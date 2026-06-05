// SPDX-License-Identifier: MIT
//
// submission_batcher_whitebox.rs -- Whitebox structural tests for T-WGPU-P1.4.3
// (Submission Batching).
//
// These tests exercise the internal implementation of SubmissionBatcher,
// covering all code paths in batching configuration, metrics tracking,
// threshold-based flushing, and thread safety.
//
// WHITEBOX coverage plan:
//   - Path A: BatcherConfig::default() returns correct default values
//   - Path B: BatcherConfig::new() with custom thresholds
//   - Path C: BatcherConfig::low_latency() preset configuration
//   - Path D: BatcherConfig::high_throughput() preset configuration
//   - Path E: BatcherConfig::time_threshold() Duration conversion
//   - Path F: BatcherMetrics::default() all fields zero
//   - Path G: BatcherMetrics::avg_batch_size() with zero submissions
//   - Path H: BatcherMetrics::avg_batch_size() with non-empty submissions
//   - Path I: BatcherMetrics::avg_batch_size() excludes empty flushes
//   - Path J: BatcherMetrics::batching_efficiency() with zero submissions
//   - Path K: BatcherMetrics::batching_efficiency() calculation
//   - Path L: BatcherMetrics::summary() format
//   - Path M: SubmissionBatcher::new() construction
//   - Path N: SubmissionBatcher::with_defaults() convenience constructor
//   - Path O: SubmissionBatcher::add() single buffer
//   - Path P: SubmissionBatcher::add() triggers count threshold flush
//   - Path Q: SubmissionBatcher::add() triggers time threshold flush
//   - Path R: SubmissionBatcher::add_many() multiple buffers
//   - Path S: SubmissionBatcher::flush() manual flush
//   - Path T: SubmissionBatcher::force_flush() for sync operations
//   - Path U: SubmissionBatcher::end_frame() frame boundary flush
//   - Path V: SubmissionBatcher::pending_count() returns correct count
//   - Path W: SubmissionBatcher::has_pending() boolean check
//   - Path X: SubmissionBatcher::metrics() returns snapshot
//   - Path Y: SubmissionBatcher::reset_metrics() clears counters
//   - Path Z: SubmissionBatcher::queue() accessor
//   - Path AA: SubmissionBatcher::config() accessor
//   - Path AB: SubmissionBatcher::time_threshold_elapsed() check
//   - Path AC: SubmissionBatcher::poll() time-based flush trigger
//   - Path AD: SubmissionBatcher Debug trait
//   - Path AE: SubmissionBatcher Send + Sync bounds
//   - Path AF: Empty flush increments empty_flushes counter
//   - Path AG: Batch size min/max tracking
//   - Path AH: FlushReason counter increments for each reason type
//   - Path AI: Edge case - zero count threshold
//   - Path AJ: Edge case - zero time threshold
//   - Path AK: Concurrent add operations (thread safety)
//
// Acceptance criteria (T-WGPU-P1.4.3):
//   1. Batch by command buffer count (threshold: 8)
//   2. Batch by time (threshold: 2ms)
//   3. Flush on frame end
//   4. Force flush API for synchronous operations
//   5. Metrics for batch effectiveness

use renderer_backend::device::{
    BatcherConfig, BatcherMetrics, SubmissionBatcher, TrinityInstance, TrinityQueue,
    DEFAULT_BATCH_COUNT_THRESHOLD, DEFAULT_BATCH_TIME_THRESHOLD_MS,
};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

// ===========================================================================
// Test Helpers
// ===========================================================================

/// Helper to create a real wgpu device and queue for testing.
/// Returns None if no GPU adapter is available (CI environment).
fn create_test_device_and_queue() -> Option<(wgpu::Device, Arc<TrinityQueue>)> {
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
            label: Some("whitebox_batcher_test_device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::downlevel_webgl2_defaults(),
            memory_hints: Default::default(),
        },
        None,
    ))
    .ok()?;

    Some((device, Arc::new(TrinityQueue::new(queue))))
}

/// Helper to create a command buffer from a device.
fn create_command_buffer(device: &wgpu::Device, label: &str) -> wgpu::CommandBuffer {
    let encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some(label),
    });
    encoder.finish()
}

/// Helper to create multiple command buffers.
fn create_command_buffers(device: &wgpu::Device, prefix: &str, count: usize) -> Vec<wgpu::CommandBuffer> {
    (0..count)
        .map(|i| create_command_buffer(device, &format!("{}_{}", prefix, i)))
        .collect()
}

// ===========================================================================
// Path A: BatcherConfig::default() returns correct default values
// ===========================================================================

#[test]
fn test_batcher_config_default_count_threshold() {
    let config = BatcherConfig::default();
    assert_eq!(
        config.count_threshold, DEFAULT_BATCH_COUNT_THRESHOLD,
        "Default count_threshold should be {}",
        DEFAULT_BATCH_COUNT_THRESHOLD
    );
}

#[test]
fn test_batcher_config_default_time_threshold() {
    let config = BatcherConfig::default();
    assert_eq!(
        config.time_threshold_ms, DEFAULT_BATCH_TIME_THRESHOLD_MS,
        "Default time_threshold_ms should be {}",
        DEFAULT_BATCH_TIME_THRESHOLD_MS
    );
}

#[test]
fn test_batcher_config_default_matches_constants() {
    let config = BatcherConfig::default();
    assert_eq!(config.count_threshold, 8);
    assert_eq!(config.time_threshold_ms, 2);
}

// ===========================================================================
// Path B: BatcherConfig::new() with custom thresholds
// ===========================================================================

#[test]
fn test_batcher_config_new_custom_values() {
    let config = BatcherConfig::new(16, 5);
    assert_eq!(config.count_threshold, 16);
    assert_eq!(config.time_threshold_ms, 5);
}

#[test]
fn test_batcher_config_new_extreme_values() {
    // Very small thresholds
    let small = BatcherConfig::new(1, 0);
    assert_eq!(small.count_threshold, 1);
    assert_eq!(small.time_threshold_ms, 0);

    // Very large thresholds
    let large = BatcherConfig::new(1000, 10000);
    assert_eq!(large.count_threshold, 1000);
    assert_eq!(large.time_threshold_ms, 10000);
}

#[test]
fn test_batcher_config_new_zero_thresholds() {
    let config = BatcherConfig::new(0, 0);
    assert_eq!(config.count_threshold, 0);
    assert_eq!(config.time_threshold_ms, 0);
}

// ===========================================================================
// Path C: BatcherConfig::low_latency() preset configuration
// ===========================================================================

#[test]
fn test_batcher_config_low_latency_count() {
    let config = BatcherConfig::low_latency();
    assert_eq!(config.count_threshold, 4, "Low latency should use count threshold 4");
}

#[test]
fn test_batcher_config_low_latency_time() {
    let config = BatcherConfig::low_latency();
    assert_eq!(config.time_threshold_ms, 1, "Low latency should use time threshold 1ms");
}

#[test]
fn test_batcher_config_low_latency_smaller_than_default() {
    let low_lat = BatcherConfig::low_latency();
    let default = BatcherConfig::default();

    assert!(
        low_lat.count_threshold < default.count_threshold,
        "Low latency count should be smaller than default"
    );
    assert!(
        low_lat.time_threshold_ms < default.time_threshold_ms,
        "Low latency time should be smaller than default"
    );
}

// ===========================================================================
// Path D: BatcherConfig::high_throughput() preset configuration
// ===========================================================================

#[test]
fn test_batcher_config_high_throughput_count() {
    let config = BatcherConfig::high_throughput();
    assert_eq!(config.count_threshold, 16, "High throughput should use count threshold 16");
}

#[test]
fn test_batcher_config_high_throughput_time() {
    let config = BatcherConfig::high_throughput();
    assert_eq!(config.time_threshold_ms, 4, "High throughput should use time threshold 4ms");
}

#[test]
fn test_batcher_config_high_throughput_larger_than_default() {
    let high_tp = BatcherConfig::high_throughput();
    let default = BatcherConfig::default();

    assert!(
        high_tp.count_threshold > default.count_threshold,
        "High throughput count should be larger than default"
    );
    assert!(
        high_tp.time_threshold_ms > default.time_threshold_ms,
        "High throughput time should be larger than default"
    );
}

// ===========================================================================
// Path E: BatcherConfig::time_threshold() Duration conversion
// ===========================================================================

#[test]
fn test_batcher_config_time_threshold_conversion() {
    let config = BatcherConfig::new(8, 5);
    assert_eq!(config.time_threshold(), Duration::from_millis(5));
}

#[test]
fn test_batcher_config_time_threshold_zero() {
    let config = BatcherConfig::new(8, 0);
    assert_eq!(config.time_threshold(), Duration::ZERO);
}

#[test]
fn test_batcher_config_time_threshold_large() {
    let config = BatcherConfig::new(8, 1000);
    assert_eq!(config.time_threshold(), Duration::from_secs(1));
}

#[test]
fn test_batcher_config_time_threshold_default_value() {
    let config = BatcherConfig::default();
    assert_eq!(config.time_threshold(), Duration::from_millis(DEFAULT_BATCH_TIME_THRESHOLD_MS));
}

// ===========================================================================
// Path F: BatcherMetrics::default() all fields zero
// ===========================================================================

#[test]
fn test_batcher_metrics_default_all_zero() {
    let metrics = BatcherMetrics::default();

    assert_eq!(metrics.total_submissions, 0);
    assert_eq!(metrics.total_batched_buffers, 0);
    assert_eq!(metrics.count_triggered_flushes, 0);
    assert_eq!(metrics.time_triggered_flushes, 0);
    assert_eq!(metrics.frame_end_flushes, 0);
    assert_eq!(metrics.force_flushes, 0);
    assert_eq!(metrics.empty_flushes, 0);
    assert_eq!(metrics.max_batch_size, 0);
    assert_eq!(metrics.min_batch_size, 0);
}

#[test]
fn test_batcher_metrics_default_clone() {
    let metrics = BatcherMetrics::default();
    let cloned = metrics.clone();

    assert_eq!(metrics.total_submissions, cloned.total_submissions);
    assert_eq!(metrics.total_batched_buffers, cloned.total_batched_buffers);
}

// ===========================================================================
// Path G: BatcherMetrics::avg_batch_size() with zero submissions
// ===========================================================================

#[test]
fn test_batcher_metrics_avg_batch_size_zero_submissions() {
    let metrics = BatcherMetrics::default();
    assert_eq!(
        metrics.avg_batch_size(),
        0.0,
        "avg_batch_size should return 0.0 when no submissions"
    );
}

#[test]
fn test_batcher_metrics_avg_batch_size_all_empty_flushes() {
    let mut metrics = BatcherMetrics::default();
    metrics.total_submissions = 5;
    metrics.empty_flushes = 5;
    metrics.total_batched_buffers = 0;

    // non_empty_submissions = 5 - 5 = 0
    assert_eq!(
        metrics.avg_batch_size(),
        0.0,
        "avg_batch_size should return 0.0 when all flushes are empty"
    );
}

// ===========================================================================
// Path H: BatcherMetrics::avg_batch_size() with non-empty submissions
// ===========================================================================

#[test]
fn test_batcher_metrics_avg_batch_size_calculation() {
    let mut metrics = BatcherMetrics::default();
    metrics.total_submissions = 4;
    metrics.total_batched_buffers = 20;
    metrics.empty_flushes = 0;

    // avg = 20 / (4 - 0) = 5.0
    let avg = metrics.avg_batch_size();
    assert!(
        (avg - 5.0).abs() < f64::EPSILON,
        "Expected avg_batch_size 5.0, got {}",
        avg
    );
}

#[test]
fn test_batcher_metrics_avg_batch_size_single_submission() {
    let mut metrics = BatcherMetrics::default();
    metrics.total_submissions = 1;
    metrics.total_batched_buffers = 8;
    metrics.empty_flushes = 0;

    let avg = metrics.avg_batch_size();
    assert!(
        (avg - 8.0).abs() < f64::EPSILON,
        "Expected avg_batch_size 8.0, got {}",
        avg
    );
}

// ===========================================================================
// Path I: BatcherMetrics::avg_batch_size() excludes empty flushes
// ===========================================================================

#[test]
fn test_batcher_metrics_avg_batch_size_excludes_empty() {
    let mut metrics = BatcherMetrics::default();
    metrics.total_submissions = 5;
    metrics.total_batched_buffers = 20;
    metrics.empty_flushes = 1;

    // non_empty_submissions = 5 - 1 = 4
    // avg = 20 / 4 = 5.0
    let avg = metrics.avg_batch_size();
    assert!(
        (avg - 5.0).abs() < f64::EPSILON,
        "Expected avg_batch_size 5.0 (excluding empty), got {}",
        avg
    );
}

#[test]
fn test_batcher_metrics_avg_batch_size_multiple_empty() {
    let mut metrics = BatcherMetrics::default();
    metrics.total_submissions = 10;
    metrics.total_batched_buffers = 40;
    metrics.empty_flushes = 2;

    // non_empty_submissions = 10 - 2 = 8
    // avg = 40 / 8 = 5.0
    let avg = metrics.avg_batch_size();
    assert!(
        (avg - 5.0).abs() < f64::EPSILON,
        "Expected avg_batch_size 5.0, got {}",
        avg
    );
}

// ===========================================================================
// Path J: BatcherMetrics::batching_efficiency() with zero submissions
// ===========================================================================

#[test]
fn test_batcher_metrics_efficiency_zero_submissions() {
    let metrics = BatcherMetrics::default();
    assert_eq!(
        metrics.batching_efficiency(),
        0.0,
        "batching_efficiency should return 0.0 when no submissions"
    );
}

// ===========================================================================
// Path K: BatcherMetrics::batching_efficiency() calculation
// ===========================================================================

#[test]
fn test_batcher_metrics_efficiency_calculation() {
    let mut metrics = BatcherMetrics::default();
    metrics.total_submissions = 5;
    metrics.total_batched_buffers = 25;

    // efficiency = 25 / 5 = 5.0
    let eff = metrics.batching_efficiency();
    assert!(
        (eff - 5.0).abs() < f64::EPSILON,
        "Expected batching_efficiency 5.0, got {}",
        eff
    );
}

#[test]
fn test_batcher_metrics_efficiency_no_batching() {
    let mut metrics = BatcherMetrics::default();
    metrics.total_submissions = 10;
    metrics.total_batched_buffers = 10;

    // efficiency = 10 / 10 = 1.0 (no batching benefit)
    let eff = metrics.batching_efficiency();
    assert!(
        (eff - 1.0).abs() < f64::EPSILON,
        "Expected batching_efficiency 1.0 (no batching), got {}",
        eff
    );
}

#[test]
fn test_batcher_metrics_efficiency_high_batching() {
    let mut metrics = BatcherMetrics::default();
    metrics.total_submissions = 2;
    metrics.total_batched_buffers = 16;

    // efficiency = 16 / 2 = 8.0 (excellent batching)
    let eff = metrics.batching_efficiency();
    assert!(
        (eff - 8.0).abs() < f64::EPSILON,
        "Expected batching_efficiency 8.0, got {}",
        eff
    );
}

// ===========================================================================
// Path L: BatcherMetrics::summary() format
// ===========================================================================

#[test]
fn test_batcher_metrics_summary_contains_submissions() {
    let mut metrics = BatcherMetrics::default();
    metrics.total_submissions = 10;

    let summary = metrics.summary();
    assert!(
        summary.contains("submissions: 10"),
        "Summary should contain submissions count"
    );
}

#[test]
fn test_batcher_metrics_summary_contains_buffers() {
    let mut metrics = BatcherMetrics::default();
    metrics.total_batched_buffers = 50;

    let summary = metrics.summary();
    assert!(
        summary.contains("buffers: 50"),
        "Summary should contain buffers count"
    );
}

#[test]
fn test_batcher_metrics_summary_contains_triggers() {
    let mut metrics = BatcherMetrics::default();
    metrics.count_triggered_flushes = 3;
    metrics.time_triggered_flushes = 2;
    metrics.frame_end_flushes = 4;
    metrics.force_flushes = 1;

    let summary = metrics.summary();
    assert!(summary.contains("count: 3"), "Summary should contain count triggers");
    assert!(summary.contains("time: 2"), "Summary should contain time triggers");
    assert!(summary.contains("frame: 4"), "Summary should contain frame triggers");
    assert!(summary.contains("force: 1"), "Summary should contain force triggers");
}

#[test]
fn test_batcher_metrics_summary_format() {
    let metrics = BatcherMetrics::default();
    let summary = metrics.summary();

    assert!(
        summary.starts_with("BatcherMetrics"),
        "Summary should start with struct name"
    );
    assert!(summary.contains("avg_size:"), "Summary should contain avg_size");
    assert!(summary.contains("efficiency:"), "Summary should contain efficiency");
    assert!(summary.contains("triggers:"), "Summary should contain triggers section");
}

// ===========================================================================
// Path M: SubmissionBatcher::new() construction
// ===========================================================================

#[test]
fn test_submission_batcher_new_empty_pending() {
    if let Some((_device, queue)) = create_test_device_and_queue() {
        let batcher = SubmissionBatcher::new(queue, BatcherConfig::default());
        assert_eq!(
            batcher.pending_count(),
            0,
            "New batcher should have zero pending buffers"
        );
    }
}

#[test]
fn test_submission_batcher_new_has_no_pending() {
    if let Some((_device, queue)) = create_test_device_and_queue() {
        let batcher = SubmissionBatcher::new(queue, BatcherConfig::default());
        assert!(
            !batcher.has_pending(),
            "New batcher should report no pending"
        );
    }
}

#[test]
fn test_submission_batcher_new_with_custom_config() {
    if let Some((_device, queue)) = create_test_device_and_queue() {
        let config = BatcherConfig::new(4, 1);
        let batcher = SubmissionBatcher::new(queue, config);

        assert_eq!(batcher.config().count_threshold, 4);
        assert_eq!(batcher.config().time_threshold_ms, 1);
    }
}

#[test]
fn test_submission_batcher_new_metrics_zeroed() {
    if let Some((_device, queue)) = create_test_device_and_queue() {
        let batcher = SubmissionBatcher::new(queue, BatcherConfig::default());
        let metrics = batcher.metrics();

        assert_eq!(metrics.total_submissions, 0);
        assert_eq!(metrics.total_batched_buffers, 0);
    }
}

// ===========================================================================
// Path N: SubmissionBatcher::with_defaults() convenience constructor
// ===========================================================================

#[test]
fn test_submission_batcher_with_defaults() {
    if let Some((_device, queue)) = create_test_device_and_queue() {
        let batcher = SubmissionBatcher::with_defaults(queue);

        assert_eq!(batcher.config().count_threshold, DEFAULT_BATCH_COUNT_THRESHOLD);
        assert_eq!(batcher.config().time_threshold_ms, DEFAULT_BATCH_TIME_THRESHOLD_MS);
    }
}

#[test]
fn test_submission_batcher_with_defaults_equivalent_to_new() {
    if let Some((_device, queue)) = create_test_device_and_queue() {
        let batcher = SubmissionBatcher::with_defaults(Arc::clone(&queue));
        let explicit = SubmissionBatcher::new(queue, BatcherConfig::default());

        assert_eq!(batcher.config().count_threshold, explicit.config().count_threshold);
        assert_eq!(batcher.config().time_threshold_ms, explicit.config().time_threshold_ms);
    }
}

// ===========================================================================
// Path O: SubmissionBatcher::add() single buffer
// ===========================================================================

#[test]
fn test_submission_batcher_add_single_buffer() {
    if let Some((device, queue)) = create_test_device_and_queue() {
        let batcher = SubmissionBatcher::new(queue, BatcherConfig::default());

        let cmd_buf = create_command_buffer(&device, "add_single");
        let flushed = batcher.add(cmd_buf);

        assert!(
            !flushed,
            "Single buffer should not trigger flush (threshold is 8)"
        );
        assert_eq!(batcher.pending_count(), 1, "Should have 1 pending buffer");
    }
}

#[test]
fn test_submission_batcher_add_increments_pending() {
    if let Some((device, queue)) = create_test_device_and_queue() {
        let batcher = SubmissionBatcher::new(queue, BatcherConfig::default());

        for i in 1..=5 {
            let cmd_buf = create_command_buffer(&device, &format!("add_inc_{}", i));
            batcher.add(cmd_buf);
            assert_eq!(batcher.pending_count(), i, "Pending should be {} after {} adds", i, i);
        }
    }
}

#[test]
fn test_submission_batcher_add_returns_false_below_threshold() {
    if let Some((device, queue)) = create_test_device_and_queue() {
        let batcher = SubmissionBatcher::new(queue, BatcherConfig::new(8, 10000)); // High time threshold

        for i in 1..8 {
            let cmd_buf = create_command_buffer(&device, &format!("below_threshold_{}", i));
            let flushed = batcher.add(cmd_buf);
            assert!(
                !flushed,
                "Buffer {} should not trigger flush (below count threshold)",
                i
            );
        }
    }
}

// ===========================================================================
// Path P: SubmissionBatcher::add() triggers count threshold flush
// ===========================================================================

#[test]
fn test_submission_batcher_add_triggers_count_threshold() {
    if let Some((device, queue)) = create_test_device_and_queue() {
        // Use count threshold of 4 for easier testing
        let batcher = SubmissionBatcher::new(queue, BatcherConfig::new(4, 10000));

        // Add 3 buffers - no flush
        for i in 0..3 {
            let cmd_buf = create_command_buffer(&device, &format!("count_thresh_{}", i));
            let flushed = batcher.add(cmd_buf);
            assert!(!flushed, "Buffer {} should not trigger flush", i);
        }

        // 4th buffer triggers flush
        let cmd_buf = create_command_buffer(&device, "count_thresh_trigger");
        let flushed = batcher.add(cmd_buf);

        assert!(flushed, "4th buffer should trigger count threshold flush");
        assert_eq!(batcher.pending_count(), 0, "After flush, pending should be 0");
    }
}

#[test]
fn test_submission_batcher_count_threshold_metrics() {
    if let Some((device, queue)) = create_test_device_and_queue() {
        let batcher = SubmissionBatcher::new(queue, BatcherConfig::new(4, 10000));

        // Add enough buffers to trigger one count-based flush
        for i in 0..4 {
            batcher.add(create_command_buffer(&device, &format!("ct_metrics_{}", i)));
        }

        let metrics = batcher.metrics();
        assert_eq!(
            metrics.count_triggered_flushes, 1,
            "Should have 1 count-triggered flush"
        );
        assert_eq!(metrics.total_submissions, 1, "Should have 1 total submission");
        assert_eq!(metrics.total_batched_buffers, 4, "Should have batched 4 buffers");
    }
}

#[test]
fn test_submission_batcher_count_threshold_boundary() {
    if let Some((device, queue)) = create_test_device_and_queue() {
        // Test exactly at threshold
        let batcher = SubmissionBatcher::new(queue, BatcherConfig::new(3, 10000));

        // Add 2 - no flush
        batcher.add(create_command_buffer(&device, "boundary_1"));
        batcher.add(create_command_buffer(&device, "boundary_2"));
        assert_eq!(batcher.pending_count(), 2);

        // Add 3rd - triggers flush
        let flushed = batcher.add(create_command_buffer(&device, "boundary_3"));
        assert!(flushed, "Exactly at threshold should trigger flush");
        assert_eq!(batcher.pending_count(), 0);
    }
}

// ===========================================================================
// Path Q: SubmissionBatcher::add() triggers time threshold flush
// ===========================================================================

#[test]
fn test_submission_batcher_add_triggers_time_threshold() {
    if let Some((device, queue)) = create_test_device_and_queue() {
        // Use high count threshold (won't hit) and short time threshold (will hit)
        let batcher = SubmissionBatcher::new(queue, BatcherConfig::new(1000, 1)); // 1ms time threshold

        // Add first buffer
        let cmd_buf = create_command_buffer(&device, "time_thresh_1");
        batcher.add(cmd_buf);

        // Wait for time threshold to elapse
        std::thread::sleep(Duration::from_millis(5));

        // Add another buffer - should trigger time threshold
        let cmd_buf = create_command_buffer(&device, "time_thresh_2");
        let flushed = batcher.add(cmd_buf);

        // Time threshold should have triggered flush
        assert!(flushed, "Should trigger time threshold flush after waiting");
    }
}

#[test]
fn test_submission_batcher_time_threshold_metrics() {
    if let Some((device, queue)) = create_test_device_and_queue() {
        let batcher = SubmissionBatcher::new(queue, BatcherConfig::new(1000, 1));

        batcher.add(create_command_buffer(&device, "tt_metrics_1"));
        std::thread::sleep(Duration::from_millis(5));
        batcher.add(create_command_buffer(&device, "tt_metrics_2"));

        let metrics = batcher.metrics();
        assert_eq!(
            metrics.time_triggered_flushes, 1,
            "Should have 1 time-triggered flush"
        );
    }
}

// ===========================================================================
// Path R: SubmissionBatcher::add_many() multiple buffers
// ===========================================================================

#[test]
fn test_submission_batcher_add_many_increments_pending() {
    if let Some((device, queue)) = create_test_device_and_queue() {
        let batcher = SubmissionBatcher::new(queue, BatcherConfig::new(100, 10000));

        let buffers = create_command_buffers(&device, "add_many", 5);
        batcher.add_many(buffers);

        assert_eq!(batcher.pending_count(), 5, "Should have 5 pending after add_many");
    }
}

#[test]
fn test_submission_batcher_add_many_triggers_threshold() {
    if let Some((device, queue)) = create_test_device_and_queue() {
        let batcher = SubmissionBatcher::new(queue, BatcherConfig::new(4, 10000));

        // Add 6 buffers at once - should trigger flush at 4
        let buffers = create_command_buffers(&device, "many_trigger", 6);
        let flushed = batcher.add_many(buffers);

        assert!(flushed, "add_many should trigger flush when exceeding threshold");
        // After flushing 6 buffers, pending should be 0
        assert_eq!(batcher.pending_count(), 0);
    }
}

#[test]
fn test_submission_batcher_add_many_empty() {
    if let Some((_device, queue)) = create_test_device_and_queue() {
        let batcher = SubmissionBatcher::new(queue, BatcherConfig::default());

        let empty: Vec<wgpu::CommandBuffer> = vec![];
        let flushed = batcher.add_many(empty);

        assert!(!flushed, "Empty add_many should not trigger flush");
        assert_eq!(batcher.pending_count(), 0, "Pending should still be 0");
    }
}

// ===========================================================================
// Path S: SubmissionBatcher::flush() manual flush
// ===========================================================================

#[test]
fn test_submission_batcher_flush_returns_count() {
    if let Some((device, queue)) = create_test_device_and_queue() {
        let batcher = SubmissionBatcher::new(queue, BatcherConfig::default());

        batcher.add(create_command_buffer(&device, "flush_1"));
        batcher.add(create_command_buffer(&device, "flush_2"));
        batcher.add(create_command_buffer(&device, "flush_3"));

        let count = batcher.flush();
        assert_eq!(count, 3, "flush() should return number of buffers flushed");
    }
}

#[test]
fn test_submission_batcher_flush_clears_pending() {
    if let Some((device, queue)) = create_test_device_and_queue() {
        let batcher = SubmissionBatcher::new(queue, BatcherConfig::default());

        batcher.add(create_command_buffer(&device, "flush_clear_1"));
        batcher.add(create_command_buffer(&device, "flush_clear_2"));
        assert_eq!(batcher.pending_count(), 2);

        batcher.flush();
        assert_eq!(batcher.pending_count(), 0, "flush() should clear pending");
    }
}

#[test]
fn test_submission_batcher_flush_empty() {
    if let Some((_device, queue)) = create_test_device_and_queue() {
        let batcher = SubmissionBatcher::new(queue, BatcherConfig::default());

        let count = batcher.flush();
        assert_eq!(count, 0, "flush() on empty batcher should return 0");
    }
}

#[test]
fn test_submission_batcher_flush_updates_metrics() {
    if let Some((device, queue)) = create_test_device_and_queue() {
        let batcher = SubmissionBatcher::new(queue, BatcherConfig::default());

        batcher.add(create_command_buffer(&device, "flush_metrics_1"));
        batcher.add(create_command_buffer(&device, "flush_metrics_2"));
        batcher.flush();

        let metrics = batcher.metrics();
        assert_eq!(metrics.total_submissions, 1);
        assert_eq!(metrics.total_batched_buffers, 2);
    }
}

// ===========================================================================
// Path T: SubmissionBatcher::force_flush() for sync operations
// ===========================================================================

#[test]
fn test_submission_batcher_force_flush_returns_count() {
    if let Some((device, queue)) = create_test_device_and_queue() {
        let batcher = SubmissionBatcher::new(queue, BatcherConfig::default());

        batcher.add(create_command_buffer(&device, "force_1"));
        batcher.add(create_command_buffer(&device, "force_2"));

        let count = batcher.force_flush();
        assert_eq!(count, 2, "force_flush() should return count");
    }
}

#[test]
fn test_submission_batcher_force_flush_clears_pending() {
    if let Some((device, queue)) = create_test_device_and_queue() {
        let batcher = SubmissionBatcher::new(queue, BatcherConfig::default());

        batcher.add(create_command_buffer(&device, "force_clear"));
        assert_eq!(batcher.pending_count(), 1);

        batcher.force_flush();
        assert_eq!(batcher.pending_count(), 0);
    }
}

#[test]
fn test_submission_batcher_force_flush_updates_metrics() {
    if let Some((device, queue)) = create_test_device_and_queue() {
        let batcher = SubmissionBatcher::new(queue, BatcherConfig::default());

        batcher.add(create_command_buffer(&device, "force_metrics"));
        batcher.force_flush();

        let metrics = batcher.metrics();
        assert_eq!(
            metrics.force_flushes, 1,
            "force_flush should increment force_flushes counter"
        );
    }
}

#[test]
fn test_submission_batcher_force_flush_empty() {
    if let Some((_device, queue)) = create_test_device_and_queue() {
        let batcher = SubmissionBatcher::new(queue, BatcherConfig::default());

        let count = batcher.force_flush();
        assert_eq!(count, 0, "force_flush on empty should return 0");

        let metrics = batcher.metrics();
        assert_eq!(metrics.force_flushes, 1, "force_flush should still count");
        assert_eq!(metrics.empty_flushes, 1, "Empty force_flush should count as empty");
    }
}

// ===========================================================================
// Path U: SubmissionBatcher::end_frame() frame boundary flush
// ===========================================================================

#[test]
fn test_submission_batcher_end_frame_returns_count() {
    if let Some((device, queue)) = create_test_device_and_queue() {
        let batcher = SubmissionBatcher::new(queue, BatcherConfig::default());

        batcher.add(create_command_buffer(&device, "frame_1"));
        batcher.add(create_command_buffer(&device, "frame_2"));
        batcher.add(create_command_buffer(&device, "frame_3"));

        let count = batcher.end_frame();
        assert_eq!(count, 3, "end_frame() should return count of flushed buffers");
    }
}

#[test]
fn test_submission_batcher_end_frame_clears_pending() {
    if let Some((device, queue)) = create_test_device_and_queue() {
        let batcher = SubmissionBatcher::new(queue, BatcherConfig::default());

        batcher.add(create_command_buffer(&device, "frame_clear"));
        assert_eq!(batcher.pending_count(), 1);

        batcher.end_frame();
        assert_eq!(batcher.pending_count(), 0, "end_frame() should clear pending");
    }
}

#[test]
fn test_submission_batcher_end_frame_updates_metrics() {
    if let Some((device, queue)) = create_test_device_and_queue() {
        let batcher = SubmissionBatcher::new(queue, BatcherConfig::default());

        batcher.add(create_command_buffer(&device, "frame_metrics"));
        batcher.end_frame();

        let metrics = batcher.metrics();
        assert_eq!(
            metrics.frame_end_flushes, 1,
            "end_frame should increment frame_end_flushes counter"
        );
    }
}

#[test]
fn test_submission_batcher_end_frame_empty() {
    if let Some((_device, queue)) = create_test_device_and_queue() {
        let batcher = SubmissionBatcher::new(queue, BatcherConfig::default());

        let count = batcher.end_frame();
        assert_eq!(count, 0, "end_frame on empty should return 0");

        let metrics = batcher.metrics();
        assert_eq!(metrics.frame_end_flushes, 1);
        assert_eq!(metrics.empty_flushes, 1, "Empty end_frame should count as empty");
    }
}

// ===========================================================================
// Path V: SubmissionBatcher::pending_count() returns correct count
// ===========================================================================

#[test]
fn test_submission_batcher_pending_count_zero_initially() {
    if let Some((_device, queue)) = create_test_device_and_queue() {
        let batcher = SubmissionBatcher::new(queue, BatcherConfig::default());
        assert_eq!(batcher.pending_count(), 0);
    }
}

#[test]
fn test_submission_batcher_pending_count_after_adds() {
    if let Some((device, queue)) = create_test_device_and_queue() {
        let batcher = SubmissionBatcher::new(queue, BatcherConfig::default());

        for expected in 1..=5 {
            batcher.add(create_command_buffer(&device, &format!("pending_{}", expected)));
            assert_eq!(batcher.pending_count(), expected);
        }
    }
}

#[test]
fn test_submission_batcher_pending_count_after_flush() {
    if let Some((device, queue)) = create_test_device_and_queue() {
        let batcher = SubmissionBatcher::new(queue, BatcherConfig::default());

        batcher.add(create_command_buffer(&device, "pending_flush"));
        assert_eq!(batcher.pending_count(), 1);

        batcher.flush();
        assert_eq!(batcher.pending_count(), 0);
    }
}

// ===========================================================================
// Path W: SubmissionBatcher::has_pending() boolean check
// ===========================================================================

#[test]
fn test_submission_batcher_has_pending_false_initially() {
    if let Some((_device, queue)) = create_test_device_and_queue() {
        let batcher = SubmissionBatcher::new(queue, BatcherConfig::default());
        assert!(!batcher.has_pending());
    }
}

#[test]
fn test_submission_batcher_has_pending_true_after_add() {
    if let Some((device, queue)) = create_test_device_and_queue() {
        let batcher = SubmissionBatcher::new(queue, BatcherConfig::default());

        batcher.add(create_command_buffer(&device, "has_pending"));
        assert!(batcher.has_pending());
    }
}

#[test]
fn test_submission_batcher_has_pending_false_after_flush() {
    if let Some((device, queue)) = create_test_device_and_queue() {
        let batcher = SubmissionBatcher::new(queue, BatcherConfig::default());

        batcher.add(create_command_buffer(&device, "has_pending_flush"));
        assert!(batcher.has_pending());

        batcher.flush();
        assert!(!batcher.has_pending());
    }
}

// ===========================================================================
// Path X: SubmissionBatcher::metrics() returns snapshot
// ===========================================================================

#[test]
fn test_submission_batcher_metrics_returns_clone() {
    if let Some((device, queue)) = create_test_device_and_queue() {
        let batcher = SubmissionBatcher::new(queue, BatcherConfig::new(4, 10000));

        // Add and flush to generate metrics
        for i in 0..4 {
            batcher.add(create_command_buffer(&device, &format!("metrics_{}", i)));
        }

        // Get snapshot
        let metrics1 = batcher.metrics();

        // Add more and flush
        for i in 0..4 {
            batcher.add(create_command_buffer(&device, &format!("metrics2_{}", i)));
        }

        let metrics2 = batcher.metrics();

        // metrics1 should be unchanged (it's a clone)
        assert_eq!(metrics1.total_submissions, 1);
        assert_eq!(metrics2.total_submissions, 2);
    }
}

// ===========================================================================
// Path Y: SubmissionBatcher::reset_metrics() clears counters
// ===========================================================================

#[test]
fn test_submission_batcher_reset_metrics() {
    if let Some((device, queue)) = create_test_device_and_queue() {
        let batcher = SubmissionBatcher::new(queue, BatcherConfig::new(4, 10000));

        // Generate some metrics
        for i in 0..4 {
            batcher.add(create_command_buffer(&device, &format!("reset_{}", i)));
        }

        // Verify non-zero metrics
        let before = batcher.metrics();
        assert!(before.total_submissions > 0);

        // Reset
        batcher.reset_metrics();

        // Verify zeroed
        let after = batcher.metrics();
        assert_eq!(after.total_submissions, 0);
        assert_eq!(after.total_batched_buffers, 0);
        assert_eq!(after.count_triggered_flushes, 0);
    }
}

// ===========================================================================
// Path Z: SubmissionBatcher::queue() accessor
// ===========================================================================

#[test]
fn test_submission_batcher_queue_accessor() {
    if let Some((_device, queue)) = create_test_device_and_queue() {
        let queue_clone = Arc::clone(&queue);
        let batcher = SubmissionBatcher::new(queue, BatcherConfig::default());

        let retrieved_queue = batcher.queue();

        // Should point to same queue (Arc comparison)
        assert!(Arc::ptr_eq(retrieved_queue, &queue_clone));
    }
}

// ===========================================================================
// Path AA: SubmissionBatcher::config() accessor
// ===========================================================================

#[test]
fn test_submission_batcher_config_accessor() {
    if let Some((_device, queue)) = create_test_device_and_queue() {
        let config = BatcherConfig::new(12, 3);
        let batcher = SubmissionBatcher::new(queue, config);

        let retrieved_config = batcher.config();
        assert_eq!(retrieved_config.count_threshold, 12);
        assert_eq!(retrieved_config.time_threshold_ms, 3);
    }
}

// ===========================================================================
// Path AB: SubmissionBatcher::time_threshold_elapsed() check
// ===========================================================================

#[test]
fn test_submission_batcher_time_threshold_elapsed_false_initially() {
    if let Some((_device, queue)) = create_test_device_and_queue() {
        let batcher = SubmissionBatcher::new(queue, BatcherConfig::new(100, 10));

        // No buffers added, no batch start time
        assert!(
            !batcher.time_threshold_elapsed(),
            "Should be false when no buffers added"
        );
    }
}

#[test]
fn test_submission_batcher_time_threshold_elapsed_false_before_timeout() {
    if let Some((device, queue)) = create_test_device_and_queue() {
        let batcher = SubmissionBatcher::new(queue, BatcherConfig::new(100, 1000)); // 1 second timeout

        batcher.add(create_command_buffer(&device, "elapsed_false"));

        // Immediately after adding, should not be elapsed
        assert!(
            !batcher.time_threshold_elapsed(),
            "Should be false immediately after add"
        );
    }
}

#[test]
fn test_submission_batcher_time_threshold_elapsed_true_after_timeout() {
    if let Some((device, queue)) = create_test_device_and_queue() {
        let batcher = SubmissionBatcher::new(queue, BatcherConfig::new(100, 1)); // 1ms timeout

        batcher.add(create_command_buffer(&device, "elapsed_true"));

        // Wait for threshold
        std::thread::sleep(Duration::from_millis(5));

        assert!(
            batcher.time_threshold_elapsed(),
            "Should be true after waiting past threshold"
        );
    }
}

// ===========================================================================
// Path AC: SubmissionBatcher::poll() time-based flush trigger
// ===========================================================================

#[test]
fn test_submission_batcher_poll_returns_zero_when_empty() {
    if let Some((_device, queue)) = create_test_device_and_queue() {
        let batcher = SubmissionBatcher::new(queue, BatcherConfig::new(100, 1));

        let count = batcher.poll();
        assert_eq!(count, 0, "poll() on empty batcher should return 0");
    }
}

#[test]
fn test_submission_batcher_poll_returns_zero_before_timeout() {
    if let Some((device, queue)) = create_test_device_and_queue() {
        let batcher = SubmissionBatcher::new(queue, BatcherConfig::new(100, 1000)); // 1 second

        batcher.add(create_command_buffer(&device, "poll_before"));

        let count = batcher.poll();
        assert_eq!(count, 0, "poll() before timeout should return 0");
        assert_eq!(batcher.pending_count(), 1, "Buffer should still be pending");
    }
}

#[test]
fn test_submission_batcher_poll_triggers_flush_after_timeout() {
    if let Some((device, queue)) = create_test_device_and_queue() {
        let batcher = SubmissionBatcher::new(queue, BatcherConfig::new(100, 1)); // 1ms

        batcher.add(create_command_buffer(&device, "poll_trigger_1"));
        batcher.add(create_command_buffer(&device, "poll_trigger_2"));

        // Wait for threshold
        std::thread::sleep(Duration::from_millis(5));

        let count = batcher.poll();
        assert_eq!(count, 2, "poll() after timeout should flush all pending");
        assert_eq!(batcher.pending_count(), 0, "Should be empty after poll flush");
    }
}

#[test]
fn test_submission_batcher_poll_updates_time_metrics() {
    if let Some((device, queue)) = create_test_device_and_queue() {
        let batcher = SubmissionBatcher::new(queue, BatcherConfig::new(100, 1));

        batcher.add(create_command_buffer(&device, "poll_metrics"));
        std::thread::sleep(Duration::from_millis(5));
        batcher.poll();

        let metrics = batcher.metrics();
        assert_eq!(
            metrics.time_triggered_flushes, 1,
            "poll flush should count as time-triggered"
        );
    }
}

// ===========================================================================
// Path AD: SubmissionBatcher Debug trait
// ===========================================================================

#[test]
fn test_submission_batcher_debug_contains_name() {
    if let Some((_device, queue)) = create_test_device_and_queue() {
        let batcher = SubmissionBatcher::new(queue, BatcherConfig::default());
        let debug_str = format!("{:?}", batcher);

        assert!(
            debug_str.contains("SubmissionBatcher"),
            "Debug should contain struct name"
        );
    }
}

#[test]
fn test_submission_batcher_debug_contains_pending() {
    if let Some((device, queue)) = create_test_device_and_queue() {
        let batcher = SubmissionBatcher::new(queue, BatcherConfig::default());

        batcher.add(create_command_buffer(&device, "debug_pending"));

        let debug_str = format!("{:?}", batcher);
        assert!(
            debug_str.contains("pending_count"),
            "Debug should contain pending_count"
        );
    }
}

#[test]
fn test_submission_batcher_debug_contains_config() {
    if let Some((_device, queue)) = create_test_device_and_queue() {
        let batcher = SubmissionBatcher::new(queue, BatcherConfig::default());
        let debug_str = format!("{:?}", batcher);

        assert!(debug_str.contains("config"), "Debug should contain config");
    }
}

#[test]
fn test_submission_batcher_debug_contains_metrics() {
    if let Some((_device, queue)) = create_test_device_and_queue() {
        let batcher = SubmissionBatcher::new(queue, BatcherConfig::default());
        let debug_str = format!("{:?}", batcher);

        assert!(debug_str.contains("metrics"), "Debug should contain metrics");
    }
}

// ===========================================================================
// Path AE: SubmissionBatcher Send + Sync bounds
// ===========================================================================

#[test]
fn test_submission_batcher_is_send() {
    fn assert_send<T: Send>() {}
    assert_send::<SubmissionBatcher>();
}

#[test]
fn test_submission_batcher_is_sync() {
    fn assert_sync<T: Sync>() {}
    assert_sync::<SubmissionBatcher>();
}

#[test]
fn test_submission_batcher_can_be_arc_shared() {
    if let Some((device, queue)) = create_test_device_and_queue() {
        let batcher = Arc::new(SubmissionBatcher::new(queue, BatcherConfig::default()));
        let batcher_clone = Arc::clone(&batcher);

        // Use from main thread
        batcher.add(create_command_buffer(&device, "arc_main"));

        // Use from another thread
        let device = Arc::new(device);
        let device_clone = Arc::clone(&device);

        let handle = std::thread::spawn(move || {
            let encoder = device_clone.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("arc_thread"),
            });
            batcher_clone.add(encoder.finish());
            batcher_clone.pending_count()
        });

        let count = handle.join().expect("Thread should complete");
        assert!(count >= 1, "Thread should have added to batcher");
    }
}

// ===========================================================================
// Path AF: Empty flush increments empty_flushes counter
// ===========================================================================

#[test]
fn test_submission_batcher_empty_flush_counted() {
    if let Some((_device, queue)) = create_test_device_and_queue() {
        let batcher = SubmissionBatcher::new(queue, BatcherConfig::default());

        // Multiple empty flushes
        batcher.flush();
        batcher.flush();
        batcher.end_frame();

        let metrics = batcher.metrics();
        assert_eq!(
            metrics.empty_flushes, 3,
            "Should count 3 empty flushes"
        );
    }
}

#[test]
fn test_submission_batcher_empty_flush_no_submission_increment() {
    if let Some((_device, queue)) = create_test_device_and_queue() {
        let batcher = SubmissionBatcher::new(queue, BatcherConfig::default());

        batcher.flush();

        let metrics = batcher.metrics();
        assert_eq!(
            metrics.total_submissions, 0,
            "Empty flush should not increment total_submissions"
        );
    }
}

// ===========================================================================
// Path AG: Batch size min/max tracking
// ===========================================================================

#[test]
fn test_submission_batcher_max_batch_size_tracking() {
    if let Some((device, queue)) = create_test_device_and_queue() {
        let batcher = SubmissionBatcher::new(queue, BatcherConfig::default());

        // First flush with 2 buffers
        batcher.add(create_command_buffer(&device, "max_1"));
        batcher.add(create_command_buffer(&device, "max_2"));
        batcher.flush();

        // Second flush with 5 buffers
        for i in 0..5 {
            batcher.add(create_command_buffer(&device, &format!("max_5_{}", i)));
        }
        batcher.flush();

        let metrics = batcher.metrics();
        assert_eq!(metrics.max_batch_size, 5, "Max batch size should be 5");
    }
}

#[test]
fn test_submission_batcher_min_batch_size_tracking() {
    if let Some((device, queue)) = create_test_device_and_queue() {
        let batcher = SubmissionBatcher::new(queue, BatcherConfig::default());

        // First flush with 5 buffers
        for i in 0..5 {
            batcher.add(create_command_buffer(&device, &format!("min_5_{}", i)));
        }
        batcher.flush();

        // Second flush with 2 buffers
        batcher.add(create_command_buffer(&device, "min_1"));
        batcher.add(create_command_buffer(&device, "min_2"));
        batcher.flush();

        let metrics = batcher.metrics();
        assert_eq!(metrics.min_batch_size, 2, "Min batch size should be 2");
    }
}

#[test]
fn test_submission_batcher_single_buffer_batch_size() {
    if let Some((device, queue)) = create_test_device_and_queue() {
        let batcher = SubmissionBatcher::new(queue, BatcherConfig::default());

        batcher.add(create_command_buffer(&device, "single"));
        batcher.flush();

        let metrics = batcher.metrics();
        assert_eq!(metrics.min_batch_size, 1);
        assert_eq!(metrics.max_batch_size, 1);
    }
}

// ===========================================================================
// Path AH: FlushReason counter increments for each reason type
// ===========================================================================

#[test]
fn test_submission_batcher_flush_reason_count_threshold() {
    if let Some((device, queue)) = create_test_device_and_queue() {
        let batcher = SubmissionBatcher::new(queue, BatcherConfig::new(2, 10000));

        batcher.add(create_command_buffer(&device, "reason_1"));
        batcher.add(create_command_buffer(&device, "reason_2")); // Triggers count threshold

        let metrics = batcher.metrics();
        assert_eq!(metrics.count_triggered_flushes, 1);
        assert_eq!(metrics.time_triggered_flushes, 0);
        assert_eq!(metrics.frame_end_flushes, 0);
        assert_eq!(metrics.force_flushes, 0);
    }
}

#[test]
fn test_submission_batcher_flush_reason_frame_end() {
    if let Some((device, queue)) = create_test_device_and_queue() {
        let batcher = SubmissionBatcher::new(queue, BatcherConfig::default());

        batcher.add(create_command_buffer(&device, "frame_reason"));
        batcher.end_frame();

        let metrics = batcher.metrics();
        assert_eq!(metrics.frame_end_flushes, 1);
        assert_eq!(metrics.count_triggered_flushes, 0);
    }
}

#[test]
fn test_submission_batcher_flush_reason_force() {
    if let Some((device, queue)) = create_test_device_and_queue() {
        let batcher = SubmissionBatcher::new(queue, BatcherConfig::default());

        batcher.add(create_command_buffer(&device, "force_reason"));
        batcher.force_flush();

        let metrics = batcher.metrics();
        assert_eq!(metrics.force_flushes, 1);
        assert_eq!(metrics.frame_end_flushes, 0);
    }
}

#[test]
fn test_submission_batcher_all_flush_reasons_combined() {
    if let Some((device, queue)) = create_test_device_and_queue() {
        let batcher = SubmissionBatcher::new(queue, BatcherConfig::new(2, 1));

        // Count threshold
        batcher.add(create_command_buffer(&device, "all_1"));
        batcher.add(create_command_buffer(&device, "all_2"));

        // Frame end
        batcher.add(create_command_buffer(&device, "all_3"));
        batcher.end_frame();

        // Force flush
        batcher.add(create_command_buffer(&device, "all_4"));
        batcher.force_flush();

        // Time threshold
        batcher.add(create_command_buffer(&device, "all_5"));
        std::thread::sleep(Duration::from_millis(5));
        batcher.poll();

        let metrics = batcher.metrics();
        assert_eq!(metrics.count_triggered_flushes, 1);
        assert_eq!(metrics.frame_end_flushes, 1);
        assert_eq!(metrics.force_flushes, 1);
        assert_eq!(metrics.time_triggered_flushes, 1);
        assert_eq!(metrics.total_submissions, 4);
    }
}

// ===========================================================================
// Path AI: Edge case - zero count threshold
// ===========================================================================

#[test]
fn test_submission_batcher_zero_count_threshold_immediate_flush() {
    if let Some((device, queue)) = create_test_device_and_queue() {
        // With count threshold 0, every add should... actually NOT trigger
        // because we check >= threshold, and 1 >= 0 is true
        // So threshold=0 means NEVER flush due to count (need at least 0 buffers)
        // Actually, let's test the actual behavior:
        let batcher = SubmissionBatcher::new(queue, BatcherConfig::new(0, 10000));

        // With threshold 0, the condition `pending_buffers.len() >= threshold`
        // becomes `1 >= 0` which is always true
        let flushed = batcher.add(create_command_buffer(&device, "zero_thresh"));

        // Every single add will trigger flush when threshold is 0
        assert!(flushed, "With threshold 0, every add triggers flush");
        assert_eq!(batcher.pending_count(), 0);
    }
}

// ===========================================================================
// Path AJ: Edge case - zero time threshold
// ===========================================================================

#[test]
fn test_submission_batcher_zero_time_threshold() {
    if let Some((device, queue)) = create_test_device_and_queue() {
        // With time threshold 0ms, time-based flush could trigger immediately
        let batcher = SubmissionBatcher::new(queue, BatcherConfig::new(1000, 0));

        batcher.add(create_command_buffer(&device, "zero_time_1"));

        // Even without sleep, Instant::now() elapsed >= 0ms is always true
        let flushed = batcher.add(create_command_buffer(&device, "zero_time_2"));

        // Second add should trigger time-based flush since any elapsed time >= 0
        assert!(flushed, "With 0ms threshold, second add should flush");
    }
}

// ===========================================================================
// Path AK: Concurrent add operations (thread safety)
// ===========================================================================

#[test]
fn test_submission_batcher_concurrent_adds() {
    if let Some((device, queue)) = create_test_device_and_queue() {
        let batcher = Arc::new(SubmissionBatcher::new(queue, BatcherConfig::new(1000, 10000)));
        let device = Arc::new(device);

        const NUM_THREADS: usize = 4;
        const ADDS_PER_THREAD: usize = 5;

        let add_count = Arc::new(AtomicUsize::new(0));

        let mut handles = Vec::new();

        for t in 0..NUM_THREADS {
            let batcher_clone = Arc::clone(&batcher);
            let device_clone = Arc::clone(&device);
            let add_count_clone = Arc::clone(&add_count);

            let handle = std::thread::spawn(move || {
                for i in 0..ADDS_PER_THREAD {
                    let encoder = device_clone.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some(&format!("concurrent_t{}_i{}", t, i)),
                    });
                    batcher_clone.add(encoder.finish());
                    add_count_clone.fetch_add(1, Ordering::SeqCst);
                }
            });

            handles.push(handle);
        }

        // Wait for all threads
        for handle in handles {
            handle.join().expect("Thread should complete");
        }

        // Total adds should equal expected
        assert_eq!(
            add_count.load(Ordering::SeqCst),
            NUM_THREADS * ADDS_PER_THREAD,
            "All adds should complete"
        );

        // All buffers should either be pending or have been flushed
        // (no buffers lost due to race conditions)
        let pending = batcher.pending_count();
        let metrics = batcher.metrics();
        let total = pending + metrics.total_batched_buffers as usize;

        assert_eq!(
            total,
            NUM_THREADS * ADDS_PER_THREAD,
            "No buffers should be lost in concurrent adds"
        );
    }
}

#[test]
fn test_submission_batcher_concurrent_add_and_flush() {
    if let Some((device, queue)) = create_test_device_and_queue() {
        let batcher = Arc::new(SubmissionBatcher::new(queue, BatcherConfig::new(100, 10000)));
        let device = Arc::new(device);

        let batcher_add = Arc::clone(&batcher);
        let device_clone = Arc::clone(&device);

        // Thread 1: Add buffers
        let add_handle = std::thread::spawn(move || {
            for i in 0..10 {
                let encoder = device_clone.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some(&format!("add_thread_{}", i)),
                });
                batcher_add.add(encoder.finish());
                std::thread::sleep(Duration::from_micros(100));
            }
        });

        let batcher_flush = Arc::clone(&batcher);

        // Thread 2: Periodic flushes
        let flush_handle = std::thread::spawn(move || {
            for _ in 0..5 {
                std::thread::sleep(Duration::from_micros(200));
                batcher_flush.flush();
            }
        });

        add_handle.join().expect("Add thread should complete");
        flush_handle.join().expect("Flush thread should complete");

        // Final flush to get all remaining
        batcher.flush();

        // Should have no lost buffers
        let metrics = batcher.metrics();
        assert_eq!(
            metrics.total_batched_buffers, 10,
            "All 10 buffers should be batched"
        );
    }
}

// ===========================================================================
// Acceptance Criteria Tests
// ===========================================================================

#[test]
fn acceptance_criteria_1_batch_by_count() {
    // AC1: Batch by command buffer count (threshold: 8)
    if let Some((device, queue)) = create_test_device_and_queue() {
        let batcher = SubmissionBatcher::with_defaults(queue);

        // Add 7 buffers - should not flush
        for i in 0..7 {
            let flushed = batcher.add(create_command_buffer(&device, &format!("ac1_{}", i)));
            assert!(!flushed, "Buffer {} should not trigger flush", i);
        }
        assert_eq!(batcher.pending_count(), 7);

        // Add 8th buffer - should flush
        let flushed = batcher.add(create_command_buffer(&device, "ac1_trigger"));
        assert!(flushed, "8th buffer should trigger count-based flush");
        assert_eq!(batcher.pending_count(), 0);
    }
}

#[test]
fn acceptance_criteria_2_batch_by_time() {
    // AC2: Batch by time (threshold: 2ms)
    if let Some((device, queue)) = create_test_device_and_queue() {
        let batcher = SubmissionBatcher::with_defaults(queue);

        batcher.add(create_command_buffer(&device, "ac2_1"));

        // Wait past 2ms threshold
        std::thread::sleep(Duration::from_millis(5));

        // poll() should trigger time-based flush
        let count = batcher.poll();
        assert_eq!(count, 1, "Should flush 1 buffer after time threshold");

        let metrics = batcher.metrics();
        assert_eq!(metrics.time_triggered_flushes, 1);
    }
}

#[test]
fn acceptance_criteria_3_flush_on_frame_end() {
    // AC3: Flush on frame end
    if let Some((device, queue)) = create_test_device_and_queue() {
        let batcher = SubmissionBatcher::with_defaults(queue);

        batcher.add(create_command_buffer(&device, "ac3_1"));
        batcher.add(create_command_buffer(&device, "ac3_2"));
        assert_eq!(batcher.pending_count(), 2);

        let count = batcher.end_frame();
        assert_eq!(count, 2, "end_frame should flush 2 pending buffers");
        assert_eq!(batcher.pending_count(), 0, "No pending after end_frame");
    }
}

#[test]
fn acceptance_criteria_4_force_flush_api() {
    // AC4: Force flush API for synchronous operations
    if let Some((device, queue)) = create_test_device_and_queue() {
        let batcher = SubmissionBatcher::with_defaults(queue);

        batcher.add(create_command_buffer(&device, "ac4_1"));

        let count = batcher.force_flush();
        assert_eq!(count, 1, "force_flush should return flushed count");
        assert_eq!(batcher.pending_count(), 0);

        let metrics = batcher.metrics();
        assert_eq!(metrics.force_flushes, 1, "force_flush should be tracked");
    }
}

#[test]
fn acceptance_criteria_5_batch_effectiveness_metrics() {
    // AC5: Metrics for batch effectiveness
    if let Some((device, queue)) = create_test_device_and_queue() {
        let batcher = SubmissionBatcher::new(queue, BatcherConfig::new(4, 10000));

        // Create some batches
        for batch in 0..3 {
            for i in 0..4 {
                batcher.add(create_command_buffer(&device, &format!("ac5_b{}_i{}", batch, i)));
            }
        }

        let metrics = batcher.metrics();

        // Verify all metric fields are populated
        assert_eq!(metrics.total_submissions, 3, "Should have 3 submissions");
        assert_eq!(metrics.total_batched_buffers, 12, "Should batch 12 buffers");
        assert_eq!(metrics.count_triggered_flushes, 3, "All 3 should be count-triggered");

        // Verify calculated metrics
        let avg = metrics.avg_batch_size();
        assert!((avg - 4.0).abs() < f64::EPSILON, "Avg batch size should be 4.0");

        let eff = metrics.batching_efficiency();
        assert!((eff - 4.0).abs() < f64::EPSILON, "Efficiency should be 4.0");

        assert_eq!(metrics.min_batch_size, 4);
        assert_eq!(metrics.max_batch_size, 4);
    }
}
