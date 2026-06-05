// Blackbox contract tests for T-WGPU-P7.1.8 Frame Pacing API (presentation/surface.rs)
//
// CLEANROOM: No access to implementation files. Tests use only the public API
// exported by `renderer_backend::presentation`.
//
// Forbidden files (per TESTDEV_BLACKBOX prompt):
//   - crates/renderer-backend/src/presentation/surface.rs (implementation)
//   - Any WHITEBOX test file for this task
//
// Contract sources:
//   - Task acceptance criteria (T-WGPU-P7.1.8)
//
// Public API under test:
//   - FramePacer: new(), begin_frame(), end_frame(), wait_for_target(), should_skip_frame(), statistics()
//   - FrameStatistics: fps(), percentile_frame_time(), frame_time_variance(), is_consistent()
//   - FrameTiming: new(), begin_frame(), end_frame(), frame_delta(), set_target_fps()
//   - DEFAULT_FRAME_HISTORY_SIZE constant
//
// Test design rationale:
//   Equivalence partitioning:
//     - FPS targets: 60, 30, 120, None (unlimited)
//     - Frame timing variations: stable, unstable, edge cases
//   Boundary cases:
//     - Empty frame history
//     - Single frame
//     - History overflow
//     - Zero/negative percentiles
//   Statistical validation:
//     - min/max/avg accuracy
//     - percentile correctness
//     - variance calculation
//     - consistency detection

use std::thread;
use std::time::{Duration, Instant};

use renderer_backend::presentation::surface::{
    FramePacer, FrameStatistics, FrameTiming, DEFAULT_FRAME_HISTORY_SIZE,
};

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

/// Creates a FrameStatistics from a slice of millisecond values.
fn stats_from_ms(times_ms: &[u64]) -> FrameStatistics {
    let durations: Vec<Duration> = times_ms.iter().map(|&ms| Duration::from_millis(ms)).collect();
    FrameStatistics::from_times(durations)
}

// =============================================================================
// SECTION 1 -- DEFAULT_FRAME_HISTORY_SIZE Constant Tests
// =============================================================================

#[test]
fn test_default_frame_history_size_constant() {
    assert_eq!(DEFAULT_FRAME_HISTORY_SIZE, 100);
}

#[test]
fn test_default_frame_history_size_reasonable() {
    assert!(DEFAULT_FRAME_HISTORY_SIZE >= 10, "History too small for statistics");
    assert!(DEFAULT_FRAME_HISTORY_SIZE <= 1000, "History too large, memory concern");
}

// =============================================================================
// SECTION 2 -- FrameTiming Basic Tests
// =============================================================================

#[test]
fn test_frame_timing_new_defaults() {
    let timing = FrameTiming::new();
    assert_eq!(timing.frame_count(), 0);
    assert_eq!(timing.frame_delta(), Duration::ZERO);
    assert!(timing.target_frame_time().is_none());
    assert!(!timing.in_frame());
    assert_eq!(timing.history_size(), DEFAULT_FRAME_HISTORY_SIZE);
}

#[test]
fn test_frame_timing_with_history_size() {
    let sizes = [1, 10, 50, 200, 500];
    for size in sizes {
        let timing = FrameTiming::with_history_size(size);
        assert_eq!(timing.history_size(), size, "History size {} not set", size);
    }
}

#[test]
fn test_frame_timing_default_matches_new() {
    let default_timing = FrameTiming::default();
    let new_timing = FrameTiming::new();
    assert_eq!(default_timing.frame_count(), new_timing.frame_count());
    assert_eq!(default_timing.history_size(), new_timing.history_size());
}

#[test]
fn test_frame_timing_set_target_fps() {
    let mut timing = FrameTiming::new();
    let fps_targets = [30, 60, 120, 144, 240];
    for fps in fps_targets {
        timing.set_target_fps(Some(fps));
        let target = timing.target_frame_time().unwrap();
        let expected_ms = 1000.0 / fps as f64;
        let actual_ms = target.as_secs_f64() * 1000.0;
        assert!(
            (actual_ms - expected_ms).abs() < 0.1,
            "FPS {} -> expected {:.3}ms, got {:.3}ms",
            fps, expected_ms, actual_ms
        );
    }
}

#[test]
fn test_frame_timing_set_target_fps_none() {
    let mut timing = FrameTiming::new();
    timing.set_target_fps(Some(60));
    assert!(timing.target_frame_time().is_some());
    timing.set_target_fps(None);
    assert!(timing.target_frame_time().is_none());
}

#[test]
fn test_frame_timing_set_target_frame_time() {
    let mut timing = FrameTiming::new();
    let target = Duration::from_millis(16);
    timing.set_target_frame_time(Some(target));
    assert_eq!(timing.target_frame_time(), Some(target));
}

#[test]
fn test_frame_timing_begin_frame() {
    let mut timing = FrameTiming::new();
    assert!(!timing.in_frame());
    timing.begin_frame();
    assert!(timing.in_frame());
}

#[test]
fn test_frame_timing_end_frame() {
    let mut timing = FrameTiming::new();
    timing.begin_frame();
    thread::sleep(Duration::from_millis(2));
    timing.end_frame();
    assert!(!timing.in_frame());
    assert_eq!(timing.frame_count(), 1);
    assert!(timing.frame_delta() >= Duration::from_millis(2));
}

#[test]
fn test_frame_timing_end_frame_without_begin() {
    let mut timing = FrameTiming::new();
    timing.end_frame(); // Should not panic or change state
    assert_eq!(timing.frame_count(), 0);
    assert!(!timing.in_frame());
}

#[test]
fn test_frame_timing_double_begin() {
    let mut timing = FrameTiming::new();
    timing.begin_frame();
    thread::sleep(Duration::from_millis(1));
    timing.begin_frame(); // Should auto-end previous
    assert_eq!(timing.frame_count(), 1);
    assert!(timing.in_frame());
}

#[test]
fn test_frame_timing_elapsed_not_in_frame() {
    let timing = FrameTiming::new();
    assert_eq!(timing.elapsed(), Duration::ZERO);
}

#[test]
fn test_frame_timing_elapsed_in_frame() {
    let mut timing = FrameTiming::new();
    timing.begin_frame();
    thread::sleep(Duration::from_millis(2));
    let elapsed = timing.elapsed();
    assert!(elapsed >= Duration::from_millis(2));
}

#[test]
fn test_frame_timing_history_limit() {
    let mut timing = FrameTiming::with_history_size(5);
    for _ in 0..10 {
        timing.begin_frame();
        timing.end_frame();
    }
    assert_eq!(timing.frame_times().len(), 5);
    assert_eq!(timing.frame_count(), 10);
}

#[test]
fn test_frame_timing_reset() {
    let mut timing = FrameTiming::new();
    timing.set_target_fps(Some(60));
    timing.begin_frame();
    timing.end_frame();
    timing.begin_frame();
    timing.end_frame();
    timing.reset();
    assert_eq!(timing.frame_count(), 0);
    assert_eq!(timing.frame_delta(), Duration::ZERO);
    assert!(timing.frame_times().is_empty());
    assert!(!timing.in_frame());
}

#[test]
fn test_frame_timing_clone() {
    let mut timing = FrameTiming::new();
    timing.set_target_fps(Some(60));
    timing.begin_frame();
    timing.end_frame();
    let cloned = timing.clone();
    assert_eq!(cloned.frame_count(), timing.frame_count());
    assert_eq!(cloned.target_frame_time(), timing.target_frame_time());
}

#[test]
fn test_frame_timing_debug() {
    let timing = FrameTiming::new();
    let debug = format!("{:?}", timing);
    assert!(debug.contains("FrameTiming"));
}

// =============================================================================
// SECTION 3 -- FrameStatistics Empty/Single Case Tests
// =============================================================================

#[test]
fn test_frame_statistics_empty() {
    let stats = FrameStatistics::from_times(std::iter::empty::<Duration>());
    assert_eq!(stats.sample_count, 0);
    assert_eq!(stats.min_frame_time, Duration::ZERO);
    assert_eq!(stats.max_frame_time, Duration::ZERO);
    assert_eq!(stats.avg_frame_time, Duration::ZERO);
    assert_eq!(stats.fps(), 0.0);
}

#[test]
fn test_frame_statistics_default() {
    let stats = FrameStatistics::default();
    assert_eq!(stats.sample_count, 0);
    assert_eq!(stats.fps(), 0.0);
}

#[test]
fn test_frame_statistics_single_sample() {
    let stats = stats_from_ms(&[16]);
    assert_eq!(stats.sample_count, 1);
    assert_eq!(stats.min_frame_time, Duration::from_millis(16));
    assert_eq!(stats.max_frame_time, Duration::from_millis(16));
    assert_eq!(stats.avg_frame_time, Duration::from_millis(16));
}

#[test]
fn test_frame_statistics_two_samples() {
    let stats = stats_from_ms(&[10, 20]);
    assert_eq!(stats.sample_count, 2);
    assert_eq!(stats.min_frame_time, Duration::from_millis(10));
    assert_eq!(stats.max_frame_time, Duration::from_millis(20));
    assert_eq!(stats.avg_frame_time, Duration::from_millis(15));
}

// =============================================================================
// SECTION 4 -- FrameStatistics FPS Calculation Tests
// =============================================================================

#[test]
fn test_frame_statistics_fps_60() {
    let stats = stats_from_ms(&[16, 17, 16, 17, 17]); // ~16.6ms avg
    let fps = stats.fps();
    assert!(fps > 58.0 && fps < 63.0, "FPS {} not in 60 FPS range", fps);
}

#[test]
fn test_frame_statistics_fps_30() {
    let stats = stats_from_ms(&[33, 33, 34, 33]); // ~33.25ms avg
    let fps = stats.fps();
    assert!(fps > 29.0 && fps < 31.0, "FPS {} not in 30 FPS range", fps);
}

#[test]
fn test_frame_statistics_fps_zero_time() {
    let stats = stats_from_ms(&[0]);
    assert_eq!(stats.fps(), 0.0);
}

#[test]
fn test_frame_statistics_min_fps() {
    let stats = stats_from_ms(&[10, 50]); // max = 50ms = 20 FPS
    let min_fps = stats.min_fps();
    assert!((min_fps - 20.0).abs() < 1.0, "min_fps {} not ~20", min_fps);
}

#[test]
fn test_frame_statistics_max_fps() {
    let stats = stats_from_ms(&[10, 50]); // min = 10ms = 100 FPS
    let max_fps = stats.max_fps();
    assert!((max_fps - 100.0).abs() < 1.0, "max_fps {} not ~100", max_fps);
}

#[test]
fn test_frame_statistics_min_fps_zero() {
    let stats = stats_from_ms(&[0]);
    assert_eq!(stats.min_fps(), 0.0);
}

#[test]
fn test_frame_statistics_max_fps_zero() {
    let stats = stats_from_ms(&[0]);
    assert_eq!(stats.max_fps(), 0.0);
}

// =============================================================================
// SECTION 5 -- FrameStatistics Percentile Tests
// =============================================================================

#[test]
fn test_frame_statistics_percentile_empty() {
    let stats = FrameStatistics::default();
    assert_eq!(stats.percentile_frame_time(0.5), Duration::ZERO);
    assert_eq!(stats.percentile_frame_time(0.95), Duration::ZERO);
}

#[test]
fn test_frame_statistics_percentile_p0() {
    let stats = stats_from_ms(&[10, 20, 30, 40, 50]);
    assert_eq!(stats.percentile_frame_time(0.0), Duration::from_millis(10));
}

#[test]
fn test_frame_statistics_percentile_p100() {
    let stats = stats_from_ms(&[10, 20, 30, 40, 50]);
    assert_eq!(stats.percentile_frame_time(1.0), Duration::from_millis(50));
}

#[test]
fn test_frame_statistics_percentile_p50_odd() {
    let stats = stats_from_ms(&[10, 20, 30, 40, 50]);
    assert_eq!(stats.percentile_frame_time(0.5), Duration::from_millis(30));
}

#[test]
fn test_frame_statistics_percentile_p50_even() {
    let stats = stats_from_ms(&[10, 20, 30, 40]);
    let p50 = stats.percentile_frame_time(0.5);
    assert!(p50 >= Duration::from_millis(20) && p50 <= Duration::from_millis(30));
}

#[test]
fn test_frame_statistics_percentile_p95() {
    let times: Vec<u64> = (1..=100).collect();
    let stats = stats_from_ms(&times);
    let p95 = stats.percentile_frame_time(0.95);
    assert!(
        p95 >= Duration::from_millis(94) && p95 <= Duration::from_millis(96),
        "p95 {:?} not in expected range",
        p95
    );
}

#[test]
fn test_frame_statistics_percentile_clamp_negative() {
    let stats = stats_from_ms(&[10, 20, 30]);
    assert_eq!(stats.percentile_frame_time(-0.5), Duration::from_millis(10));
}

#[test]
fn test_frame_statistics_percentile_clamp_above_one() {
    let stats = stats_from_ms(&[10, 20, 30]);
    assert_eq!(stats.percentile_frame_time(1.5), Duration::from_millis(30));
}

#[test]
fn test_frame_statistics_median() {
    let stats = stats_from_ms(&[10, 20, 30, 40, 50]);
    assert_eq!(stats.median_frame_time(), stats.percentile_frame_time(0.5));
}

// =============================================================================
// SECTION 6 -- FrameStatistics Variance and Consistency Tests
// =============================================================================

#[test]
fn test_frame_statistics_variance_uniform() {
    let stats = stats_from_ms(&[16, 16, 16, 16, 16]);
    let variance = stats.frame_time_variance();
    assert!(variance < 0.00001, "Uniform variance {} not near zero", variance);
}

#[test]
fn test_frame_statistics_variance_varied() {
    let stats = stats_from_ms(&[10, 20, 30]);
    let variance = stats.frame_time_variance();
    assert!(variance > 0.0, "Variance should be positive");
}

#[test]
fn test_frame_statistics_variance_single_sample() {
    let stats = stats_from_ms(&[16]);
    assert_eq!(stats.frame_time_variance(), 0.0);
}

#[test]
fn test_frame_statistics_variance_empty() {
    let stats = FrameStatistics::default();
    assert_eq!(stats.frame_time_variance(), 0.0);
}

#[test]
fn test_frame_statistics_std_dev() {
    let stats = stats_from_ms(&[10, 20, 30, 40]);
    let variance = stats.frame_time_variance();
    let std_dev = stats.frame_time_std_dev();
    let expected_std_dev = variance.sqrt();
    assert!(
        (std_dev - expected_std_dev).abs() < 0.00001,
        "std_dev {} != sqrt(variance) {}",
        std_dev, expected_std_dev
    );
}

#[test]
fn test_frame_statistics_is_consistent_stable() {
    let stats = stats_from_ms(&[16, 16, 17, 16, 17]);
    assert!(stats.is_consistent(0.1), "Stable times should be consistent");
}

#[test]
fn test_frame_statistics_is_consistent_unstable() {
    let stats = stats_from_ms(&[10, 100, 15, 90]);
    assert!(!stats.is_consistent(0.1), "Unstable times should not be consistent");
}

#[test]
fn test_frame_statistics_is_consistent_zero_avg() {
    let stats = stats_from_ms(&[0, 0, 0]);
    assert!(stats.is_consistent(0.1));
}

#[test]
fn test_frame_statistics_clone_eq() {
    let stats = stats_from_ms(&[10, 20, 30]);
    let cloned = stats.clone();
    assert_eq!(stats, cloned);
}

#[test]
fn test_frame_statistics_display() {
    let stats = stats_from_ms(&[16]);
    let display = format!("{}", stats);
    assert!(display.contains("FPS"), "Display should contain FPS");
}

#[test]
fn test_frame_statistics_debug() {
    let stats = stats_from_ms(&[16]);
    let debug = format!("{:?}", stats);
    assert!(debug.contains("FrameStatistics"));
}

// =============================================================================
// SECTION 7 -- FramePacer Construction Tests
// =============================================================================

#[test]
fn test_frame_pacer_new_60fps() {
    let pacer = FramePacer::new(Some(60));
    assert!(pacer.is_limiting_enabled());
    let target = pacer.target_fps().unwrap();
    assert!((target - 60.0).abs() < 0.5, "Target FPS {} not ~60", target);
}

#[test]
fn test_frame_pacer_new_30fps() {
    let pacer = FramePacer::new(Some(30));
    assert!(pacer.is_limiting_enabled());
    let target = pacer.target_fps().unwrap();
    assert!((target - 30.0).abs() < 0.5, "Target FPS {} not ~30", target);
}

#[test]
fn test_frame_pacer_new_unlimited() {
    let pacer = FramePacer::new(None);
    assert!(!pacer.is_limiting_enabled());
    assert!(pacer.target_fps().is_none());
}

#[test]
fn test_frame_pacer_with_history_size() {
    let pacer = FramePacer::with_history_size(Some(60), 50);
    assert_eq!(pacer.timing().history_size(), 50);
    assert!(pacer.is_limiting_enabled());
}

#[test]
fn test_frame_pacer_default() {
    let pacer = FramePacer::default();
    assert!(!pacer.is_limiting_enabled());
    assert!(pacer.target_fps().is_none());
    assert_eq!(pacer.frame_count(), 0);
}

// =============================================================================
// SECTION 8 -- FramePacer Begin/End Frame Tests
// =============================================================================

#[test]
fn test_frame_pacer_begin_end_frame() {
    let mut pacer = FramePacer::new(None);
    assert_eq!(pacer.frame_count(), 0);
    pacer.begin_frame();
    pacer.end_frame();
    assert_eq!(pacer.frame_count(), 1);
    pacer.begin_frame();
    pacer.end_frame();
    assert_eq!(pacer.frame_count(), 2);
}

#[test]
fn test_frame_pacer_frame_delta() {
    let mut pacer = FramePacer::new(None);
    pacer.begin_frame();
    thread::sleep(Duration::from_millis(5));
    pacer.end_frame();
    let delta = pacer.frame_delta();
    assert!(delta >= Duration::from_millis(5), "Delta {:?} too short", delta);
}

#[test]
fn test_frame_pacer_current_fps_no_frames() {
    let pacer = FramePacer::new(None);
    assert_eq!(pacer.current_fps(), 0.0);
}

#[test]
fn test_frame_pacer_current_fps_after_frames() {
    let mut pacer = FramePacer::new(None);
    pacer.begin_frame();
    thread::sleep(Duration::from_millis(10));
    pacer.end_frame();
    let fps = pacer.current_fps();
    assert!(fps > 50.0 && fps < 200.0, "FPS {} not in expected range", fps);
}

// =============================================================================
// SECTION 9 -- FramePacer Target FPS Tests
// =============================================================================

#[test]
fn test_frame_pacer_set_target_fps_enables() {
    let mut pacer = FramePacer::new(None);
    assert!(!pacer.is_limiting_enabled());
    pacer.set_target_fps(Some(60));
    assert!(pacer.is_limiting_enabled());
}

#[test]
fn test_frame_pacer_set_target_fps_disables() {
    let mut pacer = FramePacer::new(Some(60));
    assert!(pacer.is_limiting_enabled());
    pacer.set_target_fps(None);
    assert!(!pacer.is_limiting_enabled());
}

#[test]
fn test_frame_pacer_set_limiting_enabled() {
    let mut pacer = FramePacer::new(Some(60));
    assert!(pacer.is_limiting_enabled());
    pacer.set_limiting_enabled(false);
    assert!(!pacer.is_limiting_enabled());
    pacer.set_limiting_enabled(true);
    assert!(pacer.is_limiting_enabled());
}

#[test]
fn test_frame_pacer_set_limiting_enabled_no_target() {
    let mut pacer = FramePacer::new(None);
    pacer.set_limiting_enabled(true);
    assert!(!pacer.is_limiting_enabled(), "Can't enable without target");
}

// =============================================================================
// SECTION 10 -- FramePacer Wait for Target Tests
// =============================================================================

#[test]
fn test_frame_pacer_wait_for_target_unlimited() {
    let mut pacer = FramePacer::new(None);
    pacer.begin_frame();
    pacer.end_frame();
    let waited = pacer.wait_for_target();
    assert_eq!(waited, Duration::ZERO);
}

#[test]
fn test_frame_pacer_wait_for_target_limiting_disabled() {
    let mut pacer = FramePacer::new(Some(60));
    pacer.set_limiting_enabled(false);
    pacer.begin_frame();
    pacer.end_frame();
    let waited = pacer.wait_for_target();
    assert_eq!(waited, Duration::ZERO);
}

#[test]
fn test_frame_pacer_wait_for_target_waits() {
    let mut pacer = FramePacer::new(Some(30)); // 33.3ms target
    pacer.begin_frame();
    thread::sleep(Duration::from_millis(5)); // Much faster than target
    pacer.end_frame();
    let start = Instant::now();
    let waited = pacer.wait_for_target();
    let actual_wait = start.elapsed();
    // Should have waited roughly 28ms (33.3 - 5)
    assert!(
        waited >= Duration::from_millis(20),
        "Waited {:?}, expected ~28ms",
        waited
    );
    assert!(
        actual_wait >= Duration::from_millis(20),
        "Actual wait {:?}, expected ~28ms",
        actual_wait
    );
}

#[test]
fn test_frame_pacer_wait_for_target_behind_schedule() {
    let mut pacer = FramePacer::new(Some(120)); // 8.3ms target
    pacer.begin_frame();
    thread::sleep(Duration::from_millis(20)); // Slower than target
    pacer.end_frame();
    let waited = pacer.wait_for_target();
    assert_eq!(waited, Duration::ZERO, "Should not wait when behind");
}

// =============================================================================
// SECTION 11 -- FramePacer Frame Skipping Tests
// =============================================================================

#[test]
fn test_frame_pacer_should_skip_frame_unlimited() {
    let mut pacer = FramePacer::new(None);
    assert!(!pacer.should_skip_frame());
}

#[test]
fn test_frame_pacer_should_skip_frame_not_behind() {
    let mut pacer = FramePacer::new(Some(60));
    pacer.begin_frame();
    pacer.end_frame();
    assert!(!pacer.should_skip_frame());
}

#[test]
fn test_frame_pacer_skip_threshold() {
    let mut pacer = FramePacer::new(Some(60));
    assert_eq!(pacer.skip_threshold(), 2); // Default
    pacer.set_skip_threshold(5);
    assert_eq!(pacer.skip_threshold(), 5);
}

#[test]
fn test_frame_pacer_skip_threshold_minimum() {
    let mut pacer = FramePacer::new(Some(60));
    pacer.set_skip_threshold(0);
    assert_eq!(pacer.skip_threshold(), 1, "Should clamp to 1");
}

#[test]
fn test_frame_pacer_time_debt_initial() {
    let pacer = FramePacer::new(Some(60));
    assert_eq!(pacer.time_debt(), Duration::ZERO);
}

#[test]
fn test_frame_pacer_frames_skipped_initial() {
    let pacer = FramePacer::new(Some(60));
    assert_eq!(pacer.frames_skipped(), 0);
}

// =============================================================================
// SECTION 12 -- FramePacer Statistics Tests
// =============================================================================

#[test]
fn test_frame_pacer_statistics_no_frames() {
    let pacer = FramePacer::new(None);
    let stats = pacer.statistics();
    assert_eq!(stats.sample_count, 0);
}

#[test]
fn test_frame_pacer_statistics_after_frames() {
    let mut pacer = FramePacer::new(None);
    for _ in 0..5 {
        pacer.begin_frame();
        pacer.end_frame();
    }
    let stats = pacer.statistics();
    assert_eq!(stats.sample_count, 5);
}

#[test]
fn test_frame_pacer_statistics_history_limit() {
    let mut pacer = FramePacer::with_history_size(None, 5);
    for _ in 0..10 {
        pacer.begin_frame();
        pacer.end_frame();
    }
    let stats = pacer.statistics();
    assert_eq!(stats.sample_count, 5);
    assert_eq!(pacer.frame_count(), 10);
}

// =============================================================================
// SECTION 13 -- FramePacer Timing Access Tests
// =============================================================================

#[test]
fn test_frame_pacer_timing_read() {
    let pacer = FramePacer::new(Some(60));
    let timing = pacer.timing();
    assert_eq!(timing.frame_count(), 0);
}

#[test]
fn test_frame_pacer_timing_mut_write() {
    let mut pacer = FramePacer::new(None);
    {
        let timing = pacer.timing_mut();
        timing.set_target_fps(Some(30));
    }
    assert!(pacer.timing().target_frame_time().is_some());
}

// =============================================================================
// SECTION 14 -- FramePacer Reset Tests
// =============================================================================

#[test]
fn test_frame_pacer_reset() {
    let mut pacer = FramePacer::new(Some(60));
    pacer.begin_frame();
    pacer.end_frame();
    pacer.begin_frame();
    pacer.end_frame();
    assert_eq!(pacer.frame_count(), 2);
    pacer.reset();
    assert_eq!(pacer.frame_count(), 0);
    assert_eq!(pacer.frames_skipped(), 0);
    assert_eq!(pacer.time_debt(), Duration::ZERO);
}

// =============================================================================
// SECTION 15 -- FramePacer Clone/Debug Tests
// =============================================================================

#[test]
fn test_frame_pacer_clone() {
    let mut pacer = FramePacer::new(Some(60));
    pacer.begin_frame();
    pacer.end_frame();
    let cloned = pacer.clone();
    assert_eq!(cloned.frame_count(), pacer.frame_count());
    assert_eq!(cloned.target_fps(), pacer.target_fps());
}

#[test]
fn test_frame_pacer_debug() {
    let pacer = FramePacer::new(Some(60));
    let debug = format!("{:?}", pacer);
    assert!(debug.contains("FramePacer"));
}

// =============================================================================
// SECTION 16 -- 60 FPS Target Scenario Tests
// =============================================================================

#[test]
fn test_scenario_60fps_stable() {
    let mut pacer = FramePacer::new(Some(60));
    for _ in 0..10 {
        pacer.begin_frame();
        thread::sleep(Duration::from_millis(5));
        pacer.end_frame();
        pacer.wait_for_target();
    }
    let stats = pacer.statistics();
    assert_eq!(stats.sample_count, 10);
}

#[test]
fn test_scenario_60fps_timing_accuracy() {
    let pacer = FramePacer::new(Some(60));
    let target_time = pacer.timing().target_frame_time().unwrap();
    let expected_ms = 1000.0 / 60.0;
    let actual_ms = target_time.as_secs_f64() * 1000.0;
    assert!(
        (actual_ms - expected_ms).abs() < 0.1,
        "60 FPS target {:.3}ms != expected {:.3}ms",
        actual_ms, expected_ms
    );
}

// =============================================================================
// SECTION 17 -- 30 FPS Target Scenario Tests (Half Refresh Rate)
// =============================================================================

#[test]
fn test_scenario_30fps_target() {
    let pacer = FramePacer::new(Some(30));
    let target = pacer.target_fps().unwrap();
    assert!((target - 30.0).abs() < 0.5, "Target FPS {} not ~30", target);
    let target_time = pacer.timing().target_frame_time().unwrap();
    let expected_ms = 1000.0 / 30.0;
    let actual_ms = target_time.as_secs_f64() * 1000.0;
    assert!(
        (actual_ms - expected_ms).abs() < 0.1,
        "30 FPS target {:.3}ms != expected {:.3}ms",
        actual_ms, expected_ms
    );
}

#[test]
fn test_scenario_30fps_is_half_60fps() {
    let pacer_60 = FramePacer::new(Some(60));
    let pacer_30 = FramePacer::new(Some(30));
    let target_60 = pacer_60.timing().target_frame_time().unwrap();
    let target_30 = pacer_30.timing().target_frame_time().unwrap();
    let ratio = target_30.as_secs_f64() / target_60.as_secs_f64();
    assert!(
        (ratio - 2.0).abs() < 0.01,
        "30 FPS target should be 2x 60 FPS, ratio = {}",
        ratio
    );
}

// =============================================================================
// SECTION 18 -- Unlimited Scenario Tests
// =============================================================================

#[test]
fn test_scenario_unlimited_no_wait() {
    let mut pacer = FramePacer::new(None);
    let start = Instant::now();
    for _ in 0..10 {
        pacer.begin_frame();
        pacer.end_frame();
        let waited = pacer.wait_for_target();
        assert_eq!(waited, Duration::ZERO);
    }
    let elapsed = start.elapsed();
    assert!(elapsed < Duration::from_millis(100), "Unlimited mode too slow: {:?}", elapsed);
}

#[test]
fn test_scenario_unlimited_tracks_stats() {
    let mut pacer = FramePacer::new(None);
    for _ in 0..10 {
        pacer.begin_frame();
        thread::sleep(Duration::from_millis(1));
        pacer.end_frame();
    }
    let stats = pacer.statistics();
    assert_eq!(stats.sample_count, 10);
    assert!(stats.avg_frame_time > Duration::ZERO);
}

// =============================================================================
// SECTION 19 -- Frame Skipping Scenario Tests
// =============================================================================

#[test]
fn test_scenario_frame_skip_detection() {
    let mut pacer = FramePacer::new(Some(120)); // 8.33ms target
    pacer.set_skip_threshold(2);
    for _ in 0..5 {
        pacer.begin_frame();
        thread::sleep(Duration::from_millis(50)); // Way behind
        pacer.end_frame();
    }
    // After several slow frames, should_skip_frame is callable
    let _ = pacer.should_skip_frame();
}

#[test]
fn test_scenario_frame_skip_count() {
    let mut pacer = FramePacer::new(Some(60));
    let initial_skipped = pacer.frames_skipped();
    assert_eq!(initial_skipped, 0);
    pacer.begin_frame();
    pacer.end_frame();
    assert_eq!(pacer.frames_skipped(), 0);
}

// =============================================================================
// SECTION 20 -- Statistics Accuracy Tests
// =============================================================================

#[test]
fn test_statistics_accuracy_min_max_avg() {
    let stats = stats_from_ms(&[10, 15, 20, 25, 30]);
    assert_eq!(stats.min_frame_time, Duration::from_millis(10));
    assert_eq!(stats.max_frame_time, Duration::from_millis(30));
    assert_eq!(stats.avg_frame_time, Duration::from_millis(20)); // (10+15+20+25+30)/5 = 20
}

#[test]
fn test_statistics_accuracy_large_sample() {
    let times: Vec<u64> = (1..=100).collect();
    let stats = stats_from_ms(&times);
    assert_eq!(stats.sample_count, 100);
    assert_eq!(stats.min_frame_time, Duration::from_millis(1));
    assert_eq!(stats.max_frame_time, Duration::from_millis(100));
    let avg_ms = stats.avg_frame_time.as_millis();
    assert!(avg_ms == 50 || avg_ms == 51, "avg {} not ~50.5", avg_ms);
}

#[test]
fn test_statistics_accuracy_with_outliers() {
    let stats = stats_from_ms(&[16, 16, 17, 16, 100]); // outlier: 100
    assert_eq!(stats.min_frame_time, Duration::from_millis(16));
    assert_eq!(stats.max_frame_time, Duration::from_millis(100));
    assert_eq!(stats.avg_frame_time, Duration::from_millis(33));
}

// =============================================================================
// SECTION 21 -- Percentile Accuracy Tests
// =============================================================================

#[test]
fn test_percentile_accuracy_p95() {
    let mut times: Vec<u64> = vec![16; 95];
    times.extend(vec![100; 5]);
    let stats = stats_from_ms(&times);
    let p95 = stats.percentile_frame_time(0.95);
    assert!(
        p95 <= Duration::from_millis(100),
        "p95 {:?} should be <= 100ms",
        p95
    );
}

#[test]
fn test_percentile_accuracy_p99() {
    let times: Vec<u64> = (1..=100).collect();
    let stats = stats_from_ms(&times);
    let p99 = stats.percentile_frame_time(0.99);
    assert!(
        p99 >= Duration::from_millis(98) && p99 <= Duration::from_millis(100),
        "p99 {:?} not in expected range",
        p99
    );
}

#[test]
fn test_percentile_two_samples() {
    let stats = stats_from_ms(&[10, 100]);
    assert_eq!(stats.percentile_frame_time(0.0), Duration::from_millis(10));
    assert_eq!(stats.percentile_frame_time(1.0), Duration::from_millis(100));
}

// =============================================================================
// SECTION 22 -- Consistency Check Tests
// =============================================================================

#[test]
fn test_consistency_perfect() {
    let stats = stats_from_ms(&[16, 16, 16, 16, 16, 16, 16, 16, 16, 16]);
    assert!(stats.is_consistent(0.01), "Perfect stability should pass 1% threshold");
}

#[test]
fn test_consistency_minor_variation() {
    let stats = stats_from_ms(&[16, 17, 16, 17, 16, 17, 16, 17]);
    assert!(stats.is_consistent(0.1), "Minor variation should pass 10% threshold");
}

#[test]
fn test_consistency_high_variation() {
    let stats = stats_from_ms(&[5, 50, 10, 45, 8, 55]);
    assert!(!stats.is_consistent(0.1), "High variation should fail 10% threshold");
}

// =============================================================================
// SECTION 23 -- VRR Simulation Tests
// =============================================================================

#[test]
fn test_vrr_simulation_varying_times() {
    let mut pacer = FramePacer::new(None);
    let frame_times = [8, 10, 12, 9, 11, 8, 10, 12, 9, 11];
    for &ms in &frame_times {
        pacer.begin_frame();
        thread::sleep(Duration::from_millis(ms));
        pacer.end_frame();
    }
    let stats = pacer.statistics();
    assert_eq!(stats.sample_count, 10);
    assert!(
        stats.max_frame_time > stats.min_frame_time,
        "VRR should have variable frame times"
    );
}

#[test]
fn test_vrr_statistics_variance() {
    let stats = stats_from_ms(&[8, 10, 12, 14, 16]);
    let variance = stats.frame_time_variance();
    assert!(variance > 0.0, "VRR should have positive variance");
}

#[test]
fn test_vrr_fps_calculation() {
    let stats = stats_from_ms(&[8, 10, 12]);
    let fps = stats.fps();
    assert!(fps > 80.0 && fps < 130.0, "VRR FPS {} not in expected range", fps);
}

// =============================================================================
// SECTION 24 -- Edge Case Tests
// =============================================================================

#[test]
fn test_edge_case_high_fps_target() {
    let pacer = FramePacer::new(Some(240));
    let target = pacer.target_fps().unwrap();
    assert!((target - 240.0).abs() < 1.0, "Target FPS {} not ~240", target);
}

#[test]
fn test_edge_case_low_fps_target() {
    let pacer = FramePacer::new(Some(1));
    let target = pacer.target_fps().unwrap();
    assert!((target - 1.0).abs() < 0.1, "Target FPS {} not ~1", target);
}

#[test]
fn test_edge_case_all_same_values() {
    let stats = stats_from_ms(&[16, 16, 16, 16, 16]);
    assert_eq!(stats.min_frame_time, stats.max_frame_time);
    assert_eq!(stats.min_frame_time, stats.avg_frame_time);
    assert!(stats.frame_time_variance() < 0.00001);
    assert!(stats.is_consistent(0.001));
}

#[test]
fn test_edge_case_all_zero_values() {
    let stats = stats_from_ms(&[0, 0, 0]);
    assert_eq!(stats.min_frame_time, Duration::ZERO);
    assert_eq!(stats.max_frame_time, Duration::ZERO);
    assert_eq!(stats.fps(), 0.0);
}

#[test]
fn test_edge_case_history_size_one() {
    let mut pacer = FramePacer::with_history_size(None, 1);
    for _ in 0..10 {
        pacer.begin_frame();
        pacer.end_frame();
    }
    let stats = pacer.statistics();
    assert_eq!(stats.sample_count, 1);
    assert_eq!(pacer.frame_count(), 10);
}

#[test]
fn test_edge_case_large_history_size() {
    let timing = FrameTiming::with_history_size(1000);
    assert_eq!(timing.history_size(), 1000);
}

#[test]
fn test_edge_case_rapid_cycles() {
    let mut pacer = FramePacer::new(None);
    let start = Instant::now();
    for _ in 0..100 {
        pacer.begin_frame();
        pacer.end_frame();
    }
    let elapsed = start.elapsed();
    assert_eq!(pacer.frame_count(), 100);
    assert!(elapsed < Duration::from_secs(1), "Rapid cycles too slow: {:?}", elapsed);
}

// =============================================================================
// SECTION 25 -- API Completeness Tests
// =============================================================================

#[test]
fn test_frame_timing_api_completeness() {
    let mut timing = FrameTiming::new();
    let _ = timing.frame_count();
    let _ = timing.frame_delta();
    let _ = timing.target_frame_time();
    let _ = timing.in_frame();
    let _ = timing.elapsed();
    let _ = timing.history_size();
    let _ = timing.frame_times();
    timing.set_target_fps(Some(60));
    timing.set_target_frame_time(Some(Duration::from_millis(16)));
    timing.begin_frame();
    timing.end_frame();
    timing.reset();
}

#[test]
fn test_frame_statistics_api_completeness() {
    let stats = stats_from_ms(&[16, 17, 18]);
    let _ = stats.fps();
    let _ = stats.min_fps();
    let _ = stats.max_fps();
    let _ = stats.percentile_frame_time(0.5);
    let _ = stats.median_frame_time();
    let _ = stats.frame_time_variance();
    let _ = stats.frame_time_std_dev();
    let _ = stats.is_consistent(0.1);
    let _ = stats.min_frame_time;
    let _ = stats.max_frame_time;
    let _ = stats.avg_frame_time;
    let _ = stats.sample_count;
}

#[test]
fn test_frame_pacer_api_completeness() {
    let mut pacer = FramePacer::new(Some(60));
    let _ = pacer.target_fps();
    let _ = pacer.is_limiting_enabled();
    let _ = pacer.skip_threshold();
    let _ = pacer.frames_skipped();
    let _ = pacer.time_debt();
    let _ = pacer.frame_count();
    let _ = pacer.frame_delta();
    let _ = pacer.current_fps();
    let _ = pacer.timing();
    let _ = pacer.timing_mut();
    let _ = pacer.statistics();
    pacer.set_target_fps(Some(30));
    pacer.set_skip_threshold(3);
    pacer.set_limiting_enabled(true);
    pacer.begin_frame();
    pacer.end_frame();
    let _ = pacer.wait_for_target();
    let _ = pacer.should_skip_frame();
    pacer.reset();
}

// =============================================================================
// SECTION 26 -- Trait Implementation Tests
// =============================================================================

#[test]
fn test_frame_timing_is_clone() {
    fn assert_clone<T: Clone>() {}
    assert_clone::<FrameTiming>();
}

#[test]
fn test_frame_timing_is_debug() {
    fn assert_debug<T: std::fmt::Debug>() {}
    assert_debug::<FrameTiming>();
}

#[test]
fn test_frame_timing_is_default() {
    fn assert_default<T: Default>() {}
    assert_default::<FrameTiming>();
}

#[test]
fn test_frame_statistics_is_clone() {
    fn assert_clone<T: Clone>() {}
    assert_clone::<FrameStatistics>();
}

#[test]
fn test_frame_statistics_is_debug() {
    fn assert_debug<T: std::fmt::Debug>() {}
    assert_debug::<FrameStatistics>();
}

#[test]
fn test_frame_statistics_is_default() {
    fn assert_default<T: Default>() {}
    assert_default::<FrameStatistics>();
}

#[test]
fn test_frame_statistics_is_partialeq() {
    fn assert_eq<T: PartialEq>() {}
    assert_eq::<FrameStatistics>();
}

#[test]
fn test_frame_statistics_is_display() {
    fn assert_display<T: std::fmt::Display>() {}
    assert_display::<FrameStatistics>();
}

#[test]
fn test_frame_pacer_is_clone() {
    fn assert_clone<T: Clone>() {}
    assert_clone::<FramePacer>();
}

#[test]
fn test_frame_pacer_is_debug() {
    fn assert_debug<T: std::fmt::Debug>() {}
    assert_debug::<FramePacer>();
}

#[test]
fn test_frame_pacer_is_default() {
    fn assert_default<T: Default>() {}
    assert_default::<FramePacer>();
}

// =============================================================================
// SECTION 27 -- Integration Scenario Tests
// =============================================================================

#[test]
fn test_integration_complete_workflow() {
    let mut pacer = FramePacer::new(Some(60));
    for frame_num in 0..20 {
        pacer.begin_frame();
        if pacer.should_skip_frame() {
            continue;
        }
        let work_time = if frame_num % 5 == 0 { 5 } else { 3 };
        thread::sleep(Duration::from_millis(work_time));
        pacer.end_frame();
        let _ = pacer.wait_for_target();
    }
    let stats = pacer.statistics();
    assert!(stats.sample_count > 0);
    assert!(stats.fps() > 0.0);
}

#[test]
fn test_integration_fps_switching() {
    let mut pacer = FramePacer::new(Some(60));
    for _ in 0..5 {
        pacer.begin_frame();
        pacer.end_frame();
    }
    pacer.set_target_fps(Some(30));
    let target = pacer.target_fps().unwrap();
    assert!((target - 30.0).abs() < 0.5);
    for _ in 0..5 {
        pacer.begin_frame();
        pacer.end_frame();
    }
    let stats = pacer.statistics();
    assert_eq!(stats.sample_count, 10);
}

#[test]
fn test_integration_toggle_limiting() {
    let mut pacer = FramePacer::new(Some(60));
    assert!(pacer.is_limiting_enabled());
    pacer.set_limiting_enabled(false);
    assert!(!pacer.is_limiting_enabled());
    pacer.begin_frame();
    pacer.end_frame();
    let waited = pacer.wait_for_target();
    assert_eq!(waited, Duration::ZERO, "Should not wait when disabled");
    pacer.set_limiting_enabled(true);
    assert!(pacer.is_limiting_enabled());
}

// =============================================================================
// SECTION 28 -- Additional Edge Cases
// =============================================================================

#[test]
fn test_percentile_single_sample_all_same() {
    let stats = stats_from_ms(&[42]);
    assert_eq!(stats.percentile_frame_time(0.0), Duration::from_millis(42));
    assert_eq!(stats.percentile_frame_time(0.5), Duration::from_millis(42));
    assert_eq!(stats.percentile_frame_time(1.0), Duration::from_millis(42));
}

#[test]
fn test_frame_pacer_multiple_resets() {
    let mut pacer = FramePacer::new(Some(60));
    for _ in 0..3 {
        pacer.begin_frame();
        pacer.end_frame();
    }
    pacer.reset();
    assert_eq!(pacer.frame_count(), 0);
    for _ in 0..2 {
        pacer.begin_frame();
        pacer.end_frame();
    }
    assert_eq!(pacer.frame_count(), 2);
    pacer.reset();
    assert_eq!(pacer.frame_count(), 0);
}

#[test]
fn test_frame_timing_history_boundary() {
    let mut timing = FrameTiming::with_history_size(3);
    // Record exactly 3 frames
    for _ in 0..3 {
        timing.begin_frame();
        timing.end_frame();
    }
    assert_eq!(timing.frame_times().len(), 3);
    // Record one more
    timing.begin_frame();
    timing.end_frame();
    assert_eq!(timing.frame_times().len(), 3); // Still 3
    assert_eq!(timing.frame_count(), 4);
}

#[test]
fn test_statistics_from_iterator() {
    let durations = vec![
        Duration::from_millis(10),
        Duration::from_millis(20),
        Duration::from_millis(30),
    ];
    let stats = FrameStatistics::from_times(durations.into_iter());
    assert_eq!(stats.sample_count, 3);
    assert_eq!(stats.avg_frame_time, Duration::from_millis(20));
}

#[test]
fn test_frame_pacer_timing_access_consistency() {
    let mut pacer = FramePacer::new(Some(60));
    pacer.begin_frame();
    thread::sleep(Duration::from_millis(2));
    pacer.end_frame();

    // Both should report the same frame count
    assert_eq!(pacer.frame_count(), pacer.timing().frame_count());
    // Frame delta should also match
    assert_eq!(pacer.frame_delta(), pacer.timing().frame_delta());
}

#[test]
fn test_statistics_variance_calculation_accuracy() {
    // Values: 10, 20, 30 -> mean = 20
    // Variance = ((10-20)^2 + (20-20)^2 + (30-20)^2) / (3-1) = (100 + 0 + 100) / 2 = 100
    // In seconds: 0.01, 0.02, 0.03 -> mean = 0.02
    // Variance = ((0.01-0.02)^2 + (0.02-0.02)^2 + (0.03-0.02)^2) / 2
    //          = (0.0001 + 0 + 0.0001) / 2 = 0.0001
    let stats = stats_from_ms(&[10, 20, 30]);
    let variance = stats.frame_time_variance();
    let expected = 0.0001; // in seconds^2
    assert!(
        (variance - expected).abs() < 0.00001,
        "Variance {} != expected {}",
        variance, expected
    );
}

#[test]
fn test_120_fps_target() {
    let pacer = FramePacer::new(Some(120));
    let target = pacer.target_fps().unwrap();
    assert!((target - 120.0).abs() < 0.5, "Target FPS {} not ~120", target);
    let target_time = pacer.timing().target_frame_time().unwrap();
    let expected_ms = 1000.0 / 120.0; // ~8.33ms
    let actual_ms = target_time.as_secs_f64() * 1000.0;
    assert!(
        (actual_ms - expected_ms).abs() < 0.1,
        "120 FPS target {:.3}ms != expected {:.3}ms",
        actual_ms, expected_ms
    );
}

#[test]
fn test_144_fps_target() {
    let pacer = FramePacer::new(Some(144));
    let target = pacer.target_fps().unwrap();
    assert!((target - 144.0).abs() < 0.5, "Target FPS {} not ~144", target);
}

#[test]
fn test_frame_times_ordering_preserved_in_stats() {
    // FrameStatistics sorts times internally but we verify the results are consistent
    let stats = stats_from_ms(&[50, 10, 30, 20, 40]);
    // After sorting: 10, 20, 30, 40, 50
    assert_eq!(stats.min_frame_time, Duration::from_millis(10));
    assert_eq!(stats.max_frame_time, Duration::from_millis(50));
    assert_eq!(stats.median_frame_time(), Duration::from_millis(30));
}
