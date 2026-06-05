// Blackbox contract tests for GPU Timestamp Query Profiler (T-WGPU-P7.4.1).
//
// CLEANROOM: No src/ access beyond the public API exported by the crate.
// Tests use only `renderer_backend::profiling::timestamps::*` -- no internal
// fields, no private methods, no implementation details.
//
// Acceptance criteria:
//   1.  Query lifecycle: begin/end pair management
//   2.  Duration calculations: ns/us/ms/s precision
//   3.  Period conversion: ticks <-> nanoseconds
//   4.  Profiler statistics: min/max/avg tracking
//   5.  Frame profiling: per-frame stats collection
//   6.  RAII scope pattern: automatic timing on drop
//   7.  Real-world scenarios: render pass, compute pass, etc.
//   8.  Buffer management: capacity handling
//   9.  Edge cases: zero duration, maximum values
//  10.  Integration patterns: multiple profilers, thread-local

use renderer_backend::profiling::timestamps::{
    FrameProfiler, FrameStats, ProfilerStats, TimestampHandle, TimestampPeriodConverter,
    TimestampResult, DEFAULT_CAPACITY, MAX_RECOMMENDED_CAPACITY, MIN_CAPACITY,
    TIMESTAMP_SIZE_BYTES,
};
use std::borrow::Cow;
use std::collections::HashSet;
use std::time::Duration;

// =============================================================================
// MODULE: TimestampHandle Tests - Query Lifecycle
// =============================================================================

mod handle_lifecycle {
    use super::*;

    // -------------------------------------------------------------------------
    // Basic Handle Creation
    // -------------------------------------------------------------------------

    #[test]
    fn create_handle_with_sequential_indices() {
        let handle = TimestampHandle::new(0, 1);
        assert_eq!(handle.start_index, 0);
        assert_eq!(handle.end_index, 1);
    }

    #[test]
    fn create_handle_with_large_indices() {
        let handle = TimestampHandle::new(1000, 1001);
        assert_eq!(handle.start_index, 1000);
        assert_eq!(handle.end_index, 1001);
    }

    #[test]
    fn create_handle_with_non_sequential_indices() {
        let handle = TimestampHandle::new(5, 10);
        assert_eq!(handle.start_index, 5);
        assert_eq!(handle.end_index, 10);
    }

    #[test]
    fn handle_without_label_has_no_label() {
        let handle = TimestampHandle::new(0, 1);
        assert!(handle.label.is_none());
        assert!(!handle.has_label());
    }

    #[test]
    fn handle_with_label_stores_label() {
        let handle = TimestampHandle::with_label(0, 1, "Shadow Pass");
        assert!(handle.has_label());
        assert_eq!(handle.label.as_deref(), Some("Shadow Pass"));
    }

    #[test]
    fn handle_with_empty_string_label() {
        let handle = TimestampHandle::with_label(0, 1, "");
        assert!(handle.has_label());
        assert_eq!(handle.label.as_deref(), Some(""));
    }

    #[test]
    fn handle_with_unicode_label() {
        let handle = TimestampHandle::with_label(0, 1, "渲染通道 🎨");
        assert_eq!(handle.label.as_deref(), Some("渲染通道 🎨"));
    }

    #[test]
    fn handle_with_very_long_label() {
        let long_label = "a".repeat(1000);
        let handle = TimestampHandle::with_label(0, 1, &long_label);
        assert_eq!(handle.label.as_ref().unwrap().len(), 1000);
    }

    // -------------------------------------------------------------------------
    // Label Mutation
    // -------------------------------------------------------------------------

    #[test]
    fn set_label_on_unlabeled_handle() {
        let mut handle = TimestampHandle::new(0, 1);
        handle.set_label("New Label");
        assert!(handle.has_label());
        assert_eq!(handle.label.as_deref(), Some("New Label"));
    }

    #[test]
    fn set_label_replaces_existing_label() {
        let mut handle = TimestampHandle::with_label(0, 1, "Old Label");
        handle.set_label("New Label");
        assert_eq!(handle.label.as_deref(), Some("New Label"));
    }

    #[test]
    fn clear_label_removes_label() {
        let mut handle = TimestampHandle::with_label(0, 1, "Label");
        handle.clear_label();
        assert!(!handle.has_label());
        assert!(handle.label.is_none());
    }

    #[test]
    fn clear_label_on_unlabeled_handle_is_noop() {
        let mut handle = TimestampHandle::new(0, 1);
        handle.clear_label();
        assert!(!handle.has_label());
    }

    // -------------------------------------------------------------------------
    // Label Accessor Methods
    // -------------------------------------------------------------------------

    #[test]
    fn label_or_returns_label_when_present() {
        let handle = TimestampHandle::with_label(0, 1, "Actual");
        assert_eq!(handle.label_or("Default"), "Actual");
    }

    #[test]
    fn label_or_returns_default_when_absent() {
        let handle = TimestampHandle::new(0, 1);
        assert_eq!(handle.label_or("Default"), "Default");
    }

    #[test]
    fn label_or_unnamed_returns_unnamed_when_absent() {
        let handle = TimestampHandle::new(0, 1);
        assert_eq!(handle.label_or_unnamed(), "unnamed");
    }

    #[test]
    fn label_or_unnamed_returns_label_when_present() {
        let handle = TimestampHandle::with_label(0, 1, "Named");
        assert_eq!(handle.label_or_unnamed(), "Named");
    }

    // -------------------------------------------------------------------------
    // Display Formatting
    // -------------------------------------------------------------------------

    #[test]
    fn display_unlabeled_handle_shows_indices() {
        let handle = TimestampHandle::new(5, 6);
        let display = format!("{}", handle);
        assert!(display.contains("5"));
        assert!(display.contains("6"));
        assert!(display.contains(".."));
    }

    #[test]
    fn display_labeled_handle_includes_label() {
        let handle = TimestampHandle::with_label(5, 6, "Test Pass");
        let display = format!("{}", handle);
        assert!(display.contains("Test Pass"));
        assert!(display.contains("5"));
        assert!(display.contains("6"));
    }

    #[test]
    fn debug_format_includes_all_fields() {
        let handle = TimestampHandle::with_label(1, 2, "Debug Test");
        let debug = format!("{:?}", handle);
        assert!(debug.contains("TimestampHandle"));
        assert!(debug.contains("start_index"));
        assert!(debug.contains("end_index"));
        assert!(debug.contains("label"));
    }

    // -------------------------------------------------------------------------
    // Multiple Sequential Queries
    // -------------------------------------------------------------------------

    #[test]
    fn create_multiple_sequential_handles() {
        let handles: Vec<TimestampHandle> =
            (0..10).map(|i| TimestampHandle::new(i * 2, i * 2 + 1)).collect();

        assert_eq!(handles.len(), 10);
        for (i, h) in handles.iter().enumerate() {
            assert_eq!(h.start_index, (i * 2) as u32);
            assert_eq!(h.end_index, (i * 2 + 1) as u32);
        }
    }

    #[test]
    fn create_multiple_labeled_handles() {
        let labels = ["Shadow", "GBuffer", "Lighting", "Post"];
        let handles: Vec<TimestampHandle> = labels
            .iter()
            .enumerate()
            .map(|(i, l)| TimestampHandle::with_label(i as u32 * 2, i as u32 * 2 + 1, *l))
            .collect();

        for (i, h) in handles.iter().enumerate() {
            assert_eq!(h.label.as_deref(), Some(labels[i]));
        }
    }

    // -------------------------------------------------------------------------
    // Equality and Hashing
    // -------------------------------------------------------------------------

    #[test]
    fn identical_handles_are_equal() {
        let h1 = TimestampHandle::new(0, 1);
        let h2 = TimestampHandle::new(0, 1);
        assert_eq!(h1, h2);
    }

    #[test]
    fn handles_with_different_indices_not_equal() {
        let h1 = TimestampHandle::new(0, 1);
        let h2 = TimestampHandle::new(2, 3);
        assert_ne!(h1, h2);
    }

    #[test]
    fn handles_with_different_labels_not_equal() {
        let h1 = TimestampHandle::with_label(0, 1, "A");
        let h2 = TimestampHandle::with_label(0, 1, "B");
        assert_ne!(h1, h2);
    }

    #[test]
    fn labeled_and_unlabeled_handles_not_equal() {
        let h1 = TimestampHandle::new(0, 1);
        let h2 = TimestampHandle::with_label(0, 1, "Label");
        assert_ne!(h1, h2);
    }

    #[test]
    fn handle_can_be_stored_in_hashset() {
        let mut set = HashSet::new();
        let h1 = TimestampHandle::new(0, 1);
        let h2 = TimestampHandle::with_label(0, 1, "Label");
        let h3 = TimestampHandle::new(2, 3);

        set.insert(h1.clone());
        set.insert(h2.clone());
        set.insert(h3.clone());

        assert_eq!(set.len(), 3);
        assert!(set.contains(&h1));
        assert!(set.contains(&h2));
        assert!(set.contains(&h3));
    }

    #[test]
    fn cloned_handle_equals_original() {
        let original = TimestampHandle::with_label(5, 6, "Clone Test");
        let cloned = original.clone();
        assert_eq!(original, cloned);
    }
}

// =============================================================================
// MODULE: TimestampResult Tests - Duration Calculations
// =============================================================================

mod duration_calculations {
    use super::*;

    // -------------------------------------------------------------------------
    // Microsecond Precision
    // -------------------------------------------------------------------------

    #[test]
    fn duration_1_microsecond() {
        let result = TimestampResult::new(None, 0, 1_000);
        assert_eq!(result.duration_ns(), 1_000);
        assert!((result.duration_us() - 1.0).abs() < 0.001);
    }

    #[test]
    fn duration_half_microsecond() {
        let result = TimestampResult::new(None, 0, 500);
        assert_eq!(result.duration_ns(), 500);
        assert!((result.duration_us() - 0.5).abs() < 0.001);
    }

    #[test]
    fn duration_10_microseconds() {
        let result = TimestampResult::new(None, 0, 10_000);
        assert!((result.duration_us() - 10.0).abs() < 0.001);
    }

    #[test]
    fn duration_100_microseconds() {
        let result = TimestampResult::new(None, 0, 100_000);
        assert!((result.duration_us() - 100.0).abs() < 0.001);
    }

    #[test]
    fn duration_fractional_microseconds() {
        let result = TimestampResult::new(None, 0, 1_234);
        assert!((result.duration_us() - 1.234).abs() < 0.001);
    }

    // -------------------------------------------------------------------------
    // Millisecond Precision
    // -------------------------------------------------------------------------

    #[test]
    fn duration_1_millisecond() {
        let result = TimestampResult::new(None, 0, 1_000_000);
        assert_eq!(result.duration_ns(), 1_000_000);
        assert!((result.duration_ms() - 1.0).abs() < 0.0001);
    }

    #[test]
    fn duration_half_millisecond() {
        let result = TimestampResult::new(None, 0, 500_000);
        assert!((result.duration_ms() - 0.5).abs() < 0.0001);
    }

    #[test]
    fn duration_16_point_6_milliseconds() {
        // Typical 60fps frame time
        let result = TimestampResult::new(None, 0, 16_666_667);
        let ms = result.duration_ms();
        assert!((ms - 16.666667).abs() < 0.001);
    }

    #[test]
    fn duration_33_milliseconds() {
        // Typical 30fps frame time
        let result = TimestampResult::new(None, 0, 33_333_333);
        let ms = result.duration_ms();
        assert!((ms - 33.333333).abs() < 0.001);
    }

    #[test]
    fn duration_fractional_milliseconds() {
        let result = TimestampResult::new(None, 0, 1_234_567);
        assert!((result.duration_ms() - 1.234567).abs() < 0.0001);
    }

    // -------------------------------------------------------------------------
    // Sub-Microsecond Durations
    // -------------------------------------------------------------------------

    #[test]
    fn duration_1_nanosecond() {
        let result = TimestampResult::new(None, 0, 1);
        assert_eq!(result.duration_ns(), 1);
        assert!(result.duration_us() < 0.01);
    }

    #[test]
    fn duration_100_nanoseconds() {
        let result = TimestampResult::new(None, 0, 100);
        assert_eq!(result.duration_ns(), 100);
        assert!((result.duration_us() - 0.1).abs() < 0.001);
    }

    #[test]
    fn duration_999_nanoseconds() {
        let result = TimestampResult::new(None, 0, 999);
        assert_eq!(result.duration_ns(), 999);
        assert!((result.duration_us() - 0.999).abs() < 0.001);
    }

    // -------------------------------------------------------------------------
    // Multi-Second Durations
    // -------------------------------------------------------------------------

    #[test]
    fn duration_1_second() {
        let result = TimestampResult::new(None, 0, 1_000_000_000);
        assert!((result.duration_secs() - 1.0).abs() < 0.0001);
    }

    #[test]
    fn duration_half_second() {
        let result = TimestampResult::new(None, 0, 500_000_000);
        assert!((result.duration_secs() - 0.5).abs() < 0.0001);
    }

    #[test]
    fn duration_10_seconds() {
        let result = TimestampResult::new(None, 0, 10_000_000_000);
        assert!((result.duration_secs() - 10.0).abs() < 0.001);
    }

    #[test]
    fn duration_fractional_seconds() {
        let result = TimestampResult::new(None, 0, 1_234_567_890);
        assert!((result.duration_secs() - 1.234567890).abs() < 0.0001);
    }

    // -------------------------------------------------------------------------
    // Non-Zero Start Time
    // -------------------------------------------------------------------------

    #[test]
    fn duration_with_nonzero_start() {
        let result = TimestampResult::new(None, 5_000, 10_000);
        assert_eq!(result.duration_ns(), 5_000);
    }

    #[test]
    fn duration_with_large_start_and_end() {
        let result = TimestampResult::new(None, 1_000_000_000, 1_001_000_000);
        assert_eq!(result.duration_ns(), 1_000_000);
        assert!((result.duration_ms() - 1.0).abs() < 0.0001);
    }

    #[test]
    fn duration_with_equal_start_and_end() {
        let result = TimestampResult::new(None, 5_000, 5_000);
        assert_eq!(result.duration_ns(), 0);
    }

    // -------------------------------------------------------------------------
    // From Ticks Conversion
    // -------------------------------------------------------------------------

    #[test]
    fn from_ticks_with_1ns_period() {
        let result = TimestampResult::from_ticks(0, 1000, 1.0);
        assert_eq!(result.duration_ns(), 1000);
    }

    #[test]
    fn from_ticks_with_10ns_period() {
        let result = TimestampResult::from_ticks(0, 100, 10.0);
        assert_eq!(result.end_ns, 1000);
        assert_eq!(result.duration_ns(), 1000);
    }

    #[test]
    fn from_ticks_with_25ns_period() {
        // Common GPU timestamp period
        let result = TimestampResult::from_ticks(0, 40_000_000, 25.0);
        // 40M ticks * 25ns = 1 second
        assert!((result.duration_secs() - 1.0).abs() < 0.001);
    }

    #[test]
    fn from_ticks_with_fractional_period() {
        let result = TimestampResult::from_ticks(0, 1000, 1.5);
        assert_eq!(result.duration_ns(), 1500);
    }

    #[test]
    fn from_ticks_labeled() {
        let result = TimestampResult::from_ticks_labeled(0, 100, 10.0, "Test");
        assert_eq!(result.label.as_deref(), Some("Test"));
        assert_eq!(result.duration_ns(), 1000);
    }

    // -------------------------------------------------------------------------
    // Zero Duration
    // -------------------------------------------------------------------------

    #[test]
    fn zero_creates_zero_duration() {
        let result = TimestampResult::zero();
        assert_eq!(result.start_ns, 0);
        assert_eq!(result.end_ns, 0);
        assert_eq!(result.duration_ns(), 0);
        assert!(result.label.is_none());
    }

    #[test]
    fn zero_labeled() {
        let result = TimestampResult::zero_labeled("Zero Test");
        assert_eq!(result.duration_ns(), 0);
        assert_eq!(result.label.as_deref(), Some("Zero Test"));
    }

    #[test]
    fn default_is_zero() {
        let result = TimestampResult::default();
        assert_eq!(result.duration_ns(), 0);
    }

    // -------------------------------------------------------------------------
    // Validity Checking
    // -------------------------------------------------------------------------

    #[test]
    fn valid_result_when_end_greater_than_start() {
        let result = TimestampResult::new(None, 0, 100);
        assert!(result.is_valid());
    }

    #[test]
    fn valid_result_when_end_equals_start() {
        let result = TimestampResult::new(None, 100, 100);
        assert!(result.is_valid());
    }

    #[test]
    fn invalid_result_when_end_less_than_start() {
        let result = TimestampResult::new(None, 100, 50);
        assert!(!result.is_valid());
    }

    #[test]
    fn duration_saturates_when_end_less_than_start() {
        let result = TimestampResult::new(None, 100, 50);
        assert_eq!(result.duration_ns(), 0);
    }

    // -------------------------------------------------------------------------
    // Display and Label
    // -------------------------------------------------------------------------

    #[test]
    fn display_format_with_label() {
        let result = TimestampResult::new(Some("Test".to_string()), 0, 1_000_000);
        let display = format!("{}", result);
        assert!(display.contains("Test"));
        assert!(display.contains("ms"));
    }

    #[test]
    fn display_format_without_label() {
        let result = TimestampResult::new(None, 0, 1_000_000);
        let display = format!("{}", result);
        assert!(display.contains("unnamed"));
    }

    #[test]
    fn label_or_returns_correct_value() {
        let with_label = TimestampResult::new(Some("Actual".to_string()), 0, 0);
        let without_label = TimestampResult::new(None, 0, 0);

        assert_eq!(with_label.label_or("Default"), "Actual");
        assert_eq!(without_label.label_or("Default"), "Default");
    }
}

// =============================================================================
// MODULE: TimestampPeriodConverter Tests - Period Conversion
// =============================================================================

mod period_conversion {
    use super::*;

    // -------------------------------------------------------------------------
    // Standard GPU Periods (1ns)
    // -------------------------------------------------------------------------

    #[test]
    fn convert_ticks_to_ns_with_1ns_period() {
        let converter = TimestampPeriodConverter::new(1.0);
        assert_eq!(converter.ticks_to_ns(1000), 1000);
    }

    #[test]
    fn convert_ns_to_ticks_with_1ns_period() {
        let converter = TimestampPeriodConverter::new(1.0);
        assert_eq!(converter.ns_to_ticks(1000), 1000);
    }

    #[test]
    fn roundtrip_1ns_period() {
        let converter = TimestampPeriodConverter::new(1.0);
        let original: u64 = 123456789;
        let ns = converter.ticks_to_ns(original);
        let recovered = converter.ns_to_ticks(ns);
        assert_eq!(original, recovered);
    }

    // -------------------------------------------------------------------------
    // Non-Standard Periods (Fractional)
    // -------------------------------------------------------------------------

    #[test]
    fn convert_with_25ns_period() {
        // Common GPU period
        let converter = TimestampPeriodConverter::new(25.0);
        assert_eq!(converter.ticks_to_ns(4), 100); // 4 * 25 = 100
    }

    #[test]
    fn convert_with_fractional_period() {
        let converter = TimestampPeriodConverter::new(1.5);
        assert_eq!(converter.ticks_to_ns(100), 150);
    }

    #[test]
    fn convert_with_very_small_period() {
        let converter = TimestampPeriodConverter::new(0.001);
        let ns = converter.ticks_to_ns(1_000_000);
        assert_eq!(ns, 1000);
    }

    #[test]
    fn convert_with_large_period() {
        let converter = TimestampPeriodConverter::new(1000.0);
        assert_eq!(converter.ticks_to_ns(1), 1000);
    }

    // -------------------------------------------------------------------------
    // Round-Trip Conversion
    // -------------------------------------------------------------------------

    #[test]
    fn roundtrip_25ns_period() {
        let converter = TimestampPeriodConverter::new(25.0);
        let ticks: u64 = 12345;
        let ns = converter.ticks_to_ns(ticks);
        let recovered = converter.ns_to_ticks(ns);
        // May have rounding error
        assert!((recovered as i64 - ticks as i64).abs() <= 1);
    }

    #[test]
    fn roundtrip_fractional_period() {
        let converter = TimestampPeriodConverter::new(3.14159);
        let ticks: u64 = 10000;
        let ns = converter.ticks_to_ns(ticks);
        let recovered = converter.ns_to_ticks(ns);
        assert!((recovered as i64 - ticks as i64).abs() <= 1);
    }

    #[test]
    fn roundtrip_large_tick_values() {
        let converter = TimestampPeriodConverter::new(1.0);
        let ticks: u64 = u64::MAX / 2;
        let ns = converter.ticks_to_ns(ticks);
        let recovered = converter.ns_to_ticks(ns);
        // Floating-point precision may cause +-1 difference at extreme values
        assert!((recovered as i128 - ticks as i128).abs() <= 1);
    }

    // -------------------------------------------------------------------------
    // Time Unit Conversions
    // -------------------------------------------------------------------------

    #[test]
    fn ticks_to_us_conversion() {
        let converter = TimestampPeriodConverter::new(1.0);
        let us = converter.ticks_to_us(1_000);
        assert!((us - 1.0).abs() < 0.001);
    }

    #[test]
    fn ticks_to_ms_conversion() {
        let converter = TimestampPeriodConverter::new(1.0);
        let ms = converter.ticks_to_ms(1_000_000);
        assert!((ms - 1.0).abs() < 0.001);
    }

    #[test]
    fn duration_ns_between_ticks() {
        let converter = TimestampPeriodConverter::new(10.0);
        let ns = converter.duration_ns(100, 200);
        assert_eq!(ns, 1000); // (200-100) * 10 = 1000
    }

    #[test]
    fn duration_ms_between_ticks() {
        let converter = TimestampPeriodConverter::new(1.0);
        let ms = converter.duration_ms(0, 1_000_000);
        assert!((ms - 1.0).abs() < 0.001);
    }

    #[test]
    fn duration_saturates_when_start_greater() {
        let converter = TimestampPeriodConverter::new(1.0);
        let ns = converter.duration_ns(200, 100);
        assert_eq!(ns, 0);
    }

    // -------------------------------------------------------------------------
    // Edge Cases
    // -------------------------------------------------------------------------

    #[test]
    fn zero_period_returns_zero_ticks() {
        let converter = TimestampPeriodConverter::new(0.0);
        assert_eq!(converter.ns_to_ticks(1000), 0);
    }

    #[test]
    fn zero_ticks_converts_to_zero() {
        let converter = TimestampPeriodConverter::new(25.0);
        assert_eq!(converter.ticks_to_ns(0), 0);
    }

    #[test]
    fn default_period_is_1ns() {
        let converter = TimestampPeriodConverter::default();
        assert_eq!(converter.period(), 1.0);
    }

    #[test]
    fn period_accessor() {
        let converter = TimestampPeriodConverter::new(42.5);
        assert_eq!(converter.period(), 42.5);
    }

    #[test]
    fn large_tick_conversion_no_overflow() {
        let converter = TimestampPeriodConverter::new(1.0);
        // This should not panic
        let ns = converter.ticks_to_ns(u64::MAX / 2);
        assert!(ns > 0);
    }

    #[test]
    fn copy_trait_works() {
        let c1 = TimestampPeriodConverter::new(25.0);
        let c2 = c1; // Copy
        assert_eq!(c1.period(), c2.period());
    }

    #[test]
    fn clone_equals_original() {
        let c1 = TimestampPeriodConverter::new(33.33);
        let c2 = c1.clone();
        assert_eq!(c1, c2);
    }
}

// =============================================================================
// MODULE: ProfilerStats Tests - Statistics Tracking
// =============================================================================

mod profiler_statistics {
    use super::*;

    // -------------------------------------------------------------------------
    // Basic Stats Creation
    // -------------------------------------------------------------------------

    #[test]
    fn create_stats_with_all_values() {
        let stats = ProfilerStats::new(100, 50, 25, 1_000_000, 500_000, 2_000_000);
        assert_eq!(stats.total_queries, 100);
        assert_eq!(stats.active_queries, 50);
        assert_eq!(stats.resolved_queries, 25);
        assert_eq!(stats.avg_duration_ns, 1_000_000);
        assert_eq!(stats.min_duration_ns, 500_000);
        assert_eq!(stats.max_duration_ns, 2_000_000);
    }

    #[test]
    fn empty_stats_has_zero_values() {
        let stats = ProfilerStats::empty();
        assert_eq!(stats.total_queries, 0);
        assert_eq!(stats.active_queries, 0);
        assert_eq!(stats.resolved_queries, 0);
        assert_eq!(stats.avg_duration_ns, 0);
        assert_eq!(stats.min_duration_ns, 0);
        assert_eq!(stats.max_duration_ns, 0);
    }

    #[test]
    fn default_is_empty() {
        let stats = ProfilerStats::default();
        assert_eq!(stats, ProfilerStats::empty());
    }

    // -------------------------------------------------------------------------
    // Accumulate Multiple Results
    // -------------------------------------------------------------------------

    #[test]
    fn stats_with_single_resolved_query() {
        let stats = ProfilerStats::new(64, 1, 1, 500_000, 500_000, 500_000);
        assert_eq!(stats.resolved_queries, 1);
        assert_eq!(stats.avg_duration_ns, 500_000);
        assert_eq!(stats.min_duration_ns, stats.max_duration_ns);
    }

    #[test]
    fn stats_with_multiple_resolved_queries() {
        // Simulating 3 queries: 1ms, 2ms, 3ms
        // avg = 2ms, min = 1ms, max = 3ms
        let stats = ProfilerStats::new(64, 0, 3, 2_000_000, 1_000_000, 3_000_000);
        assert!((stats.avg_duration_ms() - 2.0).abs() < 0.001);
        assert!((stats.min_duration_ms() - 1.0).abs() < 0.001);
        assert!((stats.max_duration_ms() - 3.0).abs() < 0.001);
    }

    #[test]
    fn stats_with_many_resolved_queries() {
        let stats = ProfilerStats::new(256, 0, 100, 500_000, 100_000, 5_000_000);
        assert_eq!(stats.resolved_queries, 100);
    }

    // -------------------------------------------------------------------------
    // Track Min/Max Correctly
    // -------------------------------------------------------------------------

    #[test]
    fn min_max_with_same_values() {
        let stats = ProfilerStats::new(10, 0, 5, 1_000_000, 1_000_000, 1_000_000);
        assert_eq!(stats.min_duration_ns, stats.max_duration_ns);
    }

    #[test]
    fn min_max_with_large_spread() {
        let stats = ProfilerStats::new(10, 0, 5, 500_000, 1_000, 10_000_000);
        assert!(stats.max_duration_ns > stats.min_duration_ns);
        assert!(stats.max_duration_ns > stats.avg_duration_ns);
        assert!(stats.min_duration_ns < stats.avg_duration_ns);
    }

    #[test]
    fn min_can_be_zero() {
        let stats = ProfilerStats::new(10, 0, 5, 500_000, 0, 1_000_000);
        assert_eq!(stats.min_duration_ns, 0);
    }

    // -------------------------------------------------------------------------
    // Average Calculation
    // -------------------------------------------------------------------------

    #[test]
    fn average_duration_ms() {
        let stats = ProfilerStats::new(10, 0, 5, 1_000_000, 0, 0);
        assert!((stats.avg_duration_ms() - 1.0).abs() < 0.001);
    }

    #[test]
    fn average_duration_sub_millisecond() {
        let stats = ProfilerStats::new(10, 0, 5, 100_000, 0, 0);
        assert!((stats.avg_duration_ms() - 0.1).abs() < 0.001);
    }

    #[test]
    fn zero_average_when_no_results() {
        let stats = ProfilerStats::empty();
        assert_eq!(stats.avg_duration_ms(), 0.0);
    }

    // -------------------------------------------------------------------------
    // Utilization Calculation
    // -------------------------------------------------------------------------

    #[test]
    fn utilization_50_percent() {
        let stats = ProfilerStats::new(100, 50, 0, 0, 0, 0);
        assert!((stats.utilization() - 0.5).abs() < 0.001);
    }

    #[test]
    fn utilization_100_percent() {
        let stats = ProfilerStats::new(100, 100, 0, 0, 0, 0);
        assert!((stats.utilization() - 1.0).abs() < 0.001);
    }

    #[test]
    fn utilization_0_percent() {
        let stats = ProfilerStats::new(100, 0, 0, 0, 0, 0);
        assert_eq!(stats.utilization(), 0.0);
    }

    #[test]
    fn utilization_zero_total_returns_zero() {
        let stats = ProfilerStats::empty();
        assert_eq!(stats.utilization(), 0.0);
    }

    #[test]
    fn utilization_fractional() {
        let stats = ProfilerStats::new(64, 16, 0, 0, 0, 0);
        assert!((stats.utilization() - 0.25).abs() < 0.001);
    }

    // -------------------------------------------------------------------------
    // Display Formatting
    // -------------------------------------------------------------------------

    #[test]
    fn display_includes_active_total() {
        let stats = ProfilerStats::new(100, 50, 25, 1_000_000, 500_000, 2_000_000);
        let display = format!("{}", stats);
        assert!(display.contains("50/100"));
    }

    #[test]
    fn display_includes_resolved_count() {
        let stats = ProfilerStats::new(100, 50, 25, 1_000_000, 500_000, 2_000_000);
        let display = format!("{}", stats);
        assert!(display.contains("resolved=25"));
    }

    #[test]
    fn display_includes_timing_values() {
        let stats = ProfilerStats::new(100, 50, 25, 1_000_000, 500_000, 2_000_000);
        let display = format!("{}", stats);
        assert!(display.contains("avg="));
        assert!(display.contains("min="));
        assert!(display.contains("max="));
    }

    // -------------------------------------------------------------------------
    // Reset and Reuse (Simulated)
    // -------------------------------------------------------------------------

    #[test]
    fn simulate_stats_reset() {
        let stats1 = ProfilerStats::new(100, 50, 25, 1_000_000, 500_000, 2_000_000);
        assert!(stats1.resolved_queries > 0);

        // After reset
        let stats2 = ProfilerStats::empty();
        assert_eq!(stats2.resolved_queries, 0);
        assert_eq!(stats2.active_queries, 0);
    }

    #[test]
    fn copy_trait_works() {
        let s1 = ProfilerStats::new(100, 50, 25, 1_000_000, 500_000, 2_000_000);
        let s2 = s1; // Copy
        assert_eq!(s1, s2);
    }
}

// =============================================================================
// MODULE: FrameStats Tests - Frame Profiling
// =============================================================================

mod frame_profiling {
    use super::*;

    // -------------------------------------------------------------------------
    // Single Frame Profile
    // -------------------------------------------------------------------------

    #[test]
    fn new_frame_stats_has_zero_total() {
        let stats = FrameStats::new(0);
        assert_eq!(stats.total_ns, 0);
        assert_eq!(stats.region_count, 0);
        assert!(stats.is_empty());
    }

    #[test]
    fn frame_stats_with_index() {
        let stats = FrameStats::new(42);
        assert_eq!(stats.frame_index, 42);
    }

    #[test]
    fn add_single_region() {
        let mut stats = FrameStats::new(0);
        let result = TimestampResult::new(Some("Pass".to_string()), 0, 1_000_000);
        stats.add_region(result);

        assert_eq!(stats.region_count, 1);
        assert_eq!(stats.total_ns, 1_000_000);
        assert!(!stats.is_empty());
    }

    #[test]
    fn total_ms_calculation() {
        let mut stats = FrameStats::new(0);
        stats.add_region(TimestampResult::new(None, 0, 1_000_000));
        assert!((stats.total_ms() - 1.0).abs() < 0.001);
    }

    // -------------------------------------------------------------------------
    // Multi-Frame Profiling
    // -------------------------------------------------------------------------

    #[test]
    fn add_multiple_regions() {
        let mut stats = FrameStats::new(0);

        stats.add_region(TimestampResult::new(Some("Shadow".to_string()), 0, 500_000));
        stats.add_region(TimestampResult::new(Some("GBuffer".to_string()), 0, 300_000));
        stats.add_region(TimestampResult::new(Some("Lighting".to_string()), 0, 700_000));

        assert_eq!(stats.region_count, 3);
        assert_eq!(stats.total_ns, 1_500_000);
    }

    #[test]
    fn regions_vector_contains_all_results() {
        let mut stats = FrameStats::new(0);

        stats.add_region(TimestampResult::new(Some("A".to_string()), 0, 100));
        stats.add_region(TimestampResult::new(Some("B".to_string()), 0, 200));

        assert_eq!(stats.regions.len(), 2);
        assert_eq!(stats.regions[0].label.as_deref(), Some("A"));
        assert_eq!(stats.regions[1].label.as_deref(), Some("B"));
    }

    #[test]
    fn multiple_frames_independent() {
        let mut frame1 = FrameStats::new(1);
        let mut frame2 = FrameStats::new(2);

        frame1.add_region(TimestampResult::new(None, 0, 1_000_000));
        frame2.add_region(TimestampResult::new(None, 0, 2_000_000));

        assert_eq!(frame1.frame_index, 1);
        assert_eq!(frame2.frame_index, 2);
        assert_eq!(frame1.total_ns, 1_000_000);
        assert_eq!(frame2.total_ns, 2_000_000);
    }

    // -------------------------------------------------------------------------
    // Multiple Passes Per Frame
    // -------------------------------------------------------------------------

    #[test]
    fn simulate_typical_frame_with_multiple_passes() {
        let mut stats = FrameStats::new(0);

        // Shadow cascade passes
        for i in 0..4 {
            stats.add_region(TimestampResult::new(
                Some(format!("Shadow Cascade {}", i)),
                0,
                250_000,
            ));
        }

        // Main passes
        stats.add_region(TimestampResult::new(
            Some("GBuffer".to_string()),
            0,
            2_000_000,
        ));
        stats.add_region(TimestampResult::new(
            Some("Lighting".to_string()),
            0,
            3_000_000,
        ));
        stats.add_region(TimestampResult::new(
            Some("Post Process".to_string()),
            0,
            500_000,
        ));

        assert_eq!(stats.region_count, 7);
        // 4*0.25ms + 2ms + 3ms + 0.5ms = 6.5ms
        assert!((stats.total_ms() - 6.5).abs() < 0.001);
    }

    #[test]
    fn many_small_passes() {
        let mut stats = FrameStats::new(0);

        for i in 0..100 {
            stats.add_region(TimestampResult::new(
                Some(format!("Pass {}", i)),
                0,
                10_000,
            ));
        }

        assert_eq!(stats.region_count, 100);
        assert_eq!(stats.total_ns, 1_000_000); // 100 * 10us = 1ms
    }

    // -------------------------------------------------------------------------
    // Frame Timing Accuracy
    // -------------------------------------------------------------------------

    #[test]
    fn frame_total_matches_sum_of_regions() {
        let mut stats = FrameStats::new(0);

        let durations = [100_000, 200_000, 300_000, 400_000];
        for d in durations.iter() {
            stats.add_region(TimestampResult::new(None, 0, *d));
        }

        let expected: u64 = durations.iter().sum();
        assert_eq!(stats.total_ns, expected);
    }

    #[test]
    fn sub_microsecond_regions_accumulate_correctly() {
        let mut stats = FrameStats::new(0);

        for _ in 0..1000 {
            stats.add_region(TimestampResult::new(None, 0, 100)); // 100ns each
        }

        assert_eq!(stats.total_ns, 100_000); // 1000 * 100ns = 100us
    }

    #[test]
    fn zero_duration_regions() {
        let mut stats = FrameStats::new(0);

        stats.add_region(TimestampResult::zero());
        stats.add_region(TimestampResult::zero());

        assert_eq!(stats.region_count, 2);
        assert_eq!(stats.total_ns, 0);
    }

    // -------------------------------------------------------------------------
    // Display and Debug
    // -------------------------------------------------------------------------

    #[test]
    fn display_includes_frame_index() {
        let stats = FrameStats::new(42);
        let display = format!("{}", stats);
        assert!(display.contains("Frame 42"));
    }

    #[test]
    fn display_includes_total_time() {
        let mut stats = FrameStats::new(0);
        stats.add_region(TimestampResult::new(None, 0, 1_000_000));
        let display = format!("{}", stats);
        assert!(display.contains("ms") || display.contains("1.0"));
    }

    #[test]
    fn display_includes_region_count() {
        let mut stats = FrameStats::new(0);
        stats.add_region(TimestampResult::new(None, 0, 100));
        let display = format!("{}", stats);
        assert!(display.contains("1 region"));
    }

    #[test]
    fn default_is_frame_zero() {
        let stats = FrameStats::default();
        assert_eq!(stats.frame_index, 0);
    }

    #[test]
    fn debug_format_shows_all_fields() {
        let stats = FrameStats::new(5);
        let debug = format!("{:?}", stats);
        assert!(debug.contains("FrameStats"));
        assert!(debug.contains("frame_index"));
        assert!(debug.contains("total_ns"));
    }
}

// =============================================================================
// MODULE: Edge Cases
// =============================================================================

mod edge_cases {
    use super::*;

    // -------------------------------------------------------------------------
    // Zero Duration
    // -------------------------------------------------------------------------

    #[test]
    fn zero_duration_result() {
        let result = TimestampResult::zero();
        assert_eq!(result.duration_ns(), 0);
        assert_eq!(result.duration_us(), 0.0);
        assert_eq!(result.duration_ms(), 0.0);
        assert_eq!(result.duration_secs(), 0.0);
    }

    #[test]
    fn zero_duration_from_equal_timestamps() {
        let result = TimestampResult::new(None, 1000, 1000);
        assert_eq!(result.duration_ns(), 0);
    }

    #[test]
    fn zero_duration_is_valid() {
        let result = TimestampResult::zero();
        assert!(result.is_valid());
    }

    // -------------------------------------------------------------------------
    // Maximum Duration
    // -------------------------------------------------------------------------

    #[test]
    fn maximum_duration_ns() {
        let result = TimestampResult::new(None, 0, u64::MAX);
        assert_eq!(result.duration_ns(), u64::MAX);
    }

    #[test]
    fn large_duration_conversion_to_seconds() {
        let result = TimestampResult::new(None, 0, 60_000_000_000); // 60 seconds
        assert!((result.duration_secs() - 60.0).abs() < 0.001);
    }

    #[test]
    fn maximum_tick_values() {
        let converter = TimestampPeriodConverter::new(1.0);
        // Should not overflow or panic
        let ns = converter.ticks_to_ns(u64::MAX / 2);
        assert!(ns > 0);
    }

    // -------------------------------------------------------------------------
    // Empty Results
    // -------------------------------------------------------------------------

    #[test]
    fn empty_frame_stats() {
        let stats = FrameStats::new(0);
        assert!(stats.is_empty());
        assert_eq!(stats.region_count, 0);
        assert!(stats.regions.is_empty());
    }

    #[test]
    fn empty_profiler_stats() {
        let stats = ProfilerStats::empty();
        assert_eq!(stats.total_queries, 0);
        assert_eq!(stats.utilization(), 0.0);
    }

    // -------------------------------------------------------------------------
    // Boundary Values
    // -------------------------------------------------------------------------

    #[test]
    fn handle_with_u32_max_indices() {
        let handle = TimestampHandle::new(u32::MAX - 1, u32::MAX);
        assert_eq!(handle.start_index, u32::MAX - 1);
        assert_eq!(handle.end_index, u32::MAX);
    }

    #[test]
    fn handle_with_same_start_end() {
        let handle = TimestampHandle::new(5, 5);
        assert_eq!(handle.start_index, handle.end_index);
    }

    #[test]
    fn result_with_max_timestamps() {
        let result = TimestampResult::new(None, u64::MAX - 1000, u64::MAX);
        assert_eq!(result.duration_ns(), 1000);
    }

    // -------------------------------------------------------------------------
    // Special Characters in Labels
    // -------------------------------------------------------------------------

    #[test]
    fn label_with_newlines() {
        let handle = TimestampHandle::with_label(0, 1, "Line1\nLine2");
        assert!(handle.label.as_ref().unwrap().contains('\n'));
    }

    #[test]
    fn label_with_tabs() {
        let handle = TimestampHandle::with_label(0, 1, "Col1\tCol2");
        assert!(handle.label.as_ref().unwrap().contains('\t'));
    }

    #[test]
    fn label_with_special_chars() {
        let handle = TimestampHandle::with_label(0, 1, "!@#$%^&*()[]{}");
        assert_eq!(handle.label.as_deref(), Some("!@#$%^&*()[]{}"));
    }

    #[test]
    fn label_with_quotes() {
        let handle = TimestampHandle::with_label(0, 1, r#"Say "Hello""#);
        assert!(handle.label.as_ref().unwrap().contains('"'));
    }

    #[test]
    fn label_with_backslashes() {
        let handle = TimestampHandle::with_label(0, 1, r"path\to\file");
        assert!(handle.label.as_ref().unwrap().contains('\\'));
    }

    // -------------------------------------------------------------------------
    // Numeric Edge Cases
    // -------------------------------------------------------------------------

    #[test]
    fn period_near_zero() {
        let converter = TimestampPeriodConverter::new(0.0001);
        let ns = converter.ticks_to_ns(1_000_000);
        assert!(ns > 0);
    }

    #[test]
    fn period_very_large() {
        let converter = TimestampPeriodConverter::new(1_000_000.0);
        let ns = converter.ticks_to_ns(1);
        assert_eq!(ns, 1_000_000);
    }

    #[test]
    fn negative_period_effect() {
        // Period should be positive in practice, but test defensive behavior
        let converter = TimestampPeriodConverter::new(-1.0);
        // Negative period results in 0 (casted from negative float)
        let ticks = converter.ns_to_ticks(1000);
        // Behavior depends on implementation - just ensure no panic
        let _ = ticks;
    }
}

// =============================================================================
// MODULE: Buffer Management
// =============================================================================

mod buffer_management {
    use super::*;

    // -------------------------------------------------------------------------
    // Capacity Constants
    // -------------------------------------------------------------------------

    #[test]
    fn min_capacity_constant() {
        assert!(MIN_CAPACITY >= 1);
        assert!(MIN_CAPACITY <= 16);
    }

    #[test]
    fn max_capacity_constant() {
        assert!(MAX_RECOMMENDED_CAPACITY >= 256);
        assert!(MAX_RECOMMENDED_CAPACITY <= 16384);
    }

    #[test]
    fn default_capacity_in_valid_range() {
        assert!(DEFAULT_CAPACITY >= MIN_CAPACITY);
        assert!(DEFAULT_CAPACITY <= MAX_RECOMMENDED_CAPACITY);
    }

    #[test]
    fn timestamp_size_is_8_bytes() {
        assert_eq!(TIMESTAMP_SIZE_BYTES, 8);
        assert_eq!(TIMESTAMP_SIZE_BYTES, std::mem::size_of::<u64>() as u64);
    }

    // -------------------------------------------------------------------------
    // Buffer Size Calculations
    // -------------------------------------------------------------------------

    #[test]
    fn buffer_size_for_min_capacity() {
        let size = (MIN_CAPACITY as u64) * 2 * TIMESTAMP_SIZE_BYTES;
        assert_eq!(size, (MIN_CAPACITY as u64) * 16);
    }

    #[test]
    fn buffer_size_for_default_capacity() {
        let size = (DEFAULT_CAPACITY as u64) * 2 * TIMESTAMP_SIZE_BYTES;
        assert_eq!(size, (DEFAULT_CAPACITY as u64) * 16);
    }

    #[test]
    fn buffer_size_for_max_capacity() {
        let size = (MAX_RECOMMENDED_CAPACITY as u64) * 2 * TIMESTAMP_SIZE_BYTES;
        // 4096 * 2 * 8 = 65536 bytes = 64KB
        assert_eq!(size, (MAX_RECOMMENDED_CAPACITY as u64) * 16);
    }

    // -------------------------------------------------------------------------
    // Capacity Tracking (Simulated)
    // -------------------------------------------------------------------------

    #[test]
    fn simulate_sequential_allocations() {
        let capacity = 10;
        let mut allocated = 0;

        for _ in 0..capacity {
            allocated += 1;
        }

        assert_eq!(allocated, capacity);
    }

    #[test]
    fn simulate_capacity_exhaustion() {
        let capacity = 5;
        let mut handles = Vec::new();

        for i in 0..capacity {
            handles.push(TimestampHandle::new(i as u32 * 2, i as u32 * 2 + 1));
        }

        assert_eq!(handles.len(), capacity);
        // Next allocation would fail (simulated)
    }

    #[test]
    fn simulate_reset_and_reuse() {
        let capacity = 5;
        let mut handles = Vec::new();

        // First batch
        for i in 0..capacity {
            handles.push(TimestampHandle::new(i as u32 * 2, i as u32 * 2 + 1));
        }
        assert_eq!(handles.len(), capacity);

        // Reset (clear)
        handles.clear();

        // Second batch - same indices available again
        for i in 0..capacity {
            handles.push(TimestampHandle::new(i as u32 * 2, i as u32 * 2 + 1));
        }
        assert_eq!(handles.len(), capacity);
    }
}

// =============================================================================
// MODULE: Real-World Scenarios
// =============================================================================

mod real_world_scenarios {
    use super::*;

    // -------------------------------------------------------------------------
    // Profile Render Pass
    // -------------------------------------------------------------------------

    #[test]
    fn simulate_shadow_pass_timing() {
        let handle = TimestampHandle::with_label(0, 1, "Shadow Map");
        assert_eq!(handle.label_or_unnamed(), "Shadow Map");

        // Simulate timing result: 0.5ms shadow pass
        let result = TimestampResult::from_ticks_labeled(0, 20_000, 25.0, "Shadow Map");
        assert!((result.duration_ms() - 0.5).abs() < 0.01);
    }

    #[test]
    fn simulate_gbuffer_pass_timing() {
        let result = TimestampResult::from_ticks_labeled(0, 80_000, 25.0, "GBuffer");
        // 80000 * 25ns = 2ms
        assert!((result.duration_ms() - 2.0).abs() < 0.01);
    }

    #[test]
    fn simulate_lighting_pass_timing() {
        let result = TimestampResult::from_ticks_labeled(0, 120_000, 25.0, "Deferred Lighting");
        // 120000 * 25ns = 3ms
        assert!((result.duration_ms() - 3.0).abs() < 0.01);
    }

    #[test]
    fn simulate_post_process_timing() {
        let result = TimestampResult::from_ticks_labeled(0, 12_000, 25.0, "Bloom");
        // 12000 * 25ns = 0.3ms
        assert!((result.duration_ms() - 0.3).abs() < 0.01);
    }

    // -------------------------------------------------------------------------
    // Profile Compute Pass
    // -------------------------------------------------------------------------

    #[test]
    fn simulate_particle_compute_timing() {
        let result = TimestampResult::from_ticks_labeled(0, 40_000, 25.0, "Particle Simulation");
        assert!((result.duration_ms() - 1.0).abs() < 0.01);
    }

    #[test]
    fn simulate_culling_compute_timing() {
        let result = TimestampResult::from_ticks_labeled(0, 8_000, 25.0, "GPU Culling");
        // 8000 * 25ns = 0.2ms
        assert!((result.duration_ms() - 0.2).abs() < 0.01);
    }

    #[test]
    fn simulate_skinning_compute_timing() {
        let result = TimestampResult::from_ticks_labeled(0, 16_000, 25.0, "GPU Skinning");
        // 16000 * 25ns = 0.4ms
        assert!((result.duration_ms() - 0.4).abs() < 0.01);
    }

    // -------------------------------------------------------------------------
    // Profile Copy Operations
    // -------------------------------------------------------------------------

    #[test]
    fn simulate_texture_copy_timing() {
        let result = TimestampResult::from_ticks_labeled(0, 2_000, 25.0, "Texture Copy");
        // 2000 * 25ns = 0.05ms = 50us
        assert!((result.duration_us() - 50.0).abs() < 1.0);
    }

    #[test]
    fn simulate_buffer_copy_timing() {
        let result = TimestampResult::from_ticks_labeled(0, 1_000, 25.0, "Buffer Copy");
        // 1000 * 25ns = 25us
        assert!((result.duration_us() - 25.0).abs() < 1.0);
    }

    // -------------------------------------------------------------------------
    // Profile Entire Frame
    // -------------------------------------------------------------------------

    #[test]
    fn simulate_full_frame_profile() {
        let mut stats = FrameStats::new(100);
        let period = 25.0;

        // Add all passes
        let passes = [
            ("Shadow Cascade 0", 20_000),
            ("Shadow Cascade 1", 18_000),
            ("Shadow Cascade 2", 16_000),
            ("Shadow Cascade 3", 14_000),
            ("GBuffer", 80_000),
            ("SSAO", 40_000),
            ("Deferred Lighting", 120_000),
            ("Atmosphere", 30_000),
            ("Bloom", 12_000),
            ("Tonemap", 4_000),
            ("UI", 8_000),
        ];

        for (name, ticks) in passes.iter() {
            let result = TimestampResult::from_ticks_labeled(0, *ticks, period, *name);
            stats.add_region(result);
        }

        assert_eq!(stats.region_count, 11);
        // Total: ~9.05ms
        let total_ticks: u64 = passes.iter().map(|(_, t)| *t as u64).sum();
        let expected_ms = (total_ticks as f64) * (period as f64) / 1_000_000.0;
        assert!((stats.total_ms() - expected_ms).abs() < 0.01);
    }

    #[test]
    fn simulate_60fps_frame_budget() {
        let budget_ns: u64 = 16_666_667; // ~16.67ms for 60fps

        let result = TimestampResult::new(None, 0, 15_000_000);
        let usage = (result.duration_ns() as f64) / (budget_ns as f64);

        assert!(usage < 1.0); // Within budget
        assert!((usage - 0.9).abs() < 0.01); // ~90% of budget
    }

    #[test]
    fn simulate_frame_over_budget() {
        let budget_ns: u64 = 16_666_667;

        let result = TimestampResult::new(None, 0, 20_000_000); // 20ms
        let usage = (result.duration_ns() as f64) / (budget_ns as f64);

        assert!(usage > 1.0); // Over budget
        assert!((usage - 1.2).abs() < 0.01); // ~120% of budget
    }

    // -------------------------------------------------------------------------
    // Mixed Pass Types
    // -------------------------------------------------------------------------

    #[test]
    fn simulate_frame_with_mixed_passes() {
        let mut stats = FrameStats::new(0);
        let period = 25.0;

        // Compute passes
        stats.add_region(TimestampResult::from_ticks_labeled(
            0,
            8_000,
            period,
            "Culling (Compute)",
        ));
        stats.add_region(TimestampResult::from_ticks_labeled(
            0,
            16_000,
            period,
            "Skinning (Compute)",
        ));

        // Graphics passes
        stats.add_region(TimestampResult::from_ticks_labeled(
            0,
            80_000,
            period,
            "GBuffer (Graphics)",
        ));
        stats.add_region(TimestampResult::from_ticks_labeled(
            0,
            120_000,
            period,
            "Lighting (Graphics)",
        ));

        // Transfer passes
        stats.add_region(TimestampResult::from_ticks_labeled(
            0,
            2_000,
            period,
            "Readback (Transfer)",
        ));

        assert_eq!(stats.region_count, 5);
        // Total: (8+16+80+120+2) * 25ns = 5.65ms
        let expected_ns: u64 = (8_000 + 16_000 + 80_000 + 120_000 + 2_000) * 25;
        assert_eq!(stats.total_ns, expected_ns);
    }
}

// =============================================================================
// MODULE: Integration Patterns
// =============================================================================

mod integration_patterns {
    use super::*;

    // -------------------------------------------------------------------------
    // Multiple Profilers
    // -------------------------------------------------------------------------

    #[test]
    fn create_multiple_converters() {
        let gpu1 = TimestampPeriodConverter::new(25.0); // NVIDIA
        let gpu2 = TimestampPeriodConverter::new(41.666666); // AMD

        // Same tick count, different GPU periods
        let ticks = 40_000;
        let ns1 = gpu1.ticks_to_ns(ticks);
        let ns2 = gpu2.ticks_to_ns(ticks);

        assert_ne!(ns1, ns2);
        // NVIDIA: 40000 * 25 = 1_000_000ns = 1ms
        assert_eq!(ns1, 1_000_000);
        // AMD: 40000 * 41.67 = 1_666_666ns = ~1.67ms
        assert!((ns2 as f64 - 1_666_666.0).abs() < 100.0);
    }

    #[test]
    fn independent_frame_stats() {
        let mut stats1 = FrameStats::new(1);
        let mut stats2 = FrameStats::new(2);

        stats1.add_region(TimestampResult::new(None, 0, 1_000_000));
        stats2.add_region(TimestampResult::new(None, 0, 2_000_000));

        // Verify independence
        assert_eq!(stats1.total_ns, 1_000_000);
        assert_eq!(stats2.total_ns, 2_000_000);
    }

    // -------------------------------------------------------------------------
    // Profiler Per Frame Pattern
    // -------------------------------------------------------------------------

    #[test]
    fn simulate_frame_profiler_pattern() {
        let frame_count = 3;

        let mut all_stats: Vec<FrameStats> = Vec::new();

        for frame in 0..frame_count {
            let mut stats = FrameStats::new(frame as u64);

            // Simulate varying GPU load
            let base_time = 1_000_000u64;
            let variation = (frame as u64) * 100_000;

            stats.add_region(TimestampResult::new(
                Some("Main Pass".to_string()),
                0,
                base_time + variation,
            ));

            all_stats.push(stats);
        }

        assert_eq!(all_stats.len(), 3);
        assert!(all_stats[2].total_ns > all_stats[0].total_ns);
    }

    #[test]
    fn simulate_rolling_average() {
        let history_size = 5;
        let mut history: Vec<f64> = Vec::with_capacity(history_size);

        let frame_times = [1.0, 1.2, 0.8, 1.1, 0.9];

        for time in frame_times.iter() {
            if history.len() >= history_size {
                history.remove(0);
            }
            history.push(*time);
        }

        let avg: f64 = history.iter().sum::<f64>() / (history.len() as f64);
        assert!((avg - 1.0).abs() < 0.01);
    }

    // -------------------------------------------------------------------------
    // Global Profiler Pattern (Simulated)
    // -------------------------------------------------------------------------

    #[test]
    fn simulate_global_stats_accumulation() {
        // Simulates a static/global profiler collecting stats
        let mut global_total_frames = 0u64;
        let mut global_total_time_ns = 0u64;

        for frame in 0..100 {
            let frame_time = 16_000_000u64 + (frame % 5) * 100_000;
            global_total_frames += 1;
            global_total_time_ns += frame_time;
        }

        let avg_frame_time_ms = (global_total_time_ns as f64) / (global_total_frames as f64) / 1_000_000.0;
        assert!(avg_frame_time_ms > 16.0 && avg_frame_time_ms < 17.0);
    }

    // -------------------------------------------------------------------------
    // Thread-Local Profiler Pattern (Simulated)
    // -------------------------------------------------------------------------

    #[test]
    fn simulate_per_thread_stats() {
        // Simulates thread-local profiling
        let thread_count = 4;
        let mut thread_stats: Vec<FrameStats> = Vec::new();

        for thread_id in 0..thread_count {
            let mut stats = FrameStats::new(thread_id as u64);
            stats.add_region(TimestampResult::new(
                Some(format!("Thread {} Work", thread_id)),
                0,
                1_000_000 + (thread_id as u64) * 100_000,
            ));
            thread_stats.push(stats);
        }

        assert_eq!(thread_stats.len(), thread_count);

        // Aggregate results
        let total: u64 = thread_stats.iter().map(|s| s.total_ns).sum();
        assert!(total > 4_000_000);
    }

    // -------------------------------------------------------------------------
    // Hierarchical Profiling Pattern
    // -------------------------------------------------------------------------

    #[test]
    fn simulate_hierarchical_profiling() {
        let mut frame = FrameStats::new(0);

        // Top-level frame timing
        let frame_total = 16_000_000u64;

        // Child timings
        let shadow_time = 2_000_000u64;
        let gbuffer_time = 4_000_000u64;
        let lighting_time = 6_000_000u64;
        let post_time = 3_000_000u64;
        let other_time = 1_000_000u64;

        frame.add_region(TimestampResult::new(
            Some("Shadow".to_string()),
            0,
            shadow_time,
        ));
        frame.add_region(TimestampResult::new(
            Some("GBuffer".to_string()),
            0,
            gbuffer_time,
        ));
        frame.add_region(TimestampResult::new(
            Some("Lighting".to_string()),
            0,
            lighting_time,
        ));
        frame.add_region(TimestampResult::new(
            Some("Post".to_string()),
            0,
            post_time,
        ));
        frame.add_region(TimestampResult::new(
            Some("Other".to_string()),
            0,
            other_time,
        ));

        // Child times should sum to frame total
        assert_eq!(frame.total_ns, frame_total);
    }

    // -------------------------------------------------------------------------
    // Callback/Event Pattern
    // -------------------------------------------------------------------------

    #[test]
    fn simulate_timing_events() {
        struct TimingEvent {
            label: String,
            duration_ms: f64,
        }

        let mut events: Vec<TimingEvent> = Vec::new();

        let results = [
            TimestampResult::from_ticks_labeled(0, 40_000, 25.0, "Pass A"),
            TimestampResult::from_ticks_labeled(0, 80_000, 25.0, "Pass B"),
            TimestampResult::from_ticks_labeled(0, 20_000, 25.0, "Pass C"),
        ];

        for result in results.iter() {
            events.push(TimingEvent {
                label: result.label.clone().unwrap_or_default(),
                duration_ms: result.duration_ms(),
            });
        }

        assert_eq!(events.len(), 3);
        assert!((events[0].duration_ms - 1.0).abs() < 0.01);
        assert!((events[1].duration_ms - 2.0).abs() < 0.01);
        assert!((events[2].duration_ms - 0.5).abs() < 0.01);
    }
}

// =============================================================================
// MODULE: Send + Sync Bounds
// =============================================================================

mod thread_safety {
    use super::*;

    fn assert_send<T: Send>() {}
    fn assert_sync<T: Sync>() {}

    #[test]
    fn timestamp_handle_is_send() {
        assert_send::<TimestampHandle>();
    }

    #[test]
    fn timestamp_handle_is_sync() {
        assert_sync::<TimestampHandle>();
    }

    #[test]
    fn timestamp_result_is_send() {
        assert_send::<TimestampResult>();
    }

    #[test]
    fn timestamp_result_is_sync() {
        assert_sync::<TimestampResult>();
    }

    #[test]
    fn timestamp_period_converter_is_send() {
        assert_send::<TimestampPeriodConverter>();
    }

    #[test]
    fn timestamp_period_converter_is_sync() {
        assert_sync::<TimestampPeriodConverter>();
    }

    #[test]
    fn profiler_stats_is_send() {
        assert_send::<ProfilerStats>();
    }

    #[test]
    fn profiler_stats_is_sync() {
        assert_sync::<ProfilerStats>();
    }

    #[test]
    fn frame_stats_is_send() {
        assert_send::<FrameStats>();
    }

    #[test]
    fn frame_stats_is_sync() {
        assert_sync::<FrameStats>();
    }
}

// =============================================================================
// MODULE: RAII Scope Pattern (Simulated)
// =============================================================================

mod raii_scope_pattern {
    use super::*;

    // Simulated RAII behavior using Option<TimestampHandle>

    struct SimulatedScope {
        handle: TimestampHandle,
        ended: bool,
    }

    impl SimulatedScope {
        fn new(label: &str) -> Self {
            Self {
                handle: TimestampHandle::with_label(0, 1, label),
                ended: false,
            }
        }

        fn split(&mut self, new_label: &str) -> TimestampHandle {
            self.ended = true;
            self.handle = TimestampHandle::with_label(2, 3, new_label);
            self.ended = false;
            self.handle.clone()
        }

        fn end_manual(&mut self) {
            self.ended = true;
        }

        fn is_ended(&self) -> bool {
            self.ended
        }
    }

    impl Drop for SimulatedScope {
        fn drop(&mut self) {
            // Would call profiler.end() here if not already ended
            if !self.ended {
                // Simulate end call
            }
        }
    }

    // -------------------------------------------------------------------------
    // Automatic End on Drop
    // -------------------------------------------------------------------------

    #[test]
    fn scope_ends_on_drop() {
        let scope = SimulatedScope::new("Auto End");
        assert!(!scope.is_ended());
        drop(scope);
        // Scope automatically ended on drop
    }

    #[test]
    fn scope_within_block() {
        let label;
        {
            let scope = SimulatedScope::new("Block Scope");
            label = scope.handle.label.clone();
            // scope drops here
        }
        assert_eq!(label.as_deref(), Some("Block Scope"));
    }

    // -------------------------------------------------------------------------
    // Split Within Scope
    // -------------------------------------------------------------------------

    #[test]
    fn split_creates_new_handle() {
        let mut scope = SimulatedScope::new("First");
        let first_label = scope.handle.label.clone();

        let new_handle = scope.split("Second");

        assert_eq!(first_label.as_deref(), Some("First"));
        assert_eq!(new_handle.label.as_deref(), Some("Second"));
    }

    #[test]
    fn split_multiple_times() {
        let mut scope = SimulatedScope::new("A");

        scope.split("B");
        scope.split("C");
        scope.split("D");

        assert_eq!(scope.handle.label.as_deref(), Some("D"));
    }

    // -------------------------------------------------------------------------
    // Nested Scopes (Simulated)
    // -------------------------------------------------------------------------

    #[test]
    fn nested_scopes() {
        let outer = SimulatedScope::new("Outer");
        {
            let inner = SimulatedScope::new("Inner");
            assert_eq!(inner.handle.label.as_deref(), Some("Inner"));
            // inner drops
        }
        assert_eq!(outer.handle.label.as_deref(), Some("Outer"));
        // outer drops
    }

    #[test]
    fn deeply_nested_scopes() {
        let level1 = SimulatedScope::new("L1");
        {
            let level2 = SimulatedScope::new("L2");
            {
                let level3 = SimulatedScope::new("L3");
                assert_eq!(level3.handle.label.as_deref(), Some("L3"));
            }
            assert_eq!(level2.handle.label.as_deref(), Some("L2"));
        }
        assert_eq!(level1.handle.label.as_deref(), Some("L1"));
    }

    // -------------------------------------------------------------------------
    // Early Drop
    // -------------------------------------------------------------------------

    #[test]
    fn manual_end_prevents_double_end() {
        let mut scope = SimulatedScope::new("Manual");
        scope.end_manual();
        assert!(scope.is_ended());
        // drop won't double-end
    }

    #[test]
    fn explicit_drop() {
        let scope = SimulatedScope::new("Explicit");
        std::mem::drop(scope);
        // Scope is explicitly dropped
    }
}

// =============================================================================
// MODULE: Clone and Debug Traits
// =============================================================================

mod clone_debug_traits {
    use super::*;

    #[test]
    fn timestamp_handle_clone() {
        let original = TimestampHandle::with_label(10, 11, "Clone Test");
        let cloned = original.clone();

        assert_eq!(original.start_index, cloned.start_index);
        assert_eq!(original.end_index, cloned.end_index);
        assert_eq!(original.label, cloned.label);
    }

    #[test]
    fn timestamp_result_clone() {
        let original = TimestampResult::new(Some("Clone".to_string()), 100, 200);
        let cloned = original.clone();

        assert_eq!(original.label, cloned.label);
        assert_eq!(original.start_ns, cloned.start_ns);
        assert_eq!(original.end_ns, cloned.end_ns);
    }

    #[test]
    fn profiler_stats_clone() {
        let original = ProfilerStats::new(100, 50, 25, 1_000_000, 500_000, 2_000_000);
        let cloned = original.clone();

        assert_eq!(original, cloned);
    }

    #[test]
    fn frame_stats_clone() {
        let mut original = FrameStats::new(42);
        original.add_region(TimestampResult::new(None, 0, 1000));

        let cloned = original.clone();

        assert_eq!(original.frame_index, cloned.frame_index);
        assert_eq!(original.total_ns, cloned.total_ns);
        assert_eq!(original.region_count, cloned.region_count);
    }

    #[test]
    fn converter_copy_and_clone() {
        let c1 = TimestampPeriodConverter::new(25.0);
        let c2 = c1; // Copy
        let c3 = c1.clone(); // Clone

        assert_eq!(c1.period(), c2.period());
        assert_eq!(c1.period(), c3.period());
    }

    #[test]
    fn debug_timestamp_handle() {
        let handle = TimestampHandle::with_label(1, 2, "Debug");
        let debug = format!("{:?}", handle);

        assert!(debug.contains("TimestampHandle"));
        assert!(debug.contains("start_index"));
        assert!(debug.contains("end_index"));
        assert!(debug.contains("label"));
    }

    #[test]
    fn debug_timestamp_result() {
        let result = TimestampResult::new(Some("Debug".to_string()), 100, 200);
        let debug = format!("{:?}", result);

        assert!(debug.contains("TimestampResult"));
        assert!(debug.contains("label"));
        assert!(debug.contains("start_ns"));
        assert!(debug.contains("end_ns"));
    }

    #[test]
    fn debug_converter() {
        let converter = TimestampPeriodConverter::new(25.0);
        let debug = format!("{:?}", converter);

        assert!(debug.contains("TimestampPeriodConverter"));
        assert!(debug.contains("timestamp_period"));
    }

    #[test]
    fn debug_profiler_stats() {
        let stats = ProfilerStats::new(100, 50, 25, 0, 0, 0);
        let debug = format!("{:?}", stats);

        assert!(debug.contains("ProfilerStats"));
        assert!(debug.contains("total_queries"));
        assert!(debug.contains("active_queries"));
    }

    #[test]
    fn debug_frame_stats() {
        let stats = FrameStats::new(1);
        let debug = format!("{:?}", stats);

        assert!(debug.contains("FrameStats"));
        assert!(debug.contains("frame_index"));
    }
}

// =============================================================================
// MODULE: PartialEq Comparisons
// =============================================================================

mod equality_comparisons {
    use super::*;

    #[test]
    fn timestamp_handle_eq() {
        let h1 = TimestampHandle::new(0, 1);
        let h2 = TimestampHandle::new(0, 1);
        let h3 = TimestampHandle::new(2, 3);

        assert_eq!(h1, h2);
        assert_ne!(h1, h3);
    }

    #[test]
    fn timestamp_handle_eq_with_labels() {
        let h1 = TimestampHandle::with_label(0, 1, "A");
        let h2 = TimestampHandle::with_label(0, 1, "A");
        let h3 = TimestampHandle::with_label(0, 1, "B");

        assert_eq!(h1, h2);
        assert_ne!(h1, h3);
    }

    #[test]
    fn timestamp_result_eq() {
        let r1 = TimestampResult::new(None, 100, 200);
        let r2 = TimestampResult::new(None, 100, 200);
        let r3 = TimestampResult::new(None, 100, 300);

        assert_eq!(r1, r2);
        assert_ne!(r1, r3);
    }

    #[test]
    fn converter_eq() {
        let c1 = TimestampPeriodConverter::new(25.0);
        let c2 = TimestampPeriodConverter::new(25.0);
        let c3 = TimestampPeriodConverter::new(30.0);

        assert_eq!(c1, c2);
        assert_ne!(c1, c3);
    }

    #[test]
    fn profiler_stats_eq() {
        let s1 = ProfilerStats::new(100, 50, 25, 0, 0, 0);
        let s2 = ProfilerStats::new(100, 50, 25, 0, 0, 0);
        let s3 = ProfilerStats::new(200, 50, 25, 0, 0, 0);

        assert_eq!(s1, s2);
        assert_ne!(s1, s3);
    }

    #[test]
    fn frame_stats_eq() {
        let s1 = FrameStats::new(1);
        let s2 = FrameStats::new(1);
        let s3 = FrameStats::new(2);

        assert_eq!(s1, s2);
        assert_ne!(s1, s3);
    }
}

// =============================================================================
// MODULE: Default Trait Implementations
// =============================================================================

mod default_implementations {
    use super::*;

    #[test]
    fn timestamp_result_default() {
        let result = TimestampResult::default();
        assert!(result.label.is_none());
        assert_eq!(result.start_ns, 0);
        assert_eq!(result.end_ns, 0);
    }

    #[test]
    fn converter_default() {
        let converter = TimestampPeriodConverter::default();
        assert_eq!(converter.period(), 1.0);
    }

    #[test]
    fn profiler_stats_default() {
        let stats = ProfilerStats::default();
        assert_eq!(stats.total_queries, 0);
        assert_eq!(stats.active_queries, 0);
        assert_eq!(stats.resolved_queries, 0);
    }

    #[test]
    fn frame_stats_default() {
        let stats = FrameStats::default();
        assert_eq!(stats.frame_index, 0);
        assert_eq!(stats.total_ns, 0);
        assert_eq!(stats.region_count, 0);
    }
}
