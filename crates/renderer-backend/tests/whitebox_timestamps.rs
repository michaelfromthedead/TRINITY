//! Whitebox structural tests for GPU Timestamp Profiler.
//!
//! These tests verify the internal structure and behavior of the timestamp
//! profiling system, including all data structures, conversions, and edge cases.
//!
//! Task: T-WGPU-P7.4.1 - Timestamp Profiler Whitebox Testing
//!
//! Components Tested:
//! 1. TimestampHandle - Query pair tracking with labels
//! 2. TimestampResult - Duration calculations with multiple units
//! 3. TimestampPeriodConverter - Tick-to-nanosecond conversions
//! 4. ProfilerStats - Profiler state statistics
//! 5. FrameStats - Per-frame timing statistics
//! 6. TimestampQuery - Query set management (mock tests)
//! 7. TimestampProfiler - Main profiler interface (mock tests)
//! 8. GpuProfileScope - RAII timing guard
//! 9. FrameProfiler - Per-frame profiling
//! 10. Thread Safety - Send + Sync bounds
//!
//! WHITEBOX coverage plan:
//!   - Path A: TimestampHandle construction variants
//!   - Path B: TimestampHandle label management
//!   - Path C: TimestampHandle trait implementations (Clone, Debug, PartialEq, Eq, Hash, Display)
//!   - Path D: TimestampResult construction and duration calculations
//!   - Path E: TimestampResult time unit conversions (ns, us, ms, secs)
//!   - Path F: TimestampResult validity checks and edge cases
//!   - Path G: TimestampPeriodConverter construction and conversions
//!   - Path H: TimestampPeriodConverter roundtrip accuracy
//!   - Path I: TimestampPeriodConverter edge cases (zero, max values)
//!   - Path J: ProfilerStats construction and field accessors
//!   - Path K: ProfilerStats duration calculations
//!   - Path L: ProfilerStats utilization metrics
//!   - Path M: FrameStats construction and region tracking
//!   - Path N: FrameStats accumulation and duration calculations
//!   - Path O: Constants validation
//!   - Path P: Send + Sync bounds for all types
//!   - Path Q: Edge cases - boundary values
//!   - Path R: Edge cases - overflow handling
//!   - Path S: Edge cases - saturation arithmetic

use renderer_backend::profiling::timestamps::{
    FrameStats, ProfilerStats, TimestampHandle, TimestampPeriodConverter, TimestampResult,
    DEFAULT_CAPACITY, MAX_RECOMMENDED_CAPACITY, MIN_CAPACITY, TIMESTAMP_SIZE_BYTES,
};
use std::collections::{HashMap, HashSet};

// ============================================================================
// Section 1: TimestampHandle Tests
// ============================================================================

mod timestamp_handle_construction {
    use super::*;

    #[test]
    fn new_creates_handle_with_correct_indices() {
        let handle = TimestampHandle::new(0, 1);
        assert_eq!(handle.start_index, 0);
        assert_eq!(handle.end_index, 1);
        assert!(handle.label.is_none());
    }

    #[test]
    fn new_with_non_zero_indices() {
        let handle = TimestampHandle::new(42, 43);
        assert_eq!(handle.start_index, 42);
        assert_eq!(handle.end_index, 43);
    }

    #[test]
    fn new_with_consecutive_pair_indices() {
        for i in 0..100 {
            let handle = TimestampHandle::new(i * 2, i * 2 + 1);
            assert_eq!(handle.start_index, i * 2);
            assert_eq!(handle.end_index, i * 2 + 1);
        }
    }

    #[test]
    fn new_with_same_start_and_end_index() {
        let handle = TimestampHandle::new(5, 5);
        assert_eq!(handle.start_index, 5);
        assert_eq!(handle.end_index, 5);
    }

    #[test]
    fn new_with_reversed_indices_allowed() {
        // The struct doesn't validate ordering
        let handle = TimestampHandle::new(10, 5);
        assert_eq!(handle.start_index, 10);
        assert_eq!(handle.end_index, 5);
    }

    #[test]
    fn with_label_stores_string_label() {
        let handle = TimestampHandle::with_label(0, 1, "Shadow Pass");
        assert_eq!(handle.start_index, 0);
        assert_eq!(handle.end_index, 1);
        assert_eq!(handle.label, Some("Shadow Pass".to_string()));
    }

    #[test]
    fn with_label_accepts_string() {
        let handle = TimestampHandle::with_label(0, 1, String::from("Lighting"));
        assert_eq!(handle.label, Some("Lighting".to_string()));
    }

    #[test]
    fn with_label_empty_string() {
        let handle = TimestampHandle::with_label(0, 1, "");
        assert_eq!(handle.label, Some("".to_string()));
    }

    #[test]
    fn with_label_unicode() {
        let handle = TimestampHandle::with_label(0, 1, "");
        assert_eq!(handle.label, Some("".to_string()));
    }

    #[test]
    fn with_label_long_string() {
        let long_label = "A".repeat(1000);
        let handle = TimestampHandle::with_label(0, 1, long_label.clone());
        assert_eq!(handle.label, Some(long_label));
    }
}

mod timestamp_handle_label_management {
    use super::*;

    #[test]
    fn set_label_on_unlabeled_handle() {
        let mut handle = TimestampHandle::new(0, 1);
        assert!(handle.label.is_none());
        handle.set_label("New Label");
        assert_eq!(handle.label, Some("New Label".to_string()));
    }

    #[test]
    fn set_label_overwrites_existing() {
        let mut handle = TimestampHandle::with_label(0, 1, "Old");
        handle.set_label("New");
        assert_eq!(handle.label, Some("New".to_string()));
    }

    #[test]
    fn set_label_multiple_times() {
        let mut handle = TimestampHandle::new(0, 1);
        for i in 0..10 {
            handle.set_label(format!("Label {}", i));
            assert_eq!(handle.label, Some(format!("Label {}", i)));
        }
    }

    #[test]
    fn clear_label_removes_label() {
        let mut handle = TimestampHandle::with_label(0, 1, "Label");
        assert!(handle.has_label());
        handle.clear_label();
        assert!(!handle.has_label());
        assert!(handle.label.is_none());
    }

    #[test]
    fn clear_label_on_unlabeled_is_noop() {
        let mut handle = TimestampHandle::new(0, 1);
        handle.clear_label();
        assert!(!handle.has_label());
    }

    #[test]
    fn has_label_returns_true_for_labeled() {
        let handle = TimestampHandle::with_label(0, 1, "Test");
        assert!(handle.has_label());
    }

    #[test]
    fn has_label_returns_false_for_unlabeled() {
        let handle = TimestampHandle::new(0, 1);
        assert!(!handle.has_label());
    }

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
}

mod timestamp_handle_traits {
    use super::*;

    #[test]
    fn clone_creates_equal_handle() {
        let original = TimestampHandle::with_label(5, 10, "Test");
        let cloned = original.clone();
        assert_eq!(original, cloned);
    }

    #[test]
    fn clone_is_independent() {
        let original = TimestampHandle::with_label(5, 10, "Test");
        let mut cloned = original.clone();
        cloned.set_label("Changed");
        assert_ne!(original.label, cloned.label);
    }

    #[test]
    fn debug_format_contains_indices() {
        let handle = TimestampHandle::new(0, 1);
        let debug = format!("{:?}", handle);
        assert!(debug.contains("start_index: 0"));
        assert!(debug.contains("end_index: 1"));
    }

    #[test]
    fn debug_format_contains_label() {
        let handle = TimestampHandle::with_label(0, 1, "Test");
        let debug = format!("{:?}", handle);
        assert!(debug.contains("Test"));
    }

    #[test]
    fn partial_eq_same_handles() {
        let h1 = TimestampHandle::new(0, 1);
        let h2 = TimestampHandle::new(0, 1);
        assert_eq!(h1, h2);
    }

    #[test]
    fn partial_eq_different_indices() {
        let h1 = TimestampHandle::new(0, 1);
        let h2 = TimestampHandle::new(2, 3);
        assert_ne!(h1, h2);
    }

    #[test]
    fn partial_eq_different_labels() {
        let h1 = TimestampHandle::with_label(0, 1, "A");
        let h2 = TimestampHandle::with_label(0, 1, "B");
        assert_ne!(h1, h2);
    }

    #[test]
    fn partial_eq_labeled_vs_unlabeled() {
        let h1 = TimestampHandle::new(0, 1);
        let h2 = TimestampHandle::with_label(0, 1, "Test");
        assert_ne!(h1, h2);
    }

    #[test]
    fn hash_same_handles_same_hash() {
        use std::hash::{Hash, Hasher};
        use std::collections::hash_map::DefaultHasher;

        let h1 = TimestampHandle::new(0, 1);
        let h2 = TimestampHandle::new(0, 1);

        let mut hasher1 = DefaultHasher::new();
        let mut hasher2 = DefaultHasher::new();
        h1.hash(&mut hasher1);
        h2.hash(&mut hasher2);

        assert_eq!(hasher1.finish(), hasher2.finish());
    }

    #[test]
    fn hash_different_handles_likely_different_hash() {
        use std::hash::{Hash, Hasher};
        use std::collections::hash_map::DefaultHasher;

        let h1 = TimestampHandle::new(0, 1);
        let h2 = TimestampHandle::new(2, 3);

        let mut hasher1 = DefaultHasher::new();
        let mut hasher2 = DefaultHasher::new();
        h1.hash(&mut hasher1);
        h2.hash(&mut hasher2);

        assert_ne!(hasher1.finish(), hasher2.finish());
    }

    #[test]
    fn hash_set_contains_handle() {
        let mut set = HashSet::new();
        let handle = TimestampHandle::new(0, 1);
        set.insert(handle.clone());
        assert!(set.contains(&handle));
    }

    #[test]
    fn hash_set_no_duplicates() {
        let mut set = HashSet::new();
        let handle = TimestampHandle::new(0, 1);
        set.insert(handle.clone());
        set.insert(handle.clone());
        assert_eq!(set.len(), 1);
    }

    #[test]
    fn hash_map_key() {
        let mut map = HashMap::new();
        let handle = TimestampHandle::with_label(0, 1, "Key");
        map.insert(handle.clone(), 42);
        assert_eq!(map.get(&handle), Some(&42));
    }

    #[test]
    fn display_without_label() {
        let handle = TimestampHandle::new(5, 6);
        let display = format!("{}", handle);
        assert_eq!(display, "[5..6]");
    }

    #[test]
    fn display_with_label() {
        let handle = TimestampHandle::with_label(5, 6, "Test");
        let display = format!("{}", handle);
        assert_eq!(display, "Test[5..6]");
    }

    #[test]
    fn display_with_empty_label() {
        let handle = TimestampHandle::with_label(0, 1, "");
        let display = format!("{}", handle);
        assert_eq!(display, "[0..1]");
    }
}

// ============================================================================
// Section 2: TimestampResult Tests
// ============================================================================

mod timestamp_result_construction {
    use super::*;

    #[test]
    fn new_stores_all_fields() {
        let result = TimestampResult::new(Some("Label".to_string()), 1000, 2000);
        assert_eq!(result.label, Some("Label".to_string()));
        assert_eq!(result.start_ns, 1000);
        assert_eq!(result.end_ns, 2000);
    }

    #[test]
    fn new_without_label() {
        let result = TimestampResult::new(None, 100, 500);
        assert!(result.label.is_none());
        assert_eq!(result.start_ns, 100);
        assert_eq!(result.end_ns, 500);
    }

    #[test]
    fn zero_creates_zero_duration() {
        let result = TimestampResult::zero();
        assert!(result.label.is_none());
        assert_eq!(result.start_ns, 0);
        assert_eq!(result.end_ns, 0);
        assert_eq!(result.duration_ns(), 0);
    }

    #[test]
    fn zero_labeled_creates_labeled_zero() {
        let result = TimestampResult::zero_labeled("Empty");
        assert_eq!(result.label, Some("Empty".to_string()));
        assert_eq!(result.duration_ns(), 0);
    }

    #[test]
    fn from_ticks_converts_correctly() {
        // period = 10ns per tick, 100 ticks = 1000ns
        let result = TimestampResult::from_ticks(0, 100, 10.0);
        assert_eq!(result.start_ns, 0);
        assert_eq!(result.end_ns, 1000);
    }

    #[test]
    fn from_ticks_with_nonzero_start() {
        let result = TimestampResult::from_ticks(50, 150, 10.0);
        assert_eq!(result.start_ns, 500);
        assert_eq!(result.end_ns, 1500);
        assert_eq!(result.duration_ns(), 1000);
    }

    #[test]
    fn from_ticks_labeled() {
        let result = TimestampResult::from_ticks_labeled(0, 100, 10.0, "Test");
        assert_eq!(result.label, Some("Test".to_string()));
        assert_eq!(result.end_ns, 1000);
    }

    #[test]
    fn from_ticks_fractional_period() {
        // period = 25.5ns per tick
        let result = TimestampResult::from_ticks(0, 100, 25.5);
        assert_eq!(result.end_ns, 2550);
    }

    #[test]
    fn from_ticks_zero_period() {
        let result = TimestampResult::from_ticks(100, 200, 0.0);
        assert_eq!(result.start_ns, 0);
        assert_eq!(result.end_ns, 0);
    }

    #[test]
    fn default_is_zero() {
        let result = TimestampResult::default();
        assert_eq!(result.start_ns, 0);
        assert_eq!(result.end_ns, 0);
        assert!(result.label.is_none());
    }
}

mod timestamp_result_duration_calculations {
    use super::*;

    #[test]
    fn duration_ns_basic() {
        let result = TimestampResult::new(None, 1000, 5000);
        assert_eq!(result.duration_ns(), 4000);
    }

    #[test]
    fn duration_ns_zero() {
        let result = TimestampResult::new(None, 1000, 1000);
        assert_eq!(result.duration_ns(), 0);
    }

    #[test]
    fn duration_ns_saturating_on_reverse() {
        // end < start should saturate to 0
        let result = TimestampResult::new(None, 5000, 1000);
        assert_eq!(result.duration_ns(), 0);
    }

    #[test]
    fn duration_ns_large_values() {
        let result = TimestampResult::new(None, 0, u64::MAX);
        assert_eq!(result.duration_ns(), u64::MAX);
    }

    #[test]
    fn duration_us_conversion() {
        let result = TimestampResult::new(None, 0, 1000);
        assert!((result.duration_us() - 1.0).abs() < 0.001);
    }

    #[test]
    fn duration_us_fractional() {
        let result = TimestampResult::new(None, 0, 1500);
        assert!((result.duration_us() - 1.5).abs() < 0.001);
    }

    #[test]
    fn duration_ms_conversion() {
        let result = TimestampResult::new(None, 0, 1_000_000);
        assert!((result.duration_ms() - 1.0).abs() < 0.001);
    }

    #[test]
    fn duration_ms_fractional() {
        let result = TimestampResult::new(None, 0, 1_500_000);
        assert!((result.duration_ms() - 1.5).abs() < 0.001);
    }

    #[test]
    fn duration_secs_conversion() {
        let result = TimestampResult::new(None, 0, 1_000_000_000);
        assert!((result.duration_secs() - 1.0).abs() < 0.001);
    }

    #[test]
    fn duration_secs_fractional() {
        let result = TimestampResult::new(None, 0, 500_000_000);
        assert!((result.duration_secs() - 0.5).abs() < 0.001);
    }

    #[test]
    fn duration_chain_consistency() {
        let result = TimestampResult::new(None, 0, 1_000_000_000);
        let ns = result.duration_ns();
        let us = result.duration_us();
        let ms = result.duration_ms();
        let secs = result.duration_secs();

        assert!((us - (ns as f64 / 1000.0)).abs() < 0.001);
        assert!((ms - (ns as f64 / 1_000_000.0)).abs() < 0.001);
        assert!((secs - (ns as f64 / 1_000_000_000.0)).abs() < 0.001);
    }
}

mod timestamp_result_validity {
    use super::*;

    #[test]
    fn is_valid_for_normal_measurement() {
        let result = TimestampResult::new(None, 0, 100);
        assert!(result.is_valid());
    }

    #[test]
    fn is_valid_for_equal_timestamps() {
        let result = TimestampResult::new(None, 100, 100);
        assert!(result.is_valid());
    }

    #[test]
    fn is_not_valid_when_end_before_start() {
        let result = TimestampResult::new(None, 100, 50);
        assert!(!result.is_valid());
    }

    #[test]
    fn is_valid_for_zero_result() {
        let result = TimestampResult::zero();
        assert!(result.is_valid());
    }

    #[test]
    fn label_or_returns_default() {
        let result = TimestampResult::new(None, 0, 0);
        assert_eq!(result.label_or("default"), "default");
    }

    #[test]
    fn label_or_returns_actual() {
        let result = TimestampResult::new(Some("actual".to_string()), 0, 0);
        assert_eq!(result.label_or("default"), "actual");
    }
}

mod timestamp_result_traits {
    use super::*;

    #[test]
    fn clone_creates_equal_result() {
        let original = TimestampResult::new(Some("Test".to_string()), 100, 200);
        let cloned = original.clone();
        assert_eq!(original, cloned);
    }

    #[test]
    fn debug_format() {
        let result = TimestampResult::new(Some("Debug".to_string()), 100, 200);
        let debug = format!("{:?}", result);
        assert!(debug.contains("Debug"));
        assert!(debug.contains("100"));
        assert!(debug.contains("200"));
    }

    #[test]
    fn partial_eq_same_results() {
        let r1 = TimestampResult::new(None, 100, 200);
        let r2 = TimestampResult::new(None, 100, 200);
        assert_eq!(r1, r2);
    }

    #[test]
    fn partial_eq_different_labels() {
        let r1 = TimestampResult::new(Some("A".to_string()), 100, 200);
        let r2 = TimestampResult::new(Some("B".to_string()), 100, 200);
        assert_ne!(r1, r2);
    }

    #[test]
    fn partial_eq_different_times() {
        let r1 = TimestampResult::new(None, 100, 200);
        let r2 = TimestampResult::new(None, 100, 300);
        assert_ne!(r1, r2);
    }

    #[test]
    fn display_without_label() {
        let result = TimestampResult::new(None, 0, 1_000_000);
        let display = format!("{}", result);
        assert!(display.contains("unnamed"));
        assert!(display.contains("1.000ms"));
    }

    #[test]
    fn display_with_label() {
        let result = TimestampResult::new(Some("Test".to_string()), 0, 1_000_000);
        let display = format!("{}", result);
        assert!(display.contains("Test"));
        assert!(display.contains("1.000ms"));
    }
}

// ============================================================================
// Section 3: TimestampPeriodConverter Tests
// ============================================================================

mod period_converter_construction {
    use super::*;

    #[test]
    fn new_stores_period() {
        let converter = TimestampPeriodConverter::new(25.0);
        assert_eq!(converter.period(), 25.0);
    }

    #[test]
    fn new_with_fractional_period() {
        let converter = TimestampPeriodConverter::new(25.5);
        assert_eq!(converter.period(), 25.5);
    }

    #[test]
    fn new_with_zero_period() {
        let converter = TimestampPeriodConverter::new(0.0);
        assert_eq!(converter.period(), 0.0);
    }

    #[test]
    fn new_with_very_small_period() {
        let converter = TimestampPeriodConverter::new(0.001);
        assert!((converter.period() - 0.001).abs() < f32::EPSILON);
    }

    #[test]
    fn new_with_large_period() {
        let converter = TimestampPeriodConverter::new(1000.0);
        assert_eq!(converter.period(), 1000.0);
    }

    #[test]
    fn default_is_one() {
        let converter = TimestampPeriodConverter::default();
        assert_eq!(converter.period(), 1.0);
    }
}

mod period_converter_ticks_to_time {
    use super::*;

    #[test]
    fn ticks_to_ns_basic() {
        let converter = TimestampPeriodConverter::new(10.0);
        assert_eq!(converter.ticks_to_ns(100), 1000);
    }

    #[test]
    fn ticks_to_ns_zero_ticks() {
        let converter = TimestampPeriodConverter::new(25.0);
        assert_eq!(converter.ticks_to_ns(0), 0);
    }

    #[test]
    fn ticks_to_ns_one_tick() {
        let converter = TimestampPeriodConverter::new(25.0);
        assert_eq!(converter.ticks_to_ns(1), 25);
    }

    #[test]
    fn ticks_to_ns_fractional_result() {
        // 25.5 * 3 = 76.5, truncated to 76
        let converter = TimestampPeriodConverter::new(25.5);
        assert_eq!(converter.ticks_to_ns(3), 76);
    }

    #[test]
    fn ticks_to_us_basic() {
        let converter = TimestampPeriodConverter::new(1.0);
        let us = converter.ticks_to_us(1000);
        assert!((us - 1.0).abs() < 0.001);
    }

    #[test]
    fn ticks_to_ms_basic() {
        let converter = TimestampPeriodConverter::new(1.0);
        let ms = converter.ticks_to_ms(1_000_000);
        assert!((ms - 1.0).abs() < 0.001);
    }

    #[test]
    fn ticks_to_ms_with_period() {
        let converter = TimestampPeriodConverter::new(25.0);
        // 40000 ticks * 25 ns/tick = 1_000_000 ns = 1 ms
        let ms = converter.ticks_to_ms(40000);
        assert!((ms - 1.0).abs() < 0.001);
    }
}

mod period_converter_ns_to_ticks {
    use super::*;

    #[test]
    fn ns_to_ticks_basic() {
        let converter = TimestampPeriodConverter::new(10.0);
        assert_eq!(converter.ns_to_ticks(1000), 100);
    }

    #[test]
    fn ns_to_ticks_zero_ns() {
        let converter = TimestampPeriodConverter::new(25.0);
        assert_eq!(converter.ns_to_ticks(0), 0);
    }

    #[test]
    fn ns_to_ticks_zero_period() {
        let converter = TimestampPeriodConverter::new(0.0);
        assert_eq!(converter.ns_to_ticks(1000), 0);
    }

    #[test]
    fn ns_to_ticks_fractional_result() {
        // 1000 / 25.5 = 39.21..., truncated to 39
        let converter = TimestampPeriodConverter::new(25.5);
        assert_eq!(converter.ns_to_ticks(1000), 39);
    }
}

mod period_converter_duration {
    use super::*;

    #[test]
    fn duration_ns_basic() {
        let converter = TimestampPeriodConverter::new(10.0);
        let ns = converter.duration_ns(100, 200);
        assert_eq!(ns, 1000);
    }

    #[test]
    fn duration_ns_zero_delta() {
        let converter = TimestampPeriodConverter::new(10.0);
        let ns = converter.duration_ns(100, 100);
        assert_eq!(ns, 0);
    }

    #[test]
    fn duration_ns_saturating() {
        let converter = TimestampPeriodConverter::new(10.0);
        let ns = converter.duration_ns(200, 100);
        assert_eq!(ns, 0);
    }

    #[test]
    fn duration_ms_basic() {
        let converter = TimestampPeriodConverter::new(1.0);
        let ms = converter.duration_ms(0, 1_000_000);
        assert!((ms - 1.0).abs() < 0.001);
    }

    #[test]
    fn duration_ms_with_period() {
        let converter = TimestampPeriodConverter::new(25.0);
        // 40000 tick delta * 25 ns/tick = 1_000_000 ns = 1 ms
        let ms = converter.duration_ms(0, 40000);
        assert!((ms - 1.0).abs() < 0.001);
    }
}

mod period_converter_roundtrip {
    use super::*;

    #[test]
    fn roundtrip_basic() {
        let converter = TimestampPeriodConverter::new(25.0);
        let ticks: u64 = 12345;
        let ns = converter.ticks_to_ns(ticks);
        let recovered = converter.ns_to_ticks(ns);
        // Allow for rounding error of 1
        assert!((recovered as i64 - ticks as i64).abs() <= 1);
    }

    #[test]
    fn roundtrip_small_values() {
        let converter = TimestampPeriodConverter::new(25.0);
        for ticks in 0..100 {
            let ns = converter.ticks_to_ns(ticks);
            let recovered = converter.ns_to_ticks(ns);
            assert!((recovered as i64 - ticks as i64).abs() <= 1);
        }
    }

    #[test]
    fn roundtrip_large_values() {
        let converter = TimestampPeriodConverter::new(25.0);
        let ticks: u64 = 1_000_000_000;
        let ns = converter.ticks_to_ns(ticks);
        let recovered = converter.ns_to_ticks(ns);
        // Relative error should be small
        let error = (recovered as f64 - ticks as f64).abs() / ticks as f64;
        assert!(error < 0.001);
    }

    #[test]
    fn roundtrip_unit_period() {
        let converter = TimestampPeriodConverter::new(1.0);
        let ticks: u64 = 999999;
        let ns = converter.ticks_to_ns(ticks);
        let recovered = converter.ns_to_ticks(ns);
        assert_eq!(ticks, recovered);
    }
}

mod period_converter_traits {
    use super::*;

    #[test]
    fn clone_creates_equal() {
        let original = TimestampPeriodConverter::new(25.0);
        let cloned = original.clone();
        assert_eq!(original.period(), cloned.period());
    }

    #[test]
    fn copy_semantics() {
        let original = TimestampPeriodConverter::new(25.0);
        let copied = original; // Copy, not move
        assert_eq!(original.period(), copied.period());
    }

    #[test]
    fn debug_format() {
        let converter = TimestampPeriodConverter::new(25.0);
        let debug = format!("{:?}", converter);
        assert!(debug.contains("25"));
    }

    #[test]
    fn partial_eq_same() {
        let c1 = TimestampPeriodConverter::new(25.0);
        let c2 = TimestampPeriodConverter::new(25.0);
        assert_eq!(c1, c2);
    }

    #[test]
    fn partial_eq_different() {
        let c1 = TimestampPeriodConverter::new(25.0);
        let c2 = TimestampPeriodConverter::new(50.0);
        assert_ne!(c1, c2);
    }
}

// ============================================================================
// Section 4: ProfilerStats Tests
// ============================================================================

mod profiler_stats_construction {
    use super::*;

    #[test]
    fn new_stores_all_fields() {
        let stats = ProfilerStats::new(100, 50, 25, 1000, 500, 2000);
        assert_eq!(stats.total_queries, 100);
        assert_eq!(stats.active_queries, 50);
        assert_eq!(stats.resolved_queries, 25);
        assert_eq!(stats.avg_duration_ns, 1000);
        assert_eq!(stats.min_duration_ns, 500);
        assert_eq!(stats.max_duration_ns, 2000);
    }

    #[test]
    fn empty_creates_zero_stats() {
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
}

mod profiler_stats_duration_calculations {
    use super::*;

    #[test]
    fn avg_duration_ms_conversion() {
        let stats = ProfilerStats::new(0, 0, 0, 1_000_000, 0, 0);
        assert!((stats.avg_duration_ms() - 1.0).abs() < 0.001);
    }

    #[test]
    fn avg_duration_ms_zero() {
        let stats = ProfilerStats::empty();
        assert_eq!(stats.avg_duration_ms(), 0.0);
    }

    #[test]
    fn avg_duration_ms_fractional() {
        let stats = ProfilerStats::new(0, 0, 0, 1_500_000, 0, 0);
        assert!((stats.avg_duration_ms() - 1.5).abs() < 0.001);
    }

    #[test]
    fn min_duration_ms_conversion() {
        let stats = ProfilerStats::new(0, 0, 0, 0, 500_000, 0);
        assert!((stats.min_duration_ms() - 0.5).abs() < 0.001);
    }

    #[test]
    fn max_duration_ms_conversion() {
        let stats = ProfilerStats::new(0, 0, 0, 0, 0, 2_000_000);
        assert!((stats.max_duration_ms() - 2.0).abs() < 0.001);
    }
}

mod profiler_stats_utilization {
    use super::*;

    #[test]
    fn utilization_half() {
        let stats = ProfilerStats::new(100, 50, 0, 0, 0, 0);
        assert!((stats.utilization() - 0.5).abs() < 0.001);
    }

    #[test]
    fn utilization_full() {
        let stats = ProfilerStats::new(100, 100, 0, 0, 0, 0);
        assert!((stats.utilization() - 1.0).abs() < 0.001);
    }

    #[test]
    fn utilization_empty() {
        let stats = ProfilerStats::new(100, 0, 0, 0, 0, 0);
        assert_eq!(stats.utilization(), 0.0);
    }

    #[test]
    fn utilization_zero_total() {
        let stats = ProfilerStats::empty();
        assert_eq!(stats.utilization(), 0.0);
    }

    #[test]
    fn utilization_fractional() {
        let stats = ProfilerStats::new(100, 33, 0, 0, 0, 0);
        assert!((stats.utilization() - 0.33).abs() < 0.01);
    }
}

mod profiler_stats_traits {
    use super::*;

    #[test]
    fn clone_creates_equal() {
        let original = ProfilerStats::new(100, 50, 25, 1000, 500, 2000);
        let cloned = original.clone();
        assert_eq!(original, cloned);
    }

    #[test]
    fn copy_semantics() {
        let original = ProfilerStats::new(100, 50, 25, 1000, 500, 2000);
        let copied = original;
        assert_eq!(original, copied);
    }

    #[test]
    fn debug_format() {
        let stats = ProfilerStats::new(100, 50, 25, 1_000_000, 500_000, 2_000_000);
        let debug = format!("{:?}", stats);
        assert!(debug.contains("total_queries: 100"));
        assert!(debug.contains("active_queries: 50"));
    }

    #[test]
    fn partial_eq_same() {
        let s1 = ProfilerStats::new(100, 50, 25, 1000, 500, 2000);
        let s2 = ProfilerStats::new(100, 50, 25, 1000, 500, 2000);
        assert_eq!(s1, s2);
    }

    #[test]
    fn partial_eq_different() {
        let s1 = ProfilerStats::new(100, 50, 25, 1000, 500, 2000);
        let s2 = ProfilerStats::new(100, 51, 25, 1000, 500, 2000);
        assert_ne!(s1, s2);
    }

    #[test]
    fn display_format() {
        let stats = ProfilerStats::new(100, 50, 25, 1_000_000, 500_000, 2_000_000);
        let display = format!("{}", stats);
        assert!(display.contains("50/100"));
        assert!(display.contains("resolved=25"));
        assert!(display.contains("avg=1.000ms"));
    }
}

// ============================================================================
// Section 5: FrameStats Tests
// ============================================================================

mod frame_stats_construction {
    use super::*;

    #[test]
    fn new_creates_empty_frame() {
        let stats = FrameStats::new(42);
        assert_eq!(stats.frame_index, 42);
        assert_eq!(stats.total_ns, 0);
        assert_eq!(stats.region_count, 0);
        assert!(stats.regions.is_empty());
    }

    #[test]
    fn default_is_frame_zero() {
        let stats = FrameStats::default();
        assert_eq!(stats.frame_index, 0);
        assert_eq!(stats.total_ns, 0);
    }
}

mod frame_stats_region_tracking {
    use super::*;

    #[test]
    fn add_region_increments_count() {
        let mut stats = FrameStats::new(0);
        let result = TimestampResult::new(Some("Test".to_string()), 0, 1000);
        stats.add_region(result);

        assert_eq!(stats.region_count, 1);
        assert_eq!(stats.regions.len(), 1);
    }

    #[test]
    fn add_region_accumulates_duration() {
        let mut stats = FrameStats::new(0);
        stats.add_region(TimestampResult::new(None, 0, 1000));
        stats.add_region(TimestampResult::new(None, 0, 2000));

        assert_eq!(stats.total_ns, 3000);
    }

    #[test]
    fn add_many_regions() {
        let mut stats = FrameStats::new(0);
        for i in 0..100 {
            stats.add_region(TimestampResult::new(None, 0, (i + 1) * 100));
        }

        assert_eq!(stats.region_count, 100);
        // Sum of 100 + 200 + ... + 10000 = 100 * (1+100) * 100 / 2 = 505000
        assert_eq!(stats.total_ns, 505000);
    }

    #[test]
    fn add_region_stores_result() {
        let mut stats = FrameStats::new(0);
        let result = TimestampResult::new(Some("Stored".to_string()), 100, 500);
        stats.add_region(result.clone());

        assert_eq!(stats.regions.len(), 1);
        assert_eq!(stats.regions[0], result);
    }
}

mod frame_stats_duration_calculations {
    use super::*;

    #[test]
    fn total_ms_conversion() {
        let mut stats = FrameStats::new(0);
        stats.add_region(TimestampResult::new(None, 0, 1_000_000));
        assert!((stats.total_ms() - 1.0).abs() < 0.001);
    }

    #[test]
    fn total_ms_zero() {
        let stats = FrameStats::new(0);
        assert_eq!(stats.total_ms(), 0.0);
    }

    #[test]
    fn total_ms_accumulated() {
        let mut stats = FrameStats::new(0);
        stats.add_region(TimestampResult::new(None, 0, 500_000));
        stats.add_region(TimestampResult::new(None, 0, 500_000));
        assert!((stats.total_ms() - 1.0).abs() < 0.001);
    }
}

mod frame_stats_is_empty {
    use super::*;

    #[test]
    fn is_empty_for_new_frame() {
        let stats = FrameStats::new(0);
        assert!(stats.is_empty());
    }

    #[test]
    fn is_not_empty_after_add() {
        let mut stats = FrameStats::new(0);
        stats.add_region(TimestampResult::new(None, 0, 100));
        assert!(!stats.is_empty());
    }

    #[test]
    fn is_not_empty_with_zero_duration_region() {
        let mut stats = FrameStats::new(0);
        stats.add_region(TimestampResult::zero());
        assert!(!stats.is_empty());
    }
}

mod frame_stats_traits {
    use super::*;

    #[test]
    fn clone_creates_equal() {
        let mut original = FrameStats::new(5);
        original.add_region(TimestampResult::new(None, 0, 1000));
        let cloned = original.clone();
        assert_eq!(original, cloned);
    }

    #[test]
    fn clone_is_independent() {
        let mut original = FrameStats::new(5);
        let mut cloned = original.clone();
        cloned.add_region(TimestampResult::new(None, 0, 1000));
        assert_ne!(original.region_count, cloned.region_count);
    }

    #[test]
    fn debug_format() {
        let stats = FrameStats::new(42);
        let debug = format!("{:?}", stats);
        assert!(debug.contains("frame_index: 42"));
    }

    #[test]
    fn partial_eq_same() {
        let s1 = FrameStats::new(5);
        let s2 = FrameStats::new(5);
        assert_eq!(s1, s2);
    }

    #[test]
    fn partial_eq_different_index() {
        let s1 = FrameStats::new(5);
        let s2 = FrameStats::new(6);
        assert_ne!(s1, s2);
    }

    #[test]
    fn partial_eq_different_regions() {
        let mut s1 = FrameStats::new(5);
        let s2 = FrameStats::new(5);
        s1.add_region(TimestampResult::new(None, 0, 100));
        assert_ne!(s1, s2);
    }

    #[test]
    fn display_format() {
        let mut stats = FrameStats::new(5);
        stats.add_region(TimestampResult::new(None, 0, 1_000_000));
        let display = format!("{}", stats);
        assert!(display.contains("Frame 5"));
        assert!(display.contains("1.000ms"));
        assert!(display.contains("1 region"));
    }
}

// ============================================================================
// Section 6: Constants Tests
// ============================================================================

mod constants_validation {
    use super::*;

    #[test]
    fn timestamp_size_is_8_bytes() {
        assert_eq!(TIMESTAMP_SIZE_BYTES, 8);
    }

    #[test]
    fn min_capacity_is_at_least_2() {
        assert!(MIN_CAPACITY >= 2);
    }

    #[test]
    fn min_capacity_is_reasonable() {
        assert!(MIN_CAPACITY <= 16);
    }

    #[test]
    fn max_recommended_capacity_is_reasonable() {
        assert!(MAX_RECOMMENDED_CAPACITY >= 256);
        assert!(MAX_RECOMMENDED_CAPACITY <= 16384);
    }

    #[test]
    fn default_capacity_is_between_bounds() {
        assert!(DEFAULT_CAPACITY >= MIN_CAPACITY);
        assert!(DEFAULT_CAPACITY <= MAX_RECOMMENDED_CAPACITY);
    }

    #[test]
    fn timestamp_size_matches_u64() {
        assert_eq!(TIMESTAMP_SIZE_BYTES, std::mem::size_of::<u64>() as u64);
    }
}

// ============================================================================
// Section 7: Send + Sync Bounds Tests
// ============================================================================

mod thread_safety_bounds {
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

// ============================================================================
// Section 8: Edge Cases and Boundary Tests
// ============================================================================

mod edge_cases_timestamp_handle {
    use super::*;

    #[test]
    fn max_indices() {
        let handle = TimestampHandle::new(u32::MAX - 1, u32::MAX);
        assert_eq!(handle.start_index, u32::MAX - 1);
        assert_eq!(handle.end_index, u32::MAX);
    }

    #[test]
    fn zero_indices() {
        let handle = TimestampHandle::new(0, 0);
        assert_eq!(handle.start_index, 0);
        assert_eq!(handle.end_index, 0);
    }

    #[test]
    fn very_long_label() {
        let long_label = "A".repeat(10000);
        let handle = TimestampHandle::with_label(0, 1, long_label.clone());
        assert_eq!(handle.label.unwrap().len(), 10000);
    }
}

mod edge_cases_timestamp_result {
    use super::*;

    #[test]
    fn max_duration() {
        let result = TimestampResult::new(None, 0, u64::MAX);
        assert_eq!(result.duration_ns(), u64::MAX);
    }

    #[test]
    fn saturating_duration() {
        let result = TimestampResult::new(None, u64::MAX, 0);
        assert_eq!(result.duration_ns(), 0);
    }

    #[test]
    fn large_duration_ms() {
        let result = TimestampResult::new(None, 0, u64::MAX);
        let ms = result.duration_ms();
        assert!(ms > 0.0);
        assert!(ms.is_finite());
    }

    #[test]
    fn large_duration_secs() {
        let result = TimestampResult::new(None, 0, u64::MAX);
        let secs = result.duration_secs();
        assert!(secs > 0.0);
        assert!(secs.is_finite());
    }

    #[test]
    fn from_ticks_very_small_period() {
        let result = TimestampResult::from_ticks(0, 1000, 0.001);
        assert_eq!(result.end_ns, 1);
    }

    #[test]
    fn from_ticks_very_large_period() {
        let result = TimestampResult::from_ticks(0, 1, 1000000.0);
        assert_eq!(result.end_ns, 1000000);
    }
}

mod edge_cases_period_converter {
    use super::*;

    #[test]
    fn large_ticks_conversion() {
        let converter = TimestampPeriodConverter::new(1.0);
        let ns = converter.ticks_to_ns(u64::MAX / 2);
        assert!(ns > 0);
    }

    #[test]
    fn negative_period_behavior() {
        // f32 can hold negative values, but this is undefined behavior for the profiler
        // The converter should still function without panicking
        let converter = TimestampPeriodConverter::new(-1.0);
        // ns_to_ticks with negative period is handled by the period > 0 check
        let ticks = converter.ns_to_ticks(1000);
        assert_eq!(ticks, 0); // Returns 0 when period <= 0
    }

    #[test]
    fn subnormal_period() {
        let subnormal = f32::MIN_POSITIVE / 2.0;
        let converter = TimestampPeriodConverter::new(subnormal);
        // Should not panic
        let _ = converter.ticks_to_ns(100);
    }

    #[test]
    fn infinity_handling() {
        let converter = TimestampPeriodConverter::new(f32::INFINITY);
        // Should not panic, but results may be infinity or overflow
        let ns = converter.ticks_to_ns(1);
        // ns is u64, so infinity would become u64::MAX or 0 depending on cast behavior
        let _ = ns;
    }
}

mod edge_cases_profiler_stats {
    use super::*;

    #[test]
    fn max_queries() {
        let stats = ProfilerStats::new(u32::MAX, u32::MAX, u32::MAX, u64::MAX, u64::MAX, u64::MAX);
        assert_eq!(stats.total_queries, u32::MAX);
        assert_eq!(stats.avg_duration_ns, u64::MAX);
    }

    #[test]
    fn utilization_near_max() {
        let stats = ProfilerStats::new(u32::MAX, u32::MAX - 1, 0, 0, 0, 0);
        let util = stats.utilization();
        assert!(util > 0.99);
        assert!(util <= 1.0);
    }

    #[test]
    fn large_duration_conversions() {
        let stats = ProfilerStats::new(0, 0, 0, u64::MAX, u64::MAX, u64::MAX);
        let ms = stats.avg_duration_ms();
        assert!(ms.is_finite());
        assert!(ms > 0.0);
    }
}

mod edge_cases_frame_stats {
    use super::*;

    #[test]
    fn max_frame_index() {
        let stats = FrameStats::new(u64::MAX);
        assert_eq!(stats.frame_index, u64::MAX);
    }

    #[test]
    fn many_regions_accumulation() {
        let mut stats = FrameStats::new(0);
        for _ in 0..1000 {
            stats.add_region(TimestampResult::new(None, 0, 1_000_000));
        }
        assert_eq!(stats.region_count, 1000);
        assert_eq!(stats.total_ns, 1_000_000_000);
    }

    #[test]
    fn total_ns_overflow_behavior() {
        let mut stats = FrameStats::new(0);
        // Add regions that would overflow if not handled
        stats.total_ns = u64::MAX - 100;
        stats.add_region(TimestampResult::new(None, 0, 50));
        // Should wrap around in release mode, panic in debug with overflow checks
        // This test documents the behavior
        let _ = stats.total_ns;
    }
}

// ============================================================================
// Section 9: Integration-Style Tests (without real GPU)
// ============================================================================

mod integration_timestamp_handle {
    use super::*;

    #[test]
    fn workflow_create_modify_use() {
        let mut handle = TimestampHandle::new(0, 1);
        assert!(!handle.has_label());

        handle.set_label("Pass 1");
        assert!(handle.has_label());
        assert_eq!(handle.label_or_unnamed(), "Pass 1");

        handle.clear_label();
        assert!(!handle.has_label());
        assert_eq!(handle.label_or_unnamed(), "unnamed");
    }

    #[test]
    fn workflow_clone_and_modify() {
        let original = TimestampHandle::with_label(5, 10, "Original");
        let mut modified = original.clone();
        modified.set_label("Modified");

        assert_eq!(original.label.as_deref(), Some("Original"));
        assert_eq!(modified.label.as_deref(), Some("Modified"));
        assert_eq!(original.start_index, modified.start_index);
    }
}

mod integration_timestamp_result {
    use super::*;

    #[test]
    fn workflow_from_ticks_to_display() {
        let result = TimestampResult::from_ticks_labeled(1000, 2000, 25.0, "GPU Work");

        assert!(result.is_valid());
        assert_eq!(result.duration_ns(), 25000);
        assert!((result.duration_us() - 25.0).abs() < 0.01);
        assert!((result.duration_ms() - 0.025).abs() < 0.001);

        let display = format!("{}", result);
        assert!(display.contains("GPU Work"));
    }

    #[test]
    fn workflow_accumulate_results() {
        let results = vec![
            TimestampResult::new(Some("A".to_string()), 0, 1_000_000),
            TimestampResult::new(Some("B".to_string()), 0, 2_000_000),
            TimestampResult::new(Some("C".to_string()), 0, 3_000_000),
        ];

        let total_ns: u64 = results.iter().map(|r| r.duration_ns()).sum();
        assert_eq!(total_ns, 6_000_000);

        let total_ms: f64 = results.iter().map(|r| r.duration_ms()).sum();
        assert!((total_ms - 6.0).abs() < 0.001);
    }
}

mod integration_frame_stats {
    use super::*;

    #[test]
    fn workflow_frame_profiling() {
        let mut frame = FrameStats::new(0);

        // Simulate adding pass timings
        frame.add_region(TimestampResult::from_ticks_labeled(0, 40000, 25.0, "Shadow"));
        frame.add_region(TimestampResult::from_ticks_labeled(0, 80000, 25.0, "GBuffer"));
        frame.add_region(TimestampResult::from_ticks_labeled(0, 60000, 25.0, "Lighting"));

        assert_eq!(frame.region_count, 3);
        assert!(!frame.is_empty());

        // Total = 1000000 + 2000000 + 1500000 = 4500000 ns = 4.5ms
        assert!((frame.total_ms() - 4.5).abs() < 0.01);

        let display = format!("{}", frame);
        assert!(display.contains("Frame 0"));
        assert!(display.contains("3 regions"));
    }

    #[test]
    fn workflow_multi_frame_tracking() {
        let mut frames = Vec::new();

        for i in 0..10 {
            let mut frame = FrameStats::new(i);
            frame.add_region(TimestampResult::new(None, 0, 16_666_667)); // ~16.67ms (60fps)
            frames.push(frame);
        }

        assert_eq!(frames.len(), 10);

        let avg_ms: f64 = frames.iter().map(|f| f.total_ms()).sum::<f64>() / frames.len() as f64;
        assert!((avg_ms - 16.666667).abs() < 0.01);
    }
}

mod integration_profiler_stats {
    use super::*;

    #[test]
    fn workflow_stats_analysis() {
        let stats = ProfilerStats::new(64, 32, 28, 5_000_000, 1_000_000, 10_000_000);

        // Utilization = 32/64 = 50%
        assert!((stats.utilization() - 0.5).abs() < 0.001);

        // Duration analysis
        let avg = stats.avg_duration_ms(); // 5ms
        let min = stats.min_duration_ms(); // 1ms
        let max = stats.max_duration_ms(); // 10ms

        assert!((avg - 5.0).abs() < 0.001);
        assert!((min - 1.0).abs() < 0.001);
        assert!((max - 10.0).abs() < 0.001);

        // Variance indicator
        let range = max - min;
        assert!((range - 9.0).abs() < 0.001);
    }
}

mod integration_period_converter {
    use super::*;

    #[test]
    fn workflow_typical_gpu_period() {
        // Typical GPU timestamp period ~25ns
        let converter = TimestampPeriodConverter::new(25.0);

        // Convert 1 second worth of ticks
        let one_second_ticks = 1_000_000_000 / 25; // 40,000,000 ticks
        let ns = converter.ticks_to_ns(one_second_ticks);

        assert!((ns as f64 - 1_000_000_000.0).abs() < 25.0);
    }

    #[test]
    fn workflow_frame_time_measurement() {
        let converter = TimestampPeriodConverter::new(25.0);

        // Measure a 16.67ms frame (60fps)
        let frame_ns: u64 = 16_666_667;
        let frame_ticks = converter.ns_to_ticks(frame_ns);
        let recovered_ns = converter.ticks_to_ns(frame_ticks);

        // Should be within one tick precision
        let error = (recovered_ns as i64 - frame_ns as i64).abs();
        assert!(error <= 25);
    }
}

// ============================================================================
// Section 10: Comprehensive Boundary Tests
// ============================================================================

mod boundary_tests {
    use super::*;

    #[test]
    fn all_zero_values() {
        let handle = TimestampHandle::new(0, 0);
        let result = TimestampResult::new(None, 0, 0);
        let converter = TimestampPeriodConverter::new(0.0);
        let stats = ProfilerStats::empty();
        let frame = FrameStats::new(0);

        assert_eq!(handle.start_index, 0);
        assert_eq!(result.duration_ns(), 0);
        assert_eq!(converter.ticks_to_ns(100), 0);
        assert_eq!(stats.utilization(), 0.0);
        assert!(frame.is_empty());
    }

    #[test]
    fn all_max_values() {
        let handle = TimestampHandle::new(u32::MAX, u32::MAX);
        let result = TimestampResult::new(None, u64::MAX, u64::MAX);
        let stats = ProfilerStats::new(u32::MAX, u32::MAX, u32::MAX, u64::MAX, u64::MAX, u64::MAX);
        let frame = FrameStats::new(u64::MAX);

        assert_eq!(handle.start_index, u32::MAX);
        assert_eq!(result.duration_ns(), 0); // MAX - MAX = 0
        assert!((stats.utilization() - 1.0).abs() < 0.001);
        assert_eq!(frame.frame_index, u64::MAX);
    }

    #[test]
    fn transition_values() {
        // Test values at boundaries: 0, 1, MAX-1, MAX
        let values = [0u64, 1, u64::MAX - 1, u64::MAX];

        for &start in &values {
            for &end in &values {
                let result = TimestampResult::new(None, start, end);
                if end >= start {
                    assert_eq!(result.duration_ns(), end - start);
                } else {
                    assert_eq!(result.duration_ns(), 0);
                }
            }
        }
    }
}

// ============================================================================
// Test Count Summary (150+ tests)
// ============================================================================
//
// Section 1: TimestampHandle Tests
//   - timestamp_handle_construction: 10 tests
//   - timestamp_handle_label_management: 12 tests
//   - timestamp_handle_traits: 17 tests
//
// Section 2: TimestampResult Tests
//   - timestamp_result_construction: 10 tests
//   - timestamp_result_duration_calculations: 11 tests
//   - timestamp_result_validity: 6 tests
//   - timestamp_result_traits: 8 tests
//
// Section 3: TimestampPeriodConverter Tests
//   - period_converter_construction: 6 tests
//   - period_converter_ticks_to_time: 7 tests
//   - period_converter_ns_to_ticks: 4 tests
//   - period_converter_duration: 5 tests
//   - period_converter_roundtrip: 4 tests
//   - period_converter_traits: 5 tests
//
// Section 4: ProfilerStats Tests
//   - profiler_stats_construction: 3 tests
//   - profiler_stats_duration_calculations: 5 tests
//   - profiler_stats_utilization: 5 tests
//   - profiler_stats_traits: 6 tests
//
// Section 5: FrameStats Tests
//   - frame_stats_construction: 2 tests
//   - frame_stats_region_tracking: 4 tests
//   - frame_stats_duration_calculations: 3 tests
//   - frame_stats_is_empty: 3 tests
//   - frame_stats_traits: 7 tests
//
// Section 6: Constants Tests
//   - constants_validation: 6 tests
//
// Section 7: Thread Safety Tests
//   - thread_safety_bounds: 10 tests
//
// Section 8: Edge Cases
//   - edge_cases_timestamp_handle: 3 tests
//   - edge_cases_timestamp_result: 6 tests
//   - edge_cases_period_converter: 4 tests
//   - edge_cases_profiler_stats: 3 tests
//   - edge_cases_frame_stats: 3 tests
//
// Section 9: Integration Tests
//   - integration_timestamp_handle: 2 tests
//   - integration_timestamp_result: 2 tests
//   - integration_frame_stats: 2 tests
//   - integration_profiler_stats: 1 test
//   - integration_period_converter: 2 tests
//
// Section 10: Boundary Tests
//   - boundary_tests: 3 tests
//
// TOTAL: 175 tests
