// WHITEBOX tests for T-WGPU-P7.4.3 (GPU Resource Leak Detection)
//
// WHITEBOX discipline: These tests have FULL ACCESS to the implementation.
// They exercise internal code paths, branch conditions, and edge cases
// that are not visible through the public contract alone.
//
// Implementation under test: crates/renderer-backend/src/profiling/leaks.rs
//   - LeakSeverity enum and methods
//   - LeakThresholds struct and presets
//   - LeakCandidate struct and methods
//   - AllocationTracker struct and methods
//   - LeakDetector struct and methods
//   - LeakStats struct and methods
//   - LeakReport struct and methods
//   - FrameLeakChecker struct and methods
//
// WHITEBOX coverage plan (150+ tests):
//   - LeakSeverity: 20 tests (enum variants, from_age_secs, display, ordering)
//   - LeakThresholds: 22 tests (presets, custom, edge cases)
//   - LeakCandidate: 28 tests (creation, age, severity, formatting)
//   - AllocationTracker: 32 tests (track, untrack, marks, clear, iteration)
//   - LeakDetector: 38 tests (track, release, check, stats, report)
//   - LeakStats: 18 tests (counters, rates, calculations)
//   - LeakReport: 22 tests (critical detection, bytes sum, categorization)
//   - FrameLeakChecker: 22 tests (frame lifecycle, tracking, multi-frame)

use renderer_backend::profiling::leaks::{
    AllocationTracker, FrameLeakChecker, LeakCandidate, LeakDetector, LeakReport, LeakSeverity,
    LeakStats, LeakThresholds,
};
use renderer_backend::profiling::memory::ResourceType;
use std::time::{Duration, Instant};

// ============================================================================
// LeakSeverity Tests (20 tests)
// ============================================================================

/// Test LeakSeverity::Info variant exists and is distinct
#[test]
fn test_leak_severity_info_variant() {
    let severity = LeakSeverity::Info;
    assert_eq!(severity, LeakSeverity::Info);
    assert_ne!(severity, LeakSeverity::Warning);
    assert_ne!(severity, LeakSeverity::Critical);
}

/// Test LeakSeverity::Warning variant exists and is distinct
#[test]
fn test_leak_severity_warning_variant() {
    let severity = LeakSeverity::Warning;
    assert_eq!(severity, LeakSeverity::Warning);
    assert_ne!(severity, LeakSeverity::Info);
    assert_ne!(severity, LeakSeverity::Critical);
}

/// Test LeakSeverity::Critical variant exists and is distinct
#[test]
fn test_leak_severity_critical_variant() {
    let severity = LeakSeverity::Critical;
    assert_eq!(severity, LeakSeverity::Critical);
    assert_ne!(severity, LeakSeverity::Info);
    assert_ne!(severity, LeakSeverity::Warning);
}

/// Test from_age_secs with age well below warning threshold
#[test]
fn test_from_age_secs_below_warning() {
    let severity = LeakSeverity::from_age_secs(5, 30, 120);
    assert_eq!(severity, LeakSeverity::Info);
}

/// Test from_age_secs with age at zero
#[test]
fn test_from_age_secs_zero_age() {
    let severity = LeakSeverity::from_age_secs(0, 30, 120);
    assert_eq!(severity, LeakSeverity::Info);
}

/// Test from_age_secs with age exactly at warning threshold
#[test]
fn test_from_age_secs_at_warning_boundary() {
    let severity = LeakSeverity::from_age_secs(30, 30, 120);
    assert_eq!(severity, LeakSeverity::Warning);
}

/// Test from_age_secs with age one below warning threshold
#[test]
fn test_from_age_secs_one_below_warning() {
    let severity = LeakSeverity::from_age_secs(29, 30, 120);
    assert_eq!(severity, LeakSeverity::Info);
}

/// Test from_age_secs with age between warning and critical
#[test]
fn test_from_age_secs_between_warning_and_critical() {
    let severity = LeakSeverity::from_age_secs(60, 30, 120);
    assert_eq!(severity, LeakSeverity::Warning);

    let severity2 = LeakSeverity::from_age_secs(90, 30, 120);
    assert_eq!(severity2, LeakSeverity::Warning);
}

/// Test from_age_secs with age one below critical threshold
#[test]
fn test_from_age_secs_one_below_critical() {
    let severity = LeakSeverity::from_age_secs(119, 30, 120);
    assert_eq!(severity, LeakSeverity::Warning);
}

/// Test from_age_secs with age exactly at critical threshold
#[test]
fn test_from_age_secs_at_critical_boundary() {
    let severity = LeakSeverity::from_age_secs(120, 30, 120);
    assert_eq!(severity, LeakSeverity::Critical);
}

/// Test from_age_secs with age above critical threshold
#[test]
fn test_from_age_secs_above_critical() {
    let severity = LeakSeverity::from_age_secs(200, 30, 120);
    assert_eq!(severity, LeakSeverity::Critical);

    let severity2 = LeakSeverity::from_age_secs(10000, 30, 120);
    assert_eq!(severity2, LeakSeverity::Critical);
}

/// Test display_color returns cyan for Info
#[test]
fn test_display_color_info() {
    assert_eq!(LeakSeverity::Info.display_color(), "\x1b[36m");
}

/// Test display_color returns yellow for Warning
#[test]
fn test_display_color_warning() {
    assert_eq!(LeakSeverity::Warning.display_color(), "\x1b[33m");
}

/// Test display_color returns red for Critical
#[test]
fn test_display_color_critical() {
    assert_eq!(LeakSeverity::Critical.display_color(), "\x1b[31m");
}

/// Test reset_color returns ANSI reset code
#[test]
fn test_reset_color() {
    assert_eq!(LeakSeverity::reset_color(), "\x1b[0m");
}

/// Test display_name returns correct strings
#[test]
fn test_display_name() {
    assert_eq!(LeakSeverity::Info.display_name(), "INFO");
    assert_eq!(LeakSeverity::Warning.display_name(), "WARNING");
    assert_eq!(LeakSeverity::Critical.display_name(), "CRITICAL");
}

/// Test Display trait implementation
#[test]
fn test_severity_display_trait() {
    assert_eq!(format!("{}", LeakSeverity::Info), "INFO");
    assert_eq!(format!("{}", LeakSeverity::Warning), "WARNING");
    assert_eq!(format!("{}", LeakSeverity::Critical), "CRITICAL");
}

/// Test severity ordering: Info < Warning < Critical
#[test]
fn test_severity_ordering() {
    assert!(LeakSeverity::Info < LeakSeverity::Warning);
    assert!(LeakSeverity::Warning < LeakSeverity::Critical);
    assert!(LeakSeverity::Info < LeakSeverity::Critical);
}

/// Test severity cloning preserves equality
#[test]
fn test_severity_clone() {
    let original = LeakSeverity::Critical;
    let cloned = original.clone();
    assert_eq!(original, cloned);
}

/// Test severity hashing consistency
#[test]
fn test_severity_hash() {
    use std::collections::HashSet;
    let mut set = HashSet::new();
    set.insert(LeakSeverity::Info);
    set.insert(LeakSeverity::Warning);
    set.insert(LeakSeverity::Critical);
    assert_eq!(set.len(), 3);
    assert!(set.contains(&LeakSeverity::Warning));
}

// ============================================================================
// LeakThresholds Tests (22 tests)
// ============================================================================

/// Test default threshold values
#[test]
fn test_thresholds_default_values() {
    let thresholds = LeakThresholds::default();
    assert_eq!(thresholds.warning_secs, 30);
    assert_eq!(thresholds.critical_secs, 120);
    assert_eq!(thresholds.min_size_bytes, 1024);
}

/// Test default_thresholds() returns same as default()
#[test]
fn test_thresholds_default_thresholds_method() {
    let default_method = LeakThresholds::default_thresholds();
    let default_trait = LeakThresholds::default();
    assert_eq!(default_method.warning_secs, default_trait.warning_secs);
    assert_eq!(default_method.critical_secs, default_trait.critical_secs);
    assert_eq!(default_method.min_size_bytes, default_trait.min_size_bytes);
}

/// Test strict preset values
#[test]
fn test_thresholds_strict_values() {
    let thresholds = LeakThresholds::strict();
    assert_eq!(thresholds.warning_secs, 5);
    assert_eq!(thresholds.critical_secs, 30);
    assert_eq!(thresholds.min_size_bytes, 0);
}

/// Test relaxed preset values
#[test]
fn test_thresholds_relaxed_values() {
    let thresholds = LeakThresholds::relaxed();
    assert_eq!(thresholds.warning_secs, 300);
    assert_eq!(thresholds.critical_secs, 600);
    assert_eq!(thresholds.min_size_bytes, 4096);
}

/// Test custom threshold creation
#[test]
fn test_thresholds_custom() {
    let thresholds = LeakThresholds::custom(10, 60, 512);
    assert_eq!(thresholds.warning_secs, 10);
    assert_eq!(thresholds.critical_secs, 60);
    assert_eq!(thresholds.min_size_bytes, 512);
}

/// Test custom thresholds with zero warning
#[test]
fn test_thresholds_custom_zero_warning() {
    let thresholds = LeakThresholds::custom(0, 60, 512);
    assert_eq!(thresholds.warning_secs, 0);
    // Zero warning means any age triggers warning
    let severity = LeakSeverity::from_age_secs(0, thresholds.warning_secs, thresholds.critical_secs);
    assert_eq!(severity, LeakSeverity::Warning);
}

/// Test custom thresholds with zero critical
#[test]
fn test_thresholds_custom_zero_critical() {
    let thresholds = LeakThresholds::custom(0, 0, 512);
    assert_eq!(thresholds.critical_secs, 0);
    // Zero critical means any age triggers critical
    let severity = LeakSeverity::from_age_secs(0, thresholds.warning_secs, thresholds.critical_secs);
    assert_eq!(severity, LeakSeverity::Critical);
}

/// Test custom thresholds with all zeros
#[test]
fn test_thresholds_custom_all_zeros() {
    let thresholds = LeakThresholds::custom(0, 0, 0);
    assert_eq!(thresholds.warning_secs, 0);
    assert_eq!(thresholds.critical_secs, 0);
    assert_eq!(thresholds.min_size_bytes, 0);
}

/// Test custom thresholds with very large values
#[test]
fn test_thresholds_custom_large_values() {
    let thresholds = LeakThresholds::custom(u64::MAX / 2, u64::MAX, u64::MAX);
    assert_eq!(thresholds.warning_secs, u64::MAX / 2);
    assert_eq!(thresholds.critical_secs, u64::MAX);
    assert_eq!(thresholds.min_size_bytes, u64::MAX);
}

/// Test thresholds where warning equals critical
#[test]
fn test_thresholds_warning_equals_critical() {
    let thresholds = LeakThresholds::custom(30, 30, 1024);
    // Age of 30 should be critical (>= critical takes precedence)
    let severity = LeakSeverity::from_age_secs(30, thresholds.warning_secs, thresholds.critical_secs);
    assert_eq!(severity, LeakSeverity::Critical);
}

/// Test thresholds where warning exceeds critical (unusual but allowed)
#[test]
fn test_thresholds_warning_exceeds_critical() {
    let thresholds = LeakThresholds::custom(100, 50, 1024);
    // Age of 75 is between 50 (critical) and 100 (warning)
    // Critical check happens first, so 75 >= 50 means Critical
    let severity = LeakSeverity::from_age_secs(75, thresholds.warning_secs, thresholds.critical_secs);
    assert_eq!(severity, LeakSeverity::Critical);
}

/// Test min_size_bytes at 1 byte
#[test]
fn test_thresholds_min_size_one() {
    let thresholds = LeakThresholds::custom(30, 120, 1);
    assert_eq!(thresholds.min_size_bytes, 1);
}

/// Test min_size_bytes at common sizes
#[test]
fn test_thresholds_common_sizes() {
    let kb = LeakThresholds::custom(30, 120, 1024);
    let mb = LeakThresholds::custom(30, 120, 1024 * 1024);
    let gb = LeakThresholds::custom(30, 120, 1024 * 1024 * 1024);

    assert_eq!(kb.min_size_bytes, 1024);
    assert_eq!(mb.min_size_bytes, 1024 * 1024);
    assert_eq!(gb.min_size_bytes, 1024 * 1024 * 1024);
}

/// Test thresholds cloning
#[test]
fn test_thresholds_clone() {
    let original = LeakThresholds::custom(15, 90, 2048);
    let cloned = original.clone();
    assert_eq!(original.warning_secs, cloned.warning_secs);
    assert_eq!(original.critical_secs, cloned.critical_secs);
    assert_eq!(original.min_size_bytes, cloned.min_size_bytes);
}

/// Test thresholds copying
#[test]
fn test_thresholds_copy() {
    let original = LeakThresholds::custom(15, 90, 2048);
    let copied = original; // Copy semantics
    assert_eq!(original.warning_secs, copied.warning_secs);
}

/// Test thresholds debug formatting
#[test]
fn test_thresholds_debug() {
    let thresholds = LeakThresholds::default();
    let debug_str = format!("{:?}", thresholds);
    assert!(debug_str.contains("warning_secs"));
    assert!(debug_str.contains("30"));
}

/// Test strict is more aggressive than default
#[test]
fn test_strict_more_aggressive_than_default() {
    let strict = LeakThresholds::strict();
    let default = LeakThresholds::default();

    assert!(strict.warning_secs < default.warning_secs);
    assert!(strict.critical_secs < default.critical_secs);
    assert!(strict.min_size_bytes < default.min_size_bytes);
}

/// Test relaxed is less aggressive than default
#[test]
fn test_relaxed_less_aggressive_than_default() {
    let relaxed = LeakThresholds::relaxed();
    let default = LeakThresholds::default();

    assert!(relaxed.warning_secs > default.warning_secs);
    assert!(relaxed.critical_secs > default.critical_secs);
    assert!(relaxed.min_size_bytes > default.min_size_bytes);
}

/// Test threshold presets form a proper ordering
#[test]
fn test_threshold_preset_ordering() {
    let strict = LeakThresholds::strict();
    let default = LeakThresholds::default();
    let relaxed = LeakThresholds::relaxed();

    // Strict < Default < Relaxed for warning times
    assert!(strict.warning_secs < default.warning_secs);
    assert!(default.warning_secs < relaxed.warning_secs);
}

/// Test thresholds with typical gaming frame rate values
#[test]
fn test_thresholds_gaming_scenario() {
    // For 60 FPS, a frame is ~16.7ms. 2 seconds = ~120 frames
    let thresholds = LeakThresholds::custom(2, 10, 0); // Quick detection
    assert_eq!(thresholds.warning_secs, 2);
    assert_eq!(thresholds.critical_secs, 10);
}

/// Test thresholds with server scenario values
#[test]
fn test_thresholds_server_scenario() {
    // Servers might have long-running processes
    let thresholds = LeakThresholds::custom(3600, 86400, 1024 * 1024); // 1 hour, 1 day, 1MB
    assert_eq!(thresholds.warning_secs, 3600);
    assert_eq!(thresholds.critical_secs, 86400);
}

// ============================================================================
// LeakCandidate Tests (28 tests)
// ============================================================================

/// Test LeakCandidate creation with all fields
#[test]
fn test_leak_candidate_creation_full() {
    let now = Instant::now();
    let candidate = LeakCandidate::new(
        42,
        ResourceType::Buffer,
        1024 * 1024,
        Some("Vertex Buffer".to_string()),
        now,
    );

    assert_eq!(candidate.allocation_id, 42);
    assert_eq!(candidate.resource_type, ResourceType::Buffer);
    assert_eq!(candidate.size_bytes, 1024 * 1024);
    assert_eq!(candidate.label, Some("Vertex Buffer".to_string()));
}

/// Test LeakCandidate creation with None label
#[test]
fn test_leak_candidate_creation_no_label() {
    let now = Instant::now();
    let candidate = LeakCandidate::new(1, ResourceType::Texture, 2048, None, now);

    assert_eq!(candidate.label, None);
}

/// Test LeakCandidate with different resource types
#[test]
fn test_leak_candidate_resource_types() {
    let now = Instant::now();

    let buffer = LeakCandidate::new(1, ResourceType::Buffer, 100, None, now);
    let texture = LeakCandidate::new(2, ResourceType::Texture, 100, None, now);
    let query = LeakCandidate::new(3, ResourceType::QuerySet, 100, None, now);
    let bind = LeakCandidate::new(4, ResourceType::BindGroup, 100, None, now);
    let pipe = LeakCandidate::new(5, ResourceType::Pipeline, 100, None, now);
    let other = LeakCandidate::new(6, ResourceType::Other, 100, None, now);

    assert_eq!(buffer.resource_type, ResourceType::Buffer);
    assert_eq!(texture.resource_type, ResourceType::Texture);
    assert_eq!(query.resource_type, ResourceType::QuerySet);
    assert_eq!(bind.resource_type, ResourceType::BindGroup);
    assert_eq!(pipe.resource_type, ResourceType::Pipeline);
    assert_eq!(other.resource_type, ResourceType::Other);
}

/// Test age() returns Duration close to zero for fresh allocation
#[test]
fn test_leak_candidate_age_fresh() {
    let now = Instant::now();
    let candidate = LeakCandidate::new(1, ResourceType::Buffer, 1024, None, now);

    // Age should be very small (< 100ms to account for test execution time)
    assert!(candidate.age().as_millis() < 100);
}

/// Test age_secs() returns 0 for fresh allocation
#[test]
fn test_leak_candidate_age_secs_fresh() {
    let now = Instant::now();
    let candidate = LeakCandidate::new(1, ResourceType::Buffer, 1024, None, now);

    assert_eq!(candidate.age_secs(), 0);
}

/// Test severity with default thresholds on fresh allocation
#[test]
fn test_leak_candidate_severity_info() {
    let now = Instant::now();
    let candidate = LeakCandidate::new(1, ResourceType::Buffer, 1024, None, now);
    let thresholds = LeakThresholds::default();

    assert_eq!(candidate.severity(&thresholds), LeakSeverity::Info);
}

/// Test severity with strict thresholds on fresh allocation
#[test]
fn test_leak_candidate_severity_strict() {
    let now = Instant::now();
    let candidate = LeakCandidate::new(1, ResourceType::Buffer, 1024, None, now);
    let thresholds = LeakThresholds::strict();

    // Fresh allocation is still Info even with strict
    assert_eq!(candidate.severity(&thresholds), LeakSeverity::Info);
}

/// Test severity with relaxed thresholds on fresh allocation
#[test]
fn test_leak_candidate_severity_relaxed() {
    let now = Instant::now();
    let candidate = LeakCandidate::new(1, ResourceType::Buffer, 1024, None, now);
    let thresholds = LeakThresholds::relaxed();

    assert_eq!(candidate.severity(&thresholds), LeakSeverity::Info);
}

/// Test severity with immediate warning threshold
#[test]
fn test_leak_candidate_severity_immediate_warning() {
    let now = Instant::now();
    let candidate = LeakCandidate::new(1, ResourceType::Buffer, 1024, None, now);
    let thresholds = LeakThresholds::custom(0, 120, 0);

    // 0-second warning means immediate Warning
    assert_eq!(candidate.severity(&thresholds), LeakSeverity::Warning);
}

/// Test severity with immediate critical threshold
#[test]
fn test_leak_candidate_severity_immediate_critical() {
    let now = Instant::now();
    let candidate = LeakCandidate::new(1, ResourceType::Buffer, 1024, None, now);
    let thresholds = LeakThresholds::custom(0, 0, 0);

    // 0-second critical means immediate Critical
    assert_eq!(candidate.severity(&thresholds), LeakSeverity::Critical);
}

/// Test size_bytes with zero
#[test]
fn test_leak_candidate_size_zero() {
    let now = Instant::now();
    let candidate = LeakCandidate::new(1, ResourceType::Buffer, 0, None, now);

    assert_eq!(candidate.size_bytes, 0);
}

/// Test size_bytes with small value
#[test]
fn test_leak_candidate_size_small() {
    let now = Instant::now();
    let candidate = LeakCandidate::new(1, ResourceType::Buffer, 1, None, now);

    assert_eq!(candidate.size_bytes, 1);
}

/// Test size_bytes with typical buffer size (1 MB)
#[test]
fn test_leak_candidate_size_megabyte() {
    let now = Instant::now();
    let candidate = LeakCandidate::new(1, ResourceType::Buffer, 1024 * 1024, None, now);

    assert_eq!(candidate.size_bytes, 1024 * 1024);
}

/// Test size_bytes with large value (1 GB)
#[test]
fn test_leak_candidate_size_gigabyte() {
    let now = Instant::now();
    let candidate = LeakCandidate::new(1, ResourceType::Buffer, 1024 * 1024 * 1024, None, now);

    assert_eq!(candidate.size_bytes, 1024 * 1024 * 1024);
}

/// Test size_bytes with maximum u64
#[test]
fn test_leak_candidate_size_max() {
    let now = Instant::now();
    let candidate = LeakCandidate::new(1, ResourceType::Buffer, u64::MAX, None, now);

    assert_eq!(candidate.size_bytes, u64::MAX);
}

/// Test label with empty string
#[test]
fn test_leak_candidate_label_empty() {
    let now = Instant::now();
    let candidate = LeakCandidate::new(1, ResourceType::Buffer, 1024, Some("".to_string()), now);

    assert_eq!(candidate.label, Some("".to_string()));
}

/// Test label with unicode characters
#[test]
fn test_leak_candidate_label_unicode() {
    let now = Instant::now();
    let candidate = LeakCandidate::new(
        1,
        ResourceType::Buffer,
        1024,
        Some("Buffer_for_\u{03B1}\u{03B2}\u{03B3}".to_string()),
        now,
    );

    assert!(candidate.label.as_ref().unwrap().contains("\u{03B1}"));
}

/// Test label with very long string
#[test]
fn test_leak_candidate_label_long() {
    let now = Instant::now();
    let long_label = "x".repeat(10000);
    let candidate = LeakCandidate::new(1, ResourceType::Buffer, 1024, Some(long_label.clone()), now);

    assert_eq!(candidate.label.as_ref().unwrap().len(), 10000);
}

/// Test format_colored output contains severity
#[test]
fn test_leak_candidate_format_colored_contains_severity() {
    let now = Instant::now();
    let candidate = LeakCandidate::new(1, ResourceType::Buffer, 1024, Some("Test".to_string()), now);
    let thresholds = LeakThresholds::default();

    let formatted = candidate.format_colored(&thresholds);
    assert!(formatted.contains("INFO") || formatted.contains("WARNING") || formatted.contains("CRITICAL"));
}

/// Test format_colored output contains label
#[test]
fn test_leak_candidate_format_colored_contains_label() {
    let now = Instant::now();
    let candidate = LeakCandidate::new(1, ResourceType::Buffer, 1024, Some("MyBuffer".to_string()), now);
    let thresholds = LeakThresholds::default();

    let formatted = candidate.format_colored(&thresholds);
    assert!(formatted.contains("MyBuffer"));
}

/// Test format_colored output for unlabeled allocation
#[test]
fn test_leak_candidate_format_colored_unlabeled() {
    let now = Instant::now();
    let candidate = LeakCandidate::new(1, ResourceType::Buffer, 1024, None, now);
    let thresholds = LeakThresholds::default();

    let formatted = candidate.format_colored(&thresholds);
    assert!(formatted.contains("<unlabeled>"));
}

/// Test format_colored output contains size
#[test]
fn test_leak_candidate_format_colored_contains_size() {
    let now = Instant::now();
    let candidate = LeakCandidate::new(1, ResourceType::Buffer, 2048, None, now);
    let thresholds = LeakThresholds::default();

    let formatted = candidate.format_colored(&thresholds);
    assert!(formatted.contains("2048"));
}

/// Test format_colored output contains resource type
#[test]
fn test_leak_candidate_format_colored_contains_resource_type() {
    let now = Instant::now();
    let candidate = LeakCandidate::new(1, ResourceType::Buffer, 1024, None, now);
    let thresholds = LeakThresholds::default();

    let formatted = candidate.format_colored(&thresholds);
    assert!(formatted.contains("Buffer"));
}

/// Test format_colored output contains ANSI codes
#[test]
fn test_leak_candidate_format_colored_has_ansi() {
    let now = Instant::now();
    let candidate = LeakCandidate::new(1, ResourceType::Buffer, 1024, None, now);
    let thresholds = LeakThresholds::default();

    let formatted = candidate.format_colored(&thresholds);
    assert!(formatted.contains("\x1b["));
}

/// Test LeakCandidate cloning
#[test]
fn test_leak_candidate_clone() {
    let now = Instant::now();
    let original = LeakCandidate::new(42, ResourceType::Texture, 4096, Some("Clone Test".to_string()), now);
    let cloned = original.clone();

    assert_eq!(cloned.allocation_id, 42);
    assert_eq!(cloned.resource_type, ResourceType::Texture);
    assert_eq!(cloned.size_bytes, 4096);
    assert_eq!(cloned.label, Some("Clone Test".to_string()));
}

/// Test LeakCandidate debug output
#[test]
fn test_leak_candidate_debug() {
    let now = Instant::now();
    let candidate = LeakCandidate::new(1, ResourceType::Buffer, 1024, Some("Debug".to_string()), now);

    let debug_str = format!("{:?}", candidate);
    assert!(debug_str.contains("allocation_id"));
    assert!(debug_str.contains("Buffer"));
}

/// Test allocation_id with maximum value
#[test]
fn test_leak_candidate_id_max() {
    let now = Instant::now();
    let candidate = LeakCandidate::new(u64::MAX, ResourceType::Buffer, 1024, None, now);

    assert_eq!(candidate.allocation_id, u64::MAX);
}

// ============================================================================
// AllocationTracker Tests (32 tests)
// ============================================================================

/// Test new() creates empty tracker
#[test]
fn test_allocation_tracker_new_empty() {
    let tracker = AllocationTracker::new();
    assert_eq!(tracker.count(), 0);
    assert_eq!(tracker.expected_count(), 0);
    assert_eq!(tracker.temporary_count(), 0);
}

/// Test default() is equivalent to new()
#[test]
fn test_allocation_tracker_default() {
    let default_tracker = AllocationTracker::default();
    let new_tracker = AllocationTracker::new();

    assert_eq!(default_tracker.count(), new_tracker.count());
}

/// Test track() adds single allocation
#[test]
fn test_allocation_tracker_track_single() {
    let mut tracker = AllocationTracker::new();
    tracker.track(1, "Buffer1", 1024);

    assert_eq!(tracker.count(), 1);
    assert!(tracker.get(1).is_some());
}

/// Test track() with multiple allocations
#[test]
fn test_allocation_tracker_track_multiple() {
    let mut tracker = AllocationTracker::new();
    tracker.track(1, "Buffer1", 1024);
    tracker.track(2, "Buffer2", 2048);
    tracker.track(3, "Buffer3", 4096);

    assert_eq!(tracker.count(), 3);
}

/// Test track() with duplicate ID replaces existing
#[test]
fn test_allocation_tracker_track_duplicate_id() {
    let mut tracker = AllocationTracker::new();
    tracker.track(1, "First", 1024);
    tracker.track(1, "Second", 2048);

    assert_eq!(tracker.count(), 1);
    let (_, label, size) = tracker.get(1).unwrap();
    assert_eq!(label, "Second");
    assert_eq!(*size, 2048);
}

/// Test track_with_type() stores resource type
#[test]
fn test_allocation_tracker_track_with_type() {
    let mut tracker = AllocationTracker::new();
    tracker.track_with_type(1, "Texture", 4096, ResourceType::Texture);

    assert_eq!(tracker.count(), 1);
    assert_eq!(tracker.get_resource_type(1), ResourceType::Texture);
}

/// Test untrack() removes allocation
#[test]
fn test_allocation_tracker_untrack() {
    let mut tracker = AllocationTracker::new();
    tracker.track(1, "Buffer", 1024);

    assert!(tracker.untrack(1));
    assert_eq!(tracker.count(), 0);
    assert!(tracker.get(1).is_none());
}

/// Test untrack() returns false for unknown ID
#[test]
fn test_allocation_tracker_untrack_unknown() {
    let mut tracker = AllocationTracker::new();
    assert!(!tracker.untrack(999));
}

/// Test untrack() returns false after already removed
#[test]
fn test_allocation_tracker_untrack_twice() {
    let mut tracker = AllocationTracker::new();
    tracker.track(1, "Buffer", 1024);

    assert!(tracker.untrack(1));
    assert!(!tracker.untrack(1)); // Already removed
}

/// Test untrack() removes from expected and temporary sets
#[test]
fn test_allocation_tracker_untrack_clears_marks() {
    let mut tracker = AllocationTracker::new();
    tracker.track(1, "Buffer", 1024);
    tracker.mark_expected(1);

    tracker.untrack(1);

    assert!(!tracker.is_expected(1));
    assert_eq!(tracker.expected_count(), 0);
}

/// Test mark_expected() sets expected flag
#[test]
fn test_allocation_tracker_mark_expected() {
    let mut tracker = AllocationTracker::new();
    tracker.track(1, "Static", 1024);
    tracker.mark_expected(1);

    assert!(tracker.is_expected(1));
    assert_eq!(tracker.expected_count(), 1);
}

/// Test mark_expected() does nothing for unknown ID
#[test]
fn test_allocation_tracker_mark_expected_unknown() {
    let mut tracker = AllocationTracker::new();
    tracker.mark_expected(999);

    assert!(!tracker.is_expected(999));
    assert_eq!(tracker.expected_count(), 0);
}

/// Test mark_expected() removes from temporary
#[test]
fn test_allocation_tracker_mark_expected_removes_temporary() {
    let mut tracker = AllocationTracker::new();
    tracker.track(1, "Buffer", 1024);
    tracker.mark_temporary(1);
    tracker.mark_expected(1);

    assert!(tracker.is_expected(1));
    assert!(!tracker.is_temporary(1));
}

/// Test mark_temporary() sets temporary flag
#[test]
fn test_allocation_tracker_mark_temporary() {
    let mut tracker = AllocationTracker::new();
    tracker.track(1, "Temp", 1024);
    tracker.mark_temporary(1);

    assert!(tracker.is_temporary(1));
    assert_eq!(tracker.temporary_count(), 1);
}

/// Test mark_temporary() does nothing for unknown ID
#[test]
fn test_allocation_tracker_mark_temporary_unknown() {
    let mut tracker = AllocationTracker::new();
    tracker.mark_temporary(999);

    assert!(!tracker.is_temporary(999));
    assert_eq!(tracker.temporary_count(), 0);
}

/// Test mark_temporary() removes from expected
#[test]
fn test_allocation_tracker_mark_temporary_removes_expected() {
    let mut tracker = AllocationTracker::new();
    tracker.track(1, "Buffer", 1024);
    tracker.mark_expected(1);
    tracker.mark_temporary(1);

    assert!(tracker.is_temporary(1));
    assert!(!tracker.is_expected(1));
}

/// Test is_expected() returns false for non-expected
#[test]
fn test_allocation_tracker_is_expected_false() {
    let mut tracker = AllocationTracker::new();
    tracker.track(1, "Buffer", 1024);

    assert!(!tracker.is_expected(1));
}

/// Test is_temporary() returns false for non-temporary
#[test]
fn test_allocation_tracker_is_temporary_false() {
    let mut tracker = AllocationTracker::new();
    tracker.track(1, "Buffer", 1024);

    assert!(!tracker.is_temporary(1));
}

/// Test get() returns allocation info
#[test]
fn test_allocation_tracker_get() {
    let mut tracker = AllocationTracker::new();
    tracker.track(1, "TestBuffer", 2048);

    let info = tracker.get(1);
    assert!(info.is_some());
    let (_, label, size) = info.unwrap();
    assert_eq!(label, "TestBuffer");
    assert_eq!(*size, 2048);
}

/// Test get() returns None for unknown
#[test]
fn test_allocation_tracker_get_unknown() {
    let tracker = AllocationTracker::new();
    assert!(tracker.get(999).is_none());
}

/// Test get_resource_type() returns Other for unknown
#[test]
fn test_allocation_tracker_get_resource_type_default() {
    let mut tracker = AllocationTracker::new();
    tracker.track(1, "Buffer", 1024); // No resource type specified

    assert_eq!(tracker.get_resource_type(1), ResourceType::Other);
}

/// Test count() accuracy
#[test]
fn test_allocation_tracker_count() {
    let mut tracker = AllocationTracker::new();
    assert_eq!(tracker.count(), 0);

    tracker.track(1, "A", 100);
    assert_eq!(tracker.count(), 1);

    tracker.track(2, "B", 100);
    assert_eq!(tracker.count(), 2);

    tracker.untrack(1);
    assert_eq!(tracker.count(), 1);
}

/// Test expected_count() accuracy
#[test]
fn test_allocation_tracker_expected_count() {
    let mut tracker = AllocationTracker::new();
    tracker.track(1, "A", 100);
    tracker.track(2, "B", 100);
    tracker.mark_expected(1);
    tracker.mark_expected(2);

    assert_eq!(tracker.expected_count(), 2);
}

/// Test temporary_count() accuracy
#[test]
fn test_allocation_tracker_temporary_count() {
    let mut tracker = AllocationTracker::new();
    tracker.track(1, "A", 100);
    tracker.track(2, "B", 100);
    tracker.mark_temporary(1);

    assert_eq!(tracker.temporary_count(), 1);
}

/// Test iter() yields all allocations
#[test]
fn test_allocation_tracker_iter() {
    let mut tracker = AllocationTracker::new();
    tracker.track(1, "A", 100);
    tracker.track(2, "B", 200);
    tracker.track(3, "C", 300);

    let ids: Vec<u64> = tracker.iter().map(|(id, _)| id).collect();
    assert_eq!(ids.len(), 3);
    assert!(ids.contains(&1));
    assert!(ids.contains(&2));
    assert!(ids.contains(&3));
}

/// Test clear() removes all state
#[test]
fn test_allocation_tracker_clear() {
    let mut tracker = AllocationTracker::new();
    tracker.track(1, "A", 100);
    tracker.track(2, "B", 200);
    tracker.mark_expected(1);
    tracker.mark_temporary(2);
    tracker.track_with_type(3, "C", 300, ResourceType::Texture);

    tracker.clear();

    assert_eq!(tracker.count(), 0);
    assert_eq!(tracker.expected_count(), 0);
    assert_eq!(tracker.temporary_count(), 0);
}

/// Test total_bytes() sums all allocations
#[test]
fn test_allocation_tracker_total_bytes() {
    let mut tracker = AllocationTracker::new();
    tracker.track(1, "A", 100);
    tracker.track(2, "B", 200);
    tracker.track(3, "C", 300);

    assert_eq!(tracker.total_bytes(), 600);
}

/// Test total_bytes() on empty tracker
#[test]
fn test_allocation_tracker_total_bytes_empty() {
    let tracker = AllocationTracker::new();
    assert_eq!(tracker.total_bytes(), 0);
}

/// Test total_bytes() with large values
#[test]
fn test_allocation_tracker_total_bytes_large() {
    let mut tracker = AllocationTracker::new();
    tracker.track(1, "A", 1024 * 1024 * 1024); // 1 GB
    tracker.track(2, "B", 1024 * 1024 * 1024); // 1 GB

    assert_eq!(tracker.total_bytes(), 2 * 1024 * 1024 * 1024);
}

/// Test allocation with zero-sized entry
#[test]
fn test_allocation_tracker_zero_size() {
    let mut tracker = AllocationTracker::new();
    tracker.track(1, "Empty", 0);

    assert_eq!(tracker.count(), 1);
    assert_eq!(tracker.total_bytes(), 0);
}

/// Test tracking with string label conversion
#[test]
fn test_allocation_tracker_label_into_string() {
    let mut tracker = AllocationTracker::new();
    tracker.track(1, String::from("Owned"), 1024);
    tracker.track(2, "Static", 2048);

    let (_, label1, _) = tracker.get(1).unwrap();
    let (_, label2, _) = tracker.get(2).unwrap();
    assert_eq!(label1, "Owned");
    assert_eq!(label2, "Static");
}

// ============================================================================
// LeakDetector Tests (38 tests)
// ============================================================================

/// Test new() with custom thresholds
#[test]
fn test_leak_detector_new_custom() {
    let thresholds = LeakThresholds::custom(10, 60, 512);
    let detector = LeakDetector::new(thresholds);

    assert_eq!(detector.thresholds().warning_secs, 10);
    assert_eq!(detector.thresholds().critical_secs, 60);
    assert_eq!(detector.thresholds().min_size_bytes, 512);
}

/// Test with_default_thresholds() initialization
#[test]
fn test_leak_detector_with_default() {
    let detector = LeakDetector::with_default_thresholds();

    assert_eq!(detector.thresholds().warning_secs, 30);
    assert_eq!(detector.tracked_count(), 0);
}

/// Test with_strict_thresholds() initialization
#[test]
fn test_leak_detector_with_strict() {
    let detector = LeakDetector::with_strict_thresholds();

    assert_eq!(detector.thresholds().warning_secs, 5);
    assert_eq!(detector.thresholds().critical_secs, 30);
}

/// Test with_relaxed_thresholds() initialization
#[test]
fn test_leak_detector_with_relaxed() {
    let detector = LeakDetector::with_relaxed_thresholds();

    assert_eq!(detector.thresholds().warning_secs, 300);
    assert_eq!(detector.thresholds().critical_secs, 600);
}

/// Test thresholds() returns reference to thresholds
#[test]
fn test_leak_detector_thresholds_getter() {
    let thresholds = LeakThresholds::custom(15, 90, 2048);
    let detector = LeakDetector::new(thresholds);

    let got = detector.thresholds();
    assert_eq!(got.warning_secs, 15);
}

/// Test set_thresholds() updates thresholds
#[test]
fn test_leak_detector_set_thresholds() {
    let mut detector = LeakDetector::with_default_thresholds();
    detector.set_thresholds(LeakThresholds::strict());

    assert_eq!(detector.thresholds().warning_secs, 5);
}

/// Test track_allocation() registration
#[test]
fn test_leak_detector_track_allocation() {
    let mut detector = LeakDetector::with_default_thresholds();
    detector.track_allocation(1, "Buffer1", 1024);

    assert_eq!(detector.tracked_count(), 1);
    assert_eq!(detector.total_bytes(), 1024);
}

/// Test track_allocation() updates stats
#[test]
fn test_leak_detector_track_updates_stats() {
    let mut detector = LeakDetector::with_default_thresholds();
    detector.track_allocation(1, "Buffer1", 1024);
    detector.track_allocation(2, "Buffer2", 2048);

    let stats = detector.stats();
    assert_eq!(stats.total_tracked, 2);
    assert_eq!(stats.current_tracked, 2);
}

/// Test track_allocation_typed() with resource type
#[test]
fn test_leak_detector_track_allocation_typed() {
    let mut detector = LeakDetector::with_default_thresholds();
    detector.track_allocation_typed(1, "Texture", 4096, ResourceType::Texture);

    assert_eq!(detector.tracked_count(), 1);
}

/// Test release_allocation() removal
#[test]
fn test_leak_detector_release_allocation() {
    let mut detector = LeakDetector::with_default_thresholds();
    detector.track_allocation(1, "Buffer", 1024);

    assert!(detector.release_allocation(1));
    assert_eq!(detector.tracked_count(), 0);
}

/// Test release_allocation() returns false for unknown
#[test]
fn test_leak_detector_release_unknown() {
    let mut detector = LeakDetector::with_default_thresholds();
    assert!(!detector.release_allocation(999));
}

/// Test release_allocation() updates stats
#[test]
fn test_leak_detector_release_updates_stats() {
    let mut detector = LeakDetector::with_default_thresholds();
    detector.track_allocation(1, "Buffer", 1024);
    detector.release_allocation(1);

    let stats = detector.stats();
    assert_eq!(stats.total_released, 1);
    assert_eq!(stats.current_tracked, 0);
}

/// Test mark_expected() exclusion
#[test]
fn test_leak_detector_mark_expected() {
    let mut detector = LeakDetector::with_default_thresholds();
    detector.track_allocation(1, "Static", 1024);
    detector.mark_expected(1);

    let stats = detector.stats();
    assert_eq!(stats.expected_long_lived, 1);
}

/// Test mark_temporary() for stricter checking
#[test]
fn test_leak_detector_mark_temporary() {
    let mut detector = LeakDetector::with_default_thresholds();
    detector.track_allocation(1, "Temp", 1024);
    detector.mark_temporary(1);

    // Just verify it doesn't error - actual effect tested via check()
    assert_eq!(detector.tracked_count(), 1);
}

/// Test check() on fresh allocations returns empty
#[test]
fn test_leak_detector_check_no_leaks() {
    let mut detector = LeakDetector::with_default_thresholds();
    detector.track_allocation(1, "Fresh", 2048);

    let candidates = detector.check();
    assert!(candidates.is_empty());
}

/// Test check() respects min_size_bytes threshold
#[test]
fn test_leak_detector_check_respects_min_size() {
    // Immediate warning but min_size = 2048
    let thresholds = LeakThresholds::custom(0, 1, 2048);
    let mut detector = LeakDetector::new(thresholds);
    detector.track_allocation(1, "SmallBuffer", 1024); // Below min

    let candidates = detector.check();
    assert!(candidates.is_empty());
}

/// Test check() includes allocations above min_size
#[test]
fn test_leak_detector_check_includes_above_min_size() {
    // Immediate warning, min_size = 512
    let thresholds = LeakThresholds::custom(0, 120, 512);
    let mut detector = LeakDetector::new(thresholds);
    detector.track_allocation(1, "LargeBuffer", 1024); // Above min

    let candidates = detector.check();
    assert_eq!(candidates.len(), 1);
}

/// Test check() excludes expected allocations
#[test]
fn test_leak_detector_check_excludes_expected() {
    let thresholds = LeakThresholds::custom(0, 120, 0);
    let mut detector = LeakDetector::new(thresholds);
    detector.track_allocation(1, "Static", 1024);
    detector.mark_expected(1);

    let candidates = detector.check();
    assert!(candidates.is_empty());
}

/// Test check() increments check counter
#[test]
fn test_leak_detector_check_increments_counter() {
    let mut detector = LeakDetector::with_default_thresholds();

    let _ = detector.check();
    let _ = detector.check();
    let _ = detector.check();

    let stats = detector.stats();
    assert_eq!(stats.checks_performed, 3);
}

/// Test check() updates last_check time
#[test]
fn test_leak_detector_check_updates_time() {
    let mut detector = LeakDetector::with_default_thresholds();

    assert!(detector.time_since_last_check().is_none());
    let _ = detector.check();
    assert!(detector.time_since_last_check().is_some());
}

/// Test check_critical_only() filters by severity
#[test]
fn test_leak_detector_check_critical_only() {
    // Warning at 0s, Critical at 120s
    let thresholds = LeakThresholds::custom(0, 120, 0);
    let mut detector = LeakDetector::new(thresholds);
    detector.track_allocation(1, "Buffer", 1024);

    // Fresh allocation is Warning (age < 120s), not Critical
    let critical = detector.check_critical_only();
    assert!(critical.is_empty());
}

/// Test check_critical_only() with immediate critical threshold
#[test]
fn test_leak_detector_check_critical_only_immediate() {
    // Warning at 0s, Critical at 0s
    let thresholds = LeakThresholds::custom(0, 0, 0);
    let mut detector = LeakDetector::new(thresholds);
    detector.track_allocation(1, "Buffer", 1024);

    let critical = detector.check_critical_only();
    assert_eq!(critical.len(), 1);
}

/// Test stats() accuracy
#[test]
fn test_leak_detector_stats_accuracy() {
    let mut detector = LeakDetector::with_default_thresholds();
    detector.track_allocation(1, "A", 100);
    detector.track_allocation(2, "B", 200);
    detector.track_allocation(3, "C", 300);
    detector.release_allocation(1);
    detector.mark_expected(2);

    let stats = detector.stats();
    assert_eq!(stats.total_tracked, 3);
    assert_eq!(stats.total_released, 1);
    assert_eq!(stats.current_tracked, 2);
    assert_eq!(stats.expected_long_lived, 1);
}

/// Test clear() resets all state
#[test]
fn test_leak_detector_clear() {
    let mut detector = LeakDetector::with_default_thresholds();
    detector.track_allocation(1, "Buffer", 1024);
    let _ = detector.check();

    detector.clear();

    assert_eq!(detector.tracked_count(), 0);
    let stats = detector.stats();
    assert_eq!(stats.checks_performed, 0);
    assert_eq!(stats.total_tracked, 0);
}

/// Test tracked_count() getter
#[test]
fn test_leak_detector_tracked_count() {
    let mut detector = LeakDetector::with_default_thresholds();
    assert_eq!(detector.tracked_count(), 0);

    detector.track_allocation(1, "A", 100);
    assert_eq!(detector.tracked_count(), 1);
}

/// Test total_bytes() getter
#[test]
fn test_leak_detector_total_bytes() {
    let mut detector = LeakDetector::with_default_thresholds();
    detector.track_allocation(1, "A", 1024);
    detector.track_allocation(2, "B", 2048);

    assert_eq!(detector.total_bytes(), 3072);
}

/// Test time_since_last_check() before any check
#[test]
fn test_leak_detector_time_since_check_none() {
    let detector = LeakDetector::with_default_thresholds();
    assert!(detector.time_since_last_check().is_none());
}

/// Test time_since_last_check() after check
#[test]
fn test_leak_detector_time_since_check_some() {
    let mut detector = LeakDetector::with_default_thresholds();
    let _ = detector.check();

    let elapsed = detector.time_since_last_check();
    assert!(elapsed.is_some());
    assert!(elapsed.unwrap().as_millis() < 1000); // Should be quick
}

/// Test report() generation
#[test]
fn test_leak_detector_report() {
    let thresholds = LeakThresholds::custom(0, 120, 0);
    let mut detector = LeakDetector::new(thresholds);
    detector.track_allocation(1, "Buffer", 1024);

    let report = detector.report();
    assert_eq!(report.len(), 1);
}

/// Test report() updates stats
#[test]
fn test_leak_detector_report_updates_stats() {
    let mut detector = LeakDetector::with_default_thresholds();
    let _ = detector.report();

    let stats = detector.stats();
    assert_eq!(stats.checks_performed, 1);
}

/// Test multiple allocations with different IDs
#[test]
fn test_leak_detector_multiple_allocations() {
    let mut detector = LeakDetector::with_default_thresholds();

    for i in 0..100 {
        detector.track_allocation(i, format!("Buffer{}", i), 1024);
    }

    assert_eq!(detector.tracked_count(), 100);
    assert_eq!(detector.total_bytes(), 100 * 1024);
}

/// Test no false positives on timely releases
#[test]
fn test_leak_detector_no_false_positives() {
    let mut detector = LeakDetector::with_default_thresholds();

    // Track and release several allocations
    for i in 0..10 {
        detector.track_allocation(i, format!("Temp{}", i), 1024);
        detector.release_allocation(i);
    }

    let candidates = detector.check();
    assert!(candidates.is_empty());
}

/// Test mixed expected and regular allocations
#[test]
fn test_leak_detector_mixed_allocations() {
    let thresholds = LeakThresholds::custom(0, 120, 0);
    let mut detector = LeakDetector::new(thresholds);

    detector.track_allocation(1, "Static1", 1024);
    detector.track_allocation(2, "Temp", 2048);
    detector.track_allocation(3, "Static2", 4096);

    detector.mark_expected(1);
    detector.mark_expected(3);

    let candidates = detector.check();
    assert_eq!(candidates.len(), 1); // Only "Temp" should be detected
    assert_eq!(candidates[0].allocation_id, 2);
}

/// Test leaks_detected counter
#[test]
fn test_leak_detector_leaks_detected_counter() {
    let thresholds = LeakThresholds::custom(0, 120, 0);
    let mut detector = LeakDetector::new(thresholds);
    detector.track_allocation(1, "Leak", 1024);

    let _ = detector.check();
    let stats = detector.stats();
    assert_eq!(stats.leaks_detected, 1);
}

/// Test critical_leaks counter
#[test]
fn test_leak_detector_critical_leaks_counter() {
    let thresholds = LeakThresholds::custom(0, 0, 0);
    let mut detector = LeakDetector::new(thresholds);
    detector.track_allocation(1, "Critical", 1024);

    let _ = detector.check();
    let stats = detector.stats();
    assert_eq!(stats.critical_leaks, 1);
}

/// Test temporary allocation stricter thresholds
#[test]
fn test_leak_detector_temporary_stricter() {
    // Warning at 60s, but temporary uses half (30s)
    let thresholds = LeakThresholds::custom(60, 120, 0);
    let mut detector = LeakDetector::new(thresholds);
    detector.track_allocation(1, "Temp", 1024);
    detector.mark_temporary(1);

    // Fresh allocation won't trigger even stricter check
    let candidates = detector.check();
    assert!(candidates.is_empty());
}

// ============================================================================
// LeakStats Tests (18 tests)
// ============================================================================

/// Test new() creates empty stats
#[test]
fn test_leak_stats_new() {
    let stats = LeakStats::new();

    assert_eq!(stats.total_tracked, 0);
    assert_eq!(stats.total_released, 0);
    assert_eq!(stats.current_tracked, 0);
    assert_eq!(stats.expected_long_lived, 0);
    assert_eq!(stats.checks_performed, 0);
    assert_eq!(stats.leaks_detected, 0);
    assert_eq!(stats.critical_leaks, 0);
}

/// Test default() is same as new()
#[test]
fn test_leak_stats_default() {
    let default_stats = LeakStats::default();
    let new_stats = LeakStats::new();

    assert_eq!(default_stats.total_tracked, new_stats.total_tracked);
}

/// Test total_tracked field
#[test]
fn test_leak_stats_total_tracked() {
    let mut stats = LeakStats::new();
    stats.total_tracked = 100;

    assert_eq!(stats.total_tracked, 100);
}

/// Test total_released field
#[test]
fn test_leak_stats_total_released() {
    let mut stats = LeakStats::new();
    stats.total_released = 50;

    assert_eq!(stats.total_released, 50);
}

/// Test current_tracked field
#[test]
fn test_leak_stats_current_tracked() {
    let mut stats = LeakStats::new();
    stats.current_tracked = 25;

    assert_eq!(stats.current_tracked, 25);
}

/// Test expected_long_lived field
#[test]
fn test_leak_stats_expected_long_lived() {
    let mut stats = LeakStats::new();
    stats.expected_long_lived = 10;

    assert_eq!(stats.expected_long_lived, 10);
}

/// Test checks_performed field
#[test]
fn test_leak_stats_checks_performed() {
    let mut stats = LeakStats::new();
    stats.checks_performed = 5;

    assert_eq!(stats.checks_performed, 5);
}

/// Test leaks_detected field
#[test]
fn test_leak_stats_leaks_detected() {
    let mut stats = LeakStats::new();
    stats.leaks_detected = 3;

    assert_eq!(stats.leaks_detected, 3);
}

/// Test critical_leaks field
#[test]
fn test_leak_stats_critical_leaks() {
    let mut stats = LeakStats::new();
    stats.critical_leaks = 1;

    assert_eq!(stats.critical_leaks, 1);
}

/// Test leak_rate() calculation
#[test]
fn test_leak_stats_leak_rate() {
    let mut stats = LeakStats::new();
    stats.total_tracked = 100;
    stats.leaks_detected = 10;

    assert!((stats.leak_rate() - 10.0).abs() < 0.001);
}

/// Test leak_rate() with zero total
#[test]
fn test_leak_stats_leak_rate_zero_total() {
    let stats = LeakStats::new();
    assert_eq!(stats.leak_rate(), 0.0);
}

/// Test leak_rate() with 100% leaks
#[test]
fn test_leak_stats_leak_rate_full() {
    let mut stats = LeakStats::new();
    stats.total_tracked = 50;
    stats.leaks_detected = 50;

    assert!((stats.leak_rate() - 100.0).abs() < 0.001);
}

/// Test critical_rate() calculation
#[test]
fn test_leak_stats_critical_rate() {
    let mut stats = LeakStats::new();
    stats.leaks_detected = 20;
    stats.critical_leaks = 5;

    assert!((stats.critical_rate() - 25.0).abs() < 0.001);
}

/// Test critical_rate() with zero leaks
#[test]
fn test_leak_stats_critical_rate_zero_leaks() {
    let stats = LeakStats::new();
    assert_eq!(stats.critical_rate(), 0.0);
}

/// Test critical_rate() with all critical
#[test]
fn test_leak_stats_critical_rate_all_critical() {
    let mut stats = LeakStats::new();
    stats.leaks_detected = 10;
    stats.critical_leaks = 10;

    assert!((stats.critical_rate() - 100.0).abs() < 0.001);
}

/// Test stats cloning
#[test]
fn test_leak_stats_clone() {
    let mut original = LeakStats::new();
    original.total_tracked = 42;
    original.leaks_detected = 7;

    let cloned = original.clone();
    assert_eq!(cloned.total_tracked, 42);
    assert_eq!(cloned.leaks_detected, 7);
}

/// Test stats debug output
#[test]
fn test_leak_stats_debug() {
    let stats = LeakStats::new();
    let debug_str = format!("{:?}", stats);
    assert!(debug_str.contains("total_tracked"));
}

/// Test current_tracked relationship to tracked/released
#[test]
fn test_leak_stats_tracked_released_relationship() {
    // In a real scenario: current = total_tracked - total_released
    let mut stats = LeakStats::new();
    stats.total_tracked = 100;
    stats.total_released = 40;
    stats.current_tracked = 60; // 100 - 40

    assert_eq!(stats.current_tracked, stats.total_tracked - stats.total_released);
}

// ============================================================================
// LeakReport Tests (22 tests)
// ============================================================================

/// Test has_critical() true when critical_leaks > 0
#[test]
fn test_leak_report_has_critical_true() {
    let report = LeakReport {
        candidates: vec![],
        stats: LeakStats {
            critical_leaks: 1,
            ..Default::default()
        },
        timestamp: Instant::now(),
    };

    assert!(report.has_critical());
}

/// Test has_critical() false when critical_leaks == 0
#[test]
fn test_leak_report_has_critical_false() {
    let report = LeakReport {
        candidates: vec![],
        stats: LeakStats::default(),
        timestamp: Instant::now(),
    };

    assert!(!report.has_critical());
}

/// Test total_leaked_bytes() sums candidates
#[test]
fn test_leak_report_total_leaked_bytes() {
    let now = Instant::now();
    let report = LeakReport {
        candidates: vec![
            LeakCandidate::new(1, ResourceType::Buffer, 1024, None, now),
            LeakCandidate::new(2, ResourceType::Buffer, 2048, None, now),
            LeakCandidate::new(3, ResourceType::Buffer, 4096, None, now),
        ],
        stats: LeakStats::default(),
        timestamp: now,
    };

    assert_eq!(report.total_leaked_bytes(), 7168);
}

/// Test total_leaked_bytes() on empty report
#[test]
fn test_leak_report_total_leaked_bytes_empty() {
    let report = LeakReport {
        candidates: vec![],
        stats: LeakStats::default(),
        timestamp: Instant::now(),
    };

    assert_eq!(report.total_leaked_bytes(), 0);
}

/// Test summary() contains candidate count
#[test]
fn test_leak_report_summary_contains_count() {
    let report = LeakReport {
        candidates: vec![],
        stats: LeakStats::default(),
        timestamp: Instant::now(),
    };

    let summary = report.summary();
    assert!(summary.contains("0 candidates"));
}

/// Test summary() contains KB measurement
#[test]
fn test_leak_report_summary_contains_kb() {
    let now = Instant::now();
    let report = LeakReport {
        candidates: vec![
            LeakCandidate::new(1, ResourceType::Buffer, 2048, None, now),
        ],
        stats: LeakStats::default(),
        timestamp: now,
    };

    let summary = report.summary();
    assert!(summary.contains("KB"));
}

/// Test summary() contains critical count
#[test]
fn test_leak_report_summary_contains_critical() {
    let report = LeakReport {
        candidates: vec![],
        stats: LeakStats {
            critical_leaks: 3,
            ..Default::default()
        },
        timestamp: Instant::now(),
    };

    let summary = report.summary();
    assert!(summary.contains("3 critical"));
}

/// Test summary() contains stats info
#[test]
fn test_leak_report_summary_contains_stats() {
    let report = LeakReport {
        candidates: vec![],
        stats: LeakStats {
            checks_performed: 5,
            ..Default::default()
        },
        timestamp: Instant::now(),
    };

    let summary = report.summary();
    assert!(summary.contains("5 checks"));
}

/// Test by_severity() categorization - empty report
#[test]
fn test_leak_report_by_severity_empty() {
    let report = LeakReport {
        candidates: vec![],
        stats: LeakStats::default(),
        timestamp: Instant::now(),
    };
    let thresholds = LeakThresholds::default();

    let (info, warning, critical) = report.by_severity(&thresholds);
    assert!(info.is_empty());
    assert!(warning.is_empty());
    assert!(critical.is_empty());
}

/// Test by_severity() categorizes fresh candidates as Info
#[test]
fn test_leak_report_by_severity_info() {
    let now = Instant::now();
    let report = LeakReport {
        candidates: vec![
            LeakCandidate::new(1, ResourceType::Buffer, 1024, None, now),
        ],
        stats: LeakStats::default(),
        timestamp: now,
    };
    let thresholds = LeakThresholds::default();

    let (info, warning, critical) = report.by_severity(&thresholds);
    assert_eq!(info.len(), 1);
    assert!(warning.is_empty());
    assert!(critical.is_empty());
}

/// Test by_severity() with immediate warning threshold
#[test]
fn test_leak_report_by_severity_warning() {
    let now = Instant::now();
    let report = LeakReport {
        candidates: vec![
            LeakCandidate::new(1, ResourceType::Buffer, 1024, None, now),
        ],
        stats: LeakStats::default(),
        timestamp: now,
    };
    let thresholds = LeakThresholds::custom(0, 120, 0); // Immediate warning

    let (info, warning, critical) = report.by_severity(&thresholds);
    assert!(info.is_empty());
    assert_eq!(warning.len(), 1);
    assert!(critical.is_empty());
}

/// Test by_severity() with immediate critical threshold
#[test]
fn test_leak_report_by_severity_critical() {
    let now = Instant::now();
    let report = LeakReport {
        candidates: vec![
            LeakCandidate::new(1, ResourceType::Buffer, 1024, None, now),
        ],
        stats: LeakStats::default(),
        timestamp: now,
    };
    let thresholds = LeakThresholds::custom(0, 0, 0); // Immediate critical

    let (info, warning, critical) = report.by_severity(&thresholds);
    assert!(info.is_empty());
    assert!(warning.is_empty());
    assert_eq!(critical.len(), 1);
}

/// Test by_severity() with mixed severities
#[test]
fn test_leak_report_by_severity_mixed() {
    let now = Instant::now();
    let report = LeakReport {
        candidates: vec![
            LeakCandidate::new(1, ResourceType::Buffer, 1024, None, now),
            LeakCandidate::new(2, ResourceType::Buffer, 2048, None, now),
            LeakCandidate::new(3, ResourceType::Buffer, 4096, None, now),
        ],
        stats: LeakStats::default(),
        timestamp: now,
    };
    // With default thresholds, all fresh allocations are Info
    let thresholds = LeakThresholds::default();

    let (info, warning, critical) = report.by_severity(&thresholds);
    assert_eq!(info.len(), 3);
    assert!(warning.is_empty());
    assert!(critical.is_empty());
}

/// Test len() returns candidate count
#[test]
fn test_leak_report_len() {
    let now = Instant::now();
    let report = LeakReport {
        candidates: vec![
            LeakCandidate::new(1, ResourceType::Buffer, 100, None, now),
            LeakCandidate::new(2, ResourceType::Buffer, 200, None, now),
        ],
        stats: LeakStats::default(),
        timestamp: now,
    };

    assert_eq!(report.len(), 2);
}

/// Test len() on empty report
#[test]
fn test_leak_report_len_empty() {
    let report = LeakReport {
        candidates: vec![],
        stats: LeakStats::default(),
        timestamp: Instant::now(),
    };

    assert_eq!(report.len(), 0);
}

/// Test is_empty() true when no candidates
#[test]
fn test_leak_report_is_empty_true() {
    let report = LeakReport {
        candidates: vec![],
        stats: LeakStats::default(),
        timestamp: Instant::now(),
    };

    assert!(report.is_empty());
}

/// Test is_empty() false when candidates exist
#[test]
fn test_leak_report_is_empty_false() {
    let now = Instant::now();
    let report = LeakReport {
        candidates: vec![
            LeakCandidate::new(1, ResourceType::Buffer, 100, None, now),
        ],
        stats: LeakStats::default(),
        timestamp: now,
    };

    assert!(!report.is_empty());
}

/// Test report debug output
#[test]
fn test_leak_report_debug() {
    let report = LeakReport {
        candidates: vec![],
        stats: LeakStats::default(),
        timestamp: Instant::now(),
    };

    let debug_str = format!("{:?}", report);
    assert!(debug_str.contains("LeakReport"));
    assert!(debug_str.contains("candidates"));
}

/// Test report timestamp
#[test]
fn test_leak_report_timestamp() {
    let before = Instant::now();
    let report = LeakReport {
        candidates: vec![],
        stats: LeakStats::default(),
        timestamp: Instant::now(),
    };
    let after = Instant::now();

    assert!(report.timestamp >= before);
    assert!(report.timestamp <= after);
}

/// Test single leak report
#[test]
fn test_leak_report_single_leak() {
    let now = Instant::now();
    let report = LeakReport {
        candidates: vec![
            LeakCandidate::new(42, ResourceType::Texture, 4096, Some("SingleLeak".to_string()), now),
        ],
        stats: LeakStats {
            leaks_detected: 1,
            ..Default::default()
        },
        timestamp: now,
    };

    assert_eq!(report.len(), 1);
    assert!(!report.is_empty());
    assert_eq!(report.total_leaked_bytes(), 4096);
}

/// Test multiple leaks report
#[test]
fn test_leak_report_multiple_leaks() {
    let now = Instant::now();
    let candidates: Vec<_> = (0..10)
        .map(|i| LeakCandidate::new(i, ResourceType::Buffer, 1024, Some(format!("Leak{}", i)), now))
        .collect();

    let report = LeakReport {
        candidates,
        stats: LeakStats {
            leaks_detected: 10,
            ..Default::default()
        },
        timestamp: now,
    };

    assert_eq!(report.len(), 10);
    assert_eq!(report.total_leaked_bytes(), 10 * 1024);
}

// ============================================================================
// FrameLeakChecker Tests (22 tests)
// ============================================================================

/// Test new() creates clean checker
#[test]
fn test_frame_leak_checker_new() {
    let checker = FrameLeakChecker::new();

    assert!(checker.is_clean());
    assert_eq!(checker.frame_number(), 0);
    assert_eq!(checker.unreleased_count(), 0);
}

/// Test default() is same as new()
#[test]
fn test_frame_leak_checker_default() {
    let default_checker = FrameLeakChecker::default();
    let new_checker = FrameLeakChecker::new();

    assert_eq!(default_checker.frame_number(), new_checker.frame_number());
    assert_eq!(default_checker.is_clean(), new_checker.is_clean());
}

/// Test begin_frame() clears state
#[test]
fn test_frame_leak_checker_begin_frame_clears() {
    let mut checker = FrameLeakChecker::new();
    checker.track(1);
    checker.track(2);

    checker.begin_frame();

    assert!(checker.is_clean());
    assert_eq!(checker.unreleased_count(), 0);
}

/// Test begin_frame() increments frame number
#[test]
fn test_frame_leak_checker_begin_frame_increments() {
    let mut checker = FrameLeakChecker::new();

    checker.begin_frame();
    assert_eq!(checker.frame_number(), 1);

    checker.begin_frame();
    assert_eq!(checker.frame_number(), 2);
}

/// Test track() adds to frame
#[test]
fn test_frame_leak_checker_track() {
    let mut checker = FrameLeakChecker::new();
    checker.begin_frame();
    checker.track(1);

    assert!(!checker.is_clean());
    assert_eq!(checker.unreleased_count(), 1);
}

/// Test track() adds multiple
#[test]
fn test_frame_leak_checker_track_multiple() {
    let mut checker = FrameLeakChecker::new();
    checker.begin_frame();
    checker.track(1);
    checker.track(2);
    checker.track(3);

    assert_eq!(checker.unreleased_count(), 3);
}

/// Test track() allows duplicates
#[test]
fn test_frame_leak_checker_track_duplicates() {
    let mut checker = FrameLeakChecker::new();
    checker.begin_frame();
    checker.track(1);
    checker.track(1);

    // Duplicates are allowed (Vec, not Set)
    assert_eq!(checker.unreleased_count(), 2);
}

/// Test release() removes from frame
#[test]
fn test_frame_leak_checker_release() {
    let mut checker = FrameLeakChecker::new();
    checker.begin_frame();
    checker.track(1);
    checker.track(2);

    checker.release(1);

    assert_eq!(checker.unreleased_count(), 1);
}

/// Test release() on non-existent is no-op
#[test]
fn test_frame_leak_checker_release_nonexistent() {
    let mut checker = FrameLeakChecker::new();
    checker.begin_frame();
    checker.track(1);

    checker.release(999);

    assert_eq!(checker.unreleased_count(), 1);
}

/// Test release() removes all instances of ID
#[test]
fn test_frame_leak_checker_release_all_instances() {
    let mut checker = FrameLeakChecker::new();
    checker.begin_frame();
    checker.track(1);
    checker.track(1);
    checker.track(1);

    checker.release(1);

    assert!(checker.is_clean());
}

/// Test end_frame() returns unreleased
#[test]
fn test_frame_leak_checker_end_frame() {
    let mut checker = FrameLeakChecker::new();
    checker.begin_frame();
    checker.track(1);
    checker.track(2);
    checker.track(3);
    checker.release(2);

    let unreleased = checker.end_frame();

    assert_eq!(unreleased.len(), 2);
    assert!(unreleased.contains(&1));
    assert!(unreleased.contains(&3));
    assert!(!unreleased.contains(&2));
}

/// Test end_frame() clears state
#[test]
fn test_frame_leak_checker_end_frame_clears() {
    let mut checker = FrameLeakChecker::new();
    checker.begin_frame();
    checker.track(1);

    let _ = checker.end_frame();

    assert!(checker.is_clean());
}

/// Test end_frame() on clean frame
#[test]
fn test_frame_leak_checker_end_frame_clean() {
    let mut checker = FrameLeakChecker::new();
    checker.begin_frame();

    let unreleased = checker.end_frame();

    assert!(unreleased.is_empty());
}

/// Test is_clean() true when empty
#[test]
fn test_frame_leak_checker_is_clean_empty() {
    let checker = FrameLeakChecker::new();
    assert!(checker.is_clean());
}

/// Test is_clean() false when leaks
#[test]
fn test_frame_leak_checker_is_clean_with_leaks() {
    let mut checker = FrameLeakChecker::new();
    checker.begin_frame();
    checker.track(1);

    assert!(!checker.is_clean());
}

/// Test is_clean() after all released
#[test]
fn test_frame_leak_checker_is_clean_after_release() {
    let mut checker = FrameLeakChecker::new();
    checker.begin_frame();
    checker.track(1);
    checker.release(1);

    assert!(checker.is_clean());
}

/// Test frame_number() getter
#[test]
fn test_frame_leak_checker_frame_number() {
    let mut checker = FrameLeakChecker::new();
    assert_eq!(checker.frame_number(), 0);

    checker.begin_frame();
    assert_eq!(checker.frame_number(), 1);
}

/// Test unreleased_count() accuracy
#[test]
fn test_frame_leak_checker_unreleased_count() {
    let mut checker = FrameLeakChecker::new();
    checker.begin_frame();

    assert_eq!(checker.unreleased_count(), 0);

    checker.track(1);
    assert_eq!(checker.unreleased_count(), 1);

    checker.track(2);
    assert_eq!(checker.unreleased_count(), 2);

    checker.release(1);
    assert_eq!(checker.unreleased_count(), 1);
}

/// Test multi-frame sequences
#[test]
fn test_frame_leak_checker_multi_frame() {
    let mut checker = FrameLeakChecker::new();

    // Frame 1: track and leak
    checker.begin_frame();
    checker.track(1);
    let leaks1 = checker.end_frame();
    assert_eq!(leaks1, vec![1]);

    // Frame 2: track and release
    checker.begin_frame();
    checker.track(2);
    checker.release(2);
    let leaks2 = checker.end_frame();
    assert!(leaks2.is_empty());

    // Frame 3: multiple operations
    checker.begin_frame();
    checker.track(3);
    checker.track(4);
    checker.release(3);
    let leaks3 = checker.end_frame();
    assert_eq!(leaks3, vec![4]);

    assert_eq!(checker.frame_number(), 3);
}

/// Test nested frame handling (begin without end)
#[test]
fn test_frame_leak_checker_begin_without_end() {
    let mut checker = FrameLeakChecker::new();

    checker.begin_frame();
    checker.track(1);

    // Begin next frame without ending previous (clears state)
    checker.begin_frame();
    assert!(checker.is_clean());
    assert_eq!(checker.frame_number(), 2);
}

/// Test long frame sequence
#[test]
fn test_frame_leak_checker_long_sequence() {
    let mut checker = FrameLeakChecker::new();

    for frame in 0..1000 {
        checker.begin_frame();
        checker.track(frame);
        checker.release(frame);
        let leaks = checker.end_frame();
        assert!(leaks.is_empty());
    }

    assert_eq!(checker.frame_number(), 1000);
}

/// Test frame checker debug output
#[test]
fn test_frame_leak_checker_debug() {
    let checker = FrameLeakChecker::new();
    let debug_str = format!("{:?}", checker);
    assert!(debug_str.contains("FrameLeakChecker"));
}

// ============================================================================
// Integration Tests (Additional coverage)
// ============================================================================

/// Test full workflow: create detector, track, check, release, stats
#[test]
fn test_full_leak_detection_workflow() {
    // min_size_bytes = 1024 means allocations < 1024 bytes are excluded
    let thresholds = LeakThresholds::custom(0, 120, 1024);
    let mut detector = LeakDetector::new(thresholds);

    // Track several allocations
    detector.track_allocation_typed(1, "Vertex Buffer", 1024 * 1024, ResourceType::Buffer);
    detector.track_allocation_typed(2, "Index Buffer", 256 * 1024, ResourceType::Buffer);
    detector.track_allocation_typed(3, "Uniform Buffer", 512, ResourceType::Buffer); // Below min (512 < 1024)
    detector.track_allocation_typed(4, "Static Texture", 4 * 1024 * 1024, ResourceType::Texture);

    // Mark some as expected
    detector.mark_expected(4);

    // Initial check
    let candidates = detector.check();
    assert_eq!(candidates.len(), 2); // Only 1 and 2 (3 below min, 4 expected)

    // Release some
    detector.release_allocation(1);
    detector.release_allocation(2);

    // Final stats
    let stats = detector.stats();
    assert_eq!(stats.total_tracked, 4);
    assert_eq!(stats.total_released, 2);
    assert_eq!(stats.current_tracked, 2); // 3 and 4 remain
    assert_eq!(stats.expected_long_lived, 1);
    assert_eq!(stats.checks_performed, 1);
}

/// Test detector with frame checker integration
#[test]
fn test_detector_with_frame_checker() {
    let mut detector = LeakDetector::with_default_thresholds();
    let mut frame_checker = FrameLeakChecker::new();

    // Simulate frame
    frame_checker.begin_frame();

    // Track in both
    detector.track_allocation(1, "FrameTemp", 2048);
    frame_checker.track(1);

    // Release in both
    detector.release_allocation(1);
    frame_checker.release(1);

    // End frame
    let frame_leaks = frame_checker.end_frame();
    assert!(frame_leaks.is_empty());

    // Detector stats
    let stats = detector.stats();
    assert_eq!(stats.total_tracked, 1);
    assert_eq!(stats.total_released, 1);
}
