// Blackbox contract tests for T-WGPU-P7.3.2 DebugUtils (debug::utils module).
//
// CLEANROOM: No src/ access beyond the public API exported by the crate.
// Tests use only `renderer_backend::debug::*` -- no internal fields,
// no private methods, no implementation details.
//
// NOTE: This tests the new debug::utils module (T-WGPU-P7.3.2), NOT the older
// debug_utils module (T-WGPU-P4.5.1) which has its own tests in blackbox_debug_utils.rs.
//
// Contract:
//   debug::utils provides GPU debugging utilities including:
//   - DeviceLostReason: Extended device loss reasons with recoverability
//   - DeviceLostInfo: Detailed device loss information with timestamps
//   - ErrorScope: Error scope with accumulated error collection
//   - ErrorFilter: Error type filtering with match semantics
//   - GpuError: Structured GPU error with source locations
//   - GpuErrorType: Error type classification with severity
//   - SourceLocation: Source code location for debugging
//   - ErrorCallbackRegistry: Thread-safe callback management
//   - DebugUtils: Main utility struct for error handling
//
// Coverage:
//   1.  Device Loss Handling (20 tests)
//   2.  Error Scope Workflows (15 tests)
//   3.  Error Collection (12 tests)
//   4.  Error Callback System (15 tests)
//   5.  Error Severity Classification (10 tests)
//   6.  RAII Error Capture (8 tests)
//   7.  Real-World Scenarios (12 tests)
//   8.  Source Location Tracking (10 tests)
//   9.  Thread Safety (8 tests)
//  10.  Edge Cases and Negative Tests (15 tests)
//
// Total: 125+ tests

use renderer_backend::debug::{
    DebugUtils, DeviceLostInfo, DeviceLostReason, ErrorCallbackFn, ErrorCallbackRegistry,
    ErrorFilter, ErrorScope, GpuError, GpuErrorType, Severity, SourceLocation,
};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

// =============================================================================
// SECTION 1 -- DeviceLostReason Tests
// =============================================================================

mod device_lost_reason {
    use super::*;

    #[test]
    fn reason_unknown_exists() {
        let reason = DeviceLostReason::Unknown;
        assert_eq!(format!("{}", reason), "Unknown");
    }

    #[test]
    fn reason_destroyed_exists() {
        let reason = DeviceLostReason::Destroyed;
        assert_eq!(format!("{}", reason), "Destroyed");
    }

    #[test]
    fn reason_device_invalid_exists() {
        let reason = DeviceLostReason::DeviceInvalid;
        assert_eq!(format!("{}", reason), "DeviceInvalid");
    }

    #[test]
    fn reason_driver_error_exists() {
        let reason = DeviceLostReason::DriverError;
        assert_eq!(format!("{}", reason), "DriverError");
    }

    #[test]
    fn destroyed_is_not_recoverable() {
        assert!(!DeviceLostReason::Destroyed.is_recoverable());
    }

    #[test]
    fn unknown_is_recoverable() {
        assert!(DeviceLostReason::Unknown.is_recoverable());
    }

    #[test]
    fn device_invalid_is_recoverable() {
        assert!(DeviceLostReason::DeviceInvalid.is_recoverable());
    }

    #[test]
    fn driver_error_is_recoverable() {
        assert!(DeviceLostReason::DriverError.is_recoverable());
    }

    #[test]
    fn destroyed_description_mentions_destroyed() {
        let desc = DeviceLostReason::Destroyed.description();
        assert!(
            desc.to_lowercase().contains("destroy"),
            "Destroyed description should mention destroy: {}",
            desc
        );
    }

    #[test]
    fn driver_error_description_mentions_driver() {
        let desc = DeviceLostReason::DriverError.description();
        assert!(
            desc.to_lowercase().contains("driver"),
            "DriverError description should mention driver: {}",
            desc
        );
    }

    #[test]
    fn device_invalid_description_mentions_invalid() {
        let desc = DeviceLostReason::DeviceInvalid.description();
        assert!(
            desc.to_lowercase().contains("invalid"),
            "DeviceInvalid description should mention invalid: {}",
            desc
        );
    }

    #[test]
    fn unknown_description_mentions_unknown() {
        let desc = DeviceLostReason::Unknown.description();
        assert!(
            desc.to_lowercase().contains("unknown"),
            "Unknown description should mention unknown: {}",
            desc
        );
    }

    #[test]
    fn default_is_unknown() {
        assert_eq!(DeviceLostReason::default(), DeviceLostReason::Unknown);
    }

    #[test]
    fn reasons_are_clone() {
        let reason = DeviceLostReason::DriverError;
        let cloned = reason.clone();
        assert_eq!(reason, cloned);
    }

    #[test]
    fn reasons_are_copy() {
        let reason = DeviceLostReason::DriverError;
        let copied: DeviceLostReason = reason;
        assert_eq!(reason, copied);
    }

    #[test]
    fn reasons_are_eq() {
        assert_eq!(DeviceLostReason::Unknown, DeviceLostReason::Unknown);
        assert_ne!(DeviceLostReason::Unknown, DeviceLostReason::Destroyed);
    }

    #[test]
    fn reasons_are_debug() {
        let debug_str = format!("{:?}", DeviceLostReason::DriverError);
        assert!(debug_str.contains("DriverError"));
    }

    #[test]
    fn reasons_are_hashable() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(DeviceLostReason::Unknown);
        set.insert(DeviceLostReason::Destroyed);
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn all_reasons_have_descriptions() {
        let reasons = [
            DeviceLostReason::Unknown,
            DeviceLostReason::Destroyed,
            DeviceLostReason::DeviceInvalid,
            DeviceLostReason::DriverError,
        ];
        for reason in reasons {
            let desc = reason.description();
            assert!(!desc.is_empty(), "Description for {:?} should not be empty", reason);
        }
    }

    #[test]
    fn all_reasons_have_display() {
        let reasons = [
            DeviceLostReason::Unknown,
            DeviceLostReason::Destroyed,
            DeviceLostReason::DeviceInvalid,
            DeviceLostReason::DriverError,
        ];
        for reason in reasons {
            let display = format!("{}", reason);
            assert!(!display.is_empty(), "Display for {:?} should not be empty", reason);
        }
    }
}

// =============================================================================
// SECTION 2 -- DeviceLostInfo Tests
// =============================================================================

mod device_lost_info {
    use super::*;

    #[test]
    fn new_captures_reason() {
        let info = DeviceLostInfo::new(DeviceLostReason::DriverError, "Test message".to_string());
        assert_eq!(info.reason, DeviceLostReason::DriverError);
    }

    #[test]
    fn new_captures_message() {
        let info = DeviceLostInfo::new(DeviceLostReason::Unknown, "Error occurred".to_string());
        assert_eq!(info.message, "Error occurred");
    }

    #[test]
    fn new_captures_timestamp() {
        let info = DeviceLostInfo::new(DeviceLostReason::Unknown, String::new());
        // Timestamp should be very recent
        assert!(info.elapsed() < Duration::from_secs(1));
    }

    #[test]
    fn elapsed_increases_over_time() {
        let info = DeviceLostInfo::new(DeviceLostReason::Unknown, String::new());
        let elapsed1 = info.elapsed();
        std::thread::sleep(Duration::from_millis(10));
        let elapsed2 = info.elapsed();
        assert!(elapsed2 > elapsed1);
    }

    #[test]
    fn should_attempt_recovery_for_recoverable() {
        let info = DeviceLostInfo::new(DeviceLostReason::DriverError, String::new());
        assert!(info.should_attempt_recovery());
    }

    #[test]
    fn should_not_attempt_recovery_for_destroyed() {
        let info = DeviceLostInfo::new(DeviceLostReason::Destroyed, String::new());
        assert!(!info.should_attempt_recovery());
    }

    #[test]
    fn display_contains_reason() {
        let info = DeviceLostInfo::new(DeviceLostReason::DriverError, "Test".to_string());
        let display = format!("{}", info);
        assert!(display.contains("DriverError"));
    }

    #[test]
    fn display_contains_message() {
        let info = DeviceLostInfo::new(DeviceLostReason::Unknown, "Custom message".to_string());
        let display = format!("{}", info);
        assert!(display.contains("Custom message"));
    }

    #[test]
    fn info_is_clone() {
        let info = DeviceLostInfo::new(DeviceLostReason::DriverError, "Test".to_string());
        let cloned = info.clone();
        assert_eq!(cloned.reason, info.reason);
        assert_eq!(cloned.message, info.message);
    }

    #[test]
    fn info_is_debug() {
        let info = DeviceLostInfo::new(DeviceLostReason::Unknown, "Test".to_string());
        let debug_str = format!("{:?}", info);
        assert!(debug_str.contains("DeviceLostInfo"));
    }

    #[test]
    fn empty_message_is_valid() {
        let info = DeviceLostInfo::new(DeviceLostReason::Unknown, String::new());
        assert!(info.message.is_empty());
    }

    #[test]
    fn long_message_is_preserved() {
        let long_msg = "A".repeat(10000);
        let info = DeviceLostInfo::new(DeviceLostReason::Unknown, long_msg.clone());
        assert_eq!(info.message, long_msg);
    }
}

// =============================================================================
// SECTION 3 -- ErrorFilter Tests
// =============================================================================

mod error_filter {
    use super::*;

    #[test]
    fn validation_filter_matches_validation() {
        assert!(ErrorFilter::Validation.matches(&GpuErrorType::Validation));
    }

    #[test]
    fn validation_filter_rejects_oom() {
        assert!(!ErrorFilter::Validation.matches(&GpuErrorType::OutOfMemory));
    }

    #[test]
    fn validation_filter_rejects_internal() {
        assert!(!ErrorFilter::Validation.matches(&GpuErrorType::Internal));
    }

    #[test]
    fn validation_filter_rejects_lost() {
        assert!(!ErrorFilter::Validation.matches(&GpuErrorType::Lost));
    }

    #[test]
    fn oom_filter_matches_oom() {
        assert!(ErrorFilter::OutOfMemory.matches(&GpuErrorType::OutOfMemory));
    }

    #[test]
    fn oom_filter_rejects_validation() {
        assert!(!ErrorFilter::OutOfMemory.matches(&GpuErrorType::Validation));
    }

    #[test]
    fn oom_filter_rejects_internal() {
        assert!(!ErrorFilter::OutOfMemory.matches(&GpuErrorType::Internal));
    }

    #[test]
    fn oom_filter_rejects_lost() {
        assert!(!ErrorFilter::OutOfMemory.matches(&GpuErrorType::Lost));
    }

    #[test]
    fn internal_filter_matches_internal() {
        assert!(ErrorFilter::Internal.matches(&GpuErrorType::Internal));
    }

    #[test]
    fn internal_filter_rejects_validation() {
        assert!(!ErrorFilter::Internal.matches(&GpuErrorType::Validation));
    }

    #[test]
    fn internal_filter_rejects_oom() {
        assert!(!ErrorFilter::Internal.matches(&GpuErrorType::OutOfMemory));
    }

    #[test]
    fn all_filter_matches_validation() {
        assert!(ErrorFilter::All.matches(&GpuErrorType::Validation));
    }

    #[test]
    fn all_filter_matches_oom() {
        assert!(ErrorFilter::All.matches(&GpuErrorType::OutOfMemory));
    }

    #[test]
    fn all_filter_matches_internal() {
        assert!(ErrorFilter::All.matches(&GpuErrorType::Internal));
    }

    #[test]
    fn all_filter_matches_lost() {
        assert!(ErrorFilter::All.matches(&GpuErrorType::Lost));
    }

    #[test]
    fn default_filter_is_all() {
        assert_eq!(ErrorFilter::default(), ErrorFilter::All);
    }

    #[test]
    fn filter_display_validation() {
        assert_eq!(format!("{}", ErrorFilter::Validation), "Validation");
    }

    #[test]
    fn filter_display_oom() {
        assert_eq!(format!("{}", ErrorFilter::OutOfMemory), "OutOfMemory");
    }

    #[test]
    fn filter_display_internal() {
        assert_eq!(format!("{}", ErrorFilter::Internal), "Internal");
    }

    #[test]
    fn filter_display_all() {
        assert_eq!(format!("{}", ErrorFilter::All), "All");
    }

    #[test]
    fn filters_are_eq() {
        assert_eq!(ErrorFilter::Validation, ErrorFilter::Validation);
        assert_ne!(ErrorFilter::Validation, ErrorFilter::OutOfMemory);
    }

    #[test]
    fn filters_are_copy() {
        let filter = ErrorFilter::Validation;
        let copied: ErrorFilter = filter;
        assert_eq!(filter, copied);
    }
}

// =============================================================================
// SECTION 4 -- GpuErrorType Tests
// =============================================================================

mod gpu_error_type {
    use super::*;

    #[test]
    fn validation_severity_is_warning() {
        assert_eq!(GpuErrorType::Validation.severity(), Severity::Warning);
    }

    #[test]
    fn oom_severity_is_error() {
        assert_eq!(GpuErrorType::OutOfMemory.severity(), Severity::Error);
    }

    #[test]
    fn internal_severity_is_error() {
        assert_eq!(GpuErrorType::Internal.severity(), Severity::Error);
    }

    #[test]
    fn lost_severity_is_error() {
        assert_eq!(GpuErrorType::Lost.severity(), Severity::Error);
    }

    #[test]
    fn validation_is_not_fatal() {
        assert!(!GpuErrorType::Validation.is_fatal());
    }

    #[test]
    fn oom_is_not_fatal() {
        assert!(!GpuErrorType::OutOfMemory.is_fatal());
    }

    #[test]
    fn internal_is_not_fatal() {
        assert!(!GpuErrorType::Internal.is_fatal());
    }

    #[test]
    fn lost_is_fatal() {
        assert!(GpuErrorType::Lost.is_fatal());
    }

    #[test]
    fn is_validation_helper() {
        assert!(GpuErrorType::Validation.is_validation());
        assert!(!GpuErrorType::OutOfMemory.is_validation());
        assert!(!GpuErrorType::Internal.is_validation());
        assert!(!GpuErrorType::Lost.is_validation());
    }

    #[test]
    fn is_oom_helper() {
        assert!(GpuErrorType::OutOfMemory.is_oom());
        assert!(!GpuErrorType::Validation.is_oom());
        assert!(!GpuErrorType::Internal.is_oom());
        assert!(!GpuErrorType::Lost.is_oom());
    }

    #[test]
    fn is_internal_helper() {
        assert!(GpuErrorType::Internal.is_internal());
        assert!(!GpuErrorType::Validation.is_internal());
        assert!(!GpuErrorType::OutOfMemory.is_internal());
        assert!(!GpuErrorType::Lost.is_internal());
    }

    #[test]
    fn display_validation() {
        assert_eq!(format!("{}", GpuErrorType::Validation), "Validation");
    }

    #[test]
    fn display_oom() {
        assert_eq!(format!("{}", GpuErrorType::OutOfMemory), "OutOfMemory");
    }

    #[test]
    fn display_internal() {
        assert_eq!(format!("{}", GpuErrorType::Internal), "Internal");
    }

    #[test]
    fn display_lost() {
        assert_eq!(format!("{}", GpuErrorType::Lost), "DeviceLost");
    }

    #[test]
    fn error_types_are_eq() {
        assert_eq!(GpuErrorType::Validation, GpuErrorType::Validation);
        assert_ne!(GpuErrorType::Validation, GpuErrorType::OutOfMemory);
    }

    #[test]
    fn error_types_are_copy() {
        let error_type = GpuErrorType::Validation;
        let copied: GpuErrorType = error_type;
        assert_eq!(error_type, copied);
    }

    #[test]
    fn error_types_are_hashable() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(GpuErrorType::Validation);
        set.insert(GpuErrorType::OutOfMemory);
        set.insert(GpuErrorType::Internal);
        set.insert(GpuErrorType::Lost);
        assert_eq!(set.len(), 4);
    }
}

// =============================================================================
// SECTION 5 -- Severity Tests
// =============================================================================

mod severity {
    use super::*;

    #[test]
    fn error_is_greater_than_warning() {
        assert!(Severity::Error > Severity::Warning);
    }

    #[test]
    fn warning_is_greater_than_info() {
        assert!(Severity::Warning > Severity::Info);
    }

    #[test]
    fn error_is_greatest() {
        assert!(Severity::Error > Severity::Warning);
        assert!(Severity::Error > Severity::Info);
    }

    #[test]
    fn info_is_least() {
        assert!(Severity::Info < Severity::Warning);
        assert!(Severity::Info < Severity::Error);
    }

    #[test]
    fn severity_display_info() {
        assert_eq!(format!("{}", Severity::Info), "INFO");
    }

    #[test]
    fn severity_display_warning() {
        assert_eq!(format!("{}", Severity::Warning), "WARN");
    }

    #[test]
    fn severity_display_error() {
        assert_eq!(format!("{}", Severity::Error), "ERROR");
    }

    #[test]
    fn severity_is_eq() {
        assert_eq!(Severity::Error, Severity::Error);
        assert_ne!(Severity::Error, Severity::Warning);
    }

    #[test]
    fn severity_is_ord() {
        let mut severities = vec![Severity::Warning, Severity::Error, Severity::Info];
        severities.sort();
        assert_eq!(severities, vec![Severity::Info, Severity::Warning, Severity::Error]);
    }

    #[test]
    fn severity_is_hashable() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(Severity::Info);
        set.insert(Severity::Warning);
        set.insert(Severity::Error);
        assert_eq!(set.len(), 3);
    }
}

// =============================================================================
// SECTION 6 -- SourceLocation Tests
// =============================================================================

mod source_location {
    use super::*;

    #[test]
    fn new_has_no_file() {
        let loc = SourceLocation::new();
        assert!(loc.file.is_none());
    }

    #[test]
    fn new_has_no_line() {
        let loc = SourceLocation::new();
        assert!(loc.line.is_none());
    }

    #[test]
    fn new_has_no_column() {
        let loc = SourceLocation::new();
        assert!(loc.column.is_none());
    }

    #[test]
    fn new_has_no_function() {
        let loc = SourceLocation::new();
        assert!(loc.function.is_none());
    }

    #[test]
    fn new_is_not_available() {
        let loc = SourceLocation::new();
        assert!(!loc.is_available());
    }

    #[test]
    fn here_has_file() {
        let loc = SourceLocation::here();
        assert!(loc.file.is_some());
    }

    #[test]
    fn here_has_line() {
        let loc = SourceLocation::here();
        assert!(loc.line.is_some());
    }

    #[test]
    fn here_has_column() {
        let loc = SourceLocation::here();
        assert!(loc.column.is_some());
    }

    #[test]
    fn here_is_available() {
        let loc = SourceLocation::here();
        assert!(loc.is_available());
    }

    #[test]
    fn with_file_sets_file() {
        let loc = SourceLocation::new().with_file("test.rs");
        assert_eq!(loc.file.as_deref(), Some("test.rs"));
    }

    #[test]
    fn with_line_sets_line() {
        let loc = SourceLocation::new().with_line(42);
        assert_eq!(loc.line, Some(42));
    }

    #[test]
    fn with_column_sets_column() {
        let loc = SourceLocation::new().with_column(10);
        assert_eq!(loc.column, Some(10));
    }

    #[test]
    fn with_function_sets_function() {
        let loc = SourceLocation::new().with_function("render_frame");
        assert_eq!(loc.function.as_deref(), Some("render_frame"));
    }

    #[test]
    fn builder_chain_works() {
        let loc = SourceLocation::new()
            .with_file("src/renderer.rs")
            .with_line(100)
            .with_column(5)
            .with_function("draw_mesh");

        assert_eq!(loc.file.as_deref(), Some("src/renderer.rs"));
        assert_eq!(loc.line, Some(100));
        assert_eq!(loc.column, Some(5));
        assert_eq!(loc.function.as_deref(), Some("draw_mesh"));
    }

    #[test]
    fn is_available_with_file_only() {
        let loc = SourceLocation::new().with_file("test.rs");
        assert!(loc.is_available());
    }

    #[test]
    fn is_available_with_line_only() {
        let loc = SourceLocation::new().with_line(42);
        assert!(loc.is_available());
    }

    #[test]
    fn is_available_with_function_only() {
        let loc = SourceLocation::new().with_function("test_fn");
        assert!(loc.is_available());
    }

    #[test]
    fn display_empty_shows_unknown() {
        let loc = SourceLocation::new();
        assert_eq!(format!("{}", loc), "<unknown location>");
    }

    #[test]
    fn display_file_only() {
        let loc = SourceLocation::new().with_file("test.rs");
        let display = format!("{}", loc);
        assert!(display.contains("test.rs"));
    }

    #[test]
    fn display_file_and_line() {
        let loc = SourceLocation::new().with_file("test.rs").with_line(42);
        let display = format!("{}", loc);
        assert!(display.contains("test.rs"));
        assert!(display.contains("42"));
    }

    #[test]
    fn display_file_line_column() {
        let loc = SourceLocation::new()
            .with_file("test.rs")
            .with_line(42)
            .with_column(10);
        let display = format!("{}", loc);
        assert!(display.contains("test.rs"));
        assert!(display.contains("42"));
        assert!(display.contains("10"));
    }

    #[test]
    fn display_with_function() {
        let loc = SourceLocation::new()
            .with_file("test.rs")
            .with_line(42)
            .with_function("render");
        let display = format!("{}", loc);
        assert!(display.contains("render"));
    }

    #[test]
    fn display_function_only() {
        let loc = SourceLocation::new().with_function("test_fn");
        let display = format!("{}", loc);
        assert!(display.contains("test_fn"));
    }

    #[test]
    fn location_is_clone() {
        let loc = SourceLocation::new().with_file("test.rs").with_line(42);
        let cloned = loc.clone();
        assert_eq!(loc, cloned);
    }

    #[test]
    fn location_is_debug() {
        let loc = SourceLocation::new().with_file("test.rs");
        let debug_str = format!("{:?}", loc);
        assert!(debug_str.contains("SourceLocation"));
    }

    #[test]
    fn location_is_default() {
        let loc = SourceLocation::default();
        assert!(!loc.is_available());
    }
}

// =============================================================================
// SECTION 7 -- GpuError Tests
// =============================================================================

mod gpu_error {
    use super::*;

    #[test]
    fn new_sets_error_type() {
        let error = GpuError::new(GpuErrorType::Validation, "Test".to_string());
        assert_eq!(error.error_type, GpuErrorType::Validation);
    }

    #[test]
    fn new_sets_message() {
        let error = GpuError::new(GpuErrorType::Validation, "Test message".to_string());
        assert_eq!(error.message, "Test message");
    }

    #[test]
    fn new_has_no_source_location() {
        let error = GpuError::new(GpuErrorType::Validation, "Test".to_string());
        assert!(error.source_location.is_none());
    }

    #[test]
    fn new_has_timestamp() {
        let error = GpuError::new(GpuErrorType::Validation, "Test".to_string());
        // Timestamp should be very recent
        assert!(error.timestamp.elapsed() < Duration::from_secs(1));
    }

    #[test]
    fn with_source_location_sets_location() {
        let error = GpuError::new(GpuErrorType::Validation, "Test".to_string())
            .with_source_location(SourceLocation::here());
        assert!(error.source_location.is_some());
    }

    #[test]
    fn is_validation_for_validation_error() {
        let error = GpuError::new(GpuErrorType::Validation, String::new());
        assert!(error.is_validation());
    }

    #[test]
    fn is_validation_false_for_oom() {
        let error = GpuError::new(GpuErrorType::OutOfMemory, String::new());
        assert!(!error.is_validation());
    }

    #[test]
    fn is_oom_for_oom_error() {
        let error = GpuError::new(GpuErrorType::OutOfMemory, String::new());
        assert!(error.is_oom());
    }

    #[test]
    fn is_oom_false_for_validation() {
        let error = GpuError::new(GpuErrorType::Validation, String::new());
        assert!(!error.is_oom());
    }

    #[test]
    fn is_internal_for_internal_error() {
        let error = GpuError::new(GpuErrorType::Internal, String::new());
        assert!(error.is_internal());
    }

    #[test]
    fn is_internal_false_for_validation() {
        let error = GpuError::new(GpuErrorType::Validation, String::new());
        assert!(!error.is_internal());
    }

    #[test]
    fn severity_returns_correct_value() {
        let validation = GpuError::new(GpuErrorType::Validation, String::new());
        assert_eq!(validation.severity(), Severity::Warning);

        let oom = GpuError::new(GpuErrorType::OutOfMemory, String::new());
        assert_eq!(oom.severity(), Severity::Error);
    }

    #[test]
    fn display_contains_severity() {
        let error = GpuError::new(GpuErrorType::Validation, "Test".to_string());
        let display = format!("{}", error);
        assert!(display.contains("WARN"));
    }

    #[test]
    fn display_contains_error_type() {
        let error = GpuError::new(GpuErrorType::Validation, "Test".to_string());
        let display = format!("{}", error);
        assert!(display.contains("Validation"));
    }

    #[test]
    fn display_contains_message() {
        let error = GpuError::new(GpuErrorType::Validation, "Buffer too large".to_string());
        let display = format!("{}", error);
        assert!(display.contains("Buffer too large"));
    }

    #[test]
    fn display_contains_source_location() {
        let error = GpuError::new(GpuErrorType::Validation, "Test".to_string())
            .with_source_location(SourceLocation::new().with_file("render.rs").with_line(100));
        let display = format!("{}", error);
        assert!(display.contains("render.rs"));
        assert!(display.contains("100"));
    }

    #[test]
    fn error_is_clone() {
        let error = GpuError::new(GpuErrorType::Validation, "Test".to_string());
        let cloned = error.clone();
        assert_eq!(cloned.error_type, error.error_type);
        assert_eq!(cloned.message, error.message);
    }

    #[test]
    fn error_is_debug() {
        let error = GpuError::new(GpuErrorType::Validation, "Test".to_string());
        let debug_str = format!("{:?}", error);
        assert!(debug_str.contains("GpuError"));
    }

    #[test]
    fn error_implements_std_error() {
        fn assert_error<E: std::error::Error>() {}
        assert_error::<GpuError>();
    }
}

// =============================================================================
// SECTION 8 -- ErrorScope Tests
// =============================================================================

mod error_scope {
    use super::*;

    #[test]
    fn new_has_correct_filter() {
        let scope = ErrorScope::new(ErrorFilter::Validation);
        assert_eq!(scope.filter, ErrorFilter::Validation);
    }

    #[test]
    fn new_has_no_label() {
        let scope = ErrorScope::new(ErrorFilter::All);
        assert!(scope.label.is_none());
    }

    #[test]
    fn new_has_no_errors() {
        let scope = ErrorScope::new(ErrorFilter::All);
        assert!(!scope.has_errors());
    }

    #[test]
    fn new_has_zero_error_count() {
        let scope = ErrorScope::new(ErrorFilter::All);
        assert_eq!(scope.error_count(), 0);
    }

    #[test]
    fn with_label_sets_label() {
        let scope = ErrorScope::with_label(ErrorFilter::All, "shadow_pass");
        assert_eq!(scope.label.as_deref(), Some("shadow_pass"));
    }

    #[test]
    fn push_accepts_matching_error() {
        let mut scope = ErrorScope::new(ErrorFilter::Validation);
        let accepted = scope.push(GpuError::new(GpuErrorType::Validation, "Test".to_string()));
        assert!(accepted);
        assert!(scope.has_errors());
        assert_eq!(scope.error_count(), 1);
    }

    #[test]
    fn push_rejects_non_matching_error() {
        let mut scope = ErrorScope::new(ErrorFilter::Validation);
        let rejected = scope.push(GpuError::new(GpuErrorType::OutOfMemory, "OOM".to_string()));
        assert!(!rejected);
        assert!(!scope.has_errors());
        assert_eq!(scope.error_count(), 0);
    }

    #[test]
    fn push_all_filter_accepts_all_types() {
        let mut scope = ErrorScope::new(ErrorFilter::All);

        scope.push(GpuError::new(GpuErrorType::Validation, "1".to_string()));
        scope.push(GpuError::new(GpuErrorType::OutOfMemory, "2".to_string()));
        scope.push(GpuError::new(GpuErrorType::Internal, "3".to_string()));
        scope.push(GpuError::new(GpuErrorType::Lost, "4".to_string()));

        assert_eq!(scope.error_count(), 4);
    }

    #[test]
    fn errors_returns_collected_errors() {
        let mut scope = ErrorScope::new(ErrorFilter::All);
        scope.push(GpuError::new(GpuErrorType::Validation, "First".to_string()));
        scope.push(GpuError::new(GpuErrorType::OutOfMemory, "Second".to_string()));

        let errors = scope.errors();
        assert_eq!(errors.len(), 2);
        assert_eq!(errors[0].message, "First");
        assert_eq!(errors[1].message, "Second");
    }

    #[test]
    fn take_errors_removes_errors() {
        let mut scope = ErrorScope::new(ErrorFilter::All);
        scope.push(GpuError::new(GpuErrorType::Validation, "Test".to_string()));

        let errors = scope.take_errors();
        assert_eq!(errors.len(), 1);
        assert!(!scope.has_errors());
        assert_eq!(scope.error_count(), 0);
    }

    #[test]
    fn clear_removes_all_errors() {
        let mut scope = ErrorScope::new(ErrorFilter::All);
        scope.push(GpuError::new(GpuErrorType::Validation, "1".to_string()));
        scope.push(GpuError::new(GpuErrorType::OutOfMemory, "2".to_string()));

        scope.clear();
        assert!(!scope.has_errors());
        assert_eq!(scope.error_count(), 0);
    }

    #[test]
    fn elapsed_increases_over_time() {
        let scope = ErrorScope::new(ErrorFilter::All);
        let elapsed1 = scope.elapsed();
        std::thread::sleep(Duration::from_millis(10));
        let elapsed2 = scope.elapsed();
        assert!(elapsed2 > elapsed1);
    }

    #[test]
    fn most_severe_returns_none_for_empty() {
        let scope = ErrorScope::new(ErrorFilter::All);
        assert!(scope.most_severe().is_none());
    }

    #[test]
    fn most_severe_returns_error_over_warning() {
        let mut scope = ErrorScope::new(ErrorFilter::All);
        scope.push(GpuError::new(GpuErrorType::Validation, "warning".to_string())); // Warning
        scope.push(GpuError::new(GpuErrorType::OutOfMemory, "error".to_string())); // Error

        let most_severe = scope.most_severe().unwrap();
        assert_eq!(most_severe.error_type, GpuErrorType::OutOfMemory);
    }

    #[test]
    fn most_severe_with_multiple_errors() {
        let mut scope = ErrorScope::new(ErrorFilter::All);
        scope.push(GpuError::new(GpuErrorType::Validation, "1".to_string()));
        scope.push(GpuError::new(GpuErrorType::Internal, "2".to_string()));
        scope.push(GpuError::new(GpuErrorType::Lost, "3".to_string()));

        let most_severe = scope.most_severe().unwrap();
        // All three (Internal, Lost) have Error severity; any of them is valid
        assert_eq!(most_severe.severity(), Severity::Error);
    }

    #[test]
    fn display_contains_label() {
        let scope = ErrorScope::with_label(ErrorFilter::Validation, "render_pass");
        let display = format!("{}", scope);
        assert!(display.contains("render_pass"));
    }

    #[test]
    fn display_contains_filter() {
        let scope = ErrorScope::new(ErrorFilter::Validation);
        let display = format!("{}", scope);
        assert!(display.contains("Validation"));
    }

    #[test]
    fn display_contains_error_count() {
        let mut scope = ErrorScope::new(ErrorFilter::All);
        scope.push(GpuError::new(GpuErrorType::Validation, "1".to_string()));
        scope.push(GpuError::new(GpuErrorType::Validation, "2".to_string()));
        let display = format!("{}", scope);
        assert!(display.contains("2"));
    }

    #[test]
    fn scope_is_clone() {
        let mut scope = ErrorScope::with_label(ErrorFilter::All, "test");
        scope.push(GpuError::new(GpuErrorType::Validation, "error".to_string()));
        let cloned = scope.clone();
        assert_eq!(cloned.error_count(), scope.error_count());
    }

    #[test]
    fn scope_is_debug() {
        let scope = ErrorScope::new(ErrorFilter::All);
        let debug_str = format!("{:?}", scope);
        assert!(debug_str.contains("ErrorScope"));
    }
}

// =============================================================================
// SECTION 9 -- ErrorCallbackRegistry Tests
// =============================================================================

mod error_callback_registry {
    use super::*;

    #[test]
    fn new_is_empty() {
        let registry = ErrorCallbackRegistry::new();
        assert!(registry.is_empty());
        assert_eq!(registry.len(), 0);
    }

    #[test]
    fn register_increases_len() {
        let registry = ErrorCallbackRegistry::new();
        registry.register(Arc::new(|_| {}));
        assert_eq!(registry.len(), 1);
        assert!(!registry.is_empty());
    }

    #[test]
    fn register_returns_unique_ids() {
        let registry = ErrorCallbackRegistry::new();
        let id1 = registry.register(Arc::new(|_| {}));
        let id2 = registry.register(Arc::new(|_| {}));
        assert_ne!(id1, id2);
    }

    #[test]
    fn unregister_removes_callback() {
        let registry = ErrorCallbackRegistry::new();
        let id = registry.register(Arc::new(|_| {}));
        assert_eq!(registry.len(), 1);

        let removed = registry.unregister(id);
        assert!(removed);
        assert_eq!(registry.len(), 0);
    }

    #[test]
    fn unregister_returns_false_for_unknown_id() {
        let registry = ErrorCallbackRegistry::new();
        let removed = registry.unregister(999);
        assert!(!removed);
    }

    #[test]
    fn unregister_same_id_twice_returns_false_second_time() {
        let registry = ErrorCallbackRegistry::new();
        let id = registry.register(Arc::new(|_| {}));

        assert!(registry.unregister(id));
        assert!(!registry.unregister(id));
    }

    #[test]
    fn invoke_calls_callback() {
        let called = Arc::new(AtomicBool::new(false));
        let called_clone = called.clone();

        let registry = ErrorCallbackRegistry::new();
        registry.register(Arc::new(move |_| {
            called_clone.store(true, Ordering::SeqCst);
        }));

        let error = GpuError::new(GpuErrorType::Validation, "Test".to_string());
        registry.invoke(&error);

        assert!(called.load(Ordering::SeqCst));
    }

    #[test]
    fn invoke_returns_callback_count() {
        let registry = ErrorCallbackRegistry::new();
        registry.register(Arc::new(|_| {}));
        registry.register(Arc::new(|_| {}));

        let error = GpuError::new(GpuErrorType::Validation, "Test".to_string());
        let count = registry.invoke(&error);

        assert_eq!(count, 2);
    }

    #[test]
    fn invoke_calls_multiple_callbacks() {
        let counter = Arc::new(AtomicU32::new(0));

        let registry = ErrorCallbackRegistry::new();
        for _ in 0..5 {
            let counter_clone = counter.clone();
            registry.register(Arc::new(move |_| {
                counter_clone.fetch_add(1, Ordering::SeqCst);
            }));
        }

        let error = GpuError::new(GpuErrorType::Validation, "Test".to_string());
        registry.invoke(&error);

        assert_eq!(counter.load(Ordering::SeqCst), 5);
    }

    #[test]
    fn invoke_receives_correct_error() {
        let received_message = Arc::new(std::sync::Mutex::new(String::new()));
        let received_message_clone = received_message.clone();

        let registry = ErrorCallbackRegistry::new();
        registry.register(Arc::new(move |error| {
            *received_message_clone.lock().unwrap() = error.message.clone();
        }));

        let error = GpuError::new(GpuErrorType::Validation, "Specific message".to_string());
        registry.invoke(&error);

        assert_eq!(*received_message.lock().unwrap(), "Specific message");
    }

    #[test]
    fn clear_removes_all_callbacks() {
        let registry = ErrorCallbackRegistry::new();
        registry.register(Arc::new(|_| {}));
        registry.register(Arc::new(|_| {}));
        registry.register(Arc::new(|_| {}));

        registry.clear();
        assert!(registry.is_empty());
        assert_eq!(registry.len(), 0);
    }

    #[test]
    fn registry_is_clone() {
        let registry = ErrorCallbackRegistry::new();
        let id = registry.register(Arc::new(|_| {}));
        let cloned = registry.clone();

        // Both registries share the same underlying state
        assert_eq!(cloned.len(), 1);
        registry.unregister(id);
        assert_eq!(cloned.len(), 0);
    }

    #[test]
    fn registry_is_default() {
        let registry = ErrorCallbackRegistry::default();
        assert!(registry.is_empty());
    }

    #[test]
    fn registry_is_debug() {
        let registry = ErrorCallbackRegistry::new();
        registry.register(Arc::new(|_| {}));
        let debug_str = format!("{:?}", registry);
        assert!(debug_str.contains("ErrorCallbackRegistry"));
    }
}

// =============================================================================
// SECTION 10 -- DebugUtils Tests
// =============================================================================

mod debug_utils {
    use super::*;

    #[test]
    fn new_has_zero_scope_depth() {
        let debug = DebugUtils::new();
        assert_eq!(debug.scope_depth(), 0);
    }

    #[test]
    fn new_has_zero_total_errors() {
        let debug = DebugUtils::new();
        assert_eq!(debug.total_errors(), 0);
    }

    #[test]
    fn push_error_scope_increases_depth() {
        let mut debug = DebugUtils::new();
        debug.push_error_scope(ErrorFilter::Validation);
        assert_eq!(debug.scope_depth(), 1);
    }

    #[test]
    fn push_multiple_scopes() {
        let mut debug = DebugUtils::new();
        debug.push_error_scope(ErrorFilter::Validation);
        debug.push_error_scope(ErrorFilter::OutOfMemory);
        debug.push_error_scope(ErrorFilter::All);
        assert_eq!(debug.scope_depth(), 3);
    }

    #[test]
    fn pop_error_scope_decreases_depth() {
        let mut debug = DebugUtils::new();
        debug.push_error_scope(ErrorFilter::Validation);
        debug.pop_error_scope();
        assert_eq!(debug.scope_depth(), 0);
    }

    #[test]
    fn pop_error_scope_returns_correct_scope() {
        let mut debug = DebugUtils::new();
        debug.push_error_scope(ErrorFilter::Validation);
        debug.push_error_scope(ErrorFilter::OutOfMemory);

        let scope = debug.pop_error_scope().unwrap();
        assert_eq!(scope.filter, ErrorFilter::OutOfMemory);

        let scope = debug.pop_error_scope().unwrap();
        assert_eq!(scope.filter, ErrorFilter::Validation);
    }

    #[test]
    fn pop_error_scope_returns_none_when_empty() {
        let mut debug = DebugUtils::new();
        assert!(debug.pop_error_scope().is_none());
    }

    #[test]
    fn push_error_scope_labeled_sets_label() {
        let mut debug = DebugUtils::new();
        debug.push_error_scope_labeled(ErrorFilter::All, "shadow_pass");
        let scope = debug.pop_error_scope().unwrap();
        assert_eq!(scope.label.as_deref(), Some("shadow_pass"));
    }

    #[test]
    fn push_error_without_scope_returns_false() {
        let mut debug = DebugUtils::new();
        let captured = debug.push_error(GpuError::new(GpuErrorType::Validation, "Test".to_string()));
        assert!(!captured);
    }

    #[test]
    fn push_error_without_scope_still_counts_error() {
        let mut debug = DebugUtils::new();
        debug.push_error(GpuError::new(GpuErrorType::Validation, "Test".to_string()));
        assert_eq!(debug.total_errors(), 1);
    }

    #[test]
    fn push_error_with_scope_returns_true() {
        let mut debug = DebugUtils::new();
        debug.push_error_scope(ErrorFilter::All);
        let captured = debug.push_error(GpuError::new(GpuErrorType::Validation, "Test".to_string()));
        assert!(captured);
    }

    #[test]
    fn push_error_increments_total_errors() {
        let mut debug = DebugUtils::new();
        debug.push_error_scope(ErrorFilter::All);

        debug.push_error(GpuError::new(GpuErrorType::Validation, "1".to_string()));
        debug.push_error(GpuError::new(GpuErrorType::OutOfMemory, "2".to_string()));

        assert_eq!(debug.total_errors(), 2);
    }

    #[test]
    fn push_error_to_top_scope() {
        let mut debug = DebugUtils::new();
        debug.push_error_scope(ErrorFilter::Validation); // Bottom scope
        debug.push_error_scope(ErrorFilter::OutOfMemory); // Top scope

        // Push OOM error - should go to top scope
        debug.push_error(GpuError::new(GpuErrorType::OutOfMemory, "OOM".to_string()));

        // Top scope should have the error
        let top = debug.pop_error_scope().unwrap();
        assert_eq!(top.error_count(), 1);

        // Bottom scope should be empty
        let bottom = debug.pop_error_scope().unwrap();
        assert_eq!(bottom.error_count(), 0);
    }

    #[test]
    fn device_lost_handler_is_called() {
        let called = Arc::new(AtomicBool::new(false));
        let called_clone = called.clone();

        let mut debug = DebugUtils::new();
        debug.set_device_lost_handler(move |_info| {
            called_clone.store(true, Ordering::SeqCst);
        });

        debug.notify_device_lost(DeviceLostInfo::new(
            DeviceLostReason::DriverError,
            "Test".to_string(),
        ));

        assert!(called.load(Ordering::SeqCst));
    }

    #[test]
    fn device_lost_handler_receives_correct_info() {
        let received_reason = Arc::new(std::sync::Mutex::new(DeviceLostReason::Unknown));
        let received_reason_clone = received_reason.clone();

        let mut debug = DebugUtils::new();
        debug.set_device_lost_handler(move |info| {
            *received_reason_clone.lock().unwrap() = info.reason;
        });

        debug.notify_device_lost(DeviceLostInfo::new(
            DeviceLostReason::DriverError,
            "Test".to_string(),
        ));

        assert_eq!(*received_reason.lock().unwrap(), DeviceLostReason::DriverError);
    }

    #[test]
    fn notify_without_handler_does_not_panic() {
        let debug = DebugUtils::new();
        // Should not panic
        debug.notify_device_lost(DeviceLostInfo::new(
            DeviceLostReason::Unknown,
            "Test".to_string(),
        ));
    }

    #[test]
    fn clear_device_lost_handler_stops_notifications() {
        let called = Arc::new(AtomicBool::new(false));
        let called_clone = called.clone();

        let mut debug = DebugUtils::new();
        debug.set_device_lost_handler(move |_info| {
            called_clone.store(true, Ordering::SeqCst);
        });
        debug.clear_device_lost_handler();

        debug.notify_device_lost(DeviceLostInfo::new(
            DeviceLostReason::Unknown,
            "Test".to_string(),
        ));

        assert!(!called.load(Ordering::SeqCst));
    }

    #[test]
    fn callbacks_returns_registry() {
        let debug = DebugUtils::new();
        let _callbacks = debug.callbacks();
        // Just verify we can access it
    }

    #[test]
    fn callbacks_mut_allows_modification() {
        let mut debug = DebugUtils::new();
        debug.callbacks_mut().register(Arc::new(|_| {}));
        assert_eq!(debug.callbacks().len(), 1);
    }

    #[test]
    fn push_error_invokes_callbacks() {
        let called = Arc::new(AtomicBool::new(false));
        let called_clone = called.clone();

        let mut debug = DebugUtils::new();
        debug.callbacks_mut().register(Arc::new(move |_| {
            called_clone.store(true, Ordering::SeqCst);
        }));

        debug.push_error(GpuError::new(GpuErrorType::Validation, "Test".to_string()));

        assert!(called.load(Ordering::SeqCst));
    }

    #[test]
    fn debug_utils_is_default() {
        let debug = DebugUtils::default();
        assert_eq!(debug.scope_depth(), 0);
    }

    #[test]
    fn debug_utils_is_debug() {
        let debug = DebugUtils::new();
        let debug_str = format!("{:?}", debug);
        assert!(debug_str.contains("DebugUtils"));
    }
}

// =============================================================================
// SECTION 11 -- RAII Error Capture Guard Tests
// =============================================================================

mod error_capture_guard {
    use super::*;

    #[test]
    fn capture_errors_creates_scope() {
        let mut debug = DebugUtils::new();
        {
            let _guard = debug.capture_errors(ErrorFilter::Validation);
            // Guard exists, scope should be active
        }
        // Guard dropped, scope should be popped
        assert_eq!(debug.scope_depth(), 0);
    }

    #[test]
    fn capture_errors_guard_pops_on_drop() {
        let mut debug = DebugUtils::new();
        debug.push_error_scope(ErrorFilter::All); // Existing scope

        {
            let _guard = debug.capture_errors(ErrorFilter::Validation);
            // Now we have 2 scopes, but we can't check inside the guard
            // because of borrow rules
        }

        // After guard drops, only the original scope remains
        assert_eq!(debug.scope_depth(), 1);
    }

    #[test]
    fn nested_guards_work_correctly() {
        let mut debug = DebugUtils::new();

        // We can only have one guard at a time due to borrow rules
        // But we can test sequential guards
        {
            let _guard = debug.capture_errors(ErrorFilter::Validation);
        }
        assert_eq!(debug.scope_depth(), 0);

        {
            let _guard = debug.capture_errors(ErrorFilter::OutOfMemory);
        }
        assert_eq!(debug.scope_depth(), 0);
    }
}

// =============================================================================
// SECTION 12 -- Real-World Scenario Tests
// =============================================================================

mod real_world_scenarios {
    use super::*;

    #[test]
    fn shader_compilation_error_scenario() {
        let mut debug = DebugUtils::new();
        debug.push_error_scope(ErrorFilter::Validation);

        // Simulate shader compilation error
        let error = GpuError::new(
            GpuErrorType::Validation,
            "WGSL shader compilation failed: unexpected token at line 42".to_string(),
        )
        .with_source_location(SourceLocation::new().with_file("shaders/pbr.wgsl").with_line(42));

        debug.push_error(error);

        let scope = debug.pop_error_scope().unwrap();
        assert!(scope.has_errors());
        let errors = scope.errors();
        assert_eq!(errors.len(), 1);
        assert!(errors[0].message.contains("shader"));
        assert!(errors[0].source_location.is_some());
    }

    #[test]
    fn buffer_creation_oom_scenario() {
        let mut debug = DebugUtils::new();
        debug.push_error_scope(ErrorFilter::OutOfMemory);

        // Simulate buffer allocation failure
        let error = GpuError::new(
            GpuErrorType::OutOfMemory,
            "Failed to allocate 4GB buffer: insufficient GPU memory".to_string(),
        )
        .with_source_location(SourceLocation::new().with_function("create_vertex_buffer"));

        debug.push_error(error);

        let scope = debug.pop_error_scope().unwrap();
        assert!(scope.has_errors());
        assert!(scope.errors()[0].is_oom());
    }

    #[test]
    fn pipeline_validation_failure_scenario() {
        let mut debug = DebugUtils::new();
        debug.push_error_scope(ErrorFilter::Validation);

        // Simulate pipeline validation errors
        debug.push_error(GpuError::new(
            GpuErrorType::Validation,
            "Vertex buffer stride mismatch".to_string(),
        ));
        debug.push_error(GpuError::new(
            GpuErrorType::Validation,
            "Missing required vertex attribute: POSITION".to_string(),
        ));

        let scope = debug.pop_error_scope().unwrap();
        assert_eq!(scope.error_count(), 2);
    }

    #[test]
    fn device_lost_during_render_scenario() {
        let device_lost = Arc::new(AtomicBool::new(false));
        let device_lost_clone = device_lost.clone();

        let mut debug = DebugUtils::new();
        debug.set_device_lost_handler(move |info| {
            if info.reason == DeviceLostReason::DriverError {
                device_lost_clone.store(true, Ordering::SeqCst);
            }
        });

        // Simulate device lost
        debug.notify_device_lost(DeviceLostInfo::new(
            DeviceLostReason::DriverError,
            "GPU driver crashed during draw call".to_string(),
        ));

        assert!(device_lost.load(Ordering::SeqCst));
    }

    #[test]
    fn multiple_concurrent_errors_scenario() {
        let mut debug = DebugUtils::new();
        debug.push_error_scope(ErrorFilter::All);

        // Simulate multiple errors from different sources
        let error_types = [
            (GpuErrorType::Validation, "Invalid bind group"),
            (GpuErrorType::OutOfMemory, "Texture allocation failed"),
            (GpuErrorType::Internal, "Driver internal error"),
            (GpuErrorType::Validation, "Missing sampler"),
        ];

        for (error_type, message) in error_types {
            debug.push_error(GpuError::new(error_type, message.to_string()));
        }

        let scope = debug.pop_error_scope().unwrap();
        assert_eq!(scope.error_count(), 4);

        // Most severe should be an Error-level severity
        let most_severe = scope.most_severe().unwrap();
        assert_eq!(most_severe.severity(), Severity::Error);
    }

    #[test]
    fn error_callback_logging_scenario() {
        let error_log = Arc::new(std::sync::Mutex::new(Vec::new()));
        let error_log_clone = error_log.clone();

        let mut debug = DebugUtils::new();
        debug.callbacks_mut().register(Arc::new(move |error| {
            error_log_clone.lock().unwrap().push(format!(
                "[{}] {}: {}",
                error.error_type, error.severity(), error.message
            ));
        }));

        debug.push_error_scope(ErrorFilter::All);
        debug.push_error(GpuError::new(
            GpuErrorType::Validation,
            "Invalid operation".to_string(),
        ));
        debug.push_error(GpuError::new(
            GpuErrorType::OutOfMemory,
            "Allocation failed".to_string(),
        ));

        let log = error_log.lock().unwrap();
        assert_eq!(log.len(), 2);
        assert!(log[0].contains("Validation"));
        assert!(log[1].contains("OutOfMemory"));
    }

    #[test]
    fn nested_render_passes_scenario() {
        let mut debug = DebugUtils::new();

        // Outer render pass scope
        debug.push_error_scope_labeled(ErrorFilter::All, "main_pass");
        debug.push_error(GpuError::new(
            GpuErrorType::Validation,
            "Main pass error".to_string(),
        ));

        // Inner shadow pass scope
        debug.push_error_scope_labeled(ErrorFilter::All, "shadow_pass");
        debug.push_error(GpuError::new(
            GpuErrorType::Validation,
            "Shadow pass error".to_string(),
        ));

        // Pop shadow pass
        let shadow_scope = debug.pop_error_scope().unwrap();
        assert_eq!(shadow_scope.label.as_deref(), Some("shadow_pass"));
        assert_eq!(shadow_scope.error_count(), 1);

        // Pop main pass
        let main_scope = debug.pop_error_scope().unwrap();
        assert_eq!(main_scope.label.as_deref(), Some("main_pass"));
        assert_eq!(main_scope.error_count(), 1);
    }

    #[test]
    fn error_recovery_decision_scenario() {
        // Test using device lost info to decide recovery strategy
        let recoverable_reasons = [
            DeviceLostReason::Unknown,
            DeviceLostReason::DeviceInvalid,
            DeviceLostReason::DriverError,
        ];

        for reason in recoverable_reasons {
            let info = DeviceLostInfo::new(reason, String::new());
            assert!(
                info.should_attempt_recovery(),
                "{:?} should be recoverable",
                reason
            );
        }

        let info = DeviceLostInfo::new(DeviceLostReason::Destroyed, String::new());
        assert!(!info.should_attempt_recovery());
    }

    #[test]
    fn filtered_error_collection_scenario() {
        let mut debug = DebugUtils::new();

        // Only collect validation errors
        debug.push_error_scope(ErrorFilter::Validation);

        // Try to push different error types
        let val_captured = debug.push_error(GpuError::new(
            GpuErrorType::Validation,
            "Validation error".to_string(),
        ));
        let oom_captured = debug.push_error(GpuError::new(
            GpuErrorType::OutOfMemory,
            "OOM error".to_string(),
        ));

        assert!(val_captured);
        assert!(!oom_captured);

        let scope = debug.pop_error_scope().unwrap();
        assert_eq!(scope.error_count(), 1);
        assert!(scope.errors()[0].is_validation());
    }

    #[test]
    fn total_error_tracking_across_scopes_scenario() {
        let mut debug = DebugUtils::new();

        // First scope
        debug.push_error_scope(ErrorFilter::All);
        debug.push_error(GpuError::new(GpuErrorType::Validation, "1".to_string()));
        debug.push_error(GpuError::new(GpuErrorType::Validation, "2".to_string()));
        debug.pop_error_scope();

        // Second scope
        debug.push_error_scope(ErrorFilter::All);
        debug.push_error(GpuError::new(GpuErrorType::Validation, "3".to_string()));
        debug.pop_error_scope();

        // Total errors should be cumulative
        assert_eq!(debug.total_errors(), 3);
    }

    #[test]
    fn source_location_debugging_scenario() {
        let error = GpuError::new(
            GpuErrorType::Validation,
            "Invalid vertex format".to_string(),
        )
        .with_source_location(
            SourceLocation::new()
                .with_file("src/mesh.rs")
                .with_line(156)
                .with_column(24)
                .with_function("create_vertex_buffer"),
        );

        let display = format!("{}", error);
        assert!(display.contains("src/mesh.rs"));
        assert!(display.contains("156"));
        assert!(display.contains("create_vertex_buffer"));
    }
}

// =============================================================================
// SECTION 13 -- Thread Safety Tests
// =============================================================================

mod thread_safety {
    use super::*;

    #[test]
    fn callback_registry_register_from_multiple_threads() {
        let registry = ErrorCallbackRegistry::new();
        let registry_clone = registry.clone();

        let handles: Vec<_> = (0..10)
            .map(|_| {
                let r = registry_clone.clone();
                thread::spawn(move || {
                    r.register(Arc::new(|_| {}));
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }

        assert_eq!(registry.len(), 10);
    }

    #[test]
    fn callback_registry_invoke_from_multiple_threads() {
        let counter = Arc::new(AtomicU64::new(0));
        let registry = ErrorCallbackRegistry::new();

        let counter_clone = counter.clone();
        registry.register(Arc::new(move |_| {
            counter_clone.fetch_add(1, Ordering::SeqCst);
        }));

        let registry_clone = registry.clone();
        let handles: Vec<_> = (0..10)
            .map(|_| {
                let r = registry_clone.clone();
                thread::spawn(move || {
                    let error = GpuError::new(GpuErrorType::Validation, "Test".to_string());
                    r.invoke(&error);
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }

        assert_eq!(counter.load(Ordering::SeqCst), 10);
    }

    #[test]
    fn callback_registry_concurrent_register_unregister() {
        let registry = ErrorCallbackRegistry::new();
        let ids = Arc::new(std::sync::Mutex::new(Vec::new()));

        // Register from multiple threads
        let registry_clone = registry.clone();
        let ids_clone = ids.clone();
        let handles: Vec<_> = (0..5)
            .map(|_| {
                let r = registry_clone.clone();
                let i = ids_clone.clone();
                thread::spawn(move || {
                    let id = r.register(Arc::new(|_| {}));
                    i.lock().unwrap().push(id);
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }

        // Unregister from multiple threads
        let registry_clone = registry.clone();
        let handles: Vec<_> = ids
            .lock()
            .unwrap()
            .clone()
            .into_iter()
            .map(|id| {
                let r = registry_clone.clone();
                thread::spawn(move || {
                    r.unregister(id);
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }

        assert!(registry.is_empty());
    }

    #[test]
    fn callback_registry_clear_while_invoking() {
        let registry = ErrorCallbackRegistry::new();
        let counter = Arc::new(AtomicU32::new(0));

        let counter_clone = counter.clone();
        registry.register(Arc::new(move |_| {
            counter_clone.fetch_add(1, Ordering::SeqCst);
            // Simulate some work
            thread::sleep(Duration::from_millis(1));
        }));

        let registry_invoke = registry.clone();
        let registry_clear = registry.clone();

        let invoke_handle = thread::spawn(move || {
            for _ in 0..10 {
                let error = GpuError::new(GpuErrorType::Validation, "Test".to_string());
                registry_invoke.invoke(&error);
            }
        });

        let clear_handle = thread::spawn(move || {
            thread::sleep(Duration::from_millis(5));
            registry_clear.clear();
        });

        invoke_handle.join().unwrap();
        clear_handle.join().unwrap();

        // Counter should have been incremented at least once
        assert!(counter.load(Ordering::SeqCst) > 0);
    }

    #[test]
    fn source_location_here_in_thread() {
        let handle = thread::spawn(|| {
            let loc = SourceLocation::here();
            assert!(loc.is_available());
            assert!(loc.file.is_some());
            assert!(loc.line.is_some());
        });

        handle.join().unwrap();
    }

    #[test]
    fn device_lost_info_clone_across_threads() {
        let info = DeviceLostInfo::new(DeviceLostReason::DriverError, "Test".to_string());

        let handles: Vec<_> = (0..10)
            .map(|_| {
                let info_clone = info.clone();
                thread::spawn(move || {
                    assert_eq!(info_clone.reason, DeviceLostReason::DriverError);
                    assert_eq!(info_clone.message, "Test");
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }
    }

    #[test]
    fn gpu_error_clone_across_threads() {
        let error = GpuError::new(GpuErrorType::Validation, "Test error".to_string())
            .with_source_location(SourceLocation::here());

        let handles: Vec<_> = (0..10)
            .map(|_| {
                let error_clone = error.clone();
                thread::spawn(move || {
                    assert!(error_clone.is_validation());
                    assert_eq!(error_clone.message, "Test error");
                    assert!(error_clone.source_location.is_some());
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }
    }

    #[test]
    fn error_scope_clone_across_threads() {
        let mut scope = ErrorScope::with_label(ErrorFilter::All, "test_scope");
        scope.push(GpuError::new(GpuErrorType::Validation, "Error 1".to_string()));
        scope.push(GpuError::new(GpuErrorType::OutOfMemory, "Error 2".to_string()));

        let handles: Vec<_> = (0..10)
            .map(|_| {
                let scope_clone = scope.clone();
                thread::spawn(move || {
                    assert_eq!(scope_clone.error_count(), 2);
                    assert_eq!(scope_clone.label.as_deref(), Some("test_scope"));
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }
    }
}

// =============================================================================
// SECTION 14 -- Edge Cases and Negative Tests
// =============================================================================

mod edge_cases {
    use super::*;

    #[test]
    fn empty_message_in_gpu_error() {
        let error = GpuError::new(GpuErrorType::Validation, String::new());
        assert!(error.message.is_empty());
        // Should still display properly
        let display = format!("{}", error);
        assert!(display.contains("Validation"));
    }

    #[test]
    fn very_long_message_in_gpu_error() {
        let long_msg = "A".repeat(100000);
        let error = GpuError::new(GpuErrorType::Validation, long_msg.clone());
        assert_eq!(error.message.len(), 100000);
    }

    #[test]
    fn unicode_message_in_gpu_error() {
        let unicode_msg = "GPU Error: \u{1F4A5} Explosion! \u{1F525}".to_string();
        let error = GpuError::new(GpuErrorType::Internal, unicode_msg.clone());
        assert_eq!(error.message, unicode_msg);
    }

    #[test]
    fn empty_label_in_error_scope() {
        let scope = ErrorScope::with_label(ErrorFilter::All, "");
        assert_eq!(scope.label.as_deref(), Some(""));
    }

    #[test]
    fn special_chars_in_source_location_file() {
        let loc = SourceLocation::new().with_file("path/to/file with spaces.rs");
        assert_eq!(loc.file.as_deref(), Some("path/to/file with spaces.rs"));
    }

    #[test]
    fn zero_line_in_source_location() {
        let loc = SourceLocation::new().with_line(0);
        assert_eq!(loc.line, Some(0));
    }

    #[test]
    fn max_line_in_source_location() {
        let loc = SourceLocation::new().with_line(u32::MAX);
        assert_eq!(loc.line, Some(u32::MAX));
    }

    #[test]
    fn pop_from_empty_scope_stack() {
        let mut debug = DebugUtils::new();
        assert!(debug.pop_error_scope().is_none());
        assert!(debug.pop_error_scope().is_none());
    }

    #[test]
    fn push_error_to_filtered_scope_does_not_match() {
        let mut debug = DebugUtils::new();
        debug.push_error_scope(ErrorFilter::OutOfMemory);

        // Push validation error to OOM-only scope
        let captured = debug.push_error(GpuError::new(
            GpuErrorType::Validation,
            "Should not be captured".to_string(),
        ));

        assert!(!captured);
        let scope = debug.pop_error_scope().unwrap();
        assert!(!scope.has_errors());
    }

    #[test]
    fn take_errors_twice() {
        let mut scope = ErrorScope::new(ErrorFilter::All);
        scope.push(GpuError::new(GpuErrorType::Validation, "Test".to_string()));

        let first = scope.take_errors();
        assert_eq!(first.len(), 1);

        let second = scope.take_errors();
        assert!(second.is_empty());
    }

    #[test]
    fn clear_empty_scope() {
        let mut scope = ErrorScope::new(ErrorFilter::All);
        scope.clear(); // Should not panic
        assert!(!scope.has_errors());
    }

    #[test]
    fn unregister_never_registered_id() {
        let registry = ErrorCallbackRegistry::new();
        assert!(!registry.unregister(12345));
    }

    #[test]
    fn invoke_empty_registry() {
        let registry = ErrorCallbackRegistry::new();
        let error = GpuError::new(GpuErrorType::Validation, "Test".to_string());
        let count = registry.invoke(&error);
        assert_eq!(count, 0);
    }

    #[test]
    fn most_severe_single_error() {
        let mut scope = ErrorScope::new(ErrorFilter::All);
        scope.push(GpuError::new(GpuErrorType::Validation, "Only error".to_string()));
        let most_severe = scope.most_severe().unwrap();
        assert_eq!(most_severe.message, "Only error");
    }

    #[test]
    fn elapsed_after_long_time() {
        let info = DeviceLostInfo::new(DeviceLostReason::Unknown, String::new());
        thread::sleep(Duration::from_millis(50));
        let elapsed = info.elapsed();
        assert!(elapsed >= Duration::from_millis(50));
    }

    #[test]
    fn display_source_location_with_column_no_line() {
        // Column without line is unusual but should handle gracefully
        let loc = SourceLocation::new().with_file("test.rs").with_column(10);
        let display = format!("{}", loc);
        assert!(display.contains("test.rs"));
        // Column should not appear without line
    }

    #[test]
    fn callback_receives_error_with_all_fields() {
        let received = Arc::new(std::sync::Mutex::new(None));
        let received_clone = received.clone();

        let registry = ErrorCallbackRegistry::new();
        registry.register(Arc::new(move |error| {
            *received_clone.lock().unwrap() = Some((
                error.error_type,
                error.message.clone(),
                error.source_location.is_some(),
            ));
        }));

        let error = GpuError::new(GpuErrorType::Internal, "Full error".to_string())
            .with_source_location(SourceLocation::here());

        registry.invoke(&error);

        let received = received.lock().unwrap().clone().unwrap();
        assert_eq!(received.0, GpuErrorType::Internal);
        assert_eq!(received.1, "Full error");
        assert!(received.2);
    }
}

// =============================================================================
// SECTION 15 -- Additional Coverage Tests
// =============================================================================

mod additional_coverage {
    use super::*;

    #[test]
    fn all_error_types_have_correct_severity() {
        let test_cases = [
            (GpuErrorType::Validation, Severity::Warning),
            (GpuErrorType::OutOfMemory, Severity::Error),
            (GpuErrorType::Internal, Severity::Error),
            (GpuErrorType::Lost, Severity::Error),
        ];

        for (error_type, expected_severity) in test_cases {
            assert_eq!(
                error_type.severity(),
                expected_severity,
                "{:?} should have {:?} severity",
                error_type,
                expected_severity
            );
        }
    }

    #[test]
    fn error_filter_and_error_type_compatibility() {
        // Validation filter
        assert!(ErrorFilter::Validation.matches(&GpuErrorType::Validation));
        assert!(!ErrorFilter::Validation.matches(&GpuErrorType::OutOfMemory));
        assert!(!ErrorFilter::Validation.matches(&GpuErrorType::Internal));
        assert!(!ErrorFilter::Validation.matches(&GpuErrorType::Lost));

        // OutOfMemory filter
        assert!(!ErrorFilter::OutOfMemory.matches(&GpuErrorType::Validation));
        assert!(ErrorFilter::OutOfMemory.matches(&GpuErrorType::OutOfMemory));
        assert!(!ErrorFilter::OutOfMemory.matches(&GpuErrorType::Internal));
        assert!(!ErrorFilter::OutOfMemory.matches(&GpuErrorType::Lost));

        // Internal filter
        assert!(!ErrorFilter::Internal.matches(&GpuErrorType::Validation));
        assert!(!ErrorFilter::Internal.matches(&GpuErrorType::OutOfMemory));
        assert!(ErrorFilter::Internal.matches(&GpuErrorType::Internal));
        assert!(!ErrorFilter::Internal.matches(&GpuErrorType::Lost));

        // All filter matches everything
        assert!(ErrorFilter::All.matches(&GpuErrorType::Validation));
        assert!(ErrorFilter::All.matches(&GpuErrorType::OutOfMemory));
        assert!(ErrorFilter::All.matches(&GpuErrorType::Internal));
        assert!(ErrorFilter::All.matches(&GpuErrorType::Lost));
    }

    #[test]
    fn multiple_handlers_replace_each_other() {
        let first_called = Arc::new(AtomicBool::new(false));
        let second_called = Arc::new(AtomicBool::new(false));

        let first_clone = first_called.clone();
        let second_clone = second_called.clone();

        let mut debug = DebugUtils::new();

        debug.set_device_lost_handler(move |_| {
            first_clone.store(true, Ordering::SeqCst);
        });

        debug.set_device_lost_handler(move |_| {
            second_clone.store(true, Ordering::SeqCst);
        });

        debug.notify_device_lost(DeviceLostInfo::new(DeviceLostReason::Unknown, String::new()));

        // Only second handler should be called
        assert!(!first_called.load(Ordering::SeqCst));
        assert!(second_called.load(Ordering::SeqCst));
    }

    #[test]
    fn scope_elapsed_is_monotonic() {
        let scope = ErrorScope::new(ErrorFilter::All);
        let mut previous = scope.elapsed();

        for _ in 0..10 {
            thread::sleep(Duration::from_micros(100));
            let current = scope.elapsed();
            assert!(current >= previous);
            previous = current;
        }
    }

    #[test]
    fn device_lost_info_elapsed_is_monotonic() {
        let info = DeviceLostInfo::new(DeviceLostReason::Unknown, String::new());
        let mut previous = info.elapsed();

        for _ in 0..10 {
            thread::sleep(Duration::from_micros(100));
            let current = info.elapsed();
            assert!(current >= previous);
            previous = current;
        }
    }

    #[test]
    fn callbacks_are_invoked_in_registration_order() {
        let order = Arc::new(std::sync::Mutex::new(Vec::new()));

        let registry = ErrorCallbackRegistry::new();

        for i in 0..5 {
            let order_clone = order.clone();
            registry.register(Arc::new(move |_| {
                order_clone.lock().unwrap().push(i);
            }));
        }

        let error = GpuError::new(GpuErrorType::Validation, "Test".to_string());
        registry.invoke(&error);

        let order = order.lock().unwrap();
        assert_eq!(*order, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn total_errors_includes_filtered_errors() {
        let mut debug = DebugUtils::new();
        debug.push_error_scope(ErrorFilter::Validation);

        // This error will be rejected by the scope but still counted
        debug.push_error(GpuError::new(GpuErrorType::OutOfMemory, "OOM".to_string()));

        let scope = debug.pop_error_scope().unwrap();
        assert_eq!(scope.error_count(), 0); // Filtered out
        assert_eq!(debug.total_errors(), 1); // Still counted
    }

    #[test]
    fn debug_utils_debug_output_contains_state() {
        let mut debug = DebugUtils::new();
        debug.push_error_scope(ErrorFilter::All);
        debug.push_error(GpuError::new(GpuErrorType::Validation, "Test".to_string()));
        debug.callbacks_mut().register(Arc::new(|_| {}));

        let debug_str = format!("{:?}", debug);
        assert!(debug_str.contains("scope_depth"));
        assert!(debug_str.contains("callback_count"));
        assert!(debug_str.contains("total_errors"));
    }

    #[test]
    fn error_callback_fn_type_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<ErrorCallbackFn>();
    }

    #[test]
    fn error_callback_registry_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<ErrorCallbackRegistry>();
    }
}
