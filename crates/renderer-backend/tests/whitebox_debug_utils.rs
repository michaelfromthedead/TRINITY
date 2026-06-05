//! Whitebox tests for GPU debugging utilities.
//!
//! Tests cover all components in `debug::utils`:
//! - DeviceLostReason
//! - DeviceLostInfo
//! - ErrorFilter
//! - GpuErrorType
//! - Severity
//! - SourceLocation
//! - GpuError
//! - ErrorScope
//! - ErrorCallbackRegistry
//! - DebugUtils
//! - ErrorCaptureGuard

use renderer_backend::debug::utils::*;
use std::sync::atomic::{AtomicU32, AtomicU64, AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;
use std::thread;

// ============================================================================
// DeviceLostReason Tests (20 tests)
// ============================================================================

mod device_lost_reason_tests {
    use super::*;

    #[test]
    fn test_unknown_variant_exists() {
        let reason = DeviceLostReason::Unknown;
        assert_eq!(format!("{:?}", reason), "Unknown");
    }

    #[test]
    fn test_destroyed_variant_exists() {
        let reason = DeviceLostReason::Destroyed;
        assert_eq!(format!("{:?}", reason), "Destroyed");
    }

    #[test]
    fn test_device_invalid_variant_exists() {
        let reason = DeviceLostReason::DeviceInvalid;
        assert_eq!(format!("{:?}", reason), "DeviceInvalid");
    }

    #[test]
    fn test_driver_error_variant_exists() {
        let reason = DeviceLostReason::DriverError;
        assert_eq!(format!("{:?}", reason), "DriverError");
    }

    #[test]
    fn test_from_wgpu_destroyed() {
        let reason = DeviceLostReason::from_wgpu(wgpu::DeviceLostReason::Destroyed);
        assert_eq!(reason, DeviceLostReason::Destroyed);
    }

    #[test]
    fn test_from_wgpu_replaced_callback() {
        let reason = DeviceLostReason::from_wgpu(wgpu::DeviceLostReason::ReplacedCallback);
        assert_eq!(reason, DeviceLostReason::Unknown);
    }

    #[test]
    fn test_unknown_is_recoverable() {
        assert!(DeviceLostReason::Unknown.is_recoverable());
    }

    #[test]
    fn test_destroyed_not_recoverable() {
        assert!(!DeviceLostReason::Destroyed.is_recoverable());
    }

    #[test]
    fn test_device_invalid_is_recoverable() {
        assert!(DeviceLostReason::DeviceInvalid.is_recoverable());
    }

    #[test]
    fn test_driver_error_is_recoverable() {
        assert!(DeviceLostReason::DriverError.is_recoverable());
    }

    #[test]
    fn test_unknown_description_not_empty() {
        let desc = DeviceLostReason::Unknown.description();
        assert!(!desc.is_empty());
        assert!(desc.contains("unknown"));
    }

    #[test]
    fn test_destroyed_description_contains_destroyed() {
        let desc = DeviceLostReason::Destroyed.description();
        assert!(desc.contains("destroyed") || desc.contains("Destroyed"));
    }

    #[test]
    fn test_device_invalid_description_contains_invalid() {
        let desc = DeviceLostReason::DeviceInvalid.description();
        assert!(desc.contains("invalid") || desc.contains("Invalid") || desc.contains("corruption"));
    }

    #[test]
    fn test_driver_error_description_contains_driver() {
        let desc = DeviceLostReason::DriverError.description();
        assert!(desc.contains("driver"));
    }

    #[test]
    fn test_display_unknown() {
        assert_eq!(format!("{}", DeviceLostReason::Unknown), "Unknown");
    }

    #[test]
    fn test_display_destroyed() {
        assert_eq!(format!("{}", DeviceLostReason::Destroyed), "Destroyed");
    }

    #[test]
    fn test_display_device_invalid() {
        assert_eq!(format!("{}", DeviceLostReason::DeviceInvalid), "DeviceInvalid");
    }

    #[test]
    fn test_display_driver_error() {
        assert_eq!(format!("{}", DeviceLostReason::DriverError), "DriverError");
    }

    #[test]
    fn test_default_is_unknown() {
        assert_eq!(DeviceLostReason::default(), DeviceLostReason::Unknown);
    }

    #[test]
    fn test_clone_and_copy() {
        let original = DeviceLostReason::DriverError;
        let cloned = original.clone();
        let copied = original;
        assert_eq!(original, cloned);
        assert_eq!(original, copied);
    }

    #[test]
    fn test_eq_and_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(DeviceLostReason::Unknown);
        set.insert(DeviceLostReason::Destroyed);
        set.insert(DeviceLostReason::DeviceInvalid);
        set.insert(DeviceLostReason::DriverError);
        assert_eq!(set.len(), 4);
        assert!(set.contains(&DeviceLostReason::Unknown));
    }
}

// ============================================================================
// DeviceLostInfo Tests (15 tests)
// ============================================================================

mod device_lost_info_tests {
    use super::*;

    #[test]
    fn test_new_captures_timestamp() {
        let before = std::time::Instant::now();
        let info = DeviceLostInfo::new(DeviceLostReason::Unknown, "test".to_string());
        let after = std::time::Instant::now();

        // timestamp should be between before and after
        assert!(info.elapsed() <= after.duration_since(before) + Duration::from_millis(10));
    }

    #[test]
    fn test_new_stores_reason() {
        let info = DeviceLostInfo::new(DeviceLostReason::DriverError, String::new());
        assert_eq!(info.reason, DeviceLostReason::DriverError);
    }

    #[test]
    fn test_new_stores_message() {
        let info = DeviceLostInfo::new(DeviceLostReason::Unknown, "GPU crashed".to_string());
        assert_eq!(info.message, "GPU crashed");
    }

    #[test]
    fn test_new_with_empty_message() {
        let info = DeviceLostInfo::new(DeviceLostReason::Unknown, String::new());
        assert!(info.message.is_empty());
    }

    #[test]
    fn test_from_wgpu_converts_reason() {
        let info = DeviceLostInfo::from_wgpu(
            wgpu::DeviceLostReason::Destroyed,
            "device dropped".to_string(),
        );
        assert_eq!(info.reason, DeviceLostReason::Destroyed);
        assert_eq!(info.message, "device dropped");
    }

    #[test]
    fn test_elapsed_increases_over_time() {
        let info = DeviceLostInfo::new(DeviceLostReason::Unknown, String::new());
        let elapsed1 = info.elapsed();
        thread::sleep(Duration::from_millis(5));
        let elapsed2 = info.elapsed();
        assert!(elapsed2 >= elapsed1);
    }

    #[test]
    fn test_should_attempt_recovery_for_driver_error() {
        let info = DeviceLostInfo::new(DeviceLostReason::DriverError, String::new());
        assert!(info.should_attempt_recovery());
    }

    #[test]
    fn test_should_not_attempt_recovery_for_destroyed() {
        let info = DeviceLostInfo::new(DeviceLostReason::Destroyed, String::new());
        assert!(!info.should_attempt_recovery());
    }

    #[test]
    fn test_should_attempt_recovery_for_unknown() {
        let info = DeviceLostInfo::new(DeviceLostReason::Unknown, String::new());
        assert!(info.should_attempt_recovery());
    }

    #[test]
    fn test_should_attempt_recovery_for_device_invalid() {
        let info = DeviceLostInfo::new(DeviceLostReason::DeviceInvalid, String::new());
        assert!(info.should_attempt_recovery());
    }

    #[test]
    fn test_clone() {
        let info = DeviceLostInfo::new(DeviceLostReason::DriverError, "error".to_string());
        let cloned = info.clone();
        assert_eq!(cloned.reason, info.reason);
        assert_eq!(cloned.message, info.message);
    }

    #[test]
    fn test_debug_trait() {
        let info = DeviceLostInfo::new(DeviceLostReason::Unknown, "test".to_string());
        let debug_str = format!("{:?}", info);
        assert!(debug_str.contains("DeviceLostInfo"));
    }

    #[test]
    fn test_display_contains_reason() {
        let info = DeviceLostInfo::new(DeviceLostReason::DriverError, "crash".to_string());
        let display = format!("{}", info);
        assert!(display.contains("DriverError"));
    }

    #[test]
    fn test_display_contains_message() {
        let info = DeviceLostInfo::new(DeviceLostReason::Unknown, "custom error message".to_string());
        let display = format!("{}", info);
        assert!(display.contains("custom error message"));
    }

    #[test]
    fn test_display_contains_elapsed() {
        let info = DeviceLostInfo::new(DeviceLostReason::Unknown, String::new());
        let display = format!("{}", info);
        assert!(display.contains("elapsed"));
    }
}

// ============================================================================
// ErrorFilter Tests (18 tests)
// ============================================================================

mod error_filter_tests {
    use super::*;

    #[test]
    fn test_validation_variant() {
        assert_eq!(format!("{:?}", ErrorFilter::Validation), "Validation");
    }

    #[test]
    fn test_out_of_memory_variant() {
        assert_eq!(format!("{:?}", ErrorFilter::OutOfMemory), "OutOfMemory");
    }

    #[test]
    fn test_internal_variant() {
        assert_eq!(format!("{:?}", ErrorFilter::Internal), "Internal");
    }

    #[test]
    fn test_all_variant() {
        assert_eq!(format!("{:?}", ErrorFilter::All), "All");
    }

    #[test]
    fn test_validation_matches_validation() {
        assert!(ErrorFilter::Validation.matches(&GpuErrorType::Validation));
    }

    #[test]
    fn test_validation_not_matches_oom() {
        assert!(!ErrorFilter::Validation.matches(&GpuErrorType::OutOfMemory));
    }

    #[test]
    fn test_validation_not_matches_internal() {
        assert!(!ErrorFilter::Validation.matches(&GpuErrorType::Internal));
    }

    #[test]
    fn test_validation_not_matches_lost() {
        assert!(!ErrorFilter::Validation.matches(&GpuErrorType::Lost));
    }

    #[test]
    fn test_oom_matches_oom() {
        assert!(ErrorFilter::OutOfMemory.matches(&GpuErrorType::OutOfMemory));
    }

    #[test]
    fn test_oom_not_matches_validation() {
        assert!(!ErrorFilter::OutOfMemory.matches(&GpuErrorType::Validation));
    }

    #[test]
    fn test_internal_matches_internal() {
        assert!(ErrorFilter::Internal.matches(&GpuErrorType::Internal));
    }

    #[test]
    fn test_internal_not_matches_lost() {
        assert!(!ErrorFilter::Internal.matches(&GpuErrorType::Lost));
    }

    #[test]
    fn test_all_matches_validation() {
        assert!(ErrorFilter::All.matches(&GpuErrorType::Validation));
    }

    #[test]
    fn test_all_matches_oom() {
        assert!(ErrorFilter::All.matches(&GpuErrorType::OutOfMemory));
    }

    #[test]
    fn test_all_matches_internal() {
        assert!(ErrorFilter::All.matches(&GpuErrorType::Internal));
    }

    #[test]
    fn test_all_matches_lost() {
        assert!(ErrorFilter::All.matches(&GpuErrorType::Lost));
    }

    #[test]
    fn test_to_wgpu_validation() {
        assert_eq!(ErrorFilter::Validation.to_wgpu(), wgpu::ErrorFilter::Validation);
    }

    #[test]
    fn test_to_wgpu_oom() {
        assert_eq!(ErrorFilter::OutOfMemory.to_wgpu(), wgpu::ErrorFilter::OutOfMemory);
    }

    #[test]
    fn test_to_wgpu_internal_defaults_to_validation() {
        // Internal doesn't have a direct wgpu equivalent
        assert_eq!(ErrorFilter::Internal.to_wgpu(), wgpu::ErrorFilter::Validation);
    }

    #[test]
    fn test_to_wgpu_all_defaults_to_validation() {
        // All doesn't have a direct wgpu equivalent
        assert_eq!(ErrorFilter::All.to_wgpu(), wgpu::ErrorFilter::Validation);
    }

    #[test]
    fn test_display_all() {
        assert_eq!(format!("{}", ErrorFilter::All), "All");
    }

    #[test]
    fn test_display_validation() {
        assert_eq!(format!("{}", ErrorFilter::Validation), "Validation");
    }

    #[test]
    fn test_display_oom() {
        assert_eq!(format!("{}", ErrorFilter::OutOfMemory), "OutOfMemory");
    }

    #[test]
    fn test_display_internal() {
        assert_eq!(format!("{}", ErrorFilter::Internal), "Internal");
    }

    #[test]
    fn test_default_is_all() {
        assert_eq!(ErrorFilter::default(), ErrorFilter::All);
    }

    #[test]
    fn test_clone_copy_eq_hash() {
        use std::collections::HashSet;
        let filter = ErrorFilter::Validation;
        let cloned = filter.clone();
        let copied = filter;
        assert_eq!(filter, cloned);
        assert_eq!(filter, copied);

        let mut set = HashSet::new();
        set.insert(ErrorFilter::Validation);
        set.insert(ErrorFilter::OutOfMemory);
        assert_eq!(set.len(), 2);
    }
}

// ============================================================================
// GpuErrorType Tests (18 tests)
// ============================================================================

mod gpu_error_type_tests {
    use super::*;

    #[test]
    fn test_validation_variant() {
        let t = GpuErrorType::Validation;
        assert_eq!(format!("{:?}", t), "Validation");
    }

    #[test]
    fn test_out_of_memory_variant() {
        let t = GpuErrorType::OutOfMemory;
        assert_eq!(format!("{:?}", t), "OutOfMemory");
    }

    #[test]
    fn test_internal_variant() {
        let t = GpuErrorType::Internal;
        assert_eq!(format!("{:?}", t), "Internal");
    }

    #[test]
    fn test_lost_variant() {
        let t = GpuErrorType::Lost;
        assert_eq!(format!("{:?}", t), "Lost");
    }

    #[test]
    fn test_validation_severity_is_warning() {
        assert_eq!(GpuErrorType::Validation.severity(), Severity::Warning);
    }

    #[test]
    fn test_oom_severity_is_error() {
        assert_eq!(GpuErrorType::OutOfMemory.severity(), Severity::Error);
    }

    #[test]
    fn test_internal_severity_is_error() {
        assert_eq!(GpuErrorType::Internal.severity(), Severity::Error);
    }

    #[test]
    fn test_lost_severity_is_error() {
        assert_eq!(GpuErrorType::Lost.severity(), Severity::Error);
    }

    #[test]
    fn test_validation_not_fatal() {
        assert!(!GpuErrorType::Validation.is_fatal());
    }

    #[test]
    fn test_oom_not_fatal() {
        assert!(!GpuErrorType::OutOfMemory.is_fatal());
    }

    #[test]
    fn test_internal_not_fatal() {
        assert!(!GpuErrorType::Internal.is_fatal());
    }

    #[test]
    fn test_lost_is_fatal() {
        assert!(GpuErrorType::Lost.is_fatal());
    }

    #[test]
    fn test_is_validation_true() {
        assert!(GpuErrorType::Validation.is_validation());
    }

    #[test]
    fn test_is_validation_false_for_others() {
        assert!(!GpuErrorType::OutOfMemory.is_validation());
        assert!(!GpuErrorType::Internal.is_validation());
        assert!(!GpuErrorType::Lost.is_validation());
    }

    #[test]
    fn test_is_oom_true() {
        assert!(GpuErrorType::OutOfMemory.is_oom());
    }

    #[test]
    fn test_is_oom_false_for_others() {
        assert!(!GpuErrorType::Validation.is_oom());
        assert!(!GpuErrorType::Internal.is_oom());
        assert!(!GpuErrorType::Lost.is_oom());
    }

    #[test]
    fn test_is_internal_true() {
        assert!(GpuErrorType::Internal.is_internal());
    }

    #[test]
    fn test_is_internal_false_for_others() {
        assert!(!GpuErrorType::Validation.is_internal());
        assert!(!GpuErrorType::OutOfMemory.is_internal());
        assert!(!GpuErrorType::Lost.is_internal());
    }

    #[test]
    fn test_display_validation() {
        assert_eq!(format!("{}", GpuErrorType::Validation), "Validation");
    }

    #[test]
    fn test_display_oom() {
        assert_eq!(format!("{}", GpuErrorType::OutOfMemory), "OutOfMemory");
    }

    #[test]
    fn test_display_internal() {
        assert_eq!(format!("{}", GpuErrorType::Internal), "Internal");
    }

    #[test]
    fn test_display_lost() {
        assert_eq!(format!("{}", GpuErrorType::Lost), "DeviceLost");
    }

    #[test]
    fn test_clone_copy_eq_hash() {
        use std::collections::HashSet;
        let t = GpuErrorType::Lost;
        let cloned = t.clone();
        let copied = t;
        assert_eq!(t, cloned);
        assert_eq!(t, copied);

        let mut set = HashSet::new();
        set.insert(GpuErrorType::Validation);
        set.insert(GpuErrorType::OutOfMemory);
        set.insert(GpuErrorType::Internal);
        set.insert(GpuErrorType::Lost);
        assert_eq!(set.len(), 4);
    }
}

// ============================================================================
// Severity Tests (10 tests)
// ============================================================================

mod severity_tests {
    use super::*;

    #[test]
    fn test_info_variant() {
        assert_eq!(format!("{:?}", Severity::Info), "Info");
    }

    #[test]
    fn test_warning_variant() {
        assert_eq!(format!("{:?}", Severity::Warning), "Warning");
    }

    #[test]
    fn test_error_variant() {
        assert_eq!(format!("{:?}", Severity::Error), "Error");
    }

    #[test]
    fn test_ordering_info_less_than_warning() {
        assert!(Severity::Info < Severity::Warning);
    }

    #[test]
    fn test_ordering_warning_less_than_error() {
        assert!(Severity::Warning < Severity::Error);
    }

    #[test]
    fn test_ordering_info_less_than_error() {
        assert!(Severity::Info < Severity::Error);
    }

    #[test]
    fn test_ordering_error_greater_than_all() {
        assert!(Severity::Error > Severity::Warning);
        assert!(Severity::Error > Severity::Info);
    }

    #[test]
    fn test_display_info() {
        assert_eq!(format!("{}", Severity::Info), "INFO");
    }

    #[test]
    fn test_display_warning() {
        assert_eq!(format!("{}", Severity::Warning), "WARN");
    }

    #[test]
    fn test_display_error() {
        assert_eq!(format!("{}", Severity::Error), "ERROR");
    }

    #[test]
    fn test_clone_copy_eq_hash() {
        use std::collections::HashSet;
        let s = Severity::Warning;
        let cloned = s.clone();
        let copied = s;
        assert_eq!(s, cloned);
        assert_eq!(s, copied);

        let mut set = HashSet::new();
        set.insert(Severity::Info);
        set.insert(Severity::Warning);
        set.insert(Severity::Error);
        assert_eq!(set.len(), 3);
    }
}

// ============================================================================
// SourceLocation Tests (20 tests)
// ============================================================================

mod source_location_tests {
    use super::*;

    #[test]
    fn test_new_empty() {
        let loc = SourceLocation::new();
        assert!(loc.file.is_none());
        assert!(loc.line.is_none());
        assert!(loc.column.is_none());
        assert!(loc.function.is_none());
    }

    #[test]
    fn test_new_not_available() {
        let loc = SourceLocation::new();
        assert!(!loc.is_available());
    }

    #[test]
    fn test_here_captures_file() {
        let loc = SourceLocation::here();
        assert!(loc.file.is_some());
        assert!(loc.file.as_ref().unwrap().contains("whitebox_debug_utils"));
    }

    #[test]
    fn test_here_captures_line() {
        let loc = SourceLocation::here();
        assert!(loc.line.is_some());
        assert!(loc.line.unwrap() > 0);
    }

    #[test]
    fn test_here_captures_column() {
        let loc = SourceLocation::here();
        assert!(loc.column.is_some());
        assert!(loc.column.unwrap() > 0);
    }

    #[test]
    fn test_here_function_is_none() {
        let loc = SourceLocation::here();
        // Function is not captured by Location::caller()
        assert!(loc.function.is_none());
    }

    #[test]
    fn test_here_is_available() {
        let loc = SourceLocation::here();
        assert!(loc.is_available());
    }

    #[test]
    fn test_with_file() {
        let loc = SourceLocation::new().with_file("my_file.rs");
        assert_eq!(loc.file.as_deref(), Some("my_file.rs"));
    }

    #[test]
    fn test_with_line() {
        let loc = SourceLocation::new().with_line(42);
        assert_eq!(loc.line, Some(42));
    }

    #[test]
    fn test_with_column() {
        let loc = SourceLocation::new().with_column(15);
        assert_eq!(loc.column, Some(15));
    }

    #[test]
    fn test_with_function() {
        let loc = SourceLocation::new().with_function("render_frame");
        assert_eq!(loc.function.as_deref(), Some("render_frame"));
    }

    #[test]
    fn test_builder_chain() {
        let loc = SourceLocation::new()
            .with_file("src/main.rs")
            .with_line(100)
            .with_column(5)
            .with_function("main");

        assert_eq!(loc.file.as_deref(), Some("src/main.rs"));
        assert_eq!(loc.line, Some(100));
        assert_eq!(loc.column, Some(5));
        assert_eq!(loc.function.as_deref(), Some("main"));
    }

    #[test]
    fn test_is_available_with_file_only() {
        let loc = SourceLocation::new().with_file("test.rs");
        assert!(loc.is_available());
    }

    #[test]
    fn test_is_available_with_line_only() {
        let loc = SourceLocation::new().with_line(1);
        assert!(loc.is_available());
    }

    #[test]
    fn test_is_available_with_function_only() {
        let loc = SourceLocation::new().with_function("test");
        assert!(loc.is_available());
    }

    #[test]
    fn test_is_available_with_column_only() {
        // Column alone does not make location available (per implementation)
        let loc = SourceLocation::new().with_column(10);
        // This depends on implementation - file, line, or function required
        // Based on the code: file.is_some() || line.is_some() || function.is_some()
        assert!(!loc.is_available());
    }

    #[test]
    fn test_display_file_only() {
        let loc = SourceLocation::new().with_file("test.rs");
        let display = format!("{}", loc);
        assert_eq!(display, "test.rs");
    }

    #[test]
    fn test_display_file_and_line() {
        let loc = SourceLocation::new().with_file("test.rs").with_line(42);
        let display = format!("{}", loc);
        assert_eq!(display, "test.rs:42");
    }

    #[test]
    fn test_display_file_line_column() {
        let loc = SourceLocation::new()
            .with_file("test.rs")
            .with_line(42)
            .with_column(10);
        let display = format!("{}", loc);
        assert_eq!(display, "test.rs:42:10");
    }

    #[test]
    fn test_display_with_function() {
        let loc = SourceLocation::new()
            .with_file("test.rs")
            .with_line(42)
            .with_function("render");
        let display = format!("{}", loc);
        assert!(display.contains("test.rs:42"));
        assert!(display.contains("render()"));
    }

    #[test]
    fn test_display_function_only() {
        let loc = SourceLocation::new().with_function("my_func");
        let display = format!("{}", loc);
        assert_eq!(display, "my_func()");
    }

    #[test]
    fn test_display_empty() {
        let loc = SourceLocation::new();
        let display = format!("{}", loc);
        assert_eq!(display, "<unknown location>");
    }

    #[test]
    fn test_default() {
        let loc = SourceLocation::default();
        assert!(loc.file.is_none());
        assert!(loc.line.is_none());
        assert!(!loc.is_available());
    }

    #[test]
    fn test_eq() {
        let loc1 = SourceLocation::new().with_file("a.rs").with_line(10);
        let loc2 = SourceLocation::new().with_file("a.rs").with_line(10);
        let loc3 = SourceLocation::new().with_file("b.rs").with_line(10);
        assert_eq!(loc1, loc2);
        assert_ne!(loc1, loc3);
    }

    #[test]
    fn test_clone() {
        let loc = SourceLocation::new().with_file("test.rs").with_line(1);
        let cloned = loc.clone();
        assert_eq!(loc, cloned);
    }
}

// ============================================================================
// GpuError Tests (20 tests)
// ============================================================================

mod gpu_error_tests {
    use super::*;

    #[test]
    fn test_new_stores_error_type() {
        let error = GpuError::new(GpuErrorType::Validation, String::new());
        assert_eq!(error.error_type, GpuErrorType::Validation);
    }

    #[test]
    fn test_new_stores_message() {
        let error = GpuError::new(GpuErrorType::OutOfMemory, "Out of VRAM".to_string());
        assert_eq!(error.message, "Out of VRAM");
    }

    #[test]
    fn test_new_source_location_none() {
        let error = GpuError::new(GpuErrorType::Internal, String::new());
        assert!(error.source_location.is_none());
    }

    #[test]
    fn test_new_captures_timestamp() {
        let before = std::time::Instant::now();
        let error = GpuError::new(GpuErrorType::Validation, String::new());
        let after = std::time::Instant::now();

        // Timestamp should be between before and after
        assert!(error.timestamp >= before);
        assert!(error.timestamp <= after);
    }

    #[test]
    fn test_with_source_location() {
        let loc = SourceLocation::here();
        let error = GpuError::new(GpuErrorType::Validation, "test".to_string())
            .with_source_location(loc.clone());

        assert!(error.source_location.is_some());
        assert_eq!(error.source_location.as_ref().unwrap().line, loc.line);
    }

    #[test]
    fn test_is_validation_true() {
        let error = GpuError::new(GpuErrorType::Validation, String::new());
        assert!(error.is_validation());
    }

    #[test]
    fn test_is_validation_false() {
        let error = GpuError::new(GpuErrorType::OutOfMemory, String::new());
        assert!(!error.is_validation());
    }

    #[test]
    fn test_is_oom_true() {
        let error = GpuError::new(GpuErrorType::OutOfMemory, String::new());
        assert!(error.is_oom());
    }

    #[test]
    fn test_is_oom_false() {
        let error = GpuError::new(GpuErrorType::Internal, String::new());
        assert!(!error.is_oom());
    }

    #[test]
    fn test_is_internal_true() {
        let error = GpuError::new(GpuErrorType::Internal, String::new());
        assert!(error.is_internal());
    }

    #[test]
    fn test_is_internal_false() {
        let error = GpuError::new(GpuErrorType::Lost, String::new());
        assert!(!error.is_internal());
    }

    #[test]
    fn test_severity_validation() {
        let error = GpuError::new(GpuErrorType::Validation, String::new());
        assert_eq!(error.severity(), Severity::Warning);
    }

    #[test]
    fn test_severity_oom() {
        let error = GpuError::new(GpuErrorType::OutOfMemory, String::new());
        assert_eq!(error.severity(), Severity::Error);
    }

    #[test]
    fn test_severity_lost() {
        let error = GpuError::new(GpuErrorType::Lost, String::new());
        assert_eq!(error.severity(), Severity::Error);
    }

    #[test]
    fn test_display_contains_severity() {
        let error = GpuError::new(GpuErrorType::Validation, "test".to_string());
        let display = format!("{}", error);
        assert!(display.contains("WARN"));
    }

    #[test]
    fn test_display_contains_error_type() {
        let error = GpuError::new(GpuErrorType::OutOfMemory, "allocation failed".to_string());
        let display = format!("{}", error);
        assert!(display.contains("OutOfMemory"));
    }

    #[test]
    fn test_display_contains_message() {
        let error = GpuError::new(GpuErrorType::Internal, "driver bug".to_string());
        let display = format!("{}", error);
        assert!(display.contains("driver bug"));
    }

    #[test]
    fn test_display_with_source_location() {
        let error = GpuError::new(GpuErrorType::Validation, "bad params".to_string())
            .with_source_location(SourceLocation::new().with_file("test.rs").with_line(100));
        let display = format!("{}", error);
        assert!(display.contains("test.rs:100"));
    }

    #[test]
    fn test_error_trait() {
        fn assert_error<E: std::error::Error>(_: E) {}
        let error = GpuError::new(GpuErrorType::Validation, String::new());
        assert_error(error);
    }

    #[test]
    fn test_clone() {
        let error = GpuError::new(GpuErrorType::Internal, "test".to_string())
            .with_source_location(SourceLocation::here());
        let cloned = error.clone();
        assert_eq!(cloned.error_type, error.error_type);
        assert_eq!(cloned.message, error.message);
    }

    #[test]
    fn test_debug_trait() {
        let error = GpuError::new(GpuErrorType::Lost, "device lost".to_string());
        let debug_str = format!("{:?}", error);
        assert!(debug_str.contains("GpuError"));
    }
}

// ============================================================================
// ErrorScope Tests (25 tests)
// ============================================================================

mod error_scope_tests {
    use super::*;

    #[test]
    fn test_new_with_validation_filter() {
        let scope = ErrorScope::new(ErrorFilter::Validation);
        assert_eq!(scope.filter, ErrorFilter::Validation);
        assert!(scope.label.is_none());
    }

    #[test]
    fn test_new_with_all_filter() {
        let scope = ErrorScope::new(ErrorFilter::All);
        assert_eq!(scope.filter, ErrorFilter::All);
    }

    #[test]
    fn test_new_empty_errors() {
        let scope = ErrorScope::new(ErrorFilter::All);
        assert!(!scope.has_errors());
        assert_eq!(scope.error_count(), 0);
    }

    #[test]
    fn test_with_label() {
        let scope = ErrorScope::with_label(ErrorFilter::Validation, "shadow_pass");
        assert_eq!(scope.label.as_deref(), Some("shadow_pass"));
    }

    #[test]
    fn test_with_label_string() {
        let scope = ErrorScope::with_label(ErrorFilter::All, String::from("render"));
        assert_eq!(scope.label.as_deref(), Some("render"));
    }

    #[test]
    fn test_push_matching_error() {
        let mut scope = ErrorScope::new(ErrorFilter::Validation);
        let accepted = scope.push(GpuError::new(GpuErrorType::Validation, "test".to_string()));
        assert!(accepted);
        assert!(scope.has_errors());
        assert_eq!(scope.error_count(), 1);
    }

    #[test]
    fn test_push_non_matching_error() {
        let mut scope = ErrorScope::new(ErrorFilter::Validation);
        let accepted = scope.push(GpuError::new(GpuErrorType::OutOfMemory, "oom".to_string()));
        assert!(!accepted);
        assert!(!scope.has_errors());
        assert_eq!(scope.error_count(), 0);
    }

    #[test]
    fn test_push_multiple_errors() {
        let mut scope = ErrorScope::new(ErrorFilter::All);
        scope.push(GpuError::new(GpuErrorType::Validation, "1".to_string()));
        scope.push(GpuError::new(GpuErrorType::OutOfMemory, "2".to_string()));
        scope.push(GpuError::new(GpuErrorType::Internal, "3".to_string()));
        assert_eq!(scope.error_count(), 3);
    }

    #[test]
    fn test_all_filter_accepts_validation() {
        let mut scope = ErrorScope::new(ErrorFilter::All);
        assert!(scope.push(GpuError::new(GpuErrorType::Validation, String::new())));
    }

    #[test]
    fn test_all_filter_accepts_oom() {
        let mut scope = ErrorScope::new(ErrorFilter::All);
        assert!(scope.push(GpuError::new(GpuErrorType::OutOfMemory, String::new())));
    }

    #[test]
    fn test_all_filter_accepts_internal() {
        let mut scope = ErrorScope::new(ErrorFilter::All);
        assert!(scope.push(GpuError::new(GpuErrorType::Internal, String::new())));
    }

    #[test]
    fn test_all_filter_accepts_lost() {
        let mut scope = ErrorScope::new(ErrorFilter::All);
        assert!(scope.push(GpuError::new(GpuErrorType::Lost, String::new())));
    }

    #[test]
    fn test_errors_returns_slice() {
        let mut scope = ErrorScope::new(ErrorFilter::All);
        scope.push(GpuError::new(GpuErrorType::Validation, "first".to_string()));
        scope.push(GpuError::new(GpuErrorType::Internal, "second".to_string()));

        let errors = scope.errors();
        assert_eq!(errors.len(), 2);
        assert_eq!(errors[0].message, "first");
        assert_eq!(errors[1].message, "second");
    }

    #[test]
    fn test_take_errors_drains() {
        let mut scope = ErrorScope::new(ErrorFilter::All);
        scope.push(GpuError::new(GpuErrorType::Validation, String::new()));
        scope.push(GpuError::new(GpuErrorType::Internal, String::new()));

        let taken = scope.take_errors();
        assert_eq!(taken.len(), 2);
        assert!(!scope.has_errors());
        assert_eq!(scope.error_count(), 0);
    }

    #[test]
    fn test_take_errors_returns_empty_when_none() {
        let mut scope = ErrorScope::new(ErrorFilter::All);
        let taken = scope.take_errors();
        assert!(taken.is_empty());
    }

    #[test]
    fn test_clear() {
        let mut scope = ErrorScope::new(ErrorFilter::All);
        scope.push(GpuError::new(GpuErrorType::Validation, String::new()));
        scope.push(GpuError::new(GpuErrorType::Internal, String::new()));
        scope.clear();
        assert!(!scope.has_errors());
        assert_eq!(scope.error_count(), 0);
    }

    #[test]
    fn test_elapsed() {
        let scope = ErrorScope::new(ErrorFilter::All);
        thread::sleep(Duration::from_millis(5));
        let elapsed = scope.elapsed();
        assert!(elapsed >= Duration::from_millis(4)); // Allow some tolerance
    }

    #[test]
    fn test_most_severe_returns_none_when_empty() {
        let scope = ErrorScope::new(ErrorFilter::All);
        assert!(scope.most_severe().is_none());
    }

    #[test]
    fn test_most_severe_single_error() {
        let mut scope = ErrorScope::new(ErrorFilter::All);
        scope.push(GpuError::new(GpuErrorType::Validation, "only".to_string()));
        let most = scope.most_severe().unwrap();
        assert_eq!(most.message, "only");
    }

    #[test]
    fn test_most_severe_selects_error_over_warning() {
        let mut scope = ErrorScope::new(ErrorFilter::All);
        scope.push(GpuError::new(GpuErrorType::Validation, "warning".to_string())); // Warning
        scope.push(GpuError::new(GpuErrorType::OutOfMemory, "error".to_string())); // Error

        let most = scope.most_severe().unwrap();
        assert_eq!(most.message, "error");
    }

    #[test]
    fn test_most_severe_with_multiple_same_severity() {
        let mut scope = ErrorScope::new(ErrorFilter::All);
        scope.push(GpuError::new(GpuErrorType::OutOfMemory, "first error".to_string()));
        scope.push(GpuError::new(GpuErrorType::Internal, "second error".to_string()));

        let most = scope.most_severe().unwrap();
        // Both are Error severity, one of them should be selected
        assert!(most.message == "first error" || most.message == "second error");
    }

    #[test]
    fn test_display_with_label() {
        let scope = ErrorScope::with_label(ErrorFilter::Validation, "test_scope");
        let display = format!("{}", scope);
        assert!(display.contains("test_scope"));
        assert!(display.contains("Validation"));
    }

    #[test]
    fn test_display_without_label() {
        let scope = ErrorScope::new(ErrorFilter::OutOfMemory);
        let display = format!("{}", scope);
        assert!(display.contains("unnamed"));
        assert!(display.contains("OutOfMemory"));
    }

    #[test]
    fn test_display_shows_error_count() {
        let mut scope = ErrorScope::new(ErrorFilter::All);
        scope.push(GpuError::new(GpuErrorType::Validation, String::new()));
        scope.push(GpuError::new(GpuErrorType::Internal, String::new()));
        let display = format!("{}", scope);
        assert!(display.contains("2"));
    }

    #[test]
    fn test_clone() {
        let mut scope = ErrorScope::with_label(ErrorFilter::All, "original");
        scope.push(GpuError::new(GpuErrorType::Validation, "error".to_string()));

        let cloned = scope.clone();
        assert_eq!(cloned.label, scope.label);
        assert_eq!(cloned.filter, scope.filter);
        assert_eq!(cloned.error_count(), scope.error_count());
    }
}

// ============================================================================
// ErrorCallbackRegistry Tests (20 tests)
// ============================================================================

mod error_callback_registry_tests {
    use super::*;

    #[test]
    fn test_new_is_empty() {
        let registry = ErrorCallbackRegistry::new();
        assert!(registry.is_empty());
        assert_eq!(registry.len(), 0);
    }

    #[test]
    fn test_register_returns_unique_id() {
        let registry = ErrorCallbackRegistry::new();
        let id1 = registry.register(Arc::new(|_| {}));
        let id2 = registry.register(Arc::new(|_| {}));
        let id3 = registry.register(Arc::new(|_| {}));

        assert_ne!(id1, id2);
        assert_ne!(id2, id3);
        assert_ne!(id1, id3);
    }

    #[test]
    fn test_register_increments_len() {
        let registry = ErrorCallbackRegistry::new();
        assert_eq!(registry.len(), 0);
        registry.register(Arc::new(|_| {}));
        assert_eq!(registry.len(), 1);
        registry.register(Arc::new(|_| {}));
        assert_eq!(registry.len(), 2);
    }

    #[test]
    fn test_register_not_empty_after() {
        let registry = ErrorCallbackRegistry::new();
        registry.register(Arc::new(|_| {}));
        assert!(!registry.is_empty());
    }

    #[test]
    fn test_unregister_existing() {
        let registry = ErrorCallbackRegistry::new();
        let id = registry.register(Arc::new(|_| {}));

        assert!(registry.unregister(id));
        assert!(registry.is_empty());
    }

    #[test]
    fn test_unregister_nonexistent() {
        let registry = ErrorCallbackRegistry::new();
        assert!(!registry.unregister(999));
    }

    #[test]
    fn test_unregister_already_removed() {
        let registry = ErrorCallbackRegistry::new();
        let id = registry.register(Arc::new(|_| {}));

        assert!(registry.unregister(id));
        assert!(!registry.unregister(id)); // Second unregister fails
    }

    #[test]
    fn test_invoke_calls_callback() {
        let counter = Arc::new(AtomicU32::new(0));
        let registry = ErrorCallbackRegistry::new();

        let counter_clone = counter.clone();
        registry.register(Arc::new(move |_| {
            counter_clone.fetch_add(1, Ordering::SeqCst);
        }));

        let error = GpuError::new(GpuErrorType::Validation, String::new());
        registry.invoke(&error);

        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn test_invoke_calls_all_callbacks() {
        let counter = Arc::new(AtomicU32::new(0));
        let registry = ErrorCallbackRegistry::new();

        for _ in 0..5 {
            let counter_clone = counter.clone();
            registry.register(Arc::new(move |_| {
                counter_clone.fetch_add(1, Ordering::SeqCst);
            }));
        }

        let error = GpuError::new(GpuErrorType::Validation, String::new());
        let count = registry.invoke(&error);

        assert_eq!(count, 5);
        assert_eq!(counter.load(Ordering::SeqCst), 5);
    }

    #[test]
    fn test_invoke_returns_callback_count() {
        let registry = ErrorCallbackRegistry::new();
        registry.register(Arc::new(|_| {}));
        registry.register(Arc::new(|_| {}));
        registry.register(Arc::new(|_| {}));

        let error = GpuError::new(GpuErrorType::Validation, String::new());
        assert_eq!(registry.invoke(&error), 3);
    }

    #[test]
    fn test_invoke_with_no_callbacks() {
        let registry = ErrorCallbackRegistry::new();
        let error = GpuError::new(GpuErrorType::Validation, String::new());
        assert_eq!(registry.invoke(&error), 0);
    }

    #[test]
    fn test_invoke_passes_error_to_callback() {
        let captured_type = Arc::new(Mutex::new(None));
        let registry = ErrorCallbackRegistry::new();

        let captured_clone = captured_type.clone();
        registry.register(Arc::new(move |error| {
            *captured_clone.lock().unwrap() = Some(error.error_type);
        }));

        let error = GpuError::new(GpuErrorType::OutOfMemory, String::new());
        registry.invoke(&error);

        assert_eq!(*captured_type.lock().unwrap(), Some(GpuErrorType::OutOfMemory));
    }

    #[test]
    fn test_clear() {
        let registry = ErrorCallbackRegistry::new();
        registry.register(Arc::new(|_| {}));
        registry.register(Arc::new(|_| {}));

        registry.clear();

        assert!(registry.is_empty());
        assert_eq!(registry.len(), 0);
    }

    #[test]
    fn test_default() {
        let registry = ErrorCallbackRegistry::default();
        assert!(registry.is_empty());
    }

    #[test]
    fn test_debug_trait() {
        let registry = ErrorCallbackRegistry::new();
        registry.register(Arc::new(|_| {}));

        let debug_str = format!("{:?}", registry);
        assert!(debug_str.contains("ErrorCallbackRegistry"));
        assert!(debug_str.contains("1")); // callback_count
    }

    #[test]
    fn test_clone() {
        let registry = ErrorCallbackRegistry::new();
        let counter = Arc::new(AtomicU32::new(0));

        let counter_clone = counter.clone();
        registry.register(Arc::new(move |_| {
            counter_clone.fetch_add(1, Ordering::SeqCst);
        }));

        let cloned = registry.clone();

        // Both should share the same underlying storage
        let error = GpuError::new(GpuErrorType::Validation, String::new());
        cloned.invoke(&error);

        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn test_thread_safety_register() {
        let registry = Arc::new(ErrorCallbackRegistry::new());
        let mut handles = vec![];

        for _ in 0..10 {
            let reg = registry.clone();
            handles.push(thread::spawn(move || {
                for _ in 0..10 {
                    reg.register(Arc::new(|_| {}));
                }
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }

        assert_eq!(registry.len(), 100);
    }

    #[test]
    fn test_thread_safety_invoke() {
        let registry = Arc::new(ErrorCallbackRegistry::new());
        let counter = Arc::new(AtomicU64::new(0));

        let counter_clone = counter.clone();
        registry.register(Arc::new(move |_| {
            counter_clone.fetch_add(1, Ordering::SeqCst);
        }));

        let mut handles = vec![];
        for _ in 0..10 {
            let reg = registry.clone();
            handles.push(thread::spawn(move || {
                let error = GpuError::new(GpuErrorType::Validation, String::new());
                for _ in 0..100 {
                    reg.invoke(&error);
                }
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }

        assert_eq!(counter.load(Ordering::SeqCst), 1000);
    }

    #[test]
    fn test_unregister_during_invoke() {
        // This tests that we can safely unregister while iterating
        let registry = ErrorCallbackRegistry::new();
        let id1 = registry.register(Arc::new(|_| {}));
        let _id2 = registry.register(Arc::new(|_| {}));

        // Unregister one callback
        registry.unregister(id1);

        // Should still be able to invoke remaining callbacks
        let error = GpuError::new(GpuErrorType::Validation, String::new());
        let count = registry.invoke(&error);
        assert_eq!(count, 1);
    }
}

// ============================================================================
// DebugUtils Tests (25 tests)
// ============================================================================

mod debug_utils_tests {
    use super::*;

    #[test]
    fn test_new() {
        let debug = DebugUtils::new();
        assert_eq!(debug.scope_depth(), 0);
        assert_eq!(debug.total_errors(), 0);
    }

    #[test]
    fn test_default() {
        let debug = DebugUtils::default();
        assert_eq!(debug.scope_depth(), 0);
    }

    #[test]
    fn test_push_error_scope() {
        let mut debug = DebugUtils::new();
        debug.push_error_scope(ErrorFilter::Validation);
        assert_eq!(debug.scope_depth(), 1);
    }

    #[test]
    fn test_push_multiple_error_scopes() {
        let mut debug = DebugUtils::new();
        debug.push_error_scope(ErrorFilter::Validation);
        debug.push_error_scope(ErrorFilter::OutOfMemory);
        debug.push_error_scope(ErrorFilter::Internal);
        assert_eq!(debug.scope_depth(), 3);
    }

    #[test]
    fn test_pop_error_scope_returns_scope() {
        let mut debug = DebugUtils::new();
        debug.push_error_scope(ErrorFilter::OutOfMemory);

        let scope = debug.pop_error_scope();
        assert!(scope.is_some());
        assert_eq!(scope.unwrap().filter, ErrorFilter::OutOfMemory);
    }

    #[test]
    fn test_pop_error_scope_decrements_depth() {
        let mut debug = DebugUtils::new();
        debug.push_error_scope(ErrorFilter::Validation);
        debug.push_error_scope(ErrorFilter::Internal);

        debug.pop_error_scope();
        assert_eq!(debug.scope_depth(), 1);
        debug.pop_error_scope();
        assert_eq!(debug.scope_depth(), 0);
    }

    #[test]
    fn test_pop_error_scope_empty_returns_none() {
        let mut debug = DebugUtils::new();
        assert!(debug.pop_error_scope().is_none());
    }

    #[test]
    fn test_push_error_scope_labeled() {
        let mut debug = DebugUtils::new();
        debug.push_error_scope_labeled(ErrorFilter::All, "main_render");

        let scope = debug.pop_error_scope().unwrap();
        assert_eq!(scope.label.as_deref(), Some("main_render"));
    }

    #[test]
    fn test_push_error_no_scope() {
        let mut debug = DebugUtils::new();
        let error = GpuError::new(GpuErrorType::Validation, "test".to_string());

        let captured = debug.push_error(error);
        assert!(!captured);
        // Error is still counted
        assert_eq!(debug.total_errors(), 1);
    }

    #[test]
    fn test_push_error_with_scope() {
        let mut debug = DebugUtils::new();
        debug.push_error_scope(ErrorFilter::All);

        let error = GpuError::new(GpuErrorType::Validation, "test".to_string());
        let captured = debug.push_error(error);

        assert!(captured);
        assert_eq!(debug.total_errors(), 1);
    }

    #[test]
    fn test_push_error_filtered_out() {
        let mut debug = DebugUtils::new();
        debug.push_error_scope(ErrorFilter::Validation);

        let error = GpuError::new(GpuErrorType::OutOfMemory, "oom".to_string());
        let captured = debug.push_error(error);

        // Not captured by scope but still counted
        assert!(!captured);
        assert_eq!(debug.total_errors(), 1);
    }

    #[test]
    fn test_push_error_invokes_callbacks() {
        let mut debug = DebugUtils::new();
        let counter = Arc::new(AtomicU32::new(0));

        let counter_clone = counter.clone();
        debug.callbacks_mut().register(Arc::new(move |_| {
            counter_clone.fetch_add(1, Ordering::SeqCst);
        }));

        debug.push_error_scope(ErrorFilter::All);
        debug.push_error(GpuError::new(GpuErrorType::Validation, String::new()));

        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn test_callbacks_accessor() {
        let debug = DebugUtils::new();
        assert_eq!(debug.callbacks().len(), 0);
    }

    #[test]
    fn test_callbacks_mut_accessor() {
        let mut debug = DebugUtils::new();
        debug.callbacks_mut().register(Arc::new(|_| {}));
        assert_eq!(debug.callbacks().len(), 1);
    }

    #[test]
    fn test_total_errors_increments() {
        let mut debug = DebugUtils::new();
        debug.push_error_scope(ErrorFilter::All);

        for i in 0..10 {
            debug.push_error(GpuError::new(GpuErrorType::Validation, format!("{}", i)));
        }

        assert_eq!(debug.total_errors(), 10);
    }

    #[test]
    fn test_set_device_lost_handler() {
        let mut debug = DebugUtils::new();
        let called = Arc::new(AtomicBool::new(false));

        let called_clone = called.clone();
        debug.set_device_lost_handler(move |_| {
            called_clone.store(true, Ordering::SeqCst);
        });

        debug.notify_device_lost(DeviceLostInfo::new(
            DeviceLostReason::DriverError,
            String::new(),
        ));

        assert!(called.load(Ordering::SeqCst));
    }

    #[test]
    fn test_clear_device_lost_handler() {
        let mut debug = DebugUtils::new();
        let called = Arc::new(AtomicBool::new(false));

        let called_clone = called.clone();
        debug.set_device_lost_handler(move |_| {
            called_clone.store(true, Ordering::SeqCst);
        });

        debug.clear_device_lost_handler();

        debug.notify_device_lost(DeviceLostInfo::new(
            DeviceLostReason::DriverError,
            String::new(),
        ));

        assert!(!called.load(Ordering::SeqCst));
    }

    #[test]
    fn test_notify_device_lost_no_handler() {
        let debug = DebugUtils::new();
        // Should not panic
        debug.notify_device_lost(DeviceLostInfo::new(
            DeviceLostReason::Unknown,
            String::new(),
        ));
    }

    #[test]
    fn test_notify_device_lost_passes_info() {
        let mut debug = DebugUtils::new();
        let captured_reason = Arc::new(Mutex::new(None));

        let captured_clone = captured_reason.clone();
        debug.set_device_lost_handler(move |info| {
            *captured_clone.lock().unwrap() = Some(info.reason);
        });

        debug.notify_device_lost(DeviceLostInfo::new(
            DeviceLostReason::DeviceInvalid,
            String::new(),
        ));

        assert_eq!(
            *captured_reason.lock().unwrap(),
            Some(DeviceLostReason::DeviceInvalid)
        );
    }

    #[test]
    fn test_capture_errors_creates_scope() {
        let mut debug = DebugUtils::new();
        {
            let _guard = debug.capture_errors(ErrorFilter::Validation);
            // Cannot access debug.scope_depth() here due to borrow rules
            // but the scope should exist
        }
        // After guard drops, scope should be popped
        assert_eq!(debug.scope_depth(), 0);
    }

    #[test]
    fn test_capture_errors_guard_pops_on_drop() {
        let mut debug = DebugUtils::new();
        debug.push_error_scope(ErrorFilter::All); // Pre-existing scope

        {
            let _guard = debug.capture_errors(ErrorFilter::Validation);
        }

        // Only pre-existing scope should remain
        assert_eq!(debug.scope_depth(), 1);
    }

    #[test]
    fn test_debug_trait() {
        let mut debug = DebugUtils::new();
        debug.push_error_scope(ErrorFilter::Validation);
        debug.push_error(GpuError::new(GpuErrorType::Validation, String::new()));

        let debug_str = format!("{:?}", debug);
        assert!(debug_str.contains("DebugUtils"));
        assert!(debug_str.contains("scope_depth"));
        assert!(debug_str.contains("total_errors"));
    }

    #[test]
    fn test_nested_scopes_lifo_order() {
        let mut debug = DebugUtils::new();

        debug.push_error_scope_labeled(ErrorFilter::Validation, "outer");
        debug.push_error_scope_labeled(ErrorFilter::OutOfMemory, "inner");

        let inner = debug.pop_error_scope().unwrap();
        assert_eq!(inner.label.as_deref(), Some("inner"));

        let outer = debug.pop_error_scope().unwrap();
        assert_eq!(outer.label.as_deref(), Some("outer"));
    }

    #[test]
    fn test_errors_accumulate_in_correct_scope() {
        let mut debug = DebugUtils::new();

        debug.push_error_scope(ErrorFilter::All);
        debug.push_error(GpuError::new(GpuErrorType::Validation, "outer".to_string()));

        debug.push_error_scope(ErrorFilter::All);
        debug.push_error(GpuError::new(GpuErrorType::OutOfMemory, "inner".to_string()));

        let inner_scope = debug.pop_error_scope().unwrap();
        assert_eq!(inner_scope.error_count(), 1);
        assert_eq!(inner_scope.errors()[0].message, "inner");

        let outer_scope = debug.pop_error_scope().unwrap();
        assert_eq!(outer_scope.error_count(), 1);
        assert_eq!(outer_scope.errors()[0].message, "outer");
    }
}

// ============================================================================
// ErrorCaptureGuard Tests (10 tests)
// ============================================================================

mod error_capture_guard_tests {
    use super::*;

    #[test]
    fn test_has_errors_false_initially() {
        let mut debug = DebugUtils::new();
        let guard = debug.capture_errors(ErrorFilter::All);
        assert!(!guard.has_errors());
    }

    #[test]
    fn test_error_count_zero_initially() {
        let mut debug = DebugUtils::new();
        let guard = debug.capture_errors(ErrorFilter::All);
        assert_eq!(guard.error_count(), 0);
    }

    #[test]
    fn test_scope_returns_some() {
        let mut debug = DebugUtils::new();
        let guard = debug.capture_errors(ErrorFilter::Validation);
        assert!(guard.scope().is_some());
    }

    #[test]
    fn test_scope_has_correct_filter() {
        let mut debug = DebugUtils::new();
        let guard = debug.capture_errors(ErrorFilter::OutOfMemory);
        assert_eq!(guard.scope().unwrap().filter, ErrorFilter::OutOfMemory);
    }

    #[test]
    fn test_auto_pops_on_drop() {
        let mut debug = DebugUtils::new();
        assert_eq!(debug.scope_depth(), 0);

        {
            let _guard = debug.capture_errors(ErrorFilter::All);
            // Guard exists, scope should be pushed
        }

        // Guard dropped, scope should be popped
        assert_eq!(debug.scope_depth(), 0);
    }

    #[test]
    fn test_nested_guards() {
        // Due to borrow rules, we cannot nest guards directly
        // Instead, test that multiple sequential guards work
        let mut debug = DebugUtils::new();

        {
            let _outer = debug.capture_errors(ErrorFilter::Validation);
            // Outer guard exists
        }
        // Outer guard dropped

        {
            let _inner = debug.capture_errors(ErrorFilter::OutOfMemory);
            // Inner guard exists
        }
        // Inner guard dropped

        assert_eq!(debug.scope_depth(), 0);
    }

    #[test]
    fn test_finish_returns_none() {
        // The current implementation of finish() returns None
        // as noted in the source code comments
        let mut debug = DebugUtils::new();
        let guard = debug.capture_errors(ErrorFilter::All);
        let result = guard.finish();
        assert!(result.is_none());
    }

    #[test]
    fn test_guard_drops_safely() {
        // Test that guard drops safely in normal circumstances
        let mut debug = DebugUtils::new();

        // Pre-check state
        assert_eq!(debug.scope_depth(), 0);

        {
            let _guard = debug.capture_errors(ErrorFilter::All);
            // Guard exists, scope pushed
        }
        // Guard dropped, scope popped

        // State should be clean
        assert_eq!(debug.scope_depth(), 0);
    }

    #[test]
    fn test_has_errors_after_push() {
        // This is a bit tricky because we can't push errors while guard is held
        // due to borrow rules, but we can test the initial state
        let mut debug = DebugUtils::new();
        let guard = debug.capture_errors(ErrorFilter::All);
        // Initially no errors
        assert!(!guard.has_errors());
        assert_eq!(guard.error_count(), 0);
    }

    #[test]
    fn test_guard_tracks_errors_correctly() {
        // Due to borrow rules, we need a different approach to test error tracking
        let mut debug = DebugUtils::new();

        // Push an error before creating guard
        debug.push_error_scope(ErrorFilter::All);
        debug.push_error(GpuError::new(GpuErrorType::Validation, String::new()));
        debug.pop_error_scope();

        // Now create a new guard and verify it starts fresh
        let guard = debug.capture_errors(ErrorFilter::All);
        assert!(!guard.has_errors());
        assert_eq!(guard.error_count(), 0);
    }
}

// ============================================================================
// Integration Tests (10 tests)
// ============================================================================

mod integration_tests {
    use super::*;

    #[test]
    fn test_full_error_capture_workflow() {
        let mut debug = DebugUtils::new();

        // Set up callback
        let errors_logged = Arc::new(AtomicU32::new(0));
        let errors_logged_clone = errors_logged.clone();
        debug.callbacks_mut().register(Arc::new(move |_| {
            errors_logged_clone.fetch_add(1, Ordering::SeqCst);
        }));

        // Push scope and errors
        debug.push_error_scope(ErrorFilter::All);
        debug.push_error(GpuError::new(GpuErrorType::Validation, "error 1".to_string()));
        debug.push_error(GpuError::new(GpuErrorType::OutOfMemory, "error 2".to_string()));

        // Pop and analyze
        let scope = debug.pop_error_scope().unwrap();
        assert_eq!(scope.error_count(), 2);
        assert_eq!(errors_logged.load(Ordering::SeqCst), 2);

        let most_severe = scope.most_severe().unwrap();
        assert_eq!(most_severe.error_type, GpuErrorType::OutOfMemory);
    }

    #[test]
    fn test_device_lost_handling_workflow() {
        let mut debug = DebugUtils::new();

        let recovery_attempted = Arc::new(AtomicBool::new(false));
        let recovery_clone = recovery_attempted.clone();

        debug.set_device_lost_handler(move |info| {
            if info.should_attempt_recovery() {
                recovery_clone.store(true, Ordering::SeqCst);
            }
        });

        debug.notify_device_lost(DeviceLostInfo::new(
            DeviceLostReason::DriverError,
            "GPU timeout".to_string(),
        ));

        assert!(recovery_attempted.load(Ordering::SeqCst));
    }

    #[test]
    fn test_error_filtering_workflow() {
        let mut debug = DebugUtils::new();

        debug.push_error_scope(ErrorFilter::Validation);

        // Only validation errors should be captured
        debug.push_error(GpuError::new(GpuErrorType::Validation, "val".to_string()));
        debug.push_error(GpuError::new(GpuErrorType::OutOfMemory, "oom".to_string()));
        debug.push_error(GpuError::new(GpuErrorType::Internal, "int".to_string()));
        debug.push_error(GpuError::new(GpuErrorType::Validation, "val2".to_string()));

        let scope = debug.pop_error_scope().unwrap();

        // Only validation errors captured
        assert_eq!(scope.error_count(), 2);
        for error in scope.errors() {
            assert!(error.is_validation());
        }

        // But all errors are counted
        assert_eq!(debug.total_errors(), 4);
    }

    #[test]
    fn test_source_location_tracking() {
        let loc = SourceLocation::here();
        let error = GpuError::new(GpuErrorType::Validation, "test error".to_string())
            .with_source_location(loc);

        let display = format!("{}", error);
        assert!(display.contains("whitebox_debug_utils.rs"));
    }

    #[test]
    fn test_multiple_callbacks() {
        let mut debug = DebugUtils::new();

        let counter1 = Arc::new(AtomicU32::new(0));
        let counter2 = Arc::new(AtomicU32::new(0));
        let counter3 = Arc::new(AtomicU32::new(0));

        let c1 = counter1.clone();
        let c2 = counter2.clone();
        let c3 = counter3.clone();

        debug.callbacks_mut().register(Arc::new(move |_| { c1.fetch_add(1, Ordering::SeqCst); }));
        debug.callbacks_mut().register(Arc::new(move |_| { c2.fetch_add(1, Ordering::SeqCst); }));
        debug.callbacks_mut().register(Arc::new(move |_| { c3.fetch_add(1, Ordering::SeqCst); }));

        debug.push_error_scope(ErrorFilter::All);
        debug.push_error(GpuError::new(GpuErrorType::Validation, String::new()));

        assert_eq!(counter1.load(Ordering::SeqCst), 1);
        assert_eq!(counter2.load(Ordering::SeqCst), 1);
        assert_eq!(counter3.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn test_deep_scope_nesting() {
        let mut debug = DebugUtils::new();

        for i in 0..10 {
            debug.push_error_scope_labeled(ErrorFilter::All, format!("scope_{}", i));
        }

        assert_eq!(debug.scope_depth(), 10);

        for i in (0..10).rev() {
            let scope = debug.pop_error_scope().unwrap();
            assert_eq!(scope.label.as_deref(), Some(format!("scope_{}", i).as_str()));
        }

        assert_eq!(debug.scope_depth(), 0);
    }

    #[test]
    fn test_error_with_all_fields() {
        let loc = SourceLocation::new()
            .with_file("src/renderer.rs")
            .with_line(100)
            .with_column(5)
            .with_function("render_frame");

        let error = GpuError::new(
            GpuErrorType::Validation,
            "Invalid bind group layout".to_string(),
        ).with_source_location(loc);

        assert!(error.is_validation());
        assert!(!error.is_oom());
        assert_eq!(error.severity(), Severity::Warning);

        let display = format!("{}", error);
        assert!(display.contains("Validation"));
        assert!(display.contains("Invalid bind group layout"));
        assert!(display.contains("src/renderer.rs:100:5"));
        assert!(display.contains("render_frame()"));
    }

    #[test]
    fn test_callback_registry_isolation() {
        let mut debug1 = DebugUtils::new();
        let mut debug2 = DebugUtils::new();

        let counter1 = Arc::new(AtomicU32::new(0));
        let counter2 = Arc::new(AtomicU32::new(0));

        let c1 = counter1.clone();
        let c2 = counter2.clone();

        debug1.callbacks_mut().register(Arc::new(move |_| { c1.fetch_add(1, Ordering::SeqCst); }));
        debug2.callbacks_mut().register(Arc::new(move |_| { c2.fetch_add(1, Ordering::SeqCst); }));

        debug1.push_error_scope(ErrorFilter::All);
        debug1.push_error(GpuError::new(GpuErrorType::Validation, String::new()));

        // Only debug1's callback should be invoked
        assert_eq!(counter1.load(Ordering::SeqCst), 1);
        assert_eq!(counter2.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn test_error_scope_elapsed_tracking() {
        let scope = ErrorScope::new(ErrorFilter::All);
        let elapsed1 = scope.elapsed();

        thread::sleep(Duration::from_millis(10));

        let elapsed2 = scope.elapsed();
        assert!(elapsed2 > elapsed1);
        assert!(elapsed2 >= Duration::from_millis(10));
    }

    #[test]
    fn test_device_lost_recovery_decision() {
        // Test that different device lost reasons lead to different recovery decisions
        let recoverable_reasons = [
            DeviceLostReason::Unknown,
            DeviceLostReason::DeviceInvalid,
            DeviceLostReason::DriverError,
        ];

        let non_recoverable_reasons = [
            DeviceLostReason::Destroyed,
        ];

        for reason in recoverable_reasons {
            let info = DeviceLostInfo::new(reason, String::new());
            assert!(info.should_attempt_recovery(), "Expected {:?} to be recoverable", reason);
        }

        for reason in non_recoverable_reasons {
            let info = DeviceLostInfo::new(reason, String::new());
            assert!(!info.should_attempt_recovery(), "Expected {:?} to not be recoverable", reason);
        }
    }
}

use std::sync::Mutex;
