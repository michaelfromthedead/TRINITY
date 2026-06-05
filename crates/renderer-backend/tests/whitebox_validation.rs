//! Whitebox tests for GPU validation layer.
//!
//! Tests all components of the validation module:
//! - ValidationLevel
//! - ValidationFeatures
//! - ValidationSeverity
//! - ValidationMessageType
//! - ValidationObjectType
//! - ValidationObject
//! - ValidationMessage
//! - ValidationCallbackRegistry
//! - ValidationLayer
//! - ValidationScope
//!
//! Target: 150+ tests

use renderer_backend::debug::validation::*;
use renderer_backend::debug::SourceLocation;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

// ============================================================================
// ValidationLevel Tests (35 tests)
// ============================================================================

mod validation_level_tests {
    use super::*;

    // ---- Variant Tests ----

    #[test]
    fn test_disabled_variant() {
        let level = ValidationLevel::Disabled;
        assert!(!level.is_enabled());
        assert_eq!(format!("{:?}", level), "Disabled");
    }

    #[test]
    fn test_basic_variant() {
        let level = ValidationLevel::Basic;
        assert!(level.is_enabled());
        assert_eq!(format!("{:?}", level), "Basic");
    }

    #[test]
    fn test_full_variant() {
        let level = ValidationLevel::Full;
        assert!(level.is_enabled());
        assert_eq!(format!("{:?}", level), "Full");
    }

    #[test]
    fn test_verbose_variant() {
        let level = ValidationLevel::Verbose;
        assert!(level.is_enabled());
        assert_eq!(format!("{:?}", level), "Verbose");
    }

    // ---- from_str Tests ----

    #[test]
    fn test_from_str_disabled_variants() {
        assert_eq!(ValidationLevel::from_str("disabled"), ValidationLevel::Disabled);
        assert_eq!(ValidationLevel::from_str("none"), ValidationLevel::Disabled);
        assert_eq!(ValidationLevel::from_str("off"), ValidationLevel::Disabled);
        assert_eq!(ValidationLevel::from_str("0"), ValidationLevel::Disabled);
        assert_eq!(ValidationLevel::from_str("false"), ValidationLevel::Disabled);
    }

    #[test]
    fn test_from_str_basic_variants() {
        assert_eq!(ValidationLevel::from_str("basic"), ValidationLevel::Basic);
        assert_eq!(ValidationLevel::from_str("min"), ValidationLevel::Basic);
        assert_eq!(ValidationLevel::from_str("1"), ValidationLevel::Basic);
    }

    #[test]
    fn test_from_str_full_variants() {
        assert_eq!(ValidationLevel::from_str("full"), ValidationLevel::Full);
        assert_eq!(ValidationLevel::from_str("on"), ValidationLevel::Full);
        assert_eq!(ValidationLevel::from_str("true"), ValidationLevel::Full);
        assert_eq!(ValidationLevel::from_str("2"), ValidationLevel::Full);
    }

    #[test]
    fn test_from_str_verbose_variants() {
        assert_eq!(ValidationLevel::from_str("verbose"), ValidationLevel::Verbose);
        assert_eq!(ValidationLevel::from_str("max"), ValidationLevel::Verbose);
        assert_eq!(ValidationLevel::from_str("debug"), ValidationLevel::Verbose);
        assert_eq!(ValidationLevel::from_str("3"), ValidationLevel::Verbose);
    }

    #[test]
    fn test_from_str_case_insensitive() {
        assert_eq!(ValidationLevel::from_str("DISABLED"), ValidationLevel::Disabled);
        assert_eq!(ValidationLevel::from_str("BASIC"), ValidationLevel::Basic);
        assert_eq!(ValidationLevel::from_str("FULL"), ValidationLevel::Full);
        assert_eq!(ValidationLevel::from_str("VERBOSE"), ValidationLevel::Verbose);
        assert_eq!(ValidationLevel::from_str("FuLl"), ValidationLevel::Full);
        assert_eq!(ValidationLevel::from_str("VeRbOsE"), ValidationLevel::Verbose);
    }

    #[test]
    fn test_from_str_unknown_defaults_to_basic() {
        assert_eq!(ValidationLevel::from_str("unknown"), ValidationLevel::Basic);
        assert_eq!(ValidationLevel::from_str("invalid"), ValidationLevel::Basic);
        assert_eq!(ValidationLevel::from_str(""), ValidationLevel::Basic);
        assert_eq!(ValidationLevel::from_str("xyz"), ValidationLevel::Basic);
        assert_eq!(ValidationLevel::from_str("4"), ValidationLevel::Basic);
    }

    // ---- is_enabled Tests ----

    #[test]
    fn test_is_enabled_only_disabled_returns_false() {
        assert!(!ValidationLevel::Disabled.is_enabled());
        assert!(ValidationLevel::Basic.is_enabled());
        assert!(ValidationLevel::Full.is_enabled());
        assert!(ValidationLevel::Verbose.is_enabled());
    }

    // ---- severity_threshold Tests ----

    #[test]
    fn test_severity_threshold_disabled() {
        assert_eq!(
            ValidationLevel::Disabled.severity_threshold(),
            ValidationSeverity::Error
        );
    }

    #[test]
    fn test_severity_threshold_basic() {
        assert_eq!(
            ValidationLevel::Basic.severity_threshold(),
            ValidationSeverity::Warning
        );
    }

    #[test]
    fn test_severity_threshold_full() {
        assert_eq!(
            ValidationLevel::Full.severity_threshold(),
            ValidationSeverity::Info
        );
    }

    #[test]
    fn test_severity_threshold_verbose() {
        assert_eq!(
            ValidationLevel::Verbose.severity_threshold(),
            ValidationSeverity::Verbose
        );
    }

    // ---- default_features Tests ----

    #[test]
    fn test_default_features_disabled() {
        let features = ValidationLevel::Disabled.default_features();
        assert!(!features.any_enabled());
        assert_eq!(features.enabled_count(), 0);
    }

    #[test]
    fn test_default_features_basic() {
        let features = ValidationLevel::Basic.default_features();
        assert!(features.shader_validation);
        assert!(features.descriptor_indexing_validation);
        assert!(!features.gpu_based_validation);
        assert!(!features.synchronization_validation);
    }

    #[test]
    fn test_default_features_full() {
        let features = ValidationLevel::Full.default_features();
        assert!(features.shader_validation);
        assert!(features.synchronization_validation);
        assert!(features.best_practices_warnings);
        assert!(!features.gpu_based_validation);
    }

    #[test]
    fn test_default_features_verbose() {
        let features = ValidationLevel::Verbose.default_features();
        assert!(features.gpu_based_validation);
        assert!(features.synchronization_validation);
        assert!(features.shader_validation);
        assert!(features.descriptor_indexing_validation);
        assert!(features.best_practices_warnings);
        assert!(features.printf_to_stdout);
    }

    // ---- Trait Tests ----

    #[test]
    fn test_clone() {
        let level = ValidationLevel::Full;
        let cloned = level.clone();
        assert_eq!(level, cloned);
    }

    #[test]
    fn test_copy() {
        let level = ValidationLevel::Full;
        let copied: ValidationLevel = level;
        assert_eq!(level, copied);
    }

    #[test]
    fn test_debug() {
        assert_eq!(format!("{:?}", ValidationLevel::Disabled), "Disabled");
        assert_eq!(format!("{:?}", ValidationLevel::Basic), "Basic");
        assert_eq!(format!("{:?}", ValidationLevel::Full), "Full");
        assert_eq!(format!("{:?}", ValidationLevel::Verbose), "Verbose");
    }

    #[test]
    fn test_eq() {
        assert_eq!(ValidationLevel::Full, ValidationLevel::Full);
        assert_ne!(ValidationLevel::Full, ValidationLevel::Basic);
    }

    #[test]
    fn test_ord() {
        assert!(ValidationLevel::Disabled < ValidationLevel::Basic);
        assert!(ValidationLevel::Basic < ValidationLevel::Full);
        assert!(ValidationLevel::Full < ValidationLevel::Verbose);
        assert!(ValidationLevel::Disabled <= ValidationLevel::Disabled);
        assert!(ValidationLevel::Verbose >= ValidationLevel::Full);
    }

    #[test]
    fn test_display() {
        assert_eq!(format!("{}", ValidationLevel::Disabled), "Disabled");
        assert_eq!(format!("{}", ValidationLevel::Basic), "Basic");
        assert_eq!(format!("{}", ValidationLevel::Full), "Full");
        assert_eq!(format!("{}", ValidationLevel::Verbose), "Verbose");
    }

    #[test]
    fn test_default() {
        let level = ValidationLevel::default();
        assert_eq!(level, ValidationLevel::Disabled);
    }

    #[test]
    fn test_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(ValidationLevel::Disabled);
        set.insert(ValidationLevel::Basic);
        set.insert(ValidationLevel::Full);
        set.insert(ValidationLevel::Verbose);
        assert_eq!(set.len(), 4);

        // Inserting duplicate should not increase count
        set.insert(ValidationLevel::Full);
        assert_eq!(set.len(), 4);
    }

    // ---- Boundary Tests ----

    #[test]
    fn test_from_str_whitespace_handling() {
        // Note: from_str does not trim whitespace, so these should default to Basic
        assert_eq!(ValidationLevel::from_str(" full"), ValidationLevel::Basic);
        assert_eq!(ValidationLevel::from_str("full "), ValidationLevel::Basic);
    }

    #[test]
    fn test_ordering_transitivity() {
        let levels = [
            ValidationLevel::Disabled,
            ValidationLevel::Basic,
            ValidationLevel::Full,
            ValidationLevel::Verbose,
        ];

        for i in 0..levels.len() {
            for j in (i + 1)..levels.len() {
                assert!(levels[i] < levels[j], "{:?} should be < {:?}", levels[i], levels[j]);
            }
        }
    }
}

// ============================================================================
// ValidationFeatures Tests (25 tests)
// ============================================================================

mod validation_features_tests {
    use super::*;

    // ---- Field Initialization Tests ----

    #[test]
    fn test_all_fields_initialization() {
        let features = ValidationFeatures {
            gpu_based_validation: true,
            synchronization_validation: true,
            shader_validation: true,
            descriptor_indexing_validation: true,
            best_practices_warnings: true,
            printf_to_stdout: true,
        };
        assert!(features.gpu_based_validation);
        assert!(features.synchronization_validation);
        assert!(features.shader_validation);
        assert!(features.descriptor_indexing_validation);
        assert!(features.best_practices_warnings);
        assert!(features.printf_to_stdout);
    }

    #[test]
    fn test_all_fields_disabled() {
        let features = ValidationFeatures {
            gpu_based_validation: false,
            synchronization_validation: false,
            shader_validation: false,
            descriptor_indexing_validation: false,
            best_practices_warnings: false,
            printf_to_stdout: false,
        };
        assert!(!features.any_enabled());
        assert_eq!(features.enabled_count(), 0);
    }

    // ---- default() Tests ----

    #[test]
    fn test_default_values() {
        let features = ValidationFeatures::default();
        assert!(!features.gpu_based_validation);
        assert!(!features.synchronization_validation);
        assert!(features.shader_validation);
        assert!(features.descriptor_indexing_validation);
        // best_practices_warnings depends on debug_assertions
        assert!(!features.printf_to_stdout);
    }

    #[test]
    fn test_new_equals_default() {
        let new_features = ValidationFeatures::new();
        let default_features = ValidationFeatures::default();
        assert_eq!(new_features, default_features);
    }

    // ---- all_enabled() Tests ----

    #[test]
    fn test_all_enabled_sets_all_true() {
        let features = ValidationFeatures::all_enabled();
        assert!(features.gpu_based_validation);
        assert!(features.synchronization_validation);
        assert!(features.shader_validation);
        assert!(features.descriptor_indexing_validation);
        assert!(features.best_practices_warnings);
        assert!(features.printf_to_stdout);
        assert_eq!(features.enabled_count(), 6);
    }

    // ---- for_level() Progression Tests ----

    #[test]
    fn test_for_level_disabled() {
        let features = ValidationFeatures::for_level(ValidationLevel::Disabled);
        assert!(!features.gpu_based_validation);
        assert!(!features.synchronization_validation);
        assert!(!features.shader_validation);
        assert!(!features.descriptor_indexing_validation);
        assert!(!features.best_practices_warnings);
        assert!(!features.printf_to_stdout);
    }

    #[test]
    fn test_for_level_basic() {
        let features = ValidationFeatures::for_level(ValidationLevel::Basic);
        assert!(!features.gpu_based_validation);
        assert!(!features.synchronization_validation);
        assert!(features.shader_validation);
        assert!(features.descriptor_indexing_validation);
        assert!(!features.best_practices_warnings);
        assert!(!features.printf_to_stdout);
    }

    #[test]
    fn test_for_level_full() {
        let features = ValidationFeatures::for_level(ValidationLevel::Full);
        assert!(!features.gpu_based_validation);
        assert!(features.synchronization_validation);
        assert!(features.shader_validation);
        assert!(features.descriptor_indexing_validation);
        assert!(features.best_practices_warnings);
        assert!(!features.printf_to_stdout);
    }

    #[test]
    fn test_for_level_verbose_equals_all_enabled() {
        let verbose = ValidationFeatures::for_level(ValidationLevel::Verbose);
        let all = ValidationFeatures::all_enabled();
        assert_eq!(verbose, all);
    }

    #[test]
    fn test_for_level_progression() {
        let disabled = ValidationFeatures::for_level(ValidationLevel::Disabled);
        let basic = ValidationFeatures::for_level(ValidationLevel::Basic);
        let full = ValidationFeatures::for_level(ValidationLevel::Full);
        let verbose = ValidationFeatures::for_level(ValidationLevel::Verbose);

        assert!(disabled.enabled_count() <= basic.enabled_count());
        assert!(basic.enabled_count() <= full.enabled_count());
        assert!(full.enabled_count() <= verbose.enabled_count());
    }

    // ---- any_enabled() Tests ----

    #[test]
    fn test_any_enabled_none() {
        let features = ValidationFeatures::for_level(ValidationLevel::Disabled);
        assert!(!features.any_enabled());
    }

    #[test]
    fn test_any_enabled_single_field() {
        let features = ValidationFeatures {
            gpu_based_validation: true,
            synchronization_validation: false,
            shader_validation: false,
            descriptor_indexing_validation: false,
            best_practices_warnings: false,
            printf_to_stdout: false,
        };
        assert!(features.any_enabled());
    }

    #[test]
    fn test_any_enabled_all_fields() {
        let features = ValidationFeatures::all_enabled();
        assert!(features.any_enabled());
    }

    // ---- enabled_count() Tests ----

    #[test]
    fn test_enabled_count_zero() {
        let features = ValidationFeatures::for_level(ValidationLevel::Disabled);
        assert_eq!(features.enabled_count(), 0);
    }

    #[test]
    fn test_enabled_count_max() {
        let features = ValidationFeatures::all_enabled();
        assert_eq!(features.enabled_count(), 6);
    }

    #[test]
    fn test_enabled_count_partial() {
        let features = ValidationFeatures {
            gpu_based_validation: true,
            synchronization_validation: false,
            shader_validation: true,
            descriptor_indexing_validation: false,
            best_practices_warnings: true,
            printf_to_stdout: false,
        };
        assert_eq!(features.enabled_count(), 3);
    }

    // ---- Display Tests ----

    #[test]
    fn test_display_none() {
        let features = ValidationFeatures::for_level(ValidationLevel::Disabled);
        assert_eq!(format!("{}", features), "None");
    }

    #[test]
    fn test_display_all_enabled() {
        let features = ValidationFeatures::all_enabled();
        let display = format!("{}", features);
        assert!(display.contains("GPU"));
        assert!(display.contains("Sync"));
        assert!(display.contains("Shader"));
        assert!(display.contains("Descriptor"));
        assert!(display.contains("BestPractices"));
        assert!(display.contains("Printf"));
    }

    #[test]
    fn test_display_partial() {
        let features = ValidationFeatures {
            gpu_based_validation: true,
            synchronization_validation: false,
            shader_validation: true,
            descriptor_indexing_validation: false,
            best_practices_warnings: false,
            printf_to_stdout: false,
        };
        let display = format!("{}", features);
        assert!(display.contains("GPU"));
        assert!(display.contains("Shader"));
        assert!(!display.contains("Sync"));
        assert!(display.contains("+"));
    }

    // ---- Trait Tests ----

    #[test]
    fn test_features_clone() {
        let features = ValidationFeatures::all_enabled();
        let cloned = features.clone();
        assert_eq!(features, cloned);
    }

    #[test]
    fn test_features_copy() {
        let features = ValidationFeatures::all_enabled();
        let copied: ValidationFeatures = features;
        assert_eq!(features, copied);
    }

    #[test]
    fn test_features_eq() {
        let a = ValidationFeatures::all_enabled();
        let b = ValidationFeatures::all_enabled();
        let c = ValidationFeatures::default();
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn test_features_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(ValidationFeatures::all_enabled());
        set.insert(ValidationFeatures::default());
        set.insert(ValidationFeatures::for_level(ValidationLevel::Disabled));
        // Should have at least 2 unique entries
        assert!(set.len() >= 2);
    }

    #[test]
    fn test_features_debug() {
        let features = ValidationFeatures::default();
        let debug = format!("{:?}", features);
        assert!(debug.contains("ValidationFeatures"));
        assert!(debug.contains("gpu_based_validation"));
        assert!(debug.contains("shader_validation"));
    }
}

// ============================================================================
// ValidationSeverity Tests (25 tests)
// ============================================================================

mod validation_severity_tests {
    use super::*;

    // ---- Variant Tests ----

    #[test]
    fn test_verbose_variant() {
        let severity = ValidationSeverity::Verbose;
        assert_eq!(format!("{:?}", severity), "Verbose");
        assert_eq!(severity.as_str(), "VERBOSE");
    }

    #[test]
    fn test_info_variant() {
        let severity = ValidationSeverity::Info;
        assert_eq!(format!("{:?}", severity), "Info");
        assert_eq!(severity.as_str(), "INFO");
    }

    #[test]
    fn test_warning_variant() {
        let severity = ValidationSeverity::Warning;
        assert_eq!(format!("{:?}", severity), "Warning");
        assert_eq!(severity.as_str(), "WARN");
    }

    #[test]
    fn test_error_variant() {
        let severity = ValidationSeverity::Error;
        assert_eq!(format!("{:?}", severity), "Error");
        assert_eq!(severity.as_str(), "ERROR");
    }

    // ---- as_log_level() Tests ----

    #[test]
    fn test_as_log_level_verbose() {
        assert_eq!(ValidationSeverity::Verbose.as_log_level(), log::Level::Trace);
    }

    #[test]
    fn test_as_log_level_info() {
        assert_eq!(ValidationSeverity::Info.as_log_level(), log::Level::Info);
    }

    #[test]
    fn test_as_log_level_warning() {
        assert_eq!(ValidationSeverity::Warning.as_log_level(), log::Level::Warn);
    }

    #[test]
    fn test_as_log_level_error() {
        assert_eq!(ValidationSeverity::Error.as_log_level(), log::Level::Error);
    }

    // ---- should_break() Tests ----

    #[test]
    fn test_should_break_only_error() {
        assert!(!ValidationSeverity::Verbose.should_break());
        assert!(!ValidationSeverity::Info.should_break());
        assert!(!ValidationSeverity::Warning.should_break());
        assert!(ValidationSeverity::Error.should_break());
    }

    // ---- meets_threshold() Tests ----

    #[test]
    fn test_meets_threshold_error() {
        assert!(ValidationSeverity::Error.meets_threshold(ValidationSeverity::Error));
        assert!(ValidationSeverity::Error.meets_threshold(ValidationSeverity::Warning));
        assert!(ValidationSeverity::Error.meets_threshold(ValidationSeverity::Info));
        assert!(ValidationSeverity::Error.meets_threshold(ValidationSeverity::Verbose));
    }

    #[test]
    fn test_meets_threshold_warning() {
        assert!(!ValidationSeverity::Warning.meets_threshold(ValidationSeverity::Error));
        assert!(ValidationSeverity::Warning.meets_threshold(ValidationSeverity::Warning));
        assert!(ValidationSeverity::Warning.meets_threshold(ValidationSeverity::Info));
        assert!(ValidationSeverity::Warning.meets_threshold(ValidationSeverity::Verbose));
    }

    #[test]
    fn test_meets_threshold_info() {
        assert!(!ValidationSeverity::Info.meets_threshold(ValidationSeverity::Error));
        assert!(!ValidationSeverity::Info.meets_threshold(ValidationSeverity::Warning));
        assert!(ValidationSeverity::Info.meets_threshold(ValidationSeverity::Info));
        assert!(ValidationSeverity::Info.meets_threshold(ValidationSeverity::Verbose));
    }

    #[test]
    fn test_meets_threshold_verbose() {
        assert!(!ValidationSeverity::Verbose.meets_threshold(ValidationSeverity::Error));
        assert!(!ValidationSeverity::Verbose.meets_threshold(ValidationSeverity::Warning));
        assert!(!ValidationSeverity::Verbose.meets_threshold(ValidationSeverity::Info));
        assert!(ValidationSeverity::Verbose.meets_threshold(ValidationSeverity::Verbose));
    }

    // ---- Ordering Tests ----

    #[test]
    fn test_ordering_verbose_lowest() {
        assert!(ValidationSeverity::Verbose < ValidationSeverity::Info);
        assert!(ValidationSeverity::Verbose < ValidationSeverity::Warning);
        assert!(ValidationSeverity::Verbose < ValidationSeverity::Error);
    }

    #[test]
    fn test_ordering_info() {
        assert!(ValidationSeverity::Info > ValidationSeverity::Verbose);
        assert!(ValidationSeverity::Info < ValidationSeverity::Warning);
        assert!(ValidationSeverity::Info < ValidationSeverity::Error);
    }

    #[test]
    fn test_ordering_warning() {
        assert!(ValidationSeverity::Warning > ValidationSeverity::Verbose);
        assert!(ValidationSeverity::Warning > ValidationSeverity::Info);
        assert!(ValidationSeverity::Warning < ValidationSeverity::Error);
    }

    #[test]
    fn test_ordering_error_highest() {
        assert!(ValidationSeverity::Error > ValidationSeverity::Verbose);
        assert!(ValidationSeverity::Error > ValidationSeverity::Info);
        assert!(ValidationSeverity::Error > ValidationSeverity::Warning);
    }

    #[test]
    fn test_ordering_self_equality() {
        assert!(ValidationSeverity::Verbose <= ValidationSeverity::Verbose);
        assert!(ValidationSeverity::Error >= ValidationSeverity::Error);
    }

    // ---- Trait Tests ----

    #[test]
    fn test_severity_clone() {
        let severity = ValidationSeverity::Warning;
        let cloned = severity.clone();
        assert_eq!(severity, cloned);
    }

    #[test]
    fn test_severity_copy() {
        let severity = ValidationSeverity::Error;
        let copied: ValidationSeverity = severity;
        assert_eq!(severity, copied);
    }

    #[test]
    fn test_severity_default() {
        let severity = ValidationSeverity::default();
        assert_eq!(severity, ValidationSeverity::Info);
    }

    #[test]
    fn test_severity_display() {
        assert_eq!(format!("{}", ValidationSeverity::Verbose), "VERBOSE");
        assert_eq!(format!("{}", ValidationSeverity::Info), "INFO");
        assert_eq!(format!("{}", ValidationSeverity::Warning), "WARN");
        assert_eq!(format!("{}", ValidationSeverity::Error), "ERROR");
    }

    #[test]
    fn test_severity_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(ValidationSeverity::Verbose);
        set.insert(ValidationSeverity::Info);
        set.insert(ValidationSeverity::Warning);
        set.insert(ValidationSeverity::Error);
        assert_eq!(set.len(), 4);
    }

    #[test]
    fn test_severity_ordering_transitivity() {
        let severities = [
            ValidationSeverity::Verbose,
            ValidationSeverity::Info,
            ValidationSeverity::Warning,
            ValidationSeverity::Error,
        ];

        for i in 0..severities.len() {
            for j in (i + 1)..severities.len() {
                assert!(
                    severities[i] < severities[j],
                    "{:?} should be < {:?}",
                    severities[i],
                    severities[j]
                );
            }
        }
    }
}

// ============================================================================
// ValidationMessageType Tests (15 tests)
// ============================================================================

mod validation_message_type_tests {
    use super::*;

    // ---- Variant Tests ----

    #[test]
    fn test_general_variant() {
        let msg_type = ValidationMessageType::General;
        assert_eq!(format!("{:?}", msg_type), "General");
        assert_eq!(format!("{}", msg_type), "General");
    }

    #[test]
    fn test_validation_variant() {
        let msg_type = ValidationMessageType::Validation;
        assert_eq!(format!("{:?}", msg_type), "Validation");
        assert_eq!(format!("{}", msg_type), "Validation");
    }

    #[test]
    fn test_performance_variant() {
        let msg_type = ValidationMessageType::Performance;
        assert_eq!(format!("{:?}", msg_type), "Performance");
        assert_eq!(format!("{}", msg_type), "Performance");
    }

    #[test]
    fn test_debug_marker_variant() {
        let msg_type = ValidationMessageType::DebugMarker;
        assert_eq!(format!("{:?}", msg_type), "DebugMarker");
        assert_eq!(format!("{}", msg_type), "DebugMarker");
    }

    // ---- is_error() Tests ----

    #[test]
    fn test_is_error_only_validation() {
        assert!(!ValidationMessageType::General.is_error());
        assert!(ValidationMessageType::Validation.is_error());
        assert!(!ValidationMessageType::Performance.is_error());
        assert!(!ValidationMessageType::DebugMarker.is_error());
    }

    // ---- is_performance() Tests ----

    #[test]
    fn test_is_performance_only_performance() {
        assert!(!ValidationMessageType::General.is_performance());
        assert!(!ValidationMessageType::Validation.is_performance());
        assert!(ValidationMessageType::Performance.is_performance());
        assert!(!ValidationMessageType::DebugMarker.is_performance());
    }

    // ---- default_severity() Tests ----

    #[test]
    fn test_default_severity_general() {
        assert_eq!(
            ValidationMessageType::General.default_severity(),
            ValidationSeverity::Info
        );
    }

    #[test]
    fn test_default_severity_validation() {
        assert_eq!(
            ValidationMessageType::Validation.default_severity(),
            ValidationSeverity::Error
        );
    }

    #[test]
    fn test_default_severity_performance() {
        assert_eq!(
            ValidationMessageType::Performance.default_severity(),
            ValidationSeverity::Warning
        );
    }

    #[test]
    fn test_default_severity_debug_marker() {
        assert_eq!(
            ValidationMessageType::DebugMarker.default_severity(),
            ValidationSeverity::Verbose
        );
    }

    // ---- Trait Tests ----

    #[test]
    fn test_message_type_clone() {
        let msg_type = ValidationMessageType::Validation;
        let cloned = msg_type.clone();
        assert_eq!(msg_type, cloned);
    }

    #[test]
    fn test_message_type_copy() {
        let msg_type = ValidationMessageType::Performance;
        let copied: ValidationMessageType = msg_type;
        assert_eq!(msg_type, copied);
    }

    #[test]
    fn test_message_type_eq() {
        assert_eq!(ValidationMessageType::General, ValidationMessageType::General);
        assert_ne!(ValidationMessageType::General, ValidationMessageType::Validation);
    }

    #[test]
    fn test_message_type_default() {
        let default = ValidationMessageType::default();
        assert_eq!(default, ValidationMessageType::General);
    }

    #[test]
    fn test_message_type_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(ValidationMessageType::General);
        set.insert(ValidationMessageType::Validation);
        set.insert(ValidationMessageType::Performance);
        set.insert(ValidationMessageType::DebugMarker);
        assert_eq!(set.len(), 4);
    }
}

// ============================================================================
// ValidationObjectType Tests (30 tests)
// ============================================================================

mod validation_object_type_tests {
    use super::*;

    // ---- All 20+ Object Types ----

    #[test]
    fn test_unknown_type() {
        let obj_type = ValidationObjectType::Unknown;
        assert_eq!(format!("{}", obj_type), "Unknown");
        assert!(!obj_type.is_pipeline());
        assert!(!obj_type.is_resource());
        assert!(!obj_type.is_binding());
        assert!(!obj_type.is_command());
    }

    #[test]
    fn test_buffer_type() {
        let obj_type = ValidationObjectType::Buffer;
        assert_eq!(format!("{}", obj_type), "Buffer");
        assert!(obj_type.is_resource());
        assert!(!obj_type.is_pipeline());
    }

    #[test]
    fn test_texture_type() {
        let obj_type = ValidationObjectType::Texture;
        assert_eq!(format!("{}", obj_type), "Texture");
        assert!(obj_type.is_resource());
    }

    #[test]
    fn test_texture_view_type() {
        let obj_type = ValidationObjectType::TextureView;
        assert_eq!(format!("{}", obj_type), "TextureView");
        assert!(obj_type.is_resource());
    }

    #[test]
    fn test_sampler_type() {
        let obj_type = ValidationObjectType::Sampler;
        assert_eq!(format!("{}", obj_type), "Sampler");
        assert!(obj_type.is_resource());
    }

    #[test]
    fn test_bind_group_type() {
        let obj_type = ValidationObjectType::BindGroup;
        assert_eq!(format!("{}", obj_type), "BindGroup");
        assert!(obj_type.is_binding());
        assert!(!obj_type.is_resource());
    }

    #[test]
    fn test_bind_group_layout_type() {
        let obj_type = ValidationObjectType::BindGroupLayout;
        assert_eq!(format!("{}", obj_type), "BindGroupLayout");
        assert!(obj_type.is_binding());
    }

    #[test]
    fn test_render_pipeline_type() {
        let obj_type = ValidationObjectType::RenderPipeline;
        assert_eq!(format!("{}", obj_type), "RenderPipeline");
        assert!(obj_type.is_pipeline());
        assert!(!obj_type.is_resource());
    }

    #[test]
    fn test_compute_pipeline_type() {
        let obj_type = ValidationObjectType::ComputePipeline;
        assert_eq!(format!("{}", obj_type), "ComputePipeline");
        assert!(obj_type.is_pipeline());
    }

    #[test]
    fn test_pipeline_layout_type() {
        let obj_type = ValidationObjectType::PipelineLayout;
        assert_eq!(format!("{}", obj_type), "PipelineLayout");
        assert!(obj_type.is_pipeline());
    }

    #[test]
    fn test_shader_module_type() {
        let obj_type = ValidationObjectType::ShaderModule;
        assert_eq!(format!("{}", obj_type), "ShaderModule");
        assert!(!obj_type.is_pipeline());
        assert!(!obj_type.is_resource());
    }

    #[test]
    fn test_command_buffer_type() {
        let obj_type = ValidationObjectType::CommandBuffer;
        assert_eq!(format!("{}", obj_type), "CommandBuffer");
        assert!(obj_type.is_command());
    }

    #[test]
    fn test_command_encoder_type() {
        let obj_type = ValidationObjectType::CommandEncoder;
        assert_eq!(format!("{}", obj_type), "CommandEncoder");
        assert!(obj_type.is_command());
    }

    #[test]
    fn test_render_pass_type() {
        let obj_type = ValidationObjectType::RenderPass;
        assert_eq!(format!("{}", obj_type), "RenderPass");
        assert!(obj_type.is_command());
    }

    #[test]
    fn test_compute_pass_type() {
        let obj_type = ValidationObjectType::ComputePass;
        assert_eq!(format!("{}", obj_type), "ComputePass");
        assert!(obj_type.is_command());
    }

    #[test]
    fn test_query_set_type() {
        let obj_type = ValidationObjectType::QuerySet;
        assert_eq!(format!("{}", obj_type), "QuerySet");
        assert!(!obj_type.is_command());
    }

    #[test]
    fn test_surface_type() {
        let obj_type = ValidationObjectType::Surface;
        assert_eq!(format!("{}", obj_type), "Surface");
    }

    #[test]
    fn test_device_type() {
        let obj_type = ValidationObjectType::Device;
        assert_eq!(format!("{}", obj_type), "Device");
    }

    #[test]
    fn test_queue_type() {
        let obj_type = ValidationObjectType::Queue;
        assert_eq!(format!("{}", obj_type), "Queue");
        assert!(!obj_type.is_command());
    }

    #[test]
    fn test_adapter_type() {
        let obj_type = ValidationObjectType::Adapter;
        assert_eq!(format!("{}", obj_type), "Adapter");
    }

    #[test]
    fn test_instance_type() {
        let obj_type = ValidationObjectType::Instance;
        assert_eq!(format!("{}", obj_type), "Instance");
    }

    // ---- Category Helper Tests ----

    #[test]
    fn test_is_pipeline_exhaustive() {
        assert!(ValidationObjectType::RenderPipeline.is_pipeline());
        assert!(ValidationObjectType::ComputePipeline.is_pipeline());
        assert!(ValidationObjectType::PipelineLayout.is_pipeline());
        // Non-pipeline types
        assert!(!ValidationObjectType::Buffer.is_pipeline());
        assert!(!ValidationObjectType::ShaderModule.is_pipeline());
    }

    #[test]
    fn test_is_resource_exhaustive() {
        assert!(ValidationObjectType::Buffer.is_resource());
        assert!(ValidationObjectType::Texture.is_resource());
        assert!(ValidationObjectType::TextureView.is_resource());
        assert!(ValidationObjectType::Sampler.is_resource());
        // Non-resource types
        assert!(!ValidationObjectType::BindGroup.is_resource());
        assert!(!ValidationObjectType::RenderPipeline.is_resource());
    }

    #[test]
    fn test_is_binding_exhaustive() {
        assert!(ValidationObjectType::BindGroup.is_binding());
        assert!(ValidationObjectType::BindGroupLayout.is_binding());
        // Non-binding types
        assert!(!ValidationObjectType::Buffer.is_binding());
        assert!(!ValidationObjectType::Sampler.is_binding());
    }

    #[test]
    fn test_is_command_exhaustive() {
        assert!(ValidationObjectType::CommandBuffer.is_command());
        assert!(ValidationObjectType::CommandEncoder.is_command());
        assert!(ValidationObjectType::RenderPass.is_command());
        assert!(ValidationObjectType::ComputePass.is_command());
        // Non-command types
        assert!(!ValidationObjectType::Queue.is_command());
        assert!(!ValidationObjectType::Device.is_command());
    }

    // ---- Trait Tests ----

    #[test]
    fn test_object_type_default() {
        let default = ValidationObjectType::default();
        assert_eq!(default, ValidationObjectType::Unknown);
    }

    #[test]
    fn test_object_type_clone() {
        let obj_type = ValidationObjectType::Buffer;
        let cloned = obj_type.clone();
        assert_eq!(obj_type, cloned);
    }

    #[test]
    fn test_object_type_copy() {
        let obj_type = ValidationObjectType::Texture;
        let copied: ValidationObjectType = obj_type;
        assert_eq!(obj_type, copied);
    }

    #[test]
    fn test_object_type_hash() {
        use std::collections::HashSet;
        let types = [
            ValidationObjectType::Unknown,
            ValidationObjectType::Buffer,
            ValidationObjectType::Texture,
            ValidationObjectType::RenderPipeline,
            ValidationObjectType::BindGroup,
            ValidationObjectType::CommandBuffer,
        ];
        let set: HashSet<_> = types.iter().collect();
        assert_eq!(set.len(), types.len());
    }

    #[test]
    fn test_all_categories_mutually_exclusive() {
        // Test that no object type belongs to multiple categories
        let all_types = [
            ValidationObjectType::Unknown,
            ValidationObjectType::Buffer,
            ValidationObjectType::Texture,
            ValidationObjectType::TextureView,
            ValidationObjectType::Sampler,
            ValidationObjectType::BindGroup,
            ValidationObjectType::BindGroupLayout,
            ValidationObjectType::RenderPipeline,
            ValidationObjectType::ComputePipeline,
            ValidationObjectType::PipelineLayout,
            ValidationObjectType::ShaderModule,
            ValidationObjectType::CommandBuffer,
            ValidationObjectType::CommandEncoder,
            ValidationObjectType::RenderPass,
            ValidationObjectType::ComputePass,
            ValidationObjectType::QuerySet,
            ValidationObjectType::Surface,
            ValidationObjectType::Device,
            ValidationObjectType::Queue,
            ValidationObjectType::Adapter,
            ValidationObjectType::Instance,
        ];

        for obj_type in all_types.iter() {
            let category_count = [
                obj_type.is_pipeline(),
                obj_type.is_resource(),
                obj_type.is_binding(),
                obj_type.is_command(),
            ]
            .iter()
            .filter(|&&b| b)
            .count();

            assert!(
                category_count <= 1,
                "{:?} belongs to multiple categories",
                obj_type
            );
        }
    }
}

// ============================================================================
// ValidationObject Tests (15 tests)
// ============================================================================

mod validation_object_tests {
    use super::*;

    // ---- Construction Tests ----

    #[test]
    fn test_new_basic() {
        let obj = ValidationObject::new(ValidationObjectType::Buffer, 0x1234);
        assert_eq!(obj.object_type, ValidationObjectType::Buffer);
        assert_eq!(obj.handle, 0x1234);
        assert!(obj.name.is_none());
        assert!(!obj.has_name());
    }

    #[test]
    fn test_new_with_zero_handle() {
        let obj = ValidationObject::new(ValidationObjectType::Texture, 0);
        assert_eq!(obj.handle, 0);
    }

    #[test]
    fn test_new_with_max_handle() {
        let obj = ValidationObject::new(ValidationObjectType::Buffer, u64::MAX);
        assert_eq!(obj.handle, u64::MAX);
    }

    #[test]
    fn test_unknown_constructor() {
        let obj = ValidationObject::unknown(0xDEADBEEF);
        assert_eq!(obj.object_type, ValidationObjectType::Unknown);
        assert_eq!(obj.handle, 0xDEADBEEF);
        assert!(obj.name.is_none());
    }

    // ---- with_name Tests ----

    #[test]
    fn test_with_name_string() {
        let obj = ValidationObject::new(ValidationObjectType::Texture, 0x5678)
            .with_name("ShadowMap");
        assert_eq!(obj.name.as_deref(), Some("ShadowMap"));
        assert!(obj.has_name());
    }

    #[test]
    fn test_with_name_empty() {
        let obj = ValidationObject::new(ValidationObjectType::Buffer, 0x1)
            .with_name("");
        assert_eq!(obj.name.as_deref(), Some(""));
        assert!(obj.has_name());
    }

    #[test]
    fn test_with_name_string_owned() {
        let name = String::from("VertexBuffer");
        let obj = ValidationObject::new(ValidationObjectType::Buffer, 0x1)
            .with_name(name);
        assert_eq!(obj.name.as_deref(), Some("VertexBuffer"));
    }

    #[test]
    fn test_set_name() {
        let mut obj = ValidationObject::new(ValidationObjectType::Buffer, 0x1);
        assert!(!obj.has_name());

        obj.set_name("NewName");
        assert!(obj.has_name());
        assert_eq!(obj.name.as_deref(), Some("NewName"));
    }

    // ---- display_string Tests ----

    #[test]
    fn test_display_string_unnamed() {
        let obj = ValidationObject::new(ValidationObjectType::Buffer, 0x1234);
        let display = obj.display_string();
        assert_eq!(display, "Buffer(0x1234)");
    }

    #[test]
    fn test_display_string_named() {
        let obj = ValidationObject::new(ValidationObjectType::Texture, 0xABCD)
            .with_name("Albedo");
        let display = obj.display_string();
        assert_eq!(display, "Texture(0xabcd): Albedo");
    }

    #[test]
    fn test_display_trait() {
        let obj = ValidationObject::new(ValidationObjectType::Sampler, 0xFF)
            .with_name("LinearSampler");
        let display = format!("{}", obj);
        assert!(display.contains("Sampler"));
        assert!(display.contains("0xff"));
        assert!(display.contains("LinearSampler"));
    }

    // ---- Trait Tests ----

    #[test]
    fn test_validation_object_clone() {
        let obj = ValidationObject::new(ValidationObjectType::Buffer, 0x1234)
            .with_name("Test");
        let cloned = obj.clone();
        assert_eq!(obj, cloned);
    }

    #[test]
    fn test_validation_object_eq() {
        let obj1 = ValidationObject::new(ValidationObjectType::Buffer, 0x1234);
        let obj2 = ValidationObject::new(ValidationObjectType::Buffer, 0x1234);
        let obj3 = ValidationObject::new(ValidationObjectType::Buffer, 0x5678);

        assert_eq!(obj1, obj2);
        assert_ne!(obj1, obj3);
    }

    #[test]
    fn test_validation_object_debug() {
        let obj = ValidationObject::new(ValidationObjectType::Buffer, 0x1234)
            .with_name("Debug");
        let debug = format!("{:?}", obj);
        assert!(debug.contains("ValidationObject"));
        assert!(debug.contains("Buffer"));
        assert!(debug.contains("Debug"));
    }

    #[test]
    fn test_validation_object_different_types_not_equal() {
        let obj1 = ValidationObject::new(ValidationObjectType::Buffer, 0x1234);
        let obj2 = ValidationObject::new(ValidationObjectType::Texture, 0x1234);
        assert_ne!(obj1, obj2);
    }
}

// ============================================================================
// ValidationMessage Tests (25 tests)
// ============================================================================

mod validation_message_tests {
    use super::*;

    // ---- Factory Method Tests ----

    #[test]
    fn test_new_basic() {
        let msg = ValidationMessage::new(
            ValidationSeverity::Warning,
            ValidationMessageType::Performance,
            "Test message",
        );
        assert_eq!(msg.severity, ValidationSeverity::Warning);
        assert_eq!(msg.message_type, ValidationMessageType::Performance);
        assert_eq!(msg.message, "Test message");
        assert!(msg.message_id.is_none());
        assert!(msg.objects.is_empty());
        assert!(msg.location.is_none());
    }

    #[test]
    fn test_error_factory() {
        let msg = ValidationMessage::error("Error message");
        assert_eq!(msg.severity, ValidationSeverity::Error);
        assert_eq!(msg.message_type, ValidationMessageType::Validation);
        assert_eq!(msg.message, "Error message");
    }

    #[test]
    fn test_warning_factory() {
        let msg = ValidationMessage::warning("Warning message");
        assert_eq!(msg.severity, ValidationSeverity::Warning);
        assert_eq!(msg.message_type, ValidationMessageType::Validation);
        assert_eq!(msg.message, "Warning message");
    }

    #[test]
    fn test_performance_factory() {
        let msg = ValidationMessage::performance("Perf message");
        assert_eq!(msg.severity, ValidationSeverity::Warning);
        assert_eq!(msg.message_type, ValidationMessageType::Performance);
        assert_eq!(msg.message, "Perf message");
    }

    #[test]
    fn test_info_factory() {
        let msg = ValidationMessage::info("Info message");
        assert_eq!(msg.severity, ValidationSeverity::Info);
        assert_eq!(msg.message_type, ValidationMessageType::General);
        assert_eq!(msg.message, "Info message");
    }

    // ---- Builder Pattern Tests ----

    #[test]
    fn test_with_id() {
        let msg = ValidationMessage::error("Test").with_id(42);
        assert_eq!(msg.message_id, Some(42));
    }

    #[test]
    fn test_with_id_negative() {
        let msg = ValidationMessage::error("Test").with_id(-1);
        assert_eq!(msg.message_id, Some(-1));
    }

    #[test]
    fn test_with_object() {
        let obj = ValidationObject::new(ValidationObjectType::Buffer, 0x1);
        let msg = ValidationMessage::error("Test").with_object(obj);
        assert_eq!(msg.objects.len(), 1);
        assert_eq!(msg.objects[0].object_type, ValidationObjectType::Buffer);
    }

    #[test]
    fn test_with_objects() {
        let objects = vec![
            ValidationObject::new(ValidationObjectType::Buffer, 0x1),
            ValidationObject::new(ValidationObjectType::Texture, 0x2),
        ];
        let msg = ValidationMessage::error("Test").with_objects(objects);
        assert_eq!(msg.objects.len(), 2);
    }

    #[test]
    fn test_with_object_chaining() {
        let msg = ValidationMessage::error("Test")
            .with_object(ValidationObject::new(ValidationObjectType::Buffer, 0x1))
            .with_objects(vec![
                ValidationObject::new(ValidationObjectType::Texture, 0x2),
                ValidationObject::new(ValidationObjectType::Sampler, 0x3),
            ]);
        assert_eq!(msg.objects.len(), 3);
    }

    #[test]
    fn test_with_location() {
        let loc = SourceLocation::new()
            .with_file("test.rs")
            .with_line(42);
        let msg = ValidationMessage::error("Test").with_location(loc);
        assert!(msg.location.is_some());
        assert_eq!(msg.location.as_ref().unwrap().line, Some(42));
    }

    #[test]
    fn test_full_builder_chain() {
        let msg = ValidationMessage::error("Full test")
            .with_id(100)
            .with_object(ValidationObject::new(ValidationObjectType::Buffer, 0x1).with_name("VB"))
            .with_location(SourceLocation::new().with_file("shader.wgsl").with_line(10));

        assert_eq!(msg.message_id, Some(100));
        assert_eq!(msg.objects.len(), 1);
        assert!(msg.location.is_some());
    }

    // ---- Query Method Tests ----

    #[test]
    fn test_is_error() {
        let error = ValidationMessage::error("Error");
        let warning = ValidationMessage::warning("Warning");

        assert!(error.is_error());
        assert!(!warning.is_error());
    }

    #[test]
    fn test_is_warning() {
        let error = ValidationMessage::error("Error");
        let warning = ValidationMessage::warning("Warning");

        assert!(!error.is_warning());
        assert!(warning.is_warning());
    }

    #[test]
    fn test_meets_threshold() {
        let error = ValidationMessage::error("Error");
        let warning = ValidationMessage::warning("Warning");
        let info = ValidationMessage::info("Info");

        assert!(error.meets_threshold(ValidationSeverity::Warning));
        assert!(warning.meets_threshold(ValidationSeverity::Warning));
        assert!(!info.meets_threshold(ValidationSeverity::Warning));
    }

    // ---- Timestamp Tests ----

    #[test]
    fn test_timestamp_exists() {
        let msg = ValidationMessage::info("Test");
        // Elapsed should be very small
        assert!(msg.elapsed() < Duration::from_secs(1));
    }

    #[test]
    fn test_elapsed_increases() {
        let msg = ValidationMessage::info("Test");
        let elapsed1 = msg.elapsed();
        thread::sleep(Duration::from_millis(10));
        let elapsed2 = msg.elapsed();
        assert!(elapsed2 >= elapsed1);
    }

    // ---- Display Tests ----

    #[test]
    fn test_display_basic() {
        let msg = ValidationMessage::error("Test error");
        let display = format!("{}", msg);
        assert!(display.contains("ERROR"));
        assert!(display.contains("Validation"));
        assert!(display.contains("Test error"));
    }

    #[test]
    fn test_display_format() {
        let msg = ValidationMessage::new(
            ValidationSeverity::Warning,
            ValidationMessageType::Performance,
            "Performance hint",
        );
        let display = format!("{}", msg);
        assert!(display.contains("WARN"));
        assert!(display.contains("Performance"));
    }

    // ---- Clone Tests ----

    #[test]
    fn test_message_clone() {
        let msg = ValidationMessage::error("Clone test")
            .with_id(42)
            .with_object(ValidationObject::new(ValidationObjectType::Buffer, 0x1));

        let cloned = msg.clone();
        assert_eq!(cloned.severity, msg.severity);
        assert_eq!(cloned.message, msg.message);
        assert_eq!(cloned.message_id, msg.message_id);
        assert_eq!(cloned.objects.len(), msg.objects.len());
    }

    #[test]
    fn test_debug_format() {
        let msg = ValidationMessage::warning("Debug test");
        let debug = format!("{:?}", msg);
        assert!(debug.contains("ValidationMessage"));
        assert!(debug.contains("Warning"));
    }

    // ---- Edge Cases ----

    #[test]
    fn test_empty_message() {
        let msg = ValidationMessage::info("");
        assert_eq!(msg.message, "");
    }

    #[test]
    fn test_long_message() {
        let long_msg = "A".repeat(10000);
        let msg = ValidationMessage::info(&long_msg);
        assert_eq!(msg.message.len(), 10000);
    }

    #[test]
    fn test_special_characters_in_message() {
        let special = "Test\n\t\r\0\u{1234}";
        let msg = ValidationMessage::info(special);
        assert_eq!(msg.message, special);
    }
}

// ============================================================================
// ValidationCallbackRegistry Tests (12 tests)
// ============================================================================

mod validation_callback_registry_tests {
    use super::*;
    use std::sync::atomic::AtomicBool;

    #[test]
    fn test_new_empty() {
        let registry = ValidationCallbackRegistry::new();
        assert!(registry.is_empty());
        assert_eq!(registry.len(), 0);
    }

    #[test]
    fn test_default_empty() {
        let registry = ValidationCallbackRegistry::default();
        assert!(registry.is_empty());
    }

    #[test]
    fn test_register_single() {
        let registry = ValidationCallbackRegistry::new();
        let index = registry.register(Box::new(|_| {}));
        assert_eq!(index, 0);
        assert_eq!(registry.len(), 1);
        assert!(!registry.is_empty());
    }

    #[test]
    fn test_register_multiple() {
        let registry = ValidationCallbackRegistry::new();
        for i in 0..5 {
            let index = registry.register(Box::new(|_| {}));
            assert_eq!(index, i);
        }
        assert_eq!(registry.len(), 5);
    }

    #[test]
    fn test_invoke_single() {
        let registry = ValidationCallbackRegistry::new();
        let counter = Arc::new(AtomicU64::new(0));
        let counter_clone = counter.clone();

        registry.register(Box::new(move |_| {
            counter_clone.fetch_add(1, Ordering::SeqCst);
        }));

        let msg = ValidationMessage::info("Test");
        registry.invoke(&msg);

        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn test_invoke_multiple() {
        let registry = ValidationCallbackRegistry::new();
        let counter = Arc::new(AtomicU64::new(0));

        for _ in 0..3 {
            let c = counter.clone();
            registry.register(Box::new(move |_| {
                c.fetch_add(1, Ordering::SeqCst);
            }));
        }

        let msg = ValidationMessage::info("Test");
        registry.invoke(&msg);

        assert_eq!(counter.load(Ordering::SeqCst), 3);
    }

    #[test]
    fn test_invoke_with_message_data() {
        let registry = ValidationCallbackRegistry::new();
        let received_error = Arc::new(AtomicBool::new(false));
        let received_clone = received_error.clone();

        registry.register(Box::new(move |msg| {
            if msg.is_error() {
                received_clone.store(true, Ordering::SeqCst);
            }
        }));

        registry.invoke(&ValidationMessage::error("Test error"));
        assert!(received_error.load(Ordering::SeqCst));
    }

    #[test]
    fn test_clear() {
        let registry = ValidationCallbackRegistry::new();
        registry.register(Box::new(|_| {}));
        registry.register(Box::new(|_| {}));
        assert_eq!(registry.len(), 2);

        registry.clear();
        assert!(registry.is_empty());
        assert_eq!(registry.len(), 0);
    }

    #[test]
    fn test_invoke_after_clear() {
        let registry = ValidationCallbackRegistry::new();
        let counter = Arc::new(AtomicU64::new(0));
        let counter_clone = counter.clone();

        registry.register(Box::new(move |_| {
            counter_clone.fetch_add(1, Ordering::SeqCst);
        }));

        registry.clear();
        registry.invoke(&ValidationMessage::info("Test"));

        // Should not increment since callbacks were cleared
        assert_eq!(counter.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn test_debug_format() {
        let registry = ValidationCallbackRegistry::new();
        registry.register(Box::new(|_| {}));
        registry.register(Box::new(|_| {}));

        let debug = format!("{:?}", registry);
        assert!(debug.contains("ValidationCallbackRegistry"));
        assert!(debug.contains("2"));
    }

    #[test]
    fn test_thread_safe_register() {
        let registry = Arc::new(ValidationCallbackRegistry::new());
        let mut handles = vec![];

        for _ in 0..4 {
            let reg = registry.clone();
            handles.push(thread::spawn(move || {
                reg.register(Box::new(|_| {}));
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }

        assert_eq!(registry.len(), 4);
    }

    #[test]
    fn test_thread_safe_invoke() {
        let registry = Arc::new(ValidationCallbackRegistry::new());
        let counter = Arc::new(AtomicU64::new(0));
        let counter_clone = counter.clone();

        registry.register(Box::new(move |_| {
            counter_clone.fetch_add(1, Ordering::SeqCst);
        }));

        let mut handles = vec![];
        for _ in 0..4 {
            let reg = registry.clone();
            handles.push(thread::spawn(move || {
                reg.invoke(&ValidationMessage::info("Test"));
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }

        assert_eq!(counter.load(Ordering::SeqCst), 4);
    }
}

// ============================================================================
// ValidationLayer Tests (25 tests)
// ============================================================================

mod validation_layer_tests {
    use super::*;

    // ---- Construction Tests ----

    #[test]
    fn test_new_with_level() {
        let layer = ValidationLayer::new(ValidationLevel::Full);
        assert_eq!(layer.level(), ValidationLevel::Full);
        assert!(layer.is_enabled());
    }

    #[test]
    fn test_new_disabled() {
        let layer = ValidationLayer::new(ValidationLevel::Disabled);
        assert!(!layer.is_enabled());
    }

    #[test]
    fn test_with_features() {
        let features = ValidationFeatures::all_enabled();
        let layer = ValidationLayer::with_features(ValidationLevel::Full, features);
        assert!(layer.features().gpu_based_validation);
        assert!(layer.features().synchronization_validation);
    }

    #[test]
    fn test_initial_counters_zero() {
        let layer = ValidationLayer::new(ValidationLevel::Full);
        assert_eq!(layer.message_count(), 0);
        assert_eq!(layer.error_count(), 0);
        assert_eq!(layer.warning_count(), 0);
        assert!(!layer.has_errors());
        assert!(!layer.has_warnings());
    }

    // ---- on_message Tests ----

    #[test]
    fn test_on_message_warning() {
        let layer = ValidationLayer::new(ValidationLevel::Full);
        let msg = ValidationMessage::warning("Test warning");
        layer.on_message(&msg);

        assert_eq!(layer.warning_count(), 1);
        assert_eq!(layer.message_count(), 1);
        assert!(!layer.has_errors());
        assert!(layer.has_warnings());
    }

    #[test]
    fn test_on_message_error() {
        let layer = ValidationLayer::new(ValidationLevel::Full);
        let msg = ValidationMessage::error("Test error");
        layer.on_message(&msg);

        assert_eq!(layer.error_count(), 1);
        assert_eq!(layer.message_count(), 1);
        assert!(layer.has_errors());
    }

    #[test]
    fn test_on_message_info() {
        let layer = ValidationLayer::new(ValidationLevel::Full);
        let msg = ValidationMessage::info("Test info");
        layer.on_message(&msg);

        assert_eq!(layer.message_count(), 1);
        assert_eq!(layer.error_count(), 0);
        assert_eq!(layer.warning_count(), 0);
    }

    #[test]
    fn test_on_message_multiple() {
        let layer = ValidationLayer::new(ValidationLevel::Full);

        layer.on_message(&ValidationMessage::error("E1"));
        layer.on_message(&ValidationMessage::error("E2"));
        layer.on_message(&ValidationMessage::warning("W1"));
        layer.on_message(&ValidationMessage::info("I1"));

        assert_eq!(layer.error_count(), 2);
        assert_eq!(layer.warning_count(), 1);
        assert_eq!(layer.message_count(), 4);
    }

    // ---- Threshold Filtering Tests ----

    #[test]
    fn test_threshold_basic_filters_info() {
        let layer = ValidationLayer::new(ValidationLevel::Basic);

        layer.on_message(&ValidationMessage::info("Info"));
        assert_eq!(layer.message_count(), 0);

        layer.on_message(&ValidationMessage::warning("Warning"));
        assert_eq!(layer.message_count(), 1);
    }

    #[test]
    fn test_threshold_disabled_filters_all_but_error() {
        let layer = ValidationLayer::new(ValidationLevel::Disabled);

        layer.on_message(&ValidationMessage::info("Info"));
        layer.on_message(&ValidationMessage::warning("Warning"));
        assert_eq!(layer.message_count(), 0);

        layer.on_message(&ValidationMessage::error("Error"));
        assert_eq!(layer.message_count(), 1);
    }

    #[test]
    fn test_threshold_verbose_allows_all() {
        let layer = ValidationLayer::new(ValidationLevel::Verbose);

        let verbose_msg = ValidationMessage::new(
            ValidationSeverity::Verbose,
            ValidationMessageType::DebugMarker,
            "Verbose",
        );
        layer.on_message(&verbose_msg);
        assert_eq!(layer.message_count(), 1);
    }

    // ---- Counter Tests ----

    #[test]
    fn test_reset_counts() {
        let layer = ValidationLayer::new(ValidationLevel::Full);

        layer.on_message(&ValidationMessage::error("Error"));
        layer.on_message(&ValidationMessage::warning("Warning"));

        assert!(layer.has_errors());
        assert!(layer.has_warnings());

        layer.reset_counts();

        assert!(!layer.has_errors());
        assert!(!layer.has_warnings());
        assert_eq!(layer.message_count(), 0);
        assert_eq!(layer.error_count(), 0);
        assert_eq!(layer.warning_count(), 0);
    }

    // ---- Summary Tests ----

    #[test]
    fn test_summary_format() {
        let layer = ValidationLayer::new(ValidationLevel::Full);
        layer.on_message(&ValidationMessage::error("E"));
        layer.on_message(&ValidationMessage::warning("W"));

        let summary = layer.summary();
        assert!(summary.contains("Full"));
        assert!(summary.contains("1 errors"));
        assert!(summary.contains("1 warnings"));
        assert!(summary.contains("2 total"));
    }

    #[test]
    fn test_summary_clean() {
        let layer = ValidationLayer::new(ValidationLevel::Basic);
        let summary = layer.summary();
        assert!(summary.contains("0 errors"));
        assert!(summary.contains("0 warnings"));
        assert!(summary.contains("0 total"));
    }

    // ---- break_on_error Tests ----

    #[test]
    fn test_break_on_error_default_false() {
        let layer = ValidationLayer::new(ValidationLevel::Full);
        assert!(!layer.break_on_error());
    }

    #[test]
    fn test_set_break_on_error() {
        let layer = ValidationLayer::new(ValidationLevel::Full);
        layer.set_break_on_error(true);
        assert!(layer.break_on_error());

        layer.set_break_on_error(false);
        assert!(!layer.break_on_error());
    }

    // ---- Callback Tests ----

    #[test]
    fn test_register_callback() {
        let layer = ValidationLayer::new(ValidationLevel::Full);
        let counter = Arc::new(AtomicU64::new(0));
        let counter_clone = counter.clone();

        layer.register_callback(Box::new(move |_| {
            counter_clone.fetch_add(1, Ordering::SeqCst);
        }));

        layer.on_message(&ValidationMessage::error("Test"));
        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn test_callbacks_method() {
        let layer = ValidationLayer::new(ValidationLevel::Full);
        let callbacks = layer.callbacks();
        assert!(callbacks.is_empty());
    }

    // ---- Scope Tests ----

    #[test]
    fn test_scope_creation() {
        let layer = ValidationLayer::new(ValidationLevel::Full);
        let scope = layer.scope("Test Scope");
        assert_eq!(scope.name(), "Test Scope");
    }

    // ---- Trait Tests ----

    #[test]
    fn test_debug_format() {
        let layer = ValidationLayer::new(ValidationLevel::Full);
        let debug = format!("{:?}", layer);
        assert!(debug.contains("ValidationLayer"));
        assert!(debug.contains("Full"));
    }

    #[test]
    fn test_level_accessor() {
        let layer = ValidationLayer::new(ValidationLevel::Verbose);
        assert_eq!(layer.level(), ValidationLevel::Verbose);
    }

    #[test]
    fn test_features_accessor() {
        let layer = ValidationLayer::new(ValidationLevel::Basic);
        assert!(layer.features().shader_validation);
        assert!(!layer.features().gpu_based_validation);
    }

    #[test]
    fn test_is_enabled() {
        let enabled = ValidationLayer::new(ValidationLevel::Full);
        let disabled = ValidationLayer::new(ValidationLevel::Disabled);

        assert!(enabled.is_enabled());
        assert!(!disabled.is_enabled());
    }

    #[test]
    fn test_concurrent_on_message() {
        let layer = Arc::new(ValidationLayer::new(ValidationLevel::Full));
        let mut handles = vec![];

        for _ in 0..4 {
            let l = layer.clone();
            handles.push(thread::spawn(move || {
                for _ in 0..100 {
                    l.on_message(&ValidationMessage::warning("Concurrent"));
                }
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }

        assert_eq!(layer.warning_count(), 400);
        assert_eq!(layer.message_count(), 400);
    }
}

// ============================================================================
// ValidationScope Tests (20 tests)
// ============================================================================

mod validation_scope_tests {
    use super::*;

    // ---- Construction Tests ----

    #[test]
    fn test_new_scope() {
        let layer = ValidationLayer::new(ValidationLevel::Full);
        let scope = ValidationScope::new(&layer, "Test");
        assert_eq!(scope.name(), "Test");
    }

    #[test]
    fn test_scope_initial_counters() {
        let layer = ValidationLayer::new(ValidationLevel::Full);
        let scope = ValidationScope::new(&layer, "Test").silent();

        assert_eq!(scope.scope_errors(), 0);
        assert_eq!(scope.scope_warnings(), 0);
        assert_eq!(scope.scope_messages(), 0);
        assert!(!scope.has_errors());
        assert!(!scope.has_warnings());
    }

    // ---- Counter Tracking Tests ----

    #[test]
    fn test_scope_tracks_errors() {
        let layer = ValidationLayer::new(ValidationLevel::Full);

        {
            let scope = ValidationScope::new(&layer, "Test").silent();
            layer.on_message(&ValidationMessage::error("E"));
            assert_eq!(scope.scope_errors(), 1);
            assert!(scope.has_errors());
        }
    }

    #[test]
    fn test_scope_tracks_warnings() {
        let layer = ValidationLayer::new(ValidationLevel::Full);

        {
            let scope = ValidationScope::new(&layer, "Test").silent();
            layer.on_message(&ValidationMessage::warning("W"));
            assert_eq!(scope.scope_warnings(), 1);
            assert!(scope.has_warnings());
        }
    }

    #[test]
    fn test_scope_tracks_messages() {
        let layer = ValidationLayer::new(ValidationLevel::Full);

        {
            let scope = ValidationScope::new(&layer, "Test").silent();
            layer.on_message(&ValidationMessage::info("I"));
            layer.on_message(&ValidationMessage::warning("W"));
            layer.on_message(&ValidationMessage::error("E"));
            assert_eq!(scope.scope_messages(), 3);
        }
    }

    #[test]
    fn test_scope_ignores_pre_scope_messages() {
        let layer = ValidationLayer::new(ValidationLevel::Full);

        // Messages before scope
        layer.on_message(&ValidationMessage::error("Pre-scope"));

        {
            let scope = ValidationScope::new(&layer, "Test").silent();
            assert_eq!(scope.scope_errors(), 0);

            layer.on_message(&ValidationMessage::error("In-scope"));
            assert_eq!(scope.scope_errors(), 1);
        }

        // Total should include all
        assert_eq!(layer.error_count(), 2);
    }

    // ---- Silent Mode Tests ----

    #[test]
    fn test_silent_mode() {
        let layer = ValidationLayer::new(ValidationLevel::Full);
        let scope = ValidationScope::new(&layer, "Test").silent();

        // Should complete without logging on drop
        drop(scope);
    }

    // ---- Elapsed Time Tests ----

    #[test]
    fn test_elapsed_time() {
        let layer = ValidationLayer::new(ValidationLevel::Full);
        let scope = ValidationScope::new(&layer, "Test").silent();

        let elapsed = scope.elapsed();
        assert!(elapsed < Duration::from_secs(1));
    }

    #[test]
    fn test_elapsed_increases() {
        let layer = ValidationLayer::new(ValidationLevel::Full);
        let scope = ValidationScope::new(&layer, "Test").silent();

        let elapsed1 = scope.elapsed();
        thread::sleep(Duration::from_millis(10));
        let elapsed2 = scope.elapsed();

        assert!(elapsed2 >= elapsed1);
    }

    // ---- Summary Tests ----

    #[test]
    fn test_scope_summary() {
        let layer = ValidationLayer::new(ValidationLevel::Full);
        let scope = ValidationScope::new(&layer, "Summary Test").silent();

        layer.on_message(&ValidationMessage::error("E"));
        layer.on_message(&ValidationMessage::warning("W"));

        let summary = scope.summary();
        assert!(summary.contains("Summary Test"));
        assert!(summary.contains("1 errors"));
        assert!(summary.contains("1 warnings"));
    }

    // ---- end() Method Tests ----

    #[test]
    fn test_end_returns_result() {
        let layer = ValidationLayer::new(ValidationLevel::Full);
        let scope = ValidationScope::new(&layer, "End Test").silent();

        layer.on_message(&ValidationMessage::warning("W"));

        let result = scope.end();
        assert_eq!(result.name, "End Test");
        assert_eq!(result.warnings, 1);
        assert_eq!(result.errors, 0);
    }

    #[test]
    fn test_end_disables_report_on_drop() {
        let layer = ValidationLayer::new(ValidationLevel::Full);
        let scope = ValidationScope::new(&layer, "Test");

        // Call end() - this consumes the scope and should not report on drop
        let _result = scope.end();
    }

    // ---- ValidationScopeResult Tests ----

    #[test]
    fn test_result_has_errors() {
        let layer = ValidationLayer::new(ValidationLevel::Full);
        let scope = ValidationScope::new(&layer, "Test").silent();

        layer.on_message(&ValidationMessage::error("E"));

        let result = scope.end();
        assert!(result.has_errors());
        assert!(!result.is_clean());
    }

    #[test]
    fn test_result_has_warnings() {
        let layer = ValidationLayer::new(ValidationLevel::Full);
        let scope = ValidationScope::new(&layer, "Test").silent();

        layer.on_message(&ValidationMessage::warning("W"));

        let result = scope.end();
        assert!(result.has_warnings());
        assert!(!result.is_clean());
    }

    #[test]
    fn test_result_is_clean() {
        let layer = ValidationLayer::new(ValidationLevel::Full);
        let scope = ValidationScope::new(&layer, "Clean").silent();

        let result = scope.end();
        assert!(result.is_clean());
        assert!(!result.has_errors());
        assert!(!result.has_warnings());
    }

    #[test]
    fn test_result_display() {
        let layer = ValidationLayer::new(ValidationLevel::Full);
        let scope = ValidationScope::new(&layer, "Display").silent();

        layer.on_message(&ValidationMessage::error("E"));

        let result = scope.end();
        let display = format!("{}", result);
        assert!(display.contains("Display"));
        assert!(display.contains("1 errors"));
    }

    // ---- Debug Format Tests ----

    #[test]
    fn test_scope_debug() {
        let layer = ValidationLayer::new(ValidationLevel::Full);
        let scope = ValidationScope::new(&layer, "Debug Test").silent();

        let debug = format!("{:?}", scope);
        assert!(debug.contains("ValidationScope"));
        assert!(debug.contains("Debug Test"));
    }

    #[test]
    fn test_result_debug() {
        let layer = ValidationLayer::new(ValidationLevel::Full);
        let scope = ValidationScope::new(&layer, "Test").silent();
        let result = scope.end();

        let debug = format!("{:?}", result);
        assert!(debug.contains("ValidationScopeResult"));
    }

    #[test]
    fn test_result_clone() {
        let layer = ValidationLayer::new(ValidationLevel::Full);
        let scope = ValidationScope::new(&layer, "Clone").silent();
        let result = scope.end();

        let cloned = result.clone();
        assert_eq!(cloned.name, result.name);
        assert_eq!(cloned.errors, result.errors);
    }

    // ---- Nested Scope Tests ----

    #[test]
    fn test_nested_scopes() {
        let layer = ValidationLayer::new(ValidationLevel::Full);

        let outer = ValidationScope::new(&layer, "Outer").silent();
        layer.on_message(&ValidationMessage::error("Outer error"));

        {
            let inner = ValidationScope::new(&layer, "Inner").silent();
            layer.on_message(&ValidationMessage::error("Inner error"));

            assert_eq!(inner.scope_errors(), 1);
        }

        assert_eq!(outer.scope_errors(), 2);
    }
}

// ============================================================================
// Additional Edge Case Tests (10 tests)
// ============================================================================

mod edge_case_tests {
    use super::*;

    #[test]
    fn test_empty_scope_name() {
        let layer = ValidationLayer::new(ValidationLevel::Full);
        let scope = ValidationScope::new(&layer, "").silent();
        assert_eq!(scope.name(), "");
    }

    #[test]
    fn test_unicode_scope_name() {
        let layer = ValidationLayer::new(ValidationLevel::Full);
        let scope = ValidationScope::new(&layer, "Test \u{1F4A1} Unicode").silent();
        assert!(scope.name().contains("\u{1F4A1}"));
    }

    #[test]
    fn test_very_long_scope_name() {
        let layer = ValidationLayer::new(ValidationLevel::Full);
        let long_name = "X".repeat(10000);
        let scope = ValidationScope::new(&layer, &long_name).silent();
        assert_eq!(scope.name().len(), 10000);
    }

    #[test]
    fn test_validation_message_empty_objects() {
        let msg = ValidationMessage::error("Test")
            .with_objects(vec![]);
        assert!(msg.objects.is_empty());
    }

    #[test]
    fn test_validation_object_max_u64_handle() {
        let obj = ValidationObject::new(ValidationObjectType::Buffer, u64::MAX);
        let display = obj.display_string();
        assert!(display.contains("0xffffffffffffffff"));
    }

    #[test]
    fn test_source_location_in_message() {
        let loc = SourceLocation::new()
            .with_file("test.wgsl")
            .with_line(100)
            .with_column(25)
            .with_function("main");

        let msg = ValidationMessage::error("Shader error")
            .with_location(loc);

        assert!(msg.location.is_some());
        let loc = msg.location.as_ref().unwrap();
        assert_eq!(loc.file.as_deref(), Some("test.wgsl"));
        assert_eq!(loc.line, Some(100));
        assert_eq!(loc.column, Some(25));
        assert_eq!(loc.function.as_deref(), Some("main"));
    }

    #[test]
    fn test_many_callbacks() {
        let registry = ValidationCallbackRegistry::new();
        let counter = Arc::new(AtomicU64::new(0));

        for _ in 0..100 {
            let c = counter.clone();
            registry.register(Box::new(move |_| {
                c.fetch_add(1, Ordering::SeqCst);
            }));
        }

        registry.invoke(&ValidationMessage::info("Test"));
        assert_eq!(counter.load(Ordering::SeqCst), 100);
    }

    #[test]
    fn test_rapid_message_processing() {
        let layer = ValidationLayer::new(ValidationLevel::Full);

        for i in 0..1000 {
            let msg = if i % 3 == 0 {
                ValidationMessage::error("Error")
            } else if i % 3 == 1 {
                ValidationMessage::warning("Warning")
            } else {
                ValidationMessage::info("Info")
            };
            layer.on_message(&msg);
        }

        assert_eq!(layer.message_count(), 1000);
        assert_eq!(layer.error_count(), 334);  // 0, 3, 6, ... (333 + 1 for 0)
        assert_eq!(layer.warning_count(), 333); // 1, 4, 7, ...
    }

    #[test]
    fn test_validation_features_individual_toggle() {
        let mut features = ValidationFeatures::default();

        features.gpu_based_validation = true;
        assert!(features.any_enabled());

        features.gpu_based_validation = false;
        features.shader_validation = false;
        features.descriptor_indexing_validation = false;
        features.best_practices_warnings = false;

        assert!(!features.any_enabled());
    }

    #[test]
    fn test_all_severity_log_levels() {
        // Verify the log level mapping is complete
        let severities = [
            ValidationSeverity::Verbose,
            ValidationSeverity::Info,
            ValidationSeverity::Warning,
            ValidationSeverity::Error,
        ];

        for severity in severities.iter() {
            let _level = severity.as_log_level();
            // Just verify it doesn't panic
        }
    }
}
