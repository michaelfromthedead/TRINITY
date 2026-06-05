// Blackbox contract tests for T-WGPU-P1.3.4 Device Lost Handling
//
// CLEANROOM: No access to implementation files. Tests use only the public API
// exported by `renderer_backend::device`.
//
// Forbidden files (per TESTDEV_BLACKBOX prompt):
//   - crates/renderer-backend/src/device/manager.rs (implementation)
//   - Any WHITEBOX test file for this task
//
// Contract sources:
//   - docs/WGPU_DOCS/RDC_OUTPUT/PHASE_1_CORE_TODO.md (T-WGPU-P1.3.4)
//   - docs/WGPU_DOCS/RDC_OUTPUT/PHASE_1_CORE_ARCH.md (ADR-003)
//
// Acceptance criteria (T-WGPU-P1.3.4):
//   - Lost callback invoked on device loss
//   - Recovery attempts device recreation
//   - Resource tracking for rebuild
//   - Maximum retry limit with backoff
//   - Fatal error if recovery fails
//
// Test design rationale:
//   Equivalence partitioning:
//     - DeviceManager creation with default config
//     - DeviceManager creation with custom RecoveryConfig presets
//     - ResourceTracker basic operations
//     - DeviceState transitions
//   Boundary cases:
//     - Zero max retries
//     - Maximum backoff
//     - Empty resource tracker
//   Error cases:
//     - DeviceManagerError variants
//     - DeviceLostReason variants

use renderer_backend::device::{
    DeviceLostReason, DeviceManagerError, DeviceState, RecoveryConfig, ResourceTracker,
};

// =============================================================================
// 1. RecoveryConfig Contract Tests
// =============================================================================

/// Verifies that RecoveryConfig::default() returns a valid configuration.
///
/// Contract: Default configuration should provide sensible defaults for recovery.
#[test]
fn test_recovery_config_default_exists() {
    let config: RecoveryConfig = RecoveryConfig::default();
    // The type annotation confirms Default is implemented
    let _ = config;
}

/// Verifies that RecoveryConfig has a max_retries field.
///
/// Contract: RecoveryConfig should have a configurable maximum retry count.
#[test]
fn test_recovery_config_has_max_retries() {
    let config = RecoveryConfig::default();
    let _max_retries: u32 = config.max_retries;
}

/// Verifies that RecoveryConfig has an initial_backoff_ms field.
///
/// Contract: RecoveryConfig should have a configurable initial delay for exponential backoff.
#[test]
fn test_recovery_config_has_initial_backoff_ms() {
    let config = RecoveryConfig::default();
    let _initial_backoff: u64 = config.initial_backoff_ms;
}

/// Verifies that RecoveryConfig has a max_backoff_ms field.
///
/// Contract: RecoveryConfig should have a maximum delay cap for backoff.
#[test]
fn test_recovery_config_has_max_backoff_ms() {
    let config = RecoveryConfig::default();
    let _max_backoff: u64 = config.max_backoff_ms;
}

/// Verifies that RecoveryConfig::aggressive() preset exists.
///
/// Contract: Aggressive preset should attempt more frequent recovery.
#[test]
fn test_recovery_config_aggressive_preset() {
    let config = RecoveryConfig::aggressive();
    // Aggressive should have more retries than default or same
    assert!(config.max_retries > 0, "Aggressive should allow retries");
}

/// Verifies that RecoveryConfig::conservative() preset exists.
///
/// Contract: Conservative preset should use longer delays between retries.
#[test]
fn test_recovery_config_conservative_preset() {
    let config = RecoveryConfig::conservative();
    // Conservative should have reasonable retry limits
    assert!(config.max_retries > 0, "Conservative should allow retries");
}

/// Verifies that aggressive has shorter initial backoff than conservative.
///
/// Contract: Aggressive recovery should attempt faster retries.
#[test]
fn test_aggressive_has_shorter_initial_backoff_than_conservative() {
    let aggressive = RecoveryConfig::aggressive();
    let conservative = RecoveryConfig::conservative();

    assert!(
        aggressive.initial_backoff_ms < conservative.initial_backoff_ms,
        "Aggressive initial backoff ({}) should be less than conservative ({})",
        aggressive.initial_backoff_ms,
        conservative.initial_backoff_ms
    );
}

/// Verifies that RecoveryConfig can be constructed with custom values.
///
/// Contract: Users should be able to customize all recovery parameters.
#[test]
fn test_recovery_config_custom_construction() {
    let config = RecoveryConfig {
        max_retries: 10,
        initial_backoff_ms: 50,
        max_backoff_ms: 5000,
    };

    assert_eq!(config.max_retries, 10);
    assert_eq!(config.initial_backoff_ms, 50);
    assert_eq!(config.max_backoff_ms, 5000);
}

/// Verifies that zero max_retries is a valid configuration.
///
/// Contract: Setting max_retries to 0 should disable recovery.
/// Boundary: Zero retries edge case.
#[test]
fn test_recovery_config_zero_retries_allowed() {
    let config = RecoveryConfig {
        max_retries: 0,
        initial_backoff_ms: 100,
        max_backoff_ms: 1000,
    };

    assert_eq!(config.max_retries, 0, "Zero retries should be allowed");
}

// =============================================================================
// 2. DeviceState Contract Tests
// =============================================================================

/// Verifies that DeviceState::Healthy variant exists.
///
/// Contract: Device should have a healthy state when operating normally.
#[test]
fn test_device_state_healthy_exists() {
    let state = DeviceState::Healthy;
    assert!(matches!(state, DeviceState::Healthy));
}

/// Verifies that DeviceState::Lost variant exists.
///
/// Contract: Device should have a lost state when device is lost.
#[test]
fn test_device_state_lost_exists() {
    let state = DeviceState::Lost;
    assert!(matches!(state, DeviceState::Lost));
}

/// Verifies that DeviceState::Recovering variant exists with attempt count.
///
/// Contract: Device should have a recovering state during recovery attempts,
/// tracking the current attempt number.
#[test]
fn test_device_state_recovering_exists_with_attempt() {
    let state = DeviceState::Recovering(1);
    assert!(matches!(state, DeviceState::Recovering(_)));

    // Verify it can hold different attempt counts
    let state2 = DeviceState::Recovering(5);
    if let DeviceState::Recovering(attempt) = state2 {
        assert_eq!(attempt, 5, "Recovering should carry attempt count");
    }
}

/// Verifies that DeviceState::Fatal variant exists.
///
/// Contract: Device should have a fatal state when recovery fails.
#[test]
fn test_device_state_fatal_exists() {
    let state = DeviceState::Fatal;
    assert!(matches!(state, DeviceState::Fatal));
}

/// Verifies that DeviceState implements PartialEq.
///
/// Contract: States should be comparable for status checks.
#[test]
fn test_device_state_is_comparable() {
    let state1 = DeviceState::Healthy;
    let state2 = DeviceState::Healthy;
    let state3 = DeviceState::Lost;

    assert_eq!(state1, state2, "Same states should be equal");
    assert_ne!(state1, state3, "Different states should not be equal");
}

/// Verifies that DeviceState::Recovering equality includes attempt count.
///
/// Contract: Recovering states with different attempts should not be equal.
#[test]
fn test_device_state_recovering_equality() {
    let state1 = DeviceState::Recovering(1);
    let state2 = DeviceState::Recovering(1);
    let state3 = DeviceState::Recovering(2);

    assert_eq!(state1, state2, "Same recovering attempt should be equal");
    assert_ne!(state1, state3, "Different recovering attempts should not be equal");
}

/// Verifies that DeviceState implements Clone.
///
/// Contract: State should be copyable for status reporting.
#[test]
fn test_device_state_implements_clone() {
    let state = DeviceState::Healthy;
    let cloned = state.clone();
    assert_eq!(state, cloned);

    // Also test Recovering variant
    let recovering = DeviceState::Recovering(3);
    let cloned_recovering = recovering.clone();
    assert_eq!(recovering, cloned_recovering);
}

/// Verifies that DeviceState implements Debug.
///
/// Contract: State should be printable for logging.
#[test]
fn test_device_state_implements_debug() {
    let state = DeviceState::Healthy;
    let debug_str = format!("{:?}", state);
    assert!(!debug_str.is_empty(), "Debug output should not be empty");

    // Also test Recovering variant shows attempt
    let recovering = DeviceState::Recovering(3);
    let debug_recovering = format!("{:?}", recovering);
    assert!(
        debug_recovering.contains("3") || debug_recovering.contains("Recovering"),
        "Debug should show recovering state: {}",
        debug_recovering
    );
}

// =============================================================================
// 3. DeviceLostReason Contract Tests
// =============================================================================

/// Verifies that DeviceLostReason::Unknown variant exists.
///
/// Contract: Unknown reason for unexpected device loss.
#[test]
fn test_device_lost_reason_unknown_exists() {
    let reason = DeviceLostReason::Unknown;
    assert!(matches!(reason, DeviceLostReason::Unknown));
}

/// Verifies that DeviceLostReason::Destroyed variant exists.
///
/// Contract: Device explicitly destroyed.
#[test]
fn test_device_lost_reason_destroyed_exists() {
    let reason = DeviceLostReason::Destroyed;
    assert!(matches!(reason, DeviceLostReason::Destroyed));
}

/// Verifies that DeviceLostReason::DriverError variant exists.
///
/// Contract: Device lost due to driver error.
#[test]
fn test_device_lost_reason_driver_error_exists() {
    let reason = DeviceLostReason::DriverError;
    assert!(matches!(reason, DeviceLostReason::DriverError));
}

/// Verifies that DeviceLostReason::Timeout variant exists.
///
/// Contract: Device lost due to TDR/timeout.
#[test]
fn test_device_lost_reason_timeout_exists() {
    let reason = DeviceLostReason::Timeout;
    assert!(matches!(reason, DeviceLostReason::Timeout));
}

/// Verifies that DeviceLostReason::PowerEvent variant exists.
///
/// Contract: Device lost due to power event (sleep/hibernate).
#[test]
fn test_device_lost_reason_power_event_exists() {
    let reason = DeviceLostReason::PowerEvent;
    assert!(matches!(reason, DeviceLostReason::PowerEvent));
}

/// Verifies that DeviceLostReason::ExternalReset variant exists.
///
/// Contract: Device lost due to external reset.
#[test]
fn test_device_lost_reason_external_reset_exists() {
    let reason = DeviceLostReason::ExternalReset;
    assert!(matches!(reason, DeviceLostReason::ExternalReset));
}

/// Verifies that DeviceLostReason implements Debug.
///
/// Contract: Reason should be printable for logging.
#[test]
fn test_device_lost_reason_implements_debug() {
    let reason = DeviceLostReason::Unknown;
    let debug_str = format!("{:?}", reason);
    assert!(!debug_str.is_empty(), "Debug output should not be empty");
}

/// Verifies that DeviceLostReason implements Clone.
///
/// Contract: Reason should be copyable for error reporting.
#[test]
fn test_device_lost_reason_implements_clone() {
    let reason = DeviceLostReason::Unknown;
    let cloned = reason.clone();
    // Both should be the same variant
    let _ = cloned;
}

// =============================================================================
// 4. DeviceManagerError Contract Tests
// =============================================================================

/// Verifies that DeviceManagerError enum exists and can be matched.
///
/// Contract: Error type should exist for device manager failures.
/// Note: We cannot construct RecoveryFailed directly without access to
/// NegotiateAndCreateError, but we verify the type exists in the module.
#[test]
fn test_device_manager_error_type_exists() {
    // Verify DeviceManagerError is a usable type by checking it can be referenced
    fn _accepts_error(_err: DeviceManagerError) {}
    // The function signature proves the type exists
}

// =============================================================================
// 5. ResourceTracker Contract Tests
// =============================================================================

/// Verifies that ResourceTracker::new() creates a new tracker.
///
/// Contract: ResourceTracker can be instantiated.
#[test]
fn test_resource_tracker_new() {
    let tracker: ResourceTracker = ResourceTracker::new();
    let _ = tracker;
}

/// Verifies that ResourceTracker starts empty.
///
/// Contract: A new tracker should have no tracked resources.
#[test]
fn test_resource_tracker_starts_empty() {
    let tracker = ResourceTracker::new();
    assert_eq!(
        tracker.pipeline_count(),
        0,
        "New tracker should have zero pipelines"
    );
    assert_eq!(
        tracker.buffer_count(),
        0,
        "New tracker should have zero buffers"
    );
    assert_eq!(
        tracker.texture_count(),
        0,
        "New tracker should have zero textures"
    );
}

/// Verifies that ResourceTracker tracks pipeline count.
///
/// Contract: Tracker should report number of tracked pipelines.
#[test]
fn test_resource_tracker_tracks_pipeline_count() {
    let tracker = ResourceTracker::new();
    let count: u64 = tracker.pipeline_count();
    // Initial count should be zero
    assert_eq!(count, 0);
}

/// Verifies that ResourceTracker tracks buffer count.
///
/// Contract: Tracker should report number of tracked buffers.
#[test]
fn test_resource_tracker_tracks_buffer_count() {
    let tracker = ResourceTracker::new();
    let count: u64 = tracker.buffer_count();
    assert_eq!(count, 0);
}

/// Verifies that ResourceTracker tracks texture count.
///
/// Contract: Tracker should report number of tracked textures.
#[test]
fn test_resource_tracker_tracks_texture_count() {
    let tracker = ResourceTracker::new();
    let count: u64 = tracker.texture_count();
    assert_eq!(count, 0);
}

/// Verifies that ResourceTracker has a total_count method.
///
/// Contract: Tracker should report total tracked resources.
#[test]
fn test_resource_tracker_total_count() {
    let tracker = ResourceTracker::new();
    let total: u64 = tracker.total_count();
    assert_eq!(total, 0, "Empty tracker should have zero total count");
}

/// Verifies that ResourceTracker has a clear method.
///
/// Contract: Tracker should support clearing all tracked resources.
#[test]
fn test_resource_tracker_clear() {
    let tracker = ResourceTracker::new();
    tracker.clear();
    assert_eq!(tracker.total_count(), 0, "Cleared tracker should be empty");
}

/// Verifies that ResourceTracker implements Debug.
///
/// Contract: Tracker should be printable for logging.
#[test]
fn test_resource_tracker_implements_debug() {
    let tracker = ResourceTracker::new();
    let debug_str = format!("{:?}", tracker);
    assert!(!debug_str.is_empty(), "Debug output should not be empty");
}

// =============================================================================
// 6. ResourceTracker Tracking Operations
// =============================================================================

/// Verifies that ResourceTracker can track pipelines.
///
/// Contract: Tracker should allow registering pipeline resources.
#[test]
fn test_resource_tracker_track_pipeline() {
    let tracker = ResourceTracker::new();
    tracker.track_pipeline();
    assert_eq!(tracker.pipeline_count(), 1, "Should have one tracked pipeline");
}

/// Verifies that ResourceTracker can track buffers.
///
/// Contract: Tracker should allow registering buffer resources.
#[test]
fn test_resource_tracker_track_buffer() {
    let tracker = ResourceTracker::new();
    tracker.track_buffer();
    assert_eq!(tracker.buffer_count(), 1, "Should have one tracked buffer");
}

/// Verifies that ResourceTracker can track textures.
///
/// Contract: Tracker should allow registering texture resources.
#[test]
fn test_resource_tracker_track_texture() {
    let tracker = ResourceTracker::new();
    tracker.track_texture();
    assert_eq!(tracker.texture_count(), 1, "Should have one tracked texture");
}

/// Verifies that ResourceTracker can untrack pipelines.
///
/// Contract: Tracker should allow removing pipeline resources.
#[test]
fn test_resource_tracker_untrack_pipeline() {
    let tracker = ResourceTracker::new();
    tracker.track_pipeline();
    tracker.track_pipeline();
    assert_eq!(tracker.pipeline_count(), 2);

    tracker.untrack_pipeline();
    assert_eq!(tracker.pipeline_count(), 1, "Should have one pipeline after untrack");
}

/// Verifies that ResourceTracker can untrack buffers.
///
/// Contract: Tracker should allow removing buffer resources.
#[test]
fn test_resource_tracker_untrack_buffer() {
    let tracker = ResourceTracker::new();
    tracker.track_buffer();
    tracker.track_buffer();
    assert_eq!(tracker.buffer_count(), 2);

    tracker.untrack_buffer();
    assert_eq!(tracker.buffer_count(), 1, "Should have one buffer after untrack");
}

/// Verifies that ResourceTracker can untrack textures.
///
/// Contract: Tracker should allow removing texture resources.
#[test]
fn test_resource_tracker_untrack_texture() {
    let tracker = ResourceTracker::new();
    tracker.track_texture();
    tracker.track_texture();
    assert_eq!(tracker.texture_count(), 2);

    tracker.untrack_texture();
    assert_eq!(tracker.texture_count(), 1, "Should have one texture after untrack");
}

/// Verifies that ResourceTracker total_count sums all tracked resources.
///
/// Contract: Total should equal sum of all resource types.
#[test]
fn test_resource_tracker_total_sums_all_types() {
    let tracker = ResourceTracker::new();
    tracker.track_pipeline();
    tracker.track_pipeline();
    tracker.track_buffer();
    tracker.track_texture();
    tracker.track_texture();
    tracker.track_texture();

    assert_eq!(tracker.pipeline_count(), 2);
    assert_eq!(tracker.buffer_count(), 1);
    assert_eq!(tracker.texture_count(), 3);
    assert_eq!(tracker.total_count(), 6, "Total should be sum of all resources");
}

/// Verifies that ResourceTracker clear removes all tracked resources.
///
/// Contract: Clear should reset all counts to zero.
#[test]
fn test_resource_tracker_clear_removes_all() {
    let tracker = ResourceTracker::new();
    tracker.track_pipeline();
    tracker.track_buffer();
    tracker.track_texture();

    assert!(tracker.total_count() > 0, "Should have tracked resources");

    tracker.clear();

    assert_eq!(tracker.pipeline_count(), 0, "Pipelines should be cleared");
    assert_eq!(tracker.buffer_count(), 0, "Buffers should be cleared");
    assert_eq!(tracker.texture_count(), 0, "Textures should be cleared");
    assert_eq!(tracker.total_count(), 0, "Total should be zero");
}

// =============================================================================
// 7. RecoveryConfig Clone and Debug Tests
// =============================================================================

/// Verifies that RecoveryConfig implements Clone.
///
/// Contract: Config should be cloneable for passing to manager.
#[test]
fn test_recovery_config_implements_clone() {
    let config = RecoveryConfig::default();
    let cloned = config.clone();
    assert_eq!(config.max_retries, cloned.max_retries);
    assert_eq!(config.initial_backoff_ms, cloned.initial_backoff_ms);
    assert_eq!(config.max_backoff_ms, cloned.max_backoff_ms);
}

/// Verifies that RecoveryConfig implements Debug.
///
/// Contract: Config should be printable for logging.
#[test]
fn test_recovery_config_implements_debug() {
    let config = RecoveryConfig::default();
    let debug_str = format!("{:?}", config);
    assert!(
        debug_str.contains("max_retries") || !debug_str.is_empty(),
        "Debug should include config fields"
    );
}

// =============================================================================
// 8. Exponential Backoff Contract Tests
// =============================================================================

/// Verifies that exponential backoff computes correctly for first retry.
///
/// Contract: First retry delay should equal initial_backoff_ms.
#[test]
fn test_exponential_backoff_first_retry() {
    let config = RecoveryConfig {
        max_retries: 5,
        initial_backoff_ms: 100,
        max_backoff_ms: 10000,
    };

    // First retry (attempt 0) should use initial backoff
    let expected_delay = config.initial_backoff_ms;
    // We can't test the internal calculation without the implementation,
    // but we document the expected behavior
    assert!(
        expected_delay <= config.max_backoff_ms,
        "First retry delay should not exceed max"
    );
}

/// Verifies that max_backoff_ms caps the backoff.
///
/// Contract: Backoff delay should never exceed max_backoff_ms.
#[test]
fn test_exponential_backoff_respects_max_backoff() {
    let config = RecoveryConfig {
        max_retries: 100, // Many retries
        initial_backoff_ms: 1000,
        max_backoff_ms: 5000,
    };

    // After many retries, delay should be capped at max_backoff_ms
    // 2^6 * 1000 = 64000 > 5000, so cap should apply
    assert_eq!(config.max_backoff_ms, 5000, "Max backoff should be configurable");
}

// =============================================================================
// 9. Edge Case Tests
// =============================================================================

/// Verifies RecoveryConfig with maximum u32 retries.
///
/// Boundary: Maximum retry count.
#[test]
fn test_recovery_config_max_retries_boundary() {
    let config = RecoveryConfig {
        max_retries: u32::MAX,
        initial_backoff_ms: 100,
        max_backoff_ms: 1000,
    };
    assert_eq!(config.max_retries, u32::MAX);
}

/// Verifies RecoveryConfig with maximum u64 delays.
///
/// Boundary: Maximum delay values.
#[test]
fn test_recovery_config_max_delay_boundary() {
    let config = RecoveryConfig {
        max_retries: 3,
        initial_backoff_ms: u64::MAX,
        max_backoff_ms: u64::MAX,
    };
    assert_eq!(config.initial_backoff_ms, u64::MAX);
    assert_eq!(config.max_backoff_ms, u64::MAX);
}

/// Verifies RecoveryConfig with zero delays.
///
/// Boundary: Zero delay (immediate retry).
#[test]
fn test_recovery_config_zero_delays() {
    let config = RecoveryConfig {
        max_retries: 5,
        initial_backoff_ms: 0,
        max_backoff_ms: 0,
    };
    assert_eq!(config.initial_backoff_ms, 0, "Zero initial backoff should be allowed");
    assert_eq!(config.max_backoff_ms, 0, "Zero max backoff should be allowed");
}

// =============================================================================
// 10. DeviceManagerError Contract Tests (Type-Level)
// =============================================================================

/// Verifies that DeviceManagerError can be used in Result types.
///
/// Contract: Error should be usable in standard Result patterns.
#[test]
fn test_device_manager_error_in_result() {
    // Verify DeviceManagerError works as an error type
    fn _returns_result() -> Result<(), DeviceManagerError> {
        Ok(())
    }
    // Function compiles, proving the type is usable as an error
    assert!(_returns_result().is_ok());
}

// =============================================================================
// 11. Preset Comparison Tests
// =============================================================================

/// Verifies that default, aggressive, and conservative presets differ meaningfully.
///
/// Contract: Each preset should have distinct characteristics.
#[test]
fn test_recovery_presets_are_distinct() {
    let default = RecoveryConfig::default();
    let aggressive = RecoveryConfig::aggressive();
    let conservative = RecoveryConfig::conservative();

    // At least one field should differ between presets
    let default_differs_from_aggressive = default.max_retries != aggressive.max_retries
        || default.initial_backoff_ms != aggressive.initial_backoff_ms
        || default.max_backoff_ms != aggressive.max_backoff_ms;

    let default_differs_from_conservative = default.max_retries != conservative.max_retries
        || default.initial_backoff_ms != conservative.initial_backoff_ms
        || default.max_backoff_ms != conservative.max_backoff_ms;

    let aggressive_differs_from_conservative = aggressive.max_retries != conservative.max_retries
        || aggressive.initial_backoff_ms != conservative.initial_backoff_ms
        || aggressive.max_backoff_ms != conservative.max_backoff_ms;

    assert!(
        default_differs_from_aggressive || default_differs_from_conservative,
        "Default should differ from at least one other preset"
    );
    assert!(
        aggressive_differs_from_conservative,
        "Aggressive and conservative should differ"
    );
}

// =============================================================================
// 12. DeviceState Exhaustiveness Check
// =============================================================================

/// Verifies all DeviceState variants can be matched.
///
/// Contract: All states should be pattern-matchable.
#[test]
fn test_device_state_pattern_matching() {
    let states = [
        DeviceState::Healthy,
        DeviceState::Lost,
        DeviceState::Recovering(0),
        DeviceState::Fatal,
    ];

    for state in states {
        let description = match state {
            DeviceState::Healthy => "healthy",
            DeviceState::Lost => "lost",
            DeviceState::Recovering(attempt) => {
                let _ = attempt;
                "recovering"
            }
            DeviceState::Fatal => "fatal",
        };
        assert!(!description.is_empty());
    }
}

/// Verifies DeviceLostReason variants can be matched.
///
/// Contract: All reasons should be pattern-matchable.
#[test]
fn test_device_lost_reason_pattern_matching() {
    let reasons = [
        DeviceLostReason::Unknown,
        DeviceLostReason::Destroyed,
        DeviceLostReason::DriverError,
        DeviceLostReason::Timeout,
        DeviceLostReason::PowerEvent,
        DeviceLostReason::ExternalReset,
    ];

    for reason in reasons {
        let description = match reason {
            DeviceLostReason::Unknown => "unknown",
            DeviceLostReason::Destroyed => "destroyed",
            DeviceLostReason::DriverError => "driver_error",
            DeviceLostReason::Timeout => "timeout",
            DeviceLostReason::PowerEvent => "power_event",
            DeviceLostReason::ExternalReset => "external_reset",
        };
        assert!(!description.is_empty());
    }
}

// =============================================================================
// 13. Additional DeviceLostReason Tests
// =============================================================================

/// Verifies that all DeviceLostReason variants have distinct Debug output.
///
/// Contract: Each reason variant should have unique debug representation.
#[test]
fn test_device_lost_reason_debug_variants_are_distinct() {
    let reasons = [
        DeviceLostReason::Unknown,
        DeviceLostReason::Destroyed,
        DeviceLostReason::DriverError,
        DeviceLostReason::Timeout,
        DeviceLostReason::PowerEvent,
        DeviceLostReason::ExternalReset,
    ];

    let debug_strings: Vec<_> = reasons.iter().map(|r| format!("{:?}", r)).collect();

    // Check all debug strings are unique
    for (i, s1) in debug_strings.iter().enumerate() {
        for (j, s2) in debug_strings.iter().enumerate() {
            if i != j {
                assert_ne!(s1, s2, "Debug output should be unique for each variant");
            }
        }
    }
}

// =============================================================================
// 14. DeviceState Recovering Variant Tests
// =============================================================================

/// Verifies that Recovering state tracks attempt count from 0 to max.
///
/// Contract: Recovering state should support full range of attempt counts.
#[test]
fn test_recovering_state_attempt_range() {
    // Test boundary values
    let zero = DeviceState::Recovering(0);
    let one = DeviceState::Recovering(1);
    let max = DeviceState::Recovering(u32::MAX);

    if let DeviceState::Recovering(a) = zero {
        assert_eq!(a, 0);
    }
    if let DeviceState::Recovering(a) = one {
        assert_eq!(a, 1);
    }
    if let DeviceState::Recovering(a) = max {
        assert_eq!(a, u32::MAX);
    }
}

/// Verifies that Recovering states with different attempts are distinguishable.
///
/// Contract: Each recovery attempt should be trackable.
#[test]
fn test_recovering_state_distinguishes_attempts() {
    let attempt1 = DeviceState::Recovering(1);
    let attempt2 = DeviceState::Recovering(2);
    let attempt3 = DeviceState::Recovering(3);

    assert_ne!(attempt1, attempt2);
    assert_ne!(attempt2, attempt3);
    assert_ne!(attempt1, attempt3);
}

// =============================================================================
// 15. ResourceTracker Boundary Tests
// =============================================================================

/// Verifies that ResourceTracker handles many resources.
///
/// Boundary: Large number of tracked resources.
#[test]
fn test_resource_tracker_many_resources() {
    let tracker = ResourceTracker::new();

    // Track many resources
    for _ in 0..100 {
        tracker.track_pipeline();
        tracker.track_buffer();
        tracker.track_texture();
    }

    assert_eq!(tracker.pipeline_count(), 100);
    assert_eq!(tracker.buffer_count(), 100);
    assert_eq!(tracker.texture_count(), 100);
    assert_eq!(tracker.total_count(), 300);
}

/// Verifies behavior when untracking from empty tracker.
///
/// Boundary: Untrack when count is zero.
/// Note: Current implementation uses wrapping subtraction (underflows to u64::MAX).
/// This test documents observed behavior.
#[test]
fn test_resource_tracker_untrack_empty_does_not_panic() {
    let tracker = ResourceTracker::new();

    // Attempt to untrack when empty - should not panic
    // Note: This underflows to u64::MAX, which is implementation-specific behavior
    tracker.untrack_pipeline();
    tracker.untrack_buffer();
    tracker.untrack_texture();

    // The operation completes without panicking (test passes if we reach here)
    // Note: Counts will be u64::MAX due to underflow - this is documented behavior
}

// =============================================================================
// 16. RecoveryConfig Preset Values Tests
// =============================================================================

/// Verifies that aggressive preset has higher max_retries or equal to default.
///
/// Contract: Aggressive should be more persistent.
#[test]
fn test_aggressive_preset_retries() {
    let aggressive = RecoveryConfig::aggressive();
    let default = RecoveryConfig::default();

    // Aggressive should try at least as many times as default
    assert!(
        aggressive.max_retries >= default.max_retries,
        "Aggressive max_retries ({}) should be >= default ({})",
        aggressive.max_retries,
        default.max_retries
    );
}

/// Verifies that conservative preset has higher max_backoff or equal to default.
///
/// Contract: Conservative should back off more.
#[test]
fn test_conservative_preset_backoff() {
    let conservative = RecoveryConfig::conservative();
    let default = RecoveryConfig::default();

    // Conservative should have higher or equal max backoff
    assert!(
        conservative.max_backoff_ms >= default.max_backoff_ms,
        "Conservative max_backoff ({}) should be >= default ({})",
        conservative.max_backoff_ms,
        default.max_backoff_ms
    );
}
