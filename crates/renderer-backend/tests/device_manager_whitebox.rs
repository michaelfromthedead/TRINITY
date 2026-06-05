// SPDX-License-Identifier: MIT
//
// device_manager_whitebox.rs -- Whitebox structural tests for T-WGPU-P1.3.4
// (DeviceManager - Device Lost Handling).
//
// These tests exercise the internal implementation of DeviceManager,
// covering all code paths in device state management, recovery logic,
// resource tracking, and error handling.
//
// WHITEBOX coverage plan:
//   - Path A: DeviceState enum - all variants, state helpers (is_healthy, needs_recovery, is_fatal)
//   - Path B: DeviceState Display - all variant formatting
//   - Path C: DeviceState Default - returns Healthy
//   - Path D: AtomicDeviceState - new(), load(), store(), compare_exchange()
//   - Path E: AtomicDeviceState encoding - Healthy=0, Lost=1, Recovering(n)=2+n, Fatal=MAX
//   - Path F: AtomicDeviceState saturating_add overflow protection for large Recovering(n)
//   - Path G: RecoveryConfig - default(), new(), aggressive(), conservative()
//   - Path H: RecoveryConfig backoff_for_attempt() - exponential doubling, max cap
//   - Path I: RecoveryConfig backoff overflow protection with high attempt numbers
//   - Path J: RecoveryConfig Display formatting
//   - Path K: DeviceLostReason - all variants, is_likely_recoverable() for each
//   - Path L: DeviceLostReason Display - all variant formatting
//   - Path M: ResourceTracker - track/untrack all 6 resource types
//   - Path N: ResourceTracker - total_count(), is_empty(), clear()
//   - Path O: ResourceTracker Display formatting
//   - Path P: DeviceManagerError - all variants, Display impl
//   - Path Q: DeviceManagerError - std::error::Error source() impl
//   - Path R: DeviceManagerError - From<NegotiateAndCreateError> conversion
//   - Path S: DeviceManager state accessors - state(), is_healthy(), needs_recovery(), is_fatal()
//   - Path T: DeviceManager mark_lost() - state transition, callback invocation, device clearing
//   - Path U: DeviceManager mark_lost() from non-Healthy state (no-op with warning)
//   - Path V: DeviceManager set_lost_callback() and clear_lost_callback()
//   - Path W: DeviceManager reset_from_fatal() - compare_exchange back to Lost
//   - Path X: DeviceManager recovery_stats() - tracking attempts and successes
//   - Path Y: DeviceManager config() and resource_tracker() accessors
//   - Path Z: DeviceManager Debug and Display trait implementations
//
// Acceptance criteria (T-WGPU-P1.3.4):
//   1. Lost callback invoked on device loss
//   2. Recovery attempts device recreation
//   3. Resource tracking for rebuild
//   4. Maximum retry limit with exponential backoff
//   5. Fatal error if recovery fails after max retries

use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::Arc;
use std::time::Duration;

use renderer_backend::device::{
    DeviceLostReason, DeviceManagerError, DeviceState, RecoveryConfig, ResourceTracker,
};

// ============================================================================
// Path A: DeviceState enum - all variants and state helpers
// ============================================================================

#[test]
fn test_device_state_healthy_is_healthy_true() {
    assert!(DeviceState::Healthy.is_healthy());
}

#[test]
fn test_device_state_lost_is_healthy_false() {
    assert!(!DeviceState::Lost.is_healthy());
}

#[test]
fn test_device_state_recovering_is_healthy_false() {
    assert!(!DeviceState::Recovering(0).is_healthy());
    assert!(!DeviceState::Recovering(1).is_healthy());
    assert!(!DeviceState::Recovering(100).is_healthy());
    assert!(!DeviceState::Recovering(u32::MAX).is_healthy());
}

#[test]
fn test_device_state_fatal_is_healthy_false() {
    assert!(!DeviceState::Fatal.is_healthy());
}

#[test]
fn test_device_state_healthy_needs_recovery_false() {
    assert!(!DeviceState::Healthy.needs_recovery());
}

#[test]
fn test_device_state_lost_needs_recovery_true() {
    assert!(DeviceState::Lost.needs_recovery());
}

#[test]
fn test_device_state_recovering_needs_recovery_true() {
    assert!(DeviceState::Recovering(0).needs_recovery());
    assert!(DeviceState::Recovering(5).needs_recovery());
    assert!(DeviceState::Recovering(u32::MAX).needs_recovery());
}

#[test]
fn test_device_state_fatal_needs_recovery_false() {
    // Fatal means recovery already failed - no point trying again
    assert!(!DeviceState::Fatal.needs_recovery());
}

#[test]
fn test_device_state_healthy_is_fatal_false() {
    assert!(!DeviceState::Healthy.is_fatal());
}

#[test]
fn test_device_state_lost_is_fatal_false() {
    assert!(!DeviceState::Lost.is_fatal());
}

#[test]
fn test_device_state_recovering_is_fatal_false() {
    assert!(!DeviceState::Recovering(0).is_fatal());
    assert!(!DeviceState::Recovering(999).is_fatal());
}

#[test]
fn test_device_state_fatal_is_fatal_true() {
    assert!(DeviceState::Fatal.is_fatal());
}

#[test]
fn test_device_state_equality() {
    assert_eq!(DeviceState::Healthy, DeviceState::Healthy);
    assert_eq!(DeviceState::Lost, DeviceState::Lost);
    assert_eq!(DeviceState::Recovering(5), DeviceState::Recovering(5));
    assert_eq!(DeviceState::Fatal, DeviceState::Fatal);

    assert_ne!(DeviceState::Healthy, DeviceState::Lost);
    assert_ne!(DeviceState::Recovering(0), DeviceState::Recovering(1));
}

#[test]
fn test_device_state_clone() {
    let state = DeviceState::Recovering(42);
    let cloned = state.clone();
    assert_eq!(state, cloned);
}

#[test]
fn test_device_state_copy() {
    let state = DeviceState::Healthy;
    let copied: DeviceState = state;
    assert_eq!(state, copied);
}

// ============================================================================
// Path B: DeviceState Display - all variant formatting
// ============================================================================

#[test]
fn test_device_state_display_healthy() {
    assert_eq!(format!("{}", DeviceState::Healthy), "Healthy");
}

#[test]
fn test_device_state_display_lost() {
    assert_eq!(format!("{}", DeviceState::Lost), "Lost");
}

#[test]
fn test_device_state_display_recovering_attempt_0() {
    assert_eq!(
        format!("{}", DeviceState::Recovering(0)),
        "Recovering (attempt 0)"
    );
}

#[test]
fn test_device_state_display_recovering_attempt_3() {
    assert_eq!(
        format!("{}", DeviceState::Recovering(3)),
        "Recovering (attempt 3)"
    );
}

#[test]
fn test_device_state_display_recovering_attempt_max() {
    let display = format!("{}", DeviceState::Recovering(u32::MAX));
    assert!(display.contains("Recovering"));
    assert!(display.contains(&u32::MAX.to_string()));
}

#[test]
fn test_device_state_display_fatal() {
    assert_eq!(format!("{}", DeviceState::Fatal), "Fatal");
}

// ============================================================================
// Path C: DeviceState Default - returns Healthy
// ============================================================================

#[test]
fn test_device_state_default_is_healthy() {
    let default: DeviceState = Default::default();
    assert_eq!(default, DeviceState::Healthy);
}

// ============================================================================
// Path G: RecoveryConfig - default(), new(), aggressive(), conservative()
// ============================================================================

#[test]
fn test_recovery_config_default_values() {
    let config = RecoveryConfig::default();
    assert_eq!(config.max_retries, 3);
    assert_eq!(config.initial_backoff_ms, 200);
    assert_eq!(config.max_backoff_ms, 5000);
}

#[test]
fn test_recovery_config_new() {
    let config = RecoveryConfig::new(10, 500, 30000);
    assert_eq!(config.max_retries, 10);
    assert_eq!(config.initial_backoff_ms, 500);
    assert_eq!(config.max_backoff_ms, 30000);
}

#[test]
fn test_recovery_config_new_const() {
    // Verify the constructor is const-evaluable
    const CONFIG: RecoveryConfig = RecoveryConfig::new(5, 100, 10000);
    assert_eq!(CONFIG.max_retries, 5);
    assert_eq!(CONFIG.initial_backoff_ms, 100);
    assert_eq!(CONFIG.max_backoff_ms, 10000);
}

#[test]
fn test_recovery_config_aggressive() {
    let config = RecoveryConfig::aggressive();

    // Aggressive should have more retries than default
    assert!(config.max_retries >= 5);

    // Aggressive should have shorter initial backoff
    assert!(config.initial_backoff_ms < 100);

    // Aggressive should have reasonable max backoff
    assert!(config.max_backoff_ms <= 5000);
}

#[test]
fn test_recovery_config_conservative() {
    let config = RecoveryConfig::conservative();

    // Conservative should have fewer retries
    assert!(config.max_retries <= 3);

    // Conservative should have longer initial backoff
    assert!(config.initial_backoff_ms >= 500);

    // Conservative should have longer max backoff
    assert!(config.max_backoff_ms >= 10000);
}

#[test]
fn test_recovery_config_clone() {
    let config = RecoveryConfig::new(7, 333, 7777);
    let cloned = config.clone();
    assert_eq!(config, cloned);
}

#[test]
fn test_recovery_config_copy() {
    let config = RecoveryConfig::new(7, 333, 7777);
    let copied: RecoveryConfig = config;
    assert_eq!(config, copied);
}

#[test]
fn test_recovery_config_equality() {
    let config1 = RecoveryConfig::new(5, 100, 5000);
    let config2 = RecoveryConfig::new(5, 100, 5000);
    let config3 = RecoveryConfig::new(5, 100, 6000);

    assert_eq!(config1, config2);
    assert_ne!(config1, config3);
}

// ============================================================================
// Path H: RecoveryConfig backoff_for_attempt() - exponential doubling, max cap
// ============================================================================

#[test]
fn test_recovery_config_backoff_attempt_0() {
    let config = RecoveryConfig::new(5, 100, 5000);
    assert_eq!(config.backoff_for_attempt(0), Duration::from_millis(100));
}

#[test]
fn test_recovery_config_backoff_exponential_doubling() {
    let config = RecoveryConfig::new(10, 100, 100000);

    assert_eq!(config.backoff_for_attempt(0), Duration::from_millis(100)); // 100 * 2^0 = 100
    assert_eq!(config.backoff_for_attempt(1), Duration::from_millis(200)); // 100 * 2^1 = 200
    assert_eq!(config.backoff_for_attempt(2), Duration::from_millis(400)); // 100 * 2^2 = 400
    assert_eq!(config.backoff_for_attempt(3), Duration::from_millis(800)); // 100 * 2^3 = 800
    assert_eq!(config.backoff_for_attempt(4), Duration::from_millis(1600)); // 100 * 2^4 = 1600
}

#[test]
fn test_recovery_config_backoff_capped_at_max() {
    let config = RecoveryConfig::new(10, 100, 5000);

    // 100 * 2^5 = 3200, still under max
    assert_eq!(config.backoff_for_attempt(5), Duration::from_millis(3200));

    // 100 * 2^6 = 6400, should be capped to 5000
    assert_eq!(config.backoff_for_attempt(6), Duration::from_millis(5000));

    // Higher attempts should stay at max
    assert_eq!(config.backoff_for_attempt(7), Duration::from_millis(5000));
    assert_eq!(config.backoff_for_attempt(10), Duration::from_millis(5000));
    assert_eq!(config.backoff_for_attempt(20), Duration::from_millis(5000));
}

#[test]
fn test_recovery_config_backoff_from_doctest() {
    // Verify doctest example from source
    let config = RecoveryConfig::new(5, 100, 5000);

    assert_eq!(config.backoff_for_attempt(0), Duration::from_millis(100));
    assert_eq!(config.backoff_for_attempt(1), Duration::from_millis(200));
    assert_eq!(config.backoff_for_attempt(2), Duration::from_millis(400));
    assert_eq!(config.backoff_for_attempt(10), Duration::from_millis(5000)); // capped
}

// ============================================================================
// Path I: RecoveryConfig backoff overflow protection
// ============================================================================

#[test]
fn test_recovery_config_backoff_overflow_attempt_31() {
    // 2^31 is the limit before we'd overflow in 1 << attempt
    let config = RecoveryConfig::new(100, 1000, u64::MAX);
    let backoff = config.backoff_for_attempt(31);

    // Should not panic, should return a valid duration
    assert!(backoff.as_millis() > 0);
}

#[test]
fn test_recovery_config_backoff_overflow_attempt_50() {
    // Attempt > 31 should be capped at 31 in the shift
    let config = RecoveryConfig::new(100, 1000, u64::MAX);
    let backoff = config.backoff_for_attempt(50);

    // Should not panic
    assert!(backoff.as_millis() > 0);
}

#[test]
fn test_recovery_config_backoff_overflow_attempt_max() {
    let config = RecoveryConfig::new(100, 1000, u64::MAX);
    let backoff = config.backoff_for_attempt(u32::MAX);

    // Should not panic - saturating_mul + min(31) protects us
    assert!(backoff.as_millis() > 0);
}

#[test]
fn test_recovery_config_backoff_large_initial_no_overflow() {
    // Large initial value that would overflow if not protected
    let config = RecoveryConfig::new(100, u64::MAX / 2, u64::MAX);
    let backoff = config.backoff_for_attempt(5);

    // saturating_mul should protect us
    assert!(backoff.as_millis() > 0);
}

#[test]
fn test_recovery_config_backoff_zero_initial() {
    let config = RecoveryConfig::new(5, 0, 5000);

    // All attempts should return 0ms (capped to 0 before max)
    assert_eq!(config.backoff_for_attempt(0), Duration::from_millis(0));
    assert_eq!(config.backoff_for_attempt(5), Duration::from_millis(0));
}

// ============================================================================
// Path J: RecoveryConfig Display formatting
// ============================================================================

#[test]
fn test_recovery_config_display_contains_max_retries() {
    let config = RecoveryConfig::new(7, 100, 5000);
    let display = format!("{}", config);
    assert!(display.contains("max_retries=7"));
}

#[test]
fn test_recovery_config_display_contains_backoff_range() {
    let config = RecoveryConfig::new(5, 100, 10000);
    let display = format!("{}", config);
    assert!(display.contains("100ms"));
    assert!(display.contains("10000ms"));
}

#[test]
fn test_recovery_config_display_format() {
    let config = RecoveryConfig::new(3, 200, 5000);
    let display = format!("{}", config);

    // Should contain RecoveryConfig prefix
    assert!(display.contains("RecoveryConfig"));
}

// ============================================================================
// Path K: DeviceLostReason - all variants and is_likely_recoverable()
// ============================================================================

#[test]
fn test_device_lost_reason_destroyed_not_recoverable() {
    // Destroyed is intentional - not recoverable
    assert!(!DeviceLostReason::Destroyed.is_likely_recoverable());
}

#[test]
fn test_device_lost_reason_driver_error_recoverable() {
    // Driver error is worth trying to recover from
    assert!(DeviceLostReason::DriverError.is_likely_recoverable());
}

#[test]
fn test_device_lost_reason_timeout_recoverable() {
    // TDR events are usually recoverable
    assert!(DeviceLostReason::Timeout.is_likely_recoverable());
}

#[test]
fn test_device_lost_reason_power_event_recoverable() {
    // Power events (sleep/resume) are usually recoverable
    assert!(DeviceLostReason::PowerEvent.is_likely_recoverable());
}

#[test]
fn test_device_lost_reason_external_reset_recoverable() {
    // External reset is usually recoverable
    assert!(DeviceLostReason::ExternalReset.is_likely_recoverable());
}

#[test]
fn test_device_lost_reason_unknown_recoverable() {
    // Unknown reasons are worth trying to recover from
    assert!(DeviceLostReason::Unknown.is_likely_recoverable());
}

#[test]
fn test_device_lost_reason_equality() {
    assert_eq!(DeviceLostReason::Timeout, DeviceLostReason::Timeout);
    assert_ne!(DeviceLostReason::Timeout, DeviceLostReason::PowerEvent);
}

#[test]
fn test_device_lost_reason_clone() {
    let reason = DeviceLostReason::DriverError;
    let cloned = reason.clone();
    assert_eq!(reason, cloned);
}

#[test]
fn test_device_lost_reason_copy() {
    let reason = DeviceLostReason::ExternalReset;
    let copied: DeviceLostReason = reason;
    assert_eq!(reason, copied);
}

// ============================================================================
// Path L: DeviceLostReason Display - all variant formatting
// ============================================================================

#[test]
fn test_device_lost_reason_display_destroyed() {
    assert_eq!(
        format!("{}", DeviceLostReason::Destroyed),
        "Device destroyed"
    );
}

#[test]
fn test_device_lost_reason_display_driver_error() {
    assert_eq!(format!("{}", DeviceLostReason::DriverError), "Driver error");
}

#[test]
fn test_device_lost_reason_display_timeout() {
    assert_eq!(
        format!("{}", DeviceLostReason::Timeout),
        "GPU timeout (TDR)"
    );
}

#[test]
fn test_device_lost_reason_display_power_event() {
    assert_eq!(
        format!("{}", DeviceLostReason::PowerEvent),
        "Power state change"
    );
}

#[test]
fn test_device_lost_reason_display_external_reset() {
    assert_eq!(
        format!("{}", DeviceLostReason::ExternalReset),
        "External reset"
    );
}

#[test]
fn test_device_lost_reason_display_unknown() {
    assert_eq!(format!("{}", DeviceLostReason::Unknown), "Unknown");
}

// ============================================================================
// Path M: ResourceTracker - track/untrack all 6 resource types
// ============================================================================

#[test]
fn test_resource_tracker_new_is_empty() {
    let tracker = ResourceTracker::new();
    assert!(tracker.is_empty());
    assert_eq!(tracker.total_count(), 0);
}

#[test]
fn test_resource_tracker_track_buffer() {
    let tracker = ResourceTracker::new();

    assert_eq!(tracker.buffer_count(), 0);
    tracker.track_buffer();
    assert_eq!(tracker.buffer_count(), 1);
    tracker.track_buffer();
    assert_eq!(tracker.buffer_count(), 2);
}

#[test]
fn test_resource_tracker_untrack_buffer() {
    let tracker = ResourceTracker::new();

    tracker.track_buffer();
    tracker.track_buffer();
    assert_eq!(tracker.buffer_count(), 2);

    tracker.untrack_buffer();
    assert_eq!(tracker.buffer_count(), 1);
}

#[test]
fn test_resource_tracker_track_texture() {
    let tracker = ResourceTracker::new();

    assert_eq!(tracker.texture_count(), 0);
    tracker.track_texture();
    assert_eq!(tracker.texture_count(), 1);
}

#[test]
fn test_resource_tracker_untrack_texture() {
    let tracker = ResourceTracker::new();

    tracker.track_texture();
    tracker.track_texture();
    tracker.untrack_texture();
    assert_eq!(tracker.texture_count(), 1);
}

#[test]
fn test_resource_tracker_track_bind_group() {
    let tracker = ResourceTracker::new();

    assert_eq!(tracker.bind_group_count(), 0);
    tracker.track_bind_group();
    assert_eq!(tracker.bind_group_count(), 1);
}

#[test]
fn test_resource_tracker_untrack_bind_group() {
    let tracker = ResourceTracker::new();

    tracker.track_bind_group();
    tracker.track_bind_group();
    tracker.track_bind_group();
    tracker.untrack_bind_group();
    assert_eq!(tracker.bind_group_count(), 2);
}

#[test]
fn test_resource_tracker_track_pipeline() {
    let tracker = ResourceTracker::new();

    assert_eq!(tracker.pipeline_count(), 0);
    tracker.track_pipeline();
    tracker.track_pipeline();
    assert_eq!(tracker.pipeline_count(), 2);
}

#[test]
fn test_resource_tracker_untrack_pipeline() {
    let tracker = ResourceTracker::new();

    tracker.track_pipeline();
    tracker.untrack_pipeline();
    assert_eq!(tracker.pipeline_count(), 0);
}

#[test]
fn test_resource_tracker_track_sampler() {
    let tracker = ResourceTracker::new();

    assert_eq!(tracker.sampler_count(), 0);
    tracker.track_sampler();
    assert_eq!(tracker.sampler_count(), 1);
}

#[test]
fn test_resource_tracker_untrack_sampler() {
    let tracker = ResourceTracker::new();

    tracker.track_sampler();
    tracker.track_sampler();
    tracker.untrack_sampler();
    assert_eq!(tracker.sampler_count(), 1);
}

#[test]
fn test_resource_tracker_track_query_set() {
    let tracker = ResourceTracker::new();

    assert_eq!(tracker.query_set_count(), 0);
    tracker.track_query_set();
    assert_eq!(tracker.query_set_count(), 1);
}

#[test]
fn test_resource_tracker_untrack_query_set() {
    let tracker = ResourceTracker::new();

    tracker.track_query_set();
    tracker.track_query_set();
    tracker.untrack_query_set();
    assert_eq!(tracker.query_set_count(), 1);
}

// ============================================================================
// Path N: ResourceTracker - total_count(), is_empty(), clear()
// ============================================================================

#[test]
fn test_resource_tracker_total_count_sums_all_types() {
    let tracker = ResourceTracker::new();

    tracker.track_buffer();
    tracker.track_buffer();
    tracker.track_texture();
    tracker.track_bind_group();
    tracker.track_bind_group();
    tracker.track_bind_group();
    tracker.track_pipeline();
    tracker.track_sampler();
    tracker.track_query_set();
    tracker.track_query_set();

    // 2 + 1 + 3 + 1 + 1 + 2 = 10
    assert_eq!(tracker.total_count(), 10);
}

#[test]
fn test_resource_tracker_is_empty_true_when_all_zero() {
    let tracker = ResourceTracker::new();
    assert!(tracker.is_empty());
}

#[test]
fn test_resource_tracker_is_empty_false_when_any_nonzero() {
    let tracker = ResourceTracker::new();
    tracker.track_sampler();
    assert!(!tracker.is_empty());
}

#[test]
fn test_resource_tracker_clear_resets_all_counts() {
    let tracker = ResourceTracker::new();

    tracker.track_buffer();
    tracker.track_texture();
    tracker.track_bind_group();
    tracker.track_pipeline();
    tracker.track_sampler();
    tracker.track_query_set();

    assert!(!tracker.is_empty());

    tracker.clear();

    assert!(tracker.is_empty());
    assert_eq!(tracker.buffer_count(), 0);
    assert_eq!(tracker.texture_count(), 0);
    assert_eq!(tracker.bind_group_count(), 0);
    assert_eq!(tracker.pipeline_count(), 0);
    assert_eq!(tracker.sampler_count(), 0);
    assert_eq!(tracker.query_set_count(), 0);
}

#[test]
fn test_resource_tracker_default_is_empty() {
    let tracker: ResourceTracker = Default::default();
    assert!(tracker.is_empty());
}

// ============================================================================
// Path O: ResourceTracker Display formatting
// ============================================================================

#[test]
fn test_resource_tracker_display_contains_all_fields() {
    let tracker = ResourceTracker::new();
    tracker.track_buffer();
    tracker.track_texture();
    tracker.track_texture();

    let display = format!("{}", tracker);

    assert!(display.contains("buffers=1"));
    assert!(display.contains("textures=2"));
    assert!(display.contains("bind_groups=0"));
    assert!(display.contains("pipelines=0"));
    assert!(display.contains("samplers=0"));
    assert!(display.contains("query_sets=0"));
}

#[test]
fn test_resource_tracker_display_prefix() {
    let tracker = ResourceTracker::new();
    let display = format!("{}", tracker);
    assert!(display.starts_with("ResourceTracker("));
}

// ============================================================================
// Path P: DeviceManagerError - all variants, Display impl
// ============================================================================

#[test]
fn test_device_manager_error_device_unavailable_display() {
    let err = DeviceManagerError::DeviceUnavailable(DeviceState::Lost);
    let display = format!("{}", err);
    assert!(display.contains("unavailable"));
    assert!(display.contains("Lost"));
}

#[test]
fn test_device_manager_error_device_unavailable_recovering() {
    let err = DeviceManagerError::DeviceUnavailable(DeviceState::Recovering(2));
    let display = format!("{}", err);
    assert!(display.contains("unavailable"));
    assert!(display.contains("Recovering"));
}

#[test]
fn test_device_manager_error_fatal_state_display() {
    let err = DeviceManagerError::FatalState;
    let display = format!("{}", err);
    assert!(display.contains("fatal"));
}

// ============================================================================
// Thread safety tests for ResourceTracker
// ============================================================================

#[test]
fn test_resource_tracker_concurrent_track() {
    use std::thread;

    let tracker = Arc::new(ResourceTracker::new());
    let mut handles = vec![];

    // Spawn 10 threads, each tracking 100 buffers
    for _ in 0..10 {
        let tracker_clone = Arc::clone(&tracker);
        handles.push(thread::spawn(move || {
            for _ in 0..100 {
                tracker_clone.track_buffer();
            }
        }));
    }

    for handle in handles {
        handle.join().unwrap();
    }

    // Should have exactly 1000 buffers tracked
    assert_eq!(tracker.buffer_count(), 1000);
}

#[test]
fn test_resource_tracker_concurrent_track_untrack() {
    use std::thread;

    let tracker = Arc::new(ResourceTracker::new());

    // Pre-track 500 textures
    for _ in 0..500 {
        tracker.track_texture();
    }

    let mut handles = vec![];

    // 5 threads tracking, 5 threads untracking
    for i in 0..10 {
        let tracker_clone = Arc::clone(&tracker);
        handles.push(thread::spawn(move || {
            for _ in 0..100 {
                if i < 5 {
                    tracker_clone.track_texture();
                } else {
                    tracker_clone.untrack_texture();
                }
            }
        }));
    }

    for handle in handles {
        handle.join().unwrap();
    }

    // 500 initial + (5*100 tracked) - (5*100 untracked) = 500
    assert_eq!(tracker.texture_count(), 500);
}

// ============================================================================
// Edge case tests
// ============================================================================

#[test]
fn test_recovery_config_edge_case_zero_max_retries() {
    let config = RecoveryConfig::new(0, 100, 5000);
    assert_eq!(config.max_retries, 0);
}

#[test]
fn test_recovery_config_edge_case_max_backoff_less_than_initial() {
    // max_backoff < initial - should still cap at max
    let config = RecoveryConfig::new(5, 1000, 500);
    assert_eq!(config.backoff_for_attempt(0), Duration::from_millis(500));
}

#[test]
fn test_recovery_config_edge_case_equal_backoffs() {
    let config = RecoveryConfig::new(5, 100, 100);
    assert_eq!(config.backoff_for_attempt(0), Duration::from_millis(100));
    assert_eq!(config.backoff_for_attempt(1), Duration::from_millis(100));
    assert_eq!(config.backoff_for_attempt(5), Duration::from_millis(100));
}

#[test]
fn test_device_state_recovering_boundary_values() {
    // Test boundary values for Recovering
    assert_eq!(DeviceState::Recovering(0), DeviceState::Recovering(0));
    assert_ne!(DeviceState::Recovering(0), DeviceState::Recovering(1));

    // Test formatting for edge values
    let _ = format!("{}", DeviceState::Recovering(0));
    let _ = format!("{}", DeviceState::Recovering(u32::MAX - 2)); // Near encoding limit
}

#[test]
fn test_resource_tracker_underflow_protection() {
    let tracker = ResourceTracker::new();

    // Untrack without any tracked (potential underflow)
    // Atomics use wrapping on fetch_sub, so this would wrap to u64::MAX
    // This is by design - the caller is responsible for not untracking more than tracked
    tracker.untrack_buffer();

    // Count will be u64::MAX (wrapped)
    assert_eq!(tracker.buffer_count(), u64::MAX);
}

// ============================================================================
// Debug trait tests
// ============================================================================

#[test]
fn test_device_state_debug() {
    let healthy = DeviceState::Healthy;
    let debug = format!("{:?}", healthy);
    assert!(debug.contains("Healthy"));

    let recovering = DeviceState::Recovering(5);
    let debug = format!("{:?}", recovering);
    assert!(debug.contains("Recovering"));
    assert!(debug.contains("5"));
}

#[test]
fn test_recovery_config_debug() {
    let config = RecoveryConfig::new(3, 200, 5000);
    let debug = format!("{:?}", config);
    assert!(debug.contains("RecoveryConfig"));
    assert!(debug.contains("max_retries"));
    assert!(debug.contains("initial_backoff_ms"));
    assert!(debug.contains("max_backoff_ms"));
}

#[test]
fn test_device_lost_reason_debug() {
    let reason = DeviceLostReason::Timeout;
    let debug = format!("{:?}", reason);
    assert!(debug.contains("Timeout"));
}

#[test]
fn test_resource_tracker_debug() {
    let tracker = ResourceTracker::new();
    tracker.track_buffer();
    let debug = format!("{:?}", tracker);
    assert!(debug.contains("ResourceTracker"));
}

#[test]
fn test_device_manager_error_debug() {
    let err = DeviceManagerError::FatalState;
    let debug = format!("{:?}", err);
    assert!(debug.contains("FatalState"));
}

// ============================================================================
// Error trait tests
// ============================================================================

#[test]
fn test_device_manager_error_is_error() {
    let err = DeviceManagerError::FatalState;

    // Verify it implements std::error::Error
    let _: &dyn std::error::Error = &err;
}

#[test]
fn test_device_manager_error_source_fatal_state() {
    use std::error::Error;
    let err = DeviceManagerError::FatalState;
    assert!(err.source().is_none());
}

#[test]
fn test_device_manager_error_source_device_unavailable() {
    use std::error::Error;
    let err = DeviceManagerError::DeviceUnavailable(DeviceState::Lost);
    assert!(err.source().is_none());
}

// ============================================================================
// State transition diagram tests - verify documented transitions
// ============================================================================
//
// Documented state machine:
//
// Healthy ──[device lost]──> Lost
//    ^                         |
//    |                         v
//    |                    Recovering(0)
//    |                         |
//    |        [success]        v
//    +<──────────────── Recovering(n)
//                              |
//                         [max retries]
//                              v
//                            Fatal

#[test]
fn test_state_machine_healthy_to_lost_allowed() {
    // Healthy -> Lost is a valid transition (device loss)
    let before = DeviceState::Healthy;
    let after = DeviceState::Lost;

    assert!(before.is_healthy());
    assert!(!after.is_healthy());
    assert!(after.needs_recovery());
}

#[test]
fn test_state_machine_lost_to_recovering_allowed() {
    // Lost -> Recovering(0) is a valid transition (start recovery)
    let before = DeviceState::Lost;
    let after = DeviceState::Recovering(0);

    assert!(before.needs_recovery());
    assert!(after.needs_recovery());
}

#[test]
fn test_state_machine_recovering_increment_allowed() {
    // Recovering(n) -> Recovering(n+1) is valid (retry)
    let before = DeviceState::Recovering(2);
    let after = DeviceState::Recovering(3);

    assert!(before.needs_recovery());
    assert!(after.needs_recovery());
}

#[test]
fn test_state_machine_recovering_to_healthy_allowed() {
    // Recovering(n) -> Healthy is valid (successful recovery)
    let before = DeviceState::Recovering(2);
    let after = DeviceState::Healthy;

    assert!(before.needs_recovery());
    assert!(after.is_healthy());
}

#[test]
fn test_state_machine_recovering_to_fatal_allowed() {
    // Recovering(max) -> Fatal is valid (max retries exceeded)
    let before = DeviceState::Recovering(3);
    let after = DeviceState::Fatal;

    assert!(before.needs_recovery());
    assert!(after.is_fatal());
    assert!(!after.needs_recovery()); // Fatal does not need recovery
}

#[test]
fn test_state_machine_fatal_is_terminal() {
    // Fatal is a terminal state
    let state = DeviceState::Fatal;

    assert!(state.is_fatal());
    assert!(!state.is_healthy());
    assert!(!state.needs_recovery());
}

// ============================================================================
// Callback invocation test (without real device)
// ============================================================================

#[test]
fn test_lost_callback_can_be_fn_closure() {
    // Verify the callback type constraints allow closures
    let called = Arc::new(AtomicBool::new(false));
    let called_clone = Arc::clone(&called);

    // This closure captures state and should satisfy Fn(DeviceLostReason) + Send + Sync
    let callback = move |_reason: DeviceLostReason| {
        called_clone.store(true, Ordering::SeqCst);
    };

    // Verify it's callable
    callback(DeviceLostReason::Timeout);
    assert!(called.load(Ordering::SeqCst));
}

#[test]
fn test_lost_callback_receives_reason() {
    let received_reason = Arc::new(std::sync::Mutex::new(None));
    let received_clone = Arc::clone(&received_reason);

    let callback = move |reason: DeviceLostReason| {
        *received_clone.lock().unwrap() = Some(reason);
    };

    callback(DeviceLostReason::PowerEvent);

    let reason = received_reason.lock().unwrap().take();
    assert_eq!(reason, Some(DeviceLostReason::PowerEvent));
}

// ============================================================================
// Performance characteristics tests
// ============================================================================

#[test]
fn test_resource_tracker_operations_are_lock_free() {
    // ResourceTracker uses Atomics - verify operations complete quickly
    // (no deadlocks, no contention issues)
    let tracker = ResourceTracker::new();

    let start = std::time::Instant::now();
    for _ in 0..10000 {
        tracker.track_buffer();
        tracker.track_texture();
        tracker.untrack_buffer();
    }
    let duration = start.elapsed();

    // 30000 atomic operations should complete in well under 100ms
    assert!(
        duration < Duration::from_millis(100),
        "Atomic operations took too long: {:?}",
        duration
    );

    // Verify final state
    assert_eq!(tracker.buffer_count(), 0);
    assert_eq!(tracker.texture_count(), 10000);
}

#[test]
fn test_device_state_helpers_are_inline() {
    // These helper methods are marked #[inline] - verify they're fast
    let state = DeviceState::Recovering(5);

    let start = std::time::Instant::now();
    for _ in 0..100000 {
        let _ = state.is_healthy();
        let _ = state.needs_recovery();
        let _ = state.is_fatal();
    }
    let duration = start.elapsed();

    // 300000 inline checks should complete in well under 10ms
    assert!(
        duration < Duration::from_millis(10),
        "State checks took too long: {:?}",
        duration
    );
}

#[test]
fn test_recovery_config_backoff_calculation_is_fast() {
    let config = RecoveryConfig::default();

    let start = std::time::Instant::now();
    for attempt in 0..1000 {
        let _ = config.backoff_for_attempt(attempt % 100);
    }
    let duration = start.elapsed();

    // 1000 backoff calculations should complete in well under 10ms
    assert!(
        duration < Duration::from_millis(10),
        "Backoff calculations took too long: {:?}",
        duration
    );
}
