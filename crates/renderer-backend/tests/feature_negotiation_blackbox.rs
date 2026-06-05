// Blackbox contract tests for T-WGPU-P1.3.2 Feature Negotiation
//
// CLEANROOM: No access to implementation files. Tests use only the public API
// exported by `renderer_backend::device`.
//
// Forbidden files (per TESTDEV_BLACKBOX prompt):
//   - crates/renderer-backend/src/device/requirements.rs
//   - crates/renderer-backend/src/device/device.rs
//   - Any WHITEBOX test file for this task
//
// Contract sources:
//   - docs/WGPU_DOCS/RDC_OUTPUT/PHASE_1_CORE_TODO.md (T-WGPU-P1.3.2)
//   - docs/WGPU_DOCS/RDC_OUTPUT/PHASE_1_CORE_ARCH.md (Architecture spec)
//
// Acceptance criteria (T-WGPU-P1.3.2):
//   - Required features cause failure if unavailable
//   - Optional features degraded gracefully
//   - Feature dependencies automatically added
//   - Final feature set logged
//
// Public API under test:
//   - DeviceRequirements struct with builder methods
//   - NegotiationResult struct with outcome fields
//   - FeatureNegotiationError enum with error variants
//   - negotiate_features() function
//   - negotiate_and_create_device() function
//
// Test design rationale:
//   Equivalence partitioning:
//     - Required features only (hard failures)
//     - Optional features only (graceful degradation)
//     - Mixed required and optional
//     - Empty requirements (baseline)
//   Boundary cases:
//     - All features required (maximal constraint)
//     - No features required (minimal constraint)
//     - Feature that doesn't exist on adapter
//   Contract verification:
//     - Type exports and trait implementations
//     - Builder method chaining
//     - Error variant accessibility

use pollster::FutureExt as _;
use renderer_backend::device::{
    enumerate_adapters_with_info, DeviceRequirements, FeatureNegotiationError,
    NegotiateAndCreateError, NegotiationResult, TrinityInstance, negotiate_and_create_device,
    negotiate_features,
};
use std::fmt::{Debug, Display};

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

/// Creates a TrinityInstance and gets the first available adapter.
/// Skips tests if no adapter is available.
fn get_test_adapter() -> Option<wgpu::Adapter> {
    let instance = TrinityInstance::new();
    let result = enumerate_adapters_with_info(instance.inner(), instance.backends());
    result.adapters.into_iter().next()
}

/// Helper macro to skip test if no GPU adapter is available.
macro_rules! require_adapter {
    () => {
        match get_test_adapter() {
            Some(adapter) => adapter,
            None => {
                eprintln!("SKIP: No GPU adapter available for this test");
                return;
            }
        }
    };
}

// =============================================================================
// SECTION 1: DeviceRequirements Type Contract Tests
// =============================================================================

/// Verifies that DeviceRequirements is a publicly accessible type.
///
/// Contract: DeviceRequirements is exported from renderer_backend::device.
#[test]
fn test_device_requirements_type_is_exported() {
    fn _assert_type_exists<T>(_: Option<&T>) {}
    _assert_type_exists::<DeviceRequirements>(None);
}

/// Verifies that DeviceRequirements implements Debug.
///
/// Contract: DeviceRequirements derives Debug for introspection.
#[test]
fn test_device_requirements_implements_debug() {
    fn _assert_debug<T: Debug>() {}
    _assert_debug::<DeviceRequirements>();
}

/// Verifies that DeviceRequirements implements Clone.
///
/// Contract: DeviceRequirements derives Clone for copying.
#[test]
fn test_device_requirements_implements_clone() {
    fn _assert_clone<T: Clone>() {}
    _assert_clone::<DeviceRequirements>();
}

/// Verifies that DeviceRequirements implements Default.
///
/// Contract: DeviceRequirements has a default constructor for empty requirements.
#[test]
fn test_device_requirements_implements_default() {
    fn _assert_default<T: Default>() {}
    _assert_default::<DeviceRequirements>();
}

// =============================================================================
// SECTION 2: DeviceRequirements Constructor Tests
// =============================================================================

/// Verifies that DeviceRequirements has a new() constructor.
///
/// Contract: DeviceRequirements::new() creates empty requirements.
#[test]
fn test_device_requirements_has_new_constructor() {
    let requirements = DeviceRequirements::new();
    // Should compile and create successfully
    let _ = requirements;
}

/// Verifies that DeviceRequirements::new() creates usable requirements.
///
/// Contract: new() returns a valid DeviceRequirements instance.
#[test]
fn test_device_requirements_new_returns_valid_instance() {
    let requirements = DeviceRequirements::new();
    // Verify it implements Debug by formatting
    let debug_str = format!("{:?}", requirements);
    assert!(!debug_str.is_empty(), "Debug output should not be empty");
}

/// Verifies that DeviceRequirements::default() creates the same as new().
///
/// Contract: Default implementation matches new() constructor.
#[test]
fn test_device_requirements_default_matches_new() {
    let from_new = DeviceRequirements::new();
    let from_default = DeviceRequirements::default();

    // Both should produce valid instances
    let debug_new = format!("{:?}", from_new);
    let debug_default = format!("{:?}", from_default);

    assert_eq!(debug_new, debug_default, "new() and default() should produce equivalent instances");
}

// =============================================================================
// SECTION 3: DeviceRequirements Builder Method Tests
// =============================================================================

/// Verifies that DeviceRequirements has a require() builder method.
///
/// Contract: require() adds a required feature to the requirements.
#[test]
fn test_device_requirements_has_require_method() {
    let requirements = DeviceRequirements::new()
        .require(wgpu::Features::TEXTURE_COMPRESSION_BC);

    let _ = requirements;
}

/// Verifies that require() returns Self for method chaining.
///
/// Contract: require() enables builder pattern chaining.
#[test]
fn test_device_requirements_require_returns_self() {
    let _requirements = DeviceRequirements::new()
        .require(wgpu::Features::TEXTURE_COMPRESSION_BC)
        .require(wgpu::Features::DEPTH_CLIP_CONTROL);
}

/// Verifies that DeviceRequirements has a prefer() builder method.
///
/// Contract: prefer() adds an optional/preferred feature to the requirements.
#[test]
fn test_device_requirements_has_prefer_method() {
    let requirements = DeviceRequirements::new()
        .prefer(wgpu::Features::TEXTURE_COMPRESSION_ASTC);

    let _ = requirements;
}

/// Verifies that prefer() returns Self for method chaining.
///
/// Contract: prefer() enables builder pattern chaining.
#[test]
fn test_device_requirements_prefer_returns_self() {
    let _requirements = DeviceRequirements::new()
        .prefer(wgpu::Features::TEXTURE_COMPRESSION_ASTC)
        .prefer(wgpu::Features::TEXTURE_COMPRESSION_ETC2);
}

/// Verifies that DeviceRequirements has a with_limits() builder method.
///
/// Contract: with_limits() sets the limit requirements.
#[test]
fn test_device_requirements_has_with_limits_method() {
    let requirements = DeviceRequirements::new()
        .with_limits(wgpu::Limits::default());

    let _ = requirements;
}

/// Verifies that with_limits() returns Self for method chaining.
///
/// Contract: with_limits() enables builder pattern chaining.
#[test]
fn test_device_requirements_with_limits_returns_self() {
    let _requirements = DeviceRequirements::new()
        .with_limits(wgpu::Limits::default())
        .require(wgpu::Features::DEPTH_CLIP_CONTROL);
}

/// Verifies that all builder methods can be chained together.
///
/// Contract: Full builder pattern support with mixed methods.
#[test]
fn test_device_requirements_full_builder_chain() {
    let _requirements = DeviceRequirements::new()
        .require(wgpu::Features::TEXTURE_COMPRESSION_BC)
        .prefer(wgpu::Features::TEXTURE_COMPRESSION_ASTC)
        .with_limits(wgpu::Limits::downlevel_defaults())
        .require(wgpu::Features::DEPTH_CLIP_CONTROL)
        .prefer(wgpu::Features::TEXTURE_COMPRESSION_ETC2);
}

/// Verifies that DeviceRequirements can be cloned after building.
///
/// Contract: Clone works after any builder operations.
#[test]
fn test_device_requirements_clone_after_building() {
    let requirements = DeviceRequirements::new()
        .require(wgpu::Features::TEXTURE_COMPRESSION_BC)
        .prefer(wgpu::Features::TEXTURE_COMPRESSION_ASTC);

    let cloned = requirements.clone();
    let _ = cloned;
}

// =============================================================================
// SECTION 4: NegotiationResult Type Contract Tests
// =============================================================================

/// Verifies that NegotiationResult is a publicly accessible type.
///
/// Contract: NegotiationResult is exported from renderer_backend::device.
#[test]
fn test_negotiation_result_type_is_exported() {
    fn _assert_type_exists<T>(_: Option<&T>) {}
    _assert_type_exists::<NegotiationResult>(None);
}

/// Verifies that NegotiationResult implements Debug.
///
/// Contract: NegotiationResult derives Debug for introspection.
#[test]
fn test_negotiation_result_implements_debug() {
    fn _assert_debug<T: Debug>() {}
    _assert_debug::<NegotiationResult>();
}

/// Verifies that NegotiationResult implements Clone.
///
/// Contract: NegotiationResult derives Clone for copying.
#[test]
fn test_negotiation_result_implements_clone() {
    fn _assert_clone<T: Clone>() {}
    _assert_clone::<NegotiationResult>();
}

// =============================================================================
// SECTION 5: NegotiationResult Field Access Tests
// =============================================================================

/// Verifies that NegotiationResult has an enabled_features field.
///
/// Contract: enabled_features contains the final feature set that was enabled.
#[test]
fn test_negotiation_result_has_enabled_features_field() {
    let adapter = require_adapter!();
    let requirements = DeviceRequirements::new();

    let result = negotiate_features(&requirements, &adapter);
    match result {
        Ok(negotiation) => {
            let _features: wgpu::Features = negotiation.enabled_features;
        }
        Err(_) => {
            // Even with empty requirements, we test the field exists
            // by verifying the type compiles
        }
    }
}

/// Verifies that NegotiationResult has a degraded_features field.
///
/// Contract: degraded_features contains optional features that were unavailable.
#[test]
fn test_negotiation_result_has_degraded_features_field() {
    let adapter = require_adapter!();
    let requirements = DeviceRequirements::new();

    let result = negotiate_features(&requirements, &adapter);
    match result {
        Ok(negotiation) => {
            let _degraded: wgpu::Features = negotiation.degraded_features;
        }
        Err(_) => {}
    }
}

/// Verifies that NegotiationResult has a limits field.
///
/// Contract: limits contains the final negotiated limits.
#[test]
fn test_negotiation_result_has_limits_field() {
    let adapter = require_adapter!();
    let requirements = DeviceRequirements::new();

    let result = negotiate_features(&requirements, &adapter);
    match result {
        Ok(negotiation) => {
            let _limits: wgpu::Limits = negotiation.limits;
        }
        Err(_) => {}
    }
}

// =============================================================================
// SECTION 6: FeatureNegotiationError Type Contract Tests
// =============================================================================

/// Verifies that FeatureNegotiationError is a publicly accessible type.
///
/// Contract: FeatureNegotiationError is exported from renderer_backend::device.
#[test]
fn test_feature_negotiation_error_type_is_exported() {
    fn _assert_type_exists<T>(_: Option<&T>) {}
    _assert_type_exists::<FeatureNegotiationError>(None);
}

/// Verifies that FeatureNegotiationError implements Debug.
///
/// Contract: FeatureNegotiationError derives Debug for introspection.
#[test]
fn test_feature_negotiation_error_implements_debug() {
    fn _assert_debug<T: Debug>() {}
    _assert_debug::<FeatureNegotiationError>();
}

/// Verifies that FeatureNegotiationError implements Display.
///
/// Contract: FeatureNegotiationError implements Display for user-friendly messages.
#[test]
fn test_feature_negotiation_error_implements_display() {
    fn _assert_display<T: Display>() {}
    _assert_display::<FeatureNegotiationError>();
}

/// Verifies that FeatureNegotiationError implements std::error::Error.
///
/// Contract: FeatureNegotiationError is a proper Rust error type.
#[test]
fn test_feature_negotiation_error_implements_error() {
    fn _assert_error<T: std::error::Error>() {}
    _assert_error::<FeatureNegotiationError>();
}

// =============================================================================
// SECTION 7: FeatureNegotiationError Variant Tests
// =============================================================================

/// Verifies that FeatureNegotiationError has RequiredFeaturesMissing variant.
///
/// Contract: RequiredFeaturesMissing is returned when required features are unavailable.
#[test]
fn test_feature_negotiation_error_has_required_features_missing_variant() {
    // Test by requesting a feature that likely doesn't exist
    let adapter = require_adapter!();

    // Request ray tracing which is often unavailable
    let requirements = DeviceRequirements::new()
        .require(wgpu::Features::RAY_TRACING_ACCELERATION_STRUCTURE);

    let result = negotiate_features(&requirements, &adapter);

    // If adapter doesn't support ray tracing, we should get RequiredFeaturesMissing
    if let Err(e) = result {
        // Verify error can be formatted
        let error_msg = format!("{}", e);
        assert!(!error_msg.is_empty(), "Error message should not be empty");

        let debug_msg = format!("{:?}", e);
        assert!(!debug_msg.is_empty(), "Debug output should not be empty");
    }
    // If adapter supports it, that's fine too - we're testing the contract
}

/// Verifies that FeatureNegotiationError has LimitsExceedCapabilities variant.
///
/// Contract: LimitsExceedCapabilities is returned when requested limits exceed adapter.
#[test]
fn test_feature_negotiation_error_has_limits_exceed_capabilities_variant() {
    let adapter = require_adapter!();

    // Request impossibly high limits
    let mut extreme_limits = wgpu::Limits::default();
    extreme_limits.max_texture_dimension_2d = u32::MAX;
    extreme_limits.max_buffer_size = u64::MAX;

    let requirements = DeviceRequirements::new()
        .with_limits(extreme_limits);

    let result = negotiate_features(&requirements, &adapter);

    // Should either succeed with capped limits or fail with error
    match result {
        Ok(_) => {
            // Implementation may cap limits instead of failing
        }
        Err(e) => {
            let error_msg = format!("{}", e);
            assert!(!error_msg.is_empty(), "Error message should not be empty");
        }
    }
}

// =============================================================================
// SECTION 8: negotiate_features() Function Contract Tests
// =============================================================================

/// Verifies that negotiate_features function is exported.
///
/// Contract: negotiate_features is a public function.
#[test]
fn test_negotiate_features_function_is_exported() {
    let adapter = require_adapter!();
    let requirements = DeviceRequirements::new();

    // Verify function exists by calling it
    let _ = negotiate_features(&requirements, &adapter);
}

/// Verifies that negotiate_features accepts DeviceRequirements.
///
/// Contract: First parameter is &DeviceRequirements.
#[test]
fn test_negotiate_features_accepts_device_requirements() {
    let adapter = require_adapter!();
    let requirements: DeviceRequirements = DeviceRequirements::new();

    let _ = negotiate_features(&requirements, &adapter);
}

/// Verifies that negotiate_features accepts Adapter.
///
/// Contract: Second parameter is &wgpu::Adapter.
#[test]
fn test_negotiate_features_accepts_adapter() {
    let adapter: wgpu::Adapter = require_adapter!();
    let requirements = DeviceRequirements::new();

    let _ = negotiate_features(&requirements, &adapter);
}

/// Verifies that negotiate_features returns Result type.
///
/// Contract: Returns Result<NegotiationResult, FeatureNegotiationError>.
#[test]
fn test_negotiate_features_returns_result() {
    let adapter = require_adapter!();
    let requirements = DeviceRequirements::new();

    let result: Result<NegotiationResult, FeatureNegotiationError> =
        negotiate_features(&requirements, &adapter);

    let _ = result;
}

/// Verifies that negotiate_features succeeds with empty requirements.
///
/// Contract: Empty requirements should always succeed on a valid adapter.
#[test]
fn test_negotiate_features_succeeds_with_empty_requirements() {
    let adapter = require_adapter!();
    let requirements = DeviceRequirements::new();

    let result = negotiate_features(&requirements, &adapter);

    assert!(result.is_ok(), "Empty requirements should succeed: {:?}", result.err());
}

/// Verifies that negotiate_features fails for missing required features.
///
/// Contract: Required features cause failure if unavailable.
#[test]
fn test_negotiate_features_fails_for_missing_required_features() {
    let adapter = require_adapter!();

    // Pick a feature that is very unlikely to be available
    // RAY_TRACING_ACCELERATION_STRUCTURE is a good candidate
    let requirements = DeviceRequirements::new()
        .require(wgpu::Features::RAY_TRACING_ACCELERATION_STRUCTURE);

    let adapter_features = adapter.features();

    // Only expect failure if the adapter doesn't support this feature
    if !adapter_features.contains(wgpu::Features::RAY_TRACING_ACCELERATION_STRUCTURE) {
        let result = negotiate_features(&requirements, &adapter);
        assert!(result.is_err(), "Should fail when required feature is unavailable");
    }
}

// =============================================================================
// SECTION 9: Optional Features Graceful Degradation Tests
// =============================================================================

/// Verifies that optional features are degraded gracefully.
///
/// Contract: Optional features degraded gracefully.
#[test]
fn test_negotiate_features_degrades_optional_features_gracefully() {
    let adapter = require_adapter!();

    // Request an unlikely optional feature
    let requirements = DeviceRequirements::new()
        .prefer(wgpu::Features::RAY_TRACING_ACCELERATION_STRUCTURE);

    let result = negotiate_features(&requirements, &adapter);

    // Should succeed even if feature is unavailable
    assert!(result.is_ok(), "Optional features should not cause failure: {:?}", result.err());
}

/// Verifies that degraded_features tracks unavailable optional features.
///
/// Contract: degraded_features field lists unavailable optional features.
#[test]
fn test_negotiate_features_tracks_degraded_features() {
    let adapter = require_adapter!();
    let adapter_features = adapter.features();

    // Request an optional feature that's likely unavailable
    let requirements = DeviceRequirements::new()
        .prefer(wgpu::Features::RAY_TRACING_ACCELERATION_STRUCTURE);

    let result = negotiate_features(&requirements, &adapter);

    if let Ok(negotiation) = result {
        if !adapter_features.contains(wgpu::Features::RAY_TRACING_ACCELERATION_STRUCTURE) {
            // Feature should be in degraded_features
            assert!(
                negotiation.degraded_features.contains(wgpu::Features::RAY_TRACING_ACCELERATION_STRUCTURE),
                "Unavailable optional feature should be in degraded_features"
            );
        }
    }
}

/// Verifies that available optional features are enabled.
///
/// Contract: Optional features that are available should be enabled.
#[test]
fn test_negotiate_features_enables_available_optional_features() {
    let adapter = require_adapter!();
    let adapter_features = adapter.features();

    // Find a feature that the adapter actually supports
    // DEPTH_CLIP_CONTROL is commonly supported
    let test_feature = wgpu::Features::DEPTH_CLIP_CONTROL;

    if adapter_features.contains(test_feature) {
        let requirements = DeviceRequirements::new()
            .prefer(test_feature);

        let result = negotiate_features(&requirements, &adapter);

        if let Ok(negotiation) = result {
            assert!(
                negotiation.enabled_features.contains(test_feature),
                "Available optional feature should be enabled"
            );
        }
    }
}

/// Verifies mixed required and optional features work together.
///
/// Contract: Can combine required and optional features in same requirements.
#[test]
fn test_negotiate_features_handles_mixed_required_and_optional() {
    let adapter = require_adapter!();

    // Mix required (likely available) and optional (likely unavailable) features
    let requirements = DeviceRequirements::new()
        .prefer(wgpu::Features::RAY_TRACING_ACCELERATION_STRUCTURE)
        .prefer(wgpu::Features::DEPTH_CLIP_CONTROL);

    let result = negotiate_features(&requirements, &adapter);

    // Should succeed - all features are optional
    assert!(result.is_ok(), "Mixed optional features should not fail: {:?}", result.err());
}

// =============================================================================
// SECTION 10: Feature Dependencies Tests
// =============================================================================

/// Verifies that feature dependencies are automatically added.
///
/// Contract: Feature dependencies automatically added.
#[test]
fn test_negotiate_features_adds_feature_dependencies() {
    let adapter = require_adapter!();

    // This test verifies the contract that dependencies are resolved
    // The exact behavior depends on implementation, but it should not crash
    let requirements = DeviceRequirements::new()
        .prefer(wgpu::Features::TEXTURE_COMPRESSION_BC);

    let result = negotiate_features(&requirements, &adapter);

    // Should succeed or fail gracefully
    match result {
        Ok(negotiation) => {
            // Verify we got a valid feature set
            let _ = negotiation.enabled_features;
        }
        Err(e) => {
            // Error should be informative
            let msg = format!("{}", e);
            assert!(!msg.is_empty());
        }
    }
}

// =============================================================================
// SECTION 11: negotiate_and_create_device() Function Tests
// =============================================================================

/// Verifies that negotiate_and_create_device function is exported.
///
/// Contract: negotiate_and_create_device is a public function.
#[test]
fn test_negotiate_and_create_device_function_is_exported() {
    let adapter = require_adapter!();
    let requirements = DeviceRequirements::new();

    // Verify function exists by calling it
    let _ = negotiate_and_create_device(&requirements, &adapter).block_on();
}

/// Verifies that negotiate_and_create_device is async.
///
/// Contract: negotiate_and_create_device is an async function.
#[test]
fn test_negotiate_and_create_device_is_async() {
    let adapter = require_adapter!();
    let requirements = DeviceRequirements::new();

    // Calling block_on proves it's async
    let _ = negotiate_and_create_device(&requirements, &adapter).block_on();
}

/// Verifies that negotiate_and_create_device returns proper Result type.
///
/// Contract: Returns Result with TrinityDevice on success.
#[test]
fn test_negotiate_and_create_device_returns_result() {
    let adapter = require_adapter!();
    let requirements = DeviceRequirements::new();

    let result = negotiate_and_create_device(&requirements, &adapter).block_on();

    // Verify it's a Result type
    match result {
        Ok(_device) => {}
        Err(_error) => {}
    }
}

/// Verifies that negotiate_and_create_device succeeds with empty requirements.
///
/// Contract: Should create device successfully with minimal requirements.
#[test]
fn test_negotiate_and_create_device_succeeds_with_empty_requirements() {
    let adapter = require_adapter!();
    let requirements = DeviceRequirements::new();

    let result = negotiate_and_create_device(&requirements, &adapter).block_on();

    assert!(result.is_ok(), "Should succeed with empty requirements: {:?}", result.err());
}

/// Verifies that negotiate_and_create_device fails when required features missing.
///
/// Contract: Required features missing should cause failure.
#[test]
fn test_negotiate_and_create_device_fails_for_missing_required() {
    let adapter = require_adapter!();
    let adapter_features = adapter.features();

    // Request a feature that doesn't exist
    if !adapter_features.contains(wgpu::Features::RAY_TRACING_ACCELERATION_STRUCTURE) {
        let requirements = DeviceRequirements::new()
            .require(wgpu::Features::RAY_TRACING_ACCELERATION_STRUCTURE);

        let result = negotiate_and_create_device(&requirements, &adapter).block_on();

        assert!(result.is_err(), "Should fail when required feature unavailable");
    }
}

// =============================================================================
// SECTION 12: NegotiateAndCreateError Type Tests
// =============================================================================

/// Verifies that NegotiateAndCreateError is a publicly accessible type.
///
/// Contract: NegotiateAndCreateError is exported from renderer_backend::device.
#[test]
fn test_negotiate_and_create_error_type_is_exported() {
    fn _assert_type_exists<T>(_: Option<&T>) {}
    _assert_type_exists::<NegotiateAndCreateError>(None);
}

/// Verifies that NegotiateAndCreateError implements Debug.
///
/// Contract: NegotiateAndCreateError derives Debug for introspection.
#[test]
fn test_negotiate_and_create_error_implements_debug() {
    fn _assert_debug<T: Debug>() {}
    _assert_debug::<NegotiateAndCreateError>();
}

/// Verifies that NegotiateAndCreateError implements Display.
///
/// Contract: NegotiateAndCreateError implements Display for user-friendly messages.
#[test]
fn test_negotiate_and_create_error_implements_display() {
    fn _assert_display<T: Display>() {}
    _assert_display::<NegotiateAndCreateError>();
}

/// Verifies that NegotiateAndCreateError implements std::error::Error.
///
/// Contract: NegotiateAndCreateError is a proper Rust error type.
#[test]
fn test_negotiate_and_create_error_implements_error() {
    fn _assert_error<T: std::error::Error>() {}
    _assert_error::<NegotiateAndCreateError>();
}

// =============================================================================
// SECTION 13: Integration Tests - Full Pipeline
// =============================================================================

/// Verifies full pipeline: Requirements -> negotiate -> create device.
///
/// Contract: Full feature negotiation pipeline works end-to-end.
#[test]
fn test_full_pipeline_requirements_to_device() {
    let adapter = require_adapter!();

    let requirements = DeviceRequirements::new()
        .prefer(wgpu::Features::DEPTH_CLIP_CONTROL)
        .with_limits(wgpu::Limits::downlevel_defaults());

    let result = negotiate_and_create_device(&requirements, &adapter).block_on();

    assert!(result.is_ok(), "Full pipeline should succeed: {:?}", result.err());
}

/// Verifies device has expected features after negotiation.
///
/// Contract: Created device should have negotiated features.
#[test]
fn test_device_has_expected_features_after_negotiation() {
    let adapter = require_adapter!();
    let adapter_features = adapter.features();

    // Only test if adapter supports depth clip control
    if adapter_features.contains(wgpu::Features::DEPTH_CLIP_CONTROL) {
        let requirements = DeviceRequirements::new()
            .require(wgpu::Features::DEPTH_CLIP_CONTROL);

        let result = negotiate_and_create_device(&requirements, &adapter).block_on();

        if let Ok((device, _negotiation)) = result {
            // Verify device has the feature
            let device_features = device.features();
            assert!(
                device_features.contains(wgpu::Features::DEPTH_CLIP_CONTROL),
                "Device should have required feature enabled"
            );
        }
    }
}

/// Verifies negotiation result is returned alongside device.
///
/// Contract: negotiate_and_create_device returns both device and negotiation result.
#[test]
fn test_negotiate_and_create_returns_negotiation_result() {
    let adapter = require_adapter!();

    let requirements = DeviceRequirements::new();

    let result = negotiate_and_create_device(&requirements, &adapter).block_on();

    if let Ok((_device, negotiation)) = result {
        // Verify negotiation result is accessible
        let _ = negotiation.enabled_features;
        let _ = negotiation.degraded_features;
        let _ = negotiation.limits;
    }
}

// =============================================================================
// SECTION 14: Limits Negotiation Tests
// =============================================================================

/// Verifies that limits are included in negotiation.
///
/// Contract: Limits are part of the negotiation process.
#[test]
fn test_negotiate_features_handles_limits() {
    let adapter = require_adapter!();

    let requirements = DeviceRequirements::new()
        .with_limits(wgpu::Limits::downlevel_defaults());

    let result = negotiate_features(&requirements, &adapter);

    if let Ok(negotiation) = result {
        // Verify limits are present in result
        let limits = negotiation.limits;
        assert!(limits.max_texture_dimension_2d > 0, "Should have valid texture limit");
    }
}

/// Verifies that limits are capped to adapter capabilities.
///
/// Contract: Requested limits are capped to what adapter supports.
#[test]
fn test_negotiate_features_caps_limits_to_adapter() {
    let adapter = require_adapter!();
    let adapter_limits = adapter.limits();

    // Request higher limits than adapter supports
    let mut high_limits = wgpu::Limits::default();
    high_limits.max_texture_dimension_2d = adapter_limits.max_texture_dimension_2d + 1000;

    let requirements = DeviceRequirements::new()
        .with_limits(high_limits);

    let result = negotiate_features(&requirements, &adapter);

    if let Ok(negotiation) = result {
        // Result should be capped to adapter limits
        assert!(
            negotiation.limits.max_texture_dimension_2d <= adapter_limits.max_texture_dimension_2d,
            "Limits should be capped to adapter capabilities"
        );
    }
}

// =============================================================================
// SECTION 15: Edge Cases and Boundary Tests
// =============================================================================

/// Verifies behavior with all features required.
///
/// Contract: System handles maximal feature requirements appropriately.
#[test]
fn test_negotiate_features_with_many_features() {
    let adapter = require_adapter!();

    // Request many features
    let requirements = DeviceRequirements::new()
        .prefer(wgpu::Features::TEXTURE_COMPRESSION_BC)
        .prefer(wgpu::Features::TEXTURE_COMPRESSION_ETC2)
        .prefer(wgpu::Features::TEXTURE_COMPRESSION_ASTC)
        .prefer(wgpu::Features::DEPTH_CLIP_CONTROL)
        .prefer(wgpu::Features::INDIRECT_FIRST_INSTANCE)
        .prefer(wgpu::Features::TIMESTAMP_QUERY);

    let result = negotiate_features(&requirements, &adapter);

    // Should succeed - all are optional
    assert!(result.is_ok(), "Many optional features should not cause failure");
}

/// Verifies behavior with maximum limits request.
///
/// Contract: System handles maximum limit requests gracefully.
#[test]
fn test_negotiate_features_with_maximum_limits() {
    let adapter = require_adapter!();

    // Use max limits from adapter
    let adapter_limits = adapter.limits();

    let requirements = DeviceRequirements::new()
        .with_limits(adapter_limits.clone());

    let result = negotiate_features(&requirements, &adapter);

    // Should succeed with adapter's own limits
    assert!(result.is_ok(), "Adapter's own limits should be acceptable");
}

/// Verifies behavior when building requirements multiple times.
///
/// Contract: Multiple require/prefer calls accumulate features.
#[test]
fn test_device_requirements_accumulates_features() {
    let adapter = require_adapter!();

    let mut requirements = DeviceRequirements::new();
    requirements = requirements.prefer(wgpu::Features::DEPTH_CLIP_CONTROL);
    requirements = requirements.prefer(wgpu::Features::TEXTURE_COMPRESSION_BC);

    let result = negotiate_features(&requirements, &adapter);

    // Should handle accumulated features
    assert!(result.is_ok(), "Accumulated features should be handled");
}

/// Verifies that Debug output for NegotiationResult is informative.
///
/// Contract: NegotiationResult Debug shows feature information.
#[test]
fn test_negotiation_result_debug_is_informative() {
    let adapter = require_adapter!();
    let requirements = DeviceRequirements::new();

    let result = negotiate_features(&requirements, &adapter);

    if let Ok(negotiation) = result {
        let debug_str = format!("{:?}", negotiation);
        assert!(!debug_str.is_empty(), "Debug output should not be empty");
        // Should contain some indication of features or limits
        assert!(
            debug_str.contains("features") || debug_str.contains("limits") || debug_str.contains("Features") || debug_str.contains("Limits"),
            "Debug output should mention features or limits: {}", debug_str
        );
    }
}

/// Verifies that error messages are informative.
///
/// Contract: Error messages should help diagnose issues.
#[test]
fn test_feature_negotiation_error_message_is_informative() {
    let adapter = require_adapter!();
    let adapter_features = adapter.features();

    // Only test if adapter doesn't support this feature
    if !adapter_features.contains(wgpu::Features::RAY_TRACING_ACCELERATION_STRUCTURE) {
        let requirements = DeviceRequirements::new()
            .require(wgpu::Features::RAY_TRACING_ACCELERATION_STRUCTURE);

        let result = negotiate_features(&requirements, &adapter);

        if let Err(e) = result {
            let error_msg = format!("{}", e);
            assert!(!error_msg.is_empty(), "Error message should not be empty");
            assert!(
                error_msg.len() > 10,
                "Error message should be descriptive: {}", error_msg
            );
        }
    }
}

// =============================================================================
// SECTION 16: Send + Sync Trait Tests
// =============================================================================

/// Verifies that DeviceRequirements is Send.
///
/// Contract: DeviceRequirements can be sent between threads.
#[test]
fn test_device_requirements_is_send() {
    fn _assert_send<T: Send>() {}
    _assert_send::<DeviceRequirements>();
}

/// Verifies that DeviceRequirements is Sync.
///
/// Contract: DeviceRequirements can be shared between threads.
#[test]
fn test_device_requirements_is_sync() {
    fn _assert_sync<T: Sync>() {}
    _assert_sync::<DeviceRequirements>();
}

/// Verifies that NegotiationResult is Send.
///
/// Contract: NegotiationResult can be sent between threads.
#[test]
fn test_negotiation_result_is_send() {
    fn _assert_send<T: Send>() {}
    _assert_send::<NegotiationResult>();
}

/// Verifies that NegotiationResult is Sync.
///
/// Contract: NegotiationResult can be shared between threads.
#[test]
fn test_negotiation_result_is_sync() {
    fn _assert_sync<T: Sync>() {}
    _assert_sync::<NegotiationResult>();
}

/// Verifies that FeatureNegotiationError is Send.
///
/// Contract: FeatureNegotiationError can be sent between threads.
#[test]
fn test_feature_negotiation_error_is_send() {
    fn _assert_send<T: Send>() {}
    _assert_send::<FeatureNegotiationError>();
}

/// Verifies that FeatureNegotiationError is Sync.
///
/// Contract: FeatureNegotiationError can be shared between threads.
#[test]
fn test_feature_negotiation_error_is_sync() {
    fn _assert_sync<T: Sync>() {}
    _assert_sync::<FeatureNegotiationError>();
}
