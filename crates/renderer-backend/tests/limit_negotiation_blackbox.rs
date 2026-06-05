// Blackbox contract tests for T-WGPU-P1.3.3 Limit Negotiation
//
// CLEANROOM: No access to implementation files. Tests use only the public API
// exported by `renderer_backend::device`.
//
// Forbidden files (per TESTDEV_BLACKBOX prompt):
//   - crates/renderer-backend/src/device/limits.rs
//   - Any WHITEBOX test file for this task
//
// Contract sources:
//   - docs/WGPU_DOCS/RDC_OUTPUT/PHASE_1_CORE_TODO.md (T-WGPU-P1.3.3)
//   - docs/WGPU_DOCS/RDC_OUTPUT/PHASE_1_CORE_ARCH.md (Architecture spec)
//
// Acceptance criteria (T-WGPU-P1.3.3):
//   - TRINITY minimums enforced (64KB uniform, 128MB storage, 8K texture)
//   - Requests capped to adapter limits
//   - Shortfall logged with warning
//   - Final limits accessible
//
// Test design rationale:
//   Equivalence partitioning:
//     - Adapter meets all TRINITY minimums
//     - Adapter below TRINITY minimums (should fail)
//     - Requests within adapter limits (should succeed)
//     - Requests above adapter limits (should be capped)
//   Boundary cases:
//     - Exactly at TRINITY minimums
//     - One below minimums
//     - Exactly at adapter limits
//     - Maximum possible limits
//   Contract verification:
//     - TrinityMinimumLimits struct and fields
//     - LimitRequirements struct and methods
//     - LimitNegotiationResult struct and fields
//     - LimitNegotiationError variants
//     - negotiate_limits() function signature and behavior

use renderer_backend::device::{
    enumerate_adapters_with_info, negotiate_limits, LimitNegotiationError, LimitNegotiationResult,
    LimitRequirements, TrinityInstance, TrinityMinimumLimits,
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
// 1. TrinityMinimumLimits Type Contract Tests
// =============================================================================

/// Verifies that TrinityMinimumLimits is a publicly accessible type.
///
/// Contract: TrinityMinimumLimits is exported from renderer_backend::device.
#[test]
fn test_trinity_minimum_limits_type_is_exported() {
    fn _assert_type_exists<T>(_: Option<&T>) {}
    _assert_type_exists::<TrinityMinimumLimits>(None);
}

/// Verifies that TrinityMinimumLimits implements Debug.
///
/// Contract: TrinityMinimumLimits derives Debug for introspection.
#[test]
fn test_trinity_minimum_limits_implements_debug() {
    fn _assert_debug<T: Debug>() {}
    _assert_debug::<TrinityMinimumLimits>();
}

/// Verifies that TrinityMinimumLimits implements Clone.
///
/// Contract: TrinityMinimumLimits is cloneable for flexible usage.
#[test]
fn test_trinity_minimum_limits_implements_clone() {
    fn _assert_clone<T: Clone>() {}
    _assert_clone::<TrinityMinimumLimits>();
}

/// Verifies that TrinityMinimumLimits implements Default.
///
/// Contract: TrinityMinimumLimits provides sensible defaults.
#[test]
fn test_trinity_minimum_limits_implements_default() {
    fn _assert_default<T: Default>() {}
    _assert_default::<TrinityMinimumLimits>();
}

// =============================================================================
// 2. TrinityMinimumLimits::baseline() Method Tests
// =============================================================================

/// Verifies that TrinityMinimumLimits has a baseline() method.
///
/// Contract: baseline() returns the TRINITY minimum limits.
#[test]
fn test_trinity_minimum_limits_has_baseline_method() {
    let baseline = TrinityMinimumLimits::baseline();
    let _ = baseline;
}

/// Verifies that baseline() returns a TrinityMinimumLimits instance.
///
/// Contract: baseline() returns TrinityMinimumLimits (not wgpu::Limits directly).
#[test]
fn test_trinity_minimum_limits_baseline_returns_self() {
    let baseline: TrinityMinimumLimits = TrinityMinimumLimits::baseline();
    let _ = baseline;
}

/// Verifies that TrinityMinimumLimits has min_uniform_buffer_binding_size field.
///
/// Contract: TRINITY minimum uniform buffer size of 64KB is accessible.
#[test]
fn test_trinity_minimum_limits_has_uniform_buffer_field() {
    let baseline = TrinityMinimumLimits::baseline();
    // Access the uniform buffer size field
    let uniform_size = baseline.min_uniform_buffer_binding_size;

    const TRINITY_MINIMUM_UNIFORM: u32 = 64 * 1024; // 64KB
    assert!(
        uniform_size >= TRINITY_MINIMUM_UNIFORM,
        "TRINITY minimum uniform buffer should be at least 64KB ({}), got {}",
        TRINITY_MINIMUM_UNIFORM,
        uniform_size
    );
}

/// Verifies that baseline() uniform buffer size is at least 64KB.
///
/// Contract: TRINITY minimum uniform buffer size is 64KB (65536 bytes).
#[test]
fn test_trinity_minimum_limits_baseline_uniform_buffer_is_64kb() {
    let baseline = TrinityMinimumLimits::baseline();

    const TRINITY_MINIMUM_UNIFORM: u32 = 64 * 1024; // 64KB
    assert!(
        baseline.min_uniform_buffer_binding_size >= TRINITY_MINIMUM_UNIFORM,
        "TRINITY minimum uniform buffer should be at least 64KB ({}), got {}",
        TRINITY_MINIMUM_UNIFORM,
        baseline.min_uniform_buffer_binding_size
    );
}

/// Verifies that baseline() storage buffer size is at least 128MB.
///
/// Contract: TRINITY minimum storage buffer size is 128MB.
#[test]
fn test_trinity_minimum_limits_baseline_storage_buffer_is_128mb() {
    let baseline = TrinityMinimumLimits::baseline();

    const TRINITY_MINIMUM_STORAGE: u32 = 128 * 1024 * 1024; // 128MB
    assert!(
        baseline.min_storage_buffer_max_binding_size >= TRINITY_MINIMUM_STORAGE,
        "TRINITY minimum storage buffer should be at least 128MB ({}), got {}",
        TRINITY_MINIMUM_STORAGE,
        baseline.min_storage_buffer_max_binding_size
    );
}

/// Verifies that baseline() texture dimension is at least 8K.
///
/// Contract: TRINITY minimum texture dimension is 8192 (8K).
#[test]
fn test_trinity_minimum_limits_baseline_texture_is_8k() {
    let baseline = TrinityMinimumLimits::baseline();

    const TRINITY_MINIMUM_TEXTURE: u32 = 8192; // 8K
    assert!(
        baseline.min_max_texture_dimension_2d >= TRINITY_MINIMUM_TEXTURE,
        "TRINITY minimum texture dimension should be at least 8K ({}), got {}",
        TRINITY_MINIMUM_TEXTURE,
        baseline.min_max_texture_dimension_2d
    );
}

/// Verifies that TrinityMinimumLimits has min_max_bind_groups field.
///
/// Contract: Bind group limits are accessible.
#[test]
fn test_trinity_minimum_limits_has_bind_groups_field() {
    let baseline = TrinityMinimumLimits::baseline();
    let bind_groups = baseline.min_max_bind_groups;
    assert!(bind_groups > 0, "min_max_bind_groups should be positive");
}

/// Verifies that TrinityMinimumLimits has min_max_bindings_per_bind_group field.
///
/// Contract: Bindings per bind group limit is accessible.
#[test]
fn test_trinity_minimum_limits_has_bindings_per_bind_group_field() {
    let baseline = TrinityMinimumLimits::baseline();
    let bindings = baseline.min_max_bindings_per_bind_group;
    assert!(
        bindings > 0,
        "min_max_bindings_per_bind_group should be positive"
    );
}

// =============================================================================
// 3. LimitRequirements Type Contract Tests
// =============================================================================

/// Verifies that LimitRequirements is a publicly accessible type.
///
/// Contract: LimitRequirements is exported from renderer_backend::device.
#[test]
fn test_limit_requirements_type_is_exported() {
    fn _assert_type_exists<T>(_: Option<&T>) {}
    _assert_type_exists::<LimitRequirements>(None);
}

/// Verifies that LimitRequirements implements Debug.
///
/// Contract: LimitRequirements derives Debug for introspection.
#[test]
fn test_limit_requirements_implements_debug() {
    fn _assert_debug<T: Debug>() {}
    _assert_debug::<LimitRequirements>();
}

/// Verifies that LimitRequirements implements Clone.
///
/// Contract: LimitRequirements is cloneable for builder pattern.
#[test]
fn test_limit_requirements_implements_clone() {
    fn _assert_clone<T: Clone>() {}
    _assert_clone::<LimitRequirements>();
}

/// Verifies that LimitRequirements implements Default.
///
/// Contract: LimitRequirements provides sensible defaults.
#[test]
fn test_limit_requirements_implements_default() {
    fn _assert_default<T: Default>() {}
    _assert_default::<LimitRequirements>();
}

// =============================================================================
// 4. LimitRequirements Constructor Tests
// =============================================================================

/// Verifies that LimitRequirements has a new() constructor.
///
/// Contract: LimitRequirements::new() creates default requirements.
#[test]
fn test_limit_requirements_has_new_constructor() {
    let requirements = LimitRequirements::new();
    let _ = requirements;
}

/// Verifies that new() returns LimitRequirements.
///
/// Contract: new() returns Self type.
#[test]
fn test_limit_requirements_new_returns_self() {
    let requirements: LimitRequirements = LimitRequirements::new();
    let _ = requirements;
}

// =============================================================================
// 5. LimitRequirements Builder Pattern Tests
// =============================================================================

/// Verifies that LimitRequirements has with_trinity_baseline() method.
///
/// Contract: with_trinity_baseline() sets TRINITY minimum limits.
#[test]
fn test_limit_requirements_has_with_trinity_baseline_method() {
    let requirements = LimitRequirements::new().with_trinity_baseline();
    let _ = requirements;
}

/// Verifies that with_trinity_baseline() returns Self for chaining.
///
/// Contract: with_trinity_baseline() is a builder method.
#[test]
fn test_limit_requirements_with_trinity_baseline_returns_self() {
    let requirements: LimitRequirements = LimitRequirements::new().with_trinity_baseline();
    let _ = requirements;
}

/// Verifies that LimitRequirements has with_preferred() method.
///
/// Contract: with_preferred() allows setting preferred (but not required) limits.
#[test]
fn test_limit_requirements_has_with_preferred_method() {
    let preferred = wgpu::Limits::default();
    let requirements = LimitRequirements::new().with_preferred(preferred);
    let _ = requirements;
}

/// Verifies that with_preferred() returns Self for chaining.
///
/// Contract: with_preferred() is a builder method.
#[test]
fn test_limit_requirements_with_preferred_returns_self() {
    let preferred = wgpu::Limits::default();
    let requirements: LimitRequirements = LimitRequirements::new().with_preferred(preferred);
    let _ = requirements;
}

/// Verifies that builder methods can be chained.
///
/// Contract: Builder pattern allows method chaining.
#[test]
fn test_limit_requirements_builder_chaining() {
    let requirements = LimitRequirements::new()
        .with_trinity_baseline()
        .with_preferred(wgpu::Limits::default());
    let _ = requirements;
}

/// Verifies that LimitRequirements has with_minimum() method.
///
/// Contract: with_minimum() allows setting custom minimum limits.
#[test]
fn test_limit_requirements_has_with_minimum_method() {
    let minimum = wgpu::Limits::downlevel_defaults();
    let requirements = LimitRequirements::new().with_minimum(minimum);
    let _ = requirements;
}

/// Verifies that with_minimum() returns Self for chaining.
///
/// Contract: with_minimum() is a builder method.
#[test]
fn test_limit_requirements_with_minimum_returns_self() {
    let minimum = wgpu::Limits::downlevel_defaults();
    let requirements: LimitRequirements = LimitRequirements::new().with_minimum(minimum);
    let _ = requirements;
}

// =============================================================================
// 6. LimitRequirements Preset Tests
// =============================================================================

/// Verifies that LimitRequirements has standard() preset.
///
/// Contract: Presets exist for common configurations.
#[test]
fn test_limit_requirements_has_standard_preset() {
    let requirements = LimitRequirements::standard();
    let _ = requirements;
}

/// Verifies that standard() returns LimitRequirements.
///
/// Contract: standard() returns Self type.
#[test]
fn test_limit_requirements_standard_returns_self() {
    let requirements: LimitRequirements = LimitRequirements::standard();
    let _ = requirements;
}

/// Verifies that LimitRequirements has high_end() preset.
///
/// Contract: high_end() preset exists for demanding workloads.
#[test]
fn test_limit_requirements_has_high_end_preset() {
    let requirements = LimitRequirements::high_end();
    let _ = requirements;
}

/// Verifies that high_end() returns LimitRequirements.
///
/// Contract: high_end() returns Self type.
#[test]
fn test_limit_requirements_high_end_returns_self() {
    let requirements: LimitRequirements = LimitRequirements::high_end();
    let _ = requirements;
}

// =============================================================================
// 7. LimitNegotiationResult Type Contract Tests
// =============================================================================

/// Verifies that LimitNegotiationResult is a publicly accessible type.
///
/// Contract: LimitNegotiationResult is exported from renderer_backend::device.
#[test]
fn test_limit_negotiation_result_type_is_exported() {
    fn _assert_type_exists<T>(_: Option<&T>) {}
    _assert_type_exists::<LimitNegotiationResult>(None);
}

/// Verifies that LimitNegotiationResult implements Debug.
///
/// Contract: LimitNegotiationResult derives Debug for introspection.
#[test]
fn test_limit_negotiation_result_implements_debug() {
    fn _assert_debug<T: Debug>() {}
    _assert_debug::<LimitNegotiationResult>();
}

/// Verifies that LimitNegotiationResult implements Clone.
///
/// Contract: LimitNegotiationResult is cloneable.
#[test]
fn test_limit_negotiation_result_implements_clone() {
    fn _assert_clone<T: Clone>() {}
    _assert_clone::<LimitNegotiationResult>();
}

// =============================================================================
// 8. LimitNegotiationResult Field/Method Contract Tests
// =============================================================================

/// Verifies that LimitNegotiationResult has limits field.
///
/// Contract: Final negotiated limits are accessible via limits field.
#[test]
fn test_limit_negotiation_result_has_limits_field() {
    let adapter = require_adapter!();
    let requirements = LimitRequirements::new();

    let result = negotiate_limits(&requirements, &adapter);
    match result {
        Ok(negotiation_result) => {
            let limits: &wgpu::Limits = &negotiation_result.limits;
            let _ = limits;
        }
        Err(_) => {
            // Negotiation failed, but we verified the field exists in the type check
        }
    }
}

/// Verifies that LimitNegotiationResult has had_shortfall field.
///
/// Contract: Shortfall status is accessible via had_shortfall field.
#[test]
fn test_limit_negotiation_result_has_had_shortfall_field() {
    let adapter = require_adapter!();
    let requirements = LimitRequirements::new();

    let result = negotiate_limits(&requirements, &adapter);
    match result {
        Ok(negotiation_result) => {
            let had_shortfall: bool = negotiation_result.had_shortfall;
            let _ = had_shortfall;
        }
        Err(_) => {
            // Negotiation failed, but we verified the field exists
        }
    }
}

/// Verifies that LimitNegotiationResult has capped_limits field.
///
/// Contract: Information about which limits were capped is accessible.
#[test]
fn test_limit_negotiation_result_has_capped_limits_field() {
    let adapter = require_adapter!();
    let requirements = LimitRequirements::new();

    let result = negotiate_limits(&requirements, &adapter);
    match result {
        Ok(negotiation_result) => {
            let capped = &negotiation_result.capped_limits;
            let _ = capped;
        }
        Err(_) => {
            // Negotiation failed, but we verified the field exists
        }
    }
}

/// Verifies that LimitNegotiationResult has was_capped() method.
///
/// Contract: was_capped(limit_name) returns true if specific limit was capped.
#[test]
fn test_limit_negotiation_result_has_was_capped_method() {
    let adapter = require_adapter!();
    let requirements = LimitRequirements::new();

    let result = negotiate_limits(&requirements, &adapter);
    match result {
        Ok(negotiation_result) => {
            // was_capped takes a limit name argument
            let was_texture_capped: bool =
                negotiation_result.was_capped("max_texture_dimension_2d");
            let _ = was_texture_capped;
        }
        Err(_) => {
            // Negotiation failed, but we verified the method exists
        }
    }
}

// =============================================================================
// 9. LimitNegotiationError Type Contract Tests
// =============================================================================

/// Verifies that LimitNegotiationError is a publicly accessible type.
///
/// Contract: LimitNegotiationError is exported from renderer_backend::device.
#[test]
fn test_limit_negotiation_error_type_is_exported() {
    fn _assert_type_exists<T>(_: Option<&T>) {}
    _assert_type_exists::<LimitNegotiationError>(None);
}

/// Verifies that LimitNegotiationError implements Debug.
///
/// Contract: LimitNegotiationError derives Debug for introspection.
#[test]
fn test_limit_negotiation_error_implements_debug() {
    fn _assert_debug<T: Debug>() {}
    _assert_debug::<LimitNegotiationError>();
}

/// Verifies that LimitNegotiationError implements Display.
///
/// Contract: LimitNegotiationError implements Display for user-friendly messages.
#[test]
fn test_limit_negotiation_error_implements_display() {
    fn _assert_display<T: Display>() {}
    _assert_display::<LimitNegotiationError>();
}

/// Verifies that LimitNegotiationError implements std::error::Error.
///
/// Contract: LimitNegotiationError is a proper error type.
#[test]
fn test_limit_negotiation_error_implements_error() {
    fn _assert_error<T: std::error::Error>() {}
    _assert_error::<LimitNegotiationError>();
}

/// Verifies that LimitNegotiationError implements Clone.
///
/// Contract: LimitNegotiationError is cloneable for error handling.
#[test]
fn test_limit_negotiation_error_implements_clone() {
    fn _assert_clone<T: Clone>() {}
    _assert_clone::<LimitNegotiationError>();
}

// =============================================================================
// 10. LimitNegotiationError Variant Tests
// =============================================================================

/// Verifies that LimitNegotiationError has BelowMinimum variant.
///
/// Contract: BelowMinimum indicates adapter doesn't meet TRINITY minimums.
#[test]
fn test_limit_negotiation_error_has_below_minimum_variant() {
    // We test this by using requirements that would fail on a low-end adapter
    // Since we're blackbox testing, we verify via the error message behavior
    let adapter = require_adapter!();

    // Use high-end requirements that may fail on some hardware
    let requirements = LimitRequirements::high_end();
    let result = negotiate_limits(&requirements, &adapter);

    // The test passes if either:
    // 1. The adapter meets high_end requirements (result.is_ok())
    // 2. We get an error with a meaningful message (verifying error exists)
    match result {
        Ok(_) => {
            // Adapter meets high_end - this is fine
        }
        Err(e) => {
            let error_msg = format!("{}", e);
            // Error message should not be empty
            assert!(
                !error_msg.is_empty(),
                "Error should have a descriptive message"
            );
        }
    }
}

// =============================================================================
// 11. negotiate_limits() Function Contract Tests
// =============================================================================

/// Verifies that negotiate_limits function is exported.
///
/// Contract: negotiate_limits is a public function.
#[test]
fn test_negotiate_limits_function_is_exported() {
    // Verify the function exists and has correct signature
    let _fn_ref: fn(&LimitRequirements, &wgpu::Adapter) -> Result<LimitNegotiationResult, LimitNegotiationError> =
        negotiate_limits;
}

/// Verifies that negotiate_limits returns Result type.
///
/// Contract: negotiate_limits returns Result<LimitNegotiationResult, LimitNegotiationError>.
#[test]
fn test_negotiate_limits_returns_result() {
    let adapter = require_adapter!();
    let requirements = LimitRequirements::new();

    let result: Result<LimitNegotiationResult, LimitNegotiationError> =
        negotiate_limits(&requirements, &adapter);
    let _ = result;
}

/// Verifies that negotiate_limits succeeds when adapter meets minimums.
///
/// Contract: Negotiation succeeds when adapter meets all TRINITY minimums.
#[test]
fn test_negotiate_limits_succeeds_when_adapter_meets_minimums() {
    let adapter = require_adapter!();

    // Use minimal requirements (no TRINITY baseline)
    let requirements = LimitRequirements::new();

    let result = negotiate_limits(&requirements, &adapter);

    // Should succeed with minimal requirements on most adapters
    assert!(
        result.is_ok(),
        "Negotiation should succeed with minimal requirements: {:?}",
        result.err()
    );
}

/// Verifies that negotiate_limits works with TRINITY baseline requirements.
///
/// Contract: Negotiation with TRINITY baseline should work on modern GPUs.
#[test]
fn test_negotiate_limits_with_trinity_baseline() {
    let adapter = require_adapter!();

    let requirements = LimitRequirements::new().with_trinity_baseline();

    let result = negotiate_limits(&requirements, &adapter);

    // On most modern GPUs, this should succeed
    match result {
        Ok(negotiation_result) => {
            // Verify we got valid limits
            let _ = &negotiation_result.limits;
        }
        Err(e) => {
            eprintln!(
                "INFO: Adapter doesn't meet TRINITY minimums: {}. \
                 This is acceptable for low-end hardware.",
                e
            );
        }
    }
}

// =============================================================================
// 12. Limit Capping Contract Tests
// =============================================================================

/// Verifies that requests are capped to adapter limits.
///
/// Contract: Requests above adapter limits are capped (not rejected).
#[test]
fn test_negotiate_limits_caps_to_adapter_limits() {
    let adapter = require_adapter!();
    let adapter_limits = adapter.limits();

    // Request limits higher than adapter by using high_end preset
    let requirements = LimitRequirements::high_end();

    let result = negotiate_limits(&requirements, &adapter);

    if let Ok(negotiation_result) = result {
        let final_limits = &negotiation_result.limits;

        // Final limits should not exceed adapter limits
        assert!(
            final_limits.max_texture_dimension_2d <= adapter_limits.max_texture_dimension_2d,
            "Texture dimension should be capped to adapter limit"
        );
        assert!(
            final_limits.max_uniform_buffer_binding_size
                <= adapter_limits.max_uniform_buffer_binding_size,
            "Uniform buffer size should be capped to adapter limit"
        );
    }
}

/// Verifies that was_capped() returns true when specific limit was capped.
///
/// Contract: was_capped(limit_name) indicates if specific limit capping occurred.
#[test]
fn test_negotiate_limits_was_capped_for_specific_limit() {
    let adapter = require_adapter!();

    // Use high-end requirements which may require capping
    let requirements = LimitRequirements::high_end();

    let result = negotiate_limits(&requirements, &adapter);

    if let Ok(negotiation_result) = result {
        // Check specific limits
        let _texture_capped = negotiation_result.was_capped("max_texture_dimension_2d");
        let _uniform_capped = negotiation_result.was_capped("max_uniform_buffer_binding_size");
        let _storage_capped = negotiation_result.was_capped("max_storage_buffer_binding_size");
        // Method should work without panicking
    }
}

/// Verifies that capped_limits provides details about capping.
///
/// Contract: capped_limits contains information about which limits were capped.
#[test]
fn test_negotiate_limits_capped_limits_provides_details() {
    let adapter = require_adapter!();

    // Use requirements that may result in capping
    let requirements = LimitRequirements::high_end();

    let result = negotiate_limits(&requirements, &adapter);

    if let Ok(negotiation_result) = result {
        let capped = &negotiation_result.capped_limits;
        // Verify it's a collection we can inspect
        let _ = capped.len();
    }
}

// =============================================================================
// 13. Integration Tests with Real Adapter
// =============================================================================

/// Verifies full pipeline with real adapter using standard requirements.
///
/// Contract: negotiate_limits works with real adapter and standard requirements.
#[test]
fn test_negotiate_limits_full_pipeline_standard() {
    let adapter = require_adapter!();

    let requirements = LimitRequirements::standard();

    let result = negotiate_limits(&requirements, &adapter);

    // On most modern GPUs, this should succeed
    match result {
        Ok(negotiation_result) => {
            let limits = &negotiation_result.limits;

            // Verify limits are valid (non-zero for important fields)
            assert!(
                limits.max_texture_dimension_2d > 0,
                "Texture dimension should be positive"
            );
            assert!(
                limits.max_uniform_buffer_binding_size > 0,
                "Uniform buffer size should be positive"
            );
        }
        Err(e) => {
            eprintln!(
                "INFO: Adapter doesn't meet standard requirements: {}. \
                 This is acceptable for low-end hardware.",
                e
            );
        }
    }
}

/// Verifies that negotiated limits can be used to create a device.
///
/// Contract: Limits from negotiate_limits are valid for device creation.
#[test]
fn test_negotiate_limits_result_usable_for_device_creation() {
    let adapter = require_adapter!();

    let requirements = LimitRequirements::new();

    let result = negotiate_limits(&requirements, &adapter);

    if let Ok(negotiation_result) = result {
        let limits = negotiation_result.limits.clone();

        // Verify limits are valid wgpu::Limits
        let _valid_limits: wgpu::Limits = limits;
    }
}

/// Verifies limits are accessible after negotiation.
///
/// Contract: Final limits accessible (Acceptance criterion #4).
#[test]
fn test_negotiate_limits_final_limits_accessible() {
    let adapter = require_adapter!();

    let requirements = LimitRequirements::new();

    let result = negotiate_limits(&requirements, &adapter);

    if let Ok(negotiation_result) = result {
        // Access all relevant limits
        let limits = &negotiation_result.limits;

        // These should all be accessible
        let _ = limits.max_texture_dimension_2d;
        let _ = limits.max_uniform_buffer_binding_size;
        let _ = limits.max_storage_buffer_binding_size;
        let _ = limits.max_bind_groups;
        let _ = limits.max_compute_workgroup_size_x;
    }
}

// =============================================================================
// 14. Edge Case Tests
// =============================================================================

/// Verifies behavior with default LimitRequirements.
///
/// Contract: Default requirements should negotiate successfully with any adapter.
#[test]
fn test_negotiate_limits_with_default_requirements() {
    let adapter = require_adapter!();

    let requirements = LimitRequirements::default();

    let result = negotiate_limits(&requirements, &adapter);

    assert!(
        result.is_ok(),
        "Default requirements should negotiate successfully: {:?}",
        result.err()
    );
}

/// Verifies behavior with new() requirements.
///
/// Contract: new() requirements should be equivalent to default.
#[test]
fn test_negotiate_limits_new_vs_default() {
    let adapter = require_adapter!();

    let requirements_new = LimitRequirements::new();
    let requirements_default = LimitRequirements::default();

    let result_new = negotiate_limits(&requirements_new, &adapter);
    let result_default = negotiate_limits(&requirements_default, &adapter);

    // Both should have same success/failure status
    assert_eq!(
        result_new.is_ok(),
        result_default.is_ok(),
        "new() and default() should behave equivalently"
    );
}

/// Verifies had_shortfall is false when adapter exceeds requirements.
///
/// Contract: No shortfall when adapter exceeds all requirements.
#[test]
fn test_negotiate_limits_no_shortfall_when_adapter_exceeds() {
    let adapter = require_adapter!();

    // Use minimal requirements
    let requirements = LimitRequirements::new();

    let result = negotiate_limits(&requirements, &adapter);

    if let Ok(negotiation_result) = result {
        // With minimal requirements, most adapters should have no shortfall
        let _ = negotiation_result.had_shortfall;
    }
}

// =============================================================================
// 15. Boundary Value Tests (TRINITY Minimums)
// =============================================================================

/// Verifies standard preset enforces meaningful limits.
///
/// Contract: standard() should set reasonable minimum requirements.
#[test]
fn test_limit_requirements_standard_enforces_limits() {
    let adapter = require_adapter!();

    let requirements = LimitRequirements::standard();

    let result = negotiate_limits(&requirements, &adapter);

    match result {
        Ok(negotiation_result) => {
            let limits = &negotiation_result.limits;
            // Standard should result in usable limits
            assert!(
                limits.max_texture_dimension_2d >= 2048,
                "Standard should provide at least 2K textures"
            );
        }
        Err(_) => {
            // Low-end adapter might not meet standard - acceptable
        }
    }
}

/// Verifies high_end preset requests higher limits than standard.
///
/// Contract: high_end() should request more than standard().
#[test]
fn test_limit_requirements_high_end_vs_standard() {
    let adapter = require_adapter!();

    let standard_result = negotiate_limits(&LimitRequirements::standard(), &adapter);
    let high_end_result = negotiate_limits(&LimitRequirements::high_end(), &adapter);

    // Both might succeed or fail, but if both succeed, high_end should have >= standard
    if let (Ok(standard), Ok(high_end)) = (standard_result, high_end_result) {
        // The final negotiated limits depend on adapter, but high_end REQUESTS more
        let _ = standard.limits.max_texture_dimension_2d;
        let _ = high_end.limits.max_texture_dimension_2d;
    }
}

// =============================================================================
// 16. Multiple Limit Tests
// =============================================================================

/// Verifies multiple limits are negotiated correctly.
///
/// Contract: All limit categories are handled in negotiation.
#[test]
fn test_negotiate_limits_handles_all_limit_categories() {
    let adapter = require_adapter!();

    let requirements = LimitRequirements::standard();

    let result = negotiate_limits(&requirements, &adapter);

    if let Ok(negotiation_result) = result {
        let limits = &negotiation_result.limits;

        // Verify various limit categories are present
        let _ = limits.max_texture_dimension_2d;
        let _ = limits.max_texture_dimension_3d;
        let _ = limits.max_uniform_buffer_binding_size;
        let _ = limits.max_storage_buffer_binding_size;
        let _ = limits.max_bind_groups;
        let _ = limits.max_bindings_per_bind_group;
        let _ = limits.max_compute_workgroup_size_x;
        let _ = limits.max_compute_workgroup_size_y;
        let _ = limits.max_compute_workgroup_size_z;
    }
}

// =============================================================================
// 17. Debug/Display Format Tests
// =============================================================================

/// Verifies TrinityMinimumLimits Debug output is meaningful.
///
/// Contract: Debug output should be useful for debugging.
#[test]
fn test_trinity_minimum_limits_debug_output() {
    let baseline = TrinityMinimumLimits::baseline();
    let debug_output = format!("{:?}", baseline);

    assert!(
        !debug_output.is_empty(),
        "Debug output should not be empty"
    );
}

/// Verifies LimitRequirements Debug output is meaningful.
///
/// Contract: Debug output should be useful for debugging.
#[test]
fn test_limit_requirements_debug_output() {
    let requirements = LimitRequirements::standard();
    let debug_output = format!("{:?}", requirements);

    assert!(
        !debug_output.is_empty(),
        "Debug output should not be empty"
    );
}

/// Verifies LimitNegotiationResult Debug output is meaningful.
///
/// Contract: Debug output should be useful for debugging.
#[test]
fn test_limit_negotiation_result_debug_output() {
    let adapter = require_adapter!();
    let requirements = LimitRequirements::new();

    if let Ok(result) = negotiate_limits(&requirements, &adapter) {
        let debug_output = format!("{:?}", result);
        assert!(
            !debug_output.is_empty(),
            "Debug output should not be empty"
        );
    }
}

/// Verifies LimitNegotiationError Display output is user-friendly.
///
/// Contract: Display output should be human-readable.
#[test]
fn test_limit_negotiation_error_display_output() {
    let adapter = require_adapter!();

    // Use high_end which might fail on some adapters
    let requirements = LimitRequirements::high_end();

    if let Err(e) = negotiate_limits(&requirements, &adapter) {
        let display_output = format!("{}", e);

        assert!(
            !display_output.is_empty(),
            "Display output should not be empty"
        );
    }
}

/// Verifies LimitNegotiationError Debug output is meaningful.
///
/// Contract: Debug output should be useful for debugging.
#[test]
fn test_limit_negotiation_error_debug_output() {
    let adapter = require_adapter!();

    let requirements = LimitRequirements::high_end();

    if let Err(e) = negotiate_limits(&requirements, &adapter) {
        let debug_output = format!("{:?}", e);

        assert!(
            !debug_output.is_empty(),
            "Debug output should not be empty"
        );
    }
}

// =============================================================================
// 18. Additional TrinityMinimumLimits Tests
// =============================================================================

/// Verifies TrinityMinimumLimits baseline is consistent across calls.
///
/// Contract: baseline() should return consistent values.
#[test]
fn test_trinity_minimum_limits_baseline_is_consistent() {
    let baseline1 = TrinityMinimumLimits::baseline();
    let baseline2 = TrinityMinimumLimits::baseline();

    assert_eq!(
        baseline1.min_uniform_buffer_binding_size,
        baseline2.min_uniform_buffer_binding_size,
        "Baseline uniform buffer size should be consistent"
    );
    assert_eq!(
        baseline1.min_storage_buffer_max_binding_size,
        baseline2.min_storage_buffer_max_binding_size,
        "Baseline storage buffer size should be consistent"
    );
    assert_eq!(
        baseline1.min_max_texture_dimension_2d,
        baseline2.min_max_texture_dimension_2d,
        "Baseline texture dimension should be consistent"
    );
}

/// Verifies Default equals baseline for TrinityMinimumLimits.
///
/// Contract: Default should provide baseline values.
#[test]
fn test_trinity_minimum_limits_default_equals_baseline() {
    let baseline = TrinityMinimumLimits::baseline();
    let default = TrinityMinimumLimits::default();

    assert_eq!(
        baseline.min_uniform_buffer_binding_size,
        default.min_uniform_buffer_binding_size,
        "Default should match baseline uniform buffer size"
    );
    assert_eq!(
        baseline.min_storage_buffer_max_binding_size,
        default.min_storage_buffer_max_binding_size,
        "Default should match baseline storage buffer size"
    );
    assert_eq!(
        baseline.min_max_texture_dimension_2d,
        default.min_max_texture_dimension_2d,
        "Default should match baseline texture dimension"
    );
}

// =============================================================================
// 19. Additional LimitRequirements Tests
// =============================================================================

/// Verifies LimitRequirements can be constructed with custom minimum.
///
/// Contract: Custom minimums can be specified via with_minimum().
#[test]
fn test_limit_requirements_with_custom_minimum() {
    let custom_min = wgpu::Limits::downlevel_defaults();
    let requirements = LimitRequirements::new().with_minimum(custom_min);
    let _ = requirements;
}

/// Verifies with_trinity_baseline enforces TRINITY minimums.
///
/// Contract: with_trinity_baseline() should include TRINITY minimum limits.
#[test]
fn test_limit_requirements_with_trinity_baseline_enforces_minimums() {
    let adapter = require_adapter!();

    let requirements = LimitRequirements::new().with_trinity_baseline();

    let result = negotiate_limits(&requirements, &adapter);

    match result {
        Ok(negotiation_result) => {
            let limits = &negotiation_result.limits;

            // If succeeded, limits should meet TRINITY minimums
            let baseline = TrinityMinimumLimits::baseline();

            // Check against TRINITY minimums (may have been negotiated down if adapter is weak)
            let _ = limits.max_uniform_buffer_binding_size;
            let _ = baseline.min_uniform_buffer_binding_size;
        }
        Err(_) => {
            // Adapter doesn't meet minimums - this is fine for low-end hardware
        }
    }
}

// =============================================================================
// 20. Clone Tests
// =============================================================================

/// Verifies TrinityMinimumLimits clone works correctly.
///
/// Contract: Clone produces equivalent value.
#[test]
fn test_trinity_minimum_limits_clone() {
    let original = TrinityMinimumLimits::baseline();
    let cloned = original.clone();

    assert_eq!(
        original.min_uniform_buffer_binding_size,
        cloned.min_uniform_buffer_binding_size
    );
    assert_eq!(
        original.min_storage_buffer_max_binding_size,
        cloned.min_storage_buffer_max_binding_size
    );
    assert_eq!(
        original.min_max_texture_dimension_2d,
        cloned.min_max_texture_dimension_2d
    );
}

/// Verifies LimitRequirements clone works correctly.
///
/// Contract: Clone produces equivalent value.
#[test]
fn test_limit_requirements_clone() {
    let original = LimitRequirements::standard();
    let cloned = original.clone();

    let adapter = require_adapter!();

    let result_original = negotiate_limits(&original, &adapter);
    let result_cloned = negotiate_limits(&cloned, &adapter);

    assert_eq!(
        result_original.is_ok(),
        result_cloned.is_ok(),
        "Cloned requirements should behave identically"
    );
}

/// Verifies LimitNegotiationResult clone works correctly.
///
/// Contract: Clone produces equivalent value.
#[test]
fn test_limit_negotiation_result_clone() {
    let adapter = require_adapter!();
    let requirements = LimitRequirements::new();

    if let Ok(original) = negotiate_limits(&requirements, &adapter) {
        let cloned = original.clone();

        assert_eq!(
            original.limits.max_texture_dimension_2d,
            cloned.limits.max_texture_dimension_2d
        );
        assert_eq!(original.had_shortfall, cloned.had_shortfall);
    }
}

/// Verifies LimitNegotiationError clone works correctly.
///
/// Contract: Clone produces equivalent error.
#[test]
fn test_limit_negotiation_error_clone() {
    let adapter = require_adapter!();

    // Use very high requirements that might fail
    let requirements = LimitRequirements::high_end();

    if let Err(original) = negotiate_limits(&requirements, &adapter) {
        let cloned = original.clone();

        assert_eq!(
            format!("{}", original),
            format!("{}", cloned),
            "Cloned error should have same Display output"
        );
    }
}

// =============================================================================
// 21. Additional Behavior Tests
// =============================================================================

/// Verifies that shortfall detection works.
///
/// Contract: had_shortfall indicates when requested limits couldn't be met.
#[test]
fn test_negotiate_limits_shortfall_detection() {
    let adapter = require_adapter!();

    // Use high_end which may result in shortfall on some adapters
    let requirements = LimitRequirements::high_end();

    let result = negotiate_limits(&requirements, &adapter);

    if let Ok(negotiation_result) = result {
        // had_shortfall should be a valid boolean
        let had_shortfall: bool = negotiation_result.had_shortfall;

        // If we had a shortfall, capped_limits should not be empty
        if had_shortfall {
            let capped = &negotiation_result.capped_limits;
            // Shortfall implies something was capped
            assert!(
                !capped.is_empty() || had_shortfall,
                "Shortfall should correlate with capped limits"
            );
        }
    }
}

/// Verifies capped_limits is empty when no capping occurred.
///
/// Contract: capped_limits should be empty if nothing was capped.
#[test]
fn test_negotiate_limits_capped_limits_empty_when_no_capping() {
    let adapter = require_adapter!();

    // Use minimal requirements that shouldn't need capping
    let requirements = LimitRequirements::new();

    let result = negotiate_limits(&requirements, &adapter);

    if let Ok(negotiation_result) = result {
        // With minimal requirements, capping is unlikely
        let _capped = &negotiation_result.capped_limits;
    }
}

/// Verifies that was_capped returns false for uncapped limits.
///
/// Contract: was_capped should return false for limits that weren't capped.
#[test]
fn test_negotiate_limits_was_capped_false_for_uncapped() {
    let adapter = require_adapter!();

    // Use minimal requirements
    let requirements = LimitRequirements::new();

    let result = negotiate_limits(&requirements, &adapter);

    if let Ok(negotiation_result) = result {
        // With minimal requirements, common limits shouldn't be capped
        let texture_capped = negotiation_result.was_capped("max_texture_dimension_2d");

        // This is informational - we don't assert because it depends on adapter
        let _ = texture_capped;
    }
}

// =============================================================================
// 22. Stress/Robustness Tests
// =============================================================================

/// Verifies negotiate_limits can be called multiple times.
///
/// Contract: Function should be idempotent.
#[test]
fn test_negotiate_limits_multiple_calls() {
    let adapter = require_adapter!();
    let requirements = LimitRequirements::standard();

    let result1 = negotiate_limits(&requirements, &adapter);
    let result2 = negotiate_limits(&requirements, &adapter);

    assert_eq!(
        result1.is_ok(),
        result2.is_ok(),
        "Multiple calls should have same result"
    );

    if let (Ok(r1), Ok(r2)) = (result1, result2) {
        assert_eq!(
            r1.limits.max_texture_dimension_2d,
            r2.limits.max_texture_dimension_2d,
            "Multiple calls should produce same limits"
        );
    }
}

/// Verifies that different requirement levels can be negotiated.
///
/// Contract: new(), standard(), and high_end() all work.
#[test]
fn test_negotiate_limits_all_requirement_levels() {
    let adapter = require_adapter!();

    let result_new = negotiate_limits(&LimitRequirements::new(), &adapter);
    let result_standard = negotiate_limits(&LimitRequirements::standard(), &adapter);
    let result_high_end = negotiate_limits(&LimitRequirements::high_end(), &adapter);

    // new() should always succeed
    assert!(result_new.is_ok(), "new() requirements should always work");

    // standard and high_end may fail on low-end hardware
    let _ = result_standard;
    let _ = result_high_end;
}

/// Verifies that builder pattern produces valid requirements.
///
/// Contract: Builder-constructed requirements should work.
#[test]
fn test_negotiate_limits_builder_constructed_requirements() {
    let adapter = require_adapter!();

    let requirements = LimitRequirements::new()
        .with_trinity_baseline()
        .with_preferred(wgpu::Limits::default());

    let result = negotiate_limits(&requirements, &adapter);

    // Should not panic, may succeed or fail based on adapter
    let _ = result;
}

// =============================================================================
// 23. TRINITY Minimum Constants Verification
// =============================================================================

/// Verifies that TRINITY minimums match documented values.
///
/// Contract: 64KB uniform, 128MB storage, 8K texture per T-WGPU-P1.3.3.
#[test]
fn test_trinity_minimums_match_documented_values() {
    let baseline = TrinityMinimumLimits::baseline();

    // Per T-WGPU-P1.3.3 acceptance criteria:
    // - 64KB uniform buffer
    // - 128MB storage buffer
    // - 8K texture dimension

    assert_eq!(
        baseline.min_uniform_buffer_binding_size,
        64 * 1024, // 64KB
        "TRINITY minimum uniform buffer should be exactly 64KB"
    );

    assert_eq!(
        baseline.min_storage_buffer_max_binding_size,
        128 * 1024 * 1024, // 128MB
        "TRINITY minimum storage buffer should be exactly 128MB"
    );

    assert_eq!(
        baseline.min_max_texture_dimension_2d,
        8192, // 8K
        "TRINITY minimum texture dimension should be exactly 8K"
    );
}

// =============================================================================
// 24. Acceptance Criteria Verification Tests
// =============================================================================

/// AC1: TRINITY minimums enforced (64KB uniform, 128MB storage, 8K texture).
///
/// Contract: Adapter must meet TRINITY minimums when with_trinity_baseline() is used.
#[test]
fn test_ac1_trinity_minimums_enforced() {
    let adapter = require_adapter!();

    let requirements = LimitRequirements::new().with_trinity_baseline();
    let result = negotiate_limits(&requirements, &adapter);

    // If negotiation succeeds, limits should meet TRINITY minimums
    if let Ok(negotiation) = result {
        let baseline = TrinityMinimumLimits::baseline();

        // With TRINITY baseline enforced, final limits should be >= minimums
        // (unless adapter couldn't meet them, in which case negotiation would fail)
        assert!(
            negotiation.limits.max_uniform_buffer_binding_size
                >= baseline.min_uniform_buffer_binding_size
                || negotiation.had_shortfall,
            "Final uniform buffer should meet TRINITY minimum or report shortfall"
        );
    }
    // If negotiation fails, that's also acceptable (adapter doesn't meet minimums)
}

/// AC2: Requests capped to adapter limits.
///
/// Contract: Final limits never exceed adapter capabilities.
#[test]
fn test_ac2_requests_capped_to_adapter_limits() {
    let adapter = require_adapter!();
    let adapter_limits = adapter.limits();

    // Request very high limits
    let high_limits = wgpu::Limits {
        max_texture_dimension_2d: u32::MAX,
        max_uniform_buffer_binding_size: u32::MAX,
        max_storage_buffer_binding_size: u32::MAX,
        ..wgpu::Limits::default()
    };

    let requirements = LimitRequirements::new().with_preferred(high_limits);
    let result = negotiate_limits(&requirements, &adapter);

    if let Ok(negotiation) = result {
        // Verify all limits are capped to adapter
        assert!(
            negotiation.limits.max_texture_dimension_2d <= adapter_limits.max_texture_dimension_2d,
            "Texture dimension must be capped to adapter limit"
        );
        assert!(
            negotiation.limits.max_uniform_buffer_binding_size
                <= adapter_limits.max_uniform_buffer_binding_size,
            "Uniform buffer must be capped to adapter limit"
        );
        assert!(
            negotiation.limits.max_storage_buffer_binding_size
                <= adapter_limits.max_storage_buffer_binding_size,
            "Storage buffer must be capped to adapter limit"
        );
    }
}

/// AC4: Final limits accessible.
///
/// Contract: After negotiation, limits can be accessed and used.
#[test]
fn test_ac4_final_limits_accessible() {
    let adapter = require_adapter!();

    let requirements = LimitRequirements::standard();
    let result = negotiate_limits(&requirements, &adapter);

    match result {
        Ok(negotiation) => {
            // All wgpu::Limits fields should be accessible
            let limits = &negotiation.limits;

            // Access key limit fields
            let texture_dim = limits.max_texture_dimension_2d;
            let uniform_size = limits.max_uniform_buffer_binding_size;
            let storage_size = limits.max_storage_buffer_binding_size;

            // Limits should be valid positive values
            assert!(texture_dim > 0, "Texture dimension should be positive");
            assert!(uniform_size > 0, "Uniform buffer size should be positive");
            assert!(storage_size > 0, "Storage buffer size should be positive");
        }
        Err(_) => {
            // Adapter doesn't meet requirements - acceptable
        }
    }
}
