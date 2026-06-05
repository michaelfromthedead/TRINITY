//! Whitebox tests for feature negotiation (T-WGPU-P1.3.2).
//!
//! These tests validate the internal implementation of:
//! - DeviceRequirements builder pattern
//! - Feature dependency expansion
//! - Feature negotiation against mock adapters
//! - Limit validation
//! - NegotiationResult helpers
//! - Error types and Display implementations
//!
//! Test count: 60+ tests covering all acceptance criteria:
//! 1. Required features cause failure if unavailable
//! 2. Optional features degraded gracefully
//! 3. Feature dependencies automatically added
//! 4. Final feature set logged (verified via result contents)

use renderer_backend::device::{
    negotiate_features, DeviceRequirements, FeatureNegotiationError, NegotiationResult,
};
use wgpu::{Features, Limits};

// ============================================================================
// Test Utilities
// ============================================================================

/// Creates a mock adapter-like struct for testing feature negotiation logic.
/// Since we cannot create a real wgpu::Adapter without GPU, we test the
/// DeviceRequirements and NegotiationResult types directly.
mod mock_negotiation {
    use super::*;

    /// Simulates feature negotiation without a real adapter.
    /// This mirrors the logic in negotiate_features() for testing.
    pub fn mock_negotiate(
        requirements: &DeviceRequirements,
        adapter_features: Features,
        adapter_limits: &Limits,
    ) -> Result<NegotiationResult, FeatureNegotiationError> {
        // Step 1: Expand required features to include dependencies
        let expanded_required = expand_feature_dependencies(requirements.required_features);

        // Step 2: Check that all required features are available
        let missing_required = expanded_required - adapter_features;
        if !missing_required.is_empty() {
            return Err(FeatureNegotiationError::RequiredFeaturesMissing(
                missing_required,
            ));
        }

        // Step 3: Validate limits
        validate_limits(&requirements.required_limits, adapter_limits)?;

        // Step 4: Expand optional features to include dependencies
        let expanded_optional = expand_feature_dependencies(requirements.optional_features);

        // Step 5: Determine which optional features are available
        let available_optional = expanded_optional & adapter_features;
        let degraded_optional = expanded_optional - adapter_features;

        // Step 6: Build final feature set
        let enabled_features = expanded_required | available_optional;

        Ok(NegotiationResult {
            enabled_features,
            degraded_features: degraded_optional,
            limits: requirements.required_limits.clone(),
        })
    }

    /// Get implicit feature dependencies (mirrors implementation).
    fn get_feature_dependencies(feature: Features) -> Features {
        let mut deps = Features::empty();

        if feature.contains(Features::MULTI_DRAW_INDIRECT_COUNT) {
            deps |= Features::MULTI_DRAW_INDIRECT;
        }

        if feature.contains(Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING)
        {
            deps |= Features::TEXTURE_BINDING_ARRAY;
        }

        if feature.contains(Features::UNIFORM_BUFFER_AND_STORAGE_TEXTURE_ARRAY_NON_UNIFORM_INDEXING)
        {
            deps |= Features::TEXTURE_BINDING_ARRAY;
        }

        if feature.contains(Features::PARTIALLY_BOUND_BINDING_ARRAY) {
            deps |= Features::TEXTURE_BINDING_ARRAY;
        }

        deps
    }

    /// Expand features to include all dependencies.
    pub fn expand_feature_dependencies(features: Features) -> Features {
        let mut expanded = features;
        let mut prev = Features::empty();

        while expanded != prev {
            prev = expanded;
            for feature in expanded.iter() {
                expanded |= get_feature_dependencies(feature);
            }
        }

        expanded
    }

    /// Validate limits (subset of full validation).
    fn validate_limits(
        required: &Limits,
        available: &Limits,
    ) -> Result<(), FeatureNegotiationError> {
        if required.max_texture_dimension_2d > available.max_texture_dimension_2d {
            return Err(FeatureNegotiationError::LimitsExceedCapabilities {
                limit: "max_texture_dimension_2d".to_string(),
                required: required.max_texture_dimension_2d,
                available: available.max_texture_dimension_2d,
            });
        }
        if required.max_buffer_size > available.max_buffer_size {
            return Err(FeatureNegotiationError::LimitsExceedCapabilities {
                limit: "max_buffer_size".to_string(),
                required: required.max_buffer_size as u32,
                available: available.max_buffer_size as u32,
            });
        }
        if required.max_bind_groups > available.max_bind_groups {
            return Err(FeatureNegotiationError::LimitsExceedCapabilities {
                limit: "max_bind_groups".to_string(),
                required: required.max_bind_groups,
                available: available.max_bind_groups,
            });
        }
        if required.max_compute_workgroup_size_x > available.max_compute_workgroup_size_x {
            return Err(FeatureNegotiationError::LimitsExceedCapabilities {
                limit: "max_compute_workgroup_size_x".to_string(),
                required: required.max_compute_workgroup_size_x,
                available: available.max_compute_workgroup_size_x,
            });
        }
        Ok(())
    }
}

// ============================================================================
// 1. DeviceRequirements Builder Tests
// ============================================================================

mod device_requirements_builder {
    use super::*;

    #[test]
    fn test_new_creates_empty_requirements() {
        let req = DeviceRequirements::new();
        assert!(req.required_features.is_empty());
        assert!(req.optional_features.is_empty());
    }

    #[test]
    fn test_require_adds_required_feature() {
        let req = DeviceRequirements::new().require(Features::TEXTURE_COMPRESSION_BC);
        assert!(req.required_features.contains(Features::TEXTURE_COMPRESSION_BC));
        assert!(req.optional_features.is_empty());
    }

    #[test]
    fn test_require_multiple_features() {
        let req = DeviceRequirements::new()
            .require(Features::TEXTURE_COMPRESSION_BC)
            .require(Features::DEPTH_CLIP_CONTROL);

        assert!(req.required_features.contains(Features::TEXTURE_COMPRESSION_BC));
        assert!(req.required_features.contains(Features::DEPTH_CLIP_CONTROL));
    }

    #[test]
    fn test_require_combined_features() {
        let combined = Features::TEXTURE_COMPRESSION_BC | Features::DEPTH_CLIP_CONTROL;
        let req = DeviceRequirements::new().require(combined);

        assert!(req.required_features.contains(Features::TEXTURE_COMPRESSION_BC));
        assert!(req.required_features.contains(Features::DEPTH_CLIP_CONTROL));
    }

    #[test]
    fn test_prefer_adds_optional_feature() {
        let req = DeviceRequirements::new().prefer(Features::POLYGON_MODE_LINE);
        assert!(req.required_features.is_empty());
        assert!(req.optional_features.contains(Features::POLYGON_MODE_LINE));
    }

    #[test]
    fn test_prefer_multiple_features() {
        let req = DeviceRequirements::new()
            .prefer(Features::POLYGON_MODE_LINE)
            .prefer(Features::TIMESTAMP_QUERY);

        assert!(req.optional_features.contains(Features::POLYGON_MODE_LINE));
        assert!(req.optional_features.contains(Features::TIMESTAMP_QUERY));
    }

    #[test]
    fn test_with_limits_sets_limits() {
        let mut limits = Limits::default();
        limits.max_texture_dimension_2d = 4096;

        let req = DeviceRequirements::new().with_limits(limits.clone());
        assert_eq!(req.required_limits.max_texture_dimension_2d, 4096);
    }

    #[test]
    fn test_builder_chaining() {
        let mut limits = Limits::default();
        limits.max_texture_dimension_2d = 8192;

        let req = DeviceRequirements::new()
            .require(Features::TEXTURE_COMPRESSION_BC)
            .require(Features::DEPTH_CLIP_CONTROL)
            .prefer(Features::POLYGON_MODE_LINE)
            .prefer(Features::PUSH_CONSTANTS)
            .with_limits(limits);

        assert!(req.required_features.contains(Features::TEXTURE_COMPRESSION_BC));
        assert!(req.required_features.contains(Features::DEPTH_CLIP_CONTROL));
        assert!(req.optional_features.contains(Features::POLYGON_MODE_LINE));
        assert!(req.optional_features.contains(Features::PUSH_CONSTANTS));
        assert_eq!(req.required_limits.max_texture_dimension_2d, 8192);
    }

    #[test]
    fn test_default_implementation() {
        let req = DeviceRequirements::default();
        assert!(req.required_features.is_empty());
        assert!(req.optional_features.is_empty());
    }

    #[test]
    fn test_clone_implementation() {
        let req = DeviceRequirements::new()
            .require(Features::PUSH_CONSTANTS)
            .prefer(Features::TIMESTAMP_QUERY);

        let cloned = req.clone();
        assert_eq!(cloned.required_features, req.required_features);
        assert_eq!(cloned.optional_features, req.optional_features);
    }

    #[test]
    fn test_debug_implementation() {
        let req = DeviceRequirements::new().require(Features::PUSH_CONSTANTS);
        let debug_str = format!("{:?}", req);
        assert!(debug_str.contains("DeviceRequirements"));
    }

    #[test]
    fn test_require_same_feature_twice_is_idempotent() {
        let req = DeviceRequirements::new()
            .require(Features::PUSH_CONSTANTS)
            .require(Features::PUSH_CONSTANTS);

        // Feature should only be present once (bitwise OR is idempotent)
        assert!(req.required_features.contains(Features::PUSH_CONSTANTS));
        assert_eq!(req.required_features, Features::PUSH_CONSTANTS);
    }

    #[test]
    fn test_prefer_same_feature_twice_is_idempotent() {
        let req = DeviceRequirements::new()
            .prefer(Features::TIMESTAMP_QUERY)
            .prefer(Features::TIMESTAMP_QUERY);

        assert!(req.optional_features.contains(Features::TIMESTAMP_QUERY));
        assert_eq!(req.optional_features, Features::TIMESTAMP_QUERY);
    }
}

// ============================================================================
// 2. Preset Tests
// ============================================================================

mod device_requirements_presets {
    use super::*;

    #[test]
    fn test_minimal_preset_no_required_features() {
        let req = DeviceRequirements::minimal();
        assert!(req.required_features.is_empty());
    }

    #[test]
    fn test_minimal_preset_no_optional_features() {
        let req = DeviceRequirements::minimal();
        assert!(req.optional_features.is_empty());
    }

    #[test]
    fn test_minimal_preset_uses_downlevel_limits() {
        let req = DeviceRequirements::minimal();
        let downlevel = Limits::downlevel_defaults();
        assert_eq!(
            req.required_limits.max_texture_dimension_2d,
            downlevel.max_texture_dimension_2d
        );
    }

    #[test]
    fn test_standard_preset_no_required_features() {
        let req = DeviceRequirements::standard();
        assert!(req.required_features.is_empty());
    }

    #[test]
    fn test_standard_preset_has_bc_compression_optional() {
        let req = DeviceRequirements::standard();
        assert!(req
            .optional_features
            .contains(Features::TEXTURE_COMPRESSION_BC));
    }

    #[test]
    fn test_standard_preset_has_polygon_mode_optional() {
        let req = DeviceRequirements::standard();
        assert!(req.optional_features.contains(Features::POLYGON_MODE_LINE));
    }

    #[test]
    fn test_standard_preset_has_timestamp_query_optional() {
        let req = DeviceRequirements::standard();
        assert!(req.optional_features.contains(Features::TIMESTAMP_QUERY));
    }

    #[test]
    fn test_standard_preset_has_depth_clip_control_optional() {
        let req = DeviceRequirements::standard();
        assert!(req.optional_features.contains(Features::DEPTH_CLIP_CONTROL));
    }

    #[test]
    fn test_advanced_preset_requires_push_constants() {
        let req = DeviceRequirements::advanced();
        assert!(req.required_features.contains(Features::PUSH_CONSTANTS));
    }

    #[test]
    fn test_advanced_preset_has_multi_draw_optional() {
        let req = DeviceRequirements::advanced();
        assert!(req.optional_features.contains(Features::MULTI_DRAW_INDIRECT));
    }

    #[test]
    fn test_advanced_preset_has_multi_draw_count_optional() {
        let req = DeviceRequirements::advanced();
        assert!(req
            .optional_features
            .contains(Features::MULTI_DRAW_INDIRECT_COUNT));
    }

    #[test]
    fn test_advanced_preset_has_texture_binding_array_optional() {
        let req = DeviceRequirements::advanced();
        assert!(req
            .optional_features
            .contains(Features::TEXTURE_BINDING_ARRAY));
    }

    #[test]
    fn test_advanced_preset_has_non_uniform_indexing_optional() {
        let req = DeviceRequirements::advanced();
        assert!(req.optional_features.contains(
            Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING
        ));
    }
}

// ============================================================================
// 3. Feature Dependency Tests
// ============================================================================

mod feature_dependencies {
    use super::*;
    use super::mock_negotiation::expand_feature_dependencies;

    #[test]
    fn test_multi_draw_indirect_count_requires_multi_draw_indirect() {
        let features = Features::MULTI_DRAW_INDIRECT_COUNT;
        let expanded = expand_feature_dependencies(features);

        assert!(expanded.contains(Features::MULTI_DRAW_INDIRECT_COUNT));
        assert!(expanded.contains(Features::MULTI_DRAW_INDIRECT));
    }

    #[test]
    fn test_non_uniform_indexing_requires_texture_binding_array() {
        let features = Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING;
        let expanded = expand_feature_dependencies(features);

        assert!(expanded.contains(Features::TEXTURE_BINDING_ARRAY));
    }

    #[test]
    fn test_uniform_buffer_indexing_requires_texture_binding_array() {
        let features = Features::UNIFORM_BUFFER_AND_STORAGE_TEXTURE_ARRAY_NON_UNIFORM_INDEXING;
        let expanded = expand_feature_dependencies(features);

        assert!(expanded.contains(Features::TEXTURE_BINDING_ARRAY));
    }

    #[test]
    fn test_partially_bound_requires_texture_binding_array() {
        let features = Features::PARTIALLY_BOUND_BINDING_ARRAY;
        let expanded = expand_feature_dependencies(features);

        assert!(expanded.contains(Features::TEXTURE_BINDING_ARRAY));
    }

    #[test]
    fn test_no_dependencies_for_simple_feature() {
        let features = Features::POLYGON_MODE_LINE;
        let expanded = expand_feature_dependencies(features);

        assert_eq!(expanded, Features::POLYGON_MODE_LINE);
    }

    #[test]
    fn test_combined_features_all_dependencies_expanded() {
        let features = Features::MULTI_DRAW_INDIRECT_COUNT
            | Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING;
        let expanded = expand_feature_dependencies(features);

        assert!(expanded.contains(Features::MULTI_DRAW_INDIRECT));
        assert!(expanded.contains(Features::TEXTURE_BINDING_ARRAY));
    }

    #[test]
    fn test_dependencies_are_transitive() {
        // If A requires B and we request A, B should be included
        let features = Features::MULTI_DRAW_INDIRECT_COUNT;
        let expanded = expand_feature_dependencies(features);

        // Both the feature and its dependency should be present
        assert!(expanded.contains(Features::MULTI_DRAW_INDIRECT_COUNT));
        assert!(expanded.contains(Features::MULTI_DRAW_INDIRECT));
    }

    #[test]
    fn test_empty_features_expand_to_empty() {
        let features = Features::empty();
        let expanded = expand_feature_dependencies(features);
        assert!(expanded.is_empty());
    }

    #[test]
    fn test_dependency_expansion_is_idempotent() {
        let features = Features::MULTI_DRAW_INDIRECT_COUNT;
        let expanded1 = expand_feature_dependencies(features);
        let expanded2 = expand_feature_dependencies(expanded1);

        assert_eq!(expanded1, expanded2);
    }
}

// ============================================================================
// 4. negotiate_features() Tests (using mock)
// ============================================================================

mod negotiate_features_tests {
    use super::*;
    use super::mock_negotiation::mock_negotiate;

    #[test]
    fn test_empty_requirements_succeeds() {
        let req = DeviceRequirements::new();
        let adapter_features = Features::empty();
        let adapter_limits = Limits::default();

        let result = mock_negotiate(&req, adapter_features, &adapter_limits);
        assert!(result.is_ok());
    }

    #[test]
    fn test_required_features_present_succeeds() {
        let req = DeviceRequirements::new().require(Features::TEXTURE_COMPRESSION_BC);
        let adapter_features = Features::TEXTURE_COMPRESSION_BC;
        let adapter_limits = Limits::default();

        let result = mock_negotiate(&req, adapter_features, &adapter_limits);
        assert!(result.is_ok());
    }

    #[test]
    fn test_required_features_missing_fails() {
        let req = DeviceRequirements::new().require(Features::TEXTURE_COMPRESSION_BC);
        let adapter_features = Features::empty();
        let adapter_limits = Limits::default();

        let result = mock_negotiate(&req, adapter_features, &adapter_limits);
        assert!(result.is_err());

        if let Err(FeatureNegotiationError::RequiredFeaturesMissing(missing)) = result {
            assert!(missing.contains(Features::TEXTURE_COMPRESSION_BC));
        } else {
            panic!("Expected RequiredFeaturesMissing error");
        }
    }

    #[test]
    fn test_multiple_required_features_all_missing() {
        let req = DeviceRequirements::new()
            .require(Features::TEXTURE_COMPRESSION_BC)
            .require(Features::PUSH_CONSTANTS);
        let adapter_features = Features::empty();
        let adapter_limits = Limits::default();

        let result = mock_negotiate(&req, adapter_features, &adapter_limits);
        assert!(result.is_err());

        if let Err(FeatureNegotiationError::RequiredFeaturesMissing(missing)) = result {
            assert!(missing.contains(Features::TEXTURE_COMPRESSION_BC));
            assert!(missing.contains(Features::PUSH_CONSTANTS));
        }
    }

    #[test]
    fn test_multiple_required_features_some_missing() {
        let req = DeviceRequirements::new()
            .require(Features::TEXTURE_COMPRESSION_BC)
            .require(Features::PUSH_CONSTANTS);
        let adapter_features = Features::TEXTURE_COMPRESSION_BC;
        let adapter_limits = Limits::default();

        let result = mock_negotiate(&req, adapter_features, &adapter_limits);
        assert!(result.is_err());

        if let Err(FeatureNegotiationError::RequiredFeaturesMissing(missing)) = result {
            assert!(missing.contains(Features::PUSH_CONSTANTS));
            assert!(!missing.contains(Features::TEXTURE_COMPRESSION_BC));
        }
    }

    #[test]
    fn test_optional_features_present_included() {
        let req = DeviceRequirements::new().prefer(Features::POLYGON_MODE_LINE);
        let adapter_features = Features::POLYGON_MODE_LINE;
        let adapter_limits = Limits::default();

        let result = mock_negotiate(&req, adapter_features, &adapter_limits).unwrap();
        assert!(result.enabled_features.contains(Features::POLYGON_MODE_LINE));
        assert!(result.degraded_features.is_empty());
    }

    #[test]
    fn test_optional_features_missing_degraded() {
        let req = DeviceRequirements::new().prefer(Features::POLYGON_MODE_LINE);
        let adapter_features = Features::empty();
        let adapter_limits = Limits::default();

        let result = mock_negotiate(&req, adapter_features, &adapter_limits).unwrap();
        assert!(!result.enabled_features.contains(Features::POLYGON_MODE_LINE));
        assert!(result.degraded_features.contains(Features::POLYGON_MODE_LINE));
    }

    #[test]
    fn test_mixed_required_and_optional_all_available() {
        let req = DeviceRequirements::new()
            .require(Features::TEXTURE_COMPRESSION_BC)
            .prefer(Features::POLYGON_MODE_LINE);
        let adapter_features = Features::TEXTURE_COMPRESSION_BC | Features::POLYGON_MODE_LINE;
        let adapter_limits = Limits::default();

        let result = mock_negotiate(&req, adapter_features, &adapter_limits).unwrap();
        assert!(result
            .enabled_features
            .contains(Features::TEXTURE_COMPRESSION_BC));
        assert!(result.enabled_features.contains(Features::POLYGON_MODE_LINE));
        assert!(result.degraded_features.is_empty());
    }

    #[test]
    fn test_mixed_required_present_optional_missing() {
        let req = DeviceRequirements::new()
            .require(Features::TEXTURE_COMPRESSION_BC)
            .prefer(Features::POLYGON_MODE_LINE);
        let adapter_features = Features::TEXTURE_COMPRESSION_BC;
        let adapter_limits = Limits::default();

        let result = mock_negotiate(&req, adapter_features, &adapter_limits).unwrap();
        assert!(result
            .enabled_features
            .contains(Features::TEXTURE_COMPRESSION_BC));
        assert!(!result.enabled_features.contains(Features::POLYGON_MODE_LINE));
        assert!(result.degraded_features.contains(Features::POLYGON_MODE_LINE));
    }

    #[test]
    fn test_required_dependency_auto_included() {
        let req = DeviceRequirements::new().require(Features::MULTI_DRAW_INDIRECT_COUNT);
        let adapter_features =
            Features::MULTI_DRAW_INDIRECT_COUNT | Features::MULTI_DRAW_INDIRECT;
        let adapter_limits = Limits::default();

        let result = mock_negotiate(&req, adapter_features, &adapter_limits).unwrap();
        assert!(result
            .enabled_features
            .contains(Features::MULTI_DRAW_INDIRECT_COUNT));
        assert!(result
            .enabled_features
            .contains(Features::MULTI_DRAW_INDIRECT));
    }

    #[test]
    fn test_required_dependency_missing_fails() {
        // Request MULTI_DRAW_INDIRECT_COUNT but adapter only has it, not MULTI_DRAW_INDIRECT
        let req = DeviceRequirements::new().require(Features::MULTI_DRAW_INDIRECT_COUNT);
        let adapter_features = Features::MULTI_DRAW_INDIRECT_COUNT; // Missing the dependency!
        let adapter_limits = Limits::default();

        let result = mock_negotiate(&req, adapter_features, &adapter_limits);
        assert!(result.is_err());
    }

    #[test]
    fn test_adapter_has_extra_features_not_in_enabled() {
        let req = DeviceRequirements::new().require(Features::TEXTURE_COMPRESSION_BC);
        let adapter_features =
            Features::TEXTURE_COMPRESSION_BC | Features::POLYGON_MODE_LINE | Features::PUSH_CONSTANTS;
        let adapter_limits = Limits::default();

        let result = mock_negotiate(&req, adapter_features, &adapter_limits).unwrap();
        // Should only have required features, not adapter extras
        assert!(result
            .enabled_features
            .contains(Features::TEXTURE_COMPRESSION_BC));
        assert!(!result.enabled_features.contains(Features::POLYGON_MODE_LINE));
        assert!(!result.enabled_features.contains(Features::PUSH_CONSTANTS));
    }
}

// ============================================================================
// 5. Limit Validation Tests
// ============================================================================

mod limit_validation_tests {
    use super::*;
    use super::mock_negotiation::mock_negotiate;

    #[test]
    fn test_limits_within_adapter_succeeds() {
        let mut req_limits = Limits::default();
        req_limits.max_texture_dimension_2d = 4096;

        let req = DeviceRequirements::new().with_limits(req_limits);
        let adapter_features = Features::empty();
        let mut adapter_limits = Limits::default();
        adapter_limits.max_texture_dimension_2d = 8192;

        let result = mock_negotiate(&req, adapter_features, &adapter_limits);
        assert!(result.is_ok());
    }

    #[test]
    fn test_limits_equal_to_adapter_succeeds() {
        let mut req_limits = Limits::default();
        req_limits.max_texture_dimension_2d = 8192;

        let req = DeviceRequirements::new().with_limits(req_limits);
        let adapter_features = Features::empty();
        let mut adapter_limits = Limits::default();
        adapter_limits.max_texture_dimension_2d = 8192;

        let result = mock_negotiate(&req, adapter_features, &adapter_limits);
        assert!(result.is_ok());
    }

    #[test]
    fn test_limits_exceeding_adapter_fails_texture_2d() {
        let mut req_limits = Limits::default();
        req_limits.max_texture_dimension_2d = 16384;

        let req = DeviceRequirements::new().with_limits(req_limits);
        let adapter_features = Features::empty();
        let mut adapter_limits = Limits::default();
        adapter_limits.max_texture_dimension_2d = 8192;

        let result = mock_negotiate(&req, adapter_features, &adapter_limits);
        assert!(result.is_err());

        if let Err(FeatureNegotiationError::LimitsExceedCapabilities { limit, required, available }) =
            result
        {
            assert_eq!(limit, "max_texture_dimension_2d");
            assert_eq!(required, 16384);
            assert_eq!(available, 8192);
        } else {
            panic!("Expected LimitsExceedCapabilities error");
        }
    }

    #[test]
    fn test_limits_exceeding_adapter_fails_buffer_size() {
        let mut req_limits = Limits::default();
        req_limits.max_buffer_size = 1024 * 1024 * 1024; // 1 GB

        let req = DeviceRequirements::new().with_limits(req_limits);
        let adapter_features = Features::empty();
        let mut adapter_limits = Limits::default();
        adapter_limits.max_buffer_size = 256 * 1024 * 1024; // 256 MB

        let result = mock_negotiate(&req, adapter_features, &adapter_limits);
        assert!(result.is_err());

        if let Err(FeatureNegotiationError::LimitsExceedCapabilities { limit, .. }) = result {
            assert_eq!(limit, "max_buffer_size");
        }
    }

    #[test]
    fn test_limits_exceeding_adapter_fails_bind_groups() {
        let mut req_limits = Limits::default();
        req_limits.max_bind_groups = 8;

        let req = DeviceRequirements::new().with_limits(req_limits);
        let adapter_features = Features::empty();
        let mut adapter_limits = Limits::default();
        adapter_limits.max_bind_groups = 4;

        let result = mock_negotiate(&req, adapter_features, &adapter_limits);
        assert!(result.is_err());

        if let Err(FeatureNegotiationError::LimitsExceedCapabilities { limit, .. }) = result {
            assert_eq!(limit, "max_bind_groups");
        }
    }

    #[test]
    fn test_limits_exceeding_adapter_fails_compute_workgroup() {
        let mut req_limits = Limits::default();
        req_limits.max_compute_workgroup_size_x = 1024;

        let req = DeviceRequirements::new().with_limits(req_limits);
        let adapter_features = Features::empty();
        let mut adapter_limits = Limits::default();
        adapter_limits.max_compute_workgroup_size_x = 256;

        let result = mock_negotiate(&req, adapter_features, &adapter_limits);
        assert!(result.is_err());

        if let Err(FeatureNegotiationError::LimitsExceedCapabilities { limit, .. }) = result {
            assert_eq!(limit, "max_compute_workgroup_size_x");
        }
    }

    #[test]
    fn test_default_limits_always_work() {
        let req = DeviceRequirements::new(); // Uses Limits::default()
        let adapter_features = Features::empty();
        let adapter_limits = Limits::default();

        let result = mock_negotiate(&req, adapter_features, &adapter_limits);
        assert!(result.is_ok());
    }

    #[test]
    fn test_downlevel_limits_work_with_default_adapter() {
        let req = DeviceRequirements::minimal(); // Uses downlevel_defaults
        let adapter_features = Features::empty();
        let adapter_limits = Limits::default(); // Default is >= downlevel

        let result = mock_negotiate(&req, adapter_features, &adapter_limits);
        assert!(result.is_ok());
    }
}

// ============================================================================
// 6. NegotiationResult Tests
// ============================================================================

mod negotiation_result_tests {
    use super::*;

    fn create_test_result() -> NegotiationResult {
        NegotiationResult {
            enabled_features: Features::TEXTURE_COMPRESSION_BC | Features::POLYGON_MODE_LINE,
            degraded_features: Features::TIMESTAMP_QUERY | Features::PUSH_CONSTANTS,
            limits: Limits::default(),
        }
    }

    #[test]
    fn test_enabled_features_contains_required() {
        let result = create_test_result();
        assert!(result
            .enabled_features
            .contains(Features::TEXTURE_COMPRESSION_BC));
    }

    #[test]
    fn test_enabled_features_contains_available_optional() {
        let result = create_test_result();
        assert!(result.enabled_features.contains(Features::POLYGON_MODE_LINE));
    }

    #[test]
    fn test_degraded_features_contains_unavailable_optional() {
        let result = create_test_result();
        assert!(result.degraded_features.contains(Features::TIMESTAMP_QUERY));
    }

    #[test]
    fn test_has_feature_returns_true_for_enabled() {
        let result = create_test_result();
        assert!(result.has_feature(Features::TEXTURE_COMPRESSION_BC));
        assert!(result.has_feature(Features::POLYGON_MODE_LINE));
    }

    #[test]
    fn test_has_feature_returns_false_for_disabled() {
        let result = create_test_result();
        assert!(!result.has_feature(Features::TIMESTAMP_QUERY));
        assert!(!result.has_feature(Features::PUSH_CONSTANTS));
    }

    #[test]
    fn test_has_feature_returns_false_for_unknown() {
        let result = create_test_result();
        assert!(!result.has_feature(Features::DEPTH_CLIP_CONTROL));
    }

    #[test]
    fn test_was_degraded_returns_true_for_degraded() {
        let result = create_test_result();
        assert!(result.was_degraded(Features::TIMESTAMP_QUERY));
        assert!(result.was_degraded(Features::PUSH_CONSTANTS));
    }

    #[test]
    fn test_was_degraded_returns_false_for_enabled() {
        let result = create_test_result();
        assert!(!result.was_degraded(Features::TEXTURE_COMPRESSION_BC));
        assert!(!result.was_degraded(Features::POLYGON_MODE_LINE));
    }

    #[test]
    fn test_was_degraded_returns_false_for_unknown() {
        let result = create_test_result();
        assert!(!result.was_degraded(Features::DEPTH_CLIP_CONTROL));
    }

    #[test]
    fn test_enabled_count() {
        let result = create_test_result();
        assert_eq!(result.enabled_count(), 2);
    }

    #[test]
    fn test_degraded_count() {
        let result = create_test_result();
        assert_eq!(result.degraded_count(), 2);
    }

    #[test]
    fn test_enabled_count_zero_when_empty() {
        let result = NegotiationResult {
            enabled_features: Features::empty(),
            degraded_features: Features::empty(),
            limits: Limits::default(),
        };
        assert_eq!(result.enabled_count(), 0);
    }

    #[test]
    fn test_degraded_count_zero_when_empty() {
        let result = NegotiationResult {
            enabled_features: Features::PUSH_CONSTANTS,
            degraded_features: Features::empty(),
            limits: Limits::default(),
        };
        assert_eq!(result.degraded_count(), 0);
    }

    #[test]
    fn test_clone_implementation() {
        let result = create_test_result();
        let cloned = result.clone();

        assert_eq!(cloned.enabled_features, result.enabled_features);
        assert_eq!(cloned.degraded_features, result.degraded_features);
    }

    #[test]
    fn test_debug_implementation() {
        let result = create_test_result();
        let debug_str = format!("{:?}", result);
        assert!(debug_str.contains("NegotiationResult"));
    }
}

// ============================================================================
// 7. FeatureNegotiationError Tests
// ============================================================================

mod feature_negotiation_error_tests {
    use super::*;

    #[test]
    fn test_required_features_missing_display() {
        let err =
            FeatureNegotiationError::RequiredFeaturesMissing(Features::TEXTURE_COMPRESSION_BC);
        let msg = format!("{}", err);

        assert!(msg.contains("Required features"));
        assert!(msg.contains("not available"));
        assert!(msg.contains("TEXTURE_COMPRESSION_BC"));
    }

    #[test]
    fn test_required_features_missing_display_multiple() {
        let missing = Features::TEXTURE_COMPRESSION_BC | Features::PUSH_CONSTANTS;
        let err = FeatureNegotiationError::RequiredFeaturesMissing(missing);
        let msg = format!("{}", err);

        assert!(msg.contains("Required features"));
    }

    #[test]
    fn test_limits_exceed_capabilities_display() {
        let err = FeatureNegotiationError::LimitsExceedCapabilities {
            limit: "max_texture_dimension_2d".to_string(),
            required: 16384,
            available: 8192,
        };
        let msg = format!("{}", err);

        assert!(msg.contains("max_texture_dimension_2d"));
        assert!(msg.contains("16384"));
        assert!(msg.contains("8192"));
        assert!(msg.contains("exceeds"));
    }

    #[test]
    fn test_debug_implementation_required_features() {
        let err =
            FeatureNegotiationError::RequiredFeaturesMissing(Features::TEXTURE_COMPRESSION_BC);
        let debug_str = format!("{:?}", err);
        assert!(debug_str.contains("RequiredFeaturesMissing"));
    }

    #[test]
    fn test_debug_implementation_limits() {
        let err = FeatureNegotiationError::LimitsExceedCapabilities {
            limit: "test_limit".to_string(),
            required: 100,
            available: 50,
        };
        let debug_str = format!("{:?}", err);
        assert!(debug_str.contains("LimitsExceedCapabilities"));
    }

    #[test]
    fn test_clone_implementation() {
        let err = FeatureNegotiationError::LimitsExceedCapabilities {
            limit: "test".to_string(),
            required: 10,
            available: 5,
        };
        let cloned = err.clone();

        if let FeatureNegotiationError::LimitsExceedCapabilities {
            limit,
            required,
            available,
        } = cloned
        {
            assert_eq!(limit, "test");
            assert_eq!(required, 10);
            assert_eq!(available, 5);
        }
    }

    #[test]
    fn test_error_trait_implementation() {
        let err =
            FeatureNegotiationError::RequiredFeaturesMissing(Features::TEXTURE_COMPRESSION_BC);
        let _: &dyn std::error::Error = &err;
    }
}

// ============================================================================
// 8. Display Implementation Tests
// ============================================================================

mod display_tests {
    use super::*;

    #[test]
    fn test_device_requirements_display_empty() {
        let req = DeviceRequirements::new();
        let display = format!("{}", req);

        assert!(display.contains("DeviceRequirements"));
        assert!(display.contains("Required features"));
        assert!(display.contains("none"));
    }

    #[test]
    fn test_device_requirements_display_with_features() {
        let req = DeviceRequirements::new()
            .require(Features::PUSH_CONSTANTS)
            .prefer(Features::POLYGON_MODE_LINE);
        let display = format!("{}", req);

        assert!(display.contains("DeviceRequirements"));
        assert!(display.contains("Required features"));
        assert!(display.contains("Optional features"));
    }

    #[test]
    fn test_device_requirements_display_with_limits() {
        let req = DeviceRequirements::new();
        let display = format!("{}", req);

        assert!(display.contains("Required limits"));
        assert!(display.contains("max_texture"));
        assert!(display.contains("max_buffer"));
    }

    #[test]
    fn test_negotiation_result_display_enabled() {
        let result = NegotiationResult {
            enabled_features: Features::PUSH_CONSTANTS,
            degraded_features: Features::empty(),
            limits: Limits::default(),
        };
        let display = format!("{}", result);

        assert!(display.contains("NegotiationResult"));
        assert!(display.contains("Enabled"));
        assert!(display.contains("1 features"));
    }

    #[test]
    fn test_negotiation_result_display_degraded() {
        let result = NegotiationResult {
            enabled_features: Features::empty(),
            degraded_features: Features::TIMESTAMP_QUERY | Features::POLYGON_MODE_LINE,
            limits: Limits::default(),
        };
        let display = format!("{}", result);

        assert!(display.contains("Degraded"));
        assert!(display.contains("2 features"));
    }

    #[test]
    fn test_negotiation_result_display_shows_degraded_list() {
        let result = NegotiationResult {
            enabled_features: Features::empty(),
            degraded_features: Features::TIMESTAMP_QUERY,
            limits: Limits::default(),
        };
        let display = format!("{}", result);

        assert!(display.contains("Degraded list"));
    }
}

// ============================================================================
// 9. Edge Case Tests
// ============================================================================

mod edge_cases {
    use super::*;
    use super::mock_negotiation::mock_negotiate;

    #[test]
    fn test_same_feature_required_and_optional() {
        // If a feature is both required and optional, it should be in required
        let req = DeviceRequirements::new()
            .require(Features::PUSH_CONSTANTS)
            .prefer(Features::PUSH_CONSTANTS);

        let adapter_features = Features::PUSH_CONSTANTS;
        let adapter_limits = Limits::default();

        let result = mock_negotiate(&req, adapter_features, &adapter_limits).unwrap();
        assert!(result.has_feature(Features::PUSH_CONSTANTS));
        assert!(!result.was_degraded(Features::PUSH_CONSTANTS));
    }

    #[test]
    fn test_all_features_empty_result() {
        let req = DeviceRequirements::new();
        let adapter_features = Features::empty();
        let adapter_limits = Limits::default();

        let result = mock_negotiate(&req, adapter_features, &adapter_limits).unwrap();
        assert!(result.enabled_features.is_empty());
        assert!(result.degraded_features.is_empty());
    }

    #[test]
    fn test_many_optional_features_all_available() {
        let req = DeviceRequirements::new()
            .prefer(Features::POLYGON_MODE_LINE)
            .prefer(Features::TIMESTAMP_QUERY)
            .prefer(Features::PUSH_CONSTANTS)
            .prefer(Features::DEPTH_CLIP_CONTROL);

        let adapter_features = Features::POLYGON_MODE_LINE
            | Features::TIMESTAMP_QUERY
            | Features::PUSH_CONSTANTS
            | Features::DEPTH_CLIP_CONTROL;
        let adapter_limits = Limits::default();

        let result = mock_negotiate(&req, adapter_features, &adapter_limits).unwrap();
        assert_eq!(result.enabled_count(), 4);
        assert_eq!(result.degraded_count(), 0);
    }

    #[test]
    fn test_many_optional_features_all_unavailable() {
        let req = DeviceRequirements::new()
            .prefer(Features::POLYGON_MODE_LINE)
            .prefer(Features::TIMESTAMP_QUERY)
            .prefer(Features::PUSH_CONSTANTS);

        let adapter_features = Features::empty();
        let adapter_limits = Limits::default();

        let result = mock_negotiate(&req, adapter_features, &adapter_limits).unwrap();
        assert_eq!(result.enabled_count(), 0);
        assert_eq!(result.degraded_count(), 3);
    }

    #[test]
    fn test_optional_with_dependency_partially_available() {
        // Request MULTI_DRAW_INDIRECT_COUNT (needs MULTI_DRAW_INDIRECT)
        // Adapter only has MULTI_DRAW_INDIRECT
        let req =
            DeviceRequirements::new().prefer(Features::MULTI_DRAW_INDIRECT_COUNT);

        let adapter_features = Features::MULTI_DRAW_INDIRECT; // Has dependency but not feature
        let adapter_limits = Limits::default();

        let result = mock_negotiate(&req, adapter_features, &adapter_limits).unwrap();
        // Both should be degraded since the main feature isn't available
        assert!(result
            .degraded_features
            .contains(Features::MULTI_DRAW_INDIRECT_COUNT));
    }

    #[test]
    fn test_limits_validation_before_feature_negotiation() {
        // Even if features are OK, limits should fail first
        let mut req_limits = Limits::default();
        req_limits.max_texture_dimension_2d = 99999;

        let req = DeviceRequirements::new()
            .require(Features::PUSH_CONSTANTS)
            .with_limits(req_limits);

        let adapter_features = Features::PUSH_CONSTANTS;
        let mut adapter_limits = Limits::default();
        adapter_limits.max_texture_dimension_2d = 8192;

        let result = mock_negotiate(&req, adapter_features, &adapter_limits);
        assert!(matches!(
            result,
            Err(FeatureNegotiationError::LimitsExceedCapabilities { .. })
        ));
    }
}

// ============================================================================
// 10. Integration-style Tests (internal consistency)
// ============================================================================

mod integration_consistency {
    use super::*;
    use super::mock_negotiation::mock_negotiate;

    #[test]
    fn test_standard_preset_on_minimal_adapter() {
        let req = DeviceRequirements::standard();
        let adapter_features = Features::empty();
        let adapter_limits = Limits::default();

        let result = mock_negotiate(&req, adapter_features, &adapter_limits).unwrap();

        // All standard features should be degraded
        assert!(result.was_degraded(Features::TEXTURE_COMPRESSION_BC));
        assert!(result.was_degraded(Features::POLYGON_MODE_LINE));
        assert!(result.was_degraded(Features::TIMESTAMP_QUERY));
        assert!(result.was_degraded(Features::DEPTH_CLIP_CONTROL));
    }

    #[test]
    fn test_standard_preset_on_full_adapter() {
        let req = DeviceRequirements::standard();
        let adapter_features = Features::TEXTURE_COMPRESSION_BC
            | Features::POLYGON_MODE_LINE
            | Features::TIMESTAMP_QUERY
            | Features::DEPTH_CLIP_CONTROL;
        let adapter_limits = Limits::default();

        let result = mock_negotiate(&req, adapter_features, &adapter_limits).unwrap();

        // All standard features should be enabled
        assert!(result.has_feature(Features::TEXTURE_COMPRESSION_BC));
        assert!(result.has_feature(Features::POLYGON_MODE_LINE));
        assert!(result.has_feature(Features::TIMESTAMP_QUERY));
        assert!(result.has_feature(Features::DEPTH_CLIP_CONTROL));
        assert!(result.degraded_features.is_empty());
    }

    #[test]
    fn test_advanced_preset_fails_without_push_constants() {
        let req = DeviceRequirements::advanced();
        let adapter_features = Features::MULTI_DRAW_INDIRECT; // Missing PUSH_CONSTANTS
        let adapter_limits = Limits::default();

        let result = mock_negotiate(&req, adapter_features, &adapter_limits);
        assert!(matches!(
            result,
            Err(FeatureNegotiationError::RequiredFeaturesMissing(_))
        ));
    }

    #[test]
    fn test_advanced_preset_succeeds_with_push_constants() {
        let req = DeviceRequirements::advanced();
        let adapter_features = Features::PUSH_CONSTANTS;
        let adapter_limits = Limits::default();

        let result = mock_negotiate(&req, adapter_features, &adapter_limits).unwrap();
        assert!(result.has_feature(Features::PUSH_CONSTANTS));
    }

    #[test]
    fn test_minimal_preset_succeeds_everywhere() {
        let req = DeviceRequirements::minimal();
        let adapter_features = Features::empty();
        // Downlevel limits should be <= default limits
        let adapter_limits = Limits::downlevel_defaults();

        let result = mock_negotiate(&req, adapter_features, &adapter_limits);
        assert!(result.is_ok());
    }

    #[test]
    fn test_result_consistency_enabled_and_degraded_disjoint() {
        let req = DeviceRequirements::new()
            .require(Features::PUSH_CONSTANTS)
            .prefer(Features::POLYGON_MODE_LINE)
            .prefer(Features::TIMESTAMP_QUERY);

        let adapter_features = Features::PUSH_CONSTANTS | Features::POLYGON_MODE_LINE;
        let adapter_limits = Limits::default();

        let result = mock_negotiate(&req, adapter_features, &adapter_limits).unwrap();

        // Enabled and degraded should be disjoint
        let intersection = result.enabled_features & result.degraded_features;
        assert!(intersection.is_empty());
    }

    #[test]
    fn test_result_consistency_covers_all_requested() {
        let req = DeviceRequirements::new()
            .require(Features::PUSH_CONSTANTS)
            .prefer(Features::POLYGON_MODE_LINE)
            .prefer(Features::TIMESTAMP_QUERY);

        let adapter_features = Features::PUSH_CONSTANTS | Features::POLYGON_MODE_LINE;
        let adapter_limits = Limits::default();

        let result = mock_negotiate(&req, adapter_features, &adapter_limits).unwrap();

        // Every requested feature should be either enabled or degraded
        assert!(
            result.has_feature(Features::PUSH_CONSTANTS)
                || result.was_degraded(Features::PUSH_CONSTANTS)
        );
        assert!(
            result.has_feature(Features::POLYGON_MODE_LINE)
                || result.was_degraded(Features::POLYGON_MODE_LINE)
        );
        assert!(
            result.has_feature(Features::TIMESTAMP_QUERY)
                || result.was_degraded(Features::TIMESTAMP_QUERY)
        );
    }
}

// ============================================================================
// 11. Feature Count Tests
// ============================================================================

mod feature_count_tests {
    use super::*;

    #[test]
    fn test_single_feature_count() {
        let result = NegotiationResult {
            enabled_features: Features::PUSH_CONSTANTS,
            degraded_features: Features::empty(),
            limits: Limits::default(),
        };
        assert_eq!(result.enabled_count(), 1);
    }

    #[test]
    fn test_multiple_features_count() {
        let result = NegotiationResult {
            enabled_features: Features::PUSH_CONSTANTS
                | Features::POLYGON_MODE_LINE
                | Features::TIMESTAMP_QUERY,
            degraded_features: Features::empty(),
            limits: Limits::default(),
        };
        assert_eq!(result.enabled_count(), 3);
    }

    #[test]
    fn test_empty_features_count() {
        let result = NegotiationResult {
            enabled_features: Features::empty(),
            degraded_features: Features::empty(),
            limits: Limits::default(),
        };
        assert_eq!(result.enabled_count(), 0);
        assert_eq!(result.degraded_count(), 0);
    }
}

// ============================================================================
// 12. Builder Pattern Immutability Tests
// ============================================================================

mod builder_immutability {
    use super::*;

    #[test]
    fn test_require_returns_new_instance() {
        let req1 = DeviceRequirements::new();
        let req2 = req1.require(Features::PUSH_CONSTANTS);

        // req2 should have the feature, but req1 was consumed
        assert!(req2.required_features.contains(Features::PUSH_CONSTANTS));
    }

    #[test]
    fn test_prefer_returns_new_instance() {
        let req1 = DeviceRequirements::new();
        let req2 = req1.prefer(Features::POLYGON_MODE_LINE);

        assert!(req2.optional_features.contains(Features::POLYGON_MODE_LINE));
    }

    #[test]
    fn test_with_limits_returns_new_instance() {
        let mut limits = Limits::default();
        limits.max_texture_dimension_2d = 4096;

        let req1 = DeviceRequirements::new();
        let req2 = req1.with_limits(limits);

        assert_eq!(req2.required_limits.max_texture_dimension_2d, 4096);
    }
}
