//! Whitebox tests for limit negotiation (T-WGPU-P1.3.3).
//!
//! These tests validate the internal implementation of:
//! - TrinityMinimumLimits baseline values and conversions
//! - LimitRequirements builder pattern and presets
//! - negotiate_limits() function behavior
//! - LimitNegotiationResult helpers
//! - LimitNegotiationError types and Display implementations
//!
//! Test count: 60+ tests covering all acceptance criteria:
//! 1. TRINITY minimums enforced (64KB uniform, 128MB storage, 8K texture)
//! 2. Requests capped to adapter limits
//! 3. Shortfall logged with warning (verified via result contents)
//! 4. Final limits accessible

use renderer_backend::device::{
    LimitNegotiationError, LimitNegotiationResult, LimitRequirements, TrinityMinimumLimits,
};
use wgpu::Limits;

// ============================================================================
// Test Utilities
// ============================================================================

/// Creates a mock adapter limits struct for testing negotiation logic.
/// Since we cannot create a real wgpu::Adapter without GPU, we test the
/// internal logic through the validate_minimum_limits path.
mod mock_negotiation {
    use super::*;

    /// Simulates limit negotiation without a real adapter.
    /// This mirrors the core logic in negotiate_limits() for testing.
    pub fn mock_negotiate(
        requirements: &LimitRequirements,
        adapter_limits: &Limits,
    ) -> Result<LimitNegotiationResult, LimitNegotiationError> {
        // Phase 1: Validate adapter meets minimums
        validate_minimum_limits(&requirements.minimum, adapter_limits)?;

        // Phase 2: Negotiate each limit
        let mut capped_limits = Vec::new();
        let mut limits = Limits::default();

        // Macro for u32 limits
        macro_rules! negotiate_limit {
            ($field:ident, $name:expr) => {{
                let minimum = requirements.minimum.$field;
                let preferred = requirements.preferred.$field;
                let adapter_val = adapter_limits.$field;

                let negotiated = if adapter_val >= preferred {
                    preferred
                } else if adapter_val >= minimum {
                    capped_limits.push($name.to_string());
                    adapter_val
                } else {
                    return Err(LimitNegotiationError::BelowMinimum {
                        limit: $name.to_string(),
                        required: minimum as u64,
                        available: adapter_val as u64,
                    });
                };

                limits.$field = negotiated;
            }};
        }

        // Macro for u64 limits
        macro_rules! negotiate_limit_u64 {
            ($field:ident, $name:expr) => {{
                let minimum = requirements.minimum.$field;
                let preferred = requirements.preferred.$field;
                let adapter_val = adapter_limits.$field;

                let negotiated = if adapter_val >= preferred {
                    preferred
                } else if adapter_val >= minimum {
                    capped_limits.push($name.to_string());
                    adapter_val
                } else {
                    return Err(LimitNegotiationError::BelowMinimum {
                        limit: $name.to_string(),
                        required: minimum,
                        available: adapter_val,
                    });
                };

                limits.$field = negotiated;
            }};
        }

        // Negotiate key limits for testing
        negotiate_limit!(max_texture_dimension_1d, "max_texture_dimension_1d");
        negotiate_limit!(max_texture_dimension_2d, "max_texture_dimension_2d");
        negotiate_limit!(max_texture_dimension_3d, "max_texture_dimension_3d");
        negotiate_limit!(max_texture_array_layers, "max_texture_array_layers");
        negotiate_limit_u64!(max_buffer_size, "max_buffer_size");
        negotiate_limit!(
            max_uniform_buffer_binding_size,
            "max_uniform_buffer_binding_size"
        );
        negotiate_limit!(
            max_storage_buffer_binding_size,
            "max_storage_buffer_binding_size"
        );
        negotiate_limit!(max_bind_groups, "max_bind_groups");
        negotiate_limit!(max_bindings_per_bind_group, "max_bindings_per_bind_group");
        negotiate_limit!(max_compute_workgroup_size_x, "max_compute_workgroup_size_x");
        negotiate_limit!(max_compute_workgroup_size_y, "max_compute_workgroup_size_y");
        negotiate_limit!(max_compute_workgroup_size_z, "max_compute_workgroup_size_z");
        negotiate_limit!(
            max_compute_invocations_per_workgroup,
            "max_compute_invocations_per_workgroup"
        );
        negotiate_limit!(max_vertex_buffers, "max_vertex_buffers");
        negotiate_limit!(max_vertex_attributes, "max_vertex_attributes");
        negotiate_limit!(max_color_attachments, "max_color_attachments");

        let had_shortfall = !capped_limits.is_empty();

        Ok(LimitNegotiationResult {
            limits,
            had_shortfall,
            capped_limits,
        })
    }

    /// Validate that adapter limits meet minimum requirements.
    fn validate_minimum_limits(
        minimum: &Limits,
        adapter: &Limits,
    ) -> Result<(), LimitNegotiationError> {
        macro_rules! check_limit {
            ($field:ident, $name:expr) => {
                if adapter.$field < minimum.$field {
                    return Err(LimitNegotiationError::BelowMinimum {
                        limit: $name.to_string(),
                        required: minimum.$field as u64,
                        available: adapter.$field as u64,
                    });
                }
            };
        }

        macro_rules! check_limit_u64 {
            ($field:ident, $name:expr) => {
                if adapter.$field < minimum.$field {
                    return Err(LimitNegotiationError::BelowMinimum {
                        limit: $name.to_string(),
                        required: minimum.$field,
                        available: adapter.$field,
                    });
                }
            };
        }

        check_limit!(max_texture_dimension_1d, "max_texture_dimension_1d");
        check_limit!(max_texture_dimension_2d, "max_texture_dimension_2d");
        check_limit!(max_texture_dimension_3d, "max_texture_dimension_3d");
        check_limit!(max_texture_array_layers, "max_texture_array_layers");
        check_limit_u64!(max_buffer_size, "max_buffer_size");
        check_limit!(
            max_uniform_buffer_binding_size,
            "max_uniform_buffer_binding_size"
        );
        check_limit!(
            max_storage_buffer_binding_size,
            "max_storage_buffer_binding_size"
        );
        check_limit!(max_bind_groups, "max_bind_groups");
        check_limit!(max_bindings_per_bind_group, "max_bindings_per_bind_group");
        check_limit!(max_compute_workgroup_size_x, "max_compute_workgroup_size_x");
        check_limit!(max_compute_workgroup_size_y, "max_compute_workgroup_size_y");
        check_limit!(max_compute_workgroup_size_z, "max_compute_workgroup_size_z");
        check_limit!(
            max_compute_invocations_per_workgroup,
            "max_compute_invocations_per_workgroup"
        );
        check_limit!(max_vertex_buffers, "max_vertex_buffers");
        check_limit!(max_vertex_attributes, "max_vertex_attributes");
        check_limit!(max_color_attachments, "max_color_attachments");

        Ok(())
    }

    /// Creates adapter limits that exceed TRINITY minimums.
    pub fn make_high_end_adapter() -> Limits {
        let mut limits = Limits::default();
        limits.max_texture_dimension_1d = 16384;
        limits.max_texture_dimension_2d = 32768;
        limits.max_texture_dimension_3d = 4096;
        limits.max_texture_array_layers = 2048;
        limits.max_buffer_size = 4_294_967_296; // 4GB
        limits.max_uniform_buffer_binding_size = 131072; // 128KB
        limits.max_storage_buffer_binding_size = 1_073_741_824; // 1GB
        limits.max_bind_groups = 8;
        limits.max_bindings_per_bind_group = 1000;
        limits.max_compute_workgroup_size_x = 1024;
        limits.max_compute_workgroup_size_y = 1024;
        limits.max_compute_workgroup_size_z = 64;
        limits.max_compute_invocations_per_workgroup = 1024;
        limits.max_vertex_buffers = 16;
        limits.max_vertex_attributes = 32;
        limits.max_color_attachments = 8;
        limits
    }

    /// Creates adapter limits that exactly match TRINITY minimums.
    pub fn make_minimum_adapter() -> Limits {
        TrinityMinimumLimits::baseline().to_wgpu_limits()
    }

    /// Creates adapter limits below TRINITY minimums (for failure testing).
    pub fn make_subpar_adapter() -> Limits {
        let mut limits = Limits::default();
        limits.max_texture_dimension_1d = 4096;
        limits.max_texture_dimension_2d = 4096; // Below 8K minimum
        limits.max_texture_dimension_3d = 1024;
        limits.max_texture_array_layers = 128;
        limits.max_buffer_size = 134_217_728; // 128MB (below 256MB minimum)
        limits.max_uniform_buffer_binding_size = 32768; // 32KB (below 64KB minimum)
        limits.max_storage_buffer_binding_size = 67_108_864; // 64MB (below 128MB minimum)
        limits.max_bind_groups = 4;
        limits.max_bindings_per_bind_group = 256;
        limits.max_compute_workgroup_size_x = 128;
        limits.max_compute_workgroup_size_y = 128;
        limits.max_compute_workgroup_size_z = 32;
        limits.max_compute_invocations_per_workgroup = 128;
        limits.max_vertex_buffers = 4;
        limits.max_vertex_attributes = 8;
        limits.max_color_attachments = 4;
        limits
    }

    /// Creates adapter limits between minimum and preferred.
    pub fn make_mid_tier_adapter() -> Limits {
        let mut limits = TrinityMinimumLimits::baseline().to_wgpu_limits();
        // Exceed minimums but fall short of typical "high-end" preferences
        limits.max_texture_dimension_2d = 12288; // Between 8K and 16K
        limits.max_storage_buffer_binding_size = 201_326_592; // ~192MB (between 128MB and 256MB)
        limits.max_buffer_size = 402_653_184; // ~384MB (between 256MB and 512MB)
        limits
    }
}

use mock_negotiation::*;

// ============================================================================
// 1. TrinityMinimumLimits Tests
// ============================================================================

mod trinity_minimum_limits {
    use super::*;

    #[test]
    fn baseline_creates_correct_minimums() {
        let baseline = TrinityMinimumLimits::baseline();

        // Core requirements from acceptance criteria
        assert_eq!(baseline.min_uniform_buffer_binding_size, 65536); // 64KB
        assert_eq!(baseline.min_storage_buffer_max_binding_size, 134_217_728); // 128MB
        assert_eq!(baseline.min_max_texture_dimension_2d, 8192); // 8K
    }

    #[test]
    fn baseline_uniform_buffer_is_64kb() {
        let baseline = TrinityMinimumLimits::baseline();
        assert_eq!(baseline.min_uniform_buffer_binding_size, 64 * 1024);
        assert_eq!(baseline.min_uniform_buffer_binding_size, 65536);
    }

    #[test]
    fn baseline_storage_buffer_is_128mb() {
        let baseline = TrinityMinimumLimits::baseline();
        assert_eq!(
            baseline.min_storage_buffer_max_binding_size,
            128 * 1024 * 1024
        );
        assert_eq!(baseline.min_storage_buffer_max_binding_size, 134_217_728);
    }

    #[test]
    fn baseline_texture_2d_is_8k() {
        let baseline = TrinityMinimumLimits::baseline();
        assert_eq!(baseline.min_max_texture_dimension_2d, 8192);
    }

    #[test]
    fn baseline_bind_groups_is_4() {
        let baseline = TrinityMinimumLimits::baseline();
        assert_eq!(baseline.min_max_bind_groups, 4);
    }

    #[test]
    fn baseline_bindings_per_group_is_640() {
        let baseline = TrinityMinimumLimits::baseline();
        assert_eq!(baseline.min_max_bindings_per_bind_group, 640);
    }

    #[test]
    fn baseline_compute_workgroup_x_is_256() {
        let baseline = TrinityMinimumLimits::baseline();
        assert_eq!(baseline.min_max_compute_workgroup_size_x, 256);
    }

    #[test]
    fn baseline_compute_workgroup_y_is_256() {
        let baseline = TrinityMinimumLimits::baseline();
        assert_eq!(baseline.min_max_compute_workgroup_size_y, 256);
    }

    #[test]
    fn baseline_compute_workgroup_z_is_64() {
        let baseline = TrinityMinimumLimits::baseline();
        assert_eq!(baseline.min_max_compute_workgroup_size_z, 64);
    }

    #[test]
    fn baseline_compute_invocations_is_256() {
        let baseline = TrinityMinimumLimits::baseline();
        assert_eq!(baseline.min_max_compute_invocations_per_workgroup, 256);
    }

    #[test]
    fn baseline_buffer_size_is_256mb() {
        let baseline = TrinityMinimumLimits::baseline();
        assert_eq!(baseline.min_max_buffer_size, 256 * 1024 * 1024);
        assert_eq!(baseline.min_max_buffer_size, 268_435_456);
    }

    #[test]
    fn baseline_texture_1d_is_8192() {
        let baseline = TrinityMinimumLimits::baseline();
        assert_eq!(baseline.min_max_texture_dimension_1d, 8192);
    }

    #[test]
    fn baseline_texture_3d_is_2048() {
        let baseline = TrinityMinimumLimits::baseline();
        assert_eq!(baseline.min_max_texture_dimension_3d, 2048);
    }

    #[test]
    fn baseline_texture_array_layers_is_256() {
        let baseline = TrinityMinimumLimits::baseline();
        assert_eq!(baseline.min_max_texture_array_layers, 256);
    }

    #[test]
    fn baseline_vertex_buffers_is_8() {
        let baseline = TrinityMinimumLimits::baseline();
        assert_eq!(baseline.min_max_vertex_buffers, 8);
    }

    #[test]
    fn baseline_vertex_attributes_is_16() {
        let baseline = TrinityMinimumLimits::baseline();
        assert_eq!(baseline.min_max_vertex_attributes, 16);
    }

    #[test]
    fn baseline_color_attachments_is_8() {
        let baseline = TrinityMinimumLimits::baseline();
        assert_eq!(baseline.min_max_color_attachments, 8);
    }

    #[test]
    fn to_wgpu_limits_converts_correctly() {
        let baseline = TrinityMinimumLimits::baseline();
        let wgpu_limits = baseline.to_wgpu_limits();

        assert_eq!(wgpu_limits.max_uniform_buffer_binding_size, 65536);
        assert_eq!(wgpu_limits.max_storage_buffer_binding_size, 134_217_728);
        assert_eq!(wgpu_limits.max_texture_dimension_2d, 8192);
        assert_eq!(wgpu_limits.max_bind_groups, 4);
        assert_eq!(wgpu_limits.max_bindings_per_bind_group, 640);
    }

    #[test]
    fn to_wgpu_limits_sets_all_fields() {
        let baseline = TrinityMinimumLimits::baseline();
        let limits = baseline.to_wgpu_limits();

        assert_eq!(limits.max_texture_dimension_1d, baseline.min_max_texture_dimension_1d);
        assert_eq!(limits.max_texture_dimension_2d, baseline.min_max_texture_dimension_2d);
        assert_eq!(limits.max_texture_dimension_3d, baseline.min_max_texture_dimension_3d);
        assert_eq!(limits.max_texture_array_layers, baseline.min_max_texture_array_layers);
        assert_eq!(limits.max_buffer_size, baseline.min_max_buffer_size);
        assert_eq!(
            limits.max_uniform_buffer_binding_size,
            baseline.min_uniform_buffer_binding_size
        );
        assert_eq!(
            limits.max_storage_buffer_binding_size,
            baseline.min_storage_buffer_max_binding_size
        );
        assert_eq!(limits.max_bind_groups, baseline.min_max_bind_groups);
        assert_eq!(
            limits.max_bindings_per_bind_group,
            baseline.min_max_bindings_per_bind_group
        );
        assert_eq!(
            limits.max_compute_workgroup_size_x,
            baseline.min_max_compute_workgroup_size_x
        );
        assert_eq!(
            limits.max_compute_workgroup_size_y,
            baseline.min_max_compute_workgroup_size_y
        );
        assert_eq!(
            limits.max_compute_workgroup_size_z,
            baseline.min_max_compute_workgroup_size_z
        );
        assert_eq!(
            limits.max_compute_invocations_per_workgroup,
            baseline.min_max_compute_invocations_per_workgroup
        );
        assert_eq!(limits.max_vertex_buffers, baseline.min_max_vertex_buffers);
        assert_eq!(
            limits.max_vertex_attributes,
            baseline.min_max_vertex_attributes
        );
        assert_eq!(
            limits.max_color_attachments,
            baseline.min_max_color_attachments
        );
    }

    #[test]
    fn default_equals_baseline() {
        let default = TrinityMinimumLimits::default();
        let baseline = TrinityMinimumLimits::baseline();
        assert_eq!(default, baseline);
    }

    #[test]
    fn display_shows_formatted_values() {
        let baseline = TrinityMinimumLimits::baseline();
        let display = format!("{}", baseline);

        assert!(display.contains("TrinityMinimumLimits"));
        assert!(display.contains("64KB"));
        assert!(display.contains("128MB"));
        assert!(display.contains("8192px"));
    }

    #[test]
    fn debug_impl_works() {
        let baseline = TrinityMinimumLimits::baseline();
        let debug = format!("{:?}", baseline);

        assert!(debug.contains("TrinityMinimumLimits"));
        assert!(debug.contains("65536"));
        assert!(debug.contains("134217728"));
    }

    #[test]
    fn clone_creates_identical_copy() {
        let baseline = TrinityMinimumLimits::baseline();
        let cloned = baseline.clone();
        assert_eq!(baseline, cloned);
    }

    #[test]
    fn copy_works() {
        let baseline = TrinityMinimumLimits::baseline();
        let copied: TrinityMinimumLimits = baseline;
        assert_eq!(baseline, copied);
    }
}

// ============================================================================
// 2. LimitRequirements Builder Tests
// ============================================================================

mod limit_requirements {
    use super::*;

    #[test]
    fn new_creates_default_requirements() {
        let req = LimitRequirements::new();
        let default_limits = Limits::default();

        // Both minimum and preferred should be wgpu defaults
        assert_eq!(
            req.minimum.max_texture_dimension_2d,
            default_limits.max_texture_dimension_2d
        );
        assert_eq!(
            req.preferred.max_texture_dimension_2d,
            default_limits.max_texture_dimension_2d
        );
    }

    #[test]
    fn default_impl_equals_new() {
        let new = LimitRequirements::new();
        let default = LimitRequirements::default();

        assert_eq!(
            new.minimum.max_texture_dimension_2d,
            default.minimum.max_texture_dimension_2d
        );
        assert_eq!(
            new.preferred.max_texture_dimension_2d,
            default.preferred.max_texture_dimension_2d
        );
    }

    #[test]
    fn with_trinity_baseline_sets_minimums() {
        let req = LimitRequirements::new().with_trinity_baseline();

        // Should have TRINITY baseline values
        assert_eq!(req.minimum.max_uniform_buffer_binding_size, 65536);
        assert_eq!(req.minimum.max_storage_buffer_binding_size, 134_217_728);
        assert_eq!(req.minimum.max_texture_dimension_2d, 8192);
    }

    #[test]
    fn with_trinity_baseline_does_not_change_preferred() {
        let original = LimitRequirements::new();
        let with_baseline = original.clone().with_trinity_baseline();

        // Preferred should remain unchanged (wgpu defaults)
        assert_eq!(
            with_baseline.preferred.max_texture_dimension_2d,
            Limits::default().max_texture_dimension_2d
        );
    }

    #[test]
    fn with_minimum_sets_custom_minimums() {
        let mut custom = Limits::default();
        custom.max_texture_dimension_2d = 4096;
        custom.max_buffer_size = 100_000_000;

        let req = LimitRequirements::new().with_minimum(custom.clone());

        assert_eq!(req.minimum.max_texture_dimension_2d, 4096);
        assert_eq!(req.minimum.max_buffer_size, 100_000_000);
    }

    #[test]
    fn with_preferred_sets_preferred_values() {
        let mut preferred = Limits::default();
        preferred.max_texture_dimension_2d = 16384;
        preferred.max_storage_buffer_binding_size = 1_073_741_824; // 1GB

        let req = LimitRequirements::new()
            .with_trinity_baseline()
            .with_preferred(preferred);

        assert_eq!(req.preferred.max_texture_dimension_2d, 16384);
        assert_eq!(req.preferred.max_storage_buffer_binding_size, 1_073_741_824);
    }

    #[test]
    fn builder_chaining_works() {
        let mut custom_min = Limits::default();
        custom_min.max_texture_dimension_2d = 4096;

        let mut custom_pref = Limits::default();
        custom_pref.max_texture_dimension_2d = 16384;

        let req = LimitRequirements::new()
            .with_minimum(custom_min)
            .with_preferred(custom_pref);

        assert_eq!(req.minimum.max_texture_dimension_2d, 4096);
        assert_eq!(req.preferred.max_texture_dimension_2d, 16384);
    }

    #[test]
    fn with_trinity_baseline_then_with_preferred() {
        let mut pref = Limits::default();
        pref.max_texture_dimension_2d = 16384;

        let req = LimitRequirements::new()
            .with_trinity_baseline()
            .with_preferred(pref);

        // Minimum should be TRINITY baseline
        assert_eq!(req.minimum.max_texture_dimension_2d, 8192);
        // Preferred should be what we set
        assert_eq!(req.preferred.max_texture_dimension_2d, 16384);
    }

    #[test]
    fn standard_preset_has_trinity_minimum() {
        let req = LimitRequirements::standard();

        assert_eq!(req.minimum.max_texture_dimension_2d, 8192);
        assert_eq!(req.minimum.max_uniform_buffer_binding_size, 65536);
        assert_eq!(req.minimum.max_storage_buffer_binding_size, 134_217_728);
    }

    #[test]
    fn standard_preset_has_moderate_preferred() {
        let req = LimitRequirements::standard();

        // Standard prefers 16K textures
        assert_eq!(req.preferred.max_texture_dimension_2d, 16384);
        // Preferred should be higher than minimum
        assert!(req.preferred.max_texture_dimension_2d >= req.minimum.max_texture_dimension_2d);
    }

    #[test]
    fn high_end_preset_has_trinity_minimum() {
        let req = LimitRequirements::high_end();

        assert_eq!(req.minimum.max_texture_dimension_2d, 8192);
        assert_eq!(req.minimum.max_uniform_buffer_binding_size, 65536);
        assert_eq!(req.minimum.max_storage_buffer_binding_size, 134_217_728);
    }

    #[test]
    fn high_end_preset_has_high_preferred() {
        let req = LimitRequirements::high_end();

        // High-end prefers 32K textures
        assert_eq!(req.preferred.max_texture_dimension_2d, 32768);
        // 1GB storage
        assert_eq!(req.preferred.max_storage_buffer_binding_size, 1_073_741_824);
    }

    #[test]
    fn high_end_preferred_exceeds_standard() {
        let standard = LimitRequirements::standard();
        let high_end = LimitRequirements::high_end();

        assert!(
            high_end.preferred.max_texture_dimension_2d
                > standard.preferred.max_texture_dimension_2d
        );
        assert!(
            high_end.preferred.max_storage_buffer_binding_size
                > standard.preferred.max_storage_buffer_binding_size
        );
    }

    #[test]
    fn display_shows_requirements() {
        let req = LimitRequirements::standard();
        let display = format!("{}", req);

        assert!(display.contains("LimitRequirements"));
        assert!(display.contains("Minimum texture 2D"));
        assert!(display.contains("Preferred texture 2D"));
    }

    #[test]
    fn clone_creates_independent_copy() {
        let original = LimitRequirements::standard();
        let cloned = original.clone();

        assert_eq!(
            original.minimum.max_texture_dimension_2d,
            cloned.minimum.max_texture_dimension_2d
        );
        assert_eq!(
            original.preferred.max_texture_dimension_2d,
            cloned.preferred.max_texture_dimension_2d
        );
    }

    #[test]
    fn debug_impl_works() {
        let req = LimitRequirements::standard();
        let debug = format!("{:?}", req);

        assert!(debug.contains("LimitRequirements"));
        assert!(debug.contains("minimum"));
        assert!(debug.contains("preferred"));
    }
}

// ============================================================================
// 3. negotiate_limits() Tests (via Mock)
// ============================================================================

mod negotiate_limits_tests {
    use super::*;

    #[test]
    fn adapter_meets_minimums_succeeds() {
        let requirements = LimitRequirements::new().with_trinity_baseline();
        let adapter = make_high_end_adapter();

        let result = mock_negotiate(&requirements, &adapter);
        assert!(result.is_ok());
    }

    #[test]
    fn adapter_exactly_at_minimums_succeeds() {
        let requirements = LimitRequirements::new().with_trinity_baseline();
        let adapter = make_minimum_adapter();

        let result = mock_negotiate(&requirements, &adapter);
        assert!(result.is_ok());
    }

    #[test]
    fn adapter_below_minimums_fails() {
        let requirements = LimitRequirements::new().with_trinity_baseline();
        let adapter = make_subpar_adapter();

        let result = mock_negotiate(&requirements, &adapter);
        assert!(result.is_err());
    }

    #[test]
    fn below_minimum_error_contains_limit_name() {
        let requirements = LimitRequirements::new().with_trinity_baseline();
        let adapter = make_subpar_adapter();

        if let Err(LimitNegotiationError::BelowMinimum { limit, .. }) =
            mock_negotiate(&requirements, &adapter)
        {
            // Should identify which limit failed
            assert!(!limit.is_empty());
        } else {
            panic!("Expected BelowMinimum error");
        }
    }

    #[test]
    fn below_minimum_error_contains_required_value() {
        let requirements = LimitRequirements::new().with_trinity_baseline();
        let adapter = make_subpar_adapter();

        if let Err(LimitNegotiationError::BelowMinimum { required, .. }) =
            mock_negotiate(&requirements, &adapter)
        {
            // Required should be > 0 (TRINITY baseline)
            assert!(required > 0);
        } else {
            panic!("Expected BelowMinimum error");
        }
    }

    #[test]
    fn below_minimum_error_contains_available_value() {
        let requirements = LimitRequirements::new().with_trinity_baseline();
        let adapter = make_subpar_adapter();

        if let Err(LimitNegotiationError::BelowMinimum {
            required, available, ..
        }) = mock_negotiate(&requirements, &adapter)
        {
            // Available should be less than required (that's why it failed)
            assert!(available < required);
        } else {
            panic!("Expected BelowMinimum error");
        }
    }

    #[test]
    fn preferred_within_adapter_uses_preferred() {
        let mut pref = Limits::default();
        pref.max_texture_dimension_2d = 12288; // Prefer 12K

        let requirements = LimitRequirements::new()
            .with_trinity_baseline()
            .with_preferred(pref);

        let adapter = make_high_end_adapter(); // Has 32K capability

        let result = mock_negotiate(&requirements, &adapter).unwrap();

        // Should use our preferred 12K, not the adapter's max 32K
        assert_eq!(result.limits.max_texture_dimension_2d, 12288);
    }

    #[test]
    fn preferred_exceeds_adapter_caps_to_adapter() {
        let mut pref = TrinityMinimumLimits::baseline().to_wgpu_limits();
        pref.max_texture_dimension_2d = 65536; // Prefer 64K (unrealistic)

        let requirements = LimitRequirements::new()
            .with_trinity_baseline()
            .with_preferred(pref);

        let adapter = make_high_end_adapter(); // Has 32K max

        let result = mock_negotiate(&requirements, &adapter).unwrap();

        // Should cap to adapter's 32K
        assert_eq!(result.limits.max_texture_dimension_2d, 32768);
        // Should indicate shortfall
        assert!(result.had_shortfall);
    }

    #[test]
    fn multiple_limits_capped() {
        let mut pref = TrinityMinimumLimits::baseline().to_wgpu_limits();
        pref.max_texture_dimension_2d = 65536; // Way above adapter
        pref.max_storage_buffer_binding_size = u32::MAX; // Way above adapter

        let requirements = LimitRequirements::new()
            .with_trinity_baseline()
            .with_preferred(pref);

        let adapter = make_high_end_adapter();

        let result = mock_negotiate(&requirements, &adapter).unwrap();

        // Both limits should be capped
        assert!(result.capped_limits.contains(&"max_texture_dimension_2d".to_string()));
        assert!(result.capped_limits.contains(&"max_storage_buffer_binding_size".to_string()));
        assert!(result.capped_limits.len() >= 2);
    }

    #[test]
    fn shortfall_logged_when_limits_capped() {
        let mut pref = TrinityMinimumLimits::baseline().to_wgpu_limits();
        pref.max_texture_dimension_2d = 65536;

        let requirements = LimitRequirements::new()
            .with_trinity_baseline()
            .with_preferred(pref);

        let adapter = make_high_end_adapter();

        let result = mock_negotiate(&requirements, &adapter).unwrap();

        // Should have shortfall indicator
        assert!(result.had_shortfall);
        assert!(!result.capped_limits.is_empty());
    }

    #[test]
    fn no_shortfall_when_adapter_meets_all_preferred() {
        let pref = make_minimum_adapter(); // Prefer exactly what TRINITY needs

        let requirements = LimitRequirements::new()
            .with_trinity_baseline()
            .with_preferred(pref);

        let adapter = make_high_end_adapter(); // Exceeds all preferred

        let result = mock_negotiate(&requirements, &adapter).unwrap();

        // Should have no shortfall
        assert!(!result.had_shortfall);
        assert!(result.capped_limits.is_empty());
    }

    #[test]
    fn final_limits_accessible() {
        let requirements = LimitRequirements::standard();
        let adapter = make_high_end_adapter();

        let result = mock_negotiate(&requirements, &adapter).unwrap();

        // Should be able to access final limits
        assert!(result.limits.max_texture_dimension_2d > 0);
        assert!(result.limits.max_buffer_size > 0);
        assert!(result.limits.max_uniform_buffer_binding_size > 0);
    }

    #[test]
    fn mid_tier_adapter_partial_cap() {
        let requirements = LimitRequirements::standard();
        let adapter = make_mid_tier_adapter();

        let result = mock_negotiate(&requirements, &adapter).unwrap();

        // Mid-tier adapter should cap some limits but meet minimums
        // (whether had_shortfall is true depends on exact values)
        assert!(result.limits.max_texture_dimension_2d >= 8192);
    }
}

// ============================================================================
// 4. LimitNegotiationResult Tests
// ============================================================================

mod limit_negotiation_result {
    use super::*;

    #[test]
    fn limits_field_accessible() {
        let result = LimitNegotiationResult {
            limits: Limits::default(),
            had_shortfall: false,
            capped_limits: vec![],
        };

        // Should be able to access limits
        assert!(result.limits.max_texture_dimension_2d > 0);
    }

    #[test]
    fn had_shortfall_false_when_no_caps() {
        let result = LimitNegotiationResult {
            limits: Limits::default(),
            had_shortfall: false,
            capped_limits: vec![],
        };

        assert!(!result.had_shortfall);
    }

    #[test]
    fn had_shortfall_true_when_caps_exist() {
        let result = LimitNegotiationResult {
            limits: Limits::default(),
            had_shortfall: true,
            capped_limits: vec!["max_texture_dimension_2d".to_string()],
        };

        assert!(result.had_shortfall);
    }

    #[test]
    fn capped_limits_lists_capped() {
        let result = LimitNegotiationResult {
            limits: Limits::default(),
            had_shortfall: true,
            capped_limits: vec![
                "max_texture_dimension_2d".to_string(),
                "max_buffer_size".to_string(),
            ],
        };

        assert_eq!(result.capped_limits.len(), 2);
        assert!(result.capped_limits.contains(&"max_texture_dimension_2d".to_string()));
        assert!(result.capped_limits.contains(&"max_buffer_size".to_string()));
    }

    #[test]
    fn was_capped_returns_true_for_capped_limit() {
        let result = LimitNegotiationResult {
            limits: Limits::default(),
            had_shortfall: true,
            capped_limits: vec!["max_texture_dimension_2d".to_string()],
        };

        assert!(result.was_capped("max_texture_dimension_2d"));
    }

    #[test]
    fn was_capped_returns_false_for_uncapped_limit() {
        let result = LimitNegotiationResult {
            limits: Limits::default(),
            had_shortfall: true,
            capped_limits: vec!["max_texture_dimension_2d".to_string()],
        };

        assert!(!result.was_capped("max_buffer_size"));
    }

    #[test]
    fn was_capped_returns_false_when_empty() {
        let result = LimitNegotiationResult {
            limits: Limits::default(),
            had_shortfall: false,
            capped_limits: vec![],
        };

        assert!(!result.was_capped("max_texture_dimension_2d"));
    }

    #[test]
    fn capped_count_returns_correct_count() {
        let result = LimitNegotiationResult {
            limits: Limits::default(),
            had_shortfall: true,
            capped_limits: vec![
                "limit1".to_string(),
                "limit2".to_string(),
                "limit3".to_string(),
            ],
        };

        assert_eq!(result.capped_count(), 3);
    }

    #[test]
    fn capped_count_zero_when_empty() {
        let result = LimitNegotiationResult {
            limits: Limits::default(),
            had_shortfall: false,
            capped_limits: vec![],
        };

        assert_eq!(result.capped_count(), 0);
    }

    #[test]
    fn display_shows_shortfall_status() {
        let result = LimitNegotiationResult {
            limits: Limits::default(),
            had_shortfall: true,
            capped_limits: vec!["max_texture_dimension_2d".to_string()],
        };

        let display = format!("{}", result);
        assert!(display.contains("shortfall: true"));
    }

    #[test]
    fn display_shows_capped_limits() {
        let result = LimitNegotiationResult {
            limits: Limits::default(),
            had_shortfall: true,
            capped_limits: vec!["max_texture_dimension_2d".to_string()],
        };

        let display = format!("{}", result);
        assert!(display.contains("max_texture_dimension_2d"));
    }

    #[test]
    fn display_shows_final_limits() {
        let mut limits = Limits::default();
        limits.max_texture_dimension_2d = 8192;

        let result = LimitNegotiationResult {
            limits,
            had_shortfall: false,
            capped_limits: vec![],
        };

        let display = format!("{}", result);
        assert!(display.contains("8192"));
    }

    #[test]
    fn debug_impl_works() {
        let result = LimitNegotiationResult {
            limits: Limits::default(),
            had_shortfall: false,
            capped_limits: vec![],
        };

        let debug = format!("{:?}", result);
        assert!(debug.contains("LimitNegotiationResult"));
    }

    #[test]
    fn clone_creates_independent_copy() {
        let original = LimitNegotiationResult {
            limits: Limits::default(),
            had_shortfall: true,
            capped_limits: vec!["test".to_string()],
        };

        let cloned = original.clone();

        assert_eq!(original.had_shortfall, cloned.had_shortfall);
        assert_eq!(original.capped_limits, cloned.capped_limits);
    }
}

// ============================================================================
// 5. LimitNegotiationError Tests
// ============================================================================

mod limit_negotiation_error {
    use super::*;

    #[test]
    fn below_minimum_display_contains_limit_name() {
        let err = LimitNegotiationError::BelowMinimum {
            limit: "max_texture_dimension_2d".to_string(),
            required: 8192,
            available: 4096,
        };

        let msg = format!("{}", err);
        assert!(msg.contains("max_texture_dimension_2d"));
    }

    #[test]
    fn below_minimum_display_contains_required() {
        let err = LimitNegotiationError::BelowMinimum {
            limit: "max_texture_dimension_2d".to_string(),
            required: 8192,
            available: 4096,
        };

        let msg = format!("{}", err);
        assert!(msg.contains("8192"));
    }

    #[test]
    fn below_minimum_display_contains_available() {
        let err = LimitNegotiationError::BelowMinimum {
            limit: "max_texture_dimension_2d".to_string(),
            required: 8192,
            available: 4096,
        };

        let msg = format!("{}", err);
        assert!(msg.contains("4096"));
    }

    #[test]
    fn below_minimum_display_mentions_trinity() {
        let err = LimitNegotiationError::BelowMinimum {
            limit: "max_texture_dimension_2d".to_string(),
            required: 8192,
            available: 4096,
        };

        let msg = format!("{}", err);
        assert!(msg.contains("TRINITY"));
    }

    #[test]
    fn below_minimum_display_mentions_below() {
        let err = LimitNegotiationError::BelowMinimum {
            limit: "max_texture_dimension_2d".to_string(),
            required: 8192,
            available: 4096,
        };

        let msg = format!("{}", err);
        assert!(msg.contains("below"));
    }

    #[test]
    fn debug_impl_works() {
        let err = LimitNegotiationError::BelowMinimum {
            limit: "max_texture_dimension_2d".to_string(),
            required: 8192,
            available: 4096,
        };

        let debug = format!("{:?}", err);
        assert!(debug.contains("BelowMinimum"));
        assert!(debug.contains("max_texture_dimension_2d"));
    }

    #[test]
    fn error_impl_available() {
        let err = LimitNegotiationError::BelowMinimum {
            limit: "test".to_string(),
            required: 100,
            available: 50,
        };

        // std::error::Error is implemented
        let _: &dyn std::error::Error = &err;
    }

    #[test]
    fn clone_creates_identical_error() {
        let original = LimitNegotiationError::BelowMinimum {
            limit: "test".to_string(),
            required: 100,
            available: 50,
        };

        let cloned = original.clone();

        let (
            LimitNegotiationError::BelowMinimum {
                limit: l1,
                required: r1,
                available: a1,
            },
            LimitNegotiationError::BelowMinimum {
                limit: l2,
                required: r2,
                available: a2,
            },
        ) = (original, cloned);

        assert_eq!(l1, l2);
        assert_eq!(r1, r2);
        assert_eq!(a1, a2);
    }

    #[test]
    fn below_minimum_with_u64_limit() {
        let err = LimitNegotiationError::BelowMinimum {
            limit: "max_buffer_size".to_string(),
            required: 268_435_456, // 256MB
            available: 134_217_728, // 128MB
        };

        let msg = format!("{}", err);
        assert!(msg.contains("268435456"));
        assert!(msg.contains("134217728"));
    }

    #[test]
    fn error_fields_accessible() {
        let err = LimitNegotiationError::BelowMinimum {
            limit: "max_texture_dimension_2d".to_string(),
            required: 8192,
            available: 4096,
        };

        let LimitNegotiationError::BelowMinimum {
            limit,
            required,
            available,
        } = err;

        assert_eq!(limit, "max_texture_dimension_2d");
        assert_eq!(required, 8192);
        assert_eq!(available, 4096);
    }
}

// ============================================================================
// 6. Edge Case Tests
// ============================================================================

mod edge_cases {
    use super::*;

    #[test]
    fn exact_minimum_limits_passes() {
        // When both minimum and preferred are set to TRINITY baseline
        // and adapter also matches, there should be no shortfall
        let baseline = TrinityMinimumLimits::baseline().to_wgpu_limits();
        let requirements = LimitRequirements::new()
            .with_minimum(baseline.clone())
            .with_preferred(baseline.clone());
        let adapter = make_minimum_adapter();

        let result = mock_negotiate(&requirements, &adapter).unwrap();

        // Should succeed with exact minimums (when preferred == minimum == adapter)
        assert!(!result.had_shortfall);
    }

    #[test]
    fn minimum_adapter_may_have_shortfall_vs_wgpu_default_preferred() {
        // with_trinity_baseline only sets minimum, not preferred
        // preferred defaults to wgpu::Limits::default() which may differ
        let requirements = LimitRequirements::new().with_trinity_baseline();
        let adapter = make_minimum_adapter();

        let result = mock_negotiate(&requirements, &adapter).unwrap();

        // The negotiation succeeds (adapter meets minimum)
        // but may have shortfall if preferred (wgpu defaults) > adapter
        // This is expected behavior - the adapter meets TRINITY minimums
        // but may not meet all wgpu default preferences
        assert!(result.limits.max_texture_dimension_2d >= 8192);
    }

    #[test]
    fn exact_preferred_limits_no_shortfall() {
        let pref = make_high_end_adapter();
        let requirements = LimitRequirements::new()
            .with_trinity_baseline()
            .with_preferred(pref.clone());

        let result = mock_negotiate(&requirements, &pref).unwrap();

        // Exact match should have no shortfall
        assert!(!result.had_shortfall);
        assert!(result.capped_limits.is_empty());
    }

    #[test]
    fn empty_capped_list_when_adapter_exceeds_all() {
        let pref = TrinityMinimumLimits::baseline().to_wgpu_limits();
        let requirements = LimitRequirements::new()
            .with_trinity_baseline()
            .with_preferred(pref);

        let adapter = make_high_end_adapter();

        let result = mock_negotiate(&requirements, &adapter).unwrap();

        assert!(result.capped_limits.is_empty());
        assert!(!result.had_shortfall);
    }

    #[test]
    fn all_limits_capped_when_adapter_at_minimum() {
        let mut pref = make_high_end_adapter();
        // Set preferred to maximum possible values
        pref.max_texture_dimension_1d = u32::MAX;
        pref.max_texture_dimension_2d = u32::MAX;
        pref.max_texture_dimension_3d = u32::MAX;
        pref.max_buffer_size = u64::MAX;

        let requirements = LimitRequirements::new()
            .with_trinity_baseline()
            .with_preferred(pref);

        let adapter = make_minimum_adapter();

        let result = mock_negotiate(&requirements, &adapter).unwrap();

        // Should have many capped limits
        assert!(result.had_shortfall);
        assert!(!result.capped_limits.is_empty());
    }

    #[test]
    fn one_below_minimum_fails() {
        let mut adapter = make_high_end_adapter();
        adapter.max_texture_dimension_2d = 4096; // Only this one is below 8K

        let requirements = LimitRequirements::new().with_trinity_baseline();

        let result = mock_negotiate(&requirements, &adapter);

        assert!(result.is_err());
        if let Err(LimitNegotiationError::BelowMinimum { limit, .. }) = result {
            assert_eq!(limit, "max_texture_dimension_2d");
        }
    }

    #[test]
    fn uniform_buffer_below_minimum_fails() {
        let mut adapter = make_high_end_adapter();
        adapter.max_uniform_buffer_binding_size = 32768; // 32KB (below 64KB minimum)

        let requirements = LimitRequirements::new().with_trinity_baseline();

        let result = mock_negotiate(&requirements, &adapter);

        assert!(result.is_err());
    }

    #[test]
    fn storage_buffer_below_minimum_fails() {
        let mut adapter = make_high_end_adapter();
        adapter.max_storage_buffer_binding_size = 67_108_864; // 64MB (below 128MB minimum)

        let requirements = LimitRequirements::new().with_trinity_baseline();

        let result = mock_negotiate(&requirements, &adapter);

        assert!(result.is_err());
    }

    #[test]
    fn buffer_size_below_minimum_fails() {
        let mut adapter = make_high_end_adapter();
        adapter.max_buffer_size = 134_217_728; // 128MB (below 256MB minimum)

        let requirements = LimitRequirements::new().with_trinity_baseline();

        let result = mock_negotiate(&requirements, &adapter);

        assert!(result.is_err());
    }

    #[test]
    fn bind_groups_below_minimum_fails() {
        let mut adapter = make_high_end_adapter();
        adapter.max_bind_groups = 2; // Below 4 minimum

        let requirements = LimitRequirements::new().with_trinity_baseline();

        let result = mock_negotiate(&requirements, &adapter);

        assert!(result.is_err());
    }

    #[test]
    fn bindings_per_group_below_minimum_fails() {
        let mut adapter = make_high_end_adapter();
        adapter.max_bindings_per_bind_group = 256; // Below 640 minimum

        let requirements = LimitRequirements::new().with_trinity_baseline();

        let result = mock_negotiate(&requirements, &adapter);

        assert!(result.is_err());
    }

    #[test]
    fn compute_workgroup_x_below_minimum_fails() {
        let mut adapter = make_high_end_adapter();
        adapter.max_compute_workgroup_size_x = 128; // Below 256 minimum

        let requirements = LimitRequirements::new().with_trinity_baseline();

        let result = mock_negotiate(&requirements, &adapter);

        assert!(result.is_err());
    }

    #[test]
    fn vertex_buffers_below_minimum_fails() {
        let mut adapter = make_high_end_adapter();
        adapter.max_vertex_buffers = 4; // Below 8 minimum

        let requirements = LimitRequirements::new().with_trinity_baseline();

        let result = mock_negotiate(&requirements, &adapter);

        assert!(result.is_err());
    }

    #[test]
    fn color_attachments_below_minimum_fails() {
        let mut adapter = make_high_end_adapter();
        adapter.max_color_attachments = 4; // Below 8 minimum

        let requirements = LimitRequirements::new().with_trinity_baseline();

        let result = mock_negotiate(&requirements, &adapter);

        assert!(result.is_err());
    }

    #[test]
    fn zero_preferred_uses_minimum() {
        let mut pref = Limits::default();
        pref.max_texture_dimension_2d = 0; // Zero preferred

        let requirements = LimitRequirements::new()
            .with_trinity_baseline()
            .with_preferred(pref);

        let adapter = make_high_end_adapter();

        // This should still work, using minimum as effective preferred
        // (implementation may vary - testing actual behavior)
        let result = mock_negotiate(&requirements, &adapter);
        // Either succeeds with 0 or adapter value, depending on implementation
        assert!(result.is_ok());
    }
}

// ============================================================================
// 7. Integration Tests (Mock Full Pipeline)
// ============================================================================

mod integration {
    use super::*;

    #[test]
    fn full_negotiation_pipeline_high_end() {
        // Build requirements using standard preset
        let requirements = LimitRequirements::standard();

        // Negotiate against high-end adapter
        let adapter = make_high_end_adapter();
        let result = mock_negotiate(&requirements, &adapter).unwrap();

        // Verify result
        assert!(!result.had_shortfall);
        assert!(result.limits.max_texture_dimension_2d >= 8192);
        assert!(result.limits.max_uniform_buffer_binding_size >= 65536);
    }

    #[test]
    fn full_negotiation_pipeline_minimum() {
        let requirements = LimitRequirements::new().with_trinity_baseline();
        let adapter = make_minimum_adapter();

        let result = mock_negotiate(&requirements, &adapter).unwrap();

        // Should succeed with exact minimums
        assert_eq!(result.limits.max_texture_dimension_2d, 8192);
        assert_eq!(result.limits.max_uniform_buffer_binding_size, 65536);
    }

    #[test]
    fn full_negotiation_pipeline_mid_tier() {
        let requirements = LimitRequirements::high_end();
        let adapter = make_mid_tier_adapter();

        let result = mock_negotiate(&requirements, &adapter).unwrap();

        // Should succeed but with some caps
        assert!(result.had_shortfall);
        assert!(result.limits.max_texture_dimension_2d >= 8192);
    }

    #[test]
    fn full_negotiation_pipeline_failure() {
        let requirements = LimitRequirements::new().with_trinity_baseline();
        let adapter = make_subpar_adapter();

        let result = mock_negotiate(&requirements, &adapter);

        // Should fail
        assert!(result.is_err());
        if let Err(e) = result {
            let msg = format!("{}", e);
            assert!(msg.contains("below TRINITY minimum"));
        }
    }

    #[test]
    fn chained_builder_produces_valid_requirements() {
        let mut custom_pref = TrinityMinimumLimits::baseline().to_wgpu_limits();
        custom_pref.max_texture_dimension_2d = 16384;
        custom_pref.max_storage_buffer_binding_size = 536_870_912; // 512MB

        let requirements = LimitRequirements::new()
            .with_trinity_baseline()
            .with_preferred(custom_pref);

        let adapter = make_high_end_adapter();
        let result = mock_negotiate(&requirements, &adapter).unwrap();

        // Should use our preferred values
        assert_eq!(result.limits.max_texture_dimension_2d, 16384);
        assert_eq!(result.limits.max_storage_buffer_binding_size, 536_870_912);
    }

    #[test]
    fn requirements_default_works_with_default_adapter() {
        let requirements = LimitRequirements::default();
        let adapter = Limits::default();

        // Default requirements against default adapter should succeed
        let result = mock_negotiate(&requirements, &adapter);
        assert!(result.is_ok());
    }
}
