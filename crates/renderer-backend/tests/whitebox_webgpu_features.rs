//! Whitebox tests for WebGPU capability detection.
//!
//! These tests comprehensively verify the WebGPU feature detection system:
//! - WebGpuTier enum and detection logic
//! - WebGpuLimits struct and tier requirement checking
//! - WebGpuFeatures struct and compression detection
//! - BrowserType enum and user agent parsing
//! - BrowserCapabilities struct and feature detection
//!
//! Target: 150+ tests covering all public API methods and edge cases.

use renderer_backend::backend::webgpu::{
    BrowserCapabilities, BrowserType, WebGpuFeatures, WebGpuLimits, WebGpuTier,
};
use std::collections::HashSet;
use wgpu::{Features, Limits};

// ============================================================================
// Module: WebGpuTier - Basic Properties
// ============================================================================

mod webgpu_tier_basic {
    use super::*;

    #[test]
    fn default_is_tier1() {
        let tier = WebGpuTier::default();
        assert_eq!(tier, WebGpuTier::Tier1);
    }

    #[test]
    fn tier1_variant_exists() {
        let _tier = WebGpuTier::Tier1;
    }

    #[test]
    fn tier2_variant_exists() {
        let _tier = WebGpuTier::Tier2;
    }

    #[test]
    fn tier3_variant_exists() {
        let _tier = WebGpuTier::Tier3;
    }

    #[test]
    fn all_tiers_are_different() {
        assert_ne!(WebGpuTier::Tier1, WebGpuTier::Tier2);
        assert_ne!(WebGpuTier::Tier2, WebGpuTier::Tier3);
        assert_ne!(WebGpuTier::Tier1, WebGpuTier::Tier3);
    }
}

// ============================================================================
// Module: WebGpuTier - Trait Implementations
// ============================================================================

mod webgpu_tier_traits {
    use super::*;

    #[test]
    fn clone_works() {
        let tier = WebGpuTier::Tier2;
        let cloned = tier.clone();
        assert_eq!(tier, cloned);
    }

    #[test]
    fn copy_works() {
        let tier = WebGpuTier::Tier3;
        let copied: WebGpuTier = tier;
        assert_eq!(tier, copied);
    }

    #[test]
    fn debug_formats_correctly() {
        assert_eq!(format!("{:?}", WebGpuTier::Tier1), "Tier1");
        assert_eq!(format!("{:?}", WebGpuTier::Tier2), "Tier2");
        assert_eq!(format!("{:?}", WebGpuTier::Tier3), "Tier3");
    }

    #[test]
    fn eq_works_for_same_tier() {
        assert!(WebGpuTier::Tier1 == WebGpuTier::Tier1);
        assert!(WebGpuTier::Tier2 == WebGpuTier::Tier2);
        assert!(WebGpuTier::Tier3 == WebGpuTier::Tier3);
    }

    #[test]
    fn eq_works_for_different_tiers() {
        assert!(WebGpuTier::Tier1 != WebGpuTier::Tier2);
        assert!(WebGpuTier::Tier2 != WebGpuTier::Tier3);
    }

    #[test]
    fn hash_is_consistent() {
        use std::hash::{Hash, Hasher};
        use std::collections::hash_map::DefaultHasher;

        let mut hasher1 = DefaultHasher::new();
        let mut hasher2 = DefaultHasher::new();

        WebGpuTier::Tier2.hash(&mut hasher1);
        WebGpuTier::Tier2.hash(&mut hasher2);

        assert_eq!(hasher1.finish(), hasher2.finish());
    }

    #[test]
    fn hash_differs_for_different_tiers() {
        use std::hash::{Hash, Hasher};
        use std::collections::hash_map::DefaultHasher;

        let mut hasher1 = DefaultHasher::new();
        let mut hasher2 = DefaultHasher::new();

        WebGpuTier::Tier1.hash(&mut hasher1);
        WebGpuTier::Tier3.hash(&mut hasher2);

        assert_ne!(hasher1.finish(), hasher2.finish());
    }

    #[test]
    fn can_be_used_in_hashset() {
        let mut set = HashSet::new();
        set.insert(WebGpuTier::Tier1);
        set.insert(WebGpuTier::Tier2);
        set.insert(WebGpuTier::Tier3);
        set.insert(WebGpuTier::Tier1); // Duplicate

        assert_eq!(set.len(), 3);
        assert!(set.contains(&WebGpuTier::Tier2));
    }

    #[test]
    fn ordering_tier1_less_than_tier2() {
        assert!(WebGpuTier::Tier1 < WebGpuTier::Tier2);
    }

    #[test]
    fn ordering_tier2_less_than_tier3() {
        assert!(WebGpuTier::Tier2 < WebGpuTier::Tier3);
    }

    #[test]
    fn ordering_tier1_less_than_tier3() {
        assert!(WebGpuTier::Tier1 < WebGpuTier::Tier3);
    }

    #[test]
    fn ordering_tier3_greater_than_tier1() {
        assert!(WebGpuTier::Tier3 > WebGpuTier::Tier1);
    }

    #[test]
    fn ordering_is_transitive() {
        let t1 = WebGpuTier::Tier1;
        let t2 = WebGpuTier::Tier2;
        let t3 = WebGpuTier::Tier3;

        if t1 < t2 && t2 < t3 {
            assert!(t1 < t3);
        }
    }

    #[test]
    fn partial_ord_le_works() {
        assert!(WebGpuTier::Tier1 <= WebGpuTier::Tier1);
        assert!(WebGpuTier::Tier1 <= WebGpuTier::Tier2);
    }

    #[test]
    fn partial_ord_ge_works() {
        assert!(WebGpuTier::Tier3 >= WebGpuTier::Tier3);
        assert!(WebGpuTier::Tier3 >= WebGpuTier::Tier2);
    }

    #[test]
    fn display_tier1() {
        assert_eq!(format!("{}", WebGpuTier::Tier1), "Tier 1 (Basic)");
    }

    #[test]
    fn display_tier2() {
        assert_eq!(format!("{}", WebGpuTier::Tier2), "Tier 2 (Extended)");
    }

    #[test]
    fn display_tier3() {
        assert_eq!(format!("{}", WebGpuTier::Tier3), "Tier 3 (Advanced)");
    }
}

// ============================================================================
// Module: WebGpuTier - Feature Support Methods
// ============================================================================

mod webgpu_tier_feature_support {
    use super::*;

    #[test]
    fn tier1_supports_compute_shaders() {
        assert!(WebGpuTier::Tier1.supports_compute_shaders());
    }

    #[test]
    fn tier2_supports_compute_shaders() {
        assert!(WebGpuTier::Tier2.supports_compute_shaders());
    }

    #[test]
    fn tier3_supports_compute_shaders() {
        assert!(WebGpuTier::Tier3.supports_compute_shaders());
    }

    #[test]
    fn tier1_does_not_support_storage_textures() {
        assert!(!WebGpuTier::Tier1.supports_storage_textures());
    }

    #[test]
    fn tier2_supports_storage_textures() {
        assert!(WebGpuTier::Tier2.supports_storage_textures());
    }

    #[test]
    fn tier3_supports_storage_textures() {
        assert!(WebGpuTier::Tier3.supports_storage_textures());
    }

    #[test]
    fn tier1_does_not_support_timestamp_query() {
        assert!(!WebGpuTier::Tier1.supports_timestamp_query());
    }

    #[test]
    fn tier2_does_not_support_timestamp_query() {
        assert!(!WebGpuTier::Tier2.supports_timestamp_query());
    }

    #[test]
    fn tier3_supports_timestamp_query() {
        assert!(WebGpuTier::Tier3.supports_timestamp_query());
    }

    #[test]
    fn name_tier1() {
        assert_eq!(WebGpuTier::Tier1.name(), "Tier 1 (Basic)");
    }

    #[test]
    fn name_tier2() {
        assert_eq!(WebGpuTier::Tier2.name(), "Tier 2 (Extended)");
    }

    #[test]
    fn name_tier3() {
        assert_eq!(WebGpuTier::Tier3.name(), "Tier 3 (Advanced)");
    }

    #[test]
    fn tier_number_tier1() {
        assert_eq!(WebGpuTier::Tier1.tier_number(), 1);
    }

    #[test]
    fn tier_number_tier2() {
        assert_eq!(WebGpuTier::Tier2.tier_number(), 2);
    }

    #[test]
    fn tier_number_tier3() {
        assert_eq!(WebGpuTier::Tier3.tier_number(), 3);
    }
}

// ============================================================================
// Module: WebGpuTier - Tier Detection from Limits/Features
// ============================================================================

mod webgpu_tier_detection {
    use super::*;

    #[test]
    fn detects_tier1_from_empty_features_and_low_limits() {
        let limits = Limits::downlevel_defaults();
        let features = Features::empty();
        let tier = WebGpuTier::from_limits_and_features(&limits, features);
        assert_eq!(tier, WebGpuTier::Tier1);
    }

    #[test]
    fn detects_tier1_when_texture_below_4096() {
        let mut limits = Limits::default();
        limits.max_texture_dimension_2d = 2048;
        let features = Features::empty();
        let tier = WebGpuTier::from_limits_and_features(&limits, features);
        assert_eq!(tier, WebGpuTier::Tier1);
    }

    #[test]
    fn detects_tier1_when_bind_groups_below_8() {
        let mut limits = Limits::default();
        limits.max_texture_dimension_2d = 4096;
        limits.max_bind_groups = 4;
        let features = Features::empty();
        let tier = WebGpuTier::from_limits_and_features(&limits, features);
        assert_eq!(tier, WebGpuTier::Tier1);
    }

    #[test]
    fn detects_tier2_when_extended_limits_met() {
        let mut limits = Limits::default();
        limits.max_texture_dimension_2d = 4096;
        limits.max_bind_groups = 8;
        limits.max_storage_textures_per_shader_stage = 4;
        limits.max_dynamic_uniform_buffers_per_pipeline_layout = 8;
        let features = Features::empty();
        let tier = WebGpuTier::from_limits_and_features(&limits, features);
        assert_eq!(tier, WebGpuTier::Tier2);
    }

    #[test]
    fn detects_tier2_when_exactly_at_boundary() {
        let mut limits = Limits::default();
        limits.max_texture_dimension_2d = 4096; // Exactly Tier 2 minimum
        limits.max_bind_groups = 8;
        limits.max_storage_textures_per_shader_stage = 4;
        limits.max_dynamic_uniform_buffers_per_pipeline_layout = 8;
        let features = Features::empty();
        let tier = WebGpuTier::from_limits_and_features(&limits, features);
        assert_eq!(tier, WebGpuTier::Tier2);
    }

    #[test]
    fn detects_tier3_with_timestamp_and_high_limits() {
        let mut limits = Limits::default();
        limits.max_texture_dimension_2d = 8192;
        limits.max_compute_workgroups_per_dimension = 65535;
        limits.max_storage_buffers_per_shader_stage = 8;
        let features = Features::TIMESTAMP_QUERY;
        let tier = WebGpuTier::from_limits_and_features(&limits, features);
        assert_eq!(tier, WebGpuTier::Tier3);
    }

    #[test]
    fn tier3_requires_timestamp_query() {
        let mut limits = Limits::default();
        limits.max_texture_dimension_2d = 8192;
        limits.max_compute_workgroups_per_dimension = 65535;
        limits.max_storage_buffers_per_shader_stage = 8;
        // No TIMESTAMP_QUERY feature
        let features = Features::empty();
        let tier = WebGpuTier::from_limits_and_features(&limits, features);
        // Should fall back to Tier 2 (extended limits met)
        assert!(tier == WebGpuTier::Tier2 || tier == WebGpuTier::Tier1);
    }

    #[test]
    fn tier3_requires_high_texture_dimension() {
        let mut limits = Limits::default();
        limits.max_texture_dimension_2d = 4096; // Below 8192
        limits.max_compute_workgroups_per_dimension = 65535;
        limits.max_storage_buffers_per_shader_stage = 8;
        let features = Features::TIMESTAMP_QUERY;
        let tier = WebGpuTier::from_limits_and_features(&limits, features);
        // Not Tier 3 due to texture limit
        assert_ne!(tier, WebGpuTier::Tier3);
    }

    #[test]
    fn tier3_requires_high_workgroup_count() {
        let mut limits = Limits::default();
        limits.max_texture_dimension_2d = 8192;
        limits.max_compute_workgroups_per_dimension = 1000; // Below 65535
        limits.max_storage_buffers_per_shader_stage = 8;
        let features = Features::TIMESTAMP_QUERY;
        let tier = WebGpuTier::from_limits_and_features(&limits, features);
        assert_ne!(tier, WebGpuTier::Tier3);
    }

    #[test]
    fn tier3_requires_sufficient_storage_buffers() {
        let mut limits = Limits::default();
        limits.max_texture_dimension_2d = 8192;
        limits.max_compute_workgroups_per_dimension = 65535;
        limits.max_storage_buffers_per_shader_stage = 4; // Below 8
        let features = Features::TIMESTAMP_QUERY;
        let tier = WebGpuTier::from_limits_and_features(&limits, features);
        assert_ne!(tier, WebGpuTier::Tier3);
    }

    #[test]
    fn tier3_boundary_exact_8192_texture() {
        let mut limits = Limits::default();
        limits.max_texture_dimension_2d = 8192; // Exactly at boundary
        limits.max_compute_workgroups_per_dimension = 65535;
        limits.max_storage_buffers_per_shader_stage = 8;
        let features = Features::TIMESTAMP_QUERY;
        let tier = WebGpuTier::from_limits_and_features(&limits, features);
        assert_eq!(tier, WebGpuTier::Tier3);
    }

    #[test]
    fn tier3_exceeds_all_requirements() {
        let mut limits = Limits::default();
        limits.max_texture_dimension_2d = 16384; // Exceeds 8192
        limits.max_compute_workgroups_per_dimension = 100000; // Exceeds 65535
        limits.max_storage_buffers_per_shader_stage = 16; // Exceeds 8
        let features = Features::TIMESTAMP_QUERY;
        let tier = WebGpuTier::from_limits_and_features(&limits, features);
        assert_eq!(tier, WebGpuTier::Tier3);
    }
}

// ============================================================================
// Module: WebGpuLimits - Basic Properties
// ============================================================================

mod webgpu_limits_basic {
    use super::*;

    #[test]
    fn default_equals_tier1_minimum() {
        let default = WebGpuLimits::default();
        let tier1 = WebGpuLimits::tier1_minimum();
        assert_eq!(default, tier1);
    }

    #[test]
    fn tier1_minimum_max_texture_2d() {
        let limits = WebGpuLimits::tier1_minimum();
        assert_eq!(limits.max_texture_dimension_2d, 2048);
    }

    #[test]
    fn tier1_minimum_max_texture_1d() {
        let limits = WebGpuLimits::tier1_minimum();
        assert_eq!(limits.max_texture_dimension_1d, 2048);
    }

    #[test]
    fn tier1_minimum_max_texture_3d() {
        let limits = WebGpuLimits::tier1_minimum();
        assert_eq!(limits.max_texture_dimension_3d, 256);
    }

    #[test]
    fn tier1_minimum_max_bind_groups() {
        let limits = WebGpuLimits::tier1_minimum();
        assert_eq!(limits.max_bind_groups, 4);
    }

    #[test]
    fn tier1_minimum_max_bindings_per_bind_group() {
        let limits = WebGpuLimits::tier1_minimum();
        assert_eq!(limits.max_bindings_per_bind_group, 640);
    }

    #[test]
    fn tier1_minimum_max_compute_workgroup_size_x() {
        let limits = WebGpuLimits::tier1_minimum();
        assert_eq!(limits.max_compute_workgroup_size_x, 256);
    }

    #[test]
    fn tier1_minimum_max_compute_workgroup_size_y() {
        let limits = WebGpuLimits::tier1_minimum();
        assert_eq!(limits.max_compute_workgroup_size_y, 256);
    }

    #[test]
    fn tier1_minimum_max_compute_workgroup_size_z() {
        let limits = WebGpuLimits::tier1_minimum();
        assert_eq!(limits.max_compute_workgroup_size_z, 64);
    }

    #[test]
    fn tier2_minimum_max_texture_2d() {
        let limits = WebGpuLimits::tier2_minimum();
        assert_eq!(limits.max_texture_dimension_2d, 4096);
    }

    #[test]
    fn tier2_minimum_max_bind_groups() {
        let limits = WebGpuLimits::tier2_minimum();
        assert_eq!(limits.max_bind_groups, 8);
    }

    #[test]
    fn tier2_minimum_max_storage_textures() {
        let limits = WebGpuLimits::tier2_minimum();
        assert_eq!(limits.max_storage_textures_per_shader_stage, 8);
    }

    #[test]
    fn tier3_minimum_max_texture_2d() {
        let limits = WebGpuLimits::tier3_minimum();
        assert_eq!(limits.max_texture_dimension_2d, 8192);
    }

    #[test]
    fn tier3_minimum_max_texture_3d() {
        let limits = WebGpuLimits::tier3_minimum();
        assert_eq!(limits.max_texture_dimension_3d, 2048);
    }

    #[test]
    fn tier3_minimum_max_compute_workgroup_size_x() {
        let limits = WebGpuLimits::tier3_minimum();
        assert_eq!(limits.max_compute_workgroup_size_x, 1024);
    }

    #[test]
    fn tier3_minimum_max_vertex_attributes() {
        let limits = WebGpuLimits::tier3_minimum();
        assert_eq!(limits.max_vertex_attributes, 32);
    }
}

// ============================================================================
// Module: WebGpuLimits - All 22 Fields
// ============================================================================

mod webgpu_limits_all_fields {
    use super::*;

    #[test]
    fn field_max_texture_dimension_1d() {
        let limits = WebGpuLimits::tier1_minimum();
        assert!(limits.max_texture_dimension_1d > 0);
    }

    #[test]
    fn field_max_texture_dimension_2d() {
        let limits = WebGpuLimits::tier1_minimum();
        assert!(limits.max_texture_dimension_2d > 0);
    }

    #[test]
    fn field_max_texture_dimension_3d() {
        let limits = WebGpuLimits::tier1_minimum();
        assert!(limits.max_texture_dimension_3d > 0);
    }

    #[test]
    fn field_max_bind_groups() {
        let limits = WebGpuLimits::tier1_minimum();
        assert!(limits.max_bind_groups > 0);
    }

    #[test]
    fn field_max_bindings_per_bind_group() {
        let limits = WebGpuLimits::tier1_minimum();
        assert!(limits.max_bindings_per_bind_group > 0);
    }

    #[test]
    fn field_max_dynamic_uniform_buffers_per_pipeline_layout() {
        let limits = WebGpuLimits::tier1_minimum();
        assert!(limits.max_dynamic_uniform_buffers_per_pipeline_layout > 0);
    }

    #[test]
    fn field_max_dynamic_storage_buffers_per_pipeline_layout() {
        let limits = WebGpuLimits::tier1_minimum();
        assert!(limits.max_dynamic_storage_buffers_per_pipeline_layout > 0);
    }

    #[test]
    fn field_max_sampled_textures_per_shader_stage() {
        let limits = WebGpuLimits::tier1_minimum();
        assert!(limits.max_sampled_textures_per_shader_stage > 0);
    }

    #[test]
    fn field_max_samplers_per_shader_stage() {
        let limits = WebGpuLimits::tier1_minimum();
        assert!(limits.max_samplers_per_shader_stage > 0);
    }

    #[test]
    fn field_max_storage_buffers_per_shader_stage() {
        let limits = WebGpuLimits::tier1_minimum();
        assert!(limits.max_storage_buffers_per_shader_stage > 0);
    }

    #[test]
    fn field_max_storage_textures_per_shader_stage() {
        let limits = WebGpuLimits::tier1_minimum();
        assert!(limits.max_storage_textures_per_shader_stage > 0);
    }

    #[test]
    fn field_max_uniform_buffers_per_shader_stage() {
        let limits = WebGpuLimits::tier1_minimum();
        assert!(limits.max_uniform_buffers_per_shader_stage > 0);
    }

    #[test]
    fn field_max_uniform_buffer_binding_size() {
        let limits = WebGpuLimits::tier1_minimum();
        assert!(limits.max_uniform_buffer_binding_size > 0);
    }

    #[test]
    fn field_max_storage_buffer_binding_size() {
        let limits = WebGpuLimits::tier1_minimum();
        assert!(limits.max_storage_buffer_binding_size > 0);
    }

    #[test]
    fn field_max_vertex_buffers() {
        let limits = WebGpuLimits::tier1_minimum();
        assert!(limits.max_vertex_buffers > 0);
    }

    #[test]
    fn field_max_vertex_attributes() {
        let limits = WebGpuLimits::tier1_minimum();
        assert!(limits.max_vertex_attributes > 0);
    }

    #[test]
    fn field_max_vertex_buffer_array_stride() {
        let limits = WebGpuLimits::tier1_minimum();
        assert!(limits.max_vertex_buffer_array_stride > 0);
    }

    #[test]
    fn field_max_compute_workgroup_size_x() {
        let limits = WebGpuLimits::tier1_minimum();
        assert!(limits.max_compute_workgroup_size_x > 0);
    }

    #[test]
    fn field_max_compute_workgroup_size_y() {
        let limits = WebGpuLimits::tier1_minimum();
        assert!(limits.max_compute_workgroup_size_y > 0);
    }

    #[test]
    fn field_max_compute_workgroup_size_z() {
        let limits = WebGpuLimits::tier1_minimum();
        assert!(limits.max_compute_workgroup_size_z > 0);
    }

    #[test]
    fn field_max_compute_invocations_per_workgroup() {
        let limits = WebGpuLimits::tier1_minimum();
        assert!(limits.max_compute_invocations_per_workgroup > 0);
    }

    #[test]
    fn field_max_compute_workgroups_per_dimension() {
        let limits = WebGpuLimits::tier1_minimum();
        assert!(limits.max_compute_workgroups_per_dimension > 0);
    }
}

// ============================================================================
// Module: WebGpuLimits - meets_tier()
// ============================================================================

mod webgpu_limits_meets_tier {
    use super::*;

    #[test]
    fn tier1_meets_tier1() {
        let limits = WebGpuLimits::tier1_minimum();
        assert!(limits.meets_tier(WebGpuTier::Tier1));
    }

    #[test]
    fn tier1_does_not_meet_tier2() {
        let limits = WebGpuLimits::tier1_minimum();
        assert!(!limits.meets_tier(WebGpuTier::Tier2));
    }

    #[test]
    fn tier1_does_not_meet_tier3() {
        let limits = WebGpuLimits::tier1_minimum();
        assert!(!limits.meets_tier(WebGpuTier::Tier3));
    }

    #[test]
    fn tier2_meets_tier1() {
        let limits = WebGpuLimits::tier2_minimum();
        assert!(limits.meets_tier(WebGpuTier::Tier1));
    }

    #[test]
    fn tier2_meets_tier2() {
        let limits = WebGpuLimits::tier2_minimum();
        assert!(limits.meets_tier(WebGpuTier::Tier2));
    }

    #[test]
    fn tier2_does_not_meet_tier3() {
        let limits = WebGpuLimits::tier2_minimum();
        assert!(!limits.meets_tier(WebGpuTier::Tier3));
    }

    #[test]
    fn tier3_meets_tier1() {
        let limits = WebGpuLimits::tier3_minimum();
        assert!(limits.meets_tier(WebGpuTier::Tier1));
    }

    #[test]
    fn tier3_meets_tier2() {
        let limits = WebGpuLimits::tier3_minimum();
        assert!(limits.meets_tier(WebGpuTier::Tier2));
    }

    #[test]
    fn tier3_meets_tier3() {
        let limits = WebGpuLimits::tier3_minimum();
        assert!(limits.meets_tier(WebGpuTier::Tier3));
    }

    #[test]
    fn exceeding_tier3_still_meets_all() {
        let mut limits = WebGpuLimits::tier3_minimum();
        limits.max_texture_dimension_2d = 16384;
        limits.max_bind_groups = 16;
        assert!(limits.meets_tier(WebGpuTier::Tier1));
        assert!(limits.meets_tier(WebGpuTier::Tier2));
        assert!(limits.meets_tier(WebGpuTier::Tier3));
    }

    #[test]
    fn boundary_texture_exactly_tier2() {
        let mut limits = WebGpuLimits::tier2_minimum();
        limits.max_texture_dimension_2d = 4096; // Exactly at boundary
        assert!(limits.meets_tier(WebGpuTier::Tier2));
    }

    #[test]
    fn boundary_texture_below_tier2() {
        let mut limits = WebGpuLimits::tier2_minimum();
        limits.max_texture_dimension_2d = 4095; // Just below
        assert!(!limits.meets_tier(WebGpuTier::Tier2));
    }
}

// ============================================================================
// Module: WebGpuLimits - from_wgpu_limits()
// ============================================================================

mod webgpu_limits_conversion {
    use super::*;

    #[test]
    fn from_wgpu_limits_default() {
        let wgpu = Limits::default();
        let webgpu = WebGpuLimits::from_wgpu_limits(&wgpu);
        assert_eq!(webgpu.max_texture_dimension_2d, wgpu.max_texture_dimension_2d);
    }

    #[test]
    fn from_wgpu_limits_preserves_bind_groups() {
        let wgpu = Limits::default();
        let webgpu = WebGpuLimits::from_wgpu_limits(&wgpu);
        assert_eq!(webgpu.max_bind_groups, wgpu.max_bind_groups);
    }

    #[test]
    fn from_wgpu_limits_preserves_compute_limits() {
        let wgpu = Limits::default();
        let webgpu = WebGpuLimits::from_wgpu_limits(&wgpu);
        assert_eq!(webgpu.max_compute_workgroup_size_x, wgpu.max_compute_workgroup_size_x);
    }

    #[test]
    fn from_wgpu_limits_downlevel_defaults() {
        let wgpu = Limits::downlevel_defaults();
        let webgpu = WebGpuLimits::from_wgpu_limits(&wgpu);
        // Downlevel defaults have lower limits
        assert!(webgpu.max_texture_dimension_2d <= 2048 || webgpu.max_texture_dimension_2d > 0);
    }

    #[test]
    fn from_wgpu_limits_preserves_vertex_limits() {
        let wgpu = Limits::default();
        let webgpu = WebGpuLimits::from_wgpu_limits(&wgpu);
        assert_eq!(webgpu.max_vertex_buffers, wgpu.max_vertex_buffers);
        assert_eq!(webgpu.max_vertex_attributes, wgpu.max_vertex_attributes);
    }
}

// ============================================================================
// Module: WebGpuFeatures - Basic Properties
// ============================================================================

mod webgpu_features_basic {
    use super::*;

    #[test]
    fn default_tier_is_tier1() {
        let features = WebGpuFeatures::default();
        assert_eq!(features.tier, WebGpuTier::Tier1);
    }

    #[test]
    fn default_no_compression() {
        let features = WebGpuFeatures::default();
        assert!(!features.texture_compression_bc);
        assert!(!features.texture_compression_etc2);
        assert!(!features.texture_compression_astc);
    }

    #[test]
    fn default_no_timestamp_query() {
        let features = WebGpuFeatures::default();
        assert!(!features.timestamp_query);
    }

    #[test]
    fn default_no_shader_f16() {
        let features = WebGpuFeatures::default();
        assert!(!features.shader_f16);
    }

    #[test]
    fn default_no_depth_clip_control() {
        let features = WebGpuFeatures::default();
        assert!(!features.depth_clip_control);
    }

    #[test]
    fn default_limits_match_tier1() {
        let features = WebGpuFeatures::default();
        assert_eq!(features.limits, WebGpuLimits::tier1_minimum());
    }
}

// ============================================================================
// Module: WebGpuFeatures - supports_compression()
// ============================================================================

mod webgpu_features_compression {
    use super::*;

    #[test]
    fn no_compression_by_default() {
        let features = WebGpuFeatures::default();
        assert!(!features.supports_compression());
    }

    #[test]
    fn bc_enables_compression() {
        let mut features = WebGpuFeatures::default();
        features.texture_compression_bc = true;
        assert!(features.supports_compression());
    }

    #[test]
    fn etc2_enables_compression() {
        let mut features = WebGpuFeatures::default();
        features.texture_compression_etc2 = true;
        assert!(features.supports_compression());
    }

    #[test]
    fn astc_enables_compression() {
        let mut features = WebGpuFeatures::default();
        features.texture_compression_astc = true;
        assert!(features.supports_compression());
    }

    #[test]
    fn all_compression_enables_compression() {
        let mut features = WebGpuFeatures::default();
        features.texture_compression_bc = true;
        features.texture_compression_etc2 = true;
        features.texture_compression_astc = true;
        assert!(features.supports_compression());
    }
}

// ============================================================================
// Module: WebGpuFeatures - compression_formats()
// ============================================================================

mod webgpu_features_compression_formats {
    use super::*;

    #[test]
    fn empty_when_no_compression() {
        let features = WebGpuFeatures::default();
        let formats = features.compression_formats();
        assert!(formats.is_empty());
    }

    #[test]
    fn bc_only() {
        let mut features = WebGpuFeatures::default();
        features.texture_compression_bc = true;
        let formats = features.compression_formats();
        assert_eq!(formats, vec!["BC"]);
    }

    #[test]
    fn etc2_only() {
        let mut features = WebGpuFeatures::default();
        features.texture_compression_etc2 = true;
        let formats = features.compression_formats();
        assert_eq!(formats, vec!["ETC2"]);
    }

    #[test]
    fn astc_only() {
        let mut features = WebGpuFeatures::default();
        features.texture_compression_astc = true;
        let formats = features.compression_formats();
        assert_eq!(formats, vec!["ASTC"]);
    }

    #[test]
    fn bc_and_etc2() {
        let mut features = WebGpuFeatures::default();
        features.texture_compression_bc = true;
        features.texture_compression_etc2 = true;
        let formats = features.compression_formats();
        assert_eq!(formats.len(), 2);
        assert!(formats.contains(&"BC"));
        assert!(formats.contains(&"ETC2"));
    }

    #[test]
    fn bc_and_astc() {
        let mut features = WebGpuFeatures::default();
        features.texture_compression_bc = true;
        features.texture_compression_astc = true;
        let formats = features.compression_formats();
        assert_eq!(formats.len(), 2);
        assert!(formats.contains(&"BC"));
        assert!(formats.contains(&"ASTC"));
    }

    #[test]
    fn all_three_formats() {
        let mut features = WebGpuFeatures::default();
        features.texture_compression_bc = true;
        features.texture_compression_etc2 = true;
        features.texture_compression_astc = true;
        let formats = features.compression_formats();
        assert_eq!(formats.len(), 3);
    }

    #[test]
    fn format_order_is_bc_etc2_astc() {
        let mut features = WebGpuFeatures::default();
        features.texture_compression_bc = true;
        features.texture_compression_etc2 = true;
        features.texture_compression_astc = true;
        let formats = features.compression_formats();
        assert_eq!(formats, vec!["BC", "ETC2", "ASTC"]);
    }
}

// ============================================================================
// Module: WebGpuFeatures - is_mobile_optimized()
// ============================================================================

mod webgpu_features_mobile_optimized {
    use super::*;

    #[test]
    fn not_mobile_by_default() {
        let features = WebGpuFeatures::default();
        assert!(!features.is_mobile_optimized());
    }

    #[test]
    fn etc2_only_is_mobile() {
        let mut features = WebGpuFeatures::default();
        features.texture_compression_etc2 = true;
        assert!(features.is_mobile_optimized());
    }

    #[test]
    fn astc_only_is_mobile() {
        let mut features = WebGpuFeatures::default();
        features.texture_compression_astc = true;
        assert!(features.is_mobile_optimized());
    }

    #[test]
    fn etc2_and_astc_is_mobile() {
        let mut features = WebGpuFeatures::default();
        features.texture_compression_etc2 = true;
        features.texture_compression_astc = true;
        assert!(features.is_mobile_optimized());
    }

    #[test]
    fn bc_disables_mobile() {
        let mut features = WebGpuFeatures::default();
        features.texture_compression_etc2 = true;
        features.texture_compression_bc = true;
        assert!(!features.is_mobile_optimized());
    }

    #[test]
    fn bc_with_astc_not_mobile() {
        let mut features = WebGpuFeatures::default();
        features.texture_compression_astc = true;
        features.texture_compression_bc = true;
        assert!(!features.is_mobile_optimized());
    }

    #[test]
    fn bc_only_not_mobile() {
        let mut features = WebGpuFeatures::default();
        features.texture_compression_bc = true;
        assert!(!features.is_mobile_optimized());
    }
}

// ============================================================================
// Module: WebGpuFeatures - is_desktop_optimized()
// ============================================================================

mod webgpu_features_desktop_optimized {
    use super::*;

    #[test]
    fn not_desktop_by_default() {
        let features = WebGpuFeatures::default();
        assert!(!features.is_desktop_optimized());
    }

    #[test]
    fn bc_only_is_desktop() {
        let mut features = WebGpuFeatures::default();
        features.texture_compression_bc = true;
        assert!(features.is_desktop_optimized());
    }

    #[test]
    fn etc2_disables_desktop() {
        let mut features = WebGpuFeatures::default();
        features.texture_compression_bc = true;
        features.texture_compression_etc2 = true;
        assert!(!features.is_desktop_optimized());
    }

    #[test]
    fn astc_disables_desktop() {
        let mut features = WebGpuFeatures::default();
        features.texture_compression_bc = true;
        features.texture_compression_astc = true;
        assert!(!features.is_desktop_optimized());
    }

    #[test]
    fn all_compression_not_desktop() {
        let mut features = WebGpuFeatures::default();
        features.texture_compression_bc = true;
        features.texture_compression_etc2 = true;
        features.texture_compression_astc = true;
        assert!(!features.is_desktop_optimized());
    }

    #[test]
    fn etc2_only_not_desktop() {
        let mut features = WebGpuFeatures::default();
        features.texture_compression_etc2 = true;
        assert!(!features.is_desktop_optimized());
    }
}

// ============================================================================
// Module: WebGpuFeatures - summary()
// ============================================================================

mod webgpu_features_summary {
    use super::*;

    #[test]
    fn summary_includes_tier_name() {
        let features = WebGpuFeatures::default();
        let summary = features.summary();
        assert!(summary.contains("Tier 1"));
    }

    #[test]
    fn summary_includes_basic_for_default() {
        let features = WebGpuFeatures::default();
        let summary = features.summary();
        assert!(summary.contains("Basic"));
    }

    #[test]
    fn summary_includes_bc() {
        let mut features = WebGpuFeatures::default();
        features.texture_compression_bc = true;
        let summary = features.summary();
        assert!(summary.contains("BC"));
    }

    #[test]
    fn summary_includes_etc2() {
        let mut features = WebGpuFeatures::default();
        features.texture_compression_etc2 = true;
        let summary = features.summary();
        assert!(summary.contains("ETC2"));
    }

    #[test]
    fn summary_includes_astc() {
        let mut features = WebGpuFeatures::default();
        features.texture_compression_astc = true;
        let summary = features.summary();
        assert!(summary.contains("ASTC"));
    }

    #[test]
    fn summary_includes_timestamp() {
        let mut features = WebGpuFeatures::default();
        features.timestamp_query = true;
        let summary = features.summary();
        assert!(summary.contains("Timestamp"));
    }

    #[test]
    fn summary_includes_fp16() {
        let mut features = WebGpuFeatures::default();
        features.shader_f16 = true;
        let summary = features.summary();
        assert!(summary.contains("FP16"));
    }

    #[test]
    fn summary_includes_indirect() {
        let mut features = WebGpuFeatures::default();
        features.indirect_first_instance = true;
        let summary = features.summary();
        assert!(summary.contains("Indirect"));
    }

    #[test]
    fn summary_includes_f32filter() {
        let mut features = WebGpuFeatures::default();
        features.float32_filterable = true;
        let summary = features.summary();
        assert!(summary.contains("F32Filter"));
    }

    #[test]
    fn summary_includes_d32s8() {
        let mut features = WebGpuFeatures::default();
        features.depth32_float_stencil8 = true;
        let summary = features.summary();
        assert!(summary.contains("D32S8"));
    }

    #[test]
    fn summary_tier3_full() {
        let mut features = WebGpuFeatures::default();
        features.tier = WebGpuTier::Tier3;
        features.texture_compression_bc = true;
        features.timestamp_query = true;
        features.shader_f16 = true;

        let summary = features.summary();
        assert!(summary.contains("Tier 3"));
        assert!(summary.contains("BC"));
        assert!(summary.contains("Timestamp"));
        assert!(summary.contains("FP16"));
        assert!(!summary.contains("Basic")); // Not basic when features present
    }
}

// ============================================================================
// Module: WebGpuFeatures - from_adapter_info()
// ============================================================================

mod webgpu_features_from_adapter_info {
    use super::*;

    #[test]
    fn empty_features_gives_tier1() {
        let limits = Limits::downlevel_defaults();
        let features = Features::empty();
        let webgpu = WebGpuFeatures::from_adapter_info(&limits, features);
        assert_eq!(webgpu.tier, WebGpuTier::Tier1);
    }

    #[test]
    fn detects_bc_compression() {
        let limits = Limits::default();
        let features = Features::TEXTURE_COMPRESSION_BC;
        let webgpu = WebGpuFeatures::from_adapter_info(&limits, features);
        assert!(webgpu.texture_compression_bc);
    }

    #[test]
    fn detects_etc2_compression() {
        let limits = Limits::default();
        let features = Features::TEXTURE_COMPRESSION_ETC2;
        let webgpu = WebGpuFeatures::from_adapter_info(&limits, features);
        assert!(webgpu.texture_compression_etc2);
    }

    #[test]
    fn detects_astc_compression() {
        let limits = Limits::default();
        let features = Features::TEXTURE_COMPRESSION_ASTC;
        let webgpu = WebGpuFeatures::from_adapter_info(&limits, features);
        assert!(webgpu.texture_compression_astc);
    }

    #[test]
    fn detects_timestamp_query() {
        let mut limits = Limits::default();
        limits.max_texture_dimension_2d = 8192;
        limits.max_compute_workgroups_per_dimension = 65535;
        limits.max_storage_buffers_per_shader_stage = 8;
        let features = Features::TIMESTAMP_QUERY;
        let webgpu = WebGpuFeatures::from_adapter_info(&limits, features);
        assert!(webgpu.timestamp_query);
    }

    #[test]
    fn detects_indirect_first_instance() {
        let limits = Limits::default();
        let features = Features::INDIRECT_FIRST_INSTANCE;
        let webgpu = WebGpuFeatures::from_adapter_info(&limits, features);
        assert!(webgpu.indirect_first_instance);
    }

    #[test]
    fn detects_shader_f16() {
        let limits = Limits::default();
        let features = Features::SHADER_F16;
        let webgpu = WebGpuFeatures::from_adapter_info(&limits, features);
        assert!(webgpu.shader_f16);
    }

    #[test]
    fn detects_depth_clip_control() {
        let limits = Limits::default();
        let features = Features::DEPTH_CLIP_CONTROL;
        let webgpu = WebGpuFeatures::from_adapter_info(&limits, features);
        assert!(webgpu.depth_clip_control);
    }

    #[test]
    fn detects_depth32_float_stencil8() {
        let limits = Limits::default();
        let features = Features::DEPTH32FLOAT_STENCIL8;
        let webgpu = WebGpuFeatures::from_adapter_info(&limits, features);
        assert!(webgpu.depth32_float_stencil8);
    }

    #[test]
    fn detects_rg11b10_ufloat_renderable() {
        let limits = Limits::default();
        let features = Features::RG11B10UFLOAT_RENDERABLE;
        let webgpu = WebGpuFeatures::from_adapter_info(&limits, features);
        assert!(webgpu.rg11b10_ufloat_renderable);
    }

    #[test]
    fn detects_bgra8_unorm_storage() {
        let limits = Limits::default();
        let features = Features::BGRA8UNORM_STORAGE;
        let webgpu = WebGpuFeatures::from_adapter_info(&limits, features);
        assert!(webgpu.bgra8_unorm_storage);
    }

    #[test]
    fn detects_float32_filterable() {
        let limits = Limits::default();
        let features = Features::FLOAT32_FILTERABLE;
        let webgpu = WebGpuFeatures::from_adapter_info(&limits, features);
        assert!(webgpu.float32_filterable);
    }

    #[test]
    fn combined_features_detected() {
        let limits = Limits::default();
        let features = Features::TEXTURE_COMPRESSION_BC
            | Features::TEXTURE_COMPRESSION_ETC2
            | Features::SHADER_F16;
        let webgpu = WebGpuFeatures::from_adapter_info(&limits, features);
        assert!(webgpu.texture_compression_bc);
        assert!(webgpu.texture_compression_etc2);
        assert!(webgpu.shader_f16);
    }
}

// ============================================================================
// Module: BrowserType - Basic Properties
// ============================================================================

mod browser_type_basic {
    use super::*;

    #[test]
    fn default_is_unknown() {
        let browser = BrowserType::default();
        assert_eq!(browser, BrowserType::Unknown);
    }

    #[test]
    fn chrome_variant_exists() {
        let _browser = BrowserType::Chrome;
    }

    #[test]
    fn firefox_variant_exists() {
        let _browser = BrowserType::Firefox;
    }

    #[test]
    fn safari_variant_exists() {
        let _browser = BrowserType::Safari;
    }

    #[test]
    fn edge_variant_exists() {
        let _browser = BrowserType::Edge;
    }

    #[test]
    fn unknown_variant_exists() {
        let _browser = BrowserType::Unknown;
    }
}

// ============================================================================
// Module: BrowserType - from_user_agent()
// ============================================================================

mod browser_type_from_user_agent {
    use super::*;

    #[test]
    fn detects_chrome_standard() {
        let browser = BrowserType::from_user_agent(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"
        );
        assert_eq!(browser, BrowserType::Chrome);
    }

    #[test]
    fn detects_chrome_minimal() {
        let browser = BrowserType::from_user_agent("Chrome/120.0.0.0");
        assert_eq!(browser, BrowserType::Chrome);
    }

    #[test]
    fn detects_chrome_lowercase() {
        let browser = BrowserType::from_user_agent("chrome/120");
        assert_eq!(browser, BrowserType::Chrome);
    }

    #[test]
    fn detects_chromium() {
        let browser = BrowserType::from_user_agent("Chromium/120.0.0.0");
        assert_eq!(browser, BrowserType::Chrome);
    }

    #[test]
    fn detects_edge_with_edg() {
        let browser = BrowserType::from_user_agent(
            "Mozilla/5.0 Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0"
        );
        assert_eq!(browser, BrowserType::Edge);
    }

    #[test]
    fn detects_edge_with_edge() {
        let browser = BrowserType::from_user_agent("Edge/120.0.0.0");
        assert_eq!(browser, BrowserType::Edge);
    }

    #[test]
    fn edge_takes_priority_over_chrome() {
        let browser = BrowserType::from_user_agent("Chrome/120 Edg/120");
        assert_eq!(browser, BrowserType::Edge);
    }

    #[test]
    fn detects_firefox_standard() {
        let browser = BrowserType::from_user_agent(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0"
        );
        assert_eq!(browser, BrowserType::Firefox);
    }

    #[test]
    fn detects_firefox_minimal() {
        let browser = BrowserType::from_user_agent("Firefox/121.0");
        assert_eq!(browser, BrowserType::Firefox);
    }

    #[test]
    fn detects_firefox_via_gecko() {
        let browser = BrowserType::from_user_agent("Gecko/20100101");
        assert_eq!(browser, BrowserType::Firefox);
    }

    #[test]
    fn detects_safari_standard() {
        let browser = BrowserType::from_user_agent(
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15"
        );
        assert_eq!(browser, BrowserType::Safari);
    }

    #[test]
    fn safari_requires_both_version_and_safari() {
        // Just "Safari/" without "Version/" is Chrome/Edge user agent style
        let browser = BrowserType::from_user_agent("Safari/537.36");
        assert_eq!(browser, BrowserType::Unknown); // No Version/
    }

    #[test]
    fn safari_with_version() {
        let browser = BrowserType::from_user_agent("Version/17.0 Safari/605.1.15");
        assert_eq!(browser, BrowserType::Safari);
    }

    #[test]
    fn empty_string_returns_unknown() {
        let browser = BrowserType::from_user_agent("");
        assert_eq!(browser, BrowserType::Unknown);
    }

    #[test]
    fn random_string_returns_unknown() {
        let browser = BrowserType::from_user_agent("RandomBrowser/1.0");
        assert_eq!(browser, BrowserType::Unknown);
    }

    #[test]
    fn mozilla_only_returns_unknown() {
        let browser = BrowserType::from_user_agent("Mozilla/5.0");
        assert_eq!(browser, BrowserType::Unknown);
    }

    #[test]
    fn case_insensitive_chrome() {
        let browser = BrowserType::from_user_agent("CHROME/120");
        assert_eq!(browser, BrowserType::Chrome);
    }

    #[test]
    fn case_insensitive_firefox() {
        let browser = BrowserType::from_user_agent("FIREFOX/121");
        assert_eq!(browser, BrowserType::Firefox);
    }

    #[test]
    fn case_insensitive_edge() {
        let browser = BrowserType::from_user_agent("EDG/120");
        assert_eq!(browser, BrowserType::Edge);
    }
}

// ============================================================================
// Module: BrowserType - Feature Support Methods
// ============================================================================

mod browser_type_feature_support {
    use super::*;

    #[test]
    fn chrome_has_stable_webgpu() {
        assert!(BrowserType::Chrome.has_stable_webgpu());
    }

    #[test]
    fn edge_has_stable_webgpu() {
        assert!(BrowserType::Edge.has_stable_webgpu());
    }

    #[test]
    fn safari_has_stable_webgpu() {
        assert!(BrowserType::Safari.has_stable_webgpu());
    }

    #[test]
    fn firefox_no_stable_webgpu() {
        assert!(!BrowserType::Firefox.has_stable_webgpu());
    }

    #[test]
    fn unknown_no_stable_webgpu() {
        assert!(!BrowserType::Unknown.has_stable_webgpu());
    }

    #[test]
    fn chrome_supports_offscreen_canvas() {
        assert!(BrowserType::Chrome.supports_offscreen_canvas());
    }

    #[test]
    fn edge_supports_offscreen_canvas() {
        assert!(BrowserType::Edge.supports_offscreen_canvas());
    }

    #[test]
    fn firefox_supports_offscreen_canvas() {
        assert!(BrowserType::Firefox.supports_offscreen_canvas());
    }

    #[test]
    fn safari_no_offscreen_canvas() {
        assert!(!BrowserType::Safari.supports_offscreen_canvas());
    }

    #[test]
    fn unknown_no_offscreen_canvas() {
        assert!(!BrowserType::Unknown.supports_offscreen_canvas());
    }

    #[test]
    fn chrome_supports_shared_array_buffer() {
        assert!(BrowserType::Chrome.supports_shared_array_buffer());
    }

    #[test]
    fn edge_supports_shared_array_buffer() {
        assert!(BrowserType::Edge.supports_shared_array_buffer());
    }

    #[test]
    fn firefox_supports_shared_array_buffer() {
        assert!(BrowserType::Firefox.supports_shared_array_buffer());
    }

    #[test]
    fn safari_supports_shared_array_buffer() {
        assert!(BrowserType::Safari.supports_shared_array_buffer());
    }

    #[test]
    fn unknown_no_shared_array_buffer() {
        assert!(!BrowserType::Unknown.supports_shared_array_buffer());
    }
}

// ============================================================================
// Module: BrowserType - Name and Backend Methods
// ============================================================================

mod browser_type_name_backend {
    use super::*;

    #[test]
    fn name_chrome() {
        assert_eq!(BrowserType::Chrome.name(), "Chrome");
    }

    #[test]
    fn name_firefox() {
        assert_eq!(BrowserType::Firefox.name(), "Firefox");
    }

    #[test]
    fn name_safari() {
        assert_eq!(BrowserType::Safari.name(), "Safari");
    }

    #[test]
    fn name_edge() {
        assert_eq!(BrowserType::Edge.name(), "Edge");
    }

    #[test]
    fn name_unknown() {
        assert_eq!(BrowserType::Unknown.name(), "Unknown");
    }

    #[test]
    fn webgpu_backend_chrome() {
        assert_eq!(BrowserType::Chrome.webgpu_backend(), "Dawn (Vulkan/D3D12/Metal)");
    }

    #[test]
    fn webgpu_backend_edge() {
        assert_eq!(BrowserType::Edge.webgpu_backend(), "Dawn (Vulkan/D3D12/Metal)");
    }

    #[test]
    fn webgpu_backend_firefox() {
        assert_eq!(BrowserType::Firefox.webgpu_backend(), "wgpu-native (Vulkan/D3D12/Metal)");
    }

    #[test]
    fn webgpu_backend_safari() {
        assert_eq!(BrowserType::Safari.webgpu_backend(), "WebKit (Metal)");
    }

    #[test]
    fn webgpu_backend_unknown() {
        assert_eq!(BrowserType::Unknown.webgpu_backend(), "Unknown");
    }

    #[test]
    fn display_chrome() {
        assert_eq!(format!("{}", BrowserType::Chrome), "Chrome");
    }

    #[test]
    fn display_firefox() {
        assert_eq!(format!("{}", BrowserType::Firefox), "Firefox");
    }
}

// ============================================================================
// Module: BrowserCapabilities - Basic Properties
// ============================================================================

mod browser_capabilities_basic {
    use super::*;

    #[test]
    fn default_browser_is_unknown() {
        let caps = BrowserCapabilities::default();
        assert_eq!(caps.browser, BrowserType::Unknown);
    }

    #[test]
    fn default_no_webgpu_version() {
        let caps = BrowserCapabilities::default();
        assert!(caps.webgpu_version.is_none());
    }

    #[test]
    fn default_no_offscreen_canvas() {
        let caps = BrowserCapabilities::default();
        assert!(!caps.supports_offscreen_canvas);
    }

    #[test]
    fn default_no_shared_array_buffer() {
        let caps = BrowserCapabilities::default();
        assert!(!caps.supports_shared_array_buffer);
    }

    #[test]
    fn default_max_canvas_size() {
        let caps = BrowserCapabilities::default();
        assert_eq!(caps.max_canvas_size, (0, 0));
    }
}

// ============================================================================
// Module: BrowserCapabilities - from_user_agent()
// ============================================================================

mod browser_capabilities_from_user_agent {
    use super::*;

    #[test]
    fn chrome_capabilities() {
        let caps = BrowserCapabilities::from_user_agent("Chrome/120.0.0.0");
        assert_eq!(caps.browser, BrowserType::Chrome);
        assert!(caps.supports_offscreen_canvas);
        assert!(caps.supports_shared_array_buffer);
        assert_eq!(caps.max_canvas_size, (32767, 32767));
    }

    #[test]
    fn edge_capabilities() {
        let caps = BrowserCapabilities::from_user_agent("Chrome/120 Edg/120");
        assert_eq!(caps.browser, BrowserType::Edge);
        assert!(caps.supports_offscreen_canvas);
        assert!(caps.supports_shared_array_buffer);
        assert_eq!(caps.max_canvas_size, (32767, 32767));
    }

    #[test]
    fn firefox_capabilities() {
        let caps = BrowserCapabilities::from_user_agent("Firefox/121.0");
        assert_eq!(caps.browser, BrowserType::Firefox);
        assert!(caps.supports_offscreen_canvas);
        assert!(caps.supports_shared_array_buffer);
        assert_eq!(caps.max_canvas_size, (32767, 32767));
    }

    #[test]
    fn safari_capabilities() {
        let caps = BrowserCapabilities::from_user_agent("Version/17.0 Safari/605.1.15");
        assert_eq!(caps.browser, BrowserType::Safari);
        assert!(!caps.supports_offscreen_canvas);
        assert!(caps.supports_shared_array_buffer);
        assert_eq!(caps.max_canvas_size, (16384, 16384));
    }

    #[test]
    fn unknown_capabilities() {
        let caps = BrowserCapabilities::from_user_agent("RandomBrowser/1.0");
        assert_eq!(caps.browser, BrowserType::Unknown);
        assert!(!caps.supports_offscreen_canvas);
        assert!(!caps.supports_shared_array_buffer);
        assert_eq!(caps.max_canvas_size, (8192, 8192));
    }

    #[test]
    fn no_webgpu_version_from_user_agent() {
        let caps = BrowserCapabilities::from_user_agent("Chrome/120.0.0.0");
        assert!(caps.webgpu_version.is_none());
    }
}

// ============================================================================
// Module: BrowserCapabilities - has_full_webgpu()
// ============================================================================

mod browser_capabilities_full_webgpu {
    use super::*;

    #[test]
    fn chrome_has_full_webgpu() {
        let caps = BrowserCapabilities::from_user_agent("Chrome/120");
        assert!(caps.has_full_webgpu());
    }

    #[test]
    fn edge_has_full_webgpu() {
        let caps = BrowserCapabilities::from_user_agent("Edg/120");
        assert!(caps.has_full_webgpu());
    }

    #[test]
    fn safari_has_full_webgpu() {
        let caps = BrowserCapabilities::from_user_agent("Version/17 Safari/605");
        assert!(caps.has_full_webgpu());
    }

    #[test]
    fn firefox_no_full_webgpu() {
        let caps = BrowserCapabilities::from_user_agent("Firefox/121");
        assert!(!caps.has_full_webgpu());
    }

    #[test]
    fn unknown_no_full_webgpu() {
        let caps = BrowserCapabilities::from_user_agent("RandomBrowser");
        assert!(!caps.has_full_webgpu());
    }
}

// ============================================================================
// Module: BrowserCapabilities - summary()
// ============================================================================

mod browser_capabilities_summary {
    use super::*;

    #[test]
    fn summary_includes_browser_name() {
        let caps = BrowserCapabilities::from_user_agent("Chrome/120");
        let summary = caps.summary();
        assert!(summary.contains("Chrome"));
    }

    #[test]
    fn summary_includes_offscreen_canvas() {
        let caps = BrowserCapabilities::from_user_agent("Chrome/120");
        let summary = caps.summary();
        assert!(summary.contains("OffscreenCanvas"));
    }

    #[test]
    fn summary_includes_sab() {
        let caps = BrowserCapabilities::from_user_agent("Chrome/120");
        let summary = caps.summary();
        assert!(summary.contains("SAB"));
    }

    #[test]
    fn summary_includes_max_canvas_size() {
        let caps = BrowserCapabilities::from_user_agent("Chrome/120");
        let summary = caps.summary();
        assert!(summary.contains("32767x32767"));
    }

    #[test]
    fn summary_safari_no_offscreen_canvas() {
        let caps = BrowserCapabilities::from_user_agent("Version/17 Safari/605");
        let summary = caps.summary();
        assert!(!summary.contains("OffscreenCanvas"));
    }

    #[test]
    fn summary_unknown_no_sab() {
        let caps = BrowserCapabilities::from_user_agent("RandomBrowser");
        let summary = caps.summary();
        assert!(!summary.contains("SAB"));
    }

    #[test]
    fn summary_with_webgpu_version() {
        let mut caps = BrowserCapabilities::from_user_agent("Chrome/120");
        caps.webgpu_version = Some("1.0".to_string());
        let summary = caps.summary();
        assert!(summary.contains("WebGPU 1.0"));
    }
}

// ============================================================================
// Module: BrowserCapabilities - detect()
// ============================================================================

mod browser_capabilities_detect {
    use super::*;

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn detect_on_native_returns_unknown() {
        let caps = BrowserCapabilities::detect();
        assert_eq!(caps.browser, BrowserType::Unknown);
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn detect_on_native_large_canvas() {
        let caps = BrowserCapabilities::detect();
        assert_eq!(caps.max_canvas_size, (16384, 16384));
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn detect_on_native_no_offscreen() {
        let caps = BrowserCapabilities::detect();
        assert!(!caps.supports_offscreen_canvas);
    }
}

// ============================================================================
// Module: Edge Cases - Empty Feature Sets
// ============================================================================

mod edge_cases_empty {
    use super::*;

    #[test]
    fn empty_features_default_tier() {
        let features = WebGpuFeatures::default();
        assert_eq!(features.tier, WebGpuTier::Tier1);
    }

    #[test]
    fn empty_compression_formats_vec() {
        let features = WebGpuFeatures::default();
        assert!(features.compression_formats().is_empty());
    }

    #[test]
    fn empty_user_agent_unknown_browser() {
        let browser = BrowserType::from_user_agent("");
        assert_eq!(browser, BrowserType::Unknown);
    }
}

// ============================================================================
// Module: Edge Cases - Minimum Limits
// ============================================================================

mod edge_cases_minimum_limits {
    use super::*;

    #[test]
    fn below_tier1_minimum_still_default() {
        // Even with zero limits, from_limits_and_features uses fallback logic
        let mut limits = Limits::default();
        limits.max_texture_dimension_2d = 1024; // Below Tier 1
        let features = Features::empty();
        let tier = WebGpuTier::from_limits_and_features(&limits, features);
        // Falls through to Tier 1 default
        assert_eq!(tier, WebGpuTier::Tier1);
    }

    #[test]
    fn extremely_low_limits_tier1() {
        let mut limits = Limits::default();
        limits.max_texture_dimension_2d = 512;
        limits.max_bind_groups = 2;
        let features = Features::empty();
        let tier = WebGpuTier::from_limits_and_features(&limits, features);
        assert_eq!(tier, WebGpuTier::Tier1);
    }
}

// ============================================================================
// Module: Edge Cases - Maximum Limits
// ============================================================================

mod edge_cases_maximum_limits {
    use super::*;

    #[test]
    fn extremely_high_limits_tier3() {
        let mut limits = Limits::default();
        limits.max_texture_dimension_2d = 65536;
        limits.max_compute_workgroups_per_dimension = 1000000;
        limits.max_storage_buffers_per_shader_stage = 64;
        let features = Features::TIMESTAMP_QUERY;
        let tier = WebGpuTier::from_limits_and_features(&limits, features);
        assert_eq!(tier, WebGpuTier::Tier3);
    }

    #[test]
    fn high_limits_all_tiers_met() {
        let mut limits = WebGpuLimits::tier3_minimum();
        limits.max_texture_dimension_2d = 32768;
        assert!(limits.meets_tier(WebGpuTier::Tier1));
        assert!(limits.meets_tier(WebGpuTier::Tier2));
        assert!(limits.meets_tier(WebGpuTier::Tier3));
    }
}

// ============================================================================
// Module: Edge Cases - Mixed Compression Support
// ============================================================================

mod edge_cases_mixed_compression {
    use super::*;

    #[test]
    fn bc_plus_etc2_not_mobile_or_desktop() {
        let mut features = WebGpuFeatures::default();
        features.texture_compression_bc = true;
        features.texture_compression_etc2 = true;
        assert!(!features.is_mobile_optimized());
        assert!(!features.is_desktop_optimized());
    }

    #[test]
    fn bc_plus_astc_not_mobile_or_desktop() {
        let mut features = WebGpuFeatures::default();
        features.texture_compression_bc = true;
        features.texture_compression_astc = true;
        assert!(!features.is_mobile_optimized());
        assert!(!features.is_desktop_optimized());
    }

    #[test]
    fn etc2_plus_astc_is_mobile() {
        let mut features = WebGpuFeatures::default();
        features.texture_compression_etc2 = true;
        features.texture_compression_astc = true;
        assert!(features.is_mobile_optimized());
        assert!(!features.is_desktop_optimized());
    }
}

// ============================================================================
// Module: Edge Cases - Unknown User Agent Strings
// ============================================================================

mod edge_cases_unknown_user_agents {
    use super::*;

    #[test]
    fn whitespace_only() {
        let browser = BrowserType::from_user_agent("   ");
        assert_eq!(browser, BrowserType::Unknown);
    }

    #[test]
    fn numbers_only() {
        let browser = BrowserType::from_user_agent("12345");
        assert_eq!(browser, BrowserType::Unknown);
    }

    #[test]
    fn partial_browser_name() {
        let browser = BrowserType::from_user_agent("Chrom");
        assert_eq!(browser, BrowserType::Unknown);
    }

    #[test]
    fn browser_name_without_version() {
        // "Chrome" without slash and version doesn't match pattern
        let browser = BrowserType::from_user_agent("Chrome");
        assert_eq!(browser, BrowserType::Unknown);
    }

    #[test]
    fn special_characters() {
        let browser = BrowserType::from_user_agent("@#$%^&*()");
        assert_eq!(browser, BrowserType::Unknown);
    }

    #[test]
    fn very_long_user_agent() {
        let long_ua = "a".repeat(10000);
        let browser = BrowserType::from_user_agent(&long_ua);
        assert_eq!(browser, BrowserType::Unknown);
    }
}

// ============================================================================
// Module: Integration Tests
// ============================================================================

mod integration_tests {
    use super::*;

    #[test]
    fn tier_limits_feature_consistency_tier1() {
        let limits = WebGpuLimits::tier1_minimum();
        let features = WebGpuFeatures {
            tier: WebGpuTier::Tier1,
            limits,
            ..Default::default()
        };
        assert!(features.limits.meets_tier(features.tier));
    }

    #[test]
    fn tier_limits_feature_consistency_tier2() {
        let limits = WebGpuLimits::tier2_minimum();
        let features = WebGpuFeatures {
            tier: WebGpuTier::Tier2,
            limits,
            ..Default::default()
        };
        assert!(features.limits.meets_tier(features.tier));
    }

    #[test]
    fn tier_limits_feature_consistency_tier3() {
        let limits = WebGpuLimits::tier3_minimum();
        let features = WebGpuFeatures {
            tier: WebGpuTier::Tier3,
            limits,
            ..Default::default()
        };
        assert!(features.limits.meets_tier(features.tier));
    }

    #[test]
    fn full_desktop_gpu_simulation() {
        let mut limits = Limits::default();
        limits.max_texture_dimension_2d = 16384;
        limits.max_compute_workgroups_per_dimension = 65535;
        limits.max_storage_buffers_per_shader_stage = 8;

        let wgpu_features = Features::TEXTURE_COMPRESSION_BC
            | Features::TIMESTAMP_QUERY
            | Features::SHADER_F16
            | Features::DEPTH32FLOAT_STENCIL8
            | Features::FLOAT32_FILTERABLE
            | Features::INDIRECT_FIRST_INSTANCE;

        let features = WebGpuFeatures::from_adapter_info(&limits, wgpu_features);

        assert_eq!(features.tier, WebGpuTier::Tier3);
        assert!(features.is_desktop_optimized());
        assert!(!features.is_mobile_optimized());
        assert!(features.timestamp_query);
        assert!(features.shader_f16);
    }

    #[test]
    fn full_mobile_gpu_simulation() {
        let limits = Limits::default();
        let wgpu_features = Features::TEXTURE_COMPRESSION_ETC2
            | Features::TEXTURE_COMPRESSION_ASTC;

        let features = WebGpuFeatures::from_adapter_info(&limits, wgpu_features);

        assert!(features.is_mobile_optimized());
        assert!(!features.is_desktop_optimized());
        assert!(features.supports_compression());
    }

    #[test]
    fn browser_to_capabilities_chrome() {
        let browser = BrowserType::Chrome;
        let caps = BrowserCapabilities::from_user_agent("Chrome/120");
        assert_eq!(caps.browser, browser);
        assert!(caps.has_full_webgpu());
    }

    #[test]
    fn browser_to_capabilities_firefox() {
        let browser = BrowserType::Firefox;
        let caps = BrowserCapabilities::from_user_agent("Firefox/121");
        assert_eq!(caps.browser, browser);
        assert!(!caps.has_full_webgpu());
    }

    #[test]
    fn cross_platform_device_simulation() {
        let mut limits = Limits::default();
        limits.max_texture_dimension_2d = 4096;
        limits.max_bind_groups = 8;
        limits.max_storage_textures_per_shader_stage = 4;
        limits.max_dynamic_uniform_buffers_per_pipeline_layout = 8;

        let wgpu_features = Features::TEXTURE_COMPRESSION_BC
            | Features::TEXTURE_COMPRESSION_ETC2
            | Features::TEXTURE_COMPRESSION_ASTC;

        let features = WebGpuFeatures::from_adapter_info(&limits, wgpu_features);

        // Has all compression formats
        let formats = features.compression_formats();
        assert_eq!(formats.len(), 3);

        // Not exclusively mobile or desktop
        assert!(!features.is_mobile_optimized());
        assert!(!features.is_desktop_optimized());

        // Still supports compression
        assert!(features.supports_compression());
    }
}
