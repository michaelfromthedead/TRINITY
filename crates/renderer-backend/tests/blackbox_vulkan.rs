//! Blackbox contract tests for T-WGPU-P7.2.1 Vulkan Features.
//!
//! CLEANROOM: No src/ access beyond the public API exported by the crate.
//! Tests use only `renderer_backend::backend::*` -- no internal fields,
//! no private methods, no implementation details.
//!
//! Test Categories:
//!   1. API Contract Tests (25+) - VulkanVersion, VulkanDeviceType, VulkanFeatures public API
//!   2. Feature Detection Scenarios (35+) - GPU tier simulation
//!   3. Version Compatibility (20+) - Vulkan version handling
//!   4. Extension Queries (20+) - Extension list validation
//!   5. Integration Scenarios (15+) - Combined workflows

use renderer_backend::backend::{
    VulkanDeviceType, VulkanFeatures, VulkanInfo, VulkanRayTracingTier, VulkanVersion,
};
use std::collections::HashSet;

// =============================================================================
// SECTION 1: API Contract Tests - VulkanVersion (25+)
// =============================================================================

mod vulkan_version_api_contract {
    use super::*;

    // -------------------------------------------------------------------------
    // Constants accessibility
    // -------------------------------------------------------------------------

    #[test]
    fn v1_0_constant_accessible() {
        let version = VulkanVersion::V1_0;
        assert_eq!(version.major, 1);
        assert_eq!(version.minor, 0);
        assert_eq!(version.patch, 0);
    }

    #[test]
    fn v1_1_constant_accessible() {
        let version = VulkanVersion::V1_1;
        assert_eq!(version.major, 1);
        assert_eq!(version.minor, 1);
        assert_eq!(version.patch, 0);
    }

    #[test]
    fn v1_2_constant_accessible() {
        let version = VulkanVersion::V1_2;
        assert_eq!(version.major, 1);
        assert_eq!(version.minor, 2);
        assert_eq!(version.patch, 0);
    }

    #[test]
    fn v1_3_constant_accessible() {
        let version = VulkanVersion::V1_3;
        assert_eq!(version.major, 1);
        assert_eq!(version.minor, 3);
        assert_eq!(version.patch, 0);
    }

    // -------------------------------------------------------------------------
    // Constructor patterns
    // -------------------------------------------------------------------------

    #[test]
    fn new_constructor_with_zero_values() {
        let version = VulkanVersion::new(0, 0, 0);
        assert_eq!(version.major, 0);
        assert_eq!(version.minor, 0);
        assert_eq!(version.patch, 0);
    }

    #[test]
    fn new_constructor_with_typical_values() {
        let version = VulkanVersion::new(1, 3, 250);
        assert_eq!(version.major, 1);
        assert_eq!(version.minor, 3);
        assert_eq!(version.patch, 250);
    }

    #[test]
    fn new_constructor_with_max_minor() {
        // Vulkan uses 10 bits for minor (max 1023)
        let version = VulkanVersion::new(1, 1023, 0);
        assert_eq!(version.minor, 1023);
    }

    #[test]
    fn new_constructor_with_max_patch() {
        // Vulkan uses 12 bits for patch (max 4095)
        let version = VulkanVersion::new(1, 2, 4095);
        assert_eq!(version.patch, 4095);
    }

    #[test]
    fn default_constructor_returns_v1_0() {
        let version = VulkanVersion::default();
        assert_eq!(version, VulkanVersion::V1_0);
    }

    // -------------------------------------------------------------------------
    // Comparison operators
    // -------------------------------------------------------------------------

    #[test]
    fn comparison_equal_versions() {
        let v1 = VulkanVersion::new(1, 2, 0);
        let v2 = VulkanVersion::new(1, 2, 0);
        assert_eq!(v1, v2);
    }

    #[test]
    fn comparison_less_than_major() {
        assert!(VulkanVersion::V1_0 < VulkanVersion::new(2, 0, 0));
    }

    #[test]
    fn comparison_less_than_minor() {
        assert!(VulkanVersion::V1_1 < VulkanVersion::V1_2);
    }

    #[test]
    fn comparison_less_than_patch() {
        let v1 = VulkanVersion::new(1, 2, 100);
        let v2 = VulkanVersion::new(1, 2, 200);
        assert!(v1 < v2);
    }

    #[test]
    fn comparison_greater_than() {
        assert!(VulkanVersion::V1_3 > VulkanVersion::V1_2);
    }

    #[test]
    fn comparison_ordering_chain() {
        assert!(VulkanVersion::V1_0 < VulkanVersion::V1_1);
        assert!(VulkanVersion::V1_1 < VulkanVersion::V1_2);
        assert!(VulkanVersion::V1_2 < VulkanVersion::V1_3);
    }

    #[test]
    fn comparison_partial_ord_some() {
        let v1 = VulkanVersion::V1_2;
        let v2 = VulkanVersion::V1_3;
        assert!(v1.partial_cmp(&v2).is_some());
    }

    // -------------------------------------------------------------------------
    // Display trait
    // -------------------------------------------------------------------------

    #[test]
    fn display_v1_0() {
        assert_eq!(format!("{}", VulkanVersion::V1_0), "1.0.0");
    }

    #[test]
    fn display_v1_3() {
        assert_eq!(format!("{}", VulkanVersion::V1_3), "1.3.0");
    }

    #[test]
    fn display_with_patch() {
        let version = VulkanVersion::new(1, 3, 250);
        assert_eq!(format!("{}", version), "1.3.250");
    }

    #[test]
    fn display_large_patch() {
        let version = VulkanVersion::new(1, 2, 4095);
        assert_eq!(format!("{}", version), "1.2.4095");
    }

    // -------------------------------------------------------------------------
    // Raw encoding
    // -------------------------------------------------------------------------

    #[test]
    fn to_raw_v1_0() {
        assert_eq!(VulkanVersion::V1_0.to_raw(), 0x00400000);
    }

    #[test]
    fn to_raw_v1_2() {
        assert_eq!(VulkanVersion::V1_2.to_raw(), 0x00402000);
    }

    #[test]
    fn from_raw_v1_0() {
        let version = VulkanVersion::from_raw(0x00400000);
        assert_eq!(version, VulkanVersion::V1_0);
    }

    #[test]
    fn from_raw_v1_2() {
        let version = VulkanVersion::from_raw(0x00402000);
        assert_eq!(version, VulkanVersion::V1_2);
    }

    #[test]
    fn raw_round_trip() {
        for v in [VulkanVersion::V1_0, VulkanVersion::V1_1, VulkanVersion::V1_2, VulkanVersion::V1_3] {
            let raw = v.to_raw();
            let decoded = VulkanVersion::from_raw(raw);
            assert_eq!(v, decoded, "round-trip failed for {:?}", v);
        }
    }

    #[test]
    fn raw_round_trip_with_patch() {
        let version = VulkanVersion::new(1, 3, 250);
        let raw = version.to_raw();
        let decoded = VulkanVersion::from_raw(raw);
        assert_eq!(version, decoded);
    }

    // -------------------------------------------------------------------------
    // is_at_least
    // -------------------------------------------------------------------------

    #[test]
    fn is_at_least_exact_match() {
        assert!(VulkanVersion::V1_2.is_at_least(1, 2));
    }

    #[test]
    fn is_at_least_higher_version() {
        assert!(VulkanVersion::V1_3.is_at_least(1, 2));
    }

    #[test]
    fn is_at_least_lower_version() {
        assert!(!VulkanVersion::V1_2.is_at_least(1, 3));
    }

    #[test]
    fn is_at_least_different_major() {
        assert!(!VulkanVersion::V1_3.is_at_least(2, 0));
    }

    #[test]
    fn is_at_least_higher_major() {
        let v2 = VulkanVersion::new(2, 0, 0);
        assert!(v2.is_at_least(1, 9));
    }
}

// =============================================================================
// SECTION 2: API Contract Tests - VulkanDeviceType
// =============================================================================

mod vulkan_device_type_api_contract {
    use super::*;

    // -------------------------------------------------------------------------
    // Variant accessibility
    // -------------------------------------------------------------------------

    #[test]
    fn discrete_gpu_accessible() {
        let t = VulkanDeviceType::DiscreteGpu;
        assert!(t.is_gpu());
        assert!(t.is_hardware());
    }

    #[test]
    fn integrated_gpu_accessible() {
        let t = VulkanDeviceType::IntegratedGpu;
        assert!(t.is_gpu());
        assert!(t.is_hardware());
    }

    #[test]
    fn virtual_gpu_accessible() {
        let t = VulkanDeviceType::VirtualGpu;
        assert!(t.is_gpu());
        assert!(t.is_hardware());
    }

    #[test]
    fn cpu_accessible() {
        let t = VulkanDeviceType::Cpu;
        assert!(!t.is_gpu());
        assert!(!t.is_hardware());
    }

    #[test]
    fn other_accessible() {
        let t = VulkanDeviceType::Other;
        assert!(!t.is_gpu());
        assert!(!t.is_hardware());
    }

    // -------------------------------------------------------------------------
    // Comparison
    // -------------------------------------------------------------------------

    #[test]
    fn comparison_equal() {
        assert_eq!(VulkanDeviceType::DiscreteGpu, VulkanDeviceType::DiscreteGpu);
    }

    #[test]
    fn comparison_not_equal() {
        assert_ne!(VulkanDeviceType::DiscreteGpu, VulkanDeviceType::IntegratedGpu);
    }

    #[test]
    fn comparison_all_variants_distinct() {
        let variants = [
            VulkanDeviceType::DiscreteGpu,
            VulkanDeviceType::IntegratedGpu,
            VulkanDeviceType::VirtualGpu,
            VulkanDeviceType::Cpu,
            VulkanDeviceType::Other,
        ];
        for i in 0..variants.len() {
            for j in (i + 1)..variants.len() {
                assert_ne!(variants[i], variants[j], "variants {:?} and {:?} should differ", variants[i], variants[j]);
            }
        }
    }

    // -------------------------------------------------------------------------
    // Debug output
    // -------------------------------------------------------------------------

    #[test]
    fn debug_discrete_gpu() {
        let debug = format!("{:?}", VulkanDeviceType::DiscreteGpu);
        assert!(debug.contains("DiscreteGpu"));
    }

    #[test]
    fn debug_integrated_gpu() {
        let debug = format!("{:?}", VulkanDeviceType::IntegratedGpu);
        assert!(debug.contains("IntegratedGpu"));
    }

    #[test]
    fn debug_all_variants() {
        for variant in [
            VulkanDeviceType::DiscreteGpu,
            VulkanDeviceType::IntegratedGpu,
            VulkanDeviceType::VirtualGpu,
            VulkanDeviceType::Cpu,
            VulkanDeviceType::Other,
        ] {
            let debug = format!("{:?}", variant);
            assert!(!debug.is_empty());
        }
    }

    // -------------------------------------------------------------------------
    // Display output
    // -------------------------------------------------------------------------

    #[test]
    fn display_discrete_gpu() {
        assert_eq!(format!("{}", VulkanDeviceType::DiscreteGpu), "Discrete GPU");
    }

    #[test]
    fn display_integrated_gpu() {
        assert_eq!(format!("{}", VulkanDeviceType::IntegratedGpu), "Integrated GPU");
    }

    #[test]
    fn display_virtual_gpu() {
        assert_eq!(format!("{}", VulkanDeviceType::VirtualGpu), "Virtual GPU");
    }

    #[test]
    fn display_cpu() {
        assert_eq!(format!("{}", VulkanDeviceType::Cpu), "CPU");
    }

    #[test]
    fn display_other() {
        assert_eq!(format!("{}", VulkanDeviceType::Other), "Other");
    }

    // -------------------------------------------------------------------------
    // name() method
    // -------------------------------------------------------------------------

    #[test]
    fn name_discrete_gpu() {
        assert_eq!(VulkanDeviceType::DiscreteGpu.name(), "Discrete GPU");
    }

    #[test]
    fn name_all_variants_non_empty() {
        for variant in [
            VulkanDeviceType::DiscreteGpu,
            VulkanDeviceType::IntegratedGpu,
            VulkanDeviceType::VirtualGpu,
            VulkanDeviceType::Cpu,
            VulkanDeviceType::Other,
        ] {
            assert!(!variant.name().is_empty());
        }
    }

    // -------------------------------------------------------------------------
    // Default
    // -------------------------------------------------------------------------

    #[test]
    fn default_is_other() {
        assert_eq!(VulkanDeviceType::default(), VulkanDeviceType::Other);
    }
}

// =============================================================================
// SECTION 3: API Contract Tests - VulkanFeatures
// =============================================================================

mod vulkan_features_api_contract {
    use super::*;

    // -------------------------------------------------------------------------
    // Default construction
    // -------------------------------------------------------------------------

    #[test]
    fn default_all_false() {
        let f = VulkanFeatures::default();
        assert!(!f.ray_tracing);
        assert!(!f.ray_query);
        assert!(!f.descriptor_indexing);
        assert!(!f.timeline_semaphores);
        assert!(!f.buffer_device_address);
        assert!(!f.mesh_shading);
        assert!(!f.dynamic_rendering);
        assert!(!f.synchronization2);
        assert!(!f.extended_dynamic_state);
        assert!(!f.maintenance4);
    }

    // -------------------------------------------------------------------------
    // Field access
    // -------------------------------------------------------------------------

    #[test]
    fn field_ray_tracing_readable() {
        let f = VulkanFeatures::default();
        let _ = f.ray_tracing;
    }

    #[test]
    fn field_ray_query_readable() {
        let f = VulkanFeatures::default();
        let _ = f.ray_query;
    }

    #[test]
    fn field_descriptor_indexing_readable() {
        let f = VulkanFeatures::default();
        let _ = f.descriptor_indexing;
    }

    #[test]
    fn field_timeline_semaphores_readable() {
        let f = VulkanFeatures::default();
        let _ = f.timeline_semaphores;
    }

    #[test]
    fn field_buffer_device_address_readable() {
        let f = VulkanFeatures::default();
        let _ = f.buffer_device_address;
    }

    #[test]
    fn field_mesh_shading_readable() {
        let f = VulkanFeatures::default();
        let _ = f.mesh_shading;
    }

    #[test]
    fn field_dynamic_rendering_readable() {
        let f = VulkanFeatures::default();
        let _ = f.dynamic_rendering;
    }

    #[test]
    fn field_synchronization2_readable() {
        let f = VulkanFeatures::default();
        let _ = f.synchronization2;
    }

    #[test]
    fn field_extended_dynamic_state_readable() {
        let f = VulkanFeatures::default();
        let _ = f.extended_dynamic_state;
    }

    #[test]
    fn field_maintenance4_readable() {
        let f = VulkanFeatures::default();
        let _ = f.maintenance4;
    }

    // -------------------------------------------------------------------------
    // Method calls
    // -------------------------------------------------------------------------

    #[test]
    fn method_supports_rt_pipeline_callable() {
        let f = VulkanFeatures::default();
        let _ = f.supports_rt_pipeline();
    }

    #[test]
    fn method_supports_ray_query_callable() {
        let f = VulkanFeatures::default();
        let _ = f.supports_ray_query();
    }

    #[test]
    fn method_supports_any_rt_callable() {
        let f = VulkanFeatures::default();
        let _ = f.supports_any_rt();
    }

    #[test]
    fn method_supports_bindless_callable() {
        let f = VulkanFeatures::default();
        let _ = f.supports_bindless();
    }

    #[test]
    fn method_supports_vulkan_1_2_callable() {
        let f = VulkanFeatures::default();
        let _ = f.supports_vulkan_1_2();
    }

    #[test]
    fn method_supports_vulkan_1_3_callable() {
        let f = VulkanFeatures::default();
        let _ = f.supports_vulkan_1_3();
    }

    #[test]
    fn method_ray_tracing_tier_callable() {
        let f = VulkanFeatures::default();
        let _ = f.ray_tracing_tier();
    }

    #[test]
    fn method_summary_callable() {
        let f = VulkanFeatures::default();
        let _ = f.summary();
    }

    #[test]
    fn method_supports_mesh_shading_callable() {
        let f = VulkanFeatures::default();
        let _ = f.supports_mesh_shading();
    }

    #[test]
    fn method_supports_modern_sync_callable() {
        let f = VulkanFeatures::default();
        let _ = f.supports_modern_sync();
    }
}

// =============================================================================
// SECTION 4: Feature Detection Scenarios (35+)
// =============================================================================

mod feature_detection_scenarios {
    use super::*;

    // -------------------------------------------------------------------------
    // High-end discrete GPU scenario
    // -------------------------------------------------------------------------

    fn high_end_gpu_features() -> VulkanFeatures {
        VulkanFeatures {
            ray_tracing: true,
            ray_query: true,
            descriptor_indexing: true,
            timeline_semaphores: true,
            buffer_device_address: true,
            mesh_shading: true,
            dynamic_rendering: true,
            synchronization2: true,
            extended_dynamic_state: true,
            maintenance4: true,
        }
    }

    #[test]
    fn high_end_supports_rt_pipeline() {
        let f = high_end_gpu_features();
        assert!(f.supports_rt_pipeline());
    }

    #[test]
    fn high_end_supports_ray_query() {
        let f = high_end_gpu_features();
        assert!(f.supports_ray_query());
    }

    #[test]
    fn high_end_supports_any_rt() {
        let f = high_end_gpu_features();
        assert!(f.supports_any_rt());
    }

    #[test]
    fn high_end_supports_bindless() {
        let f = high_end_gpu_features();
        assert!(f.supports_bindless());
    }

    #[test]
    fn high_end_supports_mesh_shading() {
        let f = high_end_gpu_features();
        assert!(f.supports_mesh_shading());
    }

    #[test]
    fn high_end_supports_vulkan_1_2() {
        let f = high_end_gpu_features();
        assert!(f.supports_vulkan_1_2());
    }

    #[test]
    fn high_end_supports_vulkan_1_3() {
        let f = high_end_gpu_features();
        assert!(f.supports_vulkan_1_3());
    }

    #[test]
    fn high_end_supports_modern_sync() {
        let f = high_end_gpu_features();
        assert!(f.supports_modern_sync());
    }

    #[test]
    fn high_end_ray_tracing_tier_is_full() {
        let f = high_end_gpu_features();
        assert_eq!(f.ray_tracing_tier(), VulkanRayTracingTier::Full);
    }

    #[test]
    fn high_end_summary_contains_all_major_features() {
        let f = high_end_gpu_features();
        let summary = f.summary();
        assert!(summary.contains("RT-Pipeline"));
        assert!(summary.contains("RT-Query"));
        assert!(summary.contains("Descriptor-Indexing"));
        assert!(summary.contains("Timeline-Semaphores"));
        assert!(summary.contains("BDA"));
        assert!(summary.contains("Mesh-Shaders"));
        assert!(summary.contains("Dynamic-Rendering"));
    }

    // -------------------------------------------------------------------------
    // Mid-range GPU scenario
    // -------------------------------------------------------------------------

    fn mid_range_gpu_features() -> VulkanFeatures {
        VulkanFeatures {
            ray_tracing: false,
            ray_query: false,
            descriptor_indexing: true,
            timeline_semaphores: true,
            buffer_device_address: true,
            mesh_shading: false,
            dynamic_rendering: true,
            synchronization2: true,
            extended_dynamic_state: true,
            maintenance4: false,
        }
    }

    #[test]
    fn mid_range_no_rt_pipeline() {
        let f = mid_range_gpu_features();
        assert!(!f.supports_rt_pipeline());
    }

    #[test]
    fn mid_range_no_ray_query() {
        let f = mid_range_gpu_features();
        assert!(!f.supports_ray_query());
    }

    #[test]
    fn mid_range_no_any_rt() {
        let f = mid_range_gpu_features();
        assert!(!f.supports_any_rt());
    }

    #[test]
    fn mid_range_supports_bindless() {
        let f = mid_range_gpu_features();
        assert!(f.supports_bindless());
    }

    #[test]
    fn mid_range_no_mesh_shading() {
        let f = mid_range_gpu_features();
        assert!(!f.supports_mesh_shading());
    }

    #[test]
    fn mid_range_supports_vulkan_1_2() {
        let f = mid_range_gpu_features();
        assert!(f.supports_vulkan_1_2());
    }

    #[test]
    fn mid_range_supports_vulkan_1_3() {
        let f = mid_range_gpu_features();
        assert!(f.supports_vulkan_1_3());
    }

    #[test]
    fn mid_range_ray_tracing_tier_is_none() {
        let f = mid_range_gpu_features();
        assert_eq!(f.ray_tracing_tier(), VulkanRayTracingTier::None);
    }

    // -------------------------------------------------------------------------
    // Integrated GPU scenario
    // -------------------------------------------------------------------------

    fn integrated_gpu_features() -> VulkanFeatures {
        VulkanFeatures {
            ray_tracing: false,
            ray_query: false,
            descriptor_indexing: true,
            timeline_semaphores: true,
            buffer_device_address: false,
            mesh_shading: false,
            dynamic_rendering: false,
            synchronization2: false,
            extended_dynamic_state: true,
            maintenance4: false,
        }
    }

    #[test]
    fn integrated_no_rt() {
        let f = integrated_gpu_features();
        assert!(!f.supports_any_rt());
    }

    #[test]
    fn integrated_no_bindless() {
        // Bindless requires both descriptor_indexing AND buffer_device_address
        let f = integrated_gpu_features();
        assert!(!f.supports_bindless());
    }

    #[test]
    fn integrated_no_vulkan_1_2() {
        // Vulkan 1.2 requires timeline_semaphores AND buffer_device_address
        let f = integrated_gpu_features();
        assert!(!f.supports_vulkan_1_2());
    }

    #[test]
    fn integrated_no_vulkan_1_3() {
        let f = integrated_gpu_features();
        assert!(!f.supports_vulkan_1_3());
    }

    #[test]
    fn integrated_no_modern_sync() {
        let f = integrated_gpu_features();
        assert!(!f.supports_modern_sync());
    }

    // -------------------------------------------------------------------------
    // Legacy GPU scenario (Vulkan 1.0 baseline)
    // -------------------------------------------------------------------------

    fn legacy_gpu_features() -> VulkanFeatures {
        VulkanFeatures::default()
    }

    #[test]
    fn legacy_all_false() {
        let f = legacy_gpu_features();
        assert!(!f.ray_tracing);
        assert!(!f.ray_query);
        assert!(!f.descriptor_indexing);
        assert!(!f.timeline_semaphores);
        assert!(!f.buffer_device_address);
        assert!(!f.mesh_shading);
        assert!(!f.dynamic_rendering);
        assert!(!f.synchronization2);
    }

    #[test]
    fn legacy_no_rt() {
        let f = legacy_gpu_features();
        assert!(!f.supports_any_rt());
        assert_eq!(f.ray_tracing_tier(), VulkanRayTracingTier::None);
    }

    #[test]
    fn legacy_no_bindless() {
        let f = legacy_gpu_features();
        assert!(!f.supports_bindless());
    }

    #[test]
    fn legacy_no_vulkan_1_2() {
        let f = legacy_gpu_features();
        assert!(!f.supports_vulkan_1_2());
    }

    #[test]
    fn legacy_no_vulkan_1_3() {
        let f = legacy_gpu_features();
        assert!(!f.supports_vulkan_1_3());
    }

    #[test]
    fn legacy_summary_is_none() {
        let f = legacy_gpu_features();
        assert_eq!(f.summary(), "None");
    }

    // -------------------------------------------------------------------------
    // Ray query only scenario
    // -------------------------------------------------------------------------

    fn ray_query_only_features() -> VulkanFeatures {
        VulkanFeatures {
            ray_tracing: false,
            ray_query: true,
            descriptor_indexing: true,
            timeline_semaphores: true,
            buffer_device_address: true,
            mesh_shading: false,
            dynamic_rendering: true,
            synchronization2: true,
            extended_dynamic_state: true,
            maintenance4: false,
        }
    }

    #[test]
    fn ray_query_only_no_rt_pipeline() {
        let f = ray_query_only_features();
        assert!(!f.supports_rt_pipeline());
    }

    #[test]
    fn ray_query_only_supports_ray_query() {
        let f = ray_query_only_features();
        assert!(f.supports_ray_query());
    }

    #[test]
    fn ray_query_only_supports_any_rt() {
        let f = ray_query_only_features();
        assert!(f.supports_any_rt());
    }

    #[test]
    fn ray_query_only_tier_is_query() {
        let f = ray_query_only_features();
        assert_eq!(f.ray_tracing_tier(), VulkanRayTracingTier::Query);
    }
}

// =============================================================================
// SECTION 5: Version Compatibility (20+)
// =============================================================================

mod version_compatibility {
    use super::*;

    // -------------------------------------------------------------------------
    // Vulkan 1.0 baseline
    // -------------------------------------------------------------------------

    #[test]
    fn v1_0_is_baseline() {
        let v = VulkanVersion::V1_0;
        assert!(v.is_at_least(1, 0));
        assert!(!v.is_at_least(1, 1));
    }

    #[test]
    fn v1_0_less_than_all_others() {
        assert!(VulkanVersion::V1_0 < VulkanVersion::V1_1);
        assert!(VulkanVersion::V1_0 < VulkanVersion::V1_2);
        assert!(VulkanVersion::V1_0 < VulkanVersion::V1_3);
    }

    // -------------------------------------------------------------------------
    // Vulkan 1.1 features
    // -------------------------------------------------------------------------

    #[test]
    fn v1_1_is_at_least_1_1() {
        let v = VulkanVersion::V1_1;
        assert!(v.is_at_least(1, 1));
    }

    #[test]
    fn v1_1_not_at_least_1_2() {
        let v = VulkanVersion::V1_1;
        assert!(!v.is_at_least(1, 2));
    }

    // -------------------------------------------------------------------------
    // Vulkan 1.2 features (timeline semaphores)
    // -------------------------------------------------------------------------

    #[test]
    fn v1_2_is_at_least_1_2() {
        let v = VulkanVersion::V1_2;
        assert!(v.is_at_least(1, 2));
    }

    #[test]
    fn v1_2_features_support_check() {
        // A feature set representing Vulkan 1.2 minimum
        let features = VulkanFeatures {
            ray_tracing: false,
            ray_query: false,
            descriptor_indexing: false,
            timeline_semaphores: true,
            buffer_device_address: true,
            mesh_shading: false,
            dynamic_rendering: false,
            synchronization2: false,
            extended_dynamic_state: false,
            maintenance4: false,
        };
        assert!(features.supports_vulkan_1_2());
    }

    // -------------------------------------------------------------------------
    // Vulkan 1.3 features (dynamic rendering)
    // -------------------------------------------------------------------------

    #[test]
    fn v1_3_is_at_least_1_3() {
        let v = VulkanVersion::V1_3;
        assert!(v.is_at_least(1, 3));
    }

    #[test]
    fn v1_3_is_at_least_1_2() {
        let v = VulkanVersion::V1_3;
        assert!(v.is_at_least(1, 2));
    }

    #[test]
    fn v1_3_features_support_check() {
        // A feature set representing Vulkan 1.3 minimum
        let features = VulkanFeatures {
            ray_tracing: false,
            ray_query: false,
            descriptor_indexing: false,
            timeline_semaphores: false,
            buffer_device_address: false,
            mesh_shading: false,
            dynamic_rendering: true,
            synchronization2: true,
            extended_dynamic_state: false,
            maintenance4: false,
        };
        assert!(features.supports_vulkan_1_3());
    }

    // -------------------------------------------------------------------------
    // Version ordering correctness
    // -------------------------------------------------------------------------

    #[test]
    fn version_ordering_is_transitive() {
        let v1 = VulkanVersion::V1_0;
        let v2 = VulkanVersion::V1_1;
        let v3 = VulkanVersion::V1_2;

        assert!(v1 < v2);
        assert!(v2 < v3);
        assert!(v1 < v3); // Transitivity
    }

    #[test]
    fn version_ordering_antisymmetric() {
        let v1 = VulkanVersion::V1_2;
        let v2 = VulkanVersion::V1_3;

        assert!(v1 < v2);
        assert!(!(v2 < v1));
    }

    #[test]
    fn version_ordering_reflexive() {
        let v = VulkanVersion::V1_2;
        assert!(v == v);
        assert!(v <= v);
        assert!(v >= v);
    }

    // -------------------------------------------------------------------------
    // Version comparison in conditionals
    // -------------------------------------------------------------------------

    #[test]
    fn version_conditional_1_2_check() {
        let version = VulkanVersion::V1_3;
        if version >= VulkanVersion::V1_2 {
            // Should enter this branch
            assert!(true);
        } else {
            panic!("1.3 should be >= 1.2");
        }
    }

    #[test]
    fn version_conditional_1_3_check() {
        let version = VulkanVersion::V1_2;
        if version >= VulkanVersion::V1_3 {
            panic!("1.2 should not be >= 1.3");
        } else {
            // Should enter this branch
            assert!(true);
        }
    }

    #[test]
    fn version_conditional_chain() {
        let version = VulkanVersion::V1_2;

        let tier = if version >= VulkanVersion::V1_3 {
            "1.3+"
        } else if version >= VulkanVersion::V1_2 {
            "1.2"
        } else if version >= VulkanVersion::V1_1 {
            "1.1"
        } else {
            "1.0"
        };

        assert_eq!(tier, "1.2");
    }

    #[test]
    fn version_with_patch_comparison() {
        let v1 = VulkanVersion::new(1, 3, 100);
        let v2 = VulkanVersion::new(1, 3, 200);
        let v3 = VulkanVersion::V1_3; // patch 0

        assert!(v3 < v1);
        assert!(v1 < v2);
    }

    #[test]
    fn version_max_selection() {
        let versions = [
            VulkanVersion::V1_0,
            VulkanVersion::V1_2,
            VulkanVersion::V1_1,
            VulkanVersion::V1_3,
        ];
        let max = versions.iter().max().unwrap();
        assert_eq!(*max, VulkanVersion::V1_3);
    }

    #[test]
    fn version_min_selection() {
        let versions = [
            VulkanVersion::V1_3,
            VulkanVersion::V1_2,
            VulkanVersion::V1_1,
            VulkanVersion::V1_0,
        ];
        let min = versions.iter().min().unwrap();
        assert_eq!(*min, VulkanVersion::V1_0);
    }
}

// =============================================================================
// SECTION 6: Extension Queries (20+)
// =============================================================================

mod extension_queries {
    use super::*;

    // -------------------------------------------------------------------------
    // required_instance_extensions()
    // -------------------------------------------------------------------------

    #[test]
    fn required_instance_extensions_non_empty() {
        let extensions = VulkanFeatures::required_instance_extensions();
        assert!(!extensions.is_empty());
    }

    #[test]
    fn required_instance_extensions_contains_physical_device_properties2() {
        let extensions = VulkanFeatures::required_instance_extensions();
        assert!(extensions.contains(&"VK_KHR_get_physical_device_properties2"));
    }

    #[test]
    fn required_instance_extensions_contains_debug_utils() {
        let extensions = VulkanFeatures::required_instance_extensions();
        assert!(extensions.contains(&"VK_EXT_debug_utils"));
    }

    #[test]
    fn required_instance_extensions_no_duplicates() {
        let extensions = VulkanFeatures::required_instance_extensions();
        let set: HashSet<_> = extensions.iter().collect();
        assert_eq!(extensions.len(), set.len(), "duplicates found in required_instance_extensions");
    }

    #[test]
    fn required_instance_extensions_all_vk_prefixed() {
        let extensions = VulkanFeatures::required_instance_extensions();
        for ext in extensions {
            assert!(ext.starts_with("VK_"), "extension {} does not start with VK_", ext);
        }
    }

    // -------------------------------------------------------------------------
    // ray_tracing_extensions()
    // -------------------------------------------------------------------------

    #[test]
    fn ray_tracing_extensions_non_empty() {
        let extensions = VulkanFeatures::ray_tracing_extensions();
        assert!(!extensions.is_empty());
    }

    #[test]
    fn ray_tracing_extensions_contains_pipeline() {
        let extensions = VulkanFeatures::ray_tracing_extensions();
        assert!(extensions.contains(&"VK_KHR_ray_tracing_pipeline"));
    }

    #[test]
    fn ray_tracing_extensions_contains_acceleration_structure() {
        let extensions = VulkanFeatures::ray_tracing_extensions();
        assert!(extensions.contains(&"VK_KHR_acceleration_structure"));
    }

    #[test]
    fn ray_tracing_extensions_contains_ray_query() {
        let extensions = VulkanFeatures::ray_tracing_extensions();
        assert!(extensions.contains(&"VK_KHR_ray_query"));
    }

    #[test]
    fn ray_tracing_extensions_contains_buffer_device_address() {
        let extensions = VulkanFeatures::ray_tracing_extensions();
        assert!(extensions.contains(&"VK_KHR_buffer_device_address"));
    }

    #[test]
    fn ray_tracing_extensions_no_duplicates() {
        let extensions = VulkanFeatures::ray_tracing_extensions();
        let set: HashSet<_> = extensions.iter().collect();
        assert_eq!(extensions.len(), set.len());
    }

    #[test]
    fn ray_tracing_extensions_all_vk_prefixed() {
        let extensions = VulkanFeatures::ray_tracing_extensions();
        for ext in extensions {
            assert!(ext.starts_with("VK_"));
        }
    }

    // -------------------------------------------------------------------------
    // mesh_shader_extensions()
    // -------------------------------------------------------------------------

    #[test]
    fn mesh_shader_extensions_non_empty() {
        let extensions = VulkanFeatures::mesh_shader_extensions();
        assert!(!extensions.is_empty());
    }

    #[test]
    fn mesh_shader_extensions_contains_ext_mesh_shader() {
        let extensions = VulkanFeatures::mesh_shader_extensions();
        assert!(extensions.contains(&"VK_EXT_mesh_shader"));
    }

    #[test]
    fn mesh_shader_extensions_no_duplicates() {
        let extensions = VulkanFeatures::mesh_shader_extensions();
        let set: HashSet<_> = extensions.iter().collect();
        assert_eq!(extensions.len(), set.len());
    }

    // -------------------------------------------------------------------------
    // bindless_extensions()
    // -------------------------------------------------------------------------

    #[test]
    fn bindless_extensions_non_empty() {
        let extensions = VulkanFeatures::bindless_extensions();
        assert!(!extensions.is_empty());
    }

    #[test]
    fn bindless_extensions_contains_descriptor_indexing() {
        let extensions = VulkanFeatures::bindless_extensions();
        assert!(extensions.contains(&"VK_EXT_descriptor_indexing"));
    }

    #[test]
    fn bindless_extensions_contains_buffer_device_address() {
        let extensions = VulkanFeatures::bindless_extensions();
        assert!(extensions.contains(&"VK_KHR_buffer_device_address"));
    }

    #[test]
    fn bindless_extensions_no_duplicates() {
        let extensions = VulkanFeatures::bindless_extensions();
        let set: HashSet<_> = extensions.iter().collect();
        assert_eq!(extensions.len(), set.len());
    }

    #[test]
    fn bindless_extensions_all_vk_prefixed() {
        let extensions = VulkanFeatures::bindless_extensions();
        for ext in extensions {
            assert!(ext.starts_with("VK_"));
        }
    }

    // -------------------------------------------------------------------------
    // Cross-extension relationships
    // -------------------------------------------------------------------------

    #[test]
    fn rt_and_bindless_share_bda() {
        let rt_ext = VulkanFeatures::ray_tracing_extensions();
        let bindless_ext = VulkanFeatures::bindless_extensions();

        // Both should contain buffer_device_address
        assert!(rt_ext.contains(&"VK_KHR_buffer_device_address"));
        assert!(bindless_ext.contains(&"VK_KHR_buffer_device_address"));
    }
}

// =============================================================================
// SECTION 7: VulkanRayTracingTier Tests
// =============================================================================

mod ray_tracing_tier_tests {
    use super::*;

    #[test]
    fn tier_none_default() {
        assert_eq!(VulkanRayTracingTier::default(), VulkanRayTracingTier::None);
    }

    #[test]
    fn tier_none_not_available() {
        let tier = VulkanRayTracingTier::None;
        assert!(!tier.is_available());
        assert!(!tier.is_full());
    }

    #[test]
    fn tier_query_available_not_full() {
        let tier = VulkanRayTracingTier::Query;
        assert!(tier.is_available());
        assert!(!tier.is_full());
    }

    #[test]
    fn tier_full_available_and_full() {
        let tier = VulkanRayTracingTier::Full;
        assert!(tier.is_available());
        assert!(tier.is_full());
    }

    #[test]
    fn tier_none_name() {
        assert_eq!(VulkanRayTracingTier::None.name(), "None");
    }

    #[test]
    fn tier_query_name() {
        assert_eq!(VulkanRayTracingTier::Query.name(), "Ray Query");
    }

    #[test]
    fn tier_full_name() {
        assert_eq!(VulkanRayTracingTier::Full.name(), "Full Pipeline");
    }

    #[test]
    fn tier_display_none() {
        assert_eq!(format!("{}", VulkanRayTracingTier::None), "None");
    }

    #[test]
    fn tier_display_query() {
        assert_eq!(format!("{}", VulkanRayTracingTier::Query), "Ray Query");
    }

    #[test]
    fn tier_display_full() {
        assert_eq!(format!("{}", VulkanRayTracingTier::Full), "Full Pipeline");
    }

    #[test]
    fn tier_comparison_equal() {
        assert_eq!(VulkanRayTracingTier::Full, VulkanRayTracingTier::Full);
    }

    #[test]
    fn tier_comparison_not_equal() {
        assert_ne!(VulkanRayTracingTier::None, VulkanRayTracingTier::Full);
    }
}

// =============================================================================
// SECTION 8: Integration Scenarios (15+)
// =============================================================================

mod integration_scenarios {
    use super::*;

    // -------------------------------------------------------------------------
    // Feature detection workflow
    // -------------------------------------------------------------------------

    #[test]
    fn workflow_feature_detection_high_end() {
        // Simulate detecting features on a high-end GPU
        let features = VulkanFeatures {
            ray_tracing: true,
            ray_query: true,
            descriptor_indexing: true,
            timeline_semaphores: true,
            buffer_device_address: true,
            mesh_shading: true,
            dynamic_rendering: true,
            synchronization2: true,
            extended_dynamic_state: true,
            maintenance4: true,
        };

        // Check what rendering paths are available
        let can_use_rt = features.supports_rt_pipeline();
        let can_use_bindless = features.supports_bindless();
        let can_use_mesh = features.supports_mesh_shading();

        assert!(can_use_rt);
        assert!(can_use_bindless);
        assert!(can_use_mesh);
    }

    #[test]
    fn workflow_feature_detection_fallback() {
        // Simulate fallback path selection
        let features = VulkanFeatures::default();

        let render_path = if features.supports_rt_pipeline() {
            "ray_traced"
        } else if features.supports_bindless() {
            "gpu_driven"
        } else {
            "traditional"
        };

        assert_eq!(render_path, "traditional");
    }

    // -------------------------------------------------------------------------
    // Version + features combination
    // -------------------------------------------------------------------------

    #[test]
    fn version_and_features_1_2_combo() {
        let version = VulkanVersion::V1_2;
        let features = VulkanFeatures {
            ray_tracing: false,
            ray_query: false,
            descriptor_indexing: false,
            timeline_semaphores: true,
            buffer_device_address: true,
            mesh_shading: false,
            dynamic_rendering: false,
            synchronization2: false,
            extended_dynamic_state: false,
            maintenance4: false,
        };

        assert!(version.is_at_least(1, 2));
        assert!(features.supports_vulkan_1_2());
    }

    #[test]
    fn version_and_features_1_3_combo() {
        let version = VulkanVersion::V1_3;
        let features = VulkanFeatures {
            ray_tracing: false,
            ray_query: false,
            descriptor_indexing: false,
            timeline_semaphores: true,
            buffer_device_address: true,
            mesh_shading: false,
            dynamic_rendering: true,
            synchronization2: true,
            extended_dynamic_state: false,
            maintenance4: false,
        };

        assert!(version.is_at_least(1, 3));
        assert!(features.supports_vulkan_1_2());
        assert!(features.supports_vulkan_1_3());
    }

    // -------------------------------------------------------------------------
    // Device type + features combination
    // -------------------------------------------------------------------------

    #[test]
    fn device_type_and_features_discrete() {
        let device_type = VulkanDeviceType::DiscreteGpu;
        let features = VulkanFeatures {
            ray_tracing: true,
            ray_query: true,
            descriptor_indexing: true,
            timeline_semaphores: true,
            buffer_device_address: true,
            mesh_shading: true,
            dynamic_rendering: true,
            synchronization2: true,
            extended_dynamic_state: true,
            maintenance4: true,
        };

        assert!(device_type.is_hardware());
        assert!(device_type.is_gpu());
        assert!(features.supports_rt_pipeline());
    }

    #[test]
    fn device_type_and_features_integrated() {
        let device_type = VulkanDeviceType::IntegratedGpu;
        let features = VulkanFeatures {
            ray_tracing: false,
            ray_query: false,
            descriptor_indexing: true,
            timeline_semaphores: true,
            buffer_device_address: false,
            mesh_shading: false,
            dynamic_rendering: false,
            synchronization2: false,
            extended_dynamic_state: false,
            maintenance4: false,
        };

        assert!(device_type.is_hardware());
        assert!(device_type.is_gpu());
        assert!(!features.supports_rt_pipeline());
    }

    // -------------------------------------------------------------------------
    // Complete VulkanInfo construction
    // -------------------------------------------------------------------------

    #[test]
    fn vulkan_info_construction() {
        let info = VulkanInfo {
            version: VulkanVersion::V1_3,
            features: VulkanFeatures {
                ray_tracing: true,
                ray_query: true,
                descriptor_indexing: true,
                timeline_semaphores: true,
                buffer_device_address: true,
                mesh_shading: false,
                dynamic_rendering: true,
                synchronization2: true,
                extended_dynamic_state: true,
                maintenance4: true,
            },
            driver_name: "NVIDIA".to_string(),
            driver_version: 536870912,
            device_name: "GeForce RTX 4090".to_string(),
            device_type: VulkanDeviceType::DiscreteGpu,
            vendor_id: 0x10DE,
            device_id: 0x2684,
        };

        assert_eq!(info.version, VulkanVersion::V1_3);
        assert!(info.features.supports_rt_pipeline());
        assert!(info.device_type.is_hardware());
    }

    #[test]
    fn vulkan_info_summary() {
        let info = VulkanInfo {
            version: VulkanVersion::V1_3,
            features: VulkanFeatures::default(),
            driver_name: "NVIDIA".to_string(),
            driver_version: 0,
            device_name: "RTX 4090".to_string(),
            device_type: VulkanDeviceType::DiscreteGpu,
            vendor_id: 0,
            device_id: 0,
        };

        let summary = info.summary();
        assert!(summary.contains("RTX 4090"));
        assert!(summary.contains("NVIDIA"));
        assert!(summary.contains("1.3.0"));
    }

    #[test]
    fn vulkan_info_is_suitable_hardware_1_2() {
        let mut info = VulkanInfo::default();
        info.device_type = VulkanDeviceType::DiscreteGpu;
        info.features.timeline_semaphores = true;
        info.features.buffer_device_address = true;

        assert!(info.is_suitable());
    }

    #[test]
    fn vulkan_info_not_suitable_no_hardware() {
        let mut info = VulkanInfo::default();
        info.device_type = VulkanDeviceType::Cpu;
        info.features.timeline_semaphores = true;
        info.features.buffer_device_address = true;

        assert!(!info.is_suitable());
    }

    #[test]
    fn vulkan_info_not_suitable_no_1_2() {
        let mut info = VulkanInfo::default();
        info.device_type = VulkanDeviceType::DiscreteGpu;
        // Missing Vulkan 1.2 features

        assert!(!info.is_suitable());
    }

    #[test]
    fn vulkan_info_supports_ray_tracing() {
        let mut info = VulkanInfo::default();
        info.features.ray_tracing = true;
        info.features.buffer_device_address = true;

        assert!(info.supports_ray_tracing());
    }

    #[test]
    fn vulkan_info_default_values() {
        let info = VulkanInfo::default();

        assert_eq!(info.version, VulkanVersion::V1_0);
        assert!(info.driver_name.is_empty());
        assert!(info.device_name.is_empty());
        assert_eq!(info.device_type, VulkanDeviceType::Other);
        assert_eq!(info.vendor_id, 0);
        assert_eq!(info.device_id, 0);
    }

    // -------------------------------------------------------------------------
    // Info aggregation patterns
    // -------------------------------------------------------------------------

    #[test]
    fn aggregation_best_gpu_selection() {
        let gpus = vec![
            VulkanInfo {
                version: VulkanVersion::V1_2,
                features: VulkanFeatures::default(),
                driver_name: "Intel".to_string(),
                driver_version: 0,
                device_name: "Intel UHD".to_string(),
                device_type: VulkanDeviceType::IntegratedGpu,
                vendor_id: 0x8086,
                device_id: 0,
            },
            VulkanInfo {
                version: VulkanVersion::V1_3,
                features: VulkanFeatures {
                    ray_tracing: true,
                    ray_query: true,
                    descriptor_indexing: true,
                    timeline_semaphores: true,
                    buffer_device_address: true,
                    mesh_shading: false,
                    dynamic_rendering: true,
                    synchronization2: true,
                    extended_dynamic_state: true,
                    maintenance4: true,
                },
                driver_name: "NVIDIA".to_string(),
                driver_version: 0,
                device_name: "RTX 4090".to_string(),
                device_type: VulkanDeviceType::DiscreteGpu,
                vendor_id: 0x10DE,
                device_id: 0,
            },
        ];

        // Select GPU with ray tracing support
        let best = gpus.iter().find(|g| g.features.supports_rt_pipeline());
        assert!(best.is_some());
        assert_eq!(best.unwrap().device_name, "RTX 4090");
    }

    #[test]
    fn aggregation_feature_requirement_check() {
        let required_features = VulkanFeatures {
            ray_tracing: false,
            ray_query: false,
            descriptor_indexing: true,
            timeline_semaphores: true,
            buffer_device_address: true,
            mesh_shading: false,
            dynamic_rendering: false,
            synchronization2: false,
            extended_dynamic_state: false,
            maintenance4: false,
        };

        let available_features = VulkanFeatures {
            ray_tracing: true,
            ray_query: true,
            descriptor_indexing: true,
            timeline_semaphores: true,
            buffer_device_address: true,
            mesh_shading: true,
            dynamic_rendering: true,
            synchronization2: true,
            extended_dynamic_state: true,
            maintenance4: true,
        };

        // Check that required features are met
        let meets_requirements =
            (!required_features.descriptor_indexing || available_features.descriptor_indexing) &&
            (!required_features.timeline_semaphores || available_features.timeline_semaphores) &&
            (!required_features.buffer_device_address || available_features.buffer_device_address);

        assert!(meets_requirements);
    }
}

// =============================================================================
// SECTION 9: Edge Cases and Boundary Conditions
// =============================================================================

mod edge_cases {
    use super::*;

    #[test]
    fn version_zero_major() {
        let v = VulkanVersion::new(0, 1, 0);
        assert_eq!(v.major, 0);
        assert!(!v.is_at_least(1, 0));
    }

    #[test]
    fn version_large_patch_number() {
        let v = VulkanVersion::new(1, 3, 4095);
        assert_eq!(v.patch, 4095);
        let raw = v.to_raw();
        let decoded = VulkanVersion::from_raw(raw);
        assert_eq!(v, decoded);
    }

    #[test]
    fn version_raw_zero() {
        let v = VulkanVersion::from_raw(0);
        assert_eq!(v.major, 0);
        assert_eq!(v.minor, 0);
        assert_eq!(v.patch, 0);
    }

    #[test]
    fn version_raw_max() {
        let v = VulkanVersion::from_raw(0xFFFFFFFF);
        // Should decode without panicking
        let _ = v.major;
        let _ = v.minor;
        let _ = v.patch;
    }

    #[test]
    fn features_summary_only_one_feature() {
        let features = VulkanFeatures {
            ray_tracing: true,
            ray_query: false,
            descriptor_indexing: false,
            timeline_semaphores: false,
            buffer_device_address: false,
            mesh_shading: false,
            dynamic_rendering: false,
            synchronization2: false,
            extended_dynamic_state: false,
            maintenance4: false,
        };
        let summary = features.summary();
        assert_eq!(summary, "RT-Pipeline");
    }

    #[test]
    fn features_partial_rt_with_bda() {
        // Ray tracing requires BDA
        let features = VulkanFeatures {
            ray_tracing: true,
            ray_query: false,
            descriptor_indexing: false,
            timeline_semaphores: false,
            buffer_device_address: true,
            mesh_shading: false,
            dynamic_rendering: false,
            synchronization2: false,
            extended_dynamic_state: false,
            maintenance4: false,
        };
        assert!(features.supports_rt_pipeline());
        assert_eq!(features.ray_tracing_tier(), VulkanRayTracingTier::Full);
    }

    #[test]
    fn features_rt_without_bda_not_full_tier() {
        // Ray tracing flag set but no BDA means no full RT pipeline
        let features = VulkanFeatures {
            ray_tracing: true,
            ray_query: false,
            descriptor_indexing: false,
            timeline_semaphores: false,
            buffer_device_address: false,
            mesh_shading: false,
            dynamic_rendering: false,
            synchronization2: false,
            extended_dynamic_state: false,
            maintenance4: false,
        };
        assert!(!features.supports_rt_pipeline());
        // Still counts as "any RT" because ray_tracing is set
        assert!(features.supports_any_rt());
    }

    #[test]
    fn device_type_hash_distinct() {
        use std::hash::{Hash, Hasher};
        use std::collections::hash_map::DefaultHasher;

        fn hash_it<H: Hash>(h: &H) -> u64 {
            let mut hasher = DefaultHasher::new();
            h.hash(&mut hasher);
            hasher.finish()
        }

        let hashes: Vec<_> = [
            VulkanDeviceType::DiscreteGpu,
            VulkanDeviceType::IntegratedGpu,
            VulkanDeviceType::VirtualGpu,
            VulkanDeviceType::Cpu,
            VulkanDeviceType::Other,
        ].iter().map(hash_it).collect();

        // All hashes should be distinct
        let unique: HashSet<_> = hashes.iter().collect();
        assert_eq!(unique.len(), hashes.len());
    }

    #[test]
    fn version_hash_distinct() {
        use std::hash::{Hash, Hasher};
        use std::collections::hash_map::DefaultHasher;

        fn hash_it<H: Hash>(h: &H) -> u64 {
            let mut hasher = DefaultHasher::new();
            h.hash(&mut hasher);
            hasher.finish()
        }

        let hashes: Vec<_> = [
            VulkanVersion::V1_0,
            VulkanVersion::V1_1,
            VulkanVersion::V1_2,
            VulkanVersion::V1_3,
        ].iter().map(hash_it).collect();

        let unique: HashSet<_> = hashes.iter().collect();
        assert_eq!(unique.len(), hashes.len());
    }

    #[test]
    fn tier_hash_distinct() {
        use std::hash::{Hash, Hasher};
        use std::collections::hash_map::DefaultHasher;

        fn hash_it<H: Hash>(h: &H) -> u64 {
            let mut hasher = DefaultHasher::new();
            h.hash(&mut hasher);
            hasher.finish()
        }

        let hashes: Vec<_> = [
            VulkanRayTracingTier::None,
            VulkanRayTracingTier::Query,
            VulkanRayTracingTier::Full,
        ].iter().map(hash_it).collect();

        let unique: HashSet<_> = hashes.iter().collect();
        assert_eq!(unique.len(), hashes.len());
    }

    #[test]
    fn vulkan_info_clone() {
        let info = VulkanInfo {
            version: VulkanVersion::V1_3,
            features: VulkanFeatures::default(),
            driver_name: "Test".to_string(),
            driver_version: 123,
            device_name: "Test GPU".to_string(),
            device_type: VulkanDeviceType::DiscreteGpu,
            vendor_id: 0x1234,
            device_id: 0x5678,
        };

        let cloned = info.clone();
        assert_eq!(info.version, cloned.version);
        assert_eq!(info.driver_name, cloned.driver_name);
        assert_eq!(info.device_name, cloned.device_name);
    }

    #[test]
    fn features_copy_trait() {
        let features = VulkanFeatures {
            ray_tracing: true,
            ray_query: true,
            descriptor_indexing: true,
            timeline_semaphores: true,
            buffer_device_address: true,
            mesh_shading: true,
            dynamic_rendering: true,
            synchronization2: true,
            extended_dynamic_state: true,
            maintenance4: true,
        };

        let copied = features;
        assert_eq!(features, copied);
    }

    #[test]
    fn version_copy_trait() {
        let version = VulkanVersion::V1_3;
        let copied = version;
        assert_eq!(version, copied);
    }

    #[test]
    fn device_type_copy_trait() {
        let device_type = VulkanDeviceType::DiscreteGpu;
        let copied = device_type;
        assert_eq!(device_type, copied);
    }

    #[test]
    fn tier_copy_trait() {
        let tier = VulkanRayTracingTier::Full;
        let copied = tier;
        assert_eq!(tier, copied);
    }
}

// =============================================================================
// SECTION 10: Debug and Display Trait Coverage
// =============================================================================

mod debug_display_coverage {
    use super::*;

    #[test]
    fn vulkan_info_debug() {
        let info = VulkanInfo::default();
        let debug = format!("{:?}", info);
        assert!(debug.contains("VulkanInfo"));
    }

    #[test]
    fn vulkan_features_debug() {
        let features = VulkanFeatures::default();
        let debug = format!("{:?}", features);
        assert!(debug.contains("VulkanFeatures"));
    }

    #[test]
    fn vulkan_version_debug() {
        let version = VulkanVersion::V1_3;
        let debug = format!("{:?}", version);
        assert!(debug.contains("VulkanVersion"));
    }

    #[test]
    fn vulkan_device_type_debug() {
        let device_type = VulkanDeviceType::DiscreteGpu;
        let debug = format!("{:?}", device_type);
        assert!(debug.contains("DiscreteGpu"));
    }

    #[test]
    fn vulkan_ray_tracing_tier_debug() {
        let tier = VulkanRayTracingTier::Full;
        let debug = format!("{:?}", tier);
        assert!(debug.contains("Full"));
    }

    #[test]
    fn version_display_format() {
        let version = VulkanVersion::new(1, 3, 275);
        let display = format!("{}", version);
        assert_eq!(display, "1.3.275");
    }

    #[test]
    fn device_type_display_matches_name() {
        for variant in [
            VulkanDeviceType::DiscreteGpu,
            VulkanDeviceType::IntegratedGpu,
            VulkanDeviceType::VirtualGpu,
            VulkanDeviceType::Cpu,
            VulkanDeviceType::Other,
        ] {
            assert_eq!(format!("{}", variant), variant.name());
        }
    }

    #[test]
    fn tier_display_matches_name() {
        for tier in [
            VulkanRayTracingTier::None,
            VulkanRayTracingTier::Query,
            VulkanRayTracingTier::Full,
        ] {
            assert_eq!(format!("{}", tier), tier.name());
        }
    }
}
