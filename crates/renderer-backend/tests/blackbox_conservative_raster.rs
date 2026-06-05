// SPDX-License-Identifier: MIT
//
// blackbox_conservative_raster.rs -- Blackbox tests for T-WGPU-P3.4.3 Conservative Rasterization.
//
// CLEANROOM: No src/ access beyond the public API exported by the crate.
// Tests use only the documented public types and functions:
//
//   - ConservativeRasterization -- Core configuration struct
//   - ConservativeRasterBuilder -- Fluent builder for configuration creation
//   - ConservativeRasterError -- Error type for validation failures
//   - ConservativeRasterInfo -- Metadata about use cases
//   - ConservativeUseCase -- Enum of predefined use cases
//   - CONSERVATIVE_RASTERIZATION_FEATURE -- wgpu feature constant
//   - CONSERVATIVE_RASTER_USE_CASES -- Array of use case info
//   - is_supported -- Check adapter support
//   - is_enabled_on_device -- Check device feature
//   - required_features -- Get required wgpu features
//   - get_conservative_raster_info -- Lookup info by name
//   - get_info_for_use_case -- Lookup info by use case enum
//   - use_case_names -- Iterator over use case names
//
// ACCEPTANCE CRITERIA:
//   1. Feature check (CONSERVATIVE_RASTERIZATION)
//   2. PrimitiveState.conservative flag
//   3. Use case documentation
//
// Additional test categories:
//   4. API tests -- Public interface accessibility
//   5. Builder API -- Fluent interface
//   6. Enable/disable states -- State management
//   7. Hardware compatibility info -- Info struct coverage
//   8. Real-world scenarios -- Common usage patterns
//   9. Error handling -- All error conditions
//   10. wgpu interoperability -- Feature flag conversion
//
// Total target: 60+ tests across 10 categories

use renderer_backend::render_pipeline::{
    get_conservative_raster_info, get_info_for_use_case, is_enabled_on_device, is_supported,
    required_features, use_case_names, ConservativeRasterBuilder, ConservativeRasterError,
    ConservativeRasterInfo, ConservativeRasterization, ConservativeUseCase,
    CONSERVATIVE_RASTERIZATION_FEATURE, CONSERVATIVE_RASTER_USE_CASES,
};
use std::collections::HashSet;

// =============================================================================
// CATEGORY 1: API SURFACE TESTS
// =============================================================================
// Verify all public types and functions are accessible from the blackbox.

#[test]
fn test_api_conservative_rasterization_accessible() {
    let config = ConservativeRasterization::new();
    assert!(!config.enabled());
}

#[test]
fn test_api_conservative_raster_builder_accessible() {
    let builder = ConservativeRasterBuilder::new();
    let config = builder.build();
    assert!(!config.enabled());
}

#[test]
fn test_api_conservative_raster_error_accessible() {
    // Error type is accessible, though we cannot create the error directly
    // without a device context. Verify the type exists in the API.
    let _: fn() -> ConservativeRasterError = || ConservativeRasterError::FeatureNotEnabled;
    let _: fn() -> ConservativeRasterError = || ConservativeRasterError::FeatureNotSupported;
}

#[test]
fn test_api_conservative_use_case_accessible() {
    let use_case = ConservativeUseCase::Voxelization;
    assert_eq!(use_case.name(), "Voxelization");
}

#[test]
fn test_api_conservative_raster_info_accessible() {
    let info = get_conservative_raster_info("GPU Voxelization");
    assert!(info.is_some());
}

#[test]
fn test_api_conservative_raster_use_cases_accessible() {
    assert!(!CONSERVATIVE_RASTER_USE_CASES.is_empty());
    assert_eq!(CONSERVATIVE_RASTER_USE_CASES.len(), 7);
}

#[test]
fn test_api_feature_constant_accessible() {
    let feature = CONSERVATIVE_RASTERIZATION_FEATURE;
    assert_eq!(feature, wgpu::Features::CONSERVATIVE_RASTERIZATION);
}

#[test]
fn test_api_required_features_accessible() {
    let features = required_features();
    assert!(features.contains(wgpu::Features::CONSERVATIVE_RASTERIZATION));
}

#[test]
fn test_api_use_case_names_accessible() {
    let names: Vec<_> = use_case_names().collect();
    assert!(!names.is_empty());
}

// =============================================================================
// CATEGORY 2: FEATURE CHECK (CONSERVATIVE_RASTERIZATION)
// =============================================================================
// Tests for wgpu feature constant and feature detection.

#[test]
fn test_feature_constant_matches_wgpu() {
    assert_eq!(
        CONSERVATIVE_RASTERIZATION_FEATURE,
        wgpu::Features::CONSERVATIVE_RASTERIZATION
    );
}

#[test]
fn test_required_features_contains_conservative_rasterization() {
    let features = required_features();
    assert!(features.contains(wgpu::Features::CONSERVATIVE_RASTERIZATION));
}

#[test]
fn test_required_features_is_only_conservative_rasterization() {
    // The required_features() should return exactly CONSERVATIVE_RASTERIZATION
    let features = required_features();
    assert_eq!(features, wgpu::Features::CONSERVATIVE_RASTERIZATION);
}

#[test]
fn test_is_supported_function_exists() {
    // We cannot test actual adapter support without GPU context,
    // but verify the function signature is correct.
    fn check_signature(_f: fn(&wgpu::Adapter) -> bool) {}
    check_signature(is_supported);
}

#[test]
fn test_is_enabled_on_device_function_exists() {
    // Verify the function signature is correct.
    fn check_signature(_f: fn(&wgpu::Device) -> bool) {}
    check_signature(is_enabled_on_device);
}

// =============================================================================
// CATEGORY 3: PRIMITIVESTATE.CONSERVATIVE FLAG
// =============================================================================
// Tests for the boolean flag conversion to wgpu's PrimitiveState.

#[test]
fn test_as_wgpu_flag_enabled() {
    let config = ConservativeRasterization::enabled_config();
    assert!(config.as_wgpu_flag());
}

#[test]
fn test_as_wgpu_flag_disabled() {
    let config = ConservativeRasterization::disabled();
    assert!(!config.as_wgpu_flag());
}

#[test]
fn test_as_wgpu_flag_default() {
    let config = ConservativeRasterization::default();
    assert!(!config.as_wgpu_flag());
}

#[test]
fn test_as_wgpu_flag_voxelization() {
    let config = ConservativeRasterization::voxelization();
    assert!(config.as_wgpu_flag());
}

#[test]
fn test_as_wgpu_flag_after_enable() {
    let config = ConservativeRasterization::new().enable();
    assert!(config.as_wgpu_flag());
}

#[test]
fn test_as_wgpu_flag_after_disable() {
    let config = ConservativeRasterization::voxelization().disable();
    assert!(!config.as_wgpu_flag());
}

#[test]
fn test_wgpu_primitive_state_integration() {
    // Demonstrate how as_wgpu_flag() integrates with wgpu's PrimitiveState
    let config = ConservativeRasterization::voxelization();
    let primitive_state = wgpu::PrimitiveState {
        conservative: config.as_wgpu_flag(),
        ..Default::default()
    };
    assert!(primitive_state.conservative);
}

#[test]
fn test_wgpu_primitive_state_disabled() {
    let config = ConservativeRasterization::disabled();
    let primitive_state = wgpu::PrimitiveState {
        conservative: config.as_wgpu_flag(),
        ..Default::default()
    };
    assert!(!primitive_state.conservative);
}

// =============================================================================
// CATEGORY 4: USE CASE DOCUMENTATION
// =============================================================================
// Tests verifying all use cases are documented with info.

#[test]
fn test_all_use_cases_have_info() {
    for info in &CONSERVATIVE_RASTER_USE_CASES {
        assert!(!info.name.is_empty(), "Use case has empty name");
        assert!(!info.description.is_empty(), "Use case {} has empty description", info.name);
        assert!(!info.benefits.is_empty(), "Use case {} has no benefits", info.name);
        assert!(!info.performance_notes.is_empty(), "Use case {} has no performance notes", info.name);
    }
}

#[test]
fn test_use_case_count() {
    assert_eq!(CONSERVATIVE_RASTER_USE_CASES.len(), 7);
}

#[test]
fn test_use_case_voxelization_documented() {
    let info = get_conservative_raster_info("GPU Voxelization")
        .expect("GPU Voxelization info should exist");
    assert_eq!(info.use_case, ConservativeUseCase::Voxelization);
    assert!(info.benefits.iter().any(|b| b.contains("voxel") || b.contains("geometry")));
}

#[test]
fn test_use_case_occlusion_culling_documented() {
    let info = get_conservative_raster_info("Software Occlusion Culling")
        .expect("Occlusion Culling info should exist");
    assert_eq!(info.use_case, ConservativeUseCase::OcclusionCulling);
    assert!(info.benefits.iter().any(|b| b.contains("triangle") || b.contains("coverage")));
}

#[test]
fn test_use_case_collision_detection_documented() {
    let info = get_conservative_raster_info("Collision Detection")
        .expect("Collision Detection info should exist");
    assert_eq!(info.use_case, ConservativeUseCase::CollisionDetection);
}

#[test]
fn test_use_case_visibility_buffer_documented() {
    let info = get_conservative_raster_info("Visibility Buffer")
        .expect("Visibility Buffer info should exist");
    assert_eq!(info.use_case, ConservativeUseCase::VisibilityBuffer);
}

#[test]
fn test_use_case_shadow_mapping_documented() {
    let info = get_conservative_raster_info("Shadow Mapping")
        .expect("Shadow Mapping info should exist");
    assert_eq!(info.use_case, ConservativeUseCase::ShadowMapping);
}

#[test]
fn test_use_case_ray_tracing_prep_documented() {
    let info = get_conservative_raster_info("Ray Tracing Preparation")
        .expect("Ray Tracing Preparation info should exist");
    assert_eq!(info.use_case, ConservativeUseCase::RayTracingPrep);
}

#[test]
fn test_use_case_pathfinding_documented() {
    let info = get_conservative_raster_info("Pathfinding/Navigation")
        .expect("Pathfinding/Navigation info should exist");
    assert_eq!(info.use_case, ConservativeUseCase::Pathfinding);
}

#[test]
fn test_get_info_for_use_case_all_variants() {
    for use_case in [
        ConservativeUseCase::Voxelization,
        ConservativeUseCase::OcclusionCulling,
        ConservativeUseCase::CollisionDetection,
        ConservativeUseCase::VisibilityBuffer,
        ConservativeUseCase::ShadowMapping,
        ConservativeUseCase::RayTracingPrep,
        ConservativeUseCase::Pathfinding,
    ] {
        let info = get_info_for_use_case(use_case);
        assert!(
            info.is_some(),
            "No info found for use case: {:?}",
            use_case
        );
    }
}

#[test]
fn test_get_info_for_custom_returns_none() {
    // Custom use case doesn't have predefined info
    let info = get_info_for_use_case(ConservativeUseCase::Custom);
    assert!(info.is_none());
}

#[test]
fn test_use_case_names_iterator() {
    let names: Vec<_> = use_case_names().collect();
    assert_eq!(names.len(), 7);
    assert!(names.contains(&"GPU Voxelization"));
    assert!(names.contains(&"Software Occlusion Culling"));
    assert!(names.contains(&"Collision Detection"));
    assert!(names.contains(&"Visibility Buffer"));
    assert!(names.contains(&"Shadow Mapping"));
    assert!(names.contains(&"Ray Tracing Preparation"));
    assert!(names.contains(&"Pathfinding/Navigation"));
}

#[test]
fn test_use_case_enum_name_method() {
    assert_eq!(ConservativeUseCase::Voxelization.name(), "Voxelization");
    assert_eq!(ConservativeUseCase::OcclusionCulling.name(), "Occlusion Culling");
    assert_eq!(ConservativeUseCase::CollisionDetection.name(), "Collision Detection");
    assert_eq!(ConservativeUseCase::VisibilityBuffer.name(), "Visibility Buffer");
    assert_eq!(ConservativeUseCase::ShadowMapping.name(), "Shadow Mapping");
    assert_eq!(ConservativeUseCase::RayTracingPrep.name(), "Ray Tracing Prep");
    assert_eq!(ConservativeUseCase::Pathfinding.name(), "Pathfinding");
    assert_eq!(ConservativeUseCase::Custom.name(), "Custom");
}

#[test]
fn test_use_case_enum_description_method() {
    assert!(!ConservativeUseCase::Voxelization.description().is_empty());
    assert!(!ConservativeUseCase::OcclusionCulling.description().is_empty());
    assert!(!ConservativeUseCase::CollisionDetection.description().is_empty());
    assert!(!ConservativeUseCase::VisibilityBuffer.description().is_empty());
    assert!(!ConservativeUseCase::ShadowMapping.description().is_empty());
    assert!(!ConservativeUseCase::RayTracingPrep.description().is_empty());
    assert!(!ConservativeUseCase::Pathfinding.description().is_empty());
    assert!(!ConservativeUseCase::Custom.description().is_empty());
}

// =============================================================================
// CATEGORY 5: BUILDER API (ConservativeRasterBuilder)
// =============================================================================
// Tests for the fluent builder interface.

#[test]
fn test_builder_new_creates_default() {
    let config = ConservativeRasterBuilder::new().build();
    assert!(!config.enabled());
    assert!(config.use_case().is_none());
}

#[test]
fn test_builder_enable() {
    let config = ConservativeRasterBuilder::new().enable().build();
    assert!(config.enabled());
}

#[test]
fn test_builder_disable() {
    let config = ConservativeRasterBuilder::new()
        .enable()
        .disable()
        .build();
    assert!(!config.enabled());
}

#[test]
fn test_builder_enabled_explicit_true() {
    let config = ConservativeRasterBuilder::new().enabled(true).build();
    assert!(config.enabled());
}

#[test]
fn test_builder_enabled_explicit_false() {
    let config = ConservativeRasterBuilder::new()
        .enable()
        .enabled(false)
        .build();
    assert!(!config.enabled());
}

#[test]
fn test_builder_for_voxelization() {
    let config = ConservativeRasterBuilder::new()
        .enable()
        .for_voxelization()
        .build();
    assert!(config.enabled());
    assert_eq!(config.use_case(), Some(ConservativeUseCase::Voxelization));
}

#[test]
fn test_builder_for_occlusion_culling() {
    let config = ConservativeRasterBuilder::new()
        .enable()
        .for_occlusion_culling()
        .build();
    assert_eq!(config.use_case(), Some(ConservativeUseCase::OcclusionCulling));
}

#[test]
fn test_builder_for_collision_detection() {
    let config = ConservativeRasterBuilder::new()
        .enable()
        .for_collision_detection()
        .build();
    assert_eq!(config.use_case(), Some(ConservativeUseCase::CollisionDetection));
}

#[test]
fn test_builder_for_visibility_buffer() {
    let config = ConservativeRasterBuilder::new()
        .enable()
        .for_visibility_buffer()
        .build();
    assert_eq!(config.use_case(), Some(ConservativeUseCase::VisibilityBuffer));
}

#[test]
fn test_builder_for_shadow_mapping() {
    let config = ConservativeRasterBuilder::new()
        .enable()
        .for_shadow_mapping()
        .build();
    assert_eq!(config.use_case(), Some(ConservativeUseCase::ShadowMapping));
}

#[test]
fn test_builder_for_ray_tracing_prep() {
    let config = ConservativeRasterBuilder::new()
        .enable()
        .for_ray_tracing_prep()
        .build();
    assert_eq!(config.use_case(), Some(ConservativeUseCase::RayTracingPrep));
}

#[test]
fn test_builder_for_pathfinding() {
    let config = ConservativeRasterBuilder::new()
        .enable()
        .for_pathfinding()
        .build();
    assert_eq!(config.use_case(), Some(ConservativeUseCase::Pathfinding));
}

#[test]
fn test_builder_for_custom() {
    let config = ConservativeRasterBuilder::new()
        .enable()
        .for_custom()
        .build();
    assert_eq!(config.use_case(), Some(ConservativeUseCase::Custom));
}

#[test]
fn test_builder_use_case_explicit() {
    let config = ConservativeRasterBuilder::new()
        .enable()
        .use_case(ConservativeUseCase::Voxelization)
        .build();
    assert_eq!(config.use_case(), Some(ConservativeUseCase::Voxelization));
}

#[test]
fn test_builder_from_config() {
    let original = ConservativeRasterization::voxelization();
    let modified = ConservativeRasterBuilder::from_config(original)
        .disable()
        .build();

    assert!(!modified.enabled());
    assert_eq!(modified.use_case(), Some(ConservativeUseCase::Voxelization));
}

#[test]
fn test_builder_default_trait() {
    let builder: ConservativeRasterBuilder = Default::default();
    let config = builder.build();
    assert!(!config.enabled());
}

#[test]
fn test_builder_fluent_chain() {
    let config = ConservativeRasterBuilder::new()
        .enable()
        .for_voxelization()
        .disable()
        .enable()
        .for_shadow_mapping()
        .build();

    assert!(config.enabled());
    assert_eq!(config.use_case(), Some(ConservativeUseCase::ShadowMapping));
}

// =============================================================================
// CATEGORY 6: ENABLE/DISABLE STATES
// =============================================================================
// Tests for state management.

#[test]
fn test_default_is_disabled() {
    let config = ConservativeRasterization::default();
    assert!(!config.enabled());
    assert!(config.use_case().is_none());
}

#[test]
fn test_new_is_disabled() {
    let config = ConservativeRasterization::new();
    assert!(!config.enabled());
}

#[test]
fn test_disabled_constructor() {
    let config = ConservativeRasterization::disabled();
    assert!(!config.enabled());
}

#[test]
fn test_enabled_config_constructor() {
    let config = ConservativeRasterization::enabled_config();
    assert!(config.enabled());
    assert!(config.use_case().is_none());
}

#[test]
fn test_enable_method() {
    let config = ConservativeRasterization::new().enable();
    assert!(config.enabled());
}

#[test]
fn test_disable_method() {
    let config = ConservativeRasterization::voxelization().disable();
    assert!(!config.enabled());
}

#[test]
fn test_enable_disable_chain() {
    let config = ConservativeRasterization::new()
        .enable()
        .disable()
        .enable();
    assert!(config.enabled());
}

#[test]
fn test_with_use_case() {
    let config = ConservativeRasterization::new()
        .enable()
        .with_use_case(ConservativeUseCase::Voxelization);

    assert!(config.enabled());
    assert_eq!(config.use_case(), Some(ConservativeUseCase::Voxelization));
}

#[test]
fn test_without_use_case() {
    let config = ConservativeRasterization::voxelization()
        .without_use_case();

    assert!(config.enabled());
    assert!(config.use_case().is_none());
}

// =============================================================================
// CATEGORY 7: HARDWARE COMPATIBILITY INFO
// =============================================================================
// Tests for ConservativeRasterInfo struct coverage.

#[test]
fn test_info_struct_fields() {
    let info = &CONSERVATIVE_RASTER_USE_CASES[0];
    let _name: &str = info.name;
    let _description: &str = info.description;
    let _benefits: &[&str] = info.benefits;
    let _performance_notes: &str = info.performance_notes;
    let _use_case: ConservativeUseCase = info.use_case;
}

#[test]
fn test_all_info_entries_have_benefits() {
    for info in &CONSERVATIVE_RASTER_USE_CASES {
        assert!(
            info.benefits.len() >= 1,
            "Info '{}' should have at least one benefit",
            info.name
        );
    }
}

#[test]
fn test_all_info_entries_have_performance_notes() {
    for info in &CONSERVATIVE_RASTER_USE_CASES {
        assert!(
            info.performance_notes.len() > 10,
            "Info '{}' should have meaningful performance notes",
            info.name
        );
    }
}

#[test]
fn test_info_benefits_are_meaningful() {
    for info in &CONSERVATIVE_RASTER_USE_CASES {
        for benefit in info.benefits {
            assert!(
                benefit.len() > 5,
                "Benefit '{}' in '{}' is too short",
                benefit,
                info.name
            );
        }
    }
}

#[test]
fn test_get_conservative_raster_info_by_name() {
    let info = get_conservative_raster_info("GPU Voxelization");
    assert!(info.is_some());
    assert_eq!(info.unwrap().name, "GPU Voxelization");
}

#[test]
fn test_get_conservative_raster_info_not_found() {
    let info = get_conservative_raster_info("NonExistent Use Case");
    assert!(info.is_none());
}

#[test]
fn test_get_info_for_use_case_returns_matching() {
    let info = get_info_for_use_case(ConservativeUseCase::Voxelization).unwrap();
    assert_eq!(info.use_case, ConservativeUseCase::Voxelization);
    assert_eq!(info.name, "GPU Voxelization");
}

// =============================================================================
// CATEGORY 8: REAL-WORLD SCENARIOS
// =============================================================================
// Tests demonstrating common usage patterns.

#[test]
fn test_scenario_voxelization_pipeline() {
    // SVO/SDF construction requires conservative rasterization
    let config = ConservativeRasterization::voxelization();
    assert!(config.enabled());
    assert!(config.as_wgpu_flag());

    let primitive_state = wgpu::PrimitiveState {
        conservative: config.as_wgpu_flag(),
        topology: wgpu::PrimitiveTopology::TriangleList,
        ..Default::default()
    };
    assert!(primitive_state.conservative);
}

#[test]
fn test_scenario_occlusion_culling_buffer() {
    // Software occlusion culling preparation pass
    let config = ConservativeRasterization::occlusion_culling();
    assert!(config.enabled());
    assert_eq!(config.use_case(), Some(ConservativeUseCase::OcclusionCulling));
}

#[test]
fn test_scenario_collision_detection() {
    // GPU-accelerated physics broadphase
    let config = ConservativeRasterization::collision_detection();
    assert!(config.enabled());
}

#[test]
fn test_scenario_visibility_buffer() {
    // Forward+ rendering visibility pass
    let config = ConservativeRasterization::visibility_buffer();
    assert!(config.enabled());
}

#[test]
fn test_scenario_shadow_mapping() {
    // Improved shadow map precision for thin geometry
    let config = ConservativeRasterization::shadow_mapping();
    assert!(config.enabled());
}

#[test]
fn test_scenario_ray_tracing_prep() {
    // BVH construction acceleration
    let config = ConservativeRasterization::ray_tracing_prep();
    assert!(config.enabled());
}

#[test]
fn test_scenario_pathfinding() {
    // Navigation mesh rasterization
    let config = ConservativeRasterization::pathfinding();
    assert!(config.enabled());
}

#[test]
fn test_scenario_standard_rendering_no_conservative() {
    // Standard opaque rendering doesn't need conservative rasterization
    let config = ConservativeRasterization::disabled();
    assert!(!config.enabled());
    assert!(!config.as_wgpu_flag());
}

#[test]
fn test_scenario_custom_use_case() {
    // User-defined requirement
    let config = ConservativeRasterization::new()
        .enable()
        .with_use_case(ConservativeUseCase::Custom);

    assert!(config.enabled());
    assert_eq!(config.use_case(), Some(ConservativeUseCase::Custom));
}

#[test]
fn test_scenario_conditional_enable() {
    // Enable based on runtime condition
    let should_enable = true;
    let config = ConservativeRasterBuilder::new()
        .enabled(should_enable)
        .for_voxelization()
        .build();

    assert_eq!(config.enabled(), should_enable);
}

#[test]
fn test_scenario_modify_existing_config() {
    // Start from preset, modify for specific needs
    let base_config = ConservativeRasterization::voxelization();
    let modified = ConservativeRasterBuilder::from_config(base_config)
        .for_shadow_mapping() // Change use case
        .build();

    assert!(modified.enabled());
    assert_eq!(modified.use_case(), Some(ConservativeUseCase::ShadowMapping));
}

// =============================================================================
// CATEGORY 9: ERROR HANDLING
// =============================================================================
// Tests for all error conditions.

#[test]
fn test_error_feature_not_enabled_display() {
    let err = ConservativeRasterError::FeatureNotEnabled;
    let msg = format!("{}", err);
    assert!(msg.contains("CONSERVATIVE_RASTERIZATION"));
    assert!(msg.contains("not enabled"));
}

#[test]
fn test_error_feature_not_supported_display() {
    let err = ConservativeRasterError::FeatureNotSupported;
    let msg = format!("{}", err);
    assert!(msg.contains("not supported"));
}

#[test]
fn test_error_is_std_error() {
    fn assert_error<E: std::error::Error>() {}
    assert_error::<ConservativeRasterError>();
}

#[test]
fn test_error_debug_trait() {
    let err = ConservativeRasterError::FeatureNotEnabled;
    let debug = format!("{:?}", err);
    assert!(debug.contains("FeatureNotEnabled"));
}

#[test]
fn test_error_clone() {
    let err = ConservativeRasterError::FeatureNotEnabled;
    let cloned = err.clone();
    assert_eq!(err, cloned);
}

#[test]
fn test_error_copy() {
    let err = ConservativeRasterError::FeatureNotEnabled;
    let copied = err;
    assert_eq!(err, copied);
}

#[test]
fn test_error_partial_eq() {
    assert_eq!(
        ConservativeRasterError::FeatureNotEnabled,
        ConservativeRasterError::FeatureNotEnabled
    );
    assert_ne!(
        ConservativeRasterError::FeatureNotEnabled,
        ConservativeRasterError::FeatureNotSupported
    );
}

// =============================================================================
// CATEGORY 10: WGPU INTEROPERABILITY
// =============================================================================
// Tests for wgpu feature flag conversion and integration.

#[test]
fn test_wgpu_feature_flag_equality() {
    assert_eq!(
        CONSERVATIVE_RASTERIZATION_FEATURE,
        wgpu::Features::CONSERVATIVE_RASTERIZATION
    );
}

#[test]
fn test_required_features_usable_in_device_descriptor() {
    // Verify required_features() returns a wgpu::Features that can be used
    let features = required_features();
    let _descriptor = wgpu::DeviceDescriptor {
        required_features: features,
        ..Default::default()
    };
}

#[test]
fn test_required_features_combinable_with_other_features() {
    let features = required_features() | wgpu::Features::POLYGON_MODE_LINE;
    assert!(features.contains(wgpu::Features::CONSERVATIVE_RASTERIZATION));
    assert!(features.contains(wgpu::Features::POLYGON_MODE_LINE));
}

#[test]
fn test_primitive_state_conservative_flag_type() {
    // Verify as_wgpu_flag() returns bool compatible with wgpu::PrimitiveState.conservative
    let config = ConservativeRasterization::voxelization();
    let flag: bool = config.as_wgpu_flag();
    assert!(flag);

    let primitive = wgpu::PrimitiveState {
        conservative: flag,
        ..Default::default()
    };
    assert!(primitive.conservative);
}

// =============================================================================
// CATEGORY 11: PRESET TESTS
// =============================================================================
// Tests for all preset constructors.

#[test]
fn test_preset_voxelization() {
    let config = ConservativeRasterization::voxelization();
    assert!(config.enabled());
    assert_eq!(config.use_case(), Some(ConservativeUseCase::Voxelization));
}

#[test]
fn test_preset_occlusion_culling() {
    let config = ConservativeRasterization::occlusion_culling();
    assert!(config.enabled());
    assert_eq!(config.use_case(), Some(ConservativeUseCase::OcclusionCulling));
}

#[test]
fn test_preset_collision_detection() {
    let config = ConservativeRasterization::collision_detection();
    assert!(config.enabled());
    assert_eq!(config.use_case(), Some(ConservativeUseCase::CollisionDetection));
}

#[test]
fn test_preset_visibility_buffer() {
    let config = ConservativeRasterization::visibility_buffer();
    assert!(config.enabled());
    assert_eq!(config.use_case(), Some(ConservativeUseCase::VisibilityBuffer));
}

#[test]
fn test_preset_shadow_mapping() {
    let config = ConservativeRasterization::shadow_mapping();
    assert!(config.enabled());
    assert_eq!(config.use_case(), Some(ConservativeUseCase::ShadowMapping));
}

#[test]
fn test_preset_ray_tracing_prep() {
    let config = ConservativeRasterization::ray_tracing_prep();
    assert!(config.enabled());
    assert_eq!(config.use_case(), Some(ConservativeUseCase::RayTracingPrep));
}

#[test]
fn test_preset_pathfinding() {
    let config = ConservativeRasterization::pathfinding();
    assert!(config.enabled());
    assert_eq!(config.use_case(), Some(ConservativeUseCase::Pathfinding));
}

// =============================================================================
// CATEGORY 12: TRAITS AND ATTRIBUTES
// =============================================================================
// Tests for derived traits.

#[test]
fn test_conservative_rasterization_debug() {
    let config = ConservativeRasterization::voxelization();
    let debug = format!("{:?}", config);
    assert!(debug.contains("ConservativeRasterization"));
}

#[test]
fn test_conservative_rasterization_clone() {
    let original = ConservativeRasterization::voxelization();
    let cloned = original.clone();
    assert_eq!(original, cloned);
}

#[test]
fn test_conservative_rasterization_copy() {
    let original = ConservativeRasterization::voxelization();
    let copied = original;
    assert_eq!(original, copied);
}

#[test]
fn test_conservative_rasterization_partial_eq() {
    let config1 = ConservativeRasterization::voxelization();
    let config2 = ConservativeRasterization::voxelization();
    let config3 = ConservativeRasterization::occlusion_culling();

    assert_eq!(config1, config2);
    assert_ne!(config1, config3);
}

#[test]
fn test_conservative_rasterization_hash() {
    let mut set = HashSet::new();
    set.insert(ConservativeRasterization::voxelization());
    set.insert(ConservativeRasterization::occlusion_culling());

    assert!(set.contains(&ConservativeRasterization::voxelization()));
    assert!(set.contains(&ConservativeRasterization::occlusion_culling()));
    assert!(!set.contains(&ConservativeRasterization::disabled()));
}

#[test]
fn test_conservative_rasterization_send() {
    fn assert_send<T: Send>() {}
    assert_send::<ConservativeRasterization>();
}

#[test]
fn test_conservative_rasterization_sync() {
    fn assert_sync<T: Sync>() {}
    assert_sync::<ConservativeRasterization>();
}

#[test]
fn test_builder_debug() {
    let builder = ConservativeRasterBuilder::new();
    let debug = format!("{:?}", builder);
    assert!(debug.contains("ConservativeRasterBuilder"));
}

#[test]
fn test_builder_clone() {
    let builder = ConservativeRasterBuilder::new().enable().for_voxelization();
    let cloned = builder.clone();
    assert_eq!(builder.build(), cloned.build());
}

#[test]
fn test_use_case_enum_debug() {
    let use_case = ConservativeUseCase::Voxelization;
    let debug = format!("{:?}", use_case);
    assert!(debug.contains("Voxelization"));
}

#[test]
fn test_use_case_enum_clone() {
    let use_case = ConservativeUseCase::Voxelization;
    let cloned = use_case.clone();
    assert_eq!(use_case, cloned);
}

#[test]
fn test_use_case_enum_copy() {
    let use_case = ConservativeUseCase::Voxelization;
    let copied = use_case;
    assert_eq!(use_case, copied);
}

#[test]
fn test_use_case_enum_hash() {
    let mut set = HashSet::new();
    set.insert(ConservativeUseCase::Voxelization);
    set.insert(ConservativeUseCase::OcclusionCulling);

    assert!(set.contains(&ConservativeUseCase::Voxelization));
    assert!(set.contains(&ConservativeUseCase::OcclusionCulling));
    assert!(!set.contains(&ConservativeUseCase::Custom));
}

// =============================================================================
// CATEGORY 13: DISPLAY TESTS
// =============================================================================
// Tests for Display trait implementation.

#[test]
fn test_display_disabled() {
    let config = ConservativeRasterization::disabled();
    let display = format!("{}", config);
    assert!(display.contains("disabled"));
}

#[test]
fn test_display_enabled_no_use_case() {
    let config = ConservativeRasterization::enabled_config();
    let display = format!("{}", config);
    assert!(display.contains("enabled"));
    assert!(!display.contains("Voxelization"));
}

#[test]
fn test_display_enabled_with_use_case() {
    let config = ConservativeRasterization::voxelization();
    let display = format!("{}", config);
    assert!(display.contains("enabled"));
    assert!(display.contains("Voxelization"));
}

// =============================================================================
// CATEGORY 14: EDGE CASES
// =============================================================================
// Tests for boundary conditions and edge cases.

#[test]
fn test_builder_overwrite_use_case() {
    let config = ConservativeRasterBuilder::new()
        .for_voxelization()
        .for_shadow_mapping()
        .for_occlusion_culling()
        .build();

    assert_eq!(config.use_case(), Some(ConservativeUseCase::OcclusionCulling));
}

#[test]
fn test_builder_overwrite_enabled() {
    let config = ConservativeRasterBuilder::new()
        .enable()
        .disable()
        .enable()
        .build();

    assert!(config.enabled());
}

#[test]
fn test_config_use_case_override() {
    let config = ConservativeRasterization::voxelization()
        .with_use_case(ConservativeUseCase::ShadowMapping);

    assert_eq!(config.use_case(), Some(ConservativeUseCase::ShadowMapping));
}

#[test]
fn test_disabled_config_with_use_case() {
    // Can have a use case tag even when disabled
    let config = ConservativeRasterization::new()
        .with_use_case(ConservativeUseCase::Voxelization);

    assert!(!config.enabled());
    assert_eq!(config.use_case(), Some(ConservativeUseCase::Voxelization));
}

#[test]
fn test_enabled_config_without_use_case() {
    let config = ConservativeRasterization::enabled_config();
    assert!(config.enabled());
    assert!(config.use_case().is_none());
}

// =============================================================================
// ACCEPTANCE CRITERIA VERIFICATION
// =============================================================================

#[test]
fn test_acceptance_criteria_feature_check() {
    // AC: Feature check (CONSERVATIVE_RASTERIZATION)
    assert_eq!(
        CONSERVATIVE_RASTERIZATION_FEATURE,
        wgpu::Features::CONSERVATIVE_RASTERIZATION
    );
    let features = required_features();
    assert!(features.contains(wgpu::Features::CONSERVATIVE_RASTERIZATION));
}

#[test]
fn test_acceptance_criteria_primitive_state_flag() {
    // AC: PrimitiveState.conservative flag
    let enabled_config = ConservativeRasterization::voxelization();
    let disabled_config = ConservativeRasterization::disabled();

    assert!(enabled_config.as_wgpu_flag());
    assert!(!disabled_config.as_wgpu_flag());

    // Integration with wgpu::PrimitiveState
    let primitive = wgpu::PrimitiveState {
        conservative: enabled_config.as_wgpu_flag(),
        ..Default::default()
    };
    assert!(primitive.conservative);
}

#[test]
fn test_acceptance_criteria_use_case_documentation() {
    // AC: Use case documentation
    assert_eq!(CONSERVATIVE_RASTER_USE_CASES.len(), 7);

    for info in &CONSERVATIVE_RASTER_USE_CASES {
        assert!(!info.name.is_empty());
        assert!(!info.description.is_empty());
        assert!(!info.benefits.is_empty());
        assert!(!info.performance_notes.is_empty());
    }

    // All use cases should be documented
    for use_case in [
        ConservativeUseCase::Voxelization,
        ConservativeUseCase::OcclusionCulling,
        ConservativeUseCase::CollisionDetection,
        ConservativeUseCase::VisibilityBuffer,
        ConservativeUseCase::ShadowMapping,
        ConservativeUseCase::RayTracingPrep,
        ConservativeUseCase::Pathfinding,
    ] {
        let info = get_info_for_use_case(use_case);
        assert!(info.is_some(), "Use case {:?} should have documentation", use_case);
    }
}
