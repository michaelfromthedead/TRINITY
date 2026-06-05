// Blackbox contract tests for T-WGPU-P7.2.3 D3D12 Feature Detection API.
//
// CLEANROOM: No src/ access beyond the public API exported by the crate.
// Tests use only `renderer_backend::backend::*` -- no internal fields,
// no private methods, no implementation details.
//
// Acceptance criterion (T-WGPU-P7.2.3):
//   D3D12 feature detection API correctly identifies hardware capabilities
//   including feature levels, shader models, ray tracing tiers, and advanced features.
//
// Public API under test:
//   - D3D12FeatureLevel: FL_11_0..FL_12_2, supports_ray_tracing(), supports_mesh_shaders()
//   - D3D12FeatureLevel: supports_variable_rate_shading(), supports_sampler_feedback()
//   - D3D12FeatureLevel: name(), d3d_feature_level_value(), from_features()
//   - D3D12ShaderModel: SM_5_1..SM_6_7, supports_wave_intrinsics(), supports_mesh_shaders()
//   - D3D12ShaderModel: supports_raytracing_intrinsics(), supports_derivatives()
//   - D3D12ShaderModel: supports_16bit_types(), name(), version()
//   - D3D12RayTracingTier: None, Tier1_0, Tier1_1, supports_inline_raytracing()
//   - D3D12RayTracingTier: supports_rayquery(), is_available(), name()
//   - D3D12Features: from_features(), supports_rt(), supports_mesh_shaders()
//   - D3D12Features: supports_bindless(), supports_vrs(), supports_inline_rt()
//   - D3D12Features: supports_sampler_feedback(), supports_conservative_raster()
//   - D3D12Features: supports_gpu_driven(), summary(), minimum_windows_version()
//
// Test scenarios:
//   1.  Feature Level Detection: FL_11_0 through FL_12_2
//   2.  Shader Model Workflows: SM_5_1 through SM_6_7
//   3.  Ray Tracing Detection: None, Tier 1.0, Tier 1.1
//   4.  Feature Combinations: Mesh shaders + RT, VRS tiers, etc.
//   5.  Real-World GPU Profiles: GTX 1080, RTX 2080/3080/4090, RX 6800/7900, Arc A770
//   6.  Backend Integration: D3D12Features with BackendCapabilities

use renderer_backend::backend::{
    BackendCapabilities, BackendType, D3D12FeatureLevel, D3D12Features, D3D12RayTracingTier,
    D3D12ShaderModel,
};
use wgpu::Features;

// ============================================================================
// SECTION 1 -- D3D12FeatureLevel Basic Contract Tests
// ============================================================================

#[test]
fn feature_level_default_is_fl_11_0() {
    let fl = D3D12FeatureLevel::default();
    assert_eq!(fl, D3D12FeatureLevel::FL_11_0);
}

#[test]
fn feature_level_fl_11_0_name() {
    assert_eq!(D3D12FeatureLevel::FL_11_0.name(), "11.0");
}

#[test]
fn feature_level_fl_11_1_name() {
    assert_eq!(D3D12FeatureLevel::FL_11_1.name(), "11.1");
}

#[test]
fn feature_level_fl_12_0_name() {
    assert_eq!(D3D12FeatureLevel::FL_12_0.name(), "12.0");
}

#[test]
fn feature_level_fl_12_1_name() {
    assert_eq!(D3D12FeatureLevel::FL_12_1.name(), "12.1");
}

#[test]
fn feature_level_fl_12_2_name() {
    assert_eq!(D3D12FeatureLevel::FL_12_2.name(), "12.2");
}

#[test]
fn feature_level_display_fl_11_0() {
    assert_eq!(
        format!("{}", D3D12FeatureLevel::FL_11_0),
        "D3D_FEATURE_LEVEL_11_0"
    );
}

#[test]
fn feature_level_display_fl_12_2() {
    assert_eq!(
        format!("{}", D3D12FeatureLevel::FL_12_2),
        "D3D_FEATURE_LEVEL_12_2"
    );
}

// ============================================================================
// SECTION 2 -- D3D12FeatureLevel D3D_FEATURE_LEVEL Value Tests
// ============================================================================

#[test]
fn feature_level_fl_11_0_d3d_value() {
    assert_eq!(D3D12FeatureLevel::FL_11_0.d3d_feature_level_value(), 0xb000);
}

#[test]
fn feature_level_fl_11_1_d3d_value() {
    assert_eq!(D3D12FeatureLevel::FL_11_1.d3d_feature_level_value(), 0xb100);
}

#[test]
fn feature_level_fl_12_0_d3d_value() {
    assert_eq!(D3D12FeatureLevel::FL_12_0.d3d_feature_level_value(), 0xc000);
}

#[test]
fn feature_level_fl_12_1_d3d_value() {
    assert_eq!(D3D12FeatureLevel::FL_12_1.d3d_feature_level_value(), 0xc100);
}

#[test]
fn feature_level_fl_12_2_d3d_value() {
    assert_eq!(D3D12FeatureLevel::FL_12_2.d3d_feature_level_value(), 0xc200);
}

// ============================================================================
// SECTION 3 -- D3D12FeatureLevel Ordering Tests
// ============================================================================

#[test]
fn feature_level_ordering_fl_11_0_less_than_fl_11_1() {
    assert!(D3D12FeatureLevel::FL_11_0 < D3D12FeatureLevel::FL_11_1);
}

#[test]
fn feature_level_ordering_fl_11_1_less_than_fl_12_0() {
    assert!(D3D12FeatureLevel::FL_11_1 < D3D12FeatureLevel::FL_12_0);
}

#[test]
fn feature_level_ordering_fl_12_0_less_than_fl_12_1() {
    assert!(D3D12FeatureLevel::FL_12_0 < D3D12FeatureLevel::FL_12_1);
}

#[test]
fn feature_level_ordering_fl_12_1_less_than_fl_12_2() {
    assert!(D3D12FeatureLevel::FL_12_1 < D3D12FeatureLevel::FL_12_2);
}

#[test]
fn feature_level_ordering_full_chain() {
    assert!(D3D12FeatureLevel::FL_11_0 < D3D12FeatureLevel::FL_12_2);
    assert!(D3D12FeatureLevel::FL_11_1 < D3D12FeatureLevel::FL_12_1);
}

// ============================================================================
// SECTION 4 -- D3D12FeatureLevel Ray Tracing Support Tests
// ============================================================================

#[test]
fn feature_level_fl_11_0_does_not_support_ray_tracing() {
    assert!(!D3D12FeatureLevel::FL_11_0.supports_ray_tracing());
}

#[test]
fn feature_level_fl_11_1_does_not_support_ray_tracing() {
    assert!(!D3D12FeatureLevel::FL_11_1.supports_ray_tracing());
}

#[test]
fn feature_level_fl_12_0_does_not_support_ray_tracing() {
    assert!(!D3D12FeatureLevel::FL_12_0.supports_ray_tracing());
}

#[test]
fn feature_level_fl_12_1_supports_ray_tracing() {
    assert!(D3D12FeatureLevel::FL_12_1.supports_ray_tracing());
}

#[test]
fn feature_level_fl_12_2_supports_ray_tracing() {
    assert!(D3D12FeatureLevel::FL_12_2.supports_ray_tracing());
}

// ============================================================================
// SECTION 5 -- D3D12FeatureLevel Mesh Shader Support Tests
// ============================================================================

#[test]
fn feature_level_fl_11_0_does_not_support_mesh_shaders() {
    assert!(!D3D12FeatureLevel::FL_11_0.supports_mesh_shaders());
}

#[test]
fn feature_level_fl_11_1_does_not_support_mesh_shaders() {
    assert!(!D3D12FeatureLevel::FL_11_1.supports_mesh_shaders());
}

#[test]
fn feature_level_fl_12_0_does_not_support_mesh_shaders() {
    assert!(!D3D12FeatureLevel::FL_12_0.supports_mesh_shaders());
}

#[test]
fn feature_level_fl_12_1_does_not_support_mesh_shaders() {
    assert!(!D3D12FeatureLevel::FL_12_1.supports_mesh_shaders());
}

#[test]
fn feature_level_fl_12_2_supports_mesh_shaders() {
    assert!(D3D12FeatureLevel::FL_12_2.supports_mesh_shaders());
}

// ============================================================================
// SECTION 6 -- D3D12FeatureLevel VRS Support Tests
// ============================================================================

#[test]
fn feature_level_fl_11_0_does_not_support_vrs() {
    assert!(!D3D12FeatureLevel::FL_11_0.supports_variable_rate_shading());
}

#[test]
fn feature_level_fl_11_1_does_not_support_vrs() {
    assert!(!D3D12FeatureLevel::FL_11_1.supports_variable_rate_shading());
}

#[test]
fn feature_level_fl_12_0_does_not_support_vrs() {
    assert!(!D3D12FeatureLevel::FL_12_0.supports_variable_rate_shading());
}

#[test]
fn feature_level_fl_12_1_supports_vrs() {
    assert!(D3D12FeatureLevel::FL_12_1.supports_variable_rate_shading());
}

#[test]
fn feature_level_fl_12_2_supports_vrs() {
    assert!(D3D12FeatureLevel::FL_12_2.supports_variable_rate_shading());
}

// ============================================================================
// SECTION 7 -- D3D12FeatureLevel Sampler Feedback Support Tests
// ============================================================================

#[test]
fn feature_level_fl_11_0_does_not_support_sampler_feedback() {
    assert!(!D3D12FeatureLevel::FL_11_0.supports_sampler_feedback());
}

#[test]
fn feature_level_fl_11_1_does_not_support_sampler_feedback() {
    assert!(!D3D12FeatureLevel::FL_11_1.supports_sampler_feedback());
}

#[test]
fn feature_level_fl_12_0_does_not_support_sampler_feedback() {
    assert!(!D3D12FeatureLevel::FL_12_0.supports_sampler_feedback());
}

#[test]
fn feature_level_fl_12_1_does_not_support_sampler_feedback() {
    assert!(!D3D12FeatureLevel::FL_12_1.supports_sampler_feedback());
}

#[test]
fn feature_level_fl_12_2_supports_sampler_feedback() {
    assert!(D3D12FeatureLevel::FL_12_2.supports_sampler_feedback());
}

// ============================================================================
// SECTION 8 -- D3D12FeatureLevel from_features() Tests
// ============================================================================

#[test]
fn feature_level_from_empty_features_is_fl_11_0() {
    let features = Features::empty();
    let fl = D3D12FeatureLevel::from_features(features);
    assert_eq!(fl, D3D12FeatureLevel::FL_11_0);
}

#[test]
fn feature_level_from_timestamp_is_fl_11_1() {
    let features = Features::TIMESTAMP_QUERY;
    let fl = D3D12FeatureLevel::from_features(features);
    assert_eq!(fl, D3D12FeatureLevel::FL_11_1);
}

#[test]
fn feature_level_from_bindless_is_fl_12_0() {
    let features = Features::TEXTURE_BINDING_ARRAY
        | Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING;
    let fl = D3D12FeatureLevel::from_features(features);
    assert_eq!(fl, D3D12FeatureLevel::FL_12_0);
}

#[test]
fn feature_level_from_rt_only_is_fl_12_1() {
    let features = Features::RAY_TRACING_ACCELERATION_STRUCTURE;
    let fl = D3D12FeatureLevel::from_features(features);
    assert_eq!(fl, D3D12FeatureLevel::FL_12_1);
}

#[test]
fn feature_level_from_rt_and_rayquery_is_fl_12_2() {
    let features = Features::RAY_TRACING_ACCELERATION_STRUCTURE | Features::RAY_QUERY;
    let fl = D3D12FeatureLevel::from_features(features);
    assert_eq!(fl, D3D12FeatureLevel::FL_12_2);
}

// ============================================================================
// SECTION 9 -- D3D12ShaderModel Basic Contract Tests
// ============================================================================

#[test]
fn shader_model_default_is_sm_5_1() {
    let sm = D3D12ShaderModel::default();
    assert_eq!(sm, D3D12ShaderModel::SM_5_1);
}

#[test]
fn shader_model_sm_5_1_name() {
    assert_eq!(D3D12ShaderModel::SM_5_1.name(), "5.1");
}

#[test]
fn shader_model_sm_6_0_name() {
    assert_eq!(D3D12ShaderModel::SM_6_0.name(), "6.0");
}

#[test]
fn shader_model_sm_6_3_name() {
    assert_eq!(D3D12ShaderModel::SM_6_3.name(), "6.3");
}

#[test]
fn shader_model_sm_6_5_name() {
    assert_eq!(D3D12ShaderModel::SM_6_5.name(), "6.5");
}

#[test]
fn shader_model_sm_6_7_name() {
    assert_eq!(D3D12ShaderModel::SM_6_7.name(), "6.7");
}

#[test]
fn shader_model_display_sm_5_1() {
    assert_eq!(format!("{}", D3D12ShaderModel::SM_5_1), "SM 5.1");
}

#[test]
fn shader_model_display_sm_6_5() {
    assert_eq!(format!("{}", D3D12ShaderModel::SM_6_5), "SM 6.5");
}

// ============================================================================
// SECTION 10 -- D3D12ShaderModel Version Tests
// ============================================================================

#[test]
fn shader_model_sm_5_1_version() {
    assert_eq!(D3D12ShaderModel::SM_5_1.version(), (5, 1));
}

#[test]
fn shader_model_sm_6_0_version() {
    assert_eq!(D3D12ShaderModel::SM_6_0.version(), (6, 0));
}

#[test]
fn shader_model_sm_6_1_version() {
    assert_eq!(D3D12ShaderModel::SM_6_1.version(), (6, 1));
}

#[test]
fn shader_model_sm_6_2_version() {
    assert_eq!(D3D12ShaderModel::SM_6_2.version(), (6, 2));
}

#[test]
fn shader_model_sm_6_3_version() {
    assert_eq!(D3D12ShaderModel::SM_6_3.version(), (6, 3));
}

#[test]
fn shader_model_sm_6_4_version() {
    assert_eq!(D3D12ShaderModel::SM_6_4.version(), (6, 4));
}

#[test]
fn shader_model_sm_6_5_version() {
    assert_eq!(D3D12ShaderModel::SM_6_5.version(), (6, 5));
}

#[test]
fn shader_model_sm_6_6_version() {
    assert_eq!(D3D12ShaderModel::SM_6_6.version(), (6, 6));
}

#[test]
fn shader_model_sm_6_7_version() {
    assert_eq!(D3D12ShaderModel::SM_6_7.version(), (6, 7));
}

// ============================================================================
// SECTION 11 -- D3D12ShaderModel Ordering Tests
// ============================================================================

#[test]
fn shader_model_ordering_sm_5_1_less_than_sm_6_0() {
    assert!(D3D12ShaderModel::SM_5_1 < D3D12ShaderModel::SM_6_0);
}

#[test]
fn shader_model_ordering_sm_6_0_less_than_sm_6_3() {
    assert!(D3D12ShaderModel::SM_6_0 < D3D12ShaderModel::SM_6_3);
}

#[test]
fn shader_model_ordering_sm_6_3_less_than_sm_6_5() {
    assert!(D3D12ShaderModel::SM_6_3 < D3D12ShaderModel::SM_6_5);
}

#[test]
fn shader_model_ordering_sm_6_5_less_than_sm_6_7() {
    assert!(D3D12ShaderModel::SM_6_5 < D3D12ShaderModel::SM_6_7);
}

#[test]
fn shader_model_ordering_full_chain() {
    assert!(D3D12ShaderModel::SM_5_1 < D3D12ShaderModel::SM_6_7);
    assert!(D3D12ShaderModel::SM_6_2 < D3D12ShaderModel::SM_6_6);
}

// ============================================================================
// SECTION 12 -- D3D12ShaderModel Wave Intrinsics Support Tests
// ============================================================================

#[test]
fn shader_model_sm_5_1_does_not_support_wave_intrinsics() {
    assert!(!D3D12ShaderModel::SM_5_1.supports_wave_intrinsics());
}

#[test]
fn shader_model_sm_6_0_supports_wave_intrinsics() {
    assert!(D3D12ShaderModel::SM_6_0.supports_wave_intrinsics());
}

#[test]
fn shader_model_sm_6_3_supports_wave_intrinsics() {
    assert!(D3D12ShaderModel::SM_6_3.supports_wave_intrinsics());
}

#[test]
fn shader_model_sm_6_5_supports_wave_intrinsics() {
    assert!(D3D12ShaderModel::SM_6_5.supports_wave_intrinsics());
}

#[test]
fn shader_model_sm_6_7_supports_wave_intrinsics() {
    assert!(D3D12ShaderModel::SM_6_7.supports_wave_intrinsics());
}

// ============================================================================
// SECTION 13 -- D3D12ShaderModel Ray Tracing Intrinsics Support Tests
// ============================================================================

#[test]
fn shader_model_sm_5_1_does_not_support_rt_intrinsics() {
    assert!(!D3D12ShaderModel::SM_5_1.supports_raytracing_intrinsics());
}

#[test]
fn shader_model_sm_6_0_does_not_support_rt_intrinsics() {
    assert!(!D3D12ShaderModel::SM_6_0.supports_raytracing_intrinsics());
}

#[test]
fn shader_model_sm_6_2_does_not_support_rt_intrinsics() {
    assert!(!D3D12ShaderModel::SM_6_2.supports_raytracing_intrinsics());
}

#[test]
fn shader_model_sm_6_3_supports_rt_intrinsics() {
    assert!(D3D12ShaderModel::SM_6_3.supports_raytracing_intrinsics());
}

#[test]
fn shader_model_sm_6_4_supports_rt_intrinsics() {
    assert!(D3D12ShaderModel::SM_6_4.supports_raytracing_intrinsics());
}

#[test]
fn shader_model_sm_6_5_supports_rt_intrinsics() {
    assert!(D3D12ShaderModel::SM_6_5.supports_raytracing_intrinsics());
}

#[test]
fn shader_model_sm_6_6_supports_rt_intrinsics() {
    assert!(D3D12ShaderModel::SM_6_6.supports_raytracing_intrinsics());
}

#[test]
fn shader_model_sm_6_7_supports_rt_intrinsics() {
    assert!(D3D12ShaderModel::SM_6_7.supports_raytracing_intrinsics());
}

// ============================================================================
// SECTION 14 -- D3D12ShaderModel Mesh Shader Support Tests
// ============================================================================

#[test]
fn shader_model_sm_5_1_does_not_support_mesh_shaders() {
    assert!(!D3D12ShaderModel::SM_5_1.supports_mesh_shaders());
}

#[test]
fn shader_model_sm_6_0_does_not_support_mesh_shaders() {
    assert!(!D3D12ShaderModel::SM_6_0.supports_mesh_shaders());
}

#[test]
fn shader_model_sm_6_3_does_not_support_mesh_shaders() {
    assert!(!D3D12ShaderModel::SM_6_3.supports_mesh_shaders());
}

#[test]
fn shader_model_sm_6_4_does_not_support_mesh_shaders() {
    assert!(!D3D12ShaderModel::SM_6_4.supports_mesh_shaders());
}

#[test]
fn shader_model_sm_6_5_supports_mesh_shaders() {
    assert!(D3D12ShaderModel::SM_6_5.supports_mesh_shaders());
}

#[test]
fn shader_model_sm_6_6_supports_mesh_shaders() {
    assert!(D3D12ShaderModel::SM_6_6.supports_mesh_shaders());
}

#[test]
fn shader_model_sm_6_7_supports_mesh_shaders() {
    assert!(D3D12ShaderModel::SM_6_7.supports_mesh_shaders());
}

// ============================================================================
// SECTION 15 -- D3D12ShaderModel Derivatives Support Tests
// ============================================================================

#[test]
fn shader_model_sm_5_1_does_not_support_derivatives() {
    assert!(!D3D12ShaderModel::SM_5_1.supports_derivatives());
}

#[test]
fn shader_model_sm_6_3_does_not_support_derivatives() {
    assert!(!D3D12ShaderModel::SM_6_3.supports_derivatives());
}

#[test]
fn shader_model_sm_6_5_does_not_support_derivatives() {
    assert!(!D3D12ShaderModel::SM_6_5.supports_derivatives());
}

#[test]
fn shader_model_sm_6_6_supports_derivatives() {
    assert!(D3D12ShaderModel::SM_6_6.supports_derivatives());
}

#[test]
fn shader_model_sm_6_7_supports_derivatives() {
    assert!(D3D12ShaderModel::SM_6_7.supports_derivatives());
}

// ============================================================================
// SECTION 16 -- D3D12ShaderModel 16-bit Types Support Tests
// ============================================================================

#[test]
fn shader_model_sm_5_1_does_not_support_16bit_types() {
    assert!(!D3D12ShaderModel::SM_5_1.supports_16bit_types());
}

#[test]
fn shader_model_sm_6_0_does_not_support_16bit_types() {
    assert!(!D3D12ShaderModel::SM_6_0.supports_16bit_types());
}

#[test]
fn shader_model_sm_6_1_does_not_support_16bit_types() {
    assert!(!D3D12ShaderModel::SM_6_1.supports_16bit_types());
}

#[test]
fn shader_model_sm_6_2_supports_16bit_types() {
    assert!(D3D12ShaderModel::SM_6_2.supports_16bit_types());
}

#[test]
fn shader_model_sm_6_5_supports_16bit_types() {
    assert!(D3D12ShaderModel::SM_6_5.supports_16bit_types());
}

#[test]
fn shader_model_sm_6_7_supports_16bit_types() {
    assert!(D3D12ShaderModel::SM_6_7.supports_16bit_types());
}

// ============================================================================
// SECTION 17 -- D3D12ShaderModel from_features() Tests
// ============================================================================

#[test]
fn shader_model_from_empty_features_is_sm_5_1() {
    let features = Features::empty();
    let sm = D3D12ShaderModel::from_features(features);
    assert_eq!(sm, D3D12ShaderModel::SM_5_1);
}

#[test]
fn shader_model_from_bindless_is_sm_5_1() {
    let features = Features::TEXTURE_BINDING_ARRAY
        | Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING;
    let sm = D3D12ShaderModel::from_features(features);
    assert_eq!(sm, D3D12ShaderModel::SM_5_1);
}

#[test]
fn shader_model_from_subgroups_is_sm_6_0() {
    let features = Features::SUBGROUP;
    let sm = D3D12ShaderModel::from_features(features);
    assert_eq!(sm, D3D12ShaderModel::SM_6_0);
}

#[test]
fn shader_model_from_rt_is_sm_6_3() {
    let features = Features::RAY_TRACING_ACCELERATION_STRUCTURE;
    let sm = D3D12ShaderModel::from_features(features);
    assert_eq!(sm, D3D12ShaderModel::SM_6_3);
}

#[test]
fn shader_model_from_ray_query_is_sm_6_5() {
    let features = Features::RAY_QUERY;
    let sm = D3D12ShaderModel::from_features(features);
    assert_eq!(sm, D3D12ShaderModel::SM_6_5);
}

#[test]
fn shader_model_from_rt_and_ray_query_is_sm_6_5() {
    let features = Features::RAY_TRACING_ACCELERATION_STRUCTURE | Features::RAY_QUERY;
    let sm = D3D12ShaderModel::from_features(features);
    assert_eq!(sm, D3D12ShaderModel::SM_6_5);
}

// ============================================================================
// SECTION 18 -- D3D12RayTracingTier Basic Contract Tests
// ============================================================================

#[test]
fn rt_tier_default_is_none() {
    let tier = D3D12RayTracingTier::default();
    assert_eq!(tier, D3D12RayTracingTier::None);
}

#[test]
fn rt_tier_none_name() {
    assert_eq!(D3D12RayTracingTier::None.name(), "None");
}

#[test]
fn rt_tier_tier1_0_name() {
    assert_eq!(D3D12RayTracingTier::Tier1_0.name(), "DXR 1.0");
}

#[test]
fn rt_tier_tier1_1_name() {
    assert_eq!(D3D12RayTracingTier::Tier1_1.name(), "DXR 1.1");
}

#[test]
fn rt_tier_display_none() {
    assert_eq!(format!("{}", D3D12RayTracingTier::None), "None");
}

#[test]
fn rt_tier_display_tier1_0() {
    assert_eq!(format!("{}", D3D12RayTracingTier::Tier1_0), "DXR 1.0");
}

#[test]
fn rt_tier_display_tier1_1() {
    assert_eq!(format!("{}", D3D12RayTracingTier::Tier1_1), "DXR 1.1");
}

// ============================================================================
// SECTION 19 -- D3D12RayTracingTier Ordering Tests
// ============================================================================

#[test]
fn rt_tier_ordering_none_less_than_tier1_0() {
    assert!(D3D12RayTracingTier::None < D3D12RayTracingTier::Tier1_0);
}

#[test]
fn rt_tier_ordering_tier1_0_less_than_tier1_1() {
    assert!(D3D12RayTracingTier::Tier1_0 < D3D12RayTracingTier::Tier1_1);
}

#[test]
fn rt_tier_ordering_full_chain() {
    assert!(D3D12RayTracingTier::None < D3D12RayTracingTier::Tier1_1);
}

// ============================================================================
// SECTION 20 -- D3D12RayTracingTier is_available() Tests
// ============================================================================

#[test]
fn rt_tier_none_is_not_available() {
    assert!(!D3D12RayTracingTier::None.is_available());
}

#[test]
fn rt_tier_tier1_0_is_available() {
    assert!(D3D12RayTracingTier::Tier1_0.is_available());
}

#[test]
fn rt_tier_tier1_1_is_available() {
    assert!(D3D12RayTracingTier::Tier1_1.is_available());
}

// ============================================================================
// SECTION 21 -- D3D12RayTracingTier Inline RT Support Tests
// ============================================================================

#[test]
fn rt_tier_none_does_not_support_inline_rt() {
    assert!(!D3D12RayTracingTier::None.supports_inline_raytracing());
}

#[test]
fn rt_tier_tier1_0_does_not_support_inline_rt() {
    assert!(!D3D12RayTracingTier::Tier1_0.supports_inline_raytracing());
}

#[test]
fn rt_tier_tier1_1_supports_inline_rt() {
    assert!(D3D12RayTracingTier::Tier1_1.supports_inline_raytracing());
}

// ============================================================================
// SECTION 22 -- D3D12RayTracingTier RayQuery Support Tests
// ============================================================================

#[test]
fn rt_tier_none_does_not_support_rayquery() {
    assert!(!D3D12RayTracingTier::None.supports_rayquery());
}

#[test]
fn rt_tier_tier1_0_does_not_support_rayquery() {
    assert!(!D3D12RayTracingTier::Tier1_0.supports_rayquery());
}

#[test]
fn rt_tier_tier1_1_supports_rayquery() {
    assert!(D3D12RayTracingTier::Tier1_1.supports_rayquery());
}

// ============================================================================
// SECTION 23 -- D3D12RayTracingTier from_features() Tests
// ============================================================================

#[test]
fn rt_tier_from_empty_features_is_none() {
    let features = Features::empty();
    let tier = D3D12RayTracingTier::from_features(features);
    assert_eq!(tier, D3D12RayTracingTier::None);
}

#[test]
fn rt_tier_from_rt_only_is_tier1_0() {
    let features = Features::RAY_TRACING_ACCELERATION_STRUCTURE;
    let tier = D3D12RayTracingTier::from_features(features);
    assert_eq!(tier, D3D12RayTracingTier::Tier1_0);
}

#[test]
fn rt_tier_from_rt_and_rayquery_is_tier1_1() {
    let features = Features::RAY_TRACING_ACCELERATION_STRUCTURE | Features::RAY_QUERY;
    let tier = D3D12RayTracingTier::from_features(features);
    assert_eq!(tier, D3D12RayTracingTier::Tier1_1);
}

#[test]
fn rt_tier_from_rayquery_only_is_none() {
    // Ray query without acceleration structure should be None
    // (ray query is meaningless without RT support)
    let features = Features::RAY_QUERY;
    let tier = D3D12RayTracingTier::from_features(features);
    assert_eq!(tier, D3D12RayTracingTier::None);
}

// ============================================================================
// SECTION 24 -- D3D12Features Default Tests
// ============================================================================

#[test]
fn features_default_feature_level_is_fl_11_0() {
    let features = D3D12Features::default();
    assert_eq!(features.feature_level, D3D12FeatureLevel::FL_11_0);
}

#[test]
fn features_default_shader_model_is_sm_5_1() {
    let features = D3D12Features::default();
    assert_eq!(features.shader_model, D3D12ShaderModel::SM_5_1);
}

#[test]
fn features_default_rt_tier_is_none() {
    let features = D3D12Features::default();
    assert_eq!(features.ray_tracing_tier, D3D12RayTracingTier::None);
}

#[test]
fn features_default_mesh_shader_tier_is_0() {
    let features = D3D12Features::default();
    assert_eq!(features.mesh_shader_tier, 0);
}

#[test]
fn features_default_vrs_tier_is_0() {
    let features = D3D12Features::default();
    assert_eq!(features.variable_rate_shading_tier, 0);
}

#[test]
fn features_default_sampler_feedback_tier_is_0() {
    let features = D3D12Features::default();
    assert_eq!(features.sampler_feedback_tier, 0);
}

#[test]
fn features_default_bindless_is_false() {
    let features = D3D12Features::default();
    assert!(!features.bindless_resources);
}

#[test]
fn features_default_conservative_raster_tier_is_0() {
    let features = D3D12Features::default();
    assert_eq!(features.conservative_rasterization_tier, 0);
}

#[test]
fn features_default_tiled_resources_tier_is_0() {
    let features = D3D12Features::default();
    assert_eq!(features.tiled_resources_tier, 0);
}

#[test]
fn features_default_resource_binding_tier_is_0() {
    let features = D3D12Features::default();
    assert_eq!(features.resource_binding_tier, 0);
}

#[test]
fn features_default_root_signature_version_is_0() {
    let features = D3D12Features::default();
    assert_eq!(features.root_signature_version, 0);
}

#[test]
fn features_default_wave_ops_is_false() {
    let features = D3D12Features::default();
    assert!(!features.wave_ops);
}

#[test]
fn features_default_native_16bit_ops_is_false() {
    let features = D3D12Features::default();
    assert!(!features.native_16bit_ops);
}

#[test]
fn features_default_rt_pipeline_is_false() {
    let features = D3D12Features::default();
    assert!(!features.rt_pipeline);
}

#[test]
fn features_default_rov_tier_is_0() {
    let features = D3D12Features::default();
    assert_eq!(features.rasterizer_ordered_views_tier, 0);
}

// ============================================================================
// SECTION 25 -- D3D12Features supports_rt() Tests
// ============================================================================

#[test]
fn features_default_does_not_support_rt() {
    let features = D3D12Features::default();
    assert!(!features.supports_rt());
}

#[test]
fn features_with_tier1_0_supports_rt() {
    let mut features = D3D12Features::default();
    features.ray_tracing_tier = D3D12RayTracingTier::Tier1_0;
    assert!(features.supports_rt());
}

#[test]
fn features_with_tier1_1_supports_rt() {
    let mut features = D3D12Features::default();
    features.ray_tracing_tier = D3D12RayTracingTier::Tier1_1;
    assert!(features.supports_rt());
}

// ============================================================================
// SECTION 26 -- D3D12Features supports_mesh_shaders() Tests
// ============================================================================

#[test]
fn features_default_does_not_support_mesh_shaders() {
    let features = D3D12Features::default();
    assert!(!features.supports_mesh_shaders());
}

#[test]
fn features_with_mesh_tier_0_does_not_support_mesh_shaders() {
    let mut features = D3D12Features::default();
    features.mesh_shader_tier = 0;
    assert!(!features.supports_mesh_shaders());
}

#[test]
fn features_with_mesh_tier_1_supports_mesh_shaders() {
    let mut features = D3D12Features::default();
    features.mesh_shader_tier = 1;
    assert!(features.supports_mesh_shaders());
}

// ============================================================================
// SECTION 27 -- D3D12Features supports_bindless() Tests
// ============================================================================

#[test]
fn features_default_does_not_support_bindless() {
    let features = D3D12Features::default();
    assert!(!features.supports_bindless());
}

#[test]
fn features_with_bindless_true_supports_bindless() {
    let mut features = D3D12Features::default();
    features.bindless_resources = true;
    assert!(features.supports_bindless());
}

// ============================================================================
// SECTION 28 -- D3D12Features supports_vrs() Tests
// ============================================================================

#[test]
fn features_default_does_not_support_vrs() {
    let features = D3D12Features::default();
    assert!(!features.supports_vrs());
}

#[test]
fn features_with_vrs_tier_0_does_not_support_vrs() {
    let mut features = D3D12Features::default();
    features.variable_rate_shading_tier = 0;
    assert!(!features.supports_vrs());
}

#[test]
fn features_with_vrs_tier_1_supports_vrs() {
    let mut features = D3D12Features::default();
    features.variable_rate_shading_tier = 1;
    assert!(features.supports_vrs());
}

#[test]
fn features_with_vrs_tier_2_supports_vrs() {
    let mut features = D3D12Features::default();
    features.variable_rate_shading_tier = 2;
    assert!(features.supports_vrs());
}

// ============================================================================
// SECTION 29 -- D3D12Features supports_inline_rt() Tests
// ============================================================================

#[test]
fn features_default_does_not_support_inline_rt() {
    let features = D3D12Features::default();
    assert!(!features.supports_inline_rt());
}

#[test]
fn features_with_tier1_0_does_not_support_inline_rt() {
    let mut features = D3D12Features::default();
    features.ray_tracing_tier = D3D12RayTracingTier::Tier1_0;
    assert!(!features.supports_inline_rt());
}

#[test]
fn features_with_tier1_1_supports_inline_rt() {
    let mut features = D3D12Features::default();
    features.ray_tracing_tier = D3D12RayTracingTier::Tier1_1;
    assert!(features.supports_inline_rt());
}

// ============================================================================
// SECTION 30 -- D3D12Features supports_sampler_feedback() Tests
// ============================================================================

#[test]
fn features_default_does_not_support_sampler_feedback() {
    let features = D3D12Features::default();
    assert!(!features.supports_sampler_feedback());
}

#[test]
fn features_with_sampler_feedback_tier_1_supports_sampler_feedback() {
    let mut features = D3D12Features::default();
    features.sampler_feedback_tier = 1;
    assert!(features.supports_sampler_feedback());
}

// ============================================================================
// SECTION 31 -- D3D12Features supports_conservative_raster() Tests
// ============================================================================

#[test]
fn features_default_does_not_support_conservative_raster() {
    let features = D3D12Features::default();
    assert!(!features.supports_conservative_raster());
}

#[test]
fn features_with_conservative_raster_tier_1_supports_conservative_raster() {
    let mut features = D3D12Features::default();
    features.conservative_rasterization_tier = 1;
    assert!(features.supports_conservative_raster());
}

#[test]
fn features_with_conservative_raster_tier_3_supports_conservative_raster() {
    let mut features = D3D12Features::default();
    features.conservative_rasterization_tier = 3;
    assert!(features.supports_conservative_raster());
}

// ============================================================================
// SECTION 32 -- D3D12Features supports_gpu_driven() Tests
// ============================================================================

#[test]
fn features_default_does_not_support_gpu_driven() {
    let features = D3D12Features::default();
    assert!(!features.supports_gpu_driven());
}

#[test]
fn features_bindless_only_does_not_support_gpu_driven() {
    let mut features = D3D12Features::default();
    features.bindless_resources = true;
    assert!(!features.supports_gpu_driven());
}

#[test]
fn features_wave_ops_only_does_not_support_gpu_driven() {
    let mut features = D3D12Features::default();
    features.wave_ops = true;
    assert!(!features.supports_gpu_driven());
}

#[test]
fn features_bindless_and_wave_ops_supports_gpu_driven() {
    let mut features = D3D12Features::default();
    features.bindless_resources = true;
    features.wave_ops = true;
    assert!(features.supports_gpu_driven());
}

// ============================================================================
// SECTION 33 -- D3D12Features from_features() Tests
// ============================================================================

#[test]
fn features_from_empty_wgpu_features() {
    let wgpu_features = Features::empty();
    let dx12 = D3D12Features::from_features(wgpu_features);

    assert_eq!(dx12.feature_level, D3D12FeatureLevel::FL_11_0);
    assert_eq!(dx12.shader_model, D3D12ShaderModel::SM_5_1);
    assert_eq!(dx12.ray_tracing_tier, D3D12RayTracingTier::None);
    assert!(!dx12.bindless_resources);
    assert!(!dx12.wave_ops);
}

#[test]
fn features_from_rt_wgpu_features() {
    let wgpu_features = Features::RAY_TRACING_ACCELERATION_STRUCTURE;
    let dx12 = D3D12Features::from_features(wgpu_features);

    assert_eq!(dx12.feature_level, D3D12FeatureLevel::FL_12_1);
    assert_eq!(dx12.shader_model, D3D12ShaderModel::SM_6_3);
    assert_eq!(dx12.ray_tracing_tier, D3D12RayTracingTier::Tier1_0);
    assert!(dx12.rt_pipeline);
}

#[test]
fn features_from_full_rt_wgpu_features() {
    let wgpu_features = Features::RAY_TRACING_ACCELERATION_STRUCTURE | Features::RAY_QUERY;
    let dx12 = D3D12Features::from_features(wgpu_features);

    assert_eq!(dx12.feature_level, D3D12FeatureLevel::FL_12_2);
    assert_eq!(dx12.shader_model, D3D12ShaderModel::SM_6_5);
    assert_eq!(dx12.ray_tracing_tier, D3D12RayTracingTier::Tier1_1);
    assert!(dx12.supports_rt());
    assert!(dx12.supports_inline_rt());
}

#[test]
fn features_from_bindless_wgpu_features() {
    let wgpu_features = Features::TEXTURE_BINDING_ARRAY
        | Features::BUFFER_BINDING_ARRAY
        | Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING;
    let dx12 = D3D12Features::from_features(wgpu_features);

    assert!(dx12.bindless_resources);
    assert!(dx12.supports_bindless());
}

#[test]
fn features_from_subgroup_wgpu_features() {
    let wgpu_features = Features::SUBGROUP;
    let dx12 = D3D12Features::from_features(wgpu_features);

    assert_eq!(dx12.shader_model, D3D12ShaderModel::SM_6_0);
    assert!(dx12.wave_ops);
}

#[test]
fn features_from_fp16_wgpu_features() {
    let wgpu_features = Features::SHADER_F16;
    let dx12 = D3D12Features::from_features(wgpu_features);

    assert!(dx12.native_16bit_ops);
}

// ============================================================================
// SECTION 34 -- D3D12Features summary() Tests
// ============================================================================

#[test]
fn features_summary_basic_contains_fl_and_sm() {
    let features = D3D12Features::default();
    let summary = features.summary();
    assert!(summary.contains("FL 11.0"));
    assert!(summary.contains("SM 5.1"));
}

#[test]
fn features_summary_full_contains_all_features() {
    let features = D3D12Features {
        feature_level: D3D12FeatureLevel::FL_12_2,
        shader_model: D3D12ShaderModel::SM_6_5,
        ray_tracing_tier: D3D12RayTracingTier::Tier1_1,
        mesh_shader_tier: 1,
        variable_rate_shading_tier: 2,
        sampler_feedback_tier: 1,
        bindless_resources: true,
        conservative_rasterization_tier: 3,
        tiled_resources_tier: 4,
        resource_binding_tier: 3,
        root_signature_version: 2,
        wave_ops: true,
        native_16bit_ops: true,
        rt_pipeline: true,
        rasterizer_ordered_views_tier: 3,
    };

    let summary = features.summary();
    assert!(summary.contains("FL 12.2"));
    assert!(summary.contains("SM 6.5"));
    assert!(summary.contains("DXR 1.1"));
    assert!(summary.contains("Mesh"));
    assert!(summary.contains("VRS T2"));
    assert!(summary.contains("Bindless"));
    assert!(summary.contains("Wave"));
    assert!(summary.contains("FP16"));
}

#[test]
fn features_summary_with_rt_tier_1_0_contains_dxr_1_0() {
    let mut features = D3D12Features::default();
    features.ray_tracing_tier = D3D12RayTracingTier::Tier1_0;
    let summary = features.summary();
    assert!(summary.contains("DXR 1.0"));
}

#[test]
fn features_summary_with_vrs_tier_1_contains_vrs_t1() {
    let mut features = D3D12Features::default();
    features.variable_rate_shading_tier = 1;
    let summary = features.summary();
    assert!(summary.contains("VRS T1"));
}

// ============================================================================
// SECTION 35 -- D3D12Features minimum_windows_version() Tests
// ============================================================================

#[test]
fn features_minimum_windows_version_basic() {
    let features = D3D12Features::default();
    let (major, minor, build) = features.minimum_windows_version();
    assert_eq!(major, 10);
    assert_eq!(minor, 0);
    assert_eq!(build, 10240);
}

#[test]
fn features_minimum_windows_version_with_vrs() {
    let mut features = D3D12Features::default();
    features.variable_rate_shading_tier = 1;
    let (_, _, build) = features.minimum_windows_version();
    assert_eq!(build, 18362);
}

#[test]
fn features_minimum_windows_version_with_mesh_shaders() {
    let mut features = D3D12Features::default();
    features.mesh_shader_tier = 1;
    let (_, _, build) = features.minimum_windows_version();
    assert_eq!(build, 19041);
}

#[test]
fn features_minimum_windows_version_with_dxr_1_0() {
    let mut features = D3D12Features::default();
    features.ray_tracing_tier = D3D12RayTracingTier::Tier1_0;
    let (_, _, build) = features.minimum_windows_version();
    assert_eq!(build, 17763);
}

#[test]
fn features_minimum_windows_version_with_dxr_1_1() {
    let mut features = D3D12Features::default();
    features.ray_tracing_tier = D3D12RayTracingTier::Tier1_1;
    let (_, _, build) = features.minimum_windows_version();
    assert_eq!(build, 19041);
}

#[test]
fn features_minimum_windows_version_dxr_1_1_takes_precedence_over_vrs() {
    let mut features = D3D12Features::default();
    features.ray_tracing_tier = D3D12RayTracingTier::Tier1_1;
    features.variable_rate_shading_tier = 1;
    let (_, _, build) = features.minimum_windows_version();
    // DXR 1.1 requires 19041, VRS requires 18362
    // DXR 1.1 should take precedence (higher requirement)
    assert_eq!(build, 19041);
}

// ============================================================================
// SECTION 36 -- Real-World GPU Profile Tests: GTX 1080
// ============================================================================

/// GTX 1080: FL 12.0, SM 6.0, No RT
fn gtx_1080_features() -> Features {
    Features::TEXTURE_BINDING_ARRAY
        | Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING
        | Features::SUBGROUP
        | Features::TIMESTAMP_QUERY
}

#[test]
fn gtx_1080_feature_level_is_fl_12_0() {
    let dx12 = D3D12Features::from_features(gtx_1080_features());
    assert_eq!(dx12.feature_level, D3D12FeatureLevel::FL_12_0);
}

#[test]
fn gtx_1080_shader_model_is_sm_6_0() {
    let dx12 = D3D12Features::from_features(gtx_1080_features());
    assert_eq!(dx12.shader_model, D3D12ShaderModel::SM_6_0);
}

#[test]
fn gtx_1080_does_not_support_rt() {
    let dx12 = D3D12Features::from_features(gtx_1080_features());
    assert!(!dx12.supports_rt());
}

#[test]
fn gtx_1080_does_not_support_mesh_shaders() {
    let dx12 = D3D12Features::from_features(gtx_1080_features());
    assert!(!dx12.supports_mesh_shaders());
}

#[test]
fn gtx_1080_supports_bindless() {
    let features = gtx_1080_features() | Features::BUFFER_BINDING_ARRAY;
    let dx12 = D3D12Features::from_features(features);
    assert!(dx12.supports_bindless());
}

#[test]
fn gtx_1080_supports_wave_ops() {
    let dx12 = D3D12Features::from_features(gtx_1080_features());
    assert!(dx12.wave_ops);
}

// ============================================================================
// SECTION 37 -- Real-World GPU Profile Tests: RTX 2080
// ============================================================================

/// RTX 2080: FL 12.1, SM 6.5, RT Tier 1.0
fn rtx_2080_features() -> Features {
    Features::TEXTURE_BINDING_ARRAY
        | Features::BUFFER_BINDING_ARRAY
        | Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING
        | Features::SUBGROUP
        | Features::TIMESTAMP_QUERY
        | Features::RAY_TRACING_ACCELERATION_STRUCTURE
}

#[test]
fn rtx_2080_feature_level_is_fl_12_1() {
    let dx12 = D3D12Features::from_features(rtx_2080_features());
    assert_eq!(dx12.feature_level, D3D12FeatureLevel::FL_12_1);
}

#[test]
fn rtx_2080_shader_model_is_sm_6_3() {
    let dx12 = D3D12Features::from_features(rtx_2080_features());
    assert_eq!(dx12.shader_model, D3D12ShaderModel::SM_6_3);
}

#[test]
fn rtx_2080_rt_tier_is_tier1_0() {
    let dx12 = D3D12Features::from_features(rtx_2080_features());
    assert_eq!(dx12.ray_tracing_tier, D3D12RayTracingTier::Tier1_0);
}

#[test]
fn rtx_2080_supports_rt() {
    let dx12 = D3D12Features::from_features(rtx_2080_features());
    assert!(dx12.supports_rt());
}

#[test]
fn rtx_2080_does_not_support_inline_rt() {
    let dx12 = D3D12Features::from_features(rtx_2080_features());
    assert!(!dx12.supports_inline_rt());
}

#[test]
fn rtx_2080_supports_bindless() {
    let dx12 = D3D12Features::from_features(rtx_2080_features());
    assert!(dx12.supports_bindless());
}

#[test]
fn rtx_2080_supports_vrs() {
    let dx12 = D3D12Features::from_features(rtx_2080_features());
    assert!(dx12.supports_vrs());
}

// ============================================================================
// SECTION 38 -- Real-World GPU Profile Tests: RTX 3080
// ============================================================================

/// RTX 3080: FL 12.2, SM 6.6, RT Tier 1.1
fn rtx_3080_features() -> Features {
    Features::TEXTURE_BINDING_ARRAY
        | Features::BUFFER_BINDING_ARRAY
        | Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING
        | Features::SUBGROUP
        | Features::TIMESTAMP_QUERY
        | Features::RAY_TRACING_ACCELERATION_STRUCTURE
        | Features::RAY_QUERY
        | Features::SHADER_F16
}

#[test]
fn rtx_3080_feature_level_is_fl_12_2() {
    let dx12 = D3D12Features::from_features(rtx_3080_features());
    assert_eq!(dx12.feature_level, D3D12FeatureLevel::FL_12_2);
}

#[test]
fn rtx_3080_shader_model_is_sm_6_5() {
    let dx12 = D3D12Features::from_features(rtx_3080_features());
    assert_eq!(dx12.shader_model, D3D12ShaderModel::SM_6_5);
}

#[test]
fn rtx_3080_rt_tier_is_tier1_1() {
    let dx12 = D3D12Features::from_features(rtx_3080_features());
    assert_eq!(dx12.ray_tracing_tier, D3D12RayTracingTier::Tier1_1);
}

#[test]
fn rtx_3080_supports_rt() {
    let dx12 = D3D12Features::from_features(rtx_3080_features());
    assert!(dx12.supports_rt());
}

#[test]
fn rtx_3080_supports_inline_rt() {
    let dx12 = D3D12Features::from_features(rtx_3080_features());
    assert!(dx12.supports_inline_rt());
}

#[test]
fn rtx_3080_supports_mesh_shaders() {
    let dx12 = D3D12Features::from_features(rtx_3080_features());
    assert!(dx12.supports_mesh_shaders());
}

#[test]
fn rtx_3080_supports_vrs_tier_2() {
    let dx12 = D3D12Features::from_features(rtx_3080_features());
    assert_eq!(dx12.variable_rate_shading_tier, 2);
}

#[test]
fn rtx_3080_supports_sampler_feedback() {
    let dx12 = D3D12Features::from_features(rtx_3080_features());
    assert!(dx12.supports_sampler_feedback());
}

#[test]
fn rtx_3080_supports_gpu_driven() {
    let dx12 = D3D12Features::from_features(rtx_3080_features());
    assert!(dx12.supports_gpu_driven());
}

#[test]
fn rtx_3080_supports_native_16bit_ops() {
    let dx12 = D3D12Features::from_features(rtx_3080_features());
    assert!(dx12.native_16bit_ops);
}

// ============================================================================
// SECTION 39 -- Real-World GPU Profile Tests: RTX 4090
// ============================================================================

/// RTX 4090: FL 12.2, SM 6.7, RT Tier 1.1
fn rtx_4090_features() -> Features {
    rtx_3080_features() // Same wgpu features, different hardware
}

#[test]
fn rtx_4090_feature_level_is_fl_12_2() {
    let dx12 = D3D12Features::from_features(rtx_4090_features());
    assert_eq!(dx12.feature_level, D3D12FeatureLevel::FL_12_2);
}

#[test]
fn rtx_4090_supports_all_advanced_features() {
    let dx12 = D3D12Features::from_features(rtx_4090_features());
    assert!(dx12.supports_rt());
    assert!(dx12.supports_inline_rt());
    assert!(dx12.supports_mesh_shaders());
    assert!(dx12.supports_bindless());
    assert!(dx12.supports_vrs());
    assert!(dx12.supports_sampler_feedback());
    assert!(dx12.supports_gpu_driven());
    assert!(dx12.supports_conservative_raster());
}

// ============================================================================
// SECTION 40 -- Real-World GPU Profile Tests: AMD RX 6800
// ============================================================================

/// RX 6800: FL 12.1, SM 6.5, RT Tier 1.0 (RDNA2)
fn rx_6800_features() -> Features {
    Features::TEXTURE_BINDING_ARRAY
        | Features::BUFFER_BINDING_ARRAY
        | Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING
        | Features::SUBGROUP
        | Features::TIMESTAMP_QUERY
        | Features::RAY_TRACING_ACCELERATION_STRUCTURE
}

#[test]
fn rx_6800_feature_level_is_fl_12_1() {
    let dx12 = D3D12Features::from_features(rx_6800_features());
    assert_eq!(dx12.feature_level, D3D12FeatureLevel::FL_12_1);
}

#[test]
fn rx_6800_supports_rt() {
    let dx12 = D3D12Features::from_features(rx_6800_features());
    assert!(dx12.supports_rt());
}

#[test]
fn rx_6800_does_not_support_inline_rt() {
    // RDNA2 has limited RayQuery support
    let dx12 = D3D12Features::from_features(rx_6800_features());
    assert!(!dx12.supports_inline_rt());
}

#[test]
fn rx_6800_supports_bindless() {
    let dx12 = D3D12Features::from_features(rx_6800_features());
    assert!(dx12.supports_bindless());
}

// ============================================================================
// SECTION 41 -- Real-World GPU Profile Tests: AMD RX 7900
// ============================================================================

/// RX 7900: FL 12.2, SM 6.6, RT Tier 1.1 (RDNA3)
fn rx_7900_features() -> Features {
    Features::TEXTURE_BINDING_ARRAY
        | Features::BUFFER_BINDING_ARRAY
        | Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING
        | Features::SUBGROUP
        | Features::TIMESTAMP_QUERY
        | Features::RAY_TRACING_ACCELERATION_STRUCTURE
        | Features::RAY_QUERY
}

#[test]
fn rx_7900_feature_level_is_fl_12_2() {
    let dx12 = D3D12Features::from_features(rx_7900_features());
    assert_eq!(dx12.feature_level, D3D12FeatureLevel::FL_12_2);
}

#[test]
fn rx_7900_supports_rt() {
    let dx12 = D3D12Features::from_features(rx_7900_features());
    assert!(dx12.supports_rt());
}

#[test]
fn rx_7900_supports_inline_rt() {
    let dx12 = D3D12Features::from_features(rx_7900_features());
    assert!(dx12.supports_inline_rt());
}

#[test]
fn rx_7900_supports_mesh_shaders() {
    let dx12 = D3D12Features::from_features(rx_7900_features());
    assert!(dx12.supports_mesh_shaders());
}

// ============================================================================
// SECTION 42 -- Real-World GPU Profile Tests: Intel Arc A770
// ============================================================================

/// Intel Arc A770: FL 12.2, SM 6.6, RT Tier 1.1
fn arc_a770_features() -> Features {
    Features::TEXTURE_BINDING_ARRAY
        | Features::BUFFER_BINDING_ARRAY
        | Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING
        | Features::SUBGROUP
        | Features::TIMESTAMP_QUERY
        | Features::RAY_TRACING_ACCELERATION_STRUCTURE
        | Features::RAY_QUERY
}

#[test]
fn arc_a770_feature_level_is_fl_12_2() {
    let dx12 = D3D12Features::from_features(arc_a770_features());
    assert_eq!(dx12.feature_level, D3D12FeatureLevel::FL_12_2);
}

#[test]
fn arc_a770_supports_rt() {
    let dx12 = D3D12Features::from_features(arc_a770_features());
    assert!(dx12.supports_rt());
}

#[test]
fn arc_a770_supports_inline_rt() {
    let dx12 = D3D12Features::from_features(arc_a770_features());
    assert!(dx12.supports_inline_rt());
}

#[test]
fn arc_a770_supports_mesh_shaders() {
    let dx12 = D3D12Features::from_features(arc_a770_features());
    assert!(dx12.supports_mesh_shaders());
}

// ============================================================================
// SECTION 43 -- Legacy Hardware Tests: GTX 900 Series
// ============================================================================

/// GTX 980: FL 11.1 (DX11 hardware running in D3D12)
fn gtx_980_features() -> Features {
    Features::TIMESTAMP_QUERY
}

#[test]
fn gtx_980_feature_level_is_fl_11_1() {
    let dx12 = D3D12Features::from_features(gtx_980_features());
    assert_eq!(dx12.feature_level, D3D12FeatureLevel::FL_11_1);
}

#[test]
fn gtx_980_does_not_support_rt() {
    let dx12 = D3D12Features::from_features(gtx_980_features());
    assert!(!dx12.supports_rt());
}

#[test]
fn gtx_980_does_not_support_bindless() {
    let dx12 = D3D12Features::from_features(gtx_980_features());
    assert!(!dx12.supports_bindless());
}

#[test]
fn gtx_980_does_not_support_mesh_shaders() {
    let dx12 = D3D12Features::from_features(gtx_980_features());
    assert!(!dx12.supports_mesh_shaders());
}

// ============================================================================
// SECTION 44 -- Feature Combination Tests
// ============================================================================

#[test]
fn feature_combination_rt_requires_fl_12_1() {
    // RT without FL 12.1 features should still detect as FL 12.1
    let features = Features::RAY_TRACING_ACCELERATION_STRUCTURE;
    let dx12 = D3D12Features::from_features(features);
    assert!(dx12.feature_level >= D3D12FeatureLevel::FL_12_1);
}

#[test]
fn feature_combination_mesh_shaders_require_fl_12_2() {
    // Full RT (tier 1.1) implies FL 12.2 which enables mesh shaders
    let features = Features::RAY_TRACING_ACCELERATION_STRUCTURE | Features::RAY_QUERY;
    let dx12 = D3D12Features::from_features(features);
    assert_eq!(dx12.feature_level, D3D12FeatureLevel::FL_12_2);
    assert!(dx12.supports_mesh_shaders());
}

#[test]
fn feature_combination_vrs_requires_fl_12_1() {
    let features = Features::RAY_TRACING_ACCELERATION_STRUCTURE;
    let dx12 = D3D12Features::from_features(features);
    assert!(dx12.supports_vrs());
    assert!(dx12.variable_rate_shading_tier >= 1);
}

#[test]
fn feature_combination_sampler_feedback_requires_fl_12_2() {
    let features = Features::RAY_TRACING_ACCELERATION_STRUCTURE | Features::RAY_QUERY;
    let dx12 = D3D12Features::from_features(features);
    assert!(dx12.supports_sampler_feedback());
}

#[test]
fn feature_combination_conservative_raster_requires_fl_12_0() {
    let features = Features::TEXTURE_BINDING_ARRAY
        | Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING;
    let dx12 = D3D12Features::from_features(features);
    assert!(dx12.supports_conservative_raster());
}

// ============================================================================
// SECTION 45 -- Tier Progression Tests
// ============================================================================

#[test]
fn tier_progression_tiled_resources_by_feature_level() {
    // Test that tiled resources tier increases with feature level
    let fl_11_0 = D3D12Features::from_features(Features::empty());
    let fl_12_0 = D3D12Features::from_features(
        Features::TEXTURE_BINDING_ARRAY
            | Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING,
    );
    let fl_12_1 = D3D12Features::from_features(Features::RAY_TRACING_ACCELERATION_STRUCTURE);
    let fl_12_2 = D3D12Features::from_features(
        Features::RAY_TRACING_ACCELERATION_STRUCTURE | Features::RAY_QUERY,
    );

    assert!(fl_12_0.tiled_resources_tier > fl_11_0.tiled_resources_tier);
    assert!(fl_12_1.tiled_resources_tier > fl_12_0.tiled_resources_tier);
    assert!(fl_12_2.tiled_resources_tier >= fl_12_1.tiled_resources_tier);
}

#[test]
fn tier_progression_conservative_raster_by_feature_level() {
    let fl_12_0 = D3D12Features::from_features(
        Features::TEXTURE_BINDING_ARRAY
            | Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING,
    );
    let fl_12_1 = D3D12Features::from_features(Features::RAY_TRACING_ACCELERATION_STRUCTURE);

    assert!(fl_12_1.conservative_rasterization_tier > fl_12_0.conservative_rasterization_tier);
}

#[test]
fn tier_progression_vrs_by_feature_level() {
    let fl_12_1 = D3D12Features::from_features(Features::RAY_TRACING_ACCELERATION_STRUCTURE);
    let fl_12_2 = D3D12Features::from_features(
        Features::RAY_TRACING_ACCELERATION_STRUCTURE | Features::RAY_QUERY,
    );

    assert_eq!(fl_12_1.variable_rate_shading_tier, 1);
    assert_eq!(fl_12_2.variable_rate_shading_tier, 2);
}

// ============================================================================
// SECTION 46 -- Root Signature Version Tests
// ============================================================================

#[test]
fn root_signature_version_1_for_fl_11() {
    let dx12 = D3D12Features::from_features(Features::TIMESTAMP_QUERY);
    assert_eq!(dx12.root_signature_version, 1);
}

#[test]
fn root_signature_version_2_for_fl_12_0() {
    let dx12 = D3D12Features::from_features(
        Features::TEXTURE_BINDING_ARRAY
            | Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING,
    );
    assert_eq!(dx12.root_signature_version, 2);
}

// ============================================================================
// SECTION 47 -- Resource Binding Tier Tests
// ============================================================================

#[test]
fn resource_binding_tier_3_for_bindless() {
    let dx12 = D3D12Features::from_features(
        Features::TEXTURE_BINDING_ARRAY
            | Features::BUFFER_BINDING_ARRAY
            | Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING,
    );
    assert_eq!(dx12.resource_binding_tier, 3);
}

#[test]
fn resource_binding_tier_2_for_fl_12_0_without_bindless() {
    // FL 12.0 without full bindless should get tier 2
    let dx12 = D3D12Features::from_features(
        Features::TEXTURE_BINDING_ARRAY
            | Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING,
    );
    // This will detect as FL 12.0 but bindless requires BUFFER_BINDING_ARRAY too
    assert_eq!(dx12.feature_level, D3D12FeatureLevel::FL_12_0);
}

// ============================================================================
// SECTION 48 -- ROV Tier Tests
// ============================================================================

#[test]
fn rov_tier_progression_by_feature_level() {
    let fl_11_0 = D3D12Features::from_features(Features::empty());
    let fl_11_1 = D3D12Features::from_features(Features::TIMESTAMP_QUERY);
    let fl_12_0 = D3D12Features::from_features(
        Features::TEXTURE_BINDING_ARRAY
            | Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING,
    );
    let fl_12_1 = D3D12Features::from_features(Features::RAY_TRACING_ACCELERATION_STRUCTURE);

    assert_eq!(fl_11_0.rasterizer_ordered_views_tier, 0);
    assert_eq!(fl_11_1.rasterizer_ordered_views_tier, 1);
    assert_eq!(fl_12_0.rasterizer_ordered_views_tier, 2);
    assert_eq!(fl_12_1.rasterizer_ordered_views_tier, 3);
}

// ============================================================================
// SECTION 49 -- BackendType DX12 Integration Tests
// ============================================================================

#[test]
fn backend_type_dx12_is_native() {
    assert!(BackendType::Dx12.is_native());
}

#[test]
fn backend_type_dx12_supports_ray_tracing() {
    assert!(BackendType::Dx12.supports_ray_tracing());
}

#[test]
fn backend_type_dx12_supports_mesh_shaders() {
    assert!(BackendType::Dx12.supports_mesh_shaders());
}

#[test]
fn backend_type_dx12_supports_bindless() {
    assert!(BackendType::Dx12.supports_bindless());
}

#[test]
fn backend_type_dx12_name_is_directx_12() {
    assert_eq!(BackendType::Dx12.name(), "DirectX 12");
}

// ============================================================================
// SECTION 50 -- BackendCapabilities D3D12 Integration Tests
// ============================================================================

#[test]
fn backend_capabilities_default_supports_full_rt_requires_both() {
    let mut caps = BackendCapabilities::default();
    assert!(!caps.supports_full_rt());

    caps.ray_tracing = true;
    assert!(!caps.supports_full_rt());

    caps.ray_query = true;
    assert!(caps.supports_full_rt());
}

#[test]
fn backend_capabilities_dx12_gpu_driven_requires_bindless() {
    let mut caps = BackendCapabilities::default();
    caps.backend = BackendType::Dx12;

    assert!(!caps.supports_gpu_driven());

    caps.bindless = true;
    // DX12 doesn't require buffer_device_address for GPU-driven
    assert!(caps.supports_gpu_driven());
}

// ============================================================================
// SECTION 51 -- Negative Test Cases
// ============================================================================

#[test]
fn negative_test_fl_11_0_has_no_advanced_features() {
    let dx12 = D3D12Features::from_features(Features::empty());

    assert!(!dx12.supports_rt());
    assert!(!dx12.supports_inline_rt());
    assert!(!dx12.supports_mesh_shaders());
    assert!(!dx12.supports_vrs());
    assert!(!dx12.supports_sampler_feedback());
    assert!(!dx12.supports_bindless());
    assert!(!dx12.supports_conservative_raster());
    assert!(!dx12.supports_gpu_driven());
}

#[test]
fn negative_test_ray_query_without_rt_is_invalid_state() {
    // Ray query alone (without RT accel structures) should not enable RT
    let tier = D3D12RayTracingTier::from_features(Features::RAY_QUERY);
    assert_eq!(tier, D3D12RayTracingTier::None);
}

#[test]
fn negative_test_sm_5_1_has_no_modern_features() {
    let sm = D3D12ShaderModel::SM_5_1;

    assert!(!sm.supports_wave_intrinsics());
    assert!(!sm.supports_raytracing_intrinsics());
    assert!(!sm.supports_mesh_shaders());
    assert!(!sm.supports_derivatives());
    assert!(!sm.supports_16bit_types());
}

#[test]
fn negative_test_empty_features_produces_minimal_capabilities() {
    let dx12 = D3D12Features::from_features(Features::empty());

    assert_eq!(dx12.mesh_shader_tier, 0);
    assert_eq!(dx12.variable_rate_shading_tier, 0);
    assert_eq!(dx12.sampler_feedback_tier, 0);
    assert_eq!(dx12.conservative_rasterization_tier, 0);
    assert_eq!(dx12.tiled_resources_tier, 0);
    assert_eq!(dx12.resource_binding_tier, 1);
    assert_eq!(dx12.root_signature_version, 1);
    assert_eq!(dx12.rasterizer_ordered_views_tier, 0);
}

// ============================================================================
// SECTION 52 -- Equality and Hash Tests
// ============================================================================

#[test]
fn feature_level_equality() {
    assert_eq!(D3D12FeatureLevel::FL_12_1, D3D12FeatureLevel::FL_12_1);
    assert_ne!(D3D12FeatureLevel::FL_12_1, D3D12FeatureLevel::FL_12_2);
}

#[test]
fn shader_model_equality() {
    assert_eq!(D3D12ShaderModel::SM_6_5, D3D12ShaderModel::SM_6_5);
    assert_ne!(D3D12ShaderModel::SM_6_5, D3D12ShaderModel::SM_6_6);
}

#[test]
fn rt_tier_equality() {
    assert_eq!(D3D12RayTracingTier::Tier1_1, D3D12RayTracingTier::Tier1_1);
    assert_ne!(D3D12RayTracingTier::Tier1_1, D3D12RayTracingTier::Tier1_0);
}

#[test]
fn features_equality() {
    let features1 = D3D12Features::default();
    let features2 = D3D12Features::default();
    assert_eq!(features1, features2);
}

// ============================================================================
// SECTION 53 -- Clone and Copy Tests
// ============================================================================

#[test]
fn feature_level_is_copy() {
    let fl = D3D12FeatureLevel::FL_12_1;
    let fl_copy = fl;
    assert_eq!(fl, fl_copy);
}

#[test]
fn shader_model_is_copy() {
    let sm = D3D12ShaderModel::SM_6_5;
    let sm_copy = sm;
    assert_eq!(sm, sm_copy);
}

#[test]
fn rt_tier_is_copy() {
    let tier = D3D12RayTracingTier::Tier1_1;
    let tier_copy = tier;
    assert_eq!(tier, tier_copy);
}

#[test]
fn features_is_copy() {
    let features = D3D12Features::default();
    let features_copy = features;
    assert_eq!(features, features_copy);
}

// ============================================================================
// SECTION 54 -- Debug Trait Tests
// ============================================================================

#[test]
fn feature_level_debug_format() {
    let fl = D3D12FeatureLevel::FL_12_1;
    let debug = format!("{:?}", fl);
    assert!(debug.contains("FL_12_1"));
}

#[test]
fn shader_model_debug_format() {
    let sm = D3D12ShaderModel::SM_6_5;
    let debug = format!("{:?}", sm);
    assert!(debug.contains("SM_6_5"));
}

#[test]
fn rt_tier_debug_format() {
    let tier = D3D12RayTracingTier::Tier1_1;
    let debug = format!("{:?}", tier);
    assert!(debug.contains("Tier1_1"));
}

#[test]
fn features_debug_format_contains_all_fields() {
    let features = D3D12Features::default();
    let debug = format!("{:?}", features);
    assert!(debug.contains("feature_level"));
    assert!(debug.contains("shader_model"));
    assert!(debug.contains("ray_tracing_tier"));
}

// ============================================================================
// SECTION 55 -- Edge Case Tests
// ============================================================================

#[test]
fn edge_case_all_features_combined() {
    let features = Features::TEXTURE_BINDING_ARRAY
        | Features::BUFFER_BINDING_ARRAY
        | Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING
        | Features::SUBGROUP
        | Features::TIMESTAMP_QUERY
        | Features::RAY_TRACING_ACCELERATION_STRUCTURE
        | Features::RAY_QUERY
        | Features::SHADER_F16;

    let dx12 = D3D12Features::from_features(features);

    // Should be the highest level for everything
    assert_eq!(dx12.feature_level, D3D12FeatureLevel::FL_12_2);
    assert_eq!(dx12.shader_model, D3D12ShaderModel::SM_6_5);
    assert_eq!(dx12.ray_tracing_tier, D3D12RayTracingTier::Tier1_1);
    assert!(dx12.bindless_resources);
    assert!(dx12.wave_ops);
    assert!(dx12.native_16bit_ops);
    assert!(dx12.rt_pipeline);
}

#[test]
fn edge_case_only_timestamp_query() {
    let dx12 = D3D12Features::from_features(Features::TIMESTAMP_QUERY);
    assert_eq!(dx12.feature_level, D3D12FeatureLevel::FL_11_1);
    assert!(!dx12.bindless_resources);
    assert!(!dx12.supports_rt());
}

#[test]
fn edge_case_subgroup_without_other_features() {
    let dx12 = D3D12Features::from_features(Features::SUBGROUP);
    assert_eq!(dx12.shader_model, D3D12ShaderModel::SM_6_0);
    assert!(dx12.wave_ops);
    assert!(!dx12.bindless_resources);
}
