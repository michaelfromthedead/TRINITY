//! Whitebox tests for D3D12-specific feature detection.
//!
//! This module provides comprehensive testing of the D3D12 feature detection
//! system including feature levels, shader models, ray tracing tiers, and
//! the combined D3D12Features struct.
//!
//! Test coverage targets:
//! - All enum variants
//! - All public methods
//! - Trait implementations (Clone, Copy, Debug, Eq, Hash, Display, Ord)
//! - Edge cases and boundary conditions
//! - Feature detection from wgpu Features

use renderer_backend::backend::dx12::{
    D3D12FeatureLevel, D3D12Features, D3D12RayTracingTier, D3D12ShaderModel,
};
use std::collections::HashSet;
use std::hash::Hash;
use wgpu::Features;

// ============================================================================
// D3D12FeatureLevel Tests - Variants and Basic Properties
// ============================================================================

mod feature_level_variants {
    use super::*;

    #[test]
    fn fl_11_0_exists() {
        let fl = D3D12FeatureLevel::FL_11_0;
        assert_eq!(fl.name(), "11.0");
    }

    #[test]
    fn fl_11_1_exists() {
        let fl = D3D12FeatureLevel::FL_11_1;
        assert_eq!(fl.name(), "11.1");
    }

    #[test]
    fn fl_12_0_exists() {
        let fl = D3D12FeatureLevel::FL_12_0;
        assert_eq!(fl.name(), "12.0");
    }

    #[test]
    fn fl_12_1_exists() {
        let fl = D3D12FeatureLevel::FL_12_1;
        assert_eq!(fl.name(), "12.1");
    }

    #[test]
    fn fl_12_2_exists() {
        let fl = D3D12FeatureLevel::FL_12_2;
        assert_eq!(fl.name(), "12.2");
    }

    #[test]
    fn default_is_fl_11_0() {
        assert_eq!(D3D12FeatureLevel::default(), D3D12FeatureLevel::FL_11_0);
    }

    #[test]
    fn all_variants_have_different_names() {
        let variants = [
            D3D12FeatureLevel::FL_11_0,
            D3D12FeatureLevel::FL_11_1,
            D3D12FeatureLevel::FL_12_0,
            D3D12FeatureLevel::FL_12_1,
            D3D12FeatureLevel::FL_12_2,
        ];

        let names: HashSet<&str> = variants.iter().map(|v| v.name()).collect();
        assert_eq!(names.len(), 5, "All variants should have unique names");
    }
}

// ============================================================================
// D3D12FeatureLevel Tests - from_features() Detection
// ============================================================================

mod feature_level_from_features {
    use super::*;

    #[test]
    fn empty_features_gives_fl_11_0() {
        let features = Features::empty();
        assert_eq!(
            D3D12FeatureLevel::from_features(features),
            D3D12FeatureLevel::FL_11_0
        );
    }

    #[test]
    fn timestamp_query_gives_fl_11_1() {
        let features = Features::TIMESTAMP_QUERY;
        assert_eq!(
            D3D12FeatureLevel::from_features(features),
            D3D12FeatureLevel::FL_11_1
        );
    }

    #[test]
    fn texture_binding_array_alone_is_fl_11_0() {
        // Just TEXTURE_BINDING_ARRAY without non-uniform indexing
        let features = Features::TEXTURE_BINDING_ARRAY;
        assert_eq!(
            D3D12FeatureLevel::from_features(features),
            D3D12FeatureLevel::FL_11_0
        );
    }

    #[test]
    fn bindless_combo_gives_fl_12_0() {
        let features = Features::TEXTURE_BINDING_ARRAY
            | Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING;
        assert_eq!(
            D3D12FeatureLevel::from_features(features),
            D3D12FeatureLevel::FL_12_0
        );
    }

    #[test]
    fn ray_tracing_gives_fl_12_1() {
        let features = Features::RAY_TRACING_ACCELERATION_STRUCTURE;
        assert_eq!(
            D3D12FeatureLevel::from_features(features),
            D3D12FeatureLevel::FL_12_1
        );
    }

    #[test]
    fn ray_tracing_with_ray_query_gives_fl_12_2() {
        let features = Features::RAY_TRACING_ACCELERATION_STRUCTURE | Features::RAY_QUERY;
        assert_eq!(
            D3D12FeatureLevel::from_features(features),
            D3D12FeatureLevel::FL_12_2
        );
    }

    #[test]
    fn ray_query_alone_is_not_fl_12_2() {
        // RAY_QUERY without RAY_TRACING_ACCELERATION_STRUCTURE
        let features = Features::RAY_QUERY;
        // Should not be FL_12_2 since both are required
        let fl = D3D12FeatureLevel::from_features(features);
        assert_ne!(fl, D3D12FeatureLevel::FL_12_2);
    }

    #[test]
    fn rt_takes_priority_over_bindless() {
        // RT features should override bindless detection
        let features = Features::RAY_TRACING_ACCELERATION_STRUCTURE
            | Features::TEXTURE_BINDING_ARRAY
            | Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING;
        assert_eq!(
            D3D12FeatureLevel::from_features(features),
            D3D12FeatureLevel::FL_12_1
        );
    }

    #[test]
    fn full_rt_takes_priority_over_all() {
        let features = Features::RAY_TRACING_ACCELERATION_STRUCTURE
            | Features::RAY_QUERY
            | Features::TEXTURE_BINDING_ARRAY
            | Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING
            | Features::TIMESTAMP_QUERY;
        assert_eq!(
            D3D12FeatureLevel::from_features(features),
            D3D12FeatureLevel::FL_12_2
        );
    }

    #[test]
    fn timestamp_takes_priority_over_nothing() {
        let features = Features::TIMESTAMP_QUERY | Features::DEPTH_CLIP_CONTROL;
        assert_eq!(
            D3D12FeatureLevel::from_features(features),
            D3D12FeatureLevel::FL_11_1
        );
    }

    #[test]
    fn bindless_takes_priority_over_timestamp() {
        let features = Features::TEXTURE_BINDING_ARRAY
            | Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING
            | Features::TIMESTAMP_QUERY;
        assert_eq!(
            D3D12FeatureLevel::from_features(features),
            D3D12FeatureLevel::FL_12_0
        );
    }
}

// ============================================================================
// D3D12FeatureLevel Tests - supports_ray_tracing()
// ============================================================================

mod feature_level_supports_ray_tracing {
    use super::*;

    #[test]
    fn fl_11_0_does_not_support_rt() {
        assert!(!D3D12FeatureLevel::FL_11_0.supports_ray_tracing());
    }

    #[test]
    fn fl_11_1_does_not_support_rt() {
        assert!(!D3D12FeatureLevel::FL_11_1.supports_ray_tracing());
    }

    #[test]
    fn fl_12_0_does_not_support_rt() {
        assert!(!D3D12FeatureLevel::FL_12_0.supports_ray_tracing());
    }

    #[test]
    fn fl_12_1_supports_rt() {
        assert!(D3D12FeatureLevel::FL_12_1.supports_ray_tracing());
    }

    #[test]
    fn fl_12_2_supports_rt() {
        assert!(D3D12FeatureLevel::FL_12_2.supports_ray_tracing());
    }

    #[test]
    fn only_fl_12_1_and_12_2_support_rt() {
        let variants = [
            D3D12FeatureLevel::FL_11_0,
            D3D12FeatureLevel::FL_11_1,
            D3D12FeatureLevel::FL_12_0,
            D3D12FeatureLevel::FL_12_1,
            D3D12FeatureLevel::FL_12_2,
        ];

        let rt_support: Vec<bool> = variants.iter().map(|v| v.supports_ray_tracing()).collect();
        assert_eq!(rt_support, vec![false, false, false, true, true]);
    }
}

// ============================================================================
// D3D12FeatureLevel Tests - supports_mesh_shaders()
// ============================================================================

mod feature_level_supports_mesh_shaders {
    use super::*;

    #[test]
    fn fl_11_0_does_not_support_mesh() {
        assert!(!D3D12FeatureLevel::FL_11_0.supports_mesh_shaders());
    }

    #[test]
    fn fl_11_1_does_not_support_mesh() {
        assert!(!D3D12FeatureLevel::FL_11_1.supports_mesh_shaders());
    }

    #[test]
    fn fl_12_0_does_not_support_mesh() {
        assert!(!D3D12FeatureLevel::FL_12_0.supports_mesh_shaders());
    }

    #[test]
    fn fl_12_1_does_not_support_mesh() {
        assert!(!D3D12FeatureLevel::FL_12_1.supports_mesh_shaders());
    }

    #[test]
    fn fl_12_2_supports_mesh() {
        assert!(D3D12FeatureLevel::FL_12_2.supports_mesh_shaders());
    }

    #[test]
    fn only_fl_12_2_supports_mesh() {
        let variants = [
            D3D12FeatureLevel::FL_11_0,
            D3D12FeatureLevel::FL_11_1,
            D3D12FeatureLevel::FL_12_0,
            D3D12FeatureLevel::FL_12_1,
            D3D12FeatureLevel::FL_12_2,
        ];

        let mesh_support: Vec<bool> = variants.iter().map(|v| v.supports_mesh_shaders()).collect();
        assert_eq!(mesh_support, vec![false, false, false, false, true]);
    }
}

// ============================================================================
// D3D12FeatureLevel Tests - supports_variable_rate_shading()
// ============================================================================

mod feature_level_supports_vrs {
    use super::*;

    #[test]
    fn fl_11_0_does_not_support_vrs() {
        assert!(!D3D12FeatureLevel::FL_11_0.supports_variable_rate_shading());
    }

    #[test]
    fn fl_11_1_does_not_support_vrs() {
        assert!(!D3D12FeatureLevel::FL_11_1.supports_variable_rate_shading());
    }

    #[test]
    fn fl_12_0_does_not_support_vrs() {
        assert!(!D3D12FeatureLevel::FL_12_0.supports_variable_rate_shading());
    }

    #[test]
    fn fl_12_1_supports_vrs() {
        assert!(D3D12FeatureLevel::FL_12_1.supports_variable_rate_shading());
    }

    #[test]
    fn fl_12_2_supports_vrs() {
        assert!(D3D12FeatureLevel::FL_12_2.supports_variable_rate_shading());
    }

    #[test]
    fn vrs_and_rt_both_start_at_fl_12_1() {
        // VRS and RT both start at FL 12.1
        let variants = [
            D3D12FeatureLevel::FL_11_0,
            D3D12FeatureLevel::FL_11_1,
            D3D12FeatureLevel::FL_12_0,
            D3D12FeatureLevel::FL_12_1,
            D3D12FeatureLevel::FL_12_2,
        ];

        for v in &variants {
            assert_eq!(
                v.supports_variable_rate_shading(),
                v.supports_ray_tracing(),
                "VRS and RT support should match for {:?}",
                v
            );
        }
    }
}

// ============================================================================
// D3D12FeatureLevel Tests - supports_sampler_feedback()
// ============================================================================

mod feature_level_supports_sampler_feedback {
    use super::*;

    #[test]
    fn fl_11_0_does_not_support_sampler_feedback() {
        assert!(!D3D12FeatureLevel::FL_11_0.supports_sampler_feedback());
    }

    #[test]
    fn fl_11_1_does_not_support_sampler_feedback() {
        assert!(!D3D12FeatureLevel::FL_11_1.supports_sampler_feedback());
    }

    #[test]
    fn fl_12_0_does_not_support_sampler_feedback() {
        assert!(!D3D12FeatureLevel::FL_12_0.supports_sampler_feedback());
    }

    #[test]
    fn fl_12_1_does_not_support_sampler_feedback() {
        assert!(!D3D12FeatureLevel::FL_12_1.supports_sampler_feedback());
    }

    #[test]
    fn fl_12_2_supports_sampler_feedback() {
        assert!(D3D12FeatureLevel::FL_12_2.supports_sampler_feedback());
    }

    #[test]
    fn sampler_feedback_same_as_mesh_shaders() {
        // Both require FL 12.2
        for fl in [
            D3D12FeatureLevel::FL_11_0,
            D3D12FeatureLevel::FL_11_1,
            D3D12FeatureLevel::FL_12_0,
            D3D12FeatureLevel::FL_12_1,
            D3D12FeatureLevel::FL_12_2,
        ] {
            assert_eq!(
                fl.supports_sampler_feedback(),
                fl.supports_mesh_shaders(),
                "Sampler feedback and mesh shader support should match for {:?}",
                fl
            );
        }
    }
}

// ============================================================================
// D3D12FeatureLevel Tests - name()
// ============================================================================

mod feature_level_name {
    use super::*;

    #[test]
    fn name_format_is_major_dot_minor() {
        let variants = [
            (D3D12FeatureLevel::FL_11_0, "11.0"),
            (D3D12FeatureLevel::FL_11_1, "11.1"),
            (D3D12FeatureLevel::FL_12_0, "12.0"),
            (D3D12FeatureLevel::FL_12_1, "12.1"),
            (D3D12FeatureLevel::FL_12_2, "12.2"),
        ];

        for (fl, expected) in variants {
            assert_eq!(fl.name(), expected);
        }
    }

    #[test]
    fn name_returns_static_str() {
        let name: &'static str = D3D12FeatureLevel::FL_12_0.name();
        assert_eq!(name, "12.0");
    }
}

// ============================================================================
// D3D12FeatureLevel Tests - d3d_feature_level_value()
// ============================================================================

mod feature_level_d3d_value {
    use super::*;

    #[test]
    fn fl_11_0_is_0xb000() {
        assert_eq!(D3D12FeatureLevel::FL_11_0.d3d_feature_level_value(), 0xb000);
    }

    #[test]
    fn fl_11_1_is_0xb100() {
        assert_eq!(D3D12FeatureLevel::FL_11_1.d3d_feature_level_value(), 0xb100);
    }

    #[test]
    fn fl_12_0_is_0xc000() {
        assert_eq!(D3D12FeatureLevel::FL_12_0.d3d_feature_level_value(), 0xc000);
    }

    #[test]
    fn fl_12_1_is_0xc100() {
        assert_eq!(D3D12FeatureLevel::FL_12_1.d3d_feature_level_value(), 0xc100);
    }

    #[test]
    fn fl_12_2_is_0xc200() {
        assert_eq!(D3D12FeatureLevel::FL_12_2.d3d_feature_level_value(), 0xc200);
    }

    #[test]
    fn d3d_values_are_ordered() {
        let values: Vec<u32> = [
            D3D12FeatureLevel::FL_11_0,
            D3D12FeatureLevel::FL_11_1,
            D3D12FeatureLevel::FL_12_0,
            D3D12FeatureLevel::FL_12_1,
            D3D12FeatureLevel::FL_12_2,
        ]
        .iter()
        .map(|v| v.d3d_feature_level_value())
        .collect();

        for i in 1..values.len() {
            assert!(
                values[i] > values[i - 1],
                "D3D values should be strictly increasing"
            );
        }
    }

    #[test]
    fn d3d_values_match_windows_sdk_definitions() {
        // D3D_FEATURE_LEVEL values from the Windows SDK
        assert_eq!(D3D12FeatureLevel::FL_11_0.d3d_feature_level_value(), 45056); // 0xb000
        assert_eq!(D3D12FeatureLevel::FL_11_1.d3d_feature_level_value(), 45312); // 0xb100
        assert_eq!(D3D12FeatureLevel::FL_12_0.d3d_feature_level_value(), 49152); // 0xc000
        assert_eq!(D3D12FeatureLevel::FL_12_1.d3d_feature_level_value(), 49408); // 0xc100
        assert_eq!(D3D12FeatureLevel::FL_12_2.d3d_feature_level_value(), 49664); // 0xc200
    }
}

// ============================================================================
// D3D12FeatureLevel Tests - Trait Implementations
// ============================================================================

mod feature_level_traits {
    use super::*;

    #[test]
    fn clone_works() {
        let fl = D3D12FeatureLevel::FL_12_1;
        let cloned = fl.clone();
        assert_eq!(fl, cloned);
    }

    #[test]
    fn copy_works() {
        let fl = D3D12FeatureLevel::FL_12_1;
        let copied = fl;
        assert_eq!(fl, copied);
    }

    #[test]
    fn debug_format_is_readable() {
        let debug_str = format!("{:?}", D3D12FeatureLevel::FL_12_1);
        assert!(debug_str.contains("FL_12_1"));
    }

    #[test]
    fn eq_works_for_same_variants() {
        assert_eq!(D3D12FeatureLevel::FL_12_0, D3D12FeatureLevel::FL_12_0);
    }

    #[test]
    fn ne_works_for_different_variants() {
        assert_ne!(D3D12FeatureLevel::FL_11_0, D3D12FeatureLevel::FL_12_0);
    }

    #[test]
    fn hash_is_different_for_different_variants() {
        fn hash_value<T: Hash>(t: &T) -> u64 {
            use std::collections::hash_map::DefaultHasher;
            use std::hash::Hasher;
            let mut hasher = DefaultHasher::new();
            t.hash(&mut hasher);
            hasher.finish()
        }

        let h1 = hash_value(&D3D12FeatureLevel::FL_11_0);
        let h2 = hash_value(&D3D12FeatureLevel::FL_12_0);
        assert_ne!(h1, h2);
    }

    #[test]
    fn can_use_in_hashset() {
        let mut set: HashSet<D3D12FeatureLevel> = HashSet::new();
        set.insert(D3D12FeatureLevel::FL_11_0);
        set.insert(D3D12FeatureLevel::FL_12_0);
        set.insert(D3D12FeatureLevel::FL_11_0); // duplicate

        assert_eq!(set.len(), 2);
    }

    #[test]
    fn display_format() {
        assert_eq!(
            format!("{}", D3D12FeatureLevel::FL_11_0),
            "D3D_FEATURE_LEVEL_11_0"
        );
        assert_eq!(
            format!("{}", D3D12FeatureLevel::FL_11_1),
            "D3D_FEATURE_LEVEL_11_1"
        );
        assert_eq!(
            format!("{}", D3D12FeatureLevel::FL_12_0),
            "D3D_FEATURE_LEVEL_12_0"
        );
        assert_eq!(
            format!("{}", D3D12FeatureLevel::FL_12_1),
            "D3D_FEATURE_LEVEL_12_1"
        );
        assert_eq!(
            format!("{}", D3D12FeatureLevel::FL_12_2),
            "D3D_FEATURE_LEVEL_12_2"
        );
    }

    #[test]
    fn ord_less_than() {
        assert!(D3D12FeatureLevel::FL_11_0 < D3D12FeatureLevel::FL_11_1);
        assert!(D3D12FeatureLevel::FL_11_1 < D3D12FeatureLevel::FL_12_0);
        assert!(D3D12FeatureLevel::FL_12_0 < D3D12FeatureLevel::FL_12_1);
        assert!(D3D12FeatureLevel::FL_12_1 < D3D12FeatureLevel::FL_12_2);
    }

    #[test]
    fn ord_greater_than() {
        assert!(D3D12FeatureLevel::FL_12_2 > D3D12FeatureLevel::FL_12_1);
        assert!(D3D12FeatureLevel::FL_12_1 > D3D12FeatureLevel::FL_12_0);
        assert!(D3D12FeatureLevel::FL_12_0 > D3D12FeatureLevel::FL_11_1);
        assert!(D3D12FeatureLevel::FL_11_1 > D3D12FeatureLevel::FL_11_0);
    }

    #[test]
    fn ord_less_than_or_equal() {
        assert!(D3D12FeatureLevel::FL_12_0 <= D3D12FeatureLevel::FL_12_0);
        assert!(D3D12FeatureLevel::FL_12_0 <= D3D12FeatureLevel::FL_12_1);
    }

    #[test]
    fn ord_greater_than_or_equal() {
        assert!(D3D12FeatureLevel::FL_12_0 >= D3D12FeatureLevel::FL_12_0);
        assert!(D3D12FeatureLevel::FL_12_1 >= D3D12FeatureLevel::FL_12_0);
    }

    #[test]
    fn can_sort_feature_levels() {
        let mut levels = vec![
            D3D12FeatureLevel::FL_12_2,
            D3D12FeatureLevel::FL_11_0,
            D3D12FeatureLevel::FL_12_0,
        ];
        levels.sort();

        assert_eq!(
            levels,
            vec![
                D3D12FeatureLevel::FL_11_0,
                D3D12FeatureLevel::FL_12_0,
                D3D12FeatureLevel::FL_12_2,
            ]
        );
    }
}

// ============================================================================
// D3D12ShaderModel Tests - Variants and Basic Properties
// ============================================================================

mod shader_model_variants {
    use super::*;

    #[test]
    fn sm_5_1_exists() {
        assert_eq!(D3D12ShaderModel::SM_5_1.name(), "5.1");
    }

    #[test]
    fn sm_6_0_exists() {
        assert_eq!(D3D12ShaderModel::SM_6_0.name(), "6.0");
    }

    #[test]
    fn sm_6_1_exists() {
        assert_eq!(D3D12ShaderModel::SM_6_1.name(), "6.1");
    }

    #[test]
    fn sm_6_2_exists() {
        assert_eq!(D3D12ShaderModel::SM_6_2.name(), "6.2");
    }

    #[test]
    fn sm_6_3_exists() {
        assert_eq!(D3D12ShaderModel::SM_6_3.name(), "6.3");
    }

    #[test]
    fn sm_6_4_exists() {
        assert_eq!(D3D12ShaderModel::SM_6_4.name(), "6.4");
    }

    #[test]
    fn sm_6_5_exists() {
        assert_eq!(D3D12ShaderModel::SM_6_5.name(), "6.5");
    }

    #[test]
    fn sm_6_6_exists() {
        assert_eq!(D3D12ShaderModel::SM_6_6.name(), "6.6");
    }

    #[test]
    fn sm_6_7_exists() {
        assert_eq!(D3D12ShaderModel::SM_6_7.name(), "6.7");
    }

    #[test]
    fn default_is_sm_5_1() {
        assert_eq!(D3D12ShaderModel::default(), D3D12ShaderModel::SM_5_1);
    }

    #[test]
    fn all_variants_have_different_names() {
        let variants = [
            D3D12ShaderModel::SM_5_1,
            D3D12ShaderModel::SM_6_0,
            D3D12ShaderModel::SM_6_1,
            D3D12ShaderModel::SM_6_2,
            D3D12ShaderModel::SM_6_3,
            D3D12ShaderModel::SM_6_4,
            D3D12ShaderModel::SM_6_5,
            D3D12ShaderModel::SM_6_6,
            D3D12ShaderModel::SM_6_7,
        ];

        let names: HashSet<&str> = variants.iter().map(|v| v.name()).collect();
        assert_eq!(names.len(), 9, "All shader model variants should have unique names");
    }
}

// ============================================================================
// D3D12ShaderModel Tests - from_features() Detection
// ============================================================================

mod shader_model_from_features {
    use super::*;

    #[test]
    fn empty_features_gives_sm_5_1() {
        let features = Features::empty();
        assert_eq!(
            D3D12ShaderModel::from_features(features),
            D3D12ShaderModel::SM_5_1
        );
    }

    #[test]
    fn subgroup_feature_gives_sm_6_0() {
        let features = Features::SUBGROUP;
        assert_eq!(
            D3D12ShaderModel::from_features(features),
            D3D12ShaderModel::SM_6_0
        );
    }

    #[test]
    fn ray_tracing_gives_sm_6_3() {
        let features = Features::RAY_TRACING_ACCELERATION_STRUCTURE;
        assert_eq!(
            D3D12ShaderModel::from_features(features),
            D3D12ShaderModel::SM_6_3
        );
    }

    #[test]
    fn ray_query_gives_sm_6_5() {
        let features = Features::RAY_QUERY;
        assert_eq!(
            D3D12ShaderModel::from_features(features),
            D3D12ShaderModel::SM_6_5
        );
    }

    #[test]
    fn full_rt_gives_sm_6_5() {
        let features = Features::RAY_TRACING_ACCELERATION_STRUCTURE | Features::RAY_QUERY;
        assert_eq!(
            D3D12ShaderModel::from_features(features),
            D3D12ShaderModel::SM_6_5
        );
    }

    #[test]
    fn bindless_only_gives_sm_5_1() {
        let features = Features::TEXTURE_BINDING_ARRAY
            | Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING;
        assert_eq!(
            D3D12ShaderModel::from_features(features),
            D3D12ShaderModel::SM_5_1
        );
    }

    #[test]
    fn ray_query_takes_priority() {
        let features =
            Features::RAY_QUERY | Features::RAY_TRACING_ACCELERATION_STRUCTURE | Features::SUBGROUP;
        assert_eq!(
            D3D12ShaderModel::from_features(features),
            D3D12ShaderModel::SM_6_5
        );
    }
}

// ============================================================================
// D3D12ShaderModel Tests - supports_wave_intrinsics()
// ============================================================================

mod shader_model_supports_wave_intrinsics {
    use super::*;

    #[test]
    fn sm_5_1_does_not_support_wave() {
        assert!(!D3D12ShaderModel::SM_5_1.supports_wave_intrinsics());
    }

    #[test]
    fn sm_6_0_supports_wave() {
        assert!(D3D12ShaderModel::SM_6_0.supports_wave_intrinsics());
    }

    #[test]
    fn sm_6_1_supports_wave() {
        assert!(D3D12ShaderModel::SM_6_1.supports_wave_intrinsics());
    }

    #[test]
    fn sm_6_2_supports_wave() {
        assert!(D3D12ShaderModel::SM_6_2.supports_wave_intrinsics());
    }

    #[test]
    fn sm_6_3_supports_wave() {
        assert!(D3D12ShaderModel::SM_6_3.supports_wave_intrinsics());
    }

    #[test]
    fn sm_6_4_supports_wave() {
        assert!(D3D12ShaderModel::SM_6_4.supports_wave_intrinsics());
    }

    #[test]
    fn sm_6_5_supports_wave() {
        assert!(D3D12ShaderModel::SM_6_5.supports_wave_intrinsics());
    }

    #[test]
    fn sm_6_6_supports_wave() {
        assert!(D3D12ShaderModel::SM_6_6.supports_wave_intrinsics());
    }

    #[test]
    fn sm_6_7_supports_wave() {
        assert!(D3D12ShaderModel::SM_6_7.supports_wave_intrinsics());
    }

    #[test]
    fn only_sm_5_1_lacks_wave_support() {
        let variants = [
            D3D12ShaderModel::SM_5_1,
            D3D12ShaderModel::SM_6_0,
            D3D12ShaderModel::SM_6_1,
            D3D12ShaderModel::SM_6_2,
            D3D12ShaderModel::SM_6_3,
            D3D12ShaderModel::SM_6_4,
            D3D12ShaderModel::SM_6_5,
            D3D12ShaderModel::SM_6_6,
            D3D12ShaderModel::SM_6_7,
        ];

        let wave_support: Vec<bool> = variants
            .iter()
            .map(|v| v.supports_wave_intrinsics())
            .collect();
        assert_eq!(
            wave_support,
            vec![false, true, true, true, true, true, true, true, true]
        );
    }
}

// ============================================================================
// D3D12ShaderModel Tests - supports_raytracing_intrinsics()
// ============================================================================

mod shader_model_supports_raytracing_intrinsics {
    use super::*;

    #[test]
    fn sm_5_1_does_not_support_rt() {
        assert!(!D3D12ShaderModel::SM_5_1.supports_raytracing_intrinsics());
    }

    #[test]
    fn sm_6_0_does_not_support_rt() {
        assert!(!D3D12ShaderModel::SM_6_0.supports_raytracing_intrinsics());
    }

    #[test]
    fn sm_6_1_does_not_support_rt() {
        assert!(!D3D12ShaderModel::SM_6_1.supports_raytracing_intrinsics());
    }

    #[test]
    fn sm_6_2_does_not_support_rt() {
        assert!(!D3D12ShaderModel::SM_6_2.supports_raytracing_intrinsics());
    }

    #[test]
    fn sm_6_3_supports_rt() {
        assert!(D3D12ShaderModel::SM_6_3.supports_raytracing_intrinsics());
    }

    #[test]
    fn sm_6_4_supports_rt() {
        assert!(D3D12ShaderModel::SM_6_4.supports_raytracing_intrinsics());
    }

    #[test]
    fn sm_6_5_supports_rt() {
        assert!(D3D12ShaderModel::SM_6_5.supports_raytracing_intrinsics());
    }

    #[test]
    fn sm_6_6_supports_rt() {
        assert!(D3D12ShaderModel::SM_6_6.supports_raytracing_intrinsics());
    }

    #[test]
    fn sm_6_7_supports_rt() {
        assert!(D3D12ShaderModel::SM_6_7.supports_raytracing_intrinsics());
    }

    #[test]
    fn rt_starts_at_sm_6_3() {
        let variants = [
            D3D12ShaderModel::SM_5_1,
            D3D12ShaderModel::SM_6_0,
            D3D12ShaderModel::SM_6_1,
            D3D12ShaderModel::SM_6_2,
            D3D12ShaderModel::SM_6_3,
            D3D12ShaderModel::SM_6_4,
            D3D12ShaderModel::SM_6_5,
            D3D12ShaderModel::SM_6_6,
            D3D12ShaderModel::SM_6_7,
        ];

        let rt_support: Vec<bool> = variants
            .iter()
            .map(|v| v.supports_raytracing_intrinsics())
            .collect();
        assert_eq!(
            rt_support,
            vec![false, false, false, false, true, true, true, true, true]
        );
    }
}

// ============================================================================
// D3D12ShaderModel Tests - supports_mesh_shaders()
// ============================================================================

mod shader_model_supports_mesh_shaders {
    use super::*;

    #[test]
    fn sm_5_1_does_not_support_mesh() {
        assert!(!D3D12ShaderModel::SM_5_1.supports_mesh_shaders());
    }

    #[test]
    fn sm_6_0_does_not_support_mesh() {
        assert!(!D3D12ShaderModel::SM_6_0.supports_mesh_shaders());
    }

    #[test]
    fn sm_6_4_does_not_support_mesh() {
        assert!(!D3D12ShaderModel::SM_6_4.supports_mesh_shaders());
    }

    #[test]
    fn sm_6_5_supports_mesh() {
        assert!(D3D12ShaderModel::SM_6_5.supports_mesh_shaders());
    }

    #[test]
    fn sm_6_6_supports_mesh() {
        assert!(D3D12ShaderModel::SM_6_6.supports_mesh_shaders());
    }

    #[test]
    fn sm_6_7_supports_mesh() {
        assert!(D3D12ShaderModel::SM_6_7.supports_mesh_shaders());
    }

    #[test]
    fn mesh_starts_at_sm_6_5() {
        let variants = [
            D3D12ShaderModel::SM_5_1,
            D3D12ShaderModel::SM_6_0,
            D3D12ShaderModel::SM_6_1,
            D3D12ShaderModel::SM_6_2,
            D3D12ShaderModel::SM_6_3,
            D3D12ShaderModel::SM_6_4,
            D3D12ShaderModel::SM_6_5,
            D3D12ShaderModel::SM_6_6,
            D3D12ShaderModel::SM_6_7,
        ];

        let mesh_support: Vec<bool> = variants
            .iter()
            .map(|v| v.supports_mesh_shaders())
            .collect();
        assert_eq!(
            mesh_support,
            vec![false, false, false, false, false, false, true, true, true]
        );
    }
}

// ============================================================================
// D3D12ShaderModel Tests - supports_derivatives()
// ============================================================================

mod shader_model_supports_derivatives {
    use super::*;

    #[test]
    fn sm_5_1_does_not_support_derivatives() {
        assert!(!D3D12ShaderModel::SM_5_1.supports_derivatives());
    }

    #[test]
    fn sm_6_5_does_not_support_derivatives() {
        assert!(!D3D12ShaderModel::SM_6_5.supports_derivatives());
    }

    #[test]
    fn sm_6_6_supports_derivatives() {
        assert!(D3D12ShaderModel::SM_6_6.supports_derivatives());
    }

    #[test]
    fn sm_6_7_supports_derivatives() {
        assert!(D3D12ShaderModel::SM_6_7.supports_derivatives());
    }

    #[test]
    fn derivatives_start_at_sm_6_6() {
        let variants = [
            D3D12ShaderModel::SM_5_1,
            D3D12ShaderModel::SM_6_0,
            D3D12ShaderModel::SM_6_1,
            D3D12ShaderModel::SM_6_2,
            D3D12ShaderModel::SM_6_3,
            D3D12ShaderModel::SM_6_4,
            D3D12ShaderModel::SM_6_5,
            D3D12ShaderModel::SM_6_6,
            D3D12ShaderModel::SM_6_7,
        ];

        let deriv_support: Vec<bool> = variants.iter().map(|v| v.supports_derivatives()).collect();
        assert_eq!(
            deriv_support,
            vec![false, false, false, false, false, false, false, true, true]
        );
    }
}

// ============================================================================
// D3D12ShaderModel Tests - supports_16bit_types()
// ============================================================================

mod shader_model_supports_16bit_types {
    use super::*;

    #[test]
    fn sm_5_1_does_not_support_16bit() {
        assert!(!D3D12ShaderModel::SM_5_1.supports_16bit_types());
    }

    #[test]
    fn sm_6_0_does_not_support_16bit() {
        assert!(!D3D12ShaderModel::SM_6_0.supports_16bit_types());
    }

    #[test]
    fn sm_6_1_does_not_support_16bit() {
        assert!(!D3D12ShaderModel::SM_6_1.supports_16bit_types());
    }

    #[test]
    fn sm_6_2_supports_16bit() {
        assert!(D3D12ShaderModel::SM_6_2.supports_16bit_types());
    }

    #[test]
    fn sm_6_3_supports_16bit() {
        assert!(D3D12ShaderModel::SM_6_3.supports_16bit_types());
    }

    #[test]
    fn sm_6_7_supports_16bit() {
        assert!(D3D12ShaderModel::SM_6_7.supports_16bit_types());
    }

    #[test]
    fn type_16bit_starts_at_sm_6_2() {
        let variants = [
            D3D12ShaderModel::SM_5_1,
            D3D12ShaderModel::SM_6_0,
            D3D12ShaderModel::SM_6_1,
            D3D12ShaderModel::SM_6_2,
            D3D12ShaderModel::SM_6_3,
            D3D12ShaderModel::SM_6_4,
            D3D12ShaderModel::SM_6_5,
            D3D12ShaderModel::SM_6_6,
            D3D12ShaderModel::SM_6_7,
        ];

        let support_16bit: Vec<bool> =
            variants.iter().map(|v| v.supports_16bit_types()).collect();
        assert_eq!(
            support_16bit,
            vec![false, false, false, true, true, true, true, true, true]
        );
    }
}

// ============================================================================
// D3D12ShaderModel Tests - name() and version()
// ============================================================================

mod shader_model_name_version {
    use super::*;

    #[test]
    fn name_format() {
        assert_eq!(D3D12ShaderModel::SM_5_1.name(), "5.1");
        assert_eq!(D3D12ShaderModel::SM_6_0.name(), "6.0");
        assert_eq!(D3D12ShaderModel::SM_6_7.name(), "6.7");
    }

    #[test]
    fn version_tuple_sm_5_1() {
        assert_eq!(D3D12ShaderModel::SM_5_1.version(), (5, 1));
    }

    #[test]
    fn version_tuple_sm_6_0() {
        assert_eq!(D3D12ShaderModel::SM_6_0.version(), (6, 0));
    }

    #[test]
    fn version_tuple_sm_6_7() {
        assert_eq!(D3D12ShaderModel::SM_6_7.version(), (6, 7));
    }

    #[test]
    fn all_versions() {
        let versions: Vec<(u8, u8)> = [
            D3D12ShaderModel::SM_5_1,
            D3D12ShaderModel::SM_6_0,
            D3D12ShaderModel::SM_6_1,
            D3D12ShaderModel::SM_6_2,
            D3D12ShaderModel::SM_6_3,
            D3D12ShaderModel::SM_6_4,
            D3D12ShaderModel::SM_6_5,
            D3D12ShaderModel::SM_6_6,
            D3D12ShaderModel::SM_6_7,
        ]
        .iter()
        .map(|v| v.version())
        .collect();

        assert_eq!(
            versions,
            vec![
                (5, 1),
                (6, 0),
                (6, 1),
                (6, 2),
                (6, 3),
                (6, 4),
                (6, 5),
                (6, 6),
                (6, 7)
            ]
        );
    }
}

// ============================================================================
// D3D12ShaderModel Tests - Ordering Comparisons
// ============================================================================

mod shader_model_ordering {
    use super::*;

    #[test]
    fn sm_6_7_greater_than_sm_6_0() {
        assert!(D3D12ShaderModel::SM_6_7 > D3D12ShaderModel::SM_6_0);
    }

    #[test]
    fn sm_6_0_greater_than_sm_5_1() {
        assert!(D3D12ShaderModel::SM_6_0 > D3D12ShaderModel::SM_5_1);
    }

    #[test]
    fn sm_6_5_equals_sm_6_5() {
        assert!(D3D12ShaderModel::SM_6_5 == D3D12ShaderModel::SM_6_5);
    }

    #[test]
    fn strict_ordering_all_variants() {
        let variants = [
            D3D12ShaderModel::SM_5_1,
            D3D12ShaderModel::SM_6_0,
            D3D12ShaderModel::SM_6_1,
            D3D12ShaderModel::SM_6_2,
            D3D12ShaderModel::SM_6_3,
            D3D12ShaderModel::SM_6_4,
            D3D12ShaderModel::SM_6_5,
            D3D12ShaderModel::SM_6_6,
            D3D12ShaderModel::SM_6_7,
        ];

        for i in 1..variants.len() {
            assert!(
                variants[i] > variants[i - 1],
                "{:?} should be > {:?}",
                variants[i],
                variants[i - 1]
            );
        }
    }

    #[test]
    fn can_sort_shader_models() {
        let mut models = vec![
            D3D12ShaderModel::SM_6_7,
            D3D12ShaderModel::SM_5_1,
            D3D12ShaderModel::SM_6_3,
        ];
        models.sort();

        assert_eq!(
            models,
            vec![
                D3D12ShaderModel::SM_5_1,
                D3D12ShaderModel::SM_6_3,
                D3D12ShaderModel::SM_6_7,
            ]
        );
    }

    #[test]
    fn display_format() {
        assert_eq!(format!("{}", D3D12ShaderModel::SM_5_1), "SM 5.1");
        assert_eq!(format!("{}", D3D12ShaderModel::SM_6_5), "SM 6.5");
    }
}

// ============================================================================
// D3D12RayTracingTier Tests - Variants and Basic Properties
// ============================================================================

mod ray_tracing_tier_variants {
    use super::*;

    #[test]
    fn none_exists() {
        assert_eq!(D3D12RayTracingTier::None.name(), "None");
    }

    #[test]
    fn tier_1_0_exists() {
        assert_eq!(D3D12RayTracingTier::Tier1_0.name(), "DXR 1.0");
    }

    #[test]
    fn tier_1_1_exists() {
        assert_eq!(D3D12RayTracingTier::Tier1_1.name(), "DXR 1.1");
    }

    #[test]
    fn default_is_none() {
        assert_eq!(D3D12RayTracingTier::default(), D3D12RayTracingTier::None);
    }
}

// ============================================================================
// D3D12RayTracingTier Tests - from_features() Detection
// ============================================================================

mod ray_tracing_tier_from_features {
    use super::*;

    #[test]
    fn empty_features_gives_none() {
        let features = Features::empty();
        assert_eq!(
            D3D12RayTracingTier::from_features(features),
            D3D12RayTracingTier::None
        );
    }

    #[test]
    fn rt_only_gives_tier_1_0() {
        let features = Features::RAY_TRACING_ACCELERATION_STRUCTURE;
        assert_eq!(
            D3D12RayTracingTier::from_features(features),
            D3D12RayTracingTier::Tier1_0
        );
    }

    #[test]
    fn rt_with_ray_query_gives_tier_1_1() {
        let features = Features::RAY_TRACING_ACCELERATION_STRUCTURE | Features::RAY_QUERY;
        assert_eq!(
            D3D12RayTracingTier::from_features(features),
            D3D12RayTracingTier::Tier1_1
        );
    }

    #[test]
    fn ray_query_alone_gives_none() {
        // RAY_QUERY without RT acceleration structure
        let features = Features::RAY_QUERY;
        assert_eq!(
            D3D12RayTracingTier::from_features(features),
            D3D12RayTracingTier::None
        );
    }
}

// ============================================================================
// D3D12RayTracingTier Tests - supports_inline_raytracing()
// ============================================================================

mod ray_tracing_tier_supports_inline_raytracing {
    use super::*;

    #[test]
    fn none_does_not_support_inline() {
        assert!(!D3D12RayTracingTier::None.supports_inline_raytracing());
    }

    #[test]
    fn tier_1_0_does_not_support_inline() {
        assert!(!D3D12RayTracingTier::Tier1_0.supports_inline_raytracing());
    }

    #[test]
    fn tier_1_1_supports_inline() {
        assert!(D3D12RayTracingTier::Tier1_1.supports_inline_raytracing());
    }
}

// ============================================================================
// D3D12RayTracingTier Tests - supports_rayquery()
// ============================================================================

mod ray_tracing_tier_supports_rayquery {
    use super::*;

    #[test]
    fn none_does_not_support_rayquery() {
        assert!(!D3D12RayTracingTier::None.supports_rayquery());
    }

    #[test]
    fn tier_1_0_does_not_support_rayquery() {
        assert!(!D3D12RayTracingTier::Tier1_0.supports_rayquery());
    }

    #[test]
    fn tier_1_1_supports_rayquery() {
        assert!(D3D12RayTracingTier::Tier1_1.supports_rayquery());
    }

    #[test]
    fn inline_and_rayquery_match() {
        // Both should be exclusive to Tier1_1
        for tier in [
            D3D12RayTracingTier::None,
            D3D12RayTracingTier::Tier1_0,
            D3D12RayTracingTier::Tier1_1,
        ] {
            assert_eq!(
                tier.supports_inline_raytracing(),
                tier.supports_rayquery(),
                "inline and rayquery should match for {:?}",
                tier
            );
        }
    }
}

// ============================================================================
// D3D12RayTracingTier Tests - is_available()
// ============================================================================

mod ray_tracing_tier_is_available {
    use super::*;

    #[test]
    fn none_is_not_available() {
        assert!(!D3D12RayTracingTier::None.is_available());
    }

    #[test]
    fn tier_1_0_is_available() {
        assert!(D3D12RayTracingTier::Tier1_0.is_available());
    }

    #[test]
    fn tier_1_1_is_available() {
        assert!(D3D12RayTracingTier::Tier1_1.is_available());
    }
}

// ============================================================================
// D3D12RayTracingTier Tests - Trait Implementations
// ============================================================================

mod ray_tracing_tier_traits {
    use super::*;

    #[test]
    fn ordering() {
        assert!(D3D12RayTracingTier::None < D3D12RayTracingTier::Tier1_0);
        assert!(D3D12RayTracingTier::Tier1_0 < D3D12RayTracingTier::Tier1_1);
    }

    #[test]
    fn display_format() {
        assert_eq!(format!("{}", D3D12RayTracingTier::None), "None");
        assert_eq!(format!("{}", D3D12RayTracingTier::Tier1_0), "DXR 1.0");
        assert_eq!(format!("{}", D3D12RayTracingTier::Tier1_1), "DXR 1.1");
    }

    #[test]
    fn can_use_in_hashset() {
        let mut set: HashSet<D3D12RayTracingTier> = HashSet::new();
        set.insert(D3D12RayTracingTier::None);
        set.insert(D3D12RayTracingTier::Tier1_0);
        set.insert(D3D12RayTracingTier::Tier1_1);
        set.insert(D3D12RayTracingTier::None); // duplicate

        assert_eq!(set.len(), 3);
    }
}

// ============================================================================
// D3D12Features Tests - Default and Initialization
// ============================================================================

mod features_default {
    use super::*;

    #[test]
    fn default_feature_level() {
        let features = D3D12Features::default();
        assert_eq!(features.feature_level, D3D12FeatureLevel::FL_11_0);
    }

    #[test]
    fn default_shader_model() {
        let features = D3D12Features::default();
        assert_eq!(features.shader_model, D3D12ShaderModel::SM_5_1);
    }

    #[test]
    fn default_ray_tracing_tier() {
        let features = D3D12Features::default();
        assert_eq!(features.ray_tracing_tier, D3D12RayTracingTier::None);
    }

    #[test]
    fn default_mesh_shader_tier() {
        let features = D3D12Features::default();
        assert_eq!(features.mesh_shader_tier, 0);
    }

    #[test]
    fn default_vrs_tier() {
        let features = D3D12Features::default();
        assert_eq!(features.variable_rate_shading_tier, 0);
    }

    #[test]
    fn default_sampler_feedback_tier() {
        let features = D3D12Features::default();
        assert_eq!(features.sampler_feedback_tier, 0);
    }

    #[test]
    fn default_bindless() {
        let features = D3D12Features::default();
        assert!(!features.bindless_resources);
    }

    #[test]
    fn default_conservative_raster() {
        let features = D3D12Features::default();
        assert_eq!(features.conservative_rasterization_tier, 0);
    }

    #[test]
    fn default_tiled_resources() {
        let features = D3D12Features::default();
        assert_eq!(features.tiled_resources_tier, 0);
    }

    #[test]
    fn default_resource_binding() {
        let features = D3D12Features::default();
        assert_eq!(features.resource_binding_tier, 0);
    }

    #[test]
    fn default_root_signature() {
        let features = D3D12Features::default();
        assert_eq!(features.root_signature_version, 0);
    }

    #[test]
    fn default_wave_ops() {
        let features = D3D12Features::default();
        assert!(!features.wave_ops);
    }

    #[test]
    fn default_native_16bit() {
        let features = D3D12Features::default();
        assert!(!features.native_16bit_ops);
    }

    #[test]
    fn default_rt_pipeline() {
        let features = D3D12Features::default();
        assert!(!features.rt_pipeline);
    }

    #[test]
    fn default_rov() {
        let features = D3D12Features::default();
        assert_eq!(features.rasterizer_ordered_views_tier, 0);
    }
}

// ============================================================================
// D3D12Features Tests - from_features() Detection
// ============================================================================

mod features_from_features {
    use super::*;

    #[test]
    fn empty_features() {
        let wgpu_features = Features::empty();
        let dx12 = D3D12Features::from_features(wgpu_features);

        assert_eq!(dx12.feature_level, D3D12FeatureLevel::FL_11_0);
        assert_eq!(dx12.shader_model, D3D12ShaderModel::SM_5_1);
        assert_eq!(dx12.ray_tracing_tier, D3D12RayTracingTier::None);
        assert!(!dx12.bindless_resources);
        assert!(!dx12.wave_ops);
    }

    #[test]
    fn bindless_features() {
        let wgpu_features = Features::TEXTURE_BINDING_ARRAY
            | Features::BUFFER_BINDING_ARRAY
            | Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING;
        let dx12 = D3D12Features::from_features(wgpu_features);

        assert!(dx12.bindless_resources);
        assert_eq!(dx12.resource_binding_tier, 3);
    }

    #[test]
    fn subgroup_feature() {
        let wgpu_features = Features::SUBGROUP;
        let dx12 = D3D12Features::from_features(wgpu_features);

        assert!(dx12.wave_ops);
        assert_eq!(dx12.shader_model, D3D12ShaderModel::SM_6_0);
    }

    #[test]
    fn shader_f16_feature() {
        let wgpu_features = Features::SHADER_F16;
        let dx12 = D3D12Features::from_features(wgpu_features);

        assert!(dx12.native_16bit_ops);
    }

    #[test]
    fn rt_acceleration_structure_feature() {
        let wgpu_features = Features::RAY_TRACING_ACCELERATION_STRUCTURE;
        let dx12 = D3D12Features::from_features(wgpu_features);

        assert!(dx12.rt_pipeline);
        assert_eq!(dx12.ray_tracing_tier, D3D12RayTracingTier::Tier1_0);
        assert_eq!(dx12.feature_level, D3D12FeatureLevel::FL_12_1);
    }

    #[test]
    fn full_rt_features() {
        let wgpu_features = Features::RAY_TRACING_ACCELERATION_STRUCTURE | Features::RAY_QUERY;
        let dx12 = D3D12Features::from_features(wgpu_features);

        assert_eq!(dx12.ray_tracing_tier, D3D12RayTracingTier::Tier1_1);
        assert_eq!(dx12.feature_level, D3D12FeatureLevel::FL_12_2);
        assert_eq!(dx12.shader_model, D3D12ShaderModel::SM_6_5);
    }

    #[test]
    fn fl_12_1_sets_vrs_tier_1() {
        let wgpu_features = Features::RAY_TRACING_ACCELERATION_STRUCTURE;
        let dx12 = D3D12Features::from_features(wgpu_features);

        assert_eq!(dx12.variable_rate_shading_tier, 1);
    }

    #[test]
    fn fl_12_2_sets_vrs_tier_2() {
        let wgpu_features = Features::RAY_TRACING_ACCELERATION_STRUCTURE | Features::RAY_QUERY;
        let dx12 = D3D12Features::from_features(wgpu_features);

        assert_eq!(dx12.variable_rate_shading_tier, 2);
    }

    #[test]
    fn fl_12_2_sets_sampler_feedback_tier_1() {
        let wgpu_features = Features::RAY_TRACING_ACCELERATION_STRUCTURE | Features::RAY_QUERY;
        let dx12 = D3D12Features::from_features(wgpu_features);

        assert_eq!(dx12.sampler_feedback_tier, 1);
    }

    #[test]
    fn mesh_shader_tier_requires_fl_12_1_and_sm_6_5() {
        // FL 12.2 + SM 6.5 should give mesh_shader_tier = 1
        let wgpu_features = Features::RAY_TRACING_ACCELERATION_STRUCTURE | Features::RAY_QUERY;
        let dx12 = D3D12Features::from_features(wgpu_features);

        assert_eq!(dx12.mesh_shader_tier, 1);
    }

    #[test]
    fn conservative_raster_tier_by_feature_level() {
        // FL 12.1+ gets tier 3
        let features = Features::RAY_TRACING_ACCELERATION_STRUCTURE;
        let dx12 = D3D12Features::from_features(features);
        assert_eq!(dx12.conservative_rasterization_tier, 3);
    }

    #[test]
    fn tiled_resources_tier_progression() {
        // FL 11.1 -> tier 1
        let features = Features::TIMESTAMP_QUERY;
        let dx12 = D3D12Features::from_features(features);
        assert_eq!(dx12.tiled_resources_tier, 1);

        // FL 12.0 -> tier 2
        let features = Features::TEXTURE_BINDING_ARRAY
            | Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING;
        let dx12 = D3D12Features::from_features(features);
        assert_eq!(dx12.tiled_resources_tier, 2);

        // FL 12.1 -> tier 3
        let features = Features::RAY_TRACING_ACCELERATION_STRUCTURE;
        let dx12 = D3D12Features::from_features(features);
        assert_eq!(dx12.tiled_resources_tier, 3);

        // FL 12.2 -> tier 4
        let features = Features::RAY_TRACING_ACCELERATION_STRUCTURE | Features::RAY_QUERY;
        let dx12 = D3D12Features::from_features(features);
        assert_eq!(dx12.tiled_resources_tier, 4);
    }
}

// ============================================================================
// D3D12Features Tests - Feature Detection Methods
// ============================================================================

mod features_detection_methods {
    use super::*;

    #[test]
    fn supports_rt_false_for_none() {
        let mut features = D3D12Features::default();
        features.ray_tracing_tier = D3D12RayTracingTier::None;
        assert!(!features.supports_rt());
    }

    #[test]
    fn supports_rt_true_for_tier_1_0() {
        let mut features = D3D12Features::default();
        features.ray_tracing_tier = D3D12RayTracingTier::Tier1_0;
        assert!(features.supports_rt());
    }

    #[test]
    fn supports_rt_true_for_tier_1_1() {
        let mut features = D3D12Features::default();
        features.ray_tracing_tier = D3D12RayTracingTier::Tier1_1;
        assert!(features.supports_rt());
    }

    #[test]
    fn supports_mesh_shaders_false_for_tier_0() {
        let mut features = D3D12Features::default();
        features.mesh_shader_tier = 0;
        assert!(!features.supports_mesh_shaders());
    }

    #[test]
    fn supports_mesh_shaders_true_for_tier_1() {
        let mut features = D3D12Features::default();
        features.mesh_shader_tier = 1;
        assert!(features.supports_mesh_shaders());
    }

    #[test]
    fn supports_bindless_false_when_disabled() {
        let mut features = D3D12Features::default();
        features.bindless_resources = false;
        assert!(!features.supports_bindless());
    }

    #[test]
    fn supports_bindless_true_when_enabled() {
        let mut features = D3D12Features::default();
        features.bindless_resources = true;
        assert!(features.supports_bindless());
    }

    #[test]
    fn supports_vrs_false_for_tier_0() {
        let mut features = D3D12Features::default();
        features.variable_rate_shading_tier = 0;
        assert!(!features.supports_vrs());
    }

    #[test]
    fn supports_vrs_true_for_tier_1() {
        let mut features = D3D12Features::default();
        features.variable_rate_shading_tier = 1;
        assert!(features.supports_vrs());
    }

    #[test]
    fn supports_vrs_true_for_tier_2() {
        let mut features = D3D12Features::default();
        features.variable_rate_shading_tier = 2;
        assert!(features.supports_vrs());
    }

    #[test]
    fn supports_inline_rt_false_for_tier_1_0() {
        let mut features = D3D12Features::default();
        features.ray_tracing_tier = D3D12RayTracingTier::Tier1_0;
        assert!(!features.supports_inline_rt());
    }

    #[test]
    fn supports_inline_rt_true_for_tier_1_1() {
        let mut features = D3D12Features::default();
        features.ray_tracing_tier = D3D12RayTracingTier::Tier1_1;
        assert!(features.supports_inline_rt());
    }

    #[test]
    fn supports_sampler_feedback_false_for_tier_0() {
        let mut features = D3D12Features::default();
        features.sampler_feedback_tier = 0;
        assert!(!features.supports_sampler_feedback());
    }

    #[test]
    fn supports_sampler_feedback_true_for_tier_1() {
        let mut features = D3D12Features::default();
        features.sampler_feedback_tier = 1;
        assert!(features.supports_sampler_feedback());
    }

    #[test]
    fn supports_conservative_raster_false_for_tier_0() {
        let mut features = D3D12Features::default();
        features.conservative_rasterization_tier = 0;
        assert!(!features.supports_conservative_raster());
    }

    #[test]
    fn supports_conservative_raster_true_for_tier_1() {
        let mut features = D3D12Features::default();
        features.conservative_rasterization_tier = 1;
        assert!(features.supports_conservative_raster());
    }

    #[test]
    fn supports_gpu_driven_requires_both() {
        let mut features = D3D12Features::default();
        assert!(!features.supports_gpu_driven());

        features.bindless_resources = true;
        assert!(!features.supports_gpu_driven());

        features.wave_ops = true;
        assert!(features.supports_gpu_driven());

        features.bindless_resources = false;
        assert!(!features.supports_gpu_driven());
    }
}

// ============================================================================
// D3D12Features Tests - minimum_windows_version()
// ============================================================================

mod features_minimum_windows_version {
    use super::*;

    #[test]
    fn base_d3d12_requires_windows_10() {
        let features = D3D12Features::default();
        let (major, minor, build) = features.minimum_windows_version();
        assert_eq!(major, 10);
        assert_eq!(minor, 0);
        assert_eq!(build, 10240);
    }

    #[test]
    fn vrs_requires_build_18362() {
        let mut features = D3D12Features::default();
        features.variable_rate_shading_tier = 1;
        let (_, _, build) = features.minimum_windows_version();
        assert_eq!(build, 18362);
    }

    #[test]
    fn dxr_1_0_requires_build_17763() {
        let mut features = D3D12Features::default();
        features.ray_tracing_tier = D3D12RayTracingTier::Tier1_0;
        let (_, _, build) = features.minimum_windows_version();
        assert_eq!(build, 17763);
    }

    #[test]
    fn dxr_1_1_requires_build_19041() {
        let mut features = D3D12Features::default();
        features.ray_tracing_tier = D3D12RayTracingTier::Tier1_1;
        let (_, _, build) = features.minimum_windows_version();
        assert_eq!(build, 19041);
    }

    #[test]
    fn mesh_shaders_require_build_19041() {
        let mut features = D3D12Features::default();
        features.mesh_shader_tier = 1;
        let (_, _, build) = features.minimum_windows_version();
        assert_eq!(build, 19041);
    }

    #[test]
    fn dxr_1_1_takes_priority_over_mesh() {
        let mut features = D3D12Features::default();
        features.ray_tracing_tier = D3D12RayTracingTier::Tier1_1;
        features.mesh_shader_tier = 1;
        let (_, _, build) = features.minimum_windows_version();
        assert_eq!(build, 19041);
    }

    #[test]
    fn dxr_1_0_takes_priority_over_vrs() {
        let mut features = D3D12Features::default();
        features.ray_tracing_tier = D3D12RayTracingTier::Tier1_0;
        features.variable_rate_shading_tier = 1;
        let (_, _, build) = features.minimum_windows_version();
        assert_eq!(build, 17763);
    }
}

// ============================================================================
// D3D12Features Tests - summary()
// ============================================================================

mod features_summary {
    use super::*;

    #[test]
    fn summary_contains_feature_level() {
        let features = D3D12Features::default();
        assert!(features.summary().contains("FL 11.0"));
    }

    #[test]
    fn summary_contains_shader_model() {
        let features = D3D12Features::default();
        assert!(features.summary().contains("SM 5.1"));
    }

    #[test]
    fn summary_contains_dxr_when_available() {
        let mut features = D3D12Features::default();
        features.ray_tracing_tier = D3D12RayTracingTier::Tier1_1;
        assert!(features.summary().contains("DXR 1.1"));
    }

    #[test]
    fn summary_omits_dxr_when_none() {
        let features = D3D12Features::default();
        assert!(!features.summary().contains("DXR"));
    }

    #[test]
    fn summary_contains_mesh_when_available() {
        let mut features = D3D12Features::default();
        features.mesh_shader_tier = 1;
        assert!(features.summary().contains("Mesh"));
    }

    #[test]
    fn summary_contains_vrs_with_tier() {
        let mut features = D3D12Features::default();
        features.variable_rate_shading_tier = 2;
        assert!(features.summary().contains("VRS T2"));
    }

    #[test]
    fn summary_contains_bindless_when_available() {
        let mut features = D3D12Features::default();
        features.bindless_resources = true;
        assert!(features.summary().contains("Bindless"));
    }

    #[test]
    fn summary_contains_wave_when_available() {
        let mut features = D3D12Features::default();
        features.wave_ops = true;
        assert!(features.summary().contains("Wave"));
    }

    #[test]
    fn summary_contains_fp16_when_available() {
        let mut features = D3D12Features::default();
        features.native_16bit_ops = true;
        assert!(features.summary().contains("FP16"));
    }

    #[test]
    fn summary_full_features() {
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
}

// ============================================================================
// D3D12Features Tests - Tier Fields
// ============================================================================

mod features_tier_fields {
    use super::*;

    #[test]
    fn mesh_shader_tier_range() {
        let mut features = D3D12Features::default();
        features.mesh_shader_tier = 0;
        assert!(!features.supports_mesh_shaders());
        features.mesh_shader_tier = 1;
        assert!(features.supports_mesh_shaders());
    }

    #[test]
    fn vrs_tier_values() {
        // 0 = none, 1 = per-draw, 2 = per-draw + per-primitive
        let mut features = D3D12Features::default();

        features.variable_rate_shading_tier = 0;
        assert!(!features.supports_vrs());

        features.variable_rate_shading_tier = 1;
        assert!(features.supports_vrs());

        features.variable_rate_shading_tier = 2;
        assert!(features.supports_vrs());
    }

    #[test]
    fn conservative_raster_tier_values() {
        let mut features = D3D12Features::default();

        for tier in [0, 1, 2, 3] {
            features.conservative_rasterization_tier = tier;
            assert_eq!(
                features.supports_conservative_raster(),
                tier >= 1,
                "tier {} should {}support conservative raster",
                tier,
                if tier >= 1 { "" } else { "not " }
            );
        }
    }

    #[test]
    fn tiled_resources_tier_range() {
        let mut features = D3D12Features::default();

        for tier in 0..=4 {
            features.tiled_resources_tier = tier;
            // Just verifying the field can hold these values
            assert_eq!(features.tiled_resources_tier, tier);
        }
    }

    #[test]
    fn resource_binding_tier_range() {
        let mut features = D3D12Features::default();

        for tier in 1..=3 {
            features.resource_binding_tier = tier;
            assert_eq!(features.resource_binding_tier, tier);
        }
    }

    #[test]
    fn rov_tier_range() {
        let mut features = D3D12Features::default();

        for tier in 0..=3 {
            features.rasterizer_ordered_views_tier = tier;
            assert_eq!(features.rasterizer_ordered_views_tier, tier);
        }
    }
}

// ============================================================================
// Edge Case Tests
// ============================================================================

mod edge_cases {
    use super::*;

    #[test]
    fn minimum_feature_level_scenario() {
        let features = D3D12Features::default();
        assert_eq!(features.feature_level, D3D12FeatureLevel::FL_11_0);
        assert!(!features.supports_rt());
        assert!(!features.supports_mesh_shaders());
        assert!(!features.supports_vrs());
        assert!(!features.supports_bindless());
    }

    #[test]
    fn maximum_feature_level_scenario() {
        let wgpu_features = Features::RAY_TRACING_ACCELERATION_STRUCTURE
            | Features::RAY_QUERY
            | Features::TEXTURE_BINDING_ARRAY
            | Features::BUFFER_BINDING_ARRAY
            | Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING
            | Features::SUBGROUP
            | Features::SHADER_F16;

        let dx12 = D3D12Features::from_features(wgpu_features);

        assert_eq!(dx12.feature_level, D3D12FeatureLevel::FL_12_2);
        assert_eq!(dx12.shader_model, D3D12ShaderModel::SM_6_5);
        assert_eq!(dx12.ray_tracing_tier, D3D12RayTracingTier::Tier1_1);
        assert!(dx12.supports_rt());
        assert!(dx12.supports_inline_rt());
        assert!(dx12.supports_mesh_shaders());
        assert!(dx12.supports_vrs());
        assert!(dx12.supports_bindless());
        assert!(dx12.supports_gpu_driven());
        assert!(dx12.supports_sampler_feedback());
    }

    #[test]
    fn tier_boundary_mesh_shader_0_to_1() {
        let mut features = D3D12Features::default();

        features.mesh_shader_tier = 0;
        assert!(!features.supports_mesh_shaders());

        features.mesh_shader_tier = 1;
        assert!(features.supports_mesh_shaders());
    }

    #[test]
    fn tier_boundary_vrs_0_to_1() {
        let mut features = D3D12Features::default();

        features.variable_rate_shading_tier = 0;
        assert!(!features.supports_vrs());

        features.variable_rate_shading_tier = 1;
        assert!(features.supports_vrs());
    }

    #[test]
    fn tier_boundary_sampler_feedback_0_to_1() {
        let mut features = D3D12Features::default();

        features.sampler_feedback_tier = 0;
        assert!(!features.supports_sampler_feedback());

        features.sampler_feedback_tier = 1;
        assert!(features.supports_sampler_feedback());
    }

    #[test]
    fn feature_level_boundary_fl_12_0_to_fl_12_1() {
        // FL 12.0 does not support RT
        assert!(!D3D12FeatureLevel::FL_12_0.supports_ray_tracing());
        // FL 12.1 does support RT
        assert!(D3D12FeatureLevel::FL_12_1.supports_ray_tracing());
    }

    #[test]
    fn feature_level_boundary_fl_12_1_to_fl_12_2() {
        // FL 12.1 does not support mesh shaders
        assert!(!D3D12FeatureLevel::FL_12_1.supports_mesh_shaders());
        // FL 12.2 does support mesh shaders
        assert!(D3D12FeatureLevel::FL_12_2.supports_mesh_shaders());
    }

    #[test]
    fn shader_model_boundary_sm_6_2_to_sm_6_3() {
        // SM 6.2 does not support RT intrinsics
        assert!(!D3D12ShaderModel::SM_6_2.supports_raytracing_intrinsics());
        // SM 6.3 does support RT intrinsics
        assert!(D3D12ShaderModel::SM_6_3.supports_raytracing_intrinsics());
    }

    #[test]
    fn shader_model_boundary_sm_6_4_to_sm_6_5() {
        // SM 6.4 does not support mesh shaders
        assert!(!D3D12ShaderModel::SM_6_4.supports_mesh_shaders());
        // SM 6.5 does support mesh shaders
        assert!(D3D12ShaderModel::SM_6_5.supports_mesh_shaders());
    }

    #[test]
    fn shader_model_boundary_sm_6_5_to_sm_6_6() {
        // SM 6.5 does not support derivatives
        assert!(!D3D12ShaderModel::SM_6_5.supports_derivatives());
        // SM 6.6 does support derivatives
        assert!(D3D12ShaderModel::SM_6_6.supports_derivatives());
    }

    #[test]
    fn rt_tier_boundary_none_to_tier_1_0() {
        assert!(!D3D12RayTracingTier::None.is_available());
        assert!(D3D12RayTracingTier::Tier1_0.is_available());
    }

    #[test]
    fn rt_tier_boundary_tier_1_0_to_tier_1_1() {
        assert!(!D3D12RayTracingTier::Tier1_0.supports_inline_raytracing());
        assert!(D3D12RayTracingTier::Tier1_1.supports_inline_raytracing());
    }

    #[test]
    fn unknown_feature_combinations() {
        // Unusual combination: subgroup without bindless
        let features = Features::SUBGROUP;
        let dx12 = D3D12Features::from_features(features);
        assert!(dx12.wave_ops);
        assert!(!dx12.bindless_resources);
        assert_eq!(dx12.shader_model, D3D12ShaderModel::SM_6_0);
    }

    #[test]
    fn partial_bindless_features() {
        // Only TEXTURE_BINDING_ARRAY without non-uniform indexing
        let features = Features::TEXTURE_BINDING_ARRAY | Features::BUFFER_BINDING_ARRAY;
        let dx12 = D3D12Features::from_features(features);
        assert!(!dx12.bindless_resources);
    }
}

// ============================================================================
// Integration Tests
// 