//! Whitebox tests for CapabilityManager.
//!
//! These tests verify the internal structure and behavior of CapabilityManager,
//! including all feature query methods, render path selection, texture compression
//! selection, and capability reporting.
//!
//! Task: T-WGPU-P1.5.2 - CapabilityManager
//!
//! WHITEBOX coverage plan:
//!   - Path A: RenderPath enum - all variants, names, descriptions, Display
//!   - Path B: TextureCompression enum - all variants, names, descriptions,
//!             is_hardware_accelerated, Display
//!   - Path C: CapabilityManager construction - from_features_and_limits for each tier
//!   - Path D: supports_ray_tracing() - true only with RT feature
//!   - Path E: supports_bindless() - true only with TEXTURE_BINDING_ARRAY feature
//!   - Path F: supports_gpu_culling() - true only with MULTI_DRAW_INDIRECT_COUNT feature
//!   - Path G: supports_timestamp_queries() - true only with TIMESTAMP_QUERY feature
//!   - Path H: select_render_path() - maps tier to correct RenderPath
//!   - Path I: select_texture_compression() - priority BC > ASTC > ETC2 > None
//!   - Path J: max_bindless_textures() - 0 if no bindless, limit otherwise
//!   - Path K: report() - generates CapabilityReport with correct values
//!   - Path L: Basic accessors (tier, features, limits, adapter_name)
//!   - Path M: Additional methods (push_constants, storage_binding_array, etc.)
//!   - Path N: Edge cases (boundary limits, empty features, combined features)
//!   - Path O: Trait implementations (Debug, Clone)
//!   - Path P: meets_tier() boundary conditions

use renderer_backend::device::{
    CapabilityManager, CapabilityReport, CapabilityTier, RenderPath, TextureCompression,
};
use std::collections::{HashMap, HashSet};
use wgpu::{Features, Limits};

// ============================================================================
// Constants (matching the source implementation)
// ============================================================================

/// Minimum 2D texture dimension required for Standard tier (8K).
const STANDARD_TIER_MIN_TEXTURE_2D: u32 = 8192;

/// Minimum compute workgroup invocations for Advanced tier.
const ADVANCED_TIER_MIN_WORKGROUP_INVOCATIONS: u32 = 1024;

/// Minimum storage buffer binding size for Standard tier (128 MB).
const STANDARD_TIER_MIN_STORAGE_BUFFER_SIZE: u32 = 128 * 1024 * 1024;

// ============================================================================
// Helper Functions
// ============================================================================

/// Create minimal limits that result in Minimal tier.
fn minimal_limits() -> Limits {
    Limits::downlevel_webgl2_defaults()
}

/// Create limits that meet Standard tier requirements exactly.
fn standard_limits() -> Limits {
    let mut limits = Limits::default();
    limits.max_texture_dimension_2d = STANDARD_TIER_MIN_TEXTURE_2D;
    limits.max_storage_buffer_binding_size = STANDARD_TIER_MIN_STORAGE_BUFFER_SIZE;
    limits
}

/// Create limits that meet Advanced tier requirements.
fn advanced_limits() -> Limits {
    let mut limits = Limits::default();
    limits.max_texture_dimension_2d = 16384;
    limits.max_compute_invocations_per_workgroup = ADVANCED_TIER_MIN_WORKGROUP_INVOCATIONS;
    limits.max_storage_buffer_binding_size = 256 * 1024 * 1024;
    limits.max_sampled_textures_per_shader_stage = 16384;
    limits
}

/// Create features for Advanced tier.
fn advanced_features() -> Features {
    Features::TEXTURE_BINDING_ARRAY | Features::MULTI_DRAW_INDIRECT_COUNT
}

/// Create features for Full tier.
fn full_features() -> Features {
    Features::RAY_TRACING_ACCELERATION_STRUCTURE
        | Features::TEXTURE_BINDING_ARRAY
        | Features::MULTI_DRAW_INDIRECT_COUNT
        | Features::TIMESTAMP_QUERY
}

// ============================================================================
// 1. RenderPath Enum Tests
// ============================================================================

mod render_path_enum {
    use super::*;

    #[test]
    fn all_four_variants_exist() {
        let ray_traced = RenderPath::RayTraced;
        let gpu_driven = RenderPath::GPUDriven;
        let traditional = RenderPath::Traditional;
        let fallback = RenderPath::Fallback;

        // Each variant is self-equal
        assert_eq!(ray_traced, RenderPath::RayTraced);
        assert_eq!(gpu_driven, RenderPath::GPUDriven);
        assert_eq!(traditional, RenderPath::Traditional);
        assert_eq!(fallback, RenderPath::Fallback);
    }

    #[test]
    fn variants_are_mutually_distinct() {
        let variants = [
            RenderPath::RayTraced,
            RenderPath::GPUDriven,
            RenderPath::Traditional,
            RenderPath::Fallback,
        ];

        for (i, a) in variants.iter().enumerate() {
            for (j, b) in variants.iter().enumerate() {
                if i == j {
                    assert_eq!(a, b, "Same variant should be equal");
                } else {
                    assert_ne!(a, b, "Different variants should not be equal");
                }
            }
        }
    }

    #[test]
    fn name_returns_correct_strings() {
        assert_eq!(RenderPath::RayTraced.name(), "RayTraced");
        assert_eq!(RenderPath::GPUDriven.name(), "GPUDriven");
        assert_eq!(RenderPath::Traditional.name(), "Traditional");
        assert_eq!(RenderPath::Fallback.name(), "Fallback");
    }

    #[test]
    fn names_are_unique() {
        let names: HashSet<&str> = [
            RenderPath::RayTraced.name(),
            RenderPath::GPUDriven.name(),
            RenderPath::Traditional.name(),
            RenderPath::Fallback.name(),
        ]
        .into_iter()
        .collect();

        assert_eq!(names.len(), 4, "All render path names should be unique");
    }

    #[test]
    fn description_is_non_empty() {
        assert!(!RenderPath::RayTraced.description().is_empty());
        assert!(!RenderPath::GPUDriven.description().is_empty());
        assert!(!RenderPath::Traditional.description().is_empty());
        assert!(!RenderPath::Fallback.description().is_empty());
    }

    #[test]
    fn description_is_meaningful() {
        // RayTraced description should mention ray tracing
        assert!(
            RenderPath::RayTraced
                .description()
                .to_lowercase()
                .contains("ray")
        );

        // GPUDriven description should mention gpu or bindless
        let gpu_desc = RenderPath::GPUDriven.description().to_lowercase();
        assert!(gpu_desc.contains("gpu") || gpu_desc.contains("bindless"));

        // Traditional description should mention forward or deferred
        let trad_desc = RenderPath::Traditional.description().to_lowercase();
        assert!(trad_desc.contains("forward") || trad_desc.contains("deferred") || trad_desc.contains("traditional"));

        // Fallback description should mention fallback or minimal
        let fallback_desc = RenderPath::Fallback.description().to_lowercase();
        assert!(fallback_desc.contains("fallback") || fallback_desc.contains("minimal"));
    }

    #[test]
    fn display_matches_name() {
        assert_eq!(format!("{}", RenderPath::RayTraced), "RayTraced");
        assert_eq!(format!("{}", RenderPath::GPUDriven), "GPUDriven");
        assert_eq!(format!("{}", RenderPath::Traditional), "Traditional");
        assert_eq!(format!("{}", RenderPath::Fallback), "Fallback");
    }

    #[test]
    fn debug_is_implemented() {
        let debug_str = format!("{:?}", RenderPath::RayTraced);
        assert!(debug_str.contains("RayTraced"));

        assert!(!format!("{:?}", RenderPath::GPUDriven).is_empty());
        assert!(!format!("{:?}", RenderPath::Traditional).is_empty());
        assert!(!format!("{:?}", RenderPath::Fallback).is_empty());
    }

    #[test]
    fn clone_works() {
        let original = RenderPath::GPUDriven;
        let cloned = original.clone();
        assert_eq!(original, cloned);
    }

    #[test]
    fn copy_works() {
        let path = RenderPath::Traditional;
        let copied = path;
        assert_eq!(path, copied);
        // Original still usable (Copy trait)
        assert_eq!(path.name(), "Traditional");
    }

    #[test]
    fn hash_is_usable() {
        let mut set: HashSet<RenderPath> = HashSet::new();
        set.insert(RenderPath::RayTraced);
        set.insert(RenderPath::GPUDriven);
        set.insert(RenderPath::Traditional);
        set.insert(RenderPath::Fallback);

        assert_eq!(set.len(), 4);
        assert!(set.contains(&RenderPath::RayTraced));
        assert!(set.contains(&RenderPath::Fallback));
    }

    #[test]
    fn hash_is_usable_in_hashmap() {
        let mut map: HashMap<RenderPath, &str> = HashMap::new();
        map.insert(RenderPath::RayTraced, "highest quality");
        map.insert(RenderPath::GPUDriven, "high performance");
        map.insert(RenderPath::Traditional, "compatible");
        map.insert(RenderPath::Fallback, "lowest");

        assert_eq!(map.get(&RenderPath::RayTraced), Some(&"highest quality"));
        assert_eq!(map.get(&RenderPath::Fallback), Some(&"lowest"));
    }
}

// ============================================================================
// 2. TextureCompression Enum Tests
// ============================================================================

mod texture_compression_enum {
    use super::*;

    #[test]
    fn all_four_variants_exist() {
        let bc = TextureCompression::BC;
        let astc = TextureCompression::ASTC;
        let etc2 = TextureCompression::ETC2;
        let none = TextureCompression::None;

        assert_eq!(bc, TextureCompression::BC);
        assert_eq!(astc, TextureCompression::ASTC);
        assert_eq!(etc2, TextureCompression::ETC2);
        assert_eq!(none, TextureCompression::None);
    }

    #[test]
    fn variants_are_mutually_distinct() {
        let variants = [
            TextureCompression::BC,
            TextureCompression::ASTC,
            TextureCompression::ETC2,
            TextureCompression::None,
        ];

        for (i, a) in variants.iter().enumerate() {
            for (j, b) in variants.iter().enumerate() {
                if i == j {
                    assert_eq!(a, b, "Same variant should be equal");
                } else {
                    assert_ne!(a, b, "Different variants should not be equal");
                }
            }
        }
    }

    #[test]
    fn name_returns_correct_strings() {
        assert_eq!(TextureCompression::BC.name(), "BC");
        assert_eq!(TextureCompression::ASTC.name(), "ASTC");
        assert_eq!(TextureCompression::ETC2.name(), "ETC2");
        assert_eq!(TextureCompression::None.name(), "None");
    }

    #[test]
    fn names_are_unique() {
        let names: HashSet<&str> = [
            TextureCompression::BC.name(),
            TextureCompression::ASTC.name(),
            TextureCompression::ETC2.name(),
            TextureCompression::None.name(),
        ]
        .into_iter()
        .collect();

        assert_eq!(names.len(), 4, "All compression names should be unique");
    }

    #[test]
    fn description_is_non_empty() {
        assert!(!TextureCompression::BC.description().is_empty());
        assert!(!TextureCompression::ASTC.description().is_empty());
        assert!(!TextureCompression::ETC2.description().is_empty());
        assert!(!TextureCompression::None.description().is_empty());
    }

    #[test]
    fn description_is_meaningful() {
        // BC description should mention block compression or desktop
        let bc_desc = TextureCompression::BC.description().to_lowercase();
        assert!(bc_desc.contains("block") || bc_desc.contains("desktop") || bc_desc.contains("bc"));

        // ASTC description should mention adaptive or mobile
        let astc_desc = TextureCompression::ASTC.description().to_lowercase();
        assert!(astc_desc.contains("adaptive") || astc_desc.contains("mobile") || astc_desc.contains("scalable"));

        // ETC2 description should mention ericsson or mobile
        let etc2_desc = TextureCompression::ETC2.description().to_lowercase();
        assert!(etc2_desc.contains("ericsson") || etc2_desc.contains("mobile") || etc2_desc.contains("etc"));

        // None description should mention no compression or compatibility
        let none_desc = TextureCompression::None.description().to_lowercase();
        assert!(none_desc.contains("no") || none_desc.contains("none") || none_desc.contains("compat"));
    }

    #[test]
    fn is_hardware_accelerated_bc() {
        assert!(TextureCompression::BC.is_hardware_accelerated());
    }

    #[test]
    fn is_hardware_accelerated_astc() {
        assert!(TextureCompression::ASTC.is_hardware_accelerated());
    }

    #[test]
    fn is_hardware_accelerated_etc2() {
        assert!(TextureCompression::ETC2.is_hardware_accelerated());
    }

    #[test]
    fn is_hardware_accelerated_none_is_false() {
        assert!(!TextureCompression::None.is_hardware_accelerated());
    }

    #[test]
    fn display_matches_name() {
        assert_eq!(format!("{}", TextureCompression::BC), "BC");
        assert_eq!(format!("{}", TextureCompression::ASTC), "ASTC");
        assert_eq!(format!("{}", TextureCompression::ETC2), "ETC2");
        assert_eq!(format!("{}", TextureCompression::None), "None");
    }

    #[test]
    fn debug_is_implemented() {
        assert!(!format!("{:?}", TextureCompression::BC).is_empty());
        assert!(!format!("{:?}", TextureCompression::ASTC).is_empty());
        assert!(!format!("{:?}", TextureCompression::ETC2).is_empty());
        assert!(!format!("{:?}", TextureCompression::None).is_empty());
    }

    #[test]
    fn clone_works() {
        let original = TextureCompression::ASTC;
        let cloned = original.clone();
        assert_eq!(original, cloned);
    }

    #[test]
    fn copy_works() {
        let compression = TextureCompression::ETC2;
        let copied = compression;
        assert_eq!(compression, copied);
        assert_eq!(compression.name(), "ETC2");
    }

    #[test]
    fn hash_is_usable() {
        let mut set: HashSet<TextureCompression> = HashSet::new();
        set.insert(TextureCompression::BC);
        set.insert(TextureCompression::ASTC);
        set.insert(TextureCompression::ETC2);
        set.insert(TextureCompression::None);

        assert_eq!(set.len(), 4);
        assert!(set.contains(&TextureCompression::BC));
        assert!(set.contains(&TextureCompression::None));
    }
}

// ============================================================================
// 3. CapabilityManager Construction Tests
// ============================================================================

mod capability_manager_construction {
    use super::*;

    #[test]
    fn from_features_and_limits_minimal_tier() {
        let features = Features::empty();
        let limits = minimal_limits();

        let manager = CapabilityManager::from_features_and_limits(
            features,
            limits,
            "Test Minimal GPU",
        );

        assert_eq!(manager.tier(), CapabilityTier::Minimal);
        assert_eq!(manager.adapter_name(), "Test Minimal GPU");
    }

    #[test]
    fn from_features_and_limits_standard_tier() {
        let features = Features::empty();
        let limits = standard_limits();

        let manager = CapabilityManager::from_features_and_limits(
            features,
            limits,
            "Test Standard GPU",
        );

        assert_eq!(manager.tier(), CapabilityTier::Standard);
    }

    #[test]
    fn from_features_and_limits_advanced_tier() {
        let features = advanced_features();
        let limits = advanced_limits();

        let manager = CapabilityManager::from_features_and_limits(
            features,
            limits,
            "Test Advanced GPU",
        );

        assert_eq!(manager.tier(), CapabilityTier::Advanced);
    }

    #[test]
    fn from_features_and_limits_full_tier() {
        let features = full_features();
        let mut limits = advanced_limits();
        limits.max_compute_invocations_per_workgroup = 1024;

        let manager = CapabilityManager::from_features_and_limits(
            features,
            limits,
            "Test Full GPU",
        );

        assert_eq!(manager.tier(), CapabilityTier::Full);
    }

    #[test]
    fn adapter_name_accepts_string() {
        let manager = CapabilityManager::from_features_and_limits(
            Features::empty(),
            Limits::default(),
            String::from("Dynamic Name"),
        );

        assert_eq!(manager.adapter_name(), "Dynamic Name");
    }

    #[test]
    fn adapter_name_accepts_str() {
        let manager = CapabilityManager::from_features_and_limits(
            Features::empty(),
            Limits::default(),
            "Static Name",
        );

        assert_eq!(manager.adapter_name(), "Static Name");
    }

    #[test]
    fn empty_adapter_name() {
        let manager = CapabilityManager::from_features_and_limits(
            Features::empty(),
            Limits::default(),
            "",
        );

        assert_eq!(manager.adapter_name(), "");
    }
}

// ============================================================================
// 4. supports_ray_tracing() Tests
// ============================================================================

mod supports_ray_tracing {
    use super::*;

    #[test]
    fn true_with_rt_feature() {
        let features = Features::RAY_TRACING_ACCELERATION_STRUCTURE;
        let limits = Limits::default();
        let manager = CapabilityManager::from_features_and_limits(features, limits, "RT GPU");

        assert!(manager.supports_ray_tracing());
    }

    #[test]
    fn true_with_rt_and_other_features() {
        let features = Features::RAY_TRACING_ACCELERATION_STRUCTURE
            | Features::TEXTURE_BINDING_ARRAY
            | Features::MULTI_DRAW_INDIRECT_COUNT;
        let limits = Limits::default();
        let manager = CapabilityManager::from_features_and_limits(features, limits, "Full GPU");

        assert!(manager.supports_ray_tracing());
    }

    #[test]
    fn false_without_rt_feature_minimal() {
        let features = Features::empty();
        let limits = minimal_limits();
        let manager = CapabilityManager::from_features_and_limits(features, limits, "Minimal GPU");

        assert!(!manager.supports_ray_tracing());
    }

    #[test]
    fn false_without_rt_feature_standard() {
        let features = Features::empty();
        let limits = standard_limits();
        let manager = CapabilityManager::from_features_and_limits(features, limits, "Standard GPU");

        assert!(!manager.supports_ray_tracing());
    }

    #[test]
    fn false_without_rt_feature_advanced() {
        let features = advanced_features();
        let limits = advanced_limits();
        let manager = CapabilityManager::from_features_and_limits(features, limits, "Advanced GPU");

        assert!(!manager.supports_ray_tracing());
    }

    #[test]
    fn false_with_all_other_features_but_not_rt() {
        let features = Features::TEXTURE_BINDING_ARRAY
            | Features::MULTI_DRAW_INDIRECT_COUNT
            | Features::TIMESTAMP_QUERY
            | Features::PUSH_CONSTANTS;
        let limits = advanced_limits();
        let manager = CapabilityManager::from_features_and_limits(features, limits, "No RT GPU");

        assert!(!manager.supports_ray_tracing());
    }
}

// ============================================================================
// 5. supports_bindless() Tests
// ============================================================================

mod supports_bindless {
    use super::*;

    #[test]
    fn true_with_texture_binding_array() {
        let features = Features::TEXTURE_BINDING_ARRAY;
        let limits = Limits::default();
        let manager = CapabilityManager::from_features_and_limits(features, limits, "Bindless GPU");

        assert!(manager.supports_bindless());
    }

    #[test]
    fn true_with_advanced_features() {
        let features = advanced_features();
        let limits = advanced_limits();
        let manager = CapabilityManager::from_features_and_limits(features, limits, "Advanced GPU");

        assert!(manager.supports_bindless());
    }

    #[test]
    fn true_with_full_features() {
        let features = full_features();
        let limits = advanced_limits();
        let manager = CapabilityManager::from_features_and_limits(features, limits, "Full GPU");

        assert!(manager.supports_bindless());
    }

    #[test]
    fn false_without_texture_binding_array() {
        let features = Features::empty();
        let limits = Limits::default();
        let manager = CapabilityManager::from_features_and_limits(features, limits, "No Bindless GPU");

        assert!(!manager.supports_bindless());
    }

    #[test]
    fn false_with_multi_draw_but_no_bindless() {
        let features = Features::MULTI_DRAW_INDIRECT_COUNT;
        let limits = Limits::default();
        let manager = CapabilityManager::from_features_and_limits(features, limits, "Multi-Draw GPU");

        assert!(!manager.supports_bindless());
    }

    #[test]
    fn false_with_rt_but_no_bindless() {
        let features = Features::RAY_TRACING_ACCELERATION_STRUCTURE;
        let limits = Limits::default();
        let manager = CapabilityManager::from_features_and_limits(features, limits, "RT Only GPU");

        assert!(!manager.supports_bindless());
    }
}

// ============================================================================
// 6. supports_gpu_culling() Tests
// ============================================================================

mod supports_gpu_culling {
    use super::*;

    #[test]
    fn true_with_multi_draw_indirect_count() {
        let features = Features::MULTI_DRAW_INDIRECT_COUNT;
        let limits = Limits::default();
        let manager = CapabilityManager::from_features_and_limits(features, limits, "GPU Culling GPU");

        assert!(manager.supports_gpu_culling());
    }

    #[test]
    fn true_with_advanced_features() {
        let features = advanced_features();
        let limits = advanced_limits();
        let manager = CapabilityManager::from_features_and_limits(features, limits, "Advanced GPU");

        assert!(manager.supports_gpu_culling());
    }

    #[test]
    fn true_with_full_features() {
        let features = full_features();
        let limits = advanced_limits();
        let manager = CapabilityManager::from_features_and_limits(features, limits, "Full GPU");

        assert!(manager.supports_gpu_culling());
    }

    #[test]
    fn false_without_multi_draw() {
        let features = Features::empty();
        let limits = Limits::default();
        let manager = CapabilityManager::from_features_and_limits(features, limits, "Basic GPU");

        assert!(!manager.supports_gpu_culling());
    }

    #[test]
    fn false_with_bindless_but_no_multi_draw() {
        let features = Features::TEXTURE_BINDING_ARRAY;
        let limits = Limits::default();
        let manager = CapabilityManager::from_features_and_limits(features, limits, "Bindless Only GPU");

        assert!(!manager.supports_gpu_culling());
    }

    #[test]
    fn false_with_rt_but_no_multi_draw() {
        // RT only, no multi-draw indirect count
        let features = Features::RAY_TRACING_ACCELERATION_STRUCTURE;
        let limits = Limits::default();
        let manager = CapabilityManager::from_features_and_limits(features, limits, "RT Only GPU");

        assert!(!manager.supports_gpu_culling());
    }
}

// ============================================================================
// 7. supports_timestamp_queries() Tests
// ============================================================================

mod supports_timestamp_queries {
    use super::*;

    #[test]
    fn true_with_timestamp_query_feature() {
        let features = Features::TIMESTAMP_QUERY;
        let limits = Limits::default();
        let manager = CapabilityManager::from_features_and_limits(features, limits, "Timestamp GPU");

        assert!(manager.supports_timestamp_queries());
    }

    #[test]
    fn true_with_full_features() {
        let features = full_features();
        let limits = advanced_limits();
        let manager = CapabilityManager::from_features_and_limits(features, limits, "Full GPU");

        assert!(manager.supports_timestamp_queries());
    }

    #[test]
    fn false_without_timestamp_query() {
        let features = Features::empty();
        let limits = Limits::default();
        let manager = CapabilityManager::from_features_and_limits(features, limits, "No Timestamp GPU");

        assert!(!manager.supports_timestamp_queries());
    }

    #[test]
    fn false_with_advanced_features_no_timestamp() {
        let features = advanced_features(); // Does not include TIMESTAMP_QUERY
        let limits = advanced_limits();
        let manager = CapabilityManager::from_features_and_limits(features, limits, "Advanced GPU");

        assert!(!manager.supports_timestamp_queries());
    }

    #[test]
    fn false_with_rt_but_no_timestamp() {
        let features = Features::RAY_TRACING_ACCELERATION_STRUCTURE;
        let limits = Limits::default();
        let manager = CapabilityManager::from_features_and_limits(features, limits, "RT GPU");

        assert!(!manager.supports_timestamp_queries());
    }

    #[test]
    fn true_with_timestamp_and_other_features() {
        let features = Features::TIMESTAMP_QUERY
            | Features::TEXTURE_BINDING_ARRAY
            | Features::PUSH_CONSTANTS;
        let limits = Limits::default();
        let manager = CapabilityManager::from_features_and_limits(features, limits, "Mixed Features GPU");

        assert!(manager.supports_timestamp_queries());
    }
}

// ============================================================================
// 8. select_render_path() Tests
// ============================================================================

mod select_render_path {
    use super::*;

    #[test]
    fn full_tier_selects_ray_traced() {
        let features = Features::RAY_TRACING_ACCELERATION_STRUCTURE;
        let limits = Limits::default();
        let manager = CapabilityManager::from_features_and_limits(features, limits, "Full GPU");

        assert_eq!(manager.select_render_path(), RenderPath::RayTraced);
    }

    #[test]
    fn advanced_tier_selects_gpu_driven() {
        let features = advanced_features();
        let limits = advanced_limits();
        let manager = CapabilityManager::from_features_and_limits(features, limits, "Advanced GPU");

        assert_eq!(manager.select_render_path(), RenderPath::GPUDriven);
    }

    #[test]
    fn standard_tier_selects_traditional() {
        let features = Features::empty();
        let limits = standard_limits();
        let manager = CapabilityManager::from_features_and_limits(features, limits, "Standard GPU");

        assert_eq!(manager.select_render_path(), RenderPath::Traditional);
    }

    #[test]
    fn minimal_tier_selects_fallback() {
        let features = Features::empty();
        let limits = minimal_limits();
        let manager = CapabilityManager::from_features_and_limits(features, limits, "Minimal GPU");

        assert_eq!(manager.select_render_path(), RenderPath::Fallback);
    }

    #[test]
    fn full_tier_with_all_features_still_ray_traced() {
        let features = full_features();
        let limits = advanced_limits();
        let manager = CapabilityManager::from_features_and_limits(features, limits, "Full GPU");

        assert_eq!(manager.select_render_path(), RenderPath::RayTraced);
    }

    #[test]
    fn render_path_maps_to_tier_correctly() {
        // Verify the mapping is bijective (one-to-one)
        let test_cases = [
            (CapabilityTier::Full, RenderPath::RayTraced),
            (CapabilityTier::Advanced, RenderPath::GPUDriven),
            (CapabilityTier::Standard, RenderPath::Traditional),
            (CapabilityTier::Minimal, RenderPath::Fallback),
        ];

        for (expected_tier, expected_path) in test_cases {
            let (features, limits) = match expected_tier {
                CapabilityTier::Full => (
                    Features::RAY_TRACING_ACCELERATION_STRUCTURE,
                    Limits::default(),
                ),
                CapabilityTier::Advanced => (advanced_features(), advanced_limits()),
                CapabilityTier::Standard => (Features::empty(), standard_limits()),
                CapabilityTier::Minimal => (Features::empty(), minimal_limits()),
            };

            let manager = CapabilityManager::from_features_and_limits(features, limits, "Test GPU");
            assert_eq!(manager.tier(), expected_tier);
            assert_eq!(manager.select_render_path(), expected_path);
        }
    }
}

// ============================================================================
// 9. select_texture_compression() Tests
// ============================================================================

mod select_texture_compression {
    use super::*;

    #[test]
    fn selects_bc_when_available() {
        let features = Features::TEXTURE_COMPRESSION_BC;
        let limits = Limits::default();
        let manager = CapabilityManager::from_features_and_limits(features, limits, "Desktop GPU");

        assert_eq!(manager.select_texture_compression(), TextureCompression::BC);
    }

    #[test]
    fn selects_astc_when_no_bc() {
        let features = Features::TEXTURE_COMPRESSION_ASTC;
        let limits = Limits::default();
        let manager = CapabilityManager::from_features_and_limits(features, limits, "Mobile GPU");

        assert_eq!(manager.select_texture_compression(), TextureCompression::ASTC);
    }

    #[test]
    fn selects_etc2_when_no_bc_or_astc() {
        let features = Features::TEXTURE_COMPRESSION_ETC2;
        let limits = Limits::default();
        let manager = CapabilityManager::from_features_and_limits(features, limits, "ETC2 GPU");

        assert_eq!(manager.select_texture_compression(), TextureCompression::ETC2);
    }

    #[test]
    fn selects_none_when_no_compression() {
        let features = Features::empty();
        let limits = Limits::default();
        let manager = CapabilityManager::from_features_and_limits(features, limits, "Basic GPU");

        assert_eq!(manager.select_texture_compression(), TextureCompression::None);
    }

    #[test]
    fn bc_preferred_over_astc() {
        let features = Features::TEXTURE_COMPRESSION_BC | Features::TEXTURE_COMPRESSION_ASTC;
        let limits = Limits::default();
        let manager = CapabilityManager::from_features_and_limits(features, limits, "Multi GPU");

        assert_eq!(manager.select_texture_compression(), TextureCompression::BC);
    }

    #[test]
    fn bc_preferred_over_etc2() {
        let features = Features::TEXTURE_COMPRESSION_BC | Features::TEXTURE_COMPRESSION_ETC2;
        let limits = Limits::default();
        let manager = CapabilityManager::from_features_and_limits(features, limits, "Multi GPU");

        assert_eq!(manager.select_texture_compression(), TextureCompression::BC);
    }

    #[test]
    fn astc_preferred_over_etc2() {
        let features = Features::TEXTURE_COMPRESSION_ASTC | Features::TEXTURE_COMPRESSION_ETC2;
        let limits = Limits::default();
        let manager = CapabilityManager::from_features_and_limits(features, limits, "Multi GPU");

        assert_eq!(manager.select_texture_compression(), TextureCompression::ASTC);
    }

    #[test]
    fn bc_preferred_over_all() {
        let features = Features::TEXTURE_COMPRESSION_BC
            | Features::TEXTURE_COMPRESSION_ASTC
            | Features::TEXTURE_COMPRESSION_ETC2;
        let limits = Limits::default();
        let manager = CapabilityManager::from_features_and_limits(features, limits, "Full Compression GPU");

        assert_eq!(manager.select_texture_compression(), TextureCompression::BC);
    }

    #[test]
    fn compression_selection_independent_of_tier() {
        // Minimal tier with BC compression
        let features = Features::TEXTURE_COMPRESSION_BC;
        let limits = minimal_limits();
        let manager = CapabilityManager::from_features_and_limits(features, limits, "Minimal with BC");

        assert_eq!(manager.tier(), CapabilityTier::Minimal);
        assert_eq!(manager.select_texture_compression(), TextureCompression::BC);
    }
}

// ============================================================================
// 10. max_bindless_textures() Tests
// ============================================================================

mod max_bindless_textures {
    use super::*;

    #[test]
    fn returns_limit_when_bindless_supported() {
        let features = Features::TEXTURE_BINDING_ARRAY;
        let mut limits = Limits::default();
        limits.max_sampled_textures_per_shader_stage = 16384;
        let manager = CapabilityManager::from_features_and_limits(features, limits, "Bindless GPU");

        assert_eq!(manager.max_bindless_textures(), 16384);
    }

    #[test]
    fn returns_zero_when_bindless_not_supported() {
        let features = Features::empty();
        let mut limits = Limits::default();
        limits.max_sampled_textures_per_shader_stage = 16384;
        let manager = CapabilityManager::from_features_and_limits(features, limits, "No Bindless GPU");

        assert_eq!(manager.max_bindless_textures(), 0);
    }

    #[test]
    fn returns_actual_limit_value() {
        let features = Features::TEXTURE_BINDING_ARRAY;
        let mut limits = Limits::default();
        limits.max_sampled_textures_per_shader_stage = 1000;
        let manager = CapabilityManager::from_features_and_limits(features, limits, "Limited GPU");

        assert_eq!(manager.max_bindless_textures(), 1000);
    }

    #[test]
    fn returns_limit_from_advanced_tier() {
        let features = advanced_features();
        let limits = advanced_limits();
        let manager = CapabilityManager::from_features_and_limits(features, limits, "Advanced GPU");

        assert_eq!(manager.max_bindless_textures(), 16384);
    }

    #[test]
    fn returns_limit_from_full_tier() {
        let features = full_features();
        let limits = advanced_limits();
        let manager = CapabilityManager::from_features_and_limits(features, limits, "Full GPU");

        assert_eq!(manager.max_bindless_textures(), 16384);
    }

    #[test]
    fn returns_zero_for_minimal_tier() {
        let features = Features::empty();
        let limits = minimal_limits();
        let manager = CapabilityManager::from_features_and_limits(features, limits, "Minimal GPU");

        assert_eq!(manager.max_bindless_textures(), 0);
    }

    #[test]
    fn returns_zero_for_standard_tier() {
        let features = Features::empty();
        let limits = standard_limits();
        let manager = CapabilityManager::from_features_and_limits(features, limits, "Standard GPU");

        assert_eq!(manager.max_bindless_textures(), 0);
    }

    #[test]
    fn handles_very_large_limit() {
        let features = Features::TEXTURE_BINDING_ARRAY;
        let mut limits = Limits::default();
        limits.max_sampled_textures_per_shader_stage = u32::MAX;
        let manager = CapabilityManager::from_features_and_limits(features, limits, "Huge GPU");

        assert_eq!(manager.max_bindless_textures(), u32::MAX);
    }
}

// ============================================================================
// 11. report() Tests
// ============================================================================

mod report_generation {
    use super::*;

    #[test]
    fn report_contains_correct_tier() {
        let features = advanced_features();
        let limits = advanced_limits();
        let manager = CapabilityManager::from_features_and_limits(features, limits, "Test GPU");

        let report = manager.report();
        assert_eq!(report.tier, CapabilityTier::Advanced);
    }

    #[test]
    fn report_contains_correct_adapter_name() {
        let features = Features::empty();
        let limits = Limits::default();
        let manager = CapabilityManager::from_features_and_limits(features, limits, "My Custom GPU");

        let report = manager.report();
        assert_eq!(report.adapter_name, "My Custom GPU");
    }

    #[test]
    fn report_has_ray_tracing_matches_supports_ray_tracing() {
        let features = Features::RAY_TRACING_ACCELERATION_STRUCTURE;
        let limits = Limits::default();
        let manager = CapabilityManager::from_features_and_limits(features, limits, "RT GPU");

        let report = manager.report();
        assert_eq!(report.has_ray_tracing, manager.supports_ray_tracing());
        assert!(report.has_ray_tracing);
    }

    #[test]
    fn report_has_bindless_matches_supports_bindless() {
        let features = Features::TEXTURE_BINDING_ARRAY;
        let limits = Limits::default();
        let manager = CapabilityManager::from_features_and_limits(features, limits, "Bindless GPU");

        let report = manager.report();
        assert_eq!(report.has_bindless, manager.supports_bindless());
        assert!(report.has_bindless);
    }

    #[test]
    fn report_has_multi_draw_matches_supports_gpu_culling() {
        let features = Features::MULTI_DRAW_INDIRECT_COUNT;
        let limits = Limits::default();
        let manager = CapabilityManager::from_features_and_limits(features, limits, "Multi-Draw GPU");

        let report = manager.report();
        assert_eq!(report.has_multi_draw_indirect_count, manager.supports_gpu_culling());
        assert!(report.has_multi_draw_indirect_count);
    }

    #[test]
    fn report_contains_correct_limits() {
        let features = Features::empty();
        let mut limits = Limits::default();
        limits.max_texture_dimension_2d = 16384;
        limits.max_compute_invocations_per_workgroup = 1024;
        limits.max_storage_buffer_binding_size = 256 * 1024 * 1024;
        let manager = CapabilityManager::from_features_and_limits(features, limits, "Test GPU");

        let report = manager.report();
        assert_eq!(report.max_texture_dimension_2d, 16384);
        assert_eq!(report.max_compute_invocations, 1024);
        assert_eq!(report.max_storage_buffer_binding_size, 256 * 1024 * 1024);
    }

    #[test]
    fn report_has_storage_binding_array() {
        let features = Features::STORAGE_RESOURCE_BINDING_ARRAY;
        let limits = Limits::default();
        let manager = CapabilityManager::from_features_and_limits(features, limits, "Storage Array GPU");

        let report = manager.report();
        assert!(report.has_storage_binding_array);
    }

    #[test]
    fn report_display_contains_tier_info() {
        let features = Features::empty();
        let limits = standard_limits();
        let manager = CapabilityManager::from_features_and_limits(features, limits, "Display Test GPU");

        let report = manager.report();
        let display = format!("{}", report);
        assert!(display.contains("Standard"));
        assert!(display.contains("Display Test GPU"));
    }

    #[test]
    fn report_is_cloneable() {
        let features = Features::empty();
        let limits = Limits::default();
        let manager = CapabilityManager::from_features_and_limits(features, limits, "Clone Test GPU");

        let report = manager.report();
        let cloned = report.clone();
        assert_eq!(cloned.tier, report.tier);
        assert_eq!(cloned.adapter_name, report.adapter_name);
        assert_eq!(cloned.max_texture_dimension_2d, report.max_texture_dimension_2d);
    }

    #[test]
    fn report_debug_is_implemented() {
        let features = Features::empty();
        let limits = Limits::default();
        let manager = CapabilityManager::from_features_and_limits(features, limits, "Debug Test GPU");

        let report = manager.report();
        let debug = format!("{:?}", report);
        assert!(!debug.is_empty());
    }

    #[test]
    fn report_for_full_tier() {
        let features = full_features();
        let limits = advanced_limits();
        let manager = CapabilityManager::from_features_and_limits(features, limits, "Full GPU");

        let report = manager.report();
        assert_eq!(report.tier, CapabilityTier::Full);
        assert!(report.has_ray_tracing);
        assert!(report.has_bindless);
        assert!(report.has_multi_draw_indirect_count);
    }

    #[test]
    fn report_for_minimal_tier() {
        let features = Features::empty();
        let limits = minimal_limits();
        let manager = CapabilityManager::from_features_and_limits(features, limits, "Minimal GPU");

        let report = manager.report();
        assert_eq!(report.tier, CapabilityTier::Minimal);
        assert!(!report.has_ray_tracing);
        assert!(!report.has_bindless);
        assert!(!report.has_multi_draw_indirect_count);
    }
}

// ============================================================================
// 12. Basic Accessors Tests
// ============================================================================

mod basic_accessors {
    use super::*;

    #[test]
    fn tier_returns_correct_tier() {
        let test_cases = [
            (Features::empty(), minimal_limits(), CapabilityTier::Minimal),
            (Features::empty(), standard_limits(), CapabilityTier::Standard),
            (advanced_features(), advanced_limits(), CapabilityTier::Advanced),
            (Features::RAY_TRACING_ACCELERATION_STRUCTURE, Limits::default(), CapabilityTier::Full),
        ];

        for (features, limits, expected_tier) in test_cases {
            let manager = CapabilityManager::from_features_and_limits(features, limits, "Test GPU");
            assert_eq!(manager.tier(), expected_tier);
        }
    }

    #[test]
    fn features_returns_stored_features() {
        let features = Features::TEXTURE_BINDING_ARRAY | Features::TIMESTAMP_QUERY;
        let limits = Limits::default();
        let manager = CapabilityManager::from_features_and_limits(features, limits, "Test GPU");

        let stored_features = manager.features();
        assert!(stored_features.contains(Features::TEXTURE_BINDING_ARRAY));
        assert!(stored_features.contains(Features::TIMESTAMP_QUERY));
    }

    #[test]
    fn limits_returns_stored_limits() {
        let features = Features::empty();
        let mut limits = Limits::default();
        limits.max_texture_dimension_2d = 12345;
        let manager = CapabilityManager::from_features_and_limits(features, limits, "Test GPU");

        let stored_limits = manager.limits();
        assert_eq!(stored_limits.max_texture_dimension_2d, 12345);
    }

    #[test]
    fn adapter_name_returns_correct_name() {
        let features = Features::empty();
        let limits = Limits::default();
        let manager = CapabilityManager::from_features_and_limits(features, limits, "NVIDIA GeForce RTX 4090");

        assert_eq!(manager.adapter_name(), "NVIDIA GeForce RTX 4090");
    }

    #[test]
    fn adapter_name_handles_unicode() {
        let features = Features::empty();
        let limits = Limits::default();
        let manager = CapabilityManager::from_features_and_limits(features, limits, "GPU with Unicode");

        assert_eq!(manager.adapter_name(), "GPU with Unicode");
    }
}

// ============================================================================
// 13. Additional Methods Tests
// ============================================================================

mod additional_methods {
    use super::*;

    // -------------------------------------------------------------------------
    // supports_push_constants()
    // -------------------------------------------------------------------------

    #[test]
    fn supports_push_constants_true_with_feature() {
        let features = Features::PUSH_CONSTANTS;
        let limits = Limits::default();
        let manager = CapabilityManager::from_features_and_limits(features, limits, "Push Constants GPU");

        assert!(manager.supports_push_constants());
    }

    #[test]
    fn supports_push_constants_false_without_feature() {
        let features = Features::empty();
        let limits = Limits::default();
        let manager = CapabilityManager::from_features_and_limits(features, limits, "No Push Constants GPU");

        assert!(!manager.supports_push_constants());
    }

    // -------------------------------------------------------------------------
    // max_push_constant_size()
    // -------------------------------------------------------------------------

    #[test]
    fn max_push_constant_size_with_feature() {
        let features = Features::PUSH_CONSTANTS;
        let mut limits = Limits::default();
        limits.max_push_constant_size = 256;
        let manager = CapabilityManager::from_features_and_limits(features, limits, "Push Constants GPU");

        assert_eq!(manager.max_push_constant_size(), 256);
    }

    #[test]
    fn max_push_constant_size_zero_without_feature() {
        let features = Features::empty();
        let mut limits = Limits::default();
        limits.max_push_constant_size = 256; // Limit is set, but feature is not
        let manager = CapabilityManager::from_features_and_limits(features, limits, "No Push Constants GPU");

        assert_eq!(manager.max_push_constant_size(), 0);
    }

    // -------------------------------------------------------------------------
    // supports_storage_binding_array()
    // -------------------------------------------------------------------------

    #[test]
    fn supports_storage_binding_array_true_with_feature() {
        let features = Features::STORAGE_RESOURCE_BINDING_ARRAY;
        let limits = Limits::default();
        let manager = CapabilityManager::from_features_and_limits(features, limits, "Storage Array GPU");

        assert!(manager.supports_storage_binding_array());
    }

    #[test]
    fn supports_storage_binding_array_false_without_feature() {
        let features = Features::empty();
        let limits = Limits::default();
        let manager = CapabilityManager::from_features_and_limits(features, limits, "No Storage Array GPU");

        assert!(!manager.supports_storage_binding_array());
    }

    // -------------------------------------------------------------------------
    // supports_indirect_first_instance()
    // -------------------------------------------------------------------------

    #[test]
    fn supports_indirect_first_instance_true_with_feature() {
        let features = Features::INDIRECT_FIRST_INSTANCE;
        let limits = Limits::default();
        let manager = CapabilityManager::from_features_and_limits(features, limits, "Indirect GPU");

        assert!(manager.supports_indirect_first_instance());
    }

    #[test]
    fn supports_indirect_first_instance_false_without_feature() {
        let features = Features::empty();
        let limits = Limits::default();
        let manager = CapabilityManager::from_features_and_limits(features, limits, "No Indirect GPU");

        assert!(!manager.supports_indirect_first_instance());
    }

    // -------------------------------------------------------------------------
    // max_texture_2d()
    // -------------------------------------------------------------------------

    #[test]
    fn max_texture_2d_returns_limit() {
        let features = Features::empty();
        let mut limits = Limits::default();
        limits.max_texture_dimension_2d = 16384;
        let manager = CapabilityManager::from_features_and_limits(features, limits, "Test GPU");

        assert_eq!(manager.max_texture_2d(), 16384);
    }

    // -------------------------------------------------------------------------
    // max_workgroup_invocations()
    // -------------------------------------------------------------------------

    #[test]
    fn max_workgroup_invocations_returns_limit() {
        let features = Features::empty();
        let mut limits = Limits::default();
        limits.max_compute_invocations_per_workgroup = 2048;
        let manager = CapabilityManager::from_features_and_limits(features, limits, "Test GPU");

        assert_eq!(manager.max_workgroup_invocations(), 2048);
    }

    // -------------------------------------------------------------------------
    // max_storage_buffer_size()
    // -------------------------------------------------------------------------

    #[test]
    fn max_storage_buffer_size_returns_limit() {
        let features = Features::empty();
        let mut limits = Limits::default();
        limits.max_storage_buffer_binding_size = 512 * 1024 * 1024;
        let manager = CapabilityManager::from_features_and_limits(features, limits, "Test GPU");

        assert_eq!(manager.max_storage_buffer_size(), 512 * 1024 * 1024);
    }
}

// ============================================================================
// 14. meets_tier() Tests
// ============================================================================

mod meets_tier {
    use super::*;

    #[test]
    fn full_meets_all_tiers() {
        let features = Features::RAY_TRACING_ACCELERATION_STRUCTURE;
        let limits = Limits::default();
        let manager = CapabilityManager::from_features_and_limits(features, limits, "Full GPU");

        assert!(manager.meets_tier(CapabilityTier::Minimal));
        assert!(manager.meets_tier(CapabilityTier::Standard));
        assert!(manager.meets_tier(CapabilityTier::Advanced));
        assert!(manager.meets_tier(CapabilityTier::Full));
    }

    #[test]
    fn advanced_meets_advanced_and_below() {
        let features = advanced_features();
        let limits = advanced_limits();
        let manager = CapabilityManager::from_features_and_limits(features, limits, "Advanced GPU");

        assert!(manager.meets_tier(CapabilityTier::Minimal));
        assert!(manager.meets_tier(CapabilityTier::Standard));
        assert!(manager.meets_tier(CapabilityTier::Advanced));
        assert!(!manager.meets_tier(CapabilityTier::Full));
    }

    #[test]
    fn standard_meets_standard_and_below() {
        let features = Features::empty();
        let limits = standard_limits();
        let manager = CapabilityManager::from_features_and_limits(features, limits, "Standard GPU");

        assert!(manager.meets_tier(CapabilityTier::Minimal));
        assert!(manager.meets_tier(CapabilityTier::Standard));
        assert!(!manager.meets_tier(CapabilityTier::Advanced));
        assert!(!manager.meets_tier(CapabilityTier::Full));
    }

    #[test]
    fn minimal_meets_only_minimal() {
        let features = Features::empty();
        let limits = minimal_limits();
        let manager = CapabilityManager::from_features_and_limits(features, limits, "Minimal GPU");

        assert!(manager.meets_tier(CapabilityTier::Minimal));
        assert!(!manager.meets_tier(CapabilityTier::Standard));
        assert!(!manager.meets_tier(CapabilityTier::Advanced));
        assert!(!manager.meets_tier(CapabilityTier::Full));
    }

    #[test]
    fn meets_tier_is_same_as_ge_comparison() {
        let test_cases = [
            (Features::empty(), minimal_limits()),
            (Features::empty(), standard_limits()),
            (advanced_features(), advanced_limits()),
            (Features::RAY_TRACING_ACCELERATION_STRUCTURE, Limits::default()),
        ];

        let tiers = [
            CapabilityTier::Minimal,
            CapabilityTier::Standard,
            CapabilityTier::Advanced,
            CapabilityTier::Full,
        ];

        for (features, limits) in test_cases {
            let manager = CapabilityManager::from_features_and_limits(features, limits, "Test GPU");
            for &required in &tiers {
                assert_eq!(
                    manager.meets_tier(required),
                    manager.tier() >= required,
                    "meets_tier should match >= comparison"
                );
            }
        }
    }
}

// ============================================================================
// 15. Edge Cases Tests
// ============================================================================

mod edge_cases {
    use super::*;

    #[test]
    fn zero_limits() {
        let features = Features::empty();
        let mut limits = Limits::default();
        limits.max_texture_dimension_2d = 0;
        limits.max_storage_buffer_binding_size = 0;
        limits.max_compute_invocations_per_workgroup = 0;
        limits.max_sampled_textures_per_shader_stage = 0;

        let manager = CapabilityManager::from_features_and_limits(features, limits, "Zero GPU");

        assert_eq!(manager.tier(), CapabilityTier::Minimal);
        assert_eq!(manager.max_texture_2d(), 0);
        assert_eq!(manager.max_workgroup_invocations(), 0);
        assert_eq!(manager.max_storage_buffer_size(), 0);
    }

    #[test]
    fn max_limits() {
        let features = Features::empty();
        let mut limits = Limits::default();
        limits.max_texture_dimension_2d = u32::MAX;
        limits.max_storage_buffer_binding_size = u32::MAX;
        limits.max_compute_invocations_per_workgroup = u32::MAX;

        let manager = CapabilityManager::from_features_and_limits(features, limits, "Max GPU");

        // Without Advanced features, should be Standard
        assert_eq!(manager.tier(), CapabilityTier::Standard);
        assert_eq!(manager.max_texture_2d(), u32::MAX);
    }

    #[test]
    fn all_features() {
        let features = Features::all();
        let limits = advanced_limits();
        let manager = CapabilityManager::from_features_and_limits(features, limits, "All Features GPU");

        // Should be Full tier (has RT)
        assert_eq!(manager.tier(), CapabilityTier::Full);
        assert!(manager.supports_ray_tracing());
        assert!(manager.supports_bindless());
        assert!(manager.supports_gpu_culling());
        assert!(manager.supports_timestamp_queries());
        assert!(manager.supports_push_constants());
        assert!(manager.supports_storage_binding_array());
        assert!(manager.supports_indirect_first_instance());
    }

    #[test]
    fn boundary_standard_texture_exact() {
        let features = Features::empty();
        let mut limits = Limits::default();
        limits.max_texture_dimension_2d = STANDARD_TIER_MIN_TEXTURE_2D;
        limits.max_storage_buffer_binding_size = STANDARD_TIER_MIN_STORAGE_BUFFER_SIZE;

        let manager = CapabilityManager::from_features_and_limits(features, limits, "Boundary GPU");
        assert_eq!(manager.tier(), CapabilityTier::Standard);
    }

    #[test]
    fn boundary_standard_texture_one_below() {
        let features = Features::empty();
        let mut limits = Limits::default();
        limits.max_texture_dimension_2d = STANDARD_TIER_MIN_TEXTURE_2D - 1;
        limits.max_storage_buffer_binding_size = STANDARD_TIER_MIN_STORAGE_BUFFER_SIZE;

        let manager = CapabilityManager::from_features_and_limits(features, limits, "Boundary GPU");
        assert_eq!(manager.tier(), CapabilityTier::Minimal);
    }

    #[test]
    fn boundary_standard_storage_one_below() {
        let features = Features::empty();
        let mut limits = Limits::default();
        limits.max_texture_dimension_2d = STANDARD_TIER_MIN_TEXTURE_2D;
        limits.max_storage_buffer_binding_size = STANDARD_TIER_MIN_STORAGE_BUFFER_SIZE - 1;

        let manager = CapabilityManager::from_features_and_limits(features, limits, "Boundary GPU");
        assert_eq!(manager.tier(), CapabilityTier::Minimal);
    }

    #[test]
    fn boundary_advanced_workgroup_exact() {
        let features = advanced_features();
        let mut limits = standard_limits();
        limits.max_compute_invocations_per_workgroup = ADVANCED_TIER_MIN_WORKGROUP_INVOCATIONS;

        let manager = CapabilityManager::from_features_and_limits(features, limits, "Boundary GPU");
        assert_eq!(manager.tier(), CapabilityTier::Advanced);
    }

    #[test]
    fn boundary_advanced_workgroup_one_below() {
        let features = advanced_features();
        let mut limits = standard_limits();
        limits.max_compute_invocations_per_workgroup = ADVANCED_TIER_MIN_WORKGROUP_INVOCATIONS - 1;

        let manager = CapabilityManager::from_features_and_limits(features, limits, "Boundary GPU");
        assert_eq!(manager.tier(), CapabilityTier::Standard);
    }

    #[test]
    fn rt_overrides_low_limits() {
        let features = Features::RAY_TRACING_ACCELERATION_STRUCTURE;
        let limits = minimal_limits();

        let manager = CapabilityManager::from_features_and_limits(features, limits, "RT with Low Limits");
        assert_eq!(manager.tier(), CapabilityTier::Full);
    }

    #[test]
    fn advanced_features_with_minimal_limits() {
        let features = advanced_features();
        let limits = minimal_limits();

        let manager = CapabilityManager::from_features_and_limits(features, limits, "Advanced Features Low Limits");
        // Should not be Advanced because workgroup is too small
        assert_eq!(manager.tier(), CapabilityTier::Minimal);
    }
}

// ============================================================================
// 16. Trait Implementations Tests
// ============================================================================

mod trait_implementations {
    use super::*;

    #[test]
    fn debug_is_implemented() {
        let features = Features::empty();
        let limits = Limits::default();
        let manager = CapabilityManager::from_features_and_limits(features, limits, "Debug Test GPU");

        let debug_str = format!("{:?}", manager);
        assert!(!debug_str.is_empty());
        // Should contain struct name or fields
        assert!(debug_str.contains("CapabilityManager") || debug_str.contains("tier"));
    }

    #[test]
    fn clone_is_deep() {
        let features = Features::TEXTURE_BINDING_ARRAY | Features::TIMESTAMP_QUERY;
        let mut limits = Limits::default();
        limits.max_texture_dimension_2d = 16384;
        let manager = CapabilityManager::from_features_and_limits(features, limits, "Clone Test GPU");

        let cloned = manager.clone();

        assert_eq!(cloned.tier(), manager.tier());
        assert_eq!(cloned.adapter_name(), manager.adapter_name());
        assert_eq!(cloned.max_texture_2d(), manager.max_texture_2d());
        assert_eq!(cloned.supports_bindless(), manager.supports_bindless());
        assert_eq!(cloned.supports_timestamp_queries(), manager.supports_timestamp_queries());
    }

    #[test]
    fn clone_is_independent() {
        let features = Features::empty();
        let limits = Limits::default();
        let manager = CapabilityManager::from_features_and_limits(features, limits, "Clone Test GPU");

        let cloned = manager.clone();

        // Both should work independently
        assert_eq!(manager.tier(), cloned.tier());
        assert_eq!(manager.adapter_name(), cloned.adapter_name());
    }
}

// ============================================================================
// 17. Consistency Tests
// ============================================================================

mod consistency {
    use super::*;

    #[test]
    fn tier_and_supports_ray_tracing_consistent() {
        // Full tier should support RT
        let features = Features::RAY_TRACING_ACCELERATION_STRUCTURE;
        let limits = Limits::default();
        let manager = CapabilityManager::from_features_and_limits(features, limits, "Full GPU");

        assert_eq!(manager.tier(), CapabilityTier::Full);
        assert!(manager.supports_ray_tracing());
    }

    #[test]
    fn tier_and_supports_bindless_consistent() {
        // Advanced tier should support bindless
        let features = advanced_features();
        let limits = advanced_limits();
        let manager = CapabilityManager::from_features_and_limits(features, limits, "Advanced GPU");

        assert_eq!(manager.tier(), CapabilityTier::Advanced);
        assert!(manager.supports_bindless());
    }

    #[test]
    fn tier_and_supports_gpu_culling_consistent() {
        // Advanced tier should support GPU culling
        let features = advanced_features();
        let limits = advanced_limits();
        let manager = CapabilityManager::from_features_and_limits(features, limits, "Advanced GPU");

        assert_eq!(manager.tier(), CapabilityTier::Advanced);
        assert!(manager.supports_gpu_culling());
    }

    #[test]
    fn report_consistent_with_manager() {
        let features = full_features();
        let limits = advanced_limits();
        let manager = CapabilityManager::from_features_and_limits(features, limits, "Consistency GPU");

        let report = manager.report();

        assert_eq!(report.tier, manager.tier());
        assert_eq!(report.has_ray_tracing, manager.supports_ray_tracing());
        assert_eq!(report.has_bindless, manager.supports_bindless());
        assert_eq!(report.has_multi_draw_indirect_count, manager.supports_gpu_culling());
        assert_eq!(report.max_texture_dimension_2d, manager.max_texture_2d());
        assert_eq!(report.max_compute_invocations, manager.max_workgroup_invocations());
        assert_eq!(report.max_storage_buffer_binding_size, manager.max_storage_buffer_size());
    }

    #[test]
    fn render_path_consistent_with_tier() {
        let test_cases = [
            (Features::RAY_TRACING_ACCELERATION_STRUCTURE, Limits::default(), CapabilityTier::Full, RenderPath::RayTraced),
            (advanced_features(), advanced_limits(), CapabilityTier::Advanced, RenderPath::GPUDriven),
            (Features::empty(), standard_limits(), CapabilityTier::Standard, RenderPath::Traditional),
            (Features::empty(), minimal_limits(), CapabilityTier::Minimal, RenderPath::Fallback),
        ];

        for (features, limits, expected_tier, expected_path) in test_cases {
            let manager = CapabilityManager::from_features_and_limits(features, limits, "Test GPU");
            assert_eq!(manager.tier(), expected_tier);
            assert_eq!(manager.select_render_path(), expected_path);
        }
    }

    #[test]
    fn max_bindless_consistent_with_supports_bindless() {
        // With bindless: max_bindless_textures > 0
        let features = Features::TEXTURE_BINDING_ARRAY;
        let mut limits = Limits::default();
        limits.max_sampled_textures_per_shader_stage = 100;
        let manager = CapabilityManager::from_features_and_limits(features, limits.clone(), "Bindless GPU");

        assert!(manager.supports_bindless());
        assert!(manager.max_bindless_textures() > 0);

        // Without bindless: max_bindless_textures == 0
        let features = Features::empty();
        let manager = CapabilityManager::from_features_and_limits(features, limits, "No Bindless GPU");

        assert!(!manager.supports_bindless());
        assert_eq!(manager.max_bindless_textures(), 0);
    }
}
