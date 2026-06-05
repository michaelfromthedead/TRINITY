//! Whitebox tests for T-WGPU-P1.2.4 Feature Detection
//!
//! Tests the internal implementation of feature detection including:
//! - FeatureTier enum and ordering
//! - AdapterFeatures construction and methods
//! - Individual feature detection methods (16+)
//! - Utility methods (tier, count, supports, etc.)
//! - FeaturesSummary structure
//! - inspect_features() function
//!
//! These tests verify the internal logic by constructing AdapterFeatures
//! with synthetic wgpu::Features bitflags.

use renderer_backend::device::{AdapterFeatures, FeaturesSummary, FeatureTier};

// ============================================================================
// FeatureTier Tests
// ============================================================================

mod feature_tier_tests {
    use super::*;

    #[test]
    fn test_feature_tier_ordering_minimal_less_than_standard() {
        assert!(FeatureTier::Minimal < FeatureTier::Standard);
    }

    #[test]
    fn test_feature_tier_ordering_standard_less_than_advanced() {
        assert!(FeatureTier::Standard < FeatureTier::Advanced);
    }

    #[test]
    fn test_feature_tier_ordering_advanced_less_than_full() {
        assert!(FeatureTier::Advanced < FeatureTier::Full);
    }

    #[test]
    fn test_feature_tier_ordering_full_chain() {
        assert!(FeatureTier::Minimal < FeatureTier::Standard);
        assert!(FeatureTier::Standard < FeatureTier::Advanced);
        assert!(FeatureTier::Advanced < FeatureTier::Full);
        assert!(FeatureTier::Minimal < FeatureTier::Full);
    }

    #[test]
    fn test_feature_tier_equality_same() {
        assert_eq!(FeatureTier::Minimal, FeatureTier::Minimal);
        assert_eq!(FeatureTier::Standard, FeatureTier::Standard);
        assert_eq!(FeatureTier::Advanced, FeatureTier::Advanced);
        assert_eq!(FeatureTier::Full, FeatureTier::Full);
    }

    #[test]
    fn test_feature_tier_inequality_different() {
        assert_ne!(FeatureTier::Minimal, FeatureTier::Standard);
        assert_ne!(FeatureTier::Standard, FeatureTier::Advanced);
        assert_ne!(FeatureTier::Advanced, FeatureTier::Full);
        assert_ne!(FeatureTier::Minimal, FeatureTier::Full);
    }

    #[test]
    fn test_feature_tier_description_minimal() {
        let desc = FeatureTier::Minimal.description();
        assert!(desc.contains("Minimal"));
        assert!(desc.contains("core WebGPU"));
    }

    #[test]
    fn test_feature_tier_description_standard() {
        let desc = FeatureTier::Standard.description();
        assert!(desc.contains("Standard"));
        assert!(desc.contains("optional features"));
    }

    #[test]
    fn test_feature_tier_description_advanced() {
        let desc = FeatureTier::Advanced.description();
        assert!(desc.contains("Advanced"));
        assert!(desc.contains("modern GPU"));
    }

    #[test]
    fn test_feature_tier_description_full() {
        let desc = FeatureTier::Full.description();
        assert!(desc.contains("Full"));
        assert!(desc.contains("all features"));
    }

    #[test]
    fn test_feature_tier_display_minimal() {
        assert_eq!(format!("{}", FeatureTier::Minimal), "Minimal");
    }

    #[test]
    fn test_feature_tier_display_standard() {
        assert_eq!(format!("{}", FeatureTier::Standard), "Standard");
    }

    #[test]
    fn test_feature_tier_display_advanced() {
        assert_eq!(format!("{}", FeatureTier::Advanced), "Advanced");
    }

    #[test]
    fn test_feature_tier_display_full() {
        assert_eq!(format!("{}", FeatureTier::Full), "Full");
    }

    #[test]
    fn test_feature_tier_hash_uniqueness() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(FeatureTier::Minimal);
        set.insert(FeatureTier::Standard);
        set.insert(FeatureTier::Advanced);
        set.insert(FeatureTier::Full);
        assert_eq!(set.len(), 4, "All tiers should have unique hashes");
    }

    #[test]
    fn test_feature_tier_hash_duplicates() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(FeatureTier::Minimal);
        set.insert(FeatureTier::Minimal); // Duplicate
        assert_eq!(set.len(), 1, "Duplicate should not increase set size");
    }

    #[test]
    fn test_feature_tier_clone() {
        let tier = FeatureTier::Advanced;
        let cloned = tier.clone();
        assert_eq!(tier, cloned);
    }

    #[test]
    fn test_feature_tier_copy() {
        let tier = FeatureTier::Full;
        let copied = tier; // Copy, not move
        assert_eq!(tier, copied);
        // Both should be usable after copy
        assert_eq!(tier.description(), copied.description());
    }

    #[test]
    fn test_feature_tier_debug() {
        let debug_str = format!("{:?}", FeatureTier::Advanced);
        assert!(debug_str.contains("Advanced"));
    }
}

// ============================================================================
// AdapterFeatures Construction Tests
// ============================================================================

mod adapter_features_construction_tests {
    use super::*;

    #[test]
    fn test_adapter_features_from_empty() {
        let features = AdapterFeatures {
            raw: wgpu::Features::empty(),
        };
        assert_eq!(features.count(), 0);
    }

    #[test]
    fn test_adapter_features_raw_field_accessible() {
        let raw = wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::PUSH_CONSTANTS;
        let features = AdapterFeatures { raw };
        assert!(features.raw.contains(wgpu::Features::TIMESTAMP_QUERY));
        assert!(features.raw.contains(wgpu::Features::PUSH_CONSTANTS));
    }

    #[test]
    fn test_adapter_features_clone() {
        let original = AdapterFeatures {
            raw: wgpu::Features::SHADER_F16 | wgpu::Features::MULTIVIEW,
        };
        let cloned = original.clone();
        assert_eq!(original.count(), cloned.count());
        assert_eq!(original.has_shader_f16(), cloned.has_shader_f16());
        assert_eq!(original.has_multiview(), cloned.has_multiview());
    }

    #[test]
    fn test_adapter_features_debug() {
        let features = AdapterFeatures {
            raw: wgpu::Features::TIMESTAMP_QUERY,
        };
        let debug_str = format!("{:?}", features);
        assert!(debug_str.contains("AdapterFeatures"));
    }
}

// ============================================================================
// Individual Feature Method Tests (16+ features)
// ============================================================================

mod individual_feature_tests {
    use super::*;

    // --- Core Features ---

    #[test]
    fn test_has_depth_clip_control_present() {
        let features = AdapterFeatures {
            raw: wgpu::Features::DEPTH_CLIP_CONTROL,
        };
        assert!(features.has_depth_clip_control());
    }

    #[test]
    fn test_has_depth_clip_control_absent() {
        let features = AdapterFeatures {
            raw: wgpu::Features::empty(),
        };
        assert!(!features.has_depth_clip_control());
    }

    #[test]
    fn test_has_depth32float_stencil8_present() {
        let features = AdapterFeatures {
            raw: wgpu::Features::DEPTH32FLOAT_STENCIL8,
        };
        assert!(features.has_depth32float_stencil8());
    }

    #[test]
    fn test_has_depth32float_stencil8_absent() {
        let features = AdapterFeatures {
            raw: wgpu::Features::empty(),
        };
        assert!(!features.has_depth32float_stencil8());
    }

    // --- Texture Compression Features ---

    #[test]
    fn test_has_texture_compression_bc_present() {
        let features = AdapterFeatures {
            raw: wgpu::Features::TEXTURE_COMPRESSION_BC,
        };
        assert!(features.has_texture_compression_bc());
    }

    #[test]
    fn test_has_texture_compression_bc_absent() {
        let features = AdapterFeatures {
            raw: wgpu::Features::empty(),
        };
        assert!(!features.has_texture_compression_bc());
    }

    #[test]
    fn test_has_texture_compression_etc2_present() {
        let features = AdapterFeatures {
            raw: wgpu::Features::TEXTURE_COMPRESSION_ETC2,
        };
        assert!(features.has_texture_compression_etc2());
    }

    #[test]
    fn test_has_texture_compression_etc2_absent() {
        let features = AdapterFeatures {
            raw: wgpu::Features::empty(),
        };
        assert!(!features.has_texture_compression_etc2());
    }

    #[test]
    fn test_has_texture_compression_astc_present() {
        let features = AdapterFeatures {
            raw: wgpu::Features::TEXTURE_COMPRESSION_ASTC,
        };
        assert!(features.has_texture_compression_astc());
    }

    #[test]
    fn test_has_texture_compression_astc_absent() {
        let features = AdapterFeatures {
            raw: wgpu::Features::empty(),
        };
        assert!(!features.has_texture_compression_astc());
    }

    #[test]
    fn test_has_texture_compression_astc_hdr_present() {
        let features = AdapterFeatures {
            raw: wgpu::Features::TEXTURE_COMPRESSION_ASTC_HDR,
        };
        assert!(features.has_texture_compression_astc_hdr());
    }

    #[test]
    fn test_has_texture_compression_astc_hdr_absent() {
        let features = AdapterFeatures {
            raw: wgpu::Features::empty(),
        };
        assert!(!features.has_texture_compression_astc_hdr());
    }

    // --- Rendering Features ---

    #[test]
    fn test_has_indirect_first_instance_present() {
        let features = AdapterFeatures {
            raw: wgpu::Features::INDIRECT_FIRST_INSTANCE,
        };
        assert!(features.has_indirect_first_instance());
    }

    #[test]
    fn test_has_indirect_first_instance_absent() {
        let features = AdapterFeatures {
            raw: wgpu::Features::empty(),
        };
        assert!(!features.has_indirect_first_instance());
    }

    #[test]
    fn test_has_multiview_present() {
        let features = AdapterFeatures {
            raw: wgpu::Features::MULTIVIEW,
        };
        assert!(features.has_multiview());
    }

    #[test]
    fn test_has_multiview_absent() {
        let features = AdapterFeatures {
            raw: wgpu::Features::empty(),
        };
        assert!(!features.has_multiview());
    }

    // --- Query Features ---

    #[test]
    fn test_has_timestamp_query_present() {
        let features = AdapterFeatures {
            raw: wgpu::Features::TIMESTAMP_QUERY,
        };
        assert!(features.has_timestamp_query());
    }

    #[test]
    fn test_has_timestamp_query_absent() {
        let features = AdapterFeatures {
            raw: wgpu::Features::empty(),
        };
        assert!(!features.has_timestamp_query());
    }

    #[test]
    fn test_has_pipeline_statistics_query_present() {
        let features = AdapterFeatures {
            raw: wgpu::Features::PIPELINE_STATISTICS_QUERY,
        };
        assert!(features.has_pipeline_statistics_query());
    }

    #[test]
    fn test_has_pipeline_statistics_query_absent() {
        let features = AdapterFeatures {
            raw: wgpu::Features::empty(),
        };
        assert!(!features.has_pipeline_statistics_query());
    }

    // --- Shader Features ---

    #[test]
    fn test_has_shader_f16_present() {
        let features = AdapterFeatures {
            raw: wgpu::Features::SHADER_F16,
        };
        assert!(features.has_shader_f16());
    }

    #[test]
    fn test_has_shader_f16_absent() {
        let features = AdapterFeatures {
            raw: wgpu::Features::empty(),
        };
        assert!(!features.has_shader_f16());
    }

    #[test]
    fn test_has_push_constants_present() {
        let features = AdapterFeatures {
            raw: wgpu::Features::PUSH_CONSTANTS,
        };
        assert!(features.has_push_constants());
    }

    #[test]
    fn test_has_push_constants_absent() {
        let features = AdapterFeatures {
            raw: wgpu::Features::empty(),
        };
        assert!(!features.has_push_constants());
    }

    // --- Format Features ---

    #[test]
    fn test_has_rg11b10ufloat_renderable_present() {
        let features = AdapterFeatures {
            raw: wgpu::Features::RG11B10UFLOAT_RENDERABLE,
        };
        assert!(features.has_rg11b10ufloat_renderable());
    }

    #[test]
    fn test_has_rg11b10ufloat_renderable_absent() {
        let features = AdapterFeatures {
            raw: wgpu::Features::empty(),
        };
        assert!(!features.has_rg11b10ufloat_renderable());
    }

    #[test]
    fn test_has_bgra8unorm_storage_present() {
        let features = AdapterFeatures {
            raw: wgpu::Features::BGRA8UNORM_STORAGE,
        };
        assert!(features.has_bgra8unorm_storage());
    }

    #[test]
    fn test_has_bgra8unorm_storage_absent() {
        let features = AdapterFeatures {
            raw: wgpu::Features::empty(),
        };
        assert!(!features.has_bgra8unorm_storage());
    }

    #[test]
    fn test_has_float32_filterable_present() {
        let features = AdapterFeatures {
            raw: wgpu::Features::FLOAT32_FILTERABLE,
        };
        assert!(features.has_float32_filterable());
    }

    #[test]
    fn test_has_float32_filterable_absent() {
        let features = AdapterFeatures {
            raw: wgpu::Features::empty(),
        };
        assert!(!features.has_float32_filterable());
    }

    #[test]
    fn test_has_texture_format_16bit_norm_present() {
        let features = AdapterFeatures {
            raw: wgpu::Features::TEXTURE_FORMAT_16BIT_NORM,
        };
        assert!(features.has_texture_format_16bit_norm());
    }

    #[test]
    fn test_has_texture_format_16bit_norm_absent() {
        let features = AdapterFeatures {
            raw: wgpu::Features::empty(),
        };
        assert!(!features.has_texture_format_16bit_norm());
    }

    // --- Multiple Features ---

    #[test]
    fn test_multiple_features_all_present() {
        let features = AdapterFeatures {
            raw: wgpu::Features::DEPTH_CLIP_CONTROL
                | wgpu::Features::DEPTH32FLOAT_STENCIL8
                | wgpu::Features::TEXTURE_COMPRESSION_BC
                | wgpu::Features::TEXTURE_COMPRESSION_ETC2
                | wgpu::Features::TEXTURE_COMPRESSION_ASTC
                | wgpu::Features::TEXTURE_COMPRESSION_ASTC_HDR
                | wgpu::Features::INDIRECT_FIRST_INSTANCE
                | wgpu::Features::MULTIVIEW
                | wgpu::Features::TIMESTAMP_QUERY
                | wgpu::Features::PIPELINE_STATISTICS_QUERY
                | wgpu::Features::SHADER_F16
                | wgpu::Features::PUSH_CONSTANTS
                | wgpu::Features::RG11B10UFLOAT_RENDERABLE
                | wgpu::Features::BGRA8UNORM_STORAGE
                | wgpu::Features::FLOAT32_FILTERABLE
                | wgpu::Features::TEXTURE_FORMAT_16BIT_NORM,
        };

        assert!(features.has_depth_clip_control());
        assert!(features.has_depth32float_stencil8());
        assert!(features.has_texture_compression_bc());
        assert!(features.has_texture_compression_etc2());
        assert!(features.has_texture_compression_astc());
        assert!(features.has_texture_compression_astc_hdr());
        assert!(features.has_indirect_first_instance());
        assert!(features.has_multiview());
        assert!(features.has_timestamp_query());
        assert!(features.has_pipeline_statistics_query());
        assert!(features.has_shader_f16());
        assert!(features.has_push_constants());
        assert!(features.has_rg11b10ufloat_renderable());
        assert!(features.has_bgra8unorm_storage());
        assert!(features.has_float32_filterable());
        assert!(features.has_texture_format_16bit_norm());
    }
}

// ============================================================================
// Utility Method Tests
// ============================================================================

mod utility_method_tests {
    use super::*;

    // --- has_any_texture_compression() ---

    #[test]
    fn test_has_any_texture_compression_none() {
        let features = AdapterFeatures {
            raw: wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::SHADER_F16,
        };
        assert!(!features.has_any_texture_compression());
    }

    #[test]
    fn test_has_any_texture_compression_bc_only() {
        let features = AdapterFeatures {
            raw: wgpu::Features::TEXTURE_COMPRESSION_BC,
        };
        assert!(features.has_any_texture_compression());
    }

    #[test]
    fn test_has_any_texture_compression_etc2_only() {
        let features = AdapterFeatures {
            raw: wgpu::Features::TEXTURE_COMPRESSION_ETC2,
        };
        assert!(features.has_any_texture_compression());
    }

    #[test]
    fn test_has_any_texture_compression_astc_only() {
        let features = AdapterFeatures {
            raw: wgpu::Features::TEXTURE_COMPRESSION_ASTC,
        };
        assert!(features.has_any_texture_compression());
    }

    #[test]
    fn test_has_any_texture_compression_all() {
        let features = AdapterFeatures {
            raw: wgpu::Features::TEXTURE_COMPRESSION_BC
                | wgpu::Features::TEXTURE_COMPRESSION_ETC2
                | wgpu::Features::TEXTURE_COMPRESSION_ASTC,
        };
        assert!(features.has_any_texture_compression());
    }

    // --- count() ---

    #[test]
    fn test_count_zero_features() {
        let features = AdapterFeatures {
            raw: wgpu::Features::empty(),
        };
        assert_eq!(features.count(), 0);
    }

    #[test]
    fn test_count_one_feature() {
        let features = AdapterFeatures {
            raw: wgpu::Features::TIMESTAMP_QUERY,
        };
        assert_eq!(features.count(), 1);
    }

    #[test]
    fn test_count_three_features() {
        let features = AdapterFeatures {
            raw: wgpu::Features::TIMESTAMP_QUERY
                | wgpu::Features::PUSH_CONSTANTS
                | wgpu::Features::SHADER_F16,
        };
        assert_eq!(features.count(), 3);
    }

    #[test]
    fn test_count_many_features() {
        let features = AdapterFeatures {
            raw: wgpu::Features::DEPTH_CLIP_CONTROL
                | wgpu::Features::DEPTH32FLOAT_STENCIL8
                | wgpu::Features::TEXTURE_COMPRESSION_BC
                | wgpu::Features::TEXTURE_COMPRESSION_ETC2
                | wgpu::Features::TEXTURE_COMPRESSION_ASTC
                | wgpu::Features::INDIRECT_FIRST_INSTANCE
                | wgpu::Features::MULTIVIEW
                | wgpu::Features::TIMESTAMP_QUERY
                | wgpu::Features::PIPELINE_STATISTICS_QUERY
                | wgpu::Features::SHADER_F16
                | wgpu::Features::PUSH_CONSTANTS
                | wgpu::Features::RG11B10UFLOAT_RENDERABLE,
        };
        assert_eq!(features.count(), 12);
    }

    // --- tier() classification ---

    #[test]
    fn test_tier_minimal_zero_features() {
        let features = AdapterFeatures {
            raw: wgpu::Features::empty(),
        };
        assert_eq!(features.tier(), FeatureTier::Minimal);
    }

    #[test]
    fn test_tier_minimal_three_features() {
        let features = AdapterFeatures {
            raw: wgpu::Features::DEPTH_CLIP_CONTROL
                | wgpu::Features::DEPTH32FLOAT_STENCIL8
                | wgpu::Features::TEXTURE_COMPRESSION_BC,
        };
        assert_eq!(features.tier(), FeatureTier::Minimal);
    }

    #[test]
    fn test_tier_standard_four_features_no_advanced() {
        let features = AdapterFeatures {
            raw: wgpu::Features::DEPTH_CLIP_CONTROL
                | wgpu::Features::DEPTH32FLOAT_STENCIL8
                | wgpu::Features::TEXTURE_COMPRESSION_BC
                | wgpu::Features::INDIRECT_FIRST_INSTANCE,
        };
        assert_eq!(features.tier(), FeatureTier::Standard);
    }

    #[test]
    fn test_tier_standard_seven_features_no_advanced() {
        let features = AdapterFeatures {
            raw: wgpu::Features::DEPTH_CLIP_CONTROL
                | wgpu::Features::DEPTH32FLOAT_STENCIL8
                | wgpu::Features::TEXTURE_COMPRESSION_BC
                | wgpu::Features::TEXTURE_COMPRESSION_ETC2
                | wgpu::Features::INDIRECT_FIRST_INSTANCE
                | wgpu::Features::MULTIVIEW
                | wgpu::Features::RG11B10UFLOAT_RENDERABLE,
        };
        assert_eq!(features.tier(), FeatureTier::Standard);
    }

    #[test]
    fn test_tier_advanced_eight_features_with_timestamp() {
        let features = AdapterFeatures {
            raw: wgpu::Features::DEPTH_CLIP_CONTROL
                | wgpu::Features::DEPTH32FLOAT_STENCIL8
                | wgpu::Features::TEXTURE_COMPRESSION_BC
                | wgpu::Features::TEXTURE_COMPRESSION_ETC2
                | wgpu::Features::INDIRECT_FIRST_INSTANCE
                | wgpu::Features::MULTIVIEW
                | wgpu::Features::RG11B10UFLOAT_RENDERABLE
                | wgpu::Features::TIMESTAMP_QUERY,
        };
        assert_eq!(features.tier(), FeatureTier::Advanced);
    }

    #[test]
    fn test_tier_advanced_eight_features_with_pipeline_stats() {
        let features = AdapterFeatures {
            raw: wgpu::Features::DEPTH_CLIP_CONTROL
                | wgpu::Features::DEPTH32FLOAT_STENCIL8
                | wgpu::Features::TEXTURE_COMPRESSION_BC
                | wgpu::Features::TEXTURE_COMPRESSION_ETC2
                | wgpu::Features::INDIRECT_FIRST_INSTANCE
                | wgpu::Features::MULTIVIEW
                | wgpu::Features::RG11B10UFLOAT_RENDERABLE
                | wgpu::Features::PIPELINE_STATISTICS_QUERY,
        };
        assert_eq!(features.tier(), FeatureTier::Advanced);
    }

    #[test]
    fn test_tier_advanced_eight_features_with_f16() {
        let features = AdapterFeatures {
            raw: wgpu::Features::DEPTH_CLIP_CONTROL
                | wgpu::Features::DEPTH32FLOAT_STENCIL8
                | wgpu::Features::TEXTURE_COMPRESSION_BC
                | wgpu::Features::TEXTURE_COMPRESSION_ETC2
                | wgpu::Features::INDIRECT_FIRST_INSTANCE
                | wgpu::Features::MULTIVIEW
                | wgpu::Features::RG11B10UFLOAT_RENDERABLE
                | wgpu::Features::SHADER_F16,
        };
        assert_eq!(features.tier(), FeatureTier::Advanced);
    }

    #[test]
    fn test_tier_advanced_eleven_features_with_advanced() {
        let features = AdapterFeatures {
            raw: wgpu::Features::DEPTH_CLIP_CONTROL
                | wgpu::Features::DEPTH32FLOAT_STENCIL8
                | wgpu::Features::TEXTURE_COMPRESSION_BC
                | wgpu::Features::TEXTURE_COMPRESSION_ETC2
                | wgpu::Features::TEXTURE_COMPRESSION_ASTC
                | wgpu::Features::INDIRECT_FIRST_INSTANCE
                | wgpu::Features::MULTIVIEW
                | wgpu::Features::RG11B10UFLOAT_RENDERABLE
                | wgpu::Features::BGRA8UNORM_STORAGE
                | wgpu::Features::FLOAT32_FILTERABLE
                | wgpu::Features::TIMESTAMP_QUERY,
        };
        assert_eq!(features.tier(), FeatureTier::Advanced);
    }

    #[test]
    fn test_tier_full_twelve_features_with_advanced() {
        let features = AdapterFeatures {
            raw: wgpu::Features::DEPTH_CLIP_CONTROL
                | wgpu::Features::DEPTH32FLOAT_STENCIL8
                | wgpu::Features::TEXTURE_COMPRESSION_BC
                | wgpu::Features::TEXTURE_COMPRESSION_ETC2
                | wgpu::Features::TEXTURE_COMPRESSION_ASTC
                | wgpu::Features::INDIRECT_FIRST_INSTANCE
                | wgpu::Features::MULTIVIEW
                | wgpu::Features::RG11B10UFLOAT_RENDERABLE
                | wgpu::Features::BGRA8UNORM_STORAGE
                | wgpu::Features::FLOAT32_FILTERABLE
                | wgpu::Features::PUSH_CONSTANTS
                | wgpu::Features::TIMESTAMP_QUERY,
        };
        assert_eq!(features.tier(), FeatureTier::Full);
    }

    #[test]
    fn test_tier_full_sixteen_features() {
        let features = AdapterFeatures {
            raw: wgpu::Features::DEPTH_CLIP_CONTROL
                | wgpu::Features::DEPTH32FLOAT_STENCIL8
                | wgpu::Features::TEXTURE_COMPRESSION_BC
                | wgpu::Features::TEXTURE_COMPRESSION_ETC2
                | wgpu::Features::TEXTURE_COMPRESSION_ASTC
                | wgpu::Features::TEXTURE_COMPRESSION_ASTC_HDR
                | wgpu::Features::INDIRECT_FIRST_INSTANCE
                | wgpu::Features::MULTIVIEW
                | wgpu::Features::TIMESTAMP_QUERY
                | wgpu::Features::PIPELINE_STATISTICS_QUERY
                | wgpu::Features::SHADER_F16
                | wgpu::Features::PUSH_CONSTANTS
                | wgpu::Features::RG11B10UFLOAT_RENDERABLE
                | wgpu::Features::BGRA8UNORM_STORAGE
                | wgpu::Features::FLOAT32_FILTERABLE
                | wgpu::Features::TEXTURE_FORMAT_16BIT_NORM,
        };
        assert_eq!(features.tier(), FeatureTier::Full);
    }

    #[test]
    fn test_tier_many_features_without_advanced_is_standard() {
        // 8 features but NO advanced features (timestamp, pipeline stats, f16)
        let features = AdapterFeatures {
            raw: wgpu::Features::DEPTH_CLIP_CONTROL
                | wgpu::Features::DEPTH32FLOAT_STENCIL8
                | wgpu::Features::TEXTURE_COMPRESSION_BC
                | wgpu::Features::TEXTURE_COMPRESSION_ETC2
                | wgpu::Features::TEXTURE_COMPRESSION_ASTC
                | wgpu::Features::INDIRECT_FIRST_INSTANCE
                | wgpu::Features::MULTIVIEW
                | wgpu::Features::RG11B10UFLOAT_RENDERABLE,
        };
        // Has 8 features but no advanced features, so should be Standard
        assert_eq!(features.tier(), FeatureTier::Standard);
    }

    // --- supports() ---

    #[test]
    fn test_supports_single_feature_present() {
        let features = AdapterFeatures {
            raw: wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::PUSH_CONSTANTS,
        };
        assert!(features.supports(wgpu::Features::TIMESTAMP_QUERY));
    }

    #[test]
    fn test_supports_single_feature_absent() {
        let features = AdapterFeatures {
            raw: wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::PUSH_CONSTANTS,
        };
        assert!(!features.supports(wgpu::Features::SHADER_F16));
    }

    #[test]
    fn test_supports_multiple_features_all_present() {
        let features = AdapterFeatures {
            raw: wgpu::Features::TIMESTAMP_QUERY
                | wgpu::Features::PUSH_CONSTANTS
                | wgpu::Features::SHADER_F16,
        };
        assert!(features.supports(wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::PUSH_CONSTANTS));
    }

    #[test]
    fn test_supports_multiple_features_some_absent() {
        let features = AdapterFeatures {
            raw: wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::PUSH_CONSTANTS,
        };
        assert!(!features.supports(wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::SHADER_F16));
    }

    #[test]
    fn test_supports_empty_features() {
        let features = AdapterFeatures {
            raw: wgpu::Features::TIMESTAMP_QUERY,
        };
        assert!(features.supports(wgpu::Features::empty()));
    }

    // --- best_compression_format() ---

    #[test]
    fn test_best_compression_format_none() {
        let features = AdapterFeatures {
            raw: wgpu::Features::TIMESTAMP_QUERY,
        };
        assert_eq!(features.best_compression_format(), "none");
    }

    #[test]
    fn test_best_compression_format_bc_preferred() {
        let features = AdapterFeatures {
            raw: wgpu::Features::TEXTURE_COMPRESSION_BC
                | wgpu::Features::TEXTURE_COMPRESSION_ETC2
                | wgpu::Features::TEXTURE_COMPRESSION_ASTC,
        };
        assert_eq!(features.best_compression_format(), "BC");
    }

    #[test]
    fn test_best_compression_format_astc_when_no_bc() {
        let features = AdapterFeatures {
            raw: wgpu::Features::TEXTURE_COMPRESSION_ETC2
                | wgpu::Features::TEXTURE_COMPRESSION_ASTC,
        };
        assert_eq!(features.best_compression_format(), "ASTC");
    }

    #[test]
    fn test_best_compression_format_etc2_fallback() {
        let features = AdapterFeatures {
            raw: wgpu::Features::TEXTURE_COMPRESSION_ETC2,
        };
        assert_eq!(features.best_compression_format(), "ETC2");
    }

    #[test]
    fn test_best_compression_format_bc_only() {
        let features = AdapterFeatures {
            raw: wgpu::Features::TEXTURE_COMPRESSION_BC,
        };
        assert_eq!(features.best_compression_format(), "BC");
    }

    #[test]
    fn test_best_compression_format_astc_only() {
        let features = AdapterFeatures {
            raw: wgpu::Features::TEXTURE_COMPRESSION_ASTC,
        };
        assert_eq!(features.best_compression_format(), "ASTC");
    }
}

// ============================================================================
// FeaturesSummary Tests
// ============================================================================

mod features_summary_tests {
    use super::*;

    #[test]
    fn test_summary_returns_correct_struct() {
        let features = AdapterFeatures {
            raw: wgpu::Features::TEXTURE_COMPRESSION_BC
                | wgpu::Features::TEXTURE_COMPRESSION_ETC2
                | wgpu::Features::TIMESTAMP_QUERY
                | wgpu::Features::SHADER_F16
                | wgpu::Features::PUSH_CONSTANTS
                | wgpu::Features::INDIRECT_FIRST_INSTANCE
                | wgpu::Features::MULTIVIEW,
        };
        let summary = features.summary();

        assert_eq!(summary.total_count, 7);
        assert!(summary.has_compression_bc);
        assert!(summary.has_compression_etc2);
        assert!(!summary.has_compression_astc);
        assert!(!summary.has_compression_astc_hdr);
        assert!(summary.has_timestamp_query);
        assert!(!summary.has_pipeline_statistics);
        assert!(summary.has_shader_f16);
        assert!(summary.has_push_constants);
        assert!(summary.has_multiview);
        assert!(summary.has_indirect_first_instance);
    }

    #[test]
    fn test_summary_tier_matches_features() {
        let features = AdapterFeatures {
            raw: wgpu::Features::DEPTH_CLIP_CONTROL
                | wgpu::Features::DEPTH32FLOAT_STENCIL8
                | wgpu::Features::TEXTURE_COMPRESSION_BC
                | wgpu::Features::TEXTURE_COMPRESSION_ETC2
                | wgpu::Features::INDIRECT_FIRST_INSTANCE
                | wgpu::Features::MULTIVIEW
                | wgpu::Features::RG11B10UFLOAT_RENDERABLE
                | wgpu::Features::TIMESTAMP_QUERY,
        };
        let summary = features.summary();

        assert_eq!(summary.tier, features.tier());
        assert_eq!(summary.tier, FeatureTier::Advanced);
    }

    #[test]
    fn test_summary_has_any_compression_true() {
        let features = AdapterFeatures {
            raw: wgpu::Features::TEXTURE_COMPRESSION_BC,
        };
        let summary = features.summary();
        assert!(summary.has_any_compression());
    }

    #[test]
    fn test_summary_has_any_compression_false() {
        let features = AdapterFeatures {
            raw: wgpu::Features::TIMESTAMP_QUERY,
        };
        let summary = features.summary();
        assert!(!summary.has_any_compression());
    }

    #[test]
    fn test_summary_has_profiling_true() {
        let features = AdapterFeatures {
            raw: wgpu::Features::TIMESTAMP_QUERY,
        };
        let summary = features.summary();
        assert!(summary.has_profiling());
    }

    #[test]
    fn test_summary_has_profiling_false() {
        let features = AdapterFeatures {
            raw: wgpu::Features::SHADER_F16,
        };
        let summary = features.summary();
        assert!(!summary.has_profiling());
    }

    #[test]
    fn test_summary_has_gpu_driven_true() {
        let features = AdapterFeatures {
            raw: wgpu::Features::INDIRECT_FIRST_INSTANCE,
        };
        let summary = features.summary();
        assert!(summary.has_gpu_driven());
    }

    #[test]
    fn test_summary_has_gpu_driven_false() {
        let features = AdapterFeatures {
            raw: wgpu::Features::TIMESTAMP_QUERY,
        };
        let summary = features.summary();
        assert!(!summary.has_gpu_driven());
    }

    #[test]
    fn test_summary_display_contains_tier() {
        let features = AdapterFeatures {
            raw: wgpu::Features::TEXTURE_COMPRESSION_BC | wgpu::Features::TIMESTAMP_QUERY,
        };
        let summary = features.summary();
        let display = format!("{}", summary);

        assert!(display.contains("features"));
    }

    #[test]
    fn test_summary_display_contains_compression_info() {
        let features = AdapterFeatures {
            raw: wgpu::Features::TEXTURE_COMPRESSION_BC | wgpu::Features::TEXTURE_COMPRESSION_ETC2,
        };
        let summary = features.summary();
        let display = format!("{}", summary);

        assert!(display.contains("BC"));
        assert!(display.contains("ETC2"));
    }

    #[test]
    fn test_summary_display_contains_profiling() {
        let features = AdapterFeatures {
            raw: wgpu::Features::TIMESTAMP_QUERY,
        };
        let summary = features.summary();
        let display = format!("{}", summary);

        assert!(display.contains("Profiling"));
    }

    #[test]
    fn test_summary_display_contains_f16() {
        let features = AdapterFeatures {
            raw: wgpu::Features::SHADER_F16,
        };
        let summary = features.summary();
        let display = format!("{}", summary);

        assert!(display.contains("F16"));
    }

    #[test]
    fn test_summary_copy_semantics() {
        let features = AdapterFeatures {
            raw: wgpu::Features::TIMESTAMP_QUERY,
        };
        let summary = features.summary();
        let copy = summary; // Copy semantics

        assert_eq!(summary.total_count, copy.total_count);
        assert_eq!(summary.tier, copy.tier);
        assert_eq!(summary.has_timestamp_query, copy.has_timestamp_query);
    }

    #[test]
    fn test_summary_clone() {
        let features = AdapterFeatures {
            raw: wgpu::Features::SHADER_F16,
        };
        let summary = features.summary();
        let cloned = summary.clone();

        assert_eq!(summary.total_count, cloned.total_count);
        assert_eq!(summary.has_shader_f16, cloned.has_shader_f16);
    }
}

// ============================================================================
// AdapterFeatures Display Tests
// ============================================================================

mod adapter_features_display_tests {
    use super::*;

    #[test]
    fn test_display_contains_feature_count() {
        let features = AdapterFeatures {
            raw: wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::PUSH_CONSTANTS,
        };
        let display = format!("{}", features);
        assert!(display.contains("2 available"));
    }

    #[test]
    fn test_display_contains_tier() {
        let features = AdapterFeatures {
            raw: wgpu::Features::TIMESTAMP_QUERY,
        };
        let display = format!("{}", features);
        assert!(display.contains("Tier:"));
    }

    #[test]
    fn test_display_contains_texture_compression_section() {
        let features = AdapterFeatures {
            raw: wgpu::Features::TEXTURE_COMPRESSION_BC,
        };
        let display = format!("{}", features);
        assert!(display.contains("Texture Compression"));
        assert!(display.contains("BC (Desktop)"));
        assert!(display.contains("ETC2 (Mobile)"));
        assert!(display.contains("ASTC"));
    }

    #[test]
    fn test_display_contains_rendering_section() {
        let features = AdapterFeatures {
            raw: wgpu::Features::INDIRECT_FIRST_INSTANCE,
        };
        let display = format!("{}", features);
        assert!(display.contains("Rendering"));
        assert!(display.contains("Indirect"));
        assert!(display.contains("Multiview"));
    }

    #[test]
    fn test_display_contains_queries_section() {
        let features = AdapterFeatures {
            raw: wgpu::Features::TIMESTAMP_QUERY,
        };
        let display = format!("{}", features);
        assert!(display.contains("Queries"));
        assert!(display.contains("Timestamp"));
        assert!(display.contains("Pipeline stats"));
    }

    #[test]
    fn test_display_contains_shader_section() {
        let features = AdapterFeatures {
            raw: wgpu::Features::SHADER_F16,
        };
        let display = format!("{}", features);
        assert!(display.contains("Shader"));
        assert!(display.contains("F16"));
        assert!(display.contains("Push constants"));
    }

    #[test]
    fn test_display_contains_formats_section() {
        let features = AdapterFeatures {
            raw: wgpu::Features::DEPTH_CLIP_CONTROL,
        };
        let display = format!("{}", features);
        assert!(display.contains("Formats"));
        assert!(display.contains("Depth clip control"));
    }

    #[test]
    fn test_display_contains_best_format() {
        let features = AdapterFeatures {
            raw: wgpu::Features::TEXTURE_COMPRESSION_BC,
        };
        let display = format!("{}", features);
        assert!(display.contains("Best format"));
        assert!(display.contains("BC"));
    }

    #[test]
    fn test_display_empty_features() {
        let features = AdapterFeatures {
            raw: wgpu::Features::empty(),
        };
        let display = format!("{}", features);
        assert!(display.contains("0 available"));
        assert!(display.contains("Minimal"));
    }
}

// ============================================================================
// Edge Case Tests
// ============================================================================

mod edge_case_tests {
    use super::*;

    #[test]
    fn test_tier_boundary_three_to_four_features() {
        // 3 features = Minimal
        let three_features = AdapterFeatures {
            raw: wgpu::Features::DEPTH_CLIP_CONTROL
                | wgpu::Features::DEPTH32FLOAT_STENCIL8
                | wgpu::Features::TEXTURE_COMPRESSION_BC,
        };
        assert_eq!(three_features.tier(), FeatureTier::Minimal);

        // 4 features = Standard
        let four_features = AdapterFeatures {
            raw: wgpu::Features::DEPTH_CLIP_CONTROL
                | wgpu::Features::DEPTH32FLOAT_STENCIL8
                | wgpu::Features::TEXTURE_COMPRESSION_BC
                | wgpu::Features::INDIRECT_FIRST_INSTANCE,
        };
        assert_eq!(four_features.tier(), FeatureTier::Standard);
    }

    #[test]
    fn test_tier_boundary_seven_to_eight_features_with_advanced() {
        // 7 features with advanced feature = Standard (not enough features)
        let seven_features = AdapterFeatures {
            raw: wgpu::Features::DEPTH_CLIP_CONTROL
                | wgpu::Features::DEPTH32FLOAT_STENCIL8
                | wgpu::Features::TEXTURE_COMPRESSION_BC
                | wgpu::Features::TEXTURE_COMPRESSION_ETC2
                | wgpu::Features::INDIRECT_FIRST_INSTANCE
                | wgpu::Features::MULTIVIEW
                | wgpu::Features::TIMESTAMP_QUERY, // Advanced feature
        };
        assert_eq!(seven_features.tier(), FeatureTier::Standard);

        // 8 features with advanced feature = Advanced
        let eight_features = AdapterFeatures {
            raw: wgpu::Features::DEPTH_CLIP_CONTROL
                | wgpu::Features::DEPTH32FLOAT_STENCIL8
                | wgpu::Features::TEXTURE_COMPRESSION_BC
                | wgpu::Features::TEXTURE_COMPRESSION_ETC2
                | wgpu::Features::INDIRECT_FIRST_INSTANCE
                | wgpu::Features::MULTIVIEW
                | wgpu::Features::RG11B10UFLOAT_RENDERABLE
                | wgpu::Features::TIMESTAMP_QUERY, // Advanced feature
        };
        assert_eq!(eight_features.tier(), FeatureTier::Advanced);
    }

    #[test]
    fn test_tier_boundary_eleven_to_twelve_features() {
        // 11 features with advanced = Advanced
        let eleven_features = AdapterFeatures {
            raw: wgpu::Features::DEPTH_CLIP_CONTROL
                | wgpu::Features::DEPTH32FLOAT_STENCIL8
                | wgpu::Features::TEXTURE_COMPRESSION_BC
                | wgpu::Features::TEXTURE_COMPRESSION_ETC2
                | wgpu::Features::TEXTURE_COMPRESSION_ASTC
                | wgpu::Features::INDIRECT_FIRST_INSTANCE
                | wgpu::Features::MULTIVIEW
                | wgpu::Features::RG11B10UFLOAT_RENDERABLE
                | wgpu::Features::BGRA8UNORM_STORAGE
                | wgpu::Features::FLOAT32_FILTERABLE
                | wgpu::Features::TIMESTAMP_QUERY, // Advanced feature
        };
        assert_eq!(eleven_features.tier(), FeatureTier::Advanced);

        // 12 features with advanced = Full
        let twelve_features = AdapterFeatures {
            raw: wgpu::Features::DEPTH_CLIP_CONTROL
                | wgpu::Features::DEPTH32FLOAT_STENCIL8
                | wgpu::Features::TEXTURE_COMPRESSION_BC
                | wgpu::Features::TEXTURE_COMPRESSION_ETC2
                | wgpu::Features::TEXTURE_COMPRESSION_ASTC
                | wgpu::Features::INDIRECT_FIRST_INSTANCE
                | wgpu::Features::MULTIVIEW
                | wgpu::Features::RG11B10UFLOAT_RENDERABLE
                | wgpu::Features::BGRA8UNORM_STORAGE
                | wgpu::Features::FLOAT32_FILTERABLE
                | wgpu::Features::PUSH_CONSTANTS
                | wgpu::Features::TIMESTAMP_QUERY, // Advanced feature
        };
        assert_eq!(twelve_features.tier(), FeatureTier::Full);
    }

    #[test]
    fn test_advanced_feature_alone_not_enough_for_advanced_tier() {
        // Timestamp query alone = Minimal (only 1 feature)
        let timestamp_only = AdapterFeatures {
            raw: wgpu::Features::TIMESTAMP_QUERY,
        };
        assert_eq!(timestamp_only.tier(), FeatureTier::Minimal);

        // Pipeline stats alone = Minimal (only 1 feature)
        let pipeline_stats_only = AdapterFeatures {
            raw: wgpu::Features::PIPELINE_STATISTICS_QUERY,
        };
        assert_eq!(pipeline_stats_only.tier(), FeatureTier::Minimal);

        // F16 alone = Minimal (only 1 feature)
        let f16_only = AdapterFeatures {
            raw: wgpu::Features::SHADER_F16,
        };
        assert_eq!(f16_only.tier(), FeatureTier::Minimal);
    }

    #[test]
    fn test_all_compression_formats_with_hdr() {
        let features = AdapterFeatures {
            raw: wgpu::Features::TEXTURE_COMPRESSION_BC
                | wgpu::Features::TEXTURE_COMPRESSION_ETC2
                | wgpu::Features::TEXTURE_COMPRESSION_ASTC
                | wgpu::Features::TEXTURE_COMPRESSION_ASTC_HDR,
        };

        assert!(features.has_texture_compression_bc());
        assert!(features.has_texture_compression_etc2());
        assert!(features.has_texture_compression_astc());
        assert!(features.has_texture_compression_astc_hdr());
        assert!(features.has_any_texture_compression());
        assert_eq!(features.best_compression_format(), "BC");
    }

    #[test]
    fn test_supports_with_all_required_features() {
        let features = AdapterFeatures {
            raw: wgpu::Features::TIMESTAMP_QUERY
                | wgpu::Features::PUSH_CONSTANTS
                | wgpu::Features::SHADER_F16
                | wgpu::Features::MULTIVIEW,
        };

        let required = wgpu::Features::TIMESTAMP_QUERY
            | wgpu::Features::PUSH_CONSTANTS
            | wgpu::Features::SHADER_F16
            | wgpu::Features::MULTIVIEW;

        assert!(features.supports(required));
    }

    #[test]
    fn test_supports_with_superset_of_features() {
        let features = AdapterFeatures {
            raw: wgpu::Features::TIMESTAMP_QUERY
                | wgpu::Features::PUSH_CONSTANTS
                | wgpu::Features::SHADER_F16
                | wgpu::Features::MULTIVIEW
                | wgpu::Features::TEXTURE_COMPRESSION_BC,
        };

        let required = wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::PUSH_CONSTANTS;
        assert!(features.supports(required));
    }
}

// ============================================================================
// Platform-Specific Feature Documentation Tests
// ============================================================================

mod platform_specific_documentation_tests {
    use super::*;

    /// Test that BC compression is typically desktop (documented in code).
    #[test]
    fn test_bc_compression_platform_documentation() {
        let features = AdapterFeatures {
            raw: wgpu::Features::TEXTURE_COMPRESSION_BC,
        };
        let display = format!("{}", features);
        assert!(display.contains("Desktop"));
    }

    /// Test that ETC2 compression is typically mobile (documented in code).
    #[test]
    fn test_etc2_compression_platform_documentation() {
        let features = AdapterFeatures {
            raw: wgpu::Features::TEXTURE_COMPRESSION_ETC2,
        };
        let display = format!("{}", features);
        assert!(display.contains("Mobile"));
    }

    /// Test that multiview is documented for VR.
    #[test]
    fn test_multiview_vr_documentation() {
        let features = AdapterFeatures {
            raw: wgpu::Features::MULTIVIEW,
        };
        let display = format!("{}", features);
        assert!(display.contains("VR"));
    }
}

// ============================================================================
// Regression Tests
// ============================================================================

mod regression_tests {
    use super::*;

    /// Ensure count() doesn't overflow on empty features.
    #[test]
    fn test_count_no_overflow_empty() {
        let features = AdapterFeatures {
            raw: wgpu::Features::empty(),
        };
        assert_eq!(features.count(), 0);
    }

    /// Ensure tier() is deterministic.
    #[test]
    fn test_tier_deterministic() {
        let features = AdapterFeatures {
            raw: wgpu::Features::TIMESTAMP_QUERY
                | wgpu::Features::PUSH_CONSTANTS
                | wgpu::Features::SHADER_F16
                | wgpu::Features::MULTIVIEW
                | wgpu::Features::TEXTURE_COMPRESSION_BC
                | wgpu::Features::TEXTURE_COMPRESSION_ETC2
                | wgpu::Features::INDIRECT_FIRST_INSTANCE
                | wgpu::Features::RG11B10UFLOAT_RENDERABLE,
        };

        let tier1 = features.tier();
        let tier2 = features.tier();
        let tier3 = features.tier();

        assert_eq!(tier1, tier2);
        assert_eq!(tier2, tier3);
    }

    /// Ensure summary() returns consistent results.
    #[test]
    fn test_summary_consistent() {
        let features = AdapterFeatures {
            raw: wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::TEXTURE_COMPRESSION_BC,
        };

        let summary1 = features.summary();
        let summary2 = features.summary();

        assert_eq!(summary1.total_count, summary2.total_count);
        assert_eq!(summary1.tier, summary2.tier);
        assert_eq!(summary1.has_timestamp_query, summary2.has_timestamp_query);
        assert_eq!(summary1.has_compression_bc, summary2.has_compression_bc);
    }
}
