//! Whitebox structural tests for CapabilityTier and CapabilityReport.
//!
//! These tests verify the internal structure and behavior of the capability tier
//! detection algorithm, including tier ordering, feature requirements, and
//! edge cases in the detection logic.
//!
//! Task: T-WGPU-P1.5.1 - Capability Tier Detection
//!
//! Acceptance Criteria Tested:
//! 1. Full tier: RT (ray tracing) features present
//! 2. Advanced tier: bindless + multi-draw + large workgroup
//! 3. Standard tier: 8K textures + compute
//! 4. Minimal tier: fallback
//! 5. Tier ordering (Full > Advanced > Standard > Minimal)
//!
//! WHITEBOX coverage plan:
//!   - Path A: CapabilityTier enum variants construction and identity
//!   - Path B: Tier ordering via Ord/PartialOrd trait implementations
//!   - Path C: from_features_and_limits() with Full tier (RT feature present)
//!   - Path D: from_features_and_limits() with Advanced tier (all 3 requirements)
//!   - Path E: from_features_and_limits() with Standard tier (8K + storage)
//!   - Path F: from_features_and_limits() with Minimal tier (fallback)
//!   - Path G: tier_name() returns correct strings for all variants
//!   - Path H: supports_ray_tracing() only true for Full
//!   - Path I: supports_bindless() true for Advanced and Full
//!   - Path J: supports_gpu_driven() true for Advanced and Full
//!   - Path K: supports_8k_textures() true for Standard, Advanced, Full
//!   - Path L: supports_compute() true for Standard, Advanced, Full
//!   - Path M: meets_requirement() boundary conditions
//!   - Path N: Edge cases - boundary limit values
//!   - Path O: Edge cases - partial feature combinations
//!   - Path P: Trait implementations (Debug, Clone, Copy, Eq, Hash, Default)
//!   - Path Q: CapabilityReport generation and display
//!   - Path R: Utility functions (features_for_tier, can_achieve_tier)
//!   - Path S: Tier rank() internal method for ordering

use renderer_backend::device::{
    can_achieve_tier, features_for_tier, CapabilityReport, CapabilityTier,
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

/// Create limits that meet Advanced tier requirements exactly.
fn advanced_limits() -> Limits {
    let mut limits = Limits::default();
    limits.max_texture_dimension_2d = 16384;
    limits.max_compute_invocations_per_workgroup = ADVANCED_TIER_MIN_WORKGROUP_INVOCATIONS;
    limits.max_storage_buffer_binding_size = 256 * 1024 * 1024;
    limits
}

/// Create limits just below Standard tier thresholds.
fn below_standard_limits() -> Limits {
    let mut limits = Limits::default();
    limits.max_texture_dimension_2d = STANDARD_TIER_MIN_TEXTURE_2D - 1; // 8191
    limits.max_storage_buffer_binding_size = STANDARD_TIER_MIN_STORAGE_BUFFER_SIZE;
    limits
}

/// Create limits just below Advanced tier workgroup threshold.
fn below_advanced_workgroup_limits() -> Limits {
    let mut limits = Limits::default();
    limits.max_texture_dimension_2d = 16384;
    limits.max_compute_invocations_per_workgroup = ADVANCED_TIER_MIN_WORKGROUP_INVOCATIONS - 1;
    limits.max_storage_buffer_binding_size = 256 * 1024 * 1024;
    limits
}

// ============================================================================
// 1. CapabilityTier Enum Variants Tests
// ============================================================================

mod capability_tier_variants {
    use super::*;

    #[test]
    fn all_four_variants_exist() {
        // Verify all four tier variants exist and are distinct
        let minimal = CapabilityTier::Minimal;
        let standard = CapabilityTier::Standard;
        let advanced = CapabilityTier::Advanced;
        let full = CapabilityTier::Full;

        // Each variant is self-equal
        assert_eq!(minimal, CapabilityTier::Minimal);
        assert_eq!(standard, CapabilityTier::Standard);
        assert_eq!(advanced, CapabilityTier::Advanced);
        assert_eq!(full, CapabilityTier::Full);
    }

    #[test]
    fn variants_are_mutually_distinct() {
        let variants = [
            CapabilityTier::Minimal,
            CapabilityTier::Standard,
            CapabilityTier::Advanced,
            CapabilityTier::Full,
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
    fn default_is_minimal() {
        let tier: CapabilityTier = Default::default();
        assert_eq!(tier, CapabilityTier::Minimal);
    }
}

// ============================================================================
// 2. Tier Ordering (Ord/PartialOrd) Tests
// ============================================================================

mod tier_ordering {
    use super::*;
    use std::cmp::Ordering;

    #[test]
    fn minimal_is_lowest() {
        assert!(CapabilityTier::Minimal < CapabilityTier::Standard);
        assert!(CapabilityTier::Minimal < CapabilityTier::Advanced);
        assert!(CapabilityTier::Minimal < CapabilityTier::Full);
    }

    #[test]
    fn full_is_highest() {
        assert!(CapabilityTier::Full > CapabilityTier::Minimal);
        assert!(CapabilityTier::Full > CapabilityTier::Standard);
        assert!(CapabilityTier::Full > CapabilityTier::Advanced);
    }

    #[test]
    fn strict_ordering_chain() {
        // Minimal < Standard < Advanced < Full
        assert!(CapabilityTier::Minimal < CapabilityTier::Standard);
        assert!(CapabilityTier::Standard < CapabilityTier::Advanced);
        assert!(CapabilityTier::Advanced < CapabilityTier::Full);
    }

    #[test]
    fn transitive_ordering() {
        // If A < B and B < C, then A < C
        let a = CapabilityTier::Minimal;
        let b = CapabilityTier::Standard;
        let c = CapabilityTier::Advanced;
        let d = CapabilityTier::Full;

        assert!(a < b && b < c && a < c);
        assert!(b < c && c < d && b < d);
        assert!(a < d); // Minimal < Full (full transitive chain)
    }

    #[test]
    fn ordering_is_reflexive() {
        let variants = [
            CapabilityTier::Minimal,
            CapabilityTier::Standard,
            CapabilityTier::Advanced,
            CapabilityTier::Full,
        ];

        for tier in &variants {
            assert!(tier <= tier);
            assert!(tier >= tier);
            assert_eq!(tier.cmp(tier), Ordering::Equal);
        }
    }

    #[test]
    fn ordering_is_antisymmetric() {
        // If A <= B and B <= A, then A == B
        let a = CapabilityTier::Standard;
        let b = CapabilityTier::Standard;
        assert!(a <= b && b <= a);
        assert_eq!(a, b);
    }

    #[test]
    fn partial_cmp_matches_cmp() {
        let variants = [
            CapabilityTier::Minimal,
            CapabilityTier::Standard,
            CapabilityTier::Advanced,
            CapabilityTier::Full,
        ];

        for a in &variants {
            for b in &variants {
                assert_eq!(a.partial_cmp(b), Some(a.cmp(b)));
            }
        }
    }

    #[test]
    fn greater_than_or_equal_works() {
        assert!(CapabilityTier::Full >= CapabilityTier::Full);
        assert!(CapabilityTier::Full >= CapabilityTier::Advanced);
        assert!(CapabilityTier::Full >= CapabilityTier::Standard);
        assert!(CapabilityTier::Full >= CapabilityTier::Minimal);

        assert!(CapabilityTier::Advanced >= CapabilityTier::Advanced);
        assert!(CapabilityTier::Advanced >= CapabilityTier::Standard);
        assert!(CapabilityTier::Advanced >= CapabilityTier::Minimal);

        assert!(CapabilityTier::Standard >= CapabilityTier::Standard);
        assert!(CapabilityTier::Standard >= CapabilityTier::Minimal);

        assert!(CapabilityTier::Minimal >= CapabilityTier::Minimal);
    }

    #[test]
    fn less_than_or_equal_works() {
        assert!(CapabilityTier::Minimal <= CapabilityTier::Minimal);
        assert!(CapabilityTier::Minimal <= CapabilityTier::Standard);
        assert!(CapabilityTier::Minimal <= CapabilityTier::Advanced);
        assert!(CapabilityTier::Minimal <= CapabilityTier::Full);

        assert!(CapabilityTier::Standard <= CapabilityTier::Standard);
        assert!(CapabilityTier::Standard <= CapabilityTier::Advanced);
        assert!(CapabilityTier::Standard <= CapabilityTier::Full);

        assert!(CapabilityTier::Advanced <= CapabilityTier::Advanced);
        assert!(CapabilityTier::Advanced <= CapabilityTier::Full);

        assert!(CapabilityTier::Full <= CapabilityTier::Full);
    }

    #[test]
    fn tiers_are_sortable() {
        let mut tiers = vec![
            CapabilityTier::Full,
            CapabilityTier::Minimal,
            CapabilityTier::Advanced,
            CapabilityTier::Standard,
        ];

        tiers.sort();

        assert_eq!(
            tiers,
            vec![
                CapabilityTier::Minimal,
                CapabilityTier::Standard,
                CapabilityTier::Advanced,
                CapabilityTier::Full,
            ]
        );
    }

    #[test]
    fn min_max_work_correctly() {
        let tiers = [
            CapabilityTier::Advanced,
            CapabilityTier::Minimal,
            CapabilityTier::Full,
            CapabilityTier::Standard,
        ];

        assert_eq!(*tiers.iter().min().unwrap(), CapabilityTier::Minimal);
        assert_eq!(*tiers.iter().max().unwrap(), CapabilityTier::Full);
    }
}

// ============================================================================
// 3. from_features_and_limits() Detection Tests
// ============================================================================

mod tier_detection {
    use super::*;

    // ------------------------------------------------------------------------
    // Full Tier Detection (RT present)
    // ------------------------------------------------------------------------

    #[test]
    fn detects_full_tier_with_rt_only() {
        let features = Features::RAY_TRACING_ACCELERATION_STRUCTURE;
        let limits = minimal_limits();

        let tier = CapabilityTier::from_features_and_limits(&features, &limits);
        assert_eq!(tier, CapabilityTier::Full);
    }

    #[test]
    fn detects_full_tier_with_rt_and_all_other_features() {
        let features = Features::RAY_TRACING_ACCELERATION_STRUCTURE
            | Features::TEXTURE_BINDING_ARRAY
            | Features::MULTI_DRAW_INDIRECT_COUNT;
        let limits = advanced_limits();

        let tier = CapabilityTier::from_features_and_limits(&features, &limits);
        assert_eq!(tier, CapabilityTier::Full);
    }

    #[test]
    fn full_tier_takes_precedence_over_advanced() {
        // Has all Advanced requirements plus RT
        let features = Features::RAY_TRACING_ACCELERATION_STRUCTURE
            | Features::TEXTURE_BINDING_ARRAY
            | Features::MULTI_DRAW_INDIRECT_COUNT;
        let limits = advanced_limits();

        let tier = CapabilityTier::from_features_and_limits(&features, &limits);
        assert_eq!(
            tier,
            CapabilityTier::Full,
            "RT should always result in Full tier"
        );
    }

    // ------------------------------------------------------------------------
    // Advanced Tier Detection (bindless + multi-draw + large workgroup)
    // ------------------------------------------------------------------------

    #[test]
    fn detects_advanced_tier_with_all_requirements() {
        let features = Features::TEXTURE_BINDING_ARRAY | Features::MULTI_DRAW_INDIRECT_COUNT;
        let limits = advanced_limits();

        let tier = CapabilityTier::from_features_and_limits(&features, &limits);
        assert_eq!(tier, CapabilityTier::Advanced);
    }

    #[test]
    fn advanced_requires_texture_binding_array() {
        // Missing TEXTURE_BINDING_ARRAY
        let features = Features::MULTI_DRAW_INDIRECT_COUNT;
        let limits = advanced_limits();

        let tier = CapabilityTier::from_features_and_limits(&features, &limits);
        assert_ne!(tier, CapabilityTier::Advanced);
        assert_eq!(tier, CapabilityTier::Standard);
    }

    #[test]
    fn advanced_requires_multi_draw_indirect_count() {
        // Missing MULTI_DRAW_INDIRECT_COUNT
        let features = Features::TEXTURE_BINDING_ARRAY;
        let limits = advanced_limits();

        let tier = CapabilityTier::from_features_and_limits(&features, &limits);
        assert_ne!(tier, CapabilityTier::Advanced);
        assert_eq!(tier, CapabilityTier::Standard);
    }

    #[test]
    fn advanced_requires_large_workgroup() {
        // Has features but small workgroup
        let features = Features::TEXTURE_BINDING_ARRAY | Features::MULTI_DRAW_INDIRECT_COUNT;
        let limits = below_advanced_workgroup_limits();

        let tier = CapabilityTier::from_features_and_limits(&features, &limits);
        assert_ne!(tier, CapabilityTier::Advanced);
        assert_eq!(tier, CapabilityTier::Standard);
    }

    #[test]
    fn advanced_workgroup_boundary_exact() {
        // Exactly at threshold
        let features = Features::TEXTURE_BINDING_ARRAY | Features::MULTI_DRAW_INDIRECT_COUNT;
        let mut limits = advanced_limits();
        limits.max_compute_invocations_per_workgroup = ADVANCED_TIER_MIN_WORKGROUP_INVOCATIONS;

        let tier = CapabilityTier::from_features_and_limits(&features, &limits);
        assert_eq!(tier, CapabilityTier::Advanced);
    }

    #[test]
    fn advanced_workgroup_boundary_one_below() {
        // One below threshold
        let features = Features::TEXTURE_BINDING_ARRAY | Features::MULTI_DRAW_INDIRECT_COUNT;
        let mut limits = advanced_limits();
        limits.max_compute_invocations_per_workgroup = ADVANCED_TIER_MIN_WORKGROUP_INVOCATIONS - 1;

        let tier = CapabilityTier::from_features_and_limits(&features, &limits);
        assert_ne!(tier, CapabilityTier::Advanced);
    }

    // ------------------------------------------------------------------------
    // Standard Tier Detection (8K textures + 128MB storage)
    // ------------------------------------------------------------------------

    #[test]
    fn detects_standard_tier_with_8k_textures_and_storage() {
        let features = Features::empty();
        let limits = standard_limits();

        let tier = CapabilityTier::from_features_and_limits(&features, &limits);
        assert_eq!(tier, CapabilityTier::Standard);
    }

    #[test]
    fn standard_requires_8k_textures() {
        let features = Features::empty();
        let mut limits = standard_limits();
        limits.max_texture_dimension_2d = STANDARD_TIER_MIN_TEXTURE_2D - 1;

        let tier = CapabilityTier::from_features_and_limits(&features, &limits);
        assert_eq!(tier, CapabilityTier::Minimal);
    }

    #[test]
    fn standard_requires_large_storage_buffer() {
        let features = Features::empty();
        let mut limits = standard_limits();
        limits.max_storage_buffer_binding_size = STANDARD_TIER_MIN_STORAGE_BUFFER_SIZE - 1;

        let tier = CapabilityTier::from_features_and_limits(&features, &limits);
        assert_eq!(tier, CapabilityTier::Minimal);
    }

    #[test]
    fn standard_texture_boundary_exact() {
        let features = Features::empty();
        let mut limits = Limits::default();
        limits.max_texture_dimension_2d = STANDARD_TIER_MIN_TEXTURE_2D; // Exactly 8192
        limits.max_storage_buffer_binding_size = STANDARD_TIER_MIN_STORAGE_BUFFER_SIZE;

        let tier = CapabilityTier::from_features_and_limits(&features, &limits);
        assert_eq!(tier, CapabilityTier::Standard);
    }

    #[test]
    fn standard_texture_boundary_one_below() {
        let features = Features::empty();
        let mut limits = Limits::default();
        limits.max_texture_dimension_2d = STANDARD_TIER_MIN_TEXTURE_2D - 1; // 8191
        limits.max_storage_buffer_binding_size = STANDARD_TIER_MIN_STORAGE_BUFFER_SIZE;

        let tier = CapabilityTier::from_features_and_limits(&features, &limits);
        assert_eq!(tier, CapabilityTier::Minimal);
    }

    #[test]
    fn standard_storage_boundary_exact() {
        let features = Features::empty();
        let mut limits = Limits::default();
        limits.max_texture_dimension_2d = STANDARD_TIER_MIN_TEXTURE_2D;
        limits.max_storage_buffer_binding_size = STANDARD_TIER_MIN_STORAGE_BUFFER_SIZE; // Exactly 128MB

        let tier = CapabilityTier::from_features_and_limits(&features, &limits);
        assert_eq!(tier, CapabilityTier::Standard);
    }

    #[test]
    fn standard_storage_boundary_one_below() {
        let features = Features::empty();
        let mut limits = Limits::default();
        limits.max_texture_dimension_2d = STANDARD_TIER_MIN_TEXTURE_2D;
        limits.max_storage_buffer_binding_size = STANDARD_TIER_MIN_STORAGE_BUFFER_SIZE - 1;

        let tier = CapabilityTier::from_features_and_limits(&features, &limits);
        assert_eq!(tier, CapabilityTier::Minimal);
    }

    // ------------------------------------------------------------------------
    // Minimal Tier Detection (fallback)
    // ------------------------------------------------------------------------

    #[test]
    fn detects_minimal_tier_with_no_features() {
        let features = Features::empty();
        let limits = minimal_limits();

        let tier = CapabilityTier::from_features_and_limits(&features, &limits);
        assert_eq!(tier, CapabilityTier::Minimal);
    }

    #[test]
    fn detects_minimal_tier_with_webgl2_defaults() {
        let features = Features::empty();
        let limits = Limits::downlevel_webgl2_defaults();

        let tier = CapabilityTier::from_features_and_limits(&features, &limits);
        assert_eq!(tier, CapabilityTier::Minimal);
    }

    #[test]
    fn minimal_when_both_standard_requirements_missing() {
        let features = Features::empty();
        let mut limits = Limits::default();
        limits.max_texture_dimension_2d = 4096; // Below 8K
        limits.max_storage_buffer_binding_size = 64 * 1024 * 1024; // Below 128MB

        let tier = CapabilityTier::from_features_and_limits(&features, &limits);
        assert_eq!(tier, CapabilityTier::Minimal);
    }
}

// ============================================================================
// 4. tier_name() Tests
// ============================================================================

mod tier_name {
    use super::*;

    #[test]
    fn full_tier_name_is_full() {
        assert_eq!(CapabilityTier::Full.tier_name(), "Full");
    }

    #[test]
    fn advanced_tier_name_is_advanced() {
        assert_eq!(CapabilityTier::Advanced.tier_name(), "Advanced");
    }

    #[test]
    fn standard_tier_name_is_standard() {
        assert_eq!(CapabilityTier::Standard.tier_name(), "Standard");
    }

    #[test]
    fn minimal_tier_name_is_minimal() {
        assert_eq!(CapabilityTier::Minimal.tier_name(), "Minimal");
    }

    #[test]
    fn tier_names_are_static_str() {
        // Verify that tier_name returns a &'static str (compile-time check)
        let _: &'static str = CapabilityTier::Full.tier_name();
        let _: &'static str = CapabilityTier::Advanced.tier_name();
        let _: &'static str = CapabilityTier::Standard.tier_name();
        let _: &'static str = CapabilityTier::Minimal.tier_name();
    }

    #[test]
    fn tier_names_are_unique() {
        let names: HashSet<&str> = [
            CapabilityTier::Full.tier_name(),
            CapabilityTier::Advanced.tier_name(),
            CapabilityTier::Standard.tier_name(),
            CapabilityTier::Minimal.tier_name(),
        ]
        .into_iter()
        .collect();

        assert_eq!(names.len(), 4, "All tier names should be unique");
    }
}

// ============================================================================
// 5. Capability Query Helper Tests
// ============================================================================

mod capability_queries {
    use super::*;

    // ------------------------------------------------------------------------
    // supports_ray_tracing()
    // ------------------------------------------------------------------------

    #[test]
    fn supports_ray_tracing_full_only() {
        assert!(CapabilityTier::Full.supports_ray_tracing());
        assert!(!CapabilityTier::Advanced.supports_ray_tracing());
        assert!(!CapabilityTier::Standard.supports_ray_tracing());
        assert!(!CapabilityTier::Minimal.supports_ray_tracing());
    }

    // ------------------------------------------------------------------------
    // supports_bindless()
    // ------------------------------------------------------------------------

    #[test]
    fn supports_bindless_advanced_and_full() {
        assert!(CapabilityTier::Full.supports_bindless());
        assert!(CapabilityTier::Advanced.supports_bindless());
        assert!(!CapabilityTier::Standard.supports_bindless());
        assert!(!CapabilityTier::Minimal.supports_bindless());
    }

    // ------------------------------------------------------------------------
    // supports_gpu_driven()
    // ------------------------------------------------------------------------

    #[test]
    fn supports_gpu_driven_advanced_and_full() {
        assert!(CapabilityTier::Full.supports_gpu_driven());
        assert!(CapabilityTier::Advanced.supports_gpu_driven());
        assert!(!CapabilityTier::Standard.supports_gpu_driven());
        assert!(!CapabilityTier::Minimal.supports_gpu_driven());
    }

    // ------------------------------------------------------------------------
    // supports_8k_textures()
    // ------------------------------------------------------------------------

    #[test]
    fn supports_8k_textures_all_except_minimal() {
        assert!(CapabilityTier::Full.supports_8k_textures());
        assert!(CapabilityTier::Advanced.supports_8k_textures());
        assert!(CapabilityTier::Standard.supports_8k_textures());
        assert!(!CapabilityTier::Minimal.supports_8k_textures());
    }

    // ------------------------------------------------------------------------
    // supports_compute()
    // ------------------------------------------------------------------------

    #[test]
    fn supports_compute_all_except_minimal() {
        assert!(CapabilityTier::Full.supports_compute());
        assert!(CapabilityTier::Advanced.supports_compute());
        assert!(CapabilityTier::Standard.supports_compute());
        assert!(!CapabilityTier::Minimal.supports_compute());
    }

    // ------------------------------------------------------------------------
    // Capability consistency with tier ordering
    // ------------------------------------------------------------------------

    #[test]
    fn higher_tier_has_superset_of_capabilities() {
        // Full has all capabilities
        assert!(CapabilityTier::Full.supports_ray_tracing());
        assert!(CapabilityTier::Full.supports_bindless());
        assert!(CapabilityTier::Full.supports_gpu_driven());
        assert!(CapabilityTier::Full.supports_8k_textures());
        assert!(CapabilityTier::Full.supports_compute());

        // Advanced has bindless, gpu_driven, 8k, compute (not RT)
        assert!(CapabilityTier::Advanced.supports_bindless());
        assert!(CapabilityTier::Advanced.supports_gpu_driven());
        assert!(CapabilityTier::Advanced.supports_8k_textures());
        assert!(CapabilityTier::Advanced.supports_compute());

        // Standard has 8k, compute
        assert!(CapabilityTier::Standard.supports_8k_textures());
        assert!(CapabilityTier::Standard.supports_compute());

        // Minimal has nothing
        assert!(!CapabilityTier::Minimal.supports_ray_tracing());
        assert!(!CapabilityTier::Minimal.supports_bindless());
        assert!(!CapabilityTier::Minimal.supports_gpu_driven());
        assert!(!CapabilityTier::Minimal.supports_8k_textures());
        assert!(!CapabilityTier::Minimal.supports_compute());
    }
}

// ============================================================================
// 6. meets_requirement() Tests
// ============================================================================

mod meets_requirement {
    use super::*;

    #[test]
    fn full_meets_all_requirements() {
        let full = CapabilityTier::Full;
        assert!(full.meets_requirement(CapabilityTier::Minimal));
        assert!(full.meets_requirement(CapabilityTier::Standard));
        assert!(full.meets_requirement(CapabilityTier::Advanced));
        assert!(full.meets_requirement(CapabilityTier::Full));
    }

    #[test]
    fn advanced_meets_advanced_and_below() {
        let advanced = CapabilityTier::Advanced;
        assert!(advanced.meets_requirement(CapabilityTier::Minimal));
        assert!(advanced.meets_requirement(CapabilityTier::Standard));
        assert!(advanced.meets_requirement(CapabilityTier::Advanced));
        assert!(!advanced.meets_requirement(CapabilityTier::Full));
    }

    #[test]
    fn standard_meets_standard_and_below() {
        let standard = CapabilityTier::Standard;
        assert!(standard.meets_requirement(CapabilityTier::Minimal));
        assert!(standard.meets_requirement(CapabilityTier::Standard));
        assert!(!standard.meets_requirement(CapabilityTier::Advanced));
        assert!(!standard.meets_requirement(CapabilityTier::Full));
    }

    #[test]
    fn minimal_meets_only_minimal() {
        let minimal = CapabilityTier::Minimal;
        assert!(minimal.meets_requirement(CapabilityTier::Minimal));
        assert!(!minimal.meets_requirement(CapabilityTier::Standard));
        assert!(!minimal.meets_requirement(CapabilityTier::Advanced));
        assert!(!minimal.meets_requirement(CapabilityTier::Full));
    }

    #[test]
    fn meets_requirement_is_same_as_ge_comparison() {
        let tiers = [
            CapabilityTier::Minimal,
            CapabilityTier::Standard,
            CapabilityTier::Advanced,
            CapabilityTier::Full,
        ];

        for actual in &tiers {
            for required in &tiers {
                assert_eq!(
                    actual.meets_requirement(*required),
                    *actual >= *required,
                    "meets_requirement should match >= operator"
                );
            }
        }
    }
}

// ============================================================================
// 7. Edge Cases Tests
// ============================================================================

mod edge_cases {
    use super::*;

    #[test]
    fn zero_limits_results_in_minimal() {
        let features = Features::empty();
        let mut limits = Limits::default();
        limits.max_texture_dimension_2d = 0;
        limits.max_storage_buffer_binding_size = 0;
        limits.max_compute_invocations_per_workgroup = 0;

        let tier = CapabilityTier::from_features_and_limits(&features, &limits);
        assert_eq!(tier, CapabilityTier::Minimal);
    }

    #[test]
    fn max_limits_without_features_is_standard() {
        let features = Features::empty();
        let mut limits = Limits::default();
        limits.max_texture_dimension_2d = u32::MAX;
        limits.max_storage_buffer_binding_size = u32::MAX;
        limits.max_compute_invocations_per_workgroup = u32::MAX;

        let tier = CapabilityTier::from_features_and_limits(&features, &limits);
        // Without Advanced features, falls to Standard (if 8K+ texture)
        assert_eq!(tier, CapabilityTier::Standard);
    }

    #[test]
    fn max_limits_with_advanced_features_is_advanced() {
        let features = Features::TEXTURE_BINDING_ARRAY | Features::MULTI_DRAW_INDIRECT_COUNT;
        let mut limits = Limits::default();
        limits.max_texture_dimension_2d = u32::MAX;
        limits.max_storage_buffer_binding_size = u32::MAX;
        limits.max_compute_invocations_per_workgroup = u32::MAX;

        let tier = CapabilityTier::from_features_and_limits(&features, &limits);
        assert_eq!(tier, CapabilityTier::Advanced);
    }

    #[test]
    fn partial_advanced_features_missing_bindless() {
        let features = Features::MULTI_DRAW_INDIRECT_COUNT;
        let limits = advanced_limits();

        let tier = CapabilityTier::from_features_and_limits(&features, &limits);
        assert_ne!(tier, CapabilityTier::Advanced);
    }

    #[test]
    fn partial_advanced_features_missing_multi_draw() {
        let features = Features::TEXTURE_BINDING_ARRAY;
        let limits = advanced_limits();

        let tier = CapabilityTier::from_features_and_limits(&features, &limits);
        assert_ne!(tier, CapabilityTier::Advanced);
    }

    #[test]
    fn partial_standard_limits_missing_texture() {
        let features = Features::empty();
        let mut limits = Limits::default();
        limits.max_texture_dimension_2d = 4096;
        limits.max_storage_buffer_binding_size = STANDARD_TIER_MIN_STORAGE_BUFFER_SIZE;

        let tier = CapabilityTier::from_features_and_limits(&features, &limits);
        assert_eq!(tier, CapabilityTier::Minimal);
    }

    #[test]
    fn partial_standard_limits_missing_storage() {
        let features = Features::empty();
        let mut limits = Limits::default();
        limits.max_texture_dimension_2d = STANDARD_TIER_MIN_TEXTURE_2D;
        limits.max_storage_buffer_binding_size = 64 * 1024 * 1024;

        let tier = CapabilityTier::from_features_and_limits(&features, &limits);
        assert_eq!(tier, CapabilityTier::Minimal);
    }

    #[test]
    fn advanced_features_with_minimal_limits_falls_to_minimal() {
        // Has Advanced features but limits are minimal
        let features = Features::TEXTURE_BINDING_ARRAY | Features::MULTI_DRAW_INDIRECT_COUNT;
        let limits = minimal_limits();

        let tier = CapabilityTier::from_features_and_limits(&features, &limits);
        // Should fall to Minimal because workgroup is too small
        assert_eq!(tier, CapabilityTier::Minimal);
    }

    #[test]
    fn rt_overrides_even_minimal_limits() {
        // RT present but limits are minimal
        let features = Features::RAY_TRACING_ACCELERATION_STRUCTURE;
        let limits = minimal_limits();

        let tier = CapabilityTier::from_features_and_limits(&features, &limits);
        assert_eq!(tier, CapabilityTier::Full);
    }

    #[test]
    fn empty_features_object() {
        let features = Features::empty();
        let limits = standard_limits();

        let tier = CapabilityTier::from_features_and_limits(&features, &limits);
        assert_eq!(tier, CapabilityTier::Standard);
    }
}

// ============================================================================
// 8. Trait Implementations Tests
// ============================================================================

mod trait_implementations {
    use super::*;

    // ------------------------------------------------------------------------
    // Debug
    // ------------------------------------------------------------------------

    #[test]
    fn debug_is_implemented() {
        let full = CapabilityTier::Full;
        let debug_str = format!("{:?}", full);
        assert!(debug_str.contains("Full"));
    }

    #[test]
    fn debug_for_all_variants() {
        assert!(!format!("{:?}", CapabilityTier::Minimal).is_empty());
        assert!(!format!("{:?}", CapabilityTier::Standard).is_empty());
        assert!(!format!("{:?}", CapabilityTier::Advanced).is_empty());
        assert!(!format!("{:?}", CapabilityTier::Full).is_empty());
    }

    // ------------------------------------------------------------------------
    // Clone
    // ------------------------------------------------------------------------

    #[test]
    fn clone_is_equal_to_original() {
        let original = CapabilityTier::Advanced;
        let cloned = original.clone();
        assert_eq!(original, cloned);
    }

    #[test]
    fn clone_all_variants() {
        let variants = [
            CapabilityTier::Minimal,
            CapabilityTier::Standard,
            CapabilityTier::Advanced,
            CapabilityTier::Full,
        ];

        for tier in &variants {
            let cloned = tier.clone();
            assert_eq!(*tier, cloned);
        }
    }

    // ------------------------------------------------------------------------
    // Copy
    // ------------------------------------------------------------------------

    #[test]
    fn copy_is_implicit() {
        let tier = CapabilityTier::Standard;
        let copied = tier; // Copy trait allows this
        assert_eq!(tier, copied);

        // Original still usable (Copy, not moved)
        assert_eq!(tier.tier_name(), "Standard");
    }

    // ------------------------------------------------------------------------
    // Eq
    // ------------------------------------------------------------------------

    #[test]
    fn eq_is_reflexive() {
        assert!(CapabilityTier::Full == CapabilityTier::Full);
        assert!(CapabilityTier::Advanced == CapabilityTier::Advanced);
        assert!(CapabilityTier::Standard == CapabilityTier::Standard);
        assert!(CapabilityTier::Minimal == CapabilityTier::Minimal);
    }

    #[test]
    fn eq_is_symmetric() {
        let a = CapabilityTier::Standard;
        let b = CapabilityTier::Standard;
        assert!(a == b && b == a);
    }

    #[test]
    fn ne_for_different_variants() {
        assert!(CapabilityTier::Full != CapabilityTier::Advanced);
        assert!(CapabilityTier::Full != CapabilityTier::Standard);
        assert!(CapabilityTier::Full != CapabilityTier::Minimal);
    }

    // ------------------------------------------------------------------------
    // Hash
    // ------------------------------------------------------------------------

    #[test]
    fn hash_is_consistent() {
        use std::hash::{Hash, Hasher};
        use std::collections::hash_map::DefaultHasher;

        fn hash_tier(tier: &CapabilityTier) -> u64 {
            let mut hasher = DefaultHasher::new();
            tier.hash(&mut hasher);
            hasher.finish()
        }

        let tier = CapabilityTier::Advanced;
        let hash1 = hash_tier(&tier);
        let hash2 = hash_tier(&tier);
        assert_eq!(hash1, hash2, "Same value should produce same hash");
    }

    #[test]
    fn hash_usable_in_hashset() {
        let mut set: HashSet<CapabilityTier> = HashSet::new();
        set.insert(CapabilityTier::Minimal);
        set.insert(CapabilityTier::Standard);
        set.insert(CapabilityTier::Advanced);
        set.insert(CapabilityTier::Full);

        assert_eq!(set.len(), 4);
        assert!(set.contains(&CapabilityTier::Full));
    }

    #[test]
    fn hash_usable_in_hashmap() {
        let mut map: HashMap<CapabilityTier, &str> = HashMap::new();
        map.insert(CapabilityTier::Full, "ray tracing");
        map.insert(CapabilityTier::Advanced, "bindless");
        map.insert(CapabilityTier::Standard, "compute");
        map.insert(CapabilityTier::Minimal, "basic");

        assert_eq!(map.get(&CapabilityTier::Full), Some(&"ray tracing"));
        assert_eq!(map.get(&CapabilityTier::Minimal), Some(&"basic"));
    }

    // ------------------------------------------------------------------------
    // Default
    // ------------------------------------------------------------------------

    #[test]
    fn default_returns_minimal() {
        let default_tier: CapabilityTier = Default::default();
        assert_eq!(default_tier, CapabilityTier::Minimal);
    }

    // ------------------------------------------------------------------------
    // Display
    // ------------------------------------------------------------------------

    #[test]
    fn display_matches_tier_name() {
        assert_eq!(format!("{}", CapabilityTier::Full), "Full");
        assert_eq!(format!("{}", CapabilityTier::Advanced), "Advanced");
        assert_eq!(format!("{}", CapabilityTier::Standard), "Standard");
        assert_eq!(format!("{}", CapabilityTier::Minimal), "Minimal");
    }

    #[test]
    fn display_in_format_string() {
        let tier = CapabilityTier::Advanced;
        let formatted = format!("Current tier: {}", tier);
        assert!(formatted.contains("Advanced"));
    }
}

// ============================================================================
// 9. CapabilityReport Tests
// ============================================================================

mod capability_report {
    use super::*;

    fn make_test_report(tier: CapabilityTier) -> CapabilityReport {
        CapabilityReport {
            tier,
            has_ray_tracing: tier == CapabilityTier::Full,
            has_bindless: matches!(tier, CapabilityTier::Full | CapabilityTier::Advanced),
            has_multi_draw_indirect_count: matches!(
                tier,
                CapabilityTier::Full | CapabilityTier::Advanced
            ),
            has_storage_binding_array: false,
            max_texture_dimension_2d: 16384,
            max_compute_invocations: 1024,
            max_storage_buffer_binding_size: 256 * 1024 * 1024,
            adapter_name: "Test GPU".to_string(),
        }
    }

    #[test]
    fn report_stores_tier_correctly() {
        let report = make_test_report(CapabilityTier::Advanced);
        assert_eq!(report.tier, CapabilityTier::Advanced);
    }

    #[test]
    fn report_stores_features_correctly() {
        let report = make_test_report(CapabilityTier::Full);
        assert!(report.has_ray_tracing);
        assert!(report.has_bindless);
        assert!(report.has_multi_draw_indirect_count);
    }

    #[test]
    fn report_stores_limits_correctly() {
        let report = make_test_report(CapabilityTier::Standard);
        assert_eq!(report.max_texture_dimension_2d, 16384);
        assert_eq!(report.max_compute_invocations, 1024);
        assert_eq!(report.max_storage_buffer_binding_size, 256 * 1024 * 1024);
    }

    #[test]
    fn report_stores_adapter_name() {
        let report = make_test_report(CapabilityTier::Standard);
        assert_eq!(report.adapter_name, "Test GPU");
    }

    #[test]
    fn report_is_cloneable() {
        let report = make_test_report(CapabilityTier::Advanced);
        let cloned = report.clone();
        assert_eq!(cloned.tier, report.tier);
        assert_eq!(cloned.adapter_name, report.adapter_name);
    }

    #[test]
    fn report_display_contains_tier() {
        let report = make_test_report(CapabilityTier::Advanced);
        let display = format!("{}", report);
        assert!(display.contains("Advanced"));
    }

    #[test]
    fn report_display_contains_adapter_name() {
        let report = make_test_report(CapabilityTier::Standard);
        let display = format!("{}", report);
        assert!(display.contains("Test GPU"));
    }

    #[test]
    fn report_display_contains_limits() {
        let report = make_test_report(CapabilityTier::Standard);
        let display = format!("{}", report);
        assert!(display.contains("16384"));
        assert!(display.contains("1024"));
    }

    #[test]
    fn report_debug_works() {
        let report = make_test_report(CapabilityTier::Full);
        let debug = format!("{:?}", report);
        assert!(!debug.is_empty());
    }
}

// ============================================================================
// 10. Utility Functions Tests
// ============================================================================

mod utility_functions {
    use super::*;

    // ------------------------------------------------------------------------
    // features_for_tier()
    // ------------------------------------------------------------------------

    #[test]
    fn features_for_full_tier_includes_rt() {
        let features = features_for_tier(CapabilityTier::Full);
        assert!(features.contains(Features::RAY_TRACING_ACCELERATION_STRUCTURE));
    }

    #[test]
    fn features_for_advanced_tier_includes_bindless_and_multi_draw() {
        let features = features_for_tier(CapabilityTier::Advanced);
        assert!(features.contains(Features::TEXTURE_BINDING_ARRAY));
        assert!(features.contains(Features::MULTI_DRAW_INDIRECT_COUNT));
    }

    #[test]
    fn features_for_standard_tier_is_empty() {
        let features = features_for_tier(CapabilityTier::Standard);
        assert_eq!(features, Features::empty());
    }

    #[test]
    fn features_for_minimal_tier_is_empty() {
        let features = features_for_tier(CapabilityTier::Minimal);
        assert_eq!(features, Features::empty());
    }

    // ------------------------------------------------------------------------
    // can_achieve_tier()
    // ------------------------------------------------------------------------

    #[test]
    fn can_achieve_minimal_with_any_config() {
        let features = Features::empty();
        let limits = minimal_limits();
        assert!(can_achieve_tier(&features, &limits, CapabilityTier::Minimal));
    }

    #[test]
    fn can_achieve_standard_with_standard_limits() {
        let features = Features::empty();
        let limits = standard_limits();
        assert!(can_achieve_tier(&features, &limits, CapabilityTier::Standard));
        assert!(can_achieve_tier(&features, &limits, CapabilityTier::Minimal));
    }

    #[test]
    fn cannot_achieve_standard_with_minimal_limits() {
        let features = Features::empty();
        let limits = minimal_limits();
        assert!(!can_achieve_tier(
            &features,
            &limits,
            CapabilityTier::Standard
        ));
    }

    #[test]
    fn can_achieve_advanced_with_advanced_config() {
        let features = Features::TEXTURE_BINDING_ARRAY | Features::MULTI_DRAW_INDIRECT_COUNT;
        let limits = advanced_limits();
        assert!(can_achieve_tier(&features, &limits, CapabilityTier::Advanced));
        assert!(can_achieve_tier(&features, &limits, CapabilityTier::Standard));
        assert!(can_achieve_tier(&features, &limits, CapabilityTier::Minimal));
    }

    #[test]
    fn cannot_achieve_advanced_without_features() {
        let features = Features::empty();
        let limits = advanced_limits();
        assert!(!can_achieve_tier(
            &features,
            &limits,
            CapabilityTier::Advanced
        ));
    }

    #[test]
    fn can_achieve_full_with_rt() {
        let features = Features::RAY_TRACING_ACCELERATION_STRUCTURE;
        let limits = Limits::default();
        assert!(can_achieve_tier(&features, &limits, CapabilityTier::Full));
        assert!(can_achieve_tier(&features, &limits, CapabilityTier::Advanced));
        assert!(can_achieve_tier(&features, &limits, CapabilityTier::Standard));
        assert!(can_achieve_tier(&features, &limits, CapabilityTier::Minimal));
    }

    #[test]
    fn cannot_achieve_full_without_rt() {
        let features = Features::TEXTURE_BINDING_ARRAY | Features::MULTI_DRAW_INDIRECT_COUNT;
        let limits = advanced_limits();
        assert!(!can_achieve_tier(&features, &limits, CapabilityTier::Full));
    }
}

// ============================================================================
// 11. Description Tests
// ============================================================================

mod description {
    use super::*;

    #[test]
    fn full_description_mentions_ray_tracing() {
        let desc = CapabilityTier::Full.description();
        assert!(
            desc.to_lowercase().contains("ray tracing")
                || desc.to_lowercase().contains("ray-tracing")
        );
    }

    #[test]
    fn advanced_description_mentions_bindless() {
        let desc = CapabilityTier::Advanced.description();
        assert!(
            desc.to_lowercase().contains("bindless")
                || desc.to_lowercase().contains("gpu-driven")
        );
    }

    #[test]
    fn standard_description_mentions_textures_or_compute() {
        let desc = CapabilityTier::Standard.description();
        assert!(
            desc.to_lowercase().contains("8k")
                || desc.to_lowercase().contains("texture")
                || desc.to_lowercase().contains("compute")
        );
    }

    #[test]
    fn minimal_description_mentions_fallback_or_basic() {
        let desc = CapabilityTier::Minimal.description();
        assert!(
            desc.to_lowercase().contains("minimal")
                || desc.to_lowercase().contains("basic")
                || desc.to_lowercase().contains("fallback")
        );
    }

    #[test]
    fn all_descriptions_are_non_empty() {
        assert!(!CapabilityTier::Full.description().is_empty());
        assert!(!CapabilityTier::Advanced.description().is_empty());
        assert!(!CapabilityTier::Standard.description().is_empty());
        assert!(!CapabilityTier::Minimal.description().is_empty());
    }

    #[test]
    fn descriptions_are_static_str() {
        let _: &'static str = CapabilityTier::Full.description();
        let _: &'static str = CapabilityTier::Advanced.description();
        let _: &'static str = CapabilityTier::Standard.description();
        let _: &'static str = CapabilityTier::Minimal.description();
    }
}

// ============================================================================
// 12. Boundary Value Tests
// ============================================================================

mod boundary_values {
    use super::*;

    #[test]
    fn texture_dimension_exact_8192_is_standard() {
        let features = Features::empty();
        let mut limits = Limits::default();
        limits.max_texture_dimension_2d = 8192;
        limits.max_storage_buffer_binding_size = 128 * 1024 * 1024;

        let tier = CapabilityTier::from_features_and_limits(&features, &limits);
        assert_eq!(tier, CapabilityTier::Standard);
    }

    #[test]
    fn texture_dimension_8191_is_minimal() {
        let features = Features::empty();
        let mut limits = Limits::default();
        limits.max_texture_dimension_2d = 8191;
        limits.max_storage_buffer_binding_size = 128 * 1024 * 1024;

        let tier = CapabilityTier::from_features_and_limits(&features, &limits);
        assert_eq!(tier, CapabilityTier::Minimal);
    }

    #[test]
    fn texture_dimension_8193_is_standard() {
        let features = Features::empty();
        let mut limits = Limits::default();
        limits.max_texture_dimension_2d = 8193;
        limits.max_storage_buffer_binding_size = 128 * 1024 * 1024;

        let tier = CapabilityTier::from_features_and_limits(&features, &limits);
        assert_eq!(tier, CapabilityTier::Standard);
    }

    #[test]
    fn storage_buffer_exact_128mb_is_standard() {
        let features = Features::empty();
        let mut limits = Limits::default();
        limits.max_texture_dimension_2d = 8192;
        limits.max_storage_buffer_binding_size = 128 * 1024 * 1024;

        let tier = CapabilityTier::from_features_and_limits(&features, &limits);
        assert_eq!(tier, CapabilityTier::Standard);
    }

    #[test]
    fn storage_buffer_one_below_128mb_is_minimal() {
        let features = Features::empty();
        let mut limits = Limits::default();
        limits.max_texture_dimension_2d = 8192;
        limits.max_storage_buffer_binding_size = 128 * 1024 * 1024 - 1;

        let tier = CapabilityTier::from_features_and_limits(&features, &limits);
        assert_eq!(tier, CapabilityTier::Minimal);
    }

    #[test]
    fn workgroup_invocations_exact_1024_is_advanced() {
        let features = Features::TEXTURE_BINDING_ARRAY | Features::MULTI_DRAW_INDIRECT_COUNT;
        let mut limits = standard_limits();
        limits.max_compute_invocations_per_workgroup = 1024;

        let tier = CapabilityTier::from_features_and_limits(&features, &limits);
        assert_eq!(tier, CapabilityTier::Advanced);
    }

    #[test]
    fn workgroup_invocations_1023_is_not_advanced() {
        let features = Features::TEXTURE_BINDING_ARRAY | Features::MULTI_DRAW_INDIRECT_COUNT;
        let mut limits = standard_limits();
        limits.max_compute_invocations_per_workgroup = 1023;

        let tier = CapabilityTier::from_features_and_limits(&features, &limits);
        assert_ne!(tier, CapabilityTier::Advanced);
    }

    #[test]
    fn workgroup_invocations_1025_is_advanced() {
        let features = Features::TEXTURE_BINDING_ARRAY | Features::MULTI_DRAW_INDIRECT_COUNT;
        let mut limits = standard_limits();
        limits.max_compute_invocations_per_workgroup = 1025;

        let tier = CapabilityTier::from_features_and_limits(&features, &limits);
        assert_eq!(tier, CapabilityTier::Advanced);
    }
}

// ============================================================================
// 13. Feature Combination Matrix Tests
// ============================================================================

mod feature_combinations {
    use super::*;

    #[test]
    fn rt_alone() {
        let features = Features::RAY_TRACING_ACCELERATION_STRUCTURE;
        let limits = minimal_limits();
        let tier = CapabilityTier::from_features_and_limits(&features, &limits);
        assert_eq!(tier, CapabilityTier::Full);
    }

    #[test]
    fn rt_plus_bindless() {
        let features =
            Features::RAY_TRACING_ACCELERATION_STRUCTURE | Features::TEXTURE_BINDING_ARRAY;
        let limits = minimal_limits();
        let tier = CapabilityTier::from_features_and_limits(&features, &limits);
        assert_eq!(tier, CapabilityTier::Full);
    }

    #[test]
    fn rt_plus_multi_draw() {
        let features =
            Features::RAY_TRACING_ACCELERATION_STRUCTURE | Features::MULTI_DRAW_INDIRECT_COUNT;
        let limits = minimal_limits();
        let tier = CapabilityTier::from_features_and_limits(&features, &limits);
        assert_eq!(tier, CapabilityTier::Full);
    }

    #[test]
    fn rt_plus_all_advanced_features() {
        let features = Features::RAY_TRACING_ACCELERATION_STRUCTURE
            | Features::TEXTURE_BINDING_ARRAY
            | Features::MULTI_DRAW_INDIRECT_COUNT;
        let limits = advanced_limits();
        let tier = CapabilityTier::from_features_and_limits(&features, &limits);
        assert_eq!(tier, CapabilityTier::Full);
    }

    #[test]
    fn bindless_only_with_adequate_limits() {
        let features = Features::TEXTURE_BINDING_ARRAY;
        let limits = advanced_limits();
        let tier = CapabilityTier::from_features_and_limits(&features, &limits);
        // Missing multi-draw, should fall to Standard
        assert_eq!(tier, CapabilityTier::Standard);
    }

    #[test]
    fn multi_draw_only_with_adequate_limits() {
        let features = Features::MULTI_DRAW_INDIRECT_COUNT;
        let limits = advanced_limits();
        let tier = CapabilityTier::from_features_and_limits(&features, &limits);
        // Missing bindless, should fall to Standard
        assert_eq!(tier, CapabilityTier::Standard);
    }

    #[test]
    fn both_advanced_features_but_small_workgroup() {
        let features = Features::TEXTURE_BINDING_ARRAY | Features::MULTI_DRAW_INDIRECT_COUNT;
        let mut limits = standard_limits();
        limits.max_compute_invocations_per_workgroup = 256;
        let tier = CapabilityTier::from_features_and_limits(&features, &limits);
        // Missing workgroup size, should fall to Standard
        assert_eq!(tier, CapabilityTier::Standard);
    }

    #[test]
    fn no_features_with_standard_limits() {
        let features = Features::empty();
        let limits = standard_limits();
        let tier = CapabilityTier::from_features_and_limits(&features, &limits);
        assert_eq!(tier, CapabilityTier::Standard);
    }

    #[test]
    fn no_features_with_advanced_limits() {
        let features = Features::empty();
        let limits = advanced_limits();
        let tier = CapabilityTier::from_features_and_limits(&features, &limits);
        // No features, so Standard (8K textures + compute available)
        assert_eq!(tier, CapabilityTier::Standard);
    }

    #[test]
    fn advanced_features_with_minimal_limits() {
        let features = Features::TEXTURE_BINDING_ARRAY | Features::MULTI_DRAW_INDIRECT_COUNT;
        let limits = minimal_limits();
        let tier = CapabilityTier::from_features_and_limits(&features, &limits);
        // Has features but limits are too low
        assert_eq!(tier, CapabilityTier::Minimal);
    }
}
