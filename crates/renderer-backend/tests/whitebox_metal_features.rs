//! Whitebox tests for Metal-specific feature detection.
//!
//! This module provides comprehensive testing of the Metal GPU family detection,
//! Apple Silicon generation identification, and Metal feature capabilities.
//!
//! Tests cover:
//! - MetalGpuFamily enum variants and detection
//! - AppleSiliconGeneration enum variants and parsing
//! - MetalFeatures struct initialization and feature detection
//! - Edge cases including unknown devices, empty strings, mixed case
//!
//! Target: 150+ tests minimum

use renderer_backend::backend::metal::{
    AppleSiliconGeneration, MetalFeatures, MetalGpuFamily,
};
use wgpu::Features;

// ============================================================================
// Module: MetalGpuFamily Tests
// ============================================================================

mod metal_gpu_family {
    use super::*;

    // ========================================================================
    // Default and Enum Variants
    // ========================================================================

    #[test]
    fn test_default_is_unknown() {
        let family = MetalGpuFamily::default();
        assert_eq!(family, MetalGpuFamily::Unknown);
    }

    #[test]
    fn test_unknown_variant_exists() {
        let _unknown = MetalGpuFamily::Unknown;
    }

    #[test]
    fn test_all_apple_variants_exist() {
        let variants = [
            MetalGpuFamily::Apple1,
            MetalGpuFamily::Apple2,
            MetalGpuFamily::Apple3,
            MetalGpuFamily::Apple4,
            MetalGpuFamily::Apple5,
            MetalGpuFamily::Apple6,
            MetalGpuFamily::Apple7,
            MetalGpuFamily::Apple8,
            MetalGpuFamily::Apple9,
        ];
        assert_eq!(variants.len(), 9);
    }

    #[test]
    fn test_mac_variants_exist() {
        let variants = [MetalGpuFamily::Mac1, MetalGpuFamily::Mac2];
        assert_eq!(variants.len(), 2);
    }

    #[test]
    fn test_common_variants_exist() {
        let variants = [
            MetalGpuFamily::Common1,
            MetalGpuFamily::Common2,
            MetalGpuFamily::Common3,
        ];
        assert_eq!(variants.len(), 3);
    }

    #[test]
    fn test_metal3_variant_exists() {
        let _metal3 = MetalGpuFamily::Metal3;
    }

    // ========================================================================
    // from_device_name: M-series Chips
    // ========================================================================

    #[test]
    fn test_from_device_name_m1_base() {
        assert_eq!(
            MetalGpuFamily::from_device_name("Apple M1"),
            MetalGpuFamily::Apple7
        );
    }

    #[test]
    fn test_from_device_name_m1_pro() {
        assert_eq!(
            MetalGpuFamily::from_device_name("Apple M1 Pro"),
            MetalGpuFamily::Apple7
        );
    }

    #[test]
    fn test_from_device_name_m1_max() {
        assert_eq!(
            MetalGpuFamily::from_device_name("Apple M1 Max"),
            MetalGpuFamily::Apple7
        );
    }

    #[test]
    fn test_from_device_name_m1_ultra() {
        assert_eq!(
            MetalGpuFamily::from_device_name("Apple M1 Ultra"),
            MetalGpuFamily::Apple7
        );
    }

    #[test]
    fn test_from_device_name_m2_base() {
        assert_eq!(
            MetalGpuFamily::from_device_name("Apple M2"),
            MetalGpuFamily::Apple8
        );
    }

    #[test]
    fn test_from_device_name_m2_pro() {
        assert_eq!(
            MetalGpuFamily::from_device_name("Apple M2 Pro"),
            MetalGpuFamily::Apple8
        );
    }

    #[test]
    fn test_from_device_name_m2_max() {
        assert_eq!(
            MetalGpuFamily::from_device_name("Apple M2 Max"),
            MetalGpuFamily::Apple8
        );
    }

    #[test]
    fn test_from_device_name_m2_ultra() {
        assert_eq!(
            MetalGpuFamily::from_device_name("Apple M2 Ultra"),
            MetalGpuFamily::Apple8
        );
    }

    #[test]
    fn test_from_device_name_m3_base() {
        assert_eq!(
            MetalGpuFamily::from_device_name("Apple M3"),
            MetalGpuFamily::Apple9
        );
    }

    #[test]
    fn test_from_device_name_m3_pro() {
        assert_eq!(
            MetalGpuFamily::from_device_name("Apple M3 Pro"),
            MetalGpuFamily::Apple9
        );
    }

    #[test]
    fn test_from_device_name_m3_max() {
        assert_eq!(
            MetalGpuFamily::from_device_name("Apple M3 Max"),
            MetalGpuFamily::Apple9
        );
    }

    #[test]
    fn test_from_device_name_m4_base() {
        assert_eq!(
            MetalGpuFamily::from_device_name("Apple M4"),
            MetalGpuFamily::Apple9
        );
    }

    #[test]
    fn test_from_device_name_m4_pro() {
        assert_eq!(
            MetalGpuFamily::from_device_name("Apple M4 Pro"),
            MetalGpuFamily::Apple9
        );
    }

    #[test]
    fn test_from_device_name_m4_max() {
        assert_eq!(
            MetalGpuFamily::from_device_name("Apple M4 Max"),
            MetalGpuFamily::Apple9
        );
    }

    // ========================================================================
    // from_device_name: A-series Chips
    // ========================================================================

    #[test]
    fn test_from_device_name_a7() {
        assert_eq!(
            MetalGpuFamily::from_device_name("Apple A7"),
            MetalGpuFamily::Apple1
        );
    }

    #[test]
    fn test_from_device_name_a8() {
        assert_eq!(
            MetalGpuFamily::from_device_name("Apple A8"),
            MetalGpuFamily::Apple2
        );
    }

    #[test]
    fn test_from_device_name_a9() {
        assert_eq!(
            MetalGpuFamily::from_device_name("Apple A9"),
            MetalGpuFamily::Apple3
        );
    }

    #[test]
    fn test_from_device_name_a10() {
        assert_eq!(
            MetalGpuFamily::from_device_name("Apple A10"),
            MetalGpuFamily::Apple3
        );
    }

    #[test]
    fn test_from_device_name_a11() {
        assert_eq!(
            MetalGpuFamily::from_device_name("Apple A11 Bionic"),
            MetalGpuFamily::Apple4
        );
    }

    #[test]
    fn test_from_device_name_a12() {
        assert_eq!(
            MetalGpuFamily::from_device_name("Apple A12 Bionic"),
            MetalGpuFamily::Apple5
        );
    }

    #[test]
    fn test_from_device_name_a13() {
        assert_eq!(
            MetalGpuFamily::from_device_name("Apple A13 Bionic"),
            MetalGpuFamily::Apple6
        );
    }

    #[test]
    fn test_from_device_name_a14() {
        assert_eq!(
            MetalGpuFamily::from_device_name("Apple A14 Bionic"),
            MetalGpuFamily::Apple7
        );
    }

    #[test]
    fn test_from_device_name_a15() {
        assert_eq!(
            MetalGpuFamily::from_device_name("Apple A15 Bionic"),
            MetalGpuFamily::Apple8
        );
    }

    #[test]
    fn test_from_device_name_a16() {
        assert_eq!(
            MetalGpuFamily::from_device_name("Apple A16 Bionic"),
            MetalGpuFamily::Apple9
        );
    }

    #[test]
    fn test_from_device_name_a17_pro() {
        assert_eq!(
            MetalGpuFamily::from_device_name("Apple A17 Pro"),
            MetalGpuFamily::Apple9
        );
    }

    // ========================================================================
    // from_device_name: Intel Mac
    // ========================================================================

    #[test]
    fn test_from_device_name_intel_uhd() {
        assert_eq!(
            MetalGpuFamily::from_device_name("Intel UHD Graphics 630"),
            MetalGpuFamily::Mac2
        );
    }

    #[test]
    fn test_from_device_name_intel_iris() {
        assert_eq!(
            MetalGpuFamily::from_device_name("Intel Iris Pro Graphics"),
            MetalGpuFamily::Mac2
        );
    }

    #[test]
    fn test_from_device_name_intel_iris_plus() {
        assert_eq!(
            MetalGpuFamily::from_device_name("Intel Iris Plus Graphics"),
            MetalGpuFamily::Mac2
        );
    }

    #[test]
    fn test_from_device_name_intel_hd_old() {
        assert_eq!(
            MetalGpuFamily::from_device_name("Intel HD Graphics 4000"),
            MetalGpuFamily::Mac1
        );
    }

    #[test]
    fn test_from_device_name_intel_hd_5000() {
        assert_eq!(
            MetalGpuFamily::from_device_name("Intel HD Graphics 5000"),
            MetalGpuFamily::Mac1
        );
    }

    #[test]
    fn test_from_device_name_intel_generic() {
        assert_eq!(
            MetalGpuFamily::from_device_name("Intel Graphics"),
            MetalGpuFamily::Mac1
        );
    }

    // ========================================================================
    // from_device_name: AMD
    // ========================================================================

    #[test]
    fn test_from_device_name_amd_radeon() {
        assert_eq!(
            MetalGpuFamily::from_device_name("AMD Radeon Pro 5500M"),
            MetalGpuFamily::Mac2
        );
    }

    #[test]
    fn test_from_device_name_amd_radeon_rx() {
        assert_eq!(
            MetalGpuFamily::from_device_name("AMD Radeon RX 580"),
            MetalGpuFamily::Mac2
        );
    }

    #[test]
    fn test_from_device_name_radeon_only() {
        assert_eq!(
            MetalGpuFamily::from_device_name("Radeon Pro Vega 56"),
            MetalGpuFamily::Mac2
        );
    }

    #[test]
    fn test_from_device_name_amd_only() {
        assert_eq!(
            MetalGpuFamily::from_device_name("AMD W5700X"),
            MetalGpuFamily::Mac2
        );
    }

    // ========================================================================
    // from_device_name: Generic Apple
    // ========================================================================

    #[test]
    fn test_from_device_name_generic_apple() {
        assert_eq!(
            MetalGpuFamily::from_device_name("Apple GPU"),
            MetalGpuFamily::Apple7
        );
    }

    #[test]
    fn test_from_device_name_apple_only() {
        assert_eq!(
            MetalGpuFamily::from_device_name("Apple"),
            MetalGpuFamily::Apple7
        );
    }

    // ========================================================================
    // from_device_name: Case Insensitivity
    // ========================================================================

    #[test]
    fn test_from_device_name_lowercase_m3() {
        assert_eq!(
            MetalGpuFamily::from_device_name("apple m3 pro"),
            MetalGpuFamily::Apple9
        );
    }

    #[test]
    fn test_from_device_name_uppercase_m3() {
        assert_eq!(
            MetalGpuFamily::from_device_name("APPLE M3 PRO"),
            MetalGpuFamily::Apple9
        );
    }

    #[test]
    fn test_from_device_name_mixed_case_m2() {
        assert_eq!(
            MetalGpuFamily::from_device_name("aPpLe M2 mAx"),
            MetalGpuFamily::Apple8
        );
    }

    #[test]
    fn test_from_device_name_lowercase_intel() {
        assert_eq!(
            MetalGpuFamily::from_device_name("intel uhd graphics 630"),
            MetalGpuFamily::Mac2
        );
    }

    // ========================================================================
    // from_device_name: Edge Cases
    // ========================================================================

    #[test]
    fn test_from_device_name_empty_string() {
        assert_eq!(
            MetalGpuFamily::from_device_name(""),
            MetalGpuFamily::Unknown
        );
    }

    #[test]
    fn test_from_device_name_unknown_device() {
        assert_eq!(
            MetalGpuFamily::from_device_name("Unknown GPU"),
            MetalGpuFamily::Unknown
        );
    }

    #[test]
    fn test_from_device_name_nvidia() {
        // NVIDIA not supported on Metal
        assert_eq!(
            MetalGpuFamily::from_device_name("NVIDIA GeForce RTX 4090"),
            MetalGpuFamily::Unknown
        );
    }

    #[test]
    fn test_from_device_name_whitespace_only() {
        assert_eq!(
            MetalGpuFamily::from_device_name("   "),
            MetalGpuFamily::Unknown
        );
    }

    #[test]
    fn test_from_device_name_partial_m1_match() {
        // "m1" should still match
        assert_eq!(
            MetalGpuFamily::from_device_name("m1"),
            MetalGpuFamily::Apple7
        );
    }

    #[test]
    fn test_from_device_name_m1_in_longer_string() {
        assert_eq!(
            MetalGpuFamily::from_device_name("MacBook Pro (M1, 2020)"),
            MetalGpuFamily::Apple7
        );
    }

    // ========================================================================
    // is_apple_silicon
    // ========================================================================

    #[test]
    fn test_apple7_is_apple_silicon() {
        assert!(MetalGpuFamily::Apple7.is_apple_silicon());
    }

    #[test]
    fn test_apple8_is_apple_silicon() {
        assert!(MetalGpuFamily::Apple8.is_apple_silicon());
    }

    #[test]
    fn test_apple9_is_apple_silicon() {
        assert!(MetalGpuFamily::Apple9.is_apple_silicon());
    }

    #[test]
    fn test_metal3_is_apple_silicon() {
        assert!(MetalGpuFamily::Metal3.is_apple_silicon());
    }

    #[test]
    fn test_apple1_not_apple_silicon() {
        assert!(!MetalGpuFamily::Apple1.is_apple_silicon());
    }

    #[test]
    fn test_apple6_not_apple_silicon() {
        assert!(!MetalGpuFamily::Apple6.is_apple_silicon());
    }

    #[test]
    fn test_mac1_not_apple_silicon() {
        assert!(!MetalGpuFamily::Mac1.is_apple_silicon());
    }

    #[test]
    fn test_mac2_not_apple_silicon() {
        assert!(!MetalGpuFamily::Mac2.is_apple_silicon());
    }

    #[test]
    fn test_common1_not_apple_silicon() {
        assert!(!MetalGpuFamily::Common1.is_apple_silicon());
    }

    #[test]
    fn test_unknown_not_apple_silicon() {
        assert!(!MetalGpuFamily::Unknown.is_apple_silicon());
    }

    // ========================================================================
    // is_intel_mac
    // ========================================================================

    #[test]
    fn test_mac1_is_intel_mac() {
        assert!(MetalGpuFamily::Mac1.is_intel_mac());
    }

    #[test]
    fn test_mac2_is_intel_mac() {
        assert!(MetalGpuFamily::Mac2.is_intel_mac());
    }

    #[test]
    fn test_apple7_not_intel_mac() {
        assert!(!MetalGpuFamily::Apple7.is_intel_mac());
    }

    #[test]
    fn test_common1_not_intel_mac() {
        assert!(!MetalGpuFamily::Common1.is_intel_mac());
    }

    #[test]
    fn test_unknown_not_intel_mac() {
        assert!(!MetalGpuFamily::Unknown.is_intel_mac());
    }

    // ========================================================================
    // supports_metal3
    // ========================================================================

    #[test]
    fn test_apple7_supports_metal3() {
        assert!(MetalGpuFamily::Apple7.supports_metal3());
    }

    #[test]
    fn test_apple8_supports_metal3() {
        assert!(MetalGpuFamily::Apple8.supports_metal3());
    }

    #[test]
    fn test_apple9_supports_metal3() {
        assert!(MetalGpuFamily::Apple9.supports_metal3());
    }

    #[test]
    fn test_metal3_supports_metal3() {
        assert!(MetalGpuFamily::Metal3.supports_metal3());
    }

    #[test]
    fn test_apple6_not_supports_metal3() {
        assert!(!MetalGpuFamily::Apple6.supports_metal3());
    }

    #[test]
    fn test_mac2_not_supports_metal3() {
        assert!(!MetalGpuFamily::Mac2.supports_metal3());
    }

    #[test]
    fn test_common3_not_supports_metal3() {
        assert!(!MetalGpuFamily::Common3.supports_metal3());
    }

    // ========================================================================
    // supports_ray_tracing
    // ========================================================================

    #[test]
    fn test_apple7_supports_ray_tracing() {
        assert!(MetalGpuFamily::Apple7.supports_ray_tracing());
    }

    #[test]
    fn test_apple8_supports_ray_tracing() {
        assert!(MetalGpuFamily::Apple8.supports_ray_tracing());
    }

    #[test]
    fn test_apple9_supports_ray_tracing() {
        assert!(MetalGpuFamily::Apple9.supports_ray_tracing());
    }

    #[test]
    fn test_metal3_supports_ray_tracing() {
        assert!(MetalGpuFamily::Metal3.supports_ray_tracing());
    }

    #[test]
    fn test_apple6_not_supports_ray_tracing() {
        assert!(!MetalGpuFamily::Apple6.supports_ray_tracing());
    }

    #[test]
    fn test_mac2_not_supports_ray_tracing() {
        assert!(!MetalGpuFamily::Mac2.supports_ray_tracing());
    }

    // ========================================================================
    // supports_mesh_shaders
    // ========================================================================

    #[test]
    fn test_apple7_supports_mesh_shaders() {
        assert!(MetalGpuFamily::Apple7.supports_mesh_shaders());
    }

    #[test]
    fn test_apple9_supports_mesh_shaders() {
        assert!(MetalGpuFamily::Apple9.supports_mesh_shaders());
    }

    #[test]
    fn test_apple5_not_supports_mesh_shaders() {
        assert!(!MetalGpuFamily::Apple5.supports_mesh_shaders());
    }

    #[test]
    fn test_mac1_not_supports_mesh_shaders() {
        assert!(!MetalGpuFamily::Mac1.supports_mesh_shaders());
    }

    // ========================================================================
    // apple_version
    // ========================================================================

    #[test]
    fn test_apple_version_apple1() {
        assert_eq!(MetalGpuFamily::Apple1.apple_version(), 1);
    }

    #[test]
    fn test_apple_version_apple2() {
        assert_eq!(MetalGpuFamily::Apple2.apple_version(), 2);
    }

    #[test]
    fn test_apple_version_apple3() {
        assert_eq!(MetalGpuFamily::Apple3.apple_version(), 3);
    }

    #[test]
    fn test_apple_version_apple4() {
        assert_eq!(MetalGpuFamily::Apple4.apple_version(), 4);
    }

    #[test]
    fn test_apple_version_apple5() {
        assert_eq!(MetalGpuFamily::Apple5.apple_version(), 5);
    }

    #[test]
    fn test_apple_version_apple6() {
        assert_eq!(MetalGpuFamily::Apple6.apple_version(), 6);
    }

    #[test]
    fn test_apple_version_apple7() {
        assert_eq!(MetalGpuFamily::Apple7.apple_version(), 7);
    }

    #[test]
    fn test_apple_version_apple8() {
        assert_eq!(MetalGpuFamily::Apple8.apple_version(), 8);
    }

    #[test]
    fn test_apple_version_apple9() {
        assert_eq!(MetalGpuFamily::Apple9.apple_version(), 9);
    }

    #[test]
    fn test_apple_version_mac1_returns_zero() {
        assert_eq!(MetalGpuFamily::Mac1.apple_version(), 0);
    }

    #[test]
    fn test_apple_version_mac2_returns_zero() {
        assert_eq!(MetalGpuFamily::Mac2.apple_version(), 0);
    }

    #[test]
    fn test_apple_version_common_returns_zero() {
        assert_eq!(MetalGpuFamily::Common1.apple_version(), 0);
        assert_eq!(MetalGpuFamily::Common2.apple_version(), 0);
        assert_eq!(MetalGpuFamily::Common3.apple_version(), 0);
    }

    #[test]
    fn test_apple_version_metal3_returns_zero() {
        assert_eq!(MetalGpuFamily::Metal3.apple_version(), 0);
    }

    #[test]
    fn test_apple_version_unknown_returns_zero() {
        assert_eq!(MetalGpuFamily::Unknown.apple_version(), 0);
    }

    // ========================================================================
    // name
    // ========================================================================

    #[test]
    fn test_name_all_variants() {
        assert_eq!(MetalGpuFamily::Apple1.name(), "Apple1");
        assert_eq!(MetalGpuFamily::Apple2.name(), "Apple2");
        assert_eq!(MetalGpuFamily::Apple3.name(), "Apple3");
        assert_eq!(MetalGpuFamily::Apple4.name(), "Apple4");
        assert_eq!(MetalGpuFamily::Apple5.name(), "Apple5");
        assert_eq!(MetalGpuFamily::Apple6.name(), "Apple6");
        assert_eq!(MetalGpuFamily::Apple7.name(), "Apple7");
        assert_eq!(MetalGpuFamily::Apple8.name(), "Apple8");
        assert_eq!(MetalGpuFamily::Apple9.name(), "Apple9");
        assert_eq!(MetalGpuFamily::Mac1.name(), "Mac1");
        assert_eq!(MetalGpuFamily::Mac2.name(), "Mac2");
        assert_eq!(MetalGpuFamily::Common1.name(), "Common1");
        assert_eq!(MetalGpuFamily::Common2.name(), "Common2");
        assert_eq!(MetalGpuFamily::Common3.name(), "Common3");
        assert_eq!(MetalGpuFamily::Metal3.name(), "Metal3");
        assert_eq!(MetalGpuFamily::Unknown.name(), "Unknown");
    }

    // ========================================================================
    // Display Trait
    // ========================================================================

    #[test]
    fn test_display_apple7() {
        assert_eq!(format!("{}", MetalGpuFamily::Apple7), "Apple7");
    }

    #[test]
    fn test_display_mac2() {
        assert_eq!(format!("{}", MetalGpuFamily::Mac2), "Mac2");
    }

    #[test]
    fn test_display_metal3() {
        assert_eq!(format!("{}", MetalGpuFamily::Metal3), "Metal3");
    }

    #[test]
    fn test_display_unknown() {
        assert_eq!(format!("{}", MetalGpuFamily::Unknown), "Unknown");
    }

    // ========================================================================
    // Trait Implementations
    // ========================================================================

    #[test]
    fn test_clone() {
        let original = MetalGpuFamily::Apple9;
        let cloned = original.clone();
        assert_eq!(original, cloned);
    }

    #[test]
    fn test_copy() {
        let original = MetalGpuFamily::Apple9;
        let copied: MetalGpuFamily = original; // Copy
        assert_eq!(original, copied);
    }

    #[test]
    fn test_debug() {
        let family = MetalGpuFamily::Apple7;
        let debug_str = format!("{:?}", family);
        assert!(debug_str.contains("Apple7"));
    }

    #[test]
    fn test_eq() {
        assert_eq!(MetalGpuFamily::Apple7, MetalGpuFamily::Apple7);
        assert_ne!(MetalGpuFamily::Apple7, MetalGpuFamily::Apple8);
    }

    #[test]
    fn test_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(MetalGpuFamily::Apple7);
        set.insert(MetalGpuFamily::Apple8);
        set.insert(MetalGpuFamily::Apple7); // Duplicate
        assert_eq!(set.len(), 2);
    }

    // ========================================================================
    // Feature Progression
    // ========================================================================

    #[test]
    fn test_feature_progression_apple7_plus_supports_metal3() {
        assert!(MetalGpuFamily::Apple7.supports_metal3());
        assert!(MetalGpuFamily::Apple8.supports_metal3());
        assert!(MetalGpuFamily::Apple9.supports_metal3());
    }

    #[test]
    fn test_feature_progression_pre_apple7_no_metal3() {
        assert!(!MetalGpuFamily::Apple1.supports_metal3());
        assert!(!MetalGpuFamily::Apple2.supports_metal3());
        assert!(!MetalGpuFamily::Apple3.supports_metal3());
        assert!(!MetalGpuFamily::Apple4.supports_metal3());
        assert!(!MetalGpuFamily::Apple5.supports_metal3());
        assert!(!MetalGpuFamily::Apple6.supports_metal3());
    }
}

// ============================================================================
// Module: AppleSiliconGeneration Tests
// ============================================================================

mod apple_silicon_generation {
    use super::*;

    // ========================================================================
    // Default and Enum Variants
    // ========================================================================

    #[test]
    fn test_default_is_unknown() {
        let gen = AppleSiliconGeneration::default();
        assert_eq!(gen, AppleSiliconGeneration::Unknown);
    }

    #[test]
    fn test_all_a_series_variants_exist() {
        let variants = [
            AppleSiliconGeneration::A14,
            AppleSiliconGeneration::A15,
            AppleSiliconGeneration::A16,
            AppleSiliconGeneration::A17Pro,
        ];
        assert_eq!(variants.len(), 4);
    }

    #[test]
    fn test_all_m1_variants_exist() {
        let variants = [
            AppleSiliconGeneration::M1,
            AppleSiliconGeneration::M1Pro,
            AppleSiliconGeneration::M1Max,
            AppleSiliconGeneration::M1Ultra,
        ];
        assert_eq!(variants.len(), 4);
    }

    #[test]
    fn test_all_m2_variants_exist() {
        let variants = [
            AppleSiliconGeneration::M2,
            AppleSiliconGeneration::M2Pro,
            AppleSiliconGeneration::M2Max,
            AppleSiliconGeneration::M2Ultra,
        ];
        assert_eq!(variants.len(), 4);
    }

    #[test]
    fn test_all_m3_variants_exist() {
        let variants = [
            AppleSiliconGeneration::M3,
            AppleSiliconGeneration::M3Pro,
            AppleSiliconGeneration::M3Max,
        ];
        assert_eq!(variants.len(), 3);
    }

    #[test]
    fn test_all_m4_variants_exist() {
        let variants = [
            AppleSiliconGeneration::M4,
            AppleSiliconGeneration::M4Pro,
            AppleSiliconGeneration::M4Max,
        ];
        assert_eq!(variants.len(), 3);
    }

    // ========================================================================
    // from_device_name: M1 Series
    // ========================================================================

    #[test]
    fn test_from_device_name_m1_base() {
        assert_eq!(
            AppleSiliconGeneration::from_device_name("Apple M1"),
            AppleSiliconGeneration::M1
        );
    }

    #[test]
    fn test_from_device_name_m1_pro() {
        assert_eq!(
            AppleSiliconGeneration::from_device_name("Apple M1 Pro"),
            AppleSiliconGeneration::M1Pro
        );
    }

    #[test]
    fn test_from_device_name_m1_max() {
        assert_eq!(
            AppleSiliconGeneration::from_device_name("Apple M1 Max"),
            AppleSiliconGeneration::M1Max
        );
    }

    #[test]
    fn test_from_device_name_m1_ultra() {
        assert_eq!(
            AppleSiliconGeneration::from_device_name("Apple M1 Ultra"),
            AppleSiliconGeneration::M1Ultra
        );
    }

    // ========================================================================
    // from_device_name: M2 Series
    // ========================================================================

    #[test]
    fn test_from_device_name_m2_base() {
        assert_eq!(
            AppleSiliconGeneration::from_device_name("Apple M2"),
            AppleSiliconGeneration::M2
        );
    }

    #[test]
    fn test_from_device_name_m2_pro() {
        assert_eq!(
            AppleSiliconGeneration::from_device_name("Apple M2 Pro"),
            AppleSiliconGeneration::M2Pro
        );
    }

    #[test]
    fn test_from_device_name_m2_max() {
        assert_eq!(
            AppleSiliconGeneration::from_device_name("Apple M2 Max"),
            AppleSiliconGeneration::M2Max
        );
    }

    #[test]
    fn test_from_device_name_m2_ultra() {
        assert_eq!(
            AppleSiliconGeneration::from_device_name("Apple M2 Ultra"),
            AppleSiliconGeneration::M2Ultra
        );
    }

    // ========================================================================
    // from_device_name: M3 Series
    // ========================================================================

    #[test]
    fn test_from_device_name_m3_base() {
        assert_eq!(
            AppleSiliconGeneration::from_device_name("Apple M3"),
            AppleSiliconGeneration::M3
        );
    }

    #[test]
    fn test_from_device_name_m3_pro() {
        assert_eq!(
            AppleSiliconGeneration::from_device_name("Apple M3 Pro"),
            AppleSiliconGeneration::M3Pro
        );
    }

    #[test]
    fn test_from_device_name_m3_max() {
        assert_eq!(
            AppleSiliconGeneration::from_device_name("Apple M3 Max"),
            AppleSiliconGeneration::M3Max
        );
    }

    // ========================================================================
    // from_device_name: M4 Series
    // ========================================================================

    #[test]
    fn test_from_device_name_m4_base() {
        assert_eq!(
            AppleSiliconGeneration::from_device_name("Apple M4"),
            AppleSiliconGeneration::M4
        );
    }

    #[test]
    fn test_from_device_name_m4_pro() {
        assert_eq!(
            AppleSiliconGeneration::from_device_name("Apple M4 Pro"),
            AppleSiliconGeneration::M4Pro
        );
    }

    #[test]
    fn test_from_device_name_m4_max() {
        assert_eq!(
            AppleSiliconGeneration::from_device_name("Apple M4 Max"),
            AppleSiliconGeneration::M4Max
        );
    }

    // ========================================================================
    // from_device_name: A-series
    // ========================================================================

    #[test]
    fn test_from_device_name_a14() {
        assert_eq!(
            AppleSiliconGeneration::from_device_name("Apple A14 Bionic"),
            AppleSiliconGeneration::A14
        );
    }

    #[test]
    fn test_from_device_name_a15() {
        assert_eq!(
            AppleSiliconGeneration::from_device_name("Apple A15 Bionic"),
            AppleSiliconGeneration::A15
        );
    }

    #[test]
    fn test_from_device_name_a16() {
        assert_eq!(
            AppleSiliconGeneration::from_device_name("Apple A16 Bionic"),
            AppleSiliconGeneration::A16
        );
    }

    #[test]
    fn test_from_device_name_a17_pro() {
        assert_eq!(
            AppleSiliconGeneration::from_device_name("Apple A17 Pro"),
            AppleSiliconGeneration::A17Pro
        );
    }

    // ========================================================================
    // from_device_name: Case Insensitivity
    // ========================================================================

    #[test]
    fn test_from_device_name_lowercase_m3_pro() {
        assert_eq!(
            AppleSiliconGeneration::from_device_name("apple m3 pro"),
            AppleSiliconGeneration::M3Pro
        );
    }

    #[test]
    fn test_from_device_name_uppercase_m4_max() {
        assert_eq!(
            AppleSiliconGeneration::from_device_name("APPLE M4 MAX"),
            AppleSiliconGeneration::M4Max
        );
    }

    #[test]
    fn test_from_device_name_mixed_case_m2_ultra() {
        assert_eq!(
            AppleSiliconGeneration::from_device_name("aPpLe M2 uLtRa"),
            AppleSiliconGeneration::M2Ultra
        );
    }

    // ========================================================================
    // from_device_name: Edge Cases
    // ========================================================================

    #[test]
    fn test_from_device_name_empty_string() {
        assert_eq!(
            AppleSiliconGeneration::from_device_name(""),
            AppleSiliconGeneration::Unknown
        );
    }

    #[test]
    fn test_from_device_name_unknown_device() {
        assert_eq!(
            AppleSiliconGeneration::from_device_name("Intel Core i9"),
            AppleSiliconGeneration::Unknown
        );
    }

    #[test]
    fn test_from_device_name_whitespace_only() {
        assert_eq!(
            AppleSiliconGeneration::from_device_name("   "),
            AppleSiliconGeneration::Unknown
        );
    }

    #[test]
    fn test_from_device_name_older_a_series() {
        // A13 and earlier not tracked by AppleSiliconGeneration
        assert_eq!(
            AppleSiliconGeneration::from_device_name("Apple A13 Bionic"),
            AppleSiliconGeneration::Unknown
        );
    }

    #[test]
    fn test_from_device_name_partial_m2() {
        assert_eq!(
            AppleSiliconGeneration::from_device_name("m2"),
            AppleSiliconGeneration::M2
        );
    }

    #[test]
    fn test_from_device_name_extra_whitespace() {
        assert_eq!(
            AppleSiliconGeneration::from_device_name("  Apple  M3  Pro  "),
            AppleSiliconGeneration::M3Pro
        );
    }

    // ========================================================================
    // generation_number
    // ========================================================================

    #[test]
    fn test_generation_number_m1_series() {
        assert_eq!(AppleSiliconGeneration::M1.generation_number(), 1);
        assert_eq!(AppleSiliconGeneration::M1Pro.generation_number(), 1);
        assert_eq!(AppleSiliconGeneration::M1Max.generation_number(), 1);
        assert_eq!(AppleSiliconGeneration::M1Ultra.generation_number(), 1);
    }

    #[test]
    fn test_generation_number_m2_series() {
        assert_eq!(AppleSiliconGeneration::M2.generation_number(), 2);
        assert_eq!(AppleSiliconGeneration::M2Pro.generation_number(), 2);
        assert_eq!(AppleSiliconGeneration::M2Max.generation_number(), 2);
        assert_eq!(AppleSiliconGeneration::M2Ultra.generation_number(), 2);
    }

    #[test]
    fn test_generation_number_m3_series() {
        assert_eq!(AppleSiliconGeneration::M3.generation_number(), 3);
        assert_eq!(AppleSiliconGeneration::M3Pro.generation_number(), 3);
        assert_eq!(AppleSiliconGeneration::M3Max.generation_number(), 3);
    }

    #[test]
    fn test_generation_number_m4_series() {
        assert_eq!(AppleSiliconGeneration::M4.generation_number(), 4);
        assert_eq!(AppleSiliconGeneration::M4Pro.generation_number(), 4);
        assert_eq!(AppleSiliconGeneration::M4Max.generation_number(), 4);
    }

    #[test]
    fn test_generation_number_a_series_returns_zero() {
        assert_eq!(AppleSiliconGeneration::A14.generation_number(), 0);
        assert_eq!(AppleSiliconGeneration::A15.generation_number(), 0);
        assert_eq!(AppleSiliconGeneration::A16.generation_number(), 0);
        assert_eq!(AppleSiliconGeneration::A17Pro.generation_number(), 0);
    }

    #[test]
    fn test_generation_number_unknown_returns_zero() {
        assert_eq!(AppleSiliconGeneration::Unknown.generation_number(), 0);
    }

    // ========================================================================
    // is_m_series
    // ========================================================================

    #[test]
    fn test_is_m_series_m1() {
        assert!(AppleSiliconGeneration::M1.is_m_series());
        assert!(AppleSiliconGeneration::M1Pro.is_m_series());
        assert!(AppleSiliconGeneration::M1Max.is_m_series());
        assert!(AppleSiliconGeneration::M1Ultra.is_m_series());
    }

    #[test]
    fn test_is_m_series_m2() {
        assert!(AppleSiliconGeneration::M2.is_m_series());
        assert!(AppleSiliconGeneration::M2Pro.is_m_series());
        assert!(AppleSiliconGeneration::M2Max.is_m_series());
        assert!(AppleSiliconGeneration::M2Ultra.is_m_series());
    }

    #[test]
    fn test_is_m_series_m3() {
        assert!(AppleSiliconGeneration::M3.is_m_series());
        assert!(AppleSiliconGeneration::M3Pro.is_m_series());
        assert!(AppleSiliconGeneration::M3Max.is_m_series());
    }

    #[test]
    fn test_is_m_series_m4() {
        assert!(AppleSiliconGeneration::M4.is_m_series());
        assert!(AppleSiliconGeneration::M4Pro.is_m_series());
        assert!(AppleSiliconGeneration::M4Max.is_m_series());
    }

    #[test]
    fn test_is_m_series_a_series_false() {
        assert!(!AppleSiliconGeneration::A14.is_m_series());
        assert!(!AppleSiliconGeneration::A15.is_m_series());
        assert!(!AppleSiliconGeneration::A16.is_m_series());
        assert!(!AppleSiliconGeneration::A17Pro.is_m_series());
    }

    #[test]
    fn test_is_m_series_unknown_false() {
        assert!(!AppleSiliconGeneration::Unknown.is_m_series());
    }

    // ========================================================================
    // is_a_series
    // ========================================================================

    #[test]
    fn test_is_a_series_a14() {
        assert!(AppleSiliconGeneration::A14.is_a_series());
    }

    #[test]
    fn test_is_a_series_a15() {
        assert!(AppleSiliconGeneration::A15.is_a_series());
    }

    #[test]
    fn test_is_a_series_a16() {
        assert!(AppleSiliconGeneration::A16.is_a_series());
    }

    #[test]
    fn test_is_a_series_a17_pro() {
        assert!(AppleSiliconGeneration::A17Pro.is_a_series());
    }

    #[test]
    fn test_is_a_series_m_series_false() {
        assert!(!AppleSiliconGeneration::M1.is_a_series());
        assert!(!AppleSiliconGeneration::M2Pro.is_a_series());
        assert!(!AppleSiliconGeneration::M3Max.is_a_series());
    }

    #[test]
    fn test_is_a_series_unknown_false() {
        assert!(!AppleSiliconGeneration::Unknown.is_a_series());
    }

    // ========================================================================
    // is_pro_tier
    // ========================================================================

    #[test]
    fn test_is_pro_tier_m1_series() {
        assert!(!AppleSiliconGeneration::M1.is_pro_tier());
        assert!(AppleSiliconGeneration::M1Pro.is_pro_tier());
        assert!(AppleSiliconGeneration::M1Max.is_pro_tier());
        assert!(AppleSiliconGeneration::M1Ultra.is_pro_tier());
    }

    #[test]
    fn test_is_pro_tier_m2_series() {
        assert!(!AppleSiliconGeneration::M2.is_pro_tier());
        assert!(AppleSiliconGeneration::M2Pro.is_pro_tier());
        assert!(AppleSiliconGeneration::M2Max.is_pro_tier());
        assert!(AppleSiliconGeneration::M2Ultra.is_pro_tier());
    }

    #[test]
    fn test_is_pro_tier_m3_series() {
        assert!(!AppleSiliconGeneration::M3.is_pro_tier());
        assert!(AppleSiliconGeneration::M3Pro.is_pro_tier());
        assert!(AppleSiliconGeneration::M3Max.is_pro_tier());
    }

    #[test]
    fn test_is_pro_tier_m4_series() {
        assert!(!AppleSiliconGeneration::M4.is_pro_tier());
        assert!(AppleSiliconGeneration::M4Pro.is_pro_tier());
        assert!(AppleSiliconGeneration::M4Max.is_pro_tier());
    }

    #[test]
    fn test_is_pro_tier_a17_pro() {
        assert!(AppleSiliconGeneration::A17Pro.is_pro_tier());
    }

    #[test]
    fn test_is_pro_tier_other_a_series_false() {
        assert!(!AppleSiliconGeneration::A14.is_pro_tier());
        assert!(!AppleSiliconGeneration::A15.is_pro_tier());
        assert!(!AppleSiliconGeneration::A16.is_pro_tier());
    }

    #[test]
    fn test_is_pro_tier_unknown_false() {
        assert!(!AppleSiliconGeneration::Unknown.is_pro_tier());
    }

    // ========================================================================
    // estimated_gpu_cores
    // ========================================================================

    #[test]
    fn test_estimated_gpu_cores_a_series() {
        assert_eq!(AppleSiliconGeneration::A14.estimated_gpu_cores(), 4);
        assert_eq!(AppleSiliconGeneration::A15.estimated_gpu_cores(), 5);
        assert_eq!(AppleSiliconGeneration::A16.estimated_gpu_cores(), 5);
        assert_eq!(AppleSiliconGeneration::A17Pro.estimated_gpu_cores(), 6);
    }

    #[test]
    fn test_estimated_gpu_cores_m1_series() {
        assert_eq!(AppleSiliconGeneration::M1.estimated_gpu_cores(), 8);
        assert_eq!(AppleSiliconGeneration::M1Pro.estimated_gpu_cores(), 16);
        assert_eq!(AppleSiliconGeneration::M1Max.estimated_gpu_cores(), 32);
        assert_eq!(AppleSiliconGeneration::M1Ultra.estimated_gpu_cores(), 64);
    }

    #[test]
    fn test_estimated_gpu_cores_m2_series() {
        assert_eq!(AppleSiliconGeneration::M2.estimated_gpu_cores(), 10);
        assert_eq!(AppleSiliconGeneration::M2Pro.estimated_gpu_cores(), 19);
        assert_eq!(AppleSiliconGeneration::M2Max.estimated_gpu_cores(), 38);
        assert_eq!(AppleSiliconGeneration::M2Ultra.estimated_gpu_cores(), 76);
    }

    #[test]
    fn test_estimated_gpu_cores_m3_series() {
        assert_eq!(AppleSiliconGeneration::M3.estimated_gpu_cores(), 10);
        assert_eq!(AppleSiliconGeneration::M3Pro.estimated_gpu_cores(), 18);
        assert_eq!(AppleSiliconGeneration::M3Max.estimated_gpu_cores(), 40);
    }

    #[test]
    fn test_estimated_gpu_cores_m4_series() {
        assert_eq!(AppleSiliconGeneration::M4.estimated_gpu_cores(), 10);
        assert_eq!(AppleSiliconGeneration::M4Pro.estimated_gpu_cores(), 20);
        assert_eq!(AppleSiliconGeneration::M4Max.estimated_gpu_cores(), 40);
    }

    #[test]
    fn test_estimated_gpu_cores_unknown_returns_zero() {
        assert_eq!(AppleSiliconGeneration::Unknown.estimated_gpu_cores(), 0);
    }

    // ========================================================================
    // name
    // ========================================================================

    #[test]
    fn test_name_a_series() {
        assert_eq!(AppleSiliconGeneration::A14.name(), "A14 Bionic");
        assert_eq!(AppleSiliconGeneration::A15.name(), "A15 Bionic");
        assert_eq!(AppleSiliconGeneration::A16.name(), "A16 Bionic");
        assert_eq!(AppleSiliconGeneration::A17Pro.name(), "A17 Pro");
    }

    #[test]
    fn test_name_m1_series() {
        assert_eq!(AppleSiliconGeneration::M1.name(), "M1");
        assert_eq!(AppleSiliconGeneration::M1Pro.name(), "M1 Pro");
        assert_eq!(AppleSiliconGeneration::M1Max.name(), "M1 Max");
        assert_eq!(AppleSiliconGeneration::M1Ultra.name(), "M1 Ultra");
    }

    #[test]
    fn test_name_m2_series() {
        assert_eq!(AppleSiliconGeneration::M2.name(), "M2");
        assert_eq!(AppleSiliconGeneration::M2Pro.name(), "M2 Pro");
        assert_eq!(AppleSiliconGeneration::M2Max.name(), "M2 Max");
        assert_eq!(AppleSiliconGeneration::M2Ultra.name(), "M2 Ultra");
    }

    #[test]
    fn test_name_m3_series() {
        assert_eq!(AppleSiliconGeneration::M3.name(), "M3");
        assert_eq!(AppleSiliconGeneration::M3Pro.name(), "M3 Pro");
        assert_eq!(AppleSiliconGeneration::M3Max.name(), "M3 Max");
    }

    #[test]
    fn test_name_m4_series() {
        assert_eq!(AppleSiliconGeneration::M4.name(), "M4");
        assert_eq!(AppleSiliconGeneration::M4Pro.name(), "M4 Pro");
        assert_eq!(AppleSiliconGeneration::M4Max.name(), "M4 Max");
    }

    #[test]
    fn test_name_unknown() {
        assert_eq!(AppleSiliconGeneration::Unknown.name(), "Unknown");
    }

    // ========================================================================
    // Display Trait
    // ========================================================================

    #[test]
    fn test_display_m3_max() {
        assert_eq!(format!("{}", AppleSiliconGeneration::M3Max), "M3 Max");
    }

    #[test]
    fn test_display_a17_pro() {
        assert_eq!(format!("{}", AppleSiliconGeneration::A17Pro), "A17 Pro");
    }

    #[test]
    fn test_display_m1_ultra() {
        assert_eq!(format!("{}", AppleSiliconGeneration::M1Ultra), "M1 Ultra");
    }

    // ========================================================================
    // Trait Implementations
    // ========================================================================

    #[test]
    fn test_clone() {
        let original = AppleSiliconGeneration::M3Pro;
        let cloned = original.clone();
        assert_eq!(original, cloned);
    }

    #[test]
    fn test_copy() {
        let original = AppleSiliconGeneration::M3Pro;
        let copied: AppleSiliconGeneration = original;
        assert_eq!(original, copied);
    }

    #[test]
    fn test_debug() {
        let gen = AppleSiliconGeneration::M2Max;
        let debug_str = format!("{:?}", gen);
        assert!(debug_str.contains("M2Max"));
    }

    #[test]
    fn test_eq() {
        assert_eq!(AppleSiliconGeneration::M4, AppleSiliconGeneration::M4);
        assert_ne!(AppleSiliconGeneration::M4, AppleSiliconGeneration::M4Pro);
    }

    #[test]
    fn test_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(AppleSiliconGeneration::M3);
        set.insert(AppleSiliconGeneration::M3Pro);
        set.insert(AppleSiliconGeneration::M3); // Duplicate
        assert_eq!(set.len(), 2);
    }
}

// ============================================================================
// Module: MetalFeatures Tests
// ============================================================================

mod metal_features {
    use super::*;

    // ========================================================================
    // Default
    // ========================================================================

    #[test]
    fn test_default_gpu_family_unknown() {
        let features = MetalFeatures::default();
        assert_eq!(features.gpu_family, MetalGpuFamily::Unknown);
    }

    #[test]
    fn test_default_silicon_generation_unknown() {
        let features = MetalFeatures::default();
        assert_eq!(features.silicon_generation, AppleSiliconGeneration::Unknown);
    }

    #[test]
    fn test_default_argument_buffers_tier_zero() {
        let features = MetalFeatures::default();
        assert_eq!(features.argument_buffers_tier, 0);
    }

    #[test]
    fn test_default_no_ray_tracing() {
        let features = MetalFeatures::default();
        assert!(!features.ray_tracing);
    }

    #[test]
    fn test_default_no_mesh_shaders() {
        let features = MetalFeatures::default();
        assert!(!features.mesh_shaders);
    }

    #[test]
    fn test_default_all_features_false() {
        let features = MetalFeatures::default();
        assert!(!features.float32_msaa_resolve);
        assert!(!features.sparse_textures);
        assert!(!features.primitive_motion_blur);
        assert!(!features.memoryless_render_targets);
        assert!(!features.lossless_compression);
        assert!(!features.function_pointers);
        assert!(!features.primitive_restart_32bit);
        assert!(!features.simd_group);
        assert!(!features.read_write_textures);
        assert!(!features.tile_shaders);
        assert!(!features.imageblock);
    }

    // ========================================================================
    // from_device_info: M-series
    // ========================================================================

    #[test]
    fn test_from_device_info_m1_gpu_family() {
        let features = MetalFeatures::from_device_info("Apple M1");
        assert_eq!(features.gpu_family, MetalGpuFamily::Apple7);
    }

    #[test]
    fn test_from_device_info_m1_silicon_generation() {
        let features = MetalFeatures::from_device_info("Apple M1");
        assert_eq!(features.silicon_generation, AppleSiliconGeneration::M1);
    }

    #[test]
    fn test_from_device_info_m1_argument_buffers_tier_2() {
        let features = MetalFeatures::from_device_info("Apple M1");
        assert_eq!(features.argument_buffers_tier, 2);
    }

    #[test]
    fn test_from_device_info_m1_ray_tracing() {
        let features = MetalFeatures::from_device_info("Apple M1");
        assert!(features.ray_tracing);
    }

    #[test]
    fn test_from_device_info_m1_mesh_shaders() {
        let features = MetalFeatures::from_device_info("Apple M1");
        assert!(features.mesh_shaders);
    }

    #[test]
    fn test_from_device_info_m1_memoryless_render_targets() {
        let features = MetalFeatures::from_device_info("Apple M1");
        assert!(features.memoryless_render_targets);
    }

    #[test]
    fn test_from_device_info_m1_lossless_compression() {
        let features = MetalFeatures::from_device_info("Apple M1");
        assert!(features.lossless_compression);
    }

    #[test]
    fn test_from_device_info_m2_pro() {
        let features = MetalFeatures::from_device_info("Apple M2 Pro");
        assert_eq!(features.gpu_family, MetalGpuFamily::Apple8);
        assert_eq!(features.silicon_generation, AppleSiliconGeneration::M2Pro);
    }

    #[test]
    fn test_from_device_info_m3_max() {
        let features = MetalFeatures::from_device_info("Apple M3 Max");
        assert_eq!(features.gpu_family, MetalGpuFamily::Apple9);
        assert_eq!(features.silicon_generation, AppleSiliconGeneration::M3Max);
    }

    #[test]
    fn test_from_device_info_m3_max_all_metal3_features() {
        let features = MetalFeatures::from_device_info("Apple M3 Max");
        assert!(features.ray_tracing);
        assert!(features.mesh_shaders);
        assert!(features.primitive_motion_blur);
        assert!(features.function_pointers);
    }

    #[test]
    fn test_from_device_info_m4() {
        let features = MetalFeatures::from_device_info("Apple M4");
        assert_eq!(features.gpu_family, MetalGpuFamily::Apple9);
        assert_eq!(features.silicon_generation, AppleSiliconGeneration::M4);
    }

    #[test]
    fn test_from_device_info_m4_max() {
        let features = MetalFeatures::from_device_info("Apple M4 Max");
        assert_eq!(features.gpu_family, MetalGpuFamily::Apple9);
        assert_eq!(features.silicon_generation, AppleSiliconGeneration::M4Max);
    }

    // ========================================================================
    // from_device_info: Intel Mac
    // ========================================================================

    #[test]
    fn test_from_device_info_intel_uhd_gpu_family() {
        let features = MetalFeatures::from_device_info("Intel UHD Graphics 630");
        assert_eq!(features.gpu_family, MetalGpuFamily::Mac2);
    }

    #[test]
    fn test_from_device_info_intel_uhd_silicon_generation_unknown() {
        let features = MetalFeatures::from_device_info("Intel UHD Graphics 630");
        assert_eq!(features.silicon_generation, AppleSiliconGeneration::Unknown);
    }

    #[test]
    fn test_from_device_info_intel_no_ray_tracing() {
        let features = MetalFeatures::from_device_info("Intel UHD Graphics 630");
        assert!(!features.ray_tracing);
    }

    #[test]
    fn test_from_device_info_intel_no_mesh_shaders() {
        let features = MetalFeatures::from_device_info("Intel UHD Graphics 630");
        assert!(!features.mesh_shaders);
    }

    #[test]
    fn test_from_device_info_intel_simd_group() {
        let features = MetalFeatures::from_device_info("Intel UHD Graphics 630");
        assert!(features.simd_group);
    }

    #[test]
    fn test_from_device_info_intel_read_write_textures() {
        let features = MetalFeatures::from_device_info("Intel UHD Graphics 630");
        assert!(features.read_write_textures);
    }

    #[test]
    fn test_from_device_info_intel_float32_msaa_resolve() {
        let features = MetalFeatures::from_device_info("Intel UHD Graphics 630");
        assert!(features.float32_msaa_resolve);
    }

    // ========================================================================
    // from_device_info: A-series
    // ========================================================================

    #[test]
    fn test_from_device_info_a14_ray_tracing() {
        let features = MetalFeatures::from_device_info("Apple A14 Bionic");
        assert!(features.ray_tracing);
    }

    #[test]
    fn test_from_device_info_a13_no_ray_tracing() {
        let features = MetalFeatures::from_device_info("Apple A13 Bionic");
        assert!(!features.ray_tracing);
    }

    #[test]
    fn test_from_device_info_a12_sparse_textures() {
        let features = MetalFeatures::from_device_info("Apple A12 Bionic");
        assert!(features.sparse_textures);
    }

    #[test]
    fn test_from_device_info_a11_no_sparse_textures() {
        let features = MetalFeatures::from_device_info("Apple A11 Bionic");
        assert!(!features.sparse_textures);
    }

    #[test]
    fn test_from_device_info_a11_tile_shaders() {
        let features = MetalFeatures::from_device_info("Apple A11 Bionic");
        assert!(features.tile_shaders);
    }

    #[test]
    fn test_from_device_info_a11_imageblock() {
        let features = MetalFeatures::from_device_info("Apple A11 Bionic");
        assert!(features.imageblock);
    }

    #[test]
    fn test_from_device_info_a10_no_tile_shaders() {
        let features = MetalFeatures::from_device_info("Apple A10");
        assert!(!features.tile_shaders);
    }

    #[test]
    fn test_from_device_info_a9_simd_group() {
        let features = MetalFeatures::from_device_info("Apple A9");
        assert!(features.simd_group);
    }

    // ========================================================================
    // from_adapter_info with wgpu Features
    // ========================================================================

    #[test]
    fn test_from_adapter_info_with_ray_tracing_feature() {
        let features = MetalFeatures::from_adapter_info(
            "Apple A12 Bionic",
            Features::RAY_TRACING_ACCELERATION_STRUCTURE,
        );
        // A12 doesn't normally have RT, but wgpu feature flag enables it
        assert!(features.ray_tracing);
    }

    #[test]
    fn test_from_adapter_info_with_ray_query_feature() {
        let features =
            MetalFeatures::from_adapter_info("Apple A12 Bionic", Features::RAY_QUERY);
        assert!(features.ray_tracing);
    }

    #[test]
    fn test_from_adapter_info_empty_features() {
        let features =
            MetalFeatures::from_adapter_info("Apple M1", Features::empty());
        // M1 still has RT from family detection
        assert!(features.ray_tracing);
    }

    // ========================================================================
    // supports_rt
    // ========================================================================

    #[test]
    fn test_supports_rt_m1() {
        let features = MetalFeatures::from_device_info("Apple M1");
        assert!(features.supports_rt());
    }

    #[test]
    fn test_supports_rt_m4_max() {
        let features = MetalFeatures::from_device_info("Apple M4 Max");
        assert!(features.supports_rt());
    }

    #[test]
    fn test_supports_rt_intel_false() {
        let features = MetalFeatures::from_device_info("Intel UHD Graphics");
        assert!(!features.supports_rt());
    }

    #[test]
    fn test_supports_rt_a13_false() {
        let features = MetalFeatures::from_device_info("Apple A13");
        assert!(!features.supports_rt());
    }

    // ========================================================================
    // supports_mesh_shaders
    // ========================================================================

    #[test]
    fn test_supports_mesh_shaders_m1() {
        let features = MetalFeatures::from_device_info("Apple M1");
        assert!(features.supports_mesh_shaders());
    }

    #[test]
    fn test_supports_mesh_shaders_m3() {
        let features = MetalFeatures::from_device_info("Apple M3");
        assert!(features.supports_mesh_shaders());
    }

    #[test]
    fn test_supports_mesh_shaders_a13_false() {
        let features = MetalFeatures::from_device_info("Apple A13");
        assert!(!features.supports_mesh_shaders());
    }

    // ========================================================================
    // supports_argument_buffers
    // ========================================================================

    #[test]
    fn test_supports_argument_buffers_m1() {
        let features = MetalFeatures::from_device_info("Apple M1");
        assert!(features.supports_argument_buffers());
    }

    #[test]
    fn test_supports_argument_buffers_a12() {
        let features = MetalFeatures::from_device_info("Apple A12");
        assert!(features.supports_argument_buffers());
    }

    #[test]
    fn test_supports_argument_buffers_default_false() {
        let features = MetalFeatures::default();
        assert!(!features.supports_argument_buffers());
    }

    // ========================================================================
    // supports_bindless
    // ========================================================================

    #[test]
    fn test_supports_bindless_m1() {
        let features = MetalFeatures::from_device_info("Apple M1");
        assert!(features.supports_bindless());
    }

    #[test]
    fn test_supports_bindless_m3_pro() {
        let features = MetalFeatures::from_device_info("Apple M3 Pro");
        assert!(features.supports_bindless());
    }

    #[test]
    fn test_supports_bindless_default_false() {
        let features = MetalFeatures::default();
        assert!(!features.supports_bindless());
    }

    // ========================================================================
    // is_m_series
    // ========================================================================

    #[test]
    fn test_is_m_series_m1() {
        let features = MetalFeatures::from_device_info("Apple M1");
        assert!(features.is_m_series());
    }

    #[test]
    fn test_is_m_series_m4_pro() {
        let features = MetalFeatures::from_device_info("Apple M4 Pro");
        assert!(features.is_m_series());
    }

    #[test]
    fn test_is_m_series_a15_false() {
        let features = MetalFeatures::from_device_info("Apple A15");
        assert!(!features.is_m_series());
    }

    #[test]
    fn test_is_m_series_intel_false() {
        let features = MetalFeatures::from_device_info("Intel Iris Pro");
        assert!(!features.is_m_series());
    }

    // ========================================================================
    // minimum_macos_version
    // ========================================================================

    #[test]
    fn test_minimum_macos_version_m3() {
        let features = MetalFeatures::from_device_info("Apple M3");
        assert_eq!(features.minimum_macos_version(), (14, 0));
    }

    #[test]
    fn test_minimum_macos_version_m3_pro() {
        let features = MetalFeatures::from_device_info("Apple M3 Pro");
        assert_eq!(features.minimum_macos_version(), (14, 0));
    }

    #[test]
    fn test_minimum_macos_version_m3_max() {
        let features = MetalFeatures::from_device_info("Apple M3 Max");
        assert_eq!(features.minimum_macos_version(), (14, 0));
    }

    #[test]
    fn test_minimum_macos_version_m4() {
        let features = MetalFeatures::from_device_info("Apple M4");
        assert_eq!(features.minimum_macos_version(), (14, 0));
    }

    #[test]
    fn test_minimum_macos_version_m4_pro() {
        let features = MetalFeatures::from_device_info("Apple M4 Pro");
        assert_eq!(features.minimum_macos_version(), (14, 0));
    }

    #[test]
    fn test_minimum_macos_version_m4_max() {
        let features = MetalFeatures::from_device_info("Apple M4 Max");
        assert_eq!(features.minimum_macos_version(), (14, 0));
    }

    #[test]
    fn test_minimum_macos_version_m1() {
        let features = MetalFeatures::from_device_info("Apple M1");
        // M1 with RT requires Ventura (13.0)
        assert_eq!(features.minimum_macos_version(), (13, 0));
    }

    #[test]
    fn test_minimum_macos_version_m2() {
        let features = MetalFeatures::from_device_info("Apple M2");
        assert_eq!(features.minimum_macos_version(), (13, 0));
    }

    #[test]
    fn test_minimum_macos_version_intel_with_arg_buffers() {
        let features = MetalFeatures::from_device_info("Intel UHD Graphics 630");
        // Intel Mac2 doesn't have RT/mesh, fallback to Catalina
        assert_eq!(features.minimum_macos_version(), (10, 15));
    }

    // ========================================================================
    // summary
    // ========================================================================

    #[test]
    fn test_summary_m3_pro_contains_family() {
        let features = MetalFeatures::from_device_info("Apple M3 Pro");
        let summary = features.summary();
        assert!(summary.contains("Apple9"));
    }

    #[test]
    fn test_summary_m3_pro_contains_chip_name() {
        let features = MetalFeatures::from_device_info("Apple M3 Pro");
        let summary = features.summary();
        assert!(summary.contains("M3 Pro"));
    }

    #[test]
    fn test_summary_m3_pro_contains_rt() {
        let features = MetalFeatures::from_device_info("Apple M3 Pro");
        let summary = features.summary();
        assert!(summary.contains("RT"));
    }

    #[test]
    fn test_summary_m3_pro_contains_mesh() {
        let features = MetalFeatures::from_device_info("Apple M3 Pro");
        let summary = features.summary();
        assert!(summary.contains("Mesh"));
    }

    #[test]
    fn test_summary_m3_pro_contains_arg_buf_t2() {
        let features = MetalFeatures::from_device_info("Apple M3 Pro");
        let summary = features.summary();
        assert!(summary.contains("ArgBuf-T2"));
    }

    #[test]
    fn test_summary_m3_pro_contains_sparse() {
        let features = MetalFeatures::from_device_info("Apple M3 Pro");
        let summary = features.summary();
        assert!(summary.contains("Sparse"));
    }

    #[test]
    fn test_summary_m3_pro_contains_memoryless() {
        let features = MetalFeatures::from_device_info("Apple M3 Pro");
        let summary = features.summary();
        assert!(summary.contains("Memoryless"));
    }

    #[test]
    fn test_summary_m3_pro_contains_lossless_comp() {
        let features = MetalFeatures::from_device_info("Apple M3 Pro");
        let summary = features.summary();
        assert!(summary.contains("LosslessComp"));
    }

    #[test]
    fn test_summary_intel_no_rt() {
        let features = MetalFeatures::from_device_info("Intel UHD Graphics 630");
        let summary = features.summary();
        assert!(!summary.contains("RT"));
    }

    #[test]
    fn test_summary_intel_no_mesh() {
        let features = MetalFeatures::from_device_info("Intel UHD Graphics 630");
        let summary = features.summary();
        assert!(!summary.contains("Mesh"));
    }

    #[test]
    fn test_summary_intel_contains_mac2() {
        let features = MetalFeatures::from_device_info("Intel UHD Graphics 630");
        let summary = features.summary();
        assert!(summary.contains("Mac2"));
    }

    // ========================================================================
    // Field Initialization: Edge Cases
    // ========================================================================

    #[test]
    fn test_from_device_info_empty_string() {
        let features = MetalFeatures::from_device_info("");
        assert_eq!(features.gpu_family, MetalGpuFamily::Unknown);
        assert_eq!(features.silicon_generation, AppleSiliconGeneration::Unknown);
    }

    #[test]
    fn test_from_device_info_unknown_device() {
        let features = MetalFeatures::from_device_info("Unknown GPU Model");
        assert_eq!(features.gpu_family, MetalGpuFamily::Unknown);
    }

    // ========================================================================
    // Feature Progression Tests
    // ========================================================================

    #[test]
    fn test_m4_has_all_features() {
        let features = MetalFeatures::from_device_info("Apple M4 Max");
        assert!(features.supports_rt());
        assert!(features.supports_mesh_shaders());
        assert!(features.supports_bindless());
        assert!(features.lossless_compression);
        assert!(features.primitive_motion_blur);
        assert!(features.function_pointers);
        assert!(features.sparse_textures);
        assert!(features.tile_shaders);
        assert!(features.imageblock);
        assert!(features.simd_group);
        assert!(features.read_write_textures);
        assert!(features.memoryless_render_targets);
        assert!(features.float32_msaa_resolve);
        assert!(features.primitive_restart_32bit);
    }

    #[test]
    fn test_m3_plus_has_lossless_compression() {
        let m3 = MetalFeatures::from_device_info("Apple M3");
        let m4 = MetalFeatures::from_device_info("Apple M4");
        assert!(m3.lossless_compression);
        assert!(m4.lossless_compression);
    }

    #[test]
    fn test_a14_plus_has_ray_tracing() {
        let a14 = MetalFeatures::from_device_info("Apple A14");
        let a15 = MetalFeatures::from_device_info("Apple A15");
        let a16 = MetalFeatures::from_device_info("Apple A16");
        let a17 = MetalFeatures::from_device_info("Apple A17 Pro");
        assert!(a14.supports_rt());
        assert!(a15.supports_rt());
        assert!(a16.supports_rt());
        assert!(a17.supports_rt());
    }

    #[test]
    fn test_a13_and_below_no_ray_tracing() {
        let a13 = MetalFeatures::from_device_info("Apple A13");
        let a12 = MetalFeatures::from_device_info("Apple A12");
        let a11 = MetalFeatures::from_device_info("Apple A11");
        assert!(!a13.supports_rt());
        assert!(!a12.supports_rt());
        assert!(!a11.supports_rt());
    }

    // ========================================================================
    // Trait Implementations
    // ========================================================================

    #[test]
    fn test_clone() {
        let original = MetalFeatures::from_device_info("Apple M3 Pro");
        let cloned = original.clone();
        assert_eq!(original, cloned);
    }

    #[test]
    fn test_copy() {
        let original = MetalFeatures::from_device_info("Apple M3 Pro");
        let copied: MetalFeatures = original;
        assert_eq!(original, copied);
    }

    #[test]
    fn test_debug() {
        let features = MetalFeatures::from_device_info("Apple M2");
        let debug_str = format!("{:?}", features);
        assert!(debug_str.contains("MetalFeatures"));
        assert!(debug_str.contains("gpu_family"));
    }

    #[test]
    fn test_eq() {
        let features1 = MetalFeatures::from_device_info("Apple M3");
        let features2 = MetalFeatures::from_device_info("Apple M3");
        let features3 = MetalFeatures::from_device_info("Apple M4");
        assert_eq!(features1, features2);
        assert_ne!(features1, features3);
    }
}

// ============================================================================
// Module: Integration Tests
// ============================================================================

mod integration {
    use super::*;

    #[test]
    fn test_full_detection_chain_m3_max() {
        let device_name = "Apple M3 Max";

        // GPU family detection
        let family = MetalGpuFamily::from_device_name(device_name);
        assert_eq!(family, MetalGpuFamily::Apple9);
        assert!(family.is_apple_silicon());
        assert!(family.supports_metal3());
        assert!(family.supports_ray_tracing());
        assert!(family.supports_mesh_shaders());

        // Silicon generation detection
        let gen = AppleSiliconGeneration::from_device_name(device_name);
        assert_eq!(gen, AppleSiliconGeneration::M3Max);
        assert!(gen.is_m_series());
        assert!(gen.is_pro_tier());
        assert_eq!(gen.generation_number(), 3);
        assert_eq!(gen.estimated_gpu_cores(), 40);

        // Full features
        let features = MetalFeatures::from_device_info(device_name);
        assert_eq!(features.gpu_family, family);
        assert_eq!(features.silicon_generation, gen);
        assert!(features.supports_rt());
        assert!(features.supports_mesh_shaders());
        assert!(features.supports_bindless());
        assert_eq!(features.minimum_macos_version(), (14, 0));
    }

    #[test]
    fn test_full_detection_chain_intel() {
        let device_name = "Intel UHD Graphics 630";

        // GPU family detection
        let family = MetalGpuFamily::from_device_name(device_name);
        assert_eq!(family, MetalGpuFamily::Mac2);
        assert!(!family.is_apple_silicon());
        assert!(family.is_intel_mac());
        assert!(!family.supports_metal3());

        // Silicon generation detection
        let gen = AppleSiliconGeneration::from_device_name(device_name);
        assert_eq!(gen, AppleSiliconGeneration::Unknown);

        // Full features
        let features = MetalFeatures::from_device_info(device_name);
        assert_eq!(features.gpu_family, family);
        assert_eq!(features.silicon_generation, gen);
        assert!(!features.supports_rt());
        assert!(!features.supports_mesh_shaders());
    }

    #[test]
    fn test_case_insensitivity_consistency() {
        let variants = [
            "Apple M3 Pro",
            "apple m3 pro",
            "APPLE M3 PRO",
            "ApPlE m3 PrO",
        ];

        let reference_family = MetalGpuFamily::from_device_name(variants[0]);
        let reference_gen = AppleSiliconGeneration::from_device_name(variants[0]);
        let reference_features = MetalFeatures::from_device_info(variants[0]);

        for variant in &variants[1..] {
            assert_eq!(
                MetalGpuFamily::from_device_name(variant),
                reference_family,
                "Family mismatch for: {}",
                variant
            );
            assert_eq!(
                AppleSiliconGeneration::from_device_name(variant),
                reference_gen,
                "Generation mismatch for: {}",
                variant
            );
            assert_eq!(
                MetalFeatures::from_device_info(variant),
                reference_features,
                "Features mismatch for: {}",
                variant
            );
        }
    }

    #[test]
    fn test_all_m_series_detected_correctly() {
        let m_series = [
            ("Apple M1", AppleSiliconGeneration::M1, MetalGpuFamily::Apple7),
            ("Apple M1 Pro", AppleSiliconGeneration::M1Pro, MetalGpuFamily::Apple7),
            ("Apple M1 Max", AppleSiliconGeneration::M1Max, MetalGpuFamily::Apple7),
            ("Apple M1 Ultra", AppleSiliconGeneration::M1Ultra, MetalGpuFamily::Apple7),
            ("Apple M2", AppleSiliconGeneration::M2, MetalGpuFamily::Apple8),
            ("Apple M2 Pro", AppleSiliconGeneration::M2Pro, MetalGpuFamily::Apple8),
            ("Apple M2 Max", AppleSiliconGeneration::M2Max, MetalGpuFamily::Apple8),
            ("Apple M2 Ultra", AppleSiliconGeneration::M2Ultra, MetalGpuFamily::Apple8),
            ("Apple M3", AppleSiliconGeneration::M3, MetalGpuFamily::Apple9),
            ("Apple M3 Pro", AppleSiliconGeneration::M3Pro, MetalGpuFamily::Apple9),
            ("Apple M3 Max", AppleSiliconGeneration::M3Max, MetalGpuFamily::Apple9),
            ("Apple M4", AppleSiliconGeneration::M4, MetalGpuFamily::Apple9),
            ("Apple M4 Pro", AppleSiliconGeneration::M4Pro, MetalGpuFamily::Apple9),
            ("Apple M4 Max", AppleSiliconGeneration::M4Max, MetalGpuFamily::Apple9),
        ];

        for (name, expected_gen, expected_family) in m_series {
            let gen = AppleSiliconGeneration::from_device_name(name);
            let family = MetalGpuFamily::from_device_name(name);
            assert_eq!(gen, expected_gen, "Generation mismatch for: {}", name);
            assert_eq!(family, expected_family, "Family mismatch for: {}", name);
        }
    }

    #[test]
    fn test_all_a_series_detected_correctly() {
        let a_series = [
            ("Apple A7", MetalGpuFamily::Apple1),
            ("Apple A8", MetalGpuFamily::Apple2),
            ("Apple A9", MetalGpuFamily::Apple3),
            ("Apple A10", MetalGpuFamily::Apple3),
            ("Apple A11 Bionic", MetalGpuFamily::Apple4),
            ("Apple A12 Bionic", MetalGpuFamily::Apple5),
            ("Apple A13 Bionic", MetalGpuFamily::Apple6),
            ("Apple A14 Bionic", MetalGpuFamily::Apple7),
            ("Apple A15 Bionic", MetalGpuFamily::Apple8),
            ("Apple A16 Bionic", MetalGpuFamily::Apple9),
            ("Apple A17 Pro", MetalGpuFamily::Apple9),
        ];

        for (name, expected_family) in a_series {
            let family = MetalGpuFamily::from_device_name(name);
            assert_eq!(family, expected_family, "Family mismatch for: {}", name);
        }
    }

    #[test]
    fn test_feature_tier_consistency() {
        // All Apple7+ families should have Metal 3 features
        let metal3_families = [
            MetalGpuFamily::Apple7,
            MetalGpuFamily::Apple8,
            MetalGpuFamily::Apple9,
            MetalGpuFamily::Metal3,
        ];

        for family in metal3_families {
            assert!(
                family.supports_metal3(),
                "{:?} should support Metal3",
                family
            );
            assert!(
                family.supports_ray_tracing(),
                "{:?} should support ray tracing",
                family
            );
            assert!(
                family.supports_mesh_shaders(),
                "{:?} should support mesh shaders",
                family
            );
            assert!(
                family.is_apple_silicon(),
                "{:?} should be Apple Silicon",
                family
            );
        }
    }

    #[test]
    fn test_gpu_core_estimates_increase_with_tier() {
        // Within each generation, cores should increase: base < Pro < Max < Ultra
        assert!(
            AppleSiliconGeneration::M1.estimated_gpu_cores()
                < AppleSiliconGeneration::M1Pro.estimated_gpu_cores()
        );
        assert!(
            AppleSiliconGeneration::M1Pro.estimated_gpu_cores()
                < AppleSiliconGeneration::M1Max.estimated_gpu_cores()
        );
        assert!(
            AppleSiliconGeneration::M1Max.estimated_gpu_cores()
                < AppleSiliconGeneration::M1Ultra.estimated_gpu_cores()
        );
    }
}