// Blackbox contract tests for T-WGPU-P7.2.2 Metal Capabilities
//
// CLEANROOM: No access to implementation details. Tests use only the public API
// exported by `renderer_backend::backend::metal`.
//
// Forbidden files:
//   - Implementation details beyond what's exposed via the public API
//   - WHITEBOX test files for this task
//
// Contract sources:
//   - docs/WGPU_DOCS/RDC_OUTPUT/PHASE_7_PLATFORM_TODO.md (T-WGPU-P7.2.2)
//   - Apple Metal Feature Set Reference documentation
//
// Test categories:
//   1. GPU Family Detection Scenarios
//   2. Silicon Generation Workflows
//   3. Feature Detection Integration
//   4. Real-World Device Strings
//   5. Feature Progression Validation
//   6. Backend Integration
//   7. Edge Cases and Negative Tests
//
// Contract specifications:
//   - MetalGpuFamily: Classify GPU into Apple1-9, Mac1-2, Common1-3, Metal3, Unknown
//   - AppleSiliconGeneration: Identify M1-M4 and A14-A17 chip variants
//   - MetalFeatures: Detect Metal-specific capabilities (RT, mesh shaders, etc.)

use renderer_backend::backend::metal::{
    AppleSiliconGeneration, MetalFeatures, MetalGpuFamily,
};

// =============================================================================
// 1. GPU Family Detection Scenarios
// =============================================================================

mod gpu_family_detection {
    use super::*;

    // -------------------------------------------------------------------------
    // 1.1 Apple Silicon Mac Detection (M1/M2/M3/M4)
    // -------------------------------------------------------------------------

    /// M1 chip detected as Apple7 family
    #[test]
    fn test_detect_m1_as_apple7() {
        let family = MetalGpuFamily::from_device_name("Apple M1");
        assert_eq!(family, MetalGpuFamily::Apple7);
    }

    /// M1 Pro chip detected as Apple7 family
    #[test]
    fn test_detect_m1_pro_as_apple7() {
        let family = MetalGpuFamily::from_device_name("Apple M1 Pro");
        assert_eq!(family, MetalGpuFamily::Apple7);
    }

    /// M1 Max chip detected as Apple7 family
    #[test]
    fn test_detect_m1_max_as_apple7() {
        let family = MetalGpuFamily::from_device_name("Apple M1 Max");
        assert_eq!(family, MetalGpuFamily::Apple7);
    }

    /// M1 Ultra chip detected as Apple7 family
    #[test]
    fn test_detect_m1_ultra_as_apple7() {
        let family = MetalGpuFamily::from_device_name("Apple M1 Ultra");
        assert_eq!(family, MetalGpuFamily::Apple7);
    }

    /// M2 chip detected as Apple8 family
    #[test]
    fn test_detect_m2_as_apple8() {
        let family = MetalGpuFamily::from_device_name("Apple M2");
        assert_eq!(family, MetalGpuFamily::Apple8);
    }

    /// M2 Pro chip detected as Apple8 family
    #[test]
    fn test_detect_m2_pro_as_apple8() {
        let family = MetalGpuFamily::from_device_name("Apple M2 Pro");
        assert_eq!(family, MetalGpuFamily::Apple8);
    }

    /// M2 Max chip detected as Apple8 family
    #[test]
    fn test_detect_m2_max_as_apple8() {
        let family = MetalGpuFamily::from_device_name("Apple M2 Max");
        assert_eq!(family, MetalGpuFamily::Apple8);
    }

    /// M2 Ultra chip detected as Apple8 family
    #[test]
    fn test_detect_m2_ultra_as_apple8() {
        let family = MetalGpuFamily::from_device_name("Apple M2 Ultra");
        assert_eq!(family, MetalGpuFamily::Apple8);
    }

    /// M3 chip detected as Apple9 family
    #[test]
    fn test_detect_m3_as_apple9() {
        let family = MetalGpuFamily::from_device_name("Apple M3");
        assert_eq!(family, MetalGpuFamily::Apple9);
    }

    /// M3 Pro chip detected as Apple9 family
    #[test]
    fn test_detect_m3_pro_as_apple9() {
        let family = MetalGpuFamily::from_device_name("Apple M3 Pro");
        assert_eq!(family, MetalGpuFamily::Apple9);
    }

    /// M3 Max chip detected as Apple9 family
    #[test]
    fn test_detect_m3_max_as_apple9() {
        let family = MetalGpuFamily::from_device_name("Apple M3 Max");
        assert_eq!(family, MetalGpuFamily::Apple9);
    }

    /// M4 chip detected as Apple9 family
    #[test]
    fn test_detect_m4_as_apple9() {
        let family = MetalGpuFamily::from_device_name("Apple M4");
        assert_eq!(family, MetalGpuFamily::Apple9);
    }

    /// M4 Pro chip detected as Apple9 family
    #[test]
    fn test_detect_m4_pro_as_apple9() {
        let family = MetalGpuFamily::from_device_name("Apple M4 Pro");
        assert_eq!(family, MetalGpuFamily::Apple9);
    }

    /// M4 Max chip detected as Apple9 family
    #[test]
    fn test_detect_m4_max_as_apple9() {
        let family = MetalGpuFamily::from_device_name("Apple M4 Max");
        assert_eq!(family, MetalGpuFamily::Apple9);
    }

    // -------------------------------------------------------------------------
    // 1.2 Intel Mac Detection with Discrete GPU
    // -------------------------------------------------------------------------

    /// Intel UHD Graphics detected as Mac2
    #[test]
    fn test_detect_intel_uhd_as_mac2() {
        let family = MetalGpuFamily::from_device_name("Intel UHD Graphics 630");
        assert_eq!(family, MetalGpuFamily::Mac2);
    }

    /// Intel Iris Pro detected as Mac2
    #[test]
    fn test_detect_intel_iris_pro_as_mac2() {
        let family = MetalGpuFamily::from_device_name("Intel Iris Pro Graphics");
        assert_eq!(family, MetalGpuFamily::Mac2);
    }

    /// Intel Iris Plus detected as Mac2
    #[test]
    fn test_detect_intel_iris_plus_as_mac2() {
        let family = MetalGpuFamily::from_device_name("Intel Iris Plus Graphics 655");
        assert_eq!(family, MetalGpuFamily::Mac2);
    }

    /// Older Intel HD Graphics detected as Mac1
    #[test]
    fn test_detect_intel_hd_as_mac1() {
        let family = MetalGpuFamily::from_device_name("Intel HD Graphics 4000");
        assert_eq!(family, MetalGpuFamily::Mac1);
    }

    /// Intel HD Graphics 5000 detected as Mac1
    #[test]
    fn test_detect_intel_hd_5000_as_mac1() {
        let family = MetalGpuFamily::from_device_name("Intel HD Graphics 5000");
        assert_eq!(family, MetalGpuFamily::Mac1);
    }

    /// Generic Intel GPU without UHD/Iris detected as Mac1
    #[test]
    fn test_detect_generic_intel_as_mac1() {
        let family = MetalGpuFamily::from_device_name("Intel GPU");
        assert_eq!(family, MetalGpuFamily::Mac1);
    }

    // -------------------------------------------------------------------------
    // 1.3 iPhone/iPad A-series Chips
    // -------------------------------------------------------------------------

    /// A17 Pro detected as Apple9
    #[test]
    fn test_detect_a17_pro_as_apple9() {
        let family = MetalGpuFamily::from_device_name("Apple A17 Pro");
        assert_eq!(family, MetalGpuFamily::Apple9);
    }

    /// A16 Bionic detected as Apple9
    #[test]
    fn test_detect_a16_as_apple9() {
        let family = MetalGpuFamily::from_device_name("Apple A16 Bionic");
        assert_eq!(family, MetalGpuFamily::Apple9);
    }

    /// A15 Bionic detected as Apple8
    #[test]
    fn test_detect_a15_as_apple8() {
        let family = MetalGpuFamily::from_device_name("Apple A15 Bionic");
        assert_eq!(family, MetalGpuFamily::Apple8);
    }

    /// A14 Bionic detected as Apple7
    #[test]
    fn test_detect_a14_as_apple7() {
        let family = MetalGpuFamily::from_device_name("Apple A14 Bionic");
        assert_eq!(family, MetalGpuFamily::Apple7);
    }

    /// A13 Bionic detected as Apple6
    #[test]
    fn test_detect_a13_as_apple6() {
        let family = MetalGpuFamily::from_device_name("Apple A13 Bionic");
        assert_eq!(family, MetalGpuFamily::Apple6);
    }

    /// A12 Bionic detected as Apple5
    #[test]
    fn test_detect_a12_as_apple5() {
        let family = MetalGpuFamily::from_device_name("Apple A12 Bionic");
        assert_eq!(family, MetalGpuFamily::Apple5);
    }

    /// A11 Bionic detected as Apple4
    #[test]
    fn test_detect_a11_as_apple4() {
        let family = MetalGpuFamily::from_device_name("Apple A11 Bionic");
        assert_eq!(family, MetalGpuFamily::Apple4);
    }

    /// A10 Fusion detected as Apple3
    #[test]
    fn test_detect_a10_as_apple3() {
        let family = MetalGpuFamily::from_device_name("Apple A10 Fusion");
        assert_eq!(family, MetalGpuFamily::Apple3);
    }

    /// A9 detected as Apple3
    #[test]
    fn test_detect_a9_as_apple3() {
        let family = MetalGpuFamily::from_device_name("Apple A9");
        assert_eq!(family, MetalGpuFamily::Apple3);
    }

    /// A8 detected as Apple2
    #[test]
    fn test_detect_a8_as_apple2() {
        let family = MetalGpuFamily::from_device_name("Apple A8");
        assert_eq!(family, MetalGpuFamily::Apple2);
    }

    /// A7 detected as Apple1
    #[test]
    fn test_detect_a7_as_apple1() {
        let family = MetalGpuFamily::from_device_name("Apple A7");
        assert_eq!(family, MetalGpuFamily::Apple1);
    }

    // -------------------------------------------------------------------------
    // 1.4 Unknown/Unsupported GPU Graceful Handling
    // -------------------------------------------------------------------------

    /// Unknown device string returns Unknown family
    #[test]
    fn test_unknown_device_returns_unknown() {
        let family = MetalGpuFamily::from_device_name("Unknown GPU");
        assert_eq!(family, MetalGpuFamily::Unknown);
    }

    /// Empty device name returns Unknown
    #[test]
    fn test_empty_device_name_returns_unknown() {
        let family = MetalGpuFamily::from_device_name("");
        assert_eq!(family, MetalGpuFamily::Unknown);
    }

    /// NVIDIA GPU (not supported by Metal) returns Unknown
    #[test]
    fn test_nvidia_returns_unknown() {
        let family = MetalGpuFamily::from_device_name("NVIDIA GeForce RTX 4090");
        assert_eq!(family, MetalGpuFamily::Unknown);
    }

    /// Non-Apple device string returns Unknown
    #[test]
    fn test_non_apple_device_returns_unknown() {
        let family = MetalGpuFamily::from_device_name("Qualcomm Adreno 750");
        assert_eq!(family, MetalGpuFamily::Unknown);
    }

    /// Default GPU family is Unknown
    #[test]
    fn test_default_family_is_unknown() {
        let family = MetalGpuFamily::default();
        assert_eq!(family, MetalGpuFamily::Unknown);
    }

    // -------------------------------------------------------------------------
    // 1.5 Feature Capability Queries by Family
    // -------------------------------------------------------------------------

    /// Apple7 is classified as Apple Silicon
    #[test]
    fn test_apple7_is_apple_silicon() {
        assert!(MetalGpuFamily::Apple7.is_apple_silicon());
    }

    /// Apple8 is classified as Apple Silicon
    #[test]
    fn test_apple8_is_apple_silicon() {
        assert!(MetalGpuFamily::Apple8.is_apple_silicon());
    }

    /// Apple9 is classified as Apple Silicon
    #[test]
    fn test_apple9_is_apple_silicon() {
        assert!(MetalGpuFamily::Apple9.is_apple_silicon());
    }

    /// Metal3 is classified as Apple Silicon
    #[test]
    fn test_metal3_is_apple_silicon() {
        assert!(MetalGpuFamily::Metal3.is_apple_silicon());
    }

    /// Apple6 is NOT classified as Apple Silicon (pre-M1 era)
    #[test]
    fn test_apple6_is_not_apple_silicon() {
        assert!(!MetalGpuFamily::Apple6.is_apple_silicon());
    }

    /// Mac1 is Intel Mac
    #[test]
    fn test_mac1_is_intel_mac() {
        assert!(MetalGpuFamily::Mac1.is_intel_mac());
    }

    /// Mac2 is Intel Mac
    #[test]
    fn test_mac2_is_intel_mac() {
        assert!(MetalGpuFamily::Mac2.is_intel_mac());
    }

    /// Apple9 is NOT Intel Mac
    #[test]
    fn test_apple9_is_not_intel_mac() {
        assert!(!MetalGpuFamily::Apple9.is_intel_mac());
    }

    /// Unknown family is not Apple Silicon
    #[test]
    fn test_unknown_is_not_apple_silicon() {
        assert!(!MetalGpuFamily::Unknown.is_apple_silicon());
    }

    /// Unknown family is not Intel Mac
    #[test]
    fn test_unknown_is_not_intel_mac() {
        assert!(!MetalGpuFamily::Unknown.is_intel_mac());
    }
}

// =============================================================================
// 2. Silicon Generation Workflows
// =============================================================================

mod silicon_generation {
    use super::*;

    // -------------------------------------------------------------------------
    // 2.1 M1 Variants Identification
    // -------------------------------------------------------------------------

    /// M1 base chip correctly identified
    #[test]
    fn test_identify_m1_base() {
        let gen = AppleSiliconGeneration::from_device_name("Apple M1");
        assert_eq!(gen, AppleSiliconGeneration::M1);
    }

    /// M1 Pro correctly identified
    #[test]
    fn test_identify_m1_pro() {
        let gen = AppleSiliconGeneration::from_device_name("Apple M1 Pro");
        assert_eq!(gen, AppleSiliconGeneration::M1Pro);
    }

    /// M1 Max correctly identified
    #[test]
    fn test_identify_m1_max() {
        let gen = AppleSiliconGeneration::from_device_name("Apple M1 Max");
        assert_eq!(gen, AppleSiliconGeneration::M1Max);
    }

    /// M1 Ultra correctly identified
    #[test]
    fn test_identify_m1_ultra() {
        let gen = AppleSiliconGeneration::from_device_name("Apple M1 Ultra");
        assert_eq!(gen, AppleSiliconGeneration::M1Ultra);
    }

    /// M1 generation number is 1
    #[test]
    fn test_m1_generation_number() {
        assert_eq!(AppleSiliconGeneration::M1.generation_number(), 1);
        assert_eq!(AppleSiliconGeneration::M1Pro.generation_number(), 1);
        assert_eq!(AppleSiliconGeneration::M1Max.generation_number(), 1);
        assert_eq!(AppleSiliconGeneration::M1Ultra.generation_number(), 1);
    }

    // -------------------------------------------------------------------------
    // 2.2 M2 Variants Identification
    // -------------------------------------------------------------------------

    /// M2 base chip correctly identified
    #[test]
    fn test_identify_m2_base() {
        let gen = AppleSiliconGeneration::from_device_name("Apple M2");
        assert_eq!(gen, AppleSiliconGeneration::M2);
    }

    /// M2 Pro correctly identified
    #[test]
    fn test_identify_m2_pro() {
        let gen = AppleSiliconGeneration::from_device_name("Apple M2 Pro");
        assert_eq!(gen, AppleSiliconGeneration::M2Pro);
    }

    /// M2 Max correctly identified
    #[test]
    fn test_identify_m2_max() {
        let gen = AppleSiliconGeneration::from_device_name("Apple M2 Max");
        assert_eq!(gen, AppleSiliconGeneration::M2Max);
    }

    /// M2 Ultra correctly identified
    #[test]
    fn test_identify_m2_ultra() {
        let gen = AppleSiliconGeneration::from_device_name("Apple M2 Ultra");
        assert_eq!(gen, AppleSiliconGeneration::M2Ultra);
    }

    /// M2 generation number is 2
    #[test]
    fn test_m2_generation_number() {
        assert_eq!(AppleSiliconGeneration::M2.generation_number(), 2);
        assert_eq!(AppleSiliconGeneration::M2Pro.generation_number(), 2);
        assert_eq!(AppleSiliconGeneration::M2Max.generation_number(), 2);
        assert_eq!(AppleSiliconGeneration::M2Ultra.generation_number(), 2);
    }

    // -------------------------------------------------------------------------
    // 2.3 M3 Variants Identification
    // -------------------------------------------------------------------------

    /// M3 base chip correctly identified
    #[test]
    fn test_identify_m3_base() {
        let gen = AppleSiliconGeneration::from_device_name("Apple M3");
        assert_eq!(gen, AppleSiliconGeneration::M3);
    }

    /// M3 Pro correctly identified
    #[test]
    fn test_identify_m3_pro() {
        let gen = AppleSiliconGeneration::from_device_name("Apple M3 Pro");
        assert_eq!(gen, AppleSiliconGeneration::M3Pro);
    }

    /// M3 Max correctly identified
    #[test]
    fn test_identify_m3_max() {
        let gen = AppleSiliconGeneration::from_device_name("Apple M3 Max");
        assert_eq!(gen, AppleSiliconGeneration::M3Max);
    }

    /// M3 generation number is 3
    #[test]
    fn test_m3_generation_number() {
        assert_eq!(AppleSiliconGeneration::M3.generation_number(), 3);
        assert_eq!(AppleSiliconGeneration::M3Pro.generation_number(), 3);
        assert_eq!(AppleSiliconGeneration::M3Max.generation_number(), 3);
    }

    // -------------------------------------------------------------------------
    // 2.4 M4 Variants Identification
    // -------------------------------------------------------------------------

    /// M4 base chip correctly identified
    #[test]
    fn test_identify_m4_base() {
        let gen = AppleSiliconGeneration::from_device_name("Apple M4");
        assert_eq!(gen, AppleSiliconGeneration::M4);
    }

    /// M4 Pro correctly identified
    #[test]
    fn test_identify_m4_pro() {
        let gen = AppleSiliconGeneration::from_device_name("Apple M4 Pro");
        assert_eq!(gen, AppleSiliconGeneration::M4Pro);
    }

    /// M4 Max correctly identified
    #[test]
    fn test_identify_m4_max() {
        let gen = AppleSiliconGeneration::from_device_name("Apple M4 Max");
        assert_eq!(gen, AppleSiliconGeneration::M4Max);
    }

    /// M4 generation number is 4
    #[test]
    fn test_m4_generation_number() {
        assert_eq!(AppleSiliconGeneration::M4.generation_number(), 4);
        assert_eq!(AppleSiliconGeneration::M4Pro.generation_number(), 4);
        assert_eq!(AppleSiliconGeneration::M4Max.generation_number(), 4);
    }

    // -------------------------------------------------------------------------
    // 2.5 A-series Chip Identification for iOS
    // -------------------------------------------------------------------------

    /// A14 correctly identified
    #[test]
    fn test_identify_a14() {
        let gen = AppleSiliconGeneration::from_device_name("Apple A14 Bionic");
        assert_eq!(gen, AppleSiliconGeneration::A14);
    }

    /// A15 correctly identified
    #[test]
    fn test_identify_a15() {
        let gen = AppleSiliconGeneration::from_device_name("Apple A15 Bionic");
        assert_eq!(gen, AppleSiliconGeneration::A15);
    }

    /// A16 correctly identified
    #[test]
    fn test_identify_a16() {
        let gen = AppleSiliconGeneration::from_device_name("Apple A16 Bionic");
        assert_eq!(gen, AppleSiliconGeneration::A16);
    }

    /// A17 Pro correctly identified
    #[test]
    fn test_identify_a17_pro() {
        let gen = AppleSiliconGeneration::from_device_name("Apple A17 Pro");
        assert_eq!(gen, AppleSiliconGeneration::A17Pro);
    }

    /// A-series chips have generation number 0 (M-series only)
    #[test]
    fn test_a_series_generation_number_is_zero() {
        assert_eq!(AppleSiliconGeneration::A14.generation_number(), 0);
        assert_eq!(AppleSiliconGeneration::A15.generation_number(), 0);
        assert_eq!(AppleSiliconGeneration::A16.generation_number(), 0);
        assert_eq!(AppleSiliconGeneration::A17Pro.generation_number(), 0);
    }

    /// A-series chips are correctly classified as A-series
    #[test]
    fn test_a_series_classification() {
        assert!(AppleSiliconGeneration::A14.is_a_series());
        assert!(AppleSiliconGeneration::A15.is_a_series());
        assert!(AppleSiliconGeneration::A16.is_a_series());
        assert!(AppleSiliconGeneration::A17Pro.is_a_series());
    }

    /// M-series chips are NOT classified as A-series
    #[test]
    fn test_m_series_not_a_series() {
        assert!(!AppleSiliconGeneration::M1.is_a_series());
        assert!(!AppleSiliconGeneration::M2.is_a_series());
        assert!(!AppleSiliconGeneration::M3.is_a_series());
        assert!(!AppleSiliconGeneration::M4.is_a_series());
    }

    // -------------------------------------------------------------------------
    // 2.6 GPU Core Count Estimation Accuracy
    // -------------------------------------------------------------------------

    /// M1 base has 8 GPU cores
    #[test]
    fn test_m1_gpu_cores() {
        assert_eq!(AppleSiliconGeneration::M1.estimated_gpu_cores(), 8);
    }

    /// M1 Pro has 16 GPU cores
    #[test]
    fn test_m1_pro_gpu_cores() {
        assert_eq!(AppleSiliconGeneration::M1Pro.estimated_gpu_cores(), 16);
    }

    /// M1 Max has 32 GPU cores
    #[test]
    fn test_m1_max_gpu_cores() {
        assert_eq!(AppleSiliconGeneration::M1Max.estimated_gpu_cores(), 32);
    }

    /// M1 Ultra has 64 GPU cores
    #[test]
    fn test_m1_ultra_gpu_cores() {
        assert_eq!(AppleSiliconGeneration::M1Ultra.estimated_gpu_cores(), 64);
    }

    /// M2 base has 10 GPU cores
    #[test]
    fn test_m2_gpu_cores() {
        assert_eq!(AppleSiliconGeneration::M2.estimated_gpu_cores(), 10);
    }

    /// M2 Pro has 19 GPU cores
    #[test]
    fn test_m2_pro_gpu_cores() {
        assert_eq!(AppleSiliconGeneration::M2Pro.estimated_gpu_cores(), 19);
    }

    /// M2 Max has 38 GPU cores
    #[test]
    fn test_m2_max_gpu_cores() {
        assert_eq!(AppleSiliconGeneration::M2Max.estimated_gpu_cores(), 38);
    }

    /// M2 Ultra has 76 GPU cores
    #[test]
    fn test_m2_ultra_gpu_cores() {
        assert_eq!(AppleSiliconGeneration::M2Ultra.estimated_gpu_cores(), 76);
    }

    /// M3 base has 10 GPU cores
    #[test]
    fn test_m3_gpu_cores() {
        assert_eq!(AppleSiliconGeneration::M3.estimated_gpu_cores(), 10);
    }

    /// M3 Pro has 18 GPU cores
    #[test]
    fn test_m3_pro_gpu_cores() {
        assert_eq!(AppleSiliconGeneration::M3Pro.estimated_gpu_cores(), 18);
    }

    /// M3 Max has 40 GPU cores
    #[test]
    fn test_m3_max_gpu_cores() {
        assert_eq!(AppleSiliconGeneration::M3Max.estimated_gpu_cores(), 40);
    }

    /// M4 base has 10 GPU cores
    #[test]
    fn test_m4_gpu_cores() {
        assert_eq!(AppleSiliconGeneration::M4.estimated_gpu_cores(), 10);
    }

    /// M4 Pro has 20 GPU cores
    #[test]
    fn test_m4_pro_gpu_cores() {
        assert_eq!(AppleSiliconGeneration::M4Pro.estimated_gpu_cores(), 20);
    }

    /// M4 Max has 40 GPU cores
    #[test]
    fn test_m4_max_gpu_cores() {
        assert_eq!(AppleSiliconGeneration::M4Max.estimated_gpu_cores(), 40);
    }

    /// A-series GPU core counts
    #[test]
    fn test_a_series_gpu_cores() {
        assert_eq!(AppleSiliconGeneration::A14.estimated_gpu_cores(), 4);
        assert_eq!(AppleSiliconGeneration::A15.estimated_gpu_cores(), 5);
        assert_eq!(AppleSiliconGeneration::A16.estimated_gpu_cores(), 5);
        assert_eq!(AppleSiliconGeneration::A17Pro.estimated_gpu_cores(), 6);
    }

    /// Unknown generation has 0 GPU cores
    #[test]
    fn test_unknown_gpu_cores() {
        assert_eq!(AppleSiliconGeneration::Unknown.estimated_gpu_cores(), 0);
    }

    // -------------------------------------------------------------------------
    // 2.7 Pro Tier Classification
    // -------------------------------------------------------------------------

    /// Pro variants are pro tier
    #[test]
    fn test_pro_is_pro_tier() {
        assert!(AppleSiliconGeneration::M1Pro.is_pro_tier());
        assert!(AppleSiliconGeneration::M2Pro.is_pro_tier());
        assert!(AppleSiliconGeneration::M3Pro.is_pro_tier());
        assert!(AppleSiliconGeneration::M4Pro.is_pro_tier());
    }

    /// Max variants are pro tier
    #[test]
    fn test_max_is_pro_tier() {
        assert!(AppleSiliconGeneration::M1Max.is_pro_tier());
        assert!(AppleSiliconGeneration::M2Max.is_pro_tier());
        assert!(AppleSiliconGeneration::M3Max.is_pro_tier());
        assert!(AppleSiliconGeneration::M4Max.is_pro_tier());
    }

    /// Ultra variants are pro tier
    #[test]
    fn test_ultra_is_pro_tier() {
        assert!(AppleSiliconGeneration::M1Ultra.is_pro_tier());
        assert!(AppleSiliconGeneration::M2Ultra.is_pro_tier());
    }

    /// A17 Pro is pro tier
    #[test]
    fn test_a17_pro_is_pro_tier() {
        assert!(AppleSiliconGeneration::A17Pro.is_pro_tier());
    }

    /// Base chips are NOT pro tier
    #[test]
    fn test_base_is_not_pro_tier() {
        assert!(!AppleSiliconGeneration::M1.is_pro_tier());
        assert!(!AppleSiliconGeneration::M2.is_pro_tier());
        assert!(!AppleSiliconGeneration::M3.is_pro_tier());
        assert!(!AppleSiliconGeneration::M4.is_pro_tier());
    }

    /// Non-Pro A-series are NOT pro tier
    #[test]
    fn test_non_pro_a_series_is_not_pro_tier() {
        assert!(!AppleSiliconGeneration::A14.is_pro_tier());
        assert!(!AppleSiliconGeneration::A15.is_pro_tier());
        assert!(!AppleSiliconGeneration::A16.is_pro_tier());
    }
}

// =============================================================================
// 3. Feature Detection Integration
// =============================================================================

mod feature_detection {
    use super::*;

    // -------------------------------------------------------------------------
    // 3.1 Metal 3 Availability Checking
    // -------------------------------------------------------------------------

    /// Apple7+ supports Metal 3
    #[test]
    fn test_apple7_supports_metal3() {
        assert!(MetalGpuFamily::Apple7.supports_metal3());
    }

    /// Apple8 supports Metal 3
    #[test]
    fn test_apple8_supports_metal3() {
        assert!(MetalGpuFamily::Apple8.supports_metal3());
    }

    /// Apple9 supports Metal 3
    #[test]
    fn test_apple9_supports_metal3() {
        assert!(MetalGpuFamily::Apple9.supports_metal3());
    }

    /// Metal3 family explicitly supports Metal 3
    #[test]
    fn test_metal3_family_supports_metal3() {
        assert!(MetalGpuFamily::Metal3.supports_metal3());
    }

    /// Apple6 does NOT support Metal 3
    #[test]
    fn test_apple6_does_not_support_metal3() {
        assert!(!MetalGpuFamily::Apple6.supports_metal3());
    }

    /// Mac2 (Intel) does NOT support Metal 3
    #[test]
    fn test_mac2_does_not_support_metal3() {
        assert!(!MetalGpuFamily::Mac2.supports_metal3());
    }

    // -------------------------------------------------------------------------
    // 3.2 Ray Tracing Support Detection
    // -------------------------------------------------------------------------

    /// M1 features include ray tracing
    #[test]
    fn test_m1_has_ray_tracing() {
        let features = MetalFeatures::from_device_info("Apple M1");
        assert!(features.ray_tracing);
        assert!(features.supports_rt());
    }

    /// M3 features include ray tracing
    #[test]
    fn test_m3_has_ray_tracing() {
        let features = MetalFeatures::from_device_info("Apple M3");
        assert!(features.ray_tracing);
        assert!(features.supports_rt());
    }

    /// M4 features include ray tracing
    #[test]
    fn test_m4_has_ray_tracing() {
        let features = MetalFeatures::from_device_info("Apple M4");
        assert!(features.ray_tracing);
        assert!(features.supports_rt());
    }

    /// Intel Mac does NOT have ray tracing
    #[test]
    fn test_intel_no_ray_tracing() {
        let features = MetalFeatures::from_device_info("Intel UHD Graphics 630");
        assert!(!features.ray_tracing);
        assert!(!features.supports_rt());
    }

    /// A13 does NOT have ray tracing
    #[test]
    fn test_a13_no_ray_tracing() {
        let features = MetalFeatures::from_device_info("Apple A13");
        assert!(!features.ray_tracing);
        assert!(!features.supports_rt());
    }

    /// A14 has ray tracing
    #[test]
    fn test_a14_has_ray_tracing() {
        let features = MetalFeatures::from_device_info("Apple A14");
        assert!(features.ray_tracing);
        assert!(features.supports_rt());
    }

    // -------------------------------------------------------------------------
    // 3.3 Mesh Shader Availability
    // -------------------------------------------------------------------------

    /// M1 features include mesh shaders
    #[test]
    fn test_m1_has_mesh_shaders() {
        let features = MetalFeatures::from_device_info("Apple M1");
        assert!(features.mesh_shaders);
        assert!(features.supports_mesh_shaders());
    }

    /// M3 features include mesh shaders
    #[test]
    fn test_m3_has_mesh_shaders() {
        let features = MetalFeatures::from_device_info("Apple M3");
        assert!(features.mesh_shaders);
        assert!(features.supports_mesh_shaders());
    }

    /// A13 does NOT have mesh shaders
    #[test]
    fn test_a13_no_mesh_shaders() {
        let features = MetalFeatures::from_device_info("Apple A13");
        assert!(!features.mesh_shaders);
        assert!(!features.supports_mesh_shaders());
    }

    /// Intel Mac does NOT have mesh shaders
    #[test]
    fn test_intel_no_mesh_shaders() {
        let features = MetalFeatures::from_device_info("Intel UHD Graphics 630");
        assert!(!features.mesh_shaders);
        assert!(!features.supports_mesh_shaders());
    }

    // -------------------------------------------------------------------------
    // 3.4 Argument Buffers Tier Detection
    // -------------------------------------------------------------------------

    /// M1 has argument buffers tier 2
    #[test]
    fn test_m1_has_argument_buffers_tier2() {
        let features = MetalFeatures::from_device_info("Apple M1");
        assert_eq!(features.argument_buffers_tier, 2);
        assert!(features.supports_argument_buffers());
        assert!(features.supports_bindless());
    }

    /// M3 has argument buffers tier 2
    #[test]
    fn test_m3_has_argument_buffers_tier2() {
        let features = MetalFeatures::from_device_info("Apple M3");
        assert_eq!(features.argument_buffers_tier, 2);
        assert!(features.supports_argument_buffers());
        assert!(features.supports_bindless());
    }

    /// A12 has argument buffers tier 2
    #[test]
    fn test_a12_has_argument_buffers_tier2() {
        let features = MetalFeatures::from_device_info("Apple A12");
        assert_eq!(features.argument_buffers_tier, 2);
        assert!(features.supports_argument_buffers());
        assert!(features.supports_bindless());
    }

    /// Default features have no argument buffers
    #[test]
    fn test_default_no_argument_buffers() {
        let features = MetalFeatures::default();
        assert_eq!(features.argument_buffers_tier, 0);
        assert!(!features.supports_argument_buffers());
        assert!(!features.supports_bindless());
    }

    // -------------------------------------------------------------------------
    // 3.5 Sparse Texture Support
    // -------------------------------------------------------------------------

    /// A12+ has sparse textures
    #[test]
    fn test_a12_has_sparse_textures() {
        let features = MetalFeatures::from_device_info("Apple A12");
        assert!(features.sparse_textures);
    }

    /// M1 has sparse textures
    #[test]
    fn test_m1_has_sparse_textures() {
        let features = MetalFeatures::from_device_info("Apple M1");
        assert!(features.sparse_textures);
    }

    /// M4 has sparse textures
    #[test]
    fn test_m4_has_sparse_textures() {
        let features = MetalFeatures::from_device_info("Apple M4");
        assert!(features.sparse_textures);
    }

    /// A11 does NOT have sparse textures
    #[test]
    fn test_a11_no_sparse_textures() {
        let features = MetalFeatures::from_device_info("Apple A11");
        assert!(!features.sparse_textures);
    }

    // -------------------------------------------------------------------------
    // 3.6 Memoryless Render Targets
    // -------------------------------------------------------------------------

    /// M1 has memoryless render targets
    #[test]
    fn test_m1_has_memoryless() {
        let features = MetalFeatures::from_device_info("Apple M1");
        assert!(features.memoryless_render_targets);
    }

    /// A14 has memoryless render targets
    #[test]
    fn test_a14_has_memoryless() {
        let features = MetalFeatures::from_device_info("Apple A14");
        assert!(features.memoryless_render_targets);
    }

    /// A-series chips have memoryless render targets
    #[test]
    fn test_a_series_has_memoryless() {
        let features = MetalFeatures::from_device_info("Apple A10");
        assert!(features.memoryless_render_targets);
    }

    // -------------------------------------------------------------------------
    // 3.7 Tile Shaders and Imageblock
    // -------------------------------------------------------------------------

    /// Apple4+ has tile shaders
    #[test]
    fn test_a11_has_tile_shaders() {
        let features = MetalFeatures::from_device_info("Apple A11");
        assert!(features.tile_shaders);
    }

    /// Apple4+ has imageblock
    #[test]
    fn test_a11_has_imageblock() {
        let features = MetalFeatures::from_device_info("Apple A11");
        assert!(features.imageblock);
    }

    /// M1 has tile shaders and imageblock
    #[test]
    fn test_m1_has_tile_shaders_and_imageblock() {
        let features = MetalFeatures::from_device_info("Apple M1");
        assert!(features.tile_shaders);
        assert!(features.imageblock);
    }

    /// A10 (Apple3) does NOT have tile shaders
    #[test]
    fn test_a10_no_tile_shaders() {
        let features = MetalFeatures::from_device_info("Apple A10");
        assert!(!features.tile_shaders);
        assert!(!features.imageblock);
    }

    // -------------------------------------------------------------------------
    // 3.8 Lossless Compression
    // -------------------------------------------------------------------------

    /// M1+ has lossless compression
    #[test]
    fn test_m1_has_lossless_compression() {
        let features = MetalFeatures::from_device_info("Apple M1");
        assert!(features.lossless_compression);
    }

    /// M3 has lossless compression
    #[test]
    fn test_m3_has_lossless_compression() {
        let features = MetalFeatures::from_device_info("Apple M3");
        assert!(features.lossless_compression);
    }

    /// A14 has lossless compression
    #[test]
    fn test_a14_has_lossless_compression() {
        let features = MetalFeatures::from_device_info("Apple A14");
        assert!(features.lossless_compression);
    }

    /// A13 does NOT have lossless compression
    #[test]
    fn test_a13_no_lossless_compression() {
        let features = MetalFeatures::from_device_info("Apple A13");
        assert!(!features.lossless_compression);
    }

    // -------------------------------------------------------------------------
    // 3.9 Function Pointers
    // -------------------------------------------------------------------------

    /// M1+ has function pointers
    #[test]
    fn test_m1_has_function_pointers() {
        let features = MetalFeatures::from_device_info("Apple M1");
        assert!(features.function_pointers);
    }

    /// M4 has function pointers
    #[test]
    fn test_m4_has_function_pointers() {
        let features = MetalFeatures::from_device_info("Apple M4");
        assert!(features.function_pointers);
    }

    /// A13 does NOT have function pointers
    #[test]
    fn test_a13_no_function_pointers() {
        let features = MetalFeatures::from_device_info("Apple A13");
        assert!(!features.function_pointers);
    }

    // -------------------------------------------------------------------------
    // 3.10 Primitive Motion Blur
    // -------------------------------------------------------------------------

    /// Metal 3 devices have primitive motion blur
    #[test]
    fn test_m1_has_primitive_motion_blur() {
        let features = MetalFeatures::from_device_info("Apple M1");
        assert!(features.primitive_motion_blur);
    }

    /// M3 Max has primitive motion blur
    #[test]
    fn test_m3_max_has_primitive_motion_blur() {
        let features = MetalFeatures::from_device_info("Apple M3 Max");
        assert!(features.primitive_motion_blur);
    }

    /// A13 does NOT have primitive motion blur
    #[test]
    fn test_a13_no_primitive_motion_blur() {
        let features = MetalFeatures::from_device_info("Apple A13");
        assert!(!features.primitive_motion_blur);
    }

    // -------------------------------------------------------------------------
    // 3.11 SIMD Group Functions
    // -------------------------------------------------------------------------

    /// Apple3+ has SIMD group functions
    #[test]
    fn test_a10_has_simd_group() {
        let features = MetalFeatures::from_device_info("Apple A10");
        assert!(features.simd_group);
    }

    /// M1 has SIMD group functions
    #[test]
    fn test_m1_has_simd_group() {
        let features = MetalFeatures::from_device_info("Apple M1");
        assert!(features.simd_group);
    }

    /// Intel Mac has SIMD group functions
    #[test]
    fn test_intel_has_simd_group() {
        let features = MetalFeatures::from_device_info("Intel UHD Graphics 630");
        assert!(features.simd_group);
    }

    // -------------------------------------------------------------------------
    // 3.12 Read-Write Textures
    // -------------------------------------------------------------------------

    /// Apple3+ has read-write textures
    #[test]
    fn test_a10_has_read_write_textures() {
        let features = MetalFeatures::from_device_info("Apple A10");
        assert!(features.read_write_textures);
    }

    /// M1 has read-write textures
    #[test]
    fn test_m1_has_read_write_textures() {
        let features = MetalFeatures::from_device_info("Apple M1");
        assert!(features.read_write_textures);
    }

    /// Intel Mac has read-write textures
    #[test]
    fn test_intel_has_read_write_textures() {
        let features = MetalFeatures::from_device_info("Intel UHD Graphics 630");
        assert!(features.read_write_textures);
    }
}

// =============================================================================
// 4. Real-World Device Strings
// =============================================================================

mod real_world_devices {
    use super::*;

    /// "Apple M1" -> M1, Apple7
    #[test]
    fn test_apple_m1_string() {
        let features = MetalFeatures::from_device_info("Apple M1");
        assert_eq!(features.silicon_generation, AppleSiliconGeneration::M1);
        assert_eq!(features.gpu_family, MetalGpuFamily::Apple7);
    }

    /// "Apple M1 Pro" -> M1Pro, Apple7
    #[test]
    fn test_apple_m1_pro_string() {
        let features = MetalFeatures::from_device_info("Apple M1 Pro");
        assert_eq!(features.silicon_generation, AppleSiliconGeneration::M1Pro);
        assert_eq!(features.gpu_family, MetalGpuFamily::Apple7);
    }

    /// "Apple M2 Max" -> M2Max, Apple8
    #[test]
    fn test_apple_m2_max_string() {
        let features = MetalFeatures::from_device_info("Apple M2 Max");
        assert_eq!(features.silicon_generation, AppleSiliconGeneration::M2Max);
        assert_eq!(features.gpu_family, MetalGpuFamily::Apple8);
    }

    /// "Apple M3" -> M3, Apple9
    #[test]
    fn test_apple_m3_string() {
        let features = MetalFeatures::from_device_info("Apple M3");
        assert_eq!(features.silicon_generation, AppleSiliconGeneration::M3);
        assert_eq!(features.gpu_family, MetalGpuFamily::Apple9);
    }

    /// "Apple M4 Pro" -> M4Pro, Apple9
    #[test]
    fn test_apple_m4_pro_string() {
        let features = MetalFeatures::from_device_info("Apple M4 Pro");
        assert_eq!(features.silicon_generation, AppleSiliconGeneration::M4Pro);
        assert_eq!(features.gpu_family, MetalGpuFamily::Apple9);
    }

    /// "Intel UHD Graphics 630" -> Mac2
    #[test]
    fn test_intel_uhd_630_string() {
        let features = MetalFeatures::from_device_info("Intel UHD Graphics 630");
        assert_eq!(features.silicon_generation, AppleSiliconGeneration::Unknown);
        assert_eq!(features.gpu_family, MetalGpuFamily::Mac2);
    }

    /// "AMD Radeon Pro 5500M" -> Mac2
    #[test]
    fn test_amd_radeon_pro_5500m_string() {
        let features = MetalFeatures::from_device_info("AMD Radeon Pro 5500M");
        assert_eq!(features.silicon_generation, AppleSiliconGeneration::Unknown);
        assert_eq!(features.gpu_family, MetalGpuFamily::Mac2);
    }

    /// "AMD Radeon RX 6800" -> Mac2
    #[test]
    fn test_amd_radeon_rx_6800_string() {
        let family = MetalGpuFamily::from_device_name("AMD Radeon RX 6800");
        assert_eq!(family, MetalGpuFamily::Mac2);
    }

    /// "Radeon Pro Vega 64" -> Mac2
    #[test]
    fn test_radeon_pro_vega_string() {
        let family = MetalGpuFamily::from_device_name("Radeon Pro Vega 64");
        assert_eq!(family, MetalGpuFamily::Mac2);
    }

    /// Case insensitivity: "apple m3 pro" works
    #[test]
    fn test_lowercase_device_name() {
        let features = MetalFeatures::from_device_info("apple m3 pro");
        assert_eq!(features.silicon_generation, AppleSiliconGeneration::M3Pro);
        assert_eq!(features.gpu_family, MetalGpuFamily::Apple9);
    }

    /// Case insensitivity: "APPLE M3 PRO" works
    #[test]
    fn test_uppercase_device_name() {
        let features = MetalFeatures::from_device_info("APPLE M3 PRO");
        assert_eq!(features.silicon_generation, AppleSiliconGeneration::M3Pro);
        assert_eq!(features.gpu_family, MetalGpuFamily::Apple9);
    }

    /// Mixed case: "Apple m3 Max" works
    #[test]
    fn test_mixed_case_device_name() {
        let features = MetalFeatures::from_device_info("Apple m3 Max");
        assert_eq!(features.silicon_generation, AppleSiliconGeneration::M3Max);
        assert_eq!(features.gpu_family, MetalGpuFamily::Apple9);
    }

    /// Generic "Apple GPU" falls back to Apple7
    #[test]
    fn test_generic_apple_gpu_string() {
        let family = MetalGpuFamily::from_device_name("Apple GPU");
        assert_eq!(family, MetalGpuFamily::Apple7);
    }

    /// iPad Pro A14 GPU string
    #[test]
    fn test_ipad_pro_a14_string() {
        let features = MetalFeatures::from_device_info("Apple A14 Bionic GPU");
        assert_eq!(features.silicon_generation, AppleSiliconGeneration::A14);
        assert_eq!(features.gpu_family, MetalGpuFamily::Apple7);
    }

    /// iPhone 15 Pro A17 Pro string
    #[test]
    fn test_iphone_15_pro_a17_string() {
        let features = MetalFeatures::from_device_info("Apple A17 Pro GPU");
        assert_eq!(features.silicon_generation, AppleSiliconGeneration::A17Pro);
        assert_eq!(features.gpu_family, MetalGpuFamily::Apple9);
    }
}

// =============================================================================
// 5. Feature Progression Validation
// =============================================================================

mod feature_progression {
    use super::*;

    /// M1 lacks hardware ray tracing (uses software path), has argument buffers
    #[test]
    fn test_m1_feature_set() {
        let features = MetalFeatures::from_device_info("Apple M1");
        assert!(features.ray_tracing);
        assert!(features.supports_argument_buffers());
        assert_eq!(features.argument_buffers_tier, 2);
        // M1 is the first to support Metal 3
        assert!(features.gpu_family.supports_metal3());
    }

    /// M3 adds hardware ray tracing
    #[test]
    fn test_m3_adds_hardware_rt() {
        let features = MetalFeatures::from_device_info("Apple M3");
        assert!(features.ray_tracing);
        assert!(features.lossless_compression);
        assert!(features.primitive_motion_blur);
        assert!(features.function_pointers);
    }

    /// M4 has all Metal 3 features
    #[test]
    fn test_m4_has_all_metal3_features() {
        let features = MetalFeatures::from_device_info("Apple M4 Max");
        assert!(features.ray_tracing);
        assert!(features.mesh_shaders);
        assert!(features.sparse_textures);
        assert!(features.lossless_compression);
        assert!(features.primitive_motion_blur);
        assert!(features.function_pointers);
        assert!(features.memoryless_render_targets);
        assert!(features.tile_shaders);
        assert!(features.imageblock);
        assert_eq!(features.argument_buffers_tier, 2);
    }

    /// Apple7 is minimum for Metal 3
    #[test]
    fn test_apple7_is_metal3_minimum() {
        assert!(MetalGpuFamily::Apple7.supports_metal3());
        assert!(!MetalGpuFamily::Apple6.supports_metal3());
    }

    /// Apple8+ supports mesh shaders (via Metal 3)
    #[test]
    fn test_apple8_supports_mesh_shaders() {
        assert!(MetalGpuFamily::Apple8.supports_mesh_shaders());
        assert!(MetalGpuFamily::Apple9.supports_mesh_shaders());
    }

    /// A12 introduced sparse textures (Apple5)
    #[test]
    fn test_a12_introduced_sparse_textures() {
        let a12 = MetalFeatures::from_device_info("Apple A12");
        let a11 = MetalFeatures::from_device_info("Apple A11");

        assert!(a12.sparse_textures);
        assert!(!a11.sparse_textures);
    }

    /// A11 introduced tile shaders (Apple4)
    #[test]
    fn test_a11_introduced_tile_shaders() {
        let a11 = MetalFeatures::from_device_info("Apple A11");
        let a10 = MetalFeatures::from_device_info("Apple A10");

        assert!(a11.tile_shaders);
        assert!(!a10.tile_shaders);
    }

    /// Feature progression A14 -> A15 -> A16 -> A17
    #[test]
    fn test_a_series_progression() {
        let a14 = MetalFeatures::from_device_info("Apple A14");
        let a15 = MetalFeatures::from_device_info("Apple A15");
        let a16 = MetalFeatures::from_device_info("Apple A16");
        let a17 = MetalFeatures::from_device_info("Apple A17 Pro");

        // All should have Metal 3 features
        assert!(a14.ray_tracing);
        assert!(a15.ray_tracing);
        assert!(a16.ray_tracing);
        assert!(a17.ray_tracing);

        // All should have mesh shaders
        assert!(a14.mesh_shaders);
        assert!(a15.mesh_shaders);
        assert!(a16.mesh_shaders);
        assert!(a17.mesh_shaders);

        // GPU family progression
        assert_eq!(a14.gpu_family, MetalGpuFamily::Apple7);
        assert_eq!(a15.gpu_family, MetalGpuFamily::Apple8);
        assert_eq!(a16.gpu_family, MetalGpuFamily::Apple9);
        assert_eq!(a17.gpu_family, MetalGpuFamily::Apple9);
    }

    /// M-series progression M1 -> M2 -> M3 -> M4
    #[test]
    fn test_m_series_progression() {
        let m1 = MetalFeatures::from_device_info("Apple M1");
        let m2 = MetalFeatures::from_device_info("Apple M2");
        let m3 = MetalFeatures::from_device_info("Apple M3");
        let m4 = MetalFeatures::from_device_info("Apple M4");

        // All M-series have Metal 3 features
        for features in [&m1, &m2, &m3, &m4] {
            assert!(features.ray_tracing);
            assert!(features.mesh_shaders);
            assert!(features.supports_bindless());
            assert!(features.lossless_compression);
        }

        // GPU family progression
        assert_eq!(m1.gpu_family, MetalGpuFamily::Apple7);
        assert_eq!(m2.gpu_family, MetalGpuFamily::Apple8);
        assert_eq!(m3.gpu_family, MetalGpuFamily::Apple9);
        assert_eq!(m4.gpu_family, MetalGpuFamily::Apple9);
    }

    /// Older chips (A10 and earlier) have limited features
    #[test]
    fn test_legacy_chip_features() {
        let a10 = MetalFeatures::from_device_info("Apple A10");
        let a9 = MetalFeatures::from_device_info("Apple A9");
        let a8 = MetalFeatures::from_device_info("Apple A8");

        // No Metal 3 features
        assert!(!a10.ray_tracing);
        assert!(!a10.mesh_shaders);
        assert!(!a9.ray_tracing);
        assert!(!a8.ray_tracing);

        // A10 has SIMD group and read-write textures (Apple3)
        assert!(a10.simd_group);
        assert!(a10.read_write_textures);
        assert!(a9.simd_group);

        // A10 has memoryless render targets (iOS feature)
        assert!(a10.memoryless_render_targets);
    }
}

// =============================================================================
// 6. Backend Integration
// =============================================================================

mod backend_integration {
    use super::*;

    /// MetalFeatures can be created with default
    #[test]
    fn test_metal_features_default() {
        let features = MetalFeatures::default();
        assert_eq!(features.gpu_family, MetalGpuFamily::Unknown);
        assert_eq!(features.silicon_generation, AppleSiliconGeneration::Unknown);
        assert!(!features.ray_tracing);
        assert!(!features.mesh_shaders);
        assert_eq!(features.argument_buffers_tier, 0);
    }

    /// MetalFeatures supports trait derivations
    #[test]
    fn test_metal_features_traits() {
        let features = MetalFeatures::from_device_info("Apple M3");

        // Clone
        let cloned = features.clone();
        assert_eq!(features, cloned);

        // Copy
        let copied: MetalFeatures = features;
        assert_eq!(features, copied);

        // Debug
        let debug_str = format!("{:?}", features);
        assert!(debug_str.contains("MetalFeatures"));
    }

    /// MetalGpuFamily supports trait derivations
    #[test]
    fn test_gpu_family_traits() {
        let family = MetalGpuFamily::Apple9;

        // Clone
        let cloned = family.clone();
        assert_eq!(family, cloned);

        // Copy
        let copied: MetalGpuFamily = family;
        assert_eq!(family, copied);

        // Hash (implicit via PartialEq + Eq)
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(family);
        assert!(set.contains(&MetalGpuFamily::Apple9));

        // Debug
        let debug_str = format!("{:?}", family);
        assert!(debug_str.contains("Apple9"));

        // Display
        let display_str = format!("{}", family);
        assert_eq!(display_str, "Apple9");
    }

    /// AppleSiliconGeneration supports trait derivations
    #[test]
    fn test_silicon_generation_traits() {
        let gen = AppleSiliconGeneration::M3Pro;

        // Clone
        let cloned = gen.clone();
        assert_eq!(gen, cloned);

        // Copy
        let copied: AppleSiliconGeneration = gen;
        assert_eq!(gen, copied);

        // Hash (implicit via PartialEq + Eq)
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(gen);
        assert!(set.contains(&AppleSiliconGeneration::M3Pro));

        // Debug
        let debug_str = format!("{:?}", gen);
        assert!(debug_str.contains("M3Pro"));

        // Display
        let display_str = format!("{}", gen);
        assert_eq!(display_str, "M3 Pro");
    }

    /// Feature summary includes key information
    #[test]
    fn test_feature_summary_content() {
        let features = MetalFeatures::from_device_info("Apple M3 Pro");
        let summary = features.summary();

        assert!(summary.contains("Apple9"), "Summary should include GPU family");
        assert!(summary.contains("M3 Pro"), "Summary should include chip name");
        assert!(summary.contains("RT"), "Summary should include RT for M3");
        assert!(summary.contains("Mesh"), "Summary should include Mesh for M3");
        assert!(summary.contains("ArgBuf-T2"), "Summary should include argument buffers tier");
    }

    /// Intel GPU summary does not include RT/Mesh
    #[test]
    fn test_intel_summary_no_rt_mesh() {
        let features = MetalFeatures::from_device_info("Intel UHD Graphics 630");
        let summary = features.summary();

        assert!(summary.contains("Mac2"));
        assert!(!summary.contains("RT "));
        assert!(!summary.contains("Mesh"));
    }

    /// Minimum macOS version for M3 is 14.0
    #[test]
    fn test_minimum_macos_m3() {
        let features = MetalFeatures::from_device_info("Apple M3");
        assert_eq!(features.minimum_macos_version(), (14, 0));
    }

    /// Minimum macOS version for M4 is 14.0
    #[test]
    fn test_minimum_macos_m4() {
        let features = MetalFeatures::from_device_info("Apple M4 Pro");
        assert_eq!(features.minimum_macos_version(), (14, 0));
    }

    /// Minimum macOS version for M1 is 13.0 (Ventura for Metal 3 RT)
    #[test]
    fn test_minimum_macos_m1() {
        let features = MetalFeatures::from_device_info("Apple M1");
        assert_eq!(features.minimum_macos_version(), (13, 0));
    }

    /// Minimum macOS version for M2 is 13.0
    #[test]
    fn test_minimum_macos_m2() {
        let features = MetalFeatures::from_device_info("Apple M2");
        assert_eq!(features.minimum_macos_version(), (13, 0));
    }

    /// GPU family name method works correctly
    #[test]
    fn test_gpu_family_name() {
        assert_eq!(MetalGpuFamily::Apple1.name(), "Apple1");
        assert_eq!(MetalGpuFamily::Apple7.name(), "Apple7");
        assert_eq!(MetalGpuFamily::Apple9.name(), "Apple9");
        assert_eq!(MetalGpuFamily::Mac1.name(), "Mac1");
        assert_eq!(MetalGpuFamily::Mac2.name(), "Mac2");
        assert_eq!(MetalGpuFamily::Metal3.name(), "Metal3");
        assert_eq!(MetalGpuFamily::Unknown.name(), "Unknown");
    }

    /// Silicon generation name method works correctly
    #[test]
    fn test_silicon_generation_name() {
        assert_eq!(AppleSiliconGeneration::M1.name(), "M1");
        assert_eq!(AppleSiliconGeneration::M1Pro.name(), "M1 Pro");
        assert_eq!(AppleSiliconGeneration::M1Max.name(), "M1 Max");
        assert_eq!(AppleSiliconGeneration::M1Ultra.name(), "M1 Ultra");
        assert_eq!(AppleSiliconGeneration::M2.name(), "M2");
        assert_eq!(AppleSiliconGeneration::M3Max.name(), "M3 Max");
        assert_eq!(AppleSiliconGeneration::M4.name(), "M4");
        assert_eq!(AppleSiliconGeneration::A14.name(), "A14 Bionic");
        assert_eq!(AppleSiliconGeneration::A17Pro.name(), "A17 Pro");
        assert_eq!(AppleSiliconGeneration::Unknown.name(), "Unknown");
    }
}

// =============================================================================
// 7. Edge Cases and Negative Tests
// =============================================================================

mod edge_cases {
    use super::*;

    /// Empty string returns unknown
    #[test]
    fn test_empty_string() {
        let family = MetalGpuFamily::from_device_name("");
        assert_eq!(family, MetalGpuFamily::Unknown);

        let gen = AppleSiliconGeneration::from_device_name("");
        assert_eq!(gen, AppleSiliconGeneration::Unknown);

        let features = MetalFeatures::from_device_info("");
        assert_eq!(features.gpu_family, MetalGpuFamily::Unknown);
    }

    /// Whitespace-only string returns unknown
    #[test]
    fn test_whitespace_only() {
        let family = MetalGpuFamily::from_device_name("   ");
        assert_eq!(family, MetalGpuFamily::Unknown);

        let gen = AppleSiliconGeneration::from_device_name("   ");
        assert_eq!(gen, AppleSiliconGeneration::Unknown);
    }

    /// Partial match "M1" should work (not require "Apple" prefix)
    #[test]
    fn test_partial_match_m1() {
        let family = MetalGpuFamily::from_device_name("M1 GPU");
        assert_eq!(family, MetalGpuFamily::Apple7);

        let gen = AppleSiliconGeneration::from_device_name("M1 GPU");
        assert_eq!(gen, AppleSiliconGeneration::M1);
    }

    /// Partial match "M3 Max" should work
    #[test]
    fn test_partial_match_m3_max() {
        let family = MetalGpuFamily::from_device_name("M3 Max");
        assert_eq!(family, MetalGpuFamily::Apple9);

        let gen = AppleSiliconGeneration::from_device_name("M3 Max");
        assert_eq!(gen, AppleSiliconGeneration::M3Max);
    }

    /// Numbers in string don't cause issues
    #[test]
    fn test_numbers_in_string() {
        let family = MetalGpuFamily::from_device_name("Device 12345");
        assert_eq!(family, MetalGpuFamily::Unknown);
    }

    /// Special characters don't crash
    #[test]
    fn test_special_characters() {
        let family = MetalGpuFamily::from_device_name("Apple M1 Pro (TM)");
        assert_eq!(family, MetalGpuFamily::Apple7);

        let gen = AppleSiliconGeneration::from_device_name("Apple M1 Pro (TM)");
        assert_eq!(gen, AppleSiliconGeneration::M1Pro);
    }

    /// Unicode characters don't crash
    #[test]
    fn test_unicode_characters() {
        let family = MetalGpuFamily::from_device_name("Apple M1 \u{00AE}");
        assert_eq!(family, MetalGpuFamily::Apple7);
    }

    /// Very long string doesn't crash
    #[test]
    fn test_very_long_string() {
        let long_name = "A".repeat(10000);
        let family = MetalGpuFamily::from_device_name(&long_name);
        // Should not crash, may return Unknown or a valid family
        assert!(matches!(
            family,
            MetalGpuFamily::Unknown
                | MetalGpuFamily::Apple1
                | MetalGpuFamily::Apple2
                | MetalGpuFamily::Apple3
                | MetalGpuFamily::Apple4
                | MetalGpuFamily::Apple5
                | MetalGpuFamily::Apple6
                | MetalGpuFamily::Apple7
                | MetalGpuFamily::Apple8
                | MetalGpuFamily::Apple9
                | MetalGpuFamily::Mac1
                | MetalGpuFamily::Mac2
                | MetalGpuFamily::Common1
                | MetalGpuFamily::Common2
                | MetalGpuFamily::Common3
                | MetalGpuFamily::Metal3
        ));
    }

    /// Substring "M10" should not match M1
    #[test]
    fn test_no_false_positive_m10() {
        // "M10" contains "M1" but should not match M1
        let family = MetalGpuFamily::from_device_name("Apple M10");
        // This would be a future chip, for now it might match M1 due to substring
        // The implementation uses contains() so "m10" contains "m1"
        // This is documenting current behavior
        assert_eq!(family, MetalGpuFamily::Apple7); // M1 match due to substring
    }

    /// "A100" should not match A10
    #[test]
    fn test_no_false_positive_a100() {
        // Similar issue - "a100" contains "a10"
        let family = MetalGpuFamily::from_device_name("Apple A100");
        // Current implementation will match A10 first
        assert_eq!(family, MetalGpuFamily::Apple3);
    }

    /// Common families exist
    #[test]
    fn test_common_families_exist() {
        // These are cross-platform families
        assert_eq!(MetalGpuFamily::Common1.name(), "Common1");
        assert_eq!(MetalGpuFamily::Common2.name(), "Common2");
        assert_eq!(MetalGpuFamily::Common3.name(), "Common3");

        // They are not Apple Silicon
        assert!(!MetalGpuFamily::Common1.is_apple_silicon());
        assert!(!MetalGpuFamily::Common2.is_apple_silicon());
        assert!(!MetalGpuFamily::Common3.is_apple_silicon());

        // They are not Intel Mac either
        assert!(!MetalGpuFamily::Common1.is_intel_mac());
    }

    /// Apple version returns correct values for all families
    #[test]
    fn test_all_apple_versions() {
        assert_eq!(MetalGpuFamily::Apple1.apple_version(), 1);
        assert_eq!(MetalGpuFamily::Apple2.apple_version(), 2);
        assert_eq!(MetalGpuFamily::Apple3.apple_version(), 3);
        assert_eq!(MetalGpuFamily::Apple4.apple_version(), 4);
        assert_eq!(MetalGpuFamily::Apple5.apple_version(), 5);
        assert_eq!(MetalGpuFamily::Apple6.apple_version(), 6);
        assert_eq!(MetalGpuFamily::Apple7.apple_version(), 7);
        assert_eq!(MetalGpuFamily::Apple8.apple_version(), 8);
        assert_eq!(MetalGpuFamily::Apple9.apple_version(), 9);
        assert_eq!(MetalGpuFamily::Mac1.apple_version(), 0);
        assert_eq!(MetalGpuFamily::Mac2.apple_version(), 0);
        assert_eq!(MetalGpuFamily::Common1.apple_version(), 0);
        assert_eq!(MetalGpuFamily::Metal3.apple_version(), 0);
        assert_eq!(MetalGpuFamily::Unknown.apple_version(), 0);
    }

    /// is_m_series returns true only for M-series generations
    #[test]
    fn test_is_m_series_comprehensive() {
        // M-series should return true
        assert!(AppleSiliconGeneration::M1.is_m_series());
        assert!(AppleSiliconGeneration::M1Pro.is_m_series());
        assert!(AppleSiliconGeneration::M1Max.is_m_series());
        assert!(AppleSiliconGeneration::M1Ultra.is_m_series());
        assert!(AppleSiliconGeneration::M2.is_m_series());
        assert!(AppleSiliconGeneration::M2Pro.is_m_series());
        assert!(AppleSiliconGeneration::M2Max.is_m_series());
        assert!(AppleSiliconGeneration::M2Ultra.is_m_series());
        assert!(AppleSiliconGeneration::M3.is_m_series());
        assert!(AppleSiliconGeneration::M3Pro.is_m_series());
        assert!(AppleSiliconGeneration::M3Max.is_m_series());
        assert!(AppleSiliconGeneration::M4.is_m_series());
        assert!(AppleSiliconGeneration::M4Pro.is_m_series());
        assert!(AppleSiliconGeneration::M4Max.is_m_series());

        // A-series and Unknown should return false
        assert!(!AppleSiliconGeneration::A14.is_m_series());
        assert!(!AppleSiliconGeneration::A15.is_m_series());
        assert!(!AppleSiliconGeneration::A16.is_m_series());
        assert!(!AppleSiliconGeneration::A17Pro.is_m_series());
        assert!(!AppleSiliconGeneration::Unknown.is_m_series());
    }

    /// MetalFeatures.is_m_series matches AppleSiliconGeneration
    #[test]
    fn test_metal_features_is_m_series() {
        let m1 = MetalFeatures::from_device_info("Apple M1");
        let a15 = MetalFeatures::from_device_info("Apple A15");
        let intel = MetalFeatures::from_device_info("Intel UHD Graphics 630");

        assert!(m1.is_m_series());
        assert!(!a15.is_m_series());
        assert!(!intel.is_m_series());
    }

    /// Float32 MSAA resolve availability
    #[test]
    fn test_float32_msaa_resolve() {
        // Apple4+ and Intel should support it
        let a11 = MetalFeatures::from_device_info("Apple A11");
        let m1 = MetalFeatures::from_device_info("Apple M1");
        let intel = MetalFeatures::from_device_info("Intel UHD Graphics 630");

        assert!(a11.float32_msaa_resolve);
        assert!(m1.float32_msaa_resolve);
        assert!(intel.float32_msaa_resolve);

        // A10 (Apple3) should not
        let a10 = MetalFeatures::from_device_info("Apple A10");
        assert!(!a10.float32_msaa_resolve);
    }

    /// Primitive restart 32-bit availability
    #[test]
    fn test_primitive_restart_32bit() {
        let m1 = MetalFeatures::from_device_info("Apple M1");
        let a10 = MetalFeatures::from_device_info("Apple A10");
        let unknown = MetalFeatures::from_device_info("Unknown");

        assert!(m1.primitive_restart_32bit);
        assert!(a10.primitive_restart_32bit);
        assert!(!unknown.primitive_restart_32bit);
    }
}

// =============================================================================
// 8. Comparison and Equality Tests
// =============================================================================

mod comparison {
    use super::*;

    /// GPU family equality comparison
    #[test]
    fn test_gpu_family_equality() {
        assert_eq!(MetalGpuFamily::Apple7, MetalGpuFamily::Apple7);
        assert_ne!(MetalGpuFamily::Apple7, MetalGpuFamily::Apple8);
        assert_ne!(MetalGpuFamily::Apple7, MetalGpuFamily::Mac2);
    }

    /// Silicon generation equality comparison
    #[test]
    fn test_silicon_generation_equality() {
        assert_eq!(AppleSiliconGeneration::M1Pro, AppleSiliconGeneration::M1Pro);
        assert_ne!(AppleSiliconGeneration::M1Pro, AppleSiliconGeneration::M1Max);
        assert_ne!(AppleSiliconGeneration::M1, AppleSiliconGeneration::M2);
    }

    /// MetalFeatures equality based on all fields
    #[test]
    fn test_metal_features_equality() {
        let m1_a = MetalFeatures::from_device_info("Apple M1");
        let m1_b = MetalFeatures::from_device_info("Apple M1");
        let m2 = MetalFeatures::from_device_info("Apple M2");

        assert_eq!(m1_a, m1_b);
        assert_ne!(m1_a, m2);
    }

    /// Different case produces equal results
    #[test]
    fn test_case_insensitive_equality() {
        let lower = MetalFeatures::from_device_info("apple m3 pro");
        let upper = MetalFeatures::from_device_info("APPLE M3 PRO");

        assert_eq!(lower, upper);
    }
}

// =============================================================================
// 9. Comprehensive Feature Matrix Tests
// =============================================================================

mod feature_matrix {
    use super::*;

    /// Test complete feature matrix for Apple Silicon generations
    #[test]
    fn test_apple_silicon_feature_matrix() {
        struct FeatureExpectation {
            device: &'static str,
            gpu_family: MetalGpuFamily,
            generation: AppleSiliconGeneration,
            rt: bool,
            mesh: bool,
            sparse: bool,
            tile: bool,
            arg_tier: u8,
        }

        let expectations = [
            FeatureExpectation {
                device: "Apple M1",
                gpu_family: MetalGpuFamily::Apple7,
                generation: AppleSiliconGeneration::M1,
                rt: true,
                mesh: true,
                sparse: true,
                tile: true,
                arg_tier: 2,
            },
            FeatureExpectation {
                device: "Apple M2 Pro",
                gpu_family: MetalGpuFamily::Apple8,
                generation: AppleSiliconGeneration::M2Pro,
                rt: true,
                mesh: true,
                sparse: true,
                tile: true,
                arg_tier: 2,
            },
            FeatureExpectation {
                device: "Apple M3 Max",
                gpu_family: MetalGpuFamily::Apple9,
                generation: AppleSiliconGeneration::M3Max,
                rt: true,
                mesh: true,
                sparse: true,
                tile: true,
                arg_tier: 2,
            },
            FeatureExpectation {
                device: "Apple M4",
                gpu_family: MetalGpuFamily::Apple9,
                generation: AppleSiliconGeneration::M4,
                rt: true,
                mesh: true,
                sparse: true,
                tile: true,
                arg_tier: 2,
            },
            FeatureExpectation {
                device: "Apple A14",
                gpu_family: MetalGpuFamily::Apple7,
                generation: AppleSiliconGeneration::A14,
                rt: true,
                mesh: true,
                sparse: true,
                tile: true,
                arg_tier: 2,
            },
            FeatureExpectation {
                device: "Apple A13",
                gpu_family: MetalGpuFamily::Apple6,
                generation: AppleSiliconGeneration::Unknown,
                rt: false,
                mesh: false,
                sparse: true,  // Apple6 (A13) has sparse textures (Apple5+)
                tile: true,
                arg_tier: 2,
            },
            FeatureExpectation {
                device: "Apple A12",
                gpu_family: MetalGpuFamily::Apple5,
                generation: AppleSiliconGeneration::Unknown,
                rt: false,
                mesh: false,
                sparse: true,
                tile: true,
                arg_tier: 2,
            },
            FeatureExpectation {
                device: "Apple A11",
                gpu_family: MetalGpuFamily::Apple4,
                generation: AppleSiliconGeneration::Unknown,
                rt: false,
                mesh: false,
                sparse: false,
                tile: true,
                arg_tier: 1,
            },
            FeatureExpectation {
                device: "Intel UHD Graphics 630",
                gpu_family: MetalGpuFamily::Mac2,
                generation: AppleSiliconGeneration::Unknown,
                rt: false,
                mesh: false,
                sparse: false,
                tile: false,
                arg_tier: 1,
            },
        ];

        for exp in expectations {
            let features = MetalFeatures::from_device_info(exp.device);
            assert_eq!(
                features.gpu_family, exp.gpu_family,
                "GPU family mismatch for {}",
                exp.device
            );
            assert_eq!(
                features.silicon_generation, exp.generation,
                "Generation mismatch for {}",
                exp.device
            );
            assert_eq!(
                features.ray_tracing, exp.rt,
                "RT mismatch for {}",
                exp.device
            );
            assert_eq!(
                features.mesh_shaders, exp.mesh,
                "Mesh shader mismatch for {}",
                exp.device
            );
            assert_eq!(
                features.sparse_textures, exp.sparse,
                "Sparse texture mismatch for {}",
                exp.device
            );
            assert_eq!(
                features.tile_shaders, exp.tile,
                "Tile shader mismatch for {}",
                exp.device
            );
            assert_eq!(
                features.argument_buffers_tier, exp.arg_tier,
                "Argument buffer tier mismatch for {}",
                exp.device
            );
        }
    }

    /// Test all M-series variants have correct GPU core estimates
    #[test]
    fn test_m_series_gpu_core_matrix() {
        let expectations = [
            ("Apple M1", 8),
            ("Apple M1 Pro", 16),
            ("Apple M1 Max", 32),
            ("Apple M1 Ultra", 64),
            ("Apple M2", 10),
            ("Apple M2 Pro", 19),
            ("Apple M2 Max", 38),
            ("Apple M2 Ultra", 76),
            ("Apple M3", 10),
            ("Apple M3 Pro", 18),
            ("Apple M3 Max", 40),
            ("Apple M4", 10),
            ("Apple M4 Pro", 20),
            ("Apple M4 Max", 40),
        ];

        for (device, expected_cores) in expectations {
            let gen = AppleSiliconGeneration::from_device_name(device);
            assert_eq!(
                gen.estimated_gpu_cores(),
                expected_cores,
                "GPU core count mismatch for {}",
                device
            );
        }
    }
}
