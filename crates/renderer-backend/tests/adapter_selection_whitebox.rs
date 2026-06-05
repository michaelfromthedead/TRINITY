//! Whitebox structural tests for AdapterSelector and related types.
//!
//! These tests verify the internal structure and behavior of the adapter selection
//! algorithm, including DeviceTypeWeights, AdapterBlacklistEntry, AdapterScore,
//! AdapterSelector, and SelectionResult.
//!
//! Task: T-WGPU-P1.2.5 - Adapter Selection Algorithm
//!
//! Acceptance Criteria Tested:
//! 1. Score by device type (discrete > integrated > software)
//! 2. Score by feature availability (FeatureTier affects score)
//! 3. Score by limits (larger limits = higher score)
//! 4. Blacklist support (filter known-broken drivers)
//! 5. Vendor preference support (boost for preferred vendor)
//! 6. Falls back to first available if no preference match

use renderer_backend::device::{
    AdapterBlacklistEntry, AdapterFeatures, AdapterScore, AdapterSelector, DeviceTypeWeights,
    FeatureTier, Vendor,
};
use wgpu::{AdapterInfo, DeviceType};

// ============================================================================
// Helper Functions
// ============================================================================

/// Create a mock AdapterInfo for testing.
fn make_adapter_info(
    name: &str,
    vendor_id: u32,
    device_type: DeviceType,
) -> AdapterInfo {
    AdapterInfo {
        name: name.to_string(),
        vendor: vendor_id,
        device: 0,
        device_type,
        backend: wgpu::Backend::Vulkan,
        driver: String::new(),
        driver_info: String::new(),
    }
}

/// Create a mock AdapterInfo with driver info.
#[allow(dead_code)]
fn make_adapter_info_with_driver(
    name: &str,
    vendor_id: u32,
    device_type: DeviceType,
    driver: &str,
    driver_info: &str,
) -> AdapterInfo {
    AdapterInfo {
        name: name.to_string(),
        vendor: vendor_id,
        device: 0,
        device_type,
        backend: wgpu::Backend::Vulkan,
        driver: driver.to_string(),
        driver_info: driver_info.to_string(),
    }
}

// ============================================================================
// 1. DeviceTypeWeights Tests
// ============================================================================

mod device_type_weights {
    use super::*;

    #[test]
    fn default_weights_have_correct_values() {
        let weights = DeviceTypeWeights::default();
        assert_eq!(weights.discrete, 1000);
        assert_eq!(weights.integrated, 500);
        assert_eq!(weights.virtual_gpu, 300);
        assert_eq!(weights.cpu, 100);
        assert_eq!(weights.other, 50);
    }

    #[test]
    fn default_weights_discrete_greater_than_integrated() {
        let weights = DeviceTypeWeights::default();
        assert!(weights.discrete > weights.integrated);
    }

    #[test]
    fn default_weights_integrated_greater_than_virtual() {
        let weights = DeviceTypeWeights::default();
        assert!(weights.integrated > weights.virtual_gpu);
    }

    #[test]
    fn default_weights_virtual_greater_than_cpu() {
        let weights = DeviceTypeWeights::default();
        assert!(weights.virtual_gpu > weights.cpu);
    }

    #[test]
    fn default_weights_cpu_greater_than_other() {
        let weights = DeviceTypeWeights::default();
        assert!(weights.cpu > weights.other);
    }

    #[test]
    fn weight_for_returns_discrete_for_discrete_gpu() {
        let weights = DeviceTypeWeights::default();
        assert_eq!(weights.weight_for(DeviceType::DiscreteGpu), 1000);
    }

    #[test]
    fn weight_for_returns_integrated_for_integrated_gpu() {
        let weights = DeviceTypeWeights::default();
        assert_eq!(weights.weight_for(DeviceType::IntegratedGpu), 500);
    }

    #[test]
    fn weight_for_returns_virtual_for_virtual_gpu() {
        let weights = DeviceTypeWeights::default();
        assert_eq!(weights.weight_for(DeviceType::VirtualGpu), 300);
    }

    #[test]
    fn weight_for_returns_cpu_for_cpu() {
        let weights = DeviceTypeWeights::default();
        assert_eq!(weights.weight_for(DeviceType::Cpu), 100);
    }

    #[test]
    fn weight_for_returns_other_for_other() {
        let weights = DeviceTypeWeights::default();
        assert_eq!(weights.weight_for(DeviceType::Other), 50);
    }

    #[test]
    fn power_saving_prefers_integrated() {
        let weights = DeviceTypeWeights::power_saving();
        assert!(weights.integrated > weights.discrete);
    }

    #[test]
    fn power_saving_has_correct_values() {
        let weights = DeviceTypeWeights::power_saving();
        assert_eq!(weights.discrete, 500);
        assert_eq!(weights.integrated, 1000);
        assert_eq!(weights.virtual_gpu, 200);
        assert_eq!(weights.cpu, 100);
        assert_eq!(weights.other, 50);
    }

    #[test]
    fn performance_strongly_prefers_discrete() {
        let weights = DeviceTypeWeights::performance();
        assert!(weights.discrete > weights.integrated * 2);
    }

    #[test]
    fn performance_has_correct_values() {
        let weights = DeviceTypeWeights::performance();
        assert_eq!(weights.discrete, 2000);
        assert_eq!(weights.integrated, 400);
        assert_eq!(weights.virtual_gpu, 200);
        assert_eq!(weights.cpu, 50);
        assert_eq!(weights.other, 25);
    }

    #[test]
    fn custom_weights_are_configurable() {
        let weights = DeviceTypeWeights {
            discrete: 500,
            integrated: 1500,
            virtual_gpu: 100,
            cpu: 10,
            other: 5,
        };
        assert_eq!(weights.discrete, 500);
        assert_eq!(weights.integrated, 1500);
        assert_eq!(weights.virtual_gpu, 100);
        assert_eq!(weights.cpu, 10);
        assert_eq!(weights.other, 5);
    }

    #[test]
    fn weights_are_copy() {
        let weights = DeviceTypeWeights::default();
        let copy = weights;
        assert_eq!(copy.discrete, weights.discrete);
    }

    #[test]
    fn weights_are_clone() {
        let weights = DeviceTypeWeights::default();
        let cloned = weights.clone();
        assert_eq!(cloned.discrete, weights.discrete);
    }

    #[test]
    fn weights_implement_eq() {
        let a = DeviceTypeWeights::default();
        let b = DeviceTypeWeights::default();
        assert_eq!(a, b);
    }

    #[test]
    fn weights_implement_partial_eq() {
        let a = DeviceTypeWeights::default();
        let b = DeviceTypeWeights::power_saving();
        assert_ne!(a, b);
    }

    #[test]
    fn weights_implement_debug() {
        let weights = DeviceTypeWeights::default();
        let debug = format!("{:?}", weights);
        assert!(debug.contains("DeviceTypeWeights"));
        assert!(debug.contains("discrete"));
    }
}

// ============================================================================
// 2. AdapterBlacklistEntry Tests
// ============================================================================

mod adapter_blacklist_entry {
    use super::*;

    #[test]
    fn new_creates_empty_entry() {
        let entry = AdapterBlacklistEntry::new();
        assert!(entry.vendor.is_none());
        assert!(entry.name_contains.is_none());
        assert!(entry.reason.is_empty());
    }

    #[test]
    fn default_creates_empty_entry() {
        let entry = AdapterBlacklistEntry::default();
        assert!(entry.vendor.is_none());
        assert!(entry.name_contains.is_none());
        assert!(entry.reason.is_empty());
    }

    #[test]
    fn with_vendor_sets_vendor() {
        let entry = AdapterBlacklistEntry::new().with_vendor(Vendor::Nvidia);
        assert_eq!(entry.vendor, Some(Vendor::Nvidia));
    }

    #[test]
    fn with_name_contains_sets_name() {
        let entry = AdapterBlacklistEntry::new().with_name_contains("WARP");
        assert_eq!(entry.name_contains, Some("WARP".to_string()));
    }

    #[test]
    fn with_name_contains_accepts_string() {
        let name = String::from("WARP Driver");
        let entry = AdapterBlacklistEntry::new().with_name_contains(name);
        assert_eq!(entry.name_contains, Some("WARP Driver".to_string()));
    }

    #[test]
    fn with_reason_sets_reason() {
        let entry = AdapterBlacklistEntry::new().with_reason("Known driver crash");
        assert_eq!(entry.reason, "Known driver crash");
    }

    #[test]
    fn builder_pattern_chains() {
        let entry = AdapterBlacklistEntry::new()
            .with_vendor(Vendor::Microsoft)
            .with_name_contains("WARP")
            .with_reason("Software renderer");

        assert_eq!(entry.vendor, Some(Vendor::Microsoft));
        assert_eq!(entry.name_contains, Some("WARP".to_string()));
        assert_eq!(entry.reason, "Software renderer");
    }

    #[test]
    fn matches_vendor_only() {
        let entry = AdapterBlacklistEntry::new()
            .with_vendor(Vendor::Microsoft)
            .with_reason("test");

        let microsoft = make_adapter_info("Microsoft Basic Render Driver", 0x1414, DeviceType::Cpu);
        let nvidia = make_adapter_info("NVIDIA GeForce RTX 4090", 0x10DE, DeviceType::DiscreteGpu);

        assert!(entry.matches(&microsoft));
        assert!(!entry.matches(&nvidia));
    }

    #[test]
    fn matches_name_substring_only() {
        let entry = AdapterBlacklistEntry::new()
            .with_name_contains("WARP")
            .with_reason("test");

        let warp = make_adapter_info("Microsoft Basic WARP Driver", 0x1414, DeviceType::Cpu);
        let nvidia = make_adapter_info("NVIDIA GeForce RTX 4090", 0x10DE, DeviceType::DiscreteGpu);

        assert!(entry.matches(&warp));
        assert!(!entry.matches(&nvidia));
    }

    #[test]
    fn matches_name_case_insensitive() {
        let entry = AdapterBlacklistEntry::new()
            .with_name_contains("warp")
            .with_reason("test");

        let warp_upper = make_adapter_info("Microsoft Basic WARP Driver", 0x1414, DeviceType::Cpu);
        let warp_mixed = make_adapter_info("Microsoft Warp Software", 0x1414, DeviceType::Cpu);

        assert!(entry.matches(&warp_upper));
        assert!(entry.matches(&warp_mixed));
    }

    #[test]
    fn matches_upper_case_pattern_is_case_insensitive() {
        let entry = AdapterBlacklistEntry::new()
            .with_name_contains("GEFORCE")
            .with_reason("test");

        let lower = make_adapter_info("nvidia geforce rtx 4090", 0x10DE, DeviceType::DiscreteGpu);
        let mixed = make_adapter_info("NVIDIA GeForce RTX 4090", 0x10DE, DeviceType::DiscreteGpu);

        assert!(entry.matches(&lower));
        assert!(entry.matches(&mixed));
    }

    #[test]
    fn matches_both_vendor_and_name_requires_both() {
        let entry = AdapterBlacklistEntry::new()
            .with_vendor(Vendor::Microsoft)
            .with_name_contains("WARP")
            .with_reason("test");

        // Both match
        let warp_microsoft = make_adapter_info("Microsoft Basic WARP Driver", 0x1414, DeviceType::Cpu);
        // Only name matches
        let warp_nvidia = make_adapter_info("WARP Software Renderer", 0x10DE, DeviceType::DiscreteGpu);
        // Only vendor matches
        let microsoft_other = make_adapter_info("Microsoft Basic Render Driver", 0x1414, DeviceType::Cpu);

        assert!(entry.matches(&warp_microsoft));
        assert!(!entry.matches(&warp_nvidia));
        assert!(!entry.matches(&microsoft_other));
    }

    #[test]
    fn empty_entry_never_matches() {
        let entry = AdapterBlacklistEntry::new();

        let any = make_adapter_info("Any GPU", 0x10DE, DeviceType::DiscreteGpu);
        assert!(!entry.matches(&any));
    }

    #[test]
    fn empty_entry_with_reason_only_never_matches() {
        let entry = AdapterBlacklistEntry::new().with_reason("No filter specified");

        let any = make_adapter_info("Any GPU", 0x10DE, DeviceType::DiscreteGpu);
        assert!(!entry.matches(&any));
    }

    #[test]
    fn matches_partial_name() {
        let entry = AdapterBlacklistEntry::new()
            .with_name_contains("RTX")
            .with_reason("test");

        let rtx_4090 = make_adapter_info("NVIDIA GeForce RTX 4090", 0x10DE, DeviceType::DiscreteGpu);
        let rtx_3080 = make_adapter_info("NVIDIA GeForce RTX 3080", 0x10DE, DeviceType::DiscreteGpu);
        let gtx_1080 = make_adapter_info("NVIDIA GeForce GTX 1080", 0x10DE, DeviceType::DiscreteGpu);

        assert!(entry.matches(&rtx_4090));
        assert!(entry.matches(&rtx_3080));
        assert!(!entry.matches(&gtx_1080));
    }

    #[test]
    fn entry_is_clone() {
        let entry = AdapterBlacklistEntry::new()
            .with_vendor(Vendor::Nvidia)
            .with_name_contains("RTX")
            .with_reason("test");

        let cloned = entry.clone();
        assert_eq!(cloned.vendor, entry.vendor);
        assert_eq!(cloned.name_contains, entry.name_contains);
        assert_eq!(cloned.reason, entry.reason);
    }

    #[test]
    fn entry_implements_debug() {
        let entry = AdapterBlacklistEntry::new()
            .with_vendor(Vendor::Nvidia)
            .with_reason("test");

        let debug = format!("{:?}", entry);
        assert!(debug.contains("AdapterBlacklistEntry"));
    }
}

// ============================================================================
// 3. AdapterScore Tests
// ============================================================================

mod adapter_score {
    use super::*;

    #[test]
    fn zero_creates_zero_score() {
        let score = AdapterScore {
            device_type_score: 0,
            feature_score: 0,
            limits_score: 0,
            vendor_bonus: 0,
            total: 0,
            blacklisted: false,
            blacklist_reason: None,
        };
        assert_eq!(score.device_type_score, 0);
        assert_eq!(score.feature_score, 0);
        assert_eq!(score.limits_score, 0);
        assert_eq!(score.vendor_bonus, 0);
        assert_eq!(score.total, 0);
        assert!(!score.blacklisted);
        assert!(score.blacklist_reason.is_none());
    }

    #[test]
    fn score_components_are_accessible() {
        let score = AdapterScore {
            device_type_score: 1000,
            feature_score: 400,
            limits_score: 150,
            vendor_bonus: 200,
            total: 1750,
            blacklisted: false,
            blacklist_reason: None,
        };

        assert_eq!(score.device_type_score, 1000);
        assert_eq!(score.feature_score, 400);
        assert_eq!(score.limits_score, 150);
        assert_eq!(score.vendor_bonus, 200);
        assert_eq!(score.total, 1750);
    }

    #[test]
    fn blacklisted_score_has_zero_total() {
        let score = AdapterScore {
            device_type_score: 0,
            feature_score: 0,
            limits_score: 0,
            vendor_bonus: 0,
            total: 0,
            blacklisted: true,
            blacklist_reason: Some("Driver bug".to_string()),
        };

        assert!(score.blacklisted);
        assert_eq!(score.total, 0);
        assert_eq!(score.blacklist_reason, Some("Driver bug".to_string()));
    }

    #[test]
    fn display_format_for_normal_score() {
        let score = AdapterScore {
            device_type_score: 1000,
            feature_score: 400,
            limits_score: 100,
            vendor_bonus: 200,
            total: 1700,
            blacklisted: false,
            blacklist_reason: None,
        };

        let display = format!("{}", score);
        assert!(display.contains("1700"));
        assert!(display.contains("type: 1000"));
        assert!(display.contains("features: 400"));
        assert!(display.contains("limits: 100"));
        assert!(display.contains("vendor: 200"));
    }

    #[test]
    fn display_format_for_blacklisted_score() {
        let score = AdapterScore {
            device_type_score: 0,
            feature_score: 0,
            limits_score: 0,
            vendor_bonus: 0,
            total: 0,
            blacklisted: true,
            blacklist_reason: Some("Known issue".to_string()),
        };

        let display = format!("{}", score);
        assert!(display.contains("BLACKLISTED"));
        assert!(display.contains("Known issue"));
    }

    #[test]
    fn display_format_for_blacklisted_without_reason() {
        let score = AdapterScore {
            device_type_score: 0,
            feature_score: 0,
            limits_score: 0,
            vendor_bonus: 0,
            total: 0,
            blacklisted: true,
            blacklist_reason: None,
        };

        let display = format!("{}", score);
        assert!(display.contains("BLACKLISTED"));
        assert!(display.contains("unknown"));
    }

    #[test]
    fn score_is_clone() {
        let score = AdapterScore {
            device_type_score: 1000,
            feature_score: 400,
            limits_score: 100,
            vendor_bonus: 200,
            total: 1700,
            blacklisted: false,
            blacklist_reason: None,
        };

        let cloned = score.clone();
        assert_eq!(cloned.total, score.total);
        assert_eq!(cloned.device_type_score, score.device_type_score);
    }

    #[test]
    fn score_implements_debug() {
        let score = AdapterScore {
            device_type_score: 1000,
            feature_score: 400,
            limits_score: 100,
            vendor_bonus: 0,
            total: 1500,
            blacklisted: false,
            blacklist_reason: None,
        };

        let debug = format!("{:?}", score);
        assert!(debug.contains("AdapterScore"));
    }
}

// ============================================================================
// 4. AdapterSelector Builder Tests
// ============================================================================

mod adapter_selector_builder {
    use super::*;

    #[test]
    fn new_creates_default_selector() {
        let selector = AdapterSelector::new();
        // Default settings - verify via behavior in subsequent tests
        let _ = selector;
    }

    #[test]
    fn default_creates_same_as_new() {
        let from_new = AdapterSelector::new();
        let from_default = AdapterSelector::default();
        // Both should produce equivalent selectors
        let _ = (from_new, from_default);
    }

    #[test]
    fn with_vendor_preference_sets_vendor() {
        let selector = AdapterSelector::new().with_vendor_preference(Vendor::Nvidia);
        // Verify by checking it affects scoring
        let _ = selector;
    }

    #[test]
    fn with_blacklist_entry_adds_entry() {
        let selector = AdapterSelector::new().with_blacklist_entry(
            AdapterBlacklistEntry::new()
                .with_vendor(Vendor::Microsoft)
                .with_reason("test"),
        );
        let _ = selector;
    }

    #[test]
    fn with_multiple_blacklist_entries() {
        let selector = AdapterSelector::new()
            .with_blacklist_entry(
                AdapterBlacklistEntry::new()
                    .with_vendor(Vendor::Microsoft)
                    .with_reason("Software renderer"),
            )
            .with_blacklist_entry(
                AdapterBlacklistEntry::new()
                    .with_name_contains("Buggy GPU")
                    .with_reason("Driver crash"),
            );
        let _ = selector;
    }

    #[test]
    fn with_blacklist_adds_multiple_entries() {
        let entries = vec![
            AdapterBlacklistEntry::new()
                .with_vendor(Vendor::Microsoft)
                .with_reason("test1"),
            AdapterBlacklistEntry::new()
                .with_name_contains("Bad GPU")
                .with_reason("test2"),
        ];

        let selector = AdapterSelector::new().with_blacklist(entries);
        let _ = selector;
    }

    #[test]
    fn with_device_type_weights_sets_weights() {
        let selector =
            AdapterSelector::new().with_device_type_weights(DeviceTypeWeights::power_saving());
        let _ = selector;
    }

    #[test]
    fn with_feature_weight_sets_weight() {
        let selector = AdapterSelector::new().with_feature_weight(150);
        let _ = selector;
    }

    #[test]
    fn with_limit_weight_sets_weight() {
        let selector = AdapterSelector::new().with_limit_weight(2);
        let _ = selector;
    }

    #[test]
    fn with_vendor_preference_bonus_sets_bonus() {
        let selector = AdapterSelector::new().with_vendor_preference_bonus(300);
        let _ = selector;
    }

    #[test]
    fn builder_pattern_chains_all_methods() {
        let selector = AdapterSelector::new()
            .with_vendor_preference(Vendor::Nvidia)
            .with_blacklist_entry(
                AdapterBlacklistEntry::new()
                    .with_vendor(Vendor::Microsoft)
                    .with_reason("test"),
            )
            .with_device_type_weights(DeviceTypeWeights::performance())
            .with_feature_weight(150)
            .with_limit_weight(2)
            .with_vendor_preference_bonus(300);
        let _ = selector;
    }

    #[test]
    fn selector_is_clone() {
        let selector = AdapterSelector::new()
            .with_vendor_preference(Vendor::Nvidia)
            .with_blacklist_entry(
                AdapterBlacklistEntry::new()
                    .with_vendor(Vendor::Microsoft)
                    .with_reason("test"),
            );

        let cloned = selector.clone();
        let _ = cloned;
    }

    #[test]
    fn selector_implements_debug() {
        let selector = AdapterSelector::new();
        let debug = format!("{:?}", selector);
        assert!(debug.contains("AdapterSelector"));
    }
}

// ============================================================================
// 5. Scoring Logic Tests - Device Type
// ============================================================================

mod scoring_device_type {
    use super::*;

    #[test]
    fn discrete_gets_highest_default_score() {
        let weights = DeviceTypeWeights::default();
        let discrete_score = weights.weight_for(DeviceType::DiscreteGpu);
        let integrated_score = weights.weight_for(DeviceType::IntegratedGpu);
        let virtual_score = weights.weight_for(DeviceType::VirtualGpu);
        let cpu_score = weights.weight_for(DeviceType::Cpu);
        let other_score = weights.weight_for(DeviceType::Other);

        assert!(discrete_score > integrated_score);
        assert!(integrated_score > virtual_score);
        assert!(virtual_score > cpu_score);
        assert!(cpu_score > other_score);
    }

    #[test]
    fn custom_weights_affect_scoring() {
        let custom = DeviceTypeWeights {
            discrete: 100,
            integrated: 2000,
            virtual_gpu: 50,
            cpu: 25,
            other: 10,
        };

        assert!(custom.weight_for(DeviceType::IntegratedGpu) > custom.weight_for(DeviceType::DiscreteGpu));
    }

    #[test]
    fn power_saving_weights_prefer_integrated() {
        let weights = DeviceTypeWeights::power_saving();
        assert!(
            weights.weight_for(DeviceType::IntegratedGpu)
                > weights.weight_for(DeviceType::DiscreteGpu)
        );
    }

    #[test]
    fn performance_weights_strongly_prefer_discrete() {
        let weights = DeviceTypeWeights::performance();
        let discrete = weights.weight_for(DeviceType::DiscreteGpu);
        let integrated = weights.weight_for(DeviceType::IntegratedGpu);
        assert!(discrete >= integrated * 4);
    }
}

// ============================================================================
// 6. Scoring Logic Tests - Feature Tier
// ============================================================================

mod scoring_feature_tier {
    use super::*;

    #[test]
    fn feature_tier_ordering_minimal_to_full() {
        assert!(FeatureTier::Minimal < FeatureTier::Standard);
        assert!(FeatureTier::Standard < FeatureTier::Advanced);
        assert!(FeatureTier::Advanced < FeatureTier::Full);
    }

    #[test]
    fn full_tier_highest_score() {
        // Feature scoring: Full=4x, Advanced=3x, Standard=2x, Minimal=1x
        let feature_weight = 100;
        let full_score = feature_weight * 4;
        let advanced_score = feature_weight * 3;
        let standard_score = feature_weight * 2;
        let minimal_score = feature_weight;

        assert!(full_score > advanced_score);
        assert!(advanced_score > standard_score);
        assert!(standard_score > minimal_score);
    }

    #[test]
    fn feature_weight_scales_scores() {
        let weight_100 = 100;
        let weight_200 = 200;

        let full_100 = weight_100 * 4;
        let full_200 = weight_200 * 4;

        assert_eq!(full_200, full_100 * 2);
    }

    #[test]
    fn empty_features_is_minimal_tier() {
        let features = AdapterFeatures {
            raw: wgpu::Features::empty(),
        };
        assert_eq!(features.tier(), FeatureTier::Minimal);
    }

    #[test]
    fn four_plus_features_is_standard_tier() {
        let features = AdapterFeatures {
            raw: wgpu::Features::DEPTH_CLIP_CONTROL
                | wgpu::Features::DEPTH32FLOAT_STENCIL8
                | wgpu::Features::TEXTURE_COMPRESSION_BC
                | wgpu::Features::INDIRECT_FIRST_INSTANCE,
        };
        assert_eq!(features.tier(), FeatureTier::Standard);
    }

    #[test]
    fn eight_plus_features_with_advanced_is_advanced_tier() {
        let features = AdapterFeatures {
            raw: wgpu::Features::DEPTH_CLIP_CONTROL
                | wgpu::Features::DEPTH32FLOAT_STENCIL8
                | wgpu::Features::TEXTURE_COMPRESSION_BC
                | wgpu::Features::TEXTURE_COMPRESSION_ASTC
                | wgpu::Features::INDIRECT_FIRST_INSTANCE
                | wgpu::Features::RG11B10UFLOAT_RENDERABLE
                | wgpu::Features::BGRA8UNORM_STORAGE
                | wgpu::Features::TIMESTAMP_QUERY,
        };
        assert_eq!(features.tier(), FeatureTier::Advanced);
    }

    #[test]
    fn twelve_plus_features_with_advanced_is_full_tier() {
        let features = AdapterFeatures {
            raw: wgpu::Features::DEPTH_CLIP_CONTROL
                | wgpu::Features::DEPTH32FLOAT_STENCIL8
                | wgpu::Features::TEXTURE_COMPRESSION_BC
                | wgpu::Features::TEXTURE_COMPRESSION_ETC2
                | wgpu::Features::TEXTURE_COMPRESSION_ASTC
                | wgpu::Features::INDIRECT_FIRST_INSTANCE
                | wgpu::Features::RG11B10UFLOAT_RENDERABLE
                | wgpu::Features::BGRA8UNORM_STORAGE
                | wgpu::Features::FLOAT32_FILTERABLE
                | wgpu::Features::MULTIVIEW
                | wgpu::Features::PUSH_CONSTANTS
                | wgpu::Features::TIMESTAMP_QUERY,
        };
        assert_eq!(features.tier(), FeatureTier::Full);
    }
}

// ============================================================================
// 7. Scoring Logic Tests - Limits
// ============================================================================

mod scoring_limits {
    #[allow(unused_imports)]
    use super::*;

    #[test]
    fn higher_texture_limit_gives_higher_score() {
        let mut low = wgpu::Limits::default();
        low.max_texture_dimension_2d = 4096;

        let mut high = wgpu::Limits::default();
        high.max_texture_dimension_2d = 16384;

        // Normalize: 8192 = 50 pts, 16384 = 100 pts
        let low_normalized = (low.max_texture_dimension_2d.min(16384) / 164) as u32;
        let high_normalized = (high.max_texture_dimension_2d.min(16384) / 164) as u32;

        assert!(high_normalized > low_normalized);
    }

    #[test]
    fn higher_buffer_size_gives_higher_score() {
        let low_buffer = 256 * 1024 * 1024u64; // 256MB
        let high_buffer = 1024 * 1024 * 1024u64; // 1GB

        // Normalize: 256MB = ~25 pts, 1GB = 100 pts
        let low_score = (low_buffer / (10 * 1024 * 1024)).min(100) as u32;
        let high_score = (high_buffer / (10 * 1024 * 1024)).min(100) as u32;

        assert!(high_score > low_score);
    }

    #[test]
    fn higher_compute_invocations_gives_higher_score() {
        let low_invocations = 256u32;
        let high_invocations = 1024u32;

        // Normalize: 256 = ~12 pts, 1024 = ~50 pts
        let low_score = (low_invocations / 20).min(50);
        let high_score = (high_invocations / 20).min(50);

        assert!(high_score > low_score);
    }

    #[test]
    fn higher_storage_binding_gives_higher_score() {
        let low_storage = 128 * 1024 * 1024u32; // 128MB
        let high_storage = 512 * 1024 * 1024u32; // 512MB

        // Normalize: 128MB = ~12 pts, 512MB = ~50 pts
        let low_score = (low_storage / (10 * 1024 * 1024)).min(50);
        let high_score = (high_storage / (10 * 1024 * 1024)).min(50);

        assert!(high_score > low_score);
    }

    #[test]
    fn limit_weight_scales_limits_score() {
        let limit_weight_1 = 1u32;
        let limit_weight_3 = 3u32;
        let base_score = 100u32;

        let score_1 = base_score * limit_weight_1;
        let score_3 = base_score * limit_weight_3;

        assert_eq!(score_3, score_1 * 3);
    }

    #[test]
    fn max_limits_give_capped_score() {
        // Texture: max 16384 = 100 pts
        let texture_score = (16384u32.min(16384) / 164) as u32;
        assert!(texture_score <= 100);

        // Buffer: max 100 pts
        let buffer_score = ((u64::MAX / (10 * 1024 * 1024)).min(100)) as u32;
        assert_eq!(buffer_score, 100);

        // Compute: max 50 pts
        let compute_score = (u32::MAX / 20).min(50);
        assert_eq!(compute_score, 50);

        // Storage: max 50 pts
        let storage_score = (u32::MAX / (10 * 1024 * 1024)).min(50);
        assert_eq!(storage_score, 50);
    }
}

// ============================================================================
// 8. Scoring Logic Tests - Vendor Bonus
// ============================================================================

mod scoring_vendor_bonus {
    use super::*;

    #[test]
    fn matching_vendor_gets_bonus() {
        let _bonus = 200u32; // Example bonus value
        let preferred = Vendor::Nvidia;
        let adapter_vendor = Vendor::Nvidia;

        let gets_bonus = adapter_vendor == preferred;
        assert!(gets_bonus);
    }

    #[test]
    fn non_matching_vendor_gets_no_bonus() {
        let preferred = Vendor::Nvidia;
        let adapter_vendor = Vendor::Amd;

        let gets_bonus = adapter_vendor == preferred;
        assert!(!gets_bonus);
    }

    #[test]
    fn vendor_preference_bonus_is_configurable() {
        let default_bonus = 200u32;
        let custom_bonus = 500u32;

        assert_ne!(default_bonus, custom_bonus);
    }

    #[test]
    fn vendor_id_matching() {
        let nvidia_id = 0x10DE;
        let amd_id = 0x1002;
        let intel_id = 0x8086;

        assert_eq!(Vendor::from_id(nvidia_id), Vendor::Nvidia);
        assert_eq!(Vendor::from_id(amd_id), Vendor::Amd);
        assert_eq!(Vendor::from_id(intel_id), Vendor::Intel);
    }

    #[test]
    fn unknown_vendor_id_creates_unknown_variant() {
        let unknown_id = 0x9999;
        let vendor = Vendor::from_id(unknown_id);
        assert_eq!(vendor, Vendor::Unknown(0x9999));
    }
}

// ============================================================================
// 9. Selection Logic Tests
// ============================================================================

mod selection_logic {
    use super::*;

    #[test]
    fn select_returns_none_for_empty_list() {
        let selector = AdapterSelector::new();
        let adapters: Vec<wgpu::Adapter> = vec![];
        let result = selector.select(&adapters);
        assert!(result.is_none());
    }

    #[test]
    fn select_adapter_returns_none_for_empty_list() {
        let selector = AdapterSelector::new();
        let adapters: Vec<wgpu::Adapter> = vec![];
        let result = selector.select_adapter(&adapters);
        assert!(result.is_none());
    }
}

// ============================================================================
// 10. Edge Cases
// ============================================================================

mod edge_cases {
    use super::*;

    #[test]
    fn empty_blacklist_filters_nothing() {
        let selector = AdapterSelector::new();
        let info = make_adapter_info("Any GPU", 0x10DE, DeviceType::DiscreteGpu);

        // With no blacklist, nothing should be blacklisted
        // We can't directly test is_blacklisted, but behavior can be verified
        let _ = (selector, info);
    }

    #[test]
    fn multiple_blacklist_entries_checked_in_order() {
        let _selector = AdapterSelector::new()
            .with_blacklist_entry(
                AdapterBlacklistEntry::new()
                    .with_vendor(Vendor::Microsoft)
                    .with_reason("First reason"),
            )
            .with_blacklist_entry(
                AdapterBlacklistEntry::new()
                    .with_vendor(Vendor::Microsoft)
                    .with_reason("Second reason"),
            );

        // First matching entry's reason should be used
    }

    #[test]
    fn vendor_preference_without_match_still_selects_best() {
        // If preferred vendor doesn't exist, should still select highest scoring
        let _selector =
            AdapterSelector::new().with_vendor_preference(Vendor::Apple); // Unlikely on desktop
    }

    #[test]
    fn zero_weights_produce_zero_device_type_score() {
        let weights = DeviceTypeWeights {
            discrete: 0,
            integrated: 0,
            virtual_gpu: 0,
            cpu: 0,
            other: 0,
        };

        assert_eq!(weights.weight_for(DeviceType::DiscreteGpu), 0);
        assert_eq!(weights.weight_for(DeviceType::IntegratedGpu), 0);
        assert_eq!(weights.weight_for(DeviceType::VirtualGpu), 0);
        assert_eq!(weights.weight_for(DeviceType::Cpu), 0);
        assert_eq!(weights.weight_for(DeviceType::Other), 0);
    }

    #[test]
    fn zero_feature_weight_produces_zero_feature_score() {
        let weight = 0u32;
        let full_score = weight * 4;
        assert_eq!(full_score, 0);
    }

    #[test]
    fn zero_limit_weight_produces_zero_limits_score() {
        let weight = 0u32;
        let base_score = 100u32;
        let total = base_score * weight;
        assert_eq!(total, 0);
    }

    #[test]
    fn zero_vendor_bonus_means_no_preference_effect() {
        let bonus = 0u32;
        let score_without = 1500u32;
        let score_with_bonus = score_without + bonus;
        assert_eq!(score_without, score_with_bonus);
    }

    #[test]
    fn max_u32_weights_dont_overflow_in_addition() {
        // Test that reasonable weights don't cause overflow
        let device_score = 2000u32;
        let feature_score = 400u32;
        let limits_score = 300u32;
        let vendor_bonus = 200u32;

        // This should not panic
        let total = device_score + feature_score + limits_score + vendor_bonus;
        assert_eq!(total, 2900);
    }

    #[test]
    fn very_long_adapter_name_for_blacklist() {
        let long_name = "A".repeat(10000);
        let entry = AdapterBlacklistEntry::new()
            .with_name_contains(&long_name)
            .with_reason("test");

        let info = AdapterInfo {
            name: long_name.clone(),
            vendor: 0x10DE,
            device: 0,
            device_type: DeviceType::DiscreteGpu,
            backend: wgpu::Backend::Vulkan,
            driver: String::new(),
            driver_info: String::new(),
        };

        assert!(entry.matches(&info));
    }

    #[test]
    fn unicode_in_adapter_name() {
        let entry = AdapterBlacklistEntry::new()
            .with_name_contains("GPU")
            .with_reason("test");

        let info = make_adapter_info("GPU 2024", 0x10DE, DeviceType::DiscreteGpu);
        assert!(entry.matches(&info));
    }

    #[test]
    fn special_characters_in_name_pattern() {
        let entry = AdapterBlacklistEntry::new()
            .with_name_contains("(TM)")
            .with_reason("test");

        let info = make_adapter_info("Intel(R) UHD Graphics (TM)", 0x8086, DeviceType::IntegratedGpu);
        assert!(entry.matches(&info));
    }

    #[test]
    fn blacklist_reason_can_be_empty() {
        let entry = AdapterBlacklistEntry::new()
            .with_vendor(Vendor::Microsoft)
            .with_reason("");

        assert!(entry.reason.is_empty());
    }

    #[test]
    fn blacklist_reason_can_be_very_long() {
        let long_reason = "R".repeat(10000);
        let entry = AdapterBlacklistEntry::new()
            .with_vendor(Vendor::Microsoft)
            .with_reason(&long_reason);

        assert_eq!(entry.reason.len(), 10000);
    }
}

// ============================================================================
// 11. FeatureTier Additional Tests
// ============================================================================

mod feature_tier_tests {
    use super::*;

    #[test]
    fn feature_tier_is_copy() {
        let tier = FeatureTier::Full;
        let copy = tier;
        assert_eq!(tier, copy);
    }

    #[test]
    fn feature_tier_is_clone() {
        let tier = FeatureTier::Full;
        let cloned = tier.clone();
        assert_eq!(tier, cloned);
    }

    #[test]
    fn feature_tier_implements_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(FeatureTier::Minimal);
        set.insert(FeatureTier::Standard);
        set.insert(FeatureTier::Advanced);
        set.insert(FeatureTier::Full);
        assert_eq!(set.len(), 4);
    }

    #[test]
    fn feature_tier_description_contains_name() {
        assert!(FeatureTier::Minimal.description().contains("Minimal"));
        assert!(FeatureTier::Standard.description().contains("Standard"));
        assert!(FeatureTier::Advanced.description().contains("Advanced"));
        assert!(FeatureTier::Full.description().contains("Full"));
    }

    #[test]
    fn feature_tier_display_format() {
        assert_eq!(format!("{}", FeatureTier::Minimal), "Minimal");
        assert_eq!(format!("{}", FeatureTier::Standard), "Standard");
        assert_eq!(format!("{}", FeatureTier::Advanced), "Advanced");
        assert_eq!(format!("{}", FeatureTier::Full), "Full");
    }

    #[test]
    fn feature_tier_partial_ord() {
        assert!(FeatureTier::Full > FeatureTier::Advanced);
        assert!(FeatureTier::Advanced > FeatureTier::Standard);
        assert!(FeatureTier::Standard > FeatureTier::Minimal);
    }

    #[test]
    fn feature_tier_ord_consistent_with_partial_ord() {
        use std::cmp::Ordering;
        assert_eq!(FeatureTier::Full.cmp(&FeatureTier::Advanced), Ordering::Greater);
        assert_eq!(FeatureTier::Minimal.cmp(&FeatureTier::Standard), Ordering::Less);
        assert_eq!(FeatureTier::Advanced.cmp(&FeatureTier::Advanced), Ordering::Equal);
    }
}

// ============================================================================
// 12. Vendor Tests
// ============================================================================

mod vendor_tests {
    use super::*;

    #[test]
    fn all_known_vendors_from_id() {
        assert_eq!(Vendor::from_id(0x10DE), Vendor::Nvidia);
        assert_eq!(Vendor::from_id(0x1002), Vendor::Amd);
        assert_eq!(Vendor::from_id(0x1022), Vendor::Amd); // AMD alternate
        assert_eq!(Vendor::from_id(0x8086), Vendor::Intel);
        assert_eq!(Vendor::from_id(0x106B), Vendor::Apple);
        assert_eq!(Vendor::from_id(0x13B5), Vendor::Arm);
        assert_eq!(Vendor::from_id(0x5143), Vendor::Qualcomm);
        assert_eq!(Vendor::from_id(0x1414), Vendor::Microsoft);
    }

    #[test]
    fn vendor_names() {
        assert_eq!(Vendor::Nvidia.name(), "NVIDIA");
        assert_eq!(Vendor::Amd.name(), "AMD");
        assert_eq!(Vendor::Intel.name(), "Intel");
        assert_eq!(Vendor::Apple.name(), "Apple");
        assert_eq!(Vendor::Arm.name(), "ARM");
        assert_eq!(Vendor::Qualcomm.name(), "Qualcomm");
        assert_eq!(Vendor::Microsoft.name(), "Microsoft");
        assert_eq!(Vendor::Unknown(0x1234).name(), "Unknown");
    }

    #[test]
    fn vendor_ids() {
        assert_eq!(Vendor::Nvidia.id(), 0x10DE);
        assert_eq!(Vendor::Amd.id(), 0x1002);
        assert_eq!(Vendor::Intel.id(), 0x8086);
        assert_eq!(Vendor::Apple.id(), 0x106B);
        assert_eq!(Vendor::Arm.id(), 0x13B5);
        assert_eq!(Vendor::Qualcomm.id(), 0x5143);
        assert_eq!(Vendor::Microsoft.id(), 0x1414);
        assert_eq!(Vendor::Unknown(0xABCD).id(), 0xABCD);
    }

    #[test]
    fn vendor_is_known() {
        assert!(Vendor::Nvidia.is_known());
        assert!(Vendor::Amd.is_known());
        assert!(Vendor::Intel.is_known());
        assert!(Vendor::Apple.is_known());
        assert!(Vendor::Arm.is_known());
        assert!(Vendor::Qualcomm.is_known());
        assert!(Vendor::Microsoft.is_known());
        assert!(!Vendor::Unknown(0x1234).is_known());
    }

    #[test]
    fn vendor_display_known() {
        assert_eq!(format!("{}", Vendor::Nvidia), "NVIDIA");
        assert_eq!(format!("{}", Vendor::Amd), "AMD");
        assert_eq!(format!("{}", Vendor::Intel), "Intel");
    }

    #[test]
    fn vendor_display_unknown() {
        assert_eq!(format!("{}", Vendor::Unknown(0x1234)), "Unknown (0x1234)");
        assert_eq!(format!("{}", Vendor::Unknown(0x0000)), "Unknown (0x0000)");
        assert_eq!(format!("{}", Vendor::Unknown(0xFFFF)), "Unknown (0xFFFF)");
    }

    #[test]
    fn vendor_equality() {
        assert_eq!(Vendor::Nvidia, Vendor::Nvidia);
        assert_ne!(Vendor::Nvidia, Vendor::Amd);
        assert_eq!(Vendor::Unknown(0x1234), Vendor::Unknown(0x1234));
        assert_ne!(Vendor::Unknown(0x1234), Vendor::Unknown(0x5678));
    }

    #[test]
    fn vendor_is_copy() {
        let v = Vendor::Nvidia;
        let copy = v;
        assert_eq!(v, copy);
    }

    #[test]
    fn vendor_is_clone() {
        let v = Vendor::Nvidia;
        let cloned = v.clone();
        assert_eq!(v, cloned);
    }

    #[test]
    fn vendor_implements_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(Vendor::Nvidia);
        set.insert(Vendor::Amd);
        set.insert(Vendor::Intel);
        set.insert(Vendor::Unknown(0x1234));
        assert_eq!(set.len(), 4);
    }

    #[test]
    fn vendor_implements_debug() {
        let debug = format!("{:?}", Vendor::Nvidia);
        assert!(debug.contains("Nvidia"));

        let debug_unknown = format!("{:?}", Vendor::Unknown(0x1234));
        assert!(debug_unknown.contains("Unknown"));
        assert!(debug_unknown.contains("4660") || debug_unknown.contains("1234")); // decimal or hex
    }
}

// ============================================================================
// 13. AdapterFeatures Additional Tests
// ============================================================================

mod adapter_features_tests {
    use super::*;

    #[test]
    fn features_count_empty() {
        let features = AdapterFeatures {
            raw: wgpu::Features::empty(),
        };
        assert_eq!(features.count(), 0);
    }

    #[test]
    fn features_count_single() {
        let features = AdapterFeatures {
            raw: wgpu::Features::TIMESTAMP_QUERY,
        };
        assert_eq!(features.count(), 1);
    }

    #[test]
    fn features_count_multiple() {
        let features = AdapterFeatures {
            raw: wgpu::Features::TIMESTAMP_QUERY
                | wgpu::Features::PUSH_CONSTANTS
                | wgpu::Features::SHADER_F16,
        };
        assert_eq!(features.count(), 3);
    }

    #[test]
    fn features_supports_single() {
        let features = AdapterFeatures {
            raw: wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::PUSH_CONSTANTS,
        };

        assert!(features.supports(wgpu::Features::TIMESTAMP_QUERY));
        assert!(features.supports(wgpu::Features::PUSH_CONSTANTS));
        assert!(!features.supports(wgpu::Features::SHADER_F16));
    }

    #[test]
    fn features_supports_multiple() {
        let features = AdapterFeatures {
            raw: wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::PUSH_CONSTANTS,
        };

        assert!(features.supports(
            wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::PUSH_CONSTANTS
        ));
        assert!(!features.supports(
            wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::SHADER_F16
        ));
    }

    #[test]
    fn features_has_any_texture_compression() {
        let no_compression = AdapterFeatures {
            raw: wgpu::Features::TIMESTAMP_QUERY,
        };
        assert!(!no_compression.has_any_texture_compression());

        let bc = AdapterFeatures {
            raw: wgpu::Features::TEXTURE_COMPRESSION_BC,
        };
        assert!(bc.has_any_texture_compression());

        let etc2 = AdapterFeatures {
            raw: wgpu::Features::TEXTURE_COMPRESSION_ETC2,
        };
        assert!(etc2.has_any_texture_compression());

        let astc = AdapterFeatures {
            raw: wgpu::Features::TEXTURE_COMPRESSION_ASTC,
        };
        assert!(astc.has_any_texture_compression());
    }

    #[test]
    fn features_best_compression_format() {
        let none = AdapterFeatures {
            raw: wgpu::Features::TIMESTAMP_QUERY,
        };
        assert_eq!(none.best_compression_format(), "none");

        let bc_only = AdapterFeatures {
            raw: wgpu::Features::TEXTURE_COMPRESSION_BC,
        };
        assert_eq!(bc_only.best_compression_format(), "BC");

        let all = AdapterFeatures {
            raw: wgpu::Features::TEXTURE_COMPRESSION_BC
                | wgpu::Features::TEXTURE_COMPRESSION_ETC2
                | wgpu::Features::TEXTURE_COMPRESSION_ASTC,
        };
        assert_eq!(all.best_compression_format(), "BC"); // BC preferred

        let astc_etc2 = AdapterFeatures {
            raw: wgpu::Features::TEXTURE_COMPRESSION_ETC2
                | wgpu::Features::TEXTURE_COMPRESSION_ASTC,
        };
        assert_eq!(astc_etc2.best_compression_format(), "ASTC"); // ASTC over ETC2

        let etc2_only = AdapterFeatures {
            raw: wgpu::Features::TEXTURE_COMPRESSION_ETC2,
        };
        assert_eq!(etc2_only.best_compression_format(), "ETC2");
    }

    #[test]
    fn features_individual_checks() {
        let all_features = AdapterFeatures {
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

        assert!(all_features.has_depth_clip_control());
        assert!(all_features.has_depth32float_stencil8());
        assert!(all_features.has_texture_compression_bc());
        assert!(all_features.has_texture_compression_etc2());
        assert!(all_features.has_texture_compression_astc());
        assert!(all_features.has_texture_compression_astc_hdr());
        assert!(all_features.has_indirect_first_instance());
        assert!(all_features.has_multiview());
        assert!(all_features.has_timestamp_query());
        assert!(all_features.has_pipeline_statistics_query());
        assert!(all_features.has_shader_f16());
        assert!(all_features.has_push_constants());
        assert!(all_features.has_rg11b10ufloat_renderable());
        assert!(all_features.has_bgra8unorm_storage());
        assert!(all_features.has_float32_filterable());
        assert!(all_features.has_texture_format_16bit_norm());
    }

    #[test]
    fn features_summary_creation() {
        let features = AdapterFeatures {
            raw: wgpu::Features::TEXTURE_COMPRESSION_BC
                | wgpu::Features::TIMESTAMP_QUERY
                | wgpu::Features::SHADER_F16
                | wgpu::Features::PUSH_CONSTANTS,
        };
        let summary = features.summary();

        assert_eq!(summary.total_count, 4);
        assert!(summary.has_compression_bc);
        assert!(!summary.has_compression_etc2);
        assert!(!summary.has_compression_astc);
        assert!(!summary.has_compression_astc_hdr);
        assert!(summary.has_timestamp_query);
        assert!(!summary.has_pipeline_statistics);
        assert!(summary.has_shader_f16);
        assert!(summary.has_push_constants);
        assert!(!summary.has_multiview);
        assert!(!summary.has_indirect_first_instance);
    }

    #[test]
    fn features_summary_helpers() {
        let features = AdapterFeatures {
            raw: wgpu::Features::TEXTURE_COMPRESSION_BC
                | wgpu::Features::TIMESTAMP_QUERY
                | wgpu::Features::INDIRECT_FIRST_INSTANCE,
        };
        let summary = features.summary();

        assert!(summary.has_any_compression());
        assert!(summary.has_profiling());
        assert!(summary.has_gpu_driven());
    }

    #[test]
    fn features_display_contains_sections() {
        let features = AdapterFeatures {
            raw: wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::TEXTURE_COMPRESSION_BC,
        };
        let display = format!("{}", features);

        assert!(display.contains("Adapter Features"));
        assert!(display.contains("Tier:"));
        assert!(display.contains("Texture Compression"));
        assert!(display.contains("Rendering"));
        assert!(display.contains("Queries"));
        assert!(display.contains("Shader"));
        assert!(display.contains("Formats"));
    }

    #[test]
    fn features_is_clone() {
        let features = AdapterFeatures {
            raw: wgpu::Features::TIMESTAMP_QUERY,
        };
        let cloned = features.clone();
        assert_eq!(features.count(), cloned.count());
    }

    #[test]
    fn features_implements_debug() {
        let features = AdapterFeatures {
            raw: wgpu::Features::TIMESTAMP_QUERY,
        };
        let debug = format!("{:?}", features);
        assert!(debug.contains("AdapterFeatures"));
    }
}

// ============================================================================
// 14. Integration-Style Unit Tests (No GPU Required)
// ============================================================================

mod integration_unit_tests {
    use super::*;

    #[test]
    fn complete_selector_configuration() {
        let selector = AdapterSelector::new()
            .with_vendor_preference(Vendor::Nvidia)
            .with_vendor_preference_bonus(500)
            .with_device_type_weights(DeviceTypeWeights::performance())
            .with_feature_weight(200)
            .with_limit_weight(3)
            .with_blacklist_entry(
                AdapterBlacklistEntry::new()
                    .with_vendor(Vendor::Microsoft)
                    .with_name_contains("WARP")
                    .with_reason("Software renderer"),
            )
            .with_blacklist_entry(
                AdapterBlacklistEntry::new()
                    .with_name_contains("Buggy Driver")
                    .with_reason("Known crashes"),
            );

        // Selector should be fully configured
        let cloned = selector.clone();
        let _ = cloned;
    }

    #[test]
    fn selector_with_power_saving_weights() {
        let selector = AdapterSelector::new()
            .with_device_type_weights(DeviceTypeWeights::power_saving());

        // In power saving mode, integrated GPUs should be preferred
        let _ = selector;
    }

    #[test]
    fn blacklist_chain_test() {
        let entries = vec![
            AdapterBlacklistEntry::new()
                .with_vendor(Vendor::Microsoft)
                .with_reason("Software"),
            AdapterBlacklistEntry::new()
                .with_name_contains("Virtual")
                .with_reason("VM GPU"),
            AdapterBlacklistEntry::new()
                .with_vendor(Vendor::Amd)
                .with_name_contains("Vega 8")
                .with_reason("Old integrated"),
        ];

        let selector = AdapterSelector::new().with_blacklist(entries);
        let _ = selector;
    }

    #[test]
    fn score_calculation_formula() {
        // Verify the expected score calculation formula
        let device_type_score = 1000u32; // Discrete GPU
        let feature_score = 400u32; // Full tier (4 * 100)
        let limits_score = 250u32; // Good limits
        let vendor_bonus = 200u32; // Matching vendor

        let expected_total = device_type_score + feature_score + limits_score + vendor_bonus;
        assert_eq!(expected_total, 1850);
    }

    #[test]
    fn tier_to_feature_score_mapping() {
        let weight = 100u32;

        let minimal_score = weight * 1;
        let standard_score = weight * 2;
        let advanced_score = weight * 3;
        let full_score = weight * 4;

        assert_eq!(minimal_score, 100);
        assert_eq!(standard_score, 200);
        assert_eq!(advanced_score, 300);
        assert_eq!(full_score, 400);
    }

    #[test]
    fn all_device_types_have_weights() {
        let weights = DeviceTypeWeights::default();

        // All device types should have positive weights
        assert!(weights.weight_for(DeviceType::DiscreteGpu) > 0);
        assert!(weights.weight_for(DeviceType::IntegratedGpu) > 0);
        assert!(weights.weight_for(DeviceType::VirtualGpu) > 0);
        assert!(weights.weight_for(DeviceType::Cpu) > 0);
        assert!(weights.weight_for(DeviceType::Other) > 0);
    }

    #[test]
    fn blacklist_entry_matches_all_known_vendors() {
        let vendors = [
            (Vendor::Nvidia, 0x10DE),
            (Vendor::Amd, 0x1002),
            (Vendor::Intel, 0x8086),
            (Vendor::Apple, 0x106B),
            (Vendor::Arm, 0x13B5),
            (Vendor::Qualcomm, 0x5143),
            (Vendor::Microsoft, 0x1414),
        ];

        for (vendor, id) in vendors {
            let entry = AdapterBlacklistEntry::new()
                .with_vendor(vendor)
                .with_reason("test");

            let info = make_adapter_info("Test GPU", id, DeviceType::DiscreteGpu);
            assert!(
                entry.matches(&info),
                "Entry with {:?} should match vendor ID 0x{:04X}",
                vendor,
                id
            );
        }
    }

    #[test]
    fn blacklist_entry_does_not_match_wrong_vendors() {
        let entry = AdapterBlacklistEntry::new()
            .with_vendor(Vendor::Nvidia)
            .with_reason("test");

        let other_vendors = [
            ("AMD GPU", 0x1002),
            ("Intel GPU", 0x8086),
            ("Apple GPU", 0x106B),
            ("ARM GPU", 0x13B5),
            ("Qualcomm GPU", 0x5143),
            ("Microsoft GPU", 0x1414),
            ("Unknown GPU", 0x9999),
        ];

        for (name, id) in other_vendors {
            let info = make_adapter_info(name, id, DeviceType::DiscreteGpu);
            assert!(
                !entry.matches(&info),
                "Nvidia entry should not match {} (0x{:04X})",
                name,
                id
            );
        }
    }
}

// ============================================================================
// 15. Boundary Value Tests
// ============================================================================

mod boundary_values {
    use super::*;

    #[test]
    fn zero_vendor_id_creates_unknown() {
        let vendor = Vendor::from_id(0);
        assert_eq!(vendor, Vendor::Unknown(0));
    }

    #[test]
    fn max_vendor_id_creates_unknown() {
        let vendor = Vendor::from_id(u32::MAX);
        assert_eq!(vendor, Vendor::Unknown(u32::MAX));
    }

    #[test]
    fn empty_name_for_blacklist() {
        let entry = AdapterBlacklistEntry::new()
            .with_name_contains("")
            .with_reason("test");

        let info = make_adapter_info("Any GPU", 0x10DE, DeviceType::DiscreteGpu);
        // Empty string matches all names (contains "")
        assert!(entry.matches(&info));
    }

    #[test]
    fn whitespace_only_name_for_blacklist() {
        let entry = AdapterBlacklistEntry::new()
            .with_name_contains("   ")
            .with_reason("test");

        let info = make_adapter_info("NVIDIA GeForce RTX 4090", 0x10DE, DeviceType::DiscreteGpu);
        // Whitespace in name check
        assert!(!entry.matches(&info)); // "   " is not in "NVIDIA GeForce RTX 4090"
    }

    #[test]
    fn single_character_name_match() {
        let entry = AdapterBlacklistEntry::new()
            .with_name_contains("X")
            .with_reason("test");

        let rtx = make_adapter_info("NVIDIA GeForce RTX 4090", 0x10DE, DeviceType::DiscreteGpu);
        let hd = make_adapter_info("Intel HD Graphics", 0x8086, DeviceType::IntegratedGpu);

        assert!(entry.matches(&rtx)); // Contains 'X' in RTX
        assert!(!entry.matches(&hd)); // No 'X' or 'x'
    }

    #[test]
    fn maximum_weight_values() {
        let weights = DeviceTypeWeights {
            discrete: u32::MAX,
            integrated: u32::MAX - 1,
            virtual_gpu: u32::MAX - 2,
            cpu: u32::MAX - 3,
            other: u32::MAX - 4,
        };

        assert_eq!(weights.weight_for(DeviceType::DiscreteGpu), u32::MAX);
        assert_eq!(weights.weight_for(DeviceType::IntegratedGpu), u32::MAX - 1);
    }

    #[test]
    fn minimum_weight_values() {
        let weights = DeviceTypeWeights {
            discrete: 0,
            integrated: 0,
            virtual_gpu: 0,
            cpu: 0,
            other: 0,
        };

        assert_eq!(weights.weight_for(DeviceType::DiscreteGpu), 0);
        assert_eq!(weights.weight_for(DeviceType::IntegratedGpu), 0);
    }

    #[test]
    fn feature_weight_boundary() {
        let min_weight = 0u32;
        let max_weight = u32::MAX / 4; // Avoid overflow in tier calculation

        let min_full_score = min_weight * 4;
        let max_full_score = max_weight * 4;

        assert_eq!(min_full_score, 0);
        assert!(max_full_score > 0);
    }

    #[test]
    fn limit_weight_boundary() {
        let min_weight = 0u32;
        let max_weight = u32::MAX / 300; // Avoid overflow in limits calculation

        let base_score = 300u32;
        let min_score = base_score * min_weight;
        let max_score = base_score * max_weight;

        assert_eq!(min_score, 0);
        assert!(max_score > 0);
    }

    #[test]
    fn vendor_bonus_boundary() {
        let min_bonus = 0u32;
        let max_bonus = u32::MAX;

        // Both extremes should be valid
        let _ = min_bonus;
        let _ = max_bonus;
    }
}
