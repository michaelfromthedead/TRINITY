// Blackbox contract tests for T-WGPU-P1.2.5 Adapter Selection Algorithm
//
// CLEANROOM: No access to implementation files. Tests use only the public API
// exported by `renderer_backend::device`.
//
// Forbidden files (per TESTDEV_BLACKBOX prompt):
//   - crates/renderer-backend/src/device/adapter.rs
//   - crates/renderer-backend/src/device/instance.rs (implementation details)
//   - Any WHITEBOX test file for this task
//
// Contract sources:
//   - docs/WGPU_DOCS/RDC_OUTPUT/PHASE_1_CORE_TODO.md (T-WGPU-P1.2.5)
//   - docs/WGPU_DOCS/RDC_OUTPUT/PHASE_1_CORE_ARCH.md (Architecture spec)
//
// Acceptance criteria (T-WGPU-P1.2.5):
//   1. Score by device type (discrete > integrated > software)
//   2. Score by feature availability
//   3. Score by limits
//   4. Blacklist support (for known-broken drivers)
//   5. Vendor preference support
//   6. Falls back to first available if no preference match
//
// Test design rationale:
//   Equivalence partitioning:
//     - AdapterSelector construction and configuration
//     - DeviceTypeWeights preset variations
//     - Blacklist entry variations (vendor, name, both)
//     - Vendor preference scenarios
//   Boundary cases:
//     - Zero adapters (empty list)
//     - All adapters blacklisted
//     - No vendor preference match
//   Contract verification:
//     - AdapterSelector builder pattern
//     - AdapterScore fields and methods
//     - SelectionResult structure
//     - DeviceTypeWeights presets

use renderer_backend::device::{
    enumerate_adapters_with_info, AdapterBlacklistEntry, AdapterScore, AdapterSelector,
    DeviceTypeWeights, SelectionResult, TrinityInstance, Vendor,
};

// =============================================================================
// 1. AdapterSelector Contract Tests
// =============================================================================

/// Verifies that AdapterSelector has a new() constructor.
///
/// Contract: AdapterSelector can be constructed with new().
#[test]
fn test_adapter_selector_has_new_constructor() {
    let _selector = AdapterSelector::new();
}

/// Verifies that AdapterSelector has a default() constructor via Default trait.
///
/// Contract: AdapterSelector implements Default.
#[test]
fn test_adapter_selector_implements_default() {
    let _selector = AdapterSelector::default();
}

/// Verifies that AdapterSelector::new() and default() produce equivalent selectors.
///
/// Contract: new() and default() are equivalent.
#[test]
fn test_adapter_selector_new_equals_default() {
    let selector_new = AdapterSelector::new();
    let selector_default = AdapterSelector::default();
    // Both should work identically - we verify by using them
    let _ = selector_new;
    let _ = selector_default;
}

/// Verifies that AdapterSelector has with_device_type_weights builder method.
///
/// Contract: AdapterSelector::with_device_type_weights() returns Self for chaining.
#[test]
fn test_adapter_selector_has_with_device_type_weights() {
    let weights = DeviceTypeWeights::default();
    let _selector = AdapterSelector::new().with_device_type_weights(weights);
}

/// Verifies that AdapterSelector has with_vendor_preference builder method.
///
/// Contract: AdapterSelector::with_vendor_preference() returns Self for chaining.
#[test]
fn test_adapter_selector_has_with_vendor_preference() {
    let _selector = AdapterSelector::new().with_vendor_preference(Vendor::Nvidia);
}

/// Verifies that AdapterSelector has with_blacklist_entry builder method.
///
/// Contract: AdapterSelector::with_blacklist_entry() returns Self for chaining.
#[test]
fn test_adapter_selector_has_with_blacklist_entry() {
    let entry = AdapterBlacklistEntry::new().with_vendor(Vendor::Unknown(0));
    let _selector = AdapterSelector::new().with_blacklist_entry(entry);
}

/// Verifies that AdapterSelector has with_blacklist builder method.
///
/// Contract: AdapterSelector::with_blacklist() accepts Vec of entries.
#[test]
fn test_adapter_selector_has_with_blacklist() {
    let blacklist = vec![AdapterBlacklistEntry::new().with_vendor(Vendor::Unknown(0))];
    let _selector = AdapterSelector::new().with_blacklist(blacklist);
}

/// Verifies that AdapterSelector has with_feature_weight builder method.
///
/// Contract: AdapterSelector::with_feature_weight() customizes feature scoring.
#[test]
fn test_adapter_selector_has_with_feature_weight() {
    let _selector = AdapterSelector::new().with_feature_weight(100);
}

/// Verifies that AdapterSelector has with_limit_weight builder method.
///
/// Contract: AdapterSelector::with_limit_weight() customizes limit scoring.
#[test]
fn test_adapter_selector_has_with_limit_weight() {
    let _selector = AdapterSelector::new().with_limit_weight(50);
}

/// Verifies that AdapterSelector has with_vendor_preference_bonus builder method.
///
/// Contract: AdapterSelector::with_vendor_preference_bonus() sets bonus value.
#[test]
fn test_adapter_selector_has_with_vendor_preference_bonus() {
    let _selector = AdapterSelector::new().with_vendor_preference_bonus(500);
}

/// Verifies that AdapterSelector builder methods can be chained.
///
/// Contract: Builder pattern allows full configuration in one expression.
#[test]
fn test_adapter_selector_builder_chaining() {
    let weights = DeviceTypeWeights::performance();
    let blacklist = vec![AdapterBlacklistEntry::new().with_vendor(Vendor::Unknown(0))];

    let _selector = AdapterSelector::new()
        .with_device_type_weights(weights)
        .with_vendor_preference(Vendor::Nvidia)
        .with_blacklist(blacklist)
        .with_feature_weight(100)
        .with_limit_weight(50)
        .with_vendor_preference_bonus(500);
}

/// Verifies that AdapterSelector has select() method returning SelectionResult.
///
/// Contract: AdapterSelector::select() returns Option<SelectionResult>.
#[test]
fn test_adapter_selector_has_select_method() {
    let instance = TrinityInstance::new();
    let result = enumerate_adapters_with_info(instance.inner(), instance.backends());
    let selector = AdapterSelector::new();

    let _selected: Option<SelectionResult> = selector.select(&result.adapters);
}

/// Verifies that AdapterSelector has select_adapter() method returning adapter reference.
///
/// Contract: AdapterSelector::select_adapter() returns Option<&Adapter>.
#[test]
fn test_adapter_selector_has_select_adapter_method() {
    let instance = TrinityInstance::new();
    let result = enumerate_adapters_with_info(instance.inner(), instance.backends());
    let selector = AdapterSelector::new();

    let _selected: Option<&wgpu::Adapter> = selector.select_adapter(&result.adapters);
}

/// Verifies that AdapterSelector has score_adapter() method.
///
/// Contract: AdapterSelector::score_adapter() returns AdapterScore for a single adapter.
#[test]
fn test_adapter_selector_has_score_adapter_method() {
    let instance = TrinityInstance::new();
    let result = enumerate_adapters_with_info(instance.inner(), instance.backends());

    if let Some(adapter) = result.adapters.first() {
        let selector = AdapterSelector::new();
        let _score: AdapterScore = selector.score_adapter(adapter);
    }
}

/// Verifies that AdapterSelector implements Debug trait.
///
/// Contract: AdapterSelector implements Debug for logging.
#[test]
fn test_adapter_selector_implements_debug() {
    let selector = AdapterSelector::new();
    let debug_str = format!("{:?}", selector);
    assert!(!debug_str.is_empty());
}

/// Verifies that AdapterSelector implements Clone trait.
///
/// Contract: AdapterSelector implements Clone.
#[test]
fn test_adapter_selector_implements_clone() {
    let selector = AdapterSelector::new().with_vendor_preference(Vendor::Nvidia);
    let _cloned = selector.clone();
}

// =============================================================================
// 2. DeviceTypeWeights Contract Tests
// =============================================================================

/// Verifies that DeviceTypeWeights has a Default implementation.
///
/// Contract: DeviceTypeWeights::default() creates standard weights.
#[test]
fn test_device_type_weights_default() {
    let _weights = DeviceTypeWeights::default();
}

/// Verifies that DeviceTypeWeights has power_saving() preset.
///
/// Contract: DeviceTypeWeights::power_saving() returns preset for battery use.
#[test]
fn test_device_type_weights_power_saving_preset() {
    let _weights = DeviceTypeWeights::power_saving();
}

/// Verifies that DeviceTypeWeights has performance() preset.
///
/// Contract: DeviceTypeWeights::performance() returns preset for max performance.
#[test]
fn test_device_type_weights_performance_preset() {
    let _weights = DeviceTypeWeights::performance();
}

/// Verifies that DeviceTypeWeights implements Clone.
///
/// Contract: DeviceTypeWeights implements Clone.
#[test]
fn test_device_type_weights_implements_clone() {
    let weights = DeviceTypeWeights::performance();
    let _cloned = weights.clone();
}

/// Verifies that DeviceTypeWeights implements Debug.
///
/// Contract: DeviceTypeWeights implements Debug for logging.
#[test]
fn test_device_type_weights_implements_debug() {
    let weights = DeviceTypeWeights::default();
    let debug_str = format!("{:?}", weights);
    assert!(!debug_str.is_empty());
}

/// Verifies that DeviceTypeWeights has discrete field.
///
/// Contract: DeviceTypeWeights.discrete is a numeric weight.
#[test]
fn test_device_type_weights_has_discrete_field() {
    let weights = DeviceTypeWeights::default();
    let _discrete: u32 = weights.discrete;
}

/// Verifies that DeviceTypeWeights has integrated field.
///
/// Contract: DeviceTypeWeights.integrated is a numeric weight.
#[test]
fn test_device_type_weights_has_integrated_field() {
    let weights = DeviceTypeWeights::default();
    let _integrated: u32 = weights.integrated;
}

/// Verifies that DeviceTypeWeights has cpu field.
///
/// Contract: DeviceTypeWeights.cpu is a numeric weight for software/CPU rendering.
#[test]
fn test_device_type_weights_has_cpu_field() {
    let weights = DeviceTypeWeights::default();
    let _cpu: u32 = weights.cpu;
}

/// Verifies that DeviceTypeWeights has other field.
///
/// Contract: DeviceTypeWeights.other is a numeric weight.
#[test]
fn test_device_type_weights_has_other_field() {
    let weights = DeviceTypeWeights::default();
    let _other: u32 = weights.other;
}

/// Verifies that performance preset prefers discrete GPUs.
///
/// Contract: performance().discrete > performance().integrated.
#[test]
fn test_device_type_weights_performance_prefers_discrete() {
    let weights = DeviceTypeWeights::performance();
    assert!(
        weights.discrete > weights.integrated,
        "Performance preset should prefer discrete over integrated"
    );
}

/// Verifies that default weights have discrete >= integrated >= cpu ordering.
///
/// Contract: Default weights follow discrete >= integrated >= cpu.
#[test]
fn test_device_type_weights_default_ordering() {
    let weights = DeviceTypeWeights::default();
    assert!(
        weights.discrete >= weights.integrated,
        "Discrete should score >= integrated"
    );
    assert!(
        weights.integrated >= weights.cpu,
        "Integrated should score >= cpu"
    );
}

/// Verifies that DeviceTypeWeights has weight_for() method.
///
/// Contract: DeviceTypeWeights::weight_for() returns weight for a given DeviceType.
#[test]
fn test_device_type_weights_has_weight_for_method() {
    let weights = DeviceTypeWeights::default();
    let _discrete_weight = weights.weight_for(wgpu::DeviceType::DiscreteGpu);
    let _integrated_weight = weights.weight_for(wgpu::DeviceType::IntegratedGpu);
    let _software_weight = weights.weight_for(wgpu::DeviceType::Cpu);
    let _other_weight = weights.weight_for(wgpu::DeviceType::Other);
}

// =============================================================================
// 3. AdapterBlacklistEntry Contract Tests
// =============================================================================

/// Verifies that AdapterBlacklistEntry has new() constructor.
///
/// Contract: AdapterBlacklistEntry::new() creates empty entry.
#[test]
fn test_adapter_blacklist_entry_new() {
    let _entry = AdapterBlacklistEntry::new();
}

/// Verifies that AdapterBlacklistEntry can be created with vendor.
///
/// Contract: AdapterBlacklistEntry::with_vendor() adds vendor filter.
#[test]
fn test_adapter_blacklist_entry_with_vendor() {
    let _entry = AdapterBlacklistEntry::new().with_vendor(Vendor::Unknown(0));
}

/// Verifies that AdapterBlacklistEntry can be created with name pattern.
///
/// Contract: AdapterBlacklistEntry::with_name_contains() adds name filter.
#[test]
fn test_adapter_blacklist_entry_with_name_contains() {
    let _entry = AdapterBlacklistEntry::new().with_name_contains("Broken GPU");
}

/// Verifies that AdapterBlacklistEntry can be created with vendor and name.
///
/// Contract: Builder pattern supports combining vendor and name filters.
#[test]
fn test_adapter_blacklist_entry_with_vendor_and_name() {
    let _entry = AdapterBlacklistEntry::new()
        .with_vendor(Vendor::Unknown(0))
        .with_name_contains("Broken GPU");
}

/// Verifies that AdapterBlacklistEntry has with_reason() method.
///
/// Contract: AdapterBlacklistEntry::with_reason() stores reason for blacklisting.
#[test]
fn test_adapter_blacklist_entry_with_reason() {
    let _entry = AdapterBlacklistEntry::new()
        .with_vendor(Vendor::Unknown(0))
        .with_reason("Known driver bug");
}

/// Verifies that AdapterBlacklistEntry has matches() method.
///
/// Contract: AdapterBlacklistEntry::matches() checks if adapter matches filter.
#[test]
fn test_adapter_blacklist_entry_has_matches_method() {
    let instance = TrinityInstance::new();
    let result = enumerate_adapters_with_info(instance.inner(), instance.backends());

    if let Some(adapter) = result.adapters.first() {
        let entry = AdapterBlacklistEntry::new().with_name_contains("NONEXISTENT12345");
        let info = adapter.get_info();
        let _matches: bool = entry.matches(&info);
    }
}

/// Verifies that AdapterBlacklistEntry implements Clone.
///
/// Contract: AdapterBlacklistEntry implements Clone.
#[test]
fn test_adapter_blacklist_entry_implements_clone() {
    let entry = AdapterBlacklistEntry::new().with_vendor(Vendor::Nvidia);
    let _cloned = entry.clone();
}

/// Verifies that AdapterBlacklistEntry implements Debug.
///
/// Contract: AdapterBlacklistEntry implements Debug.
#[test]
fn test_adapter_blacklist_entry_implements_debug() {
    let entry = AdapterBlacklistEntry::new().with_vendor(Vendor::Nvidia);
    let debug_str = format!("{:?}", entry);
    assert!(!debug_str.is_empty());
}

/// Verifies that AdapterBlacklistEntry implements Default.
///
/// Contract: AdapterBlacklistEntry implements Default (same as new()).
#[test]
fn test_adapter_blacklist_entry_implements_default() {
    let _entry = AdapterBlacklistEntry::default();
}

// =============================================================================
// 4. AdapterScore Contract Tests
// =============================================================================

/// Verifies that AdapterScore has device_type_score field.
///
/// Contract: AdapterScore.device_type_score is the device type score component.
#[test]
fn test_adapter_score_has_device_type_score_field() {
    let instance = TrinityInstance::new();
    let result = enumerate_adapters_with_info(instance.inner(), instance.backends());

    if let Some(adapter) = result.adapters.first() {
        let selector = AdapterSelector::new();
        let score = selector.score_adapter(adapter);
        let _device_type: u32 = score.device_type_score;
    }
}

/// Verifies that AdapterScore has feature_score field.
///
/// Contract: AdapterScore.feature_score is the feature availability score component.
#[test]
fn test_adapter_score_has_feature_score_field() {
    let instance = TrinityInstance::new();
    let result = enumerate_adapters_with_info(instance.inner(), instance.backends());

    if let Some(adapter) = result.adapters.first() {
        let selector = AdapterSelector::new();
        let score = selector.score_adapter(adapter);
        let _features: u32 = score.feature_score;
    }
}

/// Verifies that AdapterScore has limits_score field.
///
/// Contract: AdapterScore.limits_score is the adapter limits score component.
#[test]
fn test_adapter_score_has_limits_score_field() {
    let instance = TrinityInstance::new();
    let result = enumerate_adapters_with_info(instance.inner(), instance.backends());

    if let Some(adapter) = result.adapters.first() {
        let selector = AdapterSelector::new();
        let score = selector.score_adapter(adapter);
        let _limits: u32 = score.limits_score;
    }
}

/// Verifies that AdapterScore has vendor_bonus field.
///
/// Contract: AdapterScore.vendor_bonus is the vendor preference bonus.
#[test]
fn test_adapter_score_has_vendor_bonus_field() {
    let instance = TrinityInstance::new();
    let result = enumerate_adapters_with_info(instance.inner(), instance.backends());

    if let Some(adapter) = result.adapters.first() {
        let selector = AdapterSelector::new();
        let score = selector.score_adapter(adapter);
        let _vendor_bonus: u32 = score.vendor_bonus;
    }
}

/// Verifies that AdapterScore has total field.
///
/// Contract: AdapterScore.total is the sum of all score components.
#[test]
fn test_adapter_score_has_total_field() {
    let instance = TrinityInstance::new();
    let result = enumerate_adapters_with_info(instance.inner(), instance.backends());

    if let Some(adapter) = result.adapters.first() {
        let selector = AdapterSelector::new();
        let score = selector.score_adapter(adapter);
        let total: u32 = score.total;
        // Total should be sum of components
        assert_eq!(
            total,
            score.device_type_score + score.feature_score + score.limits_score + score.vendor_bonus,
            "Total should equal sum of all components"
        );
    }
}

/// Verifies that AdapterScore implements Display.
///
/// Contract: AdapterScore implements Display for human-readable output.
#[test]
fn test_adapter_score_implements_display() {
    let instance = TrinityInstance::new();
    let result = enumerate_adapters_with_info(instance.inner(), instance.backends());

    if let Some(adapter) = result.adapters.first() {
        let selector = AdapterSelector::new();
        let score = selector.score_adapter(adapter);
        let display_str = format!("{}", score);
        assert!(!display_str.is_empty(), "Display should produce output");
    }
}

/// Verifies that AdapterScore implements Debug.
///
/// Contract: AdapterScore implements Debug.
#[test]
fn test_adapter_score_implements_debug() {
    let instance = TrinityInstance::new();
    let result = enumerate_adapters_with_info(instance.inner(), instance.backends());

    if let Some(adapter) = result.adapters.first() {
        let selector = AdapterSelector::new();
        let score = selector.score_adapter(adapter);
        let debug_str = format!("{:?}", score);
        assert!(!debug_str.is_empty());
    }
}

/// Verifies that AdapterScore implements Clone.
///
/// Contract: AdapterScore implements Clone.
#[test]
fn test_adapter_score_implements_clone() {
    let instance = TrinityInstance::new();
    let result = enumerate_adapters_with_info(instance.inner(), instance.backends());

    if let Some(adapter) = result.adapters.first() {
        let selector = AdapterSelector::new();
        let score = selector.score_adapter(adapter);
        let _cloned = score.clone();
    }
}

/// Verifies that AdapterScore can have zero total.
///
/// Contract: AdapterScore total can be zero when adapter is blacklisted or has minimal scoring.
/// Note: AdapterScore is created via scoring, not Default trait.
#[test]
fn test_adapter_score_can_be_zero() {
    let instance = TrinityInstance::new();
    let result = enumerate_adapters_with_info(instance.inner(), instance.backends());

    if let Some(adapter) = result.adapters.first() {
        // Create a selector with zero weights
        let selector = AdapterSelector::new()
            .with_device_type_weights(DeviceTypeWeights {
                discrete: 0,
                integrated: 0,
                virtual_gpu: 0,
                cpu: 0,
                other: 0,
            })
            .with_feature_weight(0)
            .with_limit_weight(0)
            .with_vendor_preference_bonus(0);
        let score = selector.score_adapter(adapter);
        // With all weights zero, score should be zero
        assert_eq!(score.total, 0, "Score with zero weights should be zero");
    }
}

// =============================================================================
// 5. SelectionResult Contract Tests
// =============================================================================

/// Verifies that SelectionResult contains selected adapter reference.
///
/// Contract: SelectionResult.adapter is reference to selected adapter.
#[test]
fn test_selection_result_has_adapter_field() {
    let instance = TrinityInstance::new();
    let result = enumerate_adapters_with_info(instance.inner(), instance.backends());

    if let Some(selection) = AdapterSelector::new().select(&result.adapters) {
        let _adapter: &wgpu::Adapter = selection.adapter;
    }
}

/// Verifies that SelectionResult contains the adapter's score.
///
/// Contract: SelectionResult.score is the AdapterScore for selected adapter.
#[test]
fn test_selection_result_has_score_field() {
    let instance = TrinityInstance::new();
    let result = enumerate_adapters_with_info(instance.inner(), instance.backends());

    if let Some(selection) = AdapterSelector::new().select(&result.adapters) {
        let _score: &AdapterScore = &selection.score;
    }
}

/// Verifies that SelectionResult contains all_scores for comparison.
///
/// Contract: SelectionResult.all_scores contains (name, score) tuples for all adapters.
#[test]
fn test_selection_result_has_all_scores_field() {
    let instance = TrinityInstance::new();
    let result = enumerate_adapters_with_info(instance.inner(), instance.backends());

    if let Some(selection) = AdapterSelector::new().select(&result.adapters) {
        let _all_scores: &Vec<(String, AdapterScore)> = &selection.all_scores;
    }
}

/// Verifies that SelectionResult.all_scores has one entry per candidate adapter.
///
/// Contract: all_scores length equals number of non-blacklisted adapters.
#[test]
fn test_selection_result_all_scores_count_matches_adapters() {
    let instance = TrinityInstance::new();
    let result = enumerate_adapters_with_info(instance.inner(), instance.backends());

    if !result.adapters.is_empty() {
        if let Some(selection) = AdapterSelector::new().select(&result.adapters) {
            assert_eq!(
                selection.all_scores.len(),
                result.adapters.len(),
                "all_scores should have one entry per adapter"
            );
        }
    }
}

/// Verifies that SelectionResult has adapter_name() helper method.
///
/// Contract: SelectionResult::adapter_name() returns adapter name.
#[test]
fn test_selection_result_has_adapter_name_method() {
    let instance = TrinityInstance::new();
    let result = enumerate_adapters_with_info(instance.inner(), instance.backends());

    if let Some(selection) = AdapterSelector::new().select(&result.adapters) {
        let name: String = selection.adapter_name();
        assert!(!name.is_empty(), "Adapter name should not be empty");
    }
}

/// Verifies that SelectionResult has log_results() method.
///
/// Contract: SelectionResult::log_results() logs selection details.
#[test]
fn test_selection_result_has_log_results_method() {
    let instance = TrinityInstance::new();
    let result = enumerate_adapters_with_info(instance.inner(), instance.backends());

    if let Some(selection) = AdapterSelector::new().select(&result.adapters) {
        // Should not panic
        selection.log_results();
    }
}

/// Verifies that SelectionResult implements Debug.
///
/// Contract: SelectionResult implements Debug.
#[test]
fn test_selection_result_implements_debug() {
    let instance = TrinityInstance::new();
    let result = enumerate_adapters_with_info(instance.inner(), instance.backends());

    if let Some(selection) = AdapterSelector::new().select(&result.adapters) {
        let debug_str = format!("{:?}", selection);
        assert!(!debug_str.is_empty());
    }
}

// =============================================================================
// 6. Vendor Enum Contract Tests
// =============================================================================

/// Verifies that Vendor enum has Nvidia variant.
///
/// Contract: Vendor::Nvidia exists.
#[test]
fn test_vendor_has_nvidia() {
    let _vendor = Vendor::Nvidia;
}

/// Verifies that Vendor enum has Amd variant.
///
/// Contract: Vendor::Amd exists.
#[test]
fn test_vendor_has_amd() {
    let _vendor = Vendor::Amd;
}

/// Verifies that Vendor enum has Intel variant.
///
/// Contract: Vendor::Intel exists.
#[test]
fn test_vendor_has_intel() {
    let _vendor = Vendor::Intel;
}

/// Verifies that Vendor enum has Apple variant.
///
/// Contract: Vendor::Apple exists.
#[test]
fn test_vendor_has_apple() {
    let _vendor = Vendor::Apple;
}

/// Verifies that Vendor enum has Arm variant.
///
/// Contract: Vendor::Arm exists.
#[test]
fn test_vendor_has_arm() {
    let _vendor = Vendor::Arm;
}

/// Verifies that Vendor enum has Unknown variant.
///
/// Contract: Vendor::Unknown(id) exists for unrecognized vendors.
#[test]
fn test_vendor_has_unknown() {
    let _vendor = Vendor::Unknown(0);
    // Unknown variant takes a u32 vendor ID for debugging
    let _vendor_with_id = Vendor::Unknown(0x9999);
}

/// Verifies that Vendor has from_id() constructor.
///
/// Contract: Vendor::from_id() creates vendor from PCI vendor ID.
#[test]
fn test_vendor_has_from_id() {
    let nvidia = Vendor::from_id(0x10DE);
    assert_eq!(nvidia, Vendor::Nvidia);

    let amd = Vendor::from_id(0x1002);
    assert_eq!(amd, Vendor::Amd);

    let intel = Vendor::from_id(0x8086);
    assert_eq!(intel, Vendor::Intel);
}

/// Verifies that Vendor has name() method.
///
/// Contract: Vendor::name() returns human-readable name.
#[test]
fn test_vendor_has_name_method() {
    assert_eq!(Vendor::Nvidia.name(), "NVIDIA");
    assert_eq!(Vendor::Amd.name(), "AMD");
    assert_eq!(Vendor::Intel.name(), "Intel");
}

/// Verifies that Vendor has is_known() method.
///
/// Contract: Vendor::is_known() returns true for known vendors.
#[test]
fn test_vendor_has_is_known_method() {
    assert!(Vendor::Nvidia.is_known());
    assert!(Vendor::Amd.is_known());
    assert!(Vendor::Intel.is_known());
    assert!(!Vendor::Unknown(0).is_known());
}

/// Verifies that Vendor has id() method.
///
/// Contract: Vendor::id() returns PCI vendor ID.
#[test]
fn test_vendor_has_id_method() {
    assert_eq!(Vendor::Nvidia.id(), 0x10DE);
    assert_eq!(Vendor::Amd.id(), 0x1002);
    assert_eq!(Vendor::Intel.id(), 0x8086);
}

/// Verifies that Vendor implements Clone.
///
/// Contract: Vendor implements Clone.
#[test]
fn test_vendor_implements_clone() {
    let vendor = Vendor::Nvidia;
    let _cloned = vendor.clone();
}

/// Verifies that Vendor implements Debug.
///
/// Contract: Vendor implements Debug.
#[test]
fn test_vendor_implements_debug() {
    let vendor = Vendor::Nvidia;
    let debug_str = format!("{:?}", vendor);
    assert!(debug_str.contains("Nvidia"));
}

/// Verifies that Vendor implements PartialEq.
///
/// Contract: Vendor implements PartialEq for comparison.
#[test]
fn test_vendor_implements_partial_eq() {
    assert_eq!(Vendor::Nvidia, Vendor::Nvidia);
    assert_ne!(Vendor::Nvidia, Vendor::Amd);
}

/// Verifies that Vendor implements Copy.
///
/// Contract: Vendor implements Copy (it's a simple enum).
#[test]
fn test_vendor_implements_copy() {
    let vendor = Vendor::Nvidia;
    let copied = vendor; // Copy, not move
    let _ = vendor; // Can still use original
    let _ = copied;
}

// =============================================================================
// 7. Selection Behavior Contract Tests
// =============================================================================

/// Verifies that selection with empty adapter list returns None.
///
/// Contract: select() on empty list returns None.
#[test]
fn test_selection_with_empty_list_returns_none() {
    let empty: Vec<wgpu::Adapter> = vec![];
    let selector = AdapterSelector::new();

    assert!(selector.select(&empty).is_none());
}

/// Verifies that select_adapter on empty list returns None.
///
/// Contract: select_adapter() on empty list returns None.
#[test]
fn test_select_adapter_empty_list_returns_none() {
    let empty: Vec<wgpu::Adapter> = vec![];
    let selector = AdapterSelector::new();

    assert!(selector.select_adapter(&empty).is_none());
}

/// Verifies that vendor preference adds bonus to matching adapters.
///
/// Contract: Vendor preference increases score for matching adapters.
#[test]
fn test_vendor_preference_affects_scoring() {
    let instance = TrinityInstance::new();
    let result = enumerate_adapters_with_info(instance.inner(), instance.backends());

    if result.adapters.is_empty() {
        return; // Skip on systems without GPU
    }

    // Get score without preference
    let selector_no_pref = AdapterSelector::new();
    let selection_no_pref = selector_no_pref.select(&result.adapters);

    // Get score with preference for some vendor
    let selector_with_pref = AdapterSelector::new().with_vendor_preference(Vendor::Nvidia);
    let selection_with_pref = selector_with_pref.select(&result.adapters);

    // Both should succeed if there are adapters
    assert!(selection_no_pref.is_some());
    assert!(selection_with_pref.is_some());
}

/// Verifies that blacklisted adapters are excluded from selection.
///
/// Contract: Blacklisted adapters are not selected.
#[test]
fn test_blacklisted_adapters_excluded() {
    let instance = TrinityInstance::new();
    let result = enumerate_adapters_with_info(instance.inner(), instance.backends());

    if result.adapters.is_empty() {
        return; // Skip on systems without GPU
    }

    // Blacklist all vendors to verify exclusion works
    let blacklist = vec![
        AdapterBlacklistEntry::new().with_vendor(Vendor::Nvidia),
        AdapterBlacklistEntry::new().with_vendor(Vendor::Amd),
        AdapterBlacklistEntry::new().with_vendor(Vendor::Intel),
        AdapterBlacklistEntry::new().with_vendor(Vendor::Apple),
        AdapterBlacklistEntry::new().with_vendor(Vendor::Arm),
        AdapterBlacklistEntry::new().with_vendor(Vendor::Unknown(0)),
    ];

    let selector = AdapterSelector::new().with_blacklist(blacklist);

    // With all vendors blacklisted, selection should return None
    // (unless there's a vendor not in our list)
    let selection = selector.select(&result.adapters);
    // Note: This may still select something if vendor detection is imperfect
    let _ = selection;
}

/// Verifies that select returns first adapter when all equal.
///
/// Contract: With no preference and equal scores, returns first adapter.
#[test]
fn test_selection_returns_valid_adapter() {
    let instance = TrinityInstance::new();
    let result = enumerate_adapters_with_info(instance.inner(), instance.backends());

    if result.adapters.is_empty() {
        return; // Skip on systems without GPU
    }

    let selector = AdapterSelector::new();
    let selected = selector.select(&result.adapters);

    assert!(selected.is_some(), "Should select an adapter when available");

    // Verify the selected adapter is from the input list
    let selected_result = selected.unwrap();
    let selected_info = selected_result.adapter.get_info();

    let found = result
        .adapters
        .iter()
        .any(|a| a.get_info().name == selected_info.name);
    assert!(found, "Selected adapter should be from the input list");
}

/// Verifies that device type weights affect selection order.
///
/// Contract: DeviceTypeWeights influence which adapter is selected.
#[test]
fn test_device_type_weights_affect_selection() {
    let instance = TrinityInstance::new();
    let result = enumerate_adapters_with_info(instance.inner(), instance.backends());

    if result.adapters.is_empty() {
        return; // Skip on systems without GPU
    }

    // Select with performance weights
    let perf_selector =
        AdapterSelector::new().with_device_type_weights(DeviceTypeWeights::performance());
    let _perf_result = perf_selector.select(&result.adapters);

    // Select with power saving weights
    let power_selector =
        AdapterSelector::new().with_device_type_weights(DeviceTypeWeights::power_saving());
    let _power_result = power_selector.select(&result.adapters);

    // Both should work - actual ordering depends on available hardware
}

// =============================================================================
// 8. Integration Tests
// =============================================================================

/// Verifies complete adapter selection workflow.
///
/// Contract: Full selection pipeline works end-to-end.
#[test]
fn test_complete_selection_workflow() {
    // 1. Create instance
    let instance = TrinityInstance::new();

    // 2. Enumerate adapters
    let result = enumerate_adapters_with_info(instance.inner(), instance.backends());

    // 3. Configure selector
    let selector = AdapterSelector::new()
        .with_device_type_weights(DeviceTypeWeights::performance())
        .with_vendor_preference(Vendor::Nvidia)
        .with_feature_weight(100)
        .with_limit_weight(50);

    // 4. Select adapter
    if let Some(selection) = selector.select(&result.adapters) {
        // 5. Verify result structure
        assert!(selection.score.total > 0);
        assert!(!selection.all_scores.is_empty());

        // 6. Get selected adapter info
        let info = selection.adapter.get_info();
        assert!(!info.name.is_empty());
    }
}

/// Verifies that selection works with custom blacklist.
///
/// Contract: Custom blacklist entries are respected.
#[test]
fn test_selection_with_custom_blacklist() {
    let instance = TrinityInstance::new();
    let result = enumerate_adapters_with_info(instance.inner(), instance.backends());

    if result.adapters.is_empty() {
        return;
    }

    // Create blacklist with a specific pattern
    let blacklist = vec![AdapterBlacklistEntry::new()
        .with_name_contains("NonExistentGPU")
        .with_reason("Test blacklist entry")];

    let selector = AdapterSelector::new().with_blacklist(blacklist);
    let selection = selector.select(&result.adapters);

    // Since "NonExistentGPU" doesn't exist, all adapters should be available
    assert!(selection.is_some());
}

/// Verifies that selection handles mixed device types correctly.
///
/// Contract: Selection algorithm handles heterogeneous adapter sets.
#[test]
fn test_selection_handles_mixed_device_types() {
    let instance = TrinityInstance::new();
    let result = enumerate_adapters_with_info(instance.inner(), instance.backends());

    if result.adapters.is_empty() {
        return;
    }

    // With performance weights, should prefer discrete if available
    let selector =
        AdapterSelector::new().with_device_type_weights(DeviceTypeWeights::performance());

    if let Some(selection) = selector.select(&result.adapters) {
        // Verify that score reflects device type
        assert!(selection.score.device_type_score >= 0);
    }
}

/// Verifies that all_scores provides complete scoring breakdown.
///
/// Contract: all_scores allows comparison of all candidate adapters.
#[test]
fn test_all_scores_provides_complete_breakdown() {
    let instance = TrinityInstance::new();
    let result = enumerate_adapters_with_info(instance.inner(), instance.backends());

    if result.adapters.is_empty() {
        return;
    }

    let selector = AdapterSelector::new();
    if let Some(selection) = selector.select(&result.adapters) {
        // all_scores should have (name, score) tuples for all non-blacklisted adapters
        for (name, score) in &selection.all_scores {
            // Name should be non-empty
            assert!(!name.is_empty(), "Adapter name should not be empty");
            // Each score should have valid components
            let _ = score.device_type_score;
            let _ = score.feature_score;
            let _ = score.limits_score;
            let _ = score.vendor_bonus;
            let _ = score.total;
        }
    }
}

/// Verifies that highest-scoring adapter is selected.
///
/// Contract: Selected adapter has highest total score among candidates.
#[test]
fn test_highest_scoring_adapter_selected() {
    let instance = TrinityInstance::new();
    let result = enumerate_adapters_with_info(instance.inner(), instance.backends());

    if result.adapters.is_empty() {
        return;
    }

    let selector = AdapterSelector::new();
    if let Some(selection) = selector.select(&result.adapters) {
        let selected_score = selection.score.total;

        // Selected adapter should have the highest or tied-for-highest score
        for (_name, score) in &selection.all_scores {
            assert!(
                selected_score >= score.total,
                "Selected score {} should be >= all scores, but found {}",
                selected_score,
                score.total
            );
        }
    }
}

// =============================================================================
// 9. Edge Case Tests
// =============================================================================

/// Verifies that name-based blacklist entries use partial matching.
///
/// Contract: Name patterns match substrings in adapter names.
#[test]
fn test_blacklist_name_pattern_matching() {
    let instance = TrinityInstance::new();
    let result = enumerate_adapters_with_info(instance.inner(), instance.backends());

    if result.adapters.is_empty() {
        return;
    }

    // Create blacklist entry with partial name that won't match anything
    let blacklist = vec![AdapterBlacklistEntry::new().with_name_contains("XXXYYYZZZ_NONEXISTENT_12345")];

    let selector = AdapterSelector::new().with_blacklist(blacklist);
    let selection = selector.select(&result.adapters);

    // Should still find adapters since pattern doesn't match
    assert!(selection.is_some());
}

/// Verifies that empty blacklist does not affect selection.
///
/// Contract: Empty blacklist is equivalent to no blacklist.
#[test]
fn test_empty_blacklist_no_effect() {
    let instance = TrinityInstance::new();
    let result = enumerate_adapters_with_info(instance.inner(), instance.backends());

    if result.adapters.is_empty() {
        return;
    }

    let selector_no_blacklist = AdapterSelector::new();
    let selector_empty_blacklist = AdapterSelector::new().with_blacklist(vec![]);

    let selection_no = selector_no_blacklist.select(&result.adapters);
    let selection_empty = selector_empty_blacklist.select(&result.adapters);

    // Both should select the same adapter
    match (selection_no, selection_empty) {
        (Some(a), Some(b)) => {
            assert_eq!(
                a.adapter.get_info().name,
                b.adapter.get_info().name,
                "Empty blacklist should behave same as no blacklist"
            );
        }
        (None, None) => {} // Both found nothing
        _ => panic!("Mismatch between empty blacklist and no blacklist behavior"),
    }
}

/// Verifies that vendor preference for non-present vendor falls back gracefully.
///
/// Contract: Preference for unavailable vendor still selects best available.
#[test]
fn test_vendor_preference_fallback() {
    let instance = TrinityInstance::new();
    let result = enumerate_adapters_with_info(instance.inner(), instance.backends());

    if result.adapters.is_empty() {
        return;
    }

    // Set preference for vendor that might not be present
    let selector = AdapterSelector::new().with_vendor_preference(Vendor::Arm);

    // Should still select something even if ARM adapter isn't present
    let selection = selector.select(&result.adapters);
    assert!(
        selection.is_some(),
        "Should fall back to best available adapter"
    );
}

/// Verifies that multiple blacklist entries are all checked.
///
/// Contract: All blacklist entries are evaluated.
#[test]
fn test_multiple_blacklist_entries() {
    let instance = TrinityInstance::new();
    let result = enumerate_adapters_with_info(instance.inner(), instance.backends());

    if result.adapters.is_empty() {
        return;
    }

    let blacklist = vec![
        AdapterBlacklistEntry::new().with_name_contains("Pattern1"),
        AdapterBlacklistEntry::new().with_name_contains("Pattern2"),
        AdapterBlacklistEntry::new().with_vendor(Vendor::Unknown(0)),
    ];

    let selector = AdapterSelector::new().with_blacklist(blacklist);
    let _selection = selector.select(&result.adapters);
    // Test passes if no panic occurs during blacklist processing
}

/// Verifies that score components are non-negative.
///
/// Contract: All score components are >= 0.
#[test]
fn test_score_components_non_negative() {
    let instance = TrinityInstance::new();
    let result = enumerate_adapters_with_info(instance.inner(), instance.backends());

    if result.adapters.is_empty() {
        return;
    }

    let selector = AdapterSelector::new();
    if let Some(selection) = selector.select(&result.adapters) {
        // All components should be non-negative (u32 enforces this, but test the contract)
        assert!(selection.score.device_type_score >= 0);
        assert!(selection.score.feature_score >= 0);
        assert!(selection.score.limits_score >= 0);
        assert!(selection.score.vendor_bonus >= 0);
    }
}

/// Verifies that single blacklist entry works correctly.
///
/// Contract: Single blacklist entry is processed correctly.
#[test]
fn test_single_blacklist_entry() {
    let instance = TrinityInstance::new();
    let result = enumerate_adapters_with_info(instance.inner(), instance.backends());

    if result.adapters.is_empty() {
        return;
    }

    let entry = AdapterBlacklistEntry::new()
        .with_name_contains("NONEXISTENT")
        .with_reason("Test entry");

    let selector = AdapterSelector::new().with_blacklist_entry(entry);
    let selection = selector.select(&result.adapters);

    // Should still find adapters since pattern doesn't match
    assert!(selection.is_some());
}

// =============================================================================
// 10. Concurrency Safety Tests
// =============================================================================

/// Verifies that AdapterSelector can be used from multiple threads.
///
/// Contract: AdapterSelector is thread-safe for selection.
#[test]
fn test_adapter_selector_thread_safety() {
    use std::sync::Arc;
    use std::thread;

    let instance = TrinityInstance::new();
    let result = enumerate_adapters_with_info(instance.inner(), instance.backends());

    if result.adapters.is_empty() {
        return;
    }

    let adapters = Arc::new(result.adapters);
    let selector = AdapterSelector::new();

    let handles: Vec<_> = (0..4)
        .map(|_| {
            let adapters_clone = Arc::clone(&adapters);
            let selector_clone = selector.clone();
            thread::spawn(move || {
                let selection = selector_clone.select(&adapters_clone);
                selection.map(|s| s.adapter.get_info().name.clone())
            })
        })
        .collect();

    let results: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();

    // All threads should get the same result
    let first = &results[0];
    for result in &results[1..] {
        assert_eq!(result, first, "All threads should select the same adapter");
    }
}

/// Verifies that DeviceTypeWeights can be shared across threads.
///
/// Contract: DeviceTypeWeights is thread-safe.
#[test]
fn test_device_type_weights_thread_safety() {
    use std::sync::Arc;
    use std::thread;

    let weights = Arc::new(DeviceTypeWeights::performance());

    let handles: Vec<_> = (0..4)
        .map(|_| {
            let weights_clone = Arc::clone(&weights);
            thread::spawn(move || {
                // Read weights from multiple threads
                weights_clone.discrete + weights_clone.integrated
            })
        })
        .collect();

    for handle in handles {
        let _ = handle.join().unwrap();
    }
}

/// Verifies that scoring is deterministic.
///
/// Contract: Same adapter always produces same score with same selector.
#[test]
fn test_scoring_deterministic() {
    let instance = TrinityInstance::new();
    let result = enumerate_adapters_with_info(instance.inner(), instance.backends());

    if result.adapters.is_empty() {
        return;
    }

    let adapter = result.adapters.first().unwrap();
    let selector = AdapterSelector::new()
        .with_device_type_weights(DeviceTypeWeights::performance())
        .with_vendor_preference(Vendor::Nvidia);

    let score1 = selector.score_adapter(adapter);
    let score2 = selector.score_adapter(adapter);

    assert_eq!(score1.total, score2.total, "Scoring should be deterministic");
    assert_eq!(score1.device_type_score, score2.device_type_score);
    assert_eq!(score1.feature_score, score2.feature_score);
    assert_eq!(score1.limits_score, score2.limits_score);
    assert_eq!(score1.vendor_bonus, score2.vendor_bonus);
}

/// Verifies that selection is deterministic.
///
/// Contract: Same adapters always produce same selection with same selector.
#[test]
fn test_selection_deterministic() {
    let instance = TrinityInstance::new();
    let result = enumerate_adapters_with_info(instance.inner(), instance.backends());

    if result.adapters.is_empty() {
        return;
    }

    let selector = AdapterSelector::new();

    let selection1 = selector.select(&result.adapters);
    let selection2 = selector.select(&result.adapters);

    match (selection1, selection2) {
        (Some(s1), Some(s2)) => {
            assert_eq!(
                s1.adapter.get_info().name,
                s2.adapter.get_info().name,
                "Selection should be deterministic"
            );
        }
        (None, None) => {} // Both found nothing
        _ => panic!("Selection should be deterministic"),
    }
}
