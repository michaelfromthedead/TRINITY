// Blackbox contract tests for T-WGPU-P7.1.4 Present Mode Selection API.
//
// CLEANROOM: No src/ access beyond the public API exported by the crate.
// Tests use only `renderer_backend::presentation::*` -- no internal fields,
// no private methods, no implementation details.
//
// Contract:
//   PresentModePreference enum specifies the desired presentation behavior:
//     - LowLatency:   Lowest input latency (Immediate > Mailbox > FifoRelaxed > Fifo)
//     - Vsync:        Smooth vsync without tearing (Mailbox > FifoRelaxed > Fifo)
//     - PowerSaving:  Minimize power consumption (Fifo preferred)
//     - Adaptive:     Adaptive vsync with frame drops (FifoRelaxed > Mailbox > Fifo)
//     - Specific(m):  Request specific mode, fallback to Vsync if unavailable
//
//   PresentModeInfo provides metadata about a present mode:
//     - mode:             The wgpu::PresentMode
//     - name:             Human-readable name
//     - description:      Short description
//     - prevents_tearing: Whether the mode eliminates tearing
//     - latency_rank:     1 = lowest latency, 4 = highest
//     - power_efficient:  Whether GPU can idle between frames
//     - is_competitive_gaming_mode(): latency_rank <= 2
//     - is_battery_friendly():        power_efficient == true
//
//   SurfaceCapabilities contains:
//     - formats:        Vec<TextureFormat>
//     - present_modes:  Vec<PresentMode>
//     - alpha_modes:    Vec<CompositeAlphaMode>
//     - usages:         TextureUsages
//
//   SurfaceCapabilities methods:
//     - low_latency_present_mode() -> PresentMode
//     - preferred_present_mode()   -> PresentMode
//     - select_present_mode(pref)  -> PresentMode
//     - supports_immediate()       -> bool
//     - supports_mailbox()         -> bool
//     - supports_fifo_relaxed()    -> bool
//     - supports_present_mode(m)   -> bool
//     - describe_present_mode(m)   -> PresentModeInfo
//
//   SurfaceConfiguration:
//     - with_present_mode_preference(caps, pref) -> Self
//
// Scenarios (65 tests, 130+ assertions):
//   1-10:   PresentModePreference enum traits and behavior
//   11-25:  PresentModeInfo metadata and classification
//   26-45:  SurfaceCapabilities present mode selection
//   46-55:  SurfaceConfiguration with preferences
//   56-65:  Edge cases and mobile/limited surface scenarios

use renderer_backend::presentation::{
    PresentModeInfo, PresentModePreference, SurfaceCapabilities, SurfaceConfiguration,
};
use wgpu::{CompositeAlphaMode, PresentMode, TextureFormat, TextureUsages};

// ============================================================================
// Test Fixtures / Helpers
// ============================================================================

/// Create capabilities with all present modes available (desktop GPU scenario).
fn caps_all_modes() -> SurfaceCapabilities {
    SurfaceCapabilities {
        formats: vec![TextureFormat::Bgra8Unorm],
        present_modes: vec![
            PresentMode::Fifo,
            PresentMode::FifoRelaxed,
            PresentMode::Mailbox,
            PresentMode::Immediate,
        ],
        alpha_modes: vec![CompositeAlphaMode::Auto],
        usages: TextureUsages::RENDER_ATTACHMENT,
    }
}

/// Create capabilities with only Fifo (mobile/limited scenario).
fn caps_fifo_only() -> SurfaceCapabilities {
    SurfaceCapabilities {
        formats: vec![TextureFormat::Bgra8Unorm],
        present_modes: vec![PresentMode::Fifo],
        alpha_modes: vec![CompositeAlphaMode::Auto],
        usages: TextureUsages::RENDER_ATTACHMENT,
    }
}

/// Create capabilities without Immediate mode (common on some platforms).
fn caps_no_immediate() -> SurfaceCapabilities {
    SurfaceCapabilities {
        formats: vec![TextureFormat::Bgra8Unorm],
        present_modes: vec![
            PresentMode::Fifo,
            PresentMode::FifoRelaxed,
            PresentMode::Mailbox,
        ],
        alpha_modes: vec![CompositeAlphaMode::Auto],
        usages: TextureUsages::RENDER_ATTACHMENT,
    }
}

/// Create capabilities without Mailbox mode.
fn caps_no_mailbox() -> SurfaceCapabilities {
    SurfaceCapabilities {
        formats: vec![TextureFormat::Bgra8Unorm],
        present_modes: vec![
            PresentMode::Fifo,
            PresentMode::FifoRelaxed,
            PresentMode::Immediate,
        ],
        alpha_modes: vec![CompositeAlphaMode::Auto],
        usages: TextureUsages::RENDER_ATTACHMENT,
    }
}

/// Create capabilities with Fifo and FifoRelaxed only (VRR display).
fn caps_vrr_display() -> SurfaceCapabilities {
    SurfaceCapabilities {
        formats: vec![TextureFormat::Bgra8Unorm],
        present_modes: vec![PresentMode::Fifo, PresentMode::FifoRelaxed],
        alpha_modes: vec![CompositeAlphaMode::Auto],
        usages: TextureUsages::RENDER_ATTACHMENT,
    }
}

/// Create capabilities with Fifo and Mailbox only.
fn caps_fifo_mailbox() -> SurfaceCapabilities {
    SurfaceCapabilities {
        formats: vec![TextureFormat::Bgra8Unorm],
        present_modes: vec![PresentMode::Fifo, PresentMode::Mailbox],
        alpha_modes: vec![CompositeAlphaMode::Auto],
        usages: TextureUsages::RENDER_ATTACHMENT,
    }
}

/// Create capabilities with Fifo and Immediate only (no triple buffering).
fn caps_fifo_immediate() -> SurfaceCapabilities {
    SurfaceCapabilities {
        formats: vec![TextureFormat::Bgra8Unorm],
        present_modes: vec![PresentMode::Fifo, PresentMode::Immediate],
        alpha_modes: vec![CompositeAlphaMode::Auto],
        usages: TextureUsages::RENDER_ATTACHMENT,
    }
}

// ============================================================================
// 1-10: PresentModePreference Enum Traits and Behavior
// ============================================================================

#[test]
fn test_01_present_mode_preference_default_is_vsync() {
    // Default preference should be Vsync for general-purpose use.
    let pref = PresentModePreference::default();
    assert_eq!(pref, PresentModePreference::Vsync);
}

#[test]
fn test_02_present_mode_preference_derives_debug() {
    // Debug should be implemented for diagnostic output.
    let pref = PresentModePreference::LowLatency;
    let debug_str = format!("{:?}", pref);
    assert!(debug_str.contains("LowLatency"));
}

#[test]
fn test_03_present_mode_preference_derives_clone() {
    // Clone should work for all variants.
    let pref1 = PresentModePreference::PowerSaving;
    let pref2 = pref1.clone();
    assert_eq!(pref1, pref2);
}

#[test]
fn test_04_present_mode_preference_derives_copy() {
    // Copy should work (no ownership transfer).
    let pref1 = PresentModePreference::Adaptive;
    let pref2 = pref1; // Copy, not move
    assert_eq!(pref1, pref2);
}

#[test]
fn test_05_present_mode_preference_derives_eq() {
    // Eq/PartialEq should be derived.
    assert_eq!(PresentModePreference::Vsync, PresentModePreference::Vsync);
    assert_ne!(PresentModePreference::Vsync, PresentModePreference::LowLatency);
}

#[test]
fn test_06_present_mode_preference_derives_hash() {
    // Hash should be derived for use in HashMaps.
    use std::collections::HashSet;
    let mut set = HashSet::new();
    set.insert(PresentModePreference::LowLatency);
    set.insert(PresentModePreference::Vsync);
    assert!(set.contains(&PresentModePreference::LowLatency));
    assert!(set.contains(&PresentModePreference::Vsync));
    assert!(!set.contains(&PresentModePreference::PowerSaving));
}

#[test]
fn test_07_present_mode_preference_specific_equality() {
    // Specific variants should compare the inner mode.
    let a = PresentModePreference::Specific(PresentMode::Fifo);
    let b = PresentModePreference::Specific(PresentMode::Fifo);
    let c = PresentModePreference::Specific(PresentMode::Mailbox);
    assert_eq!(a, b);
    assert_ne!(a, c);
}

#[test]
fn test_08_present_mode_preference_description_non_empty() {
    // All variants should have non-empty descriptions.
    assert!(!PresentModePreference::LowLatency.description().is_empty());
    assert!(!PresentModePreference::Vsync.description().is_empty());
    assert!(!PresentModePreference::PowerSaving.description().is_empty());
    assert!(!PresentModePreference::Adaptive.description().is_empty());
    assert!(!PresentModePreference::Specific(PresentMode::Fifo)
        .description()
        .is_empty());
}

#[test]
fn test_09_present_mode_preference_display_format() {
    // Display should produce human-readable strings.
    assert_eq!(format!("{}", PresentModePreference::LowLatency), "Low Latency");
    assert_eq!(format!("{}", PresentModePreference::Vsync), "Vsync");
    assert_eq!(format!("{}", PresentModePreference::PowerSaving), "Power Saving");
    assert_eq!(format!("{}", PresentModePreference::Adaptive), "Adaptive");
    // Specific should contain "Specific"
    let specific_display = format!("{}", PresentModePreference::Specific(PresentMode::Fifo));
    assert!(specific_display.contains("Specific"));
}

#[test]
fn test_10_present_mode_preference_all_variants_distinct() {
    // All main variants should be distinct.
    let variants = [
        PresentModePreference::LowLatency,
        PresentModePreference::Vsync,
        PresentModePreference::PowerSaving,
        PresentModePreference::Adaptive,
    ];
    for i in 0..variants.len() {
        for j in (i + 1)..variants.len() {
            assert_ne!(variants[i], variants[j], "Variants {} and {} should differ", i, j);
        }
    }
}

// ============================================================================
// 11-25: PresentModeInfo Metadata and Classification
// ============================================================================

#[test]
fn test_11_present_mode_info_immediate_properties() {
    // Immediate mode: lowest latency, may tear, not power efficient.
    let info = PresentModeInfo::from_mode(PresentMode::Immediate);
    assert_eq!(info.mode, PresentMode::Immediate);
    assert!(!info.prevents_tearing);
    assert_eq!(info.latency_rank, 1);
    assert!(!info.power_efficient);
}

#[test]
fn test_12_present_mode_info_mailbox_properties() {
    // Mailbox: low latency, no tearing, not power efficient.
    let info = PresentModeInfo::from_mode(PresentMode::Mailbox);
    assert_eq!(info.mode, PresentMode::Mailbox);
    assert!(info.prevents_tearing);
    assert_eq!(info.latency_rank, 2);
    assert!(!info.power_efficient);
}

#[test]
fn test_13_present_mode_info_fifo_properties() {
    // Fifo: highest latency, no tearing, power efficient.
    let info = PresentModeInfo::from_mode(PresentMode::Fifo);
    assert_eq!(info.mode, PresentMode::Fifo);
    assert!(info.prevents_tearing);
    assert_eq!(info.latency_rank, 4);
    assert!(info.power_efficient);
}

#[test]
fn test_14_present_mode_info_fifo_relaxed_properties() {
    // FifoRelaxed: medium latency, no tearing, power efficient.
    let info = PresentModeInfo::from_mode(PresentMode::FifoRelaxed);
    assert_eq!(info.mode, PresentMode::FifoRelaxed);
    assert!(info.prevents_tearing);
    assert_eq!(info.latency_rank, 3);
    assert!(info.power_efficient);
}

#[test]
fn test_15_present_mode_info_competitive_gaming_immediate() {
    // Immediate should be suitable for competitive gaming.
    let info = PresentModeInfo::from_mode(PresentMode::Immediate);
    assert!(info.is_competitive_gaming_mode());
}

#[test]
fn test_16_present_mode_info_competitive_gaming_mailbox() {
    // Mailbox should be suitable for competitive gaming.
    let info = PresentModeInfo::from_mode(PresentMode::Mailbox);
    assert!(info.is_competitive_gaming_mode());
}

#[test]
fn test_17_present_mode_info_not_competitive_gaming_fifo() {
    // Fifo is NOT suitable for competitive gaming (too much latency).
    let info = PresentModeInfo::from_mode(PresentMode::Fifo);
    assert!(!info.is_competitive_gaming_mode());
}

#[test]
fn test_18_present_mode_info_not_competitive_gaming_fifo_relaxed() {
    // FifoRelaxed is NOT suitable for competitive gaming.
    let info = PresentModeInfo::from_mode(PresentMode::FifoRelaxed);
    assert!(!info.is_competitive_gaming_mode());
}

#[test]
fn test_19_present_mode_info_battery_friendly_fifo() {
    // Fifo should be battery-friendly.
    let info = PresentModeInfo::from_mode(PresentMode::Fifo);
    assert!(info.is_battery_friendly());
}

#[test]
fn test_20_present_mode_info_battery_friendly_fifo_relaxed() {
    // FifoRelaxed should be battery-friendly.
    let info = PresentModeInfo::from_mode(PresentMode::FifoRelaxed);
    assert!(info.is_battery_friendly());
}

#[test]
fn test_21_present_mode_info_not_battery_friendly_immediate() {
    // Immediate is NOT battery-friendly.
    let info = PresentModeInfo::from_mode(PresentMode::Immediate);
    assert!(!info.is_battery_friendly());
}

#[test]
fn test_22_present_mode_info_not_battery_friendly_mailbox() {
    // Mailbox is NOT battery-friendly (GPU always rendering).
    let info = PresentModeInfo::from_mode(PresentMode::Mailbox);
    assert!(!info.is_battery_friendly());
}

#[test]
fn test_23_present_mode_info_display_contains_name() {
    // Display should include the mode name.
    let info = PresentModeInfo::from_mode(PresentMode::Mailbox);
    let display = format!("{}", info);
    assert!(display.contains("Mailbox"));
}

#[test]
fn test_24_present_mode_info_latency_rank_ordering() {
    // Latency ranks should follow: Immediate(1) < Mailbox(2) < FifoRelaxed(3) < Fifo(4)
    let immediate = PresentModeInfo::from_mode(PresentMode::Immediate);
    let mailbox = PresentModeInfo::from_mode(PresentMode::Mailbox);
    let fifo_relaxed = PresentModeInfo::from_mode(PresentMode::FifoRelaxed);
    let fifo = PresentModeInfo::from_mode(PresentMode::Fifo);

    assert!(immediate.latency_rank < mailbox.latency_rank);
    assert!(mailbox.latency_rank < fifo_relaxed.latency_rank);
    assert!(fifo_relaxed.latency_rank < fifo.latency_rank);
}

#[test]
fn test_25_present_mode_info_describes_present_mode_static() {
    // SurfaceCapabilities::describe_present_mode should return correct info.
    let info = SurfaceCapabilities::describe_present_mode(PresentMode::Immediate);
    assert_eq!(info.mode, PresentMode::Immediate);
    assert_eq!(info.latency_rank, 1);
}

// ============================================================================
// 26-45: SurfaceCapabilities Present Mode Selection
// ============================================================================

#[test]
fn test_26_low_latency_prefers_immediate() {
    // With all modes available, low latency should select Immediate.
    let caps = caps_all_modes();
    assert_eq!(caps.low_latency_present_mode(), PresentMode::Immediate);
}

#[test]
fn test_27_low_latency_fallback_to_mailbox() {
    // Without Immediate, low latency should fall back to Mailbox.
    let caps = caps_no_immediate();
    assert_eq!(caps.low_latency_present_mode(), PresentMode::Mailbox);
}

#[test]
fn test_28_low_latency_fallback_to_fifo_relaxed() {
    // Without Immediate or Mailbox, fall back to FifoRelaxed.
    let caps = caps_vrr_display();
    assert_eq!(caps.low_latency_present_mode(), PresentMode::FifoRelaxed);
}

#[test]
fn test_29_low_latency_ultimate_fallback_to_fifo() {
    // With only Fifo available, low latency must use Fifo.
    let caps = caps_fifo_only();
    assert_eq!(caps.low_latency_present_mode(), PresentMode::Fifo);
}

#[test]
fn test_30_preferred_present_mode_prefers_mailbox() {
    // Vsync preference should select Mailbox when available.
    let caps = caps_all_modes();
    assert_eq!(caps.preferred_present_mode(), PresentMode::Mailbox);
}

#[test]
fn test_31_preferred_present_mode_fallback_to_fifo_relaxed() {
    // Without Mailbox, preferred mode should select FifoRelaxed.
    let caps = caps_vrr_display();
    assert_eq!(caps.preferred_present_mode(), PresentMode::FifoRelaxed);
}

#[test]
fn test_32_preferred_present_mode_fallback_to_fifo() {
    // With only Fifo available, preferred mode is Fifo.
    let caps = caps_fifo_only();
    assert_eq!(caps.preferred_present_mode(), PresentMode::Fifo);
}

#[test]
fn test_33_select_low_latency_preference() {
    // Select with LowLatency preference should get Immediate.
    let caps = caps_all_modes();
    assert_eq!(
        caps.select_present_mode(PresentModePreference::LowLatency),
        PresentMode::Immediate
    );
}

#[test]
fn test_34_select_vsync_preference() {
    // Select with Vsync preference should get Mailbox.
    let caps = caps_all_modes();
    assert_eq!(
        caps.select_present_mode(PresentModePreference::Vsync),
        PresentMode::Mailbox
    );
}

#[test]
fn test_35_select_power_saving_prefers_fifo() {
    // PowerSaving should select Fifo even when other modes available.
    let caps = caps_all_modes();
    assert_eq!(
        caps.select_present_mode(PresentModePreference::PowerSaving),
        PresentMode::Fifo
    );
}

#[test]
fn test_36_select_adaptive_prefers_fifo_relaxed() {
    // Adaptive should select FifoRelaxed when available.
    let caps = caps_all_modes();
    assert_eq!(
        caps.select_present_mode(PresentModePreference::Adaptive),
        PresentMode::FifoRelaxed
    );
}

#[test]
fn test_37_select_adaptive_fallback_without_fifo_relaxed() {
    // Adaptive without FifoRelaxed should fall back to preferred.
    let caps = caps_fifo_mailbox();
    // Without FifoRelaxed, falls back to preferred_present_mode() which is Mailbox
    assert_eq!(
        caps.select_present_mode(PresentModePreference::Adaptive),
        PresentMode::Mailbox
    );
}

#[test]
fn test_38_select_specific_mode_available() {
    // Specific mode that is available should be selected.
    let caps = caps_all_modes();
    assert_eq!(
        caps.select_present_mode(PresentModePreference::Specific(PresentMode::Mailbox)),
        PresentMode::Mailbox
    );
}

#[test]
fn test_39_select_specific_mode_unavailable_fallback() {
    // Specific mode not available should fall back to preferred.
    let caps = caps_fifo_only();
    // Requesting Immediate but not available -> fallback to Fifo
    assert_eq!(
        caps.select_present_mode(PresentModePreference::Specific(PresentMode::Immediate)),
        PresentMode::Fifo
    );
}

#[test]
fn test_40_supports_immediate_true() {
    let caps = caps_all_modes();
    assert!(caps.supports_immediate());
}

#[test]
fn test_41_supports_immediate_false() {
    let caps = caps_no_immediate();
    assert!(!caps.supports_immediate());
}

#[test]
fn test_42_supports_mailbox_true() {
    let caps = caps_all_modes();
    assert!(caps.supports_mailbox());
}

#[test]
fn test_43_supports_mailbox_false() {
    let caps = caps_no_mailbox();
    assert!(!caps.supports_mailbox());
}

#[test]
fn test_44_supports_fifo_relaxed_true() {
    let caps = caps_vrr_display();
    assert!(caps.supports_fifo_relaxed());
}

#[test]
fn test_45_supports_fifo_relaxed_false() {
    let caps = caps_fifo_mailbox();
    assert!(!caps.supports_fifo_relaxed());
}

// ============================================================================
// 46-55: SurfaceConfiguration with Present Mode Preferences
// ============================================================================

#[test]
fn test_46_config_with_low_latency_preference() {
    let caps = caps_all_modes();
    let config = SurfaceConfiguration::new(1920, 1080)
        .with_present_mode_preference(&caps, PresentModePreference::LowLatency);
    assert_eq!(config.present_mode, PresentMode::Immediate);
}

#[test]
fn test_47_config_with_vsync_preference() {
    let caps = caps_all_modes();
    let config = SurfaceConfiguration::new(1920, 1080)
        .with_present_mode_preference(&caps, PresentModePreference::Vsync);
    assert_eq!(config.present_mode, PresentMode::Mailbox);
}

#[test]
fn test_48_config_with_power_saving_preference() {
    let caps = caps_all_modes();
    let config = SurfaceConfiguration::new(1920, 1080)
        .with_present_mode_preference(&caps, PresentModePreference::PowerSaving);
    assert_eq!(config.present_mode, PresentMode::Fifo);
}

#[test]
fn test_49_config_with_adaptive_preference() {
    let caps = caps_all_modes();
    let config = SurfaceConfiguration::new(1920, 1080)
        .with_present_mode_preference(&caps, PresentModePreference::Adaptive);
    assert_eq!(config.present_mode, PresentMode::FifoRelaxed);
}

#[test]
fn test_50_config_with_specific_preference_available() {
    let caps = caps_all_modes();
    let config = SurfaceConfiguration::new(800, 600)
        .with_present_mode_preference(&caps, PresentModePreference::Specific(PresentMode::FifoRelaxed));
    assert_eq!(config.present_mode, PresentMode::FifoRelaxed);
}

#[test]
fn test_51_config_with_specific_preference_fallback() {
    let caps = caps_fifo_only();
    let config = SurfaceConfiguration::new(800, 600)
        .with_present_mode_preference(&caps, PresentModePreference::Specific(PresentMode::Mailbox));
    // Mailbox not available, falls back to Fifo
    assert_eq!(config.present_mode, PresentMode::Fifo);
}

#[test]
fn test_52_config_preserves_dimensions_after_preference() {
    let caps = caps_all_modes();
    let config = SurfaceConfiguration::new(2560, 1440)
        .with_present_mode_preference(&caps, PresentModePreference::LowLatency);
    assert_eq!(config.width, 2560);
    assert_eq!(config.height, 1440);
}

#[test]
fn test_53_config_chaining_format_then_preference() {
    let caps = caps_all_modes();
    let config = SurfaceConfiguration::new(1920, 1080)
        .with_format(TextureFormat::Rgba8Unorm)
        .with_present_mode_preference(&caps, PresentModePreference::Vsync);
    assert_eq!(config.format, TextureFormat::Rgba8Unorm);
    assert_eq!(config.present_mode, PresentMode::Mailbox);
}

#[test]
fn test_54_config_chaining_preference_then_alpha() {
    let caps = caps_all_modes();
    let config = SurfaceConfiguration::new(1920, 1080)
        .with_present_mode_preference(&caps, PresentModePreference::PowerSaving)
        .with_alpha_mode(CompositeAlphaMode::Opaque);
    assert_eq!(config.present_mode, PresentMode::Fifo);
    assert_eq!(config.alpha_mode, CompositeAlphaMode::Opaque);
}

#[test]
fn test_55_config_preference_overrides_manual_mode() {
    let caps = caps_all_modes();
    let config = SurfaceConfiguration::new(1920, 1080)
        .with_present_mode(PresentMode::Fifo)
        .with_present_mode_preference(&caps, PresentModePreference::LowLatency);
    // Preference should override the manual setting
    assert_eq!(config.present_mode, PresentMode::Immediate);
}

// ============================================================================
// 56-65: Edge Cases and Mobile/Limited Surface Scenarios
// ============================================================================

#[test]
fn test_56_mobile_surface_fifo_only_low_latency() {
    // Mobile surfaces often only support Fifo.
    let caps = caps_fifo_only();
    // Even with LowLatency request, must fall back to Fifo
    assert_eq!(
        caps.select_present_mode(PresentModePreference::LowLatency),
        PresentMode::Fifo
    );
}

#[test]
fn test_57_mobile_surface_fifo_only_vsync() {
    let caps = caps_fifo_only();
    assert_eq!(
        caps.select_present_mode(PresentModePreference::Vsync),
        PresentMode::Fifo
    );
}

#[test]
fn test_58_mobile_surface_fifo_only_adaptive() {
    let caps = caps_fifo_only();
    // Adaptive without FifoRelaxed falls back to preferred, which is Fifo
    assert_eq!(
        caps.select_present_mode(PresentModePreference::Adaptive),
        PresentMode::Fifo
    );
}

#[test]
fn test_59_vrr_display_adaptive_gets_fifo_relaxed() {
    // VRR displays often have Fifo + FifoRelaxed only.
    let caps = caps_vrr_display();
    assert_eq!(
        caps.select_present_mode(PresentModePreference::Adaptive),
        PresentMode::FifoRelaxed
    );
}

#[test]
fn test_60_vrr_display_vsync_gets_fifo_relaxed() {
    // Vsync preference on VRR should get FifoRelaxed (better than plain Fifo).
    let caps = caps_vrr_display();
    assert_eq!(
        caps.select_present_mode(PresentModePreference::Vsync),
        PresentMode::FifoRelaxed
    );
}

#[test]
fn test_61_supports_present_mode_generic() {
    let caps = caps_all_modes();
    assert!(caps.supports_present_mode(PresentMode::Fifo));
    assert!(caps.supports_present_mode(PresentMode::Immediate));
    assert!(caps.supports_present_mode(PresentMode::Mailbox));
    assert!(caps.supports_present_mode(PresentMode::FifoRelaxed));
}

#[test]
fn test_62_supports_present_mode_missing() {
    let caps = caps_fifo_only();
    assert!(caps.supports_present_mode(PresentMode::Fifo));
    assert!(!caps.supports_present_mode(PresentMode::Immediate));
    assert!(!caps.supports_present_mode(PresentMode::Mailbox));
    assert!(!caps.supports_present_mode(PresentMode::FifoRelaxed));
}

#[test]
fn test_63_competitive_gaming_mode_check_all() {
    // Verify competitive gaming mode classification across all modes.
    let modes = [
        (PresentMode::Immediate, true),
        (PresentMode::Mailbox, true),
        (PresentMode::FifoRelaxed, false),
        (PresentMode::Fifo, false),
    ];
    for (mode, expected) in modes {
        let info = PresentModeInfo::from_mode(mode);
        assert_eq!(
            info.is_competitive_gaming_mode(),
            expected,
            "Mode {:?} competitive gaming classification mismatch",
            mode
        );
    }
}

#[test]
fn test_64_battery_friendly_mode_check_all() {
    // Verify battery-friendly classification across all modes.
    let modes = [
        (PresentMode::Immediate, false),
        (PresentMode::Mailbox, false),
        (PresentMode::FifoRelaxed, true),
        (PresentMode::Fifo, true),
    ];
    for (mode, expected) in modes {
        let info = PresentModeInfo::from_mode(mode);
        assert_eq!(
            info.is_battery_friendly(),
            expected,
            "Mode {:?} battery-friendly classification mismatch",
            mode
        );
    }
}

#[test]
fn test_65_present_mode_info_name_non_empty() {
    // All present modes should have non-empty names.
    let modes = [
        PresentMode::Immediate,
        PresentMode::Mailbox,
        PresentMode::FifoRelaxed,
        PresentMode::Fifo,
    ];
    for mode in modes {
        let info = PresentModeInfo::from_mode(mode);
        assert!(!info.name.is_empty(), "Mode {:?} should have a name", mode);
        assert!(!info.description.is_empty(), "Mode {:?} should have a description", mode);
    }
}

// ============================================================================
// Additional Coverage: Edge Cases and Stress Tests
// ============================================================================

#[test]
fn test_66_select_all_preferences_with_single_mode() {
    // When only one mode available, all preferences should return it.
    let caps = caps_fifo_only();
    let preferences = [
        PresentModePreference::LowLatency,
        PresentModePreference::Vsync,
        PresentModePreference::PowerSaving,
        PresentModePreference::Adaptive,
        PresentModePreference::Specific(PresentMode::Immediate),
        PresentModePreference::Specific(PresentMode::Mailbox),
    ];
    for pref in preferences {
        assert_eq!(
            caps.select_present_mode(pref),
            PresentMode::Fifo,
            "Preference {:?} should fall back to Fifo when it's the only option",
            pref
        );
    }
}

#[test]
fn test_67_specific_fifo_always_succeeds() {
    // Specific(Fifo) should always succeed since Fifo is guaranteed.
    let caps_scenarios = [
        caps_all_modes(),
        caps_fifo_only(),
        caps_no_immediate(),
        caps_vrr_display(),
    ];
    for caps in caps_scenarios {
        assert_eq!(
            caps.select_present_mode(PresentModePreference::Specific(PresentMode::Fifo)),
            PresentMode::Fifo
        );
    }
}

#[test]
fn test_68_power_saving_always_prefers_fifo() {
    // PowerSaving should always select Fifo regardless of what else is available.
    let caps_scenarios = [
        caps_all_modes(),
        caps_no_immediate(),
        caps_no_mailbox(),
        caps_vrr_display(),
    ];
    for caps in caps_scenarios {
        assert_eq!(
            caps.select_present_mode(PresentModePreference::PowerSaving),
            PresentMode::Fifo
        );
    }
}

#[test]
fn test_69_low_latency_chain_without_mailbox() {
    // Without Mailbox, low latency should try FifoRelaxed after Immediate.
    let caps = caps_no_mailbox();
    // Has Immediate, so should still get Immediate
    assert_eq!(
        caps.select_present_mode(PresentModePreference::LowLatency),
        PresentMode::Immediate
    );
}

#[test]
fn test_70_low_latency_fifo_immediate_gets_immediate() {
    // With Fifo and Immediate only, low latency gets Immediate.
    let caps = caps_fifo_immediate();
    assert_eq!(caps.low_latency_present_mode(), PresentMode::Immediate);
}

#[test]
fn test_71_vsync_fifo_immediate_gets_fifo() {
    // With Fifo and Immediate only, vsync preference gets Fifo.
    let caps = caps_fifo_immediate();
    // No Mailbox, no FifoRelaxed, falls back to Fifo
    assert_eq!(caps.preferred_present_mode(), PresentMode::Fifo);
}

#[test]
fn test_72_mode_info_equality() {
    // PresentModeInfo should derive Eq.
    let info1 = PresentModeInfo::from_mode(PresentMode::Fifo);
    let info2 = PresentModeInfo::from_mode(PresentMode::Fifo);
    assert_eq!(info1, info2);
}

#[test]
fn test_73_mode_info_inequality_different_modes() {
    let info1 = PresentModeInfo::from_mode(PresentMode::Fifo);
    let info2 = PresentModeInfo::from_mode(PresentMode::Mailbox);
    assert_ne!(info1, info2);
}

#[test]
fn test_74_config_from_capabilities_uses_preferred() {
    // SurfaceConfiguration::from_capabilities should use preferred_present_mode.
    let caps = caps_all_modes();
    let config = SurfaceConfiguration::from_capabilities(&caps, 1920, 1080);
    assert_eq!(config.present_mode, PresentMode::Mailbox);
}

#[test]
fn test_75_config_from_capabilities_fifo_only() {
    let caps = caps_fifo_only();
    let config = SurfaceConfiguration::from_capabilities(&caps, 800, 600);
    assert_eq!(config.present_mode, PresentMode::Fifo);
}

// ============================================================================
// Stress Tests: Many Combinations
// ============================================================================

#[test]
fn test_76_all_preference_mode_combinations() {
    // Test all preference/capability combinations produce valid results.
    let caps_list = [
        ("all", caps_all_modes()),
        ("fifo_only", caps_fifo_only()),
        ("no_immediate", caps_no_immediate()),
        ("no_mailbox", caps_no_mailbox()),
        ("vrr", caps_vrr_display()),
        ("fifo_mailbox", caps_fifo_mailbox()),
        ("fifo_immediate", caps_fifo_immediate()),
    ];
    let preferences = [
        PresentModePreference::LowLatency,
        PresentModePreference::Vsync,
        PresentModePreference::PowerSaving,
        PresentModePreference::Adaptive,
    ];

    for (caps_name, caps) in &caps_list {
        for pref in &preferences {
            let result = caps.select_present_mode(*pref);
            // Result must be one of the supported modes
            assert!(
                caps.supports_present_mode(result),
                "Caps {:?} with pref {:?} returned unsupported mode {:?}",
                caps_name,
                pref,
                result
            );
        }
    }
}

#[test]
fn test_77_specific_mode_combinations() {
    // Test Specific preference with all modes across all capabilities.
    let all_modes = [
        PresentMode::Immediate,
        PresentMode::Mailbox,
        PresentMode::FifoRelaxed,
        PresentMode::Fifo,
    ];
    let caps_list = [
        caps_all_modes(),
        caps_fifo_only(),
        caps_no_immediate(),
        caps_vrr_display(),
    ];

    for caps in &caps_list {
        for mode in &all_modes {
            let result = caps.select_present_mode(PresentModePreference::Specific(*mode));
            // Result must be supported
            assert!(
                caps.supports_present_mode(result),
                "Specific({:?}) returned unsupported mode {:?}",
                mode,
                result
            );
            // If the mode is supported, it should be selected
            if caps.supports_present_mode(*mode) {
                assert_eq!(result, *mode);
            }
        }
    }
}

// ============================================================================
// Present Mode Info Name and Description Consistency
// ============================================================================

#[test]
fn test_78_immediate_info_contains_latency_mention() {
    let info = PresentModeInfo::from_mode(PresentMode::Immediate);
    assert!(
        info.description.to_lowercase().contains("latency")
            || info.description.to_lowercase().contains("tear"),
        "Immediate description should mention latency or tearing"
    );
}

#[test]
fn test_79_mailbox_info_contains_triple_buffer() {
    let info = PresentModeInfo::from_mode(PresentMode::Mailbox);
    assert!(
        info.name.contains("Triple") || info.description.contains("triple"),
        "Mailbox should mention triple buffering"
    );
}

#[test]
fn test_80_fifo_info_contains_vsync() {
    let info = PresentModeInfo::from_mode(PresentMode::Fifo);
    assert!(
        info.name.to_lowercase().contains("vsync")
            || info.description.to_lowercase().contains("vsync"),
        "Fifo should mention vsync"
    );
}

// ============================================================================
// Final Validation Tests
// ============================================================================

#[test]
fn test_81_prevents_tearing_consistency() {
    // All vsync modes should prevent tearing.
    let tearing_modes = [
        (PresentMode::Immediate, false), // Immediate may tear
        (PresentMode::Mailbox, true),    // Triple buffered, no tear
        (PresentMode::FifoRelaxed, true), // Vsync variant
        (PresentMode::Fifo, true),       // Standard vsync
    ];
    for (mode, prevents) in tearing_modes {
        let info = PresentModeInfo::from_mode(mode);
        assert_eq!(
            info.prevents_tearing, prevents,
            "Mode {:?} tearing prevention mismatch",
            mode
        );
    }
}

#[test]
fn test_82_config_validate_with_selected_mode() {
    // Configuration with a selected mode should validate successfully.
    let caps = caps_all_modes();
    let config = SurfaceConfiguration::new(1920, 1080)
        .with_format(TextureFormat::Bgra8Unorm)
        .with_present_mode_preference(&caps, PresentModePreference::LowLatency)
        .with_alpha_mode(CompositeAlphaMode::Auto);

    assert!(config.validate(&caps).is_ok());
}

#[test]
fn test_83_config_validate_fallback_mode() {
    // Configuration with fallback mode should still validate.
    let caps = caps_fifo_only();
    let config = SurfaceConfiguration::new(800, 600)
        .with_format(TextureFormat::Bgra8Unorm)
        .with_present_mode_preference(&caps, PresentModePreference::LowLatency)
        .with_alpha_mode(CompositeAlphaMode::Auto);

    // LowLatency fell back to Fifo, which should validate
    assert_eq!(config.present_mode, PresentMode::Fifo);
    assert!(config.validate(&caps).is_ok());
}

#[test]
fn test_84_describe_present_mode_all_modes() {
    // describe_present_mode should work for all standard modes.
    let modes = [
        PresentMode::Immediate,
        PresentMode::Mailbox,
        PresentMode::FifoRelaxed,
        PresentMode::Fifo,
    ];
    for mode in modes {
        let info = SurfaceCapabilities::describe_present_mode(mode);
        assert_eq!(info.mode, mode);
        assert!(info.latency_rank >= 1 && info.latency_rank <= 4);
    }
}

#[test]
fn test_85_present_mode_preference_specific_all_modes() {
    // Specific variant should work with all standard modes.
    let modes = [
        PresentMode::Immediate,
        PresentMode::Mailbox,
        PresentMode::FifoRelaxed,
        PresentMode::Fifo,
    ];
    for mode in modes {
        let pref = PresentModePreference::Specific(mode);
        let debug = format!("{:?}", pref);
        assert!(debug.contains("Specific"));
    }
}

// ============================================================================
// Edge Case: Empty/Unusual Capabilities (Defensive Tests)
// ============================================================================

#[test]
fn test_86_caps_with_single_immediate() {
    // Unusual: only Immediate mode (theoretical edge case).
    let caps = SurfaceCapabilities {
        formats: vec![TextureFormat::Bgra8Unorm],
        present_modes: vec![PresentMode::Immediate],
        alpha_modes: vec![CompositeAlphaMode::Auto],
        usages: TextureUsages::RENDER_ATTACHMENT,
    };
    // All preferences should return Immediate since it's the only option.
    assert_eq!(
        caps.select_present_mode(PresentModePreference::PowerSaving),
        PresentMode::Immediate
    );
    assert_eq!(caps.low_latency_present_mode(), PresentMode::Immediate);
    assert_eq!(caps.preferred_present_mode(), PresentMode::Immediate);
}

#[test]
fn test_87_low_latency_without_immediate_or_mailbox() {
    // Without Immediate or Mailbox, should get FifoRelaxed for low latency.
    let caps = SurfaceCapabilities {
        formats: vec![TextureFormat::Bgra8Unorm],
        present_modes: vec![PresentMode::Fifo, PresentMode::FifoRelaxed],
        alpha_modes: vec![CompositeAlphaMode::Auto],
        usages: TextureUsages::RENDER_ATTACHMENT,
    };
    assert_eq!(caps.low_latency_present_mode(), PresentMode::FifoRelaxed);
}

#[test]
fn test_88_present_mode_info_copy_trait() {
    // PresentModeInfo should implement Copy.
    let info1 = PresentModeInfo::from_mode(PresentMode::Fifo);
    let info2 = info1; // Copy
    assert_eq!(info1.mode, info2.mode);
}

#[test]
fn test_89_config_to_wgpu_preserves_present_mode() {
    let caps = caps_all_modes();
    let config = SurfaceConfiguration::new(1920, 1080)
        .with_present_mode_preference(&caps, PresentModePreference::LowLatency);

    let wgpu_config = config.to_wgpu();
    assert_eq!(wgpu_config.present_mode, PresentMode::Immediate);
}

#[test]
fn test_90_preference_description_unique() {
    // Each preference should have a unique description.
    let descriptions: Vec<&str> = vec![
        PresentModePreference::LowLatency.description(),
        PresentModePreference::Vsync.description(),
        PresentModePreference::PowerSaving.description(),
        PresentModePreference::Adaptive.description(),
    ];

    for i in 0..descriptions.len() {
        for j in (i + 1)..descriptions.len() {
            assert_ne!(
                descriptions[i], descriptions[j],
                "Descriptions {} and {} should differ",
                i, j
            );
        }
    }
}
