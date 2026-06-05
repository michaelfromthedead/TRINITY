//! Whitebox structural tests for Present Mode Selection Logic
//!
//! These tests verify the internal structure and behavior of the present mode
//! selection algorithm, including PresentModePreference, PresentModeInfo,
//! SurfaceCapabilities present mode methods, and SurfaceConfiguration
//! present mode preference builder.
//!
//! Task: T-WGPU-P7.1.4 - Present Mode Selection
//!
//! Acceptance Criteria Tested:
//! 1. Internal priority chains (LowLatency: Immediate > Mailbox > FifoRelaxed > Fifo)
//! 2. Capability intersection logic
//! 3. Fallback behavior when preferred unavailable
//! 4. PresentModeInfo accuracy (latency_rank, power_efficient flags)
//! 5. Edge cases: empty capabilities, single mode, all modes

use renderer_backend::presentation::{
    FormatCategory, PresentModeInfo, PresentModePreference, SurfaceCapabilities,
    SurfaceConfiguration,
};
use wgpu::{CompositeAlphaMode, PresentMode, TextureFormat, TextureUsages};

// ============================================================================
// Helper Functions
// ============================================================================

/// Create minimal SurfaceCapabilities with specified present modes.
fn caps_with_modes(modes: Vec<PresentMode>) -> SurfaceCapabilities {
    SurfaceCapabilities {
        formats: vec![TextureFormat::Bgra8Unorm],
        present_modes: modes,
        alpha_modes: vec![CompositeAlphaMode::Auto],
        usages: TextureUsages::RENDER_ATTACHMENT,
    }
}

/// Create SurfaceCapabilities with all standard present modes.
fn caps_all_modes() -> SurfaceCapabilities {
    SurfaceCapabilities {
        formats: vec![TextureFormat::Bgra8Unorm],
        present_modes: vec![
            PresentMode::Immediate,
            PresentMode::Mailbox,
            PresentMode::FifoRelaxed,
            PresentMode::Fifo,
        ],
        alpha_modes: vec![CompositeAlphaMode::Auto],
        usages: TextureUsages::RENDER_ATTACHMENT,
    }
}

/// Create SurfaceCapabilities with only Fifo (minimum guaranteed mode).
fn caps_fifo_only() -> SurfaceCapabilities {
    caps_with_modes(vec![PresentMode::Fifo])
}

/// Create empty SurfaceCapabilities (edge case).
fn caps_empty() -> SurfaceCapabilities {
    SurfaceCapabilities {
        formats: vec![],
        present_modes: vec![],
        alpha_modes: vec![],
        usages: TextureUsages::RENDER_ATTACHMENT,
    }
}

// ============================================================================
// 1. PresentModePreference Enum Tests
// ============================================================================

mod present_mode_preference_enum {
    use super::*;

    #[test]
    fn low_latency_variant_exists() {
        let pref = PresentModePreference::LowLatency;
        assert_eq!(pref, PresentModePreference::LowLatency);
    }

    #[test]
    fn vsync_variant_exists() {
        let pref = PresentModePreference::Vsync;
        assert_eq!(pref, PresentModePreference::Vsync);
    }

    #[test]
    fn power_saving_variant_exists() {
        let pref = PresentModePreference::PowerSaving;
        assert_eq!(pref, PresentModePreference::PowerSaving);
    }

    #[test]
    fn adaptive_variant_exists() {
        let pref = PresentModePreference::Adaptive;
        assert_eq!(pref, PresentModePreference::Adaptive);
    }

    #[test]
    fn specific_variant_with_immediate() {
        let pref = PresentModePreference::Specific(PresentMode::Immediate);
        if let PresentModePreference::Specific(mode) = pref {
            assert_eq!(mode, PresentMode::Immediate);
        } else {
            panic!("Expected Specific variant");
        }
    }

    #[test]
    fn specific_variant_with_mailbox() {
        let pref = PresentModePreference::Specific(PresentMode::Mailbox);
        if let PresentModePreference::Specific(mode) = pref {
            assert_eq!(mode, PresentMode::Mailbox);
        } else {
            panic!("Expected Specific variant");
        }
    }

    #[test]
    fn specific_variant_with_fifo() {
        let pref = PresentModePreference::Specific(PresentMode::Fifo);
        if let PresentModePreference::Specific(mode) = pref {
            assert_eq!(mode, PresentMode::Fifo);
        } else {
            panic!("Expected Specific variant");
        }
    }

    #[test]
    fn specific_variant_with_fifo_relaxed() {
        let pref = PresentModePreference::Specific(PresentMode::FifoRelaxed);
        if let PresentModePreference::Specific(mode) = pref {
            assert_eq!(mode, PresentMode::FifoRelaxed);
        } else {
            panic!("Expected Specific variant");
        }
    }

    #[test]
    fn default_is_vsync() {
        assert_eq!(PresentModePreference::default(), PresentModePreference::Vsync);
    }

    #[test]
    fn enum_is_copy() {
        let pref = PresentModePreference::LowLatency;
        let copy = pref;
        assert_eq!(pref, copy);
    }

    #[test]
    fn enum_is_clone() {
        let pref = PresentModePreference::Vsync;
        let cloned = pref.clone();
        assert_eq!(pref, cloned);
    }

    #[test]
    fn enum_is_eq_same_variant() {
        assert_eq!(PresentModePreference::LowLatency, PresentModePreference::LowLatency);
        assert_eq!(PresentModePreference::Vsync, PresentModePreference::Vsync);
    }

    #[test]
    fn enum_is_ne_different_variants() {
        assert_ne!(PresentModePreference::LowLatency, PresentModePreference::Vsync);
        assert_ne!(PresentModePreference::PowerSaving, PresentModePreference::Adaptive);
    }

    #[test]
    fn specific_variants_eq_same_mode() {
        assert_eq!(
            PresentModePreference::Specific(PresentMode::Immediate),
            PresentModePreference::Specific(PresentMode::Immediate)
        );
    }

    #[test]
    fn specific_variants_ne_different_modes() {
        assert_ne!(
            PresentModePreference::Specific(PresentMode::Immediate),
            PresentModePreference::Specific(PresentMode::Fifo)
        );
    }

    #[test]
    fn enum_is_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(PresentModePreference::LowLatency);
        set.insert(PresentModePreference::Vsync);
        assert!(set.contains(&PresentModePreference::LowLatency));
        assert!(set.contains(&PresentModePreference::Vsync));
    }

    #[test]
    fn description_low_latency_not_empty() {
        let desc = PresentModePreference::LowLatency.description();
        assert!(!desc.is_empty());
        assert!(desc.contains("latency") || desc.contains("Latency") || desc.contains("tear"));
    }

    #[test]
    fn description_vsync_not_empty() {
        let desc = PresentModePreference::Vsync.description();
        assert!(!desc.is_empty());
        assert!(desc.contains("vsync") || desc.contains("Vsync") || desc.contains("smooth"));
    }

    #[test]
    fn description_power_saving_not_empty() {
        let desc = PresentModePreference::PowerSaving.description();
        assert!(!desc.is_empty());
        assert!(desc.to_lowercase().contains("power") || desc.to_lowercase().contains("efficient"));
    }

    #[test]
    fn description_adaptive_not_empty() {
        let desc = PresentModePreference::Adaptive.description();
        assert!(!desc.is_empty());
        assert!(desc.to_lowercase().contains("adaptive") || desc.to_lowercase().contains("drop"));
    }

    #[test]
    fn description_specific_not_empty() {
        let desc = PresentModePreference::Specific(PresentMode::Fifo).description();
        assert!(!desc.is_empty());
    }

    #[test]
    fn display_low_latency() {
        let s = format!("{}", PresentModePreference::LowLatency);
        assert!(s.contains("Low") || s.contains("Latency"));
    }

    #[test]
    fn display_vsync() {
        let s = format!("{}", PresentModePreference::Vsync);
        assert!(s.contains("Vsync") || s.contains("vsync"));
    }

    #[test]
    fn display_power_saving() {
        let s = format!("{}", PresentModePreference::PowerSaving);
        assert!(s.contains("Power") || s.contains("Saving"));
    }

    #[test]
    fn display_adaptive() {
        let s = format!("{}", PresentModePreference::Adaptive);
        assert!(s.contains("Adaptive") || s.contains("adaptive"));
    }

    #[test]
    fn display_specific_contains_specific() {
        let s = format!("{}", PresentModePreference::Specific(PresentMode::Immediate));
        assert!(s.contains("Specific"));
    }

    #[test]
    fn display_specific_contains_mode() {
        let s = format!("{}", PresentModePreference::Specific(PresentMode::Fifo));
        assert!(s.contains("Fifo"));
    }
}

// ============================================================================
// 2. PresentModeInfo Struct Tests
// ============================================================================

mod present_mode_info_struct {
    use super::*;

    #[test]
    fn from_mode_immediate_mode_field() {
        let info = PresentModeInfo::from_mode(PresentMode::Immediate);
        assert_eq!(info.mode, PresentMode::Immediate);
    }

    #[test]
    fn from_mode_immediate_name() {
        let info = PresentModeInfo::from_mode(PresentMode::Immediate);
        assert!(info.name.contains("Immediate"));
    }

    #[test]
    fn from_mode_immediate_description() {
        let info = PresentModeInfo::from_mode(PresentMode::Immediate);
        assert!(!info.description.is_empty());
    }

    #[test]
    fn from_mode_immediate_no_tearing_prevention() {
        let info = PresentModeInfo::from_mode(PresentMode::Immediate);
        assert!(!info.prevents_tearing);
    }

    #[test]
    fn from_mode_immediate_lowest_latency_rank() {
        let info = PresentModeInfo::from_mode(PresentMode::Immediate);
        assert_eq!(info.latency_rank, 1);
    }

    #[test]
    fn from_mode_immediate_not_power_efficient() {
        let info = PresentModeInfo::from_mode(PresentMode::Immediate);
        assert!(!info.power_efficient);
    }

    #[test]
    fn from_mode_mailbox_mode_field() {
        let info = PresentModeInfo::from_mode(PresentMode::Mailbox);
        assert_eq!(info.mode, PresentMode::Mailbox);
    }

    #[test]
    fn from_mode_mailbox_name() {
        let info = PresentModeInfo::from_mode(PresentMode::Mailbox);
        assert!(info.name.contains("Mailbox") || info.name.contains("Triple"));
    }

    #[test]
    fn from_mode_mailbox_prevents_tearing() {
        let info = PresentModeInfo::from_mode(PresentMode::Mailbox);
        assert!(info.prevents_tearing);
    }

    #[test]
    fn from_mode_mailbox_latency_rank() {
        let info = PresentModeInfo::from_mode(PresentMode::Mailbox);
        assert_eq!(info.latency_rank, 2);
    }

    #[test]
    fn from_mode_mailbox_not_power_efficient() {
        let info = PresentModeInfo::from_mode(PresentMode::Mailbox);
        assert!(!info.power_efficient);
    }

    #[test]
    fn from_mode_fifo_mode_field() {
        let info = PresentModeInfo::from_mode(PresentMode::Fifo);
        assert_eq!(info.mode, PresentMode::Fifo);
    }

    #[test]
    fn from_mode_fifo_name() {
        let info = PresentModeInfo::from_mode(PresentMode::Fifo);
        assert!(info.name.contains("Fifo") || info.name.contains("Vsync"));
    }

    #[test]
    fn from_mode_fifo_prevents_tearing() {
        let info = PresentModeInfo::from_mode(PresentMode::Fifo);
        assert!(info.prevents_tearing);
    }

    #[test]
    fn from_mode_fifo_highest_latency_rank() {
        let info = PresentModeInfo::from_mode(PresentMode::Fifo);
        assert_eq!(info.latency_rank, 4);
    }

    #[test]
    fn from_mode_fifo_power_efficient() {
        let info = PresentModeInfo::from_mode(PresentMode::Fifo);
        assert!(info.power_efficient);
    }

    #[test]
    fn from_mode_fifo_relaxed_mode_field() {
        let info = PresentModeInfo::from_mode(PresentMode::FifoRelaxed);
        assert_eq!(info.mode, PresentMode::FifoRelaxed);
    }

    #[test]
    fn from_mode_fifo_relaxed_name() {
        let info = PresentModeInfo::from_mode(PresentMode::FifoRelaxed);
        assert!(info.name.contains("Relaxed") || info.name.contains("Adaptive"));
    }

    #[test]
    fn from_mode_fifo_relaxed_prevents_tearing() {
        let info = PresentModeInfo::from_mode(PresentMode::FifoRelaxed);
        assert!(info.prevents_tearing);
    }

    #[test]
    fn from_mode_fifo_relaxed_latency_rank() {
        let info = PresentModeInfo::from_mode(PresentMode::FifoRelaxed);
        assert_eq!(info.latency_rank, 3);
    }

    #[test]
    fn from_mode_fifo_relaxed_power_efficient() {
        let info = PresentModeInfo::from_mode(PresentMode::FifoRelaxed);
        assert!(info.power_efficient);
    }

    #[test]
    fn latency_rank_ordering_immediate_best() {
        let immediate = PresentModeInfo::from_mode(PresentMode::Immediate);
        let mailbox = PresentModeInfo::from_mode(PresentMode::Mailbox);
        let fifo_relaxed = PresentModeInfo::from_mode(PresentMode::FifoRelaxed);
        let fifo = PresentModeInfo::from_mode(PresentMode::Fifo);

        assert!(immediate.latency_rank < mailbox.latency_rank);
        assert!(mailbox.latency_rank < fifo_relaxed.latency_rank);
        assert!(fifo_relaxed.latency_rank < fifo.latency_rank);
    }

    #[test]
    fn is_competitive_gaming_mode_immediate() {
        let info = PresentModeInfo::from_mode(PresentMode::Immediate);
        assert!(info.is_competitive_gaming_mode());
    }

    #[test]
    fn is_competitive_gaming_mode_mailbox() {
        let info = PresentModeInfo::from_mode(PresentMode::Mailbox);
        assert!(info.is_competitive_gaming_mode());
    }

    #[test]
    fn is_not_competitive_gaming_mode_fifo_relaxed() {
        let info = PresentModeInfo::from_mode(PresentMode::FifoRelaxed);
        assert!(!info.is_competitive_gaming_mode());
    }

    #[test]
    fn is_not_competitive_gaming_mode_fifo() {
        let info = PresentModeInfo::from_mode(PresentMode::Fifo);
        assert!(!info.is_competitive_gaming_mode());
    }

    #[test]
    fn is_battery_friendly_fifo() {
        let info = PresentModeInfo::from_mode(PresentMode::Fifo);
        assert!(info.is_battery_friendly());
    }

    #[test]
    fn is_battery_friendly_fifo_relaxed() {
        let info = PresentModeInfo::from_mode(PresentMode::FifoRelaxed);
        assert!(info.is_battery_friendly());
    }

    #[test]
    fn is_not_battery_friendly_immediate() {
        let info = PresentModeInfo::from_mode(PresentMode::Immediate);
        assert!(!info.is_battery_friendly());
    }

    #[test]
    fn is_not_battery_friendly_mailbox() {
        let info = PresentModeInfo::from_mode(PresentMode::Mailbox);
        assert!(!info.is_battery_friendly());
    }

    #[test]
    fn display_format_contains_name() {
        let info = PresentModeInfo::from_mode(PresentMode::Mailbox);
        let s = format!("{}", info);
        assert!(s.contains("Mailbox") || s.contains("Triple"));
    }

    #[test]
    fn display_format_contains_description() {
        let info = PresentModeInfo::from_mode(PresentMode::Immediate);
        let s = format!("{}", info);
        assert!(s.len() > info.name.len());
    }

    #[test]
    fn struct_is_copy() {
        let info = PresentModeInfo::from_mode(PresentMode::Fifo);
        let copy = info;
        assert_eq!(info.mode, copy.mode);
    }

    #[test]
    fn struct_is_clone() {
        let info = PresentModeInfo::from_mode(PresentMode::Fifo);
        let cloned = info.clone();
        assert_eq!(info.mode, cloned.mode);
    }

    #[test]
    fn struct_is_eq() {
        let a = PresentModeInfo::from_mode(PresentMode::Mailbox);
        let b = PresentModeInfo::from_mode(PresentMode::Mailbox);
        assert_eq!(a, b);
    }

    #[test]
    fn struct_is_ne_different_modes() {
        let a = PresentModeInfo::from_mode(PresentMode::Immediate);
        let b = PresentModeInfo::from_mode(PresentMode::Fifo);
        assert_ne!(a, b);
    }

    #[test]
    fn struct_is_debug() {
        let info = PresentModeInfo::from_mode(PresentMode::Mailbox);
        let s = format!("{:?}", info);
        assert!(s.contains("PresentModeInfo"));
    }
}

// ============================================================================
// 3. LowLatency Priority Chain Tests
// ============================================================================

mod low_latency_priority_chain {
    use super::*;

    #[test]
    fn low_latency_selects_immediate_when_all_available() {
        let caps = caps_all_modes();
        assert_eq!(caps.low_latency_present_mode(), PresentMode::Immediate);
    }

    #[test]
    fn low_latency_selects_mailbox_when_no_immediate() {
        let caps = caps_with_modes(vec![
            PresentMode::Mailbox,
            PresentMode::FifoRelaxed,
            PresentMode::Fifo,
        ]);
        assert_eq!(caps.low_latency_present_mode(), PresentMode::Mailbox);
    }

    #[test]
    fn low_latency_selects_fifo_relaxed_when_no_immediate_or_mailbox() {
        let caps = caps_with_modes(vec![PresentMode::FifoRelaxed, PresentMode::Fifo]);
        assert_eq!(caps.low_latency_present_mode(), PresentMode::FifoRelaxed);
    }

    #[test]
    fn low_latency_selects_fifo_when_only_fifo() {
        let caps = caps_fifo_only();
        assert_eq!(caps.low_latency_present_mode(), PresentMode::Fifo);
    }

    #[test]
    fn low_latency_returns_first_when_empty() {
        let caps = caps_empty();
        // Should return Fifo as fallback
        assert_eq!(caps.low_latency_present_mode(), PresentMode::Fifo);
    }

    #[test]
    fn low_latency_priority_immediate_over_mailbox() {
        let caps = caps_with_modes(vec![PresentMode::Mailbox, PresentMode::Immediate]);
        assert_eq!(caps.low_latency_present_mode(), PresentMode::Immediate);
    }

    #[test]
    fn low_latency_priority_mailbox_over_fifo_relaxed() {
        let caps = caps_with_modes(vec![PresentMode::FifoRelaxed, PresentMode::Mailbox]);
        assert_eq!(caps.low_latency_present_mode(), PresentMode::Mailbox);
    }

    #[test]
    fn low_latency_priority_fifo_relaxed_over_fifo() {
        let caps = caps_with_modes(vec![PresentMode::Fifo, PresentMode::FifoRelaxed]);
        assert_eq!(caps.low_latency_present_mode(), PresentMode::FifoRelaxed);
    }

    #[test]
    fn select_present_mode_low_latency_delegates() {
        let caps = caps_all_modes();
        assert_eq!(
            caps.select_present_mode(PresentModePreference::LowLatency),
            caps.low_latency_present_mode()
        );
    }

    #[test]
    fn low_latency_with_single_mode_immediate() {
        let caps = caps_with_modes(vec![PresentMode::Immediate]);
        assert_eq!(caps.low_latency_present_mode(), PresentMode::Immediate);
    }

    #[test]
    fn low_latency_with_single_mode_mailbox() {
        let caps = caps_with_modes(vec![PresentMode::Mailbox]);
        assert_eq!(caps.low_latency_present_mode(), PresentMode::Mailbox);
    }

    #[test]
    fn low_latency_order_independence_test_1() {
        // Order: Fifo, Immediate, Mailbox
        let caps = caps_with_modes(vec![
            PresentMode::Fifo,
            PresentMode::Immediate,
            PresentMode::Mailbox,
        ]);
        assert_eq!(caps.low_latency_present_mode(), PresentMode::Immediate);
    }

    #[test]
    fn low_latency_order_independence_test_2() {
        // Order: Mailbox, FifoRelaxed, Immediate, Fifo
        let caps = caps_with_modes(vec![
            PresentMode::Mailbox,
            PresentMode::FifoRelaxed,
            PresentMode::Immediate,
            PresentMode::Fifo,
        ]);
        assert_eq!(caps.low_latency_present_mode(), PresentMode::Immediate);
    }
}

// ============================================================================
// 4. Vsync Priority Chain Tests
// ============================================================================

mod vsync_priority_chain {
    use super::*;

    #[test]
    fn vsync_selects_mailbox_when_all_available() {
        let caps = caps_all_modes();
        assert_eq!(caps.preferred_present_mode(), PresentMode::Mailbox);
    }

    #[test]
    fn vsync_selects_fifo_relaxed_when_no_mailbox() {
        let caps = caps_with_modes(vec![
            PresentMode::FifoRelaxed,
            PresentMode::Fifo,
            PresentMode::Immediate,
        ]);
        assert_eq!(caps.preferred_present_mode(), PresentMode::FifoRelaxed);
    }

    #[test]
    fn vsync_selects_fifo_when_no_mailbox_or_fifo_relaxed() {
        let caps = caps_with_modes(vec![PresentMode::Fifo, PresentMode::Immediate]);
        assert_eq!(caps.preferred_present_mode(), PresentMode::Fifo);
    }

    #[test]
    fn vsync_selects_fifo_when_only_fifo() {
        let caps = caps_fifo_only();
        assert_eq!(caps.preferred_present_mode(), PresentMode::Fifo);
    }

    #[test]
    fn vsync_priority_mailbox_over_fifo_relaxed() {
        let caps = caps_with_modes(vec![PresentMode::FifoRelaxed, PresentMode::Mailbox]);
        assert_eq!(caps.preferred_present_mode(), PresentMode::Mailbox);
    }

    #[test]
    fn vsync_priority_fifo_relaxed_over_fifo() {
        let caps = caps_with_modes(vec![PresentMode::Fifo, PresentMode::FifoRelaxed]);
        assert_eq!(caps.preferred_present_mode(), PresentMode::FifoRelaxed);
    }

    #[test]
    fn vsync_ignores_immediate() {
        let caps = caps_with_modes(vec![PresentMode::Immediate, PresentMode::Fifo]);
        // Should not select Immediate for vsync preference
        assert_eq!(caps.preferred_present_mode(), PresentMode::Fifo);
    }

    #[test]
    fn select_present_mode_vsync_delegates() {
        let caps = caps_all_modes();
        assert_eq!(
            caps.select_present_mode(PresentModePreference::Vsync),
            caps.preferred_present_mode()
        );
    }

    #[test]
    fn vsync_with_single_mode_mailbox() {
        let caps = caps_with_modes(vec![PresentMode::Mailbox]);
        assert_eq!(caps.preferred_present_mode(), PresentMode::Mailbox);
    }

    #[test]
    fn vsync_order_independence_test() {
        let caps = caps_with_modes(vec![
            PresentMode::Fifo,
            PresentMode::Immediate,
            PresentMode::FifoRelaxed,
            PresentMode::Mailbox,
        ]);
        assert_eq!(caps.preferred_present_mode(), PresentMode::Mailbox);
    }

    #[test]
    fn vsync_returns_first_when_empty() {
        let caps = caps_empty();
        // Should return Fifo as fallback
        assert_eq!(caps.preferred_present_mode(), PresentMode::Fifo);
    }
}

// ============================================================================
// 5. PowerSaving Priority Chain Tests
// ============================================================================

mod power_saving_priority_chain {
    use super::*;

    #[test]
    fn power_saving_selects_fifo_when_available() {
        let caps = caps_all_modes();
        assert_eq!(
            caps.select_present_mode(PresentModePreference::PowerSaving),
            PresentMode::Fifo
        );
    }

    #[test]
    fn power_saving_prefers_fifo_over_mailbox() {
        let caps = caps_with_modes(vec![PresentMode::Mailbox, PresentMode::Fifo]);
        assert_eq!(
            caps.select_present_mode(PresentModePreference::PowerSaving),
            PresentMode::Fifo
        );
    }

    #[test]
    fn power_saving_prefers_fifo_over_immediate() {
        let caps = caps_with_modes(vec![PresentMode::Immediate, PresentMode::Fifo]);
        assert_eq!(
            caps.select_present_mode(PresentModePreference::PowerSaving),
            PresentMode::Fifo
        );
    }

    #[test]
    fn power_saving_prefers_fifo_over_fifo_relaxed() {
        let caps = caps_with_modes(vec![PresentMode::FifoRelaxed, PresentMode::Fifo]);
        assert_eq!(
            caps.select_present_mode(PresentModePreference::PowerSaving),
            PresentMode::Fifo
        );
    }

    #[test]
    fn power_saving_falls_back_to_vsync_when_no_fifo() {
        // This case shouldn't happen in practice (Fifo is always available)
        // but the code handles it
        let caps = caps_with_modes(vec![PresentMode::Mailbox, PresentMode::Immediate]);
        let result = caps.select_present_mode(PresentModePreference::PowerSaving);
        // Falls back to preferred_present_mode which selects Mailbox
        assert_eq!(result, PresentMode::Mailbox);
    }

    #[test]
    fn power_saving_order_independence() {
        let caps = caps_with_modes(vec![
            PresentMode::Immediate,
            PresentMode::Mailbox,
            PresentMode::Fifo,
            PresentMode::FifoRelaxed,
        ]);
        assert_eq!(
            caps.select_present_mode(PresentModePreference::PowerSaving),
            PresentMode::Fifo
        );
    }

    #[test]
    fn power_saving_with_single_mode_fifo() {
        let caps = caps_fifo_only();
        assert_eq!(
            caps.select_present_mode(PresentModePreference::PowerSaving),
            PresentMode::Fifo
        );
    }

    #[test]
    fn power_efficient_modes_are_fifo_and_fifo_relaxed() {
        let fifo_info = PresentModeInfo::from_mode(PresentMode::Fifo);
        let fifo_relaxed_info = PresentModeInfo::from_mode(PresentMode::FifoRelaxed);
        let mailbox_info = PresentModeInfo::from_mode(PresentMode::Mailbox);
        let immediate_info = PresentModeInfo::from_mode(PresentMode::Immediate);

        assert!(fifo_info.power_efficient);
        assert!(fifo_relaxed_info.power_efficient);
        assert!(!mailbox_info.power_efficient);
        assert!(!immediate_info.power_efficient);
    }
}

// ============================================================================
// 6. Adaptive Priority Chain Tests
// ============================================================================

mod adaptive_priority_chain {
    use super::*;

    #[test]
    fn adaptive_selects_fifo_relaxed_when_available() {
        let caps = caps_all_modes();
        assert_eq!(
            caps.select_present_mode(PresentModePreference::Adaptive),
            PresentMode::FifoRelaxed
        );
    }

    #[test]
    fn adaptive_prefers_fifo_relaxed_over_mailbox() {
        let caps = caps_with_modes(vec![PresentMode::Mailbox, PresentMode::FifoRelaxed]);
        assert_eq!(
            caps.select_present_mode(PresentModePreference::Adaptive),
            PresentMode::FifoRelaxed
        );
    }

    #[test]
    fn adaptive_prefers_fifo_relaxed_over_fifo() {
        let caps = caps_with_modes(vec![PresentMode::Fifo, PresentMode::FifoRelaxed]);
        assert_eq!(
            caps.select_present_mode(PresentModePreference::Adaptive),
            PresentMode::FifoRelaxed
        );
    }

    #[test]
    fn adaptive_falls_back_to_vsync_when_no_fifo_relaxed() {
        let caps = caps_with_modes(vec![PresentMode::Mailbox, PresentMode::Fifo]);
        // Falls back to preferred_present_mode
        assert_eq!(
            caps.select_present_mode(PresentModePreference::Adaptive),
            PresentMode::Mailbox
        );
    }

    #[test]
    fn adaptive_falls_back_to_fifo_when_only_fifo() {
        let caps = caps_fifo_only();
        assert_eq!(
            caps.select_present_mode(PresentModePreference::Adaptive),
            PresentMode::Fifo
        );
    }

    #[test]
    fn adaptive_order_independence() {
        let caps = caps_with_modes(vec![
            PresentMode::Fifo,
            PresentMode::Immediate,
            PresentMode::FifoRelaxed,
            PresentMode::Mailbox,
        ]);
        assert_eq!(
            caps.select_present_mode(PresentModePreference::Adaptive),
            PresentMode::FifoRelaxed
        );
    }

    #[test]
    fn adaptive_with_single_mode_fifo_relaxed() {
        let caps = caps_with_modes(vec![PresentMode::FifoRelaxed]);
        assert_eq!(
            caps.select_present_mode(PresentModePreference::Adaptive),
            PresentMode::FifoRelaxed
        );
    }
}

// ============================================================================
// 7. Specific Mode Selection Tests
// ============================================================================

mod specific_mode_selection {
    use super::*;

    #[test]
    fn specific_immediate_when_available() {
        let caps = caps_all_modes();
        assert_eq!(
            caps.select_present_mode(PresentModePreference::Specific(PresentMode::Immediate)),
            PresentMode::Immediate
        );
    }

    #[test]
    fn specific_mailbox_when_available() {
        let caps = caps_all_modes();
        assert_eq!(
            caps.select_present_mode(PresentModePreference::Specific(PresentMode::Mailbox)),
            PresentMode::Mailbox
        );
    }

    #[test]
    fn specific_fifo_when_available() {
        let caps = caps_all_modes();
        assert_eq!(
            caps.select_present_mode(PresentModePreference::Specific(PresentMode::Fifo)),
            PresentMode::Fifo
        );
    }

    #[test]
    fn specific_fifo_relaxed_when_available() {
        let caps = caps_all_modes();
        assert_eq!(
            caps.select_present_mode(PresentModePreference::Specific(PresentMode::FifoRelaxed)),
            PresentMode::FifoRelaxed
        );
    }

    #[test]
    fn specific_immediate_fallback_when_unavailable() {
        let caps = caps_fifo_only();
        // Immediate not available, falls back to vsync preference
        assert_eq!(
            caps.select_present_mode(PresentModePreference::Specific(PresentMode::Immediate)),
            PresentMode::Fifo
        );
    }

    #[test]
    fn specific_mailbox_fallback_when_unavailable() {
        let caps = caps_fifo_only();
        assert_eq!(
            caps.select_present_mode(PresentModePreference::Specific(PresentMode::Mailbox)),
            PresentMode::Fifo
        );
    }

    #[test]
    fn specific_fifo_relaxed_fallback_when_unavailable() {
        let caps = caps_fifo_only();
        assert_eq!(
            caps.select_present_mode(PresentModePreference::Specific(PresentMode::FifoRelaxed)),
            PresentMode::Fifo
        );
    }

    #[test]
    fn specific_with_partial_availability() {
        let caps = caps_with_modes(vec![PresentMode::Mailbox, PresentMode::Fifo]);
        // Request Immediate, not available, falls back
        assert_eq!(
            caps.select_present_mode(PresentModePreference::Specific(PresentMode::Immediate)),
            PresentMode::Mailbox // preferred_present_mode selects Mailbox
        );
    }

    #[test]
    fn specific_respects_exact_mode() {
        let caps = caps_with_modes(vec![
            PresentMode::Immediate,
            PresentMode::Mailbox,
            PresentMode::Fifo,
        ]);
        // Request Fifo specifically even though lower priority modes available
        assert_eq!(
            caps.select_present_mode(PresentModePreference::Specific(PresentMode::Fifo)),
            PresentMode::Fifo
        );
    }
}

// ============================================================================
// 8. Capability Support Query Tests
// ============================================================================

mod capability_support_queries {
    use super::*;

    #[test]
    fn supports_immediate_true_when_present() {
        let caps = caps_with_modes(vec![PresentMode::Immediate, PresentMode::Fifo]);
        assert!(caps.supports_immediate());
    }

    #[test]
    fn supports_immediate_false_when_absent() {
        let caps = caps_fifo_only();
        assert!(!caps.supports_immediate());
    }

    #[test]
    fn supports_mailbox_true_when_present() {
        let caps = caps_with_modes(vec![PresentMode::Mailbox, PresentMode::Fifo]);
        assert!(caps.supports_mailbox());
    }

    #[test]
    fn supports_mailbox_false_when_absent() {
        let caps = caps_fifo_only();
        assert!(!caps.supports_mailbox());
    }

    #[test]
    fn supports_fifo_relaxed_true_when_present() {
        let caps = caps_with_modes(vec![PresentMode::FifoRelaxed, PresentMode::Fifo]);
        assert!(caps.supports_fifo_relaxed());
    }

    #[test]
    fn supports_fifo_relaxed_false_when_absent() {
        let caps = caps_fifo_only();
        assert!(!caps.supports_fifo_relaxed());
    }

    #[test]
    fn supports_present_mode_fifo() {
        let caps = caps_fifo_only();
        assert!(caps.supports_present_mode(PresentMode::Fifo));
    }

    #[test]
    fn supports_present_mode_false_for_missing() {
        let caps = caps_fifo_only();
        assert!(!caps.supports_present_mode(PresentMode::Immediate));
    }

    #[test]
    fn supports_all_modes_when_all_present() {
        let caps = caps_all_modes();
        assert!(caps.supports_immediate());
        assert!(caps.supports_mailbox());
        assert!(caps.supports_fifo_relaxed());
        assert!(caps.supports_present_mode(PresentMode::Fifo));
    }

    #[test]
    fn supports_none_when_empty() {
        let caps = caps_empty();
        assert!(!caps.supports_immediate());
        assert!(!caps.supports_mailbox());
        assert!(!caps.supports_fifo_relaxed());
        assert!(!caps.supports_present_mode(PresentMode::Fifo));
    }
}

// ============================================================================
// 9. describe_present_mode Static Helper Tests
// ============================================================================

mod describe_present_mode {
    use super::*;

    #[test]
    fn describe_immediate() {
        let info = SurfaceCapabilities::describe_present_mode(PresentMode::Immediate);
        assert_eq!(info.mode, PresentMode::Immediate);
        assert_eq!(info.latency_rank, 1);
    }

    #[test]
    fn describe_mailbox() {
        let info = SurfaceCapabilities::describe_present_mode(PresentMode::Mailbox);
        assert_eq!(info.mode, PresentMode::Mailbox);
        assert_eq!(info.latency_rank, 2);
    }

    #[test]
    fn describe_fifo() {
        let info = SurfaceCapabilities::describe_present_mode(PresentMode::Fifo);
        assert_eq!(info.mode, PresentMode::Fifo);
        assert_eq!(info.latency_rank, 4);
    }

    #[test]
    fn describe_fifo_relaxed() {
        let info = SurfaceCapabilities::describe_present_mode(PresentMode::FifoRelaxed);
        assert_eq!(info.mode, PresentMode::FifoRelaxed);
        assert_eq!(info.latency_rank, 3);
    }

    #[test]
    fn describe_returns_same_as_from_mode() {
        let via_caps = SurfaceCapabilities::describe_present_mode(PresentMode::Mailbox);
        let via_info = PresentModeInfo::from_mode(PresentMode::Mailbox);
        assert_eq!(via_caps, via_info);
    }
}

// ============================================================================
// 10. SurfaceConfiguration Present Mode Preference Builder Tests
// ============================================================================

mod surface_configuration_builder {
    use super::*;

    #[test]
    fn with_present_mode_preference_low_latency() {
        let caps = caps_all_modes();
        let config = SurfaceConfiguration::new(1920, 1080)
            .with_present_mode_preference(&caps, PresentModePreference::LowLatency);
        assert_eq!(config.present_mode, PresentMode::Immediate);
    }

    #[test]
    fn with_present_mode_preference_vsync() {
        let caps = caps_all_modes();
        let config = SurfaceConfiguration::new(1920, 1080)
            .with_present_mode_preference(&caps, PresentModePreference::Vsync);
        assert_eq!(config.present_mode, PresentMode::Mailbox);
    }

    #[test]
    fn with_present_mode_preference_power_saving() {
        let caps = caps_all_modes();
        let config = SurfaceConfiguration::new(1920, 1080)
            .with_present_mode_preference(&caps, PresentModePreference::PowerSaving);
        assert_eq!(config.present_mode, PresentMode::Fifo);
    }

    #[test]
    fn with_present_mode_preference_adaptive() {
        let caps = caps_all_modes();
        let config = SurfaceConfiguration::new(1920, 1080)
            .with_present_mode_preference(&caps, PresentModePreference::Adaptive);
        assert_eq!(config.present_mode, PresentMode::FifoRelaxed);
    }

    #[test]
    fn with_present_mode_preference_specific() {
        let caps = caps_all_modes();
        let config = SurfaceConfiguration::new(1920, 1080)
            .with_present_mode_preference(&caps, PresentModePreference::Specific(PresentMode::Fifo));
        assert_eq!(config.present_mode, PresentMode::Fifo);
    }

    #[test]
    fn with_present_mode_preference_fallback() {
        let caps = caps_fifo_only();
        let config = SurfaceConfiguration::new(1920, 1080)
            .with_present_mode_preference(&caps, PresentModePreference::LowLatency);
        // Falls back to Fifo since that's all that's available
        assert_eq!(config.present_mode, PresentMode::Fifo);
    }

    #[test]
    fn with_present_mode_direct_vs_preference() {
        let caps = caps_all_modes();

        let config_direct = SurfaceConfiguration::new(1920, 1080)
            .with_present_mode(PresentMode::Immediate);

        let config_pref = SurfaceConfiguration::new(1920, 1080)
            .with_present_mode_preference(&caps, PresentModePreference::LowLatency);

        assert_eq!(config_direct.present_mode, config_pref.present_mode);
    }

    #[test]
    fn builder_chain_preserves_dimensions() {
        let caps = caps_all_modes();
        let config = SurfaceConfiguration::new(1920, 1080)
            .with_present_mode_preference(&caps, PresentModePreference::Vsync);
        assert_eq!(config.width, 1920);
        assert_eq!(config.height, 1080);
    }

    #[test]
    fn builder_chain_preserves_other_settings() {
        let caps = caps_all_modes();
        let config = SurfaceConfiguration::new(1920, 1080)
            .with_format(TextureFormat::Rgba8UnormSrgb)
            .with_alpha_mode(CompositeAlphaMode::Opaque)
            .with_frame_latency(3)
            .with_present_mode_preference(&caps, PresentModePreference::Vsync);

        assert_eq!(config.format, TextureFormat::Rgba8UnormSrgb);
        assert_eq!(config.alpha_mode, CompositeAlphaMode::Opaque);
        assert_eq!(config.desired_maximum_frame_latency, 3);
        assert_eq!(config.present_mode, PresentMode::Mailbox);
    }

    #[test]
    fn builder_chain_order_matters_for_present_mode() {
        let caps = caps_all_modes();

        // Set direct mode first, then preference
        let config1 = SurfaceConfiguration::new(1920, 1080)
            .with_present_mode(PresentMode::Fifo)
            .with_present_mode_preference(&caps, PresentModePreference::LowLatency);
        assert_eq!(config1.present_mode, PresentMode::Immediate);

        // Set preference first, then direct mode
        let config2 = SurfaceConfiguration::new(1920, 1080)
            .with_present_mode_preference(&caps, PresentModePreference::LowLatency)
            .with_present_mode(PresentMode::Fifo);
        assert_eq!(config2.present_mode, PresentMode::Fifo);
    }

    #[test]
    fn from_capabilities_uses_preferred() {
        let caps = caps_all_modes();
        let config = SurfaceConfiguration::from_capabilities(&caps, 800, 600);
        // from_capabilities uses preferred_present_mode which selects Mailbox
        assert_eq!(config.present_mode, PresentMode::Mailbox);
    }
}

// ============================================================================
// 11. Edge Cases Tests
// ============================================================================

mod edge_cases {
    use super::*;

    #[test]
    fn empty_present_modes_low_latency_fallback() {
        let caps = caps_empty();
        // Should return Fifo as ultimate fallback
        assert_eq!(caps.low_latency_present_mode(), PresentMode::Fifo);
    }

    #[test]
    fn empty_present_modes_preferred_fallback() {
        let caps = caps_empty();
        assert_eq!(caps.preferred_present_mode(), PresentMode::Fifo);
    }

    #[test]
    fn empty_present_modes_select_low_latency() {
        let caps = caps_empty();
        assert_eq!(
            caps.select_present_mode(PresentModePreference::LowLatency),
            PresentMode::Fifo
        );
    }

    #[test]
    fn empty_present_modes_select_specific() {
        let caps = caps_empty();
        // Specific mode not available, falls back
        assert_eq!(
            caps.select_present_mode(PresentModePreference::Specific(PresentMode::Immediate)),
            PresentMode::Fifo
        );
    }

    #[test]
    fn single_mode_immediate_only() {
        let caps = caps_with_modes(vec![PresentMode::Immediate]);
        assert_eq!(caps.low_latency_present_mode(), PresentMode::Immediate);
        // preferred_present_mode doesn't prefer Immediate for vsync, but it's all we have
        // Actually looking at the code, it returns first when nothing matches
        let result = caps.preferred_present_mode();
        assert_eq!(result, PresentMode::Immediate);
    }

    #[test]
    fn single_mode_mailbox_only() {
        let caps = caps_with_modes(vec![PresentMode::Mailbox]);
        assert_eq!(caps.preferred_present_mode(), PresentMode::Mailbox);
        assert_eq!(caps.low_latency_present_mode(), PresentMode::Mailbox);
    }

    #[test]
    fn single_mode_fifo_relaxed_only() {
        let caps = caps_with_modes(vec![PresentMode::FifoRelaxed]);
        assert_eq!(caps.preferred_present_mode(), PresentMode::FifoRelaxed);
        assert_eq!(caps.low_latency_present_mode(), PresentMode::FifoRelaxed);
    }

    #[test]
    fn duplicate_modes_in_capabilities() {
        // wgpu shouldn't return duplicates, but test robustness
        let caps = caps_with_modes(vec![
            PresentMode::Fifo,
            PresentMode::Fifo,
            PresentMode::Mailbox,
        ]);
        assert_eq!(caps.preferred_present_mode(), PresentMode::Mailbox);
    }

    #[test]
    fn all_four_standard_modes() {
        let caps = caps_all_modes();
        // Verify all are detected
        assert!(caps.supports_immediate());
        assert!(caps.supports_mailbox());
        assert!(caps.supports_fifo_relaxed());
        assert!(caps.supports_present_mode(PresentMode::Fifo));
    }

    #[test]
    fn caps_with_only_non_standard_modes() {
        // AutoVsync and AutoNoVsync are platform-specific, rarely seen
        // Test with just Fifo which should always be present
        let caps = caps_fifo_only();
        assert!(caps.supports_present_mode(PresentMode::Fifo));
        assert!(!caps.supports_immediate());
    }

    #[test]
    fn select_present_mode_all_preferences_with_empty_caps() {
        let caps = caps_empty();
        // All should return Fifo as fallback
        assert_eq!(
            caps.select_present_mode(PresentModePreference::LowLatency),
            PresentMode::Fifo
        );
        assert_eq!(
            caps.select_present_mode(PresentModePreference::Vsync),
            PresentMode::Fifo
        );
        assert_eq!(
            caps.select_present_mode(PresentModePreference::PowerSaving),
            PresentMode::Fifo
        );
        assert_eq!(
            caps.select_present_mode(PresentModePreference::Adaptive),
            PresentMode::Fifo
        );
    }
}

// ============================================================================
// 12. Tearing Prevention Tests
// ============================================================================

mod tearing_prevention {
    use super::*;

    #[test]
    fn immediate_does_not_prevent_tearing() {
        let info = PresentModeInfo::from_mode(PresentMode::Immediate);
        assert!(!info.prevents_tearing);
    }

    #[test]
    fn mailbox_prevents_tearing() {
        let info = PresentModeInfo::from_mode(PresentMode::Mailbox);
        assert!(info.prevents_tearing);
    }

    #[test]
    fn fifo_prevents_tearing() {
        let info = PresentModeInfo::from_mode(PresentMode::Fifo);
        assert!(info.prevents_tearing);
    }

    #[test]
    fn fifo_relaxed_prevents_tearing() {
        let info = PresentModeInfo::from_mode(PresentMode::FifoRelaxed);
        assert!(info.prevents_tearing);
    }

    #[test]
    fn only_immediate_allows_tearing() {
        let modes = [
            PresentMode::Immediate,
            PresentMode::Mailbox,
            PresentMode::Fifo,
            PresentMode::FifoRelaxed,
        ];

        for mode in &modes {
            let info = PresentModeInfo::from_mode(*mode);
            if *mode == PresentMode::Immediate {
                assert!(!info.prevents_tearing, "Immediate should allow tearing");
            } else {
                assert!(info.prevents_tearing, "{:?} should prevent tearing", mode);
            }
        }
    }
}

// ============================================================================
// 13. Latency Rank Consistency Tests
// ============================================================================

mod latency_rank_consistency {
    use super::*;

    #[test]
    fn latency_ranks_are_unique() {
        let ranks: Vec<u8> = [
            PresentMode::Immediate,
            PresentMode::Mailbox,
            PresentMode::FifoRelaxed,
            PresentMode::Fifo,
        ]
        .iter()
        .map(|m| PresentModeInfo::from_mode(*m).latency_rank)
        .collect();

        // Check all unique
        let mut sorted = ranks.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), ranks.len(), "Latency ranks should be unique");
    }

    #[test]
    fn latency_ranks_are_1_to_4() {
        let modes = [
            PresentMode::Immediate,
            PresentMode::Mailbox,
            PresentMode::FifoRelaxed,
            PresentMode::Fifo,
        ];

        for mode in &modes {
            let rank = PresentModeInfo::from_mode(*mode).latency_rank;
            assert!(rank >= 1 && rank <= 4, "Rank {} out of range 1-4", rank);
        }
    }

    #[test]
    fn latency_rank_1_is_best() {
        // Rank 1 means lowest latency
        let immediate = PresentModeInfo::from_mode(PresentMode::Immediate);
        assert_eq!(immediate.latency_rank, 1);
    }

    #[test]
    fn latency_rank_4_is_worst() {
        // Rank 4 means highest latency
        let fifo = PresentModeInfo::from_mode(PresentMode::Fifo);
        assert_eq!(fifo.latency_rank, 4);
    }

    #[test]
    fn competitive_gaming_threshold_is_rank_2() {
        // Modes with rank <= 2 are considered competitive gaming modes
        let immediate = PresentModeInfo::from_mode(PresentMode::Immediate);
        let mailbox = PresentModeInfo::from_mode(PresentMode::Mailbox);
        let fifo_relaxed = PresentModeInfo::from_mode(PresentMode::FifoRelaxed);
        let fifo = PresentModeInfo::from_mode(PresentMode::Fifo);

        assert!(immediate.is_competitive_gaming_mode());
        assert!(mailbox.is_competitive_gaming_mode());
        assert!(!fifo_relaxed.is_competitive_gaming_mode());
        assert!(!fifo.is_competitive_gaming_mode());
    }
}

// ============================================================================
// 14. Power Efficiency Consistency Tests
// ============================================================================

mod power_efficiency_consistency {
    use super::*;

    #[test]
    fn power_efficient_modes_count() {
        // Only Fifo and FifoRelaxed are power efficient
        let efficient_count: usize = [
            PresentMode::Immediate,
            PresentMode::Mailbox,
            PresentMode::FifoRelaxed,
            PresentMode::Fifo,
        ]
        .iter()
        .filter(|m| PresentModeInfo::from_mode(**m).power_efficient)
        .count();

        assert_eq!(efficient_count, 2);
    }

    #[test]
    fn fifo_modes_are_power_efficient() {
        let fifo = PresentModeInfo::from_mode(PresentMode::Fifo);
        let fifo_relaxed = PresentModeInfo::from_mode(PresentMode::FifoRelaxed);

        assert!(fifo.power_efficient);
        assert!(fifo_relaxed.power_efficient);
    }

    #[test]
    fn non_fifo_modes_are_not_power_efficient() {
        let immediate = PresentModeInfo::from_mode(PresentMode::Immediate);
        let mailbox = PresentModeInfo::from_mode(PresentMode::Mailbox);

        assert!(!immediate.power_efficient);
        assert!(!mailbox.power_efficient);
    }

    #[test]
    fn battery_friendly_equals_power_efficient() {
        let modes = [
            PresentMode::Immediate,
            PresentMode::Mailbox,
            PresentMode::FifoRelaxed,
            PresentMode::Fifo,
        ];

        for mode in &modes {
            let info = PresentModeInfo::from_mode(*mode);
            assert_eq!(
                info.is_battery_friendly(),
                info.power_efficient,
                "{:?} battery_friendly != power_efficient",
                mode
            );
        }
    }
}

// ============================================================================
// 15. Cross-Cutting Integration Tests
// ============================================================================

mod cross_cutting_integration {
    use super::*;

    #[test]
    fn preference_to_info_consistency_low_latency() {
        let caps = caps_all_modes();
        let mode = caps.select_present_mode(PresentModePreference::LowLatency);
        let info = PresentModeInfo::from_mode(mode);

        // Low latency should select a competitive gaming mode
        assert!(info.is_competitive_gaming_mode());
    }

    #[test]
    fn preference_to_info_consistency_power_saving() {
        let caps = caps_all_modes();
        let mode = caps.select_present_mode(PresentModePreference::PowerSaving);
        let info = PresentModeInfo::from_mode(mode);

        // Power saving should select a power efficient mode
        assert!(info.power_efficient);
    }

    #[test]
    fn vsync_preference_prevents_tearing() {
        let caps = caps_all_modes();
        let mode = caps.select_present_mode(PresentModePreference::Vsync);
        let info = PresentModeInfo::from_mode(mode);

        // Vsync should prevent tearing
        assert!(info.prevents_tearing);
    }

    #[test]
    fn adaptive_preference_prevents_tearing() {
        let caps = caps_all_modes();
        let mode = caps.select_present_mode(PresentModePreference::Adaptive);
        let info = PresentModeInfo::from_mode(mode);

        // Adaptive should prevent tearing
        assert!(info.prevents_tearing);
    }

    #[test]
    fn full_pipeline_low_latency() {
        let caps = caps_all_modes();

        // Select mode via preference
        let mode = caps.select_present_mode(PresentModePreference::LowLatency);

        // Build config with preference
        let config = SurfaceConfiguration::new(1920, 1080)
            .with_present_mode_preference(&caps, PresentModePreference::LowLatency);

        // Get info about the mode
        let info = SurfaceCapabilities::describe_present_mode(mode);

        // Verify consistency
        assert_eq!(config.present_mode, mode);
        assert_eq!(info.mode, mode);
        assert_eq!(mode, PresentMode::Immediate);
    }

    #[test]
    fn full_pipeline_vsync() {
        let caps = caps_all_modes();
        let mode = caps.select_present_mode(PresentModePreference::Vsync);
        let config = SurfaceConfiguration::new(1920, 1080)
            .with_present_mode_preference(&caps, PresentModePreference::Vsync);
        let info = SurfaceCapabilities::describe_present_mode(mode);

        assert_eq!(config.present_mode, mode);
        assert_eq!(info.mode, mode);
        assert_eq!(mode, PresentMode::Mailbox);
    }

    #[test]
    fn full_pipeline_power_saving() {
        let caps = caps_all_modes();
        let mode = caps.select_present_mode(PresentModePreference::PowerSaving);
        let config = SurfaceConfiguration::new(1920, 1080)
            .with_present_mode_preference(&caps, PresentModePreference::PowerSaving);
        let info = SurfaceCapabilities::describe_present_mode(mode);

        assert_eq!(config.present_mode, mode);
        assert_eq!(info.mode, mode);
        assert_eq!(mode, PresentMode::Fifo);
        assert!(info.power_efficient);
    }

    #[test]
    fn full_pipeline_adaptive() {
        let caps = caps_all_modes();
        let mode = caps.select_present_mode(PresentModePreference::Adaptive);
        let config = SurfaceConfiguration::new(1920, 1080)
            .with_present_mode_preference(&caps, PresentModePreference::Adaptive);
        let info = SurfaceCapabilities::describe_present_mode(mode);

        assert_eq!(config.present_mode, mode);
        assert_eq!(info.mode, mode);
        assert_eq!(mode, PresentMode::FifoRelaxed);
    }

    #[test]
    fn capability_limited_scenario() {
        // Simulate typical mobile device: only Fifo available
        let caps = caps_fifo_only();

        // All preferences should gracefully degrade to Fifo
        assert_eq!(
            caps.select_present_mode(PresentModePreference::LowLatency),
            PresentMode::Fifo
        );
        assert_eq!(
            caps.select_present_mode(PresentModePreference::Vsync),
            PresentMode::Fifo
        );
        assert_eq!(
            caps.select_present_mode(PresentModePreference::PowerSaving),
            PresentMode::Fifo
        );
        assert_eq!(
            caps.select_present_mode(PresentModePreference::Adaptive),
            PresentMode::Fifo
        );
    }

    #[test]
    fn desktop_gaming_scenario() {
        // Simulate high-end desktop: Immediate + Mailbox + Fifo
        let caps = caps_with_modes(vec![
            PresentMode::Immediate,
            PresentMode::Mailbox,
            PresentMode::Fifo,
        ]);

        // Low latency gaming: Immediate
        assert_eq!(
            caps.select_present_mode(PresentModePreference::LowLatency),
            PresentMode::Immediate
        );

        // Smooth gaming: Mailbox
        assert_eq!(
            caps.select_present_mode(PresentModePreference::Vsync),
            PresentMode::Mailbox
        );

        // Power saving: Fifo
        assert_eq!(
            caps.select_present_mode(PresentModePreference::PowerSaving),
            PresentMode::Fifo
        );

        // Adaptive without FifoRelaxed: falls back to Mailbox
        assert_eq!(
            caps.select_present_mode(PresentModePreference::Adaptive),
            PresentMode::Mailbox
        );
    }
}

// ============================================================================
// 16. Config Validation Tests with Present Modes
// ============================================================================

mod config_validation_present_modes {
    use super::*;

    #[test]
    fn validate_succeeds_with_supported_present_mode() {
        let caps = caps_all_modes();
        let config = SurfaceConfiguration::new(800, 600)
            .with_format(TextureFormat::Bgra8Unorm)
            .with_present_mode(PresentMode::Mailbox)
            .with_alpha_mode(CompositeAlphaMode::Auto);

        assert!(config.validate(&caps).is_ok());
    }

    #[test]
    fn validate_fails_with_unsupported_present_mode() {
        let caps = caps_fifo_only();
        let config = SurfaceConfiguration::new(800, 600)
            .with_format(TextureFormat::Bgra8Unorm)
            .with_present_mode(PresentMode::Immediate)
            .with_alpha_mode(CompositeAlphaMode::Auto);

        let result = config.validate(&caps);
        assert!(result.is_err());
    }

    #[test]
    fn validate_error_message_contains_present_mode() {
        let caps = caps_fifo_only();
        let config = SurfaceConfiguration::new(800, 600)
            .with_format(TextureFormat::Bgra8Unorm)
            .with_present_mode(PresentMode::Mailbox)
            .with_alpha_mode(CompositeAlphaMode::Auto);

        let err = config.validate(&caps).unwrap_err();
        let msg = format!("{}", err);
        assert!(msg.contains("present mode") || msg.contains("PresentMode"));
    }

    #[test]
    fn preference_builder_always_produces_valid_config() {
        let caps = caps_fifo_only();

        // Using preference should always produce a valid config
        let config = SurfaceConfiguration::new(800, 600)
            .with_format(TextureFormat::Bgra8Unorm)
            .with_present_mode_preference(&caps, PresentModePreference::LowLatency)
            .with_alpha_mode(CompositeAlphaMode::Auto);

        // This should validate because the preference builder selected a supported mode
        assert!(config.validate(&caps).is_ok());
        assert_eq!(config.present_mode, PresentMode::Fifo);
    }

    #[test]
    fn all_preferences_produce_valid_configs() {
        let caps = caps_with_modes(vec![PresentMode::Mailbox, PresentMode::Fifo]);
        let preferences = [
            PresentModePreference::LowLatency,
            PresentModePreference::Vsync,
            PresentModePreference::PowerSaving,
            PresentModePreference::Adaptive,
            PresentModePreference::Specific(PresentMode::Immediate), // Not available, will fallback
        ];

        for pref in &preferences {
            let config = SurfaceConfiguration::new(800, 600)
                .with_format(TextureFormat::Bgra8Unorm)
                .with_present_mode_preference(&caps, *pref)
                .with_alpha_mode(CompositeAlphaMode::Auto);

            assert!(
                config.validate(&caps).is_ok(),
                "Preference {:?} produced invalid config",
                pref
            );
        }
    }
}

// ============================================================================
// 17. to_wgpu Conversion Tests
// ============================================================================

mod to_wgpu_conversion {
    use super::*;

    #[test]
    fn to_wgpu_preserves_present_mode() {
        let config = SurfaceConfiguration::new(1920, 1080)
            .with_present_mode(PresentMode::Mailbox);
        let wgpu_config = config.to_wgpu();
        assert_eq!(wgpu_config.present_mode, PresentMode::Mailbox);
    }

    #[test]
    fn to_wgpu_with_all_present_modes() {
        let modes = [
            PresentMode::Immediate,
            PresentMode::Mailbox,
            PresentMode::FifoRelaxed,
            PresentMode::Fifo,
        ];

        for mode in &modes {
            let config = SurfaceConfiguration::new(1920, 1080).with_present_mode(*mode);
            let wgpu_config = config.to_wgpu();
            assert_eq!(wgpu_config.present_mode, *mode);
        }
    }

    #[test]
    fn to_wgpu_after_preference_selection() {
        let caps = caps_all_modes();
        let config = SurfaceConfiguration::new(1920, 1080)
            .with_present_mode_preference(&caps, PresentModePreference::LowLatency);
        let wgpu_config = config.to_wgpu();
        assert_eq!(wgpu_config.present_mode, PresentMode::Immediate);
    }
}
