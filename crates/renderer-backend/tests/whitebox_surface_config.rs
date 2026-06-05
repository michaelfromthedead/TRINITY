//! Whitebox tests for T-WGPU-P7.1.5 (Surface Configuration)
//!
//! WHITEBOX discipline: These tests have FULL ACCESS to the implementation.
//! They exercise internal code paths, branch conditions, and edge cases
//! that are not visible through the public contract alone.
//!
//! Implementation under test: crates/renderer-backend/src/presentation/surface.rs
//!   - AlphaModePreference: Opaque, PreMultiplied, PostMultiplied, Inherit, Auto
//!   - AlphaModePreference methods: description(), requires_alpha(), to_concrete_mode()
//!   - SurfaceCapabilities: supports_alpha_mode(), select_alpha_mode(preference)
//!   - SurfaceConfiguration: from_window_size(), with_alpha_mode(), with_alpha_mode_preference()
//!   - SurfaceConfiguration: with_view_formats(), with_srgb_view_format(), has_srgb_view_format()
//!   - SurfaceConfiguration: srgb_format(), linear_format(), configure()
//!   - Helper functions: get_srgb_companion_format(), are_srgb_companions()
//!   - TrinitySurface: alpha_mode(), view_formats(), has_srgb_view(), is_configured(), dimensions()
//!
//! Test Categories:
//!   1. AlphaModePreference selection priority (Auto: PreMultiplied > PostMultiplied > Opaque > Inherit)
//!   2. Alpha mode capability intersection
//!   3. sRGB/linear companion format pairing
//!   4. view_formats accumulation (multiple calls)
//!   5. Configuration validation with alpha modes
//!   6. Edge cases: empty capabilities, single alpha mode

use renderer_backend::presentation::{
    AlphaModePreference, PlatformTarget, SurfaceCapabilities, SurfaceConfiguration, SurfaceError,
    FormatCategory, PresentModePreference, PresentModeInfo,
    get_srgb_companion_format, are_srgb_companions,
};
use wgpu::{CompositeAlphaMode, PresentMode, TextureFormat, TextureUsages};

// ============================================================================
// Category 1: AlphaModePreference Selection Priority Tests
// ============================================================================

mod alpha_mode_preference_priority {
    use super::*;

    // -------------------------------------------------------------------------
    // 1.1 AlphaModePreference::Auto priority: Opaque first (best performance)
    // -------------------------------------------------------------------------

    #[test]
    fn auto_selects_opaque_when_available() {
        let caps = SurfaceCapabilities {
            formats: vec![TextureFormat::Bgra8Unorm],
            present_modes: vec![PresentMode::Fifo],
            alpha_modes: vec![
                CompositeAlphaMode::PreMultiplied,
                CompositeAlphaMode::Opaque,
                CompositeAlphaMode::PostMultiplied,
            ],
            usages: TextureUsages::RENDER_ATTACHMENT,
        };
        // Auto should prefer Opaque for best performance
        assert_eq!(caps.select_alpha_mode(AlphaModePreference::Auto), CompositeAlphaMode::Opaque);
    }

    #[test]
    fn auto_falls_back_to_first_when_opaque_unavailable() {
        let caps = SurfaceCapabilities {
            formats: vec![TextureFormat::Bgra8Unorm],
            present_modes: vec![PresentMode::Fifo],
            alpha_modes: vec![
                CompositeAlphaMode::PreMultiplied,
                CompositeAlphaMode::PostMultiplied,
            ],
            usages: TextureUsages::RENDER_ATTACHMENT,
        };
        // Should fall back to first available (PreMultiplied)
        assert_eq!(caps.select_alpha_mode(AlphaModePreference::Auto), CompositeAlphaMode::PreMultiplied);
    }

    #[test]
    fn auto_with_single_mode_returns_that_mode() {
        let caps = SurfaceCapabilities {
            formats: vec![TextureFormat::Bgra8Unorm],
            present_modes: vec![PresentMode::Fifo],
            alpha_modes: vec![CompositeAlphaMode::Inherit],
            usages: TextureUsages::RENDER_ATTACHMENT,
        };
        assert_eq!(caps.select_alpha_mode(AlphaModePreference::Auto), CompositeAlphaMode::Inherit);
    }

    #[test]
    fn auto_with_only_auto_mode_returns_auto() {
        let caps = SurfaceCapabilities {
            formats: vec![TextureFormat::Bgra8Unorm],
            present_modes: vec![PresentMode::Fifo],
            alpha_modes: vec![CompositeAlphaMode::Auto],
            usages: TextureUsages::RENDER_ATTACHMENT,
        };
        assert_eq!(caps.select_alpha_mode(AlphaModePreference::Auto), CompositeAlphaMode::Auto);
    }

    // -------------------------------------------------------------------------
    // 1.2 AlphaModePreference::Opaque direct selection
    // -------------------------------------------------------------------------

    #[test]
    fn opaque_preference_selects_opaque_when_available() {
        let caps = SurfaceCapabilities {
            formats: vec![TextureFormat::Bgra8Unorm],
            present_modes: vec![PresentMode::Fifo],
            alpha_modes: vec![
                CompositeAlphaMode::PreMultiplied,
                CompositeAlphaMode::Opaque,
            ],
            usages: TextureUsages::RENDER_ATTACHMENT,
        };
        assert_eq!(caps.select_alpha_mode(AlphaModePreference::Opaque), CompositeAlphaMode::Opaque);
    }

    #[test]
    fn opaque_preference_falls_back_when_opaque_unavailable() {
        let caps = SurfaceCapabilities {
            formats: vec![TextureFormat::Bgra8Unorm],
            present_modes: vec![PresentMode::Fifo],
            alpha_modes: vec![
                CompositeAlphaMode::PreMultiplied,
                CompositeAlphaMode::PostMultiplied,
            ],
            usages: TextureUsages::RENDER_ATTACHMENT,
        };
        // Falls back to preferred_alpha_mode(), which returns first available
        assert_eq!(caps.select_alpha_mode(AlphaModePreference::Opaque), CompositeAlphaMode::PreMultiplied);
    }

    // -------------------------------------------------------------------------
    // 1.3 AlphaModePreference::PreMultiplied selection with fallback chain
    // -------------------------------------------------------------------------

    #[test]
    fn premultiplied_preference_selects_premultiplied_when_available() {
        let caps = SurfaceCapabilities {
            formats: vec![TextureFormat::Bgra8Unorm],
            present_modes: vec![PresentMode::Fifo],
            alpha_modes: vec![
                CompositeAlphaMode::Opaque,
                CompositeAlphaMode::PreMultiplied,
            ],
            usages: TextureUsages::RENDER_ATTACHMENT,
        };
        assert_eq!(caps.select_alpha_mode(AlphaModePreference::PreMultiplied), CompositeAlphaMode::PreMultiplied);
    }

    #[test]
    fn premultiplied_falls_back_to_postmultiplied() {
        let caps = SurfaceCapabilities {
            formats: vec![TextureFormat::Bgra8Unorm],
            present_modes: vec![PresentMode::Fifo],
            alpha_modes: vec![
                CompositeAlphaMode::Opaque,
                CompositeAlphaMode::PostMultiplied,
            ],
            usages: TextureUsages::RENDER_ATTACHMENT,
        };
        // PreMultiplied not available, falls back to PostMultiplied
        assert_eq!(caps.select_alpha_mode(AlphaModePreference::PreMultiplied), CompositeAlphaMode::PostMultiplied);
    }

    #[test]
    fn premultiplied_falls_back_to_opaque_when_no_multiplied_modes() {
        let caps = SurfaceCapabilities {
            formats: vec![TextureFormat::Bgra8Unorm],
            present_modes: vec![PresentMode::Fifo],
            alpha_modes: vec![CompositeAlphaMode::Opaque],
            usages: TextureUsages::RENDER_ATTACHMENT,
        };
        // Neither PreMultiplied nor PostMultiplied available
        assert_eq!(caps.select_alpha_mode(AlphaModePreference::PreMultiplied), CompositeAlphaMode::Opaque);
    }

    // -------------------------------------------------------------------------
    // 1.4 AlphaModePreference::PostMultiplied selection with fallback chain
    // -------------------------------------------------------------------------

    #[test]
    fn postmultiplied_preference_selects_postmultiplied_when_available() {
        let caps = SurfaceCapabilities {
            formats: vec![TextureFormat::Bgra8Unorm],
            present_modes: vec![PresentMode::Fifo],
            alpha_modes: vec![
                CompositeAlphaMode::Opaque,
                CompositeAlphaMode::PostMultiplied,
            ],
            usages: TextureUsages::RENDER_ATTACHMENT,
        };
        assert_eq!(caps.select_alpha_mode(AlphaModePreference::PostMultiplied), CompositeAlphaMode::PostMultiplied);
    }

    #[test]
    fn postmultiplied_falls_back_to_premultiplied() {
        let caps = SurfaceCapabilities {
            formats: vec![TextureFormat::Bgra8Unorm],
            present_modes: vec![PresentMode::Fifo],
            alpha_modes: vec![
                CompositeAlphaMode::Opaque,
                CompositeAlphaMode::PreMultiplied,
            ],
            usages: TextureUsages::RENDER_ATTACHMENT,
        };
        // PostMultiplied not available, falls back to PreMultiplied
        assert_eq!(caps.select_alpha_mode(AlphaModePreference::PostMultiplied), CompositeAlphaMode::PreMultiplied);
    }

    #[test]
    fn postmultiplied_falls_back_to_opaque_when_no_multiplied_modes() {
        let caps = SurfaceCapabilities {
            formats: vec![TextureFormat::Bgra8Unorm],
            present_modes: vec![PresentMode::Fifo],
            alpha_modes: vec![CompositeAlphaMode::Opaque],
            usages: TextureUsages::RENDER_ATTACHMENT,
        };
        assert_eq!(caps.select_alpha_mode(AlphaModePreference::PostMultiplied), CompositeAlphaMode::Opaque);
    }

    // -------------------------------------------------------------------------
    // 1.5 AlphaModePreference::Inherit selection
    // -------------------------------------------------------------------------

    #[test]
    fn inherit_preference_selects_inherit_when_available() {
        let caps = SurfaceCapabilities {
            formats: vec![TextureFormat::Bgra8Unorm],
            present_modes: vec![PresentMode::Fifo],
            alpha_modes: vec![
                CompositeAlphaMode::Opaque,
                CompositeAlphaMode::Inherit,
            ],
            usages: TextureUsages::RENDER_ATTACHMENT,
        };
        assert_eq!(caps.select_alpha_mode(AlphaModePreference::Inherit), CompositeAlphaMode::Inherit);
    }

    #[test]
    fn inherit_preference_falls_back_when_unavailable() {
        let caps = SurfaceCapabilities {
            formats: vec![TextureFormat::Bgra8Unorm],
            present_modes: vec![PresentMode::Fifo],
            alpha_modes: vec![
                CompositeAlphaMode::Opaque,
                CompositeAlphaMode::PreMultiplied,
            ],
            usages: TextureUsages::RENDER_ATTACHMENT,
        };
        // Inherit not available, falls back to preferred (Opaque)
        assert_eq!(caps.select_alpha_mode(AlphaModePreference::Inherit), CompositeAlphaMode::Opaque);
    }

    // -------------------------------------------------------------------------
    // 1.6 Complete priority chain verification
    // -------------------------------------------------------------------------

    #[test]
    fn all_alpha_modes_available_respects_each_preference() {
        let caps = SurfaceCapabilities {
            formats: vec![TextureFormat::Bgra8Unorm],
            present_modes: vec![PresentMode::Fifo],
            alpha_modes: vec![
                CompositeAlphaMode::Auto,
                CompositeAlphaMode::Opaque,
                CompositeAlphaMode::PreMultiplied,
                CompositeAlphaMode::PostMultiplied,
                CompositeAlphaMode::Inherit,
            ],
            usages: TextureUsages::RENDER_ATTACHMENT,
        };

        // Each preference should select its own mode
        assert_eq!(caps.select_alpha_mode(AlphaModePreference::Opaque), CompositeAlphaMode::Opaque);
        assert_eq!(caps.select_alpha_mode(AlphaModePreference::PreMultiplied), CompositeAlphaMode::PreMultiplied);
        assert_eq!(caps.select_alpha_mode(AlphaModePreference::PostMultiplied), CompositeAlphaMode::PostMultiplied);
        assert_eq!(caps.select_alpha_mode(AlphaModePreference::Inherit), CompositeAlphaMode::Inherit);
        // Auto prefers Opaque for performance
        assert_eq!(caps.select_alpha_mode(AlphaModePreference::Auto), CompositeAlphaMode::Opaque);
    }
}

// ============================================================================
// Category 2: Alpha Mode Capability Intersection Tests
// ============================================================================

mod alpha_mode_capability_intersection {
    use super::*;

    // -------------------------------------------------------------------------
    // 2.1 supports_alpha_mode() basic checks
    // -------------------------------------------------------------------------

    #[test]
    fn supports_alpha_mode_returns_true_when_present() {
        let caps = SurfaceCapabilities {
            formats: vec![TextureFormat::Bgra8Unorm],
            present_modes: vec![PresentMode::Fifo],
            alpha_modes: vec![CompositeAlphaMode::Opaque, CompositeAlphaMode::PreMultiplied],
            usages: TextureUsages::RENDER_ATTACHMENT,
        };
        assert!(caps.supports_alpha_mode(CompositeAlphaMode::Opaque));
        assert!(caps.supports_alpha_mode(CompositeAlphaMode::PreMultiplied));
    }

    #[test]
    fn supports_alpha_mode_returns_false_when_absent() {
        let caps = SurfaceCapabilities {
            formats: vec![TextureFormat::Bgra8Unorm],
            present_modes: vec![PresentMode::Fifo],
            alpha_modes: vec![CompositeAlphaMode::Opaque],
            usages: TextureUsages::RENDER_ATTACHMENT,
        };
        assert!(!caps.supports_alpha_mode(CompositeAlphaMode::PreMultiplied));
        assert!(!caps.supports_alpha_mode(CompositeAlphaMode::PostMultiplied));
        assert!(!caps.supports_alpha_mode(CompositeAlphaMode::Inherit));
    }

    #[test]
    fn supports_alpha_mode_auto_vs_explicit() {
        let caps = SurfaceCapabilities {
            formats: vec![TextureFormat::Bgra8Unorm],
            present_modes: vec![PresentMode::Fifo],
            alpha_modes: vec![CompositeAlphaMode::Auto],
            usages: TextureUsages::RENDER_ATTACHMENT,
        };
        // Auto mode explicitly in list
        assert!(caps.supports_alpha_mode(CompositeAlphaMode::Auto));
        // But explicit modes not available
        assert!(!caps.supports_alpha_mode(CompositeAlphaMode::Opaque));
    }

    // -------------------------------------------------------------------------
    // 2.2 Alpha mode selection with various capability sets
    // -------------------------------------------------------------------------

    #[test]
    fn select_with_only_opaque_capability() {
        let caps = SurfaceCapabilities {
            formats: vec![TextureFormat::Bgra8Unorm],
            present_modes: vec![PresentMode::Fifo],
            alpha_modes: vec![CompositeAlphaMode::Opaque],
            usages: TextureUsages::RENDER_ATTACHMENT,
        };
        // All preferences should fall back to Opaque
        assert_eq!(caps.select_alpha_mode(AlphaModePreference::Opaque), CompositeAlphaMode::Opaque);
        assert_eq!(caps.select_alpha_mode(AlphaModePreference::PreMultiplied), CompositeAlphaMode::Opaque);
        assert_eq!(caps.select_alpha_mode(AlphaModePreference::PostMultiplied), CompositeAlphaMode::Opaque);
        assert_eq!(caps.select_alpha_mode(AlphaModePreference::Inherit), CompositeAlphaMode::Opaque);
        assert_eq!(caps.select_alpha_mode(AlphaModePreference::Auto), CompositeAlphaMode::Opaque);
    }

    #[test]
    fn select_with_only_premultiplied_capability() {
        let caps = SurfaceCapabilities {
            formats: vec![TextureFormat::Bgra8Unorm],
            present_modes: vec![PresentMode::Fifo],
            alpha_modes: vec![CompositeAlphaMode::PreMultiplied],
            usages: TextureUsages::RENDER_ATTACHMENT,
        };
        assert_eq!(caps.select_alpha_mode(AlphaModePreference::PreMultiplied), CompositeAlphaMode::PreMultiplied);
        // Others fall back
        assert_eq!(caps.select_alpha_mode(AlphaModePreference::Opaque), CompositeAlphaMode::PreMultiplied);
        assert_eq!(caps.select_alpha_mode(AlphaModePreference::Auto), CompositeAlphaMode::PreMultiplied);
    }

    #[test]
    fn select_with_only_postmultiplied_capability() {
        let caps = SurfaceCapabilities {
            formats: vec![TextureFormat::Bgra8Unorm],
            present_modes: vec![PresentMode::Fifo],
            alpha_modes: vec![CompositeAlphaMode::PostMultiplied],
            usages: TextureUsages::RENDER_ATTACHMENT,
        };
        assert_eq!(caps.select_alpha_mode(AlphaModePreference::PostMultiplied), CompositeAlphaMode::PostMultiplied);
        // PreMultiplied fallback finds PostMultiplied
        assert_eq!(caps.select_alpha_mode(AlphaModePreference::PreMultiplied), CompositeAlphaMode::PostMultiplied);
    }

    #[test]
    fn select_with_only_inherit_capability() {
        let caps = SurfaceCapabilities {
            formats: vec![TextureFormat::Bgra8Unorm],
            present_modes: vec![PresentMode::Fifo],
            alpha_modes: vec![CompositeAlphaMode::Inherit],
            usages: TextureUsages::RENDER_ATTACHMENT,
        };
        assert_eq!(caps.select_alpha_mode(AlphaModePreference::Inherit), CompositeAlphaMode::Inherit);
        // Others fall back to first available
        assert_eq!(caps.select_alpha_mode(AlphaModePreference::Auto), CompositeAlphaMode::Inherit);
    }

    // -------------------------------------------------------------------------
    // 2.3 Intersection of transparency modes
    // -------------------------------------------------------------------------

    #[test]
    fn both_multiplied_modes_available_premultiplied_prefers_itself() {
        let caps = SurfaceCapabilities {
            formats: vec![TextureFormat::Bgra8Unorm],
            present_modes: vec![PresentMode::Fifo],
            alpha_modes: vec![
                CompositeAlphaMode::PreMultiplied,
                CompositeAlphaMode::PostMultiplied,
            ],
            usages: TextureUsages::RENDER_ATTACHMENT,
        };
        assert_eq!(caps.select_alpha_mode(AlphaModePreference::PreMultiplied), CompositeAlphaMode::PreMultiplied);
        assert_eq!(caps.select_alpha_mode(AlphaModePreference::PostMultiplied), CompositeAlphaMode::PostMultiplied);
    }

    #[test]
    fn opaque_and_inherit_only() {
        let caps = SurfaceCapabilities {
            formats: vec![TextureFormat::Bgra8Unorm],
            present_modes: vec![PresentMode::Fifo],
            alpha_modes: vec![
                CompositeAlphaMode::Opaque,
                CompositeAlphaMode::Inherit,
            ],
            usages: TextureUsages::RENDER_ATTACHMENT,
        };
        assert_eq!(caps.select_alpha_mode(AlphaModePreference::Opaque), CompositeAlphaMode::Opaque);
        assert_eq!(caps.select_alpha_mode(AlphaModePreference::Inherit), CompositeAlphaMode::Inherit);
        // PreMultiplied/PostMultiplied fall back to Opaque (preferred)
        assert_eq!(caps.select_alpha_mode(AlphaModePreference::PreMultiplied), CompositeAlphaMode::Opaque);
        assert_eq!(caps.select_alpha_mode(AlphaModePreference::PostMultiplied), CompositeAlphaMode::Opaque);
    }
}

// ============================================================================
// Category 3: sRGB/Linear Companion Format Pairing Tests
// ============================================================================

mod srgb_companion_format_pairing {
    use super::*;

    // -------------------------------------------------------------------------
    // 3.1 get_srgb_companion_format() for all known pairs
    // -------------------------------------------------------------------------

    #[test]
    fn bgra8_unorm_companion_is_bgra8_srgb() {
        assert_eq!(
            get_srgb_companion_format(TextureFormat::Bgra8Unorm),
            Some(TextureFormat::Bgra8UnormSrgb)
        );
    }

    #[test]
    fn bgra8_srgb_companion_is_bgra8_unorm() {
        assert_eq!(
            get_srgb_companion_format(TextureFormat::Bgra8UnormSrgb),
            Some(TextureFormat::Bgra8Unorm)
        );
    }

    #[test]
    fn rgba8_unorm_companion_is_rgba8_srgb() {
        assert_eq!(
            get_srgb_companion_format(TextureFormat::Rgba8Unorm),
            Some(TextureFormat::Rgba8UnormSrgb)
        );
    }

    #[test]
    fn rgba8_srgb_companion_is_rgba8_unorm() {
        assert_eq!(
            get_srgb_companion_format(TextureFormat::Rgba8UnormSrgb),
            Some(TextureFormat::Rgba8Unorm)
        );
    }

    // -------------------------------------------------------------------------
    // 3.2 Formats without companions
    // -------------------------------------------------------------------------

    #[test]
    fn rgba16_float_has_no_companion() {
        assert_eq!(get_srgb_companion_format(TextureFormat::Rgba16Float), None);
    }

    #[test]
    fn depth32_float_has_no_companion() {
        assert_eq!(get_srgb_companion_format(TextureFormat::Depth32Float), None);
    }

    #[test]
    fn r8_unorm_has_no_companion() {
        assert_eq!(get_srgb_companion_format(TextureFormat::R8Unorm), None);
    }

    #[test]
    fn rg11b10_float_has_no_companion() {
        assert_eq!(get_srgb_companion_format(TextureFormat::Rg11b10Float), None);
    }

    #[test]
    fn rgb10a2_unorm_has_no_companion() {
        assert_eq!(get_srgb_companion_format(TextureFormat::Rgb10a2Unorm), None);
    }

    // -------------------------------------------------------------------------
    // 3.3 are_srgb_companions() symmetry tests
    // -------------------------------------------------------------------------

    #[test]
    fn are_companions_bgra_pair_both_directions() {
        assert!(are_srgb_companions(TextureFormat::Bgra8Unorm, TextureFormat::Bgra8UnormSrgb));
        assert!(are_srgb_companions(TextureFormat::Bgra8UnormSrgb, TextureFormat::Bgra8Unorm));
    }

    #[test]
    fn are_companions_rgba_pair_both_directions() {
        assert!(are_srgb_companions(TextureFormat::Rgba8Unorm, TextureFormat::Rgba8UnormSrgb));
        assert!(are_srgb_companions(TextureFormat::Rgba8UnormSrgb, TextureFormat::Rgba8Unorm));
    }

    #[test]
    fn are_not_companions_different_families() {
        // Bgra and Rgba are not companions of each other
        assert!(!are_srgb_companions(TextureFormat::Bgra8Unorm, TextureFormat::Rgba8Unorm));
        assert!(!are_srgb_companions(TextureFormat::Bgra8UnormSrgb, TextureFormat::Rgba8UnormSrgb));
    }

    #[test]
    fn are_not_companions_same_format() {
        // A format is not its own companion
        assert!(!are_srgb_companions(TextureFormat::Bgra8Unorm, TextureFormat::Bgra8Unorm));
        assert!(!are_srgb_companions(TextureFormat::Rgba8UnormSrgb, TextureFormat::Rgba8UnormSrgb));
    }

    #[test]
    fn are_not_companions_hdr_formats() {
        assert!(!are_srgb_companions(TextureFormat::Rgba16Float, TextureFormat::Rg11b10Float));
        assert!(!are_srgb_companions(TextureFormat::Rgba16Float, TextureFormat::Rgb10a2Unorm));
    }

    // -------------------------------------------------------------------------
    // 3.4 Companion retrieval exhaustiveness
    // -------------------------------------------------------------------------

    #[test]
    fn all_8bit_color_formats_have_companions() {
        let formats_with_companions = [
            TextureFormat::Bgra8Unorm,
            TextureFormat::Bgra8UnormSrgb,
            TextureFormat::Rgba8Unorm,
            TextureFormat::Rgba8UnormSrgb,
        ];
        for format in formats_with_companions {
            assert!(
                get_srgb_companion_format(format).is_some(),
                "{:?} should have a companion",
                format
            );
        }
    }

    #[test]
    fn companion_of_companion_is_original() {
        let original = TextureFormat::Bgra8Unorm;
        let companion = get_srgb_companion_format(original).unwrap();
        let back_to_original = get_srgb_companion_format(companion).unwrap();
        assert_eq!(original, back_to_original);
    }
}

// ============================================================================
// Category 4: view_formats Accumulation Tests
// ============================================================================

mod view_formats_accumulation {
    use super::*;

    // -------------------------------------------------------------------------
    // 4.1 with_view_formats() basic behavior
    // -------------------------------------------------------------------------

    #[test]
    fn view_formats_empty_by_default() {
        let config = SurfaceConfiguration::new(1920, 1080);
        assert!(config.view_formats.is_empty());
    }

    #[test]
    fn with_view_formats_sets_formats() {
        let config = SurfaceConfiguration::new(1920, 1080)
            .with_view_formats(&[TextureFormat::Bgra8UnormSrgb]);
        assert_eq!(config.view_formats.len(), 1);
        assert_eq!(config.view_formats[0], TextureFormat::Bgra8UnormSrgb);
    }

    #[test]
    fn with_view_formats_multiple_formats() {
        let config = SurfaceConfiguration::new(1920, 1080)
            .with_view_formats(&[
                TextureFormat::Bgra8UnormSrgb,
                TextureFormat::Rgba8UnormSrgb,
            ]);
        assert_eq!(config.view_formats.len(), 2);
        assert!(config.view_formats.contains(&TextureFormat::Bgra8UnormSrgb));
        assert!(config.view_formats.contains(&TextureFormat::Rgba8UnormSrgb));
    }

    #[test]
    fn with_view_formats_replaces_previous() {
        let config = SurfaceConfiguration::new(1920, 1080)
            .with_view_formats(&[TextureFormat::Bgra8UnormSrgb])
            .with_view_formats(&[TextureFormat::Rgba8UnormSrgb]);
        // Second call replaces, not appends
        assert_eq!(config.view_formats.len(), 1);
        assert_eq!(config.view_formats[0], TextureFormat::Rgba8UnormSrgb);
    }

    #[test]
    fn with_view_formats_empty_clears() {
        let config = SurfaceConfiguration::new(1920, 1080)
            .with_view_formats(&[TextureFormat::Bgra8UnormSrgb])
            .with_view_formats(&[]);
        assert!(config.view_formats.is_empty());
    }

    // -------------------------------------------------------------------------
    // 4.2 with_srgb_view_format() auto-companion behavior
    // -------------------------------------------------------------------------

    #[test]
    fn srgb_view_format_adds_srgb_companion_for_linear_base() {
        let config = SurfaceConfiguration::new(1920, 1080)
            .with_format(TextureFormat::Bgra8Unorm)
            .with_srgb_view_format();
        assert!(config.view_formats.contains(&TextureFormat::Bgra8UnormSrgb));
    }

    #[test]
    fn srgb_view_format_adds_linear_companion_for_srgb_base() {
        let config = SurfaceConfiguration::new(1920, 1080)
            .with_format(TextureFormat::Bgra8UnormSrgb)
            .with_srgb_view_format();
        assert!(config.view_formats.contains(&TextureFormat::Bgra8Unorm));
    }

    #[test]
    fn srgb_view_format_rgba_linear_base() {
        let config = SurfaceConfiguration::new(1920, 1080)
            .with_format(TextureFormat::Rgba8Unorm)
            .with_srgb_view_format();
        assert!(config.view_formats.contains(&TextureFormat::Rgba8UnormSrgb));
    }

    #[test]
    fn srgb_view_format_rgba_srgb_base() {
        let config = SurfaceConfiguration::new(1920, 1080)
            .with_format(TextureFormat::Rgba8UnormSrgb)
            .with_srgb_view_format();
        assert!(config.view_formats.contains(&TextureFormat::Rgba8Unorm));
    }

    #[test]
    fn srgb_view_format_no_op_for_hdr_format() {
        let config = SurfaceConfiguration::new(1920, 1080)
            .with_format(TextureFormat::Rgba16Float)
            .with_srgb_view_format();
        assert!(config.view_formats.is_empty());
    }

    #[test]
    fn srgb_view_format_no_duplicate_on_multiple_calls() {
        let config = SurfaceConfiguration::new(1920, 1080)
            .with_format(TextureFormat::Bgra8Unorm)
            .with_srgb_view_format()
            .with_srgb_view_format()
            .with_srgb_view_format();
        // Should not duplicate
        assert_eq!(config.view_formats.len(), 1);
        assert_eq!(config.view_formats[0], TextureFormat::Bgra8UnormSrgb);
    }

    // -------------------------------------------------------------------------
    // 4.3 Combining with_view_formats and with_srgb_view_format
    // -------------------------------------------------------------------------

    #[test]
    fn view_formats_then_srgb_view_format_replaces() {
        let config = SurfaceConfiguration::new(1920, 1080)
            .with_format(TextureFormat::Bgra8Unorm)
            .with_view_formats(&[TextureFormat::Rgba8UnormSrgb])
            .with_srgb_view_format();
        // with_srgb_view_format appends, doesn't replace
        // The Rgba8UnormSrgb from with_view_formats is replaced
        // But then with_srgb_view_format adds Bgra8UnormSrgb
        // Actually: with_view_formats replaces, with_srgb_view_format appends
        // Wait, re-reading code: with_view_formats() sets the vec
        // with_srgb_view_format() pushes to existing vec
        // So: vec becomes [Rgba8UnormSrgb], then push Bgra8UnormSrgb
        // Let me re-read... No, with_view_formats sets self.view_formats = formats.to_vec()
        // So [Rgba8UnormSrgb] is the vec, then with_srgb_view_format adds Bgra8UnormSrgb
        // Result should be [Rgba8UnormSrgb, Bgra8UnormSrgb]
        assert!(config.view_formats.contains(&TextureFormat::Rgba8UnormSrgb));
        assert!(config.view_formats.contains(&TextureFormat::Bgra8UnormSrgb));
    }

    #[test]
    fn srgb_view_format_then_view_formats_replaces() {
        let config = SurfaceConfiguration::new(1920, 1080)
            .with_format(TextureFormat::Bgra8Unorm)
            .with_srgb_view_format()  // Adds Bgra8UnormSrgb
            .with_view_formats(&[TextureFormat::Rgba8UnormSrgb]);  // Replaces
        assert!(!config.view_formats.contains(&TextureFormat::Bgra8UnormSrgb));
        assert!(config.view_formats.contains(&TextureFormat::Rgba8UnormSrgb));
        assert_eq!(config.view_formats.len(), 1);
    }

    // -------------------------------------------------------------------------
    // 4.4 has_srgb_view_format() detection
    // -------------------------------------------------------------------------

    #[test]
    fn has_srgb_view_format_detects_bgra_srgb() {
        let config = SurfaceConfiguration::new(1920, 1080)
            .with_view_formats(&[TextureFormat::Bgra8UnormSrgb]);
        assert!(config.has_srgb_view_format());
    }

    #[test]
    fn has_srgb_view_format_detects_rgba_srgb() {
        let config = SurfaceConfiguration::new(1920, 1080)
            .with_view_formats(&[TextureFormat::Rgba8UnormSrgb]);
        assert!(config.has_srgb_view_format());
    }

    #[test]
    fn has_srgb_view_format_false_for_linear_only() {
        let config = SurfaceConfiguration::new(1920, 1080)
            .with_view_formats(&[TextureFormat::Bgra8Unorm, TextureFormat::Rgba8Unorm]);
        assert!(!config.has_srgb_view_format());
    }

    #[test]
    fn has_srgb_view_format_false_for_empty() {
        let config = SurfaceConfiguration::new(1920, 1080);
        assert!(!config.has_srgb_view_format());
    }

    #[test]
    fn has_srgb_view_format_false_for_hdr() {
        let config = SurfaceConfiguration::new(1920, 1080)
            .with_view_formats(&[TextureFormat::Rgba16Float]);
        assert!(!config.has_srgb_view_format());
    }

    // -------------------------------------------------------------------------
    // 4.5 srgb_format() and linear_format() accessors
    // -------------------------------------------------------------------------

    #[test]
    fn srgb_format_returns_main_when_main_is_srgb() {
        let config = SurfaceConfiguration::new(1920, 1080)
            .with_format(TextureFormat::Bgra8UnormSrgb);
        assert_eq!(config.srgb_format(), Some(TextureFormat::Bgra8UnormSrgb));
    }

    #[test]
    fn srgb_format_returns_view_format_when_main_is_linear() {
        let config = SurfaceConfiguration::new(1920, 1080)
            .with_format(TextureFormat::Bgra8Unorm)
            .with_srgb_view_format();
        assert_eq!(config.srgb_format(), Some(TextureFormat::Bgra8UnormSrgb));
    }

    #[test]
    fn srgb_format_none_when_no_srgb_available() {
        let config = SurfaceConfiguration::new(1920, 1080)
            .with_format(TextureFormat::Rgba16Float);
        assert_eq!(config.srgb_format(), None);
    }

    #[test]
    fn linear_format_returns_main_when_main_is_linear() {
        let config = SurfaceConfiguration::new(1920, 1080)
            .with_format(TextureFormat::Bgra8Unorm);
        assert_eq!(config.linear_format(), Some(TextureFormat::Bgra8Unorm));
    }

    #[test]
    fn linear_format_returns_view_format_when_main_is_srgb() {
        let config = SurfaceConfiguration::new(1920, 1080)
            .with_format(TextureFormat::Bgra8UnormSrgb)
            .with_srgb_view_format();  // Adds linear companion
        assert_eq!(config.linear_format(), Some(TextureFormat::Bgra8Unorm));
    }

    #[test]
    fn linear_format_none_when_no_linear_available() {
        let config = SurfaceConfiguration::new(1920, 1080)
            .with_format(TextureFormat::Rgba16Float);
        assert_eq!(config.linear_format(), None);
    }

    #[test]
    fn srgb_format_prefers_main_over_view() {
        // If main is sRGB, should return main even if view has different sRGB
        let config = SurfaceConfiguration::new(1920, 1080)
            .with_format(TextureFormat::Bgra8UnormSrgb)
            .with_view_formats(&[TextureFormat::Rgba8UnormSrgb]);
        assert_eq!(config.srgb_format(), Some(TextureFormat::Bgra8UnormSrgb));
    }
}

// ============================================================================
// Category 5: Configuration Validation with Alpha Modes Tests
// ============================================================================

mod configuration_validation_alpha_modes {
    use super::*;

    // -------------------------------------------------------------------------
    // 5.1 Validation success cases with various alpha modes
    // -------------------------------------------------------------------------

    #[test]
    fn validate_succeeds_with_opaque_alpha_mode() {
        let caps = SurfaceCapabilities {
            formats: vec![TextureFormat::Bgra8Unorm],
            present_modes: vec![PresentMode::Fifo],
            alpha_modes: vec![CompositeAlphaMode::Opaque],
            usages: TextureUsages::RENDER_ATTACHMENT,
        };
        let config = SurfaceConfiguration::new(800, 600)
            .with_format(TextureFormat::Bgra8Unorm)
            .with_present_mode(PresentMode::Fifo)
            .with_alpha_mode(CompositeAlphaMode::Opaque);
        assert!(config.validate(&caps).is_ok());
    }

    #[test]
    fn validate_succeeds_with_premultiplied_alpha_mode() {
        let caps = SurfaceCapabilities {
            formats: vec![TextureFormat::Bgra8Unorm],
            present_modes: vec![PresentMode::Fifo],
            alpha_modes: vec![CompositeAlphaMode::PreMultiplied],
            usages: TextureUsages::RENDER_ATTACHMENT,
        };
        let config = SurfaceConfiguration::new(800, 600)
            .with_format(TextureFormat::Bgra8Unorm)
            .with_present_mode(PresentMode::Fifo)
            .with_alpha_mode(CompositeAlphaMode::PreMultiplied);
        assert!(config.validate(&caps).is_ok());
    }

    #[test]
    fn validate_succeeds_with_postmultiplied_alpha_mode() {
        let caps = SurfaceCapabilities {
            formats: vec![TextureFormat::Bgra8Unorm],
            present_modes: vec![PresentMode::Fifo],
            alpha_modes: vec![CompositeAlphaMode::PostMultiplied],
            usages: TextureUsages::RENDER_ATTACHMENT,
        };
        let config = SurfaceConfiguration::new(800, 600)
            .with_format(TextureFormat::Bgra8Unorm)
            .with_present_mode(PresentMode::Fifo)
            .with_alpha_mode(CompositeAlphaMode::PostMultiplied);
        assert!(config.validate(&caps).is_ok());
    }

    #[test]
    fn validate_succeeds_with_inherit_alpha_mode() {
        let caps = SurfaceCapabilities {
            formats: vec![TextureFormat::Bgra8Unorm],
            present_modes: vec![PresentMode::Fifo],
            alpha_modes: vec![CompositeAlphaMode::Inherit],
            usages: TextureUsages::RENDER_ATTACHMENT,
        };
        let config = SurfaceConfiguration::new(800, 600)
            .with_format(TextureFormat::Bgra8Unorm)
            .with_present_mode(PresentMode::Fifo)
            .with_alpha_mode(CompositeAlphaMode::Inherit);
        assert!(config.validate(&caps).is_ok());
    }

    #[test]
    fn validate_succeeds_with_auto_alpha_mode() {
        let caps = SurfaceCapabilities {
            formats: vec![TextureFormat::Bgra8Unorm],
            present_modes: vec![PresentMode::Fifo],
            alpha_modes: vec![CompositeAlphaMode::Auto],
            usages: TextureUsages::RENDER_ATTACHMENT,
        };
        let config = SurfaceConfiguration::new(800, 600)
            .with_format(TextureFormat::Bgra8Unorm)
            .with_present_mode(PresentMode::Fifo)
            .with_alpha_mode(CompositeAlphaMode::Auto);
        assert!(config.validate(&caps).is_ok());
    }

    // -------------------------------------------------------------------------
    // 5.2 Validation failure cases with unsupported alpha modes
    // -------------------------------------------------------------------------

    #[test]
    fn validate_fails_when_alpha_mode_not_supported() {
        let caps = SurfaceCapabilities {
            formats: vec![TextureFormat::Bgra8Unorm],
            present_modes: vec![PresentMode::Fifo],
            alpha_modes: vec![CompositeAlphaMode::Opaque],  // Only Opaque
            usages: TextureUsages::RENDER_ATTACHMENT,
        };
        let config = SurfaceConfiguration::new(800, 600)
            .with_format(TextureFormat::Bgra8Unorm)
            .with_present_mode(PresentMode::Fifo)
            .with_alpha_mode(CompositeAlphaMode::PreMultiplied);  // Not supported

        let result = config.validate(&caps);
        assert!(result.is_err());
        let err_msg = format!("{}", result.unwrap_err());
        assert!(err_msg.contains("alpha mode"), "Error should mention alpha mode: {}", err_msg);
    }

    #[test]
    fn validate_fails_with_postmultiplied_when_only_premultiplied() {
        let caps = SurfaceCapabilities {
            formats: vec![TextureFormat::Bgra8Unorm],
            present_modes: vec![PresentMode::Fifo],
            alpha_modes: vec![CompositeAlphaMode::PreMultiplied],
            usages: TextureUsages::RENDER_ATTACHMENT,
        };
        let config = SurfaceConfiguration::new(800, 600)
            .with_format(TextureFormat::Bgra8Unorm)
            .with_present_mode(PresentMode::Fifo)
            .with_alpha_mode(CompositeAlphaMode::PostMultiplied);

        assert!(config.validate(&caps).is_err());
    }

    // -------------------------------------------------------------------------
    // 5.3 with_alpha_mode_preference() integration
    // -------------------------------------------------------------------------

    #[test]
    fn with_alpha_mode_preference_produces_valid_config() {
        let caps = SurfaceCapabilities {
            formats: vec![TextureFormat::Bgra8Unorm],
            present_modes: vec![PresentMode::Fifo],
            alpha_modes: vec![
                CompositeAlphaMode::Opaque,
                CompositeAlphaMode::PreMultiplied,
            ],
            usages: TextureUsages::RENDER_ATTACHMENT,
        };

        let config = SurfaceConfiguration::new(1920, 1080)
            .with_format(TextureFormat::Bgra8Unorm)
            .with_present_mode(PresentMode::Fifo)
            .with_alpha_mode_preference(&caps, AlphaModePreference::PreMultiplied);

        // Should select PreMultiplied and validate successfully
        assert_eq!(config.alpha_mode, CompositeAlphaMode::PreMultiplied);
        assert!(config.validate(&caps).is_ok());
    }

    #[test]
    fn with_alpha_mode_preference_fallback_produces_valid_config() {
        let caps = SurfaceCapabilities {
            formats: vec![TextureFormat::Bgra8Unorm],
            present_modes: vec![PresentMode::Fifo],
            alpha_modes: vec![CompositeAlphaMode::Opaque],  // Only Opaque
            usages: TextureUsages::RENDER_ATTACHMENT,
        };

        let config = SurfaceConfiguration::new(1920, 1080)
            .with_format(TextureFormat::Bgra8Unorm)
            .with_present_mode(PresentMode::Fifo)
            .with_alpha_mode_preference(&caps, AlphaModePreference::PreMultiplied);  // Not available

        // Should fall back to Opaque and validate successfully
        assert_eq!(config.alpha_mode, CompositeAlphaMode::Opaque);
        assert!(config.validate(&caps).is_ok());
    }

    // -------------------------------------------------------------------------
    // 5.4 Validation with other failing conditions
    // -------------------------------------------------------------------------

    #[test]
    fn validate_fails_format_before_alpha_mode_check() {
        let caps = SurfaceCapabilities {
            formats: vec![TextureFormat::Rgba8Unorm],  // Bgra8Unorm not supported
            present_modes: vec![PresentMode::Fifo],
            alpha_modes: vec![CompositeAlphaMode::Opaque],
            usages: TextureUsages::RENDER_ATTACHMENT,
        };
        let config = SurfaceConfiguration::new(800, 600)
            .with_format(TextureFormat::Bgra8Unorm)  // Not supported
            .with_present_mode(PresentMode::Fifo)
            .with_alpha_mode(CompositeAlphaMode::PreMultiplied);  // Also not supported

        let result = config.validate(&caps);
        assert!(result.is_err());
        let err_msg = format!("{}", result.unwrap_err());
        // Format check should fail first
        assert!(err_msg.contains("format"), "Error should mention format: {}", err_msg);
    }

    #[test]
    fn validate_fails_present_mode_before_alpha_mode_check() {
        let caps = SurfaceCapabilities {
            formats: vec![TextureFormat::Bgra8Unorm],
            present_modes: vec![PresentMode::Fifo],  // Mailbox not supported
            alpha_modes: vec![CompositeAlphaMode::Opaque],
            usages: TextureUsages::RENDER_ATTACHMENT,
        };
        let config = SurfaceConfiguration::new(800, 600)
            .with_format(TextureFormat::Bgra8Unorm)
            .with_present_mode(PresentMode::Mailbox)  // Not supported
            .with_alpha_mode(CompositeAlphaMode::PreMultiplied);  // Also not supported

        let result = config.validate(&caps);
        assert!(result.is_err());
        let err_msg = format!("{}", result.unwrap_err());
        // Present mode check should fail before alpha mode
        assert!(err_msg.contains("present mode"), "Error should mention present mode: {}", err_msg);
    }
}

// ============================================================================
// Category 6: Edge Cases Tests
// ============================================================================

mod edge_cases {
    use super::*;

    // -------------------------------------------------------------------------
    // 6.1 Empty capabilities
    // -------------------------------------------------------------------------

    #[test]
    fn empty_alpha_modes_returns_auto_for_preferred() {
        let caps = SurfaceCapabilities {
            formats: vec![TextureFormat::Bgra8Unorm],
            present_modes: vec![PresentMode::Fifo],
            alpha_modes: vec![],
            usages: TextureUsages::RENDER_ATTACHMENT,
        };
        // preferred_alpha_mode with empty list returns Auto
        assert_eq!(caps.preferred_alpha_mode(), CompositeAlphaMode::Auto);
    }

    #[test]
    fn empty_formats_returns_none_for_preferred() {
        let caps = SurfaceCapabilities {
            formats: vec![],
            present_modes: vec![PresentMode::Fifo],
            alpha_modes: vec![CompositeAlphaMode::Opaque],
            usages: TextureUsages::RENDER_ATTACHMENT,
        };
        assert_eq!(caps.preferred_format(), None);
    }

    #[test]
    fn empty_present_modes_returns_fifo_for_preferred() {
        let caps = SurfaceCapabilities {
            formats: vec![TextureFormat::Bgra8Unorm],
            present_modes: vec![],
            alpha_modes: vec![CompositeAlphaMode::Opaque],
            usages: TextureUsages::RENDER_ATTACHMENT,
        };
        // Falls back to Fifo when list is empty
        assert_eq!(caps.preferred_present_mode(), PresentMode::Fifo);
    }

    // -------------------------------------------------------------------------
    // 6.2 Single alpha mode only
    // -------------------------------------------------------------------------

    #[test]
    fn single_alpha_mode_auto() {
        let caps = SurfaceCapabilities {
            formats: vec![TextureFormat::Bgra8Unorm],
            present_modes: vec![PresentMode::Fifo],
            alpha_modes: vec![CompositeAlphaMode::Auto],
            usages: TextureUsages::RENDER_ATTACHMENT,
        };
        // All preferences fall back to Auto
        assert_eq!(caps.select_alpha_mode(AlphaModePreference::Opaque), CompositeAlphaMode::Auto);
        assert_eq!(caps.select_alpha_mode(AlphaModePreference::PreMultiplied), CompositeAlphaMode::Auto);
        assert_eq!(caps.select_alpha_mode(AlphaModePreference::Auto), CompositeAlphaMode::Auto);
    }

    #[test]
    fn single_alpha_mode_inherit() {
        let caps = SurfaceCapabilities {
            formats: vec![TextureFormat::Bgra8Unorm],
            present_modes: vec![PresentMode::Fifo],
            alpha_modes: vec![CompositeAlphaMode::Inherit],
            usages: TextureUsages::RENDER_ATTACHMENT,
        };
        assert_eq!(caps.select_alpha_mode(AlphaModePreference::Inherit), CompositeAlphaMode::Inherit);
        // All others also return Inherit (only option)
        assert_eq!(caps.select_alpha_mode(AlphaModePreference::Opaque), CompositeAlphaMode::Inherit);
    }

    // -------------------------------------------------------------------------
    // 6.3 Configuration dimension edge cases
    // -------------------------------------------------------------------------

    #[test]
    fn config_zero_width_clamped_to_one() {
        let config = SurfaceConfiguration::new(0, 100);
        assert_eq!(config.width, 1);
        assert_eq!(config.height, 100);
    }

    #[test]
    fn config_zero_height_clamped_to_one() {
        let config = SurfaceConfiguration::new(100, 0);
        assert_eq!(config.width, 100);
        assert_eq!(config.height, 1);
    }

    #[test]
    fn config_both_zero_clamped() {
        let config = SurfaceConfiguration::new(0, 0);
        assert_eq!(config.width, 1);
        assert_eq!(config.height, 1);
    }

    #[test]
    fn config_max_dimensions() {
        let config = SurfaceConfiguration::new(u32::MAX, u32::MAX);
        assert_eq!(config.width, u32::MAX);
        assert_eq!(config.height, u32::MAX);
    }

    #[test]
    fn from_window_size_clamps_zero() {
        let caps = SurfaceCapabilities {
            formats: vec![TextureFormat::Bgra8Unorm],
            present_modes: vec![PresentMode::Fifo],
            alpha_modes: vec![CompositeAlphaMode::Opaque],
            usages: TextureUsages::RENDER_ATTACHMENT,
        };
        let config = SurfaceConfiguration::from_window_size(0, 0, &caps);
        assert_eq!(config.width, 1);
        assert_eq!(config.height, 1);
    }

    // -------------------------------------------------------------------------
    // 6.4 Frame latency edge cases
    // -------------------------------------------------------------------------

    #[test]
    fn frame_latency_zero_clamped_to_one() {
        let config = SurfaceConfiguration::new(800, 600).with_frame_latency(0);
        assert_eq!(config.desired_maximum_frame_latency, 1);
    }

    #[test]
    fn frame_latency_one_preserved() {
        let config = SurfaceConfiguration::new(800, 600).with_frame_latency(1);
        assert_eq!(config.desired_maximum_frame_latency, 1);
    }

    #[test]
    fn frame_latency_high_value_preserved() {
        let config = SurfaceConfiguration::new(800, 600).with_frame_latency(10);
        assert_eq!(config.desired_maximum_frame_latency, 10);
    }

    // -------------------------------------------------------------------------
    // 6.5 to_wgpu() conversion preserves all fields
    // -------------------------------------------------------------------------

    #[test]
    fn to_wgpu_preserves_all_configuration_fields() {
        let config = SurfaceConfiguration::new(1920, 1080)
            .with_format(TextureFormat::Bgra8UnormSrgb)
            .with_present_mode(PresentMode::Mailbox)
            .with_alpha_mode(CompositeAlphaMode::PreMultiplied)
            .with_frame_latency(3)
            .with_view_formats(&[TextureFormat::Bgra8Unorm]);

        let wgpu_config = config.to_wgpu();

        assert_eq!(wgpu_config.format, TextureFormat::Bgra8UnormSrgb);
        assert_eq!(wgpu_config.width, 1920);
        assert_eq!(wgpu_config.height, 1080);
        assert_eq!(wgpu_config.present_mode, PresentMode::Mailbox);
        assert_eq!(wgpu_config.alpha_mode, CompositeAlphaMode::PreMultiplied);
        assert_eq!(wgpu_config.desired_maximum_frame_latency, 3);
        assert_eq!(wgpu_config.view_formats.len(), 1);
        assert_eq!(wgpu_config.view_formats[0], TextureFormat::Bgra8Unorm);
        assert_eq!(wgpu_config.usage, TextureUsages::RENDER_ATTACHMENT);
    }

    #[test]
    fn to_wgpu_empty_view_formats() {
        let config = SurfaceConfiguration::new(800, 600);
        let wgpu_config = config.to_wgpu();
        assert!(wgpu_config.view_formats.is_empty());
    }
}

// ============================================================================
// AlphaModePreference Method Tests
// ============================================================================

mod alpha_mode_preference_methods {
    use super::*;

    // -------------------------------------------------------------------------
    // description() tests
    // -------------------------------------------------------------------------

    #[test]
    fn description_opaque() {
        let desc = AlphaModePreference::Opaque.description();
        assert!(!desc.is_empty());
        assert!(desc.contains("Opaque") || desc.contains("alpha"));
    }

    #[test]
    fn description_premultiplied() {
        let desc = AlphaModePreference::PreMultiplied.description();
        assert!(!desc.is_empty());
        assert!(desc.contains("Pre-multiplied") || desc.contains("compositing") || desc.contains("alpha"));
    }

    #[test]
    fn description_postmultiplied() {
        let desc = AlphaModePreference::PostMultiplied.description();
        assert!(!desc.is_empty());
        assert!(desc.contains("Post-multiplied") || desc.contains("straight") || desc.contains("alpha"));
    }

    #[test]
    fn description_inherit() {
        let desc = AlphaModePreference::Inherit.description();
        assert!(!desc.is_empty());
        assert!(desc.contains("Inherit") || desc.contains("surface") || desc.contains("alpha"));
    }

    #[test]
    fn description_auto() {
        let desc = AlphaModePreference::Auto.description();
        assert!(!desc.is_empty());
        assert!(desc.contains("Auto") || desc.contains("select") || desc.contains("best"));
    }

    #[test]
    fn all_descriptions_are_unique() {
        let descs = [
            AlphaModePreference::Opaque.description(),
            AlphaModePreference::PreMultiplied.description(),
            AlphaModePreference::PostMultiplied.description(),
            AlphaModePreference::Inherit.description(),
            AlphaModePreference::Auto.description(),
        ];
        for i in 0..descs.len() {
            for j in (i + 1)..descs.len() {
                assert_ne!(descs[i], descs[j], "Descriptions should be unique");
            }
        }
    }

    // -------------------------------------------------------------------------
    // requires_alpha() tests
    // -------------------------------------------------------------------------

    #[test]
    fn requires_alpha_opaque_false() {
        assert!(!AlphaModePreference::Opaque.requires_alpha());
    }

    #[test]
    fn requires_alpha_premultiplied_true() {
        assert!(AlphaModePreference::PreMultiplied.requires_alpha());
    }

    #[test]
    fn requires_alpha_postmultiplied_true() {
        assert!(AlphaModePreference::PostMultiplied.requires_alpha());
    }

    #[test]
    fn requires_alpha_inherit_true() {
        assert!(AlphaModePreference::Inherit.requires_alpha());
    }

    #[test]
    fn requires_alpha_auto_true() {
        assert!(AlphaModePreference::Auto.requires_alpha());
    }

    #[test]
    fn only_opaque_does_not_require_alpha() {
        let all_prefs = [
            AlphaModePreference::Opaque,
            AlphaModePreference::PreMultiplied,
            AlphaModePreference::PostMultiplied,
            AlphaModePreference::Inherit,
            AlphaModePreference::Auto,
        ];
        let non_alpha_count = all_prefs.iter().filter(|p| !p.requires_alpha()).count();
        assert_eq!(non_alpha_count, 1, "Only Opaque should not require alpha");
    }

    // -------------------------------------------------------------------------
    // to_concrete_mode() tests
    // -------------------------------------------------------------------------

    #[test]
    fn to_concrete_mode_opaque() {
        assert_eq!(
            AlphaModePreference::Opaque.to_concrete_mode(),
            Some(CompositeAlphaMode::Opaque)
        );
    }

    #[test]
    fn to_concrete_mode_premultiplied() {
        assert_eq!(
            AlphaModePreference::PreMultiplied.to_concrete_mode(),
            Some(CompositeAlphaMode::PreMultiplied)
        );
    }

    #[test]
    fn to_concrete_mode_postmultiplied() {
        assert_eq!(
            AlphaModePreference::PostMultiplied.to_concrete_mode(),
            Some(CompositeAlphaMode::PostMultiplied)
        );
    }

    #[test]
    fn to_concrete_mode_inherit() {
        assert_eq!(
            AlphaModePreference::Inherit.to_concrete_mode(),
            Some(CompositeAlphaMode::Inherit)
        );
    }

    #[test]
    fn to_concrete_mode_auto_is_none() {
        assert_eq!(AlphaModePreference::Auto.to_concrete_mode(), None);
    }

    #[test]
    fn only_auto_returns_none_from_to_concrete_mode() {
        let all_prefs = [
            AlphaModePreference::Opaque,
            AlphaModePreference::PreMultiplied,
            AlphaModePreference::PostMultiplied,
            AlphaModePreference::Inherit,
            AlphaModePreference::Auto,
        ];
        let none_count = all_prefs.iter().filter(|p| p.to_concrete_mode().is_none()).count();
        assert_eq!(none_count, 1, "Only Auto should return None");
    }

    // -------------------------------------------------------------------------
    // Default trait tests
    // -------------------------------------------------------------------------

    #[test]
    fn default_is_auto() {
        assert_eq!(AlphaModePreference::default(), AlphaModePreference::Auto);
    }

    // -------------------------------------------------------------------------
    // Display trait tests
    // -------------------------------------------------------------------------

    #[test]
    fn display_opaque() {
        assert_eq!(format!("{}", AlphaModePreference::Opaque), "Opaque");
    }

    #[test]
    fn display_premultiplied() {
        assert_eq!(format!("{}", AlphaModePreference::PreMultiplied), "Pre-Multiplied");
    }

    #[test]
    fn display_postmultiplied() {
        assert_eq!(format!("{}", AlphaModePreference::PostMultiplied), "Post-Multiplied");
    }

    #[test]
    fn display_inherit() {
        assert_eq!(format!("{}", AlphaModePreference::Inherit), "Inherit");
    }

    #[test]
    fn display_auto() {
        assert_eq!(format!("{}", AlphaModePreference::Auto), "Auto");
    }
}

// ============================================================================
// SurfaceConfiguration Builder Chain Tests
// ============================================================================

mod surface_configuration_builder_chain {
    use super::*;

    #[test]
    fn builder_chain_all_methods() {
        let caps = SurfaceCapabilities {
            formats: vec![TextureFormat::Bgra8Unorm],
            present_modes: vec![PresentMode::Fifo, PresentMode::Mailbox],
            alpha_modes: vec![CompositeAlphaMode::Opaque, CompositeAlphaMode::PreMultiplied],
            usages: TextureUsages::RENDER_ATTACHMENT,
        };

        let config = SurfaceConfiguration::new(1920, 1080)
            .with_format(TextureFormat::Bgra8Unorm)
            .with_present_mode(PresentMode::Mailbox)
            .with_alpha_mode(CompositeAlphaMode::PreMultiplied)
            .with_frame_latency(2)
            .with_view_formats(&[TextureFormat::Bgra8UnormSrgb]);

        assert_eq!(config.format, TextureFormat::Bgra8Unorm);
        assert_eq!(config.width, 1920);
        assert_eq!(config.height, 1080);
        assert_eq!(config.present_mode, PresentMode::Mailbox);
        assert_eq!(config.alpha_mode, CompositeAlphaMode::PreMultiplied);
        assert_eq!(config.desired_maximum_frame_latency, 2);
        assert_eq!(config.view_formats.len(), 1);
        assert!(config.validate(&caps).is_ok());
    }

    #[test]
    fn builder_with_preference_methods() {
        let caps = SurfaceCapabilities {
            formats: vec![TextureFormat::Bgra8Unorm],
            present_modes: vec![PresentMode::Fifo, PresentMode::Immediate],
            alpha_modes: vec![CompositeAlphaMode::Opaque, CompositeAlphaMode::PreMultiplied],
            usages: TextureUsages::RENDER_ATTACHMENT,
        };

        let config = SurfaceConfiguration::new(1920, 1080)
            .with_format(TextureFormat::Bgra8Unorm)
            .with_present_mode_preference(&caps, PresentModePreference::LowLatency)
            .with_alpha_mode_preference(&caps, AlphaModePreference::PreMultiplied);

        assert_eq!(config.present_mode, PresentMode::Immediate);
        assert_eq!(config.alpha_mode, CompositeAlphaMode::PreMultiplied);
        assert!(config.validate(&caps).is_ok());
    }

    #[test]
    fn builder_from_capabilities() {
        let caps = SurfaceCapabilities {
            formats: vec![TextureFormat::Bgra8UnormSrgb, TextureFormat::Bgra8Unorm],
            present_modes: vec![PresentMode::Fifo, PresentMode::Mailbox],
            alpha_modes: vec![CompositeAlphaMode::Opaque],
            usages: TextureUsages::RENDER_ATTACHMENT,
        };

        let config = SurfaceConfiguration::from_capabilities(&caps, 1280, 720);

        // Should select preferred format (sRGB)
        assert_eq!(config.format, TextureFormat::Bgra8UnormSrgb);
        // Should select preferred present mode (Mailbox)
        assert_eq!(config.present_mode, PresentMode::Mailbox);
        // Should select preferred alpha mode (Opaque)
        assert_eq!(config.alpha_mode, CompositeAlphaMode::Opaque);
        assert_eq!(config.width, 1280);
        assert_eq!(config.height, 720);
        assert!(config.validate(&caps).is_ok());
    }

    #[test]
    fn builder_default() {
        let config = SurfaceConfiguration::default();
        assert_eq!(config.width, 1);
        assert_eq!(config.height, 1);
        assert_eq!(config.present_mode, PresentMode::Fifo);
        assert_eq!(config.alpha_mode, CompositeAlphaMode::Auto);
        assert_eq!(config.desired_maximum_frame_latency, 2);
        assert!(config.view_formats.is_empty());
    }
}

// ============================================================================
// FormatCategory Integration Tests
// ============================================================================

mod format_category_integration {
    use super::*;

    #[test]
    fn format_category_srgb_formats() {
        assert_eq!(FormatCategory::from_format(TextureFormat::Bgra8UnormSrgb), FormatCategory::Srgb);
        assert_eq!(FormatCategory::from_format(TextureFormat::Rgba8UnormSrgb), FormatCategory::Srgb);
    }

    #[test]
    fn format_category_linear_formats() {
        assert_eq!(FormatCategory::from_format(TextureFormat::Bgra8Unorm), FormatCategory::Linear);
        assert_eq!(FormatCategory::from_format(TextureFormat::Rgba8Unorm), FormatCategory::Linear);
    }

    #[test]
    fn format_category_hdr_formats() {
        assert_eq!(FormatCategory::from_format(TextureFormat::Rgba16Float), FormatCategory::Hdr);
        assert_eq!(FormatCategory::from_format(TextureFormat::Rgb10a2Unorm), FormatCategory::Hdr);
        assert_eq!(FormatCategory::from_format(TextureFormat::Rg11b10Float), FormatCategory::Hdr);
    }

    #[test]
    fn format_category_other_formats() {
        assert_eq!(FormatCategory::from_format(TextureFormat::R8Unorm), FormatCategory::Other);
        assert_eq!(FormatCategory::from_format(TextureFormat::Depth32Float), FormatCategory::Other);
    }

    #[test]
    fn format_category_is_gamma_corrected() {
        assert!(FormatCategory::Srgb.is_gamma_corrected());
        assert!(!FormatCategory::Linear.is_gamma_corrected());
        assert!(!FormatCategory::Hdr.is_gamma_corrected());
        assert!(!FormatCategory::Other.is_gamma_corrected());
    }

    #[test]
    fn format_category_is_hdr() {
        assert!(!FormatCategory::Srgb.is_hdr());
        assert!(!FormatCategory::Linear.is_hdr());
        assert!(FormatCategory::Hdr.is_hdr());
        assert!(!FormatCategory::Other.is_hdr());
    }

    #[test]
    fn format_category_name() {
        assert_eq!(FormatCategory::Srgb.name(), "sRGB");
        assert_eq!(FormatCategory::Linear.name(), "Linear");
        assert_eq!(FormatCategory::Hdr.name(), "HDR");
        assert_eq!(FormatCategory::Other.name(), "Other");
    }

    #[test]
    fn format_category_display() {
        assert_eq!(format!("{}", FormatCategory::Srgb), "sRGB");
        assert_eq!(format!("{}", FormatCategory::Linear), "Linear");
        assert_eq!(format!("{}", FormatCategory::Hdr), "HDR");
        assert_eq!(format!("{}", FormatCategory::Other), "Other");
    }
}

// ============================================================================
// PresentModePreference Integration Tests
// ============================================================================

mod present_mode_preference_integration {
    use super::*;

    #[test]
    fn present_mode_preference_default() {
        assert_eq!(PresentModePreference::default(), PresentModePreference::Vsync);
    }

    #[test]
    fn present_mode_preference_description_not_empty() {
        assert!(!PresentModePreference::LowLatency.description().is_empty());
        assert!(!PresentModePreference::Vsync.description().is_empty());
        assert!(!PresentModePreference::PowerSaving.description().is_empty());
        assert!(!PresentModePreference::Adaptive.description().is_empty());
        assert!(!PresentModePreference::Specific(PresentMode::Fifo).description().is_empty());
    }

    #[test]
    fn present_mode_preference_display() {
        assert_eq!(format!("{}", PresentModePreference::LowLatency), "Low Latency");
        assert_eq!(format!("{}", PresentModePreference::Vsync), "Vsync");
        assert_eq!(format!("{}", PresentModePreference::PowerSaving), "Power Saving");
        assert_eq!(format!("{}", PresentModePreference::Adaptive), "Adaptive");
        assert!(format!("{}", PresentModePreference::Specific(PresentMode::Fifo)).contains("Specific"));
    }

    #[test]
    fn select_present_mode_low_latency() {
        let caps = SurfaceCapabilities {
            formats: vec![TextureFormat::Bgra8Unorm],
            present_modes: vec![PresentMode::Fifo, PresentMode::Immediate],
            alpha_modes: vec![CompositeAlphaMode::Opaque],
            usages: TextureUsages::RENDER_ATTACHMENT,
        };
        assert_eq!(caps.select_present_mode(PresentModePreference::LowLatency), PresentMode::Immediate);
    }

    #[test]
    fn select_present_mode_vsync() {
        let caps = SurfaceCapabilities {
            formats: vec![TextureFormat::Bgra8Unorm],
            present_modes: vec![PresentMode::Fifo, PresentMode::Mailbox],
            alpha_modes: vec![CompositeAlphaMode::Opaque],
            usages: TextureUsages::RENDER_ATTACHMENT,
        };
        assert_eq!(caps.select_present_mode(PresentModePreference::Vsync), PresentMode::Mailbox);
    }

    #[test]
    fn select_present_mode_power_saving() {
        let caps = SurfaceCapabilities {
            formats: vec![TextureFormat::Bgra8Unorm],
            present_modes: vec![PresentMode::Fifo, PresentMode::Mailbox, PresentMode::Immediate],
            alpha_modes: vec![CompositeAlphaMode::Opaque],
            usages: TextureUsages::RENDER_ATTACHMENT,
        };
        // Power saving prefers Fifo
        assert_eq!(caps.select_present_mode(PresentModePreference::PowerSaving), PresentMode::Fifo);
    }

    #[test]
    fn select_present_mode_adaptive() {
        let caps = SurfaceCapabilities {
            formats: vec![TextureFormat::Bgra8Unorm],
            present_modes: vec![PresentMode::Fifo, PresentMode::FifoRelaxed, PresentMode::Mailbox],
            alpha_modes: vec![CompositeAlphaMode::Opaque],
            usages: TextureUsages::RENDER_ATTACHMENT,
        };
        assert_eq!(caps.select_present_mode(PresentModePreference::Adaptive), PresentMode::FifoRelaxed);
    }

    #[test]
    fn select_present_mode_specific_available() {
        let caps = SurfaceCapabilities {
            formats: vec![TextureFormat::Bgra8Unorm],
            present_modes: vec![PresentMode::Fifo, PresentMode::Immediate],
            alpha_modes: vec![CompositeAlphaMode::Opaque],
            usages: TextureUsages::RENDER_ATTACHMENT,
        };
        assert_eq!(
            caps.select_present_mode(PresentModePreference::Specific(PresentMode::Immediate)),
            PresentMode::Immediate
        );
    }

    #[test]
    fn select_present_mode_specific_unavailable_fallback() {
        let caps = SurfaceCapabilities {
            formats: vec![TextureFormat::Bgra8Unorm],
            present_modes: vec![PresentMode::Fifo],
            alpha_modes: vec![CompositeAlphaMode::Opaque],
            usages: TextureUsages::RENDER_ATTACHMENT,
        };
        // Mailbox not available, falls back to preferred (Fifo)
        assert_eq!(
            caps.select_present_mode(PresentModePreference::Specific(PresentMode::Mailbox)),
            PresentMode::Fifo
        );
    }
}

// ============================================================================
// PresentModeInfo Tests
// ============================================================================

mod present_mode_info_tests {
    use super::*;

    #[test]
    fn immediate_mode_info() {
        let info = PresentModeInfo::from_mode(PresentMode::Immediate);
        assert_eq!(info.mode, PresentMode::Immediate);
        assert!(!info.prevents_tearing);
        assert_eq!(info.latency_rank, 1);
        assert!(!info.power_efficient);
        assert!(info.is_competitive_gaming_mode());
        assert!(!info.is_battery_friendly());
    }

    #[test]
    fn mailbox_mode_info() {
        let info = PresentModeInfo::from_mode(PresentMode::Mailbox);
        assert_eq!(info.mode, PresentMode::Mailbox);
        assert!(info.prevents_tearing);
        assert_eq!(info.latency_rank, 2);
        assert!(!info.power_efficient);
        assert!(info.is_competitive_gaming_mode());
        assert!(!info.is_battery_friendly());
    }

    #[test]
    fn fifo_mode_info() {
        let info = PresentModeInfo::from_mode(PresentMode::Fifo);
        assert_eq!(info.mode, PresentMode::Fifo);
        assert!(info.prevents_tearing);
        assert_eq!(info.latency_rank, 4);
        assert!(info.power_efficient);
        assert!(!info.is_competitive_gaming_mode());
        assert!(info.is_battery_friendly());
    }

    #[test]
    fn fifo_relaxed_mode_info() {
        let info = PresentModeInfo::from_mode(PresentMode::FifoRelaxed);
        assert_eq!(info.mode, PresentMode::FifoRelaxed);
        assert!(info.prevents_tearing);
        assert_eq!(info.latency_rank, 3);
        assert!(info.power_efficient);
        assert!(!info.is_competitive_gaming_mode());
        assert!(info.is_battery_friendly());
    }

    #[test]
    fn present_mode_info_display() {
        let info = PresentModeInfo::from_mode(PresentMode::Mailbox);
        let display = format!("{}", info);
        assert!(display.contains("Mailbox"));
    }

    #[test]
    fn describe_present_mode_via_capabilities() {
        let info = SurfaceCapabilities::describe_present_mode(PresentMode::Immediate);
        assert_eq!(info.name, "Immediate");
        assert_eq!(info.mode, PresentMode::Immediate);
    }
}

// ============================================================================
// SurfaceError Tests
// ============================================================================

mod surface_error_tests {
    use super::*;

    #[test]
    fn surface_error_unsupported_platform() {
        let err = SurfaceError::unsupported();
        assert!(err.is_platform_error());
        assert!(!err.is_recoverable());
    }

    #[test]
    fn surface_error_window_handle() {
        let err = SurfaceError::window_handle("test error");
        assert!(!err.is_platform_error());
        assert!(!err.is_recoverable());
        assert!(format!("{}", err).contains("test error"));
    }

    #[test]
    fn surface_error_display_handle() {
        let err = SurfaceError::display_handle("display error");
        assert!(format!("{}", err).contains("display error"));
    }

    #[test]
    fn surface_error_creation_failed() {
        let err = SurfaceError::creation_failed("wgpu error");
        assert!(format!("{}", err).contains("wgpu error"));
    }

    #[test]
    fn surface_error_invalid_config() {
        let err = SurfaceError::invalid_config("bad format");
        assert!(format!("{}", err).contains("bad format"));
    }

    #[test]
    fn surface_error_lost_is_recoverable() {
        let err = SurfaceError::SurfaceLost {
            reason: "test".to_string(),
        };
        assert!(err.is_recoverable());
    }

    #[test]
    fn surface_error_outdated_is_recoverable() {
        let err = SurfaceError::SurfaceOutdated;
        assert!(err.is_recoverable());
    }
}

// ============================================================================
// PlatformTarget Tests
// ============================================================================

mod platform_target_tests {
    use super::*;

    #[test]
    fn platform_target_names() {
        assert_eq!(PlatformTarget::Wayland.name(), "Linux (Wayland)");
        assert_eq!(PlatformTarget::X11.name(), "Linux (X11)");
        assert_eq!(PlatformTarget::Windows.name(), "Windows");
        assert_eq!(PlatformTarget::MacOS.name(), "macOS");
        assert_eq!(PlatformTarget::IOS.name(), "iOS");
        assert_eq!(PlatformTarget::Android.name(), "Android");
        assert_eq!(PlatformTarget::Web.name(), "Web");
        assert_eq!(PlatformTarget::Unknown.name(), "Unknown");
    }

    #[test]
    fn platform_target_support() {
        assert!(PlatformTarget::Wayland.is_supported());
        assert!(PlatformTarget::X11.is_supported());
        assert!(PlatformTarget::Windows.is_supported());
        assert!(PlatformTarget::MacOS.is_supported());
        assert!(PlatformTarget::IOS.is_supported());
        assert!(PlatformTarget::Android.is_supported());
        assert!(PlatformTarget::Web.is_supported());
        assert!(!PlatformTarget::Unknown.is_supported());
    }

    #[test]
    fn platform_target_display() {
        assert_eq!(format!("{}", PlatformTarget::Windows), "Windows");
        assert_eq!(format!("{}", PlatformTarget::MacOS), "macOS");
    }

    #[test]
    fn platform_target_current() {
        let platform = PlatformTarget::current();
        #[cfg(any(target_os = "linux", target_os = "windows", target_os = "macos"))]
        assert!(platform.is_supported());
    }
}

// ============================================================================
// SurfaceCapabilities Additional Tests
// ============================================================================

mod surface_capabilities_additional {
    use super::*;

    #[test]
    fn supports_format() {
        let caps = SurfaceCapabilities {
            formats: vec![TextureFormat::Bgra8Unorm, TextureFormat::Rgba8Unorm],
            present_modes: vec![PresentMode::Fifo],
            alpha_modes: vec![CompositeAlphaMode::Opaque],
            usages: TextureUsages::RENDER_ATTACHMENT,
        };
        assert!(caps.supports_format(TextureFormat::Bgra8Unorm));
        assert!(caps.supports_format(TextureFormat::Rgba8Unorm));
        assert!(!caps.supports_format(TextureFormat::Rgba16Float));
    }

    #[test]
    fn supports_present_mode() {
        let caps = SurfaceCapabilities {
            formats: vec![TextureFormat::Bgra8Unorm],
            present_modes: vec![PresentMode::Fifo, PresentMode::Mailbox],
            alpha_modes: vec![CompositeAlphaMode::Opaque],
            usages: TextureUsages::RENDER_ATTACHMENT,
        };
        assert!(caps.supports_present_mode(PresentMode::Fifo));
        assert!(caps.supports_present_mode(PresentMode::Mailbox));
        assert!(!caps.supports_present_mode(PresentMode::Immediate));
    }

    #[test]
    fn supports_immediate() {
        let caps_with = SurfaceCapabilities {
            formats: vec![TextureFormat::Bgra8Unorm],
            present_modes: vec![PresentMode::Fifo, PresentMode::Immediate],
            alpha_modes: vec![CompositeAlphaMode::Opaque],
            usages: TextureUsages::RENDER_ATTACHMENT,
        };
        assert!(caps_with.supports_immediate());

        let caps_without = SurfaceCapabilities {
            formats: vec![TextureFormat::Bgra8Unorm],
            present_modes: vec![PresentMode::Fifo],
            alpha_modes: vec![CompositeAlphaMode::Opaque],
            usages: TextureUsages::RENDER_ATTACHMENT,
        };
        assert!(!caps_without.supports_immediate());
    }

    #[test]
    fn supports_mailbox() {
        let caps = SurfaceCapabilities {
            formats: vec![TextureFormat::Bgra8Unorm],
            present_modes: vec![PresentMode::Fifo, PresentMode::Mailbox],
            alpha_modes: vec![CompositeAlphaMode::Opaque],
            usages: TextureUsages::RENDER_ATTACHMENT,
        };
        assert!(caps.supports_mailbox());
    }

    #[test]
    fn supports_fifo_relaxed() {
        let caps = SurfaceCapabilities {
            formats: vec![TextureFormat::Bgra8Unorm],
            present_modes: vec![PresentMode::Fifo, PresentMode::FifoRelaxed],
            alpha_modes: vec![CompositeAlphaMode::Opaque],
            usages: TextureUsages::RENDER_ATTACHMENT,
        };
        assert!(caps.supports_fifo_relaxed());
    }

    #[test]
    fn supports_hdr() {
        let caps_no_hdr = SurfaceCapabilities {
            formats: vec![TextureFormat::Bgra8Unorm],
            present_modes: vec![PresentMode::Fifo],
            alpha_modes: vec![CompositeAlphaMode::Opaque],
            usages: TextureUsages::RENDER_ATTACHMENT,
        };
        assert!(!caps_no_hdr.supports_hdr());

        let caps_hdr = SurfaceCapabilities {
            formats: vec![TextureFormat::Bgra8Unorm, TextureFormat::Rgba16Float],
            present_modes: vec![PresentMode::Fifo],
            alpha_modes: vec![CompositeAlphaMode::Opaque],
            usages: TextureUsages::RENDER_ATTACHMENT,
        };
        assert!(caps_hdr.supports_hdr());
    }

    #[test]
    fn preferred_hdr_format() {
        let caps = SurfaceCapabilities {
            formats: vec![
                TextureFormat::Bgra8Unorm,
                TextureFormat::Rgb10a2Unorm,
                TextureFormat::Rgba16Float,
            ],
            present_modes: vec![PresentMode::Fifo],
            alpha_modes: vec![CompositeAlphaMode::Opaque],
            usages: TextureUsages::RENDER_ATTACHMENT,
        };
        // Prefers Rgba16Float over others
        assert_eq!(caps.preferred_hdr_format(), Some(TextureFormat::Rgba16Float));
    }

    #[test]
    fn preferred_hdr_format_fallback() {
        let caps = SurfaceCapabilities {
            formats: vec![TextureFormat::Bgra8Unorm, TextureFormat::Rg11b10Float],
            present_modes: vec![PresentMode::Fifo],
            alpha_modes: vec![CompositeAlphaMode::Opaque],
            usages: TextureUsages::RENDER_ATTACHMENT,
        };
        assert_eq!(caps.preferred_hdr_format(), Some(TextureFormat::Rg11b10Float));
    }

    #[test]
    fn preferred_hdr_format_none() {
        let caps = SurfaceCapabilities {
            formats: vec![TextureFormat::Bgra8Unorm],
            present_modes: vec![PresentMode::Fifo],
            alpha_modes: vec![CompositeAlphaMode::Opaque],
            usages: TextureUsages::RENDER_ATTACHMENT,
        };
        assert_eq!(caps.preferred_hdr_format(), None);
    }

    #[test]
    fn formats_in_category() {
        let caps = SurfaceCapabilities {
            formats: vec![
                TextureFormat::Bgra8Unorm,
                TextureFormat::Bgra8UnormSrgb,
                TextureFormat::Rgba16Float,
            ],
            present_modes: vec![PresentMode::Fifo],
            alpha_modes: vec![CompositeAlphaMode::Opaque],
            usages: TextureUsages::RENDER_ATTACHMENT,
        };

        let srgb = caps.formats_in_category(FormatCategory::Srgb);
        assert_eq!(srgb, vec![TextureFormat::Bgra8UnormSrgb]);

        let linear = caps.formats_in_category(FormatCategory::Linear);
        assert_eq!(linear, vec![TextureFormat::Bgra8Unorm]);

        let hdr = caps.formats_in_category(FormatCategory::Hdr);
        assert_eq!(hdr, vec![TextureFormat::Rgba16Float]);
    }

    #[test]
    fn select_format_prefer_hdr() {
        let caps = SurfaceCapabilities {
            formats: vec![TextureFormat::Bgra8UnormSrgb, TextureFormat::Rgba16Float],
            present_modes: vec![PresentMode::Fifo],
            alpha_modes: vec![CompositeAlphaMode::Opaque],
            usages: TextureUsages::RENDER_ATTACHMENT,
        };

        assert_eq!(caps.select_format(true), Some(TextureFormat::Rgba16Float));
        assert_eq!(caps.select_format(false), Some(TextureFormat::Bgra8UnormSrgb));
    }

    #[test]
    fn select_format_hdr_fallback() {
        let caps = SurfaceCapabilities {
            formats: vec![TextureFormat::Bgra8UnormSrgb],
            present_modes: vec![PresentMode::Fifo],
            alpha_modes: vec![CompositeAlphaMode::Opaque],
            usages: TextureUsages::RENDER_ATTACHMENT,
        };
        // HDR preferred but not available
        assert_eq!(caps.select_format(true), Some(TextureFormat::Bgra8UnormSrgb));
    }
}
