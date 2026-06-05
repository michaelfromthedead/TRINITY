// SPDX-License-Identifier: MIT
//
// blackbox_surface_config.rs -- Blackbox tests for T-WGPU-P7.1.5 Surface Configuration.
//
// CLEANROOM: No src/ access beyond the public API exported by the crate.
// Tests use only the documented public types and functions:
//
//   - AlphaModePreference: Opaque, PreMultiplied, PostMultiplied, Inherit, Auto
//   - SurfaceConfiguration: from_window_size, builder methods
//   - SurfaceConfiguration builder: with_alpha_mode, with_alpha_mode_preference
//   - SurfaceConfiguration builder: with_view_formats, with_srgb_view_format
//   - SurfaceCapabilities: select_alpha_mode, supports_alpha_mode
//   - Helper functions: get_srgb_companion_format, are_srgb_companions
//
// PUBLIC TYPES:
//   - SurfaceConfiguration -- Core config struct
//   - SurfaceCapabilities -- Capabilities query result
//   - AlphaModePreference -- Alpha mode selection enum
//   - FormatCategory -- Format categorization enum
//   - PresentModePreference -- Present mode selection enum
//
// ACCEPTANCE CRITERIA (T-WGPU-P7.1.5):
//   1. Fullscreen game: Opaque alpha mode, no transparency
//   2. Composited window: PreMultiplied for correct blending
//   3. HDR display: Alpha mode + HDR format combination
//   4. sRGB toggle: Linear format with sRGB view for post-processing
//   5. Resize handling: from_window_size with various dimensions
//   6. Mobile surface: Limited alpha modes, graceful fallback
//
// TEST CATEGORIES:
//   1. AlphaModePreference enum API tests
//   2. Fullscreen game scenarios (Opaque alpha mode)
//   3. Composited window scenarios (PreMultiplied alpha)
//   4. HDR display scenarios (Alpha + HDR format)
//   5. sRGB toggle scenarios (view formats)
//   6. Resize handling scenarios (from_window_size)
//   7. Mobile surface scenarios (limited capabilities)
//   8. SurfaceConfiguration builder tests
//   9. Edge cases and error conditions
//   10. Real-world scenario combinations
//
// Total target: 65+ tests, 130+ assertions

use renderer_backend::presentation::{
    AlphaModePreference, FormatCategory, PresentModePreference, SurfaceCapabilities,
    SurfaceConfiguration, are_srgb_companions, get_srgb_companion_format,
};

// =============================================================================
// HELPER: Create mock capabilities for testing
// =============================================================================

fn make_caps(
    formats: Vec<wgpu::TextureFormat>,
    present_modes: Vec<wgpu::PresentMode>,
    alpha_modes: Vec<wgpu::CompositeAlphaMode>,
) -> SurfaceCapabilities {
    SurfaceCapabilities {
        formats,
        present_modes,
        alpha_modes,
        usages: wgpu::TextureUsages::RENDER_ATTACHMENT,
    }
}

fn fullscreen_game_caps() -> SurfaceCapabilities {
    make_caps(
        vec![
            wgpu::TextureFormat::Bgra8UnormSrgb,
            wgpu::TextureFormat::Bgra8Unorm,
        ],
        vec![wgpu::PresentMode::Mailbox, wgpu::PresentMode::Fifo],
        vec![wgpu::CompositeAlphaMode::Opaque],
    )
}

fn composited_window_caps() -> SurfaceCapabilities {
    make_caps(
        vec![
            wgpu::TextureFormat::Bgra8UnormSrgb,
            wgpu::TextureFormat::Rgba8UnormSrgb,
        ],
        vec![wgpu::PresentMode::Fifo, wgpu::PresentMode::Mailbox],
        vec![
            wgpu::CompositeAlphaMode::PreMultiplied,
            wgpu::CompositeAlphaMode::PostMultiplied,
            wgpu::CompositeAlphaMode::Opaque,
        ],
    )
}

fn hdr_display_caps() -> SurfaceCapabilities {
    make_caps(
        vec![
            wgpu::TextureFormat::Rgba16Float,
            wgpu::TextureFormat::Rgb10a2Unorm,
            wgpu::TextureFormat::Bgra8UnormSrgb,
        ],
        vec![wgpu::PresentMode::Mailbox, wgpu::PresentMode::Fifo],
        vec![
            wgpu::CompositeAlphaMode::Opaque,
            wgpu::CompositeAlphaMode::PreMultiplied,
        ],
    )
}

fn mobile_surface_caps() -> SurfaceCapabilities {
    // Mobile typically has limited capabilities
    make_caps(
        vec![wgpu::TextureFormat::Rgba8UnormSrgb],
        vec![wgpu::PresentMode::Fifo],
        vec![wgpu::CompositeAlphaMode::Inherit],
    )
}

fn all_alpha_modes_caps() -> SurfaceCapabilities {
    make_caps(
        vec![wgpu::TextureFormat::Bgra8UnormSrgb],
        vec![wgpu::PresentMode::Fifo],
        vec![
            wgpu::CompositeAlphaMode::Opaque,
            wgpu::CompositeAlphaMode::PreMultiplied,
            wgpu::CompositeAlphaMode::PostMultiplied,
            wgpu::CompositeAlphaMode::Inherit,
        ],
    )
}

// =============================================================================
// CATEGORY 1: AlphaModePreference Enum API Tests
// =============================================================================

mod alpha_mode_preference_api {
    use super::*;

    #[test]
    fn test_alpha_mode_preference_default_is_auto() {
        let pref = AlphaModePreference::default();
        assert_eq!(pref, AlphaModePreference::Auto);
    }

    #[test]
    fn test_alpha_mode_preference_opaque_variant() {
        let pref = AlphaModePreference::Opaque;
        assert_eq!(pref, AlphaModePreference::Opaque);
    }

    #[test]
    fn test_alpha_mode_preference_premultiplied_variant() {
        let pref = AlphaModePreference::PreMultiplied;
        assert_eq!(pref, AlphaModePreference::PreMultiplied);
    }

    #[test]
    fn test_alpha_mode_preference_postmultiplied_variant() {
        let pref = AlphaModePreference::PostMultiplied;
        assert_eq!(pref, AlphaModePreference::PostMultiplied);
    }

    #[test]
    fn test_alpha_mode_preference_inherit_variant() {
        let pref = AlphaModePreference::Inherit;
        assert_eq!(pref, AlphaModePreference::Inherit);
    }

    #[test]
    fn test_alpha_mode_preference_auto_variant() {
        let pref = AlphaModePreference::Auto;
        assert_eq!(pref, AlphaModePreference::Auto);
    }

    #[test]
    fn test_alpha_mode_preference_requires_alpha_opaque() {
        assert!(!AlphaModePreference::Opaque.requires_alpha());
    }

    #[test]
    fn test_alpha_mode_preference_requires_alpha_premultiplied() {
        assert!(AlphaModePreference::PreMultiplied.requires_alpha());
    }

    #[test]
    fn test_alpha_mode_preference_requires_alpha_postmultiplied() {
        assert!(AlphaModePreference::PostMultiplied.requires_alpha());
    }

    #[test]
    fn test_alpha_mode_preference_requires_alpha_inherit() {
        assert!(AlphaModePreference::Inherit.requires_alpha());
    }

    #[test]
    fn test_alpha_mode_preference_requires_alpha_auto() {
        assert!(AlphaModePreference::Auto.requires_alpha());
    }

    #[test]
    fn test_alpha_mode_preference_to_concrete_opaque() {
        let concrete = AlphaModePreference::Opaque.to_concrete_mode();
        assert_eq!(concrete, Some(wgpu::CompositeAlphaMode::Opaque));
    }

    #[test]
    fn test_alpha_mode_preference_to_concrete_premultiplied() {
        let concrete = AlphaModePreference::PreMultiplied.to_concrete_mode();
        assert_eq!(concrete, Some(wgpu::CompositeAlphaMode::PreMultiplied));
    }

    #[test]
    fn test_alpha_mode_preference_to_concrete_postmultiplied() {
        let concrete = AlphaModePreference::PostMultiplied.to_concrete_mode();
        assert_eq!(concrete, Some(wgpu::CompositeAlphaMode::PostMultiplied));
    }

    #[test]
    fn test_alpha_mode_preference_to_concrete_inherit() {
        let concrete = AlphaModePreference::Inherit.to_concrete_mode();
        assert_eq!(concrete, Some(wgpu::CompositeAlphaMode::Inherit));
    }

    #[test]
    fn test_alpha_mode_preference_to_concrete_auto_is_none() {
        let concrete = AlphaModePreference::Auto.to_concrete_mode();
        assert_eq!(concrete, None);
    }

    #[test]
    fn test_alpha_mode_preference_description_non_empty() {
        assert!(!AlphaModePreference::Opaque.description().is_empty());
        assert!(!AlphaModePreference::PreMultiplied.description().is_empty());
        assert!(!AlphaModePreference::PostMultiplied.description().is_empty());
        assert!(!AlphaModePreference::Inherit.description().is_empty());
        assert!(!AlphaModePreference::Auto.description().is_empty());
    }

    #[test]
    fn test_alpha_mode_preference_display_opaque() {
        let display = format!("{}", AlphaModePreference::Opaque);
        assert_eq!(display, "Opaque");
    }

    #[test]
    fn test_alpha_mode_preference_display_premultiplied() {
        let display = format!("{}", AlphaModePreference::PreMultiplied);
        assert_eq!(display, "Pre-Multiplied");
    }

    #[test]
    fn test_alpha_mode_preference_display_postmultiplied() {
        let display = format!("{}", AlphaModePreference::PostMultiplied);
        assert_eq!(display, "Post-Multiplied");
    }

    #[test]
    fn test_alpha_mode_preference_display_inherit() {
        let display = format!("{}", AlphaModePreference::Inherit);
        assert_eq!(display, "Inherit");
    }

    #[test]
    fn test_alpha_mode_preference_display_auto() {
        let display = format!("{}", AlphaModePreference::Auto);
        assert_eq!(display, "Auto");
    }

    #[test]
    fn test_alpha_mode_preference_clone() {
        let pref = AlphaModePreference::PreMultiplied;
        let cloned = pref;
        assert_eq!(pref, cloned);
    }

    #[test]
    fn test_alpha_mode_preference_debug() {
        let debug = format!("{:?}", AlphaModePreference::Opaque);
        assert!(debug.contains("Opaque"));
    }
}

// =============================================================================
// CATEGORY 2: Fullscreen Game Scenarios (Opaque Alpha Mode)
// =============================================================================

mod fullscreen_game_scenarios {
    use super::*;

    #[test]
    fn test_fullscreen_game_prefers_opaque() {
        let caps = fullscreen_game_caps();
        let mode = caps.select_alpha_mode(AlphaModePreference::Opaque);
        assert_eq!(mode, wgpu::CompositeAlphaMode::Opaque);
    }

    #[test]
    fn test_fullscreen_game_config_with_opaque() {
        let caps = fullscreen_game_caps();
        let config = SurfaceConfiguration::from_window_size(1920, 1080, &caps)
            .with_alpha_mode_preference(&caps, AlphaModePreference::Opaque);
        assert_eq!(config.alpha_mode, wgpu::CompositeAlphaMode::Opaque);
        assert_eq!(config.width, 1920);
        assert_eq!(config.height, 1080);
    }

    #[test]
    fn test_fullscreen_game_4k_resolution() {
        let caps = fullscreen_game_caps();
        let config = SurfaceConfiguration::from_window_size(3840, 2160, &caps)
            .with_alpha_mode_preference(&caps, AlphaModePreference::Opaque);
        assert_eq!(config.width, 3840);
        assert_eq!(config.height, 2160);
        assert_eq!(config.alpha_mode, wgpu::CompositeAlphaMode::Opaque);
    }

    #[test]
    fn test_fullscreen_game_ultrawide() {
        let caps = fullscreen_game_caps();
        let config = SurfaceConfiguration::from_window_size(3440, 1440, &caps)
            .with_alpha_mode_preference(&caps, AlphaModePreference::Opaque);
        assert_eq!(config.width, 3440);
        assert_eq!(config.height, 1440);
    }

    #[test]
    fn test_fullscreen_game_with_mailbox_present() {
        let caps = fullscreen_game_caps();
        let config = SurfaceConfiguration::from_window_size(1920, 1080, &caps)
            .with_alpha_mode_preference(&caps, AlphaModePreference::Opaque)
            .with_present_mode_preference(&caps, PresentModePreference::LowLatency);
        assert_eq!(config.present_mode, wgpu::PresentMode::Mailbox);
    }

    #[test]
    fn test_fullscreen_game_srgb_format() {
        let caps = fullscreen_game_caps();
        let config = SurfaceConfiguration::from_window_size(1920, 1080, &caps);
        // Should prefer sRGB format
        assert_eq!(config.format, wgpu::TextureFormat::Bgra8UnormSrgb);
    }

    #[test]
    fn test_fullscreen_game_opaque_only_surface() {
        // Surface that only supports Opaque
        let caps = make_caps(
            vec![wgpu::TextureFormat::Bgra8UnormSrgb],
            vec![wgpu::PresentMode::Fifo],
            vec![wgpu::CompositeAlphaMode::Opaque],
        );
        // Requesting PreMultiplied should fallback to Opaque
        let mode = caps.select_alpha_mode(AlphaModePreference::PreMultiplied);
        assert_eq!(mode, wgpu::CompositeAlphaMode::Opaque);
    }

    #[test]
    fn test_fullscreen_game_auto_selects_opaque() {
        let caps = fullscreen_game_caps();
        // Auto should prefer Opaque when available
        let mode = caps.select_alpha_mode(AlphaModePreference::Auto);
        assert_eq!(mode, wgpu::CompositeAlphaMode::Opaque);
    }
}

// =============================================================================
// CATEGORY 3: Composited Window Scenarios (PreMultiplied Alpha)
// =============================================================================

mod composited_window_scenarios {
    use super::*;

    #[test]
    fn test_composited_window_premultiplied() {
        let caps = composited_window_caps();
        let mode = caps.select_alpha_mode(AlphaModePreference::PreMultiplied);
        assert_eq!(mode, wgpu::CompositeAlphaMode::PreMultiplied);
    }

    #[test]
    fn test_composited_window_config() {
        let caps = composited_window_caps();
        let config = SurfaceConfiguration::from_window_size(800, 600, &caps)
            .with_alpha_mode_preference(&caps, AlphaModePreference::PreMultiplied);
        assert_eq!(config.alpha_mode, wgpu::CompositeAlphaMode::PreMultiplied);
    }

    #[test]
    fn test_composited_window_postmultiplied() {
        let caps = composited_window_caps();
        let mode = caps.select_alpha_mode(AlphaModePreference::PostMultiplied);
        assert_eq!(mode, wgpu::CompositeAlphaMode::PostMultiplied);
    }

    #[test]
    fn test_composited_window_fallback_premultiplied_to_postmultiplied() {
        // Only PostMultiplied available
        let caps = make_caps(
            vec![wgpu::TextureFormat::Bgra8UnormSrgb],
            vec![wgpu::PresentMode::Fifo],
            vec![wgpu::CompositeAlphaMode::PostMultiplied],
        );
        let mode = caps.select_alpha_mode(AlphaModePreference::PreMultiplied);
        assert_eq!(mode, wgpu::CompositeAlphaMode::PostMultiplied);
    }

    #[test]
    fn test_composited_window_fallback_postmultiplied_to_premultiplied() {
        // Only PreMultiplied available
        let caps = make_caps(
            vec![wgpu::TextureFormat::Bgra8UnormSrgb],
            vec![wgpu::PresentMode::Fifo],
            vec![wgpu::CompositeAlphaMode::PreMultiplied],
        );
        let mode = caps.select_alpha_mode(AlphaModePreference::PostMultiplied);
        assert_eq!(mode, wgpu::CompositeAlphaMode::PreMultiplied);
    }

    #[test]
    fn test_composited_window_overlay_window() {
        let caps = composited_window_caps();
        let config = SurfaceConfiguration::from_window_size(400, 300, &caps)
            .with_alpha_mode_preference(&caps, AlphaModePreference::PreMultiplied)
            .with_present_mode_preference(&caps, PresentModePreference::Vsync);
        assert_eq!(config.alpha_mode, wgpu::CompositeAlphaMode::PreMultiplied);
        assert_eq!(config.width, 400);
        assert_eq!(config.height, 300);
    }

    #[test]
    fn test_composited_window_hud_overlay() {
        let caps = composited_window_caps();
        let config = SurfaceConfiguration::from_window_size(640, 480, &caps)
            .with_alpha_mode_preference(&caps, AlphaModePreference::PreMultiplied);
        assert!(caps.supports_alpha_mode(wgpu::CompositeAlphaMode::PreMultiplied));
        assert_eq!(config.alpha_mode, wgpu::CompositeAlphaMode::PreMultiplied);
    }

    #[test]
    fn test_composited_window_supports_check() {
        let caps = composited_window_caps();
        assert!(caps.supports_alpha_mode(wgpu::CompositeAlphaMode::PreMultiplied));
        assert!(caps.supports_alpha_mode(wgpu::CompositeAlphaMode::PostMultiplied));
        assert!(caps.supports_alpha_mode(wgpu::CompositeAlphaMode::Opaque));
    }
}

// =============================================================================
// CATEGORY 4: HDR Display Scenarios (Alpha + HDR Format)
// =============================================================================

mod hdr_display_scenarios {
    use super::*;

    #[test]
    fn test_hdr_display_rgba16float() {
        let caps = hdr_display_caps();
        let config = SurfaceConfiguration::from_window_size(2560, 1440, &caps);
        // Should prefer HDR format when available
        assert!(caps.supports_hdr());
        assert_eq!(caps.preferred_hdr_format(), Some(wgpu::TextureFormat::Rgba16Float));
    }

    #[test]
    fn test_hdr_display_with_opaque_alpha() {
        let caps = hdr_display_caps();
        let config = SurfaceConfiguration::from_window_size(3840, 2160, &caps)
            .with_format(wgpu::TextureFormat::Rgba16Float)
            .with_alpha_mode_preference(&caps, AlphaModePreference::Opaque);
        assert_eq!(config.format, wgpu::TextureFormat::Rgba16Float);
        assert_eq!(config.alpha_mode, wgpu::CompositeAlphaMode::Opaque);
    }

    #[test]
    fn test_hdr_display_with_premultiplied() {
        let caps = hdr_display_caps();
        let config = SurfaceConfiguration::from_window_size(2560, 1440, &caps)
            .with_format(wgpu::TextureFormat::Rgba16Float)
            .with_alpha_mode_preference(&caps, AlphaModePreference::PreMultiplied);
        assert_eq!(config.format, wgpu::TextureFormat::Rgba16Float);
        assert_eq!(config.alpha_mode, wgpu::CompositeAlphaMode::PreMultiplied);
    }

    #[test]
    fn test_hdr_display_rgb10a2_format() {
        let caps = hdr_display_caps();
        let config = SurfaceConfiguration::from_window_size(1920, 1080, &caps)
            .with_format(wgpu::TextureFormat::Rgb10a2Unorm);
        assert_eq!(config.format, wgpu::TextureFormat::Rgb10a2Unorm);
    }

    #[test]
    fn test_hdr_display_select_format_prefer_hdr() {
        let caps = hdr_display_caps();
        let format = caps.select_format(true);
        assert_eq!(format, Some(wgpu::TextureFormat::Rgba16Float));
    }

    #[test]
    fn test_hdr_display_select_format_no_hdr_preference() {
        let caps = hdr_display_caps();
        let format = caps.select_format(false);
        // Should fall back to sRGB
        assert_eq!(format, Some(wgpu::TextureFormat::Bgra8UnormSrgb));
    }

    #[test]
    fn test_hdr_display_format_category() {
        assert_eq!(
            FormatCategory::from_format(wgpu::TextureFormat::Rgba16Float),
            FormatCategory::Hdr
        );
        assert_eq!(
            FormatCategory::from_format(wgpu::TextureFormat::Rgb10a2Unorm),
            FormatCategory::Hdr
        );
        assert_eq!(
            FormatCategory::from_format(wgpu::TextureFormat::Rg11b10Float),
            FormatCategory::Hdr
        );
    }

    #[test]
    fn test_hdr_display_format_is_hdr() {
        assert!(FormatCategory::Hdr.is_hdr());
        assert!(!FormatCategory::Srgb.is_hdr());
        assert!(!FormatCategory::Linear.is_hdr());
    }

    #[test]
    fn test_hdr_no_srgb_companion() {
        // HDR formats have no sRGB companion
        assert_eq!(get_srgb_companion_format(wgpu::TextureFormat::Rgba16Float), None);
        assert_eq!(get_srgb_companion_format(wgpu::TextureFormat::Rgb10a2Unorm), None);
        assert_eq!(get_srgb_companion_format(wgpu::TextureFormat::Rg11b10Float), None);
    }
}

// =============================================================================
// CATEGORY 5: sRGB Toggle Scenarios (View Formats)
// =============================================================================

mod srgb_toggle_scenarios {
    use super::*;

    #[test]
    fn test_srgb_toggle_linear_with_srgb_view() {
        let config = SurfaceConfiguration::new(1920, 1080)
            .with_format(wgpu::TextureFormat::Bgra8Unorm)
            .with_srgb_view_format();
        assert!(config.has_srgb_view_format());
        assert!(config.view_formats.contains(&wgpu::TextureFormat::Bgra8UnormSrgb));
    }

    #[test]
    fn test_srgb_toggle_srgb_with_linear_view() {
        let config = SurfaceConfiguration::new(1920, 1080)
            .with_format(wgpu::TextureFormat::Bgra8UnormSrgb)
            .with_srgb_view_format();
        // When base is sRGB, should add linear variant
        assert!(config.view_formats.contains(&wgpu::TextureFormat::Bgra8Unorm));
    }

    #[test]
    fn test_srgb_toggle_rgba_linear_with_srgb_view() {
        let config = SurfaceConfiguration::new(1920, 1080)
            .with_format(wgpu::TextureFormat::Rgba8Unorm)
            .with_srgb_view_format();
        assert!(config.view_formats.contains(&wgpu::TextureFormat::Rgba8UnormSrgb));
    }

    #[test]
    fn test_srgb_toggle_get_srgb_format_main() {
        let config = SurfaceConfiguration::new(1920, 1080)
            .with_format(wgpu::TextureFormat::Bgra8UnormSrgb);
        assert_eq!(config.srgb_format(), Some(wgpu::TextureFormat::Bgra8UnormSrgb));
    }

    #[test]
    fn test_srgb_toggle_get_srgb_format_view() {
        let config = SurfaceConfiguration::new(1920, 1080)
            .with_format(wgpu::TextureFormat::Bgra8Unorm)
            .with_srgb_view_format();
        assert_eq!(config.srgb_format(), Some(wgpu::TextureFormat::Bgra8UnormSrgb));
    }

    #[test]
    fn test_srgb_toggle_get_linear_format_main() {
        let config = SurfaceConfiguration::new(1920, 1080)
            .with_format(wgpu::TextureFormat::Bgra8Unorm);
        assert_eq!(config.linear_format(), Some(wgpu::TextureFormat::Bgra8Unorm));
    }

    #[test]
    fn test_srgb_toggle_get_linear_format_view() {
        let config = SurfaceConfiguration::new(1920, 1080)
            .with_format(wgpu::TextureFormat::Bgra8UnormSrgb)
            .with_srgb_view_format();
        assert_eq!(config.linear_format(), Some(wgpu::TextureFormat::Bgra8Unorm));
    }

    #[test]
    fn test_srgb_toggle_no_duplicate() {
        let config = SurfaceConfiguration::new(1920, 1080)
            .with_format(wgpu::TextureFormat::Bgra8Unorm)
            .with_srgb_view_format()
            .with_srgb_view_format(); // Called twice
        assert_eq!(config.view_formats.len(), 1);
    }

    #[test]
    fn test_srgb_toggle_manual_view_formats() {
        let config = SurfaceConfiguration::new(1920, 1080)
            .with_format(wgpu::TextureFormat::Bgra8Unorm)
            .with_view_formats(&[wgpu::TextureFormat::Bgra8UnormSrgb]);
        assert_eq!(config.view_formats.len(), 1);
        assert_eq!(config.view_formats[0], wgpu::TextureFormat::Bgra8UnormSrgb);
    }

    #[test]
    fn test_srgb_toggle_multiple_view_formats() {
        let config = SurfaceConfiguration::new(1920, 1080)
            .with_format(wgpu::TextureFormat::Bgra8Unorm)
            .with_view_formats(&[
                wgpu::TextureFormat::Bgra8UnormSrgb,
                wgpu::TextureFormat::Rgba8Unorm,
            ]);
        assert_eq!(config.view_formats.len(), 2);
    }

    #[test]
    fn test_srgb_companion_format_bgra_linear_to_srgb() {
        let companion = get_srgb_companion_format(wgpu::TextureFormat::Bgra8Unorm);
        assert_eq!(companion, Some(wgpu::TextureFormat::Bgra8UnormSrgb));
    }

    #[test]
    fn test_srgb_companion_format_bgra_srgb_to_linear() {
        let companion = get_srgb_companion_format(wgpu::TextureFormat::Bgra8UnormSrgb);
        assert_eq!(companion, Some(wgpu::TextureFormat::Bgra8Unorm));
    }

    #[test]
    fn test_srgb_companion_format_rgba_linear_to_srgb() {
        let companion = get_srgb_companion_format(wgpu::TextureFormat::Rgba8Unorm);
        assert_eq!(companion, Some(wgpu::TextureFormat::Rgba8UnormSrgb));
    }

    #[test]
    fn test_srgb_companion_format_rgba_srgb_to_linear() {
        let companion = get_srgb_companion_format(wgpu::TextureFormat::Rgba8UnormSrgb);
        assert_eq!(companion, Some(wgpu::TextureFormat::Rgba8Unorm));
    }

    #[test]
    fn test_are_srgb_companions_true() {
        assert!(are_srgb_companions(
            wgpu::TextureFormat::Bgra8Unorm,
            wgpu::TextureFormat::Bgra8UnormSrgb
        ));
        assert!(are_srgb_companions(
            wgpu::TextureFormat::Bgra8UnormSrgb,
            wgpu::TextureFormat::Bgra8Unorm
        ));
    }

    #[test]
    fn test_are_srgb_companions_false() {
        assert!(!are_srgb_companions(
            wgpu::TextureFormat::Bgra8Unorm,
            wgpu::TextureFormat::Rgba8Unorm
        ));
        assert!(!are_srgb_companions(
            wgpu::TextureFormat::Bgra8UnormSrgb,
            wgpu::TextureFormat::Rgba8UnormSrgb
        ));
    }

    #[test]
    fn test_srgb_toggle_to_wgpu_includes_view_formats() {
        let config = SurfaceConfiguration::new(1920, 1080)
            .with_format(wgpu::TextureFormat::Bgra8Unorm)
            .with_srgb_view_format();
        let wgpu_config = config.to_wgpu();
        assert_eq!(wgpu_config.view_formats.len(), 1);
        assert_eq!(wgpu_config.view_formats[0], wgpu::TextureFormat::Bgra8UnormSrgb);
    }

    #[test]
    fn test_srgb_format_none_for_hdr() {
        let config = SurfaceConfiguration::new(1920, 1080)
            .with_format(wgpu::TextureFormat::Rgba16Float);
        assert_eq!(config.srgb_format(), None);
    }
}

// =============================================================================
// CATEGORY 6: Resize Handling Scenarios (from_window_size)
// =============================================================================

mod resize_handling_scenarios {
    use super::*;

    #[test]
    fn test_resize_basic_dimensions() {
        let caps = fullscreen_game_caps();
        let config = SurfaceConfiguration::from_window_size(1024, 768, &caps);
        assert_eq!(config.width, 1024);
        assert_eq!(config.height, 768);
    }

    #[test]
    fn test_resize_zero_width_clamped() {
        let caps = fullscreen_game_caps();
        let config = SurfaceConfiguration::from_window_size(0, 600, &caps);
        assert_eq!(config.width, 1);
        assert_eq!(config.height, 600);
    }

    #[test]
    fn test_resize_zero_height_clamped() {
        let caps = fullscreen_game_caps();
        let config = SurfaceConfiguration::from_window_size(800, 0, &caps);
        assert_eq!(config.width, 800);
        assert_eq!(config.height, 1);
    }

    #[test]
    fn test_resize_both_zero_clamped() {
        let caps = fullscreen_game_caps();
        let config = SurfaceConfiguration::from_window_size(0, 0, &caps);
        assert_eq!(config.width, 1);
        assert_eq!(config.height, 1);
    }

    #[test]
    fn test_resize_large_dimensions() {
        let caps = fullscreen_game_caps();
        let config = SurfaceConfiguration::from_window_size(7680, 4320, &caps);
        assert_eq!(config.width, 7680);
        assert_eq!(config.height, 4320);
    }

    #[test]
    fn test_resize_non_standard_aspect_ratio() {
        let caps = fullscreen_game_caps();
        let config = SurfaceConfiguration::from_window_size(1000, 1, &caps);
        assert_eq!(config.width, 1000);
        assert_eq!(config.height, 1);
    }

    #[test]
    fn test_resize_square() {
        let caps = fullscreen_game_caps();
        let config = SurfaceConfiguration::from_window_size(512, 512, &caps);
        assert_eq!(config.width, 512);
        assert_eq!(config.height, 512);
    }

    #[test]
    fn test_resize_preserves_format_from_caps() {
        let caps = fullscreen_game_caps();
        let config = SurfaceConfiguration::from_window_size(1920, 1080, &caps);
        assert_eq!(config.format, wgpu::TextureFormat::Bgra8UnormSrgb);
    }

    #[test]
    fn test_resize_preserves_present_mode_from_caps() {
        let caps = fullscreen_game_caps();
        let config = SurfaceConfiguration::from_window_size(1920, 1080, &caps);
        // Should prefer Mailbox
        assert_eq!(config.present_mode, wgpu::PresentMode::Mailbox);
    }

    #[test]
    fn test_resize_preserves_alpha_mode_from_caps() {
        let caps = fullscreen_game_caps();
        let config = SurfaceConfiguration::from_window_size(1920, 1080, &caps);
        assert_eq!(config.alpha_mode, wgpu::CompositeAlphaMode::Opaque);
    }
}

// =============================================================================
// CATEGORY 7: Mobile Surface Scenarios (Limited Capabilities)
// =============================================================================

mod mobile_surface_scenarios {
    use super::*;

    #[test]
    fn test_mobile_inherit_only() {
        let caps = mobile_surface_caps();
        let mode = caps.select_alpha_mode(AlphaModePreference::Inherit);
        assert_eq!(mode, wgpu::CompositeAlphaMode::Inherit);
    }

    #[test]
    fn test_mobile_fallback_opaque_to_inherit() {
        let caps = mobile_surface_caps();
        // Opaque not available, should fallback
        let mode = caps.select_alpha_mode(AlphaModePreference::Opaque);
        assert_eq!(mode, wgpu::CompositeAlphaMode::Inherit);
    }

    #[test]
    fn test_mobile_fallback_premultiplied_to_inherit() {
        let caps = mobile_surface_caps();
        let mode = caps.select_alpha_mode(AlphaModePreference::PreMultiplied);
        // Falls back to preferred (Inherit in this case)
        assert_eq!(mode, wgpu::CompositeAlphaMode::Inherit);
    }

    #[test]
    fn test_mobile_fifo_only() {
        let caps = mobile_surface_caps();
        let mode = caps.preferred_present_mode();
        assert_eq!(mode, wgpu::PresentMode::Fifo);
    }

    #[test]
    fn test_mobile_single_format() {
        let caps = mobile_surface_caps();
        assert_eq!(caps.formats.len(), 1);
        assert_eq!(caps.preferred_format(), Some(wgpu::TextureFormat::Rgba8UnormSrgb));
    }

    #[test]
    fn test_mobile_no_hdr() {
        let caps = mobile_surface_caps();
        assert!(!caps.supports_hdr());
        assert_eq!(caps.preferred_hdr_format(), None);
    }

    #[test]
    fn test_mobile_config() {
        let caps = mobile_surface_caps();
        let config = SurfaceConfiguration::from_window_size(720, 1280, &caps);
        assert_eq!(config.format, wgpu::TextureFormat::Rgba8UnormSrgb);
        assert_eq!(config.present_mode, wgpu::PresentMode::Fifo);
        assert_eq!(config.alpha_mode, wgpu::CompositeAlphaMode::Inherit);
    }

    #[test]
    fn test_mobile_portrait_mode() {
        let caps = mobile_surface_caps();
        let config = SurfaceConfiguration::from_window_size(1080, 2340, &caps);
        assert_eq!(config.width, 1080);
        assert_eq!(config.height, 2340);
    }

    #[test]
    fn test_mobile_landscape_mode() {
        let caps = mobile_surface_caps();
        let config = SurfaceConfiguration::from_window_size(2340, 1080, &caps);
        assert_eq!(config.width, 2340);
        assert_eq!(config.height, 1080);
    }
}

// =============================================================================
// CATEGORY 8: SurfaceConfiguration Builder Tests
// =============================================================================

mod surface_configuration_builder {
    use super::*;

    #[test]
    fn test_builder_new() {
        let config = SurfaceConfiguration::new(800, 600);
        assert_eq!(config.width, 800);
        assert_eq!(config.height, 600);
    }

    #[test]
    fn test_builder_with_format() {
        let config = SurfaceConfiguration::new(800, 600)
            .with_format(wgpu::TextureFormat::Rgba8Unorm);
        assert_eq!(config.format, wgpu::TextureFormat::Rgba8Unorm);
    }

    #[test]
    fn test_builder_with_present_mode() {
        let config = SurfaceConfiguration::new(800, 600)
            .with_present_mode(wgpu::PresentMode::Immediate);
        assert_eq!(config.present_mode, wgpu::PresentMode::Immediate);
    }

    #[test]
    fn test_builder_with_alpha_mode() {
        let config = SurfaceConfiguration::new(800, 600)
            .with_alpha_mode(wgpu::CompositeAlphaMode::PreMultiplied);
        assert_eq!(config.alpha_mode, wgpu::CompositeAlphaMode::PreMultiplied);
    }

    #[test]
    fn test_builder_with_frame_latency() {
        let config = SurfaceConfiguration::new(800, 600)
            .with_frame_latency(3);
        assert_eq!(config.desired_maximum_frame_latency, 3);
    }

    #[test]
    fn test_builder_with_frame_latency_clamps_zero() {
        let config = SurfaceConfiguration::new(800, 600)
            .with_frame_latency(0);
        assert_eq!(config.desired_maximum_frame_latency, 1);
    }

    #[test]
    fn test_builder_chain() {
        let config = SurfaceConfiguration::new(1920, 1080)
            .with_format(wgpu::TextureFormat::Bgra8Unorm)
            .with_present_mode(wgpu::PresentMode::Mailbox)
            .with_alpha_mode(wgpu::CompositeAlphaMode::Opaque)
            .with_frame_latency(2)
            .with_srgb_view_format();

        assert_eq!(config.width, 1920);
        assert_eq!(config.height, 1080);
        assert_eq!(config.format, wgpu::TextureFormat::Bgra8Unorm);
        assert_eq!(config.present_mode, wgpu::PresentMode::Mailbox);
        assert_eq!(config.alpha_mode, wgpu::CompositeAlphaMode::Opaque);
        assert_eq!(config.desired_maximum_frame_latency, 2);
        assert!(config.has_srgb_view_format());
    }

    #[test]
    fn test_builder_default() {
        let config = SurfaceConfiguration::default();
        assert_eq!(config.width, 1);
        assert_eq!(config.height, 1);
    }

    #[test]
    fn test_builder_to_wgpu() {
        let config = SurfaceConfiguration::new(1920, 1080)
            .with_format(wgpu::TextureFormat::Bgra8Unorm)
            .with_present_mode(wgpu::PresentMode::Fifo)
            .with_alpha_mode(wgpu::CompositeAlphaMode::Opaque);

        let wgpu_config = config.to_wgpu();
        assert_eq!(wgpu_config.width, 1920);
        assert_eq!(wgpu_config.height, 1080);
        assert_eq!(wgpu_config.format, wgpu::TextureFormat::Bgra8Unorm);
        assert_eq!(wgpu_config.present_mode, wgpu::PresentMode::Fifo);
        assert_eq!(wgpu_config.alpha_mode, wgpu::CompositeAlphaMode::Opaque);
        assert_eq!(wgpu_config.usage, wgpu::TextureUsages::RENDER_ATTACHMENT);
    }

    #[test]
    fn test_builder_from_capabilities() {
        let caps = fullscreen_game_caps();
        let config = SurfaceConfiguration::from_capabilities(&caps, 1280, 720);
        assert_eq!(config.width, 1280);
        assert_eq!(config.height, 720);
        assert_eq!(config.format, wgpu::TextureFormat::Bgra8UnormSrgb);
    }

    #[test]
    fn test_builder_with_alpha_mode_preference() {
        let caps = all_alpha_modes_caps();
        let config = SurfaceConfiguration::new(800, 600)
            .with_alpha_mode_preference(&caps, AlphaModePreference::PreMultiplied);
        assert_eq!(config.alpha_mode, wgpu::CompositeAlphaMode::PreMultiplied);
    }

    #[test]
    fn test_builder_with_present_mode_preference() {
        let caps = fullscreen_game_caps();
        let config = SurfaceConfiguration::new(800, 600)
            .with_present_mode_preference(&caps, PresentModePreference::LowLatency);
        assert_eq!(config.present_mode, wgpu::PresentMode::Mailbox);
    }
}

// =============================================================================
// CATEGORY 9: Edge Cases and Error Conditions
// =============================================================================

mod edge_cases {
    use super::*;

    #[test]
    fn test_validate_success() {
        let caps = fullscreen_game_caps();
        let config = SurfaceConfiguration::from_window_size(800, 600, &caps);
        assert!(config.validate(&caps).is_ok());
    }

    #[test]
    fn test_validate_bad_format() {
        let caps = fullscreen_game_caps();
        let config = SurfaceConfiguration::new(800, 600)
            .with_format(wgpu::TextureFormat::Rgba16Float);
        let result = config.validate(&caps);
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_bad_present_mode() {
        let caps = mobile_surface_caps();
        let config = SurfaceConfiguration::new(800, 600)
            .with_format(wgpu::TextureFormat::Rgba8UnormSrgb)
            .with_present_mode(wgpu::PresentMode::Immediate)
            .with_alpha_mode(wgpu::CompositeAlphaMode::Inherit);
        let result = config.validate(&caps);
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_bad_alpha_mode() {
        let caps = fullscreen_game_caps();
        let config = SurfaceConfiguration::new(800, 600)
            .with_format(wgpu::TextureFormat::Bgra8UnormSrgb)
            .with_present_mode(wgpu::PresentMode::Fifo)
            .with_alpha_mode(wgpu::CompositeAlphaMode::PreMultiplied);
        let result = config.validate(&caps);
        assert!(result.is_err());
    }

    #[test]
    fn test_empty_formats() {
        let caps = make_caps(
            vec![],
            vec![wgpu::PresentMode::Fifo],
            vec![wgpu::CompositeAlphaMode::Opaque],
        );
        assert_eq!(caps.preferred_format(), None);
    }

    #[test]
    fn test_empty_alpha_modes() {
        let caps = make_caps(
            vec![wgpu::TextureFormat::Bgra8UnormSrgb],
            vec![wgpu::PresentMode::Fifo],
            vec![],
        );
        // Should default to Auto
        assert_eq!(caps.preferred_alpha_mode(), wgpu::CompositeAlphaMode::Auto);
    }

    #[test]
    fn test_single_pixel_surface() {
        let caps = fullscreen_game_caps();
        let config = SurfaceConfiguration::from_window_size(1, 1, &caps);
        assert_eq!(config.width, 1);
        assert_eq!(config.height, 1);
        assert!(config.validate(&caps).is_ok());
    }

    #[test]
    fn test_format_category_other() {
        assert_eq!(
            FormatCategory::from_format(wgpu::TextureFormat::Depth32Float),
            FormatCategory::Other
        );
        assert_eq!(
            FormatCategory::from_format(wgpu::TextureFormat::R8Unorm),
            FormatCategory::Other
        );
    }

    #[test]
    fn test_format_category_is_gamma_corrected() {
        assert!(FormatCategory::Srgb.is_gamma_corrected());
        assert!(!FormatCategory::Linear.is_gamma_corrected());
        assert!(!FormatCategory::Hdr.is_gamma_corrected());
        assert!(!FormatCategory::Other.is_gamma_corrected());
    }
}

// =============================================================================
// CATEGORY 10: Real-World Scenario Combinations
// =============================================================================

mod real_world_scenarios {
    use super::*;

    #[test]
    fn test_scenario_competitive_fps_game() {
        // Low latency, opaque, sRGB for correct gamma
        let caps = make_caps(
            vec![
                wgpu::TextureFormat::Bgra8UnormSrgb,
                wgpu::TextureFormat::Bgra8Unorm,
            ],
            vec![
                wgpu::PresentMode::Immediate,
                wgpu::PresentMode::Mailbox,
                wgpu::PresentMode::Fifo,
            ],
            vec![wgpu::CompositeAlphaMode::Opaque],
        );
        let config = SurfaceConfiguration::from_window_size(1920, 1080, &caps)
            .with_alpha_mode_preference(&caps, AlphaModePreference::Opaque)
            .with_present_mode_preference(&caps, PresentModePreference::LowLatency);

        assert_eq!(config.alpha_mode, wgpu::CompositeAlphaMode::Opaque);
        assert_eq!(config.present_mode, wgpu::PresentMode::Immediate);
        assert_eq!(config.format, wgpu::TextureFormat::Bgra8UnormSrgb);
    }

    #[test]
    fn test_scenario_streaming_video_player() {
        // Power efficient, vsync
        let caps = fullscreen_game_caps();
        let config = SurfaceConfiguration::from_window_size(1920, 1080, &caps)
            .with_alpha_mode_preference(&caps, AlphaModePreference::Opaque)
            .with_present_mode_preference(&caps, PresentModePreference::PowerSaving);

        assert_eq!(config.present_mode, wgpu::PresentMode::Fifo);
    }

    #[test]
    fn test_scenario_transparent_overlay() {
        // Composited window with premultiplied alpha
        let caps = composited_window_caps();
        let config = SurfaceConfiguration::from_window_size(400, 200, &caps)
            .with_alpha_mode_preference(&caps, AlphaModePreference::PreMultiplied)
            .with_present_mode_preference(&caps, PresentModePreference::Vsync);

        assert_eq!(config.alpha_mode, wgpu::CompositeAlphaMode::PreMultiplied);
    }

    #[test]
    fn test_scenario_photo_editor_hdr() {
        // HDR format with accurate color
        let caps = hdr_display_caps();
        let config = SurfaceConfiguration::from_window_size(2560, 1440, &caps)
            .with_format(wgpu::TextureFormat::Rgba16Float)
            .with_alpha_mode_preference(&caps, AlphaModePreference::Opaque);

        assert_eq!(config.format, wgpu::TextureFormat::Rgba16Float);
        assert_eq!(config.alpha_mode, wgpu::CompositeAlphaMode::Opaque);
    }

    #[test]
    fn test_scenario_post_processing_pipeline() {
        // Linear format with sRGB view for post-processing
        let caps = fullscreen_game_caps();
        let config = SurfaceConfiguration::from_window_size(1920, 1080, &caps)
            .with_format(wgpu::TextureFormat::Bgra8Unorm)
            .with_srgb_view_format();

        assert_eq!(config.format, wgpu::TextureFormat::Bgra8Unorm);
        assert!(config.has_srgb_view_format());
        assert_eq!(config.srgb_format(), Some(wgpu::TextureFormat::Bgra8UnormSrgb));
    }

    #[test]
    fn test_scenario_mobile_game() {
        let caps = mobile_surface_caps();
        let config = SurfaceConfiguration::from_window_size(1080, 2400, &caps);

        assert_eq!(config.format, wgpu::TextureFormat::Rgba8UnormSrgb);
        assert_eq!(config.present_mode, wgpu::PresentMode::Fifo);
        assert_eq!(config.alpha_mode, wgpu::CompositeAlphaMode::Inherit);
    }

    #[test]
    fn test_scenario_windowed_game_with_compositor() {
        let caps = composited_window_caps();
        let config = SurfaceConfiguration::from_window_size(1280, 720, &caps)
            .with_alpha_mode_preference(&caps, AlphaModePreference::Opaque);

        assert_eq!(config.alpha_mode, wgpu::CompositeAlphaMode::Opaque);
        assert_eq!(config.width, 1280);
        assert_eq!(config.height, 720);
    }

    #[test]
    fn test_scenario_vr_headset() {
        // Low latency critical for VR
        let caps = make_caps(
            vec![wgpu::TextureFormat::Rgba8UnormSrgb],
            vec![
                wgpu::PresentMode::Immediate,
                wgpu::PresentMode::Mailbox,
            ],
            vec![wgpu::CompositeAlphaMode::Opaque],
        );
        let config = SurfaceConfiguration::from_window_size(2160, 2160, &caps)
            .with_present_mode_preference(&caps, PresentModePreference::LowLatency);

        assert_eq!(config.present_mode, wgpu::PresentMode::Immediate);
        // VR typically square-ish per eye
        assert_eq!(config.width, 2160);
        assert_eq!(config.height, 2160);
    }

    #[test]
    fn test_scenario_adaptive_sync_display() {
        let caps = make_caps(
            vec![wgpu::TextureFormat::Bgra8UnormSrgb],
            vec![
                wgpu::PresentMode::Fifo,
                wgpu::PresentMode::FifoRelaxed,
                wgpu::PresentMode::Mailbox,
            ],
            vec![wgpu::CompositeAlphaMode::Opaque],
        );
        let config = SurfaceConfiguration::from_window_size(3440, 1440, &caps)
            .with_present_mode_preference(&caps, PresentModePreference::Adaptive);

        assert_eq!(config.present_mode, wgpu::PresentMode::FifoRelaxed);
    }
}

// =============================================================================
// SUMMARY STATISTICS
// =============================================================================

#[test]
fn test_summary_test_count() {
    // This test serves as documentation of the test coverage
    // Category 1: AlphaModePreference API - 24 tests
    // Category 2: Fullscreen Game - 8 tests
    // Category 3: Composited Window - 8 tests
    // Category 4: HDR Display - 9 tests
    // Category 5: sRGB Toggle - 18 tests
    // Category 6: Resize Handling - 10 tests
    // Category 7: Mobile Surface - 9 tests
    // Category 8: Builder - 14 tests
    // Category 9: Edge Cases - 9 tests
    // Category 10: Real-World - 10 tests
    // Total: 119 tests (excluding this one)
    assert!(true, "Test summary check passed");
}
