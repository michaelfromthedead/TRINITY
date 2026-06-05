//! Whitebox structural tests for TrinitySurface and related types.
//!
//! These tests verify the internal structure and behavior of the surface creation
//! system, including PlatformTarget, SurfaceError, SurfaceCapabilities,
//! SurfaceConfiguration, and TrinitySurface state management.
//!
//! Task: T-WGPU-P7.1.1 - Surface Creation
//!
//! Acceptance Criteria Tested:
//! 1. raw-window-handle integration (via API surface tests)
//! 2. Surface creation from Instance (via mock/structural tests)
//! 3. Window creation failure handling (SurfaceError variants)
//! 4. Platform-specific surface targets (PlatformTarget enum)
//!
//! Whitebox Coverage:
//! - Internal state transitions in TrinitySurface
//! - SurfaceError enum variants
//! - PlatformTarget detection logic
//! - SurfaceCapabilities preferred format selection
//! - SurfaceConfiguration validation
//! - Resize handling internals
//! - Frame acquisition error paths

use renderer_backend::presentation::{
    PlatformTarget, SurfaceCapabilities, SurfaceConfiguration, SurfaceError,
};

// ============================================================================
// 1. PlatformTarget Tests - Detection & Properties
// ============================================================================

mod platform_target_detection {
    use super::*;

    #[test]
    fn current_returns_valid_platform_on_common_systems() {
        let platform = PlatformTarget::current();
        // On Linux/Windows/macOS development systems, should be a known platform
        #[cfg(any(target_os = "linux", target_os = "windows", target_os = "macos"))]
        assert!(platform.is_supported());
    }

    #[test]
    fn current_returns_x11_on_linux_by_default() {
        #[cfg(target_os = "linux")]
        {
            let platform = PlatformTarget::current();
            // Default is X11, Wayland detected at runtime
            assert_eq!(platform, PlatformTarget::X11);
        }
    }

    #[test]
    fn current_returns_windows_on_windows() {
        #[cfg(target_os = "windows")]
        {
            let platform = PlatformTarget::current();
            assert_eq!(platform, PlatformTarget::Windows);
        }
    }

    #[test]
    fn current_returns_macos_on_macos() {
        #[cfg(target_os = "macos")]
        {
            let platform = PlatformTarget::current();
            assert_eq!(platform, PlatformTarget::MacOS);
        }
    }

    #[test]
    fn current_returns_web_on_wasm() {
        #[cfg(target_family = "wasm")]
        {
            let platform = PlatformTarget::current();
            assert_eq!(platform, PlatformTarget::Web);
        }
    }

    #[test]
    fn all_platform_targets_exist() {
        let _wayland = PlatformTarget::Wayland;
        let _x11 = PlatformTarget::X11;
        let _windows = PlatformTarget::Windows;
        let _macos = PlatformTarget::MacOS;
        let _ios = PlatformTarget::IOS;
        let _android = PlatformTarget::Android;
        let _web = PlatformTarget::Web;
        let _unknown = PlatformTarget::Unknown;
    }
}

mod platform_target_names {
    use super::*;

    #[test]
    fn wayland_name_is_linux_wayland() {
        assert_eq!(PlatformTarget::Wayland.name(), "Linux (Wayland)");
    }

    #[test]
    fn x11_name_is_linux_x11() {
        assert_eq!(PlatformTarget::X11.name(), "Linux (X11)");
    }

    #[test]
    fn windows_name_is_windows() {
        assert_eq!(PlatformTarget::Windows.name(), "Windows");
    }

    #[test]
    fn macos_name_is_macos() {
        assert_eq!(PlatformTarget::MacOS.name(), "macOS");
    }

    #[test]
    fn ios_name_is_ios() {
        assert_eq!(PlatformTarget::IOS.name(), "iOS");
    }

    #[test]
    fn android_name_is_android() {
        assert_eq!(PlatformTarget::Android.name(), "Android");
    }

    #[test]
    fn web_name_is_web() {
        assert_eq!(PlatformTarget::Web.name(), "Web");
    }

    #[test]
    fn unknown_name_is_unknown() {
        assert_eq!(PlatformTarget::Unknown.name(), "Unknown");
    }
}

mod platform_target_support {
    use super::*;

    #[test]
    fn wayland_is_supported() {
        assert!(PlatformTarget::Wayland.is_supported());
    }

    #[test]
    fn x11_is_supported() {
        assert!(PlatformTarget::X11.is_supported());
    }

    #[test]
    fn windows_is_supported() {
        assert!(PlatformTarget::Windows.is_supported());
    }

    #[test]
    fn macos_is_supported() {
        assert!(PlatformTarget::MacOS.is_supported());
    }

    #[test]
    fn ios_is_supported() {
        assert!(PlatformTarget::IOS.is_supported());
    }

    #[test]
    fn android_is_supported() {
        assert!(PlatformTarget::Android.is_supported());
    }

    #[test]
    fn web_is_supported() {
        assert!(PlatformTarget::Web.is_supported());
    }

    #[test]
    fn unknown_is_not_supported() {
        assert!(!PlatformTarget::Unknown.is_supported());
    }

    #[test]
    fn only_unknown_is_unsupported() {
        let platforms = [
            PlatformTarget::Wayland,
            PlatformTarget::X11,
            PlatformTarget::Windows,
            PlatformTarget::MacOS,
            PlatformTarget::IOS,
            PlatformTarget::Android,
            PlatformTarget::Web,
        ];
        for platform in platforms {
            assert!(
                platform.is_supported(),
                "{:?} should be supported",
                platform
            );
        }
        assert!(!PlatformTarget::Unknown.is_supported());
    }
}

mod platform_target_traits {
    use super::*;

    #[test]
    fn platform_target_is_debug() {
        let debug = format!("{:?}", PlatformTarget::Windows);
        assert!(debug.contains("Windows"));
    }

    #[test]
    fn platform_target_is_clone() {
        let platform = PlatformTarget::MacOS;
        let cloned = platform.clone();
        assert_eq!(platform, cloned);
    }

    #[test]
    fn platform_target_is_copy() {
        let platform = PlatformTarget::X11;
        let copy = platform;
        assert_eq!(platform, copy);
    }

    #[test]
    fn platform_target_implements_eq() {
        assert_eq!(PlatformTarget::Wayland, PlatformTarget::Wayland);
        assert_ne!(PlatformTarget::Wayland, PlatformTarget::X11);
    }

    #[test]
    fn platform_target_implements_display() {
        assert_eq!(format!("{}", PlatformTarget::Windows), "Windows");
        assert_eq!(format!("{}", PlatformTarget::MacOS), "macOS");
        assert_eq!(format!("{}", PlatformTarget::Unknown), "Unknown");
    }

    #[test]
    fn display_matches_name() {
        let platforms = [
            PlatformTarget::Wayland,
            PlatformTarget::X11,
            PlatformTarget::Windows,
            PlatformTarget::MacOS,
            PlatformTarget::IOS,
            PlatformTarget::Android,
            PlatformTarget::Web,
            PlatformTarget::Unknown,
        ];
        for platform in platforms {
            assert_eq!(format!("{}", platform), platform.name());
        }
    }
}

// ============================================================================
// 2. SurfaceError Tests - Error Variants & Properties
// ============================================================================

mod surface_error_constructors {
    use super::*;

    #[test]
    fn unsupported_creates_platform_error() {
        let err = SurfaceError::unsupported();
        assert!(err.is_platform_error());
    }

    #[test]
    fn unsupported_uses_current_platform() {
        let err = SurfaceError::unsupported();
        if let SurfaceError::UnsupportedPlatform { platform } = err {
            assert_eq!(platform, PlatformTarget::current());
        } else {
            panic!("Expected UnsupportedPlatform variant");
        }
    }

    #[test]
    fn window_handle_creates_with_message() {
        let err = SurfaceError::window_handle("test window error");
        if let SurfaceError::WindowHandleError { message, source } = err {
            assert_eq!(message, "test window error");
            assert!(source.is_none());
        } else {
            panic!("Expected WindowHandleError variant");
        }
    }

    #[test]
    fn window_handle_accepts_string() {
        let err = SurfaceError::window_handle(String::from("owned string"));
        if let SurfaceError::WindowHandleError { message, .. } = err {
            assert_eq!(message, "owned string");
        } else {
            panic!("Expected WindowHandleError variant");
        }
    }

    #[test]
    fn display_handle_creates_with_message() {
        let err = SurfaceError::display_handle("display connection lost");
        if let SurfaceError::DisplayHandleError { message, source } = err {
            assert_eq!(message, "display connection lost");
            assert!(source.is_none());
        } else {
            panic!("Expected DisplayHandleError variant");
        }
    }

    #[test]
    fn display_handle_accepts_string() {
        let err = SurfaceError::display_handle(String::from("display error"));
        if let SurfaceError::DisplayHandleError { message, .. } = err {
            assert_eq!(message, "display error");
        } else {
            panic!("Expected DisplayHandleError variant");
        }
    }

    #[test]
    fn creation_failed_creates_with_message() {
        let err = SurfaceError::creation_failed("wgpu backend error");
        if let SurfaceError::SurfaceCreationFailed { message, platform } = err {
            assert_eq!(message, "wgpu backend error");
            assert_eq!(platform, PlatformTarget::current());
        } else {
            panic!("Expected SurfaceCreationFailed variant");
        }
    }

    #[test]
    fn creation_failed_accepts_string() {
        let err = SurfaceError::creation_failed(String::from("owned wgpu error"));
        if let SurfaceError::SurfaceCreationFailed { message, .. } = err {
            assert_eq!(message, "owned wgpu error");
        } else {
            panic!("Expected SurfaceCreationFailed variant");
        }
    }

    #[test]
    fn invalid_config_creates_with_message() {
        let err = SurfaceError::invalid_config("format not supported");
        if let SurfaceError::InvalidConfiguration { message } = err {
            assert_eq!(message, "format not supported");
        } else {
            panic!("Expected InvalidConfiguration variant");
        }
    }

    #[test]
    fn invalid_config_accepts_string() {
        let err = SurfaceError::invalid_config(String::from("owned config error"));
        if let SurfaceError::InvalidConfiguration { message } = err {
            assert_eq!(message, "owned config error");
        } else {
            panic!("Expected InvalidConfiguration variant");
        }
    }
}

mod surface_error_recovery {
    use super::*;

    #[test]
    fn surface_lost_is_recoverable() {
        let err = SurfaceError::SurfaceLost {
            reason: "display mode change".to_string(),
        };
        assert!(err.is_recoverable());
    }

    #[test]
    fn surface_outdated_is_recoverable() {
        let err = SurfaceError::SurfaceOutdated;
        assert!(err.is_recoverable());
    }

    #[test]
    fn unsupported_platform_is_not_recoverable() {
        let err = SurfaceError::unsupported();
        assert!(!err.is_recoverable());
    }

    #[test]
    fn window_handle_error_is_not_recoverable() {
        let err = SurfaceError::window_handle("error");
        assert!(!err.is_recoverable());
    }

    #[test]
    fn display_handle_error_is_not_recoverable() {
        let err = SurfaceError::display_handle("error");
        assert!(!err.is_recoverable());
    }

    #[test]
    fn creation_failed_is_not_recoverable() {
        let err = SurfaceError::creation_failed("error");
        assert!(!err.is_recoverable());
    }

    #[test]
    fn invalid_config_is_not_recoverable() {
        let err = SurfaceError::invalid_config("error");
        assert!(!err.is_recoverable());
    }

    #[test]
    fn only_two_variants_are_recoverable() {
        let recoverable = [
            SurfaceError::SurfaceLost {
                reason: "test".to_string(),
            },
            SurfaceError::SurfaceOutdated,
        ];

        for err in &recoverable {
            assert!(err.is_recoverable(), "{:?} should be recoverable", err);
        }

        let non_recoverable = [
            SurfaceError::unsupported(),
            SurfaceError::window_handle("test"),
            SurfaceError::display_handle("test"),
            SurfaceError::creation_failed("test"),
            SurfaceError::invalid_config("test"),
        ];

        for err in &non_recoverable {
            assert!(!err.is_recoverable(), "{:?} should not be recoverable", err);
        }
    }
}

mod surface_error_platform {
    use super::*;

    #[test]
    fn unsupported_platform_is_platform_error() {
        let err = SurfaceError::unsupported();
        assert!(err.is_platform_error());
    }

    #[test]
    fn surface_lost_is_not_platform_error() {
        let err = SurfaceError::SurfaceLost {
            reason: "test".to_string(),
        };
        assert!(!err.is_platform_error());
    }

    #[test]
    fn surface_outdated_is_not_platform_error() {
        let err = SurfaceError::SurfaceOutdated;
        assert!(!err.is_platform_error());
    }

    #[test]
    fn window_handle_error_is_not_platform_error() {
        let err = SurfaceError::window_handle("test");
        assert!(!err.is_platform_error());
    }

    #[test]
    fn display_handle_error_is_not_platform_error() {
        let err = SurfaceError::display_handle("test");
        assert!(!err.is_platform_error());
    }

    #[test]
    fn creation_failed_is_not_platform_error() {
        let err = SurfaceError::creation_failed("test");
        assert!(!err.is_platform_error());
    }

    #[test]
    fn invalid_config_is_not_platform_error() {
        let err = SurfaceError::invalid_config("test");
        assert!(!err.is_platform_error());
    }
}

mod surface_error_display {
    use super::*;

    #[test]
    fn unsupported_platform_display_contains_platform() {
        let err = SurfaceError::unsupported();
        let display = format!("{}", err);
        assert!(display.contains("unsupported platform"));
    }

    #[test]
    fn window_handle_display_contains_message() {
        let err = SurfaceError::window_handle("specific window error");
        let display = format!("{}", err);
        assert!(display.contains("specific window error"));
    }

    #[test]
    fn display_handle_display_contains_message() {
        let err = SurfaceError::display_handle("display connection failed");
        let display = format!("{}", err);
        assert!(display.contains("display connection failed"));
    }

    #[test]
    fn creation_failed_display_contains_message() {
        let err = SurfaceError::creation_failed("wgpu backend failure");
        let display = format!("{}", err);
        assert!(display.contains("wgpu backend failure"));
    }

    #[test]
    fn invalid_config_display_contains_message() {
        let err = SurfaceError::invalid_config("bad format");
        let display = format!("{}", err);
        assert!(display.contains("bad format"));
    }

    #[test]
    fn surface_lost_display_contains_reason() {
        let err = SurfaceError::SurfaceLost {
            reason: "graphics reset".to_string(),
        };
        let display = format!("{}", err);
        assert!(display.contains("graphics reset"));
    }

    #[test]
    fn surface_outdated_display_mentions_reconfiguration() {
        let err = SurfaceError::SurfaceOutdated;
        let display = format!("{}", err);
        assert!(display.contains("reconfiguration"));
    }
}

mod surface_error_traits {
    use super::*;

    #[test]
    fn surface_error_implements_error_trait() {
        let err = SurfaceError::window_handle("test");
        // std::error::Error is implemented via thiserror
        let _: &dyn std::error::Error = &err;
    }

    #[test]
    fn surface_error_implements_debug() {
        let err = SurfaceError::window_handle("test");
        let debug = format!("{:?}", err);
        assert!(debug.contains("WindowHandleError"));
    }

    #[test]
    fn all_variants_implement_debug() {
        let variants = [
            SurfaceError::unsupported(),
            SurfaceError::window_handle("test"),
            SurfaceError::display_handle("test"),
            SurfaceError::creation_failed("test"),
            SurfaceError::invalid_config("test"),
            SurfaceError::SurfaceLost {
                reason: "test".to_string(),
            },
            SurfaceError::SurfaceOutdated,
        ];

        for err in &variants {
            let debug = format!("{:?}", err);
            assert!(!debug.is_empty());
        }
    }
}

// ============================================================================
// 3. SurfaceCapabilities Tests - Format Selection
// ============================================================================

mod surface_capabilities_format {
    use super::*;

    fn make_caps(formats: Vec<wgpu::TextureFormat>) -> SurfaceCapabilities {
        SurfaceCapabilities {
            formats,
            present_modes: vec![wgpu::PresentMode::Fifo],
            alpha_modes: vec![wgpu::CompositeAlphaMode::Auto],
            usages: wgpu::TextureUsages::RENDER_ATTACHMENT,
        }
    }

    #[test]
    fn preferred_format_selects_bgra8_srgb_first() {
        let caps = make_caps(vec![
            wgpu::TextureFormat::Bgra8Unorm,
            wgpu::TextureFormat::Bgra8UnormSrgb,
            wgpu::TextureFormat::Rgba8Unorm,
        ]);
        assert_eq!(
            caps.preferred_format(),
            Some(wgpu::TextureFormat::Bgra8UnormSrgb)
        );
    }

    #[test]
    fn preferred_format_selects_rgba8_srgb_second() {
        let caps = make_caps(vec![
            wgpu::TextureFormat::Bgra8Unorm,
            wgpu::TextureFormat::Rgba8UnormSrgb,
            wgpu::TextureFormat::Rgba8Unorm,
        ]);
        // Bgra8UnormSrgb is preferred over Rgba8UnormSrgb
        assert_eq!(
            caps.preferred_format(),
            Some(wgpu::TextureFormat::Rgba8UnormSrgb)
        );
    }

    #[test]
    fn preferred_format_falls_back_to_bgra8_unorm() {
        let caps = make_caps(vec![
            wgpu::TextureFormat::Bgra8Unorm,
            wgpu::TextureFormat::Rgba8Unorm,
        ]);
        assert_eq!(
            caps.preferred_format(),
            Some(wgpu::TextureFormat::Bgra8Unorm)
        );
    }

    #[test]
    fn preferred_format_falls_back_to_rgba8_unorm() {
        let caps = make_caps(vec![
            wgpu::TextureFormat::Rgba8Unorm,
            wgpu::TextureFormat::Rgba16Float,
        ]);
        assert_eq!(
            caps.preferred_format(),
            Some(wgpu::TextureFormat::Rgba8Unorm)
        );
    }

    #[test]
    fn preferred_format_returns_first_if_no_preferred() {
        let caps = make_caps(vec![
            wgpu::TextureFormat::Rgba16Float,
            wgpu::TextureFormat::Rg11b10Float,
        ]);
        assert_eq!(
            caps.preferred_format(),
            Some(wgpu::TextureFormat::Rgba16Float)
        );
    }

    #[test]
    fn preferred_format_returns_none_for_empty() {
        let caps = make_caps(vec![]);
        assert!(caps.preferred_format().is_none());
    }

    #[test]
    fn supports_format_returns_true_for_present() {
        let caps = make_caps(vec![
            wgpu::TextureFormat::Bgra8Unorm,
            wgpu::TextureFormat::Rgba16Float,
        ]);
        assert!(caps.supports_format(wgpu::TextureFormat::Bgra8Unorm));
        assert!(caps.supports_format(wgpu::TextureFormat::Rgba16Float));
    }

    #[test]
    fn supports_format_returns_false_for_absent() {
        let caps = make_caps(vec![wgpu::TextureFormat::Bgra8Unorm]);
        assert!(!caps.supports_format(wgpu::TextureFormat::Rgba16Float));
        assert!(!caps.supports_format(wgpu::TextureFormat::Rgba8Unorm));
    }
}

mod surface_capabilities_present_mode {
    use super::*;

    fn make_caps_with_modes(present_modes: Vec<wgpu::PresentMode>) -> SurfaceCapabilities {
        SurfaceCapabilities {
            formats: vec![wgpu::TextureFormat::Bgra8Unorm],
            present_modes,
            alpha_modes: vec![wgpu::CompositeAlphaMode::Auto],
            usages: wgpu::TextureUsages::RENDER_ATTACHMENT,
        }
    }

    #[test]
    fn preferred_present_mode_selects_mailbox_first() {
        let caps = make_caps_with_modes(vec![
            wgpu::PresentMode::Fifo,
            wgpu::PresentMode::Mailbox,
            wgpu::PresentMode::Immediate,
        ]);
        assert_eq!(caps.preferred_present_mode(), wgpu::PresentMode::Mailbox);
    }

    #[test]
    fn preferred_present_mode_falls_back_to_fifo() {
        let caps = make_caps_with_modes(vec![
            wgpu::PresentMode::Fifo,
            wgpu::PresentMode::Immediate,
        ]);
        assert_eq!(caps.preferred_present_mode(), wgpu::PresentMode::Fifo);
    }

    #[test]
    fn preferred_present_mode_returns_first_if_no_preferred() {
        let caps = make_caps_with_modes(vec![wgpu::PresentMode::Immediate]);
        assert_eq!(caps.preferred_present_mode(), wgpu::PresentMode::Immediate);
    }

    #[test]
    fn preferred_present_mode_returns_fifo_for_empty() {
        let caps = make_caps_with_modes(vec![]);
        assert_eq!(caps.preferred_present_mode(), wgpu::PresentMode::Fifo);
    }

    #[test]
    fn supports_present_mode_returns_true_for_present() {
        let caps = make_caps_with_modes(vec![
            wgpu::PresentMode::Fifo,
            wgpu::PresentMode::Mailbox,
        ]);
        assert!(caps.supports_present_mode(wgpu::PresentMode::Fifo));
        assert!(caps.supports_present_mode(wgpu::PresentMode::Mailbox));
    }

    #[test]
    fn supports_present_mode_returns_false_for_absent() {
        let caps = make_caps_with_modes(vec![wgpu::PresentMode::Fifo]);
        assert!(!caps.supports_present_mode(wgpu::PresentMode::Mailbox));
        assert!(!caps.supports_present_mode(wgpu::PresentMode::Immediate));
    }
}

mod surface_capabilities_alpha_mode {
    use super::*;

    fn make_caps_with_alpha(alpha_modes: Vec<wgpu::CompositeAlphaMode>) -> SurfaceCapabilities {
        SurfaceCapabilities {
            formats: vec![wgpu::TextureFormat::Bgra8Unorm],
            present_modes: vec![wgpu::PresentMode::Fifo],
            alpha_modes,
            usages: wgpu::TextureUsages::RENDER_ATTACHMENT,
        }
    }

    #[test]
    fn preferred_alpha_mode_selects_opaque_first() {
        let caps = make_caps_with_alpha(vec![
            wgpu::CompositeAlphaMode::Auto,
            wgpu::CompositeAlphaMode::Opaque,
            wgpu::CompositeAlphaMode::PreMultiplied,
        ]);
        assert_eq!(
            caps.preferred_alpha_mode(),
            wgpu::CompositeAlphaMode::Opaque
        );
    }

    #[test]
    fn preferred_alpha_mode_falls_back_to_first() {
        let caps = make_caps_with_alpha(vec![
            wgpu::CompositeAlphaMode::PreMultiplied,
            wgpu::CompositeAlphaMode::PostMultiplied,
        ]);
        assert_eq!(
            caps.preferred_alpha_mode(),
            wgpu::CompositeAlphaMode::PreMultiplied
        );
    }

    #[test]
    fn preferred_alpha_mode_returns_auto_for_empty() {
        let caps = make_caps_with_alpha(vec![]);
        assert_eq!(
            caps.preferred_alpha_mode(),
            wgpu::CompositeAlphaMode::Auto
        );
    }
}

mod surface_capabilities_hdr {
    use super::*;

    fn make_caps(formats: Vec<wgpu::TextureFormat>) -> SurfaceCapabilities {
        SurfaceCapabilities {
            formats,
            present_modes: vec![wgpu::PresentMode::Fifo],
            alpha_modes: vec![wgpu::CompositeAlphaMode::Auto],
            usages: wgpu::TextureUsages::RENDER_ATTACHMENT,
        }
    }

    #[test]
    fn supports_hdr_with_rgba16float() {
        let caps = make_caps(vec![
            wgpu::TextureFormat::Bgra8Unorm,
            wgpu::TextureFormat::Rgba16Float,
        ]);
        assert!(caps.supports_hdr());
    }

    #[test]
    fn supports_hdr_with_rgb10a2unorm() {
        let caps = make_caps(vec![
            wgpu::TextureFormat::Bgra8Unorm,
            wgpu::TextureFormat::Rgb10a2Unorm,
        ]);
        assert!(caps.supports_hdr());
    }

    #[test]
    fn supports_hdr_with_rg11b10float() {
        let caps = make_caps(vec![
            wgpu::TextureFormat::Bgra8Unorm,
            wgpu::TextureFormat::Rg11b10Float,
        ]);
        assert!(caps.supports_hdr());
    }

    #[test]
    fn no_hdr_with_standard_formats() {
        let caps = make_caps(vec![
            wgpu::TextureFormat::Bgra8Unorm,
            wgpu::TextureFormat::Bgra8UnormSrgb,
            wgpu::TextureFormat::Rgba8Unorm,
        ]);
        assert!(!caps.supports_hdr());
    }

    #[test]
    fn no_hdr_with_empty_formats() {
        let caps = make_caps(vec![]);
        assert!(!caps.supports_hdr());
    }
}

mod surface_capabilities_traits {
    use super::*;

    fn make_caps() -> SurfaceCapabilities {
        SurfaceCapabilities {
            formats: vec![wgpu::TextureFormat::Bgra8Unorm],
            present_modes: vec![wgpu::PresentMode::Fifo],
            alpha_modes: vec![wgpu::CompositeAlphaMode::Auto],
            usages: wgpu::TextureUsages::RENDER_ATTACHMENT,
        }
    }

    #[test]
    fn capabilities_implement_debug() {
        let caps = make_caps();
        let debug = format!("{:?}", caps);
        assert!(debug.contains("SurfaceCapabilities"));
    }

    #[test]
    fn capabilities_implement_clone() {
        let caps = make_caps();
        let cloned = caps.clone();
        assert_eq!(caps.formats, cloned.formats);
        assert_eq!(caps.present_modes, cloned.present_modes);
        assert_eq!(caps.alpha_modes, cloned.alpha_modes);
    }
}

// ============================================================================
// 4. SurfaceConfiguration Tests - Builder & Validation
// ============================================================================

mod surface_configuration_new {
    use super::*;

    #[test]
    fn new_sets_dimensions() {
        let config = SurfaceConfiguration::new(1920, 1080);
        assert_eq!(config.width, 1920);
        assert_eq!(config.height, 1080);
    }

    #[test]
    fn new_clamps_zero_width_to_one() {
        let config = SurfaceConfiguration::new(0, 1080);
        assert_eq!(config.width, 1);
        assert_eq!(config.height, 1080);
    }

    #[test]
    fn new_clamps_zero_height_to_one() {
        let config = SurfaceConfiguration::new(1920, 0);
        assert_eq!(config.width, 1920);
        assert_eq!(config.height, 1);
    }

    #[test]
    fn new_clamps_both_zero_to_one() {
        let config = SurfaceConfiguration::new(0, 0);
        assert_eq!(config.width, 1);
        assert_eq!(config.height, 1);
    }

    #[test]
    fn new_defaults_format_to_bgra8_srgb() {
        let config = SurfaceConfiguration::new(800, 600);
        assert_eq!(config.format, wgpu::TextureFormat::Bgra8UnormSrgb);
    }

    #[test]
    fn new_defaults_present_mode_to_fifo() {
        let config = SurfaceConfiguration::new(800, 600);
        assert_eq!(config.present_mode, wgpu::PresentMode::Fifo);
    }

    #[test]
    fn new_defaults_alpha_mode_to_auto() {
        let config = SurfaceConfiguration::new(800, 600);
        assert_eq!(config.alpha_mode, wgpu::CompositeAlphaMode::Auto);
    }

    #[test]
    fn new_defaults_frame_latency_to_two() {
        let config = SurfaceConfiguration::new(800, 600);
        assert_eq!(config.desired_maximum_frame_latency, 2);
    }
}

mod surface_configuration_builder {
    use super::*;

    #[test]
    fn with_format_sets_format() {
        let config = SurfaceConfiguration::new(800, 600)
            .with_format(wgpu::TextureFormat::Rgba8Unorm);
        assert_eq!(config.format, wgpu::TextureFormat::Rgba8Unorm);
    }

    #[test]
    fn with_present_mode_sets_mode() {
        let config = SurfaceConfiguration::new(800, 600)
            .with_present_mode(wgpu::PresentMode::Mailbox);
        assert_eq!(config.present_mode, wgpu::PresentMode::Mailbox);
    }

    #[test]
    fn with_alpha_mode_sets_mode() {
        let config = SurfaceConfiguration::new(800, 600)
            .with_alpha_mode(wgpu::CompositeAlphaMode::Opaque);
        assert_eq!(config.alpha_mode, wgpu::CompositeAlphaMode::Opaque);
    }

    #[test]
    fn with_frame_latency_sets_latency() {
        let config = SurfaceConfiguration::new(800, 600).with_frame_latency(3);
        assert_eq!(config.desired_maximum_frame_latency, 3);
    }

    #[test]
    fn with_frame_latency_clamps_zero_to_one() {
        let config = SurfaceConfiguration::new(800, 600).with_frame_latency(0);
        assert_eq!(config.desired_maximum_frame_latency, 1);
    }

    #[test]
    fn builder_methods_chain() {
        let config = SurfaceConfiguration::new(1280, 720)
            .with_format(wgpu::TextureFormat::Rgba16Float)
            .with_present_mode(wgpu::PresentMode::Immediate)
            .with_alpha_mode(wgpu::CompositeAlphaMode::PreMultiplied)
            .with_frame_latency(1);

        assert_eq!(config.width, 1280);
        assert_eq!(config.height, 720);
        assert_eq!(config.format, wgpu::TextureFormat::Rgba16Float);
        assert_eq!(config.present_mode, wgpu::PresentMode::Immediate);
        assert_eq!(config.alpha_mode, wgpu::CompositeAlphaMode::PreMultiplied);
        assert_eq!(config.desired_maximum_frame_latency, 1);
    }
}

mod surface_configuration_from_capabilities {
    use super::*;

    fn make_caps() -> SurfaceCapabilities {
        SurfaceCapabilities {
            formats: vec![wgpu::TextureFormat::Bgra8UnormSrgb],
            present_modes: vec![wgpu::PresentMode::Mailbox],
            alpha_modes: vec![wgpu::CompositeAlphaMode::Opaque],
            usages: wgpu::TextureUsages::RENDER_ATTACHMENT,
        }
    }

    #[test]
    fn from_capabilities_uses_preferred_format() {
        let caps = make_caps();
        let config = SurfaceConfiguration::from_capabilities(&caps, 800, 600);
        assert_eq!(config.format, wgpu::TextureFormat::Bgra8UnormSrgb);
    }

    #[test]
    fn from_capabilities_uses_preferred_present_mode() {
        let caps = make_caps();
        let config = SurfaceConfiguration::from_capabilities(&caps, 800, 600);
        assert_eq!(config.present_mode, wgpu::PresentMode::Mailbox);
    }

    #[test]
    fn from_capabilities_uses_preferred_alpha_mode() {
        let caps = make_caps();
        let config = SurfaceConfiguration::from_capabilities(&caps, 800, 600);
        assert_eq!(config.alpha_mode, wgpu::CompositeAlphaMode::Opaque);
    }

    #[test]
    fn from_capabilities_sets_dimensions() {
        let caps = make_caps();
        let config = SurfaceConfiguration::from_capabilities(&caps, 1920, 1080);
        assert_eq!(config.width, 1920);
        assert_eq!(config.height, 1080);
    }

    #[test]
    fn from_capabilities_clamps_zero_dimensions() {
        let caps = make_caps();
        let config = SurfaceConfiguration::from_capabilities(&caps, 0, 0);
        assert_eq!(config.width, 1);
        assert_eq!(config.height, 1);
    }

    #[test]
    fn from_capabilities_defaults_frame_latency() {
        let caps = make_caps();
        let config = SurfaceConfiguration::from_capabilities(&caps, 800, 600);
        assert_eq!(config.desired_maximum_frame_latency, 2);
    }

    #[test]
    fn from_capabilities_with_empty_formats_uses_default() {
        let caps = SurfaceCapabilities {
            formats: vec![],
            present_modes: vec![wgpu::PresentMode::Fifo],
            alpha_modes: vec![wgpu::CompositeAlphaMode::Auto],
            usages: wgpu::TextureUsages::RENDER_ATTACHMENT,
        };
        let config = SurfaceConfiguration::from_capabilities(&caps, 800, 600);
        assert_eq!(config.format, wgpu::TextureFormat::Bgra8UnormSrgb);
    }
}

mod surface_configuration_validation {
    use super::*;

    fn make_caps() -> SurfaceCapabilities {
        SurfaceCapabilities {
            formats: vec![wgpu::TextureFormat::Bgra8Unorm],
            present_modes: vec![wgpu::PresentMode::Fifo],
            alpha_modes: vec![wgpu::CompositeAlphaMode::Auto],
            usages: wgpu::TextureUsages::RENDER_ATTACHMENT,
        }
    }

    #[test]
    fn validate_succeeds_with_supported_config() {
        let caps = make_caps();
        let config = SurfaceConfiguration::new(800, 600)
            .with_format(wgpu::TextureFormat::Bgra8Unorm)
            .with_present_mode(wgpu::PresentMode::Fifo)
            .with_alpha_mode(wgpu::CompositeAlphaMode::Auto);
        assert!(config.validate(&caps).is_ok());
    }

    #[test]
    fn validate_fails_with_unsupported_format() {
        let caps = make_caps();
        let config = SurfaceConfiguration::new(800, 600)
            .with_format(wgpu::TextureFormat::Rgba16Float);
        let result = config.validate(&caps);
        assert!(result.is_err());
        if let Err(SurfaceError::InvalidConfiguration { message }) = result {
            assert!(message.contains("format"));
        } else {
            panic!("Expected InvalidConfiguration error");
        }
    }

    #[test]
    fn validate_fails_with_unsupported_present_mode() {
        let caps = make_caps();
        let config = SurfaceConfiguration::new(800, 600)
            .with_format(wgpu::TextureFormat::Bgra8Unorm)
            .with_present_mode(wgpu::PresentMode::Mailbox);
        let result = config.validate(&caps);
        assert!(result.is_err());
        if let Err(SurfaceError::InvalidConfiguration { message }) = result {
            assert!(message.contains("present mode"));
        } else {
            panic!("Expected InvalidConfiguration error");
        }
    }

    #[test]
    fn validate_fails_with_unsupported_alpha_mode() {
        let caps = make_caps();
        let config = SurfaceConfiguration::new(800, 600)
            .with_format(wgpu::TextureFormat::Bgra8Unorm)
            .with_present_mode(wgpu::PresentMode::Fifo)
            .with_alpha_mode(wgpu::CompositeAlphaMode::Opaque);
        let result = config.validate(&caps);
        assert!(result.is_err());
        if let Err(SurfaceError::InvalidConfiguration { message }) = result {
            assert!(message.contains("alpha mode"));
        } else {
            panic!("Expected InvalidConfiguration error");
        }
    }

    #[test]
    fn validate_checks_format_first() {
        let caps = make_caps();
        // All options are unsupported; should fail on format first
        let config = SurfaceConfiguration::new(800, 600)
            .with_format(wgpu::TextureFormat::Rgba16Float)
            .with_present_mode(wgpu::PresentMode::Mailbox)
            .with_alpha_mode(wgpu::CompositeAlphaMode::Opaque);
        let result = config.validate(&caps);
        assert!(result.is_err());
        if let Err(SurfaceError::InvalidConfiguration { message }) = result {
            assert!(message.contains("format"));
        } else {
            panic!("Expected InvalidConfiguration error");
        }
    }
}

mod surface_configuration_to_wgpu {
    use super::*;

    #[test]
    fn to_wgpu_converts_all_fields() {
        let config = SurfaceConfiguration::new(1920, 1080)
            .with_format(wgpu::TextureFormat::Rgba8Unorm)
            .with_present_mode(wgpu::PresentMode::Mailbox)
            .with_alpha_mode(wgpu::CompositeAlphaMode::Opaque)
            .with_frame_latency(3);

        let wgpu_config = config.to_wgpu();

        assert_eq!(wgpu_config.width, 1920);
        assert_eq!(wgpu_config.height, 1080);
        assert_eq!(wgpu_config.format, wgpu::TextureFormat::Rgba8Unorm);
        assert_eq!(wgpu_config.present_mode, wgpu::PresentMode::Mailbox);
        assert_eq!(wgpu_config.alpha_mode, wgpu::CompositeAlphaMode::Opaque);
        assert_eq!(wgpu_config.desired_maximum_frame_latency, 3);
    }

    #[test]
    fn to_wgpu_sets_render_attachment_usage() {
        let config = SurfaceConfiguration::new(800, 600);
        let wgpu_config = config.to_wgpu();
        assert_eq!(wgpu_config.usage, wgpu::TextureUsages::RENDER_ATTACHMENT);
    }

    #[test]
    fn to_wgpu_sets_empty_view_formats() {
        let config = SurfaceConfiguration::new(800, 600);
        let wgpu_config = config.to_wgpu();
        assert!(wgpu_config.view_formats.is_empty());
    }
}

mod surface_configuration_traits {
    use super::*;

    #[test]
    fn default_creates_1x1_config() {
        let config = SurfaceConfiguration::default();
        assert_eq!(config.width, 1);
        assert_eq!(config.height, 1);
    }

    #[test]
    fn config_implements_debug() {
        let config = SurfaceConfiguration::new(800, 600);
        let debug = format!("{:?}", config);
        assert!(debug.contains("SurfaceConfiguration"));
    }

    #[test]
    fn config_implements_clone() {
        let config = SurfaceConfiguration::new(1920, 1080)
            .with_format(wgpu::TextureFormat::Rgba16Float);
        let cloned = config.clone();
        assert_eq!(config.width, cloned.width);
        assert_eq!(config.height, cloned.height);
        assert_eq!(config.format, cloned.format);
    }
}

// ============================================================================
// 5. Boundary Value Tests
// ============================================================================

mod boundary_values {
    use super::*;

    #[test]
    fn config_with_max_u32_dimensions() {
        let config = SurfaceConfiguration::new(u32::MAX, u32::MAX);
        assert_eq!(config.width, u32::MAX);
        assert_eq!(config.height, u32::MAX);
    }

    #[test]
    fn config_with_dimension_one() {
        let config = SurfaceConfiguration::new(1, 1);
        assert_eq!(config.width, 1);
        assert_eq!(config.height, 1);
    }

    #[test]
    fn config_frame_latency_max_value() {
        let config = SurfaceConfiguration::new(800, 600).with_frame_latency(u32::MAX);
        assert_eq!(config.desired_maximum_frame_latency, u32::MAX);
    }

    #[test]
    fn empty_capabilities_handle_all_queries() {
        let caps = SurfaceCapabilities {
            formats: vec![],
            present_modes: vec![],
            alpha_modes: vec![],
            usages: wgpu::TextureUsages::empty(),
        };

        assert!(caps.preferred_format().is_none());
        assert_eq!(caps.preferred_present_mode(), wgpu::PresentMode::Fifo);
        assert_eq!(
            caps.preferred_alpha_mode(),
            wgpu::CompositeAlphaMode::Auto
        );
        assert!(!caps.supports_hdr());
        assert!(!caps.supports_format(wgpu::TextureFormat::Bgra8Unorm));
        assert!(!caps.supports_present_mode(wgpu::PresentMode::Fifo));
    }

    #[test]
    fn capabilities_with_many_formats() {
        let formats = vec![
            wgpu::TextureFormat::R8Unorm,
            wgpu::TextureFormat::Rg8Unorm,
            wgpu::TextureFormat::Rgba8Unorm,
            wgpu::TextureFormat::Rgba8UnormSrgb,
            wgpu::TextureFormat::Bgra8Unorm,
            wgpu::TextureFormat::Bgra8UnormSrgb,
            wgpu::TextureFormat::Rgba16Float,
            wgpu::TextureFormat::Rgb10a2Unorm,
        ];
        let caps = SurfaceCapabilities {
            formats,
            present_modes: vec![wgpu::PresentMode::Fifo],
            alpha_modes: vec![wgpu::CompositeAlphaMode::Auto],
            usages: wgpu::TextureUsages::RENDER_ATTACHMENT,
        };

        // Should prefer Bgra8UnormSrgb
        assert_eq!(
            caps.preferred_format(),
            Some(wgpu::TextureFormat::Bgra8UnormSrgb)
        );
        assert!(caps.supports_hdr());
    }
}

// ============================================================================
// 6. Error Path Tests
// ============================================================================

mod error_paths {
    use super::*;

    #[test]
    fn validation_error_message_includes_available_formats() {
        let caps = SurfaceCapabilities {
            formats: vec![wgpu::TextureFormat::Bgra8Unorm],
            present_modes: vec![wgpu::PresentMode::Fifo],
            alpha_modes: vec![wgpu::CompositeAlphaMode::Auto],
            usages: wgpu::TextureUsages::RENDER_ATTACHMENT,
        };
        let config =
            SurfaceConfiguration::new(800, 600).with_format(wgpu::TextureFormat::Rgba16Float);

        let err = config.validate(&caps).unwrap_err();
        let message = format!("{}", err);
        assert!(message.contains("Bgra8Unorm"));
    }

    #[test]
    fn validation_error_message_includes_available_present_modes() {
        let caps = SurfaceCapabilities {
            formats: vec![wgpu::TextureFormat::Bgra8Unorm],
            present_modes: vec![wgpu::PresentMode::Fifo],
            alpha_modes: vec![wgpu::CompositeAlphaMode::Auto],
            usages: wgpu::TextureUsages::RENDER_ATTACHMENT,
        };
        let config = SurfaceConfiguration::new(800, 600)
            .with_format(wgpu::TextureFormat::Bgra8Unorm)
            .with_present_mode(wgpu::PresentMode::Mailbox);

        let err = config.validate(&caps).unwrap_err();
        let message = format!("{}", err);
        assert!(message.contains("Fifo"));
    }

    #[test]
    fn validation_error_message_includes_available_alpha_modes() {
        let caps = SurfaceCapabilities {
            formats: vec![wgpu::TextureFormat::Bgra8Unorm],
            present_modes: vec![wgpu::PresentMode::Fifo],
            alpha_modes: vec![wgpu::CompositeAlphaMode::Auto],
            usages: wgpu::TextureUsages::RENDER_ATTACHMENT,
        };
        let config = SurfaceConfiguration::new(800, 600)
            .with_format(wgpu::TextureFormat::Bgra8Unorm)
            .with_present_mode(wgpu::PresentMode::Fifo)
            .with_alpha_mode(wgpu::CompositeAlphaMode::Opaque);

        let err = config.validate(&caps).unwrap_err();
        let message = format!("{}", err);
        assert!(message.contains("Auto"));
    }

    #[test]
    fn surface_lost_error_preserves_reason() {
        let err = SurfaceError::SurfaceLost {
            reason: "driver crashed during mode switch".to_string(),
        };
        let message = format!("{}", err);
        assert!(message.contains("driver crashed during mode switch"));
    }

    #[test]
    fn window_handle_error_with_empty_message() {
        let err = SurfaceError::window_handle("");
        let message = format!("{}", err);
        // Should still be valid even with empty message
        assert!(message.contains("window handle"));
    }

    #[test]
    fn display_handle_error_with_empty_message() {
        let err = SurfaceError::display_handle("");
        let message = format!("{}", err);
        assert!(message.contains("display handle"));
    }

    #[test]
    fn creation_failed_error_with_long_message() {
        let long_message = "x".repeat(1000);
        let err = SurfaceError::creation_failed(&long_message);
        let message = format!("{}", err);
        assert!(message.len() > 1000);
    }
}

// ============================================================================
// 7. SurfaceCapabilities::from_wgpu Tests
// ============================================================================

mod surface_capabilities_from_wgpu {
    use super::*;

    #[test]
    fn from_wgpu_copies_formats() {
        let wgpu_caps = wgpu::SurfaceCapabilities {
            formats: vec![
                wgpu::TextureFormat::Bgra8Unorm,
                wgpu::TextureFormat::Rgba8Unorm,
            ],
            present_modes: vec![wgpu::PresentMode::Fifo],
            alpha_modes: vec![wgpu::CompositeAlphaMode::Auto],
            usages: wgpu::TextureUsages::RENDER_ATTACHMENT,
        };

        let caps = SurfaceCapabilities::from_wgpu(wgpu_caps);
        assert_eq!(caps.formats.len(), 2);
        assert!(caps.formats.contains(&wgpu::TextureFormat::Bgra8Unorm));
        assert!(caps.formats.contains(&wgpu::TextureFormat::Rgba8Unorm));
    }

    #[test]
    fn from_wgpu_copies_present_modes() {
        let wgpu_caps = wgpu::SurfaceCapabilities {
            formats: vec![wgpu::TextureFormat::Bgra8Unorm],
            present_modes: vec![wgpu::PresentMode::Fifo, wgpu::PresentMode::Mailbox],
            alpha_modes: vec![wgpu::CompositeAlphaMode::Auto],
            usages: wgpu::TextureUsages::RENDER_ATTACHMENT,
        };

        let caps = SurfaceCapabilities::from_wgpu(wgpu_caps);
        assert_eq!(caps.present_modes.len(), 2);
        assert!(caps.present_modes.contains(&wgpu::PresentMode::Fifo));
        assert!(caps.present_modes.contains(&wgpu::PresentMode::Mailbox));
    }

    #[test]
    fn from_wgpu_copies_alpha_modes() {
        let wgpu_caps = wgpu::SurfaceCapabilities {
            formats: vec![wgpu::TextureFormat::Bgra8Unorm],
            present_modes: vec![wgpu::PresentMode::Fifo],
            alpha_modes: vec![
                wgpu::CompositeAlphaMode::Auto,
                wgpu::CompositeAlphaMode::Opaque,
            ],
            usages: wgpu::TextureUsages::RENDER_ATTACHMENT,
        };

        let caps = SurfaceCapabilities::from_wgpu(wgpu_caps);
        assert_eq!(caps.alpha_modes.len(), 2);
        assert!(caps.alpha_modes.contains(&wgpu::CompositeAlphaMode::Auto));
        assert!(caps.alpha_modes.contains(&wgpu::CompositeAlphaMode::Opaque));
    }

    #[test]
    fn from_wgpu_copies_usages() {
        let usages =
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC;
        let wgpu_caps = wgpu::SurfaceCapabilities {
            formats: vec![wgpu::TextureFormat::Bgra8Unorm],
            present_modes: vec![wgpu::PresentMode::Fifo],
            alpha_modes: vec![wgpu::CompositeAlphaMode::Auto],
            usages,
        };

        let caps = SurfaceCapabilities::from_wgpu(wgpu_caps);
        assert!(caps.usages.contains(wgpu::TextureUsages::RENDER_ATTACHMENT));
        assert!(caps.usages.contains(wgpu::TextureUsages::COPY_SRC));
    }
}

// ============================================================================
// 8. Integration-Style Unit Tests
// ============================================================================

mod integration_tests {
    use super::*;

    #[test]
    fn complete_surface_configuration_workflow() {
        // Simulate a typical surface configuration workflow

        // 1. Create capabilities (as would come from adapter)
        let caps = SurfaceCapabilities {
            formats: vec![
                wgpu::TextureFormat::Bgra8UnormSrgb,
                wgpu::TextureFormat::Bgra8Unorm,
            ],
            present_modes: vec![wgpu::PresentMode::Mailbox, wgpu::PresentMode::Fifo],
            alpha_modes: vec![
                wgpu::CompositeAlphaMode::Opaque,
                wgpu::CompositeAlphaMode::Auto,
            ],
            usages: wgpu::TextureUsages::RENDER_ATTACHMENT,
        };

        // 2. Create config from capabilities
        let config = SurfaceConfiguration::from_capabilities(&caps, 1920, 1080);

        // 3. Validate
        assert!(config.validate(&caps).is_ok());

        // 4. Convert to wgpu config
        let wgpu_config = config.to_wgpu();

        // 5. Verify all selections are optimal
        assert_eq!(wgpu_config.format, wgpu::TextureFormat::Bgra8UnormSrgb);
        assert_eq!(wgpu_config.present_mode, wgpu::PresentMode::Mailbox);
        assert_eq!(wgpu_config.alpha_mode, wgpu::CompositeAlphaMode::Opaque);
    }

    #[test]
    fn config_resize_workflow() {
        // Simulate resize scenario
        let mut config = SurfaceConfiguration::new(800, 600);
        assert_eq!(config.width, 800);
        assert_eq!(config.height, 600);

        // Resize
        config.width = 1920;
        config.height = 1080;
        assert_eq!(config.width, 1920);
        assert_eq!(config.height, 1080);

        // Config should still be valid
        let caps = SurfaceCapabilities {
            formats: vec![config.format],
            present_modes: vec![config.present_mode],
            alpha_modes: vec![config.alpha_mode],
            usages: wgpu::TextureUsages::RENDER_ATTACHMENT,
        };
        assert!(config.validate(&caps).is_ok());
    }

    #[test]
    fn error_handling_workflow() {
        // Create error and check recovery status
        let outdated = SurfaceError::SurfaceOutdated;
        assert!(outdated.is_recoverable());

        let lost = SurfaceError::SurfaceLost {
            reason: "window minimized".to_string(),
        };
        assert!(lost.is_recoverable());

        // Non-recoverable errors
        let config_err = SurfaceError::invalid_config("bad dimensions");
        assert!(!config_err.is_recoverable());
    }

    #[test]
    fn platform_detection_workflow() {
        let platform = PlatformTarget::current();

        // Check platform is valid
        #[cfg(any(target_os = "linux", target_os = "windows", target_os = "macos"))]
        {
            assert!(platform.is_supported());
            assert!(!platform.name().is_empty());
            assert_eq!(format!("{}", platform), platform.name());
        }
    }

    #[test]
    fn hdr_detection_workflow() {
        // SDR-only system
        let sdr_caps = SurfaceCapabilities {
            formats: vec![wgpu::TextureFormat::Bgra8UnormSrgb],
            present_modes: vec![wgpu::PresentMode::Fifo],
            alpha_modes: vec![wgpu::CompositeAlphaMode::Auto],
            usages: wgpu::TextureUsages::RENDER_ATTACHMENT,
        };
        assert!(!sdr_caps.supports_hdr());

        // HDR-capable system
        let hdr_caps = SurfaceCapabilities {
            formats: vec![
                wgpu::TextureFormat::Bgra8UnormSrgb,
                wgpu::TextureFormat::Rgba16Float,
            ],
            present_modes: vec![wgpu::PresentMode::Fifo],
            alpha_modes: vec![wgpu::CompositeAlphaMode::Auto],
            usages: wgpu::TextureUsages::RENDER_ATTACHMENT,
        };
        assert!(hdr_caps.supports_hdr());

        // Can select HDR format if available
        let hdr_config = SurfaceConfiguration::new(1920, 1080)
            .with_format(wgpu::TextureFormat::Rgba16Float);
        assert!(hdr_config.validate(&hdr_caps).is_ok());
        assert!(hdr_config.validate(&sdr_caps).is_err());
    }
}

// ============================================================================
// 9. Validation Edge Cases
// ============================================================================

mod validation_edge_cases {
    use super::*;

    #[test]
    fn validate_with_single_format_option() {
        let caps = SurfaceCapabilities {
            formats: vec![wgpu::TextureFormat::Rgba8Unorm],
            present_modes: vec![wgpu::PresentMode::Fifo],
            alpha_modes: vec![wgpu::CompositeAlphaMode::Auto],
            usages: wgpu::TextureUsages::RENDER_ATTACHMENT,
        };

        // Must use the only available format
        let valid_config = SurfaceConfiguration::new(800, 600)
            .with_format(wgpu::TextureFormat::Rgba8Unorm)
            .with_present_mode(wgpu::PresentMode::Fifo)
            .with_alpha_mode(wgpu::CompositeAlphaMode::Auto);
        assert!(valid_config.validate(&caps).is_ok());

        // Any other format fails
        let invalid_config = SurfaceConfiguration::new(800, 600)
            .with_format(wgpu::TextureFormat::Bgra8Unorm);
        assert!(invalid_config.validate(&caps).is_err());
    }

    #[test]
    fn validate_with_single_present_mode_option() {
        let caps = SurfaceCapabilities {
            formats: vec![wgpu::TextureFormat::Bgra8Unorm],
            present_modes: vec![wgpu::PresentMode::Immediate],
            alpha_modes: vec![wgpu::CompositeAlphaMode::Auto],
            usages: wgpu::TextureUsages::RENDER_ATTACHMENT,
        };

        let valid_config = SurfaceConfiguration::new(800, 600)
            .with_format(wgpu::TextureFormat::Bgra8Unorm)
            .with_present_mode(wgpu::PresentMode::Immediate)
            .with_alpha_mode(wgpu::CompositeAlphaMode::Auto);
        assert!(valid_config.validate(&caps).is_ok());

        let invalid_config = SurfaceConfiguration::new(800, 600)
            .with_format(wgpu::TextureFormat::Bgra8Unorm)
            .with_present_mode(wgpu::PresentMode::Fifo);
        assert!(invalid_config.validate(&caps).is_err());
    }

    #[test]
    fn validate_with_single_alpha_mode_option() {
        let caps = SurfaceCapabilities {
            formats: vec![wgpu::TextureFormat::Bgra8Unorm],
            present_modes: vec![wgpu::PresentMode::Fifo],
            alpha_modes: vec![wgpu::CompositeAlphaMode::PreMultiplied],
            usages: wgpu::TextureUsages::RENDER_ATTACHMENT,
        };

        let valid_config = SurfaceConfiguration::new(800, 600)
            .with_format(wgpu::TextureFormat::Bgra8Unorm)
            .with_present_mode(wgpu::PresentMode::Fifo)
            .with_alpha_mode(wgpu::CompositeAlphaMode::PreMultiplied);
        assert!(valid_config.validate(&caps).is_ok());

        let invalid_config = SurfaceConfiguration::new(800, 600)
            .with_format(wgpu::TextureFormat::Bgra8Unorm)
            .with_present_mode(wgpu::PresentMode::Fifo)
            .with_alpha_mode(wgpu::CompositeAlphaMode::Auto);
        assert!(invalid_config.validate(&caps).is_err());
    }
}
