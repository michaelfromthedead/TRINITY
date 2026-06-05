// SPDX-License-Identifier: MIT
//
// blackbox_surface_capabilities.rs -- Blackbox tests for T-WGPU-P7.1.2 Surface Capabilities.
//
// CLEANROOM: No src/ access beyond the public API exported by the crate.
// Tests use only `renderer_backend::presentation::*` -- no internal fields,
// no private methods, no implementation details.
//
// Acceptance criteria (T-WGPU-P7.1.2):
//   1. get_capabilities() / capabilities() method on TrinitySurface
//   2. Query supported formats list
//   3. Query supported present modes
//   4. Query supported alpha modes
//
// Blackbox Test Requirements:
//   - Test SurfaceCapabilities public API only
//   - Test from_wgpu() factory method
//   - Test formats, present_modes, alpha_modes field access
//   - Test preferred_format() returns valid format
//   - Test preferred_present_mode() returns valid mode
//   - Test preferred_alpha_mode() returns valid mode
//   - Test supports_format() for various TextureFormat values
//   - Test supports_present_mode() for various PresentMode values
//   - Test supports_hdr() detection
//   - Test SurfaceConfiguration builder with capabilities
//
// Target: 40+ test assertions covering public capability API.

use renderer_backend::presentation::{SurfaceCapabilities, SurfaceConfiguration};
use wgpu::{CompositeAlphaMode, PresentMode, TextureFormat, TextureUsages};

// ============================================================================
// SECTION 1 -- SurfaceCapabilities::from_wgpu factory method
// ============================================================================

mod from_wgpu_tests {
    use super::*;

    #[test]
    fn from_wgpu_preserves_formats() {
        let wgpu_caps = wgpu::SurfaceCapabilities {
            formats: vec![TextureFormat::Bgra8Unorm, TextureFormat::Rgba8UnormSrgb],
            present_modes: vec![PresentMode::Fifo],
            alpha_modes: vec![CompositeAlphaMode::Auto],
            usages: TextureUsages::RENDER_ATTACHMENT,
        };
        let caps = SurfaceCapabilities::from_wgpu(wgpu_caps);
        // Assertion 1
        assert_eq!(caps.formats.len(), 2);
        // Assertion 2
        assert!(caps.formats.contains(&TextureFormat::Bgra8Unorm));
        // Assertion 3
        assert!(caps.formats.contains(&TextureFormat::Rgba8UnormSrgb));
    }

    #[test]
    fn from_wgpu_preserves_present_modes() {
        let wgpu_caps = wgpu::SurfaceCapabilities {
            formats: vec![TextureFormat::Bgra8Unorm],
            present_modes: vec![PresentMode::Fifo, PresentMode::Mailbox, PresentMode::Immediate],
            alpha_modes: vec![CompositeAlphaMode::Auto],
            usages: TextureUsages::RENDER_ATTACHMENT,
        };
        let caps = SurfaceCapabilities::from_wgpu(wgpu_caps);
        // Assertion 4
        assert_eq!(caps.present_modes.len(), 3);
        // Assertion 5
        assert!(caps.present_modes.contains(&PresentMode::Fifo));
        // Assertion 6
        assert!(caps.present_modes.contains(&PresentMode::Mailbox));
        // Assertion 7
        assert!(caps.present_modes.contains(&PresentMode::Immediate));
    }

    #[test]
    fn from_wgpu_preserves_alpha_modes() {
        let wgpu_caps = wgpu::SurfaceCapabilities {
            formats: vec![TextureFormat::Bgra8Unorm],
            present_modes: vec![PresentMode::Fifo],
            alpha_modes: vec![
                CompositeAlphaMode::Auto,
                CompositeAlphaMode::Opaque,
                CompositeAlphaMode::PreMultiplied,
            ],
            usages: TextureUsages::RENDER_ATTACHMENT,
        };
        let caps = SurfaceCapabilities::from_wgpu(wgpu_caps);
        // Assertion 8
        assert_eq!(caps.alpha_modes.len(), 3);
        // Assertion 9
        assert!(caps.alpha_modes.contains(&CompositeAlphaMode::Auto));
        // Assertion 10
        assert!(caps.alpha_modes.contains(&CompositeAlphaMode::Opaque));
        // Assertion 11
        assert!(caps.alpha_modes.contains(&CompositeAlphaMode::PreMultiplied));
    }

    #[test]
    fn from_wgpu_preserves_usages() {
        let wgpu_caps = wgpu::SurfaceCapabilities {
            formats: vec![TextureFormat::Bgra8Unorm],
            present_modes: vec![PresentMode::Fifo],
            alpha_modes: vec![CompositeAlphaMode::Auto],
            usages: TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC | TextureUsages::COPY_DST,
        };
        let caps = SurfaceCapabilities::from_wgpu(wgpu_caps);
        // Assertion 12
        assert!(caps.usages.contains(TextureUsages::RENDER_ATTACHMENT));
        // Assertion 13
        assert!(caps.usages.contains(TextureUsages::COPY_SRC));
        // Assertion 14
        assert!(caps.usages.contains(TextureUsages::COPY_DST));
    }

    #[test]
    fn from_wgpu_handles_empty_formats() {
        let wgpu_caps = wgpu::SurfaceCapabilities {
            formats: vec![],
            present_modes: vec![PresentMode::Fifo],
            alpha_modes: vec![CompositeAlphaMode::Auto],
            usages: TextureUsages::RENDER_ATTACHMENT,
        };
        let caps = SurfaceCapabilities::from_wgpu(wgpu_caps);
        // Assertion 15
        assert!(caps.formats.is_empty());
    }

    #[test]
    fn from_wgpu_handles_single_format() {
        let wgpu_caps = wgpu::SurfaceCapabilities {
            formats: vec![TextureFormat::Rgba8Unorm],
            present_modes: vec![PresentMode::Fifo],
            alpha_modes: vec![CompositeAlphaMode::Auto],
            usages: TextureUsages::RENDER_ATTACHMENT,
        };
        let caps = SurfaceCapabilities::from_wgpu(wgpu_caps);
        // Assertion 16
        assert_eq!(caps.formats.len(), 1);
        // Assertion 17
        assert_eq!(caps.formats[0], TextureFormat::Rgba8Unorm);
    }
}

// ============================================================================
// SECTION 2 -- Field access tests
// ============================================================================

mod field_access_tests {
    use super::*;

    #[test]
    fn formats_field_is_public() {
        let caps = SurfaceCapabilities {
            formats: vec![TextureFormat::Bgra8Unorm],
            present_modes: vec![PresentMode::Fifo],
            alpha_modes: vec![CompositeAlphaMode::Auto],
            usages: TextureUsages::RENDER_ATTACHMENT,
        };
        // Assertion 18: formats field is publicly accessible
        assert!(!caps.formats.is_empty());
    }

    #[test]
    fn present_modes_field_is_public() {
        let caps = SurfaceCapabilities {
            formats: vec![TextureFormat::Bgra8Unorm],
            present_modes: vec![PresentMode::Fifo, PresentMode::Mailbox],
            alpha_modes: vec![CompositeAlphaMode::Auto],
            usages: TextureUsages::RENDER_ATTACHMENT,
        };
        // Assertion 19: present_modes field is publicly accessible
        assert_eq!(caps.present_modes.len(), 2);
    }

    #[test]
    fn alpha_modes_field_is_public() {
        let caps = SurfaceCapabilities {
            formats: vec![TextureFormat::Bgra8Unorm],
            present_modes: vec![PresentMode::Fifo],
            alpha_modes: vec![CompositeAlphaMode::Auto, CompositeAlphaMode::Opaque],
            usages: TextureUsages::RENDER_ATTACHMENT,
        };
        // Assertion 20: alpha_modes field is publicly accessible
        assert_eq!(caps.alpha_modes.len(), 2);
    }

    #[test]
    fn usages_field_is_public() {
        let caps = SurfaceCapabilities {
            formats: vec![TextureFormat::Bgra8Unorm],
            present_modes: vec![PresentMode::Fifo],
            alpha_modes: vec![CompositeAlphaMode::Auto],
            usages: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
        };
        // Assertion 21: usages field is publicly accessible
        assert!(caps.usages.contains(TextureUsages::RENDER_ATTACHMENT));
        // Assertion 22
        assert!(caps.usages.contains(TextureUsages::TEXTURE_BINDING));
    }
}

// ============================================================================
// SECTION 3 -- preferred_format() tests
// ============================================================================

mod preferred_format_tests {
    use super::*;

    #[test]
    fn prefers_bgra8_srgb_first() {
        let caps = SurfaceCapabilities {
            formats: vec![
                TextureFormat::Rgba8Unorm,
                TextureFormat::Bgra8UnormSrgb,
                TextureFormat::Rgba8UnormSrgb,
            ],
            present_modes: vec![PresentMode::Fifo],
            alpha_modes: vec![CompositeAlphaMode::Auto],
            usages: TextureUsages::RENDER_ATTACHMENT,
        };
        // Assertion 23: BGRA sRGB is preferred over RGBA sRGB
        assert_eq!(caps.preferred_format(), Some(TextureFormat::Bgra8UnormSrgb));
    }

    #[test]
    fn prefers_rgba8_srgb_when_bgra_srgb_absent() {
        let caps = SurfaceCapabilities {
            formats: vec![
                TextureFormat::Rgba8Unorm,
                TextureFormat::Bgra8Unorm,
                TextureFormat::Rgba8UnormSrgb,
            ],
            present_modes: vec![PresentMode::Fifo],
            alpha_modes: vec![CompositeAlphaMode::Auto],
            usages: TextureUsages::RENDER_ATTACHMENT,
        };
        // Assertion 24: RGBA sRGB is preferred when BGRA sRGB is absent
        assert_eq!(caps.preferred_format(), Some(TextureFormat::Rgba8UnormSrgb));
    }

    #[test]
    fn falls_back_to_bgra_linear() {
        let caps = SurfaceCapabilities {
            formats: vec![TextureFormat::Rgba8Unorm, TextureFormat::Bgra8Unorm],
            present_modes: vec![PresentMode::Fifo],
            alpha_modes: vec![CompositeAlphaMode::Auto],
            usages: TextureUsages::RENDER_ATTACHMENT,
        };
        // Assertion 25: BGRA linear preferred over RGBA linear
        assert_eq!(caps.preferred_format(), Some(TextureFormat::Bgra8Unorm));
    }

    #[test]
    fn falls_back_to_rgba_linear() {
        let caps = SurfaceCapabilities {
            formats: vec![TextureFormat::Rgba8Unorm],
            present_modes: vec![PresentMode::Fifo],
            alpha_modes: vec![CompositeAlphaMode::Auto],
            usages: TextureUsages::RENDER_ATTACHMENT,
        };
        // Assertion 26: RGBA linear when only option
        assert_eq!(caps.preferred_format(), Some(TextureFormat::Rgba8Unorm));
    }

    #[test]
    fn returns_first_available_for_exotic_formats() {
        let caps = SurfaceCapabilities {
            formats: vec![TextureFormat::Rgba16Float, TextureFormat::Rg11b10Float],
            present_modes: vec![PresentMode::Fifo],
            alpha_modes: vec![CompositeAlphaMode::Auto],
            usages: TextureUsages::RENDER_ATTACHMENT,
        };
        // Assertion 27: returns first available exotic format
        assert_eq!(caps.preferred_format(), Some(TextureFormat::Rgba16Float));
    }

    #[test]
    fn returns_none_for_empty_formats() {
        let caps = SurfaceCapabilities {
            formats: vec![],
            present_modes: vec![PresentMode::Fifo],
            alpha_modes: vec![CompositeAlphaMode::Auto],
            usages: TextureUsages::RENDER_ATTACHMENT,
        };
        // Assertion 28: returns None when no formats available
        assert_eq!(caps.preferred_format(), None);
    }
}

// ============================================================================
// SECTION 4 -- preferred_present_mode() tests
// ============================================================================

mod preferred_present_mode_tests {
    use super::*;

    #[test]
    fn prefers_mailbox_over_fifo() {
        let caps = SurfaceCapabilities {
            formats: vec![TextureFormat::Bgra8Unorm],
            present_modes: vec![PresentMode::Fifo, PresentMode::Mailbox],
            alpha_modes: vec![CompositeAlphaMode::Auto],
            usages: TextureUsages::RENDER_ATTACHMENT,
        };
        // Assertion 29: Mailbox preferred for triple buffering
        assert_eq!(caps.preferred_present_mode(), PresentMode::Mailbox);
    }

    #[test]
    fn falls_back_to_fifo() {
        let caps = SurfaceCapabilities {
            formats: vec![TextureFormat::Bgra8Unorm],
            present_modes: vec![PresentMode::Fifo],
            alpha_modes: vec![CompositeAlphaMode::Auto],
            usages: TextureUsages::RENDER_ATTACHMENT,
        };
        // Assertion 30: Fifo when Mailbox not available
        assert_eq!(caps.preferred_present_mode(), PresentMode::Fifo);
    }

    #[test]
    fn uses_first_available_when_neither_mailbox_nor_fifo() {
        let caps = SurfaceCapabilities {
            formats: vec![TextureFormat::Bgra8Unorm],
            present_modes: vec![PresentMode::Immediate],
            alpha_modes: vec![CompositeAlphaMode::Auto],
            usages: TextureUsages::RENDER_ATTACHMENT,
        };
        // Assertion 31: Immediate when only option
        assert_eq!(caps.preferred_present_mode(), PresentMode::Immediate);
    }

    #[test]
    fn defaults_to_fifo_when_empty() {
        let caps = SurfaceCapabilities {
            formats: vec![TextureFormat::Bgra8Unorm],
            present_modes: vec![],
            alpha_modes: vec![CompositeAlphaMode::Auto],
            usages: TextureUsages::RENDER_ATTACHMENT,
        };
        // Assertion 32: defaults to Fifo when present_modes is empty
        assert_eq!(caps.preferred_present_mode(), PresentMode::Fifo);
    }
}

// ============================================================================
// SECTION 5 -- preferred_alpha_mode() tests
// ============================================================================

mod preferred_alpha_mode_tests {
    use super::*;

    #[test]
    fn prefers_opaque_over_auto() {
        let caps = SurfaceCapabilities {
            formats: vec![TextureFormat::Bgra8Unorm],
            present_modes: vec![PresentMode::Fifo],
            alpha_modes: vec![CompositeAlphaMode::Auto, CompositeAlphaMode::Opaque],
            usages: TextureUsages::RENDER_ATTACHMENT,
        };
        // Assertion 33: Opaque preferred for performance
        assert_eq!(caps.preferred_alpha_mode(), CompositeAlphaMode::Opaque);
    }

    #[test]
    fn falls_back_to_first_available() {
        let caps = SurfaceCapabilities {
            formats: vec![TextureFormat::Bgra8Unorm],
            present_modes: vec![PresentMode::Fifo],
            alpha_modes: vec![CompositeAlphaMode::PreMultiplied],
            usages: TextureUsages::RENDER_ATTACHMENT,
        };
        // Assertion 34: returns first available when Opaque not present
        assert_eq!(caps.preferred_alpha_mode(), CompositeAlphaMode::PreMultiplied);
    }

    #[test]
    fn defaults_to_auto_when_empty() {
        let caps = SurfaceCapabilities {
            formats: vec![TextureFormat::Bgra8Unorm],
            present_modes: vec![PresentMode::Fifo],
            alpha_modes: vec![],
            usages: TextureUsages::RENDER_ATTACHMENT,
        };
        // Assertion 35: defaults to Auto when alpha_modes is empty
        assert_eq!(caps.preferred_alpha_mode(), CompositeAlphaMode::Auto);
    }
}

// ============================================================================
// SECTION 6 -- supports_format() tests
// ============================================================================

mod supports_format_tests {
    use super::*;

    #[test]
    fn returns_true_for_present_format() {
        let caps = SurfaceCapabilities {
            formats: vec![TextureFormat::Bgra8Unorm, TextureFormat::Rgba8Unorm],
            present_modes: vec![PresentMode::Fifo],
            alpha_modes: vec![CompositeAlphaMode::Auto],
            usages: TextureUsages::RENDER_ATTACHMENT,
        };
        // Assertion 36
        assert!(caps.supports_format(TextureFormat::Bgra8Unorm));
        // Assertion 37
        assert!(caps.supports_format(TextureFormat::Rgba8Unorm));
    }

    #[test]
    fn returns_false_for_absent_format() {
        let caps = SurfaceCapabilities {
            formats: vec![TextureFormat::Bgra8Unorm],
            present_modes: vec![PresentMode::Fifo],
            alpha_modes: vec![CompositeAlphaMode::Auto],
            usages: TextureUsages::RENDER_ATTACHMENT,
        };
        // Assertion 38
        assert!(!caps.supports_format(TextureFormat::Rgba16Float));
        // Assertion 39
        assert!(!caps.supports_format(TextureFormat::Rgba8UnormSrgb));
    }

    #[test]
    fn handles_various_texture_formats() {
        let caps = SurfaceCapabilities {
            formats: vec![
                TextureFormat::Bgra8UnormSrgb,
                TextureFormat::Rgba16Float,
                TextureFormat::Rgb10a2Unorm,
            ],
            present_modes: vec![PresentMode::Fifo],
            alpha_modes: vec![CompositeAlphaMode::Auto],
            usages: TextureUsages::RENDER_ATTACHMENT,
        };
        // Assertion 40
        assert!(caps.supports_format(TextureFormat::Bgra8UnormSrgb));
        // Assertion 41
        assert!(caps.supports_format(TextureFormat::Rgba16Float));
        // Assertion 42
        assert!(caps.supports_format(TextureFormat::Rgb10a2Unorm));
        // Assertion 43
        assert!(!caps.supports_format(TextureFormat::Bgra8Unorm));
    }
}

// ============================================================================
// SECTION 7 -- supports_present_mode() tests
// ============================================================================

mod supports_present_mode_tests {
    use super::*;

    #[test]
    fn returns_true_for_present_mode() {
        let caps = SurfaceCapabilities {
            formats: vec![TextureFormat::Bgra8Unorm],
            present_modes: vec![PresentMode::Fifo, PresentMode::Mailbox],
            alpha_modes: vec![CompositeAlphaMode::Auto],
            usages: TextureUsages::RENDER_ATTACHMENT,
        };
        // Assertion 44
        assert!(caps.supports_present_mode(PresentMode::Fifo));
        // Assertion 45
        assert!(caps.supports_present_mode(PresentMode::Mailbox));
    }

    #[test]
    fn returns_false_for_absent_mode() {
        let caps = SurfaceCapabilities {
            formats: vec![TextureFormat::Bgra8Unorm],
            present_modes: vec![PresentMode::Fifo],
            alpha_modes: vec![CompositeAlphaMode::Auto],
            usages: TextureUsages::RENDER_ATTACHMENT,
        };
        // Assertion 46
        assert!(!caps.supports_present_mode(PresentMode::Immediate));
        // Assertion 47
        assert!(!caps.supports_present_mode(PresentMode::Mailbox));
    }

    #[test]
    fn handles_all_present_modes() {
        let caps = SurfaceCapabilities {
            formats: vec![TextureFormat::Bgra8Unorm],
            present_modes: vec![
                PresentMode::Fifo,
                PresentMode::FifoRelaxed,
                PresentMode::Immediate,
                PresentMode::Mailbox,
            ],
            alpha_modes: vec![CompositeAlphaMode::Auto],
            usages: TextureUsages::RENDER_ATTACHMENT,
        };
        // Assertion 48
        assert!(caps.supports_present_mode(PresentMode::Fifo));
        // Assertion 49
        assert!(caps.supports_present_mode(PresentMode::FifoRelaxed));
        // Assertion 50
        assert!(caps.supports_present_mode(PresentMode::Immediate));
        // Assertion 51
        assert!(caps.supports_present_mode(PresentMode::Mailbox));
    }
}

// ============================================================================
// SECTION 8 -- supports_hdr() detection tests
// ============================================================================

mod supports_hdr_tests {
    use super::*;

    #[test]
    fn detects_rgba16float_as_hdr() {
        let caps = SurfaceCapabilities {
            formats: vec![TextureFormat::Bgra8Unorm, TextureFormat::Rgba16Float],
            present_modes: vec![PresentMode::Fifo],
            alpha_modes: vec![CompositeAlphaMode::Auto],
            usages: TextureUsages::RENDER_ATTACHMENT,
        };
        // Assertion 52
        assert!(caps.supports_hdr());
    }

    #[test]
    fn detects_rgb10a2unorm_as_hdr() {
        let caps = SurfaceCapabilities {
            formats: vec![TextureFormat::Bgra8Unorm, TextureFormat::Rgb10a2Unorm],
            present_modes: vec![PresentMode::Fifo],
            alpha_modes: vec![CompositeAlphaMode::Auto],
            usages: TextureUsages::RENDER_ATTACHMENT,
        };
        // Assertion 53
        assert!(caps.supports_hdr());
    }

    #[test]
    fn detects_rg11b10float_as_hdr() {
        let caps = SurfaceCapabilities {
            formats: vec![TextureFormat::Bgra8Unorm, TextureFormat::Rg11b10Float],
            present_modes: vec![PresentMode::Fifo],
            alpha_modes: vec![CompositeAlphaMode::Auto],
            usages: TextureUsages::RENDER_ATTACHMENT,
        };
        // Assertion 54
        assert!(caps.supports_hdr());
    }

    #[test]
    fn returns_false_for_sdr_only() {
        let caps = SurfaceCapabilities {
            formats: vec![
                TextureFormat::Bgra8Unorm,
                TextureFormat::Rgba8Unorm,
                TextureFormat::Bgra8UnormSrgb,
            ],
            present_modes: vec![PresentMode::Fifo],
            alpha_modes: vec![CompositeAlphaMode::Auto],
            usages: TextureUsages::RENDER_ATTACHMENT,
        };
        // Assertion 55
        assert!(!caps.supports_hdr());
    }

    #[test]
    fn returns_false_for_empty_formats() {
        let caps = SurfaceCapabilities {
            formats: vec![],
            present_modes: vec![PresentMode::Fifo],
            alpha_modes: vec![CompositeAlphaMode::Auto],
            usages: TextureUsages::RENDER_ATTACHMENT,
        };
        // Assertion 56
        assert!(!caps.supports_hdr());
    }
}

// ============================================================================
// SECTION 9 -- SurfaceConfiguration::from_capabilities tests
// ============================================================================

mod configuration_from_capabilities_tests {
    use super::*;

    #[test]
    fn uses_preferred_format_from_caps() {
        let caps = SurfaceCapabilities {
            formats: vec![TextureFormat::Rgba8Unorm, TextureFormat::Bgra8UnormSrgb],
            present_modes: vec![PresentMode::Fifo],
            alpha_modes: vec![CompositeAlphaMode::Auto],
            usages: TextureUsages::RENDER_ATTACHMENT,
        };
        let config = SurfaceConfiguration::from_capabilities(&caps, 1920, 1080);
        // Assertion 57: uses preferred format
        assert_eq!(config.format, TextureFormat::Bgra8UnormSrgb);
    }

    #[test]
    fn uses_preferred_present_mode_from_caps() {
        let caps = SurfaceCapabilities {
            formats: vec![TextureFormat::Bgra8Unorm],
            present_modes: vec![PresentMode::Fifo, PresentMode::Mailbox],
            alpha_modes: vec![CompositeAlphaMode::Auto],
            usages: TextureUsages::RENDER_ATTACHMENT,
        };
        let config = SurfaceConfiguration::from_capabilities(&caps, 1920, 1080);
        // Assertion 58: uses preferred present mode (Mailbox)
        assert_eq!(config.present_mode, PresentMode::Mailbox);
    }

    #[test]
    fn uses_preferred_alpha_mode_from_caps() {
        let caps = SurfaceCapabilities {
            formats: vec![TextureFormat::Bgra8Unorm],
            present_modes: vec![PresentMode::Fifo],
            alpha_modes: vec![CompositeAlphaMode::Auto, CompositeAlphaMode::Opaque],
            usages: TextureUsages::RENDER_ATTACHMENT,
        };
        let config = SurfaceConfiguration::from_capabilities(&caps, 1920, 1080);
        // Assertion 59: uses preferred alpha mode (Opaque)
        assert_eq!(config.alpha_mode, CompositeAlphaMode::Opaque);
    }

    #[test]
    fn preserves_dimensions() {
        let caps = SurfaceCapabilities {
            formats: vec![TextureFormat::Bgra8Unorm],
            present_modes: vec![PresentMode::Fifo],
            alpha_modes: vec![CompositeAlphaMode::Auto],
            usages: TextureUsages::RENDER_ATTACHMENT,
        };
        let config = SurfaceConfiguration::from_capabilities(&caps, 2560, 1440);
        // Assertion 60
        assert_eq!(config.width, 2560);
        // Assertion 61
        assert_eq!(config.height, 1440);
    }

    #[test]
    fn clamps_zero_dimensions() {
        let caps = SurfaceCapabilities {
            formats: vec![TextureFormat::Bgra8Unorm],
            present_modes: vec![PresentMode::Fifo],
            alpha_modes: vec![CompositeAlphaMode::Auto],
            usages: TextureUsages::RENDER_ATTACHMENT,
        };
        let config = SurfaceConfiguration::from_capabilities(&caps, 0, 0);
        // Assertion 62
        assert_eq!(config.width, 1);
        // Assertion 63
        assert_eq!(config.height, 1);
    }

    #[test]
    fn uses_default_format_when_formats_empty() {
        let caps = SurfaceCapabilities {
            formats: vec![],
            present_modes: vec![PresentMode::Fifo],
            alpha_modes: vec![CompositeAlphaMode::Auto],
            usages: TextureUsages::RENDER_ATTACHMENT,
        };
        let config = SurfaceConfiguration::from_capabilities(&caps, 800, 600);
        // Assertion 64: falls back to default format
        assert_eq!(config.format, TextureFormat::Bgra8UnormSrgb);
    }

    #[test]
    fn validates_successfully_with_matching_capabilities() {
        let caps = SurfaceCapabilities {
            formats: vec![TextureFormat::Bgra8UnormSrgb],
            present_modes: vec![PresentMode::Mailbox],
            alpha_modes: vec![CompositeAlphaMode::Opaque],
            usages: TextureUsages::RENDER_ATTACHMENT,
        };
        let config = SurfaceConfiguration::from_capabilities(&caps, 1920, 1080);
        // Assertion 65: configuration validates against same capabilities
        assert!(config.validate(&caps).is_ok());
    }
}

// ============================================================================
// SECTION 10 -- Clone and Debug trait tests
// ============================================================================

mod trait_tests {
    use super::*;

    #[test]
    fn surface_capabilities_clone_preserves_all_fields() {
        let caps = SurfaceCapabilities {
            formats: vec![TextureFormat::Bgra8Unorm, TextureFormat::Rgba16Float],
            present_modes: vec![PresentMode::Fifo, PresentMode::Mailbox],
            alpha_modes: vec![CompositeAlphaMode::Auto, CompositeAlphaMode::Opaque],
            usages: TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC,
        };
        let cloned = caps.clone();
        // Assertion 66
        assert_eq!(caps.formats, cloned.formats);
        // Assertion 67
        assert_eq!(caps.present_modes, cloned.present_modes);
        // Assertion 68
        assert_eq!(caps.alpha_modes, cloned.alpha_modes);
        // Assertion 69
        assert_eq!(caps.usages, cloned.usages);
    }

    #[test]
    fn surface_capabilities_debug_contains_format_info() {
        let caps = SurfaceCapabilities {
            formats: vec![TextureFormat::Bgra8Unorm],
            present_modes: vec![PresentMode::Fifo],
            alpha_modes: vec![CompositeAlphaMode::Auto],
            usages: TextureUsages::RENDER_ATTACHMENT,
        };
        let debug_str = format!("{:?}", caps);
        // Assertion 70: Debug output contains format information
        assert!(debug_str.contains("Bgra8Unorm"));
    }

    #[test]
    fn surface_capabilities_debug_contains_present_mode_info() {
        let caps = SurfaceCapabilities {
            formats: vec![TextureFormat::Bgra8Unorm],
            present_modes: vec![PresentMode::Mailbox],
            alpha_modes: vec![CompositeAlphaMode::Auto],
            usages: TextureUsages::RENDER_ATTACHMENT,
        };
        let debug_str = format!("{:?}", caps);
        // Assertion 71: Debug output contains present mode information
        assert!(debug_str.contains("Mailbox"));
    }
}

// ============================================================================
// SECTION 11 -- Integration: capabilities + configuration workflow
// ============================================================================

mod workflow_tests {
    use super::*;

    #[test]
    fn full_workflow_query_to_configure() {
        // Simulate querying capabilities and creating configuration
        let wgpu_caps = wgpu::SurfaceCapabilities {
            formats: vec![TextureFormat::Bgra8UnormSrgb, TextureFormat::Rgba8Unorm],
            present_modes: vec![PresentMode::Fifo, PresentMode::Mailbox],
            alpha_modes: vec![CompositeAlphaMode::Auto, CompositeAlphaMode::Opaque],
            usages: TextureUsages::RENDER_ATTACHMENT,
        };

        let caps = SurfaceCapabilities::from_wgpu(wgpu_caps);

        // Assertion 72: Query formats
        assert!(caps.supports_format(TextureFormat::Bgra8UnormSrgb));
        // Assertion 73: Query present modes
        assert!(caps.supports_present_mode(PresentMode::Mailbox));
        // Assertion 74: Check HDR support
        assert!(!caps.supports_hdr());

        let config = SurfaceConfiguration::from_capabilities(&caps, 1920, 1080);

        // Assertion 75: Config uses best format
        assert_eq!(config.format, TextureFormat::Bgra8UnormSrgb);
        // Assertion 76: Config uses best present mode
        assert_eq!(config.present_mode, PresentMode::Mailbox);
        // Assertion 77: Config uses best alpha mode
        assert_eq!(config.alpha_mode, CompositeAlphaMode::Opaque);
        // Assertion 78: Config has correct dimensions
        assert_eq!(config.width, 1920);
        // Assertion 79
        assert_eq!(config.height, 1080);

        // Assertion 80: Config validates successfully
        assert!(config.validate(&caps).is_ok());
    }

    #[test]
    fn configuration_with_builder_overrides() {
        let caps = SurfaceCapabilities {
            formats: vec![
                TextureFormat::Bgra8UnormSrgb,
                TextureFormat::Rgba8Unorm,
                TextureFormat::Rgba16Float,
            ],
            present_modes: vec![PresentMode::Fifo, PresentMode::Immediate],
            alpha_modes: vec![CompositeAlphaMode::Auto, CompositeAlphaMode::PreMultiplied],
            usages: TextureUsages::RENDER_ATTACHMENT,
        };

        // Start with capabilities, then override
        let config = SurfaceConfiguration::from_capabilities(&caps, 800, 600)
            .with_format(TextureFormat::Rgba16Float) // Override to HDR
            .with_present_mode(PresentMode::Immediate) // Override to no vsync
            .with_alpha_mode(CompositeAlphaMode::PreMultiplied);

        // Assertion 81: Override format works
        assert_eq!(config.format, TextureFormat::Rgba16Float);
        // Assertion 82: Override present mode works
        assert_eq!(config.present_mode, PresentMode::Immediate);
        // Assertion 83: Override alpha mode works
        assert_eq!(config.alpha_mode, CompositeAlphaMode::PreMultiplied);

        // Assertion 84: Still validates because overrides are in capabilities
        assert!(config.validate(&caps).is_ok());
    }

    #[test]
    fn configuration_to_wgpu_conversion() {
        let caps = SurfaceCapabilities {
            formats: vec![TextureFormat::Bgra8UnormSrgb],
            present_modes: vec![PresentMode::Mailbox],
            alpha_modes: vec![CompositeAlphaMode::Opaque],
            usages: TextureUsages::RENDER_ATTACHMENT,
        };

        let config = SurfaceConfiguration::from_capabilities(&caps, 1280, 720);
        let wgpu_config = config.to_wgpu();

        // Assertion 85: wgpu config has correct format
        assert_eq!(wgpu_config.format, TextureFormat::Bgra8UnormSrgb);
        // Assertion 86: wgpu config has correct width
        assert_eq!(wgpu_config.width, 1280);
        // Assertion 87: wgpu config has correct height
        assert_eq!(wgpu_config.height, 720);
        // Assertion 88: wgpu config has correct present mode
        assert_eq!(wgpu_config.present_mode, PresentMode::Mailbox);
        // Assertion 89: wgpu config has correct alpha mode
        assert_eq!(wgpu_config.alpha_mode, CompositeAlphaMode::Opaque);
        // Assertion 90: wgpu config has RENDER_ATTACHMENT usage
        assert!(wgpu_config.usage.contains(TextureUsages::RENDER_ATTACHMENT));
    }
}
