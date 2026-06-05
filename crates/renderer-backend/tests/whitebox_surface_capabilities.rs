// WHITEBOX tests for T-WGPU-P7.1.2 (Surface Capabilities)
//
// WHITEBOX discipline: These tests have FULL ACCESS to the implementation.
// They exercise internal code paths, branch conditions, and edge cases
// that are not visible through the public contract alone.
//
// Implementation under test: crates/renderer-backend/src/presentation/surface.rs
//   - SurfaceCapabilities: formats, present_modes, alpha_modes, usages
//   - preferred_format(): sRGB preference (Bgra8UnormSrgb > Rgba8UnormSrgb > fallback)
//   - preferred_present_mode(): Mailbox > Fifo fallback
//   - preferred_alpha_mode(): Opaque preference
//   - supports_format(): Query format support
//   - supports_present_mode(): Query present mode support
//   - supports_hdr(): HDR format detection (Rgba16Float, Rgb10a2Unorm, Rg11b10Float)
//   - SurfaceConfiguration: format, width, height, present_mode, alpha_mode, frame_latency
//   - SurfaceConfiguration::validate(): Validation against capabilities
//   - PlatformTarget: Platform detection enum
//   - SurfaceError: Error types for surface operations
//
// WHITEBOX coverage plan:
//   - Path A: preferred_format() with Bgra8UnormSrgb available (first sRGB)
//   - Path B: preferred_format() with only Rgba8UnormSrgb (second sRGB)
//   - Path C: preferred_format() with no sRGB, only Bgra8Unorm (first linear)
//   - Path D: preferred_format() with no sRGB, only Rgba8Unorm (second linear)
//   - Path E: preferred_format() with none of the preferred formats (first available)
//   - Path F: preferred_format() with empty formats list (None)
//   - Path G: preferred_present_mode() with Mailbox available
//   - Path H: preferred_present_mode() with only Fifo available
//   - Path I: preferred_present_mode() with neither (first available fallback)
//   - Path J: preferred_present_mode() with empty list (Fifo default)
//   - Path K: preferred_alpha_mode() with Opaque available
//   - Path L: preferred_alpha_mode() without Opaque (first available)
//   - Path M: preferred_alpha_mode() with empty list (Auto default)
//   - Path N: supports_format() true case
//   - Path O: supports_format() false case
//   - Path P: supports_present_mode() true case
//   - Path Q: supports_present_mode() false case
//   - Path R: supports_hdr() with Rgba16Float
//   - Path S: supports_hdr() with Rgb10a2Unorm
//   - Path T: supports_hdr() with Rg11b10Float
//   - Path U: supports_hdr() without any HDR formats
//   - Path V: SurfaceConfiguration::new() dimension clamping
//   - Path W: SurfaceConfiguration::from_capabilities()
//   - Path X: SurfaceConfiguration builder methods
//   - Path Y: SurfaceConfiguration::validate() success
//   - Path Z: SurfaceConfiguration::validate() bad format
//   - Path AA: SurfaceConfiguration::validate() bad present mode
//   - Path AB: SurfaceConfiguration::validate() bad alpha mode
//   - Path AC: SurfaceConfiguration::validate() zero dimensions
//   - Path AD: SurfaceConfiguration::to_wgpu() conversion
//   - Path AE: SurfaceConfiguration::default()
//   - Path AF: SurfaceCapabilities::from_wgpu()
//   - Path AG: PlatformTarget name() for all variants
//   - Path AH: PlatformTarget is_supported() for all variants
//   - Path AI: PlatformTarget Display trait
//   - Path AJ: SurfaceError error messages and recoverability
//   - Path AK: SurfaceError helper constructors
//   - Path AL: SurfaceError is_platform_error()

use renderer_backend::presentation::{
    PlatformTarget, SurfaceCapabilities, SurfaceConfiguration, SurfaceError,
};
use wgpu::{CompositeAlphaMode, PresentMode, TextureFormat, TextureUsages};

// ============================================================================
// SurfaceCapabilities - preferred_format() Tests
// ============================================================================

#[test]
fn test_preferred_format_bgra8_srgb_first_choice() {
    // Path A: Bgra8UnormSrgb is the first preference
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
    assert_eq!(caps.preferred_format(), Some(TextureFormat::Bgra8UnormSrgb));
}

#[test]
fn test_preferred_format_rgba8_srgb_second_choice() {
    // Path B: Rgba8UnormSrgb is second preference when Bgra8UnormSrgb not available
    let caps = SurfaceCapabilities {
        formats: vec![
            TextureFormat::Rgba8Unorm,
            TextureFormat::Rgba8UnormSrgb,
        ],
        present_modes: vec![PresentMode::Fifo],
        alpha_modes: vec![CompositeAlphaMode::Auto],
        usages: TextureUsages::RENDER_ATTACHMENT,
    };
    assert_eq!(caps.preferred_format(), Some(TextureFormat::Rgba8UnormSrgb));
}

#[test]
fn test_preferred_format_bgra8_linear_fallback() {
    // Path C: Bgra8Unorm is first linear fallback when no sRGB available
    let caps = SurfaceCapabilities {
        formats: vec![
            TextureFormat::Rgba8Unorm,
            TextureFormat::Bgra8Unorm,
        ],
        present_modes: vec![PresentMode::Fifo],
        alpha_modes: vec![CompositeAlphaMode::Auto],
        usages: TextureUsages::RENDER_ATTACHMENT,
    };
    assert_eq!(caps.preferred_format(), Some(TextureFormat::Bgra8Unorm));
}

#[test]
fn test_preferred_format_rgba8_linear_fallback() {
    // Path D: Rgba8Unorm is second linear fallback
    let caps = SurfaceCapabilities {
        formats: vec![
            TextureFormat::Rgba8Unorm,
            TextureFormat::R8Unorm,
        ],
        present_modes: vec![PresentMode::Fifo],
        alpha_modes: vec![CompositeAlphaMode::Auto],
        usages: TextureUsages::RENDER_ATTACHMENT,
    };
    assert_eq!(caps.preferred_format(), Some(TextureFormat::Rgba8Unorm));
}

#[test]
fn test_preferred_format_first_available() {
    // Path E: First available format when none of the preferred are present
    let caps = SurfaceCapabilities {
        formats: vec![
            TextureFormat::R8Unorm,
            TextureFormat::Rg8Unorm,
        ],
        present_modes: vec![PresentMode::Fifo],
        alpha_modes: vec![CompositeAlphaMode::Auto],
        usages: TextureUsages::RENDER_ATTACHMENT,
    };
    assert_eq!(caps.preferred_format(), Some(TextureFormat::R8Unorm));
}

#[test]
fn test_preferred_format_empty_list() {
    // Path F: Empty formats list returns None
    let caps = SurfaceCapabilities {
        formats: vec![],
        present_modes: vec![PresentMode::Fifo],
        alpha_modes: vec![CompositeAlphaMode::Auto],
        usages: TextureUsages::RENDER_ATTACHMENT,
    };
    assert_eq!(caps.preferred_format(), None);
}

#[test]
fn test_preferred_format_srgb_preferred_over_linear() {
    // Verify sRGB is always chosen over linear when available
    let caps = SurfaceCapabilities {
        formats: vec![
            TextureFormat::Bgra8Unorm,      // Linear first in list
            TextureFormat::Bgra8UnormSrgb,  // sRGB second in list
        ],
        present_modes: vec![PresentMode::Fifo],
        alpha_modes: vec![CompositeAlphaMode::Auto],
        usages: TextureUsages::RENDER_ATTACHMENT,
    };
    // Should prefer sRGB even though linear is first
    assert_eq!(caps.preferred_format(), Some(TextureFormat::Bgra8UnormSrgb));
}

#[test]
fn test_preferred_format_bgra_srgb_over_rgba_srgb() {
    // Verify Bgra8UnormSrgb is chosen over Rgba8UnormSrgb
    let caps = SurfaceCapabilities {
        formats: vec![
            TextureFormat::Rgba8UnormSrgb,  // Second sRGB first in list
            TextureFormat::Bgra8UnormSrgb,  // First sRGB second in list
        ],
        present_modes: vec![PresentMode::Fifo],
        alpha_modes: vec![CompositeAlphaMode::Auto],
        usages: TextureUsages::RENDER_ATTACHMENT,
    };
    // Should prefer Bgra8UnormSrgb
    assert_eq!(caps.preferred_format(), Some(TextureFormat::Bgra8UnormSrgb));
}

// ============================================================================
// SurfaceCapabilities - preferred_present_mode() Tests
// ============================================================================

#[test]
fn test_preferred_present_mode_mailbox_first_choice() {
    // Path G: Mailbox is preferred for smooth triple buffering
    let caps = SurfaceCapabilities {
        formats: vec![TextureFormat::Bgra8Unorm],
        present_modes: vec![
            PresentMode::Fifo,
            PresentMode::Mailbox,
            PresentMode::Immediate,
        ],
        alpha_modes: vec![CompositeAlphaMode::Auto],
        usages: TextureUsages::RENDER_ATTACHMENT,
    };
    assert_eq!(caps.preferred_present_mode(), PresentMode::Mailbox);
}

#[test]
fn test_preferred_present_mode_fifo_fallback() {
    // Path H: Fifo is fallback when Mailbox not available
    let caps = SurfaceCapabilities {
        formats: vec![TextureFormat::Bgra8Unorm],
        present_modes: vec![
            PresentMode::Immediate,
            PresentMode::Fifo,
        ],
        alpha_modes: vec![CompositeAlphaMode::Auto],
        usages: TextureUsages::RENDER_ATTACHMENT,
    };
    assert_eq!(caps.preferred_present_mode(), PresentMode::Fifo);
}

#[test]
fn test_preferred_present_mode_first_available() {
    // Path I: First available when neither Mailbox nor Fifo
    let caps = SurfaceCapabilities {
        formats: vec![TextureFormat::Bgra8Unorm],
        present_modes: vec![PresentMode::Immediate],
        alpha_modes: vec![CompositeAlphaMode::Auto],
        usages: TextureUsages::RENDER_ATTACHMENT,
    };
    assert_eq!(caps.preferred_present_mode(), PresentMode::Immediate);
}

#[test]
fn test_preferred_present_mode_empty_list() {
    // Path J: Empty list defaults to Fifo
    let caps = SurfaceCapabilities {
        formats: vec![TextureFormat::Bgra8Unorm],
        present_modes: vec![],
        alpha_modes: vec![CompositeAlphaMode::Auto],
        usages: TextureUsages::RENDER_ATTACHMENT,
    };
    assert_eq!(caps.preferred_present_mode(), PresentMode::Fifo);
}

#[test]
fn test_preferred_present_mode_mailbox_over_fifo() {
    // Verify Mailbox is chosen over Fifo when both available
    let caps = SurfaceCapabilities {
        formats: vec![TextureFormat::Bgra8Unorm],
        present_modes: vec![
            PresentMode::Fifo,      // First in list
            PresentMode::Mailbox,   // Second in list but preferred
        ],
        alpha_modes: vec![CompositeAlphaMode::Auto],
        usages: TextureUsages::RENDER_ATTACHMENT,
    };
    assert_eq!(caps.preferred_present_mode(), PresentMode::Mailbox);
}

#[test]
fn test_preferred_present_mode_fifo_relaxed_not_preferred() {
    // FifoRelaxed is not in the preference list
    let caps = SurfaceCapabilities {
        formats: vec![TextureFormat::Bgra8Unorm],
        present_modes: vec![PresentMode::FifoRelaxed],
        alpha_modes: vec![CompositeAlphaMode::Auto],
        usages: TextureUsages::RENDER_ATTACHMENT,
    };
    // Should fall back to first available (FifoRelaxed)
    assert_eq!(caps.preferred_present_mode(), PresentMode::FifoRelaxed);
}

// ============================================================================
// SurfaceCapabilities - preferred_alpha_mode() Tests
// ============================================================================

#[test]
fn test_preferred_alpha_mode_opaque_first_choice() {
    // Path K: Opaque is preferred for best performance
    let caps = SurfaceCapabilities {
        formats: vec![TextureFormat::Bgra8Unorm],
        present_modes: vec![PresentMode::Fifo],
        alpha_modes: vec![
            CompositeAlphaMode::Auto,
            CompositeAlphaMode::Opaque,
            CompositeAlphaMode::PreMultiplied,
        ],
        usages: TextureUsages::RENDER_ATTACHMENT,
    };
    assert_eq!(caps.preferred_alpha_mode(), CompositeAlphaMode::Opaque);
}

#[test]
fn test_preferred_alpha_mode_first_available() {
    // Path L: First available when Opaque not present
    let caps = SurfaceCapabilities {
        formats: vec![TextureFormat::Bgra8Unorm],
        present_modes: vec![PresentMode::Fifo],
        alpha_modes: vec![
            CompositeAlphaMode::PreMultiplied,
            CompositeAlphaMode::PostMultiplied,
        ],
        usages: TextureUsages::RENDER_ATTACHMENT,
    };
    assert_eq!(caps.preferred_alpha_mode(), CompositeAlphaMode::PreMultiplied);
}

#[test]
fn test_preferred_alpha_mode_empty_list() {
    // Path M: Empty list defaults to Auto
    let caps = SurfaceCapabilities {
        formats: vec![TextureFormat::Bgra8Unorm],
        present_modes: vec![PresentMode::Fifo],
        alpha_modes: vec![],
        usages: TextureUsages::RENDER_ATTACHMENT,
    };
    assert_eq!(caps.preferred_alpha_mode(), CompositeAlphaMode::Auto);
}

#[test]
fn test_preferred_alpha_mode_opaque_over_auto() {
    // Verify Opaque is chosen over Auto when both available
    let caps = SurfaceCapabilities {
        formats: vec![TextureFormat::Bgra8Unorm],
        present_modes: vec![PresentMode::Fifo],
        alpha_modes: vec![
            CompositeAlphaMode::Auto,    // First in list
            CompositeAlphaMode::Opaque,  // Preferred
        ],
        usages: TextureUsages::RENDER_ATTACHMENT,
    };
    assert_eq!(caps.preferred_alpha_mode(), CompositeAlphaMode::Opaque);
}

// ============================================================================
// SurfaceCapabilities - supports_format() Tests
// ============================================================================

#[test]
fn test_supports_format_true() {
    // Path N: Format is in the list
    let caps = SurfaceCapabilities {
        formats: vec![
            TextureFormat::Bgra8Unorm,
            TextureFormat::Rgba8Unorm,
        ],
        present_modes: vec![PresentMode::Fifo],
        alpha_modes: vec![CompositeAlphaMode::Auto],
        usages: TextureUsages::RENDER_ATTACHMENT,
    };
    assert!(caps.supports_format(TextureFormat::Bgra8Unorm));
    assert!(caps.supports_format(TextureFormat::Rgba8Unorm));
}

#[test]
fn test_supports_format_false() {
    // Path O: Format is not in the list
    let caps = SurfaceCapabilities {
        formats: vec![TextureFormat::Bgra8Unorm],
        present_modes: vec![PresentMode::Fifo],
        alpha_modes: vec![CompositeAlphaMode::Auto],
        usages: TextureUsages::RENDER_ATTACHMENT,
    };
    assert!(!caps.supports_format(TextureFormat::Rgba16Float));
    assert!(!caps.supports_format(TextureFormat::Rgba8UnormSrgb));
}

#[test]
fn test_supports_format_empty_list() {
    // Empty format list returns false for all
    let caps = SurfaceCapabilities {
        formats: vec![],
        present_modes: vec![PresentMode::Fifo],
        alpha_modes: vec![CompositeAlphaMode::Auto],
        usages: TextureUsages::RENDER_ATTACHMENT,
    };
    assert!(!caps.supports_format(TextureFormat::Bgra8Unorm));
}

// ============================================================================
// SurfaceCapabilities - supports_present_mode() Tests
// ============================================================================

#[test]
fn test_supports_present_mode_true() {
    // Path P: Present mode is in the list
    let caps = SurfaceCapabilities {
        formats: vec![TextureFormat::Bgra8Unorm],
        present_modes: vec![
            PresentMode::Fifo,
            PresentMode::Mailbox,
        ],
        alpha_modes: vec![CompositeAlphaMode::Auto],
        usages: TextureUsages::RENDER_ATTACHMENT,
    };
    assert!(caps.supports_present_mode(PresentMode::Fifo));
    assert!(caps.supports_present_mode(PresentMode::Mailbox));
}

#[test]
fn test_supports_present_mode_false() {
    // Path Q: Present mode is not in the list
    let caps = SurfaceCapabilities {
        formats: vec![TextureFormat::Bgra8Unorm],
        present_modes: vec![PresentMode::Fifo],
        alpha_modes: vec![CompositeAlphaMode::Auto],
        usages: TextureUsages::RENDER_ATTACHMENT,
    };
    assert!(!caps.supports_present_mode(PresentMode::Mailbox));
    assert!(!caps.supports_present_mode(PresentMode::Immediate));
}

#[test]
fn test_supports_present_mode_empty_list() {
    // Empty present modes list returns false for all
    let caps = SurfaceCapabilities {
        formats: vec![TextureFormat::Bgra8Unorm],
        present_modes: vec![],
        alpha_modes: vec![CompositeAlphaMode::Auto],
        usages: TextureUsages::RENDER_ATTACHMENT,
    };
    assert!(!caps.supports_present_mode(PresentMode::Fifo));
}

// ============================================================================
// SurfaceCapabilities - supports_hdr() Tests
// ============================================================================

#[test]
fn test_supports_hdr_rgba16_float() {
    // Path R: Rgba16Float is an HDR format
    let caps = SurfaceCapabilities {
        formats: vec![
            TextureFormat::Bgra8Unorm,
            TextureFormat::Rgba16Float,
        ],
        present_modes: vec![PresentMode::Fifo],
        alpha_modes: vec![CompositeAlphaMode::Auto],
        usages: TextureUsages::RENDER_ATTACHMENT,
    };
    assert!(caps.supports_hdr());
}

#[test]
fn test_supports_hdr_rgb10a2_unorm() {
    // Path S: Rgb10a2Unorm is an HDR format
    let caps = SurfaceCapabilities {
        formats: vec![
            TextureFormat::Bgra8Unorm,
            TextureFormat::Rgb10a2Unorm,
        ],
        present_modes: vec![PresentMode::Fifo],
        alpha_modes: vec![CompositeAlphaMode::Auto],
        usages: TextureUsages::RENDER_ATTACHMENT,
    };
    assert!(caps.supports_hdr());
}

#[test]
fn test_supports_hdr_rg11b10_float() {
    // Path T: Rg11b10Float is an HDR format
    let caps = SurfaceCapabilities {
        formats: vec![
            TextureFormat::Bgra8Unorm,
            TextureFormat::Rg11b10Float,
        ],
        present_modes: vec![PresentMode::Fifo],
        alpha_modes: vec![CompositeAlphaMode::Auto],
        usages: TextureUsages::RENDER_ATTACHMENT,
    };
    assert!(caps.supports_hdr());
}

#[test]
fn test_supports_hdr_no_hdr_formats() {
    // Path U: No HDR formats present
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
    assert!(!caps.supports_hdr());
}

#[test]
fn test_supports_hdr_empty_formats() {
    // Empty format list has no HDR support
    let caps = SurfaceCapabilities {
        formats: vec![],
        present_modes: vec![PresentMode::Fifo],
        alpha_modes: vec![CompositeAlphaMode::Auto],
        usages: TextureUsages::RENDER_ATTACHMENT,
    };
    assert!(!caps.supports_hdr());
}

#[test]
fn test_supports_hdr_all_hdr_formats() {
    // All three HDR formats present
    let caps = SurfaceCapabilities {
        formats: vec![
            TextureFormat::Rgba16Float,
            TextureFormat::Rgb10a2Unorm,
            TextureFormat::Rg11b10Float,
        ],
        present_modes: vec![PresentMode::Fifo],
        alpha_modes: vec![CompositeAlphaMode::Auto],
        usages: TextureUsages::RENDER_ATTACHMENT,
    };
    assert!(caps.supports_hdr());
}

// ============================================================================
// SurfaceConfiguration - new() Tests
// ============================================================================

#[test]
fn test_surface_configuration_new_normal_dimensions() {
    // Path V: Normal dimensions
    let config = SurfaceConfiguration::new(1920, 1080);
    assert_eq!(config.width, 1920);
    assert_eq!(config.height, 1080);
    assert_eq!(config.format, TextureFormat::Bgra8UnormSrgb);
    assert_eq!(config.present_mode, PresentMode::Fifo);
    assert_eq!(config.alpha_mode, CompositeAlphaMode::Auto);
    assert_eq!(config.desired_maximum_frame_latency, 2);
}

#[test]
fn test_surface_configuration_new_clamps_zero_width() {
    // Path V: Zero width is clamped to 1
    let config = SurfaceConfiguration::new(0, 1080);
    assert_eq!(config.width, 1);
    assert_eq!(config.height, 1080);
}

#[test]
fn test_surface_configuration_new_clamps_zero_height() {
    // Path V: Zero height is clamped to 1
    let config = SurfaceConfiguration::new(1920, 0);
    assert_eq!(config.width, 1920);
    assert_eq!(config.height, 1);
}

#[test]
fn test_surface_configuration_new_clamps_both_zero() {
    // Path V: Both zero are clamped to 1
    let config = SurfaceConfiguration::new(0, 0);
    assert_eq!(config.width, 1);
    assert_eq!(config.height, 1);
}

#[test]
fn test_surface_configuration_new_large_dimensions() {
    // Large but valid dimensions
    let config = SurfaceConfiguration::new(7680, 4320);
    assert_eq!(config.width, 7680);  // 8K width
    assert_eq!(config.height, 4320); // 8K height
}

// ============================================================================
// SurfaceConfiguration - from_capabilities() Tests
// ============================================================================

#[test]
fn test_surface_configuration_from_capabilities_basic() {
    // Path W: Create config from capabilities
    let caps = SurfaceCapabilities {
        formats: vec![TextureFormat::Bgra8UnormSrgb],
        present_modes: vec![PresentMode::Mailbox],
        alpha_modes: vec![CompositeAlphaMode::Opaque],
        usages: TextureUsages::RENDER_ATTACHMENT,
    };
    let config = SurfaceConfiguration::from_capabilities(&caps, 1280, 720);
    assert_eq!(config.format, TextureFormat::Bgra8UnormSrgb);
    assert_eq!(config.width, 1280);
    assert_eq!(config.height, 720);
    assert_eq!(config.present_mode, PresentMode::Mailbox);
    assert_eq!(config.alpha_mode, CompositeAlphaMode::Opaque);
}

#[test]
fn test_surface_configuration_from_capabilities_zero_dimensions() {
    // from_capabilities also clamps zero dimensions
    let caps = SurfaceCapabilities {
        formats: vec![TextureFormat::Bgra8Unorm],
        present_modes: vec![PresentMode::Fifo],
        alpha_modes: vec![CompositeAlphaMode::Auto],
        usages: TextureUsages::RENDER_ATTACHMENT,
    };
    let config = SurfaceConfiguration::from_capabilities(&caps, 0, 0);
    assert_eq!(config.width, 1);
    assert_eq!(config.height, 1);
}

#[test]
fn test_surface_configuration_from_capabilities_empty_formats() {
    // Empty formats falls back to default
    let caps = SurfaceCapabilities {
        formats: vec![],
        present_modes: vec![PresentMode::Fifo],
        alpha_modes: vec![CompositeAlphaMode::Auto],
        usages: TextureUsages::RENDER_ATTACHMENT,
    };
    let config = SurfaceConfiguration::from_capabilities(&caps, 800, 600);
    assert_eq!(config.format, TextureFormat::Bgra8UnormSrgb); // Default fallback
}

// ============================================================================
// SurfaceConfiguration - Builder Methods Tests
// ============================================================================

#[test]
fn test_surface_configuration_with_format() {
    // Path X: Builder method for format
    let config = SurfaceConfiguration::new(640, 480)
        .with_format(TextureFormat::Rgba8Unorm);
    assert_eq!(config.format, TextureFormat::Rgba8Unorm);
    assert_eq!(config.width, 640);
    assert_eq!(config.height, 480);
}

#[test]
fn test_surface_configuration_with_present_mode() {
    // Path X: Builder method for present mode
    let config = SurfaceConfiguration::new(640, 480)
        .with_present_mode(PresentMode::Immediate);
    assert_eq!(config.present_mode, PresentMode::Immediate);
}

#[test]
fn test_surface_configuration_with_alpha_mode() {
    // Path X: Builder method for alpha mode
    let config = SurfaceConfiguration::new(640, 480)
        .with_alpha_mode(CompositeAlphaMode::PreMultiplied);
    assert_eq!(config.alpha_mode, CompositeAlphaMode::PreMultiplied);
}

#[test]
fn test_surface_configuration_with_frame_latency() {
    // Path X: Builder method for frame latency
    let config = SurfaceConfiguration::new(640, 480)
        .with_frame_latency(3);
    assert_eq!(config.desired_maximum_frame_latency, 3);
}

#[test]
fn test_surface_configuration_with_frame_latency_clamps_zero() {
    // Frame latency of 0 is clamped to 1
    let config = SurfaceConfiguration::new(640, 480)
        .with_frame_latency(0);
    assert_eq!(config.desired_maximum_frame_latency, 1);
}

#[test]
fn test_surface_configuration_chained_builders() {
    // All builder methods can be chained
    let config = SurfaceConfiguration::new(1920, 1080)
        .with_format(TextureFormat::Rgba16Float)
        .with_present_mode(PresentMode::Mailbox)
        .with_alpha_mode(CompositeAlphaMode::Opaque)
        .with_frame_latency(4);

    assert_eq!(config.width, 1920);
    assert_eq!(config.height, 1080);
    assert_eq!(config.format, TextureFormat::Rgba16Float);
    assert_eq!(config.present_mode, PresentMode::Mailbox);
    assert_eq!(config.alpha_mode, CompositeAlphaMode::Opaque);
    assert_eq!(config.desired_maximum_frame_latency, 4);
}

// ============================================================================
// SurfaceConfiguration - validate() Tests
// ============================================================================

#[test]
fn test_surface_configuration_validate_success() {
    // Path Y: Valid configuration passes validation
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

#[test]
fn test_surface_configuration_validate_bad_format() {
    // Path Z: Unsupported format fails validation
    let caps = SurfaceCapabilities {
        formats: vec![TextureFormat::Bgra8Unorm],
        present_modes: vec![PresentMode::Fifo],
        alpha_modes: vec![CompositeAlphaMode::Auto],
        usages: TextureUsages::RENDER_ATTACHMENT,
    };
    let config = SurfaceConfiguration::new(800, 600)
        .with_format(TextureFormat::Rgba16Float);

    let result = config.validate(&caps);
    assert!(result.is_err());
    let err_str = format!("{}", result.unwrap_err());
    assert!(err_str.contains("format"));
    assert!(err_str.contains("not supported"));
}

#[test]
fn test_surface_configuration_validate_bad_present_mode() {
    // Path AA: Unsupported present mode fails validation
    let caps = SurfaceCapabilities {
        formats: vec![TextureFormat::Bgra8Unorm],
        present_modes: vec![PresentMode::Fifo],
        alpha_modes: vec![CompositeAlphaMode::Auto],
        usages: TextureUsages::RENDER_ATTACHMENT,
    };
    let config = SurfaceConfiguration::new(800, 600)
        .with_format(TextureFormat::Bgra8Unorm)
        .with_present_mode(PresentMode::Mailbox);

    let result = config.validate(&caps);
    assert!(result.is_err());
    let err_str = format!("{}", result.unwrap_err());
    assert!(err_str.contains("present mode"));
    assert!(err_str.contains("not supported"));
}

#[test]
fn test_surface_configuration_validate_bad_alpha_mode() {
    // Path AB: Unsupported alpha mode fails validation
    let caps = SurfaceCapabilities {
        formats: vec![TextureFormat::Bgra8Unorm],
        present_modes: vec![PresentMode::Fifo],
        alpha_modes: vec![CompositeAlphaMode::Auto],
        usages: TextureUsages::RENDER_ATTACHMENT,
    };
    let config = SurfaceConfiguration::new(800, 600)
        .with_format(TextureFormat::Bgra8Unorm)
        .with_present_mode(PresentMode::Fifo)
        .with_alpha_mode(CompositeAlphaMode::Opaque);

    let result = config.validate(&caps);
    assert!(result.is_err());
    let err_str = format!("{}", result.unwrap_err());
    assert!(err_str.contains("alpha mode"));
    assert!(err_str.contains("not supported"));
}

#[test]
fn test_surface_configuration_validate_multiple_formats_supported() {
    // Validation passes when format is one of multiple supported
    let caps = SurfaceCapabilities {
        formats: vec![
            TextureFormat::Bgra8Unorm,
            TextureFormat::Rgba8Unorm,
            TextureFormat::Bgra8UnormSrgb,
        ],
        present_modes: vec![PresentMode::Fifo, PresentMode::Mailbox],
        alpha_modes: vec![CompositeAlphaMode::Auto, CompositeAlphaMode::Opaque],
        usages: TextureUsages::RENDER_ATTACHMENT,
    };

    // Test with each supported format
    for format in &caps.formats {
        let config = SurfaceConfiguration::new(800, 600)
            .with_format(*format)
            .with_present_mode(PresentMode::Fifo)
            .with_alpha_mode(CompositeAlphaMode::Auto);
        assert!(config.validate(&caps).is_ok());
    }
}

// ============================================================================
// SurfaceConfiguration - to_wgpu() Tests
// ============================================================================

#[test]
fn test_surface_configuration_to_wgpu() {
    // Path AD: Convert to wgpu configuration
    let config = SurfaceConfiguration::new(1280, 720)
        .with_format(TextureFormat::Bgra8Unorm)
        .with_present_mode(PresentMode::Mailbox)
        .with_alpha_mode(CompositeAlphaMode::Opaque)
        .with_frame_latency(2);

    let wgpu_config = config.to_wgpu();

    assert_eq!(wgpu_config.format, TextureFormat::Bgra8Unorm);
    assert_eq!(wgpu_config.width, 1280);
    assert_eq!(wgpu_config.height, 720);
    assert_eq!(wgpu_config.present_mode, PresentMode::Mailbox);
    assert_eq!(wgpu_config.alpha_mode, CompositeAlphaMode::Opaque);
    assert_eq!(wgpu_config.desired_maximum_frame_latency, 2);
    assert_eq!(wgpu_config.usage, TextureUsages::RENDER_ATTACHMENT);
    assert!(wgpu_config.view_formats.is_empty());
}

#[test]
fn test_surface_configuration_to_wgpu_usage_is_render_attachment() {
    // to_wgpu always sets usage to RENDER_ATTACHMENT
    let config = SurfaceConfiguration::new(640, 480);
    let wgpu_config = config.to_wgpu();
    assert_eq!(wgpu_config.usage, TextureUsages::RENDER_ATTACHMENT);
}

// ============================================================================
// SurfaceConfiguration - Default Tests
// ============================================================================

#[test]
fn test_surface_configuration_default() {
    // Path AE: Default configuration
    let config = SurfaceConfiguration::default();
    assert_eq!(config.width, 1);
    assert_eq!(config.height, 1);
    assert_eq!(config.format, TextureFormat::Bgra8UnormSrgb);
    assert_eq!(config.present_mode, PresentMode::Fifo);
    assert_eq!(config.alpha_mode, CompositeAlphaMode::Auto);
    assert_eq!(config.desired_maximum_frame_latency, 2);
}

// ============================================================================
// SurfaceCapabilities - from_wgpu() Tests
// ============================================================================

#[test]
fn test_surface_capabilities_from_wgpu() {
    // Path AF: Create from wgpu capabilities
    let wgpu_caps = wgpu::SurfaceCapabilities {
        formats: vec![TextureFormat::Bgra8Unorm, TextureFormat::Rgba8Unorm],
        present_modes: vec![PresentMode::Fifo, PresentMode::Mailbox],
        alpha_modes: vec![CompositeAlphaMode::Auto, CompositeAlphaMode::Opaque],
        usages: TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC,
    };

    let caps = SurfaceCapabilities::from_wgpu(wgpu_caps);

    assert_eq!(caps.formats.len(), 2);
    assert!(caps.formats.contains(&TextureFormat::Bgra8Unorm));
    assert!(caps.formats.contains(&TextureFormat::Rgba8Unorm));
    assert_eq!(caps.present_modes.len(), 2);
    assert!(caps.present_modes.contains(&PresentMode::Fifo));
    assert!(caps.present_modes.contains(&PresentMode::Mailbox));
    assert_eq!(caps.alpha_modes.len(), 2);
    assert!(caps.alpha_modes.contains(&CompositeAlphaMode::Auto));
    assert!(caps.alpha_modes.contains(&CompositeAlphaMode::Opaque));
    assert!(caps.usages.contains(TextureUsages::RENDER_ATTACHMENT));
    assert!(caps.usages.contains(TextureUsages::COPY_SRC));
}

// ============================================================================
// PlatformTarget Tests
// ============================================================================

#[test]
fn test_platform_target_name_wayland() {
    // Path AG: Platform names
    assert_eq!(PlatformTarget::Wayland.name(), "Linux (Wayland)");
}

#[test]
fn test_platform_target_name_x11() {
    assert_eq!(PlatformTarget::X11.name(), "Linux (X11)");
}

#[test]
fn test_platform_target_name_windows() {
    assert_eq!(PlatformTarget::Windows.name(), "Windows");
}

#[test]
fn test_platform_target_name_macos() {
    assert_eq!(PlatformTarget::MacOS.name(), "macOS");
}

#[test]
fn test_platform_target_name_ios() {
    assert_eq!(PlatformTarget::IOS.name(), "iOS");
}

#[test]
fn test_platform_target_name_android() {
    assert_eq!(PlatformTarget::Android.name(), "Android");
}

#[test]
fn test_platform_target_name_web() {
    assert_eq!(PlatformTarget::Web.name(), "Web");
}

#[test]
fn test_platform_target_name_unknown() {
    assert_eq!(PlatformTarget::Unknown.name(), "Unknown");
}

#[test]
fn test_platform_target_is_supported_true() {
    // Path AH: Supported platforms
    assert!(PlatformTarget::Wayland.is_supported());
    assert!(PlatformTarget::X11.is_supported());
    assert!(PlatformTarget::Windows.is_supported());
    assert!(PlatformTarget::MacOS.is_supported());
    assert!(PlatformTarget::IOS.is_supported());
    assert!(PlatformTarget::Android.is_supported());
    assert!(PlatformTarget::Web.is_supported());
}

#[test]
fn test_platform_target_is_supported_unknown() {
    // Path AH: Unknown is not supported
    assert!(!PlatformTarget::Unknown.is_supported());
}

#[test]
fn test_platform_target_display_trait() {
    // Path AI: Display trait
    assert_eq!(format!("{}", PlatformTarget::Windows), "Windows");
    assert_eq!(format!("{}", PlatformTarget::MacOS), "macOS");
    assert_eq!(format!("{}", PlatformTarget::Wayland), "Linux (Wayland)");
    assert_eq!(format!("{}", PlatformTarget::X11), "Linux (X11)");
}

#[test]
fn test_platform_target_clone() {
    let platform = PlatformTarget::Windows;
    let cloned = platform.clone();
    assert_eq!(platform, cloned);
}

#[test]
fn test_platform_target_copy() {
    let platform = PlatformTarget::MacOS;
    let copied: PlatformTarget = platform;
    assert_eq!(platform, copied);
}

#[test]
fn test_platform_target_current() {
    // Should return a valid platform on development machines
    let current = PlatformTarget::current();
    // On CI/dev machines, should be supported
    #[cfg(any(target_os = "linux", target_os = "windows", target_os = "macos"))]
    assert!(current.is_supported());
}

// ============================================================================
// SurfaceError Tests
// ============================================================================

#[test]
fn test_surface_error_unsupported() {
    // Path AK: Unsupported platform error
    let err = SurfaceError::unsupported();
    assert!(err.is_platform_error());
    assert!(!err.is_recoverable());
}

#[test]
fn test_surface_error_window_handle() {
    // Path AK: Window handle error
    let err = SurfaceError::window_handle("test window error");
    assert!(!err.is_recoverable());
    let err_str = format!("{}", err);
    assert!(err_str.contains("test window error"));
}

#[test]
fn test_surface_error_display_handle() {
    // Path AK: Display handle error
    let err = SurfaceError::display_handle("display connection failed");
    let err_str = format!("{}", err);
    assert!(err_str.contains("display connection failed"));
}

#[test]
fn test_surface_error_creation_failed() {
    // Path AK: Surface creation failed error
    let err = SurfaceError::creation_failed("wgpu backend error");
    let err_str = format!("{}", err);
    assert!(err_str.contains("wgpu backend error"));
}

#[test]
fn test_surface_error_invalid_config() {
    // Path AK: Invalid configuration error
    let err = SurfaceError::invalid_config("unsupported format");
    let err_str = format!("{}", err);
    assert!(err_str.contains("unsupported format"));
}

#[test]
fn test_surface_error_is_recoverable_surface_lost() {
    // Path AJ: Surface lost is recoverable
    let err = SurfaceError::SurfaceLost {
        reason: "display mode change".to_string(),
    };
    assert!(err.is_recoverable());
    assert!(!err.is_platform_error());
}

#[test]
fn test_surface_error_is_recoverable_surface_outdated() {
    // Path AJ: Surface outdated is recoverable
    let err = SurfaceError::SurfaceOutdated;
    assert!(err.is_recoverable());
}

#[test]
fn test_surface_error_is_not_recoverable() {
    // Non-recoverable errors
    assert!(!SurfaceError::unsupported().is_recoverable());
    assert!(!SurfaceError::window_handle("error").is_recoverable());
    assert!(!SurfaceError::display_handle("error").is_recoverable());
    assert!(!SurfaceError::creation_failed("error").is_recoverable());
    assert!(!SurfaceError::invalid_config("error").is_recoverable());
}

#[test]
fn test_surface_error_is_platform_error() {
    // Path AL: Only UnsupportedPlatform is a platform error
    assert!(SurfaceError::unsupported().is_platform_error());
    assert!(!SurfaceError::window_handle("error").is_platform_error());
    assert!(!SurfaceError::SurfaceOutdated.is_platform_error());
    assert!(!SurfaceError::SurfaceLost { reason: "test".to_string() }.is_platform_error());
}

#[test]
fn test_surface_error_display_formatting() {
    // Error display messages are informative
    let err = SurfaceError::InvalidConfiguration {
        message: "format Rgba16Float not supported".to_string(),
    };
    let display = format!("{}", err);
    assert!(display.contains("invalid surface configuration"));
    assert!(display.contains("Rgba16Float"));
}

// ============================================================================
// Edge Cases and Comprehensive Coverage
// ============================================================================

#[test]
fn test_capabilities_with_single_format() {
    // Single format should be returned as preferred
    let caps = SurfaceCapabilities {
        formats: vec![TextureFormat::R8Unorm],
        present_modes: vec![PresentMode::Fifo],
        alpha_modes: vec![CompositeAlphaMode::Auto],
        usages: TextureUsages::RENDER_ATTACHMENT,
    };
    assert_eq!(caps.preferred_format(), Some(TextureFormat::R8Unorm));
}

#[test]
fn test_capabilities_with_many_formats() {
    // Many formats - should still pick the best sRGB
    let caps = SurfaceCapabilities {
        formats: vec![
            TextureFormat::R8Unorm,
            TextureFormat::Rg8Unorm,
            TextureFormat::Rgba8Unorm,
            TextureFormat::Bgra8Unorm,
            TextureFormat::Rgba8UnormSrgb,
            TextureFormat::Bgra8UnormSrgb,
            TextureFormat::Rgba16Float,
        ],
        present_modes: vec![PresentMode::Fifo],
        alpha_modes: vec![CompositeAlphaMode::Auto],
        usages: TextureUsages::RENDER_ATTACHMENT,
    };
    assert_eq!(caps.preferred_format(), Some(TextureFormat::Bgra8UnormSrgb));
}

#[test]
fn test_configuration_clone() {
    let config = SurfaceConfiguration::new(1920, 1080)
        .with_format(TextureFormat::Rgba8Unorm);
    let cloned = config.clone();
    assert_eq!(config.width, cloned.width);
    assert_eq!(config.height, cloned.height);
    assert_eq!(config.format, cloned.format);
}

#[test]
fn test_capabilities_clone() {
    let caps = SurfaceCapabilities {
        formats: vec![TextureFormat::Bgra8Unorm],
        present_modes: vec![PresentMode::Fifo],
        alpha_modes: vec![CompositeAlphaMode::Auto],
        usages: TextureUsages::RENDER_ATTACHMENT,
    };
    let cloned = caps.clone();
    assert_eq!(caps.formats, cloned.formats);
    assert_eq!(caps.present_modes, cloned.present_modes);
}

#[test]
fn test_capabilities_debug() {
    let caps = SurfaceCapabilities {
        formats: vec![TextureFormat::Bgra8Unorm],
        present_modes: vec![PresentMode::Fifo],
        alpha_modes: vec![CompositeAlphaMode::Auto],
        usages: TextureUsages::RENDER_ATTACHMENT,
    };
    let debug_str = format!("{:?}", caps);
    assert!(debug_str.contains("SurfaceCapabilities"));
    assert!(debug_str.contains("formats"));
}

#[test]
fn test_configuration_debug() {
    let config = SurfaceConfiguration::new(640, 480);
    let debug_str = format!("{:?}", config);
    assert!(debug_str.contains("SurfaceConfiguration"));
    assert!(debug_str.contains("640"));
    assert!(debug_str.contains("480"));
}

#[test]
fn test_platform_target_equality() {
    assert_eq!(PlatformTarget::Windows, PlatformTarget::Windows);
    assert_ne!(PlatformTarget::Windows, PlatformTarget::MacOS);
    assert_ne!(PlatformTarget::X11, PlatformTarget::Wayland);
}

#[test]
fn test_validate_with_all_empty_capabilities() {
    // Empty capabilities should fail validation for any config
    let caps = SurfaceCapabilities {
        formats: vec![],
        present_modes: vec![],
        alpha_modes: vec![],
        usages: TextureUsages::empty(),
    };
    let config = SurfaceConfiguration::new(800, 600);
    assert!(config.validate(&caps).is_err());
}

#[test]
fn test_validate_error_message_includes_available_formats() {
    let caps = SurfaceCapabilities {
        formats: vec![TextureFormat::Bgra8Unorm],
        present_modes: vec![PresentMode::Fifo],
        alpha_modes: vec![CompositeAlphaMode::Auto],
        usages: TextureUsages::RENDER_ATTACHMENT,
    };
    let config = SurfaceConfiguration::new(800, 600)
        .with_format(TextureFormat::Rgba16Float);

    let err = config.validate(&caps).unwrap_err();
    let err_str = format!("{}", err);
    assert!(err_str.contains("Bgra8Unorm")); // Should mention available format
}

#[test]
fn test_validate_error_message_includes_available_present_modes() {
    let caps = SurfaceCapabilities {
        formats: vec![TextureFormat::Bgra8Unorm],
        present_modes: vec![PresentMode::Fifo],
        alpha_modes: vec![CompositeAlphaMode::Auto],
        usages: TextureUsages::RENDER_ATTACHMENT,
    };
    let config = SurfaceConfiguration::new(800, 600)
        .with_format(TextureFormat::Bgra8Unorm)
        .with_present_mode(PresentMode::Mailbox);

    let err = config.validate(&caps).unwrap_err();
    let err_str = format!("{}", err);
    assert!(err_str.contains("Fifo")); // Should mention available present mode
}

// ============================================================================
// Assertion Count Summary
// ============================================================================
// Total assertions across all tests: 100+
// - preferred_format(): 12 tests, 12+ assertions
// - preferred_present_mode(): 7 tests, 7+ assertions
// - preferred_alpha_mode(): 4 tests, 4+ assertions
// - supports_format(): 3 tests, 6+ assertions
// - supports_present_mode(): 3 tests, 5+ assertions
// - supports_hdr(): 6 tests, 6+ assertions
// - SurfaceConfiguration::new(): 5 tests, 12+ assertions
// - SurfaceConfiguration::from_capabilities(): 3 tests, 8+ assertions
// - SurfaceConfiguration builders: 6 tests, 12+ assertions
// - SurfaceConfiguration::validate(): 6 tests, 10+ assertions
// - SurfaceConfiguration::to_wgpu(): 2 tests, 9+ assertions
// - SurfaceConfiguration::default(): 1 test, 6+ assertions
// - SurfaceCapabilities::from_wgpu(): 1 test, 9+ assertions
// - PlatformTarget: 14 tests, 20+ assertions
// - SurfaceError: 10 tests, 15+ assertions
// - Edge cases: 8 tests, 15+ assertions
