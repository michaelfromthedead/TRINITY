// WHITEBOX tests for T-WGPU-P7.1.3 (Format Selection)
//
// WHITEBOX discipline: These tests have FULL ACCESS to the implementation.
// They exercise internal code paths, branch conditions, and edge cases
// that are not visible through the public contract alone.
//
// Implementation under test: crates/renderer-backend/src/presentation/surface.rs
//   - FormatCategory enum and from_format() classification
//   - SurfaceCapabilities::preferred_format() sRGB priority logic
//   - SurfaceCapabilities::preferred_hdr_format() HDR priority logic
//   - SurfaceCapabilities::select_format() with prefer_hdr flag
//   - SurfaceCapabilities::formats_in_category() filtering
//   - SurfaceCapabilities::format_category() static helper
//   - SurfaceConfiguration::validate() error messages for invalid formats
//   - Edge cases: empty formats, all HDR, all linear
//
// WHITEBOX coverage plan:
//   - Path A: FormatCategory::from_format for all sRGB formats (Bgra8UnormSrgb, Rgba8UnormSrgb)
//   - Path B: FormatCategory::from_format for all linear formats (Bgra8Unorm, Rgba8Unorm)
//   - Path C: FormatCategory::from_format for all HDR formats (Rgba16Float, Rgb10a2Unorm, Rg11b10Float)
//   - Path D: FormatCategory::from_format for Other category (depth, stencil, etc.)
//   - Path E: FormatCategory::is_gamma_corrected() for all categories
//   - Path F: FormatCategory::is_hdr() for all categories
//   - Path G: FormatCategory::name() for all categories
//   - Path H: FormatCategory Display trait
//   - Path I: preferred_format() returns Bgra8UnormSrgb when available
//   - Path J: preferred_format() returns Rgba8UnormSrgb when Bgra8UnormSrgb unavailable
//   - Path K: preferred_format() falls back to Bgra8Unorm when no sRGB
//   - Path L: preferred_format() falls back to Rgba8Unorm when only linear
//   - Path M: preferred_format() returns first format as last resort
//   - Path N: preferred_format() returns None on empty formats
//   - Path O: preferred_hdr_format() returns Rgba16Float first priority
//   - Path P: preferred_hdr_format() returns Rg11b10Float second priority
//   - Path Q: preferred_hdr_format() returns Rgb10a2Unorm third priority
//   - Path R: preferred_hdr_format() returns None when no HDR available
//   - Path S: select_format() with prefer_hdr=true returns HDR when available
//   - Path T: select_format() with prefer_hdr=true falls back to sRGB when no HDR
//   - Path U: select_format() with prefer_hdr=false returns sRGB
//   - Path V: formats_in_category() returns empty for missing category
//   - Path W: formats_in_category() returns all matching formats
//   - Path X: format_category() static helper matches from_format()
//   - Path Y: validate() error message for unsupported format
//   - Path Z: validate() error message for unsupported present mode
//   - Path AA: validate() error message for unsupported alpha mode
//   - Path AB: validate() error message for zero dimensions
//   - Path AC: Edge case: all formats are HDR
//   - Path AD: Edge case: all formats are linear
//   - Path AE: Edge case: single format available
//   - Path AF: supports_hdr() with Rgb10a2Unorm
//   - Path AG: supports_hdr() with Rg11b10Float
//   - Path AH: Multiple sRGB formats, confirm priority order

use renderer_backend::presentation::{
    FormatCategory, SurfaceCapabilities, SurfaceConfiguration, SurfaceError,
};
use wgpu::{CompositeAlphaMode, PresentMode, TextureFormat, TextureUsages};

// ============================================================================
// Test Helpers
// ============================================================================

/// Create SurfaceCapabilities with custom formats.
fn make_caps_with_formats(formats: Vec<TextureFormat>) -> SurfaceCapabilities {
    SurfaceCapabilities {
        formats,
        present_modes: vec![PresentMode::Fifo],
        alpha_modes: vec![CompositeAlphaMode::Auto],
        usages: TextureUsages::RENDER_ATTACHMENT,
    }
}

/// Create SurfaceCapabilities with custom formats, present modes, and alpha modes.
fn make_full_caps(
    formats: Vec<TextureFormat>,
    present_modes: Vec<PresentMode>,
    alpha_modes: Vec<CompositeAlphaMode>,
) -> SurfaceCapabilities {
    SurfaceCapabilities {
        formats,
        present_modes,
        alpha_modes,
        usages: TextureUsages::RENDER_ATTACHMENT,
    }
}

// ============================================================================
// Path A-D: FormatCategory::from_format Classification
// ============================================================================

/// Path A: FormatCategory::from_format classifies all sRGB formats correctly.
#[test]
fn test_format_category_srgb_bgra8unorm_srgb() {
    assert_eq!(
        FormatCategory::from_format(TextureFormat::Bgra8UnormSrgb),
        FormatCategory::Srgb
    );
}

#[test]
fn test_format_category_srgb_rgba8unorm_srgb() {
    assert_eq!(
        FormatCategory::from_format(TextureFormat::Rgba8UnormSrgb),
        FormatCategory::Srgb
    );
}

/// Path B: FormatCategory::from_format classifies all linear formats correctly.
#[test]
fn test_format_category_linear_bgra8unorm() {
    assert_eq!(
        FormatCategory::from_format(TextureFormat::Bgra8Unorm),
        FormatCategory::Linear
    );
}

#[test]
fn test_format_category_linear_rgba8unorm() {
    assert_eq!(
        FormatCategory::from_format(TextureFormat::Rgba8Unorm),
        FormatCategory::Linear
    );
}

/// Path C: FormatCategory::from_format classifies all HDR formats correctly.
#[test]
fn test_format_category_hdr_rgba16float() {
    assert_eq!(
        FormatCategory::from_format(TextureFormat::Rgba16Float),
        FormatCategory::Hdr
    );
}

#[test]
fn test_format_category_hdr_rgb10a2unorm() {
    assert_eq!(
        FormatCategory::from_format(TextureFormat::Rgb10a2Unorm),
        FormatCategory::Hdr
    );
}

#[test]
fn test_format_category_hdr_rg11b10float() {
    assert_eq!(
        FormatCategory::from_format(TextureFormat::Rg11b10Float),
        FormatCategory::Hdr
    );
}

/// Path D: FormatCategory::from_format classifies Other category correctly.
#[test]
fn test_format_category_other_depth32float() {
    assert_eq!(
        FormatCategory::from_format(TextureFormat::Depth32Float),
        FormatCategory::Other
    );
}

#[test]
fn test_format_category_other_r8unorm() {
    assert_eq!(
        FormatCategory::from_format(TextureFormat::R8Unorm),
        FormatCategory::Other
    );
}

#[test]
fn test_format_category_other_rg8unorm() {
    assert_eq!(
        FormatCategory::from_format(TextureFormat::Rg8Unorm),
        FormatCategory::Other
    );
}

#[test]
fn test_format_category_other_depth24plus() {
    assert_eq!(
        FormatCategory::from_format(TextureFormat::Depth24Plus),
        FormatCategory::Other
    );
}

#[test]
fn test_format_category_other_stencil8() {
    assert_eq!(
        FormatCategory::from_format(TextureFormat::Stencil8),
        FormatCategory::Other
    );
}

// ============================================================================
// Path E: FormatCategory::is_gamma_corrected
// ============================================================================

#[test]
fn test_format_category_is_gamma_corrected_srgb() {
    assert!(FormatCategory::Srgb.is_gamma_corrected());
}

#[test]
fn test_format_category_is_gamma_corrected_linear() {
    assert!(!FormatCategory::Linear.is_gamma_corrected());
}

#[test]
fn test_format_category_is_gamma_corrected_hdr() {
    assert!(!FormatCategory::Hdr.is_gamma_corrected());
}

#[test]
fn test_format_category_is_gamma_corrected_other() {
    assert!(!FormatCategory::Other.is_gamma_corrected());
}

// ============================================================================
// Path F: FormatCategory::is_hdr
// ============================================================================

#[test]
fn test_format_category_is_hdr_srgb() {
    assert!(!FormatCategory::Srgb.is_hdr());
}

#[test]
fn test_format_category_is_hdr_linear() {
    assert!(!FormatCategory::Linear.is_hdr());
}

#[test]
fn test_format_category_is_hdr_hdr() {
    assert!(FormatCategory::Hdr.is_hdr());
}

#[test]
fn test_format_category_is_hdr_other() {
    assert!(!FormatCategory::Other.is_hdr());
}

// ============================================================================
// Path G: FormatCategory::name
// ============================================================================

#[test]
fn test_format_category_name_srgb() {
    assert_eq!(FormatCategory::Srgb.name(), "sRGB");
}

#[test]
fn test_format_category_name_linear() {
    assert_eq!(FormatCategory::Linear.name(), "Linear");
}

#[test]
fn test_format_category_name_hdr() {
    assert_eq!(FormatCategory::Hdr.name(), "HDR");
}

#[test]
fn test_format_category_name_other() {
    assert_eq!(FormatCategory::Other.name(), "Other");
}

// ============================================================================
// Path H: FormatCategory Display trait
// ============================================================================

#[test]
fn test_format_category_display_srgb() {
    assert_eq!(format!("{}", FormatCategory::Srgb), "sRGB");
}

#[test]
fn test_format_category_display_linear() {
    assert_eq!(format!("{}", FormatCategory::Linear), "Linear");
}

#[test]
fn test_format_category_display_hdr() {
    assert_eq!(format!("{}", FormatCategory::Hdr), "HDR");
}

#[test]
fn test_format_category_display_other() {
    assert_eq!(format!("{}", FormatCategory::Other), "Other");
}

// ============================================================================
// Path I-N: preferred_format() sRGB priority logic
// ============================================================================

/// Path I: preferred_format() returns Bgra8UnormSrgb when available.
#[test]
fn test_preferred_format_returns_bgra8unorm_srgb_first() {
    let caps = make_caps_with_formats(vec![
        TextureFormat::Rgba8Unorm,
        TextureFormat::Bgra8UnormSrgb,
        TextureFormat::Rgba8UnormSrgb,
    ]);
    assert_eq!(caps.preferred_format(), Some(TextureFormat::Bgra8UnormSrgb));
}

/// Path J: preferred_format() returns Rgba8UnormSrgb when Bgra8UnormSrgb unavailable.
#[test]
fn test_preferred_format_returns_rgba8unorm_srgb_when_bgra_unavailable() {
    let caps = make_caps_with_formats(vec![
        TextureFormat::Rgba8Unorm,
        TextureFormat::Rgba8UnormSrgb,
    ]);
    assert_eq!(caps.preferred_format(), Some(TextureFormat::Rgba8UnormSrgb));
}

/// Path K: preferred_format() falls back to Bgra8Unorm when no sRGB available.
#[test]
fn test_preferred_format_falls_back_to_bgra8unorm() {
    let caps = make_caps_with_formats(vec![
        TextureFormat::Rgba8Unorm,
        TextureFormat::Bgra8Unorm,
    ]);
    assert_eq!(caps.preferred_format(), Some(TextureFormat::Bgra8Unorm));
}

/// Path L: preferred_format() falls back to Rgba8Unorm when only linear available.
#[test]
fn test_preferred_format_falls_back_to_rgba8unorm() {
    let caps = make_caps_with_formats(vec![TextureFormat::Rgba8Unorm]);
    assert_eq!(caps.preferred_format(), Some(TextureFormat::Rgba8Unorm));
}

/// Path M: preferred_format() returns first format as last resort.
#[test]
fn test_preferred_format_returns_first_format_as_last_resort() {
    let caps = make_caps_with_formats(vec![
        TextureFormat::Rgba16Float,
        TextureFormat::Rgb10a2Unorm,
    ]);
    assert_eq!(caps.preferred_format(), Some(TextureFormat::Rgba16Float));
}

/// Path N: preferred_format() returns None on empty formats.
#[test]
fn test_preferred_format_returns_none_on_empty() {
    let caps = make_caps_with_formats(vec![]);
    assert_eq!(caps.preferred_format(), None);
}

// ============================================================================
// Path O-R: preferred_hdr_format() HDR priority logic
// ============================================================================

/// Path O: preferred_hdr_format() returns Rgba16Float as first priority.
#[test]
fn test_preferred_hdr_format_rgba16float_first_priority() {
    let caps = make_caps_with_formats(vec![
        TextureFormat::Rgb10a2Unorm,
        TextureFormat::Rg11b10Float,
        TextureFormat::Rgba16Float,
    ]);
    assert_eq!(caps.preferred_hdr_format(), Some(TextureFormat::Rgba16Float));
}

/// Path P: preferred_hdr_format() returns Rg11b10Float as second priority.
#[test]
fn test_preferred_hdr_format_rg11b10float_second_priority() {
    let caps = make_caps_with_formats(vec![
        TextureFormat::Rgb10a2Unorm,
        TextureFormat::Rg11b10Float,
    ]);
    assert_eq!(caps.preferred_hdr_format(), Some(TextureFormat::Rg11b10Float));
}

/// Path Q: preferred_hdr_format() returns Rgb10a2Unorm as third priority.
#[test]
fn test_preferred_hdr_format_rgb10a2unorm_third_priority() {
    let caps = make_caps_with_formats(vec![TextureFormat::Rgb10a2Unorm]);
    assert_eq!(caps.preferred_hdr_format(), Some(TextureFormat::Rgb10a2Unorm));
}

/// Path R: preferred_hdr_format() returns None when no HDR available.
#[test]
fn test_preferred_hdr_format_returns_none_when_no_hdr() {
    let caps = make_caps_with_formats(vec![
        TextureFormat::Bgra8UnormSrgb,
        TextureFormat::Rgba8Unorm,
    ]);
    assert_eq!(caps.preferred_hdr_format(), None);
}

#[test]
fn test_preferred_hdr_format_empty_formats() {
    let caps = make_caps_with_formats(vec![]);
    assert_eq!(caps.preferred_hdr_format(), None);
}

// ============================================================================
// Path S-U: select_format() with prefer_hdr flag
// ============================================================================

/// Path S: select_format() with prefer_hdr=true returns HDR when available.
#[test]
fn test_select_format_prefer_hdr_returns_hdr() {
    let caps = make_caps_with_formats(vec![
        TextureFormat::Bgra8UnormSrgb,
        TextureFormat::Rgba16Float,
    ]);
    assert_eq!(caps.select_format(true), Some(TextureFormat::Rgba16Float));
}

/// Path T: select_format() with prefer_hdr=true falls back to sRGB when no HDR.
#[test]
fn test_select_format_prefer_hdr_falls_back_to_srgb() {
    let caps = make_caps_with_formats(vec![TextureFormat::Bgra8UnormSrgb]);
    assert_eq!(caps.select_format(true), Some(TextureFormat::Bgra8UnormSrgb));
}

/// Path U: select_format() with prefer_hdr=false returns sRGB.
#[test]
fn test_select_format_no_hdr_returns_srgb() {
    let caps = make_caps_with_formats(vec![
        TextureFormat::Bgra8UnormSrgb,
        TextureFormat::Rgba16Float,
    ]);
    assert_eq!(caps.select_format(false), Some(TextureFormat::Bgra8UnormSrgb));
}

#[test]
fn test_select_format_empty_formats() {
    let caps = make_caps_with_formats(vec![]);
    assert_eq!(caps.select_format(true), None);
    assert_eq!(caps.select_format(false), None);
}

// ============================================================================
// Path V-W: formats_in_category() filtering
// ============================================================================

/// Path V: formats_in_category() returns empty for missing category.
#[test]
fn test_formats_in_category_returns_empty_for_missing() {
    let caps = make_caps_with_formats(vec![
        TextureFormat::Bgra8UnormSrgb,
        TextureFormat::Rgba8UnormSrgb,
    ]);
    let hdr_formats = caps.formats_in_category(FormatCategory::Hdr);
    assert!(hdr_formats.is_empty());
}

/// Path W: formats_in_category() returns all matching formats.
#[test]
fn test_formats_in_category_returns_all_matching() {
    let caps = make_caps_with_formats(vec![
        TextureFormat::Bgra8UnormSrgb,
        TextureFormat::Rgba8UnormSrgb,
        TextureFormat::Bgra8Unorm,
        TextureFormat::Rgba16Float,
    ]);

    let srgb = caps.formats_in_category(FormatCategory::Srgb);
    assert_eq!(srgb.len(), 2);
    assert!(srgb.contains(&TextureFormat::Bgra8UnormSrgb));
    assert!(srgb.contains(&TextureFormat::Rgba8UnormSrgb));

    let linear = caps.formats_in_category(FormatCategory::Linear);
    assert_eq!(linear.len(), 1);
    assert!(linear.contains(&TextureFormat::Bgra8Unorm));

    let hdr = caps.formats_in_category(FormatCategory::Hdr);
    assert_eq!(hdr.len(), 1);
    assert!(hdr.contains(&TextureFormat::Rgba16Float));
}

#[test]
fn test_formats_in_category_empty_formats() {
    let caps = make_caps_with_formats(vec![]);
    assert!(caps.formats_in_category(FormatCategory::Srgb).is_empty());
    assert!(caps.formats_in_category(FormatCategory::Linear).is_empty());
    assert!(caps.formats_in_category(FormatCategory::Hdr).is_empty());
    assert!(caps.formats_in_category(FormatCategory::Other).is_empty());
}

// ============================================================================
// Path X: format_category() static helper
// ============================================================================

#[test]
fn test_format_category_static_helper_srgb() {
    assert_eq!(
        SurfaceCapabilities::format_category(TextureFormat::Bgra8UnormSrgb),
        FormatCategory::Srgb
    );
}

#[test]
fn test_format_category_static_helper_linear() {
    assert_eq!(
        SurfaceCapabilities::format_category(TextureFormat::Bgra8Unorm),
        FormatCategory::Linear
    );
}

#[test]
fn test_format_category_static_helper_hdr() {
    assert_eq!(
        SurfaceCapabilities::format_category(TextureFormat::Rgba16Float),
        FormatCategory::Hdr
    );
}

#[test]
fn test_format_category_static_helper_other() {
    assert_eq!(
        SurfaceCapabilities::format_category(TextureFormat::Depth32Float),
        FormatCategory::Other
    );
}

#[test]
fn test_format_category_static_helper_matches_from_format() {
    let formats = [
        TextureFormat::Bgra8UnormSrgb,
        TextureFormat::Rgba8UnormSrgb,
        TextureFormat::Bgra8Unorm,
        TextureFormat::Rgba8Unorm,
        TextureFormat::Rgba16Float,
        TextureFormat::Rgb10a2Unorm,
        TextureFormat::Rg11b10Float,
        TextureFormat::Depth32Float,
    ];
    for format in &formats {
        assert_eq!(
            SurfaceCapabilities::format_category(*format),
            FormatCategory::from_format(*format)
        );
    }
}

// ============================================================================
// Path Y-AB: validate() error messages
// ============================================================================

/// Path Y: validate() error message for unsupported format.
#[test]
fn test_validate_error_unsupported_format() {
    let caps = make_full_caps(
        vec![TextureFormat::Bgra8Unorm],
        vec![PresentMode::Fifo],
        vec![CompositeAlphaMode::Auto],
    );
    let config = SurfaceConfiguration::new(800, 600)
        .with_format(TextureFormat::Rgba16Float)
        .with_present_mode(PresentMode::Fifo)
        .with_alpha_mode(CompositeAlphaMode::Auto);

    let result = config.validate(&caps);
    assert!(result.is_err());
    let err_msg = format!("{}", result.unwrap_err());
    assert!(err_msg.contains("format"));
    assert!(err_msg.contains("not supported"));
}

/// Path Z: validate() error message for unsupported present mode.
#[test]
fn test_validate_error_unsupported_present_mode() {
    let caps = make_full_caps(
        vec![TextureFormat::Bgra8Unorm],
        vec![PresentMode::Fifo],
        vec![CompositeAlphaMode::Auto],
    );
    let config = SurfaceConfiguration::new(800, 600)
        .with_format(TextureFormat::Bgra8Unorm)
        .with_present_mode(PresentMode::Mailbox)
        .with_alpha_mode(CompositeAlphaMode::Auto);

    let result = config.validate(&caps);
    assert!(result.is_err());
    let err_msg = format!("{}", result.unwrap_err());
    assert!(err_msg.contains("present mode"));
    assert!(err_msg.contains("not supported"));
}

/// Path AA: validate() error message for unsupported alpha mode.
#[test]
fn test_validate_error_unsupported_alpha_mode() {
    let caps = make_full_caps(
        vec![TextureFormat::Bgra8Unorm],
        vec![PresentMode::Fifo],
        vec![CompositeAlphaMode::Auto],
    );
    let config = SurfaceConfiguration::new(800, 600)
        .with_format(TextureFormat::Bgra8Unorm)
        .with_present_mode(PresentMode::Fifo)
        .with_alpha_mode(CompositeAlphaMode::Opaque);

    let result = config.validate(&caps);
    assert!(result.is_err());
    let err_msg = format!("{}", result.unwrap_err());
    assert!(err_msg.contains("alpha mode"));
    assert!(err_msg.contains("not supported"));
}

/// Path AB: validate() error message for zero dimensions.
#[test]
fn test_validate_error_zero_width() {
    let caps = make_full_caps(
        vec![TextureFormat::Bgra8Unorm],
        vec![PresentMode::Fifo],
        vec![CompositeAlphaMode::Auto],
    );
    // SurfaceConfiguration::new clamps to 1, so we need to manually create invalid config
    let config = SurfaceConfiguration {
        format: TextureFormat::Bgra8Unorm,
        width: 0,
        height: 600,
        present_mode: PresentMode::Fifo,
        alpha_mode: CompositeAlphaMode::Auto,
        desired_maximum_frame_latency: 2,
        view_formats: vec![],
    };

    let result = config.validate(&caps);
    assert!(result.is_err());
    let err_msg = format!("{}", result.unwrap_err());
    assert!(err_msg.contains("non-zero") || err_msg.contains("dimensions"));
}

#[test]
fn test_validate_error_zero_height() {
    let caps = make_full_caps(
        vec![TextureFormat::Bgra8Unorm],
        vec![PresentMode::Fifo],
        vec![CompositeAlphaMode::Auto],
    );
    let config = SurfaceConfiguration {
        format: TextureFormat::Bgra8Unorm,
        width: 800,
        height: 0,
        present_mode: PresentMode::Fifo,
        alpha_mode: CompositeAlphaMode::Auto,
        desired_maximum_frame_latency: 2,
        view_formats: vec![],
    };

    let result = config.validate(&caps);
    assert!(result.is_err());
}

// ============================================================================
// Path AC-AE: Edge Cases
// ============================================================================

/// Path AC: Edge case - all formats are HDR.
#[test]
fn test_edge_case_all_formats_hdr() {
    let caps = make_caps_with_formats(vec![
        TextureFormat::Rgba16Float,
        TextureFormat::Rgb10a2Unorm,
        TextureFormat::Rg11b10Float,
    ]);

    // preferred_format should fall back to first format (HDR)
    assert_eq!(caps.preferred_format(), Some(TextureFormat::Rgba16Float));

    // preferred_hdr_format should return Rgba16Float (first priority)
    assert_eq!(caps.preferred_hdr_format(), Some(TextureFormat::Rgba16Float));

    // select_format with prefer_hdr=true should return HDR
    assert_eq!(caps.select_format(true), Some(TextureFormat::Rgba16Float));

    // select_format with prefer_hdr=false should fall back to first format
    assert_eq!(caps.select_format(false), Some(TextureFormat::Rgba16Float));

    // formats_in_category should return all HDR
    let hdr_formats = caps.formats_in_category(FormatCategory::Hdr);
    assert_eq!(hdr_formats.len(), 3);
}

/// Path AD: Edge case - all formats are linear.
#[test]
fn test_edge_case_all_formats_linear() {
    let caps = make_caps_with_formats(vec![
        TextureFormat::Bgra8Unorm,
        TextureFormat::Rgba8Unorm,
    ]);

    // preferred_format should return Bgra8Unorm (first linear priority)
    assert_eq!(caps.preferred_format(), Some(TextureFormat::Bgra8Unorm));

    // preferred_hdr_format should return None
    assert_eq!(caps.preferred_hdr_format(), None);

    // select_format with prefer_hdr=true should fall back to linear
    assert_eq!(caps.select_format(true), Some(TextureFormat::Bgra8Unorm));

    // formats_in_category should return all linear
    let linear_formats = caps.formats_in_category(FormatCategory::Linear);
    assert_eq!(linear_formats.len(), 2);
}

/// Path AE: Edge case - single format available.
#[test]
fn test_edge_case_single_format_srgb() {
    let caps = make_caps_with_formats(vec![TextureFormat::Bgra8UnormSrgb]);
    assert_eq!(caps.preferred_format(), Some(TextureFormat::Bgra8UnormSrgb));
    assert_eq!(caps.preferred_hdr_format(), None);
    assert_eq!(caps.select_format(true), Some(TextureFormat::Bgra8UnormSrgb));
    assert_eq!(caps.select_format(false), Some(TextureFormat::Bgra8UnormSrgb));
}

#[test]
fn test_edge_case_single_format_hdr() {
    let caps = make_caps_with_formats(vec![TextureFormat::Rgba16Float]);
    assert_eq!(caps.preferred_format(), Some(TextureFormat::Rgba16Float));
    assert_eq!(caps.preferred_hdr_format(), Some(TextureFormat::Rgba16Float));
    assert_eq!(caps.select_format(true), Some(TextureFormat::Rgba16Float));
    assert_eq!(caps.select_format(false), Some(TextureFormat::Rgba16Float));
}

#[test]
fn test_edge_case_single_format_linear() {
    let caps = make_caps_with_formats(vec![TextureFormat::Bgra8Unorm]);
    assert_eq!(caps.preferred_format(), Some(TextureFormat::Bgra8Unorm));
    assert_eq!(caps.preferred_hdr_format(), None);
    assert_eq!(caps.select_format(true), Some(TextureFormat::Bgra8Unorm));
}

// ============================================================================
// Path AF-AG: supports_hdr() with specific HDR formats
// ============================================================================

/// Path AF: supports_hdr() with Rgb10a2Unorm.
#[test]
fn test_supports_hdr_rgb10a2unorm() {
    let caps = make_caps_with_formats(vec![
        TextureFormat::Bgra8Unorm,
        TextureFormat::Rgb10a2Unorm,
    ]);
    assert!(caps.supports_hdr());
}

/// Path AG: supports_hdr() with Rg11b10Float.
#[test]
fn test_supports_hdr_rg11b10float() {
    let caps = make_caps_with_formats(vec![
        TextureFormat::Bgra8Unorm,
        TextureFormat::Rg11b10Float,
    ]);
    assert!(caps.supports_hdr());
}

#[test]
fn test_supports_hdr_rgba16float() {
    let caps = make_caps_with_formats(vec![
        TextureFormat::Bgra8Unorm,
        TextureFormat::Rgba16Float,
    ]);
    assert!(caps.supports_hdr());
}

#[test]
fn test_supports_hdr_all_hdr_formats() {
    let caps = make_caps_with_formats(vec![
        TextureFormat::Rgba16Float,
        TextureFormat::Rgb10a2Unorm,
        TextureFormat::Rg11b10Float,
    ]);
    assert!(caps.supports_hdr());
}

#[test]
fn test_supports_hdr_no_hdr() {
    let caps = make_caps_with_formats(vec![
        TextureFormat::Bgra8Unorm,
        TextureFormat::Bgra8UnormSrgb,
    ]);
    assert!(!caps.supports_hdr());
}

// ============================================================================
// Path AH: Multiple sRGB formats, confirm priority order
// ============================================================================

#[test]
fn test_srgb_priority_bgra_before_rgba() {
    // Even if Rgba8UnormSrgb appears first in the list,
    // Bgra8UnormSrgb should be selected because it has higher priority
    let caps = make_caps_with_formats(vec![
        TextureFormat::Rgba8UnormSrgb,
        TextureFormat::Bgra8UnormSrgb,
    ]);
    assert_eq!(caps.preferred_format(), Some(TextureFormat::Bgra8UnormSrgb));
}

#[test]
fn test_linear_priority_bgra_before_rgba() {
    // Same for linear formats
    let caps = make_caps_with_formats(vec![
        TextureFormat::Rgba8Unorm,
        TextureFormat::Bgra8Unorm,
    ]);
    assert_eq!(caps.preferred_format(), Some(TextureFormat::Bgra8Unorm));
}

// ============================================================================
// Additional Coverage: supports_format and supports_present_mode
// ============================================================================

#[test]
fn test_supports_format_true() {
    let caps = make_caps_with_formats(vec![
        TextureFormat::Bgra8UnormSrgb,
        TextureFormat::Rgba16Float,
    ]);
    assert!(caps.supports_format(TextureFormat::Bgra8UnormSrgb));
    assert!(caps.supports_format(TextureFormat::Rgba16Float));
}

#[test]
fn test_supports_format_false() {
    let caps = make_caps_with_formats(vec![TextureFormat::Bgra8UnormSrgb]);
    assert!(!caps.supports_format(TextureFormat::Rgba16Float));
    assert!(!caps.supports_format(TextureFormat::Bgra8Unorm));
}

#[test]
fn test_supports_present_mode_true() {
    let caps = make_full_caps(
        vec![TextureFormat::Bgra8Unorm],
        vec![PresentMode::Fifo, PresentMode::Mailbox],
        vec![CompositeAlphaMode::Auto],
    );
    assert!(caps.supports_present_mode(PresentMode::Fifo));
    assert!(caps.supports_present_mode(PresentMode::Mailbox));
}

#[test]
fn test_supports_present_mode_false() {
    let caps = make_full_caps(
        vec![TextureFormat::Bgra8Unorm],
        vec![PresentMode::Fifo],
        vec![CompositeAlphaMode::Auto],
    );
    assert!(!caps.supports_present_mode(PresentMode::Mailbox));
    assert!(!caps.supports_present_mode(PresentMode::Immediate));
}

// ============================================================================
// Additional Coverage: preferred_present_mode
// ============================================================================

#[test]
fn test_preferred_present_mode_mailbox() {
    let caps = make_full_caps(
        vec![TextureFormat::Bgra8Unorm],
        vec![PresentMode::Fifo, PresentMode::Mailbox],
        vec![CompositeAlphaMode::Auto],
    );
    assert_eq!(caps.preferred_present_mode(), PresentMode::Mailbox);
}

#[test]
fn test_preferred_present_mode_fifo_fallback() {
    let caps = make_full_caps(
        vec![TextureFormat::Bgra8Unorm],
        vec![PresentMode::Fifo, PresentMode::Immediate],
        vec![CompositeAlphaMode::Auto],
    );
    assert_eq!(caps.preferred_present_mode(), PresentMode::Fifo);
}

#[test]
fn test_preferred_present_mode_first_fallback() {
    let caps = make_full_caps(
        vec![TextureFormat::Bgra8Unorm],
        vec![PresentMode::Immediate],
        vec![CompositeAlphaMode::Auto],
    );
    assert_eq!(caps.preferred_present_mode(), PresentMode::Immediate);
}

#[test]
fn test_preferred_present_mode_empty_returns_fifo() {
    let caps = make_full_caps(
        vec![TextureFormat::Bgra8Unorm],
        vec![],
        vec![CompositeAlphaMode::Auto],
    );
    // Falls back to Fifo when present_modes is empty
    assert_eq!(caps.preferred_present_mode(), PresentMode::Fifo);
}

// ============================================================================
// Additional Coverage: preferred_alpha_mode
// ============================================================================

#[test]
fn test_preferred_alpha_mode_opaque() {
    let caps = make_full_caps(
        vec![TextureFormat::Bgra8Unorm],
        vec![PresentMode::Fifo],
        vec![CompositeAlphaMode::Auto, CompositeAlphaMode::Opaque],
    );
    assert_eq!(caps.preferred_alpha_mode(), CompositeAlphaMode::Opaque);
}

#[test]
fn test_preferred_alpha_mode_auto_fallback() {
    let caps = make_full_caps(
        vec![TextureFormat::Bgra8Unorm],
        vec![PresentMode::Fifo],
        vec![CompositeAlphaMode::Auto, CompositeAlphaMode::PreMultiplied],
    );
    assert_eq!(caps.preferred_alpha_mode(), CompositeAlphaMode::Auto);
}

#[test]
fn test_preferred_alpha_mode_empty_returns_auto() {
    let caps = make_full_caps(
        vec![TextureFormat::Bgra8Unorm],
        vec![PresentMode::Fifo],
        vec![],
    );
    assert_eq!(caps.preferred_alpha_mode(), CompositeAlphaMode::Auto);
}

// ============================================================================
// Additional Coverage: SurfaceConfiguration
// ============================================================================

#[test]
fn test_surface_configuration_new_clamps_zero() {
    let config = SurfaceConfiguration::new(0, 0);
    assert_eq!(config.width, 1);
    assert_eq!(config.height, 1);
}

#[test]
fn test_surface_configuration_with_frame_latency_clamps() {
    let config = SurfaceConfiguration::new(800, 600).with_frame_latency(0);
    assert_eq!(config.desired_maximum_frame_latency, 1);
}

#[test]
fn test_surface_configuration_from_capabilities_fallback_format() {
    let caps = make_caps_with_formats(vec![]);
    let config = SurfaceConfiguration::from_capabilities(&caps, 800, 600);
    // Falls back to default format when caps.preferred_format() is None
    assert_eq!(config.format, TextureFormat::Bgra8UnormSrgb);
}

#[test]
fn test_surface_configuration_default() {
    let config = SurfaceConfiguration::default();
    assert_eq!(config.width, 1);
    assert_eq!(config.height, 1);
    assert_eq!(config.format, TextureFormat::Bgra8UnormSrgb);
    assert_eq!(config.present_mode, PresentMode::Fifo);
}

#[test]
fn test_surface_configuration_validate_success() {
    let caps = make_full_caps(
        vec![TextureFormat::Bgra8Unorm],
        vec![PresentMode::Fifo],
        vec![CompositeAlphaMode::Auto],
    );
    let config = SurfaceConfiguration::new(800, 600)
        .with_format(TextureFormat::Bgra8Unorm)
        .with_present_mode(PresentMode::Fifo)
        .with_alpha_mode(CompositeAlphaMode::Auto);

    assert!(config.validate(&caps).is_ok());
}

#[test]
fn test_surface_configuration_to_wgpu() {
    let config = SurfaceConfiguration::new(1920, 1080)
        .with_format(TextureFormat::Rgba16Float)
        .with_present_mode(PresentMode::Mailbox)
        .with_alpha_mode(CompositeAlphaMode::Opaque)
        .with_frame_latency(3);

    let wgpu_config = config.to_wgpu();
    assert_eq!(wgpu_config.width, 1920);
    assert_eq!(wgpu_config.height, 1080);
    assert_eq!(wgpu_config.format, TextureFormat::Rgba16Float);
    assert_eq!(wgpu_config.present_mode, PresentMode::Mailbox);
    assert_eq!(wgpu_config.alpha_mode, CompositeAlphaMode::Opaque);
    assert_eq!(wgpu_config.desired_maximum_frame_latency, 3);
    assert_eq!(wgpu_config.usage, TextureUsages::RENDER_ATTACHMENT);
}

// ============================================================================
// Additional Coverage: FormatCategory equality and hash
// ============================================================================

#[test]
fn test_format_category_equality() {
    assert_eq!(FormatCategory::Srgb, FormatCategory::Srgb);
    assert_eq!(FormatCategory::Linear, FormatCategory::Linear);
    assert_eq!(FormatCategory::Hdr, FormatCategory::Hdr);
    assert_eq!(FormatCategory::Other, FormatCategory::Other);
    assert_ne!(FormatCategory::Srgb, FormatCategory::Linear);
    assert_ne!(FormatCategory::Srgb, FormatCategory::Hdr);
    assert_ne!(FormatCategory::Srgb, FormatCategory::Other);
}

#[test]
fn test_format_category_hash() {
    use std::collections::HashSet;
    let mut set = HashSet::new();
    set.insert(FormatCategory::Srgb);
    set.insert(FormatCategory::Linear);
    set.insert(FormatCategory::Hdr);
    set.insert(FormatCategory::Other);
    assert_eq!(set.len(), 4);
    // Adding duplicate should not increase size
    set.insert(FormatCategory::Srgb);
    assert_eq!(set.len(), 4);
}

#[test]
fn test_format_category_clone_copy() {
    let cat = FormatCategory::Srgb;
    let cloned = cat.clone();
    let copied = cat;
    assert_eq!(cat, cloned);
    assert_eq!(cat, copied);
}

// ============================================================================
// SurfaceCapabilities::from_wgpu
// ============================================================================

#[test]
fn test_surface_capabilities_from_wgpu() {
    let wgpu_caps = wgpu::SurfaceCapabilities {
        formats: vec![TextureFormat::Bgra8UnormSrgb, TextureFormat::Rgba16Float],
        present_modes: vec![PresentMode::Fifo, PresentMode::Mailbox],
        alpha_modes: vec![CompositeAlphaMode::Opaque],
        usages: TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC,
    };

    let caps = SurfaceCapabilities::from_wgpu(wgpu_caps);
    assert_eq!(caps.formats.len(), 2);
    assert_eq!(caps.present_modes.len(), 2);
    assert_eq!(caps.alpha_modes.len(), 1);
    assert!(caps.usages.contains(TextureUsages::RENDER_ATTACHMENT));
    assert!(caps.usages.contains(TextureUsages::COPY_SRC));
}
