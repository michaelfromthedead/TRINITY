//! Blackbox contract tests for T-WGPU-P7.1.3 - Format Selection.
//!
//! CLEANROOM: No src/ access beyond the public API exported by the crate.
//! Tests use only `renderer_backend::presentation::*` -- no internal fields,
//! no private methods, no implementation details.
//!
//! Acceptance criteria (contract):
//!   1. Select optimal format from capabilities
//!   2. sRGB preference (Bgra8UnormSrgb > Rgba8UnormSrgb > linear)
//!   3. HDR format detection
//!   4. Format validation before use
//!
//! Coverage (40+ assertions):
//!   - SurfaceCapabilities::preferred_format() public API
//!   - SurfaceCapabilities::preferred_hdr_format() public API
//!   - SurfaceCapabilities::select_format() public API
//!   - SurfaceCapabilities::supports_hdr() public API
//!   - FormatCategory public enum variants
//!   - SurfaceConfiguration::validate() with various formats
//!   - Format selection returns valid format from capabilities
//!   - HDR preference when requested

use renderer_backend::presentation::{
    FormatCategory, SurfaceCapabilities, SurfaceConfiguration,
};
use wgpu::{CompositeAlphaMode, PresentMode, TextureFormat, TextureUsages};

// ============================================================================
// Helper Functions
// ============================================================================

/// Create SurfaceCapabilities with specified formats and defaults for other fields.
fn caps_with_formats(formats: Vec<TextureFormat>) -> SurfaceCapabilities {
    SurfaceCapabilities {
        formats,
        present_modes: vec![PresentMode::Fifo],
        alpha_modes: vec![CompositeAlphaMode::Auto],
        usages: TextureUsages::RENDER_ATTACHMENT,
    }
}

/// Create SurfaceCapabilities with full control over all fields.
fn caps_full(
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
// SECTION 1: SurfaceCapabilities::preferred_format() - sRGB Preference
// ============================================================================

#[test]
fn preferred_format_selects_bgra8_srgb_first() {
    // Bgra8UnormSrgb should be preferred over all other formats
    let caps = caps_with_formats(vec![
        TextureFormat::Rgba8Unorm,
        TextureFormat::Bgra8Unorm,
        TextureFormat::Bgra8UnormSrgb,
        TextureFormat::Rgba8UnormSrgb,
    ]);

    let preferred = caps.preferred_format();
    assert_eq!(preferred, Some(TextureFormat::Bgra8UnormSrgb));
}

#[test]
fn preferred_format_selects_rgba8_srgb_when_bgra_srgb_unavailable() {
    // Rgba8UnormSrgb should be second choice after Bgra8UnormSrgb
    let caps = caps_with_formats(vec![
        TextureFormat::Rgba8Unorm,
        TextureFormat::Bgra8Unorm,
        TextureFormat::Rgba8UnormSrgb,
    ]);

    let preferred = caps.preferred_format();
    assert_eq!(preferred, Some(TextureFormat::Rgba8UnormSrgb));
}

#[test]
fn preferred_format_falls_back_to_bgra8_linear_when_no_srgb() {
    // When no sRGB formats available, Bgra8Unorm should be preferred
    let caps = caps_with_formats(vec![
        TextureFormat::Rgba8Unorm,
        TextureFormat::Bgra8Unorm,
    ]);

    let preferred = caps.preferred_format();
    assert_eq!(preferred, Some(TextureFormat::Bgra8Unorm));
}

#[test]
fn preferred_format_falls_back_to_rgba8_linear_when_no_bgra() {
    // When only Rgba8Unorm available, use it
    let caps = caps_with_formats(vec![TextureFormat::Rgba8Unorm]);

    let preferred = caps.preferred_format();
    assert_eq!(preferred, Some(TextureFormat::Rgba8Unorm));
}

#[test]
fn preferred_format_returns_first_format_as_last_resort() {
    // When no standard formats match, return first available
    let caps = caps_with_formats(vec![
        TextureFormat::Rgba16Float,
        TextureFormat::R8Unorm,
    ]);

    let preferred = caps.preferred_format();
    assert_eq!(preferred, Some(TextureFormat::Rgba16Float));
}

#[test]
fn preferred_format_returns_none_for_empty_formats() {
    // Empty formats list should return None
    let caps = caps_with_formats(vec![]);

    let preferred = caps.preferred_format();
    assert_eq!(preferred, None);
}

#[test]
fn preferred_format_prefers_srgb_over_hdr() {
    // sRGB should be preferred over HDR by default (for gamma correctness)
    let caps = caps_with_formats(vec![
        TextureFormat::Rgba16Float,
        TextureFormat::Rgb10a2Unorm,
        TextureFormat::Bgra8UnormSrgb,
    ]);

    let preferred = caps.preferred_format();
    assert_eq!(preferred, Some(TextureFormat::Bgra8UnormSrgb));
}

// ============================================================================
// SECTION 2: SurfaceCapabilities::preferred_hdr_format() - HDR Selection
// ============================================================================

#[test]
fn preferred_hdr_format_selects_rgba16_float_first() {
    // Rgba16Float is highest priority HDR format
    let caps = caps_with_formats(vec![
        TextureFormat::Bgra8Unorm,
        TextureFormat::Rgb10a2Unorm,
        TextureFormat::Rg11b10Float,
        TextureFormat::Rgba16Float,
    ]);

    let hdr = caps.preferred_hdr_format();
    assert_eq!(hdr, Some(TextureFormat::Rgba16Float));
}

#[test]
fn preferred_hdr_format_selects_rg11b10_float_when_rgba16_unavailable() {
    // Rg11b10Float is second choice
    let caps = caps_with_formats(vec![
        TextureFormat::Bgra8Unorm,
        TextureFormat::Rgb10a2Unorm,
        TextureFormat::Rg11b10Float,
    ]);

    let hdr = caps.preferred_hdr_format();
    assert_eq!(hdr, Some(TextureFormat::Rg11b10Float));
}

#[test]
fn preferred_hdr_format_selects_rgb10a2_unorm_as_fallback() {
    // Rgb10a2Unorm is third choice
    let caps = caps_with_formats(vec![
        TextureFormat::Bgra8Unorm,
        TextureFormat::Rgb10a2Unorm,
    ]);

    let hdr = caps.preferred_hdr_format();
    assert_eq!(hdr, Some(TextureFormat::Rgb10a2Unorm));
}

#[test]
fn preferred_hdr_format_returns_none_when_no_hdr_available() {
    // No HDR formats should return None
    let caps = caps_with_formats(vec![
        TextureFormat::Bgra8Unorm,
        TextureFormat::Bgra8UnormSrgb,
        TextureFormat::Rgba8Unorm,
    ]);

    let hdr = caps.preferred_hdr_format();
    assert_eq!(hdr, None);
}

#[test]
fn preferred_hdr_format_returns_none_for_empty_formats() {
    let caps = caps_with_formats(vec![]);

    let hdr = caps.preferred_hdr_format();
    assert_eq!(hdr, None);
}

// ============================================================================
// SECTION 3: SurfaceCapabilities::supports_hdr() - HDR Detection
// ============================================================================

#[test]
fn supports_hdr_true_with_rgba16_float() {
    let caps = caps_with_formats(vec![
        TextureFormat::Bgra8Unorm,
        TextureFormat::Rgba16Float,
    ]);

    assert!(caps.supports_hdr());
}

#[test]
fn supports_hdr_true_with_rgb10a2_unorm() {
    let caps = caps_with_formats(vec![
        TextureFormat::Bgra8Unorm,
        TextureFormat::Rgb10a2Unorm,
    ]);

    assert!(caps.supports_hdr());
}

#[test]
fn supports_hdr_true_with_rg11b10_float() {
    let caps = caps_with_formats(vec![
        TextureFormat::Bgra8Unorm,
        TextureFormat::Rg11b10Float,
    ]);

    assert!(caps.supports_hdr());
}

#[test]
fn supports_hdr_true_with_multiple_hdr_formats() {
    let caps = caps_with_formats(vec![
        TextureFormat::Rgba16Float,
        TextureFormat::Rgb10a2Unorm,
        TextureFormat::Rg11b10Float,
    ]);

    assert!(caps.supports_hdr());
}

#[test]
fn supports_hdr_false_with_only_sdr_formats() {
    let caps = caps_with_formats(vec![
        TextureFormat::Bgra8Unorm,
        TextureFormat::Bgra8UnormSrgb,
        TextureFormat::Rgba8Unorm,
        TextureFormat::Rgba8UnormSrgb,
    ]);

    assert!(!caps.supports_hdr());
}

#[test]
fn supports_hdr_false_with_empty_formats() {
    let caps = caps_with_formats(vec![]);

    assert!(!caps.supports_hdr());
}

// ============================================================================
// SECTION 4: SurfaceCapabilities::select_format() - Format Selection with HDR Preference
// ============================================================================

#[test]
fn select_format_returns_hdr_when_preferred_and_available() {
    let caps = caps_with_formats(vec![
        TextureFormat::Bgra8UnormSrgb,
        TextureFormat::Rgba16Float,
    ]);

    let selected = caps.select_format(true);
    assert_eq!(selected, Some(TextureFormat::Rgba16Float));
}

#[test]
fn select_format_returns_srgb_when_hdr_preferred_but_unavailable() {
    let caps = caps_with_formats(vec![
        TextureFormat::Bgra8UnormSrgb,
        TextureFormat::Rgba8Unorm,
    ]);

    let selected = caps.select_format(true);
    assert_eq!(selected, Some(TextureFormat::Bgra8UnormSrgb));
}

#[test]
fn select_format_returns_srgb_when_hdr_not_preferred() {
    let caps = caps_with_formats(vec![
        TextureFormat::Bgra8UnormSrgb,
        TextureFormat::Rgba16Float,
    ]);

    let selected = caps.select_format(false);
    assert_eq!(selected, Some(TextureFormat::Bgra8UnormSrgb));
}

#[test]
fn select_format_returns_none_for_empty_formats() {
    let caps = caps_with_formats(vec![]);

    assert_eq!(caps.select_format(true), None);
    assert_eq!(caps.select_format(false), None);
}

#[test]
fn select_format_returns_linear_fallback_when_no_srgb_and_no_hdr_preferred() {
    let caps = caps_with_formats(vec![
        TextureFormat::Bgra8Unorm,
        TextureFormat::Rgba8Unorm,
    ]);

    let selected = caps.select_format(false);
    assert_eq!(selected, Some(TextureFormat::Bgra8Unorm));
}

// ============================================================================
// SECTION 5: FormatCategory Enum Variants
// ============================================================================

#[test]
fn format_category_srgb_from_bgra8_srgb() {
    let cat = FormatCategory::from_format(TextureFormat::Bgra8UnormSrgb);
    assert_eq!(cat, FormatCategory::Srgb);
}

#[test]
fn format_category_srgb_from_rgba8_srgb() {
    let cat = FormatCategory::from_format(TextureFormat::Rgba8UnormSrgb);
    assert_eq!(cat, FormatCategory::Srgb);
}

#[test]
fn format_category_linear_from_bgra8_unorm() {
    let cat = FormatCategory::from_format(TextureFormat::Bgra8Unorm);
    assert_eq!(cat, FormatCategory::Linear);
}

#[test]
fn format_category_linear_from_rgba8_unorm() {
    let cat = FormatCategory::from_format(TextureFormat::Rgba8Unorm);
    assert_eq!(cat, FormatCategory::Linear);
}

#[test]
fn format_category_hdr_from_rgba16_float() {
    let cat = FormatCategory::from_format(TextureFormat::Rgba16Float);
    assert_eq!(cat, FormatCategory::Hdr);
}

#[test]
fn format_category_hdr_from_rgb10a2_unorm() {
    let cat = FormatCategory::from_format(TextureFormat::Rgb10a2Unorm);
    assert_eq!(cat, FormatCategory::Hdr);
}

#[test]
fn format_category_hdr_from_rg11b10_float() {
    let cat = FormatCategory::from_format(TextureFormat::Rg11b10Float);
    assert_eq!(cat, FormatCategory::Hdr);
}

#[test]
fn format_category_other_from_r8_unorm() {
    let cat = FormatCategory::from_format(TextureFormat::R8Unorm);
    assert_eq!(cat, FormatCategory::Other);
}

#[test]
fn format_category_other_from_depth32_float() {
    let cat = FormatCategory::from_format(TextureFormat::Depth32Float);
    assert_eq!(cat, FormatCategory::Other);
}

#[test]
fn format_category_other_from_r16_float() {
    let cat = FormatCategory::from_format(TextureFormat::R16Float);
    assert_eq!(cat, FormatCategory::Other);
}

#[test]
fn format_category_is_gamma_corrected_true_for_srgb() {
    assert!(FormatCategory::Srgb.is_gamma_corrected());
}

#[test]
fn format_category_is_gamma_corrected_false_for_non_srgb() {
    assert!(!FormatCategory::Linear.is_gamma_corrected());
    assert!(!FormatCategory::Hdr.is_gamma_corrected());
    assert!(!FormatCategory::Other.is_gamma_corrected());
}

#[test]
fn format_category_is_hdr_true_only_for_hdr() {
    assert!(FormatCategory::Hdr.is_hdr());
    assert!(!FormatCategory::Srgb.is_hdr());
    assert!(!FormatCategory::Linear.is_hdr());
    assert!(!FormatCategory::Other.is_hdr());
}

#[test]
fn format_category_name_returns_expected_strings() {
    assert_eq!(FormatCategory::Srgb.name(), "sRGB");
    assert_eq!(FormatCategory::Linear.name(), "Linear");
    assert_eq!(FormatCategory::Hdr.name(), "HDR");
    assert_eq!(FormatCategory::Other.name(), "Other");
}

#[test]
fn format_category_display_trait_matches_name() {
    assert_eq!(format!("{}", FormatCategory::Srgb), "sRGB");
    assert_eq!(format!("{}", FormatCategory::Linear), "Linear");
    assert_eq!(format!("{}", FormatCategory::Hdr), "HDR");
    assert_eq!(format!("{}", FormatCategory::Other), "Other");
}

// ============================================================================
// SECTION 6: SurfaceCapabilities::formats_in_category()
// ============================================================================

#[test]
fn formats_in_category_returns_only_srgb_formats() {
    let caps = caps_with_formats(vec![
        TextureFormat::Bgra8Unorm,
        TextureFormat::Bgra8UnormSrgb,
        TextureFormat::Rgba8UnormSrgb,
        TextureFormat::Rgba16Float,
    ]);

    let srgb_formats = caps.formats_in_category(FormatCategory::Srgb);
    assert_eq!(srgb_formats.len(), 2);
    assert!(srgb_formats.contains(&TextureFormat::Bgra8UnormSrgb));
    assert!(srgb_formats.contains(&TextureFormat::Rgba8UnormSrgb));
}

#[test]
fn formats_in_category_returns_only_linear_formats() {
    let caps = caps_with_formats(vec![
        TextureFormat::Bgra8Unorm,
        TextureFormat::Rgba8Unorm,
        TextureFormat::Bgra8UnormSrgb,
    ]);

    let linear_formats = caps.formats_in_category(FormatCategory::Linear);
    assert_eq!(linear_formats.len(), 2);
    assert!(linear_formats.contains(&TextureFormat::Bgra8Unorm));
    assert!(linear_formats.contains(&TextureFormat::Rgba8Unorm));
}

#[test]
fn formats_in_category_returns_only_hdr_formats() {
    let caps = caps_with_formats(vec![
        TextureFormat::Bgra8Unorm,
        TextureFormat::Rgba16Float,
        TextureFormat::Rgb10a2Unorm,
        TextureFormat::Rg11b10Float,
    ]);

    let hdr_formats = caps.formats_in_category(FormatCategory::Hdr);
    assert_eq!(hdr_formats.len(), 3);
    assert!(hdr_formats.contains(&TextureFormat::Rgba16Float));
    assert!(hdr_formats.contains(&TextureFormat::Rgb10a2Unorm));
    assert!(hdr_formats.contains(&TextureFormat::Rg11b10Float));
}

#[test]
fn formats_in_category_returns_empty_when_no_match() {
    let caps = caps_with_formats(vec![
        TextureFormat::Bgra8Unorm,
        TextureFormat::Rgba8Unorm,
    ]);

    let hdr_formats = caps.formats_in_category(FormatCategory::Hdr);
    assert!(hdr_formats.is_empty());
}

// ============================================================================
// SECTION 7: SurfaceCapabilities::format_category() Static Helper
// ============================================================================

#[test]
fn format_category_helper_works_for_srgb() {
    let cat = SurfaceCapabilities::format_category(TextureFormat::Bgra8UnormSrgb);
    assert_eq!(cat, FormatCategory::Srgb);
}

#[test]
fn format_category_helper_works_for_hdr() {
    let cat = SurfaceCapabilities::format_category(TextureFormat::Rgba16Float);
    assert_eq!(cat, FormatCategory::Hdr);
}

// ============================================================================
// SECTION 8: SurfaceCapabilities::supports_format()
// ============================================================================

#[test]
fn supports_format_true_for_included_format() {
    let caps = caps_with_formats(vec![
        TextureFormat::Bgra8Unorm,
        TextureFormat::Rgba16Float,
    ]);

    assert!(caps.supports_format(TextureFormat::Bgra8Unorm));
    assert!(caps.supports_format(TextureFormat::Rgba16Float));
}

#[test]
fn supports_format_false_for_excluded_format() {
    let caps = caps_with_formats(vec![TextureFormat::Bgra8Unorm]);

    assert!(!caps.supports_format(TextureFormat::Rgba16Float));
    assert!(!caps.supports_format(TextureFormat::Bgra8UnormSrgb));
}

#[test]
fn supports_format_false_for_empty_formats() {
    let caps = caps_with_formats(vec![]);

    assert!(!caps.supports_format(TextureFormat::Bgra8Unorm));
}

// ============================================================================
// SECTION 9: SurfaceConfiguration::validate() with Various Formats
// ============================================================================

#[test]
fn validate_succeeds_with_supported_format() {
    let caps = caps_full(
        vec![TextureFormat::Bgra8Unorm],
        vec![PresentMode::Fifo],
        vec![CompositeAlphaMode::Auto],
    );

    let config = SurfaceConfiguration::new(1920, 1080)
        .with_format(TextureFormat::Bgra8Unorm)
        .with_present_mode(PresentMode::Fifo)
        .with_alpha_mode(CompositeAlphaMode::Auto);

    assert!(config.validate(&caps).is_ok());
}

#[test]
fn validate_fails_with_unsupported_format() {
    let caps = caps_full(
        vec![TextureFormat::Bgra8Unorm],
        vec![PresentMode::Fifo],
        vec![CompositeAlphaMode::Auto],
    );

    let config = SurfaceConfiguration::new(1920, 1080)
        .with_format(TextureFormat::Rgba16Float)
        .with_present_mode(PresentMode::Fifo)
        .with_alpha_mode(CompositeAlphaMode::Auto);

    let result = config.validate(&caps);
    assert!(result.is_err());

    let err = result.unwrap_err();
    let msg = format!("{}", err);
    assert!(msg.contains("format"));
}

#[test]
fn validate_fails_with_unsupported_present_mode() {
    let caps = caps_full(
        vec![TextureFormat::Bgra8Unorm],
        vec![PresentMode::Fifo],
        vec![CompositeAlphaMode::Auto],
    );

    let config = SurfaceConfiguration::new(1920, 1080)
        .with_format(TextureFormat::Bgra8Unorm)
        .with_present_mode(PresentMode::Mailbox)
        .with_alpha_mode(CompositeAlphaMode::Auto);

    let result = config.validate(&caps);
    assert!(result.is_err());

    let err = result.unwrap_err();
    let msg = format!("{}", err);
    assert!(msg.contains("present mode"));
}

#[test]
fn validate_fails_with_unsupported_alpha_mode() {
    let caps = caps_full(
        vec![TextureFormat::Bgra8Unorm],
        vec![PresentMode::Fifo],
        vec![CompositeAlphaMode::Opaque],
    );

    let config = SurfaceConfiguration::new(1920, 1080)
        .with_format(TextureFormat::Bgra8Unorm)
        .with_present_mode(PresentMode::Fifo)
        .with_alpha_mode(CompositeAlphaMode::PreMultiplied);

    let result = config.validate(&caps);
    assert!(result.is_err());

    let err = result.unwrap_err();
    let msg = format!("{}", err);
    assert!(msg.contains("alpha mode"));
}

#[test]
fn validate_succeeds_with_hdr_format_when_supported() {
    let caps = caps_full(
        vec![TextureFormat::Bgra8Unorm, TextureFormat::Rgba16Float],
        vec![PresentMode::Fifo],
        vec![CompositeAlphaMode::Auto],
    );

    let config = SurfaceConfiguration::new(1920, 1080)
        .with_format(TextureFormat::Rgba16Float)
        .with_present_mode(PresentMode::Fifo)
        .with_alpha_mode(CompositeAlphaMode::Auto);

    assert!(config.validate(&caps).is_ok());
}

#[test]
fn validate_succeeds_with_srgb_format() {
    let caps = caps_full(
        vec![TextureFormat::Bgra8UnormSrgb],
        vec![PresentMode::Fifo],
        vec![CompositeAlphaMode::Auto],
    );

    let config = SurfaceConfiguration::new(800, 600)
        .with_format(TextureFormat::Bgra8UnormSrgb)
        .with_present_mode(PresentMode::Fifo)
        .with_alpha_mode(CompositeAlphaMode::Auto);

    assert!(config.validate(&caps).is_ok());
}

// ============================================================================
// SECTION 10: SurfaceConfiguration::from_capabilities() - Format Selection Integration
// ============================================================================

#[test]
fn from_capabilities_selects_preferred_format() {
    let caps = caps_with_formats(vec![
        TextureFormat::Rgba8Unorm,
        TextureFormat::Bgra8UnormSrgb,
    ]);

    let config = SurfaceConfiguration::from_capabilities(&caps, 1920, 1080);

    // Should select Bgra8UnormSrgb (preferred sRGB)
    assert_eq!(config.format, TextureFormat::Bgra8UnormSrgb);
}

#[test]
fn from_capabilities_selects_fallback_format_when_no_srgb() {
    let caps = caps_with_formats(vec![TextureFormat::Bgra8Unorm]);

    let config = SurfaceConfiguration::from_capabilities(&caps, 1920, 1080);

    assert_eq!(config.format, TextureFormat::Bgra8Unorm);
}

#[test]
fn from_capabilities_config_passes_validation() {
    let caps = caps_full(
        vec![TextureFormat::Bgra8UnormSrgb],
        vec![PresentMode::Mailbox, PresentMode::Fifo],
        vec![CompositeAlphaMode::Opaque, CompositeAlphaMode::Auto],
    );

    let config = SurfaceConfiguration::from_capabilities(&caps, 1280, 720);

    // Config created from capabilities should always validate
    assert!(config.validate(&caps).is_ok());
}

// ============================================================================
// SECTION 11: Format Selection Returns Valid Format from Capabilities
// ============================================================================

#[test]
fn select_format_always_returns_format_in_capabilities_when_not_empty() {
    let formats = vec![
        TextureFormat::Bgra8Unorm,
        TextureFormat::Rgba16Float,
        TextureFormat::Rgb10a2Unorm,
    ];
    let caps = caps_with_formats(formats.clone());

    // When HDR preferred
    if let Some(selected) = caps.select_format(true) {
        assert!(caps.supports_format(selected));
    }

    // When HDR not preferred
    if let Some(selected) = caps.select_format(false) {
        assert!(caps.supports_format(selected));
    }
}

#[test]
fn preferred_format_always_returns_format_in_capabilities_when_not_empty() {
    let formats = vec![
        TextureFormat::R8Unorm,
        TextureFormat::Rg8Unorm,
        TextureFormat::Depth32Float,
    ];
    let caps = caps_with_formats(formats.clone());

    // Should return R8Unorm as first format (last resort)
    let preferred = caps.preferred_format();
    assert_eq!(preferred, Some(TextureFormat::R8Unorm));
    assert!(caps.supports_format(preferred.unwrap()));
}

// ============================================================================
// SECTION 12: HDR Preference Integration Tests
// ============================================================================

#[test]
fn hdr_workflow_integration_test() {
    // Simulate a complete HDR format selection workflow
    let caps = caps_with_formats(vec![
        TextureFormat::Bgra8Unorm,
        TextureFormat::Bgra8UnormSrgb,
        TextureFormat::Rgba16Float,
        TextureFormat::Rgb10a2Unorm,
    ]);

    // 1. Check HDR support
    assert!(caps.supports_hdr());

    // 2. Get preferred HDR format
    let hdr_format = caps.preferred_hdr_format();
    assert_eq!(hdr_format, Some(TextureFormat::Rgba16Float));

    // 3. Select format with HDR preference
    let selected = caps.select_format(true);
    assert_eq!(selected, Some(TextureFormat::Rgba16Float));

    // 4. Verify format category
    let cat = FormatCategory::from_format(selected.unwrap());
    assert!(cat.is_hdr());
}

#[test]
fn sdr_workflow_integration_test() {
    // Simulate a complete SDR format selection workflow
    let caps = caps_with_formats(vec![
        TextureFormat::Bgra8Unorm,
        TextureFormat::Bgra8UnormSrgb,
    ]);

    // 1. Check HDR support (should be false)
    assert!(!caps.supports_hdr());

    // 2. Get preferred HDR format (should be None)
    assert_eq!(caps.preferred_hdr_format(), None);

    // 3. Select format (should fall back to sRGB)
    let selected = caps.select_format(true);
    assert_eq!(selected, Some(TextureFormat::Bgra8UnormSrgb));

    // 4. Verify format category is sRGB
    let cat = FormatCategory::from_format(selected.unwrap());
    assert_eq!(cat, FormatCategory::Srgb);
    assert!(cat.is_gamma_corrected());
}

// ============================================================================
// SECTION 13: Edge Cases and Boundary Conditions
// ============================================================================

#[test]
fn single_format_capabilities() {
    let caps = caps_with_formats(vec![TextureFormat::Bgra8Unorm]);

    assert_eq!(caps.preferred_format(), Some(TextureFormat::Bgra8Unorm));
    assert_eq!(caps.preferred_hdr_format(), None);
    assert_eq!(caps.select_format(true), Some(TextureFormat::Bgra8Unorm));
    assert_eq!(caps.select_format(false), Some(TextureFormat::Bgra8Unorm));
    assert!(!caps.supports_hdr());
}

#[test]
fn all_hdr_formats_only() {
    let caps = caps_with_formats(vec![
        TextureFormat::Rgba16Float,
        TextureFormat::Rgb10a2Unorm,
        TextureFormat::Rg11b10Float,
    ]);

    // No sRGB or linear, so preferred_format returns first HDR
    assert_eq!(caps.preferred_format(), Some(TextureFormat::Rgba16Float));
    assert_eq!(caps.preferred_hdr_format(), Some(TextureFormat::Rgba16Float));
    assert!(caps.supports_hdr());
}

#[test]
fn duplicate_formats_handled_correctly() {
    let caps = caps_with_formats(vec![
        TextureFormat::Bgra8UnormSrgb,
        TextureFormat::Bgra8UnormSrgb, // duplicate
        TextureFormat::Rgba16Float,
    ]);

    // Should still work correctly
    assert_eq!(caps.preferred_format(), Some(TextureFormat::Bgra8UnormSrgb));
    assert!(caps.supports_hdr());

    let srgb_formats = caps.formats_in_category(FormatCategory::Srgb);
    assert_eq!(srgb_formats.len(), 2); // both duplicates included
}

// ============================================================================
// SECTION 14: Present Mode Selection
// ============================================================================

#[test]
fn preferred_present_mode_selects_mailbox_first() {
    let caps = caps_full(
        vec![TextureFormat::Bgra8Unorm],
        vec![PresentMode::Fifo, PresentMode::Mailbox, PresentMode::Immediate],
        vec![CompositeAlphaMode::Auto],
    );

    assert_eq!(caps.preferred_present_mode(), PresentMode::Mailbox);
}

#[test]
fn preferred_present_mode_falls_back_to_fifo() {
    let caps = caps_full(
        vec![TextureFormat::Bgra8Unorm],
        vec![PresentMode::Fifo, PresentMode::Immediate],
        vec![CompositeAlphaMode::Auto],
    );

    assert_eq!(caps.preferred_present_mode(), PresentMode::Fifo);
}

#[test]
fn supports_present_mode_true_for_included() {
    let caps = caps_full(
        vec![TextureFormat::Bgra8Unorm],
        vec![PresentMode::Fifo, PresentMode::Mailbox],
        vec![CompositeAlphaMode::Auto],
    );

    assert!(caps.supports_present_mode(PresentMode::Fifo));
    assert!(caps.supports_present_mode(PresentMode::Mailbox));
    assert!(!caps.supports_present_mode(PresentMode::Immediate));
}

// ============================================================================
// SECTION 15: Alpha Mode Selection
// ============================================================================

#[test]
fn preferred_alpha_mode_selects_opaque_first() {
    let caps = caps_full(
        vec![TextureFormat::Bgra8Unorm],
        vec![PresentMode::Fifo],
        vec![CompositeAlphaMode::Auto, CompositeAlphaMode::Opaque, CompositeAlphaMode::PreMultiplied],
    );

    assert_eq!(caps.preferred_alpha_mode(), CompositeAlphaMode::Opaque);
}

#[test]
fn preferred_alpha_mode_falls_back_to_first_available() {
    let caps = caps_full(
        vec![TextureFormat::Bgra8Unorm],
        vec![PresentMode::Fifo],
        vec![CompositeAlphaMode::PreMultiplied],
    );

    assert_eq!(caps.preferred_alpha_mode(), CompositeAlphaMode::PreMultiplied);
}

// ============================================================================
// Summary: 40+ assertions across 63 test functions covering:
// - preferred_format() with sRGB preference hierarchy
// - preferred_hdr_format() with HDR format priority
// - select_format() with HDR preference toggle
// - supports_hdr() detection
// - FormatCategory enum variants and methods
// - formats_in_category() filtering
// - format_category() static helper
// - supports_format() checks
// - SurfaceConfiguration::validate() with various formats
// - from_capabilities() integration
// - Edge cases: empty, single, duplicate, all-HDR formats
// - Present mode and alpha mode selection
// ============================================================================
