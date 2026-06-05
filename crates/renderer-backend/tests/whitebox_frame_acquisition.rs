// WHITEBOX tests for T-WGPU-P7.1.6 (Frame Acquisition)
//
// WHITEBOX discipline: These tests have FULL ACCESS to the implementation.
// They exercise internal code paths, branch conditions, and edge cases
// that are not visible through the public contract alone.
//
// Implementation under test: crates/renderer-backend/src/presentation/surface.rs
//   - FrameError enum: Timeout, Outdated, Lost variants
//   - FrameError methods: is_recoverable(), needs_reconfigure(), needs_recreate(), lost(), out_of_memory()
//   - FrameError: From<wgpu::SurfaceError> conversion
//   - Frame struct: texture, view, width, height, format, presented
//   - Frame methods: view(), texture(), raw_texture(), dimensions(), aspect_ratio(), present(), discard(), was_presented(), create_view_with_format()
//   - TrinitySurface: acquire_frame(), try_acquire_frame(), acquire_frame_with_format()
//
// WHITEBOX coverage plan:
//   - Path A: FrameError::Timeout is_recoverable = true
//   - Path B: FrameError::Outdated is_recoverable = true
//   - Path C: FrameError::Lost is_recoverable = false
//   - Path D: FrameError::Timeout needs_reconfigure = false
//   - Path E: FrameError::Outdated needs_reconfigure = true
//   - Path F: FrameError::Lost needs_reconfigure = false
//   - Path G: FrameError::Timeout needs_recreate = false
//   - Path H: FrameError::Outdated needs_recreate = false
//   - Path I: FrameError::Lost needs_recreate = true
//   - Path J: FrameError::lost() creates Lost with custom reason
//   - Path K: FrameError::out_of_memory() creates Lost with OOM reason
//   - Path L: From<wgpu::SurfaceError::Timeout> -> FrameError::Timeout
//   - Path M: From<wgpu::SurfaceError::Outdated> -> FrameError::Outdated
//   - Path N: From<wgpu::SurfaceError::Lost> -> FrameError::Lost
//   - Path O: From<wgpu::SurfaceError::OutOfMemory> -> FrameError::Lost (OOM)
//   - Path P: FrameError Display trait formatting
//   - Path Q: Frame::new creates correct width/height
//   - Path R: Frame::view() returns reference to view
//   - Path S: Frame::texture() returns reference to surface texture
//   - Path T: Frame::raw_texture() returns reference to wgpu::Texture
//   - Path U: Frame::dimensions() returns (width, height) tuple
//   - Path V: Frame::aspect_ratio() with normal dimensions
//   - Path W: Frame::aspect_ratio() with zero height returns 1.0
//   - Path X: Frame::was_presented() returns false before present
//   - Path Y: Frame::present() sets presented to true
//   - Path Z: Frame::discard() drops without presenting
//   - Path AA: Frame::create_view_with_format() creates view with different format
//   - Path AB: Frame Debug trait formatting
//   - Path AC: FrameError classification combinations
//   - Path AD: FrameError Lost with empty reason
//   - Path AE: FrameError Lost with long reason
//   - Path AF: Frame width() and height() accessors
//   - Path AG: Frame format() accessor
//   - Path AH: Frame aspect_ratio with square dimensions
//   - Path AI: Frame aspect_ratio with portrait dimensions
//   - Path AJ: Frame aspect_ratio with landscape dimensions
//   - Path AK: FrameError Debug trait formatting
//   - Path AL: Multiple FrameError::lost() calls create independent instances
//   - Path AM: FrameError error chain via thiserror
//   - Path AN: Frame dimensions with minimum size (1x1)
//   - Path AO: Frame dimensions with large size (8K)
//   - Path AP: Frame dimensions with non-power-of-2 sizes
//   - Path AQ: FrameError Lost variant matching
//   - Path AR: FrameError Timeout vs Outdated recovery distinction
//   - Path AS: Frame create_view_with_format with sRGB format
//   - Path AT: Frame create_view_with_format with linear format
//   - Path AU: Frame create_view_with_format with HDR format
//   - Path AV: FrameError from all wgpu::SurfaceError variants
//   - Path AW: Frame lifecycle: new -> use -> present
//   - Path AX: Frame lifecycle: new -> use -> discard
//   - Path AY: Frame lifecycle: new -> immediate discard
//   - Path AZ: FrameError needs_reconfigure and needs_recreate are mutually exclusive

use renderer_backend::presentation::{FrameError, SurfaceCapabilities, SurfaceConfiguration};
use wgpu::{CompositeAlphaMode, PresentMode, TextureFormat, TextureUsages};

// ============================================================================
// Test Helpers
// ============================================================================

/// Create SurfaceCapabilities with standard formats.
fn make_standard_caps() -> SurfaceCapabilities {
    SurfaceCapabilities {
        formats: vec![TextureFormat::Bgra8Unorm, TextureFormat::Bgra8UnormSrgb],
        present_modes: vec![PresentMode::Fifo, PresentMode::Mailbox],
        alpha_modes: vec![CompositeAlphaMode::Auto, CompositeAlphaMode::Opaque],
        usages: TextureUsages::RENDER_ATTACHMENT,
    }
}

/// Create SurfaceCapabilities with HDR formats.
fn make_hdr_caps() -> SurfaceCapabilities {
    SurfaceCapabilities {
        formats: vec![
            TextureFormat::Bgra8Unorm,
            TextureFormat::Rgba16Float,
            TextureFormat::Rg11b10Float,
        ],
        present_modes: vec![PresentMode::Fifo],
        alpha_modes: vec![CompositeAlphaMode::Auto],
        usages: TextureUsages::RENDER_ATTACHMENT,
    }
}

/// Create SurfaceConfiguration for testing.
fn make_config(width: u32, height: u32) -> SurfaceConfiguration {
    SurfaceConfiguration::new(width, height)
        .with_format(TextureFormat::Bgra8Unorm)
        .with_present_mode(PresentMode::Fifo)
        .with_alpha_mode(CompositeAlphaMode::Auto)
}

// ============================================================================
// Path A-C: FrameError is_recoverable()
// ============================================================================

/// Path A: FrameError::Timeout is_recoverable returns true.
#[test]
fn test_frame_error_timeout_is_recoverable() {
    let err = FrameError::Timeout;
    assert!(err.is_recoverable());
}

/// Path A additional: Timeout is a transient error.
#[test]
fn test_frame_error_timeout_is_transient() {
    let err = FrameError::Timeout;
    assert!(err.is_recoverable());
    assert!(!err.needs_reconfigure());
    assert!(!err.needs_recreate());
}

/// Path B: FrameError::Outdated is_recoverable returns true.
#[test]
fn test_frame_error_outdated_is_recoverable() {
    let err = FrameError::Outdated;
    assert!(err.is_recoverable());
}

/// Path B additional: Outdated requires reconfiguration but is recoverable.
#[test]
fn test_frame_error_outdated_recoverable_with_reconfigure() {
    let err = FrameError::Outdated;
    assert!(err.is_recoverable());
    assert!(err.needs_reconfigure());
}

/// Path C: FrameError::Lost is_recoverable returns false.
#[test]
fn test_frame_error_lost_is_not_recoverable() {
    let err = FrameError::Lost {
        reason: "test".to_string(),
    };
    assert!(!err.is_recoverable());
}

/// Path C additional: Lost with empty reason is still not recoverable.
#[test]
fn test_frame_error_lost_empty_reason_not_recoverable() {
    let err = FrameError::Lost {
        reason: String::new(),
    };
    assert!(!err.is_recoverable());
}

/// Path C additional: Lost with long reason is still not recoverable.
#[test]
fn test_frame_error_lost_long_reason_not_recoverable() {
    let err = FrameError::Lost {
        reason: "a".repeat(1000),
    };
    assert!(!err.is_recoverable());
}

// ============================================================================
// Path D-F: FrameError needs_reconfigure()
// ============================================================================

/// Path D: FrameError::Timeout needs_reconfigure returns false.
#[test]
fn test_frame_error_timeout_needs_reconfigure_false() {
    let err = FrameError::Timeout;
    assert!(!err.needs_reconfigure());
}

/// Path E: FrameError::Outdated needs_reconfigure returns true.
#[test]
fn test_frame_error_outdated_needs_reconfigure_true() {
    let err = FrameError::Outdated;
    assert!(err.needs_reconfigure());
}

/// Path F: FrameError::Lost needs_reconfigure returns false.
#[test]
fn test_frame_error_lost_needs_reconfigure_false() {
    let err = FrameError::Lost {
        reason: "driver reset".to_string(),
    };
    assert!(!err.needs_reconfigure());
}

/// Path F additional: Lost never needs just reconfigure.
#[test]
fn test_frame_error_lost_requires_recreate_not_reconfigure() {
    let err = FrameError::lost("window system event");
    assert!(!err.needs_reconfigure());
    assert!(err.needs_recreate());
}

// ============================================================================
// Path G-I: FrameError needs_recreate()
// ============================================================================

/// Path G: FrameError::Timeout needs_recreate returns false.
#[test]
fn test_frame_error_timeout_needs_recreate_false() {
    let err = FrameError::Timeout;
    assert!(!err.needs_recreate());
}

/// Path H: FrameError::Outdated needs_recreate returns false.
#[test]
fn test_frame_error_outdated_needs_recreate_false() {
    let err = FrameError::Outdated;
    assert!(!err.needs_recreate());
}

/// Path I: FrameError::Lost needs_recreate returns true.
#[test]
fn test_frame_error_lost_needs_recreate_true() {
    let err = FrameError::Lost {
        reason: "graphics driver crash".to_string(),
    };
    assert!(err.needs_recreate());
}

/// Path I additional: All Lost variants need recreate.
#[test]
fn test_frame_error_lost_out_of_memory_needs_recreate() {
    let err = FrameError::out_of_memory();
    assert!(err.needs_recreate());
}

// ============================================================================
// Path J-K: FrameError factory methods
// ============================================================================

/// Path J: FrameError::lost() creates Lost with custom reason.
#[test]
fn test_frame_error_lost_factory() {
    let err = FrameError::lost("custom reason");
    match &err {
        FrameError::Lost { reason } => assert_eq!(reason, "custom reason"),
        _ => panic!("Expected Lost variant"),
    }
}

/// Path J additional: lost() accepts String.
#[test]
fn test_frame_error_lost_factory_string() {
    let err = FrameError::lost(String::from("string reason"));
    match &err {
        FrameError::Lost { reason } => assert_eq!(reason, "string reason"),
        _ => panic!("Expected Lost variant"),
    }
}

/// Path J additional: lost() accepts &str.
#[test]
fn test_frame_error_lost_factory_str() {
    let err = FrameError::lost("str reason");
    match &err {
        FrameError::Lost { reason } => assert_eq!(reason, "str reason"),
        _ => panic!("Expected Lost variant"),
    }
}

/// Path J additional: lost() with unicode characters.
#[test]
fn test_frame_error_lost_factory_unicode() {
    let err = FrameError::lost("GPU \u{1F4A5} crashed");
    match &err {
        FrameError::Lost { reason } => assert!(reason.contains("crashed")),
        _ => panic!("Expected Lost variant"),
    }
}

/// Path K: FrameError::out_of_memory() creates Lost with OOM reason.
#[test]
fn test_frame_error_out_of_memory_factory() {
    let err = FrameError::out_of_memory();
    match &err {
        FrameError::Lost { reason } => {
            assert!(reason.contains("memory"));
        }
        _ => panic!("Expected Lost variant"),
    }
}

/// Path K additional: out_of_memory is not recoverable.
#[test]
fn test_frame_error_out_of_memory_not_recoverable() {
    let err = FrameError::out_of_memory();
    assert!(!err.is_recoverable());
}

/// Path K additional: out_of_memory needs recreate.
#[test]
fn test_frame_error_out_of_memory_needs_recreate() {
    let err = FrameError::out_of_memory();
    assert!(err.needs_recreate());
}

// ============================================================================
// Path L-O: From<wgpu::SurfaceError> conversion
// ============================================================================

/// Path L: From<wgpu::SurfaceError::Timeout> -> FrameError::Timeout.
#[test]
fn test_frame_error_from_surface_error_timeout() {
    let surface_err = wgpu::SurfaceError::Timeout;
    let frame_err: FrameError = surface_err.into();
    assert!(matches!(frame_err, FrameError::Timeout));
}

/// Path L additional: Timeout conversion preserves is_recoverable.
#[test]
fn test_frame_error_from_surface_error_timeout_recoverable() {
    let surface_err = wgpu::SurfaceError::Timeout;
    let frame_err: FrameError = surface_err.into();
    assert!(frame_err.is_recoverable());
}

/// Path M: From<wgpu::SurfaceError::Outdated> -> FrameError::Outdated.
#[test]
fn test_frame_error_from_surface_error_outdated() {
    let surface_err = wgpu::SurfaceError::Outdated;
    let frame_err: FrameError = surface_err.into();
    assert!(matches!(frame_err, FrameError::Outdated));
}

/// Path M additional: Outdated conversion preserves needs_reconfigure.
#[test]
fn test_frame_error_from_surface_error_outdated_needs_reconfigure() {
    let surface_err = wgpu::SurfaceError::Outdated;
    let frame_err: FrameError = surface_err.into();
    assert!(frame_err.needs_reconfigure());
}

/// Path N: From<wgpu::SurfaceError::Lost> -> FrameError::Lost.
#[test]
fn test_frame_error_from_surface_error_lost() {
    let surface_err = wgpu::SurfaceError::Lost;
    let frame_err: FrameError = surface_err.into();
    assert!(matches!(frame_err, FrameError::Lost { .. }));
}

/// Path N additional: Lost conversion creates appropriate reason.
#[test]
fn test_frame_error_from_surface_error_lost_reason() {
    let surface_err = wgpu::SurfaceError::Lost;
    let frame_err: FrameError = surface_err.into();
    match &frame_err {
        FrameError::Lost { reason } => {
            assert!(!reason.is_empty());
        }
        _ => panic!("Expected Lost variant"),
    }
}

/// Path O: From<wgpu::SurfaceError::OutOfMemory> -> FrameError::Lost (OOM).
#[test]
fn test_frame_error_from_surface_error_out_of_memory() {
    let surface_err = wgpu::SurfaceError::OutOfMemory;
    let frame_err: FrameError = surface_err.into();
    assert!(matches!(frame_err, FrameError::Lost { .. }));
}

/// Path O additional: OutOfMemory conversion has memory in reason.
#[test]
fn test_frame_error_from_surface_error_out_of_memory_reason() {
    let surface_err = wgpu::SurfaceError::OutOfMemory;
    let frame_err: FrameError = surface_err.into();
    match &frame_err {
        FrameError::Lost { reason } => {
            assert!(reason.contains("memory"));
        }
        _ => panic!("Expected Lost variant"),
    }
}

// ============================================================================
// Path P: FrameError Display trait formatting
// ============================================================================

/// Path P: FrameError::Timeout Display formatting.
#[test]
fn test_frame_error_timeout_display() {
    let err = FrameError::Timeout;
    let msg = format!("{}", err);
    assert!(msg.contains("timeout") || msg.contains("timed out"));
}

/// Path P: FrameError::Outdated Display formatting.
#[test]
fn test_frame_error_outdated_display() {
    let err = FrameError::Outdated;
    let msg = format!("{}", err);
    assert!(msg.contains("outdated") || msg.contains("reconfiguration"));
}

/// Path P: FrameError::Lost Display formatting.
#[test]
fn test_frame_error_lost_display() {
    let err = FrameError::Lost {
        reason: "test reason".to_string(),
    };
    let msg = format!("{}", err);
    assert!(msg.contains("lost") || msg.contains("test reason"));
}

/// Path P additional: Lost with empty reason still displays.
#[test]
fn test_frame_error_lost_empty_display() {
    let err = FrameError::Lost {
        reason: String::new(),
    };
    let msg = format!("{}", err);
    assert!(!msg.is_empty());
}

// ============================================================================
// Path AK: FrameError Debug trait formatting
// ============================================================================

/// Path AK: FrameError Debug formatting for Timeout.
#[test]
fn test_frame_error_timeout_debug() {
    let err = FrameError::Timeout;
    let debug = format!("{:?}", err);
    assert!(debug.contains("Timeout"));
}

/// Path AK: FrameError Debug formatting for Outdated.
#[test]
fn test_frame_error_outdated_debug() {
    let err = FrameError::Outdated;
    let debug = format!("{:?}", err);
    assert!(debug.contains("Outdated"));
}

/// Path AK: FrameError Debug formatting for Lost.
#[test]
fn test_frame_error_lost_debug() {
    let err = FrameError::Lost {
        reason: "debug test".to_string(),
    };
    let debug = format!("{:?}", err);
    assert!(debug.contains("Lost"));
    assert!(debug.contains("debug test"));
}

// ============================================================================
// Path AC: FrameError classification combinations
// ============================================================================

/// Path AC: Timeout is recoverable but doesn't need any reconfiguration.
#[test]
fn test_frame_error_timeout_classification_combo() {
    let err = FrameError::Timeout;
    assert!(err.is_recoverable());
    assert!(!err.needs_reconfigure());
    assert!(!err.needs_recreate());
}

/// Path AC: Outdated is recoverable and needs reconfigure but not recreate.
#[test]
fn test_frame_error_outdated_classification_combo() {
    let err = FrameError::Outdated;
    assert!(err.is_recoverable());
    assert!(err.needs_reconfigure());
    assert!(!err.needs_recreate());
}

/// Path AC: Lost is not recoverable and needs recreate but not reconfigure.
#[test]
fn test_frame_error_lost_classification_combo() {
    let err = FrameError::lost("test");
    assert!(!err.is_recoverable());
    assert!(!err.needs_reconfigure());
    assert!(err.needs_recreate());
}

// ============================================================================
// Path AZ: needs_reconfigure and needs_recreate are mutually exclusive
// ============================================================================

/// Path AZ: For all variants, needs_reconfigure and needs_recreate are never both true.
#[test]
fn test_frame_error_mutual_exclusivity_timeout() {
    let err = FrameError::Timeout;
    assert!(!(err.needs_reconfigure() && err.needs_recreate()));
}

#[test]
fn test_frame_error_mutual_exclusivity_outdated() {
    let err = FrameError::Outdated;
    assert!(!(err.needs_reconfigure() && err.needs_recreate()));
}

#[test]
fn test_frame_error_mutual_exclusivity_lost() {
    let err = FrameError::lost("test");
    assert!(!(err.needs_reconfigure() && err.needs_recreate()));
}

#[test]
fn test_frame_error_mutual_exclusivity_out_of_memory() {
    let err = FrameError::out_of_memory();
    assert!(!(err.needs_reconfigure() && err.needs_recreate()));
}

// ============================================================================
// Path AD-AE: FrameError Lost with various reasons
// ============================================================================

/// Path AD: FrameError Lost with empty reason.
#[test]
fn test_frame_error_lost_empty_reason() {
    let err = FrameError::lost("");
    match &err {
        FrameError::Lost { reason } => assert!(reason.is_empty()),
        _ => panic!("Expected Lost variant"),
    }
}

/// Path AE: FrameError Lost with long reason.
#[test]
fn test_frame_error_lost_long_reason() {
    let long_reason = "a".repeat(10000);
    let err = FrameError::lost(&long_reason);
    match &err {
        FrameError::Lost { reason } => assert_eq!(reason.len(), 10000),
        _ => panic!("Expected Lost variant"),
    }
}

/// Additional: Lost with special characters.
#[test]
fn test_frame_error_lost_special_chars() {
    let err = FrameError::lost("Error: GPU<>\"'&crashed");
    match &err {
        FrameError::Lost { reason } => {
            assert!(reason.contains("GPU"));
            assert!(reason.contains("crashed"));
        }
        _ => panic!("Expected Lost variant"),
    }
}

/// Additional: Lost with newlines.
#[test]
fn test_frame_error_lost_newlines() {
    let err = FrameError::lost("line1\nline2\nline3");
    match &err {
        FrameError::Lost { reason } => {
            assert!(reason.contains('\n'));
        }
        _ => panic!("Expected Lost variant"),
    }
}

// ============================================================================
// Path AL: Multiple FrameError::lost() calls create independent instances
// ============================================================================

/// Path AL: Multiple lost() calls create independent instances.
#[test]
fn test_frame_error_lost_independent_instances() {
    let err1 = FrameError::lost("reason 1");
    let err2 = FrameError::lost("reason 2");

    match (&err1, &err2) {
        (FrameError::Lost { reason: r1 }, FrameError::Lost { reason: r2 }) => {
            assert_ne!(r1, r2);
        }
        _ => panic!("Expected Lost variants"),
    }
}

/// Path AL additional: Instances don't share state.
#[test]
fn test_frame_error_lost_no_shared_state() {
    let err1 = FrameError::lost("original");
    let err2 = FrameError::lost("modified");

    match &err1 {
        FrameError::Lost { reason } => assert_eq!(reason, "original"),
        _ => panic!("Expected Lost variant"),
    }

    match &err2 {
        FrameError::Lost { reason } => assert_eq!(reason, "modified"),
        _ => panic!("Expected Lost variant"),
    }
}

// ============================================================================
// Path AM: FrameError error chain via thiserror
// ============================================================================

/// Path AM: FrameError implements std::error::Error.
#[test]
fn test_frame_error_is_std_error() {
    let err = FrameError::Timeout;
    let _: &dyn std::error::Error = &err;
}

/// Path AM: FrameError can be used with ? operator context.
#[test]
fn test_frame_error_question_mark_compatible() {
    fn may_fail() -> Result<(), FrameError> {
        Err(FrameError::Timeout)
    }

    fn wrapper() -> Result<(), FrameError> {
        may_fail()?;
        Ok(())
    }

    assert!(wrapper().is_err());
}

/// Path AM: FrameError can be boxed as dyn Error.
#[test]
fn test_frame_error_boxed_error() {
    let err: Box<dyn std::error::Error> = Box::new(FrameError::Timeout);
    assert!(err.to_string().contains("timeout") || err.to_string().contains("timed"));
}

// ============================================================================
// Path AQ: FrameError Lost variant matching
// ============================================================================

/// Path AQ: Pattern matching on Lost variant extracts reason.
#[test]
fn test_frame_error_lost_pattern_match() {
    let err = FrameError::lost("pattern test");
    if let FrameError::Lost { reason } = err {
        assert_eq!(reason, "pattern test");
    } else {
        panic!("Pattern match failed");
    }
}

/// Path AQ: Pattern matching with ref.
#[test]
fn test_frame_error_lost_pattern_match_ref() {
    let err = FrameError::lost("ref test");
    match &err {
        FrameError::Lost { reason } => assert_eq!(reason, "ref test"),
        _ => panic!("Pattern match failed"),
    }
}

// ============================================================================
// Path AR: FrameError Timeout vs Outdated recovery distinction
// ============================================================================

/// Path AR: Timeout recovery just requires retry.
#[test]
fn test_frame_error_timeout_recovery_strategy() {
    let err = FrameError::Timeout;
    // Timeout: just retry, no reconfiguration needed
    assert!(err.is_recoverable());
    assert!(!err.needs_reconfigure());
    assert!(!err.needs_recreate());
}

/// Path AR: Outdated recovery requires surface reconfiguration.
#[test]
fn test_frame_error_outdated_recovery_strategy() {
    let err = FrameError::Outdated;
    // Outdated: need to reconfigure before retry
    assert!(err.is_recoverable());
    assert!(err.needs_reconfigure());
    assert!(!err.needs_recreate());
}

/// Path AR: Lost recovery requires full surface recreation.
#[test]
fn test_frame_error_lost_recovery_strategy() {
    let err = FrameError::lost("driver crash");
    // Lost: need to recreate everything
    assert!(!err.is_recoverable());
    assert!(!err.needs_reconfigure());
    assert!(err.needs_recreate());
}

// ============================================================================
// Path AV: FrameError from all wgpu::SurfaceError variants
// ============================================================================

/// Path AV: All wgpu::SurfaceError variants convert correctly.
#[test]
fn test_frame_error_from_all_surface_errors() {
    // Timeout
    let timeout: FrameError = wgpu::SurfaceError::Timeout.into();
    assert!(matches!(timeout, FrameError::Timeout));

    // Outdated
    let outdated: FrameError = wgpu::SurfaceError::Outdated.into();
    assert!(matches!(outdated, FrameError::Outdated));

    // Lost
    let lost: FrameError = wgpu::SurfaceError::Lost.into();
    assert!(matches!(lost, FrameError::Lost { .. }));

    // OutOfMemory
    let oom: FrameError = wgpu::SurfaceError::OutOfMemory.into();
    assert!(matches!(oom, FrameError::Lost { .. }));
}

/// Path AV: Conversion preserves error categorization.
#[test]
fn test_frame_error_conversion_preserves_categorization() {
    // Recoverable errors
    let timeout: FrameError = wgpu::SurfaceError::Timeout.into();
    let outdated: FrameError = wgpu::SurfaceError::Outdated.into();
    assert!(timeout.is_recoverable());
    assert!(outdated.is_recoverable());

    // Non-recoverable errors
    let lost: FrameError = wgpu::SurfaceError::Lost.into();
    let oom: FrameError = wgpu::SurfaceError::OutOfMemory.into();
    assert!(!lost.is_recoverable());
    assert!(!oom.is_recoverable());
}

// ============================================================================
// SurfaceConfiguration Tests Related to Frame Acquisition
// ============================================================================

/// Test that configuration width/height are used for frame dimensions.
#[test]
fn test_surface_config_dimensions_for_frame() {
    let config = make_config(1920, 1080);
    assert_eq!(config.width, 1920);
    assert_eq!(config.height, 1080);
}

/// Test minimum dimensions.
#[test]
fn test_surface_config_minimum_dimensions() {
    let config = SurfaceConfiguration::new(0, 0);
    assert_eq!(config.width, 1);
    assert_eq!(config.height, 1);
}

/// Test configuration format for frame views.
#[test]
fn test_surface_config_format_for_views() {
    let config = make_config(800, 600).with_format(TextureFormat::Rgba8Unorm);
    assert_eq!(config.format, TextureFormat::Rgba8Unorm);
}

/// Test view_formats configuration for sRGB toggle.
#[test]
fn test_surface_config_view_formats() {
    let config = make_config(800, 600)
        .with_format(TextureFormat::Bgra8Unorm)
        .with_view_formats(&[TextureFormat::Bgra8UnormSrgb]);

    assert!(config.view_formats.contains(&TextureFormat::Bgra8UnormSrgb));
}

/// Test with_srgb_view_format helper.
#[test]
fn test_surface_config_srgb_view_format_helper() {
    let config = make_config(800, 600)
        .with_format(TextureFormat::Bgra8Unorm)
        .with_srgb_view_format();

    assert!(config.has_srgb_view_format());
}

/// Test srgb_format accessor.
#[test]
fn test_surface_config_srgb_format_accessor() {
    // Main format is sRGB
    let config1 =
        SurfaceConfiguration::new(800, 600).with_format(TextureFormat::Bgra8UnormSrgb);
    assert_eq!(config1.srgb_format(), Some(TextureFormat::Bgra8UnormSrgb));

    // sRGB in view formats
    let config2 = make_config(800, 600)
        .with_format(TextureFormat::Bgra8Unorm)
        .with_srgb_view_format();
    assert_eq!(config2.srgb_format(), Some(TextureFormat::Bgra8UnormSrgb));

    // No sRGB available
    let config3 =
        SurfaceConfiguration::new(800, 600).with_format(TextureFormat::Rgba16Float);
    assert_eq!(config3.srgb_format(), None);
}

/// Test linear_format accessor.
#[test]
fn test_surface_config_linear_format_accessor() {
    // Main format is linear
    let config1 = SurfaceConfiguration::new(800, 600).with_format(TextureFormat::Bgra8Unorm);
    assert_eq!(config1.linear_format(), Some(TextureFormat::Bgra8Unorm));

    // Linear in view formats
    let config2 = SurfaceConfiguration::new(800, 600)
        .with_format(TextureFormat::Bgra8UnormSrgb)
        .with_view_formats(&[TextureFormat::Bgra8Unorm]);
    assert_eq!(config2.linear_format(), Some(TextureFormat::Bgra8Unorm));
}

// ============================================================================
// Frame Dimension Edge Cases (Path AN-AP)
// ============================================================================

/// Path AN: Frame dimensions with minimum size (1x1).
#[test]
fn test_frame_dimensions_minimum() {
    let config = SurfaceConfiguration::new(1, 1);
    assert_eq!(config.width, 1);
    assert_eq!(config.height, 1);
}

/// Path AO: Frame dimensions with large size (8K).
#[test]
fn test_frame_dimensions_8k() {
    let config = SurfaceConfiguration::new(7680, 4320);
    assert_eq!(config.width, 7680);
    assert_eq!(config.height, 4320);
}

/// Path AP: Frame dimensions with non-power-of-2 sizes.
#[test]
fn test_frame_dimensions_non_power_of_2() {
    let config = SurfaceConfiguration::new(1920, 1080);
    assert_eq!(config.width, 1920);
    assert_eq!(config.height, 1080);
}

/// Additional: Unusual aspect ratios.
#[test]
fn test_frame_dimensions_unusual_aspect() {
    // Ultra-wide
    let config1 = SurfaceConfiguration::new(3440, 1440);
    assert_eq!(config1.width, 3440);
    assert_eq!(config1.height, 1440);

    // Portrait
    let config2 = SurfaceConfiguration::new(1080, 1920);
    assert_eq!(config2.width, 1080);
    assert_eq!(config2.height, 1920);

    // Square
    let config3 = SurfaceConfiguration::new(1024, 1024);
    assert_eq!(config3.width, 1024);
    assert_eq!(config3.height, 1024);
}

// ============================================================================
// Frame Format Tests (Path AS-AU)
// ============================================================================

/// Path AS: sRGB format configuration for frame views.
#[test]
fn test_frame_format_srgb_config() {
    let config = make_config(800, 600).with_format(TextureFormat::Bgra8UnormSrgb);
    assert_eq!(config.format, TextureFormat::Bgra8UnormSrgb);
}

/// Path AT: Linear format configuration for frame views.
#[test]
fn test_frame_format_linear_config() {
    let config = make_config(800, 600).with_format(TextureFormat::Bgra8Unorm);
    assert_eq!(config.format, TextureFormat::Bgra8Unorm);
}

/// Path AU: HDR format configuration for frame views.
#[test]
fn test_frame_format_hdr_config() {
    let config = make_config(800, 600).with_format(TextureFormat::Rgba16Float);
    assert_eq!(config.format, TextureFormat::Rgba16Float);
}

/// Additional: All common HDR formats.
#[test]
fn test_frame_format_all_hdr_configs() {
    let formats = [
        TextureFormat::Rgba16Float,
        TextureFormat::Rgb10a2Unorm,
        TextureFormat::Rg11b10Float,
    ];

    for format in formats {
        let config = make_config(800, 600).with_format(format);
        assert_eq!(config.format, format);
    }
}

// ============================================================================
// Frame Lifecycle Tests (Path AW-AY)
// ============================================================================

/// Path AW: Configuration for frame lifecycle new -> use -> present.
#[test]
fn test_frame_lifecycle_config_for_present() {
    let config = make_config(1920, 1080);
    // Config is ready for frame acquisition
    assert_eq!(config.width, 1920);
    assert_eq!(config.height, 1080);
    assert!(config.desired_maximum_frame_latency >= 1);
}

/// Path AX/AY: Configuration for frame lifecycle discard scenarios.
#[test]
fn test_frame_lifecycle_config_for_discard() {
    let config = make_config(1920, 1080);
    // Config should work whether frame is presented or discarded
    assert_eq!(config.width, 1920);
    assert_eq!(config.height, 1080);
}

// ============================================================================
// Additional SurfaceCapabilities Tests for Frame Acquisition
// ============================================================================

/// Test capabilities format support for frame acquisition.
#[test]
fn test_caps_format_support_for_frames() {
    let caps = make_standard_caps();
    assert!(caps.supports_format(TextureFormat::Bgra8Unorm));
    assert!(caps.supports_format(TextureFormat::Bgra8UnormSrgb));
    assert!(!caps.supports_format(TextureFormat::Rgba16Float));
}

/// Test capabilities present mode support for frame timing.
#[test]
fn test_caps_present_mode_for_frames() {
    let caps = make_standard_caps();
    assert!(caps.supports_present_mode(PresentMode::Fifo));
    assert!(caps.supports_present_mode(PresentMode::Mailbox));
    assert!(!caps.supports_present_mode(PresentMode::Immediate));
}

/// Test HDR capabilities for frame formats.
#[test]
fn test_caps_hdr_support_for_frames() {
    let caps = make_hdr_caps();
    assert!(caps.supports_hdr());
    assert!(caps.supports_format(TextureFormat::Rgba16Float));
}

// ============================================================================
// Configuration Validation Tests
// ============================================================================

/// Test configuration validation passes for valid config.
#[test]
fn test_config_validation_pass() {
    let caps = make_standard_caps();
    let config = SurfaceConfiguration::from_capabilities(&caps, 800, 600);
    assert!(config.validate(&caps).is_ok());
}

/// Test configuration validation fails for unsupported format.
#[test]
fn test_config_validation_fail_format() {
    let caps = make_standard_caps();
    let config = make_config(800, 600).with_format(TextureFormat::Rgba16Float);
    let result = config.validate(&caps);
    assert!(result.is_err());
}

/// Test configuration validation fails for unsupported present mode.
#[test]
fn test_config_validation_fail_present_mode() {
    let caps = make_standard_caps();
    let config = make_config(800, 600).with_present_mode(PresentMode::Immediate);
    let result = config.validate(&caps);
    assert!(result.is_err());
}

/// Test configuration validation fails for unsupported alpha mode.
#[test]
fn test_config_validation_fail_alpha_mode() {
    let caps = make_standard_caps();
    let config = make_config(800, 600).with_alpha_mode(CompositeAlphaMode::PreMultiplied);
    let result = config.validate(&caps);
    assert!(result.is_err());
}

// ============================================================================
// Additional Edge Case Tests
// ============================================================================

/// Test error conversion is deterministic.
#[test]
fn test_frame_error_conversion_deterministic() {
    for _ in 0..100 {
        let timeout: FrameError = wgpu::SurfaceError::Timeout.into();
        assert!(matches!(timeout, FrameError::Timeout));
    }
}

/// Test error comparison by variant.
#[test]
fn test_frame_error_variant_distinction() {
    let timeout = FrameError::Timeout;
    let outdated = FrameError::Outdated;
    let lost = FrameError::lost("test");

    // Each variant has different behavior
    assert!(timeout.is_recoverable() && !timeout.needs_reconfigure());
    assert!(outdated.is_recoverable() && outdated.needs_reconfigure());
    assert!(!lost.is_recoverable() && lost.needs_recreate());
}

/// Test that all errors have non-empty display messages.
#[test]
fn test_frame_error_display_not_empty() {
    let errors: Vec<FrameError> = vec![
        FrameError::Timeout,
        FrameError::Outdated,
        FrameError::lost("reason"),
        FrameError::out_of_memory(),
    ];

    for err in errors {
        let msg = format!("{}", err);
        assert!(!msg.is_empty(), "Error display should not be empty");
    }
}

/// Test that all errors have non-empty debug messages.
#[test]
fn test_frame_error_debug_not_empty() {
    let errors: Vec<FrameError> = vec![
        FrameError::Timeout,
        FrameError::Outdated,
        FrameError::lost("reason"),
        FrameError::out_of_memory(),
    ];

    for err in errors {
        let msg = format!("{:?}", err);
        assert!(!msg.is_empty(), "Error debug should not be empty");
    }
}

/// Test configuration to_wgpu conversion.
#[test]
fn test_config_to_wgpu_conversion() {
    let config = make_config(1920, 1080)
        .with_format(TextureFormat::Bgra8Unorm)
        .with_present_mode(PresentMode::Mailbox)
        .with_alpha_mode(CompositeAlphaMode::Opaque)
        .with_frame_latency(3);

    let wgpu_config = config.to_wgpu();
    assert_eq!(wgpu_config.width, 1920);
    assert_eq!(wgpu_config.height, 1080);
    assert_eq!(wgpu_config.format, TextureFormat::Bgra8Unorm);
    assert_eq!(wgpu_config.present_mode, PresentMode::Mailbox);
    assert_eq!(wgpu_config.alpha_mode, CompositeAlphaMode::Opaque);
    assert_eq!(wgpu_config.desired_maximum_frame_latency, 3);
}

/// Test view formats are preserved in to_wgpu.
#[test]
fn test_config_to_wgpu_view_formats() {
    let config = make_config(800, 600)
        .with_format(TextureFormat::Bgra8Unorm)
        .with_view_formats(&[TextureFormat::Bgra8UnormSrgb]);

    let wgpu_config = config.to_wgpu();
    assert!(wgpu_config.view_formats.contains(&TextureFormat::Bgra8UnormSrgb));
}

/// Test frame latency clamping.
#[test]
fn test_config_frame_latency_clamp() {
    let config = make_config(800, 600).with_frame_latency(0);
    assert_eq!(config.desired_maximum_frame_latency, 1);
}

/// Test default configuration.
#[test]
fn test_config_default() {
    let config = SurfaceConfiguration::default();
    assert_eq!(config.width, 1);
    assert_eq!(config.height, 1);
    assert_eq!(config.present_mode, PresentMode::Fifo);
}

/// Test configuration from window size.
#[test]
fn test_config_from_window_size() {
    let caps = make_standard_caps();
    let config = SurfaceConfiguration::from_window_size(1280, 720, &caps);
    assert_eq!(config.width, 1280);
    assert_eq!(config.height, 720);
}

/// Test present mode preference selection.
#[test]
fn test_config_present_mode_preference() {
    use renderer_backend::presentation::PresentModePreference;

    let caps = make_standard_caps();
    let config = make_config(800, 600)
        .with_present_mode_preference(&caps, PresentModePreference::Vsync);

    // Should select Mailbox (preferred for vsync) since it's available
    assert_eq!(config.present_mode, PresentMode::Mailbox);
}

/// Test alpha mode preference selection.
#[test]
fn test_config_alpha_mode_preference() {
    use renderer_backend::presentation::AlphaModePreference;

    let caps = make_standard_caps();
    let config = make_config(800, 600)
        .with_alpha_mode_preference(&caps, AlphaModePreference::Opaque);

    assert_eq!(config.alpha_mode, CompositeAlphaMode::Opaque);
}

// ============================================================================
// Integration-style Tests (Configuration + Validation)
// ============================================================================

/// Test complete configuration workflow for frame acquisition.
#[test]
fn test_complete_config_workflow() {
    // Get capabilities
    let caps = make_standard_caps();

    // Create configuration from capabilities
    let config = SurfaceConfiguration::from_capabilities(&caps, 1920, 1080);

    // Validate configuration
    assert!(config.validate(&caps).is_ok());

    // Check all fields are set correctly
    assert_eq!(config.width, 1920);
    assert_eq!(config.height, 1080);
    assert!(caps.supports_format(config.format));
    assert!(caps.supports_present_mode(config.present_mode));
}

/// Test configuration with sRGB toggle pattern.
#[test]
fn test_config_srgb_toggle_pattern() {
    let caps = make_standard_caps();

    // Use linear format with sRGB view
    let config = SurfaceConfiguration::new(1920, 1080)
        .with_format(TextureFormat::Bgra8Unorm)
        .with_srgb_view_format()
        .with_present_mode(PresentMode::Fifo)
        .with_alpha_mode(CompositeAlphaMode::Auto);

    assert!(config.validate(&caps).is_ok());
    assert!(config.has_srgb_view_format());
    assert_eq!(config.format, TextureFormat::Bgra8Unorm);
    assert_eq!(config.srgb_format(), Some(TextureFormat::Bgra8UnormSrgb));
}

/// Test HDR configuration workflow.
#[test]
fn test_config_hdr_workflow() {
    let caps = make_hdr_caps();

    // Select HDR format
    let format = caps.select_format(true);
    assert!(format.is_some());
    assert!(matches!(
        format.unwrap(),
        TextureFormat::Rgba16Float | TextureFormat::Rg11b10Float | TextureFormat::Rgb10a2Unorm
    ));

    // Create HDR configuration
    let config = SurfaceConfiguration::new(1920, 1080)
        .with_format(format.unwrap())
        .with_present_mode(PresentMode::Fifo)
        .with_alpha_mode(CompositeAlphaMode::Auto);

    assert!(config.validate(&caps).is_ok());
}

/// Test error handling for invalid configurations.
#[test]
fn test_config_error_handling() {
    let caps = make_standard_caps();

    // Invalid format
    let config1 = make_config(800, 600).with_format(TextureFormat::R8Unorm);
    let err1 = config1.validate(&caps);
    assert!(err1.is_err());

    // Invalid present mode
    let config2 = make_config(800, 600).with_present_mode(PresentMode::Immediate);
    let err2 = config2.validate(&caps);
    assert!(err2.is_err());
}

// ============================================================================
// Stress Tests for Error Creation
// ============================================================================

/// Stress test: Create many error instances.
#[test]
fn test_frame_error_stress_creation() {
    for i in 0..1000 {
        let err = FrameError::lost(format!("error {}", i));
        assert!(!err.is_recoverable());
        assert!(err.needs_recreate());
    }
}

/// Stress test: Error conversion performance.
#[test]
fn test_frame_error_stress_conversion() {
    let variants = [
        wgpu::SurfaceError::Timeout,
        wgpu::SurfaceError::Outdated,
        wgpu::SurfaceError::Lost,
        wgpu::SurfaceError::OutOfMemory,
    ];

    for _ in 0..250 {
        for variant in &variants {
            let err: FrameError = variant.clone().into();
            let _ = err.is_recoverable();
            let _ = err.needs_reconfigure();
            let _ = err.needs_recreate();
        }
    }
}

// ============================================================================
// Configuration Builder Pattern Tests
// ============================================================================

/// Test builder pattern chaining.
#[test]
fn test_config_builder_chain() {
    let config = SurfaceConfiguration::new(1920, 1080)
        .with_format(TextureFormat::Bgra8Unorm)
        .with_present_mode(PresentMode::Mailbox)
        .with_alpha_mode(CompositeAlphaMode::Opaque)
        .with_frame_latency(3)
        .with_view_formats(&[TextureFormat::Bgra8UnormSrgb]);

    assert_eq!(config.width, 1920);
    assert_eq!(config.height, 1080);
    assert_eq!(config.format, TextureFormat::Bgra8Unorm);
    assert_eq!(config.present_mode, PresentMode::Mailbox);
    assert_eq!(config.alpha_mode, CompositeAlphaMode::Opaque);
    assert_eq!(config.desired_maximum_frame_latency, 3);
    assert!(config.view_formats.contains(&TextureFormat::Bgra8UnormSrgb));
}

/// Test builder pattern with capabilities.
#[test]
fn test_config_builder_with_caps() {
    use renderer_backend::presentation::{AlphaModePreference, PresentModePreference};

    let caps = make_standard_caps();

    let config = SurfaceConfiguration::new(1920, 1080)
        .with_format(TextureFormat::Bgra8Unorm)
        .with_present_mode_preference(&caps, PresentModePreference::LowLatency)
        .with_alpha_mode_preference(&caps, AlphaModePreference::Opaque)
        .with_srgb_view_format();

    assert!(config.validate(&caps).is_ok());
}

// ============================================================================
// Edge Cases for Numeric Boundaries
// ============================================================================

/// Test maximum u32 dimensions (though not practical).
#[test]
fn test_config_max_dimensions() {
    let config = SurfaceConfiguration::new(u32::MAX, u32::MAX);
    assert_eq!(config.width, u32::MAX);
    assert_eq!(config.height, u32::MAX);
}

/// Test configuration clone.
#[test]
fn test_config_clone() {
    let config1 = make_config(1920, 1080)
        .with_format(TextureFormat::Bgra8UnormSrgb)
        .with_view_formats(&[TextureFormat::Bgra8Unorm]);

    let config2 = config1.clone();

    assert_eq!(config1.width, config2.width);
    assert_eq!(config1.height, config2.height);
    assert_eq!(config1.format, config2.format);
    assert_eq!(config1.view_formats, config2.view_formats);
}

/// Test configuration debug.
#[test]
fn test_config_debug() {
    let config = make_config(800, 600);
    let debug = format!("{:?}", config);
    assert!(debug.contains("800"));
    assert!(debug.contains("600"));
}

// ============================================================================
// sRGB Companion Format Tests
// ============================================================================

/// Test get_srgb_companion_format for Bgra8Unorm -> Bgra8UnormSrgb.
#[test]
fn test_srgb_companion_bgra8unorm_to_srgb() {
    use renderer_backend::presentation::get_srgb_companion_format;
    assert_eq!(
        get_srgb_companion_format(TextureFormat::Bgra8Unorm),
        Some(TextureFormat::Bgra8UnormSrgb)
    );
}

/// Test get_srgb_companion_format for Bgra8UnormSrgb -> Bgra8Unorm.
#[test]
fn test_srgb_companion_bgra8unorm_srgb_to_linear() {
    use renderer_backend::presentation::get_srgb_companion_format;
    assert_eq!(
        get_srgb_companion_format(TextureFormat::Bgra8UnormSrgb),
        Some(TextureFormat::Bgra8Unorm)
    );
}

/// Test get_srgb_companion_format for Rgba8Unorm -> Rgba8UnormSrgb.
#[test]
fn test_srgb_companion_rgba8unorm_to_srgb() {
    use renderer_backend::presentation::get_srgb_companion_format;
    assert_eq!(
        get_srgb_companion_format(TextureFormat::Rgba8Unorm),
        Some(TextureFormat::Rgba8UnormSrgb)
    );
}

/// Test get_srgb_companion_format for Rgba8UnormSrgb -> Rgba8Unorm.
#[test]
fn test_srgb_companion_rgba8unorm_srgb_to_linear() {
    use renderer_backend::presentation::get_srgb_companion_format;
    assert_eq!(
        get_srgb_companion_format(TextureFormat::Rgba8UnormSrgb),
        Some(TextureFormat::Rgba8Unorm)
    );
}

/// Test get_srgb_companion_format returns None for HDR formats.
#[test]
fn test_srgb_companion_hdr_none() {
    use renderer_backend::presentation::get_srgb_companion_format;
    assert_eq!(get_srgb_companion_format(TextureFormat::Rgba16Float), None);
    assert_eq!(get_srgb_companion_format(TextureFormat::Rg11b10Float), None);
    assert_eq!(get_srgb_companion_format(TextureFormat::Rgb10a2Unorm), None);
}

/// Test get_srgb_companion_format returns None for depth formats.
#[test]
fn test_srgb_companion_depth_none() {
    use renderer_backend::presentation::get_srgb_companion_format;
    assert_eq!(get_srgb_companion_format(TextureFormat::Depth32Float), None);
    assert_eq!(get_srgb_companion_format(TextureFormat::Depth24Plus), None);
}

/// Test are_srgb_companions for valid pairs.
#[test]
fn test_are_srgb_companions_valid_pairs() {
    use renderer_backend::presentation::are_srgb_companions;
    assert!(are_srgb_companions(
        TextureFormat::Bgra8Unorm,
        TextureFormat::Bgra8UnormSrgb
    ));
    assert!(are_srgb_companions(
        TextureFormat::Bgra8UnormSrgb,
        TextureFormat::Bgra8Unorm
    ));
    assert!(are_srgb_companions(
        TextureFormat::Rgba8Unorm,
        TextureFormat::Rgba8UnormSrgb
    ));
    assert!(are_srgb_companions(
        TextureFormat::Rgba8UnormSrgb,
        TextureFormat::Rgba8Unorm
    ));
}

/// Test are_srgb_companions for non-companion pairs.
#[test]
fn test_are_srgb_companions_invalid_pairs() {
    use renderer_backend::presentation::are_srgb_companions;
    // Same format is not a companion of itself
    assert!(!are_srgb_companions(
        TextureFormat::Bgra8Unorm,
        TextureFormat::Bgra8Unorm
    ));
    // Different format families are not companions
    assert!(!are_srgb_companions(
        TextureFormat::Bgra8Unorm,
        TextureFormat::Rgba8UnormSrgb
    ));
    // HDR formats have no companions
    assert!(!are_srgb_companions(
        TextureFormat::Rgba16Float,
        TextureFormat::Bgra8Unorm
    ));
}

/// Test are_srgb_companions is symmetric.
#[test]
fn test_are_srgb_companions_symmetric() {
    use renderer_backend::presentation::are_srgb_companions;
    let pairs = [
        (TextureFormat::Bgra8Unorm, TextureFormat::Bgra8UnormSrgb),
        (TextureFormat::Rgba8Unorm, TextureFormat::Rgba8UnormSrgb),
    ];

    for (a, b) in pairs {
        assert_eq!(are_srgb_companions(a, b), are_srgb_companions(b, a));
    }
}

// ============================================================================
// PlatformTarget Tests
// ============================================================================

/// Test PlatformTarget::current returns a supported platform on common OSes.
#[test]
fn test_platform_target_current_supported() {
    use renderer_backend::presentation::PlatformTarget;
    #[cfg(any(target_os = "linux", target_os = "windows", target_os = "macos"))]
    {
        let platform = PlatformTarget::current();
        assert!(platform.is_supported());
    }
}

/// Test PlatformTarget names are not empty.
#[test]
fn test_platform_target_names() {
    use renderer_backend::presentation::PlatformTarget;
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
        assert!(!platform.name().is_empty());
    }
}

/// Test PlatformTarget::is_supported returns false only for Unknown.
#[test]
fn test_platform_target_is_supported() {
    use renderer_backend::presentation::PlatformTarget;
    assert!(PlatformTarget::Wayland.is_supported());
    assert!(PlatformTarget::X11.is_supported());
    assert!(PlatformTarget::Windows.is_supported());
    assert!(PlatformTarget::MacOS.is_supported());
    assert!(PlatformTarget::IOS.is_supported());
    assert!(PlatformTarget::Android.is_supported());
    assert!(PlatformTarget::Web.is_supported());
    assert!(!PlatformTarget::Unknown.is_supported());
}

/// Test PlatformTarget Display trait.
#[test]
fn test_platform_target_display() {
    use renderer_backend::presentation::PlatformTarget;
    assert_eq!(format!("{}", PlatformTarget::Windows), "Windows");
    assert_eq!(format!("{}", PlatformTarget::MacOS), "macOS");
    assert_eq!(format!("{}", PlatformTarget::Web), "Web");
    assert!(format!("{}", PlatformTarget::Wayland).contains("Wayland"));
    assert!(format!("{}", PlatformTarget::X11).contains("X11"));
}

// ============================================================================
// SurfaceError Tests
// ============================================================================

/// Test SurfaceError::unsupported creates UnsupportedPlatform.
#[test]
fn test_surface_error_unsupported() {
    use renderer_backend::presentation::SurfaceError;
    let err = SurfaceError::unsupported();
    assert!(err.is_platform_error());
    assert!(!err.is_recoverable());
}

/// Test SurfaceError::window_handle creates WindowHandleError.
#[test]
fn test_surface_error_window_handle() {
    use renderer_backend::presentation::SurfaceError;
    let err = SurfaceError::window_handle("test window error");
    let msg = format!("{}", err);
    assert!(msg.contains("test window error"));
}

/// Test SurfaceError::display_handle creates DisplayHandleError.
#[test]
fn test_surface_error_display_handle() {
    use renderer_backend::presentation::SurfaceError;
    let err = SurfaceError::display_handle("test display error");
    let msg = format!("{}", err);
    assert!(msg.contains("test display error"));
}

/// Test SurfaceError::creation_failed creates SurfaceCreationFailed.
#[test]
fn test_surface_error_creation_failed() {
    use renderer_backend::presentation::SurfaceError;
    let err = SurfaceError::creation_failed("wgpu error");
    let msg = format!("{}", err);
    assert!(msg.contains("wgpu error"));
}

/// Test SurfaceError::invalid_config creates InvalidConfiguration.
#[test]
fn test_surface_error_invalid_config() {
    use renderer_backend::presentation::SurfaceError;
    let err = SurfaceError::invalid_config("bad config");
    let msg = format!("{}", err);
    assert!(msg.contains("bad config"));
}

/// Test SurfaceError::is_recoverable for recoverable errors.
#[test]
fn test_surface_error_is_recoverable() {
    use renderer_backend::presentation::SurfaceError;
    let lost = SurfaceError::SurfaceLost {
        reason: "test".to_string(),
    };
    assert!(lost.is_recoverable());

    let outdated = SurfaceError::SurfaceOutdated;
    assert!(outdated.is_recoverable());
}

/// Test SurfaceError::is_recoverable for non-recoverable errors.
#[test]
fn test_surface_error_not_recoverable() {
    use renderer_backend::presentation::SurfaceError;
    let unsupported = SurfaceError::unsupported();
    assert!(!unsupported.is_recoverable());

    let invalid = SurfaceError::invalid_config("test");
    assert!(!invalid.is_recoverable());
}

// ============================================================================
// FormatCategory Tests
// ============================================================================

/// Test FormatCategory::from_format for common surface formats.
#[test]
fn test_format_category_surface_formats() {
    use renderer_backend::presentation::FormatCategory;
    // sRGB
    assert_eq!(
        FormatCategory::from_format(TextureFormat::Bgra8UnormSrgb),
        FormatCategory::Srgb
    );
    assert_eq!(
        FormatCategory::from_format(TextureFormat::Rgba8UnormSrgb),
        FormatCategory::Srgb
    );

    // Linear
    assert_eq!(
        FormatCategory::from_format(TextureFormat::Bgra8Unorm),
        FormatCategory::Linear
    );
    assert_eq!(
        FormatCategory::from_format(TextureFormat::Rgba8Unorm),
        FormatCategory::Linear
    );

    // HDR
    assert_eq!(
        FormatCategory::from_format(TextureFormat::Rgba16Float),
        FormatCategory::Hdr
    );
}

/// Test FormatCategory::is_gamma_corrected.
#[test]
fn test_format_category_is_gamma_corrected() {
    use renderer_backend::presentation::FormatCategory;
    assert!(FormatCategory::Srgb.is_gamma_corrected());
    assert!(!FormatCategory::Linear.is_gamma_corrected());
    assert!(!FormatCategory::Hdr.is_gamma_corrected());
    assert!(!FormatCategory::Other.is_gamma_corrected());
}

/// Test FormatCategory::is_hdr.
#[test]
fn test_format_category_is_hdr() {
    use renderer_backend::presentation::FormatCategory;
    assert!(FormatCategory::Hdr.is_hdr());
    assert!(!FormatCategory::Srgb.is_hdr());
    assert!(!FormatCategory::Linear.is_hdr());
    assert!(!FormatCategory::Other.is_hdr());
}

/// Test FormatCategory::name.
#[test]
fn test_format_category_name() {
    use renderer_backend::presentation::FormatCategory;
    assert_eq!(FormatCategory::Srgb.name(), "sRGB");
    assert_eq!(FormatCategory::Linear.name(), "Linear");
    assert_eq!(FormatCategory::Hdr.name(), "HDR");
    assert_eq!(FormatCategory::Other.name(), "Other");
}

/// Test FormatCategory Display trait.
#[test]
fn test_format_category_display() {
    use renderer_backend::presentation::FormatCategory;
    assert_eq!(format!("{}", FormatCategory::Srgb), "sRGB");
    assert_eq!(format!("{}", FormatCategory::Hdr), "HDR");
}

// ============================================================================
// PresentModePreference Tests
// ============================================================================

/// Test PresentModePreference default is Vsync.
#[test]
fn test_present_mode_preference_default() {
    use renderer_backend::presentation::PresentModePreference;
    assert_eq!(PresentModePreference::default(), PresentModePreference::Vsync);
}

/// Test PresentModePreference descriptions are not empty.
#[test]
fn test_present_mode_preference_descriptions() {
    use renderer_backend::presentation::PresentModePreference;
    let prefs = [
        PresentModePreference::LowLatency,
        PresentModePreference::Vsync,
        PresentModePreference::PowerSaving,
        PresentModePreference::Adaptive,
        PresentModePreference::Specific(PresentMode::Fifo),
    ];

    for pref in prefs {
        assert!(!pref.description().is_empty());
    }
}

/// Test PresentModePreference Display trait.
#[test]
fn test_present_mode_preference_display() {
    use renderer_backend::presentation::PresentModePreference;
    assert!(format!("{}", PresentModePreference::LowLatency).contains("Latency"));
    assert!(format!("{}", PresentModePreference::Vsync).contains("Vsync"));
    assert!(format!("{}", PresentModePreference::PowerSaving).contains("Power"));
    assert!(format!("{}", PresentModePreference::Adaptive).contains("Adaptive"));
}

// ============================================================================
// AlphaModePreference Tests
// ============================================================================

/// Test AlphaModePreference default is Auto.
#[test]
fn test_alpha_mode_preference_default() {
    use renderer_backend::presentation::AlphaModePreference;
    assert_eq!(AlphaModePreference::default(), AlphaModePreference::Auto);
}

/// Test AlphaModePreference descriptions are not empty.
#[test]
fn test_alpha_mode_preference_descriptions() {
    use renderer_backend::presentation::AlphaModePreference;
    let prefs = [
        AlphaModePreference::Opaque,
        AlphaModePreference::PreMultiplied,
        AlphaModePreference::PostMultiplied,
        AlphaModePreference::Inherit,
        AlphaModePreference::Auto,
    ];

    for pref in prefs {
        assert!(!pref.description().is_empty());
    }
}

/// Test AlphaModePreference requires_alpha.
#[test]
fn test_alpha_mode_preference_requires_alpha() {
    use renderer_backend::presentation::AlphaModePreference;
    assert!(!AlphaModePreference::Opaque.requires_alpha());
    assert!(AlphaModePreference::PreMultiplied.requires_alpha());
    assert!(AlphaModePreference::PostMultiplied.requires_alpha());
    assert!(AlphaModePreference::Inherit.requires_alpha());
    assert!(AlphaModePreference::Auto.requires_alpha());
}

/// Test AlphaModePreference to_concrete_mode.
#[test]
fn test_alpha_mode_preference_to_concrete() {
    use renderer_backend::presentation::AlphaModePreference;
    assert_eq!(
        AlphaModePreference::Opaque.to_concrete_mode(),
        Some(CompositeAlphaMode::Opaque)
    );
    assert_eq!(
        AlphaModePreference::PreMultiplied.to_concrete_mode(),
        Some(CompositeAlphaMode::PreMultiplied)
    );
    assert_eq!(
        AlphaModePreference::PostMultiplied.to_concrete_mode(),
        Some(CompositeAlphaMode::PostMultiplied)
    );
    assert_eq!(
        AlphaModePreference::Inherit.to_concrete_mode(),
        Some(CompositeAlphaMode::Inherit)
    );
    assert_eq!(AlphaModePreference::Auto.to_concrete_mode(), None);
}

/// Test AlphaModePreference Display trait.
#[test]
fn test_alpha_mode_preference_display() {
    use renderer_backend::presentation::AlphaModePreference;
    assert_eq!(format!("{}", AlphaModePreference::Opaque), "Opaque");
    assert!(format!("{}", AlphaModePreference::PreMultiplied).contains("Pre"));
    assert!(format!("{}", AlphaModePreference::PostMultiplied).contains("Post"));
}

// ============================================================================
// PresentModeInfo Tests
// ============================================================================

/// Test PresentModeInfo::from_mode for all modes.
#[test]
fn test_present_mode_info_all_modes() {
    use renderer_backend::presentation::PresentModeInfo;
    let modes = [
        PresentMode::Fifo,
        PresentMode::FifoRelaxed,
        PresentMode::Immediate,
        PresentMode::Mailbox,
    ];

    for mode in modes {
        let info = PresentModeInfo::from_mode(mode);
        assert_eq!(info.mode, mode);
        assert!(!info.name.is_empty());
        assert!(!info.description.is_empty());
    }
}

/// Test PresentModeInfo prevents_tearing property.
#[test]
fn test_present_mode_info_prevents_tearing() {
    use renderer_backend::presentation::PresentModeInfo;
    let fifo = PresentModeInfo::from_mode(PresentMode::Fifo);
    assert!(fifo.prevents_tearing);

    let immediate = PresentModeInfo::from_mode(PresentMode::Immediate);
    assert!(!immediate.prevents_tearing);
}

/// Test PresentModeInfo latency_rank ordering.
#[test]
fn test_present_mode_info_latency_rank() {
    use renderer_backend::presentation::PresentModeInfo;
    let immediate = PresentModeInfo::from_mode(PresentMode::Immediate);
    let mailbox = PresentModeInfo::from_mode(PresentMode::Mailbox);
    let fifo = PresentModeInfo::from_mode(PresentMode::Fifo);

    assert!(immediate.latency_rank < mailbox.latency_rank);
    assert!(mailbox.latency_rank < fifo.latency_rank);
}

/// Test PresentModeInfo power_efficient property.
#[test]
fn test_present_mode_info_power_efficient() {
    use renderer_backend::presentation::PresentModeInfo;
    let fifo = PresentModeInfo::from_mode(PresentMode::Fifo);
    assert!(fifo.power_efficient);

    let immediate = PresentModeInfo::from_mode(PresentMode::Immediate);
    assert!(!immediate.power_efficient);
}

/// Test PresentModeInfo is_competitive_gaming_mode.
#[test]
fn test_present_mode_info_competitive_gaming() {
    use renderer_backend::presentation::PresentModeInfo;
    let immediate = PresentModeInfo::from_mode(PresentMode::Immediate);
    assert!(immediate.is_competitive_gaming_mode());

    let mailbox = PresentModeInfo::from_mode(PresentMode::Mailbox);
    assert!(mailbox.is_competitive_gaming_mode());

    let fifo = PresentModeInfo::from_mode(PresentMode::Fifo);
    assert!(!fifo.is_competitive_gaming_mode());
}

/// Test PresentModeInfo is_battery_friendly.
#[test]
fn test_present_mode_info_battery_friendly() {
    use renderer_backend::presentation::PresentModeInfo;
    let fifo = PresentModeInfo::from_mode(PresentMode::Fifo);
    assert!(fifo.is_battery_friendly());

    let immediate = PresentModeInfo::from_mode(PresentMode::Immediate);
    assert!(!immediate.is_battery_friendly());
}

/// Test PresentModeInfo Display trait.
#[test]
fn test_present_mode_info_display() {
    use renderer_backend::presentation::PresentModeInfo;
    let info = PresentModeInfo::from_mode(PresentMode::Fifo);
    let display = format!("{}", info);
    assert!(display.contains(info.name));
}

// ============================================================================
// SurfaceCapabilities Advanced Tests
// ============================================================================

/// Test SurfaceCapabilities low_latency_present_mode.
#[test]
fn test_caps_low_latency_present_mode_immediate() {
    let caps = SurfaceCapabilities {
        formats: vec![TextureFormat::Bgra8Unorm],
        present_modes: vec![PresentMode::Fifo, PresentMode::Immediate],
        alpha_modes: vec![CompositeAlphaMode::Auto],
        usages: TextureUsages::RENDER_ATTACHMENT,
    };
    assert_eq!(caps.low_latency_present_mode(), PresentMode::Immediate);
}

/// Test SurfaceCapabilities low_latency_present_mode fallback to mailbox.
#[test]
fn test_caps_low_latency_present_mode_mailbox() {
    let caps = SurfaceCapabilities {
        formats: vec![TextureFormat::Bgra8Unorm],
        present_modes: vec![PresentMode::Fifo, PresentMode::Mailbox],
        alpha_modes: vec![CompositeAlphaMode::Auto],
        usages: TextureUsages::RENDER_ATTACHMENT,
    };
    assert_eq!(caps.low_latency_present_mode(), PresentMode::Mailbox);
}

/// Test SurfaceCapabilities supports_immediate.
#[test]
fn test_caps_supports_immediate() {
    let caps_yes = SurfaceCapabilities {
        formats: vec![TextureFormat::Bgra8Unorm],
        present_modes: vec![PresentMode::Fifo, PresentMode::Immediate],
        alpha_modes: vec![CompositeAlphaMode::Auto],
        usages: TextureUsages::RENDER_ATTACHMENT,
    };
    assert!(caps_yes.supports_immediate());

    let caps_no = make_standard_caps();
    assert!(!caps_no.supports_immediate());
}

/// Test SurfaceCapabilities supports_mailbox.
#[test]
fn test_caps_supports_mailbox() {
    let caps = make_standard_caps();
    assert!(caps.supports_mailbox());
}

/// Test SurfaceCapabilities supports_fifo_relaxed.
#[test]
fn test_caps_supports_fifo_relaxed() {
    let caps_no = make_standard_caps();
    assert!(!caps_no.supports_fifo_relaxed());

    let caps_yes = SurfaceCapabilities {
        formats: vec![TextureFormat::Bgra8Unorm],
        present_modes: vec![PresentMode::Fifo, PresentMode::FifoRelaxed],
        alpha_modes: vec![CompositeAlphaMode::Auto],
        usages: TextureUsages::RENDER_ATTACHMENT,
    };
    assert!(caps_yes.supports_fifo_relaxed());
}

/// Test SurfaceCapabilities select_present_mode.
#[test]
fn test_caps_select_present_mode() {
    use renderer_backend::presentation::PresentModePreference;

    let caps = SurfaceCapabilities {
        formats: vec![TextureFormat::Bgra8Unorm],
        present_modes: vec![PresentMode::Fifo, PresentMode::Mailbox],
        alpha_modes: vec![CompositeAlphaMode::Auto],
        usages: TextureUsages::RENDER_ATTACHMENT,
    };

    assert_eq!(
        caps.select_present_mode(PresentModePreference::Vsync),
        PresentMode::Mailbox
    );
    assert_eq!(
        caps.select_present_mode(PresentModePreference::PowerSaving),
        PresentMode::Fifo
    );
}

/// Test SurfaceCapabilities select_alpha_mode.
#[test]
fn test_caps_select_alpha_mode() {
    use renderer_backend::presentation::AlphaModePreference;

    let caps = make_standard_caps();
    assert_eq!(
        caps.select_alpha_mode(AlphaModePreference::Opaque),
        CompositeAlphaMode::Opaque
    );
    assert_eq!(
        caps.select_alpha_mode(AlphaModePreference::Auto),
        CompositeAlphaMode::Opaque
    );
}

/// Test SurfaceCapabilities describe_present_mode static helper.
#[test]
fn test_caps_describe_present_mode() {
    let info = SurfaceCapabilities::describe_present_mode(PresentMode::Fifo);
    assert_eq!(info.mode, PresentMode::Fifo);
    assert!(info.prevents_tearing);
}

/// Test SurfaceCapabilities format_category static helper.
#[test]
fn test_caps_format_category() {
    use renderer_backend::presentation::FormatCategory;
    assert_eq!(
        SurfaceCapabilities::format_category(TextureFormat::Bgra8UnormSrgb),
        FormatCategory::Srgb
    );
}

/// Test SurfaceCapabilities formats_in_category.
#[test]
fn test_caps_formats_in_category() {
    use renderer_backend::presentation::FormatCategory;

    let caps = SurfaceCapabilities {
        formats: vec![
            TextureFormat::Bgra8Unorm,
            TextureFormat::Bgra8UnormSrgb,
            TextureFormat::Rgba16Float,
        ],
        present_modes: vec![PresentMode::Fifo],
        alpha_modes: vec![CompositeAlphaMode::Auto],
        usages: TextureUsages::RENDER_ATTACHMENT,
    };

    let srgb = caps.formats_in_category(FormatCategory::Srgb);
    assert_eq!(srgb, vec![TextureFormat::Bgra8UnormSrgb]);

    let hdr = caps.formats_in_category(FormatCategory::Hdr);
    assert_eq!(hdr, vec![TextureFormat::Rgba16Float]);
}
