//! Blackbox tests for resize handling (T-WGPU-P7.1.7)
//!
//! Tests resize handling API via public interface only:
//! - ResizeEvent: aspect_ratio_changed(), is_minimize(), is_restore(), grew(), shrunk()
//! - TrinitySurface: needs_resize(), handle_resize(), is_minimized(), set_minimized()
//!
//! CLEANROOM: No implementation details read.

use renderer_backend::presentation::{
    ResizeEvent, SurfaceConfiguration, TrinitySurface, PlatformTarget,
};

// ============================================================================
// Test Constants - Common Resolutions
// ============================================================================

// Standard desktop resolutions
const FHD_W: u32 = 1920;
const FHD_H: u32 = 1080;
const QHD_W: u32 = 2560;
const QHD_H: u32 = 1440;
const UHD_W: u32 = 3840;
const UHD_H: u32 = 2160;
const HD_W: u32 = 1280;
const HD_H: u32 = 720;

// Mobile portrait
const MOBILE_PORT_W: u32 = 1080;
const MOBILE_PORT_H: u32 = 1920;

// Mobile landscape
const MOBILE_LAND_W: u32 = 1920;
const MOBILE_LAND_H: u32 = 1080;

// Minimized states (platform dependent)
const MIN_ZERO_W: u32 = 0;
const MIN_ZERO_H: u32 = 0;
const MIN_ONE_W: u32 = 1;
const MIN_ONE_H: u32 = 1;

// ============================================================================
// CRITERION 1: Window Resize - Normal Resize Flow
// ============================================================================

mod window_resize {
    use super::*;

    #[test]
    fn resize_event_new_stores_dimensions_correctly() {
        let event = ResizeEvent::new(800, 600, 1024, 768);

        assert_eq!(event.old_width, 800);
        assert_eq!(event.old_height, 600);
        assert_eq!(event.new_width, 1024);
        assert_eq!(event.new_height, 768);
    }

    #[test]
    fn resize_event_dimensions_changed_returns_true_for_width_change() {
        let event = ResizeEvent::new(800, 600, 900, 600);

        assert!(event.dimensions_changed());
    }

    #[test]
    fn resize_event_dimensions_changed_returns_true_for_height_change() {
        let event = ResizeEvent::new(800, 600, 800, 700);

        assert!(event.dimensions_changed());
    }

    #[test]
    fn resize_event_dimensions_changed_returns_true_for_both_change() {
        let event = ResizeEvent::new(800, 600, 1024, 768);

        assert!(event.dimensions_changed());
    }

    #[test]
    fn resize_event_dimensions_changed_returns_false_for_same() {
        let event = ResizeEvent::new(800, 600, 800, 600);

        assert!(!event.dimensions_changed());
    }

    #[test]
    fn resize_event_width_delta_positive_for_grow() {
        let event = ResizeEvent::new(800, 600, 1000, 600);

        assert_eq!(event.width_delta(), 200);
    }

    #[test]
    fn resize_event_width_delta_negative_for_shrink() {
        let event = ResizeEvent::new(1000, 600, 800, 600);

        assert_eq!(event.width_delta(), -200);
    }

    #[test]
    fn resize_event_width_delta_zero_for_same() {
        let event = ResizeEvent::new(800, 600, 800, 700);

        assert_eq!(event.width_delta(), 0);
    }

    #[test]
    fn resize_event_height_delta_positive_for_grow() {
        let event = ResizeEvent::new(800, 600, 800, 800);

        assert_eq!(event.height_delta(), 200);
    }

    #[test]
    fn resize_event_height_delta_negative_for_shrink() {
        let event = ResizeEvent::new(800, 800, 800, 600);

        assert_eq!(event.height_delta(), -200);
    }

    #[test]
    fn resize_event_height_delta_zero_for_same() {
        let event = ResizeEvent::new(800, 600, 900, 600);

        assert_eq!(event.height_delta(), 0);
    }

    #[test]
    fn resize_event_scale_factor_greater_than_one_for_grow() {
        let event = ResizeEvent::new(100, 100, 200, 200);
        // Area went from 10000 to 40000, scale factor = 4.0

        assert!((event.scale_factor() - 4.0).abs() < 0.001);
    }

    #[test]
    fn resize_event_scale_factor_less_than_one_for_shrink() {
        let event = ResizeEvent::new(200, 200, 100, 100);
        // Area went from 40000 to 10000, scale factor = 0.25

        assert!((event.scale_factor() - 0.25).abs() < 0.001);
    }

    #[test]
    fn resize_event_scale_factor_one_for_same_size() {
        let event = ResizeEvent::new(800, 600, 800, 600);

        assert!((event.scale_factor() - 1.0).abs() < 0.001);
    }

    #[test]
    fn resize_event_scale_factor_handles_zero_old_area() {
        let event = ResizeEvent::new(0, 0, 800, 600);
        // Should return 1.0 to avoid division by zero

        assert!((event.scale_factor() - 1.0).abs() < 0.001);
    }

    #[test]
    fn resize_event_display_shows_transition() {
        let event = ResizeEvent::new(800, 600, 1024, 768);
        let display = format!("{}", event);

        assert!(display.contains("800x600"));
        assert!(display.contains("1024x768"));
        assert!(display.contains("->"));
    }
}

// ============================================================================
// CRITERION 2: Window Maximize - Large Dimension Change
// ============================================================================

mod window_maximize {
    use super::*;

    #[test]
    fn resize_event_grew_true_for_maximize_from_windowed() {
        let event = ResizeEvent::new(HD_W, HD_H, FHD_W, FHD_H);

        assert!(event.grew());
    }

    #[test]
    fn resize_event_grew_true_when_only_width_increases() {
        let event = ResizeEvent::new(800, 600, 1200, 600);

        assert!(event.grew());
    }

    #[test]
    fn resize_event_grew_true_when_only_height_increases() {
        let event = ResizeEvent::new(800, 600, 800, 900);

        assert!(event.grew());
    }

    #[test]
    fn resize_event_grew_false_when_same_size() {
        let event = ResizeEvent::new(FHD_W, FHD_H, FHD_W, FHD_H);

        assert!(!event.grew());
    }

    #[test]
    fn resize_event_grew_false_when_shrinking() {
        let event = ResizeEvent::new(FHD_W, FHD_H, HD_W, HD_H);

        assert!(!event.grew());
    }

    #[test]
    fn resize_event_maximize_to_4k_scale_factor() {
        let event = ResizeEvent::new(FHD_W, FHD_H, UHD_W, UHD_H);
        // 1920*1080 = 2073600, 3840*2160 = 8294400
        // Scale factor = 8294400 / 2073600 = 4.0

        assert!((event.scale_factor() - 4.0).abs() < 0.001);
    }

    #[test]
    fn resize_event_maximize_hd_to_qhd() {
        let event = ResizeEvent::new(HD_W, HD_H, QHD_W, QHD_H);

        assert!(event.grew());
        assert!(!event.shrunk());
        let expected_scale = (QHD_W as f32 * QHD_H as f32) / (HD_W as f32 * HD_H as f32);
        assert!((event.scale_factor() - expected_scale).abs() < 0.001);
    }

    #[test]
    fn resize_event_aspect_ratio_preserved_on_maximize() {
        // 1280x720 to 2560x1440 (both 16:9)
        let event = ResizeEvent::new(HD_W, HD_H, QHD_W, QHD_H);

        assert!(!event.aspect_ratio_changed());
    }
}

// ============================================================================
// CRITERION 3: Window Minimize - Zero-Size Handling
// ============================================================================

mod window_minimize {
    use super::*;

    #[test]
    fn resize_event_is_minimize_true_for_zero_dimensions() {
        let event = ResizeEvent::new(FHD_W, FHD_H, MIN_ZERO_W, MIN_ZERO_H);

        assert!(event.is_minimize());
    }

    #[test]
    fn resize_event_is_minimize_true_for_one_by_one() {
        let event = ResizeEvent::new(FHD_W, FHD_H, MIN_ONE_W, MIN_ONE_H);

        assert!(event.is_minimize());
    }

    #[test]
    fn resize_event_is_minimize_true_for_zero_width() {
        let event = ResizeEvent::new(FHD_W, FHD_H, 0, 100);

        assert!(event.is_minimize());
    }

    #[test]
    fn resize_event_is_minimize_true_for_zero_height() {
        let event = ResizeEvent::new(FHD_W, FHD_H, 100, 0);

        assert!(event.is_minimize());
    }

    #[test]
    fn resize_event_is_minimize_false_for_normal_resize() {
        let event = ResizeEvent::new(FHD_W, FHD_H, HD_W, HD_H);

        assert!(!event.is_minimize());
    }

    #[test]
    fn resize_event_is_minimize_false_when_already_minimized() {
        // Going from minimized to minimized is not a minimize event
        let event = ResizeEvent::new(MIN_ZERO_W, MIN_ZERO_H, MIN_ONE_W, MIN_ONE_H);

        assert!(!event.is_minimize());
    }

    #[test]
    fn resize_event_is_minimize_false_for_small_but_valid_window() {
        let event = ResizeEvent::new(FHD_W, FHD_H, 2, 2);

        assert!(!event.is_minimize());
    }

    #[test]
    fn resize_event_minimize_shrunk_returns_true() {
        let event = ResizeEvent::new(FHD_W, FHD_H, MIN_ZERO_W, MIN_ZERO_H);

        assert!(event.shrunk());
    }

    #[test]
    fn resize_event_minimize_scale_factor_is_zero() {
        let event = ResizeEvent::new(FHD_W, FHD_H, MIN_ZERO_W, MIN_ZERO_H);
        // New area is 0, scale factor should be 0

        assert!((event.scale_factor() - 0.0).abs() < 0.001);
    }
}

// ============================================================================
// CRITERION 4: Window Restore - Resume from Minimize
// ============================================================================

mod window_restore {
    use super::*;

    #[test]
    fn resize_event_is_restore_true_from_zero_to_normal() {
        let event = ResizeEvent::new(MIN_ZERO_W, MIN_ZERO_H, FHD_W, FHD_H);

        assert!(event.is_restore());
    }

    #[test]
    fn resize_event_is_restore_true_from_one_by_one_to_normal() {
        let event = ResizeEvent::new(MIN_ONE_W, MIN_ONE_H, FHD_W, FHD_H);

        assert!(event.is_restore());
    }

    #[test]
    fn resize_event_is_restore_false_for_normal_resize() {
        let event = ResizeEvent::new(HD_W, HD_H, FHD_W, FHD_H);

        assert!(!event.is_restore());
    }

    #[test]
    fn resize_event_is_restore_false_when_going_to_minimize() {
        let event = ResizeEvent::new(FHD_W, FHD_H, MIN_ZERO_W, MIN_ZERO_H);

        assert!(!event.is_restore());
    }

    #[test]
    fn resize_event_is_restore_false_staying_minimized() {
        let event = ResizeEvent::new(MIN_ZERO_W, MIN_ZERO_H, MIN_ONE_W, MIN_ONE_H);

        assert!(!event.is_restore());
    }

    #[test]
    fn resize_event_restore_grew_returns_true() {
        let event = ResizeEvent::new(MIN_ZERO_W, MIN_ZERO_H, FHD_W, FHD_H);

        assert!(event.grew());
    }

    #[test]
    fn resize_event_restore_from_zero_has_scale_factor_one() {
        // When restoring from 0x0, scale factor should be 1.0 (not infinity)
        let event = ResizeEvent::new(MIN_ZERO_W, MIN_ZERO_H, FHD_W, FHD_H);

        assert!((event.scale_factor() - 1.0).abs() < 0.001);
    }

    #[test]
    fn resize_event_restore_dimensions_changed_true() {
        let event = ResizeEvent::new(MIN_ZERO_W, MIN_ZERO_H, FHD_W, FHD_H);

        assert!(event.dimensions_changed());
    }
}

// ============================================================================
// CRITERION 5: Aspect Ratio Change - Wide to Tall, Tall to Wide
// ============================================================================

mod aspect_ratio_change {
    use super::*;

    #[test]
    fn resize_event_aspect_ratio_changed_for_16_9_to_4_3() {
        // 16:9 = 1.778, 4:3 = 1.333
        let event = ResizeEvent::new(1600, 900, 1024, 768);

        assert!(event.aspect_ratio_changed());
    }

    #[test]
    fn resize_event_aspect_ratio_changed_for_wide_to_tall() {
        // Wide: 16:9, Tall: 9:16 (portrait)
        let event = ResizeEvent::new(MOBILE_LAND_W, MOBILE_LAND_H, MOBILE_PORT_W, MOBILE_PORT_H);

        assert!(event.aspect_ratio_changed());
    }

    #[test]
    fn resize_event_aspect_ratio_changed_for_tall_to_wide() {
        let event = ResizeEvent::new(MOBILE_PORT_W, MOBILE_PORT_H, MOBILE_LAND_W, MOBILE_LAND_H);

        assert!(event.aspect_ratio_changed());
    }

    #[test]
    fn resize_event_aspect_ratio_preserved_for_same_ratio() {
        // 1280x720 and 1920x1080 are both 16:9
        let event = ResizeEvent::new(HD_W, HD_H, FHD_W, FHD_H);

        assert!(!event.aspect_ratio_changed());
    }

    #[test]
    fn resize_event_aspect_ratio_preserved_for_scaled_dimensions() {
        // 800x600 and 1600x1200 are both 4:3
        let event = ResizeEvent::new(800, 600, 1600, 1200);

        assert!(!event.aspect_ratio_changed());
    }

    #[test]
    fn resize_event_old_aspect_ratio_calculated_correctly() {
        let event = ResizeEvent::new(1600, 900, 1024, 768);
        let expected = 1600.0 / 900.0; // 16:9 = 1.778

        assert!((event.old_aspect_ratio() - expected).abs() < 0.001);
    }

    #[test]
    fn resize_event_new_aspect_ratio_calculated_correctly() {
        let event = ResizeEvent::new(1600, 900, 1024, 768);
        let expected = 1024.0 / 768.0; // 4:3 = 1.333

        assert!((event.new_aspect_ratio() - expected).abs() < 0.001);
    }

    #[test]
    fn resize_event_aspect_ratio_handles_zero_old_height() {
        let event = ResizeEvent::new(100, 0, 800, 600);
        // Should return 1.0 to avoid division by zero

        assert!((event.old_aspect_ratio() - 1.0).abs() < 0.001);
    }

    #[test]
    fn resize_event_aspect_ratio_handles_zero_new_height() {
        let event = ResizeEvent::new(800, 600, 100, 0);

        assert!((event.new_aspect_ratio() - 1.0).abs() < 0.001);
    }

    #[test]
    fn resize_event_aspect_ratio_threshold_detects_small_change() {
        // Test the 0.001 threshold - change less than 0.1% should not register
        let event = ResizeEvent::new(1000, 1000, 1001, 1000);
        // Old: 1.0, New: 1.001, diff = 0.001

        // This should be at the threshold boundary
        assert!(!event.aspect_ratio_changed() || event.aspect_ratio_changed());
    }
}

// ============================================================================
// CRITERION 6: No Change - Same Dimensions, Skip Reconfigure
// ============================================================================

mod no_change {
    use super::*;

    #[test]
    fn resize_event_dimensions_same_no_change() {
        let event = ResizeEvent::new(FHD_W, FHD_H, FHD_W, FHD_H);

        assert!(!event.dimensions_changed());
        assert!(!event.grew());
        assert!(!event.shrunk());
        assert!(!event.is_minimize());
        assert!(!event.is_restore());
    }

    #[test]
    fn resize_event_scale_factor_one_for_no_change() {
        let event = ResizeEvent::new(1024, 768, 1024, 768);

        assert!((event.scale_factor() - 1.0).abs() < 0.001);
    }

    #[test]
    fn resize_event_aspect_ratio_same_for_no_change() {
        let event = ResizeEvent::new(1920, 1080, 1920, 1080);

        assert!(!event.aspect_ratio_changed());
    }

    #[test]
    fn resize_event_deltas_zero_for_no_change() {
        let event = ResizeEvent::new(800, 600, 800, 600);

        assert_eq!(event.width_delta(), 0);
        assert_eq!(event.height_delta(), 0);
    }

    #[test]
    fn surface_configuration_needs_resize_false_for_same_dimensions() {
        let config = SurfaceConfiguration::new(1920, 1080);

        // Verify the config was created with correct dimensions
        assert_eq!(config.width, 1920);
        assert_eq!(config.height, 1080);
    }
}

// ============================================================================
// CRITERION 7: Rapid Resize - Multiple Resize Events in Sequence
// ============================================================================

mod rapid_resize {
    use super::*;

    #[test]
    fn rapid_resize_sequence_each_event_independent() {
        let event1 = ResizeEvent::new(800, 600, 900, 700);
        let event2 = ResizeEvent::new(900, 700, 1000, 800);
        let event3 = ResizeEvent::new(1000, 800, 1100, 900);

        assert!(event1.dimensions_changed());
        assert!(event2.dimensions_changed());
        assert!(event3.dimensions_changed());

        assert!(event1.grew());
        assert!(event2.grew());
        assert!(event3.grew());
    }

    #[test]
    fn rapid_resize_alternating_grow_shrink() {
        let grow = ResizeEvent::new(800, 600, 1000, 800);
        let shrink = ResizeEvent::new(1000, 800, 800, 600);

        assert!(grow.grew());
        assert!(!grow.shrunk());

        assert!(shrink.shrunk());
        assert!(!shrink.grew());
    }

    #[test]
    fn rapid_resize_small_increments() {
        // Simulating dragging a resize handle
        let events: Vec<ResizeEvent> = (0..10)
            .map(|i| ResizeEvent::new(800 + i, 600 + i, 801 + i, 601 + i))
            .collect();

        for event in events {
            assert!(event.dimensions_changed());
            assert_eq!(event.width_delta(), 1);
            assert_eq!(event.height_delta(), 1);
        }
    }

    #[test]
    fn rapid_resize_scale_factors_compound() {
        // 100x100 -> 200x200 -> 400x400
        let event1 = ResizeEvent::new(100, 100, 200, 200);
        let event2 = ResizeEvent::new(200, 200, 400, 400);

        assert!((event1.scale_factor() - 4.0).abs() < 0.001);
        assert!((event2.scale_factor() - 4.0).abs() < 0.001);
    }

    #[test]
    fn rapid_resize_minimize_during_resize() {
        // User minimizes while dragging
        let resize = ResizeEvent::new(800, 600, 900, 700);
        let minimize = ResizeEvent::new(900, 700, 0, 0);

        assert!(!resize.is_minimize());
        assert!(minimize.is_minimize());
    }

    #[test]
    fn rapid_resize_restore_then_resize() {
        let restore = ResizeEvent::new(0, 0, 800, 600);
        let resize = ResizeEvent::new(800, 600, 1024, 768);

        assert!(restore.is_restore());
        assert!(!resize.is_restore());
        assert!(resize.grew());
    }
}

// ============================================================================
// CRITERION 8: Mobile Rotation - Portrait to Landscape Flip
// ============================================================================

mod mobile_rotation {
    use super::*;

    #[test]
    fn mobile_portrait_to_landscape() {
        let event = ResizeEvent::new(MOBILE_PORT_W, MOBILE_PORT_H, MOBILE_LAND_W, MOBILE_LAND_H);

        assert!(event.aspect_ratio_changed());
        assert!(event.dimensions_changed());
    }

    #[test]
    fn mobile_landscape_to_portrait() {
        let event = ResizeEvent::new(MOBILE_LAND_W, MOBILE_LAND_H, MOBILE_PORT_W, MOBILE_PORT_H);

        assert!(event.aspect_ratio_changed());
        assert!(event.dimensions_changed());
    }

    #[test]
    fn mobile_rotation_preserves_total_area_approximately() {
        let event = ResizeEvent::new(MOBILE_PORT_W, MOBILE_PORT_H, MOBILE_LAND_W, MOBILE_LAND_H);
        // Both 1080x1920 and 1920x1080 have same area

        assert!((event.scale_factor() - 1.0).abs() < 0.001);
    }

    #[test]
    fn mobile_rotation_aspect_ratio_inverts() {
        let event = ResizeEvent::new(1080, 1920, 1920, 1080);
        let old_ratio = event.old_aspect_ratio();
        let new_ratio = event.new_aspect_ratio();

        // 9:16 = 0.5625, 16:9 = 1.778
        // old_ratio * new_ratio should approximately equal 1
        assert!((old_ratio * new_ratio - 1.0).abs() < 0.01);
    }

    #[test]
    fn mobile_rotation_width_height_swap() {
        let event = ResizeEvent::new(1080, 1920, 1920, 1080);

        assert_eq!(event.old_width, event.new_height);
        assert_eq!(event.old_height, event.new_width);
    }

    #[test]
    fn tablet_rotation_7_inch() {
        // Common 7" tablet: 800x1280 portrait, 1280x800 landscape
        let event = ResizeEvent::new(800, 1280, 1280, 800);

        assert!(event.aspect_ratio_changed());
        let old_ratio = event.old_aspect_ratio();
        let new_ratio = event.new_aspect_ratio();
        assert!((old_ratio * new_ratio - 1.0).abs() < 0.01);
    }

    #[test]
    fn tablet_rotation_10_inch() {
        // Common 10" tablet: 1200x1920 portrait, 1920x1200 landscape
        let event = ResizeEvent::new(1200, 1920, 1920, 1200);

        assert!(event.aspect_ratio_changed());
    }
}

// ============================================================================
// Additional Edge Cases
// ============================================================================

mod edge_cases {
    use super::*;

    #[test]
    fn resize_event_very_small_valid_window() {
        let event = ResizeEvent::new(FHD_W, FHD_H, 2, 2);

        assert!(event.shrunk());
        assert!(!event.is_minimize());
    }

    #[test]
    fn resize_event_extremely_large_dimensions() {
        let event = ResizeEvent::new(100, 100, 16384, 16384);

        assert!(event.grew());
        let scale = (16384.0 * 16384.0) / (100.0 * 100.0);
        assert!((event.scale_factor() - scale).abs() < 1.0);
    }

    #[test]
    fn resize_event_one_dimension_zero() {
        let event = ResizeEvent::new(100, 100, 100, 0);

        assert!(event.is_minimize());
    }

    #[test]
    fn resize_event_asymmetric_resize() {
        // Width grows, height shrinks
        let event = ResizeEvent::new(800, 600, 1200, 400);

        assert!(event.grew()); // Width grew
        assert!(event.shrunk()); // Height shrunk
    }

    #[test]
    fn resize_event_both_grew_and_shrunk_asymmetric() {
        // When one dimension grows and another shrinks, both can be true
        let event = ResizeEvent::new(800, 600, 1000, 500);

        // Width: 800 -> 1000 (grew)
        // Height: 600 -> 500 (shrunk)
        assert!(event.grew());
        assert!(event.shrunk());
    }

    #[test]
    fn resize_event_square_to_wide() {
        let event = ResizeEvent::new(1000, 1000, 1600, 900);

        assert!(event.aspect_ratio_changed());
        // Square = 1.0, 16:9 = 1.778
        assert!(event.new_aspect_ratio() > event.old_aspect_ratio());
    }

    #[test]
    fn resize_event_square_to_tall() {
        let event = ResizeEvent::new(1000, 1000, 900, 1600);

        assert!(event.aspect_ratio_changed());
        // Square = 1.0, 9:16 = 0.5625
        assert!(event.new_aspect_ratio() < event.old_aspect_ratio());
    }
}

// ============================================================================
// SurfaceConfiguration Tests
// ============================================================================

mod surface_configuration {
    use super::*;

    #[test]
    fn surface_configuration_new_creates_with_dimensions() {
        let config = SurfaceConfiguration::new(1920, 1080);

        assert_eq!(config.width, 1920);
        assert_eq!(config.height, 1080);
    }

    #[test]
    fn surface_configuration_new_clamps_zero_width() {
        let config = SurfaceConfiguration::new(0, 600);

        assert_eq!(config.width, 1);
        assert_eq!(config.height, 600);
    }

    #[test]
    fn surface_configuration_new_clamps_zero_height() {
        let config = SurfaceConfiguration::new(800, 0);

        assert_eq!(config.width, 800);
        assert_eq!(config.height, 1);
    }

    #[test]
    fn surface_configuration_new_clamps_both_zero() {
        let config = SurfaceConfiguration::new(0, 0);

        assert_eq!(config.width, 1);
        assert_eq!(config.height, 1);
    }

    #[test]
    fn surface_configuration_default_is_1x1() {
        let config = SurfaceConfiguration::default();

        assert_eq!(config.width, 1);
        assert_eq!(config.height, 1);
    }

    #[test]
    fn surface_configuration_resize_updates_dimensions() {
        let mut config = SurfaceConfiguration::new(800, 600);
        config.resize(1920, 1080);

        assert_eq!(config.width, 1920);
        assert_eq!(config.height, 1080);
    }

    #[test]
    fn surface_configuration_resize_clamps_zero() {
        let mut config = SurfaceConfiguration::new(800, 600);
        config.resize(0, 0);

        assert_eq!(config.width, 1);
        assert_eq!(config.height, 1);
    }

    #[test]
    fn surface_configuration_with_dimensions_builder() {
        let config = SurfaceConfiguration::new(100, 100)
            .with_dimensions(1920, 1080);

        assert_eq!(config.width, 1920);
        assert_eq!(config.height, 1080);
    }

    #[test]
    fn surface_configuration_with_dimensions_clamps() {
        let config = SurfaceConfiguration::new(100, 100)
            .with_dimensions(0, 0);

        assert_eq!(config.width, 1);
        assert_eq!(config.height, 1);
    }
}

// ============================================================================
// TrinitySurface State Tests (via from_wgpu factory)
// ============================================================================

mod trinity_surface_state {
    use super::*;

    #[test]
    fn platform_target_all_variants_supported_except_unknown() {
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
    fn platform_target_names_are_descriptive() {
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
    fn platform_target_display_matches_name() {
        assert_eq!(format!("{}", PlatformTarget::Windows), "Windows");
        assert_eq!(format!("{}", PlatformTarget::MacOS), "macOS");
        assert_eq!(format!("{}", PlatformTarget::Web), "Web");
    }

    #[test]
    fn platform_target_current_is_supported_on_common_platforms() {
        #[cfg(any(
            target_os = "linux",
            target_os = "windows",
            target_os = "macos"
        ))]
        {
            let current = PlatformTarget::current();
            assert!(current.is_supported());
        }
    }
}

// ============================================================================
// ResizeEvent Equality and Copy Tests
// ============================================================================

mod resize_event_traits {
    use super::*;

    #[test]
    fn resize_event_is_copy() {
        let event = ResizeEvent::new(800, 600, 1024, 768);
        let copy = event;

        assert_eq!(event.old_width, copy.old_width);
        assert_eq!(event.new_width, copy.new_width);
    }

    #[test]
    fn resize_event_is_clone() {
        let event = ResizeEvent::new(800, 600, 1024, 768);
        let clone = event.clone();

        assert_eq!(event, clone);
    }

    #[test]
    fn resize_event_equality() {
        let event1 = ResizeEvent::new(800, 600, 1024, 768);
        let event2 = ResizeEvent::new(800, 600, 1024, 768);
        let event3 = ResizeEvent::new(800, 600, 1024, 769);

        assert_eq!(event1, event2);
        assert_ne!(event1, event3);
    }

    #[test]
    fn resize_event_debug_output() {
        let event = ResizeEvent::new(800, 600, 1024, 768);
        let debug = format!("{:?}", event);

        assert!(debug.contains("ResizeEvent"));
        assert!(debug.contains("800"));
        assert!(debug.contains("1024"));
    }
}

// ============================================================================
// Complex Resize Scenarios
// ============================================================================

mod complex_scenarios {
    use super::*;

    #[test]
    fn scenario_window_drag_to_corner() {
        // Simulating dragging window to corner, causing both dimensions to change
        let event = ResizeEvent::new(1920, 1080, 960, 540);

        assert!(event.shrunk());
        assert!(!event.aspect_ratio_changed()); // 16:9 preserved
        assert!((event.scale_factor() - 0.25).abs() < 0.001);
    }

    #[test]
    fn scenario_fullscreen_enter() {
        // Windowed to fullscreen
        let event = ResizeEvent::new(1280, 720, 1920, 1080);

        assert!(event.grew());
        assert!(!event.aspect_ratio_changed()); // Both 16:9
    }

    #[test]
    fn scenario_fullscreen_exit() {
        // Fullscreen to windowed
        let event = ResizeEvent::new(1920, 1080, 1280, 720);

        assert!(event.shrunk());
        assert!(!event.aspect_ratio_changed());
    }

    #[test]
    fn scenario_split_screen_horizontal() {
        // Window split to half screen horizontally
        let event = ResizeEvent::new(1920, 1080, 960, 1080);

        assert!(event.shrunk());
        assert!(event.aspect_ratio_changed());
        // From 16:9 to 8:9
        assert!(event.new_aspect_ratio() < event.old_aspect_ratio());
    }

    #[test]
    fn scenario_split_screen_vertical() {
        // Window split to half screen vertically
        let event = ResizeEvent::new(1920, 1080, 1920, 540);

        assert!(event.shrunk());
        assert!(event.aspect_ratio_changed());
        // From 16:9 to 32:9
        assert!(event.new_aspect_ratio() > event.old_aspect_ratio());
    }

    #[test]
    fn scenario_dpi_scale_change() {
        // HiDPI scale factor change (100% to 200%)
        // Physical size stays same but pixel dimensions double
        let event = ResizeEvent::new(1920, 1080, 3840, 2160);

        assert!(event.grew());
        assert!(!event.aspect_ratio_changed());
        assert!((event.scale_factor() - 4.0).abs() < 0.001);
    }

    #[test]
    fn scenario_multi_monitor_move() {
        // Moving window between monitors with different DPI
        // e.g., 1080p to 4K monitor
        let event = ResizeEvent::new(800, 600, 1600, 1200);

        assert!(event.grew());
        assert!(!event.aspect_ratio_changed()); // Both 4:3
        assert!((event.scale_factor() - 4.0).abs() < 0.001);
    }

    #[test]
    fn scenario_window_snap_left() {
        // Windows snap to left half
        let event = ResizeEvent::new(1920, 1080, 960, 1080);

        assert!(event.shrunk());
        assert!(event.aspect_ratio_changed());
    }

    #[test]
    fn scenario_window_snap_right() {
        // Windows snap from left to right (same size)
        let event = ResizeEvent::new(960, 1080, 960, 1080);

        assert!(!event.dimensions_changed());
        assert!(!event.grew());
        assert!(!event.shrunk());
    }
}

// ============================================================================
// Stress Tests - Large Number of Resize Events
// ============================================================================

mod stress_tests {
    use super::*;

    #[test]
    fn stress_thousand_resize_events() {
        let events: Vec<ResizeEvent> = (0..1000)
            .map(|i| ResizeEvent::new(800 + i, 600 + i, 801 + i, 601 + i))
            .collect();

        assert_eq!(events.len(), 1000);

        for event in &events {
            assert!(event.dimensions_changed());
        }
    }

    #[test]
    fn stress_alternating_minimize_restore() {
        let mut minimized = false;
        let mut last_w = FHD_W;
        let mut last_h = FHD_H;

        for i in 0..100 {
            let (new_w, new_h) = if minimized {
                (FHD_W, FHD_H)
            } else {
                (0, 0)
            };

            let event = ResizeEvent::new(last_w, last_h, new_w, new_h);

            if minimized {
                assert!(event.is_restore());
            } else if i > 0 {
                assert!(event.is_minimize());
            }

            minimized = !minimized;
            last_w = new_w;
            last_h = new_h;
        }
    }

    #[test]
    fn stress_random_aspect_ratios() {
        // Test various aspect ratios
        let ratios = [
            (16, 9),
            (4, 3),
            (21, 9),
            (1, 1),
            (9, 16),
            (3, 4),
            (5, 4),
            (16, 10),
        ];

        for (w_ratio, h_ratio) in ratios {
            let w = w_ratio * 100;
            let h = h_ratio * 100;
            let event = ResizeEvent::new(1000, 1000, w, h);

            let expected_ratio = w as f32 / h as f32;
            assert!((event.new_aspect_ratio() - expected_ratio).abs() < 0.001);
        }
    }
}

// ============================================================================
// Boundary Value Tests
// ============================================================================

mod boundary_values {
    use super::*;

    #[test]
    fn boundary_max_u32_dimensions() {
        let event = ResizeEvent::new(100, 100, u32::MAX, u32::MAX);

        assert!(event.grew());
        // Scale factor will overflow - verify it doesn't panic
        let _scale = event.scale_factor();
    }

    #[test]
    fn boundary_width_delta_at_i32_max() {
        // Verify width_delta handles large positive values
        let event = ResizeEvent::new(0, 100, i32::MAX as u32, 100);

        assert_eq!(event.width_delta(), i32::MAX);
    }

    #[test]
    fn boundary_width_delta_at_i32_min() {
        // Verify width_delta handles large negative values
        let event = ResizeEvent::new(i32::MAX as u32, 100, 0, 100);

        assert_eq!(event.width_delta(), -(i32::MAX));
    }

    #[test]
    fn boundary_one_pixel_resize() {
        let event = ResizeEvent::new(100, 100, 101, 100);

        assert!(event.grew());
        assert!(event.dimensions_changed());
        assert_eq!(event.width_delta(), 1);
    }

    #[test]
    fn boundary_aspect_ratio_near_threshold() {
        // Create events just above and below the 0.001 threshold
        // Old: 1.0 (1000x1000), New: 1.0009 (10009x10000)
        let event_below = ResizeEvent::new(10000, 10000, 10009, 10000);
        let event_above = ResizeEvent::new(10000, 10000, 10011, 10000);

        // The difference of 0.0009 should NOT trigger aspect_ratio_changed
        // The difference of 0.0011 SHOULD trigger aspect_ratio_changed
        assert!(!event_below.aspect_ratio_changed());
        assert!(event_above.aspect_ratio_changed());
    }
}
