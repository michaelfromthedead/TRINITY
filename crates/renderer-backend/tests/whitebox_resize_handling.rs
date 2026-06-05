//! Whitebox structural tests for resize handling in the presentation module.
//!
//! These tests verify the internal structure and behavior of:
//! - ResizeEvent: aspect ratio detection, minimize/restore, grow/shrink, deltas
//! - SurfaceConfiguration: with_dimensions(), resize()
//! - TrinitySurface: needs_resize(), handle_resize(), minimized state, aspect_ratio()
//!
//! T-WGPU-P7.1.7 Resize Handling Whitebox Tests

use renderer_backend::presentation::{
    ResizeEvent, SurfaceConfiguration, SurfaceCapabilities,
    PlatformTarget, TrinitySurface,
};

// ============================================================================
// Helper Functions
// ============================================================================

/// Create a ResizeEvent with the given dimensions.
fn make_resize_event(old_w: u32, old_h: u32, new_w: u32, new_h: u32) -> ResizeEvent {
    ResizeEvent::new(old_w, old_h, new_w, new_h)
}

/// Create default test capabilities for surface configuration tests.
fn make_test_capabilities() -> SurfaceCapabilities {
    SurfaceCapabilities {
        formats: vec![
            wgpu::TextureFormat::Bgra8UnormSrgb,
            wgpu::TextureFormat::Bgra8Unorm,
        ],
        present_modes: vec![
            wgpu::PresentMode::Fifo,
            wgpu::PresentMode::Mailbox,
        ],
        alpha_modes: vec![
            wgpu::CompositeAlphaMode::Auto,
            wgpu::CompositeAlphaMode::Opaque,
        ],
        usages: wgpu::TextureUsages::RENDER_ATTACHMENT,
    }
}

// ============================================================================
// 1. ResizeEvent Construction Tests
// ============================================================================

mod resize_event_construction {
    use super::*;

    #[test]
    fn new_stores_all_dimensions() {
        let event = make_resize_event(800, 600, 1920, 1080);
        assert_eq!(event.old_width, 800);
        assert_eq!(event.old_height, 600);
        assert_eq!(event.new_width, 1920);
        assert_eq!(event.new_height, 1080);
    }

    #[test]
    fn new_with_zero_dimensions() {
        let event = make_resize_event(0, 0, 0, 0);
        assert_eq!(event.old_width, 0);
        assert_eq!(event.old_height, 0);
        assert_eq!(event.new_width, 0);
        assert_eq!(event.new_height, 0);
    }

    #[test]
    fn new_with_max_u32_dimensions() {
        let event = make_resize_event(u32::MAX, u32::MAX, u32::MAX, u32::MAX);
        assert_eq!(event.old_width, u32::MAX);
        assert_eq!(event.old_height, u32::MAX);
        assert_eq!(event.new_width, u32::MAX);
        assert_eq!(event.new_height, u32::MAX);
    }

    #[test]
    fn new_preserves_asymmetric_dimensions() {
        let event = make_resize_event(100, 200, 300, 400);
        assert_eq!(event.old_width, 100);
        assert_eq!(event.old_height, 200);
        assert_eq!(event.new_width, 300);
        assert_eq!(event.new_height, 400);
    }

    #[test]
    fn struct_is_copy() {
        let event1 = make_resize_event(800, 600, 1920, 1080);
        let event2 = event1; // Copy
        assert_eq!(event1.old_width, event2.old_width);
        assert_eq!(event1.new_width, event2.new_width);
    }

    #[test]
    fn struct_is_clone() {
        let event1 = make_resize_event(800, 600, 1920, 1080);
        let event2 = event1.clone();
        assert_eq!(event1, event2);
    }

    #[test]
    fn struct_implements_eq() {
        let event1 = make_resize_event(800, 600, 1920, 1080);
        let event2 = make_resize_event(800, 600, 1920, 1080);
        assert_eq!(event1, event2);
    }

    #[test]
    fn struct_ne_for_different_values() {
        let event1 = make_resize_event(800, 600, 1920, 1080);
        let event2 = make_resize_event(800, 600, 1920, 1081);
        assert_ne!(event1, event2);
    }

    #[test]
    fn struct_implements_debug() {
        let event = make_resize_event(800, 600, 1920, 1080);
        let debug_str = format!("{:?}", event);
        assert!(debug_str.contains("800"));
        assert!(debug_str.contains("600"));
        assert!(debug_str.contains("1920"));
        assert!(debug_str.contains("1080"));
    }
}

// ============================================================================
// 2. ResizeEvent Aspect Ratio Tests
// ============================================================================

mod resize_event_aspect_ratio {
    use super::*;

    // -- old_aspect_ratio tests --

    #[test]
    fn old_aspect_ratio_16_9() {
        let event = make_resize_event(1920, 1080, 800, 600);
        let ratio = event.old_aspect_ratio();
        assert!((ratio - 16.0 / 9.0).abs() < 0.001);
    }

    #[test]
    fn old_aspect_ratio_4_3() {
        let event = make_resize_event(1024, 768, 800, 600);
        let ratio = event.old_aspect_ratio();
        assert!((ratio - 4.0 / 3.0).abs() < 0.001);
    }

    #[test]
    fn old_aspect_ratio_1_1_square() {
        let event = make_resize_event(500, 500, 800, 600);
        let ratio = event.old_aspect_ratio();
        assert!((ratio - 1.0).abs() < 0.001);
    }

    #[test]
    fn old_aspect_ratio_portrait() {
        let event = make_resize_event(1080, 1920, 800, 600);
        let ratio = event.old_aspect_ratio();
        assert!((ratio - 1080.0 / 1920.0).abs() < 0.001);
        assert!(ratio < 1.0);
    }

    #[test]
    fn old_aspect_ratio_zero_height_returns_1() {
        let event = make_resize_event(1920, 0, 800, 600);
        assert_eq!(event.old_aspect_ratio(), 1.0);
    }

    #[test]
    fn old_aspect_ratio_zero_width_zero_height() {
        let event = make_resize_event(0, 0, 800, 600);
        assert_eq!(event.old_aspect_ratio(), 1.0);
    }

    // -- new_aspect_ratio tests --

    #[test]
    fn new_aspect_ratio_16_9() {
        let event = make_resize_event(800, 600, 1920, 1080);
        let ratio = event.new_aspect_ratio();
        assert!((ratio - 16.0 / 9.0).abs() < 0.001);
    }

    #[test]
    fn new_aspect_ratio_4_3() {
        let event = make_resize_event(800, 600, 1024, 768);
        let ratio = event.new_aspect_ratio();
        assert!((ratio - 4.0 / 3.0).abs() < 0.001);
    }

    #[test]
    fn new_aspect_ratio_1_1_square() {
        let event = make_resize_event(800, 600, 500, 500);
        let ratio = event.new_aspect_ratio();
        assert!((ratio - 1.0).abs() < 0.001);
    }

    #[test]
    fn new_aspect_ratio_portrait() {
        let event = make_resize_event(800, 600, 1080, 1920);
        let ratio = event.new_aspect_ratio();
        assert!((ratio - 1080.0 / 1920.0).abs() < 0.001);
        assert!(ratio < 1.0);
    }

    #[test]
    fn new_aspect_ratio_zero_height_returns_1() {
        let event = make_resize_event(800, 600, 1920, 0);
        assert_eq!(event.new_aspect_ratio(), 1.0);
    }

    #[test]
    fn new_aspect_ratio_zero_width_zero_height() {
        let event = make_resize_event(800, 600, 0, 0);
        assert_eq!(event.new_aspect_ratio(), 1.0);
    }

    // -- aspect_ratio_changed tests with 0.1% threshold (0.001) --

    #[test]
    fn aspect_ratio_changed_false_for_same_ratio() {
        // 16:9 to 16:9 (different resolution, same aspect)
        let event = make_resize_event(1920, 1080, 3840, 2160);
        assert!(!event.aspect_ratio_changed());
    }

    #[test]
    fn aspect_ratio_changed_false_for_identical_dimensions() {
        let event = make_resize_event(1920, 1080, 1920, 1080);
        assert!(!event.aspect_ratio_changed());
    }

    #[test]
    fn aspect_ratio_changed_true_for_16_9_to_4_3() {
        // 16:9 to 4:3
        let event = make_resize_event(1920, 1080, 1024, 768);
        assert!(event.aspect_ratio_changed());
    }

    #[test]
    fn aspect_ratio_changed_true_for_landscape_to_portrait() {
        // Landscape to portrait
        let event = make_resize_event(1920, 1080, 1080, 1920);
        assert!(event.aspect_ratio_changed());
    }

    #[test]
    fn aspect_ratio_changed_false_within_threshold() {
        // Create dimensions where ratio difference is < 0.001
        // 1920/1080 = 1.7777...
        // 1921/1081 = 1.7770... (difference ~0.0007, within threshold)
        let event = make_resize_event(1920, 1080, 1921, 1081);
        // This should be close enough to not trigger
        let old_ratio = 1920.0_f32 / 1080.0;
        let new_ratio = 1921.0_f32 / 1081.0;
        if (old_ratio - new_ratio).abs() <= 0.001 {
            assert!(!event.aspect_ratio_changed());
        }
    }

    #[test]
    fn aspect_ratio_changed_true_at_threshold_boundary() {
        // Create dimensions where ratio difference is > 0.001
        // We need old_ratio - new_ratio > 0.001
        // 1000/1000 = 1.0
        // 1000/999 = 1.001001... (difference = 0.001001, just over)
        let event = make_resize_event(1000, 1000, 1000, 999);
        // 1.0 vs 1.001001... difference is 0.001001 > 0.001
        assert!(event.aspect_ratio_changed());
    }

    #[test]
    fn aspect_ratio_changed_handles_zero_dimensions() {
        let event = make_resize_event(0, 0, 1920, 1080);
        // 0/0 returns 1.0, 1920/1080 is ~1.777, difference > 0.001
        assert!(event.aspect_ratio_changed());
    }

    #[test]
    fn aspect_ratio_changed_from_normal_to_zero() {
        let event = make_resize_event(1920, 1080, 0, 0);
        // ~1.777 to 1.0, difference > 0.001
        assert!(event.aspect_ratio_changed());
    }

    #[test]
    fn aspect_ratio_changed_zero_to_zero() {
        let event = make_resize_event(0, 0, 0, 0);
        // Both return 1.0, no change
        assert!(!event.aspect_ratio_changed());
    }

    #[test]
    fn aspect_ratio_changed_precision_edge_case() {
        // Test floating point precision at the boundary
        // 3840/2160 = 1.777... (same as 1920/1080)
        let event = make_resize_event(1920, 1080, 3840, 2160);
        assert!(!event.aspect_ratio_changed());
    }
}

// ============================================================================
// 3. ResizeEvent Minimize Detection Tests
// ============================================================================

mod resize_event_minimize {
    use super::*;

    // -- is_minimize: new dimensions are minimized, old are not --

    #[test]
    fn is_minimize_true_for_0x0() {
        let event = make_resize_event(1920, 1080, 0, 0);
        assert!(event.is_minimize());
    }

    #[test]
    fn is_minimize_true_for_1x1() {
        let event = make_resize_event(1920, 1080, 1, 1);
        assert!(event.is_minimize());
    }

    #[test]
    fn is_minimize_true_for_0_width() {
        let event = make_resize_event(1920, 1080, 0, 100);
        assert!(event.is_minimize());
    }

    #[test]
    fn is_minimize_true_for_0_height() {
        let event = make_resize_event(1920, 1080, 100, 0);
        assert!(event.is_minimize());
    }

    #[test]
    fn is_minimize_false_for_normal_resize() {
        let event = make_resize_event(1920, 1080, 1280, 720);
        assert!(!event.is_minimize());
    }

    #[test]
    fn is_minimize_false_for_2x2() {
        // 2x2 is not considered minimized (only 0x0 or 1x1)
        let event = make_resize_event(1920, 1080, 2, 2);
        assert!(!event.is_minimize());
    }

    #[test]
    fn is_minimize_false_when_old_was_already_minimized() {
        // If old was minimized (0x0), going to another minimized state is not "minimizing"
        let event = make_resize_event(0, 0, 0, 0);
        assert!(!event.is_minimize());
    }

    #[test]
    fn is_minimize_false_when_both_1x1() {
        let event = make_resize_event(1, 1, 1, 1);
        assert!(!event.is_minimize());
    }

    #[test]
    fn is_minimize_false_from_0x0_to_1x1() {
        // Both are minimized states, so not "becoming" minimized
        let event = make_resize_event(0, 0, 1, 1);
        assert!(!event.is_minimize());
    }

    #[test]
    fn is_minimize_with_1x0_old_dimension() {
        // 1x0 is minimized (height = 0), so going to 0x0 is not "minimizing"
        let event = make_resize_event(1, 0, 0, 0);
        assert!(!event.is_minimize());
    }

    #[test]
    fn is_minimize_with_0x1_old_dimension() {
        // 0x1 is minimized (width = 0), so going to 0x0 is not "minimizing"
        let event = make_resize_event(0, 1, 0, 0);
        assert!(!event.is_minimize());
    }
}

// ============================================================================
// 4. ResizeEvent Restore Detection Tests
// ============================================================================

mod resize_event_restore {
    use super::*;

    // -- is_restore: old dimensions are minimized, new are not --

    #[test]
    fn is_restore_true_from_0x0() {
        let event = make_resize_event(0, 0, 1920, 1080);
        assert!(event.is_restore());
    }

    #[test]
    fn is_restore_true_from_1x1() {
        let event = make_resize_event(1, 1, 1920, 1080);
        assert!(event.is_restore());
    }

    #[test]
    fn is_restore_true_from_0_width() {
        let event = make_resize_event(0, 100, 1920, 1080);
        assert!(event.is_restore());
    }

    #[test]
    fn is_restore_true_from_0_height() {
        let event = make_resize_event(100, 0, 1920, 1080);
        assert!(event.is_restore());
    }

    #[test]
    fn is_restore_false_for_normal_resize() {
        let event = make_resize_event(1280, 720, 1920, 1080);
        assert!(!event.is_restore());
    }

    #[test]
    fn is_restore_false_when_new_is_minimized() {
        // Going from minimized to minimized
        let event = make_resize_event(0, 0, 0, 0);
        assert!(!event.is_restore());
    }

    #[test]
    fn is_restore_false_when_new_is_1x1() {
        let event = make_resize_event(0, 0, 1, 1);
        assert!(!event.is_restore());
    }

    #[test]
    fn is_restore_false_when_old_was_valid() {
        // Normal resize is not a restore
        let event = make_resize_event(1920, 1080, 1280, 720);
        assert!(!event.is_restore());
    }

    #[test]
    fn is_restore_true_from_1x0() {
        // 1x0 has height=0, considered minimized
        let event = make_resize_event(1, 0, 1920, 1080);
        assert!(event.is_restore());
    }

    #[test]
    fn is_restore_true_from_0x1() {
        // 0x1 has width=0, considered minimized
        let event = make_resize_event(0, 1, 1920, 1080);
        assert!(event.is_restore());
    }

    #[test]
    fn is_restore_to_2x2() {
        // 2x2 is valid (not minimized)
        let event = make_resize_event(0, 0, 2, 2);
        assert!(event.is_restore());
    }
}

// ============================================================================
// 5. ResizeEvent Grow/Shrink Detection Tests
// ============================================================================

mod resize_event_grow_shrink {
    use super::*;

    // -- grew() tests --

    #[test]
    fn grew_true_width_increased() {
        let event = make_resize_event(800, 600, 1000, 600);
        assert!(event.grew());
    }

    #[test]
    fn grew_true_height_increased() {
        let event = make_resize_event(800, 600, 800, 800);
        assert!(event.grew());
    }

    #[test]
    fn grew_true_both_increased() {
        let event = make_resize_event(800, 600, 1920, 1080);
        assert!(event.grew());
    }

    #[test]
    fn grew_true_width_increased_height_decreased() {
        let event = make_resize_event(800, 600, 1000, 500);
        assert!(event.grew());
    }

    #[test]
    fn grew_true_height_increased_width_decreased() {
        let event = make_resize_event(800, 600, 700, 800);
        assert!(event.grew());
    }

    #[test]
    fn grew_false_same_dimensions() {
        let event = make_resize_event(800, 600, 800, 600);
        assert!(!event.grew());
    }

    #[test]
    fn grew_false_both_decreased() {
        let event = make_resize_event(1920, 1080, 800, 600);
        assert!(!event.grew());
    }

    #[test]
    fn grew_false_width_decreased_height_same() {
        let event = make_resize_event(800, 600, 700, 600);
        assert!(!event.grew());
    }

    #[test]
    fn grew_false_height_decreased_width_same() {
        let event = make_resize_event(800, 600, 800, 500);
        assert!(!event.grew());
    }

    #[test]
    fn grew_from_zero() {
        let event = make_resize_event(0, 0, 100, 100);
        assert!(event.grew());
    }

    // -- shrunk() tests --

    #[test]
    fn shrunk_true_width_decreased() {
        let event = make_resize_event(1000, 600, 800, 600);
        assert!(event.shrunk());
    }

    #[test]
    fn shrunk_true_height_decreased() {
        let event = make_resize_event(800, 800, 800, 600);
        assert!(event.shrunk());
    }

    #[test]
    fn shrunk_true_both_decreased() {
        let event = make_resize_event(1920, 1080, 800, 600);
        assert!(event.shrunk());
    }

    #[test]
    fn shrunk_true_width_decreased_height_increased() {
        let event = make_resize_event(1000, 500, 800, 600);
        assert!(event.shrunk());
    }

    #[test]
    fn shrunk_true_height_decreased_width_increased() {
        let event = make_resize_event(700, 800, 800, 600);
        assert!(event.shrunk());
    }

    #[test]
    fn shrunk_false_same_dimensions() {
        let event = make_resize_event(800, 600, 800, 600);
        assert!(!event.shrunk());
    }

    #[test]
    fn shrunk_false_both_increased() {
        let event = make_resize_event(800, 600, 1920, 1080);
        assert!(!event.shrunk());
    }

    #[test]
    fn shrunk_false_width_increased_height_same() {
        let event = make_resize_event(700, 600, 800, 600);
        assert!(!event.shrunk());
    }

    #[test]
    fn shrunk_false_height_increased_width_same() {
        let event = make_resize_event(800, 500, 800, 600);
        assert!(!event.shrunk());
    }

    #[test]
    fn shrunk_to_zero() {
        let event = make_resize_event(100, 100, 0, 0);
        assert!(event.shrunk());
    }

    // -- Both grew and shrunk --

    #[test]
    fn grew_and_shrunk_width_up_height_down() {
        let event = make_resize_event(800, 600, 1000, 500);
        assert!(event.grew());
        assert!(event.shrunk());
    }

    #[test]
    fn grew_and_shrunk_width_down_height_up() {
        let event = make_resize_event(800, 600, 600, 800);
        assert!(event.grew());
        assert!(event.shrunk());
    }

    #[test]
    fn neither_grew_nor_shrunk_same() {
        let event = make_resize_event(800, 600, 800, 600);
        assert!(!event.grew());
        assert!(!event.shrunk());
    }
}

// ============================================================================
// 6. ResizeEvent Scale Factor Tests
// ============================================================================

mod resize_event_scale_factor {
    use super::*;

    #[test]
    fn scale_factor_2x_growth() {
        // 800x600 = 480,000 pixels
        // 1600x1200 = 1,920,000 pixels
        // scale = 4.0
        let event = make_resize_event(800, 600, 1600, 1200);
        assert!((event.scale_factor() - 4.0).abs() < 0.001);
    }

    #[test]
    fn scale_factor_half() {
        // 1600x1200 = 1,920,000 pixels
        // 800x600 = 480,000 pixels
        // scale = 0.25
        let event = make_resize_event(1600, 1200, 800, 600);
        assert!((event.scale_factor() - 0.25).abs() < 0.001);
    }

    #[test]
    fn scale_factor_1_same_dimensions() {
        let event = make_resize_event(1920, 1080, 1920, 1080);
        assert!((event.scale_factor() - 1.0).abs() < 0.001);
    }

    #[test]
    fn scale_factor_1_same_area_different_aspect() {
        // 1000x1000 = 1,000,000
        // 2000x500 = 1,000,000
        let event = make_resize_event(1000, 1000, 2000, 500);
        assert!((event.scale_factor() - 1.0).abs() < 0.001);
    }

    #[test]
    fn scale_factor_returns_1_for_zero_old_area() {
        let event = make_resize_event(0, 0, 1920, 1080);
        assert_eq!(event.scale_factor(), 1.0);
    }

    #[test]
    fn scale_factor_returns_1_for_zero_old_width() {
        let event = make_resize_event(0, 1080, 1920, 1080);
        assert_eq!(event.scale_factor(), 1.0);
    }

    #[test]
    fn scale_factor_returns_1_for_zero_old_height() {
        let event = make_resize_event(1920, 0, 1920, 1080);
        assert_eq!(event.scale_factor(), 1.0);
    }

    #[test]
    fn scale_factor_to_zero_is_zero() {
        let event = make_resize_event(1920, 1080, 0, 0);
        assert_eq!(event.scale_factor(), 0.0);
    }

    #[test]
    fn scale_factor_large_values() {
        // 4K to 8K = 4x
        let event = make_resize_event(3840, 2160, 7680, 4320);
        assert!((event.scale_factor() - 4.0).abs() < 0.001);
    }

    #[test]
    fn scale_factor_tiny_values() {
        let event = make_resize_event(1, 1, 2, 2);
        assert!((event.scale_factor() - 4.0).abs() < 0.001);
    }

    #[test]
    fn scale_factor_uses_u64_for_area_calculation() {
        // Very large dimensions that would overflow u32 when multiplied
        // 65536 * 65536 = 4,294,967,296 (overflows u32)
        // Should still work due to u64 area calculation
        let event = make_resize_event(32768, 32768, 65536, 65536);
        assert!((event.scale_factor() - 4.0).abs() < 0.001);
    }
}

// ============================================================================
// 7. ResizeEvent Delta Calculations Tests
// ============================================================================

mod resize_event_deltas {
    use super::*;

    // -- dimensions_changed tests --

    #[test]
    fn dimensions_changed_true_width_different() {
        let event = make_resize_event(800, 600, 1000, 600);
        assert!(event.dimensions_changed());
    }

    #[test]
    fn dimensions_changed_true_height_different() {
        let event = make_resize_event(800, 600, 800, 800);
        assert!(event.dimensions_changed());
    }

    #[test]
    fn dimensions_changed_true_both_different() {
        let event = make_resize_event(800, 600, 1920, 1080);
        assert!(event.dimensions_changed());
    }

    #[test]
    fn dimensions_changed_false_same() {
        let event = make_resize_event(1920, 1080, 1920, 1080);
        assert!(!event.dimensions_changed());
    }

    #[test]
    fn dimensions_changed_false_zero_to_zero() {
        let event = make_resize_event(0, 0, 0, 0);
        assert!(!event.dimensions_changed());
    }

    #[test]
    fn dimensions_changed_true_from_zero() {
        let event = make_resize_event(0, 0, 100, 100);
        assert!(event.dimensions_changed());
    }

    // -- width_delta tests --

    #[test]
    fn width_delta_positive_when_grew() {
        let event = make_resize_event(800, 600, 1000, 600);
        assert_eq!(event.width_delta(), 200);
    }

    #[test]
    fn width_delta_negative_when_shrunk() {
        let event = make_resize_event(1000, 600, 800, 600);
        assert_eq!(event.width_delta(), -200);
    }

    #[test]
    fn width_delta_zero_same() {
        let event = make_resize_event(800, 600, 800, 1080);
        assert_eq!(event.width_delta(), 0);
    }

    #[test]
    fn width_delta_large_positive() {
        let event = make_resize_event(100, 100, 10000, 100);
        assert_eq!(event.width_delta(), 9900);
    }

    #[test]
    fn width_delta_large_negative() {
        let event = make_resize_event(10000, 100, 100, 100);
        assert_eq!(event.width_delta(), -9900);
    }

    #[test]
    fn width_delta_from_zero() {
        let event = make_resize_event(0, 0, 1920, 1080);
        assert_eq!(event.width_delta(), 1920);
    }

    #[test]
    fn width_delta_to_zero() {
        let event = make_resize_event(1920, 1080, 0, 0);
        assert_eq!(event.width_delta(), -1920);
    }

    // -- height_delta tests --

    #[test]
    fn height_delta_positive_when_grew() {
        let event = make_resize_event(800, 600, 800, 800);
        assert_eq!(event.height_delta(), 200);
    }

    #[test]
    fn height_delta_negative_when_shrunk() {
        let event = make_resize_event(800, 800, 800, 600);
        assert_eq!(event.height_delta(), -200);
    }

    #[test]
    fn height_delta_zero_same() {
        let event = make_resize_event(800, 600, 1920, 600);
        assert_eq!(event.height_delta(), 0);
    }

    #[test]
    fn height_delta_large_positive() {
        let event = make_resize_event(100, 100, 100, 10000);
        assert_eq!(event.height_delta(), 9900);
    }

    #[test]
    fn height_delta_large_negative() {
        let event = make_resize_event(100, 10000, 100, 100);
        assert_eq!(event.height_delta(), -9900);
    }

    #[test]
    fn height_delta_from_zero() {
        let event = make_resize_event(0, 0, 1920, 1080);
        assert_eq!(event.height_delta(), 1080);
    }

    #[test]
    fn height_delta_to_zero() {
        let event = make_resize_event(1920, 1080, 0, 0);
        assert_eq!(event.height_delta(), -1080);
    }

    // -- Combined delta tests --

    #[test]
    fn deltas_both_positive() {
        let event = make_resize_event(800, 600, 1920, 1080);
        assert_eq!(event.width_delta(), 1120);
        assert_eq!(event.height_delta(), 480);
    }

    #[test]
    fn deltas_both_negative() {
        let event = make_resize_event(1920, 1080, 800, 600);
        assert_eq!(event.width_delta(), -1120);
        assert_eq!(event.height_delta(), -480);
    }

    #[test]
    fn deltas_mixed_signs() {
        let event = make_resize_event(800, 600, 1000, 500);
        assert_eq!(event.width_delta(), 200);
        assert_eq!(event.height_delta(), -100);
    }
}

// ============================================================================
// 8. ResizeEvent Display Implementation Tests
// ============================================================================

mod resize_event_display {
    use super::*;

    #[test]
    fn display_format_normal() {
        let event = make_resize_event(800, 600, 1920, 1080);
        let display = format!("{}", event);
        assert_eq!(display, "800x600 -> 1920x1080");
    }

    #[test]
    fn display_format_zero() {
        let event = make_resize_event(0, 0, 0, 0);
        let display = format!("{}", event);
        assert_eq!(display, "0x0 -> 0x0");
    }

    #[test]
    fn display_format_large() {
        let event = make_resize_event(7680, 4320, 15360, 8640);
        let display = format!("{}", event);
        assert_eq!(display, "7680x4320 -> 15360x8640");
    }

    #[test]
    fn display_format_same() {
        let event = make_resize_event(1920, 1080, 1920, 1080);
        let display = format!("{}", event);
        assert_eq!(display, "1920x1080 -> 1920x1080");
    }
}

// ============================================================================
// 9. SurfaceConfiguration with_dimensions Tests
// ============================================================================

mod surface_configuration_with_dimensions {
    use super::*;

    #[test]
    fn with_dimensions_sets_values() {
        let config = SurfaceConfiguration::new(100, 100)
            .with_dimensions(1920, 1080);
        assert_eq!(config.width, 1920);
        assert_eq!(config.height, 1080);
    }

    #[test]
    fn with_dimensions_clamps_zero_to_one() {
        let config = SurfaceConfiguration::new(100, 100)
            .with_dimensions(0, 0);
        assert_eq!(config.width, 1);
        assert_eq!(config.height, 1);
    }

    #[test]
    fn with_dimensions_clamps_width_only() {
        let config = SurfaceConfiguration::new(100, 100)
            .with_dimensions(0, 500);
        assert_eq!(config.width, 1);
        assert_eq!(config.height, 500);
    }

    #[test]
    fn with_dimensions_clamps_height_only() {
        let config = SurfaceConfiguration::new(100, 100)
            .with_dimensions(500, 0);
        assert_eq!(config.width, 500);
        assert_eq!(config.height, 1);
    }

    #[test]
    fn with_dimensions_preserves_other_fields() {
        let config = SurfaceConfiguration::new(100, 100)
            .with_format(wgpu::TextureFormat::Rgba8Unorm)
            .with_present_mode(wgpu::PresentMode::Mailbox)
            .with_dimensions(1920, 1080);

        assert_eq!(config.width, 1920);
        assert_eq!(config.height, 1080);
        assert_eq!(config.format, wgpu::TextureFormat::Rgba8Unorm);
        assert_eq!(config.present_mode, wgpu::PresentMode::Mailbox);
    }

    #[test]
    fn with_dimensions_chainable() {
        let config = SurfaceConfiguration::new(100, 100)
            .with_dimensions(800, 600)
            .with_dimensions(1920, 1080);
        assert_eq!(config.width, 1920);
        assert_eq!(config.height, 1080);
    }

    #[test]
    fn with_dimensions_max_values() {
        let config = SurfaceConfiguration::new(100, 100)
            .with_dimensions(u32::MAX, u32::MAX);
        assert_eq!(config.width, u32::MAX);
        assert_eq!(config.height, u32::MAX);
    }

    #[test]
    fn with_dimensions_1x1() {
        let config = SurfaceConfiguration::new(100, 100)
            .with_dimensions(1, 1);
        assert_eq!(config.width, 1);
        assert_eq!(config.height, 1);
    }

    #[test]
    fn with_dimensions_typical_resolutions() {
        let resolutions = [
            (640, 480),   // VGA
            (1280, 720),  // 720p
            (1920, 1080), // 1080p
            (2560, 1440), // 1440p
            (3840, 2160), // 4K
            (7680, 4320), // 8K
        ];

        for (w, h) in resolutions {
            let config = SurfaceConfiguration::new(100, 100)
                .with_dimensions(w, h);
            assert_eq!(config.width, w);
            assert_eq!(config.height, h);
        }
    }
}

// ============================================================================
// 10. SurfaceConfiguration resize() Mutating Method Tests
// ============================================================================

mod surface_configuration_resize {
    use super::*;

    #[test]
    fn resize_sets_values() {
        let mut config = SurfaceConfiguration::new(100, 100);
        config.resize(1920, 1080);
        assert_eq!(config.width, 1920);
        assert_eq!(config.height, 1080);
    }

    #[test]
    fn resize_clamps_zero_to_one() {
        let mut config = SurfaceConfiguration::new(100, 100);
        config.resize(0, 0);
        assert_eq!(config.width, 1);
        assert_eq!(config.height, 1);
    }

    #[test]
    fn resize_clamps_width_only() {
        let mut config = SurfaceConfiguration::new(100, 100);
        config.resize(0, 500);
        assert_eq!(config.width, 1);
        assert_eq!(config.height, 500);
    }

    #[test]
    fn resize_clamps_height_only() {
        let mut config = SurfaceConfiguration::new(100, 100);
        config.resize(500, 0);
        assert_eq!(config.width, 500);
        assert_eq!(config.height, 1);
    }

    #[test]
    fn resize_preserves_other_fields() {
        let mut config = SurfaceConfiguration::new(100, 100)
            .with_format(wgpu::TextureFormat::Rgba8Unorm)
            .with_present_mode(wgpu::PresentMode::Mailbox);

        config.resize(1920, 1080);

        assert_eq!(config.width, 1920);
        assert_eq!(config.height, 1080);
        assert_eq!(config.format, wgpu::TextureFormat::Rgba8Unorm);
        assert_eq!(config.present_mode, wgpu::PresentMode::Mailbox);
    }

    #[test]
    fn resize_multiple_times() {
        let mut config = SurfaceConfiguration::new(100, 100);
        config.resize(800, 600);
        assert_eq!(config.width, 800);
        config.resize(1920, 1080);
        assert_eq!(config.width, 1920);
        config.resize(3840, 2160);
        assert_eq!(config.width, 3840);
        assert_eq!(config.height, 2160);
    }

    #[test]
    fn resize_max_values() {
        let mut config = SurfaceConfiguration::new(100, 100);
        config.resize(u32::MAX, u32::MAX);
        assert_eq!(config.width, u32::MAX);
        assert_eq!(config.height, u32::MAX);
    }

    #[test]
    fn resize_to_same_dimensions() {
        let mut config = SurfaceConfiguration::new(1920, 1080);
        config.resize(1920, 1080);
        assert_eq!(config.width, 1920);
        assert_eq!(config.height, 1080);
    }

    #[test]
    fn resize_from_1x1() {
        let mut config = SurfaceConfiguration::new(1, 1);
        config.resize(1920, 1080);
        assert_eq!(config.width, 1920);
        assert_eq!(config.height, 1080);
    }
}

// ============================================================================
// 11. TrinitySurface Simulated State Tests
// ============================================================================

// Note: We cannot create a real TrinitySurface without a window, but we can
// test the helper structures and logic that TrinitySurface relies on.

mod trinity_surface_state_simulation {
    use super::*;

    /// Simulates TrinitySurface state for testing resize logic.
    struct SurfaceStateSimulator {
        current_config: Option<SurfaceConfiguration>,
        minimized: bool,
    }

    impl SurfaceStateSimulator {
        fn new() -> Self {
            Self {
                current_config: None,
                minimized: false,
            }
        }

        fn configure(&mut self, config: SurfaceConfiguration) {
            self.current_config = Some(config);
        }

        fn needs_resize(&self, width: u32, height: u32) -> bool {
            match &self.current_config {
                Some(config) => config.width != width || config.height != height,
                None => true,
            }
        }

        fn is_minimized(&self) -> bool {
            self.minimized
        }

        fn set_minimized(&mut self, minimized: bool) {
            self.minimized = minimized;
        }

        fn aspect_ratio(&self) -> f32 {
            match &self.current_config {
                Some(config) if config.height > 0 => {
                    config.width as f32 / config.height as f32
                }
                _ => 1.0,
            }
        }

        fn dimensions(&self) -> (u32, u32) {
            self.current_config
                .as_ref()
                .map_or((0, 0), |c| (c.width, c.height))
        }

        /// Simulates handle_resize behavior
        fn handle_resize(&mut self, width: u32, height: u32) -> Option<ResizeEvent> {
            // Check for minimize state (0x0 or 1x1)
            let is_minimized = width == 0 || height == 0 || (width == 1 && height == 1);
            let was_minimized = self.minimized;

            if is_minimized {
                self.minimized = true;
                return None;
            }

            let (old_width, old_height) = self.dimensions();

            // Check if resize is actually needed
            if !self.needs_resize(width, height) && !was_minimized {
                return None;
            }

            self.minimized = false;

            // Update config
            if let Some(ref mut config) = self.current_config {
                config.resize(width, height);
            } else {
                self.current_config = Some(SurfaceConfiguration::new(width, height));
            }

            Some(ResizeEvent::new(old_width, old_height, width, height))
        }
    }

    // -- needs_resize tests --

    #[test]
    fn needs_resize_true_when_not_configured() {
        let sim = SurfaceStateSimulator::new();
        assert!(sim.needs_resize(1920, 1080));
    }

    #[test]
    fn needs_resize_true_when_width_differs() {
        let mut sim = SurfaceStateSimulator::new();
        sim.configure(SurfaceConfiguration::new(800, 600));
        assert!(sim.needs_resize(1920, 600));
    }

    #[test]
    fn needs_resize_true_when_height_differs() {
        let mut sim = SurfaceStateSimulator::new();
        sim.configure(SurfaceConfiguration::new(800, 600));
        assert!(sim.needs_resize(800, 1080));
    }

    #[test]
    fn needs_resize_true_when_both_differ() {
        let mut sim = SurfaceStateSimulator::new();
        sim.configure(SurfaceConfiguration::new(800, 600));
        assert!(sim.needs_resize(1920, 1080));
    }

    #[test]
    fn needs_resize_false_when_same() {
        let mut sim = SurfaceStateSimulator::new();
        sim.configure(SurfaceConfiguration::new(1920, 1080));
        assert!(!sim.needs_resize(1920, 1080));
    }

    // -- is_minimized / set_minimized tests --

    #[test]
    fn is_minimized_false_by_default() {
        let sim = SurfaceStateSimulator::new();
        assert!(!sim.is_minimized());
    }

    #[test]
    fn set_minimized_true() {
        let mut sim = SurfaceStateSimulator::new();
        sim.set_minimized(true);
        assert!(sim.is_minimized());
    }

    #[test]
    fn set_minimized_false() {
        let mut sim = SurfaceStateSimulator::new();
        sim.set_minimized(true);
        sim.set_minimized(false);
        assert!(!sim.is_minimized());
    }

    #[test]
    fn set_minimized_multiple_toggles() {
        let mut sim = SurfaceStateSimulator::new();
        for _ in 0..10 {
            sim.set_minimized(true);
            assert!(sim.is_minimized());
            sim.set_minimized(false);
            assert!(!sim.is_minimized());
        }
    }

    // -- aspect_ratio tests --

    #[test]
    fn aspect_ratio_1_when_not_configured() {
        let sim = SurfaceStateSimulator::new();
        assert_eq!(sim.aspect_ratio(), 1.0);
    }

    #[test]
    fn aspect_ratio_16_9() {
        let mut sim = SurfaceStateSimulator::new();
        sim.configure(SurfaceConfiguration::new(1920, 1080));
        assert!((sim.aspect_ratio() - 16.0 / 9.0).abs() < 0.001);
    }

    #[test]
    fn aspect_ratio_4_3() {
        let mut sim = SurfaceStateSimulator::new();
        sim.configure(SurfaceConfiguration::new(1024, 768));
        assert!((sim.aspect_ratio() - 4.0 / 3.0).abs() < 0.001);
    }

    #[test]
    fn aspect_ratio_1_for_square() {
        let mut sim = SurfaceStateSimulator::new();
        sim.configure(SurfaceConfiguration::new(1000, 1000));
        assert!((sim.aspect_ratio() - 1.0).abs() < 0.001);
    }

    #[test]
    fn aspect_ratio_portrait() {
        let mut sim = SurfaceStateSimulator::new();
        sim.configure(SurfaceConfiguration::new(1080, 1920));
        assert!(sim.aspect_ratio() < 1.0);
    }

    // -- handle_resize tests --

    #[test]
    fn handle_resize_returns_event_for_change() {
        let mut sim = SurfaceStateSimulator::new();
        sim.configure(SurfaceConfiguration::new(800, 600));

        let result = sim.handle_resize(1920, 1080);
        assert!(result.is_some());

        let event = result.unwrap();
        assert_eq!(event.old_width, 800);
        assert_eq!(event.old_height, 600);
        assert_eq!(event.new_width, 1920);
        assert_eq!(event.new_height, 1080);
    }

    #[test]
    fn handle_resize_returns_none_for_same_dimensions() {
        let mut sim = SurfaceStateSimulator::new();
        sim.configure(SurfaceConfiguration::new(1920, 1080));

        let result = sim.handle_resize(1920, 1080);
        assert!(result.is_none());
    }

    #[test]
    fn handle_resize_returns_none_for_minimize_0x0() {
        let mut sim = SurfaceStateSimulator::new();
        sim.configure(SurfaceConfiguration::new(1920, 1080));

        let result = sim.handle_resize(0, 0);
        assert!(result.is_none());
        assert!(sim.is_minimized());
    }

    #[test]
    fn handle_resize_returns_none_for_minimize_1x1() {
        let mut sim = SurfaceStateSimulator::new();
        sim.configure(SurfaceConfiguration::new(1920, 1080));

        let result = sim.handle_resize(1, 1);
        assert!(result.is_none());
        assert!(sim.is_minimized());
    }

    #[test]
    fn handle_resize_returns_event_for_restore() {
        let mut sim = SurfaceStateSimulator::new();
        sim.configure(SurfaceConfiguration::new(1920, 1080));

        // Minimize
        sim.handle_resize(0, 0);
        assert!(sim.is_minimized());

        // Restore
        let result = sim.handle_resize(1920, 1080);
        assert!(result.is_some());
        assert!(!sim.is_minimized());
    }

    #[test]
    fn handle_resize_sets_minimized_on_0_width() {
        let mut sim = SurfaceStateSimulator::new();
        sim.configure(SurfaceConfiguration::new(1920, 1080));

        sim.handle_resize(0, 100);
        assert!(sim.is_minimized());
    }

    #[test]
    fn handle_resize_sets_minimized_on_0_height() {
        let mut sim = SurfaceStateSimulator::new();
        sim.configure(SurfaceConfiguration::new(1920, 1080));

        sim.handle_resize(100, 0);
        assert!(sim.is_minimized());
    }

    #[test]
    fn handle_resize_clears_minimized_on_valid_dimensions() {
        let mut sim = SurfaceStateSimulator::new();
        sim.configure(SurfaceConfiguration::new(1920, 1080));
        sim.set_minimized(true);

        let result = sim.handle_resize(1920, 1080);
        // Because was_minimized is true, it should return an event even for same dims
        assert!(result.is_some());
        assert!(!sim.is_minimized());
    }

    #[test]
    fn handle_resize_updates_config() {
        let mut sim = SurfaceStateSimulator::new();
        sim.configure(SurfaceConfiguration::new(800, 600));

        sim.handle_resize(1920, 1080);

        let (w, h) = sim.dimensions();
        assert_eq!(w, 1920);
        assert_eq!(h, 1080);
    }

    #[test]
    fn handle_resize_initializes_config_if_none() {
        let mut sim = SurfaceStateSimulator::new();

        let result = sim.handle_resize(1920, 1080);
        assert!(result.is_some());

        let (w, h) = sim.dimensions();
        assert_eq!(w, 1920);
        assert_eq!(h, 1080);
    }
}

// ============================================================================
// 12. Additional Edge Cases and Integration Tests
// ============================================================================

mod edge_cases {
    use super::*;

    #[test]
    fn resize_event_asymmetric_minimize_detection() {
        // Test that 0xN is considered minimized
        let event = make_resize_event(1920, 1080, 0, 1080);
        assert!(event.is_minimize());

        // Test that Nx0 is considered minimized
        let event = make_resize_event(1920, 1080, 1920, 0);
        assert!(event.is_minimize());
    }

    #[test]
    fn resize_event_2x2_not_minimized() {
        // 2x2 should be valid, not minimized
        let event = make_resize_event(1920, 1080, 2, 2);
        assert!(!event.is_minimize());
    }

    #[test]
    fn resize_event_restore_to_2x2() {
        let event = make_resize_event(0, 0, 2, 2);
        assert!(event.is_restore());
    }

    #[test]
    fn config_from_capabilities_uses_preferred_values() {
        let caps = make_test_capabilities();
        let config = SurfaceConfiguration::from_capabilities(&caps, 1920, 1080);

        assert_eq!(config.width, 1920);
        assert_eq!(config.height, 1080);
        assert_eq!(config.format, wgpu::TextureFormat::Bgra8UnormSrgb);
        assert_eq!(config.present_mode, wgpu::PresentMode::Mailbox);
    }

    #[test]
    fn config_window_size_constructor() {
        let caps = make_test_capabilities();
        let config = SurfaceConfiguration::from_window_size(1920, 1080, &caps);

        assert_eq!(config.width, 1920);
        assert_eq!(config.height, 1080);
    }

    #[test]
    fn resize_event_extreme_aspect_ratio() {
        // Very wide
        let event = make_resize_event(100, 100, 10000, 1);
        assert!(event.aspect_ratio_changed());
        assert_eq!(event.new_aspect_ratio(), 10000.0);

        // Very tall
        let event = make_resize_event(100, 100, 1, 10000);
        assert!(event.aspect_ratio_changed());
        assert!((event.new_aspect_ratio() - 0.0001).abs() < 0.0001);
    }

    #[test]
    fn surface_config_default_creates_1x1() {
        let config = SurfaceConfiguration::default();
        assert_eq!(config.width, 1);
        assert_eq!(config.height, 1);
    }

    #[test]
    fn resize_event_both_zero_aspect_ratios_equal() {
        let event = make_resize_event(0, 0, 0, 0);
        assert_eq!(event.old_aspect_ratio(), event.new_aspect_ratio());
        assert!(!event.aspect_ratio_changed());
    }
}

// ============================================================================
// 13. Platform Target Tests (for surface context)
// ============================================================================

mod platform_target_context {
    use super::*;

    #[test]
    fn platform_target_supports_resize() {
        // All supported platforms should support resize
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
            assert!(platform.is_supported());
        }
    }

    #[test]
    fn unknown_platform_not_supported() {
        assert!(!PlatformTarget::Unknown.is_supported());
    }
}

// ============================================================================
// 14. View Formats and sRGB Toggle Tests (resize context)
// ============================================================================

mod view_formats_resize {
    use super::*;

    #[test]
    fn config_preserves_view_formats_on_resize() {
        let mut config = SurfaceConfiguration::new(800, 600)
            .with_view_formats(&[wgpu::TextureFormat::Bgra8UnormSrgb]);

        config.resize(1920, 1080);

        assert_eq!(config.width, 1920);
        assert_eq!(config.height, 1080);
        assert!(!config.view_formats.is_empty());
        assert_eq!(config.view_formats[0], wgpu::TextureFormat::Bgra8UnormSrgb);
    }

    #[test]
    fn config_preserves_view_formats_with_dimensions() {
        let config = SurfaceConfiguration::new(800, 600)
            .with_view_formats(&[wgpu::TextureFormat::Bgra8UnormSrgb])
            .with_dimensions(1920, 1080);

        assert_eq!(config.width, 1920);
        assert_eq!(config.height, 1080);
        assert!(!config.view_formats.is_empty());
    }

    #[test]
    fn config_with_srgb_view_format_preserved_on_resize() {
        let mut config = SurfaceConfiguration::new(800, 600)
            .with_format(wgpu::TextureFormat::Bgra8Unorm)
            .with_srgb_view_format();

        assert!(config.has_srgb_view_format());

        config.resize(1920, 1080);

        assert!(config.has_srgb_view_format());
    }
}

// ============================================================================
// 15. Stress Tests
// ============================================================================

mod stress_tests {
    use super::*;

    #[test]
    fn many_resize_events_sequential() {
        for i in 1..=100 {
            let event = make_resize_event(i * 10, i * 10, (i + 1) * 10, (i + 1) * 10);
            assert!(event.grew());
            assert!(event.dimensions_changed());
        }
    }

    #[test]
    fn many_config_resizes() {
        let mut config = SurfaceConfiguration::new(100, 100);
        for i in 1..=100 {
            config.resize(i * 10, i * 10);
            assert_eq!(config.width, i * 10);
            assert_eq!(config.height, i * 10);
        }
    }

    #[test]
    fn alternating_minimize_restore() {
        let minimize = make_resize_event(1920, 1080, 0, 0);
        let restore = make_resize_event(0, 0, 1920, 1080);

        for _ in 0..50 {
            assert!(minimize.is_minimize());
            assert!(restore.is_restore());
        }
    }

    #[test]
    fn resize_all_common_resolutions() {
        let resolutions: Vec<(u32, u32)> = vec![
            (320, 240),    // QVGA
            (640, 480),    // VGA
            (800, 600),    // SVGA
            (1024, 768),   // XGA
            (1280, 720),   // 720p
            (1280, 1024),  // SXGA
            (1366, 768),   // HD
            (1440, 900),   // WXGA+
            (1600, 900),   // HD+
            (1680, 1050),  // WSXGA+
            (1920, 1080),  // 1080p
            (1920, 1200),  // WUXGA
            (2560, 1440),  // 1440p
            (2560, 1600),  // WQXGA
            (3440, 1440),  // UWQHD
            (3840, 2160),  // 4K UHD
            (5120, 2880),  // 5K
            (7680, 4320),  // 8K UHD
        ];

        for i in 0..resolutions.len() - 1 {
            let (old_w, old_h) = resolutions[i];
            let (new_w, new_h) = resolutions[i + 1];

            let event = make_resize_event(old_w, old_h, new_w, new_h);
            assert!(event.dimensions_changed());
            assert!(event.grew());
        }
    }
}
