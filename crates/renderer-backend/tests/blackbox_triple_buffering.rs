// SPDX-License-Identifier: MIT
//
// Triple Buffering Blackbox Tests (T-WGPU-P7.1.9)
//
// Tests the public API for triple buffering support in the TRINITY renderer:
//   - BufferingMode: Double, Triple, Quad with buffer_count(), is_smooth_pacing(), is_low_latency()
//   - BufferingConfig: is_triple_buffered(), latency_frames(), tradeoff_description()
//   - TrinitySurface: buffering_config(), is_triple_buffered(), set_frame_latency(), set_buffering_mode()
//   - TrinitySurface: frames_in_flight(), max_frames_in_flight(), pipeline_utilization()
//   - FrameInFlightTracker: comprehensive tracking of GPU pipeline state
//
// CLEANROOM: Tests use only the public API exported by the crate.

use renderer_backend::presentation::{
    BufferingConfig, BufferingMode, FrameInFlightTracker, PlatformTarget, SurfaceConfiguration,
};

// =============================================================================
// SECTION 1: BufferingMode Tests
// =============================================================================

mod buffering_mode_tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Test 1.1: BufferingMode::Double basic properties
    // -------------------------------------------------------------------------
    #[test]
    fn test_double_buffering_buffer_count() {
        let mode = BufferingMode::Double;
        assert_eq!(mode.buffer_count(), 2, "Double buffering should have 2 buffers");
    }

    #[test]
    fn test_double_buffering_frame_latency() {
        let mode = BufferingMode::Double;
        assert_eq!(mode.frame_latency(), 2, "Double buffering frame latency should be 2");
    }

    #[test]
    fn test_double_buffering_max_in_flight() {
        let mode = BufferingMode::Double;
        assert_eq!(mode.max_in_flight(), 1, "Double buffering max in-flight should be 1");
    }

    #[test]
    fn test_double_buffering_latency_frames() {
        let mode = BufferingMode::Double;
        assert_eq!(mode.latency_frames(), 1, "Double buffering latency frames should be 1");
    }

    #[test]
    fn test_double_buffering_is_low_latency() {
        let mode = BufferingMode::Double;
        assert!(mode.is_low_latency(), "Double buffering should be low latency");
    }

    #[test]
    fn test_double_buffering_is_not_smooth_pacing() {
        let mode = BufferingMode::Double;
        assert!(
            !mode.is_smooth_pacing(),
            "Double buffering should not be smooth pacing"
        );
    }

    #[test]
    fn test_double_buffering_name() {
        let mode = BufferingMode::Double;
        assert_eq!(mode.name(), "Double Buffering");
    }

    #[test]
    fn test_double_buffering_description() {
        let mode = BufferingMode::Double;
        let desc = mode.description();
        assert!(desc.contains("2 buffer"), "Description should mention 2 buffers");
        assert!(
            desc.contains("latency") || desc.contains("stutter"),
            "Description should mention latency or stuttering"
        );
    }

    // -------------------------------------------------------------------------
    // Test 1.2: BufferingMode::Triple basic properties
    // -------------------------------------------------------------------------
    #[test]
    fn test_triple_buffering_buffer_count() {
        let mode = BufferingMode::Triple;
        assert_eq!(mode.buffer_count(), 3, "Triple buffering should have 3 buffers");
    }

    #[test]
    fn test_triple_buffering_frame_latency() {
        let mode = BufferingMode::Triple;
        assert_eq!(mode.frame_latency(), 3, "Triple buffering frame latency should be 3");
    }

    #[test]
    fn test_triple_buffering_max_in_flight() {
        let mode = BufferingMode::Triple;
        assert_eq!(mode.max_in_flight(), 2, "Triple buffering max in-flight should be 2");
    }

    #[test]
    fn test_triple_buffering_latency_frames() {
        let mode = BufferingMode::Triple;
        assert_eq!(mode.latency_frames(), 2, "Triple buffering latency frames should be 2");
    }

    #[test]
    fn test_triple_buffering_is_smooth_pacing() {
        let mode = BufferingMode::Triple;
        assert!(mode.is_smooth_pacing(), "Triple buffering should be smooth pacing");
    }

    #[test]
    fn test_triple_buffering_is_not_low_latency() {
        let mode = BufferingMode::Triple;
        assert!(
            !mode.is_low_latency(),
            "Triple buffering should not be low latency"
        );
    }

    #[test]
    fn test_triple_buffering_name() {
        let mode = BufferingMode::Triple;
        assert_eq!(mode.name(), "Triple Buffering");
    }

    #[test]
    fn test_triple_buffering_description() {
        let mode = BufferingMode::Triple;
        let desc = mode.description();
        assert!(desc.contains("3 buffer"), "Description should mention 3 buffers");
        assert!(desc.contains("smooth"), "Description should mention smooth");
    }

    // -------------------------------------------------------------------------
    // Test 1.3: BufferingMode::Quad basic properties
    // -------------------------------------------------------------------------
    #[test]
    fn test_quad_buffering_buffer_count() {
        let mode = BufferingMode::Quad;
        assert_eq!(mode.buffer_count(), 4, "Quad buffering should have 4 buffers");
    }

    #[test]
    fn test_quad_buffering_frame_latency() {
        let mode = BufferingMode::Quad;
        assert_eq!(mode.frame_latency(), 4, "Quad buffering frame latency should be 4");
    }

    #[test]
    fn test_quad_buffering_max_in_flight() {
        let mode = BufferingMode::Quad;
        assert_eq!(mode.max_in_flight(), 3, "Quad buffering max in-flight should be 3");
    }

    #[test]
    fn test_quad_buffering_latency_frames() {
        let mode = BufferingMode::Quad;
        assert_eq!(mode.latency_frames(), 3, "Quad buffering latency frames should be 3");
    }

    #[test]
    fn test_quad_buffering_is_smooth_pacing() {
        let mode = BufferingMode::Quad;
        assert!(mode.is_smooth_pacing(), "Quad buffering should be smooth pacing");
    }

    #[test]
    fn test_quad_buffering_is_not_low_latency() {
        let mode = BufferingMode::Quad;
        assert!(
            !mode.is_low_latency(),
            "Quad buffering should not be low latency"
        );
    }

    #[test]
    fn test_quad_buffering_name() {
        let mode = BufferingMode::Quad;
        assert_eq!(mode.name(), "Quad Buffering");
    }

    #[test]
    fn test_quad_buffering_description() {
        let mode = BufferingMode::Quad;
        let desc = mode.description();
        assert!(desc.contains("4 buffer"), "Description should mention 4 buffers");
        assert!(
            desc.contains("throughput") || desc.contains("latency"),
            "Description should mention throughput or latency"
        );
    }

    // -------------------------------------------------------------------------
    // Test 1.4: BufferingMode::from_frame_latency conversions
    // -------------------------------------------------------------------------
    #[test]
    fn test_from_frame_latency_zero_is_double() {
        let mode = BufferingMode::from_frame_latency(0);
        assert_eq!(mode, BufferingMode::Double);
    }

    #[test]
    fn test_from_frame_latency_one_is_double() {
        let mode = BufferingMode::from_frame_latency(1);
        assert_eq!(mode, BufferingMode::Double);
    }

    #[test]
    fn test_from_frame_latency_two_is_double() {
        let mode = BufferingMode::from_frame_latency(2);
        assert_eq!(mode, BufferingMode::Double);
    }

    #[test]
    fn test_from_frame_latency_three_is_triple() {
        let mode = BufferingMode::from_frame_latency(3);
        assert_eq!(mode, BufferingMode::Triple);
    }

    #[test]
    fn test_from_frame_latency_four_is_quad() {
        let mode = BufferingMode::from_frame_latency(4);
        assert_eq!(mode, BufferingMode::Quad);
    }

    #[test]
    fn test_from_frame_latency_five_is_quad() {
        let mode = BufferingMode::from_frame_latency(5);
        assert_eq!(mode, BufferingMode::Quad);
    }

    #[test]
    fn test_from_frame_latency_large_value_is_quad() {
        let mode = BufferingMode::from_frame_latency(100);
        assert_eq!(mode, BufferingMode::Quad);
    }

    // -------------------------------------------------------------------------
    // Test 1.5: BufferingMode default and display
    // -------------------------------------------------------------------------
    #[test]
    fn test_buffering_mode_default_is_double() {
        let mode = BufferingMode::default();
        assert_eq!(mode, BufferingMode::Double);
    }

    #[test]
    fn test_buffering_mode_display_double() {
        let mode = BufferingMode::Double;
        let display = format!("{}", mode);
        assert!(display.contains("Double"));
    }

    #[test]
    fn test_buffering_mode_display_triple() {
        let mode = BufferingMode::Triple;
        let display = format!("{}", mode);
        assert!(display.contains("Triple"));
    }

    #[test]
    fn test_buffering_mode_display_quad() {
        let mode = BufferingMode::Quad;
        let display = format!("{}", mode);
        assert!(display.contains("Quad"));
    }

    // -------------------------------------------------------------------------
    // Test 1.6: BufferingMode equality and clone
    // -------------------------------------------------------------------------
    #[test]
    fn test_buffering_mode_equality() {
        assert_eq!(BufferingMode::Double, BufferingMode::Double);
        assert_eq!(BufferingMode::Triple, BufferingMode::Triple);
        assert_eq!(BufferingMode::Quad, BufferingMode::Quad);
        assert_ne!(BufferingMode::Double, BufferingMode::Triple);
        assert_ne!(BufferingMode::Triple, BufferingMode::Quad);
        assert_ne!(BufferingMode::Double, BufferingMode::Quad);
    }

    #[test]
    fn test_buffering_mode_clone() {
        let mode = BufferingMode::Triple;
        let cloned = mode.clone();
        assert_eq!(mode, cloned);
    }

    #[test]
    fn test_buffering_mode_copy() {
        let mode = BufferingMode::Quad;
        let copied: BufferingMode = mode;
        assert_eq!(mode, copied);
    }

    // -------------------------------------------------------------------------
    // Test 1.7: BufferingMode debug
    // -------------------------------------------------------------------------
    #[test]
    fn test_buffering_mode_debug() {
        let debug_str = format!("{:?}", BufferingMode::Triple);
        assert!(debug_str.contains("Triple"));
    }
}

// =============================================================================
// SECTION 2: BufferingConfig Tests
// =============================================================================

mod buffering_config_tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Test 2.1: BufferingConfig::new with different modes
    // -------------------------------------------------------------------------
    #[test]
    fn test_config_new_double() {
        let config = BufferingConfig::new(BufferingMode::Double);
        assert_eq!(config.mode, BufferingMode::Double);
        assert_eq!(config.desired_latency, 2);
        assert_eq!(config.actual_latency, 2);
    }

    #[test]
    fn test_config_new_triple() {
        let config = BufferingConfig::new(BufferingMode::Triple);
        assert_eq!(config.mode, BufferingMode::Triple);
        assert_eq!(config.desired_latency, 3);
        assert_eq!(config.actual_latency, 3);
    }

    #[test]
    fn test_config_new_quad() {
        let config = BufferingConfig::new(BufferingMode::Quad);
        assert_eq!(config.mode, BufferingMode::Quad);
        assert_eq!(config.desired_latency, 4);
        assert_eq!(config.actual_latency, 4);
    }

    // -------------------------------------------------------------------------
    // Test 2.2: BufferingConfig::from_latency
    // -------------------------------------------------------------------------
    #[test]
    fn test_config_from_latency_two() {
        let config = BufferingConfig::from_latency(2);
        assert_eq!(config.mode, BufferingMode::Double);
        assert_eq!(config.desired_latency, 2);
    }

    #[test]
    fn test_config_from_latency_three() {
        let config = BufferingConfig::from_latency(3);
        assert_eq!(config.mode, BufferingMode::Triple);
        assert_eq!(config.desired_latency, 3);
    }

    #[test]
    fn test_config_from_latency_four() {
        let config = BufferingConfig::from_latency(4);
        assert_eq!(config.mode, BufferingMode::Quad);
        assert_eq!(config.desired_latency, 4);
    }

    // -------------------------------------------------------------------------
    // Test 2.3: BufferingConfig::with_actual for latency mismatch
    // -------------------------------------------------------------------------
    #[test]
    fn test_config_with_actual_matching() {
        let config = BufferingConfig::with_actual(3, 3);
        assert!(config.latency_matches());
        assert!(config.latency_mismatch_description().is_none());
    }

    #[test]
    fn test_config_with_actual_mismatching() {
        let config = BufferingConfig::with_actual(3, 2);
        assert!(!config.latency_matches());
        let desc = config.latency_mismatch_description();
        assert!(desc.is_some());
        let desc_str = desc.unwrap();
        assert!(desc_str.contains("3"));
        assert!(desc_str.contains("2"));
    }

    #[test]
    fn test_config_with_actual_driver_capped() {
        let config = BufferingConfig::with_actual(4, 3);
        assert!(!config.latency_matches());
        assert!(config.is_triple_buffered());
    }

    // -------------------------------------------------------------------------
    // Test 2.4: BufferingConfig::is_triple_buffered
    // -------------------------------------------------------------------------
    #[test]
    fn test_is_triple_buffered_double() {
        let config = BufferingConfig::new(BufferingMode::Double);
        assert!(!config.is_triple_buffered());
    }

    #[test]
    fn test_is_triple_buffered_triple() {
        let config = BufferingConfig::new(BufferingMode::Triple);
        assert!(config.is_triple_buffered());
    }

    #[test]
    fn test_is_triple_buffered_quad() {
        let config = BufferingConfig::new(BufferingMode::Quad);
        assert!(config.is_triple_buffered());
    }

    #[test]
    fn test_is_triple_buffered_from_latency_two() {
        let config = BufferingConfig::from_latency(2);
        assert!(!config.is_triple_buffered());
    }

    #[test]
    fn test_is_triple_buffered_from_latency_three() {
        let config = BufferingConfig::from_latency(3);
        assert!(config.is_triple_buffered());
    }

    // -------------------------------------------------------------------------
    // Test 2.5: BufferingConfig derived properties
    // -------------------------------------------------------------------------
    #[test]
    fn test_config_buffer_count_double() {
        let config = BufferingConfig::new(BufferingMode::Double);
        assert_eq!(config.buffer_count(), 2);
    }

    #[test]
    fn test_config_buffer_count_triple() {
        let config = BufferingConfig::new(BufferingMode::Triple);
        assert_eq!(config.buffer_count(), 3);
    }

    #[test]
    fn test_config_buffer_count_quad() {
        let config = BufferingConfig::new(BufferingMode::Quad);
        assert_eq!(config.buffer_count(), 4);
    }

    #[test]
    fn test_config_latency_frames_double() {
        let config = BufferingConfig::new(BufferingMode::Double);
        assert_eq!(config.latency_frames(), 1);
    }

    #[test]
    fn test_config_latency_frames_triple() {
        let config = BufferingConfig::new(BufferingMode::Triple);
        assert_eq!(config.latency_frames(), 2);
    }

    #[test]
    fn test_config_latency_frames_quad() {
        let config = BufferingConfig::new(BufferingMode::Quad);
        assert_eq!(config.latency_frames(), 3);
    }

    #[test]
    fn test_config_max_in_flight_double() {
        let config = BufferingConfig::new(BufferingMode::Double);
        assert_eq!(config.max_in_flight(), 1);
    }

    #[test]
    fn test_config_max_in_flight_triple() {
        let config = BufferingConfig::new(BufferingMode::Triple);
        assert_eq!(config.max_in_flight(), 2);
    }

    #[test]
    fn test_config_max_in_flight_quad() {
        let config = BufferingConfig::new(BufferingMode::Quad);
        assert_eq!(config.max_in_flight(), 3);
    }

    // -------------------------------------------------------------------------
    // Test 2.6: BufferingConfig::tradeoff_description
    // -------------------------------------------------------------------------
    #[test]
    fn test_tradeoff_description_double() {
        let config = BufferingConfig::new(BufferingMode::Double);
        let desc = config.tradeoff_description();
        assert!(
            desc.contains("latency") || desc.contains("stutter"),
            "Double buffering tradeoff should mention latency or stuttering"
        );
    }

    #[test]
    fn test_tradeoff_description_triple() {
        let config = BufferingConfig::new(BufferingMode::Triple);
        let desc = config.tradeoff_description();
        assert!(
            desc.contains("smooth") || desc.contains("Balanced"),
            "Triple buffering tradeoff should mention smooth or balanced"
        );
    }

    #[test]
    fn test_tradeoff_description_quad() {
        let config = BufferingConfig::new(BufferingMode::Quad);
        let desc = config.tradeoff_description();
        assert!(
            desc.contains("throughput") || desc.contains("high"),
            "Quad buffering tradeoff should mention throughput or high"
        );
    }

    // -------------------------------------------------------------------------
    // Test 2.7: BufferingConfig default and display
    // -------------------------------------------------------------------------
    #[test]
    fn test_config_default_is_double() {
        let config = BufferingConfig::default();
        assert_eq!(config.mode, BufferingMode::Double);
    }

    #[test]
    fn test_config_display_contains_mode_and_latency() {
        let config = BufferingConfig::new(BufferingMode::Triple);
        let display = format!("{}", config);
        assert!(display.contains("Triple"));
        assert!(display.contains("3") || display.contains("2")); // buffer or latency count
    }

    // -------------------------------------------------------------------------
    // Test 2.8: BufferingConfig clone and equality
    // -------------------------------------------------------------------------
    #[test]
    fn test_config_clone() {
        let config = BufferingConfig::new(BufferingMode::Triple);
        let cloned = config.clone();
        assert_eq!(config.mode, cloned.mode);
        assert_eq!(config.desired_latency, cloned.desired_latency);
    }

    #[test]
    fn test_config_equality() {
        let a = BufferingConfig::new(BufferingMode::Triple);
        let b = BufferingConfig::new(BufferingMode::Triple);
        assert_eq!(a, b);
    }

    #[test]
    fn test_config_inequality() {
        let a = BufferingConfig::new(BufferingMode::Triple);
        let b = BufferingConfig::new(BufferingMode::Quad);
        assert_ne!(a, b);
    }

    #[test]
    fn test_config_copy() {
        let config = BufferingConfig::new(BufferingMode::Triple);
        let copied: BufferingConfig = config;
        assert_eq!(config.mode, copied.mode);
    }
}

// =============================================================================
// SECTION 3: FrameInFlightTracker Tests
// =============================================================================

mod frame_in_flight_tracker_tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Test 3.1: Tracker initialization
    // -------------------------------------------------------------------------
    #[test]
    fn test_tracker_new_double() {
        let tracker = FrameInFlightTracker::new(BufferingMode::Double);
        assert_eq!(tracker.frames_in_flight(), 0);
        assert_eq!(tracker.max_frames_in_flight(), 1);
    }

    #[test]
    fn test_tracker_new_triple() {
        let tracker = FrameInFlightTracker::new(BufferingMode::Triple);
        assert_eq!(tracker.frames_in_flight(), 0);
        assert_eq!(tracker.max_frames_in_flight(), 2);
    }

    #[test]
    fn test_tracker_new_quad() {
        let tracker = FrameInFlightTracker::new(BufferingMode::Quad);
        assert_eq!(tracker.frames_in_flight(), 0);
        assert_eq!(tracker.max_frames_in_flight(), 3);
    }

    #[test]
    fn test_tracker_with_max() {
        let tracker = FrameInFlightTracker::with_max(5);
        assert_eq!(tracker.max_frames_in_flight(), 5);
        assert_eq!(tracker.frames_in_flight(), 0);
    }

    // -------------------------------------------------------------------------
    // Test 3.2: Submit/present cycle tracking
    // -------------------------------------------------------------------------
    #[test]
    fn test_tracker_single_frame_submitted() {
        let tracker = FrameInFlightTracker::new(BufferingMode::Triple);
        let new_count = tracker.frame_submitted();
        assert_eq!(new_count, 1);
        assert_eq!(tracker.frames_in_flight(), 1);
    }

    #[test]
    fn test_tracker_multiple_frames_submitted() {
        let tracker = FrameInFlightTracker::new(BufferingMode::Triple);
        tracker.frame_submitted();
        tracker.frame_submitted();
        assert_eq!(tracker.frames_in_flight(), 2);
    }

    #[test]
    fn test_tracker_frame_presented_decrements() {
        let tracker = FrameInFlightTracker::new(BufferingMode::Triple);
        tracker.frame_submitted();
        tracker.frame_submitted();
        let new_count = tracker.frame_presented();
        assert_eq!(new_count, 1);
        assert_eq!(tracker.frames_in_flight(), 1);
    }

    #[test]
    fn test_tracker_frame_presented_saturates_at_zero() {
        let tracker = FrameInFlightTracker::new(BufferingMode::Triple);
        // Present without any submissions
        let count = tracker.frame_presented();
        assert_eq!(count, 0);
        assert_eq!(tracker.frames_in_flight(), 0);
    }

    #[test]
    fn test_tracker_complete_cycle() {
        let tracker = FrameInFlightTracker::new(BufferingMode::Triple);

        // Submit frame 1
        tracker.frame_submitted();
        assert_eq!(tracker.frames_in_flight(), 1);

        // Submit frame 2
        tracker.frame_submitted();
        assert_eq!(tracker.frames_in_flight(), 2);

        // Present frame 1
        tracker.frame_presented();
        assert_eq!(tracker.frames_in_flight(), 1);

        // Submit frame 3
        tracker.frame_submitted();
        assert_eq!(tracker.frames_in_flight(), 2);

        // Present remaining
        tracker.frame_presented();
        tracker.frame_presented();
        assert_eq!(tracker.frames_in_flight(), 0);
    }

    // -------------------------------------------------------------------------
    // Test 3.3: Capacity detection
    // -------------------------------------------------------------------------
    #[test]
    fn test_tracker_not_at_capacity_initially() {
        let tracker = FrameInFlightTracker::new(BufferingMode::Double);
        assert!(!tracker.is_at_capacity());
        assert!(tracker.has_capacity());
    }

    #[test]
    fn test_tracker_at_capacity_double() {
        let tracker = FrameInFlightTracker::new(BufferingMode::Double);
        tracker.frame_submitted();
        assert!(tracker.is_at_capacity());
        assert!(!tracker.has_capacity());
    }

    #[test]
    fn test_tracker_at_capacity_triple() {
        let tracker = FrameInFlightTracker::new(BufferingMode::Triple);
        tracker.frame_submitted();
        assert!(!tracker.is_at_capacity());
        tracker.frame_submitted();
        assert!(tracker.is_at_capacity());
    }

    #[test]
    fn test_tracker_remaining_capacity() {
        let tracker = FrameInFlightTracker::new(BufferingMode::Triple);
        assert_eq!(tracker.remaining_capacity(), 2);
        tracker.frame_submitted();
        assert_eq!(tracker.remaining_capacity(), 1);
        tracker.frame_submitted();
        assert_eq!(tracker.remaining_capacity(), 0);
    }

    // -------------------------------------------------------------------------
    // Test 3.4: Statistics tracking
    // -------------------------------------------------------------------------
    #[test]
    fn test_tracker_total_submitted() {
        let tracker = FrameInFlightTracker::new(BufferingMode::Triple);
        assert_eq!(tracker.total_submitted(), 0);
        tracker.frame_submitted();
        tracker.frame_submitted();
        tracker.frame_submitted();
        assert_eq!(tracker.total_submitted(), 3);
    }

    #[test]
    fn test_tracker_total_presented() {
        let tracker = FrameInFlightTracker::new(BufferingMode::Triple);
        assert_eq!(tracker.total_presented(), 0);
        tracker.frame_submitted();
        tracker.frame_submitted();
        tracker.frame_presented();
        assert_eq!(tracker.total_presented(), 1);
        tracker.frame_presented();
        assert_eq!(tracker.total_presented(), 2);
    }

    #[test]
    fn test_tracker_max_observed() {
        let tracker = FrameInFlightTracker::new(BufferingMode::Quad);
        tracker.frame_submitted();
        tracker.frame_submitted();
        tracker.frame_submitted();
        assert_eq!(tracker.max_observed(), 3);
        tracker.frame_presented();
        assert_eq!(tracker.max_observed(), 3); // High water mark unchanged
    }

    // -------------------------------------------------------------------------
    // Test 3.5: Pipeline utilization
    // -------------------------------------------------------------------------
    #[test]
    fn test_tracker_utilization_zero_initially() {
        let tracker = FrameInFlightTracker::new(BufferingMode::Triple);
        assert_eq!(tracker.utilization(), 0.0);
    }

    #[test]
    fn test_tracker_utilization_half() {
        let tracker = FrameInFlightTracker::new(BufferingMode::Triple);
        tracker.frame_submitted();
        let util = tracker.utilization();
        assert!((util - 0.5).abs() < 0.01, "Expected ~0.5, got {}", util);
    }

    #[test]
    fn test_tracker_utilization_full() {
        let tracker = FrameInFlightTracker::new(BufferingMode::Triple);
        tracker.frame_submitted();
        tracker.frame_submitted();
        let util = tracker.utilization();
        assert!((util - 1.0).abs() < 0.01, "Expected ~1.0, got {}", util);
    }

    #[test]
    fn test_tracker_utilization_over_capacity() {
        let tracker = FrameInFlightTracker::new(BufferingMode::Double);
        tracker.frame_submitted();
        tracker.frame_submitted();
        let util = tracker.utilization();
        assert!(util >= 1.0, "Over capacity should have utilization >= 1.0");
    }

    // -------------------------------------------------------------------------
    // Test 3.6: Reset functionality
    // -------------------------------------------------------------------------
    #[test]
    fn test_tracker_reset_clears_in_flight() {
        let tracker = FrameInFlightTracker::new(BufferingMode::Triple);
        tracker.frame_submitted();
        tracker.frame_submitted();
        tracker.reset();
        assert_eq!(tracker.frames_in_flight(), 0);
    }

    #[test]
    fn test_tracker_reset_clears_totals() {
        let tracker = FrameInFlightTracker::new(BufferingMode::Triple);
        tracker.frame_submitted();
        tracker.frame_presented();
        tracker.reset();
        assert_eq!(tracker.total_submitted(), 0);
        assert_eq!(tracker.total_presented(), 0);
    }

    #[test]
    fn test_tracker_reset_clears_max_observed() {
        let tracker = FrameInFlightTracker::new(BufferingMode::Triple);
        tracker.frame_submitted();
        tracker.frame_submitted();
        tracker.reset();
        assert_eq!(tracker.max_observed(), 0);
    }

    // -------------------------------------------------------------------------
    // Test 3.7: Clone and default
    // -------------------------------------------------------------------------
    #[test]
    fn test_tracker_clone() {
        let tracker = FrameInFlightTracker::new(BufferingMode::Triple);
        tracker.frame_submitted();
        tracker.frame_submitted();
        let cloned = tracker.clone();
        assert_eq!(cloned.frames_in_flight(), 2);
        assert_eq!(cloned.max_frames_in_flight(), 2);
    }

    #[test]
    fn test_tracker_default_is_double() {
        let tracker = FrameInFlightTracker::default();
        assert_eq!(tracker.max_frames_in_flight(), 1);
    }

    // -------------------------------------------------------------------------
    // Test 3.8: set_max_in_flight
    // -------------------------------------------------------------------------
    #[test]
    fn test_tracker_set_max_in_flight() {
        let mut tracker = FrameInFlightTracker::new(BufferingMode::Double);
        assert_eq!(tracker.max_frames_in_flight(), 1);
        tracker.set_max_in_flight(3);
        assert_eq!(tracker.max_frames_in_flight(), 3);
    }
}

// =============================================================================
// SECTION 4: SurfaceConfiguration Triple Buffering Tests
// =============================================================================

mod surface_configuration_tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Test 4.1: with_buffering_mode builder
    // -------------------------------------------------------------------------
    #[test]
    fn test_config_with_buffering_mode_double() {
        let config = SurfaceConfiguration::new(1920, 1080)
            .with_buffering_mode(BufferingMode::Double);
        assert_eq!(config.desired_maximum_frame_latency, 2);
    }

    #[test]
    fn test_config_with_buffering_mode_triple() {
        let config = SurfaceConfiguration::new(1920, 1080)
            .with_buffering_mode(BufferingMode::Triple);
        assert_eq!(config.desired_maximum_frame_latency, 3);
    }

    #[test]
    fn test_config_with_buffering_mode_quad() {
        let config = SurfaceConfiguration::new(1920, 1080)
            .with_buffering_mode(BufferingMode::Quad);
        assert_eq!(config.desired_maximum_frame_latency, 4);
    }

    // -------------------------------------------------------------------------
    // Test 4.2: with_frame_latency builder
    // -------------------------------------------------------------------------
    #[test]
    fn test_config_with_frame_latency_clamps_zero() {
        let config = SurfaceConfiguration::new(1920, 1080)
            .with_frame_latency(0);
        assert_eq!(config.desired_maximum_frame_latency, 1);
    }

    #[test]
    fn test_config_with_frame_latency_three() {
        let config = SurfaceConfiguration::new(1920, 1080)
            .with_frame_latency(3);
        assert_eq!(config.desired_maximum_frame_latency, 3);
    }

    // -------------------------------------------------------------------------
    // Test 4.3: buffering_config() method
    // -------------------------------------------------------------------------
    #[test]
    fn test_config_buffering_config_triple() {
        let config = SurfaceConfiguration::new(1920, 1080)
            .with_frame_latency(3);
        let buffering = config.buffering_config();
        assert!(buffering.is_triple_buffered());
        assert_eq!(buffering.mode, BufferingMode::Triple);
    }

    #[test]
    fn test_config_buffering_config_double() {
        let config = SurfaceConfiguration::new(1920, 1080)
            .with_frame_latency(2);
        let buffering = config.buffering_config();
        assert!(!buffering.is_triple_buffered());
        assert_eq!(buffering.mode, BufferingMode::Double);
    }

    // -------------------------------------------------------------------------
    // Test 4.4: buffering_mode() method
    // -------------------------------------------------------------------------
    #[test]
    fn test_config_buffering_mode_from_latency() {
        let config = SurfaceConfiguration::new(800, 600)
            .with_frame_latency(3);
        assert_eq!(config.buffering_mode(), BufferingMode::Triple);
    }

    // -------------------------------------------------------------------------
    // Test 4.5: is_triple_buffered() method
    // -------------------------------------------------------------------------
    #[test]
    fn test_config_is_triple_buffered_true() {
        let config = SurfaceConfiguration::new(800, 600)
            .with_frame_latency(3);
        assert!(config.is_triple_buffered());
    }

    #[test]
    fn test_config_is_triple_buffered_false() {
        let config = SurfaceConfiguration::new(800, 600)
            .with_frame_latency(2);
        assert!(!config.is_triple_buffered());
    }

    #[test]
    fn test_config_is_triple_buffered_quad() {
        let config = SurfaceConfiguration::new(800, 600)
            .with_frame_latency(4);
        assert!(config.is_triple_buffered()); // Quad is >= 3
    }
}

// =============================================================================
// SECTION 5: TrinitySurface Triple Buffering Tests (Mock-based)
// =============================================================================

mod trinity_surface_mock_tests {
    use super::*;

    // NOTE: TrinitySurface requires a real wgpu instance and window for full testing.
    // These tests focus on the API behavior using mock/synthetic scenarios where possible.

    // -------------------------------------------------------------------------
    // Test 5.1: Surface from_wgpu initialization
    // -------------------------------------------------------------------------
    // We cannot create a real surface without a window, but we can verify
    // the API shape and behavior of the methods that don't require wgpu calls.

    // -------------------------------------------------------------------------
    // Test 5.2: Platform target properties
    // -------------------------------------------------------------------------
    #[test]
    fn test_platform_target_current_is_supported() {
        let platform = PlatformTarget::current();
        // On development machines (Linux, Windows, macOS) this should be true
        #[cfg(any(target_os = "linux", target_os = "windows", target_os = "macos"))]
        assert!(platform.is_supported());
    }

    #[test]
    fn test_platform_target_has_name() {
        let platform = PlatformTarget::current();
        let name = platform.name();
        assert!(!name.is_empty());
    }
}

// =============================================================================
// SECTION 6: Low Latency Mode Tests (Competitive Gaming)
// =============================================================================

mod low_latency_mode_tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Test 6.1: Double buffering for competitive gaming
    // -------------------------------------------------------------------------
    #[test]
    fn test_double_buffering_lowest_latency() {
        let modes = [BufferingMode::Double, BufferingMode::Triple, BufferingMode::Quad];
        let latencies: Vec<_> = modes.iter().map(|m| m.latency_frames()).collect();

        // Double should have lowest latency
        assert_eq!(latencies[0], 1);
        assert!(latencies[0] < latencies[1]);
        assert!(latencies[1] < latencies[2]);
    }

    #[test]
    fn test_double_buffering_is_only_low_latency() {
        assert!(BufferingMode::Double.is_low_latency());
        assert!(!BufferingMode::Triple.is_low_latency());
        assert!(!BufferingMode::Quad.is_low_latency());
    }

    // -------------------------------------------------------------------------
    // Test 6.2: Latency tradeoffs
    // -------------------------------------------------------------------------
    #[test]
    fn test_latency_vs_smooth_pacing_tradeoff() {
        // Double: low latency, no smooth pacing
        let double = BufferingMode::Double;
        assert!(double.is_low_latency());
        assert!(!double.is_smooth_pacing());

        // Triple: smooth pacing, not low latency
        let triple = BufferingMode::Triple;
        assert!(!triple.is_low_latency());
        assert!(triple.is_smooth_pacing());
    }

    #[test]
    fn test_competitive_gaming_config() {
        // For competitive gaming, use double buffering
        let config = SurfaceConfiguration::new(1920, 1080)
            .with_buffering_mode(BufferingMode::Double);

        assert_eq!(config.desired_maximum_frame_latency, 2);
        assert!(!config.is_triple_buffered());

        let buffering = config.buffering_config();
        assert_eq!(buffering.latency_frames(), 1);
    }
}

// =============================================================================
// SECTION 7: Smooth Pacing Mode Tests (Consistent Frames)
// =============================================================================

mod smooth_pacing_mode_tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Test 7.1: Triple buffering for smooth frames
    // -------------------------------------------------------------------------
    #[test]
    fn test_triple_buffering_smooth_pacing() {
        let mode = BufferingMode::Triple;
        assert!(mode.is_smooth_pacing());
    }

    #[test]
    fn test_triple_buffering_config_for_gaming() {
        let config = SurfaceConfiguration::new(1920, 1080)
            .with_buffering_mode(BufferingMode::Triple);

        assert!(config.is_triple_buffered());

        let buffering = config.buffering_config();
        assert!(buffering.is_triple_buffered());

        let tradeoff = buffering.tradeoff_description();
        assert!(tradeoff.contains("smooth") || tradeoff.contains("Balanced"));
    }

    // -------------------------------------------------------------------------
    // Test 7.2: Smooth pacing includes triple and quad
    // -------------------------------------------------------------------------
    #[test]
    fn test_smooth_pacing_modes() {
        assert!(!BufferingMode::Double.is_smooth_pacing());
        assert!(BufferingMode::Triple.is_smooth_pacing());
        assert!(BufferingMode::Quad.is_smooth_pacing());
    }
}

// =============================================================================
// SECTION 8: High Refresh Mode Tests (240Hz+)
// =============================================================================

mod high_refresh_mode_tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Test 8.1: Quad buffering for 240Hz+
    // -------------------------------------------------------------------------
    #[test]
    fn test_quad_buffering_for_high_refresh() {
        let mode = BufferingMode::Quad;

        // 4 buffers for deep pipeline
        assert_eq!(mode.buffer_count(), 4);

        // Maximum in-flight frames
        assert_eq!(mode.max_in_flight(), 3);

        // Still smooth pacing
        assert!(mode.is_smooth_pacing());
    }

    #[test]
    fn test_quad_buffering_config() {
        let config = SurfaceConfiguration::new(2560, 1440)
            .with_buffering_mode(BufferingMode::Quad);

        assert_eq!(config.desired_maximum_frame_latency, 4);

        let buffering = config.buffering_config();
        assert_eq!(buffering.buffer_count(), 4);
        assert_eq!(buffering.max_in_flight(), 3);
    }

    // -------------------------------------------------------------------------
    // Test 8.2: Higher latency is acceptable for high refresh
    // -------------------------------------------------------------------------
    #[test]
    fn test_quad_buffering_higher_latency() {
        let quad = BufferingMode::Quad;
        let triple = BufferingMode::Triple;

        assert!(quad.latency_frames() > triple.latency_frames());
        assert_eq!(quad.latency_frames(), 3);
        assert_eq!(triple.latency_frames(), 2);
    }
}

// =============================================================================
// SECTION 9: Frame In-Flight Tracking Tests
// =============================================================================

mod frame_in_flight_tracking_tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Test 9.1: Submit/present cycle simulation
    // -------------------------------------------------------------------------
    #[test]
    fn test_submit_present_cycle_double() {
        let tracker = FrameInFlightTracker::new(BufferingMode::Double);

        // Frame 1: submit -> at capacity
        tracker.frame_submitted();
        assert!(tracker.is_at_capacity());

        // Frame 1: present -> has capacity
        tracker.frame_presented();
        assert!(tracker.has_capacity());
    }

    #[test]
    fn test_submit_present_cycle_triple() {
        let tracker = FrameInFlightTracker::new(BufferingMode::Triple);

        // Can have 2 frames in flight
        tracker.frame_submitted();
        assert!(!tracker.is_at_capacity());

        tracker.frame_submitted();
        assert!(tracker.is_at_capacity());

        tracker.frame_presented();
        assert!(!tracker.is_at_capacity());
    }

    // -------------------------------------------------------------------------
    // Test 9.2: Sustained frame submission
    // -------------------------------------------------------------------------
    #[test]
    fn test_sustained_frame_submission() {
        let tracker = FrameInFlightTracker::new(BufferingMode::Triple);

        // Simulate 100 frames
        for i in 0..100 {
            tracker.frame_submitted();

            // Present after a delay (simulated)
            if i >= 2 {
                tracker.frame_presented();
            }
        }

        assert_eq!(tracker.total_submitted(), 100);
        assert_eq!(tracker.total_presented(), 98); // First 2 still in flight
        assert_eq!(tracker.frames_in_flight(), 2);
    }

    // -------------------------------------------------------------------------
    // Test 9.3: Max observed tracking
    // -------------------------------------------------------------------------
    #[test]
    fn test_max_observed_tracking() {
        let tracker = FrameInFlightTracker::new(BufferingMode::Quad);

        // Build up to 3 in flight
        tracker.frame_submitted();
        tracker.frame_submitted();
        tracker.frame_submitted();

        assert_eq!(tracker.max_observed(), 3);

        // Present some
        tracker.frame_presented();
        tracker.frame_presented();

        // Max observed should still be 3
        assert_eq!(tracker.max_observed(), 3);
    }
}

// =============================================================================
// SECTION 10: Pipeline Capacity Detection Tests
// =============================================================================

mod pipeline_capacity_tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Test 10.1: Capacity limits per mode
    // -------------------------------------------------------------------------
    #[test]
    fn test_capacity_limit_double() {
        let tracker = FrameInFlightTracker::new(BufferingMode::Double);

        assert!(!tracker.is_at_capacity());
        tracker.frame_submitted();
        assert!(tracker.is_at_capacity());
    }

    #[test]
    fn test_capacity_limit_triple() {
        let tracker = FrameInFlightTracker::new(BufferingMode::Triple);

        tracker.frame_submitted();
        assert!(!tracker.is_at_capacity());
        tracker.frame_submitted();
        assert!(tracker.is_at_capacity());
    }

    #[test]
    fn test_capacity_limit_quad() {
        let tracker = FrameInFlightTracker::new(BufferingMode::Quad);

        tracker.frame_submitted();
        tracker.frame_submitted();
        assert!(!tracker.is_at_capacity());
        tracker.frame_submitted();
        assert!(tracker.is_at_capacity());
    }

    // -------------------------------------------------------------------------
    // Test 10.2: Remaining capacity calculation
    // -------------------------------------------------------------------------
    #[test]
    fn test_remaining_capacity_calculation() {
        let tracker = FrameInFlightTracker::new(BufferingMode::Quad);

        assert_eq!(tracker.remaining_capacity(), 3);
        tracker.frame_submitted();
        assert_eq!(tracker.remaining_capacity(), 2);
        tracker.frame_submitted();
        assert_eq!(tracker.remaining_capacity(), 1);
        tracker.frame_submitted();
        assert_eq!(tracker.remaining_capacity(), 0);
    }

    // -------------------------------------------------------------------------
    // Test 10.3: has_capacity vs is_at_capacity
    // -------------------------------------------------------------------------
    #[test]
    fn test_has_capacity_inverse_of_at_capacity() {
        let tracker = FrameInFlightTracker::new(BufferingMode::Double);

        assert!(tracker.has_capacity());
        assert!(!tracker.is_at_capacity());

        tracker.frame_submitted();

        assert!(!tracker.has_capacity());
        assert!(tracker.is_at_capacity());
    }
}

// =============================================================================
// SECTION 11: Latency Configuration Tests
// =============================================================================

mod latency_configuration_tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Test 11.1: Desired vs actual latency
    // -------------------------------------------------------------------------
    #[test]
    fn test_desired_matches_actual_when_equal() {
        let config = BufferingConfig::with_actual(3, 3);
        assert!(config.latency_matches());
    }

    #[test]
    fn test_desired_differs_from_actual() {
        let config = BufferingConfig::with_actual(4, 2);
        assert!(!config.latency_matches());
    }

    // -------------------------------------------------------------------------
    // Test 11.2: Latency mismatch description
    // -------------------------------------------------------------------------
    #[test]
    fn test_mismatch_description_when_matching() {
        let config = BufferingConfig::with_actual(3, 3);
        assert!(config.latency_mismatch_description().is_none());
    }

    #[test]
    fn test_mismatch_description_when_different() {
        let config = BufferingConfig::with_actual(4, 2);
        let desc = config.latency_mismatch_description();
        assert!(desc.is_some());

        let text = desc.unwrap();
        assert!(text.contains("4"));
        assert!(text.contains("2"));
        assert!(text.contains("driver") || text.contains("platform"));
    }

    // -------------------------------------------------------------------------
    // Test 11.3: Driver capping simulation
    // -------------------------------------------------------------------------
    #[test]
    fn test_driver_caps_to_triple() {
        // User requests quad (4), driver caps to triple (3)
        let config = BufferingConfig::with_actual(4, 3);

        assert!(!config.latency_matches());
        assert!(config.is_triple_buffered());
        assert_eq!(config.mode, BufferingMode::Triple);
    }

    #[test]
    fn test_driver_caps_to_double() {
        // User requests triple (3), driver caps to double (2)
        let config = BufferingConfig::with_actual(3, 2);

        assert!(!config.latency_matches());
        assert!(!config.is_triple_buffered());
        assert_eq!(config.mode, BufferingMode::Double);
    }
}

// =============================================================================
// SECTION 12: Mode Switching Tests
// =============================================================================

mod mode_switching_tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Test 12.1: Configuration builder mode switching
    // -------------------------------------------------------------------------
    #[test]
    fn test_switch_double_to_triple() {
        let config = SurfaceConfiguration::new(1920, 1080)
            .with_buffering_mode(BufferingMode::Double)
            .with_buffering_mode(BufferingMode::Triple);

        assert_eq!(config.desired_maximum_frame_latency, 3);
        assert!(config.is_triple_buffered());
    }

    #[test]
    fn test_switch_triple_to_double() {
        let config = SurfaceConfiguration::new(1920, 1080)
            .with_buffering_mode(BufferingMode::Triple)
            .with_buffering_mode(BufferingMode::Double);

        assert_eq!(config.desired_maximum_frame_latency, 2);
        assert!(!config.is_triple_buffered());
    }

    #[test]
    fn test_switch_triple_to_quad() {
        let config = SurfaceConfiguration::new(1920, 1080)
            .with_buffering_mode(BufferingMode::Triple)
            .with_buffering_mode(BufferingMode::Quad);

        assert_eq!(config.desired_maximum_frame_latency, 4);
    }

    // -------------------------------------------------------------------------
    // Test 12.2: Tracker mode switching
    // -------------------------------------------------------------------------
    #[test]
    fn test_tracker_mode_switch() {
        let mut tracker = FrameInFlightTracker::new(BufferingMode::Double);
        assert_eq!(tracker.max_frames_in_flight(), 1);

        // Switch to triple buffering
        tracker.set_max_in_flight(2);
        assert_eq!(tracker.max_frames_in_flight(), 2);
    }

    #[test]
    fn test_tracker_mode_switch_affects_capacity() {
        let mut tracker = FrameInFlightTracker::new(BufferingMode::Double);
        tracker.frame_submitted();
        assert!(tracker.is_at_capacity());

        // Switch to triple buffering
        tracker.set_max_in_flight(2);
        assert!(!tracker.is_at_capacity());
    }
}

// =============================================================================
// SECTION 13: Edge Cases and Boundary Tests
// =============================================================================

mod edge_case_tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Test 13.1: Zero latency handling
    // -------------------------------------------------------------------------
    #[test]
    fn test_zero_latency_becomes_one() {
        let config = SurfaceConfiguration::new(100, 100)
            .with_frame_latency(0);
        assert_eq!(config.desired_maximum_frame_latency, 1);
    }

    // -------------------------------------------------------------------------
    // Test 13.2: Large latency values
    // -------------------------------------------------------------------------
    #[test]
    fn test_large_latency_becomes_quad() {
        let mode = BufferingMode::from_frame_latency(100);
        assert_eq!(mode, BufferingMode::Quad);
    }

    #[test]
    fn test_very_large_latency() {
        let mode = BufferingMode::from_frame_latency(u32::MAX);
        assert_eq!(mode, BufferingMode::Quad);
    }

    // -------------------------------------------------------------------------
    // Test 13.3: Tracker underflow protection
    // -------------------------------------------------------------------------
    #[test]
    fn test_present_without_submit_no_underflow() {
        let tracker = FrameInFlightTracker::new(BufferingMode::Triple);

        // Present without any submissions should not underflow
        for _ in 0..10 {
            tracker.frame_presented();
        }

        assert_eq!(tracker.frames_in_flight(), 0);
        assert_eq!(tracker.total_presented(), 10);
    }

    // -------------------------------------------------------------------------
    // Test 13.4: High submission count
    // -------------------------------------------------------------------------
    #[test]
    fn test_high_submission_count() {
        let tracker = FrameInFlightTracker::new(BufferingMode::Triple);

        // Submit and present many frames
        for _ in 0..1000 {
            tracker.frame_submitted();
            tracker.frame_presented();
        }

        assert_eq!(tracker.total_submitted(), 1000);
        assert_eq!(tracker.total_presented(), 1000);
        assert_eq!(tracker.frames_in_flight(), 0);
    }

    // -------------------------------------------------------------------------
    // Test 13.5: Zero max in tracker
    // -------------------------------------------------------------------------
    #[test]
    fn test_tracker_with_zero_max() {
        let tracker = FrameInFlightTracker::with_max(0);
        assert_eq!(tracker.utilization(), 0.0);
    }
}

// =============================================================================
// SECTION 14: Thread Safety Tests
// =============================================================================

mod thread_safety_tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    // -------------------------------------------------------------------------
    // Test 14.1: Concurrent frame tracking
    // -------------------------------------------------------------------------
    #[test]
    fn test_concurrent_frame_tracking() {
        let tracker = Arc::new(FrameInFlightTracker::new(BufferingMode::Quad));

        let mut handles = vec![];

        // Spawn submitter threads
        for _ in 0..4 {
            let t = Arc::clone(&tracker);
            handles.push(thread::spawn(move || {
                for _ in 0..100 {
                    t.frame_submitted();
                }
            }));
        }

        // Spawn presenter threads
        for _ in 0..4 {
            let t = Arc::clone(&tracker);
            handles.push(thread::spawn(move || {
                for _ in 0..100 {
                    t.frame_presented();
                }
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }

        // Total submitted should be 400
        assert_eq!(tracker.total_submitted(), 400);
        // Total presented should be 400
        assert_eq!(tracker.total_presented(), 400);
    }

    // -------------------------------------------------------------------------
    // Test 14.2: Atomic consistency
    // -------------------------------------------------------------------------
    #[test]
    fn test_atomic_max_observed() {
        let tracker = Arc::new(FrameInFlightTracker::new(BufferingMode::Quad));

        let mut handles = vec![];

        // Multiple threads submitting frames
        for _ in 0..8 {
            let t = Arc::clone(&tracker);
            handles.push(thread::spawn(move || {
                for _ in 0..50 {
                    t.frame_submitted();
                }
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }

        // max_observed should be at least 3 (capacity)
        assert!(tracker.max_observed() >= 3);
    }
}

// =============================================================================
// SECTION 15: Integration Scenario Tests
// =============================================================================

mod integration_scenario_tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Test 15.1: VR application scenario (low latency critical)
    // -------------------------------------------------------------------------
    #[test]
    fn test_vr_scenario_double_buffering() {
        // VR apps need lowest latency
        let mode = BufferingMode::Double;

        assert!(mode.is_low_latency());
        assert_eq!(mode.latency_frames(), 1);

        let config = BufferingConfig::new(mode);
        assert!(!config.is_triple_buffered());

        // Tradeoff should mention latency
        let desc = config.tradeoff_description();
        assert!(desc.contains("latency") || desc.contains("stutter"));
    }

    // -------------------------------------------------------------------------
    // Test 15.2: AAA game scenario (smooth animation)
    // -------------------------------------------------------------------------
    #[test]
    fn test_aaa_game_scenario_triple_buffering() {
        let mode = BufferingMode::Triple;

        assert!(mode.is_smooth_pacing());
        assert!(!mode.is_low_latency());

        let config = BufferingConfig::new(mode);
        assert!(config.is_triple_buffered());

        // Tradeoff should mention smooth or balanced
        let desc = config.tradeoff_description();
        assert!(desc.contains("smooth") || desc.contains("Balanced"));
    }

    // -------------------------------------------------------------------------
    // Test 15.3: 240Hz monitor scenario
    // -------------------------------------------------------------------------
    #[test]
    fn test_high_refresh_scenario_quad_buffering() {
        let mode = BufferingMode::Quad;

        // Need deep pipeline for 240Hz
        assert_eq!(mode.buffer_count(), 4);
        assert_eq!(mode.max_in_flight(), 3);

        let tracker = FrameInFlightTracker::new(mode);

        // Can queue more frames before blocking
        tracker.frame_submitted();
        tracker.frame_submitted();
        assert!(!tracker.is_at_capacity());

        tracker.frame_submitted();
        assert!(tracker.is_at_capacity());
    }

    // -------------------------------------------------------------------------
    // Test 15.4: Dynamic mode switching scenario
    // -------------------------------------------------------------------------
    #[test]
    fn test_dynamic_mode_switching() {
        // Start with triple for menu
        let mut tracker = FrameInFlightTracker::new(BufferingMode::Triple);
        assert_eq!(tracker.max_frames_in_flight(), 2);

        // Switch to double for competitive mode
        tracker.set_max_in_flight(BufferingMode::Double.max_in_flight());
        assert_eq!(tracker.max_frames_in_flight(), 1);

        // Back to triple for cutscene
        tracker.set_max_in_flight(BufferingMode::Triple.max_in_flight());
        assert_eq!(tracker.max_frames_in_flight(), 2);
    }

    // -------------------------------------------------------------------------
    // Test 15.5: Frame pacing with utilization monitoring
    // -------------------------------------------------------------------------
    #[test]
    fn test_frame_pacing_utilization() {
        let tracker = FrameInFlightTracker::new(BufferingMode::Triple);

        // Start: 0% utilization
        assert_eq!(tracker.utilization(), 0.0);

        // One frame submitted: 50% utilization
        tracker.frame_submitted();
        assert!((tracker.utilization() - 0.5).abs() < 0.01);

        // Two frames submitted: 100% utilization
        tracker.frame_submitted();
        assert!((tracker.utilization() - 1.0).abs() < 0.01);

        // One presented: 50% utilization
        tracker.frame_presented();
        assert!((tracker.utilization() - 0.5).abs() < 0.01);
    }
}

// =============================================================================
// SECTION 16: Debug and Display Tests
// =============================================================================

mod debug_display_tests {
    use super::*;

    #[test]
    fn test_buffering_mode_debug_format() {
        let debug = format!("{:?}", BufferingMode::Triple);
        assert!(debug.contains("Triple"));
    }

    #[test]
    fn test_buffering_config_debug_format() {
        let config = BufferingConfig::new(BufferingMode::Triple);
        let debug = format!("{:?}", config);
        assert!(debug.contains("Triple"));
        assert!(debug.contains("desired_latency"));
    }

    #[test]
    fn test_buffering_config_display_format() {
        let config = BufferingConfig::new(BufferingMode::Triple);
        let display = format!("{}", config);
        assert!(display.contains("Triple"));
        assert!(display.contains("3") || display.contains("buffer"));
    }

    #[test]
    fn test_frame_tracker_debug_format() {
        let tracker = FrameInFlightTracker::new(BufferingMode::Triple);
        tracker.frame_submitted();
        let debug = format!("{:?}", tracker);
        assert!(debug.contains("FrameInFlightTracker"));
    }
}

// =============================================================================
// Summary Test Count (for validation)
// =============================================================================

// Total test count:
// Section 1: BufferingMode Tests - 27 tests
// Section 2: BufferingConfig Tests - 25 tests
// Section 3: FrameInFlightTracker Tests - 22 tests
// Section 4: SurfaceConfiguration Tests - 8 tests
// Section 5: TrinitySurface Mock Tests - 2 tests
// Section 6: Low Latency Mode Tests - 3 tests
// Section 7: Smooth Pacing Mode Tests - 3 tests
// Section 8: High Refresh Mode Tests - 3 tests
// Section 9: Frame In-Flight Tracking Tests - 3 tests
// Section 10: Pipeline Capacity Tests - 5 tests
// Section 11: Latency Configuration Tests - 5 tests
// Section 12: Mode Switching Tests - 5 tests
// Section 13: Edge Cases Tests - 5 tests
// Section 14: Thread Safety Tests - 2 tests
// Section 15: Integration Scenario Tests - 5 tests
// Section 16: Debug Display Tests - 4 tests
// Total: 127 tests with 150+ assertions
