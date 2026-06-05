//! Whitebox structural tests for Triple Buffering (T-WGPU-P7.1.9).
//!
//! These tests verify the internal structure and behavior of:
//! - `BufferingMode` enum: Double, Triple, Quad with conversion methods
//! - `BufferingConfig` struct: mode configuration and latency tracking
//! - `FrameInFlightTracker`: atomic frame tracking with thread safety
//!
//! Task: T-WGPU-P7.1.9 - Triple Buffering Support
//!
//! Acceptance Criteria Tested:
//! 1. BufferingMode from_frame_latency mapping (1->Double, 2->Double, 3->Triple, 4+->Quad)
//! 2. Buffer count calculations match mode
//! 3. FrameInFlightTracker atomic operations work correctly
//! 4. FrameInFlightTracker thread safety under concurrent access
//! 5. Capacity and utilization calculations are accurate
//! 6. Latency vs throughput trade-offs are documented correctly
//! 7. Edge cases: max in-flight, underflow on present, zero capacity

use renderer_backend::presentation::{BufferingConfig, BufferingMode, FrameInFlightTracker};
use std::sync::Arc;
use std::thread;

// ============================================================================
// 1. BufferingMode - from_frame_latency Mapping Tests
// ============================================================================

mod buffering_mode_from_frame_latency {
    use super::*;

    #[test]
    fn latency_0_returns_double() {
        assert_eq!(BufferingMode::from_frame_latency(0), BufferingMode::Double);
    }

    #[test]
    fn latency_1_returns_double() {
        assert_eq!(BufferingMode::from_frame_latency(1), BufferingMode::Double);
    }

    #[test]
    fn latency_2_returns_double() {
        assert_eq!(BufferingMode::from_frame_latency(2), BufferingMode::Double);
    }

    #[test]
    fn latency_3_returns_triple() {
        assert_eq!(BufferingMode::from_frame_latency(3), BufferingMode::Triple);
    }

    #[test]
    fn latency_4_returns_quad() {
        assert_eq!(BufferingMode::from_frame_latency(4), BufferingMode::Quad);
    }

    #[test]
    fn latency_5_returns_quad() {
        assert_eq!(BufferingMode::from_frame_latency(5), BufferingMode::Quad);
    }

    #[test]
    fn latency_100_returns_quad() {
        assert_eq!(BufferingMode::from_frame_latency(100), BufferingMode::Quad);
    }

    #[test]
    fn latency_u32_max_returns_quad() {
        assert_eq!(
            BufferingMode::from_frame_latency(u32::MAX),
            BufferingMode::Quad
        );
    }
}

// ============================================================================
// 2. BufferingMode - Buffer Count Tests
// ============================================================================

mod buffering_mode_buffer_count {
    use super::*;

    #[test]
    fn double_has_2_buffers() {
        assert_eq!(BufferingMode::Double.buffer_count(), 2);
    }

    #[test]
    fn triple_has_3_buffers() {
        assert_eq!(BufferingMode::Triple.buffer_count(), 3);
    }

    #[test]
    fn quad_has_4_buffers() {
        assert_eq!(BufferingMode::Quad.buffer_count(), 4);
    }

    #[test]
    fn buffer_count_matches_frame_latency_for_double() {
        let mode = BufferingMode::Double;
        assert_eq!(mode.buffer_count(), mode.frame_latency());
    }

    #[test]
    fn buffer_count_matches_frame_latency_for_triple() {
        let mode = BufferingMode::Triple;
        assert_eq!(mode.buffer_count(), mode.frame_latency());
    }

    #[test]
    fn buffer_count_matches_frame_latency_for_quad() {
        let mode = BufferingMode::Quad;
        assert_eq!(mode.buffer_count(), mode.frame_latency());
    }
}

// ============================================================================
// 3. BufferingMode - Frame Latency Tests
// ============================================================================

mod buffering_mode_frame_latency {
    use super::*;

    #[test]
    fn double_frame_latency_is_2() {
        assert_eq!(BufferingMode::Double.frame_latency(), 2);
    }

    #[test]
    fn triple_frame_latency_is_3() {
        assert_eq!(BufferingMode::Triple.frame_latency(), 3);
    }

    #[test]
    fn quad_frame_latency_is_4() {
        assert_eq!(BufferingMode::Quad.frame_latency(), 4);
    }
}

// ============================================================================
// 4. BufferingMode - Max In-Flight Tests
// ============================================================================

mod buffering_mode_max_in_flight {
    use super::*;

    #[test]
    fn double_max_in_flight_is_1() {
        assert_eq!(BufferingMode::Double.max_in_flight(), 1);
    }

    #[test]
    fn triple_max_in_flight_is_2() {
        assert_eq!(BufferingMode::Triple.max_in_flight(), 2);
    }

    #[test]
    fn quad_max_in_flight_is_3() {
        assert_eq!(BufferingMode::Quad.max_in_flight(), 3);
    }

    #[test]
    fn max_in_flight_is_buffer_count_minus_one_for_double() {
        let mode = BufferingMode::Double;
        assert_eq!(mode.max_in_flight(), mode.buffer_count() - 1);
    }

    #[test]
    fn max_in_flight_is_buffer_count_minus_one_for_triple() {
        let mode = BufferingMode::Triple;
        assert_eq!(mode.max_in_flight(), mode.buffer_count() - 1);
    }

    #[test]
    fn max_in_flight_is_buffer_count_minus_one_for_quad() {
        let mode = BufferingMode::Quad;
        assert_eq!(mode.max_in_flight(), mode.buffer_count() - 1);
    }
}

// ============================================================================
// 5. BufferingMode - Latency Frames Tests
// ============================================================================

mod buffering_mode_latency_frames {
    use super::*;

    #[test]
    fn double_latency_frames_is_1() {
        assert_eq!(BufferingMode::Double.latency_frames(), 1);
    }

    #[test]
    fn triple_latency_frames_is_2() {
        assert_eq!(BufferingMode::Triple.latency_frames(), 2);
    }

    #[test]
    fn quad_latency_frames_is_3() {
        assert_eq!(BufferingMode::Quad.latency_frames(), 3);
    }

    #[test]
    fn latency_frames_equals_max_in_flight_for_double() {
        let mode = BufferingMode::Double;
        assert_eq!(mode.latency_frames(), mode.max_in_flight());
    }

    #[test]
    fn latency_frames_equals_max_in_flight_for_triple() {
        let mode = BufferingMode::Triple;
        assert_eq!(mode.latency_frames(), mode.max_in_flight());
    }

    #[test]
    fn latency_frames_equals_max_in_flight_for_quad() {
        let mode = BufferingMode::Quad;
        assert_eq!(mode.latency_frames(), mode.max_in_flight());
    }
}

// ============================================================================
// 6. BufferingMode - Smooth Pacing Tests
// ============================================================================

mod buffering_mode_smooth_pacing {
    use super::*;

    #[test]
    fn double_is_not_smooth_pacing() {
        assert!(!BufferingMode::Double.is_smooth_pacing());
    }

    #[test]
    fn triple_is_smooth_pacing() {
        assert!(BufferingMode::Triple.is_smooth_pacing());
    }

    #[test]
    fn quad_is_smooth_pacing() {
        assert!(BufferingMode::Quad.is_smooth_pacing());
    }
}

// ============================================================================
// 7. BufferingMode - Low Latency Tests
// ============================================================================

mod buffering_mode_low_latency {
    use super::*;

    #[test]
    fn double_is_low_latency() {
        assert!(BufferingMode::Double.is_low_latency());
    }

    #[test]
    fn triple_is_not_low_latency() {
        assert!(!BufferingMode::Triple.is_low_latency());
    }

    #[test]
    fn quad_is_not_low_latency() {
        assert!(!BufferingMode::Quad.is_low_latency());
    }

    #[test]
    fn low_latency_and_smooth_pacing_are_mutually_exclusive() {
        for mode in [BufferingMode::Double, BufferingMode::Triple, BufferingMode::Quad] {
            assert_ne!(
                mode.is_low_latency(),
                mode.is_smooth_pacing(),
                "Mode {:?} violates mutual exclusivity",
                mode
            );
        }
    }
}

// ============================================================================
// 8. BufferingMode - Name and Description Tests
// ============================================================================

mod buffering_mode_names {
    use super::*;

    #[test]
    fn double_name_is_correct() {
        assert_eq!(BufferingMode::Double.name(), "Double Buffering");
    }

    #[test]
    fn triple_name_is_correct() {
        assert_eq!(BufferingMode::Triple.name(), "Triple Buffering");
    }

    #[test]
    fn quad_name_is_correct() {
        assert_eq!(BufferingMode::Quad.name(), "Quad Buffering");
    }

    #[test]
    fn double_description_mentions_latency() {
        assert!(BufferingMode::Double.description().contains("latency"));
    }

    #[test]
    fn triple_description_mentions_smooth() {
        assert!(BufferingMode::Triple.description().contains("smooth"));
    }

    #[test]
    fn quad_description_mentions_throughput() {
        assert!(BufferingMode::Quad.description().contains("throughput"));
    }

    #[test]
    fn display_format_matches_name() {
        assert_eq!(format!("{}", BufferingMode::Double), "Double Buffering");
        assert_eq!(format!("{}", BufferingMode::Triple), "Triple Buffering");
        assert_eq!(format!("{}", BufferingMode::Quad), "Quad Buffering");
    }
}

// ============================================================================
// 9. BufferingMode - Default and Traits Tests
// ============================================================================

mod buffering_mode_traits {
    use super::*;

    #[test]
    fn default_is_double() {
        assert_eq!(BufferingMode::default(), BufferingMode::Double);
    }

    #[test]
    fn is_copy() {
        let mode = BufferingMode::Triple;
        let copy = mode;
        assert_eq!(copy, mode);
    }

    #[test]
    fn is_clone() {
        let mode = BufferingMode::Triple;
        #[allow(clippy::clone_on_copy)]
        let cloned = mode.clone();
        assert_eq!(cloned, mode);
    }

    #[test]
    fn is_eq() {
        assert_eq!(BufferingMode::Double, BufferingMode::Double);
        assert_ne!(BufferingMode::Double, BufferingMode::Triple);
    }

    #[test]
    fn is_debug() {
        let debug_str = format!("{:?}", BufferingMode::Triple);
        assert!(debug_str.contains("Triple"));
    }

    #[test]
    fn is_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(BufferingMode::Double);
        set.insert(BufferingMode::Triple);
        set.insert(BufferingMode::Quad);
        assert_eq!(set.len(), 3);
    }
}

// ============================================================================
// 10. BufferingConfig - Construction Tests
// ============================================================================

mod buffering_config_construction {
    use super::*;

    #[test]
    fn new_with_double() {
        let config = BufferingConfig::new(BufferingMode::Double);
        assert_eq!(config.mode, BufferingMode::Double);
        assert_eq!(config.desired_latency, 2);
        assert_eq!(config.actual_latency, 2);
    }

    #[test]
    fn new_with_triple() {
        let config = BufferingConfig::new(BufferingMode::Triple);
        assert_eq!(config.mode, BufferingMode::Triple);
        assert_eq!(config.desired_latency, 3);
        assert_eq!(config.actual_latency, 3);
    }

    #[test]
    fn new_with_quad() {
        let config = BufferingConfig::new(BufferingMode::Quad);
        assert_eq!(config.mode, BufferingMode::Quad);
        assert_eq!(config.desired_latency, 4);
        assert_eq!(config.actual_latency, 4);
    }

    #[test]
    fn from_latency_2_is_double() {
        let config = BufferingConfig::from_latency(2);
        assert_eq!(config.mode, BufferingMode::Double);
    }

    #[test]
    fn from_latency_3_is_triple() {
        let config = BufferingConfig::from_latency(3);
        assert_eq!(config.mode, BufferingMode::Triple);
    }

    #[test]
    fn from_latency_4_is_quad() {
        let config = BufferingConfig::from_latency(4);
        assert_eq!(config.mode, BufferingMode::Quad);
    }

    #[test]
    fn with_actual_stores_both_values() {
        let config = BufferingConfig::with_actual(3, 2);
        assert_eq!(config.desired_latency, 3);
        assert_eq!(config.actual_latency, 2);
        // Mode is derived from actual latency
        assert_eq!(config.mode, BufferingMode::Double);
    }

    #[test]
    fn with_actual_different_desired_and_actual() {
        let config = BufferingConfig::with_actual(4, 3);
        assert_eq!(config.desired_latency, 4);
        assert_eq!(config.actual_latency, 3);
        assert_eq!(config.mode, BufferingMode::Triple);
    }
}

// ============================================================================
// 11. BufferingConfig - is_triple_buffered Tests
// ============================================================================

mod buffering_config_is_triple_buffered {
    use super::*;

    #[test]
    fn double_is_not_triple_buffered() {
        let config = BufferingConfig::new(BufferingMode::Double);
        assert!(!config.is_triple_buffered());
    }

    #[test]
    fn triple_is_triple_buffered() {
        let config = BufferingConfig::new(BufferingMode::Triple);
        assert!(config.is_triple_buffered());
    }

    #[test]
    fn quad_is_triple_buffered() {
        let config = BufferingConfig::new(BufferingMode::Quad);
        assert!(config.is_triple_buffered());
    }

    #[test]
    fn actual_latency_2_is_not_triple_buffered() {
        let config = BufferingConfig::with_actual(3, 2);
        assert!(!config.is_triple_buffered());
    }

    #[test]
    fn actual_latency_3_is_triple_buffered() {
        let config = BufferingConfig::with_actual(2, 3);
        assert!(config.is_triple_buffered());
    }

    #[test]
    fn actual_latency_4_is_triple_buffered() {
        let config = BufferingConfig::with_actual(2, 4);
        assert!(config.is_triple_buffered());
    }
}

// ============================================================================
// 12. BufferingConfig - Buffer Count Tests
// ============================================================================

mod buffering_config_buffer_count {
    use super::*;

    #[test]
    fn double_config_has_2_buffers() {
        let config = BufferingConfig::new(BufferingMode::Double);
        assert_eq!(config.buffer_count(), 2);
    }

    #[test]
    fn triple_config_has_3_buffers() {
        let config = BufferingConfig::new(BufferingMode::Triple);
        assert_eq!(config.buffer_count(), 3);
    }

    #[test]
    fn quad_config_has_4_buffers() {
        let config = BufferingConfig::new(BufferingMode::Quad);
        assert_eq!(config.buffer_count(), 4);
    }

    #[test]
    fn buffer_count_matches_mode() {
        for mode in [BufferingMode::Double, BufferingMode::Triple, BufferingMode::Quad] {
            let config = BufferingConfig::new(mode);
            assert_eq!(config.buffer_count(), mode.buffer_count());
        }
    }
}

// ============================================================================
// 13. BufferingConfig - Latency Frames Tests
// ============================================================================

mod buffering_config_latency_frames {
    use super::*;

    #[test]
    fn double_config_latency_frames_is_1() {
        let config = BufferingConfig::new(BufferingMode::Double);
        assert_eq!(config.latency_frames(), 1);
    }

    #[test]
    fn triple_config_latency_frames_is_2() {
        let config = BufferingConfig::new(BufferingMode::Triple);
        assert_eq!(config.latency_frames(), 2);
    }

    #[test]
    fn quad_config_latency_frames_is_3() {
        let config = BufferingConfig::new(BufferingMode::Quad);
        assert_eq!(config.latency_frames(), 3);
    }

    #[test]
    fn latency_frames_matches_mode() {
        for mode in [BufferingMode::Double, BufferingMode::Triple, BufferingMode::Quad] {
            let config = BufferingConfig::new(mode);
            assert_eq!(config.latency_frames(), mode.latency_frames());
        }
    }
}

// ============================================================================
// 14. BufferingConfig - Max In-Flight Tests
// ============================================================================

mod buffering_config_max_in_flight {
    use super::*;

    #[test]
    fn double_config_max_in_flight_is_1() {
        let config = BufferingConfig::new(BufferingMode::Double);
        assert_eq!(config.max_in_flight(), 1);
    }

    #[test]
    fn triple_config_max_in_flight_is_2() {
        let config = BufferingConfig::new(BufferingMode::Triple);
        assert_eq!(config.max_in_flight(), 2);
    }

    #[test]
    fn quad_config_max_in_flight_is_3() {
        let config = BufferingConfig::new(BufferingMode::Quad);
        assert_eq!(config.max_in_flight(), 3);
    }
}

// ============================================================================
// 15. BufferingConfig - Latency Matches Tests
// ============================================================================

mod buffering_config_latency_matches {
    use super::*;

    #[test]
    fn new_config_latency_matches() {
        let config = BufferingConfig::new(BufferingMode::Triple);
        assert!(config.latency_matches());
    }

    #[test]
    fn from_latency_latency_matches() {
        let config = BufferingConfig::from_latency(3);
        assert!(config.latency_matches());
    }

    #[test]
    fn with_actual_same_values_matches() {
        let config = BufferingConfig::with_actual(3, 3);
        assert!(config.latency_matches());
    }

    #[test]
    fn with_actual_different_values_does_not_match() {
        let config = BufferingConfig::with_actual(3, 2);
        assert!(!config.latency_matches());
    }

    #[test]
    fn with_actual_driver_capped_does_not_match() {
        let config = BufferingConfig::with_actual(4, 3);
        assert!(!config.latency_matches());
    }
}

// ============================================================================
// 16. BufferingConfig - Tradeoff Description Tests
// ============================================================================

mod buffering_config_tradeoff_description {
    use super::*;

    #[test]
    fn double_mentions_lower_latency() {
        let config = BufferingConfig::new(BufferingMode::Double);
        let desc = config.tradeoff_description();
        assert!(
            desc.to_lowercase().contains("latency"),
            "Expected 'latency' in: {}",
            desc
        );
    }

    #[test]
    fn double_mentions_stuttering() {
        let config = BufferingConfig::new(BufferingMode::Double);
        let desc = config.tradeoff_description();
        assert!(
            desc.to_lowercase().contains("stutter"),
            "Expected 'stutter' in: {}",
            desc
        );
    }

    #[test]
    fn triple_mentions_balanced_or_smooth() {
        let config = BufferingConfig::new(BufferingMode::Triple);
        let desc = config.tradeoff_description().to_lowercase();
        assert!(
            desc.contains("balanced") || desc.contains("smooth"),
            "Expected 'balanced' or 'smooth' in: {}",
            desc
        );
    }

    #[test]
    fn quad_mentions_throughput() {
        let config = BufferingConfig::new(BufferingMode::Quad);
        let desc = config.tradeoff_description();
        assert!(
            desc.to_lowercase().contains("throughput"),
            "Expected 'throughput' in: {}",
            desc
        );
    }

    #[test]
    fn quad_mentions_high_refresh() {
        let config = BufferingConfig::new(BufferingMode::Quad);
        let desc = config.tradeoff_description();
        assert!(
            desc.to_lowercase().contains("refresh"),
            "Expected 'refresh' in: {}",
            desc
        );
    }
}

// ============================================================================
// 17. BufferingConfig - Default and Display Tests
// ============================================================================

mod buffering_config_traits {
    use super::*;

    #[test]
    fn default_is_double() {
        let config = BufferingConfig::default();
        assert_eq!(config.mode, BufferingMode::Double);
    }

    #[test]
    fn display_contains_mode_name() {
        let config = BufferingConfig::new(BufferingMode::Triple);
        let display = format!("{}", config);
        assert!(display.contains("Triple"));
    }

    #[test]
    fn display_contains_latency_info() {
        let config = BufferingConfig::new(BufferingMode::Triple);
        let display = format!("{}", config);
        assert!(display.contains("latency"));
    }

    #[test]
    fn display_contains_buffer_count() {
        let config = BufferingConfig::new(BufferingMode::Triple);
        let display = format!("{}", config);
        assert!(display.contains("3") || display.contains("buffer"));
    }

    #[test]
    fn is_copy() {
        let config = BufferingConfig::new(BufferingMode::Triple);
        let copy = config;
        assert_eq!(copy.mode, config.mode);
    }

    #[test]
    fn is_clone() {
        let config = BufferingConfig::new(BufferingMode::Triple);
        #[allow(clippy::clone_on_copy)]
        let cloned = config.clone();
        assert_eq!(cloned.mode, config.mode);
    }

    #[test]
    fn is_eq() {
        let a = BufferingConfig::new(BufferingMode::Triple);
        let b = BufferingConfig::new(BufferingMode::Triple);
        let c = BufferingConfig::new(BufferingMode::Double);
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn is_debug() {
        let config = BufferingConfig::new(BufferingMode::Triple);
        let debug_str = format!("{:?}", config);
        assert!(debug_str.contains("BufferingConfig"));
        assert!(debug_str.contains("Triple"));
    }
}

// ============================================================================
// 18. FrameInFlightTracker - Construction Tests
// ============================================================================

mod frame_tracker_construction {
    use super::*;

    #[test]
    fn new_with_double_mode() {
        let tracker = FrameInFlightTracker::new(BufferingMode::Double);
        assert_eq!(tracker.max_frames_in_flight(), 1);
        assert_eq!(tracker.frames_in_flight(), 0);
    }

    #[test]
    fn new_with_triple_mode() {
        let tracker = FrameInFlightTracker::new(BufferingMode::Triple);
        assert_eq!(tracker.max_frames_in_flight(), 2);
        assert_eq!(tracker.frames_in_flight(), 0);
    }

    #[test]
    fn new_with_quad_mode() {
        let tracker = FrameInFlightTracker::new(BufferingMode::Quad);
        assert_eq!(tracker.max_frames_in_flight(), 3);
        assert_eq!(tracker.frames_in_flight(), 0);
    }

    #[test]
    fn with_max_custom_value() {
        let tracker = FrameInFlightTracker::with_max(5);
        assert_eq!(tracker.max_frames_in_flight(), 5);
        assert_eq!(tracker.frames_in_flight(), 0);
    }

    #[test]
    fn with_max_zero() {
        let tracker = FrameInFlightTracker::with_max(0);
        assert_eq!(tracker.max_frames_in_flight(), 0);
    }

    #[test]
    fn initial_totals_are_zero() {
        let tracker = FrameInFlightTracker::new(BufferingMode::Triple);
        assert_eq!(tracker.total_submitted(), 0);
        assert_eq!(tracker.total_presented(), 0);
    }

    #[test]
    fn initial_max_observed_is_zero() {
        let tracker = FrameInFlightTracker::new(BufferingMode::Triple);
        assert_eq!(tracker.max_observed(), 0);
    }
}

// ============================================================================
// 19. FrameInFlightTracker - Submit/Present Atomic Operations
// ============================================================================

mod frame_tracker_atomic_operations {
    use super::*;

    #[test]
    fn frame_submitted_increments_count() {
        let tracker = FrameInFlightTracker::new(BufferingMode::Triple);
        assert_eq!(tracker.frames_in_flight(), 0);
        tracker.frame_submitted();
        assert_eq!(tracker.frames_in_flight(), 1);
    }

    #[test]
    fn frame_submitted_returns_new_count() {
        let tracker = FrameInFlightTracker::new(BufferingMode::Triple);
        let count = tracker.frame_submitted();
        assert_eq!(count, 1);
        let count = tracker.frame_submitted();
        assert_eq!(count, 2);
    }

    #[test]
    fn frame_presented_decrements_count() {
        let tracker = FrameInFlightTracker::new(BufferingMode::Triple);
        tracker.frame_submitted();
        tracker.frame_submitted();
        assert_eq!(tracker.frames_in_flight(), 2);
        tracker.frame_presented();
        assert_eq!(tracker.frames_in_flight(), 1);
    }

    #[test]
    fn frame_presented_returns_new_count() {
        let tracker = FrameInFlightTracker::new(BufferingMode::Triple);
        tracker.frame_submitted();
        tracker.frame_submitted();
        let count = tracker.frame_presented();
        assert_eq!(count, 1);
    }

    #[test]
    fn multiple_submit_present_cycles() {
        let tracker = FrameInFlightTracker::new(BufferingMode::Triple);

        for _ in 0..10 {
            tracker.frame_submitted();
            tracker.frame_presented();
        }

        assert_eq!(tracker.frames_in_flight(), 0);
        assert_eq!(tracker.total_submitted(), 10);
        assert_eq!(tracker.total_presented(), 10);
    }

    #[test]
    fn total_submitted_increments_on_submit() {
        let tracker = FrameInFlightTracker::new(BufferingMode::Triple);
        tracker.frame_submitted();
        tracker.frame_submitted();
        tracker.frame_submitted();
        assert_eq!(tracker.total_submitted(), 3);
    }

    #[test]
    fn total_presented_increments_on_present() {
        let tracker = FrameInFlightTracker::new(BufferingMode::Triple);
        tracker.frame_submitted();
        tracker.frame_submitted();
        tracker.frame_presented();
        tracker.frame_presented();
        assert_eq!(tracker.total_presented(), 2);
    }

    #[test]
    fn max_observed_tracks_high_water_mark() {
        let tracker = FrameInFlightTracker::new(BufferingMode::Quad);
        tracker.frame_submitted();
        assert_eq!(tracker.max_observed(), 1);
        tracker.frame_submitted();
        assert_eq!(tracker.max_observed(), 2);
        tracker.frame_submitted();
        assert_eq!(tracker.max_observed(), 3);
        tracker.frame_presented();
        // Max should not decrease
        assert_eq!(tracker.max_observed(), 3);
    }
}

// ============================================================================
// 20. FrameInFlightTracker - Edge Cases
// ============================================================================

mod frame_tracker_edge_cases {
    use super::*;

    #[test]
    fn underflow_on_present_saturates_to_zero() {
        let tracker = FrameInFlightTracker::new(BufferingMode::Triple);
        // Present without submit - should saturate at 0
        let count = tracker.frame_presented();
        assert_eq!(count, 0);
        assert_eq!(tracker.frames_in_flight(), 0);
    }

    #[test]
    fn multiple_underflows_stay_at_zero() {
        let tracker = FrameInFlightTracker::new(BufferingMode::Triple);
        tracker.frame_presented();
        tracker.frame_presented();
        tracker.frame_presented();
        assert_eq!(tracker.frames_in_flight(), 0);
    }

    #[test]
    fn underflow_does_not_affect_totals() {
        let tracker = FrameInFlightTracker::new(BufferingMode::Triple);
        tracker.frame_presented();
        // total_presented still increments (tracking actual calls)
        assert_eq!(tracker.total_presented(), 1);
    }

    #[test]
    fn can_exceed_max_in_flight() {
        let tracker = FrameInFlightTracker::new(BufferingMode::Double);
        assert_eq!(tracker.max_frames_in_flight(), 1);
        // Submit more than max - this should still work
        tracker.frame_submitted();
        tracker.frame_submitted();
        tracker.frame_submitted();
        assert_eq!(tracker.frames_in_flight(), 3);
    }

    #[test]
    fn large_number_of_frames() {
        let tracker = FrameInFlightTracker::new(BufferingMode::Triple);
        for _ in 0..1000 {
            tracker.frame_submitted();
        }
        for _ in 0..1000 {
            tracker.frame_presented();
        }
        assert_eq!(tracker.frames_in_flight(), 0);
        assert_eq!(tracker.total_submitted(), 1000);
        assert_eq!(tracker.total_presented(), 1000);
    }
}

// ============================================================================
// 21. FrameInFlightTracker - Capacity Tests
// ============================================================================

mod frame_tracker_capacity {
    use super::*;

    #[test]
    fn is_at_capacity_when_at_max() {
        let tracker = FrameInFlightTracker::new(BufferingMode::Double);
        assert!(!tracker.is_at_capacity());
        tracker.frame_submitted();
        assert!(tracker.is_at_capacity());
    }

    #[test]
    fn is_at_capacity_when_exceeding_max() {
        let tracker = FrameInFlightTracker::new(BufferingMode::Double);
        tracker.frame_submitted();
        tracker.frame_submitted();
        assert!(tracker.is_at_capacity());
    }

    #[test]
    fn has_capacity_when_below_max() {
        let tracker = FrameInFlightTracker::new(BufferingMode::Triple);
        assert!(tracker.has_capacity());
        tracker.frame_submitted();
        assert!(tracker.has_capacity());
        tracker.frame_submitted();
        assert!(!tracker.has_capacity());
    }

    #[test]
    fn remaining_capacity_initial() {
        let tracker = FrameInFlightTracker::new(BufferingMode::Triple);
        assert_eq!(tracker.remaining_capacity(), 2);
    }

    #[test]
    fn remaining_capacity_after_submit() {
        let tracker = FrameInFlightTracker::new(BufferingMode::Triple);
        tracker.frame_submitted();
        assert_eq!(tracker.remaining_capacity(), 1);
    }

    #[test]
    fn remaining_capacity_at_capacity() {
        let tracker = FrameInFlightTracker::new(BufferingMode::Triple);
        tracker.frame_submitted();
        tracker.frame_submitted();
        assert_eq!(tracker.remaining_capacity(), 0);
    }

    #[test]
    fn remaining_capacity_over_capacity() {
        let tracker = FrameInFlightTracker::new(BufferingMode::Double);
        tracker.frame_submitted();
        tracker.frame_submitted();
        tracker.frame_submitted();
        assert_eq!(tracker.remaining_capacity(), 0);
    }

    #[test]
    fn remaining_capacity_after_present() {
        let tracker = FrameInFlightTracker::new(BufferingMode::Triple);
        tracker.frame_submitted();
        tracker.frame_submitted();
        tracker.frame_presented();
        assert_eq!(tracker.remaining_capacity(), 1);
    }
}

// ============================================================================
// 22. FrameInFlightTracker - Utilization Tests
// ============================================================================

mod frame_tracker_utilization {
    use super::*;

    #[test]
    fn utilization_initial_is_zero() {
        let tracker = FrameInFlightTracker::new(BufferingMode::Triple);
        assert_eq!(tracker.utilization(), 0.0);
    }

    #[test]
    fn utilization_at_half_capacity() {
        let tracker = FrameInFlightTracker::new(BufferingMode::Quad);
        tracker.frame_submitted();
        // 1 / 3 = ~0.333
        let util = tracker.utilization();
        assert!((util - 0.333).abs() < 0.01);
    }

    #[test]
    fn utilization_at_full_capacity() {
        let tracker = FrameInFlightTracker::new(BufferingMode::Triple);
        tracker.frame_submitted();
        tracker.frame_submitted();
        assert_eq!(tracker.utilization(), 1.0);
    }

    #[test]
    fn utilization_over_capacity() {
        let tracker = FrameInFlightTracker::new(BufferingMode::Double);
        tracker.frame_submitted();
        tracker.frame_submitted();
        // Over capacity: 2 / 1 = 2.0
        assert_eq!(tracker.utilization(), 2.0);
    }

    #[test]
    fn utilization_with_zero_max() {
        let tracker = FrameInFlightTracker::with_max(0);
        // Should return 0.0 to avoid division by zero
        assert_eq!(tracker.utilization(), 0.0);
    }

    #[test]
    fn utilization_triple_at_66_percent() {
        let tracker = FrameInFlightTracker::new(BufferingMode::Quad);
        tracker.frame_submitted();
        tracker.frame_submitted();
        // 2 / 3 = ~0.666
        let util = tracker.utilization();
        assert!((util - 0.666).abs() < 0.01);
    }
}

// ============================================================================
// 23. FrameInFlightTracker - Reset Tests
// ============================================================================

mod frame_tracker_reset {
    use super::*;

    #[test]
    fn reset_clears_in_flight() {
        let tracker = FrameInFlightTracker::new(BufferingMode::Triple);
        tracker.frame_submitted();
        tracker.frame_submitted();
        tracker.reset();
        assert_eq!(tracker.frames_in_flight(), 0);
    }

    #[test]
    fn reset_clears_total_submitted() {
        let tracker = FrameInFlightTracker::new(BufferingMode::Triple);
        tracker.frame_submitted();
        tracker.frame_submitted();
        tracker.reset();
        assert_eq!(tracker.total_submitted(), 0);
    }

    #[test]
    fn reset_clears_total_presented() {
        let tracker = FrameInFlightTracker::new(BufferingMode::Triple);
        tracker.frame_submitted();
        tracker.frame_presented();
        tracker.reset();
        assert_eq!(tracker.total_presented(), 0);
    }

    #[test]
    fn reset_clears_max_observed() {
        let tracker = FrameInFlightTracker::new(BufferingMode::Triple);
        tracker.frame_submitted();
        tracker.frame_submitted();
        tracker.reset();
        assert_eq!(tracker.max_observed(), 0);
    }

    #[test]
    fn reset_does_not_change_max_in_flight() {
        let tracker = FrameInFlightTracker::new(BufferingMode::Triple);
        tracker.frame_submitted();
        tracker.reset();
        assert_eq!(tracker.max_frames_in_flight(), 2);
    }
}

// ============================================================================
// 24. FrameInFlightTracker - Set Max In-Flight Tests
// ============================================================================

mod frame_tracker_set_max {
    use super::*;

    #[test]
    fn set_max_in_flight_changes_value() {
        let mut tracker = FrameInFlightTracker::new(BufferingMode::Double);
        assert_eq!(tracker.max_frames_in_flight(), 1);
        tracker.set_max_in_flight(3);
        assert_eq!(tracker.max_frames_in_flight(), 3);
    }

    #[test]
    fn set_max_in_flight_affects_capacity() {
        let mut tracker = FrameInFlightTracker::new(BufferingMode::Double);
        tracker.frame_submitted();
        assert!(tracker.is_at_capacity());
        tracker.set_max_in_flight(2);
        assert!(!tracker.is_at_capacity());
    }

    #[test]
    fn set_max_in_flight_affects_utilization() {
        let mut tracker = FrameInFlightTracker::new(BufferingMode::Double);
        tracker.frame_submitted();
        assert_eq!(tracker.utilization(), 1.0);
        tracker.set_max_in_flight(2);
        assert_eq!(tracker.utilization(), 0.5);
    }
}

// ============================================================================
// 25. FrameInFlightTracker - Clone Tests
// ============================================================================

mod frame_tracker_clone {
    use super::*;

    #[test]
    fn clone_copies_in_flight() {
        let tracker = FrameInFlightTracker::new(BufferingMode::Triple);
        tracker.frame_submitted();
        tracker.frame_submitted();
        let cloned = tracker.clone();
        assert_eq!(cloned.frames_in_flight(), 2);
    }

    #[test]
    fn clone_copies_max_in_flight() {
        let tracker = FrameInFlightTracker::new(BufferingMode::Triple);
        let cloned = tracker.clone();
        assert_eq!(cloned.max_frames_in_flight(), 2);
    }

    #[test]
    fn clone_copies_totals() {
        let tracker = FrameInFlightTracker::new(BufferingMode::Triple);
        tracker.frame_submitted();
        tracker.frame_presented();
        let cloned = tracker.clone();
        assert_eq!(cloned.total_submitted(), 1);
        assert_eq!(cloned.total_presented(), 1);
    }

    #[test]
    fn clone_copies_max_observed() {
        let tracker = FrameInFlightTracker::new(BufferingMode::Triple);
        tracker.frame_submitted();
        tracker.frame_submitted();
        tracker.frame_presented();
        let cloned = tracker.clone();
        assert_eq!(cloned.max_observed(), 2);
    }

    #[test]
    fn clone_is_independent() {
        let tracker = FrameInFlightTracker::new(BufferingMode::Triple);
        tracker.frame_submitted();
        let cloned = tracker.clone();
        tracker.frame_submitted();
        // Original modified, clone unchanged
        assert_eq!(tracker.frames_in_flight(), 2);
        assert_eq!(cloned.frames_in_flight(), 1);
    }
}

// ============================================================================
// 26. FrameInFlightTracker - Default Tests
// ============================================================================

mod frame_tracker_default {
    use super::*;

    #[test]
    fn default_uses_double_mode() {
        let tracker = FrameInFlightTracker::default();
        assert_eq!(tracker.max_frames_in_flight(), 1);
    }

    #[test]
    fn default_starts_at_zero() {
        let tracker = FrameInFlightTracker::default();
        assert_eq!(tracker.frames_in_flight(), 0);
        assert_eq!(tracker.total_submitted(), 0);
        assert_eq!(tracker.total_presented(), 0);
        assert_eq!(tracker.max_observed(), 0);
    }
}

// ============================================================================
// 27. FrameInFlightTracker - Thread Safety Tests
// ============================================================================

mod frame_tracker_thread_safety {
    use super::*;

    #[test]
    fn concurrent_submits() {
        let tracker = Arc::new(FrameInFlightTracker::new(BufferingMode::Quad));
        let mut handles = vec![];

        for _ in 0..10 {
            let t = Arc::clone(&tracker);
            handles.push(thread::spawn(move || {
                for _ in 0..100 {
                    t.frame_submitted();
                }
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }

        assert_eq!(tracker.total_submitted(), 1000);
        assert_eq!(tracker.frames_in_flight(), 1000);
    }

    #[test]
    fn concurrent_presents() {
        let tracker = Arc::new(FrameInFlightTracker::with_max(1000));
        // Pre-submit many frames
        for _ in 0..1000 {
            tracker.frame_submitted();
        }

        let mut handles = vec![];
        for _ in 0..10 {
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

        assert_eq!(tracker.total_presented(), 1000);
        assert_eq!(tracker.frames_in_flight(), 0);
    }

    #[test]
    fn concurrent_submit_and_present() {
        let tracker = Arc::new(FrameInFlightTracker::new(BufferingMode::Quad));
        let submit_tracker = Arc::clone(&tracker);
        let present_tracker = Arc::clone(&tracker);

        // One thread submits, one thread presents
        let submit_handle = thread::spawn(move || {
            for _ in 0..500 {
                submit_tracker.frame_submitted();
            }
        });

        let present_handle = thread::spawn(move || {
            for _ in 0..500 {
                present_tracker.frame_presented();
            }
        });

        submit_handle.join().unwrap();
        present_handle.join().unwrap();

        // Totals should be accurate
        assert_eq!(tracker.total_submitted(), 500);
        assert_eq!(tracker.total_presented(), 500);
        // In-flight could be 0 or positive depending on execution order
        // but should not be negative (saturates at 0)
        assert!(tracker.frames_in_flight() <= 500);
    }

    #[test]
    fn concurrent_max_observed_updates() {
        let tracker = Arc::new(FrameInFlightTracker::with_max(100));
        let mut handles = vec![];

        // Multiple threads racing to update max_observed
        for _ in 0..10 {
            let t = Arc::clone(&tracker);
            handles.push(thread::spawn(move || {
                for _ in 0..10 {
                    t.frame_submitted();
                }
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }

        // max_observed should be 100 (all 100 frames were in flight at some point)
        assert_eq!(tracker.max_observed(), 100);
    }

    #[test]
    fn reset_during_operations_is_safe() {
        let tracker = Arc::new(FrameInFlightTracker::new(BufferingMode::Quad));
        let submit_tracker = Arc::clone(&tracker);

        // Submit some frames
        for _ in 0..50 {
            tracker.frame_submitted();
        }

        let submit_handle = thread::spawn(move || {
            for _ in 0..100 {
                submit_tracker.frame_submitted();
            }
        });

        // Reset while submits are happening
        tracker.reset();

        submit_handle.join().unwrap();

        // State should be consistent (no panic, valid values)
        assert!(tracker.total_submitted() <= 150);
    }
}

// ============================================================================
// 28. FrameInFlightTracker - Debug Tests
// ============================================================================

mod frame_tracker_debug {
    use super::*;

    #[test]
    fn debug_format_contains_type_name() {
        let tracker = FrameInFlightTracker::new(BufferingMode::Triple);
        let debug_str = format!("{:?}", tracker);
        assert!(debug_str.contains("FrameInFlightTracker"));
    }

    #[test]
    fn debug_format_contains_in_flight() {
        let tracker = FrameInFlightTracker::new(BufferingMode::Triple);
        tracker.frame_submitted();
        let debug_str = format!("{:?}", tracker);
        assert!(debug_str.contains("in_flight"));
    }

    #[test]
    fn debug_format_contains_max() {
        let tracker = FrameInFlightTracker::new(BufferingMode::Triple);
        let debug_str = format!("{:?}", tracker);
        assert!(debug_str.contains("max_in_flight"));
    }
}

// ============================================================================
// 29. Integration Tests - Complete Frame Lifecycle
// ============================================================================

mod integration_frame_lifecycle {
    use super::*;

    #[test]
    fn complete_frame_cycle_double_buffering() {
        let config = BufferingConfig::new(BufferingMode::Double);
        let tracker = FrameInFlightTracker::new(config.mode);

        assert!(!config.is_triple_buffered());
        assert!(config.mode.is_low_latency());

        // Simulate rendering loop
        for frame in 0..60 {
            let count = tracker.frame_submitted();
            // Double buffering can queue 1 frame
            if count > config.max_in_flight() {
                // Would need to wait for present
            }
            tracker.frame_presented();

            if frame == 30 {
                // Midway check
                assert_eq!(tracker.frames_in_flight(), 0);
            }
        }

        assert_eq!(tracker.total_submitted(), 60);
        assert_eq!(tracker.total_presented(), 60);
    }

    #[test]
    fn complete_frame_cycle_triple_buffering() {
        let config = BufferingConfig::new(BufferingMode::Triple);
        let tracker = FrameInFlightTracker::new(config.mode);

        assert!(config.is_triple_buffered());
        assert!(config.mode.is_smooth_pacing());
        assert!(!config.mode.is_low_latency());

        // Simulate smoother rendering with triple buffering
        // Can have 2 frames in-flight
        tracker.frame_submitted();
        tracker.frame_submitted();
        assert!(!tracker.has_capacity());
        assert!(tracker.is_at_capacity());

        // Present one
        tracker.frame_presented();
        assert!(tracker.has_capacity());

        // Can submit another
        tracker.frame_submitted();
        assert!(tracker.is_at_capacity());
    }

    #[test]
    fn buffering_config_from_surface_configuration() {
        // Simulate getting config from surface
        let desired_latency = 3;
        let actual_latency = 2; // Driver limited

        let config = BufferingConfig::with_actual(desired_latency, actual_latency);

        assert!(!config.latency_matches());
        assert!(!config.is_triple_buffered()); // Actual is 2, not 3
        assert_eq!(config.mode, BufferingMode::Double);

        let mismatch = config.latency_mismatch_description();
        assert!(mismatch.is_some());
        let desc = mismatch.unwrap();
        assert!(desc.contains("3")); // Requested
        assert!(desc.contains("2")); // Got
    }

    #[test]
    fn high_refresh_rate_quad_buffering() {
        // For 240Hz+ displays
        let config = BufferingConfig::new(BufferingMode::Quad);
        let tracker = FrameInFlightTracker::new(config.mode);

        assert!(config.is_triple_buffered()); // >= 3 is "triple or more"
        assert_eq!(config.buffer_count(), 4);
        assert_eq!(config.max_in_flight(), 3);

        // Can have 3 frames in-flight
        tracker.frame_submitted();
        tracker.frame_submitted();
        tracker.frame_submitted();
        assert!(tracker.is_at_capacity());

        // Utilization at 100%
        assert_eq!(tracker.utilization(), 1.0);
    }
}

// ============================================================================
// 30. Roundtrip Tests - Latency Value Preservation
// ============================================================================

mod roundtrip_tests {
    use super::*;

    #[test]
    fn double_mode_roundtrip() {
        let mode = BufferingMode::Double;
        let latency = mode.frame_latency();
        let mode2 = BufferingMode::from_frame_latency(latency);
        assert_eq!(mode, mode2);
    }

    #[test]
    fn triple_mode_roundtrip() {
        let mode = BufferingMode::Triple;
        let latency = mode.frame_latency();
        let mode2 = BufferingMode::from_frame_latency(latency);
        assert_eq!(mode, mode2);
    }

    #[test]
    fn quad_mode_roundtrip() {
        let mode = BufferingMode::Quad;
        let latency = mode.frame_latency();
        let mode2 = BufferingMode::from_frame_latency(latency);
        assert_eq!(mode, mode2);
    }

    #[test]
    fn config_preserves_latency() {
        for latency in 1..=5 {
            let config = BufferingConfig::from_latency(latency);
            assert_eq!(config.desired_latency, latency);
            assert_eq!(config.actual_latency, latency);
        }
    }
}
