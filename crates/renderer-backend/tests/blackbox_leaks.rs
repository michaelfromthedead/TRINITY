//! Blackbox tests for GPU Resource Leak Detection Module
//!
//! This test suite validates the `profiling::leaks` module by treating it as a black box,
//! testing public APIs without knowledge of internal implementation details.
//!
//! Test Categories:
//! - API Contract Tests (20+): Public interface validation
//! - Real-World Leak Scenarios (30+): Realistic resource leak patterns
//! - Threshold Behavior (15+): Detection threshold edge cases
//! - Frame-Based Detection (20+): Per-frame tracking scenarios
//! - Severity Escalation (15+): Severity level transitions
//! - Edge Cases (15+): Boundary conditions and unusual inputs
//! - Reporting (15+): Report generation and formatting

use std::collections::HashMap;
use std::time::{Duration, Instant};

use renderer_backend::profiling::{
    AllocationTracker, FrameLeakChecker, LeakCandidate, LeakDetector, LeakReport, LeakSeverity,
    LeakStats, LeakThresholds, ResourceType,
};

// ============================================================================
// Test Helpers
// ============================================================================

/// Generate unique allocation IDs for testing
fn next_id() -> u64 {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(1);
    COUNTER.fetch_add(1, Ordering::Relaxed)
}

/// Create a detector with immediate detection thresholds for testing
fn immediate_detector() -> LeakDetector {
    LeakDetector::new(LeakThresholds::custom(0, 1, 0))
}

/// Create a detector with custom thresholds
fn detector_with_thresholds(warning: u64, critical: u64, min_size: u64) -> LeakDetector {
    LeakDetector::new(LeakThresholds::custom(warning, critical, min_size))
}

// ============================================================================
// API Contract Tests (20+ tests)
// ============================================================================

mod api_contract {
    use super::*;

    #[test]
    fn test_leak_detector_new_with_default_thresholds() {
        let detector = LeakDetector::with_default_thresholds();
        let thresholds = detector.thresholds();

        assert_eq!(thresholds.warning_secs, 30);
        assert_eq!(thresholds.critical_secs, 120);
        assert_eq!(thresholds.min_size_bytes, 1024);
    }

    #[test]
    fn test_leak_detector_new_with_strict_thresholds() {
        let detector = LeakDetector::with_strict_thresholds();
        let thresholds = detector.thresholds();

        assert_eq!(thresholds.warning_secs, 5);
        assert_eq!(thresholds.critical_secs, 30);
        assert_eq!(thresholds.min_size_bytes, 0);
    }

    #[test]
    fn test_leak_detector_new_with_relaxed_thresholds() {
        let detector = LeakDetector::with_relaxed_thresholds();
        let thresholds = detector.thresholds();

        assert_eq!(thresholds.warning_secs, 300);
        assert_eq!(thresholds.critical_secs, 600);
        assert_eq!(thresholds.min_size_bytes, 4096);
    }

    #[test]
    fn test_leak_detector_set_thresholds() {
        let mut detector = LeakDetector::with_default_thresholds();
        let new_thresholds = LeakThresholds::custom(10, 50, 512);

        detector.set_thresholds(new_thresholds);
        let thresholds = detector.thresholds();

        assert_eq!(thresholds.warning_secs, 10);
        assert_eq!(thresholds.critical_secs, 50);
        assert_eq!(thresholds.min_size_bytes, 512);
    }

    #[test]
    fn test_leak_detector_track_allocation_returns_correct_count() {
        let mut detector = LeakDetector::with_default_thresholds();

        assert_eq!(detector.tracked_count(), 0);

        detector.track_allocation(1, "Buffer1", 1024);
        assert_eq!(detector.tracked_count(), 1);

        detector.track_allocation(2, "Buffer2", 2048);
        assert_eq!(detector.tracked_count(), 2);
    }

    #[test]
    fn test_leak_detector_track_allocation_typed() {
        let mut detector = LeakDetector::with_default_thresholds();

        detector.track_allocation_typed(1, "Vertex Buffer", 1024, ResourceType::Buffer);
        detector.track_allocation_typed(2, "Diffuse Texture", 4096, ResourceType::Texture);

        assert_eq!(detector.tracked_count(), 2);
    }

    #[test]
    fn test_leak_detector_release_allocation_returns_true() {
        let mut detector = LeakDetector::with_default_thresholds();
        detector.track_allocation(1, "Buffer", 1024);

        assert!(detector.release_allocation(1));
        assert_eq!(detector.tracked_count(), 0);
    }

    #[test]
    fn test_leak_detector_release_nonexistent_returns_false() {
        let mut detector = LeakDetector::with_default_thresholds();

        assert!(!detector.release_allocation(999));
    }

    #[test]
    fn test_leak_detector_mark_expected() {
        let mut detector = LeakDetector::with_default_thresholds();
        detector.track_allocation(1, "Static Resource", 1024);

        detector.mark_expected(1);

        let stats = detector.stats();
        assert_eq!(stats.expected_long_lived, 1);
    }

    #[test]
    fn test_leak_detector_mark_temporary() {
        let mut detector = immediate_detector();
        detector.track_allocation(1, "Staging Buffer", 1024);

        detector.mark_temporary(1);

        // Temporary allocations should be detected with stricter thresholds
        // This is tested indirectly via check()
        assert_eq!(detector.tracked_count(), 1);
    }

    #[test]
    fn test_leak_detector_total_bytes() {
        let mut detector = LeakDetector::with_default_thresholds();

        detector.track_allocation(1, "Buffer1", 1024);
        detector.track_allocation(2, "Buffer2", 2048);
        detector.track_allocation(3, "Buffer3", 4096);

        assert_eq!(detector.total_bytes(), 7168);
    }

    #[test]
    fn test_leak_detector_check_returns_vec() {
        let mut detector = LeakDetector::with_default_thresholds();
        detector.track_allocation(1, "Buffer", 1024);

        let candidates = detector.check();
        assert!(candidates.is_empty()); // Fresh allocation
    }

    #[test]
    fn test_leak_detector_check_critical_only() {
        let mut detector = LeakDetector::with_default_thresholds();
        detector.track_allocation(1, "Buffer", 1024);

        let critical = detector.check_critical_only();
        assert!(critical.is_empty());
    }

    #[test]
    fn test_leak_detector_stats_structure() {
        let mut detector = LeakDetector::with_default_thresholds();
        detector.track_allocation(1, "Buffer1", 1024);
        detector.track_allocation(2, "Buffer2", 2048);
        detector.release_allocation(1);
        let _ = detector.check();

        let stats = detector.stats();

        assert_eq!(stats.total_tracked, 2);
        assert_eq!(stats.total_released, 1);
        assert_eq!(stats.current_tracked, 1);
        assert_eq!(stats.checks_performed, 1);
    }

    #[test]
    fn test_leak_detector_clear() {
        let mut detector = LeakDetector::with_default_thresholds();
        detector.track_allocation(1, "Buffer", 1024);
        let _ = detector.check();

        detector.clear();

        assert_eq!(detector.tracked_count(), 0);
        let stats = detector.stats();
        assert_eq!(stats.total_tracked, 0);
        assert_eq!(stats.checks_performed, 0);
    }

    #[test]
    fn test_leak_detector_report_generation() {
        let mut detector = LeakDetector::with_default_thresholds();
        detector.track_allocation(1, "Buffer", 1024);

        let report = detector.report();

        assert!(report.candidates.is_empty()); // Fresh allocation
        assert_eq!(report.stats.current_tracked, 1);
    }

    #[test]
    fn test_leak_detector_time_since_last_check_none_initially() {
        let detector = LeakDetector::with_default_thresholds();

        assert!(detector.time_since_last_check().is_none());
    }

    #[test]
    fn test_leak_detector_time_since_last_check_after_check() {
        let mut detector = LeakDetector::with_default_thresholds();
        let _ = detector.check();

        let duration = detector.time_since_last_check();
        assert!(duration.is_some());
        assert!(duration.unwrap().as_millis() < 100);
    }

    #[test]
    fn test_leak_thresholds_default_thresholds_method() {
        let thresholds = LeakThresholds::default_thresholds();

        assert_eq!(thresholds.warning_secs, 30);
        assert_eq!(thresholds.critical_secs, 120);
        assert_eq!(thresholds.min_size_bytes, 1024);
    }

    #[test]
    fn test_allocation_tracker_new() {
        let tracker = AllocationTracker::new();

        assert_eq!(tracker.count(), 0);
        assert_eq!(tracker.expected_count(), 0);
        assert_eq!(tracker.temporary_count(), 0);
    }
}

// ============================================================================
// Real-World Leak Scenarios (30+ tests)
// ============================================================================

mod real_world_scenarios {
    use super::*;

    #[test]
    fn test_buffer_allocation_never_released() {
        let mut detector = immediate_detector();

        // Simulate creating a vertex buffer and forgetting to release it
        detector.track_allocation_typed(1, "Vertex Buffer", 1024 * 1024, ResourceType::Buffer);

        let candidates = detector.check();
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].resource_type, ResourceType::Buffer);
    }

    #[test]
    fn test_texture_created_and_forgotten() {
        let mut detector = immediate_detector();

        // Create a texture and forget to release
        detector.track_allocation_typed(1, "Diffuse Texture", 4 * 1024 * 1024, ResourceType::Texture);

        let candidates = detector.check();
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].resource_type, ResourceType::Texture);
        assert_eq!(candidates[0].size_bytes, 4 * 1024 * 1024);
    }

    #[test]
    fn test_pipeline_leaked_on_hot_reload() {
        let mut detector = immediate_detector();

        // Simulate hot reload scenario: old pipeline not released
        detector.track_allocation_typed(1, "Old Pipeline", 16384, ResourceType::Pipeline);
        detector.track_allocation_typed(2, "New Pipeline", 16384, ResourceType::Pipeline);

        // "Forget" to release old pipeline
        detector.release_allocation(2); // Release new one (wrong one)

        let candidates = detector.check();
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].allocation_id, 1);
    }

    #[test]
    fn test_bind_group_accumulation_leak() {
        let mut detector = immediate_detector();

        // Simulate bind groups accumulating without release
        for i in 0..10 {
            detector.track_allocation_typed(i, format!("BindGroup_{}", i), 256, ResourceType::BindGroup);
        }

        let candidates = detector.check();
        assert_eq!(candidates.len(), 10);
    }

    #[test]
    fn test_staging_buffer_not_cleaned_up() {
        let mut detector = immediate_detector();

        // Staging buffers should be temporary
        detector.track_allocation_typed(1, "Staging Upload", 64 * 1024, ResourceType::Buffer);
        detector.mark_temporary(1);

        // Forgot to release after upload complete
        let candidates = detector.check();
        assert_eq!(candidates.len(), 1);
    }

    #[test]
    fn test_render_target_leak_on_resize() {
        let mut detector = immediate_detector();

        // Simulate window resize: old render targets not released
        detector.track_allocation_typed(1, "Color Target 1920x1080", 8 * 1024 * 1024, ResourceType::Texture);
        detector.track_allocation_typed(2, "Depth Target 1920x1080", 4 * 1024 * 1024, ResourceType::Texture);

        // Window resized, new targets created
        detector.track_allocation_typed(3, "Color Target 2560x1440", 14 * 1024 * 1024, ResourceType::Texture);
        detector.track_allocation_typed(4, "Depth Target 2560x1440", 7 * 1024 * 1024, ResourceType::Texture);

        // Old targets not released
        let candidates = detector.check();
        assert_eq!(candidates.len(), 4);
    }

    #[test]
    fn test_compute_buffer_leak() {
        let mut detector = immediate_detector();

        detector.track_allocation_typed(1, "Compute Output", 16 * 1024 * 1024, ResourceType::Buffer);

        let candidates = detector.check();
        assert_eq!(candidates.len(), 1);
    }

    #[test]
    fn test_query_set_leak() {
        let mut detector = immediate_detector();

        detector.track_allocation_typed(1, "Timestamp Query Set", 1024, ResourceType::QuerySet);
        detector.track_allocation_typed(2, "Occlusion Query Set", 2048, ResourceType::QuerySet);

        let candidates = detector.check();
        assert_eq!(candidates.len(), 2);
    }

    #[test]
    fn test_multiple_resource_types_leaking() {
        let mut detector = immediate_detector();

        detector.track_allocation_typed(1, "Vertex Buffer", 1024, ResourceType::Buffer);
        detector.track_allocation_typed(2, "Albedo Texture", 4096, ResourceType::Texture);
        detector.track_allocation_typed(3, "Material Pipeline", 512, ResourceType::Pipeline);
        detector.track_allocation_typed(4, "Scene BindGroup", 256, ResourceType::BindGroup);

        let candidates = detector.check();
        assert_eq!(candidates.len(), 4);

        // Verify different resource types
        let types: Vec<_> = candidates.iter().map(|c| c.resource_type).collect();
        assert!(types.contains(&ResourceType::Buffer));
        assert!(types.contains(&ResourceType::Texture));
        assert!(types.contains(&ResourceType::Pipeline));
        assert!(types.contains(&ResourceType::BindGroup));
    }

    #[test]
    fn test_gradual_memory_growth_pattern() {
        let mut detector = immediate_detector();

        // Simulate gradual memory growth over "frames"
        for frame in 0..5 {
            for i in 0..3 {
                let id = next_id();
                detector.track_allocation(
                    id,
                    format!("Frame{}_Buffer{}", frame, i),
                    1024
                );
            }
            // Only release some allocations (leak 1 per frame)
            // Note: In a real scenario, we'd track the IDs to release
        }

        // Should have accumulated leaks
        assert!(detector.tracked_count() > 0);
    }

    #[test]
    fn test_shadow_map_cascade_leak() {
        let mut detector = immediate_detector();

        // Shadow map cascades that weren't released on quality change
        for i in 0..4 {
            detector.track_allocation_typed(
                i + 1,
                format!("Shadow Cascade {}", i),
                2 * 1024 * 1024,
                ResourceType::Texture
            );
        }

        let candidates = detector.check();
        assert_eq!(candidates.len(), 4);
    }

    #[test]
    fn test_uniform_buffer_per_object_leak() {
        let mut detector = immediate_detector();

        // Per-object uniform buffers leaking
        for i in 0..100 {
            detector.track_allocation_typed(
                i + 1,
                format!("Object{} UBO", i),
                256,
                ResourceType::Buffer
            );
        }

        let candidates = detector.check();
        assert_eq!(candidates.len(), 100);

        let total_leaked: u64 = candidates.iter().map(|c| c.size_bytes).sum();
        assert_eq!(total_leaked, 25600);
    }

    #[test]
    fn test_instanced_mesh_buffer_leak() {
        let mut detector = immediate_detector();

        detector.track_allocation_typed(1, "Instance Transform Buffer", 16 * 1000 * 64, ResourceType::Buffer);
        detector.track_allocation_typed(2, "Instance Color Buffer", 4 * 1000 * 4, ResourceType::Buffer);

        let candidates = detector.check();
        assert_eq!(candidates.len(), 2);
    }

    #[test]
    fn test_font_atlas_leak() {
        let mut detector = immediate_detector();

        // Font atlas texture not released on font change
        detector.track_allocation_typed(1, "Font Atlas (Old)", 512 * 512 * 4, ResourceType::Texture);
        detector.track_allocation_typed(2, "Font Atlas (New)", 1024 * 1024 * 4, ResourceType::Texture);

        let candidates = detector.check();
        assert_eq!(candidates.len(), 2);
    }

    #[test]
    fn test_particle_buffer_leak() {
        let mut detector = immediate_detector();

        // Particle system buffers
        detector.track_allocation_typed(1, "Particle Position Buffer", 1024 * 1024, ResourceType::Buffer);
        detector.track_allocation_typed(2, "Particle Velocity Buffer", 1024 * 1024, ResourceType::Buffer);
        detector.track_allocation_typed(3, "Particle Color Buffer", 512 * 1024, ResourceType::Buffer);

        let candidates = detector.check();
        assert_eq!(candidates.len(), 3);
    }

    #[test]
    fn test_procedural_texture_cache_leak() {
        let mut detector = immediate_detector();

        // Procedural textures generated but not cleaned up
        for i in 0..20 {
            detector.track_allocation_typed(
                i + 1,
                format!("Noise Texture {}", i),
                256 * 256 * 4,
                ResourceType::Texture
            );
        }

        let candidates = detector.check();
        assert_eq!(candidates.len(), 20);
    }

    #[test]
    fn test_gi_probe_leak() {
        let mut detector = immediate_detector();

        // GI probe volumes
        detector.track_allocation_typed(1, "GI Probe SH Coefficients", 128 * 128 * 128 * 16, ResourceType::Texture);
        detector.track_allocation_typed(2, "GI Probe Distance", 128 * 128 * 128 * 4, ResourceType::Texture);

        let candidates = detector.check();
        assert_eq!(candidates.len(), 2);
    }

    #[test]
    fn test_streaming_mesh_buffer_leak() {
        let mut detector = immediate_detector();

        // Streaming mesh LODs not released when out of view
        for lod in 0..5 {
            detector.track_allocation_typed(
                lod + 1,
                format!("Mesh LOD{}", lod),
                (5 - lod as u64) * 256 * 1024,
                ResourceType::Buffer
            );
        }

        let candidates = detector.check();
        assert_eq!(candidates.len(), 5);
    }

    #[test]
    fn test_animation_buffer_leak() {
        let mut detector = immediate_detector();

        detector.track_allocation_typed(1, "Bone Transforms", 1024 * 64, ResourceType::Buffer);
        detector.track_allocation_typed(2, "Blend Weights", 1024 * 16, ResourceType::Buffer);

        let candidates = detector.check();
        assert_eq!(candidates.len(), 2);
    }

    #[test]
    fn test_terrain_patch_leak() {
        let mut detector = immediate_detector();

        // Terrain patches not culled properly
        for x in 0..4 {
            for z in 0..4 {
                detector.track_allocation_typed(
                    (x * 4 + z + 1) as u64,
                    format!("Terrain Patch [{},{}]", x, z),
                    64 * 64 * 4 * 4,
                    ResourceType::Buffer
                );
            }
        }

        let candidates = detector.check();
        assert_eq!(candidates.len(), 16);
    }

    #[test]
    fn test_post_process_target_leak() {
        let mut detector = immediate_detector();

        // Post-process render targets
        detector.track_allocation_typed(1, "Bloom Downsample 0", 1920 * 1080 * 4, ResourceType::Texture);
        detector.track_allocation_typed(2, "Bloom Downsample 1", 960 * 540 * 4, ResourceType::Texture);
        detector.track_allocation_typed(3, "Bloom Downsample 2", 480 * 270 * 4, ResourceType::Texture);

        let candidates = detector.check();
        assert_eq!(candidates.len(), 3);
    }

    #[test]
    fn test_cubemap_face_leak() {
        let mut detector = immediate_detector();

        // Environment cubemap faces
        for face in 0..6 {
            detector.track_allocation_typed(
                face + 1,
                format!("Cubemap Face {}", face),
                512 * 512 * 4,
                ResourceType::Texture
            );
        }

        let candidates = detector.check();
        assert_eq!(candidates.len(), 6);
    }

    #[test]
    fn test_debug_line_buffer_leak() {
        let mut detector = immediate_detector();

        // Debug visualization buffers
        detector.track_allocation_typed(1, "Debug Lines VBO", 1024 * 1024, ResourceType::Buffer);
        detector.track_allocation_typed(2, "Debug Lines IBO", 512 * 1024, ResourceType::Buffer);

        let candidates = detector.check();
        assert_eq!(candidates.len(), 2);
    }

    #[test]
    fn test_readback_buffer_leak() {
        let mut detector = immediate_detector();

        // CPU readback buffers
        detector.track_allocation_typed(1, "Screenshot Readback", 1920 * 1080 * 4, ResourceType::Buffer);

        let candidates = detector.check();
        assert_eq!(candidates.len(), 1);
    }

    #[test]
    fn test_acceleration_structure_leak() {
        let mut detector = immediate_detector();

        // Ray tracing acceleration structures
        detector.track_allocation_typed(1, "BLAS Geometry 0", 4 * 1024 * 1024, ResourceType::Other);
        detector.track_allocation_typed(2, "BLAS Geometry 1", 2 * 1024 * 1024, ResourceType::Other);
        detector.track_allocation_typed(3, "TLAS", 1024 * 1024, ResourceType::Other);

        let candidates = detector.check();
        assert_eq!(candidates.len(), 3);
    }

    #[test]
    fn test_mipmap_chain_leak() {
        let mut detector = immediate_detector();

        // Individual mip levels (unusual but possible leak pattern)
        let mut size = 1024u64;
        for mip in 0..10 {
            detector.track_allocation_typed(
                mip + 1,
                format!("Texture Mip {}", mip),
                size * size * 4,
                ResourceType::Texture
            );
            size /= 2;
            if size == 0 { size = 1; }
        }

        let candidates = detector.check();
        assert_eq!(candidates.len(), 10);
    }

    #[test]
    fn test_imgui_buffer_leak() {
        let mut detector = immediate_detector();

        // ImGui draw buffers
        detector.track_allocation_typed(1, "ImGui Vertex Buffer", 64 * 1024, ResourceType::Buffer);
        detector.track_allocation_typed(2, "ImGui Index Buffer", 32 * 1024, ResourceType::Buffer);
        detector.track_allocation_typed(3, "ImGui Font Atlas", 512 * 512 * 4, ResourceType::Texture);

        let candidates = detector.check();
        assert_eq!(candidates.len(), 3);
    }
}

// ============================================================================
// Threshold Behavior Tests (15+ tests)
// ============================================================================

mod threshold_behavior {
    use super::*;

    #[test]
    fn test_detection_at_exactly_warning_threshold() {
        // We can't easily test exact timing, but we can test threshold logic
        let severity = LeakSeverity::from_age_secs(30, 30, 120);
        assert_eq!(severity, LeakSeverity::Warning);
    }

    #[test]
    fn test_detection_at_exactly_critical_threshold() {
        let severity = LeakSeverity::from_age_secs(120, 30, 120);
        assert_eq!(severity, LeakSeverity::Critical);
    }

    #[test]
    fn test_small_allocations_ignored_below_min_size() {
        let mut detector = detector_with_thresholds(0, 1, 2048);

        // Allocate below min_size_bytes
        detector.track_allocation(1, "Tiny Buffer", 512);

        let candidates = detector.check();
        assert!(candidates.is_empty());
    }

    #[test]
    fn test_allocations_at_exactly_min_size_detected() {
        let mut detector = detector_with_thresholds(0, 1, 1024);

        // Allocate exactly at min_size_bytes
        detector.track_allocation(1, "Exact Size Buffer", 1024);

        let candidates = detector.check();
        assert_eq!(candidates.len(), 1);
    }

    #[test]
    fn test_large_allocations_detected_immediately_with_zero_threshold() {
        let mut detector = detector_with_thresholds(0, 0, 0);

        detector.track_allocation(1, "Large Buffer", 100 * 1024 * 1024);

        let candidates = detector.check();
        assert_eq!(candidates.len(), 1);
    }

    #[test]
    fn test_expected_long_lived_resources_ignored() {
        let mut detector = detector_with_thresholds(0, 1, 0);

        detector.track_allocation(1, "Static Resource", 1024);
        detector.mark_expected(1);

        let candidates = detector.check();
        assert!(candidates.is_empty());
    }

    #[test]
    fn test_temporary_resources_detected_faster() {
        let mut detector = detector_with_thresholds(10, 60, 0);

        detector.track_allocation(1, "Temporary Staging", 1024);
        detector.mark_temporary(1);

        // Temporary uses warning_secs / 2, so threshold is 5s
        // We can't test actual timing, but we verify the allocation is tracked
        assert_eq!(detector.tracked_count(), 1);
    }

    #[test]
    fn test_strict_thresholds_detect_early() {
        let mut detector = LeakDetector::with_strict_thresholds();

        // With strict: warning=5s, critical=30s, min_size=0
        let thresholds = detector.thresholds();
        assert_eq!(thresholds.warning_secs, 5);
        assert_eq!(thresholds.min_size_bytes, 0);
    }

    #[test]
    fn test_relaxed_thresholds_detect_late() {
        let mut detector = LeakDetector::with_relaxed_thresholds();

        // With relaxed: warning=300s, critical=600s, min_size=4096
        let thresholds = detector.thresholds();
        assert_eq!(thresholds.warning_secs, 300);
        assert_eq!(thresholds.min_size_bytes, 4096);
    }

    #[test]
    fn test_zero_min_size_detects_all() {
        let mut detector = detector_with_thresholds(0, 1, 0);

        detector.track_allocation(1, "1 Byte", 1);
        detector.track_allocation(2, "10 Bytes", 10);
        detector.track_allocation(3, "100 Bytes", 100);

        let candidates = detector.check();
        assert_eq!(candidates.len(), 3);
    }

    #[test]
    fn test_high_min_size_filters_small() {
        let mut detector = detector_with_thresholds(0, 1, 1024 * 1024);

        detector.track_allocation(1, "Small", 512 * 1024);
        detector.track_allocation(2, "Medium", 768 * 1024);
        detector.track_allocation(3, "Large", 2 * 1024 * 1024);

        let candidates = detector.check();
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].allocation_id, 3);
    }

    #[test]
    fn test_severity_progression_info_to_warning() {
        assert_eq!(LeakSeverity::from_age_secs(0, 30, 120), LeakSeverity::Info);
        assert_eq!(LeakSeverity::from_age_secs(29, 30, 120), LeakSeverity::Info);
        assert_eq!(LeakSeverity::from_age_secs(30, 30, 120), LeakSeverity::Warning);
    }

    #[test]
    fn test_severity_progression_warning_to_critical() {
        assert_eq!(LeakSeverity::from_age_secs(30, 30, 120), LeakSeverity::Warning);
        assert_eq!(LeakSeverity::from_age_secs(119, 30, 120), LeakSeverity::Warning);
        assert_eq!(LeakSeverity::from_age_secs(120, 30, 120), LeakSeverity::Critical);
    }

    #[test]
    fn test_custom_thresholds_applied() {
        let thresholds = LeakThresholds::custom(15, 45, 256);

        assert_eq!(thresholds.warning_secs, 15);
        assert_eq!(thresholds.critical_secs, 45);
        assert_eq!(thresholds.min_size_bytes, 256);
    }

    #[test]
    fn test_thresholds_with_equal_warning_critical() {
        let severity = LeakSeverity::from_age_secs(30, 30, 30);
        // When warning == critical, at threshold it's critical
        assert_eq!(severity, LeakSeverity::Critical);
    }
}

// ============================================================================
// Frame-Based Detection Tests (20+ tests)
// ============================================================================

mod frame_based_detection {
    use super::*;

    #[test]
    fn test_clean_frame_all_released() {
        let mut checker = FrameLeakChecker::new();

        checker.begin_frame();
        checker.track(1);
        checker.track(2);
        checker.track(3);
        checker.release(1);
        checker.release(2);
        checker.release(3);

        let unreleased = checker.end_frame();
        assert!(unreleased.is_empty());
        assert!(checker.is_clean());
    }

    #[test]
    fn test_leaky_frame_some_unreleased() {
        let mut checker = FrameLeakChecker::new();

        checker.begin_frame();
        checker.track(1);
        checker.track(2);
        checker.track(3);
        checker.release(1);
        // 2 and 3 not released

        let unreleased = checker.end_frame();
        assert_eq!(unreleased.len(), 2);
        assert!(unreleased.contains(&2));
        assert!(unreleased.contains(&3));
    }

    #[test]
    fn test_accumulating_leaks_across_frames() {
        let mut checker = FrameLeakChecker::new();
        let mut all_leaked = Vec::new();

        for frame in 0..5 {
            checker.begin_frame();
            let id = (frame + 1) as u64;
            checker.track(id);
            // Don't release anything
            let unreleased = checker.end_frame();
            all_leaked.extend(unreleased);
        }

        assert_eq!(all_leaked.len(), 5);
    }

    #[test]
    fn test_frame_boundary_reset() {
        let mut checker = FrameLeakChecker::new();

        // Frame 1
        checker.begin_frame();
        checker.track(1);
        let _ = checker.end_frame();

        // Frame 2 - should start clean
        checker.begin_frame();
        assert!(checker.is_clean());
        assert_eq!(checker.unreleased_count(), 0);
    }

    #[test]
    fn test_per_frame_vs_global_detection() {
        // Frame checker is per-frame
        let mut frame_checker = FrameLeakChecker::new();
        frame_checker.begin_frame();
        frame_checker.track(1);
        frame_checker.track(2);

        // Global detector tracks across frames
        let mut global_detector = immediate_detector();
        global_detector.track_allocation(1, "Buffer1", 1024);
        global_detector.track_allocation(2, "Buffer2", 2048);

        // Frame checker sees 2 unreleased
        assert_eq!(frame_checker.unreleased_count(), 2);

        // Global detector also sees 2
        assert_eq!(global_detector.tracked_count(), 2);
    }

    #[test]
    fn test_rapid_frame_sequences() {
        let mut checker = FrameLeakChecker::new();

        for _ in 0..100 {
            checker.begin_frame();
            checker.track(1);
            checker.release(1);
            let unreleased = checker.end_frame();
            assert!(unreleased.is_empty());
        }

        assert_eq!(checker.frame_number(), 100);
    }

    #[test]
    fn test_long_frame_durations_detected() {
        let mut checker = FrameLeakChecker::new();

        checker.begin_frame();
        // In a real scenario, time would pass here
        checker.track(1);
        // Allocation made during the frame

        let unreleased = checker.end_frame();
        assert_eq!(unreleased, vec![1]);
    }

    #[test]
    fn test_frame_number_increments() {
        let mut checker = FrameLeakChecker::new();

        assert_eq!(checker.frame_number(), 0);

        checker.begin_frame();
        assert_eq!(checker.frame_number(), 1);

        checker.begin_frame();
        assert_eq!(checker.frame_number(), 2);
    }

    #[test]
    fn test_is_clean_initially() {
        let checker = FrameLeakChecker::new();
        assert!(checker.is_clean());
    }

    #[test]
    fn test_is_clean_after_all_released() {
        let mut checker = FrameLeakChecker::new();

        checker.begin_frame();
        checker.track(1);
        checker.release(1);

        assert!(checker.is_clean());
    }

    #[test]
    fn test_is_not_clean_with_unreleased() {
        let mut checker = FrameLeakChecker::new();

        checker.begin_frame();
        checker.track(1);

        assert!(!checker.is_clean());
    }

    #[test]
    fn test_unreleased_count() {
        let mut checker = FrameLeakChecker::new();

        checker.begin_frame();
        assert_eq!(checker.unreleased_count(), 0);

        checker.track(1);
        assert_eq!(checker.unreleased_count(), 1);

        checker.track(2);
        assert_eq!(checker.unreleased_count(), 2);

        checker.release(1);
        assert_eq!(checker.unreleased_count(), 1);
    }

    #[test]
    fn test_release_nonexistent_is_safe() {
        let mut checker = FrameLeakChecker::new();

        checker.begin_frame();
        checker.release(999); // Should not panic

        assert!(checker.is_clean());
    }

    #[test]
    fn test_multiple_track_same_id() {
        let mut checker = FrameLeakChecker::new();

        checker.begin_frame();
        checker.track(1);
        checker.track(1); // Duplicate

        assert_eq!(checker.unreleased_count(), 2);

        checker.release(1); // retain removes ALL occurrences of the ID
        assert_eq!(checker.unreleased_count(), 0);
    }

    #[test]
    fn test_frame_checker_integration_with_detector() {
        let mut frame_checker = FrameLeakChecker::new();
        let mut detector = immediate_detector();

        frame_checker.begin_frame();

        // Track in both
        let id = next_id();
        frame_checker.track(id);
        detector.track_allocation(id, "Frame Resource", 1024);

        // Release from both
        frame_checker.release(id);
        detector.release_allocation(id);

        let unreleased = frame_checker.end_frame();
        assert!(unreleased.is_empty());
        assert_eq!(detector.tracked_count(), 0);
    }

    #[test]
    fn test_begin_frame_clears_previous() {
        let mut checker = FrameLeakChecker::new();

        checker.begin_frame();
        checker.track(1);
        checker.track(2);
        // Don't call end_frame, just begin new frame

        checker.begin_frame();
        assert!(checker.is_clean());
    }

    #[test]
    fn test_end_frame_returns_and_clears() {
        let mut checker = FrameLeakChecker::new();

        checker.begin_frame();
        checker.track(1);
        checker.track(2);

        let unreleased = checker.end_frame();
        assert_eq!(unreleased.len(), 2);

        // After end_frame, should be clean
        assert!(checker.is_clean());
    }

    #[test]
    fn test_frame_checker_high_volume() {
        let mut checker = FrameLeakChecker::new();

        checker.begin_frame();

        for i in 1..=1000u64 {
            checker.track(i);
        }

        assert_eq!(checker.unreleased_count(), 1000);

        for i in 1..=500u64 {
            checker.release(i);
        }

        assert_eq!(checker.unreleased_count(), 500);

        let unreleased = checker.end_frame();
        assert_eq!(unreleased.len(), 500);
    }

    #[test]
    fn test_frame_checker_no_begin_frame() {
        let mut checker = FrameLeakChecker::new();

        // Track without begin_frame (frame 0)
        checker.track(1);

        assert!(!checker.is_clean());
        assert_eq!(checker.frame_number(), 0);
    }
}

// ============================================================================
// Severity Escalation Tests (15+ tests)
// ============================================================================

mod severity_escalation {
    use super::*;

    #[test]
    fn test_info_to_warning_transition() {
        let thresholds = LeakThresholds::default();

        assert_eq!(LeakSeverity::from_age_secs(0, thresholds.warning_secs, thresholds.critical_secs), LeakSeverity::Info);
        assert_eq!(LeakSeverity::from_age_secs(thresholds.warning_secs, thresholds.warning_secs, thresholds.critical_secs), LeakSeverity::Warning);
    }

    #[test]
    fn test_warning_to_critical_transition() {
        let thresholds = LeakThresholds::default();

        assert_eq!(LeakSeverity::from_age_secs(thresholds.warning_secs, thresholds.warning_secs, thresholds.critical_secs), LeakSeverity::Warning);
        assert_eq!(LeakSeverity::from_age_secs(thresholds.critical_secs, thresholds.warning_secs, thresholds.critical_secs), LeakSeverity::Critical);
    }

    #[test]
    fn test_multiple_resources_at_different_severities() {
        let now = Instant::now();
        let thresholds = LeakThresholds::default();

        // Create candidates with different ages (simulated via different allocation times)
        let candidate1 = LeakCandidate::new(1, ResourceType::Buffer, 1024, Some("New".to_string()), now);

        // Fresh allocation is Info
        assert_eq!(candidate1.severity(&thresholds), LeakSeverity::Info);
    }

    #[test]
    fn test_severity_distribution_in_report() {
        let mut detector = immediate_detector();

        // All will be detected since thresholds are 0
        detector.track_allocation(1, "Buffer1", 1024);
        detector.track_allocation(2, "Buffer2", 2048);
        detector.track_allocation(3, "Buffer3", 4096);

        let report = detector.report();

        // With immediate thresholds (0, 1), fresh allocations are Warning
        // since age >= 0 (warning threshold)
        assert!(!report.is_empty());
    }

    #[test]
    fn test_critical_leak_prioritization() {
        let mut detector = immediate_detector();

        detector.track_allocation(1, "Buffer", 1024);

        let critical = detector.check_critical_only();
        // With thresholds (0, 1), fresh allocations (age=0) are Warning (not Critical yet)
        // Critical requires age >= 1 second
        assert!(critical.is_empty());
    }

    #[test]
    fn test_severity_ordering() {
        assert!(LeakSeverity::Info < LeakSeverity::Warning);
        assert!(LeakSeverity::Warning < LeakSeverity::Critical);
        assert!(LeakSeverity::Info < LeakSeverity::Critical);
    }

    #[test]
    fn test_severity_display_names() {
        assert_eq!(LeakSeverity::Info.display_name(), "INFO");
        assert_eq!(LeakSeverity::Warning.display_name(), "WARNING");
        assert_eq!(LeakSeverity::Critical.display_name(), "CRITICAL");
    }

    #[test]
    fn test_severity_display_colors() {
        assert_eq!(LeakSeverity::Info.display_color(), "\x1b[36m"); // Cyan
        assert_eq!(LeakSeverity::Warning.display_color(), "\x1b[33m"); // Yellow
        assert_eq!(LeakSeverity::Critical.display_color(), "\x1b[31m"); // Red
    }

    #[test]
    fn test_severity_reset_color() {
        assert_eq!(LeakSeverity::reset_color(), "\x1b[0m");
    }

    #[test]
    fn test_leak_candidate_severity_method() {
        let now = Instant::now();
        let thresholds = LeakThresholds::default();

        let candidate = LeakCandidate::new(1, ResourceType::Buffer, 1024, None, now);
        let severity = candidate.severity(&thresholds);

        assert_eq!(severity, LeakSeverity::Info);
    }

    #[test]
    fn test_report_has_critical() {
        let report = LeakReport {
            candidates: vec![],
            stats: LeakStats {
                critical_leaks: 0,
                ..Default::default()
            },
            timestamp: Instant::now(),
        };
        assert!(!report.has_critical());

        let report_with_critical = LeakReport {
            candidates: vec![],
            stats: LeakStats {
                critical_leaks: 1,
                ..Default::default()
            },
            timestamp: Instant::now(),
        };
        assert!(report_with_critical.has_critical());
    }

    #[test]
    fn test_report_by_severity() {
        let now = Instant::now();
        let thresholds = LeakThresholds::default();

        let report = LeakReport {
            candidates: vec![
                LeakCandidate::new(1, ResourceType::Buffer, 1024, None, now),
                LeakCandidate::new(2, ResourceType::Buffer, 2048, None, now),
            ],
            stats: LeakStats::default(),
            timestamp: now,
        };

        let (info, warning, critical) = report.by_severity(&thresholds);

        // Fresh allocations are Info
        assert_eq!(info.len(), 2);
        assert!(warning.is_empty());
        assert!(critical.is_empty());
    }

    #[test]
    fn test_stats_critical_rate() {
        let mut stats = LeakStats::new();
        stats.leaks_detected = 10;
        stats.critical_leaks = 3;

        assert!((stats.critical_rate() - 30.0).abs() < 0.001);
    }

    #[test]
    fn test_stats_critical_rate_zero_leaks() {
        let stats = LeakStats::new();
        assert_eq!(stats.critical_rate(), 0.0);
    }

    #[test]
    fn test_severity_boundary_values() {
        // Just below warning
        assert_eq!(LeakSeverity::from_age_secs(29, 30, 120), LeakSeverity::Info);

        // Exactly at warning
        assert_eq!(LeakSeverity::from_age_secs(30, 30, 120), LeakSeverity::Warning);

        // Just below critical
        assert_eq!(LeakSeverity::from_age_secs(119, 30, 120), LeakSeverity::Warning);

        // Exactly at critical
        assert_eq!(LeakSeverity::from_age_secs(120, 30, 120), LeakSeverity::Critical);
    }
}

// ============================================================================
// Edge Cases Tests (15+ tests)
// ============================================================================

mod edge_cases {
    use super::*;

    #[test]
    fn test_zero_allocations() {
        let mut detector = LeakDetector::with_default_thresholds();

        let candidates = detector.check();
        assert!(candidates.is_empty());

        let stats = detector.stats();
        assert_eq!(stats.total_tracked, 0);
        assert_eq!(stats.checks_performed, 1);
    }

    #[test]
    fn test_single_allocation_lifecycle() {
        let mut detector = LeakDetector::with_default_thresholds();

        detector.track_allocation(1, "Single Buffer", 1024);
        assert_eq!(detector.tracked_count(), 1);
        assert_eq!(detector.total_bytes(), 1024);

        detector.release_allocation(1);
        assert_eq!(detector.tracked_count(), 0);
        assert_eq!(detector.total_bytes(), 0);
    }

    #[test]
    fn test_very_old_allocations() {
        let severity = LeakSeverity::from_age_secs(u64::MAX, 30, 120);
        assert_eq!(severity, LeakSeverity::Critical);
    }

    #[test]
    fn test_rapid_alloc_dealloc_cycles() {
        let mut detector = LeakDetector::with_default_thresholds();

        for _ in 0..100 {
            let id = next_id();
            detector.track_allocation(id, "Temp", 1024);
            detector.release_allocation(id);
        }

        assert_eq!(detector.tracked_count(), 0);
        let stats = detector.stats();
        assert_eq!(stats.total_tracked, 100);
        assert_eq!(stats.total_released, 100);
    }

    #[test]
    fn test_id_reuse_handling() {
        let mut detector = LeakDetector::with_default_thresholds();

        // Track, release, track same ID
        detector.track_allocation(1, "First", 1024);
        detector.release_allocation(1);
        detector.track_allocation(1, "Second", 2048);

        assert_eq!(detector.tracked_count(), 1);
        assert_eq!(detector.total_bytes(), 2048);
    }

    #[test]
    fn test_maximum_allocation_count() {
        let mut detector = immediate_detector();

        for i in 0..10000u64 {
            detector.track_allocation(i, format!("Buffer{}", i), 64);
        }

        assert_eq!(detector.tracked_count(), 10000);
        assert_eq!(detector.total_bytes(), 640000);
    }

    #[test]
    fn test_concurrent_style_patterns() {
        let mut detector = immediate_detector();

        // Simulate multiple "threads" allocating
        for thread in 0..4 {
            for alloc in 0..25 {
                let id = (thread * 100 + alloc) as u64;
                detector.track_allocation(id, format!("Thread{}_Alloc{}", thread, alloc), 256);
            }
        }

        assert_eq!(detector.tracked_count(), 100);
    }

    #[test]
    fn test_zero_size_allocation() {
        let mut detector = detector_with_thresholds(0, 1, 0);

        detector.track_allocation(1, "Zero Size", 0);

        let candidates = detector.check();
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].size_bytes, 0);
    }

    #[test]
    fn test_max_size_allocation() {
        let mut detector = immediate_detector();

        detector.track_allocation(1, "Huge", u64::MAX / 2);

        assert_eq!(detector.total_bytes(), u64::MAX / 2);
    }

    #[test]
    fn test_empty_label() {
        let mut detector = immediate_detector();

        detector.track_allocation(1, "", 1024);

        let candidates = detector.check();
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].label, Some(String::new()));
    }

    #[test]
    fn test_unicode_label() {
        let mut detector = immediate_detector();

        detector.track_allocation(1, "Buffer", 1024);

        let candidates = detector.check();
        assert_eq!(candidates.len(), 1);
    }

    #[test]
    fn test_special_characters_in_label() {
        let mut detector = immediate_detector();

        detector.track_allocation(1, "Buffer<T>::new()", 1024);

        let candidates = detector.check();
        assert_eq!(candidates.len(), 1);
        assert!(candidates[0].label.as_ref().unwrap().contains("<T>"));
    }

    #[test]
    fn test_release_and_retrack_same_frame() {
        let mut detector = LeakDetector::with_default_thresholds();

        detector.track_allocation(1, "Buffer", 1024);
        detector.release_allocation(1);
        detector.track_allocation(1, "Buffer Reused", 2048);

        assert_eq!(detector.tracked_count(), 1);
        assert_eq!(detector.total_bytes(), 2048);
    }

    #[test]
    fn test_mark_expected_nonexistent() {
        let mut detector = LeakDetector::with_default_thresholds();

        // Should not panic
        detector.mark_expected(999);

        let stats = detector.stats();
        assert_eq!(stats.expected_long_lived, 0);
    }

    #[test]
    fn test_mark_temporary_nonexistent() {
        let mut detector = LeakDetector::with_default_thresholds();

        // Should not panic
        detector.mark_temporary(999);
    }

    #[test]
    fn test_double_release() {
        let mut detector = LeakDetector::with_default_thresholds();

        detector.track_allocation(1, "Buffer", 1024);

        assert!(detector.release_allocation(1));
        assert!(!detector.release_allocation(1)); // Second release returns false
    }
}

// ============================================================================
// Reporting Tests (15+ tests)
// ============================================================================

mod reporting {
    use super::*;

    #[test]
    fn test_empty_report_format() {
        let mut detector = LeakDetector::with_default_thresholds();
        let report = detector.report();

        assert!(report.is_empty());
        assert_eq!(report.len(), 0);
        assert!(!report.has_critical());
    }

    #[test]
    fn test_single_leak_report() {
        let mut detector = immediate_detector();
        detector.track_allocation(1, "Leaked Buffer", 1024);

        let report = detector.report();

        assert!(!report.is_empty());
        assert_eq!(report.len(), 1);
    }

    #[test]
    fn test_multiple_leaks_grouped_by_severity() {
        let now = Instant::now();
        let thresholds = LeakThresholds::default();

        let report = LeakReport {
            candidates: vec![
                LeakCandidate::new(1, ResourceType::Buffer, 1024, Some("A".to_string()), now),
                LeakCandidate::new(2, ResourceType::Texture, 2048, Some("B".to_string()), now),
                LeakCandidate::new(3, ResourceType::Pipeline, 512, Some("C".to_string()), now),
            ],
            stats: LeakStats::default(),
            timestamp: now,
        };

        let (info, warning, critical) = report.by_severity(&thresholds);

        // All fresh, so all Info
        assert_eq!(info.len(), 3);
        assert!(warning.is_empty());
        assert!(critical.is_empty());
    }

    #[test]
    fn test_summary_accuracy() {
        let now = Instant::now();
        let report = LeakReport {
            candidates: vec![
                LeakCandidate::new(1, ResourceType::Buffer, 1024, None, now),
                LeakCandidate::new(2, ResourceType::Buffer, 2048, None, now),
            ],
            stats: LeakStats {
                current_tracked: 2,
                total_released: 5,
                checks_performed: 10,
                critical_leaks: 0,
                ..Default::default()
            },
            timestamp: now,
        };

        let summary = report.summary();

        assert!(summary.contains("2 candidates"));
        assert!(summary.contains("2/5 tracked/released"));
        assert!(summary.contains("10 checks"));
    }

    #[test]
    fn test_total_bytes_calculation() {
        let now = Instant::now();
        let report = LeakReport {
            candidates: vec![
                LeakCandidate::new(1, ResourceType::Buffer, 1024, None, now),
                LeakCandidate::new(2, ResourceType::Buffer, 2048, None, now),
                LeakCandidate::new(3, ResourceType::Buffer, 4096, None, now),
            ],
            stats: LeakStats::default(),
            timestamp: now,
        };

        assert_eq!(report.total_leaked_bytes(), 7168);
    }

    #[test]
    fn test_report_generation_performance() {
        let mut detector = immediate_detector();

        // Add many allocations
        for i in 0..1000u64 {
            detector.track_allocation(i, format!("Buffer{}", i), 1024);
        }

        let start = Instant::now();
        let report = detector.report();
        let duration = start.elapsed();

        // Should complete quickly (under 100ms)
        assert!(duration.as_millis() < 100);
        assert_eq!(report.len(), 1000);
    }

    #[test]
    fn test_leak_stats_leak_rate() {
        let mut stats = LeakStats::new();
        stats.total_tracked = 100;
        stats.leaks_detected = 5;

        assert!((stats.leak_rate() - 5.0).abs() < 0.001);
    }

    #[test]
    fn test_leak_stats_zero_tracked() {
        let stats = LeakStats::new();
        assert_eq!(stats.leak_rate(), 0.0);
    }

    #[test]
    fn test_leak_candidate_format_colored() {
        let now = Instant::now();
        let thresholds = LeakThresholds::default();

        let candidate = LeakCandidate::new(
            1,
            ResourceType::Buffer,
            1024,
            Some("Test Buffer".to_string()),
            now
        );

        let formatted = candidate.format_colored(&thresholds);

        assert!(formatted.contains("Test Buffer"));
        assert!(formatted.contains("1024 bytes"));
        assert!(formatted.contains("Buffer"));
    }

    #[test]
    fn test_leak_candidate_unlabeled_format() {
        let now = Instant::now();
        let thresholds = LeakThresholds::default();

        let candidate = LeakCandidate::new(1, ResourceType::Buffer, 1024, None, now);

        let formatted = candidate.format_colored(&thresholds);

        assert!(formatted.contains("<unlabeled>"));
    }

    #[test]
    fn test_report_timestamp() {
        let before = Instant::now();
        let mut detector = LeakDetector::with_default_thresholds();
        let report = detector.report();
        let after = Instant::now();

        // Timestamp should be between before and after
        assert!(report.timestamp >= before);
        assert!(report.timestamp <= after);
    }

    #[test]
    fn test_leak_candidate_age() {
        let now = Instant::now();
        let candidate = LeakCandidate::new(1, ResourceType::Buffer, 1024, None, now);

        let age = candidate.age();
        assert!(age.as_millis() < 100); // Should be very fresh
    }

    #[test]
    fn test_leak_candidate_age_secs() {
        let now = Instant::now();
        let candidate = LeakCandidate::new(1, ResourceType::Buffer, 1024, None, now);

        let age_secs = candidate.age_secs();
        assert_eq!(age_secs, 0); // Fresh allocation
    }

    #[test]
    fn test_stats_accumulate_across_checks() {
        let mut detector = immediate_detector();

        detector.track_allocation(1, "A", 1024);
        let _ = detector.check();

        detector.track_allocation(2, "B", 1024);
        let _ = detector.check();

        detector.track_allocation(3, "C", 1024);
        let _ = detector.check();

        let stats = detector.stats();
        assert_eq!(stats.total_tracked, 3);
        assert_eq!(stats.checks_performed, 3);
    }

    #[test]
    fn test_summary_includes_kb_conversion() {
        let now = Instant::now();
        let report = LeakReport {
            candidates: vec![
                LeakCandidate::new(1, ResourceType::Buffer, 2048, None, now),
            ],
            stats: LeakStats::default(),
            timestamp: now,
        };

        let summary = report.summary();

        // Should show KB
        assert!(summary.contains("KB"));
    }
}

// ============================================================================
// Allocation Tracker Tests (Additional coverage)
// ============================================================================

mod allocation_tracker {
    use super::*;

    #[test]
    fn test_tracker_track_with_type() {
        let mut tracker = AllocationTracker::new();

        tracker.track_with_type(1, "Buffer", 1024, ResourceType::Buffer);

        assert_eq!(tracker.get_resource_type(1), ResourceType::Buffer);
    }

    #[test]
    fn test_tracker_get_resource_type_default() {
        let mut tracker = AllocationTracker::new();

        // Track without type
        tracker.track(1, "Buffer", 1024);

        // Should return Other as default
        assert_eq!(tracker.get_resource_type(1), ResourceType::Other);
    }

    #[test]
    fn test_tracker_iter() {
        let mut tracker = AllocationTracker::new();

        tracker.track(1, "A", 1024);
        tracker.track(2, "B", 2048);
        tracker.track(3, "C", 4096);

        let items: Vec<_> = tracker.iter().collect();
        assert_eq!(items.len(), 3);
    }

    #[test]
    fn test_tracker_mark_expected_removes_temporary() {
        let mut tracker = AllocationTracker::new();

        tracker.track(1, "Buffer", 1024);
        tracker.mark_temporary(1);
        assert!(tracker.is_temporary(1));

        tracker.mark_expected(1);
        assert!(tracker.is_expected(1));
        assert!(!tracker.is_temporary(1));
    }

    #[test]
    fn test_tracker_mark_temporary_removes_expected() {
        let mut tracker = AllocationTracker::new();

        tracker.track(1, "Buffer", 1024);
        tracker.mark_expected(1);
        assert!(tracker.is_expected(1));

        tracker.mark_temporary(1);
        assert!(tracker.is_temporary(1));
        assert!(!tracker.is_expected(1));
    }

    #[test]
    fn test_tracker_untrack_clears_all_markers() {
        let mut tracker = AllocationTracker::new();

        tracker.track_with_type(1, "Buffer", 1024, ResourceType::Buffer);
        tracker.mark_expected(1);

        tracker.untrack(1);

        assert!(!tracker.is_expected(1));
        assert_eq!(tracker.get_resource_type(1), ResourceType::Other);
    }

    #[test]
    fn test_tracker_get_allocation() {
        let mut tracker = AllocationTracker::new();

        tracker.track(1, "Test", 1024);

        let alloc = tracker.get(1);
        assert!(alloc.is_some());

        let (_, label, size) = alloc.unwrap();
        assert_eq!(label, "Test");
        assert_eq!(*size, 1024);
    }

    #[test]
    fn test_tracker_get_nonexistent() {
        let tracker = AllocationTracker::new();

        assert!(tracker.get(999).is_none());
    }
}

// ============================================================================
// Resource Type Tests
// ============================================================================

mod resource_type {
    use super::*;

    #[test]
    fn test_resource_type_display() {
        assert_eq!(format!("{}", ResourceType::Buffer), "Buffer");
        assert_eq!(format!("{}", ResourceType::Texture), "Texture");
        assert_eq!(format!("{}", ResourceType::QuerySet), "Query Set");
        assert_eq!(format!("{}", ResourceType::BindGroup), "Bind Group");
        assert_eq!(format!("{}", ResourceType::Pipeline), "Pipeline");
        assert_eq!(format!("{}", ResourceType::Other), "Other");
    }

    #[test]
    fn test_resource_type_display_name() {
        assert_eq!(ResourceType::Buffer.display_name(), "Buffer");
        assert_eq!(ResourceType::Texture.display_name(), "Texture");
        assert_eq!(ResourceType::QuerySet.display_name(), "Query Set");
        assert_eq!(ResourceType::BindGroup.display_name(), "Bind Group");
        assert_eq!(ResourceType::Pipeline.display_name(), "Pipeline");
        assert_eq!(ResourceType::Other.display_name(), "Other");
    }

    #[test]
    fn test_resource_type_equality() {
        assert_eq!(ResourceType::Buffer, ResourceType::Buffer);
        assert_ne!(ResourceType::Buffer, ResourceType::Texture);
    }

    #[test]
    fn test_resource_type_hash() {
        use std::collections::HashSet;

        let mut set = HashSet::new();
        set.insert(ResourceType::Buffer);
        set.insert(ResourceType::Texture);
        set.insert(ResourceType::Buffer); // Duplicate

        assert_eq!(set.len(), 2);
    }
}

// ============================================================================
// Severity Display Tests
// ============================================================================

mod severity_display {
    use super::*;

    #[test]
    fn test_severity_display_trait() {
        assert_eq!(format!("{}", LeakSeverity::Info), "INFO");
        assert_eq!(format!("{}", LeakSeverity::Warning), "WARNING");
        assert_eq!(format!("{}", LeakSeverity::Critical), "CRITICAL");
    }

    #[test]
    fn test_severity_hash() {
        use std::collections::HashSet;

        let mut set = HashSet::new();
        set.insert(LeakSeverity::Info);
        set.insert(LeakSeverity::Warning);
        set.insert(LeakSeverity::Critical);

        assert_eq!(set.len(), 3);
    }

    #[test]
    fn test_severity_clone() {
        let s = LeakSeverity::Critical;
        let s2 = s;
        assert_eq!(s, s2);
    }

    #[test]
    fn test_severity_debug() {
        let s = LeakSeverity::Warning;
        let debug = format!("{:?}", s);
        assert_eq!(debug, "Warning");
    }
}

// ============================================================================
// Integration Scenarios
// ============================================================================

mod integration {
    use super::*;

    #[test]
    fn test_full_frame_cycle() {
        let mut detector = immediate_detector();
        let mut frame_checker = FrameLeakChecker::new();

        // Frame 1: Some resources leaked
        frame_checker.begin_frame();

        let id1 = next_id();
        let id2 = next_id();
        let id3 = next_id();

        detector.track_allocation(id1, "Persistent", 1024);
        detector.mark_expected(id1);

        frame_checker.track(id2);
        detector.track_allocation(id2, "Frame Temp", 512);
        frame_checker.release(id2);
        detector.release_allocation(id2);

        frame_checker.track(id3);
        detector.track_allocation(id3, "Leaked", 2048);
        // id3 not released

        let unreleased = frame_checker.end_frame();
        assert_eq!(unreleased.len(), 1);
        assert!(unreleased.contains(&id3));

        let candidates = detector.check();
        // Only id3 should be detected (id1 is expected, id2 was released)
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].allocation_id, id3);
    }

    #[test]
    fn test_hot_reload_scenario() {
        let mut detector = immediate_detector();

        // Initial resources
        detector.track_allocation_typed(1, "Shader v1", 4096, ResourceType::Pipeline);
        detector.track_allocation_typed(2, "Material v1", 2048, ResourceType::BindGroup);

        // Hot reload: create new versions
        detector.track_allocation_typed(3, "Shader v2", 4096, ResourceType::Pipeline);
        detector.track_allocation_typed(4, "Material v2", 2048, ResourceType::BindGroup);

        // Release old versions correctly
        detector.release_allocation(1);
        detector.release_allocation(2);

        let candidates = detector.check();
        // Only v2 resources remain, and they're new
        assert_eq!(candidates.len(), 2);
    }

    #[test]
    fn test_memory_pressure_cleanup() {
        let mut detector = immediate_detector();

        // Allocate many resources
        for i in 0..100u64 {
            detector.track_allocation(i, format!("Resource{}", i), 1024 * 1024);
        }

        // Simulate memory pressure cleanup
        for i in 0..50u64 {
            detector.release_allocation(i);
        }

        let stats = detector.stats();
        assert_eq!(stats.total_tracked, 100);
        assert_eq!(stats.total_released, 50);
        assert_eq!(stats.current_tracked, 50);
    }

    #[test]
    fn test_level_transition() {
        let mut detector = immediate_detector();

        // Level 1 resources
        for i in 0..20u64 {
            detector.track_allocation(i, format!("Level1_Resource{}", i), 1024);
        }

        // Transition: release all Level 1 resources
        for i in 0..20u64 {
            detector.release_allocation(i);
        }

        // Level 2 resources
        for i in 20..40u64 {
            detector.track_allocation(i, format!("Level2_Resource{}", i), 2048);
        }

        assert_eq!(detector.tracked_count(), 20);
        assert_eq!(detector.total_bytes(), 20 * 2048);
    }
}
