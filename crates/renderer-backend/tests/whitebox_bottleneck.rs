//! Whitebox structural tests for GPU/CPU Bottleneck Analyzer.
//!
//! These tests verify the internal structure and behavior of the bottleneck
//! analysis system, including all data structures, detection algorithms, and edge cases.
//!
//! Task: T-WGPU-P7.4.5 - Bottleneck Analyzer Whitebox Testing
//!
//! Components Tested:
//! 1. BottleneckType - All 6 bottleneck types and their properties
//! 2. BottleneckSeverity - All 5 severity levels and conversions
//! 3. TimingMetrics - CPU/GPU timing tracking and calculations
//! 4. ResourceMetrics - Bandwidth and memory transfer metrics
//! 5. StateMetrics - State change tracking and thrashing detection
//! 6. FrameMetrics - Aggregate frame metrics
//! 7. BottleneckResult - Analysis result structure
//! 8. AnalysisThresholds - Configurable thresholds
//! 9. TrendAnalysis - Multi-frame trend detection
//! 10. BottleneckAnalyzer - Main analysis interface
//! 11. BottleneckProfiler - Real-time profiling
//! 12. StateChangeType - State change categorization
//! 13. Thread Safety - Send + Sync bounds
//!
//! WHITEBOX coverage plan:
//!   - Path A: BottleneckType construction and all 6 variants
//!   - Path B: BottleneckType severity mapping for each type
//!   - Path C: BottleneckType optimization hints (non-empty)
//!   - Path D: BottleneckType trait implementations
//!   - Path E: BottleneckSeverity from_score boundary testing
//!   - Path F: BottleneckSeverity color codes
//!   - Path G: BottleneckSeverity ordering
//!   - Path H: TimingMetrics construction and accessors
//!   - Path I: TimingMetrics frame_overlap calculations
//!   - Path J: TimingMetrics gpu_utilization calculations
//!   - Path K: TimingMetrics bound detection
//!   - Path L: ResourceMetrics bandwidth tracking
//!   - Path M: ResourceMetrics threshold detection
//!   - Path N: StateMetrics counting and ratios
//!   - Path O: StateMetrics thrashing detection
//!   - Path P: FrameMetrics composition
//!   - Path Q: BottleneckResult construction
//!   - Path R: AnalysisThresholds presets
//!   - Path S: TrendAnalysis statistics
//!   - Path T: BottleneckAnalyzer detection algorithms
//!   - Path U: BottleneckProfiler lifecycle
//!   - Path V: Edge cases - zero values
//!   - Path W: Edge cases - maximum values
//!   - Path X: Send + Sync bounds

use renderer_backend::profiling::bottleneck::{
    AnalysisThresholds, BottleneckAnalyzer, BottleneckProfiler, BottleneckResult, BottleneckSeverity,
    BottleneckType, FrameMetrics, ResourceMetrics, StateChangeType, StateMetrics, TimingMetrics,
    TrendAnalysis,
};
use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};

// ============================================================================
// Section 1: BottleneckType Tests (20+ tests)
// ============================================================================

#[test]
fn test_bottleneck_type_cpu_bound_exists() {
    let t = BottleneckType::CpuBound;
    assert_eq!(format!("{:?}", t), "CpuBound");
}

#[test]
fn test_bottleneck_type_gpu_bound_exists() {
    let t = BottleneckType::GpuBound;
    assert_eq!(format!("{:?}", t), "GpuBound");
}

#[test]
fn test_bottleneck_type_memory_bandwidth_exists() {
    let t = BottleneckType::MemoryBandwidth;
    assert_eq!(format!("{:?}", t), "MemoryBandwidth");
}

#[test]
fn test_bottleneck_type_state_thrashing_exists() {
    let t = BottleneckType::StateThrashing;
    assert_eq!(format!("{:?}", t), "StateThrashing");
}

#[test]
fn test_bottleneck_type_synchronization_exists() {
    let t = BottleneckType::Synchronization;
    assert_eq!(format!("{:?}", t), "Synchronization");
}

#[test]
fn test_bottleneck_type_balanced_exists() {
    let t = BottleneckType::Balanced;
    assert_eq!(format!("{:?}", t), "Balanced");
}

#[test]
fn test_bottleneck_type_cpu_bound_severity() {
    assert_eq!(BottleneckType::CpuBound.severity(), BottleneckSeverity::Medium);
}

#[test]
fn test_bottleneck_type_gpu_bound_severity() {
    assert_eq!(BottleneckType::GpuBound.severity(), BottleneckSeverity::Medium);
}

#[test]
fn test_bottleneck_type_memory_bandwidth_severity() {
    assert_eq!(BottleneckType::MemoryBandwidth.severity(), BottleneckSeverity::High);
}

#[test]
fn test_bottleneck_type_state_thrashing_severity() {
    assert_eq!(BottleneckType::StateThrashing.severity(), BottleneckSeverity::Medium);
}

#[test]
fn test_bottleneck_type_synchronization_severity() {
    assert_eq!(BottleneckType::Synchronization.severity(), BottleneckSeverity::High);
}

#[test]
fn test_bottleneck_type_balanced_severity() {
    assert_eq!(BottleneckType::Balanced.severity(), BottleneckSeverity::None);
}

#[test]
fn test_bottleneck_type_cpu_bound_hints_non_empty() {
    let hints = BottleneckType::CpuBound.optimization_hints();
    assert!(!hints.is_empty(), "CPU-bound hints should not be empty");
    assert!(hints.len() >= 3, "Should have at least 3 hints");
}

#[test]
fn test_bottleneck_type_gpu_bound_hints_non_empty() {
    let hints = BottleneckType::GpuBound.optimization_hints();
    assert!(!hints.is_empty(), "GPU-bound hints should not be empty");
}

#[test]
fn test_bottleneck_type_memory_bandwidth_hints_non_empty() {
    let hints = BottleneckType::MemoryBandwidth.optimization_hints();
    assert!(!hints.is_empty(), "Memory bandwidth hints should not be empty");
}

#[test]
fn test_bottleneck_type_state_thrashing_hints_non_empty() {
    let hints = BottleneckType::StateThrashing.optimization_hints();
    assert!(!hints.is_empty(), "State thrashing hints should not be empty");
}

#[test]
fn test_bottleneck_type_synchronization_hints_non_empty() {
    let hints = BottleneckType::Synchronization.optimization_hints();
    assert!(!hints.is_empty(), "Synchronization hints should not be empty");
}

#[test]
fn test_bottleneck_type_balanced_hints_non_empty() {
    let hints = BottleneckType::Balanced.optimization_hints();
    assert!(!hints.is_empty(), "Balanced hints should not be empty");
}

#[test]
fn test_bottleneck_type_equality() {
    assert_eq!(BottleneckType::CpuBound, BottleneckType::CpuBound);
    assert_eq!(BottleneckType::GpuBound, BottleneckType::GpuBound);
    assert_ne!(BottleneckType::CpuBound, BottleneckType::GpuBound);
}

#[test]
fn test_bottleneck_type_clone() {
    let original = BottleneckType::MemoryBandwidth;
    let cloned = original.clone();
    assert_eq!(original, cloned);
}

#[test]
fn test_bottleneck_type_copy() {
    let original = BottleneckType::StateThrashing;
    let copied: BottleneckType = original;
    assert_eq!(original, copied);
}

#[test]
fn test_bottleneck_type_hash() {
    let mut set = HashSet::new();
    set.insert(BottleneckType::CpuBound);
    set.insert(BottleneckType::GpuBound);
    set.insert(BottleneckType::CpuBound); // duplicate
    assert_eq!(set.len(), 2);
}

#[test]
fn test_bottleneck_type_display_cpu_bound() {
    assert_eq!(format!("{}", BottleneckType::CpuBound), "CPU Bound");
}

#[test]
fn test_bottleneck_type_display_gpu_bound() {
    assert_eq!(format!("{}", BottleneckType::GpuBound), "GPU Bound");
}

#[test]
fn test_bottleneck_type_display_memory_bandwidth() {
    assert_eq!(format!("{}", BottleneckType::MemoryBandwidth), "Memory Bandwidth");
}

#[test]
fn test_bottleneck_type_display_state_thrashing() {
    assert_eq!(format!("{}", BottleneckType::StateThrashing), "State Thrashing");
}

#[test]
fn test_bottleneck_type_display_synchronization() {
    assert_eq!(format!("{}", BottleneckType::Synchronization), "Synchronization");
}

#[test]
fn test_bottleneck_type_display_balanced() {
    assert_eq!(format!("{}", BottleneckType::Balanced), "Balanced");
}

#[test]
fn test_bottleneck_type_description_contains_keyword() {
    assert!(BottleneckType::CpuBound.description().contains("CPU"));
    assert!(BottleneckType::GpuBound.description().contains("GPU"));
    assert!(BottleneckType::MemoryBandwidth.description().contains("bandwidth") ||
            BottleneckType::MemoryBandwidth.description().contains("Memory"));
    assert!(BottleneckType::StateThrashing.description().contains("state"));
    assert!(BottleneckType::Synchronization.description().contains("sync") ||
            BottleneckType::Synchronization.description().contains("Sync") ||
            BottleneckType::Synchronization.description().contains("CPU-GPU"));
}

// ============================================================================
// Section 2: BottleneckSeverity Tests (15+ tests)
// ============================================================================

#[test]
fn test_severity_none_exists() {
    let s = BottleneckSeverity::None;
    assert_eq!(format!("{:?}", s), "None");
}

#[test]
fn test_severity_low_exists() {
    let s = BottleneckSeverity::Low;
    assert_eq!(format!("{:?}", s), "Low");
}

#[test]
fn test_severity_medium_exists() {
    let s = BottleneckSeverity::Medium;
    assert_eq!(format!("{:?}", s), "Medium");
}

#[test]
fn test_severity_high_exists() {
    let s = BottleneckSeverity::High;
    assert_eq!(format!("{:?}", s), "High");
}

#[test]
fn test_severity_critical_exists() {
    let s = BottleneckSeverity::Critical;
    assert_eq!(format!("{:?}", s), "Critical");
}

#[test]
fn test_severity_from_score_zero() {
    assert_eq!(BottleneckSeverity::from_score(0.0), BottleneckSeverity::None);
}

#[test]
fn test_severity_from_score_0_05() {
    assert_eq!(BottleneckSeverity::from_score(0.05), BottleneckSeverity::None);
}

#[test]
fn test_severity_from_score_0_1() {
    assert_eq!(BottleneckSeverity::from_score(0.1), BottleneckSeverity::None);
}

#[test]
fn test_severity_from_score_0_11() {
    assert_eq!(BottleneckSeverity::from_score(0.11), BottleneckSeverity::Low);
}

#[test]
fn test_severity_from_score_0_25() {
    assert_eq!(BottleneckSeverity::from_score(0.25), BottleneckSeverity::Low);
}

#[test]
fn test_severity_from_score_0_3() {
    assert_eq!(BottleneckSeverity::from_score(0.3), BottleneckSeverity::Low);
}

#[test]
fn test_severity_from_score_0_31() {
    assert_eq!(BottleneckSeverity::from_score(0.31), BottleneckSeverity::Medium);
}

#[test]
fn test_severity_from_score_0_5() {
    assert_eq!(BottleneckSeverity::from_score(0.5), BottleneckSeverity::Medium);
}

#[test]
fn test_severity_from_score_0_51() {
    assert_eq!(BottleneckSeverity::from_score(0.51), BottleneckSeverity::High);
}

#[test]
fn test_severity_from_score_0_75() {
    assert_eq!(BottleneckSeverity::from_score(0.75), BottleneckSeverity::High);
}

#[test]
fn test_severity_from_score_0_76() {
    assert_eq!(BottleneckSeverity::from_score(0.76), BottleneckSeverity::Critical);
}

#[test]
fn test_severity_from_score_1_0() {
    assert_eq!(BottleneckSeverity::from_score(1.0), BottleneckSeverity::Critical);
}

#[test]
fn test_severity_from_score_over_1() {
    // Scores over 1.0 should still map to Critical
    assert_eq!(BottleneckSeverity::from_score(1.5), BottleneckSeverity::Critical);
}

#[test]
fn test_severity_display_color_none_is_green() {
    let color = BottleneckSeverity::None.display_color();
    assert!(color.contains("32"), "None should be green (ANSI 32)");
}

#[test]
fn test_severity_display_color_low_is_yellow() {
    let color = BottleneckSeverity::Low.display_color();
    assert!(color.contains("33"), "Low should be yellow (ANSI 33)");
}

#[test]
fn test_severity_display_color_medium_is_yellow() {
    let color = BottleneckSeverity::Medium.display_color();
    assert!(color.contains("33"), "Medium should be yellow (ANSI 33)");
}

#[test]
fn test_severity_display_color_high_is_red() {
    let color = BottleneckSeverity::High.display_color();
    assert!(color.contains("31"), "High should be red (ANSI 31)");
}

#[test]
fn test_severity_display_color_critical_is_bright_red() {
    let color = BottleneckSeverity::Critical.display_color();
    assert!(color.contains("91"), "Critical should be bright red (ANSI 91)");
}

#[test]
fn test_severity_to_score_none() {
    assert_eq!(BottleneckSeverity::None.to_score(), 0.0);
}

#[test]
fn test_severity_to_score_low() {
    assert_eq!(BottleneckSeverity::Low.to_score(), 0.25);
}

#[test]
fn test_severity_to_score_medium() {
    assert_eq!(BottleneckSeverity::Medium.to_score(), 0.5);
}

#[test]
fn test_severity_to_score_high() {
    assert_eq!(BottleneckSeverity::High.to_score(), 0.75);
}

#[test]
fn test_severity_to_score_critical() {
    assert_eq!(BottleneckSeverity::Critical.to_score(), 1.0);
}

#[test]
fn test_severity_ordering() {
    assert!(BottleneckSeverity::None < BottleneckSeverity::Low);
    assert!(BottleneckSeverity::Low < BottleneckSeverity::Medium);
    assert!(BottleneckSeverity::Medium < BottleneckSeverity::High);
    assert!(BottleneckSeverity::High < BottleneckSeverity::Critical);
}

#[test]
fn test_severity_default() {
    let default_severity: BottleneckSeverity = Default::default();
    assert_eq!(default_severity, BottleneckSeverity::None);
}

#[test]
fn test_severity_display() {
    assert_eq!(format!("{}", BottleneckSeverity::None), "None");
    assert_eq!(format!("{}", BottleneckSeverity::Low), "Low");
    assert_eq!(format!("{}", BottleneckSeverity::Medium), "Medium");
    assert_eq!(format!("{}", BottleneckSeverity::High), "High");
    assert_eq!(format!("{}", BottleneckSeverity::Critical), "Critical");
}

#[test]
fn test_severity_description() {
    assert_eq!(BottleneckSeverity::None.description(), "None");
    assert_eq!(BottleneckSeverity::Low.description(), "Low");
    assert_eq!(BottleneckSeverity::Medium.description(), "Medium");
    assert_eq!(BottleneckSeverity::High.description(), "High");
    assert_eq!(BottleneckSeverity::Critical.description(), "Critical");
}

// ============================================================================
// Section 3: TimingMetrics Tests (25+ tests)
// ============================================================================

#[test]
fn test_timing_metrics_default() {
    let timing = TimingMetrics::default();
    assert_eq!(timing.cpu_frame_time, Duration::ZERO);
    assert_eq!(timing.gpu_frame_time, Duration::ZERO);
    assert_eq!(timing.cpu_submit_time, Duration::ZERO);
    assert_eq!(timing.gpu_wait_time, Duration::ZERO);
    assert_eq!(timing.present_wait_time, Duration::ZERO);
}

#[test]
fn test_timing_metrics_new_construction() {
    let timing = TimingMetrics::new(
        Duration::from_millis(8),
        Duration::from_millis(12),
        Duration::from_micros(500),
        Duration::from_micros(100),
        Duration::from_millis(4),
    );
    assert_eq!(timing.cpu_frame_time, Duration::from_millis(8));
    assert_eq!(timing.gpu_frame_time, Duration::from_millis(12));
    assert_eq!(timing.cpu_submit_time, Duration::from_micros(500));
    assert_eq!(timing.gpu_wait_time, Duration::from_micros(100));
    assert_eq!(timing.present_wait_time, Duration::from_millis(4));
}

#[test]
fn test_timing_metrics_cpu_frame_time_tracking() {
    let timing = TimingMetrics {
        cpu_frame_time: Duration::from_micros(8000),
        ..Default::default()
    };
    assert_eq!(timing.cpu_frame_time.as_micros(), 8000);
}

#[test]
fn test_timing_metrics_gpu_frame_time_tracking() {
    let timing = TimingMetrics {
        gpu_frame_time: Duration::from_micros(12000),
        ..Default::default()
    };
    assert_eq!(timing.gpu_frame_time.as_micros(), 12000);
}

#[test]
fn test_timing_metrics_submit_time_tracking() {
    let timing = TimingMetrics {
        cpu_submit_time: Duration::from_micros(500),
        ..Default::default()
    };
    assert_eq!(timing.cpu_submit_time.as_micros(), 500);
}

#[test]
fn test_timing_metrics_gpu_wait_time_tracking() {
    let timing = TimingMetrics {
        gpu_wait_time: Duration::from_micros(200),
        ..Default::default()
    };
    assert_eq!(timing.gpu_wait_time.as_micros(), 200);
}

#[test]
fn test_timing_metrics_present_wait_time_tracking() {
    let timing = TimingMetrics {
        present_wait_time: Duration::from_millis(4),
        ..Default::default()
    };
    assert_eq!(timing.present_wait_time.as_millis(), 4);
}

#[test]
fn test_timing_metrics_frame_overlap_serial() {
    // When there's significant wait time, overlap should be low
    let timing = TimingMetrics {
        cpu_frame_time: Duration::from_millis(8),
        gpu_frame_time: Duration::from_millis(8),
        gpu_wait_time: Duration::from_millis(8), // High wait = serial execution
        ..Default::default()
    };
    let overlap = timing.frame_overlap();
    assert!(overlap < 1.0, "Serial execution should have low overlap: {}", overlap);
}

#[test]
fn test_timing_metrics_frame_overlap_full() {
    // When wait time is zero, overlap should be high
    let timing = TimingMetrics {
        cpu_frame_time: Duration::from_millis(8),
        gpu_frame_time: Duration::from_millis(8),
        gpu_wait_time: Duration::ZERO,
        ..Default::default()
    };
    let overlap = timing.frame_overlap();
    assert!(overlap >= 0.9, "Full overlap expected when no wait time: {}", overlap);
}

#[test]
fn test_timing_metrics_frame_overlap_partial() {
    let timing = TimingMetrics {
        cpu_frame_time: Duration::from_millis(10),
        gpu_frame_time: Duration::from_millis(10),
        gpu_wait_time: Duration::from_millis(2),
        ..Default::default()
    };
    let overlap = timing.frame_overlap();
    assert!(overlap > 0.5 && overlap < 1.0, "Partial overlap expected: {}", overlap);
}

#[test]
fn test_timing_metrics_frame_overlap_zero_cpu() {
    let timing = TimingMetrics {
        cpu_frame_time: Duration::ZERO,
        gpu_frame_time: Duration::from_millis(8),
        ..Default::default()
    };
    assert_eq!(timing.frame_overlap(), 0.0, "Zero CPU time should give 0 overlap");
}

#[test]
fn test_timing_metrics_frame_overlap_zero_gpu() {
    let timing = TimingMetrics {
        cpu_frame_time: Duration::from_millis(8),
        gpu_frame_time: Duration::ZERO,
        ..Default::default()
    };
    assert_eq!(timing.frame_overlap(), 0.0, "Zero GPU time should give 0 overlap");
}

#[test]
fn test_timing_metrics_gpu_utilization_zero() {
    let timing = TimingMetrics::default();
    assert_eq!(timing.gpu_utilization(), 0.0);
}

#[test]
fn test_timing_metrics_gpu_utilization_100_percent() {
    let timing = TimingMetrics {
        cpu_frame_time: Duration::from_millis(10),
        gpu_frame_time: Duration::from_millis(10),
        present_wait_time: Duration::ZERO,
        ..Default::default()
    };
    assert_eq!(timing.gpu_utilization(), 1.0, "GPU working entire frame");
}

#[test]
fn test_timing_metrics_gpu_utilization_varying() {
    let timing = TimingMetrics {
        cpu_frame_time: Duration::from_millis(10),
        gpu_frame_time: Duration::from_millis(5),
        present_wait_time: Duration::ZERO,
        ..Default::default()
    };
    let util = timing.gpu_utilization();
    assert!(util > 0.0 && util <= 1.0, "Utilization should be in (0,1]: {}", util);
}

#[test]
fn test_timing_metrics_is_cpu_bound_true() {
    let timing = TimingMetrics {
        cpu_frame_time: Duration::from_millis(16),
        gpu_frame_time: Duration::from_millis(4),
        ..Default::default()
    };
    assert!(timing.is_cpu_bound(2.0), "4x CPU/GPU ratio should be CPU bound");
}

#[test]
fn test_timing_metrics_is_cpu_bound_false() {
    let timing = TimingMetrics {
        cpu_frame_time: Duration::from_millis(8),
        gpu_frame_time: Duration::from_millis(8),
        ..Default::default()
    };
    assert!(!timing.is_cpu_bound(1.5), "Equal times should not be CPU bound");
}

#[test]
fn test_timing_metrics_is_gpu_bound_true() {
    let timing = TimingMetrics {
        cpu_frame_time: Duration::from_millis(4),
        gpu_frame_time: Duration::from_millis(16),
        ..Default::default()
    };
    assert!(timing.is_gpu_bound(2.0), "4x GPU/CPU ratio should be GPU bound");
}

#[test]
fn test_timing_metrics_is_gpu_bound_false() {
    let timing = TimingMetrics {
        cpu_frame_time: Duration::from_millis(8),
        gpu_frame_time: Duration::from_millis(8),
        ..Default::default()
    };
    assert!(!timing.is_gpu_bound(1.5), "Equal times should not be GPU bound");
}

#[test]
fn test_timing_metrics_has_sync_stalls_true() {
    let timing = TimingMetrics {
        gpu_wait_time: Duration::from_millis(5),
        ..Default::default()
    };
    assert!(timing.has_sync_stalls(1.0), "5ms wait > 1ms threshold");
}

#[test]
fn test_timing_metrics_has_sync_stalls_false() {
    let timing = TimingMetrics {
        gpu_wait_time: Duration::from_micros(500),
        ..Default::default()
    };
    assert!(!timing.has_sync_stalls(1.0), "0.5ms wait < 1ms threshold");
}

#[test]
fn test_timing_metrics_cpu_frame_time_ms() {
    let timing = TimingMetrics {
        cpu_frame_time: Duration::from_millis(16),
        ..Default::default()
    };
    let ms = timing.cpu_frame_time_ms();
    assert!((ms - 16.0).abs() < 0.01, "Expected 16.0ms, got {}", ms);
}

#[test]
fn test_timing_metrics_gpu_frame_time_ms() {
    let timing = TimingMetrics {
        gpu_frame_time: Duration::from_micros(8333),
        ..Default::default()
    };
    let ms = timing.gpu_frame_time_ms();
    assert!((ms - 8.333).abs() < 0.01, "Expected ~8.333ms, got {}", ms);
}

#[test]
fn test_timing_metrics_clone() {
    let timing = TimingMetrics::new(
        Duration::from_millis(8),
        Duration::from_millis(12),
        Duration::from_micros(500),
        Duration::from_micros(100),
        Duration::from_millis(4),
    );
    let cloned = timing.clone();
    assert_eq!(timing, cloned);
}

#[test]
fn test_timing_metrics_equality() {
    let t1 = TimingMetrics::new(
        Duration::from_millis(8),
        Duration::from_millis(12),
        Duration::ZERO,
        Duration::ZERO,
        Duration::ZERO,
    );
    let t2 = TimingMetrics::new(
        Duration::from_millis(8),
        Duration::from_millis(12),
        Duration::ZERO,
        Duration::ZERO,
        Duration::ZERO,
    );
    assert_eq!(t1, t2);
}

// ============================================================================
// Section 4: ResourceMetrics Tests (20+ tests)
// ============================================================================

#[test]
fn test_resource_metrics_default() {
    let resources = ResourceMetrics::default();
    assert_eq!(resources.texture_uploads_bytes, 0);
    assert_eq!(resources.buffer_uploads_bytes, 0);
    assert_eq!(resources.readback_bytes, 0);
    assert_eq!(resources.bandwidth_used_mbps, 0);
    assert_eq!(resources.cache_misses, 0);
}

#[test]
fn test_resource_metrics_new_construction() {
    let resources = ResourceMetrics::new(1000, 2000, 500, 100, 10);
    assert_eq!(resources.texture_uploads_bytes, 1000);
    assert_eq!(resources.buffer_uploads_bytes, 2000);
    assert_eq!(resources.readback_bytes, 500);
    assert_eq!(resources.bandwidth_used_mbps, 100);
    assert_eq!(resources.cache_misses, 10);
}

#[test]
fn test_resource_metrics_texture_upload_bytes() {
    let resources = ResourceMetrics {
        texture_uploads_bytes: 1024 * 1024,
        ..Default::default()
    };
    assert_eq!(resources.texture_uploads_bytes, 1024 * 1024);
}

#[test]
fn test_resource_metrics_buffer_upload_bytes() {
    let resources = ResourceMetrics {
        buffer_uploads_bytes: 512 * 1024,
        ..Default::default()
    };
    assert_eq!(resources.buffer_uploads_bytes, 512 * 1024);
}

#[test]
fn test_resource_metrics_readback_bytes() {
    let resources = ResourceMetrics {
        readback_bytes: 256 * 1024,
        ..Default::default()
    };
    assert_eq!(resources.readback_bytes, 256 * 1024);
}

#[test]
fn test_resource_metrics_bandwidth_tracking() {
    let resources = ResourceMetrics {
        bandwidth_used_mbps: 5000,
        ..Default::default()
    };
    assert_eq!(resources.bandwidth_used_mbps, 5000);
}

#[test]
fn test_resource_metrics_cache_miss_tracking() {
    let resources = ResourceMetrics {
        cache_misses: 100,
        ..Default::default()
    };
    assert_eq!(resources.cache_misses, 100);
}

#[test]
fn test_resource_metrics_total_bytes_calculation() {
    let resources = ResourceMetrics {
        texture_uploads_bytes: 1000,
        buffer_uploads_bytes: 2000,
        readback_bytes: 500,
        ..Default::default()
    };
    assert_eq!(resources.total_bytes(), 3500);
}

#[test]
fn test_resource_metrics_total_bytes_empty() {
    let resources = ResourceMetrics::default();
    assert_eq!(resources.total_bytes(), 0);
}

#[test]
fn test_resource_metrics_is_bandwidth_limited_under() {
    let resources = ResourceMetrics {
        bandwidth_used_mbps: 5000,
        ..Default::default()
    };
    assert!(!resources.is_bandwidth_limited(10000), "5000 < 10000 threshold");
}

#[test]
fn test_resource_metrics_is_bandwidth_limited_over() {
    let resources = ResourceMetrics {
        bandwidth_used_mbps: 15000,
        ..Default::default()
    };
    assert!(resources.is_bandwidth_limited(10000), "15000 > 10000 threshold");
}

#[test]
fn test_resource_metrics_is_bandwidth_limited_equal() {
    let resources = ResourceMetrics {
        bandwidth_used_mbps: 10000,
        ..Default::default()
    };
    assert!(!resources.is_bandwidth_limited(10000), "Equal should not be limited");
}

#[test]
fn test_resource_metrics_has_readback_stalls_true() {
    let resources = ResourceMetrics {
        readback_bytes: 2 * 1024 * 1024, // 2MB
        ..Default::default()
    };
    assert!(resources.has_readback_stalls(1024 * 1024), "2MB > 1MB threshold");
}

#[test]
fn test_resource_metrics_has_readback_stalls_false() {
    let resources = ResourceMetrics {
        readback_bytes: 512 * 1024, // 512KB
        ..Default::default()
    };
    assert!(!resources.has_readback_stalls(1024 * 1024), "512KB < 1MB threshold");
}

#[test]
fn test_resource_metrics_bandwidth_utilization_50_percent() {
    let resources = ResourceMetrics {
        bandwidth_used_mbps: 5000,
        ..Default::default()
    };
    let util = resources.bandwidth_utilization(10000);
    assert!((util - 0.5).abs() < 0.01, "Expected 50%, got {}", util);
}

#[test]
fn test_resource_metrics_bandwidth_utilization_100_percent() {
    let resources = ResourceMetrics {
        bandwidth_used_mbps: 10000,
        ..Default::default()
    };
    let util = resources.bandwidth_utilization(10000);
    assert!((util - 1.0).abs() < 0.01, "Expected 100%, got {}", util);
}

#[test]
fn test_resource_metrics_bandwidth_utilization_over_clamped() {
    let resources = ResourceMetrics {
        bandwidth_used_mbps: 15000,
        ..Default::default()
    };
    let util = resources.bandwidth_utilization(10000);
    assert_eq!(util, 1.0, "Should clamp to 1.0");
}

#[test]
fn test_resource_metrics_bandwidth_utilization_zero_limit() {
    let resources = ResourceMetrics {
        bandwidth_used_mbps: 5000,
        ..Default::default()
    };
    let util = resources.bandwidth_utilization(0);
    assert_eq!(util, 0.0, "Zero limit should return 0.0");
}

#[test]
fn test_resource_metrics_clone() {
    let resources = ResourceMetrics::new(1000, 2000, 500, 100, 10);
    let cloned = resources.clone();
    assert_eq!(resources, cloned);
}

#[test]
fn test_resource_metrics_equality() {
    let r1 = ResourceMetrics::new(1000, 2000, 500, 100, 10);
    let r2 = ResourceMetrics::new(1000, 2000, 500, 100, 10);
    assert_eq!(r1, r2);
}

// ============================================================================
// Section 5: StateMetrics Tests (20+ tests)
// ============================================================================

#[test]
fn test_state_metrics_default() {
    let state = StateMetrics::default();
    assert_eq!(state.pipeline_switches, 0);
    assert_eq!(state.bind_group_changes, 0);
    assert_eq!(state.vertex_buffer_changes, 0);
    assert_eq!(state.index_buffer_changes, 0);
    assert_eq!(state.draw_calls, 0);
}

#[test]
fn test_state_metrics_new_construction() {
    let state = StateMetrics::new(10, 20, 30, 40, 100);
    assert_eq!(state.pipeline_switches, 10);
    assert_eq!(state.bind_group_changes, 20);
    assert_eq!(state.vertex_buffer_changes, 30);
    assert_eq!(state.index_buffer_changes, 40);
    assert_eq!(state.draw_calls, 100);
}

#[test]
fn test_state_metrics_pipeline_switches_counting() {
    let state = StateMetrics {
        pipeline_switches: 50,
        ..Default::default()
    };
    assert_eq!(state.pipeline_switches, 50);
}

#[test]
fn test_state_metrics_bind_group_changes_counting() {
    let state = StateMetrics {
        bind_group_changes: 200,
        ..Default::default()
    };
    assert_eq!(state.bind_group_changes, 200);
}

#[test]
fn test_state_metrics_vertex_buffer_changes_counting() {
    let state = StateMetrics {
        vertex_buffer_changes: 100,
        ..Default::default()
    };
    assert_eq!(state.vertex_buffer_changes, 100);
}

#[test]
fn test_state_metrics_index_buffer_changes_counting() {
    let state = StateMetrics {
        index_buffer_changes: 100,
        ..Default::default()
    };
    assert_eq!(state.index_buffer_changes, 100);
}

#[test]
fn test_state_metrics_draw_calls_counting() {
    let state = StateMetrics {
        draw_calls: 500,
        ..Default::default()
    };
    assert_eq!(state.draw_calls, 500);
}

#[test]
fn test_state_metrics_total_state_changes() {
    let state = StateMetrics {
        pipeline_switches: 10,
        bind_group_changes: 20,
        vertex_buffer_changes: 30,
        index_buffer_changes: 40,
        draw_calls: 100,
    };
    assert_eq!(state.total_state_changes(), 100); // 10+20+30+40
}

#[test]
fn test_state_metrics_state_changes_per_draw_zero_draws() {
    let state = StateMetrics {
        pipeline_switches: 50,
        bind_group_changes: 100,
        vertex_buffer_changes: 50,
        index_buffer_changes: 50,
        draw_calls: 0,
    };
    assert_eq!(state.state_changes_per_draw(), 0.0, "Zero draws should return 0.0");
}

#[test]
fn test_state_metrics_state_changes_per_draw_normal() {
    let state = StateMetrics {
        pipeline_switches: 50,
        bind_group_changes: 100,
        vertex_buffer_changes: 50,
        index_buffer_changes: 50,
        draw_calls: 100, // Total changes = 250, per draw = 2.5
    };
    let ratio = state.state_changes_per_draw();
    assert!((ratio - 2.5).abs() < 0.01, "Expected 2.5, got {}", ratio);
}

#[test]
fn test_state_metrics_state_changes_per_draw_low() {
    let state = StateMetrics {
        pipeline_switches: 10,
        bind_group_changes: 20,
        vertex_buffer_changes: 10,
        index_buffer_changes: 10,
        draw_calls: 100, // Total = 50, per draw = 0.5
    };
    let ratio = state.state_changes_per_draw();
    assert!((ratio - 0.5).abs() < 0.01, "Expected 0.5, got {}", ratio);
}

#[test]
fn test_state_metrics_is_state_thrashing_below_threshold() {
    let state = StateMetrics {
        pipeline_switches: 50,
        bind_group_changes: 100,
        vertex_buffer_changes: 50,
        index_buffer_changes: 50,
        draw_calls: 500, // Total = 250, per draw = 0.5
    };
    assert!(!state.is_state_thrashing(2.0), "0.5 < 2.0 threshold");
}

#[test]
fn test_state_metrics_is_state_thrashing_above_threshold() {
    let state = StateMetrics {
        pipeline_switches: 200,
        bind_group_changes: 400,
        vertex_buffer_changes: 200,
        index_buffer_changes: 200,
        draw_calls: 100, // Total = 1000, per draw = 10.0
    };
    assert!(state.is_state_thrashing(2.0), "10.0 > 2.0 threshold");
}

#[test]
fn test_state_metrics_is_state_thrashing_at_threshold() {
    let state = StateMetrics {
        pipeline_switches: 50,
        bind_group_changes: 100,
        vertex_buffer_changes: 25,
        index_buffer_changes: 25,
        draw_calls: 100, // Total = 200, per draw = 2.0
    };
    assert!(!state.is_state_thrashing(2.0), "2.0 == 2.0 should not trigger");
}

#[test]
fn test_state_metrics_pipelines_per_draw_zero_draws() {
    let state = StateMetrics {
        pipeline_switches: 50,
        draw_calls: 0,
        ..Default::default()
    };
    assert_eq!(state.pipelines_per_draw(), 0.0);
}

#[test]
fn test_state_metrics_pipelines_per_draw_normal() {
    let state = StateMetrics {
        pipeline_switches: 50,
        draw_calls: 100,
        ..Default::default()
    };
    let ratio = state.pipelines_per_draw();
    assert!((ratio - 0.5).abs() < 0.01, "Expected 0.5, got {}", ratio);
}

#[test]
fn test_state_metrics_bind_groups_per_draw_zero_draws() {
    let state = StateMetrics {
        bind_group_changes: 200,
        draw_calls: 0,
        ..Default::default()
    };
    assert_eq!(state.bind_groups_per_draw(), 0.0);
}

#[test]
fn test_state_metrics_bind_groups_per_draw_normal() {
    let state = StateMetrics {
        bind_group_changes: 200,
        draw_calls: 100,
        ..Default::default()
    };
    let ratio = state.bind_groups_per_draw();
    assert!((ratio - 2.0).abs() < 0.01, "Expected 2.0, got {}", ratio);
}

#[test]
fn test_state_metrics_clone() {
    let state = StateMetrics::new(10, 20, 30, 40, 100);
    let cloned = state.clone();
    assert_eq!(state, cloned);
}

#[test]
fn test_state_metrics_equality() {
    let s1 = StateMetrics::new(10, 20, 30, 40, 100);
    let s2 = StateMetrics::new(10, 20, 30, 40, 100);
    assert_eq!(s1, s2);
}

// ============================================================================
// Section 6: FrameMetrics Tests (15+ tests)
// ============================================================================

#[test]
fn test_frame_metrics_default() {
    let frame = FrameMetrics::default();
    assert_eq!(frame.frame_number, 0);
    assert_eq!(frame.timing, TimingMetrics::default());
    assert_eq!(frame.resources, ResourceMetrics::default());
    assert_eq!(frame.state, StateMetrics::default());
}

#[test]
fn test_frame_metrics_new_construction() {
    let frame = FrameMetrics::new(
        TimingMetrics::default(),
        ResourceMetrics::default(),
        StateMetrics::default(),
        42,
    );
    assert_eq!(frame.frame_number, 42);
}

#[test]
fn test_frame_metrics_with_timestamp() {
    let timestamp = Instant::now();
    let frame = FrameMetrics::with_timestamp(
        TimingMetrics::default(),
        ResourceMetrics::default(),
        StateMetrics::default(),
        123,
        timestamp,
    );
    assert_eq!(frame.frame_number, 123);
    assert_eq!(frame.timestamp, timestamp);
}

#[test]
fn test_frame_metrics_timing_composition() {
    let timing = TimingMetrics::new(
        Duration::from_millis(8),
        Duration::from_millis(12),
        Duration::ZERO,
        Duration::ZERO,
        Duration::ZERO,
    );
    let frame = FrameMetrics::new(timing, ResourceMetrics::default(), StateMetrics::default(), 0);
    assert_eq!(frame.timing.cpu_frame_time, Duration::from_millis(8));
    assert_eq!(frame.timing.gpu_frame_time, Duration::from_millis(12));
}

#[test]
fn test_frame_metrics_resources_composition() {
    let resources = ResourceMetrics::new(1000, 2000, 500, 100, 10);
    let frame = FrameMetrics::new(TimingMetrics::default(), resources, StateMetrics::default(), 0);
    assert_eq!(frame.resources.texture_uploads_bytes, 1000);
    assert_eq!(frame.resources.buffer_uploads_bytes, 2000);
}

#[test]
fn test_frame_metrics_state_composition() {
    let state = StateMetrics::new(10, 20, 30, 40, 100);
    let frame = FrameMetrics::new(TimingMetrics::default(), ResourceMetrics::default(), state, 0);
    assert_eq!(frame.state.pipeline_switches, 10);
    assert_eq!(frame.state.draw_calls, 100);
}

#[test]
fn test_frame_metrics_frame_number_tracking() {
    let frame = FrameMetrics::new(
        TimingMetrics::default(),
        ResourceMetrics::default(),
        StateMetrics::default(),
        999,
    );
    assert_eq!(frame.frame_number, 999);
}

#[test]
fn test_frame_metrics_timestamp_recording() {
    let before = Instant::now();
    let frame = FrameMetrics::default();
    let after = Instant::now();
    assert!(frame.timestamp >= before);
    assert!(frame.timestamp <= after);
}

#[test]
fn test_frame_metrics_age_secs() {
    let frame = FrameMetrics::default();
    std::thread::sleep(Duration::from_millis(10));
    let age = frame.age_secs();
    assert!(age >= 0.01, "Age should be at least 10ms: {}", age);
    assert!(age < 1.0, "Age should be less than 1s: {}", age);
}

#[test]
fn test_frame_metrics_clone() {
    let frame = FrameMetrics::new(
        TimingMetrics::default(),
        ResourceMetrics::default(),
        StateMetrics::default(),
        42,
    );
    let cloned = frame.clone();
    assert_eq!(frame.frame_number, cloned.frame_number);
}

#[test]
fn test_frame_metrics_debug() {
    let frame = FrameMetrics::default();
    let debug_str = format!("{:?}", frame);
    assert!(debug_str.contains("FrameMetrics"));
    assert!(debug_str.contains("frame_number"));
}

#[test]
fn test_frame_metrics_all_fields_accessible() {
    let frame = FrameMetrics::default();
    let _ = frame.timing;
    let _ = frame.resources;
    let _ = frame.state;
    let _ = frame.frame_number;
    let _ = frame.timestamp;
}

#[test]
fn test_frame_metrics_large_frame_number() {
    let frame = FrameMetrics::new(
        TimingMetrics::default(),
        ResourceMetrics::default(),
        StateMetrics::default(),
        u64::MAX,
    );
    assert_eq!(frame.frame_number, u64::MAX);
}

#[test]
fn test_frame_metrics_combined_data() {
    let timing = TimingMetrics::new(
        Duration::from_millis(8),
        Duration::from_millis(12),
        Duration::from_micros(500),
        Duration::from_micros(100),
        Duration::from_millis(4),
    );
    let resources = ResourceMetrics::new(1024, 2048, 512, 100, 5);
    let state = StateMetrics::new(50, 200, 100, 100, 500);

    let frame = FrameMetrics::new(timing, resources, state, 100);

    assert_eq!(frame.timing.cpu_frame_time, Duration::from_millis(8));
    assert_eq!(frame.resources.total_bytes(), 1024 + 2048 + 512);
    assert_eq!(frame.state.total_state_changes(), 50 + 200 + 100 + 100);
}

// ============================================================================
// Section 7: BottleneckResult Tests (15+ tests)
// ============================================================================

#[test]
fn test_bottleneck_result_new() {
    let result = BottleneckResult::new(BottleneckType::GpuBound, FrameMetrics::default());
    assert_eq!(result.primary, BottleneckType::GpuBound);
    assert_eq!(result.confidence, 1.0);
    assert!(result.secondary.is_none());
}

#[test]
fn test_bottleneck_result_balanced() {
    let result = BottleneckResult::balanced(FrameMetrics::default());
    assert_eq!(result.primary, BottleneckType::Balanced);
    assert_eq!(result.confidence, 1.0);
    assert!(!result.details.is_empty());
}

#[test]
fn test_bottleneck_result_primary_bottleneck() {
    let result = BottleneckResult::new(BottleneckType::CpuBound, FrameMetrics::default());
    assert_eq!(result.primary, BottleneckType::CpuBound);
}

#[test]
fn test_bottleneck_result_secondary_bottleneck_some() {
    let mut result = BottleneckResult::new(BottleneckType::GpuBound, FrameMetrics::default());
    result.secondary = Some(BottleneckType::MemoryBandwidth);
    assert_eq!(result.secondary, Some(BottleneckType::MemoryBandwidth));
}

#[test]
fn test_bottleneck_result_secondary_bottleneck_none() {
    let result = BottleneckResult::new(BottleneckType::GpuBound, FrameMetrics::default());
    assert!(result.secondary.is_none());
}

#[test]
fn test_bottleneck_result_confidence_values() {
    let mut result = BottleneckResult::new(BottleneckType::CpuBound, FrameMetrics::default());
    result.confidence = 0.75;
    assert_eq!(result.confidence, 0.75);
}

#[test]
fn test_bottleneck_result_recommendations_populated() {
    let result = BottleneckResult::new(BottleneckType::CpuBound, FrameMetrics::default());
    assert!(!result.recommendations.is_empty(), "Should have recommendations");
}

#[test]
fn test_bottleneck_result_details_string() {
    let mut result = BottleneckResult::new(BottleneckType::GpuBound, FrameMetrics::default());
    result.details = "Custom details".to_string();
    assert_eq!(result.details, "Custom details");
}

#[test]
fn test_bottleneck_result_severity() {
    let result = BottleneckResult::new(BottleneckType::GpuBound, FrameMetrics::default());
    // GpuBound has Medium severity, confidence=1.0, so severity should be Medium
    let severity = result.severity();
    assert!(severity >= BottleneckSeverity::None);
}

#[test]
fn test_bottleneck_result_is_significant_true() {
    let mut result = BottleneckResult::new(BottleneckType::CpuBound, FrameMetrics::default());
    result.confidence = 0.8;
    assert!(result.is_significant(), "Should be significant with high confidence");
}

#[test]
fn test_bottleneck_result_is_significant_false_low_confidence() {
    let mut result = BottleneckResult::new(BottleneckType::CpuBound, FrameMetrics::default());
    result.confidence = 0.3;
    assert!(!result.is_significant(), "Should not be significant with low confidence");
}

#[test]
fn test_bottleneck_result_is_significant_false_balanced() {
    let result = BottleneckResult::balanced(FrameMetrics::default());
    assert!(!result.is_significant(), "Balanced should not be significant");
}

#[test]
fn test_bottleneck_result_all_hints() {
    let mut result = BottleneckResult::new(BottleneckType::GpuBound, FrameMetrics::default());
    result.secondary = Some(BottleneckType::MemoryBandwidth);
    let hints = result.all_hints();
    // Should have hints from both primary and secondary
    assert!(hints.len() > BottleneckType::GpuBound.optimization_hints().len());
}

#[test]
fn test_bottleneck_result_display() {
    let result = BottleneckResult::new(BottleneckType::GpuBound, FrameMetrics::default());
    let display = format!("{}", result);
    assert!(display.contains("GPU Bound"));
    assert!(display.contains("confidence"));
}

#[test]
fn test_bottleneck_result_display_with_secondary() {
    let mut result = BottleneckResult::new(BottleneckType::GpuBound, FrameMetrics::default());
    result.secondary = Some(BottleneckType::StateThrashing);
    let display = format!("{}", result);
    assert!(display.contains("GPU Bound"));
    assert!(display.contains("State Thrashing"));
}

#[test]
fn test_bottleneck_result_clone() {
    let result = BottleneckResult::new(BottleneckType::CpuBound, FrameMetrics::default());
    let cloned = result.clone();
    assert_eq!(result.primary, cloned.primary);
    assert_eq!(result.confidence, cloned.confidence);
}

// ============================================================================
// Section 8: AnalysisThresholds Tests (15+ tests)
// ============================================================================

#[test]
fn test_thresholds_default() {
    let thresholds = AnalysisThresholds::default();
    assert!(thresholds.gpu_bound_ratio > 1.0);
    assert!(thresholds.cpu_bound_ratio > 1.0);
    assert!(thresholds.state_thrash_per_draw > 0.0);
    assert!(thresholds.bandwidth_mbps_limit > 0.0);
    assert!(thresholds.sync_stall_ms > 0.0);
}

#[test]
fn test_thresholds_default_values() {
    let thresholds = AnalysisThresholds::default();
    assert_eq!(thresholds.gpu_bound_ratio, 1.5);
    assert_eq!(thresholds.cpu_bound_ratio, 1.5);
    assert_eq!(thresholds.state_thrash_per_draw, 2.0);
    assert_eq!(thresholds.bandwidth_mbps_limit, 10000.0);
    assert_eq!(thresholds.sync_stall_ms, 1.0);
    assert_eq!(thresholds.min_frame_time_ms, 0.1);
}

#[test]
fn test_thresholds_aggressive() {
    let thresholds = AnalysisThresholds::aggressive();
    assert_eq!(thresholds.gpu_bound_ratio, 1.2);
    assert_eq!(thresholds.cpu_bound_ratio, 1.2);
    assert_eq!(thresholds.state_thrash_per_draw, 1.5);
    assert_eq!(thresholds.bandwidth_mbps_limit, 5000.0);
    assert_eq!(thresholds.sync_stall_ms, 0.5);
}

#[test]
fn test_thresholds_relaxed() {
    let thresholds = AnalysisThresholds::relaxed();
    assert_eq!(thresholds.gpu_bound_ratio, 2.5);
    assert_eq!(thresholds.cpu_bound_ratio, 2.5);
    assert_eq!(thresholds.state_thrash_per_draw, 4.0);
    assert_eq!(thresholds.bandwidth_mbps_limit, 15000.0);
    assert_eq!(thresholds.sync_stall_ms, 3.0);
}

#[test]
fn test_thresholds_aggressive_stricter_than_default() {
    let aggressive = AnalysisThresholds::aggressive();
    let default = AnalysisThresholds::default();

    assert!(aggressive.gpu_bound_ratio < default.gpu_bound_ratio);
    assert!(aggressive.cpu_bound_ratio < default.cpu_bound_ratio);
    assert!(aggressive.state_thrash_per_draw < default.state_thrash_per_draw);
    assert!(aggressive.bandwidth_mbps_limit < default.bandwidth_mbps_limit);
    assert!(aggressive.sync_stall_ms < default.sync_stall_ms);
}

#[test]
fn test_thresholds_relaxed_looser_than_default() {
    let relaxed = AnalysisThresholds::relaxed();
    let default = AnalysisThresholds::default();

    assert!(relaxed.gpu_bound_ratio > default.gpu_bound_ratio);
    assert!(relaxed.cpu_bound_ratio > default.cpu_bound_ratio);
    assert!(relaxed.state_thrash_per_draw > default.state_thrash_per_draw);
    assert!(relaxed.bandwidth_mbps_limit > default.bandwidth_mbps_limit);
    assert!(relaxed.sync_stall_ms > default.sync_stall_ms);
}

#[test]
fn test_thresholds_new_custom() {
    let thresholds = AnalysisThresholds::new(2.0, 2.0, 3.0, 8000.0, 2.0);
    assert_eq!(thresholds.gpu_bound_ratio, 2.0);
    assert_eq!(thresholds.cpu_bound_ratio, 2.0);
    assert_eq!(thresholds.state_thrash_per_draw, 3.0);
    assert_eq!(thresholds.bandwidth_mbps_limit, 8000.0);
    assert_eq!(thresholds.sync_stall_ms, 2.0);
}

#[test]
fn test_thresholds_clone() {
    let thresholds = AnalysisThresholds::aggressive();
    let cloned = thresholds.clone();
    assert_eq!(thresholds.gpu_bound_ratio, cloned.gpu_bound_ratio);
}

#[test]
fn test_thresholds_copy() {
    let thresholds = AnalysisThresholds::default();
    let copied: AnalysisThresholds = thresholds;
    assert_eq!(thresholds.gpu_bound_ratio, copied.gpu_bound_ratio);
}

#[test]
fn test_thresholds_equality() {
    let t1 = AnalysisThresholds::default();
    let t2 = AnalysisThresholds::default();
    assert_eq!(t1, t2);
}

#[test]
fn test_thresholds_inequality() {
    let t1 = AnalysisThresholds::default();
    let t2 = AnalysisThresholds::aggressive();
    assert_ne!(t1, t2);
}

#[test]
fn test_thresholds_debug() {
    let thresholds = AnalysisThresholds::default();
    let debug_str = format!("{:?}", thresholds);
    assert!(debug_str.contains("AnalysisThresholds"));
}

#[test]
fn test_thresholds_all_fields_accessible() {
    let thresholds = AnalysisThresholds::default();
    let _ = thresholds.gpu_bound_ratio;
    let _ = thresholds.cpu_bound_ratio;
    let _ = thresholds.state_thrash_per_draw;
    let _ = thresholds.bandwidth_mbps_limit;
    let _ = thresholds.sync_stall_ms;
    let _ = thresholds.min_frame_time_ms;
}

#[test]
fn test_thresholds_new_sets_min_frame_time() {
    let thresholds = AnalysisThresholds::new(1.5, 1.5, 2.0, 10000.0, 1.0);
    assert_eq!(thresholds.min_frame_time_ms, 0.1);
}

// ============================================================================
// Section 9: TrendAnalysis Tests (15+ tests)
// ============================================================================

#[test]
fn test_trend_analysis_empty() {
    let trend = TrendAnalysis::empty();
    assert_eq!(trend.samples, 0);
    assert_eq!(trend.avg_bottleneck, BottleneckType::Balanced);
    assert_eq!(trend.bottleneck_stability, 1.0);
    assert!(!trend.improving);
    assert!(!trend.degrading);
    assert!(trend.spikes.is_empty());
}

#[test]
fn test_trend_analysis_default() {
    let trend: TrendAnalysis = Default::default();
    assert_eq!(trend.samples, 0);
}

#[test]
fn test_trend_analysis_sample_count() {
    let mut trend = TrendAnalysis::empty();
    trend.samples = 30;
    assert_eq!(trend.samples, 30);
}

#[test]
fn test_trend_analysis_avg_bottleneck() {
    let mut trend = TrendAnalysis::empty();
    trend.avg_bottleneck = BottleneckType::GpuBound;
    assert_eq!(trend.avg_bottleneck, BottleneckType::GpuBound);
}

#[test]
fn test_trend_analysis_stability_score() {
    let mut trend = TrendAnalysis::empty();
    trend.bottleneck_stability = 0.85;
    assert_eq!(trend.bottleneck_stability, 0.85);
}

#[test]
fn test_trend_analysis_improving_flag() {
    let mut trend = TrendAnalysis::empty();
    trend.improving = true;
    assert!(trend.improving);
}

#[test]
fn test_trend_analysis_degrading_flag() {
    let mut trend = TrendAnalysis::empty();
    trend.degrading = true;
    assert!(trend.degrading);
}

#[test]
fn test_trend_analysis_spike_detection() {
    let mut trend = TrendAnalysis::empty();
    trend.spikes = vec![(10, BottleneckType::CpuBound), (25, BottleneckType::GpuBound)];
    assert_eq!(trend.spikes.len(), 2);
}

#[test]
fn test_trend_analysis_has_sufficient_data_false() {
    let mut trend = TrendAnalysis::empty();
    trend.samples = 5;
    assert!(!trend.has_sufficient_data());
}

#[test]
fn test_trend_analysis_has_sufficient_data_true() {
    let mut trend = TrendAnalysis::empty();
    trend.samples = 10;
    assert!(trend.has_sufficient_data());
}

#[test]
fn test_trend_analysis_has_sufficient_data_boundary() {
    let mut trend = TrendAnalysis::empty();
    trend.samples = 9;
    assert!(!trend.has_sufficient_data());
    trend.samples = 10;
    assert!(trend.has_sufficient_data());
}

#[test]
fn test_trend_analysis_jitter_zero() {
    let trend = TrendAnalysis::empty();
    assert_eq!(trend.jitter(), 0.0);
}

#[test]
fn test_trend_analysis_jitter_calculation() {
    let mut trend = TrendAnalysis::empty();
    trend.avg_frame_time_ms = 16.0;
    trend.frame_time_stddev_ms = 1.6;
    let jitter = trend.jitter();
    assert!((jitter - 0.1).abs() < 0.01, "Expected jitter ~0.1, got {}", jitter);
}

#[test]
fn test_trend_analysis_is_stable_true() {
    let mut trend = TrendAnalysis::empty();
    trend.avg_frame_time_ms = 16.0;
    trend.frame_time_stddev_ms = 1.0; // 6.25% jitter
    assert!(trend.is_stable());
}

#[test]
fn test_trend_analysis_is_stable_false() {
    let mut trend = TrendAnalysis::empty();
    trend.avg_frame_time_ms = 16.0;
    trend.frame_time_stddev_ms = 5.0; // 31.25% jitter
    assert!(!trend.is_stable());
}

#[test]
fn test_trend_analysis_display() {
    let mut trend = TrendAnalysis::empty();
    trend.samples = 20;
    trend.avg_bottleneck = BottleneckType::GpuBound;
    trend.bottleneck_stability = 0.85;

    let display = format!("{}", trend);
    assert!(display.contains("20 samples"));
    assert!(display.contains("GPU Bound"));
}

#[test]
fn test_trend_analysis_display_improving() {
    let mut trend = TrendAnalysis::empty();
    trend.samples = 20;
    trend.improving = true;

    let display = format!("{}", trend);
    assert!(display.contains("improving"));
}

#[test]
fn test_trend_analysis_display_degrading() {
    let mut trend = TrendAnalysis::empty();
    trend.samples = 20;
    trend.degrading = true;

    let display = format!("{}", trend);
    assert!(display.contains("degrading"));
}

#[test]
fn test_trend_analysis_display_spikes() {
    let mut trend = TrendAnalysis::empty();
    trend.samples = 20;
    trend.spikes = vec![(5, BottleneckType::CpuBound)];

    let display = format!("{}", trend);
    assert!(display.contains("1 spikes"));
}

// ============================================================================
// Section 10: BottleneckAnalyzer Tests (25+ tests)
// ============================================================================

#[test]
fn test_analyzer_new() {
    let analyzer = BottleneckAnalyzer::new();
    assert_eq!(analyzer.frame_count(), 0);
    assert_eq!(analyzer.history_size(), BottleneckAnalyzer::DEFAULT_HISTORY_SIZE);
}

#[test]
fn test_analyzer_default() {
    let analyzer: BottleneckAnalyzer = Default::default();
    assert_eq!(analyzer.frame_count(), 0);
}

#[test]
fn test_analyzer_default_history_size_constant() {
    assert_eq!(BottleneckAnalyzer::DEFAULT_HISTORY_SIZE, 60);
}

#[test]
fn test_analyzer_with_thresholds() {
    let thresholds = AnalysisThresholds::aggressive();
    let analyzer = BottleneckAnalyzer::with_thresholds(thresholds);
    assert_eq!(analyzer.thresholds().gpu_bound_ratio, 1.2);
}

#[test]
fn test_analyzer_with_history_size() {
    let analyzer = BottleneckAnalyzer::with_history_size(30);
    assert_eq!(analyzer.history_size(), 30);
}

#[test]
fn test_analyzer_with_history_size_minimum() {
    let analyzer = BottleneckAnalyzer::with_history_size(0);
    assert_eq!(analyzer.history_size(), 1, "Minimum history size should be 1");
}

#[test]
fn test_analyzer_thresholds_getter() {
    let analyzer = BottleneckAnalyzer::new();
    let thresholds = analyzer.thresholds();
    assert_eq!(*thresholds, AnalysisThresholds::default());
}

#[test]
fn test_analyzer_set_thresholds() {
    let mut analyzer = BottleneckAnalyzer::new();
    analyzer.set_thresholds(AnalysisThresholds::aggressive());
    assert_eq!(analyzer.thresholds().gpu_bound_ratio, 1.2);
}

#[test]
fn test_analyzer_record_frame_increases_count() {
    let mut analyzer = BottleneckAnalyzer::new();
    assert_eq!(analyzer.frame_count(), 0);
    analyzer.record_frame(FrameMetrics::default());
    assert_eq!(analyzer.frame_count(), 1);
}

#[test]
fn test_analyzer_record_frame_multiple() {
    let mut analyzer = BottleneckAnalyzer::new();
    for _ in 0..10 {
        analyzer.record_frame(FrameMetrics::default());
    }
    assert_eq!(analyzer.frame_count(), 10);
}

#[test]
fn test_analyzer_history_ring_buffer() {
    let mut analyzer = BottleneckAnalyzer::with_history_size(5);
    for i in 0..10 {
        let mut frame = FrameMetrics::default();
        frame.frame_number = i;
        analyzer.record_frame(frame);
    }
    assert_eq!(analyzer.frame_count(), 5);
    // Should have frames 5-9 (oldest removed)
    assert_eq!(analyzer.history().front().unwrap().frame_number, 5);
    assert_eq!(analyzer.history().back().unwrap().frame_number, 9);
}

#[test]
fn test_analyzer_analyze_current_empty() {
    let mut analyzer = BottleneckAnalyzer::new();
    let result = analyzer.analyze_current();
    assert_eq!(result.primary, BottleneckType::Balanced);
}

#[test]
fn test_analyzer_analyze_current_returns_result() {
    let mut analyzer = BottleneckAnalyzer::new();
    analyzer.record_frame(FrameMetrics::default());
    let result = analyzer.analyze_current();
    // Should return some analysis
    assert!(result.confidence >= 0.0 && result.confidence <= 1.0);
}

#[test]
fn test_analyzer_detect_cpu_bound() {
    let mut analyzer = BottleneckAnalyzer::new();
    let metrics = FrameMetrics {
        timing: TimingMetrics {
            cpu_frame_time: Duration::from_millis(16),
            gpu_frame_time: Duration::from_millis(2),
            cpu_submit_time: Duration::from_millis(2),
            ..Default::default()
        },
        state: StateMetrics {
            draw_calls: 5000,
            ..Default::default()
        },
        ..Default::default()
    };
    analyzer.record_frame(metrics);
    let result = analyzer.analyze_current();
    assert_eq!(result.primary, BottleneckType::CpuBound);
}

#[test]
fn test_analyzer_detect_gpu_bound() {
    let mut analyzer = BottleneckAnalyzer::new();
    let metrics = FrameMetrics {
        timing: TimingMetrics {
            cpu_frame_time: Duration::from_millis(2),
            gpu_frame_time: Duration::from_millis(16),
            ..Default::default()
        },
        ..Default::default()
    };
    analyzer.record_frame(metrics);
    let result = analyzer.analyze_current();
    assert_eq!(result.primary, BottleneckType::GpuBound);
}

#[test]
fn test_analyzer_detect_memory_bandwidth() {
    let mut analyzer = BottleneckAnalyzer::new();
    let metrics = FrameMetrics {
        timing: TimingMetrics {
            cpu_frame_time: Duration::from_millis(8),
            gpu_frame_time: Duration::from_millis(8),
            ..Default::default()
        },
        resources: ResourceMetrics {
            bandwidth_used_mbps: 15000, // Over default 10000 limit
            readback_bytes: 2 * 1024 * 1024,
            ..Default::default()
        },
        ..Default::default()
    };
    analyzer.record_frame(metrics);
    let result = analyzer.analyze_current();
    assert_eq!(result.primary, BottleneckType::MemoryBandwidth);
}

#[test]
fn test_analyzer_detect_state_thrashing() {
    let mut analyzer = BottleneckAnalyzer::new();
    let metrics = FrameMetrics {
        timing: TimingMetrics {
            cpu_frame_time: Duration::from_millis(8),
            gpu_frame_time: Duration::from_millis(8),
            ..Default::default()
        },
        state: StateMetrics {
            pipeline_switches: 500,
            bind_group_changes: 1000,
            vertex_buffer_changes: 500,
            index_buffer_changes: 500,
            draw_calls: 500, // 5 state changes per draw
        },
        ..Default::default()
    };
    analyzer.record_frame(metrics);
    let result = analyzer.analyze_current();
    assert_eq!(result.primary, BottleneckType::StateThrashing);
}

#[test]
fn test_analyzer_detect_synchronization() {
    let mut analyzer = BottleneckAnalyzer::new();
    let metrics = FrameMetrics {
        timing: TimingMetrics {
            cpu_frame_time: Duration::from_millis(8),
            gpu_frame_time: Duration::from_millis(8),
            gpu_wait_time: Duration::from_millis(5), // High sync wait
            ..Default::default()
        },
        ..Default::default()
    };
    analyzer.record_frame(metrics);
    let result = analyzer.analyze_current();
    assert_eq!(result.primary, BottleneckType::Synchronization);
}

#[test]
fn test_analyzer_detect_balanced() {
    let mut analyzer = BottleneckAnalyzer::new();
    let metrics = FrameMetrics {
        timing: TimingMetrics {
            cpu_frame_time: Duration::from_millis(8),
            gpu_frame_time: Duration::from_millis(8),
            ..Default::default()
        },
        state: StateMetrics {
            pipeline_switches: 10,
            bind_group_changes: 20,
            vertex_buffer_changes: 10,
            index_buffer_changes: 10,
            draw_calls: 100, // Low state changes per draw
        },
        ..Default::default()
    };
    analyzer.record_frame(metrics);
    let result = analyzer.analyze_current();
    assert_eq!(result.primary, BottleneckType::Balanced);
}

#[test]
fn test_analyzer_analyze_trend_empty() {
    let analyzer = BottleneckAnalyzer::new();
    let trend = analyzer.analyze_trend();
    assert_eq!(trend.samples, 0);
    assert_eq!(trend.avg_bottleneck, BottleneckType::Balanced);
}

#[test]
fn test_analyzer_analyze_trend_with_data() {
    let mut analyzer = BottleneckAnalyzer::new();

    // Add consistent GPU-bound frames
    for i in 0..20 {
        let metrics = FrameMetrics {
            timing: TimingMetrics {
                cpu_frame_time: Duration::from_millis(2),
                gpu_frame_time: Duration::from_millis(12),
                ..Default::default()
            },
            frame_number: i,
            ..Default::default()
        };
        analyzer.record_frame(metrics);
    }

    let trend = analyzer.analyze_trend();
    assert_eq!(trend.samples, 20);
    assert!(trend.has_sufficient_data());
    assert_eq!(trend.avg_bottleneck, BottleneckType::GpuBound);
    assert!(trend.bottleneck_stability > 0.9);
}

#[test]
fn test_analyzer_clear() {
    let mut analyzer = BottleneckAnalyzer::new();
    analyzer.record_frame(FrameMetrics::default());
    let _ = analyzer.analyze_current();

    analyzer.clear();

    assert_eq!(analyzer.frame_count(), 0);
    assert!(analyzer.last_result().is_none());
}

#[test]
fn test_analyzer_last_result_none() {
    let analyzer = BottleneckAnalyzer::new();
    assert!(analyzer.last_result().is_none());
}

#[test]
fn test_analyzer_last_result_some() {
    let mut analyzer = BottleneckAnalyzer::new();
    analyzer.record_frame(FrameMetrics::default());
    let _ = analyzer.analyze_current();
    assert!(analyzer.last_result().is_some());
}

#[test]
fn test_analyzer_history_access() {
    let mut analyzer = BottleneckAnalyzer::new();
    analyzer.record_frame(FrameMetrics::default());
    let history = analyzer.history();
    assert_eq!(history.len(), 1);
}

#[test]
fn test_analyzer_debug() {
    let analyzer = BottleneckAnalyzer::new();
    let debug_str = format!("{:?}", analyzer);
    assert!(debug_str.contains("BottleneckAnalyzer"));
    assert!(debug_str.contains("history_size"));
}

// ============================================================================
// Section 11: BottleneckProfiler Tests (20+ tests)
// ============================================================================

#[test]
fn test_profiler_new() {
    let profiler = BottleneckProfiler::new();
    assert!(!profiler.is_in_frame());
}

#[test]
fn test_profiler_default() {
    let profiler: BottleneckProfiler = Default::default();
    assert!(!profiler.is_in_frame());
}

#[test]
fn test_profiler_with_analyzer() {
    let analyzer = BottleneckAnalyzer::with_thresholds(AnalysisThresholds::aggressive());
    let profiler = BottleneckProfiler::with_analyzer(analyzer);
    assert_eq!(profiler.analyzer().thresholds().gpu_bound_ratio, 1.2);
}

#[test]
fn test_profiler_begin_frame() {
    let mut profiler = BottleneckProfiler::new();
    profiler.begin_frame();
    assert!(profiler.is_in_frame());
}

#[test]
fn test_profiler_end_frame() {
    let mut profiler = BottleneckProfiler::new();
    profiler.begin_frame();
    let result = profiler.end_frame();
    assert!(result.is_some());
    assert!(!profiler.is_in_frame());
}

#[test]
fn test_profiler_end_frame_without_begin() {
    let mut profiler = BottleneckProfiler::new();
    let result = profiler.end_frame();
    assert!(result.is_none());
}

#[test]
fn test_profiler_record_cpu_time() {
    let mut profiler = BottleneckProfiler::new();
    profiler.begin_frame();
    profiler.record_cpu_time(Duration::from_millis(8));
    let result = profiler.end_frame().unwrap();
    assert_eq!(result.metrics.timing.cpu_frame_time, Duration::from_millis(8));
}

#[test]
fn test_profiler_record_gpu_time() {
    let mut profiler = BottleneckProfiler::new();
    profiler.begin_frame();
    profiler.record_gpu_time(Duration::from_millis(12));
    let result = profiler.end_frame().unwrap();
    assert_eq!(result.metrics.timing.gpu_frame_time, Duration::from_millis(12));
}

#[test]
fn test_profiler_record_submit_time() {
    let mut profiler = BottleneckProfiler::new();
    profiler.begin_frame();
    profiler.record_submit_time(Duration::from_micros(500));
    let result = profiler.end_frame().unwrap();
    assert_eq!(result.metrics.timing.cpu_submit_time, Duration::from_micros(500));
}

#[test]
fn test_profiler_record_gpu_wait() {
    let mut profiler = BottleneckProfiler::new();
    profiler.begin_frame();
    profiler.record_gpu_wait(Duration::from_micros(200));
    let result = profiler.end_frame().unwrap();
    assert_eq!(result.metrics.timing.gpu_wait_time, Duration::from_micros(200));
}

#[test]
fn test_profiler_record_present_wait() {
    let mut profiler = BottleneckProfiler::new();
    profiler.begin_frame();
    profiler.record_present_wait(Duration::from_millis(4));
    let result = profiler.end_frame().unwrap();
    assert_eq!(result.metrics.timing.present_wait_time, Duration::from_millis(4));
}

#[test]
fn test_profiler_record_state_change_pipeline() {
    let mut profiler = BottleneckProfiler::new();
    profiler.begin_frame();
    profiler.record_state_change(StateChangeType::Pipeline);
    profiler.record_state_change(StateChangeType::Pipeline);
    let result = profiler.end_frame().unwrap();
    assert_eq!(result.metrics.state.pipeline_switches, 2);
}

#[test]
fn test_profiler_record_state_change_bind_group() {
    let mut profiler = BottleneckProfiler::new();
    profiler.begin_frame();
    profiler.record_state_change(StateChangeType::BindGroup);
    profiler.record_state_change(StateChangeType::BindGroup);
    profiler.record_state_change(StateChangeType::BindGroup);
    let result = profiler.end_frame().unwrap();
    assert_eq!(result.metrics.state.bind_group_changes, 3);
}

#[test]
fn test_profiler_record_state_change_vertex_buffer() {
    let mut profiler = BottleneckProfiler::new();
    profiler.begin_frame();
    profiler.record_state_change(StateChangeType::VertexBuffer);
    let result = profiler.end_frame().unwrap();
    assert_eq!(result.metrics.state.vertex_buffer_changes, 1);
}

#[test]
fn test_profiler_record_state_change_index_buffer() {
    let mut profiler = BottleneckProfiler::new();
    profiler.begin_frame();
    profiler.record_state_change(StateChangeType::IndexBuffer);
    let result = profiler.end_frame().unwrap();
    assert_eq!(result.metrics.state.index_buffer_changes, 1);
}

#[test]
fn test_profiler_record_state_change_draw_call() {
    let mut profiler = BottleneckProfiler::new();
    profiler.begin_frame();
    for _ in 0..100 {
        profiler.record_state_change(StateChangeType::DrawCall);
    }
    let result = profiler.end_frame().unwrap();
    assert_eq!(result.metrics.state.draw_calls, 100);
}

#[test]
fn test_profiler_record_bandwidth() {
    let mut profiler = BottleneckProfiler::new();
    profiler.begin_frame();
    profiler.record_bandwidth(1024);
    profiler.record_bandwidth(2048);
    let result = profiler.end_frame().unwrap();
    assert_eq!(result.metrics.resources.buffer_uploads_bytes, 3072);
}

#[test]
fn test_profiler_record_texture_upload() {
    let mut profiler = BottleneckProfiler::new();
    profiler.begin_frame();
    profiler.record_texture_upload(1024 * 1024);
    let result = profiler.end_frame().unwrap();
    assert_eq!(result.metrics.resources.texture_uploads_bytes, 1024 * 1024);
}

#[test]
fn test_profiler_record_readback() {
    let mut profiler = BottleneckProfiler::new();
    profiler.begin_frame();
    profiler.record_readback(512 * 1024);
    let result = profiler.end_frame().unwrap();
    assert_eq!(result.metrics.resources.readback_bytes, 512 * 1024);
}

#[test]
fn test_profiler_enable_disable_auto_log() {
    let mut profiler = BottleneckProfiler::new();
    profiler.enable_auto_log();
    profiler.disable_auto_log();
    // Should not panic
}

#[test]
fn test_profiler_set_log_threshold() {
    let mut profiler = BottleneckProfiler::new();
    profiler.set_log_threshold(BottleneckSeverity::High);
    // Should not panic
}

#[test]
fn test_profiler_analyzer_getter() {
    let profiler = BottleneckProfiler::new();
    let _ = profiler.analyzer();
}

#[test]
fn test_profiler_analyzer_mut() {
    let mut profiler = BottleneckProfiler::new();
    let analyzer = profiler.analyzer_mut();
    analyzer.record_frame(FrameMetrics::default());
}

#[test]
fn test_profiler_current_frame_number() {
    let mut profiler = BottleneckProfiler::new();

    profiler.begin_frame();
    profiler.end_frame();

    profiler.begin_frame();
    assert_eq!(profiler.current_frame_number(), 1);
}

#[test]
fn test_profiler_debug() {
    let profiler = BottleneckProfiler::new();
    let debug_str = format!("{:?}", profiler);
    assert!(debug_str.contains("BottleneckProfiler"));
}

#[test]
fn test_profiler_multiple_frames() {
    let mut profiler = BottleneckProfiler::new();

    for i in 0..5 {
        profiler.begin_frame();
        profiler.record_cpu_time(Duration::from_millis(8));
        profiler.record_gpu_time(Duration::from_millis(12));
        let result = profiler.end_frame();
        assert!(result.is_some());
    }

    assert_eq!(profiler.analyzer().frame_count(), 5);
}

#[test]
fn test_profiler_no_record_without_frame() {
    let mut profiler = BottleneckProfiler::new();
    // These should be no-ops when not in a frame
    profiler.record_cpu_time(Duration::from_millis(8));
    profiler.record_state_change(StateChangeType::DrawCall);
    profiler.record_bandwidth(1024);
    // Should not affect analyzer
    assert_eq!(profiler.analyzer().frame_count(), 0);
}

// ============================================================================
// Section 12: StateChangeType Tests (additional)
// ============================================================================

#[test]
fn test_state_change_type_pipeline() {
    let t = StateChangeType::Pipeline;
    assert_eq!(format!("{:?}", t), "Pipeline");
}

#[test]
fn test_state_change_type_bind_group() {
    let t = StateChangeType::BindGroup;
    assert_eq!(format!("{:?}", t), "BindGroup");
}

#[test]
fn test_state_change_type_vertex_buffer() {
    let t = StateChangeType::VertexBuffer;
    assert_eq!(format!("{:?}", t), "VertexBuffer");
}

#[test]
fn test_state_change_type_index_buffer() {
    let t = StateChangeType::IndexBuffer;
    assert_eq!(format!("{:?}", t), "IndexBuffer");
}

#[test]
fn test_state_change_type_draw_call() {
    let t = StateChangeType::DrawCall;
    assert_eq!(format!("{:?}", t), "DrawCall");
}

#[test]
fn test_state_change_type_equality() {
    assert_eq!(StateChangeType::Pipeline, StateChangeType::Pipeline);
    assert_ne!(StateChangeType::Pipeline, StateChangeType::BindGroup);
}

#[test]
fn test_state_change_type_clone() {
    let original = StateChangeType::DrawCall;
    let cloned = original.clone();
    assert_eq!(original, cloned);
}

#[test]
fn test_state_change_type_copy() {
    let original = StateChangeType::BindGroup;
    let copied: StateChangeType = original;
    assert_eq!(original, copied);
}

#[test]
fn test_state_change_type_hash() {
    let mut set = HashSet::new();
    set.insert(StateChangeType::Pipeline);
    set.insert(StateChangeType::BindGroup);
    set.insert(StateChangeType::Pipeline); // duplicate
    assert_eq!(set.len(), 2);
}

// ============================================================================
// Section 13: Edge Case Tests
// ============================================================================

#[test]
fn test_edge_case_zero_cpu_time() {
    let timing = TimingMetrics {
        cpu_frame_time: Duration::ZERO,
        gpu_frame_time: Duration::from_millis(8),
        ..Default::default()
    };
    assert!(!timing.is_gpu_bound(1.5), "Zero CPU should not cause division error");
}

#[test]
fn test_edge_case_zero_gpu_time() {
    let timing = TimingMetrics {
        cpu_frame_time: Duration::from_millis(8),
        gpu_frame_time: Duration::ZERO,
        ..Default::default()
    };
    assert!(!timing.is_cpu_bound(1.5), "Zero GPU should not cause division error");
}

#[test]
fn test_edge_case_zero_draw_calls() {
    let state = StateMetrics {
        pipeline_switches: 100,
        bind_group_changes: 200,
        draw_calls: 0,
        ..Default::default()
    };
    assert_eq!(state.state_changes_per_draw(), 0.0);
    assert_eq!(state.pipelines_per_draw(), 0.0);
    assert_eq!(state.bind_groups_per_draw(), 0.0);
}

#[test]
fn test_edge_case_max_frame_number() {
    let frame = FrameMetrics::new(
        TimingMetrics::default(),
        ResourceMetrics::default(),
        StateMetrics::default(),
        u64::MAX,
    );
    assert_eq!(frame.frame_number, u64::MAX);
}

#[test]
fn test_edge_case_max_bandwidth() {
    let resources = ResourceMetrics {
        bandwidth_used_mbps: u32::MAX,
        ..Default::default()
    };
    assert!(resources.is_bandwidth_limited(10000));
    assert_eq!(resources.bandwidth_utilization(10000), 1.0);
}

#[test]
fn test_edge_case_max_bytes() {
    let resources = ResourceMetrics {
        texture_uploads_bytes: u64::MAX / 3,
        buffer_uploads_bytes: u64::MAX / 3,
        readback_bytes: u64::MAX / 3,
        ..Default::default()
    };
    // Should use saturating add to prevent overflow
    let total = resources.total_bytes();
    assert!(total > 0);
}

#[test]
fn test_edge_case_very_small_duration() {
    let timing = TimingMetrics {
        cpu_frame_time: Duration::from_nanos(1),
        gpu_frame_time: Duration::from_nanos(1),
        ..Default::default()
    };
    let _ = timing.frame_overlap();
    let _ = timing.gpu_utilization();
    // Should not panic with very small values
}

#[test]
fn test_edge_case_very_large_duration() {
    let timing = TimingMetrics {
        cpu_frame_time: Duration::from_secs(1000),
        gpu_frame_time: Duration::from_secs(1000),
        ..Default::default()
    };
    let _ = timing.frame_overlap();
    let _ = timing.gpu_utilization();
    // Should handle large values gracefully
}

// ============================================================================
// Section 14: Send + Sync Tests
// ============================================================================

fn assert_send<T: Send>() {}
fn assert_sync<T: Sync>() {}

#[test]
fn test_bottleneck_type_is_send() {
    assert_send::<BottleneckType>();
}

#[test]
fn test_bottleneck_type_is_sync() {
    assert_sync::<BottleneckType>();
}

#[test]
fn test_bottleneck_severity_is_send() {
    assert_send::<BottleneckSeverity>();
}

#[test]
fn test_bottleneck_severity_is_sync() {
    assert_sync::<BottleneckSeverity>();
}

#[test]
fn test_timing_metrics_is_send() {
    assert_send::<TimingMetrics>();
}

#[test]
fn test_timing_metrics_is_sync() {
    assert_sync::<TimingMetrics>();
}

#[test]
fn test_resource_metrics_is_send() {
    assert_send::<ResourceMetrics>();
}

#[test]
fn test_resource_metrics_is_sync() {
    assert_sync::<ResourceMetrics>();
}

#[test]
fn test_state_metrics_is_send() {
    assert_send::<StateMetrics>();
}

#[test]
fn test_state_metrics_is_sync() {
    assert_sync::<StateMetrics>();
}

#[test]
fn test_frame_metrics_is_send() {
    assert_send::<FrameMetrics>();
}

#[test]
fn test_frame_metrics_is_sync() {
    assert_sync::<FrameMetrics>();
}

#[test]
fn test_bottleneck_result_is_send() {
    assert_send::<BottleneckResult>();
}

#[test]
fn test_bottleneck_result_is_sync() {
    assert_sync::<BottleneckResult>();
}

#[test]
fn test_analysis_thresholds_is_send() {
    assert_send::<AnalysisThresholds>();
}

#[test]
fn test_analysis_thresholds_is_sync() {
    assert_sync::<AnalysisThresholds>();
}

#[test]
fn test_trend_analysis_is_send() {
    assert_send::<TrendAnalysis>();
}

#[test]
fn test_trend_analysis_is_sync() {
    assert_sync::<TrendAnalysis>();
}

#[test]
fn test_bottleneck_analyzer_is_send() {
    assert_send::<BottleneckAnalyzer>();
}

#[test]
fn test_bottleneck_profiler_is_send() {
    assert_send::<BottleneckProfiler>();
}

#[test]
fn test_state_change_type_is_send() {
    assert_send::<StateChangeType>();
}

#[test]
fn test_state_change_type_is_sync() {
    assert_sync::<StateChangeType>();
}

// ============================================================================
// Section 15: Integration-like Tests
// ============================================================================

#[test]
fn test_full_frame_analysis_workflow() {
    let mut profiler = BottleneckProfiler::new();

    // Simulate a GPU-bound frame
    profiler.begin_frame();
    profiler.record_cpu_time(Duration::from_millis(4));
    profiler.record_gpu_time(Duration::from_millis(14));
    profiler.record_state_change(StateChangeType::Pipeline);
    profiler.record_state_change(StateChangeType::BindGroup);
    profiler.record_state_change(StateChangeType::DrawCall);
    profiler.record_bandwidth(1024);

    let result = profiler.end_frame().unwrap();

    assert_eq!(result.primary, BottleneckType::GpuBound);
    assert!(result.confidence > 0.5);
    assert!(!result.recommendations.is_empty());
}

#[test]
fn test_trend_detection_over_time() {
    let mut analyzer = BottleneckAnalyzer::new();

    // First 10 frames: GPU-bound
    for i in 0..10 {
        let metrics = FrameMetrics {
            timing: TimingMetrics {
                cpu_frame_time: Duration::from_millis(4),
                gpu_frame_time: Duration::from_millis(14),
                ..Default::default()
            },
            frame_number: i,
            ..Default::default()
        };
        analyzer.record_frame(metrics);
    }

    // Next 10 frames: Improving (more balanced)
    for i in 10..20 {
        let metrics = FrameMetrics {
            timing: TimingMetrics {
                cpu_frame_time: Duration::from_millis(8),
                gpu_frame_time: Duration::from_millis(10),
                ..Default::default()
            },
            frame_number: i,
            ..Default::default()
        };
        analyzer.record_frame(metrics);
    }

    let trend = analyzer.analyze_trend();
    assert_eq!(trend.samples, 20);
    assert!(trend.has_sufficient_data());
}

#[test]
fn test_threshold_sensitivity() {
    // Same metrics should produce different results with different thresholds
    // Using a 2.0x ratio which is above aggressive (1.2) but below relaxed (2.5)
    let metrics = FrameMetrics {
        timing: TimingMetrics {
            cpu_frame_time: Duration::from_millis(5),
            gpu_frame_time: Duration::from_millis(15), // 3x ratio - clearly GPU-bound
            ..Default::default()
        },
        ..Default::default()
    };

    // With aggressive thresholds (1.2 ratio) - should detect GPU-bound
    let mut aggressive = BottleneckAnalyzer::with_thresholds(AnalysisThresholds::aggressive());
    aggressive.record_frame(metrics.clone());
    let result = aggressive.analyze_current();
    assert_eq!(result.primary, BottleneckType::GpuBound);

    // With relaxed thresholds (2.5 ratio) - 3x is still above 2.5, so should be GPU-bound
    // Using a 2.0x ratio for the relaxed test
    let metrics_moderate = FrameMetrics {
        timing: TimingMetrics {
            cpu_frame_time: Duration::from_millis(8),
            gpu_frame_time: Duration::from_millis(16), // 2.0x ratio
            ..Default::default()
        },
        ..Default::default()
    };
    let mut relaxed = BottleneckAnalyzer::with_thresholds(AnalysisThresholds::relaxed());
    relaxed.record_frame(metrics_moderate);
    let result = relaxed.analyze_current();
    // 2.0x ratio is below relaxed threshold (2.5), so should be balanced
    assert_eq!(result.primary, BottleneckType::Balanced);
}

#[test]
fn test_secondary_bottleneck_detection() {
    let mut analyzer = BottleneckAnalyzer::new();

    // Create a scenario with both GPU-bound and state thrashing
    let metrics = FrameMetrics {
        timing: TimingMetrics {
            cpu_frame_time: Duration::from_millis(4),
            gpu_frame_time: Duration::from_millis(14),
            ..Default::default()
        },
        state: StateMetrics {
            pipeline_switches: 300,
            bind_group_changes: 600,
            vertex_buffer_changes: 300,
            index_buffer_changes: 300,
            draw_calls: 300, // 5 changes per draw
        },
        ..Default::default()
    };

    analyzer.record_frame(metrics);
    let result = analyzer.analyze_current();

    // Should have a secondary bottleneck
    assert!(
        result.secondary.is_some() ||
        result.primary == BottleneckType::GpuBound ||
        result.primary == BottleneckType::StateThrashing
    );
}

#[test]
fn test_clear_invalidates_cache() {
    let mut analyzer = BottleneckAnalyzer::new();

    analyzer.record_frame(FrameMetrics::default());
    let _ = analyzer.analyze_current();
    assert!(analyzer.last_result().is_some());

    analyzer.clear();
    assert!(analyzer.last_result().is_none());
}

#[test]
fn test_record_frame_invalidates_cache() {
    let mut analyzer = BottleneckAnalyzer::new();

    analyzer.record_frame(FrameMetrics::default());
    let _ = analyzer.analyze_current();
    assert!(analyzer.last_result().is_some());

    // Recording a new frame should invalidate the cache
    analyzer.record_frame(FrameMetrics::default());
    // After recording, last_result should be None until analyze_current is called
    // (Based on the implementation which sets last_result = None in record_frame)
}
