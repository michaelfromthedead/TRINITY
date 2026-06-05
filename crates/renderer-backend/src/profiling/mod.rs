//! GPU Profiling Infrastructure.
//!
//! This module provides comprehensive GPU profiling capabilities for
//! measuring and analyzing GPU performance in the TRINITY engine.
//!
//! # Modules
//!
//! - [`timestamps`]: GPU timestamp query profiler for precise timing measurements
//! - [`memory`]: GPU memory tracking and statistics
//! - [`leaks`]: GPU resource leak detection system
//! - [`drawcalls`]: Draw call statistics tracking and analysis
//! - [`bottleneck`]: CPU/GPU bottleneck analysis and detection
//!
//! # Overview
//!
//! GPU profiling is essential for understanding rendering performance and
//! identifying bottlenecks. This module provides:
//!
//! - **Timestamp Queries**: Measure GPU execution time of render passes
//! - **RAII Guards**: Automatic timing with scope-based lifetime
//! - **Frame Profiling**: Per-frame statistics and aggregation
//! - **Period Conversion**: Convert between GPU ticks and time units
//! - **Memory Tracking**: Track GPU resource allocations and detect leaks
//! - **Leak Detection**: Identify unreleased GPU resources with configurable thresholds
//! - **Bottleneck Analysis**: Detect CPU/GPU/memory/state bottlenecks
//!
//! # Quick Start
//!
//! ```no_run
//! use renderer_backend::profiling::timestamps::{TimestampProfiler, GpuProfileScope, FrameProfiler};
//!
//! # fn example(device: &wgpu::Device, queue: &wgpu::Queue, adapter: &wgpu::Adapter) {
//! // Check support
//! if !TimestampProfiler::is_supported(adapter) {
//!     println!("Timestamps not supported on this device");
//!     return;
//! }
//!
//! // Option 1: Direct profiler usage
//! let mut profiler = TimestampProfiler::new(device, queue, 64);
//!
//! let mut encoder = device.create_command_encoder(&Default::default());
//!
//! // Manual timing
//! let handle = profiler.begin(&mut encoder, Some("Shadow Pass"));
//! // ... shadow commands ...
//! profiler.end(&mut encoder, handle);
//!
//! // RAII timing
//! {
//!     let _scope = GpuProfileScope::new(&mut profiler, &mut encoder, "Lighting");
//!     // ... lighting commands ...
//! } // Automatically ends timing
//!
//! profiler.resolve(&mut encoder);
//! queue.submit(std::iter::once(encoder.finish()));
//!
//! // Read results (1-3 frames later)
//! for result in profiler.read_results(queue) {
//!     println!("{}: {:.3}ms", result.label.as_deref().unwrap_or("unnamed"), result.duration_ms());
//! }
//!
//! // Option 2: Frame profiler for per-frame stats
//! let mut frame_profiler = FrameProfiler::new(device, queue, 64);
//!
//! let mut encoder = device.create_command_encoder(&Default::default());
//! frame_profiler.begin_frame(&mut encoder);
//!
//! {
//!     let _scope = frame_profiler.profile_pass(&mut encoder, "GBuffer");
//!     // ... gbuffer commands ...
//! }
//!
//! frame_profiler.end_frame(&mut encoder);
//! frame_profiler.resolve(&mut encoder);
//! queue.submit(std::iter::once(encoder.finish()));
//!
//! if let Some(stats) = frame_profiler.get_frame_stats(queue) {
//!     println!("Frame {}: {:.3}ms total", stats.frame_index, stats.total_ms());
//! }
//! # }
//! ```
//!
//! # Feature Detection
//!
//! Timestamp queries require the `TIMESTAMP_QUERY` feature, which is not
//! available on all devices. Always check support before profiling:
//!
//! ```no_run
//! use renderer_backend::profiling::timestamps::TimestampProfiler;
//!
//! # fn example(adapter: &wgpu::Adapter) {
//! if TimestampProfiler::is_supported(adapter) {
//!     // Create and use profiler
//! } else {
//!     // Fall back to CPU timing or skip profiling
//! }
//! # }
//! ```
//!
//! # Best Practices
//!
//! 1. **Check Feature Support**: Always verify `TIMESTAMP_QUERY` is available
//! 2. **Deferred Readback**: Results are available 1-3 frames after resolve
//! 3. **Reasonable Capacity**: Use appropriate capacity (64-256 typical)
//! 4. **Clear Each Frame**: Call `clear()` or use `FrameProfiler`
//! 5. **Use RAII**: Prefer `GpuProfileScope` over manual begin/end
//!
//! # wgpu 22+ Compatibility
//!
//! This module targets wgpu 22+ and uses:
//! - `Device::create_query_set()` with `QueryType::Timestamp`
//! - `CommandEncoder::write_timestamp()` for recording
//! - `CommandEncoder::resolve_query_set()` for result resolution
//! - `Queue::get_timestamp_period()` for tick-to-ns conversion

pub mod bottleneck;
pub mod draw_stats;
pub mod drawcalls;
pub mod leak_detector;
pub mod leaks;
pub mod memory;
pub mod memory_tracker;
pub mod timestamp;
pub mod timestamps;

// Re-export commonly used types from memory module
pub use memory::{
    // Enums
    MemoryType,
    ResourceType,

    // Core types
    AllocationInfo,
    AllocationStats,
    MemoryBudget,
    MemoryTracker,

    // Snapshot and diff
    MemorySnapshot,
    MemoryDiff,

    // Utilities
    format_bytes,
};

// Re-export leak detection types (enhanced version)
pub use leaks::{
    // Severity and candidates
    LeakSeverity,
    LeakCandidate,
    LeakThresholds,

    // Tracking
    AllocationTracker,

    // Main detector (shadows the simpler one from memory module)
    LeakDetector,

    // Statistics and reporting
    LeakStats,
    LeakReport,

    // Per-frame detection
    FrameLeakChecker,
};

pub use timestamps::{
    // Core types
    TimestampQuery,
    TimestampHandle,
    TimestampResult,
    TimestampProfiler,

    // RAII guard
    GpuProfileScope,

    // Conversion
    TimestampPeriodConverter,

    // Statistics
    ProfilerStats,
    FrameStats,
    FrameProfiler,

    // Constants
    TIMESTAMP_SIZE_BYTES,
    MIN_CAPACITY,
    MAX_RECOMMENDED_CAPACITY,
    DEFAULT_CAPACITY,
};

// Re-export bottleneck analysis types
pub use bottleneck::{
    // Enums
    BottleneckType,
    BottleneckSeverity,
    StateChangeType,

    // Metrics
    TimingMetrics as BottleneckTimingMetrics,
    ResourceMetrics,
    StateMetrics,
    FrameMetrics as BottleneckFrameMetrics,

    // Analysis results
    BottleneckResult,
    AnalysisThresholds,
    TrendAnalysis as BottleneckTrendAnalysis,

    // Analyzer and profiler
    BottleneckAnalyzer,
    BottleneckProfiler,

    // Simplified API (T-WGPU-P7.4.5)
    SimpleBottleneckType,
    Bottleneck,
    BottleneckAnalysis,
    BottleneckThresholds,
    BottleneckTrend,
    SimpleBottleneckAnalyzer,
};

// Re-export draw call tracking types
pub use drawcalls::{
    // Enums
    DrawType,
    PassType,
    PrimitiveTopology,

    // Core types
    DrawCall,
    DrawBatch,
    PassStats,
    DrawFrameStats,

    // Tracker
    DrawCallTracker,

    // Analysis
    DrawCallAnalyzer,
    FrameAnalysis,
    FrameComparison,
    TrendAnalysis,

    // Constants
    DEFAULT_HISTORY_SIZE,
    STATE_THRASHING_THRESHOLD,
    HEAVY_PASS_DRAW_THRESHOLD,
    HEAVY_PASS_VERTEX_THRESHOLD,
};

// Re-export simplified timestamp profiler types (T-WGPU-P7.4.1)
pub use timestamp::{
    // Core types
    TimestampQuery as SimpleTimestampQuery,
    TimestampResult as SimpleTimestampResult,
    TimestampProfiler as SimpleTimestampProfiler,

    // RAII guard
    TimestampScope,

    // Statistics
    ProfileStats,

    // Constants
    TIMESTAMP_SIZE_BYTES as SIMPLE_TIMESTAMP_SIZE_BYTES,
    MAX_QUERIES,
    DEFAULT_MAX_QUERIES,
};

// Re-export memory tracker types (T-WGPU-P7.4.2)
pub use memory_tracker::{
    // Enums
    MemoryCategory,

    // Core types
    MemoryAllocation,
    MemoryStats as CategoryMemoryStats,
    MemoryBudget as CategoryMemoryBudget,
    MemoryTracker as CategoryMemoryTracker,
};

// Re-export leak detector types (T-WGPU-P7.4.3)
pub use leak_detector::{
    // Enums
    ResourceType as LeakResourceType,

    // Core types
    TrackedResource,
    LeakReport as DetailedLeakReport,
    LeakDetector as ResourceLeakDetector,
    LeakDetectorStats,

    // RAII
    LeakScope,
};

// Re-export draw statistics types (T-WGPU-P7.4.4)
pub use draw_stats::{
    // Enums
    DrawCallType,

    // Core types
    DrawCallInfo,
    FrameDrawStats,
    DrawStatsSummary,

    // Collector
    DrawStatsCollector,

    // Constants
    DEFAULT_HISTORY_SIZE as DRAW_STATS_DEFAULT_HISTORY_SIZE,
    MIN_HISTORY_SIZE as DRAW_STATS_MIN_HISTORY_SIZE,
    MAX_HISTORY_SIZE as DRAW_STATS_MAX_HISTORY_SIZE,
};
