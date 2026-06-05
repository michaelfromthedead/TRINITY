//! GPU/CPU Bottleneck Analysis for wgpu 25.x Performance Optimization.
//!
//! This module provides comprehensive bottleneck analysis capabilities for
//! identifying and diagnosing performance limitations in the rendering pipeline.
//!
//! # Overview
//!
//! The bottleneck analyzer helps identify:
//! - **CPU-bound scenarios**: Draw call submission overhead, too many small draws
//! - **GPU-bound scenarios**: Shader/rasterization limited rendering
//! - **Memory bandwidth**: Texture/buffer transfer bottlenecks
//! - **State thrashing**: Frequent pipeline/binding changes
//! - **Synchronization stalls**: CPU-GPU pipeline bubbles
//!
//! # Usage
//!
//! ```no_run
//! use renderer_backend::profiling::bottleneck::{
//!     BottleneckAnalyzer, BottleneckProfiler, FrameMetrics,
//!     TimingMetrics, ResourceMetrics, StateMetrics,
//! };
//! use std::time::Duration;
//!
//! // Create analyzer
//! let mut analyzer = BottleneckAnalyzer::new();
//!
//! // Record frame metrics
//! let metrics = FrameMetrics {
//!     timing: TimingMetrics {
//!         cpu_frame_time: Duration::from_micros(8000),
//!         gpu_frame_time: Duration::from_micros(12000),
//!         cpu_submit_time: Duration::from_micros(500),
//!         gpu_wait_time: Duration::from_micros(100),
//!         present_wait_time: Duration::from_micros(4000),
//!     },
//!     resources: ResourceMetrics::default(),
//!     state: StateMetrics {
//!         pipeline_switches: 50,
//!         bind_group_changes: 200,
//!         vertex_buffer_changes: 100,
//!         index_buffer_changes: 100,
//!         draw_calls: 500,
//!     },
//!     frame_number: 1,
//!     timestamp: std::time::Instant::now(),
//! };
//!
//! analyzer.record_frame(metrics);
//!
//! // Analyze current bottleneck
//! let result = analyzer.analyze_current();
//! println!("Primary bottleneck: {:?}", result.primary);
//! println!("Confidence: {:.1}%", result.confidence * 100.0);
//! for hint in result.primary.optimization_hints() {
//!     println!("  Hint: {}", hint);
//! }
//! ```
//!
//! # Real-time Profiling
//!
//! ```no_run
//! use renderer_backend::profiling::bottleneck::{BottleneckProfiler, BottleneckSeverity, StateChangeType};
//! use std::time::Duration;
//!
//! let mut profiler = BottleneckProfiler::new();
//! profiler.set_log_threshold(BottleneckSeverity::Medium);
//!
//! // Per-frame profiling
//! profiler.begin_frame();
//! profiler.record_cpu_time(Duration::from_micros(5000));
//! profiler.record_gpu_time(Duration::from_micros(8000));
//! profiler.record_state_change(StateChangeType::Pipeline);
//! profiler.record_state_change(StateChangeType::BindGroup);
//! profiler.record_bandwidth(1024 * 1024); // 1MB upload
//!
//! if let Some(result) = profiler.end_frame() {
//!     println!("Frame bottleneck: {:?}", result.primary);
//! }
//! ```

use std::collections::VecDeque;
use std::fmt;
use std::time::{Duration, Instant};

// ============================================================================
// BottleneckType
// ============================================================================

/// Types of performance bottlenecks that can be detected.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BottleneckType {
    /// CPU-bound: draw call submission overhead dominates.
    ///
    /// Indicators:
    /// - CPU frame time >> GPU frame time
    /// - Many small draw calls
    /// - High driver overhead
    CpuBound,

    /// GPU-bound: shader/rasterization limited.
    ///
    /// Indicators:
    /// - GPU frame time >> CPU frame time
    /// - High GPU utilization
    /// - Complex shaders or overdraw
    GpuBound,

    /// Memory bandwidth limited: texture/buffer transfers dominate.
    ///
    /// Indicators:
    /// - High bandwidth usage
    /// - Frequent texture uploads
    /// - Large readback operations
    MemoryBandwidth,

    /// State thrashing: too many pipeline/binding changes.
    ///
    /// Indicators:
    /// - High state changes per draw call
    /// - Unsorted draw calls
    /// - Too many unique materials
    StateThrashing,

    /// Synchronization bottleneck: CPU-GPU stalls.
    ///
    /// Indicators:
    /// - High GPU wait time
    /// - Pipeline bubbles
    /// - Blocking readbacks
    Synchronization,

    /// Balanced workload: no clear bottleneck.
    ///
    /// CPU and GPU are both utilized efficiently with
    /// good overlap and no obvious limiting factor.
    Balanced,
}

impl BottleneckType {
    /// Get the severity level for this bottleneck type.
    ///
    /// Returns a baseline severity; actual severity depends on metrics.
    #[must_use]
    pub fn severity(&self) -> BottleneckSeverity {
        match self {
            BottleneckType::CpuBound => BottleneckSeverity::Medium,
            BottleneckType::GpuBound => BottleneckSeverity::Medium,
            BottleneckType::MemoryBandwidth => BottleneckSeverity::High,
            BottleneckType::StateThrashing => BottleneckSeverity::Medium,
            BottleneckType::Synchronization => BottleneckSeverity::High,
            BottleneckType::Balanced => BottleneckSeverity::None,
        }
    }

    /// Get optimization hints for this bottleneck type.
    #[must_use]
    pub fn optimization_hints(&self) -> Vec<&'static str> {
        match self {
            BottleneckType::CpuBound => vec![
                "Batch draw calls using instancing or indirect draws",
                "Reduce driver overhead with multi-draw-indirect",
                "Use GPU-driven rendering to move work to compute",
                "Merge small meshes into larger batches",
                "Reduce material/pipeline permutations",
                "Consider frustum culling on GPU instead of CPU",
            ],
            BottleneckType::GpuBound => vec![
                "Optimize shaders: reduce ALU operations and texture samples",
                "Use LOD (Level of Detail) for distant objects",
                "Reduce overdraw with depth pre-pass",
                "Lower resolution for expensive effects",
                "Use async compute for independent work",
                "Profile individual passes to find hotspots",
            ],
            BottleneckType::MemoryBandwidth => vec![
                "Compress textures (BC/ASTC formats)",
                "Use texture streaming for large assets",
                "Reduce buffer upload frequency",
                "Avoid CPU readbacks; use async readback if needed",
                "Pool and reuse staging buffers",
                "Consider bindless textures to reduce binding overhead",
            ],
            BottleneckType::StateThrashing => vec![
                "Sort draw calls by pipeline state",
                "Use render bundles for static geometry",
                "Merge materials where possible",
                "Use texture arrays instead of individual textures",
                "Batch by bind group to minimize rebinding",
                "Consider uber-shaders with specialization constants",
            ],
            BottleneckType::Synchronization => vec![
                "Use triple buffering for CPU-GPU overlap",
                "Avoid synchronous buffer mapping",
                "Use async readback with fence queries",
                "Pipeline GPU work to fill bubbles",
                "Move CPU work earlier in frame",
                "Consider ring buffers for uniform updates",
            ],
            BottleneckType::Balanced => vec![
                "Performance is well-balanced",
                "Consider increasing quality settings",
                "Monitor for regressions",
            ],
        }
    }

    /// Get a short description of this bottleneck type.
    #[must_use]
    pub fn description(&self) -> &'static str {
        match self {
            BottleneckType::CpuBound => "CPU draw call submission overhead",
            BottleneckType::GpuBound => "GPU shader/rasterization limited",
            BottleneckType::MemoryBandwidth => "Memory bandwidth limited",
            BottleneckType::StateThrashing => "Excessive state changes",
            BottleneckType::Synchronization => "CPU-GPU synchronization stalls",
            BottleneckType::Balanced => "No clear bottleneck",
        }
    }
}

impl fmt::Display for BottleneckType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BottleneckType::CpuBound => write!(f, "CPU Bound"),
            BottleneckType::GpuBound => write!(f, "GPU Bound"),
            BottleneckType::MemoryBandwidth => write!(f, "Memory Bandwidth"),
            BottleneckType::StateThrashing => write!(f, "State Thrashing"),
            BottleneckType::Synchronization => write!(f, "Synchronization"),
            BottleneckType::Balanced => write!(f, "Balanced"),
        }
    }
}

// ============================================================================
// BottleneckSeverity
// ============================================================================

/// Severity level of a detected bottleneck.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum BottleneckSeverity {
    /// No bottleneck detected.
    None,
    /// Minor bottleneck, may not require action.
    Low,
    /// Moderate bottleneck, should be addressed.
    Medium,
    /// Significant bottleneck affecting performance.
    High,
    /// Critical bottleneck causing severe performance issues.
    Critical,
}

impl BottleneckSeverity {
    /// Create severity from a score (0.0 = None, 1.0 = Critical).
    #[must_use]
    pub fn from_score(score: f32) -> Self {
        match score {
            s if s <= 0.1 => BottleneckSeverity::None,
            s if s <= 0.3 => BottleneckSeverity::Low,
            s if s <= 0.5 => BottleneckSeverity::Medium,
            s if s <= 0.75 => BottleneckSeverity::High,
            _ => BottleneckSeverity::Critical,
        }
    }

    /// Get a display color for terminal output.
    ///
    /// Returns ANSI color escape codes.
    #[must_use]
    pub fn display_color(&self) -> &'static str {
        match self {
            BottleneckSeverity::None => "\x1b[32m",     // Green
            BottleneckSeverity::Low => "\x1b[33m",      // Yellow
            BottleneckSeverity::Medium => "\x1b[33m",   // Yellow
            BottleneckSeverity::High => "\x1b[31m",     // Red
            BottleneckSeverity::Critical => "\x1b[91m", // Bright red
        }
    }

    /// Get a plain text description.
    #[must_use]
    pub fn description(&self) -> &'static str {
        match self {
            BottleneckSeverity::None => "None",
            BottleneckSeverity::Low => "Low",
            BottleneckSeverity::Medium => "Medium",
            BottleneckSeverity::High => "High",
            BottleneckSeverity::Critical => "Critical",
        }
    }

    /// Convert severity to a numeric score (0.0-1.0).
    #[must_use]
    pub fn to_score(&self) -> f32 {
        match self {
            BottleneckSeverity::None => 0.0,
            BottleneckSeverity::Low => 0.25,
            BottleneckSeverity::Medium => 0.5,
            BottleneckSeverity::High => 0.75,
            BottleneckSeverity::Critical => 1.0,
        }
    }
}

impl Default for BottleneckSeverity {
    fn default() -> Self {
        BottleneckSeverity::None
    }
}

impl fmt::Display for BottleneckSeverity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.description())
    }
}

// ============================================================================
// TimingMetrics
// ============================================================================

/// Timing metrics for CPU/GPU performance analysis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct TimingMetrics {
    /// Total CPU time spent on frame processing.
    pub cpu_frame_time: Duration,

    /// Total GPU time spent executing commands.
    pub gpu_frame_time: Duration,

    /// Time spent submitting commands to the GPU.
    pub cpu_submit_time: Duration,

    /// Time CPU spent waiting for GPU to complete.
    pub gpu_wait_time: Duration,

    /// Time spent waiting for present/vsync.
    pub present_wait_time: Duration,
}

impl TimingMetrics {
    /// Create new timing metrics.
    #[must_use]
    pub fn new(
        cpu_frame_time: Duration,
        gpu_frame_time: Duration,
        cpu_submit_time: Duration,
        gpu_wait_time: Duration,
        present_wait_time: Duration,
    ) -> Self {
        Self {
            cpu_frame_time,
            gpu_frame_time,
            cpu_submit_time,
            gpu_wait_time,
            present_wait_time,
        }
    }

    /// Calculate frame overlap ratio.
    ///
    /// Returns a value from 0.0 (completely serial execution) to 1.0 (full overlap).
    /// Higher overlap means better CPU-GPU parallelism.
    #[must_use]
    pub fn frame_overlap(&self) -> f32 {
        let cpu_us = self.cpu_frame_time.as_micros() as f32;
        let gpu_us = self.gpu_frame_time.as_micros() as f32;
        let wait_us = self.gpu_wait_time.as_micros() as f32;

        if cpu_us <= 0.0 || gpu_us <= 0.0 {
            return 0.0;
        }

        // Ideal frame time would be max(cpu, gpu)
        // Actual frame time includes stalls
        let ideal = cpu_us.max(gpu_us);
        let actual = cpu_us + wait_us;

        if actual <= 0.0 {
            return 0.0;
        }

        // Overlap = 1 - (wasted time / ideal)
        let wasted = (actual - ideal).max(0.0);
        (1.0 - wasted / ideal).clamp(0.0, 1.0)
    }

    /// Calculate GPU utilization ratio.
    ///
    /// Returns the fraction of frame time the GPU was actively working.
    #[must_use]
    pub fn gpu_utilization(&self) -> f32 {
        let gpu_us = self.gpu_frame_time.as_micros() as f32;
        let cpu_us = self.cpu_frame_time.as_micros() as f32;
        let present_us = self.present_wait_time.as_micros() as f32;

        // Total available time is max of CPU work + present
        let total_us = cpu_us + present_us;
        if total_us <= 0.0 {
            return 0.0;
        }

        (gpu_us / total_us).clamp(0.0, 1.0)
    }

    /// Check if CPU time dominates GPU time.
    #[must_use]
    pub fn is_cpu_bound(&self, ratio_threshold: f32) -> bool {
        let cpu_us = self.cpu_frame_time.as_micros() as f32;
        let gpu_us = self.gpu_frame_time.as_micros() as f32;

        gpu_us > 0.0 && cpu_us / gpu_us > ratio_threshold
    }

    /// Check if GPU time dominates CPU time.
    #[must_use]
    pub fn is_gpu_bound(&self, ratio_threshold: f32) -> bool {
        let cpu_us = self.cpu_frame_time.as_micros() as f32;
        let gpu_us = self.gpu_frame_time.as_micros() as f32;

        cpu_us > 0.0 && gpu_us / cpu_us > ratio_threshold
    }

    /// Check if synchronization wait time is significant.
    #[must_use]
    pub fn has_sync_stalls(&self, threshold_ms: f32) -> bool {
        self.gpu_wait_time.as_secs_f32() * 1000.0 > threshold_ms
    }

    /// Get CPU frame time in milliseconds.
    #[must_use]
    pub fn cpu_frame_time_ms(&self) -> f64 {
        self.cpu_frame_time.as_secs_f64() * 1000.0
    }

    /// Get GPU frame time in milliseconds.
    #[must_use]
    pub fn gpu_frame_time_ms(&self) -> f64 {
        self.gpu_frame_time.as_secs_f64() * 1000.0
    }
}

// ============================================================================
// ResourceMetrics
// ============================================================================

/// Resource transfer metrics for bandwidth analysis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ResourceMetrics {
    /// Bytes uploaded to GPU textures this frame.
    pub texture_uploads_bytes: u64,

    /// Bytes uploaded to GPU buffers this frame.
    pub buffer_uploads_bytes: u64,

    /// Bytes read back from GPU this frame.
    pub readback_bytes: u64,

    /// Estimated bandwidth used in MB/s.
    pub bandwidth_used_mbps: u32,

    /// Estimated cache misses (texture/buffer).
    pub cache_misses: u64,
}

impl ResourceMetrics {
    /// Create new resource metrics.
    #[must_use]
    pub fn new(
        texture_uploads_bytes: u64,
        buffer_uploads_bytes: u64,
        readback_bytes: u64,
        bandwidth_used_mbps: u32,
        cache_misses: u64,
    ) -> Self {
        Self {
            texture_uploads_bytes,
            buffer_uploads_bytes,
            readback_bytes,
            bandwidth_used_mbps,
            cache_misses,
        }
    }

    /// Get total bytes transferred this frame.
    #[must_use]
    pub fn total_bytes(&self) -> u64 {
        self.texture_uploads_bytes
            .saturating_add(self.buffer_uploads_bytes)
            .saturating_add(self.readback_bytes)
    }

    /// Check if bandwidth usage exceeds threshold.
    #[must_use]
    pub fn is_bandwidth_limited(&self, threshold_mbps: u32) -> bool {
        self.bandwidth_used_mbps > threshold_mbps
    }

    /// Check if there are significant readback operations.
    #[must_use]
    pub fn has_readback_stalls(&self, threshold_bytes: u64) -> bool {
        self.readback_bytes > threshold_bytes
    }

    /// Get bandwidth usage as a fraction (0.0-1.0) of a given limit.
    #[must_use]
    pub fn bandwidth_utilization(&self, limit_mbps: u32) -> f32 {
        if limit_mbps == 0 {
            return 0.0;
        }
        (self.bandwidth_used_mbps as f32 / limit_mbps as f32).clamp(0.0, 1.0)
    }
}

// ============================================================================
// StateMetrics
// ============================================================================

/// State change metrics for detecting thrashing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct StateMetrics {
    /// Number of pipeline (render/compute) switches.
    pub pipeline_switches: u64,

    /// Number of bind group changes.
    pub bind_group_changes: u64,

    /// Number of vertex buffer binding changes.
    pub vertex_buffer_changes: u64,

    /// Number of index buffer binding changes.
    pub index_buffer_changes: u64,

    /// Total number of draw calls.
    pub draw_calls: u64,
}

impl StateMetrics {
    /// Create new state metrics.
    #[must_use]
    pub fn new(
        pipeline_switches: u64,
        bind_group_changes: u64,
        vertex_buffer_changes: u64,
        index_buffer_changes: u64,
        draw_calls: u64,
    ) -> Self {
        Self {
            pipeline_switches,
            bind_group_changes,
            vertex_buffer_changes,
            index_buffer_changes,
            draw_calls,
        }
    }

    /// Calculate total state changes.
    #[must_use]
    pub fn total_state_changes(&self) -> u64 {
        self.pipeline_switches
            .saturating_add(self.bind_group_changes)
            .saturating_add(self.vertex_buffer_changes)
            .saturating_add(self.index_buffer_changes)
    }

    /// Calculate state changes per draw call.
    ///
    /// Higher values indicate poor draw call sorting.
    #[must_use]
    pub fn state_changes_per_draw(&self) -> f32 {
        if self.draw_calls == 0 {
            return 0.0;
        }
        self.total_state_changes() as f32 / self.draw_calls as f32
    }

    /// Check if state changes exceed threshold per draw.
    ///
    /// Typical good values are < 2.0 changes per draw.
    #[must_use]
    pub fn is_state_thrashing(&self, threshold: f32) -> bool {
        self.state_changes_per_draw() > threshold
    }

    /// Get pipeline switches per draw call.
    #[must_use]
    pub fn pipelines_per_draw(&self) -> f32 {
        if self.draw_calls == 0 {
            return 0.0;
        }
        self.pipeline_switches as f32 / self.draw_calls as f32
    }

    /// Get bind group changes per draw call.
    #[must_use]
    pub fn bind_groups_per_draw(&self) -> f32 {
        if self.draw_calls == 0 {
            return 0.0;
        }
        self.bind_group_changes as f32 / self.draw_calls as f32
    }
}

// ============================================================================
// FrameMetrics
// ============================================================================

/// Aggregate metrics for a single frame.
#[derive(Debug, Clone)]
pub struct FrameMetrics {
    /// Timing information.
    pub timing: TimingMetrics,

    /// Resource transfer information.
    pub resources: ResourceMetrics,

    /// State change information.
    pub state: StateMetrics,

    /// Frame number (monotonically increasing).
    pub frame_number: u64,

    /// Timestamp when this frame was recorded.
    pub timestamp: Instant,
}

impl FrameMetrics {
    /// Create new frame metrics.
    #[must_use]
    pub fn new(
        timing: TimingMetrics,
        resources: ResourceMetrics,
        state: StateMetrics,
        frame_number: u64,
    ) -> Self {
        Self {
            timing,
            resources,
            state,
            frame_number,
            timestamp: Instant::now(),
        }
    }

    /// Create frame metrics with current timestamp.
    #[must_use]
    pub fn with_timestamp(
        timing: TimingMetrics,
        resources: ResourceMetrics,
        state: StateMetrics,
        frame_number: u64,
        timestamp: Instant,
    ) -> Self {
        Self {
            timing,
            resources,
            state,
            frame_number,
            timestamp,
        }
    }

    /// Get the age of this frame in seconds.
    #[must_use]
    pub fn age_secs(&self) -> f64 {
        self.timestamp.elapsed().as_secs_f64()
    }
}

impl Default for FrameMetrics {
    fn default() -> Self {
        Self {
            timing: TimingMetrics::default(),
            resources: ResourceMetrics::default(),
            state: StateMetrics::default(),
            frame_number: 0,
            timestamp: Instant::now(),
        }
    }
}

// ============================================================================
// BottleneckResult
// ============================================================================

/// Result of bottleneck analysis.
#[derive(Debug, Clone)]
pub struct BottleneckResult {
    /// Primary detected bottleneck.
    pub primary: BottleneckType,

    /// Secondary bottleneck (if any).
    pub secondary: Option<BottleneckType>,

    /// Confidence in the analysis (0.0-1.0).
    pub confidence: f32,

    /// Metrics used for analysis.
    pub metrics: FrameMetrics,

    /// Specific recommendations based on metrics.
    pub recommendations: Vec<String>,

    /// Detailed analysis description.
    pub details: String,
}

impl BottleneckResult {
    /// Create a new bottleneck result.
    #[must_use]
    pub fn new(primary: BottleneckType, metrics: FrameMetrics) -> Self {
        Self {
            primary,
            secondary: None,
            confidence: 1.0,
            metrics,
            recommendations: primary
                .optimization_hints()
                .iter()
                .map(|s| s.to_string())
                .collect(),
            details: String::new(),
        }
    }

    /// Create a balanced result (no bottleneck).
    #[must_use]
    pub fn balanced(metrics: FrameMetrics) -> Self {
        Self {
            primary: BottleneckType::Balanced,
            secondary: None,
            confidence: 1.0,
            metrics,
            recommendations: vec![],
            details: "No significant bottleneck detected.".to_string(),
        }
    }

    /// Get the severity of the primary bottleneck.
    #[must_use]
    pub fn severity(&self) -> BottleneckSeverity {
        // Scale severity by confidence
        let base_score = self.primary.severity().to_score();
        BottleneckSeverity::from_score(base_score * self.confidence)
    }

    /// Check if this result indicates a significant bottleneck.
    #[must_use]
    pub fn is_significant(&self) -> bool {
        self.primary != BottleneckType::Balanced && self.confidence > 0.5
    }

    /// Get all optimization hints from both primary and secondary.
    #[must_use]
    pub fn all_hints(&self) -> Vec<&'static str> {
        let mut hints = self.primary.optimization_hints();
        if let Some(secondary) = &self.secondary {
            hints.extend(secondary.optimization_hints());
        }
        hints
    }
}

impl fmt::Display for BottleneckResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} (confidence: {:.0}%)",
            self.primary,
            self.confidence * 100.0
        )?;
        if let Some(secondary) = &self.secondary {
            write!(f, " + {}", secondary)?;
        }
        Ok(())
    }
}

// ============================================================================
// AnalysisThresholds
// ============================================================================

/// Configurable thresholds for bottleneck analysis.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AnalysisThresholds {
    /// GPU/CPU time ratio threshold for GPU-bound detection.
    pub gpu_bound_ratio: f32,

    /// CPU/GPU time ratio threshold for CPU-bound detection.
    pub cpu_bound_ratio: f32,

    /// State changes per draw threshold for thrashing detection.
    pub state_thrash_per_draw: f32,

    /// Bandwidth limit in MB/s for bandwidth-limited detection.
    pub bandwidth_mbps_limit: f32,

    /// Sync stall threshold in milliseconds.
    pub sync_stall_ms: f32,

    /// Minimum frame time (ms) to consider for analysis.
    pub min_frame_time_ms: f32,
}

impl AnalysisThresholds {
    /// Create new thresholds with custom values.
    #[must_use]
    pub fn new(
        gpu_bound_ratio: f32,
        cpu_bound_ratio: f32,
        state_thrash_per_draw: f32,
        bandwidth_mbps_limit: f32,
        sync_stall_ms: f32,
    ) -> Self {
        Self {
            gpu_bound_ratio,
            cpu_bound_ratio,
            state_thrash_per_draw,
            bandwidth_mbps_limit,
            sync_stall_ms,
            min_frame_time_ms: 0.1,
        }
    }

    /// Create aggressive thresholds for strict bottleneck detection.
    #[must_use]
    pub fn aggressive() -> Self {
        Self {
            gpu_bound_ratio: 1.2,
            cpu_bound_ratio: 1.2,
            state_thrash_per_draw: 1.5,
            bandwidth_mbps_limit: 5000.0,
            sync_stall_ms: 0.5,
            min_frame_time_ms: 0.05,
        }
    }

    /// Create relaxed thresholds for less sensitive detection.
    #[must_use]
    pub fn relaxed() -> Self {
        Self {
            gpu_bound_ratio: 2.5,
            cpu_bound_ratio: 2.5,
            state_thrash_per_draw: 4.0,
            bandwidth_mbps_limit: 15000.0,
            sync_stall_ms: 3.0,
            min_frame_time_ms: 0.5,
        }
    }
}

impl Default for AnalysisThresholds {
    fn default() -> Self {
        Self {
            gpu_bound_ratio: 1.5,
            cpu_bound_ratio: 1.5,
            state_thrash_per_draw: 2.0,
            bandwidth_mbps_limit: 10000.0, // 10 GB/s
            sync_stall_ms: 1.0,
            min_frame_time_ms: 0.1,
        }
    }
}

// ============================================================================
// TrendAnalysis
// ============================================================================

/// Analysis of bottleneck trends over multiple frames.
#[derive(Debug, Clone)]
pub struct TrendAnalysis {
    /// Number of frames analyzed.
    pub samples: usize,

    /// Most common bottleneck type.
    pub avg_bottleneck: BottleneckType,

    /// How consistent the bottleneck type is (0.0-1.0).
    pub bottleneck_stability: f32,

    /// Whether performance is improving.
    pub improving: bool,

    /// Whether performance is degrading.
    pub degrading: bool,

    /// Frames with significant spikes.
    pub spikes: Vec<(u64, BottleneckType)>,

    /// Average frame time over the period.
    pub avg_frame_time_ms: f64,

    /// Standard deviation of frame time.
    pub frame_time_stddev_ms: f64,
}

impl TrendAnalysis {
    /// Create empty trend analysis.
    #[must_use]
    pub fn empty() -> Self {
        Self {
            samples: 0,
            avg_bottleneck: BottleneckType::Balanced,
            bottleneck_stability: 1.0,
            improving: false,
            degrading: false,
            spikes: vec![],
            avg_frame_time_ms: 0.0,
            frame_time_stddev_ms: 0.0,
        }
    }

    /// Check if there's enough data for meaningful analysis.
    #[must_use]
    pub fn has_sufficient_data(&self) -> bool {
        self.samples >= 10
    }

    /// Get the jitter coefficient (stddev / avg).
    #[must_use]
    pub fn jitter(&self) -> f64 {
        if self.avg_frame_time_ms <= 0.0 {
            return 0.0;
        }
        self.frame_time_stddev_ms / self.avg_frame_time_ms
    }

    /// Check if frame times are stable.
    #[must_use]
    pub fn is_stable(&self) -> bool {
        self.jitter() < 0.1
    }
}

impl Default for TrendAnalysis {
    fn default() -> Self {
        Self::empty()
    }
}

impl fmt::Display for TrendAnalysis {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "TrendAnalysis({} samples, avg: {}, stability: {:.0}%",
            self.samples, self.avg_bottleneck, self.bottleneck_stability * 100.0
        )?;
        if self.improving {
            write!(f, ", improving")?;
        }
        if self.degrading {
            write!(f, ", degrading")?;
        }
        if !self.spikes.is_empty() {
            write!(f, ", {} spikes", self.spikes.len())?;
        }
        write!(f, ")")
    }
}

// ============================================================================
// BottleneckAnalyzer
// ============================================================================

/// Main interface for bottleneck analysis.
///
/// Maintains a history of frame metrics and provides analysis capabilities.
pub struct BottleneckAnalyzer {
    /// Ring buffer of frame metrics.
    history: VecDeque<FrameMetrics>,

    /// Maximum history size.
    history_size: usize,

    /// Analysis thresholds.
    thresholds: AnalysisThresholds,

    /// Cache of recent analysis results.
    last_result: Option<BottleneckResult>,
}

impl BottleneckAnalyzer {
    /// Default history size (60 frames = 1 second at 60 FPS).
    pub const DEFAULT_HISTORY_SIZE: usize = 60;

    /// Create a new bottleneck analyzer with default settings.
    #[must_use]
    pub fn new() -> Self {
        Self {
            history: VecDeque::with_capacity(Self::DEFAULT_HISTORY_SIZE),
            history_size: Self::DEFAULT_HISTORY_SIZE,
            thresholds: AnalysisThresholds::default(),
            last_result: None,
        }
    }

    /// Create a new analyzer with custom thresholds.
    #[must_use]
    pub fn with_thresholds(thresholds: AnalysisThresholds) -> Self {
        Self {
            history: VecDeque::with_capacity(Self::DEFAULT_HISTORY_SIZE),
            history_size: Self::DEFAULT_HISTORY_SIZE,
            thresholds,
            last_result: None,
        }
    }

    /// Create a new analyzer with custom history size.
    #[must_use]
    pub fn with_history_size(size: usize) -> Self {
        let size = size.max(1);
        Self {
            history: VecDeque::with_capacity(size),
            history_size: size,
            thresholds: AnalysisThresholds::default(),
            last_result: None,
        }
    }

    /// Get the current thresholds.
    #[must_use]
    pub fn thresholds(&self) -> &AnalysisThresholds {
        &self.thresholds
    }

    /// Set new thresholds.
    pub fn set_thresholds(&mut self, thresholds: AnalysisThresholds) {
        self.thresholds = thresholds;
    }

    /// Get the history size.
    #[must_use]
    pub fn history_size(&self) -> usize {
        self.history_size
    }

    /// Get the number of frames currently in history.
    #[must_use]
    pub fn frame_count(&self) -> usize {
        self.history.len()
    }

    /// Record a new frame's metrics.
    pub fn record_frame(&mut self, metrics: FrameMetrics) {
        // Remove oldest if at capacity
        if self.history.len() >= self.history_size {
            self.history.pop_front();
        }
        self.history.push_back(metrics);
        // Invalidate cached result
        self.last_result = None;
    }

    /// Analyze the most recent frame.
    ///
    /// Returns a detailed bottleneck analysis result.
    #[must_use]
    pub fn analyze_current(&mut self) -> BottleneckResult {
        if let Some(metrics) = self.history.back().cloned() {
            let result = self.analyze_frame(&metrics);
            self.last_result = Some(result.clone());
            result
        } else {
            BottleneckResult::balanced(FrameMetrics::default())
        }
    }

    /// Analyze a specific frame's metrics.
    #[must_use]
    pub fn analyze_frame(&self, metrics: &FrameMetrics) -> BottleneckResult {
        let mut scores = vec![
            (BottleneckType::CpuBound, self.score_cpu_bound(metrics)),
            (BottleneckType::GpuBound, self.score_gpu_bound(metrics)),
            (
                BottleneckType::MemoryBandwidth,
                self.score_bandwidth(metrics),
            ),
            (
                BottleneckType::StateThrashing,
                self.score_state_thrashing(metrics),
            ),
            (
                BottleneckType::Synchronization,
                self.score_synchronization(metrics),
            ),
        ];

        // Sort by score descending
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let (primary, primary_score) = scores[0];
        let (secondary_type, secondary_score) = scores[1];

        // If primary score is too low, consider balanced
        if primary_score < 0.3 {
            return BottleneckResult::balanced(metrics.clone());
        }

        let mut result = BottleneckResult::new(primary, metrics.clone());
        result.confidence = primary_score.clamp(0.0, 1.0);

        // Add secondary if it's also significant
        if secondary_score > 0.4 && secondary_score > primary_score * 0.6 {
            result.secondary = Some(secondary_type);
        }

        // Generate detailed analysis
        result.details = self.generate_details(metrics, &scores);
        result.recommendations = self.generate_recommendations(metrics, primary, result.secondary);

        result
    }

    /// Score CPU-bound likelihood.
    fn score_cpu_bound(&self, metrics: &FrameMetrics) -> f32 {
        let cpu_us = metrics.timing.cpu_frame_time.as_micros() as f32;
        let gpu_us = metrics.timing.gpu_frame_time.as_micros() as f32;

        if gpu_us <= 0.0 {
            return 0.0;
        }

        let ratio = cpu_us / gpu_us;
        let base_score = if ratio > self.thresholds.cpu_bound_ratio {
            ((ratio - self.thresholds.cpu_bound_ratio) / 2.0).clamp(0.0, 1.0)
        } else {
            0.0
        };

        // Boost score if there are many draw calls
        let draw_boost = if metrics.state.draw_calls > 1000 {
            0.1
        } else {
            0.0
        };

        // Boost score if submit time is high relative to frame time
        let submit_us = metrics.timing.cpu_submit_time.as_micros() as f32;
        let submit_boost = if cpu_us > 0.0 && submit_us / cpu_us > 0.3 {
            0.15
        } else {
            0.0
        };

        (base_score + draw_boost + submit_boost).clamp(0.0, 1.0)
    }

    /// Score GPU-bound likelihood.
    fn score_gpu_bound(&self, metrics: &FrameMetrics) -> f32 {
        let cpu_us = metrics.timing.cpu_frame_time.as_micros() as f32;
        let gpu_us = metrics.timing.gpu_frame_time.as_micros() as f32;

        if cpu_us <= 0.0 {
            return 0.0;
        }

        let ratio = gpu_us / cpu_us;
        if ratio > self.thresholds.gpu_bound_ratio {
            ((ratio - self.thresholds.gpu_bound_ratio) / 2.0).clamp(0.0, 1.0)
        } else {
            0.0
        }
    }

    /// Score memory bandwidth limitation.
    fn score_bandwidth(&self, metrics: &FrameMetrics) -> f32 {
        let bandwidth = metrics.resources.bandwidth_used_mbps as f32;
        let limit = self.thresholds.bandwidth_mbps_limit;

        if bandwidth <= limit * 0.5 {
            return 0.0;
        }

        let base_score = ((bandwidth - limit * 0.5) / (limit * 0.5)).clamp(0.0, 1.0);

        // Boost for readback operations
        let readback_boost = if metrics.resources.readback_bytes > 1024 * 1024 {
            0.2
        } else {
            0.0
        };

        (base_score + readback_boost).clamp(0.0, 1.0)
    }

    /// Score state thrashing.
    fn score_state_thrashing(&self, metrics: &FrameMetrics) -> f32 {
        let changes_per_draw = metrics.state.state_changes_per_draw();
        let threshold = self.thresholds.state_thrash_per_draw;

        if changes_per_draw <= threshold {
            return 0.0;
        }

        ((changes_per_draw - threshold) / threshold).clamp(0.0, 1.0)
    }

    /// Score synchronization bottleneck.
    fn score_synchronization(&self, metrics: &FrameMetrics) -> f32 {
        let wait_ms = metrics.timing.gpu_wait_time.as_secs_f32() * 1000.0;
        let threshold = self.thresholds.sync_stall_ms;

        if wait_ms <= threshold {
            return 0.0;
        }

        ((wait_ms - threshold) / (threshold * 2.0)).clamp(0.0, 1.0)
    }

    /// Generate detailed analysis text.
    fn generate_details(
        &self,
        metrics: &FrameMetrics,
        scores: &[(BottleneckType, f32)],
    ) -> String {
        let mut details = String::new();

        details.push_str(&format!(
            "Frame {}: CPU={:.2}ms, GPU={:.2}ms\n",
            metrics.frame_number,
            metrics.timing.cpu_frame_time_ms(),
            metrics.timing.gpu_frame_time_ms()
        ));

        details.push_str(&format!(
            "Overlap: {:.0}%, GPU Util: {:.0}%\n",
            metrics.timing.frame_overlap() * 100.0,
            metrics.timing.gpu_utilization() * 100.0
        ));

        if metrics.state.draw_calls > 0 {
            details.push_str(&format!(
                "Draw calls: {}, State changes/draw: {:.2}\n",
                metrics.state.draw_calls,
                metrics.state.state_changes_per_draw()
            ));
        }

        if metrics.resources.bandwidth_used_mbps > 0 {
            details.push_str(&format!(
                "Bandwidth: {} MB/s\n",
                metrics.resources.bandwidth_used_mbps
            ));
        }

        details.push_str("\nBottleneck Scores:\n");
        for (btype, score) in scores {
            details.push_str(&format!("  {:20} {:.2}\n", format!("{}:", btype), score));
        }

        details
    }

    /// Generate specific recommendations based on metrics.
    fn generate_recommendations(
        &self,
        metrics: &FrameMetrics,
        primary: BottleneckType,
        secondary: Option<BottleneckType>,
    ) -> Vec<String> {
        let mut recommendations = Vec::new();

        // Add generic hints from bottleneck type
        for hint in primary.optimization_hints().iter().take(3) {
            recommendations.push(hint.to_string());
        }

        // Add specific recommendations based on metrics
        if metrics.state.draw_calls > 2000 {
            recommendations.push(format!(
                "High draw call count ({}). Consider GPU-driven rendering or instancing.",
                metrics.state.draw_calls
            ));
        }

        if metrics.state.pipelines_per_draw() > 0.5 {
            recommendations.push(
                "Too many pipeline switches. Sort draws by pipeline state.".to_string(),
            );
        }

        if metrics.timing.gpu_wait_time.as_millis() > 2 {
            recommendations.push(format!(
                "GPU wait time is {:.1}ms. Consider async uploads or triple buffering.",
                metrics.timing.gpu_wait_time.as_secs_f32() * 1000.0
            ));
        }

        if let Some(sec) = secondary {
            recommendations.push(format!("Also consider addressing {} bottleneck.", sec));
        }

        recommendations
    }

    /// Analyze trends over the entire history.
    #[must_use]
    pub fn analyze_trend(&self) -> TrendAnalysis {
        if self.history.is_empty() {
            return TrendAnalysis::empty();
        }

        let samples = self.history.len();

        // Analyze each frame to get bottleneck types
        let analyses: Vec<(BottleneckType, f64)> = self
            .history
            .iter()
            .map(|m| {
                let result = self.analyze_frame(m);
                let frame_time =
                    m.timing.cpu_frame_time_ms().max(m.timing.gpu_frame_time_ms());
                (result.primary, frame_time)
            })
            .collect();

        // Count bottleneck types
        let mut type_counts = std::collections::HashMap::new();
        for (btype, _) in &analyses {
            *type_counts.entry(*btype).or_insert(0) += 1;
        }

        // Find most common
        let avg_bottleneck = *type_counts
            .iter()
            .max_by_key(|(_, count)| *count)
            .map(|(btype, _)| btype)
            .unwrap_or(&BottleneckType::Balanced);

        // Calculate stability
        let max_count = *type_counts.values().max().unwrap_or(&0);
        let bottleneck_stability = max_count as f32 / samples as f32;

        // Calculate frame time statistics
        let frame_times: Vec<f64> = analyses.iter().map(|(_, t)| *t).collect();
        let avg_frame_time_ms = frame_times.iter().sum::<f64>() / samples as f64;

        let variance: f64 = frame_times
            .iter()
            .map(|t| (t - avg_frame_time_ms).powi(2))
            .sum::<f64>()
            / samples as f64;
        let frame_time_stddev_ms = variance.sqrt();

        // Detect trend direction
        let first_half_avg: f64 = frame_times.iter().take(samples / 2).sum::<f64>()
            / (samples / 2).max(1) as f64;
        let second_half_avg: f64 = frame_times.iter().skip(samples / 2).sum::<f64>()
            / (samples - samples / 2).max(1) as f64;

        let improving = second_half_avg < first_half_avg * 0.9;
        let degrading = second_half_avg > first_half_avg * 1.1;

        // Detect spikes (>2 stddev from mean)
        let spike_threshold = avg_frame_time_ms + 2.0 * frame_time_stddev_ms;
        let spikes: Vec<(u64, BottleneckType)> = self
            .history
            .iter()
            .zip(analyses.iter())
            .filter(|(m, (_, frame_time))| {
                *frame_time > spike_threshold && m.frame_number > 0
            })
            .map(|(m, (btype, _))| (m.frame_number, *btype))
            .collect();

        TrendAnalysis {
            samples,
            avg_bottleneck,
            bottleneck_stability,
            improving,
            degrading,
            spikes,
            avg_frame_time_ms,
            frame_time_stddev_ms,
        }
    }

    /// Clear all recorded history.
    pub fn clear(&mut self) {
        self.history.clear();
        self.last_result = None;
    }

    /// Get the last analysis result (if any).
    #[must_use]
    pub fn last_result(&self) -> Option<&BottleneckResult> {
        self.last_result.as_ref()
    }

    /// Get the history as a slice.
    #[must_use]
    pub fn history(&self) -> &VecDeque<FrameMetrics> {
        &self.history
    }
}

impl Default for BottleneckAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for BottleneckAnalyzer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BottleneckAnalyzer")
            .field("history_size", &self.history_size)
            .field("frames_recorded", &self.history.len())
            .field("thresholds", &self.thresholds)
            .finish()
    }
}

// ============================================================================
// StateChangeType (for BottleneckProfiler)
// ============================================================================

/// Types of state changes that can be recorded.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StateChangeType {
    /// Pipeline switch.
    Pipeline,
    /// Bind group change.
    BindGroup,
    /// Vertex buffer change.
    VertexBuffer,
    /// Index buffer change.
    IndexBuffer,
    /// Draw call.
    DrawCall,
}

// ============================================================================
// BottleneckProfiler
// ============================================================================

/// Real-time bottleneck profiler for integration with render loops.
///
/// Provides a simple API for recording metrics during rendering and
/// automatically generates bottleneck analysis at frame boundaries.
pub struct BottleneckProfiler {
    /// The underlying analyzer.
    analyzer: BottleneckAnalyzer,

    /// Whether to automatically log results.
    auto_log: bool,

    /// Minimum severity to log.
    log_threshold: BottleneckSeverity,

    /// Current frame being built.
    current_frame: FrameMetrics,

    /// Whether we're inside a frame.
    in_frame: bool,

    /// Frame start time.
    frame_start: Instant,

    /// CPU work start time.
    cpu_start: Option<Instant>,
}

impl BottleneckProfiler {
    /// Create a new bottleneck profiler.
    #[must_use]
    pub fn new() -> Self {
        Self {
            analyzer: BottleneckAnalyzer::new(),
            auto_log: false,
            log_threshold: BottleneckSeverity::Medium,
            current_frame: FrameMetrics::default(),
            in_frame: false,
            frame_start: Instant::now(),
            cpu_start: None,
        }
    }

    /// Create a profiler with custom analyzer settings.
    #[must_use]
    pub fn with_analyzer(analyzer: BottleneckAnalyzer) -> Self {
        Self {
            analyzer,
            auto_log: false,
            log_threshold: BottleneckSeverity::Medium,
            current_frame: FrameMetrics::default(),
            in_frame: false,
            frame_start: Instant::now(),
            cpu_start: None,
        }
    }

    /// Enable automatic logging of bottleneck results.
    pub fn enable_auto_log(&mut self) {
        self.auto_log = true;
    }

    /// Disable automatic logging.
    pub fn disable_auto_log(&mut self) {
        self.auto_log = false;
    }

    /// Set the minimum severity level for logging.
    pub fn set_log_threshold(&mut self, threshold: BottleneckSeverity) {
        self.log_threshold = threshold;
    }

    /// Get a reference to the underlying analyzer.
    #[must_use]
    pub fn analyzer(&self) -> &BottleneckAnalyzer {
        &self.analyzer
    }

    /// Get a mutable reference to the underlying analyzer.
    pub fn analyzer_mut(&mut self) -> &mut BottleneckAnalyzer {
        &mut self.analyzer
    }

    /// Begin a new frame.
    ///
    /// Call this at the start of each frame.
    pub fn begin_frame(&mut self) {
        let frame_number = if let Some(last) = self.analyzer.history.back() {
            last.frame_number + 1
        } else {
            0
        };

        self.current_frame = FrameMetrics {
            timing: TimingMetrics::default(),
            resources: ResourceMetrics::default(),
            state: StateMetrics::default(),
            frame_number,
            timestamp: Instant::now(),
        };
        self.frame_start = Instant::now();
        self.in_frame = true;
        self.cpu_start = Some(Instant::now());
    }

    /// Record CPU time for the current frame.
    pub fn record_cpu_time(&mut self, duration: Duration) {
        if self.in_frame {
            self.current_frame.timing.cpu_frame_time = duration;
        }
    }

    /// Record GPU time for the current frame.
    pub fn record_gpu_time(&mut self, duration: Duration) {
        if self.in_frame {
            self.current_frame.timing.gpu_frame_time = duration;
        }
    }

    /// Record CPU submit time.
    pub fn record_submit_time(&mut self, duration: Duration) {
        if self.in_frame {
            self.current_frame.timing.cpu_submit_time = duration;
        }
    }

    /// Record GPU wait time.
    pub fn record_gpu_wait(&mut self, duration: Duration) {
        if self.in_frame {
            self.current_frame.timing.gpu_wait_time = duration;
        }
    }

    /// Record present wait time.
    pub fn record_present_wait(&mut self, duration: Duration) {
        if self.in_frame {
            self.current_frame.timing.present_wait_time = duration;
        }
    }

    /// Record a state change.
    pub fn record_state_change(&mut self, change_type: StateChangeType) {
        if !self.in_frame {
            return;
        }

        match change_type {
            StateChangeType::Pipeline => self.current_frame.state.pipeline_switches += 1,
            StateChangeType::BindGroup => self.current_frame.state.bind_group_changes += 1,
            StateChangeType::VertexBuffer => self.current_frame.state.vertex_buffer_changes += 1,
            StateChangeType::IndexBuffer => self.current_frame.state.index_buffer_changes += 1,
            StateChangeType::DrawCall => self.current_frame.state.draw_calls += 1,
        }
    }

    /// Record bandwidth usage.
    pub fn record_bandwidth(&mut self, bytes: u64) {
        if self.in_frame {
            self.current_frame.resources.buffer_uploads_bytes += bytes;
        }
    }

    /// Record texture upload.
    pub fn record_texture_upload(&mut self, bytes: u64) {
        if self.in_frame {
            self.current_frame.resources.texture_uploads_bytes += bytes;
        }
    }

    /// Record readback operation.
    pub fn record_readback(&mut self, bytes: u64) {
        if self.in_frame {
            self.current_frame.resources.readback_bytes += bytes;
        }
    }

    /// End the current frame and return analysis result.
    ///
    /// Returns `None` if no frame was started.
    pub fn end_frame(&mut self) -> Option<BottleneckResult> {
        if !self.in_frame {
            return None;
        }

        self.in_frame = false;

        // Calculate bandwidth if we have timing
        let frame_time_secs = self.frame_start.elapsed().as_secs_f64();
        if frame_time_secs > 0.0 {
            let total_bytes = self.current_frame.resources.total_bytes();
            let mbps = (total_bytes as f64 / frame_time_secs / 1_000_000.0) as u32;
            self.current_frame.resources.bandwidth_used_mbps = mbps;
        }

        // If no CPU time was recorded, use elapsed time
        if self.current_frame.timing.cpu_frame_time == Duration::ZERO {
            if let Some(start) = self.cpu_start {
                self.current_frame.timing.cpu_frame_time = start.elapsed();
            }
        }

        // Record the frame
        self.analyzer.record_frame(self.current_frame.clone());

        // Analyze
        let result = self.analyzer.analyze_current();

        // Auto-log if enabled and above threshold
        if self.auto_log && result.severity() >= self.log_threshold {
            eprintln!(
                "[Bottleneck] Frame {}: {} ({})",
                self.current_frame.frame_number,
                result.primary,
                result.severity()
            );
        }

        Some(result)
    }

    /// Get the current frame number.
    #[must_use]
    pub fn current_frame_number(&self) -> u64 {
        self.current_frame.frame_number
    }

    /// Check if currently inside a frame.
    #[must_use]
    pub fn is_in_frame(&self) -> bool {
        self.in_frame
    }
}

impl Default for BottleneckProfiler {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for BottleneckProfiler {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BottleneckProfiler")
            .field("in_frame", &self.in_frame)
            .field("current_frame", &self.current_frame.frame_number)
            .field("auto_log", &self.auto_log)
            .field("log_threshold", &self.log_threshold)
            .finish()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // BottleneckType Tests
    // ========================================================================

    #[test]
    fn test_bottleneck_type_severity() {
        assert_eq!(BottleneckType::Balanced.severity(), BottleneckSeverity::None);
        assert_eq!(BottleneckType::CpuBound.severity(), BottleneckSeverity::Medium);
        assert_eq!(BottleneckType::Synchronization.severity(), BottleneckSeverity::High);
    }

    #[test]
    fn test_bottleneck_type_hints() {
        let hints = BottleneckType::CpuBound.optimization_hints();
        assert!(!hints.is_empty());
        assert!(hints.iter().any(|h| h.contains("draw call")));
    }

    #[test]
    fn test_bottleneck_type_description() {
        assert!(BottleneckType::GpuBound.description().contains("GPU"));
        assert!(BottleneckType::CpuBound.description().contains("CPU"));
    }

    #[test]
    fn test_bottleneck_type_display() {
        assert_eq!(format!("{}", BottleneckType::CpuBound), "CPU Bound");
        assert_eq!(format!("{}", BottleneckType::GpuBound), "GPU Bound");
    }

    // ========================================================================
    // BottleneckSeverity Tests
    // ========================================================================

    #[test]
    fn test_severity_from_score() {
        assert_eq!(BottleneckSeverity::from_score(0.0), BottleneckSeverity::None);
        assert_eq!(BottleneckSeverity::from_score(0.2), BottleneckSeverity::Low);
        assert_eq!(BottleneckSeverity::from_score(0.4), BottleneckSeverity::Medium);
        assert_eq!(BottleneckSeverity::from_score(0.6), BottleneckSeverity::High);
        assert_eq!(BottleneckSeverity::from_score(1.0), BottleneckSeverity::Critical);
    }

    #[test]
    fn test_severity_to_score() {
        assert_eq!(BottleneckSeverity::None.to_score(), 0.0);
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
    fn test_severity_display_color() {
        assert!(BottleneckSeverity::None.display_color().contains("32")); // Green
        assert!(BottleneckSeverity::Critical.display_color().contains("91")); // Bright red
    }

    // ========================================================================
    // TimingMetrics Tests
    // ========================================================================

    #[test]
    fn test_timing_metrics_new() {
        let timing = TimingMetrics::new(
            Duration::from_millis(5),
            Duration::from_millis(8),
            Duration::from_micros(500),
            Duration::from_micros(100),
            Duration::from_millis(4),
        );
        assert_eq!(timing.cpu_frame_time, Duration::from_millis(5));
        assert_eq!(timing.gpu_frame_time, Duration::from_millis(8));
    }

    #[test]
    fn test_timing_metrics_frame_overlap() {
        let timing = TimingMetrics {
            cpu_frame_time: Duration::from_millis(8),
            gpu_frame_time: Duration::from_millis(8),
            cpu_submit_time: Duration::ZERO,
            gpu_wait_time: Duration::ZERO,
            present_wait_time: Duration::ZERO,
        };
        let overlap = timing.frame_overlap();
        assert!(overlap > 0.9, "Expected high overlap, got {}", overlap);
    }

    #[test]
    fn test_timing_metrics_gpu_utilization() {
        let timing = TimingMetrics {
            cpu_frame_time: Duration::from_millis(10),
            gpu_frame_time: Duration::from_millis(8),
            cpu_submit_time: Duration::ZERO,
            gpu_wait_time: Duration::ZERO,
            present_wait_time: Duration::ZERO,
        };
        let util = timing.gpu_utilization();
        assert!(util > 0.0 && util <= 1.0);
    }

    #[test]
    fn test_timing_metrics_is_cpu_bound() {
        let timing = TimingMetrics {
            cpu_frame_time: Duration::from_millis(16),
            gpu_frame_time: Duration::from_millis(5),
            cpu_submit_time: Duration::ZERO,
            gpu_wait_time: Duration::ZERO,
            present_wait_time: Duration::ZERO,
        };
        assert!(timing.is_cpu_bound(2.0));
        assert!(!timing.is_gpu_bound(2.0));
    }

    #[test]
    fn test_timing_metrics_is_gpu_bound() {
        let timing = TimingMetrics {
            cpu_frame_time: Duration::from_millis(5),
            gpu_frame_time: Duration::from_millis(16),
            cpu_submit_time: Duration::ZERO,
            gpu_wait_time: Duration::ZERO,
            present_wait_time: Duration::ZERO,
        };
        assert!(timing.is_gpu_bound(2.0));
        assert!(!timing.is_cpu_bound(2.0));
    }

    #[test]
    fn test_timing_metrics_has_sync_stalls() {
        let timing = TimingMetrics {
            cpu_frame_time: Duration::from_millis(8),
            gpu_frame_time: Duration::from_millis(8),
            cpu_submit_time: Duration::ZERO,
            gpu_wait_time: Duration::from_millis(5),
            present_wait_time: Duration::ZERO,
        };
        assert!(timing.has_sync_stalls(1.0));
        assert!(!timing.has_sync_stalls(10.0));
    }

    // ========================================================================
    // ResourceMetrics Tests
    // ========================================================================

    #[test]
    fn test_resource_metrics_total_bytes() {
        let resources = ResourceMetrics {
            texture_uploads_bytes: 1000,
            buffer_uploads_bytes: 2000,
            readback_bytes: 500,
            bandwidth_used_mbps: 100,
            cache_misses: 10,
        };
        assert_eq!(resources.total_bytes(), 3500);
    }

    #[test]
    fn test_resource_metrics_is_bandwidth_limited() {
        let resources = ResourceMetrics {
            texture_uploads_bytes: 0,
            buffer_uploads_bytes: 0,
            readback_bytes: 0,
            bandwidth_used_mbps: 15000,
            cache_misses: 0,
        };
        assert!(resources.is_bandwidth_limited(10000));
        assert!(!resources.is_bandwidth_limited(20000));
    }

    #[test]
    fn test_resource_metrics_bandwidth_utilization() {
        let resources = ResourceMetrics {
            texture_uploads_bytes: 0,
            buffer_uploads_bytes: 0,
            readback_bytes: 0,
            bandwidth_used_mbps: 5000,
            cache_misses: 0,
        };
        let util = resources.bandwidth_utilization(10000);
        assert!((util - 0.5).abs() < 0.01);
    }

    // ========================================================================
    // StateMetrics Tests
    // ========================================================================

    #[test]
    fn test_state_metrics_total_state_changes() {
        let state = StateMetrics {
            pipeline_switches: 10,
            bind_group_changes: 20,
            vertex_buffer_changes: 30,
            index_buffer_changes: 40,
            draw_calls: 100,
        };
        assert_eq!(state.total_state_changes(), 100);
    }

    #[test]
    fn test_state_metrics_state_changes_per_draw() {
        let state = StateMetrics {
            pipeline_switches: 50,
            bind_group_changes: 100,
            vertex_buffer_changes: 50,
            index_buffer_changes: 50,
            draw_calls: 100,
        };
        assert!((state.state_changes_per_draw() - 2.5).abs() < 0.01);
    }

    #[test]
    fn test_state_metrics_is_state_thrashing() {
        let state = StateMetrics {
            pipeline_switches: 200,
            bind_group_changes: 400,
            vertex_buffer_changes: 200,
            index_buffer_changes: 200,
            draw_calls: 100,
        };
        assert!(state.is_state_thrashing(2.0));
        assert!(!state.is_state_thrashing(20.0));
    }

    #[test]
    fn test_state_metrics_pipelines_per_draw() {
        let state = StateMetrics {
            pipeline_switches: 50,
            bind_group_changes: 0,
            vertex_buffer_changes: 0,
            index_buffer_changes: 0,
            draw_calls: 100,
        };
        assert!((state.pipelines_per_draw() - 0.5).abs() < 0.01);
    }

    // ========================================================================
    // FrameMetrics Tests
    // ========================================================================

    #[test]
    fn test_frame_metrics_new() {
        let frame = FrameMetrics::new(
            TimingMetrics::default(),
            ResourceMetrics::default(),
            StateMetrics::default(),
            42,
        );
        assert_eq!(frame.frame_number, 42);
    }

    #[test]
    fn test_frame_metrics_age() {
        let frame = FrameMetrics::default();
        // Age should be very small
        assert!(frame.age_secs() < 1.0);
    }

    // ========================================================================
    // AnalysisThresholds Tests
    // ========================================================================

    #[test]
    fn test_thresholds_default() {
        let thresholds = AnalysisThresholds::default();
        assert!(thresholds.gpu_bound_ratio > 1.0);
        assert!(thresholds.cpu_bound_ratio > 1.0);
    }

    #[test]
    fn test_thresholds_aggressive() {
        let aggressive = AnalysisThresholds::aggressive();
        let default = AnalysisThresholds::default();
        assert!(aggressive.state_thrash_per_draw < default.state_thrash_per_draw);
    }

    #[test]
    fn test_thresholds_relaxed() {
        let relaxed = AnalysisThresholds::relaxed();
        let default = AnalysisThresholds::default();
        assert!(relaxed.state_thrash_per_draw > default.state_thrash_per_draw);
    }

    // ========================================================================
    // BottleneckResult Tests
    // ========================================================================

    #[test]
    fn test_bottleneck_result_new() {
        let result = BottleneckResult::new(BottleneckType::GpuBound, FrameMetrics::default());
        assert_eq!(result.primary, BottleneckType::GpuBound);
        assert!(!result.recommendations.is_empty());
    }

    #[test]
    fn test_bottleneck_result_balanced() {
        let result = BottleneckResult::balanced(FrameMetrics::default());
        assert_eq!(result.primary, BottleneckType::Balanced);
        assert!(!result.is_significant());
    }

    #[test]
    fn test_bottleneck_result_is_significant() {
        let mut result = BottleneckResult::new(BottleneckType::CpuBound, FrameMetrics::default());
        result.confidence = 0.8;
        assert!(result.is_significant());

        result.confidence = 0.3;
        assert!(!result.is_significant());
    }

    #[test]
    fn test_bottleneck_result_display() {
        let result = BottleneckResult::new(BottleneckType::GpuBound, FrameMetrics::default());
        let s = format!("{}", result);
        assert!(s.contains("GPU Bound"));
    }

    // ========================================================================
    // TrendAnalysis Tests
    // ========================================================================

    #[test]
    fn test_trend_analysis_empty() {
        let trend = TrendAnalysis::empty();
        assert_eq!(trend.samples, 0);
        assert!(!trend.has_sufficient_data());
    }

    #[test]
    fn test_trend_analysis_jitter() {
        let mut trend = TrendAnalysis::empty();
        trend.avg_frame_time_ms = 16.0;
        trend.frame_time_stddev_ms = 1.6;
        assert!((trend.jitter() - 0.1).abs() < 0.01);
    }

    #[test]
    fn test_trend_analysis_is_stable() {
        let mut trend = TrendAnalysis::empty();
        trend.avg_frame_time_ms = 16.0;
        trend.frame_time_stddev_ms = 1.0;
        assert!(trend.is_stable());

        trend.frame_time_stddev_ms = 5.0;
        assert!(!trend.is_stable());
    }

    // ========================================================================
    // BottleneckAnalyzer Tests
    // ========================================================================

    #[test]
    fn test_analyzer_new() {
        let analyzer = BottleneckAnalyzer::new();
        assert_eq!(analyzer.frame_count(), 0);
        assert_eq!(analyzer.history_size(), BottleneckAnalyzer::DEFAULT_HISTORY_SIZE);
    }

    #[test]
    fn test_analyzer_with_thresholds() {
        let thresholds = AnalysisThresholds::aggressive();
        let analyzer = BottleneckAnalyzer::with_thresholds(thresholds);
        assert_eq!(analyzer.thresholds().state_thrash_per_draw, thresholds.state_thrash_per_draw);
    }

    #[test]
    fn test_analyzer_record_frame() {
        let mut analyzer = BottleneckAnalyzer::new();
        analyzer.record_frame(FrameMetrics::default());
        assert_eq!(analyzer.frame_count(), 1);
    }

    #[test]
    fn test_analyzer_history_limit() {
        let mut analyzer = BottleneckAnalyzer::with_history_size(5);
        for i in 0..10 {
            let mut frame = FrameMetrics::default();
            frame.frame_number = i;
            analyzer.record_frame(frame);
        }
        assert_eq!(analyzer.frame_count(), 5);
        // Should have frames 5-9
        assert_eq!(analyzer.history().front().unwrap().frame_number, 5);
    }

    #[test]
    fn test_analyzer_analyze_gpu_bound() {
        let mut analyzer = BottleneckAnalyzer::new();
        let metrics = FrameMetrics {
            timing: TimingMetrics {
                cpu_frame_time: Duration::from_millis(2),
                gpu_frame_time: Duration::from_millis(12),
                cpu_submit_time: Duration::ZERO,
                gpu_wait_time: Duration::ZERO,
                present_wait_time: Duration::ZERO,
            },
            resources: ResourceMetrics::default(),
            state: StateMetrics::default(),
            frame_number: 0,
            timestamp: Instant::now(),
        };

        analyzer.record_frame(metrics);
        let result = analyzer.analyze_current();
        assert_eq!(result.primary, BottleneckType::GpuBound);
    }

    #[test]
    fn test_analyzer_analyze_cpu_bound() {
        let mut analyzer = BottleneckAnalyzer::new();
        let metrics = FrameMetrics {
            timing: TimingMetrics {
                cpu_frame_time: Duration::from_millis(12),
                gpu_frame_time: Duration::from_millis(2),
                cpu_submit_time: Duration::ZERO,
                gpu_wait_time: Duration::ZERO,
                present_wait_time: Duration::ZERO,
            },
            resources: ResourceMetrics::default(),
            state: StateMetrics {
                draw_calls: 5000,
                ..StateMetrics::default()
            },
            frame_number: 0,
            timestamp: Instant::now(),
        };

        analyzer.record_frame(metrics);
        let result = analyzer.analyze_current();
        assert_eq!(result.primary, BottleneckType::CpuBound);
    }

    #[test]
    fn test_analyzer_analyze_state_thrashing() {
        let mut analyzer = BottleneckAnalyzer::new();
        let metrics = FrameMetrics {
            timing: TimingMetrics {
                cpu_frame_time: Duration::from_millis(8),
                gpu_frame_time: Duration::from_millis(8),
                cpu_submit_time: Duration::ZERO,
                gpu_wait_time: Duration::ZERO,
                present_wait_time: Duration::ZERO,
            },
            resources: ResourceMetrics::default(),
            state: StateMetrics {
                pipeline_switches: 500,
                bind_group_changes: 1000,
                vertex_buffer_changes: 500,
                index_buffer_changes: 500,
                draw_calls: 500,
            },
            frame_number: 0,
            timestamp: Instant::now(),
        };

        analyzer.record_frame(metrics);
        let result = analyzer.analyze_current();
        assert_eq!(result.primary, BottleneckType::StateThrashing);
    }

    #[test]
    fn test_analyzer_analyze_trend() {
        let mut analyzer = BottleneckAnalyzer::new();

        // Add consistent GPU-bound frames
        for i in 0..20 {
            let metrics = FrameMetrics {
                timing: TimingMetrics {
                    cpu_frame_time: Duration::from_millis(2),
                    gpu_frame_time: Duration::from_millis(12),
                    cpu_submit_time: Duration::ZERO,
                    gpu_wait_time: Duration::ZERO,
                    present_wait_time: Duration::ZERO,
                },
                resources: ResourceMetrics::default(),
                state: StateMetrics::default(),
                frame_number: i,
                timestamp: Instant::now(),
            };
            analyzer.record_frame(metrics);
        }

        let trend = analyzer.analyze_trend();
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

    // ========================================================================
    // BottleneckProfiler Tests
    // ========================================================================

    #[test]
    fn test_profiler_new() {
        let profiler = BottleneckProfiler::new();
        assert!(!profiler.is_in_frame());
    }

    #[test]
    fn test_profiler_begin_end_frame() {
        let mut profiler = BottleneckProfiler::new();

        profiler.begin_frame();
        assert!(profiler.is_in_frame());

        profiler.record_cpu_time(Duration::from_millis(5));
        profiler.record_gpu_time(Duration::from_millis(8));
        profiler.record_state_change(StateChangeType::DrawCall);

        let result = profiler.end_frame();
        assert!(result.is_some());
        assert!(!profiler.is_in_frame());
    }

    #[test]
    fn test_profiler_record_state_changes() {
        let mut profiler = BottleneckProfiler::new();
        profiler.begin_frame();

        profiler.record_state_change(StateChangeType::Pipeline);
        profiler.record_state_change(StateChangeType::BindGroup);
        profiler.record_state_change(StateChangeType::BindGroup);
        profiler.record_state_change(StateChangeType::DrawCall);

        let result = profiler.end_frame().unwrap();
        assert_eq!(result.metrics.state.pipeline_switches, 1);
        assert_eq!(result.metrics.state.bind_group_changes, 2);
        assert_eq!(result.metrics.state.draw_calls, 1);
    }

    #[test]
    fn test_profiler_record_bandwidth() {
        let mut profiler = BottleneckProfiler::new();
        profiler.begin_frame();

        profiler.record_bandwidth(1024);
        profiler.record_texture_upload(2048);
        profiler.record_readback(512);

        let result = profiler.end_frame().unwrap();
        assert_eq!(result.metrics.resources.buffer_uploads_bytes, 1024);
        assert_eq!(result.metrics.resources.texture_uploads_bytes, 2048);
        assert_eq!(result.metrics.resources.readback_bytes, 512);
    }

    #[test]
    fn test_profiler_auto_log() {
        let mut profiler = BottleneckProfiler::new();
        profiler.enable_auto_log();
        profiler.set_log_threshold(BottleneckSeverity::Critical);

        profiler.begin_frame();
        // Should not log since default metrics produce no critical bottleneck
        let _ = profiler.end_frame();
    }

    #[test]
    fn test_profiler_current_frame_number() {
        let mut profiler = BottleneckProfiler::new();

        profiler.begin_frame();
        profiler.end_frame();

        profiler.begin_frame();
        assert_eq!(profiler.current_frame_number(), 1);
    }

    // ========================================================================
    // StateChangeType Tests
    // ========================================================================

    #[test]
    fn test_state_change_type_equality() {
        assert_eq!(StateChangeType::Pipeline, StateChangeType::Pipeline);
        assert_ne!(StateChangeType::Pipeline, StateChangeType::BindGroup);
    }

    // ========================================================================
    // Edge Case Tests
    // ========================================================================

    #[test]
    fn test_timing_metrics_zero_values() {
        let timing = TimingMetrics::default();
        assert_eq!(timing.frame_overlap(), 0.0);
        assert_eq!(timing.gpu_utilization(), 0.0);
        assert!(!timing.is_cpu_bound(1.5));
        assert!(!timing.is_gpu_bound(1.5));
    }

    #[test]
    fn test_state_metrics_zero_draw_calls() {
        let state = StateMetrics::default();
        assert_eq!(state.state_changes_per_draw(), 0.0);
        assert!(!state.is_state_thrashing(2.0));
    }

    #[test]
    fn test_analyzer_empty_analysis() {
        let mut analyzer = BottleneckAnalyzer::new();
        let result = analyzer.analyze_current();
        assert_eq!(result.primary, BottleneckType::Balanced);
    }

    #[test]
    fn test_profiler_end_without_begin() {
        let mut profiler = BottleneckProfiler::new();
        assert!(profiler.end_frame().is_none());
    }

    #[test]
    fn test_profiler_record_without_frame() {
        let mut profiler = BottleneckProfiler::new();
        // These should be no-ops
        profiler.record_cpu_time(Duration::from_millis(5));
        profiler.record_state_change(StateChangeType::DrawCall);
        profiler.record_bandwidth(1024);
    }

    // ========================================================================
    // Send + Sync Tests
    // ========================================================================

    fn assert_send<T: Send>() {}
    fn assert_sync<T: Sync>() {}

    #[test]
    fn test_bottleneck_type_is_send_sync() {
        assert_send::<BottleneckType>();
        assert_sync::<BottleneckType>();
    }

    #[test]
    fn test_bottleneck_severity_is_send_sync() {
        assert_send::<BottleneckSeverity>();
        assert_sync::<BottleneckSeverity>();
    }

    #[test]
    fn test_timing_metrics_is_send_sync() {
        assert_send::<TimingMetrics>();
        assert_sync::<TimingMetrics>();
    }

    #[test]
    fn test_bottleneck_result_is_send_sync() {
        assert_send::<BottleneckResult>();
        assert_sync::<BottleneckResult>();
    }

    #[test]
    fn test_bottleneck_analyzer_is_send() {
        assert_send::<BottleneckAnalyzer>();
    }
}

// ============================================================================
// Simplified API (T-WGPU-P7.4.5)
// ============================================================================
// The types below provide a simplified interface as specified in the task
// requirements. They wrap or complement the more comprehensive types above.

/// Simplified bottleneck type enumeration for performance tuning.
///
/// This enum provides a more granular categorization of GPU bottlenecks
/// compared to [`BottleneckType`], with specific focus on individual
/// pipeline stages.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SimpleBottleneckType {
    /// CPU is limiting performance.
    CpuBound,
    /// GPU is limiting performance (general).
    GpuBound,
    /// Vertex processing is the bottleneck (vertex shaders, geometry).
    VertexProcessing,
    /// Fragment/pixel processing is the bottleneck.
    FragmentProcessing,
    /// GPU memory bandwidth or capacity limitation.
    Memory,
    /// Bus or memory bandwidth limitation.
    Bandwidth,
    /// Fill rate (pixel output) limitation.
    Fillrate,
    /// Too many draw calls causing CPU overhead.
    DrawCalls,
    /// Frequent state changes causing overhead.
    StateChanges,
    /// Texture sampling is limiting performance.
    TextureSampling,
    /// Unable to determine bottleneck.
    Unknown,
}

impl SimpleBottleneckType {
    /// Get the name of this bottleneck type.
    #[must_use]
    pub fn name(&self) -> &'static str {
        match self {
            SimpleBottleneckType::CpuBound => "CPU Bound",
            SimpleBottleneckType::GpuBound => "GPU Bound",
            SimpleBottleneckType::VertexProcessing => "Vertex Processing",
            SimpleBottleneckType::FragmentProcessing => "Fragment Processing",
            SimpleBottleneckType::Memory => "Memory",
            SimpleBottleneckType::Bandwidth => "Bandwidth",
            SimpleBottleneckType::Fillrate => "Fill Rate",
            SimpleBottleneckType::DrawCalls => "Draw Calls",
            SimpleBottleneckType::StateChanges => "State Changes",
            SimpleBottleneckType::TextureSampling => "Texture Sampling",
            SimpleBottleneckType::Unknown => "Unknown",
        }
    }

    /// Get the category this bottleneck belongs to.
    #[must_use]
    pub fn category(&self) -> &'static str {
        match self {
            SimpleBottleneckType::CpuBound | SimpleBottleneckType::DrawCalls | SimpleBottleneckType::StateChanges => {
                "CPU"
            }
            SimpleBottleneckType::GpuBound
            | SimpleBottleneckType::VertexProcessing
            | SimpleBottleneckType::FragmentProcessing
            | SimpleBottleneckType::Fillrate
            | SimpleBottleneckType::TextureSampling => "GPU",
            SimpleBottleneckType::Memory | SimpleBottleneckType::Bandwidth => "Memory",
            SimpleBottleneckType::Unknown => "Unknown",
        }
    }

    /// Get the typical fix for this bottleneck type.
    #[must_use]
    pub fn typical_fix(&self) -> &'static str {
        match self {
            SimpleBottleneckType::CpuBound => "Reduce CPU work, move computations to GPU",
            SimpleBottleneckType::GpuBound => "Optimize shaders, reduce resolution, use LOD",
            SimpleBottleneckType::VertexProcessing => "Reduce vertex count, use LOD, simplify vertex shaders",
            SimpleBottleneckType::FragmentProcessing => "Reduce overdraw, simplify fragment shaders, lower resolution",
            SimpleBottleneckType::Memory => "Reduce texture sizes, compress textures, pool allocations",
            SimpleBottleneckType::Bandwidth => "Compress data, reduce transfers, use texture streaming",
            SimpleBottleneckType::Fillrate => "Reduce overdraw, use depth pre-pass, lower resolution",
            SimpleBottleneckType::DrawCalls => "Batch draw calls, use instancing, GPU-driven rendering",
            SimpleBottleneckType::StateChanges => "Sort by state, merge materials, use texture arrays",
            SimpleBottleneckType::TextureSampling => "Use mipmaps, compress textures, reduce sampling count",
            SimpleBottleneckType::Unknown => "Profile further to identify the bottleneck",
        }
    }

    /// Convert from the comprehensive BottleneckType.
    #[must_use]
    pub fn from_bottleneck_type(bt: BottleneckType) -> Self {
        match bt {
            BottleneckType::CpuBound => SimpleBottleneckType::CpuBound,
            BottleneckType::GpuBound => SimpleBottleneckType::GpuBound,
            BottleneckType::MemoryBandwidth => SimpleBottleneckType::Bandwidth,
            BottleneckType::StateThrashing => SimpleBottleneckType::StateChanges,
            BottleneckType::Synchronization => SimpleBottleneckType::CpuBound,
            BottleneckType::Balanced => SimpleBottleneckType::Unknown,
        }
    }
}

impl Default for SimpleBottleneckType {
    fn default() -> Self {
        SimpleBottleneckType::Unknown
    }
}

impl fmt::Display for SimpleBottleneckType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// A detected bottleneck with severity and metrics.
#[derive(Clone, Debug)]
pub struct Bottleneck {
    /// The type of bottleneck detected.
    pub bottleneck_type: SimpleBottleneckType,
    /// Severity of the bottleneck.
    pub severity: BottleneckSeverity,
    /// Human-readable description.
    pub description: String,
    /// The measured metric value that triggered detection.
    pub metric_value: f64,
    /// The threshold that was exceeded.
    pub threshold: f64,
}

impl Bottleneck {
    /// Create a new bottleneck.
    #[must_use]
    pub fn new(
        bottleneck_type: SimpleBottleneckType,
        severity: BottleneckSeverity,
        description: impl Into<String>,
        metric_value: f64,
        threshold: f64,
    ) -> Self {
        Self {
            bottleneck_type,
            severity,
            description: description.into(),
            metric_value,
            threshold,
        }
    }

    /// Calculate how much the metric exceeds the threshold as a percentage.
    #[must_use]
    pub fn percentage_over_threshold(&self) -> f64 {
        if self.threshold <= 0.0 {
            return 0.0;
        }
        ((self.metric_value - self.threshold) / self.threshold * 100.0).max(0.0)
    }

    /// Check if this is a critical bottleneck.
    #[must_use]
    pub fn is_critical(&self) -> bool {
        self.severity == BottleneckSeverity::Critical
    }

    /// Check if this bottleneck exceeds its threshold.
    #[must_use]
    pub fn exceeds_threshold(&self) -> bool {
        self.metric_value > self.threshold
    }
}

impl fmt::Display for Bottleneck {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[{}] {}: {} (value: {:.2}, threshold: {:.2}, +{:.1}%)",
            self.severity,
            self.bottleneck_type,
            self.description,
            self.metric_value,
            self.threshold,
            self.percentage_over_threshold()
        )
    }
}

/// Result of a bottleneck analysis for a single frame.
#[derive(Clone, Debug)]
pub struct BottleneckAnalysis {
    /// All detected bottlenecks, sorted by severity.
    pub bottlenecks: Vec<Bottleneck>,
    /// The primary (most severe) bottleneck type, if any.
    pub primary_bottleneck: Option<SimpleBottleneckType>,
    /// CPU frame time in milliseconds.
    pub cpu_time_ms: f64,
    /// GPU frame time in milliseconds.
    pub gpu_time_ms: f64,
    /// Total frame time in milliseconds.
    pub frame_time_ms: f64,
}

impl BottleneckAnalysis {
    /// Create a new empty analysis.
    #[must_use]
    pub fn new(cpu_time_ms: f64, gpu_time_ms: f64) -> Self {
        Self {
            bottlenecks: Vec::new(),
            primary_bottleneck: None,
            cpu_time_ms,
            gpu_time_ms,
            frame_time_ms: cpu_time_ms.max(gpu_time_ms),
        }
    }

    /// Check if the frame is CPU-bound.
    #[must_use]
    pub fn is_cpu_bound(&self) -> bool {
        self.cpu_time_ms > self.gpu_time_ms * 1.2
    }

    /// Check if the frame is GPU-bound.
    #[must_use]
    pub fn is_gpu_bound(&self) -> bool {
        self.gpu_time_ms > self.cpu_time_ms * 1.2
    }

    /// Check if there are any critical issues.
    #[must_use]
    pub fn has_critical_issues(&self) -> bool {
        self.bottlenecks
            .iter()
            .any(|b| b.severity == BottleneckSeverity::Critical)
    }

    /// Get the worst (most severe) bottleneck.
    #[must_use]
    pub fn worst_bottleneck(&self) -> Option<&Bottleneck> {
        self.bottlenecks.iter().max_by_key(|b| b.severity)
    }

    /// Get the number of detected bottlenecks.
    #[must_use]
    pub fn bottleneck_count(&self) -> usize {
        self.bottlenecks.len()
    }

    /// Check if there are any bottlenecks detected.
    #[must_use]
    pub fn has_bottlenecks(&self) -> bool {
        !self.bottlenecks.is_empty()
    }

    /// Get bottlenecks filtered by minimum severity.
    #[must_use]
    pub fn bottlenecks_by_severity(&self, min_severity: BottleneckSeverity) -> Vec<&Bottleneck> {
        self.bottlenecks
            .iter()
            .filter(|b| b.severity >= min_severity)
            .collect()
    }
}

impl Default for BottleneckAnalysis {
    fn default() -> Self {
        Self::new(0.0, 0.0)
    }
}

/// Configurable thresholds for bottleneck detection.
#[derive(Clone, Debug)]
pub struct BottleneckThresholds {
    /// Warning threshold for draw call count.
    pub draw_call_warning: u32,
    /// Critical threshold for draw call count.
    pub draw_call_critical: u32,
    /// Warning threshold for vertex processing rate (vertices/ms).
    pub vertex_rate_warning: f64,
    /// Warning threshold for fill rate (pixels/ms).
    pub fillrate_warning: f64,
    /// Ratio threshold for CPU/GPU balance (1.0 = balanced).
    pub cpu_gpu_ratio_balanced: f64,
    /// Warning threshold for state changes per frame.
    pub state_changes_warning: u32,
    /// Warning threshold for bandwidth (MB/frame).
    pub bandwidth_warning_mb: f64,
    /// Critical threshold for bandwidth (MB/frame).
    pub bandwidth_critical_mb: f64,
}

impl BottleneckThresholds {
    /// Create new thresholds with custom values.
    #[must_use]
    pub fn new(
        draw_call_warning: u32,
        draw_call_critical: u32,
        vertex_rate_warning: f64,
        fillrate_warning: f64,
        cpu_gpu_ratio_balanced: f64,
    ) -> Self {
        Self {
            draw_call_warning,
            draw_call_critical,
            vertex_rate_warning,
            fillrate_warning,
            cpu_gpu_ratio_balanced,
            state_changes_warning: 1000,
            bandwidth_warning_mb: 100.0,
            bandwidth_critical_mb: 500.0,
        }
    }

    /// Create default thresholds for high-end hardware.
    #[must_use]
    pub fn high_end() -> Self {
        Self {
            draw_call_warning: 5000,
            draw_call_critical: 10000,
            vertex_rate_warning: 5_000_000.0,
            fillrate_warning: 100_000_000.0,
            cpu_gpu_ratio_balanced: 1.2,
            state_changes_warning: 2000,
            bandwidth_warning_mb: 500.0,
            bandwidth_critical_mb: 1000.0,
        }
    }

    /// Create default thresholds for low-end hardware.
    #[must_use]
    pub fn low_end() -> Self {
        Self {
            draw_call_warning: 500,
            draw_call_critical: 1000,
            vertex_rate_warning: 500_000.0,
            fillrate_warning: 10_000_000.0,
            cpu_gpu_ratio_balanced: 1.5,
            state_changes_warning: 200,
            bandwidth_warning_mb: 50.0,
            bandwidth_critical_mb: 100.0,
        }
    }
}

impl Default for BottleneckThresholds {
    fn default() -> Self {
        Self {
            draw_call_warning: 2000,
            draw_call_critical: 5000,
            vertex_rate_warning: 2_000_000.0,
            fillrate_warning: 50_000_000.0,
            cpu_gpu_ratio_balanced: 1.3,
            state_changes_warning: 500,
            bandwidth_warning_mb: 100.0,
            bandwidth_critical_mb: 300.0,
        }
    }
}

/// Direction of the bottleneck trend.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BottleneckTrend {
    /// Performance is improving.
    Improving,
    /// Performance is stable.
    Stable,
    /// Performance is degrading.
    Degrading,
    /// Not enough data to determine trend.
    Unknown,
}

impl Default for BottleneckTrend {
    fn default() -> Self {
        BottleneckTrend::Unknown
    }
}

impl fmt::Display for BottleneckTrend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BottleneckTrend::Improving => write!(f, "Improving"),
            BottleneckTrend::Stable => write!(f, "Stable"),
            BottleneckTrend::Degrading => write!(f, "Degrading"),
            BottleneckTrend::Unknown => write!(f, "Unknown"),
        }
    }
}

/// Simplified bottleneck analyzer using the straightforward API.
///
/// This provides the exact API specified in T-WGPU-P7.4.5 requirements.
pub struct SimpleBottleneckAnalyzer {
    thresholds: BottleneckThresholds,
    history: Vec<BottleneckAnalysis>,
    history_size: usize,
}

impl SimpleBottleneckAnalyzer {
    /// Default history size for analysis.
    pub const DEFAULT_HISTORY_SIZE: usize = 60;

    /// Create a new analyzer with default thresholds.
    #[must_use]
    pub fn new() -> Self {
        Self {
            thresholds: BottleneckThresholds::default(),
            history: Vec::with_capacity(Self::DEFAULT_HISTORY_SIZE),
            history_size: Self::DEFAULT_HISTORY_SIZE,
        }
    }

    /// Create an analyzer with custom thresholds.
    #[must_use]
    pub fn with_thresholds(thresholds: BottleneckThresholds) -> Self {
        Self {
            thresholds,
            history: Vec::with_capacity(Self::DEFAULT_HISTORY_SIZE),
            history_size: Self::DEFAULT_HISTORY_SIZE,
        }
    }

    /// Analyze a frame's performance data.
    pub fn analyze_frame(
        &mut self,
        cpu_time_ms: f64,
        gpu_time_ms: f64,
        draw_calls: u32,
        vertices: u64,
        pixels: u64,
    ) -> BottleneckAnalysis {
        let mut analysis = BottleneckAnalysis::new(cpu_time_ms, gpu_time_ms);
        let frame_time = cpu_time_ms.max(gpu_time_ms);

        // Analyze CPU/GPU bound
        let cpu_gpu_ratio = if gpu_time_ms > 0.0 {
            cpu_time_ms / gpu_time_ms
        } else {
            1.0
        };

        if cpu_gpu_ratio > self.thresholds.cpu_gpu_ratio_balanced * 1.5 {
            analysis.bottlenecks.push(Bottleneck::new(
                SimpleBottleneckType::CpuBound,
                BottleneckSeverity::High,
                "CPU time significantly exceeds GPU time",
                cpu_gpu_ratio,
                self.thresholds.cpu_gpu_ratio_balanced,
            ));
            analysis.primary_bottleneck = Some(SimpleBottleneckType::CpuBound);
        } else if cpu_gpu_ratio < 1.0 / (self.thresholds.cpu_gpu_ratio_balanced * 1.5) {
            analysis.bottlenecks.push(Bottleneck::new(
                SimpleBottleneckType::GpuBound,
                BottleneckSeverity::High,
                "GPU time significantly exceeds CPU time",
                1.0 / cpu_gpu_ratio,
                self.thresholds.cpu_gpu_ratio_balanced,
            ));
            analysis.primary_bottleneck = Some(SimpleBottleneckType::GpuBound);
        }

        // Analyze draw calls
        if draw_calls >= self.thresholds.draw_call_critical {
            analysis.bottlenecks.push(Bottleneck::new(
                SimpleBottleneckType::DrawCalls,
                BottleneckSeverity::Critical,
                "Draw call count is critically high",
                draw_calls as f64,
                self.thresholds.draw_call_critical as f64,
            ));
            if analysis.primary_bottleneck.is_none() {
                analysis.primary_bottleneck = Some(SimpleBottleneckType::DrawCalls);
            }
        } else if draw_calls >= self.thresholds.draw_call_warning {
            analysis.bottlenecks.push(Bottleneck::new(
                SimpleBottleneckType::DrawCalls,
                BottleneckSeverity::Medium,
                "Draw call count exceeds recommended threshold",
                draw_calls as f64,
                self.thresholds.draw_call_warning as f64,
            ));
        }

        // Analyze vertex processing
        let vertices_per_ms = if frame_time > 0.0 {
            vertices as f64 / frame_time
        } else {
            0.0
        };

        if vertices_per_ms > self.thresholds.vertex_rate_warning {
            analysis.bottlenecks.push(Bottleneck::new(
                SimpleBottleneckType::VertexProcessing,
                BottleneckSeverity::Medium,
                "High vertex processing load",
                vertices_per_ms,
                self.thresholds.vertex_rate_warning,
            ));
        }

        // Analyze fill rate
        let pixels_per_ms = if frame_time > 0.0 {
            pixels as f64 / frame_time
        } else {
            0.0
        };

        if pixels_per_ms > self.thresholds.fillrate_warning {
            analysis.bottlenecks.push(Bottleneck::new(
                SimpleBottleneckType::Fillrate,
                BottleneckSeverity::Medium,
                "High pixel fill rate load",
                pixels_per_ms,
                self.thresholds.fillrate_warning,
            ));
        }

        // Sort bottlenecks by severity (highest first)
        analysis
            .bottlenecks
            .sort_by(|a, b| b.severity.cmp(&a.severity));

        // Store in history
        if self.history.len() >= self.history_size {
            self.history.remove(0);
        }
        self.history.push(analysis.clone());

        analysis
    }

    /// Get the analysis history.
    #[must_use]
    pub fn history(&self) -> &[BottleneckAnalysis] {
        &self.history
    }

    /// Analyze the trend of bottleneck occurrences.
    #[must_use]
    pub fn trend(&self) -> BottleneckTrend {
        if self.history.len() < 10 {
            return BottleneckTrend::Unknown;
        }

        let recent_count = self.history.len().min(20);
        let recent = &self.history[self.history.len() - recent_count..];

        // Calculate average frame times for first and second half
        let half = recent_count / 2;
        let first_avg: f64 =
            recent[..half].iter().map(|a| a.frame_time_ms).sum::<f64>() / half as f64;
        let second_avg: f64 =
            recent[half..].iter().map(|a| a.frame_time_ms).sum::<f64>() / (recent_count - half) as f64;

        let change_ratio = second_avg / first_avg;

        if change_ratio < 0.9 {
            BottleneckTrend::Improving
        } else if change_ratio > 1.1 {
            BottleneckTrend::Degrading
        } else {
            BottleneckTrend::Stable
        }
    }

    /// Generate optimization suggestions based on the analysis.
    #[must_use]
    pub fn suggestions(&self) -> Vec<String> {
        let mut suggestions = Vec::new();

        if let Some(latest) = self.history.last() {
            for bottleneck in &latest.bottlenecks {
                suggestions.push(format!(
                    "{}: {}",
                    bottleneck.bottleneck_type.name(),
                    bottleneck.bottleneck_type.typical_fix()
                ));
            }

            // Add general suggestions based on the primary bottleneck
            if let Some(primary) = latest.primary_bottleneck {
                match primary {
                    SimpleBottleneckType::CpuBound => {
                        suggestions.push("Consider using GPU-driven rendering".to_string());
                        suggestions.push("Profile CPU to find hotspots".to_string());
                    }
                    SimpleBottleneckType::GpuBound => {
                        suggestions.push("Consider reducing shader complexity".to_string());
                        suggestions.push("Use LOD (Level of Detail) for distant objects".to_string());
                    }
                    SimpleBottleneckType::DrawCalls => {
                        suggestions.push("Implement indirect draw calls".to_string());
                        suggestions.push("Merge static geometry into batches".to_string());
                    }
                    _ => {}
                }
            }
        }

        // Deduplicate
        suggestions.sort();
        suggestions.dedup();

        suggestions
    }

    /// Clear all recorded history.
    pub fn reset(&mut self) {
        self.history.clear();
    }

    /// Get the configured thresholds.
    #[must_use]
    pub fn thresholds(&self) -> &BottleneckThresholds {
        &self.thresholds
    }

    /// Set new thresholds.
    pub fn set_thresholds(&mut self, thresholds: BottleneckThresholds) {
        self.thresholds = thresholds;
    }

    /// Get the history size.
    #[must_use]
    pub fn history_size(&self) -> usize {
        self.history_size
    }

    /// Set the history size.
    pub fn set_history_size(&mut self, size: usize) {
        self.history_size = size.max(1);
        while self.history.len() > self.history_size {
            self.history.remove(0);
        }
    }
}

impl Default for SimpleBottleneckAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for SimpleBottleneckAnalyzer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SimpleBottleneckAnalyzer")
            .field("history_len", &self.history.len())
            .field("history_size", &self.history_size)
            .finish()
    }
}

// ============================================================================
// Additional Tests for Simplified API
// ============================================================================

#[cfg(test)]
mod simplified_tests {
    use super::*;

    // ========================================================================
    // SimpleBottleneckType Tests
    // ========================================================================

    #[test]
    fn test_simple_bottleneck_type_all_variants() {
        let variants = [
            SimpleBottleneckType::CpuBound,
            SimpleBottleneckType::GpuBound,
            SimpleBottleneckType::VertexProcessing,
            SimpleBottleneckType::FragmentProcessing,
            SimpleBottleneckType::Memory,
            SimpleBottleneckType::Bandwidth,
            SimpleBottleneckType::Fillrate,
            SimpleBottleneckType::DrawCalls,
            SimpleBottleneckType::StateChanges,
            SimpleBottleneckType::TextureSampling,
            SimpleBottleneckType::Unknown,
        ];

        for variant in variants {
            assert!(!variant.name().is_empty());
            assert!(!variant.category().is_empty());
            assert!(!variant.typical_fix().is_empty());
        }
    }

    #[test]
    fn test_simple_bottleneck_type_name() {
        assert_eq!(SimpleBottleneckType::CpuBound.name(), "CPU Bound");
        assert_eq!(SimpleBottleneckType::GpuBound.name(), "GPU Bound");
        assert_eq!(SimpleBottleneckType::VertexProcessing.name(), "Vertex Processing");
    }

    #[test]
    fn test_simple_bottleneck_type_category() {
        assert_eq!(SimpleBottleneckType::CpuBound.category(), "CPU");
        assert_eq!(SimpleBottleneckType::DrawCalls.category(), "CPU");
        assert_eq!(SimpleBottleneckType::GpuBound.category(), "GPU");
        assert_eq!(SimpleBottleneckType::Memory.category(), "Memory");
    }

    #[test]
    fn test_simple_bottleneck_type_typical_fix() {
        let fix = SimpleBottleneckType::DrawCalls.typical_fix();
        assert!(fix.contains("instancing") || fix.contains("Batch"));
    }

    #[test]
    fn test_simple_bottleneck_type_from_bottleneck_type() {
        assert_eq!(
            SimpleBottleneckType::from_bottleneck_type(BottleneckType::CpuBound),
            SimpleBottleneckType::CpuBound
        );
        assert_eq!(
            SimpleBottleneckType::from_bottleneck_type(BottleneckType::GpuBound),
            SimpleBottleneckType::GpuBound
        );
        assert_eq!(
            SimpleBottleneckType::from_bottleneck_type(BottleneckType::StateThrashing),
            SimpleBottleneckType::StateChanges
        );
    }

    #[test]
    fn test_simple_bottleneck_type_default() {
        assert_eq!(SimpleBottleneckType::default(), SimpleBottleneckType::Unknown);
    }

    #[test]
    fn test_simple_bottleneck_type_display() {
        assert_eq!(format!("{}", SimpleBottleneckType::CpuBound), "CPU Bound");
    }

    // ========================================================================
    // Bottleneck Tests
    // ========================================================================

    #[test]
    fn test_bottleneck_new() {
        let b = Bottleneck::new(
            SimpleBottleneckType::DrawCalls,
            BottleneckSeverity::High,
            "Too many draw calls",
            3000.0,
            2000.0,
        );

        assert_eq!(b.bottleneck_type, SimpleBottleneckType::DrawCalls);
        assert_eq!(b.severity, BottleneckSeverity::High);
        assert_eq!(b.description, "Too many draw calls");
        assert_eq!(b.metric_value, 3000.0);
        assert_eq!(b.threshold, 2000.0);
    }

    #[test]
    fn test_bottleneck_percentage_over_threshold() {
        let b = Bottleneck::new(
            SimpleBottleneckType::DrawCalls,
            BottleneckSeverity::Medium,
            "test",
            3000.0,
            2000.0,
        );

        let pct = b.percentage_over_threshold();
        assert!((pct - 50.0).abs() < 0.1); // 50% over
    }

    #[test]
    fn test_bottleneck_percentage_zero_threshold() {
        let b = Bottleneck::new(
            SimpleBottleneckType::Unknown,
            BottleneckSeverity::None,
            "test",
            100.0,
            0.0,
        );

        assert_eq!(b.percentage_over_threshold(), 0.0);
    }

    #[test]
    fn test_bottleneck_is_critical() {
        let critical = Bottleneck::new(
            SimpleBottleneckType::DrawCalls,
            BottleneckSeverity::Critical,
            "critical",
            10000.0,
            5000.0,
        );
        assert!(critical.is_critical());

        let high = Bottleneck::new(
            SimpleBottleneckType::DrawCalls,
            BottleneckSeverity::High,
            "high",
            4000.0,
            2000.0,
        );
        assert!(!high.is_critical());
    }

    #[test]
    fn test_bottleneck_exceeds_threshold() {
        let exceeds = Bottleneck::new(
            SimpleBottleneckType::DrawCalls,
            BottleneckSeverity::Medium,
            "test",
            3000.0,
            2000.0,
        );
        assert!(exceeds.exceeds_threshold());

        let below = Bottleneck::new(
            SimpleBottleneckType::DrawCalls,
            BottleneckSeverity::None,
            "test",
            1000.0,
            2000.0,
        );
        assert!(!below.exceeds_threshold());
    }

    #[test]
    fn test_bottleneck_display() {
        let b = Bottleneck::new(
            SimpleBottleneckType::DrawCalls,
            BottleneckSeverity::High,
            "Too many",
            3000.0,
            2000.0,
        );
        let s = format!("{}", b);
        assert!(s.contains("Draw Calls"));
        assert!(s.contains("High"));
    }

    // ========================================================================
    // BottleneckAnalysis Tests
    // ========================================================================

    #[test]
    fn test_bottleneck_analysis_new() {
        let analysis = BottleneckAnalysis::new(5.0, 8.0);
        assert_eq!(analysis.cpu_time_ms, 5.0);
        assert_eq!(analysis.gpu_time_ms, 8.0);
        assert_eq!(analysis.frame_time_ms, 8.0);
        assert!(analysis.bottlenecks.is_empty());
        assert!(analysis.primary_bottleneck.is_none());
    }

    #[test]
    fn test_bottleneck_analysis_is_cpu_bound() {
        let analysis = BottleneckAnalysis::new(16.0, 8.0);
        assert!(analysis.is_cpu_bound());
        assert!(!analysis.is_gpu_bound());
    }

    #[test]
    fn test_bottleneck_analysis_is_gpu_bound() {
        let analysis = BottleneckAnalysis::new(5.0, 16.0);
        assert!(analysis.is_gpu_bound());
        assert!(!analysis.is_cpu_bound());
    }

    #[test]
    fn test_bottleneck_analysis_has_critical_issues() {
        let mut analysis = BottleneckAnalysis::new(8.0, 8.0);
        assert!(!analysis.has_critical_issues());

        analysis.bottlenecks.push(Bottleneck::new(
            SimpleBottleneckType::DrawCalls,
            BottleneckSeverity::Critical,
            "critical",
            10000.0,
            5000.0,
        ));
        assert!(analysis.has_critical_issues());
    }

    #[test]
    fn test_bottleneck_analysis_worst_bottleneck() {
        let mut analysis = BottleneckAnalysis::new(8.0, 8.0);
        assert!(analysis.worst_bottleneck().is_none());

        analysis.bottlenecks.push(Bottleneck::new(
            SimpleBottleneckType::DrawCalls,
            BottleneckSeverity::Low,
            "low",
            2500.0,
            2000.0,
        ));
        analysis.bottlenecks.push(Bottleneck::new(
            SimpleBottleneckType::Fillrate,
            BottleneckSeverity::High,
            "high",
            100_000_000.0,
            50_000_000.0,
        ));

        let worst = analysis.worst_bottleneck().unwrap();
        assert_eq!(worst.severity, BottleneckSeverity::High);
    }

    #[test]
    fn test_bottleneck_analysis_bottleneck_count() {
        let mut analysis = BottleneckAnalysis::new(8.0, 8.0);
        assert_eq!(analysis.bottleneck_count(), 0);

        analysis.bottlenecks.push(Bottleneck::new(
            SimpleBottleneckType::DrawCalls,
            BottleneckSeverity::Medium,
            "test",
            3000.0,
            2000.0,
        ));
        assert_eq!(analysis.bottleneck_count(), 1);
    }

    #[test]
    fn test_bottleneck_analysis_has_bottlenecks() {
        let mut analysis = BottleneckAnalysis::new(8.0, 8.0);
        assert!(!analysis.has_bottlenecks());

        analysis.bottlenecks.push(Bottleneck::new(
            SimpleBottleneckType::DrawCalls,
            BottleneckSeverity::Medium,
            "test",
            3000.0,
            2000.0,
        ));
        assert!(analysis.has_bottlenecks());
    }

    #[test]
    fn test_bottleneck_analysis_by_severity() {
        let mut analysis = BottleneckAnalysis::new(8.0, 8.0);

        analysis.bottlenecks.push(Bottleneck::new(
            SimpleBottleneckType::DrawCalls,
            BottleneckSeverity::Low,
            "low",
            2200.0,
            2000.0,
        ));
        analysis.bottlenecks.push(Bottleneck::new(
            SimpleBottleneckType::Fillrate,
            BottleneckSeverity::High,
            "high",
            100_000_000.0,
            50_000_000.0,
        ));

        let high_only = analysis.bottlenecks_by_severity(BottleneckSeverity::High);
        assert_eq!(high_only.len(), 1);

        let all = analysis.bottlenecks_by_severity(BottleneckSeverity::Low);
        assert_eq!(all.len(), 2);
    }

    // ========================================================================
    // BottleneckThresholds Tests
    // ========================================================================

    #[test]
    fn test_bottleneck_thresholds_default() {
        let t = BottleneckThresholds::default();
        assert!(t.draw_call_warning > 0);
        assert!(t.draw_call_critical > t.draw_call_warning);
    }

    #[test]
    fn test_bottleneck_thresholds_high_end() {
        let high = BottleneckThresholds::high_end();
        let default = BottleneckThresholds::default();
        assert!(high.draw_call_warning > default.draw_call_warning);
    }

    #[test]
    fn test_bottleneck_thresholds_low_end() {
        let low = BottleneckThresholds::low_end();
        let default = BottleneckThresholds::default();
        assert!(low.draw_call_warning < default.draw_call_warning);
    }

    #[test]
    fn test_bottleneck_thresholds_new() {
        let t = BottleneckThresholds::new(1000, 2000, 1_000_000.0, 50_000_000.0, 1.5);
        assert_eq!(t.draw_call_warning, 1000);
        assert_eq!(t.draw_call_critical, 2000);
    }

    // ========================================================================
    // BottleneckTrend Tests
    // ========================================================================

    #[test]
    fn test_bottleneck_trend_default() {
        assert_eq!(BottleneckTrend::default(), BottleneckTrend::Unknown);
    }

    #[test]
    fn test_bottleneck_trend_display() {
        assert_eq!(format!("{}", BottleneckTrend::Improving), "Improving");
        assert_eq!(format!("{}", BottleneckTrend::Degrading), "Degrading");
        assert_eq!(format!("{}", BottleneckTrend::Stable), "Stable");
    }

    // ========================================================================
    // SimpleBottleneckAnalyzer Tests
    // ========================================================================

    #[test]
    fn test_simple_analyzer_new() {
        let analyzer = SimpleBottleneckAnalyzer::new();
        assert!(analyzer.history().is_empty());
        assert_eq!(analyzer.history_size(), SimpleBottleneckAnalyzer::DEFAULT_HISTORY_SIZE);
    }

    #[test]
    fn test_simple_analyzer_with_thresholds() {
        let thresholds = BottleneckThresholds::high_end();
        let analyzer = SimpleBottleneckAnalyzer::with_thresholds(thresholds.clone());
        assert_eq!(analyzer.thresholds().draw_call_warning, thresholds.draw_call_warning);
    }

    #[test]
    fn test_simple_analyzer_analyze_frame_cpu_bound() {
        let mut analyzer = SimpleBottleneckAnalyzer::new();
        let result = analyzer.analyze_frame(16.0, 4.0, 100, 10000, 1000000);

        assert!(result.is_cpu_bound());
        assert!(result.has_bottlenecks());
        assert_eq!(result.primary_bottleneck, Some(SimpleBottleneckType::CpuBound));
    }

    #[test]
    fn test_simple_analyzer_analyze_frame_gpu_bound() {
        let mut analyzer = SimpleBottleneckAnalyzer::new();
        let result = analyzer.analyze_frame(4.0, 16.0, 100, 10000, 1000000);

        assert!(result.is_gpu_bound());
        assert!(result.has_bottlenecks());
        assert_eq!(result.primary_bottleneck, Some(SimpleBottleneckType::GpuBound));
    }

    #[test]
    fn test_simple_analyzer_analyze_frame_draw_call_warning() {
        let mut analyzer = SimpleBottleneckAnalyzer::new();
        let result = analyzer.analyze_frame(8.0, 8.0, 3000, 10000, 1000000);

        let draw_call_bottlenecks: Vec<_> = result
            .bottlenecks
            .iter()
            .filter(|b| b.bottleneck_type == SimpleBottleneckType::DrawCalls)
            .collect();
        assert!(!draw_call_bottlenecks.is_empty());
    }

    #[test]
    fn test_simple_analyzer_analyze_frame_draw_call_critical() {
        let mut analyzer = SimpleBottleneckAnalyzer::new();
        let result = analyzer.analyze_frame(8.0, 8.0, 6000, 10000, 1000000);

        let critical = result
            .bottlenecks
            .iter()
            .find(|b| b.bottleneck_type == SimpleBottleneckType::DrawCalls);
        assert!(critical.is_some());
        assert_eq!(critical.unwrap().severity, BottleneckSeverity::Critical);
    }

    #[test]
    fn test_simple_analyzer_analyze_frame_vertex_processing() {
        let mut analyzer = SimpleBottleneckAnalyzer::new();
        // High vertex count relative to frame time
        let result = analyzer.analyze_frame(8.0, 8.0, 100, 50_000_000, 1000000);

        let vertex_bottleneck = result
            .bottlenecks
            .iter()
            .find(|b| b.bottleneck_type == SimpleBottleneckType::VertexProcessing);
        assert!(vertex_bottleneck.is_some());
    }

    #[test]
    fn test_simple_analyzer_analyze_frame_fillrate() {
        let mut analyzer = SimpleBottleneckAnalyzer::new();
        // High pixel count relative to frame time
        let result = analyzer.analyze_frame(8.0, 8.0, 100, 10000, 1_000_000_000);

        let fillrate_bottleneck = result
            .bottlenecks
            .iter()
            .find(|b| b.bottleneck_type == SimpleBottleneckType::Fillrate);
        assert!(fillrate_bottleneck.is_some());
    }

    #[test]
    fn test_simple_analyzer_history() {
        let mut analyzer = SimpleBottleneckAnalyzer::new();

        analyzer.analyze_frame(8.0, 8.0, 100, 10000, 1000000);
        analyzer.analyze_frame(9.0, 7.0, 150, 15000, 1200000);

        assert_eq!(analyzer.history().len(), 2);
    }

    #[test]
    fn test_simple_analyzer_history_limit() {
        let mut analyzer = SimpleBottleneckAnalyzer::new();
        analyzer.set_history_size(5);

        for i in 0..10 {
            analyzer.analyze_frame(8.0 + i as f64, 8.0, 100, 10000, 1000000);
        }

        assert_eq!(analyzer.history().len(), 5);
    }

    #[test]
    fn test_simple_analyzer_trend_unknown() {
        let mut analyzer = SimpleBottleneckAnalyzer::new();
        // Not enough data
        analyzer.analyze_frame(8.0, 8.0, 100, 10000, 1000000);
        assert_eq!(analyzer.trend(), BottleneckTrend::Unknown);
    }

    #[test]
    fn test_simple_analyzer_trend_improving() {
        let mut analyzer = SimpleBottleneckAnalyzer::new();

        // Simulate improving performance (decreasing frame times)
        for i in 0..20 {
            let time = 20.0 - i as f64 * 0.5;
            analyzer.analyze_frame(time, time, 100, 10000, 1000000);
        }

        assert_eq!(analyzer.trend(), BottleneckTrend::Improving);
    }

    #[test]
    fn test_simple_analyzer_trend_degrading() {
        let mut analyzer = SimpleBottleneckAnalyzer::new();

        // Simulate degrading performance (increasing frame times)
        for i in 0..20 {
            let time = 8.0 + i as f64 * 0.5;
            analyzer.analyze_frame(time, time, 100, 10000, 1000000);
        }

        assert_eq!(analyzer.trend(), BottleneckTrend::Degrading);
    }

    #[test]
    fn test_simple_analyzer_trend_stable() {
        let mut analyzer = SimpleBottleneckAnalyzer::new();

        // Simulate stable performance
        for _ in 0..20 {
            analyzer.analyze_frame(8.0, 8.0, 100, 10000, 1000000);
        }

        assert_eq!(analyzer.trend(), BottleneckTrend::Stable);
    }

    #[test]
    fn test_simple_analyzer_suggestions() {
        let mut analyzer = SimpleBottleneckAnalyzer::new();
        analyzer.analyze_frame(16.0, 4.0, 100, 10000, 1000000);

        let suggestions = analyzer.suggestions();
        assert!(!suggestions.is_empty());
    }

    #[test]
    fn test_simple_analyzer_suggestions_empty() {
        let analyzer = SimpleBottleneckAnalyzer::new();
        let suggestions = analyzer.suggestions();
        assert!(suggestions.is_empty());
    }

    #[test]
    fn test_simple_analyzer_reset() {
        let mut analyzer = SimpleBottleneckAnalyzer::new();
        analyzer.analyze_frame(8.0, 8.0, 100, 10000, 1000000);
        assert!(!analyzer.history().is_empty());

        analyzer.reset();
        assert!(analyzer.history().is_empty());
    }

    #[test]
    fn test_simple_analyzer_set_thresholds() {
        let mut analyzer = SimpleBottleneckAnalyzer::new();
        let new_thresholds = BottleneckThresholds::low_end();
        analyzer.set_thresholds(new_thresholds.clone());
        assert_eq!(analyzer.thresholds().draw_call_warning, new_thresholds.draw_call_warning);
    }

    #[test]
    fn test_simple_analyzer_set_history_size() {
        let mut analyzer = SimpleBottleneckAnalyzer::new();
        for _ in 0..10 {
            analyzer.analyze_frame(8.0, 8.0, 100, 10000, 1000000);
        }

        analyzer.set_history_size(5);
        assert_eq!(analyzer.history_size(), 5);
        assert_eq!(analyzer.history().len(), 5);
    }

    #[test]
    fn test_simple_analyzer_debug() {
        let analyzer = SimpleBottleneckAnalyzer::new();
        let debug_str = format!("{:?}", analyzer);
        assert!(debug_str.contains("SimpleBottleneckAnalyzer"));
    }

    // ========================================================================
    // Send + Sync Tests
    // ========================================================================

    fn assert_send<T: Send>() {}
    fn assert_sync<T: Sync>() {}

    #[test]
    fn test_simple_bottleneck_type_send_sync() {
        assert_send::<SimpleBottleneckType>();
        assert_sync::<SimpleBottleneckType>();
    }

    #[test]
    fn test_bottleneck_send_sync() {
        assert_send::<Bottleneck>();
        assert_sync::<Bottleneck>();
    }

    #[test]
    fn test_bottleneck_analysis_send_sync() {
        assert_send::<BottleneckAnalysis>();
        assert_sync::<BottleneckAnalysis>();
    }

    #[test]
    fn test_bottleneck_thresholds_send_sync() {
        assert_send::<BottleneckThresholds>();
        assert_sync::<BottleneckThresholds>();
    }

    #[test]
    fn test_simple_analyzer_send() {
        assert_send::<SimpleBottleneckAnalyzer>();
    }
}
