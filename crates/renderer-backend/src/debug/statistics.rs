//! GPU Statistics Collection and Reporting
//!
//! This module provides comprehensive GPU statistics collection for performance
//! monitoring and profiling of the rendering pipeline.
//!
//! # Overview
//!
//! The statistics system tracks various GPU metrics:
//!
//! - **Frame Timing**: Frame time, GPU time, FPS calculations
//! - **Draw Statistics**: Draw calls, triangle count, vertex count
//! - **Resource Bindings**: Texture binds, buffer binds, pipeline switches
//! - **Compute Statistics**: Compute dispatch counts
//! - **Memory Usage**: GPU memory consumption tracking
//!
//! # Example
//!
//! ```ignore
//! use renderer_backend::debug::statistics::*;
//!
//! let mut collector = StatisticsCollector::new(120); // 120 frame history
//!
//! // Each frame:
//! collector.begin_frame();
//!
//! // Record statistics during rendering
//! collector.record(StatisticType::GpuTime, 8.5);
//! collector.increment(StatisticType::DrawCalls);
//! collector.record(StatisticType::TriangleCount, 150000.0);
//!
//! collector.end_frame();
//!
//! // Query statistics
//! println!("Average FPS: {:.1}", collector.average_fps());
//! println!("Average frame time: {:.2}ms", collector.average_frame_time());
//!
//! // Generate report
//! let report = StatisticsReport::from_collector(&collector);
//! println!("{}", report.to_string());
//! ```

use std::collections::{HashMap, VecDeque};
use std::fmt;
use std::time::Instant;

/// A single GPU statistic with min/max/average tracking.
///
/// Maintains running statistics for a single metric, tracking the current
/// value, minimum, maximum, and rolling average.
#[derive(Clone, Copy, Debug)]
pub struct GpuStatistic {
    /// Current value of the statistic.
    pub current: f64,
    /// Minimum value observed.
    pub min: f64,
    /// Maximum value observed.
    pub max: f64,
    /// Running average of all recorded values.
    pub average: f64,
    /// Number of samples recorded.
    pub sample_count: u32,
    /// Sum of all values for average calculation.
    sum: f64,
}

impl GpuStatistic {
    /// Creates a new statistic with default values.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let stat = GpuStatistic::new();
    /// assert_eq!(stat.sample_count, 0);
    /// ```
    #[must_use]
    pub fn new() -> Self {
        Self {
            current: 0.0,
            min: f64::MAX,
            max: f64::MIN,
            average: 0.0,
            sample_count: 0,
            sum: 0.0,
        }
    }

    /// Records a new value, updating min/max/average.
    ///
    /// # Arguments
    ///
    /// * `value` - The value to record
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let mut stat = GpuStatistic::new();
    /// stat.record(10.0);
    /// stat.record(20.0);
    /// assert_eq!(stat.current, 20.0);
    /// assert_eq!(stat.min, 10.0);
    /// assert_eq!(stat.max, 20.0);
    /// assert_eq!(stat.average, 15.0);
    /// ```
    pub fn record(&mut self, value: f64) {
        self.current = value;
        self.min = self.min.min(value);
        self.max = self.max.max(value);
        self.sum += value;
        self.sample_count += 1;
        self.average = self.sum / f64::from(self.sample_count);
    }

    /// Resets the statistic to default values.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let mut stat = GpuStatistic::new();
    /// stat.record(10.0);
    /// stat.reset();
    /// assert_eq!(stat.sample_count, 0);
    /// ```
    pub fn reset(&mut self) {
        *self = Self::new();
    }

    /// Returns the range (max - min) of recorded values.
    ///
    /// Returns 0.0 if no samples have been recorded.
    #[must_use]
    pub fn range(&self) -> f64 {
        if self.sample_count == 0 {
            0.0
        } else {
            self.max - self.min
        }
    }

    /// Returns the variance of recorded values using Welford's algorithm.
    ///
    /// Returns 0.0 if fewer than 2 samples have been recorded.
    #[must_use]
    pub fn variance(&self) -> f64 {
        if self.sample_count < 2 {
            return 0.0;
        }
        // Approximate variance (would need proper Welford's for exact)
        // For now, return a simple estimate based on range
        let range = self.range();
        (range / 4.0).powi(2) // Rough estimate: stddev ~ range/4
    }
}

impl Default for GpuStatistic {
    fn default() -> Self {
        Self::new()
    }
}

/// Types of GPU statistics that can be tracked.
///
/// This enum defines all the different metrics that can be collected
/// during rendering.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum StatisticType {
    /// Total frame time in milliseconds.
    FrameTime,
    /// GPU execution time in milliseconds.
    GpuTime,
    /// Number of draw calls per frame.
    DrawCalls,
    /// Total triangle count per frame.
    TriangleCount,
    /// Total vertex count per frame.
    VertexCount,
    /// Number of texture binding operations.
    TextureBinds,
    /// Number of buffer binding operations.
    BufferBinds,
    /// Number of pipeline state changes.
    PipelineSwitches,
    /// Number of compute shader dispatches.
    ComputeDispatches,
    /// GPU memory usage in bytes.
    MemoryUsage,
    /// Custom user-defined statistic type.
    Custom(u32),
}

impl StatisticType {
    /// Returns a human-readable name for this statistic type.
    #[must_use]
    pub fn name(&self) -> &'static str {
        match self {
            StatisticType::FrameTime => "Frame Time",
            StatisticType::GpuTime => "GPU Time",
            StatisticType::DrawCalls => "Draw Calls",
            StatisticType::TriangleCount => "Triangle Count",
            StatisticType::VertexCount => "Vertex Count",
            StatisticType::TextureBinds => "Texture Binds",
            StatisticType::BufferBinds => "Buffer Binds",
            StatisticType::PipelineSwitches => "Pipeline Switches",
            StatisticType::ComputeDispatches => "Compute Dispatches",
            StatisticType::MemoryUsage => "Memory Usage",
            StatisticType::Custom(_) => "Custom",
        }
    }

    /// Returns the unit of measurement for this statistic type.
    #[must_use]
    pub fn unit(&self) -> &'static str {
        match self {
            StatisticType::FrameTime | StatisticType::GpuTime => "ms",
            StatisticType::MemoryUsage => "bytes",
            _ => "",
        }
    }

    /// All built-in statistic types (excluding Custom).
    pub const BUILTIN: [StatisticType; 10] = [
        StatisticType::FrameTime,
        StatisticType::GpuTime,
        StatisticType::DrawCalls,
        StatisticType::TriangleCount,
        StatisticType::VertexCount,
        StatisticType::TextureBinds,
        StatisticType::BufferBinds,
        StatisticType::PipelineSwitches,
        StatisticType::ComputeDispatches,
        StatisticType::MemoryUsage,
    ];

    /// Returns true if this is a timing-related statistic.
    #[must_use]
    pub fn is_timing(&self) -> bool {
        matches!(self, StatisticType::FrameTime | StatisticType::GpuTime)
    }

    /// Returns true if this is a count-based statistic.
    #[must_use]
    pub fn is_count(&self) -> bool {
        matches!(
            self,
            StatisticType::DrawCalls
                | StatisticType::TriangleCount
                | StatisticType::VertexCount
                | StatisticType::TextureBinds
                | StatisticType::BufferBinds
                | StatisticType::PipelineSwitches
                | StatisticType::ComputeDispatches
        )
    }
}

impl fmt::Display for StatisticType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// Statistics collected for a single frame.
///
/// Contains all the metrics captured during a single frame of rendering.
#[derive(Clone, Debug, Default)]
pub struct FrameStatistics {
    /// Sequential frame number.
    pub frame_number: u64,
    /// Total frame time in milliseconds.
    pub frame_time_ms: f64,
    /// GPU execution time in milliseconds.
    pub gpu_time_ms: f64,
    /// Number of draw calls issued.
    pub draw_calls: u32,
    /// Total number of triangles rendered.
    pub triangles: u64,
    /// Total number of vertices processed.
    pub vertices: u64,
    /// Number of texture binding operations.
    pub texture_binds: u32,
    /// Number of buffer binding operations.
    pub buffer_binds: u32,
    /// Number of pipeline state changes.
    pub pipeline_switches: u32,
    /// Number of compute shader dispatches.
    pub compute_dispatches: u32,
}

impl FrameStatistics {
    /// Creates new frame statistics with the given frame number.
    #[must_use]
    pub fn new(frame_number: u64) -> Self {
        Self {
            frame_number,
            ..Default::default()
        }
    }

    /// Returns the calculated FPS based on frame time.
    ///
    /// Returns 0.0 if frame time is zero or negative.
    #[must_use]
    pub fn fps(&self) -> f64 {
        if self.frame_time_ms > 0.0 {
            1000.0 / self.frame_time_ms
        } else {
            0.0
        }
    }

    /// Returns the GPU utilization as a percentage of frame time.
    ///
    /// Returns 0.0 if frame time is zero.
    #[must_use]
    pub fn gpu_utilization(&self) -> f64 {
        if self.frame_time_ms > 0.0 {
            (self.gpu_time_ms / self.frame_time_ms) * 100.0
        } else {
            0.0
        }
    }

    /// Returns triangles per draw call.
    #[must_use]
    pub fn triangles_per_draw(&self) -> f64 {
        if self.draw_calls > 0 {
            self.triangles as f64 / f64::from(self.draw_calls)
        } else {
            0.0
        }
    }

    /// Returns vertices per triangle (useful for detecting degenerate geometry).
    #[must_use]
    pub fn vertices_per_triangle(&self) -> f64 {
        if self.triangles > 0 {
            self.vertices as f64 / self.triangles as f64
        } else {
            0.0
        }
    }
}

/// Collects and manages GPU statistics across multiple frames.
///
/// The collector maintains a rolling history of frame statistics and provides
/// aggregate metrics like averages and totals.
pub struct StatisticsCollector {
    /// Per-type statistics with running averages.
    statistics: HashMap<StatisticType, GpuStatistic>,
    /// Rolling history of per-frame statistics.
    frame_history: VecDeque<FrameStatistics>,
    /// Maximum number of frames to keep in history.
    history_size: usize,
    /// Current frame number.
    current_frame: u64,
    /// Statistics for the frame being recorded.
    current_stats: FrameStatistics,
    /// Timestamp when the current frame started.
    frame_start: Option<Instant>,
    /// Timestamp when the collector was created/reset.
    start_time: Instant,
    /// Whether we're currently recording a frame.
    recording: bool,
}

impl StatisticsCollector {
    /// Creates a new statistics collector with the specified history size.
    ///
    /// # Arguments
    ///
    /// * `history_size` - Number of frames to keep in history for averaging
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let collector = StatisticsCollector::new(120); // 2 seconds at 60fps
    /// ```
    #[must_use]
    pub fn new(history_size: usize) -> Self {
        Self {
            statistics: HashMap::new(),
            frame_history: VecDeque::with_capacity(history_size),
            history_size,
            current_frame: 0,
            current_stats: FrameStatistics::new(0),
            frame_start: None,
            start_time: Instant::now(),
            recording: false,
        }
    }

    /// Begins recording a new frame.
    ///
    /// This should be called at the start of each frame before any
    /// rendering operations. It resets per-frame counters and starts
    /// timing.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// collector.begin_frame();
    /// // ... rendering ...
    /// collector.end_frame();
    /// ```
    pub fn begin_frame(&mut self) {
        self.frame_start = Some(Instant::now());
        self.current_stats = FrameStatistics::new(self.current_frame);
        self.recording = true;
    }

    /// Ends recording the current frame.
    ///
    /// This should be called at the end of each frame after all rendering
    /// is complete. It calculates frame time and stores the statistics.
    pub fn end_frame(&mut self) {
        if !self.recording {
            return;
        }

        // Calculate frame time
        if let Some(start) = self.frame_start.take() {
            let frame_time = start.elapsed().as_secs_f64() * 1000.0;
            self.current_stats.frame_time_ms = frame_time;
            self.record(StatisticType::FrameTime, frame_time);
        }

        // Store frame statistics in history
        if self.frame_history.len() >= self.history_size {
            self.frame_history.pop_front();
        }
        self.frame_history.push_back(self.current_stats.clone());

        self.current_frame += 1;
        self.recording = false;
    }

    /// Records a value for the specified statistic type.
    ///
    /// # Arguments
    ///
    /// * `stat_type` - The type of statistic to record
    /// * `value` - The value to record
    ///
    /// # Examples
    ///
    /// ```ignore
    /// collector.record(StatisticType::GpuTime, 8.5);
    /// collector.record(StatisticType::TriangleCount, 150000.0);
    /// ```
    pub fn record(&mut self, stat_type: StatisticType, value: f64) {
        // Update running statistics
        self.statistics
            .entry(stat_type)
            .or_insert_with(GpuStatistic::new)
            .record(value);

        // Update current frame statistics
        match stat_type {
            StatisticType::FrameTime => self.current_stats.frame_time_ms = value,
            StatisticType::GpuTime => self.current_stats.gpu_time_ms = value,
            StatisticType::DrawCalls => self.current_stats.draw_calls = value as u32,
            StatisticType::TriangleCount => self.current_stats.triangles = value as u64,
            StatisticType::VertexCount => self.current_stats.vertices = value as u64,
            StatisticType::TextureBinds => self.current_stats.texture_binds = value as u32,
            StatisticType::BufferBinds => self.current_stats.buffer_binds = value as u32,
            StatisticType::PipelineSwitches => self.current_stats.pipeline_switches = value as u32,
            StatisticType::ComputeDispatches => {
                self.current_stats.compute_dispatches = value as u32
            }
            StatisticType::MemoryUsage | StatisticType::Custom(_) => {}
        }
    }

    /// Increments a count-based statistic by 1.
    ///
    /// This is a convenience method for statistics that are incremented
    /// one at a time, like draw calls or texture binds.
    ///
    /// # Arguments
    ///
    /// * `stat_type` - The type of statistic to increment
    ///
    /// # Examples
    ///
    /// ```ignore
    /// collector.increment(StatisticType::DrawCalls);
    /// collector.increment(StatisticType::TextureBinds);
    /// ```
    pub fn increment(&mut self, stat_type: StatisticType) {
        // Get current value and increment
        match stat_type {
            StatisticType::DrawCalls => {
                self.current_stats.draw_calls += 1;
            }
            StatisticType::TextureBinds => {
                self.current_stats.texture_binds += 1;
            }
            StatisticType::BufferBinds => {
                self.current_stats.buffer_binds += 1;
            }
            StatisticType::PipelineSwitches => {
                self.current_stats.pipeline_switches += 1;
            }
            StatisticType::ComputeDispatches => {
                self.current_stats.compute_dispatches += 1;
            }
            _ => {
                // For non-count statistics, increment from current value
                let current = self.statistics.get(&stat_type).map_or(0.0, |s| s.current);
                self.record(stat_type, current + 1.0);
                return;
            }
        }
    }

    /// Gets the statistic for the specified type.
    ///
    /// Returns `None` if no values have been recorded for this type.
    ///
    /// # Arguments
    ///
    /// * `stat_type` - The type of statistic to retrieve
    ///
    /// # Examples
    ///
    /// ```ignore
    /// if let Some(stat) = collector.get(StatisticType::FrameTime) {
    ///     println!("Average frame time: {:.2}ms", stat.average);
    /// }
    /// ```
    #[must_use]
    pub fn get(&self, stat_type: StatisticType) -> Option<&GpuStatistic> {
        self.statistics.get(&stat_type)
    }

    /// Returns the statistics for the current frame being recorded.
    ///
    /// If not currently recording, returns the most recently completed frame.
    #[must_use]
    pub fn current_frame_stats(&self) -> &FrameStatistics {
        &self.current_stats
    }

    /// Returns the frame history.
    #[must_use]
    pub fn frame_history(&self) -> &VecDeque<FrameStatistics> {
        &self.frame_history
    }

    /// Returns the current frame number.
    #[must_use]
    pub fn current_frame_number(&self) -> u64 {
        self.current_frame
    }

    /// Returns the average frame time in milliseconds across the history.
    ///
    /// Returns 0.0 if no frames have been recorded.
    #[must_use]
    pub fn average_frame_time(&self) -> f64 {
        if self.frame_history.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.frame_history.iter().map(|f| f.frame_time_ms).sum();
        sum / self.frame_history.len() as f64
    }

    /// Returns the average FPS across the history.
    ///
    /// Returns 0.0 if no frames have been recorded or average frame time is 0.
    #[must_use]
    pub fn average_fps(&self) -> f64 {
        let avg_frame_time = self.average_frame_time();
        if avg_frame_time > 0.0 {
            1000.0 / avg_frame_time
        } else {
            0.0
        }
    }

    /// Returns the minimum frame time in the history.
    #[must_use]
    pub fn min_frame_time(&self) -> f64 {
        self.frame_history
            .iter()
            .map(|f| f.frame_time_ms)
            .fold(f64::MAX, f64::min)
    }

    /// Returns the maximum frame time in the history.
    #[must_use]
    pub fn max_frame_time(&self) -> f64 {
        self.frame_history
            .iter()
            .map(|f| f.frame_time_ms)
            .fold(f64::MIN, f64::max)
    }

    /// Returns the 1% low FPS (99th percentile frame time).
    #[must_use]
    pub fn percentile_1_low_fps(&self) -> f64 {
        if self.frame_history.is_empty() {
            return 0.0;
        }
        let mut times: Vec<f64> = self.frame_history.iter().map(|f| f.frame_time_ms).collect();
        times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let idx = (times.len() as f64 * 0.99).floor() as usize;
        let idx = idx.min(times.len() - 1);
        if times[idx] > 0.0 {
            1000.0 / times[idx]
        } else {
            0.0
        }
    }

    /// Returns the total draw calls across all recorded frames.
    #[must_use]
    pub fn total_draw_calls(&self) -> u64 {
        self.frame_history
            .iter()
            .map(|f| u64::from(f.draw_calls))
            .sum()
    }

    /// Returns the total triangles across all recorded frames.
    #[must_use]
    pub fn total_triangles(&self) -> u64 {
        self.frame_history.iter().map(|f| f.triangles).sum()
    }

    /// Returns the total elapsed time since the collector was created/reset.
    #[must_use]
    pub fn elapsed_seconds(&self) -> f64 {
        self.start_time.elapsed().as_secs_f64()
    }

    /// Returns whether a frame is currently being recorded.
    #[must_use]
    pub fn is_recording(&self) -> bool {
        self.recording
    }

    /// Returns the history size (maximum frames stored).
    #[must_use]
    pub fn history_size(&self) -> usize {
        self.history_size
    }

    /// Resets all statistics and history.
    ///
    /// Clears the frame history, resets all running statistics,
    /// and resets the frame counter to 0.
    pub fn reset(&mut self) {
        self.statistics.clear();
        self.frame_history.clear();
        self.current_frame = 0;
        self.current_stats = FrameStatistics::new(0);
        self.frame_start = None;
        self.start_time = Instant::now();
        self.recording = false;
    }

    /// Returns an iterator over all statistic types that have been recorded.
    pub fn recorded_types(&self) -> impl Iterator<Item = &StatisticType> {
        self.statistics.keys()
    }
}

impl Default for StatisticsCollector {
    fn default() -> Self {
        Self::new(120) // Default to 2 seconds at 60fps
    }
}

/// A summary report of collected GPU statistics.
///
/// Provides a formatted overview of rendering performance and resource usage.
#[derive(Clone, Debug)]
pub struct StatisticsReport {
    /// Total number of frames recorded.
    pub frame_count: u64,
    /// Total elapsed time in seconds.
    pub total_time_s: f64,
    /// Average frames per second.
    pub avg_fps: f64,
    /// Average frame time in milliseconds.
    pub avg_frame_time_ms: f64,
    /// Average GPU time in milliseconds.
    pub avg_gpu_time_ms: f64,
    /// Total draw calls across all frames.
    pub total_draw_calls: u64,
    /// Total triangles rendered across all frames.
    pub total_triangles: u64,
    /// Minimum frame time in milliseconds.
    pub min_frame_time_ms: f64,
    /// Maximum frame time in milliseconds.
    pub max_frame_time_ms: f64,
    /// 1% low FPS.
    pub percentile_1_low_fps: f64,
    /// Total texture binds.
    pub total_texture_binds: u64,
    /// Total buffer binds.
    pub total_buffer_binds: u64,
    /// Total pipeline switches.
    pub total_pipeline_switches: u64,
    /// Total compute dispatches.
    pub total_compute_dispatches: u64,
}

impl StatisticsReport {
    /// Creates a report from a statistics collector.
    ///
    /// # Arguments
    ///
    /// * `collector` - The statistics collector to generate a report from
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let report = StatisticsReport::from_collector(&collector);
    /// println!("{}", report.to_string());
    /// ```
    #[must_use]
    pub fn from_collector(collector: &StatisticsCollector) -> Self {
        let frame_history = collector.frame_history();

        let avg_gpu_time_ms = if frame_history.is_empty() {
            0.0
        } else {
            let sum: f64 = frame_history.iter().map(|f| f.gpu_time_ms).sum();
            sum / frame_history.len() as f64
        };

        let total_texture_binds: u64 = frame_history
            .iter()
            .map(|f| u64::from(f.texture_binds))
            .sum();
        let total_buffer_binds: u64 = frame_history
            .iter()
            .map(|f| u64::from(f.buffer_binds))
            .sum();
        let total_pipeline_switches: u64 = frame_history
            .iter()
            .map(|f| u64::from(f.pipeline_switches))
            .sum();
        let total_compute_dispatches: u64 = frame_history
            .iter()
            .map(|f| u64::from(f.compute_dispatches))
            .sum();

        Self {
            frame_count: collector.current_frame_number(),
            total_time_s: collector.elapsed_seconds(),
            avg_fps: collector.average_fps(),
            avg_frame_time_ms: collector.average_frame_time(),
            avg_gpu_time_ms,
            total_draw_calls: collector.total_draw_calls(),
            total_triangles: collector.total_triangles(),
            min_frame_time_ms: collector.min_frame_time(),
            max_frame_time_ms: collector.max_frame_time(),
            percentile_1_low_fps: collector.percentile_1_low_fps(),
            total_texture_binds,
            total_buffer_binds,
            total_pipeline_switches,
            total_compute_dispatches,
        }
    }

    /// Converts the report to a formatted string.
    ///
    /// Returns a multi-line string with all statistics formatted for display.
    #[must_use]
    #[allow(clippy::inherent_to_string_shadow_display)]
    pub fn to_string(&self) -> String {
        format!(
            "GPU Statistics Report\n\
             ======================\n\
             Frames:           {}\n\
             Total Time:       {:.2}s\n\
             Average FPS:      {:.1}\n\
             1% Low FPS:       {:.1}\n\
             Frame Time:       {:.2}ms (avg), {:.2}ms (min), {:.2}ms (max)\n\
             GPU Time:         {:.2}ms (avg)\n\
             Draw Calls:       {} total\n\
             Triangles:        {} total\n\
             Texture Binds:    {} total\n\
             Buffer Binds:     {} total\n\
             Pipeline Switches:{} total\n\
             Compute Dispatches:{} total",
            self.frame_count,
            self.total_time_s,
            self.avg_fps,
            self.percentile_1_low_fps,
            self.avg_frame_time_ms,
            self.min_frame_time_ms,
            self.max_frame_time_ms,
            self.avg_gpu_time_ms,
            self.total_draw_calls,
            self.total_triangles,
            self.total_texture_binds,
            self.total_buffer_binds,
            self.total_pipeline_switches,
            self.total_compute_dispatches
        )
    }

    /// Returns the average draw calls per frame.
    #[must_use]
    pub fn avg_draw_calls_per_frame(&self) -> f64 {
        if self.frame_count > 0 {
            self.total_draw_calls as f64 / self.frame_count as f64
        } else {
            0.0
        }
    }

    /// Returns the average triangles per frame.
    #[must_use]
    pub fn avg_triangles_per_frame(&self) -> f64 {
        if self.frame_count > 0 {
            self.total_triangles as f64 / self.frame_count as f64
        } else {
            0.0
        }
    }

    /// Returns the GPU utilization percentage.
    #[must_use]
    pub fn gpu_utilization(&self) -> f64 {
        if self.avg_frame_time_ms > 0.0 {
            (self.avg_gpu_time_ms / self.avg_frame_time_ms) * 100.0
        } else {
            0.0
        }
    }
}

impl fmt::Display for StatisticsReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_string())
    }
}

impl Default for StatisticsReport {
    fn default() -> Self {
        Self {
            frame_count: 0,
            total_time_s: 0.0,
            avg_fps: 0.0,
            avg_frame_time_ms: 0.0,
            avg_gpu_time_ms: 0.0,
            total_draw_calls: 0,
            total_triangles: 0,
            min_frame_time_ms: 0.0,
            max_frame_time_ms: 0.0,
            percentile_1_low_fps: 0.0,
            total_texture_binds: 0,
            total_buffer_binds: 0,
            total_pipeline_switches: 0,
            total_compute_dispatches: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ==================== GpuStatistic Tests ====================

    #[test]
    fn test_gpu_statistic_new() {
        let stat = GpuStatistic::new();
        assert_eq!(stat.current, 0.0);
        assert_eq!(stat.min, f64::MAX);
        assert_eq!(stat.max, f64::MIN);
        assert_eq!(stat.average, 0.0);
        assert_eq!(stat.sample_count, 0);
    }

    #[test]
    fn test_gpu_statistic_single_record() {
        let mut stat = GpuStatistic::new();
        stat.record(10.0);

        assert_eq!(stat.current, 10.0);
        assert_eq!(stat.min, 10.0);
        assert_eq!(stat.max, 10.0);
        assert_eq!(stat.average, 10.0);
        assert_eq!(stat.sample_count, 1);
    }

    #[test]
    fn test_gpu_statistic_multiple_records() {
        let mut stat = GpuStatistic::new();
        stat.record(10.0);
        stat.record(20.0);
        stat.record(30.0);

        assert_eq!(stat.current, 30.0);
        assert_eq!(stat.min, 10.0);
        assert_eq!(stat.max, 30.0);
        assert_eq!(stat.average, 20.0);
        assert_eq!(stat.sample_count, 3);
    }

    #[test]
    fn test_gpu_statistic_reset() {
        let mut stat = GpuStatistic::new();
        stat.record(10.0);
        stat.record(20.0);
        stat.reset();

        assert_eq!(stat.current, 0.0);
        assert_eq!(stat.min, f64::MAX);
        assert_eq!(stat.max, f64::MIN);
        assert_eq!(stat.average, 0.0);
        assert_eq!(stat.sample_count, 0);
    }

    #[test]
    fn test_gpu_statistic_range() {
        let mut stat = GpuStatistic::new();
        assert_eq!(stat.range(), 0.0); // No samples

        stat.record(10.0);
        stat.record(30.0);
        assert_eq!(stat.range(), 20.0);
    }

    #[test]
    fn test_gpu_statistic_default() {
        let stat = GpuStatistic::default();
        assert_eq!(stat.sample_count, 0);
    }

    // ==================== StatisticType Tests ====================

    #[test]
    fn test_statistic_type_name() {
        assert_eq!(StatisticType::FrameTime.name(), "Frame Time");
        assert_eq!(StatisticType::GpuTime.name(), "GPU Time");
        assert_eq!(StatisticType::DrawCalls.name(), "Draw Calls");
        assert_eq!(StatisticType::TriangleCount.name(), "Triangle Count");
        assert_eq!(StatisticType::Custom(42).name(), "Custom");
    }

    #[test]
    fn test_statistic_type_unit() {
        assert_eq!(StatisticType::FrameTime.unit(), "ms");
        assert_eq!(StatisticType::GpuTime.unit(), "ms");
        assert_eq!(StatisticType::DrawCalls.unit(), "");
        assert_eq!(StatisticType::MemoryUsage.unit(), "bytes");
    }

    #[test]
    fn test_statistic_type_is_timing() {
        assert!(StatisticType::FrameTime.is_timing());
        assert!(StatisticType::GpuTime.is_timing());
        assert!(!StatisticType::DrawCalls.is_timing());
    }

    #[test]
    fn test_statistic_type_is_count() {
        assert!(StatisticType::DrawCalls.is_count());
        assert!(StatisticType::TriangleCount.is_count());
        assert!(StatisticType::TextureBinds.is_count());
        assert!(!StatisticType::FrameTime.is_count());
    }

    #[test]
    fn test_statistic_type_builtin_count() {
        assert_eq!(StatisticType::BUILTIN.len(), 10);
    }

    #[test]
    fn test_statistic_type_display() {
        let s = format!("{}", StatisticType::FrameTime);
        assert_eq!(s, "Frame Time");
    }

    // ==================== FrameStatistics Tests ====================

    #[test]
    fn test_frame_statistics_new() {
        let stats = FrameStatistics::new(42);
        assert_eq!(stats.frame_number, 42);
        assert_eq!(stats.frame_time_ms, 0.0);
        assert_eq!(stats.draw_calls, 0);
    }

    #[test]
    fn test_frame_statistics_fps() {
        let mut stats = FrameStatistics::new(0);
        stats.frame_time_ms = 16.666;
        let fps = stats.fps();
        assert!((fps - 60.0).abs() < 0.1);
    }

    #[test]
    fn test_frame_statistics_fps_zero_time() {
        let stats = FrameStatistics::new(0);
        assert_eq!(stats.fps(), 0.0);
    }

    #[test]
    fn test_frame_statistics_gpu_utilization() {
        let mut stats = FrameStatistics::new(0);
        stats.frame_time_ms = 16.666;
        stats.gpu_time_ms = 8.333;
        let util = stats.gpu_utilization();
        assert!((util - 50.0).abs() < 0.1);
    }

    #[test]
    fn test_frame_statistics_triangles_per_draw() {
        let mut stats = FrameStatistics::new(0);
        stats.draw_calls = 100;
        stats.triangles = 10000;
        assert_eq!(stats.triangles_per_draw(), 100.0);
    }

    // ==================== StatisticsCollector Tests ====================

    #[test]
    fn test_collector_new() {
        let collector = StatisticsCollector::new(60);
        assert_eq!(collector.history_size(), 60);
        assert_eq!(collector.current_frame_number(), 0);
        assert!(!collector.is_recording());
    }

    #[test]
    fn test_collector_begin_end_frame() {
        let mut collector = StatisticsCollector::new(10);

        collector.begin_frame();
        assert!(collector.is_recording());

        collector.end_frame();
        assert!(!collector.is_recording());
        assert_eq!(collector.current_frame_number(), 1);
    }

    #[test]
    fn test_collector_record_and_get() {
        let mut collector = StatisticsCollector::new(10);
        collector.begin_frame();
        collector.record(StatisticType::GpuTime, 8.5);
        collector.end_frame();

        let stat = collector.get(StatisticType::GpuTime);
        assert!(stat.is_some());
        assert_eq!(stat.unwrap().current, 8.5);
    }

    #[test]
    fn test_collector_increment() {
        let mut collector = StatisticsCollector::new(10);
        collector.begin_frame();

        collector.increment(StatisticType::DrawCalls);
        collector.increment(StatisticType::DrawCalls);
        collector.increment(StatisticType::DrawCalls);

        assert_eq!(collector.current_frame_stats().draw_calls, 3);
    }

    #[test]
    fn test_collector_frame_history() {
        let mut collector = StatisticsCollector::new(5);

        for _ in 0..3 {
            collector.begin_frame();
            collector.record(StatisticType::DrawCalls, 10.0);
            collector.end_frame();
        }

        assert_eq!(collector.frame_history().len(), 3);
    }

    #[test]
    fn test_collector_history_limit() {
        let mut collector = StatisticsCollector::new(3);

        for i in 0..5 {
            collector.begin_frame();
            collector.record(StatisticType::DrawCalls, i as f64);
            collector.end_frame();
        }

        // Should only keep last 3 frames
        assert_eq!(collector.frame_history().len(), 3);
        // First frame in history should be frame 2 (0-indexed)
        assert_eq!(collector.frame_history().front().unwrap().frame_number, 2);
    }

    #[test]
    fn test_collector_average_frame_time() {
        let mut collector = StatisticsCollector::new(10);

        // Manually set frame times for testing
        for i in 1..=4 {
            collector.begin_frame();
            // Directly modify current_stats since we can't control Instant timing
            collector.current_stats.frame_time_ms = i as f64 * 10.0;
            collector.record(StatisticType::FrameTime, i as f64 * 10.0);

            // Manually add to history
            if collector.frame_history.len() >= collector.history_size {
                collector.frame_history.pop_front();
            }
            collector.frame_history.push_back(collector.current_stats.clone());
            collector.current_frame += 1;
            collector.recording = false;
        }

        // Average of 10, 20, 30, 40 = 25
        assert_eq!(collector.average_frame_time(), 25.0);
    }

    #[test]
    fn test_collector_average_fps() {
        let mut collector = StatisticsCollector::new(10);

        collector.begin_frame();
        collector.current_stats.frame_time_ms = 16.666;
        collector.frame_history.push_back(collector.current_stats.clone());
        collector.current_frame += 1;
        collector.recording = false;

        let fps = collector.average_fps();
        assert!((fps - 60.0).abs() < 0.1);
    }

    #[test]
    fn test_collector_total_draw_calls() {
        let mut collector = StatisticsCollector::new(10);

        for _ in 0..3 {
            collector.begin_frame();
            collector.current_stats.draw_calls = 100;
            collector.frame_history.push_back(collector.current_stats.clone());
            collector.current_frame += 1;
            collector.recording = false;
        }

        assert_eq!(collector.total_draw_calls(), 300);
    }

    #[test]
    fn test_collector_reset() {
        let mut collector = StatisticsCollector::new(10);

        collector.begin_frame();
        collector.record(StatisticType::DrawCalls, 100.0);
        collector.end_frame();

        collector.reset();

        assert_eq!(collector.current_frame_number(), 0);
        assert!(collector.frame_history().is_empty());
        assert!(collector.get(StatisticType::DrawCalls).is_none());
    }

    #[test]
    fn test_collector_default() {
        let collector = StatisticsCollector::default();
        assert_eq!(collector.history_size(), 120);
    }

    // ==================== StatisticsReport Tests ====================

    #[test]
    fn test_report_from_collector() {
        let mut collector = StatisticsCollector::new(10);

        collector.begin_frame();
        collector.current_stats.frame_time_ms = 16.666;
        collector.current_stats.gpu_time_ms = 10.0;
        collector.current_stats.draw_calls = 100;
        collector.current_stats.triangles = 50000;
        collector.frame_history.push_back(collector.current_stats.clone());
        collector.current_frame = 1;
        collector.recording = false;

        let report = StatisticsReport::from_collector(&collector);

        assert_eq!(report.frame_count, 1);
        assert_eq!(report.total_draw_calls, 100);
        assert_eq!(report.total_triangles, 50000);
    }

    #[test]
    fn test_report_to_string() {
        let report = StatisticsReport::default();
        let s = report.to_string();

        assert!(s.contains("GPU Statistics Report"));
        assert!(s.contains("Frames:"));
        assert!(s.contains("Average FPS:"));
    }

    #[test]
    fn test_report_display() {
        let report = StatisticsReport::default();
        let s = format!("{}", report);
        assert!(s.contains("GPU Statistics Report"));
    }

    #[test]
    fn test_report_avg_draw_calls_per_frame() {
        let mut report = StatisticsReport::default();
        report.frame_count = 10;
        report.total_draw_calls = 1000;

        assert_eq!(report.avg_draw_calls_per_frame(), 100.0);
    }

    #[test]
    fn test_report_avg_triangles_per_frame() {
        let mut report = StatisticsReport::default();
        report.frame_count = 10;
        report.total_triangles = 500000;

        assert_eq!(report.avg_triangles_per_frame(), 50000.0);
    }

    #[test]
    fn test_report_gpu_utilization() {
        let mut report = StatisticsReport::default();
        report.avg_frame_time_ms = 16.666;
        report.avg_gpu_time_ms = 8.333;

        let util = report.gpu_utilization();
        assert!((util - 50.0).abs() < 0.1);
    }

    #[test]
    fn test_collector_min_max_frame_time() {
        let mut collector = StatisticsCollector::new(10);

        for time in [10.0, 20.0, 15.0, 25.0, 12.0] {
            collector.begin_frame();
            collector.current_stats.frame_time_ms = time;
            collector.frame_history.push_back(collector.current_stats.clone());
            collector.current_frame += 1;
            collector.recording = false;
        }

        assert_eq!(collector.min_frame_time(), 10.0);
        assert_eq!(collector.max_frame_time(), 25.0);
    }

    #[test]
    fn test_collector_recorded_types() {
        let mut collector = StatisticsCollector::new(10);
        collector.begin_frame();
        collector.record(StatisticType::GpuTime, 8.5);
        collector.record(StatisticType::MemoryUsage, 1024.0);
        collector.end_frame();

        let types: Vec<_> = collector.recorded_types().collect();
        assert!(types.contains(&&StatisticType::GpuTime));
        assert!(types.contains(&&StatisticType::MemoryUsage));
    }

    #[test]
    fn test_custom_statistic_type() {
        let custom1 = StatisticType::Custom(1);
        let custom2 = StatisticType::Custom(2);

        assert_ne!(custom1, custom2);
        assert_eq!(custom1.name(), "Custom");
        assert!(!custom1.is_timing());
        assert!(!custom1.is_count());
    }

    #[test]
    fn test_collector_end_frame_without_begin() {
        let mut collector = StatisticsCollector::new(10);

        // Should not crash or add to history
        collector.end_frame();

        assert_eq!(collector.frame_history().len(), 0);
        assert_eq!(collector.current_frame_number(), 0);
    }
}
