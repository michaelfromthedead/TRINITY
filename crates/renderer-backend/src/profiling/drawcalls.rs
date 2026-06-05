//! Draw Call Statistics Tracking for wgpu 25.x Rendering Analysis.
//!
//! This module provides comprehensive draw call statistics tracking for
//! measuring and analyzing GPU rendering workloads in the TRINITY engine.
//!
//! # Overview
//!
//! The module provides:
//! - **DrawType**: Enumeration of all draw/dispatch command types
//! - **DrawCall**: Individual draw call information
//! - **DrawBatch**: Batched draw calls for instancing analysis
//! - **PassStats**: Per-pass statistics aggregation
//! - **DrawFrameStats**: Per-frame aggregate statistics
//! - **DrawCallTracker**: Main interface for recording draw calls
//! - **DrawCallAnalyzer**: Analysis utilities for performance insights
//!
//! # Usage
//!
//! ```no_run
//! use renderer_backend::profiling::drawcalls::{
//!     DrawCallTracker, DrawType, PassType, DrawCallAnalyzer
//! };
//!
//! # fn example() {
//! let mut tracker = DrawCallTracker::new();
//!
//! // Begin frame tracking
//! tracker.begin_frame();
//!
//! // Begin a render pass
//! tracker.begin_pass(PassType::Render, Some("GBuffer"));
//!
//! // Record draw calls
//! tracker.record_draw_indexed(100, 300, 10); // 100 verts, 300 indices, 10 instances
//! tracker.record_pipeline_switch();
//! tracker.record_draw_indexed(50, 150, 5);
//!
//! // End pass
//! let pass_stats = tracker.end_pass();
//! println!("Pass drew {} vertices", pass_stats.unwrap().vertex_count);
//!
//! // End frame
//! let frame_stats = tracker.end_frame();
//! println!("Frame summary: {}", frame_stats.summary());
//!
//! // Analyze frame
//! let analysis = DrawCallAnalyzer::analyze_frame(&frame_stats);
//! for rec in analysis.recommendations() {
//!     println!("Recommendation: {}", rec);
//! }
//! # }
//! ```
//!
//! # Performance Considerations
//!
//! - Statistics are stored per-frame with configurable history depth
//! - Analysis methods are designed for offline/debug use
//! - Production builds can disable tracking via conditional compilation

use std::collections::VecDeque;
use std::fmt;
use std::time::{Duration, Instant};

// ============================================================================
// Constants
// ============================================================================

/// Default history size (frames to keep).
pub const DEFAULT_HISTORY_SIZE: usize = 120;

/// Minimum history size.
pub const MIN_HISTORY_SIZE: usize = 1;

/// Maximum recommended history size.
pub const MAX_HISTORY_SIZE: usize = 3600;

/// Threshold for state thrashing detection (switches per draw).
pub const STATE_THRASHING_THRESHOLD: f32 = 0.5;

/// Threshold for "heavy" pass detection (draws).
pub const HEAVY_PASS_DRAW_THRESHOLD: u64 = 1000;

/// Threshold for "heavy" pass detection (vertices).
pub const HEAVY_PASS_VERTEX_THRESHOLD: u64 = 1_000_000;

// ============================================================================
// DrawType
// ============================================================================

/// Types of draw and dispatch commands.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DrawType {
    /// Direct draw command (non-indexed).
    Draw,
    /// Indexed draw command.
    DrawIndexed,
    /// Indirect draw command (non-indexed).
    DrawIndirect,
    /// Indirect indexed draw command.
    DrawIndexedIndirect,
    /// Multi-draw indirect command (non-indexed).
    MultiDrawIndirect,
    /// Multi-draw indirect indexed command.
    MultiDrawIndexedIndirect,
    /// Compute dispatch command.
    Dispatch,
    /// Indirect compute dispatch command.
    DispatchIndirect,
}

impl DrawType {
    /// Returns true if this is an indexed draw type.
    #[inline]
    pub fn is_indexed(&self) -> bool {
        matches!(
            self,
            DrawType::DrawIndexed
                | DrawType::DrawIndexedIndirect
                | DrawType::MultiDrawIndexedIndirect
        )
    }

    /// Returns true if this is an indirect draw/dispatch type.
    #[inline]
    pub fn is_indirect(&self) -> bool {
        matches!(
            self,
            DrawType::DrawIndirect
                | DrawType::DrawIndexedIndirect
                | DrawType::MultiDrawIndirect
                | DrawType::MultiDrawIndexedIndirect
                | DrawType::DispatchIndirect
        )
    }

    /// Returns true if this is a compute dispatch type.
    #[inline]
    pub fn is_compute(&self) -> bool {
        matches!(self, DrawType::Dispatch | DrawType::DispatchIndirect)
    }

    /// Returns true if this is a render draw type (not compute).
    #[inline]
    pub fn is_render(&self) -> bool {
        !self.is_compute()
    }

    /// Returns the name of this draw type as a string.
    pub fn name(&self) -> &'static str {
        match self {
            DrawType::Draw => "Draw",
            DrawType::DrawIndexed => "DrawIndexed",
            DrawType::DrawIndirect => "DrawIndirect",
            DrawType::DrawIndexedIndirect => "DrawIndexedIndirect",
            DrawType::MultiDrawIndirect => "MultiDrawIndirect",
            DrawType::MultiDrawIndexedIndirect => "MultiDrawIndexedIndirect",
            DrawType::Dispatch => "Dispatch",
            DrawType::DispatchIndirect => "DispatchIndirect",
        }
    }
}

impl fmt::Display for DrawType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

// ============================================================================
// PrimitiveTopology
// ============================================================================

/// Primitive topology for calculating primitive counts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum PrimitiveTopology {
    /// Point list topology.
    PointList,
    /// Line list topology.
    LineList,
    /// Line strip topology.
    LineStrip,
    /// Triangle list topology (default).
    #[default]
    TriangleList,
    /// Triangle strip topology.
    TriangleStrip,
}

impl PrimitiveTopology {
    /// Calculate the number of primitives from vertex count.
    pub fn primitives_from_vertices(&self, vertex_count: u64) -> u64 {
        match self {
            PrimitiveTopology::PointList => vertex_count,
            PrimitiveTopology::LineList => vertex_count / 2,
            PrimitiveTopology::LineStrip => vertex_count.saturating_sub(1),
            PrimitiveTopology::TriangleList => vertex_count / 3,
            PrimitiveTopology::TriangleStrip => vertex_count.saturating_sub(2),
        }
    }
}

// ============================================================================
// PassType
// ============================================================================

/// Type of GPU pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum PassType {
    /// Render pass.
    #[default]
    Render,
    /// Compute pass.
    Compute,
}

impl fmt::Display for PassType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PassType::Render => write!(f, "Render"),
            PassType::Compute => write!(f, "Compute"),
        }
    }
}

// ============================================================================
// DrawCall
// ============================================================================

/// Information about a single draw call.
#[derive(Debug, Clone)]
pub struct DrawCall {
    /// Type of draw command.
    pub draw_type: DrawType,
    /// Number of vertices per instance.
    pub vertex_count: u32,
    /// Number of instances to draw.
    pub instance_count: u32,
    /// Number of indices (for indexed draws).
    pub index_count: Option<u32>,
    /// Pipeline ID (for tracking pipeline switches).
    pub pipeline_id: Option<u64>,
    /// Optional label for debugging.
    pub label: Option<String>,
    /// Timestamp when the draw was recorded.
    pub timestamp: Instant,
}

impl DrawCall {
    /// Create a new draw call.
    pub fn new(draw_type: DrawType, vertex_count: u32, instance_count: u32) -> Self {
        Self {
            draw_type,
            vertex_count,
            instance_count,
            index_count: None,
            pipeline_id: None,
            label: None,
            timestamp: Instant::now(),
        }
    }

    /// Create an indexed draw call.
    pub fn indexed(vertex_count: u32, index_count: u32, instance_count: u32) -> Self {
        Self {
            draw_type: DrawType::DrawIndexed,
            vertex_count,
            instance_count,
            index_count: Some(index_count),
            pipeline_id: None,
            label: None,
            timestamp: Instant::now(),
        }
    }

    /// Create a compute dispatch.
    pub fn dispatch(workgroups_x: u32, workgroups_y: u32, workgroups_z: u32) -> Self {
        // Store workgroup counts in vertex/instance counts for convenience
        Self {
            draw_type: DrawType::Dispatch,
            vertex_count: workgroups_x,
            instance_count: workgroups_y,
            index_count: Some(workgroups_z),
            pipeline_id: None,
            label: None,
            timestamp: Instant::now(),
        }
    }

    /// Set the pipeline ID.
    pub fn with_pipeline(mut self, pipeline_id: u64) -> Self {
        self.pipeline_id = Some(pipeline_id);
        self
    }

    /// Set the label.
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Calculate total vertices processed (vertex_count * instance_count).
    #[inline]
    pub fn total_vertices(&self) -> u64 {
        self.vertex_count as u64 * self.instance_count as u64
    }

    /// Calculate total primitives based on topology.
    pub fn total_primitives(&self, topology: PrimitiveTopology) -> u64 {
        let effective_count = if self.draw_type.is_indexed() {
            self.index_count.unwrap_or(self.vertex_count) as u64
        } else {
            self.vertex_count as u64
        };
        let primitives_per_instance = topology.primitives_from_vertices(effective_count);
        primitives_per_instance * self.instance_count as u64
    }

    /// Get workgroup counts for dispatch calls.
    pub fn workgroups(&self) -> Option<(u32, u32, u32)> {
        if self.draw_type.is_compute() {
            Some((
                self.vertex_count,
                self.instance_count,
                self.index_count.unwrap_or(1),
            ))
        } else {
            None
        }
    }
}

impl Default for DrawCall {
    fn default() -> Self {
        Self {
            draw_type: DrawType::Draw,
            vertex_count: 0,
            instance_count: 1,
            index_count: None,
            pipeline_id: None,
            label: None,
            timestamp: Instant::now(),
        }
    }
}

// ============================================================================
// DrawBatch
// ============================================================================

/// A batch of draw calls sharing the same pipeline.
#[derive(Debug, Clone)]
pub struct DrawBatch {
    /// Draw calls in this batch.
    pub draws: Vec<DrawCall>,
    /// Pipeline ID shared by all draws in the batch.
    pub pipeline_id: u64,
    /// Start time of the batch.
    pub start_time: Instant,
    /// End time of the batch.
    pub end_time: Option<Instant>,
}

impl DrawBatch {
    /// Create a new draw batch.
    pub fn new(pipeline_id: u64) -> Self {
        Self {
            draws: Vec::new(),
            pipeline_id,
            start_time: Instant::now(),
            end_time: None,
        }
    }

    /// Add a draw call to the batch.
    pub fn add_draw(&mut self, draw: DrawCall) {
        self.draws.push(draw);
    }

    /// Mark the batch as complete.
    pub fn finish(&mut self) {
        self.end_time = Some(Instant::now());
    }

    /// Get the total number of draw calls in the batch.
    #[inline]
    pub fn total_draw_count(&self) -> u64 {
        self.draws.len() as u64
    }

    /// Get the total vertex count across all draws.
    pub fn total_vertex_count(&self) -> u64 {
        self.draws.iter().map(|d| d.total_vertices()).sum()
    }

    /// Get the total instance count across all draws.
    pub fn total_instance_count(&self) -> u64 {
        self.draws.iter().map(|d| d.instance_count as u64).sum()
    }

    /// Get the average vertices per draw.
    pub fn avg_vertices_per_draw(&self) -> f32 {
        if self.draws.is_empty() {
            0.0
        } else {
            self.total_vertex_count() as f32 / self.draws.len() as f32
        }
    }

    /// Get the duration of the batch (if completed).
    pub fn duration(&self) -> Option<Duration> {
        self.end_time.map(|end| end.duration_since(self.start_time))
    }
}

impl Default for DrawBatch {
    fn default() -> Self {
        Self::new(0)
    }
}

// ============================================================================
// PassStats
// ============================================================================

/// Statistics for a single render or compute pass.
#[derive(Debug, Clone, Default)]
pub struct PassStats {
    /// Type of pass (render or compute).
    pub pass_type: PassType,
    /// Optional label for the pass.
    pub label: Option<String>,
    /// Total number of draw calls.
    pub draw_count: u64,
    /// Total vertex count (sum of vertex_count * instance_count).
    pub vertex_count: u64,
    /// Total instance count.
    pub instance_count: u64,
    /// Total number of compute dispatches.
    pub dispatch_count: u64,
    /// Total workgroup count (x * y * z for all dispatches).
    pub workgroup_count: (u32, u32, u32),
    /// Number of pipeline switches.
    pub pipeline_switches: u64,
    /// Number of bind group sets.
    pub bind_group_sets: u64,
    /// Duration of the pass.
    pub duration: Duration,
    /// Start time of the pass.
    start_time: Option<Instant>,
    /// Current pipeline ID (for tracking switches).
    current_pipeline: Option<u64>,
}

impl PassStats {
    /// Create new pass statistics.
    pub fn new(pass_type: PassType, label: Option<String>) -> Self {
        Self {
            pass_type,
            label,
            start_time: Some(Instant::now()),
            ..Default::default()
        }
    }

    /// Record a draw call.
    pub fn record_draw(&mut self, draw: &DrawCall) {
        if draw.draw_type.is_compute() {
            self.dispatch_count += 1;
            if let Some((x, y, z)) = draw.workgroups() {
                self.workgroup_count.0 = self.workgroup_count.0.saturating_add(x);
                self.workgroup_count.1 = self.workgroup_count.1.saturating_add(y);
                self.workgroup_count.2 = self.workgroup_count.2.saturating_add(z);
            }
        } else {
            self.draw_count += 1;
            self.vertex_count += draw.total_vertices();
            self.instance_count += draw.instance_count as u64;

            // Track pipeline switches
            if let Some(pipeline_id) = draw.pipeline_id {
                if self.current_pipeline != Some(pipeline_id) {
                    if self.current_pipeline.is_some() {
                        self.pipeline_switches += 1;
                    }
                    self.current_pipeline = Some(pipeline_id);
                }
            }
        }
    }

    /// Record a pipeline switch.
    pub fn record_pipeline_switch(&mut self) {
        self.pipeline_switches += 1;
    }

    /// Record a bind group set.
    pub fn record_bind_group_set(&mut self) {
        self.bind_group_sets += 1;
    }

    /// Finish the pass and calculate duration.
    pub fn finish(&mut self) {
        if let Some(start) = self.start_time {
            self.duration = start.elapsed();
        }
    }

    /// Get the total workgroup invocations.
    pub fn total_workgroup_invocations(&self) -> u64 {
        self.workgroup_count.0 as u64
            * self.workgroup_count.1 as u64
            * self.workgroup_count.2 as u64
    }

    /// Get state changes per draw ratio.
    pub fn state_changes_per_draw(&self) -> f32 {
        if self.draw_count == 0 {
            0.0
        } else {
            (self.pipeline_switches + self.bind_group_sets) as f32 / self.draw_count as f32
        }
    }

    /// Get average vertices per draw.
    pub fn avg_vertices_per_draw(&self) -> f32 {
        if self.draw_count == 0 {
            0.0
        } else {
            self.vertex_count as f32 / self.draw_count as f32
        }
    }

    /// Get average instances per draw.
    pub fn avg_instances_per_draw(&self) -> f32 {
        if self.draw_count == 0 {
            0.0
        } else {
            self.instance_count as f32 / self.draw_count as f32
        }
    }
}

impl fmt::Display for PassStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let label = self.label.as_deref().unwrap_or("unnamed");
        write!(
            f,
            "{} pass '{}': {} draws, {} vertices, {} dispatches, {:.2}ms",
            self.pass_type,
            label,
            self.draw_count,
            self.vertex_count,
            self.dispatch_count,
            self.duration.as_secs_f64() * 1000.0
        )
    }
}

// ============================================================================
// DrawFrameStats
// ============================================================================

/// Per-frame aggregate statistics for draw calls.
#[derive(Debug, Clone, Default)]
pub struct DrawFrameStats {
    /// Frame number.
    pub frame_number: u64,
    /// Statistics for each pass in the frame.
    pub passes: Vec<PassStats>,
    /// Total draw calls across all passes.
    pub total_draw_calls: u64,
    /// Total compute dispatches across all passes.
    pub total_dispatches: u64,
    /// Total vertices processed.
    pub total_vertices: u64,
    /// Total instances drawn.
    pub total_instances: u64,
    /// Total pipeline switches.
    pub total_pipeline_switches: u64,
    /// Total bind group sets.
    pub total_bind_group_sets: u64,
    /// Total frame time.
    pub frame_time: Duration,
    /// Frame start time.
    start_time: Option<Instant>,
}

impl DrawFrameStats {
    /// Create new frame statistics.
    pub fn new(frame_number: u64) -> Self {
        Self {
            frame_number,
            start_time: Some(Instant::now()),
            ..Default::default()
        }
    }

    /// Add pass statistics to the frame.
    pub fn add_pass(&mut self, pass: PassStats) {
        self.total_draw_calls += pass.draw_count;
        self.total_dispatches += pass.dispatch_count;
        self.total_vertices += pass.vertex_count;
        self.total_instances += pass.instance_count;
        self.total_pipeline_switches += pass.pipeline_switches;
        self.total_bind_group_sets += pass.bind_group_sets;
        self.passes.push(pass);
    }

    /// Finish the frame and calculate totals.
    pub fn finish(&mut self) {
        if let Some(start) = self.start_time {
            self.frame_time = start.elapsed();
        }
    }

    /// Get total state changes.
    pub fn total_state_changes(&self) -> u64 {
        self.total_pipeline_switches + self.total_bind_group_sets
    }

    /// Get state changes per draw ratio.
    pub fn state_changes_per_draw(&self) -> f32 {
        if self.total_draw_calls == 0 {
            0.0
        } else {
            self.total_state_changes() as f32 / self.total_draw_calls as f32
        }
    }

    /// Get average vertices per draw.
    pub fn avg_vertices_per_draw(&self) -> f32 {
        if self.total_draw_calls == 0 {
            0.0
        } else {
            self.total_vertices as f32 / self.total_draw_calls as f32
        }
    }

    /// Get draw calls per millisecond.
    pub fn draws_per_ms(&self) -> f32 {
        let ms = self.frame_time.as_secs_f32() * 1000.0;
        if ms <= 0.0 {
            0.0
        } else {
            self.total_draw_calls as f32 / ms
        }
    }

    /// Generate a summary string.
    pub fn summary(&self) -> String {
        format!(
            "Frame {}: {} draws, {} dispatches, {} vertices, {} instances, \
             {} pipeline switches, {} bind groups, {:.2}ms",
            self.frame_number,
            self.total_draw_calls,
            self.total_dispatches,
            self.total_vertices,
            self.total_instances,
            self.total_pipeline_switches,
            self.total_bind_group_sets,
            self.frame_time.as_secs_f64() * 1000.0
        )
    }
}

impl fmt::Display for DrawFrameStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.summary())
    }
}

// ============================================================================
// DrawCallTracker
// ============================================================================

/// Main interface for tracking draw call statistics.
#[derive(Debug)]
pub struct DrawCallTracker {
    /// Current frame number.
    current_frame: u64,
    /// Current pass being recorded.
    current_pass: Option<PassStats>,
    /// Current frame statistics.
    frame_stats: DrawFrameStats,
    /// History of past frame statistics.
    history: VecDeque<DrawFrameStats>,
    /// Maximum history size.
    history_size: usize,
    /// Whether tracking is enabled.
    enabled: bool,
}

impl DrawCallTracker {
    /// Create a new draw call tracker with default history size.
    pub fn new() -> Self {
        Self {
            current_frame: 0,
            current_pass: None,
            frame_stats: DrawFrameStats::new(0),
            history: VecDeque::with_capacity(DEFAULT_HISTORY_SIZE),
            history_size: DEFAULT_HISTORY_SIZE,
            enabled: true,
        }
    }

    /// Create a new tracker with specified history size.
    pub fn with_history_size(size: usize) -> Self {
        let size = size.clamp(MIN_HISTORY_SIZE, MAX_HISTORY_SIZE);
        Self {
            current_frame: 0,
            current_pass: None,
            frame_stats: DrawFrameStats::new(0),
            history: VecDeque::with_capacity(size),
            history_size: size,
            enabled: true,
        }
    }

    /// Enable or disable tracking.
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Check if tracking is enabled.
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Begin a new frame.
    pub fn begin_frame(&mut self) {
        if !self.enabled {
            return;
        }

        // End any open pass
        if self.current_pass.is_some() {
            self.end_pass();
        }

        self.frame_stats = DrawFrameStats::new(self.current_frame);
    }

    /// End the current frame and return statistics.
    pub fn end_frame(&mut self) -> DrawFrameStats {
        if !self.enabled {
            return DrawFrameStats::default();
        }

        // End any open pass
        if self.current_pass.is_some() {
            self.end_pass();
        }

        // Finish frame stats
        self.frame_stats.finish();

        // Store in history
        let stats = std::mem::replace(
            &mut self.frame_stats,
            DrawFrameStats::new(self.current_frame + 1),
        );

        if self.history.len() >= self.history_size {
            self.history.pop_front();
        }
        let result = stats.clone();
        self.history.push_back(stats);

        // Increment frame counter
        self.current_frame += 1;

        result
    }

    /// Begin a new pass.
    pub fn begin_pass(&mut self, pass_type: PassType, label: Option<&str>) {
        if !self.enabled {
            return;
        }

        // End any open pass first
        if self.current_pass.is_some() {
            self.end_pass();
        }

        self.current_pass = Some(PassStats::new(pass_type, label.map(String::from)));
    }

    /// End the current pass and return statistics.
    pub fn end_pass(&mut self) -> Option<PassStats> {
        if !self.enabled {
            return None;
        }

        if let Some(mut pass) = self.current_pass.take() {
            pass.finish();
            let result = pass.clone();
            self.frame_stats.add_pass(pass);
            Some(result)
        } else {
            None
        }
    }

    /// Record a draw call.
    pub fn record_draw(&mut self, draw_call: DrawCall) {
        if !self.enabled {
            return;
        }

        if let Some(ref mut pass) = self.current_pass {
            pass.record_draw(&draw_call);
        }
    }

    /// Record an indexed draw call.
    pub fn record_draw_indexed(
        &mut self,
        vertex_count: u32,
        index_count: u32,
        instance_count: u32,
    ) {
        if !self.enabled {
            return;
        }

        let draw = DrawCall::indexed(vertex_count, index_count, instance_count);
        self.record_draw(draw);
    }

    /// Record a non-indexed draw call.
    pub fn record_draw_non_indexed(&mut self, vertex_count: u32, instance_count: u32) {
        if !self.enabled {
            return;
        }

        let draw = DrawCall::new(DrawType::Draw, vertex_count, instance_count);
        self.record_draw(draw);
    }

    /// Record a compute dispatch.
    pub fn record_dispatch(&mut self, x: u32, y: u32, z: u32) {
        if !self.enabled {
            return;
        }

        let draw = DrawCall::dispatch(x, y, z);
        self.record_draw(draw);
    }

    /// Record a pipeline switch.
    pub fn record_pipeline_switch(&mut self) {
        if !self.enabled {
            return;
        }

        if let Some(ref mut pass) = self.current_pass {
            pass.record_pipeline_switch();
        }
    }

    /// Record a bind group set.
    pub fn record_bind_group_set(&mut self) {
        if !self.enabled {
            return;
        }

        if let Some(ref mut pass) = self.current_pass {
            pass.record_bind_group_set();
        }
    }

    /// Get the current frame statistics (in progress).
    pub fn current_frame_stats(&self) -> &DrawFrameStats {
        &self.frame_stats
    }

    /// Get the frame history.
    pub fn history(&self) -> &VecDeque<DrawFrameStats> {
        &self.history
    }

    /// Get the current frame number.
    pub fn current_frame_number(&self) -> u64 {
        self.current_frame
    }

    /// Calculate average draw calls per frame from history.
    pub fn avg_draw_calls_per_frame(&self) -> f32 {
        if self.history.is_empty() {
            0.0
        } else {
            let total: u64 = self.history.iter().map(|f| f.total_draw_calls).sum();
            total as f32 / self.history.len() as f32
        }
    }

    /// Calculate average vertices per frame from history.
    pub fn avg_vertices_per_frame(&self) -> f32 {
        if self.history.is_empty() {
            0.0
        } else {
            let total: u64 = self.history.iter().map(|f| f.total_vertices).sum();
            total as f32 / self.history.len() as f32
        }
    }

    /// Calculate average frame time from history.
    pub fn avg_frame_time_ms(&self) -> f32 {
        if self.history.is_empty() {
            0.0
        } else {
            let total: f64 = self
                .history
                .iter()
                .map(|f| f.frame_time.as_secs_f64() * 1000.0)
                .sum();
            total as f32 / self.history.len() as f32
        }
    }

    /// Calculate average dispatches per frame from history.
    pub fn avg_dispatches_per_frame(&self) -> f32 {
        if self.history.is_empty() {
            0.0
        } else {
            let total: u64 = self.history.iter().map(|f| f.total_dispatches).sum();
            total as f32 / self.history.len() as f32
        }
    }

    /// Get the most recent frame statistics.
    pub fn last_frame_stats(&self) -> Option<&DrawFrameStats> {
        self.history.back()
    }

    /// Clear history.
    pub fn clear_history(&mut self) {
        self.history.clear();
    }

    /// Reset the tracker completely.
    pub fn reset(&mut self) {
        self.current_frame = 0;
        self.current_pass = None;
        self.frame_stats = DrawFrameStats::new(0);
        self.history.clear();
    }
}

impl Default for DrawCallTracker {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// FrameAnalysis
// ============================================================================

/// Analysis results for a frame.
#[derive(Debug, Clone)]
pub struct FrameAnalysis {
    /// Frame number analyzed.
    pub frame_number: u64,
    /// Draw call density (draws per ms).
    pub draw_call_density: f32,
    /// State change ratio (state changes per draw).
    pub state_change_ratio: f32,
    /// Batching score (0-1, higher is better).
    pub batching_score: f32,
    /// Identified bottlenecks.
    pub bottlenecks: Vec<String>,
    /// Total draw calls.
    total_draws: u64,
    /// Total vertices.
    total_vertices: u64,
    /// Average vertices per draw.
    avg_vertices_per_draw: f32,
}

impl FrameAnalysis {
    /// Create a new frame analysis.
    fn new(stats: &DrawFrameStats) -> Self {
        let draw_call_density = stats.draws_per_ms();
        let state_change_ratio = stats.state_changes_per_draw();
        let avg_verts = stats.avg_vertices_per_draw();

        // Calculate batching score (higher avg verts = better batching)
        // Score is normalized assuming 1000 avg verts is "good"
        let batching_score = (avg_verts / 1000.0).min(1.0);

        Self {
            frame_number: stats.frame_number,
            draw_call_density,
            state_change_ratio,
            batching_score,
            bottlenecks: Vec::new(),
            total_draws: stats.total_draw_calls,
            total_vertices: stats.total_vertices,
            avg_vertices_per_draw: avg_verts,
        }
    }

    /// Returns true if the frame appears CPU-bound (many small draws).
    pub fn is_cpu_bound(&self) -> bool {
        // Many small draws with low batching suggests CPU bound
        self.total_draws > 1000 && self.avg_vertices_per_draw < 100.0
    }

    /// Returns true if the frame appears GPU-bound (few large draws).
    pub fn is_gpu_bound(&self) -> bool {
        // Few large draws with high vertex counts suggests GPU bound
        self.total_draws < 100 && self.total_vertices > 1_000_000
    }

    /// Generate recommendations based on analysis.
    pub fn recommendations(&self) -> Vec<String> {
        let mut recs = Vec::new();

        if self.is_cpu_bound() {
            recs.push("Consider instancing or batching to reduce draw calls".to_string());
        }

        if self.state_change_ratio > STATE_THRASHING_THRESHOLD {
            recs.push(format!(
                "High state change ratio ({:.2}), consider sorting draws by material",
                self.state_change_ratio
            ));
        }

        if self.batching_score < 0.3 {
            recs.push("Low batching efficiency, consider merging small meshes".to_string());
        }

        if self.draw_call_density > 100.0 {
            recs.push(
                "Very high draw call density, consider GPU-driven rendering".to_string(),
            );
        }

        recs.append(&mut self.bottlenecks.clone());
        recs
    }
}

impl fmt::Display for FrameAnalysis {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Frame {} Analysis: density={:.1} draws/ms, state_ratio={:.2}, \
             batching={:.2}, cpu_bound={}, gpu_bound={}",
            self.frame_number,
            self.draw_call_density,
            self.state_change_ratio,
            self.batching_score,
            self.is_cpu_bound(),
            self.is_gpu_bound()
        )
    }
}

// ============================================================================
// DrawCallAnalyzer
// ============================================================================

/// Utility for analyzing draw call statistics.
pub struct DrawCallAnalyzer;

impl DrawCallAnalyzer {
    /// Analyze a frame's statistics.
    pub fn analyze_frame(stats: &DrawFrameStats) -> FrameAnalysis {
        let mut analysis = FrameAnalysis::new(stats);

        // Identify bottlenecks
        if stats.total_draw_calls > 10000 {
            analysis.bottlenecks.push(format!(
                "Excessive draw calls: {}",
                stats.total_draw_calls
            ));
        }

        if stats.total_pipeline_switches > stats.total_draw_calls / 2 {
            analysis.bottlenecks.push(
                "Too many pipeline switches relative to draw count".to_string(),
            );
        }

        // Check for passes with issues
        for pass in &stats.passes {
            if pass.state_changes_per_draw() > 1.0 {
                analysis.bottlenecks.push(format!(
                    "Pass '{}' has excessive state changes ({:.2} per draw)",
                    pass.label.as_deref().unwrap_or("unnamed"),
                    pass.state_changes_per_draw()
                ));
            }
        }

        analysis
    }

    /// Find passes that exceed the given draw threshold.
    pub fn find_heavy_passes<'a>(
        stats: &'a DrawFrameStats,
        draw_threshold: u64,
    ) -> Vec<&'a PassStats> {
        stats
            .passes
            .iter()
            .filter(|p| p.draw_count >= draw_threshold || p.vertex_count >= HEAVY_PASS_VERTEX_THRESHOLD)
            .collect()
    }

    /// Detect if state thrashing is occurring.
    pub fn detect_state_thrashing(stats: &DrawFrameStats) -> bool {
        stats.state_changes_per_draw() > STATE_THRASHING_THRESHOLD
    }

    /// Calculate batching efficiency (0-1).
    pub fn batching_efficiency(stats: &DrawFrameStats) -> f32 {
        if stats.total_draw_calls == 0 {
            return 1.0; // No draws = perfect efficiency
        }

        let avg_verts = stats.avg_vertices_per_draw();

        // Assume 1000 avg verts is "efficient", scale down from there
        (avg_verts / 1000.0).min(1.0)
    }

    /// Generate recommendations for improving performance.
    pub fn recommendations(stats: &DrawFrameStats) -> Vec<String> {
        let analysis = Self::analyze_frame(stats);
        analysis.recommendations()
    }

    /// Compare two frames and identify changes.
    pub fn compare_frames(
        frame_a: &DrawFrameStats,
        frame_b: &DrawFrameStats,
    ) -> FrameComparison {
        FrameComparison {
            frame_a: frame_a.frame_number,
            frame_b: frame_b.frame_number,
            draw_delta: frame_b.total_draw_calls as i64 - frame_a.total_draw_calls as i64,
            vertex_delta: frame_b.total_vertices as i64 - frame_a.total_vertices as i64,
            time_delta_ms: (frame_b.frame_time.as_secs_f64() - frame_a.frame_time.as_secs_f64())
                * 1000.0,
            state_change_delta: frame_b.total_state_changes() as i64
                - frame_a.total_state_changes() as i64,
        }
    }

    /// Analyze trends across multiple frames.
    pub fn analyze_trend(history: &VecDeque<DrawFrameStats>) -> TrendAnalysis {
        if history.is_empty() {
            return TrendAnalysis::default();
        }

        let draw_counts: Vec<f64> = history
            .iter()
            .map(|f| f.total_draw_calls as f64)
            .collect();
        let frame_times: Vec<f64> = history
            .iter()
            .map(|f| f.frame_time.as_secs_f64() * 1000.0)
            .collect();

        let avg_draws = draw_counts.iter().sum::<f64>() / draw_counts.len() as f64;
        let avg_time = frame_times.iter().sum::<f64>() / frame_times.len() as f64;

        // Calculate variance
        let draw_variance = draw_counts
            .iter()
            .map(|x| (x - avg_draws).powi(2))
            .sum::<f64>()
            / draw_counts.len() as f64;
        let time_variance = frame_times
            .iter()
            .map(|x| (x - avg_time).powi(2))
            .sum::<f64>()
            / frame_times.len() as f64;

        // Calculate trend (simple linear regression slope approximation)
        let n = draw_counts.len() as f64;
        let x_sum: f64 = (0..draw_counts.len()).map(|i| i as f64).sum();
        let x_mean = x_sum / n;

        let draw_slope = if n > 1.0 {
            let numerator: f64 = draw_counts
                .iter()
                .enumerate()
                .map(|(i, &y)| (i as f64 - x_mean) * (y - avg_draws))
                .sum();
            let denominator: f64 = (0..draw_counts.len())
                .map(|i| (i as f64 - x_mean).powi(2))
                .sum();
            if denominator > 0.0 {
                numerator / denominator
            } else {
                0.0
            }
        } else {
            0.0
        };

        TrendAnalysis {
            avg_draw_calls: avg_draws as f32,
            avg_frame_time_ms: avg_time as f32,
            draw_variance: draw_variance as f32,
            time_variance: time_variance as f32,
            draw_trend: draw_slope as f32,
            samples: history.len(),
        }
    }
}

// ============================================================================
// FrameComparison
// ============================================================================

/// Comparison between two frames.
#[derive(Debug, Clone, Default)]
pub struct FrameComparison {
    /// First frame number.
    pub frame_a: u64,
    /// Second frame number.
    pub frame_b: u64,
    /// Change in draw calls.
    pub draw_delta: i64,
    /// Change in vertex count.
    pub vertex_delta: i64,
    /// Change in frame time (ms).
    pub time_delta_ms: f64,
    /// Change in state changes.
    pub state_change_delta: i64,
}

impl fmt::Display for FrameComparison {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Frame {} vs {}: draws {:+}, vertices {:+}, time {:+.2}ms, state {:+}",
            self.frame_a,
            self.frame_b,
            self.draw_delta,
            self.vertex_delta,
            self.time_delta_ms,
            self.state_change_delta
        )
    }
}

// ============================================================================
// TrendAnalysis
// ============================================================================

/// Analysis of trends across multiple frames.
#[derive(Debug, Clone, Default)]
pub struct TrendAnalysis {
    /// Average draw calls per frame.
    pub avg_draw_calls: f32,
    /// Average frame time in milliseconds.
    pub avg_frame_time_ms: f32,
    /// Variance in draw calls.
    pub draw_variance: f32,
    /// Variance in frame time.
    pub time_variance: f32,
    /// Trend in draw calls (positive = increasing).
    pub draw_trend: f32,
    /// Number of samples analyzed.
    pub samples: usize,
}

impl TrendAnalysis {
    /// Returns true if draw calls are increasing over time.
    pub fn is_increasing(&self) -> bool {
        self.draw_trend > 0.5
    }

    /// Returns true if draw calls are decreasing over time.
    pub fn is_decreasing(&self) -> bool {
        self.draw_trend < -0.5
    }

    /// Returns true if the workload is stable.
    pub fn is_stable(&self) -> bool {
        self.draw_variance < 100.0 && self.time_variance < 1.0
    }
}

impl fmt::Display for TrendAnalysis {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let trend = if self.is_increasing() {
            "increasing"
        } else if self.is_decreasing() {
            "decreasing"
        } else {
            "stable"
        };

        write!(
            f,
            "Trend ({} samples): avg {:.0} draws, {:.2}ms, {} (slope: {:.2})",
            self.samples, self.avg_draw_calls, self.avg_frame_time_ms, trend, self.draw_trend
        )
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_draw_type_is_indexed() {
        assert!(!DrawType::Draw.is_indexed());
        assert!(DrawType::DrawIndexed.is_indexed());
        assert!(!DrawType::DrawIndirect.is_indexed());
        assert!(DrawType::DrawIndexedIndirect.is_indexed());
        assert!(!DrawType::MultiDrawIndirect.is_indexed());
        assert!(DrawType::MultiDrawIndexedIndirect.is_indexed());
        assert!(!DrawType::Dispatch.is_indexed());
        assert!(!DrawType::DispatchIndirect.is_indexed());
    }

    #[test]
    fn test_draw_type_is_indirect() {
        assert!(!DrawType::Draw.is_indirect());
        assert!(!DrawType::DrawIndexed.is_indirect());
        assert!(DrawType::DrawIndirect.is_indirect());
        assert!(DrawType::DrawIndexedIndirect.is_indirect());
        assert!(DrawType::MultiDrawIndirect.is_indirect());
        assert!(DrawType::MultiDrawIndexedIndirect.is_indirect());
        assert!(!DrawType::Dispatch.is_indirect());
        assert!(DrawType::DispatchIndirect.is_indirect());
    }

    #[test]
    fn test_draw_type_is_compute() {
        assert!(!DrawType::Draw.is_compute());
        assert!(!DrawType::DrawIndexed.is_compute());
        assert!(!DrawType::DrawIndirect.is_compute());
        assert!(DrawType::Dispatch.is_compute());
        assert!(DrawType::DispatchIndirect.is_compute());
    }

    #[test]
    fn test_draw_type_display() {
        assert_eq!(DrawType::Draw.to_string(), "Draw");
        assert_eq!(DrawType::DrawIndexed.to_string(), "DrawIndexed");
        assert_eq!(DrawType::Dispatch.to_string(), "Dispatch");
    }

    #[test]
    fn test_primitive_topology_calculation() {
        assert_eq!(PrimitiveTopology::PointList.primitives_from_vertices(10), 10);
        assert_eq!(PrimitiveTopology::LineList.primitives_from_vertices(10), 5);
        assert_eq!(PrimitiveTopology::LineStrip.primitives_from_vertices(10), 9);
        assert_eq!(PrimitiveTopology::TriangleList.primitives_from_vertices(12), 4);
        assert_eq!(PrimitiveTopology::TriangleStrip.primitives_from_vertices(10), 8);
    }

    #[test]
    fn test_draw_call_total_vertices() {
        let draw = DrawCall::new(DrawType::Draw, 100, 10);
        assert_eq!(draw.total_vertices(), 1000);
    }

    #[test]
    fn test_draw_call_indexed() {
        let draw = DrawCall::indexed(100, 300, 5);
        assert!(draw.draw_type.is_indexed());
        assert_eq!(draw.index_count, Some(300));
        assert_eq!(draw.total_vertices(), 500);
    }

    #[test]
    fn test_draw_call_dispatch() {
        let draw = DrawCall::dispatch(8, 8, 1);
        assert!(draw.draw_type.is_compute());
        assert_eq!(draw.workgroups(), Some((8, 8, 1)));
    }

    #[test]
    fn test_draw_call_primitives() {
        let draw = DrawCall::new(DrawType::Draw, 12, 1);
        assert_eq!(draw.total_primitives(PrimitiveTopology::TriangleList), 4);
    }

    #[test]
    fn test_draw_batch() {
        let mut batch = DrawBatch::new(42);
        batch.add_draw(DrawCall::new(DrawType::Draw, 100, 1));
        batch.add_draw(DrawCall::new(DrawType::Draw, 200, 2));
        batch.finish();

        assert_eq!(batch.total_draw_count(), 2);
        assert_eq!(batch.total_vertex_count(), 500); // 100 + 200*2
        assert!(batch.duration().is_some());
        assert_eq!(batch.avg_vertices_per_draw(), 250.0);
    }

    #[test]
    fn test_pass_stats() {
        let mut pass = PassStats::new(PassType::Render, Some("test".to_string()));

        pass.record_draw(&DrawCall::new(DrawType::Draw, 100, 1));
        pass.record_draw(&DrawCall::indexed(50, 150, 2));
        pass.record_pipeline_switch();
        pass.record_bind_group_set();
        pass.finish();

        assert_eq!(pass.draw_count, 2);
        assert_eq!(pass.vertex_count, 200); // 100 + 50*2
        assert_eq!(pass.pipeline_switches, 1);
        assert_eq!(pass.bind_group_sets, 1);
    }

    #[test]
    fn test_pass_stats_compute() {
        let mut pass = PassStats::new(PassType::Compute, Some("compute".to_string()));

        pass.record_draw(&DrawCall::dispatch(8, 8, 4));
        pass.record_draw(&DrawCall::dispatch(16, 16, 1));
        pass.finish();

        assert_eq!(pass.dispatch_count, 2);
        assert_eq!(pass.draw_count, 0);
    }

    #[test]
    fn test_frame_stats() {
        let mut frame = DrawFrameStats::new(0);

        let mut pass1 = PassStats::new(PassType::Render, Some("pass1".to_string()));
        pass1.record_draw(&DrawCall::new(DrawType::Draw, 100, 1));
        pass1.finish();
        frame.add_pass(pass1);

        let mut pass2 = PassStats::new(PassType::Render, Some("pass2".to_string()));
        pass2.record_draw(&DrawCall::new(DrawType::Draw, 200, 2));
        pass2.finish();
        frame.add_pass(pass2);

        frame.finish();

        assert_eq!(frame.passes.len(), 2);
        assert_eq!(frame.total_draw_calls, 2);
        assert_eq!(frame.total_vertices, 500);
    }

    #[test]
    fn test_draw_call_tracker_basic() {
        let mut tracker = DrawCallTracker::new();

        tracker.begin_frame();
        tracker.begin_pass(PassType::Render, Some("main"));
        tracker.record_draw_indexed(100, 300, 1);
        tracker.record_draw_indexed(50, 150, 2);
        tracker.end_pass();
        let stats = tracker.end_frame();

        assert_eq!(stats.total_draw_calls, 2);
        assert_eq!(stats.total_vertices, 200); // 100 + 50*2
        assert_eq!(stats.passes.len(), 1);
    }

    #[test]
    fn test_draw_call_tracker_history() {
        let mut tracker = DrawCallTracker::with_history_size(5);

        for i in 0..10 {
            tracker.begin_frame();
            tracker.begin_pass(PassType::Render, None);
            tracker.record_draw_non_indexed((i + 1) * 10, 1);
            tracker.end_pass();
            tracker.end_frame();
        }

        assert_eq!(tracker.history().len(), 5);
        assert_eq!(tracker.current_frame_number(), 10);
    }

    #[test]
    fn test_draw_call_tracker_averages() {
        let mut tracker = DrawCallTracker::with_history_size(10);

        for _ in 0..5 {
            tracker.begin_frame();
            tracker.begin_pass(PassType::Render, None);
            tracker.record_draw_non_indexed(100, 1);
            tracker.record_draw_non_indexed(100, 1);
            tracker.end_pass();
            tracker.end_frame();
        }

        assert_eq!(tracker.avg_draw_calls_per_frame(), 2.0);
        assert_eq!(tracker.avg_vertices_per_frame(), 200.0);
    }

    #[test]
    fn test_draw_call_tracker_disabled() {
        let mut tracker = DrawCallTracker::new();
        tracker.set_enabled(false);

        tracker.begin_frame();
        tracker.begin_pass(PassType::Render, None);
        tracker.record_draw_non_indexed(100, 1);
        let stats = tracker.end_frame();

        assert_eq!(stats.total_draw_calls, 0);
    }

    #[test]
    fn test_frame_analysis() {
        let mut frame = DrawFrameStats::new(0);
        let mut pass = PassStats::new(PassType::Render, None);

        for _ in 0..100 {
            pass.record_draw(&DrawCall::new(DrawType::Draw, 10, 1));
        }
        pass.finish();
        frame.add_pass(pass);
        frame.finish();

        let analysis = DrawCallAnalyzer::analyze_frame(&frame);
        assert_eq!(analysis.frame_number, 0);
        assert!(analysis.batching_score < 0.1); // Low vertices per draw
    }

    #[test]
    fn test_frame_analysis_cpu_bound() {
        let mut frame = DrawFrameStats::new(0);
        let mut pass = PassStats::new(PassType::Render, None);

        // Many small draws = CPU bound
        for _ in 0..2000 {
            pass.record_draw(&DrawCall::new(DrawType::Draw, 10, 1));
        }
        pass.finish();
        frame.add_pass(pass);
        frame.finish();

        let analysis = DrawCallAnalyzer::analyze_frame(&frame);
        assert!(analysis.is_cpu_bound());
        assert!(!analysis.is_gpu_bound());
    }

    #[test]
    fn test_state_thrashing_detection() {
        let mut frame = DrawFrameStats::new(0);
        let mut pass = PassStats::new(PassType::Render, None);

        // Many state changes relative to draws
        pass.record_draw(&DrawCall::new(DrawType::Draw, 100, 1));
        pass.record_pipeline_switch();
        pass.record_bind_group_set();
        pass.record_draw(&DrawCall::new(DrawType::Draw, 100, 1));
        pass.finish();
        frame.add_pass(pass);
        frame.finish();

        assert!(DrawCallAnalyzer::detect_state_thrashing(&frame));
    }

    #[test]
    fn test_batching_efficiency() {
        let mut frame = DrawFrameStats::new(0);
        let mut pass = PassStats::new(PassType::Render, None);

        // High vertex count per draw = good batching
        pass.record_draw(&DrawCall::new(DrawType::Draw, 5000, 1));
        pass.finish();
        frame.add_pass(pass);
        frame.finish();

        let efficiency = DrawCallAnalyzer::batching_efficiency(&frame);
        assert!(efficiency >= 1.0);
    }

    #[test]
    fn test_find_heavy_passes() {
        let mut frame = DrawFrameStats::new(0);

        let mut light_pass = PassStats::new(PassType::Render, Some("light".to_string()));
        for _ in 0..10 {
            light_pass.record_draw(&DrawCall::new(DrawType::Draw, 100, 1));
        }
        light_pass.finish();
        frame.add_pass(light_pass);

        let mut heavy_pass = PassStats::new(PassType::Render, Some("heavy".to_string()));
        for _ in 0..2000 {
            heavy_pass.record_draw(&DrawCall::new(DrawType::Draw, 100, 1));
        }
        heavy_pass.finish();
        frame.add_pass(heavy_pass);
        frame.finish();

        let heavy = DrawCallAnalyzer::find_heavy_passes(&frame, 1000);
        assert_eq!(heavy.len(), 1);
        assert_eq!(heavy[0].label.as_deref(), Some("heavy"));
    }

    #[test]
    fn test_frame_comparison() {
        let mut frame_a = DrawFrameStats::new(0);
        let mut pass = PassStats::new(PassType::Render, None);
        pass.record_draw(&DrawCall::new(DrawType::Draw, 100, 1));
        pass.finish();
        frame_a.add_pass(pass);
        frame_a.finish();

        let mut frame_b = DrawFrameStats::new(1);
        let mut pass = PassStats::new(PassType::Render, None);
        pass.record_draw(&DrawCall::new(DrawType::Draw, 200, 2));
        pass.record_draw(&DrawCall::new(DrawType::Draw, 200, 2));
        pass.finish();
        frame_b.add_pass(pass);
        frame_b.finish();

        let comparison = DrawCallAnalyzer::compare_frames(&frame_a, &frame_b);
        assert_eq!(comparison.draw_delta, 1);
        assert_eq!(comparison.vertex_delta, 700); // 800 - 100
    }

    #[test]
    fn test_trend_analysis() {
        let mut history = VecDeque::new();

        for i in 0..10 {
            let mut frame = DrawFrameStats::new(i);
            let mut pass = PassStats::new(PassType::Render, None);
            // Increasing draw count trend
            for _ in 0..(i + 1) {
                pass.record_draw(&DrawCall::new(DrawType::Draw, 100, 1));
            }
            pass.finish();
            frame.add_pass(pass);
            frame.finish();
            history.push_back(frame);
        }

        let trend = DrawCallAnalyzer::analyze_trend(&history);
        assert!(trend.is_increasing());
        assert_eq!(trend.samples, 10);
    }

    #[test]
    fn test_recommendations() {
        let mut frame = DrawFrameStats::new(0);
        let mut pass = PassStats::new(PassType::Render, None);

        // Create conditions for recommendations
        for _ in 0..2000 {
            pass.record_draw(&DrawCall::new(DrawType::Draw, 10, 1));
            pass.record_pipeline_switch();
        }
        pass.finish();
        frame.add_pass(pass);
        frame.finish();

        let recs = DrawCallAnalyzer::recommendations(&frame);
        assert!(!recs.is_empty());
    }

    #[test]
    fn test_tracker_reset() {
        let mut tracker = DrawCallTracker::new();

        tracker.begin_frame();
        tracker.begin_pass(PassType::Render, None);
        tracker.record_draw_non_indexed(100, 1);
        tracker.end_pass();
        tracker.end_frame();

        assert_eq!(tracker.history().len(), 1);

        tracker.reset();

        assert_eq!(tracker.current_frame_number(), 0);
        assert!(tracker.history().is_empty());
    }

    #[test]
    fn test_pass_type_display() {
        assert_eq!(PassType::Render.to_string(), "Render");
        assert_eq!(PassType::Compute.to_string(), "Compute");
    }

    #[test]
    fn test_draw_call_with_builder_pattern() {
        let draw = DrawCall::new(DrawType::Draw, 100, 1)
            .with_pipeline(42)
            .with_label("test_draw");

        assert_eq!(draw.pipeline_id, Some(42));
        assert_eq!(draw.label.as_deref(), Some("test_draw"));
    }
}
