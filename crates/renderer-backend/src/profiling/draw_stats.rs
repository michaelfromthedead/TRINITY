//! Draw Call Statistics Tracking for Performance Analysis.
//!
//! This module provides lightweight draw call statistics tracking designed for
//! real-time performance analysis of GPU rendering workloads.
//!
//! # Overview
//!
//! - [`DrawCallType`]: Enumeration of draw call command types
//! - [`DrawCallInfo`]: Statistics for a single draw call
//! - [`FrameDrawStats`]: Frame-level aggregate statistics
//! - [`DrawStatsCollector`]: Main interface for tracking statistics
//! - [`DrawStatsSummary`]: Summary statistics across frames
//!
//! # Usage
//!
//! ```no_run
//! use renderer_backend::profiling::draw_stats::{
//!     DrawStatsCollector, DrawCallType, FrameDrawStats,
//! };
//!
//! # fn example() {
//! let mut collector = DrawStatsCollector::new();
//!
//! // Record draw calls during frame
//! collector.record_draw(100, 1);
//! collector.record_draw_indexed(300, 10);
//! collector.record_draw_indirect(DrawCallType::DrawIndirect, 5);
//!
//! // End frame and get statistics
//! let frame_stats = collector.end_frame();
//! println!("Frame: {} draw calls, {} vertices",
//!     frame_stats.total_draw_calls,
//!     frame_stats.total_vertices);
//!
//! // Get summary across history
//! let summary = collector.summary();
//! println!("Average: {:.1} draw calls/frame", summary.avg_draw_calls);
//! # }
//! ```
//!
//! # Performance Considerations
//!
//! - Statistics collection is designed for minimal overhead
//! - Can be disabled via `disable()` for release builds
//! - Frame history is bounded by configurable size
//! - All operations are O(1) except summary aggregation

use std::collections::HashMap;

// ============================================================================
// Constants
// ============================================================================

/// Default history size (frames to keep).
pub const DEFAULT_HISTORY_SIZE: usize = 120;

/// Minimum history size.
pub const MIN_HISTORY_SIZE: usize = 1;

/// Maximum recommended history size.
pub const MAX_HISTORY_SIZE: usize = 3600;

// ============================================================================
// DrawCallType
// ============================================================================

/// Type of draw call command.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DrawCallType {
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
}

impl DrawCallType {
    /// Returns the name of this draw call type as a string.
    #[inline]
    pub fn name(&self) -> &'static str {
        match self {
            DrawCallType::Draw => "Draw",
            DrawCallType::DrawIndexed => "DrawIndexed",
            DrawCallType::DrawIndirect => "DrawIndirect",
            DrawCallType::DrawIndexedIndirect => "DrawIndexedIndirect",
            DrawCallType::MultiDrawIndirect => "MultiDrawIndirect",
            DrawCallType::MultiDrawIndexedIndirect => "MultiDrawIndexedIndirect",
        }
    }

    /// Returns true if this is an indexed draw type.
    #[inline]
    pub fn is_indexed(&self) -> bool {
        matches!(
            self,
            DrawCallType::DrawIndexed
                | DrawCallType::DrawIndexedIndirect
                | DrawCallType::MultiDrawIndexedIndirect
        )
    }

    /// Returns true if this is an indirect draw type.
    #[inline]
    pub fn is_indirect(&self) -> bool {
        matches!(
            self,
            DrawCallType::DrawIndirect
                | DrawCallType::DrawIndexedIndirect
                | DrawCallType::MultiDrawIndirect
                | DrawCallType::MultiDrawIndexedIndirect
        )
    }

    /// Returns all draw call type variants.
    pub fn all_variants() -> &'static [DrawCallType] {
        &[
            DrawCallType::Draw,
            DrawCallType::DrawIndexed,
            DrawCallType::DrawIndirect,
            DrawCallType::DrawIndexedIndirect,
            DrawCallType::MultiDrawIndirect,
            DrawCallType::MultiDrawIndexedIndirect,
        ]
    }
}

impl std::fmt::Display for DrawCallType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

impl Default for DrawCallType {
    fn default() -> Self {
        DrawCallType::Draw
    }
}

// ============================================================================
// DrawCallInfo
// ============================================================================

/// Statistics for a single draw call.
#[derive(Clone, Debug, Default)]
pub struct DrawCallInfo {
    /// Type of draw call.
    pub call_type: DrawCallType,
    /// Number of vertices per instance.
    pub vertex_count: u32,
    /// Number of instances to draw.
    pub instance_count: u32,
    /// Number of indices (for indexed draws).
    pub index_count: Option<u32>,
    /// Pipeline ID (for tracking pipeline usage).
    pub pipeline_id: Option<u64>,
}

impl DrawCallInfo {
    /// Create a new draw call info.
    #[inline]
    pub fn new(call_type: DrawCallType, vertex_count: u32, instance_count: u32) -> Self {
        Self {
            call_type,
            vertex_count,
            instance_count,
            index_count: None,
            pipeline_id: None,
        }
    }

    /// Create an indexed draw call info.
    #[inline]
    pub fn indexed(index_count: u32, instance_count: u32) -> Self {
        Self {
            call_type: DrawCallType::DrawIndexed,
            vertex_count: 0,
            instance_count,
            index_count: Some(index_count),
            pipeline_id: None,
        }
    }

    /// Create an indirect draw call info.
    #[inline]
    pub fn indirect(call_type: DrawCallType, count: u32) -> Self {
        Self {
            call_type,
            vertex_count: 0,
            instance_count: count,
            index_count: None,
            pipeline_id: None,
        }
    }

    /// Set the pipeline ID.
    #[inline]
    pub fn with_pipeline(mut self, pipeline_id: u64) -> Self {
        self.pipeline_id = Some(pipeline_id);
        self
    }

    /// Calculate total vertices processed.
    #[inline]
    pub fn total_vertices(&self) -> u64 {
        self.vertex_count as u64 * self.instance_count as u64
    }

    /// Calculate total indices processed.
    #[inline]
    pub fn total_indices(&self) -> u64 {
        self.index_count.unwrap_or(0) as u64 * self.instance_count as u64
    }

    /// Calculate total instances.
    #[inline]
    pub fn total_instances(&self) -> u64 {
        self.instance_count as u64
    }
}

// ============================================================================
// FrameDrawStats
// ============================================================================

/// Frame-level draw statistics.
#[derive(Clone, Debug, Default)]
pub struct FrameDrawStats {
    /// Total number of draw calls.
    pub total_draw_calls: u32,
    /// Total vertices processed.
    pub total_vertices: u64,
    /// Total indices processed.
    pub total_indices: u64,
    /// Total instances drawn.
    pub total_instances: u64,
    /// Draw calls by type.
    pub by_type: HashMap<DrawCallType, u32>,
    /// Draw calls by pipeline.
    pub by_pipeline: HashMap<u64, u32>,
}

impl FrameDrawStats {
    /// Create empty frame statistics.
    #[inline]
    pub fn empty() -> Self {
        Self::default()
    }

    /// Create frame statistics from a collection of draw calls.
    pub fn from_draw_calls(draws: &[DrawCallInfo]) -> Self {
        let mut stats = Self::empty();
        for draw in draws {
            stats.add_draw(draw);
        }
        stats
    }

    /// Add a draw call to the statistics.
    #[inline]
    pub fn add_draw(&mut self, draw: &DrawCallInfo) {
        self.total_draw_calls += 1;
        self.total_vertices += draw.total_vertices();
        self.total_indices += draw.total_indices();
        self.total_instances += draw.total_instances();

        *self.by_type.entry(draw.call_type).or_insert(0) += 1;

        if let Some(pipeline_id) = draw.pipeline_id {
            *self.by_pipeline.entry(pipeline_id).or_insert(0) += 1;
        }
    }

    /// Calculate draw call rate given frame time.
    ///
    /// # Arguments
    /// * `frame_time_ms` - Frame time in milliseconds
    ///
    /// # Returns
    /// Draw calls per millisecond
    #[inline]
    pub fn draw_call_rate(&self, frame_time_ms: f64) -> f64 {
        if frame_time_ms <= 0.0 {
            0.0
        } else {
            self.total_draw_calls as f64 / frame_time_ms
        }
    }

    /// Calculate vertex processing rate given frame time.
    ///
    /// # Arguments
    /// * `frame_time_ms` - Frame time in milliseconds
    ///
    /// # Returns
    /// Vertices per millisecond
    #[inline]
    pub fn vertex_rate(&self, frame_time_ms: f64) -> f64 {
        if frame_time_ms <= 0.0 {
            0.0
        } else {
            self.total_vertices as f64 / frame_time_ms
        }
    }

    /// Calculate index processing rate given frame time.
    ///
    /// # Arguments
    /// * `frame_time_ms` - Frame time in milliseconds
    ///
    /// # Returns
    /// Indices per millisecond
    #[inline]
    pub fn index_rate(&self, frame_time_ms: f64) -> f64 {
        if frame_time_ms <= 0.0 {
            0.0
        } else {
            self.total_indices as f64 / frame_time_ms
        }
    }

    /// Get the count of a specific draw type.
    #[inline]
    pub fn type_count(&self, draw_type: DrawCallType) -> u32 {
        self.by_type.get(&draw_type).copied().unwrap_or(0)
    }

    /// Get the count for a specific pipeline.
    #[inline]
    pub fn pipeline_count(&self, pipeline_id: u64) -> u32 {
        self.by_pipeline.get(&pipeline_id).copied().unwrap_or(0)
    }

    /// Get the number of unique pipelines used.
    #[inline]
    pub fn unique_pipelines(&self) -> usize {
        self.by_pipeline.len()
    }

    /// Get the most used draw type.
    pub fn most_common_type(&self) -> Option<DrawCallType> {
        self.by_type
            .iter()
            .max_by_key(|(_, &count)| count)
            .map(|(&draw_type, _)| draw_type)
    }

    /// Get the most used pipeline.
    pub fn most_common_pipeline(&self) -> Option<u64> {
        self.by_pipeline
            .iter()
            .max_by_key(|(_, &count)| count)
            .map(|(&pipeline_id, _)| pipeline_id)
    }

    /// Merge another frame's statistics into this one.
    pub fn merge(&mut self, other: &FrameDrawStats) {
        self.total_draw_calls += other.total_draw_calls;
        self.total_vertices += other.total_vertices;
        self.total_indices += other.total_indices;
        self.total_instances += other.total_instances;

        for (&draw_type, &count) in &other.by_type {
            *self.by_type.entry(draw_type).or_insert(0) += count;
        }

        for (&pipeline_id, &count) in &other.by_pipeline {
            *self.by_pipeline.entry(pipeline_id).or_insert(0) += count;
        }
    }

    /// Calculate average vertices per draw call.
    #[inline]
    pub fn avg_vertices_per_draw(&self) -> f64 {
        if self.total_draw_calls == 0 {
            0.0
        } else {
            self.total_vertices as f64 / self.total_draw_calls as f64
        }
    }

    /// Calculate average instances per draw call.
    #[inline]
    pub fn avg_instances_per_draw(&self) -> f64 {
        if self.total_draw_calls == 0 {
            0.0
        } else {
            self.total_instances as f64 / self.total_draw_calls as f64
        }
    }

    /// Check if frame is empty (no draw calls).
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.total_draw_calls == 0
    }
}

impl std::fmt::Display for FrameDrawStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "DrawStats: {} calls, {} verts, {} indices, {} instances",
            self.total_draw_calls, self.total_vertices, self.total_indices, self.total_instances
        )
    }
}

// ============================================================================
// DrawStatsSummary
// ============================================================================

/// Summary statistics across multiple frames.
#[derive(Clone, Debug, Default)]
pub struct DrawStatsSummary {
    /// Average draw calls per frame.
    pub avg_draw_calls: f64,
    /// Average vertices per frame.
    pub avg_vertices: f64,
    /// Maximum draw calls in any frame.
    pub max_draw_calls: u32,
    /// Maximum vertices in any frame.
    pub max_vertices: u64,
    /// Most common draw type across all frames.
    pub most_common_type: DrawCallType,
    /// Number of frames analyzed.
    pub frame_count: usize,
    /// Total draw calls across all frames.
    pub total_draw_calls: u64,
    /// Total vertices across all frames.
    pub total_vertices: u64,
    /// Average indices per frame.
    pub avg_indices: f64,
    /// Maximum indices in any frame.
    pub max_indices: u64,
    /// Average instances per frame.
    pub avg_instances: f64,
    /// Maximum instances in any frame.
    pub max_instances: u64,
}

impl DrawStatsSummary {
    /// Create an empty summary.
    pub fn empty() -> Self {
        Self::default()
    }

    /// Check if the summary has data.
    #[inline]
    pub fn has_data(&self) -> bool {
        self.frame_count > 0
    }

    /// Get the variance of draw calls (requires frame history).
    pub fn draw_call_variance(&self, history: &[FrameDrawStats]) -> f64 {
        if history.is_empty() {
            return 0.0;
        }

        let mean = self.avg_draw_calls;
        let variance: f64 = history
            .iter()
            .map(|f| {
                let diff = f.total_draw_calls as f64 - mean;
                diff * diff
            })
            .sum::<f64>()
            / history.len() as f64;

        variance
    }

    /// Get the standard deviation of draw calls.
    pub fn draw_call_std_dev(&self, history: &[FrameDrawStats]) -> f64 {
        self.draw_call_variance(history).sqrt()
    }
}

impl std::fmt::Display for DrawStatsSummary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Summary ({} frames): avg {:.1} calls, {:.0} verts, max {} calls, {} verts, most common: {}",
            self.frame_count,
            self.avg_draw_calls,
            self.avg_vertices,
            self.max_draw_calls,
            self.max_vertices,
            self.most_common_type
        )
    }
}

// ============================================================================
// DrawStatsCollector
// ============================================================================

/// Tracks draw call statistics across frames.
#[derive(Debug)]
pub struct DrawStatsCollector {
    /// Current frame's draw calls.
    current_frame: Vec<DrawCallInfo>,
    /// Frame history.
    frame_history: Vec<FrameDrawStats>,
    /// Maximum history size.
    history_size: usize,
    /// Whether collection is enabled.
    enabled: bool,
    /// Current pipeline ID (for implicit tracking).
    current_pipeline: Option<u64>,
}

impl DrawStatsCollector {
    /// Create a new draw statistics collector with default history size.
    pub fn new() -> Self {
        Self {
            current_frame: Vec::new(),
            frame_history: Vec::with_capacity(DEFAULT_HISTORY_SIZE),
            history_size: DEFAULT_HISTORY_SIZE,
            enabled: true,
            current_pipeline: None,
        }
    }

    /// Create a new collector with specified history size.
    pub fn with_history_size(size: usize) -> Self {
        let size = size.clamp(MIN_HISTORY_SIZE, MAX_HISTORY_SIZE);
        Self {
            current_frame: Vec::new(),
            frame_history: Vec::with_capacity(size),
            history_size: size,
            enabled: true,
            current_pipeline: None,
        }
    }

    /// Record a non-indexed draw call.
    #[inline]
    pub fn record_draw(&mut self, vertex_count: u32, instance_count: u32) {
        if !self.enabled {
            return;
        }

        let mut info = DrawCallInfo::new(DrawCallType::Draw, vertex_count, instance_count);
        if let Some(pipeline_id) = self.current_pipeline {
            info.pipeline_id = Some(pipeline_id);
        }
        self.current_frame.push(info);
    }

    /// Record an indexed draw call.
    #[inline]
    pub fn record_draw_indexed(&mut self, index_count: u32, instance_count: u32) {
        if !self.enabled {
            return;
        }

        let mut info = DrawCallInfo::indexed(index_count, instance_count);
        if let Some(pipeline_id) = self.current_pipeline {
            info.pipeline_id = Some(pipeline_id);
        }
        self.current_frame.push(info);
    }

    /// Record an indirect draw call.
    #[inline]
    pub fn record_draw_indirect(&mut self, call_type: DrawCallType, count: u32) {
        if !self.enabled {
            return;
        }

        let mut info = DrawCallInfo::indirect(call_type, count);
        if let Some(pipeline_id) = self.current_pipeline {
            info.pipeline_id = Some(pipeline_id);
        }
        self.current_frame.push(info);
    }

    /// Record a generic draw call with full info.
    #[inline]
    pub fn record(&mut self, info: DrawCallInfo) {
        if !self.enabled {
            return;
        }

        self.current_frame.push(info);
    }

    /// Set the current pipeline for subsequent draws.
    #[inline]
    pub fn set_pipeline(&mut self, pipeline_id: u64) {
        self.current_pipeline = Some(pipeline_id);
    }

    /// Clear the current pipeline.
    #[inline]
    pub fn clear_pipeline(&mut self) {
        self.current_pipeline = None;
    }

    /// End the current frame and return statistics.
    pub fn end_frame(&mut self) -> FrameDrawStats {
        if !self.enabled {
            return FrameDrawStats::empty();
        }

        let stats = FrameDrawStats::from_draw_calls(&self.current_frame);

        // Add to history
        if self.frame_history.len() >= self.history_size {
            self.frame_history.remove(0);
        }
        self.frame_history.push(stats.clone());

        // Clear for next frame
        self.current_frame.clear();
        self.current_pipeline = None;

        stats
    }

    /// Get statistics for the current frame (in progress).
    pub fn current_frame_stats(&self) -> FrameDrawStats {
        FrameDrawStats::from_draw_calls(&self.current_frame)
    }

    /// Get the frame history.
    #[inline]
    pub fn history(&self) -> &[FrameDrawStats] {
        &self.frame_history
    }

    /// Get summary statistics across all history.
    pub fn summary(&self) -> DrawStatsSummary {
        if self.frame_history.is_empty() {
            return DrawStatsSummary::empty();
        }

        let frame_count = self.frame_history.len();
        let mut total_draw_calls: u64 = 0;
        let mut total_vertices: u64 = 0;
        let mut total_indices: u64 = 0;
        let mut total_instances: u64 = 0;
        let mut max_draw_calls: u32 = 0;
        let mut max_vertices: u64 = 0;
        let mut max_indices: u64 = 0;
        let mut max_instances: u64 = 0;
        let mut type_counts: HashMap<DrawCallType, u32> = HashMap::new();

        for frame in &self.frame_history {
            total_draw_calls += frame.total_draw_calls as u64;
            total_vertices += frame.total_vertices;
            total_indices += frame.total_indices;
            total_instances += frame.total_instances;

            max_draw_calls = max_draw_calls.max(frame.total_draw_calls);
            max_vertices = max_vertices.max(frame.total_vertices);
            max_indices = max_indices.max(frame.total_indices);
            max_instances = max_instances.max(frame.total_instances);

            for (&draw_type, &count) in &frame.by_type {
                *type_counts.entry(draw_type).or_insert(0) += count;
            }
        }

        let most_common_type = type_counts
            .iter()
            .max_by_key(|(_, &count)| count)
            .map(|(&t, _)| t)
            .unwrap_or(DrawCallType::Draw);

        DrawStatsSummary {
            avg_draw_calls: total_draw_calls as f64 / frame_count as f64,
            avg_vertices: total_vertices as f64 / frame_count as f64,
            max_draw_calls,
            max_vertices,
            most_common_type,
            frame_count,
            total_draw_calls,
            total_vertices,
            avg_indices: total_indices as f64 / frame_count as f64,
            max_indices,
            avg_instances: total_instances as f64 / frame_count as f64,
            max_instances,
        }
    }

    /// Enable statistics collection.
    #[inline]
    pub fn enable(&mut self) {
        self.enabled = true;
    }

    /// Disable statistics collection.
    #[inline]
    pub fn disable(&mut self) {
        self.enabled = false;
    }

    /// Check if collection is enabled.
    #[inline]
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Reset the collector, clearing all history.
    pub fn reset(&mut self) {
        self.current_frame.clear();
        self.frame_history.clear();
        self.current_pipeline = None;
    }

    /// Get the number of draw calls recorded in the current frame.
    #[inline]
    pub fn current_frame_draw_count(&self) -> usize {
        self.current_frame.len()
    }

    /// Get the number of frames in history.
    #[inline]
    pub fn history_size(&self) -> usize {
        self.frame_history.len()
    }

    /// Get the maximum history size.
    #[inline]
    pub fn max_history_size(&self) -> usize {
        self.history_size
    }

    /// Get the last frame's statistics.
    pub fn last_frame_stats(&self) -> Option<&FrameDrawStats> {
        self.frame_history.last()
    }

    /// Calculate average draw calls per frame from history.
    pub fn avg_draw_calls_per_frame(&self) -> f64 {
        if self.frame_history.is_empty() {
            return 0.0;
        }

        let total: u64 = self
            .frame_history
            .iter()
            .map(|f| f.total_draw_calls as u64)
            .sum();
        total as f64 / self.frame_history.len() as f64
    }

    /// Calculate average vertices per frame from history.
    pub fn avg_vertices_per_frame(&self) -> f64 {
        if self.frame_history.is_empty() {
            return 0.0;
        }

        let total: u64 = self.frame_history.iter().map(|f| f.total_vertices).sum();
        total as f64 / self.frame_history.len() as f64
    }

    /// Get frame statistics for a specific frame in history (0 = oldest).
    pub fn frame_at(&self, index: usize) -> Option<&FrameDrawStats> {
        self.frame_history.get(index)
    }
}

impl Default for DrawStatsCollector {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================
    // DrawCallType tests
    // ========================================

    #[test]
    fn test_draw_call_type_name() {
        assert_eq!(DrawCallType::Draw.name(), "Draw");
        assert_eq!(DrawCallType::DrawIndexed.name(), "DrawIndexed");
        assert_eq!(DrawCallType::DrawIndirect.name(), "DrawIndirect");
        assert_eq!(DrawCallType::DrawIndexedIndirect.name(), "DrawIndexedIndirect");
        assert_eq!(DrawCallType::MultiDrawIndirect.name(), "MultiDrawIndirect");
        assert_eq!(
            DrawCallType::MultiDrawIndexedIndirect.name(),
            "MultiDrawIndexedIndirect"
        );
    }

    #[test]
    fn test_draw_call_type_is_indexed() {
        assert!(!DrawCallType::Draw.is_indexed());
        assert!(DrawCallType::DrawIndexed.is_indexed());
        assert!(!DrawCallType::DrawIndirect.is_indexed());
        assert!(DrawCallType::DrawIndexedIndirect.is_indexed());
        assert!(!DrawCallType::MultiDrawIndirect.is_indexed());
        assert!(DrawCallType::MultiDrawIndexedIndirect.is_indexed());
    }

    #[test]
    fn test_draw_call_type_is_indirect() {
        assert!(!DrawCallType::Draw.is_indirect());
        assert!(!DrawCallType::DrawIndexed.is_indirect());
        assert!(DrawCallType::DrawIndirect.is_indirect());
        assert!(DrawCallType::DrawIndexedIndirect.is_indirect());
        assert!(DrawCallType::MultiDrawIndirect.is_indirect());
        assert!(DrawCallType::MultiDrawIndexedIndirect.is_indirect());
    }

    #[test]
    fn test_draw_call_type_display() {
        assert_eq!(DrawCallType::Draw.to_string(), "Draw");
        assert_eq!(DrawCallType::DrawIndexed.to_string(), "DrawIndexed");
    }

    #[test]
    fn test_draw_call_type_default() {
        assert_eq!(DrawCallType::default(), DrawCallType::Draw);
    }

    #[test]
    fn test_draw_call_type_all_variants() {
        let variants = DrawCallType::all_variants();
        assert_eq!(variants.len(), 6);
        assert!(variants.contains(&DrawCallType::Draw));
        assert!(variants.contains(&DrawCallType::DrawIndexed));
        assert!(variants.contains(&DrawCallType::MultiDrawIndexedIndirect));
    }

    #[test]
    fn test_draw_call_type_equality() {
        assert_eq!(DrawCallType::Draw, DrawCallType::Draw);
        assert_ne!(DrawCallType::Draw, DrawCallType::DrawIndexed);
    }

    #[test]
    fn test_draw_call_type_hash() {
        let mut map = HashMap::new();
        map.insert(DrawCallType::Draw, 1);
        map.insert(DrawCallType::DrawIndexed, 2);
        assert_eq!(map.get(&DrawCallType::Draw), Some(&1));
        assert_eq!(map.get(&DrawCallType::DrawIndexed), Some(&2));
    }

    // ========================================
    // DrawCallInfo tests
    // ========================================

    #[test]
    fn test_draw_call_info_new() {
        let info = DrawCallInfo::new(DrawCallType::Draw, 100, 5);
        assert_eq!(info.call_type, DrawCallType::Draw);
        assert_eq!(info.vertex_count, 100);
        assert_eq!(info.instance_count, 5);
        assert!(info.index_count.is_none());
        assert!(info.pipeline_id.is_none());
    }

    #[test]
    fn test_draw_call_info_indexed() {
        let info = DrawCallInfo::indexed(300, 10);
        assert_eq!(info.call_type, DrawCallType::DrawIndexed);
        assert_eq!(info.index_count, Some(300));
        assert_eq!(info.instance_count, 10);
    }

    #[test]
    fn test_draw_call_info_indirect() {
        let info = DrawCallInfo::indirect(DrawCallType::DrawIndirect, 5);
        assert_eq!(info.call_type, DrawCallType::DrawIndirect);
        assert_eq!(info.instance_count, 5);
    }

    #[test]
    fn test_draw_call_info_with_pipeline() {
        let info = DrawCallInfo::new(DrawCallType::Draw, 100, 1).with_pipeline(42);
        assert_eq!(info.pipeline_id, Some(42));
    }

    #[test]
    fn test_draw_call_info_total_vertices() {
        let info = DrawCallInfo::new(DrawCallType::Draw, 100, 10);
        assert_eq!(info.total_vertices(), 1000);
    }

    #[test]
    fn test_draw_call_info_total_indices() {
        let info = DrawCallInfo::indexed(300, 5);
        assert_eq!(info.total_indices(), 1500);

        let no_indices = DrawCallInfo::new(DrawCallType::Draw, 100, 1);
        assert_eq!(no_indices.total_indices(), 0);
    }

    #[test]
    fn test_draw_call_info_total_instances() {
        let info = DrawCallInfo::new(DrawCallType::Draw, 100, 7);
        assert_eq!(info.total_instances(), 7);
    }

    #[test]
    fn test_draw_call_info_default() {
        let info = DrawCallInfo::default();
        assert_eq!(info.call_type, DrawCallType::Draw);
        assert_eq!(info.vertex_count, 0);
        assert_eq!(info.instance_count, 0);
    }

    // ========================================
    // FrameDrawStats tests
    // ========================================

    #[test]
    fn test_frame_draw_stats_empty() {
        let stats = FrameDrawStats::empty();
        assert_eq!(stats.total_draw_calls, 0);
        assert_eq!(stats.total_vertices, 0);
        assert!(stats.is_empty());
    }

    #[test]
    fn test_frame_draw_stats_from_draw_calls() {
        let draws = vec![
            DrawCallInfo::new(DrawCallType::Draw, 100, 1),
            DrawCallInfo::new(DrawCallType::Draw, 200, 2),
            DrawCallInfo::indexed(300, 5),
        ];
        let stats = FrameDrawStats::from_draw_calls(&draws);
        assert_eq!(stats.total_draw_calls, 3);
        assert_eq!(stats.total_vertices, 500); // 100 + 400
        assert_eq!(stats.total_indices, 1500); // 300 * 5
        assert_eq!(stats.total_instances, 8); // 1 + 2 + 5
    }

    #[test]
    fn test_frame_draw_stats_add_draw() {
        let mut stats = FrameDrawStats::empty();
        stats.add_draw(&DrawCallInfo::new(DrawCallType::Draw, 100, 2));
        stats.add_draw(&DrawCallInfo::indexed(50, 3));

        assert_eq!(stats.total_draw_calls, 2);
        assert_eq!(stats.total_vertices, 200); // 100 * 2
        assert_eq!(stats.total_indices, 150); // 50 * 3
    }

    #[test]
    fn test_frame_draw_stats_draw_call_rate() {
        let mut stats = FrameDrawStats::empty();
        for _ in 0..100 {
            stats.add_draw(&DrawCallInfo::new(DrawCallType::Draw, 10, 1));
        }
        assert_eq!(stats.draw_call_rate(10.0), 10.0); // 100 calls / 10ms
        assert_eq!(stats.draw_call_rate(0.0), 0.0);
    }

    #[test]
    fn test_frame_draw_stats_vertex_rate() {
        let mut stats = FrameDrawStats::empty();
        stats.add_draw(&DrawCallInfo::new(DrawCallType::Draw, 1000, 1));
        assert_eq!(stats.vertex_rate(2.0), 500.0); // 1000 verts / 2ms
        assert_eq!(stats.vertex_rate(0.0), 0.0);
    }

    #[test]
    fn test_frame_draw_stats_by_type() {
        let mut stats = FrameDrawStats::empty();
        stats.add_draw(&DrawCallInfo::new(DrawCallType::Draw, 100, 1));
        stats.add_draw(&DrawCallInfo::new(DrawCallType::Draw, 100, 1));
        stats.add_draw(&DrawCallInfo::indexed(50, 1));

        assert_eq!(stats.type_count(DrawCallType::Draw), 2);
        assert_eq!(stats.type_count(DrawCallType::DrawIndexed), 1);
        assert_eq!(stats.type_count(DrawCallType::DrawIndirect), 0);
    }

    #[test]
    fn test_frame_draw_stats_by_pipeline() {
        let mut stats = FrameDrawStats::empty();
        stats.add_draw(&DrawCallInfo::new(DrawCallType::Draw, 100, 1).with_pipeline(1));
        stats.add_draw(&DrawCallInfo::new(DrawCallType::Draw, 100, 1).with_pipeline(1));
        stats.add_draw(&DrawCallInfo::new(DrawCallType::Draw, 100, 1).with_pipeline(2));

        assert_eq!(stats.pipeline_count(1), 2);
        assert_eq!(stats.pipeline_count(2), 1);
        assert_eq!(stats.unique_pipelines(), 2);
    }

    #[test]
    fn test_frame_draw_stats_most_common_type() {
        let mut stats = FrameDrawStats::empty();
        stats.add_draw(&DrawCallInfo::new(DrawCallType::Draw, 100, 1));
        stats.add_draw(&DrawCallInfo::new(DrawCallType::Draw, 100, 1));
        stats.add_draw(&DrawCallInfo::indexed(50, 1));

        assert_eq!(stats.most_common_type(), Some(DrawCallType::Draw));
    }

    #[test]
    fn test_frame_draw_stats_merge() {
        let mut stats1 = FrameDrawStats::empty();
        stats1.add_draw(&DrawCallInfo::new(DrawCallType::Draw, 100, 1));

        let mut stats2 = FrameDrawStats::empty();
        stats2.add_draw(&DrawCallInfo::new(DrawCallType::Draw, 200, 2));

        stats1.merge(&stats2);
        assert_eq!(stats1.total_draw_calls, 2);
        assert_eq!(stats1.total_vertices, 500);
    }

    #[test]
    fn test_frame_draw_stats_avg_vertices_per_draw() {
        let draws = vec![
            DrawCallInfo::new(DrawCallType::Draw, 100, 1),
            DrawCallInfo::new(DrawCallType::Draw, 300, 1),
        ];
        let stats = FrameDrawStats::from_draw_calls(&draws);
        assert_eq!(stats.avg_vertices_per_draw(), 200.0);
    }

    #[test]
    fn test_frame_draw_stats_display() {
        let mut stats = FrameDrawStats::empty();
        stats.add_draw(&DrawCallInfo::new(DrawCallType::Draw, 100, 2));
        let display = stats.to_string();
        assert!(display.contains("1 calls"));
        assert!(display.contains("200 verts"));
    }

    // ========================================
    // DrawStatsSummary tests
    // ========================================

    #[test]
    fn test_draw_stats_summary_empty() {
        let summary = DrawStatsSummary::empty();
        assert!(!summary.has_data());
        assert_eq!(summary.frame_count, 0);
    }

    #[test]
    fn test_draw_stats_summary_display() {
        let mut collector = DrawStatsCollector::new();
        collector.record_draw(100, 1);
        collector.end_frame();
        let summary = collector.summary();
        let display = summary.to_string();
        assert!(display.contains("1 frames"));
    }

    // ========================================
    // DrawStatsCollector tests
    // ========================================

    #[test]
    fn test_draw_stats_collector_new() {
        let collector = DrawStatsCollector::new();
        assert!(collector.is_enabled());
        assert_eq!(collector.history_size(), 0);
        assert_eq!(collector.max_history_size(), DEFAULT_HISTORY_SIZE);
    }

    #[test]
    fn test_draw_stats_collector_with_history_size() {
        let collector = DrawStatsCollector::with_history_size(10);
        assert_eq!(collector.max_history_size(), 10);
    }

    #[test]
    fn test_draw_stats_collector_with_history_size_clamped() {
        let collector = DrawStatsCollector::with_history_size(10000);
        assert_eq!(collector.max_history_size(), MAX_HISTORY_SIZE);

        let collector = DrawStatsCollector::with_history_size(0);
        assert_eq!(collector.max_history_size(), MIN_HISTORY_SIZE);
    }

    #[test]
    fn test_draw_stats_collector_record_draw() {
        let mut collector = DrawStatsCollector::new();
        collector.record_draw(100, 5);
        let stats = collector.current_frame_stats();
        assert_eq!(stats.total_draw_calls, 1);
        assert_eq!(stats.total_vertices, 500);
    }

    #[test]
    fn test_draw_stats_collector_record_draw_indexed() {
        let mut collector = DrawStatsCollector::new();
        collector.record_draw_indexed(300, 10);
        let stats = collector.current_frame_stats();
        assert_eq!(stats.total_draw_calls, 1);
        assert_eq!(stats.total_indices, 3000);
    }

    #[test]
    fn test_draw_stats_collector_record_draw_indirect() {
        let mut collector = DrawStatsCollector::new();
        collector.record_draw_indirect(DrawCallType::MultiDrawIndirect, 5);
        let stats = collector.current_frame_stats();
        assert_eq!(stats.total_draw_calls, 1);
        assert_eq!(stats.type_count(DrawCallType::MultiDrawIndirect), 1);
    }

    #[test]
    fn test_draw_stats_collector_end_frame() {
        let mut collector = DrawStatsCollector::new();
        collector.record_draw(100, 1);
        collector.record_draw(200, 2);
        let stats = collector.end_frame();
        assert_eq!(stats.total_draw_calls, 2);
        assert_eq!(stats.total_vertices, 500);
        assert_eq!(collector.history_size(), 1);
        assert_eq!(collector.current_frame_draw_count(), 0);
    }

    #[test]
    fn test_draw_stats_collector_frame_history() {
        let mut collector = DrawStatsCollector::with_history_size(5);
        for i in 0..10 {
            collector.record_draw((i + 1) * 10, 1);
            collector.end_frame();
        }
        assert_eq!(collector.history_size(), 5);
    }

    #[test]
    fn test_draw_stats_collector_enable_disable() {
        let mut collector = DrawStatsCollector::new();
        collector.disable();
        assert!(!collector.is_enabled());

        collector.record_draw(100, 1);
        let stats = collector.end_frame();
        assert_eq!(stats.total_draw_calls, 0);

        collector.enable();
        assert!(collector.is_enabled());
    }

    #[test]
    fn test_draw_stats_collector_reset() {
        let mut collector = DrawStatsCollector::new();
        collector.record_draw(100, 1);
        collector.end_frame();
        collector.record_draw(200, 1);
        collector.reset();

        assert_eq!(collector.history_size(), 0);
        assert_eq!(collector.current_frame_draw_count(), 0);
    }

    #[test]
    fn test_draw_stats_collector_summary() {
        let mut collector = DrawStatsCollector::new();
        for _ in 0..5 {
            collector.record_draw(100, 1);
            collector.record_draw(200, 2);
            collector.end_frame();
        }
        let summary = collector.summary();
        assert_eq!(summary.frame_count, 5);
        assert_eq!(summary.avg_draw_calls, 2.0);
        assert_eq!(summary.avg_vertices, 500.0);
        assert_eq!(summary.max_draw_calls, 2);
        assert_eq!(summary.max_vertices, 500);
    }

    #[test]
    fn test_draw_stats_collector_avg_per_frame() {
        let mut collector = DrawStatsCollector::new();
        collector.record_draw(100, 1);
        collector.end_frame();
        collector.record_draw(300, 1);
        collector.end_frame();

        assert_eq!(collector.avg_draw_calls_per_frame(), 1.0);
        assert_eq!(collector.avg_vertices_per_frame(), 200.0);
    }

    #[test]
    fn test_draw_stats_collector_pipeline_tracking() {
        let mut collector = DrawStatsCollector::new();
        collector.set_pipeline(42);
        collector.record_draw(100, 1);
        collector.record_draw(100, 1);
        collector.clear_pipeline();
        collector.record_draw(100, 1);
        let stats = collector.end_frame();

        assert_eq!(stats.pipeline_count(42), 2);
        assert_eq!(stats.unique_pipelines(), 1);
    }

    #[test]
    fn test_draw_stats_collector_last_frame_stats() {
        let mut collector = DrawStatsCollector::new();
        collector.record_draw(100, 1);
        collector.end_frame();
        collector.record_draw(200, 2);
        collector.end_frame();

        let last = collector.last_frame_stats().unwrap();
        assert_eq!(last.total_draw_calls, 1);
        assert_eq!(last.total_vertices, 400);
    }

    #[test]
    fn test_draw_stats_collector_frame_at() {
        let mut collector = DrawStatsCollector::new();
        collector.record_draw(100, 1);
        collector.end_frame();
        collector.record_draw(200, 1);
        collector.end_frame();

        let first = collector.frame_at(0).unwrap();
        assert_eq!(first.total_vertices, 100);

        let second = collector.frame_at(1).unwrap();
        assert_eq!(second.total_vertices, 200);

        assert!(collector.frame_at(10).is_none());
    }

    #[test]
    fn test_draw_stats_collector_default() {
        let collector = DrawStatsCollector::default();
        assert!(collector.is_enabled());
    }

    #[test]
    fn test_draw_stats_collector_record_generic() {
        let mut collector = DrawStatsCollector::new();
        let info = DrawCallInfo::new(DrawCallType::DrawIndirect, 0, 100).with_pipeline(99);
        collector.record(info);
        let stats = collector.current_frame_stats();
        assert_eq!(stats.total_draw_calls, 1);
        assert_eq!(stats.pipeline_count(99), 1);
    }

    #[test]
    fn test_draw_stats_collector_empty_summary() {
        let collector = DrawStatsCollector::new();
        let summary = collector.summary();
        assert!(!summary.has_data());
    }

    #[test]
    fn test_frame_draw_stats_index_rate() {
        let mut stats = FrameDrawStats::empty();
        stats.add_draw(&DrawCallInfo::indexed(1000, 1));
        assert_eq!(stats.index_rate(2.0), 500.0);
        assert_eq!(stats.index_rate(0.0), 0.0);
    }

    #[test]
    fn test_draw_stats_summary_variance() {
        let mut collector = DrawStatsCollector::new();
        collector.record_draw(100, 1);
        collector.end_frame();
        collector.record_draw(100, 1);
        collector.record_draw(100, 1);
        collector.record_draw(100, 1);
        collector.end_frame();

        let summary = collector.summary();
        let variance = summary.draw_call_variance(collector.history());
        // frames: [1, 3], mean = 2, variance = ((1-2)^2 + (3-2)^2)/2 = 1
        assert!((variance - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_draw_stats_collector_history_bounds() {
        let mut collector = DrawStatsCollector::with_history_size(3);
        for i in 0..10 {
            collector.record_draw((i + 1) * 10, 1);
            collector.end_frame();
        }
        assert_eq!(collector.history_size(), 3);

        // Oldest frames should be removed (frames 0-6 gone, frames 7, 8, 9 remain)
        let first = collector.frame_at(0).unwrap();
        assert_eq!(first.total_vertices, 80); // Frame 7 had 80 verts
    }

    #[test]
    fn test_frame_draw_stats_avg_instances_per_draw() {
        let draws = vec![
            DrawCallInfo::new(DrawCallType::Draw, 100, 2),
            DrawCallInfo::new(DrawCallType::Draw, 100, 4),
        ];
        let stats = FrameDrawStats::from_draw_calls(&draws);
        assert_eq!(stats.avg_instances_per_draw(), 3.0);
    }

    #[test]
    fn test_frame_draw_stats_empty_averages() {
        let stats = FrameDrawStats::empty();
        assert_eq!(stats.avg_vertices_per_draw(), 0.0);
        assert_eq!(stats.avg_instances_per_draw(), 0.0);
    }

    #[test]
    fn test_draw_stats_summary_std_dev() {
        let mut collector = DrawStatsCollector::new();
        collector.record_draw(100, 1);
        collector.end_frame();
        collector.record_draw(100, 1);
        collector.record_draw(100, 1);
        collector.record_draw(100, 1);
        collector.end_frame();

        let summary = collector.summary();
        let std_dev = summary.draw_call_std_dev(collector.history());
        assert!((std_dev - 1.0).abs() < 0.001);
    }
}
