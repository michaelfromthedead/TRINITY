//! Pipeline statistics query pool for GPU performance analysis.
//!
//! This module provides a reusable pool for wgpu pipeline statistics queries, enabling
//! detailed GPU execution statistics including shader invocation counts, primitive counts,
//! and culling efficiency metrics.
//!
//! # Overview
//!
//! Pipeline statistics queries provide insight into GPU execution:
//!
//! - **Feature detection**: Checks if `PIPELINE_STATISTICS_QUERY` is supported
//! - **QuerySet creation**: Creates a `QuerySet` with `QueryType::PipelineStatistics`
//! - **5 statistic types**: Vertex, fragment, compute shader invocations, clipper primitives/invocations
//! - **Derived metrics**: Overdraw estimation and culling efficiency
//!
//! # Usage
//!
//! ```no_run
//! use renderer_backend::pipeline_statistics::{PipelineStatisticsPool, PipelineStatisticsResult};
//!
//! # fn example(device: &wgpu::Device, queue: &wgpu::Queue) {
//! // Check if pipeline statistics are supported
//! if !PipelineStatisticsPool::is_supported(device) {
//!     println!("Pipeline statistics not supported on this device");
//!     return;
//! }
//!
//! // Create a pool with capacity for 64 queries
//! let mut pool = PipelineStatisticsPool::new(device, 64)
//!     .expect("Failed to create pipeline statistics pool");
//!
//! // Allocate an index for a query
//! let query_idx = pool.allocate().expect("Pool exhausted");
//!
//! // In render pass:
//! // render_pass.begin_pipeline_statistics_query(pool.query_set(), query_idx);
//! // ... draw calls ...
//! // render_pass.end_pipeline_statistics_query();
//!
//! // Reset pool for next frame
//! pool.reset();
//! # }
//! ```
//!
//! # Statistics Types
//!
//! The pool collects 5 statistics:
//! - **Vertex shader invocations**: Number of times vertex shader was executed
//! - **Clipper invocations**: Number of primitives entering the clipper
//! - **Clipper primitives out**: Number of primitives that passed clipping
//! - **Fragment shader invocations**: Number of times fragment shader was executed
//! - **Compute shader invocations**: Number of compute shader work group dispatches
//!
//! # Derived Metrics
//!
//! ```no_run
//! use renderer_backend::pipeline_statistics::PipelineStatisticsResult;
//!
//! # fn example(result: &PipelineStatisticsResult) {
//! // Estimate overdraw (fragment invocations vs expected pixels)
//! let expected_pixels = 1920 * 1080; // Screen resolution
//! let overdraw = result.overdraw_estimate(expected_pixels);
//! println!("Overdraw: {:.2}x", overdraw);
//!
//! // Calculate culling efficiency
//! let culling = result.culling_efficiency();
//! println!("Culling efficiency: {:.1}%", culling * 100.0);
//! # }
//! ```
//!
//! # wgpu 25.x Compatibility
//!
//! This implementation targets wgpu 22+ and follows these patterns:
//! - `device.features().contains(Features::PIPELINE_STATISTICS_QUERY)` for feature check
//! - `QueryType::PipelineStatistics(types)` for query set creation
//! - `PipelineStatisticsTypes` bitflags for selecting statistics
//! - `render_pass.begin_pipeline_statistics_query()` / `end_pipeline_statistics_query()`

use std::fmt;
use std::ops::Range;
use wgpu::{Buffer, BufferDescriptor, BufferUsages, CommandEncoder, Device, Features, MapMode, Maintain, QuerySet, QuerySetDescriptor, QueryType, PipelineStatisticsTypes, RenderPass};

// Re-export QueryError from query_pool for consistent error handling
pub use crate::query_pool::{QueryError, QueryErrorKind};

// ============================================================================
// Constants
// ============================================================================

/// Size of a single statistic value in bytes (u64).
pub const STATISTIC_SIZE_BYTES: u64 = 8;

/// Number of statistics collected per query (5 types).
pub const STATISTICS_PER_QUERY: u64 = 5;

/// Size of a complete statistics result in bytes (5 * u64 = 40 bytes).
pub const STATISTICS_RESULT_SIZE_BYTES: u64 = STATISTICS_PER_QUERY * STATISTIC_SIZE_BYTES;

/// Minimum capacity for a pipeline statistics pool.
pub const MIN_POOL_CAPACITY: u32 = 1;

/// Maximum recommended capacity for a pipeline statistics pool.
pub const MAX_RECOMMENDED_CAPACITY: u32 = 1024;

/// Default label prefix for pipeline statistics pool resources.
pub const DEFAULT_LABEL_PREFIX: &str = "PipelineStatisticsPool";

// ============================================================================
// StatisticType
// ============================================================================

/// Enumeration of individual pipeline statistic types.
///
/// These correspond to the statistics collected by `PipelineStatisticsTypes`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum StatisticType {
    /// Number of vertex shader invocations.
    VertexShaderInvocations = 0,
    /// Number of primitives entering the clipper.
    ClipperInvocations = 1,
    /// Number of primitives output by the clipper.
    ClipperPrimitivesOut = 2,
    /// Number of fragment shader invocations.
    FragmentShaderInvocations = 3,
    /// Number of compute shader invocations.
    ComputeShaderInvocations = 4,
}

impl StatisticType {
    /// Get all statistic types in order.
    pub const fn all() -> [StatisticType; 5] {
        [
            StatisticType::VertexShaderInvocations,
            StatisticType::ClipperInvocations,
            StatisticType::ClipperPrimitivesOut,
            StatisticType::FragmentShaderInvocations,
            StatisticType::ComputeShaderInvocations,
        ]
    }

    /// Get the index of this statistic type in the result array.
    #[inline]
    pub const fn index(&self) -> usize {
        *self as usize
    }

    /// Get the name of this statistic type.
    pub const fn name(&self) -> &'static str {
        match self {
            StatisticType::VertexShaderInvocations => "Vertex Shader Invocations",
            StatisticType::ClipperInvocations => "Clipper Invocations",
            StatisticType::ClipperPrimitivesOut => "Clipper Primitives Out",
            StatisticType::FragmentShaderInvocations => "Fragment Shader Invocations",
            StatisticType::ComputeShaderInvocations => "Compute Shader Invocations",
        }
    }

    /// Get a short name for this statistic type.
    pub const fn short_name(&self) -> &'static str {
        match self {
            StatisticType::VertexShaderInvocations => "VS",
            StatisticType::ClipperInvocations => "ClipIn",
            StatisticType::ClipperPrimitivesOut => "ClipOut",
            StatisticType::FragmentShaderInvocations => "FS",
            StatisticType::ComputeShaderInvocations => "CS",
        }
    }
}

impl fmt::Display for StatisticType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

// ============================================================================
// PipelineStatisticsResult
// ============================================================================

/// Result of a single pipeline statistics query.
///
/// Contains all 5 pipeline statistics collected during a render/compute pass.
///
/// # Example
///
/// ```no_run
/// use renderer_backend::pipeline_statistics::PipelineStatisticsResult;
///
/// # fn example(result: &PipelineStatisticsResult) {
/// println!("Vertex shader invocations: {}", result.vertex_shader_invocations);
/// println!("Fragment shader invocations: {}", result.fragment_shader_invocations);
/// println!("Overdraw estimate: {:.2}x", result.overdraw_estimate(1920 * 1080));
/// println!("Culling efficiency: {:.1}%", result.culling_efficiency() * 100.0);
/// # }
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct PipelineStatisticsResult {
    /// Number of vertex shader invocations.
    pub vertex_shader_invocations: u64,
    /// Number of primitives entering the clipper stage.
    pub clipper_invocations: u64,
    /// Number of primitives output by the clipper (after clipping/culling).
    pub clipper_primitives_out: u64,
    /// Number of fragment shader invocations.
    pub fragment_shader_invocations: u64,
    /// Number of compute shader invocations (workgroup dispatches).
    pub compute_shader_invocations: u64,
}

impl PipelineStatisticsResult {
    /// Create a new pipeline statistics result with all values.
    #[inline]
    pub const fn new(
        vertex_shader_invocations: u64,
        clipper_invocations: u64,
        clipper_primitives_out: u64,
        fragment_shader_invocations: u64,
        compute_shader_invocations: u64,
    ) -> Self {
        Self {
            vertex_shader_invocations,
            clipper_invocations,
            clipper_primitives_out,
            fragment_shader_invocations,
            compute_shader_invocations,
        }
    }

    /// Create a result with all zeros.
    #[inline]
    pub const fn zero() -> Self {
        Self {
            vertex_shader_invocations: 0,
            clipper_invocations: 0,
            clipper_primitives_out: 0,
            fragment_shader_invocations: 0,
            compute_shader_invocations: 0,
        }
    }

    /// Parse a result from raw u64 values in order.
    ///
    /// Expects values in this order:
    /// 1. vertex_shader_invocations
    /// 2. clipper_invocations
    /// 3. clipper_primitives_out
    /// 4. fragment_shader_invocations
    /// 5. compute_shader_invocations
    ///
    /// # Arguments
    ///
    /// * `values` - Slice of exactly 5 u64 values
    ///
    /// # Returns
    ///
    /// `Some(PipelineStatisticsResult)` if slice has 5 elements, `None` otherwise.
    pub fn from_slice(values: &[u64]) -> Option<Self> {
        if values.len() != 5 {
            return None;
        }
        Some(Self {
            vertex_shader_invocations: values[0],
            clipper_invocations: values[1],
            clipper_primitives_out: values[2],
            fragment_shader_invocations: values[3],
            compute_shader_invocations: values[4],
        })
    }

    /// Get a statistic value by type.
    #[inline]
    pub fn get(&self, stat_type: StatisticType) -> u64 {
        match stat_type {
            StatisticType::VertexShaderInvocations => self.vertex_shader_invocations,
            StatisticType::ClipperInvocations => self.clipper_invocations,
            StatisticType::ClipperPrimitivesOut => self.clipper_primitives_out,
            StatisticType::FragmentShaderInvocations => self.fragment_shader_invocations,
            StatisticType::ComputeShaderInvocations => self.compute_shader_invocations,
        }
    }

    /// Convert to an array of values in order.
    #[inline]
    pub fn to_array(&self) -> [u64; 5] {
        [
            self.vertex_shader_invocations,
            self.clipper_invocations,
            self.clipper_primitives_out,
            self.fragment_shader_invocations,
            self.compute_shader_invocations,
        ]
    }

    /// Estimate overdraw: fragment invocations / expected pixels.
    ///
    /// Overdraw occurs when the same pixel is shaded multiple times due to
    /// overlapping geometry. A value of 1.0 means no overdraw, 2.0 means
    /// each pixel was shaded twice on average.
    ///
    /// # Arguments
    ///
    /// * `expected_pixels` - Expected number of unique pixels (e.g., screen resolution)
    ///
    /// # Returns
    ///
    /// Overdraw ratio. Returns 0.0 if `expected_pixels` is 0.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use renderer_backend::pipeline_statistics::PipelineStatisticsResult;
    ///
    /// let result = PipelineStatisticsResult::new(1000, 500, 400, 4147200, 0);
    /// let screen_pixels = 1920 * 1080; // 2073600
    /// let overdraw = result.overdraw_estimate(screen_pixels);
    /// // If overdraw is 2.0, each pixel was shaded twice on average
    /// # }
    /// ```
    #[inline]
    pub fn overdraw_estimate(&self, expected_pixels: u64) -> f64 {
        if expected_pixels == 0 {
            return 0.0;
        }
        (self.fragment_shader_invocations as f64) / (expected_pixels as f64)
    }

    /// Calculate culling efficiency: 1.0 - (primitives_out / clipper_invocations).
    ///
    /// Measures how many primitives were culled (back-face, frustum, etc.)
    /// before reaching the rasterizer.
    ///
    /// # Returns
    ///
    /// Culling efficiency from 0.0 (no culling) to 1.0 (100% culled).
    /// Returns 0.0 if `clipper_invocations` is 0.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use renderer_backend::pipeline_statistics::PipelineStatisticsResult;
    ///
    /// let result = PipelineStatisticsResult::new(1000, 1000, 400, 500000, 0);
    /// let efficiency = result.culling_efficiency();
    /// // efficiency = 1.0 - (400 / 1000) = 0.6 (60% of primitives were culled)
    /// # }
    /// ```
    #[inline]
    pub fn culling_efficiency(&self) -> f64 {
        if self.clipper_invocations == 0 {
            return 0.0;
        }
        let ratio = (self.clipper_primitives_out as f64) / (self.clipper_invocations as f64);
        (1.0 - ratio).max(0.0)
    }

    /// Calculate vertex reuse ratio.
    ///
    /// Measures how many times vertices are reused across primitives.
    /// Higher values indicate better use of vertex caching.
    ///
    /// # Arguments
    ///
    /// * `vertex_count` - Number of unique vertices in the mesh
    ///
    /// # Returns
    ///
    /// Vertex reuse ratio. Returns 0.0 if `vertex_shader_invocations` is 0.
    #[inline]
    pub fn vertex_reuse_ratio(&self, vertex_count: u64) -> f64 {
        if self.vertex_shader_invocations == 0 {
            return 0.0;
        }
        (vertex_count as f64) / (self.vertex_shader_invocations as f64)
    }

    /// Calculate fragment-to-vertex ratio.
    ///
    /// Higher values indicate more fragments per vertex, which can indicate
    /// large triangles or high tessellation.
    ///
    /// # Returns
    ///
    /// Fragment-to-vertex ratio. Returns 0.0 if `vertex_shader_invocations` is 0.
    #[inline]
    pub fn fragment_to_vertex_ratio(&self) -> f64 {
        if self.vertex_shader_invocations == 0 {
            return 0.0;
        }
        (self.fragment_shader_invocations as f64) / (self.vertex_shader_invocations as f64)
    }

    /// Calculate average primitive size in fragments.
    ///
    /// # Returns
    ///
    /// Average fragments per primitive. Returns 0.0 if `clipper_primitives_out` is 0.
    #[inline]
    pub fn avg_primitive_size(&self) -> f64 {
        if self.clipper_primitives_out == 0 {
            return 0.0;
        }
        (self.fragment_shader_invocations as f64) / (self.clipper_primitives_out as f64)
    }

    /// Check if any shader was invoked.
    #[inline]
    pub fn has_activity(&self) -> bool {
        self.vertex_shader_invocations > 0
            || self.fragment_shader_invocations > 0
            || self.compute_shader_invocations > 0
    }

    /// Check if this appears to be a graphics pass (has vertex/fragment activity).
    #[inline]
    pub fn is_graphics_pass(&self) -> bool {
        self.vertex_shader_invocations > 0 || self.fragment_shader_invocations > 0
    }

    /// Check if this appears to be a compute pass (has compute activity, no graphics).
    #[inline]
    pub fn is_compute_pass(&self) -> bool {
        self.compute_shader_invocations > 0
            && self.vertex_shader_invocations == 0
            && self.fragment_shader_invocations == 0
    }

    /// Get the total shader invocations (VS + FS + CS).
    #[inline]
    pub fn total_shader_invocations(&self) -> u64 {
        self.vertex_shader_invocations
            .saturating_add(self.fragment_shader_invocations)
            .saturating_add(self.compute_shader_invocations)
    }

    /// Add another result to this one (combine statistics).
    #[inline]
    pub fn combine(&self, other: &Self) -> Self {
        Self {
            vertex_shader_invocations: self.vertex_shader_invocations
                .saturating_add(other.vertex_shader_invocations),
            clipper_invocations: self.clipper_invocations
                .saturating_add(other.clipper_invocations),
            clipper_primitives_out: self.clipper_primitives_out
                .saturating_add(other.clipper_primitives_out),
            fragment_shader_invocations: self.fragment_shader_invocations
                .saturating_add(other.fragment_shader_invocations),
            compute_shader_invocations: self.compute_shader_invocations
                .saturating_add(other.compute_shader_invocations),
        }
    }
}

impl fmt::Display for PipelineStatisticsResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "VS:{} ClipIn:{} ClipOut:{} FS:{} CS:{}",
            self.vertex_shader_invocations,
            self.clipper_invocations,
            self.clipper_primitives_out,
            self.fragment_shader_invocations,
            self.compute_shader_invocations
        )
    }
}

// ============================================================================
// LabeledStatisticsResult
// ============================================================================

/// Pipeline statistics result with an optional label.
///
/// Associates statistics with a human-readable label for identifying
/// which pass or operation was measured.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LabeledStatisticsResult {
    /// Optional human-readable label for this measurement.
    pub label: Option<String>,
    /// The statistics result.
    pub result: PipelineStatisticsResult,
    /// Query index this result came from.
    pub query_index: u32,
}

impl LabeledStatisticsResult {
    /// Create a new labeled result.
    #[inline]
    pub fn new(query_index: u32, result: PipelineStatisticsResult) -> Self {
        Self {
            label: None,
            result,
            query_index,
        }
    }

    /// Create a new labeled result with a label.
    #[inline]
    pub fn with_label(query_index: u32, result: PipelineStatisticsResult, label: impl Into<String>) -> Self {
        Self {
            label: Some(label.into()),
            result,
            query_index,
        }
    }

    /// Get the label or a default string.
    #[inline]
    pub fn label_or<'a>(&'a self, default: &'a str) -> &'a str {
        self.label.as_deref().unwrap_or(default)
    }

    /// Get the label or "unnamed".
    #[inline]
    pub fn label_or_unnamed(&self) -> &str {
        self.label_or("unnamed")
    }
}

impl Default for LabeledStatisticsResult {
    fn default() -> Self {
        Self {
            label: None,
            result: PipelineStatisticsResult::zero(),
            query_index: 0,
        }
    }
}

impl fmt::Display for LabeledStatisticsResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(ref label) = self.label {
            write!(f, "{}: {}", label, self.result)
        } else {
            write!(f, "[{}]: {}", self.query_index, self.result)
        }
    }
}

// ============================================================================
// PipelineStatisticsAllocation
// ============================================================================

/// Represents an allocated pipeline statistics query index.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PipelineStatisticsAllocation {
    /// The allocated query index.
    pub index: u32,
    /// Generation counter for validation.
    pub generation: u32,
}

impl PipelineStatisticsAllocation {
    /// Create a new allocation.
    #[inline]
    pub const fn new(index: u32, generation: u32) -> Self {
        Self { index, generation }
    }

    /// Create an allocation with default generation.
    #[inline]
    pub const fn with_index(index: u32) -> Self {
        Self { index, generation: 0 }
    }
}

impl fmt::Display for PipelineStatisticsAllocation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PipelineStatsAllocation(index={}, gen={})", self.index, self.generation)
    }
}

// ============================================================================
// PipelineStatisticsResolveParams
// ============================================================================

/// Parameters for resolving pipeline statistics queries to a buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PipelineStatisticsResolveParams {
    /// Starting query index in the QuerySet.
    pub start_query: u32,
    /// Number of queries to resolve.
    pub query_count: u32,
    /// Byte offset in the destination resolve buffer.
    pub destination_offset: u64,
}

impl PipelineStatisticsResolveParams {
    /// Create new resolve parameters.
    #[inline]
    pub const fn new(start_query: u32, query_count: u32, destination_offset: u64) -> Self {
        Self {
            start_query,
            query_count,
            destination_offset,
        }
    }

    /// Create parameters to resolve from the beginning.
    #[inline]
    pub const fn from_start(query_count: u32) -> Self {
        Self {
            start_query: 0,
            query_count,
            destination_offset: 0,
        }
    }

    /// Calculate the end query index (exclusive).
    #[inline]
    pub const fn end_query(&self) -> u32 {
        self.start_query + self.query_count
    }

    /// Calculate the required buffer size in bytes.
    #[inline]
    pub const fn required_buffer_size(&self) -> u64 {
        self.destination_offset + (self.query_count as u64) * STATISTICS_RESULT_SIZE_BYTES
    }
}

impl Default for PipelineStatisticsResolveParams {
    fn default() -> Self {
        Self {
            start_query: 0,
            query_count: 0,
            destination_offset: 0,
        }
    }
}

impl fmt::Display for PipelineStatisticsResolveParams {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "PipelineStatsResolveParams(queries={}..{}, offset={})",
            self.start_query,
            self.end_query(),
            self.destination_offset
        )
    }
}

// ============================================================================
// PipelineStatisticsData
// ============================================================================

/// Raw pipeline statistics data from readback.
///
/// Contains parsed statistics results for multiple queries.
#[derive(Debug, Clone)]
pub struct PipelineStatisticsData {
    /// Parsed statistics results.
    pub results: Vec<PipelineStatisticsResult>,
    /// Starting query index for this data.
    pub start_index: u32,
}

impl PipelineStatisticsData {
    /// Create new statistics data.
    #[inline]
    pub fn new(results: Vec<PipelineStatisticsResult>, start_index: u32) -> Self {
        Self { results, start_index }
    }

    /// Create empty statistics data.
    #[inline]
    pub fn empty() -> Self {
        Self {
            results: Vec::new(),
            start_index: 0,
        }
    }

    /// Get the number of results.
    #[inline]
    pub fn len(&self) -> usize {
        self.results.len()
    }

    /// Check if empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.results.is_empty()
    }

    /// Get result at relative index.
    #[inline]
    pub fn get(&self, relative_index: usize) -> Option<&PipelineStatisticsResult> {
        self.results.get(relative_index)
    }

    /// Get result by absolute query index.
    pub fn get_by_query_index(&self, query_index: u32) -> Option<&PipelineStatisticsResult> {
        if query_index < self.start_index {
            return None;
        }
        let relative = (query_index - self.start_index) as usize;
        self.results.get(relative)
    }

    /// Calculate aggregate statistics across all results.
    pub fn aggregate(&self) -> PipelineStatisticsResult {
        self.results.iter().fold(PipelineStatisticsResult::zero(), |acc, r| acc.combine(r))
    }

    /// Get overdraw estimates for all results.
    pub fn overdraw_estimates(&self, expected_pixels: u64) -> Vec<f64> {
        self.results.iter().map(|r| r.overdraw_estimate(expected_pixels)).collect()
    }

    /// Get culling efficiencies for all results.
    pub fn culling_efficiencies(&self) -> Vec<f64> {
        self.results.iter().map(|r| r.culling_efficiency()).collect()
    }

    /// Find the result with maximum fragment shader invocations.
    pub fn max_fragment_invocations(&self) -> Option<(u32, &PipelineStatisticsResult)> {
        self.results
            .iter()
            .enumerate()
            .max_by_key(|(_, r)| r.fragment_shader_invocations)
            .map(|(i, r)| (self.start_index + i as u32, r))
    }

    /// Find results with overdraw above a threshold.
    pub fn overdraw_hotspots(&self, expected_pixels: u64, threshold: f64) -> Vec<(u32, f64)> {
        self.results
            .iter()
            .enumerate()
            .filter_map(|(i, r)| {
                let od = r.overdraw_estimate(expected_pixels);
                if od > threshold {
                    Some((self.start_index + i as u32, od))
                } else {
                    None
                }
            })
            .collect()
    }
}

impl Default for PipelineStatisticsData {
    fn default() -> Self {
        Self::empty()
    }
}

// ============================================================================
// PipelineStatisticsPool
// ============================================================================

/// A pool for managing wgpu pipeline statistics queries.
///
/// Provides detailed GPU execution statistics including shader invocation counts,
/// primitive counts, and derived metrics like overdraw and culling efficiency.
///
/// # Feature Requirement
///
/// Pipeline statistics queries require the `PIPELINE_STATISTICS_QUERY` feature.
/// Use `is_supported()` to check if the device supports this feature.
///
/// # Example
///
/// ```no_run
/// use renderer_backend::pipeline_statistics::{PipelineStatisticsPool, PipelineStatisticsResult};
///
/// # fn example(device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder) {
/// // Check feature support
/// if !PipelineStatisticsPool::is_supported(device) {
///     return;
/// }
///
/// // Create pool with 64 query capacity
/// let mut pool = PipelineStatisticsPool::new(device, 64).unwrap();
///
/// // Allocate a query for the main pass
/// let query_idx = pool.allocate().unwrap();
///
/// // In render pass:
/// // pool.begin_query(&mut render_pass, query_idx)?;
/// // ... draw calls ...
/// // pool.end_query(&mut render_pass)?;
///
/// // Resolve results
/// pool.resolve_all(encoder).unwrap();
///
/// // Reset for next frame
/// pool.reset();
/// # }
/// ```
pub struct PipelineStatisticsPool {
    /// The wgpu QuerySet for pipeline statistics queries.
    query_set: QuerySet,
    /// Buffer for resolving query results.
    resolve_buffer: Buffer,
    /// Maximum number of queries this pool can hold.
    capacity: u32,
    /// Next index to allocate (watermark allocator).
    next_index: u32,
    /// Generation counter for allocation tracking.
    generation: u32,
    /// Optional label for debugging.
    label: Option<String>,
    /// The statistics types being collected.
    statistics_types: PipelineStatisticsTypes,
    /// Tracks if a query is currently active.
    active_query: Option<u32>,
}

impl PipelineStatisticsPool {
    /// Create a new pipeline statistics query pool.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device to create resources on
    /// * `capacity` - Maximum number of queries
    ///
    /// # Returns
    ///
    /// * `Ok(PipelineStatisticsPool)` - Successfully created pool
    /// * `Err(QueryError::FeatureNotSupported)` - Device lacks PIPELINE_STATISTICS_QUERY
    /// * `Err(QueryError::InvalidCapacity)` - Capacity is 0
    ///
    /// # Example
    ///
    /// ```no_run
    /// use renderer_backend::pipeline_statistics::PipelineStatisticsPool;
    ///
    /// # fn example(device: &wgpu::Device) {
    /// let pool = PipelineStatisticsPool::new(device, 64)?;
    /// println!("Created pool with {} capacity", pool.capacity());
    /// # Ok::<(), renderer_backend::query_pool::QueryError>(())
    /// # }
    /// ```
    pub fn new(device: &Device, capacity: u32) -> Result<Self, QueryError> {
        Self::with_label(device, capacity, None)
    }

    /// Create a new pipeline statistics query pool with a custom label.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device
    /// * `capacity` - Maximum number of queries
    /// * `label` - Optional debug label
    pub fn with_label(
        device: &Device,
        capacity: u32,
        label: Option<&str>,
    ) -> Result<Self, QueryError> {
        // Validate capacity
        if capacity < MIN_POOL_CAPACITY {
            return Err(QueryError::invalid_capacity(capacity));
        }

        // Check feature support
        if !Self::is_supported(device) {
            return Err(QueryError::feature_not_supported());
        }

        // Define all 5 statistics types
        let statistics_types = PipelineStatisticsTypes::VERTEX_SHADER_INVOCATIONS
            | PipelineStatisticsTypes::CLIPPER_INVOCATIONS
            | PipelineStatisticsTypes::CLIPPER_PRIMITIVES_OUT
            | PipelineStatisticsTypes::FRAGMENT_SHADER_INVOCATIONS
            | PipelineStatisticsTypes::COMPUTE_SHADER_INVOCATIONS;

        // Create the QuerySet
        let query_set_label = label
            .map(|l| format!("{} QuerySet", l))
            .unwrap_or_else(|| format!("{} QuerySet", DEFAULT_LABEL_PREFIX));

        let query_set = device.create_query_set(&QuerySetDescriptor {
            label: Some(&query_set_label),
            ty: QueryType::PipelineStatistics(statistics_types),
            count: capacity,
        });

        // Create resolve buffer (5 * u64 = 40 bytes per query)
        let buffer_size = (capacity as u64) * STATISTICS_RESULT_SIZE_BYTES;
        let buffer_label = label
            .map(|l| format!("{} Resolve Buffer", l))
            .unwrap_or_else(|| format!("{} Resolve Buffer", DEFAULT_LABEL_PREFIX));

        let resolve_buffer = device.create_buffer(&BufferDescriptor {
            label: Some(&buffer_label),
            size: buffer_size,
            usage: BufferUsages::QUERY_RESOLVE | BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        Ok(Self {
            query_set,
            resolve_buffer,
            capacity,
            next_index: 0,
            generation: 0,
            label: label.map(String::from),
            statistics_types,
            active_query: None,
        })
    }

    /// Check if pipeline statistics queries are supported on the device.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device to check
    ///
    /// # Returns
    ///
    /// `true` if pipeline statistics queries are supported, `false` otherwise.
    #[inline]
    pub fn is_supported(device: &Device) -> bool {
        device.features().contains(Features::PIPELINE_STATISTICS_QUERY)
    }

    /// Allocate the next available query index.
    ///
    /// # Returns
    ///
    /// * `Ok(index)` - The allocated query index
    /// * `Err(QueryError::PoolExhausted)` - No more indices available
    #[inline]
    pub fn allocate(&mut self) -> Result<u32, QueryError> {
        if self.next_index >= self.capacity {
            return Err(QueryError::pool_exhausted(self.capacity));
        }

        let index = self.next_index;
        self.next_index += 1;
        Ok(index)
    }

    /// Allocate a query index with full allocation metadata.
    pub fn allocate_tracked(&mut self) -> Result<PipelineStatisticsAllocation, QueryError> {
        let index = self.allocate()?;
        Ok(PipelineStatisticsAllocation::new(index, self.generation))
    }

    /// Try to allocate multiple consecutive query indices.
    pub fn allocate_range(&mut self, count: u32) -> Result<u32, QueryError> {
        if count == 0 {
            return Ok(self.next_index);
        }

        let end_index = self.next_index.checked_add(count).ok_or_else(|| {
            QueryError::pool_exhausted(self.capacity)
        })?;

        if end_index > self.capacity {
            return Err(QueryError::pool_exhausted(self.capacity));
        }

        let start = self.next_index;
        self.next_index = end_index;
        Ok(start)
    }

    /// Reset the pool for reuse.
    #[inline]
    pub fn reset(&mut self) {
        self.next_index = 0;
        self.generation = self.generation.wrapping_add(1);
        self.active_query = None;
    }

    // ========================================================================
    // Render Pass Query Operations
    // ========================================================================

    /// Begin a pipeline statistics query at the specified index.
    ///
    /// # Arguments
    ///
    /// * `render_pass` - The active render pass
    /// * `index` - Query index to begin
    ///
    /// # Returns
    ///
    /// * `Ok(())` - Query started successfully
    /// * `Err(QueryError::InvalidIndex)` - Index out of bounds
    /// * `Err(QueryError::ResolveFailed)` - Another query is already active
    pub fn begin_query<'a>(&mut self, render_pass: &mut RenderPass<'a>, index: u32) -> Result<(), QueryError> {
        // Validate index
        if index >= self.capacity {
            return Err(QueryError::invalid_index(index, self.capacity));
        }

        // Check if another query is active
        if self.active_query.is_some() {
            return Err(QueryError::resolve_failed(
                "another pipeline statistics query is already active; call end_query first"
            ));
        }

        render_pass.begin_pipeline_statistics_query(&self.query_set, index);
        self.active_query = Some(index);

        Ok(())
    }

    /// End the current pipeline statistics query.
    ///
    /// # Arguments
    ///
    /// * `render_pass` - The active render pass
    ///
    /// # Returns
    ///
    /// * `Ok(())` - Query ended successfully
    /// * `Err(QueryError::ResolveFailed)` - No query is currently active
    pub fn end_query<'a>(&mut self, render_pass: &mut RenderPass<'a>) -> Result<(), QueryError> {
        if self.active_query.is_none() {
            return Err(QueryError::resolve_failed(
                "no pipeline statistics query is active; call begin_query first"
            ));
        }

        render_pass.end_pipeline_statistics_query();
        self.active_query = None;

        Ok(())
    }

    /// Convenience method to run a query around a closure.
    ///
    /// Automatically handles begin/end query calls.
    ///
    /// # Arguments
    ///
    /// * `render_pass` - The active render pass
    /// * `index` - Query index to use
    /// * `draw_fn` - Closure that performs draw calls
    pub fn with_query<'a, F>(
        &mut self,
        render_pass: &mut RenderPass<'a>,
        index: u32,
        draw_fn: F,
    ) -> Result<(), QueryError>
    where
        F: FnOnce(&mut RenderPass<'a>),
    {
        self.begin_query(render_pass, index)?;
        draw_fn(render_pass);
        self.end_query(render_pass)?;
        Ok(())
    }

    // ========================================================================
    // Query Resolution
    // ========================================================================

    /// Resolve queries to the internal resolve buffer.
    ///
    /// # Timing Requirements
    ///
    /// Call **after** all render/compute passes complete,
    /// **before** the command buffer is submitted.
    ///
    /// # Arguments
    ///
    /// * `encoder` - The command encoder to record the resolve command
    /// * `range` - Range of query indices to resolve
    pub fn resolve(
        &self,
        encoder: &mut CommandEncoder,
        range: Range<u32>,
    ) -> Result<(), QueryError> {
        // Validate range
        if range.is_empty() {
            return Err(QueryError::resolve_failed("query range is empty"));
        }

        if range.end > self.capacity {
            return Err(QueryError::invalid_query_range(range.start, range.end, self.capacity));
        }

        let offset = (range.start as u64) * STATISTICS_RESULT_SIZE_BYTES;

        encoder.resolve_query_set(
            &self.query_set,
            range,
            &self.resolve_buffer,
            offset,
        );

        Ok(())
    }

    /// Resolve all used queries to the resolve buffer.
    pub fn resolve_all(&self, encoder: &mut CommandEncoder) -> Result<(), QueryError> {
        let used = self.used();
        if used == 0 {
            return Ok(());
        }

        self.resolve(encoder, 0..used)
    }

    /// Resolve queries with custom parameters.
    pub fn resolve_with_params(
        &self,
        encoder: &mut CommandEncoder,
        params: PipelineStatisticsResolveParams,
    ) -> Result<(), QueryError> {
        if params.query_count == 0 {
            return Err(QueryError::resolve_failed("query_count must be greater than 0"));
        }

        let end_query = params.start_query.checked_add(params.query_count)
            .ok_or_else(|| QueryError::resolve_failed("query range overflow"))?;

        if end_query > self.capacity {
            return Err(QueryError::resolve_failed(format!(
                "query range out of bounds: {} > {}",
                end_query, self.capacity
            )));
        }

        encoder.resolve_query_set(
            &self.query_set,
            params.start_query..end_query,
            &self.resolve_buffer,
            params.destination_offset,
        );

        Ok(())
    }

    // ========================================================================
    // Result Reading
    // ========================================================================

    /// Read resolved pipeline statistics query results.
    ///
    /// # Timing Requirements
    ///
    /// Call **after** the command buffer containing the resolve operation
    /// has been submitted and GPU work is complete.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device
    /// * `query_range` - Range of query indices to read
    ///
    /// # Returns
    ///
    /// * `Ok(PipelineStatisticsData)` - Successfully read results
    /// * `Err(QueryError)` - Read failed
    pub fn read_results(
        &self,
        device: &Device,
        query_range: Range<u32>,
    ) -> Result<PipelineStatisticsData, QueryError> {
        // Validate range
        if query_range.is_empty() {
            return Err(QueryError::invalid_query_range(
                query_range.start,
                query_range.end,
                self.capacity,
            ));
        }

        if query_range.end > self.capacity {
            return Err(QueryError::invalid_query_range(
                query_range.start,
                query_range.end,
                self.capacity,
            ));
        }

        // Calculate buffer slice
        let offset = (query_range.start as u64) * STATISTICS_RESULT_SIZE_BYTES;
        let size = ((query_range.end - query_range.start) as u64) * STATISTICS_RESULT_SIZE_BYTES;

        let slice = self.resolve_buffer.slice(offset..offset + size);

        // Map the buffer
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(MapMode::Read, move |result| {
            let _ = tx.send(result);
        });

        device.poll(Maintain::Wait);

        rx.recv()
            .map_err(|_| QueryError::buffer_mapping_failed("channel receive failed"))?
            .map_err(|e| QueryError::buffer_mapping_failed(format!("map_async failed: {:?}", e)))?;

        // Read and parse the data
        let mapped_range = slice.get_mapped_range();
        let query_count = (query_range.end - query_range.start) as usize;
        let mut results = Vec::with_capacity(query_count);

        for i in 0..query_count {
            let base_offset = i * (STATISTICS_RESULT_SIZE_BYTES as usize);
            let mut values = [0u64; 5];

            for (j, value) in values.iter_mut().enumerate() {
                let byte_offset = base_offset + j * 8;
                if byte_offset + 8 <= mapped_range.len() {
                    let bytes: [u8; 8] = mapped_range[byte_offset..byte_offset + 8]
                        .try_into()
                        .map_err(|_| QueryError::buffer_mapping_failed("failed to read u64 from buffer"))?;
                    *value = u64::from_le_bytes(bytes);
                }
            }

            if let Some(result) = PipelineStatisticsResult::from_slice(&values) {
                results.push(result);
            }
        }

        drop(mapped_range);
        self.resolve_buffer.unmap();

        Ok(PipelineStatisticsData::new(results, query_range.start))
    }

    /// Read all used query results.
    pub fn read_all_results(&self, device: &Device) -> Result<PipelineStatisticsData, QueryError> {
        let used = self.used();
        if used == 0 {
            return Ok(PipelineStatisticsData::empty());
        }

        self.read_results(device, 0..used)
    }

    /// Read results as a vector of PipelineStatisticsResult.
    pub fn read_results_vec(
        &self,
        device: &Device,
    ) -> Result<Vec<PipelineStatisticsResult>, QueryError> {
        let data = self.read_all_results(device)?;
        Ok(data.results)
    }

    // ========================================================================
    // Accessors
    // ========================================================================

    /// Get the maximum capacity of this pool.
    #[inline]
    pub fn capacity(&self) -> u32 {
        self.capacity
    }

    /// Get the number of available (unallocated) indices.
    #[inline]
    pub fn available(&self) -> u32 {
        self.capacity.saturating_sub(self.next_index)
    }

    /// Get the number of used (allocated) indices.
    #[inline]
    pub fn used(&self) -> u32 {
        self.next_index
    }

    /// Check if the pool has capacity for at least one more allocation.
    #[inline]
    pub fn has_capacity(&self) -> bool {
        self.next_index < self.capacity
    }

    /// Check if the pool has capacity for N more allocations.
    #[inline]
    pub fn has_capacity_for(&self, count: u32) -> bool {
        self.next_index.saturating_add(count) <= self.capacity
    }

    /// Check if the pool is empty (no allocations).
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.next_index == 0
    }

    /// Check if the pool is full.
    #[inline]
    pub fn is_full(&self) -> bool {
        self.next_index >= self.capacity
    }

    /// Get a reference to the underlying QuerySet.
    #[inline]
    pub fn query_set(&self) -> &QuerySet {
        &self.query_set
    }

    /// Get a reference to the resolve buffer.
    #[inline]
    pub fn resolve_buffer(&self) -> &Buffer {
        &self.resolve_buffer
    }

    /// Get the current generation counter.
    #[inline]
    pub fn generation(&self) -> u32 {
        self.generation
    }

    /// Get the debug label, if any.
    #[inline]
    pub fn label(&self) -> Option<&str> {
        self.label.as_deref()
    }

    /// Get the statistics types being collected.
    #[inline]
    pub fn statistics_types(&self) -> PipelineStatisticsTypes {
        self.statistics_types
    }

    /// Get the resolve buffer size in bytes.
    #[inline]
    pub fn resolve_buffer_size(&self) -> u64 {
        (self.capacity as u64) * STATISTICS_RESULT_SIZE_BYTES
    }

    /// Check if a query is currently active.
    #[inline]
    pub fn has_active_query(&self) -> bool {
        self.active_query.is_some()
    }

    /// Get the currently active query index, if any.
    #[inline]
    pub fn active_query_index(&self) -> Option<u32> {
        self.active_query
    }

    /// Validate an index is within bounds.
    #[inline]
    pub fn validate_index(&self, index: u32) -> Result<(), QueryError> {
        if index >= self.capacity {
            return Err(QueryError::invalid_index(index, self.capacity));
        }
        Ok(())
    }

    /// Get statistics about pool usage.
    pub fn stats(&self) -> PipelineStatisticsPoolStats {
        PipelineStatisticsPoolStats {
            capacity: self.capacity,
            used: self.next_index,
            available: self.available(),
            generation: self.generation,
            resolve_buffer_size: self.resolve_buffer_size(),
            has_active_query: self.active_query.is_some(),
        }
    }
}

impl fmt::Debug for PipelineStatisticsPool {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PipelineStatisticsPool")
            .field("capacity", &self.capacity)
            .field("used", &self.next_index)
            .field("available", &self.available())
            .field("generation", &self.generation)
            .field("label", &self.label)
            .field("active_query", &self.active_query)
            .finish()
    }
}

// ============================================================================
// PipelineStatisticsPoolStats
// ============================================================================

/// Statistics about a pipeline statistics pool's current state.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PipelineStatisticsPoolStats {
    /// Total capacity of the pool.
    pub capacity: u32,
    /// Number of allocated indices.
    pub used: u32,
    /// Number of available indices.
    pub available: u32,
    /// Current generation counter.
    pub generation: u32,
    /// Resolve buffer size in bytes.
    pub resolve_buffer_size: u64,
    /// Whether a query is currently active.
    pub has_active_query: bool,
}

impl PipelineStatisticsPoolStats {
    /// Get the utilization percentage (0.0 to 1.0).
    #[inline]
    pub fn utilization(&self) -> f32 {
        if self.capacity == 0 {
            0.0
        } else {
            (self.used as f32) / (self.capacity as f32)
        }
    }

    /// Check if the pool is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.used == 0
    }

    /// Check if the pool is full.
    #[inline]
    pub fn is_full(&self) -> bool {
        self.used >= self.capacity
    }
}

impl fmt::Display for PipelineStatisticsPoolStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "PipelineStatsPool: {}/{} used ({:.1}%), gen={}",
            self.used,
            self.capacity,
            self.utilization() * 100.0,
            self.generation
        )
    }
}

// ============================================================================
// PipelineStatisticsPoolBuilder
// ============================================================================

/// Builder for creating PipelineStatisticsPool with custom options.
///
/// # Example
///
/// ```no_run
/// use renderer_backend::pipeline_statistics::PipelineStatisticsPoolBuilder;
///
/// # fn example(device: &wgpu::Device) {
/// let pool = PipelineStatisticsPoolBuilder::new()
///     .capacity(128)
///     .label("FrameStats")
///     .build(device)?;
/// # Ok::<(), renderer_backend::query_pool::QueryError>(())
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct PipelineStatisticsPoolBuilder {
    capacity: u32,
    label: Option<String>,
}

impl Default for PipelineStatisticsPoolBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl PipelineStatisticsPoolBuilder {
    /// Create a new builder with default settings.
    pub fn new() -> Self {
        Self {
            capacity: 64,
            label: None,
        }
    }

    /// Set the pool capacity.
    pub fn capacity(mut self, capacity: u32) -> Self {
        self.capacity = capacity;
        self
    }

    /// Set a debug label for the pool.
    pub fn label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Build the PipelineStatisticsPool.
    pub fn build(self, device: &Device) -> Result<PipelineStatisticsPool, QueryError> {
        PipelineStatisticsPool::with_label(device, self.capacity, self.label.as_deref())
    }
}

// ============================================================================
// Utility Functions
// ============================================================================

/// Check if pipeline statistics queries are supported on a device.
#[inline]
pub fn is_pipeline_statistics_supported(device: &Device) -> bool {
    PipelineStatisticsPool::is_supported(device)
}

/// Calculate the required resolve buffer size for a given capacity.
#[inline]
pub const fn calculate_resolve_buffer_size(capacity: u32) -> u64 {
    (capacity as u64) * STATISTICS_RESULT_SIZE_BYTES
}

/// Get all pipeline statistics types as a bitflags value.
#[inline]
pub fn all_statistics_types() -> PipelineStatisticsTypes {
    PipelineStatisticsTypes::VERTEX_SHADER_INVOCATIONS
        | PipelineStatisticsTypes::CLIPPER_INVOCATIONS
        | PipelineStatisticsTypes::CLIPPER_PRIMITIVES_OUT
        | PipelineStatisticsTypes::FRAGMENT_SHADER_INVOCATIONS
        | PipelineStatisticsTypes::COMPUTE_SHADER_INVOCATIONS
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ===== SECTION 1: StatisticType tests =====

    #[test]
    fn statistic_type_all_returns_5() {
        let all = StatisticType::all();
        assert_eq!(all.len(), 5);
    }

    #[test]
    fn statistic_type_indices_are_sequential() {
        let all = StatisticType::all();
        for (i, st) in all.iter().enumerate() {
            assert_eq!(st.index(), i);
        }
    }

    #[test]
    fn statistic_type_names_not_empty() {
        for st in StatisticType::all() {
            assert!(!st.name().is_empty());
            assert!(!st.short_name().is_empty());
        }
    }

    #[test]
    fn statistic_type_display() {
        let st = StatisticType::VertexShaderInvocations;
        let display = format!("{}", st);
        assert!(display.contains("Vertex"));
    }

    #[test]
    fn statistic_type_equality() {
        assert_eq!(StatisticType::VertexShaderInvocations, StatisticType::VertexShaderInvocations);
        assert_ne!(StatisticType::VertexShaderInvocations, StatisticType::FragmentShaderInvocations);
    }

    #[test]
    fn statistic_type_hash_consistency() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(StatisticType::VertexShaderInvocations);
        set.insert(StatisticType::FragmentShaderInvocations);
        set.insert(StatisticType::VertexShaderInvocations); // Duplicate
        assert_eq!(set.len(), 2);
    }

    // ===== SECTION 2: PipelineStatisticsResult tests =====

    #[test]
    fn stats_result_new_sets_all_values() {
        let result = PipelineStatisticsResult::new(100, 200, 150, 1000, 50);
        assert_eq!(result.vertex_shader_invocations, 100);
        assert_eq!(result.clipper_invocations, 200);
        assert_eq!(result.clipper_primitives_out, 150);
        assert_eq!(result.fragment_shader_invocations, 1000);
        assert_eq!(result.compute_shader_invocations, 50);
    }

    #[test]
    fn stats_result_zero_is_all_zeros() {
        let result = PipelineStatisticsResult::zero();
        assert_eq!(result.vertex_shader_invocations, 0);
        assert_eq!(result.clipper_invocations, 0);
        assert_eq!(result.clipper_primitives_out, 0);
        assert_eq!(result.fragment_shader_invocations, 0);
        assert_eq!(result.compute_shader_invocations, 0);
    }

    #[test]
    fn stats_result_from_slice_valid() {
        let values = [10u64, 20, 15, 100, 5];
        let result = PipelineStatisticsResult::from_slice(&values).unwrap();
        assert_eq!(result.vertex_shader_invocations, 10);
        assert_eq!(result.compute_shader_invocations, 5);
    }

    #[test]
    fn stats_result_from_slice_invalid_length() {
        let values = [1u64, 2, 3]; // Only 3 values
        assert!(PipelineStatisticsResult::from_slice(&values).is_none());
    }

    #[test]
    fn stats_result_from_slice_too_many() {
        let values = [1u64, 2, 3, 4, 5, 6]; // 6 values
        assert!(PipelineStatisticsResult::from_slice(&values).is_none());
    }

    #[test]
    fn stats_result_get_by_type() {
        let result = PipelineStatisticsResult::new(1, 2, 3, 4, 5);
        assert_eq!(result.get(StatisticType::VertexShaderInvocations), 1);
        assert_eq!(result.get(StatisticType::ClipperInvocations), 2);
        assert_eq!(result.get(StatisticType::ClipperPrimitivesOut), 3);
        assert_eq!(result.get(StatisticType::FragmentShaderInvocations), 4);
        assert_eq!(result.get(StatisticType::ComputeShaderInvocations), 5);
    }

    #[test]
    fn stats_result_to_array() {
        let result = PipelineStatisticsResult::new(1, 2, 3, 4, 5);
        let arr = result.to_array();
        assert_eq!(arr, [1, 2, 3, 4, 5]);
    }

    #[test]
    fn stats_result_overdraw_estimate_normal() {
        let result = PipelineStatisticsResult::new(1000, 500, 400, 4000, 0);
        let overdraw = result.overdraw_estimate(2000);
        assert!((overdraw - 2.0).abs() < 0.001); // 4000 / 2000 = 2.0
    }

    #[test]
    fn stats_result_overdraw_estimate_zero_pixels() {
        let result = PipelineStatisticsResult::new(1000, 500, 400, 4000, 0);
        let overdraw = result.overdraw_estimate(0);
        assert_eq!(overdraw, 0.0);
    }

    #[test]
    fn stats_result_overdraw_estimate_no_fragments() {
        let result = PipelineStatisticsResult::new(1000, 500, 400, 0, 0);
        let overdraw = result.overdraw_estimate(2000);
        assert_eq!(overdraw, 0.0);
    }

    #[test]
    fn stats_result_culling_efficiency_half() {
        let result = PipelineStatisticsResult::new(0, 1000, 500, 0, 0);
        let efficiency = result.culling_efficiency();
        assert!((efficiency - 0.5).abs() < 0.001); // 1.0 - (500/1000) = 0.5
    }

    #[test]
    fn stats_result_culling_efficiency_zero_invocations() {
        let result = PipelineStatisticsResult::new(0, 0, 0, 0, 0);
        let efficiency = result.culling_efficiency();
        assert_eq!(efficiency, 0.0);
    }

    #[test]
    fn stats_result_culling_efficiency_no_culling() {
        let result = PipelineStatisticsResult::new(0, 100, 100, 0, 0);
        let efficiency = result.culling_efficiency();
        assert_eq!(efficiency, 0.0); // All primitives pass through
    }

    #[test]
    fn stats_result_culling_efficiency_full_culling() {
        let result = PipelineStatisticsResult::new(0, 100, 0, 0, 0);
        let efficiency = result.culling_efficiency();
        assert_eq!(efficiency, 1.0); // All primitives culled
    }

    #[test]
    fn stats_result_vertex_reuse_ratio() {
        let result = PipelineStatisticsResult::new(200, 0, 0, 0, 0);
        let ratio = result.vertex_reuse_ratio(100);
        assert!((ratio - 0.5).abs() < 0.001); // 100/200 = 0.5
    }

    #[test]
    fn stats_result_vertex_reuse_ratio_zero_invocations() {
        let result = PipelineStatisticsResult::new(0, 0, 0, 0, 0);
        let ratio = result.vertex_reuse_ratio(100);
        assert_eq!(ratio, 0.0);
    }

    #[test]
    fn stats_result_fragment_to_vertex_ratio() {
        let result = PipelineStatisticsResult::new(100, 0, 0, 1000, 0);
        let ratio = result.fragment_to_vertex_ratio();
        assert!((ratio - 10.0).abs() < 0.001);
    }

    #[test]
    fn stats_result_fragment_to_vertex_ratio_zero_vs() {
        let result = PipelineStatisticsResult::new(0, 0, 0, 1000, 0);
        let ratio = result.fragment_to_vertex_ratio();
        assert_eq!(ratio, 0.0);
    }

    #[test]
    fn stats_result_avg_primitive_size() {
        let result = PipelineStatisticsResult::new(0, 0, 100, 5000, 0);
        let size = result.avg_primitive_size();
        assert!((size - 50.0).abs() < 0.001); // 5000/100 = 50
    }

    #[test]
    fn stats_result_avg_primitive_size_zero_primitives() {
        let result = PipelineStatisticsResult::new(0, 0, 0, 5000, 0);
        let size = result.avg_primitive_size();
        assert_eq!(size, 0.0);
    }

    #[test]
    fn stats_result_has_activity_true() {
        let result = PipelineStatisticsResult::new(1, 0, 0, 0, 0);
        assert!(result.has_activity());

        let result2 = PipelineStatisticsResult::new(0, 0, 0, 1, 0);
        assert!(result2.has_activity());

        let result3 = PipelineStatisticsResult::new(0, 0, 0, 0, 1);
        assert!(result3.has_activity());
    }

    #[test]
    fn stats_result_has_activity_false() {
        let result = PipelineStatisticsResult::zero();
        assert!(!result.has_activity());
    }

    #[test]
    fn stats_result_is_graphics_pass() {
        let result = PipelineStatisticsResult::new(100, 50, 40, 500, 0);
        assert!(result.is_graphics_pass());
        assert!(!result.is_compute_pass());
    }

    #[test]
    fn stats_result_is_compute_pass() {
        let result = PipelineStatisticsResult::new(0, 0, 0, 0, 64);
        assert!(result.is_compute_pass());
        assert!(!result.is_graphics_pass());
    }

    #[test]
    fn stats_result_total_shader_invocations() {
        let result = PipelineStatisticsResult::new(100, 0, 0, 500, 50);
        assert_eq!(result.total_shader_invocations(), 650);
    }

    #[test]
    fn stats_result_total_shader_invocations_saturating() {
        let result = PipelineStatisticsResult::new(u64::MAX, 0, 0, 1, 1);
        assert_eq!(result.total_shader_invocations(), u64::MAX);
    }

    #[test]
    fn stats_result_combine() {
        let a = PipelineStatisticsResult::new(10, 20, 15, 100, 5);
        let b = PipelineStatisticsResult::new(5, 10, 5, 50, 10);
        let combined = a.combine(&b);
        assert_eq!(combined.vertex_shader_invocations, 15);
        assert_eq!(combined.clipper_invocations, 30);
        assert_eq!(combined.clipper_primitives_out, 20);
        assert_eq!(combined.fragment_shader_invocations, 150);
        assert_eq!(combined.compute_shader_invocations, 15);
    }

    #[test]
    fn stats_result_combine_saturating() {
        let a = PipelineStatisticsResult::new(u64::MAX, 0, 0, 0, 0);
        let b = PipelineStatisticsResult::new(1, 0, 0, 0, 0);
        let combined = a.combine(&b);
        assert_eq!(combined.vertex_shader_invocations, u64::MAX);
    }

    #[test]
    fn stats_result_default() {
        let result = PipelineStatisticsResult::default();
        assert_eq!(result, PipelineStatisticsResult::zero());
    }

    #[test]
    fn stats_result_display() {
        let result = PipelineStatisticsResult::new(100, 50, 40, 500, 10);
        let display = format!("{}", result);
        assert!(display.contains("100"));
        assert!(display.contains("500"));
    }

    #[test]
    fn stats_result_equality() {
        let a = PipelineStatisticsResult::new(1, 2, 3, 4, 5);
        let b = PipelineStatisticsResult::new(1, 2, 3, 4, 5);
        let c = PipelineStatisticsResult::new(1, 2, 3, 4, 6);
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn stats_result_hash_consistency() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(PipelineStatisticsResult::new(1, 2, 3, 4, 5));
        set.insert(PipelineStatisticsResult::new(5, 4, 3, 2, 1));
        set.insert(PipelineStatisticsResult::new(1, 2, 3, 4, 5)); // Duplicate
        assert_eq!(set.len(), 2);
    }

    // ===== SECTION 3: LabeledStatisticsResult tests =====

    #[test]
    fn labeled_result_new() {
        let result = PipelineStatisticsResult::new(1, 2, 3, 4, 5);
        let labeled = LabeledStatisticsResult::new(0, result);
        assert_eq!(labeled.query_index, 0);
        assert!(labeled.label.is_none());
    }

    #[test]
    fn labeled_result_with_label() {
        let result = PipelineStatisticsResult::new(1, 2, 3, 4, 5);
        let labeled = LabeledStatisticsResult::with_label(5, result, "Shadow Pass");
        assert_eq!(labeled.query_index, 5);
        assert_eq!(labeled.label, Some("Shadow Pass".to_string()));
    }

    #[test]
    fn labeled_result_label_or() {
        let result = PipelineStatisticsResult::zero();
        let labeled = LabeledStatisticsResult::new(0, result);
        assert_eq!(labeled.label_or("default"), "default");

        let labeled_with = LabeledStatisticsResult::with_label(0, result, "test");
        assert_eq!(labeled_with.label_or("default"), "test");
    }

    #[test]
    fn labeled_result_display_with_label() {
        let result = PipelineStatisticsResult::new(100, 50, 40, 500, 0);
        let labeled = LabeledStatisticsResult::with_label(0, result, "Main");
        let display = format!("{}", labeled);
        assert!(display.contains("Main"));
    }

    #[test]
    fn labeled_result_display_without_label() {
        let result = PipelineStatisticsResult::new(100, 50, 40, 500, 0);
        let labeled = LabeledStatisticsResult::new(3, result);
        let display = format!("{}", labeled);
        assert!(display.contains("[3]"));
    }

    // ===== SECTION 4: PipelineStatisticsAllocation tests =====

    #[test]
    fn stats_allocation_new() {
        let alloc = PipelineStatisticsAllocation::new(42, 7);
        assert_eq!(alloc.index, 42);
        assert_eq!(alloc.generation, 7);
    }

    #[test]
    fn stats_allocation_with_index() {
        let alloc = PipelineStatisticsAllocation::with_index(99);
        assert_eq!(alloc.index, 99);
        assert_eq!(alloc.generation, 0);
    }

    #[test]
    fn stats_allocation_display() {
        let alloc = PipelineStatisticsAllocation::new(10, 3);
        let display = format!("{}", alloc);
        assert!(display.contains("10"));
        assert!(display.contains("3"));
    }

    #[test]
    fn stats_allocation_equality() {
        let a = PipelineStatisticsAllocation::new(5, 1);
        let b = PipelineStatisticsAllocation::new(5, 1);
        let c = PipelineStatisticsAllocation::new(5, 2);
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    // ===== SECTION 5: PipelineStatisticsResolveParams tests =====

    #[test]
    fn stats_resolve_params_new() {
        let params = PipelineStatisticsResolveParams::new(4, 8, 320);
        assert_eq!(params.start_query, 4);
        assert_eq!(params.query_count, 8);
        assert_eq!(params.destination_offset, 320);
    }

    #[test]
    fn stats_resolve_params_from_start() {
        let params = PipelineStatisticsResolveParams::from_start(16);
        assert_eq!(params.start_query, 0);
        assert_eq!(params.query_count, 16);
        assert_eq!(params.destination_offset, 0);
    }

    #[test]
    fn stats_resolve_params_end_query() {
        let params = PipelineStatisticsResolveParams::new(5, 10, 0);
        assert_eq!(params.end_query(), 15);
    }

    #[test]
    fn stats_resolve_params_required_buffer_size() {
        let params = PipelineStatisticsResolveParams::new(0, 4, 40);
        // 40 offset + 4 queries * 40 bytes = 200
        assert_eq!(params.required_buffer_size(), 200);
    }

    #[test]
    fn stats_resolve_params_default() {
        let params = PipelineStatisticsResolveParams::default();
        assert_eq!(params.start_query, 0);
        assert_eq!(params.query_count, 0);
        assert_eq!(params.destination_offset, 0);
    }

    #[test]
    fn stats_resolve_params_display() {
        let params = PipelineStatisticsResolveParams::new(2, 6, 80);
        let display = format!("{}", params);
        assert!(display.contains("2..8"));
        assert!(display.contains("80"));
    }

    // ===== SECTION 6: PipelineStatisticsData tests =====

    #[test]
    fn stats_data_new() {
        let results = vec![
            PipelineStatisticsResult::new(1, 2, 3, 4, 5),
            PipelineStatisticsResult::new(10, 20, 30, 40, 50),
        ];
        let data = PipelineStatisticsData::new(results, 0);
        assert_eq!(data.len(), 2);
        assert!(!data.is_empty());
    }

    #[test]
    fn stats_data_empty() {
        let data = PipelineStatisticsData::empty();
        assert!(data.is_empty());
        assert_eq!(data.len(), 0);
    }

    #[test]
    fn stats_data_get() {
        let results = vec![
            PipelineStatisticsResult::new(1, 0, 0, 0, 0),
            PipelineStatisticsResult::new(2, 0, 0, 0, 0),
        ];
        let data = PipelineStatisticsData::new(results, 0);
        assert_eq!(data.get(0).unwrap().vertex_shader_invocations, 1);
        assert_eq!(data.get(1).unwrap().vertex_shader_invocations, 2);
        assert!(data.get(2).is_none());
    }

    #[test]
    fn stats_data_get_by_query_index() {
        let results = vec![
            PipelineStatisticsResult::new(1, 0, 0, 0, 0),
            PipelineStatisticsResult::new(2, 0, 0, 0, 0),
        ];
        let data = PipelineStatisticsData::new(results, 10);
        assert_eq!(data.get_by_query_index(10).unwrap().vertex_shader_invocations, 1);
        assert_eq!(data.get_by_query_index(11).unwrap().vertex_shader_invocations, 2);
        assert!(data.get_by_query_index(9).is_none());
        assert!(data.get_by_query_index(12).is_none());
    }

    #[test]
    fn stats_data_aggregate() {
        let results = vec![
            PipelineStatisticsResult::new(10, 20, 15, 100, 5),
            PipelineStatisticsResult::new(5, 10, 5, 50, 10),
        ];
        let data = PipelineStatisticsData::new(results, 0);
        let agg = data.aggregate();
        assert_eq!(agg.vertex_shader_invocations, 15);
        assert_eq!(agg.fragment_shader_invocations, 150);
    }

    #[test]
    fn stats_data_overdraw_estimates() {
        let results = vec![
            PipelineStatisticsResult::new(0, 0, 0, 2000, 0),
            PipelineStatisticsResult::new(0, 0, 0, 4000, 0),
        ];
        let data = PipelineStatisticsData::new(results, 0);
        let estimates = data.overdraw_estimates(1000);
        assert!((estimates[0] - 2.0).abs() < 0.001);
        assert!((estimates[1] - 4.0).abs() < 0.001);
    }

    #[test]
    fn stats_data_culling_efficiencies() {
        let results = vec![
            PipelineStatisticsResult::new(0, 100, 50, 0, 0),  // 50% culled
            PipelineStatisticsResult::new(0, 100, 25, 0, 0),  // 75% culled
        ];
        let data = PipelineStatisticsData::new(results, 0);
        let efficiencies = data.culling_efficiencies();
        assert!((efficiencies[0] - 0.5).abs() < 0.001);
        assert!((efficiencies[1] - 0.75).abs() < 0.001);
    }

    #[test]
    fn stats_data_max_fragment_invocations() {
        let results = vec![
            PipelineStatisticsResult::new(0, 0, 0, 100, 0),
            PipelineStatisticsResult::new(0, 0, 0, 500, 0),
            PipelineStatisticsResult::new(0, 0, 0, 200, 0),
        ];
        let data = PipelineStatisticsData::new(results, 5);
        let (idx, result) = data.max_fragment_invocations().unwrap();
        assert_eq!(idx, 6); // Index 1 relative, start_index 5, so 5+1=6
        assert_eq!(result.fragment_shader_invocations, 500);
    }

    #[test]
    fn stats_data_overdraw_hotspots() {
        let results = vec![
            PipelineStatisticsResult::new(0, 0, 0, 1000, 0),  // 1.0x
            PipelineStatisticsResult::new(0, 0, 0, 3000, 0),  // 3.0x
            PipelineStatisticsResult::new(0, 0, 0, 1500, 0),  // 1.5x
        ];
        let data = PipelineStatisticsData::new(results, 0);
        let hotspots = data.overdraw_hotspots(1000, 2.0);
        assert_eq!(hotspots.len(), 1);
        assert_eq!(hotspots[0].0, 1); // Query index 1
        assert!((hotspots[0].1 - 3.0).abs() < 0.001);
    }

    // ===== SECTION 7: PipelineStatisticsPoolStats tests =====

    #[test]
    fn pool_stats_utilization_empty() {
        let stats = PipelineStatisticsPoolStats {
            capacity: 100,
            used: 0,
            available: 100,
            generation: 0,
            resolve_buffer_size: 4000,
            has_active_query: false,
        };
        assert_eq!(stats.utilization(), 0.0);
        assert!(stats.is_empty());
        assert!(!stats.is_full());
    }

    #[test]
    fn pool_stats_utilization_full() {
        let stats = PipelineStatisticsPoolStats {
            capacity: 100,
            used: 100,
            available: 0,
            generation: 0,
            resolve_buffer_size: 4000,
            has_active_query: false,
        };
        assert_eq!(stats.utilization(), 1.0);
        assert!(!stats.is_empty());
        assert!(stats.is_full());
    }

    #[test]
    fn pool_stats_utilization_half() {
        let stats = PipelineStatisticsPoolStats {
            capacity: 100,
            used: 50,
            available: 50,
            generation: 0,
            resolve_buffer_size: 4000,
            has_active_query: false,
        };
        assert!((stats.utilization() - 0.5).abs() < 0.001);
    }

    #[test]
    fn pool_stats_zero_capacity() {
        let stats = PipelineStatisticsPoolStats {
            capacity: 0,
            used: 0,
            available: 0,
            generation: 0,
            resolve_buffer_size: 0,
            has_active_query: false,
        };
        assert_eq!(stats.utilization(), 0.0);
    }

    #[test]
    fn pool_stats_display() {
        let stats = PipelineStatisticsPoolStats {
            capacity: 64,
            used: 32,
            available: 32,
            generation: 5,
            resolve_buffer_size: 2560,
            has_active_query: true,
        };
        let display = format!("{}", stats);
        assert!(display.contains("32/64"));
        assert!(display.contains("50.0%"));
        assert!(display.contains("gen=5"));
    }

    // ===== SECTION 8: PipelineStatisticsPoolBuilder tests =====

    #[test]
    fn pool_builder_defaults() {
        let builder = PipelineStatisticsPoolBuilder::new();
        assert_eq!(builder.capacity, 64);
        assert!(builder.label.is_none());
    }

    #[test]
    fn pool_builder_capacity() {
        let builder = PipelineStatisticsPoolBuilder::new().capacity(256);
        assert_eq!(builder.capacity, 256);
    }

    #[test]
    fn pool_builder_label() {
        let builder = PipelineStatisticsPoolBuilder::new().label("MyPool");
        assert_eq!(builder.label, Some("MyPool".to_string()));
    }

    #[test]
    fn pool_builder_chaining() {
        let builder = PipelineStatisticsPoolBuilder::new()
            .capacity(128)
            .label("TestPool");
        assert_eq!(builder.capacity, 128);
        assert_eq!(builder.label, Some("TestPool".to_string()));
    }

    #[test]
    fn pool_builder_default_trait() {
        let builder = PipelineStatisticsPoolBuilder::default();
        assert_eq!(builder.capacity, 64);
    }

    // ===== SECTION 9: Utility function tests =====

    #[test]
    fn utility_calculate_buffer_size() {
        assert_eq!(calculate_resolve_buffer_size(1), 40);
        assert_eq!(calculate_resolve_buffer_size(64), 2560);
        assert_eq!(calculate_resolve_buffer_size(128), 5120);
    }

    #[test]
    fn utility_all_statistics_types_has_5_bits() {
        let types = all_statistics_types();
        // Verify all 5 types are present
        assert!(types.contains(PipelineStatisticsTypes::VERTEX_SHADER_INVOCATIONS));
        assert!(types.contains(PipelineStatisticsTypes::CLIPPER_INVOCATIONS));
        assert!(types.contains(PipelineStatisticsTypes::CLIPPER_PRIMITIVES_OUT));
        assert!(types.contains(PipelineStatisticsTypes::FRAGMENT_SHADER_INVOCATIONS));
        assert!(types.contains(PipelineStatisticsTypes::COMPUTE_SHADER_INVOCATIONS));
    }

    // ===== SECTION 10: Constants tests =====

    #[test]
    fn statistic_size_is_8() {
        assert_eq!(STATISTIC_SIZE_BYTES, 8);
    }

    #[test]
    fn statistics_per_query_is_5() {
        assert_eq!(STATISTICS_PER_QUERY, 5);
    }

    #[test]
    fn statistics_result_size_is_40() {
        assert_eq!(STATISTICS_RESULT_SIZE_BYTES, 40);
    }

    #[test]
    fn min_pool_capacity_is_positive() {
        assert!(MIN_POOL_CAPACITY >= 1);
    }

    #[test]
    fn max_recommended_capacity_reasonable() {
        assert!(MAX_RECOMMENDED_CAPACITY >= 64);
        assert!(MAX_RECOMMENDED_CAPACITY <= 16384);
    }

    // ===== SECTION 11: Edge case and boundary tests =====

    #[test]
    fn stats_result_overdraw_large_values() {
        let result = PipelineStatisticsResult::new(0, 0, 0, u64::MAX, 0);
        let overdraw = result.overdraw_estimate(1);
        assert!(overdraw > 0.0);
        assert!(overdraw.is_finite());
    }

    #[test]
    fn stats_result_culling_more_out_than_in() {
        // Edge case: more primitives out than in (shouldn't happen, but handle gracefully)
        let result = PipelineStatisticsResult::new(0, 100, 200, 0, 0);
        let efficiency = result.culling_efficiency();
        // 1.0 - 2.0 = -1.0, but we clamp to 0.0
        assert_eq!(efficiency, 0.0);
    }

    #[test]
    fn stats_data_empty_aggregate() {
        let data = PipelineStatisticsData::empty();
        let agg = data.aggregate();
        assert_eq!(agg, PipelineStatisticsResult::zero());
    }

    #[test]
    fn stats_data_max_fragment_empty() {
        let data = PipelineStatisticsData::empty();
        assert!(data.max_fragment_invocations().is_none());
    }

    #[test]
    fn stats_data_overdraw_hotspots_none_above_threshold() {
        let results = vec![
            PipelineStatisticsResult::new(0, 0, 0, 500, 0),
            PipelineStatisticsResult::new(0, 0, 0, 1000, 0),
        ];
        let data = PipelineStatisticsData::new(results, 0);
        let hotspots = data.overdraw_hotspots(1000, 2.0);
        assert!(hotspots.is_empty());
    }

    #[test]
    fn labeled_result_default() {
        let labeled = LabeledStatisticsResult::default();
        assert_eq!(labeled.query_index, 0);
        assert!(labeled.label.is_none());
        assert_eq!(labeled.result, PipelineStatisticsResult::zero());
    }

    #[test]
    fn resolve_params_zero_query_count_buffer_size() {
        let params = PipelineStatisticsResolveParams::new(0, 0, 0);
        assert_eq!(params.required_buffer_size(), 0);
    }

    #[test]
    fn stats_allocation_hash_consistency() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(PipelineStatisticsAllocation::new(1, 0));
        set.insert(PipelineStatisticsAllocation::new(2, 0));
        set.insert(PipelineStatisticsAllocation::new(1, 0)); // Duplicate
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn resolve_params_equality() {
        let a = PipelineStatisticsResolveParams::new(0, 10, 0);
        let b = PipelineStatisticsResolveParams::new(0, 10, 0);
        let c = PipelineStatisticsResolveParams::new(0, 10, 40);
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn resolve_params_hash_consistency() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(PipelineStatisticsResolveParams::new(0, 10, 0));
        set.insert(PipelineStatisticsResolveParams::new(5, 10, 0));
        set.insert(PipelineStatisticsResolveParams::new(0, 10, 0)); // Duplicate
        assert_eq!(set.len(), 2);
    }

    // ===== SECTION 12: Mixed graphics/compute detection tests =====

    #[test]
    fn stats_result_mixed_pass() {
        // Has both graphics and compute activity
        let result = PipelineStatisticsResult::new(100, 50, 40, 500, 10);
        assert!(result.is_graphics_pass());
        assert!(!result.is_compute_pass()); // Not pure compute
    }

    #[test]
    fn stats_result_only_clipper_activity() {
        // Only clipper, no actual shaders
        let result = PipelineStatisticsResult::new(0, 100, 80, 0, 0);
        assert!(!result.has_activity());
        assert!(!result.is_graphics_pass());
        assert!(!result.is_compute_pass());
    }
}
