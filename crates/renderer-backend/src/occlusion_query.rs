//! Occlusion query pool for GPU visibility testing.
//!
//! This module provides a reusable pool for wgpu occlusion queries, enabling
//! GPU-side visibility testing for objects. It supports both binary (visible/not visible)
//! and sample count modes.
//!
//! # Overview
//!
//! Occlusion queries allow testing if rendered geometry passes the depth test.
//! The `OcclusionQueryPool` manages:
//!
//! - **Feature detection**: Checks if `PIPELINE_STATISTICS_QUERY` is supported (for sample counts)
//! - **QuerySet creation**: Creates a `QuerySet` with `QueryType::Occlusion`
//! - **Index allocation**: Tracks and allocates query indices
//! - **Resolve buffer**: Buffer for reading back occlusion results (u64 per query)
//! - **Binary/SampleCount modes**: Support for both visibility and precise sample counting
//!
//! # Usage
//!
//! ```no_run
//! use renderer_backend::occlusion_query::{OcclusionQueryPool, OcclusionMode};
//!
//! # fn example(device: &wgpu::Device, queue: &wgpu::Queue) {
//! // Create a pool with capacity for 64 occlusion queries in binary mode
//! let mut pool = OcclusionQueryPool::new(device, 64, OcclusionMode::Binary)
//!     .expect("Failed to create occlusion query pool");
//!
//! // Allocate an index for a query
//! let query_idx = pool.allocate().expect("Pool exhausted");
//!
//! // In render pass:
//! // render_pass.begin_occlusion_query(query_idx);
//! // ... draw calls ...
//! // render_pass.end_occlusion_query();
//!
//! // Reset pool for next frame
//! pool.reset();
//! # }
//! ```
//!
//! # Occlusion Query Modes
//!
//! - **Binary**: Returns 0 (not visible) or non-zero (visible). More performant.
//! - **SampleCount**: Returns the exact number of samples that passed the depth test.
//!
//! # wgpu Compatibility
//!
//! This implementation targets wgpu 22+ and follows these patterns:
//! - `device.create_query_set()` with `QuerySetDescriptor`
//! - `QueryType::Occlusion` for occlusion queries
//! - `render_pass.begin_occlusion_query(index)` / `render_pass.end_occlusion_query()`
//! - `encoder.resolve_query_set()` to copy results to buffer

use std::fmt;
use std::ops::Range;
use wgpu::{Buffer, BufferDescriptor, BufferUsages, CommandEncoder, Device, MapMode, Maintain, QuerySet, QuerySetDescriptor, QueryType};

// Re-export QueryError from query_pool for consistent error handling
pub use crate::query_pool::{QueryError, QueryErrorKind};

// ============================================================================
// Constants
// ============================================================================

/// Size of a single occlusion query result in bytes (u64).
pub const OCCLUSION_RESULT_SIZE_BYTES: u64 = 8;

/// Minimum capacity for an occlusion query pool.
pub const MIN_POOL_CAPACITY: u32 = 1;

/// Maximum recommended capacity for an occlusion query pool.
/// Larger pools use more memory but allow more concurrent visibility tests.
pub const MAX_RECOMMENDED_CAPACITY: u32 = 4096;

/// Default label prefix for occlusion query pool resources.
pub const DEFAULT_LABEL_PREFIX: &str = "OcclusionQueryPool";

/// Threshold for considering an object visible in binary mode.
/// Any sample count > 0 is considered visible.
pub const VISIBILITY_THRESHOLD: u64 = 0;

// ============================================================================
// OcclusionMode
// ============================================================================

/// Mode for occlusion queries.
///
/// Determines how occlusion query results are interpreted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum OcclusionMode {
    /// Binary occlusion testing: returns 0 (not visible) or non-zero (visible).
    ///
    /// This mode is more performant on some hardware as the GPU can early-out
    /// once any sample passes the depth test.
    #[default]
    Binary,

    /// Precise sample counting: returns the exact number of samples that passed.
    ///
    /// Use this mode when you need to know how much of an object is visible,
    /// such as for level-of-detail selection or partial occlusion effects.
    SampleCount,
}

impl OcclusionMode {
    /// Check if this mode is binary.
    #[inline]
    pub fn is_binary(&self) -> bool {
        matches!(self, OcclusionMode::Binary)
    }

    /// Check if this mode returns sample counts.
    #[inline]
    pub fn is_sample_count(&self) -> bool {
        matches!(self, OcclusionMode::SampleCount)
    }
}

impl fmt::Display for OcclusionMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OcclusionMode::Binary => write!(f, "Binary"),
            OcclusionMode::SampleCount => write!(f, "SampleCount"),
        }
    }
}

// ============================================================================
// OcclusionResult
// ============================================================================

/// Result of a single occlusion query.
///
/// Contains the query index, raw sample count, and visibility status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct OcclusionResult {
    /// The query index this result corresponds to.
    pub query_index: u32,
    /// Number of samples that passed the depth test.
    /// In binary mode, this is 0 or non-zero.
    /// In sample count mode, this is the exact count.
    pub sample_count: u64,
    /// Whether the queried geometry is considered visible.
    /// True if sample_count > 0.
    pub is_visible: bool,
}

impl OcclusionResult {
    /// Create a new occlusion result.
    ///
    /// # Arguments
    ///
    /// * `query_index` - The query slot index
    /// * `sample_count` - Number of samples that passed depth test
    #[inline]
    pub fn new(query_index: u32, sample_count: u64) -> Self {
        Self {
            query_index,
            sample_count,
            is_visible: sample_count > VISIBILITY_THRESHOLD,
        }
    }

    /// Create an invisible result (sample_count = 0).
    #[inline]
    pub fn invisible(query_index: u32) -> Self {
        Self {
            query_index,
            sample_count: 0,
            is_visible: false,
        }
    }

    /// Create a visible result with binary mode (sample_count = 1).
    #[inline]
    pub fn visible_binary(query_index: u32) -> Self {
        Self {
            query_index,
            sample_count: 1,
            is_visible: true,
        }
    }

    /// Get the visibility ratio compared to a maximum expected sample count.
    ///
    /// Useful for partial occlusion calculations.
    ///
    /// # Arguments
    ///
    /// * `max_samples` - Maximum expected sample count (e.g., total fragment count)
    ///
    /// # Returns
    ///
    /// Visibility ratio from 0.0 (fully occluded) to 1.0 (fully visible).
    #[inline]
    pub fn visibility_ratio(&self, max_samples: u64) -> f32 {
        if max_samples == 0 {
            return 0.0;
        }
        ((self.sample_count as f32) / (max_samples as f32)).min(1.0)
    }
}

impl Default for OcclusionResult {
    fn default() -> Self {
        Self::invisible(0)
    }
}

impl fmt::Display for OcclusionResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Occlusion[{}]: {} samples, {}",
            self.query_index,
            self.sample_count,
            if self.is_visible { "visible" } else { "occluded" }
        )
    }
}

// ============================================================================
// OcclusionQueryAllocation
// ============================================================================

/// Represents an allocated occlusion query index with metadata.
///
/// Similar to `QueryAllocation` but specific to occlusion queries,
/// with additional fields for tracking associated objects.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct OcclusionQueryAllocation {
    /// The allocated query index.
    pub index: u32,
    /// Generation counter for validation.
    pub generation: u32,
    /// Optional entity or object ID associated with this query.
    pub entity_id: Option<u32>,
}

impl OcclusionQueryAllocation {
    /// Create a new occlusion query allocation.
    #[inline]
    pub const fn new(index: u32, generation: u32) -> Self {
        Self {
            index,
            generation,
            entity_id: None,
        }
    }

    /// Create an allocation with an associated entity ID.
    #[inline]
    pub const fn with_entity(index: u32, generation: u32, entity_id: u32) -> Self {
        Self {
            index,
            generation,
            entity_id: Some(entity_id),
        }
    }

    /// Create an allocation with just an index (default generation).
    #[inline]
    pub const fn with_index(index: u32) -> Self {
        Self {
            index,
            generation: 0,
            entity_id: None,
        }
    }
}

impl fmt::Display for OcclusionQueryAllocation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.entity_id {
            Some(id) => write!(f, "OcclusionQuery(index={}, gen={}, entity={})",
                self.index, self.generation, id),
            None => write!(f, "OcclusionQuery(index={}, gen={})",
                self.index, self.generation),
        }
    }
}

// ============================================================================
// OcclusionResolveParams
// ============================================================================

/// Parameters for resolving occlusion queries to a buffer.
///
/// Specifies which queries to resolve and where to write results.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct OcclusionResolveParams {
    /// Starting query index in the QuerySet.
    pub start_query: u32,
    /// Number of queries to resolve.
    pub query_count: u32,
    /// Byte offset in the destination resolve buffer.
    pub destination_offset: u64,
}

impl OcclusionResolveParams {
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
        self.destination_offset + (self.query_count as u64) * OCCLUSION_RESULT_SIZE_BYTES
    }
}

impl Default for OcclusionResolveParams {
    fn default() -> Self {
        Self {
            start_query: 0,
            query_count: 0,
            destination_offset: 0,
        }
    }
}

impl fmt::Display for OcclusionResolveParams {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "OcclusionResolveParams(queries={}..{}, offset={})",
            self.start_query,
            self.end_query(),
            self.destination_offset
        )
    }
}

// ============================================================================
// OcclusionData
// ============================================================================

/// Raw occlusion data from readback.
///
/// Contains the raw u64 sample counts read from the resolve buffer.
#[derive(Debug, Clone)]
pub struct OcclusionData {
    /// Raw sample count values.
    pub sample_counts: Vec<u64>,
    /// The occlusion mode used.
    pub mode: OcclusionMode,
    /// Starting query index for this data.
    pub start_index: u32,
}

impl OcclusionData {
    /// Create new occlusion data.
    #[inline]
    pub fn new(sample_counts: Vec<u64>, mode: OcclusionMode, start_index: u32) -> Self {
        Self {
            sample_counts,
            mode,
            start_index,
        }
    }

    /// Create empty occlusion data.
    #[inline]
    pub fn empty(mode: OcclusionMode) -> Self {
        Self {
            sample_counts: Vec::new(),
            mode,
            start_index: 0,
        }
    }

    /// Get the number of query results.
    #[inline]
    pub fn len(&self) -> usize {
        self.sample_counts.len()
    }

    /// Check if empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.sample_counts.is_empty()
    }

    /// Get sample count at relative index.
    #[inline]
    pub fn get(&self, relative_index: usize) -> Option<u64> {
        self.sample_counts.get(relative_index).copied()
    }

    /// Check if a specific query index is visible.
    #[inline]
    pub fn is_visible(&self, relative_index: usize) -> Option<bool> {
        self.get(relative_index).map(|count| count > VISIBILITY_THRESHOLD)
    }

    /// Convert to OcclusionResult instances.
    pub fn to_results(&self) -> Vec<OcclusionResult> {
        self.sample_counts
            .iter()
            .enumerate()
            .map(|(i, &count)| {
                OcclusionResult::new(self.start_index + i as u32, count)
            })
            .collect()
    }

    /// Get all visible query indices (absolute indices).
    pub fn visible_indices(&self) -> Vec<u32> {
        self.sample_counts
            .iter()
            .enumerate()
            .filter(|(_, &count)| count > VISIBILITY_THRESHOLD)
            .map(|(i, _)| self.start_index + i as u32)
            .collect()
    }

    /// Get all occluded query indices (absolute indices).
    pub fn occluded_indices(&self) -> Vec<u32> {
        self.sample_counts
            .iter()
            .enumerate()
            .filter(|(_, &count)| count == 0)
            .map(|(i, _)| self.start_index + i as u32)
            .collect()
    }

    /// Count visible queries.
    #[inline]
    pub fn visible_count(&self) -> usize {
        self.sample_counts.iter().filter(|&&c| c > VISIBILITY_THRESHOLD).count()
    }

    /// Count occluded queries.
    #[inline]
    pub fn occluded_count(&self) -> usize {
        self.sample_counts.iter().filter(|&&c| c == 0).count()
    }
}

impl Default for OcclusionData {
    fn default() -> Self {
        Self::empty(OcclusionMode::Binary)
    }
}

// ============================================================================
// OcclusionQueryPool
// ============================================================================

/// A pool for managing wgpu occlusion queries.
///
/// Provides GPU-side visibility testing with support for both binary
/// (visible/not visible) and precise sample counting modes.
///
/// # Example
///
/// ```no_run
/// use renderer_backend::occlusion_query::{OcclusionQueryPool, OcclusionMode};
///
/// # fn example(device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder) {
/// // Create pool with 128 query capacity in binary mode
/// let mut pool = OcclusionQueryPool::new(device, 128, OcclusionMode::Binary).unwrap();
///
/// // Allocate queries for objects
/// let obj1_query = pool.allocate().unwrap();
/// let obj2_query = pool.allocate().unwrap();
///
/// // In render pass:
/// // render_pass.begin_occlusion_query(obj1_query);
/// // draw_object_1();
/// // render_pass.end_occlusion_query();
/// // render_pass.begin_occlusion_query(obj2_query);
/// // draw_object_2();
/// // render_pass.end_occlusion_query();
///
/// // Resolve and read results for culling decisions
/// pool.resolve_all(encoder).unwrap();
///
/// // Reset for next frame
/// pool.reset();
/// # }
/// ```
pub struct OcclusionQueryPool {
    /// The wgpu QuerySet for occlusion queries.
    query_set: QuerySet,
    /// Buffer for resolving query results.
    resolve_buffer: Buffer,
    /// Maximum number of queries this pool can hold.
    capacity: u32,
    /// Next index to allocate (watermark allocator).
    next_index: u32,
    /// The occlusion mode (binary or sample count).
    mode: OcclusionMode,
    /// Generation counter for allocation tracking.
    generation: u32,
    /// Optional label for debugging.
    label: Option<String>,
    /// Tracks which queries are currently active (between begin/end).
    active_query: Option<u32>,
}

impl OcclusionQueryPool {
    /// Create a new occlusion query pool.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device to create resources on
    /// * `capacity` - Maximum number of occlusion queries
    /// * `mode` - Occlusion query mode (Binary or SampleCount)
    ///
    /// # Returns
    ///
    /// * `Ok(OcclusionQueryPool)` - Successfully created pool
    /// * `Err(QueryError::InvalidCapacity)` - Capacity is 0
    ///
    /// # Example
    ///
    /// ```no_run
    /// use renderer_backend::occlusion_query::{OcclusionQueryPool, OcclusionMode};
    ///
    /// # fn example(device: &wgpu::Device) {
    /// let pool = OcclusionQueryPool::new(device, 64, OcclusionMode::Binary)?;
    /// println!("Created pool with {} capacity", pool.capacity());
    /// # Ok::<(), renderer_backend::query_pool::QueryError>(())
    /// # }
    /// ```
    pub fn new(device: &Device, capacity: u32, mode: OcclusionMode) -> Result<Self, QueryError> {
        Self::with_label(device, capacity, mode, None)
    }

    /// Create a new occlusion query pool with a custom label.
    ///
    /// The label is used for debugging in graphics debuggers and profilers.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device
    /// * `capacity` - Maximum number of queries
    /// * `mode` - Occlusion query mode
    /// * `label` - Optional debug label
    pub fn with_label(
        device: &Device,
        capacity: u32,
        mode: OcclusionMode,
        label: Option<&str>,
    ) -> Result<Self, QueryError> {
        // Validate capacity
        if capacity < MIN_POOL_CAPACITY {
            return Err(QueryError::invalid_capacity(capacity));
        }

        // Create the QuerySet with QueryType::Occlusion
        let query_set_label = label
            .map(|l| format!("{} QuerySet", l))
            .unwrap_or_else(|| format!("{} QuerySet", DEFAULT_LABEL_PREFIX));

        let query_set = device.create_query_set(&QuerySetDescriptor {
            label: Some(&query_set_label),
            ty: QueryType::Occlusion,
            count: capacity,
        });

        // Create resolve buffer (u64 per query = 8 bytes)
        let buffer_size = (capacity as u64) * OCCLUSION_RESULT_SIZE_BYTES;
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
            mode,
            generation: 0,
            label: label.map(String::from),
            active_query: None,
        })
    }

    /// Allocate the next available query index.
    ///
    /// Uses a watermark allocator for efficient sequential allocation.
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
    pub fn allocate_tracked(&mut self) -> Result<OcclusionQueryAllocation, QueryError> {
        let index = self.allocate()?;
        Ok(OcclusionQueryAllocation::new(index, self.generation))
    }

    /// Allocate a query for a specific entity.
    pub fn allocate_for_entity(&mut self, entity_id: u32) -> Result<OcclusionQueryAllocation, QueryError> {
        let index = self.allocate()?;
        Ok(OcclusionQueryAllocation::with_entity(index, self.generation, entity_id))
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
    ///
    /// Resets the allocation watermark and increments the generation counter.
    #[inline]
    pub fn reset(&mut self) {
        self.next_index = 0;
        self.generation = self.generation.wrapping_add(1);
        self.active_query = None;
    }

    // ========================================================================
    // Render Pass Query Operations
    // ========================================================================

    /// Begin an occlusion query at the specified index.
    ///
    /// This should be called on a render pass before drawing the geometry
    /// you want to test for visibility.
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
    ///
    /// # Example
    ///
    /// ```no_run
    /// use renderer_backend::occlusion_query::OcclusionQueryPool;
    ///
    /// # fn example(pool: &mut OcclusionQueryPool, render_pass: &mut wgpu::RenderPass) {
    /// let query_idx = pool.allocate()?;
    /// pool.begin_query(render_pass, query_idx)?;
    /// // ... draw calls for the object being tested ...
    /// pool.end_query(render_pass)?;
    /// # Ok::<(), renderer_backend::query_pool::QueryError>(())
    /// # }
    /// ```
    pub fn begin_query<'a>(&mut self, render_pass: &mut wgpu::RenderPass<'a>, index: u32) -> Result<(), QueryError> {
        // Validate index
        if index >= self.capacity {
            return Err(QueryError::invalid_index(index, self.capacity));
        }

        // Check if another query is active
        if self.active_query.is_some() {
            return Err(QueryError::resolve_failed(
                "another occlusion query is already active; call end_query first"
            ));
        }

        // Begin the occlusion query
        render_pass.begin_occlusion_query(index);
        self.active_query = Some(index);

        Ok(())
    }

    /// End the current occlusion query.
    ///
    /// Must be called after `begin_query` and before starting another query.
    ///
    /// # Arguments
    ///
    /// * `render_pass` - The active render pass
    ///
    /// # Returns
    ///
    /// * `Ok(())` - Query ended successfully
    /// * `Err(QueryError::ResolveFailed)` - No query is currently active
    pub fn end_query<'a>(&mut self, render_pass: &mut wgpu::RenderPass<'a>) -> Result<(), QueryError> {
        // Check if a query is active
        if self.active_query.is_none() {
            return Err(QueryError::resolve_failed(
                "no occlusion query is active; call begin_query first"
            ));
        }

        render_pass.end_occlusion_query();
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
    ///
    /// # Example
    ///
    /// ```no_run
    /// use renderer_backend::occlusion_query::OcclusionQueryPool;
    ///
    /// # fn example(pool: &mut OcclusionQueryPool, render_pass: &mut wgpu::RenderPass) {
    /// let query_idx = pool.allocate()?;
    /// pool.with_query(render_pass, query_idx, |pass| {
    ///     // Draw the bounding box or object
    ///     // pass.draw(0..36, 0..1);
    /// })?;
    /// # Ok::<(), renderer_backend::query_pool::QueryError>(())
    /// # }
    /// ```
    pub fn with_query<'a, F>(
        &mut self,
        render_pass: &mut wgpu::RenderPass<'a>,
        index: u32,
        draw_fn: F,
    ) -> Result<(), QueryError>
    where
        F: FnOnce(&mut wgpu::RenderPass<'a>),
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
    /// This copies occlusion query results from the QuerySet to the resolve
    /// buffer for CPU readback.
    ///
    /// # Timing Requirements
    ///
    /// Call **after** all render passes that use occlusion queries complete,
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

        let offset = (range.start as u64) * OCCLUSION_RESULT_SIZE_BYTES;

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
        params: OcclusionResolveParams,
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

    /// Read resolved occlusion query results.
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
    /// * `Ok(OcclusionData)` - Successfully read results
    /// * `Err(QueryError)` - Read failed
    pub fn read_results(
        &self,
        device: &Device,
        query_range: Range<u32>,
    ) -> Result<OcclusionData, QueryError> {
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
        let offset = (query_range.start as u64) * OCCLUSION_RESULT_SIZE_BYTES;
        let size = ((query_range.end - query_range.start) as u64) * OCCLUSION_RESULT_SIZE_BYTES;

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

        // Read the data
        let mapped_range = slice.get_mapped_range();
        let query_count = (query_range.end - query_range.start) as usize;
        let mut sample_counts = Vec::with_capacity(query_count);

        for i in 0..query_count {
            let byte_offset = i * 8;
            if byte_offset + 8 <= mapped_range.len() {
                let bytes: [u8; 8] = mapped_range[byte_offset..byte_offset + 8]
                    .try_into()
                    .map_err(|_| QueryError::buffer_mapping_failed("failed to read u64 from buffer"))?;
                sample_counts.push(u64::from_le_bytes(bytes));
            }
        }

        drop(mapped_range);
        self.resolve_buffer.unmap();

        Ok(OcclusionData::new(sample_counts, self.mode, query_range.start))
    }

    /// Read all used query results.
    pub fn read_all_results(&self, device: &Device) -> Result<OcclusionData, QueryError> {
        let used = self.used();
        if used == 0 {
            return Ok(OcclusionData::empty(self.mode));
        }

        self.read_results(device, 0..used)
    }

    /// Check if a specific query index is visible.
    ///
    /// Convenience method that reads and checks a single result.
    ///
    /// # Arguments
    ///
    /// * `results` - Pre-read occlusion results
    /// * `index` - Query index to check
    ///
    /// # Returns
    ///
    /// `true` if the query passed any samples, `false` if fully occluded.
    #[inline]
    pub fn is_visible(&self, results: &[OcclusionResult], index: u32) -> bool {
        results
            .iter()
            .find(|r| r.query_index == index)
            .map(|r| r.is_visible)
            .unwrap_or(false)
    }

    /// Get visibility status for multiple indices.
    pub fn visibility_map(&self, results: &[OcclusionResult]) -> std::collections::HashMap<u32, bool> {
        results
            .iter()
            .map(|r| (r.query_index, r.is_visible))
            .collect()
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

    /// Get the occlusion mode.
    #[inline]
    pub fn mode(&self) -> OcclusionMode {
        self.mode
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

    /// Get the resolve buffer size in bytes.
    #[inline]
    pub fn resolve_buffer_size(&self) -> u64 {
        (self.capacity as u64) * OCCLUSION_RESULT_SIZE_BYTES
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
    pub fn stats(&self) -> OcclusionPoolStats {
        OcclusionPoolStats {
            capacity: self.capacity,
            used: self.next_index,
            available: self.available(),
            generation: self.generation,
            mode: self.mode,
            resolve_buffer_size: self.resolve_buffer_size(),
            has_active_query: self.active_query.is_some(),
        }
    }
}

impl fmt::Debug for OcclusionQueryPool {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OcclusionQueryPool")
            .field("capacity", &self.capacity)
            .field("used", &self.next_index)
            .field("available", &self.available())
            .field("mode", &self.mode)
            .field("generation", &self.generation)
            .field("label", &self.label)
            .field("active_query", &self.active_query)
            .finish()
    }
}

// ============================================================================
// OcclusionPoolStats
// ============================================================================

/// Statistics about an occlusion query pool's current state.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OcclusionPoolStats {
    /// Total capacity of the pool.
    pub capacity: u32,
    /// Number of allocated indices.
    pub used: u32,
    /// Number of available indices.
    pub available: u32,
    /// Current generation counter.
    pub generation: u32,
    /// Occlusion mode.
    pub mode: OcclusionMode,
    /// Resolve buffer size in bytes.
    pub resolve_buffer_size: u64,
    /// Whether a query is currently active.
    pub has_active_query: bool,
}

impl OcclusionPoolStats {
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

impl fmt::Display for OcclusionPoolStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "OcclusionPool: {}/{} used ({:.1}%), mode={}, gen={}",
            self.used,
            self.capacity,
            self.utilization() * 100.0,
            self.mode,
            self.generation
        )
    }
}

// ============================================================================
// OcclusionQueryPoolBuilder
// ============================================================================

/// Builder for creating OcclusionQueryPool with custom options.
///
/// # Example
///
/// ```no_run
/// use renderer_backend::occlusion_query::{OcclusionQueryPoolBuilder, OcclusionMode};
///
/// # fn example(device: &wgpu::Device) {
/// let pool = OcclusionQueryPoolBuilder::new()
///     .capacity(256)
///     .mode(OcclusionMode::SampleCount)
///     .label("ObjectVisibility")
///     .build(device)?;
/// # Ok::<(), renderer_backend::query_pool::QueryError>(())
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct OcclusionQueryPoolBuilder {
    capacity: u32,
    mode: OcclusionMode,
    label: Option<String>,
}

impl Default for OcclusionQueryPoolBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl OcclusionQueryPoolBuilder {
    /// Create a new builder with default settings.
    pub fn new() -> Self {
        Self {
            capacity: 64,
            mode: OcclusionMode::Binary,
            label: None,
        }
    }

    /// Set the pool capacity.
    pub fn capacity(mut self, capacity: u32) -> Self {
        self.capacity = capacity;
        self
    }

    /// Set the occlusion mode.
    pub fn mode(mut self, mode: OcclusionMode) -> Self {
        self.mode = mode;
        self
    }

    /// Set a debug label for the pool.
    pub fn label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Build the OcclusionQueryPool.
    pub fn build(self, device: &Device) -> Result<OcclusionQueryPool, QueryError> {
        OcclusionQueryPool::with_label(device, self.capacity, self.mode, self.label.as_deref())
    }
}

// ============================================================================
// Utility Functions
// ============================================================================

/// Calculate the required resolve buffer size for a given capacity.
#[inline]
pub const fn calculate_resolve_buffer_size(capacity: u32) -> u64 {
    (capacity as u64) * OCCLUSION_RESULT_SIZE_BYTES
}

/// Check if a sample count indicates visibility.
#[inline]
pub const fn is_sample_count_visible(sample_count: u64) -> bool {
    sample_count > VISIBILITY_THRESHOLD
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ===== SECTION 1: OcclusionMode tests =====

    #[test]
    fn occlusion_mode_binary_is_default() {
        let mode = OcclusionMode::default();
        assert!(mode.is_binary());
        assert!(!mode.is_sample_count());
    }

    #[test]
    fn occlusion_mode_sample_count_detection() {
        let mode = OcclusionMode::SampleCount;
        assert!(!mode.is_binary());
        assert!(mode.is_sample_count());
    }

    #[test]
    fn occlusion_mode_display_binary() {
        let mode = OcclusionMode::Binary;
        assert_eq!(format!("{}", mode), "Binary");
    }

    #[test]
    fn occlusion_mode_display_sample_count() {
        let mode = OcclusionMode::SampleCount;
        assert_eq!(format!("{}", mode), "SampleCount");
    }

    #[test]
    fn occlusion_mode_equality() {
        assert_eq!(OcclusionMode::Binary, OcclusionMode::Binary);
        assert_eq!(OcclusionMode::SampleCount, OcclusionMode::SampleCount);
        assert_ne!(OcclusionMode::Binary, OcclusionMode::SampleCount);
    }

    #[test]
    fn occlusion_mode_clone() {
        let mode = OcclusionMode::SampleCount;
        let cloned = mode;
        assert_eq!(mode, cloned);
    }

    #[test]
    fn occlusion_mode_hash_consistency() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(OcclusionMode::Binary);
        set.insert(OcclusionMode::SampleCount);
        set.insert(OcclusionMode::Binary); // Duplicate
        assert_eq!(set.len(), 2);
    }

    // ===== SECTION 2: OcclusionResult tests =====

    #[test]
    fn occlusion_result_new_visible() {
        let result = OcclusionResult::new(5, 100);
        assert_eq!(result.query_index, 5);
        assert_eq!(result.sample_count, 100);
        assert!(result.is_visible);
    }

    #[test]
    fn occlusion_result_new_invisible() {
        let result = OcclusionResult::new(3, 0);
        assert_eq!(result.query_index, 3);
        assert_eq!(result.sample_count, 0);
        assert!(!result.is_visible);
    }

    #[test]
    fn occlusion_result_invisible_constructor() {
        let result = OcclusionResult::invisible(10);
        assert_eq!(result.query_index, 10);
        assert_eq!(result.sample_count, 0);
        assert!(!result.is_visible);
    }

    #[test]
    fn occlusion_result_visible_binary_constructor() {
        let result = OcclusionResult::visible_binary(7);
        assert_eq!(result.query_index, 7);
        assert_eq!(result.sample_count, 1);
        assert!(result.is_visible);
    }

    #[test]
    fn occlusion_result_visibility_ratio() {
        let result = OcclusionResult::new(0, 50);
        let ratio = result.visibility_ratio(100);
        assert!((ratio - 0.5).abs() < 0.001);
    }

    #[test]
    fn occlusion_result_visibility_ratio_zero_max() {
        let result = OcclusionResult::new(0, 50);
        let ratio = result.visibility_ratio(0);
        assert_eq!(ratio, 0.0);
    }

    #[test]
    fn occlusion_result_display() {
        let result = OcclusionResult::new(2, 150);
        let display = format!("{}", result);
        assert!(display.contains("2"));
        assert!(display.contains("150"));
        assert!(display.contains("visible"));
    }

    #[test]
    fn occlusion_result_display_occluded() {
        let result = OcclusionResult::invisible(5);
        let display = format!("{}", result);
        assert!(display.contains("occluded"));
    }

    #[test]
    fn occlusion_result_default() {
        let result = OcclusionResult::default();
        assert_eq!(result.query_index, 0);
        assert_eq!(result.sample_count, 0);
        assert!(!result.is_visible);
    }

    #[test]
    fn occlusion_result_equality() {
        let a = OcclusionResult::new(1, 100);
        let b = OcclusionResult::new(1, 100);
        let c = OcclusionResult::new(2, 100);
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    // ===== SECTION 3: OcclusionQueryAllocation tests =====

    #[test]
    fn occlusion_allocation_new() {
        let alloc = OcclusionQueryAllocation::new(5, 3);
        assert_eq!(alloc.index, 5);
        assert_eq!(alloc.generation, 3);
        assert_eq!(alloc.entity_id, None);
    }

    #[test]
    fn occlusion_allocation_with_entity() {
        let alloc = OcclusionQueryAllocation::with_entity(10, 2, 42);
        assert_eq!(alloc.index, 10);
        assert_eq!(alloc.generation, 2);
        assert_eq!(alloc.entity_id, Some(42));
    }

    #[test]
    fn occlusion_allocation_with_index() {
        let alloc = OcclusionQueryAllocation::with_index(7);
        assert_eq!(alloc.index, 7);
        assert_eq!(alloc.generation, 0);
        assert_eq!(alloc.entity_id, None);
    }

    #[test]
    fn occlusion_allocation_display_without_entity() {
        let alloc = OcclusionQueryAllocation::new(5, 3);
        let display = format!("{}", alloc);
        assert!(display.contains("5"));
        assert!(display.contains("3"));
        assert!(!display.contains("entity"));
    }

    #[test]
    fn occlusion_allocation_display_with_entity() {
        let alloc = OcclusionQueryAllocation::with_entity(5, 3, 99);
        let display = format!("{}", alloc);
        assert!(display.contains("99"));
        assert!(display.contains("entity"));
    }

    // ===== SECTION 4: OcclusionResolveParams tests =====

    #[test]
    fn occlusion_resolve_params_new() {
        let params = OcclusionResolveParams::new(4, 8, 32);
        assert_eq!(params.start_query, 4);
        assert_eq!(params.query_count, 8);
        assert_eq!(params.destination_offset, 32);
    }

    #[test]
    fn occlusion_resolve_params_from_start() {
        let params = OcclusionResolveParams::from_start(16);
        assert_eq!(params.start_query, 0);
        assert_eq!(params.query_count, 16);
        assert_eq!(params.destination_offset, 0);
    }

    #[test]
    fn occlusion_resolve_params_end_query() {
        let params = OcclusionResolveParams::new(5, 10, 0);
        assert_eq!(params.end_query(), 15);
    }

    #[test]
    fn occlusion_resolve_params_required_buffer_size() {
        let params = OcclusionResolveParams::new(0, 4, 8);
        // 8 offset + 4 queries * 8 bytes = 40
        assert_eq!(params.required_buffer_size(), 40);
    }

    #[test]
    fn occlusion_resolve_params_default() {
        let params = OcclusionResolveParams::default();
        assert_eq!(params.start_query, 0);
        assert_eq!(params.query_count, 0);
        assert_eq!(params.destination_offset, 0);
    }

    #[test]
    fn occlusion_resolve_params_display() {
        let params = OcclusionResolveParams::new(2, 6, 16);
        let display = format!("{}", params);
        assert!(display.contains("2..8"));
        assert!(display.contains("16"));
    }

    // ===== SECTION 5: OcclusionData tests =====

    #[test]
    fn occlusion_data_new() {
        let data = OcclusionData::new(vec![0, 100, 50, 0], OcclusionMode::SampleCount, 0);
        assert_eq!(data.len(), 4);
        assert!(!data.is_empty());
        assert_eq!(data.mode, OcclusionMode::SampleCount);
    }

    #[test]
    fn occlusion_data_empty() {
        let data = OcclusionData::empty(OcclusionMode::Binary);
        assert!(data.is_empty());
        assert_eq!(data.len(), 0);
    }

    #[test]
    fn occlusion_data_get() {
        let data = OcclusionData::new(vec![10, 20, 30], OcclusionMode::Binary, 0);
        assert_eq!(data.get(0), Some(10));
        assert_eq!(data.get(1), Some(20));
        assert_eq!(data.get(2), Some(30));
        assert_eq!(data.get(3), None);
    }

    #[test]
    fn occlusion_data_is_visible() {
        let data = OcclusionData::new(vec![0, 100, 0, 50], OcclusionMode::Binary, 0);
        assert_eq!(data.is_visible(0), Some(false));
        assert_eq!(data.is_visible(1), Some(true));
        assert_eq!(data.is_visible(2), Some(false));
        assert_eq!(data.is_visible(3), Some(true));
        assert_eq!(data.is_visible(4), None);
    }

    #[test]
    fn occlusion_data_to_results() {
        let data = OcclusionData::new(vec![0, 100], OcclusionMode::Binary, 5);
        let results = data.to_results();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].query_index, 5);
        assert!(!results[0].is_visible);
        assert_eq!(results[1].query_index, 6);
        assert!(results[1].is_visible);
    }

    #[test]
    fn occlusion_data_visible_indices() {
        let data = OcclusionData::new(vec![0, 100, 0, 50], OcclusionMode::Binary, 10);
        let visible = data.visible_indices();
        assert_eq!(visible, vec![11, 13]); // indices 1 and 3 are visible, starting from 10
    }

    #[test]
    fn occlusion_data_occluded_indices() {
        let data = OcclusionData::new(vec![0, 100, 0, 50], OcclusionMode::Binary, 10);
        let occluded = data.occluded_indices();
        assert_eq!(occluded, vec![10, 12]); // indices 0 and 2 are occluded
    }

    #[test]
    fn occlusion_data_visible_count() {
        let data = OcclusionData::new(vec![0, 100, 0, 50, 25], OcclusionMode::Binary, 0);
        assert_eq!(data.visible_count(), 3);
    }

    #[test]
    fn occlusion_data_occluded_count() {
        let data = OcclusionData::new(vec![0, 100, 0, 50, 0], OcclusionMode::Binary, 0);
        assert_eq!(data.occluded_count(), 3);
    }

    #[test]
    fn occlusion_data_default() {
        let data = OcclusionData::default();
        assert!(data.is_empty());
        assert_eq!(data.mode, OcclusionMode::Binary);
    }

    // ===== SECTION 6: OcclusionPoolStats tests =====

    #[test]
    fn occlusion_pool_stats_utilization_empty() {
        let stats = OcclusionPoolStats {
            capacity: 100,
            used: 0,
            available: 100,
            generation: 0,
            mode: OcclusionMode::Binary,
            resolve_buffer_size: 800,
            has_active_query: false,
        };
        assert_eq!(stats.utilization(), 0.0);
        assert!(stats.is_empty());
        assert!(!stats.is_full());
    }

    #[test]
    fn occlusion_pool_stats_utilization_full() {
        let stats = OcclusionPoolStats {
            capacity: 100,
            used: 100,
            available: 0,
            generation: 0,
            mode: OcclusionMode::Binary,
            resolve_buffer_size: 800,
            has_active_query: false,
        };
        assert_eq!(stats.utilization(), 1.0);
        assert!(!stats.is_empty());
        assert!(stats.is_full());
    }

    #[test]
    fn occlusion_pool_stats_utilization_half() {
        let stats = OcclusionPoolStats {
            capacity: 100,
            used: 50,
            available: 50,
            generation: 0,
            mode: OcclusionMode::Binary,
            resolve_buffer_size: 800,
            has_active_query: false,
        };
        assert!((stats.utilization() - 0.5).abs() < 0.001);
    }

    #[test]
    fn occlusion_pool_stats_zero_capacity() {
        let stats = OcclusionPoolStats {
            capacity: 0,
            used: 0,
            available: 0,
            generation: 0,
            mode: OcclusionMode::Binary,
            resolve_buffer_size: 0,
            has_active_query: false,
        };
        assert_eq!(stats.utilization(), 0.0);
    }

    #[test]
    fn occlusion_pool_stats_display() {
        let stats = OcclusionPoolStats {
            capacity: 64,
            used: 32,
            available: 32,
            generation: 5,
            mode: OcclusionMode::SampleCount,
            resolve_buffer_size: 512,
            has_active_query: true,
        };
        let display = format!("{}", stats);
        assert!(display.contains("32/64"));
        assert!(display.contains("50.0%"));
        assert!(display.contains("SampleCount"));
    }

    // ===== SECTION 7: OcclusionQueryPoolBuilder tests =====

    #[test]
    fn occlusion_builder_defaults() {
        let builder = OcclusionQueryPoolBuilder::new();
        assert_eq!(builder.capacity, 64);
        assert_eq!(builder.mode, OcclusionMode::Binary);
        assert!(builder.label.is_none());
    }

    #[test]
    fn occlusion_builder_capacity() {
        let builder = OcclusionQueryPoolBuilder::new().capacity(256);
        assert_eq!(builder.capacity, 256);
    }

    #[test]
    fn occlusion_builder_mode() {
        let builder = OcclusionQueryPoolBuilder::new().mode(OcclusionMode::SampleCount);
        assert_eq!(builder.mode, OcclusionMode::SampleCount);
    }

    #[test]
    fn occlusion_builder_label() {
        let builder = OcclusionQueryPoolBuilder::new().label("MyPool");
        assert_eq!(builder.label, Some("MyPool".to_string()));
    }

    #[test]
    fn occlusion_builder_chaining() {
        let builder = OcclusionQueryPoolBuilder::new()
            .capacity(128)
            .mode(OcclusionMode::SampleCount)
            .label("TestPool");
        assert_eq!(builder.capacity, 128);
        assert_eq!(builder.mode, OcclusionMode::SampleCount);
        assert_eq!(builder.label, Some("TestPool".to_string()));
    }

    #[test]
    fn occlusion_builder_default_trait() {
        let builder = OcclusionQueryPoolBuilder::default();
        assert_eq!(builder.capacity, 64);
    }

    // ===== SECTION 8: Utility function tests =====

    #[test]
    fn utility_calculate_buffer_size() {
        assert_eq!(calculate_resolve_buffer_size(1), 8);
        assert_eq!(calculate_resolve_buffer_size(64), 512);
        assert_eq!(calculate_resolve_buffer_size(128), 1024);
    }

    #[test]
    fn utility_is_sample_count_visible() {
        assert!(!is_sample_count_visible(0));
        assert!(is_sample_count_visible(1));
        assert!(is_sample_count_visible(100));
        assert!(is_sample_count_visible(u64::MAX));
    }

    // ===== SECTION 9: Constants tests =====

    #[test]
    fn occlusion_result_size_is_8() {
        assert_eq!(OCCLUSION_RESULT_SIZE_BYTES, 8);
    }

    #[test]
    fn min_pool_capacity_is_positive() {
        assert!(MIN_POOL_CAPACITY >= 1);
    }

    #[test]
    fn max_recommended_capacity_reasonable() {
        assert!(MAX_RECOMMENDED_CAPACITY >= 256);
        assert!(MAX_RECOMMENDED_CAPACITY <= 65536);
    }

    #[test]
    fn visibility_threshold_is_zero() {
        assert_eq!(VISIBILITY_THRESHOLD, 0);
    }

    // ===== SECTION 10: Additional edge case tests =====

    #[test]
    fn occlusion_result_boundary_sample_count() {
        // Test with sample count = 1 (minimum visible)
        let result = OcclusionResult::new(0, 1);
        assert!(result.is_visible);

        // Test with max u64
        let result_max = OcclusionResult::new(0, u64::MAX);
        assert!(result_max.is_visible);
    }

    #[test]
    fn occlusion_data_large_start_index() {
        let data = OcclusionData::new(vec![100], OcclusionMode::Binary, u32::MAX - 1);
        let results = data.to_results();
        assert_eq!(results[0].query_index, u32::MAX - 1);
    }

    #[test]
    fn occlusion_resolve_params_zero_query_count() {
        let params = OcclusionResolveParams::new(0, 0, 0);
        assert_eq!(params.end_query(), 0);
        assert_eq!(params.required_buffer_size(), 0);
    }

    #[test]
    fn occlusion_allocation_equality() {
        let a = OcclusionQueryAllocation::new(1, 1);
        let b = OcclusionQueryAllocation::new(1, 1);
        let c = OcclusionQueryAllocation::new(1, 2);
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn occlusion_allocation_hash_consistency() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(OcclusionQueryAllocation::new(1, 0));
        set.insert(OcclusionQueryAllocation::new(2, 0));
        set.insert(OcclusionQueryAllocation::new(1, 0)); // Duplicate
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn occlusion_result_hash_consistency() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(OcclusionResult::new(1, 100));
        set.insert(OcclusionResult::new(2, 200));
        set.insert(OcclusionResult::new(1, 100)); // Duplicate
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn occlusion_data_mixed_visibility() {
        // All invisible
        let data = OcclusionData::new(vec![0, 0, 0], OcclusionMode::Binary, 0);
        assert_eq!(data.visible_count(), 0);
        assert_eq!(data.occluded_count(), 3);

        // All visible
        let data2 = OcclusionData::new(vec![1, 1, 1], OcclusionMode::Binary, 0);
        assert_eq!(data2.visible_count(), 3);
        assert_eq!(data2.occluded_count(), 0);
    }
}
