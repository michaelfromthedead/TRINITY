//! Timestamp query pool for GPU performance measurement.
//!
//! This module provides a reusable pool for wgpu timestamp queries, enabling
//! GPU-side timing measurements. It handles feature detection, capacity
//! management, index allocation, and resolve buffer creation.
//!
//! # Overview
//!
//! GPU timestamp queries allow measuring the exact time taken by GPU operations.
//! The `TimestampQueryPool` manages:
//!
//! - **Feature detection**: Checks if `TIMESTAMP_QUERY` is supported
//! - **QuerySet creation**: Creates a `QuerySet` with `QueryType::Timestamp`
//! - **Index allocation**: Tracks and allocates query indices
//! - **Resolve buffer**: Buffer for reading back timestamp results (u64 per query)
//!
//! # Usage
//!
//! ```no_run
//! use renderer_backend::query_pool::TimestampQueryPool;
//!
//! # fn example(device: &wgpu::Device, queue: &wgpu::Queue) {
//! // Check if timestamps are supported
//! if !TimestampQueryPool::is_supported(device) {
//!     println!("Timestamp queries not supported on this device");
//!     return;
//! }
//!
//! // Create a pool with capacity for 64 timestamps
//! let mut pool = TimestampQueryPool::new(device, queue, 64)
//!     .expect("Failed to create query pool");
//!
//! // Allocate indices for a pass (begin + end timestamps)
//! let begin_idx = pool.allocate().expect("Pool exhausted");
//! let end_idx = pool.allocate().expect("Pool exhausted");
//!
//! // Use with command encoder
//! // encoder.write_timestamp(pool.query_set(), begin_idx);
//! // ... render commands ...
//! // encoder.write_timestamp(pool.query_set(), end_idx);
//!
//! // Reset pool for next frame
//! pool.reset();
//! # }
//! ```
//!
//! # Timestamp Resolution
//!
//! Timestamp values are in GPU clock ticks. To convert to nanoseconds:
//!
//! ```no_run
//! use renderer_backend::query_pool::TimestampQueryPool;
//!
//! # fn example(pool: &TimestampQueryPool, begin_ticks: u64, end_ticks: u64) {
//! let period = pool.timestamp_period(); // ns per tick
//! let delta_ticks = end_ticks - begin_ticks;
//! let delta_ns = (delta_ticks as f32) * period;
//! let delta_ms = delta_ns / 1_000_000.0;
//! # }
//! ```
//!
//! # wgpu 25.x Compatibility
//!
//! This implementation targets wgpu 22+ and follows these patterns:
//! - `device.create_query_set()` with `QuerySetDescriptor`
//! - `QueryType::Timestamp` for timestamp queries
//! - `Features::TIMESTAMP_QUERY` feature flag
//! - `queue.get_timestamp_period()` for tick-to-ns conversion

use std::fmt;
use std::ops::Range;
use thiserror::Error;
use wgpu::{Buffer, BufferDescriptor, BufferUsages, CommandEncoder, Device, Features, MapMode, QuerySet, QuerySetDescriptor, QueryType, Queue, Maintain};

// ============================================================================
// Constants
// ============================================================================

/// Size of a single timestamp value in bytes (u64).
pub const TIMESTAMP_SIZE_BYTES: u64 = 8;

/// Minimum capacity for a query pool.
pub const MIN_POOL_CAPACITY: u32 = 1;

/// Maximum recommended capacity for a query pool.
/// Larger pools use more memory but allow more concurrent measurements.
pub const MAX_RECOMMENDED_CAPACITY: u32 = 8192;

/// Default label prefix for query pool resources.
pub const DEFAULT_LABEL_PREFIX: &str = "TimestampQueryPool";

// ============================================================================
// QueryError
// ============================================================================

/// Error types for query pool operations.
///
/// Provides detailed information about query pool failures, including
/// feature support issues, capacity problems, and index errors.
#[derive(Debug, Clone, Error)]
pub enum QueryError {
    /// The TIMESTAMP_QUERY feature is not supported by the device.
    #[error("timestamp query feature not supported: device does not have TIMESTAMP_QUERY capability")]
    FeatureNotSupported,

    /// The query pool has no more available indices.
    #[error("query pool exhausted: all {capacity} indices have been allocated")]
    PoolExhausted {
        /// Total capacity of the pool
        capacity: u32,
    },

    /// An invalid query index was provided.
    #[error("invalid query index: {index} is out of bounds (capacity: {capacity})")]
    InvalidIndex {
        /// The invalid index
        index: u32,
        /// Pool capacity
        capacity: u32,
    },

    /// Query resolve operation failed.
    #[error("resolve failed: {reason}")]
    ResolveFailed {
        /// Reason for failure
        reason: String,
    },

    /// Invalid capacity was specified.
    #[error("invalid capacity: {capacity} (must be >= {MIN_POOL_CAPACITY})")]
    InvalidCapacity {
        /// The invalid capacity value
        capacity: u32,
    },

    /// Buffer creation failed.
    #[error("buffer creation failed: {reason}")]
    BufferCreationFailed {
        /// Reason for failure
        reason: String,
    },

    /// Buffer mapping failed.
    #[error("buffer mapping failed: {reason}")]
    BufferMappingFailed {
        /// Reason for failure
        reason: String,
    },

    /// Invalid query range specified.
    #[error("invalid query range: {start}..{end} (capacity: {capacity})")]
    InvalidQueryRange {
        /// Start of range
        start: u32,
        /// End of range (exclusive)
        end: u32,
        /// Pool capacity
        capacity: u32,
    },

    /// Readback buffer not ready.
    #[error("readback buffer not ready: {reason}")]
    ReadbackNotReady {
        /// Reason for failure
        reason: String,
    },
}

impl QueryError {
    /// Create a feature not supported error.
    #[inline]
    pub fn feature_not_supported() -> Self {
        QueryError::FeatureNotSupported
    }

    /// Create a pool exhausted error.
    #[inline]
    pub fn pool_exhausted(capacity: u32) -> Self {
        QueryError::PoolExhausted { capacity }
    }

    /// Create an invalid index error.
    #[inline]
    pub fn invalid_index(index: u32, capacity: u32) -> Self {
        QueryError::InvalidIndex { index, capacity }
    }

    /// Create a resolve failed error.
    #[inline]
    pub fn resolve_failed(reason: impl Into<String>) -> Self {
        QueryError::ResolveFailed {
            reason: reason.into(),
        }
    }

    /// Create an invalid capacity error.
    #[inline]
    pub fn invalid_capacity(capacity: u32) -> Self {
        QueryError::InvalidCapacity { capacity }
    }

    /// Create a buffer creation failed error.
    #[inline]
    pub fn buffer_creation_failed(reason: impl Into<String>) -> Self {
        QueryError::BufferCreationFailed {
            reason: reason.into(),
        }
    }

    /// Create a buffer mapping failed error.
    #[inline]
    pub fn buffer_mapping_failed(reason: impl Into<String>) -> Self {
        QueryError::BufferMappingFailed {
            reason: reason.into(),
        }
    }

    /// Create an invalid query range error.
    #[inline]
    pub fn invalid_query_range(start: u32, end: u32, capacity: u32) -> Self {
        QueryError::InvalidQueryRange { start, end, capacity }
    }

    /// Create a readback not ready error.
    #[inline]
    pub fn readback_not_ready(reason: impl Into<String>) -> Self {
        QueryError::ReadbackNotReady {
            reason: reason.into(),
        }
    }

    /// Check if this is a feature support error.
    #[inline]
    pub fn is_feature_error(&self) -> bool {
        matches!(self, QueryError::FeatureNotSupported)
    }

    /// Check if this is a capacity error.
    #[inline]
    pub fn is_capacity_error(&self) -> bool {
        matches!(self, QueryError::PoolExhausted { .. } | QueryError::InvalidCapacity { .. })
    }
}

/// Simplified error kind enum for pattern matching.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QueryErrorKind {
    /// TIMESTAMP_QUERY feature not available.
    FeatureNotSupported,
    /// No more indices available in the pool.
    PoolExhausted,
    /// Index is out of valid range.
    InvalidIndex,
    /// Query resolve operation failed.
    ResolveFailed,
    /// Invalid capacity specified.
    InvalidCapacity,
    /// Buffer creation failed.
    BufferCreationFailed,
    /// Buffer mapping failed.
    BufferMappingFailed,
    /// Invalid query range.
    InvalidQueryRange,
    /// Readback not ready.
    ReadbackNotReady,
}

impl From<&QueryError> for QueryErrorKind {
    fn from(error: &QueryError) -> Self {
        match error {
            QueryError::FeatureNotSupported => QueryErrorKind::FeatureNotSupported,
            QueryError::PoolExhausted { .. } => QueryErrorKind::PoolExhausted,
            QueryError::InvalidIndex { .. } => QueryErrorKind::InvalidIndex,
            QueryError::ResolveFailed { .. } => QueryErrorKind::ResolveFailed,
            QueryError::InvalidCapacity { .. } => QueryErrorKind::InvalidCapacity,
            QueryError::BufferCreationFailed { .. } => QueryErrorKind::BufferCreationFailed,
            QueryError::BufferMappingFailed { .. } => QueryErrorKind::BufferMappingFailed,
            QueryError::InvalidQueryRange { .. } => QueryErrorKind::InvalidQueryRange,
            QueryError::ReadbackNotReady { .. } => QueryErrorKind::ReadbackNotReady,
        }
    }
}

impl QueryError {
    /// Get the error kind for pattern matching.
    #[inline]
    pub fn kind(&self) -> QueryErrorKind {
        QueryErrorKind::from(self)
    }
}

// ============================================================================
// QueryAllocation
// ============================================================================

/// Represents an allocated query index.
///
/// This is a simple wrapper around a query index with additional metadata.
/// It can be used to track allocations and provide better debugging.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct QueryAllocation {
    /// The allocated query index.
    pub index: u32,
    /// Generation counter for validation (optional, for advanced usage).
    pub generation: u32,
}

impl QueryAllocation {
    /// Create a new query allocation.
    #[inline]
    pub const fn new(index: u32, generation: u32) -> Self {
        Self { index, generation }
    }

    /// Create an allocation with default generation (0).
    #[inline]
    pub const fn with_index(index: u32) -> Self {
        Self { index, generation: 0 }
    }
}

impl fmt::Display for QueryAllocation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "QueryAllocation(index={}, gen={})", self.index, self.generation)
    }
}

// ============================================================================
// QueryResolveParams
// ============================================================================

/// Parameters for resolving timestamp queries to a buffer.
///
/// This struct specifies which queries to resolve and where to write
/// the results in the resolve buffer.
///
/// # Example
///
/// ```no_run
/// use renderer_backend::query_pool::QueryResolveParams;
///
/// // Resolve queries 0-3 (4 queries) to the start of the buffer
/// let params = QueryResolveParams {
///     start_query: 0,
///     query_count: 4,
///     destination_offset: 0,
/// };
///
/// // Resolve queries 4-7 (4 queries) after the first 4
/// let params2 = QueryResolveParams {
///     start_query: 4,
///     query_count: 4,
///     destination_offset: 4 * 8, // 4 u64 values * 8 bytes each
/// };
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct QueryResolveParams {
    /// Starting query index in the QuerySet.
    pub start_query: u32,
    /// Number of queries to resolve.
    pub query_count: u32,
    /// Byte offset in the destination resolve buffer.
    pub destination_offset: u64,
}

impl QueryResolveParams {
    /// Create new resolve parameters.
    ///
    /// # Arguments
    ///
    /// * `start_query` - First query index to resolve
    /// * `query_count` - Number of consecutive queries to resolve
    /// * `destination_offset` - Byte offset in the resolve buffer
    #[inline]
    pub const fn new(start_query: u32, query_count: u32, destination_offset: u64) -> Self {
        Self {
            start_query,
            query_count,
            destination_offset,
        }
    }

    /// Create parameters to resolve from the beginning.
    ///
    /// Sets `start_query` to 0 and `destination_offset` to 0.
    ///
    /// # Arguments
    ///
    /// * `query_count` - Number of queries to resolve
    #[inline]
    pub const fn from_start(query_count: u32) -> Self {
        Self {
            start_query: 0,
            query_count,
            destination_offset: 0,
        }
    }

    /// Calculate the end query index (exclusive).
    ///
    /// Returns `start_query + query_count`.
    #[inline]
    pub const fn end_query(&self) -> u32 {
        self.start_query + self.query_count
    }

    /// Calculate the required buffer size in bytes.
    ///
    /// Returns `destination_offset + query_count * 8`.
    #[inline]
    pub const fn required_buffer_size(&self) -> u64 {
        self.destination_offset + (self.query_count as u64) * TIMESTAMP_SIZE_BYTES
    }
}

impl Default for QueryResolveParams {
    fn default() -> Self {
        Self {
            start_query: 0,
            query_count: 0,
            destination_offset: 0,
        }
    }
}

impl fmt::Display for QueryResolveParams {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "QueryResolveParams(queries={}..{}, offset={})",
            self.start_query,
            self.end_query(),
            self.destination_offset
        )
    }
}

// ============================================================================
// Timestamp Result Types
// ============================================================================

/// Result of a timestamp query pair (start, end).
///
/// Contains both the raw tick values and calculated durations in
/// nanoseconds and milliseconds.
///
/// # Example
///
/// ```no_run
/// use renderer_backend::query_pool::TimestampResult;
///
/// let result = TimestampResult::new(1000, 2000, 25.0);
/// assert_eq!(result.duration_ticks, 1000);
/// println!("Duration: {:.3}ms", result.duration_ms);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TimestampResult {
    /// Raw start timestamp in GPU ticks.
    pub start_ticks: u64,
    /// Raw end timestamp in GPU ticks.
    pub end_ticks: u64,
    /// Duration in GPU ticks (end - start).
    pub duration_ticks: u64,
    /// Duration in nanoseconds.
    pub duration_ns: f64,
    /// Duration in milliseconds.
    pub duration_ms: f64,
}

impl TimestampResult {
    /// Create a new timestamp result from tick values and period.
    ///
    /// # Arguments
    ///
    /// * `start_ticks` - Starting timestamp in GPU ticks
    /// * `end_ticks` - Ending timestamp in GPU ticks
    /// * `timestamp_period` - Nanoseconds per tick
    ///
    /// # Note
    ///
    /// If `end_ticks < start_ticks`, duration will be 0 (saturating subtraction).
    #[inline]
    pub fn new(start_ticks: u64, end_ticks: u64, timestamp_period: f32) -> Self {
        let duration_ticks = end_ticks.saturating_sub(start_ticks);
        let duration_ns = (duration_ticks as f64) * (timestamp_period as f64);
        let duration_ms = duration_ns / 1_000_000.0;

        Self {
            start_ticks,
            end_ticks,
            duration_ticks,
            duration_ns,
            duration_ms,
        }
    }

    /// Create a zero-duration result.
    #[inline]
    pub const fn zero() -> Self {
        Self {
            start_ticks: 0,
            end_ticks: 0,
            duration_ticks: 0,
            duration_ns: 0.0,
            duration_ms: 0.0,
        }
    }

    /// Check if this result represents a valid measurement (non-zero duration).
    #[inline]
    pub fn is_valid(&self) -> bool {
        self.duration_ticks > 0
    }

    /// Get duration in microseconds.
    #[inline]
    pub fn duration_us(&self) -> f64 {
        self.duration_ns / 1_000.0
    }

    /// Get duration in seconds.
    #[inline]
    pub fn duration_secs(&self) -> f64 {
        self.duration_ms / 1_000.0
    }
}

impl Default for TimestampResult {
    fn default() -> Self {
        Self::zero()
    }
}

impl fmt::Display for TimestampResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{:.3}ms ({} ticks @ {}..{})",
            self.duration_ms,
            self.duration_ticks,
            self.start_ticks,
            self.end_ticks
        )
    }
}

/// Profile result with optional label.
///
/// Associates a timestamp result with a human-readable label for
/// identifying which GPU operation was measured.
///
/// # Example
///
/// ```no_run
/// use renderer_backend::query_pool::{ProfileResult, TimestampResult};
///
/// let result = TimestampResult::new(0, 1000, 25.0);
/// let profile = ProfileResult::with_label("Shadow Pass", result);
/// println!("{}: {:.3}ms", profile.label.unwrap(), profile.result.duration_ms);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct ProfileResult {
    /// Optional human-readable label for this measurement.
    pub label: Option<String>,
    /// The timestamp measurement result.
    pub result: TimestampResult,
}

impl ProfileResult {
    /// Create a new profile result without a label.
    #[inline]
    pub fn new(result: TimestampResult) -> Self {
        Self { label: None, result }
    }

    /// Create a new profile result with a label.
    #[inline]
    pub fn with_label(label: impl Into<String>, result: TimestampResult) -> Self {
        Self {
            label: Some(label.into()),
            result,
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

    /// Check if this result has a label.
    #[inline]
    pub fn has_label(&self) -> bool {
        self.label.is_some()
    }
}

impl Default for ProfileResult {
    fn default() -> Self {
        Self {
            label: None,
            result: TimestampResult::zero(),
        }
    }
}

impl fmt::Display for ProfileResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(ref label) = self.label {
            write!(f, "{}: {}", label, self.result)
        } else {
            write!(f, "{}", self.result)
        }
    }
}

/// Raw timestamp data from readback.
///
/// Contains the raw u64 timestamp values read from the resolve buffer,
/// along with the timestamp period for conversion.
///
/// # Usage
///
/// Typically obtained from `read_timestamps_async` or `read_timestamps_blocking`,
/// then passed to `parse_timestamp_pairs` for interpretation.
#[derive(Debug, Clone)]
pub struct TimestampData {
    /// Raw timestamp values in GPU ticks.
    pub timestamps: Vec<u64>,
    /// Nanoseconds per tick (for conversion).
    pub timestamp_period: f32,
}

impl TimestampData {
    /// Create new timestamp data.
    #[inline]
    pub fn new(timestamps: Vec<u64>, timestamp_period: f32) -> Self {
        Self {
            timestamps,
            timestamp_period,
        }
    }

    /// Create empty timestamp data.
    #[inline]
    pub fn empty(timestamp_period: f32) -> Self {
        Self {
            timestamps: Vec::new(),
            timestamp_period,
        }
    }

    /// Get the number of timestamps.
    #[inline]
    pub fn len(&self) -> usize {
        self.timestamps.len()
    }

    /// Check if empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.timestamps.is_empty()
    }

    /// Get timestamp at index, if present.
    #[inline]
    pub fn get(&self, index: usize) -> Option<u64> {
        self.timestamps.get(index).copied()
    }

    /// Get the number of complete timestamp pairs.
    ///
    /// Returns `len() / 2`.
    #[inline]
    pub fn pair_count(&self) -> usize {
        self.timestamps.len() / 2
    }

    /// Get a timestamp pair by index.
    ///
    /// Index 0 returns timestamps [0] and [1], index 1 returns [2] and [3], etc.
    #[inline]
    pub fn get_pair(&self, pair_index: usize) -> Option<(u64, u64)> {
        let start_idx = pair_index * 2;
        let end_idx = start_idx + 1;
        if end_idx < self.timestamps.len() {
            Some((self.timestamps[start_idx], self.timestamps[end_idx]))
        } else {
            None
        }
    }

    /// Calculate duration between two indices in milliseconds.
    #[inline]
    pub fn duration_ms(&self, start_idx: usize, end_idx: usize) -> Option<f64> {
        let start = self.get(start_idx)?;
        let end = self.get(end_idx)?;
        let delta = end.saturating_sub(start);
        let ns = (delta as f64) * (self.timestamp_period as f64);
        Some(ns / 1_000_000.0)
    }
}

impl Default for TimestampData {
    fn default() -> Self {
        Self::empty(1.0)
    }
}

// ============================================================================
// TimestampQueryPool
// ============================================================================

/// A pool for managing wgpu timestamp queries.
///
/// This struct provides:
/// - Feature checking for `TIMESTAMP_QUERY` support
/// - QuerySet creation with `QueryType::Timestamp`
/// - Index allocation/deallocation
/// - Resolve buffer for reading back timestamp values
///
/// # Example
///
/// ```no_run
/// use renderer_backend::query_pool::TimestampQueryPool;
///
/// # fn example(device: &wgpu::Device, queue: &wgpu::Queue, encoder: &mut wgpu::CommandEncoder) {
/// // Create pool with 128 timestamp capacity
/// let mut pool = TimestampQueryPool::new(device, queue, 128).unwrap();
///
/// // Allocate indices for timing a pass
/// let begin = pool.allocate().unwrap();
/// let end = pool.allocate().unwrap();
///
/// // Record timestamps
/// encoder.write_timestamp(pool.query_set(), begin);
/// // ... GPU work ...
/// encoder.write_timestamp(pool.query_set(), end);
///
/// // Resolve to buffer for readback
/// encoder.resolve_query_set(
///     pool.query_set(),
///     0..pool.used(),
///     pool.resolve_buffer(),
///     0,
/// );
///
/// // Reset for next frame
/// pool.reset();
/// # }
/// ```
pub struct TimestampQueryPool {
    /// The wgpu QuerySet for timestamp queries.
    query_set: QuerySet,
    /// Buffer for resolving query results.
    resolve_buffer: Buffer,
    /// Maximum number of queries this pool can hold.
    capacity: u32,
    /// Next index to allocate (watermark allocator).
    next_index: u32,
    /// Timestamp period in nanoseconds per tick.
    timestamp_period: f32,
    /// Generation counter for allocation tracking.
    generation: u32,
    /// Optional label for debugging.
    label: Option<String>,
}

impl TimestampQueryPool {
    /// Create a new timestamp query pool.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device to create resources on
    /// * `queue` - The wgpu queue (used to get timestamp period)
    /// * `capacity` - Maximum number of timestamp queries
    ///
    /// # Returns
    ///
    /// * `Ok(TimestampQueryPool)` - Successfully created pool
    /// * `Err(QueryError::FeatureNotSupported)` - Device lacks TIMESTAMP_QUERY
    /// * `Err(QueryError::InvalidCapacity)` - Capacity is 0
    ///
    /// # Example
    ///
    /// ```no_run
    /// use renderer_backend::query_pool::TimestampQueryPool;
    ///
    /// # fn example(device: &wgpu::Device, queue: &wgpu::Queue) {
    /// let pool = TimestampQueryPool::new(device, queue, 64)?;
    /// println!("Created pool with {} capacity", pool.capacity());
    /// # Ok::<(), renderer_backend::query_pool::QueryError>(())
    /// # }
    /// ```
    pub fn new(device: &Device, queue: &Queue, capacity: u32) -> Result<Self, QueryError> {
        Self::with_label(device, queue, capacity, None)
    }

    /// Create a new timestamp query pool with a custom label.
    ///
    /// The label is used for debugging in graphics debuggers and profilers.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device
    /// * `queue` - The wgpu queue
    /// * `capacity` - Maximum number of queries
    /// * `label` - Optional debug label
    ///
    /// # Returns
    ///
    /// * `Ok(TimestampQueryPool)` - Successfully created pool
    /// * `Err(QueryError)` - Creation failed
    pub fn with_label(
        device: &Device,
        queue: &Queue,
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

        // Get timestamp period from queue
        let timestamp_period = queue.get_timestamp_period();

        // Create the QuerySet with QueryType::Timestamp
        let query_set_label = label
            .map(|l| format!("{} QuerySet", l))
            .unwrap_or_else(|| format!("{} QuerySet", DEFAULT_LABEL_PREFIX));

        let query_set = device.create_query_set(&QuerySetDescriptor {
            label: Some(&query_set_label),
            ty: QueryType::Timestamp,
            count: capacity,
        });

        // Create resolve buffer (u64 per timestamp = 8 bytes)
        let buffer_size = (capacity as u64) * TIMESTAMP_SIZE_BYTES;
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
            timestamp_period,
            generation: 0,
            label: label.map(String::from),
        })
    }

    /// Check if timestamp queries are supported on the device.
    ///
    /// This checks if the device has the `TIMESTAMP_QUERY` feature enabled.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device to check
    ///
    /// # Returns
    ///
    /// `true` if timestamp queries are supported, `false` otherwise.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use renderer_backend::query_pool::TimestampQueryPool;
    ///
    /// # fn example(device: &wgpu::Device) {
    /// if TimestampQueryPool::is_supported(device) {
    ///     println!("GPU timestamp queries available");
    /// } else {
    ///     println!("Using CPU-based profiling instead");
    /// }
    /// # }
    /// ```
    #[inline]
    pub fn is_supported(device: &Device) -> bool {
        device.features().contains(Features::TIMESTAMP_QUERY)
    }

    /// Allocate the next available query index.
    ///
    /// Uses a watermark allocator for efficient sequential allocation.
    /// Indices are allocated from 0 up to `capacity - 1`.
    ///
    /// # Returns
    ///
    /// * `Ok(index)` - The allocated query index
    /// * `Err(QueryError::PoolExhausted)` - No more indices available
    ///
    /// # Example
    ///
    /// ```no_run
    /// use renderer_backend::query_pool::TimestampQueryPool;
    ///
    /// # fn example(pool: &mut TimestampQueryPool, encoder: &mut wgpu::CommandEncoder) {
    /// let idx = pool.allocate()?;
    /// encoder.write_timestamp(pool.query_set(), idx);
    /// # Ok::<(), renderer_backend::query_pool::QueryError>(())
    /// # }
    /// ```
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
    ///
    /// Returns a `QueryAllocation` with the index and generation counter
    /// for more advanced tracking scenarios.
    ///
    /// # Returns
    ///
    /// * `Ok(QueryAllocation)` - Allocation with index and generation
    /// * `Err(QueryError::PoolExhausted)` - No more indices available
    pub fn allocate_tracked(&mut self) -> Result<QueryAllocation, QueryError> {
        let index = self.allocate()?;
        Ok(QueryAllocation::new(index, self.generation))
    }

    /// Try to allocate multiple consecutive query indices.
    ///
    /// Useful for allocating begin/end timestamp pairs atomically.
    ///
    /// # Arguments
    ///
    /// * `count` - Number of consecutive indices to allocate
    ///
    /// # Returns
    ///
    /// * `Ok(start_index)` - Starting index of the allocated range
    /// * `Err(QueryError::PoolExhausted)` - Not enough capacity
    ///
    /// # Example
    ///
    /// ```no_run
    /// use renderer_backend::query_pool::TimestampQueryPool;
    ///
    /// # fn example(pool: &mut TimestampQueryPool) {
    /// // Allocate a pair of indices for begin/end timestamps
    /// let start = pool.allocate_range(2)?;
    /// let begin_idx = start;
    /// let end_idx = start + 1;
    /// # Ok::<(), renderer_backend::query_pool::QueryError>(())
    /// # }
    /// ```
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

    /// Deallocate a specific query index.
    ///
    /// **Note**: This uses a watermark allocator, so individual deallocation
    /// is a no-op. Use `reset()` to reclaim all indices for the next frame.
    /// This method is provided for API completeness but has no effect.
    ///
    /// # Arguments
    ///
    /// * `_index` - The index to deallocate (ignored)
    #[inline]
    pub fn deallocate(&mut self, _index: u32) {
        // Watermark allocator: individual deallocation is a no-op
        // Use reset() to reclaim all indices
    }

    /// Reset the pool for reuse.
    ///
    /// This resets the allocation watermark to 0, making all indices
    /// available again. Call this at the start of each frame after
    /// reading back results from the previous frame.
    ///
    /// Also increments the generation counter for allocation tracking.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use renderer_backend::query_pool::TimestampQueryPool;
    ///
    /// # fn example(pool: &mut TimestampQueryPool) {
    /// // At start of frame
    /// pool.reset();
    ///
    /// // Now all indices are available again
    /// let idx1 = pool.allocate()?;
    /// let idx2 = pool.allocate()?;
    /// # Ok::<(), renderer_backend::query_pool::QueryError>(())
    /// # }
    /// ```
    #[inline]
    pub fn reset(&mut self) {
        self.next_index = 0;
        self.generation = self.generation.wrapping_add(1);
    }

    /// Get the maximum capacity of this pool.
    ///
    /// This is the total number of timestamps that can be recorded
    /// before the pool is exhausted.
    #[inline]
    pub fn capacity(&self) -> u32 {
        self.capacity
    }

    /// Get the number of available (unallocated) indices.
    ///
    /// Returns `capacity - next_index`.
    #[inline]
    pub fn available(&self) -> u32 {
        self.capacity.saturating_sub(self.next_index)
    }

    /// Get the number of used (allocated) indices.
    ///
    /// Returns the current allocation watermark.
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

    /// Check if the pool is full (all indices allocated).
    #[inline]
    pub fn is_full(&self) -> bool {
        self.next_index >= self.capacity
    }

    /// Get a reference to the underlying QuerySet.
    ///
    /// Use this with `encoder.write_timestamp()` to record timestamps.
    #[inline]
    pub fn query_set(&self) -> &QuerySet {
        &self.query_set
    }

    /// Get a reference to the resolve buffer.
    ///
    /// Use this with `encoder.resolve_query_set()` and subsequent
    /// buffer copy operations for reading back timestamp values.
    ///
    /// Buffer layout: `u64[capacity]` where each entry is a timestamp tick.
    #[inline]
    pub fn resolve_buffer(&self) -> &Buffer {
        &self.resolve_buffer
    }

    /// Get the timestamp period in nanoseconds per tick.
    ///
    /// Use this to convert timestamp differences to actual time:
    /// ```no_run
    /// # fn example(period: f32, begin: u64, end: u64) {
    /// let delta_ticks = end - begin;
    /// let delta_ns = (delta_ticks as f32) * period;
    /// let delta_ms = delta_ns / 1_000_000.0;
    /// # }
    /// ```
    #[inline]
    pub fn timestamp_period(&self) -> f32 {
        self.timestamp_period
    }

    /// Get the current generation counter.
    ///
    /// Incremented on each `reset()`. Useful for validating allocations.
    #[inline]
    pub fn generation(&self) -> u32 {
        self.generation
    }

    /// Get the debug label, if any.
    #[inline]
    pub fn label(&self) -> Option<&str> {
        self.label.as_deref()
    }

    /// Calculate the resolve buffer size in bytes.
    ///
    /// This is `capacity * 8` (u64 per timestamp).
    #[inline]
    pub fn resolve_buffer_size(&self) -> u64 {
        (self.capacity as u64) * TIMESTAMP_SIZE_BYTES
    }

    /// Convert timestamp ticks to milliseconds.
    ///
    /// Convenience method using this pool's timestamp period.
    ///
    /// # Arguments
    ///
    /// * `ticks` - Number of GPU timestamp ticks
    ///
    /// # Returns
    ///
    /// Time in milliseconds.
    #[inline]
    pub fn ticks_to_ms(&self, ticks: u64) -> f32 {
        let ns = (ticks as f64) * (self.timestamp_period as f64);
        (ns / 1_000_000.0) as f32
    }

    /// Convert a tick delta to milliseconds.
    ///
    /// Convenience method for measuring elapsed time.
    ///
    /// # Arguments
    ///
    /// * `begin_ticks` - Starting timestamp
    /// * `end_ticks` - Ending timestamp
    ///
    /// # Returns
    ///
    /// Elapsed time in milliseconds.
    #[inline]
    pub fn delta_to_ms(&self, begin_ticks: u64, end_ticks: u64) -> f32 {
        let delta = end_ticks.saturating_sub(begin_ticks);
        self.ticks_to_ms(delta)
    }

    /// Validate that an index is within bounds.
    ///
    /// # Arguments
    ///
    /// * `index` - The index to validate
    ///
    /// # Returns
    ///
    /// * `Ok(())` if valid
    /// * `Err(QueryError::InvalidIndex)` if out of bounds
    #[inline]
    pub fn validate_index(&self, index: u32) -> Result<(), QueryError> {
        if index >= self.capacity {
            return Err(QueryError::invalid_index(index, self.capacity));
        }
        Ok(())
    }

    // ========================================================================
    // Query Resolution
    // ========================================================================

    /// Resolve queries to the internal resolve buffer.
    ///
    /// This method wraps `encoder.resolve_query_set()` to copy timestamp
    /// values from the QuerySet to the resolve buffer for CPU readback.
    ///
    /// # Timing Requirements
    ///
    /// This method **must** be called:
    /// - **After** all render/compute passes that write timestamps complete
    /// - **Before** the command buffer is submitted to the queue
    ///
    /// Typical frame timing:
    /// ```text
    /// 1. encoder.write_timestamp() in passes
    /// 2. pool.resolve(encoder, params)  <-- Call here
    /// 3. queue.submit([encoder.finish()])
    /// 4. Map buffer and read results (next frame)
    /// ```
    ///
    /// # Arguments
    ///
    /// * `encoder` - The command encoder to record the resolve command
    /// * `params` - Parameters specifying which queries to resolve and where
    ///
    /// # Returns
    ///
    /// * `Ok(())` - Resolve command recorded successfully
    /// * `Err(QueryError::ResolveFailed)` - Invalid parameters
    ///
    /// # Example
    ///
    /// ```no_run
    /// use renderer_backend::query_pool::{TimestampQueryPool, QueryResolveParams};
    ///
    /// # fn example(pool: &TimestampQueryPool, encoder: &mut wgpu::CommandEncoder) {
    /// // Resolve queries 0-3 to the buffer
    /// let params = QueryResolveParams::new(0, 4, 0);
    /// pool.resolve(encoder, params)?;
    /// # Ok::<(), renderer_backend::query_pool::QueryError>(())
    /// # }
    /// ```
    pub fn resolve(
        &self,
        encoder: &mut CommandEncoder,
        params: QueryResolveParams,
    ) -> Result<(), QueryError> {
        // Validate query_count > 0
        if params.query_count == 0 {
            return Err(QueryError::resolve_failed(
                "query_count must be greater than 0"
            ));
        }

        // Validate start_query + query_count <= capacity
        let end_query = params.start_query.checked_add(params.query_count)
            .ok_or_else(|| QueryError::resolve_failed(
                format!(
                    "query range overflow: start_query ({}) + query_count ({}) overflows u32",
                    params.start_query, params.query_count
                )
            ))?;

        if end_query > self.capacity {
            return Err(QueryError::resolve_failed(format!(
                "query range out of bounds: end query {} exceeds capacity {}",
                end_query, self.capacity
            )));
        }

        // Validate destination_offset + query_count * 8 <= buffer_size
        let required_size = params.required_buffer_size();
        let buffer_size = self.resolve_buffer_size();

        if required_size > buffer_size {
            return Err(QueryError::resolve_failed(format!(
                "destination overflow: required {} bytes (offset {} + {} queries * 8), \
                 but buffer is only {} bytes",
                required_size, params.destination_offset, params.query_count, buffer_size
            )));
        }

        // Issue the resolve command
        encoder.resolve_query_set(
            &self.query_set,
            params.start_query..end_query,
            &self.resolve_buffer,
            params.destination_offset,
        );

        Ok(())
    }

    /// Resolve all used queries to the resolve buffer.
    ///
    /// Convenience method that resolves queries 0..next_index (all allocated
    /// queries) to the start of the resolve buffer.
    ///
    /// # Timing Requirements
    ///
    /// Must be called after passes complete, before queue submit.
    ///
    /// # Arguments
    ///
    /// * `encoder` - The command encoder to record the resolve command
    ///
    /// # Returns
    ///
    /// * `Ok(())` - Resolve command recorded (or no queries to resolve)
    /// * `Err(QueryError::ResolveFailed)` - Resolve failed
    ///
    /// # Example
    ///
    /// ```no_run
    /// use renderer_backend::query_pool::TimestampQueryPool;
    ///
    /// # fn example(pool: &TimestampQueryPool, encoder: &mut wgpu::CommandEncoder) {
    /// // Record timestamps during rendering...
    /// // pool.allocate() called multiple times
    ///
    /// // Resolve all recorded timestamps
    /// pool.resolve_all(encoder)?;
    /// # Ok::<(), renderer_backend::query_pool::QueryError>(())
    /// # }
    /// ```
    pub fn resolve_all(&self, encoder: &mut CommandEncoder) -> Result<(), QueryError> {
        let count = self.used();

        // Nothing to resolve if pool is empty
        if count == 0 {
            return Ok(());
        }

        let params = QueryResolveParams::from_start(count);
        self.resolve(encoder, params)
    }

    /// Resolve the first N queries to the resolve buffer.
    ///
    /// Convenience method that resolves queries 0..count to the start
    /// of the resolve buffer.
    ///
    /// # Timing Requirements
    ///
    /// Must be called after passes complete, before queue submit.
    ///
    /// # Arguments
    ///
    /// * `encoder` - The command encoder to record the resolve command
    /// * `count` - Number of queries to resolve starting from index 0
    ///
    /// # Returns
    ///
    /// * `Ok(())` - Resolve command recorded
    /// * `Err(QueryError::ResolveFailed)` - Invalid count or resolve failed
    ///
    /// # Example
    ///
    /// ```no_run
    /// use renderer_backend::query_pool::TimestampQueryPool;
    ///
    /// # fn example(pool: &TimestampQueryPool, encoder: &mut wgpu::CommandEncoder) {
    /// // Resolve just the first 4 timestamps (2 begin/end pairs)
    /// pool.resolve_first(encoder, 4)?;
    /// # Ok::<(), renderer_backend::query_pool::QueryError>(())
    /// # }
    /// ```
    pub fn resolve_first(&self, encoder: &mut CommandEncoder, count: u32) -> Result<(), QueryError> {
        if count == 0 {
            return Err(QueryError::resolve_failed("count must be greater than 0"));
        }

        let params = QueryResolveParams::from_start(count);
        self.resolve(encoder, params)
    }

    /// Validate resolve parameters without issuing the resolve command.
    ///
    /// Useful for pre-flight validation before building command buffers.
    ///
    /// # Arguments
    ///
    /// * `params` - The parameters to validate
    ///
    /// # Returns
    ///
    /// * `Ok(())` - Parameters are valid
    /// * `Err(QueryError::ResolveFailed)` - Parameters are invalid
    pub fn validate_resolve_params(&self, params: &QueryResolveParams) -> Result<(), QueryError> {
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

        let required_size = params.required_buffer_size();
        let buffer_size = self.resolve_buffer_size();

        if required_size > buffer_size {
            return Err(QueryError::resolve_failed(format!(
                "destination overflow: {} > {}",
                required_size, buffer_size
            )));
        }

        Ok(())
    }

    // ========================================================================
    // Async Query Readback
    // ========================================================================

    /// Validate a query range for readback.
    ///
    /// # Arguments
    ///
    /// * `query_range` - Range of query indices to read
    ///
    /// # Returns
    ///
    /// * `Ok(())` - Range is valid
    /// * `Err(QueryError)` - Range is invalid
    fn validate_query_range(&self, query_range: &Range<u32>) -> Result<(), QueryError> {
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

        Ok(())
    }

    /// Calculate buffer offset and size for a query range.
    #[inline]
    fn calculate_buffer_range(&self, query_range: &Range<u32>) -> (u64, u64) {
        let offset = (query_range.start as u64) * TIMESTAMP_SIZE_BYTES;
        let size = ((query_range.end - query_range.start) as u64) * TIMESTAMP_SIZE_BYTES;
        (offset, size)
    }

    /// Read resolved timestamps asynchronously.
    ///
    /// Maps the resolve buffer asynchronously and reads timestamp values.
    /// This is the preferred method for non-blocking readback.
    ///
    /// # Timing Requirements
    ///
    /// This method should be called **after** the command buffer containing
    /// the resolve operation has been submitted and the GPU work is complete.
    /// Typically called on frame N+1 to read results from frame N.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device (for polling)
    /// * `query_range` - Range of query indices to read (e.g., 0..4)
    ///
    /// # Returns
    ///
    /// * `Ok(TimestampData)` - Successfully read timestamps
    /// * `Err(QueryError::InvalidQueryRange)` - Invalid range
    /// * `Err(QueryError::BufferMappingFailed)` - Buffer mapping failed
    ///
    /// # Example
    ///
    /// ```no_run
    /// use renderer_backend::query_pool::TimestampQueryPool;
    ///
    /// # async fn example(pool: &TimestampQueryPool, device: &wgpu::Device) {
    /// // Read timestamps 0-3 (2 begin/end pairs)
    /// let data = pool.read_timestamps_async(device, 0..4).await?;
    /// for i in 0..data.pair_count() {
    ///     if let Some((start, end)) = data.get_pair(i) {
    ///         let ms = data.duration_ms(i*2, i*2+1).unwrap();
    ///         println!("Pair {}: {:.3}ms", i, ms);
    ///     }
    /// }
    /// # Ok::<(), renderer_backend::query_pool::QueryError>(())
    /// # }
    /// ```
    pub async fn read_timestamps_async(
        &self,
        device: &Device,
        query_range: Range<u32>,
    ) -> Result<TimestampData, QueryError> {
        // Validate range
        self.validate_query_range(&query_range)?;

        // Calculate buffer slice parameters
        let (offset, size) = self.calculate_buffer_range(&query_range);

        // Create buffer slice
        let slice = self.resolve_buffer.slice(offset..offset + size);

        // Map the buffer asynchronously
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(MapMode::Read, move |result| {
            let _ = tx.send(result);
        });

        // Poll until mapping completes
        device.poll(Maintain::Wait);

        // Check mapping result
        rx.recv()
            .map_err(|_| QueryError::buffer_mapping_failed("channel receive failed"))?
            .map_err(|e| QueryError::buffer_mapping_failed(format!("map_async failed: {:?}", e)))?;

        // Read the data
        let mapped_range = slice.get_mapped_range();
        let query_count = (query_range.end - query_range.start) as usize;
        let mut timestamps = Vec::with_capacity(query_count);

        // Parse u64 values from the buffer
        for i in 0..query_count {
            let byte_offset = i * 8;
            if byte_offset + 8 <= mapped_range.len() {
                let bytes: [u8; 8] = mapped_range[byte_offset..byte_offset + 8]
                    .try_into()
                    .map_err(|_| QueryError::buffer_mapping_failed("failed to read u64 from buffer"))?;
                timestamps.push(u64::from_le_bytes(bytes));
            }
        }

        // Drop the mapped range before unmapping
        drop(mapped_range);

        // Unmap the buffer
        self.resolve_buffer.unmap();

        Ok(TimestampData::new(timestamps, self.timestamp_period))
    }

    /// Read timestamps synchronously (blocks until complete).
    ///
    /// This method blocks the current thread until the buffer is mapped
    /// and read. Use `read_timestamps_async` for non-blocking operation.
    ///
    /// # Timing Requirements
    ///
    /// Same as `read_timestamps_async` - call after resolve is submitted.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device
    /// * `query_range` - Range of query indices to read
    ///
    /// # Returns
    ///
    /// * `Ok(TimestampData)` - Successfully read timestamps
    /// * `Err(QueryError)` - Read failed
    ///
    /// # Example
    ///
    /// ```no_run
    /// use renderer_backend::query_pool::TimestampQueryPool;
    ///
    /// # fn example(pool: &TimestampQueryPool, device: &wgpu::Device) {
    /// let data = pool.read_timestamps_blocking(device, 0..4)?;
    /// println!("Read {} timestamps", data.len());
    /// # Ok::<(), renderer_backend::query_pool::QueryError>(())
    /// # }
    /// ```
    pub fn read_timestamps_blocking(
        &self,
        device: &Device,
        query_range: Range<u32>,
    ) -> Result<TimestampData, QueryError> {
        // Validate range
        self.validate_query_range(&query_range)?;

        // Calculate buffer slice parameters
        let (offset, size) = self.calculate_buffer_range(&query_range);

        // Create buffer slice
        let slice = self.resolve_buffer.slice(offset..offset + size);

        // Set up synchronization
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(MapMode::Read, move |result| {
            let _ = tx.send(result);
        });

        // Poll and wait for completion
        device.poll(Maintain::Wait);

        // Check result
        rx.recv()
            .map_err(|_| QueryError::buffer_mapping_failed("channel receive failed"))?
            .map_err(|e| QueryError::buffer_mapping_failed(format!("map_async failed: {:?}", e)))?;

        // Read the data
        let mapped_range = slice.get_mapped_range();
        let query_count = (query_range.end - query_range.start) as usize;
        let mut timestamps = Vec::with_capacity(query_count);

        // Parse u64 values from the buffer
        for i in 0..query_count {
            let byte_offset = i * 8;
            if byte_offset + 8 <= mapped_range.len() {
                let bytes: [u8; 8] = mapped_range[byte_offset..byte_offset + 8]
                    .try_into()
                    .map_err(|_| QueryError::buffer_mapping_failed("failed to read u64 from buffer"))?;
                timestamps.push(u64::from_le_bytes(bytes));
            }
        }

        // Drop the mapped range before unmapping
        drop(mapped_range);

        // Unmap the buffer
        self.resolve_buffer.unmap();

        Ok(TimestampData::new(timestamps, self.timestamp_period))
    }

    /// Calculate duration between two timestamp values.
    ///
    /// Convenience method that creates a `TimestampResult` from raw tick values.
    ///
    /// # Arguments
    ///
    /// * `start_ticks` - Starting timestamp
    /// * `end_ticks` - Ending timestamp
    ///
    /// # Returns
    ///
    /// A `TimestampResult` with calculated durations.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use renderer_backend::query_pool::TimestampQueryPool;
    ///
    /// # fn example(pool: &TimestampQueryPool) {
    /// let result = pool.calculate_duration(1000, 2000);
    /// println!("Duration: {:.3}ms", result.duration_ms);
    /// # }
    /// ```
    #[inline]
    pub fn calculate_duration(&self, start_ticks: u64, end_ticks: u64) -> TimestampResult {
        TimestampResult::new(start_ticks, end_ticks, self.timestamp_period)
    }

    /// Parse timestamp pairs into profile results.
    ///
    /// Interprets timestamp data as begin/end pairs and creates `ProfileResult`
    /// instances with optional labels.
    ///
    /// # Arguments
    ///
    /// * `data` - Raw timestamp data from readback
    /// * `labels` - Optional labels for each pair (must match pair count if Some)
    ///
    /// # Returns
    ///
    /// A vector of `ProfileResult` instances, one per timestamp pair.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use renderer_backend::query_pool::TimestampQueryPool;
    ///
    /// # fn example(pool: &TimestampQueryPool, device: &wgpu::Device) {
    /// let data = pool.read_timestamps_blocking(device, 0..6)?;
    /// let labels = ["Shadow Pass", "Main Pass", "Post Process"];
    /// let results = pool.parse_timestamp_pairs(&data, Some(&labels));
    /// for result in &results {
    ///     println!("{}", result);
    /// }
    /// # Ok::<(), renderer_backend::query_pool::QueryError>(())
    /// # }
    /// ```
    pub fn parse_timestamp_pairs(
        &self,
        data: &TimestampData,
        labels: Option<&[&str]>,
    ) -> Vec<ProfileResult> {
        let pair_count = data.pair_count();
        let mut results = Vec::with_capacity(pair_count);

        for i in 0..pair_count {
            if let Some((start, end)) = data.get_pair(i) {
                let timestamp_result = self.calculate_duration(start, end);
                let label = labels.and_then(|l| l.get(i).map(|s| s.to_string()));

                results.push(ProfileResult {
                    label,
                    result: timestamp_result,
                });
            }
        }

        results
    }

    /// Parse all used timestamps into profile results.
    ///
    /// Convenience method that reads all allocated timestamps and parses them.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device
    /// * `labels` - Optional labels for each pair
    ///
    /// # Returns
    ///
    /// * `Ok(Vec<ProfileResult>)` - Parsed profile results
    /// * `Err(QueryError)` - Read failed
    pub fn read_and_parse_all(
        &self,
        device: &Device,
        labels: Option<&[&str]>,
    ) -> Result<Vec<ProfileResult>, QueryError> {
        let used = self.used();
        if used == 0 {
            return Ok(Vec::new());
        }

        let data = self.read_timestamps_blocking(device, 0..used)?;
        Ok(self.parse_timestamp_pairs(&data, labels))
    }

    /// Get statistics about pool usage.
    pub fn stats(&self) -> QueryPoolStats {
        QueryPoolStats {
            capacity: self.capacity,
            used: self.next_index,
            available: self.available(),
            generation: self.generation,
            timestamp_period_ns: self.timestamp_period,
            resolve_buffer_size: self.resolve_buffer_size(),
        }
    }
}

impl fmt::Debug for TimestampQueryPool {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TimestampQueryPool")
            .field("capacity", &self.capacity)
            .field("used", &self.next_index)
            .field("available", &self.available())
            .field("generation", &self.generation)
            .field("timestamp_period_ns", &self.timestamp_period)
            .field("label", &self.label)
            .finish()
    }
}

// ============================================================================
// QueryPoolStats
// ============================================================================

/// Statistics about a query pool's current state.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QueryPoolStats {
    /// Total capacity of the pool.
    pub capacity: u32,
    /// Number of allocated indices.
    pub used: u32,
    /// Number of available indices.
    pub available: u32,
    /// Current generation counter.
    pub generation: u32,
    /// Timestamp period in nanoseconds.
    pub timestamp_period_ns: f32,
    /// Resolve buffer size in bytes.
    pub resolve_buffer_size: u64,
}

impl QueryPoolStats {
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

impl fmt::Display for QueryPoolStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "QueryPool: {}/{} used ({:.1}%), gen={}, period={:.2}ns",
            self.used,
            self.capacity,
            self.utilization() * 100.0,
            self.generation,
            self.timestamp_period_ns
        )
    }
}

// ============================================================================
// QueryPoolBuilder
// ============================================================================

/// Builder for creating TimestampQueryPool with custom options.
///
/// # Example
///
/// ```no_run
/// use renderer_backend::query_pool::QueryPoolBuilder;
///
/// # fn example(device: &wgpu::Device, queue: &wgpu::Queue) {
/// let pool = QueryPoolBuilder::new()
///     .capacity(256)
///     .label("FrameTimestamps")
///     .build(device, queue)?;
/// # Ok::<(), renderer_backend::query_pool::QueryError>(())
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct QueryPoolBuilder {
    capacity: u32,
    label: Option<String>,
}

impl Default for QueryPoolBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl QueryPoolBuilder {
    /// Create a new builder with default settings.
    pub fn new() -> Self {
        Self {
            capacity: 64, // Default capacity
            label: None,
        }
    }

    /// Set the pool capacity.
    ///
    /// # Arguments
    ///
    /// * `capacity` - Maximum number of timestamps (must be >= 1)
    pub fn capacity(mut self, capacity: u32) -> Self {
        self.capacity = capacity;
        self
    }

    /// Set a debug label for the pool.
    ///
    /// # Arguments
    ///
    /// * `label` - Debug label for graphics debuggers
    pub fn label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Build the TimestampQueryPool.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device
    /// * `queue` - The wgpu queue
    ///
    /// # Returns
    ///
    /// * `Ok(TimestampQueryPool)` - Successfully created pool
    /// * `Err(QueryError)` - Creation failed
    pub fn build(self, device: &Device, queue: &Queue) -> Result<TimestampQueryPool, QueryError> {
        TimestampQueryPool::with_label(device, queue, self.capacity, self.label.as_deref())
    }
}

// ============================================================================
// Utility Functions
// ============================================================================

/// Check if timestamp queries are supported on a device.
///
/// This is a convenience wrapper around `TimestampQueryPool::is_supported`.
#[inline]
pub fn is_timestamp_query_supported(device: &Device) -> bool {
    TimestampQueryPool::is_supported(device)
}

/// Calculate the required resolve buffer size for a given capacity.
///
/// Returns `capacity * 8` bytes (u64 per timestamp).
#[inline]
pub const fn calculate_resolve_buffer_size(capacity: u32) -> u64 {
    (capacity as u64) * TIMESTAMP_SIZE_BYTES
}

/// Convert timestamp ticks to milliseconds.
///
/// # Arguments
///
/// * `ticks` - Number of GPU timestamp ticks
/// * `period_ns` - Timestamp period in nanoseconds (from queue)
#[inline]
pub fn ticks_to_ms(ticks: u64, period_ns: f32) -> f32 {
    let ns = (ticks as f64) * (period_ns as f64);
    (ns / 1_000_000.0) as f32
}

/// Convert timestamp ticks to nanoseconds.
///
/// # Arguments
///
/// * `ticks` - Number of GPU timestamp ticks
/// * `period_ns` - Timestamp period in nanoseconds
#[inline]
pub fn ticks_to_ns(ticks: u64, period_ns: f32) -> f64 {
    (ticks as f64) * (period_ns as f64)
}

/// Convert milliseconds to approximate timestamp ticks.
///
/// # Arguments
///
/// * `ms` - Time in milliseconds
/// * `period_ns` - Timestamp period in nanoseconds
#[inline]
pub fn ms_to_ticks(ms: f32, period_ns: f32) -> u64 {
    let ns = (ms as f64) * 1_000_000.0;
    (ns / (period_ns as f64)) as u64
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ===== SECTION 1: QueryError tests =====

    #[test]
    fn query_error_feature_not_supported_message() {
        let error = QueryError::feature_not_supported();
        let msg = format!("{}", error);
        assert!(msg.contains("TIMESTAMP_QUERY"));
        assert!(msg.contains("not supported"));
    }

    #[test]
    fn query_error_pool_exhausted_shows_capacity() {
        let error = QueryError::pool_exhausted(64);
        let msg = format!("{}", error);
        assert!(msg.contains("64"));
        assert!(msg.contains("exhausted"));
    }

    #[test]
    fn query_error_invalid_index_shows_values() {
        let error = QueryError::invalid_index(100, 64);
        let msg = format!("{}", error);
        assert!(msg.contains("100"));
        assert!(msg.contains("64"));
        assert!(msg.contains("invalid"));
    }

    #[test]
    fn query_error_resolve_failed_shows_reason() {
        let error = QueryError::resolve_failed("buffer mapping timeout");
        let msg = format!("{}", error);
        assert!(msg.contains("buffer mapping timeout"));
    }

    #[test]
    fn query_error_invalid_capacity_shows_value() {
        let error = QueryError::invalid_capacity(0);
        let msg = format!("{}", error);
        assert!(msg.contains("0"));
        assert!(msg.contains("capacity"));
    }

    #[test]
    fn query_error_is_feature_error_returns_true() {
        let error = QueryError::feature_not_supported();
        assert!(error.is_feature_error());
    }

    #[test]
    fn query_error_is_feature_error_returns_false_for_other() {
        let error = QueryError::pool_exhausted(64);
        assert!(!error.is_feature_error());
    }

    #[test]
    fn query_error_is_capacity_error_for_exhausted() {
        let error = QueryError::pool_exhausted(64);
        assert!(error.is_capacity_error());
    }

    #[test]
    fn query_error_is_capacity_error_for_invalid_capacity() {
        let error = QueryError::invalid_capacity(0);
        assert!(error.is_capacity_error());
    }

    #[test]
    fn query_error_is_capacity_error_false_for_feature() {
        let error = QueryError::feature_not_supported();
        assert!(!error.is_capacity_error());
    }

    #[test]
    fn query_error_kind_matches_variant() {
        assert_eq!(
            QueryError::feature_not_supported().kind(),
            QueryErrorKind::FeatureNotSupported
        );
        assert_eq!(
            QueryError::pool_exhausted(64).kind(),
            QueryErrorKind::PoolExhausted
        );
        assert_eq!(
            QueryError::invalid_index(10, 5).kind(),
            QueryErrorKind::InvalidIndex
        );
        assert_eq!(
            QueryError::resolve_failed("test").kind(),
            QueryErrorKind::ResolveFailed
        );
        assert_eq!(
            QueryError::invalid_capacity(0).kind(),
            QueryErrorKind::InvalidCapacity
        );
        assert_eq!(
            QueryError::buffer_creation_failed("test").kind(),
            QueryErrorKind::BufferCreationFailed
        );
    }

    #[test]
    fn query_error_debug_format() {
        let error = QueryError::pool_exhausted(128);
        let debug = format!("{:?}", error);
        assert!(debug.contains("PoolExhausted"));
        assert!(debug.contains("128"));
    }

    #[test]
    fn query_error_clone_equals_original() {
        let error = QueryError::invalid_index(5, 10);
        let cloned = error.clone();
        assert_eq!(error.kind(), cloned.kind());
    }

    // ===== SECTION 2: QueryAllocation tests =====

    #[test]
    fn query_allocation_new_sets_fields() {
        let alloc = QueryAllocation::new(42, 7);
        assert_eq!(alloc.index, 42);
        assert_eq!(alloc.generation, 7);
    }

    #[test]
    fn query_allocation_with_index_default_gen() {
        let alloc = QueryAllocation::with_index(99);
        assert_eq!(alloc.index, 99);
        assert_eq!(alloc.generation, 0);
    }

    #[test]
    fn query_allocation_display_format() {
        let alloc = QueryAllocation::new(10, 3);
        let display = format!("{}", alloc);
        assert!(display.contains("10"));
        assert!(display.contains("3"));
    }

    #[test]
    fn query_allocation_equality() {
        let a = QueryAllocation::new(5, 1);
        let b = QueryAllocation::new(5, 1);
        let c = QueryAllocation::new(5, 2);
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn query_allocation_hash_consistent() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(QueryAllocation::new(1, 0));
        set.insert(QueryAllocation::new(2, 0));
        set.insert(QueryAllocation::new(1, 0)); // Duplicate
        assert_eq!(set.len(), 2);
    }

    // ===== SECTION 3: QueryPoolStats tests =====

    #[test]
    fn query_pool_stats_utilization_empty() {
        let stats = QueryPoolStats {
            capacity: 100,
            used: 0,
            available: 100,
            generation: 0,
            timestamp_period_ns: 1.0,
            resolve_buffer_size: 800,
        };
        assert_eq!(stats.utilization(), 0.0);
        assert!(stats.is_empty());
        assert!(!stats.is_full());
    }

    #[test]
    fn query_pool_stats_utilization_half() {
        let stats = QueryPoolStats {
            capacity: 100,
            used: 50,
            available: 50,
            generation: 0,
            timestamp_period_ns: 1.0,
            resolve_buffer_size: 800,
        };
        assert!((stats.utilization() - 0.5).abs() < 0.001);
        assert!(!stats.is_empty());
        assert!(!stats.is_full());
    }

    #[test]
    fn query_pool_stats_utilization_full() {
        let stats = QueryPoolStats {
            capacity: 100,
            used: 100,
            available: 0,
            generation: 0,
            timestamp_period_ns: 1.0,
            resolve_buffer_size: 800,
        };
        assert_eq!(stats.utilization(), 1.0);
        assert!(!stats.is_empty());
        assert!(stats.is_full());
    }

    #[test]
    fn query_pool_stats_utilization_zero_capacity() {
        let stats = QueryPoolStats {
            capacity: 0,
            used: 0,
            available: 0,
            generation: 0,
            timestamp_period_ns: 1.0,
            resolve_buffer_size: 0,
        };
        assert_eq!(stats.utilization(), 0.0);
    }

    #[test]
    fn query_pool_stats_display() {
        let stats = QueryPoolStats {
            capacity: 64,
            used: 32,
            available: 32,
            generation: 5,
            timestamp_period_ns: 25.0,
            resolve_buffer_size: 512,
        };
        let display = format!("{}", stats);
        assert!(display.contains("32/64"));
        assert!(display.contains("50.0%"));
        assert!(display.contains("gen=5"));
    }

    // ===== SECTION 4: Constants tests =====

    #[test]
    fn timestamp_size_bytes_is_8() {
        assert_eq!(TIMESTAMP_SIZE_BYTES, 8);
    }

    #[test]
    fn min_pool_capacity_is_positive() {
        assert!(MIN_POOL_CAPACITY >= 1);
    }

    #[test]
    fn max_recommended_capacity_is_reasonable() {
        assert!(MAX_RECOMMENDED_CAPACITY >= 256);
        assert!(MAX_RECOMMENDED_CAPACITY <= 65536);
    }

    // ===== SECTION 5: Utility function tests =====

    #[test]
    fn calculate_resolve_buffer_size_basic() {
        assert_eq!(calculate_resolve_buffer_size(1), 8);
        assert_eq!(calculate_resolve_buffer_size(64), 512);
        assert_eq!(calculate_resolve_buffer_size(128), 1024);
    }

    #[test]
    fn calculate_resolve_buffer_size_zero() {
        assert_eq!(calculate_resolve_buffer_size(0), 0);
    }

    #[test]
    fn ticks_to_ms_basic_conversion() {
        // 1 million ticks at 1ns/tick = 1ms
        let ms = ticks_to_ms(1_000_000, 1.0);
        assert!((ms - 1.0).abs() < 0.001);
    }

    #[test]
    fn ticks_to_ms_with_period() {
        // 100,000 ticks at 10ns/tick = 1ms
        let ms = ticks_to_ms(100_000, 10.0);
        assert!((ms - 1.0).abs() < 0.001);
    }

    #[test]
    fn ticks_to_ms_zero_ticks() {
        let ms = ticks_to_ms(0, 1.0);
        assert_eq!(ms, 0.0);
    }

    #[test]
    fn ticks_to_ns_basic() {
        let ns = ticks_to_ns(1000, 1.0);
        assert!((ns - 1000.0).abs() < 0.001);
    }

    #[test]
    fn ticks_to_ns_with_period() {
        let ns = ticks_to_ns(100, 10.0);
        assert!((ns - 1000.0).abs() < 0.001);
    }

    #[test]
    fn ms_to_ticks_basic() {
        // 1ms at 1ns/tick = 1,000,000 ticks
        let ticks = ms_to_ticks(1.0, 1.0);
        assert_eq!(ticks, 1_000_000);
    }

    #[test]
    fn ms_to_ticks_with_period() {
        // 1ms at 10ns/tick = 100,000 ticks
        let ticks = ms_to_ticks(1.0, 10.0);
        assert_eq!(ticks, 100_000);
    }

    #[test]
    fn ticks_ms_roundtrip() {
        let original: u64 = 123456;
        let period: f32 = 25.0;
        let ms = ticks_to_ms(original, period);
        let recovered = ms_to_ticks(ms, period);
        // Allow 1 tick of rounding error
        assert!((recovered as i64 - original as i64).abs() <= 1);
    }

    // ===== SECTION 6: QueryPoolBuilder tests =====

    #[test]
    fn builder_default_capacity() {
        let builder = QueryPoolBuilder::new();
        assert_eq!(builder.capacity, 64);
        assert!(builder.label.is_none());
    }

    #[test]
    fn builder_set_capacity() {
        let builder = QueryPoolBuilder::new().capacity(256);
        assert_eq!(builder.capacity, 256);
    }

    #[test]
    fn builder_set_label() {
        let builder = QueryPoolBuilder::new().label("MyPool");
        assert_eq!(builder.label.as_deref(), Some("MyPool"));
    }

    #[test]
    fn builder_chained() {
        let builder = QueryPoolBuilder::new()
            .capacity(128)
            .label("Chained");
        assert_eq!(builder.capacity, 128);
        assert_eq!(builder.label.as_deref(), Some("Chained"));
    }

    #[test]
    fn builder_default_trait() {
        let builder = QueryPoolBuilder::default();
        assert_eq!(builder.capacity, 64);
    }

    // ===== SECTION 7: QueryErrorKind tests =====

    #[test]
    fn query_error_kind_from_feature_not_supported() {
        let error = QueryError::feature_not_supported();
        assert_eq!(QueryErrorKind::from(&error), QueryErrorKind::FeatureNotSupported);
    }

    #[test]
    fn query_error_kind_from_pool_exhausted() {
        let error = QueryError::pool_exhausted(64);
        assert_eq!(QueryErrorKind::from(&error), QueryErrorKind::PoolExhausted);
    }

    #[test]
    fn query_error_kind_equality() {
        assert_eq!(QueryErrorKind::PoolExhausted, QueryErrorKind::PoolExhausted);
        assert_ne!(QueryErrorKind::PoolExhausted, QueryErrorKind::InvalidIndex);
    }

    #[test]
    fn query_error_kind_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(QueryErrorKind::PoolExhausted);
        set.insert(QueryErrorKind::InvalidIndex);
        set.insert(QueryErrorKind::PoolExhausted); // Duplicate
        assert_eq!(set.len(), 2);
    }

    // ===== SECTION 8: Edge case tests =====

    #[test]
    fn ticks_to_ms_large_value() {
        // Should not overflow
        let ms = ticks_to_ms(u64::MAX, 1.0);
        assert!(ms > 0.0);
        assert!(!ms.is_nan());
        assert!(!ms.is_infinite());
    }

    #[test]
    fn ticks_to_ms_very_small_period() {
        let ms = ticks_to_ms(1_000_000, 0.001);
        assert!(ms > 0.0);
        assert!(ms < 1.0);
    }

    #[test]
    fn query_allocation_copy_derives() {
        let alloc = QueryAllocation::new(1, 2);
        let copied = alloc; // Copy
        assert_eq!(alloc.index, copied.index);
    }

    #[test]
    fn query_pool_stats_copy() {
        let stats = QueryPoolStats {
            capacity: 64,
            used: 32,
            available: 32,
            generation: 0,
            timestamp_period_ns: 1.0,
            resolve_buffer_size: 512,
        };
        let copied = stats;
        assert_eq!(stats.capacity, copied.capacity);
    }

    // ===== SECTION 9: Send + Sync bounds =====

    fn assert_send<T: Send>() {}
    fn assert_sync<T: Sync>() {}

    #[test]
    fn query_error_is_send() {
        assert_send::<QueryError>();
    }

    #[test]
    fn query_error_is_sync() {
        assert_sync::<QueryError>();
    }

    #[test]
    fn query_allocation_is_send() {
        assert_send::<QueryAllocation>();
    }

    #[test]
    fn query_allocation_is_sync() {
        assert_sync::<QueryAllocation>();
    }

    #[test]
    fn query_pool_stats_is_send() {
        assert_send::<QueryPoolStats>();
    }

    #[test]
    fn query_pool_stats_is_sync() {
        assert_sync::<QueryPoolStats>();
    }

    #[test]
    fn query_pool_builder_is_send() {
        assert_send::<QueryPoolBuilder>();
    }

    #[test]
    fn query_pool_builder_is_sync() {
        assert_sync::<QueryPoolBuilder>();
    }

    // ===== SECTION 10: Integration-like tests (without actual device) =====
    // These test the logic without requiring a real wgpu device

    #[test]
    fn buffer_size_calculation_matches_constant() {
        // Verify the formula matches our constant
        let capacity: u32 = 100;
        let expected = (capacity as u64) * 8; // u64 per timestamp
        let actual = calculate_resolve_buffer_size(capacity);
        assert_eq!(expected, actual);
    }

    #[test]
    fn resolve_buffer_alignment() {
        // Verify buffer size is u64-aligned for various capacities
        for capacity in [1, 7, 15, 63, 100, 127, 255, 1000] {
            let size = calculate_resolve_buffer_size(capacity);
            assert_eq!(size % 8, 0, "Buffer size {} not u64-aligned", size);
        }
    }

    // ===== SECTION 11: Mock-based pool tests =====
    // These test pool logic using a mock structure

    struct MockPool {
        capacity: u32,
        next_index: u32,
        generation: u32,
    }

    impl MockPool {
        fn new(capacity: u32) -> Self {
            Self {
                capacity,
                next_index: 0,
                generation: 0,
            }
        }

        fn allocate(&mut self) -> Result<u32, QueryError> {
            if self.next_index >= self.capacity {
                return Err(QueryError::pool_exhausted(self.capacity));
            }
            let index = self.next_index;
            self.next_index += 1;
            Ok(index)
        }

        fn allocate_range(&mut self, count: u32) -> Result<u32, QueryError> {
            if count == 0 {
                return Ok(self.next_index);
            }
            let end = self.next_index.checked_add(count)
                .ok_or_else(|| QueryError::pool_exhausted(self.capacity))?;
            if end > self.capacity {
                return Err(QueryError::pool_exhausted(self.capacity));
            }
            let start = self.next_index;
            self.next_index = end;
            Ok(start)
        }

        fn reset(&mut self) {
            self.next_index = 0;
            self.generation = self.generation.wrapping_add(1);
        }

        fn available(&self) -> u32 {
            self.capacity.saturating_sub(self.next_index)
        }

        fn used(&self) -> u32 {
            self.next_index
        }

        fn has_capacity(&self) -> bool {
            self.next_index < self.capacity
        }

        fn is_empty(&self) -> bool {
            self.next_index == 0
        }

        fn is_full(&self) -> bool {
            self.next_index >= self.capacity
        }

        fn validate_index(&self, index: u32) -> Result<(), QueryError> {
            if index >= self.capacity {
                return Err(QueryError::invalid_index(index, self.capacity));
            }
            Ok(())
        }
    }

    #[test]
    fn mock_pool_allocate_sequential() {
        let mut pool = MockPool::new(5);
        assert_eq!(pool.allocate().unwrap(), 0);
        assert_eq!(pool.allocate().unwrap(), 1);
        assert_eq!(pool.allocate().unwrap(), 2);
        assert_eq!(pool.used(), 3);
        assert_eq!(pool.available(), 2);
    }

    #[test]
    fn mock_pool_allocate_until_exhausted() {
        let mut pool = MockPool::new(3);
        assert!(pool.allocate().is_ok());
        assert!(pool.allocate().is_ok());
        assert!(pool.allocate().is_ok());
        assert!(pool.allocate().is_err());
        assert!(pool.is_full());
    }

    #[test]
    fn mock_pool_allocate_range_success() {
        let mut pool = MockPool::new(10);
        let start = pool.allocate_range(3).unwrap();
        assert_eq!(start, 0);
        assert_eq!(pool.used(), 3);
    }

    #[test]
    fn mock_pool_allocate_range_exhausted() {
        let mut pool = MockPool::new(5);
        pool.allocate_range(3).unwrap();
        let result = pool.allocate_range(3);
        assert!(result.is_err());
    }

    #[test]
    fn mock_pool_allocate_range_zero() {
        let mut pool = MockPool::new(10);
        pool.allocate().unwrap();
        let start = pool.allocate_range(0).unwrap();
        assert_eq!(start, 1); // Returns current index
        assert_eq!(pool.used(), 1); // No change
    }

    #[test]
    fn mock_pool_reset_clears_allocation() {
        let mut pool = MockPool::new(5);
        pool.allocate().unwrap();
        pool.allocate().unwrap();
        assert_eq!(pool.used(), 2);
        pool.reset();
        assert_eq!(pool.used(), 0);
        assert!(pool.is_empty());
    }

    #[test]
    fn mock_pool_reset_increments_generation() {
        let mut pool = MockPool::new(5);
        assert_eq!(pool.generation, 0);
        pool.reset();
        assert_eq!(pool.generation, 1);
        pool.reset();
        assert_eq!(pool.generation, 2);
    }

    #[test]
    fn mock_pool_reset_allows_reallocation() {
        let mut pool = MockPool::new(2);
        pool.allocate().unwrap();
        pool.allocate().unwrap();
        assert!(pool.is_full());
        pool.reset();
        assert!(pool.is_empty());
        assert_eq!(pool.allocate().unwrap(), 0);
    }

    #[test]
    fn mock_pool_has_capacity_true_when_available() {
        let pool = MockPool::new(5);
        assert!(pool.has_capacity());
    }

    #[test]
    fn mock_pool_has_capacity_false_when_full() {
        let mut pool = MockPool::new(1);
        pool.allocate().unwrap();
        assert!(!pool.has_capacity());
    }

    #[test]
    fn mock_pool_validate_index_in_bounds() {
        let pool = MockPool::new(10);
        assert!(pool.validate_index(0).is_ok());
        assert!(pool.validate_index(9).is_ok());
    }

    #[test]
    fn mock_pool_validate_index_out_of_bounds() {
        let pool = MockPool::new(10);
        assert!(pool.validate_index(10).is_err());
        assert!(pool.validate_index(100).is_err());
    }

    #[test]
    fn mock_pool_is_empty_initially() {
        let pool = MockPool::new(10);
        assert!(pool.is_empty());
        assert!(!pool.is_full());
    }

    #[test]
    fn mock_pool_is_full_when_exhausted() {
        let mut pool = MockPool::new(2);
        pool.allocate().unwrap();
        pool.allocate().unwrap();
        assert!(!pool.is_empty());
        assert!(pool.is_full());
    }

    // ===== SECTION 12: Additional edge cases =====

    #[test]
    fn mock_pool_capacity_1() {
        let mut pool = MockPool::new(1);
        assert!(pool.has_capacity());
        assert_eq!(pool.allocate().unwrap(), 0);
        assert!(!pool.has_capacity());
        assert!(pool.is_full());
    }

    #[test]
    fn mock_pool_large_capacity() {
        let pool = MockPool::new(10000);
        assert_eq!(pool.available(), 10000);
    }

    #[test]
    fn mock_pool_generation_wraps() {
        let mut pool = MockPool::new(1);
        pool.generation = u32::MAX;
        pool.reset();
        assert_eq!(pool.generation, 0);
    }

    #[test]
    fn mock_pool_allocate_range_exactly_fills() {
        let mut pool = MockPool::new(5);
        pool.allocate_range(5).unwrap();
        assert!(pool.is_full());
    }

    // ===== SECTION 13: QueryResolveParams tests =====

    #[test]
    fn query_resolve_params_new_sets_all_fields() {
        let params = QueryResolveParams::new(5, 10, 40);
        assert_eq!(params.start_query, 5);
        assert_eq!(params.query_count, 10);
        assert_eq!(params.destination_offset, 40);
    }

    #[test]
    fn query_resolve_params_from_start() {
        let params = QueryResolveParams::from_start(8);
        assert_eq!(params.start_query, 0);
        assert_eq!(params.query_count, 8);
        assert_eq!(params.destination_offset, 0);
    }

    #[test]
    fn query_resolve_params_end_query() {
        let params = QueryResolveParams::new(10, 5, 0);
        assert_eq!(params.end_query(), 15);
    }

    #[test]
    fn query_resolve_params_end_query_zero_count() {
        let params = QueryResolveParams::new(10, 0, 0);
        assert_eq!(params.end_query(), 10);
    }

    #[test]
    fn query_resolve_params_required_buffer_size() {
        // 4 queries * 8 bytes + offset 0 = 32 bytes
        let params = QueryResolveParams::new(0, 4, 0);
        assert_eq!(params.required_buffer_size(), 32);
    }

    #[test]
    fn query_resolve_params_required_buffer_size_with_offset() {
        // 4 queries * 8 bytes + offset 32 = 64 bytes
        let params = QueryResolveParams::new(4, 4, 32);
        assert_eq!(params.required_buffer_size(), 64);
    }

    #[test]
    fn query_resolve_params_default() {
        let params = QueryResolveParams::default();
        assert_eq!(params.start_query, 0);
        assert_eq!(params.query_count, 0);
        assert_eq!(params.destination_offset, 0);
    }

    #[test]
    fn query_resolve_params_display() {
        let params = QueryResolveParams::new(2, 4, 16);
        let display = format!("{}", params);
        assert!(display.contains("2..6"));
        assert!(display.contains("16"));
    }

    #[test]
    fn query_resolve_params_equality() {
        let a = QueryResolveParams::new(0, 4, 0);
        let b = QueryResolveParams::new(0, 4, 0);
        let c = QueryResolveParams::new(1, 4, 0);
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn query_resolve_params_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(QueryResolveParams::new(0, 4, 0));
        set.insert(QueryResolveParams::new(0, 8, 0));
        set.insert(QueryResolveParams::new(0, 4, 0)); // Duplicate
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn query_resolve_params_is_send() {
        assert_send::<QueryResolveParams>();
    }

    #[test]
    fn query_resolve_params_is_sync() {
        assert_sync::<QueryResolveParams>();
    }

    #[test]
    fn query_resolve_params_copy() {
        let params = QueryResolveParams::new(1, 2, 8);
        let copied = params; // Copy trait
        assert_eq!(params.start_query, copied.start_query);
        assert_eq!(params.query_count, copied.query_count);
        assert_eq!(params.destination_offset, copied.destination_offset);
    }

    // ===== SECTION 14: MockPool with resolve validation =====

    /// Mock pool that simulates resolve validation logic without wgpu device
    struct MockPoolWithResolve {
        capacity: u32,
        next_index: u32,
        buffer_size: u64,
    }

    impl MockPoolWithResolve {
        fn new(capacity: u32) -> Self {
            Self {
                capacity,
                next_index: 0,
                buffer_size: (capacity as u64) * TIMESTAMP_SIZE_BYTES,
            }
        }

        fn used(&self) -> u32 {
            self.next_index
        }

        fn allocate(&mut self) -> Result<u32, QueryError> {
            if self.next_index >= self.capacity {
                return Err(QueryError::pool_exhausted(self.capacity));
            }
            let idx = self.next_index;
            self.next_index += 1;
            Ok(idx)
        }

        fn validate_resolve_params(&self, params: &QueryResolveParams) -> Result<(), QueryError> {
            // Zero count check
            if params.query_count == 0 {
                return Err(QueryError::resolve_failed("query_count must be greater than 0"));
            }

            // Overflow check
            let end_query = params.start_query.checked_add(params.query_count)
                .ok_or_else(|| QueryError::resolve_failed("query range overflow"))?;

            // Bounds check
            if end_query > self.capacity {
                return Err(QueryError::resolve_failed(format!(
                    "query range out of bounds: {} > {}",
                    end_query, self.capacity
                )));
            }

            // Buffer size check
            let required_size = params.required_buffer_size();
            if required_size > self.buffer_size {
                return Err(QueryError::resolve_failed(format!(
                    "destination overflow: {} > {}",
                    required_size, self.buffer_size
                )));
            }

            Ok(())
        }

        fn resolve(&self, params: QueryResolveParams) -> Result<(), QueryError> {
            self.validate_resolve_params(&params)?;
            // In real impl, would call encoder.resolve_query_set()
            Ok(())
        }

        fn resolve_all(&self) -> Result<(), QueryError> {
            let count = self.used();
            if count == 0 {
                return Ok(());
            }
            let params = QueryResolveParams::from_start(count);
            self.resolve(params)
        }

        fn resolve_first(&self, count: u32) -> Result<(), QueryError> {
            if count == 0 {
                return Err(QueryError::resolve_failed("count must be greater than 0"));
            }
            let params = QueryResolveParams::from_start(count);
            self.resolve(params)
        }
    }

    // ===== SECTION 15: Resolve validation tests =====

    #[test]
    fn resolve_valid_params_succeeds() {
        let pool = MockPoolWithResolve::new(64);
        let params = QueryResolveParams::new(0, 4, 0);
        assert!(pool.resolve(params).is_ok());
    }

    #[test]
    fn resolve_full_capacity_succeeds() {
        let pool = MockPoolWithResolve::new(64);
        let params = QueryResolveParams::new(0, 64, 0);
        assert!(pool.resolve(params).is_ok());
    }

    #[test]
    fn resolve_with_offset_succeeds() {
        let pool = MockPoolWithResolve::new(64);
        // Resolve queries 32-64 to offset 0 (fits in 64 * 8 = 512 bytes)
        let params = QueryResolveParams::new(32, 32, 0);
        assert!(pool.resolve(params).is_ok());
    }

    #[test]
    fn resolve_all_empty_pool_succeeds() {
        let pool = MockPoolWithResolve::new(64);
        // Empty pool should be a no-op success
        assert!(pool.resolve_all().is_ok());
    }

    #[test]
    fn resolve_all_with_allocations() {
        let mut pool = MockPoolWithResolve::new(64);
        pool.allocate().unwrap();
        pool.allocate().unwrap();
        pool.allocate().unwrap();
        assert!(pool.resolve_all().is_ok());
    }

    #[test]
    fn resolve_first_valid_count() {
        let pool = MockPoolWithResolve::new(64);
        assert!(pool.resolve_first(4).is_ok());
    }

    #[test]
    fn resolve_first_full_capacity() {
        let pool = MockPoolWithResolve::new(64);
        assert!(pool.resolve_first(64).is_ok());
    }

    #[test]
    fn resolve_zero_count_fails() {
        let pool = MockPoolWithResolve::new(64);
        let params = QueryResolveParams::new(0, 0, 0);
        let err = pool.resolve(params).unwrap_err();
        assert!(matches!(err, QueryError::ResolveFailed { .. }));
        assert!(format!("{}", err).contains("must be greater than 0"));
    }

    #[test]
    fn resolve_first_zero_count_fails() {
        let pool = MockPoolWithResolve::new(64);
        let err = pool.resolve_first(0).unwrap_err();
        assert!(matches!(err, QueryError::ResolveFailed { .. }));
    }

    #[test]
    fn resolve_start_query_out_of_bounds_fails() {
        let pool = MockPoolWithResolve::new(64);
        // start=64 is already out of bounds (capacity is 64, indices 0-63)
        let params = QueryResolveParams::new(64, 1, 0);
        let err = pool.resolve(params).unwrap_err();
        assert!(matches!(err, QueryError::ResolveFailed { .. }));
        assert!(format!("{}", err).contains("out of bounds"));
    }

    #[test]
    fn resolve_query_count_exceeds_capacity_fails() {
        let pool = MockPoolWithResolve::new(64);
        // start=0, count=65 exceeds capacity 64
        let params = QueryResolveParams::new(0, 65, 0);
        let err = pool.resolve(params).unwrap_err();
        assert!(matches!(err, QueryError::ResolveFailed { .. }));
    }

    #[test]
    fn resolve_start_plus_count_exceeds_capacity_fails() {
        let pool = MockPoolWithResolve::new(64);
        // start=60, count=10 = 70 > 64
        let params = QueryResolveParams::new(60, 10, 0);
        let err = pool.resolve(params).unwrap_err();
        assert!(matches!(err, QueryError::ResolveFailed { .. }));
    }

    #[test]
    fn resolve_destination_offset_overflow_fails() {
        let pool = MockPoolWithResolve::new(64); // buffer is 64 * 8 = 512 bytes
        // offset=500 + 4*8=32 = 532 > 512
        let params = QueryResolveParams::new(0, 4, 500);
        let err = pool.resolve(params).unwrap_err();
        assert!(matches!(err, QueryError::ResolveFailed { .. }));
        assert!(format!("{}", err).contains("overflow"));
    }

    #[test]
    fn resolve_exact_buffer_fit_succeeds() {
        let pool = MockPoolWithResolve::new(64); // buffer is 512 bytes
        // offset=0 + 64*8=512 = exactly 512 bytes
        let params = QueryResolveParams::new(0, 64, 0);
        assert!(pool.resolve(params).is_ok());
    }

    #[test]
    fn resolve_offset_at_buffer_end_fails() {
        let pool = MockPoolWithResolve::new(64); // buffer is 512 bytes
        // offset=512 + 1*8=8 = 520 > 512
        let params = QueryResolveParams::new(0, 1, 512);
        let err = pool.resolve(params).unwrap_err();
        assert!(matches!(err, QueryError::ResolveFailed { .. }));
    }

    #[test]
    fn resolve_partial_range_with_offset() {
        let pool = MockPoolWithResolve::new(64); // buffer is 512 bytes
        // Resolve queries 4-8 (4 queries) to offset 32 (32 + 32 = 64 bytes, fits)
        let params = QueryResolveParams::new(4, 4, 32);
        assert!(pool.resolve(params).is_ok());
    }

    #[test]
    fn validate_resolve_params_valid() {
        let pool = MockPoolWithResolve::new(64);
        let params = QueryResolveParams::new(0, 4, 0);
        assert!(pool.validate_resolve_params(&params).is_ok());
    }

    #[test]
    fn validate_resolve_params_invalid_count() {
        let pool = MockPoolWithResolve::new(64);
        let params = QueryResolveParams::new(0, 0, 0);
        assert!(pool.validate_resolve_params(&params).is_err());
    }

    #[test]
    fn validate_resolve_params_invalid_bounds() {
        let pool = MockPoolWithResolve::new(64);
        let params = QueryResolveParams::new(60, 10, 0);
        assert!(pool.validate_resolve_params(&params).is_err());
    }

    #[test]
    fn validate_resolve_params_invalid_offset() {
        let pool = MockPoolWithResolve::new(64);
        let params = QueryResolveParams::new(0, 4, 500);
        assert!(pool.validate_resolve_params(&params).is_err());
    }

    // ===== SECTION 16: Edge cases for resolve =====

    #[test]
    fn resolve_capacity_1_pool() {
        let pool = MockPoolWithResolve::new(1);
        let params = QueryResolveParams::new(0, 1, 0);
        assert!(pool.resolve(params).is_ok());
    }

    #[test]
    fn resolve_capacity_1_pool_out_of_bounds() {
        let pool = MockPoolWithResolve::new(1);
        let params = QueryResolveParams::new(1, 1, 0);
        assert!(pool.resolve(params).is_err());
    }

    #[test]
    fn resolve_large_capacity() {
        let pool = MockPoolWithResolve::new(8192);
        let params = QueryResolveParams::new(0, 8192, 0);
        assert!(pool.resolve(params).is_ok());
    }

    #[test]
    fn resolve_query_range_overflow_u32_max() {
        let pool = MockPoolWithResolve::new(64);
        // This would overflow: u32::MAX + 1
        let params = QueryResolveParams::new(u32::MAX, 1, 0);
        let err = pool.resolve(params).unwrap_err();
        assert!(matches!(err, QueryError::ResolveFailed { .. }));
    }

    #[test]
    fn resolve_params_required_size_no_overflow() {
        // Test that required_buffer_size doesn't overflow for reasonable values
        let params = QueryResolveParams::new(0, 1_000_000, 0);
        let size = params.required_buffer_size();
        assert_eq!(size, 8_000_000); // 1M queries * 8 bytes
    }

    #[test]
    fn resolve_first_exactly_at_capacity() {
        let pool = MockPoolWithResolve::new(64);
        assert!(pool.resolve_first(64).is_ok());
    }

    #[test]
    fn resolve_first_exceeds_capacity() {
        let pool = MockPoolWithResolve::new(64);
        let err = pool.resolve_first(65).unwrap_err();
        assert!(matches!(err, QueryError::ResolveFailed { .. }));
    }

    #[test]
    fn resolve_all_after_allocating_full_capacity() {
        let mut pool = MockPoolWithResolve::new(4);
        pool.allocate().unwrap();
        pool.allocate().unwrap();
        pool.allocate().unwrap();
        pool.allocate().unwrap();
        assert!(pool.resolve_all().is_ok());
    }

    // ===== SECTION 17: TimestampResult tests =====

    #[test]
    fn timestamp_result_new_calculates_correctly() {
        let result = TimestampResult::new(1000, 2000, 25.0);
        assert_eq!(result.start_ticks, 1000);
        assert_eq!(result.end_ticks, 2000);
        assert_eq!(result.duration_ticks, 1000);
        // 1000 ticks * 25 ns/tick = 25000 ns
        assert!((result.duration_ns - 25000.0).abs() < 0.001);
        // 25000 ns = 0.025 ms
        assert!((result.duration_ms - 0.025).abs() < 0.0001);
    }

    #[test]
    fn timestamp_result_new_with_zero_duration() {
        let result = TimestampResult::new(1000, 1000, 25.0);
        assert_eq!(result.duration_ticks, 0);
        assert_eq!(result.duration_ns, 0.0);
        assert_eq!(result.duration_ms, 0.0);
    }

    #[test]
    fn timestamp_result_saturating_sub_when_end_before_start() {
        // When end < start, should saturate to 0 (not panic or wrap)
        let result = TimestampResult::new(2000, 1000, 25.0);
        assert_eq!(result.duration_ticks, 0);
        assert_eq!(result.duration_ns, 0.0);
        assert_eq!(result.duration_ms, 0.0);
    }

    #[test]
    fn timestamp_result_zero() {
        let result = TimestampResult::zero();
        assert_eq!(result.start_ticks, 0);
        assert_eq!(result.end_ticks, 0);
        assert_eq!(result.duration_ticks, 0);
        assert_eq!(result.duration_ns, 0.0);
        assert_eq!(result.duration_ms, 0.0);
    }

    #[test]
    fn timestamp_result_is_valid_positive_duration() {
        let result = TimestampResult::new(0, 100, 1.0);
        assert!(result.is_valid());
    }

    #[test]
    fn timestamp_result_is_valid_zero_duration() {
        let result = TimestampResult::zero();
        assert!(!result.is_valid());
    }

    #[test]
    fn timestamp_result_duration_us() {
        let result = TimestampResult::new(0, 1_000_000, 1.0);
        // 1M ticks * 1 ns/tick = 1M ns = 1000 us
        assert!((result.duration_us() - 1000.0).abs() < 0.001);
    }

    #[test]
    fn timestamp_result_duration_secs() {
        let result = TimestampResult::new(0, 1_000_000_000, 1.0);
        // 1B ticks * 1 ns/tick = 1B ns = 1000 ms = 1 sec
        assert!((result.duration_secs() - 1.0).abs() < 0.001);
    }

    #[test]
    fn timestamp_result_default_is_zero() {
        let result = TimestampResult::default();
        assert_eq!(result, TimestampResult::zero());
    }

    #[test]
    fn timestamp_result_display_format() {
        let result = TimestampResult::new(100, 200, 10.0);
        let display = format!("{}", result);
        assert!(display.contains("100"));
        assert!(display.contains("200"));
        assert!(display.contains("ms"));
        assert!(display.contains("ticks"));
    }

    #[test]
    fn timestamp_result_copy_clone() {
        let result = TimestampResult::new(1, 2, 1.0);
        let copied = result; // Copy
        let cloned = result.clone();
        assert_eq!(result.start_ticks, copied.start_ticks);
        assert_eq!(result.start_ticks, cloned.start_ticks);
    }

    #[test]
    fn timestamp_result_large_values() {
        let result = TimestampResult::new(0, u64::MAX / 2, 1.0);
        assert!(result.is_valid());
        assert!(result.duration_ns > 0.0);
    }

    #[test]
    fn timestamp_result_with_different_periods() {
        // Same tick delta, different periods
        let result_1ns = TimestampResult::new(0, 1000, 1.0);
        let result_10ns = TimestampResult::new(0, 1000, 10.0);
        let result_25ns = TimestampResult::new(0, 1000, 25.0);

        // 1000 * 1 = 1000 ns = 0.001 ms
        assert!((result_1ns.duration_ms - 0.001).abs() < 0.0001);
        // 1000 * 10 = 10000 ns = 0.01 ms
        assert!((result_10ns.duration_ms - 0.01).abs() < 0.0001);
        // 1000 * 25 = 25000 ns = 0.025 ms
        assert!((result_25ns.duration_ms - 0.025).abs() < 0.0001);
    }

    // ===== SECTION 18: ProfileResult tests =====

    #[test]
    fn profile_result_new_without_label() {
        let ts = TimestampResult::new(0, 100, 1.0);
        let profile = ProfileResult::new(ts);
        assert!(profile.label.is_none());
        assert!(!profile.has_label());
    }

    #[test]
    fn profile_result_with_label() {
        let ts = TimestampResult::new(0, 100, 1.0);
        let profile = ProfileResult::with_label("Shadow Pass", ts);
        assert_eq!(profile.label.as_deref(), Some("Shadow Pass"));
        assert!(profile.has_label());
    }

    #[test]
    fn profile_result_label_or_default() {
        let ts = TimestampResult::zero();
        let with_label = ProfileResult::with_label("Test", ts);
        let without_label = ProfileResult::new(ts);

        assert_eq!(with_label.label_or("default"), "Test");
        assert_eq!(without_label.label_or("default"), "default");
    }

    #[test]
    fn profile_result_label_or_unnamed() {
        let ts = TimestampResult::zero();
        let with_label = ProfileResult::with_label("Named", ts);
        let without_label = ProfileResult::new(ts);

        assert_eq!(with_label.label_or_unnamed(), "Named");
        assert_eq!(without_label.label_or_unnamed(), "unnamed");
    }

    #[test]
    fn profile_result_default() {
        let profile = ProfileResult::default();
        assert!(profile.label.is_none());
        assert_eq!(profile.result.duration_ticks, 0);
    }

    #[test]
    fn profile_result_display_with_label() {
        let ts = TimestampResult::new(0, 1000, 1.0);
        let profile = ProfileResult::with_label("Main Pass", ts);
        let display = format!("{}", profile);
        assert!(display.contains("Main Pass"));
        assert!(display.contains("ms"));
    }

    #[test]
    fn profile_result_display_without_label() {
        let ts = TimestampResult::new(0, 1000, 1.0);
        let profile = ProfileResult::new(ts);
        let display = format!("{}", profile);
        assert!(display.contains("ms"));
        assert!(!display.contains(":"));
    }

    #[test]
    fn profile_result_clone() {
        let ts = TimestampResult::new(0, 100, 1.0);
        let profile = ProfileResult::with_label("Test", ts);
        let cloned = profile.clone();
        assert_eq!(profile.label, cloned.label);
        assert_eq!(profile.result.duration_ticks, cloned.result.duration_ticks);
    }

    #[test]
    fn profile_result_equality() {
        let ts = TimestampResult::new(0, 100, 1.0);
        let a = ProfileResult::with_label("Test", ts);
        let b = ProfileResult::with_label("Test", ts);
        let c = ProfileResult::with_label("Different", ts);
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    // ===== SECTION 19: TimestampData tests =====

    #[test]
    fn timestamp_data_new() {
        let data = TimestampData::new(vec![100, 200, 300, 400], 25.0);
        assert_eq!(data.len(), 4);
        assert_eq!(data.timestamp_period, 25.0);
    }

    #[test]
    fn timestamp_data_empty() {
        let data = TimestampData::empty(25.0);
        assert!(data.is_empty());
        assert_eq!(data.len(), 0);
        assert_eq!(data.timestamp_period, 25.0);
    }

    #[test]
    fn timestamp_data_get() {
        let data = TimestampData::new(vec![100, 200, 300], 1.0);
        assert_eq!(data.get(0), Some(100));
        assert_eq!(data.get(1), Some(200));
        assert_eq!(data.get(2), Some(300));
        assert_eq!(data.get(3), None);
    }

    #[test]
    fn timestamp_data_pair_count() {
        let data_4 = TimestampData::new(vec![1, 2, 3, 4], 1.0);
        let data_5 = TimestampData::new(vec![1, 2, 3, 4, 5], 1.0);
        let data_0 = TimestampData::empty(1.0);

        assert_eq!(data_4.pair_count(), 2);
        assert_eq!(data_5.pair_count(), 2); // 5/2 = 2 (integer division)
        assert_eq!(data_0.pair_count(), 0);
    }

    #[test]
    fn timestamp_data_get_pair() {
        let data = TimestampData::new(vec![100, 200, 300, 400], 1.0);
        assert_eq!(data.get_pair(0), Some((100, 200)));
        assert_eq!(data.get_pair(1), Some((300, 400)));
        assert_eq!(data.get_pair(2), None);
    }

    #[test]
    fn timestamp_data_get_pair_incomplete() {
        // Only 3 timestamps = 1 complete pair, second pair incomplete
        let data = TimestampData::new(vec![100, 200, 300], 1.0);
        assert_eq!(data.get_pair(0), Some((100, 200)));
        assert_eq!(data.get_pair(1), None); // Incomplete pair
    }

    #[test]
    fn timestamp_data_duration_ms() {
        let data = TimestampData::new(vec![0, 1_000_000], 1.0);
        // 1M ticks * 1 ns/tick = 1M ns = 1 ms
        let duration = data.duration_ms(0, 1);
        assert!(duration.is_some());
        assert!((duration.unwrap() - 1.0).abs() < 0.001);
    }

    #[test]
    fn timestamp_data_duration_ms_out_of_bounds() {
        let data = TimestampData::new(vec![100, 200], 1.0);
        assert!(data.duration_ms(0, 2).is_none());
        assert!(data.duration_ms(3, 4).is_none());
    }

    #[test]
    fn timestamp_data_default() {
        let data = TimestampData::default();
        assert!(data.is_empty());
        assert_eq!(data.timestamp_period, 1.0);
    }

    // ===== SECTION 20: Query range validation tests =====

    /// Mock structure for testing query range validation
    struct MockQueryRangeValidator {
        capacity: u32,
    }

    impl MockQueryRangeValidator {
        fn new(capacity: u32) -> Self {
            Self { capacity }
        }

        fn validate_query_range(&self, query_range: &std::ops::Range<u32>) -> Result<(), QueryError> {
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

            Ok(())
        }

        fn calculate_buffer_range(&self, query_range: &std::ops::Range<u32>) -> (u64, u64) {
            let offset = (query_range.start as u64) * TIMESTAMP_SIZE_BYTES;
            let size = ((query_range.end - query_range.start) as u64) * TIMESTAMP_SIZE_BYTES;
            (offset, size)
        }
    }

    #[test]
    fn validate_range_valid() {
        let validator = MockQueryRangeValidator::new(64);
        assert!(validator.validate_query_range(&(0..4)).is_ok());
        assert!(validator.validate_query_range(&(0..64)).is_ok());
        assert!(validator.validate_query_range(&(32..64)).is_ok());
    }

    #[test]
    fn validate_range_empty_fails() {
        let validator = MockQueryRangeValidator::new(64);
        assert!(validator.validate_query_range(&(0..0)).is_err());
        assert!(validator.validate_query_range(&(5..5)).is_err());
    }

    #[test]
    fn validate_range_out_of_bounds_fails() {
        let validator = MockQueryRangeValidator::new(64);
        assert!(validator.validate_query_range(&(0..65)).is_err());
        assert!(validator.validate_query_range(&(60..70)).is_err());
        assert!(validator.validate_query_range(&(64..65)).is_err());
    }

    #[test]
    fn calculate_buffer_range_at_start() {
        let validator = MockQueryRangeValidator::new(64);
        let (offset, size) = validator.calculate_buffer_range(&(0..4));
        assert_eq!(offset, 0);
        assert_eq!(size, 32); // 4 queries * 8 bytes
    }

    #[test]
    fn calculate_buffer_range_with_offset() {
        let validator = MockQueryRangeValidator::new(64);
        let (offset, size) = validator.calculate_buffer_range(&(8..12));
        assert_eq!(offset, 64); // 8 * 8 bytes
        assert_eq!(size, 32); // 4 queries * 8 bytes
    }

    #[test]
    fn calculate_buffer_range_full_capacity() {
        let validator = MockQueryRangeValidator::new(64);
        let (offset, size) = validator.calculate_buffer_range(&(0..64));
        assert_eq!(offset, 0);
        assert_eq!(size, 512); // 64 * 8 bytes
    }

    // ===== SECTION 21: Additional error type tests =====

    #[test]
    fn query_error_buffer_mapping_failed_message() {
        let error = QueryError::buffer_mapping_failed("timeout");
        let msg = format!("{}", error);
        assert!(msg.contains("timeout"));
        assert!(msg.contains("mapping failed"));
    }

    #[test]
    fn query_error_invalid_query_range_message() {
        let error = QueryError::invalid_query_range(0, 100, 64);
        let msg = format!("{}", error);
        assert!(msg.contains("0..100"));
        assert!(msg.contains("64"));
    }

    #[test]
    fn query_error_readback_not_ready_message() {
        let error = QueryError::readback_not_ready("buffer not mapped");
        let msg = format!("{}", error);
        assert!(msg.contains("not ready"));
        assert!(msg.contains("buffer not mapped"));
    }

    #[test]
    fn query_error_kind_buffer_mapping_failed() {
        let error = QueryError::buffer_mapping_failed("test");
        assert_eq!(error.kind(), QueryErrorKind::BufferMappingFailed);
    }

    #[test]
    fn query_error_kind_invalid_query_range() {
        let error = QueryError::invalid_query_range(0, 10, 5);
        assert_eq!(error.kind(), QueryErrorKind::InvalidQueryRange);
    }

    #[test]
    fn query_error_kind_readback_not_ready() {
        let error = QueryError::readback_not_ready("test");
        assert_eq!(error.kind(), QueryErrorKind::ReadbackNotReady);
    }

    // ===== SECTION 22: Timestamp calculation integration tests =====

    /// Mock pool for testing timestamp calculations without wgpu
    struct MockTimestampPool {
        timestamp_period: f32,
        capacity: u32,
    }

    impl MockTimestampPool {
        fn new(capacity: u32, timestamp_period: f32) -> Self {
            Self { capacity, timestamp_period }
        }

        fn calculate_duration(&self, start_ticks: u64, end_ticks: u64) -> TimestampResult {
            TimestampResult::new(start_ticks, end_ticks, self.timestamp_period)
        }

        fn parse_timestamp_pairs(
            &self,
            data: &TimestampData,
            labels: Option<&[&str]>,
        ) -> Vec<ProfileResult> {
            let pair_count = data.pair_count();
            let mut results = Vec::with_capacity(pair_count);

            for i in 0..pair_count {
                if let Some((start, end)) = data.get_pair(i) {
                    let timestamp_result = self.calculate_duration(start, end);
                    let label = labels.and_then(|l| l.get(i).map(|s| s.to_string()));

                    results.push(ProfileResult {
                        label,
                        result: timestamp_result,
                    });
                }
            }

            results
        }
    }

    #[test]
    fn mock_pool_calculate_duration() {
        let pool = MockTimestampPool::new(64, 25.0);
        let result = pool.calculate_duration(0, 40000);
        // 40000 ticks * 25 ns = 1,000,000 ns = 1 ms
        assert!((result.duration_ms - 1.0).abs() < 0.001);
    }

    #[test]
    fn mock_pool_parse_pairs_no_labels() {
        let pool = MockTimestampPool::new(64, 1.0);
        let data = TimestampData::new(vec![0, 1000, 2000, 3000], 1.0);
        let results = pool.parse_timestamp_pairs(&data, None);

        assert_eq!(results.len(), 2);
        assert!(results[0].label.is_none());
        assert!(results[1].label.is_none());
        assert_eq!(results[0].result.duration_ticks, 1000);
        assert_eq!(results[1].result.duration_ticks, 1000);
    }

    #[test]
    fn mock_pool_parse_pairs_with_labels() {
        let pool = MockTimestampPool::new(64, 1.0);
        let data = TimestampData::new(vec![0, 100, 200, 400], 1.0);
        let labels = ["Pass A", "Pass B"];
        let results = pool.parse_timestamp_pairs(&data, Some(&labels));

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].label.as_deref(), Some("Pass A"));
        assert_eq!(results[1].label.as_deref(), Some("Pass B"));
        assert_eq!(results[0].result.duration_ticks, 100);
        assert_eq!(results[1].result.duration_ticks, 200);
    }

    #[test]
    fn mock_pool_parse_pairs_fewer_labels_than_pairs() {
        let pool = MockTimestampPool::new(64, 1.0);
        let data = TimestampData::new(vec![0, 100, 200, 300, 400, 500], 1.0);
        let labels = ["First"]; // Only one label for three pairs
        let results = pool.parse_timestamp_pairs(&data, Some(&labels));

        assert_eq!(results.len(), 3);
        assert_eq!(results[0].label.as_deref(), Some("First"));
        assert!(results[1].label.is_none()); // No label available
        assert!(results[2].label.is_none()); // No label available
    }

    #[test]
    fn mock_pool_parse_pairs_empty_data() {
        let pool = MockTimestampPool::new(64, 1.0);
        let data = TimestampData::empty(1.0);
        let results = pool.parse_timestamp_pairs(&data, None);
        assert!(results.is_empty());
    }

    #[test]
    fn mock_pool_parse_pairs_odd_timestamps() {
        let pool = MockTimestampPool::new(64, 1.0);
        // 5 timestamps = 2 complete pairs (ignore last timestamp)
        let data = TimestampData::new(vec![0, 100, 200, 300, 400], 1.0);
        let results = pool.parse_timestamp_pairs(&data, None);
        assert_eq!(results.len(), 2);
    }

    // ===== SECTION 23: Send + Sync bounds for new types =====

    #[test]
    fn timestamp_result_is_send() {
        assert_send::<TimestampResult>();
    }

    #[test]
    fn timestamp_result_is_sync() {
        assert_sync::<TimestampResult>();
    }

    #[test]
    fn profile_result_is_send() {
        assert_send::<ProfileResult>();
    }

    #[test]
    fn profile_result_is_sync() {
        assert_sync::<ProfileResult>();
    }

    #[test]
    fn timestamp_data_is_send() {
        assert_send::<TimestampData>();
    }

    #[test]
    fn timestamp_data_is_sync() {
        assert_sync::<TimestampData>();
    }

    // ===== SECTION 24: Edge cases for timestamp parsing =====

    #[test]
    fn timestamp_result_with_zero_period() {
        let result = TimestampResult::new(0, 1000, 0.0);
        assert_eq!(result.duration_ticks, 1000);
        assert_eq!(result.duration_ns, 0.0); // 1000 * 0 = 0
        assert_eq!(result.duration_ms, 0.0);
    }

    #[test]
    fn timestamp_result_with_very_small_period() {
        let result = TimestampResult::new(0, 1_000_000_000, 0.001);
        // 1B ticks * 0.001 ns = 1M ns = 1 ms
        assert!((result.duration_ms - 1.0).abs() < 0.001);
    }

    #[test]
    fn timestamp_data_single_timestamp() {
        let data = TimestampData::new(vec![100], 1.0);
        assert_eq!(data.len(), 1);
        assert_eq!(data.pair_count(), 0); // No complete pairs
        assert_eq!(data.get_pair(0), None);
    }

    #[test]
    fn timestamp_data_large_timestamps() {
        let data = TimestampData::new(vec![u64::MAX - 1000, u64::MAX], 1.0);
        assert_eq!(data.len(), 2);
        let duration = data.duration_ms(0, 1);
        assert!(duration.is_some());
        // 1000 ticks * 1 ns = 1000 ns = 0.001 ms
        assert!((duration.unwrap() - 0.001).abs() < 0.0001);
    }

    #[test]
    fn profile_result_empty_string_label() {
        let ts = TimestampResult::zero();
        let profile = ProfileResult::with_label("", ts);
        assert!(profile.has_label());
        assert_eq!(profile.label.as_deref(), Some(""));
    }

    #[test]
    fn profile_result_unicode_label() {
        let ts = TimestampResult::zero();
        let profile = ProfileResult::with_label("Shadow Pass", ts);
        assert_eq!(profile.label.as_deref(), Some("Shadow Pass"));
    }
}
