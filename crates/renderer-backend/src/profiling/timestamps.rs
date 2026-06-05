//! GPU Timestamp Query Profiler for wgpu 25.x Performance Measurement.
//!
//! This module provides comprehensive GPU timestamp query profiling infrastructure
//! for measuring GPU-side execution times using hardware timestamp queries.
//!
//! # Overview
//!
//! The profiler supports:
//! - **TimestampQuery**: Low-level query set management
//! - **TimestampHandle**: Tracked query pairs with optional labels
//! - **TimestampResult**: Duration calculations with multiple time units
//! - **TimestampProfiler**: Main interface with resolve buffer management
//! - **GpuProfileScope**: RAII guard for automatic timing
//! - **FrameProfiler**: Per-frame profiling with statistics
//!
//! # Usage
//!
//! ```no_run
//! use renderer_backend::profiling::timestamps::{TimestampProfiler, GpuProfileScope};
//!
//! # fn example(device: &wgpu::Device, queue: &wgpu::Queue, adapter: &wgpu::Adapter) {
//! // Check support first
//! if !TimestampProfiler::is_supported(adapter) {
//!     println!("Timestamps not supported");
//!     return;
//! }
//!
//! // Create profiler with capacity for 64 timestamp pairs
//! let mut profiler = TimestampProfiler::new(device, queue, 64);
//!
//! // In render loop
//! let mut encoder = device.create_command_encoder(&Default::default());
//!
//! // Manual timing
//! let handle = profiler.begin(&mut encoder, Some("Shadow Pass"));
//! // ... shadow pass commands ...
//! profiler.end(&mut encoder, handle);
//!
//! // RAII scope timing
//! {
//!     let _scope = GpuProfileScope::new(&mut profiler, &mut encoder, "Lighting");
//!     // ... lighting commands ...
//! } // Automatically ends timing on drop
//!
//! // Resolve and submit
//! profiler.resolve(&mut encoder);
//! queue.submit(std::iter::once(encoder.finish()));
//!
//! // Read results (typically 1-3 frames later)
//! let results = profiler.read_results(queue);
//! for result in &results {
//!     println!("{}: {:.3}ms", result.label.as_deref().unwrap_or("unnamed"), result.duration_ms());
//! }
//! # }
//! ```
//!
//! # wgpu 22+ Compatibility
//!
//! This implementation targets wgpu 22+ and follows these patterns:
//! - `Device::create_query_set()` with `QuerySetDescriptor`
//! - `QueryType::Timestamp` for timestamp queries
//! - `Features::TIMESTAMP_QUERY` feature flag
//! - `Queue::get_timestamp_period()` for tick-to-ns conversion
//! - `CommandEncoder::write_timestamp()` for recording
//! - `CommandEncoder::resolve_query_set()` for result resolution

use std::fmt;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

// ============================================================================
// Constants
// ============================================================================

/// Size of a single timestamp value in bytes (u64).
pub const TIMESTAMP_SIZE_BYTES: u64 = 8;

/// Minimum capacity for timestamp queries.
pub const MIN_CAPACITY: u32 = 2;

/// Maximum recommended capacity (4096 pairs = 8192 timestamps).
pub const MAX_RECOMMENDED_CAPACITY: u32 = 4096;

/// Default capacity for timestamp queries.
pub const DEFAULT_CAPACITY: u32 = 128;

// ============================================================================
// TimestampQuery
// ============================================================================

/// Low-level timestamp query set wrapper.
///
/// Manages a `wgpu::QuerySet` with `QueryType::Timestamp` and tracks
/// the next available index for atomic allocation.
///
/// # Example
///
/// ```no_run
/// use renderer_backend::profiling::timestamps::TimestampQuery;
///
/// # fn example(device: &wgpu::Device) {
/// // Create query with capacity for 64 timestamp pairs (128 individual timestamps)
/// let mut query = TimestampQuery::new(device, 64);
///
/// // Allocate a pair of indices
/// if let Some((begin, end)) = query.allocate_pair() {
///     // Use begin and end indices with encoder.write_timestamp()
/// }
///
/// // Reset for next frame
/// query.reset();
/// # }
/// ```
pub struct TimestampQuery {
    /// The wgpu QuerySet for timestamp queries.
    query_set: wgpu::QuerySet,
    /// Maximum number of timestamp pairs (each pair = 2 timestamps).
    capacity: u32,
    /// Next available index (atomic for potential concurrent access).
    next_index: AtomicU32,
}

impl TimestampQuery {
    /// Create a new timestamp query set.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device to create the query set on
    /// * `capacity` - Maximum number of timestamp pairs to support
    ///
    /// # Panics
    ///
    /// Panics if capacity is 0.
    pub fn new(device: &wgpu::Device, capacity: u32) -> Self {
        assert!(capacity > 0, "TimestampQuery capacity must be > 0");

        // Each pair needs 2 timestamps (begin + end)
        let total_timestamps = capacity.saturating_mul(2);

        let query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("TimestampQuery QuerySet"),
            ty: wgpu::QueryType::Timestamp,
            count: total_timestamps,
        });

        Self {
            query_set,
            capacity,
            next_index: AtomicU32::new(0),
        }
    }

    /// Get a reference to the underlying QuerySet.
    #[inline]
    pub fn query_set(&self) -> &wgpu::QuerySet {
        &self.query_set
    }

    /// Get the capacity (number of timestamp pairs).
    #[inline]
    pub fn capacity(&self) -> u32 {
        self.capacity
    }

    /// Get the total number of individual timestamps (capacity * 2).
    #[inline]
    pub fn total_timestamps(&self) -> u32 {
        self.capacity.saturating_mul(2)
    }

    /// Get the current allocation index.
    #[inline]
    pub fn current_index(&self) -> u32 {
        self.next_index.load(Ordering::Relaxed)
    }

    /// Get the number of pairs currently allocated.
    #[inline]
    pub fn allocated_pairs(&self) -> u32 {
        self.current_index() / 2
    }

    /// Check if the query set has capacity for another pair.
    #[inline]
    pub fn has_capacity(&self) -> bool {
        self.current_index() + 2 <= self.total_timestamps()
    }

    /// Get the number of remaining pairs available.
    #[inline]
    pub fn remaining_pairs(&self) -> u32 {
        let used = self.current_index();
        let total = self.total_timestamps();
        (total.saturating_sub(used)) / 2
    }

    /// Allocate a timestamp pair (begin + end indices).
    ///
    /// Returns `Some((begin_index, end_index))` if capacity is available,
    /// `None` if the query set is exhausted.
    pub fn allocate_pair(&self) -> Option<(u32, u32)> {
        loop {
            let current = self.next_index.load(Ordering::Relaxed);
            let total = self.total_timestamps();

            if current + 2 > total {
                return None;
            }

            let new_index = current + 2;
            if self.next_index
                .compare_exchange_weak(current, new_index, Ordering::SeqCst, Ordering::Relaxed)
                .is_ok()
            {
                return Some((current, current + 1));
            }
            // CAS failed, retry
        }
    }

    /// Allocate a single timestamp index.
    ///
    /// Returns `Some(index)` if capacity is available, `None` if exhausted.
    pub fn allocate_single(&self) -> Option<u32> {
        loop {
            let current = self.next_index.load(Ordering::Relaxed);
            let total = self.total_timestamps();

            if current >= total {
                return None;
            }

            let new_index = current + 1;
            if self.next_index
                .compare_exchange_weak(current, new_index, Ordering::SeqCst, Ordering::Relaxed)
                .is_ok()
            {
                return Some(current);
            }
        }
    }

    /// Reset the query set for reuse.
    ///
    /// This resets the allocation index to 0, allowing the query set
    /// to be reused in the next frame.
    pub fn reset(&self) {
        self.next_index.store(0, Ordering::Relaxed);
    }

    /// Begin a query (write begin timestamp).
    ///
    /// Convenience method that allocates a pair and writes the begin timestamp.
    /// Returns the handle if successful.
    #[inline]
    pub fn begin_query(&self) -> Option<TimestampHandle> {
        self.allocate_pair().map(|(start, end)| TimestampHandle {
            start_index: start,
            end_index: end,
            label: None,
        })
    }
}

impl fmt::Debug for TimestampQuery {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TimestampQuery")
            .field("capacity", &self.capacity)
            .field("allocated", &self.current_index())
            .field("remaining_pairs", &self.remaining_pairs())
            .finish()
    }
}

// ============================================================================
// TimestampHandle
// ============================================================================

/// Handle representing a timestamp query pair.
///
/// Contains the indices for begin and end timestamps, plus an optional
/// label for identification in profiling results.
///
/// # Example
///
/// ```no_run
/// use renderer_backend::profiling::timestamps::TimestampHandle;
///
/// let handle = TimestampHandle::new(0, 1);
/// assert_eq!(handle.start_index, 0);
/// assert_eq!(handle.end_index, 1);
///
/// let labeled = TimestampHandle::with_label(2, 3, "Shadow Pass");
/// assert_eq!(labeled.label, Some("Shadow Pass".to_string()));
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TimestampHandle {
    /// Index of the begin timestamp in the query set.
    pub start_index: u32,
    /// Index of the end timestamp in the query set.
    pub end_index: u32,
    /// Optional label for identification.
    pub label: Option<String>,
}

impl TimestampHandle {
    /// Create a new handle without a label.
    #[inline]
    pub const fn new(start_index: u32, end_index: u32) -> Self {
        Self {
            start_index,
            end_index,
            label: None,
        }
    }

    /// Create a new handle with a label.
    #[inline]
    pub fn with_label(start_index: u32, end_index: u32, label: impl Into<String>) -> Self {
        Self {
            start_index,
            end_index,
            label: Some(label.into()),
        }
    }

    /// Set the label.
    #[inline]
    pub fn set_label(&mut self, label: impl Into<String>) {
        self.label = Some(label.into());
    }

    /// Clear the label.
    #[inline]
    pub fn clear_label(&mut self) {
        self.label = None;
    }

    /// Get the label or a default value.
    #[inline]
    pub fn label_or<'a>(&'a self, default: &'a str) -> &'a str {
        self.label.as_deref().unwrap_or(default)
    }

    /// Get the label or "unnamed".
    #[inline]
    pub fn label_or_unnamed(&self) -> &str {
        self.label_or("unnamed")
    }

    /// Check if this handle has a label.
    #[inline]
    pub fn has_label(&self) -> bool {
        self.label.is_some()
    }
}

impl fmt::Display for TimestampHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(ref label) = self.label {
            write!(f, "{}[{}..{}]", label, self.start_index, self.end_index)
        } else {
            write!(f, "[{}..{}]", self.start_index, self.end_index)
        }
    }
}

// ============================================================================
// TimestampResult
// ============================================================================

/// Result of a timestamp query measurement.
///
/// Contains the raw tick values and provides convenience methods for
/// getting the duration in various time units.
///
/// # Example
///
/// ```no_run
/// use renderer_backend::profiling::timestamps::TimestampResult;
///
/// let result = TimestampResult::from_ticks(1000, 2000, 25.0);
/// println!("Duration: {}ns = {:.3}us = {:.6}ms",
///     result.duration_ns(),
///     result.duration_us(),
///     result.duration_ms());
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct TimestampResult {
    /// Optional label for this measurement.
    pub label: Option<String>,
    /// Raw start timestamp in GPU ticks.
    pub start_ns: u64,
    /// Raw end timestamp in GPU ticks (converted to ns).
    pub end_ns: u64,
}

impl TimestampResult {
    /// Create a new timestamp result.
    #[inline]
    pub const fn new(label: Option<String>, start_ns: u64, end_ns: u64) -> Self {
        Self { label, start_ns, end_ns }
    }

    /// Create a result from raw tick values and timestamp period.
    ///
    /// # Arguments
    ///
    /// * `start_ticks` - Start timestamp in GPU ticks
    /// * `end_ticks` - End timestamp in GPU ticks
    /// * `timestamp_period` - Nanoseconds per tick
    #[inline]
    pub fn from_ticks(start_ticks: u64, end_ticks: u64, timestamp_period: f32) -> Self {
        let start_ns = ((start_ticks as f64) * (timestamp_period as f64)) as u64;
        let end_ns = ((end_ticks as f64) * (timestamp_period as f64)) as u64;
        Self {
            label: None,
            start_ns,
            end_ns,
        }
    }

    /// Create a result with a label from tick values.
    #[inline]
    pub fn from_ticks_labeled(
        start_ticks: u64,
        end_ticks: u64,
        timestamp_period: f32,
        label: impl Into<String>,
    ) -> Self {
        let mut result = Self::from_ticks(start_ticks, end_ticks, timestamp_period);
        result.label = Some(label.into());
        result
    }

    /// Create a zero-duration result.
    #[inline]
    pub const fn zero() -> Self {
        Self {
            label: None,
            start_ns: 0,
            end_ns: 0,
        }
    }

    /// Create a zero-duration result with a label.
    #[inline]
    pub fn zero_labeled(label: impl Into<String>) -> Self {
        Self {
            label: Some(label.into()),
            start_ns: 0,
            end_ns: 0,
        }
    }

    /// Get the duration in nanoseconds.
    #[inline]
    pub fn duration_ns(&self) -> u64 {
        self.end_ns.saturating_sub(self.start_ns)
    }

    /// Get the duration in microseconds.
    #[inline]
    pub fn duration_us(&self) -> f64 {
        (self.duration_ns() as f64) / 1_000.0
    }

    /// Get the duration in milliseconds.
    #[inline]
    pub fn duration_ms(&self) -> f64 {
        (self.duration_ns() as f64) / 1_000_000.0
    }

    /// Get the duration in seconds.
    #[inline]
    pub fn duration_secs(&self) -> f64 {
        (self.duration_ns() as f64) / 1_000_000_000.0
    }

    /// Check if this is a valid measurement (non-zero, end >= start).
    #[inline]
    pub fn is_valid(&self) -> bool {
        self.end_ns >= self.start_ns
    }

    /// Get the label or a default value.
    #[inline]
    pub fn label_or<'a>(&'a self, default: &'a str) -> &'a str {
        self.label.as_deref().unwrap_or(default)
    }
}

impl Default for TimestampResult {
    fn default() -> Self {
        Self::zero()
    }
}

impl fmt::Display for TimestampResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let label = self.label.as_deref().unwrap_or("unnamed");
        write!(f, "{}: {:.3}ms", label, self.duration_ms())
    }
}

// ============================================================================
// TimestampPeriodConverter
// ============================================================================

/// Converter between GPU timestamp ticks and nanoseconds.
///
/// The timestamp period is obtained from `queue.get_timestamp_period()`
/// and represents the number of nanoseconds per GPU clock tick.
///
/// # Example
///
/// ```no_run
/// use renderer_backend::profiling::timestamps::TimestampPeriodConverter;
///
/// # fn example(queue: &wgpu::Queue) {
/// let period = queue.get_timestamp_period();
/// let converter = TimestampPeriodConverter::new(period);
///
/// let ticks: u64 = 1000000;
/// let ns = converter.ticks_to_ns(ticks);
/// let recovered = converter.ns_to_ticks(ns);
/// # }
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TimestampPeriodConverter {
    /// Nanoseconds per GPU clock tick.
    timestamp_period: f32,
}

impl TimestampPeriodConverter {
    /// Create a new converter with the given timestamp period.
    ///
    /// # Arguments
    ///
    /// * `timestamp_period` - Nanoseconds per tick (from queue.get_timestamp_period())
    #[inline]
    pub const fn new(timestamp_period: f32) -> Self {
        Self { timestamp_period }
    }

    /// Get the timestamp period.
    #[inline]
    pub const fn period(&self) -> f32 {
        self.timestamp_period
    }

    /// Convert GPU ticks to nanoseconds.
    #[inline]
    pub fn ticks_to_ns(&self, ticks: u64) -> u64 {
        ((ticks as f64) * (self.timestamp_period as f64)) as u64
    }

    /// Convert nanoseconds to GPU ticks.
    #[inline]
    pub fn ns_to_ticks(&self, ns: u64) -> u64 {
        if self.timestamp_period > 0.0 {
            ((ns as f64) / (self.timestamp_period as f64)) as u64
        } else {
            0
        }
    }

    /// Convert ticks to microseconds.
    #[inline]
    pub fn ticks_to_us(&self, ticks: u64) -> f64 {
        (self.ticks_to_ns(ticks) as f64) / 1_000.0
    }

    /// Convert ticks to milliseconds.
    #[inline]
    pub fn ticks_to_ms(&self, ticks: u64) -> f64 {
        (self.ticks_to_ns(ticks) as f64) / 1_000_000.0
    }

    /// Calculate duration between two tick values in nanoseconds.
    #[inline]
    pub fn duration_ns(&self, start_ticks: u64, end_ticks: u64) -> u64 {
        let delta = end_ticks.saturating_sub(start_ticks);
        self.ticks_to_ns(delta)
    }

    /// Calculate duration between two tick values in milliseconds.
    #[inline]
    pub fn duration_ms(&self, start_ticks: u64, end_ticks: u64) -> f64 {
        let delta = end_ticks.saturating_sub(start_ticks);
        self.ticks_to_ms(delta)
    }
}

impl Default for TimestampPeriodConverter {
    fn default() -> Self {
        Self { timestamp_period: 1.0 }
    }
}

// ============================================================================
// ProfilerStats
// ============================================================================

/// Statistics about the timestamp profiler state.
///
/// Provides insight into profiler usage for debugging and monitoring.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProfilerStats {
    /// Total number of timestamp pairs the profiler can track.
    pub total_queries: u32,
    /// Number of queries currently allocated (in use).
    pub active_queries: u32,
    /// Number of queries that have been resolved and are ready to read.
    pub resolved_queries: u32,
    /// Average duration of resolved queries in nanoseconds.
    pub avg_duration_ns: u64,
    /// Minimum duration of resolved queries in nanoseconds.
    pub min_duration_ns: u64,
    /// Maximum duration of resolved queries in nanoseconds.
    pub max_duration_ns: u64,
}

impl ProfilerStats {
    /// Create new profiler stats.
    #[inline]
    pub const fn new(
        total_queries: u32,
        active_queries: u32,
        resolved_queries: u32,
        avg_duration_ns: u64,
        min_duration_ns: u64,
        max_duration_ns: u64,
    ) -> Self {
        Self {
            total_queries,
            active_queries,
            resolved_queries,
            avg_duration_ns,
            min_duration_ns,
            max_duration_ns,
        }
    }

    /// Create empty stats.
    #[inline]
    pub const fn empty() -> Self {
        Self {
            total_queries: 0,
            active_queries: 0,
            resolved_queries: 0,
            avg_duration_ns: 0,
            min_duration_ns: 0,
            max_duration_ns: 0,
        }
    }

    /// Calculate average duration in milliseconds.
    #[inline]
    pub fn avg_duration_ms(&self) -> f64 {
        (self.avg_duration_ns as f64) / 1_000_000.0
    }

    /// Calculate min duration in milliseconds.
    #[inline]
    pub fn min_duration_ms(&self) -> f64 {
        (self.min_duration_ns as f64) / 1_000_000.0
    }

    /// Calculate max duration in milliseconds.
    #[inline]
    pub fn max_duration_ms(&self) -> f64 {
        (self.max_duration_ns as f64) / 1_000_000.0
    }

    /// Calculate utilization as a fraction (active / total).
    #[inline]
    pub fn utilization(&self) -> f32 {
        if self.total_queries > 0 {
            (self.active_queries as f32) / (self.total_queries as f32)
        } else {
            0.0
        }
    }
}

impl Default for ProfilerStats {
    fn default() -> Self {
        Self::empty()
    }
}

impl fmt::Display for ProfilerStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ProfilerStats(active={}/{}, resolved={}, avg={:.3}ms, min={:.3}ms, max={:.3}ms)",
            self.active_queries,
            self.total_queries,
            self.resolved_queries,
            self.avg_duration_ms(),
            self.min_duration_ms(),
            self.max_duration_ms()
        )
    }
}

// ============================================================================
// TimestampProfiler
// ============================================================================

/// Main timestamp profiler interface.
///
/// Manages timestamp queries, resolve buffers, and result readback.
/// This is the primary interface for GPU profiling.
///
/// # Example
///
/// ```no_run
/// use renderer_backend::profiling::timestamps::TimestampProfiler;
///
/// # fn example(device: &wgpu::Device, queue: &wgpu::Queue, adapter: &wgpu::Adapter) {
/// // Check if timestamps are supported
/// if !TimestampProfiler::is_supported(adapter) {
///     return;
/// }
///
/// // Create profiler
/// let mut profiler = TimestampProfiler::new(device, queue, 64);
///
/// // Profile GPU work
/// let mut encoder = device.create_command_encoder(&Default::default());
/// let handle = profiler.begin(&mut encoder, Some("Render Pass"));
/// // ... render commands ...
/// profiler.end(&mut encoder, handle);
///
/// // Resolve and submit
/// profiler.resolve(&mut encoder);
/// queue.submit(std::iter::once(encoder.finish()));
///
/// // Read results
/// let results = profiler.read_results(queue);
/// # }
/// ```
pub struct TimestampProfiler {
    /// The timestamp query set.
    query: TimestampQuery,
    /// Buffer for resolved timestamp data.
    resolve_buffer: wgpu::Buffer,
    /// Staging buffer for CPU readback.
    staging_buffer: wgpu::Buffer,
    /// Collected results from previous frames.
    results: Vec<TimestampResult>,
    /// Handles for active queries (awaiting resolution).
    active_handles: Vec<TimestampHandle>,
    /// Timestamp period (ns per tick).
    timestamp_period: f32,
    /// Whether timestamps are supported.
    supported: bool,
    /// Whether results have been resolved and are pending readback.
    pending_readback: bool,
    /// Number of queries resolved in the current frame.
    queries_resolved: u32,
}

impl TimestampProfiler {
    /// Check if timestamp queries are supported by the adapter.
    pub fn is_supported(adapter: &wgpu::Adapter) -> bool {
        adapter.features().contains(wgpu::Features::TIMESTAMP_QUERY)
    }

    /// Create a new timestamp profiler.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device
    /// * `queue` - The wgpu queue (used to get timestamp period)
    /// * `capacity` - Maximum number of timestamp pairs to support
    ///
    /// # Note
    ///
    /// If the device doesn't support timestamp queries, the profiler
    /// operates in no-op mode (all methods are safe but return empty results).
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, capacity: u32) -> Self {
        let capacity = capacity.max(MIN_CAPACITY).min(MAX_RECOMMENDED_CAPACITY);
        let supported = device.features().contains(wgpu::Features::TIMESTAMP_QUERY);

        let query = if supported {
            TimestampQuery::new(device, capacity)
        } else {
            // Create minimal query set for no-op mode
            TimestampQuery::new(device, MIN_CAPACITY)
        };

        let buffer_size = (capacity as u64) * 2 * TIMESTAMP_SIZE_BYTES;

        let resolve_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("TimestampProfiler Resolve Buffer"),
            size: buffer_size,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("TimestampProfiler Staging Buffer"),
            size: buffer_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let timestamp_period = if supported {
            queue.get_timestamp_period()
        } else {
            1.0
        };

        Self {
            query,
            resolve_buffer,
            staging_buffer,
            results: Vec::new(),
            active_handles: Vec::with_capacity(capacity as usize),
            timestamp_period,
            supported,
            pending_readback: false,
            queries_resolved: 0,
        }
    }

    /// Check if timestamps are supported on this device.
    #[inline]
    pub fn is_supported_on_device(&self) -> bool {
        self.supported
    }

    /// Get the timestamp period (nanoseconds per tick).
    #[inline]
    pub fn timestamp_period(&self) -> f32 {
        self.timestamp_period
    }

    /// Get the capacity (maximum number of timestamp pairs).
    #[inline]
    pub fn capacity(&self) -> u32 {
        self.query.capacity()
    }

    /// Get the number of active (pending) queries.
    #[inline]
    pub fn active_count(&self) -> usize {
        self.active_handles.len()
    }

    /// Get the number of available query slots.
    #[inline]
    pub fn available_count(&self) -> u32 {
        self.query.remaining_pairs()
    }

    /// Get a reference to the underlying query set.
    #[inline]
    pub fn query_set(&self) -> &wgpu::QuerySet {
        self.query.query_set()
    }

    /// Begin a timestamp measurement.
    ///
    /// Writes the begin timestamp to the command encoder and returns
    /// a handle that must be passed to `end()`.
    ///
    /// # Arguments
    ///
    /// * `encoder` - Command encoder to write the timestamp to
    /// * `label` - Optional label for this measurement
    ///
    /// # Returns
    ///
    /// A `TimestampHandle` that must be passed to `end()`, or a dummy
    /// handle if timestamps are not supported or capacity is exhausted.
    pub fn begin(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        label: Option<&str>,
    ) -> TimestampHandle {
        if !self.supported {
            return TimestampHandle::new(0, 0);
        }

        let handle = match self.query.allocate_pair() {
            Some((start, end)) => {
                let mut handle = TimestampHandle::new(start, end);
                if let Some(l) = label {
                    handle.set_label(l);
                }
                handle
            }
            None => {
                // Capacity exhausted, return dummy handle
                return TimestampHandle::new(0, 0);
            }
        };

        encoder.write_timestamp(self.query.query_set(), handle.start_index);
        self.active_handles.push(handle.clone());

        handle
    }

    /// End a timestamp measurement.
    ///
    /// Writes the end timestamp to the command encoder.
    ///
    /// # Arguments
    ///
    /// * `encoder` - Command encoder to write the timestamp to
    /// * `handle` - The handle returned from `begin()`
    pub fn end(&mut self, encoder: &mut wgpu::CommandEncoder, handle: TimestampHandle) {
        if !self.supported {
            return;
        }

        encoder.write_timestamp(self.query.query_set(), handle.end_index);
    }

    /// Resolve timestamp queries to the readback buffer.
    ///
    /// This copies the query results to a buffer that can be read by the CPU.
    /// Call this after all queries are recorded, before submitting the command buffer.
    pub fn resolve(&mut self, encoder: &mut wgpu::CommandEncoder) {
        if !self.supported || self.active_handles.is_empty() {
            return;
        }

        let query_count = self.query.current_index();
        if query_count == 0 {
            return;
        }

        // Resolve queries to the resolve buffer
        encoder.resolve_query_set(
            self.query.query_set(),
            0..query_count,
            &self.resolve_buffer,
            0,
        );

        // Copy to staging buffer for CPU readback
        let byte_count = (query_count as u64) * TIMESTAMP_SIZE_BYTES;
        encoder.copy_buffer_to_buffer(
            &self.resolve_buffer,
            0,
            &self.staging_buffer,
            0,
            byte_count,
        );

        self.queries_resolved = query_count;
        self.pending_readback = true;
    }

    /// Read timestamp results from the GPU.
    ///
    /// This maps the staging buffer and reads the resolved timestamps.
    /// Results are returned as `TimestampResult` values.
    ///
    /// # Note
    ///
    /// This should be called 1-3 frames after `resolve()` to allow
    /// the GPU to complete the work.
    pub fn read_results(&mut self, queue: &wgpu::Queue) -> Vec<TimestampResult> {
        if !self.supported || !self.pending_readback || self.queries_resolved == 0 {
            return Vec::new();
        }

        let handles = std::mem::take(&mut self.active_handles);
        let query_count = self.queries_resolved;

        // Map the staging buffer
        let slice = self.staging_buffer.slice(..((query_count as u64) * TIMESTAMP_SIZE_BYTES));
        let (tx, rx) = std::sync::mpsc::channel();

        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });

        // Poll until mapping completes
        queue.submit([]);

        let map_result = rx.recv();
        if map_result.is_err() || map_result.unwrap().is_err() {
            self.pending_readback = false;
            return Vec::new();
        }

        let data = slice.get_mapped_range();
        let timestamps: &[u64] = bytemuck::cast_slice(&data);

        let mut results = Vec::with_capacity(handles.len());

        for handle in &handles {
            let start_idx = handle.start_index as usize;
            let end_idx = handle.end_index as usize;

            if end_idx < timestamps.len() {
                let start_ticks = timestamps[start_idx];
                let end_ticks = timestamps[end_idx];

                let mut result = TimestampResult::from_ticks(
                    start_ticks,
                    end_ticks,
                    self.timestamp_period,
                );
                result.label = handle.label.clone();
                results.push(result);
            }
        }

        drop(data);
        self.staging_buffer.unmap();

        self.pending_readback = false;
        self.results = results.clone();

        results
    }

    /// Clear the profiler state for the next frame.
    ///
    /// Resets the query allocations and active handles.
    pub fn clear(&mut self) {
        self.query.reset();
        self.active_handles.clear();
        self.results.clear();
        self.pending_readback = false;
        self.queries_resolved = 0;
    }

    /// Get statistics about the profiler state.
    pub fn stats(&self) -> ProfilerStats {
        let total_queries = self.query.capacity();
        let active_queries = self.active_handles.len() as u32;
        let resolved_queries = self.results.len() as u32;

        let (avg, min, max) = if self.results.is_empty() {
            (0, 0, 0)
        } else {
            let mut sum = 0u64;
            let mut min_val = u64::MAX;
            let mut max_val = 0u64;

            for r in &self.results {
                let dur = r.duration_ns();
                sum += dur;
                min_val = min_val.min(dur);
                max_val = max_val.max(dur);
            }

            let avg = sum / (self.results.len() as u64);
            (avg, min_val, max_val)
        };

        ProfilerStats::new(total_queries, active_queries, resolved_queries, avg, min, max)
    }

    /// Get the last results without reading from GPU.
    #[inline]
    pub fn last_results(&self) -> &[TimestampResult] {
        &self.results
    }

    /// Get a timestamp period converter.
    #[inline]
    pub fn converter(&self) -> TimestampPeriodConverter {
        TimestampPeriodConverter::new(self.timestamp_period)
    }
}

impl fmt::Debug for TimestampProfiler {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TimestampProfiler")
            .field("supported", &self.supported)
            .field("capacity", &self.query.capacity())
            .field("active", &self.active_handles.len())
            .field("results", &self.results.len())
            .field("timestamp_period", &self.timestamp_period)
            .finish()
    }
}

// ============================================================================
// GpuProfileScope
// ============================================================================

/// RAII guard for automatic GPU profiling.
///
/// Begins timing on construction and ends on drop.
///
/// # Example
///
/// ```no_run
/// use renderer_backend::profiling::timestamps::{TimestampProfiler, GpuProfileScope};
///
/// # fn example(profiler: &mut TimestampProfiler, encoder: &mut wgpu::CommandEncoder) {
/// {
///     let _scope = GpuProfileScope::new(profiler, encoder, "Shadow Pass");
///     // ... shadow pass commands ...
/// } // Timing ends automatically here
/// # }
/// ```
pub struct GpuProfileScope<'a, 'b> {
    profiler: &'a mut TimestampProfiler,
    encoder: &'b mut wgpu::CommandEncoder,
    handle: TimestampHandle,
    ended: bool,
}

impl<'a, 'b> GpuProfileScope<'a, 'b> {
    /// Create a new profile scope.
    ///
    /// Begins timing immediately.
    #[inline]
    pub fn new(
        profiler: &'a mut TimestampProfiler,
        encoder: &'b mut wgpu::CommandEncoder,
        label: &str,
    ) -> Self {
        let handle = profiler.begin(encoder, Some(label));
        Self {
            profiler,
            encoder,
            handle,
            ended: false,
        }
    }

    /// End the current timing and begin a new one with a different label.
    ///
    /// Returns the handle for the new timing.
    pub fn split(&mut self, new_label: &str) -> TimestampHandle {
        if !self.ended {
            self.profiler.end(self.encoder, self.handle.clone());
            self.ended = true;
        }

        self.handle = self.profiler.begin(self.encoder, Some(new_label));
        self.ended = false;
        self.handle.clone()
    }

    /// Get the current handle.
    #[inline]
    pub fn handle(&self) -> &TimestampHandle {
        &self.handle
    }

    /// Manually end the timing without dropping.
    pub fn end_manual(&mut self) {
        if !self.ended {
            self.profiler.end(self.encoder, self.handle.clone());
            self.ended = true;
        }
    }
}

impl<'a, 'b> Drop for GpuProfileScope<'a, 'b> {
    fn drop(&mut self) {
        if !self.ended {
            self.profiler.end(self.encoder, self.handle.clone());
        }
    }
}

// ============================================================================
// FrameStats
// ============================================================================

/// Statistics for a single profiled frame.
#[derive(Debug, Clone, PartialEq)]
pub struct FrameStats {
    /// Frame index (monotonically increasing).
    pub frame_index: u64,
    /// Total GPU time for the frame in nanoseconds.
    pub total_ns: u64,
    /// Number of profiled regions.
    pub region_count: usize,
    /// Individual region timings.
    pub regions: Vec<TimestampResult>,
}

impl FrameStats {
    /// Create new frame stats.
    pub fn new(frame_index: u64) -> Self {
        Self {
            frame_index,
            total_ns: 0,
            region_count: 0,
            regions: Vec::new(),
        }
    }

    /// Add a region timing.
    pub fn add_region(&mut self, result: TimestampResult) {
        self.total_ns += result.duration_ns();
        self.region_count += 1;
        self.regions.push(result);
    }

    /// Get total time in milliseconds.
    #[inline]
    pub fn total_ms(&self) -> f64 {
        (self.total_ns as f64) / 1_000_000.0
    }

    /// Check if this frame has any timings.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.regions.is_empty()
    }
}

impl Default for FrameStats {
    fn default() -> Self {
        Self::new(0)
    }
}

impl fmt::Display for FrameStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Frame {}: {:.3}ms ({} regions)",
            self.frame_index,
            self.total_ms(),
            self.region_count
        )
    }
}

// ============================================================================
// FrameProfiler
// ============================================================================

/// Per-frame profiling with automatic frame tracking.
///
/// Wraps `TimestampProfiler` with frame indexing and statistics.
///
/// # Example
///
/// ```no_run
/// use renderer_backend::profiling::timestamps::FrameProfiler;
///
/// # fn example(device: &wgpu::Device, queue: &wgpu::Queue) {
/// let mut profiler = FrameProfiler::new(device, queue, 64);
///
/// // Each frame
/// let mut encoder = device.create_command_encoder(&Default::default());
///
/// profiler.begin_frame(&mut encoder);
///
/// {
///     let scope = profiler.profile_pass(&mut encoder, "Shadow");
///     // ... shadow commands ...
/// }
///
/// profiler.end_frame(&mut encoder);
/// profiler.resolve(&mut encoder);
///
/// queue.submit(std::iter::once(encoder.finish()));
///
/// // Get frame stats
/// if let Some(stats) = profiler.get_frame_stats(queue) {
///     println!("{}", stats);
/// }
/// # }
/// ```
pub struct FrameProfiler {
    /// Underlying timestamp profiler.
    profiler: TimestampProfiler,
    /// Current frame index.
    frame_index: AtomicU64,
    /// Handles for the current frame.
    queries: Vec<TimestampHandle>,
    /// Frame begin handle.
    frame_begin: Option<TimestampHandle>,
    /// Whether we're inside a frame.
    in_frame: bool,
}

impl FrameProfiler {
    /// Create a new frame profiler.
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, capacity: u32) -> Self {
        Self {
            profiler: TimestampProfiler::new(device, queue, capacity),
            frame_index: AtomicU64::new(0),
            queries: Vec::with_capacity(64),
            frame_begin: None,
            in_frame: false,
        }
    }

    /// Check if timestamps are supported.
    #[inline]
    pub fn is_supported(&self) -> bool {
        self.profiler.is_supported_on_device()
    }

    /// Get the current frame index.
    #[inline]
    pub fn current_frame(&self) -> u64 {
        self.frame_index.load(Ordering::Relaxed)
    }

    /// Begin a new frame.
    ///
    /// Clears previous frame data and optionally writes a frame-start timestamp.
    pub fn begin_frame(&mut self, encoder: &mut wgpu::CommandEncoder) {
        self.profiler.clear();
        self.queries.clear();
        self.frame_begin = Some(self.profiler.begin(encoder, Some("Frame")));
        self.in_frame = true;
    }

    /// End the current frame.
    ///
    /// Writes the frame-end timestamp.
    pub fn end_frame(&mut self, encoder: &mut wgpu::CommandEncoder) {
        if let Some(handle) = self.frame_begin.take() {
            self.profiler.end(encoder, handle);
        }
        self.in_frame = false;
        self.frame_index.fetch_add(1, Ordering::Relaxed);
    }

    /// Profile a render pass within the current frame.
    ///
    /// Returns a RAII guard that ends timing on drop.
    pub fn profile_pass<'a, 'b>(
        &'a mut self,
        encoder: &'b mut wgpu::CommandEncoder,
        label: &str,
    ) -> GpuProfileScope<'a, 'b> {
        GpuProfileScope::new(&mut self.profiler, encoder, label)
    }

    /// Begin a manual timing region.
    pub fn begin_region(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        label: &str,
    ) -> TimestampHandle {
        let handle = self.profiler.begin(encoder, Some(label));
        self.queries.push(handle.clone());
        handle
    }

    /// End a manual timing region.
    pub fn end_region(&mut self, encoder: &mut wgpu::CommandEncoder, handle: TimestampHandle) {
        self.profiler.end(encoder, handle);
    }

    /// Resolve queries for readback.
    pub fn resolve(&mut self, encoder: &mut wgpu::CommandEncoder) {
        self.profiler.resolve(encoder);
    }

    /// Get frame statistics.
    ///
    /// Reads timestamp results and computes frame stats.
    pub fn get_frame_stats(&mut self, queue: &wgpu::Queue) -> Option<FrameStats> {
        let results = self.profiler.read_results(queue);
        if results.is_empty() {
            return None;
        }

        let frame_idx = self.frame_index.load(Ordering::Relaxed).saturating_sub(1);
        let mut stats = FrameStats::new(frame_idx);

        for result in results {
            stats.add_region(result);
        }

        Some(stats)
    }

    /// Get profiler statistics.
    #[inline]
    pub fn stats(&self) -> ProfilerStats {
        self.profiler.stats()
    }

    /// Get access to the underlying profiler.
    #[inline]
    pub fn profiler(&self) -> &TimestampProfiler {
        &self.profiler
    }

    /// Get mutable access to the underlying profiler.
    #[inline]
    pub fn profiler_mut(&mut self) -> &mut TimestampProfiler {
        &mut self.profiler
    }
}

impl fmt::Debug for FrameProfiler {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FrameProfiler")
            .field("frame_index", &self.frame_index.load(Ordering::Relaxed))
            .field("in_frame", &self.in_frame)
            .field("queries", &self.queries.len())
            .field("profiler", &self.profiler)
            .finish()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // =========== TimestampHandle tests ===========

    #[test]
    fn timestamp_handle_new_creates_correct_indices() {
        let handle = TimestampHandle::new(0, 1);
        assert_eq!(handle.start_index, 0);
        assert_eq!(handle.end_index, 1);
        assert!(handle.label.is_none());
    }

    #[test]
    fn timestamp_handle_with_label_stores_label() {
        let handle = TimestampHandle::with_label(2, 3, "Test Pass");
        assert_eq!(handle.start_index, 2);
        assert_eq!(handle.end_index, 3);
        assert_eq!(handle.label, Some("Test Pass".to_string()));
    }

    #[test]
    fn timestamp_handle_set_label_updates_label() {
        let mut handle = TimestampHandle::new(0, 1);
        assert!(handle.label.is_none());
        handle.set_label("New Label");
        assert_eq!(handle.label, Some("New Label".to_string()));
    }

    #[test]
    fn timestamp_handle_clear_label_removes_label() {
        let mut handle = TimestampHandle::with_label(0, 1, "Label");
        assert!(handle.has_label());
        handle.clear_label();
        assert!(!handle.has_label());
    }

    #[test]
    fn timestamp_handle_label_or_returns_default() {
        let handle = TimestampHandle::new(0, 1);
        assert_eq!(handle.label_or("default"), "default");

        let labeled = TimestampHandle::with_label(0, 1, "actual");
        assert_eq!(labeled.label_or("default"), "actual");
    }

    #[test]
    fn timestamp_handle_label_or_unnamed_returns_unnamed() {
        let handle = TimestampHandle::new(0, 1);
        assert_eq!(handle.label_or_unnamed(), "unnamed");
    }

    #[test]
    fn timestamp_handle_display_without_label() {
        let handle = TimestampHandle::new(5, 6);
        let display = format!("{}", handle);
        assert!(display.contains("[5..6]"));
    }

    #[test]
    fn timestamp_handle_display_with_label() {
        let handle = TimestampHandle::with_label(5, 6, "Test");
        let display = format!("{}", handle);
        assert!(display.contains("Test"));
        assert!(display.contains("[5..6]"));
    }

    #[test]
    fn timestamp_handle_clone_equality() {
        let handle = TimestampHandle::with_label(0, 1, "Clone");
        let cloned = handle.clone();
        assert_eq!(handle, cloned);
    }

    #[test]
    fn timestamp_handle_hash_consistency() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        let handle = TimestampHandle::new(0, 1);
        set.insert(handle.clone());
        assert!(set.contains(&handle));
    }

    // =========== TimestampResult tests ===========

    #[test]
    fn timestamp_result_new_stores_values() {
        let result = TimestampResult::new(Some("Test".to_string()), 1000, 2000);
        assert_eq!(result.label, Some("Test".to_string()));
        assert_eq!(result.start_ns, 1000);
        assert_eq!(result.end_ns, 2000);
    }

    #[test]
    fn timestamp_result_from_ticks_converts_correctly() {
        // period = 10ns per tick, 100 ticks = 1000ns
        let result = TimestampResult::from_ticks(0, 100, 10.0);
        assert_eq!(result.start_ns, 0);
        assert_eq!(result.end_ns, 1000);
        assert_eq!(result.duration_ns(), 1000);
    }

    #[test]
    fn timestamp_result_duration_ns_basic() {
        let result = TimestampResult::new(None, 1000, 5000);
        assert_eq!(result.duration_ns(), 4000);
    }

    #[test]
    fn timestamp_result_duration_ns_saturating() {
        // end < start should saturate to 0
        let result = TimestampResult::new(None, 5000, 1000);
        assert_eq!(result.duration_ns(), 0);
    }

    #[test]
    fn timestamp_result_duration_us() {
        let result = TimestampResult::new(None, 0, 1000);
        assert!((result.duration_us() - 1.0).abs() < 0.001);
    }

    #[test]
    fn timestamp_result_duration_ms() {
        let result = TimestampResult::new(None, 0, 1_000_000);
        assert!((result.duration_ms() - 1.0).abs() < 0.001);
    }

    #[test]
    fn timestamp_result_duration_secs() {
        let result = TimestampResult::new(None, 0, 1_000_000_000);
        assert!((result.duration_secs() - 1.0).abs() < 0.001);
    }

    #[test]
    fn timestamp_result_zero_creates_zero() {
        let result = TimestampResult::zero();
        assert!(result.label.is_none());
        assert_eq!(result.start_ns, 0);
        assert_eq!(result.end_ns, 0);
        assert_eq!(result.duration_ns(), 0);
    }

    #[test]
    fn timestamp_result_zero_labeled() {
        let result = TimestampResult::zero_labeled("Zero");
        assert_eq!(result.label, Some("Zero".to_string()));
        assert_eq!(result.duration_ns(), 0);
    }

    #[test]
    fn timestamp_result_is_valid() {
        let valid = TimestampResult::new(None, 0, 100);
        assert!(valid.is_valid());

        let invalid = TimestampResult::new(None, 100, 50);
        assert!(!invalid.is_valid());
    }

    #[test]
    fn timestamp_result_display() {
        let result = TimestampResult::new(Some("Test".to_string()), 0, 1_000_000);
        let display = format!("{}", result);
        assert!(display.contains("Test"));
        assert!(display.contains("1.000ms") || display.contains("ms"));
    }

    #[test]
    fn timestamp_result_default() {
        let result = TimestampResult::default();
        assert_eq!(result.duration_ns(), 0);
    }

    // =========== TimestampPeriodConverter tests ===========

    #[test]
    fn converter_new_stores_period() {
        let converter = TimestampPeriodConverter::new(25.0);
        assert_eq!(converter.period(), 25.0);
    }

    #[test]
    fn converter_ticks_to_ns() {
        let converter = TimestampPeriodConverter::new(10.0);
        assert_eq!(converter.ticks_to_ns(100), 1000);
    }

    #[test]
    fn converter_ns_to_ticks() {
        let converter = TimestampPeriodConverter::new(10.0);
        assert_eq!(converter.ns_to_ticks(1000), 100);
    }

    #[test]
    fn converter_ns_to_ticks_zero_period() {
        let converter = TimestampPeriodConverter::new(0.0);
        assert_eq!(converter.ns_to_ticks(1000), 0);
    }

    #[test]
    fn converter_ticks_to_us() {
        let converter = TimestampPeriodConverter::new(1.0);
        let us = converter.ticks_to_us(1000);
        assert!((us - 1.0).abs() < 0.001);
    }

    #[test]
    fn converter_ticks_to_ms() {
        let converter = TimestampPeriodConverter::new(1.0);
        let ms = converter.ticks_to_ms(1_000_000);
        assert!((ms - 1.0).abs() < 0.001);
    }

    #[test]
    fn converter_duration_ns() {
        let converter = TimestampPeriodConverter::new(10.0);
        let ns = converter.duration_ns(100, 200);
        assert_eq!(ns, 1000);
    }

    #[test]
    fn converter_duration_ms() {
        let converter = TimestampPeriodConverter::new(1.0);
        let ms = converter.duration_ms(0, 1_000_000);
        assert!((ms - 1.0).abs() < 0.001);
    }

    #[test]
    fn converter_default() {
        let converter = TimestampPeriodConverter::default();
        assert_eq!(converter.period(), 1.0);
    }

    #[test]
    fn converter_roundtrip() {
        let converter = TimestampPeriodConverter::new(25.0);
        let ticks: u64 = 12345;
        let ns = converter.ticks_to_ns(ticks);
        let recovered = converter.ns_to_ticks(ns);
        assert!((recovered as i64 - ticks as i64).abs() <= 1);
    }

    // =========== ProfilerStats tests ===========

    #[test]
    fn profiler_stats_new_stores_values() {
        let stats = ProfilerStats::new(100, 50, 25, 1000, 500, 2000);
        assert_eq!(stats.total_queries, 100);
        assert_eq!(stats.active_queries, 50);
        assert_eq!(stats.resolved_queries, 25);
        assert_eq!(stats.avg_duration_ns, 1000);
        assert_eq!(stats.min_duration_ns, 500);
        assert_eq!(stats.max_duration_ns, 2000);
    }

    #[test]
    fn profiler_stats_empty() {
        let stats = ProfilerStats::empty();
        assert_eq!(stats.total_queries, 0);
        assert_eq!(stats.active_queries, 0);
        assert_eq!(stats.resolved_queries, 0);
    }

    #[test]
    fn profiler_stats_avg_duration_ms() {
        let stats = ProfilerStats::new(0, 0, 0, 1_000_000, 0, 0);
        assert!((stats.avg_duration_ms() - 1.0).abs() < 0.001);
    }

    #[test]
    fn profiler_stats_min_duration_ms() {
        let stats = ProfilerStats::new(0, 0, 0, 0, 500_000, 0);
        assert!((stats.min_duration_ms() - 0.5).abs() < 0.001);
    }

    #[test]
    fn profiler_stats_max_duration_ms() {
        let stats = ProfilerStats::new(0, 0, 0, 0, 0, 2_000_000);
        assert!((stats.max_duration_ms() - 2.0).abs() < 0.001);
    }

    #[test]
    fn profiler_stats_utilization() {
        let stats = ProfilerStats::new(100, 50, 0, 0, 0, 0);
        assert!((stats.utilization() - 0.5).abs() < 0.001);
    }

    #[test]
    fn profiler_stats_utilization_zero_total() {
        let stats = ProfilerStats::empty();
        assert_eq!(stats.utilization(), 0.0);
    }

    #[test]
    fn profiler_stats_display() {
        let stats = ProfilerStats::new(100, 50, 25, 1_000_000, 500_000, 2_000_000);
        let display = format!("{}", stats);
        assert!(display.contains("50/100"));
        assert!(display.contains("resolved=25"));
    }

    #[test]
    fn profiler_stats_default() {
        let stats = ProfilerStats::default();
        assert_eq!(stats.total_queries, 0);
    }

    // =========== FrameStats tests ===========

    #[test]
    fn frame_stats_new() {
        let stats = FrameStats::new(42);
        assert_eq!(stats.frame_index, 42);
        assert_eq!(stats.total_ns, 0);
        assert_eq!(stats.region_count, 0);
        assert!(stats.regions.is_empty());
    }

    #[test]
    fn frame_stats_add_region() {
        let mut stats = FrameStats::new(0);
        let result = TimestampResult::new(Some("Test".to_string()), 0, 1000);
        stats.add_region(result);

        assert_eq!(stats.total_ns, 1000);
        assert_eq!(stats.region_count, 1);
        assert_eq!(stats.regions.len(), 1);
    }

    #[test]
    fn frame_stats_multiple_regions() {
        let mut stats = FrameStats::new(0);
        stats.add_region(TimestampResult::new(None, 0, 1000));
        stats.add_region(TimestampResult::new(None, 0, 2000));
        stats.add_region(TimestampResult::new(None, 0, 3000));

        assert_eq!(stats.total_ns, 6000);
        assert_eq!(stats.region_count, 3);
    }

    #[test]
    fn frame_stats_total_ms() {
        let mut stats = FrameStats::new(0);
        stats.add_region(TimestampResult::new(None, 0, 1_000_000));
        assert!((stats.total_ms() - 1.0).abs() < 0.001);
    }

    #[test]
    fn frame_stats_is_empty() {
        let stats = FrameStats::new(0);
        assert!(stats.is_empty());

        let mut stats2 = FrameStats::new(0);
        stats2.add_region(TimestampResult::new(None, 0, 100));
        assert!(!stats2.is_empty());
    }

    #[test]
    fn frame_stats_display() {
        let mut stats = FrameStats::new(5);
        stats.add_region(TimestampResult::new(None, 0, 1_000_000));
        let display = format!("{}", stats);
        assert!(display.contains("Frame 5"));
        assert!(display.contains("1 regions"));
    }

    #[test]
    fn frame_stats_default() {
        let stats = FrameStats::default();
        assert_eq!(stats.frame_index, 0);
    }

    // =========== Constants tests ===========

    #[test]
    fn timestamp_size_is_8_bytes() {
        assert_eq!(TIMESTAMP_SIZE_BYTES, 8);
    }

    #[test]
    fn min_capacity_is_reasonable() {
        assert!(MIN_CAPACITY >= 1);
        assert!(MIN_CAPACITY <= 16);
    }

    #[test]
    fn max_recommended_capacity_is_reasonable() {
        assert!(MAX_RECOMMENDED_CAPACITY >= 256);
        assert!(MAX_RECOMMENDED_CAPACITY <= 16384);
    }

    #[test]
    fn default_capacity_is_reasonable() {
        assert!(DEFAULT_CAPACITY >= MIN_CAPACITY);
        assert!(DEFAULT_CAPACITY <= MAX_RECOMMENDED_CAPACITY);
    }

    // =========== Send + Sync bounds ===========

    fn assert_send<T: Send>() {}
    fn assert_sync<T: Sync>() {}

    #[test]
    fn timestamp_handle_is_send_sync() {
        assert_send::<TimestampHandle>();
        assert_sync::<TimestampHandle>();
    }

    #[test]
    fn timestamp_result_is_send_sync() {
        assert_send::<TimestampResult>();
        assert_sync::<TimestampResult>();
    }

    #[test]
    fn timestamp_period_converter_is_send_sync() {
        assert_send::<TimestampPeriodConverter>();
        assert_sync::<TimestampPeriodConverter>();
    }

    #[test]
    fn profiler_stats_is_send_sync() {
        assert_send::<ProfilerStats>();
        assert_sync::<ProfilerStats>();
    }

    #[test]
    fn frame_stats_is_send_sync() {
        assert_send::<FrameStats>();
        assert_sync::<FrameStats>();
    }

    // =========== Edge case tests ===========

    #[test]
    fn timestamp_result_max_values() {
        let result = TimestampResult::new(None, 0, u64::MAX);
        assert_eq!(result.duration_ns(), u64::MAX);
    }

    #[test]
    fn timestamp_handle_max_indices() {
        let handle = TimestampHandle::new(u32::MAX - 1, u32::MAX);
        assert_eq!(handle.start_index, u32::MAX - 1);
        assert_eq!(handle.end_index, u32::MAX);
    }

    #[test]
    fn converter_large_ticks() {
        let converter = TimestampPeriodConverter::new(1.0);
        let ns = converter.ticks_to_ns(u64::MAX / 2);
        assert!(ns > 0);
    }

    #[test]
    fn timestamp_result_from_ticks_labeled() {
        let result = TimestampResult::from_ticks_labeled(0, 100, 10.0, "Labeled");
        assert_eq!(result.label, Some("Labeled".to_string()));
        assert_eq!(result.end_ns, 1000);
    }

    #[test]
    fn timestamp_result_label_or() {
        let result = TimestampResult::new(None, 0, 0);
        assert_eq!(result.label_or("default"), "default");

        let result2 = TimestampResult::new(Some("actual".to_string()), 0, 0);
        assert_eq!(result2.label_or("default"), "actual");
    }

    // =========== Integration-style tests (without real GPU) ===========

    #[test]
    fn timestamp_handle_equality_with_different_labels() {
        let h1 = TimestampHandle::with_label(0, 1, "A");
        let h2 = TimestampHandle::with_label(0, 1, "B");
        // Different labels = different handles
        assert_ne!(h1, h2);

        let h3 = TimestampHandle::with_label(0, 1, "A");
        assert_eq!(h1, h3);
    }

    #[test]
    fn frame_stats_accumulates_correctly() {
        let mut stats = FrameStats::new(100);

        for i in 0..10 {
            let result = TimestampResult::new(None, 0, (i + 1) * 1000);
            stats.add_region(result);
        }

        // Sum of 1000 + 2000 + ... + 10000 = 55000
        assert_eq!(stats.total_ns, 55000);
        assert_eq!(stats.region_count, 10);
    }
}
