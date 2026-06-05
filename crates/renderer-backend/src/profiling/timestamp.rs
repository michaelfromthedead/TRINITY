//! GPU Timestamp Profiler for Performance Measurement.
//!
//! This module provides a focused timestamp profiling system for measuring
//! GPU execution times using hardware timestamp queries. It offers a simpler
//! API than the comprehensive `timestamps` module, suitable for direct
//! performance measurement integration.
//!
//! # Overview
//!
//! The profiler supports:
//! - **TimestampQuery**: Individual GPU timestamp query representation
//! - **TimestampResult**: Duration calculations with multiple time units
//! - **TimestampProfiler**: Main interface for managing timestamp queries
//! - **TimestampScope**: RAII guard for automatic timing
//! - **ProfileStats**: Aggregated statistics for profiled regions
//!
//! # Usage
//!
//! ```no_run
//! use renderer_backend::profiling::timestamp::{TimestampProfiler, TimestampScope};
//!
//! # fn example(device: &wgpu::Device, queue: &wgpu::Queue) {
//! // Create profiler with max 64 queries
//! let mut profiler = TimestampProfiler::new(device, 64);
//!
//! // In render loop
//! let mut encoder = device.create_command_encoder(&Default::default());
//!
//! // Manual timing
//! let scope_id = profiler.begin_scope(&mut encoder, "Shadow Pass");
//! // ... shadow pass commands ...
//! profiler.end_scope(&mut encoder, scope_id);
//!
//! // Resolve and submit
//! // profiler.resolve(&mut encoder, &resolve_buffer);
//! // queue.submit(std::iter::once(encoder.finish()));
//!
//! // After readback, collect results
//! // let results = profiler.results();
//! # }
//! ```
//!
//! # Feature Detection
//!
//! Timestamp queries require `TIMESTAMP_QUERY` feature. The profiler
//! gracefully handles unsupported devices by operating in disabled mode.

use std::fmt;
use std::time::Duration;

// ============================================================================
// Constants
// ============================================================================

/// Size of a single timestamp value in bytes (u64).
pub const TIMESTAMP_SIZE_BYTES: u64 = 8;

/// Maximum number of queries per profiler (each scope uses 2 queries).
pub const MAX_QUERIES: u32 = 8192;

/// Default maximum queries.
pub const DEFAULT_MAX_QUERIES: u32 = 256;

// ============================================================================
// TimestampQuery
// ============================================================================

/// A GPU timestamp query representing a named time measurement.
///
/// Contains indices for the start and end timestamps in the query set,
/// along with a human-readable name for identification.
///
/// # Example
///
/// ```no_run
/// use renderer_backend::profiling::timestamp::TimestampQuery;
///
/// let query = TimestampQuery::new("Shadow Pass", 0, 1);
/// assert_eq!(query.name(), "Shadow Pass");
/// assert_eq!(query.start_query(), 0);
/// assert_eq!(query.end_query(), 1);
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TimestampQuery {
    /// Human-readable name for this query.
    name: String,
    /// Index of the start timestamp in the query set.
    start_query: u32,
    /// Index of the end timestamp in the query set.
    end_query: u32,
}

impl TimestampQuery {
    /// Create a new timestamp query.
    ///
    /// # Arguments
    ///
    /// * `name` - Human-readable name for this query
    /// * `start_query` - Index of the start timestamp
    /// * `end_query` - Index of the end timestamp
    #[inline]
    pub fn new(name: impl Into<String>, start_query: u32, end_query: u32) -> Self {
        Self {
            name: name.into(),
            start_query,
            end_query,
        }
    }

    /// Get the query name.
    #[inline]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get the start query index.
    #[inline]
    pub fn start_query(&self) -> u32 {
        self.start_query
    }

    /// Get the end query index.
    #[inline]
    pub fn end_query(&self) -> u32 {
        self.end_query
    }

    /// Set the name.
    #[inline]
    pub fn set_name(&mut self, name: impl Into<String>) {
        self.name = name.into();
    }

    /// Check if this query has an empty name.
    #[inline]
    pub fn is_unnamed(&self) -> bool {
        self.name.is_empty()
    }

    /// Get the query span (number of indices used).
    #[inline]
    pub fn span(&self) -> u32 {
        self.end_query.saturating_sub(self.start_query) + 1
    }
}

impl fmt::Display for TimestampQuery {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}[{}-{}]", self.name, self.start_query, self.end_query)
    }
}

// ============================================================================
// TimestampResult
// ============================================================================

/// Result of a timestamp measurement.
///
/// Contains the raw nanosecond values for start and end times, plus
/// convenience methods for calculating duration in various units.
///
/// # Example
///
/// ```no_run
/// use renderer_backend::profiling::timestamp::TimestampResult;
///
/// let result = TimestampResult::new("Shadow Pass", 0, 1_000_000);
/// assert_eq!(result.name(), "Shadow Pass");
/// assert_eq!(result.duration_ns(), 1_000_000);
/// assert!((result.duration_ms() - 1.0).abs() < 0.001);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct TimestampResult {
    /// Name of the profiled region.
    pub name: String,
    /// Start timestamp in nanoseconds.
    pub start_ns: u64,
    /// End timestamp in nanoseconds.
    pub end_ns: u64,
    /// Cached duration in nanoseconds.
    pub duration_ns: u64,
    /// Cached duration in milliseconds.
    pub duration_ms: f64,
}

impl TimestampResult {
    /// Create a new timestamp result.
    ///
    /// # Arguments
    ///
    /// * `name` - Name of the profiled region
    /// * `start_ns` - Start timestamp in nanoseconds
    /// * `end_ns` - End timestamp in nanoseconds
    #[inline]
    pub fn new(name: impl Into<String>, start_ns: u64, end_ns: u64) -> Self {
        let duration_ns = end_ns.saturating_sub(start_ns);
        let duration_ms = (duration_ns as f64) / 1_000_000.0;
        Self {
            name: name.into(),
            start_ns,
            end_ns,
            duration_ns,
            duration_ms,
        }
    }

    /// Create a result from tick values and timestamp period.
    ///
    /// # Arguments
    ///
    /// * `name` - Name of the profiled region
    /// * `start_ticks` - Start timestamp in GPU ticks
    /// * `end_ticks` - End timestamp in GPU ticks
    /// * `timestamp_period` - Nanoseconds per tick
    #[inline]
    pub fn from_ticks(
        name: impl Into<String>,
        start_ticks: u64,
        end_ticks: u64,
        timestamp_period: f32,
    ) -> Self {
        let start_ns = ((start_ticks as f64) * (timestamp_period as f64)) as u64;
        let end_ns = ((end_ticks as f64) * (timestamp_period as f64)) as u64;
        Self::new(name, start_ns, end_ns)
    }

    /// Create a zero-duration result.
    #[inline]
    pub fn zero(name: impl Into<String>) -> Self {
        Self::new(name, 0, 0)
    }

    /// Get the name.
    #[inline]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get the start timestamp in nanoseconds.
    #[inline]
    pub fn start_ns(&self) -> u64 {
        self.start_ns
    }

    /// Get the end timestamp in nanoseconds.
    #[inline]
    pub fn end_ns(&self) -> u64 {
        self.end_ns
    }

    /// Get the duration in nanoseconds.
    #[inline]
    pub fn duration_ns(&self) -> u64 {
        self.duration_ns
    }

    /// Get the duration in microseconds.
    #[inline]
    pub fn duration_us(&self) -> f64 {
        (self.duration_ns as f64) / 1_000.0
    }

    /// Get the duration in milliseconds.
    #[inline]
    pub fn duration_ms(&self) -> f64 {
        self.duration_ms
    }

    /// Get the duration in seconds.
    #[inline]
    pub fn duration_secs(&self) -> f64 {
        (self.duration_ns as f64) / 1_000_000_000.0
    }

    /// Convert to a std::time::Duration.
    #[inline]
    pub fn duration(&self) -> Duration {
        Duration::from_nanos(self.duration_ns)
    }

    /// Check if this is a valid measurement (end >= start).
    #[inline]
    pub fn is_valid(&self) -> bool {
        self.end_ns >= self.start_ns
    }

    /// Check if this is a zero-duration measurement.
    #[inline]
    pub fn is_zero(&self) -> bool {
        self.duration_ns == 0
    }
}

impl Default for TimestampResult {
    fn default() -> Self {
        Self::zero("")
    }
}

impl fmt::Display for TimestampResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {:.3}ms", self.name, self.duration_ms)
    }
}

// ============================================================================
// ProfileStats
// ============================================================================

/// Profiling statistics aggregated from multiple timestamp results.
///
/// Provides summary statistics including total, average, min, and max
/// durations for a collection of profiled regions.
///
/// # Example
///
/// ```no_run
/// use renderer_backend::profiling::timestamp::{TimestampResult, ProfileStats};
///
/// let results = vec![
///     TimestampResult::new("Pass1", 0, 1_000_000),
///     TimestampResult::new("Pass2", 0, 2_000_000),
/// ];
/// let refs: Vec<_> = results.iter().collect();
/// let stats = ProfileStats::from_results(&refs);
///
/// assert_eq!(stats.sample_count, 2);
/// assert!((stats.total_duration_ms - 3.0).abs() < 0.001);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProfileStats {
    /// Total duration of all samples in milliseconds.
    pub total_duration_ms: f64,
    /// Average duration per sample in milliseconds.
    pub avg_duration_ms: f64,
    /// Minimum duration in milliseconds.
    pub min_duration_ms: f64,
    /// Maximum duration in milliseconds.
    pub max_duration_ms: f64,
    /// Number of samples.
    pub sample_count: usize,
}

impl ProfileStats {
    /// Create new profile stats.
    #[inline]
    pub const fn new(
        total_duration_ms: f64,
        avg_duration_ms: f64,
        min_duration_ms: f64,
        max_duration_ms: f64,
        sample_count: usize,
    ) -> Self {
        Self {
            total_duration_ms,
            avg_duration_ms,
            min_duration_ms,
            max_duration_ms,
            sample_count,
        }
    }

    /// Create empty stats.
    #[inline]
    pub const fn empty() -> Self {
        Self {
            total_duration_ms: 0.0,
            avg_duration_ms: 0.0,
            min_duration_ms: 0.0,
            max_duration_ms: 0.0,
            sample_count: 0,
        }
    }

    /// Create stats from a collection of timestamp results.
    pub fn from_results(results: &[&TimestampResult]) -> Self {
        if results.is_empty() {
            return Self::empty();
        }

        let mut total_ns: u64 = 0;
        let mut min_ns = u64::MAX;
        let mut max_ns = 0u64;

        for r in results {
            let dur = r.duration_ns();
            total_ns = total_ns.saturating_add(dur);
            min_ns = min_ns.min(dur);
            max_ns = max_ns.max(dur);
        }

        let count = results.len();
        let avg_ns = total_ns / (count as u64);

        Self {
            total_duration_ms: (total_ns as f64) / 1_000_000.0,
            avg_duration_ms: (avg_ns as f64) / 1_000_000.0,
            min_duration_ms: (min_ns as f64) / 1_000_000.0,
            max_duration_ms: (max_ns as f64) / 1_000_000.0,
            sample_count: count,
        }
    }

    /// Create stats from owned results.
    pub fn from_owned_results(results: &[TimestampResult]) -> Self {
        let refs: Vec<_> = results.iter().collect();
        Self::from_results(&refs)
    }

    /// Check if stats are empty (no samples).
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.sample_count == 0
    }

    /// Get total duration as a Duration.
    #[inline]
    pub fn total_duration(&self) -> Duration {
        Duration::from_secs_f64(self.total_duration_ms / 1000.0)
    }

    /// Get average duration as a Duration.
    #[inline]
    pub fn avg_duration(&self) -> Duration {
        Duration::from_secs_f64(self.avg_duration_ms / 1000.0)
    }

    /// Get the range (max - min) in milliseconds.
    #[inline]
    pub fn range_ms(&self) -> f64 {
        self.max_duration_ms - self.min_duration_ms
    }

    /// Merge with another ProfileStats.
    pub fn merge(&self, other: &ProfileStats) -> Self {
        if other.is_empty() {
            return *self;
        }
        if self.is_empty() {
            return *other;
        }

        let total_count = self.sample_count + other.sample_count;
        let total_ms = self.total_duration_ms + other.total_duration_ms;

        Self {
            total_duration_ms: total_ms,
            avg_duration_ms: total_ms / (total_count as f64),
            min_duration_ms: self.min_duration_ms.min(other.min_duration_ms),
            max_duration_ms: self.max_duration_ms.max(other.max_duration_ms),
            sample_count: total_count,
        }
    }
}

impl Default for ProfileStats {
    fn default() -> Self {
        Self::empty()
    }
}

impl fmt::Display for ProfileStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ProfileStats(samples={}, total={:.3}ms, avg={:.3}ms, min={:.3}ms, max={:.3}ms)",
            self.sample_count,
            self.total_duration_ms,
            self.avg_duration_ms,
            self.min_duration_ms,
            self.max_duration_ms
        )
    }
}

// ============================================================================
// TimestampProfiler
// ============================================================================

/// Manages GPU timestamp queries for performance profiling.
///
/// The profiler handles query set creation, index allocation, and result
/// collection. It gracefully degrades when timestamp queries are not
/// supported by the device.
///
/// # Example
///
/// ```no_run
/// use renderer_backend::profiling::timestamp::TimestampProfiler;
///
/// # fn example(device: &wgpu::Device) {
/// let mut profiler = TimestampProfiler::new(device, 64);
///
/// if profiler.is_enabled() {
///     // Use profiler
/// }
/// # }
/// ```
pub struct TimestampProfiler {
    /// The wgpu query set for timestamps.
    query_set: Option<wgpu::QuerySet>,
    /// Current query index.
    query_count: u32,
    /// Maximum number of queries.
    max_queries: u32,
    /// Pending queries awaiting resolution.
    pending_queries: Vec<TimestampQuery>,
    /// Collected results.
    results: Vec<TimestampResult>,
    /// Whether profiling is enabled.
    enabled: bool,
    /// Timestamp period (ns per tick).
    timestamp_period: f32,
}

impl TimestampProfiler {
    /// Create a new timestamp profiler.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device to create the query set on
    /// * `max_queries` - Maximum number of individual timestamps (each scope uses 2)
    ///
    /// # Note
    ///
    /// If the device doesn't support `TIMESTAMP_QUERY`, the profiler
    /// operates in disabled mode.
    pub fn new(device: &wgpu::Device, max_queries: u32) -> Self {
        let max_queries = max_queries.min(MAX_QUERIES).max(2);
        let supported = device.features().contains(wgpu::Features::TIMESTAMP_QUERY);

        let query_set = if supported {
            Some(device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("TimestampProfiler QuerySet"),
                ty: wgpu::QueryType::Timestamp,
                count: max_queries,
            }))
        } else {
            None
        };

        Self {
            query_set,
            query_count: 0,
            max_queries,
            pending_queries: Vec::with_capacity((max_queries / 2) as usize),
            results: Vec::with_capacity((max_queries / 2) as usize),
            enabled: supported,
            timestamp_period: 1.0, // Will be set via set_timestamp_period
        }
    }

    /// Create a profiler with a specific timestamp period.
    pub fn with_timestamp_period(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        max_queries: u32,
    ) -> Self {
        let mut profiler = Self::new(device, max_queries);
        if profiler.enabled {
            profiler.timestamp_period = queue.get_timestamp_period();
        }
        profiler
    }

    /// Check if profiling is enabled.
    #[inline]
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Enable profiling (only effective if device supports timestamps).
    #[inline]
    pub fn enable(&mut self) {
        if self.query_set.is_some() {
            self.enabled = true;
        }
    }

    /// Disable profiling.
    #[inline]
    pub fn disable(&mut self) {
        self.enabled = false;
    }

    /// Get the maximum number of queries.
    #[inline]
    pub fn max_queries(&self) -> u32 {
        self.max_queries
    }

    /// Get the current query count.
    #[inline]
    pub fn query_count(&self) -> u32 {
        self.query_count
    }

    /// Get the number of remaining query slots.
    #[inline]
    pub fn remaining_queries(&self) -> u32 {
        self.max_queries.saturating_sub(self.query_count)
    }

    /// Check if there's capacity for another scope (2 queries).
    #[inline]
    pub fn has_capacity(&self) -> bool {
        self.query_count + 2 <= self.max_queries
    }

    /// Get the timestamp period (nanoseconds per tick).
    #[inline]
    pub fn timestamp_period(&self) -> f32 {
        self.timestamp_period
    }

    /// Set the timestamp period.
    #[inline]
    pub fn set_timestamp_period(&mut self, period: f32) {
        self.timestamp_period = period;
    }

    /// Get the query set (if supported).
    #[inline]
    pub fn query_set(&self) -> Option<&wgpu::QuerySet> {
        self.query_set.as_ref()
    }

    /// Begin a profiling scope.
    ///
    /// Writes the begin timestamp to the command encoder.
    ///
    /// # Arguments
    ///
    /// * `encoder` - Command encoder to write the timestamp to
    /// * `name` - Name for this scope
    ///
    /// # Returns
    ///
    /// A scope ID that must be passed to `end_scope()`.
    pub fn begin_scope(&mut self, encoder: &mut wgpu::CommandEncoder, name: &str) -> u32 {
        if !self.enabled || !self.has_capacity() {
            return u32::MAX;
        }

        let query_set = match &self.query_set {
            Some(qs) => qs,
            None => return u32::MAX,
        };

        let start_query = self.query_count;
        let end_query = start_query + 1;

        encoder.write_timestamp(query_set, start_query);

        let query = TimestampQuery::new(name, start_query, end_query);
        let scope_id = self.pending_queries.len() as u32;
        self.pending_queries.push(query);

        self.query_count += 2;

        scope_id
    }

    /// End a profiling scope.
    ///
    /// Writes the end timestamp to the command encoder.
    ///
    /// # Arguments
    ///
    /// * `encoder` - Command encoder to write the timestamp to
    /// * `scope_id` - The ID returned from `begin_scope()`
    pub fn end_scope(&mut self, encoder: &mut wgpu::CommandEncoder, scope_id: u32) {
        if !self.enabled || scope_id == u32::MAX {
            return;
        }

        let query_set = match &self.query_set {
            Some(qs) => qs,
            None => return,
        };

        let idx = scope_id as usize;
        if idx >= self.pending_queries.len() {
            return;
        }

        let end_query = self.pending_queries[idx].end_query();
        encoder.write_timestamp(query_set, end_query);
    }

    /// Create a RAII scope for automatic timing.
    ///
    /// # Note
    ///
    /// This method cannot be used directly due to borrow checker limitations.
    /// Use `TimestampScope::new()` instead.
    #[inline]
    pub fn scope_id(&mut self, encoder: &mut wgpu::CommandEncoder, name: &str) -> u32 {
        self.begin_scope(encoder, name)
    }

    /// Resolve timestamp queries to a buffer.
    ///
    /// # Arguments
    ///
    /// * `encoder` - Command encoder to record the resolve command
    /// * `buffer` - Buffer to resolve timestamps into (must have QUERY_RESOLVE usage)
    pub fn resolve(&mut self, encoder: &mut wgpu::CommandEncoder, buffer: &wgpu::Buffer) {
        if !self.enabled || self.query_count == 0 {
            return;
        }

        let query_set = match &self.query_set {
            Some(qs) => qs,
            None => return,
        };

        encoder.resolve_query_set(query_set, 0..self.query_count, buffer, 0);
    }

    /// Collect results from resolved timestamp data.
    ///
    /// # Arguments
    ///
    /// * `data` - Raw timestamp data (array of u64 values)
    /// * `timestamp_period` - Nanoseconds per GPU tick
    pub fn collect_results(&mut self, data: &[u8], timestamp_period: f32) {
        self.results.clear();

        if data.len() < (self.query_count as usize) * 8 {
            return;
        }

        let timestamps: &[u64] = bytemuck::cast_slice(data);

        for query in &self.pending_queries {
            let start_idx = query.start_query() as usize;
            let end_idx = query.end_query() as usize;

            if end_idx >= timestamps.len() {
                continue;
            }

            let start_ticks = timestamps[start_idx];
            let end_ticks = timestamps[end_idx];

            let result =
                TimestampResult::from_ticks(query.name(), start_ticks, end_ticks, timestamp_period);
            self.results.push(result);
        }
    }

    /// Collect results using the profiler's stored timestamp period.
    pub fn collect_results_auto(&mut self, data: &[u8]) {
        self.collect_results(data, self.timestamp_period);
    }

    /// Get the collected results.
    #[inline]
    pub fn results(&self) -> &[TimestampResult] {
        &self.results
    }

    /// Take ownership of the results.
    #[inline]
    pub fn take_results(&mut self) -> Vec<TimestampResult> {
        std::mem::take(&mut self.results)
    }

    /// Clear the collected results.
    #[inline]
    pub fn clear_results(&mut self) {
        self.results.clear();
    }

    /// Get statistics for a specific named scope.
    ///
    /// Filters results by name and computes aggregate statistics.
    pub fn stats_for(&self, name: &str) -> Option<ProfileStats> {
        let matching: Vec<_> = self.results.iter().filter(|r| r.name() == name).collect();

        if matching.is_empty() {
            None
        } else {
            Some(ProfileStats::from_results(&matching))
        }
    }

    /// Get statistics for all results.
    pub fn stats(&self) -> ProfileStats {
        ProfileStats::from_owned_results(&self.results)
    }

    /// Reset the profiler for a new frame.
    ///
    /// Clears pending queries and resets the query counter.
    pub fn reset(&mut self) {
        self.query_count = 0;
        self.pending_queries.clear();
    }

    /// Get the pending queries.
    #[inline]
    pub fn pending_queries(&self) -> &[TimestampQuery] {
        &self.pending_queries
    }

    /// Get the number of pending queries.
    #[inline]
    pub fn pending_count(&self) -> usize {
        self.pending_queries.len()
    }
}

impl fmt::Debug for TimestampProfiler {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TimestampProfiler")
            .field("enabled", &self.enabled)
            .field("query_count", &self.query_count)
            .field("max_queries", &self.max_queries)
            .field("pending", &self.pending_queries.len())
            .field("results", &self.results.len())
            .field("timestamp_period", &self.timestamp_period)
            .finish()
    }
}

// ============================================================================
// TimestampScope
// ============================================================================

/// RAII guard for automatic GPU timestamp profiling.
///
/// Begins timing on construction and ends on drop.
///
/// # Example
///
/// ```no_run
/// use renderer_backend::profiling::timestamp::{TimestampProfiler, TimestampScope};
///
/// # fn example(profiler: &mut TimestampProfiler, encoder: &mut wgpu::CommandEncoder) {
/// {
///     let _scope = TimestampScope::new(profiler, encoder, "Shadow Pass");
///     // ... shadow pass commands ...
/// } // Timing ends automatically here
/// # }
/// ```
pub struct TimestampScope<'a> {
    profiler: &'a mut TimestampProfiler,
    encoder: *mut wgpu::CommandEncoder,
    name: String,
    scope_id: u32,
    ended: bool,
}

impl<'a> TimestampScope<'a> {
    /// Create a new timestamp scope.
    ///
    /// Begins timing immediately.
    ///
    /// # Safety
    ///
    /// The encoder reference is stored as a raw pointer to work around
    /// borrow checker limitations. The caller must ensure the encoder
    /// outlives the scope.
    #[inline]
    pub fn new(
        profiler: &'a mut TimestampProfiler,
        encoder: &mut wgpu::CommandEncoder,
        name: &str,
    ) -> Self {
        let scope_id = profiler.begin_scope(encoder, name);
        Self {
            profiler,
            encoder: encoder as *mut _,
            name: name.to_string(),
            scope_id,
            ended: false,
        }
    }

    /// Get the scope ID.
    #[inline]
    pub fn scope_id(&self) -> u32 {
        self.scope_id
    }

    /// Get the scope name.
    #[inline]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Check if the scope has already ended.
    #[inline]
    pub fn is_ended(&self) -> bool {
        self.ended
    }

    /// Manually end the timing without dropping.
    ///
    /// This is useful when you need to end timing before the scope naturally drops.
    pub fn end_manual(&mut self) {
        if !self.ended {
            // Safety: The encoder pointer is valid for the lifetime of this scope
            let encoder = unsafe { &mut *self.encoder };
            self.profiler.end_scope(encoder, self.scope_id);
            self.ended = true;
        }
    }
}

impl<'a> Drop for TimestampScope<'a> {
    fn drop(&mut self) {
        if !self.ended {
            // Safety: The encoder pointer is valid for the lifetime of this scope
            let encoder = unsafe { &mut *self.encoder };
            self.profiler.end_scope(encoder, self.scope_id);
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // =========== TimestampQuery tests ===========

    #[test]
    fn timestamp_query_new_creates_correctly() {
        let query = TimestampQuery::new("Test", 0, 1);
        assert_eq!(query.name(), "Test");
        assert_eq!(query.start_query(), 0);
        assert_eq!(query.end_query(), 1);
    }

    #[test]
    fn timestamp_query_span_calculates_correctly() {
        let query = TimestampQuery::new("Test", 0, 1);
        assert_eq!(query.span(), 2);

        let query2 = TimestampQuery::new("Test", 5, 10);
        assert_eq!(query2.span(), 6);
    }

    #[test]
    fn timestamp_query_is_unnamed_detects_empty() {
        let unnamed = TimestampQuery::new("", 0, 1);
        assert!(unnamed.is_unnamed());

        let named = TimestampQuery::new("Named", 0, 1);
        assert!(!named.is_unnamed());
    }

    #[test]
    fn timestamp_query_set_name_updates() {
        let mut query = TimestampQuery::new("Old", 0, 1);
        query.set_name("New");
        assert_eq!(query.name(), "New");
    }

    #[test]
    fn timestamp_query_display_format() {
        let query = TimestampQuery::new("Test", 0, 1);
        let display = format!("{}", query);
        assert!(display.contains("Test"));
        assert!(display.contains("0"));
        assert!(display.contains("1"));
    }

    #[test]
    fn timestamp_query_clone_equality() {
        let query = TimestampQuery::new("Clone", 5, 6);
        let cloned = query.clone();
        assert_eq!(query, cloned);
    }

    #[test]
    fn timestamp_query_hash_consistency() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        let query = TimestampQuery::new("Hash", 0, 1);
        set.insert(query.clone());
        assert!(set.contains(&query));
    }

    // =========== TimestampResult tests ===========

    #[test]
    fn timestamp_result_new_calculates_duration() {
        let result = TimestampResult::new("Test", 0, 1_000_000);
        assert_eq!(result.duration_ns(), 1_000_000);
        assert!((result.duration_ms() - 1.0).abs() < 0.001);
    }

    #[test]
    fn timestamp_result_from_ticks_converts_correctly() {
        // 1000 ticks at 1ns/tick = 1000ns = 0.001ms
        let result = TimestampResult::from_ticks("Test", 0, 1000, 1.0);
        assert_eq!(result.duration_ns(), 1000);
        assert!((result.duration_ms() - 0.001).abs() < 0.0001);
    }

    #[test]
    fn timestamp_result_from_ticks_with_period() {
        // 100 ticks at 10ns/tick = 1000ns
        let result = TimestampResult::from_ticks("Test", 0, 100, 10.0);
        assert_eq!(result.duration_ns(), 1000);
    }

    #[test]
    fn timestamp_result_zero_creates_zero_duration() {
        let result = TimestampResult::zero("Zero");
        assert!(result.is_zero());
        assert_eq!(result.duration_ns(), 0);
    }

    #[test]
    fn timestamp_result_duration_conversions() {
        let result = TimestampResult::new("Test", 0, 1_000_000_000); // 1 second
        assert_eq!(result.duration_ns(), 1_000_000_000);
        assert!((result.duration_us() - 1_000_000.0).abs() < 0.1);
        assert!((result.duration_ms() - 1000.0).abs() < 0.001);
        assert!((result.duration_secs() - 1.0).abs() < 0.000001);
    }

    #[test]
    fn timestamp_result_to_std_duration() {
        let result = TimestampResult::new("Test", 0, 1_000_000_000);
        let duration = result.duration();
        assert_eq!(duration, Duration::from_secs(1));
    }

    #[test]
    fn timestamp_result_is_valid_checks_order() {
        let valid = TimestampResult::new("Valid", 0, 100);
        assert!(valid.is_valid());

        // Same start/end is valid (zero duration)
        let zero = TimestampResult::new("Zero", 100, 100);
        assert!(zero.is_valid());
    }

    #[test]
    fn timestamp_result_display_format() {
        let result = TimestampResult::new("Test", 0, 1_000_000);
        let display = format!("{}", result);
        assert!(display.contains("Test"));
        assert!(display.contains("ms"));
    }

    #[test]
    fn timestamp_result_default_is_zero() {
        let result = TimestampResult::default();
        assert!(result.is_zero());
    }

    #[test]
    fn timestamp_result_clone_equality() {
        let result = TimestampResult::new("Clone", 0, 1000);
        let cloned = result.clone();
        assert_eq!(result, cloned);
    }

    // =========== ProfileStats tests ===========

    #[test]
    fn profile_stats_empty_creates_zeros() {
        let stats = ProfileStats::empty();
        assert!(stats.is_empty());
        assert_eq!(stats.sample_count, 0);
        assert_eq!(stats.total_duration_ms, 0.0);
    }

    #[test]
    fn profile_stats_from_single_result() {
        let result = TimestampResult::new("Single", 0, 1_000_000);
        let stats = ProfileStats::from_results(&[&result]);
        assert_eq!(stats.sample_count, 1);
        assert!((stats.total_duration_ms - 1.0).abs() < 0.001);
        assert!((stats.avg_duration_ms - 1.0).abs() < 0.001);
    }

    #[test]
    fn profile_stats_from_multiple_results() {
        let r1 = TimestampResult::new("R1", 0, 1_000_000);
        let r2 = TimestampResult::new("R2", 0, 2_000_000);
        let r3 = TimestampResult::new("R3", 0, 3_000_000);
        let stats = ProfileStats::from_results(&[&r1, &r2, &r3]);

        assert_eq!(stats.sample_count, 3);
        assert!((stats.total_duration_ms - 6.0).abs() < 0.001);
        assert!((stats.avg_duration_ms - 2.0).abs() < 0.001);
        assert!((stats.min_duration_ms - 1.0).abs() < 0.001);
        assert!((stats.max_duration_ms - 3.0).abs() < 0.001);
    }

    #[test]
    fn profile_stats_range_calculation() {
        let r1 = TimestampResult::new("R1", 0, 1_000_000);
        let r2 = TimestampResult::new("R2", 0, 5_000_000);
        let stats = ProfileStats::from_results(&[&r1, &r2]);
        assert!((stats.range_ms() - 4.0).abs() < 0.001);
    }

    #[test]
    fn profile_stats_merge_empty() {
        let stats = ProfileStats::empty();
        let r1 = TimestampResult::new("R1", 0, 1_000_000);
        let other = ProfileStats::from_results(&[&r1]);

        let merged = stats.merge(&other);
        assert_eq!(merged.sample_count, 1);
    }

    #[test]
    fn profile_stats_merge_both() {
        let r1 = TimestampResult::new("R1", 0, 1_000_000);
        let r2 = TimestampResult::new("R2", 0, 2_000_000);
        let s1 = ProfileStats::from_results(&[&r1]);
        let s2 = ProfileStats::from_results(&[&r2]);

        let merged = s1.merge(&s2);
        assert_eq!(merged.sample_count, 2);
        assert!((merged.total_duration_ms - 3.0).abs() < 0.001);
    }

    #[test]
    fn profile_stats_to_duration() {
        let r1 = TimestampResult::new("R1", 0, 1_000_000_000); // 1 second
        let stats = ProfileStats::from_results(&[&r1]);
        assert_eq!(stats.total_duration(), Duration::from_secs(1));
    }

    #[test]
    fn profile_stats_display_format() {
        let r1 = TimestampResult::new("R1", 0, 1_000_000);
        let stats = ProfileStats::from_results(&[&r1]);
        let display = format!("{}", stats);
        assert!(display.contains("ProfileStats"));
        assert!(display.contains("samples=1"));
    }

    #[test]
    fn profile_stats_default_is_empty() {
        let stats = ProfileStats::default();
        assert!(stats.is_empty());
    }

    // =========== TimestampProfiler tests (unit tests without GPU) ===========

    #[test]
    fn profiler_constants_reasonable() {
        assert!(MAX_QUERIES >= 256);
        assert!(DEFAULT_MAX_QUERIES >= 64);
        assert!(TIMESTAMP_SIZE_BYTES == 8);
    }

    #[test]
    fn profile_stats_from_owned_results_works() {
        let results = vec![
            TimestampResult::new("R1", 0, 1_000_000),
            TimestampResult::new("R2", 0, 2_000_000),
        ];
        let stats = ProfileStats::from_owned_results(&results);
        assert_eq!(stats.sample_count, 2);
    }

    #[test]
    fn timestamp_result_saturating_sub() {
        // Test with end < start (invalid but shouldn't panic)
        // Note: We construct directly to test edge case
        let result = TimestampResult {
            name: "Invalid".to_string(),
            start_ns: 100,
            end_ns: 50,
            duration_ns: 0, // Saturating sub results in 0
            duration_ms: 0.0,
        };
        // The struct stores the values, is_valid checks them
        assert!(!result.is_valid());
    }

    #[test]
    fn timestamp_query_accepts_string_types() {
        let q1 = TimestampQuery::new("str", 0, 1);
        let q2 = TimestampQuery::new(String::from("String"), 0, 1);
        assert_eq!(q1.name(), "str");
        assert_eq!(q2.name(), "String");
    }

    #[test]
    fn timestamp_result_accepts_string_types() {
        let r1 = TimestampResult::new("str", 0, 100);
        let r2 = TimestampResult::new(String::from("String"), 0, 100);
        assert_eq!(r1.name(), "str");
        assert_eq!(r2.name(), "String");
    }

    #[test]
    fn profile_stats_avg_duration_to_duration() {
        let r1 = TimestampResult::new("R1", 0, 500_000_000); // 500ms
        let stats = ProfileStats::from_results(&[&r1]);
        let avg = stats.avg_duration();
        assert_eq!(avg, Duration::from_millis(500));
    }

    // =========== Send + Sync bounds ===========

    fn assert_send<T: Send>() {}
    fn assert_sync<T: Sync>() {}

    #[test]
    fn timestamp_query_is_send_sync() {
        assert_send::<TimestampQuery>();
        assert_sync::<TimestampQuery>();
    }

    #[test]
    fn timestamp_result_is_send_sync() {
        assert_send::<TimestampResult>();
        assert_sync::<TimestampResult>();
    }

    #[test]
    fn profile_stats_is_send_sync() {
        assert_send::<ProfileStats>();
        assert_sync::<ProfileStats>();
    }

    // =========== Edge case tests ===========

    #[test]
    fn timestamp_result_max_duration() {
        let result = TimestampResult::new("Max", 0, u64::MAX);
        assert!(result.duration_ns() > 0);
        assert!(result.duration_ms() > 0.0);
    }

    #[test]
    fn profile_stats_single_zero_duration() {
        let result = TimestampResult::zero("Zero");
        let stats = ProfileStats::from_results(&[&result]);
        assert_eq!(stats.sample_count, 1);
        assert_eq!(stats.total_duration_ms, 0.0);
    }

    #[test]
    fn timestamp_query_unicode_name() {
        let query = TimestampQuery::new("Shadow Pass [日本語]", 0, 1);
        assert!(query.name().contains("日本語"));
    }

    #[test]
    fn timestamp_result_unicode_name() {
        let result = TimestampResult::new("Тест", 0, 1000);
        assert!(result.name().contains("Тест"));
    }

    #[test]
    fn profile_stats_from_empty_slice() {
        let stats = ProfileStats::from_results(&[]);
        assert!(stats.is_empty());
    }

    #[test]
    fn profile_stats_merge_with_self() {
        let r1 = TimestampResult::new("R1", 0, 1_000_000);
        let stats = ProfileStats::from_results(&[&r1]);
        let merged = stats.merge(&stats);
        assert_eq!(merged.sample_count, 2);
    }
}
