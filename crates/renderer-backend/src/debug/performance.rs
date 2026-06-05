//! GPU Performance Markers and Timing System for wgpu 25.x
//!
//! This module provides a comprehensive GPU profiling system with timestamp queries,
//! performance markers, and frame-level statistics.
//!
//! # Overview
//!
//! - [`TimestampQuery`] - Low-level timestamp query management
//! - [`GpuTimer`] - Named timer with start/end indices
//! - [`PerformanceMarker`] - Completed timing data with category
//! - [`MarkerCategory`] - Classification of GPU work types
//! - [`PerformanceProfiler`] - High-level profiling API
//!
//! # Architecture
//!
//! ```text
//! PerformanceProfiler
//!     |-- timestamps: TimestampQuery
//!     |-- active_timers: Vec<GpuTimer>
//!     |-- completed_markers: Vec<PerformanceMarker>
//!     `-- frame_count: u64
//!
//! TimestampQuery
//!     |-- query_set: wgpu::QuerySet
//!     |-- capacity: u32
//!     |-- next_index: u32
//!     `-- resolved: bool
//!
//! GpuTimer
//!     |-- name: String
//!     |-- start_index: u32
//!     |-- end_index: Option<u32>
//!     `-- category: MarkerCategory
//!
//! PerformanceMarker
//!     |-- name: String
//!     |-- timestamp_ns: u64
//!     |-- duration_ns: Option<u64>
//!     `-- category: MarkerCategory
//! ```
//!
//! # Example
//!
//! ```ignore
//! use renderer_backend::debug::performance::*;
//!
//! // Create profiler with capacity for 256 timestamps
//! let mut profiler = PerformanceProfiler::new(&device, 256);
//!
//! // Begin frame
//! profiler.begin_frame();
//!
//! // Profile a render pass
//! let timer_id = profiler.begin_marker(&mut encoder, "GBuffer Pass", MarkerCategory::Pass);
//! // ... render commands ...
//! if let Some(id) = timer_id {
//!     profiler.end_marker(&mut encoder, id);
//! }
//!
//! // End frame and process results
//! profiler.end_frame();
//!
//! // Access completed markers
//! for marker in profiler.get_markers() {
//!     println!("{}: {:?}ns", marker.name, marker.duration_ns);
//! }
//! ```
//!
//! # Thread Safety
//!
//! - `TimestampQuery`: Not thread-safe (contains QuerySet)
//! - `GpuTimer`: `Send + Sync` (contains only owned data)
//! - `PerformanceMarker`: `Send + Sync + Clone`
//! - `MarkerCategory`: `Send + Sync + Copy`
//! - `PerformanceProfiler`: Not thread-safe (manages GPU resources)

use std::fmt;

// ============================================================================
// MarkerCategory
// ============================================================================

/// Classification of GPU work types for performance markers.
///
/// Categories help organize and filter profiling data by the type of work
/// being performed on the GPU.
///
/// # Example
///
/// ```ignore
/// use renderer_backend::debug::performance::MarkerCategory;
///
/// let category = MarkerCategory::Pass;
/// assert_eq!(category.as_str(), "Pass");
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum MarkerCategory {
    /// Full frame timing
    Frame,
    /// Render or compute pass
    #[default]
    Pass,
    /// Draw call or draw batch
    Draw,
    /// Compute dispatch
    Compute,
    /// Data transfer (copy, upload, readback)
    Transfer,
    /// Barrier or synchronization
    Barrier,
    /// User-defined custom marker
    Custom,
}

impl MarkerCategory {
    /// Get the category as a static string.
    ///
    /// # Returns
    ///
    /// A static string representation of the category.
    #[inline]
    pub const fn as_str(&self) -> &'static str {
        match self {
            MarkerCategory::Frame => "Frame",
            MarkerCategory::Pass => "Pass",
            MarkerCategory::Draw => "Draw",
            MarkerCategory::Compute => "Compute",
            MarkerCategory::Transfer => "Transfer",
            MarkerCategory::Barrier => "Barrier",
            MarkerCategory::Custom => "Custom",
        }
    }

    /// Get all marker categories.
    ///
    /// # Returns
    ///
    /// An array of all marker category variants.
    #[inline]
    pub const fn all() -> [MarkerCategory; 7] {
        [
            MarkerCategory::Frame,
            MarkerCategory::Pass,
            MarkerCategory::Draw,
            MarkerCategory::Compute,
            MarkerCategory::Transfer,
            MarkerCategory::Barrier,
            MarkerCategory::Custom,
        ]
    }

    /// Check if this is a high-level category (Frame or Pass).
    #[inline]
    pub const fn is_high_level(&self) -> bool {
        matches!(self, MarkerCategory::Frame | MarkerCategory::Pass)
    }

    /// Check if this is a low-level category (Draw, Compute, Transfer, Barrier).
    #[inline]
    pub const fn is_low_level(&self) -> bool {
        matches!(
            self,
            MarkerCategory::Draw
                | MarkerCategory::Compute
                | MarkerCategory::Transfer
                | MarkerCategory::Barrier
        )
    }
}

impl fmt::Display for MarkerCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

// ============================================================================
// TimestampQuery
// ============================================================================

/// Low-level timestamp query management.
///
/// Wraps a `wgpu::QuerySet` for timestamp queries, providing index allocation,
/// resolution, and reset functionality.
///
/// # Example
///
/// ```ignore
/// use renderer_backend::debug::performance::TimestampQuery;
///
/// let mut timestamps = TimestampQuery::new(&device, 64);
///
/// // Write timestamp
/// let idx = timestamps.write_timestamp(&mut encoder);
///
/// // Resolve to buffer
/// timestamps.resolve(&mut encoder, &resolve_buffer);
///
/// // Reset for next frame
/// timestamps.reset();
/// ```
pub struct TimestampQuery {
    /// The underlying wgpu query set
    query_set: wgpu::QuerySet,
    /// Maximum number of timestamps
    capacity: u32,
    /// Next available index
    next_index: u32,
    /// Whether results have been resolved
    resolved: bool,
}

impl TimestampQuery {
    /// Create a new timestamp query set.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device
    /// * `capacity` - Maximum number of timestamps (must be > 0)
    ///
    /// # Returns
    ///
    /// A new `TimestampQuery` instance.
    ///
    /// # Panics
    ///
    /// Panics if capacity is 0.
    pub fn new(device: &wgpu::Device, capacity: u32) -> Self {
        assert!(capacity > 0, "TimestampQuery capacity must be > 0");

        let query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("PerformanceProfiler_TimestampQuery"),
            ty: wgpu::QueryType::Timestamp,
            count: capacity,
        });

        Self {
            query_set,
            capacity,
            next_index: 0,
            resolved: false,
        }
    }

    /// Write a timestamp to the query set.
    ///
    /// # Arguments
    ///
    /// * `encoder` - The command encoder to write the timestamp
    ///
    /// # Returns
    ///
    /// The query index if successful, `None` if the query set is full.
    pub fn write_timestamp(&mut self, encoder: &mut wgpu::CommandEncoder) -> Option<u32> {
        if self.next_index >= self.capacity {
            return None;
        }

        let index = self.next_index;
        encoder.write_timestamp(&self.query_set, index);
        self.next_index += 1;
        Some(index)
    }

    /// Resolve timestamp queries to a buffer.
    ///
    /// The buffer must be at least `count * 8` bytes (u64 per timestamp).
    ///
    /// # Arguments
    ///
    /// * `encoder` - The command encoder
    /// * `buffer` - The destination buffer for resolved timestamps
    pub fn resolve(&mut self, encoder: &mut wgpu::CommandEncoder, buffer: &wgpu::Buffer) {
        if self.next_index > 0 && !self.resolved {
            encoder.resolve_query_set(&self.query_set, 0..self.next_index, buffer, 0);
            self.resolved = true;
        }
    }

    /// Reset the query set for the next frame.
    ///
    /// Resets the index counter and resolved flag. Does not clear GPU-side data.
    pub fn reset(&mut self) {
        self.next_index = 0;
        self.resolved = false;
    }

    /// Get the query set capacity.
    #[inline]
    pub fn capacity(&self) -> u32 {
        self.capacity
    }

    /// Get the number of timestamps written.
    #[inline]
    pub fn count(&self) -> u32 {
        self.next_index
    }

    /// Check if the query set is full.
    #[inline]
    pub fn is_full(&self) -> bool {
        self.next_index >= self.capacity
    }

    /// Check if queries have been resolved.
    #[inline]
    pub fn is_resolved(&self) -> bool {
        self.resolved
    }

    /// Get a reference to the underlying query set.
    #[inline]
    pub fn query_set(&self) -> &wgpu::QuerySet {
        &self.query_set
    }

    /// Get remaining capacity.
    #[inline]
    pub fn remaining(&self) -> u32 {
        self.capacity.saturating_sub(self.next_index)
    }
}

impl fmt::Debug for TimestampQuery {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TimestampQuery")
            .field("capacity", &self.capacity)
            .field("next_index", &self.next_index)
            .field("resolved", &self.resolved)
            .finish()
    }
}

// ============================================================================
// GpuTimer
// ============================================================================

/// A named GPU timer with start and optional end timestamp indices.
///
/// Used to track in-flight timing measurements before they are resolved.
///
/// # Example
///
/// ```ignore
/// use renderer_backend::debug::performance::{GpuTimer, MarkerCategory};
///
/// let mut timer = GpuTimer::new("Shadow Pass", 0, MarkerCategory::Pass);
/// assert!(!timer.is_complete());
///
/// timer.stop(1);
/// assert!(timer.is_complete());
/// ```
#[derive(Clone, Debug)]
pub struct GpuTimer {
    /// Timer name
    name: String,
    /// Start timestamp index
    start_index: u32,
    /// End timestamp index (if stopped)
    end_index: Option<u32>,
    /// Category of the timed work
    category: MarkerCategory,
}

impl GpuTimer {
    /// Create a new GPU timer.
    ///
    /// # Arguments
    ///
    /// * `name` - Name of the timer
    /// * `start_index` - Query index for the start timestamp
    /// * `category` - Category of the timed work
    ///
    /// # Returns
    ///
    /// A new `GpuTimer` instance.
    #[inline]
    pub fn new(name: &str, start_index: u32, category: MarkerCategory) -> Self {
        Self {
            name: name.to_string(),
            start_index,
            end_index: None,
            category,
        }
    }

    /// Stop the timer with an end timestamp index.
    ///
    /// # Arguments
    ///
    /// * `end_index` - Query index for the end timestamp
    #[inline]
    pub fn stop(&mut self, end_index: u32) {
        self.end_index = Some(end_index);
    }

    /// Check if the timer has both start and end timestamps.
    #[inline]
    pub fn is_complete(&self) -> bool {
        self.end_index.is_some()
    }

    /// Get the timer name.
    #[inline]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get the start timestamp index.
    #[inline]
    pub fn start_index(&self) -> u32 {
        self.start_index
    }

    /// Get the end timestamp index, if set.
    #[inline]
    pub fn end_index(&self) -> Option<u32> {
        self.end_index
    }

    /// Get the category of the timed work.
    #[inline]
    pub fn category(&self) -> MarkerCategory {
        self.category
    }

    /// Get the query index range if complete.
    #[inline]
    pub fn index_range(&self) -> Option<(u32, u32)> {
        self.end_index.map(|end| (self.start_index, end))
    }
}

impl fmt::Display for GpuTimer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.end_index {
            Some(end) => write!(
                f,
                "GpuTimer({}, [{} -> {}], {})",
                self.name, self.start_index, end, self.category
            ),
            None => write!(
                f,
                "GpuTimer({}, [{} -> ?], {})",
                self.name, self.start_index, self.category
            ),
        }
    }
}

// ============================================================================
// PerformanceMarker
// ============================================================================

/// A completed performance marker with timing data.
///
/// Created from resolved GPU timestamps, containing the actual timing
/// measurements for a profiled operation.
///
/// # Example
///
/// ```ignore
/// use renderer_backend::debug::performance::{PerformanceMarker, MarkerCategory};
///
/// let marker = PerformanceMarker::new(
///     "GBuffer Pass",
///     1000000,           // 1ms start
///     Some(2500000),     // 2.5ms end -> 1.5ms duration
///     MarkerCategory::Pass,
/// );
///
/// assert_eq!(marker.duration_ns, Some(1500000));
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PerformanceMarker {
    /// Name of the marked region
    pub name: String,
    /// Start timestamp in nanoseconds
    pub timestamp_ns: u64,
    /// Duration in nanoseconds (if end timestamp available)
    pub duration_ns: Option<u64>,
    /// Category of the marked work
    pub category: MarkerCategory,
}

impl PerformanceMarker {
    /// Create a new performance marker.
    ///
    /// # Arguments
    ///
    /// * `name` - Name of the marker
    /// * `timestamp_ns` - Start timestamp in nanoseconds
    /// * `end_timestamp_ns` - Optional end timestamp in nanoseconds
    /// * `category` - Category of the work
    ///
    /// # Returns
    ///
    /// A new `PerformanceMarker` with calculated duration.
    pub fn new(
        name: impl Into<String>,
        timestamp_ns: u64,
        end_timestamp_ns: Option<u64>,
        category: MarkerCategory,
    ) -> Self {
        let duration_ns = end_timestamp_ns.map(|end| end.saturating_sub(timestamp_ns));

        Self {
            name: name.into(),
            timestamp_ns,
            duration_ns,
            category,
        }
    }

    /// Create a point marker (no duration).
    ///
    /// # Arguments
    ///
    /// * `name` - Name of the marker
    /// * `timestamp_ns` - Timestamp in nanoseconds
    /// * `category` - Category of the work
    #[inline]
    pub fn point(name: impl Into<String>, timestamp_ns: u64, category: MarkerCategory) -> Self {
        Self {
            name: name.into(),
            timestamp_ns,
            duration_ns: None,
            category,
        }
    }

    /// Create a range marker with explicit duration.
    ///
    /// # Arguments
    ///
    /// * `name` - Name of the marker
    /// * `timestamp_ns` - Start timestamp in nanoseconds
    /// * `duration_ns` - Duration in nanoseconds
    /// * `category` - Category of the work
    #[inline]
    pub fn range(
        name: impl Into<String>,
        timestamp_ns: u64,
        duration_ns: u64,
        category: MarkerCategory,
    ) -> Self {
        Self {
            name: name.into(),
            timestamp_ns,
            duration_ns: Some(duration_ns),
            category,
        }
    }

    /// Get the duration in milliseconds.
    #[inline]
    pub fn duration_ms(&self) -> Option<f64> {
        self.duration_ns.map(|ns| ns as f64 / 1_000_000.0)
    }

    /// Get the duration in microseconds.
    #[inline]
    pub fn duration_us(&self) -> Option<f64> {
        self.duration_ns.map(|ns| ns as f64 / 1_000.0)
    }

    /// Check if this marker has duration data.
    #[inline]
    pub fn has_duration(&self) -> bool {
        self.duration_ns.is_some()
    }

    /// Get the end timestamp in nanoseconds.
    #[inline]
    pub fn end_timestamp_ns(&self) -> Option<u64> {
        self.duration_ns.map(|d| self.timestamp_ns + d)
    }
}

impl fmt::Display for PerformanceMarker {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.duration_ns {
            Some(ns) => {
                let ms = ns as f64 / 1_000_000.0;
                write!(f, "[{}] {}: {:.3}ms", self.category, self.name, ms)
            }
            None => write!(f, "[{}] {}: @{}ns", self.category, self.name, self.timestamp_ns),
        }
    }
}

// ============================================================================
// PerformanceProfiler
// ============================================================================

/// High-level GPU performance profiler.
///
/// Manages timestamp queries, active timers, and completed markers across frames.
///
/// # Example
///
/// ```ignore
/// use renderer_backend::debug::performance::*;
///
/// let mut profiler = PerformanceProfiler::new(&device, 256);
///
/// // Each frame:
/// profiler.begin_frame();
///
/// // Profile operations
/// let id = profiler.begin_marker(&mut encoder, "Shadow Pass", MarkerCategory::Pass);
/// // ... GPU commands ...
/// if let Some(id) = id {
///     profiler.end_marker(&mut encoder, id);
/// }
///
/// profiler.end_frame();
///
/// // Access results
/// for marker in profiler.get_markers() {
///     println!("{}", marker);
/// }
/// ```
pub struct PerformanceProfiler {
    /// Timestamp query management
    timestamps: TimestampQuery,
    /// Currently active (incomplete) timers
    active_timers: Vec<GpuTimer>,
    /// Completed performance markers
    completed_markers: Vec<PerformanceMarker>,
    /// Number of frames profiled
    frame_count: u64,
    /// Whether we're currently in a frame
    in_frame: bool,
}

impl PerformanceProfiler {
    /// Create a new performance profiler.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device
    /// * `capacity` - Maximum number of timestamps per frame
    ///
    /// # Returns
    ///
    /// A new `PerformanceProfiler` instance.
    pub fn new(device: &wgpu::Device, capacity: u32) -> Self {
        Self {
            timestamps: TimestampQuery::new(device, capacity),
            active_timers: Vec::with_capacity(32),
            completed_markers: Vec::with_capacity(64),
            frame_count: 0,
            in_frame: false,
        }
    }

    /// Begin a new frame.
    ///
    /// Resets timestamp queries and clears active timers.
    pub fn begin_frame(&mut self) {
        self.timestamps.reset();
        self.active_timers.clear();
        self.in_frame = true;
    }

    /// End the current frame.
    ///
    /// Increments frame counter and marks frame as ended.
    pub fn end_frame(&mut self) {
        self.frame_count += 1;
        self.in_frame = false;
    }

    /// Begin a performance marker.
    ///
    /// Writes a start timestamp and creates an active timer.
    ///
    /// # Arguments
    ///
    /// * `encoder` - The command encoder
    /// * `name` - Name of the marker
    /// * `category` - Category of the work
    ///
    /// # Returns
    ///
    /// Timer index if successful, `None` if query set is full.
    pub fn begin_marker(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        name: &str,
        category: MarkerCategory,
    ) -> Option<usize> {
        let start_index = self.timestamps.write_timestamp(encoder)?;
        let timer = GpuTimer::new(name, start_index, category);
        let timer_id = self.active_timers.len();
        self.active_timers.push(timer);
        Some(timer_id)
    }

    /// End a performance marker.
    ///
    /// Writes an end timestamp to the specified timer.
    ///
    /// # Arguments
    ///
    /// * `encoder` - The command encoder
    /// * `timer_id` - Index returned from `begin_marker`
    pub fn end_marker(&mut self, encoder: &mut wgpu::CommandEncoder, timer_id: usize) {
        if timer_id < self.active_timers.len() {
            if let Some(end_index) = self.timestamps.write_timestamp(encoder) {
                self.active_timers[timer_id].stop(end_index);
            }
        }
    }

    /// Insert a point marker (no duration).
    ///
    /// # Arguments
    ///
    /// * `encoder` - The command encoder
    /// * `name` - Name of the marker
    /// * `category` - Category of the work
    ///
    /// # Returns
    ///
    /// Query index if successful, `None` if query set is full.
    pub fn insert_marker(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        name: &str,
        category: MarkerCategory,
    ) -> Option<u32> {
        let index = self.timestamps.write_timestamp(encoder)?;
        // Point markers are stored directly as completed markers without duration
        self.completed_markers.push(PerformanceMarker::point(
            name.to_string(),
            index as u64, // Placeholder - actual timestamp from resolution
            category,
        ));
        Some(index)
    }

    /// Resolve timestamps to a buffer.
    ///
    /// # Arguments
    ///
    /// * `encoder` - The command encoder
    /// * `buffer` - The destination buffer for resolved timestamps
    pub fn resolve(&mut self, encoder: &mut wgpu::CommandEncoder, buffer: &wgpu::Buffer) {
        self.timestamps.resolve(encoder, buffer);
    }

    /// Process resolved timestamps and create performance markers.
    ///
    /// Call this after mapping the resolve buffer and reading the data.
    ///
    /// # Arguments
    ///
    /// * `timestamps` - Slice of resolved timestamp values (u64 per query)
    /// * `timestamp_period` - Nanoseconds per timestamp tick
    pub fn process_results(&mut self, timestamps: &[u64], timestamp_period: f32) {
        for timer in &self.active_timers {
            if let Some((start, end)) = timer.index_range() {
                if (start as usize) < timestamps.len() && (end as usize) < timestamps.len() {
                    let start_ticks = timestamps[start as usize];
                    let end_ticks = timestamps[end as usize];
                    let start_ns = (start_ticks as f64 * timestamp_period as f64) as u64;
                    let end_ns = (end_ticks as f64 * timestamp_period as f64) as u64;

                    self.completed_markers.push(PerformanceMarker::new(
                        timer.name(),
                        start_ns,
                        Some(end_ns),
                        timer.category(),
                    ));
                }
            }
        }
    }

    /// Get completed performance markers.
    #[inline]
    pub fn get_markers(&self) -> &[PerformanceMarker] {
        &self.completed_markers
    }

    /// Get mutable access to completed markers.
    #[inline]
    pub fn get_markers_mut(&mut self) -> &mut Vec<PerformanceMarker> {
        &mut self.completed_markers
    }

    /// Clear all completed markers.
    pub fn clear_markers(&mut self) {
        self.completed_markers.clear();
    }

    /// Get the frame count.
    #[inline]
    pub fn frame_count(&self) -> u64 {
        self.frame_count
    }

    /// Check if currently in a frame.
    #[inline]
    pub fn in_frame(&self) -> bool {
        self.in_frame
    }

    /// Get the number of active (incomplete) timers.
    #[inline]
    pub fn active_timer_count(&self) -> usize {
        self.active_timers.len()
    }

    /// Get the number of timestamps written this frame.
    #[inline]
    pub fn timestamp_count(&self) -> u32 {
        self.timestamps.count()
    }

    /// Get the timestamp query capacity.
    #[inline]
    pub fn capacity(&self) -> u32 {
        self.timestamps.capacity()
    }

    /// Get remaining timestamp capacity.
    #[inline]
    pub fn remaining_capacity(&self) -> u32 {
        self.timestamps.remaining()
    }

    /// Check if the profiler can record more timestamps.
    #[inline]
    pub fn can_record(&self) -> bool {
        !self.timestamps.is_full()
    }

    /// Get a reference to the underlying query set.
    #[inline]
    pub fn query_set(&self) -> &wgpu::QuerySet {
        self.timestamps.query_set()
    }

    /// Get markers filtered by category.
    pub fn get_markers_by_category(&self, category: MarkerCategory) -> Vec<&PerformanceMarker> {
        self.completed_markers
            .iter()
            .filter(|m| m.category == category)
            .collect()
    }

    /// Calculate total time for a category.
    pub fn total_time_for_category(&self, category: MarkerCategory) -> u64 {
        self.completed_markers
            .iter()
            .filter(|m| m.category == category)
            .filter_map(|m| m.duration_ns)
            .sum()
    }

    /// Get markers sorted by duration (descending).
    pub fn get_markers_sorted_by_duration(&self) -> Vec<&PerformanceMarker> {
        let mut markers: Vec<_> = self.completed_markers.iter().collect();
        markers.sort_by(|a, b| {
            b.duration_ns
                .unwrap_or(0)
                .cmp(&a.duration_ns.unwrap_or(0))
        });
        markers
    }
}

impl fmt::Debug for PerformanceProfiler {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PerformanceProfiler")
            .field("timestamps", &self.timestamps)
            .field("active_timers", &self.active_timers.len())
            .field("completed_markers", &self.completed_markers.len())
            .field("frame_count", &self.frame_count)
            .field("in_frame", &self.in_frame)
            .finish()
    }
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // --------------------------------------------------------------------------
    // MarkerCategory Tests
    // --------------------------------------------------------------------------

    #[test]
    fn test_marker_category_as_str() {
        assert_eq!(MarkerCategory::Frame.as_str(), "Frame");
        assert_eq!(MarkerCategory::Pass.as_str(), "Pass");
        assert_eq!(MarkerCategory::Draw.as_str(), "Draw");
        assert_eq!(MarkerCategory::Compute.as_str(), "Compute");
        assert_eq!(MarkerCategory::Transfer.as_str(), "Transfer");
        assert_eq!(MarkerCategory::Barrier.as_str(), "Barrier");
        assert_eq!(MarkerCategory::Custom.as_str(), "Custom");
    }

    #[test]
    fn test_marker_category_all() {
        let all = MarkerCategory::all();
        assert_eq!(all.len(), 7);
        assert!(all.contains(&MarkerCategory::Frame));
        assert!(all.contains(&MarkerCategory::Pass));
        assert!(all.contains(&MarkerCategory::Draw));
        assert!(all.contains(&MarkerCategory::Compute));
        assert!(all.contains(&MarkerCategory::Transfer));
        assert!(all.contains(&MarkerCategory::Barrier));
        assert!(all.contains(&MarkerCategory::Custom));
    }

    #[test]
    fn test_marker_category_is_high_level() {
        assert!(MarkerCategory::Frame.is_high_level());
        assert!(MarkerCategory::Pass.is_high_level());
        assert!(!MarkerCategory::Draw.is_high_level());
        assert!(!MarkerCategory::Compute.is_high_level());
        assert!(!MarkerCategory::Transfer.is_high_level());
        assert!(!MarkerCategory::Barrier.is_high_level());
        assert!(!MarkerCategory::Custom.is_high_level());
    }

    #[test]
    fn test_marker_category_is_low_level() {
        assert!(!MarkerCategory::Frame.is_low_level());
        assert!(!MarkerCategory::Pass.is_low_level());
        assert!(MarkerCategory::Draw.is_low_level());
        assert!(MarkerCategory::Compute.is_low_level());
        assert!(MarkerCategory::Transfer.is_low_level());
        assert!(MarkerCategory::Barrier.is_low_level());
        assert!(!MarkerCategory::Custom.is_low_level());
    }

    #[test]
    fn test_marker_category_default() {
        let category: MarkerCategory = Default::default();
        assert_eq!(category, MarkerCategory::Pass);
    }

    #[test]
    fn test_marker_category_display() {
        assert_eq!(format!("{}", MarkerCategory::Frame), "Frame");
        assert_eq!(format!("{}", MarkerCategory::Custom), "Custom");
    }

    #[test]
    fn test_marker_category_clone_copy() {
        let category = MarkerCategory::Compute;
        let cloned = category.clone();
        let copied = category;
        assert_eq!(category, cloned);
        assert_eq!(category, copied);
    }

    #[test]
    fn test_marker_category_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(MarkerCategory::Frame);
        set.insert(MarkerCategory::Pass);
        set.insert(MarkerCategory::Frame); // Duplicate
        assert_eq!(set.len(), 2);
    }

    // --------------------------------------------------------------------------
    // GpuTimer Tests
    // --------------------------------------------------------------------------

    #[test]
    fn test_gpu_timer_new() {
        let timer = GpuTimer::new("Shadow Pass", 5, MarkerCategory::Pass);
        assert_eq!(timer.name(), "Shadow Pass");
        assert_eq!(timer.start_index(), 5);
        assert_eq!(timer.end_index(), None);
        assert_eq!(timer.category(), MarkerCategory::Pass);
        assert!(!timer.is_complete());
    }

    #[test]
    fn test_gpu_timer_stop() {
        let mut timer = GpuTimer::new("Lighting", 0, MarkerCategory::Compute);
        assert!(!timer.is_complete());
        assert_eq!(timer.index_range(), None);

        timer.stop(10);
        assert!(timer.is_complete());
        assert_eq!(timer.end_index(), Some(10));
        assert_eq!(timer.index_range(), Some((0, 10)));
    }

    #[test]
    fn test_gpu_timer_display() {
        let mut timer = GpuTimer::new("Test", 2, MarkerCategory::Draw);
        let incomplete = format!("{}", timer);
        assert!(incomplete.contains("Test"));
        assert!(incomplete.contains("2 -> ?"));

        timer.stop(5);
        let complete = format!("{}", timer);
        assert!(complete.contains("2 -> 5"));
    }

    #[test]
    fn test_gpu_timer_clone() {
        let timer = GpuTimer::new("Original", 0, MarkerCategory::Transfer);
        let cloned = timer.clone();
        assert_eq!(timer.name(), cloned.name());
        assert_eq!(timer.start_index(), cloned.start_index());
        assert_eq!(timer.category(), cloned.category());
    }

    // --------------------------------------------------------------------------
    // PerformanceMarker Tests
    // --------------------------------------------------------------------------

    #[test]
    fn test_performance_marker_new() {
        let marker = PerformanceMarker::new("GBuffer", 1_000_000, Some(2_500_000), MarkerCategory::Pass);
        assert_eq!(marker.name, "GBuffer");
        assert_eq!(marker.timestamp_ns, 1_000_000);
        assert_eq!(marker.duration_ns, Some(1_500_000));
        assert_eq!(marker.category, MarkerCategory::Pass);
    }

    #[test]
    fn test_performance_marker_point() {
        let marker = PerformanceMarker::point("Sync", 5_000_000, MarkerCategory::Barrier);
        assert_eq!(marker.name, "Sync");
        assert_eq!(marker.timestamp_ns, 5_000_000);
        assert_eq!(marker.duration_ns, None);
        assert!(!marker.has_duration());
    }

    #[test]
    fn test_performance_marker_range() {
        let marker = PerformanceMarker::range("Upload", 0, 2_000_000, MarkerCategory::Transfer);
        assert_eq!(marker.name, "Upload");
        assert_eq!(marker.timestamp_ns, 0);
        assert_eq!(marker.duration_ns, Some(2_000_000));
        assert!(marker.has_duration());
    }

    #[test]
    fn test_performance_marker_duration_conversions() {
        let marker = PerformanceMarker::range("Test", 0, 1_500_000, MarkerCategory::Custom);

        let ms = marker.duration_ms().unwrap();
        assert!((ms - 1.5).abs() < 0.001);

        let us = marker.duration_us().unwrap();
        assert!((us - 1500.0).abs() < 0.1);
    }

    #[test]
    fn test_performance_marker_end_timestamp() {
        let marker = PerformanceMarker::new("Test", 1_000_000, Some(3_000_000), MarkerCategory::Frame);
        assert_eq!(marker.end_timestamp_ns(), Some(3_000_000));

        let point = PerformanceMarker::point("Point", 1_000_000, MarkerCategory::Custom);
        assert_eq!(point.end_timestamp_ns(), None);
    }

    #[test]
    fn test_performance_marker_display() {
        let with_duration = PerformanceMarker::range("Test", 0, 1_500_000, MarkerCategory::Pass);
        let display = format!("{}", with_duration);
        assert!(display.contains("[Pass]"));
        assert!(display.contains("Test"));
        assert!(display.contains("1.500ms"));

        let point = PerformanceMarker::point("Point", 5_000, MarkerCategory::Barrier);
        let point_display = format!("{}", point);
        assert!(point_display.contains("@5000ns"));
    }

    #[test]
    fn test_performance_marker_clone_eq() {
        let marker = PerformanceMarker::range("Test", 100, 200, MarkerCategory::Draw);
        let cloned = marker.clone();
        assert_eq!(marker, cloned);
    }

    #[test]
    fn test_performance_marker_saturating_sub() {
        // Test that duration calculation handles wrap-around gracefully
        let marker = PerformanceMarker::new("Test", 100, Some(50), MarkerCategory::Custom);
        assert_eq!(marker.duration_ns, Some(0)); // Saturating sub prevents underflow
    }

    // --------------------------------------------------------------------------
    // Integration-style tests (no GPU required)
    // --------------------------------------------------------------------------

    #[test]
    fn test_marker_categories_comprehensive() {
        // Verify all categories have unique string representations
        let all = MarkerCategory::all();
        let strings: Vec<_> = all.iter().map(|c| c.as_str()).collect();
        let unique: std::collections::HashSet<_> = strings.iter().collect();
        assert_eq!(strings.len(), unique.len(), "All categories should have unique strings");
    }

    #[test]
    fn test_gpu_timer_lifecycle() {
        // Simulate timer lifecycle
        let mut timer = GpuTimer::new("Full Lifecycle", 0, MarkerCategory::Frame);

        // Initial state
        assert!(!timer.is_complete());
        assert_eq!(timer.index_range(), None);

        // Stop timer
        timer.stop(99);

        // Final state
        assert!(timer.is_complete());
        assert_eq!(timer.index_range(), Some((0, 99)));
        assert_eq!(timer.start_index(), 0);
        assert_eq!(timer.end_index(), Some(99));
    }

    #[test]
    fn test_performance_marker_sorting() {
        let markers = vec![
            PerformanceMarker::range("Short", 0, 100_000, MarkerCategory::Draw),
            PerformanceMarker::range("Long", 0, 500_000, MarkerCategory::Pass),
            PerformanceMarker::range("Medium", 0, 250_000, MarkerCategory::Compute),
            PerformanceMarker::point("Point", 0, MarkerCategory::Barrier),
        ];

        let mut sorted = markers.clone();
        sorted.sort_by(|a, b| {
            b.duration_ns.unwrap_or(0).cmp(&a.duration_ns.unwrap_or(0))
        });

        assert_eq!(sorted[0].name, "Long");
        assert_eq!(sorted[1].name, "Medium");
        assert_eq!(sorted[2].name, "Short");
        assert_eq!(sorted[3].name, "Point");
    }

    #[test]
    fn test_performance_marker_filtering() {
        let markers = vec![
            PerformanceMarker::range("Draw1", 0, 100, MarkerCategory::Draw),
            PerformanceMarker::range("Pass1", 0, 200, MarkerCategory::Pass),
            PerformanceMarker::range("Draw2", 0, 150, MarkerCategory::Draw),
            PerformanceMarker::range("Compute1", 0, 300, MarkerCategory::Compute),
        ];

        let draws: Vec<_> = markers
            .iter()
            .filter(|m| m.category == MarkerCategory::Draw)
            .collect();

        assert_eq!(draws.len(), 2);
        assert!(draws.iter().all(|m| m.category == MarkerCategory::Draw));
    }

    #[test]
    fn test_total_time_calculation() {
        let markers = vec![
            PerformanceMarker::range("Draw1", 0, 100_000, MarkerCategory::Draw),
            PerformanceMarker::range("Draw2", 0, 200_000, MarkerCategory::Draw),
            PerformanceMarker::range("Pass1", 0, 300_000, MarkerCategory::Pass),
        ];

        let total_draw: u64 = markers
            .iter()
            .filter(|m| m.category == MarkerCategory::Draw)
            .filter_map(|m| m.duration_ns)
            .sum();

        assert_eq!(total_draw, 300_000);
    }
}
