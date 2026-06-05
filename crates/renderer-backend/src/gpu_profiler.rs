//! GPU profiler using wgpu timestamp queries.
//!
//! This module provides GPU-side profiling infrastructure that measures pass
//! execution times on the GPU using hardware timestamp queries. Results are
//! read back with a 1-3 frame delay to handle GPU-CPU latency.
//!
//! # Architecture
//!
//! - `GPUProfiler` manages a pool of `QuerySet` objects for timestamp queries
//! - Each render pass is bracketed by begin/end timestamp writes
//! - Query results are resolved to a buffer and read back after N frames
//! - Ring buffer handles the deferred readback latency gracefully
//!
//! # Usage
//!
//! ```ignore
//! let mut profiler = GPUProfiler::new(&device, 16);
//!
//! // In render loop:
//! profiler.begin_frame();
//!
//! profiler.begin_pass(&mut encoder, "GBuffer");
//! // ... render GBuffer pass ...
//! profiler.end_pass(&mut encoder);
//!
//! profiler.begin_pass(&mut encoder, "Lighting");
//! // ... render lighting pass ...
//! profiler.end_pass(&mut encoder);
//!
//! profiler.resolve(&mut encoder);
//!
//! // After submitting:
//! if let Some(profile) = profiler.read_results(&queue) {
//!     println!("Frame {} GPU time: {:.2}ms", profile.frame_id, profile.total_gpu_ms);
//! }
//! ```
//!
//! # Graceful Degradation
//!
//! If the device doesn't support timestamp queries, the profiler operates in
//! "no-op" mode: all methods are safe to call but return empty/zero results.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Default number of frames to buffer for deferred readback.
pub const DEFAULT_RING_SIZE: usize = 4;

/// Maximum passes per frame (determines QuerySet size).
pub const MAX_PASSES_PER_FRAME: usize = 64;

// ---------------------------------------------------------------------------
// PassTiming
// ---------------------------------------------------------------------------

/// Timing data for a single render pass.
#[derive(Debug, Clone, PartialEq)]
pub struct PassTiming {
    /// Human-readable pass name.
    pub name: String,
    /// GPU execution time in milliseconds.
    pub gpu_ms: f32,
    /// CPU submission time in milliseconds (encoder work).
    pub cpu_ms: f32,
}

impl PassTiming {
    /// Create a new pass timing entry.
    pub fn new(name: impl Into<String>, gpu_ms: f32, cpu_ms: f32) -> Self {
        Self {
            name: name.into(),
            gpu_ms,
            cpu_ms,
        }
    }
}

// ---------------------------------------------------------------------------
// FrameProfile
// ---------------------------------------------------------------------------

/// Complete GPU timing profile for a single frame.
#[derive(Debug, Clone, PartialEq)]
pub struct FrameProfile {
    /// Frame identifier (monotonically increasing).
    pub frame_id: u64,
    /// Individual pass timing measurements.
    pub pass_timings: Vec<PassTiming>,
    /// Total GPU time for the frame (sum of all passes) in milliseconds.
    pub total_gpu_ms: f32,
}

impl FrameProfile {
    /// Create an empty frame profile.
    pub fn new(frame_id: u64) -> Self {
        Self {
            frame_id,
            pass_timings: Vec::new(),
            total_gpu_ms: 0.0,
        }
    }

    /// Add a pass timing and update total.
    pub fn add_pass(&mut self, timing: PassTiming) {
        self.total_gpu_ms += timing.gpu_ms;
        self.pass_timings.push(timing);
    }

    /// Number of passes recorded.
    pub fn pass_count(&self) -> usize {
        self.pass_timings.len()
    }
}

// ---------------------------------------------------------------------------
// QueryPool
// ---------------------------------------------------------------------------

/// Pool entry tracking a single QuerySet and its resolve buffer.
struct QuerySetEntry {
    /// The wgpu QuerySet for timestamp queries.
    query_set: wgpu::QuerySet,
    /// Buffer to resolve query results into.
    resolve_buffer: wgpu::Buffer,
    /// Staging buffer for CPU readback.
    staging_buffer: wgpu::Buffer,
    /// Number of queries used in this set (2 per pass: begin + end).
    queries_used: u32,
    /// Maximum queries this set supports.
    max_queries: u32,
}

impl QuerySetEntry {
    /// Create a new QuerySet entry with associated buffers.
    fn new(device: &wgpu::Device, max_passes: usize) -> Self {
        // 2 timestamps per pass (begin + end)
        let max_queries = (max_passes * 2) as u32;

        let query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("GPU Profiler QuerySet"),
            ty: wgpu::QueryType::Timestamp,
            count: max_queries,
        });

        // Each timestamp is a u64 (8 bytes)
        let buffer_size = (max_queries as u64) * 8;

        let resolve_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GPU Profiler Resolve Buffer"),
            size: buffer_size,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GPU Profiler Staging Buffer"),
            size: buffer_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        Self {
            query_set,
            resolve_buffer,
            staging_buffer,
            queries_used: 0,
            max_queries,
        }
    }

    /// Reset for reuse in a new frame.
    fn reset(&mut self) {
        self.queries_used = 0;
    }

    /// Check if there's room for another pass (2 queries).
    fn has_capacity(&self) -> bool {
        self.queries_used + 2 <= self.max_queries
    }

    /// Get the next query index for a begin timestamp.
    fn next_begin_index(&self) -> u32 {
        self.queries_used
    }

    /// Get the next query index for an end timestamp.
    fn next_end_index(&self) -> u32 {
        self.queries_used + 1
    }

    /// Mark 2 queries as used (one pass recorded).
    fn mark_pass_recorded(&mut self) {
        self.queries_used += 2;
    }
}

// ---------------------------------------------------------------------------
// FrameData
// ---------------------------------------------------------------------------

/// Per-frame recording state.
struct FrameData {
    /// Frame identifier.
    frame_id: u64,
    /// Pass names in recording order.
    pass_names: Vec<String>,
    /// CPU timing start instants for each pass.
    cpu_starts: Vec<Instant>,
    /// CPU timing end instants for each pass.
    cpu_ends: Vec<Instant>,
    /// Whether this frame was resolved and is ready for readback.
    resolved: bool,
    /// QuerySet index in the pool used for this frame.
    query_set_index: usize,
    /// Number of passes recorded.
    pass_count: usize,
}

impl FrameData {
    fn new(frame_id: u64, query_set_index: usize) -> Self {
        Self {
            frame_id,
            pass_names: Vec::new(),
            cpu_starts: Vec::new(),
            cpu_ends: Vec::new(),
            resolved: false,
            query_set_index,
            pass_count: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// GPUProfiler
// ---------------------------------------------------------------------------

/// GPU profiler managing timestamp queries with deferred readback.
///
/// Handles the GPU→CPU latency by maintaining a ring buffer of frames
/// and reading results after a configurable delay (typically 2-3 frames).
pub struct GPUProfiler {
    /// Pool of QuerySets for recycling across frames.
    query_pool: Vec<QuerySetEntry>,
    /// Ring buffer of in-flight frame data.
    frame_ring: VecDeque<FrameData>,
    /// Maximum frames to buffer (ring size).
    ring_size: usize,
    /// Timestamp period in nanoseconds (ticks → ns conversion).
    timestamp_period: f32,
    /// Whether timestamps are supported by the device.
    timestamps_supported: bool,
    /// Current frame ID (monotonically increasing).
    frame_counter: AtomicU64,
    /// Index of current frame in the pool.
    current_pool_index: usize,
    /// Whether we're currently in a pass (between begin/end).
    in_pass: bool,
    /// CPU instant when current pass began.
    current_pass_start: Option<Instant>,
    /// Maximum passes per frame.
    max_passes: usize,
}

impl GPUProfiler {
    /// Create a new GPU profiler.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device to create query resources on.
    /// * `queue` - The wgpu queue (used to get timestamp period).
    /// * `max_passes` - Maximum number of passes to profile per frame.
    ///
    /// # Note
    ///
    /// If the device doesn't support timestamp queries, the profiler
    /// operates in no-op mode and all timing methods return zero/empty.
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, max_passes: usize) -> Self {
        Self::with_ring_size(device, queue, max_passes, DEFAULT_RING_SIZE)
    }

    /// Create a profiler with a custom ring buffer size.
    ///
    /// Larger ring sizes handle higher GPU-CPU latency but use more memory.
    pub fn with_ring_size(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        max_passes: usize,
        ring_size: usize,
    ) -> Self {
        let max_passes = max_passes.min(MAX_PASSES_PER_FRAME);
        let ring_size = ring_size.max(2); // Minimum 2 for double-buffering

        // Check timestamp support via device features
        let timestamps_supported = device.features().contains(wgpu::Features::TIMESTAMP_QUERY);

        // Get timestamp period from queue (wgpu 22+)
        let timestamp_period = if timestamps_supported {
            queue.get_timestamp_period()
        } else {
            1.0 // Dummy value, won't be used
        };

        // Create query pool (one entry per ring slot)
        let query_pool = if timestamps_supported {
            (0..ring_size)
                .map(|_| QuerySetEntry::new(device, max_passes))
                .collect()
        } else {
            Vec::new()
        };

        Self {
            query_pool,
            frame_ring: VecDeque::with_capacity(ring_size),
            ring_size,
            timestamp_period,
            timestamps_supported,
            frame_counter: AtomicU64::new(0),
            current_pool_index: 0,
            in_pass: false,
            current_pass_start: None,
            max_passes,
        }
    }

    /// Check if GPU timestamps are supported.
    pub fn is_supported(&self) -> bool {
        self.timestamps_supported
    }

    /// Get the timestamp period in nanoseconds.
    pub fn timestamp_period(&self) -> f32 {
        self.timestamp_period
    }

    /// Get the current frame ID.
    pub fn current_frame_id(&self) -> u64 {
        self.frame_counter.load(Ordering::Relaxed)
    }

    /// Begin a new profiled frame.
    ///
    /// Must be called once at the start of each frame before any begin_pass calls.
    pub fn begin_frame(&mut self) {
        if !self.timestamps_supported {
            self.frame_counter.fetch_add(1, Ordering::Relaxed);
            return;
        }

        // If ring is full, we'll overwrite the oldest (not yet read) frame
        if self.frame_ring.len() >= self.ring_size {
            // Pop the oldest frame, its data will be lost
            self.frame_ring.pop_front();
        }

        let frame_id = self.frame_counter.fetch_add(1, Ordering::Relaxed);

        // Reset the current query set
        self.query_pool[self.current_pool_index].reset();

        // Create new frame data
        let frame_data = FrameData::new(frame_id, self.current_pool_index);
        self.frame_ring.push_back(frame_data);
    }

    /// Begin profiling a render pass.
    ///
    /// Must be paired with a matching `end_pass` call.
    ///
    /// # Arguments
    ///
    /// * `encoder` - The command encoder to write timestamps to.
    /// * `name` - Human-readable name for this pass.
    pub fn begin_pass(&mut self, encoder: &mut wgpu::CommandEncoder, name: &str) {
        if !self.timestamps_supported {
            return;
        }

        if self.in_pass {
            // Nested passes not supported; end the previous one implicitly
            self.end_pass_internal(encoder, Instant::now());
        }

        let frame_data = match self.frame_ring.back_mut() {
            Some(f) => f,
            None => return, // No frame started
        };

        let query_entry = &mut self.query_pool[frame_data.query_set_index];

        if !query_entry.has_capacity() {
            return; // Max passes reached
        }

        let begin_index = query_entry.next_begin_index();
        encoder.write_timestamp(&query_entry.query_set, begin_index);

        frame_data.pass_names.push(name.to_string());
        frame_data.cpu_starts.push(Instant::now());

        self.in_pass = true;
        self.current_pass_start = Some(Instant::now());
    }

    /// End profiling the current render pass.
    pub fn end_pass(&mut self, encoder: &mut wgpu::CommandEncoder) {
        let now = Instant::now();
        self.end_pass_internal(encoder, now);
    }

    fn end_pass_internal(&mut self, encoder: &mut wgpu::CommandEncoder, end_time: Instant) {
        if !self.timestamps_supported || !self.in_pass {
            return;
        }

        let frame_data = match self.frame_ring.back_mut() {
            Some(f) => f,
            None => return,
        };

        let query_entry = &mut self.query_pool[frame_data.query_set_index];

        let end_index = query_entry.next_end_index();
        encoder.write_timestamp(&query_entry.query_set, end_index);

        query_entry.mark_pass_recorded();
        frame_data.cpu_ends.push(end_time);
        frame_data.pass_count += 1;

        self.in_pass = false;
        self.current_pass_start = None;
    }

    /// Resolve timestamp queries for the current frame.
    ///
    /// Call this after all passes are recorded, before submitting the
    /// command buffer.
    pub fn resolve(&mut self, encoder: &mut wgpu::CommandEncoder) {
        if !self.timestamps_supported {
            return;
        }

        let frame_data = match self.frame_ring.back_mut() {
            Some(f) => f,
            None => return,
        };

        if frame_data.pass_count == 0 {
            return; // Nothing to resolve
        }

        let query_entry = &self.query_pool[frame_data.query_set_index];
        let query_count = query_entry.queries_used;

        if query_count == 0 {
            return;
        }

        // Resolve queries to the resolve buffer
        encoder.resolve_query_set(
            &query_entry.query_set,
            0..query_count,
            &query_entry.resolve_buffer,
            0,
        );

        // Copy to staging buffer for CPU readback
        encoder.copy_buffer_to_buffer(
            &query_entry.resolve_buffer,
            0,
            &query_entry.staging_buffer,
            0,
            (query_count as u64) * 8,
        );

        frame_data.resolved = true;

        // Advance pool index for next frame
        self.current_pool_index = (self.current_pool_index + 1) % self.ring_size;
    }

    /// Attempt to read timing results from completed frames.
    ///
    /// Returns `Some(FrameProfile)` if a frame's results are ready,
    /// `None` if no results are available yet.
    ///
    /// This method handles the GPU→CPU latency by reading from the oldest
    /// resolved frame in the ring buffer.
    pub fn read_results(&mut self, queue: &wgpu::Queue) -> Option<FrameProfile> {
        if !self.timestamps_supported {
            return None;
        }

        // Look for the oldest resolved frame
        let front_idx = self.frame_ring.iter().position(|f| f.resolved)?;

        // Need at least 2 frames of latency before reading
        if front_idx == self.frame_ring.len() - 1 {
            return None; // This is the current frame, wait for more latency
        }

        // Remove and process the resolved frame
        let frame_data = self.frame_ring.remove(front_idx)?;

        if frame_data.pass_count == 0 {
            return Some(FrameProfile::new(frame_data.frame_id));
        }

        let query_entry = &self.query_pool[frame_data.query_set_index];

        // Map the staging buffer and read results
        let slice = query_entry.staging_buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();

        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });

        // Poll the device until mapping is complete
        queue.submit([]);
        let _ = rx.recv();

        let data = slice.get_mapped_range();
        let timestamps: &[u64] = bytemuck::cast_slice(&data);

        let mut profile = FrameProfile::new(frame_data.frame_id);

        for (i, name) in frame_data.pass_names.iter().enumerate() {
            let begin_idx = i * 2;
            let end_idx = begin_idx + 1;

            if end_idx >= timestamps.len() {
                break;
            }

            let begin_ts = timestamps[begin_idx];
            let end_ts = timestamps[end_idx];

            // Convert ticks to milliseconds
            let delta_ticks = end_ts.saturating_sub(begin_ts);
            let delta_ns = (delta_ticks as f64) * (self.timestamp_period as f64);
            let gpu_ms = (delta_ns / 1_000_000.0) as f32;

            // Calculate CPU time
            let cpu_ms = if i < frame_data.cpu_starts.len() && i < frame_data.cpu_ends.len() {
                frame_data.cpu_ends[i]
                    .duration_since(frame_data.cpu_starts[i])
                    .as_secs_f32()
                    * 1000.0
            } else {
                0.0
            };

            profile.add_pass(PassTiming::new(name.clone(), gpu_ms, cpu_ms));
        }

        drop(data);
        query_entry.staging_buffer.unmap();

        Some(profile)
    }

    /// Drain all available results.
    ///
    /// Returns all frames that have completed readback, in order.
    pub fn drain_results(&mut self, queue: &wgpu::Queue) -> Vec<FrameProfile> {
        let mut results = Vec::new();
        while let Some(profile) = self.read_results(queue) {
            results.push(profile);
        }
        results
    }

    /// Get profiling statistics.
    pub fn stats(&self) -> ProfilerStats {
        ProfilerStats {
            frames_in_flight: self.frame_ring.len(),
            ring_capacity: self.ring_size,
            max_passes: self.max_passes,
            timestamps_supported: self.timestamps_supported,
            timestamp_period_ns: self.timestamp_period,
        }
    }

    /// Create timestamp writes descriptor for a render pass.
    ///
    /// This is a convenience method for integrating with wgpu render pass
    /// descriptors that support `timestamp_writes` field.
    pub fn timestamp_writes_for_pass(
        &mut self,
        pass_name: &str,
    ) -> Option<PendingTimestampWrites> {
        if !self.timestamps_supported {
            return None;
        }

        let frame_data = self.frame_ring.back_mut()?;
        let query_entry = &mut self.query_pool[frame_data.query_set_index];

        if !query_entry.has_capacity() {
            return None;
        }

        let begin_index = query_entry.next_begin_index();
        let end_index = query_entry.next_end_index();

        // Record pass info
        frame_data.pass_names.push(pass_name.to_string());
        frame_data.cpu_starts.push(Instant::now());
        frame_data.cpu_ends.push(Instant::now()); // Will be updated
        frame_data.pass_count += 1;
        query_entry.mark_pass_recorded();

        Some(PendingTimestampWrites {
            query_set_index: frame_data.query_set_index,
            begin_query_index: begin_index,
            end_query_index: end_index,
        })
    }

    /// Get a reference to a QuerySet for timestamp writes.
    pub fn query_set(&self, index: usize) -> Option<&wgpu::QuerySet> {
        self.query_pool.get(index).map(|e| &e.query_set)
    }
}

// ---------------------------------------------------------------------------
// PendingTimestampWrites
// ---------------------------------------------------------------------------

/// Timestamp write indices for a pending pass.
///
/// Use with `RenderPassDescriptor::timestamp_writes` for integrated
/// GPU timestamp recording.
#[derive(Debug, Clone, Copy)]
pub struct PendingTimestampWrites {
    /// Index into the profiler's query pool.
    pub query_set_index: usize,
    /// Query index for the begin timestamp.
    pub begin_query_index: u32,
    /// Query index for the end timestamp.
    pub end_query_index: u32,
}

// ---------------------------------------------------------------------------
// ProfilerStats
// ---------------------------------------------------------------------------

/// Runtime statistics from the profiler.
#[derive(Debug, Clone)]
pub struct ProfilerStats {
    /// Number of frames currently in the ring buffer.
    pub frames_in_flight: usize,
    /// Maximum frames the ring can hold.
    pub ring_capacity: usize,
    /// Maximum passes tracked per frame.
    pub max_passes: usize,
    /// Whether timestamps are supported.
    pub timestamps_supported: bool,
    /// Timestamp period in nanoseconds.
    pub timestamp_period_ns: f32,
}

// ---------------------------------------------------------------------------
// Utility functions
// ---------------------------------------------------------------------------

/// Convert GPU timestamp ticks to milliseconds.
///
/// # Arguments
///
/// * `ticks` - Number of GPU timestamp ticks.
/// * `period_ns` - Timestamp period in nanoseconds (from device limits).
pub fn ticks_to_ms(ticks: u64, period_ns: f32) -> f32 {
    let ns = (ticks as f64) * (period_ns as f64);
    (ns / 1_000_000.0) as f32
}

/// Convert milliseconds to GPU timestamp ticks.
pub fn ms_to_ticks(ms: f32, period_ns: f32) -> u64 {
    let ns = (ms as f64) * 1_000_000.0;
    (ns / (period_ns as f64)) as u64
}

// ---------------------------------------------------------------------------
// NoOpProfiler
// ---------------------------------------------------------------------------

/// A no-op profiler for when GPU timestamps are not needed.
///
/// All methods are no-ops that return immediately. Useful for:
/// - Release builds where profiling is disabled
/// - Platforms that don't support timestamps
/// - Testing without GPU access
#[derive(Debug, Default)]
pub struct NoOpProfiler {
    frame_id: AtomicU64,
}

impl NoOpProfiler {
    /// Create a new no-op profiler.
    pub fn new() -> Self {
        Self::default()
    }

    /// Simulated begin_frame (just increments counter).
    pub fn begin_frame(&self) {
        self.frame_id.fetch_add(1, Ordering::Relaxed);
    }

    /// No-op begin_pass.
    #[inline]
    pub fn begin_pass(&self, _encoder: &mut wgpu::CommandEncoder, _name: &str) {}

    /// No-op end_pass.
    #[inline]
    pub fn end_pass(&self, _encoder: &mut wgpu::CommandEncoder) {}

    /// No-op resolve.
    #[inline]
    pub fn resolve(&self, _encoder: &mut wgpu::CommandEncoder) {}

    /// Always returns None.
    #[inline]
    pub fn read_results(&self, _queue: &wgpu::Queue) -> Option<FrameProfile> {
        None
    }

    /// Current frame ID.
    pub fn current_frame_id(&self) -> u64 {
        self.frame_id.load(Ordering::Relaxed)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ===== SECTION 1: PassTiming tests =====

    #[test]
    fn pass_timing_new_sets_all_fields() {
        let timing = PassTiming::new("GBuffer", 1.5, 0.3);
        assert_eq!(timing.name, "GBuffer");
        assert_eq!(timing.gpu_ms, 1.5);
        assert_eq!(timing.cpu_ms, 0.3);
    }

    #[test]
    fn pass_timing_accepts_string_slice() {
        let timing = PassTiming::new("test", 0.0, 0.0);
        assert_eq!(timing.name, "test");
    }

    #[test]
    fn pass_timing_accepts_owned_string() {
        let timing = PassTiming::new(String::from("owned"), 0.0, 0.0);
        assert_eq!(timing.name, "owned");
    }

    #[test]
    fn pass_timing_clone_is_equal() {
        let timing = PassTiming::new("Clone", 2.5, 1.0);
        let cloned = timing.clone();
        assert_eq!(timing, cloned);
    }

    #[test]
    fn pass_timing_debug_format() {
        let timing = PassTiming::new("Debug", 1.0, 0.5);
        let debug = format!("{:?}", timing);
        assert!(debug.contains("PassTiming"));
        assert!(debug.contains("Debug"));
    }

    // ===== SECTION 2: FrameProfile tests =====

    #[test]
    fn frame_profile_new_creates_empty() {
        let profile = FrameProfile::new(42);
        assert_eq!(profile.frame_id, 42);
        assert!(profile.pass_timings.is_empty());
        assert_eq!(profile.total_gpu_ms, 0.0);
    }

    #[test]
    fn frame_profile_add_pass_updates_total() {
        let mut profile = FrameProfile::new(1);
        profile.add_pass(PassTiming::new("Pass1", 2.0, 0.5));
        assert_eq!(profile.total_gpu_ms, 2.0);
        assert_eq!(profile.pass_count(), 1);
    }

    #[test]
    fn frame_profile_multiple_passes_accumulate() {
        let mut profile = FrameProfile::new(1);
        profile.add_pass(PassTiming::new("Pass1", 1.0, 0.1));
        profile.add_pass(PassTiming::new("Pass2", 2.0, 0.2));
        profile.add_pass(PassTiming::new("Pass3", 3.0, 0.3));

        assert_eq!(profile.pass_count(), 3);
        assert_eq!(profile.total_gpu_ms, 6.0);
    }

    #[test]
    fn frame_profile_preserves_pass_order() {
        let mut profile = FrameProfile::new(1);
        profile.add_pass(PassTiming::new("First", 1.0, 0.0));
        profile.add_pass(PassTiming::new("Second", 2.0, 0.0));

        assert_eq!(profile.pass_timings[0].name, "First");
        assert_eq!(profile.pass_timings[1].name, "Second");
    }

    #[test]
    fn frame_profile_zero_gpu_time_valid() {
        let mut profile = FrameProfile::new(0);
        profile.add_pass(PassTiming::new("Zero", 0.0, 0.0));
        assert_eq!(profile.total_gpu_ms, 0.0);
        assert_eq!(profile.pass_count(), 1);
    }

    #[test]
    fn frame_profile_clone_is_equal() {
        let mut profile = FrameProfile::new(99);
        profile.add_pass(PassTiming::new("Test", 1.5, 0.5));
        let cloned = profile.clone();
        assert_eq!(profile, cloned);
    }

    #[test]
    fn frame_profile_debug_format_shows_fields() {
        let profile = FrameProfile::new(123);
        let debug = format!("{:?}", profile);
        assert!(debug.contains("FrameProfile"));
        assert!(debug.contains("123"));
    }

    // ===== SECTION 3: Tick conversion tests =====

    #[test]
    fn ticks_to_ms_basic_conversion() {
        // period = 1ns, 1_000_000 ticks = 1ms
        let ms = ticks_to_ms(1_000_000, 1.0);
        assert!((ms - 1.0).abs() < 0.001);
    }

    #[test]
    fn ticks_to_ms_with_period_scaling() {
        // period = 10ns, 100_000 ticks = 1ms
        let ms = ticks_to_ms(100_000, 10.0);
        assert!((ms - 1.0).abs() < 0.001);
    }

    #[test]
    fn ticks_to_ms_zero_ticks() {
        let ms = ticks_to_ms(0, 1.0);
        assert_eq!(ms, 0.0);
    }

    #[test]
    fn ticks_to_ms_large_values() {
        // 1 billion ticks at 1ns period = 1000ms
        let ms = ticks_to_ms(1_000_000_000, 1.0);
        assert!((ms - 1000.0).abs() < 0.001);
    }

    #[test]
    fn ms_to_ticks_basic_conversion() {
        // period = 1ns, 1ms = 1_000_000 ticks
        let ticks = ms_to_ticks(1.0, 1.0);
        assert_eq!(ticks, 1_000_000);
    }

    #[test]
    fn ms_to_ticks_with_period_scaling() {
        // period = 10ns, 1ms = 100_000 ticks
        let ticks = ms_to_ticks(1.0, 10.0);
        assert_eq!(ticks, 100_000);
    }

    #[test]
    fn ms_to_ticks_roundtrip() {
        let original_ticks: u64 = 123456;
        let period: f32 = 25.0;
        let ms = ticks_to_ms(original_ticks, period);
        let recovered = ms_to_ticks(ms, period);
        assert!((recovered as i64 - original_ticks as i64).abs() <= 1);
    }

    #[test]
    fn ticks_to_ms_fractional_period() {
        // period = 0.5ns (common on high-freq GPUs)
        let ms = ticks_to_ms(2_000_000, 0.5);
        assert!((ms - 1.0).abs() < 0.001);
    }

    // ===== SECTION 4: ProfilerStats tests =====

    #[test]
    fn profiler_stats_fields_accessible() {
        let stats = ProfilerStats {
            frames_in_flight: 3,
            ring_capacity: 4,
            max_passes: 16,
            timestamps_supported: true,
            timestamp_period_ns: 1.0,
        };

        assert_eq!(stats.frames_in_flight, 3);
        assert_eq!(stats.ring_capacity, 4);
        assert_eq!(stats.max_passes, 16);
        assert!(stats.timestamps_supported);
        assert_eq!(stats.timestamp_period_ns, 1.0);
    }

    #[test]
    fn profiler_stats_clone() {
        let stats = ProfilerStats {
            frames_in_flight: 2,
            ring_capacity: 4,
            max_passes: 8,
            timestamps_supported: false,
            timestamp_period_ns: 10.0,
        };
        let cloned = stats.clone();

        assert_eq!(stats.frames_in_flight, cloned.frames_in_flight);
        assert_eq!(stats.ring_capacity, cloned.ring_capacity);
    }

    #[test]
    fn profiler_stats_debug_format() {
        let stats = ProfilerStats {
            frames_in_flight: 1,
            ring_capacity: 4,
            max_passes: 8,
            timestamps_supported: true,
            timestamp_period_ns: 1.5,
        };
        let debug = format!("{:?}", stats);
        assert!(debug.contains("ProfilerStats"));
    }

    // ===== SECTION 5: PendingTimestampWrites tests =====

    #[test]
    fn pending_timestamp_writes_fields() {
        let pending = PendingTimestampWrites {
            query_set_index: 0,
            begin_query_index: 0,
            end_query_index: 1,
        };

        assert_eq!(pending.query_set_index, 0);
        assert_eq!(pending.begin_query_index, 0);
        assert_eq!(pending.end_query_index, 1);
    }

    #[test]
    fn pending_timestamp_writes_copy() {
        let pending = PendingTimestampWrites {
            query_set_index: 1,
            begin_query_index: 2,
            end_query_index: 3,
        };
        let copied = pending;
        assert_eq!(pending.query_set_index, copied.query_set_index);
    }

    #[test]
    fn pending_timestamp_writes_debug() {
        let pending = PendingTimestampWrites {
            query_set_index: 2,
            begin_query_index: 4,
            end_query_index: 5,
        };
        let debug = format!("{:?}", pending);
        assert!(debug.contains("PendingTimestampWrites"));
    }

    // ===== SECTION 6: NoOpProfiler tests =====

    #[test]
    fn no_op_profiler_default() {
        let profiler = NoOpProfiler::default();
        assert_eq!(profiler.current_frame_id(), 0);
    }

    #[test]
    fn no_op_profiler_new() {
        let profiler = NoOpProfiler::new();
        assert_eq!(profiler.current_frame_id(), 0);
    }

    #[test]
    fn no_op_profiler_begin_frame_increments() {
        let profiler = NoOpProfiler::new();
        profiler.begin_frame();
        assert_eq!(profiler.current_frame_id(), 1);
        profiler.begin_frame();
        assert_eq!(profiler.current_frame_id(), 2);
    }

    #[test]
    fn no_op_profiler_debug_format() {
        let profiler = NoOpProfiler::new();
        let debug = format!("{:?}", profiler);
        assert!(debug.contains("NoOpProfiler"));
    }

    // ===== SECTION 7: Constants tests =====

    #[test]
    fn default_ring_size_is_reasonable() {
        assert!(DEFAULT_RING_SIZE >= 2);
        assert!(DEFAULT_RING_SIZE <= 16);
    }

    #[test]
    fn max_passes_per_frame_is_reasonable() {
        assert!(MAX_PASSES_PER_FRAME >= 8);
        assert!(MAX_PASSES_PER_FRAME <= 256);
    }

    // ===== SECTION 8: Edge case tests =====

    #[test]
    fn ticks_to_ms_max_u64() {
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
    fn frame_profile_max_frame_id() {
        let profile = FrameProfile::new(u64::MAX);
        assert_eq!(profile.frame_id, u64::MAX);
    }

    #[test]
    fn pass_timing_empty_name() {
        let timing = PassTiming::new("", 1.0, 0.5);
        assert!(timing.name.is_empty());
    }

    #[test]
    fn pass_timing_unicode_name() {
        let timing = PassTiming::new("GBuffer [日本語]", 1.0, 0.5);
        assert!(timing.name.contains("日本語"));
    }

    #[test]
    fn pass_timing_negative_times_stored() {
        // While unusual, the struct doesn't enforce non-negative
        let timing = PassTiming::new("Negative", -1.0, -0.5);
        assert_eq!(timing.gpu_ms, -1.0);
        assert_eq!(timing.cpu_ms, -0.5);
    }

    #[test]
    fn frame_profile_high_precision_timing() {
        let mut profile = FrameProfile::new(0);
        profile.add_pass(PassTiming::new("Precise", 0.001234, 0.000567));
        assert!((profile.total_gpu_ms - 0.001234).abs() < f32::EPSILON);
    }

    // ===== SECTION 9: Send + Sync bounds =====

    fn assert_send<T: Send>() {}
    fn assert_sync<T: Sync>() {}

    #[test]
    fn pass_timing_is_send() {
        assert_send::<PassTiming>();
    }

    #[test]
    fn pass_timing_is_sync() {
        assert_sync::<PassTiming>();
    }

    #[test]
    fn frame_profile_is_send() {
        assert_send::<FrameProfile>();
    }

    #[test]
    fn frame_profile_is_sync() {
        assert_sync::<FrameProfile>();
    }

    #[test]
    fn profiler_stats_is_send() {
        assert_send::<ProfilerStats>();
    }

    #[test]
    fn profiler_stats_is_sync() {
        assert_sync::<ProfilerStats>();
    }

    #[test]
    fn no_op_profiler_is_send() {
        assert_send::<NoOpProfiler>();
    }

    #[test]
    fn no_op_profiler_is_sync() {
        assert_sync::<NoOpProfiler>();
    }
}
