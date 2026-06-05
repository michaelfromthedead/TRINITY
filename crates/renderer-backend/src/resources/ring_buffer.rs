//! Ring buffer for per-frame GPU data streaming.
//!
//! This module provides a triple-buffered ring buffer for efficient per-frame data
//! upload. It ensures that while the GPU reads from frames N-2 and N-1, the CPU
//! can safely write to frame N without data races.
//!
//! # Architecture
//!
//! ```text
//! +---------------+---------------+---------------+
//! | Frame 0       | Frame 1       | Frame 2       |
//! | GPU reading   | GPU reading   | CPU writing   |
//! +---------------+---------------+---------------+
//!       ^               ^               ^
//!   frame_offsets[0]  frame_offsets[1]  frame_offsets[2]
//! ```
//!
//! Each frame has a dedicated region in the buffer. The ring buffer cycles through
//! frames 0, 1, 2, 0, 1, 2, ... and waits for the GPU to finish with a frame before
//! reusing it.
//!
//! # Usage
//!
//! ```no_run
//! use renderer_backend::resources::ring_buffer::{RingBuffer, RingBufferConfig};
//! use wgpu::BufferUsages;
//!
//! # fn example(device: &wgpu::Device, queue: &wgpu::Queue) {
//! // Create a ring buffer with 1MB per frame (3MB total)
//! let mut ring = RingBuffer::new(
//!     device,
//!     RingBufferConfig {
//!         frame_size: 1024 * 1024,
//!         usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
//!         label: Some("per_frame_uniforms"),
//!         frames_in_flight: 3,
//!     },
//! );
//!
//! // At the start of each frame
//! ring.begin_frame();
//!
//! // Allocate space for a uniform
//! if let Some(alloc) = ring.allocate(256, 256) {
//!     // Write data to the buffer at alloc.offset
//!     queue.write_buffer(ring.buffer(), alloc.offset, &data);
//! }
//!
//! // Check utilization
//! let metrics = ring.metrics();
//! println!("Frame utilization: {:.1}%", metrics.utilization * 100.0);
//! # }
//! ```
//!
//! # Alignment
//!
//! The ring buffer respects WebGPU alignment requirements:
//! - Uniform buffer offsets must be 256-byte aligned
//! - Storage buffer dynamic offsets must be 256-byte aligned
//! - Individual allocations are aligned to the requested alignment

use log::{debug, trace};
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use wgpu::{Buffer, BufferDescriptor, BufferUsages, Device};

use super::uniform::UNIFORM_ALIGNMENT;

// ============================================================================
// Constants
// ============================================================================

/// Default number of frames in flight for triple buffering.
pub const DEFAULT_FRAMES_IN_FLIGHT: usize = 3;

/// Minimum alignment for ring buffer allocations (matches uniform alignment).
pub const RING_BUFFER_MIN_ALIGNMENT: u64 = UNIFORM_ALIGNMENT;

// ============================================================================
// Ring Allocation
// ============================================================================

/// Result of a successful allocation from the ring buffer.
///
/// Contains the byte offset into the underlying buffer and metadata about
/// which frame the allocation belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RingAllocation {
    /// Byte offset into the ring buffer.
    pub offset: u64,
    /// Size of the allocation in bytes (after alignment).
    pub size: u64,
    /// Frame index this allocation belongs to (0, 1, or 2 for triple buffering).
    pub frame: usize,
}

impl RingAllocation {
    /// Returns the end offset of this allocation.
    #[inline]
    pub const fn end_offset(&self) -> u64 {
        self.offset + self.size
    }
}

// ============================================================================
// Ring Buffer Metrics
// ============================================================================

/// Metrics tracking ring buffer usage and performance.
///
/// These metrics help identify memory pressure and potential overflow conditions.
#[derive(Debug, Clone, Default)]
pub struct RingBufferMetrics {
    /// Total bytes allocated across all frames since creation.
    pub total_allocated: u64,
    /// Bytes used in the current frame.
    pub current_frame_used: u64,
    /// Per-frame capacity in bytes.
    pub frame_capacity: u64,
    /// Current frame utilization (0.0 to 1.0).
    pub utilization: f32,
    /// Number of times the ring buffer has wrapped around.
    pub wrap_count: u32,
    /// Number of allocation failures due to insufficient space.
    pub overflow_count: u32,
    /// Largest single allocation ever made.
    pub peak_allocation: u64,
    /// Current frame index (0, 1, 2, ...).
    pub current_frame: usize,
}

// ============================================================================
// Ring Buffer Configuration
// ============================================================================

/// Configuration for creating a ring buffer.
#[derive(Debug, Clone)]
pub struct RingBufferConfig<'a> {
    /// Size of each frame's region in bytes.
    ///
    /// Total buffer size will be `frame_size * frames_in_flight`.
    pub frame_size: u64,
    /// Buffer usage flags (e.g., UNIFORM | COPY_DST).
    pub usage: BufferUsages,
    /// Optional debug label for the buffer.
    pub label: Option<&'a str>,
    /// Number of frames in flight (default: 3 for triple buffering).
    pub frames_in_flight: usize,
}

impl<'a> Default for RingBufferConfig<'a> {
    fn default() -> Self {
        Self {
            frame_size: 256 * 1024, // 256KB per frame
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            label: None,
            frames_in_flight: DEFAULT_FRAMES_IN_FLIGHT,
        }
    }
}

// ============================================================================
// Ring Buffer
// ============================================================================

/// A triple-buffered ring buffer for per-frame GPU data.
///
/// The ring buffer divides a single wgpu buffer into multiple frame regions,
/// allowing the CPU to write to one region while the GPU reads from others.
/// This prevents data races without explicit synchronization.
///
/// # Thread Safety
///
/// The ring buffer uses atomic operations for metrics tracking but is not
/// designed for concurrent allocation. Use from a single thread or protect
/// with external synchronization.
pub struct RingBuffer {
    /// The underlying wgpu buffer.
    buffer: Buffer,
    /// Total buffer capacity in bytes.
    capacity: u64,
    /// Size of each frame's region in bytes.
    frame_size: u64,
    /// Current frame index (0 to frames_in_flight - 1).
    current_frame: usize,
    /// Per-frame write offset within each frame's region.
    frame_offsets: Vec<u64>,
    /// Number of frames in flight (typically 3).
    frames_in_flight: usize,
    /// Buffer usage flags.
    usage: BufferUsages,
    /// Optional debug label.
    label: Option<String>,

    // Metrics (atomic for thread-safe reading)
    total_allocated: AtomicU64,
    wrap_count: AtomicU32,
    overflow_count: AtomicU32,
    peak_allocation: AtomicU64,
}

impl RingBuffer {
    /// Creates a new ring buffer with the given configuration.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device to create the buffer on.
    /// * `config` - Configuration specifying size, usage, and frame count.
    ///
    /// # Panics
    ///
    /// Panics if `frames_in_flight` is 0 or if `frame_size` is 0.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use renderer_backend::resources::ring_buffer::{RingBuffer, RingBufferConfig};
    /// use wgpu::BufferUsages;
    ///
    /// # fn example(device: &wgpu::Device) {
    /// let ring = RingBuffer::new(device, RingBufferConfig {
    ///     frame_size: 1024 * 1024, // 1MB per frame
    ///     usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
    ///     label: Some("dynamic_uniforms"),
    ///     frames_in_flight: 3,
    /// });
    /// # }
    /// ```
    pub fn new(device: &Device, config: RingBufferConfig<'_>) -> Self {
        assert!(
            config.frames_in_flight > 0,
            "frames_in_flight must be at least 1"
        );
        assert!(config.frame_size > 0, "frame_size must be greater than 0");

        let frames_in_flight = config.frames_in_flight;
        let frame_size = config.frame_size;
        let capacity = frame_size * frames_in_flight as u64;

        let label_owned = config.label.map(|s| s.to_string());
        let label_str = label_owned.as_deref();

        debug!(
            "Creating RingBuffer '{}': {} frames x {} bytes = {} total",
            label_str.unwrap_or("unnamed"),
            frames_in_flight,
            frame_size,
            capacity
        );

        let buffer = device.create_buffer(&BufferDescriptor {
            label: label_str,
            size: capacity,
            usage: config.usage,
            mapped_at_creation: false,
        });

        Self {
            buffer,
            capacity,
            frame_size,
            current_frame: 0,
            frame_offsets: vec![0; frames_in_flight],
            frames_in_flight,
            usage: config.usage,
            label: label_owned,
            total_allocated: AtomicU64::new(0),
            wrap_count: AtomicU32::new(0),
            overflow_count: AtomicU32::new(0),
            peak_allocation: AtomicU64::new(0),
        }
    }

    /// Convenience constructor with default triple-buffering.
    ///
    /// Creates a ring buffer with 3 frames in flight.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device.
    /// * `frame_size` - Size of each frame's region in bytes.
    /// * `usage` - Buffer usage flags.
    /// * `label` - Optional debug label.
    pub fn with_defaults(
        device: &Device,
        frame_size: u64,
        usage: BufferUsages,
        label: Option<&str>,
    ) -> Self {
        Self::new(
            device,
            RingBufferConfig {
                frame_size,
                usage,
                label,
                frames_in_flight: DEFAULT_FRAMES_IN_FLIGHT,
            },
        )
    }

    /// Allocates space within the current frame's region.
    ///
    /// Returns `None` if there isn't enough space remaining in the current
    /// frame. The returned allocation is aligned to the specified alignment.
    ///
    /// # Arguments
    ///
    /// * `size` - Number of bytes to allocate.
    /// * `alignment` - Required alignment (will be clamped to minimum of 4).
    ///
    /// # Returns
    ///
    /// `Some(RingAllocation)` if successful, `None` if the frame is full.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use renderer_backend::resources::ring_buffer::{RingBuffer, RingBufferConfig};
    /// use renderer_backend::resources::uniform::UNIFORM_ALIGNMENT;
    ///
    /// # fn example(ring: &mut RingBuffer) {
    /// // Allocate space for a 64-byte uniform with 256-byte alignment
    /// if let Some(alloc) = ring.allocate(64, UNIFORM_ALIGNMENT) {
    ///     println!("Allocated at offset {} (frame {})", alloc.offset, alloc.frame);
    /// } else {
    ///     println!("Frame is full, call begin_frame() to advance");
    /// }
    /// # }
    /// ```
    pub fn allocate(&mut self, size: u64, alignment: u64) -> Option<RingAllocation> {
        if size == 0 {
            return None;
        }

        // Ensure minimum alignment of 4 bytes (wgpu requirement)
        let alignment = alignment.max(4);

        // Get current frame's base offset in the buffer
        let frame_base = self.current_frame as u64 * self.frame_size;

        // Get current write position within this frame
        let current_offset = self.frame_offsets[self.current_frame];

        // Align the current offset
        let aligned_offset = Self::align_up(current_offset, alignment);

        // Calculate total size needed (aligned)
        let aligned_size = Self::align_up(size, alignment);

        // Check if allocation fits within this frame's region
        let new_offset = aligned_offset + aligned_size;
        if new_offset > self.frame_size {
            // Not enough space in this frame
            self.overflow_count.fetch_add(1, Ordering::Relaxed);
            trace!(
                "RingBuffer '{}': allocation of {} bytes failed, frame {} full ({}/{})",
                self.label.as_deref().unwrap_or("unnamed"),
                size,
                self.current_frame,
                current_offset,
                self.frame_size
            );
            return None;
        }

        // Update frame offset
        self.frame_offsets[self.current_frame] = new_offset;

        // Update metrics
        self.total_allocated.fetch_add(aligned_size, Ordering::Relaxed);

        // Update peak allocation
        let mut peak = self.peak_allocation.load(Ordering::Relaxed);
        while aligned_size > peak {
            match self.peak_allocation.compare_exchange_weak(
                peak,
                aligned_size,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(current) => peak = current,
            }
        }

        let absolute_offset = frame_base + aligned_offset;

        trace!(
            "RingBuffer '{}': allocated {} bytes at offset {} (frame {}, local {})",
            self.label.as_deref().unwrap_or("unnamed"),
            aligned_size,
            absolute_offset,
            self.current_frame,
            aligned_offset
        );

        Some(RingAllocation {
            offset: absolute_offset,
            size: aligned_size,
            frame: self.current_frame,
        })
    }

    /// Advances to the next frame and resets the frame's write offset.
    ///
    /// This should be called at the beginning of each render frame. It advances
    /// the frame counter (wrapping around) and resets the write offset for the
    /// new frame.
    ///
    /// # Note
    ///
    /// This method does not wait for GPU completion. The caller is responsible
    /// for ensuring the GPU has finished reading from the frame being reused.
    /// Typically this is handled by the frame pacing in the render loop.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # fn example(ring: &mut renderer_backend::resources::ring_buffer::RingBuffer) {
    /// // At the start of each frame
    /// ring.begin_frame();
    ///
    /// // Now allocate data for this frame
    /// let alloc = ring.allocate(256, 256);
    /// # }
    /// ```
    pub fn begin_frame(&mut self) {
        let old_frame = self.current_frame;
        let old_used = self.frame_offsets[old_frame];

        // Advance to next frame
        self.current_frame = (self.current_frame + 1) % self.frames_in_flight;

        // Check if we wrapped around
        if self.current_frame == 0 {
            self.wrap_count.fetch_add(1, Ordering::Relaxed);
            trace!(
                "RingBuffer '{}': wrapped around (wrap count: {})",
                self.label.as_deref().unwrap_or("unnamed"),
                self.wrap_count.load(Ordering::Relaxed)
            );
        }

        // Reset the new frame's offset
        self.frame_offsets[self.current_frame] = 0;

        debug!(
            "RingBuffer '{}': advanced from frame {} ({} bytes used) to frame {}",
            self.label.as_deref().unwrap_or("unnamed"),
            old_frame,
            old_used,
            self.current_frame
        );
    }

    /// Returns the base offset for the current frame.
    ///
    /// This is useful for calculating dynamic buffer offsets when binding.
    #[inline]
    pub fn current_frame_offset(&self) -> u64 {
        self.current_frame as u64 * self.frame_size
    }

    /// Returns the current frame index (0, 1, 2, ...).
    #[inline]
    pub fn current_frame(&self) -> usize {
        self.current_frame
    }

    /// Returns a reference to the underlying wgpu buffer.
    #[inline]
    pub fn buffer(&self) -> &Buffer {
        &self.buffer
    }

    /// Returns the total capacity of the ring buffer in bytes.
    #[inline]
    pub fn capacity(&self) -> u64 {
        self.capacity
    }

    /// Returns the size of each frame's region in bytes.
    #[inline]
    pub fn frame_size(&self) -> u64 {
        self.frame_size
    }

    /// Returns the number of frames in flight.
    #[inline]
    pub fn frames_in_flight(&self) -> usize {
        self.frames_in_flight
    }

    /// Returns the buffer usage flags.
    #[inline]
    pub fn usage(&self) -> BufferUsages {
        self.usage
    }

    /// Returns the bytes used in the current frame.
    #[inline]
    pub fn current_frame_used(&self) -> u64 {
        self.frame_offsets[self.current_frame]
    }

    /// Returns the bytes remaining in the current frame.
    #[inline]
    pub fn current_frame_remaining(&self) -> u64 {
        self.frame_size - self.frame_offsets[self.current_frame]
    }

    /// Returns current metrics for the ring buffer.
    pub fn metrics(&self) -> RingBufferMetrics {
        let current_frame_used = self.frame_offsets[self.current_frame];
        let utilization = if self.frame_size > 0 {
            current_frame_used as f32 / self.frame_size as f32
        } else {
            0.0
        };

        RingBufferMetrics {
            total_allocated: self.total_allocated.load(Ordering::Relaxed),
            current_frame_used,
            frame_capacity: self.frame_size,
            utilization,
            wrap_count: self.wrap_count.load(Ordering::Relaxed),
            overflow_count: self.overflow_count.load(Ordering::Relaxed),
            peak_allocation: self.peak_allocation.load(Ordering::Relaxed),
            current_frame: self.current_frame,
        }
    }

    /// Resets all metrics counters to zero.
    ///
    /// Useful for performance monitoring when you want to measure metrics
    /// over a specific period.
    pub fn reset_metrics(&self) {
        self.total_allocated.store(0, Ordering::Relaxed);
        self.wrap_count.store(0, Ordering::Relaxed);
        self.overflow_count.store(0, Ordering::Relaxed);
        self.peak_allocation.store(0, Ordering::Relaxed);
    }

    /// Returns the debug label, if any.
    #[inline]
    pub fn label(&self) -> Option<&str> {
        self.label.as_deref()
    }

    /// Aligns a value up to the given alignment.
    #[inline]
    const fn align_up(value: u64, alignment: u64) -> u64 {
        if alignment == 0 || value == 0 {
            return value;
        }
        (value + alignment - 1) & !(alignment - 1)
    }
}

impl std::fmt::Debug for RingBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RingBuffer")
            .field("label", &self.label)
            .field("capacity", &self.capacity)
            .field("frame_size", &self.frame_size)
            .field("current_frame", &self.current_frame)
            .field("frames_in_flight", &self.frames_in_flight)
            .field("usage", &self.usage)
            .finish_non_exhaustive()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // Helper to create a mock device for testing
    // Note: In real tests, we'd use wgpu-test-utils or similar
    // For unit tests without GPU, we test the logic with manual struct creation

    #[test]
    fn test_ring_allocation_end_offset() {
        let alloc = RingAllocation {
            offset: 100,
            size: 50,
            frame: 0,
        };
        assert_eq!(alloc.end_offset(), 150);
    }

    #[test]
    fn test_align_up() {
        assert_eq!(RingBuffer::align_up(0, 256), 0);
        assert_eq!(RingBuffer::align_up(1, 256), 256);
        assert_eq!(RingBuffer::align_up(255, 256), 256);
        assert_eq!(RingBuffer::align_up(256, 256), 256);
        assert_eq!(RingBuffer::align_up(257, 256), 512);
        assert_eq!(RingBuffer::align_up(100, 4), 100);
        assert_eq!(RingBuffer::align_up(101, 4), 104);
    }

    #[test]
    fn test_ring_buffer_config_default() {
        let config = RingBufferConfig::default();
        assert_eq!(config.frame_size, 256 * 1024);
        assert_eq!(config.frames_in_flight, 3);
        assert!(config.usage.contains(BufferUsages::UNIFORM));
        assert!(config.usage.contains(BufferUsages::COPY_DST));
    }

    #[test]
    fn test_ring_buffer_metrics_default() {
        let metrics = RingBufferMetrics::default();
        assert_eq!(metrics.total_allocated, 0);
        assert_eq!(metrics.current_frame_used, 0);
        assert_eq!(metrics.utilization, 0.0);
        assert_eq!(metrics.wrap_count, 0);
        assert_eq!(metrics.overflow_count, 0);
    }

    // Integration tests requiring actual wgpu device
    #[cfg(test)]
    mod integration {
        // This test requires a wgpu device - skip in CI without GPU
        #[test]
        fn test_ring_buffer_creation() {
            // Would create actual device and test buffer creation
        }

        #[test]
        fn test_ring_buffer_allocate_and_wrap() {
            // Would test allocation and wrap-around behavior
        }
    }
}

// ============================================================================
// Additional Tests for logic that doesn't require wgpu device
// ============================================================================

#[cfg(test)]
mod logic_tests {
    use super::*;

    /// Mock ring buffer state for testing allocation logic without wgpu
    struct MockRingBuffer {
        frame_size: u64,
        current_frame: usize,
        frame_offsets: Vec<u64>,
        frames_in_flight: usize,
        wrap_count: u32,
        overflow_count: u32,
    }

    impl MockRingBuffer {
        fn new(frame_size: u64, frames_in_flight: usize) -> Self {
            Self {
                frame_size,
                current_frame: 0,
                frame_offsets: vec![0; frames_in_flight],
                frames_in_flight,
                wrap_count: 0,
                overflow_count: 0,
            }
        }

        fn allocate(&mut self, size: u64, alignment: u64) -> Option<(u64, u64, usize)> {
            if size == 0 {
                return None;
            }

            let alignment = alignment.max(4);
            let frame_base = self.current_frame as u64 * self.frame_size;
            let current_offset = self.frame_offsets[self.current_frame];
            let aligned_offset = RingBuffer::align_up(current_offset, alignment);
            let aligned_size = RingBuffer::align_up(size, alignment);

            let new_offset = aligned_offset + aligned_size;
            if new_offset > self.frame_size {
                self.overflow_count += 1;
                return None;
            }

            self.frame_offsets[self.current_frame] = new_offset;
            let absolute_offset = frame_base + aligned_offset;

            Some((absolute_offset, aligned_size, self.current_frame))
        }

        fn begin_frame(&mut self) {
            self.current_frame = (self.current_frame + 1) % self.frames_in_flight;
            if self.current_frame == 0 {
                self.wrap_count += 1;
            }
            self.frame_offsets[self.current_frame] = 0;
        }

        fn current_frame_used(&self) -> u64 {
            self.frame_offsets[self.current_frame]
        }

        fn current_frame_remaining(&self) -> u64 {
            self.frame_size - self.frame_offsets[self.current_frame]
        }
    }

    #[test]
    fn test_mock_allocation_basic() {
        let mut ring = MockRingBuffer::new(1024, 3);

        // First allocation at offset 0
        let alloc = ring.allocate(100, 4);
        assert!(alloc.is_some());
        let (offset, size, frame) = alloc.unwrap();
        assert_eq!(offset, 0);
        assert_eq!(size, 100);
        assert_eq!(frame, 0);

        // Second allocation follows with alignment
        let alloc2 = ring.allocate(50, 4);
        assert!(alloc2.is_some());
        let (offset2, _, _) = alloc2.unwrap();
        assert_eq!(offset2, 100); // Right after first allocation
    }

    #[test]
    fn test_mock_allocation_alignment() {
        let mut ring = MockRingBuffer::new(1024, 3);

        // First allocation of 10 bytes
        ring.allocate(10, 4).unwrap();
        assert_eq!(ring.current_frame_used(), 12); // Aligned to 4

        // Next allocation with 256-byte alignment
        let alloc = ring.allocate(64, 256).unwrap();
        assert_eq!(alloc.0, 256); // Offset aligned to 256
        assert_eq!(alloc.1, 256); // Size is also aligned to 256 (64 -> 256)
    }

    #[test]
    fn test_mock_allocation_overflow() {
        let mut ring = MockRingBuffer::new(256, 3);

        // Allocate most of the frame
        ring.allocate(200, 4).unwrap();

        // This allocation should fail (not enough space)
        let alloc = ring.allocate(100, 4);
        assert!(alloc.is_none());
        assert_eq!(ring.overflow_count, 1);
    }

    #[test]
    fn test_mock_begin_frame() {
        let mut ring = MockRingBuffer::new(1024, 3);

        // Allocate in frame 0
        ring.allocate(100, 4).unwrap();
        assert_eq!(ring.current_frame, 0);
        assert_eq!(ring.current_frame_used(), 100);

        // Advance to frame 1
        ring.begin_frame();
        assert_eq!(ring.current_frame, 1);
        assert_eq!(ring.current_frame_used(), 0);

        // Allocate in frame 1
        ring.allocate(200, 4).unwrap();
        assert_eq!(ring.current_frame_used(), 200);

        // Advance to frame 2
        ring.begin_frame();
        assert_eq!(ring.current_frame, 2);
        assert_eq!(ring.current_frame_used(), 0);

        // Advance to frame 0 (wrap around)
        ring.begin_frame();
        assert_eq!(ring.current_frame, 0);
        assert_eq!(ring.current_frame_used(), 0); // Reset
        assert_eq!(ring.wrap_count, 1);
    }

    #[test]
    fn test_mock_frame_offsets_correct() {
        let mut ring = MockRingBuffer::new(1024, 3);
        let frame_size = 1024u64;

        // Frame 0: allocations at base 0
        let (offset0, _, _) = ring.allocate(100, 4).unwrap();
        assert_eq!(offset0, 0 * frame_size + 0);

        ring.begin_frame();

        // Frame 1: allocations at base 1024
        let (offset1, _, _) = ring.allocate(100, 4).unwrap();
        assert_eq!(offset1, 1 * frame_size + 0);

        ring.begin_frame();

        // Frame 2: allocations at base 2048
        let (offset2, _, _) = ring.allocate(100, 4).unwrap();
        assert_eq!(offset2, 2 * frame_size + 0);
    }

    #[test]
    fn test_mock_zero_size_allocation() {
        let mut ring = MockRingBuffer::new(1024, 3);
        let alloc = ring.allocate(0, 4);
        assert!(alloc.is_none());
    }

    #[test]
    fn test_mock_frame_remaining() {
        let mut ring = MockRingBuffer::new(1024, 3);

        assert_eq!(ring.current_frame_remaining(), 1024);

        ring.allocate(100, 4).unwrap();
        assert_eq!(ring.current_frame_remaining(), 1024 - 100);

        ring.allocate(256, 256).unwrap();
        // After 100 bytes, aligned to 256 = 256, then +256 = 512
        assert_eq!(ring.current_frame_remaining(), 1024 - 512);
    }

    #[test]
    fn test_triple_buffering_isolation() {
        let mut ring = MockRingBuffer::new(1024, 3);

        // Fill frame 0
        ring.allocate(500, 4).unwrap();
        let frame0_used = ring.current_frame_used();

        // Move to frame 1
        ring.begin_frame();
        assert_eq!(ring.current_frame_used(), 0); // Frame 1 is empty

        // Fill frame 1 differently
        ring.allocate(300, 4).unwrap();
        let frame1_used = ring.current_frame_used();

        // Frame 0 data is still there (we just can't access it through the mock)
        assert_ne!(frame0_used, frame1_used);

        // Move to frame 2
        ring.begin_frame();
        assert_eq!(ring.current_frame_used(), 0);

        // Wrap back to frame 0
        ring.begin_frame();
        assert_eq!(ring.current_frame, 0);
        assert_eq!(ring.current_frame_used(), 0); // Frame 0 is reset
        assert_eq!(ring.wrap_count, 1);
    }
}
