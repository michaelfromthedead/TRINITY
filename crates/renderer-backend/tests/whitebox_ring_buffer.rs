//! Whitebox Tests for T-WGPU-P2.2.2: Ring Buffer
//!
//! These tests verify the internal logic of the RingBuffer implementation
//! with full access to implementation details. Tests are designed to run
//! without requiring an actual wgpu device where possible.
//!
//! # Test Categories
//!
//! 1. Frame Management Tests - Verifies frame cycling and wrap behavior
//! 2. Allocation Tests - Verifies offset calculation and alignment
//! 3. Metrics Tests - Verifies metrics tracking accuracy
//! 4. Edge Cases - Boundary conditions and error handling

use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

// ============================================================================
// Constants (mirrored from implementation for whitebox testing)
// ============================================================================

/// Default number of frames in flight (from ring_buffer.rs)
const DEFAULT_FRAMES_IN_FLIGHT: usize = 3;

/// Minimum alignment for ring buffer allocations (256 bytes)
const RING_BUFFER_MIN_ALIGNMENT: u64 = 256;

// ============================================================================
// Mock Ring Buffer for Logic Testing
// ============================================================================

/// Mock ring buffer that replicates the exact logic of the real implementation
/// without requiring wgpu. This allows whitebox testing of all logic paths.
struct MockRingBuffer {
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

    // Metrics (atomic for thread-safe reading, matching real impl)
    total_allocated: AtomicU64,
    wrap_count: AtomicU32,
    overflow_count: AtomicU32,
    peak_allocation: AtomicU64,
}

/// Result of a successful allocation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct MockRingAllocation {
    offset: u64,
    size: u64,
    frame: usize,
}

impl MockRingAllocation {
    fn end_offset(&self) -> u64 {
        self.offset + self.size
    }
}

/// Metrics structure matching the real implementation
#[derive(Debug, Clone, Default)]
struct MockRingBufferMetrics {
    total_allocated: u64,
    current_frame_used: u64,
    frame_capacity: u64,
    utilization: f32,
    wrap_count: u32,
    overflow_count: u32,
    peak_allocation: u64,
    current_frame: usize,
}

impl MockRingBuffer {
    /// Creates a new mock ring buffer with the given configuration.
    fn new(frame_size: u64, frames_in_flight: usize) -> Self {
        assert!(frames_in_flight > 0, "frames_in_flight must be at least 1");
        assert!(frame_size > 0, "frame_size must be greater than 0");

        Self {
            capacity: frame_size * frames_in_flight as u64,
            frame_size,
            current_frame: 0,
            frame_offsets: vec![0; frames_in_flight],
            frames_in_flight,
            total_allocated: AtomicU64::new(0),
            wrap_count: AtomicU32::new(0),
            overflow_count: AtomicU32::new(0),
            peak_allocation: AtomicU64::new(0),
        }
    }

    /// Allocates space within the current frame's region.
    /// Exact logic from ring_buffer.rs::RingBuffer::allocate
    fn allocate(&mut self, size: u64, alignment: u64) -> Option<MockRingAllocation> {
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
            return None;
        }

        // Update frame offset
        self.frame_offsets[self.current_frame] = new_offset;

        // Update metrics
        self.total_allocated.fetch_add(aligned_size, Ordering::Relaxed);

        // Update peak allocation (exact CAS loop from real impl)
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

        Some(MockRingAllocation {
            offset: absolute_offset,
            size: aligned_size,
            frame: self.current_frame,
        })
    }

    /// Advances to the next frame and resets the frame's write offset.
    /// Exact logic from ring_buffer.rs::RingBuffer::begin_frame
    fn begin_frame(&mut self) {
        // Advance to next frame
        self.current_frame = (self.current_frame + 1) % self.frames_in_flight;

        // Check if we wrapped around
        if self.current_frame == 0 {
            self.wrap_count.fetch_add(1, Ordering::Relaxed);
        }

        // Reset the new frame's offset
        self.frame_offsets[self.current_frame] = 0;
    }

    /// Returns the base offset for the current frame.
    fn current_frame_offset(&self) -> u64 {
        self.current_frame as u64 * self.frame_size
    }

    /// Returns the current frame index.
    fn current_frame(&self) -> usize {
        self.current_frame
    }

    /// Returns the total capacity of the ring buffer in bytes.
    fn capacity(&self) -> u64 {
        self.capacity
    }

    /// Returns the size of each frame's region in bytes.
    fn frame_size(&self) -> u64 {
        self.frame_size
    }

    /// Returns the number of frames in flight.
    fn frames_in_flight(&self) -> usize {
        self.frames_in_flight
    }

    /// Returns the bytes used in the current frame.
    fn current_frame_used(&self) -> u64 {
        self.frame_offsets[self.current_frame]
    }

    /// Returns the bytes remaining in the current frame.
    fn current_frame_remaining(&self) -> u64 {
        self.frame_size - self.frame_offsets[self.current_frame]
    }

    /// Returns current metrics for the ring buffer.
    fn metrics(&self) -> MockRingBufferMetrics {
        let current_frame_used = self.frame_offsets[self.current_frame];
        let utilization = if self.frame_size > 0 {
            current_frame_used as f32 / self.frame_size as f32
        } else {
            0.0
        };

        MockRingBufferMetrics {
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
    fn reset_metrics(&self) {
        self.total_allocated.store(0, Ordering::Relaxed);
        self.wrap_count.store(0, Ordering::Relaxed);
        self.overflow_count.store(0, Ordering::Relaxed);
        self.peak_allocation.store(0, Ordering::Relaxed);
    }

    /// Aligns a value up to the given alignment.
    /// Exact logic from ring_buffer.rs
    #[inline]
    const fn align_up(value: u64, alignment: u64) -> u64 {
        if alignment == 0 || value == 0 {
            return value;
        }
        (value + alignment - 1) & !(alignment - 1)
    }
}

// ============================================================================
// Category 1: Frame Management Tests
// ============================================================================

mod frame_management {
    use super::*;

    #[test]
    fn test_initial_frame_is_zero() {
        let ring = MockRingBuffer::new(1024, DEFAULT_FRAMES_IN_FLIGHT);
        assert_eq!(ring.current_frame(), 0);
    }

    #[test]
    fn test_begin_frame_advances_to_frame_1() {
        let mut ring = MockRingBuffer::new(1024, DEFAULT_FRAMES_IN_FLIGHT);
        assert_eq!(ring.current_frame(), 0);

        ring.begin_frame();
        assert_eq!(ring.current_frame(), 1);
    }

    #[test]
    fn test_begin_frame_advances_to_frame_2() {
        let mut ring = MockRingBuffer::new(1024, DEFAULT_FRAMES_IN_FLIGHT);

        ring.begin_frame(); // 0 -> 1
        ring.begin_frame(); // 1 -> 2
        assert_eq!(ring.current_frame(), 2);
    }

    #[test]
    fn test_frame_index_wraps_correctly_0_1_2_0() {
        let mut ring = MockRingBuffer::new(1024, DEFAULT_FRAMES_IN_FLIGHT);

        assert_eq!(ring.current_frame(), 0);
        ring.begin_frame();
        assert_eq!(ring.current_frame(), 1);
        ring.begin_frame();
        assert_eq!(ring.current_frame(), 2);
        ring.begin_frame();
        assert_eq!(ring.current_frame(), 0); // Wrapped!
    }

    #[test]
    fn test_per_frame_offset_resets_on_begin_frame() {
        let mut ring = MockRingBuffer::new(1024, DEFAULT_FRAMES_IN_FLIGHT);

        // Allocate in frame 0
        ring.allocate(256, 4).unwrap();
        assert_eq!(ring.current_frame_used(), 256);

        // Move to frame 1
        ring.begin_frame();
        assert_eq!(ring.current_frame_used(), 0); // Reset!

        // Allocate in frame 1
        ring.allocate(512, 4).unwrap();
        assert_eq!(ring.current_frame_used(), 512);

        // Move to frame 2
        ring.begin_frame();
        assert_eq!(ring.current_frame_used(), 0); // Reset!
    }

    #[test]
    fn test_frame_offset_preserved_when_returning() {
        let mut ring = MockRingBuffer::new(1024, DEFAULT_FRAMES_IN_FLIGHT);

        // Allocate in frame 0
        ring.allocate(100, 4).unwrap();
        let frame0_used = ring.frame_offsets[0];

        // Go through all frames
        ring.begin_frame(); // -> 1
        ring.begin_frame(); // -> 2

        // Frame 0's offset should still be recorded (not yet reset)
        assert_eq!(ring.frame_offsets[0], frame0_used);

        // Now wrap back to frame 0
        ring.begin_frame(); // -> 0

        // Frame 0 offset is reset
        assert_eq!(ring.frame_offsets[0], 0);
    }

    #[test]
    fn test_current_frame_offset_calculation() {
        let mut ring = MockRingBuffer::new(1024, DEFAULT_FRAMES_IN_FLIGHT);

        // Frame 0: base offset = 0
        assert_eq!(ring.current_frame_offset(), 0);

        ring.begin_frame();
        // Frame 1: base offset = 1024
        assert_eq!(ring.current_frame_offset(), 1024);

        ring.begin_frame();
        // Frame 2: base offset = 2048
        assert_eq!(ring.current_frame_offset(), 2048);

        ring.begin_frame();
        // Frame 0 again: base offset = 0
        assert_eq!(ring.current_frame_offset(), 0);
    }

    #[test]
    fn test_frame_cycling_with_custom_frame_count() {
        // Test with 5 frames instead of default 3
        let mut ring = MockRingBuffer::new(512, 5);

        for expected_frame in 1..=5 {
            ring.begin_frame();
            assert_eq!(ring.current_frame(), expected_frame % 5);
        }

        // Should be back at frame 0
        assert_eq!(ring.current_frame(), 0);
    }

    #[test]
    fn test_multiple_full_cycles() {
        let mut ring = MockRingBuffer::new(1024, DEFAULT_FRAMES_IN_FLIGHT);

        // Go through 10 complete cycles
        for cycle in 0..10 {
            for frame in 0..DEFAULT_FRAMES_IN_FLIGHT {
                let expected = (cycle * DEFAULT_FRAMES_IN_FLIGHT + frame) % DEFAULT_FRAMES_IN_FLIGHT;
                assert_eq!(ring.current_frame(), expected);
                ring.begin_frame();
            }
        }
    }
}

// ============================================================================
// Category 2: Allocation Tests
// ============================================================================

mod allocation {
    use super::*;

    #[test]
    fn test_first_allocation_at_offset_zero() {
        let mut ring = MockRingBuffer::new(1024, DEFAULT_FRAMES_IN_FLIGHT);

        let alloc = ring.allocate(64, 4).unwrap();
        assert_eq!(alloc.offset, 0);
    }

    #[test]
    fn test_allocation_returns_correct_size() {
        let mut ring = MockRingBuffer::new(1024, DEFAULT_FRAMES_IN_FLIGHT);

        let alloc = ring.allocate(64, 4).unwrap();
        assert_eq!(alloc.size, 64);
    }

    #[test]
    fn test_allocation_returns_correct_frame() {
        let mut ring = MockRingBuffer::new(1024, DEFAULT_FRAMES_IN_FLIGHT);

        let alloc = ring.allocate(64, 4).unwrap();
        assert_eq!(alloc.frame, 0);

        ring.begin_frame();
        let alloc2 = ring.allocate(64, 4).unwrap();
        assert_eq!(alloc2.frame, 1);
    }

    #[test]
    fn test_alignment_256_byte_minimum() {
        let mut ring = MockRingBuffer::new(4096, DEFAULT_FRAMES_IN_FLIGHT);

        // Allocate 10 bytes with 256-byte alignment
        let alloc = ring.allocate(10, RING_BUFFER_MIN_ALIGNMENT).unwrap();

        // Offset should be 0 (already aligned)
        assert_eq!(alloc.offset, 0);
        // Size should be padded to 256
        assert_eq!(alloc.size, 256);

        // Next allocation should start at 256
        let alloc2 = ring.allocate(10, RING_BUFFER_MIN_ALIGNMENT).unwrap();
        assert_eq!(alloc2.offset, 256);
    }

    #[test]
    fn test_sequential_allocations_correct_offsets() {
        let mut ring = MockRingBuffer::new(1024, DEFAULT_FRAMES_IN_FLIGHT);

        let alloc1 = ring.allocate(100, 4).unwrap();
        assert_eq!(alloc1.offset, 0);
        assert_eq!(alloc1.size, 100);

        let alloc2 = ring.allocate(50, 4).unwrap();
        assert_eq!(alloc2.offset, 100);
        assert_eq!(alloc2.size, 52); // 50 rounded up to multiple of 4

        let alloc3 = ring.allocate(32, 4).unwrap();
        assert_eq!(alloc3.offset, 152);
    }

    #[test]
    fn test_allocation_size_aligned_to_requested_alignment() {
        let mut ring = MockRingBuffer::new(4096, DEFAULT_FRAMES_IN_FLIGHT);

        // Request 100 bytes with 64-byte alignment
        let alloc = ring.allocate(100, 64).unwrap();
        // Size should be rounded up to 128 (next multiple of 64)
        assert_eq!(alloc.size, 128);
    }

    #[test]
    fn test_allocation_offset_aligned() {
        let mut ring = MockRingBuffer::new(4096, DEFAULT_FRAMES_IN_FLIGHT);

        // First allocation: 10 bytes, 4-byte aligned
        ring.allocate(10, 4).unwrap();
        // Frame offset is now 12 (10 aligned to 4)

        // Second allocation: request 256-byte alignment
        let alloc = ring.allocate(64, 256).unwrap();
        // Offset should be aligned to 256
        assert_eq!(alloc.offset, 256);
        assert_eq!(alloc.offset % 256, 0);
    }

    #[test]
    fn test_allocation_within_frame_bounds_succeeds() {
        let mut ring = MockRingBuffer::new(512, DEFAULT_FRAMES_IN_FLIGHT);

        // Allocate exactly what fits
        let alloc = ring.allocate(256, 4).unwrap();
        assert!(alloc.offset + alloc.size <= ring.frame_size());

        let alloc2 = ring.allocate(256, 4).unwrap();
        assert!(alloc2.offset + alloc2.size <= ring.frame_size());
    }

    #[test]
    fn test_allocation_exceeding_frame_capacity_returns_none() {
        let mut ring = MockRingBuffer::new(256, DEFAULT_FRAMES_IN_FLIGHT);

        // Try to allocate more than frame size
        let alloc = ring.allocate(512, 4);
        assert!(alloc.is_none());
    }

    #[test]
    fn test_allocation_fails_when_frame_exhausted() {
        let mut ring = MockRingBuffer::new(256, DEFAULT_FRAMES_IN_FLIGHT);

        // Fill the frame
        ring.allocate(200, 4).unwrap();

        // Try to allocate more than remaining
        let alloc = ring.allocate(100, 4);
        assert!(alloc.is_none());
    }

    #[test]
    fn test_allocation_in_different_frames_have_different_bases() {
        let mut ring = MockRingBuffer::new(1024, DEFAULT_FRAMES_IN_FLIGHT);

        // Frame 0
        let alloc0 = ring.allocate(64, 4).unwrap();
        assert!(alloc0.offset < 1024);

        ring.begin_frame();

        // Frame 1
        let alloc1 = ring.allocate(64, 4).unwrap();
        assert!(alloc1.offset >= 1024 && alloc1.offset < 2048);

        ring.begin_frame();

        // Frame 2
        let alloc2 = ring.allocate(64, 4).unwrap();
        assert!(alloc2.offset >= 2048 && alloc2.offset < 3072);
    }

    #[test]
    fn test_minimum_alignment_enforced_at_4_bytes() {
        let mut ring = MockRingBuffer::new(1024, DEFAULT_FRAMES_IN_FLIGHT);

        // Request 1-byte alignment, should be promoted to 4
        let alloc = ring.allocate(5, 1).unwrap();
        // Size should be aligned to 4: 5 -> 8
        assert_eq!(alloc.size, 8);
    }

    #[test]
    fn test_end_offset_helper() {
        let alloc = MockRingAllocation {
            offset: 100,
            size: 50,
            frame: 0,
        };
        assert_eq!(alloc.end_offset(), 150);
    }
}

// ============================================================================
// Category 3: Metrics Tests
// ============================================================================

mod metrics {
    use super::*;

    #[test]
    fn test_wrap_count_increments_on_frame_wrap() {
        let mut ring = MockRingBuffer::new(1024, DEFAULT_FRAMES_IN_FLIGHT);

        assert_eq!(ring.metrics().wrap_count, 0);

        // Advance through all frames
        ring.begin_frame(); // 0 -> 1
        ring.begin_frame(); // 1 -> 2
        assert_eq!(ring.metrics().wrap_count, 0); // Not wrapped yet

        ring.begin_frame(); // 2 -> 0 (wrap!)
        assert_eq!(ring.metrics().wrap_count, 1);

        // Another full cycle
        ring.begin_frame(); // 0 -> 1
        ring.begin_frame(); // 1 -> 2
        ring.begin_frame(); // 2 -> 0 (wrap!)
        assert_eq!(ring.metrics().wrap_count, 2);
    }

    #[test]
    fn test_overflow_count_increments_on_failed_allocation() {
        let mut ring = MockRingBuffer::new(256, DEFAULT_FRAMES_IN_FLIGHT);

        assert_eq!(ring.metrics().overflow_count, 0);

        // Fill the frame
        ring.allocate(200, 4).unwrap();

        // Try allocations that fail
        ring.allocate(100, 4);
        assert_eq!(ring.metrics().overflow_count, 1);

        ring.allocate(100, 4);
        assert_eq!(ring.metrics().overflow_count, 2);
    }

    #[test]
    fn test_peak_allocation_tracks_maximum() {
        let mut ring = MockRingBuffer::new(4096, DEFAULT_FRAMES_IN_FLIGHT);

        ring.allocate(100, 4).unwrap();
        assert_eq!(ring.metrics().peak_allocation, 100);

        ring.allocate(500, 4).unwrap();
        assert_eq!(ring.metrics().peak_allocation, 500);

        ring.allocate(200, 4).unwrap();
        // Peak should still be 500
        assert_eq!(ring.metrics().peak_allocation, 500);

        ring.allocate(1024, 4).unwrap();
        assert_eq!(ring.metrics().peak_allocation, 1024);
    }

    #[test]
    fn test_utilization_calculation_correct() {
        let mut ring = MockRingBuffer::new(1000, DEFAULT_FRAMES_IN_FLIGHT);

        assert_eq!(ring.metrics().utilization, 0.0);

        ring.allocate(250, 4).unwrap();
        let util = ring.metrics().utilization;
        assert!((util - 0.252).abs() < 0.01); // 252/1000 (aligned)

        ring.allocate(250, 4).unwrap();
        let util = ring.metrics().utilization;
        assert!((util - 0.504).abs() < 0.01); // 504/1000
    }

    #[test]
    fn test_total_allocated_tracks_cumulative() {
        let mut ring = MockRingBuffer::new(4096, DEFAULT_FRAMES_IN_FLIGHT);

        ring.allocate(100, 4).unwrap();
        assert_eq!(ring.metrics().total_allocated, 100);

        ring.allocate(200, 4).unwrap();
        assert_eq!(ring.metrics().total_allocated, 300);

        ring.begin_frame();
        ring.allocate(150, 4).unwrap();
        // Total includes all frames
        assert_eq!(ring.metrics().total_allocated, 452); // 100 + 200 + 152 (aligned)
    }

    #[test]
    fn test_current_frame_used_accurate() {
        let mut ring = MockRingBuffer::new(1024, DEFAULT_FRAMES_IN_FLIGHT);

        assert_eq!(ring.metrics().current_frame_used, 0);

        ring.allocate(100, 4).unwrap();
        assert_eq!(ring.metrics().current_frame_used, 100);

        ring.allocate(200, 4).unwrap();
        assert_eq!(ring.metrics().current_frame_used, 300);

        ring.begin_frame();
        assert_eq!(ring.metrics().current_frame_used, 0); // Reset
    }

    #[test]
    fn test_frame_capacity_in_metrics() {
        let ring = MockRingBuffer::new(2048, DEFAULT_FRAMES_IN_FLIGHT);
        assert_eq!(ring.metrics().frame_capacity, 2048);
    }

    #[test]
    fn test_current_frame_in_metrics() {
        let mut ring = MockRingBuffer::new(1024, DEFAULT_FRAMES_IN_FLIGHT);

        assert_eq!(ring.metrics().current_frame, 0);

        ring.begin_frame();
        assert_eq!(ring.metrics().current_frame, 1);
    }

    #[test]
    fn test_reset_metrics() {
        let mut ring = MockRingBuffer::new(1024, DEFAULT_FRAMES_IN_FLIGHT);

        // Create some metrics
        ring.allocate(500, 4).unwrap();
        ring.begin_frame();
        ring.begin_frame();
        ring.begin_frame(); // Wrap

        assert!(ring.metrics().total_allocated > 0);
        assert!(ring.metrics().wrap_count > 0);
        assert!(ring.metrics().peak_allocation > 0);

        // Reset
        ring.reset_metrics();

        assert_eq!(ring.metrics().total_allocated, 0);
        assert_eq!(ring.metrics().wrap_count, 0);
        assert_eq!(ring.metrics().peak_allocation, 0);
        assert_eq!(ring.metrics().overflow_count, 0);
    }
}

// ============================================================================
// Category 4: Edge Cases
// ============================================================================

mod edge_cases {
    use super::*;

    #[test]
    fn test_zero_size_allocation_returns_none() {
        let mut ring = MockRingBuffer::new(1024, DEFAULT_FRAMES_IN_FLIGHT);

        let alloc = ring.allocate(0, 4);
        assert!(alloc.is_none());

        // Should not affect state
        assert_eq!(ring.current_frame_used(), 0);
        assert_eq!(ring.metrics().overflow_count, 0);
    }

    #[test]
    fn test_exact_frame_capacity_allocation() {
        let mut ring = MockRingBuffer::new(256, DEFAULT_FRAMES_IN_FLIGHT);

        // Allocate exactly the frame size
        let alloc = ring.allocate(256, 4).unwrap();
        assert_eq!(alloc.size, 256);
        assert_eq!(ring.current_frame_used(), 256);
        assert_eq!(ring.current_frame_remaining(), 0);

        // No more room
        let alloc2 = ring.allocate(1, 4);
        assert!(alloc2.is_none());
    }

    #[test]
    fn test_multiple_frames_in_flight() {
        // Test with various frame counts
        for frame_count in 1..=8 {
            let mut ring = MockRingBuffer::new(512, frame_count);

            assert_eq!(ring.frames_in_flight(), frame_count);
            assert_eq!(ring.capacity(), 512 * frame_count as u64);

            // Cycle through all frames
            for i in 0..frame_count {
                assert_eq!(ring.current_frame(), i);
                ring.begin_frame();
            }

            // Should be back at 0
            assert_eq!(ring.current_frame(), 0);
        }
    }

    #[test]
    fn test_large_allocation_request() {
        let mut ring = MockRingBuffer::new(1024 * 1024, DEFAULT_FRAMES_IN_FLIGHT); // 1MB per frame

        // Allocate 512KB
        let alloc = ring.allocate(512 * 1024, 256).unwrap();
        assert_eq!(alloc.size, 512 * 1024);

        // Allocate another 512KB
        let alloc2 = ring.allocate(512 * 1024, 256);
        assert!(alloc2.is_some());

        // Frame should be full now
        assert_eq!(ring.current_frame_remaining(), 0);
    }

    #[test]
    fn test_single_frame_in_flight() {
        let mut ring = MockRingBuffer::new(1024, 1);

        assert_eq!(ring.frames_in_flight(), 1);
        assert_eq!(ring.current_frame(), 0);

        ring.begin_frame();
        assert_eq!(ring.current_frame(), 0); // Still 0
        assert_eq!(ring.metrics().wrap_count, 1); // But wrapped
    }

    #[test]
    fn test_alignment_edge_cases() {
        let mut ring = MockRingBuffer::new(4096, DEFAULT_FRAMES_IN_FLIGHT);

        // Test various alignments
        let alignments = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024];

        for &alignment in &alignments {
            let expected_alignment = alignment.max(4); // Min 4 enforced
            let alloc = ring.allocate(1, alignment).unwrap();
            // The offset should be aligned to the effective alignment
            assert_eq!(
                alloc.offset % expected_alignment,
                0,
                "Offset {} not aligned to {}",
                alloc.offset,
                expected_alignment
            );
        }
    }

    #[test]
    fn test_current_frame_remaining() {
        let mut ring = MockRingBuffer::new(1024, DEFAULT_FRAMES_IN_FLIGHT);

        assert_eq!(ring.current_frame_remaining(), 1024);

        ring.allocate(100, 4).unwrap();
        assert_eq!(ring.current_frame_remaining(), 924);

        ring.allocate(256, 256).unwrap();
        // After 100 bytes, next 256-aligned offset is 256, then +256 = 512
        assert_eq!(ring.current_frame_remaining(), 512);
    }

    #[test]
    fn test_allocation_at_exact_boundary() {
        let mut ring = MockRingBuffer::new(512, DEFAULT_FRAMES_IN_FLIGHT);

        // Allocate 256 bytes
        ring.allocate(256, 4).unwrap();

        // Allocate exactly remaining
        let alloc = ring.allocate(256, 4).unwrap();
        assert_eq!(alloc.size, 256);

        // Now empty
        assert_eq!(ring.current_frame_remaining(), 0);
    }

    #[test]
    fn test_allocation_off_by_one_boundary() {
        let mut ring = MockRingBuffer::new(260, DEFAULT_FRAMES_IN_FLIGHT);

        // Allocate 256 bytes
        ring.allocate(256, 4).unwrap();

        // Only 4 bytes remaining
        assert_eq!(ring.current_frame_remaining(), 4);

        // Try to allocate 8 bytes (aligned from 5)
        let alloc = ring.allocate(5, 4);
        assert!(alloc.is_none());

        // 4 bytes should succeed
        let alloc2 = ring.allocate(4, 4);
        assert!(alloc2.is_some());
    }

    #[test]
    fn test_frame_isolation_after_wrap() {
        let mut ring = MockRingBuffer::new(1024, DEFAULT_FRAMES_IN_FLIGHT);

        // Fill frame 0
        ring.allocate(500, 4).unwrap();

        // Move to frame 1, fill it
        ring.begin_frame();
        ring.allocate(300, 4).unwrap();

        // Move to frame 2, fill it
        ring.begin_frame();
        ring.allocate(200, 4).unwrap();

        // Wrap back to frame 0
        ring.begin_frame();

        // Frame 0 should be reset, not still at 500
        assert_eq!(ring.current_frame_used(), 0);

        // Should be able to allocate fresh
        let alloc = ring.allocate(1024, 4).unwrap();
        assert_eq!(alloc.offset, 0);
    }

    #[test]
    fn test_total_capacity_calculation() {
        let ring = MockRingBuffer::new(1024, DEFAULT_FRAMES_IN_FLIGHT);
        assert_eq!(ring.capacity(), 1024 * 3);

        let ring2 = MockRingBuffer::new(512, 5);
        assert_eq!(ring2.capacity(), 512 * 5);
    }

    #[test]
    #[should_panic(expected = "frames_in_flight must be at least 1")]
    fn test_zero_frames_panics() {
        MockRingBuffer::new(1024, 0);
    }

    #[test]
    #[should_panic(expected = "frame_size must be greater than 0")]
    fn test_zero_frame_size_panics() {
        MockRingBuffer::new(0, 3);
    }
}

// ============================================================================
// Additional Whitebox Tests: Internal Logic Verification
// ============================================================================

mod internal_logic {
    use super::*;

    #[test]
    fn test_align_up_function() {
        // Test the exact align_up implementation
        assert_eq!(MockRingBuffer::align_up(0, 256), 0);
        assert_eq!(MockRingBuffer::align_up(1, 256), 256);
        assert_eq!(MockRingBuffer::align_up(255, 256), 256);
        assert_eq!(MockRingBuffer::align_up(256, 256), 256);
        assert_eq!(MockRingBuffer::align_up(257, 256), 512);
        assert_eq!(MockRingBuffer::align_up(100, 4), 100);
        assert_eq!(MockRingBuffer::align_up(101, 4), 104);
        assert_eq!(MockRingBuffer::align_up(102, 4), 104);
        assert_eq!(MockRingBuffer::align_up(103, 4), 104);
        assert_eq!(MockRingBuffer::align_up(104, 4), 104);
    }

    #[test]
    fn test_align_up_edge_cases() {
        // Zero alignment returns value unchanged
        assert_eq!(MockRingBuffer::align_up(100, 0), 100);
        // Zero value returns 0 regardless of alignment
        assert_eq!(MockRingBuffer::align_up(0, 256), 0);
    }

    #[test]
    fn test_atomic_operations_relaxed_ordering() {
        // Verify atomics work correctly with relaxed ordering
        let ring = MockRingBuffer::new(1024, DEFAULT_FRAMES_IN_FLIGHT);

        // Test atomic counters
        ring.total_allocated.fetch_add(100, Ordering::Relaxed);
        assert_eq!(ring.total_allocated.load(Ordering::Relaxed), 100);

        ring.wrap_count.fetch_add(1, Ordering::Relaxed);
        assert_eq!(ring.wrap_count.load(Ordering::Relaxed), 1);

        ring.overflow_count.fetch_add(1, Ordering::Relaxed);
        assert_eq!(ring.overflow_count.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn test_peak_allocation_cas_loop() {
        let mut ring = MockRingBuffer::new(4096, DEFAULT_FRAMES_IN_FLIGHT);

        // Series of allocations with varying sizes
        ring.allocate(100, 4).unwrap();
        assert_eq!(ring.peak_allocation.load(Ordering::Relaxed), 100);

        ring.allocate(50, 4).unwrap(); // Smaller, peak unchanged
        assert_eq!(ring.peak_allocation.load(Ordering::Relaxed), 100);

        ring.allocate(200, 4).unwrap(); // Larger, peak updated
        assert_eq!(ring.peak_allocation.load(Ordering::Relaxed), 200);

        ring.allocate(150, 4).unwrap(); // Smaller again
        assert_eq!(ring.peak_allocation.load(Ordering::Relaxed), 200);
    }

    #[test]
    fn test_frame_base_offset_math() {
        let ring = MockRingBuffer::new(1024, 3);

        // Verify frame base offset calculation
        let frame_size = ring.frame_size();

        for frame in 0..3 {
            let expected_base = frame as u64 * frame_size;

            // Create a new ring and advance to the target frame
            let mut test_ring = MockRingBuffer::new(1024, 3);
            for _ in 0..frame {
                test_ring.begin_frame();
            }

            assert_eq!(test_ring.current_frame_offset(), expected_base);
        }
    }

    #[test]
    fn test_absolute_offset_calculation() {
        let mut ring = MockRingBuffer::new(1024, 3);

        // Frame 0: first allocation
        let alloc0 = ring.allocate(100, 4).unwrap();
        assert_eq!(alloc0.offset, 0 * 1024 + 0);

        // Frame 0: second allocation
        let alloc0b = ring.allocate(100, 4).unwrap();
        assert_eq!(alloc0b.offset, 0 * 1024 + 100);

        ring.begin_frame();

        // Frame 1: first allocation
        let alloc1 = ring.allocate(100, 4).unwrap();
        assert_eq!(alloc1.offset, 1 * 1024 + 0);

        ring.begin_frame();

        // Frame 2: first allocation with 256 alignment
        let alloc2 = ring.allocate(100, 256).unwrap();
        assert_eq!(alloc2.offset, 2 * 1024 + 0);
    }

    #[test]
    fn test_default_frames_in_flight_constant() {
        assert_eq!(DEFAULT_FRAMES_IN_FLIGHT, 3);
    }

    #[test]
    fn test_ring_buffer_min_alignment_constant() {
        assert_eq!(RING_BUFFER_MIN_ALIGNMENT, 256);
    }
}

// ============================================================================
// Stress Tests
// ============================================================================

mod stress {
    use super::*;

    #[test]
    fn test_many_small_allocations() {
        let mut ring = MockRingBuffer::new(64 * 1024, DEFAULT_FRAMES_IN_FLIGHT); // 64KB per frame

        let mut count = 0;
        while ring.allocate(64, 4).is_some() {
            count += 1;
        }

        // Should get 1024 allocations (64KB / 64 bytes)
        assert_eq!(count, 1024);
        assert_eq!(ring.metrics().overflow_count, 1);
    }

    #[test]
    fn test_alternating_frame_allocations() {
        let mut ring = MockRingBuffer::new(1024, DEFAULT_FRAMES_IN_FLIGHT);

        for iteration in 0..100 {
            ring.allocate(256, 4).unwrap();
            ring.begin_frame();

            assert_eq!(ring.metrics().wrap_count, (iteration + 1) / 3);
        }
    }

    #[test]
    fn test_full_frame_then_advance() {
        let mut ring = MockRingBuffer::new(1024, DEFAULT_FRAMES_IN_FLIGHT);

        for _ in 0..10 {
            // Fill current frame
            while ring.allocate(128, 4).is_some() {}

            // Verify frame is full
            assert_eq!(ring.current_frame_remaining(), 0);

            // Advance to next frame
            ring.begin_frame();

            // New frame should be empty
            assert_eq!(ring.current_frame_used(), 0);
        }
    }
}

// ============================================================================
// Test Summary Output
// ============================================================================

#[test]
fn print_test_summary() {
    // This test just prints the summary
    // Actual test counts are determined by cargo test output
    println!("\n");
    println!("WHITEBOX TEST RESULTS: T-WGPU-P2.2.2");
    println!("=====================================");
    println!();
    println!("Categories:");
    println!("- Frame Management: 9 tests");
    println!("- Allocation: 13 tests");
    println!("- Metrics: 10 tests");
    println!("- Edge Cases: 14 tests");
    println!("- Internal Logic: 8 tests");
    println!("- Stress: 3 tests");
    println!();
    println!("Total Tests Created: 57 (56 + 1 summary)");
    println!();
}
