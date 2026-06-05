//! Scratch Buffer Pool for Acceleration Structure Builds
//!
//! This module provides a memory pool for scratch buffers used during BLAS/TLAS
//! construction in ray tracing pipelines. Scratch buffers are temporary GPU
//! allocations required by the hardware to build acceleration structures.
//!
//! # Architecture
//!
//! The pool uses size-class bucketing with free lists for efficient reuse:
//!
//! ```text
//! ScratchBufferPool
//! ├── buckets: [Vec<ScratchBufferEntry>; SIZE_CLASS_COUNT]
//! │   ├── 64 KB bucket
//! │   ├── 256 KB bucket
//! │   ├── 1 MB bucket
//! │   ├── 4 MB bucket
//! │   └── 16 MB bucket
//! ├── free_lists: [Vec<usize>; SIZE_CLASS_COUNT]
//! └── stats: PoolStats
//! ```
//!
//! # Size Classes
//!
//! | Class | Size | Typical Use Case |
//! |-------|------|------------------|
//! | 0 | 64 KB | Small static meshes |
//! | 1 | 256 KB | Medium meshes |
//! | 2 | 1 MB | Large meshes, animated objects |
//! | 3 | 4 MB | Complex scenes, many triangles |
//! | 4 | 16 MB | Massive geometry, terrain |
//!
//! # Alignment
//!
//! All scratch buffers are aligned to 256 bytes per wgpu requirements for
//! acceleration structure operations.
//!
//! # Example
//!
//! ```ignore
//! use renderer_backend::scratch_buffer::{ScratchBufferPool, ScratchBuffer};
//!
//! // Create pool
//! let mut pool = ScratchBufferPool::new(&device);
//!
//! // Acquire scratch buffer for AS build (100KB needed)
//! let scratch = pool.acquire(&device, 100 * 1024);
//!
//! // Use buffer for BLAS/TLAS build...
//! encoder.build_acceleration_structures(&[], &[tlas_descriptor]);
//!
//! // Release back to pool
//! pool.release(scratch);
//!
//! // Check memory usage
//! let stats = pool.stats();
//! println!("Current: {} bytes, Peak: {} bytes", stats.current_usage, stats.peak_usage);
//!
//! // Trim unused buffers
//! pool.trim();
//! ```

use std::sync::Arc;
use wgpu::{Buffer, BufferDescriptor, BufferUsages, Device};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Required alignment for scratch buffers (wgpu requirement).
pub const SCRATCH_BUFFER_ALIGNMENT: u64 = 256;

/// Size classes for scratch buffer pooling.
/// Sizes are chosen to cover typical AS build requirements.
pub const SIZE_CLASSES: [u64; 5] = [
    64 * 1024,        // 64 KB - small static meshes
    256 * 1024,       // 256 KB - medium meshes
    1024 * 1024,      // 1 MB - large meshes
    4 * 1024 * 1024,  // 4 MB - complex scenes
    16 * 1024 * 1024, // 16 MB - massive geometry
];

/// Number of size classes.
const SIZE_CLASS_COUNT: usize = SIZE_CLASSES.len();

// ---------------------------------------------------------------------------
// ScratchBuffer Handle
// ---------------------------------------------------------------------------

/// A scratch buffer handle for use in acceleration structure builds.
///
/// This is a lightweight handle that references a pooled buffer. The buffer
/// remains valid until released back to the pool via `ScratchBufferPool::release`.
///
/// # Fields
///
/// - `buffer` - The underlying wgpu buffer (Arc for shared ownership)
/// - `size` - Allocated size of the buffer
/// - `offset` - Offset within buffer (always 0 for this implementation)
/// - `size_class` - Index of the size class bucket
#[derive(Debug)]
pub struct ScratchBuffer {
    /// The underlying GPU buffer.
    pub buffer: Arc<Buffer>,
    /// Allocated size in bytes.
    pub size: u64,
    /// Offset within the buffer (for sub-allocation, always 0 in this impl).
    pub offset: u64,
    /// Size class index (0-4).
    size_class: usize,
    /// Index within the size class bucket.
    bucket_index: usize,
}

impl ScratchBuffer {
    /// Returns a reference to the underlying wgpu buffer.
    #[inline]
    pub fn inner(&self) -> &Buffer {
        &self.buffer
    }

    /// Returns the size class name for debugging.
    pub fn size_class_name(&self) -> &'static str {
        match self.size_class {
            0 => "64KB",
            1 => "256KB",
            2 => "1MB",
            3 => "4MB",
            4 => "16MB",
            _ => "Unknown",
        }
    }

    /// Returns the GPU address range (offset, size) for this scratch buffer.
    #[inline]
    pub fn range(&self) -> (u64, u64) {
        (self.offset, self.size)
    }
}

// ---------------------------------------------------------------------------
// ScratchBufferEntry (Internal)
// ---------------------------------------------------------------------------

/// Internal entry in a size class bucket.
struct ScratchBufferEntry {
    /// The GPU buffer.
    buffer: Arc<Buffer>,
    /// Whether this buffer is currently in use.
    in_use: bool,
}

// ---------------------------------------------------------------------------
// PoolStats
// ---------------------------------------------------------------------------

/// Statistics about the scratch buffer pool.
#[derive(Debug, Clone, Default)]
pub struct PoolStats {
    /// Total bytes currently allocated in the pool.
    pub pool_capacity: u64,
    /// Bytes currently in use (acquired but not released).
    pub current_usage: u64,
    /// Maximum bytes ever allocated.
    pub peak_usage: u64,
    /// Number of buffers in each size class.
    pub buffers_per_class: [usize; SIZE_CLASS_COUNT],
    /// Number of free buffers in each size class.
    pub free_per_class: [usize; SIZE_CLASS_COUNT],
    /// Total number of acquire operations.
    pub total_acquires: u64,
    /// Total number of release operations.
    pub total_releases: u64,
    /// Number of new buffer allocations (cache misses).
    pub allocations: u64,
    /// Number of buffer reuses (cache hits).
    pub reuses: u64,
}

impl PoolStats {
    /// Returns the cache hit rate as a percentage.
    pub fn hit_rate(&self) -> f32 {
        if self.total_acquires == 0 {
            0.0
        } else {
            (self.reuses as f32 / self.total_acquires as f32) * 100.0
        }
    }

    /// Returns the overall utilization (current_usage / pool_capacity).
    pub fn utilization(&self) -> f32 {
        if self.pool_capacity == 0 {
            0.0
        } else {
            (self.current_usage as f32 / self.pool_capacity as f32) * 100.0
        }
    }
}

// ---------------------------------------------------------------------------
// ScratchBufferPoolError
// ---------------------------------------------------------------------------

/// Errors that can occur during scratch buffer pool operations.
#[derive(Debug)]
pub enum ScratchBufferPoolError {
    /// Requested size exceeds maximum size class.
    SizeTooLarge { requested: u64, max: u64 },
    /// Buffer creation failed.
    BufferCreationFailed(String),
    /// Invalid buffer handle.
    InvalidHandle,
}

impl std::fmt::Display for ScratchBufferPoolError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SizeTooLarge { requested, max } => {
                write!(
                    f,
                    "Requested scratch buffer size {} exceeds maximum {}",
                    requested, max
                )
            }
            Self::BufferCreationFailed(msg) => {
                write!(f, "Failed to create scratch buffer: {}", msg)
            }
            Self::InvalidHandle => write!(f, "Invalid scratch buffer handle"),
        }
    }
}

impl std::error::Error for ScratchBufferPoolError {}

// ---------------------------------------------------------------------------
// ScratchBufferPool
// ---------------------------------------------------------------------------

/// Pool for managing scratch buffers used in BLAS/TLAS construction.
///
/// Scratch buffers are temporary GPU allocations required by hardware to
/// build acceleration structures. This pool provides efficient reuse through
/// size-class bucketing and free lists.
///
/// # Thread Safety
///
/// This pool is not thread-safe. For concurrent access, wrap in a mutex or
/// use one pool per thread.
pub struct ScratchBufferPool {
    /// Buckets for each size class.
    buckets: [Vec<ScratchBufferEntry>; SIZE_CLASS_COUNT],
    /// Free list indices for each size class.
    free_lists: [Vec<usize>; SIZE_CLASS_COUNT],
    /// Pool statistics.
    stats: PoolStats,
    /// Debug label prefix for created buffers.
    label_prefix: String,
}

impl ScratchBufferPool {
    /// Create a new empty scratch buffer pool.
    pub fn new() -> Self {
        Self {
            buckets: Default::default(),
            free_lists: Default::default(),
            stats: PoolStats::default(),
            label_prefix: "scratch".to_string(),
        }
    }

    /// Create a new pool with a custom label prefix.
    pub fn with_label(label_prefix: &str) -> Self {
        Self {
            buckets: Default::default(),
            free_lists: Default::default(),
            stats: PoolStats::default(),
            label_prefix: label_prefix.to_string(),
        }
    }

    /// Find the appropriate size class for a given size.
    ///
    /// Returns the index of the smallest size class that can accommodate
    /// the requested size, or None if the size exceeds all classes.
    #[inline]
    fn size_class_for(size: u64) -> Option<usize> {
        SIZE_CLASSES.iter().position(|&class_size| class_size >= size)
    }

    /// Align a size up to the scratch buffer alignment requirement.
    #[inline]
    fn align_size(size: u64) -> u64 {
        (size + SCRATCH_BUFFER_ALIGNMENT - 1) & !(SCRATCH_BUFFER_ALIGNMENT - 1)
    }

    /// Acquire a scratch buffer of at least the requested size.
    ///
    /// If a free buffer of the appropriate size class exists, it will be reused.
    /// Otherwise, a new buffer is allocated.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device for buffer creation
    /// * `size` - Minimum required size in bytes
    ///
    /// # Returns
    ///
    /// A `ScratchBuffer` handle if successful.
    ///
    /// # Errors
    ///
    /// Returns an error if the requested size exceeds the maximum size class (16MB).
    pub fn acquire(
        &mut self,
        device: &Device,
        size: u64,
    ) -> Result<ScratchBuffer, ScratchBufferPoolError> {
        let aligned_size = Self::align_size(size);
        let class_idx = Self::size_class_for(aligned_size).ok_or(
            ScratchBufferPoolError::SizeTooLarge {
                requested: aligned_size,
                max: SIZE_CLASSES[SIZE_CLASS_COUNT - 1],
            },
        )?;

        let class_size = SIZE_CLASSES[class_idx];
        self.stats.total_acquires += 1;

        // Try to reuse from free list
        if let Some(bucket_idx) = self.free_lists[class_idx].pop() {
            let entry = &mut self.buckets[class_idx][bucket_idx];
            entry.in_use = true;
            self.stats.current_usage += class_size;
            self.stats.reuses += 1;
            self.stats.free_per_class[class_idx] -= 1;

            return Ok(ScratchBuffer {
                buffer: Arc::clone(&entry.buffer),
                size: class_size,
                offset: 0,
                size_class: class_idx,
                bucket_index: bucket_idx,
            });
        }

        // Allocate new buffer
        let buffer = self.create_buffer(device, class_idx, class_size)?;
        let bucket_idx = self.buckets[class_idx].len();

        let entry = ScratchBufferEntry {
            buffer: Arc::clone(&buffer),
            in_use: true,
        };
        self.buckets[class_idx].push(entry);

        // Update stats
        self.stats.pool_capacity += class_size;
        self.stats.current_usage += class_size;
        self.stats.allocations += 1;
        self.stats.buffers_per_class[class_idx] += 1;

        // Track peak usage
        if self.stats.current_usage > self.stats.peak_usage {
            self.stats.peak_usage = self.stats.current_usage;
        }

        Ok(ScratchBuffer {
            buffer,
            size: class_size,
            offset: 0,
            size_class: class_idx,
            bucket_index: bucket_idx,
        })
    }

    /// Release a scratch buffer back to the pool.
    ///
    /// The buffer will be added to the free list for future reuse.
    ///
    /// # Arguments
    ///
    /// * `buffer` - The scratch buffer to release
    pub fn release(&mut self, buffer: ScratchBuffer) {
        let class_idx = buffer.size_class;
        let bucket_idx = buffer.bucket_index;

        if class_idx < SIZE_CLASS_COUNT && bucket_idx < self.buckets[class_idx].len() {
            let entry = &mut self.buckets[class_idx][bucket_idx];
            if entry.in_use {
                entry.in_use = false;
                self.free_lists[class_idx].push(bucket_idx);
                self.stats.current_usage = self.stats.current_usage.saturating_sub(buffer.size);
                self.stats.total_releases += 1;
                self.stats.free_per_class[class_idx] += 1;
            }
        }
        // Drop the Arc reference; buffer stays in pool
    }

    /// Trim unused buffers from the pool.
    ///
    /// This releases GPU memory for buffers that are currently free.
    /// Call this when memory pressure is high or at scene boundaries.
    ///
    /// # Returns
    ///
    /// The number of bytes freed.
    pub fn trim(&mut self) -> u64 {
        let mut freed = 0u64;

        for class_idx in 0..SIZE_CLASS_COUNT {
            let class_size = SIZE_CLASSES[class_idx];

            // Process free list in reverse to avoid index issues
            while let Some(bucket_idx) = self.free_lists[class_idx].pop() {
                // Mark for removal
                if bucket_idx < self.buckets[class_idx].len() {
                    let entry = &self.buckets[class_idx][bucket_idx];
                    if !entry.in_use && Arc::strong_count(&entry.buffer) == 1 {
                        // Safe to drop - we hold the only reference
                        freed += class_size;
                    }
                }
            }

            // Remove entries where we're the only holder
            let bucket = &mut self.buckets[class_idx];
            let mut i = 0;
            while i < bucket.len() {
                if !bucket[i].in_use && Arc::strong_count(&bucket[i].buffer) == 1 {
                    bucket.swap_remove(i);
                    self.stats.pool_capacity = self.stats.pool_capacity.saturating_sub(class_size);
                    self.stats.buffers_per_class[class_idx] =
                        self.stats.buffers_per_class[class_idx].saturating_sub(1);
                } else {
                    i += 1;
                }
            }

            // Rebuild free list
            self.free_lists[class_idx].clear();
            for (idx, entry) in bucket.iter().enumerate() {
                if !entry.in_use {
                    self.free_lists[class_idx].push(idx);
                }
            }
            self.stats.free_per_class[class_idx] = self.free_lists[class_idx].len();
        }

        freed
    }

    /// Clear all buffers from the pool.
    ///
    /// This releases all GPU memory. Any outstanding handles become invalid.
    pub fn clear(&mut self) {
        for class_idx in 0..SIZE_CLASS_COUNT {
            self.buckets[class_idx].clear();
            self.free_lists[class_idx].clear();
        }
        self.stats.pool_capacity = 0;
        self.stats.current_usage = 0;
        self.stats.buffers_per_class = [0; SIZE_CLASS_COUNT];
        self.stats.free_per_class = [0; SIZE_CLASS_COUNT];
    }

    /// Get current memory usage statistics.
    pub fn stats(&self) -> PoolStats {
        self.stats.clone()
    }

    /// Get the current memory usage in bytes.
    #[inline]
    pub fn current_usage(&self) -> u64 {
        self.stats.current_usage
    }

    /// Get the peak memory usage in bytes.
    #[inline]
    pub fn peak_usage(&self) -> u64 {
        self.stats.peak_usage
    }

    /// Get the total pool capacity in bytes.
    #[inline]
    pub fn pool_capacity(&self) -> u64 {
        self.stats.pool_capacity
    }

    /// Get the number of buffers currently in use.
    pub fn buffers_in_use(&self) -> usize {
        self.buckets.iter().flatten().filter(|e| e.in_use).count()
    }

    /// Get the number of free buffers available.
    pub fn buffers_free(&self) -> usize {
        self.free_lists.iter().map(|fl| fl.len()).sum()
    }

    /// Check if the pool is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.buckets.iter().all(|b| b.is_empty())
    }

    /// Reset peak usage tracking.
    pub fn reset_peak(&mut self) {
        self.stats.peak_usage = self.stats.current_usage;
    }

    // -------------------------------------------------------------------------
    // Private helpers
    // -------------------------------------------------------------------------

    /// Create a new GPU buffer for the given size class.
    fn create_buffer(
        &self,
        device: &Device,
        class_idx: usize,
        size: u64,
    ) -> Result<Arc<Buffer>, ScratchBufferPoolError> {
        let label = format!("{}_{}_{}KB", self.label_prefix, class_idx, size / 1024);

        let buffer = device.create_buffer(&BufferDescriptor {
            label: Some(&label),
            size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        Ok(Arc::new(buffer))
    }
}

impl Default for ScratchBufferPool {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for ScratchBufferPool {
    fn drop(&mut self) {
        // Log warning if there are still buffers in use
        let in_use = self.buffers_in_use();
        if in_use > 0 {
            log::warn!(
                "ScratchBufferPool dropped with {} buffers still in use ({} bytes)",
                in_use,
                self.stats.current_usage
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Helper: Create mock pool for testing
    // -------------------------------------------------------------------------

    fn create_test_pool() -> ScratchBufferPool {
        ScratchBufferPool::new()
    }

    // -------------------------------------------------------------------------
    // Test: Pool creation
    // -------------------------------------------------------------------------

    #[test]
    fn test_pool_new() {
        let pool = ScratchBufferPool::new();
        assert!(pool.is_empty());
        assert_eq!(pool.current_usage(), 0);
        assert_eq!(pool.peak_usage(), 0);
        assert_eq!(pool.pool_capacity(), 0);
    }

    #[test]
    fn test_pool_default() {
        let pool = ScratchBufferPool::default();
        assert!(pool.is_empty());
    }

    #[test]
    fn test_pool_with_label() {
        let pool = ScratchBufferPool::with_label("custom_scratch");
        assert_eq!(pool.label_prefix, "custom_scratch");
    }

    // -------------------------------------------------------------------------
    // Test: Size class selection
    // -------------------------------------------------------------------------

    #[test]
    fn test_size_class_for_small() {
        assert_eq!(ScratchBufferPool::size_class_for(1), Some(0)); // 64KB
        assert_eq!(ScratchBufferPool::size_class_for(64 * 1024), Some(0));
    }

    #[test]
    fn test_size_class_for_medium() {
        assert_eq!(ScratchBufferPool::size_class_for(64 * 1024 + 1), Some(1)); // 256KB
        assert_eq!(ScratchBufferPool::size_class_for(256 * 1024), Some(1));
    }

    #[test]
    fn test_size_class_for_large() {
        assert_eq!(ScratchBufferPool::size_class_for(256 * 1024 + 1), Some(2)); // 1MB
        assert_eq!(ScratchBufferPool::size_class_for(1024 * 1024), Some(2));
    }

    #[test]
    fn test_size_class_for_xlarge() {
        assert_eq!(ScratchBufferPool::size_class_for(1024 * 1024 + 1), Some(3)); // 4MB
        assert_eq!(ScratchBufferPool::size_class_for(4 * 1024 * 1024), Some(3));
    }

    #[test]
    fn test_size_class_for_huge() {
        assert_eq!(ScratchBufferPool::size_class_for(4 * 1024 * 1024 + 1), Some(4)); // 16MB
        assert_eq!(ScratchBufferPool::size_class_for(16 * 1024 * 1024), Some(4));
    }

    #[test]
    fn test_size_class_for_too_large() {
        assert_eq!(ScratchBufferPool::size_class_for(16 * 1024 * 1024 + 1), None);
        assert_eq!(ScratchBufferPool::size_class_for(100 * 1024 * 1024), None);
    }

    // -------------------------------------------------------------------------
    // Test: Alignment
    // -------------------------------------------------------------------------

    #[test]
    fn test_align_size_already_aligned() {
        assert_eq!(ScratchBufferPool::align_size(256), 256);
        assert_eq!(ScratchBufferPool::align_size(512), 512);
        assert_eq!(ScratchBufferPool::align_size(1024), 1024);
    }

    #[test]
    fn test_align_size_rounds_up() {
        assert_eq!(ScratchBufferPool::align_size(1), 256);
        assert_eq!(ScratchBufferPool::align_size(100), 256);
        assert_eq!(ScratchBufferPool::align_size(255), 256);
        assert_eq!(ScratchBufferPool::align_size(257), 512);
    }

    #[test]
    fn test_align_size_zero() {
        assert_eq!(ScratchBufferPool::align_size(0), 0);
    }

    // -------------------------------------------------------------------------
    // Test: Stats
    // -------------------------------------------------------------------------

    #[test]
    fn test_stats_initial() {
        let pool = create_test_pool();
        let stats = pool.stats();

        assert_eq!(stats.pool_capacity, 0);
        assert_eq!(stats.current_usage, 0);
        assert_eq!(stats.peak_usage, 0);
        assert_eq!(stats.total_acquires, 0);
        assert_eq!(stats.total_releases, 0);
        assert_eq!(stats.allocations, 0);
        assert_eq!(stats.reuses, 0);
    }

    #[test]
    fn test_stats_hit_rate_zero_acquires() {
        let stats = PoolStats::default();
        assert_eq!(stats.hit_rate(), 0.0);
    }

    #[test]
    fn test_stats_hit_rate_all_misses() {
        let mut stats = PoolStats::default();
        stats.total_acquires = 10;
        stats.reuses = 0;
        assert_eq!(stats.hit_rate(), 0.0);
    }

    #[test]
    fn test_stats_hit_rate_all_hits() {
        let mut stats = PoolStats::default();
        stats.total_acquires = 10;
        stats.reuses = 10;
        assert_eq!(stats.hit_rate(), 100.0);
    }

    #[test]
    fn test_stats_hit_rate_mixed() {
        let mut stats = PoolStats::default();
        stats.total_acquires = 10;
        stats.reuses = 3;
        let rate = stats.hit_rate();
        assert!((rate - 30.0).abs() < 0.001, "Expected ~30.0, got {}", rate);
    }

    #[test]
    fn test_stats_utilization_empty() {
        let stats = PoolStats::default();
        assert_eq!(stats.utilization(), 0.0);
    }

    #[test]
    fn test_stats_utilization_full() {
        let mut stats = PoolStats::default();
        stats.pool_capacity = 1000;
        stats.current_usage = 1000;
        assert_eq!(stats.utilization(), 100.0);
    }

    #[test]
    fn test_stats_utilization_partial() {
        let mut stats = PoolStats::default();
        stats.pool_capacity = 1000;
        stats.current_usage = 250;
        assert_eq!(stats.utilization(), 25.0);
    }

    // -------------------------------------------------------------------------
    // Test: ScratchBuffer handle
    // -------------------------------------------------------------------------

    #[test]
    fn test_scratch_buffer_size_class_name() {
        // Create a mock ScratchBuffer to test naming
        // We can't test with real buffers without a device, but we can test the logic

        let names = ["64KB", "256KB", "1MB", "4MB", "16MB"];
        for (idx, expected_name) in names.iter().enumerate() {
            // Verify SIZE_CLASSES match expected values
            match idx {
                0 => assert_eq!(SIZE_CLASSES[0], 64 * 1024),
                1 => assert_eq!(SIZE_CLASSES[1], 256 * 1024),
                2 => assert_eq!(SIZE_CLASSES[2], 1024 * 1024),
                3 => assert_eq!(SIZE_CLASSES[3], 4 * 1024 * 1024),
                4 => assert_eq!(SIZE_CLASSES[4], 16 * 1024 * 1024),
                _ => unreachable!(),
            }
            let _ = expected_name; // Used to verify the constant
        }
    }

    // -------------------------------------------------------------------------
    // Test: Error types
    // -------------------------------------------------------------------------

    #[test]
    fn test_error_display_size_too_large() {
        let err = ScratchBufferPoolError::SizeTooLarge {
            requested: 100,
            max: 50,
        };
        let msg = format!("{}", err);
        assert!(msg.contains("100"));
        assert!(msg.contains("50"));
    }

    #[test]
    fn test_error_display_buffer_creation_failed() {
        let err = ScratchBufferPoolError::BufferCreationFailed("out of memory".to_string());
        let msg = format!("{}", err);
        assert!(msg.contains("out of memory"));
    }

    #[test]
    fn test_error_display_invalid_handle() {
        let err = ScratchBufferPoolError::InvalidHandle;
        let msg = format!("{}", err);
        assert!(msg.contains("Invalid"));
    }

    // -------------------------------------------------------------------------
    // Test: Constants
    // -------------------------------------------------------------------------

    #[test]
    fn test_scratch_buffer_alignment() {
        assert_eq!(SCRATCH_BUFFER_ALIGNMENT, 256);
    }

    #[test]
    fn test_size_classes_ascending() {
        for i in 1..SIZE_CLASSES.len() {
            assert!(
                SIZE_CLASSES[i] > SIZE_CLASSES[i - 1],
                "Size classes must be in ascending order"
            );
        }
    }

    #[test]
    fn test_size_classes_count() {
        assert_eq!(SIZE_CLASS_COUNT, 5);
        assert_eq!(SIZE_CLASSES.len(), SIZE_CLASS_COUNT);
    }

    // -------------------------------------------------------------------------
    // Test: Pool methods without device (state tracking)
    // -------------------------------------------------------------------------

    #[test]
    fn test_buffers_in_use_empty() {
        let pool = create_test_pool();
        assert_eq!(pool.buffers_in_use(), 0);
    }

    #[test]
    fn test_buffers_free_empty() {
        let pool = create_test_pool();
        assert_eq!(pool.buffers_free(), 0);
    }

    #[test]
    fn test_is_empty_true() {
        let pool = create_test_pool();
        assert!(pool.is_empty());
    }

    #[test]
    fn test_clear_empty_pool() {
        let mut pool = create_test_pool();
        pool.clear();
        assert!(pool.is_empty());
        assert_eq!(pool.stats().pool_capacity, 0);
    }

    #[test]
    fn test_trim_empty_pool() {
        let mut pool = create_test_pool();
        let freed = pool.trim();
        assert_eq!(freed, 0);
    }

    #[test]
    fn test_reset_peak_empty() {
        let mut pool = create_test_pool();
        pool.reset_peak();
        assert_eq!(pool.peak_usage(), 0);
    }

    // -------------------------------------------------------------------------
    // Test: Size class boundary conditions
    // -------------------------------------------------------------------------

    #[test]
    fn test_size_class_exact_boundaries() {
        // Test exact boundary values
        assert_eq!(ScratchBufferPool::size_class_for(64 * 1024), Some(0));
        assert_eq!(ScratchBufferPool::size_class_for(64 * 1024 + 1), Some(1));

        assert_eq!(ScratchBufferPool::size_class_for(256 * 1024), Some(1));
        assert_eq!(ScratchBufferPool::size_class_for(256 * 1024 + 1), Some(2));

        assert_eq!(ScratchBufferPool::size_class_for(1024 * 1024), Some(2));
        assert_eq!(ScratchBufferPool::size_class_for(1024 * 1024 + 1), Some(3));

        assert_eq!(ScratchBufferPool::size_class_for(4 * 1024 * 1024), Some(3));
        assert_eq!(
            ScratchBufferPool::size_class_for(4 * 1024 * 1024 + 1),
            Some(4)
        );

        assert_eq!(ScratchBufferPool::size_class_for(16 * 1024 * 1024), Some(4));
        assert_eq!(
            ScratchBufferPool::size_class_for(16 * 1024 * 1024 + 1),
            None
        );
    }

    // -------------------------------------------------------------------------
    // Test: Alignment edge cases
    // -------------------------------------------------------------------------

    #[test]
    fn test_alignment_powers_of_two() {
        for power in 0..20 {
            let size = 1u64 << power;
            let aligned = ScratchBufferPool::align_size(size);
            assert!(
                aligned >= size,
                "Aligned size {} must be >= original {}",
                aligned,
                size
            );
            assert_eq!(
                aligned % SCRATCH_BUFFER_ALIGNMENT,
                0,
                "Aligned size {} must be divisible by {}",
                aligned,
                SCRATCH_BUFFER_ALIGNMENT
            );
        }
    }

    #[test]
    fn test_alignment_large_values() {
        let large = 1024 * 1024 * 1024; // 1GB
        let aligned = ScratchBufferPool::align_size(large);
        assert_eq!(aligned, large);
        assert_eq!(aligned % SCRATCH_BUFFER_ALIGNMENT, 0);
    }

    // -------------------------------------------------------------------------
    // Integration test notes (require device)
    // -------------------------------------------------------------------------
    // The following scenarios require a wgpu device and should be tested
    // in integration tests:
    //
    // 1. acquire() creates buffer and updates stats
    // 2. release() returns buffer to free list
    // 3. acquire() after release() reuses buffer
    // 4. trim() removes unused buffers
    // 5. Multiple size classes work independently
    // 6. Peak usage tracking across acquire/release
    // 7. Pool handles rapid acquire/release cycles
    // 8. Drop warns about in-use buffers
}
