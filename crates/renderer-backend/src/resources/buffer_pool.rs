//! Buffer pool with size classes for efficient GPU memory management.
//!
//! This module implements a hierarchical buffer allocation strategy that reduces
//! driver allocation overhead by pooling buffers into size classes and reusing
//! them via free lists.
//!
//! # Architecture
//!
//! ```text
//! BufferPool
//! ├── pools: HashMap<SizeClass, Vec<PooledBuffer>>
//! ├── free_lists: HashMap<SizeClass, Vec<BufferHandle>>
//! ├── growth_policy: GrowthPolicy
//! ├── shrink_threshold: f32
//! └── metrics: PoolMetrics
//! ```
//!
//! # Size Classes
//!
//! Buffers are allocated in fixed size classes to enable efficient reuse:
//!
//! | Class | Size | Use Case |
//! |-------|------|----------|
//! | Tiny | 256B | Small uniforms, constants |
//! | Small | 1KB | Per-object data |
//! | Medium | 4KB | Material data, small meshes |
//! | Large | 16KB | Medium meshes, instance data |
//! | XLarge | 64KB | Large meshes |
//! | Huge | 256KB | Very large meshes, terrain patches |
//! | Massive | 1MB | Massive data, particle systems |
//!
//! # Growth Policy
//!
//! When a size class is exhausted, the pool doubles the number of buffers
//! in that class (up to a maximum). This amortizes allocation cost over time.
//!
//! # Shrink Policy
//!
//! When more than 50% of buffers in a class are free, the pool may release
//! excess buffers to reduce memory usage. This is optional and configurable.
//!
//! # Example
//!
//! ```no_run
//! use renderer_backend::resources::buffer_pool::{BufferPool, PoolConfig};
//! use wgpu::BufferUsages;
//!
//! # fn example(device: &wgpu::Device) {
//! // Create pool with default config
//! let mut pool = BufferPool::new(device, PoolConfig::default());
//!
//! // Acquire a buffer for 500 bytes of vertex data
//! let handle = pool.acquire(500, BufferUsages::VERTEX | BufferUsages::COPY_DST)
//!     .expect("size within pool limits");
//!
//! // Get the buffer reference
//! let buffer = pool.get_buffer(&handle).expect("valid handle");
//! println!("Got buffer of size {} (class: {:?})",
//!     handle.size_class().size_bytes(), handle.size_class());
//!
//! // Use the buffer...
//!
//! // Release back to pool for reuse
//! pool.release(handle);
//!
//! // Check pool metrics
//! let metrics = pool.metrics();
//! println!("Total allocated: {} bytes", metrics.total_bytes_allocated);
//! println!("Total in use: {} buffers", metrics.total_in_use);
//! # }
//! ```

use log::{debug, trace, warn};
use std::collections::HashMap;
use std::sync::Arc;
use wgpu::{Buffer, BufferDescriptor, BufferUsages, Device};

use super::buffer::TrinityBuffer;

// ============================================================================
// Size Classes
// ============================================================================

/// Buffer size classes for pooling.
///
/// Each size class represents a fixed buffer size. When acquiring a buffer,
/// the pool selects the smallest class that fits the requested size.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(u64)]
pub enum SizeClass {
    /// 256 bytes - small uniforms, constants
    Tiny = 256,
    /// 1 KB - per-object data
    Small = 1024,
    /// 4 KB - material data, small meshes
    Medium = 4096,
    /// 16 KB - medium meshes, instance data
    Large = 16384,
    /// 64 KB - large meshes
    XLarge = 65536,
    /// 256 KB - very large meshes, terrain patches
    Huge = 262144,
    /// 1 MB - massive data, particle systems
    Massive = 1048576,
}

impl SizeClass {
    /// All size classes in ascending order.
    pub const ALL: [SizeClass; 7] = [
        SizeClass::Tiny,
        SizeClass::Small,
        SizeClass::Medium,
        SizeClass::Large,
        SizeClass::XLarge,
        SizeClass::Huge,
        SizeClass::Massive,
    ];

    /// Returns the size in bytes for this class.
    #[inline]
    pub const fn size_bytes(self) -> u64 {
        self as u64
    }

    /// Finds the appropriate size class for a given size.
    ///
    /// Returns `Some(class)` if a suitable class exists, or `None` if the
    /// requested size exceeds the largest class (1MB).
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::resources::buffer_pool::SizeClass;
    ///
    /// assert_eq!(SizeClass::for_size(100), Some(SizeClass::Tiny));
    /// assert_eq!(SizeClass::for_size(256), Some(SizeClass::Tiny));
    /// assert_eq!(SizeClass::for_size(257), Some(SizeClass::Small));
    /// assert_eq!(SizeClass::for_size(1024 * 1024), Some(SizeClass::Massive));
    /// assert_eq!(SizeClass::for_size(1024 * 1024 + 1), None); // Too large
    /// ```
    pub const fn for_size(size: u64) -> Option<SizeClass> {
        if size <= SizeClass::Tiny as u64 {
            Some(SizeClass::Tiny)
        } else if size <= SizeClass::Small as u64 {
            Some(SizeClass::Small)
        } else if size <= SizeClass::Medium as u64 {
            Some(SizeClass::Medium)
        } else if size <= SizeClass::Large as u64 {
            Some(SizeClass::Large)
        } else if size <= SizeClass::XLarge as u64 {
            Some(SizeClass::XLarge)
        } else if size <= SizeClass::Huge as u64 {
            Some(SizeClass::Huge)
        } else if size <= SizeClass::Massive as u64 {
            Some(SizeClass::Massive)
        } else {
            None
        }
    }

    /// Returns the next larger size class, or None if this is the largest.
    pub const fn next_larger(self) -> Option<SizeClass> {
        match self {
            SizeClass::Tiny => Some(SizeClass::Small),
            SizeClass::Small => Some(SizeClass::Medium),
            SizeClass::Medium => Some(SizeClass::Large),
            SizeClass::Large => Some(SizeClass::XLarge),
            SizeClass::XLarge => Some(SizeClass::Huge),
            SizeClass::Huge => Some(SizeClass::Massive),
            SizeClass::Massive => None,
        }
    }
}

impl std::fmt::Display for SizeClass {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SizeClass::Tiny => write!(f, "Tiny (256B)"),
            SizeClass::Small => write!(f, "Small (1KB)"),
            SizeClass::Medium => write!(f, "Medium (4KB)"),
            SizeClass::Large => write!(f, "Large (16KB)"),
            SizeClass::XLarge => write!(f, "XLarge (64KB)"),
            SizeClass::Huge => write!(f, "Huge (256KB)"),
            SizeClass::Massive => write!(f, "Massive (1MB)"),
        }
    }
}

// ============================================================================
// Buffer Handle
// ============================================================================

/// Unique identifier for a pooled buffer.
///
/// This handle is returned by [`BufferPool::acquire`] and must be used to
/// access the buffer via [`BufferPool::get_buffer`] or released via
/// [`BufferPool::release`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BufferHandle {
    /// Size class this buffer belongs to.
    class: SizeClass,
    /// Index within the size class pool.
    index: usize,
    /// Generation counter for validation.
    generation: u64,
}

impl BufferHandle {
    /// Returns the size class of this handle.
    #[inline]
    pub fn size_class(&self) -> SizeClass {
        self.class
    }

    /// Returns the buffer size in bytes.
    #[inline]
    pub fn size(&self) -> u64 {
        self.class.size_bytes()
    }
}

// ============================================================================
// Pooled Buffer (Internal)
// ============================================================================

/// A buffer acquired from the pool.
///
/// This is returned by [`BufferPool::acquire_buffer`] and owns the buffer
/// temporarily. Use [`BufferPool::release_buffer`] or drop to return it.
pub struct PooledBuffer {
    /// The underlying wgpu buffer wrapped in Arc for sharing.
    inner: Arc<Buffer>,
    /// Size class of this buffer.
    class: SizeClass,
    /// Handle for returning to pool.
    handle: BufferHandle,
    /// Buffer usage flags.
    usage: BufferUsages,
    /// Optional debug label.
    label: Option<String>,
}

impl PooledBuffer {
    /// Returns the underlying wgpu buffer.
    #[inline]
    pub fn inner(&self) -> &Buffer {
        &self.inner
    }

    /// Returns a clone of the Arc-wrapped buffer.
    #[inline]
    pub fn buffer_arc(&self) -> Arc<Buffer> {
        Arc::clone(&self.inner)
    }

    /// Returns the buffer size in bytes.
    #[inline]
    pub fn size(&self) -> u64 {
        self.class.size_bytes()
    }

    /// Returns the size class.
    #[inline]
    pub fn size_class(&self) -> SizeClass {
        self.class
    }

    /// Returns the buffer handle for manual release.
    #[inline]
    pub fn handle(&self) -> BufferHandle {
        self.handle
    }

    /// Returns the buffer usage flags.
    #[inline]
    pub fn usage(&self) -> BufferUsages {
        self.usage
    }

    /// Returns the debug label, if any.
    #[inline]
    pub fn label(&self) -> Option<&str> {
        self.label.as_deref()
    }

    /// Converts to a TrinityBuffer.
    ///
    /// Note: This extracts the inner buffer if possible, otherwise panics
    /// if there are other Arc references.
    ///
    /// Use this when you need to keep the buffer beyond the pool's lifetime
    /// or integrate with APIs expecting TrinityBuffer.
    pub fn into_trinity_buffer(self) -> TrinityBuffer {
        let buffer = Arc::try_unwrap(self.inner)
            .expect("Cannot convert to TrinityBuffer: buffer has other references");
        TrinityBuffer::from_raw(
            buffer,
            self.class.size_bytes(),
            self.usage,
            self.label,
        )
    }
}

impl std::fmt::Debug for PooledBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PooledBuffer")
            .field("class", &self.class)
            .field("handle", &self.handle)
            .field("usage", &self.usage)
            .field("label", &self.label)
            .finish_non_exhaustive()
    }
}

// ============================================================================
// Pool Entry
// ============================================================================

/// Internal representation of a buffer in the pool.
struct PoolEntry {
    /// The wgpu buffer wrapped in Arc.
    buffer: Arc<Buffer>,
    /// Buffer usage flags.
    usage: BufferUsages,
    /// Current generation (incremented on each acquire).
    generation: u64,
    /// Whether this entry is currently in use.
    in_use: bool,
    /// Optional label.
    label: Option<String>,
}

// ============================================================================
// Class Stats
// ============================================================================

/// Statistics for a single size class.
#[derive(Debug, Clone, Default)]
pub struct ClassStats {
    /// Total buffers allocated in this class.
    pub total_allocated: usize,
    /// Buffers currently in use.
    pub in_use: usize,
    /// Buffers available in free list.
    pub free: usize,
    /// Total bytes allocated for this class.
    pub total_bytes: u64,
    /// Number of times this class was grown.
    pub growth_count: u64,
    /// Number of times acquire was called for this class.
    pub acquire_count: u64,
    /// Number of times a buffer was reused from free list.
    pub reuse_count: u64,
}

impl ClassStats {
    /// Returns the utilization ratio (in_use / total_allocated).
    pub fn utilization(&self) -> f32 {
        if self.total_allocated == 0 {
            0.0
        } else {
            self.in_use as f32 / self.total_allocated as f32
        }
    }

    /// Returns the reuse ratio (reuse_count / acquire_count).
    pub fn reuse_ratio(&self) -> f32 {
        if self.acquire_count == 0 {
            0.0
        } else {
            self.reuse_count as f32 / self.acquire_count as f32
        }
    }
}

// ============================================================================
// Pool Metrics
// ============================================================================

/// Aggregated metrics for the entire buffer pool.
#[derive(Debug, Clone, Default)]
pub struct PoolMetrics {
    /// Total buffers allocated across all classes.
    pub total_allocated: usize,
    /// Total buffers currently in use.
    pub total_in_use: usize,
    /// Total buffers free across all classes.
    pub total_free: usize,
    /// Total bytes allocated.
    pub total_bytes_allocated: u64,
    /// Total bytes currently in use.
    pub total_bytes_in_use: u64,
    /// Per-class statistics.
    pub per_class_stats: HashMap<SizeClass, ClassStats>,
    /// Number of oversized allocations (bypassing pool).
    pub oversized_allocations: u64,
}

impl PoolMetrics {
    /// Returns the overall utilization ratio.
    pub fn utilization(&self) -> f32 {
        if self.total_allocated == 0 {
            0.0
        } else {
            self.total_in_use as f32 / self.total_allocated as f32
        }
    }

    /// Returns the memory efficiency (bytes in use / bytes allocated).
    pub fn memory_efficiency(&self) -> f32 {
        if self.total_bytes_allocated == 0 {
            0.0
        } else {
            self.total_bytes_in_use as f32 / self.total_bytes_allocated as f32
        }
    }
}

// ============================================================================
// Growth Policy
// ============================================================================

/// Policy for growing the pool when a size class is exhausted.
#[derive(Debug, Clone)]
pub struct GrowthPolicy {
    /// Initial number of buffers per class.
    pub initial_count: usize,
    /// Growth factor (e.g., 2.0 = double on exhaust).
    pub growth_factor: f32,
    /// Maximum buffers per class.
    pub max_per_class: usize,
}

impl Default for GrowthPolicy {
    fn default() -> Self {
        Self {
            initial_count: 4,
            growth_factor: 2.0,
            max_per_class: 256,
        }
    }
}

impl GrowthPolicy {
    /// Creates a conservative policy with lower limits.
    pub fn conservative() -> Self {
        Self {
            initial_count: 2,
            growth_factor: 1.5,
            max_per_class: 64,
        }
    }

    /// Creates an aggressive policy for high-throughput scenarios.
    pub fn aggressive() -> Self {
        Self {
            initial_count: 8,
            growth_factor: 2.0,
            max_per_class: 512,
        }
    }

    /// Calculates the next allocation count given current count.
    pub fn next_count(&self, current: usize) -> usize {
        let next = ((current as f32) * self.growth_factor).ceil() as usize;
        next.min(self.max_per_class)
    }
}

// ============================================================================
// Pool Configuration
// ============================================================================

/// Configuration for the buffer pool.
#[derive(Debug, Clone)]
pub struct PoolConfig {
    /// Growth policy for exhausted classes.
    pub growth_policy: GrowthPolicy,
    /// Threshold for shrinking (fraction of free buffers).
    /// If free/total > threshold, consider shrinking.
    pub shrink_threshold: f32,
    /// Whether to eagerly pre-allocate initial buffers.
    pub pre_allocate: bool,
    /// Default usage flags for pooled buffers.
    pub default_usage: BufferUsages,
    /// Whether to enable debug labels.
    pub enable_labels: bool,
}

impl Default for PoolConfig {
    fn default() -> Self {
        Self {
            growth_policy: GrowthPolicy::default(),
            shrink_threshold: 0.5,
            pre_allocate: false,
            default_usage: BufferUsages::COPY_DST | BufferUsages::STORAGE,
            enable_labels: cfg!(debug_assertions),
        }
    }
}

// ============================================================================
// Buffer Pool
// ============================================================================

/// Buffer pool with size class allocation.
///
/// This pool manages GPU buffers across fixed size classes, enabling
/// efficient buffer reuse and reducing driver allocation overhead.
///
/// # Thread Safety
///
/// The pool is NOT thread-safe. If concurrent access is needed, wrap
/// it in a Mutex or use separate pools per thread.
///
/// # Example
///
/// ```no_run
/// use renderer_backend::resources::buffer_pool::{BufferPool, PoolConfig};
/// use wgpu::BufferUsages;
///
/// # fn example(device: &wgpu::Device) {
/// let mut pool = BufferPool::new(device, PoolConfig::default());
///
/// // Acquire a handle
/// let handle = pool.acquire(1000, BufferUsages::VERTEX | BufferUsages::COPY_DST)
///     .expect("size within limits");
///
/// // Get buffer reference
/// if let Some(buffer) = pool.get_buffer(&handle) {
///     // Use buffer.inner() with wgpu APIs
/// }
///
/// // Release when done
/// pool.release(handle);
/// # }
/// ```
pub struct BufferPool {
    /// Pools per size class.
    pools: HashMap<SizeClass, Vec<PoolEntry>>,
    /// Free list indices per class.
    free_lists: HashMap<SizeClass, Vec<usize>>,
    /// Pool configuration.
    config: PoolConfig,
    /// Per-class statistics.
    stats: HashMap<SizeClass, ClassStats>,
    /// Count of oversized allocations.
    oversized_count: u64,
    /// Global generation counter.
    global_generation: u64,
}

impl BufferPool {
    /// Creates a new buffer pool with the given configuration.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device (only used if pre_allocate is true)
    /// * `config` - Pool configuration
    ///
    /// # Example
    ///
    /// ```no_run
    /// use renderer_backend::resources::buffer_pool::{BufferPool, PoolConfig};
    ///
    /// # fn example(device: &wgpu::Device) {
    /// let pool = BufferPool::new(device, PoolConfig::default());
    /// # }
    /// ```
    pub fn new(device: &Device, config: PoolConfig) -> Self {
        let mut pool = Self {
            pools: HashMap::new(),
            free_lists: HashMap::new(),
            config,
            stats: HashMap::new(),
            oversized_count: 0,
            global_generation: 0,
        };

        // Initialize empty pools for all size classes
        for class in SizeClass::ALL {
            pool.pools.insert(class, Vec::new());
            pool.free_lists.insert(class, Vec::new());
            pool.stats.insert(class, ClassStats::default());
        }

        // Pre-allocate if configured
        if pool.config.pre_allocate {
            for class in SizeClass::ALL {
                pool.grow_class(device, class);
            }
        }

        debug!("BufferPool created with config: {:?}", pool.config);
        pool
    }

    /// Creates a pool with default configuration.
    pub fn with_defaults(device: &Device) -> Self {
        Self::new(device, PoolConfig::default())
    }

    /// Acquires a buffer handle from the pool.
    ///
    /// Returns a handle to a buffer of at least the requested size. The actual
    /// buffer size will be the next size class that fits the request.
    ///
    /// Use [`get_buffer`] to access the actual buffer.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device
    /// * `size` - Minimum required size in bytes
    /// * `usage` - Buffer usage flags
    ///
    /// # Returns
    ///
    /// A `BufferHandle` if a suitable class exists, or `None` if the
    /// requested size exceeds the largest class (1MB).
    ///
    /// For sizes larger than 1MB, use direct allocation via
    /// [`super::buffer::create_buffer`] instead.
    pub fn acquire(
        &mut self,
        device: &Device,
        size: u64,
        usage: BufferUsages,
    ) -> Option<BufferHandle> {
        // Find appropriate size class
        let class = SizeClass::for_size(size)?;

        // Update stats
        if let Some(stats) = self.stats.get_mut(&class) {
            stats.acquire_count += 1;
        }

        // Try to get from free list
        if let Some(index) = self.free_lists.get_mut(&class).and_then(|list| list.pop()) {
            if let Some(entries) = self.pools.get_mut(&class) {
                if let Some(entry) = entries.get_mut(index) {
                    // Check usage compatibility
                    if entry.usage.contains(usage) {
                        entry.in_use = true;
                        entry.generation += 1;
                        let gen = entry.generation;

                        trace!(
                            "Reused buffer from pool: class={}, index={}, gen={}",
                            class,
                            index,
                            gen
                        );

                        // Update stats
                        if let Some(stats) = self.stats.get_mut(&class) {
                            stats.in_use += 1;
                            stats.free -= 1;
                            stats.reuse_count += 1;
                        }

                        return Some(BufferHandle {
                            class,
                            index,
                            generation: gen,
                        });
                    } else {
                        // Usage mismatch, put back on free list and create new
                        self.free_lists.get_mut(&class).unwrap().push(index);
                        trace!(
                            "Usage mismatch for reused buffer, creating new: required={:?}, have={:?}",
                            usage,
                            entry.usage
                        );
                    }
                }
            }
        }

        // No free buffer available, grow the pool
        self.grow_class_with_usage(device, class, usage);

        // Try again from free list
        if let Some(index) = self.free_lists.get_mut(&class).and_then(|list| list.pop()) {
            if let Some(entries) = self.pools.get_mut(&class) {
                if let Some(entry) = entries.get_mut(index) {
                    entry.in_use = true;
                    entry.generation += 1;
                    let gen = entry.generation;

                    if let Some(stats) = self.stats.get_mut(&class) {
                        stats.in_use += 1;
                        stats.free -= 1;
                    }

                    return Some(BufferHandle {
                        class,
                        index,
                        generation: gen,
                    });
                }
            }
        }

        // This shouldn't happen if grow_class succeeded
        warn!("Failed to acquire buffer after growth for class {}", class);
        None
    }

    /// Acquires a buffer from the pool, returning the full PooledBuffer.
    ///
    /// This is a convenience method that combines `acquire` and `get_buffer_owned`.
    pub fn acquire_buffer(
        &mut self,
        device: &Device,
        size: u64,
        usage: BufferUsages,
    ) -> Option<PooledBuffer> {
        let handle = self.acquire(device, size, usage)?;
        self.get_buffer_owned(&handle)
    }

    /// Gets a reference to the buffer for a handle.
    ///
    /// # Arguments
    ///
    /// * `handle` - A handle returned by `acquire`
    ///
    /// # Returns
    ///
    /// The underlying wgpu buffer, or `None` if the handle is invalid.
    pub fn get_buffer(&self, handle: &BufferHandle) -> Option<&Buffer> {
        let entries = self.pools.get(&handle.class)?;
        let entry = entries.get(handle.index)?;

        // Validate generation
        if entry.generation != handle.generation {
            warn!(
                "Buffer handle generation mismatch: expected {}, got {}",
                entry.generation, handle.generation
            );
            return None;
        }

        if !entry.in_use {
            warn!("Attempted to get buffer that is not in use");
            return None;
        }

        Some(&entry.buffer)
    }

    /// Gets an Arc to the buffer for a handle.
    ///
    /// Useful when you need shared ownership of the buffer.
    pub fn get_buffer_arc(&self, handle: &BufferHandle) -> Option<Arc<Buffer>> {
        let entries = self.pools.get(&handle.class)?;
        let entry = entries.get(handle.index)?;

        if entry.generation != handle.generation || !entry.in_use {
            return None;
        }

        Some(Arc::clone(&entry.buffer))
    }

    /// Gets a PooledBuffer wrapper for a handle.
    ///
    /// This creates an owned wrapper that can be passed around.
    pub fn get_buffer_owned(&self, handle: &BufferHandle) -> Option<PooledBuffer> {
        let entries = self.pools.get(&handle.class)?;
        let entry = entries.get(handle.index)?;

        if entry.generation != handle.generation || !entry.in_use {
            return None;
        }

        Some(PooledBuffer {
            inner: Arc::clone(&entry.buffer),
            class: handle.class,
            handle: *handle,
            usage: entry.usage,
            label: entry.label.clone(),
        })
    }

    /// Releases a buffer back to the pool by handle.
    ///
    /// The buffer becomes available for reuse in future [`acquire`] calls.
    ///
    /// # Arguments
    ///
    /// * `handle` - The buffer handle to release
    ///
    /// # Panics
    ///
    /// Panics in debug mode if the buffer handle is invalid (wrong generation
    /// or already released).
    pub fn release(&mut self, handle: BufferHandle) {
        if let Some(entries) = self.pools.get_mut(&handle.class) {
            if let Some(entry) = entries.get_mut(handle.index) {
                // Validate generation
                debug_assert_eq!(
                    entry.generation, handle.generation,
                    "Buffer generation mismatch: expected {}, got {}",
                    entry.generation, handle.generation
                );

                if entry.in_use {
                    entry.in_use = false;

                    // Add to free list
                    if let Some(free_list) = self.free_lists.get_mut(&handle.class) {
                        free_list.push(handle.index);
                    }

                    // Update stats
                    if let Some(stats) = self.stats.get_mut(&handle.class) {
                        stats.in_use -= 1;
                        stats.free += 1;
                    }

                    trace!(
                        "Released buffer: class={}, index={}, gen={}",
                        handle.class,
                        handle.index,
                        handle.generation
                    );
                } else {
                    warn!(
                        "Attempted to release already-free buffer: class={}, index={}",
                        handle.class, handle.index
                    );
                }
            }
        }
    }

    /// Releases a PooledBuffer back to the pool.
    ///
    /// This is a convenience wrapper around `release` that takes the buffer wrapper.
    pub fn release_buffer(&mut self, buffer: PooledBuffer) {
        self.release(buffer.handle);
    }

    /// Returns the appropriate size class for a given size.
    ///
    /// Returns `None` if the size exceeds the largest class.
    #[inline]
    pub fn size_class_for(size: u64) -> Option<SizeClass> {
        SizeClass::for_size(size)
    }

    /// Returns current pool metrics.
    pub fn metrics(&self) -> PoolMetrics {
        let mut metrics = PoolMetrics {
            per_class_stats: self.stats.clone(),
            oversized_allocations: self.oversized_count,
            ..Default::default()
        };

        for (class, stats) in &self.stats {
            metrics.total_allocated += stats.total_allocated;
            metrics.total_in_use += stats.in_use;
            metrics.total_free += stats.free;
            metrics.total_bytes_allocated += stats.total_bytes;
            metrics.total_bytes_in_use += (stats.in_use as u64) * class.size_bytes();
        }

        metrics
    }

    /// Shrinks the pool if more than threshold fraction of buffers are free.
    ///
    /// This releases excess buffers to reduce memory usage when the pool
    /// is underutilized.
    ///
    /// # Returns
    ///
    /// The number of buffers released.
    pub fn shrink_if_needed(&mut self) -> usize {
        let threshold = self.config.shrink_threshold;
        let mut released = 0;

        for class in SizeClass::ALL {
            if let Some(stats) = self.stats.get(&class) {
                if stats.total_allocated == 0 {
                    continue;
                }

                let free_ratio = stats.free as f32 / stats.total_allocated as f32;
                if free_ratio > threshold {
                    // Calculate how many to release (keep at least half)
                    let target_free = (stats.total_allocated as f32 * threshold) as usize;
                    let to_release = stats.free.saturating_sub(target_free);

                    if to_release > 0 {
                        released += self.shrink_class(class, to_release);
                    }
                }
            }
        }

        if released > 0 {
            debug!("Shrunk pool, released {} buffers", released);
        }

        released
    }

    /// Forces shrinking of a specific class by a given count.
    fn shrink_class(&mut self, class: SizeClass, count: usize) -> usize {
        let mut released = 0;

        if let Some(free_list) = self.free_lists.get_mut(&class) {
            // Remove from free list (we can't actually deallocate wgpu buffers
            // in a targeted way, but we can mark slots as invalid)
            for _ in 0..count.min(free_list.len()) {
                if free_list.pop().is_some() {
                    released += 1;
                }
            }
        }

        if let Some(stats) = self.stats.get_mut(&class) {
            stats.free = stats.free.saturating_sub(released);
            stats.total_allocated = stats.total_allocated.saturating_sub(released);
            stats.total_bytes = stats.total_bytes.saturating_sub(
                (released as u64) * class.size_bytes()
            );
        }

        released
    }

    /// Grows a size class by allocating more buffers.
    fn grow_class(&mut self, device: &Device, class: SizeClass) {
        self.grow_class_with_usage(device, class, self.config.default_usage);
    }

    /// Grows a size class with specific usage flags.
    fn grow_class_with_usage(&mut self, device: &Device, class: SizeClass, usage: BufferUsages) {
        let entries = self.pools.entry(class).or_default();
        let current_count = entries.len();

        // Calculate growth
        let target_count = if current_count == 0 {
            self.config.growth_policy.initial_count
        } else {
            self.config.growth_policy.next_count(current_count)
        };

        let to_allocate = target_count.saturating_sub(current_count);
        if to_allocate == 0 {
            warn!("Cannot grow class {} further (at max {})", class, current_count);
            return;
        }

        debug!(
            "Growing class {} from {} to {} buffers",
            class, current_count, target_count
        );

        // Allocate new buffers
        for i in 0..to_allocate {
            let label = if self.config.enable_labels {
                Some(format!("pool_{}_{}_{}", class as u64, current_count + i, self.global_generation))
            } else {
                None
            };

            let buffer = device.create_buffer(&BufferDescriptor {
                label: label.as_deref(),
                size: class.size_bytes(),
                usage,
                mapped_at_creation: false,
            });

            let index = entries.len();
            entries.push(PoolEntry {
                buffer: Arc::new(buffer),
                usage,
                generation: 0,
                in_use: false,
                label,
            });

            // Add to free list
            if let Some(free_list) = self.free_lists.get_mut(&class) {
                free_list.push(index);
            }
        }

        self.global_generation += 1;

        // Update stats
        if let Some(stats) = self.stats.get_mut(&class) {
            stats.total_allocated += to_allocate;
            stats.free += to_allocate;
            stats.total_bytes += (to_allocate as u64) * class.size_bytes();
            stats.growth_count += 1;
        }
    }

    /// Clears all pools and releases all buffers.
    ///
    /// After calling this, the pool is empty and will need to grow
    /// again on the next acquire.
    pub fn clear(&mut self) {
        for class in SizeClass::ALL {
            self.pools.insert(class, Vec::new());
            self.free_lists.insert(class, Vec::new());
            self.stats.insert(class, ClassStats::default());
        }
        self.oversized_count = 0;
        self.global_generation = 0;

        debug!("BufferPool cleared");
    }

    /// Returns the configuration.
    pub fn config(&self) -> &PoolConfig {
        &self.config
    }

    /// Updates the configuration.
    ///
    /// Note: This does not resize existing pools; it only affects future growth.
    pub fn set_config(&mut self, config: PoolConfig) {
        self.config = config;
    }
}

impl std::fmt::Debug for BufferPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let metrics = self.metrics();
        f.debug_struct("BufferPool")
            .field("total_allocated", &metrics.total_allocated)
            .field("total_in_use", &metrics.total_in_use)
            .field("total_free", &metrics.total_free)
            .field("utilization", &format!("{:.1}%", metrics.utilization() * 100.0))
            .finish()
    }
}

// ============================================================================
// Pooled Buffer Guard (RAII)
// ============================================================================

/// RAII guard that automatically releases a buffer when dropped.
///
/// This provides automatic cleanup without needing to manually call
/// [`BufferPool::release`].
///
/// # Example
///
/// ```no_run
/// use renderer_backend::resources::buffer_pool::{BufferPool, PooledBufferGuard};
/// use wgpu::BufferUsages;
///
/// # fn example(device: &wgpu::Device) {
/// let mut pool = BufferPool::with_defaults(device);
///
/// // Create a guard that will auto-release
/// if let Some(handle) = pool.acquire(device, 1000, BufferUsages::VERTEX | BufferUsages::COPY_DST) {
///     // Use the buffer through pool.get_buffer(&handle)
///
///     // Buffer is automatically released when we call release
///     pool.release(handle);
/// }
/// # }
/// ```
pub struct PooledBufferGuard<'a> {
    pool: &'a mut BufferPool,
    handle: Option<BufferHandle>,
}

impl<'a> PooledBufferGuard<'a> {
    /// Creates a new guard for a buffer handle.
    ///
    /// The buffer will be released when the guard is dropped.
    pub fn new(pool: &'a mut BufferPool, handle: BufferHandle) -> Self {
        Self {
            pool,
            handle: Some(handle),
        }
    }

    /// Takes ownership of the handle, preventing automatic release.
    ///
    /// After calling this, the caller is responsible for releasing the buffer.
    pub fn take(mut self) -> BufferHandle {
        self.handle.take().expect("Handle already taken")
    }

    /// Returns the handle without taking ownership.
    pub fn handle(&self) -> BufferHandle {
        self.handle.expect("Handle already taken")
    }

    /// Gets a reference to the underlying buffer.
    pub fn buffer(&self) -> Option<&Buffer> {
        self.handle.and_then(|h| self.pool.get_buffer(&h))
    }
}

impl<'a> Drop for PooledBufferGuard<'a> {
    fn drop(&mut self) {
        if let Some(handle) = self.handle.take() {
            self.pool.release(handle);
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_size_class_for_size() {
        // Exact boundaries
        assert_eq!(SizeClass::for_size(256), Some(SizeClass::Tiny));
        assert_eq!(SizeClass::for_size(1024), Some(SizeClass::Small));
        assert_eq!(SizeClass::for_size(4096), Some(SizeClass::Medium));
        assert_eq!(SizeClass::for_size(16384), Some(SizeClass::Large));
        assert_eq!(SizeClass::for_size(65536), Some(SizeClass::XLarge));
        assert_eq!(SizeClass::for_size(262144), Some(SizeClass::Huge));
        assert_eq!(SizeClass::for_size(1048576), Some(SizeClass::Massive));

        // Below boundaries
        assert_eq!(SizeClass::for_size(1), Some(SizeClass::Tiny));
        assert_eq!(SizeClass::for_size(255), Some(SizeClass::Tiny));
        assert_eq!(SizeClass::for_size(257), Some(SizeClass::Small));
        assert_eq!(SizeClass::for_size(1023), Some(SizeClass::Small));
        assert_eq!(SizeClass::for_size(1025), Some(SizeClass::Medium));

        // Above max
        assert_eq!(SizeClass::for_size(1048577), None);
        assert_eq!(SizeClass::for_size(2 * 1024 * 1024), None);
    }

    #[test]
    fn test_size_class_size_bytes() {
        assert_eq!(SizeClass::Tiny.size_bytes(), 256);
        assert_eq!(SizeClass::Small.size_bytes(), 1024);
        assert_eq!(SizeClass::Medium.size_bytes(), 4096);
        assert_eq!(SizeClass::Large.size_bytes(), 16384);
        assert_eq!(SizeClass::XLarge.size_bytes(), 65536);
        assert_eq!(SizeClass::Huge.size_bytes(), 262144);
        assert_eq!(SizeClass::Massive.size_bytes(), 1048576);
    }

    #[test]
    fn test_size_class_next_larger() {
        assert_eq!(SizeClass::Tiny.next_larger(), Some(SizeClass::Small));
        assert_eq!(SizeClass::Small.next_larger(), Some(SizeClass::Medium));
        assert_eq!(SizeClass::Medium.next_larger(), Some(SizeClass::Large));
        assert_eq!(SizeClass::Large.next_larger(), Some(SizeClass::XLarge));
        assert_eq!(SizeClass::XLarge.next_larger(), Some(SizeClass::Huge));
        assert_eq!(SizeClass::Huge.next_larger(), Some(SizeClass::Massive));
        assert_eq!(SizeClass::Massive.next_larger(), None);
    }

    #[test]
    fn test_size_class_ordering() {
        // Size classes should be orderable
        assert!(SizeClass::Tiny < SizeClass::Small);
        assert!(SizeClass::Small < SizeClass::Medium);
        assert!(SizeClass::Medium < SizeClass::Large);
        assert!(SizeClass::Large < SizeClass::XLarge);
        assert!(SizeClass::XLarge < SizeClass::Huge);
        assert!(SizeClass::Huge < SizeClass::Massive);
    }

    #[test]
    fn test_size_class_display() {
        assert!(SizeClass::Tiny.to_string().contains("256B"));
        assert!(SizeClass::Small.to_string().contains("1KB"));
        assert!(SizeClass::Medium.to_string().contains("4KB"));
        assert!(SizeClass::Large.to_string().contains("16KB"));
        assert!(SizeClass::XLarge.to_string().contains("64KB"));
        assert!(SizeClass::Huge.to_string().contains("256KB"));
        assert!(SizeClass::Massive.to_string().contains("1MB"));
    }

    #[test]
    fn test_size_class_all() {
        assert_eq!(SizeClass::ALL.len(), 7);
        assert_eq!(SizeClass::ALL[0], SizeClass::Tiny);
        assert_eq!(SizeClass::ALL[6], SizeClass::Massive);
    }

    #[test]
    fn test_growth_policy_default() {
        let policy = GrowthPolicy::default();
        assert_eq!(policy.initial_count, 4);
        assert_eq!(policy.growth_factor, 2.0);
        assert_eq!(policy.max_per_class, 256);
    }

    #[test]
    fn test_growth_policy_next_count() {
        let policy = GrowthPolicy::default();

        assert_eq!(policy.next_count(0), 0); // 0 * 2 = 0 (edge case)
        assert_eq!(policy.next_count(4), 8);
        assert_eq!(policy.next_count(8), 16);
        assert_eq!(policy.next_count(128), 256);
        assert_eq!(policy.next_count(256), 256); // capped at max
        assert_eq!(policy.next_count(300), 256); // still capped
    }

    #[test]
    fn test_growth_policy_conservative() {
        let policy = GrowthPolicy::conservative();
        assert_eq!(policy.initial_count, 2);
        assert_eq!(policy.growth_factor, 1.5);
        assert_eq!(policy.max_per_class, 64);

        assert_eq!(policy.next_count(2), 3);
        assert_eq!(policy.next_count(3), 5); // 3 * 1.5 = 4.5 -> 5
        assert_eq!(policy.next_count(64), 64); // capped
    }

    #[test]
    fn test_growth_policy_aggressive() {
        let policy = GrowthPolicy::aggressive();
        assert_eq!(policy.initial_count, 8);
        assert_eq!(policy.growth_factor, 2.0);
        assert_eq!(policy.max_per_class, 512);
    }

    #[test]
    fn test_pool_config_default() {
        let config = PoolConfig::default();
        assert_eq!(config.shrink_threshold, 0.5);
        assert!(!config.pre_allocate);
        assert!(config.default_usage.contains(BufferUsages::COPY_DST));
        assert!(config.default_usage.contains(BufferUsages::STORAGE));
    }

    #[test]
    fn test_class_stats_utilization() {
        let mut stats = ClassStats::default();
        assert_eq!(stats.utilization(), 0.0);

        stats.total_allocated = 10;
        stats.in_use = 5;
        assert_eq!(stats.utilization(), 0.5);

        stats.in_use = 10;
        assert_eq!(stats.utilization(), 1.0);
    }

    #[test]
    fn test_class_stats_reuse_ratio() {
        let mut stats = ClassStats::default();
        assert_eq!(stats.reuse_ratio(), 0.0);

        stats.acquire_count = 10;
        stats.reuse_count = 7;
        assert!((stats.reuse_ratio() - 0.7).abs() < 0.001);
    }

    #[test]
    fn test_pool_metrics_utilization() {
        let mut metrics = PoolMetrics::default();
        assert_eq!(metrics.utilization(), 0.0);

        metrics.total_allocated = 20;
        metrics.total_in_use = 8;
        assert_eq!(metrics.utilization(), 0.4);
    }

    #[test]
    fn test_pool_metrics_memory_efficiency() {
        let mut metrics = PoolMetrics::default();
        assert_eq!(metrics.memory_efficiency(), 0.0);

        metrics.total_bytes_allocated = 1000;
        metrics.total_bytes_in_use = 800;
        assert_eq!(metrics.memory_efficiency(), 0.8);
    }

    #[test]
    fn test_buffer_handle_size_class() {
        let handle = BufferHandle {
            class: SizeClass::Medium,
            index: 5,
            generation: 10,
        };
        assert_eq!(handle.size_class(), SizeClass::Medium);
        assert_eq!(handle.size(), 4096);
    }

    #[test]
    fn test_size_class_hash() {
        use std::collections::HashSet;

        let mut set = HashSet::new();
        for class in SizeClass::ALL {
            set.insert(class);
        }
        assert_eq!(set.len(), 7);
    }

    #[test]
    fn test_buffer_pool_size_class_for() {
        assert_eq!(BufferPool::size_class_for(100), Some(SizeClass::Tiny));
        assert_eq!(BufferPool::size_class_for(500), Some(SizeClass::Small));
        assert_eq!(BufferPool::size_class_for(2000000), None);
    }

    // Integration tests that require a wgpu device are in separate test files
    // or are marked with #[ignore] to be run manually
}
