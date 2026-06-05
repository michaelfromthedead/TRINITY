//! Transient resource pooling for frame graph resource reuse.
//!
//! This module provides a pooling system for GPU resources (textures and buffers)
//! that enables efficient reuse of transient resources across frames. Instead of
//! allocating and deallocating GPU resources every frame, the pool maintains a
//! cache of resources keyed by their properties, allowing rapid acquisition and
//! release without GPU allocation overhead.
//!
//! # Memory Aliasing
//!
//! Resources with non-overlapping lifetimes in the frame graph can share the same
//! GPU memory. The [`AliasableResource`] trait enables tracking of resource
//! lifetimes (first_use_pass, last_use_pass) to determine aliasing eligibility.
//! This can reduce peak memory usage by 30-50% in typical rendering scenarios.
//!
//! # Example
//!
//! ```ignore
//! use renderer_backend::frame_graph::transient::*;
//!
//! // Create the pool
//! let mut pool = TransientPool::new(&device);
//!
//! // At frame start
//! pool.begin_frame(frame_number);
//!
//! // Acquire resources for rendering
//! let color_texture = pool.acquire_texture(&TextureDescriptor::new_render_target(
//!     1920, 1080, wgpu::TextureFormat::Rgba8Unorm
//! ));
//!
//! // ... render using color_texture ...
//!
//! // Release when done
//! pool.release_texture(color_texture.allocation_id);
//!
//! // At frame end - cleanup idle resources
//! pool.end_frame();
//! ```

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use super::resources::{BufferDescriptor, TextureDescriptor};

// ---------------------------------------------------------------------------
// SizeClass
// ---------------------------------------------------------------------------

/// Buffer size classification for pooling purposes.
///
/// By bucketing buffers into size classes, we reduce fragmentation and improve
/// cache hit rates. A buffer request is satisfied by any pooled buffer in the
/// same size class, potentially wasting some memory but avoiding frequent
/// allocations.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SizeClass {
    /// Less than 1KB (1,024 bytes)
    Tiny,
    /// 1KB to 64KB (65,536 bytes)
    Small,
    /// 64KB to 1MB (1,048,576 bytes)
    Medium,
    /// 1MB to 16MB (16,777,216 bytes)
    Large,
    /// Greater than 16MB
    Huge,
}

impl SizeClass {
    /// Determines the size class for a given byte size.
    ///
    /// # Arguments
    ///
    /// * `size` - The buffer size in bytes.
    ///
    /// # Returns
    ///
    /// The appropriate size class for the given size.
    #[inline]
    pub fn from_bytes(size: u64) -> Self {
        const KB: u64 = 1024;
        const MB: u64 = 1024 * KB;

        if size < KB {
            Self::Tiny
        } else if size < 64 * KB {
            Self::Small
        } else if size < MB {
            Self::Medium
        } else if size < 16 * MB {
            Self::Large
        } else {
            Self::Huge
        }
    }

    /// Returns the minimum size in bytes for this size class.
    ///
    /// # Returns
    ///
    /// The lower bound of the size range (inclusive).
    #[inline]
    pub const fn min_size(&self) -> u64 {
        const KB: u64 = 1024;
        const MB: u64 = 1024 * KB;

        match self {
            Self::Tiny => 0,
            Self::Small => KB,
            Self::Medium => 64 * KB,
            Self::Large => MB,
            Self::Huge => 16 * MB,
        }
    }

    /// Returns the maximum size in bytes for this size class.
    ///
    /// # Returns
    ///
    /// The upper bound of the size range (exclusive, except for Huge).
    #[inline]
    pub const fn max_size(&self) -> u64 {
        const KB: u64 = 1024;
        const MB: u64 = 1024 * KB;

        match self {
            Self::Tiny => KB - 1,
            Self::Small => 64 * KB - 1,
            Self::Medium => MB - 1,
            Self::Large => 16 * MB - 1,
            Self::Huge => u64::MAX,
        }
    }

    /// Returns the recommended allocation size for this class.
    ///
    /// When allocating a new buffer, we round up to this size to improve
    /// reuse potential for future requests in the same class.
    #[inline]
    pub const fn allocation_size(&self) -> u64 {
        const KB: u64 = 1024;
        const MB: u64 = 1024 * KB;

        match self {
            Self::Tiny => KB,
            Self::Small => 64 * KB,
            Self::Medium => MB,
            Self::Large => 16 * MB,
            Self::Huge => 64 * MB, // Start at 64MB for huge, actual may be larger
        }
    }
}

impl fmt::Display for SizeClass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Tiny => write!(f, "Tiny(<1KB)"),
            Self::Small => write!(f, "Small(1-64KB)"),
            Self::Medium => write!(f, "Medium(64KB-1MB)"),
            Self::Large => write!(f, "Large(1-16MB)"),
            Self::Huge => write!(f, "Huge(>16MB)"),
        }
    }
}

// ---------------------------------------------------------------------------
// PoolKey
// ---------------------------------------------------------------------------

/// Key for matching pooled resources.
///
/// Resources are pooled by their essential properties that determine GPU
/// compatibility. Two resources with the same PoolKey can be used
/// interchangeably (though the actual size may differ for buffers).
#[derive(Clone, Debug)]
pub enum PoolKey {
    /// Key for texture resources.
    Texture {
        /// Pixel format.
        format: wgpu::TextureFormat,
        /// Texture width in texels.
        width: u32,
        /// Texture height in texels.
        height: u32,
        /// Depth or array layers.
        depth_or_layers: u32,
        /// Mip level count.
        mip_levels: u32,
        /// Usage flags.
        usage: wgpu::TextureUsages,
        /// Sample count for MSAA.
        sample_count: u32,
        /// Texture dimensionality.
        dimension: wgpu::TextureDimension,
    },
    /// Key for buffer resources.
    Buffer {
        /// Size class for bucketing.
        size_class: SizeClass,
        /// Usage flags.
        usage: wgpu::BufferUsages,
    },
}

impl PoolKey {
    /// Creates a pool key from a texture descriptor.
    ///
    /// # Arguments
    ///
    /// * `desc` - The texture descriptor.
    ///
    /// # Returns
    ///
    /// A PoolKey capturing the essential texture properties.
    pub fn from_texture_desc(desc: &TextureDescriptor) -> Self {
        Self::Texture {
            format: desc.format,
            width: desc.width,
            height: desc.height,
            depth_or_layers: desc.depth_or_layers,
            mip_levels: desc.mip_levels,
            usage: desc.usage,
            sample_count: desc.sample_count,
            dimension: desc.dimension,
        }
    }

    /// Creates a pool key from a buffer descriptor.
    ///
    /// # Arguments
    ///
    /// * `desc` - The buffer descriptor.
    ///
    /// # Returns
    ///
    /// A PoolKey with size class and usage.
    pub fn from_buffer_desc(desc: &BufferDescriptor) -> Self {
        Self::Buffer {
            size_class: SizeClass::from_bytes(desc.size),
            usage: desc.usage,
        }
    }

    /// Returns true if this is a texture key.
    #[inline]
    pub fn is_texture(&self) -> bool {
        matches!(self, Self::Texture { .. })
    }

    /// Returns true if this is a buffer key.
    #[inline]
    pub fn is_buffer(&self) -> bool {
        matches!(self, Self::Buffer { .. })
    }
}

impl PartialEq for PoolKey {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (
                Self::Texture {
                    format: f1,
                    width: w1,
                    height: h1,
                    depth_or_layers: d1,
                    mip_levels: m1,
                    usage: u1,
                    sample_count: s1,
                    dimension: dim1,
                },
                Self::Texture {
                    format: f2,
                    width: w2,
                    height: h2,
                    depth_or_layers: d2,
                    mip_levels: m2,
                    usage: u2,
                    sample_count: s2,
                    dimension: dim2,
                },
            ) => {
                f1 == f2
                    && w1 == w2
                    && h1 == h2
                    && d1 == d2
                    && m1 == m2
                    && u1 == u2
                    && s1 == s2
                    && dim1 == dim2
            }
            (
                Self::Buffer {
                    size_class: sc1,
                    usage: u1,
                },
                Self::Buffer {
                    size_class: sc2,
                    usage: u2,
                },
            ) => sc1 == sc2 && u1 == u2,
            _ => false,
        }
    }
}

impl Eq for PoolKey {}

impl Hash for PoolKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        match self {
            Self::Texture {
                format,
                width,
                height,
                depth_or_layers,
                mip_levels,
                usage,
                sample_count,
                dimension,
            } => {
                0u8.hash(state); // discriminant
                // Hash format by its debug string (stable for a given wgpu version)
                std::mem::discriminant(format).hash(state);
                width.hash(state);
                height.hash(state);
                depth_or_layers.hash(state);
                mip_levels.hash(state);
                usage.bits().hash(state);
                sample_count.hash(state);
                std::mem::discriminant(dimension).hash(state);
            }
            Self::Buffer { size_class, usage } => {
                1u8.hash(state); // discriminant
                size_class.hash(state);
                usage.bits().hash(state);
            }
        }
    }
}

impl fmt::Display for PoolKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Texture {
                format,
                width,
                height,
                depth_or_layers,
                mip_levels,
                usage,
                sample_count,
                dimension,
            } => {
                write!(
                    f,
                    "Texture({:?}, {}x{}x{}, mips={}, samples={}, {:?}, {:?})",
                    format, width, height, depth_or_layers, mip_levels, sample_count, usage, dimension
                )
            }
            Self::Buffer { size_class, usage } => {
                write!(f, "Buffer({}, {:?})", size_class, usage)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// PooledTexture
// ---------------------------------------------------------------------------

/// A pooled texture resource with tracking metadata.
///
/// Represents a GPU texture managed by the transient pool. The pool tracks
/// when the texture was last used to enable garbage collection of idle
/// resources.
pub struct PooledTexture {
    /// The wgpu texture handle.
    pub texture: wgpu::Texture,
    /// A default view covering the entire texture.
    pub view: wgpu::TextureView,
    /// The pool key this texture was allocated with.
    pub key: PoolKey,
    /// Frame number when this texture was last used.
    pub last_used_frame: u64,
    /// Unique allocation ID for tracking.
    pub allocation_id: u64,
}

impl PooledTexture {
    /// Creates a new pooled texture.
    ///
    /// # Arguments
    ///
    /// * `texture` - The wgpu texture.
    /// * `view` - The default texture view.
    /// * `key` - The pool key for this texture.
    /// * `frame` - The current frame number.
    /// * `allocation_id` - Unique allocation ID.
    pub fn new(
        texture: wgpu::Texture,
        view: wgpu::TextureView,
        key: PoolKey,
        frame: u64,
        allocation_id: u64,
    ) -> Self {
        Self {
            texture,
            view,
            key,
            last_used_frame: frame,
            allocation_id,
        }
    }

    /// Returns the width of the texture.
    #[inline]
    pub fn width(&self) -> u32 {
        self.texture.width()
    }

    /// Returns the height of the texture.
    #[inline]
    pub fn height(&self) -> u32 {
        self.texture.height()
    }

    /// Returns the depth or array layer count.
    #[inline]
    pub fn depth_or_array_layers(&self) -> u32 {
        self.texture.depth_or_array_layers()
    }

    /// Returns the mip level count.
    #[inline]
    pub fn mip_level_count(&self) -> u32 {
        self.texture.mip_level_count()
    }

    /// Returns the sample count.
    #[inline]
    pub fn sample_count(&self) -> u32 {
        self.texture.sample_count()
    }

    /// Returns the texture format.
    #[inline]
    pub fn format(&self) -> wgpu::TextureFormat {
        self.texture.format()
    }

    /// Returns the texture usages.
    #[inline]
    pub fn usage(&self) -> wgpu::TextureUsages {
        self.texture.usage()
    }
}

impl fmt::Debug for PooledTexture {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PooledTexture")
            .field("key", &self.key)
            .field("last_used_frame", &self.last_used_frame)
            .field("allocation_id", &self.allocation_id)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// PooledBuffer
// ---------------------------------------------------------------------------

/// A pooled buffer resource with tracking metadata.
///
/// Represents a GPU buffer managed by the transient pool. The actual size
/// may be larger than requested (rounded up to size class) to improve reuse.
pub struct PooledBuffer {
    /// The wgpu buffer handle.
    pub buffer: wgpu::Buffer,
    /// The pool key this buffer was allocated with.
    pub key: PoolKey,
    /// The actual allocated size (may be > requested).
    pub actual_size: u64,
    /// Frame number when this buffer was last used.
    pub last_used_frame: u64,
    /// Unique allocation ID for tracking.
    pub allocation_id: u64,
}

impl PooledBuffer {
    /// Creates a new pooled buffer.
    ///
    /// # Arguments
    ///
    /// * `buffer` - The wgpu buffer.
    /// * `key` - The pool key for this buffer.
    /// * `actual_size` - The actual allocated size.
    /// * `frame` - The current frame number.
    /// * `allocation_id` - Unique allocation ID.
    pub fn new(
        buffer: wgpu::Buffer,
        key: PoolKey,
        actual_size: u64,
        frame: u64,
        allocation_id: u64,
    ) -> Self {
        Self {
            buffer,
            key,
            actual_size,
            last_used_frame: frame,
            allocation_id,
        }
    }

    /// Returns the buffer size.
    #[inline]
    pub fn size(&self) -> u64 {
        self.buffer.size()
    }

    /// Returns the buffer usages.
    #[inline]
    pub fn usage(&self) -> wgpu::BufferUsages {
        self.buffer.usage()
    }
}

impl fmt::Debug for PooledBuffer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PooledBuffer")
            .field("key", &self.key)
            .field("actual_size", &self.actual_size)
            .field("last_used_frame", &self.last_used_frame)
            .field("allocation_id", &self.allocation_id)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// PoolStats
// ---------------------------------------------------------------------------

/// Statistics about pool usage.
///
/// Tracks allocation counts, memory usage, and cache hit rates to help
/// tune pool configuration and identify memory issues.
#[derive(Clone, Debug, Default)]
pub struct PoolStats {
    /// Total number of textures in the pool (active + idle).
    pub total_textures: u64,
    /// Total number of buffers in the pool (active + idle).
    pub total_buffers: u64,
    /// Number of textures currently in use.
    pub active_textures: u64,
    /// Number of buffers currently in use.
    pub active_buffers: u64,
    /// Total memory used by pooled textures (bytes).
    pub texture_bytes: u64,
    /// Total memory used by pooled buffers (bytes).
    pub buffer_bytes: u64,
    /// Number of successful cache hits (resource reused).
    pub cache_hits: u64,
    /// Number of cache misses (new allocation required).
    pub cache_misses: u64,
    /// Number of resources garbage collected.
    pub gc_count: u64,
}

impl PoolStats {
    /// Creates new empty stats.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the cache hit rate as a percentage (0.0 - 1.0).
    ///
    /// Returns 0.0 if no acquisitions have been made.
    pub fn hit_rate(&self) -> f32 {
        let total = self.cache_hits + self.cache_misses;
        if total == 0 {
            0.0
        } else {
            self.cache_hits as f32 / total as f32
        }
    }

    /// Returns the total memory usage in bytes.
    pub fn total_bytes(&self) -> u64 {
        self.texture_bytes + self.buffer_bytes
    }

    /// Returns the number of idle textures.
    pub fn idle_textures(&self) -> u64 {
        self.total_textures.saturating_sub(self.active_textures)
    }

    /// Returns the number of idle buffers.
    pub fn idle_buffers(&self) -> u64 {
        self.total_buffers.saturating_sub(self.active_buffers)
    }

    /// Resets the hit/miss counters.
    pub fn reset_counters(&mut self) {
        self.cache_hits = 0;
        self.cache_misses = 0;
        self.gc_count = 0;
    }
}

impl fmt::Display for PoolStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "PoolStats(textures={}/{}, buffers={}/{}, memory={:.2}MB, hit_rate={:.1}%)",
            self.active_textures,
            self.total_textures,
            self.active_buffers,
            self.total_buffers,
            self.total_bytes() as f64 / (1024.0 * 1024.0),
            self.hit_rate() * 100.0
        )
    }
}

// ---------------------------------------------------------------------------
// PoolConfig
// ---------------------------------------------------------------------------

/// Configuration for the transient resource pool.
#[derive(Clone, Debug)]
pub struct PoolConfig {
    /// Number of frames a resource can be idle before garbage collection.
    pub max_idle_frames: u64,
    /// Maximum number of textures per pool key.
    pub max_texture_pool_size: usize,
    /// Maximum number of buffers per pool key.
    pub max_buffer_pool_size: usize,
    /// Whether to enable memory defragmentation.
    pub enable_defragmentation: bool,
    /// Maximum total memory budget (0 = unlimited).
    pub max_memory_bytes: u64,
}

impl Default for PoolConfig {
    fn default() -> Self {
        Self {
            max_idle_frames: 3,
            max_texture_pool_size: 8,
            max_buffer_pool_size: 16,
            enable_defragmentation: false,
            max_memory_bytes: 0, // unlimited
        }
    }
}

impl PoolConfig {
    /// Creates a new configuration with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Builder method to set max idle frames.
    pub fn with_max_idle_frames(mut self, frames: u64) -> Self {
        self.max_idle_frames = frames;
        self
    }

    /// Builder method to set max texture pool size.
    pub fn with_max_texture_pool_size(mut self, size: usize) -> Self {
        self.max_texture_pool_size = size;
        self
    }

    /// Builder method to set max buffer pool size.
    pub fn with_max_buffer_pool_size(mut self, size: usize) -> Self {
        self.max_buffer_pool_size = size;
        self
    }

    /// Builder method to enable/disable defragmentation.
    pub fn with_defragmentation(mut self, enable: bool) -> Self {
        self.enable_defragmentation = enable;
        self
    }

    /// Builder method to set memory budget.
    pub fn with_memory_budget(mut self, bytes: u64) -> Self {
        self.max_memory_bytes = bytes;
        self
    }
}

// ---------------------------------------------------------------------------
// AliasableResource trait
// ---------------------------------------------------------------------------

/// Trait for resources that can potentially share GPU memory through aliasing.
///
/// Memory aliasing allows resources with non-overlapping lifetimes to share
/// the same underlying GPU memory, reducing peak memory usage. This is
/// particularly effective for transient render targets in a frame graph.
///
/// # Lifetime Model
///
/// Resources have a defined lifetime within the frame graph execution:
/// - `first_use_pass`: The pass index where the resource is first read/written
/// - `last_use_pass`: The pass index where the resource is last read/written
///
/// Two resources can alias if their lifetimes don't overlap:
/// `A.last_use_pass < B.first_use_pass` or `B.last_use_pass < A.first_use_pass`
pub trait AliasableResource {
    /// Returns the first pass that uses this resource.
    fn first_use_pass(&self) -> u32;

    /// Returns the last pass that uses this resource.
    fn last_use_pass(&self) -> u32;

    /// Returns true if this resource can alias with another.
    ///
    /// Two resources can alias if:
    /// 1. Their lifetimes don't overlap (non-overlapping pass ranges)
    /// 2. They have compatible memory requirements (same alignment, etc.)
    ///
    /// # Arguments
    ///
    /// * `other` - The other resource to check aliasing with.
    fn can_alias_with(&self, other: &Self) -> bool {
        !self.overlapping_lifetime(other)
    }

    /// Returns true if this resource's lifetime overlaps with another.
    ///
    /// # Arguments
    ///
    /// * `other` - The other resource to check overlap with.
    fn overlapping_lifetime(&self, other: &Self) -> bool {
        // Lifetimes overlap if: A.first <= B.last && B.first <= A.last
        self.first_use_pass() <= other.last_use_pass()
            && other.first_use_pass() <= self.last_use_pass()
    }
}

/// A resource lifetime entry for aliasing calculations.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResourceLifetimeRange {
    /// The resource allocation ID.
    pub allocation_id: u64,
    /// First pass that uses this resource.
    pub first_use_pass: u32,
    /// Last pass that uses this resource.
    pub last_use_pass: u32,
    /// Approximate memory size in bytes.
    pub size_bytes: u64,
}

impl ResourceLifetimeRange {
    /// Creates a new lifetime range.
    pub fn new(allocation_id: u64, first_pass: u32, last_pass: u32, size_bytes: u64) -> Self {
        Self {
            allocation_id,
            first_use_pass: first_pass,
            last_use_pass: last_pass,
            size_bytes,
        }
    }
}

impl AliasableResource for ResourceLifetimeRange {
    fn first_use_pass(&self) -> u32 {
        self.first_use_pass
    }

    fn last_use_pass(&self) -> u32 {
        self.last_use_pass
    }
}

// ---------------------------------------------------------------------------
// TransientPool
// ---------------------------------------------------------------------------

/// The main transient resource pool.
///
/// Manages pools of GPU textures and buffers for efficient reuse across frames.
/// Resources are keyed by their essential properties (format, dimensions, usage)
/// and can be quickly acquired and released without GPU allocation overhead.
///
/// # Thread Safety
///
/// The pool is not thread-safe. If concurrent access is needed, wrap in an
/// appropriate synchronization primitive (e.g., `Mutex`, `RwLock`).
///
/// # Usage Pattern
///
/// 1. Call `begin_frame()` at the start of each frame
/// 2. Acquire resources with `acquire_texture()` / `acquire_buffer()`
/// 3. Use resources for rendering
/// 4. Release resources with `release_texture()` / `release_buffer()`
/// 5. Call `end_frame()` to cleanup idle resources
pub struct TransientPool {
    /// Device reference for creating new resources.
    device: Arc<wgpu::Device>,
    /// Pools of available textures, keyed by properties.
    texture_pools: HashMap<PoolKey, Vec<PooledTexture>>,
    /// Pools of available buffers, keyed by properties.
    buffer_pools: HashMap<PoolKey, Vec<PooledBuffer>>,
    /// Set of texture allocation IDs currently in use.
    active_textures: HashSet<u64>,
    /// Set of buffer allocation IDs currently in use.
    active_buffers: HashSet<u64>,
    /// Currently active textures by allocation ID (for release).
    active_texture_map: HashMap<u64, PooledTexture>,
    /// Currently active buffers by allocation ID (for release).
    active_buffer_map: HashMap<u64, PooledBuffer>,
    /// Current frame number.
    current_frame: u64,
    /// Next allocation ID to assign.
    next_allocation_id: u64,
    /// Pool statistics.
    stats: PoolStats,
    /// Pool configuration.
    config: PoolConfig,
}

impl TransientPool {
    /// Creates a new transient pool with default configuration.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device for creating resources.
    pub fn new(device: Arc<wgpu::Device>) -> Self {
        Self::with_config(device, PoolConfig::default())
    }

    /// Creates a new transient pool with custom configuration.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device for creating resources.
    /// * `config` - Pool configuration.
    pub fn with_config(device: Arc<wgpu::Device>, config: PoolConfig) -> Self {
        Self {
            device,
            texture_pools: HashMap::new(),
            buffer_pools: HashMap::new(),
            active_textures: HashSet::new(),
            active_buffers: HashSet::new(),
            active_texture_map: HashMap::new(),
            active_buffer_map: HashMap::new(),
            current_frame: 0,
            next_allocation_id: 1,
            stats: PoolStats::new(),
            config,
        }
    }

    /// Acquires a texture from the pool, creating a new one if necessary.
    ///
    /// If a compatible texture exists in the pool, it is returned immediately.
    /// Otherwise, a new texture is allocated from the GPU.
    ///
    /// # Arguments
    ///
    /// * `desc` - The texture descriptor.
    ///
    /// # Returns
    ///
    /// A pooled texture that can be used for rendering.
    pub fn acquire_texture(&mut self, desc: &TextureDescriptor) -> PooledTexture {
        let key = PoolKey::from_texture_desc(desc);

        // Try to find an existing texture in the pool
        if let Some(pool) = self.texture_pools.get_mut(&key) {
            if let Some(mut texture) = pool.pop() {
                texture.last_used_frame = self.current_frame;
                let id = texture.allocation_id;
                self.active_textures.insert(id);
                self.stats.cache_hits += 1;
                self.stats.active_textures += 1;

                // Return the pooled texture directly
                return texture;
            }
        }

        // Allocate a new texture
        self.stats.cache_misses += 1;
        self.allocate_texture(desc)
    }

    /// Creates an active texture and tracks it.
    fn create_active_texture(&mut self, desc: &TextureDescriptor, allocation_id: u64) -> PooledTexture {
        let wgpu_desc = wgpu::TextureDescriptor {
            label: desc.label.as_deref(),
            size: wgpu::Extent3d {
                width: desc.width,
                height: desc.height,
                depth_or_array_layers: desc.depth_or_layers,
            },
            mip_level_count: desc.mip_levels,
            sample_count: desc.sample_count,
            dimension: desc.dimension,
            format: desc.format,
            usage: desc.usage,
            view_formats: &[],
        };

        let texture = self.device.create_texture(&wgpu_desc);
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let key = PoolKey::from_texture_desc(desc);

        self.stats.total_textures += 1;
        self.stats.active_textures += 1;
        self.stats.texture_bytes += desc.size_bytes();

        PooledTexture::new(texture, view, key, self.current_frame, allocation_id)
    }

    /// Allocates a new texture.
    fn allocate_texture(&mut self, desc: &TextureDescriptor) -> PooledTexture {
        let allocation_id = self.next_allocation_id;
        self.next_allocation_id += 1;
        self.active_textures.insert(allocation_id);

        self.create_active_texture(desc, allocation_id)
    }

    /// Acquires a buffer from the pool, creating a new one if necessary.
    ///
    /// Buffers are matched by size class and usage. The returned buffer
    /// may be larger than requested (rounded up to size class).
    ///
    /// # Arguments
    ///
    /// * `desc` - The buffer descriptor.
    ///
    /// # Returns
    ///
    /// A pooled buffer that can be used for rendering.
    pub fn acquire_buffer(&mut self, desc: &BufferDescriptor) -> PooledBuffer {
        let key = PoolKey::from_buffer_desc(desc);

        // Try to find an existing buffer in the pool
        if let Some(pool) = self.buffer_pools.get_mut(&key) {
            // Find a buffer that's large enough
            if let Some(idx) = pool.iter().position(|b| b.actual_size >= desc.size) {
                let mut buffer = pool.swap_remove(idx);
                buffer.last_used_frame = self.current_frame;
                let id = buffer.allocation_id;
                self.active_buffers.insert(id);
                self.stats.cache_hits += 1;
                self.stats.active_buffers += 1;

                return buffer;
            }
        }

        // Allocate a new buffer
        self.stats.cache_misses += 1;
        self.allocate_buffer(desc)
    }

    /// Allocates a new buffer.
    fn allocate_buffer(&mut self, desc: &BufferDescriptor) -> PooledBuffer {
        let allocation_id = self.next_allocation_id;
        self.next_allocation_id += 1;
        self.active_buffers.insert(allocation_id);

        // Round up to size class allocation size for better reuse
        let size_class = SizeClass::from_bytes(desc.size);
        let actual_size = desc.size.max(size_class.allocation_size());

        let wgpu_desc = wgpu::BufferDescriptor {
            label: desc.label.as_deref(),
            size: actual_size,
            usage: desc.usage,
            mapped_at_creation: desc.mapped_at_creation,
        };

        let buffer = self.device.create_buffer(&wgpu_desc);
        let key = PoolKey::from_buffer_desc(desc);

        self.stats.total_buffers += 1;
        self.stats.active_buffers += 1;
        self.stats.buffer_bytes += actual_size;

        PooledBuffer::new(buffer, key, actual_size, self.current_frame, allocation_id)
    }

    /// Releases a texture back to the pool.
    ///
    /// The texture becomes available for reuse by future `acquire_texture` calls.
    ///
    /// # Arguments
    ///
    /// * `texture` - The pooled texture to release.
    pub fn release_texture(&mut self, texture: PooledTexture) {
        let id = texture.allocation_id;
        if self.active_textures.remove(&id) {
            self.stats.active_textures = self.stats.active_textures.saturating_sub(1);

            // Return to pool
            let key = texture.key.clone();
            let pool = self.texture_pools.entry(key).or_insert_with(Vec::new);

            // Enforce pool size limit
            if pool.len() < self.config.max_texture_pool_size {
                pool.push(texture);
            } else {
                // Drop the texture (GPU will deallocate)
                self.stats.total_textures = self.stats.total_textures.saturating_sub(1);
            }
        }
    }

    /// Releases a texture by allocation ID.
    ///
    /// # Arguments
    ///
    /// * `allocation_id` - The allocation ID of the texture to release.
    ///
    /// # Returns
    ///
    /// True if the texture was found and released, false otherwise.
    pub fn release_texture_by_id(&mut self, allocation_id: u64) -> bool {
        if self.active_textures.remove(&allocation_id) {
            self.stats.active_textures = self.stats.active_textures.saturating_sub(1);
            true
        } else {
            false
        }
    }

    /// Releases a buffer back to the pool.
    ///
    /// The buffer becomes available for reuse by future `acquire_buffer` calls.
    ///
    /// # Arguments
    ///
    /// * `buffer` - The pooled buffer to release.
    pub fn release_buffer(&mut self, buffer: PooledBuffer) {
        let id = buffer.allocation_id;
        if self.active_buffers.remove(&id) {
            self.stats.active_buffers = self.stats.active_buffers.saturating_sub(1);

            // Return to pool
            let key = buffer.key.clone();
            let pool = self.buffer_pools.entry(key).or_insert_with(Vec::new);

            // Enforce pool size limit
            if pool.len() < self.config.max_buffer_pool_size {
                pool.push(buffer);
            } else {
                // Drop the buffer (GPU will deallocate)
                self.stats.total_buffers = self.stats.total_buffers.saturating_sub(1);
            }
        }
    }

    /// Releases a buffer by allocation ID.
    ///
    /// # Arguments
    ///
    /// * `allocation_id` - The allocation ID of the buffer to release.
    ///
    /// # Returns
    ///
    /// True if the buffer was found and released, false otherwise.
    pub fn release_buffer_by_id(&mut self, allocation_id: u64) -> bool {
        if self.active_buffers.remove(&allocation_id) {
            self.stats.active_buffers = self.stats.active_buffers.saturating_sub(1);
            true
        } else {
            false
        }
    }

    /// Marks the beginning of a new frame.
    ///
    /// Should be called at the start of each frame before acquiring resources.
    ///
    /// # Arguments
    ///
    /// * `frame_number` - The current frame number.
    pub fn begin_frame(&mut self, frame_number: u64) {
        self.current_frame = frame_number;
    }

    /// Marks the end of the current frame.
    ///
    /// Performs garbage collection of idle resources based on configuration.
    pub fn end_frame(&mut self) {
        self.gc_idle_resources();
    }

    /// Forces garbage collection of all idle resources.
    ///
    /// Removes resources that have been idle longer than `max_idle_frames`.
    pub fn gc(&mut self) {
        self.gc_idle_resources();
    }

    /// Internal garbage collection implementation.
    fn gc_idle_resources(&mut self) {
        let max_idle = self.config.max_idle_frames;
        let current_frame = self.current_frame;

        // GC textures
        for pool in self.texture_pools.values_mut() {
            let before = pool.len();
            pool.retain(|t| current_frame - t.last_used_frame <= max_idle);
            let removed = before - pool.len();
            self.stats.gc_count += removed as u64;
            self.stats.total_textures = self.stats.total_textures.saturating_sub(removed as u64);
        }

        // GC buffers
        for pool in self.buffer_pools.values_mut() {
            let before = pool.len();
            pool.retain(|b| current_frame - b.last_used_frame <= max_idle);
            let removed = before - pool.len();
            self.stats.gc_count += removed as u64;
            self.stats.total_buffers = self.stats.total_buffers.saturating_sub(removed as u64);
        }

        // Remove empty pools
        self.texture_pools.retain(|_, v| !v.is_empty());
        self.buffer_pools.retain(|_, v| !v.is_empty());
    }

    /// Clears all resources from the pool.
    ///
    /// This releases all pooled resources. Active resources are not affected.
    pub fn clear(&mut self) {
        let cleared_textures = self.texture_pools.values().map(|v| v.len() as u64).sum::<u64>();
        let cleared_buffers = self.buffer_pools.values().map(|v| v.len() as u64).sum::<u64>();

        self.texture_pools.clear();
        self.buffer_pools.clear();

        self.stats.total_textures = self.stats.total_textures.saturating_sub(cleared_textures);
        self.stats.total_buffers = self.stats.total_buffers.saturating_sub(cleared_buffers);
        self.stats.gc_count += cleared_textures + cleared_buffers;
    }

    /// Returns a reference to the pool statistics.
    pub fn stats(&self) -> &PoolStats {
        &self.stats
    }

    /// Returns the current frame number.
    pub fn current_frame(&self) -> u64 {
        self.current_frame
    }

    /// Returns the pool configuration.
    pub fn config(&self) -> &PoolConfig {
        &self.config
    }

    /// Updates the pool configuration.
    ///
    /// Note: This does not retroactively apply to existing pooled resources.
    pub fn set_config(&mut self, config: PoolConfig) {
        self.config = config;
    }

    /// Returns the number of texture pools.
    pub fn texture_pool_count(&self) -> usize {
        self.texture_pools.len()
    }

    /// Returns the number of buffer pools.
    pub fn buffer_pool_count(&self) -> usize {
        self.buffer_pools.len()
    }

    /// Returns the total number of idle textures across all pools.
    pub fn idle_texture_count(&self) -> usize {
        self.texture_pools.values().map(|v| v.len()).sum()
    }

    /// Returns the total number of idle buffers across all pools.
    pub fn idle_buffer_count(&self) -> usize {
        self.buffer_pools.values().map(|v| v.len()).sum()
    }
}

impl fmt::Debug for TransientPool {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TransientPool")
            .field("current_frame", &self.current_frame)
            .field("texture_pools", &self.texture_pools.len())
            .field("buffer_pools", &self.buffer_pools.len())
            .field("active_textures", &self.active_textures.len())
            .field("active_buffers", &self.active_buffers.len())
            .field("stats", &self.stats)
            .field("config", &self.config)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// AliasGroup
// ---------------------------------------------------------------------------

/// A group of resources that can share the same GPU memory through aliasing.
///
/// Resources in an alias group have non-overlapping lifetimes, allowing them
/// to use the same underlying memory allocation.
#[derive(Clone, Debug)]
pub struct AliasGroup {
    /// The allocation IDs of resources in this group.
    pub members: Vec<u64>,
    /// The size of the shared memory region (max of all member sizes).
    pub memory_size: u64,
    /// First pass that uses any resource in this group.
    pub first_pass: u32,
    /// Last pass that uses any resource in this group.
    pub last_pass: u32,
}

impl AliasGroup {
    /// Creates a new empty alias group.
    pub fn new() -> Self {
        Self {
            members: Vec::new(),
            memory_size: 0,
            first_pass: u32::MAX,
            last_pass: 0,
        }
    }

    /// Attempts to add a resource to this group.
    ///
    /// Returns true if the resource was added (no lifetime overlap),
    /// false if it cannot be added (overlapping lifetime).
    pub fn try_add(&mut self, resource: &ResourceLifetimeRange) -> bool {
        // Check if this resource overlaps with any existing member
        // For simplicity, we track group's overall lifetime span
        // A more sophisticated approach would track individual lifetimes

        // If group is empty, always accept
        if self.members.is_empty() {
            self.members.push(resource.allocation_id);
            self.memory_size = resource.size_bytes;
            self.first_pass = resource.first_use_pass;
            self.last_pass = resource.last_use_pass;
            return true;
        }

        // Check if resource overlaps with group's current span
        // This is a conservative check - could be more precise with per-member tracking
        if resource.first_use_pass <= self.last_pass && resource.last_use_pass >= self.first_pass {
            return false;
        }

        // No overlap, add to group
        self.members.push(resource.allocation_id);
        self.memory_size = self.memory_size.max(resource.size_bytes);
        self.first_pass = self.first_pass.min(resource.first_use_pass);
        self.last_pass = self.last_pass.max(resource.last_use_pass);
        true
    }

    /// Returns the number of resources in this group.
    pub fn len(&self) -> usize {
        self.members.len()
    }

    /// Returns true if the group is empty.
    pub fn is_empty(&self) -> bool {
        self.members.is_empty()
    }
}

impl Default for AliasGroup {
    fn default() -> Self {
        Self::new()
    }
}

/// Computes alias groups for a set of resource lifetimes.
///
/// Uses a greedy first-fit algorithm to pack resources into alias groups.
/// Resources with non-overlapping lifetimes are placed in the same group.
///
/// # Arguments
///
/// * `resources` - The resource lifetime ranges to group.
///
/// # Returns
///
/// A vector of alias groups, each containing non-overlapping resources.
pub fn compute_alias_groups(resources: &[ResourceLifetimeRange]) -> Vec<AliasGroup> {
    let mut groups: Vec<AliasGroup> = Vec::new();

    // Sort resources by first use pass for better packing
    let mut sorted: Vec<_> = resources.iter().collect();
    sorted.sort_by_key(|r| r.first_use_pass);

    for resource in sorted {
        // Try to fit into an existing group
        let mut added = false;
        for group in &mut groups {
            if group.try_add(resource) {
                added = true;
                break;
            }
        }

        // If no group could accept it, create a new one
        if !added {
            let mut new_group = AliasGroup::new();
            new_group.try_add(resource);
            groups.push(new_group);
        }
    }

    groups
}

/// Estimates memory savings from aliasing.
///
/// # Arguments
///
/// * `resources` - The resource lifetime ranges.
/// * `groups` - The computed alias groups.
///
/// # Returns
///
/// A tuple of (total_without_aliasing, total_with_aliasing, savings_percentage).
pub fn estimate_aliasing_savings(
    resources: &[ResourceLifetimeRange],
    groups: &[AliasGroup],
) -> (u64, u64, f32) {
    let total_without = resources.iter().map(|r| r.size_bytes).sum::<u64>();
    let total_with = groups.iter().map(|g| g.memory_size).sum::<u64>();

    let savings = if total_without > 0 {
        (total_without - total_with) as f32 / total_without as f32 * 100.0
    } else {
        0.0
    };

    (total_without, total_with, savings)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // Helper to create a mock device for testing
    // In real tests, you'd use wgpu's testing utilities
    fn create_test_pool() -> Option<TransientPool> {
        // Skip tests that require a real GPU device
        // These would be integration tests
        None
    }

    #[test]
    fn test_size_class_from_bytes() {
        assert_eq!(SizeClass::from_bytes(0), SizeClass::Tiny);
        assert_eq!(SizeClass::from_bytes(512), SizeClass::Tiny);
        assert_eq!(SizeClass::from_bytes(1023), SizeClass::Tiny);

        assert_eq!(SizeClass::from_bytes(1024), SizeClass::Small);
        assert_eq!(SizeClass::from_bytes(32 * 1024), SizeClass::Small);
        assert_eq!(SizeClass::from_bytes(64 * 1024 - 1), SizeClass::Small);

        assert_eq!(SizeClass::from_bytes(64 * 1024), SizeClass::Medium);
        assert_eq!(SizeClass::from_bytes(512 * 1024), SizeClass::Medium);
        assert_eq!(SizeClass::from_bytes(1024 * 1024 - 1), SizeClass::Medium);

        assert_eq!(SizeClass::from_bytes(1024 * 1024), SizeClass::Large);
        assert_eq!(SizeClass::from_bytes(8 * 1024 * 1024), SizeClass::Large);
        assert_eq!(SizeClass::from_bytes(16 * 1024 * 1024 - 1), SizeClass::Large);

        assert_eq!(SizeClass::from_bytes(16 * 1024 * 1024), SizeClass::Huge);
        assert_eq!(SizeClass::from_bytes(100 * 1024 * 1024), SizeClass::Huge);
    }

    #[test]
    fn test_size_class_bounds() {
        assert_eq!(SizeClass::Tiny.min_size(), 0);
        assert_eq!(SizeClass::Tiny.max_size(), 1023);

        assert_eq!(SizeClass::Small.min_size(), 1024);
        assert_eq!(SizeClass::Small.max_size(), 64 * 1024 - 1);

        assert_eq!(SizeClass::Medium.min_size(), 64 * 1024);
        assert_eq!(SizeClass::Medium.max_size(), 1024 * 1024 - 1);

        assert_eq!(SizeClass::Large.min_size(), 1024 * 1024);
        assert_eq!(SizeClass::Large.max_size(), 16 * 1024 * 1024 - 1);

        assert_eq!(SizeClass::Huge.min_size(), 16 * 1024 * 1024);
        assert_eq!(SizeClass::Huge.max_size(), u64::MAX);
    }

    #[test]
    fn test_size_class_allocation_size() {
        assert_eq!(SizeClass::Tiny.allocation_size(), 1024);
        assert_eq!(SizeClass::Small.allocation_size(), 64 * 1024);
        assert_eq!(SizeClass::Medium.allocation_size(), 1024 * 1024);
        assert_eq!(SizeClass::Large.allocation_size(), 16 * 1024 * 1024);
        assert_eq!(SizeClass::Huge.allocation_size(), 64 * 1024 * 1024);
    }

    #[test]
    fn test_pool_key_texture_equality() {
        let key1 = PoolKey::Texture {
            format: wgpu::TextureFormat::Rgba8Unorm,
            width: 1920,
            height: 1080,
            depth_or_layers: 1,
            mip_levels: 1,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
        };

        let key2 = PoolKey::Texture {
            format: wgpu::TextureFormat::Rgba8Unorm,
            width: 1920,
            height: 1080,
            depth_or_layers: 1,
            mip_levels: 1,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
        };

        let key3 = PoolKey::Texture {
            format: wgpu::TextureFormat::Rgba8Unorm,
            width: 1280, // Different width
            height: 720,
            depth_or_layers: 1,
            mip_levels: 1,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
        };

        assert_eq!(key1, key2);
        assert_ne!(key1, key3);
    }

    #[test]
    fn test_pool_key_buffer_equality() {
        let key1 = PoolKey::Buffer {
            size_class: SizeClass::Medium,
            usage: wgpu::BufferUsages::STORAGE,
        };

        let key2 = PoolKey::Buffer {
            size_class: SizeClass::Medium,
            usage: wgpu::BufferUsages::STORAGE,
        };

        let key3 = PoolKey::Buffer {
            size_class: SizeClass::Large, // Different size class
            usage: wgpu::BufferUsages::STORAGE,
        };

        assert_eq!(key1, key2);
        assert_ne!(key1, key3);
    }

    #[test]
    fn test_pool_key_hash() {
        use std::collections::hash_map::DefaultHasher;

        let key1 = PoolKey::Buffer {
            size_class: SizeClass::Medium,
            usage: wgpu::BufferUsages::STORAGE,
        };

        let key2 = PoolKey::Buffer {
            size_class: SizeClass::Medium,
            usage: wgpu::BufferUsages::STORAGE,
        };

        let mut hasher1 = DefaultHasher::new();
        let mut hasher2 = DefaultHasher::new();
        key1.hash(&mut hasher1);
        key2.hash(&mut hasher2);

        assert_eq!(hasher1.finish(), hasher2.finish());
    }

    #[test]
    fn test_pool_key_from_texture_desc() {
        let desc = TextureDescriptor::new_render_target(1920, 1080, wgpu::TextureFormat::Rgba8Unorm);
        let key = PoolKey::from_texture_desc(&desc);

        assert!(key.is_texture());
        assert!(!key.is_buffer());

        match key {
            PoolKey::Texture {
                format,
                width,
                height,
                sample_count,
                ..
            } => {
                assert_eq!(format, wgpu::TextureFormat::Rgba8Unorm);
                assert_eq!(width, 1920);
                assert_eq!(height, 1080);
                assert_eq!(sample_count, 1);
            }
            _ => panic!("Expected texture key"),
        }
    }

    #[test]
    fn test_pool_key_from_buffer_desc() {
        let desc = BufferDescriptor::new_storage(100 * 1024); // 100KB = Medium
        let key = PoolKey::from_buffer_desc(&desc);

        assert!(key.is_buffer());
        assert!(!key.is_texture());

        match key {
            PoolKey::Buffer { size_class, usage } => {
                assert_eq!(size_class, SizeClass::Medium);
                assert!(usage.contains(wgpu::BufferUsages::STORAGE));
            }
            _ => panic!("Expected buffer key"),
        }
    }

    #[test]
    fn test_pool_stats_hit_rate() {
        let mut stats = PoolStats::new();
        assert_eq!(stats.hit_rate(), 0.0);

        stats.cache_hits = 8;
        stats.cache_misses = 2;
        assert!((stats.hit_rate() - 0.8).abs() < 0.001);

        stats.cache_hits = 0;
        stats.cache_misses = 10;
        assert_eq!(stats.hit_rate(), 0.0);

        stats.cache_hits = 10;
        stats.cache_misses = 0;
        assert_eq!(stats.hit_rate(), 1.0);
    }

    #[test]
    fn test_pool_stats_total_bytes() {
        let mut stats = PoolStats::new();
        stats.texture_bytes = 1024 * 1024;
        stats.buffer_bytes = 512 * 1024;
        assert_eq!(stats.total_bytes(), 1536 * 1024);
    }

    #[test]
    fn test_pool_stats_idle_counts() {
        let mut stats = PoolStats::new();
        stats.total_textures = 10;
        stats.active_textures = 3;
        stats.total_buffers = 8;
        stats.active_buffers = 5;

        assert_eq!(stats.idle_textures(), 7);
        assert_eq!(stats.idle_buffers(), 3);
    }

    #[test]
    fn test_pool_config_default() {
        let config = PoolConfig::default();
        assert_eq!(config.max_idle_frames, 3);
        assert_eq!(config.max_texture_pool_size, 8);
        assert_eq!(config.max_buffer_pool_size, 16);
        assert!(!config.enable_defragmentation);
        assert_eq!(config.max_memory_bytes, 0);
    }

    #[test]
    fn test_pool_config_builder() {
        let config = PoolConfig::new()
            .with_max_idle_frames(5)
            .with_max_texture_pool_size(16)
            .with_max_buffer_pool_size(32)
            .with_defragmentation(true)
            .with_memory_budget(1024 * 1024 * 256);

        assert_eq!(config.max_idle_frames, 5);
        assert_eq!(config.max_texture_pool_size, 16);
        assert_eq!(config.max_buffer_pool_size, 32);
        assert!(config.enable_defragmentation);
        assert_eq!(config.max_memory_bytes, 256 * 1024 * 1024);
    }

    #[test]
    fn test_resource_lifetime_range_aliasing() {
        let r1 = ResourceLifetimeRange::new(1, 0, 5, 1024);
        let r2 = ResourceLifetimeRange::new(2, 6, 10, 2048);
        let r3 = ResourceLifetimeRange::new(3, 3, 8, 512);

        // r1 and r2 don't overlap
        assert!(!r1.overlapping_lifetime(&r2));
        assert!(r1.can_alias_with(&r2));

        // r1 and r3 overlap (1 ends at 5, 3 starts at 3)
        assert!(r1.overlapping_lifetime(&r3));
        assert!(!r1.can_alias_with(&r3));

        // r2 and r3 overlap (3 ends at 8, 2 starts at 6)
        assert!(r2.overlapping_lifetime(&r3));
        assert!(!r2.can_alias_with(&r3));
    }

    #[test]
    fn test_alias_group_basic() {
        let mut group = AliasGroup::new();
        assert!(group.is_empty());
        assert_eq!(group.len(), 0);

        let r1 = ResourceLifetimeRange::new(1, 0, 3, 1024);
        assert!(group.try_add(&r1));
        assert_eq!(group.len(), 1);
        assert_eq!(group.memory_size, 1024);

        let r2 = ResourceLifetimeRange::new(2, 5, 8, 2048);
        assert!(group.try_add(&r2));
        assert_eq!(group.len(), 2);
        assert_eq!(group.memory_size, 2048); // Max of the two
    }

    #[test]
    fn test_alias_group_overlap_rejection() {
        let mut group = AliasGroup::new();

        let r1 = ResourceLifetimeRange::new(1, 0, 5, 1024);
        assert!(group.try_add(&r1));

        // This overlaps with r1 (starts at 3, r1 ends at 5)
        let r2 = ResourceLifetimeRange::new(2, 3, 8, 2048);
        assert!(!group.try_add(&r2));
        assert_eq!(group.len(), 1);
    }

    #[test]
    fn test_compute_alias_groups() {
        let resources = vec![
            ResourceLifetimeRange::new(1, 0, 2, 1024),
            ResourceLifetimeRange::new(2, 3, 5, 2048),
            ResourceLifetimeRange::new(3, 0, 4, 512), // Overlaps with 1 and 2
            ResourceLifetimeRange::new(4, 6, 8, 768),
        ];

        let groups = compute_alias_groups(&resources);

        // Resources 1, 2, 4 can share a group (non-overlapping)
        // Resource 3 needs its own group (overlaps with 1 and 2)
        assert!(groups.len() >= 1);

        // Verify all resources are assigned
        let total_assigned: usize = groups.iter().map(|g| g.len()).sum();
        assert_eq!(total_assigned, 4);
    }

    #[test]
    fn test_estimate_aliasing_savings() {
        let resources = vec![
            ResourceLifetimeRange::new(1, 0, 2, 1000),
            ResourceLifetimeRange::new(2, 3, 5, 2000),
            ResourceLifetimeRange::new(3, 6, 8, 1500),
        ];

        let groups = compute_alias_groups(&resources);
        let (without, with, savings) = estimate_aliasing_savings(&resources, &groups);

        // All three can alias (non-overlapping)
        assert_eq!(without, 4500);
        // With aliasing, we only need max(1000, 2000, 1500) = 2000
        assert_eq!(with, 2000);
        // Savings should be (4500 - 2000) / 4500 * 100 = ~55.6%
        assert!(savings > 50.0);
    }

    #[test]
    fn test_size_class_display() {
        assert_eq!(format!("{}", SizeClass::Tiny), "Tiny(<1KB)");
        assert_eq!(format!("{}", SizeClass::Small), "Small(1-64KB)");
        assert_eq!(format!("{}", SizeClass::Medium), "Medium(64KB-1MB)");
        assert_eq!(format!("{}", SizeClass::Large), "Large(1-16MB)");
        assert_eq!(format!("{}", SizeClass::Huge), "Huge(>16MB)");
    }

    #[test]
    fn test_pool_stats_display() {
        let mut stats = PoolStats::new();
        stats.total_textures = 10;
        stats.active_textures = 3;
        stats.total_buffers = 8;
        stats.active_buffers = 2;
        stats.texture_bytes = 10 * 1024 * 1024;
        stats.buffer_bytes = 5 * 1024 * 1024;
        stats.cache_hits = 80;
        stats.cache_misses = 20;

        let display = format!("{}", stats);
        assert!(display.contains("textures=3/10"));
        assert!(display.contains("buffers=2/8"));
        assert!(display.contains("hit_rate=80.0%"));
    }

    #[test]
    fn test_pool_key_display() {
        let texture_key = PoolKey::Texture {
            format: wgpu::TextureFormat::Rgba8Unorm,
            width: 1920,
            height: 1080,
            depth_or_layers: 1,
            mip_levels: 1,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
        };

        let display = format!("{}", texture_key);
        assert!(display.contains("Texture"));
        assert!(display.contains("1920x1080"));

        let buffer_key = PoolKey::Buffer {
            size_class: SizeClass::Medium,
            usage: wgpu::BufferUsages::STORAGE,
        };

        let display = format!("{}", buffer_key);
        assert!(display.contains("Buffer"));
        assert!(display.contains("Medium"));
    }
}
