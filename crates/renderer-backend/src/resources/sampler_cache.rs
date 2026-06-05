//! Sampler caching for TRINITY.
//!
//! This module provides a cache for wgpu samplers to avoid redundant GPU resource
//! creation. Samplers are identified by a hash key derived from their descriptor,
//! and are shared via `Arc<wgpu::Sampler>`.
//!
//! # Overview
//!
//! GPU samplers are relatively cheap to create, but redundant creation wastes
//! resources and complicates resource management. This cache:
//!
//! - Deduplicates samplers with identical configuration
//! - Provides preset samplers for common use cases
//! - Tracks cache hit/miss metrics
//! - Uses thread-safe interior mutability for concurrent access
//!
//! # Thread Safety
//!
//! The cache uses `RwLock` for concurrent access:
//! - Multiple readers can query the cache simultaneously
//! - Write lock is only held briefly when creating new samplers
//!
//! # Presets
//!
//! Common sampler configurations are pre-created and accessible via dedicated methods:
//!
//! | Preset | Description |
//! |--------|-------------|
//! | `linear_clamp` | Linear filtering, clamp to edge (default for most textures) |
//! | `linear_repeat` | Linear filtering, repeat wrapping (tiled textures) |
//! | `point_clamp` | Nearest filtering, clamp to edge (pixel art, data textures) |
//! | `point_repeat` | Nearest filtering, repeat wrapping (tiled pixel art) |
//! | `shadow` | Comparison sampler for shadow mapping |
//!
//! # Example
//!
//! ```no_run
//! use renderer_backend::resources::sampler_cache::SamplerCache;
//! use renderer_backend::resources::sampler::TrinitySamplerDescriptor;
//! use std::sync::Arc;
//! use wgpu::{FilterMode, AddressMode};
//!
//! # fn example(device: Arc<wgpu::Device>) {
//! let cache = SamplerCache::new(device);
//!
//! // Get a preset sampler (no cache lookup needed)
//! let linear = cache.linear_clamp();
//!
//! // Get or create a custom sampler
//! let desc = TrinitySamplerDescriptor::new()
//!     .filter(FilterMode::Linear)
//!     .address_mode(AddressMode::MirrorRepeat)
//!     .anisotropy(8);
//! let custom = cache.get_or_create(&desc);
//!
//! // Same descriptor returns same Arc
//! let custom2 = cache.get_or_create(&desc);
//! assert!(Arc::ptr_eq(&custom, &custom2));
//!
//! // Check cache metrics
//! let metrics = cache.metrics();
//! println!("Cache size: {}, Hit rate: {:.1}%", metrics.cache_size, metrics.hit_rate * 100.0);
//! # }
//! ```

use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use wgpu::{AddressMode, CompareFunction, Device, FilterMode, Sampler, SamplerBorderColor};

use super::sampler::{create_sampler, TrinitySamplerDescriptor};

// ============================================================================
// SamplerCacheKey
// ============================================================================

/// A hashable key derived from a sampler descriptor.
///
/// This struct compactly encodes all sampler parameters that affect GPU behavior.
/// It's used as the key in the sampler cache's hash map.
///
/// # Implementation Notes
///
/// - Enum variants are stored as `u8` for compact representation
/// - Floating-point LOD values use `f32::to_bits()` for exact hashing
/// - `Option` types use `None` as a sentinel value
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SamplerCacheKey {
    /// U address mode as compact u8
    address_mode_u: u8,
    /// V address mode as compact u8
    address_mode_v: u8,
    /// W address mode as compact u8
    address_mode_w: u8,
    /// Magnification filter as u8
    mag_filter: u8,
    /// Minification filter as u8
    min_filter: u8,
    /// Mipmap filter as u8
    mipmap_filter: u8,
    /// Comparison function (0 = None, 1-8 = function variants)
    compare: u8,
    /// Maximum anisotropy level
    anisotropy: u16,
    /// Border color (0 = None, 1-3 = color variants)
    border_color: u8,
    /// LOD minimum clamp as bit representation
    lod_min_clamp_bits: u32,
    /// LOD maximum clamp as bit representation
    lod_max_clamp_bits: u32,
}

impl SamplerCacheKey {
    /// Creates a cache key from a sampler descriptor.
    ///
    /// All descriptor fields are encoded into a compact, hashable representation.
    ///
    /// # Arguments
    ///
    /// * `desc` - The sampler descriptor to convert
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::resources::sampler::TrinitySamplerDescriptor;
    /// use renderer_backend::resources::sampler_cache::SamplerCacheKey;
    ///
    /// let desc = TrinitySamplerDescriptor::linear_clamp();
    /// let key = SamplerCacheKey::from_descriptor(&desc);
    ///
    /// // Same descriptor produces same key
    /// let key2 = SamplerCacheKey::from_descriptor(&desc);
    /// assert_eq!(key, key2);
    /// ```
    pub fn from_descriptor(desc: &TrinitySamplerDescriptor) -> Self {
        Self {
            address_mode_u: address_mode_to_u8(desc.address_mode_u),
            address_mode_v: address_mode_to_u8(desc.address_mode_v),
            address_mode_w: address_mode_to_u8(desc.address_mode_w),
            mag_filter: filter_mode_to_u8(desc.mag_filter),
            min_filter: filter_mode_to_u8(desc.min_filter),
            mipmap_filter: filter_mode_to_u8(desc.mipmap_filter),
            compare: compare_function_to_u8(desc.compare),
            anisotropy: desc.anisotropy_clamp,
            border_color: border_color_to_u8(desc.border_color),
            lod_min_clamp_bits: desc.lod_min_clamp.to_bits(),
            lod_max_clamp_bits: desc.lod_max_clamp.to_bits(),
        }
    }

    /// Returns the address mode U value.
    #[inline]
    pub fn address_mode_u(&self) -> u8 {
        self.address_mode_u
    }

    /// Returns the address mode V value.
    #[inline]
    pub fn address_mode_v(&self) -> u8 {
        self.address_mode_v
    }

    /// Returns the address mode W value.
    #[inline]
    pub fn address_mode_w(&self) -> u8 {
        self.address_mode_w
    }

    /// Returns the magnification filter value.
    #[inline]
    pub fn mag_filter(&self) -> u8 {
        self.mag_filter
    }

    /// Returns the minification filter value.
    #[inline]
    pub fn min_filter(&self) -> u8 {
        self.min_filter
    }

    /// Returns the mipmap filter value.
    #[inline]
    pub fn mipmap_filter(&self) -> u8 {
        self.mipmap_filter
    }

    /// Returns the compare function value.
    #[inline]
    pub fn compare(&self) -> u8 {
        self.compare
    }

    /// Returns the anisotropy value.
    #[inline]
    pub fn anisotropy(&self) -> u16 {
        self.anisotropy
    }

    /// Returns the border color value.
    #[inline]
    pub fn border_color(&self) -> u8 {
        self.border_color
    }

    /// Returns the LOD minimum clamp as bits.
    #[inline]
    pub fn lod_min_clamp_bits(&self) -> u32 {
        self.lod_min_clamp_bits
    }

    /// Returns the LOD maximum clamp as bits.
    #[inline]
    pub fn lod_max_clamp_bits(&self) -> u32 {
        self.lod_max_clamp_bits
    }
}

// ============================================================================
// Enum Conversion Helpers
// ============================================================================

/// Converts AddressMode to compact u8.
#[inline]
fn address_mode_to_u8(mode: AddressMode) -> u8 {
    match mode {
        AddressMode::ClampToEdge => 0,
        AddressMode::Repeat => 1,
        AddressMode::MirrorRepeat => 2,
        AddressMode::ClampToBorder => 3,
    }
}

/// Converts FilterMode to compact u8.
#[inline]
fn filter_mode_to_u8(mode: FilterMode) -> u8 {
    match mode {
        FilterMode::Nearest => 0,
        FilterMode::Linear => 1,
    }
}

/// Converts Option<CompareFunction> to compact u8.
#[inline]
fn compare_function_to_u8(func: Option<CompareFunction>) -> u8 {
    match func {
        None => 0,
        Some(CompareFunction::Never) => 1,
        Some(CompareFunction::Less) => 2,
        Some(CompareFunction::Equal) => 3,
        Some(CompareFunction::LessEqual) => 4,
        Some(CompareFunction::Greater) => 5,
        Some(CompareFunction::NotEqual) => 6,
        Some(CompareFunction::GreaterEqual) => 7,
        Some(CompareFunction::Always) => 8,
    }
}

/// Converts Option<SamplerBorderColor> to compact u8.
#[inline]
fn border_color_to_u8(color: Option<SamplerBorderColor>) -> u8 {
    match color {
        None => 0,
        Some(SamplerBorderColor::TransparentBlack) => 1,
        Some(SamplerBorderColor::OpaqueBlack) => 2,
        Some(SamplerBorderColor::OpaqueWhite) => 3,
        Some(SamplerBorderColor::Zero) => 4,
    }
}

// ============================================================================
// CachedSampler
// ============================================================================

/// A reference-counted sampler that can be shared across multiple users.
///
/// This is an `Arc<wgpu::Sampler>` allowing multiple bind groups or materials
/// to reference the same GPU sampler without duplication.
pub type CachedSampler = Arc<Sampler>;

// ============================================================================
// SamplerCacheMetrics
// ============================================================================

/// Metrics for monitoring sampler cache performance.
///
/// These metrics help identify cache efficiency and potential optimization
/// opportunities.
///
/// # Example
///
/// ```
/// use renderer_backend::resources::sampler_cache::SamplerCacheMetrics;
///
/// let metrics = SamplerCacheMetrics::default();
/// assert_eq!(metrics.cache_size, 0);
/// assert_eq!(metrics.hits, 0);
/// assert_eq!(metrics.misses, 0);
/// assert_eq!(metrics.hit_rate, 0.0);
/// ```
#[derive(Debug, Clone, Default)]
pub struct SamplerCacheMetrics {
    /// Number of unique samplers in the cache (excluding presets).
    pub cache_size: usize,
    /// Number of cache hits (requested sampler already existed).
    pub hits: u64,
    /// Number of cache misses (new sampler created).
    pub misses: u64,
    /// Hit rate as a ratio (0.0 to 1.0).
    pub hit_rate: f64,
}

impl SamplerCacheMetrics {
    /// Creates metrics with the given values.
    pub fn new(cache_size: usize, hits: u64, misses: u64) -> Self {
        let total = hits + misses;
        let hit_rate = if total > 0 {
            hits as f64 / total as f64
        } else {
            0.0
        };

        Self {
            cache_size,
            hits,
            misses,
            hit_rate,
        }
    }

    /// Returns the total number of requests (hits + misses).
    #[inline]
    pub fn total_requests(&self) -> u64 {
        self.hits + self.misses
    }

    /// Returns true if the cache is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.cache_size == 0
    }

    /// Returns the hit rate as a percentage (0.0 to 100.0).
    #[inline]
    pub fn hit_rate_percent(&self) -> f64 {
        self.hit_rate * 100.0
    }
}

// ============================================================================
// SamplerCache
// ============================================================================

/// A thread-safe cache for wgpu samplers.
///
/// The cache stores samplers keyed by their configuration, ensuring that
/// identical sampler configurations share the same GPU resource.
///
/// # Architecture
///
/// ```text
/// SamplerCache
/// ├── Presets (created on construction)
/// │   ├── linear_clamp
/// │   ├── linear_repeat
/// │   ├── point_clamp
/// │   ├── point_repeat
/// │   └── shadow
/// ├── Cache (HashMap<SamplerCacheKey, CachedSampler>)
/// │   └── User-created samplers
/// └── Metrics (hits, misses, size)
/// ```
///
/// # Thread Safety
///
/// - Uses `RwLock<HashMap>` for the cache
/// - Uses `AtomicU64` for hit/miss counters (lock-free)
/// - Multiple readers can access presets and cached samplers concurrently
/// - Write lock is only held briefly when inserting new samplers
///
/// # Example
///
/// ```no_run
/// use renderer_backend::resources::sampler_cache::SamplerCache;
/// use renderer_backend::resources::sampler::TrinitySamplerDescriptor;
/// use std::sync::Arc;
///
/// # fn example(device: Arc<wgpu::Device>) {
/// let cache = SamplerCache::new(device);
///
/// // Presets are instant (no cache lookup)
/// let shadow_sampler = cache.shadow();
///
/// // Custom samplers are cached
/// let desc = TrinitySamplerDescriptor::trilinear().anisotropy(8);
/// let sampler1 = cache.get_or_create(&desc);
/// let sampler2 = cache.get_or_create(&desc); // Cache hit
/// assert!(Arc::ptr_eq(&sampler1, &sampler2));
/// # }
/// ```
pub struct SamplerCache {
    /// The wgpu device for creating new samplers.
    device: Arc<Device>,
    /// Cache of user-created samplers.
    cache: RwLock<HashMap<SamplerCacheKey, CachedSampler>>,
    /// Hit counter (atomic for lock-free updates).
    hits: AtomicU64,
    /// Miss counter (atomic for lock-free updates).
    misses: AtomicU64,

    // Preset samplers (created on construction)
    /// Linear filtering with clamp to edge.
    linear_clamp: CachedSampler,
    /// Linear filtering with repeat wrapping.
    linear_repeat: CachedSampler,
    /// Nearest/point filtering with clamp to edge.
    point_clamp: CachedSampler,
    /// Nearest/point filtering with repeat wrapping.
    point_repeat: CachedSampler,
    /// Shadow mapping comparison sampler.
    shadow: CachedSampler,
}

impl SamplerCache {
    /// Creates a new sampler cache with preset samplers.
    ///
    /// Five preset samplers are created immediately:
    /// - `linear_clamp`: Linear filtering, clamp to edge
    /// - `linear_repeat`: Linear filtering, repeat wrapping
    /// - `point_clamp`: Nearest filtering, clamp to edge
    /// - `point_repeat`: Nearest filtering, repeat wrapping
    /// - `shadow`: Comparison sampler for shadow mapping
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device for creating samplers
    ///
    /// # Example
    ///
    /// ```no_run
    /// use renderer_backend::resources::sampler_cache::SamplerCache;
    /// use std::sync::Arc;
    ///
    /// # fn example(device: Arc<wgpu::Device>) {
    /// let cache = SamplerCache::new(device);
    /// assert_eq!(cache.len(), 0); // Presets don't count toward cache size
    /// # }
    /// ```
    pub fn new(device: Arc<Device>) -> Self {
        // Create preset samplers
        let linear_clamp = Arc::new(
            create_sampler(&device, &TrinitySamplerDescriptor::linear_clamp()).into_inner(),
        );
        let linear_repeat = Arc::new(
            create_sampler(&device, &TrinitySamplerDescriptor::linear_repeat()).into_inner(),
        );
        let point_clamp = Arc::new(
            create_sampler(&device, &TrinitySamplerDescriptor::nearest_clamp()).into_inner(),
        );
        let point_repeat = Arc::new(
            create_sampler(&device, &TrinitySamplerDescriptor::nearest_repeat()).into_inner(),
        );
        let shadow = Arc::new(
            create_sampler(&device, &TrinitySamplerDescriptor::shadow()).into_inner(),
        );

        Self {
            device,
            cache: RwLock::new(HashMap::new()),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            linear_clamp,
            linear_repeat,
            point_clamp,
            point_repeat,
            shadow,
        }
    }

    /// Gets or creates a sampler matching the descriptor.
    ///
    /// If a sampler with the same configuration already exists in the cache,
    /// it is returned. Otherwise, a new sampler is created, cached, and returned.
    ///
    /// # Arguments
    ///
    /// * `desc` - The sampler descriptor
    ///
    /// # Returns
    ///
    /// An `Arc<Sampler>` that can be shared across multiple bind groups.
    ///
    /// # Thread Safety
    ///
    /// This method uses a read lock for cache lookups and only acquires
    /// a write lock when creating a new sampler.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use renderer_backend::resources::sampler_cache::SamplerCache;
    /// use renderer_backend::resources::sampler::TrinitySamplerDescriptor;
    /// use std::sync::Arc;
    /// use wgpu::AddressMode;
    ///
    /// # fn example(device: Arc<wgpu::Device>) {
    /// let cache = SamplerCache::new(device);
    ///
    /// let desc = TrinitySamplerDescriptor::new()
    ///     .address_mode(AddressMode::MirrorRepeat)
    ///     .anisotropy(4);
    ///
    /// let sampler1 = cache.get_or_create(&desc);
    /// let sampler2 = cache.get_or_create(&desc);
    ///
    /// // Same Arc (cache hit)
    /// assert!(Arc::ptr_eq(&sampler1, &sampler2));
    /// # }
    /// ```
    pub fn get_or_create(&self, desc: &TrinitySamplerDescriptor) -> CachedSampler {
        let key = SamplerCacheKey::from_descriptor(desc);

        // Try read lock first for cache hit
        {
            let cache = self.cache.read();
            if let Some(sampler) = cache.get(&key) {
                self.hits.fetch_add(1, Ordering::Relaxed);
                return Arc::clone(sampler);
            }
        }

        // Cache miss - acquire write lock
        let mut cache = self.cache.write();

        // Double-check in case another thread inserted while we were waiting
        if let Some(sampler) = cache.get(&key) {
            self.hits.fetch_add(1, Ordering::Relaxed);
            return Arc::clone(sampler);
        }

        // Create and insert new sampler
        self.misses.fetch_add(1, Ordering::Relaxed);
        let sampler = Arc::new(create_sampler(&self.device, desc).into_inner());
        cache.insert(key, Arc::clone(&sampler));
        sampler
    }

    /// Returns the linear clamp preset sampler.
    ///
    /// Configuration:
    /// - Filter: Linear (mag, min, mipmap)
    /// - Address: ClampToEdge (U, V, W)
    ///
    /// Best for: Most textures (smooth, no wrapping).
    ///
    /// # Example
    ///
    /// ```no_run
    /// use renderer_backend::resources::sampler_cache::SamplerCache;
    /// use std::sync::Arc;
    ///
    /// # fn example(device: Arc<wgpu::Device>) {
    /// let cache = SamplerCache::new(device);
    /// let sampler = cache.linear_clamp();
    /// // Use sampler in bind group...
    /// # }
    /// ```
    #[inline]
    pub fn linear_clamp(&self) -> CachedSampler {
        Arc::clone(&self.linear_clamp)
    }

    /// Returns the linear repeat preset sampler.
    ///
    /// Configuration:
    /// - Filter: Linear (mag, min, mipmap)
    /// - Address: Repeat (U, V, W)
    ///
    /// Best for: Tiled/repeating textures (floors, walls).
    ///
    /// # Example
    ///
    /// ```no_run
    /// use renderer_backend::resources::sampler_cache::SamplerCache;
    /// use std::sync::Arc;
    ///
    /// # fn example(device: Arc<wgpu::Device>) {
    /// let cache = SamplerCache::new(device);
    /// let sampler = cache.linear_repeat();
    /// // Use sampler in bind group...
    /// # }
    /// ```
    #[inline]
    pub fn linear_repeat(&self) -> CachedSampler {
        Arc::clone(&self.linear_repeat)
    }

    /// Returns the point/nearest clamp preset sampler.
    ///
    /// Configuration:
    /// - Filter: Nearest (mag, min, mipmap)
    /// - Address: ClampToEdge (U, V, W)
    ///
    /// Best for: Pixel art, data textures, or when sharp texel boundaries
    /// are desired.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use renderer_backend::resources::sampler_cache::SamplerCache;
    /// use std::sync::Arc;
    ///
    /// # fn example(device: Arc<wgpu::Device>) {
    /// let cache = SamplerCache::new(device);
    /// let sampler = cache.point_clamp();
    /// // Use sampler for pixel art rendering...
    /// # }
    /// ```
    #[inline]
    pub fn point_clamp(&self) -> CachedSampler {
        Arc::clone(&self.point_clamp)
    }

    /// Returns the point/nearest repeat preset sampler.
    ///
    /// Configuration:
    /// - Filter: Nearest (mag, min, mipmap)
    /// - Address: Repeat (U, V, W)
    ///
    /// Best for: Tiled pixel art.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use renderer_backend::resources::sampler_cache::SamplerCache;
    /// use std::sync::Arc;
    ///
    /// # fn example(device: Arc<wgpu::Device>) {
    /// let cache = SamplerCache::new(device);
    /// let sampler = cache.point_repeat();
    /// // Use sampler for tiled pixel art...
    /// # }
    /// ```
    #[inline]
    pub fn point_repeat(&self) -> CachedSampler {
        Arc::clone(&self.point_repeat)
    }

    /// Returns the shadow mapping preset sampler.
    ///
    /// Configuration:
    /// - Filter: Linear (enables PCF when supported)
    /// - Address: ClampToEdge
    /// - Compare: Less
    /// - Mipmap: Nearest
    ///
    /// Best for: Shadow mapping with depth textures.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use renderer_backend::resources::sampler_cache::SamplerCache;
    /// use std::sync::Arc;
    ///
    /// # fn example(device: Arc<wgpu::Device>) {
    /// let cache = SamplerCache::new(device);
    /// let sampler = cache.shadow();
    /// // Use sampler with shadow map depth texture...
    /// # }
    /// ```
    #[inline]
    pub fn shadow(&self) -> CachedSampler {
        Arc::clone(&self.shadow)
    }

    /// Returns current cache metrics.
    ///
    /// Metrics include cache size, hit count, miss count, and hit rate.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use renderer_backend::resources::sampler_cache::SamplerCache;
    /// use std::sync::Arc;
    ///
    /// # fn example(device: Arc<wgpu::Device>) {
    /// let cache = SamplerCache::new(device);
    ///
    /// // After some usage...
    /// let metrics = cache.metrics();
    /// println!("Cache size: {}", metrics.cache_size);
    /// println!("Hit rate: {:.1}%", metrics.hit_rate_percent());
    /// # }
    /// ```
    pub fn metrics(&self) -> SamplerCacheMetrics {
        let cache_size = self.cache.read().len();
        let hits = self.hits.load(Ordering::Relaxed);
        let misses = self.misses.load(Ordering::Relaxed);

        SamplerCacheMetrics::new(cache_size, hits, misses)
    }

    /// Clears the cache but keeps preset samplers.
    ///
    /// This removes all user-created samplers from the cache.
    /// Preset samplers (`linear_clamp`, `linear_repeat`, etc.) are not affected.
    ///
    /// Metrics (hits/misses) are also reset to zero.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use renderer_backend::resources::sampler_cache::SamplerCache;
    /// use renderer_backend::resources::sampler::TrinitySamplerDescriptor;
    /// use std::sync::Arc;
    ///
    /// # fn example(device: Arc<wgpu::Device>) {
    /// let cache = SamplerCache::new(device);
    ///
    /// // Add some samplers
    /// cache.get_or_create(&TrinitySamplerDescriptor::trilinear());
    /// assert_eq!(cache.len(), 1);
    ///
    /// // Clear cache
    /// cache.clear();
    /// assert_eq!(cache.len(), 0);
    ///
    /// // Presets still available
    /// let _ = cache.linear_clamp();
    /// # }
    /// ```
    pub fn clear(&self) {
        self.cache.write().clear();
        self.hits.store(0, Ordering::Relaxed);
        self.misses.store(0, Ordering::Relaxed);
    }

    /// Returns the number of cached samplers (excluding presets).
    ///
    /// # Example
    ///
    /// ```no_run
    /// use renderer_backend::resources::sampler_cache::SamplerCache;
    /// use std::sync::Arc;
    ///
    /// # fn example(device: Arc<wgpu::Device>) {
    /// let cache = SamplerCache::new(device);
    /// assert_eq!(cache.len(), 0); // Presets don't count
    /// # }
    /// ```
    #[inline]
    pub fn len(&self) -> usize {
        self.cache.read().len()
    }

    /// Returns true if the cache is empty (excluding presets).
    ///
    /// # Example
    ///
    /// ```no_run
    /// use renderer_backend::resources::sampler_cache::SamplerCache;
    /// use std::sync::Arc;
    ///
    /// # fn example(device: Arc<wgpu::Device>) {
    /// let cache = SamplerCache::new(device);
    /// assert!(cache.is_empty());
    /// # }
    /// ```
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.cache.read().is_empty()
    }

    /// Returns a reference to the underlying device.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use renderer_backend::resources::sampler_cache::SamplerCache;
    /// use std::sync::Arc;
    ///
    /// # fn example(device: Arc<wgpu::Device>) {
    /// let cache = SamplerCache::new(Arc::clone(&device));
    /// let device_ref = cache.device();
    /// // Use device for other operations...
    /// # }
    /// ```
    #[inline]
    pub fn device(&self) -> &Arc<Device> {
        &self.device
    }

    // ========================================================================
    // Convenience Methods
    // ========================================================================

    /// Shorthand for linear filtering with clamp to edge.
    ///
    /// Equivalent to [`linear_clamp()`](Self::linear_clamp).
    ///
    /// # Example
    ///
    /// ```no_run
    /// use renderer_backend::resources::sampler_cache::SamplerCache;
    /// use std::sync::Arc;
    ///
    /// # fn example(device: Arc<wgpu::Device>) {
    /// let cache = SamplerCache::new(device);
    /// let sampler = cache.get_linear();
    /// # }
    /// ```
    #[inline]
    pub fn get_linear(&self) -> CachedSampler {
        self.linear_clamp()
    }

    /// Shorthand for point/nearest filtering with clamp to edge.
    ///
    /// Equivalent to [`point_clamp()`](Self::point_clamp).
    ///
    /// # Example
    ///
    /// ```no_run
    /// use renderer_backend::resources::sampler_cache::SamplerCache;
    /// use std::sync::Arc;
    ///
    /// # fn example(device: Arc<wgpu::Device>) {
    /// let cache = SamplerCache::new(device);
    /// let sampler = cache.get_point();
    /// # }
    /// ```
    #[inline]
    pub fn get_point(&self) -> CachedSampler {
        self.point_clamp()
    }

    /// Returns the number of preset samplers (always 5).
    ///
    /// # Example
    ///
    /// ```no_run
    /// use renderer_backend::resources::sampler_cache::SamplerCache;
    /// use std::sync::Arc;
    ///
    /// # fn example(device: Arc<wgpu::Device>) {
    /// let cache = SamplerCache::new(device);
    /// assert_eq!(cache.preset_count(), 5);
    /// # }
    /// ```
    #[inline]
    pub fn preset_count(&self) -> usize {
        5
    }

    /// Returns the total number of samplers (cached + presets).
    ///
    /// # Example
    ///
    /// ```no_run
    /// use renderer_backend::resources::sampler_cache::SamplerCache;
    /// use renderer_backend::resources::sampler::TrinitySamplerDescriptor;
    /// use std::sync::Arc;
    ///
    /// # fn example(device: Arc<wgpu::Device>) {
    /// let cache = SamplerCache::new(device);
    /// assert_eq!(cache.total_count(), 5); // Just presets
    ///
    /// cache.get_or_create(&TrinitySamplerDescriptor::trilinear());
    /// assert_eq!(cache.total_count(), 6); // Presets + 1
    /// # }
    /// ```
    #[inline]
    pub fn total_count(&self) -> usize {
        self.len() + self.preset_count()
    }
}

impl std::fmt::Debug for SamplerCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let metrics = self.metrics();
        f.debug_struct("SamplerCache")
            .field("cache_size", &metrics.cache_size)
            .field("preset_count", &self.preset_count())
            .field("hits", &metrics.hits)
            .field("misses", &metrics.misses)
            .field("hit_rate", &format!("{:.1}%", metrics.hit_rate_percent()))
            .finish()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // SamplerCacheKey Tests
    // ========================================================================

    #[test]
    fn test_key_from_default_descriptor() {
        let desc = TrinitySamplerDescriptor::default();
        let key = SamplerCacheKey::from_descriptor(&desc);

        // Default is linear, clamp to edge
        assert_eq!(key.address_mode_u, 0); // ClampToEdge
        assert_eq!(key.address_mode_v, 0);
        assert_eq!(key.address_mode_w, 0);
        assert_eq!(key.mag_filter, 1); // Linear
        assert_eq!(key.min_filter, 1);
        assert_eq!(key.mipmap_filter, 1);
        assert_eq!(key.compare, 0); // None
        assert_eq!(key.anisotropy, 1);
        assert_eq!(key.border_color, 0); // None
    }

    #[test]
    fn test_key_equality_same_descriptor() {
        let desc1 = TrinitySamplerDescriptor::linear_clamp();
        let desc2 = TrinitySamplerDescriptor::linear_clamp();

        let key1 = SamplerCacheKey::from_descriptor(&desc1);
        let key2 = SamplerCacheKey::from_descriptor(&desc2);

        assert_eq!(key1, key2);
    }

    #[test]
    fn test_key_inequality_different_filter() {
        let desc1 = TrinitySamplerDescriptor::linear_clamp();
        let desc2 = TrinitySamplerDescriptor::nearest_clamp();

        let key1 = SamplerCacheKey::from_descriptor(&desc1);
        let key2 = SamplerCacheKey::from_descriptor(&desc2);

        assert_ne!(key1, key2);
    }

    #[test]
    fn test_key_inequality_different_address_mode() {
        let desc1 = TrinitySamplerDescriptor::linear_clamp();
        let desc2 = TrinitySamplerDescriptor::linear_repeat();

        let key1 = SamplerCacheKey::from_descriptor(&desc1);
        let key2 = SamplerCacheKey::from_descriptor(&desc2);

        assert_ne!(key1, key2);
    }

    #[test]
    fn test_key_inequality_different_anisotropy() {
        let desc1 = TrinitySamplerDescriptor::new().anisotropy(1);
        let desc2 = TrinitySamplerDescriptor::new().anisotropy(8);

        let key1 = SamplerCacheKey::from_descriptor(&desc1);
        let key2 = SamplerCacheKey::from_descriptor(&desc2);

        assert_ne!(key1, key2);
    }

    #[test]
    fn test_key_inequality_different_compare() {
        let desc1 = TrinitySamplerDescriptor::new();
        let desc2 = TrinitySamplerDescriptor::shadow();

        let key1 = SamplerCacheKey::from_descriptor(&desc1);
        let key2 = SamplerCacheKey::from_descriptor(&desc2);

        assert_ne!(key1, key2);
    }

    #[test]
    fn test_key_inequality_different_lod() {
        let desc1 = TrinitySamplerDescriptor::new().lod_clamp(0.0, 32.0);
        let desc2 = TrinitySamplerDescriptor::new().lod_clamp(1.0, 16.0);

        let key1 = SamplerCacheKey::from_descriptor(&desc1);
        let key2 = SamplerCacheKey::from_descriptor(&desc2);

        assert_ne!(key1, key2);
    }

    #[test]
    fn test_key_hash_consistency() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let desc = TrinitySamplerDescriptor::trilinear().anisotropy(4);
        let key = SamplerCacheKey::from_descriptor(&desc);

        let mut hasher1 = DefaultHasher::new();
        let mut hasher2 = DefaultHasher::new();
        key.hash(&mut hasher1);
        key.hash(&mut hasher2);

        assert_eq!(hasher1.finish(), hasher2.finish());
    }

    #[test]
    fn test_key_clone() {
        let desc = TrinitySamplerDescriptor::shadow();
        let key1 = SamplerCacheKey::from_descriptor(&desc);
        let key2 = key1;

        assert_eq!(key1, key2);
    }

    #[test]
    fn test_key_debug_format() {
        let desc = TrinitySamplerDescriptor::new();
        let key = SamplerCacheKey::from_descriptor(&desc);
        let debug_str = format!("{:?}", key);

        assert!(debug_str.contains("SamplerCacheKey"));
    }

    // ========================================================================
    // Conversion Helper Tests
    // ========================================================================

    #[test]
    fn test_address_mode_to_u8() {
        assert_eq!(address_mode_to_u8(AddressMode::ClampToEdge), 0);
        assert_eq!(address_mode_to_u8(AddressMode::Repeat), 1);
        assert_eq!(address_mode_to_u8(AddressMode::MirrorRepeat), 2);
        assert_eq!(address_mode_to_u8(AddressMode::ClampToBorder), 3);
    }

    #[test]
    fn test_filter_mode_to_u8() {
        assert_eq!(filter_mode_to_u8(FilterMode::Nearest), 0);
        assert_eq!(filter_mode_to_u8(FilterMode::Linear), 1);
    }

    #[test]
    fn test_compare_function_to_u8() {
        assert_eq!(compare_function_to_u8(None), 0);
        assert_eq!(compare_function_to_u8(Some(CompareFunction::Never)), 1);
        assert_eq!(compare_function_to_u8(Some(CompareFunction::Less)), 2);
        assert_eq!(compare_function_to_u8(Some(CompareFunction::Equal)), 3);
        assert_eq!(compare_function_to_u8(Some(CompareFunction::LessEqual)), 4);
        assert_eq!(compare_function_to_u8(Some(CompareFunction::Greater)), 5);
        assert_eq!(compare_function_to_u8(Some(CompareFunction::NotEqual)), 6);
        assert_eq!(compare_function_to_u8(Some(CompareFunction::GreaterEqual)), 7);
        assert_eq!(compare_function_to_u8(Some(CompareFunction::Always)), 8);
    }

    #[test]
    fn test_border_color_to_u8() {
        assert_eq!(border_color_to_u8(None), 0);
        assert_eq!(border_color_to_u8(Some(SamplerBorderColor::TransparentBlack)), 1);
        assert_eq!(border_color_to_u8(Some(SamplerBorderColor::OpaqueBlack)), 2);
        assert_eq!(border_color_to_u8(Some(SamplerBorderColor::OpaqueWhite)), 3);
        assert_eq!(border_color_to_u8(Some(SamplerBorderColor::Zero)), 4);
    }

    // ========================================================================
    // SamplerCacheMetrics Tests
    // ========================================================================

    #[test]
    fn test_metrics_default() {
        let metrics = SamplerCacheMetrics::default();

        assert_eq!(metrics.cache_size, 0);
        assert_eq!(metrics.hits, 0);
        assert_eq!(metrics.misses, 0);
        assert_eq!(metrics.hit_rate, 0.0);
    }

    #[test]
    fn test_metrics_new() {
        let metrics = SamplerCacheMetrics::new(5, 80, 20);

        assert_eq!(metrics.cache_size, 5);
        assert_eq!(metrics.hits, 80);
        assert_eq!(metrics.misses, 20);
        assert!((metrics.hit_rate - 0.8).abs() < 0.001);
    }

    #[test]
    fn test_metrics_total_requests() {
        let metrics = SamplerCacheMetrics::new(3, 50, 25);
        assert_eq!(metrics.total_requests(), 75);
    }

    #[test]
    fn test_metrics_is_empty() {
        let empty = SamplerCacheMetrics::new(0, 0, 0);
        assert!(empty.is_empty());

        let non_empty = SamplerCacheMetrics::new(1, 0, 1);
        assert!(!non_empty.is_empty());
    }

    #[test]
    fn test_metrics_hit_rate_percent() {
        let metrics = SamplerCacheMetrics::new(2, 75, 25);
        assert!((metrics.hit_rate_percent() - 75.0).abs() < 0.001);
    }

    #[test]
    fn test_metrics_zero_total() {
        let metrics = SamplerCacheMetrics::new(0, 0, 0);
        assert_eq!(metrics.hit_rate, 0.0);
        assert_eq!(metrics.total_requests(), 0);
    }

    #[test]
    fn test_metrics_clone() {
        let metrics = SamplerCacheMetrics::new(10, 100, 50);
        let cloned = metrics.clone();

        assert_eq!(cloned.cache_size, metrics.cache_size);
        assert_eq!(cloned.hits, metrics.hits);
        assert_eq!(cloned.misses, metrics.misses);
        assert_eq!(cloned.hit_rate, metrics.hit_rate);
    }

    #[test]
    fn test_metrics_debug_format() {
        let metrics = SamplerCacheMetrics::new(3, 10, 5);
        let debug_str = format!("{:?}", metrics);

        assert!(debug_str.contains("SamplerCacheMetrics"));
        assert!(debug_str.contains("cache_size"));
    }

    // ========================================================================
    // SamplerCacheKey Accessor Tests
    // ========================================================================

    #[test]
    fn test_key_accessors() {
        let desc = TrinitySamplerDescriptor::new()
            .address_mode(AddressMode::Repeat)
            .filter(FilterMode::Nearest)
            .anisotropy(8)
            .lod_clamp(1.0, 8.0);

        let key = SamplerCacheKey::from_descriptor(&desc);

        assert_eq!(key.address_mode_u(), 1); // Repeat
        assert_eq!(key.address_mode_v(), 1);
        assert_eq!(key.address_mode_w(), 1);
        assert_eq!(key.mag_filter(), 0); // Nearest
        assert_eq!(key.min_filter(), 0);
        assert_eq!(key.mipmap_filter(), 0);
        assert_eq!(key.compare(), 0); // None
        assert_eq!(key.anisotropy(), 8);
        assert_eq!(key.border_color(), 0); // None
        assert_eq!(key.lod_min_clamp_bits(), 1.0_f32.to_bits());
        assert_eq!(key.lod_max_clamp_bits(), 8.0_f32.to_bits());
    }

    // ========================================================================
    // Key with Border Color Tests
    // ========================================================================

    #[test]
    fn test_key_with_border_color() {
        let desc = TrinitySamplerDescriptor::new()
            .address_mode(AddressMode::ClampToBorder)
            .border_color(SamplerBorderColor::OpaqueWhite);

        let key = SamplerCacheKey::from_descriptor(&desc);

        assert_eq!(key.address_mode_u(), 3); // ClampToBorder
        assert_eq!(key.border_color(), 3); // OpaqueWhite
    }

    // ========================================================================
    // Key with Compare Function Tests
    // ========================================================================

    #[test]
    fn test_key_with_compare_function() {
        let desc = TrinitySamplerDescriptor::new()
            .compare(CompareFunction::LessEqual);

        let key = SamplerCacheKey::from_descriptor(&desc);

        assert_eq!(key.compare(), 4); // LessEqual
    }

    // ========================================================================
    // All Preset Keys Are Different
    // ========================================================================

    #[test]
    fn test_preset_keys_unique() {
        let linear_clamp = SamplerCacheKey::from_descriptor(&TrinitySamplerDescriptor::linear_clamp());
        let linear_repeat = SamplerCacheKey::from_descriptor(&TrinitySamplerDescriptor::linear_repeat());
        let nearest_clamp = SamplerCacheKey::from_descriptor(&TrinitySamplerDescriptor::nearest_clamp());
        let nearest_repeat = SamplerCacheKey::from_descriptor(&TrinitySamplerDescriptor::nearest_repeat());
        let shadow = SamplerCacheKey::from_descriptor(&TrinitySamplerDescriptor::shadow());

        // All keys should be different
        let keys = [linear_clamp, linear_repeat, nearest_clamp, nearest_repeat, shadow];
        for i in 0..keys.len() {
            for j in (i + 1)..keys.len() {
                assert_ne!(keys[i], keys[j], "Keys {} and {} should be different", i, j);
            }
        }
    }

    // ========================================================================
    // Mixed Address Mode Tests
    // ========================================================================

    #[test]
    fn test_key_mixed_address_modes() {
        let desc = TrinitySamplerDescriptor::new().address_mode_uvw(
            AddressMode::Repeat,
            AddressMode::ClampToEdge,
            AddressMode::MirrorRepeat,
        );

        let key = SamplerCacheKey::from_descriptor(&desc);

        assert_eq!(key.address_mode_u(), 1); // Repeat
        assert_eq!(key.address_mode_v(), 0); // ClampToEdge
        assert_eq!(key.address_mode_w(), 2); // MirrorRepeat
    }

    // ========================================================================
    // Mixed Filter Mode Tests
    // ========================================================================

    #[test]
    fn test_key_mixed_filter_modes() {
        let desc = TrinitySamplerDescriptor::new().filter_separate(
            FilterMode::Linear,
            FilterMode::Nearest,
            FilterMode::Linear,
        );

        let key = SamplerCacheKey::from_descriptor(&desc);

        assert_eq!(key.mag_filter(), 1); // Linear
        assert_eq!(key.min_filter(), 0); // Nearest
        assert_eq!(key.mipmap_filter(), 1); // Linear
    }

    // ========================================================================
    // LOD Edge Cases
    // ========================================================================

    #[test]
    fn test_key_lod_zero() {
        let desc = TrinitySamplerDescriptor::new().lod_clamp(0.0, 0.0);
        let key = SamplerCacheKey::from_descriptor(&desc);

        assert_eq!(key.lod_min_clamp_bits(), 0.0_f32.to_bits());
        assert_eq!(key.lod_max_clamp_bits(), 0.0_f32.to_bits());
    }

    #[test]
    fn test_key_lod_max_32() {
        let desc = TrinitySamplerDescriptor::new().lod_clamp(0.0, 32.0);
        let key = SamplerCacheKey::from_descriptor(&desc);

        assert_eq!(key.lod_max_clamp_bits(), 32.0_f32.to_bits());
    }

    // ========================================================================
    // HashMap Behavior Tests
    // ========================================================================

    #[test]
    fn test_key_as_hashmap_key() {
        let mut map: HashMap<SamplerCacheKey, i32> = HashMap::new();

        let key1 = SamplerCacheKey::from_descriptor(&TrinitySamplerDescriptor::linear_clamp());
        let key2 = SamplerCacheKey::from_descriptor(&TrinitySamplerDescriptor::linear_repeat());

        map.insert(key1, 1);
        map.insert(key2, 2);

        assert_eq!(map.len(), 2);
        assert_eq!(map.get(&key1), Some(&1));
        assert_eq!(map.get(&key2), Some(&2));
    }

    #[test]
    fn test_key_duplicate_insert() {
        let mut map: HashMap<SamplerCacheKey, i32> = HashMap::new();

        let desc = TrinitySamplerDescriptor::linear_clamp();
        let key1 = SamplerCacheKey::from_descriptor(&desc);
        let key2 = SamplerCacheKey::from_descriptor(&desc);

        map.insert(key1, 1);
        map.insert(key2, 2); // Should overwrite

        assert_eq!(map.len(), 1);
        assert_eq!(map.get(&key1), Some(&2));
    }
}
