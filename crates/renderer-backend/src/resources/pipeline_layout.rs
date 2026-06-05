//! Pipeline layout caching for TRINITY.
//!
//! This module provides a cache for wgpu pipeline layouts to avoid redundant GPU
//! resource creation. Layouts are identified by a hash key derived from their
//! bind group layout hashes and push constant ranges.
//!
//! # Overview
//!
//! GPU pipeline layouts describe the resource binding structure for a pipeline,
//! consisting of bind group layouts and optional push constant ranges. Multiple
//! pipelines often share identical layouts. This cache:
//!
//! - Deduplicates layouts with identical bind group and push constant configurations
//! - Uses pre-computed hashes from bind group layout cache for efficient lookup
//! - Tracks cache hit/miss metrics
//! - Uses thread-safe interior mutability for concurrent access
//!
//! # Thread Safety
//!
//! The cache uses `RwLock` for concurrent access:
//! - Multiple readers can query the cache simultaneously
//! - Write lock is only held briefly when creating new layouts
//!
//! # TRINITY Standard Layouts
//!
//! The engine uses a standard bind group layout convention:
//! - Group 0 (GLOBAL): Camera, lighting, time uniforms
//! - Group 1 (MATERIAL): Material textures and uniforms
//! - Group 2 (OBJECT): Per-object transforms
//! - Group 3 (BINDLESS): Bindless texture/buffer arrays
//!
//! # Example
//!
//! ```no_run
//! use renderer_backend::resources::pipeline_layout::{PipelineLayoutCache, TrinityLayoutBuilder};
//! use wgpu::BindGroupLayout;
//!
//! # fn example(device: &wgpu::Device, global_layout: &BindGroupLayout, material_layout: &BindGroupLayout) {
//! let cache = PipelineLayoutCache::new();
//!
//! // Create pipeline layout with bind group layouts
//! let layout = cache.get_or_create(
//!     device,
//!     Some("pbr_pipeline"),
//!     &[global_layout, material_layout],
//!     &[0x1234, 0x5678], // hashes from BindGroupLayoutCache
//!     &[], // no push constants
//! );
//!
//! // Check cache metrics
//! let metrics = cache.metrics();
//! println!("Cache size: {}, Hit rate: {:.1}%", metrics.cache_size, metrics.hit_rate * 100.0);
//! # }
//! ```

use parking_lot::RwLock;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use wgpu::{BindGroupLayout, Device, PipelineLayout, PipelineLayoutDescriptor, PushConstantRange, ShaderStages};

// ============================================================================
// Constants - Standard Bind Group Indices
// ============================================================================

/// Standard bind group indices for TRINITY engine.
///
/// These indices follow a consistent convention across all pipelines:
/// - Group 0: Global resources (camera, lighting, time)
/// - Group 1: Material resources (textures, material uniforms)
/// - Group 2: Object resources (per-object transforms)
/// - Group 3: Bindless resources (texture arrays, buffer arrays)
///
/// # Example
///
/// ```
/// use renderer_backend::resources::pipeline_layout::bind_group_index;
///
/// assert_eq!(bind_group_index::GLOBAL, 0);
/// assert_eq!(bind_group_index::MATERIAL, 1);
/// assert_eq!(bind_group_index::OBJECT, 2);
/// assert_eq!(bind_group_index::BINDLESS, 3);
/// ```
pub mod bind_group_index {
    /// Index for global bind group (camera, lighting, time).
    pub const GLOBAL: u32 = 0;
    /// Index for material bind group (textures, material uniforms).
    pub const MATERIAL: u32 = 1;
    /// Index for object bind group (per-object transforms).
    pub const OBJECT: u32 = 2;
    /// Index for bindless bind group (texture/buffer arrays).
    pub const BINDLESS: u32 = 3;
}

/// Maximum push constant size in bytes (WebGPU limit).
pub const MAX_PUSH_CONSTANT_SIZE: u32 = 128;

// ============================================================================
// PipelineLayoutKey
// ============================================================================

/// A hashable key derived from pipeline layout configuration.
///
/// This struct computes a hash from bind group layout hashes (in order) and
/// push constant ranges, ensuring that layouts with the same configuration
/// produce the same key.
///
/// # Implementation Notes
///
/// - Bind group layout hashes are hashed in order (order matters for pipeline layout)
/// - Push constant ranges are hashed with stages, start offset, and end offset
/// - The hash is computed once and stored for efficient lookup
///
/// # Example
///
/// ```
/// use renderer_backend::resources::pipeline_layout::PipelineLayoutKey;
/// use wgpu::{PushConstantRange, ShaderStages};
///
/// let key = PipelineLayoutKey::new(
///     &[0x1234, 0x5678],
///     &[PushConstantRange {
///         stages: ShaderStages::VERTEX,
///         range: 0..64,
///     }],
/// );
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PipelineLayoutKey {
    /// Hash of bind group layout hashes (in order).
    bind_group_layouts_hash: u64,
    /// Hash of push constant ranges.
    push_constants_hash: u64,
}

impl PipelineLayoutKey {
    /// Creates a cache key from bind group layout hashes and push constant ranges.
    ///
    /// # Arguments
    ///
    /// * `bind_group_layout_hashes` - Hashes from `BindGroupLayoutCache` in order
    /// * `push_constant_ranges` - Push constant range specifications
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::resources::pipeline_layout::PipelineLayoutKey;
    ///
    /// // Layout with two bind groups and no push constants
    /// let key = PipelineLayoutKey::new(&[0xAAAA, 0xBBBB], &[]);
    /// ```
    pub fn new(bind_group_layout_hashes: &[u64], push_constant_ranges: &[PushConstantRange]) -> Self {
        use std::collections::hash_map::DefaultHasher;

        // Hash the layout hashes in order (order matters for pipeline layout)
        let mut hasher = DefaultHasher::new();
        bind_group_layout_hashes.len().hash(&mut hasher);
        for &layout_hash in bind_group_layout_hashes {
            layout_hash.hash(&mut hasher);
        }
        let bind_group_layouts_hash = hasher.finish();

        // Hash the push constant ranges
        let mut hasher = DefaultHasher::new();
        push_constant_ranges.len().hash(&mut hasher);
        for range in push_constant_ranges {
            range.stages.bits().hash(&mut hasher);
            range.range.start.hash(&mut hasher);
            range.range.end.hash(&mut hasher);
        }
        let push_constants_hash = hasher.finish();

        Self {
            bind_group_layouts_hash,
            push_constants_hash,
        }
    }

    /// Returns the bind group layouts hash component.
    #[inline]
    pub fn bind_group_layouts_hash(&self) -> u64 {
        self.bind_group_layouts_hash
    }

    /// Returns the push constants hash component.
    #[inline]
    pub fn push_constants_hash(&self) -> u64 {
        self.push_constants_hash
    }
}

// ============================================================================
// CachedPipelineLayout
// ============================================================================

/// A cached pipeline layout with metadata.
///
/// This struct wraps an `Arc<wgpu::PipelineLayout>` with additional metadata
/// for debugging and introspection.
pub struct CachedPipelineLayout {
    /// The actual layout, wrapped in Arc for shared ownership.
    layout: Arc<PipelineLayout>,
    /// Number of bind group layouts in this pipeline layout.
    bind_group_count: usize,
    /// Total push constant size in bytes.
    push_constant_size: u32,
    /// Optional label for debugging.
    label: Option<String>,
}

impl CachedPipelineLayout {
    /// Returns a reference to the inner pipeline layout.
    #[inline]
    pub fn inner(&self) -> &PipelineLayout {
        &self.layout
    }

    /// Returns a clone of the Arc-wrapped layout.
    #[inline]
    pub fn arc(&self) -> Arc<PipelineLayout> {
        Arc::clone(&self.layout)
    }

    /// Returns the number of bind group layouts.
    #[inline]
    pub fn bind_group_count(&self) -> usize {
        self.bind_group_count
    }

    /// Returns the total push constant size in bytes.
    #[inline]
    pub fn push_constant_size(&self) -> u32 {
        self.push_constant_size
    }

    /// Returns the optional label.
    #[inline]
    pub fn label(&self) -> Option<&str> {
        self.label.as_deref()
    }
}

impl std::fmt::Debug for CachedPipelineLayout {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CachedPipelineLayout")
            .field("bind_group_count", &self.bind_group_count)
            .field("push_constant_size", &self.push_constant_size)
            .field("label", &self.label)
            .finish()
    }
}

// ============================================================================
// PipelineLayoutCacheMetrics
// ============================================================================

/// Metrics for monitoring pipeline layout cache performance.
///
/// These metrics help identify cache efficiency and potential optimization
/// opportunities.
///
/// # Example
///
/// ```
/// use renderer_backend::resources::pipeline_layout::PipelineLayoutCacheMetrics;
///
/// let metrics = PipelineLayoutCacheMetrics::default();
/// assert_eq!(metrics.cache_size, 0);
/// assert_eq!(metrics.hits, 0);
/// assert_eq!(metrics.misses, 0);
/// assert_eq!(metrics.hit_rate, 0.0);
/// ```
#[derive(Debug, Clone, Default)]
pub struct PipelineLayoutCacheMetrics {
    /// Number of unique layouts in the cache.
    pub cache_size: usize,
    /// Number of cache hits (requested layout already existed).
    pub hits: u64,
    /// Number of cache misses (new layout created).
    pub misses: u64,
    /// Hit rate as a ratio (0.0 to 1.0).
    pub hit_rate: f64,
}

impl PipelineLayoutCacheMetrics {
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
// PipelineLayoutCache
// ============================================================================

/// A thread-safe cache for wgpu pipeline layouts.
///
/// The cache stores pipeline layouts keyed by their bind group layout hashes
/// and push constant ranges, ensuring that identical configurations share the
/// same GPU resource.
///
/// # Architecture
///
/// ```text
/// PipelineLayoutCache
/// ├── Cache (HashMap<PipelineLayoutKey, CachedPipelineLayout>)
/// │   └── Layouts keyed by (bind_group_hashes, push_constants)
/// └── Metrics (hits, misses, size)
/// ```
///
/// # Thread Safety
///
/// - Uses `RwLock<HashMap>` for the cache
/// - Uses `AtomicU64` for hit/miss counters (lock-free)
/// - Multiple readers can access cached layouts concurrently
/// - Write lock is only held briefly when inserting new layouts
///
/// # Example
///
/// ```no_run
/// use renderer_backend::resources::pipeline_layout::PipelineLayoutCache;
/// use wgpu::BindGroupLayout;
///
/// # fn example(device: &wgpu::Device, layout: &BindGroupLayout) {
/// let cache = PipelineLayoutCache::new();
///
/// // Get or create pipeline layout
/// let pipeline_layout = cache.get_or_create(
///     device,
///     Some("my_pipeline"),
///     &[layout],
///     &[0x1234], // hash from BindGroupLayoutCache
///     &[], // no push constants
/// );
/// # }
/// ```
pub struct PipelineLayoutCache {
    /// Cache of layouts keyed by configuration.
    cache: RwLock<HashMap<PipelineLayoutKey, CachedPipelineLayout>>,
    /// Hit counter (atomic for lock-free updates).
    hits: AtomicU64,
    /// Miss counter (atomic for lock-free updates).
    misses: AtomicU64,
}

impl PipelineLayoutCache {
    /// Creates a new empty pipeline layout cache.
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::resources::pipeline_layout::PipelineLayoutCache;
    ///
    /// let cache = PipelineLayoutCache::new();
    /// assert!(cache.is_empty());
    /// ```
    pub fn new() -> Self {
        Self {
            cache: RwLock::new(HashMap::new()),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
        }
    }

    /// Gets or creates a pipeline layout matching the given configuration.
    ///
    /// If a layout with the same bind group layouts and push constant ranges
    /// already exists in the cache, it is returned. Otherwise, a new layout
    /// is created, cached, and returned.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device for creating layouts
    /// * `label` - Optional debug label for the layout
    /// * `bind_group_layouts` - Array of bind group layouts (in order)
    /// * `bind_group_layout_hashes` - Corresponding hashes from `BindGroupLayoutCache`
    /// * `push_constant_ranges` - Push constant range specifications
    ///
    /// # Returns
    ///
    /// An `Arc<PipelineLayout>` that can be shared across multiple pipelines.
    ///
    /// # Thread Safety
    ///
    /// This method uses a read lock for cache lookups and only acquires
    /// a write lock when creating a new layout (double-check pattern).
    ///
    /// # Example
    ///
    /// ```no_run
    /// use renderer_backend::resources::pipeline_layout::PipelineLayoutCache;
    /// use wgpu::{BindGroupLayout, PushConstantRange, ShaderStages};
    ///
    /// # fn example(device: &wgpu::Device, global: &BindGroupLayout, material: &BindGroupLayout) {
    /// let cache = PipelineLayoutCache::new();
    ///
    /// let layout = cache.get_or_create(
    ///     device,
    ///     Some("pbr_forward"),
    ///     &[global, material],
    ///     &[0x1234, 0x5678],
    ///     &[PushConstantRange {
    ///         stages: ShaderStages::VERTEX,
    ///         range: 0..64,
    ///     }],
    /// );
    /// # }
    /// ```
    pub fn get_or_create(
        &self,
        device: &Device,
        label: Option<&str>,
        bind_group_layouts: &[&BindGroupLayout],
        bind_group_layout_hashes: &[u64],
        push_constant_ranges: &[PushConstantRange],
    ) -> Arc<PipelineLayout> {
        let key = PipelineLayoutKey::new(bind_group_layout_hashes, push_constant_ranges);

        // Fast path: read lock
        {
            let cache = self.cache.read();
            if let Some(cached) = cache.get(&key) {
                self.hits.fetch_add(1, Ordering::Relaxed);
                return cached.arc();
            }
        }

        // Slow path: write lock (double-check pattern)
        let mut cache = self.cache.write();
        if let Some(cached) = cache.get(&key) {
            self.hits.fetch_add(1, Ordering::Relaxed);
            return cached.arc();
        }

        // Create new layout
        self.misses.fetch_add(1, Ordering::Relaxed);
        let layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label,
            bind_group_layouts,
            push_constant_ranges,
        });
        let arc_layout = Arc::new(layout);

        let push_constant_size = push_constant_ranges
            .iter()
            .map(|r| r.range.end)
            .max()
            .unwrap_or(0);

        cache.insert(
            key,
            CachedPipelineLayout {
                layout: Arc::clone(&arc_layout),
                bind_group_count: bind_group_layouts.len(),
                push_constant_size,
                label: label.map(String::from),
            },
        );

        arc_layout
    }

    /// Checks if a layout exists for the given configuration.
    ///
    /// # Arguments
    ///
    /// * `bind_group_layout_hashes` - Hashes from `BindGroupLayoutCache`
    /// * `push_constant_ranges` - Push constant range specifications
    ///
    /// # Returns
    ///
    /// `true` if a layout with matching configuration exists in the cache.
    pub fn contains(
        &self,
        bind_group_layout_hashes: &[u64],
        push_constant_ranges: &[PushConstantRange],
    ) -> bool {
        let key = PipelineLayoutKey::new(bind_group_layout_hashes, push_constant_ranges);
        self.cache.read().contains_key(&key)
    }

    /// Clears the cache, removing all cached layouts.
    ///
    /// Metrics (hits/misses) are also reset to zero.
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::resources::pipeline_layout::PipelineLayoutCache;
    ///
    /// let cache = PipelineLayoutCache::new();
    /// cache.clear();
    /// assert!(cache.is_empty());
    /// ```
    pub fn clear(&self) {
        self.cache.write().clear();
        self.hits.store(0, Ordering::Relaxed);
        self.misses.store(0, Ordering::Relaxed);
    }

    /// Returns the number of cached layouts.
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::resources::pipeline_layout::PipelineLayoutCache;
    ///
    /// let cache = PipelineLayoutCache::new();
    /// assert_eq!(cache.len(), 0);
    /// ```
    #[inline]
    pub fn len(&self) -> usize {
        self.cache.read().len()
    }

    /// Returns true if the cache contains no layouts.
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::resources::pipeline_layout::PipelineLayoutCache;
    ///
    /// let cache = PipelineLayoutCache::new();
    /// assert!(cache.is_empty());
    /// ```
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.cache.read().is_empty()
    }

    /// Returns current cache metrics.
    ///
    /// Metrics include cache size, hit count, miss count, and hit rate.
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::resources::pipeline_layout::PipelineLayoutCache;
    ///
    /// let cache = PipelineLayoutCache::new();
    /// let metrics = cache.metrics();
    ///
    /// assert_eq!(metrics.cache_size, 0);
    /// assert_eq!(metrics.hits, 0);
    /// assert_eq!(metrics.misses, 0);
    /// ```
    pub fn metrics(&self) -> PipelineLayoutCacheMetrics {
        let cache_size = self.cache.read().len();
        let hits = self.hits.load(Ordering::Relaxed);
        let misses = self.misses.load(Ordering::Relaxed);

        PipelineLayoutCacheMetrics::new(cache_size, hits, misses)
    }

    /// Resets metrics counters to zero without clearing the cache.
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::resources::pipeline_layout::PipelineLayoutCache;
    ///
    /// let cache = PipelineLayoutCache::new();
    /// cache.reset_metrics();
    ///
    /// let metrics = cache.metrics();
    /// assert_eq!(metrics.hits, 0);
    /// assert_eq!(metrics.misses, 0);
    /// ```
    pub fn reset_metrics(&self) {
        self.hits.store(0, Ordering::Relaxed);
        self.misses.store(0, Ordering::Relaxed);
    }

    /// Removes a specific layout from the cache.
    ///
    /// # Arguments
    ///
    /// * `bind_group_layout_hashes` - Hashes from `BindGroupLayoutCache`
    /// * `push_constant_ranges` - Push constant range specifications
    ///
    /// # Returns
    ///
    /// `true` if a layout was removed, `false` if no matching layout was found.
    pub fn remove(
        &self,
        bind_group_layout_hashes: &[u64],
        push_constant_ranges: &[PushConstantRange],
    ) -> bool {
        let key = PipelineLayoutKey::new(bind_group_layout_hashes, push_constant_ranges);
        self.cache.write().remove(&key).is_some()
    }

    /// Returns iterator over all cached layout labels (for debugging).
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::resources::pipeline_layout::PipelineLayoutCache;
    ///
    /// let cache = PipelineLayoutCache::new();
    /// let labels: Vec<_> = cache.labels().collect();
    /// assert!(labels.is_empty());
    /// ```
    pub fn labels(&self) -> impl Iterator<Item = Option<String>> + '_ {
        let cache = self.cache.read();
        cache
            .values()
            .map(|c| c.label.clone())
            .collect::<Vec<_>>()
            .into_iter()
    }
}

impl Default for PipelineLayoutCache {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for PipelineLayoutCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let metrics = self.metrics();
        f.debug_struct("PipelineLayoutCache")
            .field("cache_size", &metrics.cache_size)
            .field("hits", &metrics.hits)
            .field("misses", &metrics.misses)
            .field("hit_rate", &format!("{:.1}%", metrics.hit_rate_percent()))
            .finish()
    }
}

// ============================================================================
// TrinityLayoutBuilder
// ============================================================================

/// Helper to create standard TRINITY pipeline layouts.
///
/// This builder provides convenient methods for creating pipeline layouts
/// that follow TRINITY's standard bind group index conventions.
///
/// # Standard Layouts
///
/// - `global_only`: Just group 0 (global uniforms)
/// - `global_material`: Groups 0-1 (global + material)
/// - `pbr`: Groups 0-2 (global + material + object)
/// - `bindless`: Groups 0-3 (global + material + object + bindless)
///
/// # Example
///
/// ```no_run
/// use renderer_backend::resources::pipeline_layout::{PipelineLayoutCache, TrinityLayoutBuilder};
/// use wgpu::BindGroupLayout;
///
/// # fn example(device: &wgpu::Device, cache: &PipelineLayoutCache, global: &BindGroupLayout, material: &BindGroupLayout, object: &BindGroupLayout) {
/// let builder = TrinityLayoutBuilder::new(device, cache);
///
/// // Create PBR pipeline layout
/// let pbr_layout = builder.pbr(global, 0x1234, material, 0x5678, object, 0x9ABC);
/// # }
/// ```
pub struct TrinityLayoutBuilder<'a> {
    device: &'a Device,
    cache: &'a PipelineLayoutCache,
}

impl<'a> TrinityLayoutBuilder<'a> {
    /// Creates a new builder with the given device and cache.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device for creating layouts
    /// * `cache` - The pipeline layout cache
    pub fn new(device: &'a Device, cache: &'a PipelineLayoutCache) -> Self {
        Self { device, cache }
    }

    /// Creates a layout with just the global bind group.
    ///
    /// This is suitable for simple shaders that only need camera/lighting data.
    ///
    /// # Arguments
    ///
    /// * `global_layout` - The global bind group layout
    /// * `global_hash` - Hash from `BindGroupLayoutCache`
    ///
    /// # Example
    ///
    /// ```no_run
    /// use renderer_backend::resources::pipeline_layout::{PipelineLayoutCache, TrinityLayoutBuilder};
    ///
    /// # fn example(device: &wgpu::Device, global: &wgpu::BindGroupLayout) {
    /// let cache = PipelineLayoutCache::new();
    /// let builder = TrinityLayoutBuilder::new(device, &cache);
    /// let layout = builder.global_only(global, 0x1234);
    /// # }
    /// ```
    pub fn global_only(
        &self,
        global_layout: &BindGroupLayout,
        global_hash: u64,
    ) -> Arc<PipelineLayout> {
        self.cache.get_or_create(
            self.device,
            Some("trinity_global_only"),
            &[global_layout],
            &[global_hash],
            &[],
        )
    }

    /// Creates a layout with global and material bind groups.
    ///
    /// This is suitable for unlit or simple shaders that don't need
    /// per-object transforms.
    ///
    /// # Arguments
    ///
    /// * `global_layout` - The global bind group layout
    /// * `global_hash` - Hash from `BindGroupLayoutCache`
    /// * `material_layout` - The material bind group layout
    /// * `material_hash` - Hash from `BindGroupLayoutCache`
    pub fn global_material(
        &self,
        global_layout: &BindGroupLayout,
        global_hash: u64,
        material_layout: &BindGroupLayout,
        material_hash: u64,
    ) -> Arc<PipelineLayout> {
        self.cache.get_or_create(
            self.device,
            Some("trinity_global_material"),
            &[global_layout, material_layout],
            &[global_hash, material_hash],
            &[],
        )
    }

    /// Creates the standard PBR pipeline layout.
    ///
    /// This layout includes global, material, and object bind groups,
    /// which is the most common configuration for 3D rendering.
    ///
    /// # Arguments
    ///
    /// * `global_layout` - The global bind group layout
    /// * `global_hash` - Hash from `BindGroupLayoutCache`
    /// * `material_layout` - The material bind group layout
    /// * `material_hash` - Hash from `BindGroupLayoutCache`
    /// * `object_layout` - The object bind group layout
    /// * `object_hash` - Hash from `BindGroupLayoutCache`
    pub fn pbr(
        &self,
        global_layout: &BindGroupLayout,
        global_hash: u64,
        material_layout: &BindGroupLayout,
        material_hash: u64,
        object_layout: &BindGroupLayout,
        object_hash: u64,
    ) -> Arc<PipelineLayout> {
        self.cache.get_or_create(
            self.device,
            Some("trinity_pbr"),
            &[global_layout, material_layout, object_layout],
            &[global_hash, material_hash, object_hash],
            &[],
        )
    }

    /// Creates a layout with all four standard bind groups including bindless.
    ///
    /// This layout supports bindless rendering with texture and buffer arrays.
    ///
    /// # Arguments
    ///
    /// * `global_layout` - The global bind group layout
    /// * `global_hash` - Hash from `BindGroupLayoutCache`
    /// * `material_layout` - The material bind group layout
    /// * `material_hash` - Hash from `BindGroupLayoutCache`
    /// * `object_layout` - The object bind group layout
    /// * `object_hash` - Hash from `BindGroupLayoutCache`
    /// * `bindless_layout` - The bindless bind group layout
    /// * `bindless_hash` - Hash from `BindGroupLayoutCache`
    pub fn bindless(
        &self,
        global_layout: &BindGroupLayout,
        global_hash: u64,
        material_layout: &BindGroupLayout,
        material_hash: u64,
        object_layout: &BindGroupLayout,
        object_hash: u64,
        bindless_layout: &BindGroupLayout,
        bindless_hash: u64,
    ) -> Arc<PipelineLayout> {
        self.cache.get_or_create(
            self.device,
            Some("trinity_bindless"),
            &[global_layout, material_layout, object_layout, bindless_layout],
            &[global_hash, material_hash, object_hash, bindless_hash],
            &[],
        )
    }

    /// Creates a layout with push constants.
    ///
    /// This is a generic method for creating layouts with custom push constant
    /// configurations.
    ///
    /// # Arguments
    ///
    /// * `label` - Optional debug label
    /// * `bind_group_layouts` - Array of bind group layouts
    /// * `bind_group_layout_hashes` - Corresponding hashes
    /// * `push_constant_ranges` - Push constant range specifications
    pub fn with_push_constants(
        &self,
        label: Option<&str>,
        bind_group_layouts: &[&BindGroupLayout],
        bind_group_layout_hashes: &[u64],
        push_constant_ranges: &[PushConstantRange],
    ) -> Arc<PipelineLayout> {
        self.cache.get_or_create(
            self.device,
            label,
            bind_group_layouts,
            bind_group_layout_hashes,
            push_constant_ranges,
        )
    }

    /// Creates a PBR layout with vertex push constants.
    ///
    /// This layout includes push constants for per-draw data like object ID
    /// or material index.
    ///
    /// # Arguments
    ///
    /// * `global_layout` - The global bind group layout
    /// * `global_hash` - Hash from `BindGroupLayoutCache`
    /// * `material_layout` - The material bind group layout
    /// * `material_hash` - Hash from `BindGroupLayoutCache`
    /// * `object_layout` - The object bind group layout
    /// * `object_hash` - Hash from `BindGroupLayoutCache`
    /// * `push_constant_size` - Size of push constants in bytes
    pub fn pbr_with_push_constants(
        &self,
        global_layout: &BindGroupLayout,
        global_hash: u64,
        material_layout: &BindGroupLayout,
        material_hash: u64,
        object_layout: &BindGroupLayout,
        object_hash: u64,
        push_constant_size: u32,
    ) -> Arc<PipelineLayout> {
        let push_constant_ranges = if push_constant_size > 0 {
            vec![PushConstantRange {
                stages: ShaderStages::VERTEX | ShaderStages::FRAGMENT,
                range: 0..push_constant_size,
            }]
        } else {
            vec![]
        };

        self.cache.get_or_create(
            self.device,
            Some("trinity_pbr_push"),
            &[global_layout, material_layout, object_layout],
            &[global_hash, material_hash, object_hash],
            &push_constant_ranges,
        )
    }

    /// Creates a compute-only layout.
    ///
    /// This layout is suitable for compute shaders that don't follow the
    /// standard rendering bind group conventions.
    ///
    /// # Arguments
    ///
    /// * `label` - Optional debug label
    /// * `bind_group_layouts` - Array of bind group layouts
    /// * `bind_group_layout_hashes` - Corresponding hashes
    pub fn compute(
        &self,
        label: Option<&str>,
        bind_group_layouts: &[&BindGroupLayout],
        bind_group_layout_hashes: &[u64],
    ) -> Arc<PipelineLayout> {
        self.cache.get_or_create(
            self.device,
            label,
            bind_group_layouts,
            bind_group_layout_hashes,
            &[],
        )
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Validates push constant ranges against WebGPU limits.
///
/// # Arguments
///
/// * `ranges` - Push constant ranges to validate
///
/// # Returns
///
/// `Ok(())` if valid, `Err` with description otherwise.
///
/// # Example
///
/// ```
/// use renderer_backend::resources::pipeline_layout::validate_push_constant_ranges;
/// use wgpu::{PushConstantRange, ShaderStages};
///
/// let ranges = &[PushConstantRange {
///     stages: ShaderStages::VERTEX,
///     range: 0..64,
/// }];
///
/// assert!(validate_push_constant_ranges(ranges).is_ok());
/// ```
pub fn validate_push_constant_ranges(ranges: &[PushConstantRange]) -> Result<(), String> {
    let total_size = ranges.iter().map(|r| r.range.end).max().unwrap_or(0);

    if total_size > MAX_PUSH_CONSTANT_SIZE {
        return Err(format!(
            "Push constant size {} exceeds WebGPU limit of {} bytes",
            total_size, MAX_PUSH_CONSTANT_SIZE
        ));
    }

    // Check for overlapping ranges within the same stages
    for (i, a) in ranges.iter().enumerate() {
        for (j, b) in ranges.iter().enumerate() {
            if i >= j {
                continue;
            }

            // If stages overlap, ranges must not overlap
            if a.stages.intersects(b.stages) {
                let a_range = a.range.start..a.range.end;
                let b_range = b.range.start..b.range.end;

                if a_range.start < b_range.end && b_range.start < a_range.end {
                    return Err(format!(
                        "Overlapping push constant ranges with overlapping stages: {:?} and {:?}",
                        a, b
                    ));
                }
            }
        }
    }

    // Check alignment (push constants should be 4-byte aligned)
    for range in ranges {
        if range.range.start % 4 != 0 || range.range.end % 4 != 0 {
            return Err(format!(
                "Push constant range {:?} is not 4-byte aligned",
                range
            ));
        }
    }

    Ok(())
}

/// Calculates the total push constant size from ranges.
///
/// # Arguments
///
/// * `ranges` - Push constant ranges
///
/// # Returns
///
/// The maximum end offset, which is the total required size.
#[inline]
pub fn total_push_constant_size(ranges: &[PushConstantRange]) -> u32 {
    ranges.iter().map(|r| r.range.end).max().unwrap_or(0)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // Bind Group Index Tests
    // ========================================================================

    #[test]
    fn test_bind_group_indices() {
        assert_eq!(bind_group_index::GLOBAL, 0);
        assert_eq!(bind_group_index::MATERIAL, 1);
        assert_eq!(bind_group_index::OBJECT, 2);
        assert_eq!(bind_group_index::BINDLESS, 3);
    }

    #[test]
    fn test_max_push_constant_size() {
        assert_eq!(MAX_PUSH_CONSTANT_SIZE, 128);
    }

    // ========================================================================
    // PipelineLayoutKey Tests
    // ========================================================================

    #[test]
    fn test_key_empty() {
        let key = PipelineLayoutKey::new(&[], &[]);
        assert_ne!(key.bind_group_layouts_hash(), 0);
        assert_ne!(key.push_constants_hash(), 0);
    }

    #[test]
    fn test_key_single_layout() {
        let key = PipelineLayoutKey::new(&[0x1234], &[]);
        assert_ne!(key.bind_group_layouts_hash(), 0);
    }

    #[test]
    fn test_key_multiple_layouts() {
        let key = PipelineLayoutKey::new(&[0x1234, 0x5678, 0x9ABC], &[]);
        assert_ne!(key.bind_group_layouts_hash(), 0);
    }

    #[test]
    fn test_key_equality_same_config() {
        let key1 = PipelineLayoutKey::new(&[0x1234, 0x5678], &[]);
        let key2 = PipelineLayoutKey::new(&[0x1234, 0x5678], &[]);
        assert_eq!(key1, key2);
    }

    #[test]
    fn test_key_inequality_different_hashes() {
        let key1 = PipelineLayoutKey::new(&[0x1234, 0x5678], &[]);
        let key2 = PipelineLayoutKey::new(&[0x1234, 0xABCD], &[]);
        assert_ne!(key1, key2);
    }

    #[test]
    fn test_key_inequality_different_order() {
        // Order matters for pipeline layouts!
        let key1 = PipelineLayoutKey::new(&[0x1234, 0x5678], &[]);
        let key2 = PipelineLayoutKey::new(&[0x5678, 0x1234], &[]);
        assert_ne!(key1, key2);
    }

    #[test]
    fn test_key_inequality_different_count() {
        let key1 = PipelineLayoutKey::new(&[0x1234], &[]);
        let key2 = PipelineLayoutKey::new(&[0x1234, 0x5678], &[]);
        assert_ne!(key1, key2);
    }

    #[test]
    fn test_key_with_push_constants() {
        let ranges = &[PushConstantRange {
            stages: ShaderStages::VERTEX,
            range: 0..64,
        }];
        let key = PipelineLayoutKey::new(&[0x1234], ranges);
        assert_ne!(key.push_constants_hash(), 0);
    }

    #[test]
    fn test_key_push_constants_equality() {
        let ranges = &[PushConstantRange {
            stages: ShaderStages::VERTEX,
            range: 0..64,
        }];
        let key1 = PipelineLayoutKey::new(&[0x1234], ranges);
        let key2 = PipelineLayoutKey::new(&[0x1234], ranges);
        assert_eq!(key1, key2);
    }

    #[test]
    fn test_key_push_constants_inequality_stages() {
        let ranges1 = &[PushConstantRange {
            stages: ShaderStages::VERTEX,
            range: 0..64,
        }];
        let ranges2 = &[PushConstantRange {
            stages: ShaderStages::FRAGMENT,
            range: 0..64,
        }];
        let key1 = PipelineLayoutKey::new(&[0x1234], ranges1);
        let key2 = PipelineLayoutKey::new(&[0x1234], ranges2);
        assert_ne!(key1, key2);
    }

    #[test]
    fn test_key_push_constants_inequality_range() {
        let ranges1 = &[PushConstantRange {
            stages: ShaderStages::VERTEX,
            range: 0..64,
        }];
        let ranges2 = &[PushConstantRange {
            stages: ShaderStages::VERTEX,
            range: 0..128,
        }];
        let key1 = PipelineLayoutKey::new(&[0x1234], ranges1);
        let key2 = PipelineLayoutKey::new(&[0x1234], ranges2);
        assert_ne!(key1, key2);
    }

    #[test]
    fn test_key_as_hashmap_key() {
        let mut map: HashMap<PipelineLayoutKey, i32> = HashMap::new();

        let key1 = PipelineLayoutKey::new(&[0x1111], &[]);
        let key2 = PipelineLayoutKey::new(&[0x2222], &[]);

        map.insert(key1.clone(), 1);
        map.insert(key2.clone(), 2);

        assert_eq!(map.len(), 2);
        assert_eq!(map.get(&key1), Some(&1));
        assert_eq!(map.get(&key2), Some(&2));
    }

    #[test]
    fn test_key_hash_stability() {
        let ranges = &[PushConstantRange {
            stages: ShaderStages::VERTEX | ShaderStages::FRAGMENT,
            range: 0..64,
        }];
        let key1 = PipelineLayoutKey::new(&[0x1234, 0x5678], ranges);
        let key2 = PipelineLayoutKey::new(&[0x1234, 0x5678], ranges);

        assert_eq!(key1.bind_group_layouts_hash(), key2.bind_group_layouts_hash());
        assert_eq!(key1.push_constants_hash(), key2.push_constants_hash());
    }

    #[test]
    fn test_key_clone() {
        let key = PipelineLayoutKey::new(&[0x1234], &[]);
        let cloned = key.clone();
        assert_eq!(key, cloned);
    }

    #[test]
    fn test_key_debug_format() {
        let key = PipelineLayoutKey::new(&[0x1234], &[]);
        let debug_str = format!("{:?}", key);
        assert!(debug_str.contains("PipelineLayoutKey"));
    }

    // ========================================================================
    // CachedPipelineLayout Tests
    // ========================================================================

    #[test]
    fn test_cached_layout_debug() {
        // We can't create a real CachedPipelineLayout without a device
        let _format = "CachedPipelineLayout { bind_group_count: 3, push_constant_size: 64 }";
    }

    // ========================================================================
    // PipelineLayoutCacheMetrics Tests
    // ========================================================================

    #[test]
    fn test_metrics_default() {
        let metrics = PipelineLayoutCacheMetrics::default();
        assert_eq!(metrics.cache_size, 0);
        assert_eq!(metrics.hits, 0);
        assert_eq!(metrics.misses, 0);
        assert_eq!(metrics.hit_rate, 0.0);
    }

    #[test]
    fn test_metrics_new() {
        let metrics = PipelineLayoutCacheMetrics::new(5, 80, 20);
        assert_eq!(metrics.cache_size, 5);
        assert_eq!(metrics.hits, 80);
        assert_eq!(metrics.misses, 20);
        assert!((metrics.hit_rate - 0.8).abs() < 0.001);
    }

    #[test]
    fn test_metrics_total_requests() {
        let metrics = PipelineLayoutCacheMetrics::new(3, 50, 25);
        assert_eq!(metrics.total_requests(), 75);
    }

    #[test]
    fn test_metrics_is_empty() {
        let empty = PipelineLayoutCacheMetrics::new(0, 0, 0);
        assert!(empty.is_empty());

        let non_empty = PipelineLayoutCacheMetrics::new(1, 0, 1);
        assert!(!non_empty.is_empty());
    }

    #[test]
    fn test_metrics_hit_rate_percent() {
        let metrics = PipelineLayoutCacheMetrics::new(2, 75, 25);
        assert!((metrics.hit_rate_percent() - 75.0).abs() < 0.001);
    }

    #[test]
    fn test_metrics_zero_total() {
        let metrics = PipelineLayoutCacheMetrics::new(0, 0, 0);
        assert_eq!(metrics.hit_rate, 0.0);
        assert_eq!(metrics.total_requests(), 0);
    }

    #[test]
    fn test_metrics_clone() {
        let metrics = PipelineLayoutCacheMetrics::new(10, 100, 50);
        let cloned = metrics.clone();

        assert_eq!(cloned.cache_size, metrics.cache_size);
        assert_eq!(cloned.hits, metrics.hits);
        assert_eq!(cloned.misses, metrics.misses);
        assert_eq!(cloned.hit_rate, metrics.hit_rate);
    }

    #[test]
    fn test_metrics_debug_format() {
        let metrics = PipelineLayoutCacheMetrics::new(3, 10, 5);
        let debug_str = format!("{:?}", metrics);
        assert!(debug_str.contains("PipelineLayoutCacheMetrics"));
    }

    // ========================================================================
    // PipelineLayoutCache Tests (no device required)
    // ========================================================================

    #[test]
    fn test_cache_new() {
        let cache = PipelineLayoutCache::new();
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn test_cache_default() {
        let cache = PipelineLayoutCache::default();
        assert!(cache.is_empty());
    }

    #[test]
    fn test_cache_metrics_initial() {
        let cache = PipelineLayoutCache::new();
        let metrics = cache.metrics();

        assert_eq!(metrics.cache_size, 0);
        assert_eq!(metrics.hits, 0);
        assert_eq!(metrics.misses, 0);
        assert_eq!(metrics.hit_rate, 0.0);
    }

    #[test]
    fn test_cache_reset_metrics() {
        let cache = PipelineLayoutCache::new();

        // Simulate some activity
        cache.hits.fetch_add(10, Ordering::Relaxed);
        cache.misses.fetch_add(5, Ordering::Relaxed);

        cache.reset_metrics();

        let metrics = cache.metrics();
        assert_eq!(metrics.hits, 0);
        assert_eq!(metrics.misses, 0);
    }

    #[test]
    fn test_cache_clear() {
        let cache = PipelineLayoutCache::new();

        // Simulate some activity
        cache.hits.fetch_add(10, Ordering::Relaxed);
        cache.misses.fetch_add(5, Ordering::Relaxed);

        cache.clear();

        assert!(cache.is_empty());
        let metrics = cache.metrics();
        assert_eq!(metrics.hits, 0);
        assert_eq!(metrics.misses, 0);
    }

    #[test]
    fn test_cache_labels_empty() {
        let cache = PipelineLayoutCache::new();
        let labels: Vec<_> = cache.labels().collect();
        assert!(labels.is_empty());
    }

    #[test]
    fn test_cache_debug_format() {
        let cache = PipelineLayoutCache::new();
        let debug_str = format!("{:?}", cache);

        assert!(debug_str.contains("PipelineLayoutCache"));
        assert!(debug_str.contains("cache_size"));
    }

    #[test]
    fn test_cache_contains_empty() {
        let cache = PipelineLayoutCache::new();
        assert!(!cache.contains(&[0x1234], &[]));
    }

    #[test]
    fn test_cache_remove_nonexistent() {
        let cache = PipelineLayoutCache::new();
        assert!(!cache.remove(&[0x1234], &[]));
    }

    // ========================================================================
    // Thread Safety Tests
    // ========================================================================

    #[test]
    fn test_cache_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<PipelineLayoutCache>();
    }

    #[test]
    fn test_metrics_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<PipelineLayoutCacheMetrics>();
    }

    // ========================================================================
    // Push Constant Validation Tests
    // ========================================================================

    #[test]
    fn test_validate_push_constants_empty() {
        assert!(validate_push_constant_ranges(&[]).is_ok());
    }

    #[test]
    fn test_validate_push_constants_valid() {
        let ranges = &[PushConstantRange {
            stages: ShaderStages::VERTEX,
            range: 0..64,
        }];
        assert!(validate_push_constant_ranges(ranges).is_ok());
    }

    #[test]
    fn test_validate_push_constants_max_size() {
        let ranges = &[PushConstantRange {
            stages: ShaderStages::VERTEX,
            range: 0..128,
        }];
        assert!(validate_push_constant_ranges(ranges).is_ok());
    }

    #[test]
    fn test_validate_push_constants_exceeds_limit() {
        let ranges = &[PushConstantRange {
            stages: ShaderStages::VERTEX,
            range: 0..256,
        }];
        let result = validate_push_constant_ranges(ranges);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("exceeds"));
    }

    #[test]
    fn test_validate_push_constants_non_overlapping() {
        let ranges = &[
            PushConstantRange {
                stages: ShaderStages::VERTEX,
                range: 0..32,
            },
            PushConstantRange {
                stages: ShaderStages::FRAGMENT,
                range: 32..64,
            },
        ];
        assert!(validate_push_constant_ranges(ranges).is_ok());
    }

    #[test]
    fn test_validate_push_constants_overlapping_different_stages() {
        // Same range but different stages is allowed
        let ranges = &[
            PushConstantRange {
                stages: ShaderStages::VERTEX,
                range: 0..64,
            },
            PushConstantRange {
                stages: ShaderStages::FRAGMENT,
                range: 64..128,
            },
        ];
        assert!(validate_push_constant_ranges(ranges).is_ok());
    }

    #[test]
    fn test_validate_push_constants_overlapping_same_stages() {
        let ranges = &[
            PushConstantRange {
                stages: ShaderStages::VERTEX,
                range: 0..64,
            },
            PushConstantRange {
                stages: ShaderStages::VERTEX,
                range: 32..96,
            },
        ];
        let result = validate_push_constant_ranges(ranges);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Overlapping"));
    }

    #[test]
    fn test_validate_push_constants_misaligned_start() {
        let ranges = &[PushConstantRange {
            stages: ShaderStages::VERTEX,
            range: 1..64,
        }];
        let result = validate_push_constant_ranges(ranges);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("aligned"));
    }

    #[test]
    fn test_validate_push_constants_misaligned_end() {
        let ranges = &[PushConstantRange {
            stages: ShaderStages::VERTEX,
            range: 0..65,
        }];
        let result = validate_push_constant_ranges(ranges);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("aligned"));
    }

    #[test]
    fn test_total_push_constant_size_empty() {
        assert_eq!(total_push_constant_size(&[]), 0);
    }

    #[test]
    fn test_total_push_constant_size_single() {
        let ranges = &[PushConstantRange {
            stages: ShaderStages::VERTEX,
            range: 0..64,
        }];
        assert_eq!(total_push_constant_size(ranges), 64);
    }

    #[test]
    fn test_total_push_constant_size_multiple() {
        let ranges = &[
            PushConstantRange {
                stages: ShaderStages::VERTEX,
                range: 0..32,
            },
            PushConstantRange {
                stages: ShaderStages::FRAGMENT,
                range: 32..96,
            },
        ];
        assert_eq!(total_push_constant_size(ranges), 96);
    }

    // ========================================================================
    // Integration Tests (require GPU device)
    // ========================================================================

    fn create_test_device() -> Option<(wgpu::Device, wgpu::Queue)> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            ..Default::default()
        });
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))?;
        Some(
            pollster::block_on(adapter.request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("test device"),
                    required_features: wgpu::Features::PUSH_CONSTANTS,
                    required_limits: wgpu::Limits {
                        max_push_constant_size: 128,
                        ..wgpu::Limits::default()
                    },
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            ))
            .ok()?,
        )
    }

    fn create_test_device_no_push_constants() -> Option<(wgpu::Device, wgpu::Queue)> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            ..Default::default()
        });
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))?;
        Some(
            pollster::block_on(adapter.request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("test device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            ))
            .expect("device creation"),
        )
    }

    #[test]
    fn test_cache_get_or_create() {
        let (device, _queue) = match create_test_device_no_push_constants() {
            Some(d) => d,
            None => {
                eprintln!("Skipping: no GPU adapter available");
                return;
            }
        };

        // Create a bind group layout
        let bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("test"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let cache = PipelineLayoutCache::new();

        // First call creates the layout
        let layout1 = cache.get_or_create(
            &device,
            Some("test"),
            &[&bind_group_layout],
            &[0x1234],
            &[],
        );
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.metrics().misses, 1);
        assert_eq!(cache.metrics().hits, 0);

        // Second call returns cached layout
        let layout2 = cache.get_or_create(
            &device,
            Some("test"),
            &[&bind_group_layout],
            &[0x1234],
            &[],
        );
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.metrics().misses, 1);
        assert_eq!(cache.metrics().hits, 1);

        // Same Arc
        assert!(Arc::ptr_eq(&layout1, &layout2));
    }

    #[test]
    fn test_cache_different_layouts() {
        let (device, _queue) = match create_test_device_no_push_constants() {
            Some(d) => d,
            None => {
                eprintln!("Skipping: no GPU adapter available");
                return;
            }
        };

        let layout1 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("layout1"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let layout2 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("layout2"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            }],
        });

        let cache = PipelineLayoutCache::new();

        // Create with layout1 only
        let _pl1 = cache.get_or_create(&device, Some("pl1"), &[&layout1], &[0x1111], &[]);
        assert_eq!(cache.len(), 1);

        // Create with layout1 + layout2
        let _pl2 = cache.get_or_create(
            &device,
            Some("pl2"),
            &[&layout1, &layout2],
            &[0x1111, 0x2222],
            &[],
        );
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn test_cache_with_push_constants() {
        let (device, _queue) = match create_test_device() {
            Some(d) => d,
            None => {
                eprintln!("Skipping: no GPU adapter with push constants available");
                return;
            }
        };

        let bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("test"),
                entries: &[],
            });

        let cache = PipelineLayoutCache::new();

        let push_constants = &[PushConstantRange {
            stages: ShaderStages::VERTEX,
            range: 0..64,
        }];

        let layout = cache.get_or_create(
            &device,
            Some("with_push"),
            &[&bind_group_layout],
            &[0x1234],
            push_constants,
        );
        assert_eq!(cache.len(), 1);

        // Without push constants should be different
        let layout_no_push = cache.get_or_create(
            &device,
            Some("without_push"),
            &[&bind_group_layout],
            &[0x1234],
            &[],
        );
        assert_eq!(cache.len(), 2);

        assert!(!Arc::ptr_eq(&layout, &layout_no_push));
    }

    #[test]
    fn test_cache_contains_and_remove() {
        let (device, _queue) = match create_test_device_no_push_constants() {
            Some(d) => d,
            None => {
                eprintln!("Skipping: no GPU adapter available");
                return;
            }
        };

        let bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("test"),
                entries: &[],
            });

        let cache = PipelineLayoutCache::new();

        // Not in cache yet
        assert!(!cache.contains(&[0x1234], &[]));

        // Create layout
        let _layout = cache.get_or_create(&device, None, &[&bind_group_layout], &[0x1234], &[]);
        assert!(cache.contains(&[0x1234], &[]));

        // Remove
        assert!(cache.remove(&[0x1234], &[]));
        assert!(!cache.contains(&[0x1234], &[]));
        assert!(cache.is_empty());

        // Can't remove again
        assert!(!cache.remove(&[0x1234], &[]));
    }

    #[test]
    fn test_trinity_layout_builder_global_only() {
        let (device, _queue) = match create_test_device_no_push_constants() {
            Some(d) => d,
            None => {
                eprintln!("Skipping: no GPU adapter available");
                return;
            }
        };

        let global_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("global"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::VERTEX | ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let cache = PipelineLayoutCache::new();
        let builder = TrinityLayoutBuilder::new(&device, &cache);

        let layout = builder.global_only(&global_layout, 0x1234);
        assert_eq!(cache.len(), 1);

        // Second call should return cached
        let layout2 = builder.global_only(&global_layout, 0x1234);
        assert!(Arc::ptr_eq(&layout, &layout2));
    }

    #[test]
    fn test_trinity_layout_builder_pbr() {
        let (device, _queue) = match create_test_device_no_push_constants() {
            Some(d) => d,
            None => {
                eprintln!("Skipping: no GPU adapter available");
                return;
            }
        };

        let global_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("global"),
            entries: &[],
        });
        let material_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("material"),
            entries: &[],
        });
        let object_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("object"),
            entries: &[],
        });

        let cache = PipelineLayoutCache::new();
        let builder = TrinityLayoutBuilder::new(&device, &cache);

        let layout = builder.pbr(
            &global_layout,
            0x1111,
            &material_layout,
            0x2222,
            &object_layout,
            0x3333,
        );
        assert_eq!(cache.len(), 1);

        // Labels should show trinity_pbr
        let labels: Vec<_> = cache.labels().collect();
        assert_eq!(labels.len(), 1);
        assert_eq!(labels[0], Some("trinity_pbr".to_string()));

        // Confirm it's there
        drop(layout);
    }

    #[test]
    fn test_trinity_layout_builder_bindless() {
        let (device, _queue) = match create_test_device_no_push_constants() {
            Some(d) => d,
            None => {
                eprintln!("Skipping: no GPU adapter available");
                return;
            }
        };

        let global = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("global"),
            entries: &[],
        });
        let material = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("material"),
            entries: &[],
        });
        let object = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("object"),
            entries: &[],
        });
        let bindless = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bindless"),
            entries: &[],
        });

        let cache = PipelineLayoutCache::new();
        let builder = TrinityLayoutBuilder::new(&device, &cache);

        let _layout = builder.bindless(
            &global, 0x1111, &material, 0x2222, &object, 0x3333, &bindless, 0x4444,
        );
        assert_eq!(cache.len(), 1);

        let labels: Vec<_> = cache.labels().collect();
        assert_eq!(labels[0], Some("trinity_bindless".to_string()));
    }

    #[test]
    fn test_trinity_layout_builder_compute() {
        let (device, _queue) = match create_test_device_no_push_constants() {
            Some(d) => d,
            None => {
                eprintln!("Skipping: no GPU adapter available");
                return;
            }
        };

        let compute_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("compute"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let cache = PipelineLayoutCache::new();
        let builder = TrinityLayoutBuilder::new(&device, &cache);

        let _layout = builder.compute(Some("my_compute"), &[&compute_layout], &[0xCCCC]);
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn test_trinity_layout_builder_with_push_constants() {
        let (device, _queue) = match create_test_device() {
            Some(d) => d,
            None => {
                eprintln!("Skipping: no GPU adapter with push constants available");
                return;
            }
        };

        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("test"),
            entries: &[],
        });

        let cache = PipelineLayoutCache::new();
        let builder = TrinityLayoutBuilder::new(&device, &cache);

        let push_constants = &[PushConstantRange {
            stages: ShaderStages::VERTEX | ShaderStages::FRAGMENT,
            range: 0..64,
        }];

        let _pl = builder.with_push_constants(
            Some("custom_push"),
            &[&layout],
            &[0x1234],
            push_constants,
        );
        assert_eq!(cache.len(), 1);
    }
}
