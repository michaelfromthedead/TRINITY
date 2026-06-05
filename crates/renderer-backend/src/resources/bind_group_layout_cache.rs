//! Bind group layout caching for TRINITY.
//!
//! This module provides a cache for wgpu bind group layouts to avoid redundant GPU
//! resource creation. Layouts are identified by a hash key derived from their binding
//! entries, and are shared via `Arc<wgpu::BindGroupLayout>`.
//!
//! # Overview
//!
//! GPU bind group layouts describe the structure of resources bound to a shader.
//! Multiple materials or shaders often need identical layouts. This cache:
//!
//! - Deduplicates layouts with identical binding configurations
//! - Sorts binding entries by index for deterministic hashing
//! - Tracks cache hit/miss metrics
//! - Uses thread-safe interior mutability for concurrent access
//!
//! # Thread Safety
//!
//! The cache uses `RwLock` for concurrent access:
//! - Multiple readers can query the cache simultaneously
//! - Write lock is only held briefly when creating new layouts
//!
//! # Example
//!
//! ```no_run
//! use renderer_backend::resources::bind_group_layout_cache::BindGroupLayoutCache;
//! use wgpu::{BindGroupLayoutEntry, BindingType, ShaderStages, BufferBindingType};
//! use std::num::NonZeroU64;
//!
//! # fn example(device: &wgpu::Device) {
//! let cache = BindGroupLayoutCache::new();
//!
//! let entries = &[
//!     BindGroupLayoutEntry {
//!         binding: 0,
//!         visibility: ShaderStages::VERTEX | ShaderStages::FRAGMENT,
//!         ty: BindingType::Buffer {
//!             ty: BufferBindingType::Uniform,
//!             has_dynamic_offset: false,
//!             min_binding_size: NonZeroU64::new(64),
//!         },
//!         count: None,
//!     },
//! ];
//!
//! // Get or create layout
//! let layout1 = cache.get_or_create(device, Some("camera"), entries);
//! let layout2 = cache.get_or_create(device, Some("camera"), entries);
//!
//! // Same Arc (cache hit)
//! assert!(std::sync::Arc::ptr_eq(&layout1, &layout2));
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
use wgpu::{BindGroupLayout, BindGroupLayoutDescriptor, BindGroupLayoutEntry, Device};

// ============================================================================
// BindGroupLayoutKey
// ============================================================================

/// A hashable key derived from bind group layout entries.
///
/// This struct computes a hash from sorted binding entries, ensuring that
/// layouts with the same bindings (regardless of order) produce the same key.
///
/// # Implementation Notes
///
/// - Entries are sorted by binding index before hashing
/// - All fields that affect GPU layout compatibility are hashed
/// - The hash is computed once and stored for efficient lookup
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BindGroupLayoutKey {
    /// Pre-computed hash of sorted binding entries
    hash: u64,
}

impl BindGroupLayoutKey {
    /// Creates a cache key from binding entries.
    ///
    /// Entries are sorted by binding index before hashing to ensure
    /// deterministic key generation regardless of input order.
    ///
    /// # Arguments
    ///
    /// * `entries` - The bind group layout entries
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::resources::bind_group_layout_cache::BindGroupLayoutKey;
    /// use wgpu::{BindGroupLayoutEntry, BindingType, ShaderStages, BufferBindingType};
    /// use std::num::NonZeroU64;
    ///
    /// let entries = &[
    ///     BindGroupLayoutEntry {
    ///         binding: 1,
    ///         visibility: ShaderStages::FRAGMENT,
    ///         ty: BindingType::Buffer {
    ///             ty: BufferBindingType::Uniform,
    ///             has_dynamic_offset: false,
    ///             min_binding_size: None,
    ///         },
    ///         count: None,
    ///     },
    ///     BindGroupLayoutEntry {
    ///         binding: 0,
    ///         visibility: ShaderStages::VERTEX,
    ///         ty: BindingType::Buffer {
    ///             ty: BufferBindingType::Uniform,
    ///             has_dynamic_offset: false,
    ///             min_binding_size: None,
    ///         },
    ///         count: None,
    ///     },
    /// ];
    ///
    /// let key = BindGroupLayoutKey::from_entries(entries);
    /// ```
    pub fn from_entries(entries: &[BindGroupLayoutEntry]) -> Self {
        use std::collections::hash_map::DefaultHasher;

        // Sort entries by binding index
        let mut sorted: Vec<_> = entries.iter().collect();
        sorted.sort_by_key(|e| e.binding);

        // Hash the sorted entries
        let mut hasher = DefaultHasher::new();

        // Hash entry count for disambiguation
        sorted.len().hash(&mut hasher);

        for entry in sorted {
            entry.binding.hash(&mut hasher);
            Self::hash_binding_type(&entry.ty, &mut hasher);
            entry.visibility.bits().hash(&mut hasher);
            // Hash count (None or Some(NonZeroU32))
            entry.count.map(|c| c.get()).hash(&mut hasher);
        }

        Self {
            hash: hasher.finish(),
        }
    }

    /// Hashes a binding type including all discriminant and key fields.
    fn hash_binding_type(ty: &wgpu::BindingType, hasher: &mut impl Hasher) {
        // Hash discriminant first
        std::mem::discriminant(ty).hash(hasher);

        match ty {
            wgpu::BindingType::Buffer {
                ty,
                has_dynamic_offset,
                min_binding_size,
            } => {
                Self::hash_buffer_binding_type(ty, hasher);
                has_dynamic_offset.hash(hasher);
                min_binding_size.map(|s| s.get()).hash(hasher);
            }
            wgpu::BindingType::Sampler(sampler_binding_type) => {
                std::mem::discriminant(sampler_binding_type).hash(hasher);
            }
            wgpu::BindingType::Texture {
                sample_type,
                view_dimension,
                multisampled,
            } => {
                Self::hash_texture_sample_type(sample_type, hasher);
                std::mem::discriminant(view_dimension).hash(hasher);
                multisampled.hash(hasher);
            }
            wgpu::BindingType::StorageTexture {
                access,
                format,
                view_dimension,
            } => {
                std::mem::discriminant(access).hash(hasher);
                std::mem::discriminant(format).hash(hasher);
                std::mem::discriminant(view_dimension).hash(hasher);
            }
            wgpu::BindingType::AccelerationStructure => {
                // No additional fields to hash
            }
        }
    }

    /// Hashes a texture sample type including filterable float distinction.
    fn hash_texture_sample_type(sample_type: &wgpu::TextureSampleType, hasher: &mut impl Hasher) {
        std::mem::discriminant(sample_type).hash(hasher);
        // For Float variant, also hash the filterable flag
        if let wgpu::TextureSampleType::Float { filterable } = sample_type {
            filterable.hash(hasher);
        }
        // For Sint/Uint/Depth, no additional fields
    }

    /// Hashes a buffer binding type including Storage read_only distinction.
    fn hash_buffer_binding_type(ty: &wgpu::BufferBindingType, hasher: &mut impl Hasher) {
        std::mem::discriminant(ty).hash(hasher);
        // For Storage variant, hash the read_only flag
        if let wgpu::BufferBindingType::Storage { read_only } = ty {
            read_only.hash(hasher);
        }
        // For Uniform, no additional fields
    }

    /// Returns the raw hash value.
    #[inline]
    pub fn hash_value(&self) -> u64 {
        self.hash
    }
}

// ============================================================================
// CachedBindGroupLayout
// ============================================================================

/// A cached bind group layout with metadata.
///
/// This struct wraps an `Arc<wgpu::BindGroupLayout>` with additional metadata
/// for debugging and introspection.
pub struct CachedBindGroupLayout {
    /// The actual layout, wrapped in Arc for shared ownership
    layout: Arc<BindGroupLayout>,
    /// Number of entries in the layout
    entry_count: usize,
    /// Optional label for debugging
    label: Option<String>,
}

impl CachedBindGroupLayout {
    /// Returns a reference to the inner bind group layout.
    #[inline]
    pub fn inner(&self) -> &BindGroupLayout {
        &self.layout
    }

    /// Returns a clone of the Arc-wrapped layout.
    #[inline]
    pub fn arc(&self) -> Arc<BindGroupLayout> {
        Arc::clone(&self.layout)
    }

    /// Returns the number of entries in the layout.
    #[inline]
    pub fn entry_count(&self) -> usize {
        self.entry_count
    }

    /// Returns the optional label.
    #[inline]
    pub fn label(&self) -> Option<&str> {
        self.label.as_deref()
    }
}

impl std::fmt::Debug for CachedBindGroupLayout {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CachedBindGroupLayout")
            .field("entry_count", &self.entry_count)
            .field("label", &self.label)
            .finish()
    }
}

// ============================================================================
// BindGroupLayoutCacheMetrics
// ============================================================================

/// Metrics for monitoring bind group layout cache performance.
///
/// These metrics help identify cache efficiency and potential optimization
/// opportunities.
///
/// # Example
///
/// ```
/// use renderer_backend::resources::bind_group_layout_cache::BindGroupLayoutCacheMetrics;
///
/// let metrics = BindGroupLayoutCacheMetrics::default();
/// assert_eq!(metrics.cache_size, 0);
/// assert_eq!(metrics.hits, 0);
/// assert_eq!(metrics.misses, 0);
/// assert_eq!(metrics.hit_rate, 0.0);
/// ```
#[derive(Debug, Clone, Default)]
pub struct BindGroupLayoutCacheMetrics {
    /// Number of unique layouts in the cache.
    pub cache_size: usize,
    /// Number of cache hits (requested layout already existed).
    pub hits: u64,
    /// Number of cache misses (new layout created).
    pub misses: u64,
    /// Hit rate as a ratio (0.0 to 1.0).
    pub hit_rate: f64,
}

impl BindGroupLayoutCacheMetrics {
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
// BindGroupLayoutCache
// ============================================================================

/// A thread-safe cache for wgpu bind group layouts.
///
/// The cache stores layouts keyed by their binding configuration, ensuring that
/// identical layout configurations share the same GPU resource.
///
/// # Architecture
///
/// ```text
/// BindGroupLayoutCache
/// ├── Cache (HashMap<BindGroupLayoutKey, CachedBindGroupLayout>)
/// │   └── Layouts keyed by sorted binding entries hash
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
/// use renderer_backend::resources::bind_group_layout_cache::BindGroupLayoutCache;
/// use wgpu::{BindGroupLayoutEntry, BindingType, ShaderStages, BufferBindingType};
/// use std::num::NonZeroU64;
///
/// # fn example(device: &wgpu::Device) {
/// let cache = BindGroupLayoutCache::new();
///
/// let entries = &[
///     BindGroupLayoutEntry {
///         binding: 0,
///         visibility: ShaderStages::VERTEX,
///         ty: BindingType::Buffer {
///             ty: BufferBindingType::Uniform,
///             has_dynamic_offset: false,
///             min_binding_size: NonZeroU64::new(64),
///         },
///         count: None,
///     },
/// ];
///
/// // First call creates the layout
/// let layout1 = cache.get_or_create(device, Some("transform"), entries);
///
/// // Second call returns cached layout
/// let layout2 = cache.get_or_create(device, Some("transform"), entries);
///
/// assert!(std::sync::Arc::ptr_eq(&layout1, &layout2));
/// # }
/// ```
pub struct BindGroupLayoutCache {
    /// Cache of layouts keyed by entry configuration.
    cache: RwLock<HashMap<BindGroupLayoutKey, CachedBindGroupLayout>>,
    /// Hit counter (atomic for lock-free updates).
    hits: AtomicU64,
    /// Miss counter (atomic for lock-free updates).
    misses: AtomicU64,
}

impl BindGroupLayoutCache {
    /// Creates a new empty bind group layout cache.
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::resources::bind_group_layout_cache::BindGroupLayoutCache;
    ///
    /// let cache = BindGroupLayoutCache::new();
    /// assert!(cache.is_empty());
    /// ```
    pub fn new() -> Self {
        Self {
            cache: RwLock::new(HashMap::new()),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
        }
    }

    /// Gets or creates a bind group layout matching the given entries.
    ///
    /// If a layout with the same binding configuration already exists in the cache,
    /// it is returned. Otherwise, a new layout is created, cached, and returned.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device for creating layouts
    /// * `label` - Optional debug label for the layout
    /// * `entries` - The bind group layout entries
    ///
    /// # Returns
    ///
    /// An `Arc<BindGroupLayout>` that can be shared across multiple pipeline layouts.
    ///
    /// # Thread Safety
    ///
    /// This method uses a read lock for cache lookups and only acquires
    /// a write lock when creating a new layout (double-check pattern).
    ///
    /// # Example
    ///
    /// ```no_run
    /// use renderer_backend::resources::bind_group_layout_cache::BindGroupLayoutCache;
    /// use wgpu::{BindGroupLayoutEntry, BindingType, ShaderStages, SamplerBindingType};
    ///
    /// # fn example(device: &wgpu::Device) {
    /// let cache = BindGroupLayoutCache::new();
    ///
    /// let entries = &[
    ///     BindGroupLayoutEntry {
    ///         binding: 0,
    ///         visibility: ShaderStages::FRAGMENT,
    ///         ty: BindingType::Sampler(SamplerBindingType::Filtering),
    ///         count: None,
    ///     },
    /// ];
    ///
    /// let layout = cache.get_or_create(device, Some("sampler_layout"), entries);
    /// # }
    /// ```
    pub fn get_or_create(
        &self,
        device: &Device,
        label: Option<&str>,
        entries: &[BindGroupLayoutEntry],
    ) -> Arc<BindGroupLayout> {
        let key = BindGroupLayoutKey::from_entries(entries);

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
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label,
            entries,
        });
        let arc_layout = Arc::new(layout);

        cache.insert(
            key,
            CachedBindGroupLayout {
                layout: Arc::clone(&arc_layout),
                entry_count: entries.len(),
                label: label.map(String::from),
            },
        );

        arc_layout
    }

    /// Checks if a layout exists for the given entries.
    ///
    /// # Arguments
    ///
    /// * `entries` - The bind group layout entries to check
    ///
    /// # Returns
    ///
    /// `true` if a layout with matching entries exists in the cache.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use renderer_backend::resources::bind_group_layout_cache::BindGroupLayoutCache;
    /// use wgpu::{BindGroupLayoutEntry, BindingType, ShaderStages, BufferBindingType};
    ///
    /// # fn example(device: &wgpu::Device) {
    /// let cache = BindGroupLayoutCache::new();
    ///
    /// let entries = &[
    ///     BindGroupLayoutEntry {
    ///         binding: 0,
    ///         visibility: ShaderStages::VERTEX,
    ///         ty: BindingType::Buffer {
    ///             ty: BufferBindingType::Uniform,
    ///             has_dynamic_offset: false,
    ///             min_binding_size: None,
    ///         },
    ///         count: None,
    ///     },
    /// ];
    ///
    /// assert!(!cache.contains(entries));
    /// cache.get_or_create(device, None, entries);
    /// assert!(cache.contains(entries));
    /// # }
    /// ```
    pub fn contains(&self, entries: &[BindGroupLayoutEntry]) -> bool {
        let key = BindGroupLayoutKey::from_entries(entries);
        self.cache.read().contains_key(&key)
    }

    /// Clears the cache, removing all cached layouts.
    ///
    /// Metrics (hits/misses) are also reset to zero.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use renderer_backend::resources::bind_group_layout_cache::BindGroupLayoutCache;
    /// use wgpu::{BindGroupLayoutEntry, BindingType, ShaderStages, BufferBindingType};
    ///
    /// # fn example(device: &wgpu::Device) {
    /// let cache = BindGroupLayoutCache::new();
    ///
    /// // Add a layout
    /// let entries = &[
    ///     BindGroupLayoutEntry {
    ///         binding: 0,
    ///         visibility: ShaderStages::VERTEX,
    ///         ty: BindingType::Buffer {
    ///             ty: BufferBindingType::Uniform,
    ///             has_dynamic_offset: false,
    ///             min_binding_size: None,
    ///         },
    ///         count: None,
    ///     },
    /// ];
    /// cache.get_or_create(device, None, entries);
    /// assert_eq!(cache.len(), 1);
    ///
    /// // Clear
    /// cache.clear();
    /// assert!(cache.is_empty());
    /// # }
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
    /// use renderer_backend::resources::bind_group_layout_cache::BindGroupLayoutCache;
    ///
    /// let cache = BindGroupLayoutCache::new();
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
    /// use renderer_backend::resources::bind_group_layout_cache::BindGroupLayoutCache;
    ///
    /// let cache = BindGroupLayoutCache::new();
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
    /// use renderer_backend::resources::bind_group_layout_cache::BindGroupLayoutCache;
    ///
    /// let cache = BindGroupLayoutCache::new();
    /// let metrics = cache.metrics();
    ///
    /// assert_eq!(metrics.cache_size, 0);
    /// assert_eq!(metrics.hits, 0);
    /// assert_eq!(metrics.misses, 0);
    /// ```
    pub fn metrics(&self) -> BindGroupLayoutCacheMetrics {
        let cache_size = self.cache.read().len();
        let hits = self.hits.load(Ordering::Relaxed);
        let misses = self.misses.load(Ordering::Relaxed);

        BindGroupLayoutCacheMetrics::new(cache_size, hits, misses)
    }

    /// Resets metrics counters to zero without clearing the cache.
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::resources::bind_group_layout_cache::BindGroupLayoutCache;
    ///
    /// let cache = BindGroupLayoutCache::new();
    /// // After some usage...
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

    /// Returns iterator over all cached layout labels (for debugging).
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::resources::bind_group_layout_cache::BindGroupLayoutCache;
    ///
    /// let cache = BindGroupLayoutCache::new();
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

    /// Removes a specific layout from the cache.
    ///
    /// # Arguments
    ///
    /// * `entries` - The bind group layout entries identifying the layout to remove
    ///
    /// # Returns
    ///
    /// `true` if a layout was removed, `false` if no matching layout was found.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use renderer_backend::resources::bind_group_layout_cache::BindGroupLayoutCache;
    /// use wgpu::{BindGroupLayoutEntry, BindingType, ShaderStages, BufferBindingType};
    ///
    /// # fn example(device: &wgpu::Device) {
    /// let cache = BindGroupLayoutCache::new();
    ///
    /// let entries = &[
    ///     BindGroupLayoutEntry {
    ///         binding: 0,
    ///         visibility: ShaderStages::VERTEX,
    ///         ty: BindingType::Buffer {
    ///             ty: BufferBindingType::Uniform,
    ///             has_dynamic_offset: false,
    ///             min_binding_size: None,
    ///         },
    ///         count: None,
    ///     },
    /// ];
    ///
    /// cache.get_or_create(device, None, entries);
    /// assert!(cache.remove(entries));
    /// assert!(!cache.remove(entries)); // Already removed
    /// # }
    /// ```
    pub fn remove(&self, entries: &[BindGroupLayoutEntry]) -> bool {
        let key = BindGroupLayoutKey::from_entries(entries);
        self.cache.write().remove(&key).is_some()
    }
}

impl Default for BindGroupLayoutCache {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for BindGroupLayoutCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let metrics = self.metrics();
        f.debug_struct("BindGroupLayoutCache")
            .field("cache_size", &metrics.cache_size)
            .field("hits", &metrics.hits)
            .field("misses", &metrics.misses)
            .field("hit_rate", &format!("{:.1}%", metrics.hit_rate_percent()))
            .finish()
    }
}

// ============================================================================
// Layout Compatibility
// ============================================================================

/// Checks if two sets of bind group layout entries are compatible.
///
/// Two layouts are compatible if they have the same bindings with matching:
/// - Binding indices
/// - Binding types (same variant)
/// - Shader stage visibility
///
/// Note: This is a simplified compatibility check. Full compatibility may
/// depend on additional factors like min_binding_size.
///
/// # Arguments
///
/// * `a` - First set of entries
/// * `b` - Second set of entries
///
/// # Returns
///
/// `true` if the layouts are compatible.
///
/// # Example
///
/// ```
/// use renderer_backend::resources::bind_group_layout_cache::layouts_compatible;
/// use wgpu::{BindGroupLayoutEntry, BindingType, ShaderStages, BufferBindingType};
///
/// let entries_a = &[
///     BindGroupLayoutEntry {
///         binding: 0,
///         visibility: ShaderStages::VERTEX,
///         ty: BindingType::Buffer {
///             ty: BufferBindingType::Uniform,
///             has_dynamic_offset: false,
///             min_binding_size: None,
///         },
///         count: None,
///     },
/// ];
///
/// let entries_b = &[
///     BindGroupLayoutEntry {
///         binding: 0,
///         visibility: ShaderStages::VERTEX,
///         ty: BindingType::Buffer {
///             ty: BufferBindingType::Uniform,
///             has_dynamic_offset: false,
///             min_binding_size: None,
///         },
///         count: None,
///     },
/// ];
///
/// assert!(layouts_compatible(entries_a, entries_b));
/// ```
pub fn layouts_compatible(a: &[BindGroupLayoutEntry], b: &[BindGroupLayoutEntry]) -> bool {
    if a.len() != b.len() {
        return false;
    }

    // Sort both by binding index
    let mut a_sorted: Vec<_> = a.iter().collect();
    let mut b_sorted: Vec<_> = b.iter().collect();
    a_sorted.sort_by_key(|e| e.binding);
    b_sorted.sort_by_key(|e| e.binding);

    a_sorted
        .iter()
        .zip(b_sorted.iter())
        .all(|(a_entry, b_entry)| {
            a_entry.binding == b_entry.binding
                && binding_types_compatible(&a_entry.ty, &b_entry.ty)
                && a_entry.visibility == b_entry.visibility
                && a_entry.count == b_entry.count
        })
}

/// Checks if two binding types are compatible.
///
/// This checks the outer discriminant and, for Buffer types, also the inner
/// BufferBindingType discriminant.
fn binding_types_compatible(a: &wgpu::BindingType, b: &wgpu::BindingType) -> bool {
    if std::mem::discriminant(a) != std::mem::discriminant(b) {
        return false;
    }

    match (a, b) {
        (
            wgpu::BindingType::Buffer { ty: ty_a, .. },
            wgpu::BindingType::Buffer { ty: ty_b, .. },
        ) => std::mem::discriminant(ty_a) == std::mem::discriminant(ty_b),
        _ => true,
    }
}

/// Checks if two sets of bind group layout entries are strictly equal.
///
/// This performs a full deep comparison including all binding type fields,
/// not just discriminants. Use this when exact layout match is required.
///
/// # Arguments
///
/// * `a` - First set of entries
/// * `b` - Second set of entries
///
/// # Returns
///
/// `true` if the layouts are strictly equal.
///
/// # Example
///
/// ```
/// use renderer_backend::resources::bind_group_layout_cache::layouts_equal;
/// use wgpu::{BindGroupLayoutEntry, BindingType, ShaderStages, BufferBindingType};
/// use std::num::NonZeroU64;
///
/// let entries_a = &[
///     BindGroupLayoutEntry {
///         binding: 0,
///         visibility: ShaderStages::VERTEX,
///         ty: BindingType::Buffer {
///             ty: BufferBindingType::Uniform,
///             has_dynamic_offset: false,
///             min_binding_size: NonZeroU64::new(64),
///         },
///         count: None,
///     },
/// ];
///
/// let entries_b = &[
///     BindGroupLayoutEntry {
///         binding: 0,
///         visibility: ShaderStages::VERTEX,
///         ty: BindingType::Buffer {
///             ty: BufferBindingType::Uniform,
///             has_dynamic_offset: false,
///             min_binding_size: NonZeroU64::new(128), // Different!
///         },
///         count: None,
///     },
/// ];
///
/// // They're compatible (same binding type variant) but not equal
/// assert!(renderer_backend::resources::bind_group_layout_cache::layouts_compatible(entries_a, entries_b));
/// assert!(!layouts_equal(entries_a, entries_b));
/// ```
pub fn layouts_equal(a: &[BindGroupLayoutEntry], b: &[BindGroupLayoutEntry]) -> bool {
    // Use the key comparison - if keys match, entries are fully equal
    BindGroupLayoutKey::from_entries(a) == BindGroupLayoutKey::from_entries(b)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::num::NonZeroU64;
    use wgpu::{
        BufferBindingType, SamplerBindingType, ShaderStages, StorageTextureAccess,
        TextureFormat, TextureSampleType, TextureViewDimension,
    };

    // ========================================================================
    // Helper Functions for Tests
    // ========================================================================

    fn uniform_entry(binding: u32, visibility: ShaderStages) -> BindGroupLayoutEntry {
        BindGroupLayoutEntry {
            binding,
            visibility,
            ty: wgpu::BindingType::Buffer {
                ty: BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }
    }

    fn storage_entry(binding: u32, visibility: ShaderStages) -> BindGroupLayoutEntry {
        BindGroupLayoutEntry {
            binding,
            visibility,
            ty: wgpu::BindingType::Buffer {
                ty: BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }
    }

    fn sampler_entry(binding: u32, visibility: ShaderStages) -> BindGroupLayoutEntry {
        BindGroupLayoutEntry {
            binding,
            visibility,
            ty: wgpu::BindingType::Sampler(SamplerBindingType::Filtering),
            count: None,
        }
    }

    fn texture_entry(binding: u32, visibility: ShaderStages) -> BindGroupLayoutEntry {
        BindGroupLayoutEntry {
            binding,
            visibility,
            ty: wgpu::BindingType::Texture {
                sample_type: TextureSampleType::Float { filterable: true },
                view_dimension: TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        }
    }

    // ========================================================================
    // BindGroupLayoutKey Tests
    // ========================================================================

    #[test]
    fn test_key_from_empty_entries() {
        let entries: &[BindGroupLayoutEntry] = &[];
        let key = BindGroupLayoutKey::from_entries(entries);
        assert_ne!(key.hash_value(), 0); // Should still produce a valid hash
    }

    #[test]
    fn test_key_from_single_entry() {
        let entries = &[uniform_entry(0, ShaderStages::VERTEX)];
        let key = BindGroupLayoutKey::from_entries(entries);
        assert_ne!(key.hash_value(), 0);
    }

    #[test]
    fn test_key_equality_same_entries() {
        let entries1 = &[uniform_entry(0, ShaderStages::VERTEX)];
        let entries2 = &[uniform_entry(0, ShaderStages::VERTEX)];

        let key1 = BindGroupLayoutKey::from_entries(entries1);
        let key2 = BindGroupLayoutKey::from_entries(entries2);

        assert_eq!(key1, key2);
    }

    #[test]
    fn test_key_equality_reordered_entries() {
        // Entries in different order should produce same key
        let entries1 = &[
            uniform_entry(0, ShaderStages::VERTEX),
            sampler_entry(1, ShaderStages::FRAGMENT),
        ];
        let entries2 = &[
            sampler_entry(1, ShaderStages::FRAGMENT),
            uniform_entry(0, ShaderStages::VERTEX),
        ];

        let key1 = BindGroupLayoutKey::from_entries(entries1);
        let key2 = BindGroupLayoutKey::from_entries(entries2);

        assert_eq!(key1, key2);
    }

    #[test]
    fn test_key_inequality_different_binding() {
        let entries1 = &[uniform_entry(0, ShaderStages::VERTEX)];
        let entries2 = &[uniform_entry(1, ShaderStages::VERTEX)];

        let key1 = BindGroupLayoutKey::from_entries(entries1);
        let key2 = BindGroupLayoutKey::from_entries(entries2);

        assert_ne!(key1, key2);
    }

    #[test]
    fn test_key_inequality_different_visibility() {
        let entries1 = &[uniform_entry(0, ShaderStages::VERTEX)];
        let entries2 = &[uniform_entry(0, ShaderStages::FRAGMENT)];

        let key1 = BindGroupLayoutKey::from_entries(entries1);
        let key2 = BindGroupLayoutKey::from_entries(entries2);

        assert_ne!(key1, key2);
    }

    #[test]
    fn test_key_inequality_different_type() {
        let entries1 = &[uniform_entry(0, ShaderStages::VERTEX)];
        let entries2 = &[storage_entry(0, ShaderStages::VERTEX)];

        let key1 = BindGroupLayoutKey::from_entries(entries1);
        let key2 = BindGroupLayoutKey::from_entries(entries2);

        assert_ne!(key1, key2);
    }

    #[test]
    fn test_key_inequality_different_count() {
        let entries1 = &[uniform_entry(0, ShaderStages::VERTEX)];
        let entries2 = &[
            uniform_entry(0, ShaderStages::VERTEX),
            uniform_entry(1, ShaderStages::VERTEX),
        ];

        let key1 = BindGroupLayoutKey::from_entries(entries1);
        let key2 = BindGroupLayoutKey::from_entries(entries2);

        assert_ne!(key1, key2);
    }

    #[test]
    fn test_key_hash_stability() {
        let entries = &[
            uniform_entry(0, ShaderStages::VERTEX),
            texture_entry(1, ShaderStages::FRAGMENT),
            sampler_entry(2, ShaderStages::FRAGMENT),
        ];

        let key1 = BindGroupLayoutKey::from_entries(entries);
        let key2 = BindGroupLayoutKey::from_entries(entries);

        assert_eq!(key1.hash_value(), key2.hash_value());
    }

    #[test]
    fn test_key_as_hashmap_key() {
        let mut map: HashMap<BindGroupLayoutKey, i32> = HashMap::new();

        let key1 = BindGroupLayoutKey::from_entries(&[uniform_entry(0, ShaderStages::VERTEX)]);
        let key2 = BindGroupLayoutKey::from_entries(&[sampler_entry(0, ShaderStages::FRAGMENT)]);

        map.insert(key1, 1);
        map.insert(key2, 2);

        assert_eq!(map.len(), 2);
        assert_eq!(map.get(&key1), Some(&1));
        assert_eq!(map.get(&key2), Some(&2));
    }

    #[test]
    fn test_key_debug_format() {
        let entries = &[uniform_entry(0, ShaderStages::VERTEX)];
        let key = BindGroupLayoutKey::from_entries(entries);
        let debug_str = format!("{:?}", key);

        assert!(debug_str.contains("BindGroupLayoutKey"));
        assert!(debug_str.contains("hash"));
    }

    // ========================================================================
    // Buffer Binding Type Tests
    // ========================================================================

    #[test]
    fn test_key_buffer_dynamic_offset_difference() {
        let entries1 = &[BindGroupLayoutEntry {
            binding: 0,
            visibility: ShaderStages::VERTEX,
            ty: wgpu::BindingType::Buffer {
                ty: BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }];
        let entries2 = &[BindGroupLayoutEntry {
            binding: 0,
            visibility: ShaderStages::VERTEX,
            ty: wgpu::BindingType::Buffer {
                ty: BufferBindingType::Uniform,
                has_dynamic_offset: true,
                min_binding_size: None,
            },
            count: None,
        }];

        let key1 = BindGroupLayoutKey::from_entries(entries1);
        let key2 = BindGroupLayoutKey::from_entries(entries2);

        assert_ne!(key1, key2);
    }

    #[test]
    fn test_key_buffer_min_binding_size_difference() {
        let entries1 = &[BindGroupLayoutEntry {
            binding: 0,
            visibility: ShaderStages::VERTEX,
            ty: wgpu::BindingType::Buffer {
                ty: BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }];
        let entries2 = &[BindGroupLayoutEntry {
            binding: 0,
            visibility: ShaderStages::VERTEX,
            ty: wgpu::BindingType::Buffer {
                ty: BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: NonZeroU64::new(64),
            },
            count: None,
        }];

        let key1 = BindGroupLayoutKey::from_entries(entries1);
        let key2 = BindGroupLayoutKey::from_entries(entries2);

        assert_ne!(key1, key2);
    }

    #[test]
    fn test_key_storage_readonly_difference() {
        let entries1 = &[BindGroupLayoutEntry {
            binding: 0,
            visibility: ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }];
        let entries2 = &[BindGroupLayoutEntry {
            binding: 0,
            visibility: ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }];

        let key1 = BindGroupLayoutKey::from_entries(entries1);
        let key2 = BindGroupLayoutKey::from_entries(entries2);

        assert_ne!(key1, key2);
    }

    // ========================================================================
    // Sampler Binding Type Tests
    // ========================================================================

    #[test]
    fn test_key_sampler_type_difference() {
        let entries1 = &[BindGroupLayoutEntry {
            binding: 0,
            visibility: ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Sampler(SamplerBindingType::Filtering),
            count: None,
        }];
        let entries2 = &[BindGroupLayoutEntry {
            binding: 0,
            visibility: ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Sampler(SamplerBindingType::NonFiltering),
            count: None,
        }];

        let key1 = BindGroupLayoutKey::from_entries(entries1);
        let key2 = BindGroupLayoutKey::from_entries(entries2);

        assert_ne!(key1, key2);
    }

    #[test]
    fn test_key_sampler_comparison() {
        let entries1 = &[BindGroupLayoutEntry {
            binding: 0,
            visibility: ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Sampler(SamplerBindingType::Filtering),
            count: None,
        }];
        let entries2 = &[BindGroupLayoutEntry {
            binding: 0,
            visibility: ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Sampler(SamplerBindingType::Comparison),
            count: None,
        }];

        let key1 = BindGroupLayoutKey::from_entries(entries1);
        let key2 = BindGroupLayoutKey::from_entries(entries2);

        assert_ne!(key1, key2);
    }

    // ========================================================================
    // Texture Binding Type Tests
    // ========================================================================

    #[test]
    fn test_key_texture_sample_type_difference() {
        let entries1 = &[BindGroupLayoutEntry {
            binding: 0,
            visibility: ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: TextureSampleType::Float { filterable: true },
                view_dimension: TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        }];
        let entries2 = &[BindGroupLayoutEntry {
            binding: 0,
            visibility: ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: TextureSampleType::Uint,
                view_dimension: TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        }];

        let key1 = BindGroupLayoutKey::from_entries(entries1);
        let key2 = BindGroupLayoutKey::from_entries(entries2);

        assert_ne!(key1, key2);
    }

    #[test]
    fn test_key_texture_filterable_difference() {
        let entries1 = &[BindGroupLayoutEntry {
            binding: 0,
            visibility: ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: TextureSampleType::Float { filterable: true },
                view_dimension: TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        }];
        let entries2 = &[BindGroupLayoutEntry {
            binding: 0,
            visibility: ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: TextureSampleType::Float { filterable: false },
                view_dimension: TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        }];

        let key1 = BindGroupLayoutKey::from_entries(entries1);
        let key2 = BindGroupLayoutKey::from_entries(entries2);

        assert_ne!(key1, key2);
    }

    #[test]
    fn test_key_texture_dimension_difference() {
        let entries1 = &[BindGroupLayoutEntry {
            binding: 0,
            visibility: ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: TextureSampleType::Float { filterable: true },
                view_dimension: TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        }];
        let entries2 = &[BindGroupLayoutEntry {
            binding: 0,
            visibility: ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: TextureSampleType::Float { filterable: true },
                view_dimension: TextureViewDimension::D3,
                multisampled: false,
            },
            count: None,
        }];

        let key1 = BindGroupLayoutKey::from_entries(entries1);
        let key2 = BindGroupLayoutKey::from_entries(entries2);

        assert_ne!(key1, key2);
    }

    #[test]
    fn test_key_texture_multisampled_difference() {
        let entries1 = &[BindGroupLayoutEntry {
            binding: 0,
            visibility: ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: TextureSampleType::Float { filterable: true },
                view_dimension: TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        }];
        let entries2 = &[BindGroupLayoutEntry {
            binding: 0,
            visibility: ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: TextureSampleType::Float { filterable: true },
                view_dimension: TextureViewDimension::D2,
                multisampled: true,
            },
            count: None,
        }];

        let key1 = BindGroupLayoutKey::from_entries(entries1);
        let key2 = BindGroupLayoutKey::from_entries(entries2);

        assert_ne!(key1, key2);
    }

    // ========================================================================
    // Storage Texture Tests
    // ========================================================================

    #[test]
    fn test_key_storage_texture_access_difference() {
        let entries1 = &[BindGroupLayoutEntry {
            binding: 0,
            visibility: ShaderStages::COMPUTE,
            ty: wgpu::BindingType::StorageTexture {
                access: StorageTextureAccess::WriteOnly,
                format: TextureFormat::Rgba8Unorm,
                view_dimension: TextureViewDimension::D2,
            },
            count: None,
        }];
        let entries2 = &[BindGroupLayoutEntry {
            binding: 0,
            visibility: ShaderStages::COMPUTE,
            ty: wgpu::BindingType::StorageTexture {
                access: StorageTextureAccess::ReadOnly,
                format: TextureFormat::Rgba8Unorm,
                view_dimension: TextureViewDimension::D2,
            },
            count: None,
        }];

        let key1 = BindGroupLayoutKey::from_entries(entries1);
        let key2 = BindGroupLayoutKey::from_entries(entries2);

        assert_ne!(key1, key2);
    }

    #[test]
    fn test_key_storage_texture_format_difference() {
        let entries1 = &[BindGroupLayoutEntry {
            binding: 0,
            visibility: ShaderStages::COMPUTE,
            ty: wgpu::BindingType::StorageTexture {
                access: StorageTextureAccess::WriteOnly,
                format: TextureFormat::Rgba8Unorm,
                view_dimension: TextureViewDimension::D2,
            },
            count: None,
        }];
        let entries2 = &[BindGroupLayoutEntry {
            binding: 0,
            visibility: ShaderStages::COMPUTE,
            ty: wgpu::BindingType::StorageTexture {
                access: StorageTextureAccess::WriteOnly,
                format: TextureFormat::Rgba32Float,
                view_dimension: TextureViewDimension::D2,
            },
            count: None,
        }];

        let key1 = BindGroupLayoutKey::from_entries(entries1);
        let key2 = BindGroupLayoutKey::from_entries(entries2);

        assert_ne!(key1, key2);
    }

    // ========================================================================
    // Metrics Tests
    // ========================================================================

    #[test]
    fn test_metrics_default() {
        let metrics = BindGroupLayoutCacheMetrics::default();

        assert_eq!(metrics.cache_size, 0);
        assert_eq!(metrics.hits, 0);
        assert_eq!(metrics.misses, 0);
        assert_eq!(metrics.hit_rate, 0.0);
    }

    #[test]
    fn test_metrics_new() {
        let metrics = BindGroupLayoutCacheMetrics::new(5, 80, 20);

        assert_eq!(metrics.cache_size, 5);
        assert_eq!(metrics.hits, 80);
        assert_eq!(metrics.misses, 20);
        assert!((metrics.hit_rate - 0.8).abs() < 0.001);
    }

    #[test]
    fn test_metrics_total_requests() {
        let metrics = BindGroupLayoutCacheMetrics::new(3, 50, 25);
        assert_eq!(metrics.total_requests(), 75);
    }

    #[test]
    fn test_metrics_is_empty() {
        let empty = BindGroupLayoutCacheMetrics::new(0, 0, 0);
        assert!(empty.is_empty());

        let non_empty = BindGroupLayoutCacheMetrics::new(1, 0, 1);
        assert!(!non_empty.is_empty());
    }

    #[test]
    fn test_metrics_hit_rate_percent() {
        let metrics = BindGroupLayoutCacheMetrics::new(2, 75, 25);
        assert!((metrics.hit_rate_percent() - 75.0).abs() < 0.001);
    }

    #[test]
    fn test_metrics_zero_total() {
        let metrics = BindGroupLayoutCacheMetrics::new(0, 0, 0);
        assert_eq!(metrics.hit_rate, 0.0);
        assert_eq!(metrics.total_requests(), 0);
    }

    #[test]
    fn test_metrics_clone() {
        let metrics = BindGroupLayoutCacheMetrics::new(10, 100, 50);
        let cloned = metrics.clone();

        assert_eq!(cloned.cache_size, metrics.cache_size);
        assert_eq!(cloned.hits, metrics.hits);
        assert_eq!(cloned.misses, metrics.misses);
        assert_eq!(cloned.hit_rate, metrics.hit_rate);
    }

    #[test]
    fn test_metrics_debug_format() {
        let metrics = BindGroupLayoutCacheMetrics::new(3, 10, 5);
        let debug_str = format!("{:?}", metrics);

        assert!(debug_str.contains("BindGroupLayoutCacheMetrics"));
        assert!(debug_str.contains("cache_size"));
    }

    // ========================================================================
    // Cache Tests (no device required)
    // ========================================================================

    #[test]
    fn test_cache_new() {
        let cache = BindGroupLayoutCache::new();
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn test_cache_default() {
        let cache = BindGroupLayoutCache::default();
        assert!(cache.is_empty());
    }

    #[test]
    fn test_cache_metrics_initial() {
        let cache = BindGroupLayoutCache::new();
        let metrics = cache.metrics();

        assert_eq!(metrics.cache_size, 0);
        assert_eq!(metrics.hits, 0);
        assert_eq!(metrics.misses, 0);
        assert_eq!(metrics.hit_rate, 0.0);
    }

    #[test]
    fn test_cache_reset_metrics() {
        let cache = BindGroupLayoutCache::new();

        // Simulate some activity by directly manipulating atomics
        cache.hits.fetch_add(10, Ordering::Relaxed);
        cache.misses.fetch_add(5, Ordering::Relaxed);

        cache.reset_metrics();

        let metrics = cache.metrics();
        assert_eq!(metrics.hits, 0);
        assert_eq!(metrics.misses, 0);
    }

    #[test]
    fn test_cache_labels_empty() {
        let cache = BindGroupLayoutCache::new();
        let labels: Vec<_> = cache.labels().collect();
        assert!(labels.is_empty());
    }

    #[test]
    fn test_cache_debug_format() {
        let cache = BindGroupLayoutCache::new();
        let debug_str = format!("{:?}", cache);

        assert!(debug_str.contains("BindGroupLayoutCache"));
        assert!(debug_str.contains("cache_size"));
    }

    // ========================================================================
    // Layout Compatibility Tests
    // ========================================================================

    #[test]
    fn test_layouts_compatible_identical() {
        let entries_a = &[uniform_entry(0, ShaderStages::VERTEX)];
        let entries_b = &[uniform_entry(0, ShaderStages::VERTEX)];

        assert!(layouts_compatible(entries_a, entries_b));
    }

    #[test]
    fn test_layouts_compatible_reordered() {
        let entries_a = &[
            uniform_entry(0, ShaderStages::VERTEX),
            sampler_entry(1, ShaderStages::FRAGMENT),
        ];
        let entries_b = &[
            sampler_entry(1, ShaderStages::FRAGMENT),
            uniform_entry(0, ShaderStages::VERTEX),
        ];

        assert!(layouts_compatible(entries_a, entries_b));
    }

    #[test]
    fn test_layouts_incompatible_different_count() {
        let entries_a = &[uniform_entry(0, ShaderStages::VERTEX)];
        let entries_b = &[
            uniform_entry(0, ShaderStages::VERTEX),
            uniform_entry(1, ShaderStages::VERTEX),
        ];

        assert!(!layouts_compatible(entries_a, entries_b));
    }

    #[test]
    fn test_layouts_incompatible_different_binding() {
        let entries_a = &[uniform_entry(0, ShaderStages::VERTEX)];
        let entries_b = &[uniform_entry(1, ShaderStages::VERTEX)];

        assert!(!layouts_compatible(entries_a, entries_b));
    }

    #[test]
    fn test_layouts_incompatible_different_type() {
        let entries_a = &[uniform_entry(0, ShaderStages::VERTEX)];
        let entries_b = &[storage_entry(0, ShaderStages::VERTEX)];

        assert!(!layouts_compatible(entries_a, entries_b));
    }

    #[test]
    fn test_layouts_incompatible_different_visibility() {
        let entries_a = &[uniform_entry(0, ShaderStages::VERTEX)];
        let entries_b = &[uniform_entry(0, ShaderStages::FRAGMENT)];

        assert!(!layouts_compatible(entries_a, entries_b));
    }

    #[test]
    fn test_layouts_compatible_empty() {
        let entries_a: &[BindGroupLayoutEntry] = &[];
        let entries_b: &[BindGroupLayoutEntry] = &[];

        assert!(layouts_compatible(entries_a, entries_b));
    }

    // ========================================================================
    // layouts_equal Tests
    // ========================================================================

    #[test]
    fn test_layouts_equal_identical() {
        let entries_a = &[uniform_entry(0, ShaderStages::VERTEX)];
        let entries_b = &[uniform_entry(0, ShaderStages::VERTEX)];

        assert!(layouts_equal(entries_a, entries_b));
    }

    #[test]
    fn test_layouts_equal_different_min_binding_size() {
        let entries_a = &[BindGroupLayoutEntry {
            binding: 0,
            visibility: ShaderStages::VERTEX,
            ty: wgpu::BindingType::Buffer {
                ty: BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: NonZeroU64::new(64),
            },
            count: None,
        }];
        let entries_b = &[BindGroupLayoutEntry {
            binding: 0,
            visibility: ShaderStages::VERTEX,
            ty: wgpu::BindingType::Buffer {
                ty: BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: NonZeroU64::new(128),
            },
            count: None,
        }];

        assert!(!layouts_equal(entries_a, entries_b));
    }

    // ========================================================================
    // CachedBindGroupLayout Tests
    // ========================================================================

    #[test]
    fn test_cached_layout_debug() {
        // Can't test with real layout without device, but can test Debug impl path
        // This test just ensures the Debug implementation compiles
        let _format_template = "CachedBindGroupLayout { entry_count: 1, label: Some(\"test\") }";
    }

    // ========================================================================
    // Key with Array Count Tests
    // ========================================================================

    #[test]
    fn test_key_array_count_difference() {
        use std::num::NonZeroU32;

        let entries1 = &[BindGroupLayoutEntry {
            binding: 0,
            visibility: ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: TextureSampleType::Float { filterable: true },
                view_dimension: TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        }];
        let entries2 = &[BindGroupLayoutEntry {
            binding: 0,
            visibility: ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: TextureSampleType::Float { filterable: true },
                view_dimension: TextureViewDimension::D2,
                multisampled: false,
            },
            count: NonZeroU32::new(4),
        }];

        let key1 = BindGroupLayoutKey::from_entries(entries1);
        let key2 = BindGroupLayoutKey::from_entries(entries2);

        assert_ne!(key1, key2);
    }

    #[test]
    fn test_key_array_count_same() {
        use std::num::NonZeroU32;

        let entries1 = &[BindGroupLayoutEntry {
            binding: 0,
            visibility: ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: TextureSampleType::Float { filterable: true },
                view_dimension: TextureViewDimension::D2,
                multisampled: false,
            },
            count: NonZeroU32::new(8),
        }];
        let entries2 = &[BindGroupLayoutEntry {
            binding: 0,
            visibility: ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: TextureSampleType::Float { filterable: true },
                view_dimension: TextureViewDimension::D2,
                multisampled: false,
            },
            count: NonZeroU32::new(8),
        }];

        let key1 = BindGroupLayoutKey::from_entries(entries1);
        let key2 = BindGroupLayoutKey::from_entries(entries2);

        assert_eq!(key1, key2);
    }

    // ========================================================================
    // Complex Layout Tests
    // ========================================================================

    #[test]
    fn test_key_complex_pbr_layout() {
        // Simulate a typical PBR material layout
        let entries = &[
            // Camera uniform
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::VERTEX | ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: NonZeroU64::new(144),
                },
                count: None,
            },
            // Albedo texture
            BindGroupLayoutEntry {
                binding: 1,
                visibility: ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: TextureSampleType::Float { filterable: true },
                    view_dimension: TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            // Sampler
            BindGroupLayoutEntry {
                binding: 2,
                visibility: ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(SamplerBindingType::Filtering),
                count: None,
            },
            // Normal map
            BindGroupLayoutEntry {
                binding: 3,
                visibility: ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: TextureSampleType::Float { filterable: true },
                    view_dimension: TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
        ];

        let key1 = BindGroupLayoutKey::from_entries(entries);
        let key2 = BindGroupLayoutKey::from_entries(entries);

        assert_eq!(key1, key2);
    }

    #[test]
    fn test_key_compute_layout() {
        // Simulate a compute shader layout
        let entries = &[
            // Input buffer
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // Output buffer
            BindGroupLayoutEntry {
                binding: 1,
                visibility: ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // Output texture
            BindGroupLayoutEntry {
                binding: 2,
                visibility: ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    access: StorageTextureAccess::WriteOnly,
                    format: TextureFormat::Rgba8Unorm,
                    view_dimension: TextureViewDimension::D2,
                },
                count: None,
            },
        ];

        let key = BindGroupLayoutKey::from_entries(entries);
        assert_ne!(key.hash_value(), 0);
    }

    // ========================================================================
    // Thread Safety Tests (no device, just cache structure)
    // ========================================================================

    #[test]
    fn test_cache_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<BindGroupLayoutCache>();
    }

    #[test]
    fn test_key_copy() {
        let entries = &[uniform_entry(0, ShaderStages::VERTEX)];
        let key1 = BindGroupLayoutKey::from_entries(entries);
        let key2 = key1; // Copy
        assert_eq!(key1, key2);
    }
}
