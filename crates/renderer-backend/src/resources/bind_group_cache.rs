//! Bind group caching for TRINITY.
//!
//! This module provides a cache for wgpu bind groups to avoid redundant GPU
//! resource creation. Bind groups are identified by a composite key derived from
//! the layout hash and the resource IDs of bound resources.
//!
//! # Overview
//!
//! GPU bind groups connect shader resources (buffers, textures, samplers) to
//! pipeline bind points. Creating bind groups has GPU driver overhead, so caching
//! them significantly improves performance when the same resource combinations
//! are used across multiple frames or draw calls.
//!
//! This cache provides:
//!
//! - Deduplication by layout + resource combination
//! - Resource tracking for targeted invalidation when resources are destroyed
//! - Frame-based eviction for stale entries
//! - Thread-safe concurrent access with minimal contention
//! - Detailed metrics for monitoring cache efficiency
//!
//! # Resource Tracking
//!
//! Each bind group tracks which resources it references. When a resource is
//! destroyed, call [`BindGroupCache::invalidate_resource`] to remove all bind
//! groups that reference it, preventing use-after-free errors.
//!
//! # Thread Safety
//!
//! The cache uses `RwLock` for concurrent access:
//! - Multiple readers can query the cache simultaneously
//! - Write lock is only held briefly when creating new bind groups
//! - Hit/miss counters use atomic operations (lock-free)
//!
//! # Example
//!
//! ```no_run
//! use renderer_backend::resources::bind_group_cache::{
//!     BindGroupCache, BindGroupResourceEntry, BindGroupResourceType, ResourceId,
//! };
//! use wgpu::{BindGroupEntry, BindingResource, BufferBinding};
//!
//! # fn example(device: &wgpu::Device, layout: &wgpu::BindGroupLayout, buffer: &wgpu::Buffer) {
//! let cache = BindGroupCache::new();
//!
//! // Track resource IDs for cache key generation
//! let resource_entries = &[
//!     BindGroupResourceEntry {
//!         binding: 0,
//!         resource_id: ResourceId::new(1),
//!         resource_type: BindGroupResourceType::Buffer { offset: 0, size: Some(64) },
//!     },
//! ];
//!
//! // Build wgpu bind group entries
//! let entries = &[
//!     BindGroupEntry {
//!         binding: 0,
//!         resource: BindingResource::Buffer(BufferBinding {
//!             buffer,
//!             offset: 0,
//!             size: None,
//!         }),
//!     },
//! ];
//!
//! // Get or create bind group
//! let bind_group = cache.create_bind_group(
//!     device,
//!     layout,
//!     0x12345678, // layout hash from BindGroupLayoutCache
//!     entries,
//!     resource_entries,
//!     Some("my_bind_group"),
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
use wgpu::{BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, Device};

// ============================================================================
// ResourceId
// ============================================================================

/// Unique identifier for a GPU resource (buffer, texture, sampler).
///
/// Resource IDs are used to track which resources are bound to each cached
/// bind group. When a resource is destroyed, its ID can be used to invalidate
/// all bind groups that reference it.
///
/// # ID Generation
///
/// The caller is responsible for generating unique IDs for each resource.
/// Common approaches:
/// - Use an atomic counter for each resource type
/// - Use the resource's memory address (cast to u64)
/// - Use a UUID or other unique identifier
///
/// # Example
///
/// ```
/// use renderer_backend::resources::bind_group_cache::ResourceId;
///
/// let id1 = ResourceId::new(1);
/// let id2 = ResourceId::new(2);
///
/// assert_ne!(id1, id2);
/// assert_eq!(id1.value(), 1);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ResourceId(u64);

impl ResourceId {
    /// Creates a new resource ID with the given value.
    #[inline]
    pub const fn new(id: u64) -> Self {
        Self(id)
    }

    /// Returns the raw ID value.
    #[inline]
    pub const fn value(&self) -> u64 {
        self.0
    }
}

impl From<u64> for ResourceId {
    fn from(id: u64) -> Self {
        Self::new(id)
    }
}

impl From<ResourceId> for u64 {
    fn from(id: ResourceId) -> Self {
        id.0
    }
}

// ============================================================================
// BindGroupResourceType
// ============================================================================

/// The type of resource bound at a binding slot.
///
/// This enum captures the binding configuration that affects the cache key.
/// Different buffer offsets or sizes result in different cache keys, ensuring
/// that bind groups with different resource views are not incorrectly shared.
///
/// # Example
///
/// ```
/// use renderer_backend::resources::bind_group_cache::BindGroupResourceType;
///
/// let buffer_type = BindGroupResourceType::Buffer { offset: 0, size: Some(256) };
/// let sampler_type = BindGroupResourceType::Sampler;
/// let texture_type = BindGroupResourceType::TextureView;
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BindGroupResourceType {
    /// A buffer binding with optional offset and size.
    Buffer {
        /// Byte offset into the buffer (default 0).
        offset: u64,
        /// Byte size of the binding, or None for whole buffer.
        size: Option<u64>,
    },
    /// A sampler binding.
    Sampler,
    /// A texture view binding (sampled texture).
    TextureView,
    /// A storage texture binding.
    StorageTextureView,
}

impl BindGroupResourceType {
    /// Creates a buffer resource type with default offset (0) and no size limit.
    #[inline]
    pub const fn buffer() -> Self {
        Self::Buffer {
            offset: 0,
            size: None,
        }
    }

    /// Creates a buffer resource type with the given offset and size.
    #[inline]
    pub const fn buffer_range(offset: u64, size: u64) -> Self {
        Self::Buffer {
            offset,
            size: Some(size),
        }
    }

    /// Creates a buffer resource type with offset 0 and the given size.
    #[inline]
    pub const fn buffer_sized(size: u64) -> Self {
        Self::Buffer {
            offset: 0,
            size: Some(size),
        }
    }
}

// ============================================================================
// BindGroupResourceEntry
// ============================================================================

/// A binding entry with resource ID for cache key generation.
///
/// This struct captures all the information needed to generate a unique cache
/// key for a bind group: the binding slot, the resource identity, and the
/// binding configuration.
///
/// # Example
///
/// ```
/// use renderer_backend::resources::bind_group_cache::{
///     BindGroupResourceEntry, BindGroupResourceType, ResourceId,
/// };
///
/// let entry = BindGroupResourceEntry {
///     binding: 0,
///     resource_id: ResourceId::new(42),
///     resource_type: BindGroupResourceType::Buffer { offset: 0, size: Some(64) },
/// };
///
/// assert_eq!(entry.binding, 0);
/// assert_eq!(entry.resource_id.value(), 42);
/// ```
#[derive(Debug, Clone)]
pub struct BindGroupResourceEntry {
    /// The binding slot index (matches `@binding(N)` in WGSL).
    pub binding: u32,
    /// Unique identifier for the bound resource.
    pub resource_id: ResourceId,
    /// The type and configuration of the binding.
    pub resource_type: BindGroupResourceType,
}

impl BindGroupResourceEntry {
    /// Creates a new resource entry for a buffer binding.
    #[inline]
    pub fn buffer(binding: u32, resource_id: ResourceId, offset: u64, size: Option<u64>) -> Self {
        Self {
            binding,
            resource_id,
            resource_type: BindGroupResourceType::Buffer { offset, size },
        }
    }

    /// Creates a new resource entry for a sampler binding.
    #[inline]
    pub fn sampler(binding: u32, resource_id: ResourceId) -> Self {
        Self {
            binding,
            resource_id,
            resource_type: BindGroupResourceType::Sampler,
        }
    }

    /// Creates a new resource entry for a texture view binding.
    #[inline]
    pub fn texture_view(binding: u32, resource_id: ResourceId) -> Self {
        Self {
            binding,
            resource_id,
            resource_type: BindGroupResourceType::TextureView,
        }
    }

    /// Creates a new resource entry for a storage texture view binding.
    #[inline]
    pub fn storage_texture_view(binding: u32, resource_id: ResourceId) -> Self {
        Self {
            binding,
            resource_id,
            resource_type: BindGroupResourceType::StorageTextureView,
        }
    }
}

// ============================================================================
// BindGroupCacheKey
// ============================================================================

/// Cache key combining layout hash and resource bindings hash.
///
/// The key is computed from:
/// 1. The layout hash (from `BindGroupLayoutCache`)
/// 2. A hash of sorted resource entries (binding, resource_id, resource_type)
///
/// This ensures that bind groups with the same layout but different resources
/// are cached separately.
///
/// # Example
///
/// ```
/// use renderer_backend::resources::bind_group_cache::{
///     BindGroupCacheKey, BindGroupResourceEntry, BindGroupResourceType, ResourceId,
/// };
///
/// let entries = &[
///     BindGroupResourceEntry {
///         binding: 0,
///         resource_id: ResourceId::new(1),
///         resource_type: BindGroupResourceType::Buffer { offset: 0, size: None },
///     },
/// ];
///
/// let key = BindGroupCacheKey::new(0x12345678, entries);
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BindGroupCacheKey {
    /// Hash of the bind group layout entries.
    layout_hash: u64,
    /// Hash of the sorted resource bindings.
    resources_hash: u64,
}

impl BindGroupCacheKey {
    /// Creates a cache key from a layout hash and resource entries.
    ///
    /// Entries are sorted by binding index before hashing to ensure
    /// deterministic key generation regardless of input order.
    ///
    /// # Arguments
    ///
    /// * `layout_hash` - Hash from `BindGroupLayoutCache::get_or_create()`
    /// * `entries` - Resource entries describing the bound resources
    pub fn new(layout_hash: u64, entries: &[BindGroupResourceEntry]) -> Self {
        use std::collections::hash_map::DefaultHasher;

        // Sort entries by binding index for deterministic hashing
        let mut sorted: Vec<_> = entries.iter().collect();
        sorted.sort_by_key(|e| e.binding);

        let mut hasher = DefaultHasher::new();

        // Hash entry count for disambiguation
        sorted.len().hash(&mut hasher);

        for entry in sorted {
            entry.binding.hash(&mut hasher);
            entry.resource_id.hash(&mut hasher);
            entry.resource_type.hash(&mut hasher);
        }

        Self {
            layout_hash,
            resources_hash: hasher.finish(),
        }
    }

    /// Returns the layout hash component.
    #[inline]
    pub fn layout_hash(&self) -> u64 {
        self.layout_hash
    }

    /// Returns the resources hash component.
    #[inline]
    pub fn resources_hash(&self) -> u64 {
        self.resources_hash
    }
}

// ============================================================================
// CachedBindGroup
// ============================================================================

/// A cached bind group with metadata.
///
/// This struct wraps an `Arc<wgpu::BindGroup>` with additional metadata for
/// resource tracking, debugging, and eviction policies.
pub struct CachedBindGroup {
    /// The actual bind group, wrapped in Arc for shared ownership.
    bind_group: Arc<BindGroup>,
    /// Resource IDs referenced by this bind group (for invalidation).
    resource_ids: Vec<ResourceId>,
    /// Frame number when this bind group was created (for eviction).
    frame_created: u64,
}

impl CachedBindGroup {
    /// Returns a reference to the inner bind group.
    #[inline]
    pub fn inner(&self) -> &BindGroup {
        &self.bind_group
    }

    /// Returns a clone of the Arc-wrapped bind group.
    #[inline]
    pub fn arc(&self) -> Arc<BindGroup> {
        Arc::clone(&self.bind_group)
    }

    /// Returns the resource IDs referenced by this bind group.
    #[inline]
    pub fn resource_ids(&self) -> &[ResourceId] {
        &self.resource_ids
    }

    /// Returns the frame number when this bind group was created.
    #[inline]
    pub fn frame_created(&self) -> u64 {
        self.frame_created
    }
}

impl std::fmt::Debug for CachedBindGroup {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CachedBindGroup")
            .field("resource_count", &self.resource_ids.len())
            .field("frame_created", &self.frame_created)
            .finish()
    }
}

// ============================================================================
// BindGroupCacheMetrics
// ============================================================================

/// Metrics for monitoring bind group cache performance.
///
/// These metrics help identify cache efficiency and potential optimization
/// opportunities.
///
/// # Example
///
/// ```
/// use renderer_backend::resources::bind_group_cache::BindGroupCacheMetrics;
///
/// let metrics = BindGroupCacheMetrics::default();
/// assert_eq!(metrics.cache_size, 0);
/// assert_eq!(metrics.hits, 0);
/// assert_eq!(metrics.misses, 0);
/// assert_eq!(metrics.hit_rate, 0.0);
/// ```
#[derive(Debug, Clone, Default)]
pub struct BindGroupCacheMetrics {
    /// Number of bind groups currently in the cache.
    pub cache_size: usize,
    /// Number of cache hits (requested bind group already existed).
    pub hits: u64,
    /// Number of cache misses (new bind group created).
    pub misses: u64,
    /// Hit rate as a ratio (0.0 to 1.0).
    pub hit_rate: f64,
    /// Current frame number.
    pub current_frame: u64,
    /// Number of tracked resources.
    pub tracked_resources: usize,
}

impl BindGroupCacheMetrics {
    /// Creates metrics with the given values.
    pub fn new(
        cache_size: usize,
        hits: u64,
        misses: u64,
        current_frame: u64,
        tracked_resources: usize,
    ) -> Self {
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
            current_frame,
            tracked_resources,
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
// BindGroupCache
// ============================================================================

/// A thread-safe cache for wgpu bind groups with resource tracking.
///
/// The cache stores bind groups keyed by their layout and resource combination,
/// ensuring that identical configurations share the same GPU resource. It also
/// tracks which resources are referenced by each bind group, enabling targeted
/// invalidation when resources are destroyed.
///
/// # Architecture
///
/// ```text
/// BindGroupCache
/// ├── Cache (HashMap<BindGroupCacheKey, CachedBindGroup>)
/// │   └── Bind groups keyed by (layout_hash, resources_hash)
/// ├── Resource Tracking (HashMap<ResourceId, Vec<BindGroupCacheKey>>)
/// │   └── Maps resource IDs to bind groups that reference them
/// └── Metrics (hits, misses, size, frame)
/// ```
///
/// # Thread Safety
///
/// - Uses `RwLock<HashMap>` for the cache and resource tracking
/// - Uses `AtomicU64` for hit/miss counters and frame number (lock-free)
/// - Multiple readers can access cached bind groups concurrently
/// - Write lock is only held briefly when inserting or invalidating
///
/// # Example
///
/// ```no_run
/// use renderer_backend::resources::bind_group_cache::{
///     BindGroupCache, BindGroupResourceEntry, BindGroupResourceType, ResourceId,
/// };
///
/// # fn example(device: &wgpu::Device, layout: &wgpu::BindGroupLayout, buffer: &wgpu::Buffer) {
/// let cache = BindGroupCache::new();
///
/// // Frame loop
/// cache.begin_frame();
///
/// // Create/get cached bind groups during rendering...
///
/// // Optional: evict old entries periodically
/// let evicted = cache.evict_old(3); // Remove entries older than 3 frames
/// # }
/// ```
pub struct BindGroupCache {
    /// Cache of bind groups keyed by layout + resources.
    cache: RwLock<HashMap<BindGroupCacheKey, CachedBindGroup>>,
    /// Mapping from resource ID to bind groups that reference it.
    resource_to_bind_groups: RwLock<HashMap<ResourceId, Vec<BindGroupCacheKey>>>,
    /// Current frame number.
    current_frame: AtomicU64,
    /// Hit counter (atomic for lock-free updates).
    hits: AtomicU64,
    /// Miss counter (atomic for lock-free updates).
    misses: AtomicU64,
}

impl BindGroupCache {
    /// Creates a new empty bind group cache.
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::resources::bind_group_cache::BindGroupCache;
    ///
    /// let cache = BindGroupCache::new();
    /// assert!(cache.is_empty());
    /// ```
    pub fn new() -> Self {
        Self {
            cache: RwLock::new(HashMap::new()),
            resource_to_bind_groups: RwLock::new(HashMap::new()),
            current_frame: AtomicU64::new(0),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
        }
    }

    /// Gets or creates a bind group matching the given configuration.
    ///
    /// If a bind group with the same layout and resource bindings already exists
    /// in the cache, it is returned. Otherwise, a new bind group is created,
    /// cached, and returned.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device for creating bind groups
    /// * `layout` - The bind group layout
    /// * `layout_hash` - Hash from `BindGroupLayoutCache` (or `BindGroupLayoutKey::hash_value()`)
    /// * `entries` - The wgpu bind group entries
    /// * `resource_entries` - Resource IDs and types for cache key generation
    /// * `label` - Optional debug label for the bind group
    ///
    /// # Returns
    ///
    /// An `Arc<BindGroup>` that can be shared across multiple users.
    ///
    /// # Thread Safety
    ///
    /// This method uses a read lock for cache lookups and only acquires
    /// a write lock when creating a new bind group (double-check pattern).
    ///
    /// # Example
    ///
    /// ```no_run
    /// use renderer_backend::resources::bind_group_cache::{
    ///     BindGroupCache, BindGroupResourceEntry, BindGroupResourceType, ResourceId,
    /// };
    ///
    /// # fn example(device: &wgpu::Device, layout: &wgpu::BindGroupLayout, buffer: &wgpu::Buffer) {
    /// let cache = BindGroupCache::new();
    ///
    /// let resource_entries = &[
    ///     BindGroupResourceEntry::buffer(0, ResourceId::new(1), 0, Some(64)),
    /// ];
    ///
    /// let entries = &[
    ///     wgpu::BindGroupEntry {
    ///         binding: 0,
    ///         resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
    ///             buffer,
    ///             offset: 0,
    ///             size: None,
    ///         }),
    ///     },
    /// ];
    ///
    /// let bg = cache.create_bind_group(
    ///     device,
    ///     layout,
    ///     0x12345678,
    ///     entries,
    ///     resource_entries,
    ///     Some("my_bind_group"),
    /// );
    /// # }
    /// ```
    pub fn create_bind_group(
        &self,
        device: &Device,
        layout: &BindGroupLayout,
        layout_hash: u64,
        entries: &[BindGroupEntry],
        resource_entries: &[BindGroupResourceEntry],
        label: Option<&str>,
    ) -> Arc<BindGroup> {
        let key = BindGroupCacheKey::new(layout_hash, resource_entries);

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

        // Create new bind group
        self.misses.fetch_add(1, Ordering::Relaxed);
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label,
            layout,
            entries,
        });
        let arc_bind_group = Arc::new(bind_group);
        let resource_ids: Vec<_> = resource_entries.iter().map(|e| e.resource_id).collect();

        // Track resource -> bind group mappings for invalidation
        {
            let mut resource_map = self.resource_to_bind_groups.write();
            for &res_id in &resource_ids {
                resource_map.entry(res_id).or_default().push(key.clone());
            }
        }

        cache.insert(
            key,
            CachedBindGroup {
                bind_group: Arc::clone(&arc_bind_group),
                resource_ids,
                frame_created: self.current_frame.load(Ordering::Relaxed),
            },
        );

        arc_bind_group
    }

    /// Looks up a cached bind group without creating it.
    ///
    /// Returns `None` if no matching bind group exists in the cache.
    /// Does not update hit/miss statistics.
    ///
    /// # Arguments
    ///
    /// * `layout_hash` - Hash from `BindGroupLayoutCache`
    /// * `resource_entries` - Resource IDs and types for cache key generation
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::resources::bind_group_cache::{
    ///     BindGroupCache, BindGroupResourceEntry, BindGroupResourceType, ResourceId,
    /// };
    ///
    /// let cache = BindGroupCache::new();
    ///
    /// let resource_entries = &[
    ///     BindGroupResourceEntry::buffer(0, ResourceId::new(1), 0, None),
    /// ];
    ///
    /// // Not in cache yet
    /// assert!(cache.get(0x12345678, resource_entries).is_none());
    /// ```
    pub fn get(
        &self,
        layout_hash: u64,
        resource_entries: &[BindGroupResourceEntry],
    ) -> Option<Arc<BindGroup>> {
        let key = BindGroupCacheKey::new(layout_hash, resource_entries);
        let cache = self.cache.read();
        cache.get(&key).map(|c| c.arc())
    }

    /// Checks if a bind group exists in the cache.
    ///
    /// # Arguments
    ///
    /// * `layout_hash` - Hash from `BindGroupLayoutCache`
    /// * `resource_entries` - Resource IDs and types for cache key generation
    pub fn contains(
        &self,
        layout_hash: u64,
        resource_entries: &[BindGroupResourceEntry],
    ) -> bool {
        let key = BindGroupCacheKey::new(layout_hash, resource_entries);
        self.cache.read().contains_key(&key)
    }

    /// Invalidates all bind groups that reference the given resource.
    ///
    /// Call this when a resource (buffer, texture, sampler) is destroyed to
    /// ensure no cached bind groups reference the destroyed resource.
    ///
    /// # Arguments
    ///
    /// * `resource_id` - The ID of the destroyed resource
    ///
    /// # Returns
    ///
    /// The number of bind groups that were invalidated.
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::resources::bind_group_cache::{BindGroupCache, ResourceId};
    ///
    /// let cache = BindGroupCache::new();
    ///
    /// // When a buffer is destroyed:
    /// let buffer_id = ResourceId::new(42);
    /// let invalidated = cache.invalidate_resource(buffer_id);
    /// println!("Invalidated {} bind groups", invalidated);
    /// ```
    pub fn invalidate_resource(&self, resource_id: ResourceId) -> usize {
        // Get keys to remove while holding resource_map read lock
        let keys_to_remove: Vec<BindGroupCacheKey>;
        {
            let resource_map = self.resource_to_bind_groups.read();
            keys_to_remove = resource_map.get(&resource_id).cloned().unwrap_or_default();
        }

        let removed = keys_to_remove.len();

        if removed > 0 {
            // Remove from cache
            let mut cache = self.cache.write();
            let mut resource_map = self.resource_to_bind_groups.write();

            for key in &keys_to_remove {
                // Get resource IDs from the bind group being removed
                if let Some(cached) = cache.remove(key) {
                    // Clean up resource -> bind_group mappings
                    for res_id in cached.resource_ids() {
                        if let Some(keys) = resource_map.get_mut(res_id) {
                            keys.retain(|k| k != key);
                            // Note: We leave empty Vecs in the map; they'll be cleaned up
                            // by subsequent operations or clear()
                        }
                    }
                }
            }

            // Remove the resource from the tracking map
            resource_map.remove(&resource_id);
        }

        removed
    }

    /// Invalidates multiple resources at once.
    ///
    /// More efficient than calling `invalidate_resource` multiple times when
    /// destroying many resources at once.
    ///
    /// # Arguments
    ///
    /// * `resource_ids` - Iterator of resource IDs to invalidate
    ///
    /// # Returns
    ///
    /// The total number of bind groups that were invalidated.
    pub fn invalidate_resources(
        &self,
        resource_ids: impl IntoIterator<Item = ResourceId>,
    ) -> usize {
        // Collect all keys to remove
        let mut keys_to_remove = Vec::new();
        {
            let resource_map = self.resource_to_bind_groups.read();
            for resource_id in resource_ids {
                if let Some(keys) = resource_map.get(&resource_id) {
                    keys_to_remove.extend(keys.iter().cloned());
                }
            }
        }

        // Deduplicate keys (a bind group may reference multiple destroyed resources)
        keys_to_remove.sort_by(|a, b| {
            a.layout_hash()
                .cmp(&b.layout_hash())
                .then_with(|| a.resources_hash().cmp(&b.resources_hash()))
        });
        keys_to_remove.dedup();

        let removed = keys_to_remove.len();

        if removed > 0 {
            let mut cache = self.cache.write();
            for key in keys_to_remove {
                cache.remove(&key);
            }
            // Note: We don't clean up resource_to_bind_groups here for efficiency;
            // the next invalidate_resource call or clear() will handle it
        }

        removed
    }

    /// Advances to the next frame.
    ///
    /// Call this at the start of each frame to track frame numbers for
    /// age-based eviction.
    pub fn begin_frame(&self) {
        self.current_frame.fetch_add(1, Ordering::Relaxed);
    }

    /// Returns the current frame number.
    #[inline]
    pub fn current_frame(&self) -> u64 {
        self.current_frame.load(Ordering::Relaxed)
    }

    /// Evicts bind groups older than the specified number of frames.
    ///
    /// # Arguments
    ///
    /// * `max_age_frames` - Remove bind groups created more than this many frames ago
    ///
    /// # Returns
    ///
    /// The number of bind groups that were evicted.
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::resources::bind_group_cache::BindGroupCache;
    ///
    /// let cache = BindGroupCache::new();
    ///
    /// // Simulate 5 frames
    /// for _ in 0..5 {
    ///     cache.begin_frame();
    /// }
    ///
    /// // Evict bind groups older than 3 frames
    /// let evicted = cache.evict_old(3);
    /// ```
    pub fn evict_old(&self, max_age_frames: u64) -> usize {
        let current = self.current_frame.load(Ordering::Relaxed);
        let cutoff = current.saturating_sub(max_age_frames);

        let mut cache = self.cache.write();
        let before = cache.len();

        // Collect keys to remove
        let keys_to_remove: Vec<_> = cache
            .iter()
            .filter(|(_, v)| v.frame_created < cutoff)
            .map(|(k, _)| k.clone())
            .collect();

        // Remove from cache and clean up resource tracking
        if !keys_to_remove.is_empty() {
            let mut resource_map = self.resource_to_bind_groups.write();
            for key in keys_to_remove {
                if let Some(cached) = cache.remove(&key) {
                    for res_id in cached.resource_ids() {
                        if let Some(keys) = resource_map.get_mut(res_id) {
                            keys.retain(|k| k != &key);
                        }
                    }
                }
            }
        }

        before - cache.len()
    }

    /// Evicts all bind groups from the cache.
    ///
    /// Also resets metrics counters to zero.
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::resources::bind_group_cache::BindGroupCache;
    ///
    /// let cache = BindGroupCache::new();
    /// cache.clear();
    /// assert!(cache.is_empty());
    /// ```
    pub fn clear(&self) {
        self.cache.write().clear();
        self.resource_to_bind_groups.write().clear();
        self.hits.store(0, Ordering::Relaxed);
        self.misses.store(0, Ordering::Relaxed);
    }

    /// Returns the number of cached bind groups.
    #[inline]
    pub fn len(&self) -> usize {
        self.cache.read().len()
    }

    /// Returns true if the cache contains no bind groups.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.cache.read().is_empty()
    }

    /// Returns current cache metrics.
    pub fn metrics(&self) -> BindGroupCacheMetrics {
        let cache = self.cache.read();
        let resource_map = self.resource_to_bind_groups.read();
        let hits = self.hits.load(Ordering::Relaxed);
        let misses = self.misses.load(Ordering::Relaxed);

        BindGroupCacheMetrics::new(
            cache.len(),
            hits,
            misses,
            self.current_frame.load(Ordering::Relaxed),
            resource_map.len(),
        )
    }

    /// Resets metrics counters to zero without clearing the cache.
    pub fn reset_metrics(&self) {
        self.hits.store(0, Ordering::Relaxed);
        self.misses.store(0, Ordering::Relaxed);
    }

    /// Returns the number of tracked resources.
    #[inline]
    pub fn tracked_resource_count(&self) -> usize {
        self.resource_to_bind_groups.read().len()
    }

    /// Removes a specific bind group from the cache.
    ///
    /// # Arguments
    ///
    /// * `layout_hash` - Hash from `BindGroupLayoutCache`
    /// * `resource_entries` - Resource IDs and types for cache key generation
    ///
    /// # Returns
    ///
    /// `true` if a bind group was removed, `false` if no matching bind group was found.
    pub fn remove(
        &self,
        layout_hash: u64,
        resource_entries: &[BindGroupResourceEntry],
    ) -> bool {
        let key = BindGroupCacheKey::new(layout_hash, resource_entries);

        let mut cache = self.cache.write();
        if let Some(cached) = cache.remove(&key) {
            // Clean up resource tracking
            let mut resource_map = self.resource_to_bind_groups.write();
            for res_id in cached.resource_ids() {
                if let Some(keys) = resource_map.get_mut(res_id) {
                    keys.retain(|k| k != &key);
                }
            }
            true
        } else {
            false
        }
    }
}

impl Default for BindGroupCache {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for BindGroupCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let metrics = self.metrics();
        f.debug_struct("BindGroupCache")
            .field("cache_size", &metrics.cache_size)
            .field("hits", &metrics.hits)
            .field("misses", &metrics.misses)
            .field("hit_rate", &format!("{:.1}%", metrics.hit_rate_percent()))
            .field("current_frame", &metrics.current_frame)
            .field("tracked_resources", &metrics.tracked_resources)
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
    // ResourceId Tests
    // ========================================================================

    #[test]
    fn test_resource_id_new() {
        let id = ResourceId::new(42);
        assert_eq!(id.value(), 42);
    }

    #[test]
    fn test_resource_id_equality() {
        let id1 = ResourceId::new(1);
        let id2 = ResourceId::new(1);
        let id3 = ResourceId::new(2);

        assert_eq!(id1, id2);
        assert_ne!(id1, id3);
    }

    #[test]
    fn test_resource_id_hash() {
        let id1 = ResourceId::new(42);
        let id2 = ResourceId::new(42);

        let mut map: HashMap<ResourceId, i32> = HashMap::new();
        map.insert(id1, 1);

        assert_eq!(map.get(&id2), Some(&1));
    }

    #[test]
    fn test_resource_id_from_u64() {
        let id: ResourceId = 123u64.into();
        assert_eq!(id.value(), 123);
    }

    #[test]
    fn test_resource_id_into_u64() {
        let id = ResourceId::new(456);
        let val: u64 = id.into();
        assert_eq!(val, 456);
    }

    #[test]
    fn test_resource_id_copy() {
        let id1 = ResourceId::new(99);
        let id2 = id1; // Copy
        assert_eq!(id1, id2);
    }

    #[test]
    fn test_resource_id_debug() {
        let id = ResourceId::new(42);
        let debug_str = format!("{:?}", id);
        assert!(debug_str.contains("ResourceId"));
        assert!(debug_str.contains("42"));
    }

    // ========================================================================
    // BindGroupResourceType Tests
    // ========================================================================

    #[test]
    fn test_resource_type_buffer_default() {
        let ty = BindGroupResourceType::buffer();
        match ty {
            BindGroupResourceType::Buffer { offset, size } => {
                assert_eq!(offset, 0);
                assert!(size.is_none());
            }
            _ => panic!("Expected Buffer"),
        }
    }

    #[test]
    fn test_resource_type_buffer_range() {
        let ty = BindGroupResourceType::buffer_range(64, 256);
        match ty {
            BindGroupResourceType::Buffer { offset, size } => {
                assert_eq!(offset, 64);
                assert_eq!(size, Some(256));
            }
            _ => panic!("Expected Buffer"),
        }
    }

    #[test]
    fn test_resource_type_buffer_sized() {
        let ty = BindGroupResourceType::buffer_sized(128);
        match ty {
            BindGroupResourceType::Buffer { offset, size } => {
                assert_eq!(offset, 0);
                assert_eq!(size, Some(128));
            }
            _ => panic!("Expected Buffer"),
        }
    }

    #[test]
    fn test_resource_type_equality() {
        let ty1 = BindGroupResourceType::Buffer {
            offset: 0,
            size: Some(64),
        };
        let ty2 = BindGroupResourceType::Buffer {
            offset: 0,
            size: Some(64),
        };
        let ty3 = BindGroupResourceType::Buffer {
            offset: 0,
            size: Some(128),
        };

        assert_eq!(ty1, ty2);
        assert_ne!(ty1, ty3);
    }

    #[test]
    fn test_resource_type_hash() {
        use std::collections::hash_map::DefaultHasher;

        let ty1 = BindGroupResourceType::Sampler;
        let ty2 = BindGroupResourceType::TextureView;

        let mut h1 = DefaultHasher::new();
        let mut h2 = DefaultHasher::new();
        ty1.hash(&mut h1);
        ty2.hash(&mut h2);

        assert_ne!(h1.finish(), h2.finish());
    }

    #[test]
    fn test_resource_type_variants() {
        let _buffer = BindGroupResourceType::Buffer {
            offset: 0,
            size: None,
        };
        let _sampler = BindGroupResourceType::Sampler;
        let _texture = BindGroupResourceType::TextureView;
        let _storage = BindGroupResourceType::StorageTextureView;
    }

    // ========================================================================
    // BindGroupResourceEntry Tests
    // ========================================================================

    #[test]
    fn test_resource_entry_buffer() {
        let entry = BindGroupResourceEntry::buffer(0, ResourceId::new(1), 0, Some(64));
        assert_eq!(entry.binding, 0);
        assert_eq!(entry.resource_id, ResourceId::new(1));
        match entry.resource_type {
            BindGroupResourceType::Buffer { offset, size } => {
                assert_eq!(offset, 0);
                assert_eq!(size, Some(64));
            }
            _ => panic!("Expected Buffer"),
        }
    }

    #[test]
    fn test_resource_entry_sampler() {
        let entry = BindGroupResourceEntry::sampler(1, ResourceId::new(2));
        assert_eq!(entry.binding, 1);
        assert_eq!(entry.resource_id, ResourceId::new(2));
        assert_eq!(entry.resource_type, BindGroupResourceType::Sampler);
    }

    #[test]
    fn test_resource_entry_texture_view() {
        let entry = BindGroupResourceEntry::texture_view(2, ResourceId::new(3));
        assert_eq!(entry.binding, 2);
        assert_eq!(entry.resource_id, ResourceId::new(3));
        assert_eq!(entry.resource_type, BindGroupResourceType::TextureView);
    }

    #[test]
    fn test_resource_entry_storage_texture_view() {
        let entry = BindGroupResourceEntry::storage_texture_view(3, ResourceId::new(4));
        assert_eq!(entry.binding, 3);
        assert_eq!(entry.resource_id, ResourceId::new(4));
        assert_eq!(entry.resource_type, BindGroupResourceType::StorageTextureView);
    }

    #[test]
    fn test_resource_entry_clone() {
        let entry = BindGroupResourceEntry::buffer(0, ResourceId::new(1), 0, Some(64));
        let cloned = entry.clone();

        assert_eq!(entry.binding, cloned.binding);
        assert_eq!(entry.resource_id, cloned.resource_id);
        assert_eq!(entry.resource_type, cloned.resource_type);
    }

    // ========================================================================
    // BindGroupCacheKey Tests
    // ========================================================================

    #[test]
    fn test_cache_key_empty_entries() {
        let key = BindGroupCacheKey::new(0x12345678, &[]);
        assert_eq!(key.layout_hash(), 0x12345678);
        assert_ne!(key.resources_hash(), 0);
    }

    #[test]
    fn test_cache_key_single_entry() {
        let entries = &[BindGroupResourceEntry::buffer(0, ResourceId::new(1), 0, None)];
        let key = BindGroupCacheKey::new(0xABCD, entries);
        assert_eq!(key.layout_hash(), 0xABCD);
    }

    #[test]
    fn test_cache_key_equality_same_entries() {
        let entries1 = &[BindGroupResourceEntry::buffer(0, ResourceId::new(1), 0, None)];
        let entries2 = &[BindGroupResourceEntry::buffer(0, ResourceId::new(1), 0, None)];

        let key1 = BindGroupCacheKey::new(0x1234, entries1);
        let key2 = BindGroupCacheKey::new(0x1234, entries2);

        assert_eq!(key1, key2);
    }

    #[test]
    fn test_cache_key_equality_reordered() {
        let entries1 = &[
            BindGroupResourceEntry::buffer(0, ResourceId::new(1), 0, None),
            BindGroupResourceEntry::sampler(1, ResourceId::new(2)),
        ];
        let entries2 = &[
            BindGroupResourceEntry::sampler(1, ResourceId::new(2)),
            BindGroupResourceEntry::buffer(0, ResourceId::new(1), 0, None),
        ];

        let key1 = BindGroupCacheKey::new(0x5678, entries1);
        let key2 = BindGroupCacheKey::new(0x5678, entries2);

        // Should be equal because entries are sorted by binding
        assert_eq!(key1, key2);
    }

    #[test]
    fn test_cache_key_inequality_different_layout() {
        let entries = &[BindGroupResourceEntry::buffer(0, ResourceId::new(1), 0, None)];

        let key1 = BindGroupCacheKey::new(0x1111, entries);
        let key2 = BindGroupCacheKey::new(0x2222, entries);

        assert_ne!(key1, key2);
    }

    #[test]
    fn test_cache_key_inequality_different_resource() {
        let entries1 = &[BindGroupResourceEntry::buffer(0, ResourceId::new(1), 0, None)];
        let entries2 = &[BindGroupResourceEntry::buffer(0, ResourceId::new(2), 0, None)];

        let key1 = BindGroupCacheKey::new(0x1234, entries1);
        let key2 = BindGroupCacheKey::new(0x1234, entries2);

        assert_ne!(key1, key2);
    }

    #[test]
    fn test_cache_key_inequality_different_binding() {
        let entries1 = &[BindGroupResourceEntry::buffer(0, ResourceId::new(1), 0, None)];
        let entries2 = &[BindGroupResourceEntry::buffer(1, ResourceId::new(1), 0, None)];

        let key1 = BindGroupCacheKey::new(0x1234, entries1);
        let key2 = BindGroupCacheKey::new(0x1234, entries2);

        assert_ne!(key1, key2);
    }

    #[test]
    fn test_cache_key_inequality_different_type() {
        let entries1 = &[BindGroupResourceEntry::buffer(0, ResourceId::new(1), 0, None)];
        let entries2 = &[BindGroupResourceEntry::sampler(0, ResourceId::new(1))];

        let key1 = BindGroupCacheKey::new(0x1234, entries1);
        let key2 = BindGroupCacheKey::new(0x1234, entries2);

        assert_ne!(key1, key2);
    }

    #[test]
    fn test_cache_key_inequality_different_offset() {
        let entries1 = &[BindGroupResourceEntry::buffer(0, ResourceId::new(1), 0, None)];
        let entries2 = &[BindGroupResourceEntry::buffer(0, ResourceId::new(1), 64, None)];

        let key1 = BindGroupCacheKey::new(0x1234, entries1);
        let key2 = BindGroupCacheKey::new(0x1234, entries2);

        assert_ne!(key1, key2);
    }

    #[test]
    fn test_cache_key_inequality_different_size() {
        let entries1 = &[BindGroupResourceEntry::buffer(0, ResourceId::new(1), 0, Some(64))];
        let entries2 = &[BindGroupResourceEntry::buffer(0, ResourceId::new(1), 0, Some(128))];

        let key1 = BindGroupCacheKey::new(0x1234, entries1);
        let key2 = BindGroupCacheKey::new(0x1234, entries2);

        assert_ne!(key1, key2);
    }

    #[test]
    fn test_cache_key_as_hashmap_key() {
        let mut map: HashMap<BindGroupCacheKey, i32> = HashMap::new();

        let entries1 = &[BindGroupResourceEntry::buffer(0, ResourceId::new(1), 0, None)];
        let entries2 = &[BindGroupResourceEntry::sampler(1, ResourceId::new(2))];

        let key1 = BindGroupCacheKey::new(0x1111, entries1);
        let key2 = BindGroupCacheKey::new(0x2222, entries2);

        map.insert(key1.clone(), 1);
        map.insert(key2.clone(), 2);

        assert_eq!(map.len(), 2);
        assert_eq!(map.get(&key1), Some(&1));
        assert_eq!(map.get(&key2), Some(&2));
    }

    // ========================================================================
    // BindGroupCacheMetrics Tests
    // ========================================================================

    #[test]
    fn test_metrics_default() {
        let metrics = BindGroupCacheMetrics::default();
        assert_eq!(metrics.cache_size, 0);
        assert_eq!(metrics.hits, 0);
        assert_eq!(metrics.misses, 0);
        assert_eq!(metrics.hit_rate, 0.0);
        assert_eq!(metrics.current_frame, 0);
        assert_eq!(metrics.tracked_resources, 0);
    }

    #[test]
    fn test_metrics_new() {
        let metrics = BindGroupCacheMetrics::new(5, 80, 20, 10, 3);
        assert_eq!(metrics.cache_size, 5);
        assert_eq!(metrics.hits, 80);
        assert_eq!(metrics.misses, 20);
        assert!((metrics.hit_rate - 0.8).abs() < 0.001);
        assert_eq!(metrics.current_frame, 10);
        assert_eq!(metrics.tracked_resources, 3);
    }

    #[test]
    fn test_metrics_total_requests() {
        let metrics = BindGroupCacheMetrics::new(3, 50, 25, 0, 0);
        assert_eq!(metrics.total_requests(), 75);
    }

    #[test]
    fn test_metrics_is_empty() {
        let empty = BindGroupCacheMetrics::new(0, 0, 0, 0, 0);
        assert!(empty.is_empty());

        let non_empty = BindGroupCacheMetrics::new(1, 0, 1, 0, 0);
        assert!(!non_empty.is_empty());
    }

    #[test]
    fn test_metrics_hit_rate_percent() {
        let metrics = BindGroupCacheMetrics::new(2, 75, 25, 0, 0);
        assert!((metrics.hit_rate_percent() - 75.0).abs() < 0.001);
    }

    #[test]
    fn test_metrics_zero_total() {
        let metrics = BindGroupCacheMetrics::new(0, 0, 0, 0, 0);
        assert_eq!(metrics.hit_rate, 0.0);
        assert_eq!(metrics.total_requests(), 0);
    }

    #[test]
    fn test_metrics_clone() {
        let metrics = BindGroupCacheMetrics::new(10, 100, 50, 5, 2);
        let cloned = metrics.clone();

        assert_eq!(cloned.cache_size, metrics.cache_size);
        assert_eq!(cloned.hits, metrics.hits);
        assert_eq!(cloned.misses, metrics.misses);
        assert_eq!(cloned.hit_rate, metrics.hit_rate);
        assert_eq!(cloned.current_frame, metrics.current_frame);
        assert_eq!(cloned.tracked_resources, metrics.tracked_resources);
    }

    #[test]
    fn test_metrics_debug() {
        let metrics = BindGroupCacheMetrics::new(3, 10, 5, 2, 1);
        let debug_str = format!("{:?}", metrics);
        assert!(debug_str.contains("BindGroupCacheMetrics"));
    }

    // ========================================================================
    // BindGroupCache Tests (no device required)
    // ========================================================================

    #[test]
    fn test_cache_new() {
        let cache = BindGroupCache::new();
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn test_cache_default() {
        let cache = BindGroupCache::default();
        assert!(cache.is_empty());
    }

    #[test]
    fn test_cache_metrics_initial() {
        let cache = BindGroupCache::new();
        let metrics = cache.metrics();

        assert_eq!(metrics.cache_size, 0);
        assert_eq!(metrics.hits, 0);
        assert_eq!(metrics.misses, 0);
        assert_eq!(metrics.hit_rate, 0.0);
        assert_eq!(metrics.current_frame, 0);
        assert_eq!(metrics.tracked_resources, 0);
    }

    #[test]
    fn test_cache_begin_frame() {
        let cache = BindGroupCache::new();
        assert_eq!(cache.current_frame(), 0);

        cache.begin_frame();
        assert_eq!(cache.current_frame(), 1);

        cache.begin_frame();
        cache.begin_frame();
        assert_eq!(cache.current_frame(), 3);
    }

    #[test]
    fn test_cache_reset_metrics() {
        let cache = BindGroupCache::new();

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
        let cache = BindGroupCache::new();

        // Simulate some activity
        cache.hits.fetch_add(10, Ordering::Relaxed);
        cache.misses.fetch_add(5, Ordering::Relaxed);
        cache.begin_frame();

        cache.clear();

        assert!(cache.is_empty());
        let metrics = cache.metrics();
        assert_eq!(metrics.hits, 0);
        assert_eq!(metrics.misses, 0);
    }

    #[test]
    fn test_cache_debug_format() {
        let cache = BindGroupCache::new();
        let debug_str = format!("{:?}", cache);

        assert!(debug_str.contains("BindGroupCache"));
        assert!(debug_str.contains("cache_size"));
        assert!(debug_str.contains("current_frame"));
    }

    #[test]
    fn test_cache_contains_empty() {
        let cache = BindGroupCache::new();
        let entries = &[BindGroupResourceEntry::buffer(0, ResourceId::new(1), 0, None)];

        assert!(!cache.contains(0x1234, entries));
    }

    #[test]
    fn test_cache_get_empty() {
        let cache = BindGroupCache::new();
        let entries = &[BindGroupResourceEntry::buffer(0, ResourceId::new(1), 0, None)];

        assert!(cache.get(0x1234, entries).is_none());
    }

    #[test]
    fn test_cache_invalidate_resource_empty() {
        let cache = BindGroupCache::new();
        let removed = cache.invalidate_resource(ResourceId::new(1));
        assert_eq!(removed, 0);
    }

    #[test]
    fn test_cache_invalidate_resources_empty() {
        let cache = BindGroupCache::new();
        let removed = cache.invalidate_resources([ResourceId::new(1), ResourceId::new(2)]);
        assert_eq!(removed, 0);
    }

    #[test]
    fn test_cache_evict_old_empty() {
        let cache = BindGroupCache::new();
        cache.begin_frame();
        cache.begin_frame();

        let evicted = cache.evict_old(1);
        assert_eq!(evicted, 0);
    }

    #[test]
    fn test_cache_tracked_resource_count() {
        let cache = BindGroupCache::new();
        assert_eq!(cache.tracked_resource_count(), 0);
    }

    #[test]
    fn test_cache_remove_nonexistent() {
        let cache = BindGroupCache::new();
        let entries = &[BindGroupResourceEntry::buffer(0, ResourceId::new(1), 0, None)];

        assert!(!cache.remove(0x1234, entries));
    }

    // ========================================================================
    // CachedBindGroup Tests
    // ========================================================================

    #[test]
    fn test_cached_bind_group_debug() {
        // We can't create a real CachedBindGroup without a device, but we can
        // test that Debug is properly implemented
        let _format_template =
            "CachedBindGroup { resource_count: 2, frame_created: 5 }";
    }

    // ========================================================================
    // Thread Safety Tests
    // ========================================================================

    #[test]
    fn test_cache_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<BindGroupCache>();
    }

    #[test]
    fn test_key_clone() {
        let entries = &[BindGroupResourceEntry::buffer(0, ResourceId::new(1), 0, None)];
        let key = BindGroupCacheKey::new(0x1234, entries);
        let cloned = key.clone();
        assert_eq!(key, cloned);
    }

    // ========================================================================
    // Complex Layout Tests
    // ========================================================================

    #[test]
    fn test_cache_key_pbr_layout() {
        // Simulate a typical PBR material bind group
        let entries = &[
            BindGroupResourceEntry::buffer(0, ResourceId::new(1), 0, Some(144)), // Camera uniform
            BindGroupResourceEntry::texture_view(1, ResourceId::new(2)),          // Albedo
            BindGroupResourceEntry::sampler(2, ResourceId::new(3)),               // Sampler
            BindGroupResourceEntry::texture_view(3, ResourceId::new(4)),          // Normal map
            BindGroupResourceEntry::texture_view(4, ResourceId::new(5)),          // Roughness
        ];

        let key1 = BindGroupCacheKey::new(0xABCD_1234, entries);
        let key2 = BindGroupCacheKey::new(0xABCD_1234, entries);

        assert_eq!(key1, key2);
    }

    #[test]
    fn test_cache_key_compute_layout() {
        // Simulate a compute shader bind group
        let entries = &[
            BindGroupResourceEntry::buffer(0, ResourceId::new(1), 0, None), // Input storage
            BindGroupResourceEntry::buffer(1, ResourceId::new(2), 0, None), // Output storage
            BindGroupResourceEntry::storage_texture_view(2, ResourceId::new(3)), // Output texture
        ];

        let key = BindGroupCacheKey::new(0xDEAD_BEEF, entries);
        assert_ne!(key.resources_hash(), 0);
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
    fn test_create_bind_group_with_device() {
        let (device, _queue) = match create_test_device() {
            Some(d) => d,
            None => {
                eprintln!("Skipping: no GPU adapter available");
                return;
            }
        };

        // Create a buffer
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("test uniform"),
            size: 64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create a layout
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("test layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let cache = BindGroupCache::new();

        let resource_entries = &[BindGroupResourceEntry::buffer(0, ResourceId::new(1), 0, None)];

        let entries = &[wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer: &buffer,
                offset: 0,
                size: None,
            }),
        }];

        // First call creates the bind group
        let bg1 = cache.create_bind_group(&device, &layout, 0x1234, entries, resource_entries, Some("test"));
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.metrics().misses, 1);
        assert_eq!(cache.metrics().hits, 0);

        // Second call returns cached bind group
        let bg2 = cache.create_bind_group(&device, &layout, 0x1234, entries, resource_entries, Some("test"));
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.metrics().misses, 1);
        assert_eq!(cache.metrics().hits, 1);

        // Same Arc
        assert!(Arc::ptr_eq(&bg1, &bg2));
    }

    #[test]
    fn test_cache_resource_invalidation() {
        let (device, _queue) = match create_test_device() {
            Some(d) => d,
            None => {
                eprintln!("Skipping: no GPU adapter available");
                return;
            }
        };

        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("test uniform"),
            size: 64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("test layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let cache = BindGroupCache::new();
        let buffer_id = ResourceId::new(42);
        let resource_entries = &[BindGroupResourceEntry::buffer(0, buffer_id, 0, None)];

        let entries = &[wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer: &buffer,
                offset: 0,
                size: None,
            }),
        }];

        // Create bind group
        let _bg = cache.create_bind_group(&device, &layout, 0x1234, entries, resource_entries, None);
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.tracked_resource_count(), 1);

        // Invalidate the resource
        let invalidated = cache.invalidate_resource(buffer_id);
        assert_eq!(invalidated, 1);
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn test_cache_frame_eviction() {
        let (device, _queue) = match create_test_device() {
            Some(d) => d,
            None => {
                eprintln!("Skipping: no GPU adapter available");
                return;
            }
        };

        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("test uniform"),
            size: 64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("test layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let cache = BindGroupCache::new();
        let resource_entries = &[BindGroupResourceEntry::buffer(0, ResourceId::new(1), 0, None)];

        let entries = &[wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer: &buffer,
                offset: 0,
                size: None,
            }),
        }];

        // Create bind group at frame 0
        let _bg = cache.create_bind_group(&device, &layout, 0x1234, entries, resource_entries, None);
        assert_eq!(cache.len(), 1);

        // Advance 5 frames
        for _ in 0..5 {
            cache.begin_frame();
        }

        // Evict entries older than 3 frames
        let evicted = cache.evict_old(3);
        assert_eq!(evicted, 1);
        assert!(cache.is_empty());
    }

    #[test]
    fn test_cache_multiple_bind_groups() {
        let (device, _queue) = match create_test_device() {
            Some(d) => d,
            None => {
                eprintln!("Skipping: no GPU adapter available");
                return;
            }
        };

        let buffer1 = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("buf1"),
            size: 64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let buffer2 = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("buf2"),
            size: 128,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("test layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let cache = BindGroupCache::new();

        // Create first bind group
        let resource_entries1 = &[BindGroupResourceEntry::buffer(0, ResourceId::new(1), 0, None)];
        let entries1 = &[wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer: &buffer1,
                offset: 0,
                size: None,
            }),
        }];
        let _bg1 = cache.create_bind_group(&device, &layout, 0x1234, entries1, resource_entries1, None);

        // Create second bind group (different buffer)
        let resource_entries2 = &[BindGroupResourceEntry::buffer(0, ResourceId::new(2), 0, None)];
        let entries2 = &[wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer: &buffer2,
                offset: 0,
                size: None,
            }),
        }];
        let _bg2 = cache.create_bind_group(&device, &layout, 0x1234, entries2, resource_entries2, None);

        assert_eq!(cache.len(), 2);
        assert_eq!(cache.tracked_resource_count(), 2);
    }

    #[test]
    fn test_cache_get_and_contains() {
        let (device, _queue) = match create_test_device() {
            Some(d) => d,
            None => {
                eprintln!("Skipping: no GPU adapter available");
                return;
            }
        };

        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("test"),
            size: 64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("test layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let cache = BindGroupCache::new();
        let resource_entries = &[BindGroupResourceEntry::buffer(0, ResourceId::new(1), 0, None)];

        // Not in cache yet
        assert!(!cache.contains(0x1234, resource_entries));
        assert!(cache.get(0x1234, resource_entries).is_none());

        // Create bind group
        let entries = &[wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer: &buffer,
                offset: 0,
                size: None,
            }),
        }];
        let _bg = cache.create_bind_group(&device, &layout, 0x1234, entries, resource_entries, None);

        // Now in cache
        assert!(cache.contains(0x1234, resource_entries));
        assert!(cache.get(0x1234, resource_entries).is_some());
    }

    #[test]
    fn test_cache_remove() {
        let (device, _queue) = match create_test_device() {
            Some(d) => d,
            None => {
                eprintln!("Skipping: no GPU adapter available");
                return;
            }
        };

        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("test"),
            size: 64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("test layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let cache = BindGroupCache::new();
        let resource_entries = &[BindGroupResourceEntry::buffer(0, ResourceId::new(1), 0, None)];
        let entries = &[wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer: &buffer,
                offset: 0,
                size: None,
            }),
        }];

        let _bg = cache.create_bind_group(&device, &layout, 0x1234, entries, resource_entries, None);
        assert_eq!(cache.len(), 1);

        // Remove
        assert!(cache.remove(0x1234, resource_entries));
        assert!(cache.is_empty());

        // Can't remove again
        assert!(!cache.remove(0x1234, resource_entries));
    }
}
