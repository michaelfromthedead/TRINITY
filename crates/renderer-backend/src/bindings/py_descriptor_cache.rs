//! Python bindings for descriptor caching (T-WGPU-P7.6.7).
//!
//! This module provides Python-accessible types for caching GPU descriptors
//! to reduce allocation overhead. It implements an LRU (Least Recently Used)
//! eviction strategy when capacity is exceeded.
//!
//! # Feature Gate
//!
//! All types are gated behind the `pyo3` feature flag:
//!
//! ```toml
//! [features]
//! pyo3 = ["dep:pyo3"]
//! ```
//!
//! # Example (Python)
//!
//! ```python
//! from trinity_renderer.bindings import (
//!     PyDescriptorCache, PyCacheKey, PyCacheStats,
//!     PyBufferDescriptor, PyTextureDescriptor, PySamplerDescriptor
//! )
//!
//! # Create a descriptor cache with capacity 1000
//! cache = PyDescriptorCache(capacity=1000)
//!
//! # Get or create a buffer descriptor
//! buffer_desc = PyBufferDescriptor.uniform(256)
//! handle = cache.get_or_create_buffer(buffer_desc)
//!
//! # Second call returns cached handle
//! handle2 = cache.get_or_create_buffer(buffer_desc)
//! assert cache.hit_rate() > 0.0
//!
//! # Get cache statistics
//! stats = cache.stats()
//! print(f"Hit rate: {stats.hit_rate():.2%}")
//! print(f"Size: {stats.current_size} / {stats.capacity}")
//!
//! # Trim old entries
//! cache.trim(max_age_secs=60.0)
//!
//! # Clear all entries
//! cache.clear()
//! ```

use pyo3::prelude::*;
use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::time::Instant;

use super::py_buffer::PyBufferDescriptor;
use super::py_resource::{PyResourceHandle, PyResourcePool, PyResourceType};

// ============================================================================
// PyTextureDescriptor (minimal placeholder)
// ============================================================================

/// Texture descriptor for GPU texture creation.
///
/// This is a minimal implementation for cache key purposes.
/// A full implementation would include format, dimensions, mip levels, etc.
#[pyclass(name = "TextureDescriptor")]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct PyTextureDescriptor {
    width: u32,
    height: u32,
    depth: u32,
    format: u32,   // Texture format enum value
    mip_levels: u32,
    sample_count: u32,
    usage: u32,
    label: Option<String>,
}

#[pymethods]
impl PyTextureDescriptor {
    /// Create a new texture descriptor.
    #[new]
    #[pyo3(signature = (width, height, depth=1, format=0, mip_levels=1, sample_count=1, usage=0, label=None))]
    pub fn new(
        width: u32,
        height: u32,
        depth: u32,
        format: u32,
        mip_levels: u32,
        sample_count: u32,
        usage: u32,
        label: Option<String>,
    ) -> Self {
        Self {
            width,
            height,
            depth,
            format,
            mip_levels,
            sample_count,
            usage,
            label,
        }
    }

    /// Create a 2D texture descriptor.
    #[staticmethod]
    #[pyo3(signature = (width, height, format=0, usage=0))]
    pub fn texture_2d(width: u32, height: u32, format: u32, usage: u32) -> Self {
        Self {
            width,
            height,
            depth: 1,
            format,
            mip_levels: 1,
            sample_count: 1,
            usage,
            label: None,
        }
    }

    /// Create a 3D texture descriptor.
    #[staticmethod]
    #[pyo3(signature = (width, height, depth, format=0, usage=0))]
    pub fn texture_3d(width: u32, height: u32, depth: u32, format: u32, usage: u32) -> Self {
        Self {
            width,
            height,
            depth,
            format,
            mip_levels: 1,
            sample_count: 1,
            usage,
            label: None,
        }
    }

    /// Create a render target texture descriptor.
    #[staticmethod]
    #[pyo3(signature = (width, height, format=0, sample_count=1))]
    pub fn render_target(width: u32, height: u32, format: u32, sample_count: u32) -> Self {
        const RENDER_ATTACHMENT: u32 = 0x10; // wgpu TextureUsages::RENDER_ATTACHMENT
        Self {
            width,
            height,
            depth: 1,
            format,
            mip_levels: 1,
            sample_count,
            usage: RENDER_ATTACHMENT,
            label: None,
        }
    }

    // Getters
    #[getter]
    pub fn width(&self) -> u32 {
        self.width
    }

    #[getter]
    pub fn height(&self) -> u32 {
        self.height
    }

    #[getter]
    pub fn depth(&self) -> u32 {
        self.depth
    }

    #[getter]
    pub fn format(&self) -> u32 {
        self.format
    }

    #[getter]
    pub fn mip_levels(&self) -> u32 {
        self.mip_levels
    }

    #[getter]
    pub fn sample_count(&self) -> u32 {
        self.sample_count
    }

    #[getter]
    pub fn usage(&self) -> u32 {
        self.usage
    }

    #[getter]
    pub fn label(&self) -> Option<String> {
        self.label.clone()
    }

    /// Builder: set label
    pub fn with_label(&self, label: &str) -> Self {
        let mut new = self.clone();
        new.label = Some(label.to_string());
        new
    }

    /// Builder: set mip levels
    pub fn with_mip_levels(&self, mip_levels: u32) -> Self {
        let mut new = self.clone();
        new.mip_levels = mip_levels;
        new
    }

    /// Builder: set usage
    pub fn with_usage(&self, usage: u32) -> Self {
        let mut new = self.clone();
        new.usage = usage;
        new
    }

    pub fn __repr__(&self) -> String {
        let label_str = self
            .label
            .as_ref()
            .map(|l| format!(", label='{}'", l))
            .unwrap_or_default();
        format!(
            "TextureDescriptor({}x{}x{}, format={}, mips={}, samples={}{})",
            self.width, self.height, self.depth, self.format, self.mip_levels, self.sample_count, label_str
        )
    }

    pub fn __hash__(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.hash(&mut hasher);
        hasher.finish()
    }

    pub fn __eq__(&self, other: &Self) -> bool {
        self == other
    }
}

impl Default for PyTextureDescriptor {
    fn default() -> Self {
        Self::texture_2d(1, 1, 0, 0)
    }
}

// ============================================================================
// PySamplerDescriptor (minimal placeholder)
// ============================================================================

/// Sampler descriptor for GPU sampler creation.
///
/// This is a minimal implementation for cache key purposes.
#[pyclass(name = "SamplerDescriptor")]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct PySamplerDescriptor {
    address_mode_u: u8,
    address_mode_v: u8,
    address_mode_w: u8,
    mag_filter: u8,
    min_filter: u8,
    mipmap_filter: u8,
    lod_min_clamp: u32, // f32 bits for hashing
    lod_max_clamp: u32, // f32 bits for hashing
    compare: Option<u8>,
    anisotropy_clamp: u8,
    label: Option<String>,
}

/// Address mode constants
mod address_mode {
    pub const CLAMP_TO_EDGE: u8 = 0;
    pub const REPEAT: u8 = 1;
    pub const MIRROR_REPEAT: u8 = 2;
    pub const CLAMP_TO_BORDER: u8 = 3;
}

/// Filter mode constants
mod filter_mode {
    pub const NEAREST: u8 = 0;
    pub const LINEAR: u8 = 1;
}

#[pymethods]
impl PySamplerDescriptor {
    /// Create a new sampler descriptor with all parameters.
    #[new]
    #[pyo3(signature = (
        address_mode_u=0,
        address_mode_v=0,
        address_mode_w=0,
        mag_filter=1,
        min_filter=1,
        mipmap_filter=1,
        lod_min_clamp=0.0,
        lod_max_clamp=32.0,
        compare=None,
        anisotropy_clamp=1,
        label=None
    ))]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        address_mode_u: u8,
        address_mode_v: u8,
        address_mode_w: u8,
        mag_filter: u8,
        min_filter: u8,
        mipmap_filter: u8,
        lod_min_clamp: f32,
        lod_max_clamp: f32,
        compare: Option<u8>,
        anisotropy_clamp: u8,
        label: Option<String>,
    ) -> Self {
        Self {
            address_mode_u,
            address_mode_v,
            address_mode_w,
            mag_filter,
            min_filter,
            mipmap_filter,
            lod_min_clamp: lod_min_clamp.to_bits(),
            lod_max_clamp: lod_max_clamp.to_bits(),
            compare,
            anisotropy_clamp,
            label,
        }
    }

    /// Create a linear filtering sampler (most common).
    #[staticmethod]
    pub fn linear() -> Self {
        Self {
            address_mode_u: address_mode::CLAMP_TO_EDGE,
            address_mode_v: address_mode::CLAMP_TO_EDGE,
            address_mode_w: address_mode::CLAMP_TO_EDGE,
            mag_filter: filter_mode::LINEAR,
            min_filter: filter_mode::LINEAR,
            mipmap_filter: filter_mode::LINEAR,
            lod_min_clamp: 0.0_f32.to_bits(),
            lod_max_clamp: 32.0_f32.to_bits(),
            compare: None,
            anisotropy_clamp: 1,
            label: None,
        }
    }

    /// Create a nearest (point) filtering sampler.
    #[staticmethod]
    pub fn nearest() -> Self {
        Self {
            address_mode_u: address_mode::CLAMP_TO_EDGE,
            address_mode_v: address_mode::CLAMP_TO_EDGE,
            address_mode_w: address_mode::CLAMP_TO_EDGE,
            mag_filter: filter_mode::NEAREST,
            min_filter: filter_mode::NEAREST,
            mipmap_filter: filter_mode::NEAREST,
            lod_min_clamp: 0.0_f32.to_bits(),
            lod_max_clamp: 32.0_f32.to_bits(),
            compare: None,
            anisotropy_clamp: 1,
            label: None,
        }
    }

    /// Create a repeating sampler (for tiled textures).
    #[staticmethod]
    pub fn repeating() -> Self {
        Self {
            address_mode_u: address_mode::REPEAT,
            address_mode_v: address_mode::REPEAT,
            address_mode_w: address_mode::REPEAT,
            mag_filter: filter_mode::LINEAR,
            min_filter: filter_mode::LINEAR,
            mipmap_filter: filter_mode::LINEAR,
            lod_min_clamp: 0.0_f32.to_bits(),
            lod_max_clamp: 32.0_f32.to_bits(),
            compare: None,
            anisotropy_clamp: 1,
            label: None,
        }
    }

    /// Create a comparison sampler (for shadow maps).
    #[staticmethod]
    #[pyo3(signature = (compare_func=1))]
    pub fn comparison(compare_func: u8) -> Self {
        Self {
            address_mode_u: address_mode::CLAMP_TO_EDGE,
            address_mode_v: address_mode::CLAMP_TO_EDGE,
            address_mode_w: address_mode::CLAMP_TO_EDGE,
            mag_filter: filter_mode::LINEAR,
            min_filter: filter_mode::LINEAR,
            mipmap_filter: filter_mode::NEAREST,
            lod_min_clamp: 0.0_f32.to_bits(),
            lod_max_clamp: 32.0_f32.to_bits(),
            compare: Some(compare_func),
            anisotropy_clamp: 1,
            label: None,
        }
    }

    // Getters
    #[getter]
    pub fn address_mode_u(&self) -> u8 {
        self.address_mode_u
    }

    #[getter]
    pub fn address_mode_v(&self) -> u8 {
        self.address_mode_v
    }

    #[getter]
    pub fn address_mode_w(&self) -> u8 {
        self.address_mode_w
    }

    #[getter]
    pub fn mag_filter(&self) -> u8 {
        self.mag_filter
    }

    #[getter]
    pub fn min_filter(&self) -> u8 {
        self.min_filter
    }

    #[getter]
    pub fn mipmap_filter(&self) -> u8 {
        self.mipmap_filter
    }

    #[getter]
    pub fn lod_min_clamp(&self) -> f32 {
        f32::from_bits(self.lod_min_clamp)
    }

    #[getter]
    pub fn lod_max_clamp(&self) -> f32 {
        f32::from_bits(self.lod_max_clamp)
    }

    #[getter]
    pub fn compare(&self) -> Option<u8> {
        self.compare
    }

    #[getter]
    pub fn anisotropy_clamp(&self) -> u8 {
        self.anisotropy_clamp
    }

    #[getter]
    pub fn label(&self) -> Option<String> {
        self.label.clone()
    }

    /// Builder: set label
    pub fn with_label(&self, label: &str) -> Self {
        let mut new = self.clone();
        new.label = Some(label.to_string());
        new
    }

    /// Builder: set anisotropy
    pub fn with_anisotropy(&self, clamp: u8) -> Self {
        let mut new = self.clone();
        new.anisotropy_clamp = clamp;
        new
    }

    pub fn __repr__(&self) -> String {
        let filter = match (self.mag_filter, self.min_filter) {
            (0, 0) => "nearest",
            (1, 1) => "linear",
            _ => "mixed",
        };
        let label_str = self
            .label
            .as_ref()
            .map(|l| format!(", label='{}'", l))
            .unwrap_or_default();
        format!("SamplerDescriptor(filter={}{})", filter, label_str)
    }

    pub fn __hash__(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.hash(&mut hasher);
        hasher.finish()
    }

    pub fn __eq__(&self, other: &Self) -> bool {
        self == other
    }
}

impl Default for PySamplerDescriptor {
    fn default() -> Self {
        Self::linear()
    }
}

// ============================================================================
// PyCacheKey
// ============================================================================

/// Cache key for descriptor lookup.
///
/// A hashable key that uniquely identifies a descriptor configuration.
/// The key is computed from the descriptor's relevant fields.
#[pyclass(name = "CacheKey")]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct PyCacheKey {
    key_type: CacheKeyType,
    hash_value: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum CacheKeyType {
    Buffer,
    Texture,
    Sampler,
    Pipeline,
}

impl CacheKeyType {
    fn name(&self) -> &str {
        match self {
            Self::Buffer => "Buffer",
            Self::Texture => "Texture",
            Self::Sampler => "Sampler",
            Self::Pipeline => "Pipeline",
        }
    }
}

#[pymethods]
impl PyCacheKey {
    /// Create a cache key from a buffer descriptor.
    #[staticmethod]
    pub fn from_buffer(desc: &PyBufferDescriptor) -> Self {
        let mut hasher = DefaultHasher::new();
        desc.size().hash(&mut hasher);
        desc.usage().bits().hash(&mut hasher);
        desc.mapped_at_creation().hash(&mut hasher);
        // Note: label is intentionally excluded from hash
        Self {
            key_type: CacheKeyType::Buffer,
            hash_value: hasher.finish(),
        }
    }

    /// Create a cache key from a texture descriptor.
    #[staticmethod]
    pub fn from_texture(desc: &PyTextureDescriptor) -> Self {
        let mut hasher = DefaultHasher::new();
        desc.width.hash(&mut hasher);
        desc.height.hash(&mut hasher);
        desc.depth.hash(&mut hasher);
        desc.format.hash(&mut hasher);
        desc.mip_levels.hash(&mut hasher);
        desc.sample_count.hash(&mut hasher);
        desc.usage.hash(&mut hasher);
        // Note: label is intentionally excluded from hash
        Self {
            key_type: CacheKeyType::Texture,
            hash_value: hasher.finish(),
        }
    }

    /// Create a cache key from a sampler descriptor.
    #[staticmethod]
    pub fn from_sampler(desc: &PySamplerDescriptor) -> Self {
        let mut hasher = DefaultHasher::new();
        desc.address_mode_u.hash(&mut hasher);
        desc.address_mode_v.hash(&mut hasher);
        desc.address_mode_w.hash(&mut hasher);
        desc.mag_filter.hash(&mut hasher);
        desc.min_filter.hash(&mut hasher);
        desc.mipmap_filter.hash(&mut hasher);
        desc.lod_min_clamp.hash(&mut hasher);
        desc.lod_max_clamp.hash(&mut hasher);
        desc.compare.hash(&mut hasher);
        desc.anisotropy_clamp.hash(&mut hasher);
        // Note: label is intentionally excluded from hash
        Self {
            key_type: CacheKeyType::Sampler,
            hash_value: hasher.finish(),
        }
    }

    /// Returns the type of descriptor this key represents.
    pub fn key_type(&self) -> String {
        self.key_type.name().to_string()
    }

    /// Returns true if this key is for a buffer descriptor.
    pub fn is_buffer(&self) -> bool {
        matches!(self.key_type, CacheKeyType::Buffer)
    }

    /// Returns true if this key is for a texture descriptor.
    pub fn is_texture(&self) -> bool {
        matches!(self.key_type, CacheKeyType::Texture)
    }

    /// Returns true if this key is for a sampler descriptor.
    pub fn is_sampler(&self) -> bool {
        matches!(self.key_type, CacheKeyType::Sampler)
    }

    pub fn __repr__(&self) -> String {
        format!(
            "CacheKey({}, hash={:#018x})",
            self.key_type.name(),
            self.hash_value
        )
    }

    pub fn __hash__(&self) -> u64 {
        self.hash_value
    }

    pub fn __eq__(&self, other: &Self) -> bool {
        self.key_type == other.key_type && self.hash_value == other.hash_value
    }
}

// ============================================================================
// PyCachedDescriptor
// ============================================================================

/// Wrapper for cached descriptors with usage tracking.
///
/// Tracks hit count and last access time for LRU eviction.
#[pyclass(name = "CachedDescriptor")]
#[derive(Clone, Debug)]
pub struct PyCachedDescriptor {
    handle: PyResourceHandle,
    hit_count: u32,
    #[allow(dead_code)]
    created_at: Instant,
    last_used: Instant,
}

#[pymethods]
impl PyCachedDescriptor {
    /// Returns the resource handle for this cached descriptor.
    #[getter]
    pub fn handle(&self) -> PyResourceHandle {
        self.handle.clone()
    }

    /// Returns the number of times this descriptor has been accessed.
    #[getter]
    pub fn hit_count(&self) -> u32 {
        self.hit_count
    }

    /// Returns the time since last use in seconds.
    pub fn age_secs(&self) -> f64 {
        self.last_used.elapsed().as_secs_f64()
    }

    /// Returns the time since creation in seconds.
    pub fn lifetime_secs(&self) -> f64 {
        self.created_at.elapsed().as_secs_f64()
    }

    pub fn __repr__(&self) -> String {
        format!(
            "CachedDescriptor(handle={}, hits={}, age={:.2}s)",
            self.handle.__repr__(),
            self.hit_count,
            self.age_secs()
        )
    }
}

impl PyCachedDescriptor {
    fn new(handle: PyResourceHandle) -> Self {
        let now = Instant::now();
        Self {
            handle,
            hit_count: 0,
            created_at: now,
            last_used: now,
        }
    }

    fn touch(&mut self) {
        self.hit_count = self.hit_count.saturating_add(1);
        self.last_used = Instant::now();
    }
}

// ============================================================================
// PyCacheStats
// ============================================================================

/// Statistics for the descriptor cache.
#[pyclass(name = "CacheStats")]
#[derive(Clone, Debug, Default)]
pub struct PyCacheStats {
    /// Total cache hits.
    #[pyo3(get)]
    pub total_hits: u64,
    /// Total cache misses.
    #[pyo3(get)]
    pub total_misses: u64,
    /// Total number of evictions.
    #[pyo3(get)]
    pub evictions: u64,
    /// Current number of cached entries.
    #[pyo3(get)]
    pub current_size: usize,
    /// Maximum capacity of the cache.
    #[pyo3(get)]
    pub capacity: usize,
}

#[pymethods]
impl PyCacheStats {
    /// Create new cache statistics.
    #[new]
    #[pyo3(signature = (total_hits=0, total_misses=0, evictions=0, current_size=0, capacity=0))]
    pub fn new(
        total_hits: u64,
        total_misses: u64,
        evictions: u64,
        current_size: usize,
        capacity: usize,
    ) -> Self {
        Self {
            total_hits,
            total_misses,
            evictions,
            current_size,
            capacity,
        }
    }

    /// Returns the cache hit rate as a value between 0.0 and 1.0.
    pub fn hit_rate(&self) -> f64 {
        let total = self.total_hits + self.total_misses;
        if total == 0 {
            0.0
        } else {
            self.total_hits as f64 / total as f64
        }
    }

    /// Returns the total number of lookups (hits + misses).
    pub fn total_lookups(&self) -> u64 {
        self.total_hits + self.total_misses
    }

    /// Returns the fill ratio (current_size / capacity).
    pub fn fill_ratio(&self) -> f64 {
        if self.capacity == 0 {
            0.0
        } else {
            self.current_size as f64 / self.capacity as f64
        }
    }

    /// Returns true if the cache is full.
    pub fn is_full(&self) -> bool {
        self.current_size >= self.capacity
    }

    /// Returns true if the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.current_size == 0
    }

    pub fn __repr__(&self) -> String {
        format!(
            "CacheStats(hits={}, misses={}, evictions={}, size={}/{}, hit_rate={:.1}%)",
            self.total_hits,
            self.total_misses,
            self.evictions,
            self.current_size,
            self.capacity,
            self.hit_rate() * 100.0
        )
    }
}

// ============================================================================
// PyDescriptorCache
// ============================================================================

/// LRU cache for GPU descriptors.
///
/// Reduces allocation overhead by caching and reusing descriptors
/// with identical configurations.
///
/// # LRU Eviction
///
/// When the cache reaches capacity, the least recently used entry
/// is evicted to make room for new entries.
///
/// # Example
///
/// ```python
/// cache = PyDescriptorCache(capacity=1000)
///
/// # First call creates a new handle
/// handle1 = cache.get_or_create_buffer(PyBufferDescriptor.uniform(256))
///
/// # Second call returns the cached handle
/// handle2 = cache.get_or_create_buffer(PyBufferDescriptor.uniform(256))
/// assert handle1 == handle2
///
/// # Check hit rate
/// print(f"Hit rate: {cache.hit_rate():.2%}")
/// ```
#[pyclass(name = "DescriptorCache")]
#[derive(Debug)]
pub struct PyDescriptorCache {
    cache: HashMap<PyCacheKey, PyCachedDescriptor>,
    capacity: usize,
    resource_pool: PyResourcePool,
    // Statistics
    total_hits: u64,
    total_misses: u64,
    evictions: u64,
}

#[pymethods]
impl PyDescriptorCache {
    /// Create a new descriptor cache with the specified capacity.
    ///
    /// # Arguments
    /// * `capacity` - Maximum number of descriptors to cache
    #[new]
    #[pyo3(signature = (capacity=1000))]
    pub fn new(capacity: usize) -> Self {
        Self {
            cache: HashMap::with_capacity(capacity),
            capacity,
            resource_pool: PyResourcePool::new(),
            total_hits: 0,
            total_misses: 0,
            evictions: 0,
        }
    }

    /// Get or create a buffer handle for the given descriptor.
    ///
    /// If a cached handle exists for an equivalent descriptor, it is returned.
    /// Otherwise, a new handle is allocated and cached.
    ///
    /// # Arguments
    /// * `desc` - Buffer descriptor
    ///
    /// # Returns
    /// Resource handle for the buffer
    pub fn get_or_create_buffer(&mut self, desc: &PyBufferDescriptor) -> PyResourceHandle {
        let key = PyCacheKey::from_buffer(desc);
        self.get_or_create_internal(key, PyResourceType::Buffer)
    }

    /// Get or create a texture handle for the given descriptor.
    ///
    /// # Arguments
    /// * `desc` - Texture descriptor
    ///
    /// # Returns
    /// Resource handle for the texture
    pub fn get_or_create_texture(&mut self, desc: &PyTextureDescriptor) -> PyResourceHandle {
        let key = PyCacheKey::from_texture(desc);
        self.get_or_create_internal(key, PyResourceType::Texture)
    }

    /// Get or create a sampler handle for the given descriptor.
    ///
    /// # Arguments
    /// * `desc` - Sampler descriptor
    ///
    /// # Returns
    /// Resource handle for the sampler
    pub fn get_or_create_sampler(&mut self, desc: &PySamplerDescriptor) -> PyResourceHandle {
        let key = PyCacheKey::from_sampler(desc);
        self.get_or_create_internal(key, PyResourceType::Sampler)
    }

    /// Returns the cache hit rate as a value between 0.0 and 1.0.
    pub fn hit_rate(&self) -> f64 {
        let total = self.total_hits + self.total_misses;
        if total == 0 {
            0.0
        } else {
            self.total_hits as f64 / total as f64
        }
    }

    /// Returns comprehensive cache statistics.
    pub fn stats(&self) -> PyCacheStats {
        PyCacheStats {
            total_hits: self.total_hits,
            total_misses: self.total_misses,
            evictions: self.evictions,
            current_size: self.cache.len(),
            capacity: self.capacity,
        }
    }

    /// Clears all cached descriptors.
    pub fn clear(&mut self) {
        self.cache.clear();
        self.resource_pool.reset();
    }

    /// Removes entries older than the specified age.
    ///
    /// # Arguments
    /// * `max_age_secs` - Maximum age in seconds; entries older than this are removed
    ///
    /// # Returns
    /// Number of entries removed
    pub fn trim(&mut self, max_age_secs: f64) -> usize {
        let initial_size = self.cache.len();
        self.cache.retain(|_, entry| entry.age_secs() < max_age_secs);
        let removed = initial_size - self.cache.len();
        self.evictions += removed as u64;
        removed
    }

    /// Returns the current number of cached entries.
    #[getter]
    pub fn size(&self) -> usize {
        self.cache.len()
    }

    /// Returns the maximum capacity of the cache.
    #[getter]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Returns true if the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }

    /// Returns true if the cache is at capacity.
    pub fn is_full(&self) -> bool {
        self.cache.len() >= self.capacity
    }

    /// Check if a key exists in the cache.
    pub fn contains(&self, key: &PyCacheKey) -> bool {
        self.cache.contains_key(key)
    }

    /// Get the cached descriptor for a key without updating usage.
    pub fn peek(&self, key: &PyCacheKey) -> Option<PyCachedDescriptor> {
        self.cache.get(key).cloned()
    }

    /// Resize the cache capacity.
    ///
    /// If the new capacity is smaller than the current size,
    /// the least recently used entries are evicted.
    ///
    /// # Arguments
    /// * `new_capacity` - New maximum capacity
    pub fn resize(&mut self, new_capacity: usize) {
        if new_capacity < self.cache.len() {
            let to_evict = self.cache.len() - new_capacity;
            self.evict_lru(to_evict);
        }
        self.capacity = new_capacity;
    }

    pub fn __repr__(&self) -> String {
        format!(
            "DescriptorCache(size={}, capacity={}, hit_rate={:.1}%)",
            self.cache.len(),
            self.capacity,
            self.hit_rate() * 100.0
        )
    }

    pub fn __len__(&self) -> usize {
        self.cache.len()
    }
}

impl PyDescriptorCache {
    /// Internal implementation of get_or_create for any resource type.
    fn get_or_create_internal(
        &mut self,
        key: PyCacheKey,
        resource_type: PyResourceType,
    ) -> PyResourceHandle {
        // Check for cache hit
        if let Some(entry) = self.cache.get_mut(&key) {
            self.total_hits += 1;
            entry.touch();
            return entry.handle.clone();
        }

        // Cache miss
        self.total_misses += 1;

        // Evict if at capacity
        if self.cache.len() >= self.capacity && self.capacity > 0 {
            self.evict_lru(1);
        }

        // Create new entry
        let handle = self.resource_pool.allocate(resource_type);
        let entry = PyCachedDescriptor::new(handle.clone());
        self.cache.insert(key, entry);

        handle
    }

    /// Evict the least recently used entries.
    fn evict_lru(&mut self, count: usize) {
        if count == 0 || self.cache.is_empty() {
            return;
        }

        // Collect entries sorted by last_used (oldest first)
        let mut entries: Vec<_> = self.cache.iter().map(|(k, v)| (k.clone(), v.last_used)).collect();
        entries.sort_by(|a, b| a.1.cmp(&b.1));

        // Remove the oldest entries
        let to_remove: Vec<_> = entries.iter().take(count).map(|(k, _)| k.clone()).collect();
        for key in to_remove {
            if let Some(entry) = self.cache.remove(&key) {
                self.resource_pool.release(&entry.handle);
                self.evictions += 1;
            }
        }
    }
}

impl Default for PyDescriptorCache {
    fn default() -> Self {
        Self::new(1000)
    }
}

// ============================================================================
// Module Registration
// ============================================================================

/// Registers the descriptor cache types with the Python module.
pub fn register_module(
    _py: Python<'_>,
    parent: &Bound<'_, pyo3::types::PyModule>,
) -> PyResult<()> {
    parent.add_class::<PyTextureDescriptor>()?;
    parent.add_class::<PySamplerDescriptor>()?;
    parent.add_class::<PyCacheKey>()?;
    parent.add_class::<PyCachedDescriptor>()?;
    parent.add_class::<PyCacheStats>()?;
    parent.add_class::<PyDescriptorCache>()?;
    Ok(())
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // PyTextureDescriptor tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_texture_descriptor_new() {
        let desc = PyTextureDescriptor::new(512, 256, 1, 44, 5, 1, 0x10, Some("test".to_string()));
        assert_eq!(desc.width(), 512);
        assert_eq!(desc.height(), 256);
        assert_eq!(desc.depth(), 1);
        assert_eq!(desc.format(), 44);
        assert_eq!(desc.mip_levels(), 5);
        assert_eq!(desc.sample_count(), 1);
        assert_eq!(desc.usage(), 0x10);
        assert_eq!(desc.label(), Some("test".to_string()));
    }

    #[test]
    fn test_texture_descriptor_2d() {
        let desc = PyTextureDescriptor::texture_2d(1024, 768, 32, 0x04);
        assert_eq!(desc.width(), 1024);
        assert_eq!(desc.height(), 768);
        assert_eq!(desc.depth(), 1);
        assert_eq!(desc.format(), 32);
        assert_eq!(desc.usage(), 0x04);
    }

    #[test]
    fn test_texture_descriptor_3d() {
        let desc = PyTextureDescriptor::texture_3d(64, 64, 64, 1, 0);
        assert_eq!(desc.width(), 64);
        assert_eq!(desc.height(), 64);
        assert_eq!(desc.depth(), 64);
    }

    #[test]
    fn test_texture_descriptor_render_target() {
        let desc = PyTextureDescriptor::render_target(1920, 1080, 44, 4);
        assert_eq!(desc.width(), 1920);
        assert_eq!(desc.height(), 1080);
        assert_eq!(desc.sample_count(), 4);
        assert_eq!(desc.usage(), 0x10); // RENDER_ATTACHMENT
    }

    #[test]
    fn test_texture_descriptor_builders() {
        let desc = PyTextureDescriptor::texture_2d(512, 512, 0, 0)
            .with_label("my_texture")
            .with_mip_levels(10)
            .with_usage(0x04);
        assert_eq!(desc.label(), Some("my_texture".to_string()));
        assert_eq!(desc.mip_levels(), 10);
        assert_eq!(desc.usage(), 0x04);
    }

    #[test]
    fn test_texture_descriptor_hash_equality() {
        let a = PyTextureDescriptor::texture_2d(512, 512, 32, 0x04);
        let b = PyTextureDescriptor::texture_2d(512, 512, 32, 0x04);
        let c = PyTextureDescriptor::texture_2d(512, 256, 32, 0x04);

        assert_eq!(a.__hash__(), b.__hash__());
        assert_ne!(a.__hash__(), c.__hash__());
        assert!(a.__eq__(&b));
        assert!(!a.__eq__(&c));
    }

    // -------------------------------------------------------------------------
    // PySamplerDescriptor tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_sampler_descriptor_linear() {
        let desc = PySamplerDescriptor::linear();
        assert_eq!(desc.mag_filter(), filter_mode::LINEAR);
        assert_eq!(desc.min_filter(), filter_mode::LINEAR);
        assert_eq!(desc.address_mode_u(), address_mode::CLAMP_TO_EDGE);
    }

    #[test]
    fn test_sampler_descriptor_nearest() {
        let desc = PySamplerDescriptor::nearest();
        assert_eq!(desc.mag_filter(), filter_mode::NEAREST);
        assert_eq!(desc.min_filter(), filter_mode::NEAREST);
    }

    #[test]
    fn test_sampler_descriptor_repeating() {
        let desc = PySamplerDescriptor::repeating();
        assert_eq!(desc.address_mode_u(), address_mode::REPEAT);
        assert_eq!(desc.address_mode_v(), address_mode::REPEAT);
        assert_eq!(desc.address_mode_w(), address_mode::REPEAT);
    }

    #[test]
    fn test_sampler_descriptor_comparison() {
        let desc = PySamplerDescriptor::comparison(2);
        assert_eq!(desc.compare(), Some(2));
    }

    #[test]
    fn test_sampler_descriptor_builders() {
        let desc = PySamplerDescriptor::linear()
            .with_label("my_sampler")
            .with_anisotropy(16);
        assert_eq!(desc.label(), Some("my_sampler".to_string()));
        assert_eq!(desc.anisotropy_clamp(), 16);
    }

    #[test]
    fn test_sampler_descriptor_lod_clamp() {
        let desc = PySamplerDescriptor::new(0, 0, 0, 1, 1, 1, 0.5, 16.0, None, 1, None);
        assert!((desc.lod_min_clamp() - 0.5).abs() < f32::EPSILON);
        assert!((desc.lod_max_clamp() - 16.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_sampler_descriptor_hash_equality() {
        let a = PySamplerDescriptor::linear();
        let b = PySamplerDescriptor::linear();
        let c = PySamplerDescriptor::nearest();

        assert_eq!(a.__hash__(), b.__hash__());
        assert_ne!(a.__hash__(), c.__hash__());
        assert!(a.__eq__(&b));
        assert!(!a.__eq__(&c));
    }

    // -------------------------------------------------------------------------
    // PyCacheKey tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_cache_key_from_buffer() {
        let desc = PyBufferDescriptor::uniform(256);
        let key = PyCacheKey::from_buffer(&desc);
        assert!(key.is_buffer());
        assert!(!key.is_texture());
        assert!(!key.is_sampler());
        assert_eq!(key.key_type(), "Buffer");
    }

    #[test]
    fn test_cache_key_from_texture() {
        let desc = PyTextureDescriptor::texture_2d(512, 512, 0, 0);
        let key = PyCacheKey::from_texture(&desc);
        assert!(key.is_texture());
        assert!(!key.is_buffer());
    }

    #[test]
    fn test_cache_key_from_sampler() {
        let desc = PySamplerDescriptor::linear();
        let key = PyCacheKey::from_sampler(&desc);
        assert!(key.is_sampler());
        assert!(!key.is_buffer());
    }

    #[test]
    fn test_cache_key_equality() {
        let desc1 = PyBufferDescriptor::uniform(256);
        let desc2 = PyBufferDescriptor::uniform(256);
        let desc3 = PyBufferDescriptor::uniform(512);

        let key1 = PyCacheKey::from_buffer(&desc1);
        let key2 = PyCacheKey::from_buffer(&desc2);
        let key3 = PyCacheKey::from_buffer(&desc3);

        assert!(key1.__eq__(&key2));
        assert!(!key1.__eq__(&key3));
        assert_eq!(key1.__hash__(), key2.__hash__());
        assert_ne!(key1.__hash__(), key3.__hash__());
    }

    #[test]
    fn test_cache_key_label_excluded() {
        // Labels should NOT affect the cache key
        let desc1 = PyBufferDescriptor::uniform(256).with_label("buffer1");
        let desc2 = PyBufferDescriptor::uniform(256).with_label("buffer2");

        let key1 = PyCacheKey::from_buffer(&desc1);
        let key2 = PyCacheKey::from_buffer(&desc2);

        assert!(key1.__eq__(&key2));
        assert_eq!(key1.__hash__(), key2.__hash__());
    }

    // -------------------------------------------------------------------------
    // PyCacheStats tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_cache_stats_new() {
        let stats = PyCacheStats::new(100, 50, 10, 500, 1000);
        assert_eq!(stats.total_hits, 100);
        assert_eq!(stats.total_misses, 50);
        assert_eq!(stats.evictions, 10);
        assert_eq!(stats.current_size, 500);
        assert_eq!(stats.capacity, 1000);
    }

    #[test]
    fn test_cache_stats_hit_rate() {
        let stats = PyCacheStats::new(80, 20, 0, 100, 1000);
        assert!((stats.hit_rate() - 0.8).abs() < f64::EPSILON);

        let empty = PyCacheStats::default();
        assert_eq!(empty.hit_rate(), 0.0);
    }

    #[test]
    fn test_cache_stats_total_lookups() {
        let stats = PyCacheStats::new(100, 50, 0, 0, 0);
        assert_eq!(stats.total_lookups(), 150);
    }

    #[test]
    fn test_cache_stats_fill_ratio() {
        let stats = PyCacheStats::new(0, 0, 0, 500, 1000);
        assert!((stats.fill_ratio() - 0.5).abs() < f64::EPSILON);

        let empty_capacity = PyCacheStats::new(0, 0, 0, 0, 0);
        assert_eq!(empty_capacity.fill_ratio(), 0.0);
    }

    #[test]
    fn test_cache_stats_is_full() {
        let full = PyCacheStats::new(0, 0, 0, 1000, 1000);
        assert!(full.is_full());

        let not_full = PyCacheStats::new(0, 0, 0, 500, 1000);
        assert!(!not_full.is_full());
    }

    #[test]
    fn test_cache_stats_is_empty() {
        let empty = PyCacheStats::new(0, 0, 0, 0, 1000);
        assert!(empty.is_empty());

        let not_empty = PyCacheStats::new(0, 0, 0, 1, 1000);
        assert!(!not_empty.is_empty());
    }

    // -------------------------------------------------------------------------
    // PyDescriptorCache tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_cache_new() {
        let cache = PyDescriptorCache::new(500);
        assert_eq!(cache.capacity(), 500);
        assert_eq!(cache.size(), 0);
        assert!(cache.is_empty());
        assert!(!cache.is_full());
    }

    #[test]
    fn test_cache_get_or_create_buffer_miss() {
        let mut cache = PyDescriptorCache::new(100);
        let desc = PyBufferDescriptor::uniform(256);

        let handle = cache.get_or_create_buffer(&desc);
        assert!(handle.is_valid());
        assert!(handle.is_buffer());
        assert_eq!(cache.size(), 1);

        let stats = cache.stats();
        assert_eq!(stats.total_misses, 1);
        assert_eq!(stats.total_hits, 0);
    }

    #[test]
    fn test_cache_get_or_create_buffer_hit() {
        let mut cache = PyDescriptorCache::new(100);
        let desc = PyBufferDescriptor::uniform(256);

        let handle1 = cache.get_or_create_buffer(&desc);
        let handle2 = cache.get_or_create_buffer(&desc);

        assert_eq!(handle1.id(), handle2.id());
        assert_eq!(cache.size(), 1);

        let stats = cache.stats();
        assert_eq!(stats.total_misses, 1);
        assert_eq!(stats.total_hits, 1);
    }

    #[test]
    fn test_cache_get_or_create_texture() {
        let mut cache = PyDescriptorCache::new(100);
        let desc = PyTextureDescriptor::texture_2d(512, 512, 0, 0);

        let handle = cache.get_or_create_texture(&desc);
        assert!(handle.is_texture());
        assert_eq!(cache.size(), 1);
    }

    #[test]
    fn test_cache_get_or_create_sampler() {
        let mut cache = PyDescriptorCache::new(100);
        let desc = PySamplerDescriptor::linear();

        let handle = cache.get_or_create_sampler(&desc);
        assert!(handle.is_sampler());
        assert_eq!(cache.size(), 1);
    }

    #[test]
    fn test_cache_hit_rate() {
        let mut cache = PyDescriptorCache::new(100);
        let desc = PyBufferDescriptor::uniform(256);

        // 1 miss
        let _ = cache.get_or_create_buffer(&desc);
        assert_eq!(cache.hit_rate(), 0.0);

        // 1 hit
        let _ = cache.get_or_create_buffer(&desc);
        assert!((cache.hit_rate() - 0.5).abs() < f64::EPSILON);

        // 3 more hits
        let _ = cache.get_or_create_buffer(&desc);
        let _ = cache.get_or_create_buffer(&desc);
        let _ = cache.get_or_create_buffer(&desc);
        assert!((cache.hit_rate() - 0.8).abs() < f64::EPSILON);
    }

    #[test]
    fn test_cache_lru_eviction() {
        let mut cache = PyDescriptorCache::new(3);

        // Fill the cache
        let desc1 = PyBufferDescriptor::uniform(100);
        let desc2 = PyBufferDescriptor::uniform(200);
        let desc3 = PyBufferDescriptor::uniform(300);

        let handle1 = cache.get_or_create_buffer(&desc1);
        let _ = cache.get_or_create_buffer(&desc2);
        let _ = cache.get_or_create_buffer(&desc3);
        assert_eq!(cache.size(), 3);

        // Access desc1 again to make it more recent
        let _ = cache.get_or_create_buffer(&desc1);

        // Add a new entry, should evict desc2 (least recently used)
        let desc4 = PyBufferDescriptor::uniform(400);
        let _ = cache.get_or_create_buffer(&desc4);

        assert_eq!(cache.size(), 3);
        assert!(cache.stats().evictions >= 1);

        // Verify desc1 is still cached (it was accessed more recently)
        let key1 = PyCacheKey::from_buffer(&desc1);
        assert!(cache.contains(&key1));
    }

    #[test]
    fn test_cache_clear() {
        let mut cache = PyDescriptorCache::new(100);
        let desc = PyBufferDescriptor::uniform(256);

        let _ = cache.get_or_create_buffer(&desc);
        assert_eq!(cache.size(), 1);

        cache.clear();
        assert_eq!(cache.size(), 0);
        assert!(cache.is_empty());
    }

    #[test]
    fn test_cache_trim() {
        let mut cache = PyDescriptorCache::new(100);

        let desc1 = PyBufferDescriptor::uniform(100);
        let desc2 = PyBufferDescriptor::uniform(200);

        let _ = cache.get_or_create_buffer(&desc1);
        let _ = cache.get_or_create_buffer(&desc2);

        // All entries are very recent, trim with 0.0 should remove all
        let removed = cache.trim(0.0);
        assert_eq!(removed, 2);
        assert!(cache.is_empty());
    }

    #[test]
    fn test_cache_trim_preserves_recent() {
        let mut cache = PyDescriptorCache::new(100);

        let desc = PyBufferDescriptor::uniform(100);
        let _ = cache.get_or_create_buffer(&desc);

        // All entries are less than 1 hour old
        let removed = cache.trim(3600.0);
        assert_eq!(removed, 0);
        assert_eq!(cache.size(), 1);
    }

    #[test]
    fn test_cache_contains() {
        let mut cache = PyDescriptorCache::new(100);
        let desc = PyBufferDescriptor::uniform(256);
        let key = PyCacheKey::from_buffer(&desc);

        assert!(!cache.contains(&key));

        let _ = cache.get_or_create_buffer(&desc);
        assert!(cache.contains(&key));
    }

    #[test]
    fn test_cache_peek() {
        let mut cache = PyDescriptorCache::new(100);
        let desc = PyBufferDescriptor::uniform(256);
        let key = PyCacheKey::from_buffer(&desc);

        assert!(cache.peek(&key).is_none());

        let _ = cache.get_or_create_buffer(&desc);
        let cached = cache.peek(&key);
        assert!(cached.is_some());
        assert!(cached.unwrap().handle().is_buffer());
    }

    #[test]
    fn test_cache_resize_larger() {
        let mut cache = PyDescriptorCache::new(10);
        cache.resize(100);
        assert_eq!(cache.capacity(), 100);
    }

    #[test]
    fn test_cache_resize_smaller() {
        let mut cache = PyDescriptorCache::new(10);

        // Fill with 5 entries
        for i in 0..5 {
            let desc = PyBufferDescriptor::uniform(100 + i * 100);
            let _ = cache.get_or_create_buffer(&desc);
        }
        assert_eq!(cache.size(), 5);

        // Resize to 3, should evict 2
        cache.resize(3);
        assert_eq!(cache.capacity(), 3);
        assert_eq!(cache.size(), 3);
        assert!(cache.stats().evictions >= 2);
    }

    #[test]
    fn test_cache_is_full() {
        let mut cache = PyDescriptorCache::new(2);

        assert!(!cache.is_full());

        let desc1 = PyBufferDescriptor::uniform(100);
        let _ = cache.get_or_create_buffer(&desc1);
        assert!(!cache.is_full());

        let desc2 = PyBufferDescriptor::uniform(200);
        let _ = cache.get_or_create_buffer(&desc2);
        assert!(cache.is_full());
    }

    #[test]
    fn test_cache_multiple_types() {
        let mut cache = PyDescriptorCache::new(100);

        let buf_desc = PyBufferDescriptor::uniform(256);
        let tex_desc = PyTextureDescriptor::texture_2d(512, 512, 0, 0);
        let sam_desc = PySamplerDescriptor::linear();

        let buf_handle = cache.get_or_create_buffer(&buf_desc);
        let tex_handle = cache.get_or_create_texture(&tex_desc);
        let sam_handle = cache.get_or_create_sampler(&sam_desc);

        assert!(buf_handle.is_buffer());
        assert!(tex_handle.is_texture());
        assert!(sam_handle.is_sampler());
        assert_eq!(cache.size(), 3);
    }

    #[test]
    fn test_cache_different_buffer_sizes() {
        let mut cache = PyDescriptorCache::new(100);

        let desc1 = PyBufferDescriptor::uniform(256);
        let desc2 = PyBufferDescriptor::uniform(512);
        let desc3 = PyBufferDescriptor::uniform(256); // Same as desc1

        let handle1 = cache.get_or_create_buffer(&desc1);
        let handle2 = cache.get_or_create_buffer(&desc2);
        let handle3 = cache.get_or_create_buffer(&desc3);

        assert_ne!(handle1.id(), handle2.id()); // Different sizes
        assert_eq!(handle1.id(), handle3.id());  // Same size (cache hit)
        assert_eq!(cache.size(), 2);
    }

    #[test]
    fn test_cache_zero_capacity() {
        let mut cache = PyDescriptorCache::new(0);
        let desc = PyBufferDescriptor::uniform(256);

        // Even with 0 capacity, we should still allocate (and immediately evict old entries)
        let handle = cache.get_or_create_buffer(&desc);
        assert!(handle.is_valid());
    }

    #[test]
    fn test_cache_stats_comprehensive() {
        let mut cache = PyDescriptorCache::new(5);

        // Create 3 unique descriptors
        for i in 0..3 {
            let desc = PyBufferDescriptor::uniform(100 + i * 100);
            let _ = cache.get_or_create_buffer(&desc);
        }

        // Hit the first one twice more
        let desc1 = PyBufferDescriptor::uniform(100);
        let _ = cache.get_or_create_buffer(&desc1);
        let _ = cache.get_or_create_buffer(&desc1);

        let stats = cache.stats();
        assert_eq!(stats.total_misses, 3);
        assert_eq!(stats.total_hits, 2);
        assert_eq!(stats.current_size, 3);
        assert_eq!(stats.capacity, 5);
        assert!((stats.hit_rate() - 0.4).abs() < f64::EPSILON);
    }

    // -------------------------------------------------------------------------
    // PyCachedDescriptor tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_cached_descriptor_hit_count() {
        let mut cache = PyDescriptorCache::new(100);
        let desc = PyBufferDescriptor::uniform(256);
        let key = PyCacheKey::from_buffer(&desc);

        // First access (miss)
        let _ = cache.get_or_create_buffer(&desc);
        let cached = cache.peek(&key).unwrap();
        assert_eq!(cached.hit_count(), 0);

        // Second access (hit)
        let _ = cache.get_or_create_buffer(&desc);
        let cached = cache.peek(&key).unwrap();
        assert_eq!(cached.hit_count(), 1);

        // Third access (hit)
        let _ = cache.get_or_create_buffer(&desc);
        let cached = cache.peek(&key).unwrap();
        assert_eq!(cached.hit_count(), 2);
    }

    #[test]
    fn test_cached_descriptor_age() {
        let mut cache = PyDescriptorCache::new(100);
        let desc = PyBufferDescriptor::uniform(256);
        let key = PyCacheKey::from_buffer(&desc);

        let _ = cache.get_or_create_buffer(&desc);
        let cached = cache.peek(&key).unwrap();

        // Age should be very small (just created)
        assert!(cached.age_secs() < 1.0);
        assert!(cached.lifetime_secs() < 1.0);
    }

    // -------------------------------------------------------------------------
    // Edge case tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_cache_single_capacity() {
        let mut cache = PyDescriptorCache::new(1);

        let desc1 = PyBufferDescriptor::uniform(100);
        let desc2 = PyBufferDescriptor::uniform(200);

        let handle1 = cache.get_or_create_buffer(&desc1);
        assert_eq!(cache.size(), 1);

        let handle2 = cache.get_or_create_buffer(&desc2);
        assert_eq!(cache.size(), 1);
        assert_ne!(handle1.id(), handle2.id());

        // desc1 should have been evicted
        let key1 = PyCacheKey::from_buffer(&desc1);
        assert!(!cache.contains(&key1));
    }

    #[test]
    fn test_cache_rapid_access() {
        let mut cache = PyDescriptorCache::new(100);
        let desc = PyBufferDescriptor::uniform(256);

        // Rapid consecutive access
        for _ in 0..1000 {
            let _ = cache.get_or_create_buffer(&desc);
        }

        let stats = cache.stats();
        assert_eq!(stats.total_misses, 1);
        assert_eq!(stats.total_hits, 999);
        assert!(stats.hit_rate() > 0.99);
    }

    #[test]
    fn test_cache_many_unique_descriptors() {
        let mut cache = PyDescriptorCache::new(50);

        // Create 100 unique descriptors (exceeds capacity)
        for i in 0..100 {
            let desc = PyBufferDescriptor::uniform(i * 100);
            let _ = cache.get_or_create_buffer(&desc);
        }

        assert_eq!(cache.size(), 50);
        assert!(cache.stats().evictions >= 50);
    }

    #[test]
    fn test_texture_descriptor_default() {
        let desc = PyTextureDescriptor::default();
        assert_eq!(desc.width(), 1);
        assert_eq!(desc.height(), 1);
        assert_eq!(desc.depth(), 1);
    }

    #[test]
    fn test_sampler_descriptor_default() {
        let desc = PySamplerDescriptor::default();
        assert_eq!(desc.mag_filter(), filter_mode::LINEAR);
        assert_eq!(desc.min_filter(), filter_mode::LINEAR);
    }

    #[test]
    fn test_cache_default() {
        let cache = PyDescriptorCache::default();
        assert_eq!(cache.capacity(), 1000);
    }

    #[test]
    fn test_cache_repr() {
        let cache = PyDescriptorCache::new(100);
        let repr = cache.__repr__();
        assert!(repr.contains("DescriptorCache"));
        assert!(repr.contains("100"));
    }

    #[test]
    fn test_cache_len() {
        let mut cache = PyDescriptorCache::new(100);
        assert_eq!(cache.__len__(), 0);

        let desc = PyBufferDescriptor::uniform(256);
        let _ = cache.get_or_create_buffer(&desc);
        assert_eq!(cache.__len__(), 1);
    }
}
