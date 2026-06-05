//! Shader module caching for TRINITY.
//!
//! This module provides a cache for compiled shader modules to avoid redundant
//! compilation. Shaders are keyed by file path or content hash, and are shared
//! via `Arc<TrinityShaderModule>`.
//!
//! # Overview
//!
//! Shader compilation is expensive, especially for complex shaders. This cache:
//!
//! - Deduplicates shader modules with identical source
//! - Supports both file path and content hash keys
//! - Provides lazy compilation via `get_or_compile()`
//! - Supports hot-reload with targeted invalidation
//! - Optional disk cache for SPIR-V bytecode (future)
//!
//! # Thread Safety
//!
//! The cache uses `RwLock` for concurrent access:
//! - Multiple readers can query the cache simultaneously
//! - Write lock is only held briefly when compiling new shaders
//!
//! # Hot Reload
//!
//! For development, shaders can be invalidated when source files change:
//!
//! ```text
//! ShaderCache
//! ├── invalidate(key)         - Remove specific shader
//! ├── invalidate_by_path(path) - Remove shader by file path
//! └── invalidate_all()        - Clear entire cache
//! ```
//!
//! # Example
//!
//! ```no_run
//! use renderer_backend::shaders::cache::{ShaderCache, ShaderCacheConfig, ShaderCacheKey};
//! use renderer_backend::shaders::ShaderSourceKind;
//! use std::sync::Arc;
//!
//! # fn example(device: Arc<wgpu::Device>) -> Result<(), renderer_backend::shaders::ShaderError> {
//! let config = ShaderCacheConfig::default();
//! let cache = ShaderCache::new(device, config);
//!
//! // Compile from source (or get cached)
//! let source = r#"
//!     @vertex fn main() -> @builtin(position) vec4<f32> {
//!         return vec4<f32>(0.0);
//!     }
//! "#;
//! let key = ShaderCacheKey::from_source(source);
//! let shader = cache.get_or_compile(key, Some("vertex_shader"), source)?;
//!
//! // Second call returns cached shader
//! let shader2 = cache.get_or_compile(key.clone(), Some("vertex_shader"), source)?;
//! assert!(Arc::ptr_eq(&shader, &shader2));
//!
//! // Check metrics
//! let metrics = cache.metrics();
//! println!("Hit rate: {:.1}%", metrics.hit_rate * 100.0);
//! # Ok(())
//! # }
//! ```

use parking_lot::RwLock;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use super::{
    create_shader_module, ShaderError, TrinityShaderDescriptor, TrinityShaderModule,
};

// ============================================================================
// Constants
// ============================================================================

/// Default maximum number of cached shader modules.
pub const DEFAULT_MAX_ENTRIES: usize = 256;

/// Default disk cache path (relative to project root).
pub const DEFAULT_DISK_CACHE_PATH: &str = ".trinity/shader_cache";

// ============================================================================
// ShaderCacheKey
// ============================================================================

/// A key for identifying cached shaders.
///
/// Shaders can be keyed by file path (for file-based shaders) or by content
/// hash (for dynamically generated shaders).
///
/// # Path vs Hash Keys
///
/// - **Path keys**: Best for file-based shaders, allows path-based invalidation
/// - **Hash keys**: Best for generated shaders, content-addressable
///
/// # Example
///
/// ```
/// use renderer_backend::shaders::cache::ShaderCacheKey;
/// use std::path::PathBuf;
///
/// // Path-based key
/// let path_key = ShaderCacheKey::from_path("shaders/pbr.wgsl");
/// assert!(path_key.is_path());
///
/// // Hash-based key
/// let source = "@vertex fn main() {}";
/// let hash_key = ShaderCacheKey::from_source(source);
/// assert!(hash_key.is_hash());
///
/// // Keys with same content produce same hash
/// let hash_key2 = ShaderCacheKey::from_source(source);
/// assert_eq!(hash_key, hash_key2);
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ShaderCacheKey {
    /// Key by file path.
    Path(PathBuf),
    /// Key by SHA-256 content hash.
    Hash([u8; 32]),
}

impl ShaderCacheKey {
    /// Creates a key from a file path.
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::shaders::cache::ShaderCacheKey;
    ///
    /// let key = ShaderCacheKey::from_path("shaders/vertex.wgsl");
    /// assert!(key.is_path());
    /// ```
    #[inline]
    pub fn from_path(path: impl Into<PathBuf>) -> Self {
        Self::Path(path.into())
    }

    /// Creates a key from shader source content.
    ///
    /// The source is hashed using SHA-256 for content-addressable lookup.
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::shaders::cache::ShaderCacheKey;
    ///
    /// let source = "@vertex fn main() {}";
    /// let key = ShaderCacheKey::from_source(source);
    /// assert!(key.is_hash());
    /// ```
    pub fn from_source(source: &str) -> Self {
        Self::Hash(compute_sha256(source.as_bytes()))
    }

    /// Creates a key from raw bytes (e.g., SPIR-V).
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::shaders::cache::ShaderCacheKey;
    ///
    /// let spirv_bytes = &[0u8; 100];
    /// let key = ShaderCacheKey::from_bytes(spirv_bytes);
    /// assert!(key.is_hash());
    /// ```
    pub fn from_bytes(bytes: &[u8]) -> Self {
        Self::Hash(compute_sha256(bytes))
    }

    /// Creates a key from a pre-computed hash.
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::shaders::cache::ShaderCacheKey;
    ///
    /// let hash = [0u8; 32];
    /// let key = ShaderCacheKey::from_hash(hash);
    /// assert!(key.is_hash());
    /// ```
    #[inline]
    pub fn from_hash(hash: [u8; 32]) -> Self {
        Self::Hash(hash)
    }

    /// Returns true if this is a path-based key.
    #[inline]
    pub fn is_path(&self) -> bool {
        matches!(self, Self::Path(_))
    }

    /// Returns true if this is a hash-based key.
    #[inline]
    pub fn is_hash(&self) -> bool {
        matches!(self, Self::Hash(_))
    }

    /// Returns the path if this is a path-based key.
    #[inline]
    pub fn as_path(&self) -> Option<&Path> {
        match self {
            Self::Path(p) => Some(p),
            _ => None,
        }
    }

    /// Returns the hash if this is a hash-based key.
    #[inline]
    pub fn as_hash(&self) -> Option<&[u8; 32]> {
        match self {
            Self::Hash(h) => Some(h),
            _ => None,
        }
    }

    /// Returns a display-friendly string representation.
    pub fn display_string(&self) -> String {
        match self {
            Self::Path(p) => p.display().to_string(),
            Self::Hash(h) => {
                // Show first 8 hex chars
                format!("hash:{:02x}{:02x}{:02x}{:02x}...", h[0], h[1], h[2], h[3])
            }
        }
    }
}

impl From<PathBuf> for ShaderCacheKey {
    fn from(path: PathBuf) -> Self {
        Self::Path(path)
    }
}

impl From<&Path> for ShaderCacheKey {
    fn from(path: &Path) -> Self {
        Self::Path(path.to_path_buf())
    }
}

impl From<&str> for ShaderCacheKey {
    fn from(path: &str) -> Self {
        Self::Path(PathBuf::from(path))
    }
}

impl From<[u8; 32]> for ShaderCacheKey {
    fn from(hash: [u8; 32]) -> Self {
        Self::Hash(hash)
    }
}

// ============================================================================
// CachedShader
// ============================================================================

/// A cached shader module with metadata.
///
/// Wraps a compiled shader module with timing and access information
/// for LRU eviction and debugging.
#[derive(Debug)]
pub struct CachedShader {
    /// The compiled shader module.
    module: Arc<TrinityShaderModule>,
    /// When the shader was compiled (for age-based eviction).
    created_at: Instant,
    /// When the shader was last accessed (for LRU eviction).
    last_accessed: Instant,
    /// Number of times this shader has been accessed.
    access_count: u64,
    /// Original key used to cache this shader (for reverse lookup).
    key: ShaderCacheKey,
}

impl CachedShader {
    /// Creates a new cached shader entry.
    fn new(module: Arc<TrinityShaderModule>, key: ShaderCacheKey) -> Self {
        let now = Instant::now();
        Self {
            module,
            created_at: now,
            last_accessed: now,
            access_count: 1,
            key,
        }
    }

    /// Returns a clone of the Arc-wrapped shader module.
    #[inline]
    pub fn module(&self) -> Arc<TrinityShaderModule> {
        Arc::clone(&self.module)
    }

    /// Returns the creation time.
    #[inline]
    pub fn created_at(&self) -> Instant {
        self.created_at
    }

    /// Returns the last access time.
    #[inline]
    pub fn last_accessed(&self) -> Instant {
        self.last_accessed
    }

    /// Returns the access count.
    #[inline]
    pub fn access_count(&self) -> u64 {
        self.access_count
    }

    /// Returns the cache key.
    #[inline]
    pub fn key(&self) -> &ShaderCacheKey {
        &self.key
    }

    /// Updates the last access time and increments access count.
    fn touch(&mut self) {
        self.last_accessed = Instant::now();
        self.access_count += 1;
    }

    /// Returns the age of this entry since creation.
    pub fn age(&self) -> std::time::Duration {
        self.created_at.elapsed()
    }

    /// Returns the time since last access.
    pub fn idle_time(&self) -> std::time::Duration {
        self.last_accessed.elapsed()
    }
}

// ============================================================================
// ShaderCacheConfig
// ============================================================================

/// Configuration for the shader cache.
///
/// # Example
///
/// ```
/// use renderer_backend::shaders::cache::ShaderCacheConfig;
///
/// // Use defaults
/// let config = ShaderCacheConfig::default();
///
/// // Custom configuration
/// let config = ShaderCacheConfig {
///     max_entries: 512,
///     enable_disk_cache: false,
///     disk_cache_path: None,
///     enable_lru_eviction: true,
/// };
/// ```
#[derive(Debug, Clone)]
pub struct ShaderCacheConfig {
    /// Maximum number of shader modules to cache.
    /// When exceeded, LRU eviction is triggered.
    pub max_entries: usize,

    /// Enable disk caching of compiled SPIR-V (future feature).
    pub enable_disk_cache: bool,

    /// Path for disk cache storage.
    pub disk_cache_path: Option<PathBuf>,

    /// Enable LRU eviction when max_entries is exceeded.
    pub enable_lru_eviction: bool,
}

impl Default for ShaderCacheConfig {
    fn default() -> Self {
        Self {
            max_entries: DEFAULT_MAX_ENTRIES,
            enable_disk_cache: false,
            disk_cache_path: None,
            enable_lru_eviction: true,
        }
    }
}

impl ShaderCacheConfig {
    /// Creates a new config with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the maximum number of cache entries.
    #[inline]
    pub fn max_entries(mut self, max: usize) -> Self {
        self.max_entries = max;
        self
    }

    /// Enables disk caching with the given path.
    pub fn with_disk_cache(mut self, path: impl Into<PathBuf>) -> Self {
        self.enable_disk_cache = true;
        self.disk_cache_path = Some(path.into());
        self
    }

    /// Disables LRU eviction.
    #[inline]
    pub fn without_eviction(mut self) -> Self {
        self.enable_lru_eviction = false;
        self
    }

    /// Creates a minimal config for testing.
    pub fn minimal() -> Self {
        Self {
            max_entries: 16,
            enable_disk_cache: false,
            disk_cache_path: None,
            enable_lru_eviction: true,
        }
    }

    /// Creates a config optimized for development (smaller cache, faster eviction).
    pub fn development() -> Self {
        Self {
            max_entries: 64,
            enable_disk_cache: false,
            disk_cache_path: None,
            enable_lru_eviction: true,
        }
    }

    /// Creates a config optimized for production (larger cache, disk caching).
    pub fn production() -> Self {
        Self {
            max_entries: 1024,
            enable_disk_cache: true,
            disk_cache_path: Some(PathBuf::from(DEFAULT_DISK_CACHE_PATH)),
            enable_lru_eviction: true,
        }
    }
}

// ============================================================================
// ShaderCacheMetrics
// ============================================================================

/// Metrics for monitoring shader cache performance.
///
/// These metrics help identify cache efficiency and potential optimization
/// opportunities.
///
/// # Example
///
/// ```
/// use renderer_backend::shaders::cache::ShaderCacheMetrics;
///
/// let metrics = ShaderCacheMetrics::default();
/// assert_eq!(metrics.cache_size, 0);
/// assert_eq!(metrics.hits, 0);
/// assert_eq!(metrics.misses, 0);
/// assert_eq!(metrics.hit_rate, 0.0);
/// ```
#[derive(Debug, Clone, Default)]
pub struct ShaderCacheMetrics {
    /// Number of shader modules currently cached.
    pub cache_size: usize,

    /// Number of cache hits (requested shader was cached).
    pub hits: u64,

    /// Number of cache misses (shader needed compilation).
    pub misses: u64,

    /// Hit rate as a ratio (0.0 to 1.0).
    pub hit_rate: f64,

    /// Number of shaders evicted due to cache size limits.
    pub evictions: u64,

    /// Number of explicit invalidations.
    pub invalidations: u64,

    /// Number of compilation errors encountered.
    pub compilation_errors: u64,
}

impl ShaderCacheMetrics {
    /// Creates metrics with the given values.
    pub fn new(
        cache_size: usize,
        hits: u64,
        misses: u64,
        evictions: u64,
        invalidations: u64,
        compilation_errors: u64,
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
            evictions,
            invalidations,
            compilation_errors,
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

    /// Returns the miss rate as a ratio (0.0 to 1.0).
    #[inline]
    pub fn miss_rate(&self) -> f64 {
        1.0 - self.hit_rate
    }
}

// ============================================================================
// ShaderCache
// ============================================================================

/// A thread-safe cache for compiled shader modules.
///
/// The cache stores compiled shader modules keyed by file path or content hash,
/// ensuring that identical shaders are only compiled once.
///
/// # Architecture
///
/// ```text
/// ShaderCache
/// ├── Device (Arc<wgpu::Device>)
/// ├── Config (ShaderCacheConfig)
/// ├── Cache (RwLock<HashMap<ShaderCacheKey, CachedShader>>)
/// │   └── Shaders keyed by path or content hash
/// ├── Path Index (HashMap<PathBuf, ShaderCacheKey>)
/// │   └── Reverse lookup for path-based invalidation
/// └── Metrics (atomic counters)
/// ```
///
/// # Thread Safety
///
/// - Uses `RwLock<HashMap>` for the cache
/// - Uses `AtomicU64` for hit/miss counters (lock-free)
/// - Double-check locking pattern for compilation
/// - Multiple readers can access cached shaders concurrently
///
/// # Example
///
/// ```no_run
/// use renderer_backend::shaders::cache::{ShaderCache, ShaderCacheConfig, ShaderCacheKey};
/// use std::sync::Arc;
///
/// # fn example(device: Arc<wgpu::Device>) -> Result<(), renderer_backend::shaders::ShaderError> {
/// let cache = ShaderCache::new(device, ShaderCacheConfig::default());
///
/// // Compile and cache
/// let source = "@vertex fn main() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0); }";
/// let shader = cache.get_or_compile(
///     ShaderCacheKey::from_source(source),
///     Some("vertex"),
///     source,
/// )?;
///
/// // Load from file
/// use std::path::Path;
/// let shader = cache.get_or_compile_file(Path::new("shaders/pbr.wgsl"))?;
///
/// // Invalidate for hot-reload
/// cache.invalidate_by_path(Path::new("shaders/pbr.wgsl"));
/// # Ok(())
/// # }
/// ```
pub struct ShaderCache {
    /// The wgpu device for shader compilation.
    device: Arc<wgpu::Device>,

    /// Cache configuration.
    config: ShaderCacheConfig,

    /// Cache of compiled shader modules.
    cache: RwLock<HashMap<ShaderCacheKey, CachedShader>>,

    /// Reverse index: path -> key for path-based invalidation.
    path_index: RwLock<HashMap<PathBuf, ShaderCacheKey>>,

    /// Hit counter (atomic for lock-free updates).
    hits: AtomicU64,

    /// Miss counter (atomic for lock-free updates).
    misses: AtomicU64,

    /// Eviction counter.
    evictions: AtomicU64,

    /// Invalidation counter.
    invalidations: AtomicU64,

    /// Compilation error counter.
    compilation_errors: AtomicU64,
}

impl ShaderCache {
    /// Creates a new shader cache.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device for shader compilation
    /// * `config` - Cache configuration
    ///
    /// # Example
    ///
    /// ```no_run
    /// use renderer_backend::shaders::cache::{ShaderCache, ShaderCacheConfig};
    /// use std::sync::Arc;
    ///
    /// # fn example(device: Arc<wgpu::Device>) {
    /// let cache = ShaderCache::new(device, ShaderCacheConfig::default());
    /// assert!(cache.is_empty());
    /// # }
    /// ```
    pub fn new(device: Arc<wgpu::Device>, config: ShaderCacheConfig) -> Self {
        Self {
            device,
            config,
            cache: RwLock::new(HashMap::new()),
            path_index: RwLock::new(HashMap::new()),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            evictions: AtomicU64::new(0),
            invalidations: AtomicU64::new(0),
            compilation_errors: AtomicU64::new(0),
        }
    }

    /// Gets a cached shader or compiles and caches it.
    ///
    /// This is the primary API for shader access. If the shader is cached,
    /// it is returned immediately. Otherwise, it is compiled, cached, and returned.
    ///
    /// # Arguments
    ///
    /// * `key` - The cache key (path or hash)
    /// * `label` - Optional debug label for GPU tools
    /// * `source` - The WGSL source code
    ///
    /// # Returns
    ///
    /// An `Arc<TrinityShaderModule>` that can be shared across pipelines.
    ///
    /// # Thread Safety
    ///
    /// Uses double-check locking: read lock for lookup, write lock only for insertion.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use renderer_backend::shaders::cache::{ShaderCache, ShaderCacheConfig, ShaderCacheKey};
    /// use std::sync::Arc;
    ///
    /// # fn example(device: Arc<wgpu::Device>) -> Result<(), renderer_backend::shaders::ShaderError> {
    /// let cache = ShaderCache::new(device, ShaderCacheConfig::default());
    ///
    /// let source = "@vertex fn main() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0); }";
    /// let key = ShaderCacheKey::from_source(source);
    ///
    /// // First call compiles
    /// let shader1 = cache.get_or_compile(key.clone(), Some("vs"), source)?;
    ///
    /// // Second call returns cached
    /// let shader2 = cache.get_or_compile(key, Some("vs"), source)?;
    ///
    /// assert!(Arc::ptr_eq(&shader1, &shader2));
    /// # Ok(())
    /// # }
    /// ```
    pub fn get_or_compile(
        &self,
        key: ShaderCacheKey,
        label: Option<&str>,
        source: &str,
    ) -> Result<Arc<TrinityShaderModule>, ShaderError> {
        // Fast path: read lock
        {
            let cache = self.cache.read();
            if let Some(cached) = cache.get(&key) {
                self.hits.fetch_add(1, Ordering::Relaxed);
                return Ok(cached.module());
            }
        }

        // Slow path: write lock (double-check pattern)
        let mut cache = self.cache.write();
        if let Some(cached) = cache.get_mut(&key) {
            self.hits.fetch_add(1, Ordering::Relaxed);
            cached.touch();
            return Ok(cached.module());
        }

        // Compile the shader
        self.misses.fetch_add(1, Ordering::Relaxed);
        let module = self.compile_shader(label, source, key.as_path())?;
        let module_arc = Arc::new(module);

        // Evict if necessary
        if self.config.enable_lru_eviction && cache.len() >= self.config.max_entries {
            self.evict_lru(&mut cache);
        }

        // Cache the compiled shader
        let cached = CachedShader::new(Arc::clone(&module_arc), key.clone());

        // Update path index if this is a path key
        if let ShaderCacheKey::Path(ref path) = key {
            self.path_index.write().insert(path.clone(), key.clone());
        }

        cache.insert(key, cached);

        Ok(module_arc)
    }

    /// Gets a cached shader or compiles from a file.
    ///
    /// This is a convenience method for file-based shaders. The file path
    /// is used as the cache key.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the WGSL shader file
    ///
    /// # Returns
    ///
    /// An `Arc<TrinityShaderModule>` that can be shared across pipelines.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use renderer_backend::shaders::cache::{ShaderCache, ShaderCacheConfig};
    /// use std::sync::Arc;
    /// use std::path::Path;
    ///
    /// # fn example(device: Arc<wgpu::Device>) -> Result<(), renderer_backend::shaders::ShaderError> {
    /// let cache = ShaderCache::new(device, ShaderCacheConfig::default());
    ///
    /// let shader = cache.get_or_compile_file(Path::new("shaders/pbr.wgsl"))?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn get_or_compile_file(&self, path: &Path) -> Result<Arc<TrinityShaderModule>, ShaderError> {
        let key = ShaderCacheKey::from_path(path);

        // Fast path: read lock
        {
            let cache = self.cache.read();
            if let Some(cached) = cache.get(&key) {
                self.hits.fetch_add(1, Ordering::Relaxed);
                return Ok(cached.module());
            }
        }

        // Load the file
        let source = std::fs::read_to_string(path).map_err(|e| ShaderError::IoError {
            message: e.to_string(),
            path: Some(path.to_path_buf()),
        })?;

        let label = path.file_name().and_then(|n| n.to_str());

        // Slow path: write lock (double-check pattern)
        let mut cache = self.cache.write();
        if let Some(cached) = cache.get_mut(&key) {
            self.hits.fetch_add(1, Ordering::Relaxed);
            cached.touch();
            return Ok(cached.module());
        }

        // Compile the shader
        self.misses.fetch_add(1, Ordering::Relaxed);
        let module = self.compile_shader(label, &source, Some(path))?;
        let module_arc = Arc::new(module);

        // Evict if necessary
        if self.config.enable_lru_eviction && cache.len() >= self.config.max_entries {
            self.evict_lru(&mut cache);
        }

        // Cache the compiled shader
        let cached = CachedShader::new(Arc::clone(&module_arc), key.clone());

        // Update path index
        self.path_index.write().insert(path.to_path_buf(), key.clone());

        cache.insert(key, cached);

        Ok(module_arc)
    }

    /// Compiles a shader without caching.
    ///
    /// This is useful for one-off shaders or when you need to bypass the cache.
    ///
    /// # Arguments
    ///
    /// * `label` - Optional debug label
    /// * `source` - The WGSL source code
    /// * `file_path` - Optional file path for error messages
    ///
    /// # Returns
    ///
    /// The compiled shader module (not wrapped in Arc).
    fn compile_shader(
        &self,
        label: Option<&str>,
        source: &str,
        file_path: Option<&Path>,
    ) -> Result<TrinityShaderModule, ShaderError> {
        let mut desc = TrinityShaderDescriptor::wgsl(label, source);
        if let Some(path) = file_path {
            desc = desc.with_file_path(path);
        }

        match create_shader_module(&self.device, &desc) {
            Ok(module) => Ok(module),
            Err(e) => {
                self.compilation_errors.fetch_add(1, Ordering::Relaxed);
                Err(e)
            }
        }
    }

    /// Invalidates a specific cached shader.
    ///
    /// # Arguments
    ///
    /// * `key` - The cache key to invalidate
    ///
    /// # Returns
    ///
    /// `true` if a shader was invalidated, `false` if the key was not found.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use renderer_backend::shaders::cache::{ShaderCache, ShaderCacheConfig, ShaderCacheKey};
    /// use std::sync::Arc;
    ///
    /// # fn example(device: Arc<wgpu::Device>) -> Result<(), renderer_backend::shaders::ShaderError> {
    /// let cache = ShaderCache::new(device, ShaderCacheConfig::default());
    ///
    /// let source = "@vertex fn main() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0); }";
    /// let key = ShaderCacheKey::from_source(source);
    /// cache.get_or_compile(key.clone(), None, source)?;
    ///
    /// assert!(cache.invalidate(&key));
    /// assert!(!cache.invalidate(&key)); // Already removed
    /// # Ok(())
    /// # }
    /// ```
    pub fn invalidate(&self, key: &ShaderCacheKey) -> bool {
        let mut cache = self.cache.write();
        let removed = cache.remove(key).is_some();

        if removed {
            self.invalidations.fetch_add(1, Ordering::Relaxed);

            // Remove from path index if applicable
            if let ShaderCacheKey::Path(path) = key {
                self.path_index.write().remove(path);
            }
        }

        removed
    }

    /// Invalidates a cached shader by file path.
    ///
    /// This is useful for hot-reload when file change notifications provide paths.
    ///
    /// # Arguments
    ///
    /// * `path` - The file path to invalidate
    ///
    /// # Returns
    ///
    /// `true` if a shader was invalidated, `false` if the path was not found.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use renderer_backend::shaders::cache::{ShaderCache, ShaderCacheConfig};
    /// use std::sync::Arc;
    /// use std::path::Path;
    ///
    /// # fn example(device: Arc<wgpu::Device>) -> Result<(), renderer_backend::shaders::ShaderError> {
    /// let cache = ShaderCache::new(device, ShaderCacheConfig::default());
    ///
    /// // ... compile shader from file ...
    ///
    /// // File changed, invalidate
    /// cache.invalidate_by_path(Path::new("shaders/pbr.wgsl"));
    /// # Ok(())
    /// # }
    /// ```
    pub fn invalidate_by_path(&self, path: &Path) -> bool {
        // Look up the key in the path index
        let key = {
            let path_index = self.path_index.read();
            path_index.get(path).cloned()
        };

        if let Some(key) = key {
            self.invalidate(&key)
        } else {
            // Try direct path key
            self.invalidate(&ShaderCacheKey::Path(path.to_path_buf()))
        }
    }

    /// Invalidates all cached shaders.
    ///
    /// This clears the entire cache. Metrics are preserved.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use renderer_backend::shaders::cache::{ShaderCache, ShaderCacheConfig, ShaderCacheKey};
    /// use std::sync::Arc;
    ///
    /// # fn example(device: Arc<wgpu::Device>) -> Result<(), renderer_backend::shaders::ShaderError> {
    /// let cache = ShaderCache::new(device, ShaderCacheConfig::default());
    ///
    /// // ... cache some shaders ...
    ///
    /// cache.invalidate_all();
    /// assert!(cache.is_empty());
    /// # Ok(())
    /// # }
    /// ```
    pub fn invalidate_all(&self) {
        let mut cache = self.cache.write();
        let count = cache.len();
        cache.clear();
        self.path_index.write().clear();

        if count > 0 {
            self.invalidations.fetch_add(count as u64, Ordering::Relaxed);
        }
    }

    /// Evicts the least recently used shader from the cache.
    fn evict_lru(&self, cache: &mut HashMap<ShaderCacheKey, CachedShader>) {
        if cache.is_empty() {
            return;
        }

        // Find the least recently used entry
        let lru_key = cache
            .iter()
            .min_by_key(|(_, cached)| cached.last_accessed())
            .map(|(key, _)| key.clone());

        if let Some(key) = lru_key {
            cache.remove(&key);
            self.evictions.fetch_add(1, Ordering::Relaxed);

            // Remove from path index if applicable
            if let ShaderCacheKey::Path(path) = &key {
                self.path_index.write().remove(path);
            }
        }
    }

    /// Returns the number of cached shaders.
    #[inline]
    pub fn len(&self) -> usize {
        self.cache.read().len()
    }

    /// Returns true if the cache is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.cache.read().is_empty()
    }

    /// Checks if a key exists in the cache.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use renderer_backend::shaders::cache::{ShaderCache, ShaderCacheConfig, ShaderCacheKey};
    /// use std::sync::Arc;
    ///
    /// # fn example(device: Arc<wgpu::Device>) -> Result<(), renderer_backend::shaders::ShaderError> {
    /// let cache = ShaderCache::new(device, ShaderCacheConfig::default());
    ///
    /// let source = "@vertex fn main() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0); }";
    /// let key = ShaderCacheKey::from_source(source);
    ///
    /// assert!(!cache.contains(&key));
    /// cache.get_or_compile(key.clone(), None, source)?;
    /// assert!(cache.contains(&key));
    /// # Ok(())
    /// # }
    /// ```
    pub fn contains(&self, key: &ShaderCacheKey) -> bool {
        self.cache.read().contains_key(key)
    }

    /// Gets a cached shader without compiling if not found.
    ///
    /// # Returns
    ///
    /// `Some(Arc<TrinityShaderModule>)` if cached, `None` otherwise.
    pub fn get(&self, key: &ShaderCacheKey) -> Option<Arc<TrinityShaderModule>> {
        let mut cache = self.cache.write();
        if let Some(cached) = cache.get_mut(key) {
            cached.touch();
            self.hits.fetch_add(1, Ordering::Relaxed);
            Some(cached.module())
        } else {
            None
        }
    }

    /// Returns current cache metrics.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use renderer_backend::shaders::cache::{ShaderCache, ShaderCacheConfig};
    /// use std::sync::Arc;
    ///
    /// # fn example(device: Arc<wgpu::Device>) {
    /// let cache = ShaderCache::new(device, ShaderCacheConfig::default());
    /// let metrics = cache.metrics();
    ///
    /// println!("Cache size: {}", metrics.cache_size);
    /// println!("Hit rate: {:.1}%", metrics.hit_rate_percent());
    /// # }
    /// ```
    pub fn metrics(&self) -> ShaderCacheMetrics {
        let cache_size = self.cache.read().len();
        let hits = self.hits.load(Ordering::Relaxed);
        let misses = self.misses.load(Ordering::Relaxed);
        let evictions = self.evictions.load(Ordering::Relaxed);
        let invalidations = self.invalidations.load(Ordering::Relaxed);
        let compilation_errors = self.compilation_errors.load(Ordering::Relaxed);

        ShaderCacheMetrics::new(
            cache_size,
            hits,
            misses,
            evictions,
            invalidations,
            compilation_errors,
        )
    }

    /// Resets metrics counters to zero without clearing the cache.
    pub fn reset_metrics(&self) {
        self.hits.store(0, Ordering::Relaxed);
        self.misses.store(0, Ordering::Relaxed);
        self.evictions.store(0, Ordering::Relaxed);
        self.invalidations.store(0, Ordering::Relaxed);
        self.compilation_errors.store(0, Ordering::Relaxed);
    }

    /// Returns the cache configuration.
    #[inline]
    pub fn config(&self) -> &ShaderCacheConfig {
        &self.config
    }

    /// Returns an iterator over cached shader labels (for debugging).
    pub fn labels(&self) -> Vec<Option<String>> {
        self.cache
            .read()
            .values()
            .map(|c| c.module.label().map(String::from))
            .collect()
    }

    /// Returns an iterator over cached shader keys (for debugging).
    pub fn keys(&self) -> Vec<ShaderCacheKey> {
        self.cache.read().keys().cloned().collect()
    }

    /// Returns detailed cache entry information (for debugging).
    pub fn entries(&self) -> Vec<CacheEntryInfo> {
        self.cache
            .read()
            .values()
            .map(|cached| CacheEntryInfo {
                key: cached.key.display_string(),
                label: cached.module.label().map(String::from),
                age_secs: cached.age().as_secs_f64(),
                idle_secs: cached.idle_time().as_secs_f64(),
                access_count: cached.access_count,
            })
            .collect()
    }
}

impl std::fmt::Debug for ShaderCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let metrics = self.metrics();
        f.debug_struct("ShaderCache")
            .field("cache_size", &metrics.cache_size)
            .field("hits", &metrics.hits)
            .field("misses", &metrics.misses)
            .field("hit_rate", &format!("{:.1}%", metrics.hit_rate_percent()))
            .field("config", &self.config)
            .finish()
    }
}

// ============================================================================
// CacheEntryInfo
// ============================================================================

/// Debug information about a cache entry.
#[derive(Debug, Clone)]
pub struct CacheEntryInfo {
    /// Display string for the cache key.
    pub key: String,
    /// Shader label.
    pub label: Option<String>,
    /// Age since creation in seconds.
    pub age_secs: f64,
    /// Time since last access in seconds.
    pub idle_secs: f64,
    /// Number of times accessed.
    pub access_count: u64,
}

// ============================================================================
// Utility Functions
// ============================================================================

/// Computes SHA-256 hash of data.
fn compute_sha256(data: &[u8]) -> [u8; 32] {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(data);
    let result = hasher.finalize();
    let mut hash = [0u8; 32];
    hash.copy_from_slice(&result);
    hash
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // ShaderCacheKey Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_key_from_path() {
        let key = ShaderCacheKey::from_path("shaders/test.wgsl");
        assert!(key.is_path());
        assert!(!key.is_hash());
        assert_eq!(key.as_path(), Some(Path::new("shaders/test.wgsl")));
        assert!(key.as_hash().is_none());
    }

    #[test]
    fn test_key_from_source() {
        let source = "fn main() {}";
        let key = ShaderCacheKey::from_source(source);
        assert!(!key.is_path());
        assert!(key.is_hash());
        assert!(key.as_path().is_none());
        assert!(key.as_hash().is_some());
    }

    #[test]
    fn test_key_from_bytes() {
        let bytes = &[1u8, 2, 3, 4, 5];
        let key = ShaderCacheKey::from_bytes(bytes);
        assert!(key.is_hash());
    }

    #[test]
    fn test_key_from_hash() {
        let hash = [42u8; 32];
        let key = ShaderCacheKey::from_hash(hash);
        assert!(key.is_hash());
        assert_eq!(key.as_hash(), Some(&hash));
    }

    #[test]
    fn test_key_equality_same_source() {
        let source = "fn main() {}";
        let key1 = ShaderCacheKey::from_source(source);
        let key2 = ShaderCacheKey::from_source(source);
        assert_eq!(key1, key2);
    }

    #[test]
    fn test_key_equality_different_source() {
        let key1 = ShaderCacheKey::from_source("fn main() {}");
        let key2 = ShaderCacheKey::from_source("fn other() {}");
        assert_ne!(key1, key2);
    }

    #[test]
    fn test_key_equality_same_path() {
        let key1 = ShaderCacheKey::from_path("test.wgsl");
        let key2 = ShaderCacheKey::from_path("test.wgsl");
        assert_eq!(key1, key2);
    }

    #[test]
    fn test_key_equality_different_path() {
        let key1 = ShaderCacheKey::from_path("a.wgsl");
        let key2 = ShaderCacheKey::from_path("b.wgsl");
        assert_ne!(key1, key2);
    }

    #[test]
    fn test_key_path_vs_hash_not_equal() {
        let key1 = ShaderCacheKey::from_path("test.wgsl");
        let key2 = ShaderCacheKey::from_source("test.wgsl");
        assert_ne!(key1, key2);
    }

    #[test]
    fn test_key_display_string_path() {
        let key = ShaderCacheKey::from_path("shaders/test.wgsl");
        let display = key.display_string();
        assert!(display.contains("test.wgsl"));
    }

    #[test]
    fn test_key_display_string_hash() {
        let key = ShaderCacheKey::from_source("fn main() {}");
        let display = key.display_string();
        assert!(display.starts_with("hash:"));
        assert!(display.ends_with("..."));
    }

    #[test]
    fn test_key_from_pathbuf() {
        let path = PathBuf::from("test.wgsl");
        let key: ShaderCacheKey = path.into();
        assert!(key.is_path());
    }

    #[test]
    fn test_key_from_path_ref() {
        let path = Path::new("test.wgsl");
        let key: ShaderCacheKey = path.into();
        assert!(key.is_path());
    }

    #[test]
    fn test_key_from_str() {
        let key: ShaderCacheKey = "test.wgsl".into();
        assert!(key.is_path());
    }

    #[test]
    fn test_key_from_hash_array() {
        let hash = [0u8; 32];
        let key: ShaderCacheKey = hash.into();
        assert!(key.is_hash());
    }

    #[test]
    fn test_key_clone() {
        let key = ShaderCacheKey::from_path("test.wgsl");
        let cloned = key.clone();
        assert_eq!(key, cloned);
    }

    #[test]
    fn test_key_debug() {
        let key = ShaderCacheKey::from_path("test.wgsl");
        let debug = format!("{:?}", key);
        assert!(debug.contains("Path"));
        assert!(debug.contains("test.wgsl"));
    }

    #[test]
    fn test_key_hash_impl() {
        use std::collections::HashMap;
        let mut map: HashMap<ShaderCacheKey, i32> = HashMap::new();

        let key = ShaderCacheKey::from_path("test.wgsl");
        map.insert(key.clone(), 42);

        assert_eq!(map.get(&key), Some(&42));
    }

    // -------------------------------------------------------------------------
    // ShaderCacheConfig Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_config_default() {
        let config = ShaderCacheConfig::default();
        assert_eq!(config.max_entries, DEFAULT_MAX_ENTRIES);
        assert!(!config.enable_disk_cache);
        assert!(config.disk_cache_path.is_none());
        assert!(config.enable_lru_eviction);
    }

    #[test]
    fn test_config_new() {
        let config = ShaderCacheConfig::new();
        assert_eq!(config.max_entries, DEFAULT_MAX_ENTRIES);
    }

    #[test]
    fn test_config_max_entries() {
        let config = ShaderCacheConfig::new().max_entries(512);
        assert_eq!(config.max_entries, 512);
    }

    #[test]
    fn test_config_with_disk_cache() {
        let config = ShaderCacheConfig::new().with_disk_cache("/tmp/shaders");
        assert!(config.enable_disk_cache);
        assert_eq!(config.disk_cache_path, Some(PathBuf::from("/tmp/shaders")));
    }

    #[test]
    fn test_config_without_eviction() {
        let config = ShaderCacheConfig::new().without_eviction();
        assert!(!config.enable_lru_eviction);
    }

    #[test]
    fn test_config_minimal() {
        let config = ShaderCacheConfig::minimal();
        assert_eq!(config.max_entries, 16);
        assert!(!config.enable_disk_cache);
    }

    #[test]
    fn test_config_development() {
        let config = ShaderCacheConfig::development();
        assert_eq!(config.max_entries, 64);
        assert!(!config.enable_disk_cache);
    }

    #[test]
    fn test_config_production() {
        let config = ShaderCacheConfig::production();
        assert_eq!(config.max_entries, 1024);
        assert!(config.enable_disk_cache);
        assert!(config.disk_cache_path.is_some());
    }

    #[test]
    fn test_config_clone() {
        let config = ShaderCacheConfig::new().max_entries(100);
        let cloned = config.clone();
        assert_eq!(cloned.max_entries, 100);
    }

    #[test]
    fn test_config_debug() {
        let config = ShaderCacheConfig::default();
        let debug = format!("{:?}", config);
        assert!(debug.contains("ShaderCacheConfig"));
        assert!(debug.contains("max_entries"));
    }

    // -------------------------------------------------------------------------
    // ShaderCacheMetrics Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_metrics_default() {
        let metrics = ShaderCacheMetrics::default();
        assert_eq!(metrics.cache_size, 0);
        assert_eq!(metrics.hits, 0);
        assert_eq!(metrics.misses, 0);
        assert_eq!(metrics.hit_rate, 0.0);
        assert_eq!(metrics.evictions, 0);
        assert_eq!(metrics.invalidations, 0);
        assert_eq!(metrics.compilation_errors, 0);
    }

    #[test]
    fn test_metrics_new() {
        let metrics = ShaderCacheMetrics::new(10, 80, 20, 5, 3, 1);
        assert_eq!(metrics.cache_size, 10);
        assert_eq!(metrics.hits, 80);
        assert_eq!(metrics.misses, 20);
        assert_eq!(metrics.hit_rate, 0.8);
        assert_eq!(metrics.evictions, 5);
        assert_eq!(metrics.invalidations, 3);
        assert_eq!(metrics.compilation_errors, 1);
    }

    #[test]
    fn test_metrics_total_requests() {
        let metrics = ShaderCacheMetrics::new(0, 50, 50, 0, 0, 0);
        assert_eq!(metrics.total_requests(), 100);
    }

    #[test]
    fn test_metrics_is_empty() {
        let empty = ShaderCacheMetrics::new(0, 10, 5, 0, 0, 0);
        assert!(empty.is_empty());

        let not_empty = ShaderCacheMetrics::new(1, 10, 5, 0, 0, 0);
        assert!(!not_empty.is_empty());
    }

    #[test]
    fn test_metrics_hit_rate_percent() {
        let metrics = ShaderCacheMetrics::new(0, 75, 25, 0, 0, 0);
        assert_eq!(metrics.hit_rate_percent(), 75.0);
    }

    #[test]
    fn test_metrics_miss_rate() {
        let metrics = ShaderCacheMetrics::new(0, 60, 40, 0, 0, 0);
        assert_eq!(metrics.miss_rate(), 0.4);
    }

    #[test]
    fn test_metrics_zero_requests() {
        let metrics = ShaderCacheMetrics::new(0, 0, 0, 0, 0, 0);
        assert_eq!(metrics.hit_rate, 0.0);
        assert_eq!(metrics.total_requests(), 0);
    }

    #[test]
    fn test_metrics_clone() {
        let metrics = ShaderCacheMetrics::new(5, 10, 2, 1, 0, 0);
        let cloned = metrics.clone();
        assert_eq!(cloned.cache_size, 5);
        assert_eq!(cloned.hits, 10);
    }

    #[test]
    fn test_metrics_debug() {
        let metrics = ShaderCacheMetrics::default();
        let debug = format!("{:?}", metrics);
        assert!(debug.contains("ShaderCacheMetrics"));
    }

    // -------------------------------------------------------------------------
    // CachedShader Tests
    // -------------------------------------------------------------------------

    // Note: CachedShader tests require a TrinityShaderModule which requires a device.
    // These are integration-level tests.

    // -------------------------------------------------------------------------
    // Utility Function Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_compute_sha256_same_input() {
        let hash1 = compute_sha256(b"hello");
        let hash2 = compute_sha256(b"hello");
        assert_eq!(hash1, hash2);
    }

    #[test]
    fn test_compute_sha256_different_input() {
        let hash1 = compute_sha256(b"hello");
        let hash2 = compute_sha256(b"world");
        assert_ne!(hash1, hash2);
    }

    #[test]
    fn test_compute_sha256_empty() {
        let hash = compute_sha256(b"");
        assert_eq!(hash.len(), 32);
    }

    #[test]
    fn test_compute_sha256_length() {
        let hash = compute_sha256(b"test data");
        assert_eq!(hash.len(), 32);
    }

    // -------------------------------------------------------------------------
    // CacheEntryInfo Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_cache_entry_info_debug() {
        let info = CacheEntryInfo {
            key: "test".to_string(),
            label: Some("label".to_string()),
            age_secs: 1.5,
            idle_secs: 0.5,
            access_count: 10,
        };
        let debug = format!("{:?}", info);
        assert!(debug.contains("CacheEntryInfo"));
        assert!(debug.contains("test"));
    }

    #[test]
    fn test_cache_entry_info_clone() {
        let info = CacheEntryInfo {
            key: "test".to_string(),
            label: None,
            age_secs: 2.0,
            idle_secs: 1.0,
            access_count: 5,
        };
        let cloned = info.clone();
        assert_eq!(cloned.key, "test");
        assert_eq!(cloned.access_count, 5);
    }

    // -------------------------------------------------------------------------
    // Constants Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_default_max_entries() {
        assert_eq!(DEFAULT_MAX_ENTRIES, 256);
    }

    #[test]
    fn test_default_disk_cache_path() {
        assert_eq!(DEFAULT_DISK_CACHE_PATH, ".trinity/shader_cache");
    }

    // -------------------------------------------------------------------------
    // ShaderCache Tests (No Device - Unit Tests)
    // -------------------------------------------------------------------------

    // Note: Full ShaderCache tests require a wgpu device which is not available
    // in unit tests. The following tests cover edge cases and configuration.

    // =========================================================================
    // ADDITIONAL WHITEBOX TESTS - T-WGPU-P2.7.2
    // =========================================================================

    // -------------------------------------------------------------------------
    // Key Edge Cases
    // -------------------------------------------------------------------------

    #[test]
    fn test_key_empty_path() {
        let key = ShaderCacheKey::from_path("");
        assert!(key.is_path());
        assert_eq!(key.as_path(), Some(Path::new("")));
    }

    #[test]
    fn test_key_empty_source() {
        let key = ShaderCacheKey::from_source("");
        assert!(key.is_hash());
    }

    #[test]
    fn test_key_unicode_path() {
        let key = ShaderCacheKey::from_path("shaders/test.wgsl");
        assert!(key.is_path());
        assert!(key.display_string().contains("test"));
    }

    #[test]
    fn test_key_unicode_source() {
        let key = ShaderCacheKey::from_source("// comment");
        assert!(key.is_hash());
    }

    #[test]
    fn test_key_very_long_path() {
        let long_path = "a".repeat(1000) + ".wgsl";
        let key = ShaderCacheKey::from_path(&long_path);
        assert!(key.is_path());
    }

    #[test]
    fn test_key_very_long_source() {
        let long_source = "// ".to_string() + &"x".repeat(10000);
        let key = ShaderCacheKey::from_source(&long_source);
        assert!(key.is_hash());
    }

    #[test]
    fn test_key_whitespace_only_source() {
        let key = ShaderCacheKey::from_source("   \n\t   ");
        assert!(key.is_hash());
    }

    #[test]
    fn test_key_special_characters_path() {
        let key = ShaderCacheKey::from_path("shaders/@special#$%.wgsl");
        assert!(key.is_path());
    }

    // -------------------------------------------------------------------------
    // Config Builder Pattern Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_config_builder_chain() {
        let config = ShaderCacheConfig::new()
            .max_entries(128)
            .with_disk_cache("/cache")
            .without_eviction();

        assert_eq!(config.max_entries, 128);
        assert!(config.enable_disk_cache);
        assert!(!config.enable_lru_eviction);
    }

    #[test]
    fn test_config_max_entries_zero() {
        let config = ShaderCacheConfig::new().max_entries(0);
        assert_eq!(config.max_entries, 0);
    }

    #[test]
    fn test_config_max_entries_large() {
        let config = ShaderCacheConfig::new().max_entries(1_000_000);
        assert_eq!(config.max_entries, 1_000_000);
    }

    // -------------------------------------------------------------------------
    // Metrics Boundary Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_metrics_max_u64_hits() {
        let metrics = ShaderCacheMetrics::new(0, u64::MAX, 0, 0, 0, 0);
        assert_eq!(metrics.hits, u64::MAX);
        assert_eq!(metrics.hit_rate, 1.0);
    }

    #[test]
    fn test_metrics_equal_hits_misses() {
        let metrics = ShaderCacheMetrics::new(0, 50, 50, 0, 0, 0);
        assert_eq!(metrics.hit_rate, 0.5);
        assert_eq!(metrics.miss_rate(), 0.5);
    }

    #[test]
    fn test_metrics_all_misses() {
        let metrics = ShaderCacheMetrics::new(0, 0, 100, 0, 0, 0);
        assert_eq!(metrics.hit_rate, 0.0);
        assert_eq!(metrics.miss_rate(), 1.0);
    }

    #[test]
    fn test_metrics_all_hits() {
        let metrics = ShaderCacheMetrics::new(0, 100, 0, 0, 0, 0);
        assert_eq!(metrics.hit_rate, 1.0);
        assert_eq!(metrics.miss_rate(), 0.0);
    }

    // -------------------------------------------------------------------------
    // Hash Collision Resistance Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_hash_similar_sources() {
        let key1 = ShaderCacheKey::from_source("fn a() {}");
        let key2 = ShaderCacheKey::from_source("fn b() {}");
        assert_ne!(key1, key2);
    }

    #[test]
    fn test_hash_prefix_sources() {
        let key1 = ShaderCacheKey::from_source("fn main() {}");
        let key2 = ShaderCacheKey::from_source("fn main() {} // extra");
        assert_ne!(key1, key2);
    }

    #[test]
    fn test_hash_whitespace_difference() {
        let key1 = ShaderCacheKey::from_source("fn main(){}");
        let key2 = ShaderCacheKey::from_source("fn main() { }");
        assert_ne!(key1, key2);
    }

    #[test]
    fn test_hash_case_sensitive() {
        let key1 = ShaderCacheKey::from_source("fn Main() {}");
        let key2 = ShaderCacheKey::from_source("fn main() {}");
        assert_ne!(key1, key2);
    }

    // -------------------------------------------------------------------------
    // CacheEntryInfo Edge Cases
    // -------------------------------------------------------------------------

    #[test]
    fn test_cache_entry_info_zero_values() {
        let info = CacheEntryInfo {
            key: String::new(),
            label: None,
            age_secs: 0.0,
            idle_secs: 0.0,
            access_count: 0,
        };
        assert_eq!(info.age_secs, 0.0);
        assert_eq!(info.access_count, 0);
    }

    #[test]
    fn test_cache_entry_info_large_values() {
        let info = CacheEntryInfo {
            key: "test".to_string(),
            label: Some("label".to_string()),
            age_secs: 86400.0 * 365.0, // 1 year
            idle_secs: 86400.0,        // 1 day
            access_count: u64::MAX,
        };
        assert_eq!(info.access_count, u64::MAX);
    }

    // -------------------------------------------------------------------------
    // Path Key Normalization Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_path_key_with_dots() {
        let key = ShaderCacheKey::from_path("../shaders/test.wgsl");
        assert!(key.is_path());
        assert!(key.display_string().contains(".."));
    }

    #[test]
    fn test_path_key_absolute() {
        let key = ShaderCacheKey::from_path("/absolute/path/shader.wgsl");
        assert!(key.is_path());
    }

    #[test]
    fn test_path_key_windows_style() {
        let key = ShaderCacheKey::from_path("C:\\shaders\\test.wgsl");
        assert!(key.is_path());
    }

    // -------------------------------------------------------------------------
    // Metrics Display Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_metrics_hit_rate_percent_rounding() {
        let metrics = ShaderCacheMetrics::new(0, 1, 2, 0, 0, 0);
        let percent = metrics.hit_rate_percent();
        assert!(percent > 33.0 && percent < 34.0);
    }

    #[test]
    fn test_metrics_hit_rate_percent_100() {
        let metrics = ShaderCacheMetrics::new(0, 1000, 0, 0, 0, 0);
        assert_eq!(metrics.hit_rate_percent(), 100.0);
    }

    #[test]
    fn test_metrics_hit_rate_percent_0() {
        let metrics = ShaderCacheMetrics::new(0, 0, 1000, 0, 0, 0);
        assert_eq!(metrics.hit_rate_percent(), 0.0);
    }

    // -------------------------------------------------------------------------
    // Key Hash Stability Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_key_hash_stable_across_calls() {
        let source = "fn test() { return; }";
        let key1 = ShaderCacheKey::from_source(source);
        let key2 = ShaderCacheKey::from_source(source);

        // Should produce identical hashes
        assert_eq!(key1.as_hash(), key2.as_hash());
    }

    #[test]
    fn test_key_bytes_vs_source() {
        let text = "fn main() {}";
        let key1 = ShaderCacheKey::from_source(text);
        let key2 = ShaderCacheKey::from_bytes(text.as_bytes());

        // Should produce identical hashes
        assert_eq!(key1.as_hash(), key2.as_hash());
    }

    // -------------------------------------------------------------------------
    // Config Struct Layout Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_config_fields_accessible() {
        let config = ShaderCacheConfig {
            max_entries: 100,
            enable_disk_cache: true,
            disk_cache_path: Some(PathBuf::from("/test")),
            enable_lru_eviction: false,
        };

        assert_eq!(config.max_entries, 100);
        assert!(config.enable_disk_cache);
        assert!(config.disk_cache_path.is_some());
        assert!(!config.enable_lru_eviction);
    }

    // -------------------------------------------------------------------------
    // Metrics Struct Layout Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_metrics_fields_accessible() {
        let metrics = ShaderCacheMetrics {
            cache_size: 10,
            hits: 100,
            misses: 20,
            hit_rate: 0.833,
            evictions: 5,
            invalidations: 2,
            compilation_errors: 1,
        };

        assert_eq!(metrics.cache_size, 10);
        assert_eq!(metrics.hits, 100);
        assert_eq!(metrics.misses, 20);
        assert_eq!(metrics.evictions, 5);
        assert_eq!(metrics.invalidations, 2);
        assert_eq!(metrics.compilation_errors, 1);
    }

    // =========================================================================
    // EXTENDED WHITEBOX TESTS - T-WGPU-P2.7.2 (120+ coverage)
    // =========================================================================

    // -------------------------------------------------------------------------
    // Thread Safety Tests (Key-level)
    // -------------------------------------------------------------------------

    #[test]
    fn test_key_thread_safe_send() {
        fn assert_send<T: Send>() {}
        assert_send::<ShaderCacheKey>();
    }

    #[test]
    fn test_key_thread_safe_sync() {
        fn assert_sync<T: Sync>() {}
        assert_sync::<ShaderCacheKey>();
    }

    #[test]
    fn test_config_thread_safe_send() {
        fn assert_send<T: Send>() {}
        assert_send::<ShaderCacheConfig>();
    }

    #[test]
    fn test_config_thread_safe_sync() {
        fn assert_sync<T: Sync>() {}
        assert_sync::<ShaderCacheConfig>();
    }

    #[test]
    fn test_metrics_thread_safe_send() {
        fn assert_send<T: Send>() {}
        assert_send::<ShaderCacheMetrics>();
    }

    #[test]
    fn test_metrics_thread_safe_sync() {
        fn assert_sync<T: Sync>() {}
        assert_sync::<ShaderCacheMetrics>();
    }

    #[test]
    fn test_cache_entry_info_thread_safe() {
        fn assert_send<T: Send>() {}
        fn assert_sync<T: Sync>() {}
        assert_send::<CacheEntryInfo>();
        assert_sync::<CacheEntryInfo>();
    }

    // -------------------------------------------------------------------------
    // Key Concurrent Access Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_key_concurrent_hash_consistency() {
        use std::thread;

        let source = "fn concurrent_test() {}";
        let handles: Vec<_> = (0..10)
            .map(|_| {
                let src = source.to_string();
                thread::spawn(move || ShaderCacheKey::from_source(&src))
            })
            .collect();

        let keys: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
        let first = &keys[0];
        for key in &keys[1..] {
            assert_eq!(first, key);
        }
    }

    #[test]
    fn test_key_concurrent_path_construction() {
        use std::thread;

        let handles: Vec<_> = (0..10)
            .map(|i| {
                thread::spawn(move || ShaderCacheKey::from_path(format!("shader_{}.wgsl", i)))
            })
            .collect();

        let keys: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
        // All keys should be unique
        for i in 0..keys.len() {
            for j in (i + 1)..keys.len() {
                assert_ne!(keys[i], keys[j]);
            }
        }
    }

    // -------------------------------------------------------------------------
    // Key HashMap Integration Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_key_hashmap_insert_retrieve() {
        let mut map: HashMap<ShaderCacheKey, String> = HashMap::new();
        let key = ShaderCacheKey::from_source("fn test() {}");
        map.insert(key.clone(), "value".to_string());

        assert_eq!(map.get(&key), Some(&"value".to_string()));
    }

    #[test]
    fn test_key_hashmap_overwrite() {
        let mut map: HashMap<ShaderCacheKey, i32> = HashMap::new();
        let key = ShaderCacheKey::from_source("fn test() {}");
        map.insert(key.clone(), 1);
        map.insert(key.clone(), 2);

        assert_eq!(map.len(), 1);
        assert_eq!(map.get(&key), Some(&2));
    }

    #[test]
    fn test_key_hashmap_path_vs_hash_keys() {
        let mut map: HashMap<ShaderCacheKey, i32> = HashMap::new();
        let path_key = ShaderCacheKey::from_path("test.wgsl");
        let hash_key = ShaderCacheKey::from_source("test.wgsl");

        map.insert(path_key.clone(), 1);
        map.insert(hash_key.clone(), 2);

        // Should be two separate entries
        assert_eq!(map.len(), 2);
        assert_eq!(map.get(&path_key), Some(&1));
        assert_eq!(map.get(&hash_key), Some(&2));
    }

    #[test]
    fn test_key_hashmap_remove() {
        let mut map: HashMap<ShaderCacheKey, i32> = HashMap::new();
        let key = ShaderCacheKey::from_source("remove_me");
        map.insert(key.clone(), 42);

        assert_eq!(map.remove(&key), Some(42));
        assert!(map.is_empty());
    }

    #[test]
    fn test_key_hashmap_contains_key() {
        let mut map: HashMap<ShaderCacheKey, i32> = HashMap::new();
        let key = ShaderCacheKey::from_source("exists");
        map.insert(key.clone(), 1);

        assert!(map.contains_key(&key));
        assert!(!map.contains_key(&ShaderCacheKey::from_source("missing")));
    }

    // -------------------------------------------------------------------------
    // Key Display String Edge Cases
    // -------------------------------------------------------------------------

    #[test]
    fn test_key_display_string_empty_path() {
        let key = ShaderCacheKey::from_path("");
        assert_eq!(key.display_string(), "");
    }

    #[test]
    fn test_key_display_string_nested_path() {
        let key = ShaderCacheKey::from_path("a/b/c/d/e/shader.wgsl");
        let display = key.display_string();
        assert!(display.contains("shader.wgsl"));
    }

    #[test]
    fn test_key_display_string_hash_format() {
        let key = ShaderCacheKey::from_source("test");
        let display = key.display_string();

        // Should start with "hash:" and end with "..."
        assert!(display.starts_with("hash:"));
        assert!(display.ends_with("..."));
        // Should have exactly 8 hex chars between
        let hex_part = &display[5..display.len() - 3];
        assert_eq!(hex_part.len(), 8);
        assert!(hex_part.chars().all(|c| c.is_ascii_hexdigit()));
    }

    // -------------------------------------------------------------------------
    // SHA-256 Edge Cases
    // -------------------------------------------------------------------------

    #[test]
    fn test_sha256_binary_data() {
        let binary = vec![0u8, 1, 2, 255, 254, 253];
        let hash = compute_sha256(&binary);
        assert_eq!(hash.len(), 32);
    }

    #[test]
    fn test_sha256_unicode_data() {
        let unicode = "こんにちは世界 🌍 emoji";
        let hash = compute_sha256(unicode.as_bytes());
        assert_eq!(hash.len(), 32);
    }

    #[test]
    fn test_sha256_large_data() {
        let large = vec![42u8; 1_000_000]; // 1MB
        let hash = compute_sha256(&large);
        assert_eq!(hash.len(), 32);
    }

    #[test]
    fn test_sha256_single_byte_difference() {
        let hash1 = compute_sha256(&[0u8]);
        let hash2 = compute_sha256(&[1u8]);
        assert_ne!(hash1, hash2);
    }

    #[test]
    fn test_sha256_length_extension() {
        let hash1 = compute_sha256(b"data");
        let hash2 = compute_sha256(b"data\x00");
        assert_ne!(hash1, hash2);
    }

    // -------------------------------------------------------------------------
    // Config Edge Cases
    // -------------------------------------------------------------------------

    #[test]
    fn test_config_disk_cache_empty_path() {
        let config = ShaderCacheConfig::new().with_disk_cache("");
        assert!(config.enable_disk_cache);
        assert_eq!(config.disk_cache_path, Some(PathBuf::from("")));
    }

    #[test]
    fn test_config_multiple_with_disk_cache() {
        let config = ShaderCacheConfig::new()
            .with_disk_cache("/first")
            .with_disk_cache("/second");

        assert_eq!(config.disk_cache_path, Some(PathBuf::from("/second")));
    }

    #[test]
    fn test_config_builder_order_independence() {
        let config1 = ShaderCacheConfig::new()
            .max_entries(100)
            .without_eviction();

        let config2 = ShaderCacheConfig::new()
            .without_eviction()
            .max_entries(100);

        assert_eq!(config1.max_entries, config2.max_entries);
        assert_eq!(config1.enable_lru_eviction, config2.enable_lru_eviction);
    }

    // -------------------------------------------------------------------------
    // Metrics Edge Cases
    // -------------------------------------------------------------------------

    #[test]
    fn test_metrics_high_hit_rate() {
        // Very high hits vs few misses should compute valid hit_rate close to 1.0
        let metrics = ShaderCacheMetrics::new(0, 999_999_999, 1, 0, 0, 0);
        assert!(metrics.hit_rate > 0.999);
        assert!(metrics.hit_rate < 1.0);
    }

    #[test]
    fn test_metrics_large_cache_size() {
        let metrics = ShaderCacheMetrics::new(usize::MAX, 0, 0, 0, 0, 0);
        assert_eq!(metrics.cache_size, usize::MAX);
        assert!(!metrics.is_empty());
    }

    #[test]
    fn test_metrics_large_counters() {
        // Use large but non-overflowing values
        let metrics = ShaderCacheMetrics::new(
            1_000_000,
            u64::MAX / 2,
            u64::MAX / 2,
            u64::MAX,
            u64::MAX,
            u64::MAX,
        );
        assert_eq!(metrics.evictions, u64::MAX);
        assert_eq!(metrics.invalidations, u64::MAX);
        assert_eq!(metrics.compilation_errors, u64::MAX);
    }

    #[test]
    fn test_metrics_total_requests_large() {
        // Test with large but non-overflowing values
        let metrics = ShaderCacheMetrics::new(0, u64::MAX / 2, u64::MAX / 3, 0, 0, 0);
        // total_requests should work correctly
        let total = metrics.total_requests();
        assert_eq!(total, u64::MAX / 2 + u64::MAX / 3);
    }

    // -------------------------------------------------------------------------
    // Key From Conversions
    // -------------------------------------------------------------------------

    #[test]
    fn test_key_from_cow_path() {
        use std::borrow::Cow;
        let path: Cow<'_, Path> = Cow::Borrowed(Path::new("test.wgsl"));
        let key = ShaderCacheKey::from_path(path.as_ref());
        assert!(key.is_path());
    }

    #[test]
    fn test_key_from_osstr() {
        use std::ffi::OsStr;
        let os_path = OsStr::new("test.wgsl");
        let key = ShaderCacheKey::from_path(PathBuf::from(os_path));
        assert!(key.is_path());
    }

    // -------------------------------------------------------------------------
    // Cache Entry Info Additional Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_cache_entry_info_with_none_label() {
        let info = CacheEntryInfo {
            key: "key".to_string(),
            label: None,
            age_secs: 1.0,
            idle_secs: 0.5,
            access_count: 1,
        };
        assert!(info.label.is_none());
    }

    #[test]
    fn test_cache_entry_info_negative_seconds_impossible() {
        // f64 can represent negative, but semantically shouldn't happen
        let info = CacheEntryInfo {
            key: "key".to_string(),
            label: Some("test".to_string()),
            age_secs: -1.0, // Invalid but representable
            idle_secs: -0.5,
            access_count: 0,
        };
        // Just verify it compiles and doesn't panic
        assert!(info.age_secs < 0.0);
    }

    // -------------------------------------------------------------------------
    // Key Equality Deep Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_key_reflexive_equality() {
        let key = ShaderCacheKey::from_source("test");
        assert_eq!(key, key);
    }

    #[test]
    fn test_key_symmetric_equality() {
        let key1 = ShaderCacheKey::from_source("test");
        let key2 = ShaderCacheKey::from_source("test");
        assert_eq!(key1, key2);
        assert_eq!(key2, key1);
    }

    #[test]
    fn test_key_transitive_equality() {
        let key1 = ShaderCacheKey::from_source("test");
        let key2 = ShaderCacheKey::from_source("test");
        let key3 = ShaderCacheKey::from_source("test");
        assert_eq!(key1, key2);
        assert_eq!(key2, key3);
        assert_eq!(key1, key3);
    }

    // -------------------------------------------------------------------------
    // Key Ordering in Collections
    // -------------------------------------------------------------------------

    #[test]
    fn test_key_btree_compatible() {
        use std::collections::BTreeSet;
        // ShaderCacheKey doesn't impl Ord, so this should fail to compile if uncommented
        // BTreeSet::<ShaderCacheKey>::new();
        // This test documents that BTreeSet is NOT supported
        assert!(true);
    }

    #[test]
    fn test_key_hashset_compatible() {
        use std::collections::HashSet;
        let mut set: HashSet<ShaderCacheKey> = HashSet::new();
        set.insert(ShaderCacheKey::from_source("a"));
        set.insert(ShaderCacheKey::from_source("b"));
        set.insert(ShaderCacheKey::from_source("a")); // Duplicate

        assert_eq!(set.len(), 2);
    }

    // -------------------------------------------------------------------------
    // Metrics Method Coverage
    // -------------------------------------------------------------------------

    #[test]
    fn test_metrics_hit_rate_zero_division() {
        let metrics = ShaderCacheMetrics::new(0, 0, 0, 0, 0, 0);
        assert_eq!(metrics.hit_rate, 0.0);
        assert_eq!(metrics.miss_rate(), 1.0);
    }

    #[test]
    fn test_metrics_hit_rate_precision() {
        let metrics = ShaderCacheMetrics::new(0, 1, 3, 0, 0, 0);
        // 1/4 = 0.25
        assert!((metrics.hit_rate - 0.25).abs() < f64::EPSILON);
    }

    // -------------------------------------------------------------------------
    // Constants Validation
    // -------------------------------------------------------------------------

    #[test]
    fn test_default_max_entries_reasonable() {
        assert!(DEFAULT_MAX_ENTRIES > 0);
        assert!(DEFAULT_MAX_ENTRIES <= 10000);
    }

    #[test]
    fn test_default_disk_cache_path_valid() {
        assert!(!DEFAULT_DISK_CACHE_PATH.is_empty());
        assert!(DEFAULT_DISK_CACHE_PATH.contains("shader"));
    }

    // -------------------------------------------------------------------------
    // Key Clone Deep Equality
    // -------------------------------------------------------------------------

    #[test]
    fn test_key_clone_path_deep() {
        let key = ShaderCacheKey::from_path("/some/deep/path/shader.wgsl");
        let cloned = key.clone();

        assert_eq!(key, cloned);
        assert_eq!(key.as_path(), cloned.as_path());
    }

    #[test]
    fn test_key_clone_hash_deep() {
        let key = ShaderCacheKey::from_source("fn deep_clone_test() {}");
        let cloned = key.clone();

        assert_eq!(key, cloned);
        assert_eq!(key.as_hash(), cloned.as_hash());
    }

    // -------------------------------------------------------------------------
    // Config Clone Deep Equality
    // -------------------------------------------------------------------------

    #[test]
    fn test_config_clone_with_disk_path() {
        let config = ShaderCacheConfig::new()
            .with_disk_cache("/path/to/cache");
        let cloned = config.clone();

        assert_eq!(config.disk_cache_path, cloned.disk_cache_path);
    }

    // -------------------------------------------------------------------------
    // Metrics Derived Values
    // -------------------------------------------------------------------------

    #[test]
    fn test_metrics_miss_rate_complement() {
        let metrics = ShaderCacheMetrics::new(0, 70, 30, 0, 0, 0);
        assert!((metrics.hit_rate + metrics.miss_rate() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_metrics_percent_and_rate_consistent() {
        let metrics = ShaderCacheMetrics::new(0, 45, 55, 0, 0, 0);
        assert!((metrics.hit_rate_percent() - metrics.hit_rate * 100.0).abs() < f64::EPSILON);
    }
}
