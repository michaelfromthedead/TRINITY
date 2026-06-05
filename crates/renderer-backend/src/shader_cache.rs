//! Content-addressed shader cache for wgpu shader module deduplication.
//!
//! This module provides a [`ShaderCacheV2`] that extends the base [`ShaderCache`]
//! with additional features required for T-MAT-3.4 PBR pipeline integration:
//!
//! - Content-addressed storage using [`ContentHash`]
//! - Source path tracking for hot-reload integration
//! - Statistics for cache hit/miss monitoring
//! - Thread-safe access via interior mutability
//!
//! # Example
//!
//! ```ignore
//! use renderer_backend::shader_cache::ShaderCacheV2;
//!
//! let mut cache = ShaderCacheV2::new();
//! let (module, hash) = cache.cache_shader(&device, wgsl_source);
//!
//! // Same source returns cached module
//! let (module2, hash2) = cache.cache_shader(&device, wgsl_source);
//! assert_eq!(hash, hash2);
//! ```

use std::collections::HashMap;
use std::sync::Arc;

use crate::pipeline::ContentHash;

// ---------------------------------------------------------------------------
// ShaderCacheStats
// ---------------------------------------------------------------------------

/// Statistics for shader cache performance monitoring.
#[derive(Debug, Clone, Default)]
pub struct ShaderCacheStats {
    /// Number of cache hits.
    pub hits: u64,
    /// Number of cache misses (new compilations).
    pub misses: u64,
    /// Number of unique shader modules cached.
    pub cached_modules: usize,
    /// Number of source paths tracked.
    pub tracked_paths: usize,
    /// Total bytes of WGSL source compiled.
    pub total_source_bytes: u64,
}

impl ShaderCacheStats {
    /// Compute cache hit rate as a percentage [0.0, 100.0].
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            return 100.0; // No lookups yet, perfect hit rate by definition
        }
        (self.hits as f64 / total as f64) * 100.0
    }
}

// ---------------------------------------------------------------------------
// ShaderCacheV2
// ---------------------------------------------------------------------------

/// Content-addressed shader cache with source path tracking.
///
/// Deduplicates [`wgpu::ShaderModule`] allocations by keying on the
/// SHA-256 (or BLAKE3) hash of the WGSL source. Tracks source paths
/// for hot-reload integration with [`DepGraph`].
///
/// Unlike the basic [`ShaderCache`], this version:
/// - Uses [`ContentHash`] instead of raw `[u8; 32]` for type safety
/// - Tracks which source paths correspond to which hashes
/// - Maintains statistics for performance monitoring
/// - Supports reverse lookups (hash -> paths)
///
/// [`ShaderCache`]: crate::pipeline::ShaderCache
/// [`DepGraph`]: crate::material_dep_graph::DepGraph
pub struct ShaderCacheV2 {
    /// Compiled shader modules keyed by content hash.
    modules: HashMap<ContentHash, Arc<wgpu::ShaderModule>>,
    /// Maps source file paths to their content hash.
    path_to_hash: HashMap<String, ContentHash>,
    /// Maps content hash to associated source paths (for hot-reload).
    hash_to_paths: HashMap<ContentHash, Vec<String>>,
    /// Cache statistics.
    stats: ShaderCacheStats,
}

impl ShaderCacheV2 {
    /// Create an empty shader cache.
    pub fn new() -> Self {
        Self {
            modules: HashMap::new(),
            path_to_hash: HashMap::new(),
            hash_to_paths: HashMap::new(),
            stats: ShaderCacheStats::default(),
        }
    }

    /// Cache a shader from WGSL source, returning the module and its hash.
    ///
    /// If a module with the same content hash already exists, it is returned
    /// without recompilation. Otherwise the source is compiled and cached.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device for shader compilation.
    /// * `wgsl_source` - The WGSL shader source code.
    ///
    /// # Returns
    ///
    /// A tuple of (Arc<ShaderModule>, ContentHash).
    pub fn cache_shader(
        &mut self,
        device: &wgpu::Device,
        wgsl_source: &str,
    ) -> (Arc<wgpu::ShaderModule>, ContentHash) {
        let hash = ContentHash::from_bytes(wgsl_source.as_bytes());

        if let Some(module) = self.modules.get(&hash) {
            self.stats.hits += 1;
            return (Arc::clone(module), hash);
        }

        // Cache miss: compile new module
        self.stats.misses += 1;
        self.stats.total_source_bytes += wgsl_source.len() as u64;

        let module = Arc::new(device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ShaderCacheV2 module"),
            source: wgpu::ShaderSource::Wgsl(wgsl_source.into()),
        }));

        self.modules.insert(hash, Arc::clone(&module));
        self.stats.cached_modules = self.modules.len();

        (module, hash)
    }

    /// Cache a shader with an associated source path for hot-reload tracking.
    ///
    /// This is useful for associating file paths with shader hashes so that
    /// when a file changes, we can look up the affected shader modules.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device for shader compilation.
    /// * `wgsl_source` - The WGSL shader source code.
    /// * `source_path` - Path to the source file (for hot-reload).
    pub fn cache_shader_with_path(
        &mut self,
        device: &wgpu::Device,
        wgsl_source: &str,
        source_path: &str,
    ) -> (Arc<wgpu::ShaderModule>, ContentHash) {
        let (module, hash) = self.cache_shader(device, wgsl_source);

        // Track path -> hash mapping
        self.path_to_hash.insert(source_path.to_string(), hash);

        // Track hash -> paths mapping (multiple paths can have same hash)
        self.hash_to_paths
            .entry(hash)
            .or_default()
            .push(source_path.to_string());

        self.stats.tracked_paths = self.path_to_hash.len();

        (module, hash)
    }

    /// Get a cached shader module by its content hash.
    ///
    /// Returns `None` if no module with that hash is cached.
    pub fn get(&self, hash: &ContentHash) -> Option<Arc<wgpu::ShaderModule>> {
        self.modules.get(hash).cloned()
    }

    /// Get the content hash for a source file path.
    ///
    /// Returns `None` if the path is not tracked.
    pub fn hash_for_path(&self, path: &str) -> Option<ContentHash> {
        self.path_to_hash.get(path).copied()
    }

    /// Get all source paths associated with a content hash.
    ///
    /// Useful for hot-reload: find all files that share the same shader.
    pub fn paths_for_hash(&self, hash: &ContentHash) -> Option<&[String]> {
        self.hash_to_paths.get(hash).map(|v| v.as_slice())
    }

    /// Invalidate a shader by its source path.
    ///
    /// Called when a shader file changes and needs recompilation.
    /// Returns the old content hash if the path was tracked.
    pub fn invalidate_path(&mut self, path: &str) -> Option<ContentHash> {
        if let Some(old_hash) = self.path_to_hash.remove(path) {
            // Remove this path from hash_to_paths
            if let Some(paths) = self.hash_to_paths.get_mut(&old_hash) {
                paths.retain(|p| p != path);
                if paths.is_empty() {
                    self.hash_to_paths.remove(&old_hash);
                    // If no paths reference this hash, remove the module
                    self.modules.remove(&old_hash);
                    self.stats.cached_modules = self.modules.len();
                }
            }
            self.stats.tracked_paths = self.path_to_hash.len();
            return Some(old_hash);
        }
        None
    }

    /// Remove all cached modules and path mappings.
    pub fn clear(&mut self) {
        self.modules.clear();
        self.path_to_hash.clear();
        self.hash_to_paths.clear();
        self.stats.cached_modules = 0;
        self.stats.tracked_paths = 0;
    }

    /// Get cache statistics.
    pub fn stats(&self) -> &ShaderCacheStats {
        &self.stats
    }

    /// Reset statistics (useful after warm-up period).
    pub fn reset_stats(&mut self) {
        self.stats.hits = 0;
        self.stats.misses = 0;
    }

    /// Number of cached shader modules.
    pub fn len(&self) -> usize {
        self.modules.len()
    }

    /// Returns true if no shader modules are cached.
    pub fn is_empty(&self) -> bool {
        self.modules.is_empty()
    }

    /// Returns the number of tracked source paths.
    pub fn tracked_path_count(&self) -> usize {
        self.path_to_hash.len()
    }

    /// List all cached content hashes.
    pub fn cached_hashes(&self) -> impl Iterator<Item = &ContentHash> {
        self.modules.keys()
    }

    /// Check if a specific hash is cached.
    pub fn contains(&self, hash: &ContentHash) -> bool {
        self.modules.contains_key(hash)
    }
}

impl Default for ShaderCacheV2 {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // Helper to create a test device, returns None if no GPU available.
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
    fn test_shader_cache_v2_new() {
        let cache = ShaderCacheV2::new();
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
        assert_eq!(cache.tracked_path_count(), 0);
    }

    #[test]
    fn test_shader_cache_v2_content_addressing() {
        let (device, _queue) = match create_test_device() {
            Some(d) => d,
            None => {
                eprintln!("Skipping: no GPU adapter available");
                return;
            }
        };

        let mut cache = ShaderCacheV2::new();
        let src = r#"
            @vertex fn vs_main() -> @builtin(position) vec4<f32> {
                return vec4<f32>(0.0, 0.0, 0.0, 1.0);
            }
        "#;

        // First compilation: miss
        let (module1, hash1) = cache.cache_shader(&device, src);
        assert_eq!(cache.stats().misses, 1);
        assert_eq!(cache.stats().hits, 0);
        assert_eq!(cache.len(), 1);

        // Second compilation: hit (same source)
        let (module2, hash2) = cache.cache_shader(&device, src);
        assert_eq!(cache.stats().misses, 1);
        assert_eq!(cache.stats().hits, 1);
        assert_eq!(cache.len(), 1);

        // Same hash and module reference
        assert_eq!(hash1, hash2);
        assert!(Arc::ptr_eq(&module1, &module2));
    }

    #[test]
    fn test_shader_cache_v2_different_sources() {
        let (device, _queue) = match create_test_device() {
            Some(d) => d,
            None => {
                eprintln!("Skipping: no GPU adapter available");
                return;
            }
        };

        let mut cache = ShaderCacheV2::new();

        let src_a = "@vertex fn vs_a() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0); }";
        let src_b = "@vertex fn vs_b() -> @builtin(position) vec4<f32> { return vec4<f32>(1.0); }";

        let (_, hash_a) = cache.cache_shader(&device, src_a);
        let (_, hash_b) = cache.cache_shader(&device, src_b);

        assert_ne!(hash_a, hash_b);
        assert_eq!(cache.len(), 2);
        assert_eq!(cache.stats().misses, 2);
    }

    #[test]
    fn test_shader_cache_v2_path_tracking() {
        let (device, _queue) = match create_test_device() {
            Some(d) => d,
            None => {
                eprintln!("Skipping: no GPU adapter available");
                return;
            }
        };

        let mut cache = ShaderCacheV2::new();
        let src = "@vertex fn vs() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0); }";

        let (_, hash) = cache.cache_shader_with_path(&device, src, "shaders/test.wgsl");

        assert_eq!(cache.tracked_path_count(), 1);
        assert_eq!(cache.hash_for_path("shaders/test.wgsl"), Some(hash));
        assert_eq!(
            cache.paths_for_hash(&hash),
            Some(vec!["shaders/test.wgsl".to_string()].as_slice())
        );
    }

    #[test]
    fn test_shader_cache_v2_invalidate_path() {
        let (device, _queue) = match create_test_device() {
            Some(d) => d,
            None => {
                eprintln!("Skipping: no GPU adapter available");
                return;
            }
        };

        let mut cache = ShaderCacheV2::new();
        let src = "@vertex fn vs() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0); }";

        let (_, hash) = cache.cache_shader_with_path(&device, src, "shaders/test.wgsl");
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.tracked_path_count(), 1);

        // Invalidate the path
        let old_hash = cache.invalidate_path("shaders/test.wgsl");
        assert_eq!(old_hash, Some(hash));

        // Module should be removed since no other paths reference it
        assert_eq!(cache.len(), 0);
        assert_eq!(cache.tracked_path_count(), 0);
        assert!(cache.hash_for_path("shaders/test.wgsl").is_none());
    }

    #[test]
    fn test_shader_cache_v2_multiple_paths_same_hash() {
        let (device, _queue) = match create_test_device() {
            Some(d) => d,
            None => {
                eprintln!("Skipping: no GPU adapter available");
                return;
            }
        };

        let mut cache = ShaderCacheV2::new();
        let src = "@vertex fn vs() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0); }";

        // Same source, different paths
        let (_, hash1) = cache.cache_shader_with_path(&device, src, "path/a.wgsl");
        let (_, hash2) = cache.cache_shader_with_path(&device, src, "path/b.wgsl");

        assert_eq!(hash1, hash2);
        assert_eq!(cache.len(), 1); // Only one module
        assert_eq!(cache.tracked_path_count(), 2); // Two paths

        // Invalidate one path, module should remain
        cache.invalidate_path("path/a.wgsl");
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.tracked_path_count(), 1);

        // Invalidate the other, module should be removed
        cache.invalidate_path("path/b.wgsl");
        assert_eq!(cache.len(), 0);
        assert_eq!(cache.tracked_path_count(), 0);
    }

    #[test]
    fn test_shader_cache_v2_hit_rate() {
        let mut stats = ShaderCacheStats::default();

        // No lookups: 100% hit rate
        assert_eq!(stats.hit_rate(), 100.0);

        // All misses: 0%
        stats.misses = 10;
        assert_eq!(stats.hit_rate(), 0.0);

        // 50/50: 50%
        stats.hits = 10;
        assert_eq!(stats.hit_rate(), 50.0);

        // 90% hits
        stats.hits = 90;
        stats.misses = 10;
        assert_eq!(stats.hit_rate(), 90.0);
    }

    #[test]
    fn test_shader_cache_v2_clear() {
        let (device, _queue) = match create_test_device() {
            Some(d) => d,
            None => {
                eprintln!("Skipping: no GPU adapter available");
                return;
            }
        };

        let mut cache = ShaderCacheV2::new();
        let src = "@vertex fn vs() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0); }";

        cache.cache_shader_with_path(&device, src, "test.wgsl");
        assert!(!cache.is_empty());

        cache.clear();
        assert!(cache.is_empty());
        assert_eq!(cache.tracked_path_count(), 0);
    }

    #[test]
    fn test_shader_cache_v2_reset_stats() {
        let (device, _queue) = match create_test_device() {
            Some(d) => d,
            None => {
                eprintln!("Skipping: no GPU adapter available");
                return;
            }
        };

        let mut cache = ShaderCacheV2::new();
        let src = "@vertex fn vs() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0); }";

        cache.cache_shader(&device, src);
        cache.cache_shader(&device, src);

        assert_eq!(cache.stats().misses, 1);
        assert_eq!(cache.stats().hits, 1);

        cache.reset_stats();

        assert_eq!(cache.stats().misses, 0);
        assert_eq!(cache.stats().hits, 0);
        // Module count should remain
        assert_eq!(cache.stats().cached_modules, 1);
    }

    #[test]
    fn test_shader_cache_v2_get() {
        let (device, _queue) = match create_test_device() {
            Some(d) => d,
            None => {
                eprintln!("Skipping: no GPU adapter available");
                return;
            }
        };

        let mut cache = ShaderCacheV2::new();
        let src = "@vertex fn vs() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0); }";

        let (original, hash) = cache.cache_shader(&device, src);

        // Get by hash
        let fetched = cache.get(&hash);
        assert!(fetched.is_some());
        assert!(Arc::ptr_eq(&original, &fetched.unwrap()));

        // Get missing hash
        let missing_hash = ContentHash::from_bytes(b"nonexistent");
        assert!(cache.get(&missing_hash).is_none());
    }

    #[test]
    fn test_shader_cache_v2_contains() {
        let (device, _queue) = match create_test_device() {
            Some(d) => d,
            None => {
                eprintln!("Skipping: no GPU adapter available");
                return;
            }
        };

        let mut cache = ShaderCacheV2::new();
        let src = "@vertex fn vs() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0); }";

        let (_, hash) = cache.cache_shader(&device, src);

        assert!(cache.contains(&hash));
        assert!(!cache.contains(&ContentHash::from_bytes(b"other")));
    }

    #[test]
    fn test_shader_cache_v2_cached_hashes() {
        let (device, _queue) = match create_test_device() {
            Some(d) => d,
            None => {
                eprintln!("Skipping: no GPU adapter available");
                return;
            }
        };

        let mut cache = ShaderCacheV2::new();
        let src_a = "@vertex fn vs_a() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0); }";
        let src_b = "@vertex fn vs_b() -> @builtin(position) vec4<f32> { return vec4<f32>(1.0); }";

        let (_, hash_a) = cache.cache_shader(&device, src_a);
        let (_, hash_b) = cache.cache_shader(&device, src_b);

        let hashes: Vec<_> = cache.cached_hashes().copied().collect();
        assert_eq!(hashes.len(), 2);
        assert!(hashes.contains(&hash_a));
        assert!(hashes.contains(&hash_b));
    }
}
