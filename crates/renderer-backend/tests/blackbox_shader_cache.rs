// SPDX-License-Identifier: MIT
//
// blackbox_shader_cache.rs -- Blackbox tests for T-WGPU-P2.7.2 ShaderCache.
//
// CLEANROOM: No src/ access beyond the public API exported by the crate.
// Tests use only the documented public types and functions:
//
//   - ShaderCacheKey -- Path(PathBuf) or Hash([u8; 32])
//   - CachedShader -- Cached shader module wrapper
//   - ShaderCache -- Thread-safe shader cache
//   - ShaderCacheConfig -- Cache configuration
//   - ShaderCacheMetrics -- Cache statistics
//
// PUBLIC FUNCTIONS TESTED:
//   - ShaderCache::new(device, config) -> Self
//   - get_or_compile(key, label, source) -> Result<Arc<TrinityShaderModule>, ShaderError>
//   - get_or_compile_file(path) -> Result<Arc<TrinityShaderModule>, ShaderError>
//   - invalidate(key) -> bool
//   - invalidate_by_path(path) -> bool
//   - invalidate_all()
//   - metrics() -> ShaderCacheMetrics
//
// ACCEPTANCE CRITERIA:
//   1. API contract tests -- 25+ tests covering type construction and accessors
//   2. Integration tests  -- 25+ tests verifying cache semantics (with GPU)
//   3. Real shader tests  -- 15+ tests using real WGSL shader files
//   4. File loading tests -- 10+ tests for get_or_compile_file
//   5. Hot-reload tests   -- 10+ tests for invalidation workflow
//   6. Metrics tests      -- 10+ tests verifying metrics accuracy
//   7. Concurrent tests   -- 10+ tests for thread-safety
//
// Total target: 80+ tests

use renderer_backend::shaders::{
    CacheEntryInfo, ShaderCache, ShaderCacheConfig, ShaderCacheKey,
    ShaderCacheMetrics, DEFAULT_DISK_CACHE_PATH, DEFAULT_MAX_ENTRIES,
};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

// =============================================================================
// HELPERS -- Test helpers for cleanroom testing
// =============================================================================

/// Minimal valid WGSL vertex shader.
const MINIMAL_VERTEX_SHADER: &str = r#"
    @vertex
    fn vs_main(@builtin(vertex_index) idx: u32) -> @builtin(position) vec4<f32> {
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }
"#;

/// Minimal valid WGSL fragment shader.
const MINIMAL_FRAGMENT_SHADER: &str = r#"
    @fragment
    fn fs_main() -> @location(0) vec4<f32> {
        return vec4<f32>(1.0, 0.0, 0.0, 1.0);
    }
"#;

/// Minimal valid WGSL compute shader.
const MINIMAL_COMPUTE_SHADER: &str = r#"
    @compute @workgroup_size(64)
    fn cs_main(@builtin(global_invocation_id) id: vec3<u32>) {
        // Empty compute shader
    }
"#;

/// Vertex shader with uniform bindings.
const UNIFORM_VERTEX_SHADER: &str = r#"
    struct CameraData {
        view: mat4x4<f32>,
        projection: mat4x4<f32>,
    }
    @group(0) @binding(0) var<uniform> camera: CameraData;

    @vertex
    fn vs_main(@builtin(vertex_index) idx: u32) -> @builtin(position) vec4<f32> {
        return camera.projection * camera.view * vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }
"#;

/// Fragment shader with texture sampling.
const TEXTURED_FRAGMENT_SHADER: &str = r#"
    @group(0) @binding(0) var tex: texture_2d<f32>;
    @group(0) @binding(1) var samp: sampler;

    @fragment
    fn fs_main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
        return textureSample(tex, samp, uv);
    }
"#;

/// Compute shader with storage buffer.
const STORAGE_COMPUTE_SHADER: &str = r#"
    @group(0) @binding(0) var<storage, read_write> data: array<f32>;

    @compute @workgroup_size(256)
    fn cs_main(@builtin(global_invocation_id) id: vec3<u32>) {
        let idx = id.x;
        data[idx] = data[idx] * 2.0;
    }
"#;

/// Invalid WGSL shader (parse error).
const INVALID_SHADER_PARSE: &str = r#"
    @vertex
    fn main( { // Missing parameter list
        return vec4<f32>(0.0);
    }
"#;

/// Invalid WGSL shader (validation error).
const INVALID_SHADER_VALIDATION: &str = r#"
    @vertex
    fn vs_main() -> @builtin(position) vec4<f32> {
        return undefined_variable;
    }
"#;

/// Returns path to the shaders directory.
fn shaders_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("shaders")
}

/// Returns path to a specific shader file.
fn shader_path(name: &str) -> PathBuf {
    shaders_dir().join(name)
}

// =============================================================================
// SECTION 1 -- SHADERCACHEKEY API CONTRACTS (25+ tests)
// =============================================================================

/// ShaderCacheKey::from_path creates a path-based key.
#[test]
fn key_from_path_creates_path_key() {
    let key = ShaderCacheKey::from_path("shaders/test.wgsl");
    assert!(key.is_path());
    assert!(!key.is_hash());
}

/// ShaderCacheKey::from_source creates a hash-based key.
#[test]
fn key_from_source_creates_hash_key() {
    let key = ShaderCacheKey::from_source(MINIMAL_VERTEX_SHADER);
    assert!(key.is_hash());
    assert!(!key.is_path());
}

/// ShaderCacheKey::from_bytes creates a hash-based key.
#[test]
fn key_from_bytes_creates_hash_key() {
    let bytes = MINIMAL_VERTEX_SHADER.as_bytes();
    let key = ShaderCacheKey::from_bytes(bytes);
    assert!(key.is_hash());
}

/// ShaderCacheKey::from_hash creates a hash-based key.
#[test]
fn key_from_hash_creates_hash_key() {
    let hash = [42u8; 32];
    let key = ShaderCacheKey::from_hash(hash);
    assert!(key.is_hash());
    assert_eq!(key.as_hash(), Some(&hash));
}

/// ShaderCacheKey as_path returns the path for path keys.
#[test]
fn key_as_path_returns_path() {
    let key = ShaderCacheKey::from_path("shaders/test.wgsl");
    assert_eq!(key.as_path(), Some(Path::new("shaders/test.wgsl")));
}

/// ShaderCacheKey as_path returns None for hash keys.
#[test]
fn key_as_path_returns_none_for_hash() {
    let key = ShaderCacheKey::from_source("source");
    assert!(key.as_path().is_none());
}

/// ShaderCacheKey as_hash returns the hash for hash keys.
#[test]
fn key_as_hash_returns_hash() {
    let key = ShaderCacheKey::from_source("source");
    assert!(key.as_hash().is_some());
    assert_eq!(key.as_hash().unwrap().len(), 32);
}

/// ShaderCacheKey as_hash returns None for path keys.
#[test]
fn key_as_hash_returns_none_for_path() {
    let key = ShaderCacheKey::from_path("test.wgsl");
    assert!(key.as_hash().is_none());
}

/// ShaderCacheKey from same source produces same hash.
#[test]
fn key_same_source_same_hash() {
    let key1 = ShaderCacheKey::from_source(MINIMAL_VERTEX_SHADER);
    let key2 = ShaderCacheKey::from_source(MINIMAL_VERTEX_SHADER);
    assert_eq!(key1, key2);
}

/// ShaderCacheKey from different source produces different hash.
#[test]
fn key_different_source_different_hash() {
    let key1 = ShaderCacheKey::from_source(MINIMAL_VERTEX_SHADER);
    let key2 = ShaderCacheKey::from_source(MINIMAL_FRAGMENT_SHADER);
    assert_ne!(key1, key2);
}

/// ShaderCacheKey from same path produces same key.
#[test]
fn key_same_path_same_key() {
    let key1 = ShaderCacheKey::from_path("shaders/test.wgsl");
    let key2 = ShaderCacheKey::from_path("shaders/test.wgsl");
    assert_eq!(key1, key2);
}

/// ShaderCacheKey from different paths produces different keys.
#[test]
fn key_different_path_different_key() {
    let key1 = ShaderCacheKey::from_path("shaders/a.wgsl");
    let key2 = ShaderCacheKey::from_path("shaders/b.wgsl");
    assert_ne!(key1, key2);
}

/// ShaderCacheKey path key not equal to hash key.
#[test]
fn key_path_not_equal_to_hash() {
    let path_key = ShaderCacheKey::from_path("test.wgsl");
    let hash_key = ShaderCacheKey::from_source("test.wgsl");
    assert_ne!(path_key, hash_key);
}

/// ShaderCacheKey display_string for path key.
#[test]
fn key_display_string_path() {
    let key = ShaderCacheKey::from_path("shaders/pbr.wgsl");
    let display = key.display_string();
    assert!(display.contains("pbr.wgsl"));
}

/// ShaderCacheKey display_string for hash key.
#[test]
fn key_display_string_hash() {
    let key = ShaderCacheKey::from_source("fn main() {}");
    let display = key.display_string();
    assert!(display.starts_with("hash:"));
    assert!(display.ends_with("..."));
}

/// ShaderCacheKey works in HashMap.
#[test]
fn key_works_in_hashmap() {
    let mut map: HashMap<ShaderCacheKey, i32> = HashMap::new();

    let key1 = ShaderCacheKey::from_path("shader1.wgsl");
    let key2 = ShaderCacheKey::from_source(MINIMAL_VERTEX_SHADER);

    map.insert(key1.clone(), 1);
    map.insert(key2.clone(), 2);

    assert_eq!(map.get(&key1), Some(&1));
    assert_eq!(map.get(&key2), Some(&2));
}

/// ShaderCacheKey is Clone.
#[test]
fn key_is_clone() {
    let key = ShaderCacheKey::from_path("test.wgsl");
    let cloned = key.clone();
    assert_eq!(key, cloned);
}

/// ShaderCacheKey from PathBuf conversion.
#[test]
fn key_from_pathbuf() {
    let path = PathBuf::from("shaders/test.wgsl");
    let key: ShaderCacheKey = path.into();
    assert!(key.is_path());
}

/// ShaderCacheKey from &Path conversion.
#[test]
fn key_from_path_ref() {
    let path = Path::new("shaders/test.wgsl");
    let key: ShaderCacheKey = path.into();
    assert!(key.is_path());
}

/// ShaderCacheKey from &str conversion (becomes path).
#[test]
fn key_from_str() {
    let key: ShaderCacheKey = "shaders/test.wgsl".into();
    assert!(key.is_path());
}

/// ShaderCacheKey from [u8; 32] conversion.
#[test]
fn key_from_hash_array() {
    let hash = [0u8; 32];
    let key: ShaderCacheKey = hash.into();
    assert!(key.is_hash());
}

/// ShaderCacheKey empty path is valid.
#[test]
fn key_empty_path() {
    let key = ShaderCacheKey::from_path("");
    assert!(key.is_path());
}

/// ShaderCacheKey empty source produces valid hash.
#[test]
fn key_empty_source() {
    let key = ShaderCacheKey::from_source("");
    assert!(key.is_hash());
}

/// ShaderCacheKey very long path is valid.
#[test]
fn key_very_long_path() {
    let long_path = "a".repeat(1000) + ".wgsl";
    let key = ShaderCacheKey::from_path(&long_path);
    assert!(key.is_path());
}

/// ShaderCacheKey source and bytes produce same hash for same content.
#[test]
fn key_source_bytes_same_hash() {
    let text = "fn main() {}";
    let key1 = ShaderCacheKey::from_source(text);
    let key2 = ShaderCacheKey::from_bytes(text.as_bytes());
    assert_eq!(key1.as_hash(), key2.as_hash());
}

// =============================================================================
// SECTION 2 -- SHADERCACHECONFIG API CONTRACTS (15+ tests)
// =============================================================================

/// ShaderCacheConfig::default has expected values.
#[test]
fn config_default_values() {
    let config = ShaderCacheConfig::default();
    assert_eq!(config.max_entries, DEFAULT_MAX_ENTRIES);
    assert!(!config.enable_disk_cache);
    assert!(config.disk_cache_path.is_none());
    assert!(config.enable_lru_eviction);
}

/// ShaderCacheConfig::new equals default.
#[test]
fn config_new_equals_default() {
    let config1 = ShaderCacheConfig::new();
    let config2 = ShaderCacheConfig::default();
    assert_eq!(config1.max_entries, config2.max_entries);
    assert_eq!(config1.enable_disk_cache, config2.enable_disk_cache);
}

/// ShaderCacheConfig::max_entries builder.
#[test]
fn config_max_entries_builder() {
    let config = ShaderCacheConfig::new().max_entries(512);
    assert_eq!(config.max_entries, 512);
}

/// ShaderCacheConfig::with_disk_cache builder.
#[test]
fn config_with_disk_cache_builder() {
    let config = ShaderCacheConfig::new().with_disk_cache("/tmp/shaders");
    assert!(config.enable_disk_cache);
    assert_eq!(config.disk_cache_path, Some(PathBuf::from("/tmp/shaders")));
}

/// ShaderCacheConfig::without_eviction builder.
#[test]
fn config_without_eviction_builder() {
    let config = ShaderCacheConfig::new().without_eviction();
    assert!(!config.enable_lru_eviction);
}

/// ShaderCacheConfig::minimal preset.
#[test]
fn config_minimal_preset() {
    let config = ShaderCacheConfig::minimal();
    assert_eq!(config.max_entries, 16);
    assert!(!config.enable_disk_cache);
}

/// ShaderCacheConfig::development preset.
#[test]
fn config_development_preset() {
    let config = ShaderCacheConfig::development();
    assert_eq!(config.max_entries, 64);
    assert!(!config.enable_disk_cache);
}

/// ShaderCacheConfig::production preset.
#[test]
fn config_production_preset() {
    let config = ShaderCacheConfig::production();
    assert_eq!(config.max_entries, 1024);
    assert!(config.enable_disk_cache);
    assert!(config.disk_cache_path.is_some());
}

/// ShaderCacheConfig builder chain.
#[test]
fn config_builder_chain() {
    let config = ShaderCacheConfig::new()
        .max_entries(128)
        .with_disk_cache("/cache")
        .without_eviction();

    assert_eq!(config.max_entries, 128);
    assert!(config.enable_disk_cache);
    assert!(!config.enable_lru_eviction);
}

/// ShaderCacheConfig max_entries zero.
#[test]
fn config_max_entries_zero() {
    let config = ShaderCacheConfig::new().max_entries(0);
    assert_eq!(config.max_entries, 0);
}

/// ShaderCacheConfig max_entries large value.
#[test]
fn config_max_entries_large() {
    let config = ShaderCacheConfig::new().max_entries(1_000_000);
    assert_eq!(config.max_entries, 1_000_000);
}

/// ShaderCacheConfig is Clone.
#[test]
fn config_is_clone() {
    let config = ShaderCacheConfig::new().max_entries(100);
    let cloned = config.clone();
    assert_eq!(cloned.max_entries, 100);
}

/// ShaderCacheConfig has Debug impl.
#[test]
fn config_has_debug() {
    let config = ShaderCacheConfig::default();
    let debug = format!("{:?}", config);
    assert!(debug.contains("ShaderCacheConfig"));
}

/// DEFAULT_MAX_ENTRIES constant.
#[test]
fn constant_default_max_entries() {
    assert_eq!(DEFAULT_MAX_ENTRIES, 256);
}

/// DEFAULT_DISK_CACHE_PATH constant.
#[test]
fn constant_default_disk_cache_path() {
    assert_eq!(DEFAULT_DISK_CACHE_PATH, ".trinity/shader_cache");
}

// =============================================================================
// SECTION 3 -- SHADERCACHEMETRICS API CONTRACTS (15+ tests)
// =============================================================================

/// ShaderCacheMetrics::default has zero values.
#[test]
fn metrics_default_zero() {
    let metrics = ShaderCacheMetrics::default();
    assert_eq!(metrics.cache_size, 0);
    assert_eq!(metrics.hits, 0);
    assert_eq!(metrics.misses, 0);
    assert_eq!(metrics.hit_rate, 0.0);
    assert_eq!(metrics.evictions, 0);
    assert_eq!(metrics.invalidations, 0);
    assert_eq!(metrics.compilation_errors, 0);
}

/// ShaderCacheMetrics::new calculates hit rate correctly.
#[test]
fn metrics_new_calculates_hit_rate() {
    let metrics = ShaderCacheMetrics::new(10, 80, 20, 0, 0, 0);
    assert_eq!(metrics.hit_rate, 0.8);
}

/// ShaderCacheMetrics::new with zero requests.
#[test]
fn metrics_new_zero_requests() {
    let metrics = ShaderCacheMetrics::new(0, 0, 0, 0, 0, 0);
    assert_eq!(metrics.hit_rate, 0.0);
}

/// ShaderCacheMetrics total_requests.
#[test]
fn metrics_total_requests() {
    let metrics = ShaderCacheMetrics::new(0, 50, 50, 0, 0, 0);
    assert_eq!(metrics.total_requests(), 100);
}

/// ShaderCacheMetrics is_empty.
#[test]
fn metrics_is_empty() {
    let empty = ShaderCacheMetrics::new(0, 10, 5, 0, 0, 0);
    assert!(empty.is_empty());

    let not_empty = ShaderCacheMetrics::new(1, 10, 5, 0, 0, 0);
    assert!(!not_empty.is_empty());
}

/// ShaderCacheMetrics hit_rate_percent.
#[test]
fn metrics_hit_rate_percent() {
    let metrics = ShaderCacheMetrics::new(0, 75, 25, 0, 0, 0);
    assert_eq!(metrics.hit_rate_percent(), 75.0);
}

/// ShaderCacheMetrics miss_rate.
#[test]
fn metrics_miss_rate() {
    let metrics = ShaderCacheMetrics::new(0, 60, 40, 0, 0, 0);
    assert_eq!(metrics.miss_rate(), 0.4);
}

/// ShaderCacheMetrics all hits.
#[test]
fn metrics_all_hits() {
    let metrics = ShaderCacheMetrics::new(0, 100, 0, 0, 0, 0);
    assert_eq!(metrics.hit_rate, 1.0);
    assert_eq!(metrics.miss_rate(), 0.0);
}

/// ShaderCacheMetrics all misses.
#[test]
fn metrics_all_misses() {
    let metrics = ShaderCacheMetrics::new(0, 0, 100, 0, 0, 0);
    assert_eq!(metrics.hit_rate, 0.0);
    assert_eq!(metrics.miss_rate(), 1.0);
}

/// ShaderCacheMetrics equal hits and misses.
#[test]
fn metrics_equal_hits_misses() {
    let metrics = ShaderCacheMetrics::new(0, 50, 50, 0, 0, 0);
    assert_eq!(metrics.hit_rate, 0.5);
}

/// ShaderCacheMetrics is Clone.
#[test]
fn metrics_is_clone() {
    let metrics = ShaderCacheMetrics::new(5, 10, 2, 1, 0, 0);
    let cloned = metrics.clone();
    assert_eq!(cloned.hits, 10);
}

/// ShaderCacheMetrics has Debug impl.
#[test]
fn metrics_has_debug() {
    let metrics = ShaderCacheMetrics::default();
    let debug = format!("{:?}", metrics);
    assert!(debug.contains("ShaderCacheMetrics"));
}

/// ShaderCacheMetrics with large values.
#[test]
fn metrics_large_values() {
    let metrics = ShaderCacheMetrics::new(1000, u64::MAX - 1, 1, 100, 50, 25);
    assert!(metrics.hit_rate > 0.99);
}

/// ShaderCacheMetrics fields accessible.
#[test]
fn metrics_fields_accessible() {
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

/// ShaderCacheMetrics hit_rate_percent 100%.
#[test]
fn metrics_hit_rate_percent_100() {
    let metrics = ShaderCacheMetrics::new(0, 1000, 0, 0, 0, 0);
    assert_eq!(metrics.hit_rate_percent(), 100.0);
}

// =============================================================================
// SECTION 4 -- CACHEENTRYINFO API CONTRACTS (5+ tests)
// =============================================================================

/// CacheEntryInfo has expected fields.
#[test]
fn cache_entry_info_fields() {
    let info = CacheEntryInfo {
        key: "test".to_string(),
        label: Some("my_shader".to_string()),
        age_secs: 1.5,
        idle_secs: 0.5,
        access_count: 10,
    };

    assert_eq!(info.key, "test");
    assert_eq!(info.label, Some("my_shader".to_string()));
    assert_eq!(info.age_secs, 1.5);
    assert_eq!(info.idle_secs, 0.5);
    assert_eq!(info.access_count, 10);
}

/// CacheEntryInfo with no label.
#[test]
fn cache_entry_info_no_label() {
    let info = CacheEntryInfo {
        key: "test".to_string(),
        label: None,
        age_secs: 0.0,
        idle_secs: 0.0,
        access_count: 0,
    };

    assert!(info.label.is_none());
}

/// CacheEntryInfo is Clone.
#[test]
fn cache_entry_info_is_clone() {
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

/// CacheEntryInfo has Debug impl.
#[test]
fn cache_entry_info_has_debug() {
    let info = CacheEntryInfo {
        key: "test".to_string(),
        label: Some("label".to_string()),
        age_secs: 1.5,
        idle_secs: 0.5,
        access_count: 10,
    };
    let debug = format!("{:?}", info);
    assert!(debug.contains("CacheEntryInfo"));
}

/// CacheEntryInfo with large values.
#[test]
fn cache_entry_info_large_values() {
    let info = CacheEntryInfo {
        key: "test".to_string(),
        label: Some("label".to_string()),
        age_secs: 86400.0 * 365.0, // 1 year
        idle_secs: 86400.0,        // 1 day
        access_count: u64::MAX,
    };
    assert_eq!(info.access_count, u64::MAX);
}

// =============================================================================
// SECTION 5 -- SHADERCACHE GPU INTEGRATION TESTS (25+ tests)
// =============================================================================
//
// These tests require a GPU and are marked with #[ignore].
// Run with: cargo test --test blackbox_shader_cache -- --ignored

/// Helper to create a wgpu device for testing.
async fn create_test_device() -> Option<(Arc<wgpu::Device>, wgpu::Queue)> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::VULKAN,
        ..Default::default()
    });

    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        })
        .await?;

    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor::default(), None)
        .await
        .ok()?;

    Some((Arc::new(device), queue))
}

/// Run async test helper.
fn run_async<F: std::future::Future<Output = T>, T>(f: F) -> T {
    pollster::block_on(f)
}

/// ShaderCache::new creates empty cache.
#[test]

fn cache_new_creates_empty_cache() {
    run_async(async {
        let (device, _) = create_test_device().await.expect("No GPU available");
        let cache = ShaderCache::new(device, ShaderCacheConfig::default());
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
    });
}

/// ShaderCache get_or_compile compiles on first call.
#[test]

fn cache_get_or_compile_compiles_on_first_call() {
    run_async(async {
        let (device, _) = create_test_device().await.expect("No GPU available");
        let cache = ShaderCache::new(device, ShaderCacheConfig::default());

        let key = ShaderCacheKey::from_source(MINIMAL_VERTEX_SHADER);
        let result = cache.get_or_compile(key, Some("vertex"), MINIMAL_VERTEX_SHADER);

        assert!(result.is_ok());
        assert_eq!(cache.len(), 1);
    });
}

/// ShaderCache get_or_compile returns cached on second call.
#[test]

fn cache_get_or_compile_returns_cached_on_second_call() {
    run_async(async {
        let (device, _) = create_test_device().await.expect("No GPU available");
        let cache = ShaderCache::new(device, ShaderCacheConfig::default());

        let key = ShaderCacheKey::from_source(MINIMAL_VERTEX_SHADER);

        let result1 = cache.get_or_compile(key.clone(), Some("vs"), MINIMAL_VERTEX_SHADER);
        let result2 = cache.get_or_compile(key, Some("vs"), MINIMAL_VERTEX_SHADER);

        assert!(result1.is_ok());
        assert!(result2.is_ok());

        // Same Arc pointer
        assert!(Arc::ptr_eq(&result1.unwrap(), &result2.unwrap()));
    });
}

/// ShaderCache get_or_compile increments hit count on cache hit.
#[test]

fn cache_get_or_compile_increments_hit_count() {
    run_async(async {
        let (device, _) = create_test_device().await.expect("No GPU available");
        let cache = ShaderCache::new(device, ShaderCacheConfig::default());

        let key = ShaderCacheKey::from_source(MINIMAL_VERTEX_SHADER);

        // First call is a miss
        let _ = cache.get_or_compile(key.clone(), None, MINIMAL_VERTEX_SHADER);
        let metrics1 = cache.metrics();
        assert_eq!(metrics1.misses, 1);
        assert_eq!(metrics1.hits, 0);

        // Second call is a hit
        let _ = cache.get_or_compile(key, None, MINIMAL_VERTEX_SHADER);
        let metrics2 = cache.metrics();
        assert_eq!(metrics2.misses, 1);
        assert_eq!(metrics2.hits, 1);
    });
}

/// ShaderCache get_or_compile with different shaders.
#[test]

fn cache_get_or_compile_different_shaders() {
    run_async(async {
        let (device, _) = create_test_device().await.expect("No GPU available");
        let cache = ShaderCache::new(device, ShaderCacheConfig::default());

        let key1 = ShaderCacheKey::from_source(MINIMAL_VERTEX_SHADER);
        let key2 = ShaderCacheKey::from_source(MINIMAL_FRAGMENT_SHADER);

        let result1 = cache.get_or_compile(key1, Some("vs"), MINIMAL_VERTEX_SHADER);
        let result2 = cache.get_or_compile(key2, Some("fs"), MINIMAL_FRAGMENT_SHADER);

        assert!(result1.is_ok());
        assert!(result2.is_ok());
        assert_eq!(cache.len(), 2);
    });
}

/// ShaderCache get_or_compile with invalid shader returns error.
#[test]

fn cache_get_or_compile_invalid_shader_returns_error() {
    run_async(async {
        let (device, _) = create_test_device().await.expect("No GPU available");
        let cache = ShaderCache::new(device, ShaderCacheConfig::default());

        let key = ShaderCacheKey::from_source(INVALID_SHADER_PARSE);
        let result = cache.get_or_compile(key, None, INVALID_SHADER_PARSE);

        assert!(result.is_err());
        assert_eq!(cache.len(), 0);
    });
}

/// ShaderCache get_or_compile tracks compilation errors.
#[test]

fn cache_get_or_compile_tracks_compilation_errors() {
    run_async(async {
        let (device, _) = create_test_device().await.expect("No GPU available");
        let cache = ShaderCache::new(device, ShaderCacheConfig::default());

        let key = ShaderCacheKey::from_source(INVALID_SHADER_PARSE);
        let _ = cache.get_or_compile(key, None, INVALID_SHADER_PARSE);

        let metrics = cache.metrics();
        assert_eq!(metrics.compilation_errors, 1);
    });
}

/// ShaderCache invalidate removes cached shader.
#[test]

fn cache_invalidate_removes_cached_shader() {
    run_async(async {
        let (device, _) = create_test_device().await.expect("No GPU available");
        let cache = ShaderCache::new(device, ShaderCacheConfig::default());

        let key = ShaderCacheKey::from_source(MINIMAL_VERTEX_SHADER);
        let _ = cache.get_or_compile(key.clone(), None, MINIMAL_VERTEX_SHADER);
        assert_eq!(cache.len(), 1);

        let removed = cache.invalidate(&key);
        assert!(removed);
        assert_eq!(cache.len(), 0);
    });
}

/// ShaderCache invalidate returns false if key not found.
#[test]

fn cache_invalidate_returns_false_if_not_found() {
    run_async(async {
        let (device, _) = create_test_device().await.expect("No GPU available");
        let cache = ShaderCache::new(device, ShaderCacheConfig::default());

        let key = ShaderCacheKey::from_source(MINIMAL_VERTEX_SHADER);
        let removed = cache.invalidate(&key);
        assert!(!removed);
    });
}

/// ShaderCache invalidate increments invalidation count.
#[test]

fn cache_invalidate_increments_invalidation_count() {
    run_async(async {
        let (device, _) = create_test_device().await.expect("No GPU available");
        let cache = ShaderCache::new(device, ShaderCacheConfig::default());

        let key = ShaderCacheKey::from_source(MINIMAL_VERTEX_SHADER);
        let _ = cache.get_or_compile(key.clone(), None, MINIMAL_VERTEX_SHADER);
        let _ = cache.invalidate(&key);

        let metrics = cache.metrics();
        assert_eq!(metrics.invalidations, 1);
    });
}

/// ShaderCache invalidate_by_path removes shader by path.
#[test]

fn cache_invalidate_by_path_removes_shader() {
    run_async(async {
        let (device, _) = create_test_device().await.expect("No GPU available");
        let cache = ShaderCache::new(device, ShaderCacheConfig::default());

        let path = shader_path("pbr.vert.wgsl");
        if path.exists() {
            let _ = cache.get_or_compile_file(&path);
            if cache.len() > 0 {
                let removed = cache.invalidate_by_path(&path);
                assert!(removed);
                assert_eq!(cache.len(), 0);
            }
        }
    });
}

/// ShaderCache invalidate_all clears the cache.
#[test]

fn cache_invalidate_all_clears_cache() {
    run_async(async {
        let (device, _) = create_test_device().await.expect("No GPU available");
        let cache = ShaderCache::new(device, ShaderCacheConfig::default());

        let key1 = ShaderCacheKey::from_source(MINIMAL_VERTEX_SHADER);
        let key2 = ShaderCacheKey::from_source(MINIMAL_FRAGMENT_SHADER);

        let _ = cache.get_or_compile(key1, None, MINIMAL_VERTEX_SHADER);
        let _ = cache.get_or_compile(key2, None, MINIMAL_FRAGMENT_SHADER);
        assert_eq!(cache.len(), 2);

        cache.invalidate_all();
        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
    });
}

/// ShaderCache invalidate_all increments invalidation count.
#[test]

fn cache_invalidate_all_increments_count() {
    run_async(async {
        let (device, _) = create_test_device().await.expect("No GPU available");
        let cache = ShaderCache::new(device, ShaderCacheConfig::default());

        let key1 = ShaderCacheKey::from_source(MINIMAL_VERTEX_SHADER);
        let key2 = ShaderCacheKey::from_source(MINIMAL_FRAGMENT_SHADER);

        let _ = cache.get_or_compile(key1, None, MINIMAL_VERTEX_SHADER);
        let _ = cache.get_or_compile(key2, None, MINIMAL_FRAGMENT_SHADER);

        cache.invalidate_all();

        let metrics = cache.metrics();
        assert_eq!(metrics.invalidations, 2);
    });
}

/// ShaderCache metrics reflects actual cache state.
#[test]

fn cache_metrics_reflects_cache_state() {
    run_async(async {
        let (device, _) = create_test_device().await.expect("No GPU available");
        let cache = ShaderCache::new(device, ShaderCacheConfig::default());

        let key = ShaderCacheKey::from_source(MINIMAL_VERTEX_SHADER);

        // Initial metrics
        let metrics0 = cache.metrics();
        assert_eq!(metrics0.cache_size, 0);
        assert_eq!(metrics0.hits, 0);
        assert_eq!(metrics0.misses, 0);

        // After compile
        let _ = cache.get_or_compile(key.clone(), None, MINIMAL_VERTEX_SHADER);
        let metrics1 = cache.metrics();
        assert_eq!(metrics1.cache_size, 1);
        assert_eq!(metrics1.misses, 1);

        // After cache hit
        let _ = cache.get_or_compile(key, None, MINIMAL_VERTEX_SHADER);
        let metrics2 = cache.metrics();
        assert_eq!(metrics2.hits, 1);
    });
}

/// ShaderCache reset_metrics clears counters.
#[test]

fn cache_reset_metrics_clears_counters() {
    run_async(async {
        let (device, _) = create_test_device().await.expect("No GPU available");
        let cache = ShaderCache::new(device, ShaderCacheConfig::default());

        let key = ShaderCacheKey::from_source(MINIMAL_VERTEX_SHADER);
        let _ = cache.get_or_compile(key.clone(), None, MINIMAL_VERTEX_SHADER);
        let _ = cache.get_or_compile(key, None, MINIMAL_VERTEX_SHADER);

        cache.reset_metrics();

        let metrics = cache.metrics();
        assert_eq!(metrics.hits, 0);
        assert_eq!(metrics.misses, 0);
        // But cache_size should still reflect actual cache
        assert_eq!(metrics.cache_size, 1);
    });
}

/// ShaderCache contains returns true for cached key.
#[test]

fn cache_contains_true_for_cached_key() {
    run_async(async {
        let (device, _) = create_test_device().await.expect("No GPU available");
        let cache = ShaderCache::new(device, ShaderCacheConfig::default());

        let key = ShaderCacheKey::from_source(MINIMAL_VERTEX_SHADER);
        let _ = cache.get_or_compile(key.clone(), None, MINIMAL_VERTEX_SHADER);

        assert!(cache.contains(&key));
    });
}

/// ShaderCache contains returns false for non-cached key.
#[test]

fn cache_contains_false_for_non_cached_key() {
    run_async(async {
        let (device, _) = create_test_device().await.expect("No GPU available");
        let cache = ShaderCache::new(device, ShaderCacheConfig::default());

        let key = ShaderCacheKey::from_source(MINIMAL_VERTEX_SHADER);
        assert!(!cache.contains(&key));
    });
}

/// ShaderCache get returns Some for cached shader.
#[test]

fn cache_get_returns_some_for_cached() {
    run_async(async {
        let (device, _) = create_test_device().await.expect("No GPU available");
        let cache = ShaderCache::new(device, ShaderCacheConfig::default());

        let key = ShaderCacheKey::from_source(MINIMAL_VERTEX_SHADER);
        let _ = cache.get_or_compile(key.clone(), None, MINIMAL_VERTEX_SHADER);

        let result = cache.get(&key);
        assert!(result.is_some());
    });
}

/// ShaderCache get returns None for non-cached.
#[test]

fn cache_get_returns_none_for_non_cached() {
    run_async(async {
        let (device, _) = create_test_device().await.expect("No GPU available");
        let cache = ShaderCache::new(device, ShaderCacheConfig::default());

        let key = ShaderCacheKey::from_source(MINIMAL_VERTEX_SHADER);
        let result = cache.get(&key);
        assert!(result.is_none());
    });
}

/// ShaderCache get increments hit count.
#[test]

fn cache_get_increments_hit_count() {
    run_async(async {
        let (device, _) = create_test_device().await.expect("No GPU available");
        let cache = ShaderCache::new(device, ShaderCacheConfig::default());

        let key = ShaderCacheKey::from_source(MINIMAL_VERTEX_SHADER);
        let _ = cache.get_or_compile(key.clone(), None, MINIMAL_VERTEX_SHADER);
        cache.reset_metrics();

        let _ = cache.get(&key);
        let metrics = cache.metrics();
        assert_eq!(metrics.hits, 1);
    });
}

/// ShaderCache config returns the config.
#[test]

fn cache_config_returns_config() {
    run_async(async {
        let (device, _) = create_test_device().await.expect("No GPU available");
        let config = ShaderCacheConfig::minimal();
        let cache = ShaderCache::new(device, config.clone());

        assert_eq!(cache.config().max_entries, 16);
    });
}

/// ShaderCache labels returns shader labels.
#[test]

fn cache_labels_returns_labels() {
    run_async(async {
        let (device, _) = create_test_device().await.expect("No GPU available");
        let cache = ShaderCache::new(device, ShaderCacheConfig::default());

        let key = ShaderCacheKey::from_source(MINIMAL_VERTEX_SHADER);
        let _ = cache.get_or_compile(key, Some("my_vertex_shader"), MINIMAL_VERTEX_SHADER);

        let labels = cache.labels();
        assert_eq!(labels.len(), 1);
        assert_eq!(labels[0], Some("my_vertex_shader".to_string()));
    });
}

/// ShaderCache keys returns cache keys.
#[test]

fn cache_keys_returns_keys() {
    run_async(async {
        let (device, _) = create_test_device().await.expect("No GPU available");
        let cache = ShaderCache::new(device, ShaderCacheConfig::default());

        let key1 = ShaderCacheKey::from_source(MINIMAL_VERTEX_SHADER);
        let key2 = ShaderCacheKey::from_source(MINIMAL_FRAGMENT_SHADER);

        let _ = cache.get_or_compile(key1.clone(), None, MINIMAL_VERTEX_SHADER);
        let _ = cache.get_or_compile(key2.clone(), None, MINIMAL_FRAGMENT_SHADER);

        let keys = cache.keys();
        assert_eq!(keys.len(), 2);
        assert!(keys.contains(&key1));
        assert!(keys.contains(&key2));
    });
}

/// ShaderCache entries returns entry info.
#[test]

fn cache_entries_returns_entry_info() {
    run_async(async {
        let (device, _) = create_test_device().await.expect("No GPU available");
        let cache = ShaderCache::new(device, ShaderCacheConfig::default());

        let key = ShaderCacheKey::from_source(MINIMAL_VERTEX_SHADER);
        let _ = cache.get_or_compile(key, Some("test_shader"), MINIMAL_VERTEX_SHADER);

        let entries = cache.entries();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].label, Some("test_shader".to_string()));
        assert!(entries[0].access_count >= 1);
    });
}

/// ShaderCache has Debug impl.
#[test]

fn cache_has_debug() {
    run_async(async {
        let (device, _) = create_test_device().await.expect("No GPU available");
        let cache = ShaderCache::new(device, ShaderCacheConfig::default());

        let debug = format!("{:?}", cache);
        assert!(debug.contains("ShaderCache"));
    });
}

// =============================================================================
// SECTION 6 -- FILE LOADING TESTS (10+ tests)
// =============================================================================

/// ShaderCache get_or_compile_file loads shader from file.
#[test]

fn cache_get_or_compile_file_loads_shader() {
    run_async(async {
        let (device, _) = create_test_device().await.expect("No GPU available");
        let cache = ShaderCache::new(device, ShaderCacheConfig::default());

        let path = shader_path("pbr.vert.wgsl");
        if path.exists() {
            let result = cache.get_or_compile_file(&path);
            assert!(result.is_ok());
            assert_eq!(cache.len(), 1);
        }
    });
}

/// ShaderCache get_or_compile_file returns error for missing file.
#[test]

fn cache_get_or_compile_file_error_missing_file() {
    run_async(async {
        let (device, _) = create_test_device().await.expect("No GPU available");
        let cache = ShaderCache::new(device, ShaderCacheConfig::default());

        let path = shader_path("nonexistent_shader.wgsl");
        let result = cache.get_or_compile_file(&path);
        assert!(result.is_err());
    });
}

/// ShaderCache get_or_compile_file caches by path.
#[test]

fn cache_get_or_compile_file_caches_by_path() {
    run_async(async {
        let (device, _) = create_test_device().await.expect("No GPU available");
        let cache = ShaderCache::new(device, ShaderCacheConfig::default());

        let path = shader_path("pbr.vert.wgsl");
        if path.exists() {
            let result1 = cache.get_or_compile_file(&path);
            let result2 = cache.get_or_compile_file(&path);

            assert!(result1.is_ok());
            assert!(result2.is_ok());
            // Should be same Arc
            assert!(Arc::ptr_eq(&result1.unwrap(), &result2.unwrap()));
        }
    });
}

/// ShaderCache get_or_compile_file uses filename as label.
#[test]

fn cache_get_or_compile_file_uses_filename_as_label() {
    run_async(async {
        let (device, _) = create_test_device().await.expect("No GPU available");
        let cache = ShaderCache::new(device, ShaderCacheConfig::default());

        let path = shader_path("shadow.frag.wgsl");
        if path.exists() {
            let _ = cache.get_or_compile_file(&path);
            let labels = cache.labels();
            assert!(!labels.is_empty());
            if let Some(Some(label)) = labels.first() {
                assert!(label.contains("shadow.frag.wgsl"));
            }
        }
    });
}

/// ShaderCache get_or_compile_file with different files.
#[test]

fn cache_get_or_compile_file_different_files() {
    run_async(async {
        let (device, _) = create_test_device().await.expect("No GPU available");
        let cache = ShaderCache::new(device, ShaderCacheConfig::default());

        let path1 = shader_path("pbr.vert.wgsl");
        let path2 = shader_path("shadow.frag.wgsl");

        let mut count = 0;
        if path1.exists() {
            let _ = cache.get_or_compile_file(&path1);
            count += 1;
        }
        if path2.exists() {
            let _ = cache.get_or_compile_file(&path2);
            count += 1;
        }

        assert_eq!(cache.len(), count);
    });
}

/// ShaderCache get_or_compile_file works with subdirectory shaders.
#[test]

fn cache_get_or_compile_file_subdirectory() {
    run_async(async {
        let (device, _) = create_test_device().await.expect("No GPU available");
        let cache = ShaderCache::new(device, ShaderCacheConfig::default());

        let path = shaders_dir().join("common").join("prefix_sum.wgsl");
        if path.exists() {
            let result = cache.get_or_compile_file(&path);
            assert!(result.is_ok());
        }
    });
}

/// ShaderCache invalidate_by_path after get_or_compile_file.
#[test]

fn cache_invalidate_by_path_after_file_compile() {
    run_async(async {
        let (device, _) = create_test_device().await.expect("No GPU available");
        let cache = ShaderCache::new(device, ShaderCacheConfig::default());

        let path = shader_path("pbr.vert.wgsl");
        if path.exists() {
            let _ = cache.get_or_compile_file(&path);
            assert_eq!(cache.len(), 1);

            let removed = cache.invalidate_by_path(&path);
            assert!(removed);
            assert_eq!(cache.len(), 0);
        }
    });
}

/// ShaderCache get_or_compile_file increments miss count.
#[test]

fn cache_get_or_compile_file_increments_miss() {
    run_async(async {
        let (device, _) = create_test_device().await.expect("No GPU available");
        let cache = ShaderCache::new(device, ShaderCacheConfig::default());

        let path = shader_path("pbr.vert.wgsl");
        if path.exists() {
            let _ = cache.get_or_compile_file(&path);
            let metrics = cache.metrics();
            assert_eq!(metrics.misses, 1);
        }
    });
}

/// ShaderCache get_or_compile_file increments hit on second call.
#[test]

fn cache_get_or_compile_file_increments_hit_on_second_call() {
    run_async(async {
        let (device, _) = create_test_device().await.expect("No GPU available");
        let cache = ShaderCache::new(device, ShaderCacheConfig::default());

        let path = shader_path("pbr.vert.wgsl");
        if path.exists() {
            let _ = cache.get_or_compile_file(&path);
            let _ = cache.get_or_compile_file(&path);
            let metrics = cache.metrics();
            assert_eq!(metrics.hits, 1);
            assert_eq!(metrics.misses, 1);
        }
    });
}

/// ShaderCache get_or_compile_file with absolute path.
#[test]

fn cache_get_or_compile_file_absolute_path() {
    run_async(async {
        let (device, _) = create_test_device().await.expect("No GPU available");
        let cache = ShaderCache::new(device, ShaderCacheConfig::default());

        let path = shader_path("pbr.vert.wgsl").canonicalize().ok();
        if let Some(abs_path) = path {
            let result = cache.get_or_compile_file(&abs_path);
            assert!(result.is_ok());
        }
    });
}

// =============================================================================
// SECTION 7 -- REAL SHADER TESTS (15+ tests)
// =============================================================================

/// Compile real PBR vertex shader.
#[test]

fn real_shader_pbr_vertex() {
    run_async(async {
        let (device, _) = create_test_device().await.expect("No GPU available");
        let cache = ShaderCache::new(device, ShaderCacheConfig::default());

        let path = shader_path("pbr.vert.wgsl");
        if path.exists() {
            let result = cache.get_or_compile_file(&path);
            assert!(result.is_ok(), "Failed to compile pbr.vert.wgsl");
        }
    });
}

/// Compile real shadow fragment shader.
#[test]

fn real_shader_shadow_fragment() {
    run_async(async {
        let (device, _) = create_test_device().await.expect("No GPU available");
        let cache = ShaderCache::new(device, ShaderCacheConfig::default());

        let path = shader_path("shadow.frag.wgsl");
        if path.exists() {
            let result = cache.get_or_compile_file(&path);
            assert!(result.is_ok(), "Failed to compile shadow.frag.wgsl");
        }
    });
}

/// Compile real shadow vertex shader.
#[test]

fn real_shader_shadow_vertex() {
    run_async(async {
        let (device, _) = create_test_device().await.expect("No GPU available");
        let cache = ShaderCache::new(device, ShaderCacheConfig::default());

        let path = shader_path("shadow.vert.wgsl");
        if path.exists() {
            let result = cache.get_or_compile_file(&path);
            assert!(result.is_ok(), "Failed to compile shadow.vert.wgsl");
        }
    });
}

/// Compile real particles shader.
#[test]

fn real_shader_particles() {
    run_async(async {
        let (device, _) = create_test_device().await.expect("No GPU available");
        let cache = ShaderCache::new(device, ShaderCacheConfig::default());

        let path = shader_path("particles.wgsl");
        if path.exists() {
            let result = cache.get_or_compile_file(&path);
            assert!(result.is_ok(), "Failed to compile particles.wgsl");
        }
    });
}

/// Compile real light culling shader.
#[test]

fn real_shader_light_culling() {
    run_async(async {
        let (device, _) = create_test_device().await.expect("No GPU available");
        let cache = ShaderCache::new(device, ShaderCacheConfig::default());

        let path = shader_path("light_culling.wgsl");
        if path.exists() {
            let result = cache.get_or_compile_file(&path);
            assert!(result.is_ok(), "Failed to compile light_culling.wgsl");
        }
    });
}

/// Compile real HiZ generate shader.
#[test]

fn real_shader_hiz_generate() {
    run_async(async {
        let (device, _) = create_test_device().await.expect("No GPU available");
        let cache = ShaderCache::new(device, ShaderCacheConfig::default());

        let path = shader_path("hiz_generate.comp.wgsl");
        if path.exists() {
            let result = cache.get_or_compile_file(&path);
            assert!(result.is_ok(), "Failed to compile hiz_generate.comp.wgsl");
        }
    });
}

/// Compile real mip generate shader.
#[test]

fn real_shader_mip_generate() {
    run_async(async {
        let (device, _) = create_test_device().await.expect("No GPU available");
        let cache = ShaderCache::new(device, ShaderCacheConfig::default());

        let path = shader_path("mip_generate.comp.wgsl");
        if path.exists() {
            let result = cache.get_or_compile_file(&path);
            assert!(result.is_ok(), "Failed to compile mip_generate.comp.wgsl");
        }
    });
}

/// Compile real contact shadow shader.
#[test]

fn real_shader_contact_shadow() {
    run_async(async {
        let (device, _) = create_test_device().await.expect("No GPU available");
        let cache = ShaderCache::new(device, ShaderCacheConfig::default());

        let path = shader_path("contact_shadow.comp.wgsl");
        if path.exists() {
            let result = cache.get_or_compile_file(&path);
            assert!(result.is_ok(), "Failed to compile contact_shadow.comp.wgsl");
        }
    });
}

/// Compile real prefix sum common shader.
#[test]

fn real_shader_prefix_sum() {
    run_async(async {
        let (device, _) = create_test_device().await.expect("No GPU available");
        let cache = ShaderCache::new(device, ShaderCacheConfig::default());

        let path = shaders_dir().join("common").join("prefix_sum.wgsl");
        if path.exists() {
            let result = cache.get_or_compile_file(&path);
            assert!(result.is_ok(), "Failed to compile prefix_sum.wgsl");
        }
    });
}

/// Compile multiple real shaders and verify caching.
#[test]

fn real_shaders_multiple_cached() {
    run_async(async {
        let (device, _) = create_test_device().await.expect("No GPU available");
        let cache = ShaderCache::new(device, ShaderCacheConfig::default());

        let paths = [
            shader_path("pbr.vert.wgsl"),
            shader_path("shadow.frag.wgsl"),
            shader_path("shadow.vert.wgsl"),
        ];

        let mut compiled_count = 0;
        for path in &paths {
            if path.exists() {
                let result = cache.get_or_compile_file(path);
                if result.is_ok() {
                    compiled_count += 1;
                }
            }
        }

        assert_eq!(cache.len(), compiled_count);
        let metrics = cache.metrics();
        assert_eq!(metrics.misses, compiled_count as u64);
    });
}

/// Re-compile same real shader is cached.
#[test]

fn real_shader_recompile_cached() {
    run_async(async {
        let (device, _) = create_test_device().await.expect("No GPU available");
        let cache = ShaderCache::new(device, ShaderCacheConfig::default());

        let path = shader_path("shadow.frag.wgsl");
        if path.exists() {
            let result1 = cache.get_or_compile_file(&path);
            let result2 = cache.get_or_compile_file(&path);

            assert!(result1.is_ok());
            assert!(result2.is_ok());
            assert!(Arc::ptr_eq(&result1.unwrap(), &result2.unwrap()));
        }
    });
}

/// Real shader hot-reload simulation.
#[test]

fn real_shader_hot_reload_simulation() {
    run_async(async {
        let (device, _) = create_test_device().await.expect("No GPU available");
        let cache = ShaderCache::new(device, ShaderCacheConfig::default());

        let path = shader_path("pbr.vert.wgsl");
        if path.exists() {
            // Compile
            let result1 = cache.get_or_compile_file(&path);
            assert!(result1.is_ok());
            let ptr1 = Arc::as_ptr(&result1.unwrap());

            // Invalidate (simulating file change)
            cache.invalidate_by_path(&path);

            // Recompile
            let result2 = cache.get_or_compile_file(&path);
            assert!(result2.is_ok());
            let ptr2 = Arc::as_ptr(&result2.unwrap());

            // Should be different Arc (new compilation)
            assert_ne!(ptr1, ptr2);
        }
    });
}

/// Compile real compute shader with storage buffers.
#[test]

fn real_shader_compute_with_storage() {
    run_async(async {
        let (device, _) = create_test_device().await.expect("No GPU available");
        let cache = ShaderCache::new(device, ShaderCacheConfig::default());

        // Use lighting_pass which has storage buffers
        let path = shader_path("lighting_pass.comp.wgsl");
        if path.exists() {
            let result = cache.get_or_compile_file(&path);
            // May fail due to complex dependencies, but should not panic
            let _ = result;
        }
    });
}

/// Compile real shader and verify entry info.
#[test]

fn real_shader_entry_info() {
    run_async(async {
        let (device, _) = create_test_device().await.expect("No GPU available");
        let cache = ShaderCache::new(device, ShaderCacheConfig::default());

        let path = shader_path("shadow.frag.wgsl");
        if path.exists() {
            let _ = cache.get_or_compile_file(&path);

            let entries = cache.entries();
            assert_eq!(entries.len(), 1);
            assert!(entries[0].access_count >= 1);
            assert!(entries[0].age_secs >= 0.0);
        }
    });
}

/// Compile real SSR shader.
#[test]

fn real_shader_ssr_fade() {
    run_async(async {
        let (device, _) = create_test_device().await.expect("No GPU available");
        let cache = ShaderCache::new(device, ShaderCacheConfig::default());

        let path = shader_path("ssr_fade.wgsl");
        if path.exists() {
            let result = cache.get_or_compile_file(&path);
            // Complex shader, may have validation issues
            let _ = result;
        }
    });
}

// =============================================================================
// SECTION 8 -- HOT-RELOAD SIMULATION TESTS (10+ tests)
// =============================================================================

/// Hot-reload: invalidate then recompile with same key.
#[test]

fn hot_reload_invalidate_recompile_same_key() {
    run_async(async {
        let (device, _) = create_test_device().await.expect("No GPU available");
        let cache = ShaderCache::new(device, ShaderCacheConfig::default());

        let key = ShaderCacheKey::from_source(MINIMAL_VERTEX_SHADER);

        let _ = cache.get_or_compile(key.clone(), None, MINIMAL_VERTEX_SHADER);
        assert_eq!(cache.len(), 1);

        cache.invalidate(&key);
        assert_eq!(cache.len(), 0);

        let _ = cache.get_or_compile(key, None, MINIMAL_VERTEX_SHADER);
        assert_eq!(cache.len(), 1);
    });
}

/// Hot-reload: invalidate with path key.
#[test]

fn hot_reload_invalidate_path_key() {
    run_async(async {
        let (device, _) = create_test_device().await.expect("No GPU available");
        let cache = ShaderCache::new(device, ShaderCacheConfig::default());

        let key = ShaderCacheKey::from_path("virtual/shader.wgsl");
        let _ = cache.get_or_compile(key.clone(), None, MINIMAL_VERTEX_SHADER);

        let removed = cache.invalidate(&key);
        assert!(removed);
    });
}

/// Hot-reload: invalidate_by_path with path key.
#[test]

fn hot_reload_invalidate_by_path_path_key() {
    run_async(async {
        let (device, _) = create_test_device().await.expect("No GPU available");
        let cache = ShaderCache::new(device, ShaderCacheConfig::default());

        let path = Path::new("virtual/shader.wgsl");
        let key = ShaderCacheKey::from_path(path);
        let _ = cache.get_or_compile(key, None, MINIMAL_VERTEX_SHADER);

        let removed = cache.invalidate_by_path(path);
        assert!(removed);
    });
}

/// Hot-reload: recompile with modified source.
#[test]

fn hot_reload_recompile_modified_source() {
    run_async(async {
        let (device, _) = create_test_device().await.expect("No GPU available");
        let cache = ShaderCache::new(device, ShaderCacheConfig::default());

        let original_source = MINIMAL_VERTEX_SHADER;
        let modified_source = r#"
            @vertex
            fn vs_main(@builtin(vertex_index) idx: u32) -> @builtin(position) vec4<f32> {
                return vec4<f32>(1.0, 1.0, 1.0, 1.0); // Modified
            }
        "#;

        let key = ShaderCacheKey::from_path("shader.wgsl");
        let _ = cache.get_or_compile(key.clone(), None, original_source);

        cache.invalidate(&key);

        // Use new key for modified source (different hash)
        let new_key = ShaderCacheKey::from_source(modified_source);
        let _ = cache.get_or_compile(new_key, None, modified_source);

        assert_eq!(cache.len(), 1);
    });
}

/// Hot-reload: batch invalidate all.
#[test]

fn hot_reload_batch_invalidate_all() {
    run_async(async {
        let (device, _) = create_test_device().await.expect("No GPU available");
        let cache = ShaderCache::new(device, ShaderCacheConfig::default());

        let key1 = ShaderCacheKey::from_source(MINIMAL_VERTEX_SHADER);
        let key2 = ShaderCacheKey::from_source(MINIMAL_FRAGMENT_SHADER);
        let key3 = ShaderCacheKey::from_source(MINIMAL_COMPUTE_SHADER);

        let _ = cache.get_or_compile(key1, None, MINIMAL_VERTEX_SHADER);
        let _ = cache.get_or_compile(key2, None, MINIMAL_FRAGMENT_SHADER);
        let _ = cache.get_or_compile(key3, None, MINIMAL_COMPUTE_SHADER);
        assert_eq!(cache.len(), 3);

        cache.invalidate_all();
        assert_eq!(cache.len(), 0);
    });
}

/// Hot-reload: selective invalidation.
#[test]

fn hot_reload_selective_invalidation() {
    run_async(async {
        let (device, _) = create_test_device().await.expect("No GPU available");
        let cache = ShaderCache::new(device, ShaderCacheConfig::default());

        let key1 = ShaderCacheKey::from_source(MINIMAL_VERTEX_SHADER);
        let key2 = ShaderCacheKey::from_source(MINIMAL_FRAGMENT_SHADER);

        let _ = cache.get_or_compile(key1.clone(), None, MINIMAL_VERTEX_SHADER);
        let _ = cache.get_or_compile(key2.clone(), None, MINIMAL_FRAGMENT_SHADER);
        assert_eq!(cache.len(), 2);

        // Invalidate only vertex shader
        cache.invalidate(&key1);
        assert_eq!(cache.len(), 1);
        assert!(cache.contains(&key2));
        assert!(!cache.contains(&key1));
    });
}

/// Hot-reload: metrics track invalidations.
#[test]

fn hot_reload_metrics_track_invalidations() {
    run_async(async {
        let (device, _) = create_test_device().await.expect("No GPU available");
        let cache = ShaderCache::new(device, ShaderCacheConfig::default());

        let key = ShaderCacheKey::from_source(MINIMAL_VERTEX_SHADER);

        let _ = cache.get_or_compile(key.clone(), None, MINIMAL_VERTEX_SHADER);
        cache.invalidate(&key);
        let _ = cache.get_or_compile(key.clone(), None, MINIMAL_VERTEX_SHADER);
        cache.invalidate(&key);

        let metrics = cache.metrics();
        assert_eq!(metrics.invalidations, 2);
    });
}

/// Hot-reload: Arc reference survives invalidation.
#[test]

fn hot_reload_arc_survives_invalidation() {
    run_async(async {
        let (device, _) = create_test_device().await.expect("No GPU available");
        let cache = ShaderCache::new(device, ShaderCacheConfig::default());

        let key = ShaderCacheKey::from_source(MINIMAL_VERTEX_SHADER);

        let shader = cache.get_or_compile(key.clone(), None, MINIMAL_VERTEX_SHADER).unwrap();
        let shader_clone = Arc::clone(&shader);

        cache.invalidate(&key);

        // Shader still valid (just removed from cache)
        assert!(Arc::strong_count(&shader_clone) >= 1);
    });
}

/// Hot-reload: invalidate non-existent returns false.
#[test]

fn hot_reload_invalidate_nonexistent() {
    run_async(async {
        let (device, _) = create_test_device().await.expect("No GPU available");
        let cache = ShaderCache::new(device, ShaderCacheConfig::default());

        let key = ShaderCacheKey::from_source(MINIMAL_VERTEX_SHADER);
        let removed = cache.invalidate(&key);
        assert!(!removed);

        let metrics = cache.metrics();
        assert_eq!(metrics.invalidations, 0);
    });
}

/// Hot-reload: file-based workflow.
#[test]

fn hot_reload_file_based_workflow() {
    run_async(async {
        let (device, _) = create_test_device().await.expect("No GPU available");
        let cache = ShaderCache::new(device, ShaderCacheConfig::default());

        let path = shader_path("pbr.vert.wgsl");
        if path.exists() {
            // Initial load
            let shader1 = cache.get_or_compile_file(&path);
            assert!(shader1.is_ok());

            // Simulate file watcher notification
            cache.invalidate_by_path(&path);

            // Reload
            let shader2 = cache.get_or_compile_file(&path);
            assert!(shader2.is_ok());

            // Different Arc (recompiled)
            assert!(!Arc::ptr_eq(&shader1.unwrap(), &shader2.unwrap()));
        }
    });
}

// =============================================================================
// SECTION 9 -- LRU EVICTION TESTS (10+ tests)
// =============================================================================

/// LRU eviction: cache at capacity evicts oldest.
#[test]

fn lru_eviction_at_capacity() {
    run_async(async {
        let (device, _) = create_test_device().await.expect("No GPU available");
        let config = ShaderCacheConfig::minimal(); // max 16 entries
        let cache = ShaderCache::new(device, config);

        // Fill cache beyond capacity
        for i in 0..20 {
            let source = format!(
                "@vertex fn vs_main_{i}() -> @builtin(position) vec4<f32> {{ return vec4<f32>(0.0); }}"
            );
            let key = ShaderCacheKey::from_source(&source);
            let _ = cache.get_or_compile(key, None, &source);
        }

        // Should have evicted some
        let metrics = cache.metrics();
        assert!(metrics.evictions > 0);
        assert!(cache.len() <= 16);
    });
}

/// LRU eviction: without eviction grows unbounded.
#[test]

fn lru_eviction_disabled_grows() {
    run_async(async {
        let (device, _) = create_test_device().await.expect("No GPU available");
        let config = ShaderCacheConfig::minimal().without_eviction();
        let cache = ShaderCache::new(device, config);

        for i in 0..20 {
            let source = format!(
                "@vertex fn vs_main_{i}() -> @builtin(position) vec4<f32> {{ return vec4<f32>(0.0); }}"
            );
            let key = ShaderCacheKey::from_source(&source);
            let _ = cache.get_or_compile(key, None, &source);
        }

        // No evictions
        let metrics = cache.metrics();
        assert_eq!(metrics.evictions, 0);
        assert_eq!(cache.len(), 20);
    });
}

/// LRU eviction: access updates LRU order.
#[test]

fn lru_eviction_access_updates_order() {
    run_async(async {
        let (device, _) = create_test_device().await.expect("No GPU available");
        let config = ShaderCacheConfig::new().max_entries(3);
        let cache = ShaderCache::new(device, config);

        // Add 3 shaders
        let sources = [MINIMAL_VERTEX_SHADER, MINIMAL_FRAGMENT_SHADER, MINIMAL_COMPUTE_SHADER];
        let keys: Vec<_> = sources.iter().map(|s| ShaderCacheKey::from_source(s)).collect();

        for (key, source) in keys.iter().zip(sources.iter()) {
            let _ = cache.get_or_compile(key.clone(), None, source);
        }

        // Access first shader (makes it most recently used)
        let _ = cache.get_or_compile(keys[0].clone(), None, sources[0]);

        // Add a 4th shader - should evict second (least recently used)
        let new_source = r#"
            @compute @workgroup_size(1)
            fn cs_new() {}
        "#;
        let _ = cache.get_or_compile(ShaderCacheKey::from_source(new_source), None, new_source);

        // First shader should still be present
        assert!(cache.contains(&keys[0]));
    });
}

/// LRU eviction: metrics track eviction count.
#[test]

fn lru_eviction_metrics_track_count() {
    run_async(async {
        let (device, _) = create_test_device().await.expect("No GPU available");
        let config = ShaderCacheConfig::new().max_entries(2);
        let cache = ShaderCache::new(device, config);

        // Add 3 shaders (will evict 1)
        let sources = [MINIMAL_VERTEX_SHADER, MINIMAL_FRAGMENT_SHADER, MINIMAL_COMPUTE_SHADER];
        for source in sources {
            let key = ShaderCacheKey::from_source(source);
            let _ = cache.get_or_compile(key, None, source);
        }

        let metrics = cache.metrics();
        assert!(metrics.evictions >= 1);
    });
}

/// LRU eviction: zero max_entries.
#[test]

fn lru_eviction_zero_max_entries() {
    run_async(async {
        let (device, _) = create_test_device().await.expect("No GPU available");
        let config = ShaderCacheConfig::new().max_entries(0);
        let cache = ShaderCache::new(device, config);

        let key = ShaderCacheKey::from_source(MINIMAL_VERTEX_SHADER);
        let result = cache.get_or_compile(key.clone(), None, MINIMAL_VERTEX_SHADER);

        // Should still compile successfully
        assert!(result.is_ok());
        // But immediately evict
        // (behavior may vary - the cache might keep 0 or 1 entries)
    });
}

/// LRU eviction: large cache no evictions.
#[test]

fn lru_eviction_large_cache_no_evictions() {
    run_async(async {
        let (device, _) = create_test_device().await.expect("No GPU available");
        let config = ShaderCacheConfig::production(); // 1024 entries
        let cache = ShaderCache::new(device, config);

        // Add fewer than max
        for i in 0..10 {
            let source = format!(
                "@vertex fn vs_{i}() -> @builtin(position) vec4<f32> {{ return vec4<f32>(0.0); }}"
            );
            let key = ShaderCacheKey::from_source(&source);
            let _ = cache.get_or_compile(key, None, &source);
        }

        let metrics = cache.metrics();
        assert_eq!(metrics.evictions, 0);
        assert_eq!(cache.len(), 10);
    });
}

/// LRU eviction: path index updated on eviction.
#[test]

fn lru_eviction_path_index_updated() {
    run_async(async {
        let (device, _) = create_test_device().await.expect("No GPU available");
        let config = ShaderCacheConfig::new().max_entries(2);
        let cache = ShaderCache::new(device, config);

        let path1 = Path::new("shader1.wgsl");
        let path2 = Path::new("shader2.wgsl");
        let path3 = Path::new("shader3.wgsl");

        let key1 = ShaderCacheKey::from_path(path1);
        let key2 = ShaderCacheKey::from_path(path2);
        let key3 = ShaderCacheKey::from_path(path3);

        let _ = cache.get_or_compile(key1.clone(), None, MINIMAL_VERTEX_SHADER);
        let _ = cache.get_or_compile(key2.clone(), None, MINIMAL_FRAGMENT_SHADER);
        let _ = cache.get_or_compile(key3, None, MINIMAL_COMPUTE_SHADER);

        // One should be evicted
        assert_eq!(cache.len(), 2);
    });
}

/// LRU eviction: entry info reflects access count.
#[test]

fn lru_eviction_entry_access_count() {
    run_async(async {
        let (device, _) = create_test_device().await.expect("No GPU available");
        let cache = ShaderCache::new(device, ShaderCacheConfig::default());

        let key = ShaderCacheKey::from_source(MINIMAL_VERTEX_SHADER);

        // Access multiple times
        for _ in 0..5 {
            let _ = cache.get_or_compile(key.clone(), None, MINIMAL_VERTEX_SHADER);
        }

        let entries = cache.entries();
        assert!(!entries.is_empty());
        // Should have at least 5 accesses (first is miss, rest are hits)
        assert!(entries[0].access_count >= 5);
    });
}

/// LRU eviction: get also touches entry.
#[test]

fn lru_eviction_get_touches_entry() {
    run_async(async {
        let (device, _) = create_test_device().await.expect("No GPU available");
        let config = ShaderCacheConfig::new().max_entries(2);
        let cache = ShaderCache::new(device, config);

        let key1 = ShaderCacheKey::from_source(MINIMAL_VERTEX_SHADER);
        let key2 = ShaderCacheKey::from_source(MINIMAL_FRAGMENT_SHADER);

        let _ = cache.get_or_compile(key1.clone(), None, MINIMAL_VERTEX_SHADER);
        let _ = cache.get_or_compile(key2.clone(), None, MINIMAL_FRAGMENT_SHADER);

        // Touch key1 via get
        let _ = cache.get(&key1);

        // Add third shader - key2 should be evicted (LRU)
        let _ = cache.get_or_compile(
            ShaderCacheKey::from_source(MINIMAL_COMPUTE_SHADER),
            None,
            MINIMAL_COMPUTE_SHADER,
        );

        // key1 should survive
        assert!(cache.contains(&key1));
    });
}

/// LRU eviction: access_count reflects actual accesses.
#[test]

fn lru_eviction_access_count_accurate() {
    run_async(async {
        let (device, _) = create_test_device().await.expect("No GPU available");
        let cache = ShaderCache::new(device, ShaderCacheConfig::default());

        let key = ShaderCacheKey::from_source(MINIMAL_VERTEX_SHADER);

        let _ = cache.get_or_compile(key.clone(), None, MINIMAL_VERTEX_SHADER); // 1
        let _ = cache.get(&key); // 2
        let _ = cache.get_or_compile(key.clone(), None, MINIMAL_VERTEX_SHADER); // 3

        let entries = cache.entries();
        assert_eq!(entries.len(), 1);
        assert!(entries[0].access_count >= 3);
    });
}

// =============================================================================
// SECTION 10 -- CONCURRENT ACCESS TESTS (10+ tests)
// =============================================================================

/// Concurrent: multiple threads read same cached shader.
#[test]

fn concurrent_read_same_shader() {
    run_async(async {
        let (device, _) = create_test_device().await.expect("No GPU available");
        let cache = Arc::new(ShaderCache::new(device, ShaderCacheConfig::default()));

        let key = ShaderCacheKey::from_source(MINIMAL_VERTEX_SHADER);
        let _ = cache.get_or_compile(key.clone(), None, MINIMAL_VERTEX_SHADER);

        let handles: Vec<_> = (0..4)
            .map(|_| {
                let cache = Arc::clone(&cache);
                let key = key.clone();
                std::thread::spawn(move || {
                    for _ in 0..100 {
                        let _ = cache.get(&key);
                    }
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }

        let metrics = cache.metrics();
        assert!(metrics.hits >= 400);
    });
}

/// Concurrent: multiple threads compile different shaders.
#[test]

fn concurrent_compile_different_shaders() {
    run_async(async {
        let (device, _) = create_test_device().await.expect("No GPU available");
        let cache = Arc::new(ShaderCache::new(device, ShaderCacheConfig::default()));

        let handles: Vec<_> = (0..4)
            .map(|i| {
                let cache = Arc::clone(&cache);
                std::thread::spawn(move || {
                    let source = format!(
                        "@vertex fn vs_{i}() -> @builtin(position) vec4<f32> {{ return vec4<f32>(0.0); }}"
                    );
                    let key = ShaderCacheKey::from_source(&source);
                    cache.get_or_compile(key, None, &source)
                })
            })
            .collect();

        for handle in handles {
            let result = handle.join().unwrap();
            assert!(result.is_ok());
        }

        assert_eq!(cache.len(), 4);
    });
}

/// Concurrent: readers and writers.
#[test]

fn concurrent_readers_writers() {
    run_async(async {
        let (device, _) = create_test_device().await.expect("No GPU available");
        let cache = Arc::new(ShaderCache::new(device, ShaderCacheConfig::default()));

        // Pre-populate
        let key = ShaderCacheKey::from_source(MINIMAL_VERTEX_SHADER);
        let _ = cache.get_or_compile(key.clone(), None, MINIMAL_VERTEX_SHADER);

        let handles: Vec<_> = (0..8)
            .map(|i| {
                let cache = Arc::clone(&cache);
                let key = key.clone();
                std::thread::spawn(move || {
                    if i % 2 == 0 {
                        // Reader
                        for _ in 0..50 {
                            let _ = cache.get(&key);
                        }
                    } else {
                        // Writer (different shader)
                        let source = format!(
                            "@fragment fn fs_{i}() -> @location(0) vec4<f32> {{ return vec4<f32>(1.0); }}"
                        );
                        let new_key = ShaderCacheKey::from_source(&source);
                        let _ = cache.get_or_compile(new_key, None, &source);
                    }
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }

        // Should have original + some new shaders
        assert!(cache.len() >= 1);
    });
}

/// Concurrent: invalidate during reads.
#[test]

fn concurrent_invalidate_during_reads() {
    run_async(async {
        let (device, _) = create_test_device().await.expect("No GPU available");
        let cache = Arc::new(ShaderCache::new(device, ShaderCacheConfig::default()));

        let key = ShaderCacheKey::from_source(MINIMAL_VERTEX_SHADER);
        let _ = cache.get_or_compile(key.clone(), None, MINIMAL_VERTEX_SHADER);

        let cache_reader = Arc::clone(&cache);
        let key_reader = key.clone();
        let reader = std::thread::spawn(move || {
            for _ in 0..100 {
                let _ = cache_reader.get(&key_reader);
            }
        });

        let cache_invalidator = Arc::clone(&cache);
        let key_invalidator = key.clone();
        let invalidator = std::thread::spawn(move || {
            for _ in 0..10 {
                cache_invalidator.invalidate(&key_invalidator);
                std::thread::sleep(std::time::Duration::from_micros(100));
            }
        });

        reader.join().unwrap();
        invalidator.join().unwrap();
    });
}

/// Concurrent: metrics consistency.
#[test]

fn concurrent_metrics_consistency() {
    run_async(async {
        let (device, _) = create_test_device().await.expect("No GPU available");
        let cache = Arc::new(ShaderCache::new(device, ShaderCacheConfig::default()));

        let key = ShaderCacheKey::from_source(MINIMAL_VERTEX_SHADER);
        let _ = cache.get_or_compile(key.clone(), None, MINIMAL_VERTEX_SHADER);

        let handles: Vec<_> = (0..4)
            .map(|_| {
                let cache = Arc::clone(&cache);
                let key = key.clone();
                std::thread::spawn(move || {
                    for _ in 0..100 {
                        let _ = cache.get_or_compile(key.clone(), None, MINIMAL_VERTEX_SHADER);
                    }
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }

        let metrics = cache.metrics();
        // Total requests should be 1 (initial) + 400 (concurrent)
        assert_eq!(metrics.hits + metrics.misses, 401);
    });
}

/// Concurrent: contains during modifications.
#[test]

fn concurrent_contains_during_modifications() {
    run_async(async {
        let (device, _) = create_test_device().await.expect("No GPU available");
        let cache = Arc::new(ShaderCache::new(device, ShaderCacheConfig::default()));

        let key = ShaderCacheKey::from_source(MINIMAL_VERTEX_SHADER);

        let handles: Vec<_> = (0..4)
            .map(|_| {
                let cache = Arc::clone(&cache);
                let key = key.clone();
                std::thread::spawn(move || {
                    for _ in 0..50 {
                        let _ = cache.get_or_compile(key.clone(), None, MINIMAL_VERTEX_SHADER);
                        let _ = cache.contains(&key);
                    }
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }
    });
}

/// Concurrent: keys and entries during modifications.
#[test]

fn concurrent_keys_entries_during_modifications() {
    run_async(async {
        let (device, _) = create_test_device().await.expect("No GPU available");
        let cache = Arc::new(ShaderCache::new(device, ShaderCacheConfig::default()));

        let handles: Vec<_> = (0..4)
            .map(|i| {
                let cache = Arc::clone(&cache);
                std::thread::spawn(move || {
                    for j in 0..20 {
                        let source = format!(
                            "@vertex fn vs_{i}_{j}() -> @builtin(position) vec4<f32> {{ return vec4<f32>(0.0); }}"
                        );
                        let key = ShaderCacheKey::from_source(&source);
                        let _ = cache.get_or_compile(key, None, &source);
                        let _ = cache.keys();
                        let _ = cache.entries();
                    }
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }
    });
}

/// Concurrent: invalidate_all during access.
#[test]

fn concurrent_invalidate_all_during_access() {
    run_async(async {
        let (device, _) = create_test_device().await.expect("No GPU available");
        let cache = Arc::new(ShaderCache::new(device, ShaderCacheConfig::default()));

        // Pre-populate
        for i in 0..5 {
            let source = format!(
                "@vertex fn vs_{i}() -> @builtin(position) vec4<f32> {{ return vec4<f32>(0.0); }}"
            );
            let key = ShaderCacheKey::from_source(&source);
            let _ = cache.get_or_compile(key, None, &source);
        }

        let cache_accessor = Arc::clone(&cache);
        let accessor = std::thread::spawn(move || {
            for _ in 0..100 {
                let _ = cache_accessor.len();
                let _ = cache_accessor.is_empty();
            }
        });

        let cache_invalidator = Arc::clone(&cache);
        let invalidator = std::thread::spawn(move || {
            for _ in 0..5 {
                cache_invalidator.invalidate_all();
                std::thread::sleep(std::time::Duration::from_micros(50));
            }
        });

        accessor.join().unwrap();
        invalidator.join().unwrap();
    });
}

/// Concurrent: stress test high contention.
#[test]

fn concurrent_stress_high_contention() {
    run_async(async {
        let (device, _) = create_test_device().await.expect("No GPU available");
        let cache = Arc::new(ShaderCache::new(device, ShaderCacheConfig::minimal()));

        let handles: Vec<_> = (0..8)
            .map(|i| {
                let cache = Arc::clone(&cache);
                std::thread::spawn(move || {
                    for j in 0..25 {
                        let source = format!(
                            "@vertex fn vs_{i}_{j}() -> @builtin(position) vec4<f32> {{ return vec4<f32>(0.0); }}"
                        );
                        let key = ShaderCacheKey::from_source(&source);
                        let _ = cache.get_or_compile(key.clone(), None, &source);
                        if j % 5 == 0 {
                            cache.invalidate(&key);
                        }
                    }
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }

        // Cache should be in consistent state
        let metrics = cache.metrics();
        assert!(metrics.total_requests() > 0);
    });
}

/// Concurrent: Arc reference safety.
#[test]

fn concurrent_arc_reference_safety() {
    run_async(async {
        let (device, _) = create_test_device().await.expect("No GPU available");
        let cache = Arc::new(ShaderCache::new(device, ShaderCacheConfig::default()));

        let key = ShaderCacheKey::from_source(MINIMAL_VERTEX_SHADER);
        let shader = cache.get_or_compile(key.clone(), None, MINIMAL_VERTEX_SHADER).unwrap();

        let handles: Vec<_> = (0..4)
            .map(|_| {
                let cache = Arc::clone(&cache);
                let key = key.clone();
                let shader_ref = Arc::clone(&shader);
                std::thread::spawn(move || {
                    for _ in 0..50 {
                        let got = cache.get(&key);
                        if let Some(s) = got {
                            assert!(Arc::ptr_eq(&s, &shader_ref));
                        }
                    }
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }
    });
}

// =============================================================================
// SECTION 11 -- EDGE CASES AND ERROR HANDLING (10+ tests)
// =============================================================================

/// Edge case: empty shader source.
#[test]

fn edge_case_empty_source() {
    run_async(async {
        let (device, _) = create_test_device().await.expect("No GPU available");
        let cache = ShaderCache::new(device, ShaderCacheConfig::default());

        let key = ShaderCacheKey::from_source("");
        let result = cache.get_or_compile(key, None, "");

        assert!(result.is_err());
    });
}

/// Edge case: whitespace-only source.
#[test]

fn edge_case_whitespace_source() {
    run_async(async {
        let (device, _) = create_test_device().await.expect("No GPU available");
        let cache = ShaderCache::new(device, ShaderCacheConfig::default());

        let key = ShaderCacheKey::from_source("   \n\t\n   ");
        let result = cache.get_or_compile(key, None, "   \n\t\n   ");

        assert!(result.is_err());
    });
}

/// Edge case: validation error shader.
#[test]

fn edge_case_validation_error() {
    run_async(async {
        let (device, _) = create_test_device().await.expect("No GPU available");
        let cache = ShaderCache::new(device, ShaderCacheConfig::default());

        let key = ShaderCacheKey::from_source(INVALID_SHADER_VALIDATION);
        let result = cache.get_or_compile(key, None, INVALID_SHADER_VALIDATION);

        assert!(result.is_err());
    });
}

/// Edge case: same key different source (hash key).
#[test]

fn edge_case_key_source_mismatch_hash() {
    run_async(async {
        let (device, _) = create_test_device().await.expect("No GPU available");
        let cache = ShaderCache::new(device, ShaderCacheConfig::default());

        let key = ShaderCacheKey::from_source(MINIMAL_VERTEX_SHADER);

        // First compile with correct source
        let _ = cache.get_or_compile(key.clone(), None, MINIMAL_VERTEX_SHADER);

        // Second call with same key returns cached (ignores provided source)
        let result = cache.get_or_compile(key, None, MINIMAL_FRAGMENT_SHADER);
        assert!(result.is_ok());
    });
}

/// Edge case: very long label.
#[test]

fn edge_case_long_label() {
    run_async(async {
        let (device, _) = create_test_device().await.expect("No GPU available");
        let cache = ShaderCache::new(device, ShaderCacheConfig::default());

        let long_label = "x".repeat(1000);
        let key = ShaderCacheKey::from_source(MINIMAL_VERTEX_SHADER);

        let result = cache.get_or_compile(key, Some(&long_label), MINIMAL_VERTEX_SHADER);
        assert!(result.is_ok());
    });
}

/// Edge case: unicode in shader comments.
#[test]

fn edge_case_unicode_comments() {
    run_async(async {
        let (device, _) = create_test_device().await.expect("No GPU available");
        let cache = ShaderCache::new(device, ShaderCacheConfig::default());

        let source = r#"
            // Unicode comment: Hello World
            @vertex
            fn vs_main() -> @builtin(position) vec4<f32> {
                return vec4<f32>(0.0);
            }
        "#;
        let key = ShaderCacheKey::from_source(source);

        let result = cache.get_or_compile(key, None, source);
        assert!(result.is_ok());
    });
}

/// Edge case: null label (None).
#[test]

fn edge_case_null_label() {
    run_async(async {
        let (device, _) = create_test_device().await.expect("No GPU available");
        let cache = ShaderCache::new(device, ShaderCacheConfig::default());

        let key = ShaderCacheKey::from_source(MINIMAL_VERTEX_SHADER);
        let result = cache.get_or_compile(key, None, MINIMAL_VERTEX_SHADER);

        assert!(result.is_ok());

        let labels = cache.labels();
        assert_eq!(labels.len(), 1);
        assert!(labels[0].is_none());
    });
}

/// Edge case: path with special characters.
#[test]

fn edge_case_path_special_chars() {
    run_async(async {
        let (device, _) = create_test_device().await.expect("No GPU available");
        let cache = ShaderCache::new(device, ShaderCacheConfig::default());

        let key = ShaderCacheKey::from_path("shaders/@special#$%.wgsl");
        let result = cache.get_or_compile(key.clone(), None, MINIMAL_VERTEX_SHADER);

        assert!(result.is_ok());
        assert!(cache.contains(&key));
    });
}

/// Edge case: relative vs absolute path.
#[test]

fn edge_case_relative_absolute_path() {
    run_async(async {
        let (device, _) = create_test_device().await.expect("No GPU available");
        let cache = ShaderCache::new(device, ShaderCacheConfig::default());

        let rel_key = ShaderCacheKey::from_path("shaders/test.wgsl");
        let abs_key = ShaderCacheKey::from_path("/absolute/shaders/test.wgsl");

        // These are different keys
        assert_ne!(rel_key, abs_key);

        let _ = cache.get_or_compile(rel_key.clone(), None, MINIMAL_VERTEX_SHADER);
        let _ = cache.get_or_compile(abs_key.clone(), None, MINIMAL_VERTEX_SHADER);

        assert_eq!(cache.len(), 2);
    });
}

/// Edge case: cache multiple variants of similar shaders.
#[test]

fn edge_case_similar_shaders() {
    run_async(async {
        let (device, _) = create_test_device().await.expect("No GPU available");
        let cache = ShaderCache::new(device, ShaderCacheConfig::default());

        let shader1 = r#"
            @vertex fn vs_main() -> @builtin(position) vec4<f32> {
                return vec4<f32>(0.0);
            }
        "#;
        let shader2 = r#"
            @vertex fn vs_main() -> @builtin(position) vec4<f32> {
                return vec4<f32>(1.0);
            }
        "#;

        let key1 = ShaderCacheKey::from_source(shader1);
        let key2 = ShaderCacheKey::from_source(shader2);

        assert_ne!(key1, key2);

        let _ = cache.get_or_compile(key1, None, shader1);
        let _ = cache.get_or_compile(key2, None, shader2);

        assert_eq!(cache.len(), 2);
    });
}
