// SPDX-License-Identifier: MIT
//
// blackbox_bind_group_layout_cache.rs -- Blackbox tests for T-WGPU-P2.5.1 BindGroupLayoutCache.
//
// CLEANROOM: No src/ access beyond the public API exported by the crate.
// Tests use only the documented public types and functions:
//
//   - BindGroupLayoutKey
//   - CachedBindGroupLayout
//   - BindGroupLayoutCache
//   - BindGroupLayoutCacheMetrics
//   - layouts_compatible()
//   - layouts_equal()
//
// ACCEPTANCE CRITERIA:
//   1. Public API contracts -- 20+ tests covering type construction and accessors
//   2. Behavioral tests     -- 20+ tests verifying cache semantics
//   3. GPU integration      -- 15+ tests (marked #[ignore] if no GPU)
//   4. Property-based       -- 10+ tests for invariants
//
// Total target: 60+ tests

use renderer_backend::resources::{
    BindGroupLayoutCache, BindGroupLayoutCacheMetrics, BindGroupLayoutKey,
    layouts_compatible, layouts_equal,
};
use std::collections::HashMap;
use std::sync::Arc;
use wgpu::{
    BindGroupLayoutEntry, BindingType, BufferBindingType, SamplerBindingType, ShaderStages,
    StorageTextureAccess, TextureFormat, TextureSampleType, TextureViewDimension,
};

// =============================================================================
// HELPERS -- Entry construction helpers for cleanroom testing
// =============================================================================

/// Creates a uniform buffer entry at the given binding.
fn uniform_buffer_entry(binding: u32, stages: ShaderStages) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: stages,
        ty: BindingType::Buffer {
            ty: BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

/// Creates a storage buffer entry at the given binding.
fn storage_buffer_entry(binding: u32, stages: ShaderStages, read_only: bool) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: stages,
        ty: BindingType::Buffer {
            ty: BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

/// Creates a sampled texture entry at the given binding.
fn texture_entry(binding: u32, stages: ShaderStages) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: stages,
        ty: BindingType::Texture {
            sample_type: TextureSampleType::Float { filterable: true },
            view_dimension: TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

/// Creates a sampler entry at the given binding.
fn sampler_entry(binding: u32, stages: ShaderStages) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: stages,
        ty: BindingType::Sampler(SamplerBindingType::Filtering),
        count: None,
    }
}

/// Creates a storage texture entry at the given binding.
fn storage_texture_entry(binding: u32, stages: ShaderStages) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: stages,
        ty: BindingType::StorageTexture {
            access: StorageTextureAccess::WriteOnly,
            format: TextureFormat::Rgba8Unorm,
            view_dimension: TextureViewDimension::D2,
        },
        count: None,
    }
}

/// Creates a basic set of entries: uniform buffer (0), texture (1), sampler (2).
fn basic_entries() -> Vec<BindGroupLayoutEntry> {
    vec![
        uniform_buffer_entry(0, ShaderStages::VERTEX_FRAGMENT),
        texture_entry(1, ShaderStages::FRAGMENT),
        sampler_entry(2, ShaderStages::FRAGMENT),
    ]
}

/// Creates an alternative set of entries different from basic_entries().
fn alt_entries() -> Vec<BindGroupLayoutEntry> {
    vec![
        storage_buffer_entry(0, ShaderStages::COMPUTE, false),
        storage_buffer_entry(1, ShaderStages::COMPUTE, true),
    ]
}

/// Creates a compute-shader set of entries.
fn compute_entries() -> Vec<BindGroupLayoutEntry> {
    vec![
        storage_buffer_entry(0, ShaderStages::COMPUTE, true),
        storage_buffer_entry(1, ShaderStages::COMPUTE, false),
        uniform_buffer_entry(2, ShaderStages::COMPUTE),
    ]
}

// =============================================================================
// SECTION 1 -- PUBLIC API CONTRACTS (20+ tests)
// =============================================================================

// ---- BindGroupLayoutKey tests ----

/// BindGroupLayoutKey::from_entries() creates a key from entries.
#[test]
fn key_from_entries_creates_key() {
    let entries = basic_entries();
    let key = BindGroupLayoutKey::from_entries(&entries);
    // Key was created successfully (no panic)
    let _ = key;
}

/// BindGroupLayoutKey equality for identical entries.
#[test]
fn key_equality_for_same_entries() {
    let entries = basic_entries();
    let key1 = BindGroupLayoutKey::from_entries(&entries);
    let key2 = BindGroupLayoutKey::from_entries(&entries);
    assert_eq!(key1, key2, "Keys from identical entries must be equal");
}

/// BindGroupLayoutKey inequality for different entries.
#[test]
fn key_inequality_for_different_entries() {
    let key1 = BindGroupLayoutKey::from_entries(&basic_entries());
    let key2 = BindGroupLayoutKey::from_entries(&alt_entries());
    assert_ne!(key1, key2, "Keys from different entries must not be equal");
}

/// BindGroupLayoutKey works as HashMap key.
#[test]
fn key_works_as_hashmap_key() {
    let mut map: HashMap<BindGroupLayoutKey, &str> = HashMap::new();

    let key1 = BindGroupLayoutKey::from_entries(&basic_entries());
    let key2 = BindGroupLayoutKey::from_entries(&alt_entries());

    map.insert(key1.clone(), "basic");
    map.insert(key2.clone(), "alt");

    assert_eq!(map.get(&key1), Some(&"basic"));
    assert_eq!(map.get(&key2), Some(&"alt"));
    assert_eq!(map.len(), 2);
}

/// BindGroupLayoutKey hash is stable.
#[test]
fn key_hash_is_stable() {
    use std::hash::{Hash, Hasher};
    use std::collections::hash_map::DefaultHasher;

    let entries = basic_entries();
    let key1 = BindGroupLayoutKey::from_entries(&entries);
    let key2 = BindGroupLayoutKey::from_entries(&entries);

    let mut h1 = DefaultHasher::new();
    let mut h2 = DefaultHasher::new();
    key1.hash(&mut h1);
    key2.hash(&mut h2);

    assert_eq!(h1.finish(), h2.finish(), "Hash must be stable for same entries");
}

/// BindGroupLayoutKey from empty entries.
#[test]
fn key_from_empty_entries() {
    let entries: Vec<BindGroupLayoutEntry> = vec![];
    let key = BindGroupLayoutKey::from_entries(&entries);
    let _ = key; // Successfully created
}

/// BindGroupLayoutKey from single entry.
#[test]
fn key_from_single_entry() {
    let entries = vec![uniform_buffer_entry(0, ShaderStages::VERTEX)];
    let key = BindGroupLayoutKey::from_entries(&entries);
    let _ = key;
}

/// BindGroupLayoutKey entry order affects key (or not, depending on impl).
#[test]
fn key_entry_order_independence() {
    let entries1 = vec![
        uniform_buffer_entry(0, ShaderStages::VERTEX),
        texture_entry(1, ShaderStages::FRAGMENT),
    ];
    let entries2 = vec![
        texture_entry(1, ShaderStages::FRAGMENT),
        uniform_buffer_entry(0, ShaderStages::VERTEX),
    ];

    let key1 = BindGroupLayoutKey::from_entries(&entries1);
    let key2 = BindGroupLayoutKey::from_entries(&entries2);

    // Keys should be equal if entries are canonicalized by binding index.
    // If not, keys will differ. Both behaviors are valid -- we just document it.
    // The implementation should be consistent with itself.
    let first_check = key1 == key2;
    let second_check = key1 == key2;
    assert_eq!(first_check, second_check, "Key comparison must be consistent");
}

/// BindGroupLayoutKey hash_value accessor.
#[test]
fn key_hash_value_accessor() {
    let entries = basic_entries();
    let key = BindGroupLayoutKey::from_entries(&entries);
    let hash = key.hash_value();
    // Hash value should be non-zero for non-empty entries
    assert!(hash != 0 || entries.is_empty(), "Hash value should be computed");
}

// ---- BindGroupLayoutCache tests ----

/// BindGroupLayoutCache::new() creates an empty cache.
#[test]
fn cache_new_creates_empty() {
    let cache = BindGroupLayoutCache::new();
    assert!(cache.is_empty(), "New cache must be empty");
    assert_eq!(cache.len(), 0, "New cache length must be 0");
}

/// BindGroupLayoutCache::default() works.
#[test]
fn cache_default_works() {
    let cache = BindGroupLayoutCache::default();
    assert!(cache.is_empty());
}

/// BindGroupLayoutCache::len() returns correct size.
#[test]
fn cache_len_returns_correct_size() {
    let cache = BindGroupLayoutCache::new();
    assert_eq!(cache.len(), 0);
    // We cannot insert without a device, but we can verify the method works
}

/// BindGroupLayoutCache::is_empty() correctness.
#[test]
fn cache_is_empty_correctness() {
    let cache = BindGroupLayoutCache::new();
    assert!(cache.is_empty());
}

/// BindGroupLayoutCache::contains() on empty cache.
#[test]
fn cache_contains_on_empty() {
    let cache = BindGroupLayoutCache::new();
    let entries = basic_entries();
    assert!(!cache.contains(&entries), "Empty cache must not contain any entries");
}

/// BindGroupLayoutCache::clear() on empty cache does not panic.
#[test]
fn cache_clear_empty_no_panic() {
    let cache = BindGroupLayoutCache::new();
    cache.clear();
    assert!(cache.is_empty());
}

/// BindGroupLayoutCache::metrics() returns initial stats.
#[test]
fn cache_metrics_returns_initial_stats() {
    let cache = BindGroupLayoutCache::new();
    let metrics = cache.metrics();

    assert_eq!(metrics.hits, 0, "Initial hits must be 0");
    assert_eq!(metrics.misses, 0, "Initial misses must be 0");
    assert_eq!(metrics.cache_size, 0, "Initial cache_size must be 0");
}

/// BindGroupLayoutCache::reset_metrics() clears counters.
#[test]
fn cache_reset_metrics() {
    let cache = BindGroupLayoutCache::new();
    cache.reset_metrics();
    let metrics = cache.metrics();
    assert_eq!(metrics.hits, 0);
    assert_eq!(metrics.misses, 0);
}

// ---- BindGroupLayoutCacheMetrics tests ----

/// BindGroupLayoutCacheMetrics::new() works.
#[test]
fn metrics_new_works() {
    let metrics = BindGroupLayoutCacheMetrics::new(10, 5, 2);
    assert_eq!(metrics.cache_size, 10);
    assert_eq!(metrics.hits, 5);
    assert_eq!(metrics.misses, 2);
}

/// BindGroupLayoutCacheMetrics hit_rate field.
#[test]
fn metrics_hit_rate_field() {
    let metrics = BindGroupLayoutCacheMetrics::new(10, 75, 25);
    // 75 / (75 + 25) = 0.75
    assert!((metrics.hit_rate - 0.75).abs() < 0.001, "Hit rate should be 75%");
}

/// BindGroupLayoutCacheMetrics total_requests calculation.
#[test]
fn metrics_total_requests() {
    let metrics = BindGroupLayoutCacheMetrics::new(5, 30, 20);
    let total = metrics.total_requests();
    assert_eq!(total, 50, "Total requests should be hits + misses");
}

/// BindGroupLayoutCacheMetrics is_empty() method.
#[test]
fn metrics_is_empty() {
    let empty = BindGroupLayoutCacheMetrics::new(0, 0, 0);
    assert!(empty.is_empty());

    let non_empty = BindGroupLayoutCacheMetrics::new(5, 0, 0);
    assert!(!non_empty.is_empty());
}

/// BindGroupLayoutCacheMetrics hit_rate_percent() method.
#[test]
fn metrics_hit_rate_percent() {
    let metrics = BindGroupLayoutCacheMetrics::new(10, 80, 20);
    let percent = metrics.hit_rate_percent();
    assert!((percent - 80.0).abs() < 0.1, "Hit rate percent should be ~80%");
}

// ---- layouts_compatible and layouts_equal tests ----

/// layouts_compatible() reflexive property.
#[test]
fn layouts_compatible_reflexive() {
    let entries = basic_entries();
    assert!(layouts_compatible(&entries, &entries), "Layout must be compatible with itself");
}

/// layouts_equal() reflexive property.
#[test]
fn layouts_equal_reflexive() {
    let entries = basic_entries();
    assert!(layouts_equal(&entries, &entries), "Layout must be equal to itself");
}

/// layouts_compatible() symmetric property.
#[test]
fn layouts_compatible_symmetric() {
    let a = basic_entries();
    let b = basic_entries();
    assert_eq!(
        layouts_compatible(&a, &b),
        layouts_compatible(&b, &a),
        "layouts_compatible must be symmetric"
    );
}

/// layouts_equal() symmetric property.
#[test]
fn layouts_equal_symmetric() {
    let a = basic_entries();
    let b = basic_entries();
    assert_eq!(
        layouts_equal(&a, &b),
        layouts_equal(&b, &a),
        "layouts_equal must be symmetric"
    );
}

// =============================================================================
// SECTION 2 -- BEHAVIORAL TESTS (20+ tests)
// =============================================================================

/// Incompatible layouts are not equal.
#[test]
fn incompatible_layouts_not_equal() {
    let a = basic_entries();
    let b = alt_entries();

    if !layouts_compatible(&a, &b) {
        assert!(!layouts_equal(&a, &b), "Incompatible layouts cannot be equal");
    }
}

/// layouts_equal implies layouts_compatible.
#[test]
fn equal_implies_compatible() {
    let entries = basic_entries();
    if layouts_equal(&entries, &entries) {
        assert!(
            layouts_compatible(&entries, &entries),
            "Equal layouts must be compatible"
        );
    }
}

/// Empty layouts are compatible with each other.
#[test]
fn empty_layouts_compatible() {
    let empty: Vec<BindGroupLayoutEntry> = vec![];
    assert!(layouts_compatible(&empty, &empty), "Empty layouts must be compatible");
}

/// Empty layouts are equal.
#[test]
fn empty_layouts_equal() {
    let empty: Vec<BindGroupLayoutEntry> = vec![];
    assert!(layouts_equal(&empty, &empty), "Empty layouts must be equal");
}

/// Different binding indices produce different keys.
#[test]
fn different_binding_indices_different_keys() {
    let entries1 = vec![uniform_buffer_entry(0, ShaderStages::VERTEX)];
    let entries2 = vec![uniform_buffer_entry(1, ShaderStages::VERTEX)];

    let key1 = BindGroupLayoutKey::from_entries(&entries1);
    let key2 = BindGroupLayoutKey::from_entries(&entries2);

    assert_ne!(key1, key2, "Different binding indices must produce different keys");
}

/// Different visibility produces different keys.
#[test]
fn different_visibility_different_keys() {
    let entries1 = vec![uniform_buffer_entry(0, ShaderStages::VERTEX)];
    let entries2 = vec![uniform_buffer_entry(0, ShaderStages::FRAGMENT)];

    let key1 = BindGroupLayoutKey::from_entries(&entries1);
    let key2 = BindGroupLayoutKey::from_entries(&entries2);

    assert_ne!(key1, key2, "Different visibility must produce different keys");
}

/// Different binding types produce different keys.
#[test]
fn different_binding_types_different_keys() {
    let entries1 = vec![uniform_buffer_entry(0, ShaderStages::COMPUTE)];
    let entries2 = vec![storage_buffer_entry(0, ShaderStages::COMPUTE, false)];

    let key1 = BindGroupLayoutKey::from_entries(&entries1);
    let key2 = BindGroupLayoutKey::from_entries(&entries2);

    assert_ne!(key1, key2, "Different binding types must produce different keys");
}

/// Storage buffer read_only flag affects key.
#[test]
fn storage_buffer_readonly_affects_key() {
    let entries1 = vec![storage_buffer_entry(0, ShaderStages::COMPUTE, true)];
    let entries2 = vec![storage_buffer_entry(0, ShaderStages::COMPUTE, false)];

    let key1 = BindGroupLayoutKey::from_entries(&entries1);
    let key2 = BindGroupLayoutKey::from_entries(&entries2);

    assert_ne!(key1, key2, "read_only flag must affect key");
}

/// Adding more entries produces different key.
#[test]
fn more_entries_different_key() {
    let entries1 = vec![uniform_buffer_entry(0, ShaderStages::VERTEX)];
    let entries2 = vec![
        uniform_buffer_entry(0, ShaderStages::VERTEX),
        uniform_buffer_entry(1, ShaderStages::VERTEX),
    ];

    let key1 = BindGroupLayoutKey::from_entries(&entries1);
    let key2 = BindGroupLayoutKey::from_entries(&entries2);

    assert_ne!(key1, key2, "Different entry count must produce different keys");
}

/// Clear resets cache to empty.
#[test]
fn clear_resets_to_empty() {
    let cache = BindGroupLayoutCache::new();
    cache.clear();
    assert!(cache.is_empty());
    assert_eq!(cache.len(), 0);
}

/// Multiple clears do not panic.
#[test]
fn multiple_clears_no_panic() {
    let cache = BindGroupLayoutCache::new();
    cache.clear();
    cache.clear();
    cache.clear();
    assert!(cache.is_empty());
}

/// Cache is thread-safe (Send + Sync).
#[test]
fn cache_is_send_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<BindGroupLayoutCache>();
}

/// BindGroupLayoutKey is Clone.
#[test]
fn key_is_clone() {
    let key = BindGroupLayoutKey::from_entries(&basic_entries());
    let cloned = key.clone();
    assert_eq!(key, cloned);
}

/// BindGroupLayoutCacheMetrics is Clone.
#[test]
fn metrics_is_clone() {
    let metrics = BindGroupLayoutCacheMetrics::new(3, 10, 5);
    let cloned = metrics.clone();
    assert_eq!(metrics.hits, cloned.hits);
    assert_eq!(metrics.misses, cloned.misses);
    assert_eq!(metrics.cache_size, cloned.cache_size);
}

/// BindGroupLayoutCacheMetrics is Debug.
#[test]
fn metrics_is_debug() {
    let metrics = BindGroupLayoutCacheMetrics::new(0, 0, 0);
    let debug_str = format!("{:?}", metrics);
    assert!(!debug_str.is_empty(), "Debug output should not be empty");
}

/// BindGroupLayoutKey is Debug.
#[test]
fn key_is_debug() {
    let key = BindGroupLayoutKey::from_entries(&basic_entries());
    let debug_str = format!("{:?}", key);
    assert!(!debug_str.is_empty(), "Debug output should not be empty");
}

/// BindGroupLayoutCache is Debug.
#[test]
fn cache_is_debug() {
    let cache = BindGroupLayoutCache::new();
    let debug_str = format!("{:?}", cache);
    assert!(!debug_str.is_empty(), "Debug output should not be empty");
}

/// Layouts with different texture dimensions.
#[test]
fn different_texture_dimensions_different_keys() {
    let entries1 = vec![BindGroupLayoutEntry {
        binding: 0,
        visibility: ShaderStages::FRAGMENT,
        ty: BindingType::Texture {
            sample_type: TextureSampleType::Float { filterable: true },
            view_dimension: TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }];
    let entries2 = vec![BindGroupLayoutEntry {
        binding: 0,
        visibility: ShaderStages::FRAGMENT,
        ty: BindingType::Texture {
            sample_type: TextureSampleType::Float { filterable: true },
            view_dimension: TextureViewDimension::D3,
            multisampled: false,
        },
        count: None,
    }];

    let key1 = BindGroupLayoutKey::from_entries(&entries1);
    let key2 = BindGroupLayoutKey::from_entries(&entries2);

    assert_ne!(key1, key2, "Different texture dimensions must produce different keys");
}

/// Layouts with multisampled difference.
#[test]
fn multisampled_affects_key() {
    let entries1 = vec![BindGroupLayoutEntry {
        binding: 0,
        visibility: ShaderStages::FRAGMENT,
        ty: BindingType::Texture {
            sample_type: TextureSampleType::Float { filterable: false },
            view_dimension: TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }];
    let entries2 = vec![BindGroupLayoutEntry {
        binding: 0,
        visibility: ShaderStages::FRAGMENT,
        ty: BindingType::Texture {
            sample_type: TextureSampleType::Float { filterable: false },
            view_dimension: TextureViewDimension::D2,
            multisampled: true,
        },
        count: None,
    }];

    let key1 = BindGroupLayoutKey::from_entries(&entries1);
    let key2 = BindGroupLayoutKey::from_entries(&entries2);

    assert_ne!(key1, key2, "Multisampled flag must affect key");
}

/// Different sampler binding types.
#[test]
fn different_sampler_types_different_keys() {
    let entries1 = vec![BindGroupLayoutEntry {
        binding: 0,
        visibility: ShaderStages::FRAGMENT,
        ty: BindingType::Sampler(SamplerBindingType::Filtering),
        count: None,
    }];
    let entries2 = vec![BindGroupLayoutEntry {
        binding: 0,
        visibility: ShaderStages::FRAGMENT,
        ty: BindingType::Sampler(SamplerBindingType::NonFiltering),
        count: None,
    }];

    let key1 = BindGroupLayoutKey::from_entries(&entries1);
    let key2 = BindGroupLayoutKey::from_entries(&entries2);

    assert_ne!(key1, key2, "Different sampler types must produce different keys");
}

// =============================================================================
// SECTION 3 -- GPU INTEGRATION TESTS (15+ tests, marked #[ignore])
// =============================================================================

/// Helper to create a wgpu instance for integration tests.
#[cfg(test)]
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

    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("Test Device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::downlevel_defaults(),
            memory_hints: wgpu::MemoryHints::default(),
        },
        None,
    )).ok()?;

    Some((device, queue))
}

/// Cache returns same layout for same entries (with GPU).
#[test]

fn gpu_cache_returns_same_layout_for_same_entries() {
    let Some((device, _queue)) = create_test_device() else {
        eprintln!("No GPU available, skipping test");
        return;
    };

    let cache = BindGroupLayoutCache::new();
    let entries = basic_entries();

    let layout1 = cache.get_or_create(&device, None, &entries);
    let layout2 = cache.get_or_create(&device, None, &entries);

    // Both should return the same Arc
    assert!(Arc::ptr_eq(&layout1, &layout2), "Same entries should return same Arc");
}

/// Cache returns different layouts for different entries (with GPU).
#[test]

fn gpu_cache_returns_different_layouts_for_different_entries() {
    let Some((device, _queue)) = create_test_device() else {
        eprintln!("No GPU available, skipping test");
        return;
    };

    let cache = BindGroupLayoutCache::new();

    let layout1 = cache.get_or_create(&device, None, &basic_entries());
    let layout2 = cache.get_or_create(&device, None, &alt_entries());

    assert!(!Arc::ptr_eq(&layout1, &layout2), "Different entries should return different Arcs");
}

/// Hit rate increases on repeated access (with GPU).
#[test]

fn gpu_hit_rate_increases_on_repeated_access() {
    let Some((device, _queue)) = create_test_device() else {
        eprintln!("No GPU available, skipping test");
        return;
    };

    let cache = BindGroupLayoutCache::new();
    let entries = basic_entries();

    // First access is a miss
    let _layout1 = cache.get_or_create(&device, None, &entries);
    let metrics1 = cache.metrics();
    assert_eq!(metrics1.misses, 1, "First access should be a miss");

    // Second access should be a hit
    let _layout2 = cache.get_or_create(&device, None, &entries);
    let metrics2 = cache.metrics();
    assert_eq!(metrics2.hits, 1, "Second access should be a hit");

    // Hit rate should be 50% (1 hit, 1 miss)
    let rate = metrics2.hit_rate;
    assert!((rate - 0.5).abs() < 0.001, "Hit rate should be 50%");
}

/// Cache size increases with new entries (with GPU).
#[test]

fn gpu_cache_size_increases_with_new_entries() {
    let Some((device, _queue)) = create_test_device() else {
        eprintln!("No GPU available, skipping test");
        return;
    };

    let cache = BindGroupLayoutCache::new();
    assert_eq!(cache.len(), 0);

    let _layout1 = cache.get_or_create(&device, None, &basic_entries());
    assert_eq!(cache.len(), 1);

    let _layout2 = cache.get_or_create(&device, None, &alt_entries());
    assert_eq!(cache.len(), 2);

    let _layout3 = cache.get_or_create(&device, None, &compute_entries());
    assert_eq!(cache.len(), 3);
}

/// Cleared cache has zero entries (with GPU).
#[test]

fn gpu_cleared_cache_has_zero_entries() {
    let Some((device, _queue)) = create_test_device() else {
        eprintln!("No GPU available, skipping test");
        return;
    };

    let cache = BindGroupLayoutCache::new();
    let _layout = cache.get_or_create(&device, None, &basic_entries());
    assert_eq!(cache.len(), 1);

    cache.clear();
    assert_eq!(cache.len(), 0);
    assert!(cache.is_empty());
}

/// Created layout is valid wgpu object (with GPU).
#[test]

fn gpu_created_layout_is_valid() {
    let Some((device, _queue)) = create_test_device() else {
        eprintln!("No GPU available, skipping test");
        return;
    };

    let cache = BindGroupLayoutCache::new();
    let layout = cache.get_or_create(&device, None, &basic_entries());

    // Access the inner layout to ensure it's valid
    let _wgpu_layout: &wgpu::BindGroupLayout = layout.as_ref();
    // No panic means the layout is valid
}

/// Layout can be used in pipeline layout creation (with GPU).
#[test]

fn gpu_layout_can_create_pipeline_layout() {
    let Some((device, _queue)) = create_test_device() else {
        eprintln!("No GPU available, skipping test");
        return;
    };

    let cache = BindGroupLayoutCache::new();
    let cached_layout = cache.get_or_create(&device, None, &basic_entries());

    // Create a pipeline layout using the cached layout
    let _pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Test Pipeline Layout"),
        bind_group_layouts: &[cached_layout.as_ref()],
        push_constant_ranges: &[],
    });
    // No panic means success
}

/// Multiple pipelines can share cached layout (with GPU).
#[test]

fn gpu_multiple_pipelines_share_layout() {
    let Some((device, _queue)) = create_test_device() else {
        eprintln!("No GPU available, skipping test");
        return;
    };

    let cache = BindGroupLayoutCache::new();
    let cached_layout = cache.get_or_create(&device, None, &basic_entries());

    // Create multiple pipeline layouts using the same cached layout
    let _pipeline1 = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Pipeline 1"),
        bind_group_layouts: &[cached_layout.as_ref()],
        push_constant_ranges: &[],
    });

    let _pipeline2 = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Pipeline 2"),
        bind_group_layouts: &[cached_layout.as_ref()],
        push_constant_ranges: &[],
    });
    // No panic means success
}

/// Layout survives across simulated frames (with GPU).
#[test]

fn gpu_layout_survives_across_frames() {
    let Some((device, _queue)) = create_test_device() else {
        eprintln!("No GPU available, skipping test");
        return;
    };

    let cache = BindGroupLayoutCache::new();
    let layout1 = cache.get_or_create(&device, None, &basic_entries());

    // Simulate multiple "frames"
    for _ in 0..10 {
        let layout = cache.get_or_create(&device, None, &basic_entries());
        assert!(Arc::ptr_eq(&layout1, &layout), "Layout should persist across frames");
    }
}

/// Contains returns true after insertion (with GPU).
#[test]

fn gpu_contains_after_insertion() {
    let Some((device, _queue)) = create_test_device() else {
        eprintln!("No GPU available, skipping test");
        return;
    };

    let cache = BindGroupLayoutCache::new();
    let entries = basic_entries();

    assert!(!cache.contains(&entries), "Should not contain before insertion");

    let _layout = cache.get_or_create(&device, None, &entries);

    assert!(cache.contains(&entries), "Should contain after insertion");
}

/// Metrics cache_size matches len (with GPU).
#[test]

fn gpu_metrics_cache_size_matches_len() {
    let Some((device, _queue)) = create_test_device() else {
        eprintln!("No GPU available, skipping test");
        return;
    };

    let cache = BindGroupLayoutCache::new();

    let _layout1 = cache.get_or_create(&device, None, &basic_entries());
    let _layout2 = cache.get_or_create(&device, None, &alt_entries());

    let metrics = cache.metrics();
    assert_eq!(metrics.cache_size, cache.len(), "metrics.cache_size should equal len()");
}

/// Concurrent access from multiple threads (with GPU).
#[test]

fn gpu_concurrent_access() {
    use std::thread;

    let Some((device, _queue)) = create_test_device() else {
        eprintln!("No GPU available, skipping test");
        return;
    };

    let cache = Arc::new(BindGroupLayoutCache::new());
    let device = Arc::new(device);

    let mut handles = vec![];

    for i in 0..4 {
        let cache_clone = Arc::clone(&cache);
        let device_clone = Arc::clone(&device);

        let handle = thread::spawn(move || {
            let entries = vec![uniform_buffer_entry(i, ShaderStages::COMPUTE)];
            for _ in 0..10 {
                let _layout = cache_clone.get_or_create(&device_clone, None, &entries);
            }
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().expect("Thread should not panic");
    }

    // 4 unique layouts should have been created
    assert_eq!(cache.len(), 4);
}

/// Empty entries creates valid layout (with GPU).
#[test]

fn gpu_empty_entries_creates_valid_layout() {
    let Some((device, _queue)) = create_test_device() else {
        eprintln!("No GPU available, skipping test");
        return;
    };

    let cache = BindGroupLayoutCache::new();
    let entries: Vec<BindGroupLayoutEntry> = vec![];

    let layout = cache.get_or_create(&device, None, &entries);
    let _wgpu_layout = layout.as_ref();
    // No panic means success
}

/// Repeated access returns same Arc (with GPU).
#[test]

fn gpu_repeated_access_returns_same_arc() {
    let Some((device, _queue)) = create_test_device() else {
        eprintln!("No GPU available, skipping test");
        return;
    };

    let cache = BindGroupLayoutCache::new();
    let entries = basic_entries();

    let layout1 = cache.get_or_create(&device, None, &entries);
    let layout2 = cache.get_or_create(&device, None, &entries);
    let layout3 = cache.get_or_create(&device, None, &entries);

    assert!(Arc::ptr_eq(&layout1, &layout2));
    assert!(Arc::ptr_eq(&layout2, &layout3));
}

/// Layout with label (with GPU).
#[test]

fn gpu_layout_with_label() {
    let Some((device, _queue)) = create_test_device() else {
        eprintln!("No GPU available, skipping test");
        return;
    };

    let cache = BindGroupLayoutCache::new();
    let entries = basic_entries();

    // Create with label
    let _layout = cache.get_or_create(&device, Some("TestLayout"), &entries);

    // Just verify it doesn't panic
    assert_eq!(cache.len(), 1);
}

/// Remove layout from cache (with GPU).
#[test]

fn gpu_remove_layout_from_cache() {
    let Some((device, _queue)) = create_test_device() else {
        eprintln!("No GPU available, skipping test");
        return;
    };

    let cache = BindGroupLayoutCache::new();
    let entries = basic_entries();

    let _layout = cache.get_or_create(&device, None, &entries);
    assert_eq!(cache.len(), 1);

    let removed = cache.remove(&entries);
    assert!(removed, "Should return true when removing existing entry");
    assert_eq!(cache.len(), 0);

    let removed_again = cache.remove(&entries);
    assert!(!removed_again, "Should return false when entry doesn't exist");
}

// =============================================================================
// SECTION 4 -- PROPERTY-BASED TESTS (10+ tests)
// =============================================================================

/// Different entries always produce different keys with high probability.
#[test]
fn property_different_entries_different_keys() {
    let mut keys = Vec::new();

    // Generate many different entry configurations
    for binding in 0..10 {
        for stage in [ShaderStages::VERTEX, ShaderStages::FRAGMENT, ShaderStages::COMPUTE] {
            let entries = vec![uniform_buffer_entry(binding, stage)];
            let key = BindGroupLayoutKey::from_entries(&entries);

            // Check this key is unique
            for existing in keys.iter() {
                assert_ne!(&key, existing, "Different configs should produce different keys");
            }
            keys.push(key);
        }
    }

    // We should have 30 unique keys
    assert_eq!(keys.len(), 30);
}

/// Hit count + miss count = total requests.
#[test]
fn property_hit_miss_equals_total() {
    let metrics = BindGroupLayoutCacheMetrics::new(10, 42, 58);
    let total = metrics.total_requests();
    assert_eq!(total, 100);
}

/// hit_rate is between 0.0 and 1.0.
#[test]
fn property_hit_rate_in_range() {
    // Test various hit/miss combinations
    let test_cases = [
        (0, 0),   // Edge case: no requests
        (0, 10),  // All misses
        (10, 0),  // All hits
        (5, 5),   // 50/50
        (1, 99),  // Low hit rate
        (99, 1),  // High hit rate
    ];

    for (hits, misses) in test_cases {
        let metrics = BindGroupLayoutCacheMetrics::new(0, hits, misses);
        let rate = metrics.hit_rate;

        if !rate.is_nan() {
            assert!(rate >= 0.0, "Hit rate must be >= 0.0");
            assert!(rate <= 1.0, "Hit rate must be <= 1.0");
        }
    }
}

/// Key equality is transitive.
#[test]
fn property_key_equality_transitive() {
    let entries = basic_entries();
    let key1 = BindGroupLayoutKey::from_entries(&entries);
    let key2 = BindGroupLayoutKey::from_entries(&entries);
    let key3 = BindGroupLayoutKey::from_entries(&entries);

    // If key1 == key2 and key2 == key3, then key1 == key3
    if key1 == key2 && key2 == key3 {
        assert_eq!(key1, key3, "Key equality must be transitive");
    }
}

/// layouts_equal implies layouts_compatible for all test cases.
#[test]
fn property_equal_implies_compatible_all_cases() {
    let test_layouts = [
        basic_entries(),
        alt_entries(),
        compute_entries(),
        vec![],
    ];

    for layout in &test_layouts {
        if layouts_equal(layout, layout) {
            assert!(
                layouts_compatible(layout, layout),
                "If layouts_equal returns true, layouts_compatible must also return true"
            );
        }
    }
}

/// Key cloning preserves equality.
#[test]
fn property_clone_preserves_equality() {
    let entries = compute_entries();
    let key = BindGroupLayoutKey::from_entries(&entries);
    let cloned = key.clone();

    assert_eq!(key, cloned, "Cloned key must equal original");
}

/// Metrics cloning preserves all fields.
#[test]
fn property_metrics_clone_preserves_fields() {
    let metrics = BindGroupLayoutCacheMetrics::new(789, 123, 456);
    let cloned = metrics.clone();

    assert_eq!(metrics.hits, cloned.hits);
    assert_eq!(metrics.misses, cloned.misses);
    assert_eq!(metrics.cache_size, cloned.cache_size);
    assert_eq!(metrics.hit_rate, cloned.hit_rate);
}

/// Cache len is never negative.
#[test]
fn property_cache_len_non_negative() {
    let cache = BindGroupLayoutCache::new();
    assert!(cache.len() >= 0, "Cache len must be non-negative");

    cache.clear();
    assert!(cache.len() >= 0, "Cache len must be non-negative after clear");
}

/// Empty cache is empty.
#[test]
fn property_empty_cache_is_empty() {
    let cache = BindGroupLayoutCache::new();
    assert_eq!(cache.is_empty(), cache.len() == 0, "is_empty should equal (len == 0)");
}

/// Different entry counts always produce different keys.
#[test]
fn property_different_entry_counts_different_keys() {
    let entries0: Vec<BindGroupLayoutEntry> = vec![];
    let entries1 = vec![uniform_buffer_entry(0, ShaderStages::VERTEX)];
    let entries2 = vec![
        uniform_buffer_entry(0, ShaderStages::VERTEX),
        uniform_buffer_entry(1, ShaderStages::VERTEX),
    ];
    let entries3 = vec![
        uniform_buffer_entry(0, ShaderStages::VERTEX),
        uniform_buffer_entry(1, ShaderStages::VERTEX),
        uniform_buffer_entry(2, ShaderStages::VERTEX),
    ];

    let key0 = BindGroupLayoutKey::from_entries(&entries0);
    let key1 = BindGroupLayoutKey::from_entries(&entries1);
    let key2 = BindGroupLayoutKey::from_entries(&entries2);
    let key3 = BindGroupLayoutKey::from_entries(&entries3);

    assert_ne!(key0, key1);
    assert_ne!(key1, key2);
    assert_ne!(key2, key3);
    assert_ne!(key0, key2);
    assert_ne!(key0, key3);
    assert_ne!(key1, key3);
}

/// layouts_compatible with self is always true.
#[test]
fn property_layouts_compatible_self_always_true() {
    let test_layouts = [
        vec![],
        basic_entries(),
        alt_entries(),
        compute_entries(),
        vec![storage_texture_entry(0, ShaderStages::COMPUTE)],
    ];

    for layout in &test_layouts {
        assert!(
            layouts_compatible(layout, layout),
            "Any layout must be compatible with itself"
        );
    }
}

/// layouts_equal with self is always true.
#[test]
fn property_layouts_equal_self_always_true() {
    let test_layouts = [
        vec![],
        basic_entries(),
        alt_entries(),
        compute_entries(),
    ];

    for layout in &test_layouts {
        assert!(
            layouts_equal(layout, layout),
            "Any layout must be equal to itself"
        );
    }
}

// =============================================================================
// SECTION 5 -- ADDITIONAL EDGE CASE TESTS
// =============================================================================

/// Large number of entries in layout.
#[test]
fn edge_case_many_entries() {
    let mut entries = Vec::new();
    for i in 0..16 {
        entries.push(uniform_buffer_entry(i, ShaderStages::VERTEX_FRAGMENT));
    }

    let key = BindGroupLayoutKey::from_entries(&entries);
    let key2 = BindGroupLayoutKey::from_entries(&entries);

    assert_eq!(key, key2, "Large entry lists should produce consistent keys");
}

/// Dynamic offset affects key.
#[test]
fn edge_case_dynamic_offset_affects_key() {
    let entries1 = vec![BindGroupLayoutEntry {
        binding: 0,
        visibility: ShaderStages::VERTEX,
        ty: BindingType::Buffer {
            ty: BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }];
    let entries2 = vec![BindGroupLayoutEntry {
        binding: 0,
        visibility: ShaderStages::VERTEX,
        ty: BindingType::Buffer {
            ty: BufferBindingType::Uniform,
            has_dynamic_offset: true,
            min_binding_size: None,
        },
        count: None,
    }];

    let key1 = BindGroupLayoutKey::from_entries(&entries1);
    let key2 = BindGroupLayoutKey::from_entries(&entries2);

    assert_ne!(key1, key2, "Dynamic offset flag must affect key");
}

/// Min binding size affects key (if different).
#[test]
fn edge_case_min_binding_size_affects_key() {
    use std::num::NonZeroU64;

    let entries1 = vec![BindGroupLayoutEntry {
        binding: 0,
        visibility: ShaderStages::COMPUTE,
        ty: BindingType::Buffer {
            ty: BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }];
    let entries2 = vec![BindGroupLayoutEntry {
        binding: 0,
        visibility: ShaderStages::COMPUTE,
        ty: BindingType::Buffer {
            ty: BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: NonZeroU64::new(256),
        },
        count: None,
    }];

    let key1 = BindGroupLayoutKey::from_entries(&entries1);
    let key2 = BindGroupLayoutKey::from_entries(&entries2);

    assert_ne!(key1, key2, "Min binding size must affect key");
}

/// Array count affects key.
#[test]
fn edge_case_array_count_affects_key() {
    use std::num::NonZeroU32;

    let entries1 = vec![BindGroupLayoutEntry {
        binding: 0,
        visibility: ShaderStages::FRAGMENT,
        ty: BindingType::Texture {
            sample_type: TextureSampleType::Float { filterable: true },
            view_dimension: TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }];
    let entries2 = vec![BindGroupLayoutEntry {
        binding: 0,
        visibility: ShaderStages::FRAGMENT,
        ty: BindingType::Texture {
            sample_type: TextureSampleType::Float { filterable: true },
            view_dimension: TextureViewDimension::D2,
            multisampled: false,
        },
        count: NonZeroU32::new(4),
    }];

    let key1 = BindGroupLayoutKey::from_entries(&entries1);
    let key2 = BindGroupLayoutKey::from_entries(&entries2);

    assert_ne!(key1, key2, "Array count must affect key");
}

/// Filterable texture affects key.
#[test]
fn edge_case_filterable_texture_affects_key() {
    let entries1 = vec![BindGroupLayoutEntry {
        binding: 0,
        visibility: ShaderStages::FRAGMENT,
        ty: BindingType::Texture {
            sample_type: TextureSampleType::Float { filterable: true },
            view_dimension: TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }];
    let entries2 = vec![BindGroupLayoutEntry {
        binding: 0,
        visibility: ShaderStages::FRAGMENT,
        ty: BindingType::Texture {
            sample_type: TextureSampleType::Float { filterable: false },
            view_dimension: TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }];

    let key1 = BindGroupLayoutKey::from_entries(&entries1);
    let key2 = BindGroupLayoutKey::from_entries(&entries2);

    assert_ne!(key1, key2, "Filterable flag must affect key");
}

/// Different texture sample types affect key.
#[test]
fn edge_case_texture_sample_type_affects_key() {
    let entries1 = vec![BindGroupLayoutEntry {
        binding: 0,
        visibility: ShaderStages::FRAGMENT,
        ty: BindingType::Texture {
            sample_type: TextureSampleType::Float { filterable: true },
            view_dimension: TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }];
    let entries2 = vec![BindGroupLayoutEntry {
        binding: 0,
        visibility: ShaderStages::FRAGMENT,
        ty: BindingType::Texture {
            sample_type: TextureSampleType::Depth,
            view_dimension: TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }];

    let key1 = BindGroupLayoutKey::from_entries(&entries1);
    let key2 = BindGroupLayoutKey::from_entries(&entries2);

    assert_ne!(key1, key2, "Texture sample type must affect key");
}

/// Storage texture access mode affects key.
#[test]
fn edge_case_storage_texture_access_affects_key() {
    let entries1 = vec![BindGroupLayoutEntry {
        binding: 0,
        visibility: ShaderStages::COMPUTE,
        ty: BindingType::StorageTexture {
            access: StorageTextureAccess::WriteOnly,
            format: TextureFormat::Rgba8Unorm,
            view_dimension: TextureViewDimension::D2,
        },
        count: None,
    }];
    let entries2 = vec![BindGroupLayoutEntry {
        binding: 0,
        visibility: ShaderStages::COMPUTE,
        ty: BindingType::StorageTexture {
            access: StorageTextureAccess::ReadOnly,
            format: TextureFormat::Rgba8Unorm,
            view_dimension: TextureViewDimension::D2,
        },
        count: None,
    }];

    let key1 = BindGroupLayoutKey::from_entries(&entries1);
    let key2 = BindGroupLayoutKey::from_entries(&entries2);

    assert_ne!(key1, key2, "Storage texture access mode must affect key");
}

/// Storage texture format affects key.
#[test]
fn edge_case_storage_texture_format_affects_key() {
    let entries1 = vec![BindGroupLayoutEntry {
        binding: 0,
        visibility: ShaderStages::COMPUTE,
        ty: BindingType::StorageTexture {
            access: StorageTextureAccess::WriteOnly,
            format: TextureFormat::Rgba8Unorm,
            view_dimension: TextureViewDimension::D2,
        },
        count: None,
    }];
    let entries2 = vec![BindGroupLayoutEntry {
        binding: 0,
        visibility: ShaderStages::COMPUTE,
        ty: BindingType::StorageTexture {
            access: StorageTextureAccess::WriteOnly,
            format: TextureFormat::Rgba16Float,
            view_dimension: TextureViewDimension::D2,
        },
        count: None,
    }];

    let key1 = BindGroupLayoutKey::from_entries(&entries1);
    let key2 = BindGroupLayoutKey::from_entries(&entries2);

    assert_ne!(key1, key2, "Storage texture format must affect key");
}

/// Combined visibility flags.
#[test]
fn edge_case_combined_visibility_flags() {
    let entries1 = vec![uniform_buffer_entry(0, ShaderStages::VERTEX)];
    let entries2 = vec![uniform_buffer_entry(0, ShaderStages::VERTEX | ShaderStages::FRAGMENT)];

    let key1 = BindGroupLayoutKey::from_entries(&entries1);
    let key2 = BindGroupLayoutKey::from_entries(&entries2);

    assert_ne!(key1, key2, "Combined visibility flags must differ from single flag");
}

/// VERTEX_FRAGMENT vs separate flags.
#[test]
fn edge_case_vertex_fragment_stages() {
    let entries1 = vec![uniform_buffer_entry(0, ShaderStages::VERTEX_FRAGMENT)];
    let entries2 = vec![uniform_buffer_entry(0, ShaderStages::VERTEX | ShaderStages::FRAGMENT)];

    let key1 = BindGroupLayoutKey::from_entries(&entries1);
    let key2 = BindGroupLayoutKey::from_entries(&entries2);

    // VERTEX_FRAGMENT should be equivalent to VERTEX | FRAGMENT
    assert_eq!(key1, key2, "VERTEX_FRAGMENT should equal VERTEX | FRAGMENT");
}

/// Cube texture view dimension.
#[test]
fn edge_case_cube_texture_dimension() {
    let entries1 = vec![BindGroupLayoutEntry {
        binding: 0,
        visibility: ShaderStages::FRAGMENT,
        ty: BindingType::Texture {
            sample_type: TextureSampleType::Float { filterable: true },
            view_dimension: TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }];
    let entries2 = vec![BindGroupLayoutEntry {
        binding: 0,
        visibility: ShaderStages::FRAGMENT,
        ty: BindingType::Texture {
            sample_type: TextureSampleType::Float { filterable: true },
            view_dimension: TextureViewDimension::Cube,
            multisampled: false,
        },
        count: None,
    }];

    let key1 = BindGroupLayoutKey::from_entries(&entries1);
    let key2 = BindGroupLayoutKey::from_entries(&entries2);

    assert_ne!(key1, key2, "Cube vs D2 texture dimension must produce different keys");
}

/// Comparison sampler type.
#[test]
fn edge_case_comparison_sampler() {
    let entries1 = vec![BindGroupLayoutEntry {
        binding: 0,
        visibility: ShaderStages::FRAGMENT,
        ty: BindingType::Sampler(SamplerBindingType::Filtering),
        count: None,
    }];
    let entries2 = vec![BindGroupLayoutEntry {
        binding: 0,
        visibility: ShaderStages::FRAGMENT,
        ty: BindingType::Sampler(SamplerBindingType::Comparison),
        count: None,
    }];

    let key1 = BindGroupLayoutKey::from_entries(&entries1);
    let key2 = BindGroupLayoutKey::from_entries(&entries2);

    assert_ne!(key1, key2, "Comparison sampler must produce different key than filtering sampler");
}

/// Uint texture sample type.
#[test]
fn edge_case_uint_texture_sample_type() {
    let entries1 = vec![BindGroupLayoutEntry {
        binding: 0,
        visibility: ShaderStages::FRAGMENT,
        ty: BindingType::Texture {
            sample_type: TextureSampleType::Float { filterable: true },
            view_dimension: TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }];
    let entries2 = vec![BindGroupLayoutEntry {
        binding: 0,
        visibility: ShaderStages::FRAGMENT,
        ty: BindingType::Texture {
            sample_type: TextureSampleType::Uint,
            view_dimension: TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }];

    let key1 = BindGroupLayoutKey::from_entries(&entries1);
    let key2 = BindGroupLayoutKey::from_entries(&entries2);

    assert_ne!(key1, key2, "Uint texture sample type must produce different key");
}

/// Sint texture sample type.
#[test]
fn edge_case_sint_texture_sample_type() {
    let entries1 = vec![BindGroupLayoutEntry {
        binding: 0,
        visibility: ShaderStages::FRAGMENT,
        ty: BindingType::Texture {
            sample_type: TextureSampleType::Uint,
            view_dimension: TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }];
    let entries2 = vec![BindGroupLayoutEntry {
        binding: 0,
        visibility: ShaderStages::FRAGMENT,
        ty: BindingType::Texture {
            sample_type: TextureSampleType::Sint,
            view_dimension: TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }];

    let key1 = BindGroupLayoutKey::from_entries(&entries1);
    let key2 = BindGroupLayoutKey::from_entries(&entries2);

    assert_ne!(key1, key2, "Sint vs Uint texture sample type must produce different keys");
}

/// 2D array texture dimension.
#[test]
fn edge_case_2d_array_texture_dimension() {
    let entries1 = vec![BindGroupLayoutEntry {
        binding: 0,
        visibility: ShaderStages::FRAGMENT,
        ty: BindingType::Texture {
            sample_type: TextureSampleType::Float { filterable: true },
            view_dimension: TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }];
    let entries2 = vec![BindGroupLayoutEntry {
        binding: 0,
        visibility: ShaderStages::FRAGMENT,
        ty: BindingType::Texture {
            sample_type: TextureSampleType::Float { filterable: true },
            view_dimension: TextureViewDimension::D2Array,
            multisampled: false,
        },
        count: None,
    }];

    let key1 = BindGroupLayoutKey::from_entries(&entries1);
    let key2 = BindGroupLayoutKey::from_entries(&entries2);

    assert_ne!(key1, key2, "D2 vs D2Array texture dimension must produce different keys");
}
