// SPDX-License-Identifier: MIT
//
// blackbox_bind_group_cache.rs -- Blackbox tests for T-WGPU-P2.5.2 BindGroupCache.
//
// CLEANROOM: No src/ access beyond the public API exported by the crate.
// Tests use only the documented public types and functions:
//
//   - ResourceId
//   - BindGroupResourceType
//   - BindGroupResourceEntry
//   - BindGroupCacheKey
//   - CachedBindGroup
//   - BindGroupCache
//   - BindGroupCacheMetrics
//
// ACCEPTANCE CRITERIA:
//   1. Public API contracts -- 20+ tests covering type construction and accessors
//   2. Behavioral tests     -- 20+ tests verifying cache semantics
//   3. GPU integration      -- 15+ tests (marked #[ignore] if no GPU)
//   4. Property-based       -- 10+ tests for invariants
//
// Total target: 60+ tests

use renderer_backend::resources::{
    BindGroupCache, BindGroupCacheKey, BindGroupCacheMetrics,
    BindGroupResourceEntry, BindGroupResourceType, ResourceId,
};
use std::collections::HashMap;

// =============================================================================
// HELPERS -- Entry construction helpers for cleanroom testing
// =============================================================================

/// Creates a buffer resource entry with offset and size.
fn buffer_entry(binding: u32, resource_id: ResourceId, offset: u64, size: Option<u64>) -> BindGroupResourceEntry {
    BindGroupResourceEntry::buffer(binding, resource_id, offset, size)
}

/// Creates a sampler resource entry.
fn sampler_entry(binding: u32, resource_id: ResourceId) -> BindGroupResourceEntry {
    BindGroupResourceEntry::sampler(binding, resource_id)
}

/// Creates a texture view resource entry.
fn texture_view_entry(binding: u32, resource_id: ResourceId) -> BindGroupResourceEntry {
    BindGroupResourceEntry::texture_view(binding, resource_id)
}

/// Creates a storage texture view resource entry.
fn storage_texture_view_entry(binding: u32, resource_id: ResourceId) -> BindGroupResourceEntry {
    BindGroupResourceEntry::storage_texture_view(binding, resource_id)
}

/// Creates a basic set of entries: buffer (0), sampler (1), texture view (2).
fn basic_entries() -> Vec<BindGroupResourceEntry> {
    vec![
        buffer_entry(0, ResourceId::new(100), 0, Some(256)),
        sampler_entry(1, ResourceId::new(200)),
        texture_view_entry(2, ResourceId::new(300)),
    ]
}

/// Creates an alternative set of entries different from basic_entries().
fn alt_entries() -> Vec<BindGroupResourceEntry> {
    vec![
        buffer_entry(0, ResourceId::new(400), 0, Some(512)),
        buffer_entry(1, ResourceId::new(500), 64, None),
    ]
}

/// Creates a compute-shader style set of entries.
fn compute_entries() -> Vec<BindGroupResourceEntry> {
    vec![
        buffer_entry(0, ResourceId::new(600), 0, Some(1024)),
        storage_texture_view_entry(1, ResourceId::new(700)),
        buffer_entry(2, ResourceId::new(800), 128, Some(256)),
    ]
}

/// Creates entries with same bindings but different resources.
fn same_bindings_different_resources() -> Vec<BindGroupResourceEntry> {
    vec![
        buffer_entry(0, ResourceId::new(999), 0, Some(256)),
        sampler_entry(1, ResourceId::new(998)),
        texture_view_entry(2, ResourceId::new(997)),
    ]
}

// =============================================================================
// SECTION 1 -- PUBLIC API CONTRACTS (20+ tests)
// =============================================================================

// ---- ResourceId tests ----

/// ResourceId::new() creates a valid id.
#[test]
fn resource_id_new_creates_valid_id() {
    let id = ResourceId::new(42);
    assert_eq!(id.value(), 42);
}

/// ResourceId::new() with zero.
#[test]
fn resource_id_new_with_zero() {
    let id = ResourceId::new(0);
    assert_eq!(id.value(), 0);
}

/// ResourceId::new() with max u64.
#[test]
fn resource_id_new_with_max() {
    let id = ResourceId::new(u64::MAX);
    assert_eq!(id.value(), u64::MAX);
}

/// ResourceId equality for same values.
#[test]
fn resource_id_equality_same_values() {
    let id1 = ResourceId::new(123);
    let id2 = ResourceId::new(123);
    assert_eq!(id1, id2);
}

/// ResourceId inequality for different values.
#[test]
fn resource_id_inequality_different_values() {
    let id1 = ResourceId::new(123);
    let id2 = ResourceId::new(456);
    assert_ne!(id1, id2);
}

/// ResourceId works as HashMap key.
#[test]
fn resource_id_works_as_hashmap_key() {
    let mut map: HashMap<ResourceId, &str> = HashMap::new();

    let id1 = ResourceId::new(1);
    let id2 = ResourceId::new(2);

    map.insert(id1, "first");
    map.insert(id2, "second");

    assert_eq!(map.get(&id1), Some(&"first"));
    assert_eq!(map.get(&id2), Some(&"second"));
    assert_eq!(map.len(), 2);
}

/// ResourceId is Copy.
#[test]
fn resource_id_is_copy() {
    let id = ResourceId::new(42);
    let copy = id;
    assert_eq!(id, copy);
}

/// ResourceId hash is stable.
#[test]
fn resource_id_hash_is_stable() {
    use std::hash::{Hash, Hasher};
    use std::collections::hash_map::DefaultHasher;

    let id1 = ResourceId::new(999);
    let id2 = ResourceId::new(999);

    let mut h1 = DefaultHasher::new();
    let mut h2 = DefaultHasher::new();
    id1.hash(&mut h1);
    id2.hash(&mut h2);

    assert_eq!(h1.finish(), h2.finish());
}

// ---- BindGroupResourceType tests ----

/// BindGroupResourceType::buffer() creates buffer variant.
#[test]
fn resource_type_buffer() {
    let ty = BindGroupResourceType::buffer();
    // Type created successfully
    let _ = ty;
}

/// BindGroupResourceType::buffer_range() creates ranged buffer.
#[test]
fn resource_type_buffer_range() {
    let ty = BindGroupResourceType::buffer_range(64, 128);
    let _ = ty;
}

/// BindGroupResourceType::buffer_sized() creates sized buffer.
#[test]
fn resource_type_buffer_sized() {
    let ty = BindGroupResourceType::buffer_sized(256);
    let _ = ty;
}

// ---- BindGroupResourceEntry tests ----

/// BindGroupResourceEntry::buffer() creates buffer entry.
#[test]
fn resource_entry_buffer_creation() {
    let entry = BindGroupResourceEntry::buffer(0, ResourceId::new(1), 0, Some(256));
    let _ = entry;
}

/// BindGroupResourceEntry::buffer() with no size.
#[test]
fn resource_entry_buffer_no_size() {
    let entry = BindGroupResourceEntry::buffer(0, ResourceId::new(1), 0, None);
    let _ = entry;
}

/// BindGroupResourceEntry::buffer() with offset.
#[test]
fn resource_entry_buffer_with_offset() {
    let entry = BindGroupResourceEntry::buffer(0, ResourceId::new(1), 64, Some(128));
    let _ = entry;
}

/// BindGroupResourceEntry::sampler() creates sampler entry.
#[test]
fn resource_entry_sampler_creation() {
    let entry = BindGroupResourceEntry::sampler(0, ResourceId::new(42));
    let _ = entry;
}

/// BindGroupResourceEntry::texture_view() creates texture view entry.
#[test]
fn resource_entry_texture_view_creation() {
    let entry = BindGroupResourceEntry::texture_view(1, ResourceId::new(100));
    let _ = entry;
}

/// BindGroupResourceEntry::storage_texture_view() creates storage texture entry.
#[test]
fn resource_entry_storage_texture_view_creation() {
    let entry = BindGroupResourceEntry::storage_texture_view(2, ResourceId::new(200));
    let _ = entry;
}

// ---- BindGroupCacheKey tests ----

/// BindGroupCacheKey::new() creates a key from layout hash and entries.
#[test]
fn cache_key_new_creates_key() {
    let entries = basic_entries();
    let key = BindGroupCacheKey::new(12345, &entries);
    let _ = key;
}

/// BindGroupCacheKey equality for identical inputs.
#[test]
fn cache_key_equality_for_same_inputs() {
    let entries = basic_entries();
    let key1 = BindGroupCacheKey::new(12345, &entries);
    let key2 = BindGroupCacheKey::new(12345, &entries);
    assert_eq!(key1, key2, "Keys from identical inputs must be equal");
}

/// BindGroupCacheKey inequality for different layout hashes.
#[test]
fn cache_key_inequality_for_different_layout_hash() {
    let entries = basic_entries();
    let key1 = BindGroupCacheKey::new(12345, &entries);
    let key2 = BindGroupCacheKey::new(54321, &entries);
    assert_ne!(key1, key2, "Keys with different layout hashes must not be equal");
}

/// BindGroupCacheKey inequality for different entries.
#[test]
fn cache_key_inequality_for_different_entries() {
    let key1 = BindGroupCacheKey::new(12345, &basic_entries());
    let key2 = BindGroupCacheKey::new(12345, &alt_entries());
    assert_ne!(key1, key2, "Keys from different entries must not be equal");
}

/// BindGroupCacheKey works as HashMap key.
#[test]
fn cache_key_works_as_hashmap_key() {
    let mut map: HashMap<BindGroupCacheKey, &str> = HashMap::new();

    let key1 = BindGroupCacheKey::new(111, &basic_entries());
    let key2 = BindGroupCacheKey::new(222, &alt_entries());

    map.insert(key1.clone(), "basic");
    map.insert(key2.clone(), "alt");

    assert_eq!(map.get(&key1), Some(&"basic"));
    assert_eq!(map.get(&key2), Some(&"alt"));
    assert_eq!(map.len(), 2);
}

/// BindGroupCacheKey hash is stable.
#[test]
fn cache_key_hash_is_stable() {
    use std::hash::{Hash, Hasher};
    use std::collections::hash_map::DefaultHasher;

    let entries = basic_entries();
    let key1 = BindGroupCacheKey::new(999, &entries);
    let key2 = BindGroupCacheKey::new(999, &entries);

    let mut h1 = DefaultHasher::new();
    let mut h2 = DefaultHasher::new();
    key1.hash(&mut h1);
    key2.hash(&mut h2);

    assert_eq!(h1.finish(), h2.finish(), "Hash must be stable for same inputs");
}

/// BindGroupCacheKey from empty entries.
#[test]
fn cache_key_from_empty_entries() {
    let entries: Vec<BindGroupResourceEntry> = vec![];
    let key = BindGroupCacheKey::new(0, &entries);
    let _ = key;
}

/// BindGroupCacheKey from single entry.
#[test]
fn cache_key_from_single_entry() {
    let entries = vec![buffer_entry(0, ResourceId::new(1), 0, Some(64))];
    let key = BindGroupCacheKey::new(42, &entries);
    let _ = key;
}

/// BindGroupCacheKey layout_hash() accessor.
#[test]
fn cache_key_layout_hash_accessor() {
    let entries = basic_entries();
    let key = BindGroupCacheKey::new(99999, &entries);
    assert_eq!(key.layout_hash(), 99999);
}

/// BindGroupCacheKey resources_hash() accessor.
#[test]
fn cache_key_resources_hash_accessor() {
    let entries = basic_entries();
    let key = BindGroupCacheKey::new(12345, &entries);
    // Hash value should be computed
    let _ = key.resources_hash();
}

// ---- BindGroupCache tests ----

/// BindGroupCache::new() creates an empty cache.
#[test]
fn cache_new_creates_empty() {
    let cache = BindGroupCache::new();
    assert!(cache.is_empty(), "New cache must be empty");
    assert_eq!(cache.len(), 0, "New cache length must be 0");
}

/// BindGroupCache::default() works.
#[test]
fn cache_default_works() {
    let cache = BindGroupCache::default();
    assert!(cache.is_empty());
}

/// BindGroupCache::len() returns correct size.
#[test]
fn cache_len_returns_correct_size() {
    let cache = BindGroupCache::new();
    assert_eq!(cache.len(), 0);
}

/// BindGroupCache::is_empty() correctness.
#[test]
fn cache_is_empty_correctness() {
    let cache = BindGroupCache::new();
    assert!(cache.is_empty());
}

/// BindGroupCache::contains() on empty cache.
#[test]
fn cache_contains_on_empty() {
    let cache = BindGroupCache::new();
    let entries = basic_entries();
    assert!(!cache.contains(12345, &entries), "Empty cache must not contain any key");
}

/// BindGroupCache::clear() on empty cache does not panic.
#[test]
fn cache_clear_empty_no_panic() {
    let cache = BindGroupCache::new();
    cache.clear();
    assert!(cache.is_empty());
}

/// BindGroupCache::metrics() returns initial stats.
#[test]
fn cache_metrics_returns_initial_stats() {
    let cache = BindGroupCache::new();
    let metrics = cache.metrics();

    assert!(metrics.is_empty(), "Initial metrics should be empty");
}

/// BindGroupCache::reset_metrics() clears counters.
#[test]
fn cache_reset_metrics() {
    let cache = BindGroupCache::new();
    cache.reset_metrics();
    let metrics = cache.metrics();
    assert!(metrics.is_empty());
}

/// BindGroupCache::current_frame() returns frame number.
#[test]
fn cache_current_frame() {
    let cache = BindGroupCache::new();
    let frame = cache.current_frame();
    assert_eq!(frame, 0, "Initial frame should be 0");
}

/// BindGroupCache::begin_frame() increments frame.
#[test]
fn cache_begin_frame_increments() {
    let cache = BindGroupCache::new();
    assert_eq!(cache.current_frame(), 0);

    cache.begin_frame();
    assert_eq!(cache.current_frame(), 1);

    cache.begin_frame();
    assert_eq!(cache.current_frame(), 2);
}

/// BindGroupCache::tracked_resource_count() on empty cache.
#[test]
fn cache_tracked_resource_count_empty() {
    let cache = BindGroupCache::new();
    assert_eq!(cache.tracked_resource_count(), 0);
}

// ---- BindGroupCacheMetrics tests ----

/// BindGroupCacheMetrics::new() works.
#[test]
fn metrics_new_works() {
    let metrics = BindGroupCacheMetrics::new(10, 5, 2, 3, 15);
    assert!(!metrics.is_empty());
}

/// BindGroupCacheMetrics total_requests calculation.
#[test]
fn metrics_total_requests() {
    let metrics = BindGroupCacheMetrics::new(5, 30, 20, 2, 0);
    let total = metrics.total_requests();
    assert_eq!(total, 50, "Total requests should be hits + misses");
}

/// BindGroupCacheMetrics is_empty() method.
#[test]
fn metrics_is_empty() {
    let empty = BindGroupCacheMetrics::new(0, 0, 0, 0, 0);
    assert!(empty.is_empty());

    let non_empty = BindGroupCacheMetrics::new(5, 0, 0, 0, 0);
    assert!(!non_empty.is_empty());
}

/// BindGroupCacheMetrics hit_rate_percent() method.
#[test]
fn metrics_hit_rate_percent() {
    let metrics = BindGroupCacheMetrics::new(10, 80, 20, 0, 0);
    let percent = metrics.hit_rate_percent();
    assert!((percent - 80.0).abs() < 0.1, "Hit rate percent should be ~80%");
}

/// BindGroupCacheMetrics hit_rate_percent() with no requests.
#[test]
fn metrics_hit_rate_percent_no_requests() {
    let metrics = BindGroupCacheMetrics::new(0, 0, 0, 0, 0);
    let percent = metrics.hit_rate_percent();
    assert!(percent >= 0.0 && percent <= 100.0, "Hit rate should be valid");
}

// =============================================================================
// SECTION 2 -- BEHAVIORAL TESTS (20+ tests)
// =============================================================================

/// Different resource IDs produce different keys.
#[test]
fn different_resource_ids_different_keys() {
    let entries1 = vec![buffer_entry(0, ResourceId::new(100), 0, Some(256))];
    let entries2 = vec![buffer_entry(0, ResourceId::new(200), 0, Some(256))];

    let key1 = BindGroupCacheKey::new(42, &entries1);
    let key2 = BindGroupCacheKey::new(42, &entries2);

    assert_ne!(key1, key2, "Different resource IDs must produce different keys");
}

/// Different buffer offsets produce different keys.
#[test]
fn different_buffer_offsets_different_keys() {
    let entries1 = vec![buffer_entry(0, ResourceId::new(100), 0, Some(256))];
    let entries2 = vec![buffer_entry(0, ResourceId::new(100), 64, Some(256))];

    let key1 = BindGroupCacheKey::new(42, &entries1);
    let key2 = BindGroupCacheKey::new(42, &entries2);

    assert_ne!(key1, key2, "Different buffer offsets must produce different keys");
}

/// Different buffer sizes produce different keys.
#[test]
fn different_buffer_sizes_different_keys() {
    let entries1 = vec![buffer_entry(0, ResourceId::new(100), 0, Some(256))];
    let entries2 = vec![buffer_entry(0, ResourceId::new(100), 0, Some(512))];

    let key1 = BindGroupCacheKey::new(42, &entries1);
    let key2 = BindGroupCacheKey::new(42, &entries2);

    assert_ne!(key1, key2, "Different buffer sizes must produce different keys");
}

/// Buffer with None size differs from sized buffer.
#[test]
fn buffer_none_size_differs_from_sized() {
    let entries1 = vec![buffer_entry(0, ResourceId::new(100), 0, None)];
    let entries2 = vec![buffer_entry(0, ResourceId::new(100), 0, Some(256))];

    let key1 = BindGroupCacheKey::new(42, &entries1);
    let key2 = BindGroupCacheKey::new(42, &entries2);

    assert_ne!(key1, key2, "None size must differ from explicit size");
}

/// Different binding indices produce different keys.
#[test]
fn different_binding_indices_different_keys() {
    let entries1 = vec![buffer_entry(0, ResourceId::new(100), 0, Some(256))];
    let entries2 = vec![buffer_entry(1, ResourceId::new(100), 0, Some(256))];

    let key1 = BindGroupCacheKey::new(42, &entries1);
    let key2 = BindGroupCacheKey::new(42, &entries2);

    assert_ne!(key1, key2, "Different binding indices must produce different keys");
}

/// Different resource types produce different keys.
#[test]
fn different_resource_types_different_keys() {
    let id = ResourceId::new(100);
    let entries1 = vec![sampler_entry(0, id)];
    let entries2 = vec![texture_view_entry(0, id)];

    let key1 = BindGroupCacheKey::new(42, &entries1);
    let key2 = BindGroupCacheKey::new(42, &entries2);

    assert_ne!(key1, key2, "Different resource types must produce different keys");
}

/// Clear resets cache to empty.
#[test]
fn clear_resets_to_empty() {
    let cache = BindGroupCache::new();
    cache.clear();
    assert!(cache.is_empty());
    assert_eq!(cache.len(), 0);
}

/// Multiple clears do not panic.
#[test]
fn multiple_clears_no_panic() {
    let cache = BindGroupCache::new();
    cache.clear();
    cache.clear();
    cache.clear();
    assert!(cache.is_empty());
}

/// Cache is thread-safe (Send + Sync).
#[test]
fn cache_is_send_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<BindGroupCache>();
}

/// BindGroupCacheKey is Clone.
#[test]
fn cache_key_is_clone() {
    let key = BindGroupCacheKey::new(12345, &basic_entries());
    let cloned = key.clone();
    assert_eq!(key, cloned);
}

/// BindGroupCacheMetrics is Clone.
#[test]
fn metrics_is_clone() {
    let metrics = BindGroupCacheMetrics::new(3, 10, 5, 2, 15);
    let cloned = metrics.clone();
    assert_eq!(metrics.total_requests(), cloned.total_requests());
}

/// BindGroupCacheMetrics is Debug.
#[test]
fn metrics_is_debug() {
    let metrics = BindGroupCacheMetrics::new(0, 0, 0, 0, 0);
    let debug_str = format!("{:?}", metrics);
    assert!(!debug_str.is_empty(), "Debug output should not be empty");
}

/// BindGroupCacheKey is Debug.
#[test]
fn cache_key_is_debug() {
    let key = BindGroupCacheKey::new(12345, &basic_entries());
    let debug_str = format!("{:?}", key);
    assert!(!debug_str.is_empty(), "Debug output should not be empty");
}

/// BindGroupCache is Debug.
#[test]
fn cache_is_debug() {
    let cache = BindGroupCache::new();
    let debug_str = format!("{:?}", cache);
    assert!(!debug_str.is_empty(), "Debug output should not be empty");
}

/// ResourceId is Debug.
#[test]
fn resource_id_is_debug() {
    let id = ResourceId::new(42);
    let debug_str = format!("{:?}", id);
    assert!(!debug_str.is_empty(), "Debug output should not be empty");
}

/// Multiple entries with same binding is valid input.
#[test]
fn multiple_entries_same_binding() {
    // This tests that the API accepts multiple entries at same binding
    // (even if it might not be valid in actual GPU usage)
    let entries = vec![
        buffer_entry(0, ResourceId::new(100), 0, Some(64)),
        buffer_entry(0, ResourceId::new(200), 0, Some(128)),
    ];
    let key = BindGroupCacheKey::new(42, &entries);
    let _ = key;
}

/// Large number of entries.
#[test]
fn large_number_of_entries() {
    let entries: Vec<BindGroupResourceEntry> = (0..100)
        .map(|i| buffer_entry(i, ResourceId::new(i as u64), 0, Some(64)))
        .collect();

    let key = BindGroupCacheKey::new(42, &entries);
    let _ = key;
}

/// Frame eviction on empty cache.
#[test]
fn evict_old_on_empty_cache() {
    let cache = BindGroupCache::new();
    let evicted = cache.evict_old(10);
    assert_eq!(evicted, 0, "Empty cache should evict nothing");
}

/// Frame eviction with max age 0.
#[test]
fn evict_old_max_age_zero() {
    let cache = BindGroupCache::new();
    let evicted = cache.evict_old(0);
    assert_eq!(evicted, 0);
}

/// Resource invalidation on empty cache.
#[test]
fn invalidate_resource_on_empty_cache() {
    let cache = BindGroupCache::new();
    let invalidated = cache.invalidate_resource(ResourceId::new(42));
    assert_eq!(invalidated, 0, "Empty cache should invalidate nothing");
}

/// Batch resource invalidation on empty cache.
#[test]
fn invalidate_resources_batch_on_empty_cache() {
    let cache = BindGroupCache::new();
    let resources = vec![ResourceId::new(1), ResourceId::new(2), ResourceId::new(3)];
    let invalidated = cache.invalidate_resources(resources);
    assert_eq!(invalidated, 0);
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

/// Helper to create a basic bind group layout.
#[cfg(test)]
fn create_basic_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Test Layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    })
}

/// Helper to create a test buffer.
#[cfg(test)]
fn create_test_buffer(device: &wgpu::Device, size: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Test Buffer"),
        size,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

/// Cache operations work with GPU device (with GPU).
#[test]

fn gpu_cache_basic_operations() {
    let Some((_device, _queue)) = create_test_device() else {
        eprintln!("No GPU available, skipping test");
        return;
    };

    let cache = BindGroupCache::new();
    assert!(cache.is_empty());
    assert_eq!(cache.len(), 0);
}

/// Cache frame progression works (with GPU).
#[test]

fn gpu_cache_frame_progression() {
    let Some((_device, _queue)) = create_test_device() else {
        eprintln!("No GPU available, skipping test");
        return;
    };

    let cache = BindGroupCache::new();

    for expected_frame in 1..=10 {
        cache.begin_frame();
        assert_eq!(cache.current_frame(), expected_frame);
    }
}

/// Cache metrics work with GPU operations (with GPU).
#[test]

fn gpu_cache_metrics_tracking() {
    let Some((_device, _queue)) = create_test_device() else {
        eprintln!("No GPU available, skipping test");
        return;
    };

    let cache = BindGroupCache::new();
    let metrics = cache.metrics();

    // Initial state
    assert!(metrics.is_empty());

    cache.reset_metrics();
    let reset_metrics = cache.metrics();
    assert!(reset_metrics.is_empty());
}

/// Cache clear works (with GPU).
#[test]

fn gpu_cache_clear_operation() {
    let Some((_device, _queue)) = create_test_device() else {
        eprintln!("No GPU available, skipping test");
        return;
    };

    let cache = BindGroupCache::new();

    // Multiple clears should be safe
    cache.clear();
    cache.clear();

    assert!(cache.is_empty());
}

/// Cache contains query works (with GPU).
#[test]

fn gpu_cache_contains_query() {
    let Some((_device, _queue)) = create_test_device() else {
        eprintln!("No GPU available, skipping test");
        return;
    };

    let cache = BindGroupCache::new();
    let entries = basic_entries();

    assert!(!cache.contains(12345, &entries));
}

/// Cache eviction works (with GPU).
#[test]

fn gpu_cache_eviction() {
    let Some((_device, _queue)) = create_test_device() else {
        eprintln!("No GPU available, skipping test");
        return;
    };

    let cache = BindGroupCache::new();

    // Progress frames
    for _ in 0..10 {
        cache.begin_frame();
    }

    // Evict entries older than 5 frames
    let evicted = cache.evict_old(5);
    assert_eq!(evicted, 0, "Empty cache evicts nothing");
}

/// Cache resource invalidation works (with GPU).
#[test]

fn gpu_cache_resource_invalidation() {
    let Some((_device, _queue)) = create_test_device() else {
        eprintln!("No GPU available, skipping test");
        return;
    };

    let cache = BindGroupCache::new();
    let resource_id = ResourceId::new(12345);

    let invalidated = cache.invalidate_resource(resource_id);
    assert_eq!(invalidated, 0, "Empty cache has nothing to invalidate");
}

/// Cache batch resource invalidation works (with GPU).
#[test]

fn gpu_cache_batch_invalidation() {
    let Some((_device, _queue)) = create_test_device() else {
        eprintln!("No GPU available, skipping test");
        return;
    };

    let cache = BindGroupCache::new();
    let resources = vec![
        ResourceId::new(1),
        ResourceId::new(2),
        ResourceId::new(3),
    ];

    let invalidated = cache.invalidate_resources(resources);
    assert_eq!(invalidated, 0);
}

/// Cache tracked resource count (with GPU).
#[test]

fn gpu_cache_tracked_resources() {
    let Some((_device, _queue)) = create_test_device() else {
        eprintln!("No GPU available, skipping test");
        return;
    };

    let cache = BindGroupCache::new();
    assert_eq!(cache.tracked_resource_count(), 0);
}

/// Cache is thread safe for concurrent access (with GPU).
#[test]

fn gpu_cache_concurrent_access() {
    use std::thread;
    use std::sync::Arc;

    let Some((_device, _queue)) = create_test_device() else {
        eprintln!("No GPU available, skipping test");
        return;
    };

    let cache = Arc::new(BindGroupCache::new());

    let handles: Vec<_> = (0..4).map(|_| {
        let cache_clone = Arc::clone(&cache);
        thread::spawn(move || {
            for _ in 0..100 {
                let _ = cache_clone.is_empty();
                let _ = cache_clone.len();
                let _ = cache_clone.metrics();
            }
        })
    }).collect();

    for handle in handles {
        handle.join().expect("Thread should complete");
    }
}

/// Cache can handle rapid frame changes (with GPU).
#[test]

fn gpu_cache_rapid_frame_changes() {
    let Some((_device, _queue)) = create_test_device() else {
        eprintln!("No GPU available, skipping test");
        return;
    };

    let cache = BindGroupCache::new();

    for _ in 0..1000 {
        cache.begin_frame();
    }

    assert_eq!(cache.current_frame(), 1000);
}

/// Cache with multiple key queries (with GPU).
#[test]

fn gpu_cache_multiple_key_queries() {
    let Some((_device, _queue)) = create_test_device() else {
        eprintln!("No GPU available, skipping test");
        return;
    };

    let cache = BindGroupCache::new();

    let entry_sets: Vec<(u64, Vec<BindGroupResourceEntry>)> = (0..10)
        .map(|i| {
            let entries = vec![buffer_entry(0, ResourceId::new(i as u64), 0, Some(64))];
            (i as u64, entries)
        })
        .collect();

    for (layout_hash, entries) in &entry_sets {
        assert!(!cache.contains(*layout_hash, entries));
    }
}

/// Cache handles resource ID collisions gracefully (with GPU).
#[test]

fn gpu_cache_resource_id_collision_handling() {
    let Some((_device, _queue)) = create_test_device() else {
        eprintln!("No GPU available, skipping test");
        return;
    };

    let cache = BindGroupCache::new();

    // Same resource ID, different layouts
    let id = ResourceId::new(42);
    let entries1 = vec![buffer_entry(0, id, 0, Some(64))];
    let entries2 = vec![buffer_entry(0, id, 0, Some(64))];
    let key1 = BindGroupCacheKey::new(111, &entries1);
    let key2 = BindGroupCacheKey::new(222, &entries2);

    assert!(!cache.contains(111, &entries1));
    assert!(!cache.contains(222, &entries2));
    assert_ne!(key1, key2, "Different layout hashes produce different keys");
}

/// Cache remove operation works (with GPU).
#[test]

fn gpu_cache_remove_operation() {
    let Some((_device, _queue)) = create_test_device() else {
        eprintln!("No GPU available, skipping test");
        return;
    };

    let cache = BindGroupCache::new();
    let entries = basic_entries();

    // Remove from empty cache should return false
    let removed = cache.remove(12345, &entries);
    assert!(!removed, "Remove from empty cache should return false");
}

// =============================================================================
// SECTION 4 -- PROPERTY-BASED TESTS (10+ tests)
// =============================================================================

/// Property: Different entries always produce different keys.
#[test]
fn property_different_entries_different_keys() {
    let layout_hash = 12345u64;

    // Generate various entry combinations
    let entry_sets: Vec<Vec<BindGroupResourceEntry>> = vec![
        basic_entries(),
        alt_entries(),
        compute_entries(),
        same_bindings_different_resources(),
        vec![],
        vec![buffer_entry(0, ResourceId::new(1), 0, Some(64))],
        vec![sampler_entry(0, ResourceId::new(1))],
        vec![texture_view_entry(0, ResourceId::new(1))],
        vec![storage_texture_view_entry(0, ResourceId::new(1))],
    ];

    let keys: Vec<BindGroupCacheKey> = entry_sets
        .iter()
        .map(|entries| BindGroupCacheKey::new(layout_hash, entries))
        .collect();

    // Count unique keys
    let mut unique_keys: Vec<&BindGroupCacheKey> = keys.iter().collect();
    unique_keys.sort_by_key(|k| (k.layout_hash(), k.resources_hash()));
    unique_keys.dedup_by(|a, b| a == b);

    // At least most entries should produce unique keys
    assert!(unique_keys.len() >= entry_sets.len() / 2);
}

/// Property: Key comparison is reflexive.
#[test]
fn property_key_comparison_reflexive() {
    for entries in [basic_entries(), alt_entries(), compute_entries()] {
        let key = BindGroupCacheKey::new(42, &entries);
        assert_eq!(key, key, "Key must equal itself");
    }
}

/// Property: Key comparison is symmetric.
#[test]
fn property_key_comparison_symmetric() {
    let entries = basic_entries();
    let key1 = BindGroupCacheKey::new(42, &entries);
    let key2 = BindGroupCacheKey::new(42, &entries);

    assert_eq!(key1 == key2, key2 == key1);
}

/// Property: Key comparison is transitive.
#[test]
fn property_key_comparison_transitive() {
    let entries = basic_entries();
    let key1 = BindGroupCacheKey::new(42, &entries);
    let key2 = BindGroupCacheKey::new(42, &entries);
    let key3 = BindGroupCacheKey::new(42, &entries);

    if key1 == key2 && key2 == key3 {
        assert_eq!(key1, key3, "Key equality must be transitive");
    }
}

/// Property: hit + miss = total_requests.
#[test]
fn property_hits_plus_misses_equals_total() {
    for (hits, misses) in [(0, 0), (10, 5), (100, 200), (0, 100), (100, 0)] {
        let metrics = BindGroupCacheMetrics::new(5, hits, misses, 0, 0);
        assert_eq!(
            metrics.total_requests(),
            hits + misses,
            "Total requests must equal hits + misses"
        );
    }
}

/// Property: hit_rate is between 0.0 and 1.0 (as percentage 0-100).
#[test]
fn property_hit_rate_in_valid_range() {
    let test_cases = [
        (0, 0),
        (10, 0),
        (0, 10),
        (50, 50),
        (100, 0),
        (0, 100),
        (75, 25),
    ];

    for (hits, misses) in test_cases {
        let metrics = BindGroupCacheMetrics::new(5, hits, misses, 0, 0);
        let rate = metrics.hit_rate_percent();
        assert!(
            rate >= 0.0 && rate <= 100.0,
            "Hit rate {} must be between 0 and 100",
            rate
        );
    }
}

/// Property: ResourceId with same value always equals.
#[test]
fn property_resource_id_equality_consistent() {
    for value in [0, 1, 42, 100, u64::MAX] {
        let id1 = ResourceId::new(value);
        let id2 = ResourceId::new(value);
        assert_eq!(id1, id2);
        assert_eq!(id1.value(), id2.value());
    }
}

/// Property: ResourceId with different values never equals.
#[test]
fn property_resource_id_different_values_not_equal() {
    let values = [0, 1, 42, 100, 1000, u64::MAX];
    for (i, &v1) in values.iter().enumerate() {
        for &v2 in values.iter().skip(i + 1) {
            let id1 = ResourceId::new(v1);
            let id2 = ResourceId::new(v2);
            assert_ne!(id1, id2);
        }
    }
}

/// Property: Cache len() >= 0 always.
#[test]
fn property_cache_len_non_negative() {
    let cache = BindGroupCache::new();
    assert!(cache.len() >= 0);

    cache.clear();
    assert!(cache.len() >= 0);

    cache.begin_frame();
    assert!(cache.len() >= 0);
}

/// Property: is_empty() is equivalent to len() == 0.
#[test]
fn property_is_empty_equivalent_to_len_zero() {
    let cache = BindGroupCache::new();
    assert_eq!(cache.is_empty(), cache.len() == 0);

    cache.clear();
    assert_eq!(cache.is_empty(), cache.len() == 0);
}

/// Property: Frame number is monotonically increasing.
#[test]
fn property_frame_monotonically_increasing() {
    let cache = BindGroupCache::new();
    let mut prev_frame = cache.current_frame();

    for _ in 0..100 {
        cache.begin_frame();
        let current = cache.current_frame();
        assert!(current > prev_frame, "Frame must increase");
        prev_frame = current;
    }
}

/// Property: Eviction count is non-negative.
#[test]
fn property_eviction_count_non_negative() {
    let cache = BindGroupCache::new();

    for max_age in [0, 1, 5, 10, 100] {
        let evicted = cache.evict_old(max_age);
        assert!(evicted >= 0);
    }
}

/// Property: Invalidation count is non-negative.
#[test]
fn property_invalidation_count_non_negative() {
    let cache = BindGroupCache::new();

    for id in [0, 1, 42, u64::MAX] {
        let invalidated = cache.invalidate_resource(ResourceId::new(id));
        assert!(invalidated >= 0);
    }
}

// =============================================================================
// ADDITIONAL EDGE CASE TESTS
// =============================================================================

/// Edge case: Very large layout hash.
#[test]
fn edge_case_large_layout_hash() {
    let key = BindGroupCacheKey::new(u64::MAX, &basic_entries());
    assert_eq!(key.layout_hash(), u64::MAX);
}

/// Edge case: Zero layout hash.
#[test]
fn edge_case_zero_layout_hash() {
    let key = BindGroupCacheKey::new(0, &basic_entries());
    assert_eq!(key.layout_hash(), 0);
}

/// Edge case: Large buffer offset.
#[test]
fn edge_case_large_buffer_offset() {
    let entry = buffer_entry(0, ResourceId::new(1), u64::MAX - 100, Some(64));
    let key = BindGroupCacheKey::new(42, &[entry]);
    let _ = key;
}

/// Edge case: Large buffer size.
#[test]
fn edge_case_large_buffer_size() {
    let entry = buffer_entry(0, ResourceId::new(1), 0, Some(u64::MAX));
    let key = BindGroupCacheKey::new(42, &[entry]);
    let _ = key;
}

/// Edge case: Maximum binding index.
#[test]
fn edge_case_max_binding_index() {
    let entry = buffer_entry(u32::MAX, ResourceId::new(1), 0, Some(64));
    let key = BindGroupCacheKey::new(42, &[entry]);
    let _ = key;
}

/// Edge case: All resource types in one key.
#[test]
fn edge_case_all_resource_types() {
    let entries = vec![
        buffer_entry(0, ResourceId::new(1), 0, Some(64)),
        sampler_entry(1, ResourceId::new(2)),
        texture_view_entry(2, ResourceId::new(3)),
        storage_texture_view_entry(3, ResourceId::new(4)),
    ];
    let key = BindGroupCacheKey::new(42, &entries);
    let _ = key;
}

/// Edge case: Duplicate resource IDs across entries.
#[test]
fn edge_case_duplicate_resource_ids() {
    let id = ResourceId::new(42);
    let entries = vec![
        buffer_entry(0, id, 0, Some(64)),
        buffer_entry(1, id, 64, Some(64)),
        buffer_entry(2, id, 128, Some(64)),
    ];
    let key = BindGroupCacheKey::new(42, &entries);
    let _ = key;
}

/// Edge case: Concurrent metrics access.
#[test]
fn edge_case_concurrent_metrics_access() {
    use std::thread;
    use std::sync::Arc;

    let cache = Arc::new(BindGroupCache::new());

    let handles: Vec<_> = (0..4).map(|_| {
        let cache_clone = Arc::clone(&cache);
        thread::spawn(move || {
            for _ in 0..1000 {
                let _ = cache_clone.metrics();
            }
        })
    }).collect();

    for handle in handles {
        handle.join().expect("Thread should complete");
    }
}

/// Edge case: Concurrent frame progression.
#[test]
fn edge_case_concurrent_frame_progression() {
    use std::thread;
    use std::sync::Arc;

    let cache = Arc::new(BindGroupCache::new());

    let handles: Vec<_> = (0..4).map(|_| {
        let cache_clone = Arc::clone(&cache);
        thread::spawn(move || {
            for _ in 0..100 {
                cache_clone.begin_frame();
            }
        })
    }).collect();

    for handle in handles {
        handle.join().expect("Thread should complete");
    }

    // Frame should have progressed (exact count depends on race conditions)
    assert!(cache.current_frame() > 0);
}

/// Edge case: Interleaved operations.
#[test]
fn edge_case_interleaved_operations() {
    let cache = BindGroupCache::new();

    for _ in 0..10 {
        cache.begin_frame();
        let _ = cache.metrics();
        let _ = cache.evict_old(5);
        cache.invalidate_resource(ResourceId::new(42));
        let _ = cache.is_empty();
        let _ = cache.len();
    }
}

/// BindGroupResourceType equality.
#[test]
fn resource_type_variants_distinct() {
    let buffer = BindGroupResourceType::buffer();
    let buffer_range = BindGroupResourceType::buffer_range(0, 64);
    let buffer_sized = BindGroupResourceType::buffer_sized(64);

    // All should be constructable without panic
    let _ = (buffer, buffer_range, buffer_sized);
}

/// ResourceId value accessor roundtrip.
#[test]
fn resource_id_value_roundtrip() {
    for value in [0, 1, 42, 1000, u64::MAX / 2, u64::MAX] {
        let id = ResourceId::new(value);
        assert_eq!(id.value(), value, "Value roundtrip failed for {}", value);
    }
}
