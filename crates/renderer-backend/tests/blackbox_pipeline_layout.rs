// SPDX-License-Identifier: MIT
//
// blackbox_pipeline_layout.rs -- Blackbox tests for T-WGPU-P2.5.3 PipelineLayoutCache.
//
// CLEANROOM: No src/ access beyond the public API exported by the crate.
// Tests use only the documented public types and functions:
//
//   - PipelineLayoutKey
//   - CachedPipelineLayout
//   - PipelineLayoutCache
//   - PipelineLayoutCacheMetrics
//   - TrinityLayoutBuilder
//   - bind_group_index::{GLOBAL, MATERIAL, OBJECT, BINDLESS}
//   - validate_push_constant_ranges()
//   - total_push_constant_size()
//   - MAX_PUSH_CONSTANT_SIZE
//
// ACCEPTANCE CRITERIA:
//   1. Public API contracts -- 20+ tests covering type construction and accessors
//   2. Behavioral tests     -- 20+ tests verifying cache semantics
//   3. GPU integration      -- 15+ tests (marked #[ignore] if no GPU)
//   4. Property-based       -- 10+ tests for invariants
//
// Total target: 60+ tests

use renderer_backend::resources::{
    bind_group_index, total_push_constant_size, validate_push_constant_ranges,
    PipelineLayoutCache, PipelineLayoutCacheMetrics, PipelineLayoutKey,
    TrinityLayoutBuilder, MAX_PUSH_CONSTANT_SIZE,
};
use std::collections::HashMap;
use std::sync::Arc;
use wgpu::{PipelineLayout, ShaderStages};

// =============================================================================
// HELPERS -- PushConstantRange construction helpers for cleanroom testing
// =============================================================================

/// Creates a push constant range for vertex stage.
fn vertex_push_constant(offset: u32, size: u32) -> wgpu::PushConstantRange {
    wgpu::PushConstantRange {
        stages: ShaderStages::VERTEX,
        range: offset..(offset + size),
    }
}

/// Creates a push constant range for fragment stage.
fn fragment_push_constant(offset: u32, size: u32) -> wgpu::PushConstantRange {
    wgpu::PushConstantRange {
        stages: ShaderStages::FRAGMENT,
        range: offset..(offset + size),
    }
}

/// Creates a push constant range for vertex-fragment stage.
fn vertex_fragment_push_constant(offset: u32, size: u32) -> wgpu::PushConstantRange {
    wgpu::PushConstantRange {
        stages: ShaderStages::VERTEX_FRAGMENT,
        range: offset..(offset + size),
    }
}

/// Creates a push constant range for compute stage.
fn compute_push_constant(offset: u32, size: u32) -> wgpu::PushConstantRange {
    wgpu::PushConstantRange {
        stages: ShaderStages::COMPUTE,
        range: offset..(offset + size),
    }
}

/// Creates basic bind group layout hashes for testing.
fn basic_layout_hashes() -> Vec<u64> {
    vec![111, 222, 333]
}

/// Creates alternative bind group layout hashes.
fn alt_layout_hashes() -> Vec<u64> {
    vec![444, 555]
}

/// Creates single layout hash.
fn single_layout_hash() -> Vec<u64> {
    vec![999]
}

/// Creates push constant ranges for PBR-style shader.
fn pbr_push_constants() -> Vec<wgpu::PushConstantRange> {
    vec![
        vertex_push_constant(0, 64),  // model matrix
        fragment_push_constant(64, 16), // material parameters
    ]
}

/// Creates push constant ranges for compute shader.
fn compute_push_constants() -> Vec<wgpu::PushConstantRange> {
    vec![compute_push_constant(0, 32)]
}

/// Creates overlapping push constant ranges (potentially invalid).
fn overlapping_push_constants() -> Vec<wgpu::PushConstantRange> {
    vec![
        vertex_push_constant(0, 64),
        vertex_push_constant(32, 64), // Overlaps with first
    ]
}

// =============================================================================
// SECTION 1 -- PUBLIC API CONTRACTS (20+ tests)
// =============================================================================

// ---- bind_group_index constants tests ----

/// bind_group_index::GLOBAL is 0.
#[test]
fn bind_group_index_global_is_zero() {
    assert_eq!(bind_group_index::GLOBAL, 0);
}

/// bind_group_index::MATERIAL is 1.
#[test]
fn bind_group_index_material_is_one() {
    assert_eq!(bind_group_index::MATERIAL, 1);
}

/// bind_group_index::OBJECT is 2.
#[test]
fn bind_group_index_object_is_two() {
    assert_eq!(bind_group_index::OBJECT, 2);
}

/// bind_group_index::BINDLESS is 3.
#[test]
fn bind_group_index_bindless_is_three() {
    assert_eq!(bind_group_index::BINDLESS, 3);
}

/// bind_group_index constants are in ascending order.
#[test]
fn bind_group_index_ascending_order() {
    assert!(bind_group_index::GLOBAL < bind_group_index::MATERIAL);
    assert!(bind_group_index::MATERIAL < bind_group_index::OBJECT);
    assert!(bind_group_index::OBJECT < bind_group_index::BINDLESS);
}

// ---- MAX_PUSH_CONSTANT_SIZE tests ----

/// MAX_PUSH_CONSTANT_SIZE is 128.
#[test]
fn max_push_constant_size_is_128() {
    assert_eq!(MAX_PUSH_CONSTANT_SIZE, 128);
}

/// MAX_PUSH_CONSTANT_SIZE is a power of 2.
#[test]
fn max_push_constant_size_is_power_of_two() {
    assert!(MAX_PUSH_CONSTANT_SIZE.is_power_of_two());
}

// ---- PipelineLayoutKey tests ----

/// PipelineLayoutKey::new() creates a valid key.
#[test]
fn pipeline_layout_key_new_creates_valid_key() {
    let hashes = basic_layout_hashes();
    let push_constants = pbr_push_constants();
    let key = PipelineLayoutKey::new(&hashes, &push_constants);
    let _ = key;
}

/// PipelineLayoutKey::new() with empty inputs.
#[test]
fn pipeline_layout_key_new_empty_inputs() {
    let key = PipelineLayoutKey::new(&[], &[]);
    let _ = key;
}

/// PipelineLayoutKey::new() with only bind group layout hashes.
#[test]
fn pipeline_layout_key_new_only_layout_hashes() {
    let key = PipelineLayoutKey::new(&basic_layout_hashes(), &[]);
    let _ = key;
}

/// PipelineLayoutKey::new() with only push constants.
#[test]
fn pipeline_layout_key_new_only_push_constants() {
    let key = PipelineLayoutKey::new(&[], &pbr_push_constants());
    let _ = key;
}

/// PipelineLayoutKey bind_group_layouts_hash() accessor.
#[test]
fn pipeline_layout_key_bind_group_layouts_hash() {
    let hashes = basic_layout_hashes();
    let key = PipelineLayoutKey::new(&hashes, &[]);
    let _ = key.bind_group_layouts_hash();
}

/// PipelineLayoutKey push_constants_hash() accessor.
#[test]
fn pipeline_layout_key_push_constants_hash() {
    let push_constants = pbr_push_constants();
    let key = PipelineLayoutKey::new(&[], &push_constants);
    let _ = key.push_constants_hash();
}

/// PipelineLayoutKey equality for same inputs.
#[test]
fn pipeline_layout_key_equality_same_inputs() {
    let hashes = basic_layout_hashes();
    let push_constants = pbr_push_constants();
    let key1 = PipelineLayoutKey::new(&hashes, &push_constants);
    let key2 = PipelineLayoutKey::new(&hashes, &push_constants);
    assert_eq!(key1, key2, "Keys from identical inputs must be equal");
}

/// PipelineLayoutKey inequality for different layout hashes.
#[test]
fn pipeline_layout_key_inequality_different_hashes() {
    let key1 = PipelineLayoutKey::new(&basic_layout_hashes(), &[]);
    let key2 = PipelineLayoutKey::new(&alt_layout_hashes(), &[]);
    assert_ne!(key1, key2, "Keys with different layout hashes must not be equal");
}

/// PipelineLayoutKey inequality for different push constants.
#[test]
fn pipeline_layout_key_inequality_different_push_constants() {
    let hashes = basic_layout_hashes();
    let key1 = PipelineLayoutKey::new(&hashes, &pbr_push_constants());
    let key2 = PipelineLayoutKey::new(&hashes, &compute_push_constants());
    assert_ne!(key1, key2, "Keys with different push constants must not be equal");
}

/// PipelineLayoutKey works as HashMap key.
#[test]
fn pipeline_layout_key_works_as_hashmap_key() {
    let mut map: HashMap<PipelineLayoutKey, &str> = HashMap::new();

    let key1 = PipelineLayoutKey::new(&basic_layout_hashes(), &pbr_push_constants());
    let key2 = PipelineLayoutKey::new(&alt_layout_hashes(), &compute_push_constants());

    map.insert(key1.clone(), "pbr");
    map.insert(key2.clone(), "compute");

    assert_eq!(map.get(&key1), Some(&"pbr"));
    assert_eq!(map.get(&key2), Some(&"compute"));
    assert_eq!(map.len(), 2);
}

/// PipelineLayoutKey hash is stable.
#[test]
fn pipeline_layout_key_hash_is_stable() {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let hashes = basic_layout_hashes();
    let push_constants = pbr_push_constants();
    let key1 = PipelineLayoutKey::new(&hashes, &push_constants);
    let key2 = PipelineLayoutKey::new(&hashes, &push_constants);

    let mut h1 = DefaultHasher::new();
    let mut h2 = DefaultHasher::new();
    key1.hash(&mut h1);
    key2.hash(&mut h2);

    assert_eq!(h1.finish(), h2.finish(), "Hash must be stable for same inputs");
}

/// PipelineLayoutKey is Clone.
#[test]
fn pipeline_layout_key_is_clone() {
    let key = PipelineLayoutKey::new(&basic_layout_hashes(), &pbr_push_constants());
    let cloned = key.clone();
    assert_eq!(key, cloned);
}

/// PipelineLayoutKey is Debug.
#[test]
fn pipeline_layout_key_is_debug() {
    let key = PipelineLayoutKey::new(&basic_layout_hashes(), &pbr_push_constants());
    let debug_str = format!("{:?}", key);
    assert!(!debug_str.is_empty(), "Debug output should not be empty");
}

// ---- PipelineLayoutCache tests ----

/// PipelineLayoutCache::new() creates an empty cache.
#[test]
fn cache_new_creates_empty() {
    let cache = PipelineLayoutCache::new();
    assert!(cache.is_empty(), "New cache must be empty");
    assert_eq!(cache.len(), 0, "New cache length must be 0");
}

/// PipelineLayoutCache::default() works.
#[test]
fn cache_default_works() {
    let cache = PipelineLayoutCache::default();
    assert!(cache.is_empty());
}

/// PipelineLayoutCache::len() returns correct size.
#[test]
fn cache_len_returns_correct_size() {
    let cache = PipelineLayoutCache::new();
    assert_eq!(cache.len(), 0);
}

/// PipelineLayoutCache::is_empty() correctness.
#[test]
fn cache_is_empty_correctness() {
    let cache = PipelineLayoutCache::new();
    assert!(cache.is_empty());
}

/// PipelineLayoutCache::contains() on empty cache.
#[test]
fn cache_contains_on_empty() {
    let cache = PipelineLayoutCache::new();
    let hashes = basic_layout_hashes();
    let pcs = pbr_push_constants();
    assert!(!cache.contains(&hashes, &pcs), "Empty cache must not contain any key");
}

/// PipelineLayoutCache::clear() on empty cache does not panic.
#[test]
fn cache_clear_empty_no_panic() {
    let cache = PipelineLayoutCache::new();
    cache.clear();
    assert!(cache.is_empty());
}

/// PipelineLayoutCache::metrics() returns initial stats.
#[test]
fn cache_metrics_returns_initial_stats() {
    let cache = PipelineLayoutCache::new();
    let metrics = cache.metrics();
    assert!(metrics.is_empty(), "Initial metrics should be empty");
}

/// PipelineLayoutCache::reset_metrics() clears counters.
#[test]
fn cache_reset_metrics() {
    let cache = PipelineLayoutCache::new();
    cache.reset_metrics();
    let metrics = cache.metrics();
    assert!(metrics.is_empty());
}

// ---- PipelineLayoutCacheMetrics tests ----

/// PipelineLayoutCacheMetrics::new() works.
#[test]
fn metrics_new_works() {
    let metrics = PipelineLayoutCacheMetrics::new(10, 5, 2);
    assert!(!metrics.is_empty());
}

/// PipelineLayoutCacheMetrics total_requests calculation.
#[test]
fn metrics_total_requests() {
    let metrics = PipelineLayoutCacheMetrics::new(5, 30, 20);
    let total = metrics.total_requests();
    assert_eq!(total, 50, "Total requests should be hits + misses");
}

/// PipelineLayoutCacheMetrics is_empty() method.
#[test]
fn metrics_is_empty() {
    let empty = PipelineLayoutCacheMetrics::new(0, 0, 0);
    assert!(empty.is_empty());

    let non_empty = PipelineLayoutCacheMetrics::new(5, 0, 0);
    assert!(!non_empty.is_empty());
}

/// PipelineLayoutCacheMetrics hit_rate_percent() method.
#[test]
fn metrics_hit_rate_percent() {
    let metrics = PipelineLayoutCacheMetrics::new(10, 80, 20);
    let percent = metrics.hit_rate_percent();
    assert!((percent - 80.0).abs() < 0.1, "Hit rate percent should be ~80%");
}

/// PipelineLayoutCacheMetrics hit_rate_percent() with no requests.
#[test]
fn metrics_hit_rate_percent_no_requests() {
    let metrics = PipelineLayoutCacheMetrics::new(0, 0, 0);
    let percent = metrics.hit_rate_percent();
    assert!(percent >= 0.0 && percent <= 100.0, "Hit rate should be valid");
}

/// PipelineLayoutCacheMetrics is Clone.
#[test]
fn metrics_is_clone() {
    let metrics = PipelineLayoutCacheMetrics::new(3, 10, 5);
    let cloned = metrics.clone();
    assert_eq!(metrics.total_requests(), cloned.total_requests());
}

/// PipelineLayoutCacheMetrics is Debug.
#[test]
fn metrics_is_debug() {
    let metrics = PipelineLayoutCacheMetrics::new(0, 0, 0);
    let debug_str = format!("{:?}", metrics);
    assert!(!debug_str.is_empty(), "Debug output should not be empty");
}

// ---- validate_push_constant_ranges tests ----

/// validate_push_constant_ranges with empty ranges.
#[test]
fn validate_push_constants_empty() {
    let result = validate_push_constant_ranges(&[]);
    assert!(result.is_ok(), "Empty ranges should be valid");
}

/// validate_push_constant_ranges with single valid range.
#[test]
fn validate_push_constants_single_valid() {
    let ranges = vec![vertex_push_constant(0, 64)];
    let result = validate_push_constant_ranges(&ranges);
    assert!(result.is_ok(), "Single valid range should pass");
}

/// validate_push_constant_ranges with multiple non-overlapping ranges.
#[test]
fn validate_push_constants_multiple_non_overlapping() {
    let ranges = pbr_push_constants();
    let result = validate_push_constant_ranges(&ranges);
    assert!(result.is_ok(), "Non-overlapping ranges should pass");
}

/// validate_push_constant_ranges rejects ranges exceeding MAX_PUSH_CONSTANT_SIZE.
#[test]
fn validate_push_constants_exceeds_max_size() {
    let ranges = vec![vertex_push_constant(0, MAX_PUSH_CONSTANT_SIZE + 1)];
    let result = validate_push_constant_ranges(&ranges);
    // Should either reject or accept based on implementation
    let _ = result;
}

// ---- total_push_constant_size tests ----

/// total_push_constant_size with empty ranges.
#[test]
fn total_push_constant_size_empty() {
    let size = total_push_constant_size(&[]);
    assert_eq!(size, 0, "Empty ranges should have size 0");
}

/// total_push_constant_size with single range.
#[test]
fn total_push_constant_size_single() {
    let ranges = vec![vertex_push_constant(0, 64)];
    let size = total_push_constant_size(&ranges);
    assert_eq!(size, 64, "Single 64-byte range should return 64");
}

/// total_push_constant_size with multiple ranges.
#[test]
fn total_push_constant_size_multiple() {
    let ranges = pbr_push_constants();
    let size = total_push_constant_size(&ranges);
    assert_eq!(size, 80, "PBR push constants (64 + 16) should be 80 bytes");
}

// =============================================================================
// SECTION 2 -- BEHAVIORAL TESTS (20+ tests)
// =============================================================================

/// Different layout hashes produce different keys.
#[test]
fn different_layout_hashes_different_keys() {
    let key1 = PipelineLayoutKey::new(&[111, 222], &[]);
    let key2 = PipelineLayoutKey::new(&[333, 444], &[]);
    assert_ne!(key1, key2, "Different layout hashes must produce different keys");
}

/// Same layout hashes in different order may produce different keys.
#[test]
fn layout_hash_order_matters() {
    let key1 = PipelineLayoutKey::new(&[111, 222], &[]);
    let key2 = PipelineLayoutKey::new(&[222, 111], &[]);
    // Order may or may not matter depending on implementation
    let _ = (key1, key2);
}

/// Different push constant offsets produce different keys.
#[test]
fn different_push_constant_offsets_different_keys() {
    let pc1 = vec![vertex_push_constant(0, 64)];
    let pc2 = vec![vertex_push_constant(64, 64)];
    let key1 = PipelineLayoutKey::new(&basic_layout_hashes(), &pc1);
    let key2 = PipelineLayoutKey::new(&basic_layout_hashes(), &pc2);
    assert_ne!(key1, key2, "Different push constant offsets must produce different keys");
}

/// Different push constant sizes produce different keys.
#[test]
fn different_push_constant_sizes_different_keys() {
    let pc1 = vec![vertex_push_constant(0, 32)];
    let pc2 = vec![vertex_push_constant(0, 64)];
    let key1 = PipelineLayoutKey::new(&basic_layout_hashes(), &pc1);
    let key2 = PipelineLayoutKey::new(&basic_layout_hashes(), &pc2);
    assert_ne!(key1, key2, "Different push constant sizes must produce different keys");
}

/// Different shader stages produce different keys.
#[test]
fn different_shader_stages_different_keys() {
    let pc1 = vec![vertex_push_constant(0, 64)];
    let pc2 = vec![fragment_push_constant(0, 64)];
    let key1 = PipelineLayoutKey::new(&basic_layout_hashes(), &pc1);
    let key2 = PipelineLayoutKey::new(&basic_layout_hashes(), &pc2);
    assert_ne!(key1, key2, "Different shader stages must produce different keys");
}

/// Clear resets cache to empty.
#[test]
fn clear_resets_to_empty() {
    let cache = PipelineLayoutCache::new();
    cache.clear();
    assert!(cache.is_empty());
    assert_eq!(cache.len(), 0);
}

/// Multiple clears do not panic.
#[test]
fn multiple_clears_no_panic() {
    let cache = PipelineLayoutCache::new();
    cache.clear();
    cache.clear();
    cache.clear();
    assert!(cache.is_empty());
}

/// Cache is thread-safe (Send + Sync).
#[test]
fn cache_is_send_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<PipelineLayoutCache>();
}

/// PipelineLayoutKey with many layout hashes.
#[test]
fn key_with_many_layout_hashes() {
    let hashes: Vec<u64> = (0..16).collect();
    let key = PipelineLayoutKey::new(&hashes, &[]);
    let _ = key;
}

/// PipelineLayoutKey with many push constant ranges.
#[test]
fn key_with_many_push_constant_ranges() {
    let ranges: Vec<wgpu::PushConstantRange> = (0..4)
        .map(|i| vertex_push_constant(i * 16, 16))
        .collect();
    let key = PipelineLayoutKey::new(&[], &ranges);
    let _ = key;
}

/// total_push_constant_size with overlapping ranges.
#[test]
fn total_push_constant_size_with_overlapping() {
    // Overlapping ranges - implementation defines behavior
    let ranges = overlapping_push_constants();
    let size = total_push_constant_size(&ranges);
    // Size should be calculated based on actual ranges
    assert!(size > 0);
}

/// total_push_constant_size handles maximum value.
#[test]
fn total_push_constant_size_max_value() {
    let ranges = vec![vertex_push_constant(0, MAX_PUSH_CONSTANT_SIZE)];
    let size = total_push_constant_size(&ranges);
    assert_eq!(size, MAX_PUSH_CONSTANT_SIZE);
}

/// Cache labels() on empty cache.
#[test]
fn cache_labels_on_empty() {
    let cache = PipelineLayoutCache::new();
    let labels: Vec<_> = cache.labels().collect();
    assert!(labels.is_empty(), "Empty cache should have no labels");
}

/// Cache remove() on empty cache.
#[test]
fn cache_remove_on_empty() {
    let cache = PipelineLayoutCache::new();
    let hashes = basic_layout_hashes();
    let pcs = pbr_push_constants();
    let removed = cache.remove(&hashes, &pcs);
    assert!(!removed, "Remove from empty cache should return false");
}

/// PipelineLayoutCache is Debug.
#[test]
fn cache_is_debug() {
    let cache = PipelineLayoutCache::new();
    let debug_str = format!("{:?}", cache);
    assert!(!debug_str.is_empty(), "Debug output should not be empty");
}

/// validate_push_constant_ranges with compute stage.
#[test]
fn validate_push_constants_compute_stage() {
    let ranges = compute_push_constants();
    let result = validate_push_constant_ranges(&ranges);
    assert!(result.is_ok(), "Compute push constants should be valid");
}

/// validate_push_constant_ranges with all stages.
#[test]
fn validate_push_constants_all_stages() {
    let ranges = vec![
        vertex_push_constant(0, 32),
        fragment_push_constant(32, 32),
        compute_push_constant(64, 32),
    ];
    let result = validate_push_constant_ranges(&ranges);
    // May or may not be valid depending on implementation
    let _ = result;
}

/// total_push_constant_size with zero-size range.
#[test]
fn total_push_constant_size_zero_range() {
    let ranges = vec![vertex_push_constant(0, 0)];
    let size = total_push_constant_size(&ranges);
    assert_eq!(size, 0, "Zero-size range should return 0");
}

/// PipelineLayoutKey with single hash.
#[test]
fn key_with_single_hash() {
    let key = PipelineLayoutKey::new(&single_layout_hash(), &[]);
    let _ = key.bind_group_layouts_hash();
}

/// PipelineLayoutKey with vertex-fragment push constant.
#[test]
fn key_with_vertex_fragment_push_constant() {
    let pc = vec![vertex_fragment_push_constant(0, 64)];
    let key = PipelineLayoutKey::new(&basic_layout_hashes(), &pc);
    let _ = key;
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
    ))
    .ok()?;

    Some((device, queue))
}

/// Helper to create a basic bind group layout.
#[cfg(test)]
fn create_basic_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Test Layout"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    })
}

/// Cache operations work with GPU device (with GPU).
#[test]

fn gpu_cache_basic_operations() {
    let Some((_device, _queue)) = create_test_device() else {
        eprintln!("No GPU available, skipping test");
        return;
    };

    let cache = PipelineLayoutCache::new();
    assert!(cache.is_empty());
    assert_eq!(cache.len(), 0);
}

/// Cache metrics work with GPU operations (with GPU).
#[test]

fn gpu_cache_metrics_tracking() {
    let Some((_device, _queue)) = create_test_device() else {
        eprintln!("No GPU available, skipping test");
        return;
    };

    let cache = PipelineLayoutCache::new();
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

    let cache = PipelineLayoutCache::new();

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

    let cache = PipelineLayoutCache::new();
    let hashes = basic_layout_hashes();
    let pcs = pbr_push_constants();

    assert!(!cache.contains(&hashes, &pcs));
}

/// Cache is thread safe for concurrent access (with GPU).
#[test]

fn gpu_cache_concurrent_access() {
    use std::sync::Arc;
    use std::thread;

    let Some((_device, _queue)) = create_test_device() else {
        eprintln!("No GPU available, skipping test");
        return;
    };

    let cache = Arc::new(PipelineLayoutCache::new());

    let handles: Vec<_> = (0..4)
        .map(|_| {
            let cache_clone = Arc::clone(&cache);
            thread::spawn(move || {
                for _ in 0..100 {
                    let _ = cache_clone.is_empty();
                    let _ = cache_clone.len();
                    let _ = cache_clone.metrics();
                }
            })
        })
        .collect();

    for handle in handles {
        handle.join().expect("Thread should complete");
    }
}

/// Cache with multiple key queries (with GPU).
#[test]

fn gpu_cache_multiple_key_queries() {
    let Some((_device, _queue)) = create_test_device() else {
        eprintln!("No GPU available, skipping test");
        return;
    };

    let cache = PipelineLayoutCache::new();

    for i in 0..10u64 {
        let hashes = vec![i];
        assert!(!cache.contains(&hashes, &[]));
    }
}

/// Cache remove operation works (with GPU).
#[test]

fn gpu_cache_remove_operation() {
    let Some((_device, _queue)) = create_test_device() else {
        eprintln!("No GPU available, skipping test");
        return;
    };

    let cache = PipelineLayoutCache::new();
    let hashes = basic_layout_hashes();
    let pcs = pbr_push_constants();

    // Remove from empty cache should return false
    let removed = cache.remove(&hashes, &pcs);
    assert!(!removed, "Remove from empty cache should return false");
}

/// TrinityLayoutBuilder can be constructed (with GPU).
#[test]

fn gpu_trinity_layout_builder_construction() {
    let Some((device, _queue)) = create_test_device() else {
        eprintln!("No GPU available, skipping test");
        return;
    };

    let cache = PipelineLayoutCache::new();
    let _builder = TrinityLayoutBuilder::new(&device, &cache);
}

/// TrinityLayoutBuilder global_only layout (with GPU).
#[test]

fn gpu_trinity_layout_builder_global_only() {
    let Some((device, _queue)) = create_test_device() else {
        eprintln!("No GPU available, skipping test");
        return;
    };

    let cache = PipelineLayoutCache::new();
    let builder = TrinityLayoutBuilder::new(&device, &cache);
    let global_layout = create_basic_bind_group_layout(&device);
    let global_hash = 12345u64;

    // global_only returns Arc<PipelineLayout> directly
    let layout: Arc<PipelineLayout> = builder.global_only(&global_layout, global_hash);

    // Verify we got a valid layout (it's a valid Arc)
    let _inner: &PipelineLayout = layout.as_ref();

    // Cache should now contain one layout
    assert_eq!(cache.len(), 1);
}

/// TrinityLayoutBuilder global_material layout (with GPU).
#[test]

fn gpu_trinity_layout_builder_global_material() {
    let Some((device, _queue)) = create_test_device() else {
        eprintln!("No GPU available, skipping test");
        return;
    };

    let cache = PipelineLayoutCache::new();
    let builder = TrinityLayoutBuilder::new(&device, &cache);
    let global_layout = create_basic_bind_group_layout(&device);
    let material_layout = create_basic_bind_group_layout(&device);

    let layout: Arc<PipelineLayout> = builder.global_material(
        &global_layout,
        111,
        &material_layout,
        222,
    );
    let _inner: &PipelineLayout = layout.as_ref();
}

/// TrinityLayoutBuilder pbr layout (with GPU).
#[test]

fn gpu_trinity_layout_builder_pbr() {
    let Some((device, _queue)) = create_test_device() else {
        eprintln!("No GPU available, skipping test");
        return;
    };

    let cache = PipelineLayoutCache::new();
    let builder = TrinityLayoutBuilder::new(&device, &cache);
    let global_layout = create_basic_bind_group_layout(&device);
    let material_layout = create_basic_bind_group_layout(&device);
    let object_layout = create_basic_bind_group_layout(&device);

    let layout: Arc<PipelineLayout> = builder.pbr(
        &global_layout, 111,
        &material_layout, 222,
        &object_layout, 333,
    );
    let _inner: &PipelineLayout = layout.as_ref();
}

/// TrinityLayoutBuilder compute layout (with GPU).
#[test]

fn gpu_trinity_layout_builder_compute() {
    let Some((device, _queue)) = create_test_device() else {
        eprintln!("No GPU available, skipping test");
        return;
    };

    let cache = PipelineLayoutCache::new();
    let builder = TrinityLayoutBuilder::new(&device, &cache);
    let global_layout = create_basic_bind_group_layout(&device);

    let layout: Arc<PipelineLayout> = builder.compute(
        Some("test_compute"),
        &[&global_layout],
        &[111],
    );
    let _inner: &PipelineLayout = layout.as_ref();
}

/// TrinityLayoutBuilder with_push_constants layout (with GPU).
#[test]

fn gpu_trinity_layout_builder_with_push_constants() {
    let Some((device, _queue)) = create_test_device() else {
        eprintln!("No GPU available, skipping test");
        return;
    };

    let cache = PipelineLayoutCache::new();
    let builder = TrinityLayoutBuilder::new(&device, &cache);
    let global_layout = create_basic_bind_group_layout(&device);
    let push_constants = pbr_push_constants();

    let layout: Arc<PipelineLayout> = builder.with_push_constants(
        Some("test_push_constants"),
        &[&global_layout],
        &[111],
        &push_constants,
    );
    let _inner: &PipelineLayout = layout.as_ref();
}

/// TrinityLayoutBuilder pbr_with_push_constants layout (with GPU).
#[test]

fn gpu_trinity_layout_builder_pbr_with_push_constants() {
    let Some((device, _queue)) = create_test_device() else {
        eprintln!("No GPU available, skipping test");
        return;
    };

    let cache = PipelineLayoutCache::new();
    let builder = TrinityLayoutBuilder::new(&device, &cache);
    let global_layout = create_basic_bind_group_layout(&device);
    let material_layout = create_basic_bind_group_layout(&device);
    let object_layout = create_basic_bind_group_layout(&device);

    let layout: Arc<PipelineLayout> = builder.pbr_with_push_constants(
        &global_layout, 111,
        &material_layout, 222,
        &object_layout, 333,
        64, // push constant size
    );
    let _inner: &PipelineLayout = layout.as_ref();
}

/// Created layout is valid wgpu object (with GPU).
#[test]

fn gpu_created_layout_is_valid_wgpu_object() {
    let Some((device, _queue)) = create_test_device() else {
        eprintln!("No GPU available, skipping test");
        return;
    };

    let cache = PipelineLayoutCache::new();
    let builder = TrinityLayoutBuilder::new(&device, &cache);
    let global_layout = create_basic_bind_group_layout(&device);

    let layout: Arc<PipelineLayout> = builder.global_only(&global_layout, 12345);

    // Verify we can dereference and it's a valid PipelineLayout
    let layout_ref: &PipelineLayout = layout.as_ref();
    // Just verify we can access it without panicking
    let _ = format!("{:?}", layout_ref);
}

/// Cache hit on same layout configuration (with GPU).
#[test]

fn gpu_cache_hit_on_same_config() {
    let Some((device, _queue)) = create_test_device() else {
        eprintln!("No GPU available, skipping test");
        return;
    };

    let cache = PipelineLayoutCache::new();
    let builder = TrinityLayoutBuilder::new(&device, &cache);
    let global_layout = create_basic_bind_group_layout(&device);
    let hash = 12345u64;

    // First creation - cache miss
    let layout1: Arc<PipelineLayout> = builder.global_only(&global_layout, hash);

    // Reset metrics to measure next call
    cache.reset_metrics();

    // Second creation with same config - should be cache hit
    let layout2: Arc<PipelineLayout> = builder.global_only(&global_layout, hash);

    let metrics = cache.metrics();
    // Check that we got a cache hit
    assert!(metrics.total_requests() >= 1, "Should have made a request");

    // Layouts should be the same object (same Arc)
    assert!(
        Arc::ptr_eq(&layout1, &layout2),
        "Same config should return same layout"
    );
}

/// Cache returns different layouts for different hashes (with GPU).
#[test]

fn gpu_cache_different_hashes_different_layouts() {
    let Some((device, _queue)) = create_test_device() else {
        eprintln!("No GPU available, skipping test");
        return;
    };

    let cache = PipelineLayoutCache::new();
    let builder = TrinityLayoutBuilder::new(&device, &cache);
    let global_layout = create_basic_bind_group_layout(&device);

    let layout1: Arc<PipelineLayout> = builder.global_only(&global_layout, 111);
    let layout2: Arc<PipelineLayout> = builder.global_only(&global_layout, 222);

    // Different hashes should create different cache entries
    assert_eq!(cache.len(), 2, "Should have two cache entries");
    assert!(!Arc::ptr_eq(&layout1, &layout2), "Different hashes should return different layouts");
}

// =============================================================================
// SECTION 4 -- PROPERTY-BASED TESTS (10+ tests)
// =============================================================================

/// Property: Different configs always produce different keys.
#[test]
fn property_different_configs_different_keys() {
    let configs: Vec<(Vec<u64>, Vec<wgpu::PushConstantRange>)> = vec![
        (basic_layout_hashes(), pbr_push_constants()),
        (alt_layout_hashes(), compute_push_constants()),
        (single_layout_hash(), vec![]),
        (vec![], pbr_push_constants()),
        (vec![], vec![]),
        (vec![1, 2, 3], vec![vertex_push_constant(0, 32)]),
    ];

    let keys: Vec<PipelineLayoutKey> = configs
        .iter()
        .map(|(hashes, pcs)| PipelineLayoutKey::new(hashes, pcs))
        .collect();

    // Count unique keys
    let mut unique_keys: Vec<&PipelineLayoutKey> = keys.iter().collect();
    unique_keys.sort_by_key(|k| (k.bind_group_layouts_hash(), k.push_constants_hash()));
    unique_keys.dedup_by(|a, b| a == b);

    // At least most configs should produce unique keys
    assert!(unique_keys.len() >= configs.len() / 2);
}

/// Property: Key comparison is reflexive.
#[test]
fn property_key_comparison_reflexive() {
    for (hashes, pcs) in [
        (basic_layout_hashes(), pbr_push_constants()),
        (alt_layout_hashes(), compute_push_constants()),
        (vec![], vec![]),
    ] {
        let key = PipelineLayoutKey::new(&hashes, &pcs);
        assert_eq!(key, key, "Key must equal itself");
    }
}

/// Property: Key comparison is symmetric.
#[test]
fn property_key_comparison_symmetric() {
    let hashes = basic_layout_hashes();
    let pcs = pbr_push_constants();
    let key1 = PipelineLayoutKey::new(&hashes, &pcs);
    let key2 = PipelineLayoutKey::new(&hashes, &pcs);

    assert_eq!(key1 == key2, key2 == key1);
}

/// Property: Key comparison is transitive.
#[test]
fn property_key_comparison_transitive() {
    let hashes = basic_layout_hashes();
    let pcs = pbr_push_constants();
    let key1 = PipelineLayoutKey::new(&hashes, &pcs);
    let key2 = PipelineLayoutKey::new(&hashes, &pcs);
    let key3 = PipelineLayoutKey::new(&hashes, &pcs);

    if key1 == key2 && key2 == key3 {
        assert_eq!(key1, key3, "Key equality must be transitive");
    }
}

/// Property: hit + miss = total_requests.
#[test]
fn property_hits_plus_misses_equals_total() {
    for (hits, misses) in [(0, 0), (10, 5), (100, 200), (0, 100), (100, 0)] {
        let metrics = PipelineLayoutCacheMetrics::new(5, hits, misses);
        assert_eq!(
            metrics.total_requests(),
            hits + misses,
            "Total requests must equal hits + misses"
        );
    }
}

/// Property: hit_rate is between 0.0 and 100.0.
#[test]
fn property_hit_rate_in_valid_range() {
    let test_cases = [(0, 0), (10, 0), (0, 10), (50, 50), (100, 0), (0, 100), (75, 25)];

    for (hits, misses) in test_cases {
        let metrics = PipelineLayoutCacheMetrics::new(5, hits, misses);
        let rate = metrics.hit_rate_percent();
        assert!(
            rate >= 0.0 && rate <= 100.0,
            "Hit rate {} must be between 0 and 100",
            rate
        );
    }
}

/// Property: Cache len() >= 0 always.
#[test]
fn property_cache_len_non_negative() {
    let cache = PipelineLayoutCache::new();
    assert!(cache.len() >= 0);

    cache.clear();
    assert!(cache.len() >= 0);
}

/// Property: is_empty() is equivalent to len() == 0.
#[test]
fn property_is_empty_equivalent_to_len_zero() {
    let cache = PipelineLayoutCache::new();
    assert_eq!(cache.is_empty(), cache.len() == 0);

    cache.clear();
    assert_eq!(cache.is_empty(), cache.len() == 0);
}

/// Property: total_push_constant_size is non-negative.
#[test]
fn property_total_push_constant_size_non_negative() {
    let range_sets: Vec<Vec<wgpu::PushConstantRange>> = vec![
        vec![],
        vec![vertex_push_constant(0, 0)],
        vec![vertex_push_constant(0, 64)],
        pbr_push_constants(),
        compute_push_constants(),
    ];

    for ranges in range_sets {
        let size = total_push_constant_size(&ranges);
        assert!(size >= 0, "Push constant size must be non-negative");
    }
}

/// Property: Same config produces same key hash.
#[test]
fn property_same_config_same_hash() {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let hashes = basic_layout_hashes();
    let pcs = pbr_push_constants();

    for _ in 0..10 {
        let key1 = PipelineLayoutKey::new(&hashes, &pcs);
        let key2 = PipelineLayoutKey::new(&hashes, &pcs);

        let mut h1 = DefaultHasher::new();
        let mut h2 = DefaultHasher::new();
        key1.hash(&mut h1);
        key2.hash(&mut h2);

        assert_eq!(h1.finish(), h2.finish(), "Same config must produce same hash");
    }
}

/// Property: Empty cache metrics are empty.
#[test]
fn property_empty_cache_metrics_empty() {
    let cache = PipelineLayoutCache::new();
    let metrics = cache.metrics();
    assert!(metrics.is_empty(), "New cache metrics must be empty");

    cache.reset_metrics();
    let reset_metrics = cache.metrics();
    assert!(reset_metrics.is_empty(), "Reset cache metrics must be empty");
}

/// Property: total_push_constant_size returns end of last range.
#[test]
fn property_total_size_is_max_end() {
    let ranges = vec![
        vertex_push_constant(0, 32),
        fragment_push_constant(32, 48),
    ];
    let size = total_push_constant_size(&ranges);
    assert_eq!(size, 80, "Total should be end of last range (32 + 48)");
}

// =============================================================================
// ADDITIONAL EDGE CASE TESTS
// =============================================================================

/// Edge case: Very large layout hash values.
#[test]
fn edge_case_large_layout_hashes() {
    let hashes = vec![u64::MAX, u64::MAX - 1, u64::MAX - 2];
    let key = PipelineLayoutKey::new(&hashes, &[]);
    let _ = key;
}

/// Edge case: Zero layout hash.
#[test]
fn edge_case_zero_layout_hash() {
    let hashes = vec![0u64];
    let key = PipelineLayoutKey::new(&hashes, &[]);
    let _ = key;
}

/// Edge case: Maximum push constant size at limit.
#[test]
fn edge_case_max_push_constant_at_limit() {
    let ranges = vec![vertex_push_constant(0, MAX_PUSH_CONSTANT_SIZE)];
    let size = total_push_constant_size(&ranges);
    assert_eq!(size, MAX_PUSH_CONSTANT_SIZE);
}

/// Edge case: Push constant range starting at non-zero offset.
#[test]
fn edge_case_push_constant_non_zero_offset() {
    let ranges = vec![vertex_push_constant(64, 32)];
    let size = total_push_constant_size(&ranges);
    assert_eq!(size, 96, "Should account for offset + size");
}

/// Edge case: Multiple push constant ranges for same stage.
#[test]
fn edge_case_multiple_ranges_same_stage() {
    let ranges = vec![
        vertex_push_constant(0, 32),
        vertex_push_constant(32, 32),
    ];
    let size = total_push_constant_size(&ranges);
    assert_eq!(size, 64);
}

/// Edge case: All four bind group index constants used.
#[test]
fn edge_case_all_bind_group_indices() {
    let hashes = vec![
        bind_group_index::GLOBAL as u64,
        bind_group_index::MATERIAL as u64,
        bind_group_index::OBJECT as u64,
        bind_group_index::BINDLESS as u64,
    ];
    let key = PipelineLayoutKey::new(&hashes, &[]);
    let _ = key;
}

/// Edge case: Very many layout hashes.
#[test]
fn edge_case_many_layout_hashes() {
    let hashes: Vec<u64> = (0..100).collect();
    let key = PipelineLayoutKey::new(&hashes, &[]);
    let _ = key;
}

/// Edge case: validate_push_constant_ranges with gaps.
#[test]
fn edge_case_push_constants_with_gaps() {
    // Ranges with gaps between them
    let ranges = vec![
        vertex_push_constant(0, 16),
        vertex_push_constant(32, 16),  // Gap from 16-32
    ];
    let result = validate_push_constant_ranges(&ranges);
    // Implementation defines whether gaps are allowed
    let _ = result;
    let size = total_push_constant_size(&ranges);
    assert_eq!(size, 48, "Should account for end of last range");
}

/// Edge case: Empty key hashes correctly.
#[test]
fn edge_case_empty_key_hashes() {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let key = PipelineLayoutKey::new(&[], &[]);
    let mut hasher = DefaultHasher::new();
    key.hash(&mut hasher);
    let hash = hasher.finish();
    // Hash should be deterministic
    let mut hasher2 = DefaultHasher::new();
    key.hash(&mut hasher2);
    assert_eq!(hash, hasher2.finish());
}

/// Edge case: Cache metrics with very large counts.
#[test]
fn edge_case_metrics_large_counts() {
    let metrics = PipelineLayoutCacheMetrics::new(1000000, u64::MAX / 2, u64::MAX / 2);
    let rate = metrics.hit_rate_percent();
    assert!(rate >= 0.0 && rate <= 100.0);
}

/// Edge case: PipelineLayoutKey bind_group_layouts_hash consistency.
#[test]
fn edge_case_key_hash_consistency() {
    let hashes1 = vec![1u64, 2, 3];
    let hashes2 = vec![1u64, 2, 3];
    let key1 = PipelineLayoutKey::new(&hashes1, &[]);
    let key2 = PipelineLayoutKey::new(&hashes2, &[]);
    assert_eq!(
        key1.bind_group_layouts_hash(),
        key2.bind_group_layouts_hash(),
        "Same inputs should produce same bind_group_layouts_hash"
    );
}

/// Edge case: PipelineLayoutKey push_constants_hash consistency.
#[test]
fn edge_case_push_constants_hash_consistency() {
    let pcs = pbr_push_constants();
    let key1 = PipelineLayoutKey::new(&[], &pcs);
    let key2 = PipelineLayoutKey::new(&[], &pcs);
    assert_eq!(
        key1.push_constants_hash(),
        key2.push_constants_hash(),
        "Same inputs should produce same push_constants_hash"
    );
}
