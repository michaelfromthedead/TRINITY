// SPDX-License-Identifier: MIT
//
// WHITEBOX T-WGPU-P2.5.3 Pipeline Layout
// Contract: Pipeline layout caching, validation, and builder patterns.
//
// This test suite has FULL SOURCE ACCESS and tests all internal implementation
// details of the pipeline layout cache system.
//
// Categories tested:
//   1. PipelineLayoutKey - Hashing, equality, HashMap usage
//   2. CachedPipelineLayout - Metadata access, Arc sharing
//   3. PipelineLayoutCache - Caching, metrics, thread safety
//   4. PipelineLayoutCacheMetrics - Statistics calculations
//   5. Push Constant Validation - WebGPU limits, alignment, overlap
//   6. TrinityLayoutBuilder - Standard layout patterns
//   7. bind_group_index constants - Standard indices

use renderer_backend::resources::pipeline_layout::{
    bind_group_index, total_push_constant_size, validate_push_constant_ranges,
    PipelineLayoutCache, PipelineLayoutCacheMetrics, PipelineLayoutKey, TrinityLayoutBuilder,
    MAX_PUSH_CONSTANT_SIZE,
};
use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use std::thread;
use wgpu::{PushConstantRange, ShaderStages};

// =============================================================================
// Helper Functions
// =============================================================================

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
                label: Some("whitebox_pipeline_layout test device"),
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
                label: Some("whitebox_pipeline_layout test device (no push)"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        ))
        .expect("device creation"),
    )
}

fn create_empty_bind_group_layout(device: &wgpu::Device, label: &str) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(label),
        entries: &[],
    })
}

fn create_uniform_bind_group_layout(device: &wgpu::Device, label: &str) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(label),
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
    })
}

fn create_storage_bind_group_layout(device: &wgpu::Device, label: &str) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(label),
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
    })
}

// =============================================================================
// SECTION 1: bind_group_index Constants (5+ tests)
// =============================================================================

#[test]
fn bind_group_index_global_is_zero() {
    assert_eq!(bind_group_index::GLOBAL, 0);
}

#[test]
fn bind_group_index_material_is_one() {
    assert_eq!(bind_group_index::MATERIAL, 1);
}

#[test]
fn bind_group_index_object_is_two() {
    assert_eq!(bind_group_index::OBJECT, 2);
}

#[test]
fn bind_group_index_bindless_is_three() {
    assert_eq!(bind_group_index::BINDLESS, 3);
}

#[test]
fn bind_group_indices_are_sequential() {
    assert_eq!(bind_group_index::GLOBAL + 1, bind_group_index::MATERIAL);
    assert_eq!(bind_group_index::MATERIAL + 1, bind_group_index::OBJECT);
    assert_eq!(bind_group_index::OBJECT + 1, bind_group_index::BINDLESS);
}

#[test]
fn bind_group_indices_are_unique() {
    let indices = [
        bind_group_index::GLOBAL,
        bind_group_index::MATERIAL,
        bind_group_index::OBJECT,
        bind_group_index::BINDLESS,
    ];
    let set: HashSet<_> = indices.iter().collect();
    assert_eq!(set.len(), 4, "All bind group indices must be unique");
}

#[test]
fn max_push_constant_size_is_webgpu_limit() {
    assert_eq!(MAX_PUSH_CONSTANT_SIZE, 128);
}

// =============================================================================
// SECTION 2: PipelineLayoutKey (20+ tests)
// =============================================================================

#[test]
fn key_empty_bind_groups_empty_push_constants() {
    let key = PipelineLayoutKey::new(&[], &[]);
    // Even empty inputs produce non-zero hashes due to length hashing
    assert_ne!(key.bind_group_layouts_hash(), 0);
    assert_ne!(key.push_constants_hash(), 0);
}

#[test]
fn key_single_bind_group_no_push_constants() {
    let key = PipelineLayoutKey::new(&[0x1234], &[]);
    assert_ne!(key.bind_group_layouts_hash(), 0);
}

#[test]
fn key_multiple_bind_groups_no_push_constants() {
    let key = PipelineLayoutKey::new(&[0x1234, 0x5678, 0x9ABC, 0xDEF0], &[]);
    assert_ne!(key.bind_group_layouts_hash(), 0);
}

#[test]
fn key_with_single_push_constant_range() {
    let ranges = &[PushConstantRange {
        stages: ShaderStages::VERTEX,
        range: 0..64,
    }];
    let key = PipelineLayoutKey::new(&[0x1234], ranges);
    assert_ne!(key.push_constants_hash(), 0);
}

#[test]
fn key_with_multiple_push_constant_ranges() {
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
    let key = PipelineLayoutKey::new(&[0x1111], ranges);
    assert_ne!(key.push_constants_hash(), 0);
}

#[test]
fn key_equality_same_config() {
    let key1 = PipelineLayoutKey::new(&[0x1234, 0x5678], &[]);
    let key2 = PipelineLayoutKey::new(&[0x1234, 0x5678], &[]);
    assert_eq!(key1, key2);
}

#[test]
fn key_equality_with_push_constants() {
    let ranges = &[PushConstantRange {
        stages: ShaderStages::VERTEX | ShaderStages::FRAGMENT,
        range: 0..128,
    }];
    let key1 = PipelineLayoutKey::new(&[0xAAAA], ranges);
    let key2 = PipelineLayoutKey::new(&[0xAAAA], ranges);
    assert_eq!(key1, key2);
}

#[test]
fn key_inequality_different_hashes() {
    let key1 = PipelineLayoutKey::new(&[0x1234, 0x5678], &[]);
    let key2 = PipelineLayoutKey::new(&[0x1234, 0xABCD], &[]);
    assert_ne!(key1, key2);
}

#[test]
fn key_inequality_different_order() {
    // Order matters for pipeline layouts
    let key1 = PipelineLayoutKey::new(&[0x1234, 0x5678], &[]);
    let key2 = PipelineLayoutKey::new(&[0x5678, 0x1234], &[]);
    assert_ne!(key1, key2);
}

#[test]
fn key_inequality_different_count() {
    let key1 = PipelineLayoutKey::new(&[0x1234], &[]);
    let key2 = PipelineLayoutKey::new(&[0x1234, 0x5678], &[]);
    assert_ne!(key1, key2);
}

#[test]
fn key_inequality_different_push_constant_stages() {
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
fn key_inequality_different_push_constant_ranges() {
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
fn key_inequality_different_push_constant_count() {
    let ranges1 = &[PushConstantRange {
        stages: ShaderStages::VERTEX,
        range: 0..64,
    }];
    let ranges2 = &[
        PushConstantRange {
            stages: ShaderStages::VERTEX,
            range: 0..32,
        },
        PushConstantRange {
            stages: ShaderStages::FRAGMENT,
            range: 32..64,
        },
    ];
    let key1 = PipelineLayoutKey::new(&[0x1234], ranges1);
    let key2 = PipelineLayoutKey::new(&[0x1234], ranges2);
    assert_ne!(key1, key2);
}

#[test]
fn key_hash_stability() {
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
fn key_hashmap_insertion_and_lookup() {
    let mut map: HashMap<PipelineLayoutKey, i32> = HashMap::new();

    let key1 = PipelineLayoutKey::new(&[0x1111], &[]);
    let key2 = PipelineLayoutKey::new(&[0x2222], &[]);
    let key3 = PipelineLayoutKey::new(&[0x3333], &[]);

    map.insert(key1.clone(), 100);
    map.insert(key2.clone(), 200);
    map.insert(key3.clone(), 300);

    assert_eq!(map.len(), 3);
    assert_eq!(map.get(&key1), Some(&100));
    assert_eq!(map.get(&key2), Some(&200));
    assert_eq!(map.get(&key3), Some(&300));
}

#[test]
fn key_hashmap_overwrite() {
    let mut map: HashMap<PipelineLayoutKey, i32> = HashMap::new();

    let key = PipelineLayoutKey::new(&[0x1234], &[]);
    map.insert(key.clone(), 100);
    map.insert(key.clone(), 200);

    assert_eq!(map.len(), 1);
    assert_eq!(map.get(&key), Some(&200));
}

#[test]
fn key_hashmap_absent_lookup() {
    let map: HashMap<PipelineLayoutKey, i32> = HashMap::new();
    let key = PipelineLayoutKey::new(&[0x1234], &[]);
    assert_eq!(map.get(&key), None);
}

#[test]
fn key_clone_trait() {
    let key = PipelineLayoutKey::new(&[0x1234, 0x5678], &[]);
    let cloned = key.clone();
    assert_eq!(key, cloned);
}

#[test]
fn key_debug_trait() {
    let key = PipelineLayoutKey::new(&[0x1234], &[]);
    let debug_str = format!("{:?}", key);
    assert!(debug_str.contains("PipelineLayoutKey"));
    assert!(debug_str.contains("bind_group_layouts_hash"));
    assert!(debug_str.contains("push_constants_hash"));
}

#[test]
fn key_hash_trait() {
    use std::collections::hash_map::DefaultHasher;

    let key = PipelineLayoutKey::new(&[0x1234], &[]);
    let mut hasher = DefaultHasher::new();
    key.hash(&mut hasher);
    let hash = hasher.finish();
    assert_ne!(hash, 0);
}

#[test]
fn key_bind_group_layouts_hash_accessor() {
    let key = PipelineLayoutKey::new(&[0xAAAA, 0xBBBB], &[]);
    let hash = key.bind_group_layouts_hash();
    assert_ne!(hash, 0);
}

#[test]
fn key_push_constants_hash_accessor() {
    let ranges = &[PushConstantRange {
        stages: ShaderStages::COMPUTE,
        range: 0..64,
    }];
    let key = PipelineLayoutKey::new(&[], ranges);
    let hash = key.push_constants_hash();
    assert_ne!(hash, 0);
}

#[test]
fn key_deterministic_across_calls() {
    for _ in 0..10 {
        let key1 = PipelineLayoutKey::new(&[0x1234, 0x5678], &[]);
        let key2 = PipelineLayoutKey::new(&[0x1234, 0x5678], &[]);
        assert_eq!(key1.bind_group_layouts_hash(), key2.bind_group_layouts_hash());
    }
}

// =============================================================================
// SECTION 3: PipelineLayoutCacheMetrics (15+ tests)
// =============================================================================

#[test]
fn metrics_default_is_empty() {
    let metrics = PipelineLayoutCacheMetrics::default();
    assert_eq!(metrics.cache_size, 0);
    assert_eq!(metrics.hits, 0);
    assert_eq!(metrics.misses, 0);
    assert_eq!(metrics.hit_rate, 0.0);
}

#[test]
fn metrics_new_calculates_hit_rate() {
    let metrics = PipelineLayoutCacheMetrics::new(5, 80, 20);
    assert_eq!(metrics.cache_size, 5);
    assert_eq!(metrics.hits, 80);
    assert_eq!(metrics.misses, 20);
    assert!((metrics.hit_rate - 0.8).abs() < 0.001);
}

#[test]
fn metrics_new_zero_total_gives_zero_hit_rate() {
    let metrics = PipelineLayoutCacheMetrics::new(0, 0, 0);
    assert_eq!(metrics.hit_rate, 0.0);
}

#[test]
fn metrics_new_all_hits() {
    let metrics = PipelineLayoutCacheMetrics::new(10, 100, 0);
    assert!((metrics.hit_rate - 1.0).abs() < 0.001);
}

#[test]
fn metrics_new_all_misses() {
    let metrics = PipelineLayoutCacheMetrics::new(10, 0, 100);
    assert_eq!(metrics.hit_rate, 0.0);
}

#[test]
fn metrics_total_requests() {
    let metrics = PipelineLayoutCacheMetrics::new(3, 50, 25);
    assert_eq!(metrics.total_requests(), 75);
}

#[test]
fn metrics_total_requests_zero() {
    let metrics = PipelineLayoutCacheMetrics::new(0, 0, 0);
    assert_eq!(metrics.total_requests(), 0);
}

#[test]
fn metrics_is_empty_when_cache_size_zero() {
    let empty = PipelineLayoutCacheMetrics::new(0, 10, 5);
    assert!(empty.is_empty());
}

#[test]
fn metrics_is_not_empty_when_cache_size_nonzero() {
    let non_empty = PipelineLayoutCacheMetrics::new(1, 0, 1);
    assert!(!non_empty.is_empty());
}

#[test]
fn metrics_hit_rate_percent() {
    let metrics = PipelineLayoutCacheMetrics::new(2, 75, 25);
    assert!((metrics.hit_rate_percent() - 75.0).abs() < 0.001);
}

#[test]
fn metrics_hit_rate_percent_zero() {
    let metrics = PipelineLayoutCacheMetrics::new(0, 0, 0);
    assert_eq!(metrics.hit_rate_percent(), 0.0);
}

#[test]
fn metrics_hit_rate_percent_hundred() {
    let metrics = PipelineLayoutCacheMetrics::new(5, 100, 0);
    assert!((metrics.hit_rate_percent() - 100.0).abs() < 0.001);
}

#[test]
fn metrics_clone_trait() {
    let metrics = PipelineLayoutCacheMetrics::new(10, 100, 50);
    let cloned = metrics.clone();

    assert_eq!(cloned.cache_size, metrics.cache_size);
    assert_eq!(cloned.hits, metrics.hits);
    assert_eq!(cloned.misses, metrics.misses);
    assert_eq!(cloned.hit_rate, metrics.hit_rate);
}

#[test]
fn metrics_debug_trait() {
    let metrics = PipelineLayoutCacheMetrics::new(3, 10, 5);
    let debug_str = format!("{:?}", metrics);
    assert!(debug_str.contains("PipelineLayoutCacheMetrics"));
    assert!(debug_str.contains("cache_size"));
    assert!(debug_str.contains("hits"));
    assert!(debug_str.contains("misses"));
    assert!(debug_str.contains("hit_rate"));
}

#[test]
fn metrics_send_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<PipelineLayoutCacheMetrics>();
}

#[test]
fn metrics_large_values() {
    let metrics = PipelineLayoutCacheMetrics::new(1_000_000, u64::MAX / 2, u64::MAX / 2);
    assert_eq!(metrics.cache_size, 1_000_000);
    assert!((metrics.hit_rate - 0.5).abs() < 0.001);
}

// =============================================================================
// SECTION 4: PipelineLayoutCache (25+ tests)
// =============================================================================

#[test]
fn cache_new_is_empty() {
    let cache = PipelineLayoutCache::new();
    assert!(cache.is_empty());
    assert_eq!(cache.len(), 0);
}

#[test]
fn cache_default_is_empty() {
    let cache = PipelineLayoutCache::default();
    assert!(cache.is_empty());
    assert_eq!(cache.len(), 0);
}

#[test]
fn cache_initial_metrics() {
    let cache = PipelineLayoutCache::new();
    let metrics = cache.metrics();

    assert_eq!(metrics.cache_size, 0);
    assert_eq!(metrics.hits, 0);
    assert_eq!(metrics.misses, 0);
    assert_eq!(metrics.hit_rate, 0.0);
}

#[test]
fn cache_len_reflects_size() {
    let cache = PipelineLayoutCache::new();
    assert_eq!(cache.len(), 0);

    // Manually insert via internal state would require device
    // Test the accessor at least
    assert_eq!(cache.len(), cache.metrics().cache_size);
}

#[test]
fn cache_is_empty_when_len_zero() {
    let cache = PipelineLayoutCache::new();
    assert!(cache.is_empty());
    assert_eq!(cache.len(), 0);
}

#[test]
fn cache_reset_metrics_zeros_counters() {
    let (device, _queue) = match create_test_device_no_push_constants() {
        Some(d) => d,
        None => {
            eprintln!("Skipping: no GPU adapter available");
            return;
        }
    };

    let layout = create_empty_bind_group_layout(&device, "test");
    let cache = PipelineLayoutCache::new();

    // Create some activity
    let _pl = cache.get_or_create(&device, Some("test"), &[&layout], &[0x1234], &[]);
    let _pl = cache.get_or_create(&device, Some("test"), &[&layout], &[0x1234], &[]);

    let before = cache.metrics();
    assert_eq!(before.misses, 1);
    assert_eq!(before.hits, 1);

    cache.reset_metrics();

    let after = cache.metrics();
    assert_eq!(after.hits, 0);
    assert_eq!(after.misses, 0);
    // Cache size should be preserved
    assert_eq!(after.cache_size, 1);
}

#[test]
fn cache_reset_metrics_preserves_cache() {
    let (device, _queue) = match create_test_device_no_push_constants() {
        Some(d) => d,
        None => {
            eprintln!("Skipping: no GPU adapter available");
            return;
        }
    };

    let layout = create_empty_bind_group_layout(&device, "test");
    let cache = PipelineLayoutCache::new();

    let _pl = cache.get_or_create(&device, Some("test"), &[&layout], &[0x1234], &[]);
    assert_eq!(cache.len(), 1);

    cache.reset_metrics();

    // Cache should still have the layout
    assert_eq!(cache.len(), 1);
    assert!(cache.contains(&[0x1234], &[]));
}

#[test]
fn cache_clear_empties_cache() {
    let (device, _queue) = match create_test_device_no_push_constants() {
        Some(d) => d,
        None => {
            eprintln!("Skipping: no GPU adapter available");
            return;
        }
    };

    let layout = create_empty_bind_group_layout(&device, "test");
    let cache = PipelineLayoutCache::new();

    let _pl = cache.get_or_create(&device, Some("test"), &[&layout], &[0x1234], &[]);

    cache.clear();

    assert!(cache.is_empty());
    let metrics = cache.metrics();
    assert_eq!(metrics.hits, 0);
    assert_eq!(metrics.misses, 0);
    assert_eq!(metrics.cache_size, 0);
}

#[test]
fn cache_labels_empty_initially() {
    let cache = PipelineLayoutCache::new();
    let labels: Vec<_> = cache.labels().collect();
    assert!(labels.is_empty());
}

#[test]
fn cache_debug_format() {
    let cache = PipelineLayoutCache::new();
    let debug_str = format!("{:?}", cache);

    assert!(debug_str.contains("PipelineLayoutCache"));
    assert!(debug_str.contains("cache_size"));
    assert!(debug_str.contains("hits"));
    assert!(debug_str.contains("misses"));
    assert!(debug_str.contains("hit_rate"));
}

#[test]
fn cache_contains_false_for_empty() {
    let cache = PipelineLayoutCache::new();
    assert!(!cache.contains(&[0x1234], &[]));
}

#[test]
fn cache_contains_false_for_nonexistent() {
    let cache = PipelineLayoutCache::new();
    assert!(!cache.contains(&[0x1234, 0x5678], &[]));
    assert!(!cache.contains(&[], &[]));
}

#[test]
fn cache_remove_nonexistent_returns_false() {
    let cache = PipelineLayoutCache::new();
    assert!(!cache.remove(&[0x1234], &[]));
}

#[test]
fn cache_send_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<PipelineLayoutCache>();
}

// Tests requiring GPU device

#[test]
fn cache_get_or_create_caches_layout() {
    let (device, _queue) = match create_test_device_no_push_constants() {
        Some(d) => d,
        None => {
            eprintln!("Skipping: no GPU adapter available");
            return;
        }
    };

    let bind_group_layout = create_empty_bind_group_layout(&device, "test");
    let cache = PipelineLayoutCache::new();

    // First call creates the layout
    let layout1 = cache.get_or_create(&device, Some("test"), &[&bind_group_layout], &[0x1234], &[]);
    assert_eq!(cache.len(), 1);
    assert_eq!(cache.metrics().misses, 1);
    assert_eq!(cache.metrics().hits, 0);

    // Second call returns cached layout
    let layout2 = cache.get_or_create(&device, Some("test"), &[&bind_group_layout], &[0x1234], &[]);
    assert_eq!(cache.len(), 1);
    assert_eq!(cache.metrics().misses, 1);
    assert_eq!(cache.metrics().hits, 1);

    // Same Arc
    assert!(Arc::ptr_eq(&layout1, &layout2));
}

#[test]
fn cache_get_or_create_different_hashes_create_different_layouts() {
    let (device, _queue) = match create_test_device_no_push_constants() {
        Some(d) => d,
        None => {
            eprintln!("Skipping: no GPU adapter available");
            return;
        }
    };

    let layout1 = create_empty_bind_group_layout(&device, "layout1");
    let layout2 = create_empty_bind_group_layout(&device, "layout2");

    let cache = PipelineLayoutCache::new();

    let pl1 = cache.get_or_create(&device, Some("pl1"), &[&layout1], &[0x1111], &[]);
    let pl2 = cache.get_or_create(&device, Some("pl2"), &[&layout2], &[0x2222], &[]);

    assert_eq!(cache.len(), 2);
    assert!(!Arc::ptr_eq(&pl1, &pl2));
}

#[test]
fn cache_get_or_create_multiple_bind_groups() {
    let (device, _queue) = match create_test_device_no_push_constants() {
        Some(d) => d,
        None => {
            eprintln!("Skipping: no GPU adapter available");
            return;
        }
    };

    let global = create_uniform_bind_group_layout(&device, "global");
    let material = create_empty_bind_group_layout(&device, "material");
    let object = create_empty_bind_group_layout(&device, "object");

    let cache = PipelineLayoutCache::new();

    let _layout = cache.get_or_create(
        &device,
        Some("pbr"),
        &[&global, &material, &object],
        &[0x1111, 0x2222, 0x3333],
        &[],
    );
    assert_eq!(cache.len(), 1);
}

#[test]
fn cache_get_or_create_with_push_constants() {
    let (device, _queue) = match create_test_device() {
        Some(d) => d,
        None => {
            eprintln!("Skipping: no GPU adapter with push constants available");
            return;
        }
    };

    let bind_group_layout = create_empty_bind_group_layout(&device, "test");
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
    let layout_no_push =
        cache.get_or_create(&device, Some("without_push"), &[&bind_group_layout], &[0x1234], &[]);
    assert_eq!(cache.len(), 2);

    assert!(!Arc::ptr_eq(&layout, &layout_no_push));
}

#[test]
fn cache_contains_true_after_creation() {
    let (device, _queue) = match create_test_device_no_push_constants() {
        Some(d) => d,
        None => {
            eprintln!("Skipping: no GPU adapter available");
            return;
        }
    };

    let bind_group_layout = create_empty_bind_group_layout(&device, "test");
    let cache = PipelineLayoutCache::new();

    assert!(!cache.contains(&[0x1234], &[]));

    let _layout = cache.get_or_create(&device, None, &[&bind_group_layout], &[0x1234], &[]);

    assert!(cache.contains(&[0x1234], &[]));
}

#[test]
fn cache_remove_existing_returns_true() {
    let (device, _queue) = match create_test_device_no_push_constants() {
        Some(d) => d,
        None => {
            eprintln!("Skipping: no GPU adapter available");
            return;
        }
    };

    let bind_group_layout = create_empty_bind_group_layout(&device, "test");
    let cache = PipelineLayoutCache::new();

    let _layout = cache.get_or_create(&device, None, &[&bind_group_layout], &[0x1234], &[]);
    assert_eq!(cache.len(), 1);

    assert!(cache.remove(&[0x1234], &[]));
    assert!(cache.is_empty());
    assert!(!cache.contains(&[0x1234], &[]));
}

#[test]
fn cache_remove_cannot_remove_again() {
    let (device, _queue) = match create_test_device_no_push_constants() {
        Some(d) => d,
        None => {
            eprintln!("Skipping: no GPU adapter available");
            return;
        }
    };

    let bind_group_layout = create_empty_bind_group_layout(&device, "test");
    let cache = PipelineLayoutCache::new();

    let _layout = cache.get_or_create(&device, None, &[&bind_group_layout], &[0x1234], &[]);
    assert!(cache.remove(&[0x1234], &[]));
    assert!(!cache.remove(&[0x1234], &[]));
}

#[test]
fn cache_labels_returns_cached_labels() {
    let (device, _queue) = match create_test_device_no_push_constants() {
        Some(d) => d,
        None => {
            eprintln!("Skipping: no GPU adapter available");
            return;
        }
    };

    let layout = create_empty_bind_group_layout(&device, "test");
    let cache = PipelineLayoutCache::new();

    let _pl1 = cache.get_or_create(&device, Some("first"), &[&layout], &[0x1111], &[]);
    let _pl2 = cache.get_or_create(&device, Some("second"), &[&layout], &[0x2222], &[]);
    let _pl3 = cache.get_or_create(&device, None, &[&layout], &[0x3333], &[]);

    let labels: Vec<_> = cache.labels().collect();
    assert_eq!(labels.len(), 3);

    // Check that expected labels are present
    assert!(labels.contains(&Some("first".to_string())));
    assert!(labels.contains(&Some("second".to_string())));
    assert!(labels.contains(&None));
}

#[test]
fn cache_hit_rate_calculation() {
    let (device, _queue) = match create_test_device_no_push_constants() {
        Some(d) => d,
        None => {
            eprintln!("Skipping: no GPU adapter available");
            return;
        }
    };

    let layout = create_empty_bind_group_layout(&device, "test");
    let cache = PipelineLayoutCache::new();

    // 1 miss
    let _pl = cache.get_or_create(&device, Some("test"), &[&layout], &[0x1234], &[]);

    // 3 hits
    for _ in 0..3 {
        let _pl = cache.get_or_create(&device, Some("test"), &[&layout], &[0x1234], &[]);
    }

    let metrics = cache.metrics();
    assert_eq!(metrics.misses, 1);
    assert_eq!(metrics.hits, 3);
    assert!((metrics.hit_rate - 0.75).abs() < 0.001);
}

#[test]
fn cache_clear_after_population() {
    let (device, _queue) = match create_test_device_no_push_constants() {
        Some(d) => d,
        None => {
            eprintln!("Skipping: no GPU adapter available");
            return;
        }
    };

    let layout = create_empty_bind_group_layout(&device, "test");
    let cache = PipelineLayoutCache::new();

    for i in 0..5 {
        let _pl = cache.get_or_create(&device, Some(&format!("pl{}", i)), &[&layout], &[i as u64], &[]);
    }

    assert_eq!(cache.len(), 5);

    cache.clear();

    assert!(cache.is_empty());
    let metrics = cache.metrics();
    assert_eq!(metrics.cache_size, 0);
    assert_eq!(metrics.hits, 0);
    assert_eq!(metrics.misses, 0);
}

#[test]
fn cache_concurrent_access() {
    let (device, _queue) = match create_test_device_no_push_constants() {
        Some(d) => d,
        None => {
            eprintln!("Skipping: no GPU adapter available");
            return;
        }
    };

    let device = Arc::new(device);
    let cache = Arc::new(PipelineLayoutCache::new());

    // Create bind group layouts for each thread
    let layouts: Vec<_> = (0..4)
        .map(|i| create_empty_bind_group_layout(&device, &format!("layout{}", i)))
        .collect();

    let handles: Vec<_> = (0..4)
        .map(|i| {
            let device = Arc::clone(&device);
            let cache = Arc::clone(&cache);
            let layout = &layouts[i];
            let layout_ptr = layout as *const _ as usize;
            let hash = (i * 0x1111) as u64;

            thread::spawn(move || {
                for _ in 0..100 {
                    let layout = unsafe { &*(layout_ptr as *const wgpu::BindGroupLayout) };
                    let _pl = cache.get_or_create(&device, None, &[layout], &[hash], &[]);
                }
            })
        })
        .collect();

    for handle in handles {
        handle.join().unwrap();
    }

    // Each thread created one unique layout (4 total), hit it 99 times
    assert_eq!(cache.len(), 4);
    let metrics = cache.metrics();
    assert_eq!(metrics.misses, 4);
    assert_eq!(metrics.hits, 396); // 4 threads * 99 hits each
}

#[test]
fn cache_double_check_pattern() {
    // Test that the double-check locking pattern works correctly
    let (device, _queue) = match create_test_device_no_push_constants() {
        Some(d) => d,
        None => {
            eprintln!("Skipping: no GPU adapter available");
            return;
        }
    };

    let layout = create_empty_bind_group_layout(&device, "test");
    let cache = PipelineLayoutCache::new();

    // Rapid successive calls should all hit the cache after the first
    let layouts: Vec<_> = (0..100)
        .map(|_| cache.get_or_create(&device, Some("test"), &[&layout], &[0x1234], &[]))
        .collect();

    assert_eq!(cache.len(), 1);
    assert_eq!(cache.metrics().misses, 1);
    assert_eq!(cache.metrics().hits, 99);

    // All should be the same Arc
    for l in &layouts {
        assert!(Arc::ptr_eq(l, &layouts[0]));
    }
}

// =============================================================================
// SECTION 5: Push Constant Validation (15+ tests)
// =============================================================================

#[test]
fn validate_push_constants_empty_is_ok() {
    assert!(validate_push_constant_ranges(&[]).is_ok());
}

#[test]
fn validate_push_constants_single_valid_range() {
    let ranges = &[PushConstantRange {
        stages: ShaderStages::VERTEX,
        range: 0..64,
    }];
    assert!(validate_push_constant_ranges(ranges).is_ok());
}

#[test]
fn validate_push_constants_max_size_ok() {
    let ranges = &[PushConstantRange {
        stages: ShaderStages::VERTEX,
        range: 0..128,
    }];
    assert!(validate_push_constant_ranges(ranges).is_ok());
}

#[test]
fn validate_push_constants_exceeds_limit() {
    let ranges = &[PushConstantRange {
        stages: ShaderStages::VERTEX,
        range: 0..256,
    }];
    let result = validate_push_constant_ranges(ranges);
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(err.contains("exceeds"));
    assert!(err.contains("256"));
    assert!(err.contains("128"));
}

#[test]
fn validate_push_constants_just_over_limit() {
    let ranges = &[PushConstantRange {
        stages: ShaderStages::VERTEX,
        range: 0..132,
    }];
    let result = validate_push_constant_ranges(ranges);
    assert!(result.is_err());
}

#[test]
fn validate_push_constants_non_overlapping_different_stages() {
    let ranges = &[
        PushConstantRange {
            stages: ShaderStages::VERTEX,
            range: 0..64,
        },
        PushConstantRange {
            stages: ShaderStages::FRAGMENT,
            range: 0..64,
        },
    ];
    // Same range but different stages - this overlaps when stages intersect
    // VERTEX and FRAGMENT don't intersect, so ranges can overlap
    assert!(validate_push_constant_ranges(ranges).is_ok());
}

#[test]
fn validate_push_constants_non_overlapping_same_stages() {
    let ranges = &[
        PushConstantRange {
            stages: ShaderStages::VERTEX,
            range: 0..32,
        },
        PushConstantRange {
            stages: ShaderStages::VERTEX,
            range: 32..64,
        },
    ];
    // Adjacent ranges (not overlapping) with same stages is ok
    assert!(validate_push_constant_ranges(ranges).is_ok());
}

#[test]
fn validate_push_constants_overlapping_same_stages() {
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
fn validate_push_constants_overlapping_intersecting_stages() {
    let ranges = &[
        PushConstantRange {
            stages: ShaderStages::VERTEX | ShaderStages::FRAGMENT,
            range: 0..64,
        },
        PushConstantRange {
            stages: ShaderStages::FRAGMENT,
            range: 32..96,
        },
    ];
    let result = validate_push_constant_ranges(ranges);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("Overlapping"));
}

#[test]
fn validate_push_constants_misaligned_start() {
    let ranges = &[PushConstantRange {
        stages: ShaderStages::VERTEX,
        range: 1..64,
    }];
    let result = validate_push_constant_ranges(ranges);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("aligned"));
}

#[test]
fn validate_push_constants_misaligned_end() {
    let ranges = &[PushConstantRange {
        stages: ShaderStages::VERTEX,
        range: 0..65,
    }];
    let result = validate_push_constant_ranges(ranges);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("aligned"));
}

#[test]
fn validate_push_constants_both_misaligned() {
    let ranges = &[PushConstantRange {
        stages: ShaderStages::VERTEX,
        range: 3..67,
    }];
    let result = validate_push_constant_ranges(ranges);
    assert!(result.is_err());
}

#[test]
fn validate_push_constants_multiple_valid_ranges() {
    let ranges = &[
        PushConstantRange {
            stages: ShaderStages::VERTEX,
            range: 0..32,
        },
        PushConstantRange {
            stages: ShaderStages::FRAGMENT,
            range: 32..64,
        },
        PushConstantRange {
            stages: ShaderStages::COMPUTE,
            range: 64..128,
        },
    ];
    assert!(validate_push_constant_ranges(ranges).is_ok());
}

#[test]
fn validate_push_constants_compute_stage() {
    let ranges = &[PushConstantRange {
        stages: ShaderStages::COMPUTE,
        range: 0..128,
    }];
    assert!(validate_push_constant_ranges(ranges).is_ok());
}

#[test]
fn validate_push_constants_all_stages() {
    let ranges = &[PushConstantRange {
        stages: ShaderStages::VERTEX | ShaderStages::FRAGMENT | ShaderStages::COMPUTE,
        range: 0..64,
    }];
    assert!(validate_push_constant_ranges(ranges).is_ok());
}

#[test]
fn total_push_constant_size_empty() {
    assert_eq!(total_push_constant_size(&[]), 0);
}

#[test]
fn total_push_constant_size_single() {
    let ranges = &[PushConstantRange {
        stages: ShaderStages::VERTEX,
        range: 0..64,
    }];
    assert_eq!(total_push_constant_size(ranges), 64);
}

#[test]
fn total_push_constant_size_multiple() {
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

#[test]
fn total_push_constant_size_non_contiguous() {
    let ranges = &[
        PushConstantRange {
            stages: ShaderStages::VERTEX,
            range: 0..32,
        },
        PushConstantRange {
            stages: ShaderStages::FRAGMENT,
            range: 64..128,
        },
    ];
    // Total size is max end, even with gaps
    assert_eq!(total_push_constant_size(ranges), 128);
}

#[test]
fn total_push_constant_size_overlapping() {
    let ranges = &[
        PushConstantRange {
            stages: ShaderStages::VERTEX,
            range: 0..64,
        },
        PushConstantRange {
            stages: ShaderStages::FRAGMENT,
            range: 32..96,
        },
    ];
    assert_eq!(total_push_constant_size(ranges), 96);
}

// =============================================================================
// SECTION 6: TrinityLayoutBuilder (15+ tests)
// =============================================================================

#[test]
fn builder_global_only() {
    let (device, _queue) = match create_test_device_no_push_constants() {
        Some(d) => d,
        None => {
            eprintln!("Skipping: no GPU adapter available");
            return;
        }
    };

    let global_layout = create_uniform_bind_group_layout(&device, "global");
    let cache = PipelineLayoutCache::new();
    let builder = TrinityLayoutBuilder::new(&device, &cache);

    let layout = builder.global_only(&global_layout, 0x1234);
    assert_eq!(cache.len(), 1);

    let labels: Vec<_> = cache.labels().collect();
    assert_eq!(labels[0], Some("trinity_global_only".to_string()));

    // Same layout on second call
    let layout2 = builder.global_only(&global_layout, 0x1234);
    assert!(Arc::ptr_eq(&layout, &layout2));
}

#[test]
fn builder_global_material() {
    let (device, _queue) = match create_test_device_no_push_constants() {
        Some(d) => d,
        None => {
            eprintln!("Skipping: no GPU adapter available");
            return;
        }
    };

    let global = create_uniform_bind_group_layout(&device, "global");
    let material = create_empty_bind_group_layout(&device, "material");

    let cache = PipelineLayoutCache::new();
    let builder = TrinityLayoutBuilder::new(&device, &cache);

    let _layout = builder.global_material(&global, 0x1111, &material, 0x2222);
    assert_eq!(cache.len(), 1);

    let labels: Vec<_> = cache.labels().collect();
    assert_eq!(labels[0], Some("trinity_global_material".to_string()));
}

#[test]
fn builder_pbr() {
    let (device, _queue) = match create_test_device_no_push_constants() {
        Some(d) => d,
        None => {
            eprintln!("Skipping: no GPU adapter available");
            return;
        }
    };

    let global = create_uniform_bind_group_layout(&device, "global");
    let material = create_empty_bind_group_layout(&device, "material");
    let object = create_empty_bind_group_layout(&device, "object");

    let cache = PipelineLayoutCache::new();
    let builder = TrinityLayoutBuilder::new(&device, &cache);

    let layout = builder.pbr(&global, 0x1111, &material, 0x2222, &object, 0x3333);
    assert_eq!(cache.len(), 1);

    let labels: Vec<_> = cache.labels().collect();
    assert_eq!(labels[0], Some("trinity_pbr".to_string()));

    drop(layout);
}

#[test]
fn builder_bindless() {
    let (device, _queue) = match create_test_device_no_push_constants() {
        Some(d) => d,
        None => {
            eprintln!("Skipping: no GPU adapter available");
            return;
        }
    };

    let global = create_uniform_bind_group_layout(&device, "global");
    let material = create_empty_bind_group_layout(&device, "material");
    let object = create_empty_bind_group_layout(&device, "object");
    let bindless = create_empty_bind_group_layout(&device, "bindless");

    let cache = PipelineLayoutCache::new();
    let builder = TrinityLayoutBuilder::new(&device, &cache);

    let _layout =
        builder.bindless(&global, 0x1111, &material, 0x2222, &object, 0x3333, &bindless, 0x4444);
    assert_eq!(cache.len(), 1);

    let labels: Vec<_> = cache.labels().collect();
    assert_eq!(labels[0], Some("trinity_bindless".to_string()));
}

#[test]
fn builder_compute() {
    let (device, _queue) = match create_test_device_no_push_constants() {
        Some(d) => d,
        None => {
            eprintln!("Skipping: no GPU adapter available");
            return;
        }
    };

    let compute_layout = create_storage_bind_group_layout(&device, "compute");

    let cache = PipelineLayoutCache::new();
    let builder = TrinityLayoutBuilder::new(&device, &cache);

    let _layout = builder.compute(Some("my_compute"), &[&compute_layout], &[0xCCCC]);
    assert_eq!(cache.len(), 1);

    let labels: Vec<_> = cache.labels().collect();
    assert_eq!(labels[0], Some("my_compute".to_string()));
}

#[test]
fn builder_compute_no_label() {
    let (device, _queue) = match create_test_device_no_push_constants() {
        Some(d) => d,
        None => {
            eprintln!("Skipping: no GPU adapter available");
            return;
        }
    };

    let compute_layout = create_storage_bind_group_layout(&device, "compute");

    let cache = PipelineLayoutCache::new();
    let builder = TrinityLayoutBuilder::new(&device, &cache);

    let _layout = builder.compute(None, &[&compute_layout], &[0xCCCC]);
    assert_eq!(cache.len(), 1);

    let labels: Vec<_> = cache.labels().collect();
    assert_eq!(labels[0], None);
}

#[test]
fn builder_with_push_constants() {
    let (device, _queue) = match create_test_device() {
        Some(d) => d,
        None => {
            eprintln!("Skipping: no GPU adapter with push constants available");
            return;
        }
    };

    let layout = create_empty_bind_group_layout(&device, "test");
    let cache = PipelineLayoutCache::new();
    let builder = TrinityLayoutBuilder::new(&device, &cache);

    let push_constants = &[PushConstantRange {
        stages: ShaderStages::VERTEX | ShaderStages::FRAGMENT,
        range: 0..64,
    }];

    let _pl = builder.with_push_constants(Some("custom_push"), &[&layout], &[0x1234], push_constants);
    assert_eq!(cache.len(), 1);

    let labels: Vec<_> = cache.labels().collect();
    assert_eq!(labels[0], Some("custom_push".to_string()));
}

#[test]
fn builder_pbr_with_push_constants_nonzero() {
    let (device, _queue) = match create_test_device() {
        Some(d) => d,
        None => {
            eprintln!("Skipping: no GPU adapter with push constants available");
            return;
        }
    };

    let global = create_uniform_bind_group_layout(&device, "global");
    let material = create_empty_bind_group_layout(&device, "material");
    let object = create_empty_bind_group_layout(&device, "object");

    let cache = PipelineLayoutCache::new();
    let builder = TrinityLayoutBuilder::new(&device, &cache);

    let _layout =
        builder.pbr_with_push_constants(&global, 0x1111, &material, 0x2222, &object, 0x3333, 64);

    assert_eq!(cache.len(), 1);

    let labels: Vec<_> = cache.labels().collect();
    assert_eq!(labels[0], Some("trinity_pbr_push".to_string()));
}

#[test]
fn builder_pbr_with_push_constants_zero() {
    let (device, _queue) = match create_test_device_no_push_constants() {
        Some(d) => d,
        None => {
            eprintln!("Skipping: no GPU adapter available");
            return;
        }
    };

    let global = create_uniform_bind_group_layout(&device, "global");
    let material = create_empty_bind_group_layout(&device, "material");
    let object = create_empty_bind_group_layout(&device, "object");

    let cache = PipelineLayoutCache::new();
    let builder = TrinityLayoutBuilder::new(&device, &cache);

    // Zero push constant size should create layout with no push constants
    let _layout =
        builder.pbr_with_push_constants(&global, 0x1111, &material, 0x2222, &object, 0x3333, 0);

    assert_eq!(cache.len(), 1);
}

#[test]
fn builder_caches_across_methods() {
    let (device, _queue) = match create_test_device_no_push_constants() {
        Some(d) => d,
        None => {
            eprintln!("Skipping: no GPU adapter available");
            return;
        }
    };

    let global = create_uniform_bind_group_layout(&device, "global");

    let cache = PipelineLayoutCache::new();
    let builder = TrinityLayoutBuilder::new(&device, &cache);

    // Create global_only
    let layout1 = builder.global_only(&global, 0x1111);

    // Create same via get_or_create (should hit cache)
    let layout2 = cache.get_or_create(
        &device,
        Some("trinity_global_only"),
        &[&global],
        &[0x1111],
        &[],
    );

    assert_eq!(cache.len(), 1);
    assert!(Arc::ptr_eq(&layout1, &layout2));
}

#[test]
fn builder_multiple_layouts_different_configs() {
    let (device, _queue) = match create_test_device_no_push_constants() {
        Some(d) => d,
        None => {
            eprintln!("Skipping: no GPU adapter available");
            return;
        }
    };

    let global = create_uniform_bind_group_layout(&device, "global");
    let material = create_empty_bind_group_layout(&device, "material");
    let object = create_empty_bind_group_layout(&device, "object");
    let bindless = create_empty_bind_group_layout(&device, "bindless");

    let cache = PipelineLayoutCache::new();
    let builder = TrinityLayoutBuilder::new(&device, &cache);

    let _global_only = builder.global_only(&global, 0x1111);
    let _global_material = builder.global_material(&global, 0x1111, &material, 0x2222);
    let _pbr = builder.pbr(&global, 0x1111, &material, 0x2222, &object, 0x3333);
    let _bindless =
        builder.bindless(&global, 0x1111, &material, 0x2222, &object, 0x3333, &bindless, 0x4444);

    assert_eq!(cache.len(), 4);
}

#[test]
fn builder_compute_empty_bind_groups() {
    let (device, _queue) = match create_test_device_no_push_constants() {
        Some(d) => d,
        None => {
            eprintln!("Skipping: no GPU adapter available");
            return;
        }
    };

    let cache = PipelineLayoutCache::new();
    let builder = TrinityLayoutBuilder::new(&device, &cache);

    // Empty bind group array
    let _layout = builder.compute(Some("empty_compute"), &[], &[]);
    assert_eq!(cache.len(), 1);
}

#[test]
fn builder_with_push_constants_multiple_ranges() {
    let (device, _queue) = match create_test_device() {
        Some(d) => d,
        None => {
            eprintln!("Skipping: no GPU adapter with push constants available");
            return;
        }
    };

    let layout = create_empty_bind_group_layout(&device, "test");
    let cache = PipelineLayoutCache::new();
    let builder = TrinityLayoutBuilder::new(&device, &cache);

    let push_constants = &[
        PushConstantRange {
            stages: ShaderStages::VERTEX,
            range: 0..32,
        },
        PushConstantRange {
            stages: ShaderStages::FRAGMENT,
            range: 32..64,
        },
    ];

    let _pl = builder.with_push_constants(Some("multi_push"), &[&layout], &[0x1234], push_constants);
    assert_eq!(cache.len(), 1);
}

// =============================================================================
// SECTION 7: CachedPipelineLayout (10+ tests)
// Note: CachedPipelineLayout metadata is tested via labels() and the cache behavior
// =============================================================================

#[test]
fn cached_layout_labels_reflect_bind_group_count_one() {
    let (device, _queue) = match create_test_device_no_push_constants() {
        Some(d) => d,
        None => {
            eprintln!("Skipping: no GPU adapter available");
            return;
        }
    };

    let layout1 = create_empty_bind_group_layout(&device, "l1");
    let cache = PipelineLayoutCache::new();

    // Create with 1 bind group layout
    let _pl = cache.get_or_create(
        &device,
        Some("one_group"),
        &[&layout1],
        &[0x1111],
        &[],
    );

    let labels: Vec<_> = cache.labels().collect();
    assert_eq!(labels.len(), 1);
    assert_eq!(labels[0], Some("one_group".to_string()));
}

#[test]
fn cached_layout_labels_reflect_bind_group_count_three() {
    let (device, _queue) = match create_test_device_no_push_constants() {
        Some(d) => d,
        None => {
            eprintln!("Skipping: no GPU adapter available");
            return;
        }
    };

    let layout1 = create_empty_bind_group_layout(&device, "l1");
    let layout2 = create_empty_bind_group_layout(&device, "l2");
    let layout3 = create_empty_bind_group_layout(&device, "l3");

    let cache = PipelineLayoutCache::new();

    // Create with 3 bind group layouts
    let _pl = cache.get_or_create(
        &device,
        Some("three_groups"),
        &[&layout1, &layout2, &layout3],
        &[0x1111, 0x2222, 0x3333],
        &[],
    );

    assert_eq!(cache.len(), 1);
    let labels: Vec<_> = cache.labels().collect();
    assert_eq!(labels[0], Some("three_groups".to_string()));
}

#[test]
fn cached_layout_with_push_constants_stored() {
    let (device, _queue) = match create_test_device() {
        Some(d) => d,
        None => {
            eprintln!("Skipping: no GPU adapter with push constants available");
            return;
        }
    };

    let layout = create_empty_bind_group_layout(&device, "test");
    let cache = PipelineLayoutCache::new();

    let push_constants = &[PushConstantRange {
        stages: ShaderStages::VERTEX,
        range: 0..64,
    }];

    let _pl = cache.get_or_create(&device, Some("with_push"), &[&layout], &[0x1234], push_constants);

    // Verify it's cached and can be retrieved again
    assert_eq!(cache.len(), 1);
    assert!(cache.contains(&[0x1234], push_constants));
}

#[test]
fn cached_layout_label_some() {
    let (device, _queue) = match create_test_device_no_push_constants() {
        Some(d) => d,
        None => {
            eprintln!("Skipping: no GPU adapter available");
            return;
        }
    };

    let layout = create_empty_bind_group_layout(&device, "test");
    let cache = PipelineLayoutCache::new();

    let _pl = cache.get_or_create(&device, Some("my_label"), &[&layout], &[0x1234], &[]);

    let labels: Vec<_> = cache.labels().collect();
    assert_eq!(labels.len(), 1);
    assert_eq!(labels[0], Some("my_label".to_string()));
}

#[test]
fn cached_layout_label_none() {
    let (device, _queue) = match create_test_device_no_push_constants() {
        Some(d) => d,
        None => {
            eprintln!("Skipping: no GPU adapter available");
            return;
        }
    };

    let layout = create_empty_bind_group_layout(&device, "test");
    let cache = PipelineLayoutCache::new();

    let _pl = cache.get_or_create(&device, None, &[&layout], &[0x1234], &[]);

    let labels: Vec<_> = cache.labels().collect();
    assert_eq!(labels.len(), 1);
    assert_eq!(labels[0], None);
}

#[test]
fn cached_layout_arc_sharing() {
    let (device, _queue) = match create_test_device_no_push_constants() {
        Some(d) => d,
        None => {
            eprintln!("Skipping: no GPU adapter available");
            return;
        }
    };

    let layout = create_empty_bind_group_layout(&device, "test");
    let cache = PipelineLayoutCache::new();

    let pl1 = cache.get_or_create(&device, Some("test"), &[&layout], &[0x1234], &[]);
    let pl2 = cache.get_or_create(&device, Some("test"), &[&layout], &[0x1234], &[]);

    // Both should be the same Arc
    assert!(Arc::ptr_eq(&pl1, &pl2));
}

#[test]
fn cached_layout_inner_reference() {
    let (device, _queue) = match create_test_device_no_push_constants() {
        Some(d) => d,
        None => {
            eprintln!("Skipping: no GPU adapter available");
            return;
        }
    };

    let layout = create_empty_bind_group_layout(&device, "test");
    let cache = PipelineLayoutCache::new();

    let pl = cache.get_or_create(&device, Some("test"), &[&layout], &[0x1234], &[]);

    // The Arc should be valid and usable
    let _inner_ref: &wgpu::PipelineLayout = &*pl;
    assert!(Arc::strong_count(&pl) >= 1);
}

#[test]
fn cached_layout_push_constants_with_sparse_ranges() {
    let (device, _queue) = match create_test_device() {
        Some(d) => d,
        None => {
            eprintln!("Skipping: no GPU adapter with push constants available");
            return;
        }
    };

    let layout = create_empty_bind_group_layout(&device, "test");
    let cache = PipelineLayoutCache::new();

    // Non-contiguous ranges
    let push_constants = &[
        PushConstantRange {
            stages: ShaderStages::VERTEX,
            range: 0..32,
        },
        PushConstantRange {
            stages: ShaderStages::FRAGMENT,
            range: 64..128,
        },
    ];

    let _pl = cache.get_or_create(&device, Some("sparse"), &[&layout], &[0x1234], push_constants);

    assert_eq!(cache.len(), 1);
    assert!(cache.contains(&[0x1234], push_constants));
}

#[test]
fn cached_layout_zero_bind_groups() {
    let (device, _queue) = match create_test_device_no_push_constants() {
        Some(d) => d,
        None => {
            eprintln!("Skipping: no GPU adapter available");
            return;
        }
    };

    let cache = PipelineLayoutCache::new();

    let _pl = cache.get_or_create(&device, Some("empty"), &[], &[], &[]);

    assert_eq!(cache.len(), 1);
    assert!(cache.contains(&[], &[]));
}

#[test]
fn cached_layout_zero_push_constants() {
    let (device, _queue) = match create_test_device_no_push_constants() {
        Some(d) => d,
        None => {
            eprintln!("Skipping: no GPU adapter available");
            return;
        }
    };

    let layout = create_empty_bind_group_layout(&device, "test");
    let cache = PipelineLayoutCache::new();

    let _pl = cache.get_or_create(&device, Some("no_push"), &[&layout], &[0x1234], &[]);

    assert_eq!(cache.len(), 1);
    assert!(cache.contains(&[0x1234], &[]));
    // Verify no push constants version is different from with push constants
    assert!(!cache.contains(&[0x1234], &[PushConstantRange {
        stages: ShaderStages::VERTEX,
        range: 0..64,
    }]));
}

#[test]
fn cached_layout_multiple_labels() {
    let (device, _queue) = match create_test_device_no_push_constants() {
        Some(d) => d,
        None => {
            eprintln!("Skipping: no GPU adapter available");
            return;
        }
    };

    let layout = create_empty_bind_group_layout(&device, "test");
    let cache = PipelineLayoutCache::new();

    let _pl1 = cache.get_or_create(&device, Some("first"), &[&layout], &[0x1111], &[]);
    let _pl2 = cache.get_or_create(&device, Some("second"), &[&layout], &[0x2222], &[]);
    let _pl3 = cache.get_or_create(&device, None, &[&layout], &[0x3333], &[]);
    let _pl4 = cache.get_or_create(&device, Some("fourth"), &[&layout], &[0x4444], &[]);

    let labels: Vec<_> = cache.labels().collect();
    assert_eq!(labels.len(), 4);
    assert!(labels.contains(&Some("first".to_string())));
    assert!(labels.contains(&Some("second".to_string())));
    assert!(labels.contains(&None));
    assert!(labels.contains(&Some("fourth".to_string())));
}

// =============================================================================
// Additional Edge Cases and Stress Tests
// =============================================================================

#[test]
fn key_with_many_bind_groups() {
    let hashes: Vec<u64> = (0..16).map(|i| 0x1000 + i as u64).collect();
    let key = PipelineLayoutKey::new(&hashes, &[]);
    assert_ne!(key.bind_group_layouts_hash(), 0);
}

#[test]
fn validate_push_constants_exactly_aligned() {
    // Every multiple of 4 should be valid for alignment
    for size in (4..=128).step_by(4) {
        let ranges = &[PushConstantRange {
            stages: ShaderStages::VERTEX,
            range: 0..size,
        }];
        assert!(
            validate_push_constant_ranges(ranges).is_ok(),
            "Size {} should be valid",
            size
        );
    }
}

#[test]
fn cache_stress_many_unique_layouts() {
    let (device, _queue) = match create_test_device_no_push_constants() {
        Some(d) => d,
        None => {
            eprintln!("Skipping: no GPU adapter available");
            return;
        }
    };

    let layout = create_empty_bind_group_layout(&device, "test");
    let cache = PipelineLayoutCache::new();

    // Create 100 unique layouts
    for i in 0..100u64 {
        let _pl = cache.get_or_create(&device, Some(&format!("layout_{}", i)), &[&layout], &[i], &[]);
    }

    assert_eq!(cache.len(), 100);
    assert_eq!(cache.metrics().misses, 100);
    assert_eq!(cache.metrics().hits, 0);
}

#[test]
fn cache_mixed_hits_and_misses() {
    let (device, _queue) = match create_test_device_no_push_constants() {
        Some(d) => d,
        None => {
            eprintln!("Skipping: no GPU adapter available");
            return;
        }
    };

    let layout = create_empty_bind_group_layout(&device, "test");
    let cache = PipelineLayoutCache::new();

    // Pattern: create 5 unique layouts, then hit each 4 times
    for i in 0..5u64 {
        for _ in 0..5 {
            let _pl = cache.get_or_create(&device, Some(&format!("layout_{}", i)), &[&layout], &[i], &[]);
        }
    }

    assert_eq!(cache.len(), 5);
    assert_eq!(cache.metrics().misses, 5);
    assert_eq!(cache.metrics().hits, 20);
    assert!((cache.metrics().hit_rate - 0.8).abs() < 0.001);
}

#[test]
fn builder_reuse_across_different_builds() {
    let (device, _queue) = match create_test_device_no_push_constants() {
        Some(d) => d,
        None => {
            eprintln!("Skipping: no GPU adapter available");
            return;
        }
    };

    let global1 = create_uniform_bind_group_layout(&device, "global1");
    let global2 = create_uniform_bind_group_layout(&device, "global2");

    let cache = PipelineLayoutCache::new();
    let builder = TrinityLayoutBuilder::new(&device, &cache);

    // Same hash but different actual layouts - still cached by hash
    let layout1 = builder.global_only(&global1, 0x1234);
    let layout2 = builder.global_only(&global2, 0x1234); // Same hash!

    // These should be the same (cached by hash, not by layout identity)
    assert!(Arc::ptr_eq(&layout1, &layout2));
}

#[test]
fn validate_push_constants_boundary_cases() {
    // Test at exactly MAX_PUSH_CONSTANT_SIZE
    let ranges_at_max = &[PushConstantRange {
        stages: ShaderStages::VERTEX,
        range: 0..MAX_PUSH_CONSTANT_SIZE,
    }];
    assert!(validate_push_constant_ranges(ranges_at_max).is_ok());

    // Test just over MAX_PUSH_CONSTANT_SIZE
    let ranges_over_max = &[PushConstantRange {
        stages: ShaderStages::VERTEX,
        range: 0..(MAX_PUSH_CONSTANT_SIZE + 4),
    }];
    assert!(validate_push_constant_ranges(ranges_over_max).is_err());
}

#[test]
fn key_symmetry_with_empty_vs_nonempty() {
    let key_empty_both = PipelineLayoutKey::new(&[], &[]);
    let key_empty_ranges = PipelineLayoutKey::new(&[0x1234], &[]);
    let key_empty_groups = PipelineLayoutKey::new(&[], &[PushConstantRange {
        stages: ShaderStages::VERTEX,
        range: 0..64,
    }]);

    assert_ne!(key_empty_both, key_empty_ranges);
    assert_ne!(key_empty_both, key_empty_groups);
    assert_ne!(key_empty_ranges, key_empty_groups);
}

#[test]
fn cache_metrics_after_clear_and_repopulate() {
    let (device, _queue) = match create_test_device_no_push_constants() {
        Some(d) => d,
        None => {
            eprintln!("Skipping: no GPU adapter available");
            return;
        }
    };

    let layout = create_empty_bind_group_layout(&device, "test");
    let cache = PipelineLayoutCache::new();

    // First population
    for i in 0..3u64 {
        let _pl = cache.get_or_create(&device, None, &[&layout], &[i], &[]);
    }
    assert_eq!(cache.len(), 3);
    assert_eq!(cache.metrics().misses, 3);

    // Clear
    cache.clear();
    assert!(cache.is_empty());
    assert_eq!(cache.metrics().misses, 0);

    // Repopulate
    for i in 0..5u64 {
        let _pl = cache.get_or_create(&device, None, &[&layout], &[i], &[]);
    }
    assert_eq!(cache.len(), 5);
    assert_eq!(cache.metrics().misses, 5);
}
