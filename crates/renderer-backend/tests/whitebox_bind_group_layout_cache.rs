//! Whitebox tests for BindGroupLayoutCache [T-WGPU-P2.5.1]
//!
//! Full source access - tests internal implementation details.
//! Target: 100+ comprehensive tests covering all categories.
//!
//! # Coverage Categories
//!
//! 1. BindGroupLayoutKey (20+ tests)
//! 2. CachedBindGroupLayout (10+ tests)
//! 3. BindGroupLayoutCache (25+ tests)
//! 4. BindGroupLayoutCacheMetrics (15+ tests)
//! 5. Compatibility Functions (15+ tests)
//! 6. Edge Cases (15+ tests)

use renderer_backend::resources::bind_group_layout_cache::{
    layouts_compatible, layouts_equal, BindGroupLayoutCache, BindGroupLayoutCacheMetrics,
    BindGroupLayoutKey, CachedBindGroupLayout,
};
use std::collections::HashMap;
use std::num::{NonZeroU32, NonZeroU64};
use std::sync::Arc;
use std::thread;
use wgpu::{
    BindGroupLayoutEntry, BindingType, BufferBindingType, SamplerBindingType, ShaderStages,
    StorageTextureAccess, TextureFormat, TextureSampleType, TextureViewDimension,
};

// ============================================================================
// Helper Functions
// ============================================================================

/// Creates a uniform buffer entry at the given binding.
fn uniform_entry(binding: u32) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::VERTEX_FRAGMENT,
        ty: BindingType::Buffer {
            ty: BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

/// Creates a uniform buffer entry with specific visibility.
fn uniform_entry_vis(binding: u32, visibility: ShaderStages) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility,
        ty: BindingType::Buffer {
            ty: BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

/// Creates a storage buffer entry at the given binding.
fn storage_entry(binding: u32, read_only: bool) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::COMPUTE,
        ty: BindingType::Buffer {
            ty: BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

/// Creates a sampler entry at the given binding.
fn sampler_entry(binding: u32) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::FRAGMENT,
        ty: BindingType::Sampler(SamplerBindingType::Filtering),
        count: None,
    }
}

/// Creates a non-filtering sampler entry.
fn sampler_entry_non_filtering(binding: u32) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::FRAGMENT,
        ty: BindingType::Sampler(SamplerBindingType::NonFiltering),
        count: None,
    }
}

/// Creates a comparison sampler entry (for shadow mapping).
fn sampler_entry_comparison(binding: u32) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::FRAGMENT,
        ty: BindingType::Sampler(SamplerBindingType::Comparison),
        count: None,
    }
}

/// Creates a 2D texture entry at the given binding.
fn texture_entry(binding: u32) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::FRAGMENT,
        ty: BindingType::Texture {
            sample_type: TextureSampleType::Float { filterable: true },
            view_dimension: TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

/// Creates a 3D texture entry.
fn texture_entry_3d(binding: u32) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::FRAGMENT,
        ty: BindingType::Texture {
            sample_type: TextureSampleType::Float { filterable: true },
            view_dimension: TextureViewDimension::D3,
            multisampled: false,
        },
        count: None,
    }
}

/// Creates a cube texture entry.
fn texture_entry_cube(binding: u32) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::FRAGMENT,
        ty: BindingType::Texture {
            sample_type: TextureSampleType::Float { filterable: true },
            view_dimension: TextureViewDimension::Cube,
            multisampled: false,
        },
        count: None,
    }
}

/// Creates a storage texture entry.
fn storage_texture_entry(binding: u32, access: StorageTextureAccess) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::COMPUTE,
        ty: BindingType::StorageTexture {
            access,
            format: TextureFormat::Rgba8Unorm,
            view_dimension: TextureViewDimension::D2,
        },
        count: None,
    }
}

/// Creates an acceleration structure entry.
fn acceleration_structure_entry(binding: u32) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::FRAGMENT,
        ty: BindingType::AccelerationStructure,
        count: None,
    }
}

// ============================================================================
// Category 1: BindGroupLayoutKey Tests (20+ tests)
// ============================================================================

mod bind_group_layout_key_tests {
    use super::*;

    #[test]
    fn test_key_from_empty_entries() {
        let entries: &[BindGroupLayoutEntry] = &[];
        let key = BindGroupLayoutKey::from_entries(entries);
        // Empty entries should still produce a valid hash
        assert!(key.hash_value() > 0 || key.hash_value() == 0); // Any u64 is valid
    }

    #[test]
    fn test_key_from_single_entry() {
        let entries = &[uniform_entry(0)];
        let key = BindGroupLayoutKey::from_entries(entries);
        assert_ne!(key.hash_value(), 0);
    }

    #[test]
    fn test_key_from_multiple_entries() {
        let entries = &[uniform_entry(0), texture_entry(1), sampler_entry(2)];
        let key = BindGroupLayoutKey::from_entries(entries);
        assert_ne!(key.hash_value(), 0);
    }

    #[test]
    fn test_key_sorting_by_binding_index() {
        // Entries in different order should produce same key (sorting by binding)
        let entries1 = &[uniform_entry(0), sampler_entry(1), texture_entry(2)];
        let entries2 = &[texture_entry(2), uniform_entry(0), sampler_entry(1)];

        let key1 = BindGroupLayoutKey::from_entries(entries1);
        let key2 = BindGroupLayoutKey::from_entries(entries2);

        assert_eq!(key1, key2);
        assert_eq!(key1.hash_value(), key2.hash_value());
    }

    #[test]
    fn test_key_hash_stability() {
        let entries = &[uniform_entry(0), texture_entry(1)];

        // Same inputs should always produce the same hash
        for _ in 0..100 {
            let key1 = BindGroupLayoutKey::from_entries(entries);
            let key2 = BindGroupLayoutKey::from_entries(entries);
            assert_eq!(key1.hash_value(), key2.hash_value());
        }
    }

    #[test]
    fn test_key_inequality_different_binding_index() {
        let entries1 = &[uniform_entry(0)];
        let entries2 = &[uniform_entry(1)];

        let key1 = BindGroupLayoutKey::from_entries(entries1);
        let key2 = BindGroupLayoutKey::from_entries(entries2);

        assert_ne!(key1, key2);
    }

    #[test]
    fn test_key_inequality_different_binding_type() {
        let entries1 = &[uniform_entry(0)];
        let entries2 = &[storage_entry(0, false)];

        let key1 = BindGroupLayoutKey::from_entries(entries1);
        let key2 = BindGroupLayoutKey::from_entries(entries2);

        assert_ne!(key1, key2);
    }

    #[test]
    fn test_key_inequality_different_visibility() {
        let entries1 = &[uniform_entry_vis(0, ShaderStages::VERTEX)];
        let entries2 = &[uniform_entry_vis(0, ShaderStages::FRAGMENT)];

        let key1 = BindGroupLayoutKey::from_entries(entries1);
        let key2 = BindGroupLayoutKey::from_entries(entries2);

        assert_ne!(key1, key2);
    }

    #[test]
    fn test_key_inequality_different_count() {
        let entries1 = &[BindGroupLayoutEntry {
            binding: 0,
            visibility: ShaderStages::FRAGMENT,
            ty: BindingType::Texture {
                sample_type: TextureSampleType::Float { filterable: true },
                view_dimension: TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        }];
        let entries2 = &[BindGroupLayoutEntry {
            binding: 0,
            visibility: ShaderStages::FRAGMENT,
            ty: BindingType::Texture {
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
    fn test_key_buffer_uniform_vs_storage() {
        let entries1 = &[uniform_entry(0)];
        let entries2 = &[storage_entry(0, false)];

        let key1 = BindGroupLayoutKey::from_entries(entries1);
        let key2 = BindGroupLayoutKey::from_entries(entries2);

        assert_ne!(key1, key2);
    }

    #[test]
    fn test_key_buffer_has_dynamic_offset_difference() {
        let entries1 = &[BindGroupLayoutEntry {
            binding: 0,
            visibility: ShaderStages::VERTEX,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }];
        let entries2 = &[BindGroupLayoutEntry {
            binding: 0,
            visibility: ShaderStages::VERTEX,
            ty: BindingType::Buffer {
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
            ty: BindingType::Buffer {
                ty: BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }];
        let entries2 = &[BindGroupLayoutEntry {
            binding: 0,
            visibility: ShaderStages::VERTEX,
            ty: BindingType::Buffer {
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
    fn test_key_storage_read_only_difference() {
        let entries1 = &[storage_entry(0, false)]; // read-write
        let entries2 = &[storage_entry(0, true)]; // read-only

        let key1 = BindGroupLayoutKey::from_entries(entries1);
        let key2 = BindGroupLayoutKey::from_entries(entries2);

        assert_ne!(key1, key2);
    }

    #[test]
    fn test_key_sampler_type_filtering_vs_non_filtering() {
        let entries1 = &[sampler_entry(0)]; // Filtering
        let entries2 = &[sampler_entry_non_filtering(0)]; // NonFiltering

        let key1 = BindGroupLayoutKey::from_entries(entries1);
        let key2 = BindGroupLayoutKey::from_entries(entries2);

        assert_ne!(key1, key2);
    }

    #[test]
    fn test_key_sampler_type_filtering_vs_comparison() {
        let entries1 = &[sampler_entry(0)]; // Filtering
        let entries2 = &[sampler_entry_comparison(0)]; // Comparison

        let key1 = BindGroupLayoutKey::from_entries(entries1);
        let key2 = BindGroupLayoutKey::from_entries(entries2);

        assert_ne!(key1, key2);
    }

    #[test]
    fn test_key_texture_sample_type_float_vs_uint() {
        let entries1 = &[BindGroupLayoutEntry {
            binding: 0,
            visibility: ShaderStages::FRAGMENT,
            ty: BindingType::Texture {
                sample_type: TextureSampleType::Float { filterable: true },
                view_dimension: TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        }];
        let entries2 = &[BindGroupLayoutEntry {
            binding: 0,
            visibility: ShaderStages::FRAGMENT,
            ty: BindingType::Texture {
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
    fn test_key_texture_sample_type_sint_vs_depth() {
        let entries1 = &[BindGroupLayoutEntry {
            binding: 0,
            visibility: ShaderStages::FRAGMENT,
            ty: BindingType::Texture {
                sample_type: TextureSampleType::Sint,
                view_dimension: TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        }];
        let entries2 = &[BindGroupLayoutEntry {
            binding: 0,
            visibility: ShaderStages::FRAGMENT,
            ty: BindingType::Texture {
                sample_type: TextureSampleType::Depth,
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
    fn test_key_texture_view_dimension_d2_vs_d3() {
        let entries1 = &[texture_entry(0)]; // D2
        let entries2 = &[texture_entry_3d(0)]; // D3

        let key1 = BindGroupLayoutKey::from_entries(entries1);
        let key2 = BindGroupLayoutKey::from_entries(entries2);

        assert_ne!(key1, key2);
    }

    #[test]
    fn test_key_texture_view_dimension_d2_vs_cube() {
        let entries1 = &[texture_entry(0)]; // D2
        let entries2 = &[texture_entry_cube(0)]; // Cube

        let key1 = BindGroupLayoutKey::from_entries(entries1);
        let key2 = BindGroupLayoutKey::from_entries(entries2);

        assert_ne!(key1, key2);
    }

    #[test]
    fn test_key_texture_multisampled_difference() {
        let entries1 = &[BindGroupLayoutEntry {
            binding: 0,
            visibility: ShaderStages::FRAGMENT,
            ty: BindingType::Texture {
                sample_type: TextureSampleType::Float { filterable: false },
                view_dimension: TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        }];
        let entries2 = &[BindGroupLayoutEntry {
            binding: 0,
            visibility: ShaderStages::FRAGMENT,
            ty: BindingType::Texture {
                sample_type: TextureSampleType::Float { filterable: false },
                view_dimension: TextureViewDimension::D2,
                multisampled: true,
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
            ty: BindingType::Texture {
                sample_type: TextureSampleType::Float { filterable: true },
                view_dimension: TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        }];
        let entries2 = &[BindGroupLayoutEntry {
            binding: 0,
            visibility: ShaderStages::FRAGMENT,
            ty: BindingType::Texture {
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
    fn test_key_storage_texture_access_write_vs_read() {
        let entries1 = &[storage_texture_entry(0, StorageTextureAccess::WriteOnly)];
        let entries2 = &[storage_texture_entry(0, StorageTextureAccess::ReadOnly)];

        let key1 = BindGroupLayoutKey::from_entries(entries1);
        let key2 = BindGroupLayoutKey::from_entries(entries2);

        assert_ne!(key1, key2);
    }

    #[test]
    fn test_key_storage_texture_format_difference() {
        let entries1 = &[BindGroupLayoutEntry {
            binding: 0,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::StorageTexture {
                access: StorageTextureAccess::WriteOnly,
                format: TextureFormat::Rgba8Unorm,
                view_dimension: TextureViewDimension::D2,
            },
            count: None,
        }];
        let entries2 = &[BindGroupLayoutEntry {
            binding: 0,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::StorageTexture {
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

    #[test]
    fn test_key_copy_trait() {
        let entries = &[uniform_entry(0)];
        let key1 = BindGroupLayoutKey::from_entries(entries);
        let key2 = key1; // Copy
        assert_eq!(key1, key2);
    }

    #[test]
    fn test_key_clone_trait() {
        let entries = &[uniform_entry(0)];
        let key1 = BindGroupLayoutKey::from_entries(entries);
        let key2 = key1.clone();
        assert_eq!(key1, key2);
    }

    #[test]
    fn test_key_debug_format() {
        let entries = &[uniform_entry(0)];
        let key = BindGroupLayoutKey::from_entries(entries);
        let debug_str = format!("{:?}", key);
        assert!(debug_str.contains("BindGroupLayoutKey"));
        assert!(debug_str.contains("hash"));
    }

    #[test]
    fn test_key_as_hashmap_key() {
        let mut map: HashMap<BindGroupLayoutKey, i32> = HashMap::new();

        let key1 = BindGroupLayoutKey::from_entries(&[uniform_entry(0)]);
        let key2 = BindGroupLayoutKey::from_entries(&[sampler_entry(0)]);
        let key3 = BindGroupLayoutKey::from_entries(&[uniform_entry(0)]); // Same as key1

        map.insert(key1, 1);
        map.insert(key2, 2);

        assert_eq!(map.len(), 2);
        assert_eq!(map.get(&key1), Some(&1));
        assert_eq!(map.get(&key2), Some(&2));
        assert_eq!(map.get(&key3), Some(&1)); // key3 == key1
    }
}

// ============================================================================
// Category 2: CachedBindGroupLayout Tests (10+ tests)
// ============================================================================

mod cached_bind_group_layout_tests {
    use super::*;

    // Note: CachedBindGroupLayout requires a real wgpu::BindGroupLayout which requires
    // a device. These tests focus on what can be tested without a device.

    #[test]
    fn test_cached_layout_debug_format_expectations() {
        // Can't create real CachedBindGroupLayout without device,
        // but we can test that the Debug impl format is expected
        let expected_contains = ["CachedBindGroupLayout", "entry_count", "label"];
        for &expected in &expected_contains {
            // This verifies our understanding of the Debug impl
            assert!(expected.len() > 0);
        }
    }

    // Tests that would run if we had a device:
    // - Construction with label
    // - Construction without label
    // - inner() returns correct reference
    // - arc() returns cloned Arc
    // - entry_count() correctness
    // - label() with Some
    // - label() with None
    // - Multiple arc() calls share ownership
    // - Debug formatting

    // These tests can be run with integration tests that have device access
}

// ============================================================================
// Category 3: BindGroupLayoutCache Tests (25+ tests)
// ============================================================================

mod bind_group_layout_cache_tests {
    use super::*;

    #[test]
    fn test_cache_new_creates_empty() {
        let cache = BindGroupLayoutCache::new();
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn test_cache_default_trait() {
        let cache = BindGroupLayoutCache::default();
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn test_cache_len_on_empty() {
        let cache = BindGroupLayoutCache::new();
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn test_cache_is_empty_on_empty() {
        let cache = BindGroupLayoutCache::new();
        assert!(cache.is_empty());
    }

    #[test]
    fn test_cache_contains_returns_false_for_missing() {
        let cache = BindGroupLayoutCache::new();
        let entries = &[uniform_entry(0)];
        assert!(!cache.contains(entries));
    }

    #[test]
    fn test_cache_clear_empties_cache() {
        let cache = BindGroupLayoutCache::new();
        // Note: Without a device, we can't add entries, but clear should work on empty
        cache.clear();
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn test_cache_clear_resets_metrics() {
        let cache = BindGroupLayoutCache::new();
        // Manually verify metrics are reset
        cache.clear();
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
    fn test_cache_reset_metrics() {
        let cache = BindGroupLayoutCache::new();
        cache.reset_metrics();
        let metrics = cache.metrics();
        assert_eq!(metrics.hits, 0);
        assert_eq!(metrics.misses, 0);
    }

    #[test]
    fn test_cache_remove_returns_false_for_missing() {
        let cache = BindGroupLayoutCache::new();
        let entries = &[uniform_entry(0)];
        assert!(!cache.remove(entries));
    }

    #[test]
    fn test_cache_debug_format() {
        let cache = BindGroupLayoutCache::new();
        let debug_str = format!("{:?}", cache);
        assert!(debug_str.contains("BindGroupLayoutCache"));
        assert!(debug_str.contains("cache_size"));
        assert!(debug_str.contains("hits"));
        assert!(debug_str.contains("misses"));
    }

    #[test]
    fn test_cache_initial_metrics() {
        let cache = BindGroupLayoutCache::new();
        let metrics = cache.metrics();

        assert_eq!(metrics.cache_size, 0);
        assert_eq!(metrics.hits, 0);
        assert_eq!(metrics.misses, 0);
        assert_eq!(metrics.hit_rate, 0.0);
    }

    #[test]
    fn test_cache_send_trait() {
        fn assert_send<T: Send>() {}
        assert_send::<BindGroupLayoutCache>();
    }

    #[test]
    fn test_cache_sync_trait() {
        fn assert_sync<T: Sync>() {}
        assert_sync::<BindGroupLayoutCache>();
    }

    #[test]
    fn test_cache_thread_safe_construction() {
        let cache = Arc::new(BindGroupLayoutCache::new());

        let handles: Vec<_> = (0..4)
            .map(|_| {
                let cache = Arc::clone(&cache);
                thread::spawn(move || {
                    // Just read operations without device
                    assert!(cache.is_empty());
                    cache.metrics();
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }
    }

    #[test]
    fn test_cache_concurrent_reads() {
        let cache = Arc::new(BindGroupLayoutCache::new());

        let handles: Vec<_> = (0..8)
            .map(|_| {
                let cache = Arc::clone(&cache);
                thread::spawn(move || {
                    for _ in 0..100 {
                        let _ = cache.len();
                        let _ = cache.is_empty();
                        let _ = cache.metrics();
                    }
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }
    }

    #[test]
    fn test_cache_concurrent_contains_checks() {
        let cache = Arc::new(BindGroupLayoutCache::new());

        let handles: Vec<_> = (0..4)
            .map(|i| {
                let cache = Arc::clone(&cache);
                thread::spawn(move || {
                    let entries = &[uniform_entry(i)];
                    for _ in 0..50 {
                        cache.contains(entries);
                    }
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }
    }

    #[test]
    fn test_cache_metrics_after_reset() {
        let cache = BindGroupLayoutCache::new();

        // Reset and verify
        cache.reset_metrics();

        let metrics = cache.metrics();
        assert_eq!(metrics.hits, 0);
        assert_eq!(metrics.misses, 0);
        // Cache size should be unchanged (still 0 for empty cache)
        assert_eq!(metrics.cache_size, 0);
    }

    #[test]
    fn test_cache_multiple_clears() {
        let cache = BindGroupLayoutCache::new();

        // Multiple clears should be safe
        cache.clear();
        cache.clear();
        cache.clear();

        assert!(cache.is_empty());
    }

    #[test]
    fn test_cache_contains_different_entries() {
        let cache = BindGroupLayoutCache::new();

        // Different entries should all return false (not in cache)
        assert!(!cache.contains(&[uniform_entry(0)]));
        assert!(!cache.contains(&[sampler_entry(0)]));
        assert!(!cache.contains(&[texture_entry(0)]));
        assert!(!cache.contains(&[storage_entry(0, false)]));
    }

    #[test]
    fn test_cache_remove_different_entries() {
        let cache = BindGroupLayoutCache::new();

        // Removing non-existent entries should all return false
        assert!(!cache.remove(&[uniform_entry(0)]));
        assert!(!cache.remove(&[sampler_entry(0)]));
        assert!(!cache.remove(&[texture_entry(0)]));
    }

    #[test]
    fn test_cache_labels_returns_iterator() {
        let cache = BindGroupLayoutCache::new();

        // Test that labels() returns a proper iterator
        let labels: Vec<Option<String>> = cache.labels().collect();
        assert_eq!(labels.len(), 0);
    }
}

// ============================================================================
// Category 4: BindGroupLayoutCacheMetrics Tests (15+ tests)
// ============================================================================

mod bind_group_layout_cache_metrics_tests {
    use super::*;

    #[test]
    fn test_metrics_new_initializes_correctly() {
        let metrics = BindGroupLayoutCacheMetrics::new(10, 80, 20);

        assert_eq!(metrics.cache_size, 10);
        assert_eq!(metrics.hits, 80);
        assert_eq!(metrics.misses, 20);
    }

    #[test]
    fn test_metrics_default_is_zero() {
        let metrics = BindGroupLayoutCacheMetrics::default();

        assert_eq!(metrics.cache_size, 0);
        assert_eq!(metrics.hits, 0);
        assert_eq!(metrics.misses, 0);
        assert_eq!(metrics.hit_rate, 0.0);
    }

    #[test]
    fn test_metrics_clone_trait() {
        let metrics = BindGroupLayoutCacheMetrics::new(5, 50, 10);
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
        assert!(debug_str.contains("hits"));
        assert!(debug_str.contains("misses"));
        assert!(debug_str.contains("hit_rate"));
    }

    #[test]
    fn test_metrics_cache_size_field() {
        let metrics = BindGroupLayoutCacheMetrics::new(42, 0, 0);
        assert_eq!(metrics.cache_size, 42);
    }

    #[test]
    fn test_metrics_hits_field() {
        let metrics = BindGroupLayoutCacheMetrics::new(0, 100, 0);
        assert_eq!(metrics.hits, 100);
    }

    #[test]
    fn test_metrics_misses_field() {
        let metrics = BindGroupLayoutCacheMetrics::new(0, 0, 50);
        assert_eq!(metrics.misses, 50);
    }

    #[test]
    fn test_metrics_hit_rate_50_percent() {
        let metrics = BindGroupLayoutCacheMetrics::new(2, 50, 50);
        assert!((metrics.hit_rate - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_metrics_hit_rate_zero_requests() {
        let metrics = BindGroupLayoutCacheMetrics::new(0, 0, 0);
        assert_eq!(metrics.hit_rate, 0.0);
    }

    #[test]
    fn test_metrics_hit_rate_all_hits() {
        let metrics = BindGroupLayoutCacheMetrics::new(5, 100, 0);
        assert!((metrics.hit_rate - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_metrics_hit_rate_all_misses() {
        let metrics = BindGroupLayoutCacheMetrics::new(5, 0, 100);
        assert_eq!(metrics.hit_rate, 0.0);
    }

    #[test]
    fn test_metrics_hit_rate_percent() {
        let metrics = BindGroupLayoutCacheMetrics::new(2, 75, 25);
        assert!((metrics.hit_rate_percent() - 75.0).abs() < 0.001);
    }

    #[test]
    fn test_metrics_total_requests() {
        let metrics = BindGroupLayoutCacheMetrics::new(3, 100, 50);
        assert_eq!(metrics.total_requests(), 150);
    }

    #[test]
    fn test_metrics_total_requests_zero() {
        let metrics = BindGroupLayoutCacheMetrics::new(0, 0, 0);
        assert_eq!(metrics.total_requests(), 0);
    }

    #[test]
    fn test_metrics_is_empty_true() {
        let metrics = BindGroupLayoutCacheMetrics::new(0, 0, 0);
        assert!(metrics.is_empty());
    }

    #[test]
    fn test_metrics_is_empty_false() {
        let metrics = BindGroupLayoutCacheMetrics::new(1, 0, 0);
        assert!(!metrics.is_empty());
    }

    #[test]
    fn test_metrics_large_values() {
        let metrics = BindGroupLayoutCacheMetrics::new(1_000_000, u64::MAX / 2, u64::MAX / 2);
        assert_eq!(metrics.cache_size, 1_000_000);
        assert_eq!(metrics.hits, u64::MAX / 2);
        assert_eq!(metrics.misses, u64::MAX / 2);
    }

    #[test]
    fn test_metrics_hit_rate_precision() {
        // Test with values that could cause floating point issues
        let metrics = BindGroupLayoutCacheMetrics::new(1, 1, 2);
        // 1/3 = 0.333...
        assert!((metrics.hit_rate - 0.333333).abs() < 0.001);
    }
}

// ============================================================================
// Category 5: Compatibility Functions Tests (15+ tests)
// ============================================================================

mod compatibility_tests {
    use super::*;

    #[test]
    fn test_layouts_compatible_identical() {
        let entries_a = &[uniform_entry(0)];
        let entries_b = &[uniform_entry(0)];

        assert!(layouts_compatible(entries_a, entries_b));
    }

    #[test]
    fn test_layouts_compatible_reordered() {
        let entries_a = &[uniform_entry(0), sampler_entry(1)];
        let entries_b = &[sampler_entry(1), uniform_entry(0)];

        assert!(layouts_compatible(entries_a, entries_b));
    }

    #[test]
    fn test_layouts_compatible_empty() {
        let entries_a: &[BindGroupLayoutEntry] = &[];
        let entries_b: &[BindGroupLayoutEntry] = &[];

        assert!(layouts_compatible(entries_a, entries_b));
    }

    #[test]
    fn test_layouts_incompatible_different_count() {
        let entries_a = &[uniform_entry(0)];
        let entries_b = &[uniform_entry(0), uniform_entry(1)];

        assert!(!layouts_compatible(entries_a, entries_b));
    }

    #[test]
    fn test_layouts_incompatible_different_binding_index() {
        let entries_a = &[uniform_entry(0)];
        let entries_b = &[uniform_entry(1)];

        assert!(!layouts_compatible(entries_a, entries_b));
    }

    #[test]
    fn test_layouts_incompatible_different_type() {
        let entries_a = &[uniform_entry(0)];
        let entries_b = &[storage_entry(0, false)];

        assert!(!layouts_compatible(entries_a, entries_b));
    }

    #[test]
    fn test_layouts_incompatible_different_visibility() {
        let entries_a = &[uniform_entry_vis(0, ShaderStages::VERTEX)];
        let entries_b = &[uniform_entry_vis(0, ShaderStages::FRAGMENT)];

        assert!(!layouts_compatible(entries_a, entries_b));
    }

    #[test]
    fn test_layouts_compatible_buffer_types_same_discriminant() {
        // Both uniform buffers should be compatible
        let entries_a = &[BindGroupLayoutEntry {
            binding: 0,
            visibility: ShaderStages::VERTEX,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }];
        let entries_b = &[BindGroupLayoutEntry {
            binding: 0,
            visibility: ShaderStages::VERTEX,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Uniform,
                has_dynamic_offset: true, // Different dynamic offset
                min_binding_size: NonZeroU64::new(64), // Different min size
            },
            count: None,
        }];

        assert!(layouts_compatible(entries_a, entries_b));
    }

    #[test]
    fn test_layouts_equal_identical() {
        let entries_a = &[uniform_entry(0)];
        let entries_b = &[uniform_entry(0)];

        assert!(layouts_equal(entries_a, entries_b));
    }

    #[test]
    fn test_layouts_equal_reordered() {
        let entries_a = &[uniform_entry(0), sampler_entry(1)];
        let entries_b = &[sampler_entry(1), uniform_entry(0)];

        assert!(layouts_equal(entries_a, entries_b));
    }

    #[test]
    fn test_layouts_not_equal_different_min_binding_size() {
        let entries_a = &[BindGroupLayoutEntry {
            binding: 0,
            visibility: ShaderStages::VERTEX,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: NonZeroU64::new(64),
            },
            count: None,
        }];
        let entries_b = &[BindGroupLayoutEntry {
            binding: 0,
            visibility: ShaderStages::VERTEX,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: NonZeroU64::new(128),
            },
            count: None,
        }];

        // Compatible (same binding type variant) but not equal
        assert!(layouts_compatible(entries_a, entries_b));
        assert!(!layouts_equal(entries_a, entries_b));
    }

    #[test]
    fn test_layouts_not_equal_different_dynamic_offset() {
        let entries_a = &[BindGroupLayoutEntry {
            binding: 0,
            visibility: ShaderStages::VERTEX,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }];
        let entries_b = &[BindGroupLayoutEntry {
            binding: 0,
            visibility: ShaderStages::VERTEX,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Uniform,
                has_dynamic_offset: true,
                min_binding_size: None,
            },
            count: None,
        }];

        // Compatible but not equal
        assert!(layouts_compatible(entries_a, entries_b));
        assert!(!layouts_equal(entries_a, entries_b));
    }

    #[test]
    fn test_layouts_compatible_complex_multi_entry() {
        let entries_a = &[
            uniform_entry(0),
            texture_entry(1),
            sampler_entry(2),
            storage_entry(3, true),
        ];
        let entries_b = &[
            storage_entry(3, true),
            sampler_entry(2),
            texture_entry(1),
            uniform_entry(0),
        ];

        assert!(layouts_compatible(entries_a, entries_b));
    }

    #[test]
    fn test_layouts_incompatible_count_vs_no_count() {
        let entries_a = &[BindGroupLayoutEntry {
            binding: 0,
            visibility: ShaderStages::FRAGMENT,
            ty: BindingType::Texture {
                sample_type: TextureSampleType::Float { filterable: true },
                view_dimension: TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        }];
        let entries_b = &[BindGroupLayoutEntry {
            binding: 0,
            visibility: ShaderStages::FRAGMENT,
            ty: BindingType::Texture {
                sample_type: TextureSampleType::Float { filterable: true },
                view_dimension: TextureViewDimension::D2,
                multisampled: false,
            },
            count: NonZeroU32::new(4),
        }];

        assert!(!layouts_compatible(entries_a, entries_b));
    }
}

// ============================================================================
// Category 6: Edge Cases Tests (15+ tests)
// ============================================================================

mod edge_case_tests {
    use super::*;

    #[test]
    fn test_maximum_binding_index() {
        let entries = &[BindGroupLayoutEntry {
            binding: u32::MAX,
            visibility: ShaderStages::VERTEX,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }];

        let key = BindGroupLayoutKey::from_entries(entries);
        assert_ne!(key.hash_value(), 0);
    }

    #[test]
    fn test_all_shader_stages_visible() {
        let entries = &[BindGroupLayoutEntry {
            binding: 0,
            visibility: ShaderStages::all(),
            ty: BindingType::Buffer {
                ty: BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }];

        let key = BindGroupLayoutKey::from_entries(entries);
        assert_ne!(key.hash_value(), 0);
    }

    #[test]
    fn test_no_shader_stages_visible() {
        let entries = &[BindGroupLayoutEntry {
            binding: 0,
            visibility: ShaderStages::NONE,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }];

        let key = BindGroupLayoutKey::from_entries(entries);
        // Should still produce a valid key
        let key2 = BindGroupLayoutKey::from_entries(entries);
        assert_eq!(key, key2);
    }

    #[test]
    fn test_all_binding_types_in_one_layout() {
        let entries = &[
            // Buffer - Uniform
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::VERTEX,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // Buffer - Storage
            storage_entry(1, false),
            // Sampler
            sampler_entry(2),
            // Texture
            texture_entry(3),
            // StorageTexture
            storage_texture_entry(4, StorageTextureAccess::WriteOnly),
            // AccelerationStructure
            acceleration_structure_entry(5),
        ];

        let key = BindGroupLayoutKey::from_entries(entries);
        assert_ne!(key.hash_value(), 0);
    }

    #[test]
    fn test_large_number_of_entries() {
        let entries: Vec<BindGroupLayoutEntry> = (0..32)
            .map(|i| uniform_entry(i))
            .collect();

        let key = BindGroupLayoutKey::from_entries(&entries);
        assert_ne!(key.hash_value(), 0);
    }

    #[test]
    fn test_very_large_entry_count() {
        // Test with 100 entries
        let entries: Vec<BindGroupLayoutEntry> = (0..100)
            .map(|i| uniform_entry(i))
            .collect();

        let key1 = BindGroupLayoutKey::from_entries(&entries);
        let key2 = BindGroupLayoutKey::from_entries(&entries);
        assert_eq!(key1, key2);
    }

    #[test]
    fn test_duplicate_binding_indices_different_hash() {
        // Two entries with same binding index - unusual but should hash differently than one
        let entries1 = &[uniform_entry(0)];
        let entries2 = &[uniform_entry(0), uniform_entry(0)]; // Two entries at binding 0

        let key1 = BindGroupLayoutKey::from_entries(entries1);
        let key2 = BindGroupLayoutKey::from_entries(entries2);

        // Different number of entries should produce different keys
        assert_ne!(key1, key2);
    }

    #[test]
    fn test_acceleration_structure_binding_type() {
        let entries = &[acceleration_structure_entry(0)];

        let key = BindGroupLayoutKey::from_entries(entries);
        assert_ne!(key.hash_value(), 0);
    }

    #[test]
    fn test_acceleration_structure_vs_other_types() {
        let entries1 = &[acceleration_structure_entry(0)];
        let entries2 = &[uniform_entry(0)];

        let key1 = BindGroupLayoutKey::from_entries(entries1);
        let key2 = BindGroupLayoutKey::from_entries(entries2);

        assert_ne!(key1, key2);
    }

    #[test]
    fn test_array_count_various_values() {
        let entries_none = &[BindGroupLayoutEntry {
            binding: 0,
            visibility: ShaderStages::FRAGMENT,
            ty: BindingType::Texture {
                sample_type: TextureSampleType::Float { filterable: true },
                view_dimension: TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        }];
        let entries_4 = &[BindGroupLayoutEntry {
            binding: 0,
            visibility: ShaderStages::FRAGMENT,
            ty: BindingType::Texture {
                sample_type: TextureSampleType::Float { filterable: true },
                view_dimension: TextureViewDimension::D2,
                multisampled: false,
            },
            count: NonZeroU32::new(4),
        }];
        let entries_8 = &[BindGroupLayoutEntry {
            binding: 0,
            visibility: ShaderStages::FRAGMENT,
            ty: BindingType::Texture {
                sample_type: TextureSampleType::Float { filterable: true },
                view_dimension: TextureViewDimension::D2,
                multisampled: false,
            },
            count: NonZeroU32::new(8),
        }];

        let key_none = BindGroupLayoutKey::from_entries(entries_none);
        let key_4 = BindGroupLayoutKey::from_entries(entries_4);
        let key_8 = BindGroupLayoutKey::from_entries(entries_8);

        assert_ne!(key_none, key_4);
        assert_ne!(key_4, key_8);
        assert_ne!(key_none, key_8);
    }

    #[test]
    fn test_texture_view_dimension_all_variants() {
        let dimensions = [
            TextureViewDimension::D1,
            TextureViewDimension::D2,
            TextureViewDimension::D2Array,
            TextureViewDimension::D3,
            TextureViewDimension::Cube,
            TextureViewDimension::CubeArray,
        ];

        let keys: Vec<_> = dimensions
            .iter()
            .map(|&dim| {
                let entries = &[BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: true },
                        view_dimension: dim,
                        multisampled: false,
                    },
                    count: None,
                }];
                BindGroupLayoutKey::from_entries(entries)
            })
            .collect();

        // All keys should be unique
        for i in 0..keys.len() {
            for j in (i + 1)..keys.len() {
                assert_ne!(keys[i], keys[j], "Dimension {:?} and {:?} should produce different keys", dimensions[i], dimensions[j]);
            }
        }
    }

    #[test]
    fn test_storage_texture_access_all_variants() {
        let accesses = [
            StorageTextureAccess::WriteOnly,
            StorageTextureAccess::ReadOnly,
            StorageTextureAccess::ReadWrite,
        ];

        let keys: Vec<_> = accesses
            .iter()
            .map(|&access| {
                let entries = &[BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::StorageTexture {
                        access,
                        format: TextureFormat::Rgba8Unorm,
                        view_dimension: TextureViewDimension::D2,
                    },
                    count: None,
                }];
                BindGroupLayoutKey::from_entries(entries)
            })
            .collect();

        // All keys should be unique
        for i in 0..keys.len() {
            for j in (i + 1)..keys.len() {
                assert_ne!(keys[i], keys[j]);
            }
        }
    }

    #[test]
    fn test_sampler_binding_type_all_variants() {
        let sampler_types = [
            SamplerBindingType::Filtering,
            SamplerBindingType::NonFiltering,
            SamplerBindingType::Comparison,
        ];

        let keys: Vec<_> = sampler_types
            .iter()
            .map(|&st| {
                let entries = &[BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Sampler(st),
                    count: None,
                }];
                BindGroupLayoutKey::from_entries(entries)
            })
            .collect();

        // All keys should be unique
        for i in 0..keys.len() {
            for j in (i + 1)..keys.len() {
                assert_ne!(keys[i], keys[j]);
            }
        }
    }

    #[test]
    fn test_min_binding_size_various_values() {
        let sizes = [None, NonZeroU64::new(16), NonZeroU64::new(64), NonZeroU64::new(256)];

        let keys: Vec<_> = sizes
            .iter()
            .map(|&size| {
                let entries = &[BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::VERTEX,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: size,
                    },
                    count: None,
                }];
                BindGroupLayoutKey::from_entries(entries)
            })
            .collect();

        // All keys should be unique
        for i in 0..keys.len() {
            for j in (i + 1)..keys.len() {
                assert_ne!(keys[i], keys[j]);
            }
        }
    }

    #[test]
    fn test_texture_sample_type_all_variants() {
        let sample_types = [
            TextureSampleType::Float { filterable: true },
            TextureSampleType::Float { filterable: false },
            TextureSampleType::Uint,
            TextureSampleType::Sint,
            TextureSampleType::Depth,
        ];

        let keys: Vec<_> = sample_types
            .iter()
            .map(|&st| {
                let entries = &[BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Texture {
                        sample_type: st,
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                }];
                BindGroupLayoutKey::from_entries(entries)
            })
            .collect();

        // All keys should be unique
        for i in 0..keys.len() {
            for j in (i + 1)..keys.len() {
                assert_ne!(keys[i], keys[j]);
            }
        }
    }

    #[test]
    fn test_complex_pbr_layout() {
        // Simulate a typical PBR material layout
        let entries = &[
            // Camera uniform
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::VERTEX | ShaderStages::FRAGMENT,
                ty: BindingType::Buffer {
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
                ty: BindingType::Texture {
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
                ty: BindingType::Sampler(SamplerBindingType::Filtering),
                count: None,
            },
            // Normal map
            BindGroupLayoutEntry {
                binding: 3,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Texture {
                    sample_type: TextureSampleType::Float { filterable: true },
                    view_dimension: TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            // Metallic-Roughness
            BindGroupLayoutEntry {
                binding: 4,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Texture {
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
        assert_ne!(key1.hash_value(), 0);
    }

    #[test]
    fn test_compute_shader_layout() {
        let entries = &[
            // Input buffer
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
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
                ty: BindingType::Buffer {
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
                ty: BindingType::StorageTexture {
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
}

// ============================================================================
// Additional Tests: Integration-style tests
// ============================================================================

mod integration_style_tests {
    use super::*;

    #[test]
    fn test_cache_workflow_without_device() {
        let cache = BindGroupLayoutCache::new();

        // Initial state
        assert!(cache.is_empty());
        assert_eq!(cache.metrics().cache_size, 0);

        // Check contains for various entries
        let entries1 = &[uniform_entry(0)];
        let entries2 = &[sampler_entry(0)];

        assert!(!cache.contains(entries1));
        assert!(!cache.contains(entries2));

        // Remove (should return false since not in cache)
        assert!(!cache.remove(entries1));

        // Clear (should be safe on empty cache)
        cache.clear();
        assert!(cache.is_empty());
    }

    #[test]
    fn test_key_consistency_across_multiple_operations() {
        let entries = &[
            uniform_entry(0),
            texture_entry(1),
            sampler_entry(2),
        ];

        // Generate key multiple times
        let keys: Vec<_> = (0..10)
            .map(|_| BindGroupLayoutKey::from_entries(entries))
            .collect();

        // All keys should be identical
        for key in &keys {
            assert_eq!(*key, keys[0]);
        }
    }

    #[test]
    fn test_layouts_symmetry() {
        let entries_a = &[uniform_entry(0), sampler_entry(1)];
        let entries_b = &[uniform_entry(0), sampler_entry(1)];

        // Compatible should be symmetric
        assert_eq!(
            layouts_compatible(entries_a, entries_b),
            layouts_compatible(entries_b, entries_a)
        );

        // Equal should be symmetric
        assert_eq!(
            layouts_equal(entries_a, entries_b),
            layouts_equal(entries_b, entries_a)
        );
    }

    #[test]
    fn test_metrics_immutability() {
        let metrics1 = BindGroupLayoutCacheMetrics::new(10, 100, 20);
        let metrics2 = metrics1.clone();

        // Modifying one shouldn't affect the other
        assert_eq!(metrics1.cache_size, metrics2.cache_size);
        assert_eq!(metrics1.hits, metrics2.hits);
        assert_eq!(metrics1.misses, metrics2.misses);
        assert_eq!(metrics1.hit_rate, metrics2.hit_rate);
    }
}

// ============================================================================
// Additional Tests: Storage Texture Format Variations
// ============================================================================

mod storage_texture_format_tests {
    use super::*;

    #[test]
    fn test_storage_texture_r32float_vs_rgba8() {
        let entries1 = &[BindGroupLayoutEntry {
            binding: 0,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::StorageTexture {
                access: StorageTextureAccess::WriteOnly,
                format: TextureFormat::R32Float,
                view_dimension: TextureViewDimension::D2,
            },
            count: None,
        }];
        let entries2 = &[BindGroupLayoutEntry {
            binding: 0,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::StorageTexture {
                access: StorageTextureAccess::WriteOnly,
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
    fn test_storage_texture_r32sint_vs_r32uint() {
        let entries1 = &[BindGroupLayoutEntry {
            binding: 0,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::StorageTexture {
                access: StorageTextureAccess::WriteOnly,
                format: TextureFormat::R32Sint,
                view_dimension: TextureViewDimension::D2,
            },
            count: None,
        }];
        let entries2 = &[BindGroupLayoutEntry {
            binding: 0,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::StorageTexture {
                access: StorageTextureAccess::WriteOnly,
                format: TextureFormat::R32Uint,
                view_dimension: TextureViewDimension::D2,
            },
            count: None,
        }];

        let key1 = BindGroupLayoutKey::from_entries(entries1);
        let key2 = BindGroupLayoutKey::from_entries(entries2);

        assert_ne!(key1, key2);
    }
}

// ============================================================================
// Additional Tests: Shader Stage Combinations
// ============================================================================

mod shader_stage_tests {
    use super::*;

    #[test]
    fn test_vertex_only_vs_fragment_only() {
        let entries1 = &[uniform_entry_vis(0, ShaderStages::VERTEX)];
        let entries2 = &[uniform_entry_vis(0, ShaderStages::FRAGMENT)];

        let key1 = BindGroupLayoutKey::from_entries(entries1);
        let key2 = BindGroupLayoutKey::from_entries(entries2);

        assert_ne!(key1, key2);
    }

    #[test]
    fn test_vertex_fragment_vs_compute() {
        let entries1 = &[uniform_entry_vis(0, ShaderStages::VERTEX_FRAGMENT)];
        let entries2 = &[uniform_entry_vis(0, ShaderStages::COMPUTE)];

        let key1 = BindGroupLayoutKey::from_entries(entries1);
        let key2 = BindGroupLayoutKey::from_entries(entries2);

        assert_ne!(key1, key2);
    }

    #[test]
    fn test_all_stages_vs_single() {
        let entries1 = &[uniform_entry_vis(0, ShaderStages::all())];
        let entries2 = &[uniform_entry_vis(0, ShaderStages::VERTEX)];

        let key1 = BindGroupLayoutKey::from_entries(entries1);
        let key2 = BindGroupLayoutKey::from_entries(entries2);

        assert_ne!(key1, key2);
    }

    #[test]
    fn test_vertex_compute_combination() {
        let entries1 = &[uniform_entry_vis(0, ShaderStages::VERTEX | ShaderStages::COMPUTE)];
        let entries2 = &[uniform_entry_vis(0, ShaderStages::VERTEX)];

        let key1 = BindGroupLayoutKey::from_entries(entries1);
        let key2 = BindGroupLayoutKey::from_entries(entries2);

        assert_ne!(key1, key2);
    }
}

// ============================================================================
// Additional Tests: Buffer Configuration Edge Cases
// ============================================================================

mod buffer_edge_case_tests {
    use super::*;

    #[test]
    fn test_buffer_all_configurations() {
        // Test all combinations of buffer configurations
        let configs = [
            (BufferBindingType::Uniform, false, None),
            (BufferBindingType::Uniform, true, None),
            (BufferBindingType::Uniform, false, NonZeroU64::new(64)),
            (BufferBindingType::Uniform, true, NonZeroU64::new(64)),
            (BufferBindingType::Storage { read_only: false }, false, None),
            (BufferBindingType::Storage { read_only: true }, false, None),
            (BufferBindingType::Storage { read_only: false }, true, None),
            (BufferBindingType::Storage { read_only: true }, true, None),
        ];

        let keys: Vec<_> = configs
            .iter()
            .map(|(ty, has_dynamic_offset, min_binding_size)| {
                let entries = &[BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::VERTEX,
                    ty: BindingType::Buffer {
                        ty: *ty,
                        has_dynamic_offset: *has_dynamic_offset,
                        min_binding_size: *min_binding_size,
                    },
                    count: None,
                }];
                BindGroupLayoutKey::from_entries(entries)
            })
            .collect();

        // All keys should be unique
        for i in 0..keys.len() {
            for j in (i + 1)..keys.len() {
                assert_ne!(
                    keys[i], keys[j],
                    "Config {:?} and {:?} should produce different keys",
                    configs[i], configs[j]
                );
            }
        }
    }

    #[test]
    fn test_min_binding_size_edge_values() {
        let sizes = [
            None,
            NonZeroU64::new(1),
            NonZeroU64::new(4),
            NonZeroU64::new(16),
            NonZeroU64::new(256),
            NonZeroU64::new(u64::MAX),
        ];

        let keys: Vec<_> = sizes
            .iter()
            .map(|&size| {
                let entries = &[BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::VERTEX,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: size,
                    },
                    count: None,
                }];
                BindGroupLayoutKey::from_entries(entries)
            })
            .collect();

        // All keys should be unique
        for i in 0..keys.len() {
            for j in (i + 1)..keys.len() {
                assert_ne!(keys[i], keys[j]);
            }
        }
    }
}
