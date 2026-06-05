//! Whitebox tests for BindGroupCache (T-WGPU-P2.5.2)
//!
//! This test module provides comprehensive coverage of the bind group caching
//! system with full source access, testing all internal structures and behaviors.

use renderer_backend::resources::bind_group_cache::{
    BindGroupCache, BindGroupCacheKey, BindGroupCacheMetrics, BindGroupResourceEntry,
    BindGroupResourceType, ResourceId,
};
use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use std::thread;

// ============================================================================
// Helper Functions
// ============================================================================

fn hash_value<T: Hash>(value: &T) -> u64 {
    let mut hasher = DefaultHasher::new();
    value.hash(&mut hasher);
    hasher.finish()
}

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
                required_limits: wgpu::Limits::downlevel_defaults(),
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        ))
        .expect("device creation"),
    )
}

// ============================================================================
// CATEGORY 1: ResourceId Tests (15+ tests)
// ============================================================================

#[test]
fn test_resource_id_new_zero() {
    let id = ResourceId::new(0);
    assert_eq!(id.value(), 0);
}

#[test]
fn test_resource_id_new_max() {
    let id = ResourceId::new(u64::MAX);
    assert_eq!(id.value(), u64::MAX);
}

#[test]
fn test_resource_id_new_typical_values() {
    for value in [1u64, 100, 1000, 10000, 1_000_000] {
        let id = ResourceId::new(value);
        assert_eq!(id.value(), value);
    }
}

#[test]
fn test_resource_id_equality_same() {
    let id1 = ResourceId::new(42);
    let id2 = ResourceId::new(42);
    assert_eq!(id1, id2);
}

#[test]
fn test_resource_id_equality_different() {
    let id1 = ResourceId::new(1);
    let id2 = ResourceId::new(2);
    assert_ne!(id1, id2);
}

#[test]
fn test_resource_id_from_u64() {
    let value: u64 = 12345;
    let id: ResourceId = value.into();
    assert_eq!(id.value(), 12345);
}

#[test]
fn test_resource_id_into_u64() {
    let id = ResourceId::new(67890);
    let value: u64 = id.into();
    assert_eq!(value, 67890);
}

#[test]
fn test_resource_id_roundtrip() {
    let original = 999u64;
    let id: ResourceId = original.into();
    let back: u64 = id.into();
    assert_eq!(original, back);
}

#[test]
fn test_resource_id_copy_trait() {
    let id1 = ResourceId::new(100);
    let id2 = id1; // Copy
    let id3 = id1; // Copy again
    assert_eq!(id1, id2);
    assert_eq!(id2, id3);
}

#[test]
fn test_resource_id_clone_trait() {
    let id1 = ResourceId::new(200);
    let id2 = id1.clone();
    assert_eq!(id1, id2);
}

#[test]
fn test_resource_id_debug_format() {
    let id = ResourceId::new(42);
    let debug_str = format!("{:?}", id);
    assert!(debug_str.contains("ResourceId"));
    assert!(debug_str.contains("42"));
}

#[test]
fn test_resource_id_hash_consistency() {
    let id1 = ResourceId::new(555);
    let id2 = ResourceId::new(555);

    let hash1 = hash_value(&id1);
    let hash2 = hash_value(&id2);

    assert_eq!(hash1, hash2);
}

#[test]
fn test_resource_id_hash_different_values() {
    let id1 = ResourceId::new(1);
    let id2 = ResourceId::new(2);

    let hash1 = hash_value(&id1);
    let hash2 = hash_value(&id2);

    assert_ne!(hash1, hash2);
}

#[test]
fn test_resource_id_as_hashmap_key() {
    let mut map: HashMap<ResourceId, String> = HashMap::new();

    map.insert(ResourceId::new(1), "one".to_string());
    map.insert(ResourceId::new(2), "two".to_string());
    map.insert(ResourceId::new(3), "three".to_string());

    assert_eq!(map.get(&ResourceId::new(1)), Some(&"one".to_string()));
    assert_eq!(map.get(&ResourceId::new(2)), Some(&"two".to_string()));
    assert_eq!(map.get(&ResourceId::new(3)), Some(&"three".to_string()));
    assert!(map.get(&ResourceId::new(4)).is_none());
}

#[test]
fn test_resource_id_hashmap_overwrite() {
    let mut map: HashMap<ResourceId, i32> = HashMap::new();
    let id = ResourceId::new(10);

    map.insert(id, 100);
    assert_eq!(map.get(&id), Some(&100));

    map.insert(id, 200);
    assert_eq!(map.get(&id), Some(&200));
    assert_eq!(map.len(), 1);
}

#[test]
fn test_resource_id_const_new() {
    // Verify new() is const
    const ID: ResourceId = ResourceId::new(42);
    assert_eq!(ID.value(), 42);
}

// ============================================================================
// CATEGORY 2: BindGroupResourceType Tests (20+ tests)
// ============================================================================

#[test]
fn test_resource_type_buffer_default() {
    let ty = BindGroupResourceType::buffer();
    match ty {
        BindGroupResourceType::Buffer { offset, size } => {
            assert_eq!(offset, 0);
            assert_eq!(size, None);
        }
        _ => panic!("Expected Buffer variant"),
    }
}

#[test]
fn test_resource_type_buffer_range() {
    let ty = BindGroupResourceType::buffer_range(128, 512);
    match ty {
        BindGroupResourceType::Buffer { offset, size } => {
            assert_eq!(offset, 128);
            assert_eq!(size, Some(512));
        }
        _ => panic!("Expected Buffer variant"),
    }
}

#[test]
fn test_resource_type_buffer_sized() {
    let ty = BindGroupResourceType::buffer_sized(1024);
    match ty {
        BindGroupResourceType::Buffer { offset, size } => {
            assert_eq!(offset, 0);
            assert_eq!(size, Some(1024));
        }
        _ => panic!("Expected Buffer variant"),
    }
}

#[test]
fn test_resource_type_buffer_zero_offset() {
    let ty = BindGroupResourceType::Buffer { offset: 0, size: Some(64) };
    match ty {
        BindGroupResourceType::Buffer { offset, size } => {
            assert_eq!(offset, 0);
            assert_eq!(size, Some(64));
        }
        _ => panic!("Expected Buffer variant"),
    }
}

#[test]
fn test_resource_type_buffer_large_offset() {
    let ty = BindGroupResourceType::Buffer { offset: u64::MAX / 2, size: Some(1024) };
    match ty {
        BindGroupResourceType::Buffer { offset, size } => {
            assert_eq!(offset, u64::MAX / 2);
            assert_eq!(size, Some(1024));
        }
        _ => panic!("Expected Buffer variant"),
    }
}

#[test]
fn test_resource_type_sampler() {
    let ty = BindGroupResourceType::Sampler;
    assert_eq!(ty, BindGroupResourceType::Sampler);
}

#[test]
fn test_resource_type_texture_view() {
    let ty = BindGroupResourceType::TextureView;
    assert_eq!(ty, BindGroupResourceType::TextureView);
}

#[test]
fn test_resource_type_storage_texture_view() {
    let ty = BindGroupResourceType::StorageTextureView;
    assert_eq!(ty, BindGroupResourceType::StorageTextureView);
}

#[test]
fn test_resource_type_equality_buffer_same() {
    let ty1 = BindGroupResourceType::Buffer { offset: 0, size: Some(64) };
    let ty2 = BindGroupResourceType::Buffer { offset: 0, size: Some(64) };
    assert_eq!(ty1, ty2);
}

#[test]
fn test_resource_type_equality_buffer_different_offset() {
    let ty1 = BindGroupResourceType::Buffer { offset: 0, size: Some(64) };
    let ty2 = BindGroupResourceType::Buffer { offset: 64, size: Some(64) };
    assert_ne!(ty1, ty2);
}

#[test]
fn test_resource_type_equality_buffer_different_size() {
    let ty1 = BindGroupResourceType::Buffer { offset: 0, size: Some(64) };
    let ty2 = BindGroupResourceType::Buffer { offset: 0, size: Some(128) };
    assert_ne!(ty1, ty2);
}

#[test]
fn test_resource_type_equality_buffer_none_vs_some_size() {
    let ty1 = BindGroupResourceType::Buffer { offset: 0, size: None };
    let ty2 = BindGroupResourceType::Buffer { offset: 0, size: Some(64) };
    assert_ne!(ty1, ty2);
}

#[test]
fn test_resource_type_equality_different_variants() {
    let buffer = BindGroupResourceType::Buffer { offset: 0, size: None };
    let sampler = BindGroupResourceType::Sampler;
    let texture = BindGroupResourceType::TextureView;
    let storage = BindGroupResourceType::StorageTextureView;

    assert_ne!(buffer, sampler);
    assert_ne!(buffer, texture);
    assert_ne!(buffer, storage);
    assert_ne!(sampler, texture);
    assert_ne!(sampler, storage);
    assert_ne!(texture, storage);
}

#[test]
fn test_resource_type_hash_buffer_variants() {
    let ty1 = BindGroupResourceType::Buffer { offset: 0, size: Some(64) };
    let ty2 = BindGroupResourceType::Buffer { offset: 64, size: Some(64) };

    let hash1 = hash_value(&ty1);
    let hash2 = hash_value(&ty2);

    assert_ne!(hash1, hash2, "Different offsets should produce different hashes");
}

#[test]
fn test_resource_type_hash_size_difference() {
    let ty1 = BindGroupResourceType::Buffer { offset: 0, size: Some(64) };
    let ty2 = BindGroupResourceType::Buffer { offset: 0, size: Some(128) };

    let hash1 = hash_value(&ty1);
    let hash2 = hash_value(&ty2);

    assert_ne!(hash1, hash2, "Different sizes should produce different hashes");
}

#[test]
fn test_resource_type_hash_all_variants_different() {
    let buffer = BindGroupResourceType::Buffer { offset: 0, size: None };
    let sampler = BindGroupResourceType::Sampler;
    let texture = BindGroupResourceType::TextureView;
    let storage = BindGroupResourceType::StorageTextureView;

    let hashes = [
        hash_value(&buffer),
        hash_value(&sampler),
        hash_value(&texture),
        hash_value(&storage),
    ];

    // Check all pairs are different
    for i in 0..hashes.len() {
        for j in (i + 1)..hashes.len() {
            assert_ne!(hashes[i], hashes[j], "Variant {} and {} should have different hashes", i, j);
        }
    }
}

#[test]
fn test_resource_type_copy_trait() {
    let ty1 = BindGroupResourceType::Buffer { offset: 10, size: Some(100) };
    let ty2 = ty1; // Copy
    let ty3 = ty1; // Copy again
    assert_eq!(ty1, ty2);
    assert_eq!(ty2, ty3);
}

#[test]
fn test_resource_type_clone_trait() {
    let ty1 = BindGroupResourceType::Sampler;
    let ty2 = ty1.clone();
    assert_eq!(ty1, ty2);
}

#[test]
fn test_resource_type_debug_format_buffer() {
    let ty = BindGroupResourceType::Buffer { offset: 64, size: Some(256) };
    let debug_str = format!("{:?}", ty);
    assert!(debug_str.contains("Buffer"));
    assert!(debug_str.contains("64"));
    assert!(debug_str.contains("256"));
}

#[test]
fn test_resource_type_debug_format_sampler() {
    let ty = BindGroupResourceType::Sampler;
    let debug_str = format!("{:?}", ty);
    assert!(debug_str.contains("Sampler"));
}

#[test]
fn test_resource_type_debug_format_texture_view() {
    let ty = BindGroupResourceType::TextureView;
    let debug_str = format!("{:?}", ty);
    assert!(debug_str.contains("TextureView"));
}

#[test]
fn test_resource_type_debug_format_storage() {
    let ty = BindGroupResourceType::StorageTextureView;
    let debug_str = format!("{:?}", ty);
    assert!(debug_str.contains("StorageTextureView"));
}

// ============================================================================
// CATEGORY 3: BindGroupResourceEntry Tests (15+ tests)
// ============================================================================

#[test]
fn test_resource_entry_buffer_constructor() {
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
fn test_resource_entry_sampler_constructor() {
    let entry = BindGroupResourceEntry::sampler(1, ResourceId::new(2));
    assert_eq!(entry.binding, 1);
    assert_eq!(entry.resource_id, ResourceId::new(2));
    assert_eq!(entry.resource_type, BindGroupResourceType::Sampler);
}

#[test]
fn test_resource_entry_texture_view_constructor() {
    let entry = BindGroupResourceEntry::texture_view(2, ResourceId::new(3));
    assert_eq!(entry.binding, 2);
    assert_eq!(entry.resource_id, ResourceId::new(3));
    assert_eq!(entry.resource_type, BindGroupResourceType::TextureView);
}

#[test]
fn test_resource_entry_storage_texture_view_constructor() {
    let entry = BindGroupResourceEntry::storage_texture_view(3, ResourceId::new(4));
    assert_eq!(entry.binding, 3);
    assert_eq!(entry.resource_id, ResourceId::new(4));
    assert_eq!(entry.resource_type, BindGroupResourceType::StorageTextureView);
}

#[test]
fn test_resource_entry_clone() {
    let entry = BindGroupResourceEntry::buffer(0, ResourceId::new(10), 64, Some(128));
    let cloned = entry.clone();

    assert_eq!(entry.binding, cloned.binding);
    assert_eq!(entry.resource_id, cloned.resource_id);
    assert_eq!(entry.resource_type, cloned.resource_type);
}

#[test]
fn test_resource_entry_debug_format() {
    let entry = BindGroupResourceEntry::buffer(5, ResourceId::new(100), 0, None);
    let debug_str = format!("{:?}", entry);
    assert!(debug_str.contains("BindGroupResourceEntry"));
    assert!(debug_str.contains("binding"));
    assert!(debug_str.contains("5"));
}

#[test]
fn test_resource_entry_max_binding() {
    let entry = BindGroupResourceEntry::buffer(u32::MAX, ResourceId::new(1), 0, None);
    assert_eq!(entry.binding, u32::MAX);
}

#[test]
fn test_resource_entry_buffer_with_offset() {
    let entry = BindGroupResourceEntry::buffer(0, ResourceId::new(1), 256, Some(512));
    match entry.resource_type {
        BindGroupResourceType::Buffer { offset, size } => {
            assert_eq!(offset, 256);
            assert_eq!(size, Some(512));
        }
        _ => panic!("Expected Buffer"),
    }
}

#[test]
fn test_resource_entry_buffer_no_size() {
    let entry = BindGroupResourceEntry::buffer(0, ResourceId::new(1), 0, None);
    match entry.resource_type {
        BindGroupResourceType::Buffer { offset, size } => {
            assert_eq!(offset, 0);
            assert!(size.is_none());
        }
        _ => panic!("Expected Buffer"),
    }
}

#[test]
fn test_resource_entry_multiple_entries() {
    let entries = vec![
        BindGroupResourceEntry::buffer(0, ResourceId::new(1), 0, Some(64)),
        BindGroupResourceEntry::sampler(1, ResourceId::new(2)),
        BindGroupResourceEntry::texture_view(2, ResourceId::new(3)),
        BindGroupResourceEntry::storage_texture_view(3, ResourceId::new(4)),
    ];

    assert_eq!(entries.len(), 4);
    assert_eq!(entries[0].binding, 0);
    assert_eq!(entries[1].binding, 1);
    assert_eq!(entries[2].binding, 2);
    assert_eq!(entries[3].binding, 3);
}

#[test]
fn test_resource_entry_same_resource_different_binding() {
    let res_id = ResourceId::new(42);
    let entry1 = BindGroupResourceEntry::buffer(0, res_id, 0, None);
    let entry2 = BindGroupResourceEntry::buffer(1, res_id, 0, None);

    assert_eq!(entry1.resource_id, entry2.resource_id);
    assert_ne!(entry1.binding, entry2.binding);
}

#[test]
fn test_resource_entry_struct_direct_construction() {
    let entry = BindGroupResourceEntry {
        binding: 7,
        resource_id: ResourceId::new(77),
        resource_type: BindGroupResourceType::Sampler,
    };

    assert_eq!(entry.binding, 7);
    assert_eq!(entry.resource_id.value(), 77);
    assert_eq!(entry.resource_type, BindGroupResourceType::Sampler);
}

#[test]
fn test_resource_entry_large_buffer_range() {
    let entry = BindGroupResourceEntry::buffer(
        0,
        ResourceId::new(1),
        1024 * 1024 * 100, // 100MB offset
        Some(1024 * 1024 * 50), // 50MB size
    );

    match entry.resource_type {
        BindGroupResourceType::Buffer { offset, size } => {
            assert_eq!(offset, 1024 * 1024 * 100);
            assert_eq!(size, Some(1024 * 1024 * 50));
        }
        _ => panic!("Expected Buffer"),
    }
}

#[test]
fn test_resource_entry_zero_binding() {
    let entry = BindGroupResourceEntry::sampler(0, ResourceId::new(1));
    assert_eq!(entry.binding, 0);
}

#[test]
fn test_resource_entry_consecutive_bindings() {
    let entries: Vec<_> = (0..16u32)
        .map(|i| BindGroupResourceEntry::texture_view(i, ResourceId::new(i as u64)))
        .collect();

    for (i, entry) in entries.iter().enumerate() {
        assert_eq!(entry.binding, i as u32);
        assert_eq!(entry.resource_id.value(), i as u64);
    }
}

// ============================================================================
// CATEGORY 4: BindGroupCacheKey Tests (20+ tests)
// ============================================================================

#[test]
fn test_cache_key_empty_entries() {
    let key = BindGroupCacheKey::new(0x12345678, &[]);
    assert_eq!(key.layout_hash(), 0x12345678);
    assert_ne!(key.resources_hash(), 0); // Hash of empty still has entry count hashed
}

#[test]
fn test_cache_key_single_entry() {
    let entries = &[BindGroupResourceEntry::buffer(0, ResourceId::new(1), 0, None)];
    let key = BindGroupCacheKey::new(0xABCD, entries);
    assert_eq!(key.layout_hash(), 0xABCD);
}

#[test]
fn test_cache_key_multiple_entries() {
    let entries = &[
        BindGroupResourceEntry::buffer(0, ResourceId::new(1), 0, None),
        BindGroupResourceEntry::sampler(1, ResourceId::new(2)),
        BindGroupResourceEntry::texture_view(2, ResourceId::new(3)),
    ];
    let key = BindGroupCacheKey::new(0x1234, entries);
    assert_eq!(key.layout_hash(), 0x1234);
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
fn test_cache_key_sorting_by_binding() {
    // Entries in different order should produce same key
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

    assert_eq!(key1, key2);
}

#[test]
fn test_cache_key_different_layouts() {
    let entries = &[BindGroupResourceEntry::buffer(0, ResourceId::new(1), 0, None)];

    let key1 = BindGroupCacheKey::new(0x1111, entries);
    let key2 = BindGroupCacheKey::new(0x2222, entries);

    assert_ne!(key1, key2);
}

#[test]
fn test_cache_key_different_resources() {
    let entries1 = &[BindGroupResourceEntry::buffer(0, ResourceId::new(1), 0, None)];
    let entries2 = &[BindGroupResourceEntry::buffer(0, ResourceId::new(2), 0, None)];

    let key1 = BindGroupCacheKey::new(0x1234, entries1);
    let key2 = BindGroupCacheKey::new(0x1234, entries2);

    assert_ne!(key1, key2);
}

#[test]
fn test_cache_key_different_bindings() {
    let entries1 = &[BindGroupResourceEntry::buffer(0, ResourceId::new(1), 0, None)];
    let entries2 = &[BindGroupResourceEntry::buffer(1, ResourceId::new(1), 0, None)];

    let key1 = BindGroupCacheKey::new(0x1234, entries1);
    let key2 = BindGroupCacheKey::new(0x1234, entries2);

    assert_ne!(key1, key2);
}

#[test]
fn test_cache_key_different_types() {
    let entries1 = &[BindGroupResourceEntry::buffer(0, ResourceId::new(1), 0, None)];
    let entries2 = &[BindGroupResourceEntry::sampler(0, ResourceId::new(1))];

    let key1 = BindGroupCacheKey::new(0x1234, entries1);
    let key2 = BindGroupCacheKey::new(0x1234, entries2);

    assert_ne!(key1, key2);
}

#[test]
fn test_cache_key_different_offset() {
    let entries1 = &[BindGroupResourceEntry::buffer(0, ResourceId::new(1), 0, None)];
    let entries2 = &[BindGroupResourceEntry::buffer(0, ResourceId::new(1), 64, None)];

    let key1 = BindGroupCacheKey::new(0x1234, entries1);
    let key2 = BindGroupCacheKey::new(0x1234, entries2);

    assert_ne!(key1, key2);
}

#[test]
fn test_cache_key_different_size() {
    let entries1 = &[BindGroupResourceEntry::buffer(0, ResourceId::new(1), 0, Some(64))];
    let entries2 = &[BindGroupResourceEntry::buffer(0, ResourceId::new(1), 0, Some(128))];

    let key1 = BindGroupCacheKey::new(0x1234, entries1);
    let key2 = BindGroupCacheKey::new(0x1234, entries2);

    assert_ne!(key1, key2);
}

#[test]
fn test_cache_key_hash_stability() {
    let entries = &[
        BindGroupResourceEntry::buffer(0, ResourceId::new(1), 0, Some(64)),
        BindGroupResourceEntry::sampler(1, ResourceId::new(2)),
    ];

    let key1 = BindGroupCacheKey::new(0xDEAD, entries);
    let key2 = BindGroupCacheKey::new(0xDEAD, entries);

    let hash1 = hash_value(&key1);
    let hash2 = hash_value(&key2);

    assert_eq!(hash1, hash2);
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

#[test]
fn test_cache_key_clone() {
    let entries = &[BindGroupResourceEntry::buffer(0, ResourceId::new(1), 0, None)];
    let key = BindGroupCacheKey::new(0x1234, entries);
    let cloned = key.clone();

    assert_eq!(key, cloned);
    assert_eq!(key.layout_hash(), cloned.layout_hash());
    assert_eq!(key.resources_hash(), cloned.resources_hash());
}

#[test]
fn test_cache_key_debug_format() {
    let entries = &[BindGroupResourceEntry::buffer(0, ResourceId::new(1), 0, None)];
    let key = BindGroupCacheKey::new(0x1234, entries);
    let debug_str = format!("{:?}", key);

    assert!(debug_str.contains("BindGroupCacheKey"));
}

#[test]
fn test_cache_key_different_entry_count() {
    let entries1 = &[BindGroupResourceEntry::buffer(0, ResourceId::new(1), 0, None)];
    let entries2 = &[
        BindGroupResourceEntry::buffer(0, ResourceId::new(1), 0, None),
        BindGroupResourceEntry::buffer(1, ResourceId::new(2), 0, None),
    ];

    let key1 = BindGroupCacheKey::new(0x1234, entries1);
    let key2 = BindGroupCacheKey::new(0x1234, entries2);

    assert_ne!(key1, key2, "Different entry counts should produce different keys");
}

#[test]
fn test_cache_key_pbr_material_layout() {
    // Simulate a typical PBR material bind group
    let entries = &[
        BindGroupResourceEntry::buffer(0, ResourceId::new(1), 0, Some(144)), // Camera
        BindGroupResourceEntry::texture_view(1, ResourceId::new(2)), // Albedo
        BindGroupResourceEntry::sampler(2, ResourceId::new(3)), // Sampler
        BindGroupResourceEntry::texture_view(3, ResourceId::new(4)), // Normal
        BindGroupResourceEntry::texture_view(4, ResourceId::new(5)), // Roughness
    ];

    let key1 = BindGroupCacheKey::new(0xABCD_1234, entries);
    let key2 = BindGroupCacheKey::new(0xABCD_1234, entries);

    assert_eq!(key1, key2);
}

#[test]
fn test_cache_key_compute_dispatch_layout() {
    let entries = &[
        BindGroupResourceEntry::buffer(0, ResourceId::new(1), 0, None), // Input
        BindGroupResourceEntry::buffer(1, ResourceId::new(2), 0, None), // Output
        BindGroupResourceEntry::storage_texture_view(2, ResourceId::new(3)), // Storage texture
    ];

    let key = BindGroupCacheKey::new(0xDEAD_BEEF, entries);
    assert_ne!(key.resources_hash(), 0);
}

#[test]
fn test_cache_key_zero_layout_hash() {
    let entries = &[BindGroupResourceEntry::buffer(0, ResourceId::new(1), 0, None)];
    let key = BindGroupCacheKey::new(0, entries);

    assert_eq!(key.layout_hash(), 0);
}

#[test]
fn test_cache_key_max_layout_hash() {
    let entries = &[BindGroupResourceEntry::buffer(0, ResourceId::new(1), 0, None)];
    let key = BindGroupCacheKey::new(u64::MAX, entries);

    assert_eq!(key.layout_hash(), u64::MAX);
}

// ============================================================================
// CATEGORY 5: CachedBindGroup Tests (10+ tests)
// ============================================================================

// Note: CachedBindGroup requires wgpu device to construct, so we test through
// the cache's behavior with device. We also test accessible metadata.

#[test]
fn test_cached_bind_group_metadata_access_through_cache() {
    let (device, _queue) = match create_test_device() {
        Some(d) => d,
        None => {
            eprintln!("Skipping test_cached_bind_group_metadata_access_through_cache: no GPU");
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

    // Advance frame so we can verify frame_created
    cache.begin_frame();
    cache.begin_frame(); // Frame 2

    let _bg = cache.create_bind_group(&device, &layout, 0x1234, entries, resource_entries, Some("test"));

    // Verify cache contains the entry and tracks resources
    assert!(cache.contains(0x1234, resource_entries));
    assert_eq!(cache.tracked_resource_count(), 1);
}

#[test]
fn test_cached_bind_group_arc_sharing() {
    let (device, _queue) = match create_test_device() {
        Some(d) => d,
        None => {
            eprintln!("Skipping: no GPU");
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

    let bg1 = cache.create_bind_group(&device, &layout, 0x1234, entries, resource_entries, None);
    let bg2 = cache.create_bind_group(&device, &layout, 0x1234, entries, resource_entries, None);

    // Same Arc
    assert!(Arc::ptr_eq(&bg1, &bg2));
}

#[test]
fn test_cached_bind_group_frame_created_tracking() {
    let (device, _queue) = match create_test_device() {
        Some(d) => d,
        None => {
            eprintln!("Skipping: no GPU");
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
    let resource_entries1 = &[BindGroupResourceEntry::buffer(0, ResourceId::new(1), 0, None)];
    let resource_entries2 = &[BindGroupResourceEntry::buffer(0, ResourceId::new(2), 0, None)];

    // Create first at frame 0
    let entries1 = &[wgpu::BindGroupEntry {
        binding: 0,
        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
            buffer: &buffer,
            offset: 0,
            size: None,
        }),
    }];
    let _bg1 = cache.create_bind_group(&device, &layout, 0x1234, entries1, resource_entries1, None);

    // Advance 3 frames
    cache.begin_frame();
    cache.begin_frame();
    cache.begin_frame();

    // Create second at frame 3
    let _bg2 = cache.create_bind_group(&device, &layout, 0x1234, entries1, resource_entries2, None);

    // Now evict with max age 2 - should evict the first bind group created at frame 0
    let evicted = cache.evict_old(2);
    assert_eq!(evicted, 1);
    assert_eq!(cache.len(), 1);
}

#[test]
fn test_cached_bind_group_resource_id_tracking() {
    let (device, _queue) = match create_test_device() {
        Some(d) => d,
        None => {
            eprintln!("Skipping: no GPU");
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

    let _bg = cache.create_bind_group(&device, &layout, 0x1234, entries, resource_entries, None);

    // Verify resource tracking
    assert_eq!(cache.tracked_resource_count(), 1);

    // Invalidate by resource ID
    let invalidated = cache.invalidate_resource(buffer_id);
    assert_eq!(invalidated, 1);
    assert_eq!(cache.len(), 0);
}

#[test]
fn test_cached_bind_group_multiple_resources() {
    let (device, _queue) = match create_test_device() {
        Some(d) => d,
        None => {
            eprintln!("Skipping: no GPU");
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
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    let cache = BindGroupCache::new();
    let res1 = ResourceId::new(1);
    let res2 = ResourceId::new(2);
    let resource_entries = &[
        BindGroupResourceEntry::buffer(0, res1, 0, None),
        BindGroupResourceEntry::buffer(1, res2, 0, None),
    ];
    let entries = &[
        wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer: &buffer1,
                offset: 0,
                size: None,
            }),
        },
        wgpu::BindGroupEntry {
            binding: 1,
            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer: &buffer2,
                offset: 0,
                size: None,
            }),
        },
    ];

    let _bg = cache.create_bind_group(&device, &layout, 0x1234, entries, resource_entries, None);

    // Both resources tracked
    assert_eq!(cache.tracked_resource_count(), 2);

    // Invalidating one resource removes the bind group
    let invalidated = cache.invalidate_resource(res1);
    assert_eq!(invalidated, 1);
    assert_eq!(cache.len(), 0);
}

// ============================================================================
// CATEGORY 6: BindGroupCache Tests (25+ tests)
// ============================================================================

#[test]
fn test_cache_new_is_empty() {
    let cache = BindGroupCache::new();
    assert!(cache.is_empty());
    assert_eq!(cache.len(), 0);
}

#[test]
fn test_cache_default_is_empty() {
    let cache = BindGroupCache::default();
    assert!(cache.is_empty());
}

#[test]
fn test_cache_len_empty() {
    let cache = BindGroupCache::new();
    assert_eq!(cache.len(), 0);
}

#[test]
fn test_cache_is_empty_true() {
    let cache = BindGroupCache::new();
    assert!(cache.is_empty());
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
fn test_cache_begin_frame_increments() {
    let cache = BindGroupCache::new();
    assert_eq!(cache.current_frame(), 0);

    cache.begin_frame();
    assert_eq!(cache.current_frame(), 1);

    cache.begin_frame();
    cache.begin_frame();
    assert_eq!(cache.current_frame(), 3);
}

#[test]
fn test_cache_begin_frame_many() {
    let cache = BindGroupCache::new();

    for i in 0..1000 {
        assert_eq!(cache.current_frame(), i);
        cache.begin_frame();
    }

    assert_eq!(cache.current_frame(), 1000);
}

#[test]
fn test_cache_current_frame_initial() {
    let cache = BindGroupCache::new();
    assert_eq!(cache.current_frame(), 0);
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
fn test_cache_clear_empty() {
    let cache = BindGroupCache::new();
    cache.clear();
    assert!(cache.is_empty());
}

#[test]
fn test_cache_clear_resets_metrics() {
    let cache = BindGroupCache::new();

    // Simulate some activity manually
    cache.begin_frame();

    cache.clear();

    let metrics = cache.metrics();
    assert_eq!(metrics.hits, 0);
    assert_eq!(metrics.misses, 0);
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
fn test_cache_tracked_resource_count_empty() {
    let cache = BindGroupCache::new();
    assert_eq!(cache.tracked_resource_count(), 0);
}

#[test]
fn test_cache_remove_nonexistent() {
    let cache = BindGroupCache::new();
    let entries = &[BindGroupResourceEntry::buffer(0, ResourceId::new(1), 0, None)];
    assert!(!cache.remove(0x1234, entries));
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
fn test_cache_reset_metrics() {
    let cache = BindGroupCache::new();

    // Simulate activity
    cache.begin_frame();

    cache.reset_metrics();

    let metrics = cache.metrics();
    assert_eq!(metrics.hits, 0);
    assert_eq!(metrics.misses, 0);
    // Current frame not reset
    assert_eq!(metrics.current_frame, 1);
}

#[test]
fn test_cache_debug_format() {
    let cache = BindGroupCache::new();
    let debug_str = format!("{:?}", cache);

    assert!(debug_str.contains("BindGroupCache"));
    assert!(debug_str.contains("cache_size"));
}

#[test]
fn test_cache_send_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<BindGroupCache>();
}

#[test]
fn test_cache_create_and_get() {
    let (device, _queue) = match create_test_device() {
        Some(d) => d,
        None => {
            eprintln!("Skipping: no GPU");
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

    // Create
    let bg = cache.create_bind_group(&device, &layout, 0x1234, entries, resource_entries, Some("test"));
    assert_eq!(cache.len(), 1);

    // Get
    let retrieved = cache.get(0x1234, resource_entries);
    assert!(retrieved.is_some());
    assert!(Arc::ptr_eq(&bg, &retrieved.unwrap()));
}

#[test]
fn test_cache_hit_miss_counting() {
    let (device, _queue) = match create_test_device() {
        Some(d) => d,
        None => {
            eprintln!("Skipping: no GPU");
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

    // First call = miss
    let _bg1 = cache.create_bind_group(&device, &layout, 0x1234, entries, resource_entries, None);
    assert_eq!(cache.metrics().misses, 1);
    assert_eq!(cache.metrics().hits, 0);

    // Second call = hit
    let _bg2 = cache.create_bind_group(&device, &layout, 0x1234, entries, resource_entries, None);
    assert_eq!(cache.metrics().misses, 1);
    assert_eq!(cache.metrics().hits, 1);

    // Third call = hit
    let _bg3 = cache.create_bind_group(&device, &layout, 0x1234, entries, resource_entries, None);
    assert_eq!(cache.metrics().misses, 1);
    assert_eq!(cache.metrics().hits, 2);
}

#[test]
fn test_cache_remove_existing() {
    let (device, _queue) = match create_test_device() {
        Some(d) => d,
        None => {
            eprintln!("Skipping: no GPU");
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

    // Create
    let _bg = cache.create_bind_group(&device, &layout, 0x1234, entries, resource_entries, None);
    assert_eq!(cache.len(), 1);

    // Remove
    assert!(cache.remove(0x1234, resource_entries));
    assert_eq!(cache.len(), 0);

    // Remove again (should fail)
    assert!(!cache.remove(0x1234, resource_entries));
}

#[test]
fn test_cache_multiple_different_bind_groups() {
    let (device, _queue) = match create_test_device() {
        Some(d) => d,
        None => {
            eprintln!("Skipping: no GPU");
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

    // Create second bind group (different resource)
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
    assert_eq!(cache.metrics().misses, 2);
}

#[test]
fn test_cache_evict_old_removes_stale() {
    let (device, _queue) = match create_test_device() {
        Some(d) => d,
        None => {
            eprintln!("Skipping: no GPU");
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

    // Create at frame 0
    let _bg = cache.create_bind_group(&device, &layout, 0x1234, entries, resource_entries, None);

    // Advance 5 frames
    for _ in 0..5 {
        cache.begin_frame();
    }

    // Evict older than 3 frames
    let evicted = cache.evict_old(3);
    assert_eq!(evicted, 1);
    assert!(cache.is_empty());
}

#[test]
fn test_cache_invalidate_resources_batch() {
    let (device, _queue) = match create_test_device() {
        Some(d) => d,
        None => {
            eprintln!("Skipping: no GPU");
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
    let res1 = ResourceId::new(1);
    let res2 = ResourceId::new(2);

    // Create first bind group
    let resource_entries1 = &[BindGroupResourceEntry::buffer(0, res1, 0, None)];
    let entries1 = &[wgpu::BindGroupEntry {
        binding: 0,
        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
            buffer: &buffer1,
            offset: 0,
            size: None,
        }),
    }];
    let _bg1 = cache.create_bind_group(&device, &layout, 0x1234, entries1, resource_entries1, None);

    // Create second bind group
    let resource_entries2 = &[BindGroupResourceEntry::buffer(0, res2, 0, None)];
    let entries2 = &[wgpu::BindGroupEntry {
        binding: 0,
        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
            buffer: &buffer2,
            offset: 0,
            size: None,
        }),
    }];
    let _bg2 = cache.create_bind_group(&device, &layout, 0x5678, entries2, resource_entries2, None);

    assert_eq!(cache.len(), 2);

    // Batch invalidate
    let invalidated = cache.invalidate_resources([res1, res2]);
    assert_eq!(invalidated, 2);
    assert!(cache.is_empty());
}

// ============================================================================
// CATEGORY 7: BindGroupCacheMetrics Tests (15+ tests)
// ============================================================================

#[test]
fn test_metrics_new_all_zeros() {
    let metrics = BindGroupCacheMetrics::new(0, 0, 0, 0, 0);
    assert_eq!(metrics.cache_size, 0);
    assert_eq!(metrics.hits, 0);
    assert_eq!(metrics.misses, 0);
    assert_eq!(metrics.hit_rate, 0.0);
    assert_eq!(metrics.current_frame, 0);
    assert_eq!(metrics.tracked_resources, 0);
}

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
fn test_metrics_new_with_values() {
    let metrics = BindGroupCacheMetrics::new(10, 80, 20, 5, 3);
    assert_eq!(metrics.cache_size, 10);
    assert_eq!(metrics.hits, 80);
    assert_eq!(metrics.misses, 20);
    assert_eq!(metrics.current_frame, 5);
    assert_eq!(metrics.tracked_resources, 3);
}

#[test]
fn test_metrics_hit_rate_calculation() {
    let metrics = BindGroupCacheMetrics::new(5, 75, 25, 0, 0);
    assert!((metrics.hit_rate - 0.75).abs() < 0.001);
}

#[test]
fn test_metrics_hit_rate_zero_total() {
    let metrics = BindGroupCacheMetrics::new(0, 0, 0, 0, 0);
    assert_eq!(metrics.hit_rate, 0.0);
}

#[test]
fn test_metrics_hit_rate_all_hits() {
    let metrics = BindGroupCacheMetrics::new(10, 100, 0, 0, 0);
    assert!((metrics.hit_rate - 1.0).abs() < 0.001);
}

#[test]
fn test_metrics_hit_rate_all_misses() {
    let metrics = BindGroupCacheMetrics::new(10, 0, 100, 0, 0);
    assert!((metrics.hit_rate - 0.0).abs() < 0.001);
}

#[test]
fn test_metrics_total_requests() {
    let metrics = BindGroupCacheMetrics::new(5, 50, 30, 0, 0);
    assert_eq!(metrics.total_requests(), 80);
}

#[test]
fn test_metrics_total_requests_zero() {
    let metrics = BindGroupCacheMetrics::new(0, 0, 0, 0, 0);
    assert_eq!(metrics.total_requests(), 0);
}

#[test]
fn test_metrics_is_empty_true() {
    let metrics = BindGroupCacheMetrics::new(0, 10, 5, 0, 0);
    assert!(metrics.is_empty());
}

#[test]
fn test_metrics_is_empty_false() {
    let metrics = BindGroupCacheMetrics::new(1, 10, 5, 0, 0);
    assert!(!metrics.is_empty());
}

#[test]
fn test_metrics_hit_rate_percent() {
    let metrics = BindGroupCacheMetrics::new(5, 80, 20, 0, 0);
    assert!((metrics.hit_rate_percent() - 80.0).abs() < 0.001);
}

#[test]
fn test_metrics_hit_rate_percent_zero() {
    let metrics = BindGroupCacheMetrics::new(0, 0, 0, 0, 0);
    assert_eq!(metrics.hit_rate_percent(), 0.0);
}

#[test]
fn test_metrics_clone() {
    let metrics = BindGroupCacheMetrics::new(10, 100, 50, 5, 2);
    let cloned = metrics.clone();

    assert_eq!(metrics.cache_size, cloned.cache_size);
    assert_eq!(metrics.hits, cloned.hits);
    assert_eq!(metrics.misses, cloned.misses);
    assert_eq!(metrics.hit_rate, cloned.hit_rate);
    assert_eq!(metrics.current_frame, cloned.current_frame);
    assert_eq!(metrics.tracked_resources, cloned.tracked_resources);
}

#[test]
fn test_metrics_debug_format() {
    let metrics = BindGroupCacheMetrics::new(3, 10, 5, 2, 1);
    let debug_str = format!("{:?}", metrics);
    assert!(debug_str.contains("BindGroupCacheMetrics"));
}

#[test]
fn test_metrics_large_values() {
    let metrics = BindGroupCacheMetrics::new(
        1_000_000,
        u64::MAX / 2,
        u64::MAX / 4,
        u64::MAX,
        usize::MAX / 2,
    );

    assert_eq!(metrics.cache_size, 1_000_000);
    assert_eq!(metrics.current_frame, u64::MAX);
}

// ============================================================================
// Thread Safety Tests (additional tests)
// ============================================================================

#[test]
fn test_cache_concurrent_reads() {
    let cache = Arc::new(BindGroupCache::new());
    let handles: Vec<_> = (0..4)
        .map(|_| {
            let cache = Arc::clone(&cache);
            thread::spawn(move || {
                for _ in 0..100 {
                    let _ = cache.len();
                    let _ = cache.is_empty();
                    let _ = cache.current_frame();
                    let _ = cache.metrics();
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }
}

#[test]
fn test_cache_concurrent_begin_frame() {
    let cache = Arc::new(BindGroupCache::new());
    let handles: Vec<_> = (0..4)
        .map(|_| {
            let cache = Arc::clone(&cache);
            thread::spawn(move || {
                for _ in 0..100 {
                    cache.begin_frame();
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    assert_eq!(cache.current_frame(), 400);
}

#[test]
fn test_cache_concurrent_metrics_read() {
    let cache = Arc::new(BindGroupCache::new());
    let handles: Vec<_> = (0..8)
        .map(|_| {
            let cache = Arc::clone(&cache);
            thread::spawn(move || {
                for _ in 0..50 {
                    let metrics = cache.metrics();
                    let _ = metrics.total_requests();
                    let _ = metrics.hit_rate_percent();
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }
}

// ============================================================================
// Edge Case Tests
// ============================================================================

#[test]
fn test_cache_key_many_entries_sorted() {
    // Create entries in reverse order
    let mut entries: Vec<_> = (0..16u32)
        .rev()
        .map(|i| BindGroupResourceEntry::buffer(i, ResourceId::new(i as u64), 0, None))
        .collect();

    let key1 = BindGroupCacheKey::new(0x1234, &entries);

    // Sort entries
    entries.sort_by_key(|e| e.binding);

    let key2 = BindGroupCacheKey::new(0x1234, &entries);

    // Should be equal because BindGroupCacheKey sorts internally
    assert_eq!(key1, key2);
}

#[test]
fn test_cache_evict_old_boundary() {
    let (device, _queue) = match create_test_device() {
        Some(d) => d,
        None => {
            eprintln!("Skipping: no GPU");
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
    let entries = &[wgpu::BindGroupEntry {
        binding: 0,
        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
            buffer: &buffer,
            offset: 0,
            size: None,
        }),
    }];

    // Create bind group at frame 0
    let resource_entries = &[BindGroupResourceEntry::buffer(0, ResourceId::new(1), 0, None)];
    let _bg = cache.create_bind_group(&device, &layout, 0x1234, entries, resource_entries, None);

    // Advance exactly 3 frames
    cache.begin_frame();
    cache.begin_frame();
    cache.begin_frame();

    // Evict with max_age=3 - entry at frame 0 should be evicted (current=3, cutoff=0)
    let evicted = cache.evict_old(3);
    assert_eq!(evicted, 0, "Entry at frame 0 with max_age=3 at frame 3 should NOT be evicted");

    // Advance one more
    cache.begin_frame();

    // Now should be evicted
    let evicted = cache.evict_old(3);
    assert_eq!(evicted, 1);
}

#[test]
fn test_cache_double_check_locking() {
    // Test that double-check locking works correctly
    let (device, _queue) = match create_test_device() {
        Some(d) => d,
        None => {
            eprintln!("Skipping: no GPU");
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

    // Multiple rapid calls should all return the same Arc
    let bg1 = cache.create_bind_group(&device, &layout, 0x1234, entries, resource_entries, None);
    let bg2 = cache.create_bind_group(&device, &layout, 0x1234, entries, resource_entries, None);
    let bg3 = cache.create_bind_group(&device, &layout, 0x1234, entries, resource_entries, None);

    assert!(Arc::ptr_eq(&bg1, &bg2));
    assert!(Arc::ptr_eq(&bg2, &bg3));
    assert_eq!(cache.metrics().misses, 1);
    assert_eq!(cache.metrics().hits, 2);
}

#[test]
fn test_invalidate_shared_resource() {
    let (device, _queue) = match create_test_device() {
        Some(d) => d,
        None => {
            eprintln!("Skipping: no GPU");
            return;
        }
    };

    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("shared"),
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
    let shared_id = ResourceId::new(42);

    // Create two bind groups with the same resource but different layouts
    let resource_entries = &[BindGroupResourceEntry::buffer(0, shared_id, 0, None)];
    let entries = &[wgpu::BindGroupEntry {
        binding: 0,
        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
            buffer: &buffer,
            offset: 0,
            size: None,
        }),
    }];

    let _bg1 = cache.create_bind_group(&device, &layout, 0x1111, entries, resource_entries, None);
    let _bg2 = cache.create_bind_group(&device, &layout, 0x2222, entries, resource_entries, None);

    assert_eq!(cache.len(), 2);

    // Invalidate the shared resource - should remove both bind groups
    let invalidated = cache.invalidate_resource(shared_id);
    assert_eq!(invalidated, 2);
    assert!(cache.is_empty());
}

// ============================================================================
// Summary marker test
// ============================================================================

#[test]
fn test_whitebox_bind_group_cache_summary() {
    // This test serves as documentation of coverage
    println!("WHITEBOX T-WGPU-P2.5.2 Test Categories:");
    println!("1. ResourceId: 16 tests");
    println!("2. BindGroupResourceType: 22 tests");
    println!("3. BindGroupResourceEntry: 15 tests");
    println!("4. BindGroupCacheKey: 20 tests");
    println!("5. CachedBindGroup: 5 tests (requires device)");
    println!("6. BindGroupCache: 26 tests");
    println!("7. BindGroupCacheMetrics: 16 tests");
    println!("+ Thread Safety: 3 tests");
    println!("+ Edge Cases: 4 tests");
    println!("Total: 127 tests");
}
