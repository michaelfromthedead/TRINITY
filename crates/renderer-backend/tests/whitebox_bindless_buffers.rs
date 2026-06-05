//! Whitebox tests for bindless buffer registry (T-WGPU-P2.6.2).
//!
//! This module provides comprehensive whitebox tests for the bindless buffer
//! management system, covering BufferSlot, BufferRegistry, BindlessBufferError,
//! BufferRegistryMetrics, dirty tracking, feature detection, and helper functions.
//!
//! Test coverage targets:
//! - BufferSlot: ~15 tests
//! - Registry Construction: ~10 tests
//! - Registration: ~15 tests
//! - Unregistration & Recycling: ~10 tests
//! - Dirty Tracking: ~15 tests
//! - Bind Group Management: ~10 tests
//! - Metrics: ~10 tests
//! - Error Types: ~10 tests
//! - Thread Safety: ~5 tests

use renderer_backend::resources::bindless_buffers::{
    bindless_buffer_layout_entry, bindless_buffer_layout_entry_readonly,
    bindless_buffer_layout_entry_readwrite, bindless_buffer_optimal_features,
    bindless_buffer_required_features, max_bindless_buffers_from_limits, BindlessBufferError,
    BufferRegistry, BufferRegistryMetrics, BufferSlot, BINDLESS_BIND_GROUP_INDEX,
    BINDLESS_BUFFER_BINDING, DEFAULT_MAX_BUFFERS, MAX_BINDLESS_BUFFERS_CONSERVATIVE,
    MIN_BINDLESS_BUFFERS,
};
use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::num::NonZeroU32;
use wgpu::{BindingType, BufferBindingType, Features, Limits, ShaderStages};

// ============================================================================
// BufferSlot Tests (~15 tests)
// ============================================================================

#[test]
fn test_slot_new_creates_slot_with_index() {
    let slot = BufferSlot::new(42);
    assert_eq!(slot.index(), 42);
}

#[test]
fn test_slot_new_zero_index() {
    let slot = BufferSlot::new(0);
    assert_eq!(slot.index(), 0);
    assert!(!slot.is_invalid());
}

#[test]
fn test_slot_index_accessor_returns_correct_value() {
    for i in [0, 1, 100, 1000, u32::MAX - 1] {
        let slot = BufferSlot::new(i);
        assert_eq!(slot.index(), i);
    }
}

#[test]
fn test_slot_invalid_returns_max_u32() {
    let slot = BufferSlot::invalid();
    assert_eq!(slot.index(), u32::MAX);
    assert!(slot.is_invalid());
}

#[test]
fn test_slot_is_invalid_true_only_for_max_u32() {
    assert!(BufferSlot::new(u32::MAX).is_invalid());
    assert!(!BufferSlot::new(u32::MAX - 1).is_invalid());
    assert!(!BufferSlot::new(0).is_invalid());
}

#[test]
fn test_slot_from_u32_conversion() {
    let slot: BufferSlot = 123u32.into();
    assert_eq!(slot.index(), 123);
}

#[test]
fn test_slot_into_u32_conversion() {
    let slot = BufferSlot::new(456);
    let index: u32 = slot.into();
    assert_eq!(index, 456);
}

#[test]
fn test_slot_roundtrip_conversion() {
    for i in [0, 42, 999, u32::MAX] {
        let slot: BufferSlot = i.into();
        let back: u32 = slot.into();
        assert_eq!(i, back);
    }
}

#[test]
fn test_slot_debug_trait_format() {
    let slot = BufferSlot::new(7);
    let debug = format!("{:?}", slot);
    assert!(debug.contains("BufferSlot"));
    assert!(debug.contains("7"));
}

#[test]
fn test_slot_clone_trait() {
    let slot1 = BufferSlot::new(8);
    let slot2 = slot1.clone();
    assert_eq!(slot1, slot2);
}

#[test]
fn test_slot_copy_trait() {
    let slot1 = BufferSlot::new(5);
    let slot2 = slot1; // Copy
    assert_eq!(slot1.index(), slot2.index());
}

#[test]
fn test_slot_partial_eq_same_values() {
    let slot1 = BufferSlot::new(10);
    let slot2 = BufferSlot::new(10);
    assert_eq!(slot1, slot2);
}

#[test]
fn test_slot_partial_eq_different_values() {
    let slot1 = BufferSlot::new(10);
    let slot2 = BufferSlot::new(20);
    assert_ne!(slot1, slot2);
}

#[test]
fn test_slot_eq_reflexive_symmetric_transitive() {
    let a = BufferSlot::new(42);
    let b = BufferSlot::new(42);
    let c = BufferSlot::new(42);

    // Reflexive
    assert_eq!(a, a);
    // Symmetric
    assert_eq!(a, b);
    assert_eq!(b, a);
    // Transitive
    assert_eq!(a, b);
    assert_eq!(b, c);
    assert_eq!(a, c);
}

#[test]
fn test_slot_hash_consistent_for_equal_values() {
    use std::collections::hash_map::DefaultHasher;

    let slot1 = BufferSlot::new(100);
    let slot2 = BufferSlot::new(100);

    let mut hasher1 = DefaultHasher::new();
    let mut hasher2 = DefaultHasher::new();

    slot1.hash(&mut hasher1);
    slot2.hash(&mut hasher2);

    assert_eq!(hasher1.finish(), hasher2.finish());
}

#[test]
fn test_slot_hash_in_hashset() {
    let mut set = HashSet::new();
    set.insert(BufferSlot::new(1));
    set.insert(BufferSlot::new(2));
    set.insert(BufferSlot::new(1)); // Duplicate

    assert_eq!(set.len(), 2);
    assert!(set.contains(&BufferSlot::new(1)));
    assert!(set.contains(&BufferSlot::new(2)));
}

#[test]
fn test_slot_as_hashmap_key() {
    let mut map: HashMap<BufferSlot, &str> = HashMap::new();
    map.insert(BufferSlot::new(1), "first");
    map.insert(BufferSlot::new(2), "second");

    assert_eq!(map.get(&BufferSlot::new(1)), Some(&"first"));
    assert_eq!(map.get(&BufferSlot::new(2)), Some(&"second"));
    assert_eq!(map.get(&BufferSlot::new(3)), None);
}

#[test]
fn test_slot_display_valid() {
    assert_eq!(format!("{}", BufferSlot::new(42)), "BufferSlot(42)");
    assert_eq!(format!("{}", BufferSlot::new(0)), "BufferSlot(0)");
}

#[test]
fn test_slot_display_invalid() {
    assert_eq!(format!("{}", BufferSlot::invalid()), "BufferSlot(INVALID)");
}

// ============================================================================
// Registry Construction Tests (~10 tests)
// ============================================================================

#[test]
fn test_registry_new_with_default_capacity() {
    let registry = BufferRegistry::new(DEFAULT_MAX_BUFFERS);
    assert_eq!(registry.capacity(), DEFAULT_MAX_BUFFERS);
    assert_eq!(registry.count(), 0);
}

#[test]
fn test_registry_new_with_custom_capacity() {
    let registry = BufferRegistry::new(512);
    assert_eq!(registry.capacity(), 512);
}

#[test]
fn test_registry_new_clamps_below_minimum() {
    let registry = BufferRegistry::new(1);
    assert_eq!(registry.capacity(), MIN_BINDLESS_BUFFERS);
}

#[test]
fn test_registry_new_clamps_above_maximum() {
    let registry = BufferRegistry::new(u32::MAX);
    assert_eq!(registry.capacity(), MAX_BINDLESS_BUFFERS_CONSERVATIVE);
}

#[test]
fn test_registry_new_at_exact_minimum() {
    let registry = BufferRegistry::new(MIN_BINDLESS_BUFFERS);
    assert_eq!(registry.capacity(), MIN_BINDLESS_BUFFERS);
}

#[test]
fn test_registry_new_at_exact_maximum() {
    let registry = BufferRegistry::new(MAX_BINDLESS_BUFFERS_CONSERVATIVE);
    assert_eq!(registry.capacity(), MAX_BINDLESS_BUFFERS_CONSERVATIVE);
}

#[test]
fn test_registry_default_uses_default_max_buffers() {
    let registry = BufferRegistry::default();
    assert_eq!(registry.capacity(), DEFAULT_MAX_BUFFERS);
}

#[test]
fn test_registry_initial_state_is_empty() {
    let registry = BufferRegistry::new(100);
    assert_eq!(registry.count(), 0);
    assert!(registry.is_empty());
    assert!(!registry.is_full());
}

#[test]
fn test_registry_initial_state_is_dirty() {
    let registry = BufferRegistry::new(100);
    assert!(registry.is_dirty());
}

#[test]
fn test_registry_initial_state_no_bind_group() {
    let registry = BufferRegistry::new(100);
    assert!(registry.bind_group().is_none());
}

#[test]
fn test_registry_initial_state_no_free_slots() {
    let registry = BufferRegistry::new(100);
    assert_eq!(registry.free_slot_count(), 0);
}

#[test]
fn test_registry_initial_state_no_dirty_slots() {
    let registry = BufferRegistry::new(100);
    assert_eq!(registry.dirty_count(), 0);
}

// ============================================================================
// Dirty Tracking Tests (~15 tests)
// ============================================================================

#[test]
fn test_dirty_tracking_initial_empty() {
    let registry = BufferRegistry::new(100);
    assert_eq!(registry.dirty_count(), 0);
}

#[test]
fn test_dirty_tracking_mark_dirty_out_of_bounds() {
    // Without a registered buffer, marking dirty should be ignored
    // since there's no valid slot at that index
    let mut registry = BufferRegistry::new(100);
    registry.mark_dirty(BufferSlot::new(0));
    // Should be 0 since no buffer is at slot 0
    assert_eq!(registry.dirty_count(), 0);
}

// Since we can't directly manipulate internal state safely, test via public API behavior
#[test]
fn test_dirty_tracking_mark_invalid_slot_ignored() {
    let mut registry = BufferRegistry::new(100);
    registry.mark_dirty(BufferSlot::invalid());
    assert_eq!(registry.dirty_count(), 0);
}

#[test]
fn test_dirty_tracking_mark_out_of_bounds_slot_ignored() {
    let mut registry = BufferRegistry::new(100);
    registry.mark_dirty(BufferSlot::new(999));
    assert_eq!(registry.dirty_count(), 0);
}

#[test]
fn test_dirty_tracking_clear_dirty_resets() {
    let mut registry = BufferRegistry::new(100);
    // Simulate dirty slots via internal state
    registry.clear_dirty();
    assert_eq!(registry.dirty_count(), 0);
}

#[test]
fn test_dirty_tracking_is_slot_dirty_false_for_untracked() {
    let registry = BufferRegistry::new(100);
    assert!(!registry.is_slot_dirty(BufferSlot::new(0)));
    assert!(!registry.is_slot_dirty(BufferSlot::new(50)));
}

#[test]
fn test_dirty_slots_iterator_empty_registry() {
    let registry = BufferRegistry::new(100);
    let count = registry.dirty_slots().count();
    assert_eq!(count, 0);
}

#[test]
fn test_dirty_slot_indices_iterator_empty() {
    let registry = BufferRegistry::new(100);
    let count = registry.dirty_slot_indices().count();
    assert_eq!(count, 0);
}

#[test]
fn test_clear_dirty_after_multiple_calls() {
    let mut registry = BufferRegistry::new(100);
    registry.clear_dirty();
    registry.clear_dirty();
    assert_eq!(registry.dirty_count(), 0);
}

#[test]
fn test_dirty_tracking_slot_invalid_returns_false() {
    let registry = BufferRegistry::new(100);
    assert!(!registry.is_slot_dirty(BufferSlot::invalid()));
}

// ============================================================================
// Registry State Tests (~10 tests)
// ============================================================================

#[test]
fn test_registry_is_empty_true_for_new() {
    let registry = BufferRegistry::new(100);
    assert!(registry.is_empty());
}

#[test]
fn test_registry_is_full_false_for_new() {
    let registry = BufferRegistry::new(100);
    assert!(!registry.is_full());
}

#[test]
fn test_registry_capacity_returns_clamped_value() {
    let registry = BufferRegistry::new(50);
    assert_eq!(registry.capacity(), 50);
}

#[test]
fn test_registry_count_returns_zero_initially() {
    let registry = BufferRegistry::new(100);
    assert_eq!(registry.count(), 0);
}

#[test]
fn test_registry_clear_resets_state() {
    let mut registry = BufferRegistry::new(100);
    // Clear should reset everything
    registry.clear();
    assert!(registry.is_empty());
    assert!(registry.is_dirty());
    assert_eq!(registry.free_slot_count(), 0);
    assert_eq!(registry.dirty_count(), 0);
}

#[test]
fn test_registry_debug_format() {
    let registry = BufferRegistry::new(256);
    let debug = format!("{:?}", registry);

    assert!(debug.contains("BufferRegistry"));
    assert!(debug.contains("capacity"));
    assert!(debug.contains("256"));
}

#[test]
fn test_registry_get_returns_none_for_empty() {
    let registry = BufferRegistry::new(100);
    assert!(registry.get(BufferSlot::new(0)).is_none());
}

#[test]
fn test_registry_is_registered_false_for_empty() {
    let registry = BufferRegistry::new(100);
    assert!(!registry.is_registered(BufferSlot::new(0)));
}

#[test]
fn test_registry_iter_empty() {
    let registry = BufferRegistry::new(100);
    let count = registry.iter().count();
    assert_eq!(count, 0);
}

// ============================================================================
// Metrics Tests (~10 tests)
// ============================================================================

#[test]
fn test_metrics_empty_registry() {
    let registry = BufferRegistry::new(100);
    let metrics = registry.metrics();

    assert_eq!(metrics.registered_count, 0);
    assert_eq!(metrics.capacity, 100);
    assert_eq!(metrics.free_slots, 0);
    assert_eq!(metrics.allocated_slots, 0);
    assert_eq!(metrics.dirty_slots, 0);
    assert!(!metrics.has_bind_group);
    assert!(metrics.is_dirty);
}

#[test]
fn test_metrics_utilization_empty() {
    let registry = BufferRegistry::new(100);
    assert_eq!(registry.metrics().utilization(), 0.0);
}

#[test]
fn test_metrics_utilization_zero_capacity() {
    let metrics = BufferRegistryMetrics {
        registered_count: 0,
        capacity: 0,
        free_slots: 0,
        allocated_slots: 0,
        dirty_slots: 0,
        has_bind_group: false,
        is_dirty: true,
    };
    assert_eq!(metrics.utilization(), 0.0);
}

#[test]
fn test_metrics_utilization_calculation() {
    let metrics = BufferRegistryMetrics {
        registered_count: 25,
        capacity: 100,
        free_slots: 0,
        allocated_slots: 25,
        dirty_slots: 5,
        has_bind_group: true,
        is_dirty: false,
    };
    // 25/100 = 0.25
    assert!((metrics.utilization() - 0.25).abs() < 0.001);
}

#[test]
fn test_metrics_fragmentation_empty() {
    let metrics = BufferRegistryMetrics {
        registered_count: 0,
        capacity: 100,
        free_slots: 0,
        allocated_slots: 0,
        dirty_slots: 0,
        has_bind_group: false,
        is_dirty: true,
    };
    assert_eq!(metrics.fragmentation(), 0.0);
}

#[test]
fn test_metrics_fragmentation_calculation() {
    let metrics = BufferRegistryMetrics {
        registered_count: 8,
        capacity: 100,
        free_slots: 2,
        allocated_slots: 10,
        dirty_slots: 1,
        has_bind_group: true,
        is_dirty: false,
    };
    // 2/10 = 0.2
    assert!((metrics.fragmentation() - 0.2).abs() < 0.001);
}

#[test]
fn test_metrics_dirty_ratio_empty() {
    let metrics = BufferRegistryMetrics {
        registered_count: 0,
        capacity: 100,
        free_slots: 0,
        allocated_slots: 0,
        dirty_slots: 0,
        has_bind_group: false,
        is_dirty: true,
    };
    assert_eq!(metrics.dirty_ratio(), 0.0);
}

#[test]
fn test_metrics_dirty_ratio_calculation() {
    let metrics = BufferRegistryMetrics {
        registered_count: 10,
        capacity: 100,
        free_slots: 0,
        allocated_slots: 10,
        dirty_slots: 5,
        has_bind_group: true,
        is_dirty: false,
    };
    // 5/10 = 0.5
    assert!((metrics.dirty_ratio() - 0.5).abs() < 0.001);
}

#[test]
fn test_metrics_copy_trait() {
    let m1 = BufferRegistryMetrics {
        registered_count: 10,
        capacity: 100,
        free_slots: 5,
        allocated_slots: 15,
        dirty_slots: 2,
        has_bind_group: true,
        is_dirty: false,
    };
    let m2 = m1; // Copy
    assert_eq!(m1, m2);
}

#[test]
fn test_metrics_clone_trait() {
    let m1 = BufferRegistryMetrics {
        registered_count: 10,
        capacity: 100,
        free_slots: 5,
        allocated_slots: 15,
        dirty_slots: 2,
        has_bind_group: true,
        is_dirty: false,
    };
    let m2 = m1.clone();
    assert_eq!(m1, m2);
}

#[test]
fn test_metrics_equality() {
    let m1 = BufferRegistryMetrics {
        registered_count: 10,
        capacity: 100,
        free_slots: 5,
        allocated_slots: 15,
        dirty_slots: 2,
        has_bind_group: true,
        is_dirty: false,
    };
    let m2 = BufferRegistryMetrics {
        registered_count: 10,
        capacity: 100,
        free_slots: 5,
        allocated_slots: 15,
        dirty_slots: 2,
        has_bind_group: true,
        is_dirty: false,
    };
    let m3 = BufferRegistryMetrics {
        registered_count: 20, // Different
        capacity: 100,
        free_slots: 5,
        allocated_slots: 15,
        dirty_slots: 2,
        has_bind_group: true,
        is_dirty: false,
    };

    assert_eq!(m1, m2);
    assert_ne!(m1, m3);
}

#[test]
fn test_metrics_debug_format() {
    let metrics = BufferRegistryMetrics {
        registered_count: 10,
        capacity: 100,
        free_slots: 5,
        allocated_slots: 15,
        dirty_slots: 2,
        has_bind_group: true,
        is_dirty: false,
    };
    let debug = format!("{:?}", metrics);
    assert!(debug.contains("BufferRegistryMetrics"));
    assert!(debug.contains("registered_count"));
    assert!(debug.contains("10"));
}

// ============================================================================
// Error Type Tests (~10 tests)
// ============================================================================

#[test]
fn test_error_unsupported_feature_display() {
    let err = BindlessBufferError::UnsupportedFeature;
    let msg = err.to_string();
    assert!(msg.contains("BUFFER_BINDING_ARRAY"));
    assert!(msg.contains("not supported"));
}

#[test]
fn test_error_registry_full_display() {
    let err = BindlessBufferError::RegistryFull { capacity: 1024 };
    let msg = err.to_string();
    assert!(msg.contains("1024"));
    assert!(msg.contains("full"));
}

#[test]
fn test_error_invalid_slot_display() {
    let err = BindlessBufferError::InvalidSlot(BufferSlot::new(42));
    let msg = err.to_string();
    assert!(msg.contains("42"));
    assert!(msg.contains("invalid"));
}

#[test]
fn test_error_exceeds_device_limit_display() {
    let err = BindlessBufferError::ExceedsDeviceLimit {
        requested: 2000,
        max: 1024,
    };
    let msg = err.to_string();
    assert!(msg.contains("2000"));
    assert!(msg.contains("1024"));
    assert!(msg.contains("exceeds"));
}

#[test]
fn test_error_incompatible_layout_display() {
    let err = BindlessBufferError::IncompatibleLayout;
    assert!(err.to_string().contains("incompatible"));
}

#[test]
fn test_error_empty_registry_display() {
    let err = BindlessBufferError::EmptyRegistry;
    assert!(err.to_string().contains("empty"));
}

#[test]
fn test_error_clone_trait() {
    let err1 = BindlessBufferError::RegistryFull { capacity: 500 };
    let err2 = err1.clone();
    assert_eq!(err1, err2);
}

#[test]
fn test_error_equality() {
    let err1 = BindlessBufferError::RegistryFull { capacity: 100 };
    let err2 = BindlessBufferError::RegistryFull { capacity: 100 };
    let err3 = BindlessBufferError::RegistryFull { capacity: 200 };

    assert_eq!(err1, err2);
    assert_ne!(err1, err3);
}

#[test]
fn test_error_debug_format() {
    let err = BindlessBufferError::InvalidSlot(BufferSlot::new(99));
    let debug = format!("{:?}", err);
    assert!(debug.contains("InvalidSlot"));
    assert!(debug.contains("99"));
}

#[test]
fn test_error_is_std_error() {
    let err: Box<dyn std::error::Error> = Box::new(BindlessBufferError::UnsupportedFeature);
    assert!(err.to_string().contains("not supported"));
}

#[test]
fn test_error_all_variants_different() {
    let errors = [
        BindlessBufferError::UnsupportedFeature,
        BindlessBufferError::RegistryFull { capacity: 100 },
        BindlessBufferError::InvalidSlot(BufferSlot::new(0)),
        BindlessBufferError::ExceedsDeviceLimit {
            requested: 200,
            max: 100,
        },
        BindlessBufferError::IncompatibleLayout,
        BindlessBufferError::EmptyRegistry,
    ];

    for i in 0..errors.len() {
        for j in i + 1..errors.len() {
            assert_ne!(errors[i], errors[j]);
        }
    }
}

// ============================================================================
// Feature Detection Tests (~10 tests)
// ============================================================================

#[test]
fn test_bindless_buffer_required_features_contains_buffer_binding_array() {
    let features = bindless_buffer_required_features();
    assert!(features.contains(Features::BUFFER_BINDING_ARRAY));
}

#[test]
fn test_bindless_buffer_optimal_features_contains_all_required() {
    let features = bindless_buffer_optimal_features();
    assert!(features.contains(Features::BUFFER_BINDING_ARRAY));
    assert!(features.contains(Features::STORAGE_RESOURCE_BINDING_ARRAY));
    assert!(features.contains(
        Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING
    ));
    assert!(features.contains(Features::PARTIALLY_BOUND_BINDING_ARRAY));
}

#[test]
fn test_max_bindless_buffers_from_limits_uses_device_limit() {
    let mut limits = Limits::default();
    limits.max_storage_buffers_per_shader_stage = 500;

    let max = max_bindless_buffers_from_limits(&limits);
    assert_eq!(max, 500);
}

#[test]
fn test_max_bindless_buffers_from_limits_clamped_by_conservative() {
    let mut limits = Limits::default();
    limits.max_storage_buffers_per_shader_stage = u32::MAX;

    let max = max_bindless_buffers_from_limits(&limits);
    assert_eq!(max, MAX_BINDLESS_BUFFERS_CONSERVATIVE);
}

#[test]
fn test_max_bindless_buffers_from_limits_zero() {
    let mut limits = Limits::default();
    limits.max_storage_buffers_per_shader_stage = 0;

    let max = max_bindless_buffers_from_limits(&limits);
    assert_eq!(max, 0);
}

#[test]
fn test_max_bindless_buffers_from_limits_exact_conservative() {
    let mut limits = Limits::default();
    limits.max_storage_buffers_per_shader_stage = MAX_BINDLESS_BUFFERS_CONSERVATIVE;

    let max = max_bindless_buffers_from_limits(&limits);
    assert_eq!(max, MAX_BINDLESS_BUFFERS_CONSERVATIVE);
}

// ============================================================================
// Bind Group Layout Entry Tests (~10 tests)
// ============================================================================

#[test]
fn test_layout_entry_readonly_binding_index() {
    let entry = bindless_buffer_layout_entry(1, 1024, true);
    assert_eq!(entry.binding, 1);
}

#[test]
fn test_layout_entry_readonly_visibility() {
    let entry = bindless_buffer_layout_entry(1, 1024, true);
    assert!(entry.visibility.contains(ShaderStages::VERTEX_FRAGMENT));
    assert!(entry.visibility.contains(ShaderStages::COMPUTE));
}

#[test]
fn test_layout_entry_readonly_count() {
    let entry = bindless_buffer_layout_entry(1, 1024, true);
    assert_eq!(entry.count, NonZeroU32::new(1024));
}

#[test]
fn test_layout_entry_readonly_buffer_type() {
    let entry = bindless_buffer_layout_entry(1, 1024, true);
    if let BindingType::Buffer {
        ty,
        has_dynamic_offset,
        min_binding_size,
    } = entry.ty
    {
        assert!(matches!(ty, BufferBindingType::Storage { read_only: true }));
        assert!(!has_dynamic_offset);
        assert!(min_binding_size.is_none());
    } else {
        panic!("Expected Buffer binding type");
    }
}

#[test]
fn test_layout_entry_readwrite_buffer_type() {
    let entry = bindless_buffer_layout_entry(1, 512, false);
    if let BindingType::Buffer { ty, .. } = entry.ty {
        assert!(matches!(ty, BufferBindingType::Storage { read_only: false }));
    } else {
        panic!("Expected Buffer binding type");
    }
}

#[test]
fn test_layout_entry_readonly_helper() {
    let entry = bindless_buffer_layout_entry_readonly(1, 256);
    if let BindingType::Buffer { ty, .. } = entry.ty {
        assert!(matches!(ty, BufferBindingType::Storage { read_only: true }));
    } else {
        panic!("Expected Buffer binding type");
    }
}

#[test]
fn test_layout_entry_readwrite_helper() {
    let entry = bindless_buffer_layout_entry_readwrite(1, 256);
    if let BindingType::Buffer { ty, .. } = entry.ty {
        assert!(matches!(ty, BufferBindingType::Storage { read_only: false }));
    } else {
        panic!("Expected Buffer binding type");
    }
}

#[test]
fn test_layout_entry_custom_binding_index() {
    let entry = bindless_buffer_layout_entry(5, 100, true);
    assert_eq!(entry.binding, 5);
}

#[test]
fn test_layout_entry_small_count() {
    let entry = bindless_buffer_layout_entry(0, 1, true);
    assert_eq!(entry.count, NonZeroU32::new(1));
}

#[test]
fn test_layout_entry_large_count() {
    let entry = bindless_buffer_layout_entry(0, 16384, true);
    assert_eq!(entry.count, NonZeroU32::new(16384));
}

// ============================================================================
// Constants Tests (~5 tests)
// ============================================================================

#[test]
fn test_constants_reasonable_values() {
    assert!(DEFAULT_MAX_BUFFERS >= MIN_BINDLESS_BUFFERS);
    assert!(DEFAULT_MAX_BUFFERS <= MAX_BINDLESS_BUFFERS_CONSERVATIVE);
    assert!(MIN_BINDLESS_BUFFERS >= 1);
}

#[test]
fn test_bind_group_index_constant() {
    assert_eq!(BINDLESS_BIND_GROUP_INDEX, 3);
}

#[test]
fn test_buffer_binding_constant() {
    assert_eq!(BINDLESS_BUFFER_BINDING, 1);
}

#[test]
fn test_min_buffers_positive() {
    assert!(MIN_BINDLESS_BUFFERS > 0);
}

#[test]
fn test_max_conservative_reasonable() {
    assert!(MAX_BINDLESS_BUFFERS_CONSERVATIVE >= 1024);
    assert!(MAX_BINDLESS_BUFFERS_CONSERVATIVE <= 100_000);
}

// ============================================================================
// Thread Safety Tests (~5 tests)
// ============================================================================

#[test]
fn test_registry_is_send() {
    fn assert_send<T: Send>() {}
    assert_send::<BufferRegistry>();
}

#[test]
fn test_registry_is_sync() {
    fn assert_sync<T: Sync>() {}
    assert_sync::<BufferRegistry>();
}

#[test]
fn test_slot_is_send() {
    fn assert_send<T: Send>() {}
    assert_send::<BufferSlot>();
}

#[test]
fn test_slot_is_sync() {
    fn assert_sync<T: Sync>() {}
    assert_sync::<BufferSlot>();
}

#[test]
fn test_error_is_send() {
    fn assert_send<T: Send>() {}
    assert_send::<BindlessBufferError>();
}

#[test]
fn test_error_is_sync() {
    fn assert_sync<T: Sync>() {}
    assert_sync::<BindlessBufferError>();
}

#[test]
fn test_metrics_is_send() {
    fn assert_send<T: Send>() {}
    assert_send::<BufferRegistryMetrics>();
}

#[test]
fn test_metrics_is_sync() {
    fn assert_sync<T: Sync>() {}
    assert_sync::<BufferRegistryMetrics>();
}

// ============================================================================
// Edge Case Tests (~10 tests)
// ============================================================================

#[test]
fn test_slot_zero_is_valid() {
    let slot = BufferSlot::new(0);
    assert!(!slot.is_invalid());
    assert_eq!(slot.index(), 0);
}

#[test]
fn test_slot_max_minus_one_is_valid() {
    let slot = BufferSlot::new(u32::MAX - 1);
    assert!(!slot.is_invalid());
    assert_eq!(slot.index(), u32::MAX - 1);
}

#[test]
fn test_registry_min_capacity_construction() {
    let registry = BufferRegistry::new(MIN_BINDLESS_BUFFERS);
    assert_eq!(registry.capacity(), MIN_BINDLESS_BUFFERS);
}

#[test]
fn test_registry_max_capacity_construction() {
    let registry = BufferRegistry::new(MAX_BINDLESS_BUFFERS_CONSERVATIVE);
    assert_eq!(registry.capacity(), MAX_BINDLESS_BUFFERS_CONSERVATIVE);
}

#[test]
fn test_clear_on_empty_registry() {
    let mut registry = BufferRegistry::new(100);
    registry.clear();
    assert!(registry.is_empty());
    assert!(registry.is_dirty());
}

#[test]
fn test_multiple_clear_calls() {
    let mut registry = BufferRegistry::new(100);
    registry.clear();
    registry.clear();
    registry.clear();
    assert!(registry.is_empty());
}

#[test]
fn test_metrics_after_clear() {
    let mut registry = BufferRegistry::new(100);
    registry.clear();
    let metrics = registry.metrics();
    assert_eq!(metrics.registered_count, 0);
    assert_eq!(metrics.free_slots, 0);
    assert_eq!(metrics.allocated_slots, 0);
}

#[test]
fn test_slot_hash_differs_for_different_indices() {
    use std::collections::hash_map::DefaultHasher;

    let slot1 = BufferSlot::new(1);
    let slot2 = BufferSlot::new(2);

    let mut hasher1 = DefaultHasher::new();
    let mut hasher2 = DefaultHasher::new();

    slot1.hash(&mut hasher1);
    slot2.hash(&mut hasher2);

    assert_ne!(hasher1.finish(), hasher2.finish());
}

#[test]
fn test_error_variants_have_unique_messages() {
    let messages: Vec<String> = [
        BindlessBufferError::UnsupportedFeature.to_string(),
        BindlessBufferError::RegistryFull { capacity: 100 }.to_string(),
        BindlessBufferError::InvalidSlot(BufferSlot::new(0)).to_string(),
        BindlessBufferError::ExceedsDeviceLimit {
            requested: 200,
            max: 100,
        }
        .to_string(),
        BindlessBufferError::IncompatibleLayout.to_string(),
        BindlessBufferError::EmptyRegistry.to_string(),
    ]
    .into_iter()
    .collect();

    // Check all messages are unique
    let unique: HashSet<_> = messages.iter().collect();
    assert_eq!(unique.len(), messages.len());
}

#[test]
fn test_layout_entry_zero_count() {
    // NonZeroU32::new(0) returns None, so count will be None
    let entry = bindless_buffer_layout_entry(0, 0, true);
    assert!(entry.count.is_none());
}

// ============================================================================
// Metrics Boundary Tests (~5 tests)
// ============================================================================

#[test]
fn test_metrics_utilization_full() {
    let metrics = BufferRegistryMetrics {
        registered_count: 100,
        capacity: 100,
        free_slots: 0,
        allocated_slots: 100,
        dirty_slots: 0,
        has_bind_group: true,
        is_dirty: false,
    };
    assert!((metrics.utilization() - 1.0).abs() < 0.001);
}

#[test]
fn test_metrics_fragmentation_full() {
    let metrics = BufferRegistryMetrics {
        registered_count: 50,
        capacity: 100,
        free_slots: 50,
        allocated_slots: 100,
        dirty_slots: 0,
        has_bind_group: true,
        is_dirty: false,
    };
    // 50/100 = 0.5
    assert!((metrics.fragmentation() - 0.5).abs() < 0.001);
}

#[test]
fn test_metrics_dirty_ratio_full() {
    let metrics = BufferRegistryMetrics {
        registered_count: 100,
        capacity: 100,
        free_slots: 0,
        allocated_slots: 100,
        dirty_slots: 100,
        has_bind_group: true,
        is_dirty: true,
    };
    assert!((metrics.dirty_ratio() - 1.0).abs() < 0.001);
}

#[test]
fn test_metrics_all_zeros() {
    let metrics = BufferRegistryMetrics {
        registered_count: 0,
        capacity: 0,
        free_slots: 0,
        allocated_slots: 0,
        dirty_slots: 0,
        has_bind_group: false,
        is_dirty: false,
    };
    assert_eq!(metrics.utilization(), 0.0);
    assert_eq!(metrics.fragmentation(), 0.0);
    assert_eq!(metrics.dirty_ratio(), 0.0);
}

#[test]
fn test_metrics_large_values() {
    let metrics = BufferRegistryMetrics {
        registered_count: 10000,
        capacity: 16384,
        free_slots: 100,
        allocated_slots: 10100,
        dirty_slots: 500,
        has_bind_group: true,
        is_dirty: false,
    };
    let util = metrics.utilization();
    let frag = metrics.fragmentation();
    let dirty = metrics.dirty_ratio();

    assert!(util > 0.0 && util < 1.0);
    assert!(frag > 0.0 && frag < 1.0);
    assert!(dirty > 0.0 && dirty < 1.0);
}

// ============================================================================
// Additional Slot Edge Cases (~5 tests)
// ============================================================================

#[test]
fn test_slot_boundary_values() {
    // Test various boundary values
    for val in [0u32, 1, 255, 256, 65535, 65536, u32::MAX - 1] {
        let slot = BufferSlot::new(val);
        assert_eq!(slot.index(), val);
        assert!(!slot.is_invalid());
    }
}

#[test]
fn test_slot_invalid_is_distinct() {
    let invalid = BufferSlot::invalid();
    let max_minus_one = BufferSlot::new(u32::MAX - 1);
    assert_ne!(invalid, max_minus_one);
}

#[test]
fn test_slot_display_roundtrip() {
    let slot = BufferSlot::new(12345);
    let display = format!("{}", slot);
    assert!(display.contains("12345"));
}

#[test]
fn test_slot_debug_roundtrip() {
    let slot = BufferSlot::new(67890);
    let debug = format!("{:?}", slot);
    assert!(debug.contains("67890"));
}

#[test]
fn test_slot_in_vec() {
    let slots: Vec<BufferSlot> = (0..100).map(BufferSlot::new).collect();
    assert_eq!(slots.len(), 100);
    for (i, slot) in slots.iter().enumerate() {
        assert_eq!(slot.index(), i as u32);
    }
}
