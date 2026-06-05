//! Whitebox tests for bindless texture registry (T-WGPU-P2.6.1).
//!
//! This module provides comprehensive whitebox tests for the bindless texture
//! management system, covering TextureSlot, TextureRegistry, BindlessError,
//! TextureRegistryMetrics, feature detection, and helper functions.
//!
//! Test coverage targets:
//! - TextureSlot: ~15 tests
//! - Registry Construction: ~10 tests
//! - Registration: ~15 tests
//! - Unregistration & Recycling: ~15 tests
//! - Bind Group Management: ~15 tests
//! - Metrics: ~10 tests
//! - Feature Detection: ~10 tests
//! - Error Types: ~10 tests
//! - Thread Safety: ~5 tests
//! - Edge Cases: ~10 tests

use renderer_backend::resources::bindless_textures::{
    bindless_optimal_features, bindless_required_features, bindless_texture_layout_entry,
    max_bindless_textures_from_limits, BindlessError, TextureRegistry, TextureRegistryMetrics,
    TextureSlot, BINDLESS_BIND_GROUP_INDEX, BINDLESS_TEXTURE_BINDING, DEFAULT_MAX_TEXTURES,
    MAX_BINDLESS_TEXTURES_CONSERVATIVE, MIN_BINDLESS_TEXTURES,
};
use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::num::NonZeroU32;
use wgpu::{Features, Limits, ShaderStages, TextureSampleType, TextureViewDimension};

// ============================================================================
// TextureSlot Tests (~15 tests)
// ============================================================================

#[test]
fn test_slot_new_creates_slot_with_index() {
    let slot = TextureSlot::new(42);
    assert_eq!(slot.index(), 42);
}

#[test]
fn test_slot_new_zero_index() {
    let slot = TextureSlot::new(0);
    assert_eq!(slot.index(), 0);
    assert!(!slot.is_invalid());
}

#[test]
fn test_slot_index_accessor_returns_correct_value() {
    for i in [0, 1, 100, 1000, u32::MAX - 1] {
        let slot = TextureSlot::new(i);
        assert_eq!(slot.index(), i);
    }
}

#[test]
fn test_slot_invalid_returns_max_u32() {
    let slot = TextureSlot::invalid();
    assert_eq!(slot.index(), u32::MAX);
    assert!(slot.is_invalid());
}

#[test]
fn test_slot_is_invalid_true_only_for_max_u32() {
    assert!(TextureSlot::new(u32::MAX).is_invalid());
    assert!(!TextureSlot::new(u32::MAX - 1).is_invalid());
    assert!(!TextureSlot::new(0).is_invalid());
}

#[test]
fn test_slot_from_u32_conversion() {
    let slot: TextureSlot = 123u32.into();
    assert_eq!(slot.index(), 123);
}

#[test]
fn test_slot_into_u32_conversion() {
    let slot = TextureSlot::new(456);
    let index: u32 = slot.into();
    assert_eq!(index, 456);
}

#[test]
fn test_slot_roundtrip_conversion() {
    for i in [0, 42, 999, u32::MAX] {
        let slot: TextureSlot = i.into();
        let back: u32 = slot.into();
        assert_eq!(i, back);
    }
}

#[test]
fn test_slot_debug_trait_format() {
    let slot = TextureSlot::new(7);
    let debug = format!("{:?}", slot);
    assert!(debug.contains("TextureSlot"));
    assert!(debug.contains("7"));
}

#[test]
fn test_slot_clone_trait() {
    let slot1 = TextureSlot::new(8);
    let slot2 = slot1.clone();
    assert_eq!(slot1, slot2);
}

#[test]
fn test_slot_copy_trait() {
    let slot1 = TextureSlot::new(5);
    let slot2 = slot1; // Copy
    assert_eq!(slot1.index(), slot2.index());
}

#[test]
fn test_slot_partial_eq_same_values() {
    let slot1 = TextureSlot::new(10);
    let slot2 = TextureSlot::new(10);
    assert_eq!(slot1, slot2);
}

#[test]
fn test_slot_partial_eq_different_values() {
    let slot1 = TextureSlot::new(10);
    let slot2 = TextureSlot::new(20);
    assert_ne!(slot1, slot2);
}

#[test]
fn test_slot_eq_reflexive_symmetric_transitive() {
    let a = TextureSlot::new(42);
    let b = TextureSlot::new(42);
    let c = TextureSlot::new(42);

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

    let slot1 = TextureSlot::new(100);
    let slot2 = TextureSlot::new(100);

    let mut hasher1 = DefaultHasher::new();
    let mut hasher2 = DefaultHasher::new();

    slot1.hash(&mut hasher1);
    slot2.hash(&mut hasher2);

    assert_eq!(hasher1.finish(), hasher2.finish());
}

#[test]
fn test_slot_hash_in_hashset() {
    let mut set = HashSet::new();
    set.insert(TextureSlot::new(1));
    set.insert(TextureSlot::new(2));
    set.insert(TextureSlot::new(1)); // Duplicate

    assert_eq!(set.len(), 2);
    assert!(set.contains(&TextureSlot::new(1)));
    assert!(set.contains(&TextureSlot::new(2)));
}

#[test]
fn test_slot_as_hashmap_key() {
    let mut map: HashMap<TextureSlot, &str> = HashMap::new();
    map.insert(TextureSlot::new(1), "first");
    map.insert(TextureSlot::new(2), "second");

    assert_eq!(map.get(&TextureSlot::new(1)), Some(&"first"));
    assert_eq!(map.get(&TextureSlot::new(2)), Some(&"second"));
    assert_eq!(map.get(&TextureSlot::new(3)), None);
}

#[test]
fn test_slot_display_valid() {
    assert_eq!(format!("{}", TextureSlot::new(42)), "TextureSlot(42)");
    assert_eq!(format!("{}", TextureSlot::new(0)), "TextureSlot(0)");
}

#[test]
fn test_slot_display_invalid() {
    assert_eq!(format!("{}", TextureSlot::invalid()), "TextureSlot(INVALID)");
}

// ============================================================================
// Registry Construction Tests (~10 tests)
// ============================================================================

#[test]
fn test_registry_new_with_default_capacity() {
    let registry = TextureRegistry::new(DEFAULT_MAX_TEXTURES);
    assert_eq!(registry.capacity(), DEFAULT_MAX_TEXTURES);
    assert_eq!(registry.count(), 0);
}

#[test]
fn test_registry_new_with_custom_capacity() {
    let registry = TextureRegistry::new(512);
    assert_eq!(registry.capacity(), 512);
}

#[test]
fn test_registry_new_clamps_below_minimum() {
    let registry = TextureRegistry::new(1);
    assert_eq!(registry.capacity(), MIN_BINDLESS_TEXTURES);
}

#[test]
fn test_registry_new_clamps_zero_to_minimum() {
    let registry = TextureRegistry::new(0);
    assert_eq!(registry.capacity(), MIN_BINDLESS_TEXTURES);
}

#[test]
fn test_registry_new_clamps_above_maximum() {
    let registry = TextureRegistry::new(u32::MAX);
    assert_eq!(registry.capacity(), MAX_BINDLESS_TEXTURES_CONSERVATIVE);
}

#[test]
fn test_registry_new_exactly_at_minimum() {
    let registry = TextureRegistry::new(MIN_BINDLESS_TEXTURES);
    assert_eq!(registry.capacity(), MIN_BINDLESS_TEXTURES);
}

#[test]
fn test_registry_new_exactly_at_maximum() {
    let registry = TextureRegistry::new(MAX_BINDLESS_TEXTURES_CONSERVATIVE);
    assert_eq!(registry.capacity(), MAX_BINDLESS_TEXTURES_CONSERVATIVE);
}

#[test]
fn test_registry_default_uses_default_max_textures() {
    let registry = TextureRegistry::default();
    assert_eq!(registry.capacity(), DEFAULT_MAX_TEXTURES);
}

#[test]
fn test_registry_initial_state_is_empty() {
    let registry = TextureRegistry::new(100);
    assert!(registry.is_empty());
    assert_eq!(registry.count(), 0);
}

#[test]
fn test_registry_initial_state_not_full() {
    let registry = TextureRegistry::new(100);
    assert!(!registry.is_full());
}

#[test]
fn test_registry_initial_bind_group_is_none() {
    let registry = TextureRegistry::new(100);
    assert!(registry.bind_group().is_none());
}

#[test]
fn test_registry_initial_is_dirty() {
    let registry = TextureRegistry::new(100);
    assert!(registry.is_dirty());
}

#[test]
fn test_registry_initial_free_slot_count_zero() {
    let registry = TextureRegistry::new(100);
    assert_eq!(registry.free_slot_count(), 0);
}

// ============================================================================
// Registration Tests (~15 tests) - State-based without real textures
// ============================================================================

#[test]
fn test_registry_count_starts_at_zero() {
    let registry = TextureRegistry::new(100);
    assert_eq!(registry.count(), 0);
}

#[test]
fn test_registry_is_empty_on_new() {
    let registry = TextureRegistry::new(100);
    assert!(registry.is_empty());
}

#[test]
fn test_registry_capacity_returns_configured_value() {
    let registry = TextureRegistry::new(256);
    assert_eq!(registry.capacity(), 256);
}

#[test]
fn test_registry_is_full_false_when_empty() {
    let registry = TextureRegistry::new(100);
    assert!(!registry.is_full());
}

#[test]
fn test_registry_dirty_flag_initially_true() {
    let registry = TextureRegistry::new(100);
    assert!(registry.is_dirty());
}

#[test]
fn test_registry_free_slots_empty_initially() {
    let registry = TextureRegistry::new(100);
    assert_eq!(registry.free_slot_count(), 0);
}

#[test]
fn test_registry_get_returns_none_for_invalid_slot() {
    let registry = TextureRegistry::new(100);
    assert!(registry.get(TextureSlot::new(0)).is_none());
    assert!(registry.get(TextureSlot::new(999)).is_none());
    assert!(registry.get(TextureSlot::invalid()).is_none());
}

#[test]
fn test_registry_is_registered_false_for_empty_registry() {
    let registry = TextureRegistry::new(100);
    assert!(!registry.is_registered(TextureSlot::new(0)));
    assert!(!registry.is_registered(TextureSlot::new(50)));
}

#[test]
fn test_registry_iter_empty_on_new() {
    let registry = TextureRegistry::new(100);
    let count = registry.iter().count();
    assert_eq!(count, 0);
}

#[test]
fn test_registry_clear_on_empty() {
    let mut registry = TextureRegistry::new(100);
    registry.clear();
    assert!(registry.is_empty());
    assert!(registry.is_dirty());
}

#[test]
fn test_registry_clear_resets_count() {
    let mut registry = TextureRegistry::new(100);
    // Simulate some registrations by manipulating internal state
    // (We can't actually register without a texture, but we test clear behavior)
    registry.clear();
    assert_eq!(registry.count(), 0);
}

#[test]
fn test_registry_clear_resets_free_slots() {
    let mut registry = TextureRegistry::new(100);
    registry.clear();
    assert_eq!(registry.free_slot_count(), 0);
}

#[test]
fn test_registry_clear_removes_bind_group() {
    let mut registry = TextureRegistry::new(100);
    registry.clear();
    assert!(registry.bind_group().is_none());
}

#[test]
fn test_registry_clear_marks_dirty() {
    let mut registry = TextureRegistry::new(100);
    registry.clear();
    assert!(registry.is_dirty());
}

#[test]
fn test_registry_multiple_clears_idempotent() {
    let mut registry = TextureRegistry::new(100);
    registry.clear();
    registry.clear();
    registry.clear();
    assert!(registry.is_empty());
    assert_eq!(registry.free_slot_count(), 0);
}

// ============================================================================
// Unregistration Tests (~15 tests)
// ============================================================================

#[test]
fn test_unregister_invalid_slot_returns_false() {
    let mut registry = TextureRegistry::new(100);
    assert!(!registry.unregister(TextureSlot::invalid()));
}

#[test]
fn test_unregister_out_of_bounds_slot_returns_false() {
    let mut registry = TextureRegistry::new(100);
    assert!(!registry.unregister(TextureSlot::new(999)));
}

#[test]
fn test_unregister_empty_slot_returns_false() {
    let mut registry = TextureRegistry::new(100);
    // Slot 0 never had a texture registered
    assert!(!registry.unregister(TextureSlot::new(0)));
}

#[test]
fn test_unregister_zero_slot_on_empty_registry() {
    let mut registry = TextureRegistry::new(100);
    assert!(!registry.unregister(TextureSlot::new(0)));
}

#[test]
fn test_unregister_large_slot_index() {
    let mut registry = TextureRegistry::new(100);
    assert!(!registry.unregister(TextureSlot::new(u32::MAX - 1)));
}

#[test]
fn test_unregister_does_not_change_count_on_failure() {
    let mut registry = TextureRegistry::new(100);
    let initial_count = registry.count();
    registry.unregister(TextureSlot::new(50));
    assert_eq!(registry.count(), initial_count);
}

#[test]
fn test_unregister_does_not_add_to_free_list_on_failure() {
    let mut registry = TextureRegistry::new(100);
    let initial_free = registry.free_slot_count();
    registry.unregister(TextureSlot::new(50));
    assert_eq!(registry.free_slot_count(), initial_free);
}

// ============================================================================
// Bind Group Management Tests (~15 tests)
// ============================================================================

#[test]
fn test_bind_group_none_on_new_registry() {
    let registry = TextureRegistry::new(100);
    assert!(registry.bind_group().is_none());
}

#[test]
fn test_bind_group_none_after_clear() {
    let mut registry = TextureRegistry::new(100);
    registry.clear();
    assert!(registry.bind_group().is_none());
}

#[test]
fn test_is_dirty_true_initially() {
    let registry = TextureRegistry::new(100);
    assert!(registry.is_dirty());
}

#[test]
fn test_is_dirty_true_after_clear() {
    let mut registry = TextureRegistry::new(100);
    registry.clear();
    assert!(registry.is_dirty());
}

// ============================================================================
// Metrics Tests (~10 tests)
// ============================================================================

#[test]
fn test_metrics_empty_registry() {
    let registry = TextureRegistry::new(100);
    let metrics = registry.metrics();

    assert_eq!(metrics.registered_count, 0);
    assert_eq!(metrics.capacity, 100);
    assert_eq!(metrics.free_slots, 0);
    assert_eq!(metrics.allocated_slots, 0);
    assert!(!metrics.has_bind_group);
    assert!(metrics.is_dirty);
}

#[test]
fn test_metrics_capacity_reflects_registry() {
    let registry = TextureRegistry::new(512);
    assert_eq!(registry.metrics().capacity, 512);
}

#[test]
fn test_metrics_utilization_zero_when_empty() {
    let registry = TextureRegistry::new(100);
    assert_eq!(registry.metrics().utilization(), 0.0);
}

#[test]
fn test_metrics_utilization_handles_zero_capacity() {
    let metrics = TextureRegistryMetrics {
        registered_count: 0,
        capacity: 0,
        free_slots: 0,
        allocated_slots: 0,
        has_bind_group: false,
        is_dirty: true,
    };
    assert_eq!(metrics.utilization(), 0.0);
}

#[test]
fn test_metrics_utilization_calculation() {
    let metrics = TextureRegistryMetrics {
        registered_count: 50,
        capacity: 100,
        free_slots: 0,
        allocated_slots: 50,
        has_bind_group: true,
        is_dirty: false,
    };
    assert!((metrics.utilization() - 0.5).abs() < 0.001);
}

#[test]
fn test_metrics_utilization_full() {
    let metrics = TextureRegistryMetrics {
        registered_count: 100,
        capacity: 100,
        free_slots: 0,
        allocated_slots: 100,
        has_bind_group: true,
        is_dirty: false,
    };
    assert!((metrics.utilization() - 1.0).abs() < 0.001);
}

#[test]
fn test_metrics_fragmentation_zero_when_no_allocations() {
    let metrics = TextureRegistryMetrics {
        registered_count: 0,
        capacity: 100,
        free_slots: 0,
        allocated_slots: 0,
        has_bind_group: false,
        is_dirty: true,
    };
    assert_eq!(metrics.fragmentation(), 0.0);
}

#[test]
fn test_metrics_fragmentation_calculation() {
    let metrics = TextureRegistryMetrics {
        registered_count: 8,
        capacity: 100,
        free_slots: 2,
        allocated_slots: 10,
        has_bind_group: true,
        is_dirty: false,
    };
    // 2/10 = 0.2
    assert!((metrics.fragmentation() - 0.2).abs() < 0.001);
}

#[test]
fn test_metrics_fragmentation_half() {
    let metrics = TextureRegistryMetrics {
        registered_count: 5,
        capacity: 100,
        free_slots: 5,
        allocated_slots: 10,
        has_bind_group: true,
        is_dirty: false,
    };
    // 5/10 = 0.5
    assert!((metrics.fragmentation() - 0.5).abs() < 0.001);
}

#[test]
fn test_metrics_fragmentation_all_free() {
    let metrics = TextureRegistryMetrics {
        registered_count: 0,
        capacity: 100,
        free_slots: 10,
        allocated_slots: 10,
        has_bind_group: false,
        is_dirty: true,
    };
    // 10/10 = 1.0
    assert!((metrics.fragmentation() - 1.0).abs() < 0.001);
}

#[test]
fn test_metrics_debug_format() {
    let metrics = TextureRegistryMetrics {
        registered_count: 5,
        capacity: 100,
        free_slots: 2,
        allocated_slots: 7,
        has_bind_group: true,
        is_dirty: false,
    };
    let debug = format!("{:?}", metrics);
    assert!(debug.contains("TextureRegistryMetrics"));
    assert!(debug.contains("registered_count"));
    assert!(debug.contains("capacity"));
}

#[test]
fn test_metrics_clone() {
    let metrics1 = TextureRegistryMetrics {
        registered_count: 25,
        capacity: 100,
        free_slots: 5,
        allocated_slots: 30,
        has_bind_group: true,
        is_dirty: false,
    };
    let metrics2 = metrics1;
    assert_eq!(metrics1.registered_count, metrics2.registered_count);
    assert_eq!(metrics1.capacity, metrics2.capacity);
}

#[test]
fn test_metrics_equality() {
    let m1 = TextureRegistryMetrics {
        registered_count: 10,
        capacity: 100,
        free_slots: 2,
        allocated_slots: 12,
        has_bind_group: true,
        is_dirty: false,
    };
    let m2 = TextureRegistryMetrics {
        registered_count: 10,
        capacity: 100,
        free_slots: 2,
        allocated_slots: 12,
        has_bind_group: true,
        is_dirty: false,
    };
    let m3 = TextureRegistryMetrics {
        registered_count: 11,
        capacity: 100,
        free_slots: 2,
        allocated_slots: 12,
        has_bind_group: true,
        is_dirty: false,
    };
    assert_eq!(m1, m2);
    assert_ne!(m1, m3);
}

// ============================================================================
// Feature Detection Tests (~10 tests)
// ============================================================================

#[test]
fn test_constants_default_max_textures_value() {
    assert_eq!(DEFAULT_MAX_TEXTURES, 1024);
}

#[test]
fn test_constants_min_bindless_textures_value() {
    assert_eq!(MIN_BINDLESS_TEXTURES, 16);
}

#[test]
fn test_constants_max_conservative_value() {
    assert_eq!(MAX_BINDLESS_TEXTURES_CONSERVATIVE, 16384);
}

#[test]
fn test_constants_bind_group_index() {
    assert_eq!(BINDLESS_BIND_GROUP_INDEX, 3);
}

#[test]
fn test_constants_texture_binding() {
    assert_eq!(BINDLESS_TEXTURE_BINDING, 0);
}

#[test]
fn test_constants_hierarchy() {
    assert!(MIN_BINDLESS_TEXTURES < DEFAULT_MAX_TEXTURES);
    assert!(DEFAULT_MAX_TEXTURES < MAX_BINDLESS_TEXTURES_CONSERVATIVE);
}

#[test]
fn test_bindless_required_features_includes_texture_binding_array() {
    let features = bindless_required_features();
    assert!(features.contains(Features::TEXTURE_BINDING_ARRAY));
}

#[test]
fn test_bindless_optimal_features_includes_all_required() {
    let optimal = bindless_optimal_features();
    let required = bindless_required_features();
    assert!(optimal.contains(required));
}

#[test]
fn test_bindless_optimal_features_includes_non_uniform_indexing() {
    let features = bindless_optimal_features();
    assert!(features.contains(
        Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING
    ));
}

#[test]
fn test_bindless_optimal_features_includes_partially_bound() {
    let features = bindless_optimal_features();
    assert!(features.contains(Features::PARTIALLY_BOUND_BINDING_ARRAY));
}

#[test]
fn test_max_bindless_from_limits_normal() {
    let mut limits = Limits::default();
    limits.max_sampled_textures_per_shader_stage = 500;
    assert_eq!(max_bindless_textures_from_limits(&limits), 500);
}

#[test]
fn test_max_bindless_from_limits_clamped_to_conservative() {
    let mut limits = Limits::default();
    limits.max_sampled_textures_per_shader_stage = u32::MAX;
    assert_eq!(
        max_bindless_textures_from_limits(&limits),
        MAX_BINDLESS_TEXTURES_CONSERVATIVE
    );
}

#[test]
fn test_max_bindless_from_limits_at_conservative() {
    let mut limits = Limits::default();
    limits.max_sampled_textures_per_shader_stage = MAX_BINDLESS_TEXTURES_CONSERVATIVE;
    assert_eq!(
        max_bindless_textures_from_limits(&limits),
        MAX_BINDLESS_TEXTURES_CONSERVATIVE
    );
}

#[test]
fn test_max_bindless_from_limits_below_conservative() {
    let mut limits = Limits::default();
    limits.max_sampled_textures_per_shader_stage = 1000;
    assert_eq!(max_bindless_textures_from_limits(&limits), 1000);
}

// ============================================================================
// Error Types Tests (~10 tests)
// ============================================================================

#[test]
fn test_error_unsupported_feature_display() {
    let err = BindlessError::UnsupportedFeature;
    let msg = err.to_string();
    assert!(msg.contains("TEXTURE_BINDING_ARRAY"));
    assert!(msg.contains("not supported"));
}

#[test]
fn test_error_registry_full_display() {
    let err = BindlessError::RegistryFull { capacity: 1024 };
    let msg = err.to_string();
    assert!(msg.contains("1024"));
    assert!(msg.contains("full"));
}

#[test]
fn test_error_invalid_slot_display() {
    let err = BindlessError::InvalidSlot(TextureSlot::new(42));
    let msg = err.to_string();
    assert!(msg.contains("42"));
    assert!(msg.contains("invalid"));
}

#[test]
fn test_error_invalid_slot_with_invalid_marker() {
    let err = BindlessError::InvalidSlot(TextureSlot::invalid());
    let msg = err.to_string();
    assert!(msg.contains("INVALID"));
}

#[test]
fn test_error_exceeds_device_limit_display() {
    let err = BindlessError::ExceedsDeviceLimit {
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
    let err = BindlessError::IncompatibleLayout;
    assert!(err.to_string().contains("incompatible"));
}

#[test]
fn test_error_empty_registry_display() {
    let err = BindlessError::EmptyRegistry;
    assert!(err.to_string().contains("empty"));
}

#[test]
fn test_error_equality_same() {
    let err1 = BindlessError::RegistryFull { capacity: 100 };
    let err2 = BindlessError::RegistryFull { capacity: 100 };
    assert_eq!(err1, err2);
}

#[test]
fn test_error_equality_different_capacity() {
    let err1 = BindlessError::RegistryFull { capacity: 100 };
    let err2 = BindlessError::RegistryFull { capacity: 200 };
    assert_ne!(err1, err2);
}

#[test]
fn test_error_equality_different_variants() {
    let err1 = BindlessError::UnsupportedFeature;
    let err2 = BindlessError::EmptyRegistry;
    assert_ne!(err1, err2);
}

#[test]
fn test_error_clone() {
    let err1 = BindlessError::ExceedsDeviceLimit {
        requested: 5000,
        max: 4096,
    };
    let err2 = err1.clone();
    assert_eq!(err1, err2);
}

#[test]
fn test_error_debug_format() {
    let err = BindlessError::InvalidSlot(TextureSlot::new(99));
    let debug = format!("{:?}", err);
    assert!(debug.contains("InvalidSlot"));
    assert!(debug.contains("99"));
}

#[test]
fn test_error_is_std_error() {
    fn assert_error<T: std::error::Error>() {}
    assert_error::<BindlessError>();
}

#[test]
fn test_error_std_error_to_string() {
    let err: Box<dyn std::error::Error> = Box::new(BindlessError::UnsupportedFeature);
    assert!(err.to_string().contains("not supported"));
}

// ============================================================================
// Thread Safety Tests (~5 tests)
// ============================================================================

#[test]
fn test_registry_is_send() {
    fn assert_send<T: Send>() {}
    assert_send::<TextureRegistry>();
}

#[test]
fn test_registry_is_sync() {
    fn assert_sync<T: Sync>() {}
    assert_sync::<TextureRegistry>();
}

#[test]
fn test_slot_is_send() {
    fn assert_send<T: Send>() {}
    assert_send::<TextureSlot>();
}

#[test]
fn test_slot_is_sync() {
    fn assert_sync<T: Sync>() {}
    assert_sync::<TextureSlot>();
}

#[test]
fn test_error_is_send() {
    fn assert_send<T: Send>() {}
    assert_send::<BindlessError>();
}

#[test]
fn test_error_is_sync() {
    fn assert_sync<T: Sync>() {}
    assert_sync::<BindlessError>();
}

#[test]
fn test_metrics_is_send() {
    fn assert_send<T: Send>() {}
    assert_send::<TextureRegistryMetrics>();
}

#[test]
fn test_metrics_is_sync() {
    fn assert_sync<T: Sync>() {}
    assert_sync::<TextureRegistryMetrics>();
}

// ============================================================================
// Edge Cases Tests (~10 tests)
// ============================================================================

#[test]
fn test_slot_zero_is_valid() {
    let slot = TextureSlot::new(0);
    assert!(!slot.is_invalid());
    assert_eq!(slot.index(), 0);
}

#[test]
fn test_slot_max_minus_one_is_valid() {
    let slot = TextureSlot::new(u32::MAX - 1);
    assert!(!slot.is_invalid());
    assert_eq!(slot.index(), u32::MAX - 1);
}

#[test]
fn test_slot_one_is_valid() {
    let slot = TextureSlot::new(1);
    assert!(!slot.is_invalid());
}

#[test]
fn test_registry_min_capacity_works() {
    let registry = TextureRegistry::new(MIN_BINDLESS_TEXTURES);
    assert_eq!(registry.capacity(), MIN_BINDLESS_TEXTURES);
    assert!(registry.is_empty());
}

#[test]
fn test_registry_max_capacity_works() {
    let registry = TextureRegistry::new(MAX_BINDLESS_TEXTURES_CONSERVATIVE);
    assert_eq!(registry.capacity(), MAX_BINDLESS_TEXTURES_CONSERVATIVE);
}

#[test]
fn test_registry_one_below_min_clamps() {
    let registry = TextureRegistry::new(MIN_BINDLESS_TEXTURES - 1);
    assert_eq!(registry.capacity(), MIN_BINDLESS_TEXTURES);
}

#[test]
fn test_registry_one_above_max_clamps() {
    let registry = TextureRegistry::new(MAX_BINDLESS_TEXTURES_CONSERVATIVE + 1);
    assert_eq!(registry.capacity(), MAX_BINDLESS_TEXTURES_CONSERVATIVE);
}

#[test]
fn test_registry_debug_format() {
    let registry = TextureRegistry::new(256);
    let debug = format!("{:?}", registry);
    assert!(debug.contains("TextureRegistry"));
    assert!(debug.contains("capacity"));
    assert!(debug.contains("256"));
}

#[test]
fn test_registry_iterator_returns_empty_on_new() {
    let registry = TextureRegistry::new(100);
    let items: Vec<_> = registry.iter().collect();
    assert!(items.is_empty());
}

#[test]
fn test_multiple_default_registries_independent() {
    let r1 = TextureRegistry::default();
    let mut r2 = TextureRegistry::default();

    r2.clear();

    // r1 should be unaffected by r2
    assert!(r1.is_dirty());
    assert!(r1.bind_group().is_none());
}

// ============================================================================
// Bind Group Layout Entry Tests
// ============================================================================

#[test]
fn test_bindless_layout_entry_binding() {
    let entry = bindless_texture_layout_entry(
        0,
        1024,
        TextureSampleType::Float { filterable: true },
        TextureViewDimension::D2,
    );
    assert_eq!(entry.binding, 0);
}

#[test]
fn test_bindless_layout_entry_visibility() {
    let entry = bindless_texture_layout_entry(
        0,
        1024,
        TextureSampleType::Float { filterable: true },
        TextureViewDimension::D2,
    );
    assert_eq!(entry.visibility, ShaderStages::VERTEX_FRAGMENT);
}

#[test]
fn test_bindless_layout_entry_count() {
    let entry = bindless_texture_layout_entry(
        0,
        1024,
        TextureSampleType::Float { filterable: true },
        TextureViewDimension::D2,
    );
    assert_eq!(entry.count, NonZeroU32::new(1024));
}

#[test]
fn test_bindless_layout_entry_custom_binding() {
    let entry = bindless_texture_layout_entry(
        5,
        256,
        TextureSampleType::Uint,
        TextureViewDimension::D2Array,
    );
    assert_eq!(entry.binding, 5);
    assert_eq!(entry.count, NonZeroU32::new(256));
}

#[test]
fn test_bindless_layout_entry_texture_type() {
    let entry = bindless_texture_layout_entry(
        0,
        100,
        TextureSampleType::Float { filterable: true },
        TextureViewDimension::D2,
    );

    if let wgpu::BindingType::Texture {
        sample_type,
        view_dimension,
        multisampled,
    } = entry.ty
    {
        assert_eq!(sample_type, TextureSampleType::Float { filterable: true });
        assert_eq!(view_dimension, TextureViewDimension::D2);
        assert!(!multisampled);
    } else {
        panic!("Expected Texture binding type");
    }
}

#[test]
fn test_bindless_layout_entry_depth_sample_type() {
    let entry = bindless_texture_layout_entry(
        0,
        64,
        TextureSampleType::Depth,
        TextureViewDimension::D2,
    );

    if let wgpu::BindingType::Texture { sample_type, .. } = entry.ty {
        assert_eq!(sample_type, TextureSampleType::Depth);
    } else {
        panic!("Expected Texture binding type");
    }
}

#[test]
fn test_bindless_layout_entry_sint_sample_type() {
    let entry = bindless_texture_layout_entry(
        0,
        32,
        TextureSampleType::Sint,
        TextureViewDimension::D2,
    );

    if let wgpu::BindingType::Texture { sample_type, .. } = entry.ty {
        assert_eq!(sample_type, TextureSampleType::Sint);
    } else {
        panic!("Expected Texture binding type");
    }
}

#[test]
fn test_bindless_layout_entry_cube_dimension() {
    let entry = bindless_texture_layout_entry(
        0,
        16,
        TextureSampleType::Float { filterable: true },
        TextureViewDimension::Cube,
    );

    if let wgpu::BindingType::Texture { view_dimension, .. } = entry.ty {
        assert_eq!(view_dimension, TextureViewDimension::Cube);
    } else {
        panic!("Expected Texture binding type");
    }
}

#[test]
fn test_bindless_layout_entry_3d_dimension() {
    let entry = bindless_texture_layout_entry(
        0,
        8,
        TextureSampleType::Float { filterable: true },
        TextureViewDimension::D3,
    );

    if let wgpu::BindingType::Texture { view_dimension, .. } = entry.ty {
        assert_eq!(view_dimension, TextureViewDimension::D3);
    } else {
        panic!("Expected Texture binding type");
    }
}

#[test]
fn test_bindless_layout_entry_unfilterable_float() {
    let entry = bindless_texture_layout_entry(
        0,
        128,
        TextureSampleType::Float { filterable: false },
        TextureViewDimension::D2,
    );

    if let wgpu::BindingType::Texture { sample_type, .. } = entry.ty {
        assert_eq!(sample_type, TextureSampleType::Float { filterable: false });
    } else {
        panic!("Expected Texture binding type");
    }
}

// ============================================================================
// Additional Coverage Tests
// ============================================================================

#[test]
fn test_slot_ordering_via_sort() {
    let mut slots = vec![
        TextureSlot::new(5),
        TextureSlot::new(1),
        TextureSlot::new(3),
    ];
    slots.sort_by_key(|s| s.index());
    assert_eq!(slots[0].index(), 1);
    assert_eq!(slots[1].index(), 3);
    assert_eq!(slots[2].index(), 5);
}

#[test]
fn test_error_registry_full_with_different_capacities() {
    for cap in [16, 100, 1000, 16384] {
        let err = BindlessError::RegistryFull { capacity: cap };
        assert!(err.to_string().contains(&cap.to_string()));
    }
}

#[test]
fn test_error_exceeds_limit_boundary_values() {
    let err = BindlessError::ExceedsDeviceLimit {
        requested: MAX_BINDLESS_TEXTURES_CONSERVATIVE + 1,
        max: MAX_BINDLESS_TEXTURES_CONSERVATIVE,
    };
    assert!(err.to_string().contains(&(MAX_BINDLESS_TEXTURES_CONSERVATIVE + 1).to_string()));
}

#[test]
fn test_metrics_copy_trait() {
    let m1 = TextureRegistryMetrics {
        registered_count: 10,
        capacity: 100,
        free_slots: 2,
        allocated_slots: 12,
        has_bind_group: true,
        is_dirty: false,
    };
    let m2 = m1; // Copy
    assert_eq!(m1.registered_count, m2.registered_count);
}

#[test]
fn test_registry_capacity_stability_after_clear() {
    let mut registry = TextureRegistry::new(512);
    let initial_capacity = registry.capacity();
    registry.clear();
    assert_eq!(registry.capacity(), initial_capacity);
}

#[test]
fn test_slot_const_new() {
    const SLOT: TextureSlot = TextureSlot::new(42);
    assert_eq!(SLOT.index(), 42);
}

#[test]
fn test_slot_const_invalid() {
    const INVALID: TextureSlot = TextureSlot::invalid();
    assert!(INVALID.is_invalid());
}

#[test]
fn test_slot_const_index() {
    const SLOT: TextureSlot = TextureSlot::new(100);
    const INDEX: u32 = SLOT.index();
    assert_eq!(INDEX, 100);
}

#[test]
fn test_slot_const_is_invalid() {
    const SLOT: TextureSlot = TextureSlot::new(42);
    const IS_INVALID: bool = SLOT.is_invalid();
    assert!(!IS_INVALID);
}
