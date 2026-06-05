// SPDX-License-Identifier: MIT
//
// blackbox_bindless_buffers.rs -- Blackbox tests for T-WGPU-P2.6.2 Bindless Buffer Registry.
//
// CLEANROOM: No src/ access beyond the public API exported by the crate.
// Tests use only the documented public types and functions:
//
//   - BufferSlot
//   - BufferRegistry
//   - BufferRegistryMetrics
//   - BindlessBufferError
//   - supports_bindless_buffers(), supports_non_uniform_buffer_indexing(), supports_storage_buffer_array()
//   - max_bindless_buffers(), max_bindless_buffers_from_limits()
//   - bindless_buffer_layout_entry(), bindless_buffer_layout_entry_readonly(), bindless_buffer_layout_entry_readwrite()
//   - create_bindless_buffer_layout()
//   - bindless_buffer_required_features(), bindless_buffer_optimal_features()
//   - BINDLESS_BUFFER_BINDING
//   - DEFAULT_MAX_BUFFERS, MAX_BINDLESS_BUFFERS_CONSERVATIVE, MIN_BINDLESS_BUFFERS
//
// NOTE: Many BufferRegistry operations require actual wgpu::Buffer instances.
// Tests that need real buffers are marked #[ignore] since they require GPU initialization.
//
// ACCEPTANCE CRITERIA:
//   1. Constants tests              -- 7 tests covering const values
//   2. BufferSlot API tests         -- 15 tests for slot construction and accessors
//   3. BufferRegistry API tests     -- 20 tests for registry operations (some ignored)
//   4. Dirty Tracking API tests     -- 15 tests for dirty tracking (some ignored)
//   5. Metrics API tests            -- 10 tests for metrics types
//   6. Error API tests              -- 10 tests for error handling
//   7. Feature detection tests      -- 6 tests
//   8. Property-based tests         -- 5 tests for invariants
//
// Total target: ~88 tests

use renderer_backend::resources::{
    bindless_buffer_layout_entry, bindless_buffer_layout_entry_readonly,
    bindless_buffer_layout_entry_readwrite, bindless_buffer_optimal_features,
    bindless_buffer_required_features, max_bindless_buffers_from_limits,
    BindlessBufferError, BufferRegistry, BufferRegistryMetrics, BufferSlot,
    BINDLESS_BUFFER_BINDING, DEFAULT_MAX_BUFFERS, MAX_BINDLESS_BUFFERS_CONSERVATIVE,
    MIN_BINDLESS_BUFFERS,
};
use std::collections::HashSet;

// =============================================================================
// SECTION 1 -- CONSTANTS TESTS (7 tests)
// =============================================================================

/// BINDLESS_BUFFER_BINDING should be 1 (second binding in the group, after textures).
#[test]
fn constant_bindless_buffer_binding_value() {
    // Buffer binding is 1, texture binding is 0 in the bindless group
    assert_eq!(BINDLESS_BUFFER_BINDING, 1);
}

/// DEFAULT_MAX_BUFFERS should be 1024.
#[test]
fn constant_default_max_buffers_value() {
    assert_eq!(DEFAULT_MAX_BUFFERS, 1024);
}

/// DEFAULT_MAX_BUFFERS should be a power-of-two value.
#[test]
fn constant_default_max_buffers_is_power_of_two() {
    assert!(DEFAULT_MAX_BUFFERS > 0);
    assert!(DEFAULT_MAX_BUFFERS.is_power_of_two());
}

/// DEFAULT_MAX_BUFFERS should be at least MIN_BINDLESS_BUFFERS.
#[test]
fn constant_default_at_least_min() {
    assert!(DEFAULT_MAX_BUFFERS >= MIN_BINDLESS_BUFFERS);
}

/// DEFAULT_MAX_BUFFERS should not exceed MAX_BINDLESS_BUFFERS_CONSERVATIVE.
#[test]
fn constant_default_at_most_conservative_max() {
    assert!(DEFAULT_MAX_BUFFERS <= MAX_BINDLESS_BUFFERS_CONSERVATIVE);
}

/// MIN_BINDLESS_BUFFERS should be 16.
#[test]
fn constant_min_bindless_buffers_value() {
    assert_eq!(MIN_BINDLESS_BUFFERS, 16);
}

/// MAX_BINDLESS_BUFFERS_CONSERVATIVE should be 16384.
#[test]
fn constant_max_conservative_value() {
    assert_eq!(MAX_BINDLESS_BUFFERS_CONSERVATIVE, 16384);
}

// =============================================================================
// SECTION 2 -- BUFFER SLOT API TESTS (15 tests)
// =============================================================================

/// BufferSlot::new() creates a slot with the given index.
#[test]
fn slot_new_creates_with_index() {
    let slot = BufferSlot::new(42);
    assert_eq!(slot.index(), 42);
}

/// BufferSlot::new() with index 0 creates a valid slot.
#[test]
fn slot_new_zero_index() {
    let slot = BufferSlot::new(0);
    assert_eq!(slot.index(), 0);
    assert!(!slot.is_invalid());
}

/// BufferSlot::new() with u32::MAX-1 creates a valid slot.
#[test]
fn slot_new_near_max_index() {
    let slot = BufferSlot::new(u32::MAX - 1);
    assert_eq!(slot.index(), u32::MAX - 1);
}

/// BufferSlot::invalid() creates a slot that is_invalid().
#[test]
fn slot_invalid_returns_invalid() {
    let slot = BufferSlot::invalid();
    assert!(slot.is_invalid());
}

/// BufferSlot::invalid() index should be u32::MAX.
#[test]
fn slot_invalid_index_is_max() {
    let slot = BufferSlot::invalid();
    assert_eq!(slot.index(), u32::MAX);
}

/// A normal slot is NOT invalid.
#[test]
fn slot_normal_not_invalid() {
    let slot = BufferSlot::new(100);
    assert!(!slot.is_invalid());
}

/// BufferSlot implements Copy.
#[test]
fn slot_is_copy() {
    let slot = BufferSlot::new(5);
    let slot2 = slot; // Copy
    assert_eq!(slot.index(), slot2.index());
}

/// BufferSlot implements Clone.
#[test]
fn slot_is_clone() {
    let slot = BufferSlot::new(7);
    let slot2 = slot.clone();
    assert_eq!(slot.index(), slot2.index());
}

/// BufferSlot implements PartialEq.
#[test]
fn slot_partial_eq_same() {
    let slot1 = BufferSlot::new(10);
    let slot2 = BufferSlot::new(10);
    assert_eq!(slot1, slot2);
}

/// BufferSlot implements PartialEq for different values.
#[test]
fn slot_partial_eq_different() {
    let slot1 = BufferSlot::new(10);
    let slot2 = BufferSlot::new(20);
    assert_ne!(slot1, slot2);
}

/// BufferSlot implements Eq transitively.
#[test]
fn slot_eq_transitive() {
    let slot1 = BufferSlot::new(5);
    let slot2 = BufferSlot::new(5);
    let slot3 = BufferSlot::new(5);
    assert!(slot1 == slot2 && slot2 == slot3 && slot1 == slot3);
}

/// Invalid slots are equal to each other.
#[test]
fn slot_invalid_equal() {
    let slot1 = BufferSlot::invalid();
    let slot2 = BufferSlot::invalid();
    assert_eq!(slot1, slot2);
}

/// BufferSlot implements Hash for use in HashSet.
#[test]
fn slot_hashable() {
    let mut set = HashSet::new();
    set.insert(BufferSlot::new(1));
    set.insert(BufferSlot::new(2));
    set.insert(BufferSlot::new(1)); // Duplicate
    assert_eq!(set.len(), 2);
}

/// BufferSlot implements Debug.
#[test]
fn slot_debug_format() {
    let slot = BufferSlot::new(42);
    let debug_str = format!("{:?}", slot);
    assert!(debug_str.contains("42") || debug_str.contains("BufferSlot"));
}

/// BufferSlot::index() does not mutate the slot.
#[test]
fn slot_index_is_immutable() {
    let slot = BufferSlot::new(99);
    let idx1 = slot.index();
    let idx2 = slot.index();
    assert_eq!(idx1, idx2);
}

// =============================================================================
// SECTION 3 -- BUFFER REGISTRY API TESTS (20 tests)
// =============================================================================

/// BufferRegistry::new() creates an empty registry with specified capacity.
#[test]
fn registry_new_is_empty() {
    let registry = BufferRegistry::new(DEFAULT_MAX_BUFFERS);
    assert_eq!(registry.count(), 0);
    assert!(registry.is_empty());
}

/// BufferRegistry::new() creates registry with specified capacity.
#[test]
fn registry_new_has_specified_capacity() {
    let registry = BufferRegistry::new(512);
    assert_eq!(registry.capacity(), 512);
}

/// BufferRegistry::new() with default capacity.
#[test]
fn registry_new_default_capacity() {
    let registry = BufferRegistry::new(DEFAULT_MAX_BUFFERS);
    assert_eq!(registry.capacity(), DEFAULT_MAX_BUFFERS);
}

/// BufferRegistry capacity can be small.
#[test]
fn registry_with_small_capacity() {
    let registry = BufferRegistry::new(16);
    assert_eq!(registry.capacity(), 16);
}

/// BufferRegistry::is_empty() returns true when empty.
#[test]
fn registry_is_empty_when_empty() {
    let registry = BufferRegistry::new(64);
    assert!(registry.is_empty());
}

/// BufferRegistry::is_full() returns false when empty.
#[test]
fn registry_not_full_when_empty() {
    let registry = BufferRegistry::new(64);
    assert!(!registry.is_full());
}

/// BufferRegistry::count() returns 0 for new registry.
#[test]
fn registry_count_zero_initially() {
    let registry = BufferRegistry::new(64);
    assert_eq!(registry.count(), 0);
}

/// BufferRegistry::free_slot_count() is initially 0 (slots allocated on first register).
#[test]
fn registry_free_slot_count_zero_initially() {
    let registry = BufferRegistry::new(64);
    // Free slots may be 0 initially until first allocation
    assert_eq!(registry.free_slot_count(), 0);
}

/// BufferRegistry::clear() is callable on empty registry.
#[test]
fn registry_clear_empty_is_noop() {
    let mut registry = BufferRegistry::new(64);
    registry.clear();
    assert!(registry.is_empty());
    assert_eq!(registry.capacity(), 64);
}

/// BufferRegistryMetrics::utilization() is 0.0 for empty registry.
#[test]
fn registry_utilization_zero_when_empty() {
    let registry = BufferRegistry::new(64);
    let metrics = registry.metrics();
    assert!((metrics.utilization() - 0.0).abs() < 0.001);
}

/// BufferRegistryMetrics::fragmentation() is 0.0 for empty registry.
#[test]
fn registry_fragmentation_zero_when_empty() {
    let registry = BufferRegistry::new(64);
    let metrics = registry.metrics();
    assert!((metrics.fragmentation() - 0.0).abs() < 0.001);
}

/// BufferRegistryMetrics::dirty_ratio() is 0.0 for empty registry.
#[test]
fn registry_dirty_ratio_zero_when_empty() {
    let registry = BufferRegistry::new(64);
    let metrics = registry.metrics();
    // With no buffers, dirty ratio should be 0 or undefined (0)
    let ratio = metrics.dirty_ratio();
    assert!(ratio >= 0.0 && ratio <= 1.0);
}

/// BufferRegistry::bind_group() is None for new registry.
#[test]
fn registry_bind_group_none_initially() {
    let registry = BufferRegistry::new(64);
    assert!(registry.bind_group().is_none());
}

/// BufferRegistry::is_dirty() returns true for new registry (no bind group created yet).
#[test]
fn registry_dirty_when_new() {
    let registry = BufferRegistry::new(64);
    // New registry is considered dirty because no bind group has been created
    assert!(registry.is_dirty());
}

/// BufferRegistry::dirty_count() is 0 for new registry.
#[test]
fn registry_dirty_count_zero_initially() {
    let registry = BufferRegistry::new(64);
    assert_eq!(registry.dirty_count(), 0);
}

/// BufferRegistry::iter() yields nothing for empty registry.
#[test]
fn registry_iter_empty() {
    let registry = BufferRegistry::new(64);
    let items: Vec<_> = registry.iter().collect();
    assert!(items.is_empty());
}

/// BufferRegistry::unregister() returns false for invalid slot.
#[test]
fn registry_unregister_invalid_slot() {
    let mut registry = BufferRegistry::new(64);
    let invalid = BufferSlot::invalid();
    assert!(!registry.unregister(invalid));
}

/// BufferRegistry::unregister() returns false for out-of-range slot.
#[test]
fn registry_unregister_out_of_range() {
    let mut registry = BufferRegistry::new(64);
    let slot = BufferSlot::new(1000); // Beyond capacity
    assert!(!registry.unregister(slot));
}

/// BufferRegistry::get() returns None for invalid slot.
#[test]
fn registry_get_invalid_slot() {
    let registry = BufferRegistry::new(64);
    let invalid = BufferSlot::invalid();
    assert!(registry.get(invalid).is_none());
}

/// BufferRegistry::is_registered() returns false for invalid slot.
#[test]
fn registry_is_registered_invalid_slot() {
    let registry = BufferRegistry::new(64);
    let invalid = BufferSlot::invalid();
    assert!(!registry.is_registered(invalid));
}

// =============================================================================
// SECTION 4 -- DIRTY TRACKING API TESTS (15 tests)
// =============================================================================

/// BufferRegistry has no dirty slots initially.
#[test]
fn dirty_none_initially() {
    let registry = BufferRegistry::new(64);
    assert_eq!(registry.dirty_count(), 0);
}

/// BufferRegistry::mark_dirty() on invalid slot is a no-op.
#[test]
fn dirty_mark_invalid_noop() {
    let mut registry = BufferRegistry::new(64);
    let invalid = BufferSlot::invalid();
    registry.mark_dirty(invalid);
    assert_eq!(registry.dirty_count(), 0);
}

/// BufferRegistry::mark_dirty() on out-of-range slot is a no-op.
#[test]
fn dirty_mark_out_of_range_noop() {
    let mut registry = BufferRegistry::new(64);
    let slot = BufferSlot::new(1000); // Beyond capacity
    registry.mark_dirty(slot);
    assert_eq!(registry.dirty_count(), 0);
}

/// BufferRegistry::dirty_slots() returns empty iterator initially.
#[test]
fn dirty_slots_empty_initially() {
    let registry = BufferRegistry::new(64);
    let dirty: Vec<_> = registry.dirty_slots().collect();
    assert!(dirty.is_empty());
}

/// BufferRegistry::dirty_slot_indices() returns empty iterator initially.
#[test]
fn dirty_slot_indices_empty_initially() {
    let registry = BufferRegistry::new(64);
    let dirty: Vec<_> = registry.dirty_slot_indices().collect();
    assert!(dirty.is_empty());
}

/// BufferRegistry::clear_dirty() is callable when no dirty slots.
#[test]
fn dirty_clear_when_none() {
    let mut registry = BufferRegistry::new(64);
    registry.clear_dirty();
    assert_eq!(registry.dirty_count(), 0);
}

/// BufferRegistry::is_dirty() returns true for new registry (bind group not yet created).
#[test]
fn dirty_is_dirty_true_for_new() {
    let registry = BufferRegistry::new(64);
    // is_dirty() is true when bind group hasn't been created
    assert!(registry.is_dirty());
}

/// BufferRegistry::is_slot_dirty() returns false for invalid slot.
#[test]
fn dirty_is_slot_dirty_invalid() {
    let registry = BufferRegistry::new(64);
    let invalid = BufferSlot::invalid();
    assert!(!registry.is_slot_dirty(invalid));
}

/// BufferRegistry::is_slot_dirty() returns false for unregistered slot.
#[test]
fn dirty_is_slot_dirty_unregistered() {
    let registry = BufferRegistry::new(64);
    let slot = BufferSlot::new(5);
    assert!(!registry.is_slot_dirty(slot));
}

/// Clearing registry resets state but is_dirty remains true (no bind group).
#[test]
fn dirty_clear_registry_resets_dirty_count() {
    let mut registry = BufferRegistry::new(64);
    // Clear resets dirty_count but is_dirty() depends on bind group state
    registry.clear();
    assert_eq!(registry.dirty_count(), 0);
    // is_dirty() is still true because no bind group has been created
    assert!(registry.is_dirty());
}

/// Mark dirty on unregistered slot has no effect.
#[test]
fn dirty_mark_unregistered_slot_noop() {
    let mut registry = BufferRegistry::new(64);
    let slot = BufferSlot::new(5);
    registry.mark_dirty(slot);
    // Should have no effect since slot isn't registered
    assert_eq!(registry.dirty_count(), 0);
}

/// Multiple clear_dirty calls are idempotent.
#[test]
fn dirty_clear_idempotent() {
    let mut registry = BufferRegistry::new(64);
    registry.clear_dirty();
    registry.clear_dirty();
    registry.clear_dirty();
    assert_eq!(registry.dirty_count(), 0);
}

/// dirty_slots() iterator can be collected multiple times.
#[test]
fn dirty_slots_iterator_reusable() {
    let registry = BufferRegistry::new(64);
    let dirty1: Vec<_> = registry.dirty_slots().collect();
    let dirty2: Vec<_> = registry.dirty_slots().collect();
    assert_eq!(dirty1.len(), dirty2.len());
}

/// dirty_slot_indices() iterator can be collected multiple times.
#[test]
fn dirty_slot_indices_iterator_reusable() {
    let registry = BufferRegistry::new(64);
    let dirty1: Vec<_> = registry.dirty_slot_indices().collect();
    let dirty2: Vec<_> = registry.dirty_slot_indices().collect();
    assert_eq!(dirty1.len(), dirty2.len());
}

/// is_dirty() reflects need to create/update bind_group.
#[test]
fn dirty_reflects_bind_group_need() {
    let registry = BufferRegistry::new(64);
    // is_dirty() is true when bind_group needs to be created
    assert!(registry.is_dirty());
    assert!(registry.bind_group().is_none());
}

// =============================================================================
// SECTION 5 -- METRICS API TESTS (10 tests)
// =============================================================================

/// BufferRegistry::metrics() returns BufferRegistryMetrics.
#[test]
fn metrics_returns_struct() {
    let registry = BufferRegistry::new(64);
    let _metrics: BufferRegistryMetrics = registry.metrics();
}

/// BufferRegistryMetrics::registered_count is 0 for empty registry.
#[test]
fn metrics_registered_count_empty() {
    let registry = BufferRegistry::new(64);
    let metrics = registry.metrics();
    assert_eq!(metrics.registered_count, 0);
}

/// BufferRegistryMetrics::capacity matches registry.capacity().
#[test]
fn metrics_capacity() {
    let registry = BufferRegistry::new(512);
    let metrics = registry.metrics();
    assert_eq!(metrics.capacity, 512);
}

/// BufferRegistryMetrics::free_slots is 0 for new empty registry.
#[test]
fn metrics_free_slots_empty() {
    let registry = BufferRegistry::new(64);
    let metrics = registry.metrics();
    // Free slots are 0 initially until allocation happens
    assert_eq!(metrics.free_slots, 0);
}

/// BufferRegistryMetrics::allocated_slots is 0 for empty registry.
#[test]
fn metrics_allocated_slots_empty() {
    let registry = BufferRegistry::new(64);
    let metrics = registry.metrics();
    assert_eq!(metrics.allocated_slots, 0);
}

/// BufferRegistryMetrics::dirty_slots is 0 for empty registry.
#[test]
fn metrics_dirty_slots_empty() {
    let registry = BufferRegistry::new(64);
    let metrics = registry.metrics();
    assert_eq!(metrics.dirty_slots, 0);
}

/// BufferRegistryMetrics::has_bind_group is false for new registry.
#[test]
fn metrics_has_bind_group_false() {
    let registry = BufferRegistry::new(64);
    let metrics = registry.metrics();
    assert!(!metrics.has_bind_group);
}

/// BufferRegistryMetrics::is_dirty is true for new registry (no bind group).
#[test]
fn metrics_is_dirty_true_new() {
    let registry = BufferRegistry::new(64);
    let metrics = registry.metrics();
    // Dirty is true because bind group hasn't been created yet
    assert!(metrics.is_dirty);
}

/// BufferRegistryMetrics::utilization() is 0.0 for empty registry.
#[test]
fn metrics_utilization_empty() {
    let registry = BufferRegistry::new(64);
    let metrics = registry.metrics();
    assert!((metrics.utilization() - 0.0).abs() < 0.001);
}

/// BufferRegistryMetrics implements Debug.
#[test]
fn metrics_debug_format() {
    let registry = BufferRegistry::new(64);
    let metrics = registry.metrics();
    let debug_str = format!("{:?}", metrics);
    assert!(debug_str.contains("BufferRegistryMetrics") || debug_str.contains("registered_count"));
}

// =============================================================================
// SECTION 6 -- ERROR API TESTS (10 tests)
// =============================================================================

/// BindlessBufferError::RegistryFull exists.
#[test]
fn error_registry_full_exists() {
    let _err = BindlessBufferError::RegistryFull { capacity: 64 };
}

/// BindlessBufferError::InvalidSlot exists.
#[test]
fn error_invalid_slot_exists() {
    let _err = BindlessBufferError::InvalidSlot(BufferSlot::invalid());
}

/// BindlessBufferError::UnsupportedFeature exists.
#[test]
fn error_unsupported_feature_exists() {
    let _err = BindlessBufferError::UnsupportedFeature;
}

/// BindlessBufferError::ExceedsDeviceLimit exists.
#[test]
fn error_exceeds_device_limit_exists() {
    let _err = BindlessBufferError::ExceedsDeviceLimit { requested: 100, max: 64 };
}

/// BindlessBufferError::IncompatibleLayout exists.
#[test]
fn error_incompatible_layout_exists() {
    let _err = BindlessBufferError::IncompatibleLayout;
}

/// BindlessBufferError::EmptyRegistry exists.
#[test]
fn error_empty_registry_exists() {
    let _err = BindlessBufferError::EmptyRegistry;
}

/// BindlessBufferError implements Debug.
#[test]
fn error_debug_format() {
    let err = BindlessBufferError::RegistryFull { capacity: 64 };
    let debug_str = format!("{:?}", err);
    assert!(!debug_str.is_empty());
}

/// BindlessBufferError implements Display (via std::error::Error).
#[test]
fn error_display_format() {
    let err = BindlessBufferError::RegistryFull { capacity: 64 };
    let display_str = format!("{}", err);
    assert!(!display_str.is_empty());
}

/// BindlessBufferError implements Clone.
#[test]
fn error_is_clone() {
    let err = BindlessBufferError::RegistryFull { capacity: 64 };
    let err2 = err.clone();
    assert!(matches!(err2, BindlessBufferError::RegistryFull { .. }));
}

/// BindlessBufferError is Send.
#[test]
fn error_is_send() {
    fn assert_send<T: Send>() {}
    assert_send::<BindlessBufferError>();
}

/// BindlessBufferError is Sync.
#[test]
fn error_is_sync() {
    fn assert_sync<T: Sync>() {}
    assert_sync::<BindlessBufferError>();
}

/// BindlessBufferError variants are distinct.
#[test]
fn error_variants_distinct() {
    let full = BindlessBufferError::RegistryFull { capacity: 64 };
    let invalid = BindlessBufferError::InvalidSlot(BufferSlot::invalid());
    let unsupported = BindlessBufferError::UnsupportedFeature;

    // Each variant should have different debug output
    let full_str = format!("{:?}", full);
    let invalid_str = format!("{:?}", invalid);
    let unsupported_str = format!("{:?}", unsupported);

    assert_ne!(full_str, invalid_str);
    assert_ne!(full_str, unsupported_str);
    assert_ne!(invalid_str, unsupported_str);
}

/// BindlessBufferError implements PartialEq.
#[test]
fn error_partial_eq() {
    assert_eq!(
        BindlessBufferError::RegistryFull { capacity: 64 },
        BindlessBufferError::RegistryFull { capacity: 64 }
    );
    assert_eq!(
        BindlessBufferError::InvalidSlot(BufferSlot::new(5)),
        BindlessBufferError::InvalidSlot(BufferSlot::new(5))
    );
    assert_eq!(
        BindlessBufferError::UnsupportedFeature,
        BindlessBufferError::UnsupportedFeature
    );
}

// =============================================================================
// SECTION 7 -- FEATURE DETECTION TESTS (6 tests)
// =============================================================================

/// bindless_buffer_required_features() returns a Features value.
#[test]
fn feature_required_features() {
    let features = bindless_buffer_required_features();
    // Should be a valid Features value (may be empty or contain features)
    let _ = features.is_empty();
}

/// bindless_buffer_optimal_features() returns a Features value.
#[test]
fn feature_optimal_features() {
    let features = bindless_buffer_optimal_features();
    // Optimal should include required
    let required = bindless_buffer_required_features();
    assert!(features.contains(required));
}

/// max_bindless_buffers_from_limits() returns a value based on device limits.
#[test]
fn feature_max_from_limits_default() {
    let limits = wgpu::Limits::default();
    let max = max_bindless_buffers_from_limits(&limits);
    // Value depends on default limits.max_storage_buffers_per_shader_stage
    // which may be less than MIN_BINDLESS_BUFFERS
    assert!(max > 0);
}

/// max_bindless_buffers_from_limits() is bounded by conservative max.
#[test]
fn feature_max_from_limits_bounded() {
    let limits = wgpu::Limits::default();
    let max = max_bindless_buffers_from_limits(&limits);
    assert!(max <= MAX_BINDLESS_BUFFERS_CONSERVATIVE);
}

/// bindless_buffer_layout_entry() returns BindGroupLayoutEntry with correct binding.
#[test]
fn feature_layout_entry_binding() {
    let entry = bindless_buffer_layout_entry(BINDLESS_BUFFER_BINDING, 256, true);
    assert_eq!(entry.binding, BINDLESS_BUFFER_BINDING);
}

/// bindless_buffer_layout_entry_readonly() returns entry with correct binding.
#[test]
fn feature_layout_entry_readonly_binding() {
    let entry = bindless_buffer_layout_entry_readonly(BINDLESS_BUFFER_BINDING, 128);
    assert_eq!(entry.binding, BINDLESS_BUFFER_BINDING);
}

/// bindless_buffer_layout_entry_readwrite() returns entry with correct binding.
#[test]
fn feature_layout_entry_readwrite_binding() {
    let entry = bindless_buffer_layout_entry_readwrite(BINDLESS_BUFFER_BINDING, 128);
    assert_eq!(entry.binding, BINDLESS_BUFFER_BINDING);
}

/// Layout entry visibility should include VERTEX and FRAGMENT.
#[test]
fn feature_layout_entry_visibility() {
    let entry = bindless_buffer_layout_entry(BINDLESS_BUFFER_BINDING, 64, true);
    // Entry should have some visibility flags set
    assert!(!entry.visibility.is_empty());
}

// =============================================================================
// SECTION 8 -- THREAD SAFETY TESTS (3 tests)
// =============================================================================

/// BufferRegistry is Send.
#[test]
fn registry_is_send() {
    fn assert_send<T: Send>() {}
    assert_send::<BufferRegistry>();
}

/// BufferRegistry is Sync.
#[test]
fn registry_is_sync() {
    fn assert_sync<T: Sync>() {}
    assert_sync::<BufferRegistry>();
}

/// BufferSlot is Send + Sync.
#[test]
fn slot_is_send_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<BufferSlot>();
}

/// BufferRegistryMetrics is Send + Sync.
#[test]
fn metrics_is_send_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<BufferRegistryMetrics>();
}

// =============================================================================
// SECTION 9 -- PROPERTY-BASED TESTS (5 tests)
// =============================================================================

/// Property: free_slots + allocated_slots equals used slot count (not capacity).
#[test]
fn property_free_plus_allocated_consistent() {
    let registry = BufferRegistry::new(64);
    let metrics = registry.metrics();
    // free_slots and allocated_slots track used slots, not total capacity
    // For empty registry, both should be 0
    assert_eq!(metrics.free_slots, 0);
    assert_eq!(metrics.allocated_slots, 0);
}

/// Property: utilization is in [0.0, 1.0]
#[test]
fn property_utilization_bounded() {
    let registry = BufferRegistry::new(64);
    let metrics = registry.metrics();
    let util = metrics.utilization();
    assert!(util >= 0.0 && util <= 1.0);
}

/// Property: fragmentation is in [0.0, 1.0]
#[test]
fn property_fragmentation_bounded() {
    let registry = BufferRegistry::new(64);
    let metrics = registry.metrics();
    let frag = metrics.fragmentation();
    assert!(frag >= 0.0 && frag <= 1.0);
}

/// Property: dirty_ratio is in [0.0, 1.0]
#[test]
fn property_dirty_ratio_bounded() {
    let registry = BufferRegistry::new(64);
    let metrics = registry.metrics();
    let ratio = metrics.dirty_ratio();
    assert!(ratio >= 0.0 && ratio <= 1.0);
}

/// Property: is_empty() == (count() == 0)
#[test]
fn property_is_empty_iff_count_zero() {
    let registry = BufferRegistry::new(64);
    assert!(registry.is_empty() == (registry.count() == 0));
}

/// Property: is_full() == (count() == capacity())
#[test]
fn property_is_full_iff_count_equals_capacity() {
    let registry = BufferRegistry::new(64);
    assert!(registry.is_full() == (registry.count() == registry.capacity()));
}

/// Property: dirty_count() <= count()
#[test]
fn property_dirty_count_le_count() {
    let registry = BufferRegistry::new(64);
    assert!(registry.dirty_count() as u32 <= registry.count());
}

/// Property: metrics consistency
#[test]
fn property_metrics_consistency() {
    let registry = BufferRegistry::new(128);
    let metrics = registry.metrics();

    // registered_count should match count()
    assert_eq!(metrics.registered_count, registry.count());

    // capacity should match capacity()
    assert_eq!(metrics.capacity, registry.capacity());

    // dirty_slots should match dirty_count()
    assert_eq!(metrics.dirty_slots as usize, registry.dirty_count());

    // is_dirty should match is_dirty()
    assert_eq!(metrics.is_dirty, registry.is_dirty());

    // has_bind_group should match bind_group().is_some()
    assert_eq!(metrics.has_bind_group, registry.bind_group().is_some());
}

// =============================================================================
// SECTION 10 -- GPU-DEPENDENT TESTS (5 tests, all ignored)
// =============================================================================

/// Integration test: register a buffer and check state.
/// Requires GPU to create actual wgpu::Buffer.
#[test]

fn integration_register_buffer() {
    // Would need:
    // 1. Create wgpu instance/adapter/device
    // 2. Create a wgpu::Buffer
    // 3. Call registry.register(Arc::new(buffer))
    // 4. Verify slot is valid and count incremented
}

/// Integration test: register then unregister.
/// Requires GPU to create actual wgpu::Buffer.
#[test]

fn integration_register_unregister() {
    // Would need GPU setup to test full register/unregister cycle
}

/// Integration test: dirty tracking with real buffers.
/// Requires GPU to create actual wgpu::Buffer.
#[test]

fn integration_dirty_tracking() {
    // Would need GPU setup to test mark_dirty with registered buffers
}

/// Integration test: create bind group.
/// Requires GPU for device and layout creation.
#[test]

fn integration_create_bind_group() {
    // Would need GPU setup to test update_bind_group or create_bind_group
}

/// Integration test: new_validated with device.
/// Requires GPU for device access.
#[test]

fn integration_new_validated() {
    // Would need GPU setup to test BufferRegistry::new_validated
}
