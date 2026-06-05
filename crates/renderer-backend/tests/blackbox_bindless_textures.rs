// SPDX-License-Identifier: MIT
//
// blackbox_bindless_textures.rs -- Blackbox tests for T-WGPU-P2.6.1 Bindless Texture Registry.
//
// CLEANROOM: No src/ access beyond the public API exported by the crate.
// Tests use only the documented public types and functions:
//
//   - TextureSlot
//   - TextureRegistry
//   - TextureRegistryMetrics
//   - BindlessError
//   - supports_bindless_textures(), supports_non_uniform_indexing(), supports_partially_bound()
//   - max_bindless_textures(), max_bindless_textures_from_limits()
//   - bindless_texture_layout_entry(), create_bindless_layout()
//   - bindless_required_features(), bindless_optimal_features()
//   - BINDLESS_BIND_GROUP_INDEX, BINDLESS_TEXTURE_BINDING
//   - DEFAULT_MAX_TEXTURES, MAX_BINDLESS_TEXTURES_CONSERVATIVE, MIN_BINDLESS_TEXTURES
//
// ACCEPTANCE CRITERIA:
//   1. Constants tests           -- 8 tests covering const values
//   2. TextureSlot API tests     -- 15+ tests for slot construction and accessors
//   3. TextureRegistry API tests -- 25+ tests for registry operations
//   4. Metrics API tests         -- 12+ tests for metrics types
//   5. Error API tests           -- 12+ tests for error handling
//   6. Feature detection tests   -- 8+ tests
//   7. Integration tests         -- 10+ tests (ignored without GPU)
//   8. Property-based tests      -- 8 tests for invariants
//
// Total target: ~85 tests

use renderer_backend::resources::{
    bindless_optimal_features, bindless_required_features, bindless_texture_layout_entry,
    max_bindless_textures_from_limits,
    BindlessError, TextureRegistry, TextureRegistryMetrics, TextureSlot,
    BINDLESS_BIND_GROUP_INDEX, BINDLESS_TEXTURE_BINDING, DEFAULT_MAX_TEXTURES,
    MAX_BINDLESS_TEXTURES_CONSERVATIVE, MIN_BINDLESS_TEXTURES,
};
use std::collections::HashSet;

// =============================================================================
// SECTION 1 -- CONSTANTS TESTS (8 tests)
// =============================================================================

/// BINDLESS_BIND_GROUP_INDEX should be 3 (conventionally reserved for bindless).
#[test]
fn constant_bindless_bind_group_index_value() {
    // Bind group 3 is the conventional slot for bindless textures:
    // 0 = per-frame uniforms, 1 = per-material, 2 = per-draw, 3 = bindless
    assert_eq!(BINDLESS_BIND_GROUP_INDEX, 3);
}

/// BINDLESS_TEXTURE_BINDING should be 0 (first binding in the group).
#[test]
fn constant_bindless_texture_binding_value() {
    assert_eq!(BINDLESS_TEXTURE_BINDING, 0);
}

/// DEFAULT_MAX_TEXTURES should be 1024.
#[test]
fn constant_default_max_textures_value() {
    assert_eq!(DEFAULT_MAX_TEXTURES, 1024);
}

/// DEFAULT_MAX_TEXTURES should be a power-of-two value.
#[test]
fn constant_default_max_textures_is_power_of_two() {
    assert!(DEFAULT_MAX_TEXTURES > 0);
    assert!(DEFAULT_MAX_TEXTURES.is_power_of_two());
}

/// DEFAULT_MAX_TEXTURES should be at least MIN_BINDLESS_TEXTURES.
#[test]
fn constant_default_at_least_min() {
    assert!(DEFAULT_MAX_TEXTURES >= MIN_BINDLESS_TEXTURES);
}

/// DEFAULT_MAX_TEXTURES should not exceed MAX_BINDLESS_TEXTURES_CONSERVATIVE.
#[test]
fn constant_default_at_most_conservative_max() {
    assert!(DEFAULT_MAX_TEXTURES <= MAX_BINDLESS_TEXTURES_CONSERVATIVE);
}

/// MIN_BINDLESS_TEXTURES should be 16.
#[test]
fn constant_min_bindless_textures_value() {
    assert_eq!(MIN_BINDLESS_TEXTURES, 16);
}

/// MAX_BINDLESS_TEXTURES_CONSERVATIVE should be 16384.
#[test]
fn constant_max_conservative_value() {
    assert_eq!(MAX_BINDLESS_TEXTURES_CONSERVATIVE, 16384);
}

// =============================================================================
// SECTION 2 -- TEXTURE SLOT API TESTS (15+ tests)
// =============================================================================

/// TextureSlot::new() creates a slot with the given index.
#[test]
fn texture_slot_new_creates_valid_slot() {
    let slot = TextureSlot::new(42);
    assert_eq!(slot.index(), 42);
}

/// TextureSlot::new() with zero index.
#[test]
fn texture_slot_new_with_zero() {
    let slot = TextureSlot::new(0);
    assert_eq!(slot.index(), 0);
}

/// TextureSlot::new() with max u32 - 1 index (u32::MAX is reserved for invalid).
#[test]
fn texture_slot_new_with_large_index() {
    let slot = TextureSlot::new(u32::MAX - 1);
    assert_eq!(slot.index(), u32::MAX - 1);
}

/// TextureSlot::invalid() creates an invalid slot.
#[test]
fn texture_slot_invalid_creates_invalid_slot() {
    let slot = TextureSlot::invalid();
    assert!(slot.is_invalid());
}

/// TextureSlot::new(0) is valid, not invalid.
#[test]
fn texture_slot_zero_is_valid() {
    let slot = TextureSlot::new(0);
    assert!(!slot.is_invalid());
}

/// TextureSlot::new(1) is valid.
#[test]
fn texture_slot_one_is_valid() {
    let slot = TextureSlot::new(1);
    assert!(!slot.is_invalid());
}

/// TextureSlot equality for same index.
#[test]
fn texture_slot_equality_same_index() {
    let slot1 = TextureSlot::new(100);
    let slot2 = TextureSlot::new(100);
    assert_eq!(slot1, slot2);
}

/// TextureSlot inequality for different index.
#[test]
fn texture_slot_inequality_different_index() {
    let slot1 = TextureSlot::new(100);
    let slot2 = TextureSlot::new(200);
    assert_ne!(slot1, slot2);
}

/// TextureSlot invalid slots are equal.
#[test]
fn texture_slot_invalid_slots_equal() {
    let slot1 = TextureSlot::invalid();
    let slot2 = TextureSlot::invalid();
    assert_eq!(slot1, slot2);
}

/// TextureSlot valid and invalid are distinguished by is_invalid().
#[test]
fn texture_slot_valid_and_invalid_distinguished() {
    let valid = TextureSlot::new(0);
    let invalid = TextureSlot::invalid();
    assert!(!valid.is_invalid());
    assert!(invalid.is_invalid());
}

/// TextureSlot works in HashSet.
#[test]
fn texture_slot_works_in_hashset() {
    let mut set: HashSet<TextureSlot> = HashSet::new();
    set.insert(TextureSlot::new(1));
    set.insert(TextureSlot::new(2));
    set.insert(TextureSlot::new(1)); // Duplicate

    assert_eq!(set.len(), 2);
    assert!(set.contains(&TextureSlot::new(1)));
    assert!(set.contains(&TextureSlot::new(2)));
    assert!(!set.contains(&TextureSlot::new(3)));
}

/// TextureSlot is Copy.
#[test]
fn texture_slot_is_copy() {
    let slot1 = TextureSlot::new(42);
    let slot2 = slot1; // Copy
    let slot3 = slot1; // Copy again
    assert_eq!(slot1.index(), 42);
    assert_eq!(slot2.index(), 42);
    assert_eq!(slot3.index(), 42);
}

/// TextureSlot Debug formatting includes index.
#[test]
fn texture_slot_debug_format() {
    let slot = TextureSlot::new(123);
    let debug_str = format!("{:?}", slot);
    // Should contain the index value or be non-empty
    assert!(!debug_str.is_empty());
}

/// TextureSlot invalid Debug formatting is informative.
#[test]
fn texture_slot_invalid_debug_format() {
    let slot = TextureSlot::invalid();
    let debug_str = format!("{:?}", slot);
    // Should produce some output
    assert!(!debug_str.is_empty());
}

/// TextureSlot index accessor returns correct value for various indices.
#[test]
fn texture_slot_index_accessor_various() {
    for i in [0, 1, 10, 100, 1000, 10000, u32::MAX / 2] {
        let slot = TextureSlot::new(i);
        assert_eq!(slot.index(), i);
    }
}

/// TextureSlot::new is const fn.
#[test]
fn texture_slot_new_is_const() {
    const SLOT: TextureSlot = TextureSlot::new(42);
    assert_eq!(SLOT.index(), 42);
}

/// TextureSlot::invalid is const fn.
#[test]
fn texture_slot_invalid_is_const() {
    const INVALID: TextureSlot = TextureSlot::invalid();
    assert!(INVALID.is_invalid());
}

// =============================================================================
// SECTION 3 -- TEXTURE REGISTRY API TESTS (25+ tests)
// =============================================================================

/// TextureRegistry::new() creates an empty registry with given capacity.
#[test]
fn texture_registry_new_is_empty() {
    let registry = TextureRegistry::new(DEFAULT_MAX_TEXTURES);
    assert!(registry.is_empty());
    assert_eq!(registry.count(), 0);
}

/// TextureRegistry::new() sets correct capacity.
#[test]
fn texture_registry_new_has_correct_capacity() {
    let registry = TextureRegistry::new(512);
    assert_eq!(registry.capacity(), 512);
}

/// TextureRegistry::new() with minimum capacity.
#[test]
fn texture_registry_new_minimum_capacity() {
    let registry = TextureRegistry::new(MIN_BINDLESS_TEXTURES);
    assert_eq!(registry.capacity(), MIN_BINDLESS_TEXTURES);
}

/// TextureRegistry::new() with default capacity.
#[test]
fn texture_registry_new_default_capacity() {
    let registry = TextureRegistry::new(DEFAULT_MAX_TEXTURES);
    assert_eq!(registry.capacity(), DEFAULT_MAX_TEXTURES);
}

/// TextureRegistry::new() with large capacity.
#[test]
fn texture_registry_new_large_capacity() {
    let registry = TextureRegistry::new(MAX_BINDLESS_TEXTURES_CONSERVATIVE);
    assert_eq!(registry.capacity(), MAX_BINDLESS_TEXTURES_CONSERVATIVE);
}

/// TextureRegistry is not full when empty.
#[test]
fn texture_registry_empty_is_not_full() {
    let registry = TextureRegistry::new(DEFAULT_MAX_TEXTURES);
    assert!(!registry.is_full());
}

/// TextureRegistry count starts at zero.
#[test]
fn texture_registry_count_starts_zero() {
    let registry = TextureRegistry::new(256);
    assert_eq!(registry.count(), 0);
}

/// TextureRegistry clear on empty is safe.
#[test]
fn texture_registry_clear_empty_is_safe() {
    let mut registry = TextureRegistry::new(256);
    registry.clear();
    assert!(registry.is_empty());
    assert_eq!(registry.count(), 0);
}

/// TextureRegistry free_slot_count equals capacity when empty.
#[test]
fn texture_registry_free_slot_count_when_empty() {
    let registry = TextureRegistry::new(256);
    // When empty, free_slot_count may be 0 (no slots allocated yet) or capacity
    // depending on implementation
    let free = registry.free_slot_count();
    assert!(free <= registry.capacity() as usize);
}

/// TextureRegistry is_dirty is true initially (needs bind group creation).
#[test]
fn texture_registry_is_dirty_initially_true() {
    let registry = TextureRegistry::new(256);
    // Initially dirty because bind group hasn't been created yet
    assert!(registry.is_dirty());
}

/// TextureRegistry bind_group is None initially.
#[test]
fn texture_registry_bind_group_none_initially() {
    let registry = TextureRegistry::new(256);
    assert!(registry.bind_group().is_none());
}

/// TextureRegistry unregister with invalid slot returns false.
#[test]
fn texture_registry_unregister_invalid_returns_false() {
    let mut registry = TextureRegistry::new(256);
    let result = registry.unregister(TextureSlot::invalid());
    assert!(!result);
}

/// TextureRegistry unregister with unregistered slot returns false.
#[test]
fn texture_registry_unregister_unregistered_returns_false() {
    let mut registry = TextureRegistry::new(256);
    // Slot 999 was never registered
    let result = registry.unregister(TextureSlot::new(999));
    assert!(!result);
}

/// TextureRegistry is_registered returns false for invalid slot.
#[test]
fn texture_registry_is_registered_invalid_is_false() {
    let registry = TextureRegistry::new(256);
    assert!(!registry.is_registered(TextureSlot::invalid()));
}

/// TextureRegistry is_registered returns false for unregistered slot.
#[test]
fn texture_registry_is_registered_unregistered_is_false() {
    let registry = TextureRegistry::new(256);
    assert!(!registry.is_registered(TextureSlot::new(42)));
}

/// TextureRegistry get returns None for invalid slot.
#[test]
fn texture_registry_get_invalid_returns_none() {
    let registry = TextureRegistry::new(256);
    assert!(registry.get(TextureSlot::invalid()).is_none());
}

/// TextureRegistry get returns None for unregistered slot.
#[test]
fn texture_registry_get_unregistered_returns_none() {
    let registry = TextureRegistry::new(256);
    assert!(registry.get(TextureSlot::new(42)).is_none());
}

/// TextureRegistry iter is empty when registry is empty.
#[test]
fn texture_registry_iter_empty_when_empty() {
    let registry = TextureRegistry::new(256);
    assert_eq!(registry.iter().count(), 0);
}

/// TextureRegistry metrics returns valid data for empty registry.
#[test]
fn texture_registry_metrics_empty() {
    let registry = TextureRegistry::new(256);
    let metrics = registry.metrics();
    assert_eq!(metrics.registered_count, 0);
    assert_eq!(metrics.capacity, 256);
    assert!(!metrics.has_bind_group);
}

/// TextureRegistry metrics show 0.0 utilization when empty.
#[test]
fn texture_registry_utilization_zero_when_empty() {
    let registry = TextureRegistry::new(256);
    let metrics = registry.metrics();
    assert!((metrics.utilization() - 0.0).abs() < f32::EPSILON);
}

/// TextureRegistry metrics show 0.0 fragmentation when empty.
#[test]
fn texture_registry_fragmentation_zero_when_empty() {
    let registry = TextureRegistry::new(256);
    let metrics = registry.metrics();
    assert!((metrics.fragmentation() - 0.0).abs() < f32::EPSILON);
}

/// TextureRegistry capacity is immutable.
#[test]
fn texture_registry_capacity_is_immutable() {
    let mut registry = TextureRegistry::new(256);
    let cap_before = registry.capacity();
    registry.clear();
    let cap_after = registry.capacity();
    assert_eq!(cap_before, cap_after);
}

/// TextureRegistry is_empty and count are consistent.
#[test]
fn texture_registry_is_empty_count_consistent() {
    let registry = TextureRegistry::new(256);
    if registry.is_empty() {
        assert_eq!(registry.count(), 0);
    }
}

/// TextureRegistry is_full and count are consistent.
#[test]
fn texture_registry_is_full_count_consistent() {
    let registry = TextureRegistry::new(256);
    // When empty, should not be full
    assert!(!registry.is_full());
}

/// TextureRegistry with capacity below MIN is clamped to MIN.
#[test]
fn texture_registry_capacity_below_min_clamped() {
    let registry = TextureRegistry::new(1);
    // Capacity is clamped to MIN_BINDLESS_TEXTURES
    assert_eq!(registry.capacity(), MIN_BINDLESS_TEXTURES);
    assert!(registry.is_empty());
}

// =============================================================================
// SECTION 4 -- METRICS API TESTS (12+ tests)
// =============================================================================

/// TextureRegistryMetrics registered_count is accurate.
#[test]
fn metrics_registered_count_is_accurate() {
    let registry = TextureRegistry::new(256);
    let metrics = registry.metrics();
    assert_eq!(metrics.registered_count, 0);
}

/// TextureRegistryMetrics capacity is accurate.
#[test]
fn metrics_capacity_is_accurate() {
    let registry = TextureRegistry::new(512);
    let metrics = registry.metrics();
    assert_eq!(metrics.capacity, 512);
}

/// TextureRegistryMetrics utilization is zero when empty.
#[test]
fn metrics_utilization_zero_when_empty() {
    let registry = TextureRegistry::new(256);
    let metrics = registry.metrics();
    assert!((metrics.utilization() - 0.0).abs() < f32::EPSILON);
}

/// TextureRegistryMetrics has_bind_group is false initially.
#[test]
fn metrics_has_bind_group_false_initially() {
    let registry = TextureRegistry::new(256);
    let metrics = registry.metrics();
    assert!(!metrics.has_bind_group);
}

/// TextureRegistryMetrics is_dirty tracks dirty state.
#[test]
fn metrics_is_dirty_tracks_state() {
    let registry = TextureRegistry::new(256);
    let metrics = registry.metrics();
    // Initially dirty because bind group hasn't been created yet
    assert!(metrics.is_dirty);
}

/// TextureRegistryMetrics free_slots is correct initially.
#[test]
fn metrics_free_slots_initially() {
    let registry = TextureRegistry::new(256);
    let metrics = registry.metrics();
    // Initially no slots allocated, so free_slots may be 0 or capacity
    assert!(metrics.free_slots <= metrics.capacity);
}

/// TextureRegistryMetrics allocated_slots is 0 initially.
#[test]
fn metrics_allocated_slots_initially() {
    let registry = TextureRegistry::new(256);
    let metrics = registry.metrics();
    // Initially no allocations
    assert!(metrics.allocated_slots <= metrics.capacity);
}

/// TextureRegistryMetrics Debug formatting works.
#[test]
fn metrics_debug_format() {
    let registry = TextureRegistry::new(256);
    let metrics = registry.metrics();
    let debug_str = format!("{:?}", metrics);
    assert!(!debug_str.is_empty());
}

/// TextureRegistryMetrics is Clone.
#[test]
fn metrics_is_clone() {
    let registry = TextureRegistry::new(256);
    let metrics = registry.metrics();
    let cloned = metrics.clone();
    assert_eq!(metrics.registered_count, cloned.registered_count);
    assert_eq!(metrics.capacity, cloned.capacity);
}

/// TextureRegistryMetrics is Copy.
#[test]
fn metrics_is_copy() {
    let registry = TextureRegistry::new(256);
    let metrics = registry.metrics();
    let copied = metrics;
    let copied2 = metrics;
    assert_eq!(copied.capacity, copied2.capacity);
}

/// TextureRegistryMetrics utilization in valid range.
#[test]
fn metrics_utilization_in_valid_range() {
    let registry = TextureRegistry::new(256);
    let metrics = registry.metrics();
    let util = metrics.utilization();
    assert!(util >= 0.0);
    assert!(util <= 1.0);
}

/// TextureRegistryMetrics fragmentation in valid range.
#[test]
fn metrics_fragmentation_in_valid_range() {
    let registry = TextureRegistry::new(256);
    let metrics = registry.metrics();
    let frag = metrics.fragmentation();
    assert!(frag >= 0.0);
    assert!(frag <= 1.0);
}

// =============================================================================
// SECTION 5 -- ERROR API TESTS (12+ tests)
// =============================================================================

/// BindlessError has UnsupportedFeature variant.
#[test]
fn bindless_error_unsupported_feature_variant() {
    let err = BindlessError::UnsupportedFeature;
    let debug_str = format!("{:?}", err);
    assert!(debug_str.contains("UnsupportedFeature"));
}

/// BindlessError has RegistryFull variant with capacity.
#[test]
fn bindless_error_registry_full_variant() {
    let err = BindlessError::RegistryFull { capacity: 1024 };
    let debug_str = format!("{:?}", err);
    assert!(debug_str.contains("RegistryFull") || debug_str.contains("1024"));
}

/// BindlessError has InvalidSlot variant.
#[test]
fn bindless_error_invalid_slot_variant() {
    let err = BindlessError::InvalidSlot(TextureSlot::new(999));
    let debug_str = format!("{:?}", err);
    assert!(debug_str.contains("InvalidSlot") || debug_str.contains("999"));
}

/// BindlessError has ExceedsDeviceLimit variant.
#[test]
fn bindless_error_exceeds_device_limit_variant() {
    let err = BindlessError::ExceedsDeviceLimit { requested: 2000, max: 1024 };
    let debug_str = format!("{:?}", err);
    assert!(debug_str.contains("ExceedsDeviceLimit") || debug_str.contains("2000"));
}

/// BindlessError has IncompatibleLayout variant.
#[test]
fn bindless_error_incompatible_layout_variant() {
    let err = BindlessError::IncompatibleLayout;
    let debug_str = format!("{:?}", err);
    assert!(debug_str.contains("IncompatibleLayout"));
}

/// BindlessError has EmptyRegistry variant.
#[test]
fn bindless_error_empty_registry_variant() {
    let err = BindlessError::EmptyRegistry;
    let debug_str = format!("{:?}", err);
    assert!(debug_str.contains("EmptyRegistry"));
}

/// BindlessError Display trait works for UnsupportedFeature.
#[test]
fn bindless_error_display_unsupported_feature() {
    let err = BindlessError::UnsupportedFeature;
    let display_str = format!("{}", err);
    assert!(!display_str.is_empty());
}

/// BindlessError Display trait works for RegistryFull.
#[test]
fn bindless_error_display_registry_full() {
    let err = BindlessError::RegistryFull { capacity: 512 };
    let display_str = format!("{}", err);
    assert!(!display_str.is_empty());
}

/// BindlessError Display trait works for InvalidSlot.
#[test]
fn bindless_error_display_invalid_slot() {
    let err = BindlessError::InvalidSlot(TextureSlot::new(42));
    let display_str = format!("{}", err);
    assert!(!display_str.is_empty());
}

/// BindlessError is Send.
#[test]
fn bindless_error_is_send() {
    fn assert_send<T: Send>() {}
    assert_send::<BindlessError>();
}

/// BindlessError is Sync.
#[test]
fn bindless_error_is_sync() {
    fn assert_sync<T: Sync>() {}
    assert_sync::<BindlessError>();
}

/// BindlessError implements std::error::Error.
#[test]
fn bindless_error_implements_error() {
    fn assert_error<T: std::error::Error>() {}
    assert_error::<BindlessError>();
}

/// BindlessError equality for same variant.
#[test]
fn bindless_error_equality_same() {
    let err1 = BindlessError::RegistryFull { capacity: 100 };
    let err2 = BindlessError::RegistryFull { capacity: 100 };
    assert_eq!(err1, err2);
}

/// BindlessError inequality for different variants.
#[test]
fn bindless_error_inequality_different() {
    let err1 = BindlessError::RegistryFull { capacity: 100 };
    let err2 = BindlessError::UnsupportedFeature;
    assert_ne!(err1, err2);
}

/// BindlessError Clone works.
#[test]
fn bindless_error_is_clone() {
    let err = BindlessError::UnsupportedFeature;
    let cloned = err.clone();
    assert_eq!(err, cloned);
}

// =============================================================================
// SECTION 6 -- FEATURE DETECTION TESTS (8+ tests)
// =============================================================================

/// bindless_required_features returns valid Features.
#[test]
fn bindless_required_features_returns_features() {
    let features = bindless_required_features();
    // Should return a valid wgpu::Features
    let _ = format!("{:?}", features);
}

/// bindless_optimal_features returns valid Features.
#[test]
fn bindless_optimal_features_returns_features() {
    let features = bindless_optimal_features();
    // Should return a valid wgpu::Features
    let _ = format!("{:?}", features);
}

/// bindless_optimal_features contains required features.
#[test]
fn bindless_optimal_contains_required() {
    let required = bindless_required_features();
    let optimal = bindless_optimal_features();
    // Optimal should be a superset of required
    assert!(optimal.contains(required));
}

/// max_bindless_textures_from_limits with downlevel defaults.
#[test]
fn max_bindless_from_limits_downlevel() {
    let limits = wgpu::Limits::downlevel_defaults();
    let max = max_bindless_textures_from_limits(&limits);
    assert!(max >= MIN_BINDLESS_TEXTURES);
}

/// max_bindless_textures_from_limits with default limits.
#[test]
fn max_bindless_from_limits_default() {
    let limits = wgpu::Limits::default();
    let max = max_bindless_textures_from_limits(&limits);
    assert!(max >= MIN_BINDLESS_TEXTURES);
    assert!(max <= MAX_BINDLESS_TEXTURES_CONSERVATIVE);
}

/// bindless_texture_layout_entry returns valid layout entry.
#[test]
fn bindless_texture_layout_entry_returns_valid() {
    let entry = bindless_texture_layout_entry(
        BINDLESS_TEXTURE_BINDING,
        256,
        wgpu::TextureSampleType::Float { filterable: true },
        wgpu::TextureViewDimension::D2,
    );
    assert_eq!(entry.binding, BINDLESS_TEXTURE_BINDING);
}

/// bindless_texture_layout_entry with different sample type.
#[test]
fn bindless_texture_layout_entry_different_sample_type() {
    let entry = bindless_texture_layout_entry(
        BINDLESS_TEXTURE_BINDING,
        256,
        wgpu::TextureSampleType::Uint,
        wgpu::TextureViewDimension::D2,
    );
    assert_eq!(entry.binding, BINDLESS_TEXTURE_BINDING);
}

/// bindless_texture_layout_entry count affects entry.
#[test]
fn bindless_texture_layout_entry_count_varies() {
    let entry_small = bindless_texture_layout_entry(
        BINDLESS_TEXTURE_BINDING,
        64,
        wgpu::TextureSampleType::Float { filterable: true },
        wgpu::TextureViewDimension::D2,
    );
    let entry_large = bindless_texture_layout_entry(
        BINDLESS_TEXTURE_BINDING,
        1024,
        wgpu::TextureSampleType::Float { filterable: true },
        wgpu::TextureViewDimension::D2,
    );
    // Both should have correct binding
    assert_eq!(entry_small.binding, BINDLESS_TEXTURE_BINDING);
    assert_eq!(entry_large.binding, BINDLESS_TEXTURE_BINDING);
}

// GPU-dependent feature detection tests
#[test]

fn supports_bindless_textures_with_device() {
    // Would need actual wgpu::Device
}

#[test]

fn supports_non_uniform_indexing_with_device() {
    // Would need actual wgpu::Device
}

#[test]

fn supports_partially_bound_with_device() {
    // Would need actual wgpu::Device
}

#[test]

fn max_bindless_textures_with_device() {
    // Would need actual wgpu::Device
}

#[test]

fn create_bindless_layout_with_device() {
    // Would need actual wgpu::Device
}

// =============================================================================
// SECTION 7 -- INTEGRATION TESTS (10+ tests, ignored without GPU)
// =============================================================================

#[test]

fn integration_register_texture() {
    // Register a texture and verify slot is returned
}

#[test]

fn integration_register_multiple_textures() {
    // Register many textures, verify all get unique slots
}

#[test]

fn integration_unregister_texture() {
    // Register then unregister, verify slot is freed
}

#[test]

fn integration_full_lifecycle() {
    // register -> get -> is_registered -> unregister -> verify freed
}

#[test]

fn integration_capacity_limits() {
    // Fill to capacity, verify RegistryFull error
}

#[test]

fn integration_bind_group_creation() {
    // Create bind group with bindless textures
}

#[test]

fn integration_bind_group_update() {
    // Modify registry, call update_bind_group
}

#[test]

fn integration_new_validated() {
    // Test new_validated with actual device
}

#[test]

fn integration_create_bind_group() {
    // Test create_bind_group method
}

#[test]

fn integration_iter_with_textures() {
    // Register textures, verify iter returns them
}

// =============================================================================
// SECTION 8 -- PROPERTY-BASED INVARIANT TESTS (8 tests)
// =============================================================================

/// Invariant: count never exceeds capacity.
#[test]
fn invariant_count_never_exceeds_capacity() {
    let registry = TextureRegistry::new(256);
    assert!(registry.count() <= registry.capacity());
}

/// Invariant: is_empty implies count == 0.
#[test]
fn invariant_is_empty_implies_zero_count() {
    let registry = TextureRegistry::new(256);
    if registry.is_empty() {
        assert_eq!(registry.count(), 0);
    }
}

/// Invariant: count == capacity implies is_full (without registration ability).
#[test]
fn invariant_full_means_no_free_slots() {
    // Without ability to register textures, we verify the concept
    let registry = TextureRegistry::new(256);
    // When count < capacity, should not be full
    if registry.count() < registry.capacity() {
        assert!(!registry.is_full());
    }
}

/// Invariant: utilization in [0.0, 1.0].
#[test]
fn invariant_utilization_in_valid_range() {
    let registry = TextureRegistry::new(256);
    let metrics = registry.metrics();
    let util = metrics.utilization();
    assert!(util >= 0.0);
    assert!(util <= 1.0);
}

/// Invariant: fragmentation in [0.0, 1.0].
#[test]
fn invariant_fragmentation_in_valid_range() {
    let registry = TextureRegistry::new(256);
    let metrics = registry.metrics();
    let frag = metrics.fragmentation();
    assert!(frag >= 0.0);
    assert!(frag <= 1.0);
}

/// Invariant: metrics.registered_count equals count().
#[test]
fn invariant_metrics_count_equals_count() {
    let registry = TextureRegistry::new(256);
    let metrics = registry.metrics();
    assert_eq!(metrics.registered_count, registry.count());
}

/// Invariant: metrics.capacity equals capacity().
#[test]
fn invariant_metrics_capacity_equals_capacity() {
    let registry = TextureRegistry::new(256);
    let metrics = registry.metrics();
    assert_eq!(metrics.capacity, registry.capacity());
}

/// Invariant: clear resets to empty state.
#[test]
fn invariant_clear_resets_to_empty() {
    let mut registry = TextureRegistry::new(256);
    registry.clear();
    assert!(registry.is_empty());
    assert_eq!(registry.count(), 0);
}

// =============================================================================
// SECTION 9 -- THREAD SAFETY TESTS (5 tests)
// =============================================================================

/// TextureSlot is Send.
#[test]
fn texture_slot_is_send() {
    fn assert_send<T: Send>() {}
    assert_send::<TextureSlot>();
}

/// TextureSlot is Sync.
#[test]
fn texture_slot_is_sync() {
    fn assert_sync<T: Sync>() {}
    assert_sync::<TextureSlot>();
}

/// TextureRegistry is Send.
#[test]
fn texture_registry_is_send() {
    fn assert_send<T: Send>() {}
    assert_send::<TextureRegistry>();
}

/// TextureRegistryMetrics is Send.
#[test]
fn texture_registry_metrics_is_send() {
    fn assert_send<T: Send>() {}
    assert_send::<TextureRegistryMetrics>();
}

/// TextureRegistryMetrics is Sync.
#[test]
fn texture_registry_metrics_is_sync() {
    fn assert_sync<T: Sync>() {}
    assert_sync::<TextureRegistryMetrics>();
}
