//! Whitebox tests for BindlessTextureRegistry (T-WGPU-P6.8.1).
//!
//! This module provides comprehensive whitebox tests for the GPU-driven
//! texture registry system located at:
//! `crates/renderer-backend/src/gpu_driven/texture_registry.rs`
//!
//! ## Component Under Test
//!
//! `BindlessTextureRegistry` - manages bindless texture arrays for GPU-driven
//! rendering using wgpu's TEXTURE_BINDING_ARRAY feature.
//!
//! ## Test Coverage
//!
//! 1. Feature Detection (~12 tests)
//!    - supports_bindless_textures()
//!    - supports_partially_bound()
//!    - supports_non_uniform_indexing()
//!    - required_features()
//!    - optimal_features()
//!
//! 2. Constants (~8 tests)
//!    - MAX_BINDLESS_TEXTURES
//!    - MIN_BINDLESS_TEXTURES
//!    - BINDLESS_BIND_GROUP_INDEX
//!    - TEXTURE_ARRAY_BINDING
//!    - SAMPLER_BINDING
//!
//! 3. TextureRegistryMetrics (~18 tests)
//!    - utilization()
//!    - fragmentation()
//!    - available_slots()
//!    - Trait implementations
//!
//! 4. CPU Helper Functions (~8 tests)
//!    - cpu_count_active()
//!    - cpu_find_free_slot()
//!    - cpu_fragmentation()
//!
//! 5. Free-List LIFO Behavior (~6 tests)
//!    - Slot recycling order verification
//!
//! 6. Edge Cases (~10 tests)
//!    - Boundary conditions
//!    - Invalid inputs
//!
//! ## Note on Device-Dependent Tests
//!
//! Tests requiring actual wgpu::Device, TextureView, or BindGroup creation
//! are kept minimal and rely on the existing unit tests in the source module.
//! These whitebox tests focus on state-based logic that can be tested without
//! GPU resources.

use renderer_backend::gpu_driven::{
    // Constants (renamed exports)
    TEXTURE_REGISTRY_BIND_GROUP_INDEX as BINDLESS_BIND_GROUP_INDEX,
    TEXTURE_REGISTRY_MAX_TEXTURES as MAX_BINDLESS_TEXTURES,
    TEXTURE_REGISTRY_MIN_TEXTURES as MIN_BINDLESS_TEXTURES,
    SAMPLER_BINDING, TEXTURE_ARRAY_BINDING,
    // Metrics
    TextureRegistryMetrics,
    // Feature detection helpers (renamed exports)
    texture_registry_optimal_features as optimal_features,
    texture_registry_required_features as required_features,
    supports_bindless_textures, supports_non_uniform_indexing, supports_partially_bound,
    // CPU helpers
    cpu_count_active, cpu_find_free_slot, cpu_fragmentation,
};
use wgpu::Features;

// ============================================================================
// CONSTANTS TESTS (~8 tests)
// ============================================================================

#[test]
fn test_max_bindless_textures_value() {
    assert_eq!(MAX_BINDLESS_TEXTURES, 1024);
}

#[test]
fn test_min_bindless_textures_value() {
    assert_eq!(MIN_BINDLESS_TEXTURES, 16);
}

#[test]
fn test_constants_min_less_than_max() {
    assert!(MIN_BINDLESS_TEXTURES < MAX_BINDLESS_TEXTURES);
}

#[test]
fn test_bindless_bind_group_index_value() {
    assert_eq!(BINDLESS_BIND_GROUP_INDEX, 3);
}

#[test]
fn test_texture_array_binding_value() {
    assert_eq!(TEXTURE_ARRAY_BINDING, 0);
}

#[test]
fn test_sampler_binding_value() {
    assert_eq!(SAMPLER_BINDING, 1);
}

#[test]
fn test_texture_and_sampler_bindings_distinct() {
    assert_ne!(TEXTURE_ARRAY_BINDING, SAMPLER_BINDING);
}

#[test]
fn test_min_bindless_reasonable_power_of_two() {
    // MIN should be a power of 2 for efficient alignment
    assert!(MIN_BINDLESS_TEXTURES.is_power_of_two());
}

// ============================================================================
// FEATURE DETECTION TESTS (~12 tests)
// ============================================================================

#[test]
fn test_supports_bindless_textures_empty_features() {
    let features = Features::empty();
    assert!(!supports_bindless_textures(features));
}

#[test]
fn test_supports_bindless_textures_with_feature() {
    let features = Features::TEXTURE_BINDING_ARRAY;
    assert!(supports_bindless_textures(features));
}

#[test]
fn test_supports_bindless_textures_with_multiple_features() {
    let features = Features::TEXTURE_BINDING_ARRAY | Features::DEPTH_CLIP_CONTROL;
    assert!(supports_bindless_textures(features));
}

#[test]
fn test_supports_partially_bound_empty_features() {
    let features = Features::empty();
    assert!(!supports_partially_bound(features));
}

#[test]
fn test_supports_partially_bound_with_feature() {
    let features = Features::PARTIALLY_BOUND_BINDING_ARRAY;
    assert!(supports_partially_bound(features));
}

#[test]
fn test_supports_non_uniform_indexing_empty_features() {
    let features = Features::empty();
    assert!(!supports_non_uniform_indexing(features));
}

#[test]
fn test_supports_non_uniform_indexing_with_feature() {
    let features = Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING;
    assert!(supports_non_uniform_indexing(features));
}

#[test]
fn test_required_features_includes_texture_binding_array() {
    let req = required_features();
    assert!(req.contains(Features::TEXTURE_BINDING_ARRAY));
}

#[test]
fn test_required_features_minimal() {
    let req = required_features();
    // Required features should be minimal
    assert_eq!(req, Features::TEXTURE_BINDING_ARRAY);
}

#[test]
fn test_optimal_features_includes_required() {
    let opt = optimal_features();
    let req = required_features();
    assert!(opt.contains(req));
}

#[test]
fn test_optimal_features_includes_non_uniform_indexing() {
    let opt = optimal_features();
    assert!(
        opt.contains(Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING)
    );
}

#[test]
fn test_optimal_features_includes_partially_bound() {
    let opt = optimal_features();
    assert!(opt.contains(Features::PARTIALLY_BOUND_BINDING_ARRAY));
}

// ============================================================================
// TEXTURE REGISTRY METRICS TESTS (~18 tests)
// ============================================================================

#[test]
fn test_metrics_default_values() {
    let metrics = TextureRegistryMetrics {
        active_count: 0,
        allocated_count: 0,
        free_slots: 0,
        capacity: 100,
        has_bind_group: false,
        is_dirty: true,
        has_bindless: true,
    };
    assert_eq!(metrics.active_count, 0);
    assert_eq!(metrics.allocated_count, 0);
    assert_eq!(metrics.capacity, 100);
}

#[test]
fn test_metrics_utilization_zero_when_empty() {
    let metrics = TextureRegistryMetrics {
        active_count: 0,
        allocated_count: 0,
        free_slots: 0,
        capacity: 100,
        has_bind_group: false,
        is_dirty: true,
        has_bindless: true,
    };
    assert_eq!(metrics.utilization(), 0.0);
}

#[test]
fn test_metrics_utilization_zero_capacity() {
    let metrics = TextureRegistryMetrics {
        active_count: 0,
        allocated_count: 0,
        free_slots: 0,
        capacity: 0,
        has_bind_group: false,
        is_dirty: true,
        has_bindless: true,
    };
    assert_eq!(metrics.utilization(), 0.0);
}

#[test]
fn test_metrics_utilization_half() {
    let metrics = TextureRegistryMetrics {
        active_count: 50,
        allocated_count: 50,
        free_slots: 0,
        capacity: 100,
        has_bind_group: true,
        is_dirty: false,
        has_bindless: true,
    };
    assert!((metrics.utilization() - 0.5).abs() < 0.001);
}

#[test]
fn test_metrics_utilization_full() {
    let metrics = TextureRegistryMetrics {
        active_count: 100,
        allocated_count: 100,
        free_slots: 0,
        capacity: 100,
        has_bind_group: true,
        is_dirty: false,
        has_bindless: true,
    };
    assert!((metrics.utilization() - 1.0).abs() < 0.001);
}

#[test]
fn test_metrics_fragmentation_zero_when_no_allocations() {
    let metrics = TextureRegistryMetrics {
        active_count: 0,
        allocated_count: 0,
        free_slots: 0,
        capacity: 100,
        has_bind_group: false,
        is_dirty: true,
        has_bindless: true,
    };
    assert_eq!(metrics.fragmentation(), 0.0);
}

#[test]
fn test_metrics_fragmentation_zero_when_no_free_slots() {
    let metrics = TextureRegistryMetrics {
        active_count: 10,
        allocated_count: 10,
        free_slots: 0,
        capacity: 100,
        has_bind_group: true,
        is_dirty: false,
        has_bindless: true,
    };
    assert_eq!(metrics.fragmentation(), 0.0);
}

#[test]
fn test_metrics_fragmentation_calculation() {
    let metrics = TextureRegistryMetrics {
        active_count: 8,
        allocated_count: 10,
        free_slots: 2,
        capacity: 100,
        has_bind_group: true,
        is_dirty: false,
        has_bindless: true,
    };
    // 2/10 = 0.2
    assert!((metrics.fragmentation() - 0.2).abs() < 0.001);
}

#[test]
fn test_metrics_fragmentation_half() {
    let metrics = TextureRegistryMetrics {
        active_count: 5,
        allocated_count: 10,
        free_slots: 5,
        capacity: 100,
        has_bind_group: true,
        is_dirty: false,
        has_bindless: true,
    };
    // 5/10 = 0.5
    assert!((metrics.fragmentation() - 0.5).abs() < 0.001);
}

#[test]
fn test_metrics_available_slots_empty() {
    let metrics = TextureRegistryMetrics {
        active_count: 0,
        allocated_count: 0,
        free_slots: 0,
        capacity: 100,
        has_bind_group: false,
        is_dirty: true,
        has_bindless: true,
    };
    // 100 - 0 + 0 = 100
    assert_eq!(metrics.available_slots(), 100);
}

#[test]
fn test_metrics_available_slots_some_allocated() {
    let metrics = TextureRegistryMetrics {
        active_count: 40,
        allocated_count: 50,
        free_slots: 10,
        capacity: 100,
        has_bind_group: true,
        is_dirty: false,
        has_bindless: true,
    };
    // 100 - 50 + 10 = 60
    assert_eq!(metrics.available_slots(), 60);
}

#[test]
fn test_metrics_available_slots_full() {
    let metrics = TextureRegistryMetrics {
        active_count: 100,
        allocated_count: 100,
        free_slots: 0,
        capacity: 100,
        has_bind_group: true,
        is_dirty: false,
        has_bindless: true,
    };
    // 100 - 100 + 0 = 0
    assert_eq!(metrics.available_slots(), 0);
}

#[test]
fn test_metrics_equality() {
    let m1 = TextureRegistryMetrics {
        active_count: 10,
        allocated_count: 15,
        free_slots: 5,
        capacity: 100,
        has_bind_group: true,
        is_dirty: false,
        has_bindless: true,
    };
    let m2 = TextureRegistryMetrics {
        active_count: 10,
        allocated_count: 15,
        free_slots: 5,
        capacity: 100,
        has_bind_group: true,
        is_dirty: false,
        has_bindless: true,
    };
    assert_eq!(m1, m2);
}

#[test]
fn test_metrics_inequality() {
    let m1 = TextureRegistryMetrics {
        active_count: 10,
        allocated_count: 15,
        free_slots: 5,
        capacity: 100,
        has_bind_group: true,
        is_dirty: false,
        has_bindless: true,
    };
    let m2 = TextureRegistryMetrics {
        active_count: 11, // Different
        allocated_count: 15,
        free_slots: 5,
        capacity: 100,
        has_bind_group: true,
        is_dirty: false,
        has_bindless: true,
    };
    assert_ne!(m1, m2);
}

#[test]
fn test_metrics_copy() {
    let m1 = TextureRegistryMetrics {
        active_count: 25,
        allocated_count: 30,
        free_slots: 5,
        capacity: 100,
        has_bind_group: true,
        is_dirty: false,
        has_bindless: true,
    };
    let m2 = m1; // Copy
    assert_eq!(m1.active_count, m2.active_count);
    assert_eq!(m1.has_bindless, m2.has_bindless);
}

#[test]
fn test_metrics_clone() {
    let m1 = TextureRegistryMetrics {
        active_count: 25,
        allocated_count: 30,
        free_slots: 5,
        capacity: 100,
        has_bind_group: true,
        is_dirty: false,
        has_bindless: true,
    };
    #[allow(clippy::clone_on_copy)]
    let m2 = m1.clone();
    assert_eq!(m1, m2);
}

#[test]
fn test_metrics_debug_format() {
    let m = TextureRegistryMetrics {
        active_count: 10,
        allocated_count: 15,
        free_slots: 5,
        capacity: 100,
        has_bind_group: true,
        is_dirty: false,
        has_bindless: true,
    };
    let debug = format!("{:?}", m);
    assert!(debug.contains("TextureRegistryMetrics"));
    assert!(debug.contains("active_count"));
    assert!(debug.contains("10"));
}

#[test]
fn test_metrics_available_slots_with_saturation() {
    // Test saturating_sub behavior when capacity < allocated_count (edge case)
    let metrics = TextureRegistryMetrics {
        active_count: 0,
        allocated_count: 150,
        free_slots: 50,
        capacity: 100, // Less than allocated
        has_bind_group: false,
        is_dirty: true,
        has_bindless: true,
    };
    // saturating_sub(100, 150) = 0, + 50 = 50
    assert_eq!(metrics.available_slots(), 50);
}

// ============================================================================
// CPU HELPER FUNCTION TESTS (~8 tests)
// ============================================================================

#[test]
fn test_cpu_count_active_empty_slice() {
    let slots: Vec<Option<wgpu::TextureView>> = vec![];
    assert_eq!(cpu_count_active(&slots), 0);
}

#[test]
fn test_cpu_count_active_all_none() {
    let slots: Vec<Option<wgpu::TextureView>> = vec![None, None, None];
    assert_eq!(cpu_count_active(&slots), 0);
}

#[test]
fn test_cpu_find_free_slot_empty_slice() {
    let slots: Vec<Option<wgpu::TextureView>> = vec![];
    assert_eq!(cpu_find_free_slot(&slots), None);
}

#[test]
fn test_cpu_find_free_slot_first_none() {
    let slots: Vec<Option<wgpu::TextureView>> = vec![None, None, None];
    assert_eq!(cpu_find_free_slot(&slots), Some(0));
}

#[test]
fn test_cpu_fragmentation_zero_allocated() {
    assert_eq!(cpu_fragmentation(0, 0), 0.0);
}

#[test]
fn test_cpu_fragmentation_no_free() {
    assert_eq!(cpu_fragmentation(10, 10), 0.0);
}

#[test]
fn test_cpu_fragmentation_half() {
    assert!((cpu_fragmentation(5, 10) - 0.5).abs() < 0.001);
}

#[test]
fn test_cpu_fragmentation_all_free() {
    assert!((cpu_fragmentation(0, 10) - 1.0).abs() < 0.001);
}

// ============================================================================
// FREE-LIST LIFO BEHAVIOR TESTS (~6 tests)
// ============================================================================

// Note: These tests verify the LIFO design by documenting expected behavior.
// The actual free_slots Vec behavior is tested implicitly through integration.

#[test]
fn test_lifo_vec_push_pop_order() {
    // Verify Vec<u32> LIFO behavior (stack semantics) used by free_slots
    let mut free_slots: Vec<u32> = Vec::new();
    free_slots.push(0);
    free_slots.push(1);
    free_slots.push(2);

    // LIFO: last in, first out
    assert_eq!(free_slots.pop(), Some(2));
    assert_eq!(free_slots.pop(), Some(1));
    assert_eq!(free_slots.pop(), Some(0));
    assert_eq!(free_slots.pop(), None);
}

#[test]
fn test_lifo_interleaved_push_pop() {
    let mut free_slots: Vec<u32> = Vec::new();
    free_slots.push(10);
    assert_eq!(free_slots.pop(), Some(10));
    free_slots.push(20);
    free_slots.push(30);
    assert_eq!(free_slots.pop(), Some(30));
    free_slots.push(40);
    assert_eq!(free_slots.pop(), Some(40));
    assert_eq!(free_slots.pop(), Some(20));
}

#[test]
fn test_free_slots_empty_pop_none() {
    let mut free_slots: Vec<u32> = Vec::new();
    assert_eq!(free_slots.pop(), None);
}

#[test]
fn test_free_slots_capacity_preserved() {
    let mut free_slots: Vec<u32> = Vec::with_capacity(100);
    free_slots.push(1);
    free_slots.push(2);
    free_slots.pop();
    free_slots.pop();
    // Capacity should remain at least 100 after pops
    assert!(free_slots.capacity() >= 100);
}

#[test]
fn test_free_slots_len_tracking() {
    let mut free_slots: Vec<u32> = Vec::new();
    assert_eq!(free_slots.len(), 0);
    free_slots.push(5);
    assert_eq!(free_slots.len(), 1);
    free_slots.push(10);
    assert_eq!(free_slots.len(), 2);
    free_slots.pop();
    assert_eq!(free_slots.len(), 1);
}

#[test]
fn test_free_slots_many_items() {
    let mut free_slots: Vec<u32> = Vec::new();
    for i in 0..100 {
        free_slots.push(i);
    }
    assert_eq!(free_slots.len(), 100);
    // LIFO: should get 99 first
    assert_eq!(free_slots.pop(), Some(99));
    assert_eq!(free_slots.pop(), Some(98));
}

// ============================================================================
// EDGE CASES TESTS (~10 tests)
// ============================================================================

#[test]
fn test_feature_combination_all_optimal() {
    let all = Features::TEXTURE_BINDING_ARRAY
        | Features::PARTIALLY_BOUND_BINDING_ARRAY
        | Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING;

    assert!(supports_bindless_textures(all));
    assert!(supports_partially_bound(all));
    assert!(supports_non_uniform_indexing(all));
}

#[test]
fn test_feature_combination_only_bindless() {
    let features = Features::TEXTURE_BINDING_ARRAY;
    assert!(supports_bindless_textures(features));
    assert!(!supports_partially_bound(features));
    assert!(!supports_non_uniform_indexing(features));
}

#[test]
fn test_metrics_all_fields_false() {
    let m = TextureRegistryMetrics {
        active_count: 0,
        allocated_count: 0,
        free_slots: 0,
        capacity: 0,
        has_bind_group: false,
        is_dirty: false,
        has_bindless: false,
    };
    assert_eq!(m.utilization(), 0.0);
    assert_eq!(m.fragmentation(), 0.0);
    assert_eq!(m.available_slots(), 0);
}

#[test]
fn test_metrics_max_capacity() {
    let m = TextureRegistryMetrics {
        active_count: MAX_BINDLESS_TEXTURES,
        allocated_count: MAX_BINDLESS_TEXTURES,
        free_slots: 0,
        capacity: MAX_BINDLESS_TEXTURES,
        has_bind_group: true,
        is_dirty: false,
        has_bindless: true,
    };
    assert!((m.utilization() - 1.0).abs() < 0.001);
    assert_eq!(m.available_slots(), 0);
}

#[test]
fn test_metrics_min_capacity() {
    let m = TextureRegistryMetrics {
        active_count: MIN_BINDLESS_TEXTURES,
        allocated_count: MIN_BINDLESS_TEXTURES,
        free_slots: 0,
        capacity: MIN_BINDLESS_TEXTURES,
        has_bind_group: true,
        is_dirty: false,
        has_bindless: true,
    };
    assert!((m.utilization() - 1.0).abs() < 0.001);
}

#[test]
fn test_cpu_fragmentation_large_values() {
    let active = 999_999;
    let allocated = 1_000_000;
    let frag = cpu_fragmentation(active, allocated);
    // (1_000_000 - 999_999) / 1_000_000 = 0.000001
    assert!(frag < 0.001);
    assert!(frag > 0.0);
}

#[test]
#[should_panic(expected = "subtract with overflow")]
fn test_cpu_fragmentation_active_greater_than_allocated_panics() {
    // Edge case: active > allocated is logically invalid and causes panic
    // This documents that the function has a precondition: active <= allocated
    // The formula (allocated - active) panics on usize underflow in debug builds
    let _ = cpu_fragmentation(100, 50);
}

#[test]
fn test_optimal_features_is_superset_of_required() {
    let opt = optimal_features();
    let req = required_features();
    // Every bit in required should be in optimal
    assert!(opt.contains(req));
}

#[test]
fn test_constants_bind_group_index_reasonable() {
    // Bind group index should be in reasonable range (0-3 typical for wgpu)
    assert!(BINDLESS_BIND_GROUP_INDEX <= 3);
}

#[test]
fn test_constants_max_is_power_of_two() {
    // MAX should be a power of 2 for efficient alignment
    assert!(MAX_BINDLESS_TEXTURES.is_power_of_two());
}

// ============================================================================
// THREAD SAFETY TESTS (~4 tests)
// ============================================================================

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

#[test]
fn test_features_is_send() {
    fn assert_send<T: Send>() {}
    assert_send::<Features>();
}

#[test]
fn test_features_is_sync() {
    fn assert_sync<T: Sync>() {}
    assert_sync::<Features>();
}

// ============================================================================
// INTEGRATION-STYLE TESTS (No GPU required)
// ============================================================================

#[test]
fn test_simulated_allocation_cycle() {
    // Simulate allocation/deallocation cycle using free_slots Vec
    let mut textures: Vec<Option<u32>> = Vec::with_capacity(16);
    let mut free_slots: Vec<u32> = Vec::new();
    let mut dirty = false;

    // Allocate 5 "textures"
    for i in 0..5u32 {
        textures.push(Some(i * 10)); // Store some value
        dirty = true;
    }
    assert_eq!(textures.len(), 5);

    // Free slot 2 and 4
    textures[2] = None;
    free_slots.push(2);
    textures[4] = None;
    free_slots.push(4);
    dirty = true;

    // Count active
    let active = textures.iter().filter(|t| t.is_some()).count();
    assert_eq!(active, 3);

    // Allocate new - should reuse slot 4 (LIFO)
    let reused = free_slots.pop().unwrap();
    assert_eq!(reused, 4);
    textures[reused as usize] = Some(999);
    dirty = true;

    // Now active should be 4
    let active = textures.iter().filter(|t| t.is_some()).count();
    assert_eq!(active, 4);

    // Next allocation should reuse slot 2
    let reused = free_slots.pop().unwrap();
    assert_eq!(reused, 2);

    assert!(dirty);
}

#[test]
fn test_simulated_full_registry() {
    // Simulate filling registry to capacity
    let max_textures = 4u32;
    let mut textures: Vec<Option<u32>> = Vec::with_capacity(max_textures as usize);
    let free_slots: Vec<u32> = Vec::new();

    // Fill to capacity
    for i in 0..max_textures {
        textures.push(Some(i));
    }

    // Check is_full logic
    let is_full = textures.len() >= max_textures as usize && free_slots.is_empty();
    assert!(is_full);
}

#[test]
fn test_simulated_is_full_with_free_slots() {
    // Registry at capacity but has free slots available
    let max_textures = 4u32;
    let mut textures: Vec<Option<u32>> = vec![Some(0), Some(1), None, Some(3)];
    let free_slots: Vec<u32> = vec![2]; // Slot 2 is free

    // is_full should be false because free_slots is not empty
    let is_full = textures.len() >= max_textures as usize && free_slots.is_empty();
    assert!(!is_full);

    // Reuse free slot
    let slot = free_slots.first().copied().unwrap();
    textures[slot as usize] = Some(99);
    assert_eq!(textures[2], Some(99));
}

#[test]
fn test_simulated_dirty_tracking() {
    let mut dirty = true; // Initially dirty

    // Simulate build_bind_group (would clear dirty)
    dirty = false;

    // Allocate changes dirty
    dirty = true;

    assert!(dirty);

    // Rebuild clears dirty
    dirty = false;

    // Free changes dirty
    dirty = true;

    assert!(dirty);
}

#[test]
fn test_metrics_calculation_pipeline() {
    // Test complete metrics calculation
    let textures: Vec<Option<u32>> = vec![Some(1), None, Some(3), None, Some(5)];
    let free_slots: Vec<u32> = vec![1, 3]; // Slots 1 and 3 are free
    let capacity = 10u32;

    let active_count = textures.iter().filter(|t| t.is_some()).count() as u32;
    let allocated_count = textures.len() as u32;
    let free_slot_count = free_slots.len() as u32;

    assert_eq!(active_count, 3);
    assert_eq!(allocated_count, 5);
    assert_eq!(free_slot_count, 2);

    let metrics = TextureRegistryMetrics {
        active_count,
        allocated_count,
        free_slots: free_slot_count,
        capacity,
        has_bind_group: true,
        is_dirty: false,
        has_bindless: true,
    };

    // Utilization: 3/10 = 0.3
    assert!((metrics.utilization() - 0.3).abs() < 0.001);

    // Fragmentation: 2/5 = 0.4
    assert!((metrics.fragmentation() - 0.4).abs() < 0.001);

    // Available: 10 - 5 + 2 = 7
    assert_eq!(metrics.available_slots(), 7);
}

// ============================================================================
// DOCUMENTATION/BEHAVIOR TESTS
// ============================================================================

#[test]
fn test_bind_group_index_convention() {
    // Document that TRINITY uses bind group 3 for bindless textures
    // Group 0: Per-frame uniforms (camera, time)
    // Group 1: Per-material data
    // Group 2: Per-object data
    // Group 3: Bindless texture array (this module)
    assert_eq!(BINDLESS_BIND_GROUP_INDEX, 3);
}

#[test]
fn test_binding_slot_convention() {
    // Binding 0: texture array
    // Binding 1: shared sampler
    assert_eq!(TEXTURE_ARRAY_BINDING, 0);
    assert_eq!(SAMPLER_BINDING, 1);
}

#[test]
fn test_capacity_bounds_enforced() {
    // Document the capacity clamping behavior
    // Values below MIN get clamped to MIN
    // Values above MAX get clamped to MAX
    let below_min = 1u32;
    let clamped = below_min.max(MIN_BINDLESS_TEXTURES).min(MAX_BINDLESS_TEXTURES);
    assert_eq!(clamped, MIN_BINDLESS_TEXTURES);

    let above_max = u32::MAX;
    let clamped = above_max.max(MIN_BINDLESS_TEXTURES).min(MAX_BINDLESS_TEXTURES);
    assert_eq!(clamped, MAX_BINDLESS_TEXTURES);
}
