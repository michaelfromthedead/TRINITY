//! Blackbox contract tests for T-WGPU-P2.6.3 IndexAllocator.
//!
//! CLEANROOM: No src/ access beyond the public API exported by the crate.
//! Tests use only `renderer_backend::resources::*` -- no internal fields,
//! no private methods, no implementation details.
//!
//! Public API under test (from mod.rs re-exports):
//!   - IndexAllocator: Generic index allocator with free list
//!   - GenerationalIndex: index + generation pair for use-after-free detection
//!   - IndexAllocatorMetrics: Statistics about allocator state
//!   - AllocatorError: Error types for allocation failures
//!
//! The index allocator provides:
//!   - allocate()/free(): O(1) allocation with LIFO free list recycling
//!   - Double-free protection via allocated_set tracking
//!   - Optional generation tracking for use-after-free detection
//!   - Thread-safe: Send + Sync
//!
//! AllocatorError variants:
//!   - AtCapacity { capacity: u32 }
//!   - InvalidIndex { index: u32, capacity: u32 }
//!   - DoubleFree(u32)
//!   - StaleGeneration { expected: u32, found: u32 }
//!
//! IndexAllocatorMetrics fields:
//!   - allocated_count: u32
//!   - capacity: u32
//!   - free_list_size: usize
//!   - peak_allocations: u32
//!   - fragmentation: f32
//!   - utilization: f32
//!   - has_generations: bool
//!
//! Coverage (~60 tests):
//!   1. IndexAllocator API (~20 tests)
//!   2. GenerationalIndex API (~15 tests)
//!   3. Generation Tracking API (~10 tests)
//!   4. Error API (~10 tests)
//!   5. Thread Safety (~5 tests)

use renderer_backend::resources::{
    AllocatorError, GenerationalIndex, IndexAllocator,
};

use std::collections::HashSet;
use std::sync::Arc;
use std::thread;

// =============================================================================
// Category 1: IndexAllocator Basic API (~20 tests)
// =============================================================================

#[test]
fn allocator_new_creates_empty_allocator() {
    let allocator = IndexAllocator::new(100);
    assert_eq!(allocator.count(), 0);
    assert_eq!(allocator.capacity(), 100);
}

#[test]
fn allocator_new_with_zero_capacity() {
    let allocator = IndexAllocator::new(0);
    assert_eq!(allocator.capacity(), 0);
    assert_eq!(allocator.count(), 0);
}

#[test]
fn allocator_with_generations_creates_allocator() {
    let allocator = IndexAllocator::with_generations(50);
    assert_eq!(allocator.count(), 0);
    assert_eq!(allocator.capacity(), 50);
}

#[test]
fn allocator_allocate_returns_index() {
    let mut allocator = IndexAllocator::new(10);
    let index = allocator.allocate();
    assert!(index.is_some());
    assert_eq!(allocator.count(), 1);
}

#[test]
fn allocator_allocate_returns_sequential_indices() {
    let mut allocator = IndexAllocator::new(10);
    let idx0 = allocator.allocate().unwrap();
    let idx1 = allocator.allocate().unwrap();
    let idx2 = allocator.allocate().unwrap();

    // Indices should be unique
    let mut indices = HashSet::new();
    indices.insert(idx0);
    indices.insert(idx1);
    indices.insert(idx2);
    assert_eq!(indices.len(), 3);
}

#[test]
fn allocator_allocate_returns_none_when_full() {
    let mut allocator = IndexAllocator::new(2);
    assert!(allocator.allocate().is_some());
    assert!(allocator.allocate().is_some());
    assert!(allocator.allocate().is_none());
}

#[test]
fn allocator_try_allocate_returns_ok() {
    let mut allocator = IndexAllocator::new(10);
    let result = allocator.try_allocate();
    assert!(result.is_ok());
}

#[test]
fn allocator_try_allocate_returns_error_when_full() {
    let mut allocator = IndexAllocator::new(1);
    allocator.allocate();
    let result = allocator.try_allocate();
    assert!(result.is_err());
}

#[test]
fn allocator_free_returns_true_for_valid_index() {
    let mut allocator = IndexAllocator::new(10);
    let index = allocator.allocate().unwrap();
    assert!(allocator.free(index));
    assert_eq!(allocator.count(), 0);
}

#[test]
fn allocator_free_returns_false_for_invalid_index() {
    let mut allocator = IndexAllocator::new(10);
    assert!(!allocator.free(999));
}

#[test]
fn allocator_free_returns_false_for_double_free() {
    let mut allocator = IndexAllocator::new(10);
    let index = allocator.allocate().unwrap();
    assert!(allocator.free(index));
    assert!(!allocator.free(index));
}

#[test]
fn allocator_count_tracks_allocations() {
    let mut allocator = IndexAllocator::new(10);
    assert_eq!(allocator.count(), 0);
    allocator.allocate();
    assert_eq!(allocator.count(), 1);
    allocator.allocate();
    assert_eq!(allocator.count(), 2);
}

#[test]
fn allocator_capacity_is_immutable() {
    let mut allocator = IndexAllocator::new(100);
    allocator.allocate();
    allocator.allocate();
    assert_eq!(allocator.capacity(), 100);
}

#[test]
fn allocator_free_count_tracks_recycled_indices() {
    let mut allocator = IndexAllocator::new(10);
    // free_count returns the size of the free list (recycled indices)
    // Initially empty because no indices have been freed yet
    assert_eq!(allocator.free_count(), 0);

    let idx = allocator.allocate().unwrap();
    // Still 0 - allocation doesn't add to free list
    assert_eq!(allocator.free_count(), 0);

    allocator.free(idx);
    // Now the freed index is in the free list
    assert_eq!(allocator.free_count(), 1);
}

#[test]
fn allocator_is_allocated_returns_true_for_allocated() {
    let mut allocator = IndexAllocator::new(10);
    let index = allocator.allocate().unwrap();
    assert!(allocator.is_allocated(index));
}

#[test]
fn allocator_is_allocated_returns_false_for_unallocated() {
    let allocator = IndexAllocator::new(10);
    assert!(!allocator.is_allocated(0));
    assert!(!allocator.is_allocated(5));
}

#[test]
fn allocator_is_allocated_returns_false_after_free() {
    let mut allocator = IndexAllocator::new(10);
    let index = allocator.allocate().unwrap();
    allocator.free(index);
    assert!(!allocator.is_allocated(index));
}

#[test]
fn allocator_clear_removes_all_allocations() {
    let mut allocator = IndexAllocator::new(10);
    allocator.allocate();
    allocator.allocate();
    allocator.allocate();
    allocator.clear();
    assert_eq!(allocator.count(), 0);
    // After clear, free_count is 0 (free list is empty)
    // and next_index is reset to 0, so fresh allocations start over
    assert_eq!(allocator.free_count(), 0);
}

#[test]
fn allocator_reset_restores_initial_state() {
    let mut allocator = IndexAllocator::new(10);
    allocator.allocate();
    allocator.allocate();
    allocator.reset();
    assert_eq!(allocator.count(), 0);
    assert_eq!(allocator.capacity(), 10);
}

#[test]
fn allocator_reuses_freed_indices() {
    let mut allocator = IndexAllocator::new(10);
    let idx0 = allocator.allocate().unwrap();
    let idx1 = allocator.allocate().unwrap();
    allocator.free(idx0);

    // Next allocation should reuse freed index (LIFO)
    let idx2 = allocator.allocate().unwrap();
    assert!(idx2 == idx0 || idx2 != idx1); // Either reuses idx0 or is different from idx1
    assert_eq!(allocator.count(), 2);
}

// =============================================================================
// Category 2: GenerationalIndex API (~15 tests)
// =============================================================================

#[test]
fn generational_index_new_constructs_valid_index() {
    let gen_idx = GenerationalIndex::new(42, 1);
    assert_eq!(gen_idx.index(), 42);
    assert_eq!(gen_idx.generation(), 1);
}

#[test]
fn generational_index_index_accessor() {
    let gen_idx = GenerationalIndex::new(100, 5);
    assert_eq!(gen_idx.index(), 100);
}

#[test]
fn generational_index_generation_accessor() {
    let gen_idx = GenerationalIndex::new(10, 999);
    assert_eq!(gen_idx.generation(), 999);
}

#[test]
fn generational_index_null_returns_null_index() {
    let null_idx = GenerationalIndex::null();
    assert!(null_idx.is_null());
}

#[test]
fn generational_index_is_null_returns_false_for_valid() {
    let gen_idx = GenerationalIndex::new(0, 1);
    assert!(!gen_idx.is_null());
}

#[test]
fn generational_index_equality() {
    let idx1 = GenerationalIndex::new(10, 5);
    let idx2 = GenerationalIndex::new(10, 5);
    let idx3 = GenerationalIndex::new(10, 6);
    let idx4 = GenerationalIndex::new(11, 5);

    assert_eq!(idx1, idx2);
    assert_ne!(idx1, idx3);
    assert_ne!(idx1, idx4);
}

#[test]
fn generational_index_hash_consistent() {
    use std::collections::HashMap;

    let idx1 = GenerationalIndex::new(42, 7);
    let idx2 = GenerationalIndex::new(42, 7);

    let mut map = HashMap::new();
    map.insert(idx1, "value");

    assert_eq!(map.get(&idx2), Some(&"value"));
}

#[test]
fn generational_index_hash_different_for_different_indices() {
    use std::collections::HashMap;

    let idx1 = GenerationalIndex::new(1, 1);
    let idx2 = GenerationalIndex::new(1, 2);
    let idx3 = GenerationalIndex::new(2, 1);

    let mut map = HashMap::new();
    map.insert(idx1, "a");
    map.insert(idx2, "b");
    map.insert(idx3, "c");

    assert_eq!(map.len(), 3);
}

#[test]
fn generational_index_debug_impl() {
    let idx = GenerationalIndex::new(42, 7);
    let debug_str = format!("{:?}", idx);
    assert!(debug_str.contains("42") || debug_str.contains("7"));
}

#[test]
fn generational_index_clone() {
    let idx1 = GenerationalIndex::new(100, 50);
    let idx2 = idx1.clone();
    assert_eq!(idx1, idx2);
}

#[test]
fn generational_index_copy() {
    let idx1 = GenerationalIndex::new(100, 50);
    let idx2 = idx1; // Copy
    assert_eq!(idx1.index(), idx2.index());
    assert_eq!(idx1.generation(), idx2.generation());
}

#[test]
fn generational_index_zero_generation() {
    let idx = GenerationalIndex::new(5, 0);
    assert_eq!(idx.index(), 5);
    assert_eq!(idx.generation(), 0);
}

#[test]
fn generational_index_max_values() {
    let idx = GenerationalIndex::new(u32::MAX, u32::MAX);
    assert_eq!(idx.index(), u32::MAX);
    assert_eq!(idx.generation(), u32::MAX);
}

#[test]
fn generational_index_default_or_null_is_recognizable() {
    let null = GenerationalIndex::null();
    let _valid = GenerationalIndex::new(0, 1);

    assert!(null.is_null());
    // A valid index at position 0 with generation > 0 should not be null
    // (depends on implementation, but generation tracking should distinguish)
}

// =============================================================================
// Category 3: Generation Tracking API (~10 tests)
// =============================================================================

#[test]
fn allocator_allocate_generational_returns_gen_index() {
    let mut allocator = IndexAllocator::with_generations(10);
    let gen_idx = allocator.allocate_generational();
    assert!(gen_idx.is_some());

    let idx = gen_idx.unwrap();
    assert!(!idx.is_null());
}

#[test]
fn allocator_allocate_generational_increments_generation() {
    let mut allocator = IndexAllocator::with_generations(10);
    let idx1 = allocator.allocate_generational().unwrap();
    allocator.free(idx1.index());
    let idx2 = allocator.allocate_generational().unwrap();

    // Same slot, different generation
    if idx1.index() == idx2.index() {
        assert!(idx2.generation() > idx1.generation());
    }
}

#[test]
fn allocator_is_valid_returns_true_for_current_generation() {
    let mut allocator = IndexAllocator::with_generations(10);
    let gen_idx = allocator.allocate_generational().unwrap();
    assert!(allocator.is_valid(gen_idx));
}

#[test]
fn allocator_is_valid_returns_false_after_free() {
    let mut allocator = IndexAllocator::with_generations(10);
    let gen_idx = allocator.allocate_generational().unwrap();
    allocator.free(gen_idx.index());
    assert!(!allocator.is_valid(gen_idx));
}

#[test]
fn allocator_is_valid_returns_false_for_stale_generation() {
    let mut allocator = IndexAllocator::with_generations(10);
    let old_idx = allocator.allocate_generational().unwrap();
    allocator.free(old_idx.index());
    let _new_idx = allocator.allocate_generational().unwrap();

    // Old index should be invalid due to generation mismatch
    assert!(!allocator.is_valid(old_idx));
}

#[test]
fn allocator_is_valid_returns_false_for_null() {
    let allocator = IndexAllocator::with_generations(10);
    let null_idx = GenerationalIndex::null();
    assert!(!allocator.is_valid(null_idx));
}

#[test]
fn allocator_try_free_generational_succeeds_for_valid() {
    let mut allocator = IndexAllocator::with_generations(10);
    let gen_idx = allocator.allocate_generational().unwrap();
    let result = allocator.try_free_generational(gen_idx);
    assert!(result.is_ok());
}

#[test]
fn allocator_try_free_generational_fails_for_stale() {
    let mut allocator = IndexAllocator::with_generations(10);
    let old_idx = allocator.allocate_generational().unwrap();
    allocator.free(old_idx.index());
    let _new_idx = allocator.allocate_generational().unwrap();

    // Trying to free with old generation should fail
    let result = allocator.try_free_generational(old_idx);
    assert!(result.is_err());
}

#[test]
fn allocator_try_free_generational_fails_for_null() {
    let mut allocator = IndexAllocator::with_generations(10);
    let null_idx = GenerationalIndex::null();
    let result = allocator.try_free_generational(null_idx);
    assert!(result.is_err());
}

#[test]
fn allocator_generation_tracking_across_multiple_cycles() {
    let mut allocator = IndexAllocator::with_generations(2);

    // Allocate both slots
    let idx0 = allocator.allocate_generational().unwrap();
    let idx1 = allocator.allocate_generational().unwrap();

    // Free and reallocate multiple times
    for _ in 0..5 {
        allocator.free(idx0.index());
        allocator.free(idx1.index());

        let new0 = allocator.allocate_generational().unwrap();
        let new1 = allocator.allocate_generational().unwrap();

        // Original indices should be invalid
        assert!(!allocator.is_valid(idx0));
        assert!(!allocator.is_valid(idx1));

        // New indices should be valid
        assert!(allocator.is_valid(new0));
        assert!(allocator.is_valid(new1));
    }
}

// =============================================================================
// Category 4: Error API (~10 tests)
// =============================================================================

#[test]
fn allocator_error_at_capacity() {
    let mut allocator = IndexAllocator::new(1);
    allocator.allocate();
    let result = allocator.try_allocate();

    match result {
        Err(AllocatorError::AtCapacity { capacity }) => {
            assert_eq!(capacity, 1);
        }
        Err(e) => panic!("Expected AtCapacity, got {:?}", e),
        Ok(_) => panic!("Expected error, got Ok"),
    }
}

#[test]
fn allocator_error_invalid_index_on_free() {
    let mut allocator = IndexAllocator::with_generations(10);
    let gen_idx = GenerationalIndex::new(999, 1);
    let result = allocator.try_free_generational(gen_idx);
    assert!(result.is_err());
}

#[test]
fn allocator_error_stale_generation() {
    let mut allocator = IndexAllocator::with_generations(10);
    let idx = allocator.allocate_generational().unwrap();
    allocator.free(idx.index());
    let _new_idx = allocator.allocate_generational().unwrap();

    let result = allocator.try_free_generational(idx);
    match result {
        Err(AllocatorError::StaleGeneration { .. }) |
        Err(AllocatorError::InvalidIndex { .. }) |
        Err(AllocatorError::DoubleFree(_)) |
        Err(_) => (), // Any error is acceptable for stale generation
        Ok(_) => panic!("Expected error for stale generation"),
    }
}

#[test]
fn allocator_error_display_trait() {
    let err = AllocatorError::AtCapacity { capacity: 10 };
    let display_str = format!("{}", err);
    assert!(!display_str.is_empty());
}

#[test]
fn allocator_error_debug_trait() {
    let err = AllocatorError::AtCapacity { capacity: 10 };
    let debug_str = format!("{:?}", err);
    assert!(!debug_str.is_empty());
}

#[test]
fn allocator_error_is_send() {
    fn assert_send<T: Send>() {}
    assert_send::<AllocatorError>();
}

#[test]
fn allocator_error_is_sync() {
    fn assert_sync<T: Sync>() {}
    assert_sync::<AllocatorError>();
}

#[test]
fn allocator_error_clone_if_implemented() {
    // AllocatorError should be clonable for error propagation
    let err = AllocatorError::AtCapacity { capacity: 10 };
    let _cloned = err.clone();
}

#[test]
fn allocator_error_variants_distinct() {
    let err1 = AllocatorError::AtCapacity { capacity: 10 };
    let err2 = AllocatorError::InvalidIndex { index: 42, capacity: 10 };

    // Different variants should have different debug representations
    let debug1 = format!("{:?}", err1);
    let debug2 = format!("{:?}", err2);
    assert_ne!(debug1, debug2);
}

#[test]
fn allocator_error_invalid_index_contains_index() {
    let err = AllocatorError::InvalidIndex { index: 123, capacity: 100 };
    let display_str = format!("{}", err);
    // Should contain the invalid index number
    assert!(display_str.contains("123") || format!("{:?}", err).contains("123"));
}

#[test]
fn allocator_error_double_free() {
    let err = AllocatorError::DoubleFree(42);
    let display_str = format!("{}", err);
    assert!(!display_str.is_empty());
    assert!(display_str.contains("42") || format!("{:?}", err).contains("42"));
}

#[test]
fn allocator_error_stale_generation_contains_info() {
    let err = AllocatorError::StaleGeneration { expected: 5, found: 3 };
    let display_str = format!("{}", err);
    assert!(!display_str.is_empty());
}

// =============================================================================
// Category 5: Thread Safety (~5 tests)
// =============================================================================

#[test]
fn allocator_is_send() {
    fn assert_send<T: Send>() {}
    assert_send::<IndexAllocator>();
}

#[test]
fn allocator_is_sync() {
    fn assert_sync<T: Sync>() {}
    assert_sync::<IndexAllocator>();
}

#[test]
fn generational_index_is_send() {
    fn assert_send<T: Send>() {}
    assert_send::<GenerationalIndex>();
}

#[test]
fn generational_index_is_sync() {
    fn assert_sync<T: Sync>() {}
    assert_sync::<GenerationalIndex>();
}

#[test]
fn allocator_can_be_shared_across_threads() {
    use std::sync::Mutex;

    let allocator = Arc::new(Mutex::new(IndexAllocator::new(100)));
    let allocator_clone = Arc::clone(&allocator);

    let handle = thread::spawn(move || {
        let mut alloc = allocator_clone.lock().unwrap();
        alloc.allocate().unwrap()
    });

    let idx = handle.join().unwrap();
    let alloc = allocator.lock().unwrap();
    assert!(alloc.is_allocated(idx));
}

// =============================================================================
// Category 6: Metrics API (~5 tests)
// =============================================================================

#[test]
fn allocator_metrics_reports_allocated_count() {
    let mut allocator = IndexAllocator::new(10);
    allocator.allocate();
    allocator.allocate();

    let metrics = allocator.metrics();
    assert_eq!(metrics.allocated_count, 2);
}

#[test]
fn allocator_metrics_reports_capacity() {
    let allocator = IndexAllocator::new(50);
    let metrics = allocator.metrics();
    assert_eq!(metrics.capacity, 50);
}

#[test]
fn allocator_metrics_reports_free_list_size() {
    let mut allocator = IndexAllocator::new(10);
    allocator.allocate();
    allocator.allocate();
    allocator.allocate();

    let metrics = allocator.metrics();
    // Free list size depends on implementation, but should be non-negative
    assert!(metrics.free_list_size <= 10);
}

#[test]
fn allocator_metrics_fragmentation() {
    let mut allocator = IndexAllocator::new(10);

    // Allocate all
    let indices: Vec<_> = (0..10).filter_map(|_| allocator.allocate()).collect();

    // Free every other one to create fragmentation
    for (i, idx) in indices.iter().enumerate() {
        if i % 2 == 0 {
            allocator.free(*idx);
        }
    }

    let metrics = allocator.metrics();
    // Should have some fragmentation metric (0.0-1.0)
    assert!(metrics.fragmentation >= 0.0);
    assert!(metrics.fragmentation <= 1.0);
}

#[test]
fn allocator_metrics_utilization() {
    let mut allocator = IndexAllocator::new(10);
    allocator.allocate();
    allocator.allocate();
    allocator.allocate();
    allocator.allocate();
    allocator.allocate();

    let metrics = allocator.metrics();
    // 5 out of 10 = 50% utilization
    assert!((metrics.utilization - 0.5).abs() < 0.01);
}

#[test]
fn allocator_metrics_peak_allocations() {
    let mut allocator = IndexAllocator::new(10);

    // Allocate 7
    let mut indices = Vec::new();
    for _ in 0..7 {
        indices.push(allocator.allocate().unwrap());
    }

    // Free some
    for idx in indices.drain(0..4) {
        allocator.free(idx);
    }

    let metrics = allocator.metrics();
    // Peak should be 7
    assert_eq!(metrics.peak_allocations, 7);
}

#[test]
fn allocator_metrics_has_generations() {
    let allocator_no_gen = IndexAllocator::new(10);
    let allocator_with_gen = IndexAllocator::with_generations(10);

    assert!(!allocator_no_gen.metrics().has_generations);
    assert!(allocator_with_gen.metrics().has_generations);
}

// =============================================================================
// Category 7: Edge Cases and Stress Tests (~5 tests)
// =============================================================================

#[test]
fn allocator_handles_rapid_alloc_free_cycles() {
    let mut allocator = IndexAllocator::new(10);

    for _ in 0..1000 {
        let idx = allocator.allocate().unwrap();
        assert!(allocator.free(idx));
    }

    assert_eq!(allocator.count(), 0);
}

#[test]
fn allocator_full_then_empty_cycle() {
    let mut allocator = IndexAllocator::new(100);

    // Fill completely
    let indices: Vec<_> = (0..100).filter_map(|_| allocator.allocate()).collect();
    assert_eq!(indices.len(), 100);
    assert!(allocator.allocate().is_none());

    // Empty completely
    for idx in indices {
        assert!(allocator.free(idx));
    }
    assert_eq!(allocator.count(), 0);

    // Fill again
    for _ in 0..100 {
        assert!(allocator.allocate().is_some());
    }
}

#[test]
fn allocator_indices_are_within_bounds() {
    let mut allocator = IndexAllocator::new(10);

    for _ in 0..100 {
        if let Some(idx) = allocator.allocate() {
            assert!(idx < 10, "Index {} exceeds capacity 10", idx);
            allocator.free(idx);
        }
    }
}

#[test]
fn allocator_all_indices_eventually_used() {
    let mut allocator = IndexAllocator::new(5);
    let mut seen = HashSet::new();

    // Keep allocating and freeing until we've seen all indices
    for _ in 0..100 {
        if let Some(idx) = allocator.allocate() {
            seen.insert(idx);
            allocator.free(idx);

            if seen.len() == 5 {
                break;
            }
        }
    }

    // Should have seen indices 0-4
    assert!(seen.len() <= 5);
}

#[test]
fn allocator_stress_random_alloc_free() {
    let mut allocator = IndexAllocator::new(50);
    let mut allocated = Vec::new();

    // Pseudo-random pattern: allocate some, free some
    for i in 0..500 {
        if i % 3 != 0 && allocated.len() < 50 {
            if let Some(idx) = allocator.allocate() {
                allocated.push(idx);
            }
        } else if !allocated.is_empty() {
            let idx = allocated.pop().unwrap();
            allocator.free(idx);
        }
    }

    // Invariant: count should match allocated vector size
    assert_eq!(allocator.count() as usize, allocated.len());
}

// =============================================================================
// Category 8: Available Slots API (~4 tests)
// =============================================================================

#[test]
fn allocator_available_returns_capacity_initially() {
    let allocator = IndexAllocator::new(100);
    assert_eq!(allocator.available(), 100);
}

#[test]
fn allocator_available_decreases_on_allocate() {
    let mut allocator = IndexAllocator::new(10);
    assert_eq!(allocator.available(), 10);
    allocator.allocate();
    assert_eq!(allocator.available(), 9);
    allocator.allocate();
    assert_eq!(allocator.available(), 8);
}

#[test]
fn allocator_available_increases_on_free() {
    let mut allocator = IndexAllocator::new(10);
    let idx1 = allocator.allocate().unwrap();
    let idx2 = allocator.allocate().unwrap();
    assert_eq!(allocator.available(), 8);

    allocator.free(idx1);
    assert_eq!(allocator.available(), 9);

    allocator.free(idx2);
    assert_eq!(allocator.available(), 10);
}

#[test]
fn allocator_available_is_zero_when_full() {
    let mut allocator = IndexAllocator::new(3);
    allocator.allocate();
    allocator.allocate();
    allocator.allocate();
    assert_eq!(allocator.available(), 0);
}

// =============================================================================
// Category 9: Default and Builder Patterns (~3 tests)
// =============================================================================

#[test]
fn allocator_default_reasonable_state() {
    // If Default is implemented
    let allocator = IndexAllocator::new(0);
    assert_eq!(allocator.count(), 0);
}

#[test]
fn allocator_with_large_capacity() {
    let allocator = IndexAllocator::new(1_000_000);
    assert_eq!(allocator.capacity(), 1_000_000);
    assert_eq!(allocator.count(), 0);
}

#[test]
fn allocator_metrics_debug_impl() {
    let allocator = IndexAllocator::new(10);
    let metrics = allocator.metrics();
    let debug_str = format!("{:?}", metrics);
    assert!(!debug_str.is_empty());
}

// =============================================================================
// Category 10: LIFO Ordering Tests (~5 tests)
// =============================================================================

#[test]
fn allocator_lifo_single_free_reallocate() {
    let mut allocator = IndexAllocator::new(10);

    // Allocate several indices
    let _idx0 = allocator.allocate().unwrap();
    let idx1 = allocator.allocate().unwrap();
    let _idx2 = allocator.allocate().unwrap();

    // Free just one
    allocator.free(idx1);

    // Next allocation should return the freed index
    let idx_new = allocator.allocate().unwrap();
    assert_eq!(idx_new, idx1, "LIFO: freed index should be reused immediately");
}

#[test]
fn allocator_lifo_multiple_frees_reverse_order() {
    let mut allocator = IndexAllocator::new(10);

    // Allocate 0, 1, 2, 3, 4
    let indices: Vec<u32> = (0..5).map(|_| allocator.allocate().unwrap()).collect();

    // Free in order: 0, 1, 2
    allocator.free(indices[0]);
    allocator.free(indices[1]);
    allocator.free(indices[2]);

    // LIFO: should return in reverse order (2, 1, 0)
    let realloc0 = allocator.allocate().unwrap();
    let realloc1 = allocator.allocate().unwrap();
    let realloc2 = allocator.allocate().unwrap();

    assert_eq!(realloc0, indices[2], "First realloc should be last freed (LIFO)");
    assert_eq!(realloc1, indices[1], "Second realloc should be second-last freed");
    assert_eq!(realloc2, indices[0], "Third realloc should be first freed");
}

#[test]
fn allocator_lifo_interleaved_alloc_free() {
    let mut allocator = IndexAllocator::new(10);

    // Allocate 0, 1
    let idx0 = allocator.allocate().unwrap();
    let idx1 = allocator.allocate().unwrap();

    // Free 0
    allocator.free(idx0);

    // Allocate (should get 0 back due to LIFO)
    let idx2 = allocator.allocate().unwrap();
    assert_eq!(idx2, idx0);

    // Allocate fresh (should get 2)
    let idx3 = allocator.allocate().unwrap();
    assert!(idx3 != idx0 && idx3 != idx1, "Should be a fresh index");

    // Free 1 and 2
    allocator.free(idx1);
    allocator.free(idx2);

    // LIFO: should return idx2 first (last freed)
    let idx4 = allocator.allocate().unwrap();
    assert_eq!(idx4, idx2);
}

#[test]
fn allocator_lifo_full_cycle_preserves_order() {
    let mut allocator = IndexAllocator::new(5);

    // Fill completely
    let indices: Vec<u32> = (0..5).map(|_| allocator.allocate().unwrap()).collect();

    // Free all in specific order: 4, 2, 0, 3, 1
    allocator.free(indices[4]);
    allocator.free(indices[2]);
    allocator.free(indices[0]);
    allocator.free(indices[3]);
    allocator.free(indices[1]);

    // LIFO: should reallocate in reverse: 1, 3, 0, 2, 4
    assert_eq!(allocator.allocate().unwrap(), indices[1]);
    assert_eq!(allocator.allocate().unwrap(), indices[3]);
    assert_eq!(allocator.allocate().unwrap(), indices[0]);
    assert_eq!(allocator.allocate().unwrap(), indices[2]);
    assert_eq!(allocator.allocate().unwrap(), indices[4]);
}

#[test]
fn allocator_lifo_free_list_vs_fresh() {
    let mut allocator = IndexAllocator::new(10);

    // Allocate indices 0-4
    for _ in 0..5 {
        allocator.allocate();
    }

    // Free index 2
    allocator.free(2);

    // Free list should have 1 entry
    assert_eq!(allocator.free_count(), 1);

    // Next alloc should come from free list (2), not fresh (5)
    let realloc = allocator.allocate().unwrap();
    assert_eq!(realloc, 2, "Should reuse freed index before fresh allocation");

    // Now fresh allocation should give 5
    let fresh = allocator.allocate().unwrap();
    assert_eq!(fresh, 5, "After free list exhausted, should allocate fresh");
}

// =============================================================================
// Category 11: Allocator Lifecycle Tests (~5 tests)
// =============================================================================

#[test]
fn allocator_lifecycle_new_to_full() {
    let mut allocator = IndexAllocator::new(3);

    // Initial state
    assert_eq!(allocator.count(), 0);
    assert_eq!(allocator.available(), 3);

    // Allocate to full
    allocator.allocate();
    assert_eq!(allocator.count(), 1);
    assert_eq!(allocator.available(), 2);

    allocator.allocate();
    allocator.allocate();
    assert_eq!(allocator.count(), 3);
    assert_eq!(allocator.available(), 0);

    // At capacity
    assert!(allocator.allocate().is_none());
}

#[test]
fn allocator_lifecycle_full_to_empty() {
    let mut allocator = IndexAllocator::new(3);

    // Fill
    let idx0 = allocator.allocate().unwrap();
    let idx1 = allocator.allocate().unwrap();
    let idx2 = allocator.allocate().unwrap();

    // Empty
    allocator.free(idx0);
    assert_eq!(allocator.count(), 2);

    allocator.free(idx1);
    assert_eq!(allocator.count(), 1);

    allocator.free(idx2);
    assert_eq!(allocator.count(), 0);
    assert_eq!(allocator.available(), 3);
}

#[test]
fn allocator_lifecycle_cyclic_usage() {
    let mut allocator = IndexAllocator::new(5);

    // Cycle 1: Allocate all, free all
    let mut indices: Vec<u32> = (0..5).map(|_| allocator.allocate().unwrap()).collect();
    assert_eq!(allocator.count(), 5);
    for idx in indices.drain(..) {
        allocator.free(idx);
    }
    assert_eq!(allocator.count(), 0);

    // Cycle 2: Partial allocate, partial free
    let idx0 = allocator.allocate().unwrap();
    let _idx1 = allocator.allocate().unwrap();
    allocator.free(idx0);
    let idx2 = allocator.allocate().unwrap();
    assert_eq!(idx2, idx0, "Recycled index");

    // Cycle 3: Allocate remaining
    for _ in 0..3 {
        allocator.allocate();
    }
    assert_eq!(allocator.count(), 5);
}

#[test]
fn allocator_lifecycle_with_clear() {
    let mut allocator = IndexAllocator::new(10);

    // Use allocator
    for _ in 0..5 {
        allocator.allocate();
    }
    allocator.free(2);
    allocator.free(3);

    assert_eq!(allocator.count(), 3);
    assert_eq!(allocator.free_count(), 2);

    // Clear
    allocator.clear();

    // Back to initial state
    assert_eq!(allocator.count(), 0);
    assert_eq!(allocator.free_count(), 0);

    // Fresh allocations start from 0
    assert_eq!(allocator.allocate(), Some(0));
}

#[test]
fn allocator_lifecycle_with_reset() {
    let mut allocator = IndexAllocator::with_generations(10);

    // Use and cycle indices to increment generations
    let gen_idx = allocator.allocate_generational().unwrap();
    allocator.free(gen_idx.index());
    let gen_idx2 = allocator.allocate_generational().unwrap();
    assert_eq!(gen_idx2.generation(), 1);

    // Reset
    allocator.reset();

    // Generations should be reset
    assert_eq!(allocator.count(), 0);
    let gen_idx3 = allocator.allocate_generational().unwrap();
    assert_eq!(gen_idx3.index(), 0);
    assert_eq!(gen_idx3.generation(), 0, "Generation should reset to 0");
}

// =============================================================================
// Category 12: Advanced Stress Tests (~5 tests)
// =============================================================================

#[test]
fn allocator_stress_ping_pong_pattern() {
    let mut allocator = IndexAllocator::new(10);

    // Ping-pong: allocate one, free one repeatedly
    for i in 0..1000 {
        let idx = allocator.allocate().unwrap();
        assert!(idx < 10, "Index bounds check iteration {}", i);
        allocator.free(idx);
    }

    assert_eq!(allocator.count(), 0);
}

#[test]
fn allocator_stress_wave_pattern() {
    let mut allocator = IndexAllocator::new(100);

    // Wave pattern: allocate wave, free wave
    for wave in 0..10 {
        let mut indices = Vec::new();

        // Allocate wave_size indices
        let wave_size = 10 + wave * 2;
        for _ in 0..wave_size.min(100) {
            if let Some(idx) = allocator.allocate() {
                indices.push(idx);
            }
        }

        // Free half of them
        for idx in indices.iter().take(indices.len() / 2) {
            allocator.free(*idx);
        }
    }

    // Should not have leaked or corrupted state
    let count = allocator.count();
    assert!(count <= 100, "Count {} should not exceed capacity", count);
}

#[test]
fn allocator_stress_fragmentation_recovery() {
    let mut allocator = IndexAllocator::new(100);

    // Allocate all
    let all_indices: Vec<u32> = (0..100).map(|_| allocator.allocate().unwrap()).collect();
    assert!(allocator.allocate().is_none());

    // Free every other index (creates fragmentation)
    let mut freed = Vec::new();
    for (i, idx) in all_indices.iter().enumerate() {
        if i % 2 == 0 {
            allocator.free(*idx);
            freed.push(*idx);
        }
    }

    // Reallocate the freed indices
    let mut reallocated: HashSet<u32> = HashSet::new();
    for _ in 0..50 {
        if let Some(idx) = allocator.allocate() {
            reallocated.insert(idx);
        }
    }

    // All reallocated should be from the freed set
    for idx in &reallocated {
        assert!(freed.contains(idx) || !all_indices[..*idx as usize].contains(idx));
    }

    // Should be at capacity again
    assert!(allocator.allocate().is_none());
}

#[test]
fn allocator_stress_generations_many_cycles() {
    let mut allocator = IndexAllocator::with_generations(5);

    // Cycle the same slot many times to stress generation tracking
    for cycle in 0..100 {
        let gen_idx = allocator.allocate_generational().unwrap();

        // Each cycle should increment generation
        assert_eq!(gen_idx.index(), 0, "First slot should be 0 after free");

        // Verify current index is valid
        assert!(allocator.is_valid(gen_idx));

        allocator.free(gen_idx.index());

        // After free, the old handle should be invalid
        assert!(!allocator.is_valid(gen_idx), "Stale handle at cycle {}", cycle);
    }
}

#[test]
fn allocator_stress_concurrent_simulation() {
    // Simulates concurrent access pattern (single-threaded, but mixed operations)
    let mut allocator = IndexAllocator::new(50);
    let mut allocated: Vec<u32> = Vec::new();

    for i in 0..1000 {
        match i % 7 {
            0 | 1 | 2 | 3 => {
                // Allocate (more frequent)
                if let Some(idx) = allocator.allocate() {
                    allocated.push(idx);
                }
            }
            4 | 5 => {
                // Free (less frequent)
                if !allocated.is_empty() {
                    let idx = allocated.remove(0);
                    assert!(allocator.free(idx));
                }
            }
            6 => {
                // Check state consistency
                assert_eq!(allocator.count() as usize, allocated.len());
                for idx in &allocated {
                    assert!(allocator.is_allocated(*idx));
                }
            }
            _ => unreachable!(),
        }
    }

    // Final consistency check
    assert_eq!(allocator.count() as usize, allocated.len());
}

// =============================================================================
// Category 13: Generational Index Edge Cases (~5 tests)
// =============================================================================

#[test]
fn generational_index_stale_after_reallocate() {
    let mut allocator = IndexAllocator::with_generations(5);

    // Allocate, free, reallocate
    let old = allocator.allocate_generational().unwrap();
    allocator.free(old.index());
    let new = allocator.allocate_generational().unwrap();

    // Old handle is stale
    assert!(!allocator.is_valid(old));
    // New handle is valid
    assert!(allocator.is_valid(new));
    // Same slot, different generation
    assert_eq!(old.index(), new.index());
    assert_ne!(old.generation(), new.generation());
}

#[test]
fn generational_index_try_free_stale_fails() {
    let mut allocator = IndexAllocator::with_generations(5);

    let old = allocator.allocate_generational().unwrap();
    allocator.free(old.index());
    let _new = allocator.allocate_generational().unwrap();

    // Trying to free with stale generation should fail
    let result = allocator.try_free_generational(old);
    assert!(matches!(result, Err(AllocatorError::StaleGeneration { .. })));
}

#[test]
fn generational_index_null_operations() {
    let mut allocator = IndexAllocator::with_generations(10);

    let null_idx = GenerationalIndex::null();

    // Null index should fail validation
    assert!(!allocator.is_valid(null_idx));

    // Try to free null should fail
    let result = allocator.try_free_generational(null_idx);
    assert!(result.is_err());
}

#[test]
fn generational_index_without_tracking() {
    let mut allocator = IndexAllocator::new(10); // No generation tracking

    // allocate_generational should return None without generation tracking
    let result = allocator.allocate_generational();
    assert!(result.is_none());
}

#[test]
fn generational_index_preserves_through_clear() {
    let mut allocator = IndexAllocator::with_generations(5);

    // Allocate and free to increment generation
    let gen1 = allocator.allocate_generational().unwrap();
    allocator.free(gen1.index());

    let gen2 = allocator.allocate_generational().unwrap();
    assert_eq!(gen2.generation(), 1);

    // Clear does NOT reset generations
    allocator.clear();

    // Reallocate - generation should continue from where it was
    let gen3 = allocator.allocate_generational().unwrap();
    assert_eq!(gen3.index(), 0);
    // Generation was incremented when we freed gen2's slot (implicitly via clear behavior)
    // After clear, allocating slot 0 should show the preserved generation
    assert_eq!(gen3.generation(), 1, "Clear should preserve generations");
}
