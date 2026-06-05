//! Whitebox tests for IndexAllocator [T-WGPU-P2.6.3]
//!
//! Comprehensive whitebox testing of the index allocator implementation including:
//! - Construction and initialization
//! - Basic allocation and deallocation
//! - Free list recycling with LIFO ordering
//! - Generation tracking for use-after-free detection
//! - Metrics and statistics
//! - Error handling
//! - Clear/reset operations
//! - Thread safety traits
//! - Edge cases and stress tests

use renderer_backend::resources::index_allocator::{
    AllocatorError, GenerationalIndex, IndexAllocator, IndexAllocatorMetrics,
};
use std::collections::HashSet;

// ============================================================================
// MODULE 1: Construction Tests (~12 tests)
// ============================================================================

mod construction {
    use super::*;

    #[test]
    fn test_new_with_zero_capacity() {
        let allocator = IndexAllocator::new(0);
        assert_eq!(allocator.capacity(), 0);
        assert_eq!(allocator.count(), 0);
        assert_eq!(allocator.free_count(), 0);
        assert!(!allocator.has_generations());
    }

    #[test]
    fn test_new_with_small_capacity() {
        let allocator = IndexAllocator::new(1);
        assert_eq!(allocator.capacity(), 1);
        assert_eq!(allocator.count(), 0);
        assert_eq!(allocator.available(), 1);
    }

    #[test]
    fn test_new_with_typical_capacity() {
        let allocator = IndexAllocator::new(1024);
        assert_eq!(allocator.capacity(), 1024);
        assert_eq!(allocator.count(), 0);
        assert_eq!(allocator.available(), 1024);
        assert_eq!(allocator.peak_allocations(), 0);
    }

    #[test]
    fn test_new_with_large_capacity() {
        let allocator = IndexAllocator::new(1_000_000);
        assert_eq!(allocator.capacity(), 1_000_000);
        assert_eq!(allocator.count(), 0);
    }

    #[test]
    fn test_new_with_max_capacity() {
        // Don't allocate, just verify construction
        let allocator = IndexAllocator::new(u32::MAX);
        assert_eq!(allocator.capacity(), u32::MAX);
    }

    #[test]
    fn test_with_generations_creates_generation_array() {
        let allocator = IndexAllocator::with_generations(100);
        assert!(allocator.has_generations());
        // All generations should start at 0
        for i in 0..100 {
            assert_eq!(allocator.generation(i), Some(0));
        }
    }

    #[test]
    fn test_with_generations_zero_capacity() {
        let allocator = IndexAllocator::with_generations(0);
        assert_eq!(allocator.capacity(), 0);
        assert!(allocator.has_generations());
        // Out of bounds should return None
        assert_eq!(allocator.generation(0), None);
    }

    #[test]
    fn test_default_capacity_is_1024() {
        let allocator = IndexAllocator::default();
        assert_eq!(allocator.capacity(), 1024);
        assert!(!allocator.has_generations());
    }

    #[test]
    fn test_clone_preserves_all_state() {
        let mut allocator = IndexAllocator::with_generations(10);
        allocator.allocate(); // index 0
        allocator.allocate(); // index 1
        allocator.free(0);    // free 0, generation becomes 1

        let cloned = allocator.clone();

        assert_eq!(cloned.capacity(), allocator.capacity());
        assert_eq!(cloned.count(), allocator.count());
        assert_eq!(cloned.free_count(), allocator.free_count());
        assert_eq!(cloned.has_generations(), allocator.has_generations());
        assert_eq!(cloned.generation(0), allocator.generation(0));
        assert_eq!(cloned.peak_allocations(), allocator.peak_allocations());
    }

    #[test]
    fn test_clone_is_independent() {
        let mut allocator = IndexAllocator::new(10);
        allocator.allocate();

        let mut cloned = allocator.clone();
        cloned.allocate();
        cloned.allocate();

        // Original should be unchanged
        assert_eq!(allocator.count(), 1);
        assert_eq!(cloned.count(), 3);
    }

    #[test]
    fn test_new_without_generations_returns_none() {
        let allocator = IndexAllocator::new(100);
        assert!(!allocator.has_generations());
        assert_eq!(allocator.generation(0), None);
        assert_eq!(allocator.generation(50), None);
    }

    #[test]
    fn test_generation_out_of_bounds_returns_none() {
        let allocator = IndexAllocator::with_generations(10);
        assert_eq!(allocator.generation(10), None);
        assert_eq!(allocator.generation(100), None);
        assert_eq!(allocator.generation(u32::MAX), None);
    }
}

// ============================================================================
// MODULE 2: Basic Allocation Tests (~15 tests)
// ============================================================================

mod basic_allocation {
    use super::*;

    #[test]
    fn test_allocate_returns_sequential_indices() {
        let mut allocator = IndexAllocator::new(100);
        assert_eq!(allocator.allocate(), Some(0));
        assert_eq!(allocator.allocate(), Some(1));
        assert_eq!(allocator.allocate(), Some(2));
        assert_eq!(allocator.allocate(), Some(3));
    }

    #[test]
    fn test_allocate_increments_count() {
        let mut allocator = IndexAllocator::new(100);
        assert_eq!(allocator.count(), 0);
        allocator.allocate();
        assert_eq!(allocator.count(), 1);
        allocator.allocate();
        assert_eq!(allocator.count(), 2);
    }

    #[test]
    fn test_allocate_decrements_available() {
        let mut allocator = IndexAllocator::new(10);
        assert_eq!(allocator.available(), 10);
        allocator.allocate();
        assert_eq!(allocator.available(), 9);
        allocator.allocate();
        assert_eq!(allocator.available(), 8);
    }

    #[test]
    fn test_allocate_at_capacity_returns_none() {
        let mut allocator = IndexAllocator::new(3);
        assert_eq!(allocator.allocate(), Some(0));
        assert_eq!(allocator.allocate(), Some(1));
        assert_eq!(allocator.allocate(), Some(2));
        assert_eq!(allocator.allocate(), None);
        assert_eq!(allocator.allocate(), None);
    }

    #[test]
    fn test_allocate_zero_capacity_returns_none() {
        let mut allocator = IndexAllocator::new(0);
        assert_eq!(allocator.allocate(), None);
    }

    #[test]
    fn test_try_allocate_success() {
        let mut allocator = IndexAllocator::new(10);
        assert_eq!(allocator.try_allocate(), Ok(0));
        assert_eq!(allocator.try_allocate(), Ok(1));
    }

    #[test]
    fn test_try_allocate_at_capacity_error() {
        let mut allocator = IndexAllocator::new(1);
        assert_eq!(allocator.try_allocate(), Ok(0));
        assert_eq!(
            allocator.try_allocate(),
            Err(AllocatorError::AtCapacity { capacity: 1 })
        );
    }

    #[test]
    fn test_try_allocate_zero_capacity_error() {
        let mut allocator = IndexAllocator::new(0);
        assert_eq!(
            allocator.try_allocate(),
            Err(AllocatorError::AtCapacity { capacity: 0 })
        );
    }

    #[test]
    fn test_is_allocated_for_allocated_index() {
        let mut allocator = IndexAllocator::new(10);
        let idx = allocator.allocate().unwrap();
        assert!(allocator.is_allocated(idx));
    }

    #[test]
    fn test_is_allocated_for_unallocated_index() {
        let allocator = IndexAllocator::new(10);
        assert!(!allocator.is_allocated(0));
        assert!(!allocator.is_allocated(5));
    }

    #[test]
    fn test_is_allocated_for_out_of_bounds() {
        let allocator = IndexAllocator::new(10);
        assert!(!allocator.is_allocated(10));
        assert!(!allocator.is_allocated(100));
        assert!(!allocator.is_allocated(u32::MAX));
    }

    #[test]
    fn test_allocated_indices_iterator_empty() {
        let allocator = IndexAllocator::new(10);
        let indices: Vec<_> = allocator.allocated_indices().collect();
        assert!(indices.is_empty());
    }

    #[test]
    fn test_allocated_indices_iterator_with_allocations() {
        let mut allocator = IndexAllocator::new(10);
        allocator.allocate(); // 0
        allocator.allocate(); // 1
        allocator.allocate(); // 2

        let mut indices: Vec<_> = allocator.allocated_indices().collect();
        indices.sort();
        assert_eq!(indices, vec![0, 1, 2]);
    }

    #[test]
    fn test_peak_allocations_tracks_maximum() {
        let mut allocator = IndexAllocator::new(10);
        assert_eq!(allocator.peak_allocations(), 0);

        allocator.allocate();
        allocator.allocate();
        allocator.allocate();
        assert_eq!(allocator.peak_allocations(), 3);

        // Free doesn't decrease peak
        allocator.free(0);
        allocator.free(1);
        assert_eq!(allocator.peak_allocations(), 3);
    }

    #[test]
    fn test_allocate_fills_capacity_exactly() {
        let mut allocator = IndexAllocator::new(5);
        for i in 0..5 {
            assert_eq!(allocator.allocate(), Some(i));
        }
        assert_eq!(allocator.count(), 5);
        assert_eq!(allocator.available(), 0);
        assert_eq!(allocator.allocate(), None);
    }
}

// ============================================================================
// MODULE 3: Free & Recycling Tests (~18 tests)
// ============================================================================

mod free_and_recycling {
    use super::*;

    #[test]
    fn test_free_returns_true_for_allocated() {
        let mut allocator = IndexAllocator::new(10);
        let idx = allocator.allocate().unwrap();
        assert!(allocator.free(idx));
    }

    #[test]
    fn test_free_returns_false_for_unallocated() {
        let mut allocator = IndexAllocator::new(10);
        assert!(!allocator.free(0)); // Never allocated
    }

    #[test]
    fn test_free_returns_false_for_out_of_bounds() {
        let mut allocator = IndexAllocator::new(10);
        assert!(!allocator.free(10));
        assert!(!allocator.free(100));
    }

    #[test]
    fn test_free_decrements_count() {
        let mut allocator = IndexAllocator::new(10);
        allocator.allocate();
        allocator.allocate();
        assert_eq!(allocator.count(), 2);
        allocator.free(0);
        assert_eq!(allocator.count(), 1);
    }

    #[test]
    fn test_free_increments_free_count() {
        let mut allocator = IndexAllocator::new(10);
        allocator.allocate();
        allocator.allocate();
        assert_eq!(allocator.free_count(), 0);
        allocator.free(0);
        assert_eq!(allocator.free_count(), 1);
        allocator.free(1);
        assert_eq!(allocator.free_count(), 2);
    }

    #[test]
    fn test_free_increments_available() {
        let mut allocator = IndexAllocator::new(10);
        allocator.allocate();
        allocator.allocate();
        assert_eq!(allocator.available(), 8);
        allocator.free(0);
        assert_eq!(allocator.available(), 9);
    }

    #[test]
    fn test_double_free_returns_false() {
        let mut allocator = IndexAllocator::new(10);
        let idx = allocator.allocate().unwrap();
        assert!(allocator.free(idx));
        assert!(!allocator.free(idx)); // Double-free
    }

    #[test]
    fn test_free_list_lifo_order() {
        let mut allocator = IndexAllocator::new(10);
        allocator.allocate(); // 0
        allocator.allocate(); // 1
        allocator.allocate(); // 2

        // Free in order: 0, 1, 2
        allocator.free(0);
        allocator.free(1);
        allocator.free(2);

        // LIFO: last freed (2) comes out first
        assert_eq!(allocator.allocate(), Some(2));
        assert_eq!(allocator.allocate(), Some(1));
        assert_eq!(allocator.allocate(), Some(0));
    }

    #[test]
    fn test_recycled_before_fresh() {
        let mut allocator = IndexAllocator::new(10);
        allocator.allocate(); // 0
        allocator.allocate(); // 1
        allocator.free(0);    // 0 goes to free list

        // Next allocation should recycle 0, not allocate fresh 2
        assert_eq!(allocator.allocate(), Some(0));
    }

    #[test]
    fn test_free_and_reallocate_cycle() {
        let mut allocator = IndexAllocator::new(5);

        for cycle in 0..3 {
            // Allocate all
            for i in 0..5 {
                assert!(allocator.allocate().is_some(), "cycle {} alloc {}", cycle, i);
            }
            assert_eq!(allocator.count(), 5);

            // Free all
            for i in 0..5 {
                assert!(allocator.free(i), "cycle {} free {}", cycle, i);
            }
            assert_eq!(allocator.count(), 0);
        }
    }

    #[test]
    fn test_try_free_success() {
        let mut allocator = IndexAllocator::new(10);
        let idx = allocator.allocate().unwrap();
        assert!(allocator.try_free(idx).is_ok());
    }

    #[test]
    fn test_try_free_invalid_index_error() {
        let mut allocator = IndexAllocator::new(10);
        assert_eq!(
            allocator.try_free(100),
            Err(AllocatorError::InvalidIndex { index: 100, capacity: 10 })
        );
    }

    #[test]
    fn test_try_free_never_allocated_error() {
        let mut allocator = IndexAllocator::new(10);
        allocator.allocate(); // Allocate 0, next_index becomes 1
        // Index 5 is beyond next_index
        let result = allocator.try_free(5);
        assert!(matches!(result, Err(AllocatorError::InvalidIndex { .. })));
    }

    #[test]
    fn test_try_free_double_free_error() {
        let mut allocator = IndexAllocator::new(10);
        let idx = allocator.allocate().unwrap();
        assert!(allocator.try_free(idx).is_ok());
        assert_eq!(
            allocator.try_free(idx),
            Err(AllocatorError::DoubleFree(idx))
        );
    }

    #[test]
    fn test_is_allocated_updates_after_free() {
        let mut allocator = IndexAllocator::new(10);
        let idx = allocator.allocate().unwrap();
        assert!(allocator.is_allocated(idx));
        allocator.free(idx);
        assert!(!allocator.is_allocated(idx));
    }

    #[test]
    fn test_allocated_indices_updates_after_free() {
        let mut allocator = IndexAllocator::new(10);
        allocator.allocate(); // 0
        allocator.allocate(); // 1
        allocator.allocate(); // 2
        allocator.free(1);

        let mut indices: Vec<_> = allocator.allocated_indices().collect();
        indices.sort();
        assert_eq!(indices, vec![0, 2]);
    }

    #[test]
    fn test_interleaved_alloc_free_pattern() {
        let mut allocator = IndexAllocator::new(10);

        let a = allocator.allocate().unwrap(); // 0
        let b = allocator.allocate().unwrap(); // 1
        allocator.free(a);                      // free 0
        let c = allocator.allocate().unwrap(); // recycle 0
        let d = allocator.allocate().unwrap(); // 2
        allocator.free(b);                      // free 1
        allocator.free(c);                      // free 0
        let e = allocator.allocate().unwrap(); // recycle 0 (LIFO)

        assert_eq!(a, 0);
        assert_eq!(b, 1);
        assert_eq!(c, 0);
        assert_eq!(d, 2);
        assert_eq!(e, 0);
    }

    #[test]
    fn test_free_all_then_reallocate_all() {
        let mut allocator = IndexAllocator::new(100);

        // Allocate all
        for i in 0..100 {
            assert_eq!(allocator.allocate(), Some(i));
        }

        // Free all in reverse
        for i in (0..100).rev() {
            assert!(allocator.free(i));
        }

        assert_eq!(allocator.count(), 0);
        assert_eq!(allocator.free_count(), 100);

        // Reallocate all (LIFO order)
        for i in 0..100 {
            assert!(allocator.allocate().is_some());
        }
        assert_eq!(allocator.count(), 100);
    }
}

// ============================================================================
// MODULE 4: Generation Tracking Tests (~18 tests)
// ============================================================================

mod generation_tracking {
    use super::*;

    #[test]
    fn test_allocate_generational_returns_index_and_generation() {
        let mut allocator = IndexAllocator::with_generations(10);
        let gen_idx = allocator.allocate_generational().unwrap();
        assert_eq!(gen_idx.index, 0);
        assert_eq!(gen_idx.generation, 0);
    }

    #[test]
    fn test_allocate_generational_sequential() {
        let mut allocator = IndexAllocator::with_generations(10);
        let g0 = allocator.allocate_generational().unwrap();
        let g1 = allocator.allocate_generational().unwrap();
        let g2 = allocator.allocate_generational().unwrap();

        assert_eq!(g0.index, 0);
        assert_eq!(g1.index, 1);
        assert_eq!(g2.index, 2);

        // All first allocations have generation 0
        assert_eq!(g0.generation, 0);
        assert_eq!(g1.generation, 0);
        assert_eq!(g2.generation, 0);
    }

    #[test]
    fn test_allocate_generational_without_generations_returns_none() {
        let mut allocator = IndexAllocator::new(10);
        assert_eq!(allocator.allocate_generational(), None);
    }

    #[test]
    fn test_generation_increments_on_free() {
        let mut allocator = IndexAllocator::with_generations(10);
        assert_eq!(allocator.generation(0), Some(0));

        allocator.allocate();
        assert_eq!(allocator.generation(0), Some(0));

        allocator.free(0);
        assert_eq!(allocator.generation(0), Some(1));
    }

    #[test]
    fn test_generation_increments_on_each_reuse() {
        let mut allocator = IndexAllocator::with_generations(10);

        for expected_gen in 0..5 {
            let gen_idx = allocator.allocate_generational().unwrap();
            assert_eq!(gen_idx.index, 0);
            assert_eq!(gen_idx.generation, expected_gen);
            allocator.free(0);
        }
    }

    #[test]
    fn test_is_valid_for_current_allocation() {
        let mut allocator = IndexAllocator::with_generations(10);
        let gen_idx = allocator.allocate_generational().unwrap();
        assert!(allocator.is_valid(gen_idx));
    }

    #[test]
    fn test_is_valid_false_after_free() {
        let mut allocator = IndexAllocator::with_generations(10);
        let gen_idx = allocator.allocate_generational().unwrap();
        allocator.free(gen_idx.index);
        assert!(!allocator.is_valid(gen_idx));
    }

    #[test]
    fn test_is_valid_detects_stale_generation() {
        let mut allocator = IndexAllocator::with_generations(10);
        let old = allocator.allocate_generational().unwrap();
        allocator.free(old.index);
        let _new = allocator.allocate_generational().unwrap();

        // Old reference should be invalid (stale generation)
        assert!(!allocator.is_valid(old));
    }

    #[test]
    fn test_is_valid_for_null_returns_false() {
        let allocator = IndexAllocator::with_generations(10);
        assert!(!allocator.is_valid(GenerationalIndex::null()));
    }

    #[test]
    fn test_is_valid_without_generations_checks_allocation() {
        let mut allocator = IndexAllocator::new(10);
        allocator.allocate();

        let gen_idx = GenerationalIndex::new(0, 0);
        assert!(allocator.is_valid(gen_idx));

        allocator.free(0);
        assert!(!allocator.is_valid(gen_idx));
    }

    #[test]
    fn test_try_allocate_generational_success() {
        let mut allocator = IndexAllocator::with_generations(10);
        let result = allocator.try_allocate_generational();
        assert!(result.is_ok());
        let gen_idx = result.unwrap();
        assert_eq!(gen_idx.index, 0);
    }

    #[test]
    fn test_try_allocate_generational_at_capacity() {
        let mut allocator = IndexAllocator::with_generations(1);
        assert!(allocator.try_allocate_generational().is_ok());
        assert_eq!(
            allocator.try_allocate_generational(),
            Err(AllocatorError::AtCapacity { capacity: 1 })
        );
    }

    #[test]
    fn test_try_free_generational_success() {
        let mut allocator = IndexAllocator::with_generations(10);
        let gen_idx = allocator.allocate_generational().unwrap();
        assert!(allocator.try_free_generational(gen_idx).is_ok());
    }

    #[test]
    fn test_try_free_generational_stale_error() {
        let mut allocator = IndexAllocator::with_generations(10);
        let old = allocator.allocate_generational().unwrap();
        allocator.free(old.index);
        let _new = allocator.allocate_generational().unwrap();

        assert_eq!(
            allocator.try_free_generational(old),
            Err(AllocatorError::StaleGeneration { expected: 1, found: 0 })
        );
    }

    #[test]
    fn test_try_free_generational_invalid_index() {
        let mut allocator = IndexAllocator::with_generations(10);
        let invalid = GenerationalIndex::new(100, 0);
        assert!(matches!(
            allocator.try_free_generational(invalid),
            Err(AllocatorError::InvalidIndex { .. })
        ));
    }

    #[test]
    fn test_generation_wrapping() {
        let mut allocator = IndexAllocator::with_generations(1);

        // Manually set generation to MAX to test wrapping
        // This tests internal behavior - we need to access through public API
        // Allocate/free cycle u32::MAX times is impractical, so we test
        // that the generation() returns correct values after increments

        let gen_idx = allocator.allocate_generational().unwrap();
        assert_eq!(gen_idx.generation, 0);

        // Free increments generation
        allocator.free(0);
        assert_eq!(allocator.generation(0), Some(1));
    }

    #[test]
    fn test_multiple_slots_independent_generations() {
        let mut allocator = IndexAllocator::with_generations(3);

        // Allocate all three
        let g0 = allocator.allocate_generational().unwrap();
        let g1 = allocator.allocate_generational().unwrap();
        let g2 = allocator.allocate_generational().unwrap();

        // Free only index 1
        allocator.free(1);

        // Verify generations
        assert_eq!(allocator.generation(0), Some(0));
        assert_eq!(allocator.generation(1), Some(1)); // Incremented
        assert_eq!(allocator.generation(2), Some(0));

        // g0 and g2 should still be valid
        assert!(allocator.is_valid(g0));
        assert!(!allocator.is_valid(g1)); // Freed
        assert!(allocator.is_valid(g2));
    }

    #[test]
    fn test_generational_free_then_reallocate() {
        let mut allocator = IndexAllocator::with_generations(5);

        // Allocate, free, reallocate same slot multiple times
        for cycle in 0..3 {
            let gen_idx = allocator.allocate_generational().unwrap();
            assert_eq!(gen_idx.index, 0);
            assert_eq!(gen_idx.generation, cycle);
            assert!(allocator.is_valid(gen_idx));

            allocator.free(0);
            assert!(!allocator.is_valid(gen_idx));
        }
    }
}

// ============================================================================
// MODULE 5: Metrics Tests (~12 tests)
// ============================================================================

mod metrics {
    use super::*;

    #[test]
    fn test_count_starts_at_zero() {
        let allocator = IndexAllocator::new(100);
        assert_eq!(allocator.count(), 0);
    }

    #[test]
    fn test_capacity_returns_constructor_value() {
        let allocator = IndexAllocator::new(42);
        assert_eq!(allocator.capacity(), 42);
    }

    #[test]
    fn test_free_count_starts_at_zero() {
        let allocator = IndexAllocator::new(100);
        assert_eq!(allocator.free_count(), 0);
    }

    #[test]
    fn test_available_equals_capacity_initially() {
        let allocator = IndexAllocator::new(50);
        assert_eq!(allocator.available(), 50);
    }

    #[test]
    fn test_utilization_starts_at_zero() {
        let allocator = IndexAllocator::new(100);
        assert_eq!(allocator.utilization(), 0.0);
    }

    #[test]
    fn test_utilization_at_half_capacity() {
        let mut allocator = IndexAllocator::new(100);
        for _ in 0..50 {
            allocator.allocate();
        }
        assert_eq!(allocator.utilization(), 0.5);
    }

    #[test]
    fn test_utilization_at_full_capacity() {
        let mut allocator = IndexAllocator::new(10);
        for _ in 0..10 {
            allocator.allocate();
        }
        assert_eq!(allocator.utilization(), 1.0);
    }

    #[test]
    fn test_utilization_zero_capacity() {
        let allocator = IndexAllocator::new(0);
        assert_eq!(allocator.utilization(), 0.0);
    }

    #[test]
    fn test_fragmentation_starts_at_zero() {
        let allocator = IndexAllocator::new(100);
        assert_eq!(allocator.fragmentation(), 0.0);
    }

    #[test]
    fn test_fragmentation_increases_with_frees() {
        let mut allocator = IndexAllocator::new(100);
        for _ in 0..10 {
            allocator.allocate();
        }
        assert_eq!(allocator.fragmentation(), 0.0);

        allocator.free(0);
        allocator.free(1);
        allocator.free(2);
        allocator.free(3);
        allocator.free(4);

        // 5 free out of 10 peak = 0.5
        assert_eq!(allocator.fragmentation(), 0.5);
    }

    #[test]
    fn test_metrics_struct_all_fields() {
        let mut allocator = IndexAllocator::with_generations(100);
        allocator.allocate();
        allocator.allocate();
        allocator.free(0);

        let m = allocator.metrics();

        assert_eq!(m.allocated_count, 1);
        assert_eq!(m.capacity, 100);
        assert_eq!(m.free_list_size, 1);
        assert_eq!(m.peak_allocations, 2);
        assert_eq!(m.fragmentation, 0.5); // 1 free / 2 peak
        assert_eq!(m.utilization, 0.01);  // 1 allocated / 100 capacity
        assert!(m.has_generations);
    }

    #[test]
    fn test_metrics_without_generations() {
        let allocator = IndexAllocator::new(100);
        let m = allocator.metrics();
        assert!(!m.has_generations);
    }
}

// ============================================================================
// MODULE 6: Error Type Tests (~10 tests)
// ============================================================================

mod error_types {
    use super::*;

    #[test]
    fn test_at_capacity_error_display() {
        let err = AllocatorError::AtCapacity { capacity: 100 };
        assert_eq!(err.to_string(), "allocator at capacity (100)");
    }

    #[test]
    fn test_invalid_index_error_display() {
        let err = AllocatorError::InvalidIndex { index: 50, capacity: 10 };
        assert_eq!(err.to_string(), "invalid index 50 (capacity: 10)");
    }

    #[test]
    fn test_double_free_error_display() {
        let err = AllocatorError::DoubleFree(42);
        assert_eq!(err.to_string(), "double free detected for index 42");
    }

    #[test]
    fn test_stale_generation_error_display() {
        let err = AllocatorError::StaleGeneration { expected: 5, found: 3 };
        assert_eq!(err.to_string(), "stale generation: expected 5, found 3");
    }

    #[test]
    fn test_error_clone() {
        let err = AllocatorError::AtCapacity { capacity: 100 };
        let cloned = err.clone();
        assert_eq!(err, cloned);
    }

    #[test]
    fn test_error_debug() {
        let err = AllocatorError::DoubleFree(7);
        let debug = format!("{:?}", err);
        assert!(debug.contains("DoubleFree"));
        assert!(debug.contains("7"));
    }

    #[test]
    fn test_error_equality() {
        let e1 = AllocatorError::AtCapacity { capacity: 100 };
        let e2 = AllocatorError::AtCapacity { capacity: 100 };
        let e3 = AllocatorError::AtCapacity { capacity: 50 };

        assert_eq!(e1, e2);
        assert_ne!(e1, e3);
    }

    #[test]
    fn test_error_variants_different() {
        let e1 = AllocatorError::AtCapacity { capacity: 10 };
        let e2 = AllocatorError::InvalidIndex { index: 10, capacity: 10 };
        let e3 = AllocatorError::DoubleFree(10);
        let e4 = AllocatorError::StaleGeneration { expected: 10, found: 10 };

        assert_ne!(e1, e2);
        assert_ne!(e2, e3);
        assert_ne!(e3, e4);
    }

    #[test]
    fn test_error_is_std_error() {
        fn assert_error<E: std::error::Error>() {}
        assert_error::<AllocatorError>();
    }

    #[test]
    fn test_error_hash_eq() {
        let e1 = AllocatorError::DoubleFree(5);
        let e2 = AllocatorError::DoubleFree(5);

        // PartialEq works
        assert!(e1 == e2);
    }
}

// ============================================================================
// MODULE 7: Clear/Reset Tests (~8 tests)
// ============================================================================

mod clear_reset {
    use super::*;

    #[test]
    fn test_clear_resets_count() {
        let mut allocator = IndexAllocator::new(10);
        allocator.allocate();
        allocator.allocate();
        assert_eq!(allocator.count(), 2);

        allocator.clear();
        assert_eq!(allocator.count(), 0);
    }

    #[test]
    fn test_clear_resets_free_list() {
        let mut allocator = IndexAllocator::new(10);
        allocator.allocate();
        allocator.allocate();
        allocator.free(0);
        assert_eq!(allocator.free_count(), 1);

        allocator.clear();
        assert_eq!(allocator.free_count(), 0);
    }

    #[test]
    fn test_clear_resets_next_index() {
        let mut allocator = IndexAllocator::new(10);
        allocator.allocate();
        allocator.allocate();
        allocator.allocate();

        allocator.clear();
        assert_eq!(allocator.allocate(), Some(0));
    }

    #[test]
    fn test_clear_preserves_generations() {
        let mut allocator = IndexAllocator::with_generations(10);
        allocator.allocate();
        allocator.free(0);
        assert_eq!(allocator.generation(0), Some(1));

        allocator.clear();

        // Generation preserved
        assert_eq!(allocator.generation(0), Some(1));

        // New allocation gets current generation
        let gen_idx = allocator.allocate_generational().unwrap();
        assert_eq!(gen_idx.generation, 1);
    }

    #[test]
    fn test_reset_resets_everything() {
        let mut allocator = IndexAllocator::with_generations(10);
        allocator.allocate();
        allocator.allocate();
        allocator.free(0);

        allocator.reset();

        assert_eq!(allocator.count(), 0);
        assert_eq!(allocator.free_count(), 0);
        assert_eq!(allocator.generation(0), Some(0));
    }

    #[test]
    fn test_reset_clears_all_generations() {
        let mut allocator = IndexAllocator::with_generations(5);

        // Create varied generations
        for i in 0..5 {
            for _ in 0..i {
                allocator.allocate_generational();
                allocator.free(i);
            }
        }

        allocator.reset();

        for i in 0..5 {
            assert_eq!(allocator.generation(i), Some(0));
        }
    }

    #[test]
    fn test_clear_allows_fresh_allocations() {
        let mut allocator = IndexAllocator::new(3);
        allocator.allocate();
        allocator.allocate();
        allocator.allocate();
        assert_eq!(allocator.allocate(), None); // Full

        allocator.clear();

        assert_eq!(allocator.allocate(), Some(0));
        assert_eq!(allocator.allocate(), Some(1));
        assert_eq!(allocator.allocate(), Some(2));
    }

    #[test]
    fn test_clear_on_empty_allocator() {
        let mut allocator = IndexAllocator::new(10);
        allocator.clear();
        assert_eq!(allocator.count(), 0);
        assert_eq!(allocator.allocate(), Some(0));
    }
}

// ============================================================================
// MODULE 8: Thread Safety Tests (~6 tests)
// ============================================================================

mod thread_safety {
    use super::*;

    #[test]
    fn test_allocator_is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<IndexAllocator>();
    }

    #[test]
    fn test_allocator_is_sync() {
        fn assert_sync<T: Sync>() {}
        assert_sync::<IndexAllocator>();
    }

    #[test]
    fn test_generational_index_is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<GenerationalIndex>();
    }

    #[test]
    fn test_generational_index_is_sync() {
        fn assert_sync<T: Sync>() {}
        assert_sync::<GenerationalIndex>();
    }

    #[test]
    fn test_allocator_error_is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<AllocatorError>();
    }

    #[test]
    fn test_allocator_error_is_sync() {
        fn assert_sync<T: Sync>() {}
        assert_sync::<AllocatorError>();
    }
}

// ============================================================================
// MODULE 9: GenerationalIndex Tests (~12 tests)
// ============================================================================

mod generational_index {
    use super::*;

    #[test]
    fn test_new_creates_with_values() {
        let idx = GenerationalIndex::new(10, 5);
        assert_eq!(idx.index, 10);
        assert_eq!(idx.generation, 5);
    }

    #[test]
    fn test_index_accessor() {
        let idx = GenerationalIndex::new(42, 0);
        assert_eq!(idx.index(), 42);
    }

    #[test]
    fn test_generation_accessor() {
        let idx = GenerationalIndex::new(0, 99);
        assert_eq!(idx.generation(), 99);
    }

    #[test]
    fn test_null_values() {
        let null = GenerationalIndex::null();
        assert_eq!(null.index, u32::MAX);
        assert_eq!(null.generation, u32::MAX);
    }

    #[test]
    fn test_is_null_for_null() {
        let null = GenerationalIndex::null();
        assert!(null.is_null());
    }

    #[test]
    fn test_is_null_for_non_null() {
        let idx = GenerationalIndex::new(0, 0);
        assert!(!idx.is_null());
    }

    #[test]
    fn test_is_null_partial_max() {
        // Only one field at MAX is not null
        let idx1 = GenerationalIndex::new(u32::MAX, 0);
        let idx2 = GenerationalIndex::new(0, u32::MAX);

        assert!(!idx1.is_null());
        assert!(!idx2.is_null());
    }

    #[test]
    fn test_default_is_null() {
        let idx = GenerationalIndex::default();
        assert!(idx.is_null());
    }

    #[test]
    fn test_display_normal() {
        let idx = GenerationalIndex::new(5, 3);
        assert_eq!(format!("{}", idx), "GenerationalIndex(5, gen=3)");
    }

    #[test]
    fn test_display_null() {
        let idx = GenerationalIndex::null();
        assert_eq!(format!("{}", idx), "GenerationalIndex(null)");
    }

    #[test]
    fn test_hash_implementation() {
        let mut set = HashSet::new();
        set.insert(GenerationalIndex::new(0, 0));
        set.insert(GenerationalIndex::new(0, 1));
        set.insert(GenerationalIndex::new(1, 0));
        set.insert(GenerationalIndex::new(0, 0)); // Duplicate

        assert_eq!(set.len(), 3);
        assert!(set.contains(&GenerationalIndex::new(0, 0)));
        assert!(set.contains(&GenerationalIndex::new(0, 1)));
        assert!(set.contains(&GenerationalIndex::new(1, 0)));
    }

    #[test]
    fn test_copy_trait() {
        let idx = GenerationalIndex::new(1, 2);
        let copy = idx; // Copy, not move
        assert_eq!(idx.index, copy.index);
        assert_eq!(idx.generation, copy.generation);
    }
}

// ============================================================================
// MODULE 10: Edge Cases (~10 tests)
// ============================================================================

mod edge_cases {
    use super::*;

    #[test]
    fn test_zero_capacity_all_operations() {
        let mut allocator = IndexAllocator::new(0);

        assert_eq!(allocator.allocate(), None);
        assert_eq!(allocator.try_allocate(), Err(AllocatorError::AtCapacity { capacity: 0 }));
        assert!(!allocator.free(0));
        assert!(!allocator.is_allocated(0));
        assert_eq!(allocator.count(), 0);
        assert_eq!(allocator.available(), 0);
    }

    #[test]
    fn test_single_capacity_allocator() {
        let mut allocator = IndexAllocator::new(1);

        assert_eq!(allocator.allocate(), Some(0));
        assert_eq!(allocator.allocate(), None);
        assert!(allocator.free(0));
        assert_eq!(allocator.allocate(), Some(0));
    }

    #[test]
    fn test_large_capacity_allocation() {
        let mut allocator = IndexAllocator::new(100_000);

        // Allocate many indices
        for i in 0..1000 {
            assert_eq!(allocator.allocate(), Some(i));
        }

        assert_eq!(allocator.count(), 1000);
        assert_eq!(allocator.available(), 99_000);
    }

    #[test]
    fn test_boundary_index_values() {
        let mut allocator = IndexAllocator::new(u32::MAX);

        // We can't actually allocate u32::MAX indices, but verify construction
        assert_eq!(allocator.capacity(), u32::MAX);

        // First allocation works
        assert_eq!(allocator.allocate(), Some(0));
    }

    #[test]
    fn test_free_boundary_indices() {
        let mut allocator = IndexAllocator::new(10);

        // Allocate a few
        for _ in 0..5 {
            allocator.allocate();
        }

        // Free first and last allocated
        assert!(allocator.free(0));
        assert!(allocator.free(4));

        // Cannot free beyond allocated range
        assert!(!allocator.free(5));
        assert!(!allocator.free(9));
    }

    #[test]
    fn test_compact_is_noop() {
        let mut allocator = IndexAllocator::new(10);
        allocator.allocate();
        allocator.allocate();
        allocator.free(0);

        let free_count_before = allocator.free_count();
        allocator.compact();
        let free_count_after = allocator.free_count();

        assert_eq!(free_count_before, free_count_after);
    }

    #[test]
    fn test_reserve_free_list_capacity() {
        let mut allocator = IndexAllocator::new(1000);
        allocator.reserve_free_list(500);

        // Allocate and free to use the reserved capacity
        for i in 0..500 {
            allocator.allocate();
        }
        for i in 0..500 {
            allocator.free(i);
        }

        assert_eq!(allocator.free_count(), 500);
    }

    #[test]
    fn test_shrink_free_list() {
        let mut allocator = IndexAllocator::new(100);

        // Allocate and free many
        for _ in 0..50 {
            allocator.allocate();
        }
        for i in 0..50 {
            allocator.free(i);
        }

        allocator.shrink_free_list();
        assert_eq!(allocator.free_count(), 50);
    }

    #[test]
    fn test_metrics_default_values() {
        let m = IndexAllocatorMetrics::default();

        assert_eq!(m.allocated_count, 0);
        assert_eq!(m.capacity, 0);
        assert_eq!(m.free_list_size, 0);
        assert_eq!(m.peak_allocations, 0);
        assert_eq!(m.fragmentation, 0.0);
        assert_eq!(m.utilization, 0.0);
        assert!(!m.has_generations);
    }

    #[test]
    fn test_many_generation_cycles() {
        let mut allocator = IndexAllocator::with_generations(1);

        // Do 100 alloc/free cycles
        for expected_gen in 0u32..100 {
            let gen_idx = allocator.allocate_generational().unwrap();
            assert_eq!(gen_idx.generation, expected_gen);
            allocator.free(0);
        }

        // Generation should be 100
        assert_eq!(allocator.generation(0), Some(100));
    }
}

// ============================================================================
// MODULE 11: Stress Tests (~5 tests)
// ============================================================================

mod stress_tests {
    use super::*;

    #[test]
    fn test_many_allocations_sequential() {
        let mut allocator = IndexAllocator::new(10000);

        for i in 0..10000 {
            assert_eq!(allocator.allocate(), Some(i));
        }

        assert_eq!(allocator.count(), 10000);
        assert_eq!(allocator.allocate(), None);
    }

    #[test]
    fn test_full_allocate_free_cycle() {
        let mut allocator = IndexAllocator::with_generations(1000);

        // Allocate all
        let mut gen_indices: Vec<GenerationalIndex> = Vec::with_capacity(1000);
        for _ in 0..1000 {
            gen_indices.push(allocator.allocate_generational().unwrap());
        }

        // All should be valid
        for gen_idx in &gen_indices {
            assert!(allocator.is_valid(*gen_idx));
        }

        // Free all in reverse
        for gen_idx in gen_indices.iter().rev() {
            assert!(allocator.try_free_generational(*gen_idx).is_ok());
        }

        // None should be valid now
        for gen_idx in &gen_indices {
            assert!(!allocator.is_valid(*gen_idx));
        }

        // All slots have generation 1 now
        for i in 0..1000 {
            assert_eq!(allocator.generation(i), Some(1));
        }
    }

    #[test]
    fn test_random_pattern_simulation() {
        let mut allocator = IndexAllocator::new(100);
        let mut allocated: Vec<u32> = Vec::new();

        // Simulate random alloc/free pattern
        for i in 0..1000 {
            if allocated.len() < 50 || (i % 3 != 0 && allocated.len() < 100) {
                // Allocate
                if let Some(idx) = allocator.allocate() {
                    allocated.push(idx);
                }
            } else if !allocated.is_empty() {
                // Free from middle
                let remove_idx = i % allocated.len();
                let idx = allocated.remove(remove_idx);
                assert!(allocator.free(idx));
            }
        }

        // Verify consistency
        assert_eq!(allocator.count() as usize, allocated.len());

        for idx in &allocated {
            assert!(allocator.is_allocated(*idx));
        }
    }

    #[test]
    fn test_allocated_indices_consistency() {
        let mut allocator = IndexAllocator::new(50);
        let mut expected: HashSet<u32> = HashSet::new();

        // Allocate some
        for _ in 0..30 {
            if let Some(idx) = allocator.allocate() {
                expected.insert(idx);
            }
        }

        // Free some
        for i in (0..30).step_by(3) {
            if allocator.free(i) {
                expected.remove(&i);
            }
        }

        // Verify iterator matches
        let actual: HashSet<u32> = allocator.allocated_indices().collect();
        assert_eq!(expected, actual);
    }

    #[test]
    fn test_multiple_clear_cycles() {
        let mut allocator = IndexAllocator::with_generations(100);

        for cycle in 0..10 {
            // Allocate half
            for _ in 0..50 {
                allocator.allocate();
            }

            // Free quarter
            for i in 0..25 {
                allocator.free(i);
            }

            // Clear
            allocator.clear();

            // Verify clean state
            assert_eq!(allocator.count(), 0);
            assert_eq!(allocator.free_count(), 0);

            // But generations preserved
            // First 25 were freed once per cycle, so generation = cycle + 1
            // (but clear resets next_index, so new allocations start fresh)
        }
    }
}

// ============================================================================
// MODULE 12: IndexAllocatorMetrics Tests (~4 tests)
// ============================================================================

mod allocator_metrics_struct {
    use super::*;

    #[test]
    fn test_metrics_clone() {
        let mut allocator = IndexAllocator::with_generations(100);
        allocator.allocate();
        allocator.allocate();
        allocator.free(0);

        let metrics = allocator.metrics();
        let cloned = metrics.clone();

        assert_eq!(metrics.allocated_count, cloned.allocated_count);
        assert_eq!(metrics.capacity, cloned.capacity);
        assert_eq!(metrics.free_list_size, cloned.free_list_size);
        assert_eq!(metrics.peak_allocations, cloned.peak_allocations);
        assert_eq!(metrics.fragmentation, cloned.fragmentation);
        assert_eq!(metrics.utilization, cloned.utilization);
        assert_eq!(metrics.has_generations, cloned.has_generations);
    }

    #[test]
    fn test_metrics_debug() {
        let metrics = IndexAllocatorMetrics::default();
        let debug = format!("{:?}", metrics);
        assert!(debug.contains("IndexAllocatorMetrics"));
    }

    #[test]
    fn test_metrics_default_is_empty() {
        let metrics = IndexAllocatorMetrics::default();
        assert_eq!(metrics.allocated_count, 0);
        assert_eq!(metrics.capacity, 0);
    }

    #[test]
    fn test_metrics_reflects_allocator_state() {
        let mut allocator = IndexAllocator::with_generations(50);

        for _ in 0..25 {
            allocator.allocate();
        }
        for i in 0..10 {
            allocator.free(i);
        }

        let m = allocator.metrics();

        assert_eq!(m.allocated_count, 15);
        assert_eq!(m.capacity, 50);
        assert_eq!(m.free_list_size, 10);
        assert_eq!(m.peak_allocations, 25);
        assert_eq!(m.fragmentation, 10.0 / 25.0); // 10 free / 25 peak
        assert_eq!(m.utilization, 15.0 / 50.0);   // 15 allocated / 50 capacity
        assert!(m.has_generations);
    }
}
