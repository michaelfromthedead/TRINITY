// Blackbox contract tests for T-WGPU-P4.4.1 Timestamp Query Pool.
//
// CLEANROOM: No src/ access beyond the public API exported by the crate.
// Tests use only `renderer_backend::query_pool::*` -- no internal fields,
// no private methods, no implementation details.
//
// BUG DETECTED: GPU tests fail due to invalid buffer usage flags in
// TimestampQueryPool::new(). The resolve buffer combines MAP_READ with
// QUERY_RESOLVE, but wgpu requires MAP usage to only be combined with
// the opposite COPY direction (MAP_READ + COPY_DST only, not COPY_SRC).
// FIX: Remove COPY_SRC from resolve buffer, or use staging buffer pattern.
//
// Acceptance criteria (T-WGPU-P4.4.1):
//   1. QuerySetDescriptor with QueryType::Timestamp -- pool creation
//   2. Feature check (TIMESTAMP_QUERY) -- is_supported()
//   3. Capacity management -- capacity(), available(), used()
//   4. Index allocation -- allocate(), deallocate(), pool exhaustion
//   5. Resolve buffer creation -- buffer exists and has correct size
//
// Coverage:
//   1.  Constants have expected values
//   2.  QueryError variants exist and are constructible
//   3.  QueryErrorKind variants exist
//   4.  QueryError::kind() returns correct variant
//   5.  QueryError display/debug traits
//   6.  QueryError is_feature_error() predicate
//   7.  QueryError is_capacity_error() predicate
//   8.  QueryAllocation struct fields accessible
//   9.  QueryPoolStats has required fields
//  10.  QueryPoolBuilder::new() creates default builder
//  11.  QueryPoolBuilder capacity() method
//  12.  QueryPoolBuilder label() method
//  13.  calculate_resolve_buffer_size() utility
//  14.  ticks_to_ms() conversion
//  15.  ticks_to_ns() conversion
//  16.  ms_to_ticks() conversion
//  17.  Conversion round-trip consistency
//  18.  [GPU] TimestampQueryPool::is_supported() returns bool
//  19.  [GPU] TimestampQueryPool::new() creates pool
//  20.  [GPU] Pool capacity() matches requested
//  21.  [GPU] Pool available() starts at capacity
//  22.  [GPU] Pool used() starts at zero
//  23.  [GPU] allocate() returns valid index
//  24.  [GPU] allocate_tracked() returns QueryAllocation
//  25.  [GPU] allocate_range() allocates multiple indices
//  26.  [GPU] Allocation updates available/used counts
//  27.  [GPU] deallocate() restores capacity
//  28.  [GPU] reset() restores full capacity
//  29.  [GPU] Pool exhaustion returns error
//  30.  [GPU] has_capacity() reflects availability
//  31.  [GPU] has_capacity_for() checks range
//  32.  [GPU] is_empty() and is_full() predicates
//  33.  [GPU] resolve_buffer() exists with correct size
//  34.  [GPU] query_set() returns reference
//  35.  [GPU] timestamp_period() returns positive value
//  36.  [GPU] generation() tracks reset cycles
//  37.  [GPU] label() returns configured label
//  38.  [GPU] stats() returns QueryPoolStats
//  39.  [GPU] stats().utilization() reflects usage
//  40.  [GPU] validate_index() checks bounds
//  41.  [GPU] ticks_to_ms() instance method
//  42.  [GPU] delta_to_ms() calculates duration
//  43.  [GPU] QueryPoolBuilder::build() creates pool

use renderer_backend::query_pool::{
    // Constants
    TIMESTAMP_SIZE_BYTES, MIN_POOL_CAPACITY, MAX_RECOMMENDED_CAPACITY, DEFAULT_LABEL_PREFIX,
    // Errors
    QueryError, QueryErrorKind,
    // Types
    QueryAllocation, QueryPoolStats, QueryPoolBuilder, TimestampQueryPool,
    // Resolve params (T-WGPU-P4.4.2)
    QueryResolveParams,
    // Utility functions
    is_timestamp_query_supported, calculate_resolve_buffer_size, ticks_to_ms, ticks_to_ns,
    ms_to_ticks,
};

// =============================================================================
// Category 1: API Contract Tests (No GPU Required)
// =============================================================================

// ---------------------------------------------------------------------------
// Constants Tests
// ---------------------------------------------------------------------------

#[test]
fn test_timestamp_size_bytes_is_eight() {
    // Timestamps in wgpu are u64, which is 8 bytes
    assert_eq!(TIMESTAMP_SIZE_BYTES, 8, "Timestamp size should be 8 bytes (u64)");
}

#[test]
fn test_min_pool_capacity_is_positive() {
    assert!(MIN_POOL_CAPACITY >= 1, "Minimum pool capacity must be at least 1");
}

#[test]
fn test_max_recommended_capacity_is_reasonable() {
    // Should be large enough for practical use but not excessive
    assert!(MAX_RECOMMENDED_CAPACITY >= 64, "Max capacity should be at least 64");
    assert!(MAX_RECOMMENDED_CAPACITY <= 65536, "Max capacity should not exceed 64K");
}

#[test]
fn test_default_label_prefix_is_non_empty() {
    assert!(!DEFAULT_LABEL_PREFIX.is_empty(), "Default label prefix should not be empty");
}

// ---------------------------------------------------------------------------
// QueryError Tests
// ---------------------------------------------------------------------------

#[test]
fn test_query_error_feature_not_supported() {
    let error = QueryError::feature_not_supported();
    assert!(
        error.is_feature_error(),
        "feature_not_supported error should return true for is_feature_error()"
    );
    assert!(
        !error.is_capacity_error(),
        "feature_not_supported error should return false for is_capacity_error()"
    );
    assert_eq!(
        error.kind(),
        QueryErrorKind::FeatureNotSupported,
        "kind() should return FeatureNotSupported"
    );
}

#[test]
fn test_query_error_pool_exhausted() {
    let error = QueryError::pool_exhausted(32);
    assert!(
        error.is_capacity_error(),
        "pool_exhausted error should return true for is_capacity_error()"
    );
    assert!(
        !error.is_feature_error(),
        "pool_exhausted error should return false for is_feature_error()"
    );
    assert_eq!(
        error.kind(),
        QueryErrorKind::PoolExhausted,
        "kind() should return PoolExhausted"
    );
}

#[test]
fn test_query_error_invalid_index() {
    let error = QueryError::invalid_index(100, 32);
    assert_eq!(
        error.kind(),
        QueryErrorKind::InvalidIndex,
        "kind() should return InvalidIndex"
    );
    assert!(
        !error.is_feature_error(),
        "invalid_index error should return false for is_feature_error()"
    );
    assert!(
        !error.is_capacity_error(),
        "invalid_index error should return false for is_capacity_error()"
    );
}

#[test]
fn test_query_error_resolve_failed() {
    let error = QueryError::resolve_failed("Test failure reason");
    assert_eq!(
        error.kind(),
        QueryErrorKind::ResolveFailed,
        "kind() should return ResolveFailed"
    );
}

#[test]
fn test_query_error_invalid_capacity() {
    let error = QueryError::invalid_capacity(0);
    assert_eq!(
        error.kind(),
        QueryErrorKind::InvalidCapacity,
        "kind() should return InvalidCapacity"
    );
}

#[test]
fn test_query_error_buffer_creation_failed() {
    let error = QueryError::buffer_creation_failed("Out of memory");
    assert_eq!(
        error.kind(),
        QueryErrorKind::BufferCreationFailed,
        "kind() should return BufferCreationFailed"
    );
}

#[test]
fn test_query_error_kind_variants_exist() {
    // Ensure all expected variants are accessible
    let _feature = QueryErrorKind::FeatureNotSupported;
    let _exhausted = QueryErrorKind::PoolExhausted;
    let _invalid_idx = QueryErrorKind::InvalidIndex;
    let _resolve = QueryErrorKind::ResolveFailed;
    let _capacity = QueryErrorKind::InvalidCapacity;
    let _buffer = QueryErrorKind::BufferCreationFailed;
}

#[test]
fn test_query_error_implements_debug() {
    let error = QueryError::feature_not_supported();
    let debug_str = format!("{:?}", error);
    assert!(!debug_str.is_empty(), "Debug output should not be empty");
}

#[test]
fn test_query_error_implements_display() {
    let error = QueryError::feature_not_supported();
    let display_str = format!("{}", error);
    assert!(!display_str.is_empty(), "Display output should not be empty");
}

// ---------------------------------------------------------------------------
// QueryPoolBuilder Tests (No GPU)
// ---------------------------------------------------------------------------

#[test]
fn test_query_pool_builder_new() {
    let builder = QueryPoolBuilder::new();
    // Builder should be constructible without panic
    let _ = builder;
}

#[test]
fn test_query_pool_builder_capacity_method() {
    let builder = QueryPoolBuilder::new().capacity(64);
    // Method should be chainable without panic
    let _ = builder;
}

#[test]
fn test_query_pool_builder_label_method() {
    let builder = QueryPoolBuilder::new().label("TestPool");
    // Method should be chainable without panic
    let _ = builder;
}

#[test]
fn test_query_pool_builder_chaining() {
    let builder = QueryPoolBuilder::new()
        .capacity(128)
        .label("ChainedPool");
    // Builder should support method chaining
    let _ = builder;
}

// ---------------------------------------------------------------------------
// Utility Function Tests (No GPU)
// ---------------------------------------------------------------------------

#[test]
fn test_calculate_resolve_buffer_size_formula() {
    // Buffer size should be capacity * TIMESTAMP_SIZE_BYTES
    let capacity = 32;
    let expected = capacity as u64 * TIMESTAMP_SIZE_BYTES;
    let actual = calculate_resolve_buffer_size(capacity);
    assert_eq!(
        actual, expected,
        "Buffer size should be capacity * 8 bytes"
    );
}

#[test]
fn test_calculate_resolve_buffer_size_various_capacities() {
    assert_eq!(calculate_resolve_buffer_size(1), 8);
    assert_eq!(calculate_resolve_buffer_size(10), 80);
    assert_eq!(calculate_resolve_buffer_size(100), 800);
    assert_eq!(calculate_resolve_buffer_size(1000), 8000);
}

#[test]
fn test_ticks_to_ms_conversion() {
    // 1 ms = 1,000,000 ns, so with period_ns = 1.0, 1,000,000 ticks = 1 ms
    let ticks = 1_000_000u64;
    let period_ns = 1.0f32;
    let ms = ticks_to_ms(ticks, period_ns);
    assert!(
        (ms - 1.0).abs() < 0.0001,
        "1,000,000 ticks with 1ns period should be 1ms, got {}",
        ms
    );
}

#[test]
fn test_ticks_to_ms_with_different_period() {
    // With period_ns = 2.0, each tick is 2ns, so 500,000 ticks = 1ms
    let ticks = 500_000u64;
    let period_ns = 2.0f32;
    let ms = ticks_to_ms(ticks, period_ns);
    assert!(
        (ms - 1.0).abs() < 0.0001,
        "500,000 ticks with 2ns period should be 1ms, got {}",
        ms
    );
}

#[test]
fn test_ticks_to_ns_conversion() {
    // With period_ns = 1.0, ticks = ns
    let ticks = 1000u64;
    let period_ns = 1.0f32;
    let ns = ticks_to_ns(ticks, period_ns);
    assert!(
        (ns - 1000.0).abs() < 0.0001,
        "1000 ticks with 1ns period should be 1000ns, got {}",
        ns
    );
}

#[test]
fn test_ticks_to_ns_with_fractional_period() {
    // With period_ns = 0.5, each tick is 0.5ns
    let ticks = 1000u64;
    let period_ns = 0.5f32;
    let ns = ticks_to_ns(ticks, period_ns);
    assert!(
        (ns - 500.0).abs() < 0.0001,
        "1000 ticks with 0.5ns period should be 500ns, got {}",
        ns
    );
}

#[test]
fn test_ms_to_ticks_conversion() {
    // 1 ms = 1,000,000 ns, so with period_ns = 1.0, 1 ms = 1,000,000 ticks
    let ms = 1.0f32;
    let period_ns = 1.0f32;
    let ticks = ms_to_ticks(ms, period_ns);
    assert_eq!(
        ticks, 1_000_000,
        "1ms with 1ns period should be 1,000,000 ticks, got {}",
        ticks
    );
}

#[test]
fn test_ms_to_ticks_with_different_period() {
    // With period_ns = 2.0, 1 ms = 500,000 ticks
    let ms = 1.0f32;
    let period_ns = 2.0f32;
    let ticks = ms_to_ticks(ms, period_ns);
    assert_eq!(
        ticks, 500_000,
        "1ms with 2ns period should be 500,000 ticks, got {}",
        ticks
    );
}

#[test]
fn test_conversion_round_trip_ticks_to_ms_to_ticks() {
    let original_ticks = 5_000_000u64;
    let period_ns = 1.0f32;
    let ms = ticks_to_ms(original_ticks, period_ns);
    let back_to_ticks = ms_to_ticks(ms, period_ns);
    assert_eq!(
        back_to_ticks, original_ticks,
        "Round-trip conversion should preserve value"
    );
}

#[test]
fn test_conversion_zero_ticks() {
    let period_ns = 1.0f32;
    assert!((ticks_to_ms(0, period_ns) - 0.0).abs() < f32::EPSILON);
    assert!((ticks_to_ns(0, period_ns) - 0.0).abs() < f64::EPSILON);
    assert_eq!(ms_to_ticks(0.0, period_ns), 0);
}

// ---------------------------------------------------------------------------
// QueryPoolStats Tests
// ---------------------------------------------------------------------------

#[test]
fn test_query_pool_stats_is_empty_method() {
    // Create stats with zero used
    let stats = QueryPoolStats {
        capacity: 32,
        used: 0,
        available: 32,
        generation: 0,
        timestamp_period_ns: 1.0,
        resolve_buffer_size: 256,
    };
    assert!(stats.is_empty(), "Stats with used=0 should be empty");
}

#[test]
fn test_query_pool_stats_is_full_method() {
    let stats = QueryPoolStats {
        capacity: 32,
        used: 32,
        available: 0,
        generation: 0,
        timestamp_period_ns: 1.0,
        resolve_buffer_size: 256,
    };
    assert!(stats.is_full(), "Stats with used=capacity should be full");
}

#[test]
fn test_query_pool_stats_fields_accessible() {
    let stats = QueryPoolStats {
        capacity: 64,
        used: 10,
        available: 54,
        generation: 3,
        timestamp_period_ns: 1.5,
        resolve_buffer_size: 512,
    };
    assert_eq!(stats.capacity, 64);
    assert_eq!(stats.used, 10);
    assert_eq!(stats.available, 54);
    assert_eq!(stats.generation, 3);
    assert!((stats.timestamp_period_ns - 1.5).abs() < f32::EPSILON);
    assert_eq!(stats.resolve_buffer_size, 512);
}

// =============================================================================
// Category 2: GPU Integration Tests
// =============================================================================

/// Helper to create a wgpu device and queue for testing.
/// Returns None if no suitable adapter is available.
fn create_test_device() -> Option<(wgpu::Device, wgpu::Queue)> {
    pollster::block_on(async {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await?;

        // Request device with timestamp query feature if available
        let features = if adapter.features().contains(wgpu::Features::TIMESTAMP_QUERY) {
            wgpu::Features::TIMESTAMP_QUERY
        } else {
            wgpu::Features::empty()
        };

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("QueryPoolTestDevice"),
                    required_features: features,
                    required_limits: wgpu::Limits::default(),
                    memory_hints: wgpu::MemoryHints::default(),
                },
                None,
            )
            .await
            .ok()?;

        Some((device, queue))
    })
}

#[test]
fn test_is_timestamp_query_supported_returns_bool() {
    if let Some((device, _queue)) = create_test_device() {
        // Function should return a boolean without panicking
        let _supported = is_timestamp_query_supported(&device);
    }
}

#[test]
fn test_timestamp_query_pool_is_supported_static() {
    if let Some((device, _queue)) = create_test_device() {
        // Static method should return a boolean
        let _supported = TimestampQueryPool::is_supported(&device);
    }
}

#[test]
fn test_timestamp_query_pool_new_creates_pool() {
    let Some((device, queue)) = create_test_device() else {
        eprintln!("Skipping GPU test: no adapter available");
        return;
    };

    if !TimestampQueryPool::is_supported(&device) {
        eprintln!("Skipping GPU test: TIMESTAMP_QUERY not supported");
        return;
    }

    let result = TimestampQueryPool::new(&device, &queue, 32);
    assert!(result.is_ok(), "Pool creation should succeed: {:?}", result.err());
}

#[test]
fn test_timestamp_query_pool_capacity_matches_requested() {
    let Some((device, queue)) = create_test_device() else {
        return;
    };

    if !TimestampQueryPool::is_supported(&device) {
        return;
    }

    let pool = TimestampQueryPool::new(&device, &queue, 64).unwrap();
    assert_eq!(pool.capacity(), 64, "Capacity should match requested value");
}

#[test]
fn test_timestamp_query_pool_available_starts_at_capacity() {
    let Some((device, queue)) = create_test_device() else {
        return;
    };

    if !TimestampQueryPool::is_supported(&device) {
        return;
    }

    let pool = TimestampQueryPool::new(&device, &queue, 32).unwrap();
    assert_eq!(
        pool.available(),
        pool.capacity(),
        "Available should equal capacity when pool is empty"
    );
}

#[test]
fn test_timestamp_query_pool_used_starts_at_zero() {
    let Some((device, queue)) = create_test_device() else {
        return;
    };

    if !TimestampQueryPool::is_supported(&device) {
        return;
    }

    let pool = TimestampQueryPool::new(&device, &queue, 32).unwrap();
    assert_eq!(pool.used(), 0, "Used should be zero when pool is empty");
}

#[test]
fn test_timestamp_query_pool_allocate_returns_valid_index() {
    let Some((device, queue)) = create_test_device() else {
        return;
    };

    if !TimestampQueryPool::is_supported(&device) {
        return;
    }

    let mut pool = TimestampQueryPool::new(&device, &queue, 32).unwrap();
    let result = pool.allocate();
    assert!(result.is_ok(), "Allocation should succeed");
    let index = result.unwrap();
    assert!(index < pool.capacity(), "Index should be within capacity");
}

#[test]
fn test_timestamp_query_pool_allocate_tracked_returns_allocation() {
    let Some((device, queue)) = create_test_device() else {
        return;
    };

    if !TimestampQueryPool::is_supported(&device) {
        return;
    }

    let mut pool = TimestampQueryPool::new(&device, &queue, 32).unwrap();
    let result = pool.allocate_tracked();
    assert!(result.is_ok(), "Tracked allocation should succeed");
    let allocation = result.unwrap();
    assert!(allocation.index < pool.capacity(), "Allocation index should be valid");
}

#[test]
fn test_timestamp_query_pool_allocate_range() {
    let Some((device, queue)) = create_test_device() else {
        return;
    };

    if !TimestampQueryPool::is_supported(&device) {
        return;
    }

    let mut pool = TimestampQueryPool::new(&device, &queue, 32).unwrap();
    let result = pool.allocate_range(4);
    assert!(result.is_ok(), "Range allocation should succeed");
    let start_index = result.unwrap();
    assert!(start_index + 4 <= pool.capacity(), "Range should fit within capacity");
}

#[test]
fn test_timestamp_query_pool_allocation_updates_counts() {
    let Some((device, queue)) = create_test_device() else {
        return;
    };

    if !TimestampQueryPool::is_supported(&device) {
        return;
    }

    let mut pool = TimestampQueryPool::new(&device, &queue, 32).unwrap();
    let initial_available = pool.available();
    let initial_used = pool.used();

    pool.allocate().unwrap();

    assert_eq!(
        pool.available(),
        initial_available - 1,
        "Available should decrease by 1"
    );
    assert_eq!(pool.used(), initial_used + 1, "Used should increase by 1");
}

#[test]
fn test_timestamp_query_pool_deallocate_restores_capacity() {
    let Some((device, queue)) = create_test_device() else {
        return;
    };

    if !TimestampQueryPool::is_supported(&device) {
        return;
    }

    let mut pool = TimestampQueryPool::new(&device, &queue, 32).unwrap();
    let index = pool.allocate().unwrap();
    let used_after_alloc = pool.used();

    pool.deallocate(index);

    assert!(
        pool.used() < used_after_alloc || pool.available() > pool.capacity() - used_after_alloc,
        "Deallocate should restore capacity or mark index as reusable"
    );
}

#[test]
fn test_timestamp_query_pool_reset_restores_full_capacity() {
    let Some((device, queue)) = create_test_device() else {
        return;
    };

    if !TimestampQueryPool::is_supported(&device) {
        return;
    }

    let mut pool = TimestampQueryPool::new(&device, &queue, 32).unwrap();

    // Allocate some indices
    for _ in 0..10 {
        pool.allocate().unwrap();
    }

    assert!(pool.used() > 0, "Pool should have allocations");

    pool.reset();

    assert_eq!(pool.used(), 0, "Reset should clear all allocations");
    assert_eq!(
        pool.available(),
        pool.capacity(),
        "Reset should restore full capacity"
    );
}

#[test]
fn test_timestamp_query_pool_exhaustion_returns_error() {
    let Some((device, queue)) = create_test_device() else {
        return;
    };

    if !TimestampQueryPool::is_supported(&device) {
        return;
    }

    let capacity = 4u32; // Small capacity for quick exhaustion
    let mut pool = TimestampQueryPool::new(&device, &queue, capacity).unwrap();

    // Allocate all indices
    for _ in 0..capacity {
        assert!(pool.allocate().is_ok(), "Allocation should succeed");
    }

    // Next allocation should fail
    let result = pool.allocate();
    assert!(result.is_err(), "Allocation should fail when pool is exhausted");

    let error = result.unwrap_err();
    assert_eq!(
        error.kind(),
        QueryErrorKind::PoolExhausted,
        "Error should be PoolExhausted"
    );
}

#[test]
fn test_timestamp_query_pool_has_capacity() {
    let Some((device, queue)) = create_test_device() else {
        return;
    };

    if !TimestampQueryPool::is_supported(&device) {
        return;
    }

    let mut pool = TimestampQueryPool::new(&device, &queue, 2).unwrap();

    assert!(pool.has_capacity(), "New pool should have capacity");

    pool.allocate().unwrap();
    assert!(pool.has_capacity(), "Pool with 1/2 used should have capacity");

    pool.allocate().unwrap();
    assert!(!pool.has_capacity(), "Full pool should not have capacity");
}

#[test]
fn test_timestamp_query_pool_has_capacity_for() {
    let Some((device, queue)) = create_test_device() else {
        return;
    };

    if !TimestampQueryPool::is_supported(&device) {
        return;
    }

    let pool = TimestampQueryPool::new(&device, &queue, 8).unwrap();

    assert!(pool.has_capacity_for(1), "Should have capacity for 1");
    assert!(pool.has_capacity_for(8), "Should have capacity for 8");
    assert!(!pool.has_capacity_for(9), "Should not have capacity for 9");
}

#[test]
fn test_timestamp_query_pool_is_empty_and_is_full() {
    let Some((device, queue)) = create_test_device() else {
        return;
    };

    if !TimestampQueryPool::is_supported(&device) {
        return;
    }

    let mut pool = TimestampQueryPool::new(&device, &queue, 2).unwrap();

    assert!(pool.is_empty(), "New pool should be empty");
    assert!(!pool.is_full(), "New pool should not be full");

    pool.allocate().unwrap();
    assert!(!pool.is_empty(), "Pool with allocations should not be empty");
    assert!(!pool.is_full(), "Pool with 1/2 used should not be full");

    pool.allocate().unwrap();
    assert!(!pool.is_empty(), "Full pool should not be empty");
    assert!(pool.is_full(), "Pool with all indices used should be full");
}

#[test]
fn test_timestamp_query_pool_resolve_buffer_exists() {
    let Some((device, queue)) = create_test_device() else {
        return;
    };

    if !TimestampQueryPool::is_supported(&device) {
        return;
    }

    let pool = TimestampQueryPool::new(&device, &queue, 32).unwrap();
    let _buffer = pool.resolve_buffer(); // Should not panic
}

#[test]
fn test_timestamp_query_pool_resolve_buffer_size() {
    let Some((device, queue)) = create_test_device() else {
        return;
    };

    if !TimestampQueryPool::is_supported(&device) {
        return;
    }

    let capacity = 32u32;
    let pool = TimestampQueryPool::new(&device, &queue, capacity).unwrap();
    let expected_size = calculate_resolve_buffer_size(capacity);

    assert_eq!(
        pool.resolve_buffer_size(),
        expected_size,
        "Resolve buffer size should match calculated size"
    );
}

#[test]
fn test_timestamp_query_pool_query_set_exists() {
    let Some((device, queue)) = create_test_device() else {
        return;
    };

    if !TimestampQueryPool::is_supported(&device) {
        return;
    }

    let pool = TimestampQueryPool::new(&device, &queue, 32).unwrap();
    let _query_set = pool.query_set(); // Should not panic
}

#[test]
fn test_timestamp_query_pool_timestamp_period_positive() {
    let Some((device, queue)) = create_test_device() else {
        return;
    };

    if !TimestampQueryPool::is_supported(&device) {
        return;
    }

    let pool = TimestampQueryPool::new(&device, &queue, 32).unwrap();
    let period = pool.timestamp_period();

    assert!(
        period > 0.0,
        "Timestamp period should be positive, got {}",
        period
    );
}

#[test]
fn test_timestamp_query_pool_generation_tracks_resets() {
    let Some((device, queue)) = create_test_device() else {
        return;
    };

    if !TimestampQueryPool::is_supported(&device) {
        return;
    }

    let mut pool = TimestampQueryPool::new(&device, &queue, 8).unwrap();
    let initial_gen = pool.generation();

    pool.reset();
    let after_reset = pool.generation();

    assert_eq!(
        after_reset,
        initial_gen + 1,
        "Generation should increment on reset"
    );
}

#[test]
fn test_timestamp_query_pool_with_label() {
    let Some((device, queue)) = create_test_device() else {
        return;
    };

    if !TimestampQueryPool::is_supported(&device) {
        return;
    }

    let label = "MyTestPool";
    let pool = TimestampQueryPool::with_label(&device, &queue, 32, Some(label)).unwrap();

    let pool_label = pool.label();
    assert!(pool_label.is_some(), "Pool should have a label");
    assert!(
        pool_label.unwrap().contains(label),
        "Label should contain the specified name"
    );
}

#[test]
fn test_timestamp_query_pool_stats() {
    let Some((device, queue)) = create_test_device() else {
        return;
    };

    if !TimestampQueryPool::is_supported(&device) {
        return;
    }

    let mut pool = TimestampQueryPool::new(&device, &queue, 16).unwrap();
    pool.allocate().unwrap();
    pool.allocate().unwrap();

    let stats = pool.stats();

    assert_eq!(stats.capacity, 16, "Stats capacity should match pool");
    assert_eq!(stats.used, 2, "Stats used should reflect allocations");
    assert_eq!(stats.available, 14, "Stats available should reflect remaining");
}

#[test]
fn test_timestamp_query_pool_utilization() {
    let Some((device, queue)) = create_test_device() else {
        return;
    };

    if !TimestampQueryPool::is_supported(&device) {
        return;
    }

    let mut pool = TimestampQueryPool::new(&device, &queue, 4).unwrap();

    assert!(
        (pool.stats().utilization() - 0.0).abs() < 0.01,
        "Empty pool should have 0% utilization"
    );

    pool.allocate().unwrap();
    pool.allocate().unwrap();

    assert!(
        (pool.stats().utilization() - 0.5).abs() < 0.01,
        "Pool with 2/4 used should have 50% utilization"
    );
}

#[test]
fn test_timestamp_query_pool_validate_index() {
    let Some((device, queue)) = create_test_device() else {
        return;
    };

    if !TimestampQueryPool::is_supported(&device) {
        return;
    }

    let pool = TimestampQueryPool::new(&device, &queue, 8).unwrap();

    assert!(pool.validate_index(0).is_ok(), "Index 0 should be valid");
    assert!(pool.validate_index(7).is_ok(), "Index 7 should be valid");
    assert!(pool.validate_index(8).is_err(), "Index 8 should be invalid");
    assert!(pool.validate_index(100).is_err(), "Index 100 should be invalid");

    let error = pool.validate_index(100).unwrap_err();
    assert_eq!(
        error.kind(),
        QueryErrorKind::InvalidIndex,
        "Error should be InvalidIndex"
    );
}

#[test]
fn test_timestamp_query_pool_ticks_to_ms_instance() {
    let Some((device, queue)) = create_test_device() else {
        return;
    };

    if !TimestampQueryPool::is_supported(&device) {
        return;
    }

    let pool = TimestampQueryPool::new(&device, &queue, 8).unwrap();
    let period = pool.timestamp_period();

    // Calculate expected value using the standalone function
    let ticks = 1_000_000u64;
    let expected_ms = ticks_to_ms(ticks, period);
    let actual_ms = pool.ticks_to_ms(ticks);

    assert!(
        (actual_ms - expected_ms).abs() < 0.0001,
        "Instance method should match standalone function"
    );
}

#[test]
fn test_timestamp_query_pool_delta_to_ms() {
    let Some((device, queue)) = create_test_device() else {
        return;
    };

    if !TimestampQueryPool::is_supported(&device) {
        return;
    }

    let pool = TimestampQueryPool::new(&device, &queue, 8).unwrap();

    let begin = 1_000_000u64;
    let end = 2_000_000u64;
    let delta_ms = pool.delta_to_ms(begin, end);

    // Delta should be positive and represent end - begin
    assert!(delta_ms >= 0.0, "Delta should be non-negative");
}

#[test]
fn test_query_pool_builder_build() {
    let Some((device, queue)) = create_test_device() else {
        return;
    };

    if !TimestampQueryPool::is_supported(&device) {
        return;
    }

    let result = QueryPoolBuilder::new()
        .capacity(16)
        .label("BuilderPool")
        .build(&device, &queue);

    assert!(result.is_ok(), "Builder should successfully create pool");

    let pool = result.unwrap();
    assert_eq!(pool.capacity(), 16, "Pool should have configured capacity");
}

#[test]
fn test_query_pool_builder_default_capacity() {
    let Some((device, queue)) = create_test_device() else {
        return;
    };

    if !TimestampQueryPool::is_supported(&device) {
        return;
    }

    // Build without setting capacity explicitly
    let result = QueryPoolBuilder::new()
        .label("DefaultCapacityPool")
        .build(&device, &queue);

    assert!(result.is_ok(), "Builder should use default capacity");
}

#[test]
fn test_timestamp_query_pool_unsupported_device() {
    let Some((device, queue)) = create_test_device() else {
        return;
    };

    // If the device doesn't support timestamps, creation should fail gracefully
    if !TimestampQueryPool::is_supported(&device) {
        let result = TimestampQueryPool::new(&device, &queue, 32);
        assert!(
            result.is_err(),
            "Pool creation should fail on unsupported device"
        );
        if let Err(e) = result {
            assert!(
                e.is_feature_error(),
                "Error should be a feature error"
            );
        }
    }
}

#[test]
fn test_query_allocation_struct_fields() {
    // Test that QueryAllocation has expected fields
    let allocation = QueryAllocation {
        index: 5,
        generation: 2,
    };
    assert_eq!(allocation.index, 5);
    assert_eq!(allocation.generation, 2);
}

#[test]
fn test_allocation_reuse_after_reset() {
    let Some((device, queue)) = create_test_device() else {
        return;
    };

    if !TimestampQueryPool::is_supported(&device) {
        return;
    }

    let mut pool = TimestampQueryPool::new(&device, &queue, 4).unwrap();

    // Allocate all
    for _ in 0..4 {
        pool.allocate().unwrap();
    }
    assert!(pool.is_full(), "Pool should be full");

    // Reset
    pool.reset();
    assert!(pool.is_empty(), "Pool should be empty after reset");

    // Should be able to allocate again
    let result = pool.allocate();
    assert!(result.is_ok(), "Should be able to allocate after reset");
}

#[test]
fn test_multiple_pools_independent() {
    let Some((device, queue)) = create_test_device() else {
        return;
    };

    if !TimestampQueryPool::is_supported(&device) {
        return;
    }

    let mut pool1 = TimestampQueryPool::new(&device, &queue, 8).unwrap();
    let mut pool2 = TimestampQueryPool::new(&device, &queue, 16).unwrap();

    pool1.allocate().unwrap();
    pool1.allocate().unwrap();

    // Pool2 should be unaffected
    assert_eq!(pool2.used(), 0, "Pool2 should be independent");
    assert_eq!(pool2.capacity(), 16, "Pool2 capacity should be unchanged");

    pool2.allocate().unwrap();
    assert_eq!(pool1.used(), 2, "Pool1 should be unaffected by pool2");
}

// =============================================================================
// Category 3: Query Resolve Tests (T-WGPU-P4.4.2)
// =============================================================================
//
// CLEANROOM: Tests exercise only the public API for query resolution.
//
// Acceptance criteria (T-WGPU-P4.4.2):
//   1. resolve_query_set() command -- resolve() method exists and works
//   2. Range specification -- test start_query and query_count params
//   3. Destination buffer offset -- test destination_offset validation
//   4. Timing: after passes, before submit -- test resolve ordering
//
// Coverage:
//   1.  QueryResolveParams::new() construction
//   2.  QueryResolveParams::from_start() construction
//   3.  QueryResolveParams field access (start_query)
//   4.  QueryResolveParams field access (query_count)
//   5.  QueryResolveParams field access (destination_offset)
//   6.  QueryResolveParams::end_query() calculation
//   7.  QueryResolveParams::required_buffer_size() calculation
//   8.  QueryResolveParams with zero start_query
//   9.  QueryResolveParams with non-zero start_query
//  10.  QueryResolveParams with large destination_offset
//  11.  [GPU] validate_resolve_params() with valid params
//  12.  [GPU] validate_resolve_params() rejects zero query_count
//  13.  [GPU] validate_resolve_params() rejects out-of-bounds range
//  14.  [GPU] validate_resolve_params() rejects buffer overflow
//  15.  [GPU] resolve() with valid params succeeds
//  16.  [GPU] resolve() rejects zero query_count
//  17.  [GPU] resolve() rejects out-of-bounds start_query
//  18.  [GPU] resolve() rejects out-of-bounds end_query
//  19.  [GPU] resolve_all() resolves entire pool
//  20.  [GPU] resolve_all() on empty pool (used=0)
//  21.  [GPU] resolve_first() with count=1
//  22.  [GPU] resolve_first() with count > capacity fails
//  23.  [GPU] resolve_first() with count=0 fails
//  24.  [GPU] resolve ordering: multiple resolves in sequence
//  25.  [GPU] destination_offset alignment
//  26.  [GPU] destination_offset overflow detection

// ---------------------------------------------------------------------------
// QueryResolveParams Tests (No GPU Required)
// ---------------------------------------------------------------------------

#[test]
fn test_resolve_params_new_construction() {
    let params = QueryResolveParams::new(0, 4, 0);
    assert_eq!(params.start_query, 0, "start_query should be 0");
    assert_eq!(params.query_count, 4, "query_count should be 4");
    assert_eq!(params.destination_offset, 0, "destination_offset should be 0");
}

#[test]
fn test_resolve_params_new_with_nonzero_values() {
    let params = QueryResolveParams::new(5, 10, 128);
    assert_eq!(params.start_query, 5, "start_query should be 5");
    assert_eq!(params.query_count, 10, "query_count should be 10");
    assert_eq!(params.destination_offset, 128, "destination_offset should be 128");
}

#[test]
fn test_resolve_params_from_start_construction() {
    let params = QueryResolveParams::from_start(8);
    assert_eq!(params.start_query, 0, "from_start should set start_query to 0");
    assert_eq!(params.query_count, 8, "query_count should be 8");
    assert_eq!(params.destination_offset, 0, "from_start should set destination_offset to 0");
}

#[test]
fn test_resolve_params_start_query_field() {
    let params = QueryResolveParams::new(10, 5, 0);
    assert_eq!(params.start_query, 10, "start_query field should be accessible");
}

#[test]
fn test_resolve_params_query_count_field() {
    let params = QueryResolveParams::new(0, 16, 0);
    assert_eq!(params.query_count, 16, "query_count field should be accessible");
}

#[test]
fn test_resolve_params_destination_offset_field() {
    let params = QueryResolveParams::new(0, 4, 256);
    assert_eq!(params.destination_offset, 256, "destination_offset field should be accessible");
}

#[test]
fn test_resolve_params_end_query_calculation() {
    let params = QueryResolveParams::new(5, 10, 0);
    assert_eq!(
        params.end_query(),
        15,
        "end_query should be start_query + query_count"
    );
}

#[test]
fn test_resolve_params_end_query_from_zero() {
    let params = QueryResolveParams::new(0, 8, 0);
    assert_eq!(params.end_query(), 8, "end_query from 0 should equal query_count");
}

#[test]
fn test_resolve_params_required_buffer_size() {
    let params = QueryResolveParams::new(0, 4, 0);
    // Each timestamp is 8 bytes (u64), so 4 queries need 32 bytes
    assert_eq!(
        params.required_buffer_size(),
        32,
        "4 queries should require 32 bytes (4 * 8)"
    );
}

#[test]
fn test_resolve_params_required_buffer_size_with_offset() {
    let params = QueryResolveParams::new(0, 4, 64);
    // Required size should account for offset + data: 64 + (4 * 8) = 96
    assert_eq!(
        params.required_buffer_size(),
        96,
        "Buffer size should include destination_offset"
    );
}

#[test]
fn test_resolve_params_required_buffer_size_single_query() {
    let params = QueryResolveParams::new(0, 1, 0);
    assert_eq!(
        params.required_buffer_size(),
        TIMESTAMP_SIZE_BYTES,
        "Single query should require TIMESTAMP_SIZE_BYTES"
    );
}

#[test]
fn test_resolve_params_with_zero_start() {
    let params = QueryResolveParams::new(0, 100, 0);
    assert_eq!(params.start_query, 0);
    assert_eq!(params.end_query(), 100);
}

#[test]
fn test_resolve_params_with_large_offset() {
    let large_offset = 1024 * 1024; // 1 MB
    let params = QueryResolveParams::new(0, 4, large_offset);
    assert_eq!(params.destination_offset, large_offset);
    assert_eq!(
        params.required_buffer_size(),
        large_offset + 32,
        "Large offset should be included in required size"
    );
}

#[test]
fn test_resolve_params_max_values() {
    // Test with large but valid values
    let params = QueryResolveParams::new(1000, 1000, 8192);
    assert_eq!(params.start_query, 1000);
    assert_eq!(params.query_count, 1000);
    assert_eq!(params.destination_offset, 8192);
    assert_eq!(params.end_query(), 2000);
}

// ---------------------------------------------------------------------------
// GPU Resolve Tests (T-WGPU-P4.4.2)
// ---------------------------------------------------------------------------

#[test]
fn test_validate_resolve_params_valid() {
    let Some((device, queue)) = create_test_device() else {
        return;
    };

    if !TimestampQueryPool::is_supported(&device) {
        eprintln!("Skipping GPU test: TIMESTAMP_QUERY not supported");
        return;
    }

    let pool = TimestampQueryPool::new(&device, &queue, 32).unwrap();
    let params = QueryResolveParams::new(0, 8, 0);

    let result = pool.validate_resolve_params(&params);
    assert!(result.is_ok(), "Valid params should pass validation: {:?}", result.err());
}

#[test]
fn test_validate_resolve_params_full_capacity() {
    let Some((device, queue)) = create_test_device() else {
        return;
    };

    if !TimestampQueryPool::is_supported(&device) {
        return;
    }

    let capacity = 16u32;
    let pool = TimestampQueryPool::new(&device, &queue, capacity).unwrap();
    let params = QueryResolveParams::new(0, capacity, 0);

    let result = pool.validate_resolve_params(&params);
    assert!(result.is_ok(), "Resolving full capacity should be valid");
}

#[test]
fn test_validate_resolve_params_rejects_zero_count() {
    let Some((device, queue)) = create_test_device() else {
        return;
    };

    if !TimestampQueryPool::is_supported(&device) {
        return;
    }

    let pool = TimestampQueryPool::new(&device, &queue, 32).unwrap();
    let params = QueryResolveParams::new(0, 0, 0); // Zero query_count

    let result = pool.validate_resolve_params(&params);
    assert!(result.is_err(), "Zero query_count should be rejected");
}

#[test]
fn test_validate_resolve_params_rejects_out_of_bounds_start() {
    let Some((device, queue)) = create_test_device() else {
        return;
    };

    if !TimestampQueryPool::is_supported(&device) {
        return;
    }

    let pool = TimestampQueryPool::new(&device, &queue, 8).unwrap();
    let params = QueryResolveParams::new(10, 2, 0); // start_query beyond capacity

    let result = pool.validate_resolve_params(&params);
    assert!(result.is_err(), "Out-of-bounds start_query should be rejected");
}

#[test]
fn test_validate_resolve_params_rejects_out_of_bounds_range() {
    let Some((device, queue)) = create_test_device() else {
        return;
    };

    if !TimestampQueryPool::is_supported(&device) {
        return;
    }

    let pool = TimestampQueryPool::new(&device, &queue, 8).unwrap();
    let params = QueryResolveParams::new(5, 10, 0); // end_query = 15 > capacity 8

    let result = pool.validate_resolve_params(&params);
    assert!(result.is_err(), "Range exceeding capacity should be rejected");
}

#[test]
fn test_validate_resolve_params_rejects_buffer_overflow() {
    let Some((device, queue)) = create_test_device() else {
        return;
    };

    if !TimestampQueryPool::is_supported(&device) {
        return;
    }

    let pool = TimestampQueryPool::new(&device, &queue, 8).unwrap();
    // Resolve buffer size = 8 * 8 = 64 bytes
    // Params request offset 56 + 16 bytes (2 queries) = 72 bytes > 64
    let params = QueryResolveParams::new(0, 2, 56);

    let result = pool.validate_resolve_params(&params);
    assert!(result.is_err(), "Buffer overflow should be rejected");
}

#[test]
fn test_resolve_with_valid_params() {
    let Some((device, queue)) = create_test_device() else {
        return;
    };

    if !TimestampQueryPool::is_supported(&device) {
        return;
    }

    let pool = TimestampQueryPool::new(&device, &queue, 16).unwrap();
    let params = QueryResolveParams::new(0, 4, 0);

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("ResolveTestEncoder"),
    });

    let result = pool.resolve(&mut encoder, params);
    assert!(result.is_ok(), "resolve() should succeed with valid params: {:?}", result.err());

    // Submit to ensure no GPU errors
    queue.submit(std::iter::once(encoder.finish()));
}

#[test]
fn test_resolve_rejects_zero_query_count() {
    let Some((device, queue)) = create_test_device() else {
        return;
    };

    if !TimestampQueryPool::is_supported(&device) {
        return;
    }

    let pool = TimestampQueryPool::new(&device, &queue, 16).unwrap();
    let params = QueryResolveParams::new(0, 0, 0);

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("ResolveZeroCountEncoder"),
    });

    let result = pool.resolve(&mut encoder, params);
    assert!(result.is_err(), "resolve() should reject zero query_count");

    if let Err(e) = result {
        assert_eq!(
            e.kind(),
            QueryErrorKind::ResolveFailed,
            "Error kind should be ResolveFailed"
        );
    }
}

#[test]
fn test_resolve_rejects_out_of_bounds_start() {
    let Some((device, queue)) = create_test_device() else {
        return;
    };

    if !TimestampQueryPool::is_supported(&device) {
        return;
    }

    let pool = TimestampQueryPool::new(&device, &queue, 8).unwrap();
    let params = QueryResolveParams::new(100, 2, 0); // start way out of bounds

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("ResolveOOBStartEncoder"),
    });

    let result = pool.resolve(&mut encoder, params);
    assert!(result.is_err(), "resolve() should reject out-of-bounds start");
}

#[test]
fn test_resolve_rejects_out_of_bounds_end() {
    let Some((device, queue)) = create_test_device() else {
        return;
    };

    if !TimestampQueryPool::is_supported(&device) {
        return;
    }

    let pool = TimestampQueryPool::new(&device, &queue, 8).unwrap();
    let params = QueryResolveParams::new(6, 5, 0); // end_query = 11 > 8

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("ResolveOOBEndEncoder"),
    });

    let result = pool.resolve(&mut encoder, params);
    assert!(result.is_err(), "resolve() should reject range exceeding capacity");
}

#[test]
fn test_resolve_all_entire_pool() {
    let Some((device, queue)) = create_test_device() else {
        return;
    };

    if !TimestampQueryPool::is_supported(&device) {
        return;
    }

    let pool = TimestampQueryPool::new(&device, &queue, 8).unwrap();

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("ResolveAllEncoder"),
    });

    let result = pool.resolve_all(&mut encoder);
    assert!(result.is_ok(), "resolve_all() should succeed: {:?}", result.err());

    queue.submit(std::iter::once(encoder.finish()));
}

#[test]
fn test_resolve_all_on_empty_pool() {
    let Some((device, queue)) = create_test_device() else {
        return;
    };

    if !TimestampQueryPool::is_supported(&device) {
        return;
    }

    // Pool with no allocations (used = 0)
    let pool = TimestampQueryPool::new(&device, &queue, 4).unwrap();
    assert!(pool.is_empty(), "Pool should be empty");

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("ResolveAllEmptyEncoder"),
    });

    // resolve_all should work regardless of allocation state
    let result = pool.resolve_all(&mut encoder);
    assert!(result.is_ok(), "resolve_all() should succeed even on empty pool");
}

#[test]
fn test_resolve_first_single_query() {
    let Some((device, queue)) = create_test_device() else {
        return;
    };

    if !TimestampQueryPool::is_supported(&device) {
        return;
    }

    let pool = TimestampQueryPool::new(&device, &queue, 16).unwrap();

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("ResolveFirstSingleEncoder"),
    });

    let result = pool.resolve_first(&mut encoder, 1);
    assert!(result.is_ok(), "resolve_first(1) should succeed: {:?}", result.err());

    queue.submit(std::iter::once(encoder.finish()));
}

#[test]
fn test_resolve_first_multiple_queries() {
    let Some((device, queue)) = create_test_device() else {
        return;
    };

    if !TimestampQueryPool::is_supported(&device) {
        return;
    }

    let pool = TimestampQueryPool::new(&device, &queue, 16).unwrap();

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("ResolveFirstMultiEncoder"),
    });

    let result = pool.resolve_first(&mut encoder, 8);
    assert!(result.is_ok(), "resolve_first(8) should succeed: {:?}", result.err());

    queue.submit(std::iter::once(encoder.finish()));
}

#[test]
fn test_resolve_first_exceeds_capacity_fails() {
    let Some((device, queue)) = create_test_device() else {
        return;
    };

    if !TimestampQueryPool::is_supported(&device) {
        return;
    }

    let pool = TimestampQueryPool::new(&device, &queue, 4).unwrap();

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("ResolveFirstOOBEncoder"),
    });

    let result = pool.resolve_first(&mut encoder, 10); // 10 > capacity 4
    assert!(result.is_err(), "resolve_first with count > capacity should fail");
}

#[test]
fn test_resolve_first_zero_count_fails() {
    let Some((device, queue)) = create_test_device() else {
        return;
    };

    if !TimestampQueryPool::is_supported(&device) {
        return;
    }

    let pool = TimestampQueryPool::new(&device, &queue, 8).unwrap();

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("ResolveFirstZeroEncoder"),
    });

    let result = pool.resolve_first(&mut encoder, 0);
    assert!(result.is_err(), "resolve_first(0) should fail");
}

#[test]
fn test_resolve_multiple_times_sequential() {
    let Some((device, queue)) = create_test_device() else {
        return;
    };

    if !TimestampQueryPool::is_supported(&device) {
        return;
    }

    let pool = TimestampQueryPool::new(&device, &queue, 16).unwrap();

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("ResolveSequentialEncoder"),
    });

    // Resolve first 4 queries
    let result1 = pool.resolve(&mut encoder, QueryResolveParams::new(0, 4, 0));
    assert!(result1.is_ok(), "First resolve should succeed");

    // Resolve next 4 queries to different offset
    let result2 = pool.resolve(&mut encoder, QueryResolveParams::new(4, 4, 32));
    assert!(result2.is_ok(), "Second resolve should succeed");

    queue.submit(std::iter::once(encoder.finish()));
}

#[test]
fn test_resolve_with_offset_alignment() {
    let Some((device, queue)) = create_test_device() else {
        return;
    };

    if !TimestampQueryPool::is_supported(&device) {
        return;
    }

    let pool = TimestampQueryPool::new(&device, &queue, 16).unwrap();

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("ResolveOffsetAlignEncoder"),
    });

    // Offset 8 is aligned to timestamp size (8 bytes)
    let params = QueryResolveParams::new(0, 2, 8);
    let result = pool.resolve(&mut encoder, params);
    assert!(result.is_ok(), "Aligned offset should succeed: {:?}", result.err());

    queue.submit(std::iter::once(encoder.finish()));
}

#[test]
fn test_resolve_destination_offset_at_boundary() {
    let Some((device, queue)) = create_test_device() else {
        return;
    };

    if !TimestampQueryPool::is_supported(&device) {
        return;
    }

    let capacity = 8u32;
    let pool = TimestampQueryPool::new(&device, &queue, capacity).unwrap();
    let buffer_size = pool.resolve_buffer_size();

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("ResolveBoundaryEncoder"),
    });

    // Resolve single query at maximum valid offset
    // Buffer size = 64 bytes, single query needs 8 bytes
    // Max offset = 64 - 8 = 56
    let max_offset = buffer_size - TIMESTAMP_SIZE_BYTES;
    let params = QueryResolveParams::new(0, 1, max_offset);
    let result = pool.resolve(&mut encoder, params);
    assert!(result.is_ok(), "Resolve at max valid offset should succeed: {:?}", result.err());
}

#[test]
fn test_resolve_destination_offset_overflow() {
    let Some((device, queue)) = create_test_device() else {
        return;
    };

    if !TimestampQueryPool::is_supported(&device) {
        return;
    }

    let pool = TimestampQueryPool::new(&device, &queue, 4).unwrap();
    let buffer_size = pool.resolve_buffer_size(); // 32 bytes

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("ResolveOverflowEncoder"),
    });

    // Try to resolve with offset that would overflow buffer
    // 2 queries = 16 bytes, offset 24 would need 40 bytes total > 32
    let params = QueryResolveParams::new(0, 2, 24);
    let result = pool.resolve(&mut encoder, params);
    assert!(result.is_err(), "Resolve with buffer overflow should fail");
}

#[test]
fn test_resolve_params_edge_case_single_at_end() {
    let Some((device, queue)) = create_test_device() else {
        return;
    };

    if !TimestampQueryPool::is_supported(&device) {
        return;
    }

    let pool = TimestampQueryPool::new(&device, &queue, 8).unwrap();

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("ResolveEdgeCaseEncoder"),
    });

    // Resolve just the last query
    let params = QueryResolveParams::new(7, 1, 0);
    let result = pool.resolve(&mut encoder, params);
    assert!(result.is_ok(), "Resolving last single query should succeed: {:?}", result.err());

    queue.submit(std::iter::once(encoder.finish()));
}

#[test]
fn test_resolve_params_range_middle_section() {
    let Some((device, queue)) = create_test_device() else {
        return;
    };

    if !TimestampQueryPool::is_supported(&device) {
        return;
    }

    let pool = TimestampQueryPool::new(&device, &queue, 16).unwrap();

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("ResolveMiddleEncoder"),
    });

    // Resolve queries 4-7 (middle section)
    let params = QueryResolveParams::new(4, 4, 0);
    let result = pool.resolve(&mut encoder, params);
    assert!(result.is_ok(), "Resolving middle section should succeed: {:?}", result.err());

    queue.submit(std::iter::once(encoder.finish()));
}

// =============================================================================
// Category 4: Async Query Readback Tests (T-WGPU-P4.4.3)
// =============================================================================
//
// CLEANROOM: Tests exercise only the public API for async timestamp readback.
//
// Acceptance criteria (T-WGPU-P4.4.3):
//   1. Map readback buffer async -- async readback API exists
//   2. Parse timestamp values -- u64 timestamps returned
//   3. Calculate duration from period -- duration calculation
//   4. Return ProfileResult vec -- labeled results
//   5. Double-buffered readback -- supports async pattern
//
// Coverage:
//   1.  TimestampResult::new() construction
//   2.  TimestampResult::is_valid() with valid ticks
//   3.  TimestampResult::is_valid() with invalid ticks (start > end)
//   4.  TimestampResult::duration_us() calculation
//   5.  TimestampResult::duration_secs() calculation
//   6.  TimestampResult field access (start_ticks, end_ticks, duration_ns, duration_ms)
//   7.  ProfileResult::new() construction without label
//   8.  ProfileResult::with_label() construction
//   9.  ProfileResult::has_label() predicate
//  10.  ProfileResult::label_or() default fallback
//  11.  ProfileResult::label_or_unnamed() convenience
//  12.  ProfileResult field access (result, label)
//  13.  TimestampData::new() construction
//  14.  TimestampData::empty() construction
//  15.  TimestampData::len() and is_empty()
//  16.  TimestampData::get() individual timestamp access
//  17.  TimestampData::pair_count() for start/end pairs
//  18.  TimestampData::get_pair() for timestamp pairs
//  19.  TimestampData::duration_ms() between indices
//  20.  QueryErrorKind::ReadbackNotReady variant exists
//  21.  QueryErrorKind::BufferMappingFailed variant exists
//  22.  QueryError::readback_not_ready() constructor
//  23.  QueryError::buffer_mapping_failed() constructor
//  24.  [GPU] calculate_duration() instance method
//  25.  [GPU] parse_timestamp_pairs() returns ProfileResult vec
//  26.  [GPU] parse_timestamp_pairs() with labels
//  27.  [GPU] parse_timestamp_pairs() with empty labels
//  28.  [GPU] read_timestamps_blocking() returns TimestampData
//  29.  [GPU] read_and_parse_all() convenience method
//  30.  [GPU] Double-buffered pattern simulation

use renderer_backend::query_pool::{
    TimestampResult, ProfileResult, TimestampData,
};

// ---------------------------------------------------------------------------
// TimestampResult Tests (No GPU Required)
// ---------------------------------------------------------------------------

#[test]
fn test_timestamp_result_new_construction() {
    // With 1ns period, 1_000_000 ticks = 1ms
    let result = TimestampResult::new(1_000_000, 2_000_000, 1.0);
    assert_eq!(result.start_ticks, 1_000_000, "start_ticks should be set");
    assert_eq!(result.end_ticks, 2_000_000, "end_ticks should be set");
}

#[test]
fn test_timestamp_result_is_valid_with_valid_ticks() {
    let result = TimestampResult::new(100, 200, 1.0);
    assert!(result.is_valid(), "Result with end > start should be valid");
}

#[test]
fn test_timestamp_result_is_valid_equal_ticks() {
    let result = TimestampResult::new(100, 100, 1.0);
    // Equal ticks are treated as invalid (no elapsed time means the measurement is incomplete)
    // This is a design choice - the API considers end_ticks > start_ticks required
    assert!(!result.is_valid(), "Result with equal ticks should be invalid (no elapsed time)");
}

#[test]
fn test_timestamp_result_is_valid_with_invalid_ticks() {
    let result = TimestampResult::new(200, 100, 1.0);
    assert!(!result.is_valid(), "Result with start > end should be invalid");
}

#[test]
fn test_timestamp_result_duration_ns_calculation() {
    // 1_000_000 tick delta with 1ns period = 1_000_000 ns = 1ms
    let result = TimestampResult::new(0, 1_000_000, 1.0);
    assert!(
        (result.duration_ns - 1_000_000.0).abs() < 0.1,
        "duration_ns should be approximately 1,000,000 ns, got {}",
        result.duration_ns
    );
}

#[test]
fn test_timestamp_result_duration_ms_calculation() {
    // 1_000_000 tick delta with 1ns period = 1ms
    let result = TimestampResult::new(0, 1_000_000, 1.0);
    assert!(
        (result.duration_ms - 1.0).abs() < 0.001,
        "duration_ms should be approximately 1.0 ms, got {}",
        result.duration_ms
    );
}

#[test]
fn test_timestamp_result_duration_us_method() {
    // 1_000_000 tick delta with 1ns period = 1000 us
    let result = TimestampResult::new(0, 1_000_000, 1.0);
    let us = result.duration_us();
    assert!(
        (us - 1000.0).abs() < 0.1,
        "duration_us() should be approximately 1000 us, got {}",
        us
    );
}

#[test]
fn test_timestamp_result_duration_secs_method() {
    // 1_000_000_000 tick delta with 1ns period = 1 second
    let result = TimestampResult::new(0, 1_000_000_000, 1.0);
    let secs = result.duration_secs();
    assert!(
        (secs - 1.0).abs() < 0.001,
        "duration_secs() should be approximately 1.0 s, got {}",
        secs
    );
}

#[test]
fn test_timestamp_result_with_different_period() {
    // With 2ns period, 500_000 ticks = 1_000_000 ns = 1ms
    let result = TimestampResult::new(0, 500_000, 2.0);
    assert!(
        (result.duration_ms - 1.0).abs() < 0.001,
        "500k ticks at 2ns period should be 1ms, got {}",
        result.duration_ms
    );
}

#[test]
fn test_timestamp_result_zero_duration() {
    // When start == end, duration is 0 but result may be considered invalid
    let result = TimestampResult::new(1000, 1000, 1.0);
    // Duration fields should still be computed as 0
    assert!(
        result.duration_ns.abs() < 0.001,
        "Zero tick delta should yield near-zero duration_ns"
    );
    assert!(
        result.duration_ms.abs() < 0.001,
        "Zero tick delta should yield near-zero duration_ms"
    );
}

#[test]
fn test_timestamp_result_large_values() {
    // Test with large tick values (near u64 max / 2)
    let start = 1_000_000_000_000u64;
    let end = 1_000_001_000_000u64; // 1_000_000 tick delta
    let result = TimestampResult::new(start, end, 1.0);
    assert!(result.is_valid());
    assert!(
        (result.duration_ms - 1.0).abs() < 0.001,
        "Large tick values should still calculate correctly"
    );
}

// ---------------------------------------------------------------------------
// ProfileResult Tests (No GPU Required)
// ---------------------------------------------------------------------------

#[test]
fn test_profile_result_new_without_label() {
    let ts_result = TimestampResult::new(0, 1_000_000, 1.0);
    let profile = ProfileResult::new(ts_result);
    assert!(!profile.has_label(), "ProfileResult::new() should not have a label");
}

#[test]
fn test_profile_result_with_label_construction() {
    let ts_result = TimestampResult::new(0, 1_000_000, 1.0);
    let profile = ProfileResult::with_label("RenderPass", ts_result);
    assert!(profile.has_label(), "ProfileResult::with_label() should have a label");
}

#[test]
fn test_profile_result_has_label_predicate() {
    let without_label = ProfileResult::new(TimestampResult::new(0, 1000, 1.0));
    assert!(!without_label.has_label(), "new() creates unlabeled result");

    let with_label = ProfileResult::with_label("Test", TimestampResult::new(0, 1000, 1.0));
    assert!(with_label.has_label(), "with_label() creates labeled result");
}

#[test]
fn test_profile_result_label_or_default_fallback() {
    let ts_result = TimestampResult::new(0, 1000, 1.0);
    let profile = ProfileResult::new(ts_result);

    let label = profile.label_or("DefaultLabel");
    assert_eq!(label, "DefaultLabel", "label_or() should return default when no label");
}

#[test]
fn test_profile_result_label_or_returns_label() {
    let ts_result = TimestampResult::new(0, 1000, 1.0);
    let profile = ProfileResult::with_label("ActualLabel", ts_result);

    let label = profile.label_or("DefaultLabel");
    assert_eq!(label, "ActualLabel", "label_or() should return actual label when present");
}

#[test]
fn test_profile_result_label_or_unnamed_convenience() {
    let ts_result = TimestampResult::new(0, 1000, 1.0);
    let profile = ProfileResult::new(ts_result);

    let label = profile.label_or_unnamed();
    assert!(!label.is_empty(), "label_or_unnamed() should return non-empty default");
}

#[test]
fn test_profile_result_field_access() {
    let ts_result = TimestampResult::new(100, 200, 1.0);
    let profile = ProfileResult::with_label("TestLabel", ts_result);

    // Access the underlying TimestampResult
    assert_eq!(profile.result.start_ticks, 100);
    assert_eq!(profile.result.end_ticks, 200);

    // Access the label
    assert!(profile.label.is_some());
    assert_eq!(profile.label.as_deref(), Some("TestLabel"));
}

#[test]
fn test_profile_result_empty_string_label() {
    let ts_result = TimestampResult::new(0, 1000, 1.0);
    let profile = ProfileResult::with_label("", ts_result);

    // Empty string is still a label
    assert!(profile.has_label(), "Empty string should still count as a label");
    assert_eq!(profile.label_or("default"), "", "Empty label should be returned");
}

// ---------------------------------------------------------------------------
// TimestampData Tests (No GPU Required)
// ---------------------------------------------------------------------------

#[test]
fn test_timestamp_data_new_construction() {
    let timestamps = vec![100u64, 200, 300, 400];
    let data = TimestampData::new(timestamps, 1.0);
    assert_eq!(data.len(), 4, "TimestampData should have 4 timestamps");
}

#[test]
fn test_timestamp_data_empty_construction() {
    let data = TimestampData::empty(1.0);
    assert!(data.is_empty(), "TimestampData::empty() should be empty");
    assert_eq!(data.len(), 0, "Empty data should have length 0");
}

#[test]
fn test_timestamp_data_len_and_is_empty() {
    let empty_data = TimestampData::new(vec![], 1.0);
    assert!(empty_data.is_empty());
    assert_eq!(empty_data.len(), 0);

    let non_empty = TimestampData::new(vec![1, 2, 3], 1.0);
    assert!(!non_empty.is_empty());
    assert_eq!(non_empty.len(), 3);
}

#[test]
fn test_timestamp_data_get_individual_timestamp() {
    let data = TimestampData::new(vec![100, 200, 300], 1.0);

    assert_eq!(data.get(0), Some(100), "First timestamp should be 100");
    assert_eq!(data.get(1), Some(200), "Second timestamp should be 200");
    assert_eq!(data.get(2), Some(300), "Third timestamp should be 300");
    assert_eq!(data.get(3), None, "Out of bounds should return None");
}

#[test]
fn test_timestamp_data_pair_count() {
    // 4 timestamps = 2 pairs (start1, end1, start2, end2)
    let data = TimestampData::new(vec![100, 200, 300, 400], 1.0);
    assert_eq!(data.pair_count(), 2, "4 timestamps should form 2 pairs");

    // 3 timestamps = 1 pair (with 1 leftover)
    let data3 = TimestampData::new(vec![100, 200, 300], 1.0);
    assert_eq!(data3.pair_count(), 1, "3 timestamps should form 1 pair");

    // 1 timestamp = 0 pairs
    let data1 = TimestampData::new(vec![100], 1.0);
    assert_eq!(data1.pair_count(), 0, "1 timestamp cannot form a pair");
}

#[test]
fn test_timestamp_data_get_pair() {
    let data = TimestampData::new(vec![100, 200, 300, 400], 1.0);

    let pair0 = data.get_pair(0);
    assert_eq!(pair0, Some((100, 200)), "First pair should be (100, 200)");

    let pair1 = data.get_pair(1);
    assert_eq!(pair1, Some((300, 400)), "Second pair should be (300, 400)");

    let pair2 = data.get_pair(2);
    assert_eq!(pair2, None, "Out of bounds pair should return None");
}

#[test]
fn test_timestamp_data_duration_ms_between_indices() {
    // With 1ns period, 1_000_000 tick difference = 1ms
    let data = TimestampData::new(vec![0, 1_000_000, 2_000_000, 3_000_000], 1.0);

    let dur_0_1 = data.duration_ms(0, 1);
    assert!(dur_0_1.is_some());
    assert!(
        (dur_0_1.unwrap() - 1.0).abs() < 0.001,
        "Duration from index 0 to 1 should be 1ms"
    );

    let dur_0_3 = data.duration_ms(0, 3);
    assert!(dur_0_3.is_some());
    assert!(
        (dur_0_3.unwrap() - 3.0).abs() < 0.001,
        "Duration from index 0 to 3 should be 3ms"
    );
}

#[test]
fn test_timestamp_data_duration_ms_out_of_bounds() {
    let data = TimestampData::new(vec![100, 200], 1.0);

    let dur = data.duration_ms(0, 10); // Index 10 is out of bounds
    assert!(dur.is_none(), "Out of bounds should return None");
}

#[test]
fn test_timestamp_data_duration_ms_reversed_order() {
    // Test when end index is before start index
    let data = TimestampData::new(vec![0, 1_000_000, 2_000_000], 1.0);

    let dur = data.duration_ms(2, 0);
    // Implementation may return negative or absolute value
    assert!(dur.is_some(), "Reversed indices should still return a value");
}

#[test]
fn test_timestamp_data_preserves_period() {
    let data = TimestampData::new(vec![0, 500_000], 2.0); // 2ns period

    // With 2ns period, 500_000 ticks = 1ms
    let dur = data.duration_ms(0, 1);
    assert!(dur.is_some());
    assert!(
        (dur.unwrap() - 1.0).abs() < 0.001,
        "Period should affect duration calculation"
    );
}

// ---------------------------------------------------------------------------
// QueryError Readback Variants Tests (No GPU Required)
// ---------------------------------------------------------------------------

#[test]
fn test_query_error_kind_readback_not_ready_exists() {
    let _kind = QueryErrorKind::ReadbackNotReady;
}

#[test]
fn test_query_error_kind_buffer_mapping_failed_exists() {
    let _kind = QueryErrorKind::BufferMappingFailed;
}

#[test]
fn test_query_error_readback_not_ready_constructor() {
    let error = QueryError::readback_not_ready("Data not yet available");
    assert_eq!(
        error.kind(),
        QueryErrorKind::ReadbackNotReady,
        "readback_not_ready should have ReadbackNotReady kind"
    );
}

#[test]
fn test_query_error_buffer_mapping_failed_constructor() {
    let error = QueryError::buffer_mapping_failed("GPU mapping error");
    assert_eq!(
        error.kind(),
        QueryErrorKind::BufferMappingFailed,
        "buffer_mapping_failed should have BufferMappingFailed kind"
    );
}

#[test]
fn test_query_error_readback_display() {
    let error = QueryError::readback_not_ready("Test reason");
    let display = format!("{}", error);
    assert!(!display.is_empty(), "Readback error should have display text");
}

#[test]
fn test_query_error_buffer_mapping_display() {
    let error = QueryError::buffer_mapping_failed("Test failure");
    let display = format!("{}", error);
    assert!(!display.is_empty(), "Buffer mapping error should have display text");
}

// ---------------------------------------------------------------------------
// GPU Readback Tests (T-WGPU-P4.4.3)
// ---------------------------------------------------------------------------

#[test]
fn test_calculate_duration_instance_method() {
    let Some((device, queue)) = create_test_device() else {
        return;
    };

    if !TimestampQueryPool::is_supported(&device) {
        eprintln!("Skipping GPU test: TIMESTAMP_QUERY not supported");
        return;
    }

    let pool = TimestampQueryPool::new(&device, &queue, 8).unwrap();

    // Calculate duration between two tick values
    let result = pool.calculate_duration(0, 1_000_000);

    assert!(result.is_valid(), "Duration result should be valid");
    assert_eq!(result.start_ticks, 0, "Start ticks should match");
    assert_eq!(result.end_ticks, 1_000_000, "End ticks should match");
    // Duration depends on the device's timestamp period
    assert!(result.duration_ns > 0.0, "Duration should be positive");
}

#[test]
fn test_calculate_duration_zero_delta() {
    let Some((device, queue)) = create_test_device() else {
        return;
    };

    if !TimestampQueryPool::is_supported(&device) {
        return;
    }

    let pool = TimestampQueryPool::new(&device, &queue, 8).unwrap();
    let result = pool.calculate_duration(1000, 1000);

    // Zero duration (start == end) is considered invalid per API design
    assert!(!result.is_valid(), "Zero duration (equal ticks) should be invalid");
    assert!(result.duration_ns.abs() < 0.001, "Zero delta should yield near-zero duration");
}

#[test]
fn test_parse_timestamp_pairs_returns_profile_results() {
    let Some((device, queue)) = create_test_device() else {
        return;
    };

    if !TimestampQueryPool::is_supported(&device) {
        return;
    }

    let pool = TimestampQueryPool::new(&device, &queue, 8).unwrap();

    // Create timestamp data with 2 pairs
    let data = TimestampData::new(vec![100, 200, 300, 400], pool.timestamp_period());
    let labels: Option<&[&str]> = None;

    let results = pool.parse_timestamp_pairs(&data, labels);

    assert_eq!(results.len(), 2, "Should return 2 ProfileResults for 4 timestamps");
    assert!(!results[0].has_label(), "Results without labels should be unlabeled");
}

#[test]
fn test_parse_timestamp_pairs_with_labels() {
    let Some((device, queue)) = create_test_device() else {
        return;
    };

    if !TimestampQueryPool::is_supported(&device) {
        return;
    }

    let pool = TimestampQueryPool::new(&device, &queue, 8).unwrap();

    let data = TimestampData::new(vec![100, 200, 300, 400], pool.timestamp_period());
    let labels: Vec<&str> = vec!["Pass1", "Pass2"];

    let results = pool.parse_timestamp_pairs(&data, Some(&labels));

    assert_eq!(results.len(), 2, "Should return 2 ProfileResults");
    assert!(results[0].has_label(), "First result should have label");
    assert_eq!(results[0].label_or(""), "Pass1", "First label should be Pass1");
    assert_eq!(results[1].label_or(""), "Pass2", "Second label should be Pass2");
}

#[test]
fn test_parse_timestamp_pairs_fewer_labels_than_pairs() {
    let Some((device, queue)) = create_test_device() else {
        return;
    };

    if !TimestampQueryPool::is_supported(&device) {
        return;
    }

    let pool = TimestampQueryPool::new(&device, &queue, 8).unwrap();

    // 4 timestamps = 2 pairs, but only 1 label
    let data = TimestampData::new(vec![100, 200, 300, 400], pool.timestamp_period());
    let labels: Vec<&str> = vec!["OnlyOne"];

    let results = pool.parse_timestamp_pairs(&data, Some(&labels));

    assert_eq!(results.len(), 2, "Should still return 2 ProfileResults");
    assert!(results[0].has_label(), "First result should have label");
    // Second result may or may not have a label depending on implementation
}

#[test]
fn test_parse_timestamp_pairs_empty_data() {
    let Some((device, queue)) = create_test_device() else {
        return;
    };

    if !TimestampQueryPool::is_supported(&device) {
        return;
    }

    let pool = TimestampQueryPool::new(&device, &queue, 8).unwrap();

    let data = TimestampData::empty(pool.timestamp_period());
    let results = pool.parse_timestamp_pairs(&data, None);

    assert!(results.is_empty(), "Empty data should return empty results");
}

#[test]
fn test_parse_timestamp_pairs_single_timestamp() {
    let Some((device, queue)) = create_test_device() else {
        return;
    };

    if !TimestampQueryPool::is_supported(&device) {
        return;
    }

    let pool = TimestampQueryPool::new(&device, &queue, 8).unwrap();

    // Single timestamp cannot form a pair
    let data = TimestampData::new(vec![100], pool.timestamp_period());
    let results = pool.parse_timestamp_pairs(&data, None);

    assert!(results.is_empty(), "Single timestamp should return no pairs");
}

#[test]
fn test_read_timestamps_blocking_returns_data() {
    let Some((device, queue)) = create_test_device() else {
        return;
    };

    if !TimestampQueryPool::is_supported(&device) {
        eprintln!("Skipping GPU test: TIMESTAMP_QUERY not supported");
        return;
    }

    let pool = TimestampQueryPool::new(&device, &queue, 8).unwrap();

    // Resolve queries first
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("ReadbackTestEncoder"),
    });
    pool.resolve_all(&mut encoder).unwrap();
    queue.submit(std::iter::once(encoder.finish()));

    // Wait for GPU to complete
    device.poll(wgpu::Maintain::Wait);

    // Read timestamps (full range)
    let result = pool.read_timestamps_blocking(&device, 0..pool.capacity());

    assert!(result.is_ok(), "Blocking read should succeed: {:?}", result.err());
    let data = result.unwrap();
    assert_eq!(data.len(), pool.capacity() as usize, "Should return capacity timestamps");
}

#[test]
fn test_read_timestamps_blocking_has_timestamp_values() {
    let Some((device, queue)) = create_test_device() else {
        return;
    };

    if !TimestampQueryPool::is_supported(&device) {
        return;
    }

    let pool = TimestampQueryPool::new(&device, &queue, 4).unwrap();

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("TimestampValuesEncoder"),
    });
    pool.resolve_all(&mut encoder).unwrap();
    queue.submit(std::iter::once(encoder.finish()));
    device.poll(wgpu::Maintain::Wait);

    let data = pool.read_timestamps_blocking(&device, 0..pool.capacity()).unwrap();

    // Timestamps should be u64 values (may be 0 if queries weren't written to)
    for i in 0..data.len() {
        let ts = data.get(i);
        assert!(ts.is_some(), "Should be able to get timestamp at index {}", i);
        // The value is a u64, which is the expected type
        let _value: u64 = ts.unwrap();
    }
}

#[test]
fn test_read_and_parse_all_convenience() {
    let Some((device, queue)) = create_test_device() else {
        return;
    };

    if !TimestampQueryPool::is_supported(&device) {
        eprintln!("Skipping GPU test: TIMESTAMP_QUERY not supported");
        return;
    }

    let pool = TimestampQueryPool::new(&device, &queue, 8).unwrap();

    // Resolve first
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("ParseAllEncoder"),
    });
    pool.resolve_all(&mut encoder).unwrap();
    queue.submit(std::iter::once(encoder.finish()));
    device.poll(wgpu::Maintain::Wait);

    // Use convenience method
    let labels: Vec<&str> = vec!["A", "B", "C", "D"];
    let result = pool.read_and_parse_all(&device, Some(&labels));

    assert!(result.is_ok(), "read_and_parse_all should succeed: {:?}", result.err());
    let profiles = result.unwrap();

    // 8 timestamps = 4 pairs
    assert_eq!(profiles.len(), 4, "Should return 4 profile results for 8 timestamps");
}

#[test]
fn test_read_and_parse_all_without_labels() {
    let Some((device, queue)) = create_test_device() else {
        return;
    };

    if !TimestampQueryPool::is_supported(&device) {
        return;
    }

    let pool = TimestampQueryPool::new(&device, &queue, 4).unwrap();

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("ParseAllNoLabelsEncoder"),
    });
    pool.resolve_all(&mut encoder).unwrap();
    queue.submit(std::iter::once(encoder.finish()));
    device.poll(wgpu::Maintain::Wait);

    let result = pool.read_and_parse_all(&device, None);

    assert!(result.is_ok(), "read_and_parse_all without labels should succeed");
    let profiles = result.unwrap();

    for profile in &profiles {
        assert!(!profile.has_label(), "Results should not have labels");
    }
}

#[test]
fn test_double_buffered_readback_pattern() {
    let Some((device, queue)) = create_test_device() else {
        return;
    };

    if !TimestampQueryPool::is_supported(&device) {
        eprintln!("Skipping GPU test: TIMESTAMP_QUERY not supported");
        return;
    }

    // Simulate double-buffered pattern with two pools
    let pool_a = TimestampQueryPool::new(&device, &queue, 4).unwrap();
    let pool_b = TimestampQueryPool::new(&device, &queue, 4).unwrap();

    // Frame N: resolve pool A while reading pool B (which was resolved last frame)
    let mut encoder_a = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("PoolAEncoder"),
    });
    pool_a.resolve_all(&mut encoder_a).unwrap();
    queue.submit(std::iter::once(encoder_a.finish()));

    // Resolve pool B for next frame
    let mut encoder_b = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("PoolBEncoder"),
    });
    pool_b.resolve_all(&mut encoder_b).unwrap();
    queue.submit(std::iter::once(encoder_b.finish()));

    device.poll(wgpu::Maintain::Wait);

    // Read from pool A (already resolved)
    let result_a = pool_a.read_timestamps_blocking(&device, 0..pool_a.capacity());
    assert!(result_a.is_ok(), "Reading from pool A should succeed");

    // Read from pool B (already resolved)
    let result_b = pool_b.read_timestamps_blocking(&device, 0..pool_b.capacity());
    assert!(result_b.is_ok(), "Reading from pool B should succeed");

    // Both pools should return valid data independently
    assert_eq!(result_a.unwrap().len(), 4);
    assert_eq!(result_b.unwrap().len(), 4);
}

#[test]
fn test_timestamp_result_with_real_period() {
    let Some((device, queue)) = create_test_device() else {
        return;
    };

    if !TimestampQueryPool::is_supported(&device) {
        return;
    }

    let pool = TimestampQueryPool::new(&device, &queue, 4).unwrap();
    let period = pool.timestamp_period();

    // Create a result using the real device period
    let result = TimestampResult::new(0, 1_000_000, period);

    assert!(result.is_valid());
    // Duration depends on actual GPU period, so just verify it's calculated
    assert!(result.duration_ns >= 0.0, "Duration should be non-negative");
    assert!(result.duration_ms >= 0.0, "Duration in ms should be non-negative");
}

#[test]
fn test_profile_result_duration_access() {
    let Some((device, queue)) = create_test_device() else {
        return;
    };

    if !TimestampQueryPool::is_supported(&device) {
        return;
    }

    let pool = TimestampQueryPool::new(&device, &queue, 4).unwrap();
    let period = pool.timestamp_period();

    let ts_result = TimestampResult::new(100, 1_000_100, period);
    let profile = ProfileResult::with_label("TestPass", ts_result);

    // Access duration through the ProfileResult
    assert!(profile.result.duration_ms >= 0.0);
    let us = profile.result.duration_us();
    assert!(us >= 0.0);
}

#[test]
fn test_multiple_read_cycles() {
    let Some((device, queue)) = create_test_device() else {
        return;
    };

    if !TimestampQueryPool::is_supported(&device) {
        return;
    }

    let mut pool = TimestampQueryPool::new(&device, &queue, 4).unwrap();

    // Cycle 1: resolve and read
    let mut encoder1 = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Cycle1Encoder"),
    });
    pool.resolve_all(&mut encoder1).unwrap();
    queue.submit(std::iter::once(encoder1.finish()));
    device.poll(wgpu::Maintain::Wait);

    let data1 = pool.read_timestamps_blocking(&device, 0..pool.capacity());
    assert!(data1.is_ok(), "First read cycle should succeed");

    // Reset the pool
    pool.reset();

    // Cycle 2: resolve and read again
    let mut encoder2 = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Cycle2Encoder"),
    });
    pool.resolve_all(&mut encoder2).unwrap();
    queue.submit(std::iter::once(encoder2.finish()));
    device.poll(wgpu::Maintain::Wait);

    let data2 = pool.read_timestamps_blocking(&device, 0..pool.capacity());
    assert!(data2.is_ok(), "Second read cycle should succeed");
}

#[test]
fn test_timestamp_data_timestamps_field_access() {
    // Test that we can access the underlying timestamps vector
    let timestamps = vec![100u64, 200, 300, 400, 500];
    let data = TimestampData::new(timestamps.clone(), 1.0);

    // Access via get() method
    for (i, &expected) in timestamps.iter().enumerate() {
        assert_eq!(data.get(i), Some(expected), "Timestamp at index {} should match", i);
    }
}

#[test]
fn test_profile_result_in_collections() {
    // ProfileResult should be usable in collections
    let profiles = vec![
        ProfileResult::new(TimestampResult::new(0, 1000, 1.0)),
        ProfileResult::with_label("Labeled", TimestampResult::new(0, 2000, 1.0)),
    ];

    assert_eq!(profiles.len(), 2);
    assert!(!profiles[0].has_label());
    assert!(profiles[1].has_label());
}
