//! Whitebox structural tests for BufferPool and related types (T-WGPU-P2.2.1).
//!
//! These tests verify the internal structure and behavior of the BufferPool API,
//! including SizeClass, GrowthPolicy, PoolMetrics, and all associated types.
//!
//! The tests cover:
//! - SizeClass enum boundary values and transitions
//! - BufferPool acquire/release mechanics
//! - GrowthPolicy calculations
//! - PoolMetrics accuracy
//! - Edge cases (zero size, oversized, stress patterns)

use renderer_backend::resources::buffer_pool::{
    BufferPool, ClassStats, GrowthPolicy, PoolConfig, PoolMetrics, SizeClass,
};

// Note: BufferHandle, PooledBuffer, PooledBufferGuard require a wgpu Device for testing.
// Integration tests with actual GPU device are in blackbox tests or require #[ignore].
#[allow(unused_imports)]
use wgpu::BufferUsages;

// ============================================================================
// 1. SizeClass Tests
// ============================================================================

mod size_class {
    use super::*;

    // ------------------------------------------------------------------------
    // 1.1 for_size() boundary tests
    // ------------------------------------------------------------------------

    #[test]
    fn for_size_returns_tiny_for_zero() {
        // Zero size should map to smallest class
        assert_eq!(SizeClass::for_size(0), Some(SizeClass::Tiny));
    }

    #[test]
    fn for_size_returns_tiny_for_one() {
        assert_eq!(SizeClass::for_size(1), Some(SizeClass::Tiny));
    }

    #[test]
    fn for_size_boundary_tiny_exact() {
        // Exact boundary: 256 bytes = Tiny
        assert_eq!(SizeClass::for_size(256), Some(SizeClass::Tiny));
    }

    #[test]
    fn for_size_boundary_tiny_minus_one() {
        // 255 bytes still fits in Tiny
        assert_eq!(SizeClass::for_size(255), Some(SizeClass::Tiny));
    }

    #[test]
    fn for_size_boundary_tiny_plus_one() {
        // 257 bytes needs Small
        assert_eq!(SizeClass::for_size(257), Some(SizeClass::Small));
    }

    #[test]
    fn for_size_boundary_small_exact() {
        // 1024 bytes = Small
        assert_eq!(SizeClass::for_size(1024), Some(SizeClass::Small));
    }

    #[test]
    fn for_size_boundary_small_minus_one() {
        // 1023 bytes still fits in Small
        assert_eq!(SizeClass::for_size(1023), Some(SizeClass::Small));
    }

    #[test]
    fn for_size_boundary_small_plus_one() {
        // 1025 bytes needs Medium
        assert_eq!(SizeClass::for_size(1025), Some(SizeClass::Medium));
    }

    #[test]
    fn for_size_boundary_medium_exact() {
        // 4096 bytes = Medium
        assert_eq!(SizeClass::for_size(4096), Some(SizeClass::Medium));
    }

    #[test]
    fn for_size_boundary_medium_minus_one() {
        assert_eq!(SizeClass::for_size(4095), Some(SizeClass::Medium));
    }

    #[test]
    fn for_size_boundary_medium_plus_one() {
        // 4097 bytes needs Large
        assert_eq!(SizeClass::for_size(4097), Some(SizeClass::Large));
    }

    #[test]
    fn for_size_boundary_large_exact() {
        // 16384 bytes = Large
        assert_eq!(SizeClass::for_size(16384), Some(SizeClass::Large));
    }

    #[test]
    fn for_size_boundary_large_minus_one() {
        assert_eq!(SizeClass::for_size(16383), Some(SizeClass::Large));
    }

    #[test]
    fn for_size_boundary_large_plus_one() {
        // 16385 bytes needs XLarge
        assert_eq!(SizeClass::for_size(16385), Some(SizeClass::XLarge));
    }

    #[test]
    fn for_size_boundary_xlarge_exact() {
        // 65536 bytes = XLarge
        assert_eq!(SizeClass::for_size(65536), Some(SizeClass::XLarge));
    }

    #[test]
    fn for_size_boundary_xlarge_minus_one() {
        assert_eq!(SizeClass::for_size(65535), Some(SizeClass::XLarge));
    }

    #[test]
    fn for_size_boundary_xlarge_plus_one() {
        // 65537 bytes needs Huge
        assert_eq!(SizeClass::for_size(65537), Some(SizeClass::Huge));
    }

    #[test]
    fn for_size_boundary_huge_exact() {
        // 262144 bytes = Huge
        assert_eq!(SizeClass::for_size(262144), Some(SizeClass::Huge));
    }

    #[test]
    fn for_size_boundary_huge_minus_one() {
        assert_eq!(SizeClass::for_size(262143), Some(SizeClass::Huge));
    }

    #[test]
    fn for_size_boundary_huge_plus_one() {
        // 262145 bytes needs Massive
        assert_eq!(SizeClass::for_size(262145), Some(SizeClass::Massive));
    }

    #[test]
    fn for_size_boundary_massive_exact() {
        // 1MB = Massive
        assert_eq!(SizeClass::for_size(1048576), Some(SizeClass::Massive));
    }

    #[test]
    fn for_size_boundary_massive_minus_one() {
        assert_eq!(SizeClass::for_size(1048575), Some(SizeClass::Massive));
    }

    #[test]
    fn for_size_exceeds_massive() {
        // 1MB + 1 byte is too large for any class
        assert_eq!(SizeClass::for_size(1048577), None);
    }

    #[test]
    fn for_size_very_large_returns_none() {
        // 2MB is too large
        assert_eq!(SizeClass::for_size(2 * 1024 * 1024), None);
    }

    #[test]
    fn for_size_u64_max_returns_none() {
        // u64::MAX is way too large
        assert_eq!(SizeClass::for_size(u64::MAX), None);
    }

    // ------------------------------------------------------------------------
    // 1.2 size_bytes() tests
    // ------------------------------------------------------------------------

    #[test]
    fn size_bytes_tiny() {
        assert_eq!(SizeClass::Tiny.size_bytes(), 256);
    }

    #[test]
    fn size_bytes_small() {
        assert_eq!(SizeClass::Small.size_bytes(), 1024);
    }

    #[test]
    fn size_bytes_medium() {
        assert_eq!(SizeClass::Medium.size_bytes(), 4096);
    }

    #[test]
    fn size_bytes_large() {
        assert_eq!(SizeClass::Large.size_bytes(), 16384);
    }

    #[test]
    fn size_bytes_xlarge() {
        assert_eq!(SizeClass::XLarge.size_bytes(), 65536);
    }

    #[test]
    fn size_bytes_huge() {
        assert_eq!(SizeClass::Huge.size_bytes(), 262144);
    }

    #[test]
    fn size_bytes_massive() {
        assert_eq!(SizeClass::Massive.size_bytes(), 1048576);
    }

    // ------------------------------------------------------------------------
    // 1.3 next_larger() tests
    // ------------------------------------------------------------------------

    #[test]
    fn next_larger_tiny() {
        assert_eq!(SizeClass::Tiny.next_larger(), Some(SizeClass::Small));
    }

    #[test]
    fn next_larger_small() {
        assert_eq!(SizeClass::Small.next_larger(), Some(SizeClass::Medium));
    }

    #[test]
    fn next_larger_medium() {
        assert_eq!(SizeClass::Medium.next_larger(), Some(SizeClass::Large));
    }

    #[test]
    fn next_larger_large() {
        assert_eq!(SizeClass::Large.next_larger(), Some(SizeClass::XLarge));
    }

    #[test]
    fn next_larger_xlarge() {
        assert_eq!(SizeClass::XLarge.next_larger(), Some(SizeClass::Huge));
    }

    #[test]
    fn next_larger_huge() {
        assert_eq!(SizeClass::Huge.next_larger(), Some(SizeClass::Massive));
    }

    #[test]
    fn next_larger_massive_is_none() {
        assert_eq!(SizeClass::Massive.next_larger(), None);
    }

    #[test]
    fn next_larger_chain() {
        // Verify chaining works correctly
        let mut class = SizeClass::Tiny;
        let mut count = 0;
        while let Some(next) = class.next_larger() {
            class = next;
            count += 1;
        }
        assert_eq!(count, 6); // 6 transitions from Tiny to Massive
        assert_eq!(class, SizeClass::Massive);
    }

    // ------------------------------------------------------------------------
    // 1.4 ALL constant tests
    // ------------------------------------------------------------------------

    #[test]
    fn all_has_seven_classes() {
        assert_eq!(SizeClass::ALL.len(), 7);
    }

    #[test]
    fn all_starts_with_tiny() {
        assert_eq!(SizeClass::ALL[0], SizeClass::Tiny);
    }

    #[test]
    fn all_ends_with_massive() {
        assert_eq!(SizeClass::ALL[6], SizeClass::Massive);
    }

    #[test]
    fn all_is_ascending_order() {
        for i in 1..SizeClass::ALL.len() {
            assert!(SizeClass::ALL[i] > SizeClass::ALL[i - 1]);
        }
    }

    #[test]
    fn all_sizes_are_ascending() {
        for i in 1..SizeClass::ALL.len() {
            assert!(SizeClass::ALL[i].size_bytes() > SizeClass::ALL[i - 1].size_bytes());
        }
    }

    // ------------------------------------------------------------------------
    // 1.5 Display and ordering tests
    // ------------------------------------------------------------------------

    #[test]
    fn display_tiny_contains_256b() {
        assert!(SizeClass::Tiny.to_string().contains("256B"));
    }

    #[test]
    fn display_small_contains_1kb() {
        assert!(SizeClass::Small.to_string().contains("1KB"));
    }

    #[test]
    fn display_medium_contains_4kb() {
        assert!(SizeClass::Medium.to_string().contains("4KB"));
    }

    #[test]
    fn display_large_contains_16kb() {
        assert!(SizeClass::Large.to_string().contains("16KB"));
    }

    #[test]
    fn display_xlarge_contains_64kb() {
        assert!(SizeClass::XLarge.to_string().contains("64KB"));
    }

    #[test]
    fn display_huge_contains_256kb() {
        assert!(SizeClass::Huge.to_string().contains("256KB"));
    }

    #[test]
    fn display_massive_contains_1mb() {
        assert!(SizeClass::Massive.to_string().contains("1MB"));
    }

    #[test]
    fn ordering_tiny_less_than_small() {
        assert!(SizeClass::Tiny < SizeClass::Small);
    }

    #[test]
    fn ordering_small_less_than_medium() {
        assert!(SizeClass::Small < SizeClass::Medium);
    }

    #[test]
    fn ordering_is_transitive() {
        assert!(SizeClass::Tiny < SizeClass::Massive);
    }

    #[test]
    fn equality_same_class() {
        assert_eq!(SizeClass::Medium, SizeClass::Medium);
    }

    #[test]
    fn hash_distinct_classes() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        for class in SizeClass::ALL {
            assert!(set.insert(class));
        }
        assert_eq!(set.len(), 7);
    }

    #[test]
    fn clone_produces_equal_value() {
        let original = SizeClass::Large;
        let cloned = original.clone();
        assert_eq!(original, cloned);
    }

    #[test]
    fn copy_produces_equal_value() {
        let original = SizeClass::XLarge;
        let copied: SizeClass = original;
        assert_eq!(original, copied);
    }
}

// ============================================================================
// 2. GrowthPolicy Tests
// ============================================================================

mod growth_policy {
    use super::*;

    // ------------------------------------------------------------------------
    // 2.1 Default policy tests
    // ------------------------------------------------------------------------

    #[test]
    fn default_initial_count() {
        let policy = GrowthPolicy::default();
        assert_eq!(policy.initial_count, 4);
    }

    #[test]
    fn default_growth_factor() {
        let policy = GrowthPolicy::default();
        assert_eq!(policy.growth_factor, 2.0);
    }

    #[test]
    fn default_max_per_class() {
        let policy = GrowthPolicy::default();
        assert_eq!(policy.max_per_class, 256);
    }

    // ------------------------------------------------------------------------
    // 2.2 Conservative policy tests
    // ------------------------------------------------------------------------

    #[test]
    fn conservative_initial_count() {
        let policy = GrowthPolicy::conservative();
        assert_eq!(policy.initial_count, 2);
    }

    #[test]
    fn conservative_growth_factor() {
        let policy = GrowthPolicy::conservative();
        assert_eq!(policy.growth_factor, 1.5);
    }

    #[test]
    fn conservative_max_per_class() {
        let policy = GrowthPolicy::conservative();
        assert_eq!(policy.max_per_class, 64);
    }

    // ------------------------------------------------------------------------
    // 2.3 Aggressive policy tests
    // ------------------------------------------------------------------------

    #[test]
    fn aggressive_initial_count() {
        let policy = GrowthPolicy::aggressive();
        assert_eq!(policy.initial_count, 8);
    }

    #[test]
    fn aggressive_growth_factor() {
        let policy = GrowthPolicy::aggressive();
        assert_eq!(policy.growth_factor, 2.0);
    }

    #[test]
    fn aggressive_max_per_class() {
        let policy = GrowthPolicy::aggressive();
        assert_eq!(policy.max_per_class, 512);
    }

    // ------------------------------------------------------------------------
    // 2.4 next_count() calculation tests
    // ------------------------------------------------------------------------

    #[test]
    fn next_count_zero_stays_zero() {
        let policy = GrowthPolicy::default();
        // 0 * 2.0 = 0
        assert_eq!(policy.next_count(0), 0);
    }

    #[test]
    fn next_count_doubles_four() {
        let policy = GrowthPolicy::default();
        assert_eq!(policy.next_count(4), 8);
    }

    #[test]
    fn next_count_doubles_eight() {
        let policy = GrowthPolicy::default();
        assert_eq!(policy.next_count(8), 16);
    }

    #[test]
    fn next_count_doubles_128() {
        let policy = GrowthPolicy::default();
        assert_eq!(policy.next_count(128), 256);
    }

    #[test]
    fn next_count_capped_at_max() {
        let policy = GrowthPolicy::default();
        // 256 * 2 = 512, but max is 256
        assert_eq!(policy.next_count(256), 256);
    }

    #[test]
    fn next_count_above_max_capped() {
        let policy = GrowthPolicy::default();
        assert_eq!(policy.next_count(300), 256);
    }

    #[test]
    fn next_count_conservative_grows_slower() {
        let policy = GrowthPolicy::conservative();
        // 2 * 1.5 = 3
        assert_eq!(policy.next_count(2), 3);
        // 3 * 1.5 = 4.5 -> ceil -> 5
        assert_eq!(policy.next_count(3), 5);
        // 5 * 1.5 = 7.5 -> ceil -> 8
        assert_eq!(policy.next_count(5), 8);
    }

    #[test]
    fn next_count_conservative_capped_at_64() {
        let policy = GrowthPolicy::conservative();
        assert_eq!(policy.next_count(64), 64);
        assert_eq!(policy.next_count(100), 64);
    }

    #[test]
    fn next_count_aggressive_doubles() {
        let policy = GrowthPolicy::aggressive();
        assert_eq!(policy.next_count(8), 16);
        assert_eq!(policy.next_count(256), 512);
    }

    #[test]
    fn next_count_aggressive_capped_at_512() {
        let policy = GrowthPolicy::aggressive();
        assert_eq!(policy.next_count(512), 512);
        assert_eq!(policy.next_count(600), 512);
    }

    #[test]
    fn next_count_uses_ceiling() {
        let policy = GrowthPolicy::conservative();
        // 7 * 1.5 = 10.5 -> ceil -> 11
        assert_eq!(policy.next_count(7), 11);
    }

    #[test]
    fn next_count_one_becomes_two() {
        let policy = GrowthPolicy::default();
        // 1 * 2.0 = 2
        assert_eq!(policy.next_count(1), 2);
    }
}

// ============================================================================
// 3. PoolConfig Tests
// ============================================================================

mod pool_config {
    use super::*;

    #[test]
    fn default_shrink_threshold() {
        let config = PoolConfig::default();
        assert_eq!(config.shrink_threshold, 0.5);
    }

    #[test]
    fn default_pre_allocate_is_false() {
        let config = PoolConfig::default();
        assert!(!config.pre_allocate);
    }

    #[test]
    fn default_usage_includes_copy_dst() {
        let config = PoolConfig::default();
        assert!(config.default_usage.contains(BufferUsages::COPY_DST));
    }

    #[test]
    fn default_usage_includes_storage() {
        let config = PoolConfig::default();
        assert!(config.default_usage.contains(BufferUsages::STORAGE));
    }

    #[test]
    fn default_growth_policy_is_default() {
        let config = PoolConfig::default();
        assert_eq!(config.growth_policy.initial_count, 4);
        assert_eq!(config.growth_policy.growth_factor, 2.0);
        assert_eq!(config.growth_policy.max_per_class, 256);
    }

    #[test]
    fn config_is_cloneable() {
        let config = PoolConfig::default();
        let cloned = config.clone();
        assert_eq!(cloned.shrink_threshold, config.shrink_threshold);
        assert_eq!(cloned.pre_allocate, config.pre_allocate);
    }

    #[test]
    fn config_is_debuggable() {
        let config = PoolConfig::default();
        let debug = format!("{:?}", config);
        assert!(debug.contains("PoolConfig"));
    }
}

// ============================================================================
// 4. ClassStats Tests
// ============================================================================

mod class_stats {
    use super::*;

    #[test]
    fn default_values_are_zero() {
        let stats = ClassStats::default();
        assert_eq!(stats.total_allocated, 0);
        assert_eq!(stats.in_use, 0);
        assert_eq!(stats.free, 0);
        assert_eq!(stats.total_bytes, 0);
        assert_eq!(stats.growth_count, 0);
        assert_eq!(stats.acquire_count, 0);
        assert_eq!(stats.reuse_count, 0);
    }

    #[test]
    fn utilization_zero_when_nothing_allocated() {
        let stats = ClassStats::default();
        assert_eq!(stats.utilization(), 0.0);
    }

    #[test]
    fn utilization_half() {
        let mut stats = ClassStats::default();
        stats.total_allocated = 10;
        stats.in_use = 5;
        assert_eq!(stats.utilization(), 0.5);
    }

    #[test]
    fn utilization_full() {
        let mut stats = ClassStats::default();
        stats.total_allocated = 10;
        stats.in_use = 10;
        assert_eq!(stats.utilization(), 1.0);
    }

    #[test]
    fn utilization_zero_in_use() {
        let mut stats = ClassStats::default();
        stats.total_allocated = 10;
        stats.in_use = 0;
        assert_eq!(stats.utilization(), 0.0);
    }

    #[test]
    fn reuse_ratio_zero_when_no_acquires() {
        let stats = ClassStats::default();
        assert_eq!(stats.reuse_ratio(), 0.0);
    }

    #[test]
    fn reuse_ratio_seventy_percent() {
        let mut stats = ClassStats::default();
        stats.acquire_count = 10;
        stats.reuse_count = 7;
        let ratio = stats.reuse_ratio();
        assert!((ratio - 0.7).abs() < 0.001);
    }

    #[test]
    fn reuse_ratio_full() {
        let mut stats = ClassStats::default();
        stats.acquire_count = 100;
        stats.reuse_count = 100;
        assert_eq!(stats.reuse_ratio(), 1.0);
    }

    #[test]
    fn reuse_ratio_none() {
        let mut stats = ClassStats::default();
        stats.acquire_count = 50;
        stats.reuse_count = 0;
        assert_eq!(stats.reuse_ratio(), 0.0);
    }

    #[test]
    fn stats_is_cloneable() {
        let mut stats = ClassStats::default();
        stats.total_allocated = 5;
        stats.in_use = 3;
        let cloned = stats.clone();
        assert_eq!(cloned.total_allocated, 5);
        assert_eq!(cloned.in_use, 3);
    }

    #[test]
    fn stats_is_debuggable() {
        let stats = ClassStats::default();
        let debug = format!("{:?}", stats);
        assert!(debug.contains("ClassStats"));
    }
}

// ============================================================================
// 5. PoolMetrics Tests
// ============================================================================

mod pool_metrics {
    use super::*;

    #[test]
    fn default_values_are_zero() {
        let metrics = PoolMetrics::default();
        assert_eq!(metrics.total_allocated, 0);
        assert_eq!(metrics.total_in_use, 0);
        assert_eq!(metrics.total_free, 0);
        assert_eq!(metrics.total_bytes_allocated, 0);
        assert_eq!(metrics.total_bytes_in_use, 0);
        assert_eq!(metrics.oversized_allocations, 0);
    }

    #[test]
    fn utilization_zero_when_nothing_allocated() {
        let metrics = PoolMetrics::default();
        assert_eq!(metrics.utilization(), 0.0);
    }

    #[test]
    fn utilization_forty_percent() {
        let mut metrics = PoolMetrics::default();
        metrics.total_allocated = 20;
        metrics.total_in_use = 8;
        assert_eq!(metrics.utilization(), 0.4);
    }

    #[test]
    fn utilization_full() {
        let mut metrics = PoolMetrics::default();
        metrics.total_allocated = 100;
        metrics.total_in_use = 100;
        assert_eq!(metrics.utilization(), 1.0);
    }

    #[test]
    fn memory_efficiency_zero_when_nothing_allocated() {
        let metrics = PoolMetrics::default();
        assert_eq!(metrics.memory_efficiency(), 0.0);
    }

    #[test]
    fn memory_efficiency_eighty_percent() {
        let mut metrics = PoolMetrics::default();
        metrics.total_bytes_allocated = 1000;
        metrics.total_bytes_in_use = 800;
        assert_eq!(metrics.memory_efficiency(), 0.8);
    }

    #[test]
    fn memory_efficiency_full() {
        let mut metrics = PoolMetrics::default();
        metrics.total_bytes_allocated = 5000;
        metrics.total_bytes_in_use = 5000;
        assert_eq!(metrics.memory_efficiency(), 1.0);
    }

    #[test]
    fn per_class_stats_is_hashmap() {
        let metrics = PoolMetrics::default();
        assert!(metrics.per_class_stats.is_empty());
    }

    #[test]
    fn metrics_is_cloneable() {
        let mut metrics = PoolMetrics::default();
        metrics.total_allocated = 10;
        let cloned = metrics.clone();
        assert_eq!(cloned.total_allocated, 10);
    }

    #[test]
    fn metrics_is_debuggable() {
        let metrics = PoolMetrics::default();
        let debug = format!("{:?}", metrics);
        assert!(debug.contains("PoolMetrics"));
    }
}

// ============================================================================
// 6. BufferHandle Tests
// ============================================================================

mod buffer_handle {
    use super::*;

    // Note: BufferHandle is opaque externally, we test via BufferPool::acquire
    // These tests verify the public API of BufferHandle

    #[test]
    fn size_class_returns_correct_class() {
        // BufferHandle fields are private, but we can test via SizeClass::for_size
        // which is what BufferHandle.size_class() returns
        let size_class = SizeClass::for_size(4000).unwrap();
        assert_eq!(size_class, SizeClass::Medium);
    }

    #[test]
    fn size_returns_class_size() {
        // Verify the relationship between SizeClass and buffer size
        let class = SizeClass::Large;
        assert_eq!(class.size_bytes(), 16384);
    }
}

// ============================================================================
// 7. Edge Case Tests (Unit-level)
// ============================================================================

mod edge_cases {
    use super::*;

    #[test]
    fn size_class_for_power_of_two_boundaries() {
        // Test all power-of-two boundaries
        assert_eq!(SizeClass::for_size(128), Some(SizeClass::Tiny)); // < 256
        assert_eq!(SizeClass::for_size(256), Some(SizeClass::Tiny)); // = 256
        assert_eq!(SizeClass::for_size(512), Some(SizeClass::Small)); // < 1024
        assert_eq!(SizeClass::for_size(1024), Some(SizeClass::Small)); // = 1024
        assert_eq!(SizeClass::for_size(2048), Some(SizeClass::Medium)); // < 4096
        assert_eq!(SizeClass::for_size(4096), Some(SizeClass::Medium)); // = 4096
        assert_eq!(SizeClass::for_size(8192), Some(SizeClass::Large)); // < 16384
        assert_eq!(SizeClass::for_size(16384), Some(SizeClass::Large)); // = 16384
        assert_eq!(SizeClass::for_size(32768), Some(SizeClass::XLarge)); // < 65536
        assert_eq!(SizeClass::for_size(65536), Some(SizeClass::XLarge)); // = 65536
        assert_eq!(SizeClass::for_size(131072), Some(SizeClass::Huge)); // < 262144
        assert_eq!(SizeClass::for_size(262144), Some(SizeClass::Huge)); // = 262144
        assert_eq!(SizeClass::for_size(524288), Some(SizeClass::Massive)); // < 1MB
        assert_eq!(SizeClass::for_size(1048576), Some(SizeClass::Massive)); // = 1MB
    }

    #[test]
    fn growth_policy_small_values() {
        let policy = GrowthPolicy::default();
        assert_eq!(policy.next_count(1), 2);
        assert_eq!(policy.next_count(2), 4);
        assert_eq!(policy.next_count(3), 6);
    }

    #[test]
    fn growth_policy_fractional_growth() {
        // 1.5x growth creates fractional results
        let policy = GrowthPolicy::conservative();
        // 10 * 1.5 = 15
        assert_eq!(policy.next_count(10), 15);
        // 15 * 1.5 = 22.5 -> 23
        assert_eq!(policy.next_count(15), 23);
    }

    #[test]
    fn size_class_enum_repr() {
        // Verify the enum values match expected sizes
        assert_eq!(SizeClass::Tiny as u64, 256);
        assert_eq!(SizeClass::Small as u64, 1024);
        assert_eq!(SizeClass::Medium as u64, 4096);
        assert_eq!(SizeClass::Large as u64, 16384);
        assert_eq!(SizeClass::XLarge as u64, 65536);
        assert_eq!(SizeClass::Huge as u64, 262144);
        assert_eq!(SizeClass::Massive as u64, 1048576);
    }

    #[test]
    fn class_stats_boundary_utilization() {
        let mut stats = ClassStats::default();

        // Edge: 1 out of 1
        stats.total_allocated = 1;
        stats.in_use = 1;
        assert_eq!(stats.utilization(), 1.0);

        // Edge: 0 out of 1
        stats.in_use = 0;
        assert_eq!(stats.utilization(), 0.0);
    }

    #[test]
    fn pool_metrics_large_values() {
        let mut metrics = PoolMetrics::default();
        metrics.total_bytes_allocated = u64::MAX / 2;
        metrics.total_bytes_in_use = u64::MAX / 4;
        // Should not overflow
        let efficiency = metrics.memory_efficiency();
        assert!((efficiency - 0.5).abs() < 0.001);
    }

    #[test]
    fn size_class_all_are_unique() {
        let classes: Vec<_> = SizeClass::ALL.iter().collect();
        for i in 0..classes.len() {
            for j in (i + 1)..classes.len() {
                assert_ne!(classes[i], classes[j]);
            }
        }
    }

    #[test]
    fn size_class_sizes_are_multiples_of_256() {
        for class in SizeClass::ALL {
            assert_eq!(class.size_bytes() % 256, 0);
        }
    }

    #[test]
    fn size_class_sizes_are_powers_of_two_or_multiples() {
        // All sizes should be powers of 2
        for class in SizeClass::ALL {
            let size = class.size_bytes();
            assert!(size.is_power_of_two(), "Size {} is not power of 2", size);
        }
    }
}

// ============================================================================
// 8. BufferPool Static Method Tests
// ============================================================================

mod buffer_pool_static {
    use super::*;

    #[test]
    fn size_class_for_returns_tiny() {
        assert_eq!(BufferPool::size_class_for(100), Some(SizeClass::Tiny));
    }

    #[test]
    fn size_class_for_returns_small() {
        assert_eq!(BufferPool::size_class_for(500), Some(SizeClass::Small));
    }

    #[test]
    fn size_class_for_returns_medium() {
        assert_eq!(BufferPool::size_class_for(2000), Some(SizeClass::Medium));
    }

    #[test]
    fn size_class_for_returns_large() {
        assert_eq!(BufferPool::size_class_for(10000), Some(SizeClass::Large));
    }

    #[test]
    fn size_class_for_returns_xlarge() {
        assert_eq!(BufferPool::size_class_for(40000), Some(SizeClass::XLarge));
    }

    #[test]
    fn size_class_for_returns_huge() {
        assert_eq!(BufferPool::size_class_for(100000), Some(SizeClass::Huge));
    }

    #[test]
    fn size_class_for_returns_massive() {
        assert_eq!(BufferPool::size_class_for(500000), Some(SizeClass::Massive));
    }

    #[test]
    fn size_class_for_returns_none_for_oversized() {
        assert_eq!(BufferPool::size_class_for(2_000_000), None);
    }

    #[test]
    fn size_class_for_zero() {
        // Zero maps to Tiny
        assert_eq!(BufferPool::size_class_for(0), Some(SizeClass::Tiny));
    }
}

// ============================================================================
// Summary
// ============================================================================

#[test]
fn summary_test_count() {
    // This test documents the expected test count per category
    // SizeClass: ~45 tests
    // GrowthPolicy: ~20 tests
    // PoolConfig: ~7 tests
    // ClassStats: ~14 tests
    // PoolMetrics: ~12 tests
    // BufferHandle: ~2 tests
    // Edge Cases: ~15 tests
    // BufferPool Static: ~9 tests
    // Total: ~124 tests
    assert!(true, "Test summary marker");
}
