// Blackbox contract tests for T-WGPU-P2.4.2 Sampler Cache.
//
// CLEANROOM: No src/ access beyond the public API exported by the crate.
// Tests use only `renderer_backend::resources::*` -- no internal fields,
// no private methods, no implementation details.
//
// Acceptance criteria (T-WGPU-P2.4.2):
//   CachedSampler type alias (Arc<wgpu::Sampler>)
//   SamplerCacheKey::from_descriptor() creates cache key
//   SamplerCacheMetrics tracks cache_size, hits, misses, hit_rate
//   SamplerCache::new(device) creates cache
//   SamplerCache::get_or_create() returns cached or new sampler
//   SamplerCache presets: linear_clamp, linear_repeat, point_clamp, point_repeat, shadow
//   SamplerCache convenience: get_linear, get_point
//   SamplerCache management: metrics, clear, len, is_empty, device
//
// Coverage:
//   1.  SamplerCacheKey::from_descriptor() exists
//   2.  SamplerCacheMetrics fields accessible
//   3.  SamplerCacheMetrics::total_requests() calculation
//   4.  SamplerCacheMetrics::is_empty() behavior
//   5.  SamplerCacheMetrics::hit_rate_percent() formatting
//   6.  CachedSampler is Arc type
//   7.  [GPU] SamplerCache::new() construction
//   8.  [GPU] get_or_create() returns sampler
//   9.  [GPU] Same descriptor returns same Arc (pointer equality)
//  10.  [GPU] Different descriptors return different samplers
//  11.  [GPU] Metrics reflect cache misses
//  12.  [GPU] Metrics reflect cache hits
//  13.  [GPU] hit_rate calculated correctly
//  14.  [GPU] linear_clamp() preset
//  15.  [GPU] linear_repeat() preset
//  16.  [GPU] point_clamp() preset
//  17.  [GPU] point_repeat() preset
//  18.  [GPU] shadow() preset
//  19.  [GPU] get_linear() == linear_clamp()
//  20.  [GPU] get_point() == point_clamp()
//  21.  [GPU] Presets are distinct from each other
//  22.  [GPU] clear() empties cache
//  23.  [GPU] len() reflects cache size
//  24.  [GPU] is_empty() reflects cache state
//  25.  [GPU] device() returns stored device
//  26.  [GPU] Concurrent access safety

use renderer_backend::resources::{
    // Sampler cache types
    CachedSampler, SamplerCache, SamplerCacheKey, SamplerCacheMetrics,
    // Sampler descriptor and helpers
    TrinitySamplerDescriptor, AddressMode, FilterMode, CompareFunction,
};
use std::sync::Arc;

// =============================================================================
// Category 1: API Contract Tests (No GPU Required)
// =============================================================================

mod api_contract_tests {
    use super::*;

    // -------------------------------------------------------------------------
    // SamplerCacheKey Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_sampler_cache_key_from_descriptor_exists() {
        // Verify the from_descriptor constructor exists
        let desc = TrinitySamplerDescriptor::new();
        let _key = SamplerCacheKey::from_descriptor(&desc);
    }

    #[test]
    fn test_sampler_cache_key_from_linear_clamp_descriptor() {
        let desc = TrinitySamplerDescriptor::linear_clamp();
        let _key = SamplerCacheKey::from_descriptor(&desc);
    }

    #[test]
    fn test_sampler_cache_key_from_linear_repeat_descriptor() {
        let desc = TrinitySamplerDescriptor::linear_repeat();
        let _key = SamplerCacheKey::from_descriptor(&desc);
    }

    #[test]
    fn test_sampler_cache_key_from_nearest_clamp_descriptor() {
        let desc = TrinitySamplerDescriptor::nearest_clamp();
        let _key = SamplerCacheKey::from_descriptor(&desc);
    }

    #[test]
    fn test_sampler_cache_key_from_nearest_repeat_descriptor() {
        let desc = TrinitySamplerDescriptor::nearest_repeat();
        let _key = SamplerCacheKey::from_descriptor(&desc);
    }

    #[test]
    fn test_sampler_cache_key_from_shadow_descriptor() {
        let desc = TrinitySamplerDescriptor::shadow();
        let _key = SamplerCacheKey::from_descriptor(&desc);
    }

    #[test]
    fn test_sampler_cache_key_from_trilinear_descriptor() {
        let desc = TrinitySamplerDescriptor::trilinear();
        let _key = SamplerCacheKey::from_descriptor(&desc);
    }

    #[test]
    fn test_sampler_cache_key_from_custom_descriptor() {
        let desc = TrinitySamplerDescriptor::new()
            .filter(FilterMode::Nearest)
            .address_mode(AddressMode::Repeat)
            .anisotropy(4);
        let _key = SamplerCacheKey::from_descriptor(&desc);
    }

    #[test]
    fn test_sampler_cache_key_equality_same_descriptor() {
        let desc = TrinitySamplerDescriptor::linear_clamp();
        let key1 = SamplerCacheKey::from_descriptor(&desc);
        let key2 = SamplerCacheKey::from_descriptor(&desc);
        assert_eq!(key1, key2, "Same descriptor should produce equal keys");
    }

    #[test]
    fn test_sampler_cache_key_equality_equivalent_descriptors() {
        let desc1 = TrinitySamplerDescriptor::linear_clamp();
        let desc2 = TrinitySamplerDescriptor::linear_clamp();
        let key1 = SamplerCacheKey::from_descriptor(&desc1);
        let key2 = SamplerCacheKey::from_descriptor(&desc2);
        assert_eq!(key1, key2, "Equivalent descriptors should produce equal keys");
    }

    #[test]
    fn test_sampler_cache_key_inequality_different_descriptors() {
        let desc1 = TrinitySamplerDescriptor::linear_clamp();
        let desc2 = TrinitySamplerDescriptor::linear_repeat();
        let key1 = SamplerCacheKey::from_descriptor(&desc1);
        let key2 = SamplerCacheKey::from_descriptor(&desc2);
        assert_ne!(key1, key2, "Different descriptors should produce different keys");
    }

    #[test]
    fn test_sampler_cache_key_hash_consistent() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let desc = TrinitySamplerDescriptor::linear_clamp();
        let key = SamplerCacheKey::from_descriptor(&desc);

        let mut hasher1 = DefaultHasher::new();
        key.hash(&mut hasher1);
        let hash1 = hasher1.finish();

        let mut hasher2 = DefaultHasher::new();
        key.hash(&mut hasher2);
        let hash2 = hasher2.finish();

        assert_eq!(hash1, hash2, "Same key should hash consistently");
    }

    #[test]
    fn test_sampler_cache_key_hash_equal_keys() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let desc = TrinitySamplerDescriptor::linear_clamp();
        let key1 = SamplerCacheKey::from_descriptor(&desc);
        let key2 = SamplerCacheKey::from_descriptor(&desc);

        let mut hasher1 = DefaultHasher::new();
        key1.hash(&mut hasher1);

        let mut hasher2 = DefaultHasher::new();
        key2.hash(&mut hasher2);

        assert_eq!(hasher1.finish(), hasher2.finish(), "Equal keys should have equal hashes");
    }

    // -------------------------------------------------------------------------
    // SamplerCacheMetrics Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_sampler_cache_metrics_fields_accessible() {
        let metrics = SamplerCacheMetrics {
            cache_size: 5,
            hits: 10,
            misses: 3,
            hit_rate: 0.769,
        };
        assert_eq!(metrics.cache_size, 5);
        assert_eq!(metrics.hits, 10);
        assert_eq!(metrics.misses, 3);
        assert!((metrics.hit_rate - 0.769).abs() < 0.001);
    }

    #[test]
    fn test_sampler_cache_metrics_total_requests_calculation() {
        let metrics = SamplerCacheMetrics {
            cache_size: 5,
            hits: 10,
            misses: 3,
            hit_rate: 0.769,
        };
        assert_eq!(metrics.total_requests(), 13, "total_requests should be hits + misses");
    }

    #[test]
    fn test_sampler_cache_metrics_total_requests_zero() {
        let metrics = SamplerCacheMetrics {
            cache_size: 0,
            hits: 0,
            misses: 0,
            hit_rate: 0.0,
        };
        assert_eq!(metrics.total_requests(), 0);
    }

    #[test]
    fn test_sampler_cache_metrics_total_requests_only_hits() {
        let metrics = SamplerCacheMetrics {
            cache_size: 1,
            hits: 100,
            misses: 0,
            hit_rate: 1.0,
        };
        assert_eq!(metrics.total_requests(), 100);
    }

    #[test]
    fn test_sampler_cache_metrics_total_requests_only_misses() {
        let metrics = SamplerCacheMetrics {
            cache_size: 5,
            hits: 0,
            misses: 50,
            hit_rate: 0.0,
        };
        assert_eq!(metrics.total_requests(), 50);
    }

    #[test]
    fn test_sampler_cache_metrics_is_empty_true() {
        let metrics = SamplerCacheMetrics {
            cache_size: 0,
            hits: 0,
            misses: 0,
            hit_rate: 0.0,
        };
        assert!(metrics.is_empty(), "Cache with size 0 should be empty");
    }

    #[test]
    fn test_sampler_cache_metrics_is_empty_false() {
        let metrics = SamplerCacheMetrics {
            cache_size: 1,
            hits: 0,
            misses: 1,
            hit_rate: 0.0,
        };
        assert!(!metrics.is_empty(), "Cache with size > 0 should not be empty");
    }

    #[test]
    fn test_sampler_cache_metrics_is_empty_with_cleared_cache() {
        // Even if there were requests, empty cache_size means empty
        let metrics = SamplerCacheMetrics {
            cache_size: 0,
            hits: 100,
            misses: 50,
            hit_rate: 0.667,
        };
        assert!(metrics.is_empty());
    }

    #[test]
    fn test_sampler_cache_metrics_hit_rate_percent_calculation() {
        let metrics = SamplerCacheMetrics {
            cache_size: 5,
            hits: 75,
            misses: 25,
            hit_rate: 0.75,
        };
        let percent = metrics.hit_rate_percent();
        // hit_rate_percent() returns hit_rate * 100.0
        assert!((percent - 75.0).abs() < 0.01, "Expected ~75%, got {}", percent);
    }

    #[test]
    fn test_sampler_cache_metrics_hit_rate_percent_zero() {
        let metrics = SamplerCacheMetrics {
            cache_size: 5,
            hits: 0,
            misses: 10,
            hit_rate: 0.0,
        };
        let percent = metrics.hit_rate_percent();
        assert!((percent - 0.0).abs() < 0.01, "Expected 0%, got {}", percent);
    }

    #[test]
    fn test_sampler_cache_metrics_hit_rate_percent_hundred() {
        let metrics = SamplerCacheMetrics {
            cache_size: 1,
            hits: 100,
            misses: 0,
            hit_rate: 1.0,
        };
        let percent = metrics.hit_rate_percent();
        assert!((percent - 100.0).abs() < 0.01, "Expected 100%, got {}", percent);
    }

    #[test]
    fn test_sampler_cache_metrics_debug_impl() {
        let metrics = SamplerCacheMetrics {
            cache_size: 5,
            hits: 10,
            misses: 3,
            hit_rate: 0.769,
        };
        let debug_str = format!("{:?}", metrics);
        assert!(!debug_str.is_empty(), "Debug output should not be empty");
    }

    #[test]
    fn test_sampler_cache_metrics_clone() {
        let metrics = SamplerCacheMetrics {
            cache_size: 5,
            hits: 10,
            misses: 3,
            hit_rate: 0.769,
        };
        let cloned = metrics.clone();
        assert_eq!(metrics.cache_size, cloned.cache_size);
        assert_eq!(metrics.hits, cloned.hits);
        assert_eq!(metrics.misses, cloned.misses);
        assert!((metrics.hit_rate - cloned.hit_rate).abs() < 0.001);
    }

    // -------------------------------------------------------------------------
    // CachedSampler Type Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_cached_sampler_is_arc_type() {
        // CachedSampler should be an Arc type - verify at compile time
        fn assert_arc_like<T: Clone + Send + Sync>() {}
        assert_arc_like::<CachedSampler>();
    }

    // -------------------------------------------------------------------------
    // SamplerCache Type Trait Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_sampler_cache_is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<SamplerCache>();
    }

    #[test]
    fn test_sampler_cache_is_sync() {
        fn assert_sync<T: Sync>() {}
        assert_sync::<SamplerCache>();
    }
}

// =============================================================================
// Category 2: Cache Behavior Tests (Unit-level, testing key behavior)
// =============================================================================

mod cache_behavior_tests {
    use super::*;

    #[test]
    fn test_same_descriptor_produces_same_key() {
        let desc = TrinitySamplerDescriptor::new()
            .filter(FilterMode::Linear)
            .address_mode(AddressMode::ClampToEdge);

        let key1 = SamplerCacheKey::from_descriptor(&desc);
        let key2 = SamplerCacheKey::from_descriptor(&desc);

        assert_eq!(key1, key2, "Same descriptor should produce identical keys");
    }

    #[test]
    fn test_different_filter_produces_different_key() {
        let desc1 = TrinitySamplerDescriptor::new().filter(FilterMode::Linear);
        let desc2 = TrinitySamplerDescriptor::new().filter(FilterMode::Nearest);

        let key1 = SamplerCacheKey::from_descriptor(&desc1);
        let key2 = SamplerCacheKey::from_descriptor(&desc2);

        assert_ne!(key1, key2, "Different filter modes should produce different keys");
    }

    #[test]
    fn test_different_address_mode_produces_different_key() {
        let desc1 = TrinitySamplerDescriptor::new().address_mode(AddressMode::ClampToEdge);
        let desc2 = TrinitySamplerDescriptor::new().address_mode(AddressMode::Repeat);

        let key1 = SamplerCacheKey::from_descriptor(&desc1);
        let key2 = SamplerCacheKey::from_descriptor(&desc2);

        assert_ne!(key1, key2, "Different address modes should produce different keys");
    }

    #[test]
    fn test_different_address_mode_u_produces_different_key() {
        let desc1 = TrinitySamplerDescriptor::new()
            .address_mode_uvw(AddressMode::Repeat, AddressMode::ClampToEdge, AddressMode::ClampToEdge);
        let desc2 = TrinitySamplerDescriptor::new()
            .address_mode_uvw(AddressMode::ClampToEdge, AddressMode::ClampToEdge, AddressMode::ClampToEdge);

        let key1 = SamplerCacheKey::from_descriptor(&desc1);
        let key2 = SamplerCacheKey::from_descriptor(&desc2);

        assert_ne!(key1, key2);
    }

    #[test]
    fn test_different_address_mode_v_produces_different_key() {
        let desc1 = TrinitySamplerDescriptor::new()
            .address_mode_uvw(AddressMode::ClampToEdge, AddressMode::Repeat, AddressMode::ClampToEdge);
        let desc2 = TrinitySamplerDescriptor::new()
            .address_mode_uvw(AddressMode::ClampToEdge, AddressMode::ClampToEdge, AddressMode::ClampToEdge);

        let key1 = SamplerCacheKey::from_descriptor(&desc1);
        let key2 = SamplerCacheKey::from_descriptor(&desc2);

        assert_ne!(key1, key2);
    }

    #[test]
    fn test_different_address_mode_w_produces_different_key() {
        let desc1 = TrinitySamplerDescriptor::new()
            .address_mode_uvw(AddressMode::ClampToEdge, AddressMode::ClampToEdge, AddressMode::Repeat);
        let desc2 = TrinitySamplerDescriptor::new()
            .address_mode_uvw(AddressMode::ClampToEdge, AddressMode::ClampToEdge, AddressMode::ClampToEdge);

        let key1 = SamplerCacheKey::from_descriptor(&desc1);
        let key2 = SamplerCacheKey::from_descriptor(&desc2);

        assert_ne!(key1, key2);
    }

    #[test]
    fn test_different_anisotropy_produces_different_key() {
        let desc1 = TrinitySamplerDescriptor::new().anisotropy(1);
        let desc2 = TrinitySamplerDescriptor::new().anisotropy(16);

        let key1 = SamplerCacheKey::from_descriptor(&desc1);
        let key2 = SamplerCacheKey::from_descriptor(&desc2);

        assert_ne!(key1, key2, "Different anisotropy levels should produce different keys");
    }

    #[test]
    fn test_different_compare_function_produces_different_key() {
        let desc1 = TrinitySamplerDescriptor::new().compare(CompareFunction::Less);
        let desc2 = TrinitySamplerDescriptor::new().compare(CompareFunction::Greater);

        let key1 = SamplerCacheKey::from_descriptor(&desc1);
        let key2 = SamplerCacheKey::from_descriptor(&desc2);

        assert_ne!(key1, key2, "Different compare functions should produce different keys");
    }

    #[test]
    fn test_compare_vs_no_compare_produces_different_key() {
        let desc1 = TrinitySamplerDescriptor::new();
        let desc2 = TrinitySamplerDescriptor::new().compare(CompareFunction::Less);

        let key1 = SamplerCacheKey::from_descriptor(&desc1);
        let key2 = SamplerCacheKey::from_descriptor(&desc2);

        assert_ne!(key1, key2, "Comparison vs non-comparison should produce different keys");
    }

    #[test]
    fn test_different_lod_clamp_produces_different_key() {
        let desc1 = TrinitySamplerDescriptor::new().lod_clamp(0.0, 10.0);
        let desc2 = TrinitySamplerDescriptor::new().lod_clamp(0.0, 5.0);

        let key1 = SamplerCacheKey::from_descriptor(&desc1);
        let key2 = SamplerCacheKey::from_descriptor(&desc2);

        assert_ne!(key1, key2, "Different LOD clamp should produce different keys");
    }

    #[test]
    fn test_distinct_presets_produce_unique_keys() {
        // Note: linear_clamp and trilinear are equivalent (both use Linear for all filters
        // and ClampToEdge for address modes), so we test only functionally distinct presets
        let keys = vec![
            ("linear_clamp", SamplerCacheKey::from_descriptor(&TrinitySamplerDescriptor::linear_clamp())),
            ("linear_repeat", SamplerCacheKey::from_descriptor(&TrinitySamplerDescriptor::linear_repeat())),
            ("nearest_clamp", SamplerCacheKey::from_descriptor(&TrinitySamplerDescriptor::nearest_clamp())),
            ("nearest_repeat", SamplerCacheKey::from_descriptor(&TrinitySamplerDescriptor::nearest_repeat())),
            ("shadow", SamplerCacheKey::from_descriptor(&TrinitySamplerDescriptor::shadow())),
        ];

        // Check all pairs are unique
        for i in 0..keys.len() {
            for j in (i + 1)..keys.len() {
                assert_ne!(keys[i].1, keys[j].1,
                    "Preset {} and {} should produce different keys", keys[i].0, keys[j].0);
            }
        }
    }

    #[test]
    fn test_linear_clamp_and_trilinear_are_equivalent() {
        // trilinear() is just linear_clamp() with explicit trilinear mipmap filtering
        // Since linear_clamp() also uses Linear for mipmap_filter, they should be equivalent
        let linear_clamp_key = SamplerCacheKey::from_descriptor(&TrinitySamplerDescriptor::linear_clamp());
        let trilinear_key = SamplerCacheKey::from_descriptor(&TrinitySamplerDescriptor::trilinear());

        assert_eq!(linear_clamp_key, trilinear_key,
            "linear_clamp and trilinear should be equivalent (both use Linear for all filters)");
    }

    #[test]
    fn test_label_does_not_affect_key() {
        // Labels should not be part of the cache key (they're metadata only)
        let desc1 = TrinitySamplerDescriptor::new().label("sampler_a");
        let desc2 = TrinitySamplerDescriptor::new().label("sampler_b");

        let key1 = SamplerCacheKey::from_descriptor(&desc1);
        let key2 = SamplerCacheKey::from_descriptor(&desc2);

        assert_eq!(key1, key2, "Label should not affect cache key");
    }
}

// =============================================================================
// Category 3: Preset Tests (Descriptor-level verification)
// =============================================================================

mod preset_tests {
    use super::*;

    #[test]
    fn test_linear_clamp_preset_exists() {
        let desc = TrinitySamplerDescriptor::linear_clamp();
        assert_eq!(desc.mag_filter, FilterMode::Linear);
        assert_eq!(desc.min_filter, FilterMode::Linear);
        assert_eq!(desc.address_mode_u, AddressMode::ClampToEdge);
        assert_eq!(desc.address_mode_v, AddressMode::ClampToEdge);
    }

    #[test]
    fn test_linear_repeat_preset_exists() {
        let desc = TrinitySamplerDescriptor::linear_repeat();
        assert_eq!(desc.mag_filter, FilterMode::Linear);
        assert_eq!(desc.min_filter, FilterMode::Linear);
        assert_eq!(desc.address_mode_u, AddressMode::Repeat);
        assert_eq!(desc.address_mode_v, AddressMode::Repeat);
    }

    #[test]
    fn test_nearest_clamp_preset_exists() {
        let desc = TrinitySamplerDescriptor::nearest_clamp();
        assert_eq!(desc.mag_filter, FilterMode::Nearest);
        assert_eq!(desc.min_filter, FilterMode::Nearest);
        assert_eq!(desc.address_mode_u, AddressMode::ClampToEdge);
        assert_eq!(desc.address_mode_v, AddressMode::ClampToEdge);
    }

    #[test]
    fn test_nearest_repeat_preset_exists() {
        let desc = TrinitySamplerDescriptor::nearest_repeat();
        assert_eq!(desc.mag_filter, FilterMode::Nearest);
        assert_eq!(desc.min_filter, FilterMode::Nearest);
        assert_eq!(desc.address_mode_u, AddressMode::Repeat);
        assert_eq!(desc.address_mode_v, AddressMode::Repeat);
    }

    #[test]
    fn test_shadow_preset_exists() {
        let desc = TrinitySamplerDescriptor::shadow();
        assert!(desc.compare.is_some(), "Shadow sampler should have compare function");
    }

    #[test]
    fn test_trilinear_preset_exists() {
        let desc = TrinitySamplerDescriptor::trilinear();
        assert_eq!(desc.mag_filter, FilterMode::Linear);
        assert_eq!(desc.min_filter, FilterMode::Linear);
        assert_eq!(desc.mipmap_filter, FilterMode::Linear);
    }

    #[test]
    fn test_linear_clamp_vs_linear_repeat_differ_in_address_mode() {
        let clamp = TrinitySamplerDescriptor::linear_clamp();
        let repeat = TrinitySamplerDescriptor::linear_repeat();

        assert_eq!(clamp.mag_filter, repeat.mag_filter, "Filter should be same");
        assert_ne!(clamp.address_mode_u, repeat.address_mode_u, "Address mode should differ");
    }

    #[test]
    fn test_point_clamp_vs_point_repeat_differ_in_address_mode() {
        let clamp = TrinitySamplerDescriptor::nearest_clamp();
        let repeat = TrinitySamplerDescriptor::nearest_repeat();

        assert_eq!(clamp.mag_filter, repeat.mag_filter, "Filter should be same");
        assert_ne!(clamp.address_mode_u, repeat.address_mode_u, "Address mode should differ");
    }

    #[test]
    fn test_linear_vs_point_differ_in_filter() {
        let linear = TrinitySamplerDescriptor::linear_clamp();
        let point = TrinitySamplerDescriptor::nearest_clamp();

        assert_ne!(linear.mag_filter, point.mag_filter, "Filter should differ");
        assert_eq!(linear.address_mode_u, point.address_mode_u, "Address mode should be same");
    }
}

// =============================================================================
// Category 4: Metrics Tests (Detailed metrics behavior)
// =============================================================================

mod metrics_tests {
    use super::*;

    #[test]
    fn test_metrics_default_values() {
        let metrics = SamplerCacheMetrics {
            cache_size: 0,
            hits: 0,
            misses: 0,
            hit_rate: 0.0,
        };

        assert_eq!(metrics.cache_size, 0);
        assert_eq!(metrics.hits, 0);
        assert_eq!(metrics.misses, 0);
        assert_eq!(metrics.hit_rate, 0.0);
        assert!(metrics.is_empty());
        assert_eq!(metrics.total_requests(), 0);
    }

    #[test]
    fn test_metrics_hit_rate_calculation() {
        // 80 hits, 20 misses = 80% hit rate
        let metrics = SamplerCacheMetrics {
            cache_size: 10,
            hits: 80,
            misses: 20,
            hit_rate: 0.8,
        };

        assert_eq!(metrics.total_requests(), 100);
        assert!((metrics.hit_rate - 0.8).abs() < 0.001);
    }

    #[test]
    fn test_metrics_all_hits() {
        let metrics = SamplerCacheMetrics {
            cache_size: 5,
            hits: 100,
            misses: 0,
            hit_rate: 1.0,
        };

        assert_eq!(metrics.total_requests(), 100);
        assert_eq!(metrics.hit_rate, 1.0);
        assert!(!metrics.is_empty());
    }

    #[test]
    fn test_metrics_all_misses() {
        let metrics = SamplerCacheMetrics {
            cache_size: 50,
            hits: 0,
            misses: 50,
            hit_rate: 0.0,
        };

        assert_eq!(metrics.total_requests(), 50);
        assert_eq!(metrics.hit_rate, 0.0);
        assert!(!metrics.is_empty());
    }

    #[test]
    fn test_metrics_large_numbers() {
        let metrics = SamplerCacheMetrics {
            cache_size: 1000,
            hits: 1_000_000,
            misses: 1000,
            hit_rate: 0.999,
        };

        assert_eq!(metrics.total_requests(), 1_001_000);
        assert!(metrics.hit_rate > 0.99);
    }

    #[test]
    fn test_metrics_hit_rate_percent_various_values() {
        let test_cases = vec![
            (0.0, 0.0),
            (0.5, 50.0),
            (0.75, 75.0),
            (1.0, 100.0),
        ];

        for (rate, expected) in test_cases {
            let metrics = SamplerCacheMetrics {
                cache_size: 1,
                hits: 0,
                misses: 0,
                hit_rate: rate,
            };
            let percent = metrics.hit_rate_percent();
            assert!((percent - expected).abs() < 0.01,
                "hit_rate {} should give {}%, got {}", rate, expected, percent);
        }
    }

    #[test]
    fn test_metrics_copy_semantics() {
        let metrics1 = SamplerCacheMetrics {
            cache_size: 5,
            hits: 10,
            misses: 2,
            hit_rate: 0.833,
        };

        let metrics2 = metrics1.clone();

        assert_eq!(metrics1.cache_size, metrics2.cache_size);
        assert_eq!(metrics1.hits, metrics2.hits);
        assert_eq!(metrics1.misses, metrics2.misses);
    }
}

// =============================================================================
// Category 5: GPU Integration Tests (Require GPU, marked #[ignore])
// =============================================================================

mod integration_tests {
    use super::*;

    /// Helper to create a wgpu device for testing.
    #[allow(dead_code)]
    fn create_test_device() -> Option<(Arc<wgpu::Device>, wgpu::Queue)> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            ..Default::default()
        });

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))?;

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("test_device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::default(),
            },
            None,
        ))
        .ok()?;

        Some((Arc::new(device), queue))
    }

    #[test]
    
    fn test_sampler_cache_new_construction() {
        let (device, _queue) = create_test_device().expect("No GPU available");
        let cache = SamplerCache::new(device);

        assert!(cache.is_empty(), "New cache should be empty");
        assert_eq!(cache.len(), 0, "New cache should have length 0");
    }

    #[test]
    
    fn test_sampler_cache_get_or_create_returns_sampler() {
        let (device, _queue) = create_test_device().expect("No GPU available");
        let cache = SamplerCache::new(device);

        let desc = TrinitySamplerDescriptor::linear_clamp();
        let sampler = cache.get_or_create(&desc);

        // Sampler should be a valid Arc
        assert!(Arc::strong_count(&sampler) >= 1);
    }

    #[test]
    
    fn test_sampler_cache_same_descriptor_returns_same_arc() {
        let (device, _queue) = create_test_device().expect("No GPU available");
        let cache = SamplerCache::new(device);

        let desc = TrinitySamplerDescriptor::linear_clamp();
        let sampler1 = cache.get_or_create(&desc);
        let sampler2 = cache.get_or_create(&desc);

        // Should return the exact same Arc (pointer equality)
        assert!(Arc::ptr_eq(&sampler1, &sampler2),
            "Same descriptor should return the same cached sampler");
    }

    #[test]
    
    fn test_sampler_cache_different_descriptors_return_different_samplers() {
        let (device, _queue) = create_test_device().expect("No GPU available");
        let cache = SamplerCache::new(device);

        let desc1 = TrinitySamplerDescriptor::linear_clamp();
        let desc2 = TrinitySamplerDescriptor::linear_repeat();

        let sampler1 = cache.get_or_create(&desc1);
        let sampler2 = cache.get_or_create(&desc2);

        assert!(!Arc::ptr_eq(&sampler1, &sampler2),
            "Different descriptors should return different samplers");
    }

    #[test]
    
    fn test_sampler_cache_metrics_initial_state() {
        let (device, _queue) = create_test_device().expect("No GPU available");
        let cache = SamplerCache::new(device);

        let metrics = cache.metrics();

        assert_eq!(metrics.cache_size, 0);
        assert_eq!(metrics.hits, 0);
        assert_eq!(metrics.misses, 0);
        assert!(metrics.is_empty());
    }

    #[test]
    
    fn test_sampler_cache_metrics_reflects_cache_miss() {
        let (device, _queue) = create_test_device().expect("No GPU available");
        let cache = SamplerCache::new(device);

        let desc = TrinitySamplerDescriptor::linear_clamp();
        let _sampler = cache.get_or_create(&desc);

        let metrics = cache.metrics();

        assert_eq!(metrics.cache_size, 1, "Cache size should be 1 after first create");
        assert_eq!(metrics.misses, 1, "Should have 1 miss (first access)");
        assert_eq!(metrics.hits, 0, "Should have 0 hits");
    }

    #[test]
    
    fn test_sampler_cache_metrics_reflects_cache_hit() {
        let (device, _queue) = create_test_device().expect("No GPU available");
        let cache = SamplerCache::new(device);

        let desc = TrinitySamplerDescriptor::linear_clamp();
        let _sampler1 = cache.get_or_create(&desc);
        let _sampler2 = cache.get_or_create(&desc);

        let metrics = cache.metrics();

        assert_eq!(metrics.cache_size, 1, "Cache size should still be 1");
        assert_eq!(metrics.misses, 1, "Should have 1 miss (first access)");
        assert_eq!(metrics.hits, 1, "Should have 1 hit (second access)");
    }

    #[test]
    
    fn test_sampler_cache_hit_rate_calculation() {
        let (device, _queue) = create_test_device().expect("No GPU available");
        let cache = SamplerCache::new(device);

        let desc = TrinitySamplerDescriptor::linear_clamp();

        // First access: miss
        let _s1 = cache.get_or_create(&desc);
        // Second access: hit
        let _s2 = cache.get_or_create(&desc);
        // Third access: hit
        let _s3 = cache.get_or_create(&desc);
        // Fourth access: hit
        let _s4 = cache.get_or_create(&desc);

        let metrics = cache.metrics();

        // 1 miss, 3 hits = 75% hit rate
        assert_eq!(metrics.total_requests(), 4);
        assert_eq!(metrics.misses, 1);
        assert_eq!(metrics.hits, 3);
        assert!((metrics.hit_rate - 0.75).abs() < 0.01,
            "Hit rate should be ~75%, got {}", metrics.hit_rate);
    }

    #[test]
    
    fn test_sampler_cache_linear_clamp_preset() {
        let (device, _queue) = create_test_device().expect("No GPU available");
        let cache = SamplerCache::new(device);

        let sampler = cache.linear_clamp();

        assert!(Arc::strong_count(&sampler) >= 1);
    }

    #[test]
    
    fn test_sampler_cache_linear_repeat_preset() {
        let (device, _queue) = create_test_device().expect("No GPU available");
        let cache = SamplerCache::new(device);

        let sampler = cache.linear_repeat();

        assert!(Arc::strong_count(&sampler) >= 1);
    }

    #[test]
    
    fn test_sampler_cache_point_clamp_preset() {
        let (device, _queue) = create_test_device().expect("No GPU available");
        let cache = SamplerCache::new(device);

        let sampler = cache.point_clamp();

        assert!(Arc::strong_count(&sampler) >= 1);
    }

    #[test]
    
    fn test_sampler_cache_point_repeat_preset() {
        let (device, _queue) = create_test_device().expect("No GPU available");
        let cache = SamplerCache::new(device);

        let sampler = cache.point_repeat();

        assert!(Arc::strong_count(&sampler) >= 1);
    }

    #[test]
    
    fn test_sampler_cache_shadow_preset() {
        let (device, _queue) = create_test_device().expect("No GPU available");
        let cache = SamplerCache::new(device);

        let sampler = cache.shadow();

        assert!(Arc::strong_count(&sampler) >= 1);
    }

    #[test]
    
    fn test_sampler_cache_get_linear_equals_linear_clamp() {
        let (device, _queue) = create_test_device().expect("No GPU available");
        let cache = SamplerCache::new(device);

        let linear = cache.get_linear();
        let linear_clamp = cache.linear_clamp();

        assert!(Arc::ptr_eq(&linear, &linear_clamp),
            "get_linear() should return the same sampler as linear_clamp()");
    }

    #[test]
    
    fn test_sampler_cache_get_point_equals_point_clamp() {
        let (device, _queue) = create_test_device().expect("No GPU available");
        let cache = SamplerCache::new(device);

        let point = cache.get_point();
        let point_clamp = cache.point_clamp();

        assert!(Arc::ptr_eq(&point, &point_clamp),
            "get_point() should return the same sampler as point_clamp()");
    }

    #[test]
    
    fn test_sampler_cache_presets_are_distinct() {
        let (device, _queue) = create_test_device().expect("No GPU available");
        let cache = SamplerCache::new(device);

        let linear_clamp = cache.linear_clamp();
        let linear_repeat = cache.linear_repeat();
        let point_clamp = cache.point_clamp();
        let point_repeat = cache.point_repeat();
        let shadow = cache.shadow();

        // All presets should be distinct
        assert!(!Arc::ptr_eq(&linear_clamp, &linear_repeat));
        assert!(!Arc::ptr_eq(&linear_clamp, &point_clamp));
        assert!(!Arc::ptr_eq(&linear_clamp, &point_repeat));
        assert!(!Arc::ptr_eq(&linear_clamp, &shadow));
        assert!(!Arc::ptr_eq(&linear_repeat, &point_clamp));
        assert!(!Arc::ptr_eq(&linear_repeat, &point_repeat));
        assert!(!Arc::ptr_eq(&linear_repeat, &shadow));
        assert!(!Arc::ptr_eq(&point_clamp, &point_repeat));
        assert!(!Arc::ptr_eq(&point_clamp, &shadow));
        assert!(!Arc::ptr_eq(&point_repeat, &shadow));
    }

    #[test]
    
    fn test_sampler_cache_presets_cached() {
        let (device, _queue) = create_test_device().expect("No GPU available");
        let cache = SamplerCache::new(device);

        // Access each preset twice
        let lc1 = cache.linear_clamp();
        let lc2 = cache.linear_clamp();
        let lr1 = cache.linear_repeat();
        let lr2 = cache.linear_repeat();

        // Same preset should return same Arc
        assert!(Arc::ptr_eq(&lc1, &lc2));
        assert!(Arc::ptr_eq(&lr1, &lr2));
    }

    #[test]
    
    fn test_sampler_cache_clear_empties_cache() {
        let (device, _queue) = create_test_device().expect("No GPU available");
        let cache = SamplerCache::new(device);

        // Populate cache via get_or_create (the main cache API)
        let _s1 = cache.get_or_create(&TrinitySamplerDescriptor::linear_clamp());
        let _s2 = cache.get_or_create(&TrinitySamplerDescriptor::linear_repeat());
        let _s3 = cache.get_or_create(&TrinitySamplerDescriptor::nearest_clamp());

        assert!(!cache.is_empty(), "Cache should not be empty after populating");
        assert!(cache.len() >= 3, "Cache should have at least 3 entries");

        // Clear cache
        cache.clear();

        assert!(cache.is_empty(), "Cache should be empty after clear");
        assert_eq!(cache.len(), 0, "Cache len should be 0 after clear");
    }

    #[test]
    
    fn test_sampler_cache_len_reflects_cache_size() {
        let (device, _queue) = create_test_device().expect("No GPU available");
        let cache = SamplerCache::new(device);

        assert_eq!(cache.len(), 0);

        let _s1 = cache.get_or_create(&TrinitySamplerDescriptor::linear_clamp());
        assert_eq!(cache.len(), 1);

        let _s2 = cache.get_or_create(&TrinitySamplerDescriptor::linear_repeat());
        assert_eq!(cache.len(), 2);

        // Accessing same descriptor should not increase len
        let _s3 = cache.get_or_create(&TrinitySamplerDescriptor::linear_clamp());
        assert_eq!(cache.len(), 2);
    }

    #[test]
    
    fn test_sampler_cache_is_empty_reflects_state() {
        let (device, _queue) = create_test_device().expect("No GPU available");
        let cache = SamplerCache::new(device);

        assert!(cache.is_empty(), "New cache should be empty");

        // Use get_or_create to populate the cache
        let _s = cache.get_or_create(&TrinitySamplerDescriptor::linear_clamp());
        assert!(!cache.is_empty(), "Cache should not be empty after get_or_create");

        cache.clear();
        assert!(cache.is_empty(), "Cache should be empty after clear");
    }

    #[test]
    
    fn test_sampler_cache_device_returns_stored_device() {
        let (device, _queue) = create_test_device().expect("No GPU available");
        let device_clone = device.clone();
        let cache = SamplerCache::new(device);

        let cache_device = cache.device();

        // Should be the same device
        assert!(Arc::ptr_eq(&device_clone, cache_device));
    }

    #[test]
    
    fn test_sampler_cache_concurrent_access() {
        use std::thread;

        let (device, _queue) = create_test_device().expect("No GPU available");
        let cache = Arc::new(SamplerCache::new(device));

        let mut handles = vec![];

        // Spawn multiple threads accessing the cache via get_or_create
        for i in 0..4 {
            let cache_clone = cache.clone();
            let handle = thread::spawn(move || {
                // Each thread accesses via get_or_create (main cache API)
                let desc1 = TrinitySamplerDescriptor::linear_clamp();
                let desc2 = TrinitySamplerDescriptor::linear_repeat();
                let _s1 = cache_clone.get_or_create(&desc1);
                let _s2 = cache_clone.get_or_create(&desc2);

                // Access custom descriptor (unique per thread)
                let desc = TrinitySamplerDescriptor::new()
                    .anisotropy((i + 1) as u16);
                let _s3 = cache_clone.get_or_create(&desc);
            });
            handles.push(handle);
        }

        // Wait for all threads
        for handle in handles {
            handle.join().expect("Thread panicked");
        }

        // Cache should have entries from all threads
        // 2 shared (linear_clamp, linear_repeat) + 4 unique (anisotropy 1,2,3,4)
        assert!(!cache.is_empty(), "Cache should not be empty after concurrent access");
        assert!(cache.len() >= 4, "Cache should have entries from concurrent threads");
    }

    #[test]
    
    fn test_sampler_cache_metrics_after_clear() {
        let (device, _queue) = create_test_device().expect("No GPU available");
        let cache = SamplerCache::new(device);

        // Generate activity via get_or_create (main cache API)
        let desc1 = TrinitySamplerDescriptor::linear_clamp();
        let desc2 = TrinitySamplerDescriptor::linear_repeat();
        let _s1 = cache.get_or_create(&desc1);
        let _s2 = cache.get_or_create(&desc1); // hit
        let _s3 = cache.get_or_create(&desc2);

        let metrics_before = cache.metrics();
        assert_eq!(metrics_before.cache_size, 2, "Should have 2 cached samplers");

        cache.clear();

        let metrics_after = cache.metrics();
        assert_eq!(metrics_after.cache_size, 0, "Cache size should be 0 after clear");
        // Note: hit/miss counters may or may not be cleared depending on implementation
    }

    #[test]
    
    fn test_sampler_cache_after_clear_creates_new_samplers() {
        let (device, _queue) = create_test_device().expect("No GPU available");
        let cache = SamplerCache::new(device);

        let desc = TrinitySamplerDescriptor::linear_clamp();
        let sampler_before = cache.get_or_create(&desc);
        cache.clear();
        let sampler_after = cache.get_or_create(&desc);

        // After clear, a new sampler should be created (different Arc)
        assert!(!Arc::ptr_eq(&sampler_before, &sampler_after),
            "After clear, get_or_create should create a new sampler");
    }

    #[test]
    
    fn test_sampler_cache_multiple_custom_descriptors() {
        let (device, _queue) = create_test_device().expect("No GPU available");
        let cache = SamplerCache::new(device);

        // Create several unique samplers
        let descriptors = vec![
            TrinitySamplerDescriptor::new().filter(FilterMode::Linear).anisotropy(1),
            TrinitySamplerDescriptor::new().filter(FilterMode::Linear).anisotropy(2),
            TrinitySamplerDescriptor::new().filter(FilterMode::Linear).anisotropy(4),
            TrinitySamplerDescriptor::new().filter(FilterMode::Linear).anisotropy(8),
            TrinitySamplerDescriptor::new().filter(FilterMode::Linear).anisotropy(16),
        ];

        let samplers: Vec<_> = descriptors.iter()
            .map(|d| cache.get_or_create(d))
            .collect();

        assert_eq!(cache.len(), 5);

        // All should be unique
        for i in 0..samplers.len() {
            for j in (i + 1)..samplers.len() {
                assert!(!Arc::ptr_eq(&samplers[i], &samplers[j]),
                    "Samplers {} and {} should be different", i, j);
            }
        }
    }

    #[test]
    
    fn test_sampler_cache_high_hit_rate_scenario() {
        let (device, _queue) = create_test_device().expect("No GPU available");
        let cache = SamplerCache::new(device);

        let desc = TrinitySamplerDescriptor::linear_clamp();

        // 1 miss (first access)
        let _first = cache.get_or_create(&desc);

        // 99 hits
        for _ in 0..99 {
            let _s = cache.get_or_create(&desc);
        }

        let metrics = cache.metrics();

        assert_eq!(metrics.total_requests(), 100);
        assert_eq!(metrics.misses, 1);
        assert_eq!(metrics.hits, 99);
        assert!((metrics.hit_rate - 0.99).abs() < 0.01);
    }

    #[test]
    
    fn test_sampler_cache_all_miss_scenario() {
        let (device, _queue) = create_test_device().expect("No GPU available");
        let cache = SamplerCache::new(device);

        // Create 10 unique samplers - all misses
        for i in 0..10 {
            let desc = TrinitySamplerDescriptor::new().anisotropy(i + 1);
            let _s = cache.get_or_create(&desc);
        }

        let metrics = cache.metrics();

        assert_eq!(metrics.cache_size, 10);
        assert_eq!(metrics.misses, 10);
        assert_eq!(metrics.hits, 0);
        assert_eq!(metrics.hit_rate, 0.0);
    }

    #[test]
    
    fn test_sampler_cache_mixed_access_pattern() {
        let (device, _queue) = create_test_device().expect("No GPU available");
        let cache = SamplerCache::new(device);

        let desc_a = TrinitySamplerDescriptor::linear_clamp();
        let desc_b = TrinitySamplerDescriptor::linear_repeat();

        // A: miss
        let _a1 = cache.get_or_create(&desc_a);
        // B: miss
        let _b1 = cache.get_or_create(&desc_b);
        // A: hit
        let _a2 = cache.get_or_create(&desc_a);
        // A: hit
        let _a3 = cache.get_or_create(&desc_a);
        // B: hit
        let _b2 = cache.get_or_create(&desc_b);

        let metrics = cache.metrics();

        assert_eq!(metrics.cache_size, 2);
        assert_eq!(metrics.total_requests(), 5);
        assert_eq!(metrics.misses, 2);
        assert_eq!(metrics.hits, 3);
        assert!((metrics.hit_rate - 0.6).abs() < 0.01);
    }
}
