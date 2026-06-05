//! Whitebox tests for SamplerCache
//!
//! T-WGPU-P2.4.2 - Sampler Cache
//!
//! These tests verify the internal implementation of the sampler caching system,
//! including cache key generation, metrics tracking, preset management, and
//! thread-safe operations.

use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use std::thread;

use wgpu::{AddressMode, CompareFunction, FilterMode, SamplerBorderColor};

use renderer_backend::resources::sampler::TrinitySamplerDescriptor;
use renderer_backend::resources::sampler_cache::{SamplerCache, SamplerCacheKey, SamplerCacheMetrics};

// ============================================================================
// Test Helpers
// ============================================================================

/// Creates a mock wgpu instance and device for testing.
/// Returns None if no adapter is available (CI without GPU).
fn create_test_device() -> Option<Arc<wgpu::Device>> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::VULKAN,
        ..Default::default()
    });

    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))?;

    let (device, _queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("test_device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::downlevel_webgl2_defaults(),
            memory_hints: Default::default(),
        },
        None,
    ))
    .ok()?;

    Some(Arc::new(device))
}

/// Computes hash for a SamplerCacheKey
fn compute_hash(key: &SamplerCacheKey) -> u64 {
    let mut hasher = DefaultHasher::new();
    key.hash(&mut hasher);
    hasher.finish()
}

// ============================================================================
// Module: cache_key_tests
// ============================================================================

mod cache_key_tests {
    use super::*;

    // ------------------------------------------------------------------------
    // from_descriptor() conversion tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_from_descriptor_default() {
        let desc = TrinitySamplerDescriptor::default();
        let key = SamplerCacheKey::from_descriptor(&desc);

        // Default is linear filtering, clamp to edge
        assert_eq!(key.address_mode_u(), 0); // ClampToEdge
        assert_eq!(key.address_mode_v(), 0);
        assert_eq!(key.address_mode_w(), 0);
        assert_eq!(key.mag_filter(), 1); // Linear
        assert_eq!(key.min_filter(), 1);
        assert_eq!(key.mipmap_filter(), 1);
        assert_eq!(key.compare(), 0); // None
        assert_eq!(key.anisotropy(), 1);
        assert_eq!(key.border_color(), 0); // None
    }

    #[test]
    fn test_from_descriptor_linear_clamp() {
        let desc = TrinitySamplerDescriptor::linear_clamp();
        let key = SamplerCacheKey::from_descriptor(&desc);

        assert_eq!(key.address_mode_u(), 0); // ClampToEdge
        assert_eq!(key.mag_filter(), 1); // Linear
        assert_eq!(key.compare(), 0); // None
    }

    #[test]
    fn test_from_descriptor_linear_repeat() {
        let desc = TrinitySamplerDescriptor::linear_repeat();
        let key = SamplerCacheKey::from_descriptor(&desc);

        assert_eq!(key.address_mode_u(), 1); // Repeat
        assert_eq!(key.mag_filter(), 1); // Linear
    }

    #[test]
    fn test_from_descriptor_nearest_clamp() {
        let desc = TrinitySamplerDescriptor::nearest_clamp();
        let key = SamplerCacheKey::from_descriptor(&desc);

        assert_eq!(key.address_mode_u(), 0); // ClampToEdge
        assert_eq!(key.mag_filter(), 0); // Nearest
    }

    #[test]
    fn test_from_descriptor_nearest_repeat() {
        let desc = TrinitySamplerDescriptor::nearest_repeat();
        let key = SamplerCacheKey::from_descriptor(&desc);

        assert_eq!(key.address_mode_u(), 1); // Repeat
        assert_eq!(key.mag_filter(), 0); // Nearest
    }

    #[test]
    fn test_from_descriptor_shadow() {
        let desc = TrinitySamplerDescriptor::shadow();
        let key = SamplerCacheKey::from_descriptor(&desc);

        assert_eq!(key.compare(), 2); // Less
        assert_eq!(key.mag_filter(), 1); // Linear
        assert_eq!(key.min_filter(), 1); // Linear
        assert_eq!(key.mipmap_filter(), 0); // Nearest
    }

    #[test]
    fn test_from_descriptor_trilinear() {
        let desc = TrinitySamplerDescriptor::trilinear();
        let key = SamplerCacheKey::from_descriptor(&desc);

        assert_eq!(key.mag_filter(), 1); // Linear
        assert_eq!(key.min_filter(), 1); // Linear
        assert_eq!(key.mipmap_filter(), 1); // Linear
    }

    #[test]
    fn test_from_descriptor_with_anisotropy() {
        let desc = TrinitySamplerDescriptor::new().anisotropy(8);
        let key = SamplerCacheKey::from_descriptor(&desc);

        assert_eq!(key.anisotropy(), 8);
    }

    #[test]
    fn test_from_descriptor_with_max_anisotropy() {
        let desc = TrinitySamplerDescriptor::new().anisotropy(16);
        let key = SamplerCacheKey::from_descriptor(&desc);

        assert_eq!(key.anisotropy(), 16);
    }

    #[test]
    fn test_from_descriptor_with_custom_lod() {
        let desc = TrinitySamplerDescriptor::new().lod_clamp(2.5, 10.0);
        let key = SamplerCacheKey::from_descriptor(&desc);

        assert_eq!(key.lod_min_clamp_bits(), 2.5_f32.to_bits());
        assert_eq!(key.lod_max_clamp_bits(), 10.0_f32.to_bits());
    }

    // ------------------------------------------------------------------------
    // Hash equality for same descriptor
    // ------------------------------------------------------------------------

    #[test]
    fn test_hash_equality_same_descriptor() {
        let desc1 = TrinitySamplerDescriptor::linear_clamp();
        let desc2 = TrinitySamplerDescriptor::linear_clamp();

        let key1 = SamplerCacheKey::from_descriptor(&desc1);
        let key2 = SamplerCacheKey::from_descriptor(&desc2);

        assert_eq!(key1, key2);
        assert_eq!(compute_hash(&key1), compute_hash(&key2));
    }

    #[test]
    fn test_hash_equality_constructed_identically() {
        let desc1 = TrinitySamplerDescriptor::new()
            .filter(FilterMode::Linear)
            .address_mode(AddressMode::Repeat)
            .anisotropy(4);

        let desc2 = TrinitySamplerDescriptor::new()
            .filter(FilterMode::Linear)
            .address_mode(AddressMode::Repeat)
            .anisotropy(4);

        let key1 = SamplerCacheKey::from_descriptor(&desc1);
        let key2 = SamplerCacheKey::from_descriptor(&desc2);

        assert_eq!(key1, key2);
    }

    #[test]
    fn test_hash_consistency_multiple_calls() {
        let desc = TrinitySamplerDescriptor::trilinear().anisotropy(8);
        let key = SamplerCacheKey::from_descriptor(&desc);

        let hash1 = compute_hash(&key);
        let hash2 = compute_hash(&key);
        let hash3 = compute_hash(&key);

        assert_eq!(hash1, hash2);
        assert_eq!(hash2, hash3);
    }

    // ------------------------------------------------------------------------
    // Hash inequality for different descriptors
    // ------------------------------------------------------------------------

    #[test]
    fn test_hash_inequality_different_filter() {
        let desc1 = TrinitySamplerDescriptor::linear_clamp();
        let desc2 = TrinitySamplerDescriptor::nearest_clamp();

        let key1 = SamplerCacheKey::from_descriptor(&desc1);
        let key2 = SamplerCacheKey::from_descriptor(&desc2);

        assert_ne!(key1, key2);
    }

    #[test]
    fn test_hash_inequality_different_address_mode() {
        let desc1 = TrinitySamplerDescriptor::linear_clamp();
        let desc2 = TrinitySamplerDescriptor::linear_repeat();

        let key1 = SamplerCacheKey::from_descriptor(&desc1);
        let key2 = SamplerCacheKey::from_descriptor(&desc2);

        assert_ne!(key1, key2);
    }

    #[test]
    fn test_hash_inequality_different_anisotropy() {
        let desc1 = TrinitySamplerDescriptor::new().anisotropy(1);
        let desc2 = TrinitySamplerDescriptor::new().anisotropy(8);

        let key1 = SamplerCacheKey::from_descriptor(&desc1);
        let key2 = SamplerCacheKey::from_descriptor(&desc2);

        assert_ne!(key1, key2);
    }

    #[test]
    fn test_hash_inequality_different_compare() {
        let desc1 = TrinitySamplerDescriptor::new();
        let desc2 = TrinitySamplerDescriptor::shadow();

        let key1 = SamplerCacheKey::from_descriptor(&desc1);
        let key2 = SamplerCacheKey::from_descriptor(&desc2);

        assert_ne!(key1, key2);
    }

    #[test]
    fn test_hash_inequality_different_lod_min() {
        let desc1 = TrinitySamplerDescriptor::new().lod_clamp(0.0, 32.0);
        let desc2 = TrinitySamplerDescriptor::new().lod_clamp(1.0, 32.0);

        let key1 = SamplerCacheKey::from_descriptor(&desc1);
        let key2 = SamplerCacheKey::from_descriptor(&desc2);

        assert_ne!(key1, key2);
    }

    #[test]
    fn test_hash_inequality_different_lod_max() {
        let desc1 = TrinitySamplerDescriptor::new().lod_clamp(0.0, 16.0);
        let desc2 = TrinitySamplerDescriptor::new().lod_clamp(0.0, 32.0);

        let key1 = SamplerCacheKey::from_descriptor(&desc1);
        let key2 = SamplerCacheKey::from_descriptor(&desc2);

        assert_ne!(key1, key2);
    }

    #[test]
    fn test_hash_inequality_different_border_color() {
        let desc1 = TrinitySamplerDescriptor::new()
            .address_mode(AddressMode::ClampToBorder)
            .border_color(SamplerBorderColor::OpaqueBlack);
        let desc2 = TrinitySamplerDescriptor::new()
            .address_mode(AddressMode::ClampToBorder)
            .border_color(SamplerBorderColor::OpaqueWhite);

        let key1 = SamplerCacheKey::from_descriptor(&desc1);
        let key2 = SamplerCacheKey::from_descriptor(&desc2);

        assert_ne!(key1, key2);
    }

    // ------------------------------------------------------------------------
    // All field accessors
    // ------------------------------------------------------------------------

    #[test]
    fn test_all_field_accessors() {
        let desc = TrinitySamplerDescriptor::new()
            .address_mode_uvw(
                AddressMode::Repeat,
                AddressMode::MirrorRepeat,
                AddressMode::ClampToBorder,
            )
            .filter_separate(FilterMode::Linear, FilterMode::Nearest, FilterMode::Linear)
            .compare(CompareFunction::LessEqual)
            .anisotropy(4)
            .border_color(SamplerBorderColor::TransparentBlack)
            .lod_clamp(1.0, 16.0);

        let key = SamplerCacheKey::from_descriptor(&desc);

        assert_eq!(key.address_mode_u(), 1); // Repeat
        assert_eq!(key.address_mode_v(), 2); // MirrorRepeat
        assert_eq!(key.address_mode_w(), 3); // ClampToBorder
        assert_eq!(key.mag_filter(), 1); // Linear
        assert_eq!(key.min_filter(), 0); // Nearest
        assert_eq!(key.mipmap_filter(), 1); // Linear
        assert_eq!(key.compare(), 4); // LessEqual
        assert_eq!(key.anisotropy(), 4);
        assert_eq!(key.border_color(), 1); // TransparentBlack
        assert_eq!(key.lod_min_clamp_bits(), 1.0_f32.to_bits());
        assert_eq!(key.lod_max_clamp_bits(), 16.0_f32.to_bits());
    }

    // ------------------------------------------------------------------------
    // Clone and Debug traits
    // ------------------------------------------------------------------------

    #[test]
    fn test_clone_trait() {
        let desc = TrinitySamplerDescriptor::shadow();
        let key1 = SamplerCacheKey::from_descriptor(&desc);
        let key2 = key1;

        assert_eq!(key1, key2);
        assert_eq!(key1.compare(), key2.compare());
    }

    #[test]
    fn test_debug_trait() {
        let desc = TrinitySamplerDescriptor::new();
        let key = SamplerCacheKey::from_descriptor(&desc);
        let debug_str = format!("{:?}", key);

        assert!(debug_str.contains("SamplerCacheKey"));
        assert!(debug_str.contains("address_mode_u"));
        assert!(debug_str.contains("mag_filter"));
    }

    // ------------------------------------------------------------------------
    // All AddressMode variants mapped correctly
    // ------------------------------------------------------------------------

    #[test]
    fn test_address_mode_clamp_to_edge() {
        let desc = TrinitySamplerDescriptor::new().address_mode(AddressMode::ClampToEdge);
        let key = SamplerCacheKey::from_descriptor(&desc);
        assert_eq!(key.address_mode_u(), 0);
    }

    #[test]
    fn test_address_mode_repeat() {
        let desc = TrinitySamplerDescriptor::new().address_mode(AddressMode::Repeat);
        let key = SamplerCacheKey::from_descriptor(&desc);
        assert_eq!(key.address_mode_u(), 1);
    }

    #[test]
    fn test_address_mode_mirror_repeat() {
        let desc = TrinitySamplerDescriptor::new().address_mode(AddressMode::MirrorRepeat);
        let key = SamplerCacheKey::from_descriptor(&desc);
        assert_eq!(key.address_mode_u(), 2);
    }

    #[test]
    fn test_address_mode_clamp_to_border() {
        let desc = TrinitySamplerDescriptor::new().address_mode(AddressMode::ClampToBorder);
        let key = SamplerCacheKey::from_descriptor(&desc);
        assert_eq!(key.address_mode_u(), 3);
    }

    // ------------------------------------------------------------------------
    // All FilterMode variants mapped correctly
    // ------------------------------------------------------------------------

    #[test]
    fn test_filter_mode_nearest() {
        let desc = TrinitySamplerDescriptor::new().filter(FilterMode::Nearest);
        let key = SamplerCacheKey::from_descriptor(&desc);
        assert_eq!(key.mag_filter(), 0);
        assert_eq!(key.min_filter(), 0);
        assert_eq!(key.mipmap_filter(), 0);
    }

    #[test]
    fn test_filter_mode_linear() {
        let desc = TrinitySamplerDescriptor::new().filter(FilterMode::Linear);
        let key = SamplerCacheKey::from_descriptor(&desc);
        assert_eq!(key.mag_filter(), 1);
        assert_eq!(key.min_filter(), 1);
        assert_eq!(key.mipmap_filter(), 1);
    }

    // ------------------------------------------------------------------------
    // All CompareFunction variants mapped correctly
    // ------------------------------------------------------------------------

    #[test]
    fn test_compare_function_none() {
        let desc = TrinitySamplerDescriptor::new();
        let key = SamplerCacheKey::from_descriptor(&desc);
        assert_eq!(key.compare(), 0);
    }

    #[test]
    fn test_compare_function_never() {
        let desc = TrinitySamplerDescriptor::new().compare(CompareFunction::Never);
        let key = SamplerCacheKey::from_descriptor(&desc);
        assert_eq!(key.compare(), 1);
    }

    #[test]
    fn test_compare_function_less() {
        let desc = TrinitySamplerDescriptor::new().compare(CompareFunction::Less);
        let key = SamplerCacheKey::from_descriptor(&desc);
        assert_eq!(key.compare(), 2);
    }

    #[test]
    fn test_compare_function_equal() {
        let desc = TrinitySamplerDescriptor::new().compare(CompareFunction::Equal);
        let key = SamplerCacheKey::from_descriptor(&desc);
        assert_eq!(key.compare(), 3);
    }

    #[test]
    fn test_compare_function_less_equal() {
        let desc = TrinitySamplerDescriptor::new().compare(CompareFunction::LessEqual);
        let key = SamplerCacheKey::from_descriptor(&desc);
        assert_eq!(key.compare(), 4);
    }

    #[test]
    fn test_compare_function_greater() {
        let desc = TrinitySamplerDescriptor::new().compare(CompareFunction::Greater);
        let key = SamplerCacheKey::from_descriptor(&desc);
        assert_eq!(key.compare(), 5);
    }

    #[test]
    fn test_compare_function_not_equal() {
        let desc = TrinitySamplerDescriptor::new().compare(CompareFunction::NotEqual);
        let key = SamplerCacheKey::from_descriptor(&desc);
        assert_eq!(key.compare(), 6);
    }

    #[test]
    fn test_compare_function_greater_equal() {
        let desc = TrinitySamplerDescriptor::new().compare(CompareFunction::GreaterEqual);
        let key = SamplerCacheKey::from_descriptor(&desc);
        assert_eq!(key.compare(), 7);
    }

    #[test]
    fn test_compare_function_always() {
        let desc = TrinitySamplerDescriptor::new().compare(CompareFunction::Always);
        let key = SamplerCacheKey::from_descriptor(&desc);
        assert_eq!(key.compare(), 8);
    }

    // ------------------------------------------------------------------------
    // All SamplerBorderColor variants mapped correctly
    // ------------------------------------------------------------------------

    #[test]
    fn test_border_color_none() {
        let desc = TrinitySamplerDescriptor::new();
        let key = SamplerCacheKey::from_descriptor(&desc);
        assert_eq!(key.border_color(), 0);
    }

    #[test]
    fn test_border_color_transparent_black() {
        let desc = TrinitySamplerDescriptor::new()
            .address_mode(AddressMode::ClampToBorder)
            .border_color(SamplerBorderColor::TransparentBlack);
        let key = SamplerCacheKey::from_descriptor(&desc);
        assert_eq!(key.border_color(), 1);
    }

    #[test]
    fn test_border_color_opaque_black() {
        let desc = TrinitySamplerDescriptor::new()
            .address_mode(AddressMode::ClampToBorder)
            .border_color(SamplerBorderColor::OpaqueBlack);
        let key = SamplerCacheKey::from_descriptor(&desc);
        assert_eq!(key.border_color(), 2);
    }

    #[test]
    fn test_border_color_opaque_white() {
        let desc = TrinitySamplerDescriptor::new()
            .address_mode(AddressMode::ClampToBorder)
            .border_color(SamplerBorderColor::OpaqueWhite);
        let key = SamplerCacheKey::from_descriptor(&desc);
        assert_eq!(key.border_color(), 3);
    }

    #[test]
    fn test_border_color_zero() {
        let desc = TrinitySamplerDescriptor::new()
            .address_mode(AddressMode::ClampToBorder)
            .border_color(SamplerBorderColor::Zero);
        let key = SamplerCacheKey::from_descriptor(&desc);
        assert_eq!(key.border_color(), 4);
    }

    // ------------------------------------------------------------------------
    // LOD values encoded via to_bits()
    // ------------------------------------------------------------------------

    #[test]
    fn test_lod_zero() {
        let desc = TrinitySamplerDescriptor::new().lod_clamp(0.0, 0.0);
        let key = SamplerCacheKey::from_descriptor(&desc);

        assert_eq!(key.lod_min_clamp_bits(), 0.0_f32.to_bits());
        assert_eq!(key.lod_max_clamp_bits(), 0.0_f32.to_bits());
    }

    #[test]
    fn test_lod_default() {
        let desc = TrinitySamplerDescriptor::new();
        let key = SamplerCacheKey::from_descriptor(&desc);

        assert_eq!(key.lod_min_clamp_bits(), 0.0_f32.to_bits());
        assert_eq!(key.lod_max_clamp_bits(), 32.0_f32.to_bits());
    }

    #[test]
    fn test_lod_custom_values() {
        let desc = TrinitySamplerDescriptor::new().lod_clamp(0.5, 12.5);
        let key = SamplerCacheKey::from_descriptor(&desc);

        assert_eq!(key.lod_min_clamp_bits(), 0.5_f32.to_bits());
        assert_eq!(key.lod_max_clamp_bits(), 12.5_f32.to_bits());
    }

    #[test]
    fn test_lod_fractional_precision() {
        let desc = TrinitySamplerDescriptor::new().lod_clamp(1.123456, 7.654321);
        let key = SamplerCacheKey::from_descriptor(&desc);

        // Verify bit-exact encoding
        assert_eq!(key.lod_min_clamp_bits(), 1.123456_f32.to_bits());
        assert_eq!(key.lod_max_clamp_bits(), 7.654321_f32.to_bits());
    }

    // ------------------------------------------------------------------------
    // HashMap behavior
    // ------------------------------------------------------------------------

    #[test]
    fn test_key_as_hashmap_key() {
        let mut map: HashMap<SamplerCacheKey, i32> = HashMap::new();

        let key1 = SamplerCacheKey::from_descriptor(&TrinitySamplerDescriptor::linear_clamp());
        let key2 = SamplerCacheKey::from_descriptor(&TrinitySamplerDescriptor::linear_repeat());
        let key3 = SamplerCacheKey::from_descriptor(&TrinitySamplerDescriptor::nearest_clamp());

        map.insert(key1, 1);
        map.insert(key2, 2);
        map.insert(key3, 3);

        assert_eq!(map.len(), 3);
        assert_eq!(map.get(&key1), Some(&1));
        assert_eq!(map.get(&key2), Some(&2));
        assert_eq!(map.get(&key3), Some(&3));
    }

    #[test]
    fn test_key_duplicate_insert_overwrites() {
        let mut map: HashMap<SamplerCacheKey, i32> = HashMap::new();

        let key1 = SamplerCacheKey::from_descriptor(&TrinitySamplerDescriptor::linear_clamp());
        let key2 = SamplerCacheKey::from_descriptor(&TrinitySamplerDescriptor::linear_clamp());

        map.insert(key1, 1);
        map.insert(key2, 2);

        assert_eq!(map.len(), 1);
        assert_eq!(map.get(&key1), Some(&2));
    }

    #[test]
    fn test_all_presets_unique_keys() {
        let linear_clamp = SamplerCacheKey::from_descriptor(&TrinitySamplerDescriptor::linear_clamp());
        let linear_repeat = SamplerCacheKey::from_descriptor(&TrinitySamplerDescriptor::linear_repeat());
        let nearest_clamp = SamplerCacheKey::from_descriptor(&TrinitySamplerDescriptor::nearest_clamp());
        let nearest_repeat = SamplerCacheKey::from_descriptor(&TrinitySamplerDescriptor::nearest_repeat());
        let shadow = SamplerCacheKey::from_descriptor(&TrinitySamplerDescriptor::shadow());
        let trilinear = SamplerCacheKey::from_descriptor(&TrinitySamplerDescriptor::trilinear());

        let keys = [
            linear_clamp,
            linear_repeat,
            nearest_clamp,
            nearest_repeat,
            shadow,
            trilinear,
        ];

        // All keys should be different (except linear_clamp == trilinear since they're same config)
        for i in 0..keys.len() {
            for j in (i + 1)..keys.len() {
                // linear_clamp and trilinear may be equal, skip that comparison
                if (i == 0 && j == 5) || (i == 5 && j == 0) {
                    continue;
                }
                assert_ne!(keys[i], keys[j], "Keys {} and {} should be different", i, j);
            }
        }
    }
}

// ============================================================================
// Module: metrics_tests
// ============================================================================

mod metrics_tests {
    use super::*;

    // ------------------------------------------------------------------------
    // Default values (all zeros)
    // ------------------------------------------------------------------------

    #[test]
    fn test_default_values() {
        let metrics = SamplerCacheMetrics::default();

        assert_eq!(metrics.cache_size, 0);
        assert_eq!(metrics.hits, 0);
        assert_eq!(metrics.misses, 0);
        assert_eq!(metrics.hit_rate, 0.0);
    }

    // ------------------------------------------------------------------------
    // hit_rate() calculation
    // ------------------------------------------------------------------------

    #[test]
    fn test_hit_rate_zero_requests() {
        let metrics = SamplerCacheMetrics::new(0, 0, 0);
        assert_eq!(metrics.hit_rate, 0.0);
    }

    #[test]
    fn test_hit_rate_all_hits() {
        let metrics = SamplerCacheMetrics::new(5, 100, 0);
        assert!((metrics.hit_rate - 1.0).abs() < 0.0001);
    }

    #[test]
    fn test_hit_rate_all_misses() {
        let metrics = SamplerCacheMetrics::new(5, 0, 100);
        assert_eq!(metrics.hit_rate, 0.0);
    }

    #[test]
    fn test_hit_rate_50_percent() {
        let metrics = SamplerCacheMetrics::new(5, 50, 50);
        assert!((metrics.hit_rate - 0.5).abs() < 0.0001);
    }

    #[test]
    fn test_hit_rate_80_percent() {
        let metrics = SamplerCacheMetrics::new(5, 80, 20);
        assert!((metrics.hit_rate - 0.8).abs() < 0.0001);
    }

    #[test]
    fn test_hit_rate_25_percent() {
        let metrics = SamplerCacheMetrics::new(3, 25, 75);
        assert!((metrics.hit_rate - 0.25).abs() < 0.0001);
    }

    // ------------------------------------------------------------------------
    // total_requests() = hits + misses
    // ------------------------------------------------------------------------

    #[test]
    fn test_total_requests_zero() {
        let metrics = SamplerCacheMetrics::new(0, 0, 0);
        assert_eq!(metrics.total_requests(), 0);
    }

    #[test]
    fn test_total_requests_only_hits() {
        let metrics = SamplerCacheMetrics::new(5, 100, 0);
        assert_eq!(metrics.total_requests(), 100);
    }

    #[test]
    fn test_total_requests_only_misses() {
        let metrics = SamplerCacheMetrics::new(5, 0, 50);
        assert_eq!(metrics.total_requests(), 50);
    }

    #[test]
    fn test_total_requests_mixed() {
        let metrics = SamplerCacheMetrics::new(5, 75, 25);
        assert_eq!(metrics.total_requests(), 100);
    }

    #[test]
    fn test_total_requests_large_values() {
        let metrics = SamplerCacheMetrics::new(100, 1_000_000, 500_000);
        assert_eq!(metrics.total_requests(), 1_500_000);
    }

    // ------------------------------------------------------------------------
    // is_empty() when no requests
    // ------------------------------------------------------------------------

    #[test]
    fn test_is_empty_true() {
        let metrics = SamplerCacheMetrics::new(0, 0, 0);
        assert!(metrics.is_empty());
    }

    #[test]
    fn test_is_empty_false_with_size() {
        let metrics = SamplerCacheMetrics::new(1, 0, 0);
        assert!(!metrics.is_empty());
    }

    #[test]
    fn test_is_empty_false_with_misses() {
        let metrics = SamplerCacheMetrics::new(1, 0, 1);
        assert!(!metrics.is_empty());
    }

    // ------------------------------------------------------------------------
    // hit_rate_percent() formatting
    // ------------------------------------------------------------------------

    #[test]
    fn test_hit_rate_percent_zero() {
        let metrics = SamplerCacheMetrics::new(0, 0, 0);
        assert_eq!(metrics.hit_rate_percent(), 0.0);
    }

    #[test]
    fn test_hit_rate_percent_100() {
        let metrics = SamplerCacheMetrics::new(5, 100, 0);
        assert!((metrics.hit_rate_percent() - 100.0).abs() < 0.01);
    }

    #[test]
    fn test_hit_rate_percent_75() {
        let metrics = SamplerCacheMetrics::new(5, 75, 25);
        assert!((metrics.hit_rate_percent() - 75.0).abs() < 0.01);
    }

    #[test]
    fn test_hit_rate_percent_33() {
        let metrics = SamplerCacheMetrics::new(3, 1, 2);
        // 1/3 = 33.333...%
        assert!((metrics.hit_rate_percent() - 33.333).abs() < 0.01);
    }

    // ------------------------------------------------------------------------
    // Clone and Debug traits
    // ------------------------------------------------------------------------

    #[test]
    fn test_metrics_clone() {
        let original = SamplerCacheMetrics::new(10, 80, 20);
        let cloned = original.clone();

        assert_eq!(cloned.cache_size, original.cache_size);
        assert_eq!(cloned.hits, original.hits);
        assert_eq!(cloned.misses, original.misses);
        assert_eq!(cloned.hit_rate, original.hit_rate);
    }

    #[test]
    fn test_metrics_debug() {
        let metrics = SamplerCacheMetrics::new(5, 10, 2);
        let debug_str = format!("{:?}", metrics);

        assert!(debug_str.contains("SamplerCacheMetrics"));
        assert!(debug_str.contains("cache_size"));
        assert!(debug_str.contains("hits"));
        assert!(debug_str.contains("misses"));
        assert!(debug_str.contains("hit_rate"));
    }

    #[test]
    fn test_metrics_new_calculates_hit_rate() {
        let metrics = SamplerCacheMetrics::new(5, 60, 40);
        // 60/100 = 0.6
        assert!((metrics.hit_rate - 0.6).abs() < 0.0001);
    }
}

// ============================================================================
// Module: construction_tests
// ============================================================================

mod construction_tests {
    use super::*;

    // ------------------------------------------------------------------------
    // new() creates all 5 presets
    // ------------------------------------------------------------------------

    #[test]
    fn test_new_creates_five_presets() {
        let Some(device) = create_test_device() else {
            eprintln!("Skipping test - no GPU adapter available");
            return;
        };

        let cache = SamplerCache::new(device);

        // Presets don't count toward cache size
        assert_eq!(cache.len(), 0);
        assert_eq!(cache.preset_count(), 5);
        assert_eq!(cache.total_count(), 5);
    }

    #[test]
    fn test_new_cache_is_empty() {
        let Some(device) = create_test_device() else {
            eprintln!("Skipping test - no GPU adapter available");
            return;
        };

        let cache = SamplerCache::new(device);
        assert!(cache.is_empty());
    }

    // ------------------------------------------------------------------------
    // Presets are distinct samplers
    // ------------------------------------------------------------------------

    #[test]
    fn test_presets_are_distinct() {
        let Some(device) = create_test_device() else {
            eprintln!("Skipping test - no GPU adapter available");
            return;
        };

        let cache = SamplerCache::new(device);

        let linear_clamp = cache.linear_clamp();
        let linear_repeat = cache.linear_repeat();
        let point_clamp = cache.point_clamp();
        let point_repeat = cache.point_repeat();
        let shadow = cache.shadow();

        // Each preset should return a different Arc (different sampler)
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

    // ------------------------------------------------------------------------
    // Device is stored correctly
    // ------------------------------------------------------------------------

    #[test]
    fn test_device_stored_correctly() {
        let Some(device) = create_test_device() else {
            eprintln!("Skipping test - no GPU adapter available");
            return;
        };

        let device_clone = Arc::clone(&device);
        let cache = SamplerCache::new(device);

        // The device() method should return a reference to the same device
        assert!(Arc::ptr_eq(cache.device(), &device_clone));
    }

    #[test]
    fn test_initial_metrics() {
        let Some(device) = create_test_device() else {
            eprintln!("Skipping test - no GPU adapter available");
            return;
        };

        let cache = SamplerCache::new(device);
        let metrics = cache.metrics();

        assert_eq!(metrics.cache_size, 0);
        assert_eq!(metrics.hits, 0);
        assert_eq!(metrics.misses, 0);
        assert_eq!(metrics.hit_rate, 0.0);
    }
}

// ============================================================================
// Module: preset_tests
// ============================================================================

mod preset_tests {
    use super::*;

    // ------------------------------------------------------------------------
    // Preset method tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_linear_clamp_returns_sampler() {
        let Some(device) = create_test_device() else {
            eprintln!("Skipping test - no GPU adapter available");
            return;
        };

        let cache = SamplerCache::new(device);
        let sampler = cache.linear_clamp();

        // Should return valid Arc
        assert!(Arc::strong_count(&sampler) >= 1);
    }

    #[test]
    fn test_linear_clamp_consistent() {
        let Some(device) = create_test_device() else {
            eprintln!("Skipping test - no GPU adapter available");
            return;
        };

        let cache = SamplerCache::new(device);
        let s1 = cache.linear_clamp();
        let s2 = cache.linear_clamp();

        // Should return same Arc
        assert!(Arc::ptr_eq(&s1, &s2));
    }

    #[test]
    fn test_linear_repeat_returns_sampler() {
        let Some(device) = create_test_device() else {
            eprintln!("Skipping test - no GPU adapter available");
            return;
        };

        let cache = SamplerCache::new(device);
        let sampler = cache.linear_repeat();

        assert!(Arc::strong_count(&sampler) >= 1);
    }

    #[test]
    fn test_linear_repeat_consistent() {
        let Some(device) = create_test_device() else {
            eprintln!("Skipping test - no GPU adapter available");
            return;
        };

        let cache = SamplerCache::new(device);
        let s1 = cache.linear_repeat();
        let s2 = cache.linear_repeat();

        assert!(Arc::ptr_eq(&s1, &s2));
    }

    #[test]
    fn test_point_clamp_returns_sampler() {
        let Some(device) = create_test_device() else {
            eprintln!("Skipping test - no GPU adapter available");
            return;
        };

        let cache = SamplerCache::new(device);
        let sampler = cache.point_clamp();

        assert!(Arc::strong_count(&sampler) >= 1);
    }

    #[test]
    fn test_point_clamp_consistent() {
        let Some(device) = create_test_device() else {
            eprintln!("Skipping test - no GPU adapter available");
            return;
        };

        let cache = SamplerCache::new(device);
        let s1 = cache.point_clamp();
        let s2 = cache.point_clamp();

        assert!(Arc::ptr_eq(&s1, &s2));
    }

    #[test]
    fn test_point_repeat_returns_sampler() {
        let Some(device) = create_test_device() else {
            eprintln!("Skipping test - no GPU adapter available");
            return;
        };

        let cache = SamplerCache::new(device);
        let sampler = cache.point_repeat();

        assert!(Arc::strong_count(&sampler) >= 1);
    }

    #[test]
    fn test_point_repeat_consistent() {
        let Some(device) = create_test_device() else {
            eprintln!("Skipping test - no GPU adapter available");
            return;
        };

        let cache = SamplerCache::new(device);
        let s1 = cache.point_repeat();
        let s2 = cache.point_repeat();

        assert!(Arc::ptr_eq(&s1, &s2));
    }

    #[test]
    fn test_shadow_returns_sampler() {
        let Some(device) = create_test_device() else {
            eprintln!("Skipping test - no GPU adapter available");
            return;
        };

        let cache = SamplerCache::new(device);
        let sampler = cache.shadow();

        assert!(Arc::strong_count(&sampler) >= 1);
    }

    #[test]
    fn test_shadow_consistent() {
        let Some(device) = create_test_device() else {
            eprintln!("Skipping test - no GPU adapter available");
            return;
        };

        let cache = SamplerCache::new(device);
        let s1 = cache.shadow();
        let s2 = cache.shadow();

        assert!(Arc::ptr_eq(&s1, &s2));
    }

    // ------------------------------------------------------------------------
    // Shorthand methods
    // ------------------------------------------------------------------------

    #[test]
    fn test_get_linear_equals_linear_clamp() {
        let Some(device) = create_test_device() else {
            eprintln!("Skipping test - no GPU adapter available");
            return;
        };

        let cache = SamplerCache::new(device);
        let linear = cache.get_linear();
        let linear_clamp = cache.linear_clamp();

        assert!(Arc::ptr_eq(&linear, &linear_clamp));
    }

    #[test]
    fn test_get_point_equals_point_clamp() {
        let Some(device) = create_test_device() else {
            eprintln!("Skipping test - no GPU adapter available");
            return;
        };

        let cache = SamplerCache::new(device);
        let point = cache.get_point();
        let point_clamp = cache.point_clamp();

        assert!(Arc::ptr_eq(&point, &point_clamp));
    }

    #[test]
    fn test_preset_count_is_five() {
        let Some(device) = create_test_device() else {
            eprintln!("Skipping test - no GPU adapter available");
            return;
        };

        let cache = SamplerCache::new(device);
        assert_eq!(cache.preset_count(), 5);
    }

    #[test]
    fn test_presets_do_not_affect_cache_size() {
        let Some(device) = create_test_device() else {
            eprintln!("Skipping test - no GPU adapter available");
            return;
        };

        let cache = SamplerCache::new(device);

        // Access all presets
        let _ = cache.linear_clamp();
        let _ = cache.linear_repeat();
        let _ = cache.point_clamp();
        let _ = cache.point_repeat();
        let _ = cache.shadow();
        let _ = cache.get_linear();
        let _ = cache.get_point();

        // Cache should still be empty
        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
    }

    #[test]
    fn test_presets_do_not_affect_metrics() {
        let Some(device) = create_test_device() else {
            eprintln!("Skipping test - no GPU adapter available");
            return;
        };

        let cache = SamplerCache::new(device);

        // Access all presets multiple times
        for _ in 0..10 {
            let _ = cache.linear_clamp();
            let _ = cache.linear_repeat();
            let _ = cache.point_clamp();
            let _ = cache.point_repeat();
            let _ = cache.shadow();
        }

        let metrics = cache.metrics();
        assert_eq!(metrics.hits, 0);
        assert_eq!(metrics.misses, 0);
    }
}

// ============================================================================
// Module: get_or_create_tests
// ============================================================================

mod get_or_create_tests {
    use super::*;

    // ------------------------------------------------------------------------
    // Creates new sampler on miss
    // ------------------------------------------------------------------------

    #[test]
    fn test_creates_new_sampler_on_miss() {
        let Some(device) = create_test_device() else {
            eprintln!("Skipping test - no GPU adapter available");
            return;
        };

        let cache = SamplerCache::new(device);
        assert_eq!(cache.len(), 0);

        let desc = TrinitySamplerDescriptor::new().anisotropy(4);
        let _ = cache.get_or_create(&desc);

        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn test_miss_increments_metrics() {
        let Some(device) = create_test_device() else {
            eprintln!("Skipping test - no GPU adapter available");
            return;
        };

        let cache = SamplerCache::new(device);

        let desc = TrinitySamplerDescriptor::new().anisotropy(4);
        let _ = cache.get_or_create(&desc);

        let metrics = cache.metrics();
        assert_eq!(metrics.misses, 1);
        assert_eq!(metrics.hits, 0);
    }

    // ------------------------------------------------------------------------
    // Returns cached sampler on hit
    // ------------------------------------------------------------------------

    #[test]
    fn test_returns_cached_sampler_on_hit() {
        let Some(device) = create_test_device() else {
            eprintln!("Skipping test - no GPU adapter available");
            return;
        };

        let cache = SamplerCache::new(device);

        let desc = TrinitySamplerDescriptor::new().anisotropy(4);
        let s1 = cache.get_or_create(&desc);
        let s2 = cache.get_or_create(&desc);

        assert!(Arc::ptr_eq(&s1, &s2));
    }

    #[test]
    fn test_hit_increments_metrics() {
        let Some(device) = create_test_device() else {
            eprintln!("Skipping test - no GPU adapter available");
            return;
        };

        let cache = SamplerCache::new(device);

        let desc = TrinitySamplerDescriptor::new().anisotropy(4);
        let _ = cache.get_or_create(&desc);
        let _ = cache.get_or_create(&desc);
        let _ = cache.get_or_create(&desc);

        let metrics = cache.metrics();
        assert_eq!(metrics.misses, 1);
        assert_eq!(metrics.hits, 2);
    }

    // ------------------------------------------------------------------------
    // Same Arc returned for same descriptor
    // ------------------------------------------------------------------------

    #[test]
    fn test_same_arc_for_same_descriptor() {
        let Some(device) = create_test_device() else {
            eprintln!("Skipping test - no GPU adapter available");
            return;
        };

        let cache = SamplerCache::new(device);

        let desc1 = TrinitySamplerDescriptor::new()
            .filter(FilterMode::Linear)
            .address_mode(AddressMode::Repeat)
            .anisotropy(8);

        let desc2 = TrinitySamplerDescriptor::new()
            .filter(FilterMode::Linear)
            .address_mode(AddressMode::Repeat)
            .anisotropy(8);

        let s1 = cache.get_or_create(&desc1);
        let s2 = cache.get_or_create(&desc2);

        assert!(Arc::ptr_eq(&s1, &s2));
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn test_different_arc_for_different_descriptor() {
        let Some(device) = create_test_device() else {
            eprintln!("Skipping test - no GPU adapter available");
            return;
        };

        let cache = SamplerCache::new(device);

        let desc1 = TrinitySamplerDescriptor::new().anisotropy(4);
        let desc2 = TrinitySamplerDescriptor::new().anisotropy(8);

        let s1 = cache.get_or_create(&desc1);
        let s2 = cache.get_or_create(&desc2);

        assert!(!Arc::ptr_eq(&s1, &s2));
        assert_eq!(cache.len(), 2);
    }

    // ------------------------------------------------------------------------
    // Metrics updated correctly
    // ------------------------------------------------------------------------

    #[test]
    fn test_metrics_accuracy() {
        let Some(device) = create_test_device() else {
            eprintln!("Skipping test - no GPU adapter available");
            return;
        };

        let cache = SamplerCache::new(device);

        // Create 3 unique samplers
        for i in 1..=3 {
            let desc = TrinitySamplerDescriptor::new().anisotropy(i);
            let _ = cache.get_or_create(&desc);
        }

        // Access first one again (hit)
        let desc = TrinitySamplerDescriptor::new().anisotropy(1);
        let _ = cache.get_or_create(&desc);

        let metrics = cache.metrics();
        assert_eq!(metrics.cache_size, 3);
        assert_eq!(metrics.misses, 3);
        assert_eq!(metrics.hits, 1);
        assert_eq!(metrics.total_requests(), 4);
        // hit_rate = 1/4 = 0.25
        assert!((metrics.hit_rate - 0.25).abs() < 0.0001);
    }

    #[test]
    fn test_hit_rate_increases_with_hits() {
        let Some(device) = create_test_device() else {
            eprintln!("Skipping test - no GPU adapter available");
            return;
        };

        let cache = SamplerCache::new(device);

        let desc = TrinitySamplerDescriptor::new().anisotropy(4);

        // First call is a miss
        let _ = cache.get_or_create(&desc);
        let m1 = cache.metrics();
        assert_eq!(m1.hit_rate, 0.0);

        // Second call is a hit
        let _ = cache.get_or_create(&desc);
        let m2 = cache.metrics();
        // 1 hit / 2 total = 0.5
        assert!((m2.hit_rate - 0.5).abs() < 0.0001);

        // More hits
        let _ = cache.get_or_create(&desc);
        let _ = cache.get_or_create(&desc);
        let m3 = cache.metrics();
        // 3 hits / 4 total = 0.75
        assert!((m3.hit_rate - 0.75).abs() < 0.0001);
    }
}

// ============================================================================
// Module: cache_management_tests
// ============================================================================

mod cache_management_tests {
    use super::*;

    // ------------------------------------------------------------------------
    // len() returns correct count
    // ------------------------------------------------------------------------

    #[test]
    fn test_len_empty() {
        let Some(device) = create_test_device() else {
            eprintln!("Skipping test - no GPU adapter available");
            return;
        };

        let cache = SamplerCache::new(device);
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn test_len_after_insertions() {
        let Some(device) = create_test_device() else {
            eprintln!("Skipping test - no GPU adapter available");
            return;
        };

        let cache = SamplerCache::new(device);

        for i in 1..=5 {
            let desc = TrinitySamplerDescriptor::new().anisotropy(i);
            let _ = cache.get_or_create(&desc);
            assert_eq!(cache.len(), i as usize);
        }
    }

    #[test]
    fn test_len_with_duplicates() {
        let Some(device) = create_test_device() else {
            eprintln!("Skipping test - no GPU adapter available");
            return;
        };

        let cache = SamplerCache::new(device);

        let desc = TrinitySamplerDescriptor::new().anisotropy(4);
        for _ in 0..10 {
            let _ = cache.get_or_create(&desc);
        }

        // Should still be 1
        assert_eq!(cache.len(), 1);
    }

    // ------------------------------------------------------------------------
    // is_empty() when cache empty
    // ------------------------------------------------------------------------

    #[test]
    fn test_is_empty_initial() {
        let Some(device) = create_test_device() else {
            eprintln!("Skipping test - no GPU adapter available");
            return;
        };

        let cache = SamplerCache::new(device);
        assert!(cache.is_empty());
    }

    #[test]
    fn test_is_empty_false_after_insert() {
        let Some(device) = create_test_device() else {
            eprintln!("Skipping test - no GPU adapter available");
            return;
        };

        let cache = SamplerCache::new(device);
        let desc = TrinitySamplerDescriptor::new().anisotropy(4);
        let _ = cache.get_or_create(&desc);

        assert!(!cache.is_empty());
    }

    // ------------------------------------------------------------------------
    // clear() removes cached samplers
    // ------------------------------------------------------------------------

    #[test]
    fn test_clear_removes_cached_samplers() {
        let Some(device) = create_test_device() else {
            eprintln!("Skipping test - no GPU adapter available");
            return;
        };

        let cache = SamplerCache::new(device);

        // Add some samplers
        for i in 1..=5 {
            let desc = TrinitySamplerDescriptor::new().anisotropy(i);
            let _ = cache.get_or_create(&desc);
        }

        assert_eq!(cache.len(), 5);

        cache.clear();

        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
    }

    #[test]
    fn test_clear_resets_metrics() {
        let Some(device) = create_test_device() else {
            eprintln!("Skipping test - no GPU adapter available");
            return;
        };

        let cache = SamplerCache::new(device);

        let desc = TrinitySamplerDescriptor::new().anisotropy(4);
        let _ = cache.get_or_create(&desc);
        let _ = cache.get_or_create(&desc);

        let m1 = cache.metrics();
        assert_eq!(m1.misses, 1);
        assert_eq!(m1.hits, 1);

        cache.clear();

        let m2 = cache.metrics();
        assert_eq!(m2.misses, 0);
        assert_eq!(m2.hits, 0);
        assert_eq!(m2.cache_size, 0);
    }

    // ------------------------------------------------------------------------
    // clear() does NOT remove presets
    // ------------------------------------------------------------------------

    #[test]
    fn test_clear_preserves_presets() {
        let Some(device) = create_test_device() else {
            eprintln!("Skipping test - no GPU adapter available");
            return;
        };

        let cache = SamplerCache::new(device);

        // Get references to presets before clear
        let linear_before = cache.linear_clamp();
        let shadow_before = cache.shadow();

        // Add custom samplers and clear
        let desc = TrinitySamplerDescriptor::new().anisotropy(4);
        let _ = cache.get_or_create(&desc);
        cache.clear();

        // Presets should still be accessible
        let linear_after = cache.linear_clamp();
        let shadow_after = cache.shadow();

        // Should be same Arc
        assert!(Arc::ptr_eq(&linear_before, &linear_after));
        assert!(Arc::ptr_eq(&shadow_before, &shadow_after));

        // Preset count unchanged
        assert_eq!(cache.preset_count(), 5);
    }

    // ------------------------------------------------------------------------
    // total_count() tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_total_count_initial() {
        let Some(device) = create_test_device() else {
            eprintln!("Skipping test - no GPU adapter available");
            return;
        };

        let cache = SamplerCache::new(device);
        assert_eq!(cache.total_count(), 5); // Just presets
    }

    #[test]
    fn test_total_count_with_cached() {
        let Some(device) = create_test_device() else {
            eprintln!("Skipping test - no GPU adapter available");
            return;
        };

        let cache = SamplerCache::new(device);

        for i in 1..=3 {
            let desc = TrinitySamplerDescriptor::new().anisotropy(i);
            let _ = cache.get_or_create(&desc);
        }

        assert_eq!(cache.total_count(), 8); // 5 presets + 3 cached
    }

    // ------------------------------------------------------------------------
    // Debug trait
    // ------------------------------------------------------------------------

    #[test]
    fn test_debug_format() {
        let Some(device) = create_test_device() else {
            eprintln!("Skipping test - no GPU adapter available");
            return;
        };

        let cache = SamplerCache::new(device);
        let debug_str = format!("{:?}", cache);

        assert!(debug_str.contains("SamplerCache"));
        assert!(debug_str.contains("cache_size"));
        assert!(debug_str.contains("preset_count"));
    }
}

// ============================================================================
// Module: thread_safety_tests
// ============================================================================

mod thread_safety_tests {
    use super::*;

    // ------------------------------------------------------------------------
    // Concurrent get_or_create calls
    // ------------------------------------------------------------------------

    #[test]
    fn test_concurrent_get_or_create_same_descriptor() {
        let Some(device) = create_test_device() else {
            eprintln!("Skipping test - no GPU adapter available");
            return;
        };

        let cache = Arc::new(SamplerCache::new(device));
        let desc = TrinitySamplerDescriptor::new().anisotropy(4);

        let mut handles = vec![];

        for _ in 0..10 {
            let cache_clone = Arc::clone(&cache);
            let desc_clone = desc.clone();

            handles.push(thread::spawn(move || {
                cache_clone.get_or_create(&desc_clone)
            }));
        }

        let samplers: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();

        // All should return the same Arc
        for sampler in &samplers[1..] {
            assert!(Arc::ptr_eq(&samplers[0], sampler));
        }

        // Only one sampler should be in cache
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn test_concurrent_get_or_create_different_descriptors() {
        let Some(device) = create_test_device() else {
            eprintln!("Skipping test - no GPU adapter available");
            return;
        };

        let cache = Arc::new(SamplerCache::new(device));

        let mut handles = vec![];

        for i in 1..=8_u16 {
            let cache_clone = Arc::clone(&cache);

            handles.push(thread::spawn(move || {
                let desc = TrinitySamplerDescriptor::new().anisotropy(i);
                cache_clone.get_or_create(&desc)
            }));
        }

        for handle in handles {
            let _ = handle.join().unwrap();
        }

        // Should have 8 unique samplers
        assert_eq!(cache.len(), 8);
    }

    // ------------------------------------------------------------------------
    // Concurrent preset access
    // ------------------------------------------------------------------------

    #[test]
    fn test_concurrent_preset_access() {
        let Some(device) = create_test_device() else {
            eprintln!("Skipping test - no GPU adapter available");
            return;
        };

        let cache = Arc::new(SamplerCache::new(device));

        let mut handles = vec![];

        for i in 0..20 {
            let cache_clone = Arc::clone(&cache);

            handles.push(thread::spawn(move || match i % 5 {
                0 => cache_clone.linear_clamp(),
                1 => cache_clone.linear_repeat(),
                2 => cache_clone.point_clamp(),
                3 => cache_clone.point_repeat(),
                _ => cache_clone.shadow(),
            }));
        }

        for handle in handles {
            let _ = handle.join().unwrap();
        }

        // Cache should still be empty (presets don't count)
        assert_eq!(cache.len(), 0);
    }

    // ------------------------------------------------------------------------
    // Concurrent metrics reading
    // ------------------------------------------------------------------------

    #[test]
    fn test_concurrent_metrics_reading() {
        let Some(device) = create_test_device() else {
            eprintln!("Skipping test - no GPU adapter available");
            return;
        };

        let cache = Arc::new(SamplerCache::new(device));

        // Add some samplers first
        for i in 1..=5_u16 {
            let desc = TrinitySamplerDescriptor::new().anisotropy(i);
            let _ = cache.get_or_create(&desc);
        }

        let mut handles = vec![];

        for _ in 0..20 {
            let cache_clone = Arc::clone(&cache);

            handles.push(thread::spawn(move || {
                let metrics = cache_clone.metrics();
                assert!(metrics.cache_size >= 5);
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }
    }

    // ------------------------------------------------------------------------
    // Mixed concurrent operations
    // ------------------------------------------------------------------------

    #[test]
    fn test_mixed_concurrent_operations() {
        let Some(device) = create_test_device() else {
            eprintln!("Skipping test - no GPU adapter available");
            return;
        };

        let cache = Arc::new(SamplerCache::new(device));

        let mut handles = vec![];

        // Readers
        for _ in 0..5 {
            let cache_clone = Arc::clone(&cache);
            handles.push(thread::spawn(move || {
                for _ in 0..10 {
                    let _ = cache_clone.metrics();
                    let _ = cache_clone.len();
                    let _ = cache_clone.is_empty();
                }
            }));
        }

        // Writers
        for i in 1..=5_u16 {
            let cache_clone = Arc::clone(&cache);
            handles.push(thread::spawn(move || {
                for j in 0..5_u16 {
                    let desc = TrinitySamplerDescriptor::new().anisotropy(i * 10 + j);
                    let _ = cache_clone.get_or_create(&desc);
                }
            }));
        }

        // Preset accessors
        for _ in 0..5 {
            let cache_clone = Arc::clone(&cache);
            handles.push(thread::spawn(move || {
                for _ in 0..10 {
                    let _ = cache_clone.linear_clamp();
                    let _ = cache_clone.shadow();
                }
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }

        // Cache should have 25 unique samplers (5 threads * 5 each)
        assert_eq!(cache.len(), 25);
    }
}

// ============================================================================
// Module: edge_case_tests
// ============================================================================

mod edge_case_tests {
    use super::*;

    // ------------------------------------------------------------------------
    // Many unique descriptors
    // ------------------------------------------------------------------------

    #[test]
    fn test_many_unique_descriptors() {
        let Some(device) = create_test_device() else {
            eprintln!("Skipping test - no GPU adapter available");
            return;
        };

        let cache = SamplerCache::new(device);

        // Create 50 unique samplers with different anisotropy values
        for i in 1..=16_u16 {
            let desc = TrinitySamplerDescriptor::new().anisotropy(i);
            let _ = cache.get_or_create(&desc);
        }

        // Create more with different LOD values
        for i in 0..10 {
            let desc = TrinitySamplerDescriptor::new().lod_clamp(i as f32, 32.0);
            let _ = cache.get_or_create(&desc);
        }

        assert!(cache.len() >= 20);
    }

    #[test]
    fn test_cache_stress_with_combinations() {
        let Some(device) = create_test_device() else {
            eprintln!("Skipping test - no GPU adapter available");
            return;
        };

        let cache = SamplerCache::new(device);

        let filters = [FilterMode::Nearest, FilterMode::Linear];
        let addresses = [
            AddressMode::ClampToEdge,
            AddressMode::Repeat,
            AddressMode::MirrorRepeat,
        ];

        for &filter in &filters {
            for &address in &addresses {
                let desc = TrinitySamplerDescriptor::new()
                    .filter(filter)
                    .address_mode(address);
                let _ = cache.get_or_create(&desc);
            }
        }

        // 2 filters * 3 addresses = 6 unique samplers
        assert_eq!(cache.len(), 6);
    }

    // ------------------------------------------------------------------------
    // Same descriptor from multiple threads
    // ------------------------------------------------------------------------

    #[test]
    fn test_same_descriptor_many_threads() {
        let Some(device) = create_test_device() else {
            eprintln!("Skipping test - no GPU adapter available");
            return;
        };

        let cache = Arc::new(SamplerCache::new(device));
        let desc = TrinitySamplerDescriptor::trilinear().anisotropy(8);

        let mut handles = vec![];

        for _ in 0..50 {
            let cache_clone = Arc::clone(&cache);
            let desc_clone = desc.clone();

            handles.push(thread::spawn(move || {
                cache_clone.get_or_create(&desc_clone)
            }));
        }

        let samplers: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();

        // All should be the same Arc
        let first = &samplers[0];
        for sampler in &samplers {
            assert!(Arc::ptr_eq(first, sampler));
        }

        // Should only have 1 sampler
        assert_eq!(cache.len(), 1);

        // Should have 1 miss and 49 hits
        let metrics = cache.metrics();
        assert_eq!(metrics.misses, 1);
        assert_eq!(metrics.hits, 49);
    }

    // ------------------------------------------------------------------------
    // Clear while accessing
    // ------------------------------------------------------------------------

    #[test]
    fn test_clear_and_recreate() {
        let Some(device) = create_test_device() else {
            eprintln!("Skipping test - no GPU adapter available");
            return;
        };

        let cache = SamplerCache::new(device);

        let desc = TrinitySamplerDescriptor::new().anisotropy(4);

        // Create sampler
        let s1 = cache.get_or_create(&desc);
        assert_eq!(cache.len(), 1);

        // Clear
        cache.clear();
        assert_eq!(cache.len(), 0);

        // Create again
        let s2 = cache.get_or_create(&desc);
        assert_eq!(cache.len(), 1);

        // Should be different Arc (new sampler created)
        assert!(!Arc::ptr_eq(&s1, &s2));
    }

    #[test]
    fn test_sampler_survives_clear() {
        let Some(device) = create_test_device() else {
            eprintln!("Skipping test - no GPU adapter available");
            return;
        };

        let cache = SamplerCache::new(device);

        let desc = TrinitySamplerDescriptor::new().anisotropy(4);
        let sampler = cache.get_or_create(&desc);

        // Strong count should be at least 2 (cache + our variable)
        assert!(Arc::strong_count(&sampler) >= 2);

        cache.clear();

        // Sampler should still be valid (our variable holds reference)
        assert!(Arc::strong_count(&sampler) >= 1);
    }

    // ------------------------------------------------------------------------
    // Extreme values
    // ------------------------------------------------------------------------

    #[test]
    fn test_anisotropy_extreme_values() {
        let Some(device) = create_test_device() else {
            eprintln!("Skipping test - no GPU adapter available");
            return;
        };

        let cache = SamplerCache::new(device);

        // Min anisotropy
        let desc1 = TrinitySamplerDescriptor::new().anisotropy(1);
        let _ = cache.get_or_create(&desc1);

        // Max anisotropy (will be clamped to 16)
        let desc2 = TrinitySamplerDescriptor::new().anisotropy(16);
        let _ = cache.get_or_create(&desc2);

        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn test_lod_extreme_values() {
        let Some(device) = create_test_device() else {
            eprintln!("Skipping test - no GPU adapter available");
            return;
        };

        let cache = SamplerCache::new(device);

        // Zero LOD range
        let desc1 = TrinitySamplerDescriptor::new().lod_clamp(0.0, 0.0);
        let _ = cache.get_or_create(&desc1);

        // Large LOD range
        let desc2 = TrinitySamplerDescriptor::new().lod_clamp(0.0, 32.0);
        let _ = cache.get_or_create(&desc2);

        assert_eq!(cache.len(), 2);
    }

    // ------------------------------------------------------------------------
    // Interleaved operations
    // ------------------------------------------------------------------------

    #[test]
    fn test_interleaved_create_and_clear() {
        let Some(device) = create_test_device() else {
            eprintln!("Skipping test - no GPU adapter available");
            return;
        };

        let cache = SamplerCache::new(device);

        for iteration in 0..5 {
            // Add samplers
            for i in 1..=3_u16 {
                let desc = TrinitySamplerDescriptor::new().anisotropy(i);
                let _ = cache.get_or_create(&desc);
            }

            assert_eq!(cache.len(), 3, "Iteration {}: expected 3 samplers", iteration);

            // Clear
            cache.clear();

            assert_eq!(cache.len(), 0, "Iteration {}: expected 0 after clear", iteration);
        }
    }

    #[test]
    fn test_preset_stability_across_clear() {
        let Some(device) = create_test_device() else {
            eprintln!("Skipping test - no GPU adapter available");
            return;
        };

        let cache = SamplerCache::new(device);

        let presets_before: Vec<_> = vec![
            cache.linear_clamp(),
            cache.linear_repeat(),
            cache.point_clamp(),
            cache.point_repeat(),
            cache.shadow(),
        ];

        // Add and clear multiple times
        for _ in 0..5 {
            let desc = TrinitySamplerDescriptor::new().anisotropy(4);
            let _ = cache.get_or_create(&desc);
            cache.clear();
        }

        let presets_after: Vec<_> = vec![
            cache.linear_clamp(),
            cache.linear_repeat(),
            cache.point_clamp(),
            cache.point_repeat(),
            cache.shadow(),
        ];

        // All presets should be identical
        for (before, after) in presets_before.iter().zip(presets_after.iter()) {
            assert!(Arc::ptr_eq(before, after));
        }
    }

    // ------------------------------------------------------------------------
    // All compare functions create unique keys
    // ------------------------------------------------------------------------

    #[test]
    fn test_all_compare_functions_unique() {
        let comparisons = [
            CompareFunction::Never,
            CompareFunction::Less,
            CompareFunction::Equal,
            CompareFunction::LessEqual,
            CompareFunction::Greater,
            CompareFunction::NotEqual,
            CompareFunction::GreaterEqual,
            CompareFunction::Always,
        ];

        let mut keys = Vec::new();

        for compare in comparisons {
            let desc = TrinitySamplerDescriptor::new().compare(compare);
            let key = SamplerCacheKey::from_descriptor(&desc);
            keys.push(key);
        }

        // All keys should be unique
        for i in 0..keys.len() {
            for j in (i + 1)..keys.len() {
                assert_ne!(keys[i], keys[j], "Compare functions {} and {} produced same key", i, j);
            }
        }
    }

    // ------------------------------------------------------------------------
    // All border colors create unique keys
    // ------------------------------------------------------------------------

    #[test]
    fn test_all_border_colors_unique() {
        let colors = [
            SamplerBorderColor::TransparentBlack,
            SamplerBorderColor::OpaqueBlack,
            SamplerBorderColor::OpaqueWhite,
            SamplerBorderColor::Zero,
        ];

        let mut keys = Vec::new();

        for color in colors {
            let desc = TrinitySamplerDescriptor::new()
                .address_mode(AddressMode::ClampToBorder)
                .border_color(color);
            let key = SamplerCacheKey::from_descriptor(&desc);
            keys.push(key);
        }

        // All keys should be unique
        for i in 0..keys.len() {
            for j in (i + 1)..keys.len() {
                assert_ne!(keys[i], keys[j], "Border colors {} and {} produced same key", i, j);
            }
        }
    }
}
