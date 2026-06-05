//! Blackbox tests for shader permutation management (T-WGPU-P2.7.6).
//!
//! CLEANROOM: Tests use ONLY the public API exported by `renderer_backend::shaders`.
//! No internal fields, no private methods, no implementation details.
//!
//! Coverage:
//!   1.  FeatureFlags -- all variants, bitwise operations, iteration, parsing
//!   2.  PermutationKey -- creation, equality, hashing, HashMap usage
//!   3.  PermutationConfig -- builder pattern, presets, validation
//!   4.  EvictionPolicy -- variants, Display, Default
//!   5.  PermutationMetrics -- calculations, Display, reset
//!   6.  PermutationError -- variants, Display, Error trait
//!   7.  ShaderPermutationManager -- creation, metrics, invalidation (no device)
//!   8.  Integration patterns -- typical engine usage scenarios

use renderer_backend::shaders::{
    EvictionPolicy, FeatureFlags, PermutationConfig, PermutationError,
    PermutationKey, PermutationMetrics, ShaderPermutationManager, DEFAULT_MAX_PERMUTATIONS,
    FEATURE_FLAG_COUNT,
};
use std::collections::{HashMap, HashSet};

// =============================================================================
// SECTION 1: API Surface Tests (FeatureFlags)
// =============================================================================

#[test]
fn api_feature_flags_skinned_is_valid() {
    let flag = FeatureFlags::SKINNED;
    assert!(!flag.is_empty());
    assert_eq!(flag.flag_count(), 1);
    assert_eq!(flag.flag_name(), "SKINNED");
}

#[test]
fn api_feature_flags_alpha_test_is_valid() {
    let flag = FeatureFlags::ALPHA_TEST;
    assert!(!flag.is_empty());
    assert_eq!(flag.flag_count(), 1);
    assert_eq!(flag.flag_name(), "ALPHA_TEST");
}

#[test]
fn api_feature_flags_normal_map_is_valid() {
    let flag = FeatureFlags::NORMAL_MAP;
    assert!(!flag.is_empty());
    assert_eq!(flag.flag_count(), 1);
    assert_eq!(flag.flag_name(), "NORMAL_MAP");
}

#[test]
fn api_feature_flags_emissive_is_valid() {
    let flag = FeatureFlags::EMISSIVE;
    assert!(!flag.is_empty());
    assert_eq!(flag.flag_count(), 1);
    assert_eq!(flag.flag_name(), "EMISSIVE");
}

#[test]
fn api_feature_flags_shadows_is_valid() {
    let flag = FeatureFlags::SHADOWS;
    assert!(!flag.is_empty());
    assert_eq!(flag.flag_count(), 1);
    assert_eq!(flag.flag_name(), "SHADOWS");
}

#[test]
fn api_feature_flags_fog_is_valid() {
    let flag = FeatureFlags::FOG;
    assert!(!flag.is_empty());
    assert_eq!(flag.flag_count(), 1);
    assert_eq!(flag.flag_name(), "FOG");
}

#[test]
fn api_feature_flags_instanced_is_valid() {
    let flag = FeatureFlags::INSTANCED;
    assert!(!flag.is_empty());
    assert_eq!(flag.flag_count(), 1);
    assert_eq!(flag.flag_name(), "INSTANCED");
}

#[test]
fn api_feature_flags_none_is_empty() {
    let flag = FeatureFlags::NONE;
    assert!(flag.is_empty());
    assert!(flag.is_none_set());
    assert_eq!(flag.flag_count(), 0);
    assert_eq!(flag.flag_name(), "NONE");
}

#[test]
fn api_feature_flags_all_contains_seven_flags() {
    let flag = FeatureFlags::ALL;
    assert!(flag.is_all_set());
    assert_eq!(flag.flag_count(), 7);
}

#[test]
fn api_feature_flags_default_is_empty() {
    let flag = FeatureFlags::default();
    assert!(flag.is_empty());
}

#[test]
fn api_permutation_key_new_creates_valid_key() {
    let key = PermutationKey::new(12345, FeatureFlags::SKINNED);
    assert_eq!(key.shader_id(), 12345);
    assert_eq!(key.features(), FeatureFlags::SKINNED);
}

#[test]
fn api_permutation_key_base_has_no_features() {
    let key = PermutationKey::base(99999);
    assert_eq!(key.shader_id(), 99999);
    assert!(key.features().is_empty());
}

// =============================================================================
// SECTION 2: Feature Flag Composition Tests
// =============================================================================

#[test]
fn feature_flags_pbr_standard_combination() {
    // PBR materials typically use: normal map + shadows + emissive
    let flags = FeatureFlags::NORMAL_MAP | FeatureFlags::SHADOWS | FeatureFlags::EMISSIVE;
    assert_eq!(flags.flag_count(), 3);
    assert!(flags.contains(FeatureFlags::NORMAL_MAP));
    assert!(flags.contains(FeatureFlags::SHADOWS));
    assert!(flags.contains(FeatureFlags::EMISSIVE));
    assert!(!flags.contains(FeatureFlags::SKINNED));
}

#[test]
fn feature_flags_skinned_mesh_combination() {
    // Skinned meshes: skinned + shadows + normal map
    let flags = FeatureFlags::SKINNED | FeatureFlags::SHADOWS | FeatureFlags::NORMAL_MAP;
    assert_eq!(flags.flag_count(), 3);
    assert!(flags.contains(FeatureFlags::SKINNED));
}

#[test]
fn feature_flags_instanced_foliage_combination() {
    // Foliage: instanced + alpha test + fog
    let flags = FeatureFlags::INSTANCED | FeatureFlags::ALPHA_TEST | FeatureFlags::FOG;
    assert_eq!(flags.flag_count(), 3);
    assert!(flags.contains(FeatureFlags::INSTANCED));
    assert!(flags.contains(FeatureFlags::ALPHA_TEST));
}

#[test]
fn feature_flags_full_featured_combination() {
    // All features enabled
    let flags = FeatureFlags::SKINNED
        | FeatureFlags::ALPHA_TEST
        | FeatureFlags::NORMAL_MAP
        | FeatureFlags::EMISSIVE
        | FeatureFlags::SHADOWS
        | FeatureFlags::FOG
        | FeatureFlags::INSTANCED;
    assert!(flags.is_all_set());
    assert_eq!(flags, FeatureFlags::ALL);
}

#[test]
fn feature_flags_bitwise_and_intersection() {
    let a = FeatureFlags::SKINNED | FeatureFlags::SHADOWS | FeatureFlags::FOG;
    let b = FeatureFlags::SHADOWS | FeatureFlags::FOG | FeatureFlags::EMISSIVE;
    let intersection = a & b;

    assert_eq!(intersection.flag_count(), 2);
    assert!(intersection.contains(FeatureFlags::SHADOWS));
    assert!(intersection.contains(FeatureFlags::FOG));
    assert!(!intersection.contains(FeatureFlags::SKINNED));
    assert!(!intersection.contains(FeatureFlags::EMISSIVE));
}

#[test]
fn feature_flags_bitwise_xor_symmetric_difference() {
    let a = FeatureFlags::SKINNED | FeatureFlags::SHADOWS;
    let b = FeatureFlags::SHADOWS | FeatureFlags::FOG;
    let xor = a ^ b;

    assert!(xor.contains(FeatureFlags::SKINNED));
    assert!(xor.contains(FeatureFlags::FOG));
    assert!(!xor.contains(FeatureFlags::SHADOWS));
}

#[test]
fn feature_flags_bitwise_not_complement() {
    let flags = FeatureFlags::SKINNED;
    let complement = !flags & FeatureFlags::ALL;

    assert!(!complement.contains(FeatureFlags::SKINNED));
    assert!(complement.contains(FeatureFlags::ALPHA_TEST));
    assert!(complement.contains(FeatureFlags::NORMAL_MAP));
    assert!(complement.contains(FeatureFlags::EMISSIVE));
    assert!(complement.contains(FeatureFlags::SHADOWS));
    assert!(complement.contains(FeatureFlags::FOG));
    assert!(complement.contains(FeatureFlags::INSTANCED));
}

#[test]
fn feature_flags_with_feature_adds_flag() {
    let flags = FeatureFlags::SKINNED;
    let with_shadows = flags.with_feature(FeatureFlags::SHADOWS);

    assert!(with_shadows.contains(FeatureFlags::SKINNED));
    assert!(with_shadows.contains(FeatureFlags::SHADOWS));
    assert_eq!(with_shadows.flag_count(), 2);
}

#[test]
fn feature_flags_without_feature_removes_flag() {
    let flags = FeatureFlags::SKINNED | FeatureFlags::SHADOWS;
    let without_shadows = flags.without_feature(FeatureFlags::SHADOWS);

    assert!(without_shadows.contains(FeatureFlags::SKINNED));
    assert!(!without_shadows.contains(FeatureFlags::SHADOWS));
    assert_eq!(without_shadows.flag_count(), 1);
}

#[test]
fn feature_flags_toggle_feature_flips_state() {
    let flags = FeatureFlags::SKINNED;

    // Toggle on
    let toggled = flags.toggle_feature(FeatureFlags::SHADOWS);
    assert!(toggled.contains(FeatureFlags::SHADOWS));

    // Toggle off
    let toggled_back = toggled.toggle_feature(FeatureFlags::SHADOWS);
    assert!(!toggled_back.contains(FeatureFlags::SHADOWS));
}

// =============================================================================
// SECTION 3: FeatureFlags Iteration and Naming Tests
// =============================================================================

#[test]
fn feature_flags_iter_enabled_returns_all_enabled() {
    let flags = FeatureFlags::SKINNED | FeatureFlags::SHADOWS | FeatureFlags::FOG;
    let enabled: Vec<_> = flags.iter_enabled().collect();

    assert_eq!(enabled.len(), 3);
    assert!(enabled.contains(&FeatureFlags::SKINNED));
    assert!(enabled.contains(&FeatureFlags::SHADOWS));
    assert!(enabled.contains(&FeatureFlags::FOG));
}

#[test]
fn feature_flags_iter_enabled_empty_for_none() {
    let flags = FeatureFlags::NONE;
    let enabled: Vec<_> = flags.iter_enabled().collect();

    assert!(enabled.is_empty());
}

#[test]
fn feature_flags_names_returns_string_list() {
    let flags = FeatureFlags::SKINNED | FeatureFlags::SHADOWS;
    let names = flags.names();

    assert_eq!(names.len(), 2);
    assert!(names.contains(&"SKINNED"));
    assert!(names.contains(&"SHADOWS"));
}

#[test]
fn feature_flags_from_names_parses_list() {
    let flags = FeatureFlags::from_names(&["SKINNED", "SHADOWS", "FOG"]);

    assert_eq!(flags.flag_count(), 3);
    assert!(flags.contains(FeatureFlags::SKINNED));
    assert!(flags.contains(FeatureFlags::SHADOWS));
    assert!(flags.contains(FeatureFlags::FOG));
}

#[test]
fn feature_flags_from_names_case_insensitive() {
    let flags = FeatureFlags::from_names(&["skinned", "SHADOWS", "Fog"]);

    assert_eq!(flags.flag_count(), 3);
    assert!(flags.contains(FeatureFlags::SKINNED));
    assert!(flags.contains(FeatureFlags::SHADOWS));
    assert!(flags.contains(FeatureFlags::FOG));
}

#[test]
fn feature_flags_from_names_ignores_unknown() {
    let flags = FeatureFlags::from_names(&["SKINNED", "UNKNOWN", "SHADOWS"]);

    assert_eq!(flags.flag_count(), 2);
    assert!(flags.contains(FeatureFlags::SKINNED));
    assert!(flags.contains(FeatureFlags::SHADOWS));
}

#[test]
fn feature_flags_from_names_empty_returns_none() {
    let flags = FeatureFlags::from_names(&[]);
    assert!(flags.is_empty());
}

#[test]
fn feature_flags_parse_name_all_variants() {
    assert_eq!(FeatureFlags::parse_name("SKINNED"), FeatureFlags::SKINNED);
    assert_eq!(FeatureFlags::parse_name("ALPHA_TEST"), FeatureFlags::ALPHA_TEST);
    assert_eq!(FeatureFlags::parse_name("NORMAL_MAP"), FeatureFlags::NORMAL_MAP);
    assert_eq!(FeatureFlags::parse_name("EMISSIVE"), FeatureFlags::EMISSIVE);
    assert_eq!(FeatureFlags::parse_name("SHADOWS"), FeatureFlags::SHADOWS);
    assert_eq!(FeatureFlags::parse_name("FOG"), FeatureFlags::FOG);
    assert_eq!(FeatureFlags::parse_name("INSTANCED"), FeatureFlags::INSTANCED);
    assert_eq!(FeatureFlags::parse_name("unknown"), FeatureFlags::NONE);
}

#[test]
fn feature_flags_total_permutations_is_power_of_two() {
    let total = FeatureFlags::total_permutations();
    assert_eq!(total, 128); // 2^7
    assert!(total.is_power_of_two());
}

#[test]
fn feature_flags_display_shows_flags() {
    let flags = FeatureFlags::SKINNED | FeatureFlags::SHADOWS;
    let display = format!("{}", flags);

    assert!(display.contains("SKINNED"));
    assert!(display.contains("SHADOWS"));
}

#[test]
fn feature_flags_display_none_shows_none() {
    let flags = FeatureFlags::NONE;
    let display = format!("{}", flags);

    assert_eq!(display, "NONE");
}

// =============================================================================
// SECTION 4: PermutationKey Tests
// =============================================================================

#[test]
fn permutation_key_equality_same_shader_same_features() {
    let key1 = PermutationKey::new(100, FeatureFlags::SKINNED | FeatureFlags::SHADOWS);
    let key2 = PermutationKey::new(100, FeatureFlags::SKINNED | FeatureFlags::SHADOWS);

    assert_eq!(key1, key2);
}

#[test]
fn permutation_key_inequality_different_shader_id() {
    let key1 = PermutationKey::new(100, FeatureFlags::SKINNED);
    let key2 = PermutationKey::new(200, FeatureFlags::SKINNED);

    assert_ne!(key1, key2);
}

#[test]
fn permutation_key_inequality_different_features() {
    let key1 = PermutationKey::new(100, FeatureFlags::SKINNED);
    let key2 = PermutationKey::new(100, FeatureFlags::SHADOWS);

    assert_ne!(key1, key2);
}

#[test]
fn permutation_key_works_as_hashmap_key() {
    let mut map: HashMap<PermutationKey, i32> = HashMap::new();

    let key1 = PermutationKey::new(100, FeatureFlags::SKINNED);
    let key2 = PermutationKey::new(100, FeatureFlags::SHADOWS);
    let key3 = PermutationKey::new(200, FeatureFlags::SKINNED);

    map.insert(key1, 1);
    map.insert(key2, 2);
    map.insert(key3, 3);

    assert_eq!(map.len(), 3);
    assert_eq!(map.get(&key1), Some(&1));
    assert_eq!(map.get(&key2), Some(&2));
    assert_eq!(map.get(&key3), Some(&3));
}

#[test]
fn permutation_key_lookup_finds_existing() {
    let mut map: HashMap<PermutationKey, &str> = HashMap::new();
    let key = PermutationKey::new(42, FeatureFlags::SKINNED | FeatureFlags::SHADOWS);
    map.insert(key, "found");

    // Create identical key for lookup
    let lookup_key = PermutationKey::new(42, FeatureFlags::SKINNED | FeatureFlags::SHADOWS);
    assert_eq!(map.get(&lookup_key), Some(&"found"));
}

#[test]
fn permutation_key_with_feature_chains() {
    let key = PermutationKey::base(100)
        .with_feature(FeatureFlags::SKINNED)
        .with_feature(FeatureFlags::SHADOWS)
        .with_feature(FeatureFlags::FOG);

    assert_eq!(key.features().flag_count(), 3);
    assert!(key.features().contains(FeatureFlags::SKINNED));
    assert!(key.features().contains(FeatureFlags::SHADOWS));
    assert!(key.features().contains(FeatureFlags::FOG));
}

#[test]
fn permutation_key_without_feature_removes() {
    let key = PermutationKey::new(100, FeatureFlags::SKINNED | FeatureFlags::SHADOWS | FeatureFlags::FOG);
    let modified = key.without_feature(FeatureFlags::SHADOWS);

    assert!(modified.features().contains(FeatureFlags::SKINNED));
    assert!(!modified.features().contains(FeatureFlags::SHADOWS));
    assert!(modified.features().contains(FeatureFlags::FOG));
}

#[test]
fn permutation_key_with_features_replaces_all() {
    let key = PermutationKey::new(100, FeatureFlags::SKINNED);
    let modified = key.with_features(FeatureFlags::SHADOWS | FeatureFlags::FOG);

    assert!(!modified.features().contains(FeatureFlags::SKINNED));
    assert!(modified.features().contains(FeatureFlags::SHADOWS));
    assert!(modified.features().contains(FeatureFlags::FOG));
}

#[test]
fn permutation_key_display_string_contains_shader_id() {
    let key = PermutationKey::new(12345, FeatureFlags::SKINNED);
    let display = key.display_string();

    assert!(display.contains("12345"));
}

#[test]
fn permutation_key_display_trait() {
    let key = PermutationKey::new(12345, FeatureFlags::SKINNED);
    let display = format!("{}", key);

    assert!(display.contains("Permutation"));
    assert!(display.contains("12345"));
}

// =============================================================================
// SECTION 5: PermutationConfig Tests
// =============================================================================

#[test]
fn permutation_config_default_values() {
    let config = PermutationConfig::default();

    assert_eq!(config.max_permutations, DEFAULT_MAX_PERMUTATIONS);
    assert!(config.enable_lazy_compilation);
    assert_eq!(config.eviction_policy, EvictionPolicy::LRU);
}

#[test]
fn permutation_config_new_equals_default() {
    let config1 = PermutationConfig::new();
    let config2 = PermutationConfig::default();

    assert_eq!(config1.max_permutations, config2.max_permutations);
    assert_eq!(config1.enable_lazy_compilation, config2.enable_lazy_compilation);
    assert_eq!(config1.eviction_policy, config2.eviction_policy);
}

#[test]
fn permutation_config_builder_max_permutations() {
    let config = PermutationConfig::new().max_permutations(512);
    assert_eq!(config.max_permutations, 512);
}

#[test]
fn permutation_config_builder_lazy_compilation() {
    let config = PermutationConfig::new().enable_lazy_compilation(false);
    assert!(!config.enable_lazy_compilation);
}

#[test]
fn permutation_config_builder_eviction_policy() {
    let config = PermutationConfig::new().eviction_policy(EvictionPolicy::LFU);
    assert_eq!(config.eviction_policy, EvictionPolicy::LFU);
}

#[test]
fn permutation_config_builder_chain() {
    let config = PermutationConfig::new()
        .max_permutations(128)
        .enable_lazy_compilation(false)
        .eviction_policy(EvictionPolicy::Oldest);

    assert_eq!(config.max_permutations, 128);
    assert!(!config.enable_lazy_compilation);
    assert_eq!(config.eviction_policy, EvictionPolicy::Oldest);
}

#[test]
fn permutation_config_minimal_preset() {
    let config = PermutationConfig::minimal();

    assert_eq!(config.max_permutations, 16);
    assert!(config.enable_lazy_compilation);
}

#[test]
fn permutation_config_development_preset() {
    let config = PermutationConfig::development();

    assert_eq!(config.max_permutations, 64);
    assert!(config.enable_lazy_compilation);
}

#[test]
fn permutation_config_production_preset() {
    let config = PermutationConfig::production();

    assert_eq!(config.max_permutations, 512);
    assert_eq!(config.eviction_policy, EvictionPolicy::LFU);
}

#[test]
fn permutation_config_validate_default_succeeds() {
    let config = PermutationConfig::default();
    assert!(config.validate().is_ok());
}

#[test]
fn permutation_config_validate_zero_fails() {
    let config = PermutationConfig::new().max_permutations(0);
    let result = config.validate();

    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(err.is_config_error());
}

#[test]
fn permutation_config_clone_preserves_values() {
    let config = PermutationConfig::new()
        .max_permutations(100)
        .eviction_policy(EvictionPolicy::Oldest);
    let cloned = config.clone();

    assert_eq!(cloned.max_permutations, 100);
    assert_eq!(cloned.eviction_policy, EvictionPolicy::Oldest);
}

// =============================================================================
// SECTION 6: EvictionPolicy Tests
// =============================================================================

#[test]
fn eviction_policy_default_is_lru() {
    assert_eq!(EvictionPolicy::default(), EvictionPolicy::LRU);
}

#[test]
fn eviction_policy_lru_name() {
    assert_eq!(EvictionPolicy::LRU.name(), "LRU");
}

#[test]
fn eviction_policy_lfu_name() {
    assert_eq!(EvictionPolicy::LFU.name(), "LFU");
}

#[test]
fn eviction_policy_oldest_name() {
    assert_eq!(EvictionPolicy::Oldest.name(), "Oldest");
}

#[test]
fn eviction_policy_display_lru() {
    let policy = EvictionPolicy::LRU;
    assert_eq!(format!("{}", policy), "LRU");
}

#[test]
fn eviction_policy_display_lfu() {
    let policy = EvictionPolicy::LFU;
    assert_eq!(format!("{}", policy), "LFU");
}

#[test]
fn eviction_policy_display_oldest() {
    let policy = EvictionPolicy::Oldest;
    assert_eq!(format!("{}", policy), "Oldest");
}

#[test]
fn eviction_policy_equality() {
    assert_eq!(EvictionPolicy::LRU, EvictionPolicy::LRU);
    assert_eq!(EvictionPolicy::LFU, EvictionPolicy::LFU);
    assert_eq!(EvictionPolicy::Oldest, EvictionPolicy::Oldest);
}

#[test]
fn eviction_policy_inequality() {
    assert_ne!(EvictionPolicy::LRU, EvictionPolicy::LFU);
    assert_ne!(EvictionPolicy::LRU, EvictionPolicy::Oldest);
    assert_ne!(EvictionPolicy::LFU, EvictionPolicy::Oldest);
}

#[test]
fn eviction_policy_copy_clone() {
    let policy = EvictionPolicy::LFU;
    let copied = policy;
    let cloned = policy.clone();

    assert_eq!(policy, copied);
    assert_eq!(policy, cloned);
}

// =============================================================================
// SECTION 7: PermutationMetrics Tests
// =============================================================================

#[test]
fn permutation_metrics_default_all_zero() {
    let metrics = PermutationMetrics::default();

    assert_eq!(metrics.cache_size, 0);
    assert_eq!(metrics.cache_hits, 0);
    assert_eq!(metrics.cache_misses, 0);
    assert_eq!(metrics.compilations, 0);
    assert_eq!(metrics.evictions, 0);
    assert_eq!(metrics.hit_rate, 0.0);
}

#[test]
fn permutation_metrics_new_calculates_hit_rate() {
    let metrics = PermutationMetrics::new(10, 80, 20, 20, 5);

    assert_eq!(metrics.cache_size, 10);
    assert_eq!(metrics.cache_hits, 80);
    assert_eq!(metrics.cache_misses, 20);
    assert_eq!(metrics.compilations, 20);
    assert_eq!(metrics.evictions, 5);
    assert_eq!(metrics.hit_rate, 0.8);
}

#[test]
fn permutation_metrics_total_requests() {
    let metrics = PermutationMetrics::new(0, 50, 50, 50, 0);
    assert_eq!(metrics.total_requests(), 100);
}

#[test]
fn permutation_metrics_hit_rate_percent() {
    let metrics = PermutationMetrics::new(0, 75, 25, 25, 0);
    assert_eq!(metrics.hit_rate_percent(), 75.0);
}

#[test]
fn permutation_metrics_miss_rate() {
    let metrics = PermutationMetrics::new(0, 60, 40, 40, 0);
    assert!((metrics.miss_rate() - 0.4).abs() < 0.001);
}

#[test]
fn permutation_metrics_is_empty_true() {
    let metrics = PermutationMetrics::new(0, 10, 5, 5, 0);
    assert!(metrics.is_empty());
}

#[test]
fn permutation_metrics_is_empty_false() {
    let metrics = PermutationMetrics::new(1, 10, 5, 5, 0);
    assert!(!metrics.is_empty());
}

#[test]
fn permutation_metrics_reset_clears_all() {
    let mut metrics = PermutationMetrics::new(10, 80, 20, 20, 5);
    metrics.reset();

    assert_eq!(metrics.cache_size, 0);
    assert_eq!(metrics.cache_hits, 0);
    assert_eq!(metrics.cache_misses, 0);
}

#[test]
fn permutation_metrics_zero_requests_zero_hit_rate() {
    let metrics = PermutationMetrics::new(0, 0, 0, 0, 0);
    assert_eq!(metrics.hit_rate, 0.0);
    assert_eq!(metrics.total_requests(), 0);
}

#[test]
fn permutation_metrics_all_hits_perfect_rate() {
    let metrics = PermutationMetrics::new(10, 100, 0, 0, 0);
    assert_eq!(metrics.hit_rate, 1.0);
    assert_eq!(metrics.hit_rate_percent(), 100.0);
}

#[test]
fn permutation_metrics_all_misses_zero_rate() {
    let metrics = PermutationMetrics::new(10, 0, 100, 100, 0);
    assert_eq!(metrics.hit_rate, 0.0);
    assert_eq!(metrics.miss_rate(), 1.0);
}

#[test]
fn permutation_metrics_display_format() {
    let metrics = PermutationMetrics::new(10, 80, 20, 20, 5);
    let display = format!("{}", metrics);

    assert!(display.contains("PermutationMetrics"));
    assert!(display.contains("size=10"));
    assert!(display.contains("hits=80"));
    assert!(display.contains("misses=20"));
    assert!(display.contains("80.0%"));
}

#[test]
fn permutation_metrics_clone() {
    let metrics = PermutationMetrics::new(10, 80, 20, 20, 5);
    let cloned = metrics.clone();

    assert_eq!(cloned.cache_size, 10);
    assert_eq!(cloned.cache_hits, 80);
}

// =============================================================================
// SECTION 8: PermutationError Tests
// =============================================================================

#[test]
fn permutation_error_max_exceeded_is_max_exceeded() {
    let err = PermutationError::MaxPermutationsExceeded {
        current: 256,
        max: 256,
    };

    assert!(err.is_max_exceeded());
    assert!(!err.is_compilation_error());
    assert!(!err.is_not_found());
    assert!(!err.is_config_error());
}

#[test]
fn permutation_error_compilation_failed_is_compilation() {
    let err = PermutationError::CompilationFailed("parse error".to_string());

    assert!(err.is_compilation_error());
    assert!(!err.is_max_exceeded());
}

#[test]
fn permutation_error_shader_not_found_is_not_found() {
    let err = PermutationError::ShaderNotFound { shader_id: 12345 };

    assert!(err.is_not_found());
    assert!(!err.is_compilation_error());
}

#[test]
fn permutation_error_config_error_is_config() {
    let err = PermutationError::ConfigError("invalid".to_string());

    assert!(err.is_config_error());
    assert!(!err.is_not_found());
}

#[test]
fn permutation_error_max_exceeded_display() {
    let err = PermutationError::MaxPermutationsExceeded {
        current: 300,
        max: 256,
    };
    let display = format!("{}", err);

    assert!(display.contains("maximum permutations exceeded"));
    assert!(display.contains("300"));
    assert!(display.contains("256"));
}

#[test]
fn permutation_error_compilation_failed_display() {
    let err = PermutationError::CompilationFailed("syntax error".to_string());
    let display = format!("{}", err);

    assert!(display.contains("compilation failed"));
    assert!(display.contains("syntax error"));
}

#[test]
fn permutation_error_shader_not_found_display() {
    let err = PermutationError::ShaderNotFound { shader_id: 42 };
    let display = format!("{}", err);

    assert!(display.contains("shader not found"));
    assert!(display.contains("42"));
}

#[test]
fn permutation_error_config_error_display() {
    let err = PermutationError::ConfigError("bad config".to_string());
    let display = format!("{}", err);

    assert!(display.contains("configuration error"));
    assert!(display.contains("bad config"));
}

#[test]
fn permutation_error_clone_equality() {
    let err1 = PermutationError::MaxPermutationsExceeded {
        current: 100,
        max: 50,
    };
    let err2 = err1.clone();

    assert_eq!(err1, err2);
}

#[test]
fn permutation_error_implements_std_error() {
    fn requires_error<T: std::error::Error>() {}
    requires_error::<PermutationError>();
}

// =============================================================================
// SECTION 9: ShaderPermutationManager Tests (Without Device)
// =============================================================================

#[test]
fn manager_new_with_default_config() {
    let manager = ShaderPermutationManager::new(PermutationConfig::default());

    assert!(manager.is_empty());
    assert_eq!(manager.permutation_count(), 0);
}

#[test]
fn manager_with_defaults_constructor() {
    let manager = ShaderPermutationManager::with_defaults();

    assert!(manager.is_empty());
    assert_eq!(manager.config().max_permutations, DEFAULT_MAX_PERMUTATIONS);
}

#[test]
fn manager_config_returns_configured_values() {
    let config = PermutationConfig::new()
        .max_permutations(100)
        .eviction_policy(EvictionPolicy::LFU);
    let manager = ShaderPermutationManager::new(config);

    assert_eq!(manager.config().max_permutations, 100);
    assert_eq!(manager.config().eviction_policy, EvictionPolicy::LFU);
}

#[test]
fn manager_metrics_initial_all_zero() {
    let manager = ShaderPermutationManager::with_defaults();
    let metrics = manager.metrics();

    assert_eq!(metrics.cache_size, 0);
    assert_eq!(metrics.cache_hits, 0);
    assert_eq!(metrics.cache_misses, 0);
    assert_eq!(metrics.compilations, 0);
    assert_eq!(metrics.evictions, 0);
}

#[test]
fn manager_reset_metrics_clears_counters() {
    let manager = ShaderPermutationManager::with_defaults();
    manager.reset_metrics();
    let metrics = manager.metrics();

    assert_eq!(metrics.cache_hits, 0);
    assert_eq!(metrics.cache_misses, 0);
}

#[test]
fn manager_keys_empty_initially() {
    let manager = ShaderPermutationManager::with_defaults();
    assert!(manager.keys().is_empty());
}

#[test]
fn manager_shader_ids_empty_initially() {
    let manager = ShaderPermutationManager::with_defaults();
    assert!(manager.shader_ids().is_empty());
}

#[test]
fn manager_invalidate_all_on_empty() {
    let manager = ShaderPermutationManager::with_defaults();
    manager.invalidate_all();

    assert!(manager.is_empty());
}

#[test]
fn manager_invalidate_nonexistent_shader() {
    let manager = ShaderPermutationManager::with_defaults();
    let count = manager.invalidate(99999);

    assert_eq!(count, 0);
}

#[test]
fn manager_invalidate_key_nonexistent() {
    let manager = ShaderPermutationManager::with_defaults();
    let key = PermutationKey::new(12345, FeatureFlags::SKINNED);

    assert!(!manager.invalidate_key(&key));
}

#[test]
fn manager_contains_false_for_empty() {
    let manager = ShaderPermutationManager::with_defaults();
    let key = PermutationKey::new(12345, FeatureFlags::SKINNED);

    assert!(!manager.contains(&key));
}

#[test]
fn manager_get_cached_none_for_empty() {
    let manager = ShaderPermutationManager::with_defaults();
    let key = PermutationKey::new(12345, FeatureFlags::SKINNED);

    assert!(manager.get_cached(&key).is_none());
}

#[test]
fn manager_permutation_count_for_shader_zero() {
    let manager = ShaderPermutationManager::with_defaults();
    assert_eq!(manager.permutation_count_for_shader(12345), 0);
}

#[test]
fn manager_debug_format() {
    let manager = ShaderPermutationManager::with_defaults();
    let debug = format!("{:?}", manager);

    assert!(debug.contains("ShaderPermutationManager"));
    assert!(debug.contains("cache_size"));
}

// =============================================================================
// SECTION 10: Thread Safety Tests
// =============================================================================

#[test]
fn feature_flags_is_send_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<FeatureFlags>();
}

#[test]
fn permutation_key_is_send_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<PermutationKey>();
}

#[test]
fn permutation_config_is_send_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<PermutationConfig>();
}

#[test]
fn eviction_policy_is_send_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<EvictionPolicy>();
}

#[test]
fn permutation_metrics_is_send_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<PermutationMetrics>();
}

#[test]
fn permutation_error_is_send_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<PermutationError>();
}

#[test]
fn manager_is_send_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<ShaderPermutationManager>();
}

#[test]
fn feature_flags_concurrent_creation() {
    use std::thread;

    let handles: Vec<_> = (0..10)
        .map(|i| {
            thread::spawn(move || {
                FeatureFlags::from_bits_truncate(i as u32)
            })
        })
        .collect();

    let results: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
    assert_eq!(results.len(), 10);
}

#[test]
fn permutation_key_concurrent_creation() {
    use std::thread;

    let handles: Vec<_> = (0..10)
        .map(|i| {
            thread::spawn(move || {
                PermutationKey::new(i as u64, FeatureFlags::from_bits_truncate(i as u32))
            })
        })
        .collect();

    let keys: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();

    // All keys should be different
    for i in 0..keys.len() {
        for j in (i + 1)..keys.len() {
            assert_ne!(keys[i], keys[j]);
        }
    }
}

// =============================================================================
// SECTION 11: Edge Case Tests
// =============================================================================

#[test]
fn feature_flags_zero_shader_id() {
    let key = PermutationKey::new(0, FeatureFlags::SKINNED);
    assert_eq!(key.shader_id(), 0);
}

#[test]
fn feature_flags_max_shader_id() {
    let key = PermutationKey::new(u64::MAX, FeatureFlags::ALL);
    assert_eq!(key.shader_id(), u64::MAX);
}

#[test]
fn feature_flags_from_bits_valid() {
    let flags = FeatureFlags::from_bits(0b0010101);
    assert!(flags.is_some());

    let flags = flags.unwrap();
    assert!(flags.contains(FeatureFlags::SKINNED));
    assert!(flags.contains(FeatureFlags::NORMAL_MAP));
    assert!(flags.contains(FeatureFlags::SHADOWS));
}

#[test]
fn feature_flags_from_bits_truncate() {
    let flags = FeatureFlags::from_bits_truncate(0b11111111);
    assert!(flags.is_all_set());
}

#[test]
fn feature_flags_bits_values_unique() {
    let bits = [
        FeatureFlags::SKINNED.bits(),
        FeatureFlags::ALPHA_TEST.bits(),
        FeatureFlags::NORMAL_MAP.bits(),
        FeatureFlags::EMISSIVE.bits(),
        FeatureFlags::SHADOWS.bits(),
        FeatureFlags::FOG.bits(),
        FeatureFlags::INSTANCED.bits(),
    ];

    let unique: HashSet<_> = bits.iter().collect();
    assert_eq!(unique.len(), 7);
}

#[test]
fn feature_flags_duplicate_names_deduplicated() {
    let flags = FeatureFlags::from_names(&["SKINNED", "SKINNED", "SKINNED"]);
    assert_eq!(flags.flag_count(), 1);
}

#[test]
fn permutation_metrics_large_values() {
    let metrics = PermutationMetrics::new(
        usize::MAX,
        u64::MAX / 2,
        u64::MAX / 2,
        u64::MAX,
        u64::MAX,
    );

    assert_eq!(metrics.cache_size, usize::MAX);
    assert_eq!(metrics.evictions, u64::MAX);
}

#[test]
fn permutation_config_large_max_permutations() {
    let config = PermutationConfig::new().max_permutations(usize::MAX);
    assert_eq!(config.max_permutations, usize::MAX);
    assert!(config.validate().is_ok());
}

// =============================================================================
// SECTION 12: Integration Pattern Tests
// =============================================================================

#[test]
fn integration_typical_material_system() {
    // Simulate how a material system would use permutation keys
    let pbr_shader_id: u64 = 1001;
    let unlit_shader_id: u64 = 1002;

    // Material variants
    let pbr_opaque = PermutationKey::new(
        pbr_shader_id,
        FeatureFlags::NORMAL_MAP | FeatureFlags::SHADOWS,
    );
    let pbr_skinned = PermutationKey::new(
        pbr_shader_id,
        FeatureFlags::NORMAL_MAP | FeatureFlags::SHADOWS | FeatureFlags::SKINNED,
    );
    let pbr_transparent = PermutationKey::new(
        pbr_shader_id,
        FeatureFlags::NORMAL_MAP | FeatureFlags::ALPHA_TEST,
    );
    let unlit_simple = PermutationKey::base(unlit_shader_id);

    // Verify they're all distinct
    let keys = [pbr_opaque, pbr_skinned, pbr_transparent, unlit_simple];
    for i in 0..keys.len() {
        for j in (i + 1)..keys.len() {
            assert_ne!(keys[i], keys[j]);
        }
    }
}

#[test]
fn integration_multiple_managers() {
    // Multiple managers can coexist (e.g., for different shader families)
    let opaque_manager = ShaderPermutationManager::new(
        PermutationConfig::new().max_permutations(256)
    );
    let transparent_manager = ShaderPermutationManager::new(
        PermutationConfig::new().max_permutations(128)
    );
    let post_process_manager = ShaderPermutationManager::new(
        PermutationConfig::new().max_permutations(32)
    );

    assert_eq!(opaque_manager.config().max_permutations, 256);
    assert_eq!(transparent_manager.config().max_permutations, 128);
    assert_eq!(post_process_manager.config().max_permutations, 32);
}

#[test]
fn integration_config_for_different_platforms() {
    // Mobile: fewer permutations
    let mobile_config = PermutationConfig::minimal();
    assert_eq!(mobile_config.max_permutations, 16);

    // Desktop development
    let dev_config = PermutationConfig::development();
    assert_eq!(dev_config.max_permutations, 64);

    // Desktop production
    let prod_config = PermutationConfig::production();
    assert_eq!(prod_config.max_permutations, 512);
}

#[test]
fn integration_feature_flag_workflow() {
    // Start with base features
    let mut features = FeatureFlags::NORMAL_MAP | FeatureFlags::SHADOWS;

    // Add skinning if mesh is skinned
    let is_skinned = true;
    if is_skinned {
        features = features.with_feature(FeatureFlags::SKINNED);
    }

    // Add fog if scene has fog
    let has_fog = false;
    if has_fog {
        features = features.with_feature(FeatureFlags::FOG);
    }

    // Verify final features
    assert!(features.contains(FeatureFlags::NORMAL_MAP));
    assert!(features.contains(FeatureFlags::SHADOWS));
    assert!(features.contains(FeatureFlags::SKINNED));
    assert!(!features.contains(FeatureFlags::FOG));
}

#[test]
fn integration_shader_lookup_pattern() {
    let mut cache: HashMap<PermutationKey, &str> = HashMap::new();

    // Pre-register common permutations
    let shader_id: u64 = 100;

    cache.insert(
        PermutationKey::base(shader_id),
        "base_shader",
    );
    cache.insert(
        PermutationKey::new(shader_id, FeatureFlags::SKINNED),
        "skinned_shader",
    );
    cache.insert(
        PermutationKey::new(shader_id, FeatureFlags::SKINNED | FeatureFlags::SHADOWS),
        "skinned_shadow_shader",
    );

    // Lookup by features
    let features = FeatureFlags::SKINNED | FeatureFlags::SHADOWS;
    let key = PermutationKey::new(shader_id, features);

    assert_eq!(cache.get(&key), Some(&"skinned_shadow_shader"));
}

#[test]
fn integration_enumerate_all_permutations() {
    // For a shader, enumerate all possible permutations
    let shader_id: u64 = 1;
    let total = FeatureFlags::total_permutations();

    let mut keys = Vec::with_capacity(total);
    for bits in 0..total {
        let features = FeatureFlags::from_bits_truncate(bits as u32);
        keys.push(PermutationKey::new(shader_id, features));
    }

    assert_eq!(keys.len(), 128);

    // Verify all unique
    let unique: HashSet<_> = keys.iter().map(|k| k.features().bits()).collect();
    assert_eq!(unique.len(), 128);
}

#[test]
fn integration_metrics_monitoring() {
    let manager = ShaderPermutationManager::with_defaults();

    // Initial state
    let metrics = manager.metrics();
    assert!(metrics.is_empty());
    assert_eq!(metrics.total_requests(), 0);

    // Would track after operations (can't test without device)
    // But verify metrics are queryable
    assert_eq!(metrics.hit_rate, 0.0);
}

// =============================================================================
// SECTION 13: Constants Tests
// =============================================================================

#[test]
fn constant_default_max_permutations() {
    assert_eq!(DEFAULT_MAX_PERMUTATIONS, 256);
}

#[test]
fn constant_feature_flag_count() {
    assert_eq!(FEATURE_FLAG_COUNT, 7);
}

#[test]
fn constant_total_permutations_matches() {
    assert_eq!(FeatureFlags::total_permutations(), 1 << FEATURE_FLAG_COUNT);
}

// =============================================================================
// SECTION 14: Pipeline Constants Integration Tests
// =============================================================================

#[test]
fn feature_flags_to_pipeline_constants_empty() {
    let flags = FeatureFlags::NONE;
    let constants = flags.to_pipeline_constants();

    // All features should be 0.0
    assert_eq!(constants.get("FEATURE_SKINNED"), Some(0.0));
    assert_eq!(constants.get("FEATURE_SHADOWS"), Some(0.0));
}

#[test]
fn feature_flags_to_pipeline_constants_some() {
    let flags = FeatureFlags::SKINNED | FeatureFlags::SHADOWS;
    let constants = flags.to_pipeline_constants();

    assert_eq!(constants.get("FEATURE_SKINNED"), Some(1.0));
    assert_eq!(constants.get("FEATURE_SHADOWS"), Some(1.0));
    assert_eq!(constants.get("FEATURE_FOG"), Some(0.0));
    assert_eq!(constants.get("FEATURE_INSTANCED"), Some(0.0));
}

#[test]
fn feature_flags_to_pipeline_constants_all() {
    let flags = FeatureFlags::ALL;
    let constants = flags.to_pipeline_constants();

    assert_eq!(constants.get("FEATURE_SKINNED"), Some(1.0));
    assert_eq!(constants.get("FEATURE_ALPHA_TEST"), Some(1.0));
    assert_eq!(constants.get("FEATURE_NORMAL_MAP"), Some(1.0));
    assert_eq!(constants.get("FEATURE_EMISSIVE"), Some(1.0));
    assert_eq!(constants.get("FEATURE_SHADOWS"), Some(1.0));
    assert_eq!(constants.get("FEATURE_FOG"), Some(1.0));
    assert_eq!(constants.get("FEATURE_INSTANCED"), Some(1.0));
}
