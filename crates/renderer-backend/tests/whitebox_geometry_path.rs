// WHITEBOX tests for T-WGPU-P6.9.3 (Geometry Path Abstraction)
//
// WHITEBOX discipline: These tests have FULL ACCESS to the implementation.
// They exercise internal code paths, branch conditions, trait implementations,
// and edge cases that are not visible through the public contract alone.
//
// Implementation under test: crates/renderer-backend/src/gpu_driven/geometry_path.rs
//   - GeometryPath: Enum with Traditional and Meshlet variants
//   - GeometryPathConfig: Configuration struct for path selection
//   - GeometryRenderable: Trait for renderable geometry
//   - TRADITIONAL_PATH_NAME: &str = "Traditional"
//   - MESHLET_PATH_NAME: &str = "Meshlet"
//
// WHITEBOX coverage plan:
//   - Path A: GeometryPath::default() returns Traditional
//   - Path B: GeometryPath::select() with empty features
//   - Path C: GeometryPath::select() with all features (still Traditional)
//   - Path D: GeometryPath::meshlet_available() always returns false
//   - Path E: GeometryPath::name() returns correct strings
//   - Path F: GeometryPath::is_traditional() checks
//   - Path G: GeometryPath::is_meshlet() checks
//   - Path H: GeometryPathConfig::default() values
//   - Path I: GeometryPathConfig::prefer_meshlet() factory
//   - Path J: GeometryPathConfig::force_traditional() factory
//   - Path K: GeometryPathConfig::resolve() with various inputs
//   - Path L: GeometryPathConfig::preferred_available() checks
//   - Path M: Clone trait for GeometryPath
//   - Path N: Copy trait for GeometryPath
//   - Path O: Debug trait for GeometryPath
//   - Path P: PartialEq and Eq traits for GeometryPath
//   - Path Q: Hash trait for GeometryPath
//   - Path R: Clone and Debug traits for GeometryPathConfig
//   - Path S: Constants TRADITIONAL_PATH_NAME and MESHLET_PATH_NAME
//   - Path T: GeometryRenderable::best_path() default implementation
//   - Path U: Enum exhaustiveness (match coverage)
//   - Path V: Various wgpu::Features combinations

use renderer_backend::gpu_driven::{
    GeometryPath, GeometryPathConfig, GeometryRenderable,
    TRADITIONAL_PATH_NAME, MESHLET_PATH_NAME,
};
use std::collections::HashSet;
use std::hash::Hash;

// ============================================================================
// Path S: Constants Verification
// ============================================================================

#[test]
fn test_traditional_path_name_constant() {
    assert_eq!(TRADITIONAL_PATH_NAME, "Traditional");
}

#[test]
fn test_meshlet_path_name_constant() {
    assert_eq!(MESHLET_PATH_NAME, "Meshlet");
}

#[test]
fn test_constants_are_static_str() {
    // Verify constants can be used in const contexts
    const _T: &str = TRADITIONAL_PATH_NAME;
    const _M: &str = MESHLET_PATH_NAME;
    assert!(!_T.is_empty());
    assert!(!_M.is_empty());
}

// ============================================================================
// Path A: GeometryPath::default() Tests
// ============================================================================

#[test]
fn test_geometry_path_default_is_traditional() {
    let path = GeometryPath::default();
    assert_eq!(path, GeometryPath::Traditional);
}

#[test]
fn test_geometry_path_default_not_meshlet() {
    let path = GeometryPath::default();
    assert_ne!(path, GeometryPath::Meshlet);
}

#[test]
fn test_geometry_path_default_is_traditional_method() {
    let path = GeometryPath::default();
    assert!(path.is_traditional());
}

#[test]
fn test_geometry_path_default_is_not_meshlet_method() {
    let path = GeometryPath::default();
    assert!(!path.is_meshlet());
}

// ============================================================================
// Path B-C: GeometryPath::select() Tests
// ============================================================================

#[test]
fn test_geometry_path_select_empty_features() {
    // Path B: With empty features, should select Traditional
    let features = wgpu::Features::empty();
    let path = GeometryPath::select(features);
    assert_eq!(path, GeometryPath::Traditional);
}

#[test]
fn test_geometry_path_select_all_features() {
    // Path C: Even with all features, should select Traditional
    // (mesh shaders not yet stable in wgpu)
    let features = wgpu::Features::all();
    let path = GeometryPath::select(features);
    assert_eq!(path, GeometryPath::Traditional);
}

#[test]
fn test_geometry_path_select_common_features() {
    // Test with common feature sets
    let features = wgpu::Features::TEXTURE_COMPRESSION_BC;
    let path = GeometryPath::select(features);
    assert_eq!(path, GeometryPath::Traditional);
}

#[test]
fn test_geometry_path_select_multiple_features() {
    // Test with multiple combined features
    let features = wgpu::Features::TEXTURE_COMPRESSION_BC
        | wgpu::Features::DEPTH_CLIP_CONTROL
        | wgpu::Features::PUSH_CONSTANTS;
    let path = GeometryPath::select(features);
    assert_eq!(path, GeometryPath::Traditional);
}

// ============================================================================
// Path D: GeometryPath::meshlet_available() Tests
// ============================================================================

#[test]
fn test_meshlet_available_empty_features() {
    let features = wgpu::Features::empty();
    assert!(!GeometryPath::meshlet_available(features));
}

#[test]
fn test_meshlet_available_all_features() {
    let features = wgpu::Features::all();
    // Currently always false since mesh shaders not stable
    assert!(!GeometryPath::meshlet_available(features));
}

#[test]
fn test_meshlet_available_single_feature() {
    let features = wgpu::Features::PUSH_CONSTANTS;
    assert!(!GeometryPath::meshlet_available(features));
}

#[test]
fn test_meshlet_available_compute_features() {
    // Even compute-related features don't enable meshlet
    let features = wgpu::Features::TIMESTAMP_QUERY
        | wgpu::Features::INDIRECT_FIRST_INSTANCE;
    assert!(!GeometryPath::meshlet_available(features));
}

// ============================================================================
// Path E: GeometryPath::name() Tests
// ============================================================================

#[test]
fn test_geometry_path_name_traditional() {
    assert_eq!(GeometryPath::Traditional.name(), "Traditional");
}

#[test]
fn test_geometry_path_name_meshlet() {
    assert_eq!(GeometryPath::Meshlet.name(), "Meshlet");
}

#[test]
fn test_geometry_path_name_matches_constants() {
    assert_eq!(GeometryPath::Traditional.name(), TRADITIONAL_PATH_NAME);
    assert_eq!(GeometryPath::Meshlet.name(), MESHLET_PATH_NAME);
}

#[test]
fn test_geometry_path_name_is_const() {
    // Verify name() can be used in const contexts
    const _T: &str = GeometryPath::Traditional.name();
    const _M: &str = GeometryPath::Meshlet.name();
    assert_eq!(_T, "Traditional");
    assert_eq!(_M, "Meshlet");
}

// ============================================================================
// Path F: GeometryPath::is_traditional() Tests
// ============================================================================

#[test]
fn test_is_traditional_for_traditional() {
    assert!(GeometryPath::Traditional.is_traditional());
}

#[test]
fn test_is_traditional_for_meshlet() {
    assert!(!GeometryPath::Meshlet.is_traditional());
}

#[test]
fn test_is_traditional_is_const() {
    // Verify is_traditional() can be used in const contexts
    const IS_TRAD: bool = GeometryPath::Traditional.is_traditional();
    const IS_MESH_TRAD: bool = GeometryPath::Meshlet.is_traditional();
    assert!(IS_TRAD);
    assert!(!IS_MESH_TRAD);
}

// ============================================================================
// Path G: GeometryPath::is_meshlet() Tests
// ============================================================================

#[test]
fn test_is_meshlet_for_meshlet() {
    assert!(GeometryPath::Meshlet.is_meshlet());
}

#[test]
fn test_is_meshlet_for_traditional() {
    assert!(!GeometryPath::Traditional.is_meshlet());
}

#[test]
fn test_is_meshlet_is_const() {
    // Verify is_meshlet() can be used in const contexts
    const IS_MESH: bool = GeometryPath::Meshlet.is_meshlet();
    const IS_TRAD_MESH: bool = GeometryPath::Traditional.is_meshlet();
    assert!(IS_MESH);
    assert!(!IS_TRAD_MESH);
}

#[test]
fn test_is_traditional_and_is_meshlet_mutually_exclusive() {
    // For any path, exactly one of is_traditional() or is_meshlet() is true
    let paths = [GeometryPath::Traditional, GeometryPath::Meshlet];
    for path in paths {
        let t = path.is_traditional();
        let m = path.is_meshlet();
        assert!(t ^ m, "Path {:?} should be exactly one of traditional or meshlet", path);
    }
}

// ============================================================================
// Path H: GeometryPathConfig::default() Tests
// ============================================================================

#[test]
fn test_geometry_path_config_default_preferred() {
    let config = GeometryPathConfig::default();
    assert_eq!(config.preferred, GeometryPath::Traditional);
}

#[test]
fn test_geometry_path_config_default_force_traditional() {
    let config = GeometryPathConfig::default();
    assert!(!config.force_traditional);
}

#[test]
fn test_geometry_path_config_default_fields() {
    let config = GeometryPathConfig::default();
    assert_eq!(config.preferred, GeometryPath::Traditional);
    assert!(!config.force_traditional);
}

// ============================================================================
// Path I: GeometryPathConfig::prefer_meshlet() Tests
// ============================================================================

#[test]
fn test_geometry_path_config_prefer_meshlet_preferred() {
    let config = GeometryPathConfig::prefer_meshlet();
    assert_eq!(config.preferred, GeometryPath::Meshlet);
}

#[test]
fn test_geometry_path_config_prefer_meshlet_force_traditional() {
    let config = GeometryPathConfig::prefer_meshlet();
    assert!(!config.force_traditional);
}

#[test]
fn test_geometry_path_config_prefer_meshlet_fields() {
    let config = GeometryPathConfig::prefer_meshlet();
    assert_eq!(config.preferred, GeometryPath::Meshlet);
    assert!(!config.force_traditional);
}

// ============================================================================
// Path J: GeometryPathConfig::force_traditional() Tests
// ============================================================================

#[test]
fn test_geometry_path_config_force_traditional_preferred() {
    let config = GeometryPathConfig::force_traditional();
    assert_eq!(config.preferred, GeometryPath::Traditional);
}

#[test]
fn test_geometry_path_config_force_traditional_flag() {
    let config = GeometryPathConfig::force_traditional();
    assert!(config.force_traditional);
}

#[test]
fn test_geometry_path_config_force_traditional_fields() {
    let config = GeometryPathConfig::force_traditional();
    assert_eq!(config.preferred, GeometryPath::Traditional);
    assert!(config.force_traditional);
}

// ============================================================================
// Path K: GeometryPathConfig::resolve() Tests
// ============================================================================

#[test]
fn test_config_resolve_default_empty_features() {
    let config = GeometryPathConfig::default();
    let features = wgpu::Features::empty();
    let path = config.resolve(features);
    assert_eq!(path, GeometryPath::Traditional);
}

#[test]
fn test_config_resolve_default_all_features() {
    let config = GeometryPathConfig::default();
    let features = wgpu::Features::all();
    let path = config.resolve(features);
    assert_eq!(path, GeometryPath::Traditional);
}

#[test]
fn test_config_resolve_prefer_meshlet_empty_features() {
    // Even if we prefer meshlet, falls back to traditional
    let config = GeometryPathConfig::prefer_meshlet();
    let features = wgpu::Features::empty();
    let path = config.resolve(features);
    assert_eq!(path, GeometryPath::Traditional);
}

#[test]
fn test_config_resolve_prefer_meshlet_all_features() {
    // Even with all features, meshlet is not available
    let config = GeometryPathConfig::prefer_meshlet();
    let features = wgpu::Features::all();
    let path = config.resolve(features);
    assert_eq!(path, GeometryPath::Traditional);
}

#[test]
fn test_config_resolve_force_traditional_empty_features() {
    let config = GeometryPathConfig::force_traditional();
    let features = wgpu::Features::empty();
    let path = config.resolve(features);
    assert_eq!(path, GeometryPath::Traditional);
}

#[test]
fn test_config_resolve_force_traditional_all_features() {
    let config = GeometryPathConfig::force_traditional();
    let features = wgpu::Features::all();
    let path = config.resolve(features);
    assert_eq!(path, GeometryPath::Traditional);
}

#[test]
fn test_config_resolve_force_traditional_overrides_prefer_meshlet() {
    // Force traditional should override even if meshlet is preferred
    let config = GeometryPathConfig {
        preferred: GeometryPath::Meshlet,
        force_traditional: true,
    };
    let features = wgpu::Features::all();
    let path = config.resolve(features);
    assert_eq!(path, GeometryPath::Traditional);
}

#[test]
fn test_config_resolve_traditional_preferred_returns_traditional() {
    let config = GeometryPathConfig {
        preferred: GeometryPath::Traditional,
        force_traditional: false,
    };
    let features = wgpu::Features::all();
    let path = config.resolve(features);
    assert_eq!(path, GeometryPath::Traditional);
}

// ============================================================================
// Path L: GeometryPathConfig::preferred_available() Tests
// ============================================================================

#[test]
fn test_config_preferred_available_traditional_empty() {
    let config = GeometryPathConfig::default();
    let features = wgpu::Features::empty();
    // Traditional is always available
    assert!(config.preferred_available(features));
}

#[test]
fn test_config_preferred_available_traditional_all() {
    let config = GeometryPathConfig::default();
    let features = wgpu::Features::all();
    // Traditional is always available
    assert!(config.preferred_available(features));
}

#[test]
fn test_config_preferred_available_meshlet_empty() {
    let config = GeometryPathConfig::prefer_meshlet();
    let features = wgpu::Features::empty();
    // Meshlet is not available
    assert!(!config.preferred_available(features));
}

#[test]
fn test_config_preferred_available_meshlet_all() {
    let config = GeometryPathConfig::prefer_meshlet();
    let features = wgpu::Features::all();
    // Meshlet is still not available (not stable in wgpu)
    assert!(!config.preferred_available(features));
}

#[test]
fn test_config_preferred_available_force_traditional() {
    // force_traditional doesn't affect preferred_available
    let config = GeometryPathConfig::force_traditional();
    let features = wgpu::Features::empty();
    assert!(config.preferred_available(features));
}

// ============================================================================
// Path M: Clone Trait for GeometryPath
// ============================================================================

#[test]
fn test_geometry_path_clone_traditional() {
    let path = GeometryPath::Traditional;
    let cloned = path.clone();
    assert_eq!(path, cloned);
}

#[test]
fn test_geometry_path_clone_meshlet() {
    let path = GeometryPath::Meshlet;
    let cloned = path.clone();
    assert_eq!(path, cloned);
}

#[test]
fn test_geometry_path_clone_independence() {
    let path1 = GeometryPath::Traditional;
    let path2 = path1.clone();
    // Modifying one shouldn't affect the other (trivial for Copy types)
    assert_eq!(path1, path2);
}

// ============================================================================
// Path N: Copy Trait for GeometryPath
// ============================================================================

#[test]
fn test_geometry_path_copy_traditional() {
    let path = GeometryPath::Traditional;
    let copied = path;
    // Original is still usable (Copy trait)
    assert_eq!(path, GeometryPath::Traditional);
    assert_eq!(copied, GeometryPath::Traditional);
}

#[test]
fn test_geometry_path_copy_meshlet() {
    let path = GeometryPath::Meshlet;
    let copied = path;
    assert_eq!(path, GeometryPath::Meshlet);
    assert_eq!(copied, GeometryPath::Meshlet);
}

#[test]
fn test_geometry_path_copy_to_function() {
    fn take_copy(p: GeometryPath) -> GeometryPath {
        p
    }
    let path = GeometryPath::Traditional;
    let result = take_copy(path);
    // Original still usable after passing to function
    assert_eq!(path, result);
}

// ============================================================================
// Path O: Debug Trait for GeometryPath
// ============================================================================

#[test]
fn test_geometry_path_debug_traditional() {
    let path = GeometryPath::Traditional;
    let debug_str = format!("{:?}", path);
    assert_eq!(debug_str, "Traditional");
}

#[test]
fn test_geometry_path_debug_meshlet() {
    let path = GeometryPath::Meshlet;
    let debug_str = format!("{:?}", path);
    assert_eq!(debug_str, "Meshlet");
}

#[test]
fn test_geometry_path_debug_contains_variant() {
    let path = GeometryPath::Traditional;
    let debug_str = format!("{:?}", path);
    assert!(debug_str.contains("Traditional"));

    let path = GeometryPath::Meshlet;
    let debug_str = format!("{:?}", path);
    assert!(debug_str.contains("Meshlet"));
}

// ============================================================================
// Path P: PartialEq and Eq Traits for GeometryPath
// ============================================================================

#[test]
fn test_geometry_path_eq_same_variant() {
    assert_eq!(GeometryPath::Traditional, GeometryPath::Traditional);
    assert_eq!(GeometryPath::Meshlet, GeometryPath::Meshlet);
}

#[test]
fn test_geometry_path_ne_different_variant() {
    assert_ne!(GeometryPath::Traditional, GeometryPath::Meshlet);
    assert_ne!(GeometryPath::Meshlet, GeometryPath::Traditional);
}

#[test]
fn test_geometry_path_eq_reflexive() {
    let path = GeometryPath::Traditional;
    assert_eq!(path, path);
}

#[test]
fn test_geometry_path_eq_symmetric() {
    let a = GeometryPath::Traditional;
    let b = GeometryPath::Traditional;
    assert_eq!(a == b, b == a);
}

#[test]
fn test_geometry_path_eq_transitive() {
    let a = GeometryPath::Meshlet;
    let b = GeometryPath::Meshlet;
    let c = GeometryPath::Meshlet;
    assert!(a == b && b == c && a == c);
}

// ============================================================================
// Path Q: Hash Trait for GeometryPath
// ============================================================================

#[test]
fn test_geometry_path_hash_in_hashset() {
    let mut set = HashSet::new();
    set.insert(GeometryPath::Traditional);
    set.insert(GeometryPath::Meshlet);
    set.insert(GeometryPath::Traditional); // Duplicate

    assert_eq!(set.len(), 2);
    assert!(set.contains(&GeometryPath::Traditional));
    assert!(set.contains(&GeometryPath::Meshlet));
}

#[test]
fn test_geometry_path_hash_consistent() {
    use std::hash::{Hash, Hasher};
    use std::collections::hash_map::DefaultHasher;

    fn hash_value<T: Hash>(t: &T) -> u64 {
        let mut s = DefaultHasher::new();
        t.hash(&mut s);
        s.finish()
    }

    let path1 = GeometryPath::Traditional;
    let path2 = GeometryPath::Traditional;
    assert_eq!(hash_value(&path1), hash_value(&path2));
}

#[test]
fn test_geometry_path_hash_different_variants() {
    use std::hash::{Hash, Hasher};
    use std::collections::hash_map::DefaultHasher;

    fn hash_value<T: Hash>(t: &T) -> u64 {
        let mut s = DefaultHasher::new();
        t.hash(&mut s);
        s.finish()
    }

    let hash_trad = hash_value(&GeometryPath::Traditional);
    let hash_mesh = hash_value(&GeometryPath::Meshlet);
    assert_ne!(hash_trad, hash_mesh);
}

// ============================================================================
// Path R: Clone and Debug Traits for GeometryPathConfig
// ============================================================================

#[test]
fn test_geometry_path_config_clone() {
    let config = GeometryPathConfig::prefer_meshlet();
    let cloned = config.clone();
    assert_eq!(cloned.preferred, config.preferred);
    assert_eq!(cloned.force_traditional, config.force_traditional);
}

#[test]
fn test_geometry_path_config_clone_independence() {
    let config = GeometryPathConfig::default();
    let cloned = config.clone();
    // Both should have same values
    assert_eq!(config.preferred, cloned.preferred);
    assert_eq!(config.force_traditional, cloned.force_traditional);
}

#[test]
fn test_geometry_path_config_debug_contains_fields() {
    let config = GeometryPathConfig::default();
    let debug_str = format!("{:?}", config);
    assert!(debug_str.contains("GeometryPathConfig"));
    assert!(debug_str.contains("preferred"));
    assert!(debug_str.contains("Traditional"));
    assert!(debug_str.contains("force_traditional"));
}

#[test]
fn test_geometry_path_config_debug_prefer_meshlet() {
    let config = GeometryPathConfig::prefer_meshlet();
    let debug_str = format!("{:?}", config);
    assert!(debug_str.contains("Meshlet"));
    assert!(debug_str.contains("false"));
}

#[test]
fn test_geometry_path_config_debug_force_traditional() {
    let config = GeometryPathConfig::force_traditional();
    let debug_str = format!("{:?}", config);
    assert!(debug_str.contains("Traditional"));
    assert!(debug_str.contains("true"));
}

// ============================================================================
// Path T: GeometryRenderable Trait Tests
// ============================================================================

/// Mock geometry that supports both paths
struct MockGeometryBoth;

impl GeometryRenderable for MockGeometryBoth {
    fn render(&self, _path: GeometryPath, _render_pass: &mut wgpu::RenderPass) {
        // No-op for testing
    }

    fn supports_path(&self, _path: GeometryPath) -> bool {
        true
    }
}

/// Mock geometry that only supports traditional
struct MockGeometryTraditionalOnly;

impl GeometryRenderable for MockGeometryTraditionalOnly {
    fn render(&self, _path: GeometryPath, _render_pass: &mut wgpu::RenderPass) {
        // No-op for testing
    }

    fn supports_path(&self, path: GeometryPath) -> bool {
        matches!(path, GeometryPath::Traditional)
    }
}

/// Mock geometry that supports neither (edge case)
struct MockGeometryNone;

impl GeometryRenderable for MockGeometryNone {
    fn render(&self, _path: GeometryPath, _render_pass: &mut wgpu::RenderPass) {
        // No-op for testing
    }

    fn supports_path(&self, _path: GeometryPath) -> bool {
        false
    }
}

#[test]
fn test_geometry_renderable_best_path_both() {
    let geom = MockGeometryBoth;
    // When both are supported, best_path returns Meshlet
    assert_eq!(geom.best_path(), GeometryPath::Meshlet);
}

#[test]
fn test_geometry_renderable_best_path_traditional_only() {
    let geom = MockGeometryTraditionalOnly;
    // When only traditional is supported, returns Traditional
    assert_eq!(geom.best_path(), GeometryPath::Traditional);
}

#[test]
fn test_geometry_renderable_best_path_none() {
    let geom = MockGeometryNone;
    // When neither is supported, returns Traditional (fallback)
    assert_eq!(geom.best_path(), GeometryPath::Traditional);
}

#[test]
fn test_geometry_renderable_supports_path_both() {
    let geom = MockGeometryBoth;
    assert!(geom.supports_path(GeometryPath::Traditional));
    assert!(geom.supports_path(GeometryPath::Meshlet));
}

#[test]
fn test_geometry_renderable_supports_path_traditional_only() {
    let geom = MockGeometryTraditionalOnly;
    assert!(geom.supports_path(GeometryPath::Traditional));
    assert!(!geom.supports_path(GeometryPath::Meshlet));
}

// ============================================================================
// Path U: Enum Exhaustiveness Tests
// ============================================================================

#[test]
fn test_geometry_path_exhaustive_match() {
    fn describe_path(path: GeometryPath) -> &'static str {
        match path {
            GeometryPath::Traditional => "traditional",
            GeometryPath::Meshlet => "meshlet",
        }
    }

    assert_eq!(describe_path(GeometryPath::Traditional), "traditional");
    assert_eq!(describe_path(GeometryPath::Meshlet), "meshlet");
}

#[test]
fn test_geometry_path_all_variants() {
    let variants = [GeometryPath::Traditional, GeometryPath::Meshlet];
    assert_eq!(variants.len(), 2);

    // Each variant is unique
    let set: HashSet<_> = variants.iter().collect();
    assert_eq!(set.len(), 2);
}

// ============================================================================
// Path V: Various wgpu::Features Combinations
// ============================================================================

#[test]
fn test_select_with_texture_features() {
    let features = wgpu::Features::TEXTURE_COMPRESSION_BC
        | wgpu::Features::TEXTURE_COMPRESSION_ETC2
        | wgpu::Features::TEXTURE_COMPRESSION_ASTC;
    let path = GeometryPath::select(features);
    assert_eq!(path, GeometryPath::Traditional);
}

#[test]
fn test_select_with_bindless_features() {
    let features = wgpu::Features::TEXTURE_BINDING_ARRAY
        | wgpu::Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING;
    let path = GeometryPath::select(features);
    assert_eq!(path, GeometryPath::Traditional);
}

#[test]
fn test_select_with_compute_features() {
    let features = wgpu::Features::TIMESTAMP_QUERY
        | wgpu::Features::PIPELINE_STATISTICS_QUERY;
    let path = GeometryPath::select(features);
    assert_eq!(path, GeometryPath::Traditional);
}

#[test]
fn test_select_with_render_features() {
    let features = wgpu::Features::DEPTH_CLIP_CONTROL
        | wgpu::Features::CONSERVATIVE_RASTERIZATION;
    let path = GeometryPath::select(features);
    assert_eq!(path, GeometryPath::Traditional);
}

#[test]
fn test_meshlet_available_with_various_features() {
    // Test multiple feature combinations
    let feature_sets = [
        wgpu::Features::empty(),
        wgpu::Features::TEXTURE_COMPRESSION_BC,
        wgpu::Features::DEPTH_CLIP_CONTROL,
        wgpu::Features::PUSH_CONSTANTS,
        wgpu::Features::MULTI_DRAW_INDIRECT,
        wgpu::Features::all(),
    ];

    for features in feature_sets {
        // Meshlet should never be available with current wgpu
        assert!(!GeometryPath::meshlet_available(features));
    }
}

// ============================================================================
// Additional Edge Cases
// ============================================================================

#[test]
fn test_config_custom_construction() {
    // Direct struct construction
    let config = GeometryPathConfig {
        preferred: GeometryPath::Meshlet,
        force_traditional: false,
    };
    assert_eq!(config.preferred, GeometryPath::Meshlet);
    assert!(!config.force_traditional);
}

#[test]
fn test_config_contradictory_settings() {
    // Prefer meshlet but force traditional - force wins
    let config = GeometryPathConfig {
        preferred: GeometryPath::Meshlet,
        force_traditional: true,
    };
    let features = wgpu::Features::all();
    let path = config.resolve(features);
    assert_eq!(path, GeometryPath::Traditional);
}

#[test]
fn test_path_in_vec() {
    let paths = vec![
        GeometryPath::Traditional,
        GeometryPath::Meshlet,
        GeometryPath::Traditional,
    ];
    assert_eq!(paths.len(), 3);
    assert_eq!(paths[0], GeometryPath::Traditional);
    assert_eq!(paths[1], GeometryPath::Meshlet);
    assert_eq!(paths[2], GeometryPath::Traditional);
}

#[test]
fn test_path_as_key_in_hashmap() {
    use std::collections::HashMap;

    let mut map = HashMap::new();
    map.insert(GeometryPath::Traditional, "vertex/index buffers");
    map.insert(GeometryPath::Meshlet, "mesh shaders");

    assert_eq!(map.get(&GeometryPath::Traditional), Some(&"vertex/index buffers"));
    assert_eq!(map.get(&GeometryPath::Meshlet), Some(&"mesh shaders"));
}

#[test]
fn test_config_resolve_consistency() {
    // Multiple calls to resolve should return the same result
    let config = GeometryPathConfig::prefer_meshlet();
    let features = wgpu::Features::all();

    let path1 = config.resolve(features);
    let path2 = config.resolve(features);
    let path3 = config.resolve(features);

    assert_eq!(path1, path2);
    assert_eq!(path2, path3);
}

#[test]
fn test_name_method_for_all_variants() {
    for path in [GeometryPath::Traditional, GeometryPath::Meshlet] {
        let name = path.name();
        assert!(!name.is_empty());
        // Name should match the debug output
        let debug = format!("{:?}", path);
        assert_eq!(name, debug);
    }
}

// ============================================================================
// Integration-style Whitebox Tests
// ============================================================================

#[test]
fn test_full_path_selection_flow() {
    // Simulate the full path selection flow

    // 1. Check device features (empty for this test)
    let features = wgpu::Features::empty();

    // 2. Create config with user preference
    let config = GeometryPathConfig::prefer_meshlet();

    // 3. Check if preferred is available
    let available = config.preferred_available(features);
    assert!(!available, "Meshlet should not be available");

    // 4. Resolve to actual path
    let path = config.resolve(features);
    assert_eq!(path, GeometryPath::Traditional, "Should fall back to Traditional");

    // 5. Verify path properties
    assert!(path.is_traditional());
    assert!(!path.is_meshlet());
    assert_eq!(path.name(), "Traditional");
}

#[test]
fn test_path_selection_with_force_override() {
    // Even if hypothetically meshlet were available, force_traditional overrides
    let features = wgpu::Features::all();

    // Config prefers meshlet but forces traditional
    let config = GeometryPathConfig {
        preferred: GeometryPath::Meshlet,
        force_traditional: true,
    };

    // Force should override
    let path = config.resolve(features);
    assert_eq!(path, GeometryPath::Traditional);
}
