// SPDX-License-Identifier: MIT
//
// blackbox_geometry_path.rs -- Blackbox tests for T-WGPU-P6.9.3 GeometryPath.
//
// CLEANROOM: No src/ access beyond the public API exported by the crate.
// Tests use only the documented public types and functions:
//
//   - GeometryPath -- Enum for path selection (Traditional/Meshlet)
//   - GeometryPathConfig -- Configuration for path selection
//   - GeometryRenderable -- Trait for renderable geometry
//
// CONSTANTS:
//   - TRADITIONAL_PATH_NAME, MESHLET_PATH_NAME
//
// ACCEPTANCE CRITERIA (T-WGPU-P6.9.3):
//   1. GeometryPath enum with Traditional and Meshlet variants
//   2. Path selection based on device features
//   3. Config-based resolution with fallback
//   4. Force traditional override
//   5. GeometryRenderable trait for dual-path geometry
//   6. Correct name constants
//
// TEST CATEGORIES:
//   1. API Tests - Public interface, enum variants
//   2. Path Selection - select() behavior with features
//   3. Configuration - Default, prefer_meshlet, force_traditional
//   4. Config Resolution - resolve() with fallback logic
//   5. Trait Verification - Clone, Copy, Debug, Hash, Eq, Default
//   6. Constants - Path name matching
//   7. Integration - Multiple paths comparison, HashSet behavior
//   8. Future-proofing - meshlet_available behavior
//
// Total target: 35+ tests

use renderer_backend::gpu_driven::{
    GeometryPath, GeometryPathConfig, GeometryRenderable,
    MESHLET_PATH_NAME, TRADITIONAL_PATH_NAME,
};
use std::collections::HashSet;

// =============================================================================
// CATEGORY 1: API TESTS - Public Interface
// =============================================================================

mod api_tests {
    use super::*;

    #[test]
    fn test_geometry_path_enum_is_public() {
        // Verify GeometryPath enum is accessible
        let _traditional = GeometryPath::Traditional;
        let _meshlet = GeometryPath::Meshlet;
    }

    #[test]
    fn test_geometry_path_config_is_public() {
        // Verify GeometryPathConfig struct is accessible
        let config = GeometryPathConfig::default();
        assert_eq!(config.preferred, GeometryPath::Traditional);
    }

    #[test]
    fn test_traditional_path_name_constant() {
        // Verify TRADITIONAL_PATH_NAME constant is accessible
        assert_eq!(TRADITIONAL_PATH_NAME, "Traditional");
    }

    #[test]
    fn test_meshlet_path_name_constant() {
        // Verify MESHLET_PATH_NAME constant is accessible
        assert_eq!(MESHLET_PATH_NAME, "Meshlet");
    }

    #[test]
    fn test_geometry_path_traditional_variant() {
        let path = GeometryPath::Traditional;
        assert!(path.is_traditional());
        assert!(!path.is_meshlet());
    }

    #[test]
    fn test_geometry_path_meshlet_variant() {
        let path = GeometryPath::Meshlet;
        assert!(!path.is_traditional());
        assert!(path.is_meshlet());
    }

    #[test]
    fn test_geometry_path_name_method() {
        assert_eq!(GeometryPath::Traditional.name(), "Traditional");
        assert_eq!(GeometryPath::Meshlet.name(), "Meshlet");
    }

    #[test]
    fn test_path_name_matches_constant_traditional() {
        assert_eq!(GeometryPath::Traditional.name(), TRADITIONAL_PATH_NAME);
    }

    #[test]
    fn test_path_name_matches_constant_meshlet() {
        assert_eq!(GeometryPath::Meshlet.name(), MESHLET_PATH_NAME);
    }
}

// =============================================================================
// CATEGORY 2: PATH SELECTION - select() Behavior
// =============================================================================

mod path_selection_tests {
    use super::*;

    #[test]
    fn test_select_with_empty_features() {
        let features = wgpu::Features::empty();
        let path = GeometryPath::select(features);
        assert_eq!(path, GeometryPath::Traditional);
    }

    #[test]
    fn test_select_with_all_features() {
        // Even with all features, meshlet is not yet available
        let features = wgpu::Features::all();
        let path = GeometryPath::select(features);
        assert_eq!(path, GeometryPath::Traditional);
    }

    #[test]
    fn test_meshlet_not_available_empty_features() {
        let features = wgpu::Features::empty();
        assert!(!GeometryPath::meshlet_available(features));
    }

    #[test]
    fn test_meshlet_not_available_all_features() {
        // Currently stubbed to always return false
        let features = wgpu::Features::all();
        assert!(!GeometryPath::meshlet_available(features));
    }

    #[test]
    fn test_select_returns_traditional_consistently() {
        // Multiple calls should return same result
        let features = wgpu::Features::empty();
        for _ in 0..10 {
            let path = GeometryPath::select(features);
            assert_eq!(path, GeometryPath::Traditional);
        }
    }

    #[test]
    fn test_select_with_various_features() {
        // Test with various feature combinations
        let feature_sets = [
            wgpu::Features::empty(),
            wgpu::Features::DEPTH_CLIP_CONTROL,
            wgpu::Features::TEXTURE_COMPRESSION_BC,
            wgpu::Features::MULTI_DRAW_INDIRECT,
        ];

        for features in feature_sets {
            let path = GeometryPath::select(features);
            // All should return Traditional since meshlet not available
            assert_eq!(path, GeometryPath::Traditional);
        }
    }
}

// =============================================================================
// CATEGORY 3: CONFIGURATION - Default and Constructors
// =============================================================================

mod config_tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = GeometryPathConfig::default();
        assert_eq!(config.preferred, GeometryPath::Traditional);
        assert!(!config.force_traditional);
    }

    #[test]
    fn test_config_prefer_meshlet() {
        let config = GeometryPathConfig::prefer_meshlet();
        assert_eq!(config.preferred, GeometryPath::Meshlet);
        assert!(!config.force_traditional);
    }

    #[test]
    fn test_config_force_traditional() {
        let config = GeometryPathConfig::force_traditional();
        assert_eq!(config.preferred, GeometryPath::Traditional);
        assert!(config.force_traditional);
    }

    #[test]
    fn test_config_fields_accessible() {
        // Verify struct fields are public
        let config = GeometryPathConfig {
            preferred: GeometryPath::Meshlet,
            force_traditional: true,
        };
        assert_eq!(config.preferred, GeometryPath::Meshlet);
        assert!(config.force_traditional);
    }

    #[test]
    fn test_config_custom_combination() {
        // Prefer meshlet but force traditional (force wins)
        let config = GeometryPathConfig {
            preferred: GeometryPath::Meshlet,
            force_traditional: true,
        };
        assert_eq!(config.preferred, GeometryPath::Meshlet);
        assert!(config.force_traditional);
    }
}

// =============================================================================
// CATEGORY 4: CONFIG RESOLUTION - resolve() Logic
// =============================================================================

mod resolution_tests {
    use super::*;

    #[test]
    fn test_resolve_default_config() {
        let config = GeometryPathConfig::default();
        let features = wgpu::Features::empty();
        let path = config.resolve(features);
        assert_eq!(path, GeometryPath::Traditional);
    }

    #[test]
    fn test_resolve_prefer_meshlet_falls_back() {
        // Meshlet not available, should fall back
        let config = GeometryPathConfig::prefer_meshlet();
        let features = wgpu::Features::all();
        let path = config.resolve(features);
        assert_eq!(path, GeometryPath::Traditional);
    }

    #[test]
    fn test_resolve_force_traditional_overrides() {
        // Force traditional even if we prefer meshlet
        let config = GeometryPathConfig {
            preferred: GeometryPath::Meshlet,
            force_traditional: true,
        };
        let features = wgpu::Features::all();
        let path = config.resolve(features);
        assert_eq!(path, GeometryPath::Traditional);
    }

    #[test]
    fn test_resolve_force_traditional_with_empty_features() {
        let config = GeometryPathConfig::force_traditional();
        let features = wgpu::Features::empty();
        let path = config.resolve(features);
        assert_eq!(path, GeometryPath::Traditional);
    }

    #[test]
    fn test_resolve_prefer_traditional_no_fallback_needed() {
        let config = GeometryPathConfig::default(); // prefers Traditional
        let features = wgpu::Features::empty();
        let path = config.resolve(features);
        assert_eq!(path, GeometryPath::Traditional);
    }

    #[test]
    fn test_preferred_available_traditional() {
        let config = GeometryPathConfig::default();
        let features = wgpu::Features::empty();
        // Traditional is always available
        assert!(config.preferred_available(features));
    }

    #[test]
    fn test_preferred_available_meshlet() {
        let config = GeometryPathConfig::prefer_meshlet();
        let features = wgpu::Features::all();
        // Meshlet is not available
        assert!(!config.preferred_available(features));
    }

    #[test]
    fn test_preferred_available_consistency() {
        let config = GeometryPathConfig::default();
        let features = wgpu::Features::empty();

        // Should be consistent across calls
        for _ in 0..5 {
            assert!(config.preferred_available(features));
        }
    }
}

// =============================================================================
// CATEGORY 5: TRAIT VERIFICATION - Derived Traits
// =============================================================================

mod trait_tests {
    use super::*;

    #[test]
    fn test_geometry_path_default() {
        let path = GeometryPath::default();
        assert_eq!(path, GeometryPath::Traditional);
    }

    #[test]
    fn test_geometry_path_clone() {
        let path = GeometryPath::Traditional;
        let cloned = path.clone();
        assert_eq!(path, cloned);
    }

    #[test]
    fn test_geometry_path_copy() {
        let path = GeometryPath::Meshlet;
        let copied = path; // Copy happens here
        let _also_path = path; // path is still valid (Copy)
        assert_eq!(copied, GeometryPath::Meshlet);
    }

    #[test]
    fn test_geometry_path_debug() {
        let traditional = format!("{:?}", GeometryPath::Traditional);
        let meshlet = format!("{:?}", GeometryPath::Meshlet);
        assert!(traditional.contains("Traditional"));
        assert!(meshlet.contains("Meshlet"));
    }

    #[test]
    fn test_geometry_path_partial_eq() {
        assert_eq!(GeometryPath::Traditional, GeometryPath::Traditional);
        assert_eq!(GeometryPath::Meshlet, GeometryPath::Meshlet);
        assert_ne!(GeometryPath::Traditional, GeometryPath::Meshlet);
    }

    #[test]
    fn test_geometry_path_hash() {
        let mut set = HashSet::new();
        set.insert(GeometryPath::Traditional);
        set.insert(GeometryPath::Meshlet);
        set.insert(GeometryPath::Traditional); // duplicate

        assert_eq!(set.len(), 2);
    }

    #[test]
    fn test_geometry_path_hash_contains() {
        let mut set = HashSet::new();
        set.insert(GeometryPath::Traditional);
        set.insert(GeometryPath::Meshlet);

        assert!(set.contains(&GeometryPath::Traditional));
        assert!(set.contains(&GeometryPath::Meshlet));
    }

    #[test]
    fn test_geometry_path_config_clone() {
        let config = GeometryPathConfig::prefer_meshlet();
        let cloned = config.clone();
        assert_eq!(cloned.preferred, GeometryPath::Meshlet);
        assert_eq!(cloned.force_traditional, false);
    }

    #[test]
    fn test_geometry_path_config_debug() {
        let config = GeometryPathConfig::default();
        let debug_str = format!("{:?}", config);
        assert!(debug_str.contains("GeometryPathConfig"));
    }
}

// =============================================================================
// CATEGORY 6: CONSTANTS - Path Names
// =============================================================================

mod constants_tests {
    use super::*;

    #[test]
    fn test_traditional_name_is_string() {
        let name: &str = TRADITIONAL_PATH_NAME;
        assert!(!name.is_empty());
    }

    #[test]
    fn test_meshlet_name_is_string() {
        let name: &str = MESHLET_PATH_NAME;
        assert!(!name.is_empty());
    }

    #[test]
    fn test_names_are_different() {
        assert_ne!(TRADITIONAL_PATH_NAME, MESHLET_PATH_NAME);
    }

    #[test]
    fn test_name_method_returns_static_str() {
        // Verify name() returns &'static str
        let name: &'static str = GeometryPath::Traditional.name();
        assert_eq!(name, TRADITIONAL_PATH_NAME);
    }

    #[test]
    fn test_names_are_human_readable() {
        // Names should be readable (no underscores, proper capitalization)
        assert!(TRADITIONAL_PATH_NAME.chars().next().unwrap().is_uppercase());
        assert!(MESHLET_PATH_NAME.chars().next().unwrap().is_uppercase());
    }
}

// =============================================================================
// CATEGORY 7: INTEGRATION - Multiple Paths
// =============================================================================

mod integration_tests {
    use super::*;

    #[test]
    fn test_all_paths_have_names() {
        let paths = [GeometryPath::Traditional, GeometryPath::Meshlet];
        for path in paths {
            assert!(!path.name().is_empty());
        }
    }

    #[test]
    fn test_all_paths_have_unique_names() {
        let paths = [GeometryPath::Traditional, GeometryPath::Meshlet];
        let names: HashSet<_> = paths.iter().map(|p| p.name()).collect();
        assert_eq!(names.len(), paths.len());
    }

    #[test]
    fn test_config_resolution_chain() {
        let features = wgpu::Features::empty();

        // Test resolution priority: force_traditional > preferred > available
        let configs = [
            GeometryPathConfig::default(),
            GeometryPathConfig::prefer_meshlet(),
            GeometryPathConfig::force_traditional(),
        ];

        for config in configs {
            let resolved = config.resolve(features);
            // All should resolve to Traditional (meshlet not available)
            assert_eq!(resolved, GeometryPath::Traditional);
        }
    }

    #[test]
    fn test_path_comparison_operators() {
        let trad1 = GeometryPath::Traditional;
        let trad2 = GeometryPath::Traditional;
        let mesh = GeometryPath::Meshlet;

        assert!(trad1 == trad2);
        assert!(trad1 != mesh);
        assert!(trad2 != mesh);
    }

    #[test]
    fn test_hashset_all_variants() {
        let mut set = HashSet::new();
        set.insert(GeometryPath::Traditional);
        set.insert(GeometryPath::Meshlet);

        assert_eq!(set.len(), 2);

        // Remove and verify
        set.remove(&GeometryPath::Traditional);
        assert_eq!(set.len(), 1);
        assert!(!set.contains(&GeometryPath::Traditional));
        assert!(set.contains(&GeometryPath::Meshlet));
    }

    #[test]
    fn test_path_used_in_match_expression() {
        let path = GeometryPath::select(wgpu::Features::empty());

        let description = match path {
            GeometryPath::Traditional => "indexed geometry",
            GeometryPath::Meshlet => "mesh shader dispatch",
        };

        assert_eq!(description, "indexed geometry");
    }
}

// =============================================================================
// CATEGORY 8: FUTURE-PROOFING - Meshlet Behavior Documentation
// =============================================================================

mod future_proofing_tests {
    use super::*;

    #[test]
    fn test_meshlet_available_is_stubbed() {
        // Document: meshlet_available currently returns false always
        // When mesh shaders become stable, this test should be updated
        let features = wgpu::Features::all();
        let available = GeometryPath::meshlet_available(features);
        assert!(!available, "meshlet_available should return false until mesh shaders are stable");
    }

    #[test]
    fn test_select_prefers_meshlet_when_available() {
        // Document expected behavior: select() should return Meshlet when available
        // Currently returns Traditional because meshlet_available is stubbed
        let features = wgpu::Features::all();
        let path = GeometryPath::select(features);

        // When mesh shaders are implemented, this should become:
        // assert_eq!(path, GeometryPath::Meshlet);
        // For now:
        assert_eq!(path, GeometryPath::Traditional);
    }

    #[test]
    fn test_meshlet_path_exists() {
        // Verify the Meshlet variant exists for future use
        let path = GeometryPath::Meshlet;
        assert!(path.is_meshlet());
        assert_eq!(path.name(), "Meshlet");
    }

    #[test]
    fn test_config_prefer_meshlet_exists() {
        // Verify prefer_meshlet() constructor exists for future use
        let config = GeometryPathConfig::prefer_meshlet();
        assert_eq!(config.preferred, GeometryPath::Meshlet);
    }

    #[test]
    fn test_is_traditional_is_meshlet_exclusive() {
        // These methods should be mutually exclusive
        let trad = GeometryPath::Traditional;
        let mesh = GeometryPath::Meshlet;

        assert!(trad.is_traditional() && !trad.is_meshlet());
        assert!(mesh.is_meshlet() && !mesh.is_traditional());
    }
}

// =============================================================================
// CATEGORY 9: GEOMETRY RENDERABLE TRAIT - Mock Implementation
// =============================================================================

mod renderable_trait_tests {
    use super::*;

    /// Mock geometry that only supports traditional rendering.
    struct TraditionalOnlyGeometry;

    impl GeometryRenderable for TraditionalOnlyGeometry {
        fn render(&self, path: GeometryPath, _render_pass: &mut wgpu::RenderPass) {
            match path {
                GeometryPath::Traditional => {
                    // Would issue draw_indexed here
                }
                GeometryPath::Meshlet => {
                    panic!("Meshlet not supported");
                }
            }
        }

        fn supports_path(&self, path: GeometryPath) -> bool {
            matches!(path, GeometryPath::Traditional)
        }
    }

    /// Mock geometry that supports both paths.
    struct DualPathGeometry {
        has_meshlets: bool,
    }

    impl GeometryRenderable for DualPathGeometry {
        fn render(&self, path: GeometryPath, _render_pass: &mut wgpu::RenderPass) {
            match path {
                GeometryPath::Traditional => {
                    // Traditional rendering
                }
                GeometryPath::Meshlet => {
                    if !self.has_meshlets {
                        panic!("No meshlet data");
                    }
                    // Meshlet rendering
                }
            }
        }

        fn supports_path(&self, path: GeometryPath) -> bool {
            match path {
                GeometryPath::Traditional => true,
                GeometryPath::Meshlet => self.has_meshlets,
            }
        }
    }

    #[test]
    fn test_traditional_only_supports_traditional() {
        let geom = TraditionalOnlyGeometry;
        assert!(geom.supports_path(GeometryPath::Traditional));
        assert!(!geom.supports_path(GeometryPath::Meshlet));
    }

    #[test]
    fn test_traditional_only_best_path() {
        let geom = TraditionalOnlyGeometry;
        assert_eq!(geom.best_path(), GeometryPath::Traditional);
    }

    #[test]
    fn test_dual_path_without_meshlets() {
        let geom = DualPathGeometry { has_meshlets: false };
        assert!(geom.supports_path(GeometryPath::Traditional));
        assert!(!geom.supports_path(GeometryPath::Meshlet));
        assert_eq!(geom.best_path(), GeometryPath::Traditional);
    }

    #[test]
    fn test_dual_path_with_meshlets() {
        let geom = DualPathGeometry { has_meshlets: true };
        assert!(geom.supports_path(GeometryPath::Traditional));
        assert!(geom.supports_path(GeometryPath::Meshlet));
        assert_eq!(geom.best_path(), GeometryPath::Meshlet);
    }

    #[test]
    fn test_best_path_prefers_meshlet() {
        // Document: best_path() should prefer Meshlet when supported
        let geom = DualPathGeometry { has_meshlets: true };
        let best = geom.best_path();
        assert_eq!(best, GeometryPath::Meshlet);
    }

    #[test]
    fn test_renderable_trait_object() {
        // Verify trait can be used as trait object
        let geom: Box<dyn GeometryRenderable> = Box::new(TraditionalOnlyGeometry);
        assert!(geom.supports_path(GeometryPath::Traditional));
    }
}

// =============================================================================
// CATEGORY 10: SEND + SYNC VERIFICATION
// =============================================================================

mod send_sync_tests {
    use super::*;

    fn assert_send<T: Send>() {}
    fn assert_sync<T: Sync>() {}

    #[test]
    fn test_geometry_path_is_send() {
        assert_send::<GeometryPath>();
    }

    #[test]
    fn test_geometry_path_is_sync() {
        assert_sync::<GeometryPath>();
    }

    #[test]
    fn test_geometry_path_config_is_send() {
        assert_send::<GeometryPathConfig>();
    }

    #[test]
    fn test_geometry_path_config_is_sync() {
        assert_sync::<GeometryPathConfig>();
    }

    #[test]
    fn test_path_can_be_shared_across_threads() {
        use std::sync::Arc;

        let path = Arc::new(GeometryPath::Traditional);
        let path_clone = Arc::clone(&path);

        std::thread::spawn(move || {
            assert!(path_clone.is_traditional());
        })
        .join()
        .unwrap();
    }
}
