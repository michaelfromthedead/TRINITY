// Blackbox contract tests for T-WGPU-P1.5.2 CapabilityManager
//
// CLEANROOM: No access to implementation files. Tests use only the public API
// exported by `renderer_backend::device`.
//
// Forbidden files (per TESTDEV_BLACKBOX prompt):
//   - crates/renderer-backend/src/device/capability.rs
//   - Any WHITEBOX test file for this task
//
// Contract sources:
//   - docs/WGPU_DOCS/RDC_OUTPUT/PHASE_1_CORE_TODO.md (T-WGPU-P1.5.2)
//   - docs/WGPU_DOCS/RDC_OUTPUT/PHASE_1_CORE_ARCH.md (Capability Detection section)
//   - crates/renderer-backend/src/device/mod.rs (public exports only)
//
// Public contract (T-WGPU-P1.5.2):
//   - CapabilityManager struct
//   - supports_ray_tracing() -> bool
//   - supports_bindless() -> bool
//   - supports_gpu_culling() -> bool
//   - supports_timestamp_queries() -> bool
//   - select_render_path() -> RenderPath
//   - select_texture_compression() -> TextureCompression
//   - max_bindless_textures() -> u32
//   - report() -> CapabilityReport
//
// Test design rationale:
//   Equivalence partitioning:
//     - RenderPath variants (RayTraced, GPUDriven, Traditional, Fallback)
//     - TextureCompression variants
//     - Support query methods (true/false paths)
//   Boundary cases:
//     - max_bindless_textures() return value >= 0
//     - select_render_path() with different tier scenarios
//   Trait contract verification:
//     - Debug, Clone, Copy, Eq for RenderPath
//     - Debug, Clone, Copy, Eq for TextureCompression
//     - Thread-safety markers (Send + Sync) for CapabilityManager

use renderer_backend::device::{
    CapabilityManager, CapabilityReport, CapabilityTier, RenderPath, TextureCompression,
    TrinityInstance,
};
use std::sync::Arc;

// =============================================================================
// 1. RenderPath Enum Variant Contract Tests
// =============================================================================

/// Verifies that RenderPath::RayTraced variant exists.
///
/// Contract: RenderPath has a RayTraced variant for ray tracing render path.
#[test]
fn test_render_path_ray_traced_variant_exists() {
    let path = RenderPath::RayTraced;
    let _ = format!("{:?}", path);
}

/// Verifies that RenderPath::GPUDriven variant exists.
///
/// Contract: RenderPath has a GPUDriven variant for GPU-driven rendering.
#[test]
fn test_render_path_gpu_driven_variant_exists() {
    let path = RenderPath::GPUDriven;
    let _ = format!("{:?}", path);
}

/// Verifies that RenderPath::Traditional variant exists.
///
/// Contract: RenderPath has a Traditional variant for standard rendering.
#[test]
fn test_render_path_traditional_variant_exists() {
    let path = RenderPath::Traditional;
    let _ = format!("{:?}", path);
}

/// Verifies that RenderPath::Fallback variant exists.
///
/// Contract: RenderPath has a Fallback variant for lowest capability path.
#[test]
fn test_render_path_fallback_variant_exists() {
    let path = RenderPath::Fallback;
    let _ = format!("{:?}", path);
}

// =============================================================================
// 2. RenderPath Trait Implementation Contract Tests
// =============================================================================

/// Verifies that RenderPath implements Debug trait.
///
/// Contract: Debug trait produces non-empty, distinct output for each variant.
#[test]
fn test_render_path_debug_trait() {
    let ray_traced_debug = format!("{:?}", RenderPath::RayTraced);
    let gpu_driven_debug = format!("{:?}", RenderPath::GPUDriven);
    let traditional_debug = format!("{:?}", RenderPath::Traditional);
    let fallback_debug = format!("{:?}", RenderPath::Fallback);

    assert!(!ray_traced_debug.is_empty());
    assert!(!gpu_driven_debug.is_empty());
    assert!(!traditional_debug.is_empty());
    assert!(!fallback_debug.is_empty());

    // Each variant should have distinct debug output
    assert_ne!(ray_traced_debug, gpu_driven_debug);
    assert_ne!(gpu_driven_debug, traditional_debug);
    assert_ne!(traditional_debug, fallback_debug);
}

/// Verifies that RenderPath Debug contains variant name.
///
/// Contract: Debug output should be informative.
#[test]
fn test_render_path_debug_contains_variant_name() {
    let ray_traced_debug = format!("{:?}", RenderPath::RayTraced);
    let gpu_driven_debug = format!("{:?}", RenderPath::GPUDriven);
    let traditional_debug = format!("{:?}", RenderPath::Traditional);
    let fallback_debug = format!("{:?}", RenderPath::Fallback);

    assert!(
        ray_traced_debug.contains("RayTraced"),
        "Debug for RayTraced should contain 'RayTraced', got: {}",
        ray_traced_debug
    );
    assert!(
        gpu_driven_debug.contains("GPUDriven"),
        "Debug for GPUDriven should contain 'GPUDriven', got: {}",
        gpu_driven_debug
    );
    assert!(
        traditional_debug.contains("Traditional"),
        "Debug for Traditional should contain 'Traditional', got: {}",
        traditional_debug
    );
    assert!(
        fallback_debug.contains("Fallback"),
        "Debug for Fallback should contain 'Fallback', got: {}",
        fallback_debug
    );
}

/// Verifies that RenderPath implements Clone trait.
///
/// Contract: Clone produces an equal value.
#[test]
fn test_render_path_clone_trait() {
    let original = RenderPath::GPUDriven;
    let cloned = original.clone();

    assert_eq!(original, cloned, "Cloned RenderPath should equal original");
}

/// Verifies that RenderPath implements Copy trait.
///
/// Contract: Copy semantics allow value duplication without explicit clone.
#[test]
fn test_render_path_copy_trait() {
    let original = RenderPath::Traditional;
    let copied = original; // Copy semantics
    let also_copied = original; // Can copy again (original still valid)

    assert_eq!(original, copied);
    assert_eq!(original, also_copied);
    assert_eq!(copied, also_copied);
}

/// Verifies that RenderPath implements Eq/PartialEq with reflexivity.
///
/// Contract: Each variant equals itself.
#[test]
fn test_render_path_eq_reflexive() {
    assert_eq!(RenderPath::RayTraced, RenderPath::RayTraced);
    assert_eq!(RenderPath::GPUDriven, RenderPath::GPUDriven);
    assert_eq!(RenderPath::Traditional, RenderPath::Traditional);
    assert_eq!(RenderPath::Fallback, RenderPath::Fallback);
}

/// Verifies that RenderPath implements PartialEq with symmetry.
///
/// Contract: If a == b then b == a.
#[test]
fn test_render_path_eq_symmetric() {
    let a = RenderPath::RayTraced;
    let b = RenderPath::RayTraced;

    assert_eq!(a, b);
    assert_eq!(b, a);
}

/// Verifies that different RenderPath variants are not equal.
///
/// Contract: Different variants produce distinct values.
#[test]
fn test_render_path_distinct_variants_not_equal() {
    assert_ne!(RenderPath::RayTraced, RenderPath::GPUDriven);
    assert_ne!(RenderPath::RayTraced, RenderPath::Traditional);
    assert_ne!(RenderPath::RayTraced, RenderPath::Fallback);
    assert_ne!(RenderPath::GPUDriven, RenderPath::Traditional);
    assert_ne!(RenderPath::GPUDriven, RenderPath::Fallback);
    assert_ne!(RenderPath::Traditional, RenderPath::Fallback);
}

/// Verifies that RenderPath can be used in HashMap (requires Hash + Eq).
///
/// Contract: RenderPath should be usable as a HashMap key.
#[test]
fn test_render_path_hash_trait() {
    use std::collections::HashMap;

    let mut map: HashMap<RenderPath, &str> = HashMap::new();
    map.insert(RenderPath::RayTraced, "rt");
    map.insert(RenderPath::GPUDriven, "gpu");
    map.insert(RenderPath::Traditional, "trad");
    map.insert(RenderPath::Fallback, "fallback");

    assert_eq!(map.get(&RenderPath::RayTraced), Some(&"rt"));
    assert_eq!(map.get(&RenderPath::GPUDriven), Some(&"gpu"));
    assert_eq!(map.get(&RenderPath::Traditional), Some(&"trad"));
    assert_eq!(map.get(&RenderPath::Fallback), Some(&"fallback"));
}

/// Verifies RenderPath can be used in match expressions exhaustively.
///
/// Contract: All four variants can be matched.
#[test]
fn test_render_path_match_exhaustive() {
    let paths = [
        RenderPath::RayTraced,
        RenderPath::GPUDriven,
        RenderPath::Traditional,
        RenderPath::Fallback,
    ];

    for path in paths {
        let description = match path {
            RenderPath::RayTraced => "ray tracing enabled",
            RenderPath::GPUDriven => "gpu-driven rendering",
            RenderPath::Traditional => "traditional forward/deferred",
            RenderPath::Fallback => "minimal fallback",
        };
        assert!(!description.is_empty());
    }
}

// =============================================================================
// 3. TextureCompression Enum Contract Tests
// =============================================================================

/// Verifies that TextureCompression variants exist and are usable.
///
/// Contract: TextureCompression enum exists with distinct compression format variants.
#[test]
fn test_texture_compression_variants_exist() {
    // Test that we can create TextureCompression values
    // The exact variants depend on implementation, but at minimum:
    // BC (DX), ASTC, ETC2, or None should be available
    let _ = format!("{:?}", TextureCompression::BC);
}

/// Verifies that TextureCompression implements Debug trait.
///
/// Contract: Debug trait produces non-empty output.
#[test]
fn test_texture_compression_debug_trait() {
    let debug_output = format!("{:?}", TextureCompression::BC);
    assert!(
        !debug_output.is_empty(),
        "Debug output should not be empty"
    );
}

/// Verifies that TextureCompression implements Clone trait.
///
/// Contract: Clone produces an equal value.
#[test]
fn test_texture_compression_clone_trait() {
    let original = TextureCompression::BC;
    let cloned = original.clone();

    assert_eq!(original, cloned, "Cloned TextureCompression should equal original");
}

/// Verifies that TextureCompression implements Copy trait.
///
/// Contract: Copy semantics work correctly.
#[test]
fn test_texture_compression_copy_trait() {
    let original = TextureCompression::BC;
    let copied = original; // Copy semantics
    let also_copied = original; // Can copy again

    assert_eq!(original, copied);
    assert_eq!(original, also_copied);
}

/// Verifies that TextureCompression implements Eq with reflexivity.
///
/// Contract: Each variant equals itself.
#[test]
fn test_texture_compression_eq_reflexive() {
    assert_eq!(TextureCompression::BC, TextureCompression::BC);
}

/// Verifies that TextureCompression can be used in HashMap.
///
/// Contract: TextureCompression should be usable as HashMap key.
#[test]
fn test_texture_compression_hash_trait() {
    use std::collections::HashMap;

    let mut map: HashMap<TextureCompression, &str> = HashMap::new();
    map.insert(TextureCompression::BC, "bc");

    assert_eq!(map.get(&TextureCompression::BC), Some(&"bc"));
}

// =============================================================================
// 4. CapabilityManager Creation Contract Tests
// =============================================================================

/// Helper to create a CapabilityManager for testing.
/// Returns None if no adapter is available.
fn try_create_capability_manager() -> Option<CapabilityManager> {
    let instance = TrinityInstance::new();
    let adapter = pollster::block_on(instance.inner().request_adapter(
        &wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        },
    ))?;

    Some(CapabilityManager::from_adapter(&adapter))
}

/// Verifies that CapabilityManager::from_adapter is the constructor.
///
/// Contract: CapabilityManager is created from an adapter.
#[test]

fn test_capability_manager_from_adapter_constructor() {
    if let Some(manager) = try_create_capability_manager() {
        // Verify we can query the manager
        let _ = manager.tier();
    }
}

/// Verifies that CapabilityManager has a tier() method.
///
/// Contract: CapabilityManager exposes the capability tier.
#[test]

fn test_capability_manager_tier_method() {
    if let Some(manager) = try_create_capability_manager() {
        let tier: CapabilityTier = manager.tier();
        // Tier should be a valid CapabilityTier variant
        let _ = format!("{:?}", tier);
    }
}

// =============================================================================
// 5. CapabilityManager Support Query Methods Contract Tests
// =============================================================================

/// Verifies that supports_ray_tracing() method exists and returns bool.
///
/// Contract: CapabilityManager has supports_ray_tracing() -> bool.
#[test]

fn test_capability_manager_supports_ray_tracing() {
    if let Some(manager) = try_create_capability_manager() {
        let supports: bool = manager.supports_ray_tracing();
        // Should return a valid boolean (true or false)
        let _ = supports;
    }
}

/// Verifies that supports_bindless() method exists and returns bool.
///
/// Contract: CapabilityManager has supports_bindless() -> bool.
#[test]

fn test_capability_manager_supports_bindless() {
    if let Some(manager) = try_create_capability_manager() {
        let supports: bool = manager.supports_bindless();
        let _ = supports;
    }
}

/// Verifies that supports_gpu_culling() method exists and returns bool.
///
/// Contract: CapabilityManager has supports_gpu_culling() -> bool.
#[test]

fn test_capability_manager_supports_gpu_culling() {
    if let Some(manager) = try_create_capability_manager() {
        let supports: bool = manager.supports_gpu_culling();
        let _ = supports;
    }
}

/// Verifies that supports_timestamp_queries() method exists and returns bool.
///
/// Contract: CapabilityManager has supports_timestamp_queries() -> bool.
#[test]

fn test_capability_manager_supports_timestamp_queries() {
    if let Some(manager) = try_create_capability_manager() {
        let supports: bool = manager.supports_timestamp_queries();
        let _ = supports;
    }
}

/// Verifies consistency between ray tracing support and Full tier.
///
/// Contract: Full tier requires ray tracing capability.
#[test]

fn test_capability_manager_rt_tier_consistency() {
    if let Some(manager) = try_create_capability_manager() {
        let tier = manager.tier();
        let supports_rt = manager.supports_ray_tracing();

        // If tier is Full, ray tracing should be supported
        if tier == CapabilityTier::Full {
            assert!(
                supports_rt,
                "Full tier should have ray tracing support"
            );
        }

        // If ray tracing is not supported, tier should not be Full
        if !supports_rt {
            assert_ne!(
                tier,
                CapabilityTier::Full,
                "Without ray tracing, tier should not be Full"
            );
        }
    }
}

/// Verifies consistency between bindless support and Advanced tier.
///
/// Contract: Advanced tier requires bindless capability.
#[test]

fn test_capability_manager_bindless_tier_consistency() {
    if let Some(manager) = try_create_capability_manager() {
        let tier = manager.tier();
        let supports_bindless = manager.supports_bindless();

        // If tier is Advanced or higher, bindless should be supported
        if tier >= CapabilityTier::Advanced {
            assert!(
                supports_bindless,
                "Advanced tier or higher should have bindless support"
            );
        }
    }
}

// =============================================================================
// 6. CapabilityManager select_render_path() Contract Tests
// =============================================================================

/// Verifies that select_render_path() method exists and returns RenderPath.
///
/// Contract: select_render_path() returns a RenderPath based on tier.
#[test]

fn test_capability_manager_select_render_path() {
    if let Some(manager) = try_create_capability_manager() {
        let path: RenderPath = manager.select_render_path();
        // Should be a valid RenderPath variant
        let _ = format!("{:?}", path);
    }
}

/// Verifies that select_render_path() is deterministic.
///
/// Contract: Same manager should always return same render path.
#[test]

fn test_capability_manager_select_render_path_deterministic() {
    if let Some(manager) = try_create_capability_manager() {
        let path1 = manager.select_render_path();
        let path2 = manager.select_render_path();
        let path3 = manager.select_render_path();

        assert_eq!(path1, path2);
        assert_eq!(path2, path3);
    }
}

/// Verifies render path selection is consistent with tier.
///
/// Contract: Higher tiers should enable higher capability render paths.
#[test]

fn test_capability_manager_render_path_tier_consistency() {
    if let Some(manager) = try_create_capability_manager() {
        let tier = manager.tier();
        let path = manager.select_render_path();

        match tier {
            CapabilityTier::Full => {
                // Full tier should use RayTraced path
                assert_eq!(
                    path,
                    RenderPath::RayTraced,
                    "Full tier should select RayTraced render path"
                );
            }
            CapabilityTier::Advanced => {
                // Advanced tier should use GPUDriven or better
                assert!(
                    path == RenderPath::GPUDriven || path == RenderPath::RayTraced,
                    "Advanced tier should select GPUDriven or RayTraced path"
                );
            }
            CapabilityTier::Standard => {
                // Standard tier should use Traditional or better
                assert!(
                    path == RenderPath::Traditional
                        || path == RenderPath::GPUDriven
                        || path == RenderPath::RayTraced,
                    "Standard tier should select Traditional or better path"
                );
            }
            CapabilityTier::Minimal => {
                // Minimal tier should use Fallback or Traditional
                assert!(
                    path == RenderPath::Fallback || path == RenderPath::Traditional,
                    "Minimal tier should select Fallback or Traditional path"
                );
            }
        }
    }
}

// =============================================================================
// 7. CapabilityManager select_texture_compression() Contract Tests
// =============================================================================

/// Verifies that select_texture_compression() method exists and returns TextureCompression.
///
/// Contract: select_texture_compression() returns a TextureCompression format.
#[test]

fn test_capability_manager_select_texture_compression() {
    if let Some(manager) = try_create_capability_manager() {
        let compression: TextureCompression = manager.select_texture_compression();
        let _ = format!("{:?}", compression);
    }
}

/// Verifies that select_texture_compression() is deterministic.
///
/// Contract: Same manager should always return same compression format.
#[test]

fn test_capability_manager_select_texture_compression_deterministic() {
    if let Some(manager) = try_create_capability_manager() {
        let compression1 = manager.select_texture_compression();
        let compression2 = manager.select_texture_compression();
        let compression3 = manager.select_texture_compression();

        assert_eq!(compression1, compression2);
        assert_eq!(compression2, compression3);
    }
}

// =============================================================================
// 8. CapabilityManager max_bindless_textures() Contract Tests
// =============================================================================

/// Verifies that max_bindless_textures() method exists and returns u32.
///
/// Contract: max_bindless_textures() returns the maximum bindless texture count.
#[test]

fn test_capability_manager_max_bindless_textures() {
    if let Some(manager) = try_create_capability_manager() {
        let max_textures: u32 = manager.max_bindless_textures();
        // Value should be >= 0 (always true for u32, but documents intent)
        let _ = max_textures;
    }
}

/// Verifies max_bindless_textures() is consistent with bindless support.
///
/// Contract: If bindless not supported, max_bindless_textures should be 0 or minimal.
#[test]

fn test_capability_manager_max_bindless_textures_consistency() {
    if let Some(manager) = try_create_capability_manager() {
        let supports_bindless = manager.supports_bindless();
        let max_textures = manager.max_bindless_textures();

        if supports_bindless {
            assert!(
                max_textures > 0,
                "If bindless is supported, max_bindless_textures should be > 0"
            );
        }
    }
}

/// Verifies that max_bindless_textures() is deterministic.
///
/// Contract: Same manager should always return same value.
#[test]

fn test_capability_manager_max_bindless_textures_deterministic() {
    if let Some(manager) = try_create_capability_manager() {
        let max1 = manager.max_bindless_textures();
        let max2 = manager.max_bindless_textures();
        let max3 = manager.max_bindless_textures();

        assert_eq!(max1, max2);
        assert_eq!(max2, max3);
    }
}

// =============================================================================
// 9. CapabilityManager report() Contract Tests
// =============================================================================

/// Verifies that report() method exists and returns CapabilityReport.
///
/// Contract: report() returns a CapabilityReport with capability details.
#[test]

fn test_capability_manager_report() {
    if let Some(manager) = try_create_capability_manager() {
        let report: CapabilityReport = manager.report();
        // Report should have a valid tier
        let _tier: CapabilityTier = report.tier;
    }
}

/// Verifies that report() tier matches manager tier.
///
/// Contract: report().tier should equal manager.tier().
#[test]

fn test_capability_manager_report_tier_consistency() {
    if let Some(manager) = try_create_capability_manager() {
        let report = manager.report();
        let tier = manager.tier();

        assert_eq!(
            report.tier, tier,
            "report().tier should match manager.tier()"
        );
    }
}

/// Verifies that report() is deterministic.
///
/// Contract: Same manager should always produce equivalent reports.
#[test]

fn test_capability_manager_report_deterministic() {
    if let Some(manager) = try_create_capability_manager() {
        let report1 = manager.report();
        let report2 = manager.report();

        assert_eq!(report1.tier, report2.tier);
        assert_eq!(report1.has_ray_tracing, report2.has_ray_tracing);
        assert_eq!(report1.has_bindless, report2.has_bindless);
    }
}

// =============================================================================
// 10. CapabilityManager Thread Safety Contract Tests
// =============================================================================

/// Verifies that CapabilityManager is Send.
///
/// Contract: CapabilityManager can be transferred between threads.
#[test]
fn test_capability_manager_is_send() {
    fn assert_send<T: Send>() {}
    assert_send::<CapabilityManager>();
}

/// Verifies that CapabilityManager is Sync.
///
/// Contract: CapabilityManager can be shared between threads.
#[test]
fn test_capability_manager_is_sync() {
    fn assert_sync<T: Sync>() {}
    assert_sync::<CapabilityManager>();
}

/// Verifies that CapabilityManager can be wrapped in Arc.
///
/// Contract: CapabilityManager can be used with Arc for shared ownership.
#[test]

fn test_capability_manager_arc_usable() {
    if let Some(manager) = try_create_capability_manager() {
        let arc_manager = Arc::new(manager);
        let arc_clone = Arc::clone(&arc_manager);

        // Should be able to query from both references
        let tier1 = arc_manager.tier();
        let tier2 = arc_clone.tier();

        assert_eq!(tier1, tier2);
    }
}

/// Verifies that CapabilityManager can be shared across threads.
///
/// Contract: CapabilityManager supports concurrent access.
#[test]

fn test_capability_manager_concurrent_access() {
    if let Some(manager) = try_create_capability_manager() {
        let arc_manager = Arc::new(manager);

        let handles: Vec<_> = (0..4)
            .map(|_| {
                let manager_clone = Arc::clone(&arc_manager);
                std::thread::spawn(move || manager_clone.tier())
            })
            .collect();

        let results: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();

        // All threads should see the same tier
        let first = results[0];
        for tier in &results {
            assert_eq!(*tier, first, "All threads should see the same tier");
        }
    }
}

// =============================================================================
// 11. RenderPath Thread Safety Contract Tests
// =============================================================================

/// Verifies that RenderPath is Send.
///
/// Contract: RenderPath can be transferred between threads.
#[test]
fn test_render_path_is_send() {
    fn assert_send<T: Send>() {}
    assert_send::<RenderPath>();
}

/// Verifies that RenderPath is Sync.
///
/// Contract: RenderPath can be shared between threads.
#[test]
fn test_render_path_is_sync() {
    fn assert_sync<T: Sync>() {}
    assert_sync::<RenderPath>();
}

// =============================================================================
// 12. TextureCompression Thread Safety Contract Tests
// =============================================================================

/// Verifies that TextureCompression is Send.
///
/// Contract: TextureCompression can be transferred between threads.
#[test]
fn test_texture_compression_is_send() {
    fn assert_send<T: Send>() {}
    assert_send::<TextureCompression>();
}

/// Verifies that TextureCompression is Sync.
///
/// Contract: TextureCompression can be shared between threads.
#[test]
fn test_texture_compression_is_sync() {
    fn assert_sync<T: Sync>() {}
    assert_sync::<TextureCompression>();
}

// =============================================================================
// 13. CapabilityManager Trait Implementation Contract Tests
// =============================================================================

/// Verifies that CapabilityManager implements Debug trait.
///
/// Contract: CapabilityManager can be debug-printed.
#[test]

fn test_capability_manager_debug_trait() {
    if let Some(manager) = try_create_capability_manager() {
        let debug_output = format!("{:?}", manager);
        assert!(!debug_output.is_empty(), "Debug output should not be empty");
    }
}

/// Verifies that CapabilityManager implements Clone trait.
///
/// Contract: CapabilityManager can be cloned.
#[test]

fn test_capability_manager_clone_trait() {
    if let Some(original) = try_create_capability_manager() {
        let cloned = original.clone();

        // Cloned manager should have same tier
        assert_eq!(original.tier(), cloned.tier());
        assert_eq!(original.supports_ray_tracing(), cloned.supports_ray_tracing());
        assert_eq!(original.supports_bindless(), cloned.supports_bindless());
    }
}

// =============================================================================
// 14. CapabilityManager Default Behavior Contract Tests
// =============================================================================

/// Verifies that all query methods work without panic on any adapter.
///
/// Contract: All CapabilityManager methods should be safe to call.
#[test]

fn test_capability_manager_all_methods_safe() {
    if let Some(manager) = try_create_capability_manager() {
        // All these should complete without panic
        let _tier = manager.tier();
        let _rt = manager.supports_ray_tracing();
        let _bindless = manager.supports_bindless();
        let _culling = manager.supports_gpu_culling();
        let _timestamps = manager.supports_timestamp_queries();
        let _path = manager.select_render_path();
        let _compression = manager.select_texture_compression();
        let _max_textures = manager.max_bindless_textures();
        let _report = manager.report();
    }
}

/// Verifies consistency across all query methods.
///
/// Contract: All capability query results should be internally consistent.
#[test]

fn test_capability_manager_internal_consistency() {
    if let Some(manager) = try_create_capability_manager() {
        let tier = manager.tier();
        let supports_rt = manager.supports_ray_tracing();
        let supports_bindless = manager.supports_bindless();
        let supports_culling = manager.supports_gpu_culling();
        let path = manager.select_render_path();
        let report = manager.report();

        // Report should match individual queries
        assert_eq!(report.tier, tier);
        assert_eq!(report.has_ray_tracing, supports_rt);
        assert_eq!(report.has_bindless, supports_bindless);

        // Render path should be appropriate for capabilities
        if path == RenderPath::RayTraced {
            assert!(
                supports_rt,
                "RayTraced path should require ray tracing support"
            );
        }
        if path == RenderPath::GPUDriven {
            assert!(
                supports_culling || supports_bindless,
                "GPUDriven path should have GPU-driven capabilities"
            );
        }
    }
}

// =============================================================================
// 15. Edge Cases and Boundary Contract Tests
// =============================================================================

/// Verifies that creating multiple managers from same adapter produces equivalent results.
///
/// Contract: CapabilityManager creation is deterministic for same adapter.
#[test]

fn test_capability_manager_creation_deterministic() {
    let instance = TrinityInstance::new();
    if let Some(adapter) = pollster::block_on(instance.inner().request_adapter(
        &wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        },
    )) {
        let manager1 = CapabilityManager::from_adapter(&adapter);
        let manager2 = CapabilityManager::from_adapter(&adapter);

        assert_eq!(manager1.tier(), manager2.tier());
        assert_eq!(manager1.supports_ray_tracing(), manager2.supports_ray_tracing());
        assert_eq!(manager1.supports_bindless(), manager2.supports_bindless());
        assert_eq!(manager1.select_render_path(), manager2.select_render_path());
    }
}

/// Verifies RenderPath ordering relationship (if Ord is implemented).
///
/// Contract: RenderPath variants have a logical capability ordering.
#[test]
fn test_render_path_capability_ordering() {
    // Test that we can compare capability levels conceptually
    // RayTraced > GPUDriven > Traditional > Fallback
    // Even if Ord isn't implemented, the conceptual ordering should be:

    let paths = [
        RenderPath::RayTraced,
        RenderPath::GPUDriven,
        RenderPath::Traditional,
        RenderPath::Fallback,
    ];

    // Each path should be distinct
    for i in 0..paths.len() {
        for j in (i + 1)..paths.len() {
            assert_ne!(paths[i], paths[j], "All render paths should be distinct");
        }
    }
}

/// Verifies that max_bindless_textures has a reasonable upper bound.
///
/// Contract: max_bindless_textures should not exceed reasonable hardware limits.
#[test]

fn test_capability_manager_max_bindless_textures_reasonable() {
    if let Some(manager) = try_create_capability_manager() {
        let max_textures = manager.max_bindless_textures();

        // Maximum bindless textures should be within hardware limits
        // Most hardware supports up to 1M textures at most
        assert!(
            max_textures <= 1_000_000,
            "max_bindless_textures should be reasonable (got {})",
            max_textures
        );
    }
}

// =============================================================================
// 16. Type Safety Contract Tests
// =============================================================================

/// Verifies that CapabilityManager methods return the documented types.
///
/// Contract: All method return types match the documented API.
#[test]

fn test_capability_manager_return_types() {
    if let Some(manager) = try_create_capability_manager() {
        // Type annotations verify correct return types
        let _tier: CapabilityTier = manager.tier();
        let _rt: bool = manager.supports_ray_tracing();
        let _bindless: bool = manager.supports_bindless();
        let _culling: bool = manager.supports_gpu_culling();
        let _timestamps: bool = manager.supports_timestamp_queries();
        let _path: RenderPath = manager.select_render_path();
        let _compression: TextureCompression = manager.select_texture_compression();
        let _max_textures: u32 = manager.max_bindless_textures();
        let _report: CapabilityReport = manager.report();
    }
}

/// Verifies that RenderPath is sized.
///
/// Contract: RenderPath has known size at compile time.
#[test]
fn test_render_path_is_sized() {
    fn assert_sized<T: Sized>() {}
    assert_sized::<RenderPath>();
}

/// Verifies that TextureCompression is sized.
///
/// Contract: TextureCompression has known size at compile time.
#[test]
fn test_texture_compression_is_sized() {
    fn assert_sized<T: Sized>() {}
    assert_sized::<TextureCompression>();
}

/// Verifies that CapabilityManager is sized.
///
/// Contract: CapabilityManager has known size at compile time.
#[test]
fn test_capability_manager_is_sized() {
    fn assert_sized<T: Sized>() {}
    assert_sized::<CapabilityManager>();
}
