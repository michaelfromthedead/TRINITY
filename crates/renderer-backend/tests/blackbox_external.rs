//! Blackbox contract tests for T-WGPU-P7.5.4 External/Imported Resources.
//!
//! CLEANROOM: No src/ access beyond the public API exported by the crate.
//! Tests use only `renderer_backend::frame_graph::external::*` -- no internal
//! fields, no private methods, no implementation details.
//!
//! These tests validate the external resource module API contracts without
//! requiring actual wgpu hardware. For resource-based tests, we test the
//! logic types (ExternalResourceType, ImportMode, ResourceBarrier, etc.)
//! and use the ExternalResourceRegistry only for operations that don't
//! require wgpu resources.
//!
//! Coverage:
//!   1-20.   API Contract Tests -- ExternalResourceRegistry public interface
//!   21-45.  Real-World Import Scenarios (logic simulation)
//!   46-65.  Barrier Computation
//!   66-80.  Swapchain Lifecycle (registry logic)
//!   81-95.  Registry Operations
//!   96-110+ Edge Cases
//!
//! Total: 110+ tests

use renderer_backend::frame_graph::external::{
    ExternalResourceType, ImportMode, ResourceBarrier, ExternalResourceRegistry,
    ExternalSynchronizer,
};
use renderer_backend::frame_graph::{ResourceHandle, ResourceAccess, PassIndex};

// =========================================================================
// SECTION 1: API Contract Tests -- ExternalResourceType (20+ tests)
// =========================================================================

#[test]
fn test_001_external_resource_type_swapchain_is_swapchain() {
    assert!(ExternalResourceType::Swapchain.is_swapchain());
}

#[test]
fn test_002_external_resource_type_user_texture_not_swapchain() {
    assert!(!ExternalResourceType::UserTexture.is_swapchain());
}

#[test]
fn test_003_external_resource_type_user_buffer_not_swapchain() {
    assert!(!ExternalResourceType::UserBuffer.is_swapchain());
}

#[test]
fn test_004_external_resource_type_shared_texture_not_swapchain() {
    assert!(!ExternalResourceType::SharedTexture.is_swapchain());
}

#[test]
fn test_005_external_resource_type_shared_buffer_not_swapchain() {
    assert!(!ExternalResourceType::SharedBuffer.is_swapchain());
}

#[test]
fn test_006_external_resource_type_user_texture_is_user_provided() {
    assert!(ExternalResourceType::UserTexture.is_user_provided());
}

#[test]
fn test_007_external_resource_type_user_buffer_is_user_provided() {
    assert!(ExternalResourceType::UserBuffer.is_user_provided());
}

#[test]
fn test_008_external_resource_type_swapchain_not_user_provided() {
    assert!(!ExternalResourceType::Swapchain.is_user_provided());
}

#[test]
fn test_009_external_resource_type_shared_texture_not_user_provided() {
    assert!(!ExternalResourceType::SharedTexture.is_user_provided());
}

#[test]
fn test_010_external_resource_type_shared_buffer_not_user_provided() {
    assert!(!ExternalResourceType::SharedBuffer.is_user_provided());
}

#[test]
fn test_011_external_resource_type_shared_texture_is_shared() {
    assert!(ExternalResourceType::SharedTexture.is_shared());
}

#[test]
fn test_012_external_resource_type_shared_buffer_is_shared() {
    assert!(ExternalResourceType::SharedBuffer.is_shared());
}

#[test]
fn test_013_external_resource_type_swapchain_not_shared() {
    assert!(!ExternalResourceType::Swapchain.is_shared());
}

#[test]
fn test_014_external_resource_type_user_texture_not_shared() {
    assert!(!ExternalResourceType::UserTexture.is_shared());
}

#[test]
fn test_015_external_resource_type_user_buffer_not_shared() {
    assert!(!ExternalResourceType::UserBuffer.is_shared());
}

#[test]
fn test_016_external_resource_type_swapchain_is_texture() {
    assert!(ExternalResourceType::Swapchain.is_texture());
}

#[test]
fn test_017_external_resource_type_user_texture_is_texture() {
    assert!(ExternalResourceType::UserTexture.is_texture());
}

#[test]
fn test_018_external_resource_type_shared_texture_is_texture() {
    assert!(ExternalResourceType::SharedTexture.is_texture());
}

#[test]
fn test_019_external_resource_type_user_buffer_not_texture() {
    assert!(!ExternalResourceType::UserBuffer.is_texture());
}

#[test]
fn test_020_external_resource_type_shared_buffer_not_texture() {
    assert!(!ExternalResourceType::SharedBuffer.is_texture());
}

#[test]
fn test_021_external_resource_type_user_buffer_is_buffer() {
    assert!(ExternalResourceType::UserBuffer.is_buffer());
}

#[test]
fn test_022_external_resource_type_shared_buffer_is_buffer() {
    assert!(ExternalResourceType::SharedBuffer.is_buffer());
}

#[test]
fn test_023_external_resource_type_swapchain_not_buffer() {
    assert!(!ExternalResourceType::Swapchain.is_buffer());
}

#[test]
fn test_024_external_resource_type_user_texture_not_buffer() {
    assert!(!ExternalResourceType::UserTexture.is_buffer());
}

#[test]
fn test_025_external_resource_type_shared_texture_not_buffer() {
    assert!(!ExternalResourceType::SharedTexture.is_buffer());
}

#[test]
fn test_026_external_resource_type_display_swapchain() {
    let display = format!("{}", ExternalResourceType::Swapchain);
    assert_eq!(display, "Swapchain");
}

#[test]
fn test_027_external_resource_type_display_user_texture() {
    let display = format!("{}", ExternalResourceType::UserTexture);
    assert_eq!(display, "UserTexture");
}

#[test]
fn test_028_external_resource_type_display_user_buffer() {
    let display = format!("{}", ExternalResourceType::UserBuffer);
    assert_eq!(display, "UserBuffer");
}

#[test]
fn test_029_external_resource_type_display_shared_texture() {
    let display = format!("{}", ExternalResourceType::SharedTexture);
    assert_eq!(display, "SharedTexture");
}

#[test]
fn test_030_external_resource_type_display_shared_buffer() {
    let display = format!("{}", ExternalResourceType::SharedBuffer);
    assert_eq!(display, "SharedBuffer");
}

#[test]
fn test_031_external_resource_type_default_is_user_texture() {
    assert_eq!(ExternalResourceType::default(), ExternalResourceType::UserTexture);
}

#[test]
fn test_032_external_resource_type_clone() {
    let original = ExternalResourceType::SharedTexture;
    let cloned = original;
    assert_eq!(original, cloned);
}

#[test]
fn test_033_external_resource_type_debug_format() {
    let debug = format!("{:?}", ExternalResourceType::Swapchain);
    assert!(debug.contains("Swapchain"));
}

#[test]
fn test_034_external_resource_type_hash_eq() {
    use std::collections::HashSet;
    let mut set = HashSet::new();
    set.insert(ExternalResourceType::Swapchain);
    set.insert(ExternalResourceType::UserTexture);
    assert!(set.contains(&ExternalResourceType::Swapchain));
    assert!(set.contains(&ExternalResourceType::UserTexture));
    assert!(!set.contains(&ExternalResourceType::SharedBuffer));
}

#[test]
fn test_035_external_resource_type_exclusive_texture_or_buffer() {
    // Every type is either texture or buffer, not both
    for ext_type in [
        ExternalResourceType::Swapchain,
        ExternalResourceType::UserTexture,
        ExternalResourceType::UserBuffer,
        ExternalResourceType::SharedTexture,
        ExternalResourceType::SharedBuffer,
    ] {
        let is_tex = ext_type.is_texture();
        let is_buf = ext_type.is_buffer();
        assert!(
            (is_tex && !is_buf) || (!is_tex && is_buf),
            "{:?} should be exclusively texture or buffer",
            ext_type
        );
    }
}

// =========================================================================
// SECTION 2: API Contract Tests -- ImportMode (15+ tests)
// =========================================================================

#[test]
fn test_036_import_mode_readonly_requires_acquire_barrier() {
    assert!(ImportMode::ReadOnly.requires_acquire_barrier());
}

#[test]
fn test_037_import_mode_writeonly_no_acquire_barrier() {
    assert!(!ImportMode::WriteOnly.requires_acquire_barrier());
}

#[test]
fn test_038_import_mode_readwrite_requires_acquire_barrier() {
    assert!(ImportMode::ReadWrite.requires_acquire_barrier());
}

#[test]
fn test_039_import_mode_readonly_no_release_barrier() {
    assert!(!ImportMode::ReadOnly.requires_release_barrier());
}

#[test]
fn test_040_import_mode_writeonly_requires_release_barrier() {
    assert!(ImportMode::WriteOnly.requires_release_barrier());
}

#[test]
fn test_041_import_mode_readwrite_requires_release_barrier() {
    assert!(ImportMode::ReadWrite.requires_release_barrier());
}

#[test]
fn test_042_import_mode_readonly_is_read() {
    assert!(ImportMode::ReadOnly.is_read());
}

#[test]
fn test_043_import_mode_writeonly_not_read() {
    assert!(!ImportMode::WriteOnly.is_read());
}

#[test]
fn test_044_import_mode_readwrite_is_read() {
    assert!(ImportMode::ReadWrite.is_read());
}

#[test]
fn test_045_import_mode_readonly_not_write() {
    assert!(!ImportMode::ReadOnly.is_write());
}

#[test]
fn test_046_import_mode_writeonly_is_write() {
    assert!(ImportMode::WriteOnly.is_write());
}

#[test]
fn test_047_import_mode_readwrite_is_write() {
    assert!(ImportMode::ReadWrite.is_write());
}

#[test]
fn test_048_import_mode_to_resource_access_readonly() {
    assert_eq!(
        ImportMode::ReadOnly.to_resource_access(),
        ResourceAccess::Read
    );
}

#[test]
fn test_049_import_mode_to_resource_access_writeonly() {
    assert_eq!(
        ImportMode::WriteOnly.to_resource_access(),
        ResourceAccess::Write
    );
}

#[test]
fn test_050_import_mode_to_resource_access_readwrite() {
    assert_eq!(
        ImportMode::ReadWrite.to_resource_access(),
        ResourceAccess::ReadWrite
    );
}

#[test]
fn test_051_import_mode_display_readonly() {
    assert_eq!(format!("{}", ImportMode::ReadOnly), "ReadOnly");
}

#[test]
fn test_052_import_mode_display_writeonly() {
    assert_eq!(format!("{}", ImportMode::WriteOnly), "WriteOnly");
}

#[test]
fn test_053_import_mode_display_readwrite() {
    assert_eq!(format!("{}", ImportMode::ReadWrite), "ReadWrite");
}

#[test]
fn test_054_import_mode_default_is_readonly() {
    assert_eq!(ImportMode::default(), ImportMode::ReadOnly);
}

#[test]
fn test_055_import_mode_from_resource_access_read() {
    assert_eq!(ImportMode::from(ResourceAccess::Read), ImportMode::ReadOnly);
}

#[test]
fn test_056_import_mode_from_resource_access_write() {
    assert_eq!(ImportMode::from(ResourceAccess::Write), ImportMode::WriteOnly);
}

#[test]
fn test_057_import_mode_from_resource_access_readwrite() {
    assert_eq!(
        ImportMode::from(ResourceAccess::ReadWrite),
        ImportMode::ReadWrite
    );
}

#[test]
fn test_058_import_mode_roundtrip_conversion() {
    for mode in [ImportMode::ReadOnly, ImportMode::WriteOnly, ImportMode::ReadWrite] {
        let access = mode.to_resource_access();
        let back = ImportMode::from(access);
        assert_eq!(mode, back, "roundtrip conversion failed for {:?}", mode);
    }
}

#[test]
fn test_059_import_mode_clone() {
    let original = ImportMode::ReadWrite;
    let cloned = original;
    assert_eq!(original, cloned);
}

#[test]
fn test_060_import_mode_debug_format() {
    let debug = format!("{:?}", ImportMode::ReadOnly);
    assert!(debug.contains("ReadOnly"));
}

#[test]
fn test_061_import_mode_hash_eq() {
    use std::collections::HashSet;
    let mut set = HashSet::new();
    set.insert(ImportMode::ReadOnly);
    set.insert(ImportMode::WriteOnly);
    assert!(set.contains(&ImportMode::ReadOnly));
    assert!(set.contains(&ImportMode::WriteOnly));
    assert!(!set.contains(&ImportMode::ReadWrite));
}

// =========================================================================
// SECTION 3: ResourceBarrier Tests (20+ tests)
// =========================================================================

#[test]
fn test_062_resource_barrier_new_creates_barrier() {
    let barrier = ResourceBarrier::new(
        ResourceHandle(1),
        ResourceAccess::Read,
        ResourceAccess::Write,
        true,
    );
    assert_eq!(barrier.resource, ResourceHandle(1));
    assert_eq!(barrier.before_access, ResourceAccess::Read);
    assert_eq!(barrier.after_access, ResourceAccess::Write);
    assert!(barrier.acquire);
}

#[test]
fn test_063_resource_barrier_acquire_factory() {
    let barrier = ResourceBarrier::acquire(ResourceHandle(5), ResourceAccess::Write);
    assert!(barrier.is_acquire());
    assert!(!barrier.is_release());
    assert_eq!(barrier.resource, ResourceHandle(5));
    assert_eq!(barrier.after_access, ResourceAccess::Write);
}

#[test]
fn test_064_resource_barrier_release_factory() {
    let barrier = ResourceBarrier::release(ResourceHandle(7), ResourceAccess::Write);
    assert!(!barrier.is_acquire());
    assert!(barrier.is_release());
    assert_eq!(barrier.resource, ResourceHandle(7));
    assert_eq!(barrier.before_access, ResourceAccess::Write);
}

#[test]
fn test_065_resource_barrier_is_acquire_true() {
    let barrier = ResourceBarrier::new(
        ResourceHandle(1),
        ResourceAccess::Read,
        ResourceAccess::Write,
        true,
    );
    assert!(barrier.is_acquire());
}

#[test]
fn test_066_resource_barrier_is_acquire_false() {
    let barrier = ResourceBarrier::new(
        ResourceHandle(1),
        ResourceAccess::Read,
        ResourceAccess::Write,
        false,
    );
    assert!(!barrier.is_acquire());
}

#[test]
fn test_067_resource_barrier_is_release_true() {
    let barrier = ResourceBarrier::new(
        ResourceHandle(1),
        ResourceAccess::Read,
        ResourceAccess::Write,
        false,
    );
    assert!(barrier.is_release());
}

#[test]
fn test_068_resource_barrier_is_release_false() {
    let barrier = ResourceBarrier::new(
        ResourceHandle(1),
        ResourceAccess::Read,
        ResourceAccess::Write,
        true,
    );
    assert!(!barrier.is_release());
}

#[test]
fn test_069_resource_barrier_display_acquire() {
    let barrier = ResourceBarrier::acquire(ResourceHandle(10), ResourceAccess::Write);
    let display = format!("{}", barrier);
    assert!(display.contains("Acquire"), "display={}", display);
}

#[test]
fn test_070_resource_barrier_display_release() {
    let barrier = ResourceBarrier::release(ResourceHandle(20), ResourceAccess::Write);
    let display = format!("{}", barrier);
    assert!(display.contains("Release"), "display={}", display);
}

#[test]
fn test_071_resource_barrier_display_contains_resource() {
    let barrier = ResourceBarrier::acquire(ResourceHandle(42), ResourceAccess::Read);
    let display = format!("{}", barrier);
    assert!(display.contains("42") || display.contains("ResourceHandle"), "display={}", display);
}

#[test]
fn test_072_resource_barrier_equality() {
    let a = ResourceBarrier::new(
        ResourceHandle(1),
        ResourceAccess::Read,
        ResourceAccess::Write,
        true,
    );
    let b = ResourceBarrier::new(
        ResourceHandle(1),
        ResourceAccess::Read,
        ResourceAccess::Write,
        true,
    );
    assert_eq!(a, b);
}

#[test]
fn test_073_resource_barrier_inequality_resource() {
    let a = ResourceBarrier::acquire(ResourceHandle(1), ResourceAccess::Write);
    let b = ResourceBarrier::acquire(ResourceHandle(2), ResourceAccess::Write);
    assert_ne!(a, b);
}

#[test]
fn test_074_resource_barrier_inequality_before_access() {
    let a = ResourceBarrier::new(ResourceHandle(1), ResourceAccess::Read, ResourceAccess::Write, true);
    let b = ResourceBarrier::new(ResourceHandle(1), ResourceAccess::Write, ResourceAccess::Write, true);
    assert_ne!(a, b);
}

#[test]
fn test_075_resource_barrier_inequality_after_access() {
    let a = ResourceBarrier::new(ResourceHandle(1), ResourceAccess::Read, ResourceAccess::Write, true);
    let b = ResourceBarrier::new(ResourceHandle(1), ResourceAccess::Read, ResourceAccess::Read, true);
    assert_ne!(a, b);
}

#[test]
fn test_076_resource_barrier_inequality_acquire_flag() {
    let a = ResourceBarrier::new(ResourceHandle(1), ResourceAccess::Read, ResourceAccess::Write, true);
    let b = ResourceBarrier::new(ResourceHandle(1), ResourceAccess::Read, ResourceAccess::Write, false);
    assert_ne!(a, b);
}

#[test]
fn test_077_resource_barrier_clone() {
    let original = ResourceBarrier::acquire(ResourceHandle(99), ResourceAccess::ReadWrite);
    let cloned = original.clone();
    assert_eq!(original, cloned);
}

#[test]
fn test_078_resource_barrier_debug_format() {
    let barrier = ResourceBarrier::release(ResourceHandle(1), ResourceAccess::Write);
    let debug = format!("{:?}", barrier);
    assert!(debug.contains("ResourceBarrier"));
}

#[test]
fn test_079_resource_barrier_all_access_combinations_acquire() {
    for access in [ResourceAccess::Read, ResourceAccess::Write, ResourceAccess::ReadWrite] {
        let barrier = ResourceBarrier::acquire(ResourceHandle(1), access);
        assert!(barrier.is_acquire());
        assert_eq!(barrier.after_access, access);
    }
}

#[test]
fn test_080_resource_barrier_all_access_combinations_release() {
    for access in [ResourceAccess::Read, ResourceAccess::Write, ResourceAccess::ReadWrite] {
        let barrier = ResourceBarrier::release(ResourceHandle(1), access);
        assert!(barrier.is_release());
        assert_eq!(barrier.before_access, access);
    }
}

#[test]
fn test_081_resource_barrier_same_access_transition() {
    // Same-state barriers are valid (no-op transitions)
    let barrier = ResourceBarrier::new(
        ResourceHandle(1),
        ResourceAccess::Read,
        ResourceAccess::Read,
        true,
    );
    assert_eq!(barrier.before_access, barrier.after_access);
}

// =========================================================================
// SECTION 4: ExternalResourceRegistry Tests (15+ tests)
// =========================================================================

#[test]
fn test_082_registry_new_is_empty() {
    let registry = ExternalResourceRegistry::new();
    assert!(registry.is_empty());
    assert_eq!(registry.count(), 0);
}

#[test]
fn test_083_registry_new_no_swapchain() {
    let registry = ExternalResourceRegistry::new();
    assert!(!registry.has_swapchain());
}

#[test]
fn test_084_registry_clear_resets_state() {
    let mut registry = ExternalResourceRegistry::new();
    registry.clear();
    assert!(registry.is_empty());
    assert_eq!(registry.count(), 0);
    assert!(!registry.has_swapchain());
}

#[test]
fn test_085_registry_display_empty() {
    let registry = ExternalResourceRegistry::new();
    let display = format!("{}", registry);
    assert!(display.contains("count=0"));
    assert!(display.contains("has_swapchain=false"));
}

#[test]
fn test_086_registry_default_is_empty() {
    let registry = ExternalResourceRegistry::default();
    assert!(registry.is_empty());
}

#[test]
fn test_087_registry_debug_format() {
    let registry = ExternalResourceRegistry::new();
    let debug = format!("{:?}", registry);
    assert!(debug.contains("ExternalResourceRegistry"));
}

#[test]
fn test_088_registry_swapchain_handle_none_when_empty() {
    let registry = ExternalResourceRegistry::new();
    assert!(registry.swapchain_handle().is_none());
}

#[test]
fn test_089_registry_get_swapchain_none_when_empty() {
    let registry = ExternalResourceRegistry::new();
    assert!(registry.get_swapchain().is_none());
}

#[test]
fn test_090_registry_get_unknown_handle_none() {
    let registry = ExternalResourceRegistry::new();
    assert!(registry.get(ResourceHandle(999)).is_none());
}

#[test]
fn test_091_registry_iter_empty() {
    let registry = ExternalResourceRegistry::new();
    assert_eq!(registry.iter().count(), 0);
}

#[test]
fn test_092_registry_handles_empty() {
    let registry = ExternalResourceRegistry::new();
    assert_eq!(registry.handles().count(), 0);
}

#[test]
fn test_093_registry_find_by_name_empty() {
    let registry = ExternalResourceRegistry::new();
    assert!(registry.find_by_name("any").is_none());
}

#[test]
fn test_094_registry_track_usage_unknown_handle_no_panic() {
    let mut registry = ExternalResourceRegistry::new();
    // Should not panic on unknown handle
    registry.track_usage(ResourceHandle(999), PassIndex(0));
}

#[test]
fn test_095_registry_release_swapchain_none_when_empty() {
    let mut registry = ExternalResourceRegistry::new();
    assert!(registry.release_swapchain().is_none());
}

#[test]
fn test_096_registry_count_accuracy() {
    let registry = ExternalResourceRegistry::new();
    // Without wgpu, we can only test empty registry
    assert_eq!(registry.count(), 0);
    assert!(registry.is_empty());
}

// =========================================================================
// SECTION 5: ExternalSynchronizer Tests (20+ tests)
// =========================================================================

#[test]
fn test_097_synchronizer_empty_registry_no_acquire_barriers() {
    let registry = ExternalResourceRegistry::new();
    let barriers = ExternalSynchronizer::compute_acquire_barriers(&registry);
    assert!(barriers.is_empty());
}

#[test]
fn test_098_synchronizer_empty_registry_no_release_barriers() {
    let registry = ExternalResourceRegistry::new();
    let barriers = ExternalSynchronizer::compute_release_barriers(&registry);
    assert!(barriers.is_empty());
}

#[test]
fn test_099_synchronizer_compute_all_barriers_empty() {
    let registry = ExternalResourceRegistry::new();
    let (acquire, release) = ExternalSynchronizer::compute_all_barriers(&registry);
    assert!(acquire.is_empty());
    assert!(release.is_empty());
}

#[test]
fn test_100_synchronizer_barriers_are_vectors() {
    let registry = ExternalResourceRegistry::new();
    let acquire = ExternalSynchronizer::compute_acquire_barriers(&registry);
    let release = ExternalSynchronizer::compute_release_barriers(&registry);
    // Verify return types are Vec<ResourceBarrier>
    let _: Vec<ResourceBarrier> = acquire;
    let _: Vec<ResourceBarrier> = release;
}

// =========================================================================
// SECTION 6: Barrier Computation Logic Tests (Simulated scenarios)
// =========================================================================

/// Test that readonly resources need acquire but not release
#[test]
fn test_101_barrier_logic_readonly_mode() {
    let mode = ImportMode::ReadOnly;
    // ReadOnly: needs acquire (to see external writes)
    assert!(mode.requires_acquire_barrier());
    // ReadOnly: no release needed (we didn't modify anything)
    assert!(!mode.requires_release_barrier());
}

/// Test that writeonly resources need release but not acquire
#[test]
fn test_102_barrier_logic_writeonly_mode() {
    let mode = ImportMode::WriteOnly;
    // WriteOnly: no acquire needed (we discard previous contents)
    assert!(!mode.requires_acquire_barrier());
    // WriteOnly: needs release (our writes must be visible)
    assert!(mode.requires_release_barrier());
}

/// Test that readwrite resources need both barriers
#[test]
fn test_103_barrier_logic_readwrite_mode() {
    let mode = ImportMode::ReadWrite;
    // ReadWrite: needs both
    assert!(mode.requires_acquire_barrier());
    assert!(mode.requires_release_barrier());
}

/// Simulate swapchain texture import each frame
#[test]
fn test_104_scenario_swapchain_texture_import() {
    // Swapchain is typically WriteOnly (we render to it, don't read previous frame)
    let mode = ImportMode::WriteOnly;
    assert!(!mode.requires_acquire_barrier());
    assert!(mode.requires_release_barrier());

    // External type check
    let ext_type = ExternalResourceType::Swapchain;
    assert!(ext_type.is_swapchain());
    assert!(ext_type.is_texture());
}

/// Simulate user-provided depth buffer import
#[test]
fn test_105_scenario_user_depth_buffer() {
    // User depth buffer might be read-write (read for transparency, write for opaque)
    let mode = ImportMode::ReadWrite;
    assert!(mode.requires_acquire_barrier());
    assert!(mode.requires_release_barrier());

    let ext_type = ExternalResourceType::UserTexture;
    assert!(ext_type.is_texture());
    assert!(ext_type.is_user_provided());
}

/// Simulate shared texture between passes
#[test]
fn test_106_scenario_shared_texture_between_passes() {
    let mode = ImportMode::ReadOnly;
    let ext_type = ExternalResourceType::SharedTexture;

    assert!(ext_type.is_shared());
    assert!(ext_type.is_texture());
    assert!(mode.requires_acquire_barrier());
}

/// Simulate compute buffer from external source
#[test]
fn test_107_scenario_compute_buffer_external() {
    let mode = ImportMode::ReadWrite;
    let ext_type = ExternalResourceType::SharedBuffer;

    assert!(ext_type.is_buffer());
    assert!(ext_type.is_shared());
    assert!(mode.requires_acquire_barrier());
    assert!(mode.requires_release_barrier());
}

/// Simulate upload staging buffer
#[test]
fn test_108_scenario_upload_staging_buffer() {
    // Upload staging: CPU writes, GPU reads -> ReadOnly from GPU perspective
    let mode = ImportMode::ReadOnly;
    let ext_type = ExternalResourceType::UserBuffer;

    assert!(ext_type.is_buffer());
    assert!(mode.requires_acquire_barrier());
    assert!(!mode.requires_release_barrier());
}

/// Simulate readback staging buffer
#[test]
fn test_109_scenario_readback_staging_buffer() {
    // Readback: GPU writes, CPU reads -> WriteOnly from GPU perspective
    let mode = ImportMode::WriteOnly;
    let ext_type = ExternalResourceType::UserBuffer;

    assert!(ext_type.is_buffer());
    assert!(!mode.requires_acquire_barrier());
    assert!(mode.requires_release_barrier());
}

/// Simulate texture atlas import
#[test]
fn test_110_scenario_texture_atlas_import() {
    // Texture atlas is typically read-only (sampling)
    let _mode = ImportMode::ReadOnly;
    let ext_type = ExternalResourceType::UserTexture;

    assert!(ext_type.is_texture());
    assert!(_mode.is_read());
    assert!(!_mode.is_write());
}

/// Simulate environment cubemap import
#[test]
fn test_111_scenario_environment_cubemap() {
    let mode = ImportMode::ReadOnly;
    let ext_type = ExternalResourceType::SharedTexture;

    assert!(ext_type.is_texture());
    assert!(ext_type.is_shared());
    assert!(mode.requires_acquire_barrier());
}

/// Simulate shadow map from previous frame
#[test]
fn test_112_scenario_shadow_map_previous_frame() {
    // Previous frame shadow map is read-only in current frame
    let mode = ImportMode::ReadOnly;
    let ext_type = ExternalResourceType::SharedTexture;

    assert!(ext_type.is_texture());
    assert!(mode.requires_acquire_barrier());
}

/// Simulate UI texture overlay
#[test]
fn test_113_scenario_ui_texture_overlay() {
    let mode = ImportMode::ReadOnly;
    let ext_type = ExternalResourceType::UserTexture;

    assert!(ext_type.is_user_provided());
    assert!(ext_type.is_texture());
}

// =========================================================================
// SECTION 7: Multiple Resource Modes (Mixed scenarios)
// =========================================================================

/// Test multiple resources with different modes
#[test]
fn test_114_multiple_resources_same_mode() {
    // Simulate multiple read-only textures
    let modes = [ImportMode::ReadOnly; 5];
    for mode in modes {
        assert!(mode.requires_acquire_barrier());
        assert!(!mode.requires_release_barrier());
    }
}

/// Test mixed modes in registry scenario
#[test]
fn test_115_mixed_modes_barrier_requirements() {
    let modes = [
        (ImportMode::ReadOnly, true, false),   // acquire, no release
        (ImportMode::WriteOnly, false, true),  // no acquire, release
        (ImportMode::ReadWrite, true, true),   // both
    ];

    for (mode, expected_acquire, expected_release) in modes {
        assert_eq!(
            mode.requires_acquire_barrier(),
            expected_acquire,
            "Mode {:?} acquire mismatch",
            mode
        );
        assert_eq!(
            mode.requires_release_barrier(),
            expected_release,
            "Mode {:?} release mismatch",
            mode
        );
    }
}

// =========================================================================
// SECTION 8: Swapchain Lifecycle (Logic tests)
// =========================================================================

/// Swapchain type predicates
#[test]
fn test_116_swapchain_type_predicates() {
    let swapchain = ExternalResourceType::Swapchain;
    assert!(swapchain.is_swapchain());
    assert!(swapchain.is_texture());
    assert!(!swapchain.is_buffer());
    assert!(!swapchain.is_user_provided());
    assert!(!swapchain.is_shared());
}

/// Swapchain default import mode
#[test]
fn test_117_swapchain_typical_import_mode() {
    // Swapchain is typically WriteOnly
    let mode = ImportMode::WriteOnly;
    assert!(mode.is_write());
    assert!(!mode.is_read());
}

/// Swapchain barrier requirements
#[test]
fn test_118_swapchain_barrier_requirements() {
    let mode = ImportMode::WriteOnly;
    // No acquire needed (fresh frame)
    assert!(!mode.requires_acquire_barrier());
    // Release needed for present
    assert!(mode.requires_release_barrier());
}

/// Registry swapchain handle when empty
#[test]
fn test_119_registry_swapchain_empty() {
    let registry = ExternalResourceRegistry::new();
    assert!(registry.swapchain_handle().is_none());
    assert!(registry.get_swapchain().is_none());
    assert!(!registry.has_swapchain());
}

/// Registry release swapchain when empty
#[test]
fn test_120_registry_release_swapchain_empty() {
    let mut registry = ExternalResourceRegistry::new();
    let result = registry.release_swapchain();
    assert!(result.is_none());
}

// =========================================================================
// SECTION 9: Edge Cases
// =========================================================================

/// Empty registry all operations
#[test]
fn test_121_empty_registry_operations() {
    let registry = ExternalResourceRegistry::new();
    assert!(registry.is_empty());
    assert_eq!(registry.count(), 0);
    assert!(!registry.has_swapchain());
    assert!(registry.get(ResourceHandle(0)).is_none());
    assert!(registry.get(ResourceHandle::NONE).is_none());
    assert!(registry.swapchain_handle().is_none());
    assert!(registry.get_swapchain().is_none());
    assert!(registry.find_by_name("test").is_none());
    assert_eq!(registry.iter().count(), 0);
    assert_eq!(registry.handles().count(), 0);
}

/// Resource handle NONE sentinel
#[test]
fn test_122_resource_handle_none_sentinel() {
    let none = ResourceHandle::NONE;
    assert_eq!(none.0, u32::MAX);
}

/// Resource handle display for NONE
#[test]
fn test_123_resource_handle_none_display() {
    let display = format!("{}", ResourceHandle::NONE);
    assert!(display.contains("NONE"));
}

/// Resource handle equality
#[test]
fn test_124_resource_handle_equality() {
    let a = ResourceHandle(42);
    let b = ResourceHandle(42);
    let c = ResourceHandle(43);
    assert_eq!(a, b);
    assert_ne!(a, c);
}

/// Resource handle ordering
#[test]
fn test_125_resource_handle_ordering() {
    let a = ResourceHandle(10);
    let b = ResourceHandle(20);
    assert!(a < b);
    assert!(b > a);
}

/// Resource handle hash
#[test]
fn test_126_resource_handle_hash() {
    use std::collections::HashSet;
    let mut set = HashSet::new();
    set.insert(ResourceHandle(1));
    set.insert(ResourceHandle(2));
    set.insert(ResourceHandle(1)); // Duplicate
    assert_eq!(set.len(), 2);
}

/// PassIndex display
#[test]
fn test_127_pass_index_display() {
    let idx = PassIndex(5);
    let display = format!("{}", idx);
    assert!(display.contains("5"));
    assert!(display.contains("PassIndex"));
}

/// PassIndex equality
#[test]
fn test_128_pass_index_equality() {
    let a = PassIndex(0);
    let b = PassIndex(0);
    let c = PassIndex(1);
    assert_eq!(a, b);
    assert_ne!(a, c);
}

/// PassIndex ordering
#[test]
fn test_129_pass_index_ordering() {
    let a = PassIndex(0);
    let b = PassIndex(100);
    assert!(a < b);
}

/// PassIndex hash
#[test]
fn test_130_pass_index_hash() {
    use std::collections::HashSet;
    let mut set = HashSet::new();
    set.insert(PassIndex(0));
    set.insert(PassIndex(1));
    assert!(set.contains(&PassIndex(0)));
    assert!(!set.contains(&PassIndex(99)));
}

/// Resource access display
#[test]
fn test_131_resource_access_display() {
    assert!(format!("{}", ResourceAccess::Read).contains("Read"));
    assert!(format!("{}", ResourceAccess::Write).contains("Write"));
    // ReadWrite display contains either both or a combined string
    let rw_display = format!("{}", ResourceAccess::ReadWrite);
    assert!(rw_display.contains("Read") || rw_display.contains("Write") || rw_display.contains("RW"));
}

/// All external types mutually exclusive categories
#[test]
fn test_132_external_types_category_exclusivity() {
    let types = [
        ExternalResourceType::Swapchain,
        ExternalResourceType::UserTexture,
        ExternalResourceType::UserBuffer,
        ExternalResourceType::SharedTexture,
        ExternalResourceType::SharedBuffer,
    ];

    for t in types {
        // Exactly one of: swapchain, user_provided, shared
        let categories = [
            t.is_swapchain(),
            t.is_user_provided(),
            t.is_shared(),
        ];
        let category_count = categories.iter().filter(|&&b| b).count();
        assert_eq!(
            category_count,
            1,
            "{:?} should be in exactly one category",
            t
        );
    }
}

/// All import modes have consistent read/write flags
#[test]
fn test_133_import_modes_read_write_consistency() {
    // ReadOnly: read=true, write=false
    assert!(ImportMode::ReadOnly.is_read());
    assert!(!ImportMode::ReadOnly.is_write());

    // WriteOnly: read=false, write=true
    assert!(!ImportMode::WriteOnly.is_read());
    assert!(ImportMode::WriteOnly.is_write());

    // ReadWrite: read=true, write=true
    assert!(ImportMode::ReadWrite.is_read());
    assert!(ImportMode::ReadWrite.is_write());
}

/// Barrier acquire flag consistency
#[test]
fn test_134_barrier_acquire_flag_consistency() {
    let acquire_barrier = ResourceBarrier::acquire(ResourceHandle(1), ResourceAccess::Read);
    let release_barrier = ResourceBarrier::release(ResourceHandle(1), ResourceAccess::Write);

    assert!(acquire_barrier.acquire);
    assert!(!release_barrier.acquire);

    assert!(acquire_barrier.is_acquire());
    assert!(!acquire_barrier.is_release());

    assert!(!release_barrier.is_acquire());
    assert!(release_barrier.is_release());
}

/// Zero resource handle
#[test]
fn test_135_resource_handle_zero() {
    let zero = ResourceHandle(0);
    assert_ne!(zero, ResourceHandle::NONE);
    assert_eq!(zero.0, 0);
}

/// Large resource handle values
#[test]
fn test_136_resource_handle_large_values() {
    let large = ResourceHandle(u32::MAX - 1);
    assert_ne!(large, ResourceHandle::NONE);
    // NONE is u32::MAX, so MAX-1 should be different
}

/// Large pass index values
#[test]
fn test_137_pass_index_large_values() {
    let large = PassIndex(usize::MAX);
    let display = format!("{}", large);
    assert!(display.contains("PassIndex"));
}

/// Barrier with NONE resource handle
#[test]
fn test_138_barrier_with_none_handle() {
    let barrier = ResourceBarrier::acquire(ResourceHandle::NONE, ResourceAccess::Read);
    assert_eq!(barrier.resource, ResourceHandle::NONE);
}

/// Registry clear multiple times
#[test]
fn test_139_registry_clear_multiple_times() {
    let mut registry = ExternalResourceRegistry::new();
    for _ in 0..10 {
        registry.clear();
        assert!(registry.is_empty());
    }
}

/// Import mode all combinations with external type
#[test]
fn test_140_import_mode_external_type_combinations() {
    let modes = [ImportMode::ReadOnly, ImportMode::WriteOnly, ImportMode::ReadWrite];
    let types = [
        ExternalResourceType::Swapchain,
        ExternalResourceType::UserTexture,
        ExternalResourceType::UserBuffer,
        ExternalResourceType::SharedTexture,
        ExternalResourceType::SharedBuffer,
    ];

    // All combinations should be valid (no panics)
    for mode in &modes {
        for ext_type in &types {
            let _ = mode.to_resource_access();
            let _ = ext_type.is_texture();
            let _ = ext_type.is_buffer();
        }
    }
}

// =========================================================================
// SECTION 10: Stress/Scale Tests (simulated)
// =========================================================================

/// Simulate many resource handles
#[test]
fn test_141_many_resource_handles() {
    let handles: Vec<ResourceHandle> = (0..1000).map(ResourceHandle).collect();
    assert_eq!(handles.len(), 1000);
    assert_eq!(handles[0], ResourceHandle(0));
    assert_eq!(handles[999], ResourceHandle(999));
}

/// Simulate many pass indices
#[test]
fn test_142_many_pass_indices() {
    let indices: Vec<PassIndex> = (0..1000).map(PassIndex).collect();
    assert_eq!(indices.len(), 1000);
    assert_eq!(indices[0], PassIndex(0));
    assert_eq!(indices[999], PassIndex(999));
}

/// Simulate many barriers
#[test]
fn test_143_many_barriers() {
    let barriers: Vec<ResourceBarrier> = (0..100)
        .map(|i| ResourceBarrier::acquire(ResourceHandle(i), ResourceAccess::Read))
        .collect();
    assert_eq!(barriers.len(), 100);
    assert!(barriers.iter().all(|b| b.is_acquire()));
}

/// Simulate barrier computation for many modes
#[test]
fn test_144_barrier_computation_many_modes() {
    let modes = [ImportMode::ReadOnly, ImportMode::WriteOnly, ImportMode::ReadWrite];
    let mut acquire_count = 0;
    let mut release_count = 0;

    for _ in 0..1000 {
        for mode in &modes {
            if mode.requires_acquire_barrier() {
                acquire_count += 1;
            }
            if mode.requires_release_barrier() {
                release_count += 1;
            }
        }
    }

    // ReadOnly: acquire only -> 1000
    // WriteOnly: release only -> 1000
    // ReadWrite: both -> 1000 each
    assert_eq!(acquire_count, 2000); // ReadOnly + ReadWrite
    assert_eq!(release_count, 2000); // WriteOnly + ReadWrite
}

// =========================================================================
// SECTION 11: Additional Contract Tests
// =========================================================================

/// External type is either texture or buffer, never neither
#[test]
fn test_145_external_type_always_texture_or_buffer() {
    for ext_type in [
        ExternalResourceType::Swapchain,
        ExternalResourceType::UserTexture,
        ExternalResourceType::UserBuffer,
        ExternalResourceType::SharedTexture,
        ExternalResourceType::SharedBuffer,
    ] {
        let is_texture = ext_type.is_texture();
        let is_buffer = ext_type.is_buffer();
        assert!(
            is_texture || is_buffer,
            "{:?} is neither texture nor buffer",
            ext_type
        );
    }
}

/// Import mode conversion is bidirectional
#[test]
fn test_146_import_mode_bidirectional_conversion() {
    for original in [ImportMode::ReadOnly, ImportMode::WriteOnly, ImportMode::ReadWrite] {
        let access = original.to_resource_access();
        let converted = ImportMode::from(access);
        assert_eq!(original, converted);
    }
}

/// Resource access conversion is bidirectional
#[test]
fn test_147_resource_access_bidirectional_conversion() {
    for original in [ResourceAccess::Read, ResourceAccess::Write, ResourceAccess::ReadWrite] {
        let mode = ImportMode::from(original);
        let converted = mode.to_resource_access();
        assert_eq!(original, converted);
    }
}

/// Barrier before/after can be same access
#[test]
fn test_148_barrier_same_before_after() {
    let barrier = ResourceBarrier::new(
        ResourceHandle(1),
        ResourceAccess::Read,
        ResourceAccess::Read,
        true,
    );
    assert_eq!(barrier.before_access, barrier.after_access);
}

/// Barrier all access state combinations
#[test]
fn test_149_barrier_all_access_state_combinations() {
    let states = [ResourceAccess::Read, ResourceAccess::Write, ResourceAccess::ReadWrite];

    for before in &states {
        for after in &states {
            let barrier = ResourceBarrier::new(
                ResourceHandle(1),
                *before,
                *after,
                true,
            );
            assert_eq!(barrier.before_access, *before);
            assert_eq!(barrier.after_access, *after);
        }
    }
}

/// Registry is_empty and count consistent
#[test]
fn test_150_registry_empty_count_consistency() {
    let registry = ExternalResourceRegistry::new();
    assert_eq!(registry.is_empty(), registry.count() == 0);
}

/// Registry has_swapchain and swapchain_handle consistent
#[test]
fn test_151_registry_swapchain_consistency() {
    let registry = ExternalResourceRegistry::new();
    assert_eq!(registry.has_swapchain(), registry.swapchain_handle().is_some());
}

/// External synchronizer returns fresh vectors
#[test]
fn test_152_synchronizer_returns_fresh_vectors() {
    let registry = ExternalResourceRegistry::new();

    let acquire1 = ExternalSynchronizer::compute_acquire_barriers(&registry);
    let acquire2 = ExternalSynchronizer::compute_acquire_barriers(&registry);

    // Should be equal but separate allocations
    assert_eq!(acquire1, acquire2);
}

/// All barrier types display correctly
#[test]
fn test_153_all_barrier_display_formats() {
    let barriers = [
        ResourceBarrier::acquire(ResourceHandle(1), ResourceAccess::Read),
        ResourceBarrier::acquire(ResourceHandle(2), ResourceAccess::Write),
        ResourceBarrier::acquire(ResourceHandle(3), ResourceAccess::ReadWrite),
        ResourceBarrier::release(ResourceHandle(4), ResourceAccess::Read),
        ResourceBarrier::release(ResourceHandle(5), ResourceAccess::Write),
        ResourceBarrier::release(ResourceHandle(6), ResourceAccess::ReadWrite),
    ];

    for barrier in &barriers {
        let display = format!("{}", barrier);
        assert!(!display.is_empty());
        // Should contain either Acquire or Release
        assert!(
            display.contains("Acquire") || display.contains("Release"),
            "Display format missing barrier type: {}",
            display
        );
    }
}

/// External resource type copy semantics
#[test]
fn test_154_external_type_copy_semantics() {
    let original = ExternalResourceType::SharedTexture;
    let copied = original;
    // Original still usable after copy
    assert!(original.is_shared());
    assert_eq!(original, copied);
}

/// Import mode copy semantics
#[test]
fn test_155_import_mode_copy_semantics() {
    let original = ImportMode::ReadWrite;
    let copied = original;
    // Original still usable after copy
    assert!(original.is_read());
    assert_eq!(original, copied);
}

/// Resource barrier clone semantics
#[test]
fn test_156_barrier_clone_semantics() {
    let original = ResourceBarrier::acquire(ResourceHandle(42), ResourceAccess::Write);
    let cloned = original.clone();

    // Both should be equal
    assert_eq!(original, cloned);
    // And independently usable
    assert!(original.is_acquire());
    assert!(cloned.is_acquire());
}

/// Registry default and new are equivalent
#[test]
fn test_157_registry_default_equals_new() {
    let default_reg = ExternalResourceRegistry::default();
    let new_reg = ExternalResourceRegistry::new();

    assert_eq!(default_reg.count(), new_reg.count());
    assert_eq!(default_reg.is_empty(), new_reg.is_empty());
    assert_eq!(default_reg.has_swapchain(), new_reg.has_swapchain());
}

/// External type eq/ne reflexivity
#[test]
fn test_158_external_type_eq_reflexive() {
    for t in [
        ExternalResourceType::Swapchain,
        ExternalResourceType::UserTexture,
        ExternalResourceType::UserBuffer,
        ExternalResourceType::SharedTexture,
        ExternalResourceType::SharedBuffer,
    ] {
        assert_eq!(t, t);
    }
}

/// Import mode eq/ne reflexivity
#[test]
fn test_159_import_mode_eq_reflexive() {
    for m in [ImportMode::ReadOnly, ImportMode::WriteOnly, ImportMode::ReadWrite] {
        assert_eq!(m, m);
    }
}

/// Resource handle eq/ne reflexivity
#[test]
fn test_160_resource_handle_eq_reflexive() {
    for h in [ResourceHandle(0), ResourceHandle(100), ResourceHandle::NONE] {
        assert_eq!(h, h);
    }
}

/// Verify barrier equality is symmetric
#[test]
fn test_161_barrier_eq_symmetric() {
    let a = ResourceBarrier::acquire(ResourceHandle(1), ResourceAccess::Read);
    let b = ResourceBarrier::acquire(ResourceHandle(1), ResourceAccess::Read);

    assert_eq!(a, b);
    assert_eq!(b, a);
}

/// Verify barrier equality is transitive
#[test]
fn test_162_barrier_eq_transitive() {
    let a = ResourceBarrier::release(ResourceHandle(5), ResourceAccess::Write);
    let b = ResourceBarrier::release(ResourceHandle(5), ResourceAccess::Write);
    let c = ResourceBarrier::release(ResourceHandle(5), ResourceAccess::Write);

    assert_eq!(a, b);
    assert_eq!(b, c);
    assert_eq!(a, c);
}
