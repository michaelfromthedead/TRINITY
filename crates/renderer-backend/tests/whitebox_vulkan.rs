// WHITEBOX tests for T-WGPU-P7.2.1 (Vulkan Features)
//
// WHITEBOX discipline: These tests have FULL ACCESS to the implementation.
// They exercise internal code paths, branch conditions, and edge cases
// that are not visible through the public contract alone.
//
// Implementation under test: crates/renderer-backend/src/backend/vulkan.rs
//   - VulkanVersion (constants, from_raw, to_raw, is_at_least, Display, Ord)
//   - VulkanDeviceType (variants, from_wgpu, is_gpu, is_hardware, Display)
//   - VulkanFeatures (detect, supports_*, ray_tracing_tier, summary, extensions)
//   - VulkanInfo (from_adapter, summary, is_suitable)
//   - VulkanRayTracingTier (is_available, is_full, name, Display)
//   - raw module (get_*_handle functions)
//
// WHITEBOX coverage plan:
//   Section 1: VulkanVersion Tests (35 tests)
//     - Version constant raw values verification
//     - from_raw() edge cases (zero, max, boundary values)
//     - to_raw() encoding verification
//     - Round-trip encoding/decoding
//     - Ordering (PartialOrd, Ord)
//     - is_at_least() boundary conditions
//     - Display formatting
//     - Default trait
//     - Clone/Copy semantics
//     - Hash trait
//
//   Section 2: VulkanDeviceType Tests (25 tests)
//     - All variant construction
//     - is_gpu() for all variants
//     - is_hardware() for all variants
//     - name() for all variants
//     - from_wgpu() conversion
//     - Default trait
//     - Display formatting
//     - Clone semantics
//     - PartialEq comparisons
//     - Hash trait
//
//   Section 3: VulkanRayTracingTier Tests (20 tests)
//     - All variant construction
//     - is_available() for all variants
//     - is_full() for all variants
//     - name() for all variants
//     - Default trait
//     - Display formatting
//     - Clone/Copy semantics
//     - PartialEq comparisons
//
//   Section 4: VulkanFeatures Tests (55 tests)
//     - Default all-false state
//     - Individual feature flags
//     - supports_rt_pipeline() combinations
//     - supports_ray_query() tests
//     - supports_any_rt() combinations
//     - supports_bindless() combinations
//     - supports_vulkan_1_2() combinations
//     - supports_vulkan_1_3() combinations
//     - supports_mesh_shading() tests
//     - supports_modern_sync() combinations
//     - ray_tracing_tier() tier determination
//     - summary() output variations
//     - from_features() wgpu feature mapping
//     - Extension list verification
//
//   Section 5: VulkanInfo Tests (25 tests)
//     - Construction with all fields
//     - Default values
//     - summary() output
//     - is_suitable() requirements
//     - supports_ray_tracing() delegation
//     - Field accessors
//
//   Section 6: Edge Cases & Boundary Tests (20 tests)
//     - Maximum version components
//     - Zero values throughout
//     - Empty strings
//     - All features enabled
//     - All features disabled
//     - Mixed feature states

use renderer_backend::backend::vulkan::{
    VulkanDeviceType, VulkanFeatures, VulkanInfo, VulkanRayTracingTier, VulkanVersion,
};
use wgpu::Features;

// ============================================================================
// Section 1: VulkanVersion Tests
// ============================================================================

// ----------------------------------------------------------------------------
// 1.1 Version Constant Values
// ----------------------------------------------------------------------------

#[test]
fn test_version_v1_0_raw_value() {
    // Vulkan 1.0.0 raw encoding: (1 << 22) | (0 << 12) | 0 = 0x00400000
    assert_eq!(VulkanVersion::V1_0.to_raw(), 0x00400000);
}

#[test]
fn test_version_v1_1_raw_value() {
    // Vulkan 1.1.0 raw encoding: (1 << 22) | (1 << 12) | 0 = 0x00401000
    assert_eq!(VulkanVersion::V1_1.to_raw(), 0x00401000);
}

#[test]
fn test_version_v1_2_raw_value() {
    // Vulkan 1.2.0 raw encoding: (1 << 22) | (2 << 12) | 0 = 0x00402000
    assert_eq!(VulkanVersion::V1_2.to_raw(), 0x00402000);
}

#[test]
fn test_version_v1_3_raw_value() {
    // Vulkan 1.3.0 raw encoding: (1 << 22) | (3 << 12) | 0 = 0x00403000
    assert_eq!(VulkanVersion::V1_3.to_raw(), 0x00403000);
}

#[test]
fn test_version_constants_have_correct_components() {
    assert_eq!(VulkanVersion::V1_0.major, 1);
    assert_eq!(VulkanVersion::V1_0.minor, 0);
    assert_eq!(VulkanVersion::V1_0.patch, 0);

    assert_eq!(VulkanVersion::V1_1.major, 1);
    assert_eq!(VulkanVersion::V1_1.minor, 1);
    assert_eq!(VulkanVersion::V1_1.patch, 0);

    assert_eq!(VulkanVersion::V1_2.major, 1);
    assert_eq!(VulkanVersion::V1_2.minor, 2);
    assert_eq!(VulkanVersion::V1_2.patch, 0);

    assert_eq!(VulkanVersion::V1_3.major, 1);
    assert_eq!(VulkanVersion::V1_3.minor, 3);
    assert_eq!(VulkanVersion::V1_3.patch, 0);
}

// ----------------------------------------------------------------------------
// 1.2 from_raw() Edge Cases
// ----------------------------------------------------------------------------

#[test]
fn test_version_from_raw_zero() {
    let version = VulkanVersion::from_raw(0);
    assert_eq!(version.major, 0);
    assert_eq!(version.minor, 0);
    assert_eq!(version.patch, 0);
}

#[test]
fn test_version_from_raw_max_major() {
    // Max major (10 bits) = 0x3FF = 1023
    let raw = 0x3FF << 22;
    let version = VulkanVersion::from_raw(raw);
    assert_eq!(version.major, 0x3FF);
    assert_eq!(version.minor, 0);
    assert_eq!(version.patch, 0);
}

#[test]
fn test_version_from_raw_max_minor() {
    // Max minor (10 bits) = 0x3FF = 1023
    let raw = 0x3FF << 12;
    let version = VulkanVersion::from_raw(raw);
    assert_eq!(version.major, 0);
    assert_eq!(version.minor, 0x3FF);
    assert_eq!(version.patch, 0);
}

#[test]
fn test_version_from_raw_max_patch() {
    // Max patch (12 bits) = 0xFFF = 4095
    let raw = 0xFFF;
    let version = VulkanVersion::from_raw(raw);
    assert_eq!(version.major, 0);
    assert_eq!(version.minor, 0);
    assert_eq!(version.patch, 0xFFF);
}

#[test]
fn test_version_from_raw_all_max() {
    // All components at max: (0x3FF << 22) | (0x3FF << 12) | 0xFFF
    let raw = 0xFFFFFFFF;
    let version = VulkanVersion::from_raw(raw);
    assert_eq!(version.major, 0x3FF);
    assert_eq!(version.minor, 0x3FF);
    assert_eq!(version.patch, 0xFFF);
}

#[test]
fn test_version_from_raw_v1_3_250() {
    // Vulkan 1.3.250 = (1 << 22) | (3 << 12) | 250
    let raw = 0x004030FA;
    let version = VulkanVersion::from_raw(raw);
    assert_eq!(version.major, 1);
    assert_eq!(version.minor, 3);
    assert_eq!(version.patch, 250);
}

#[test]
fn test_version_from_raw_v2_0_0() {
    // Hypothetical Vulkan 2.0.0 = (2 << 22) | 0 | 0
    let raw = 0x00800000;
    let version = VulkanVersion::from_raw(raw);
    assert_eq!(version.major, 2);
    assert_eq!(version.minor, 0);
    assert_eq!(version.patch, 0);
}

// ----------------------------------------------------------------------------
// 1.3 to_raw() Encoding Verification
// ----------------------------------------------------------------------------

#[test]
fn test_version_to_raw_zero_version() {
    let version = VulkanVersion::new(0, 0, 0);
    assert_eq!(version.to_raw(), 0);
}

#[test]
fn test_version_to_raw_preserves_major_only() {
    let version = VulkanVersion::new(5, 0, 0);
    assert_eq!(version.to_raw(), 5 << 22);
}

#[test]
fn test_version_to_raw_preserves_minor_only() {
    let version = VulkanVersion::new(0, 7, 0);
    assert_eq!(version.to_raw(), 7 << 12);
}

#[test]
fn test_version_to_raw_preserves_patch_only() {
    let version = VulkanVersion::new(0, 0, 123);
    assert_eq!(version.to_raw(), 123);
}

#[test]
fn test_version_to_raw_truncates_overflow_major() {
    // Major values > 0x3FF should be masked
    let version = VulkanVersion::new(0x400, 0, 0); // 1024, overflows 10 bits
    assert_eq!(version.to_raw(), 0); // Gets masked to 0
}

#[test]
fn test_version_to_raw_truncates_overflow_minor() {
    let version = VulkanVersion::new(0, 0x400, 0);
    assert_eq!(version.to_raw(), 0);
}

#[test]
fn test_version_to_raw_truncates_overflow_patch() {
    let version = VulkanVersion::new(0, 0, 0x1000); // 4096, overflows 12 bits
    assert_eq!(version.to_raw(), 0);
}

// ----------------------------------------------------------------------------
// 1.4 Round-trip Encoding/Decoding
// ----------------------------------------------------------------------------

#[test]
fn test_version_round_trip_v1_0() {
    let original = VulkanVersion::V1_0;
    let decoded = VulkanVersion::from_raw(original.to_raw());
    assert_eq!(original, decoded);
}

#[test]
fn test_version_round_trip_v1_1() {
    let original = VulkanVersion::V1_1;
    let decoded = VulkanVersion::from_raw(original.to_raw());
    assert_eq!(original, decoded);
}

#[test]
fn test_version_round_trip_v1_2() {
    let original = VulkanVersion::V1_2;
    let decoded = VulkanVersion::from_raw(original.to_raw());
    assert_eq!(original, decoded);
}

#[test]
fn test_version_round_trip_v1_3() {
    let original = VulkanVersion::V1_3;
    let decoded = VulkanVersion::from_raw(original.to_raw());
    assert_eq!(original, decoded);
}

#[test]
fn test_version_round_trip_arbitrary() {
    let original = VulkanVersion::new(1, 3, 250);
    let decoded = VulkanVersion::from_raw(original.to_raw());
    assert_eq!(original, decoded);
}

#[test]
fn test_version_round_trip_max_values() {
    let original = VulkanVersion::new(0x3FF, 0x3FF, 0xFFF);
    let decoded = VulkanVersion::from_raw(original.to_raw());
    assert_eq!(original, decoded);
}

// ----------------------------------------------------------------------------
// 1.5 Version Ordering
// ----------------------------------------------------------------------------

#[test]
fn test_version_ord_v1_0_less_than_v1_1() {
    assert!(VulkanVersion::V1_0 < VulkanVersion::V1_1);
}

#[test]
fn test_version_ord_v1_1_less_than_v1_2() {
    assert!(VulkanVersion::V1_1 < VulkanVersion::V1_2);
}

#[test]
fn test_version_ord_v1_2_less_than_v1_3() {
    assert!(VulkanVersion::V1_2 < VulkanVersion::V1_3);
}

#[test]
fn test_version_ord_greater_than() {
    assert!(VulkanVersion::V1_3 > VulkanVersion::V1_0);
    assert!(VulkanVersion::V1_3 > VulkanVersion::V1_1);
    assert!(VulkanVersion::V1_3 > VulkanVersion::V1_2);
}

#[test]
fn test_version_ord_equality() {
    assert_eq!(VulkanVersion::V1_2, VulkanVersion::V1_2);
    assert!(VulkanVersion::V1_2 <= VulkanVersion::V1_2);
    assert!(VulkanVersion::V1_2 >= VulkanVersion::V1_2);
}

#[test]
fn test_version_ord_patch_affects_ordering() {
    let v1 = VulkanVersion::new(1, 3, 0);
    let v2 = VulkanVersion::new(1, 3, 250);
    assert!(v1 < v2);
}

#[test]
fn test_version_ord_minor_takes_precedence_over_patch() {
    let v1 = VulkanVersion::new(1, 2, 999);
    let v2 = VulkanVersion::new(1, 3, 0);
    assert!(v1 < v2);
}

#[test]
fn test_version_ord_major_takes_precedence_over_all() {
    let v1 = VulkanVersion::new(1, 999, 999);
    let v2 = VulkanVersion::new(2, 0, 0);
    assert!(v1 < v2);
}

// ----------------------------------------------------------------------------
// 1.6 is_at_least() Boundary Conditions
// ----------------------------------------------------------------------------

#[test]
fn test_version_is_at_least_exact_match() {
    assert!(VulkanVersion::V1_2.is_at_least(1, 2));
}

#[test]
fn test_version_is_at_least_lower_version() {
    assert!(VulkanVersion::V1_3.is_at_least(1, 0));
    assert!(VulkanVersion::V1_3.is_at_least(1, 1));
    assert!(VulkanVersion::V1_3.is_at_least(1, 2));
    assert!(VulkanVersion::V1_3.is_at_least(1, 3));
}

#[test]
fn test_version_is_at_least_higher_version_fails() {
    assert!(!VulkanVersion::V1_2.is_at_least(1, 3));
    assert!(!VulkanVersion::V1_0.is_at_least(1, 1));
}

#[test]
fn test_version_is_at_least_higher_major_fails() {
    assert!(!VulkanVersion::V1_3.is_at_least(2, 0));
}

#[test]
fn test_version_is_at_least_lower_major_passes() {
    let v2 = VulkanVersion::new(2, 0, 0);
    assert!(v2.is_at_least(1, 3));
    assert!(v2.is_at_least(1, 0));
}

#[test]
fn test_version_is_at_least_ignores_patch() {
    let v = VulkanVersion::new(1, 2, 0);
    assert!(v.is_at_least(1, 2)); // Patch doesn't matter
}

#[test]
fn test_version_is_at_least_zero_zero() {
    assert!(VulkanVersion::V1_0.is_at_least(0, 0));
    assert!(VulkanVersion::new(0, 0, 0).is_at_least(0, 0));
}

// ----------------------------------------------------------------------------
// 1.7 Display Formatting
// ----------------------------------------------------------------------------

#[test]
fn test_version_display_v1_0() {
    assert_eq!(format!("{}", VulkanVersion::V1_0), "1.0.0");
}

#[test]
fn test_version_display_v1_1() {
    assert_eq!(format!("{}", VulkanVersion::V1_1), "1.1.0");
}

#[test]
fn test_version_display_v1_2() {
    assert_eq!(format!("{}", VulkanVersion::V1_2), "1.2.0");
}

#[test]
fn test_version_display_v1_3() {
    assert_eq!(format!("{}", VulkanVersion::V1_3), "1.3.0");
}

#[test]
fn test_version_display_with_patch() {
    let v = VulkanVersion::new(1, 3, 250);
    assert_eq!(format!("{}", v), "1.3.250");
}

#[test]
fn test_version_display_zero() {
    let v = VulkanVersion::new(0, 0, 0);
    assert_eq!(format!("{}", v), "0.0.0");
}

#[test]
fn test_version_display_large_values() {
    let v = VulkanVersion::new(999, 999, 4000);
    assert_eq!(format!("{}", v), "999.999.4000");
}

// ----------------------------------------------------------------------------
// 1.8 Default and Clone/Copy/Hash
// ----------------------------------------------------------------------------

#[test]
fn test_version_default_is_v1_0() {
    let version = VulkanVersion::default();
    assert_eq!(version, VulkanVersion::V1_0);
}

#[test]
fn test_version_clone() {
    let v1 = VulkanVersion::V1_2;
    let v2 = v1.clone();
    assert_eq!(v1, v2);
}

#[test]
fn test_version_copy() {
    let v1 = VulkanVersion::V1_3;
    let v2 = v1; // Copy
    assert_eq!(v1, v2); // v1 still usable
}

#[test]
fn test_version_hash_equality() {
    use std::collections::HashSet;
    let mut set = HashSet::new();
    set.insert(VulkanVersion::V1_2);
    assert!(set.contains(&VulkanVersion::V1_2));
    assert!(!set.contains(&VulkanVersion::V1_3));
}

#[test]
fn test_version_hash_different_versions() {
    use std::collections::HashSet;
    let mut set = HashSet::new();
    set.insert(VulkanVersion::V1_0);
    set.insert(VulkanVersion::V1_1);
    set.insert(VulkanVersion::V1_2);
    set.insert(VulkanVersion::V1_3);
    assert_eq!(set.len(), 4);
}

// ============================================================================
// Section 2: VulkanDeviceType Tests
// ============================================================================

// ----------------------------------------------------------------------------
// 2.1 Variant Construction
// ----------------------------------------------------------------------------

#[test]
fn test_device_type_discrete_gpu() {
    let dt = VulkanDeviceType::DiscreteGpu;
    assert_eq!(dt, VulkanDeviceType::DiscreteGpu);
}

#[test]
fn test_device_type_integrated_gpu() {
    let dt = VulkanDeviceType::IntegratedGpu;
    assert_eq!(dt, VulkanDeviceType::IntegratedGpu);
}

#[test]
fn test_device_type_virtual_gpu() {
    let dt = VulkanDeviceType::VirtualGpu;
    assert_eq!(dt, VulkanDeviceType::VirtualGpu);
}

#[test]
fn test_device_type_cpu() {
    let dt = VulkanDeviceType::Cpu;
    assert_eq!(dt, VulkanDeviceType::Cpu);
}

#[test]
fn test_device_type_other() {
    let dt = VulkanDeviceType::Other;
    assert_eq!(dt, VulkanDeviceType::Other);
}

// ----------------------------------------------------------------------------
// 2.2 is_gpu() for All Variants
// ----------------------------------------------------------------------------

#[test]
fn test_device_type_discrete_gpu_is_gpu() {
    assert!(VulkanDeviceType::DiscreteGpu.is_gpu());
}

#[test]
fn test_device_type_integrated_gpu_is_gpu() {
    assert!(VulkanDeviceType::IntegratedGpu.is_gpu());
}

#[test]
fn test_device_type_virtual_gpu_is_gpu() {
    assert!(VulkanDeviceType::VirtualGpu.is_gpu());
}

#[test]
fn test_device_type_cpu_is_not_gpu() {
    assert!(!VulkanDeviceType::Cpu.is_gpu());
}

#[test]
fn test_device_type_other_is_not_gpu() {
    assert!(!VulkanDeviceType::Other.is_gpu());
}

// ----------------------------------------------------------------------------
// 2.3 is_hardware() for All Variants
// ----------------------------------------------------------------------------

#[test]
fn test_device_type_discrete_gpu_is_hardware() {
    assert!(VulkanDeviceType::DiscreteGpu.is_hardware());
}

#[test]
fn test_device_type_integrated_gpu_is_hardware() {
    assert!(VulkanDeviceType::IntegratedGpu.is_hardware());
}

#[test]
fn test_device_type_virtual_gpu_is_hardware() {
    assert!(VulkanDeviceType::VirtualGpu.is_hardware());
}

#[test]
fn test_device_type_cpu_is_not_hardware() {
    assert!(!VulkanDeviceType::Cpu.is_hardware());
}

#[test]
fn test_device_type_other_is_not_hardware() {
    assert!(!VulkanDeviceType::Other.is_hardware());
}

// ----------------------------------------------------------------------------
// 2.4 name() for All Variants
// ----------------------------------------------------------------------------

#[test]
fn test_device_type_discrete_gpu_name() {
    assert_eq!(VulkanDeviceType::DiscreteGpu.name(), "Discrete GPU");
}

#[test]
fn test_device_type_integrated_gpu_name() {
    assert_eq!(VulkanDeviceType::IntegratedGpu.name(), "Integrated GPU");
}

#[test]
fn test_device_type_virtual_gpu_name() {
    assert_eq!(VulkanDeviceType::VirtualGpu.name(), "Virtual GPU");
}

#[test]
fn test_device_type_cpu_name() {
    assert_eq!(VulkanDeviceType::Cpu.name(), "CPU");
}

#[test]
fn test_device_type_other_name() {
    assert_eq!(VulkanDeviceType::Other.name(), "Other");
}

// ----------------------------------------------------------------------------
// 2.5 from_wgpu() Conversion
// ----------------------------------------------------------------------------

#[test]
fn test_device_type_from_wgpu_discrete() {
    let dt = VulkanDeviceType::from_wgpu(wgpu::DeviceType::DiscreteGpu);
    assert_eq!(dt, VulkanDeviceType::DiscreteGpu);
}

#[test]
fn test_device_type_from_wgpu_integrated() {
    let dt = VulkanDeviceType::from_wgpu(wgpu::DeviceType::IntegratedGpu);
    assert_eq!(dt, VulkanDeviceType::IntegratedGpu);
}

#[test]
fn test_device_type_from_wgpu_virtual() {
    let dt = VulkanDeviceType::from_wgpu(wgpu::DeviceType::VirtualGpu);
    assert_eq!(dt, VulkanDeviceType::VirtualGpu);
}

#[test]
fn test_device_type_from_wgpu_cpu() {
    let dt = VulkanDeviceType::from_wgpu(wgpu::DeviceType::Cpu);
    assert_eq!(dt, VulkanDeviceType::Cpu);
}

#[test]
fn test_device_type_from_wgpu_other() {
    let dt = VulkanDeviceType::from_wgpu(wgpu::DeviceType::Other);
    assert_eq!(dt, VulkanDeviceType::Other);
}

// ----------------------------------------------------------------------------
// 2.6 Default, Display, Clone, Hash
// ----------------------------------------------------------------------------

#[test]
fn test_device_type_default_is_other() {
    let dt = VulkanDeviceType::default();
    assert_eq!(dt, VulkanDeviceType::Other);
}

#[test]
fn test_device_type_display_discrete() {
    assert_eq!(format!("{}", VulkanDeviceType::DiscreteGpu), "Discrete GPU");
}

#[test]
fn test_device_type_display_integrated() {
    assert_eq!(format!("{}", VulkanDeviceType::IntegratedGpu), "Integrated GPU");
}

#[test]
fn test_device_type_display_virtual() {
    assert_eq!(format!("{}", VulkanDeviceType::VirtualGpu), "Virtual GPU");
}

#[test]
fn test_device_type_display_cpu() {
    assert_eq!(format!("{}", VulkanDeviceType::Cpu), "CPU");
}

#[test]
fn test_device_type_display_other() {
    assert_eq!(format!("{}", VulkanDeviceType::Other), "Other");
}

#[test]
fn test_device_type_clone() {
    let dt1 = VulkanDeviceType::DiscreteGpu;
    let dt2 = dt1.clone();
    assert_eq!(dt1, dt2);
}

#[test]
fn test_device_type_copy() {
    let dt1 = VulkanDeviceType::IntegratedGpu;
    let dt2 = dt1;
    assert_eq!(dt1, dt2);
}

#[test]
fn test_device_type_hash() {
    use std::collections::HashSet;
    let mut set = HashSet::new();
    set.insert(VulkanDeviceType::DiscreteGpu);
    set.insert(VulkanDeviceType::IntegratedGpu);
    set.insert(VulkanDeviceType::VirtualGpu);
    set.insert(VulkanDeviceType::Cpu);
    set.insert(VulkanDeviceType::Other);
    assert_eq!(set.len(), 5);
}

// ============================================================================
// Section 3: VulkanRayTracingTier Tests
// ============================================================================

// ----------------------------------------------------------------------------
// 3.1 Variant Construction and Default
// ----------------------------------------------------------------------------

#[test]
fn test_rt_tier_none_variant() {
    let tier = VulkanRayTracingTier::None;
    assert_eq!(tier, VulkanRayTracingTier::None);
}

#[test]
fn test_rt_tier_query_variant() {
    let tier = VulkanRayTracingTier::Query;
    assert_eq!(tier, VulkanRayTracingTier::Query);
}

#[test]
fn test_rt_tier_full_variant() {
    let tier = VulkanRayTracingTier::Full;
    assert_eq!(tier, VulkanRayTracingTier::Full);
}

#[test]
fn test_rt_tier_default_is_none() {
    let tier = VulkanRayTracingTier::default();
    assert_eq!(tier, VulkanRayTracingTier::None);
}

// ----------------------------------------------------------------------------
// 3.2 is_available() for All Variants
// ----------------------------------------------------------------------------

#[test]
fn test_rt_tier_none_not_available() {
    assert!(!VulkanRayTracingTier::None.is_available());
}

#[test]
fn test_rt_tier_query_is_available() {
    assert!(VulkanRayTracingTier::Query.is_available());
}

#[test]
fn test_rt_tier_full_is_available() {
    assert!(VulkanRayTracingTier::Full.is_available());
}

// ----------------------------------------------------------------------------
// 3.3 is_full() for All Variants
// ----------------------------------------------------------------------------

#[test]
fn test_rt_tier_none_not_full() {
    assert!(!VulkanRayTracingTier::None.is_full());
}

#[test]
fn test_rt_tier_query_not_full() {
    assert!(!VulkanRayTracingTier::Query.is_full());
}

#[test]
fn test_rt_tier_full_is_full() {
    assert!(VulkanRayTracingTier::Full.is_full());
}

// ----------------------------------------------------------------------------
// 3.4 name() for All Variants
// ----------------------------------------------------------------------------

#[test]
fn test_rt_tier_none_name() {
    assert_eq!(VulkanRayTracingTier::None.name(), "None");
}

#[test]
fn test_rt_tier_query_name() {
    assert_eq!(VulkanRayTracingTier::Query.name(), "Ray Query");
}

#[test]
fn test_rt_tier_full_name() {
    assert_eq!(VulkanRayTracingTier::Full.name(), "Full Pipeline");
}

// ----------------------------------------------------------------------------
// 3.5 Display Formatting
// ----------------------------------------------------------------------------

#[test]
fn test_rt_tier_display_none() {
    assert_eq!(format!("{}", VulkanRayTracingTier::None), "None");
}

#[test]
fn test_rt_tier_display_query() {
    assert_eq!(format!("{}", VulkanRayTracingTier::Query), "Ray Query");
}

#[test]
fn test_rt_tier_display_full() {
    assert_eq!(format!("{}", VulkanRayTracingTier::Full), "Full Pipeline");
}

// ----------------------------------------------------------------------------
// 3.6 Clone/Copy/Hash
// ----------------------------------------------------------------------------

#[test]
fn test_rt_tier_clone() {
    let tier1 = VulkanRayTracingTier::Full;
    let tier2 = tier1.clone();
    assert_eq!(tier1, tier2);
}

#[test]
fn test_rt_tier_copy() {
    let tier1 = VulkanRayTracingTier::Query;
    let tier2 = tier1;
    assert_eq!(tier1, tier2);
}

#[test]
fn test_rt_tier_hash() {
    use std::collections::HashSet;
    let mut set = HashSet::new();
    set.insert(VulkanRayTracingTier::None);
    set.insert(VulkanRayTracingTier::Query);
    set.insert(VulkanRayTracingTier::Full);
    assert_eq!(set.len(), 3);
}

// ============================================================================
// Section 4: VulkanFeatures Tests
// ============================================================================

// ----------------------------------------------------------------------------
// 4.1 Default State
// ----------------------------------------------------------------------------

#[test]
fn test_features_default_ray_tracing_false() {
    let f = VulkanFeatures::default();
    assert!(!f.ray_tracing);
}

#[test]
fn test_features_default_ray_query_false() {
    let f = VulkanFeatures::default();
    assert!(!f.ray_query);
}

#[test]
fn test_features_default_descriptor_indexing_false() {
    let f = VulkanFeatures::default();
    assert!(!f.descriptor_indexing);
}

#[test]
fn test_features_default_timeline_semaphores_false() {
    let f = VulkanFeatures::default();
    assert!(!f.timeline_semaphores);
}

#[test]
fn test_features_default_buffer_device_address_false() {
    let f = VulkanFeatures::default();
    assert!(!f.buffer_device_address);
}

#[test]
fn test_features_default_mesh_shading_false() {
    let f = VulkanFeatures::default();
    assert!(!f.mesh_shading);
}

#[test]
fn test_features_default_dynamic_rendering_false() {
    let f = VulkanFeatures::default();
    assert!(!f.dynamic_rendering);
}

#[test]
fn test_features_default_synchronization2_false() {
    let f = VulkanFeatures::default();
    assert!(!f.synchronization2);
}

#[test]
fn test_features_default_extended_dynamic_state_false() {
    let f = VulkanFeatures::default();
    assert!(!f.extended_dynamic_state);
}

#[test]
fn test_features_default_maintenance4_false() {
    let f = VulkanFeatures::default();
    assert!(!f.maintenance4);
}

// ----------------------------------------------------------------------------
// 4.2 supports_rt_pipeline() Combinations
// ----------------------------------------------------------------------------

#[test]
fn test_features_supports_rt_pipeline_both_false() {
    let f = VulkanFeatures::default();
    assert!(!f.supports_rt_pipeline());
}

#[test]
fn test_features_supports_rt_pipeline_only_ray_tracing() {
    let mut f = VulkanFeatures::default();
    f.ray_tracing = true;
    assert!(!f.supports_rt_pipeline()); // Missing BDA
}

#[test]
fn test_features_supports_rt_pipeline_only_bda() {
    let mut f = VulkanFeatures::default();
    f.buffer_device_address = true;
    assert!(!f.supports_rt_pipeline()); // Missing RT
}

#[test]
fn test_features_supports_rt_pipeline_both_true() {
    let mut f = VulkanFeatures::default();
    f.ray_tracing = true;
    f.buffer_device_address = true;
    assert!(f.supports_rt_pipeline());
}

// ----------------------------------------------------------------------------
// 4.3 supports_ray_query() Tests
// ----------------------------------------------------------------------------

#[test]
fn test_features_supports_ray_query_false() {
    let f = VulkanFeatures::default();
    assert!(!f.supports_ray_query());
}

#[test]
fn test_features_supports_ray_query_true() {
    let mut f = VulkanFeatures::default();
    f.ray_query = true;
    assert!(f.supports_ray_query());
}

// ----------------------------------------------------------------------------
// 4.4 supports_any_rt() Combinations
// ----------------------------------------------------------------------------

#[test]
fn test_features_supports_any_rt_both_false() {
    let f = VulkanFeatures::default();
    assert!(!f.supports_any_rt());
}

#[test]
fn test_features_supports_any_rt_only_ray_tracing() {
    let mut f = VulkanFeatures::default();
    f.ray_tracing = true;
    assert!(f.supports_any_rt());
}

#[test]
fn test_features_supports_any_rt_only_ray_query() {
    let mut f = VulkanFeatures::default();
    f.ray_query = true;
    assert!(f.supports_any_rt());
}

#[test]
fn test_features_supports_any_rt_both_true() {
    let mut f = VulkanFeatures::default();
    f.ray_tracing = true;
    f.ray_query = true;
    assert!(f.supports_any_rt());
}

// ----------------------------------------------------------------------------
// 4.5 supports_bindless() Combinations
// ----------------------------------------------------------------------------

#[test]
fn test_features_supports_bindless_both_false() {
    let f = VulkanFeatures::default();
    assert!(!f.supports_bindless());
}

#[test]
fn test_features_supports_bindless_only_descriptor_indexing() {
    let mut f = VulkanFeatures::default();
    f.descriptor_indexing = true;
    assert!(!f.supports_bindless()); // Missing BDA
}

#[test]
fn test_features_supports_bindless_only_bda() {
    let mut f = VulkanFeatures::default();
    f.buffer_device_address = true;
    assert!(!f.supports_bindless()); // Missing DI
}

#[test]
fn test_features_supports_bindless_both_true() {
    let mut f = VulkanFeatures::default();
    f.descriptor_indexing = true;
    f.buffer_device_address = true;
    assert!(f.supports_bindless());
}

// ----------------------------------------------------------------------------
// 4.6 supports_vulkan_1_2() Combinations
// ----------------------------------------------------------------------------

#[test]
fn test_features_supports_vulkan_1_2_both_false() {
    let f = VulkanFeatures::default();
    assert!(!f.supports_vulkan_1_2());
}

#[test]
fn test_features_supports_vulkan_1_2_only_timeline() {
    let mut f = VulkanFeatures::default();
    f.timeline_semaphores = true;
    assert!(!f.supports_vulkan_1_2());
}

#[test]
fn test_features_supports_vulkan_1_2_only_bda() {
    let mut f = VulkanFeatures::default();
    f.buffer_device_address = true;
    assert!(!f.supports_vulkan_1_2());
}

#[test]
fn test_features_supports_vulkan_1_2_both_true() {
    let mut f = VulkanFeatures::default();
    f.timeline_semaphores = true;
    f.buffer_device_address = true;
    assert!(f.supports_vulkan_1_2());
}

// ----------------------------------------------------------------------------
// 4.7 supports_vulkan_1_3() Combinations
// ----------------------------------------------------------------------------

#[test]
fn test_features_supports_vulkan_1_3_both_false() {
    let f = VulkanFeatures::default();
    assert!(!f.supports_vulkan_1_3());
}

#[test]
fn test_features_supports_vulkan_1_3_only_dynamic_rendering() {
    let mut f = VulkanFeatures::default();
    f.dynamic_rendering = true;
    assert!(!f.supports_vulkan_1_3());
}

#[test]
fn test_features_supports_vulkan_1_3_only_sync2() {
    let mut f = VulkanFeatures::default();
    f.synchronization2 = true;
    assert!(!f.supports_vulkan_1_3());
}

#[test]
fn test_features_supports_vulkan_1_3_both_true() {
    let mut f = VulkanFeatures::default();
    f.dynamic_rendering = true;
    f.synchronization2 = true;
    assert!(f.supports_vulkan_1_3());
}

// ----------------------------------------------------------------------------
// 4.8 supports_mesh_shading() Tests
// ----------------------------------------------------------------------------

#[test]
fn test_features_supports_mesh_shading_false() {
    let f = VulkanFeatures::default();
    assert!(!f.supports_mesh_shading());
}

#[test]
fn test_features_supports_mesh_shading_true() {
    let mut f = VulkanFeatures::default();
    f.mesh_shading = true;
    assert!(f.supports_mesh_shading());
}

// ----------------------------------------------------------------------------
// 4.9 supports_modern_sync() Combinations
// ----------------------------------------------------------------------------

#[test]
fn test_features_supports_modern_sync_both_false() {
    let f = VulkanFeatures::default();
    assert!(!f.supports_modern_sync());
}

#[test]
fn test_features_supports_modern_sync_only_timeline() {
    let mut f = VulkanFeatures::default();
    f.timeline_semaphores = true;
    assert!(!f.supports_modern_sync());
}

#[test]
fn test_features_supports_modern_sync_only_sync2() {
    let mut f = VulkanFeatures::default();
    f.synchronization2 = true;
    assert!(!f.supports_modern_sync());
}

#[test]
fn test_features_supports_modern_sync_both_true() {
    let mut f = VulkanFeatures::default();
    f.timeline_semaphores = true;
    f.synchronization2 = true;
    assert!(f.supports_modern_sync());
}

// ----------------------------------------------------------------------------
// 4.10 ray_tracing_tier() Tier Determination
// ----------------------------------------------------------------------------

#[test]
fn test_features_ray_tracing_tier_none() {
    let f = VulkanFeatures::default();
    assert_eq!(f.ray_tracing_tier(), VulkanRayTracingTier::None);
}

#[test]
fn test_features_ray_tracing_tier_query_only() {
    let mut f = VulkanFeatures::default();
    f.ray_query = true;
    assert_eq!(f.ray_tracing_tier(), VulkanRayTracingTier::Query);
}

#[test]
fn test_features_ray_tracing_tier_full() {
    let mut f = VulkanFeatures::default();
    f.ray_tracing = true;
    f.buffer_device_address = true;
    assert_eq!(f.ray_tracing_tier(), VulkanRayTracingTier::Full);
}

#[test]
fn test_features_ray_tracing_tier_full_with_query() {
    // When both are available, Full takes precedence
    let mut f = VulkanFeatures::default();
    f.ray_tracing = true;
    f.ray_query = true;
    f.buffer_device_address = true;
    assert_eq!(f.ray_tracing_tier(), VulkanRayTracingTier::Full);
}

#[test]
fn test_features_ray_tracing_tier_rt_without_bda_is_query() {
    // RT without BDA falls back to Query tier if ray_query is set
    let mut f = VulkanFeatures::default();
    f.ray_tracing = true;
    f.ray_query = true;
    // No buffer_device_address
    assert_eq!(f.ray_tracing_tier(), VulkanRayTracingTier::Query);
}

#[test]
fn test_features_ray_tracing_tier_rt_without_bda_no_query_is_none() {
    // RT without BDA and no ray_query = None
    let mut f = VulkanFeatures::default();
    f.ray_tracing = true;
    // No buffer_device_address, no ray_query
    assert_eq!(f.ray_tracing_tier(), VulkanRayTracingTier::None);
}

// ----------------------------------------------------------------------------
// 4.11 summary() Output Variations
// ----------------------------------------------------------------------------

#[test]
fn test_features_summary_empty() {
    let f = VulkanFeatures::default();
    assert_eq!(f.summary(), "None");
}

#[test]
fn test_features_summary_single_feature() {
    let mut f = VulkanFeatures::default();
    f.ray_tracing = true;
    assert_eq!(f.summary(), "RT-Pipeline");
}

#[test]
fn test_features_summary_multiple_features() {
    let mut f = VulkanFeatures::default();
    f.ray_tracing = true;
    f.ray_query = true;
    let summary = f.summary();
    assert!(summary.contains("RT-Pipeline"));
    assert!(summary.contains("RT-Query"));
}

#[test]
fn test_features_summary_descriptor_indexing() {
    let mut f = VulkanFeatures::default();
    f.descriptor_indexing = true;
    assert!(f.summary().contains("Descriptor-Indexing"));
}

#[test]
fn test_features_summary_timeline_semaphores() {
    let mut f = VulkanFeatures::default();
    f.timeline_semaphores = true;
    assert!(f.summary().contains("Timeline-Semaphores"));
}

#[test]
fn test_features_summary_bda() {
    let mut f = VulkanFeatures::default();
    f.buffer_device_address = true;
    assert!(f.summary().contains("BDA"));
}

#[test]
fn test_features_summary_mesh_shading() {
    let mut f = VulkanFeatures::default();
    f.mesh_shading = true;
    assert!(f.summary().contains("Mesh-Shaders"));
}

#[test]
fn test_features_summary_dynamic_rendering() {
    let mut f = VulkanFeatures::default();
    f.dynamic_rendering = true;
    assert!(f.summary().contains("Dynamic-Rendering"));
}

#[test]
fn test_features_summary_all_features() {
    let f = VulkanFeatures {
        ray_tracing: true,
        ray_query: true,
        descriptor_indexing: true,
        timeline_semaphores: true,
        buffer_device_address: true,
        mesh_shading: true,
        dynamic_rendering: true,
        synchronization2: false,
        extended_dynamic_state: false,
        maintenance4: false,
    };
    let summary = f.summary();
    assert!(summary.contains("RT-Pipeline"));
    assert!(summary.contains("RT-Query"));
    assert!(summary.contains("Descriptor-Indexing"));
    assert!(summary.contains("Timeline-Semaphores"));
    assert!(summary.contains("BDA"));
    assert!(summary.contains("Mesh-Shaders"));
    assert!(summary.contains("Dynamic-Rendering"));
}

// ----------------------------------------------------------------------------
// 4.12 from_features() wgpu Feature Mapping
// ----------------------------------------------------------------------------

#[test]
fn test_features_from_empty_wgpu_features() {
    let wgpu_features = Features::empty();
    let vk = VulkanFeatures::from_features(wgpu_features);
    assert!(!vk.ray_tracing);
    assert!(!vk.ray_query);
    assert!(!vk.descriptor_indexing);
    assert!(!vk.buffer_device_address);
}

#[test]
fn test_features_from_rt_acceleration_structure() {
    let wgpu_features = Features::RAY_TRACING_ACCELERATION_STRUCTURE;
    let vk = VulkanFeatures::from_features(wgpu_features);
    assert!(vk.ray_tracing);
    assert!(vk.buffer_device_address); // Implied by RT
    assert!(vk.timeline_semaphores); // Implied by RT
}

#[test]
fn test_features_from_ray_query() {
    let wgpu_features = Features::RAY_QUERY;
    let vk = VulkanFeatures::from_features(wgpu_features);
    assert!(vk.ray_query);
    assert!(vk.buffer_device_address); // Implied by ray query
}

#[test]
fn test_features_from_rt_and_ray_query() {
    let wgpu_features = Features::RAY_TRACING_ACCELERATION_STRUCTURE | Features::RAY_QUERY;
    let vk = VulkanFeatures::from_features(wgpu_features);
    assert!(vk.ray_tracing);
    assert!(vk.ray_query);
    assert!(vk.buffer_device_address);
}

#[test]
fn test_features_from_descriptor_indexing_partial() {
    // Missing one required feature
    let wgpu_features = Features::TEXTURE_BINDING_ARRAY | Features::BUFFER_BINDING_ARRAY;
    let vk = VulkanFeatures::from_features(wgpu_features);
    assert!(!vk.descriptor_indexing);
}

#[test]
fn test_features_from_descriptor_indexing_full() {
    let wgpu_features = Features::TEXTURE_BINDING_ARRAY
        | Features::BUFFER_BINDING_ARRAY
        | Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING;
    let vk = VulkanFeatures::from_features(wgpu_features);
    assert!(vk.descriptor_indexing);
}

#[test]
fn test_features_from_texture_binding_array_enables_eds() {
    let wgpu_features = Features::TEXTURE_BINDING_ARRAY;
    let vk = VulkanFeatures::from_features(wgpu_features);
    assert!(vk.extended_dynamic_state);
}

#[test]
fn test_features_from_rt_enables_maintenance4() {
    let wgpu_features = Features::RAY_TRACING_ACCELERATION_STRUCTURE;
    let vk = VulkanFeatures::from_features(wgpu_features);
    assert!(vk.maintenance4);
}

#[test]
fn test_features_mesh_shading_always_false() {
    // wgpu doesn't support mesh shaders yet
    let wgpu_features = Features::all();
    let vk = VulkanFeatures::from_features(wgpu_features);
    assert!(!vk.mesh_shading);
}

// ----------------------------------------------------------------------------
// 4.13 Extension List Verification
// ----------------------------------------------------------------------------

#[test]
fn test_features_required_instance_extensions_not_empty() {
    let extensions = VulkanFeatures::required_instance_extensions();
    assert!(!extensions.is_empty());
}

#[test]
fn test_features_required_instance_extensions_has_properties2() {
    let extensions = VulkanFeatures::required_instance_extensions();
    assert!(extensions.contains(&"VK_KHR_get_physical_device_properties2"));
}

#[test]
fn test_features_required_instance_extensions_has_debug_utils() {
    let extensions = VulkanFeatures::required_instance_extensions();
    assert!(extensions.contains(&"VK_EXT_debug_utils"));
}

#[test]
fn test_features_required_instance_extensions_has_external_memory() {
    let extensions = VulkanFeatures::required_instance_extensions();
    assert!(extensions.contains(&"VK_KHR_external_memory_capabilities"));
}

#[test]
fn test_features_ray_tracing_extensions_not_empty() {
    let extensions = VulkanFeatures::ray_tracing_extensions();
    assert!(!extensions.is_empty());
    assert!(extensions.len() >= 5);
}

#[test]
fn test_features_ray_tracing_extensions_has_rt_pipeline() {
    let extensions = VulkanFeatures::ray_tracing_extensions();
    assert!(extensions.contains(&"VK_KHR_ray_tracing_pipeline"));
}

#[test]
fn test_features_ray_tracing_extensions_has_acceleration_structure() {
    let extensions = VulkanFeatures::ray_tracing_extensions();
    assert!(extensions.contains(&"VK_KHR_acceleration_structure"));
}

#[test]
fn test_features_ray_tracing_extensions_has_ray_query() {
    let extensions = VulkanFeatures::ray_tracing_extensions();
    assert!(extensions.contains(&"VK_KHR_ray_query"));
}

#[test]
fn test_features_ray_tracing_extensions_has_bda() {
    let extensions = VulkanFeatures::ray_tracing_extensions();
    assert!(extensions.contains(&"VK_KHR_buffer_device_address"));
}

#[test]
fn test_features_mesh_shader_extensions_not_empty() {
    let extensions = VulkanFeatures::mesh_shader_extensions();
    assert!(!extensions.is_empty());
}

#[test]
fn test_features_mesh_shader_extensions_has_mesh_shader() {
    let extensions = VulkanFeatures::mesh_shader_extensions();
    assert!(extensions.contains(&"VK_EXT_mesh_shader"));
}

#[test]
fn test_features_bindless_extensions_not_empty() {
    let extensions = VulkanFeatures::bindless_extensions();
    assert!(!extensions.is_empty());
}

#[test]
fn test_features_bindless_extensions_has_descriptor_indexing() {
    let extensions = VulkanFeatures::bindless_extensions();
    assert!(extensions.contains(&"VK_EXT_descriptor_indexing"));
}

#[test]
fn test_features_bindless_extensions_has_bda() {
    let extensions = VulkanFeatures::bindless_extensions();
    assert!(extensions.contains(&"VK_KHR_buffer_device_address"));
}

#[test]
fn test_features_bindless_extensions_has_maintenance3() {
    let extensions = VulkanFeatures::bindless_extensions();
    assert!(extensions.contains(&"VK_KHR_maintenance3"));
}

// ----------------------------------------------------------------------------
// 4.14 Clone/Copy/PartialEq
// ----------------------------------------------------------------------------

#[test]
fn test_features_clone() {
    let f1 = VulkanFeatures {
        ray_tracing: true,
        ray_query: false,
        descriptor_indexing: true,
        ..Default::default()
    };
    let f2 = f1.clone();
    assert_eq!(f1, f2);
}

#[test]
fn test_features_copy() {
    let f1 = VulkanFeatures {
        ray_tracing: true,
        ..Default::default()
    };
    let f2 = f1; // Copy
    assert_eq!(f1, f2);
}

#[test]
fn test_features_partial_eq_same() {
    let f1 = VulkanFeatures::default();
    let f2 = VulkanFeatures::default();
    assert_eq!(f1, f2);
}

#[test]
fn test_features_partial_eq_different() {
    let f1 = VulkanFeatures::default();
    let mut f2 = VulkanFeatures::default();
    f2.ray_tracing = true;
    assert_ne!(f1, f2);
}

// ============================================================================
// Section 5: VulkanInfo Tests
// ============================================================================

// ----------------------------------------------------------------------------
// 5.1 Construction with All Fields
// ----------------------------------------------------------------------------

#[test]
fn test_info_construction_full() {
    let info = VulkanInfo {
        version: VulkanVersion::V1_3,
        features: VulkanFeatures::default(),
        driver_name: "NVIDIA".to_string(),
        driver_version: 536870912,
        device_name: "GeForce RTX 4090".to_string(),
        device_type: VulkanDeviceType::DiscreteGpu,
        vendor_id: 0x10DE,
        device_id: 0x2684,
    };

    assert_eq!(info.version, VulkanVersion::V1_3);
    assert_eq!(info.driver_name, "NVIDIA");
    assert_eq!(info.driver_version, 536870912);
    assert_eq!(info.device_name, "GeForce RTX 4090");
    assert_eq!(info.device_type, VulkanDeviceType::DiscreteGpu);
    assert_eq!(info.vendor_id, 0x10DE);
    assert_eq!(info.device_id, 0x2684);
}

// ----------------------------------------------------------------------------
// 5.2 Default Values
// ----------------------------------------------------------------------------

#[test]
fn test_info_default_version() {
    let info = VulkanInfo::default();
    assert_eq!(info.version, VulkanVersion::V1_0);
}

#[test]
fn test_info_default_driver_name_empty() {
    let info = VulkanInfo::default();
    assert!(info.driver_name.is_empty());
}

#[test]
fn test_info_default_device_name_empty() {
    let info = VulkanInfo::default();
    assert!(info.device_name.is_empty());
}

#[test]
fn test_info_default_device_type_other() {
    let info = VulkanInfo::default();
    assert_eq!(info.device_type, VulkanDeviceType::Other);
}

#[test]
fn test_info_default_vendor_id_zero() {
    let info = VulkanInfo::default();
    assert_eq!(info.vendor_id, 0);
}

#[test]
fn test_info_default_device_id_zero() {
    let info = VulkanInfo::default();
    assert_eq!(info.device_id, 0);
}

// ----------------------------------------------------------------------------
// 5.3 summary() Output
// ----------------------------------------------------------------------------

#[test]
fn test_info_summary_contains_device_name() {
    let info = VulkanInfo {
        device_name: "RTX 4090".to_string(),
        ..Default::default()
    };
    assert!(info.summary().contains("RTX 4090"));
}

#[test]
fn test_info_summary_contains_device_type() {
    let info = VulkanInfo {
        device_type: VulkanDeviceType::DiscreteGpu,
        ..Default::default()
    };
    assert!(info.summary().contains("Discrete GPU"));
}

#[test]
fn test_info_summary_contains_driver_name() {
    let info = VulkanInfo {
        driver_name: "Mesa".to_string(),
        ..Default::default()
    };
    assert!(info.summary().contains("Mesa"));
}

#[test]
fn test_info_summary_contains_version() {
    let info = VulkanInfo {
        version: VulkanVersion::V1_2,
        ..Default::default()
    };
    assert!(info.summary().contains("1.2.0"));
}

#[test]
fn test_info_summary_full() {
    let info = VulkanInfo {
        version: VulkanVersion::V1_3,
        device_name: "RTX 4090".to_string(),
        device_type: VulkanDeviceType::DiscreteGpu,
        driver_name: "NVIDIA".to_string(),
        ..Default::default()
    };
    let summary = info.summary();
    assert!(summary.contains("RTX 4090"));
    assert!(summary.contains("Discrete GPU"));
    assert!(summary.contains("NVIDIA"));
    assert!(summary.contains("1.3.0"));
}

// ----------------------------------------------------------------------------
// 5.4 is_suitable() Requirements
// ----------------------------------------------------------------------------

#[test]
fn test_info_is_suitable_default_false() {
    let info = VulkanInfo::default();
    assert!(!info.is_suitable());
}

#[test]
fn test_info_is_suitable_discrete_gpu_no_features() {
    let info = VulkanInfo {
        device_type: VulkanDeviceType::DiscreteGpu,
        ..Default::default()
    };
    assert!(!info.is_suitable()); // Missing Vulkan 1.2 features
}

#[test]
fn test_info_is_suitable_features_but_cpu() {
    let mut info = VulkanInfo::default();
    info.device_type = VulkanDeviceType::Cpu;
    info.features.timeline_semaphores = true;
    info.features.buffer_device_address = true;
    assert!(!info.is_suitable()); // CPU is not hardware
}

#[test]
fn test_info_is_suitable_features_but_other() {
    let mut info = VulkanInfo::default();
    info.device_type = VulkanDeviceType::Other;
    info.features.timeline_semaphores = true;
    info.features.buffer_device_address = true;
    assert!(!info.is_suitable()); // Other is not hardware
}

#[test]
fn test_info_is_suitable_discrete_gpu_with_1_2() {
    let mut info = VulkanInfo::default();
    info.device_type = VulkanDeviceType::DiscreteGpu;
    info.features.timeline_semaphores = true;
    info.features.buffer_device_address = true;
    assert!(info.is_suitable());
}

#[test]
fn test_info_is_suitable_integrated_gpu_with_1_2() {
    let mut info = VulkanInfo::default();
    info.device_type = VulkanDeviceType::IntegratedGpu;
    info.features.timeline_semaphores = true;
    info.features.buffer_device_address = true;
    assert!(info.is_suitable());
}

#[test]
fn test_info_is_suitable_virtual_gpu_with_1_2() {
    let mut info = VulkanInfo::default();
    info.device_type = VulkanDeviceType::VirtualGpu;
    info.features.timeline_semaphores = true;
    info.features.buffer_device_address = true;
    assert!(info.is_suitable());
}

// ----------------------------------------------------------------------------
// 5.5 supports_ray_tracing() Delegation
// ----------------------------------------------------------------------------

#[test]
fn test_info_supports_ray_tracing_false() {
    let info = VulkanInfo::default();
    assert!(!info.supports_ray_tracing());
}

#[test]
fn test_info_supports_ray_tracing_with_rt() {
    let mut info = VulkanInfo::default();
    info.features.ray_tracing = true;
    assert!(info.supports_ray_tracing());
}

#[test]
fn test_info_supports_ray_tracing_with_query() {
    let mut info = VulkanInfo::default();
    info.features.ray_query = true;
    assert!(info.supports_ray_tracing());
}

// ----------------------------------------------------------------------------
// 5.6 Clone
// ----------------------------------------------------------------------------

#[test]
fn test_info_clone() {
    let info1 = VulkanInfo {
        version: VulkanVersion::V1_3,
        driver_name: "NVIDIA".to_string(),
        device_name: "RTX 4090".to_string(),
        device_type: VulkanDeviceType::DiscreteGpu,
        ..Default::default()
    };
    let info2 = info1.clone();
    assert_eq!(info1.version, info2.version);
    assert_eq!(info1.driver_name, info2.driver_name);
    assert_eq!(info1.device_name, info2.device_name);
    assert_eq!(info1.device_type, info2.device_type);
}

// ============================================================================
// Section 6: Edge Cases & Boundary Tests
// ============================================================================

// ----------------------------------------------------------------------------
// 6.1 Maximum Version Components
// ----------------------------------------------------------------------------

#[test]
fn test_edge_version_max_components() {
    let v = VulkanVersion::new(0x3FF, 0x3FF, 0xFFF);
    assert_eq!(v.major, 0x3FF);
    assert_eq!(v.minor, 0x3FF);
    assert_eq!(v.patch, 0xFFF);
}

#[test]
fn test_edge_version_max_round_trip() {
    let v = VulkanVersion::new(0x3FF, 0x3FF, 0xFFF);
    let decoded = VulkanVersion::from_raw(v.to_raw());
    assert_eq!(v, decoded);
}

#[test]
fn test_edge_version_max_display() {
    let v = VulkanVersion::new(1023, 1023, 4095);
    assert_eq!(format!("{}", v), "1023.1023.4095");
}

// ----------------------------------------------------------------------------
// 6.2 Zero Values Throughout
// ----------------------------------------------------------------------------

#[test]
fn test_edge_version_zero() {
    let v = VulkanVersion::new(0, 0, 0);
    assert_eq!(v.to_raw(), 0);
    assert_eq!(format!("{}", v), "0.0.0");
}

#[test]
fn test_edge_info_all_zeros() {
    let info = VulkanInfo {
        version: VulkanVersion::new(0, 0, 0),
        features: VulkanFeatures::default(),
        driver_name: String::new(),
        driver_version: 0,
        device_name: String::new(),
        device_type: VulkanDeviceType::Other,
        vendor_id: 0,
        device_id: 0,
    };
    assert_eq!(info.vendor_id, 0);
    assert_eq!(info.device_id, 0);
    assert!(!info.is_suitable());
}

// ----------------------------------------------------------------------------
// 6.3 Empty Strings
// ----------------------------------------------------------------------------

#[test]
fn test_edge_info_empty_driver_name() {
    let info = VulkanInfo {
        driver_name: String::new(),
        ..Default::default()
    };
    let summary = info.summary();
    assert!(summary.contains("Vulkan")); // Still has version
}

#[test]
fn test_edge_info_empty_device_name() {
    let info = VulkanInfo {
        device_name: String::new(),
        ..Default::default()
    };
    let summary = info.summary();
    // Summary should still be valid
    assert!(!summary.is_empty());
}

// ----------------------------------------------------------------------------
// 6.4 All Features Enabled/Disabled
// ----------------------------------------------------------------------------

#[test]
fn test_edge_features_all_enabled() {
    let f = VulkanFeatures {
        ray_tracing: true,
        ray_query: true,
        descriptor_indexing: true,
        timeline_semaphores: true,
        buffer_device_address: true,
        mesh_shading: true,
        dynamic_rendering: true,
        synchronization2: true,
        extended_dynamic_state: true,
        maintenance4: true,
    };

    assert!(f.supports_rt_pipeline());
    assert!(f.supports_ray_query());
    assert!(f.supports_any_rt());
    assert!(f.supports_bindless());
    assert!(f.supports_vulkan_1_2());
    assert!(f.supports_vulkan_1_3());
    assert!(f.supports_mesh_shading());
    assert!(f.supports_modern_sync());
    assert_eq!(f.ray_tracing_tier(), VulkanRayTracingTier::Full);
}

#[test]
fn test_edge_features_all_disabled() {
    let f = VulkanFeatures::default();

    assert!(!f.supports_rt_pipeline());
    assert!(!f.supports_ray_query());
    assert!(!f.supports_any_rt());
    assert!(!f.supports_bindless());
    assert!(!f.supports_vulkan_1_2());
    assert!(!f.supports_vulkan_1_3());
    assert!(!f.supports_mesh_shading());
    assert!(!f.supports_modern_sync());
    assert_eq!(f.ray_tracing_tier(), VulkanRayTracingTier::None);
    assert_eq!(f.summary(), "None");
}

// ----------------------------------------------------------------------------
// 6.5 Mixed Feature States
// ----------------------------------------------------------------------------

#[test]
fn test_edge_features_minimal_rt() {
    // Minimal for RT pipeline: ray_tracing + buffer_device_address
    let mut f = VulkanFeatures::default();
    f.ray_tracing = true;
    f.buffer_device_address = true;

    assert!(f.supports_rt_pipeline());
    assert!(!f.supports_ray_query());
    assert!(f.supports_any_rt());
    assert!(!f.supports_bindless()); // Missing descriptor_indexing
}

#[test]
fn test_edge_features_minimal_bindless() {
    // Minimal for bindless: descriptor_indexing + buffer_device_address
    let mut f = VulkanFeatures::default();
    f.descriptor_indexing = true;
    f.buffer_device_address = true;

    assert!(f.supports_bindless());
    assert!(!f.supports_rt_pipeline()); // Missing ray_tracing
}

#[test]
fn test_edge_features_minimal_1_2() {
    // Minimal for Vulkan 1.2: timeline_semaphores + buffer_device_address
    let mut f = VulkanFeatures::default();
    f.timeline_semaphores = true;
    f.buffer_device_address = true;

    assert!(f.supports_vulkan_1_2());
    assert!(!f.supports_vulkan_1_3()); // Missing dynamic_rendering/sync2
}

#[test]
fn test_edge_features_minimal_1_3() {
    // Minimal for Vulkan 1.3: dynamic_rendering + synchronization2
    let mut f = VulkanFeatures::default();
    f.dynamic_rendering = true;
    f.synchronization2 = true;

    assert!(f.supports_vulkan_1_3());
    assert!(!f.supports_vulkan_1_2()); // Missing timeline/bda
}

// ----------------------------------------------------------------------------
// 6.6 Version Boundary Edge Cases
// ----------------------------------------------------------------------------

#[test]
fn test_edge_version_ordering_boundary() {
    // Version 1.2.4095 vs 1.3.0
    let v1 = VulkanVersion::new(1, 2, 4095);
    let v2 = VulkanVersion::new(1, 3, 0);
    assert!(v1 < v2);
}

#[test]
fn test_edge_version_is_at_least_boundary_minor() {
    let v = VulkanVersion::new(1, 2, 0);
    assert!(v.is_at_least(1, 2));
    assert!(!v.is_at_least(1, 3));
}

#[test]
fn test_edge_version_is_at_least_boundary_major() {
    let v = VulkanVersion::new(1, 999, 0);
    assert!(v.is_at_least(1, 999));
    assert!(!v.is_at_least(2, 0));
}

// ----------------------------------------------------------------------------
// 6.7 Debug Formatting
// ----------------------------------------------------------------------------

#[test]
fn test_edge_version_debug_format() {
    let v = VulkanVersion::V1_2;
    let debug = format!("{:?}", v);
    assert!(debug.contains("VulkanVersion"));
    assert!(debug.contains("major"));
    assert!(debug.contains("1"));
}

#[test]
fn test_edge_device_type_debug_format() {
    let dt = VulkanDeviceType::DiscreteGpu;
    let debug = format!("{:?}", dt);
    assert!(debug.contains("DiscreteGpu"));
}

#[test]
fn test_edge_features_debug_format() {
    let f = VulkanFeatures::default();
    let debug = format!("{:?}", f);
    assert!(debug.contains("VulkanFeatures"));
    assert!(debug.contains("ray_tracing"));
}

#[test]
fn test_edge_info_debug_format() {
    let info = VulkanInfo::default();
    let debug = format!("{:?}", info);
    assert!(debug.contains("VulkanInfo"));
    assert!(debug.contains("version"));
}

#[test]
fn test_edge_rt_tier_debug_format() {
    let tier = VulkanRayTracingTier::Full;
    let debug = format!("{:?}", tier);
    assert!(debug.contains("Full"));
}

// ----------------------------------------------------------------------------
// 6.8 Extension List Immutability
// ----------------------------------------------------------------------------

#[test]
fn test_edge_required_instance_extensions_consistent() {
    let e1 = VulkanFeatures::required_instance_extensions();
    let e2 = VulkanFeatures::required_instance_extensions();
    assert_eq!(e1.len(), e2.len());
    for (a, b) in e1.iter().zip(e2.iter()) {
        assert_eq!(a, b);
    }
}

#[test]
fn test_edge_ray_tracing_extensions_consistent() {
    let e1 = VulkanFeatures::ray_tracing_extensions();
    let e2 = VulkanFeatures::ray_tracing_extensions();
    assert_eq!(e1.len(), e2.len());
}

#[test]
fn test_edge_mesh_shader_extensions_consistent() {
    let e1 = VulkanFeatures::mesh_shader_extensions();
    let e2 = VulkanFeatures::mesh_shader_extensions();
    assert_eq!(e1.len(), e2.len());
}

#[test]
fn test_edge_bindless_extensions_consistent() {
    let e1 = VulkanFeatures::bindless_extensions();
    let e2 = VulkanFeatures::bindless_extensions();
    assert_eq!(e1.len(), e2.len());
}

// ----------------------------------------------------------------------------
// 6.9 Large Version Numbers
// ----------------------------------------------------------------------------

#[test]
fn test_edge_version_large_major() {
    let v = VulkanVersion::new(999, 0, 0);
    assert_eq!(v.major, 999);
    let decoded = VulkanVersion::from_raw(v.to_raw());
    assert_eq!(decoded.major, 999);
}

#[test]
fn test_edge_version_large_patch() {
    let v = VulkanVersion::new(1, 3, 4000);
    assert_eq!(v.patch, 4000);
    let decoded = VulkanVersion::from_raw(v.to_raw());
    assert_eq!(decoded.patch, 4000);
}

// ============================================================================
// Section 7: Additional Combinatorial Tests
// ============================================================================

#[test]
fn test_combo_rt_tiers_exhaustive() {
    // Test all RT tier combinations
    let combinations = [
        (false, false, false, VulkanRayTracingTier::None),
        (false, true, false, VulkanRayTracingTier::Query),
        (true, false, false, VulkanRayTracingTier::None), // RT without BDA
        (true, false, true, VulkanRayTracingTier::Full),
        (true, true, true, VulkanRayTracingTier::Full),
        (false, true, true, VulkanRayTracingTier::Query),
    ];

    for (rt, rq, bda, expected) in combinations {
        let f = VulkanFeatures {
            ray_tracing: rt,
            ray_query: rq,
            buffer_device_address: bda,
            ..Default::default()
        };
        assert_eq!(
            f.ray_tracing_tier(),
            expected,
            "Failed for rt={}, rq={}, bda={}",
            rt,
            rq,
            bda
        );
    }
}

#[test]
fn test_combo_is_suitable_matrix() {
    // Test is_suitable across device types and feature combinations
    let device_types = [
        (VulkanDeviceType::DiscreteGpu, true),
        (VulkanDeviceType::IntegratedGpu, true),
        (VulkanDeviceType::VirtualGpu, true),
        (VulkanDeviceType::Cpu, false),
        (VulkanDeviceType::Other, false),
    ];

    for (dt, is_hw) in device_types {
        let mut info = VulkanInfo::default();
        info.device_type = dt;

        // Without 1.2 features
        assert!(!info.is_suitable(), "{:?} should not be suitable without 1.2", dt);

        // With 1.2 features
        info.features.timeline_semaphores = true;
        info.features.buffer_device_address = true;
        assert_eq!(
            info.is_suitable(),
            is_hw,
            "{:?} with 1.2 features: expected {}",
            dt,
            is_hw
        );
    }
}

#[test]
fn test_combo_version_comparison_matrix() {
    let versions = [
        VulkanVersion::V1_0,
        VulkanVersion::V1_1,
        VulkanVersion::V1_2,
        VulkanVersion::V1_3,
    ];

    for (i, v1) in versions.iter().enumerate() {
        for (j, v2) in versions.iter().enumerate() {
            if i < j {
                assert!(v1 < v2, "{} should be < {}", v1, v2);
            } else if i > j {
                assert!(v1 > v2, "{} should be > {}", v1, v2);
            } else {
                assert_eq!(v1, v2, "{} should == {}", v1, v2);
            }
        }
    }
}
