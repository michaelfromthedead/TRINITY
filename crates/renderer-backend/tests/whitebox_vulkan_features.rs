// WHITEBOX tests for T-WGPU-P7.2.1 (Vulkan Features Detection)
//
// WHITEBOX discipline: These tests have FULL ACCESS to the implementation.
// They exercise internal code paths, branch conditions, and edge cases
// that are not visible through the public contract alone.
//
// Implementation under test:
//   - crates/renderer-backend/src/backend/mod.rs
//       * BackendType enum and all variants
//       * BackendType::from_wgpu_backend()
//       * BackendType::name()
//       * BackendType::is_native()
//       * BackendType::supports_ray_tracing()
//       * BackendType::supports_mesh_shaders()
//       * BackendType::supports_bindless()
//       * BackendType::to_backends()
//       * BackendType::is_available_on_platform()
//       * BackendCapabilities struct
//       * BackendCapabilities::supports_full_rt()
//       * BackendCapabilities::supports_gpu_driven()
//
//   - crates/renderer-backend/src/backend/vulkan.rs
//       * VulkanRayTracingTier enum
//       * VulkanRayTracingTier::is_available()
//       * VulkanRayTracingTier::is_full()
//       * VulkanRayTracingTier::name()
//       * VulkanFeatures struct
//       * VulkanFeatures::from_features()
//       * VulkanFeatures::supports_rt_pipeline()
//       * VulkanFeatures::supports_ray_query()
//       * VulkanFeatures::supports_any_rt()
//       * VulkanFeatures::supports_bindless()
//       * VulkanFeatures::supports_vulkan_1_2()
//       * VulkanFeatures::supports_vulkan_1_3()
//       * VulkanFeatures::ray_tracing_tier()
//       * VulkanFeatures::summary()
//
// WHITEBOX coverage plan:
//   - Section A: BackendType from_wgpu_backend() for all wgpu::Backend variants
//   - Section B: BackendType name() returns correct strings for all variants
//   - Section C: BackendType is_native() returns true only for native backends
//   - Section D: BackendType supports_ray_tracing() returns true for capable backends
//   - Section E: BackendType supports_mesh_shaders() returns true for capable backends
//   - Section F: BackendType supports_bindless() returns true for capable backends
//   - Section G: BackendType to_backends() returns correct wgpu::Backends bitmask
//   - Section H: BackendType is_available_on_platform() platform-specific checks
//   - Section I: BackendType trait implementations (Display, Default, Clone, etc.)
//   - Section J: BackendCapabilities default and field access
//   - Section K: BackendCapabilities supports_full_rt() logic
//   - Section L: BackendCapabilities supports_gpu_driven() logic
//   - Section M: VulkanRayTracingTier is_available() and is_full()
//   - Section N: VulkanRayTracingTier name() and Display
//   - Section O: VulkanFeatures from_features() with empty features
//   - Section P: VulkanFeatures from_features() with ray tracing features
//   - Section Q: VulkanFeatures from_features() with ray query only
//   - Section R: VulkanFeatures from_features() with bindless features
//   - Section S: VulkanFeatures from_features() with partial bindless (incomplete)
//   - Section T: VulkanFeatures supports_rt_pipeline() requires both RT and BDA
//   - Section U: VulkanFeatures supports_ray_query() single field check
//   - Section V: VulkanFeatures supports_any_rt() OR logic
//   - Section W: VulkanFeatures supports_bindless() requires both DI and BDA
//   - Section X: VulkanFeatures supports_vulkan_1_2() requires TS and BDA
//   - Section Y: VulkanFeatures supports_vulkan_1_3() requires DR and Sync2
//   - Section Z: VulkanFeatures ray_tracing_tier() tier classification
//   - Section AA: VulkanFeatures summary() string generation
//   - Section AB: Edge cases and boundary conditions

use renderer_backend::backend::{BackendCapabilities, BackendType, VulkanFeatures, VulkanRayTracingTier};
use wgpu::{Backend, Backends, Features};

// ============================================================================
// Section A: BackendType from_wgpu_backend() conversions
// ============================================================================

#[test]
fn test_from_wgpu_backend_vulkan() {
    let backend = BackendType::from_wgpu_backend(Backend::Vulkan);
    assert_eq!(backend, BackendType::Vulkan);
}

#[test]
fn test_from_wgpu_backend_metal() {
    let backend = BackendType::from_wgpu_backend(Backend::Metal);
    assert_eq!(backend, BackendType::Metal);
}

#[test]
fn test_from_wgpu_backend_dx12() {
    let backend = BackendType::from_wgpu_backend(Backend::Dx12);
    assert_eq!(backend, BackendType::Dx12);
}

#[test]
fn test_from_wgpu_backend_gl() {
    let backend = BackendType::from_wgpu_backend(Backend::Gl);
    assert_eq!(backend, BackendType::Gl);
}

#[test]
fn test_from_wgpu_backend_browser_webgpu() {
    let backend = BackendType::from_wgpu_backend(Backend::BrowserWebGpu);
    assert_eq!(backend, BackendType::WebGpu);
}

#[test]
fn test_from_wgpu_backend_empty() {
    let backend = BackendType::from_wgpu_backend(Backend::Empty);
    assert_eq!(backend, BackendType::Empty);
}

#[test]
fn test_from_wgpu_backend_all_variants_covered() {
    // Ensure all wgpu::Backend variants map to a BackendType
    let backends = [
        Backend::Vulkan,
        Backend::Metal,
        Backend::Dx12,
        Backend::Gl,
        Backend::BrowserWebGpu,
        Backend::Empty,
    ];

    for backend in backends {
        let result = BackendType::from_wgpu_backend(backend);
        // Should not panic and should return a valid variant
        assert_ne!(result.name(), "");
    }
}

// ============================================================================
// Section B: BackendType name() string outputs
// ============================================================================

#[test]
fn test_name_vulkan() {
    assert_eq!(BackendType::Vulkan.name(), "Vulkan");
}

#[test]
fn test_name_metal() {
    assert_eq!(BackendType::Metal.name(), "Metal");
}

#[test]
fn test_name_dx12() {
    assert_eq!(BackendType::Dx12.name(), "DirectX 12");
}

#[test]
fn test_name_dx11() {
    assert_eq!(BackendType::Dx11.name(), "DirectX 11");
}

#[test]
fn test_name_gl() {
    assert_eq!(BackendType::Gl.name(), "OpenGL");
}

#[test]
fn test_name_webgpu() {
    assert_eq!(BackendType::WebGpu.name(), "WebGPU");
}

#[test]
fn test_name_empty() {
    assert_eq!(BackendType::Empty.name(), "Empty");
}

#[test]
fn test_name_unknown() {
    assert_eq!(BackendType::Unknown.name(), "Unknown");
}

#[test]
fn test_name_all_variants_non_empty() {
    let backends = [
        BackendType::Vulkan,
        BackendType::Metal,
        BackendType::Dx12,
        BackendType::Dx11,
        BackendType::Gl,
        BackendType::WebGpu,
        BackendType::Empty,
        BackendType::Unknown,
    ];

    for backend in backends {
        let name = backend.name();
        assert!(!name.is_empty(), "Backend {:?} has empty name", backend);
        assert!(name.len() > 2, "Backend {:?} has too short name: {}", backend, name);
    }
}

// ============================================================================
// Section C: BackendType is_native() classification
// ============================================================================

#[test]
fn test_is_native_vulkan_true() {
    assert!(BackendType::Vulkan.is_native());
}

#[test]
fn test_is_native_metal_true() {
    assert!(BackendType::Metal.is_native());
}

#[test]
fn test_is_native_dx12_true() {
    assert!(BackendType::Dx12.is_native());
}

#[test]
fn test_is_native_dx11_true() {
    assert!(BackendType::Dx11.is_native());
}

#[test]
fn test_is_native_gl_false() {
    assert!(!BackendType::Gl.is_native());
}

#[test]
fn test_is_native_webgpu_false() {
    assert!(!BackendType::WebGpu.is_native());
}

#[test]
fn test_is_native_empty_false() {
    assert!(!BackendType::Empty.is_native());
}

#[test]
fn test_is_native_unknown_false() {
    assert!(!BackendType::Unknown.is_native());
}

#[test]
fn test_is_native_count_native_backends() {
    let all_backends = [
        BackendType::Vulkan,
        BackendType::Metal,
        BackendType::Dx12,
        BackendType::Dx11,
        BackendType::Gl,
        BackendType::WebGpu,
        BackendType::Empty,
        BackendType::Unknown,
    ];

    let native_count = all_backends.iter().filter(|b| b.is_native()).count();
    assert_eq!(native_count, 4, "Expected exactly 4 native backends");
}

// ============================================================================
// Section D: BackendType supports_ray_tracing()
// ============================================================================

#[test]
fn test_supports_ray_tracing_vulkan_true() {
    assert!(BackendType::Vulkan.supports_ray_tracing());
}

#[test]
fn test_supports_ray_tracing_metal_true() {
    assert!(BackendType::Metal.supports_ray_tracing());
}

#[test]
fn test_supports_ray_tracing_dx12_true() {
    assert!(BackendType::Dx12.supports_ray_tracing());
}

#[test]
fn test_supports_ray_tracing_dx11_false() {
    assert!(!BackendType::Dx11.supports_ray_tracing());
}

#[test]
fn test_supports_ray_tracing_gl_false() {
    assert!(!BackendType::Gl.supports_ray_tracing());
}

#[test]
fn test_supports_ray_tracing_webgpu_false() {
    assert!(!BackendType::WebGpu.supports_ray_tracing());
}

#[test]
fn test_supports_ray_tracing_empty_false() {
    assert!(!BackendType::Empty.supports_ray_tracing());
}

#[test]
fn test_supports_ray_tracing_unknown_false() {
    assert!(!BackendType::Unknown.supports_ray_tracing());
}

#[test]
fn test_supports_ray_tracing_count_rt_backends() {
    let all_backends = [
        BackendType::Vulkan,
        BackendType::Metal,
        BackendType::Dx12,
        BackendType::Dx11,
        BackendType::Gl,
        BackendType::WebGpu,
        BackendType::Empty,
        BackendType::Unknown,
    ];

    let rt_count = all_backends.iter().filter(|b| b.supports_ray_tracing()).count();
    assert_eq!(rt_count, 3, "Expected exactly 3 backends with RT support");
}

// ============================================================================
// Section E: BackendType supports_mesh_shaders()
// ============================================================================

#[test]
fn test_supports_mesh_shaders_vulkan_true() {
    assert!(BackendType::Vulkan.supports_mesh_shaders());
}

#[test]
fn test_supports_mesh_shaders_metal_true() {
    assert!(BackendType::Metal.supports_mesh_shaders());
}

#[test]
fn test_supports_mesh_shaders_dx12_true() {
    assert!(BackendType::Dx12.supports_mesh_shaders());
}

#[test]
fn test_supports_mesh_shaders_dx11_false() {
    assert!(!BackendType::Dx11.supports_mesh_shaders());
}

#[test]
fn test_supports_mesh_shaders_gl_false() {
    assert!(!BackendType::Gl.supports_mesh_shaders());
}

#[test]
fn test_supports_mesh_shaders_webgpu_false() {
    assert!(!BackendType::WebGpu.supports_mesh_shaders());
}

#[test]
fn test_supports_mesh_shaders_empty_false() {
    assert!(!BackendType::Empty.supports_mesh_shaders());
}

#[test]
fn test_supports_mesh_shaders_unknown_false() {
    assert!(!BackendType::Unknown.supports_mesh_shaders());
}

// ============================================================================
// Section F: BackendType supports_bindless()
// ============================================================================

#[test]
fn test_supports_bindless_vulkan_true() {
    assert!(BackendType::Vulkan.supports_bindless());
}

#[test]
fn test_supports_bindless_metal_true() {
    assert!(BackendType::Metal.supports_bindless());
}

#[test]
fn test_supports_bindless_dx12_true() {
    assert!(BackendType::Dx12.supports_bindless());
}

#[test]
fn test_supports_bindless_dx11_false() {
    assert!(!BackendType::Dx11.supports_bindless());
}

#[test]
fn test_supports_bindless_gl_false() {
    assert!(!BackendType::Gl.supports_bindless());
}

#[test]
fn test_supports_bindless_webgpu_false() {
    assert!(!BackendType::WebGpu.supports_bindless());
}

#[test]
fn test_supports_bindless_empty_false() {
    assert!(!BackendType::Empty.supports_bindless());
}

#[test]
fn test_supports_bindless_unknown_false() {
    assert!(!BackendType::Unknown.supports_bindless());
}

// ============================================================================
// Section G: BackendType to_backends() bitmask conversion
// ============================================================================

#[test]
fn test_to_backends_vulkan() {
    assert_eq!(BackendType::Vulkan.to_backends(), Backends::VULKAN);
}

#[test]
fn test_to_backends_metal() {
    assert_eq!(BackendType::Metal.to_backends(), Backends::METAL);
}

#[test]
fn test_to_backends_dx12() {
    assert_eq!(BackendType::Dx12.to_backends(), Backends::DX12);
}

#[test]
fn test_to_backends_dx11_empty() {
    // DX11 is not in wgpu::Backends, so it returns empty
    assert!(BackendType::Dx11.to_backends().is_empty());
}

#[test]
fn test_to_backends_gl() {
    assert_eq!(BackendType::Gl.to_backends(), Backends::GL);
}

#[test]
fn test_to_backends_webgpu() {
    assert_eq!(BackendType::WebGpu.to_backends(), Backends::BROWSER_WEBGPU);
}

#[test]
fn test_to_backends_empty() {
    assert!(BackendType::Empty.to_backends().is_empty());
}

#[test]
fn test_to_backends_unknown() {
    assert!(BackendType::Unknown.to_backends().is_empty());
}

#[test]
fn test_to_backends_non_empty_count() {
    let all_backends = [
        BackendType::Vulkan,
        BackendType::Metal,
        BackendType::Dx12,
        BackendType::Dx11,
        BackendType::Gl,
        BackendType::WebGpu,
        BackendType::Empty,
        BackendType::Unknown,
    ];

    let non_empty_count = all_backends.iter().filter(|b| !b.to_backends().is_empty()).count();
    assert_eq!(non_empty_count, 5, "Expected 5 backends with non-empty Backends mask");
}

// ============================================================================
// Section H: BackendType is_available_on_platform()
// ============================================================================

#[test]
#[cfg(target_os = "windows")]
fn test_is_available_on_platform_windows_vulkan() {
    assert!(BackendType::Vulkan.is_available_on_platform());
}

#[test]
#[cfg(target_os = "windows")]
fn test_is_available_on_platform_windows_dx12() {
    assert!(BackendType::Dx12.is_available_on_platform());
}

#[test]
#[cfg(target_os = "windows")]
fn test_is_available_on_platform_windows_dx11() {
    assert!(BackendType::Dx11.is_available_on_platform());
}

#[test]
#[cfg(target_os = "windows")]
fn test_is_available_on_platform_windows_metal_false() {
    assert!(!BackendType::Metal.is_available_on_platform());
}

#[test]
#[cfg(target_os = "linux")]
fn test_is_available_on_platform_linux_vulkan() {
    assert!(BackendType::Vulkan.is_available_on_platform());
}

#[test]
#[cfg(target_os = "linux")]
fn test_is_available_on_platform_linux_dx12_false() {
    assert!(!BackendType::Dx12.is_available_on_platform());
}

#[test]
#[cfg(target_os = "linux")]
fn test_is_available_on_platform_linux_metal_false() {
    assert!(!BackendType::Metal.is_available_on_platform());
}

#[test]
#[cfg(target_os = "macos")]
fn test_is_available_on_platform_macos_metal() {
    assert!(BackendType::Metal.is_available_on_platform());
}

#[test]
#[cfg(target_os = "macos")]
fn test_is_available_on_platform_macos_vulkan_false() {
    assert!(!BackendType::Vulkan.is_available_on_platform());
}

#[test]
fn test_is_available_on_platform_gl_always_true() {
    // GL is available on most platforms
    assert!(BackendType::Gl.is_available_on_platform());
}

#[test]
fn test_is_available_on_platform_empty_always_true() {
    assert!(BackendType::Empty.is_available_on_platform());
}

#[test]
fn test_is_available_on_platform_unknown_always_true() {
    assert!(BackendType::Unknown.is_available_on_platform());
}

#[test]
#[cfg(not(target_arch = "wasm32"))]
fn test_is_available_on_platform_webgpu_false_on_native() {
    assert!(!BackendType::WebGpu.is_available_on_platform());
}

// ============================================================================
// Section I: BackendType trait implementations
// ============================================================================

#[test]
fn test_backend_type_display_vulkan() {
    let display = format!("{}", BackendType::Vulkan);
    assert_eq!(display, "Vulkan");
}

#[test]
fn test_backend_type_display_dx12() {
    let display = format!("{}", BackendType::Dx12);
    assert_eq!(display, "DirectX 12");
}

#[test]
fn test_backend_type_display_dx11() {
    let display = format!("{}", BackendType::Dx11);
    assert_eq!(display, "DirectX 11");
}

#[test]
fn test_backend_type_display_metal() {
    let display = format!("{}", BackendType::Metal);
    assert_eq!(display, "Metal");
}

#[test]
fn test_backend_type_default_is_unknown() {
    assert_eq!(BackendType::default(), BackendType::Unknown);
}

#[test]
fn test_backend_type_clone() {
    let backend = BackendType::Vulkan;
    let cloned = backend.clone();
    assert_eq!(backend, cloned);
}

#[test]
fn test_backend_type_copy() {
    let backend = BackendType::Metal;
    let copied = backend;
    assert_eq!(backend, copied);
}

#[test]
fn test_backend_type_debug() {
    let debug = format!("{:?}", BackendType::Vulkan);
    assert!(debug.contains("Vulkan"));
}

#[test]
fn test_backend_type_eq() {
    assert!(BackendType::Vulkan == BackendType::Vulkan);
    assert!(BackendType::Vulkan != BackendType::Metal);
}

#[test]
fn test_backend_type_hash() {
    use std::collections::HashSet;
    let mut set = HashSet::new();
    set.insert(BackendType::Vulkan);
    set.insert(BackendType::Metal);
    set.insert(BackendType::Vulkan); // Duplicate

    assert_eq!(set.len(), 2);
    assert!(set.contains(&BackendType::Vulkan));
    assert!(set.contains(&BackendType::Metal));
}

// ============================================================================
// Section J: BackendCapabilities default and field access
// ============================================================================

#[test]
fn test_backend_capabilities_default_backend_unknown() {
    let caps = BackendCapabilities::default();
    assert_eq!(caps.backend, BackendType::Unknown);
}

#[test]
fn test_backend_capabilities_default_ray_tracing_false() {
    let caps = BackendCapabilities::default();
    assert!(!caps.ray_tracing);
}

#[test]
fn test_backend_capabilities_default_ray_query_false() {
    let caps = BackendCapabilities::default();
    assert!(!caps.ray_query);
}

#[test]
fn test_backend_capabilities_default_mesh_shaders_false() {
    let caps = BackendCapabilities::default();
    assert!(!caps.mesh_shaders);
}

#[test]
fn test_backend_capabilities_default_bindless_false() {
    let caps = BackendCapabilities::default();
    assert!(!caps.bindless);
}

#[test]
fn test_backend_capabilities_default_timeline_semaphores_false() {
    let caps = BackendCapabilities::default();
    assert!(!caps.timeline_semaphores);
}

#[test]
fn test_backend_capabilities_default_buffer_device_address_false() {
    let caps = BackendCapabilities::default();
    assert!(!caps.buffer_device_address);
}

#[test]
fn test_backend_capabilities_default_dynamic_rendering_false() {
    let caps = BackendCapabilities::default();
    assert!(!caps.dynamic_rendering);
}

#[test]
fn test_backend_capabilities_all_fields_settable() {
    let caps = BackendCapabilities {
        backend: BackendType::Vulkan,
        ray_tracing: true,
        ray_query: true,
        mesh_shaders: true,
        bindless: true,
        timeline_semaphores: true,
        buffer_device_address: true,
        dynamic_rendering: true,
    };

    assert_eq!(caps.backend, BackendType::Vulkan);
    assert!(caps.ray_tracing);
    assert!(caps.ray_query);
    assert!(caps.mesh_shaders);
    assert!(caps.bindless);
    assert!(caps.timeline_semaphores);
    assert!(caps.buffer_device_address);
    assert!(caps.dynamic_rendering);
}

// ============================================================================
// Section K: BackendCapabilities supports_full_rt()
// ============================================================================

#[test]
fn test_supports_full_rt_default_false() {
    let caps = BackendCapabilities::default();
    assert!(!caps.supports_full_rt());
}

#[test]
fn test_supports_full_rt_only_ray_tracing_false() {
    let caps = BackendCapabilities {
        ray_tracing: true,
        ray_query: false,
        ..Default::default()
    };
    assert!(!caps.supports_full_rt());
}

#[test]
fn test_supports_full_rt_only_ray_query_false() {
    let caps = BackendCapabilities {
        ray_tracing: false,
        ray_query: true,
        ..Default::default()
    };
    assert!(!caps.supports_full_rt());
}

#[test]
fn test_supports_full_rt_both_true() {
    let caps = BackendCapabilities {
        ray_tracing: true,
        ray_query: true,
        ..Default::default()
    };
    assert!(caps.supports_full_rt());
}

// ============================================================================
// Section L: BackendCapabilities supports_gpu_driven()
// ============================================================================

#[test]
fn test_supports_gpu_driven_default_false() {
    let caps = BackendCapabilities::default();
    assert!(!caps.supports_gpu_driven());
}

#[test]
fn test_supports_gpu_driven_non_vulkan_only_bindless() {
    // Non-Vulkan backends don't require buffer_device_address
    let caps = BackendCapabilities {
        backend: BackendType::Metal,
        bindless: true,
        buffer_device_address: false,
        ..Default::default()
    };
    assert!(caps.supports_gpu_driven());
}

#[test]
fn test_supports_gpu_driven_vulkan_only_bindless_false() {
    // Vulkan requires buffer_device_address for GPU-driven
    let caps = BackendCapabilities {
        backend: BackendType::Vulkan,
        bindless: true,
        buffer_device_address: false,
        ..Default::default()
    };
    assert!(!caps.supports_gpu_driven());
}

#[test]
fn test_supports_gpu_driven_vulkan_with_bda_true() {
    let caps = BackendCapabilities {
        backend: BackendType::Vulkan,
        bindless: true,
        buffer_device_address: true,
        ..Default::default()
    };
    assert!(caps.supports_gpu_driven());
}

#[test]
fn test_supports_gpu_driven_dx12_only_bindless() {
    let caps = BackendCapabilities {
        backend: BackendType::Dx12,
        bindless: true,
        buffer_device_address: false,
        ..Default::default()
    };
    assert!(caps.supports_gpu_driven());
}

#[test]
fn test_supports_gpu_driven_no_bindless_false() {
    let caps = BackendCapabilities {
        backend: BackendType::Metal,
        bindless: false,
        buffer_device_address: true,
        ..Default::default()
    };
    assert!(!caps.supports_gpu_driven());
}

// ============================================================================
// Section M: VulkanRayTracingTier is_available() and is_full()
// ============================================================================

#[test]
fn test_vulkan_rt_tier_none_not_available() {
    assert!(!VulkanRayTracingTier::None.is_available());
}

#[test]
fn test_vulkan_rt_tier_none_not_full() {
    assert!(!VulkanRayTracingTier::None.is_full());
}

#[test]
fn test_vulkan_rt_tier_query_is_available() {
    assert!(VulkanRayTracingTier::Query.is_available());
}

#[test]
fn test_vulkan_rt_tier_query_not_full() {
    assert!(!VulkanRayTracingTier::Query.is_full());
}

#[test]
fn test_vulkan_rt_tier_full_is_available() {
    assert!(VulkanRayTracingTier::Full.is_available());
}

#[test]
fn test_vulkan_rt_tier_full_is_full() {
    assert!(VulkanRayTracingTier::Full.is_full());
}

#[test]
fn test_vulkan_rt_tier_default_is_none() {
    assert_eq!(VulkanRayTracingTier::default(), VulkanRayTracingTier::None);
}

// ============================================================================
// Section N: VulkanRayTracingTier name() and Display
// ============================================================================

#[test]
fn test_vulkan_rt_tier_name_none() {
    assert_eq!(VulkanRayTracingTier::None.name(), "None");
}

#[test]
fn test_vulkan_rt_tier_name_query() {
    assert_eq!(VulkanRayTracingTier::Query.name(), "Ray Query");
}

#[test]
fn test_vulkan_rt_tier_name_full() {
    assert_eq!(VulkanRayTracingTier::Full.name(), "Full Pipeline");
}

#[test]
fn test_vulkan_rt_tier_display_none() {
    assert_eq!(format!("{}", VulkanRayTracingTier::None), "None");
}

#[test]
fn test_vulkan_rt_tier_display_query() {
    assert_eq!(format!("{}", VulkanRayTracingTier::Query), "Ray Query");
}

#[test]
fn test_vulkan_rt_tier_display_full() {
    assert_eq!(format!("{}", VulkanRayTracingTier::Full), "Full Pipeline");
}

#[test]
fn test_vulkan_rt_tier_clone() {
    let tier = VulkanRayTracingTier::Full;
    let cloned = tier.clone();
    assert_eq!(tier, cloned);
}

#[test]
fn test_vulkan_rt_tier_debug() {
    let debug = format!("{:?}", VulkanRayTracingTier::Query);
    assert!(debug.contains("Query"));
}

// ============================================================================
// Section O: VulkanFeatures from_features() with empty features
// ============================================================================

#[test]
fn test_vulkan_features_from_empty_features_ray_tracing_false() {
    let vk = VulkanFeatures::from_features(Features::empty());
    assert!(!vk.ray_tracing);
}

#[test]
fn test_vulkan_features_from_empty_features_ray_query_false() {
    let vk = VulkanFeatures::from_features(Features::empty());
    assert!(!vk.ray_query);
}

#[test]
fn test_vulkan_features_from_empty_features_descriptor_indexing_false() {
    let vk = VulkanFeatures::from_features(Features::empty());
    assert!(!vk.descriptor_indexing);
}

#[test]
fn test_vulkan_features_from_empty_features_timeline_semaphores_false() {
    let vk = VulkanFeatures::from_features(Features::empty());
    assert!(!vk.timeline_semaphores);
}

#[test]
fn test_vulkan_features_from_empty_features_buffer_device_address_false() {
    let vk = VulkanFeatures::from_features(Features::empty());
    assert!(!vk.buffer_device_address);
}

#[test]
fn test_vulkan_features_from_empty_features_mesh_shading_false() {
    let vk = VulkanFeatures::from_features(Features::empty());
    assert!(!vk.mesh_shading);
}

#[test]
fn test_vulkan_features_from_empty_features_dynamic_rendering_false() {
    let vk = VulkanFeatures::from_features(Features::empty());
    assert!(!vk.dynamic_rendering);
}

#[test]
fn test_vulkan_features_from_empty_features_synchronization2_false() {
    let vk = VulkanFeatures::from_features(Features::empty());
    assert!(!vk.synchronization2);
}

#[test]
fn test_vulkan_features_from_empty_features_extended_dynamic_state_false() {
    let vk = VulkanFeatures::from_features(Features::empty());
    assert!(!vk.extended_dynamic_state);
}

#[test]
fn test_vulkan_features_from_empty_features_maintenance4_false() {
    let vk = VulkanFeatures::from_features(Features::empty());
    assert!(!vk.maintenance4);
}

// ============================================================================
// Section P: VulkanFeatures from_features() with ray tracing features
// ============================================================================

#[test]
fn test_vulkan_features_with_rt_acceleration_structure() {
    let features = Features::RAY_TRACING_ACCELERATION_STRUCTURE;
    let vk = VulkanFeatures::from_features(features);
    assert!(vk.ray_tracing);
}

#[test]
fn test_vulkan_features_with_rt_implies_bda() {
    let features = Features::RAY_TRACING_ACCELERATION_STRUCTURE;
    let vk = VulkanFeatures::from_features(features);
    assert!(vk.buffer_device_address, "RT should imply buffer device address");
}

#[test]
fn test_vulkan_features_with_rt_implies_timeline_semaphores() {
    let features = Features::RAY_TRACING_ACCELERATION_STRUCTURE;
    let vk = VulkanFeatures::from_features(features);
    assert!(vk.timeline_semaphores, "RT should imply timeline semaphores");
}

#[test]
fn test_vulkan_features_with_rt_implies_dynamic_rendering() {
    let features = Features::RAY_TRACING_ACCELERATION_STRUCTURE;
    let vk = VulkanFeatures::from_features(features);
    assert!(vk.dynamic_rendering, "RT should imply dynamic rendering");
}

#[test]
fn test_vulkan_features_with_rt_implies_synchronization2() {
    let features = Features::RAY_TRACING_ACCELERATION_STRUCTURE;
    let vk = VulkanFeatures::from_features(features);
    assert!(vk.synchronization2, "RT should imply synchronization2");
}

#[test]
fn test_vulkan_features_with_rt_implies_maintenance4() {
    let features = Features::RAY_TRACING_ACCELERATION_STRUCTURE;
    let vk = VulkanFeatures::from_features(features);
    assert!(vk.maintenance4, "RT should imply maintenance4");
}

#[test]
fn test_vulkan_features_with_rt_and_ray_query() {
    let features = Features::RAY_TRACING_ACCELERATION_STRUCTURE | Features::RAY_QUERY;
    let vk = VulkanFeatures::from_features(features);
    assert!(vk.ray_tracing);
    assert!(vk.ray_query);
}

// ============================================================================
// Section Q: VulkanFeatures from_features() with ray query only
// ============================================================================

#[test]
fn test_vulkan_features_ray_query_only() {
    let features = Features::RAY_QUERY;
    let vk = VulkanFeatures::from_features(features);
    assert!(!vk.ray_tracing);
    assert!(vk.ray_query);
}

#[test]
fn test_vulkan_features_ray_query_implies_bda() {
    let features = Features::RAY_QUERY;
    let vk = VulkanFeatures::from_features(features);
    assert!(vk.buffer_device_address, "Ray query should imply BDA");
}

#[test]
fn test_vulkan_features_ray_query_implies_timeline_semaphores() {
    let features = Features::RAY_QUERY;
    let vk = VulkanFeatures::from_features(features);
    assert!(vk.timeline_semaphores);
}

#[test]
fn test_vulkan_features_ray_query_implies_dynamic_rendering() {
    let features = Features::RAY_QUERY;
    let vk = VulkanFeatures::from_features(features);
    assert!(vk.dynamic_rendering);
}

#[test]
fn test_vulkan_features_ray_query_does_not_imply_maintenance4() {
    // maintenance4 is only implied by full RT
    let features = Features::RAY_QUERY;
    let vk = VulkanFeatures::from_features(features);
    assert!(!vk.maintenance4);
}

// ============================================================================
// Section R: VulkanFeatures from_features() with bindless features
// ============================================================================

#[test]
fn test_vulkan_features_bindless_all_required_flags() {
    let features = Features::TEXTURE_BINDING_ARRAY
        | Features::BUFFER_BINDING_ARRAY
        | Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING;
    let vk = VulkanFeatures::from_features(features);
    assert!(vk.descriptor_indexing);
}

#[test]
fn test_vulkan_features_bindless_implies_extended_dynamic_state() {
    let features = Features::TEXTURE_BINDING_ARRAY
        | Features::BUFFER_BINDING_ARRAY
        | Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING;
    let vk = VulkanFeatures::from_features(features);
    assert!(vk.extended_dynamic_state);
}

// ============================================================================
// Section S: VulkanFeatures from_features() with partial bindless (incomplete)
// ============================================================================

#[test]
fn test_vulkan_features_partial_bindless_texture_only() {
    let features = Features::TEXTURE_BINDING_ARRAY;
    let vk = VulkanFeatures::from_features(features);
    assert!(!vk.descriptor_indexing);
}

#[test]
fn test_vulkan_features_partial_bindless_texture_and_buffer() {
    let features = Features::TEXTURE_BINDING_ARRAY | Features::BUFFER_BINDING_ARRAY;
    let vk = VulkanFeatures::from_features(features);
    assert!(!vk.descriptor_indexing, "Missing non-uniform indexing flag");
}

#[test]
fn test_vulkan_features_partial_bindless_texture_and_nonuniform() {
    let features = Features::TEXTURE_BINDING_ARRAY
        | Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING;
    let vk = VulkanFeatures::from_features(features);
    assert!(!vk.descriptor_indexing, "Missing buffer binding array flag");
}

#[test]
fn test_vulkan_features_partial_bindless_buffer_and_nonuniform() {
    let features = Features::BUFFER_BINDING_ARRAY
        | Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING;
    let vk = VulkanFeatures::from_features(features);
    assert!(!vk.descriptor_indexing, "Missing texture binding array flag");
}

// ============================================================================
// Section T: VulkanFeatures supports_rt_pipeline()
// ============================================================================

#[test]
fn test_vulkan_features_supports_rt_pipeline_default_false() {
    let vk = VulkanFeatures::default();
    assert!(!vk.supports_rt_pipeline());
}

#[test]
fn test_vulkan_features_supports_rt_pipeline_only_rt_false() {
    let mut vk = VulkanFeatures::default();
    vk.ray_tracing = true;
    vk.buffer_device_address = false;
    assert!(!vk.supports_rt_pipeline());
}

#[test]
fn test_vulkan_features_supports_rt_pipeline_only_bda_false() {
    let mut vk = VulkanFeatures::default();
    vk.ray_tracing = false;
    vk.buffer_device_address = true;
    assert!(!vk.supports_rt_pipeline());
}

#[test]
fn test_vulkan_features_supports_rt_pipeline_both_true() {
    let mut vk = VulkanFeatures::default();
    vk.ray_tracing = true;
    vk.buffer_device_address = true;
    assert!(vk.supports_rt_pipeline());
}

// ============================================================================
// Section U: VulkanFeatures supports_ray_query()
// ============================================================================

#[test]
fn test_vulkan_features_supports_ray_query_default_false() {
    let vk = VulkanFeatures::default();
    assert!(!vk.supports_ray_query());
}

#[test]
fn test_vulkan_features_supports_ray_query_true() {
    let mut vk = VulkanFeatures::default();
    vk.ray_query = true;
    assert!(vk.supports_ray_query());
}

// ============================================================================
// Section V: VulkanFeatures supports_any_rt()
// ============================================================================

#[test]
fn test_vulkan_features_supports_any_rt_default_false() {
    let vk = VulkanFeatures::default();
    assert!(!vk.supports_any_rt());
}

#[test]
fn test_vulkan_features_supports_any_rt_only_rt_true() {
    let mut vk = VulkanFeatures::default();
    vk.ray_tracing = true;
    assert!(vk.supports_any_rt());
}

#[test]
fn test_vulkan_features_supports_any_rt_only_query_true() {
    let mut vk = VulkanFeatures::default();
    vk.ray_query = true;
    assert!(vk.supports_any_rt());
}

#[test]
fn test_vulkan_features_supports_any_rt_both_true() {
    let mut vk = VulkanFeatures::default();
    vk.ray_tracing = true;
    vk.ray_query = true;
    assert!(vk.supports_any_rt());
}

// ============================================================================
// Section W: VulkanFeatures supports_bindless()
// ============================================================================

#[test]
fn test_vulkan_features_supports_bindless_default_false() {
    let vk = VulkanFeatures::default();
    assert!(!vk.supports_bindless());
}

#[test]
fn test_vulkan_features_supports_bindless_only_di_false() {
    let mut vk = VulkanFeatures::default();
    vk.descriptor_indexing = true;
    vk.buffer_device_address = false;
    assert!(!vk.supports_bindless());
}

#[test]
fn test_vulkan_features_supports_bindless_only_bda_false() {
    let mut vk = VulkanFeatures::default();
    vk.descriptor_indexing = false;
    vk.buffer_device_address = true;
    assert!(!vk.supports_bindless());
}

#[test]
fn test_vulkan_features_supports_bindless_both_true() {
    let mut vk = VulkanFeatures::default();
    vk.descriptor_indexing = true;
    vk.buffer_device_address = true;
    assert!(vk.supports_bindless());
}

// ============================================================================
// Section X: VulkanFeatures supports_vulkan_1_2()
// ============================================================================

#[test]
fn test_vulkan_features_supports_vulkan_1_2_default_false() {
    let vk = VulkanFeatures::default();
    assert!(!vk.supports_vulkan_1_2());
}

#[test]
fn test_vulkan_features_supports_vulkan_1_2_only_ts_false() {
    let mut vk = VulkanFeatures::default();
    vk.timeline_semaphores = true;
    vk.buffer_device_address = false;
    assert!(!vk.supports_vulkan_1_2());
}

#[test]
fn test_vulkan_features_supports_vulkan_1_2_only_bda_false() {
    let mut vk = VulkanFeatures::default();
    vk.timeline_semaphores = false;
    vk.buffer_device_address = true;
    assert!(!vk.supports_vulkan_1_2());
}

#[test]
fn test_vulkan_features_supports_vulkan_1_2_both_true() {
    let mut vk = VulkanFeatures::default();
    vk.timeline_semaphores = true;
    vk.buffer_device_address = true;
    assert!(vk.supports_vulkan_1_2());
}

// ============================================================================
// Section Y: VulkanFeatures supports_vulkan_1_3()
// ============================================================================

#[test]
fn test_vulkan_features_supports_vulkan_1_3_default_false() {
    let vk = VulkanFeatures::default();
    assert!(!vk.supports_vulkan_1_3());
}

#[test]
fn test_vulkan_features_supports_vulkan_1_3_only_dr_false() {
    let mut vk = VulkanFeatures::default();
    vk.dynamic_rendering = true;
    vk.synchronization2 = false;
    assert!(!vk.supports_vulkan_1_3());
}

#[test]
fn test_vulkan_features_supports_vulkan_1_3_only_sync2_false() {
    let mut vk = VulkanFeatures::default();
    vk.dynamic_rendering = false;
    vk.synchronization2 = true;
    assert!(!vk.supports_vulkan_1_3());
}

#[test]
fn test_vulkan_features_supports_vulkan_1_3_both_true() {
    let mut vk = VulkanFeatures::default();
    vk.dynamic_rendering = true;
    vk.synchronization2 = true;
    assert!(vk.supports_vulkan_1_3());
}

// ============================================================================
// Section Z: VulkanFeatures ray_tracing_tier()
// ============================================================================

#[test]
fn test_vulkan_features_ray_tracing_tier_none_default() {
    let vk = VulkanFeatures::default();
    assert_eq!(vk.ray_tracing_tier(), VulkanRayTracingTier::None);
}

#[test]
fn test_vulkan_features_ray_tracing_tier_query_only() {
    let mut vk = VulkanFeatures::default();
    vk.ray_query = true;
    assert_eq!(vk.ray_tracing_tier(), VulkanRayTracingTier::Query);
}

#[test]
fn test_vulkan_features_ray_tracing_tier_rt_without_bda_is_query() {
    // RT without BDA falls back to Query if ray_query is set
    let mut vk = VulkanFeatures::default();
    vk.ray_tracing = true;
    vk.buffer_device_address = false;
    vk.ray_query = true;
    assert_eq!(vk.ray_tracing_tier(), VulkanRayTracingTier::Query);
}

#[test]
fn test_vulkan_features_ray_tracing_tier_rt_without_bda_without_query_is_none() {
    // RT without BDA and without ray_query is None
    let mut vk = VulkanFeatures::default();
    vk.ray_tracing = true;
    vk.buffer_device_address = false;
    vk.ray_query = false;
    assert_eq!(vk.ray_tracing_tier(), VulkanRayTracingTier::None);
}

#[test]
fn test_vulkan_features_ray_tracing_tier_full() {
    let mut vk = VulkanFeatures::default();
    vk.ray_tracing = true;
    vk.buffer_device_address = true;
    assert_eq!(vk.ray_tracing_tier(), VulkanRayTracingTier::Full);
}

#[test]
fn test_vulkan_features_ray_tracing_tier_full_takes_precedence_over_query() {
    let mut vk = VulkanFeatures::default();
    vk.ray_tracing = true;
    vk.ray_query = true;
    vk.buffer_device_address = true;
    assert_eq!(vk.ray_tracing_tier(), VulkanRayTracingTier::Full);
}

// ============================================================================
// Section AA: VulkanFeatures summary()
// ============================================================================

#[test]
fn test_vulkan_features_summary_empty_returns_none() {
    let vk = VulkanFeatures::default();
    assert_eq!(vk.summary(), "None");
}

#[test]
fn test_vulkan_features_summary_with_rt() {
    let mut vk = VulkanFeatures::default();
    vk.ray_tracing = true;
    let summary = vk.summary();
    assert!(summary.contains("RT-Pipeline"));
}

#[test]
fn test_vulkan_features_summary_with_ray_query() {
    let mut vk = VulkanFeatures::default();
    vk.ray_query = true;
    let summary = vk.summary();
    assert!(summary.contains("RT-Query"));
}

#[test]
fn test_vulkan_features_summary_with_descriptor_indexing() {
    let mut vk = VulkanFeatures::default();
    vk.descriptor_indexing = true;
    let summary = vk.summary();
    assert!(summary.contains("Descriptor-Indexing"));
}

#[test]
fn test_vulkan_features_summary_with_timeline_semaphores() {
    let mut vk = VulkanFeatures::default();
    vk.timeline_semaphores = true;
    let summary = vk.summary();
    assert!(summary.contains("Timeline-Semaphores"));
}

#[test]
fn test_vulkan_features_summary_with_bda() {
    let mut vk = VulkanFeatures::default();
    vk.buffer_device_address = true;
    let summary = vk.summary();
    assert!(summary.contains("BDA"));
}

#[test]
fn test_vulkan_features_summary_with_mesh_shading() {
    let mut vk = VulkanFeatures::default();
    vk.mesh_shading = true;
    let summary = vk.summary();
    assert!(summary.contains("Mesh-Shaders"));
}

#[test]
fn test_vulkan_features_summary_with_dynamic_rendering() {
    let mut vk = VulkanFeatures::default();
    vk.dynamic_rendering = true;
    let summary = vk.summary();
    assert!(summary.contains("Dynamic-Rendering"));
}

#[test]
fn test_vulkan_features_summary_multiple_features() {
    let mut vk = VulkanFeatures::default();
    vk.ray_tracing = true;
    vk.ray_query = true;
    vk.descriptor_indexing = true;
    let summary = vk.summary();

    assert!(summary.contains("RT-Pipeline"));
    assert!(summary.contains("RT-Query"));
    assert!(summary.contains("Descriptor-Indexing"));
    assert!(summary.contains(", "), "Multiple features should be comma-separated");
}

#[test]
fn test_vulkan_features_summary_all_features() {
    let vk = VulkanFeatures {
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

    let summary = vk.summary();
    assert!(summary.contains("RT-Pipeline"));
    assert!(summary.contains("RT-Query"));
    assert!(summary.contains("Descriptor-Indexing"));
    assert!(summary.contains("Timeline-Semaphores"));
    assert!(summary.contains("BDA"));
    assert!(summary.contains("Mesh-Shaders"));
    assert!(summary.contains("Dynamic-Rendering"));
}

#[test]
fn test_vulkan_features_summary_does_not_include_sync2() {
    // synchronization2 is not included in summary()
    let mut vk = VulkanFeatures::default();
    vk.synchronization2 = true;
    let summary = vk.summary();
    assert!(!summary.contains("Sync"));
    assert_eq!(summary, "None");
}

#[test]
fn test_vulkan_features_summary_does_not_include_extended_dynamic_state() {
    let mut vk = VulkanFeatures::default();
    vk.extended_dynamic_state = true;
    let summary = vk.summary();
    assert!(!summary.contains("Extended"));
    assert_eq!(summary, "None");
}

#[test]
fn test_vulkan_features_summary_does_not_include_maintenance4() {
    let mut vk = VulkanFeatures::default();
    vk.maintenance4 = true;
    let summary = vk.summary();
    assert!(!summary.contains("Maintenance"));
    assert_eq!(summary, "None");
}

// ============================================================================
// Section AB: Edge cases and boundary conditions
// ============================================================================

#[test]
fn test_vulkan_features_default_equals_all_false() {
    let vk = VulkanFeatures::default();
    let expected = VulkanFeatures {
        ray_tracing: false,
        ray_query: false,
        descriptor_indexing: false,
        timeline_semaphores: false,
        buffer_device_address: false,
        mesh_shading: false,
        dynamic_rendering: false,
        synchronization2: false,
        extended_dynamic_state: false,
        maintenance4: false,
    };
    assert_eq!(vk, expected);
}

#[test]
fn test_vulkan_features_clone() {
    let mut vk = VulkanFeatures::default();
    vk.ray_tracing = true;
    vk.descriptor_indexing = true;

    let cloned = vk.clone();
    assert_eq!(vk, cloned);
    assert!(cloned.ray_tracing);
    assert!(cloned.descriptor_indexing);
}

#[test]
fn test_vulkan_features_copy() {
    let mut vk = VulkanFeatures::default();
    vk.ray_query = true;

    let copied = vk;
    assert!(copied.ray_query);
    assert!(vk.ray_query); // Original still valid (Copy)
}

#[test]
fn test_vulkan_features_debug() {
    let vk = VulkanFeatures::default();
    let debug = format!("{:?}", vk);
    assert!(debug.contains("VulkanFeatures"));
    assert!(debug.contains("ray_tracing"));
}

#[test]
fn test_vulkan_features_eq() {
    let vk1 = VulkanFeatures::default();
    let vk2 = VulkanFeatures::default();
    assert_eq!(vk1, vk2);

    let mut vk3 = VulkanFeatures::default();
    vk3.ray_tracing = true;
    assert_ne!(vk1, vk3);
}

#[test]
fn test_backend_capabilities_clone() {
    let mut caps = BackendCapabilities::default();
    caps.backend = BackendType::Vulkan;
    caps.ray_tracing = true;

    let cloned = caps.clone();
    assert_eq!(cloned.backend, BackendType::Vulkan);
    assert!(cloned.ray_tracing);
}

#[test]
fn test_backend_capabilities_debug() {
    let caps = BackendCapabilities::default();
    let debug = format!("{:?}", caps);
    assert!(debug.contains("BackendCapabilities"));
}

#[test]
fn test_vulkan_rt_tier_hash() {
    use std::collections::HashSet;
    let mut set = HashSet::new();
    set.insert(VulkanRayTracingTier::None);
    set.insert(VulkanRayTracingTier::Query);
    set.insert(VulkanRayTracingTier::Full);
    set.insert(VulkanRayTracingTier::None); // Duplicate

    assert_eq!(set.len(), 3);
}

#[test]
fn test_from_features_combines_rt_and_bindless() {
    let features = Features::RAY_TRACING_ACCELERATION_STRUCTURE
        | Features::RAY_QUERY
        | Features::TEXTURE_BINDING_ARRAY
        | Features::BUFFER_BINDING_ARRAY
        | Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING;

    let vk = VulkanFeatures::from_features(features);

    assert!(vk.ray_tracing);
    assert!(vk.ray_query);
    assert!(vk.descriptor_indexing);
    assert!(vk.buffer_device_address);
    assert!(vk.supports_rt_pipeline());
    assert!(vk.supports_bindless());
    assert_eq!(vk.ray_tracing_tier(), VulkanRayTracingTier::Full);
}

#[test]
fn test_backend_type_all_variants_different_names() {
    let backends = [
        BackendType::Vulkan,
        BackendType::Metal,
        BackendType::Dx12,
        BackendType::Dx11,
        BackendType::Gl,
        BackendType::WebGpu,
        BackendType::Empty,
        BackendType::Unknown,
    ];

    // All names should be unique
    let names: Vec<_> = backends.iter().map(|b| b.name()).collect();
    let unique_count = names.iter().collect::<std::collections::HashSet<_>>().len();
    assert_eq!(unique_count, backends.len(), "All backend names should be unique");
}

#[test]
fn test_modern_features_imply_vulkan_versions() {
    // Full RT should imply both 1.2 and 1.3 support
    let features = Features::RAY_TRACING_ACCELERATION_STRUCTURE | Features::RAY_QUERY;
    let vk = VulkanFeatures::from_features(features);

    assert!(vk.supports_vulkan_1_2(), "RT should imply Vulkan 1.2");
    assert!(vk.supports_vulkan_1_3(), "RT should imply Vulkan 1.3");
}

#[test]
fn test_backend_type_capability_consistency() {
    // Backends that support RT should also support mesh shaders and bindless
    let rt_backends = [BackendType::Vulkan, BackendType::Metal, BackendType::Dx12];

    for backend in rt_backends {
        assert!(
            backend.supports_ray_tracing(),
            "{:?} should support ray tracing",
            backend
        );
        assert!(
            backend.supports_mesh_shaders(),
            "{:?} should support mesh shaders",
            backend
        );
        assert!(
            backend.supports_bindless(),
            "{:?} should support bindless",
            backend
        );
    }
}

#[test]
fn test_non_rt_backends_no_advanced_features() {
    let non_rt_backends = [
        BackendType::Dx11,
        BackendType::Gl,
        BackendType::WebGpu,
        BackendType::Empty,
        BackendType::Unknown,
    ];

    for backend in non_rt_backends {
        assert!(
            !backend.supports_ray_tracing(),
            "{:?} should not support ray tracing",
            backend
        );
        assert!(
            !backend.supports_mesh_shaders(),
            "{:?} should not support mesh shaders",
            backend
        );
        assert!(
            !backend.supports_bindless(),
            "{:?} should not support bindless",
            backend
        );
    }
}

#[test]
fn test_vulkan_features_summary_order_is_deterministic() {
    let vk = VulkanFeatures {
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

    let summary1 = vk.summary();
    let summary2 = vk.summary();
    assert_eq!(summary1, summary2, "Summary should be deterministic");
}

#[test]
fn test_backend_capabilities_full_vulkan_setup() {
    let caps = BackendCapabilities {
        backend: BackendType::Vulkan,
        ray_tracing: true,
        ray_query: true,
        mesh_shaders: true,
        bindless: true,
        timeline_semaphores: true,
        buffer_device_address: true,
        dynamic_rendering: true,
    };

    assert!(caps.supports_full_rt());
    assert!(caps.supports_gpu_driven());
}

#[test]
fn test_backend_capabilities_minimal_dx12_setup() {
    // DX12 doesn't require buffer_device_address for GPU-driven
    let caps = BackendCapabilities {
        backend: BackendType::Dx12,
        ray_tracing: true,
        ray_query: true,
        mesh_shaders: false,
        bindless: true,
        timeline_semaphores: false,
        buffer_device_address: false,
        dynamic_rendering: false,
    };

    assert!(caps.supports_full_rt());
    assert!(caps.supports_gpu_driven());
}

#[test]
fn test_vulkan_rt_tier_progression() {
    // Test that tiers form a logical progression
    let none = VulkanRayTracingTier::None;
    let query = VulkanRayTracingTier::Query;
    let full = VulkanRayTracingTier::Full;

    // None < Query < Full in terms of capability
    assert!(!none.is_available());
    assert!(query.is_available() && !query.is_full());
    assert!(full.is_available() && full.is_full());
}
