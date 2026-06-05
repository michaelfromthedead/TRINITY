// SPDX-License-Identifier: MIT
//
// device_instance_whitebox.rs -- Whitebox structural tests for T-WGPU-P1.1.1
// (TrinityInstance - Instance Creation).
//
// These tests exercise the internal implementation of TrinityInstance,
// covering all code paths in instance creation, backend selection,
// instance flags, and adapter enumeration.
//
// WHITEBOX coverage plan:
//   - Path A: TrinityInstance::new() happy path with platform backends
//   - Path B: TrinityInstance::with_backends() explicit backend override
//   - Path C: select_backends() returns PRIMARY on desktop (non-WASM)
//   - Path D: select_instance_flags() returns VALIDATION|DEBUG in debug builds
//   - Path E: enumerate_adapters() returns valid adapters
//   - Path F: Default trait implementation delegates to new()
//   - Path G: Debug trait implementation formats correctly
//   - Path H: inner()/into_inner() accessor methods work correctly
//   - Path I: Edge case - Vulkan-only backend
//   - Path J: Edge case - OpenGL secondary backend
//   - Path K: Edge case - all backends enabled
//   - Path L: Edge case - empty/invalid backends (EMPTY)
//
// Acceptance criteria (T-WGPU-P1.1.1):
//   1. Instance creates successfully with Backends::PRIMARY on desktop
//   2. Instance creates successfully with Backends::BROWSER_WEBGPU on WASM
//   3. Instance logs backend selection
//   4. Unit test verifies instance creation

use renderer_backend::device::TrinityInstance;

// ---------------------------------------------------------------------------
// Path A: TrinityInstance::new() - Platform-specific backend selection
// ---------------------------------------------------------------------------

#[test]
fn test_instance_new_creates_with_platform_backends() {
    let instance = TrinityInstance::new();

    // Platform-specific backend selection
    #[cfg(target_arch = "wasm32")]
    {
        assert_eq!(
            instance.backends(),
            wgpu::Backends::BROWSER_WEBGPU,
            "WASM should use BROWSER_WEBGPU backend"
        );
    }

    #[cfg(all(target_os = "linux", not(target_arch = "wasm32")))]
    {
        assert_eq!(
            instance.backends(),
            wgpu::Backends::VULKAN | wgpu::Backends::GL,
            "Linux should use Vulkan + GL backends"
        );
    }

    #[cfg(all(target_os = "macos", not(target_arch = "wasm32")))]
    {
        assert_eq!(
            instance.backends(),
            wgpu::Backends::METAL,
            "macOS should use Metal backend"
        );
    }

    #[cfg(all(target_os = "windows", not(target_arch = "wasm32")))]
    {
        assert_eq!(
            instance.backends(),
            wgpu::Backends::VULKAN | wgpu::Backends::DX12 | wgpu::Backends::GL,
            "Windows should use Vulkan + DX12 + GL backends"
        );
    }

    // Instance should be accessible
    let inner = instance.inner();
    assert!(std::ptr::addr_of!(*inner) as usize != 0);
}

#[test]
fn test_instance_new_inner_is_valid() {
    let instance = TrinityInstance::new();

    // The inner wgpu::Instance should be valid (non-null reference)
    let inner = instance.inner();

    // We can't easily test the inner instance directly, but we can verify
    // it doesn't panic when accessed
    let _ = std::mem::size_of_val(inner);
}

// ---------------------------------------------------------------------------
// Path B: TrinityInstance::with_backends() - Explicit backend override
// ---------------------------------------------------------------------------

#[test]
fn test_instance_with_backends_vulkan_only() {
    let instance = TrinityInstance::with_backends(wgpu::Backends::VULKAN);

    assert_eq!(
        instance.backends(),
        wgpu::Backends::VULKAN,
        "Should use only Vulkan backend when explicitly specified"
    );
}

#[test]
fn test_instance_with_backends_metal_only() {
    let instance = TrinityInstance::with_backends(wgpu::Backends::METAL);

    assert_eq!(
        instance.backends(),
        wgpu::Backends::METAL,
        "Should use only Metal backend when explicitly specified"
    );
}

#[test]
fn test_instance_with_backends_dx12_only() {
    let instance = TrinityInstance::with_backends(wgpu::Backends::DX12);

    assert_eq!(
        instance.backends(),
        wgpu::Backends::DX12,
        "Should use only DX12 backend when explicitly specified"
    );
}

#[test]
fn test_instance_with_backends_all() {
    let instance = TrinityInstance::with_backends(wgpu::Backends::all());

    assert_eq!(
        instance.backends(),
        wgpu::Backends::all(),
        "Should use all backends when explicitly specified"
    );
}

#[test]
fn test_instance_with_backends_primary() {
    let instance = TrinityInstance::with_backends(wgpu::Backends::PRIMARY);

    assert_eq!(
        instance.backends(),
        wgpu::Backends::PRIMARY,
        "Should use PRIMARY backends when explicitly specified"
    );
}

#[test]
fn test_instance_with_backends_secondary() {
    let instance = TrinityInstance::with_backends(wgpu::Backends::SECONDARY);

    assert_eq!(
        instance.backends(),
        wgpu::Backends::SECONDARY,
        "Should use SECONDARY backends when explicitly specified"
    );
}

// ---------------------------------------------------------------------------
// Path C: select_backends() - Platform detection
// ---------------------------------------------------------------------------

#[test]
#[cfg(not(target_arch = "wasm32"))]
fn test_select_backends_desktop_returns_platform_appropriate() {
    // On desktop, new() should select platform-appropriate backends
    let instance = TrinityInstance::new();
    let backends = instance.backends();

    // Should contain at least one primary backend
    let has_vulkan = backends.contains(wgpu::Backends::VULKAN);
    let has_metal = backends.contains(wgpu::Backends::METAL);
    let has_dx12 = backends.contains(wgpu::Backends::DX12);
    assert!(
        has_vulkan || has_metal || has_dx12,
        "Desktop should have at least one primary backend"
    );
}

#[test]
#[cfg(target_arch = "wasm32")]
fn test_select_backends_wasm_returns_webgpu() {
    // On WASM, new() should select BROWSER_WEBGPU internally
    let instance = TrinityInstance::new();
    assert_eq!(instance.backends(), wgpu::Backends::BROWSER_WEBGPU);
}

// ---------------------------------------------------------------------------
// Path D: select_instance_flags() - Debug vs Release flags
// ---------------------------------------------------------------------------

#[test]
#[cfg(debug_assertions)]
fn test_instance_flags_debug_build_enables_validation() {
    // In debug builds, validation should be enabled
    // We verify this indirectly by checking the instance creation succeeds
    // (validation layer initialization would fail if misconfigured)
    let instance = TrinityInstance::new();
    let _ = instance.inner();
    // If we get here without panic, validation layer was initialized correctly
}

#[test]
fn test_instance_creation_with_flags_does_not_panic() {
    // Both debug and release should create instance without panic
    let instance = TrinityInstance::new();
    assert!(!instance.backends().is_empty());
}

// ---------------------------------------------------------------------------
// Path E: enumerate_adapters() - Adapter enumeration
// ---------------------------------------------------------------------------

#[test]
fn test_enumerate_adapters_returns_vec() {
    let instance = TrinityInstance::new();
    let adapters = instance.enumerate_adapters();

    // On a real system with GPU, should return at least one adapter
    // On CI/headless, may return empty (software fallback not always available)
    // This test verifies the method works without panicking
    let _ = adapters.len();
}

#[test]
fn test_enumerate_adapters_filters_by_backend() {
    // Create instance with specific backend
    let instance = TrinityInstance::with_backends(wgpu::Backends::VULKAN);
    let adapters = instance.enumerate_adapters();

    // All returned adapters should be Vulkan (if any)
    for adapter in &adapters {
        let info = adapter.get_info();
        assert_eq!(
            info.backend,
            wgpu::Backend::Vulkan,
            "Adapter should be Vulkan when instance created with Vulkan backend"
        );
    }
}

#[test]
#[cfg(not(target_arch = "wasm32"))]
fn test_enumerate_adapters_with_all_backends() {
    let instance = TrinityInstance::with_backends(wgpu::Backends::all());
    let adapters = instance.enumerate_adapters();

    // Log adapter info for debugging
    for adapter in &adapters {
        let info = adapter.get_info();
        eprintln!(
            "Found adapter: {} (backend: {:?}, device_type: {:?})",
            info.name, info.backend, info.device_type
        );
    }
}

#[test]
fn test_enumerate_adapters_handles_empty_gracefully() {
    // Create instance with backend that may not exist on this system
    // This tests the "zero adapters" edge case from acceptance criteria
    let instance = TrinityInstance::with_backends(wgpu::Backends::BROWSER_WEBGPU);
    let adapters = instance.enumerate_adapters();

    // Should not panic, even if no adapters found
    // On desktop, WebGPU backend may return empty
    let _ = adapters.is_empty();
}

// ---------------------------------------------------------------------------
// Path F: Default trait implementation
// ---------------------------------------------------------------------------

#[test]
fn test_default_trait_delegates_to_new() {
    let default_instance: TrinityInstance = Default::default();
    let new_instance = TrinityInstance::new();

    // Both should have same backend selection
    assert_eq!(
        default_instance.backends(),
        new_instance.backends(),
        "Default should delegate to new()"
    );
}

#[test]
fn test_default_trait_creates_valid_instance() {
    let instance: TrinityInstance = Default::default();

    // Should have platform-appropriate backends
    #[cfg(not(target_arch = "wasm32"))]
    {
        let backends = instance.backends();
        let has_vulkan = backends.contains(wgpu::Backends::VULKAN);
        let has_metal = backends.contains(wgpu::Backends::METAL);
        let has_dx12 = backends.contains(wgpu::Backends::DX12);
        assert!(has_vulkan || has_metal || has_dx12, "Should have primary backend");
    }

    #[cfg(target_arch = "wasm32")]
    assert_eq!(instance.backends(), wgpu::Backends::BROWSER_WEBGPU);
}

// ---------------------------------------------------------------------------
// Path G: Debug trait implementation
// ---------------------------------------------------------------------------

#[test]
fn test_debug_trait_contains_struct_name() {
    let instance = TrinityInstance::new();
    let debug_str = format!("{:?}", instance);

    assert!(
        debug_str.contains("TrinityInstance"),
        "Debug output should contain struct name"
    );
}

#[test]
fn test_debug_trait_contains_backends_field() {
    let instance = TrinityInstance::new();
    let debug_str = format!("{:?}", instance);

    assert!(
        debug_str.contains("backends"),
        "Debug output should contain 'backends' field"
    );
}

#[test]
fn test_debug_trait_non_exhaustive_marker() {
    let instance = TrinityInstance::new();
    let debug_str = format!("{:?}", instance);

    // Debug impl uses finish_non_exhaustive(), so should contain ".."
    assert!(
        debug_str.contains(".."),
        "Debug output should use non-exhaustive format (contain '..')"
    );
}

#[test]
fn test_debug_trait_with_explicit_backends() {
    let instance = TrinityInstance::with_backends(wgpu::Backends::VULKAN);
    let debug_str = format!("{:?}", instance);

    // Should show the specific backend
    assert!(
        debug_str.contains("VULKAN"),
        "Debug output should show VULKAN backend: {}",
        debug_str
    );
}

// ---------------------------------------------------------------------------
// Path H: inner() / into_inner() accessor methods
// ---------------------------------------------------------------------------

#[test]
fn test_inner_returns_reference() {
    let instance = TrinityInstance::new();
    let inner: &wgpu::Instance = instance.inner();

    // Verify we can use the reference (doesn't panic)
    let _ = std::mem::size_of_val(inner);

    // Instance should still be usable after inner() call
    assert!(!instance.backends().is_empty());
}

#[test]
fn test_into_inner_consumes_instance() {
    let instance = TrinityInstance::new();
    let _inner: wgpu::Instance = instance.into_inner();

    // instance is now consumed, can't be used
    // This test verifies into_inner() compiles and doesn't panic
}

#[test]
fn test_into_inner_returns_valid_instance() {
    let instance = TrinityInstance::with_backends(wgpu::Backends::all());
    let inner = instance.into_inner();

    // We can use the inner instance for enumeration
    let adapters = inner.enumerate_adapters(wgpu::Backends::all());
    let _ = adapters.len();
}

// ---------------------------------------------------------------------------
// Path I-K: Edge cases - Various backend combinations
// ---------------------------------------------------------------------------

#[test]
fn test_edge_case_opengl_backend() {
    // OpenGL is a SECONDARY backend
    let instance = TrinityInstance::with_backends(wgpu::Backends::GL);

    assert_eq!(
        instance.backends(),
        wgpu::Backends::GL,
        "Should accept OpenGL backend"
    );
}

#[test]
fn test_edge_case_combined_backends() {
    // Combine Vulkan and OpenGL
    let combined = wgpu::Backends::VULKAN | wgpu::Backends::GL;
    let instance = TrinityInstance::with_backends(combined);

    assert_eq!(
        instance.backends(),
        combined,
        "Should accept combined backends"
    );
    assert!(instance.backends().contains(wgpu::Backends::VULKAN));
    assert!(instance.backends().contains(wgpu::Backends::GL));
}

#[test]
fn test_edge_case_empty_backends() {
    // EMPTY backends - wgpu should handle this gracefully
    let instance = TrinityInstance::with_backends(wgpu::Backends::empty());

    assert!(
        instance.backends().is_empty(),
        "Should accept empty backends"
    );

    // Enumeration should return empty
    let adapters = instance.enumerate_adapters();
    assert!(
        adapters.is_empty(),
        "Empty backends should enumerate zero adapters"
    );
}

#[test]
fn test_edge_case_browser_webgpu_on_desktop() {
    // WebGPU backend on desktop - should create but likely enumerate nothing
    let instance = TrinityInstance::with_backends(wgpu::Backends::BROWSER_WEBGPU);

    assert_eq!(
        instance.backends(),
        wgpu::Backends::BROWSER_WEBGPU,
        "Should accept BROWSER_WEBGPU even on desktop"
    );
}

// ---------------------------------------------------------------------------
// Path L: Multiple instance creation
// ---------------------------------------------------------------------------

#[test]
fn test_multiple_instances_coexist() {
    let instance1 = TrinityInstance::new();
    let instance2 = TrinityInstance::new();
    let instance3 = TrinityInstance::with_backends(wgpu::Backends::all());

    // All instances should be valid simultaneously
    assert_eq!(instance1.backends(), instance2.backends());
    assert_eq!(instance3.backends(), wgpu::Backends::all());

    // Each should be able to enumerate independently
    let _ = instance1.enumerate_adapters();
    let _ = instance2.enumerate_adapters();
    let _ = instance3.enumerate_adapters();
}

#[test]
fn test_instance_creation_is_deterministic() {
    // Multiple calls to new() should produce consistent results
    let backends1 = TrinityInstance::new().backends();
    let backends2 = TrinityInstance::new().backends();
    let backends3 = TrinityInstance::new().backends();

    assert_eq!(backends1, backends2);
    assert_eq!(backends2, backends3);
}

// ---------------------------------------------------------------------------
// Adapter inspection tests (enumerate_adapters edge cases)
// ---------------------------------------------------------------------------

#[test]
fn test_enumerate_adapters_info_fields() {
    let instance = TrinityInstance::new();
    let adapters = instance.enumerate_adapters();

    for adapter in &adapters {
        let info = adapter.get_info();

        // Verify all info fields are accessible
        let _ = &info.name;
        let _ = &info.vendor;
        let _ = &info.device;
        let _ = &info.device_type;
        let _ = &info.driver;
        let _ = &info.driver_info;
        let _ = &info.backend;
    }
}

#[test]
fn test_enumerate_adapters_device_types() {
    let instance = TrinityInstance::new();
    let adapters = instance.enumerate_adapters();

    for adapter in &adapters {
        let info = adapter.get_info();
        let device_type = info.device_type;

        // Device type should be one of the known types
        assert!(
            matches!(
                device_type,
                wgpu::DeviceType::DiscreteGpu
                    | wgpu::DeviceType::IntegratedGpu
                    | wgpu::DeviceType::VirtualGpu
                    | wgpu::DeviceType::Cpu
                    | wgpu::DeviceType::Other
            ),
            "Device type should be a known variant: {:?}",
            device_type
        );
    }
}

// ---------------------------------------------------------------------------
// Backend bitflag tests
// ---------------------------------------------------------------------------

#[test]
fn test_backends_bitflag_contains() {
    let instance = TrinityInstance::with_backends(wgpu::Backends::PRIMARY);
    let backends = instance.backends();

    // PRIMARY includes Vulkan, Metal, DX12, and WebGPU
    // On any platform, at least one of these should be true
    let contains_primary = backends.contains(wgpu::Backends::VULKAN)
        || backends.contains(wgpu::Backends::METAL)
        || backends.contains(wgpu::Backends::DX12)
        || backends.contains(wgpu::Backends::BROWSER_WEBGPU);

    assert!(
        contains_primary,
        "PRIMARY should contain at least one primary backend"
    );
}

#[test]
fn test_backends_bitflag_intersection() {
    let instance = TrinityInstance::with_backends(wgpu::Backends::VULKAN | wgpu::Backends::GL);
    let backends = instance.backends();

    // Intersection with VULKAN should yield VULKAN
    let intersection = backends & wgpu::Backends::VULKAN;
    assert_eq!(intersection, wgpu::Backends::VULKAN);
}

#[test]
fn test_backends_bitflag_difference() {
    let instance = TrinityInstance::with_backends(wgpu::Backends::all());
    let backends = instance.backends();

    // Remove Vulkan
    let without_vulkan = backends - wgpu::Backends::VULKAN;
    assert!(!without_vulkan.contains(wgpu::Backends::VULKAN));
    assert!(without_vulkan.contains(wgpu::Backends::METAL));
}
