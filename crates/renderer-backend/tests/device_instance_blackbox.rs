// Blackbox contract tests for T-WGPU-P1.1.1 Instance Creation
//
// CLEANROOM: No access to implementation files. Tests use only the public API
// exported by `renderer_backend::device::TrinityInstance`.
//
// Forbidden files (per TESTDEV_BLACKBOX prompt):
//   - crates/renderer-backend/src/device/instance.rs
//   - crates/renderer-backend/src/device/mod.rs (implementation details)
//   - Any WHITEBOX test file for this task
//
// Contract sources:
//   - docs/WGPU_DOCS/RDC_OUTPUT/PHASE_1_CORE_TODO.md (T-WGPU-P1.1.1)
//   - docs/WGPU_DOCS/RDC_OUTPUT/PHASE_1_CORE_ARCH.md (TrinityInstance spec)
//
// Acceptance criteria (T-WGPU-P1.1.1):
//   - Instance creates successfully with Backends::PRIMARY on desktop
//   - Instance creates successfully with Backends::BROWSER_WEBGPU on WASM
//   - Instance logs backend selection
//   - Unit test verifies instance creation
//
// Test design rationale:
//   Equivalence partitioning:
//     - Valid construction (new, with_backends)
//     - Backend enumeration (non-WASM platform)
//   Boundary cases:
//     - Empty adapter list (no GPU)
//     - Single adapter (common laptop case)
//     - Multiple adapters (multi-GPU workstation)
//   Error cases:
//     - None applicable at instance level (wgpu::Instance always succeeds)

use renderer_backend::device::TrinityInstance;

// =============================================================================
// 1. Basic Instance Construction
// =============================================================================

/// Verifies that TrinityInstance::new() returns a valid instance.
///
/// Contract: The default constructor creates an instance with platform-appropriate
/// backends (PRIMARY on desktop, BROWSER_WEBGPU on WASM).
#[test]
fn test_instance_new_returns_valid_instance() {
    let instance = TrinityInstance::new();

    // The instance should exist and be usable - verify by checking it can
    // enumerate adapters without panicking.
    let _ = instance.enumerate_adapters();
}

/// Verifies that TrinityInstance can be created with explicit backend specification.
///
/// Contract: TrinityInstance::with_backends() accepts a Backends parameter and
/// creates an instance configured for those backends.
#[test]
fn test_instance_with_backends_primary() {
    // Contract: with_backends(PRIMARY) should work on desktop platforms
    #[cfg(not(target_arch = "wasm32"))]
    {
        let instance = TrinityInstance::with_backends(wgpu::Backends::PRIMARY);

        // Instance should be usable
        let _ = instance.enumerate_adapters();
    }
}

/// Verifies that requesting VULKAN backend specifically works (where available).
///
/// Contract: Specific backend requests are honored when the backend is available.
#[test]
#[cfg(not(target_arch = "wasm32"))]
fn test_instance_with_specific_backend_vulkan() {
    // On Linux/Windows, Vulkan is typically available
    #[cfg(any(target_os = "linux", target_os = "windows"))]
    {
        let instance = TrinityInstance::with_backends(wgpu::Backends::VULKAN);
        // Should not panic; adapters may or may not be present depending on drivers
        let _ = instance.enumerate_adapters();
    }
}

// =============================================================================
// 2. Backend Query
// =============================================================================

/// Verifies that the instance reports its configured backends.
///
/// Contract: instance.backends() returns the Backends the instance was created with.
#[test]
fn test_instance_backends_returns_configured_backends() {
    let instance = TrinityInstance::new();

    // On non-WASM, default should be PRIMARY
    #[cfg(not(target_arch = "wasm32"))]
    {
        let backends = instance.backends();
        // PRIMARY includes Vulkan, Metal, DX12
        assert!(
            backends.contains(wgpu::Backends::VULKAN)
                || backends.contains(wgpu::Backends::METAL)
                || backends.contains(wgpu::Backends::DX12),
            "Expected PRIMARY backends on desktop, got {:?}",
            backends
        );
    }
}

/// Verifies that with_backends sets the correct backend mask.
///
/// Contract: The backends returned by backends() match what was passed to with_backends().
#[test]
#[cfg(not(target_arch = "wasm32"))]
fn test_instance_backends_matches_construction() {
    let backends_requested = wgpu::Backends::VULKAN | wgpu::Backends::GL;
    let instance = TrinityInstance::with_backends(backends_requested);

    let backends_actual = instance.backends();
    assert_eq!(backends_actual, backends_requested);
}

// =============================================================================
// 3. Inner wgpu::Instance Access
// =============================================================================

/// Verifies that inner() provides access to the underlying wgpu::Instance.
///
/// Contract: instance.inner() returns a reference to wgpu::Instance that can be
/// used for operations requiring direct wgpu access.
#[test]
fn test_instance_inner_returns_wgpu_instance() {
    let instance = TrinityInstance::new();

    // inner() should return a usable wgpu::Instance reference
    let inner: &wgpu::Instance = instance.inner();

    // Verify it's usable by calling a wgpu::Instance method
    // enumerate_adapters on the raw instance should work
    let _adapters: Vec<wgpu::Adapter> = inner.enumerate_adapters(wgpu::Backends::all());
}

// =============================================================================
// 4. Adapter Enumeration
// =============================================================================

/// Verifies that enumerate_adapters returns a vector of adapters.
///
/// Contract: enumerate_adapters() returns Vec<wgpu::Adapter> containing all
/// adapters available for the configured backends.
#[test]
fn test_enumerate_adapters_returns_vec() {
    let instance = TrinityInstance::new();
    let adapters = instance.enumerate_adapters();

    // Should return a Vec (may be empty on headless CI without GPU)
    // We just verify the type is correct and operation doesn't panic
    let _: Vec<wgpu::Adapter> = adapters;
}

/// Verifies that enumerate_adapters can find at least one adapter on a typical system.
///
/// Note: This test may be skipped on headless CI environments without GPU.
#[test]
fn test_enumerate_adapters_finds_adapters_when_available() {
    let instance = TrinityInstance::new();
    let adapters = instance.enumerate_adapters();

    // On a machine with a GPU, we expect at least one adapter
    // This is a soft check - CI may not have GPUs
    if !adapters.is_empty() {
        // Each adapter should have valid info
        for adapter in &adapters {
            let info = adapter.get_info();
            // Name should be non-empty
            assert!(!info.name.is_empty(), "Adapter name should not be empty");
        }
    }
}

/// Verifies that adapters from enumerate_adapters have valid backend info.
///
/// Contract: Each enumerated adapter reports a backend that is within the
/// configured backend set.
#[test]
fn test_enumerate_adapters_reports_valid_backends() {
    let instance = TrinityInstance::new();
    let configured_backends = instance.backends();
    let adapters = instance.enumerate_adapters();

    for adapter in &adapters {
        let info = adapter.get_info();
        let adapter_backend = info.backend;

        // The adapter's backend should be one we asked for
        let backend_flag = match adapter_backend {
            wgpu::Backend::Vulkan => wgpu::Backends::VULKAN,
            wgpu::Backend::Metal => wgpu::Backends::METAL,
            wgpu::Backend::Dx12 => wgpu::Backends::DX12,
            wgpu::Backend::Gl => wgpu::Backends::GL,
            wgpu::Backend::BrowserWebGpu => wgpu::Backends::BROWSER_WEBGPU,
            wgpu::Backend::Empty => continue, // Skip null backend
        };

        assert!(
            configured_backends.contains(backend_flag),
            "Adapter {} reports backend {:?} which is not in configured backends {:?}",
            info.name,
            adapter_backend,
            configured_backends
        );
    }
}

// =============================================================================
// 5. Multiple Instance Creation
// =============================================================================

/// Verifies that multiple TrinityInstance objects can coexist.
///
/// Contract: Creating multiple instances should work without interference.
#[test]
fn test_multiple_instances_coexist() {
    let instance1 = TrinityInstance::new();
    let instance2 = TrinityInstance::new();

    // Both should be independently usable
    let adapters1 = instance1.enumerate_adapters();
    let adapters2 = instance2.enumerate_adapters();

    // They should see the same adapters (same system)
    assert_eq!(
        adapters1.len(),
        adapters2.len(),
        "Multiple instances should see the same adapters"
    );
}

/// Verifies that instances with different backends can coexist.
///
/// Contract: Instances configured for different backends are independent.
#[test]
#[cfg(not(target_arch = "wasm32"))]
fn test_instances_with_different_backends_coexist() {
    let instance_vulkan = TrinityInstance::with_backends(wgpu::Backends::VULKAN);
    let instance_all = TrinityInstance::with_backends(wgpu::Backends::all());

    // Each should enumerate according to its configuration
    let _ = instance_vulkan.enumerate_adapters();
    let _ = instance_all.enumerate_adapters();

    // They should report their respective backends
    assert_eq!(instance_vulkan.backends(), wgpu::Backends::VULKAN);
    assert_eq!(instance_all.backends(), wgpu::Backends::all());
}

// =============================================================================
// 6. Edge Cases
// =============================================================================

/// Verifies behavior when no adapters are available for requested backend.
///
/// Contract: enumerate_adapters() returns empty Vec when no adapters match,
/// rather than panicking.
#[test]
#[cfg(not(target_arch = "wasm32"))]
fn test_enumerate_adapters_empty_when_no_matching_backend() {
    // Request a backend that's unlikely to be present everywhere
    // BROWSER_WEBGPU on native should return empty
    let instance = TrinityInstance::with_backends(wgpu::Backends::BROWSER_WEBGPU);
    let adapters = instance.enumerate_adapters();

    // Should return empty, not panic
    assert!(
        adapters.is_empty(),
        "BROWSER_WEBGPU should have no adapters on native platform"
    );
}

/// Verifies that requesting Backends::empty() creates a valid but adapter-less instance.
///
/// Contract: Even with no backends, the instance should be constructible.
#[test]
fn test_instance_with_empty_backends() {
    let instance = TrinityInstance::with_backends(wgpu::Backends::empty());
    let adapters = instance.enumerate_adapters();

    // No backends means no adapters
    assert!(
        adapters.is_empty(),
        "Empty backends should yield no adapters"
    );

    // But backends() should report what was requested
    assert_eq!(instance.backends(), wgpu::Backends::empty());
}

// =============================================================================
// 7. Platform-Specific Behavior
// =============================================================================

/// Verifies that on non-WASM platforms, PRIMARY backends are used by default.
///
/// Contract: Default instance on desktop uses Backends::PRIMARY.
#[test]
#[cfg(not(target_arch = "wasm32"))]
fn test_default_backends_primary_on_desktop() {
    let instance = TrinityInstance::new();
    let backends = instance.backends();

    // PRIMARY is Vulkan | Metal | DX12
    // At least one should be set
    let has_primary = backends.contains(wgpu::Backends::VULKAN)
        || backends.contains(wgpu::Backends::METAL)
        || backends.contains(wgpu::Backends::DX12);

    assert!(has_primary, "Desktop should use PRIMARY backends by default");
}

// =============================================================================
// 8. Thread Safety
// =============================================================================

/// Verifies that TrinityInstance is Send + Sync.
///
/// Contract: Instance should be usable from multiple threads.
#[test]
fn test_instance_is_send_sync() {
    fn assert_send<T: Send>() {}
    fn assert_sync<T: Sync>() {}

    assert_send::<TrinityInstance>();
    assert_sync::<TrinityInstance>();
}

/// Verifies that enumerate_adapters can be called from multiple threads.
///
/// Contract: Concurrent enumeration should not cause data races.
#[test]
fn test_concurrent_enumeration() {
    use std::sync::Arc;
    use std::thread;

    let instance = Arc::new(TrinityInstance::new());

    let handles: Vec<_> = (0..4)
        .map(|_| {
            let inst = Arc::clone(&instance);
            thread::spawn(move || {
                let adapters = inst.enumerate_adapters();
                adapters.len()
            })
        })
        .collect();

    let results: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();

    // All threads should see the same adapter count
    let first = results[0];
    for count in &results {
        assert_eq!(*count, first, "Concurrent enumeration should be consistent");
    }
}

// =============================================================================
// 9. Debug/Display (if implemented)
// =============================================================================

/// Verifies that TrinityInstance implements Debug for logging.
///
/// Contract: Instance should be printable for debugging purposes.
#[test]
fn test_instance_debug_impl() {
    let instance = TrinityInstance::new();

    // Should compile if Debug is implemented
    let debug_str = format!("{:?}", instance);

    // Debug output should contain some identifying information
    assert!(!debug_str.is_empty(), "Debug output should not be empty");
}
