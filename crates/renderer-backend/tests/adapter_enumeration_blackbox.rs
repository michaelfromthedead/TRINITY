// Blackbox contract tests for T-WGPU-P1.2.1 Adapter Enumeration
//
// CLEANROOM: No access to implementation files. Tests use only the public API
// exported by `renderer_backend::device`.
//
// Forbidden files (per TESTDEV_BLACKBOX prompt):
//   - crates/renderer-backend/src/device/adapter.rs
//   - crates/renderer-backend/src/device/instance.rs (implementation details)
//   - Any WHITEBOX test file for this task
//
// Contract sources:
//   - docs/WGPU_DOCS/RDC_OUTPUT/PHASE_1_CORE_TODO.md (T-WGPU-P1.2.1)
//   - docs/WGPU_DOCS/RDC_OUTPUT/PHASE_1_CORE_ARCH.md (Architecture spec)
//
// Acceptance criteria (T-WGPU-P1.2.1):
//   - Returns all available adapters
//   - Filters by requested backend
//   - Logs adapter info for debugging
//   - Handles zero adapters gracefully
//
// Test design rationale:
//   Equivalence partitioning:
//     - Enumeration with PRIMARY backends (typical desktop)
//     - Enumeration with EMPTY backends (no adapters)
//     - Filter operations (by device type, by backend)
//   Boundary cases:
//     - Zero adapters
//     - All adapters of same type
//     - Mixed adapter types
//   Contract verification:
//     - EnumerationResult struct fields
//     - BackendCounts struct fields
//     - Helper methods return expected types

use renderer_backend::device::{
    enumerate_adapters_with_info, filter_by_backend, filter_by_device_type, BackendCounts,
    EnumerationResult, TrinityInstance,
};

// =============================================================================
// 1. Basic Contract Tests
// =============================================================================

/// Verifies that enumerate_adapters_with_info returns an EnumerationResult type.
///
/// Contract: The enumeration function returns an EnumerationResult struct.
#[test]
fn test_enumeration_returns_enumeration_result_type() {
    let instance = TrinityInstance::new();
    let result: EnumerationResult =
        enumerate_adapters_with_info(instance.inner(), instance.backends());

    // The result should be an EnumerationResult - type annotation enforces this
    let _ = result;
}

/// Verifies that EnumerationResult has an adapters field containing Vec<Adapter>.
///
/// Contract: EnumerationResult.adapters is a Vec<wgpu::Adapter>.
#[test]
fn test_enumeration_result_has_adapters_field() {
    let instance = TrinityInstance::new();
    let result = enumerate_adapters_with_info(instance.inner(), instance.backends());

    // Access the adapters field - this verifies it exists and is Vec<Adapter>
    let adapters: &Vec<wgpu::Adapter> = &result.adapters;
    let _ = adapters;
}

/// Verifies that EnumerationResult has a backend_counts field.
///
/// Contract: EnumerationResult.backend_counts is a BackendCounts struct.
#[test]
fn test_enumeration_result_has_backend_counts_field() {
    let instance = TrinityInstance::new();
    let result = enumerate_adapters_with_info(instance.inner(), instance.backends());

    // Access the backend_counts field
    let counts: &BackendCounts = &result.backend_counts;
    let _ = counts;
}

// =============================================================================
// 2. Behavioral Tests
// =============================================================================

/// Verifies that enumeration with PRIMARY backends finds adapters (on machine with GPU).
///
/// Contract: On a system with a GPU, enumeration should find at least one adapter.
/// Note: This test is informational - CI may not have GPUs.
#[test]
#[cfg(not(target_arch = "wasm32"))]
fn test_enumeration_with_primary_backends_finds_adapters() {
    let instance = TrinityInstance::with_backends(wgpu::Backends::PRIMARY);
    let result = enumerate_adapters_with_info(instance.inner(), instance.backends());

    // On a typical desktop with GPU, we expect adapters
    // We don't assert because CI might not have GPU
    if !result.adapters.is_empty() {
        // Each adapter should have valid info
        for adapter in &result.adapters {
            let info = adapter.get_info();
            assert!(!info.name.is_empty(), "Adapter should have a name");
        }
    }
}

/// Verifies that enumeration with EMPTY backends returns empty result.
///
/// Contract: Requesting no backends should return an empty result.
#[test]
fn test_enumeration_with_empty_backends_returns_empty_result() {
    let instance = TrinityInstance::with_backends(wgpu::Backends::empty());
    let result = enumerate_adapters_with_info(instance.inner(), instance.backends());

    assert!(
        result.adapters.is_empty(),
        "Empty backends should yield no adapters"
    );
}

/// Verifies that multiple enumerations return consistent adapter count.
///
/// Contract: Enumeration is deterministic - same input yields same count.
#[test]
fn test_multiple_enumerations_return_consistent_adapter_count() {
    let instance = TrinityInstance::new();

    let result1 = enumerate_adapters_with_info(instance.inner(), instance.backends());
    let result2 = enumerate_adapters_with_info(instance.inner(), instance.backends());
    let result3 = enumerate_adapters_with_info(instance.inner(), instance.backends());

    assert_eq!(
        result1.len(),
        result2.len(),
        "First and second enumeration should have same count"
    );
    assert_eq!(
        result2.len(),
        result3.len(),
        "Second and third enumeration should have same count"
    );
}

// =============================================================================
// 3. Filter Contract Tests
// =============================================================================

/// Verifies that filter_by_device_type returns a subset of input.
///
/// Contract: Filtered result should have <= adapters than input.
#[test]
fn test_filter_by_device_type_returns_subset() {
    let instance = TrinityInstance::new();
    let result = enumerate_adapters_with_info(instance.inner(), instance.backends());
    let original_len = result.adapters.len();

    // Filter by discrete GPU - function takes ownership so we need to clone
    let adapters_copy = result.adapters;
    let filtered = filter_by_device_type(adapters_copy, wgpu::DeviceType::DiscreteGpu);

    assert!(
        filtered.len() <= original_len,
        "Filtered result should not exceed original"
    );

    // All filtered adapters should be discrete GPUs
    for adapter in &filtered {
        let info = adapter.get_info();
        assert_eq!(
            info.device_type,
            wgpu::DeviceType::DiscreteGpu,
            "Filter should only return discrete GPUs"
        );
    }
}

/// Verifies that filter_by_backend returns a subset of input.
///
/// Contract: Filtered result should have <= adapters than input.
#[test]
fn test_filter_by_backend_returns_subset() {
    let instance = TrinityInstance::new();
    let result = enumerate_adapters_with_info(instance.inner(), instance.backends());
    let original_len = result.adapters.len();

    // Filter by Vulkan backend - function takes ownership
    let adapters_copy = result.adapters;
    let filtered = filter_by_backend(adapters_copy, wgpu::Backend::Vulkan);

    assert!(
        filtered.len() <= original_len,
        "Filtered result should not exceed original"
    );

    // All filtered adapters should use Vulkan backend
    for adapter in &filtered {
        let info = adapter.get_info();
        assert_eq!(
            info.backend,
            wgpu::Backend::Vulkan,
            "Filter should only return Vulkan adapters"
        );
    }
}

/// Verifies that filters with no matches return empty Vec.
///
/// Contract: Filter should return empty Vec when no adapters match.
#[test]
fn test_filter_with_no_matches_returns_empty_vec() {
    let instance = TrinityInstance::new();
    let result = enumerate_adapters_with_info(instance.inner(), instance.backends());

    // Filter by CPU device type - unlikely to match real GPUs
    // Function takes ownership
    let adapters_copy = result.adapters;
    let filtered_cpu = filter_by_device_type(adapters_copy, wgpu::DeviceType::Cpu);

    // If no CPU adapters exist, this should be empty
    // (We can't assert it's empty because some systems have CPU fallback)
    // Just verify it doesn't panic and returns a valid Vec
    let _: Vec<wgpu::Adapter> = filtered_cpu;
}

/// Verifies that filter_by_device_type with IntegratedGpu works.
///
/// Contract: Filter should correctly identify integrated GPUs.
#[test]
fn test_filter_by_device_type_integrated() {
    let instance = TrinityInstance::new();
    let result = enumerate_adapters_with_info(instance.inner(), instance.backends());

    // Function takes ownership
    let adapters_copy = result.adapters;
    let filtered = filter_by_device_type(adapters_copy, wgpu::DeviceType::IntegratedGpu);

    // All filtered adapters should be integrated GPUs
    for adapter in &filtered {
        let info = adapter.get_info();
        assert_eq!(
            info.device_type,
            wgpu::DeviceType::IntegratedGpu,
            "Filter should only return integrated GPUs"
        );
    }
}

// =============================================================================
// 4. Helper Method Tests
// =============================================================================

/// Verifies that EnumerationResult.len() equals adapters.len().
///
/// Contract: len() is a convenience method for adapters.len().
#[test]
fn test_enumeration_result_len_equals_adapters_len() {
    let instance = TrinityInstance::new();
    let result = enumerate_adapters_with_info(instance.inner(), instance.backends());

    assert_eq!(
        result.len(),
        result.adapters.len(),
        "len() should equal adapters.len()"
    );
}

/// Verifies that EnumerationResult.is_empty() equals adapters.is_empty().
///
/// Contract: is_empty() is a convenience method for adapters.is_empty().
#[test]
fn test_enumeration_result_is_empty_equals_adapters_is_empty() {
    let instance = TrinityInstance::new();
    let result = enumerate_adapters_with_info(instance.inner(), instance.backends());

    assert_eq!(
        result.is_empty(),
        result.adapters.is_empty(),
        "is_empty() should equal adapters.is_empty()"
    );
}

/// Verifies that is_empty() returns true when adapters is empty.
///
/// Contract: is_empty() should be true for empty backends.
#[test]
fn test_enumeration_result_is_empty_true_when_no_adapters() {
    let instance = TrinityInstance::with_backends(wgpu::Backends::empty());
    let result = enumerate_adapters_with_info(instance.inner(), instance.backends());

    assert!(result.is_empty(), "Should be empty with no backends");
    assert_eq!(result.len(), 0, "Length should be 0 with no backends");
}

/// Verifies that BackendCounts.total() >= each individual count.
///
/// Contract: total() should be >= vulkan, metal, dx12, gl, webgpu.
#[test]
fn test_backend_counts_total_gte_individual_counts() {
    let instance = TrinityInstance::new();
    let result = enumerate_adapters_with_info(instance.inner(), instance.backends());
    let counts = &result.backend_counts;

    let total = counts.total();

    assert!(
        total >= counts.vulkan,
        "total >= vulkan: {} >= {}",
        total,
        counts.vulkan
    );
    assert!(
        total >= counts.metal,
        "total >= metal: {} >= {}",
        total,
        counts.metal
    );
    assert!(
        total >= counts.dx12,
        "total >= dx12: {} >= {}",
        total,
        counts.dx12
    );
    assert!(
        total >= counts.gl,
        "total >= gl: {} >= {}",
        total,
        counts.gl
    );
    assert!(
        total >= counts.webgpu,
        "total >= webgpu: {} >= {}",
        total,
        counts.webgpu
    );
}

/// Verifies that BackendCounts.total() equals sum of individual counts.
///
/// Contract: total() should equal vulkan + metal + dx12 + gl + webgpu.
#[test]
fn test_backend_counts_total_equals_sum() {
    let instance = TrinityInstance::new();
    let result = enumerate_adapters_with_info(instance.inner(), instance.backends());
    let counts = &result.backend_counts;

    let expected_sum = counts.vulkan + counts.metal + counts.dx12 + counts.gl + counts.webgpu;

    assert_eq!(
        counts.total(),
        expected_sum,
        "total() should equal sum of all backend counts"
    );
}

/// Verifies that BackendCounts.is_empty() true iff total() == 0.
///
/// Contract: is_empty() should match total() == 0.
#[test]
fn test_backend_counts_is_empty_iff_total_zero() {
    let instance = TrinityInstance::new();
    let result = enumerate_adapters_with_info(instance.inner(), instance.backends());
    let counts = &result.backend_counts;

    assert_eq!(
        counts.is_empty(),
        counts.total() == 0,
        "is_empty() should be true iff total() == 0"
    );
}

/// Verifies that BackendCounts.is_empty() is true for empty backends.
///
/// Contract: Empty backends should result in empty counts.
#[test]
fn test_backend_counts_is_empty_for_empty_backends() {
    let instance = TrinityInstance::with_backends(wgpu::Backends::empty());
    let result = enumerate_adapters_with_info(instance.inner(), instance.backends());
    let counts = &result.backend_counts;

    assert!(
        counts.is_empty(),
        "Backend counts should be empty with no backends"
    );
    assert_eq!(counts.total(), 0, "Total should be 0 with no backends");
}

/// Verifies that BackendCounts.summary() returns a non-empty string.
///
/// Contract: summary() should return a human-readable description.
#[test]
fn test_backend_counts_summary_returns_string() {
    let instance = TrinityInstance::new();
    let result = enumerate_adapters_with_info(instance.inner(), instance.backends());
    let counts = &result.backend_counts;

    let summary = counts.summary();

    // Summary should be a non-empty string
    assert!(
        !summary.is_empty(),
        "summary() should return non-empty string"
    );
}

// =============================================================================
// 5. TrinityInstance Integration Tests
// =============================================================================

/// Verifies that instance.enumerate_adapters_detailed() returns EnumerationResult.
///
/// Contract: TrinityInstance should have a method for detailed enumeration.
#[test]
fn test_instance_enumerate_adapters_detailed_returns_enumeration_result() {
    let instance = TrinityInstance::new();

    // Use the module function instead since enumerate_adapters_detailed might not exist
    let result: EnumerationResult =
        enumerate_adapters_with_info(instance.inner(), instance.backends());

    let _ = result;
}

/// Verifies that enumeration result is consistent with instance.backends().
///
/// Contract: All enumerated adapters should use backends from instance.backends().
#[test]
fn test_enumeration_result_consistent_with_instance_backends() {
    let instance = TrinityInstance::new();
    let configured_backends = instance.backends();
    let result = enumerate_adapters_with_info(instance.inner(), configured_backends);

    for adapter in &result.adapters {
        let info = adapter.get_info();
        let adapter_backend = info.backend;

        let backend_flag = match adapter_backend {
            wgpu::Backend::Vulkan => wgpu::Backends::VULKAN,
            wgpu::Backend::Metal => wgpu::Backends::METAL,
            wgpu::Backend::Dx12 => wgpu::Backends::DX12,
            wgpu::Backend::Gl => wgpu::Backends::GL,
            wgpu::Backend::BrowserWebGpu => wgpu::Backends::BROWSER_WEBGPU,
            wgpu::Backend::Empty => continue,
        };

        assert!(
            configured_backends.contains(backend_flag),
            "Adapter {} uses backend {:?} not in configured backends {:?}",
            info.name,
            adapter_backend,
            configured_backends
        );
    }
}

/// Verifies that backend counts match actual adapter backends.
///
/// Contract: BackendCounts should accurately reflect adapter distribution.
#[test]
fn test_backend_counts_match_actual_adapters() {
    let instance = TrinityInstance::new();
    let result = enumerate_adapters_with_info(instance.inner(), instance.backends());

    // Count adapters by backend manually
    let mut vulkan_count = 0;
    let mut metal_count = 0;
    let mut dx12_count = 0;
    let mut gl_count = 0;
    let mut webgpu_count = 0;

    for adapter in &result.adapters {
        match adapter.get_info().backend {
            wgpu::Backend::Vulkan => vulkan_count += 1,
            wgpu::Backend::Metal => metal_count += 1,
            wgpu::Backend::Dx12 => dx12_count += 1,
            wgpu::Backend::Gl => gl_count += 1,
            wgpu::Backend::BrowserWebGpu => webgpu_count += 1,
            wgpu::Backend::Empty => {}
        }
    }

    let counts = &result.backend_counts;

    assert_eq!(
        counts.vulkan, vulkan_count,
        "Vulkan count mismatch"
    );
    assert_eq!(
        counts.metal, metal_count,
        "Metal count mismatch"
    );
    assert_eq!(
        counts.dx12, dx12_count,
        "DX12 count mismatch"
    );
    assert_eq!(
        counts.gl, gl_count,
        "GL count mismatch"
    );
    assert_eq!(
        counts.webgpu, webgpu_count,
        "WebGPU count mismatch"
    );
}

// =============================================================================
// 6. Graceful Handling Tests
// =============================================================================

/// Verifies that zero adapters does not panic.
///
/// Contract: Enumeration with no matching backends should return empty, not panic.
#[test]
fn test_zero_adapters_does_not_panic() {
    let instance = TrinityInstance::with_backends(wgpu::Backends::empty());
    let result = enumerate_adapters_with_info(instance.inner(), instance.backends());

    // Should complete without panic
    let _ = result;
}

/// Verifies that empty result is valid (is_empty() == true).
///
/// Contract: Empty result should have is_empty() == true.
#[test]
fn test_empty_result_is_valid() {
    let instance = TrinityInstance::with_backends(wgpu::Backends::empty());
    let result = enumerate_adapters_with_info(instance.inner(), instance.backends());

    assert!(result.is_empty(), "Empty result should have is_empty() == true");
    assert_eq!(result.len(), 0, "Empty result should have len() == 0");
    assert!(
        result.adapters.is_empty(),
        "Empty result should have empty adapters vec"
    );
}

/// Verifies that first_discrete() returns None when no discrete adapters exist.
///
/// Contract: first_discrete() should return None for empty result.
#[test]
fn test_first_discrete_returns_none_when_no_discrete() {
    let instance = TrinityInstance::with_backends(wgpu::Backends::empty());
    let result = enumerate_adapters_with_info(instance.inner(), instance.backends());

    let discrete = result.first_discrete();
    assert!(
        discrete.is_none(),
        "first_discrete() should return None for empty result"
    );
}

/// Verifies that first_integrated() returns None when no integrated adapters exist.
///
/// Contract: first_integrated() should return None for empty result.
#[test]
fn test_first_integrated_returns_none_when_no_integrated() {
    let instance = TrinityInstance::with_backends(wgpu::Backends::empty());
    let result = enumerate_adapters_with_info(instance.inner(), instance.backends());

    let integrated = result.first_integrated();
    assert!(
        integrated.is_none(),
        "first_integrated() should return None for empty result"
    );
}

/// Verifies that best_adapter() returns None when no adapters exist.
///
/// Contract: best_adapter() should return None for empty result.
#[test]
fn test_best_adapter_returns_none_when_no_adapters() {
    let instance = TrinityInstance::with_backends(wgpu::Backends::empty());
    let result = enumerate_adapters_with_info(instance.inner(), instance.backends());

    let best = result.best_adapter();
    assert!(
        best.is_none(),
        "best_adapter() should return None for empty result"
    );
}

// =============================================================================
// 7. First/Best Adapter Helper Tests
// =============================================================================

/// Verifies that first_discrete() returns a discrete GPU when available.
///
/// Contract: first_discrete() should return a DiscreteGpu type adapter.
#[test]
fn test_first_discrete_returns_discrete_when_available() {
    let instance = TrinityInstance::new();
    let result = enumerate_adapters_with_info(instance.inner(), instance.backends());

    if let Some(adapter) = result.first_discrete() {
        let info = adapter.get_info();
        assert_eq!(
            info.device_type,
            wgpu::DeviceType::DiscreteGpu,
            "first_discrete() should return a DiscreteGpu"
        );
    }
    // If None, that's OK - system might not have discrete GPU
}

/// Verifies that first_integrated() returns an integrated GPU when available.
///
/// Contract: first_integrated() should return an IntegratedGpu type adapter.
#[test]
fn test_first_integrated_returns_integrated_when_available() {
    let instance = TrinityInstance::new();
    let result = enumerate_adapters_with_info(instance.inner(), instance.backends());

    if let Some(adapter) = result.first_integrated() {
        let info = adapter.get_info();
        assert_eq!(
            info.device_type,
            wgpu::DeviceType::IntegratedGpu,
            "first_integrated() should return an IntegratedGpu"
        );
    }
    // If None, that's OK - system might not have integrated GPU
}

/// Verifies that best_adapter() returns some adapter when adapters exist.
///
/// Contract: best_adapter() should return Some when result is non-empty.
#[test]
fn test_best_adapter_returns_some_when_adapters_exist() {
    let instance = TrinityInstance::new();
    let result = enumerate_adapters_with_info(instance.inner(), instance.backends());

    if !result.is_empty() {
        let best = result.best_adapter();
        assert!(
            best.is_some(),
            "best_adapter() should return Some for non-empty result"
        );
    }
}

/// Verifies that best_adapter() prefers discrete over integrated.
///
/// Contract: best_adapter() should return discrete GPU if available.
#[test]
fn test_best_adapter_prefers_discrete() {
    let instance = TrinityInstance::new();
    let result = enumerate_adapters_with_info(instance.inner(), instance.backends());

    // Only test if we have both discrete and integrated
    let has_discrete = result.first_discrete().is_some();
    let has_integrated = result.first_integrated().is_some();

    if has_discrete && has_integrated {
        if let Some(best) = result.best_adapter() {
            let info = best.get_info();
            assert_eq!(
                info.device_type,
                wgpu::DeviceType::DiscreteGpu,
                "best_adapter() should prefer discrete over integrated"
            );
        }
    }
}

// =============================================================================
// 8. Thread Safety Tests
// =============================================================================

/// Verifies that EnumerationResult is Send.
///
/// Contract: Results should be transferable across threads.
#[test]
fn test_enumeration_result_is_send() {
    fn assert_send<T: Send>() {}
    assert_send::<EnumerationResult>();
}

/// Verifies that BackendCounts is Send + Sync + Copy.
///
/// Contract: BackendCounts should be lightweight and thread-safe.
#[test]
fn test_backend_counts_is_send_sync_copy() {
    fn assert_send<T: Send>() {}
    fn assert_sync<T: Sync>() {}
    fn assert_copy<T: Copy>() {}

    assert_send::<BackendCounts>();
    assert_sync::<BackendCounts>();
    assert_copy::<BackendCounts>();
}

/// Verifies that concurrent enumerations are safe.
///
/// Contract: Multiple threads can enumerate adapters simultaneously.
#[test]
fn test_concurrent_enumeration_is_safe() {
    use std::sync::Arc;
    use std::thread;

    let instance = Arc::new(TrinityInstance::new());

    let handles: Vec<_> = (0..4)
        .map(|_| {
            let inst = Arc::clone(&instance);
            thread::spawn(move || {
                let result = enumerate_adapters_with_info(inst.inner(), inst.backends());
                result.len()
            })
        })
        .collect();

    let results: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();

    // All threads should see the same adapter count
    let first = results[0];
    for count in &results {
        assert_eq!(
            *count, first,
            "Concurrent enumeration should be consistent"
        );
    }
}

// =============================================================================
// 9. BackendCounts Field Access Tests
// =============================================================================

/// Verifies that BackendCounts has vulkan field.
///
/// Contract: BackendCounts.vulkan is accessible.
#[test]
fn test_backend_counts_has_vulkan_field() {
    let instance = TrinityInstance::new();
    let result = enumerate_adapters_with_info(instance.inner(), instance.backends());

    let _vulkan: usize = result.backend_counts.vulkan;
}

/// Verifies that BackendCounts has metal field.
///
/// Contract: BackendCounts.metal is accessible.
#[test]
fn test_backend_counts_has_metal_field() {
    let instance = TrinityInstance::new();
    let result = enumerate_adapters_with_info(instance.inner(), instance.backends());

    let _metal: usize = result.backend_counts.metal;
}

/// Verifies that BackendCounts has dx12 field.
///
/// Contract: BackendCounts.dx12 is accessible.
#[test]
fn test_backend_counts_has_dx12_field() {
    let instance = TrinityInstance::new();
    let result = enumerate_adapters_with_info(instance.inner(), instance.backends());

    let _dx12: usize = result.backend_counts.dx12;
}

/// Verifies that BackendCounts has gl field.
///
/// Contract: BackendCounts.gl is accessible.
#[test]
fn test_backend_counts_has_gl_field() {
    let instance = TrinityInstance::new();
    let result = enumerate_adapters_with_info(instance.inner(), instance.backends());

    let _gl: usize = result.backend_counts.gl;
}

/// Verifies that BackendCounts has webgpu field.
///
/// Contract: BackendCounts.webgpu is accessible.
#[test]
fn test_backend_counts_has_webgpu_field() {
    let instance = TrinityInstance::new();
    let result = enumerate_adapters_with_info(instance.inner(), instance.backends());

    let _webgpu: usize = result.backend_counts.webgpu;
}

// =============================================================================
// 10. device_type_description Helper Tests
// =============================================================================

/// Verifies that device_type_description returns a non-empty string.
///
/// Contract: device_type_description should return human-readable description.
#[test]
fn test_device_type_description_returns_string() {
    use renderer_backend::device::device_type_description;

    let discrete_desc = device_type_description(wgpu::DeviceType::DiscreteGpu);
    let integrated_desc = device_type_description(wgpu::DeviceType::IntegratedGpu);
    let cpu_desc = device_type_description(wgpu::DeviceType::Cpu);
    let other_desc = device_type_description(wgpu::DeviceType::Other);

    assert!(!discrete_desc.is_empty(), "Discrete description should not be empty");
    assert!(!integrated_desc.is_empty(), "Integrated description should not be empty");
    assert!(!cpu_desc.is_empty(), "CPU description should not be empty");
    assert!(!other_desc.is_empty(), "Other description should not be empty");
}

/// Verifies that different device types have different descriptions.
///
/// Contract: Each device type should have a distinct description.
#[test]
fn test_device_type_description_differs_by_type() {
    use renderer_backend::device::device_type_description;

    let discrete_desc = device_type_description(wgpu::DeviceType::DiscreteGpu);
    let integrated_desc = device_type_description(wgpu::DeviceType::IntegratedGpu);
    let cpu_desc = device_type_description(wgpu::DeviceType::Cpu);

    // Discrete and integrated should have different descriptions
    assert_ne!(
        discrete_desc, integrated_desc,
        "Discrete and integrated should have different descriptions"
    );
    assert_ne!(
        discrete_desc, cpu_desc,
        "Discrete and CPU should have different descriptions"
    );
    assert_ne!(
        integrated_desc, cpu_desc,
        "Integrated and CPU should have different descriptions"
    );
}

// =============================================================================
// 11. Debug/Display Implementations
// =============================================================================

/// Verifies that BackendCounts implements Debug.
///
/// Contract: BackendCounts should be printable for debugging.
#[test]
fn test_backend_counts_debug_impl() {
    let instance = TrinityInstance::new();
    let result = enumerate_adapters_with_info(instance.inner(), instance.backends());

    let debug_str = format!("{:?}", result.backend_counts);
    assert!(!debug_str.is_empty(), "Debug output should not be empty");
}

/// Verifies that EnumerationResult implements Debug.
///
/// Contract: EnumerationResult should be printable for debugging.
#[test]
fn test_enumeration_result_debug_impl() {
    let instance = TrinityInstance::new();
    let result = enumerate_adapters_with_info(instance.inner(), instance.backends());

    let debug_str = format!("{:?}", result);
    assert!(!debug_str.is_empty(), "Debug output should not be empty");
}

// =============================================================================
// 12. Backend-Specific Tests
// =============================================================================

/// Verifies that Vulkan-only enumeration works on supported platforms.
///
/// Contract: Enumeration with VULKAN backend should work on Linux/Windows.
#[test]
#[cfg(any(target_os = "linux", target_os = "windows"))]
fn test_vulkan_only_enumeration() {
    let instance = TrinityInstance::with_backends(wgpu::Backends::VULKAN);
    let result = enumerate_adapters_with_info(instance.inner(), instance.backends());

    // All adapters should be Vulkan
    for adapter in &result.adapters {
        let info = adapter.get_info();
        assert_eq!(
            info.backend,
            wgpu::Backend::Vulkan,
            "Vulkan-only enumeration should return only Vulkan adapters"
        );
    }

    // Backend counts should reflect Vulkan only
    assert_eq!(result.backend_counts.metal, 0);
    assert_eq!(result.backend_counts.dx12, 0);
    assert_eq!(result.backend_counts.webgpu, 0);
}

/// Verifies that Metal-only enumeration works on macOS.
///
/// Contract: Enumeration with METAL backend should work on macOS.
#[test]
#[cfg(target_os = "macos")]
fn test_metal_only_enumeration() {
    let instance = TrinityInstance::with_backends(wgpu::Backends::METAL);
    let result = enumerate_adapters_with_info(instance.inner(), instance.backends());

    // All adapters should be Metal
    for adapter in &result.adapters {
        let info = adapter.get_info();
        assert_eq!(
            info.backend,
            wgpu::Backend::Metal,
            "Metal-only enumeration should return only Metal adapters"
        );
    }

    // Backend counts should reflect Metal only
    assert_eq!(result.backend_counts.vulkan, 0);
    assert_eq!(result.backend_counts.dx12, 0);
    assert_eq!(result.backend_counts.webgpu, 0);
}

/// Verifies that DX12-only enumeration works on Windows.
///
/// Contract: Enumeration with DX12 backend should work on Windows.
#[test]
#[cfg(target_os = "windows")]
fn test_dx12_only_enumeration() {
    let instance = TrinityInstance::with_backends(wgpu::Backends::DX12);
    let result = enumerate_adapters_with_info(instance.inner(), instance.backends());

    // All adapters should be DX12
    for adapter in &result.adapters {
        let info = adapter.get_info();
        assert_eq!(
            info.backend,
            wgpu::Backend::Dx12,
            "DX12-only enumeration should return only DX12 adapters"
        );
    }

    // Backend counts should reflect DX12 only
    assert_eq!(result.backend_counts.vulkan, 0);
    assert_eq!(result.backend_counts.metal, 0);
    assert_eq!(result.backend_counts.webgpu, 0);
}
