// WHITEBOX tests for T-WGPU-P1.2.1 (Adapter Enumeration)
//
// WHITEBOX discipline: These tests have FULL ACCESS to the implementation.
// They exercise internal code paths, branch conditions, and edge cases
// that are not visible through the public contract alone.
//
// Implementation under test: crates/renderer-backend/src/device/adapter.rs
//   - enumerate_adapters_with_info()
//   - filter_by_device_type()
//   - filter_by_backend()
//   - EnumerationResult methods (is_empty, len, first_discrete, first_integrated, best_adapter)
//   - BackendCounts methods (total, is_empty, summary)
//   - device_type_description()
//
// Also tests integration with: crates/renderer-backend/src/device/instance.rs
//   - TrinityInstance::enumerate_adapters_detailed()
//
// WHITEBOX coverage plan:
//   - Path A: enumerate_adapters_with_info with PRIMARY backends
//   - Path B: enumerate_adapters_with_info with specific backend (VULKAN)
//   - Path C: enumerate_adapters_with_info with EMPTY backends (returns empty)
//   - Path D: EnumerationResult::is_empty() returns true when no adapters
//   - Path E: EnumerationResult::is_empty() returns false when adapters exist
//   - Path F: EnumerationResult::len() returns correct count
//   - Path G: EnumerationResult::first_discrete() finds discrete GPU
//   - Path H: EnumerationResult::first_integrated() finds integrated GPU
//   - Path I: EnumerationResult::best_adapter() priority order (discrete > integrated > virtual > cpu > other)
//   - Path J: BackendCounts::total() sums all counts
//   - Path K: BackendCounts::is_empty() true when all counts zero
//   - Path L: BackendCounts::summary() formats correctly
//   - Path M: filter_by_device_type isolates DiscreteGpu
//   - Path N: filter_by_device_type isolates IntegratedGpu
//   - Path O: filter_by_device_type returns empty for non-matching
//   - Path P: filter_by_backend isolates Vulkan
//   - Path Q: filter_by_backend returns empty for unavailable backend
//   - Path R: TrinityInstance::enumerate_adapters_detailed() integration
//   - Path S: Enumeration is idempotent (multiple calls return consistent results)
//   - Path T: Concurrent enumeration from multiple threads
//   - Path U: device_type_description returns correct strings

use renderer_backend::device::{
    enumerate_adapters_with_info, filter_by_backend, filter_by_device_type,
    BackendCounts, EnumerationResult, TrinityInstance, device_type_description,
};
use std::sync::Arc;
use std::thread;
use wgpu::{Backends, DeviceType};

// ============================================================================
// Test Helpers
// ============================================================================

/// Create a wgpu Instance for testing.
/// Uses default configuration to match production behavior.
fn create_test_instance() -> wgpu::Instance {
    wgpu::Instance::default()
}

/// Create a TrinityInstance for testing.
fn create_trinity_instance() -> TrinityInstance {
    TrinityInstance::new()
}

/// Create an empty EnumerationResult for testing helper methods.
fn empty_enumeration_result() -> EnumerationResult {
    EnumerationResult {
        adapters: vec![],
        backend_counts: BackendCounts::default(),
    }
}

// ============================================================================
// Path A-C: Basic Enumeration Paths
// ============================================================================

/// Path A: enumerate_adapters_with_info with PRIMARY backends
///
/// Tests that enumeration with PRIMARY backends returns a valid EnumerationResult
/// structure, regardless of whether adapters are found.
#[test]
fn test_enumerate_adapters_with_primary_backends() {
    let instance = create_test_instance();
    let result = enumerate_adapters_with_info(&instance, Backends::PRIMARY);

    // Structure should be valid regardless of adapter count
    // Note: total() returns usize which is always >= 0
    assert_eq!(result.len(), result.adapters.len());
    assert_eq!(result.backend_counts.total(), result.adapters.len());
}

/// Path B: enumerate_adapters_with_info with specific backend (VULKAN)
///
/// Tests enumeration with a single specific backend. On systems with Vulkan,
/// adapters should be found. On systems without Vulkan, empty result is valid.
#[test]
#[cfg(any(target_os = "linux", target_os = "windows"))]
fn test_enumerate_adapters_with_vulkan_backend() {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: Backends::VULKAN,
        ..Default::default()
    });
    let result = enumerate_adapters_with_info(&instance, Backends::VULKAN);

    // If adapters found, they should all be Vulkan
    for adapter in &result.adapters {
        let info = adapter.get_info();
        assert_eq!(info.backend, wgpu::Backend::Vulkan);
    }

    // Backend counts should reflect Vulkan only
    assert_eq!(result.backend_counts.metal, 0);
    assert_eq!(result.backend_counts.dx12, 0);
    assert_eq!(result.backend_counts.webgpu, 0);
    // gl might be 0 or not depending on configuration
}

/// Path C: enumerate_adapters_with_info with EMPTY backends (returns empty)
///
/// Tests that requesting enumeration with no backends returns an empty result.
#[test]
fn test_enumerate_adapters_with_empty_backends() {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: Backends::empty(),
        ..Default::default()
    });
    let result = enumerate_adapters_with_info(&instance, Backends::empty());

    assert!(result.is_empty());
    assert_eq!(result.len(), 0);
    assert!(result.backend_counts.is_empty());
    assert_eq!(result.backend_counts.total(), 0);
}

// ============================================================================
// Path D-F: EnumerationResult Basic Methods
// ============================================================================

/// Path D: EnumerationResult::is_empty() returns true when no adapters
#[test]
fn test_enumeration_result_is_empty_true() {
    let result = empty_enumeration_result();
    assert!(result.is_empty());
}

/// Path E: EnumerationResult::is_empty() returns false when adapters exist
///
/// This test requires hardware. We verify the invariant holds if adapters found.
#[test]
fn test_enumeration_result_is_empty_false_with_adapters() {
    let instance = create_test_instance();
    let result = enumerate_adapters_with_info(&instance, Backends::PRIMARY);

    if !result.adapters.is_empty() {
        assert!(!result.is_empty());
    }
}

/// Path F: EnumerationResult::len() returns correct count
#[test]
fn test_enumeration_result_len_matches_adapters() {
    let instance = create_test_instance();
    let result = enumerate_adapters_with_info(&instance, Backends::PRIMARY);

    assert_eq!(result.len(), result.adapters.len());
}

#[test]
fn test_enumeration_result_len_zero_when_empty() {
    let result = empty_enumeration_result();
    assert_eq!(result.len(), 0);
}

// ============================================================================
// Path G-I: EnumerationResult Adapter Selection Methods
// ============================================================================

/// Path G: EnumerationResult::first_discrete() finds discrete GPU
///
/// If the system has a discrete GPU, first_discrete() should find it.
#[test]
fn test_enumeration_result_first_discrete() {
    let instance = create_test_instance();
    let result = enumerate_adapters_with_info(&instance, Backends::PRIMARY);

    // Check if any adapter is discrete
    let has_discrete = result.adapters.iter()
        .any(|a| a.get_info().device_type == DeviceType::DiscreteGpu);

    if has_discrete {
        let discrete = result.first_discrete();
        assert!(discrete.is_some());
        assert_eq!(discrete.unwrap().get_info().device_type, DeviceType::DiscreteGpu);
    } else {
        // No discrete GPU - first_discrete should return None
        assert!(result.first_discrete().is_none());
    }
}

/// Path H: EnumerationResult::first_integrated() finds integrated GPU
///
/// If the system has an integrated GPU, first_integrated() should find it.
#[test]
fn test_enumeration_result_first_integrated() {
    let instance = create_test_instance();
    let result = enumerate_adapters_with_info(&instance, Backends::PRIMARY);

    // Check if any adapter is integrated
    let has_integrated = result.adapters.iter()
        .any(|a| a.get_info().device_type == DeviceType::IntegratedGpu);

    if has_integrated {
        let integrated = result.first_integrated();
        assert!(integrated.is_some());
        assert_eq!(integrated.unwrap().get_info().device_type, DeviceType::IntegratedGpu);
    } else {
        // No integrated GPU - first_integrated should return None
        assert!(result.first_integrated().is_none());
    }
}

/// Path I: EnumerationResult::best_adapter() priority order
///
/// Tests that best_adapter() returns adapters in the correct priority order:
/// DiscreteGpu > IntegratedGpu > VirtualGpu > Cpu > Other
#[test]
fn test_enumeration_result_best_adapter_priority() {
    let instance = create_test_instance();
    let result = enumerate_adapters_with_info(&instance, Backends::PRIMARY);

    if result.is_empty() {
        assert!(result.best_adapter().is_none());
        return;
    }

    let best = result.best_adapter();
    assert!(best.is_some());

    let best_type = best.unwrap().get_info().device_type;

    // Check that the best adapter follows priority order
    // If a higher priority type exists, best should be that type
    if result.first_discrete().is_some() {
        assert_eq!(best_type, DeviceType::DiscreteGpu,
            "If discrete GPU exists, it should be best");
    } else if result.first_integrated().is_some() {
        assert_eq!(best_type, DeviceType::IntegratedGpu,
            "If no discrete but integrated exists, integrated should be best");
    }
    // For VirtualGpu, Cpu, Other we just verify we got something
}

/// Test best_adapter returns None for empty result
#[test]
fn test_enumeration_result_best_adapter_empty() {
    let result = empty_enumeration_result();
    assert!(result.best_adapter().is_none());
}

// ============================================================================
// Path J-L: BackendCounts Methods
// ============================================================================

/// Path J: BackendCounts::total() sums all counts
#[test]
fn test_backend_counts_total_sums_correctly() {
    let counts = BackendCounts {
        vulkan: 2,
        metal: 0,
        dx12: 1,
        gl: 3,
        webgpu: 0,
    };
    assert_eq!(counts.total(), 6);
}

#[test]
fn test_backend_counts_total_zero_when_empty() {
    let counts = BackendCounts::default();
    assert_eq!(counts.total(), 0);
}

#[test]
fn test_backend_counts_total_all_backends() {
    let counts = BackendCounts {
        vulkan: 1,
        metal: 1,
        dx12: 1,
        gl: 1,
        webgpu: 1,
    };
    assert_eq!(counts.total(), 5);
}

/// Path K: BackendCounts::is_empty() true when all counts zero
#[test]
fn test_backend_counts_is_empty_true() {
    let counts = BackendCounts::default();
    assert!(counts.is_empty());
}

#[test]
fn test_backend_counts_is_empty_false_with_vulkan() {
    let counts = BackendCounts {
        vulkan: 1,
        ..Default::default()
    };
    assert!(!counts.is_empty());
}

#[test]
fn test_backend_counts_is_empty_false_with_gl() {
    let counts = BackendCounts {
        gl: 1,
        ..Default::default()
    };
    assert!(!counts.is_empty());
}

/// Path L: BackendCounts::summary() formats correctly
#[test]
fn test_backend_counts_summary_empty() {
    let counts = BackendCounts::default();
    assert_eq!(counts.summary(), "none");
}

#[test]
fn test_backend_counts_summary_single_backend() {
    let counts = BackendCounts {
        vulkan: 2,
        ..Default::default()
    };
    assert_eq!(counts.summary(), "Vulkan: 2");
}

#[test]
fn test_backend_counts_summary_multiple_backends() {
    let counts = BackendCounts {
        vulkan: 1,
        gl: 2,
        ..Default::default()
    };
    assert_eq!(counts.summary(), "Vulkan: 1, GL: 2");
}

#[test]
fn test_backend_counts_summary_all_backends() {
    let counts = BackendCounts {
        vulkan: 1,
        metal: 2,
        dx12: 3,
        gl: 4,
        webgpu: 5,
    };
    assert_eq!(counts.summary(), "Vulkan: 1, Metal: 2, DX12: 3, GL: 4, WebGPU: 5");
}

#[test]
fn test_backend_counts_summary_preserves_order() {
    // Summary should always be in order: Vulkan, Metal, DX12, GL, WebGPU
    let counts = BackendCounts {
        webgpu: 1,
        vulkan: 1,
        gl: 1,
        dx12: 1,
        metal: 1,
    };
    // Verify order is maintained regardless of initialization order
    let summary = counts.summary();
    let vulkan_pos = summary.find("Vulkan").unwrap();
    let metal_pos = summary.find("Metal").unwrap();
    let dx12_pos = summary.find("DX12").unwrap();
    let gl_pos = summary.find("GL:").unwrap(); // Use "GL:" to avoid matching "WebGPU"
    let webgpu_pos = summary.find("WebGPU").unwrap();

    assert!(vulkan_pos < metal_pos);
    assert!(metal_pos < dx12_pos);
    assert!(dx12_pos < gl_pos);
    assert!(gl_pos < webgpu_pos);
}

// ============================================================================
// Path M-Q: Filter Functions
// ============================================================================

/// Path M: filter_by_device_type isolates DiscreteGpu
///
/// Note: wgpu::Adapter doesn't implement Clone, so we enumerate twice
/// to test the filter function with fresh adapters.
#[test]
fn test_filter_by_device_type_discrete() {
    let instance = create_test_instance();

    // First enumeration to count expected discrete adapters
    let result1 = enumerate_adapters_with_info(&instance, Backends::PRIMARY);
    let expected_discrete_count = result1.adapters.iter()
        .filter(|a| a.get_info().device_type == DeviceType::DiscreteGpu)
        .count();

    // Second enumeration to test filter
    let result2 = enumerate_adapters_with_info(&instance, Backends::PRIMARY);
    let discrete = filter_by_device_type(result2.adapters, DeviceType::DiscreteGpu);

    // All filtered adapters should be discrete
    for adapter in &discrete {
        assert_eq!(adapter.get_info().device_type, DeviceType::DiscreteGpu);
    }

    // Count should match
    assert_eq!(discrete.len(), expected_discrete_count);
}

/// Path N: filter_by_device_type isolates IntegratedGpu
#[test]
fn test_filter_by_device_type_integrated() {
    let instance = create_test_instance();

    // Enumerate to test filter
    let result = enumerate_adapters_with_info(&instance, Backends::PRIMARY);
    let integrated = filter_by_device_type(result.adapters, DeviceType::IntegratedGpu);

    // All filtered adapters should be integrated
    for adapter in &integrated {
        assert_eq!(adapter.get_info().device_type, DeviceType::IntegratedGpu);
    }
}

/// Path O: filter_by_device_type returns empty for non-matching
#[test]
fn test_filter_by_device_type_returns_empty_for_nonexistent() {
    let instance = create_test_instance();
    let result = enumerate_adapters_with_info(&instance, Backends::PRIMARY);

    // VirtualGpu is unlikely to exist on most systems
    // But we can still verify the filter logic
    let filtered = filter_by_device_type(result.adapters, DeviceType::VirtualGpu);

    // All returned adapters (if any) should match the requested type
    for adapter in &filtered {
        assert_eq!(adapter.get_info().device_type, DeviceType::VirtualGpu);
    }
}

#[test]
fn test_filter_by_device_type_empty_input() {
    let filtered = filter_by_device_type(vec![], DeviceType::DiscreteGpu);
    assert!(filtered.is_empty());
}

/// Path P: filter_by_backend isolates Vulkan
#[test]
#[cfg(any(target_os = "linux", target_os = "windows"))]
fn test_filter_by_backend_vulkan() {
    let instance = create_test_instance();

    // First enumeration to get expected count
    let result1 = enumerate_adapters_with_info(&instance, Backends::PRIMARY);
    let expected_vulkan_count = result1.backend_counts.vulkan;

    // Second enumeration to test filter
    let result2 = enumerate_adapters_with_info(&instance, Backends::PRIMARY);
    let vulkan = filter_by_backend(result2.adapters, wgpu::Backend::Vulkan);

    // All filtered adapters should be Vulkan
    for adapter in &vulkan {
        assert_eq!(adapter.get_info().backend, wgpu::Backend::Vulkan);
    }

    // Count should match backend_counts.vulkan
    assert_eq!(vulkan.len(), expected_vulkan_count);
}

/// Path Q: filter_by_backend returns empty for unavailable backend
#[test]
#[cfg(target_os = "linux")]
fn test_filter_by_backend_unavailable_metal() {
    let instance = create_test_instance();
    let result = enumerate_adapters_with_info(&instance, Backends::PRIMARY);

    // Metal is not available on Linux
    let metal = filter_by_backend(result.adapters, wgpu::Backend::Metal);
    assert!(metal.is_empty());
}

#[test]
#[cfg(target_os = "linux")]
fn test_filter_by_backend_unavailable_dx12() {
    let instance = create_test_instance();
    let result = enumerate_adapters_with_info(&instance, Backends::PRIMARY);

    // DX12 is not available on Linux
    let dx12 = filter_by_backend(result.adapters, wgpu::Backend::Dx12);
    assert!(dx12.is_empty());
}

#[test]
fn test_filter_by_backend_empty_input() {
    let filtered = filter_by_backend(vec![], wgpu::Backend::Vulkan);
    assert!(filtered.is_empty());
}

// ============================================================================
// Path R: TrinityInstance Integration
// ============================================================================

/// Path R: TrinityInstance::enumerate_adapters_detailed() integration
///
/// Tests that TrinityInstance correctly wraps enumerate_adapters_with_info.
#[test]
fn test_trinity_instance_enumerate_adapters_detailed() {
    let instance = create_trinity_instance();
    let result = instance.enumerate_adapters_detailed();

    // Should return valid EnumerationResult
    assert_eq!(result.len(), result.adapters.len());
    assert_eq!(result.backend_counts.total(), result.adapters.len());

    // Compare with direct enumeration
    let direct_result = enumerate_adapters_with_info(instance.inner(), instance.backends());

    // Results should match (adapter enumeration is deterministic)
    assert_eq!(result.len(), direct_result.len());
    assert_eq!(result.backend_counts.vulkan, direct_result.backend_counts.vulkan);
    assert_eq!(result.backend_counts.gl, direct_result.backend_counts.gl);
}

/// Test that enumerate_adapters and enumerate_adapters_detailed return same adapter count
#[test]
fn test_trinity_instance_enumeration_consistency() {
    let instance = create_trinity_instance();

    let simple = instance.enumerate_adapters();
    let detailed = instance.enumerate_adapters_detailed();

    assert_eq!(simple.len(), detailed.len());
}

// ============================================================================
// Path S-T: Idempotency and Concurrency
// ============================================================================

/// Path S: Enumeration is idempotent (multiple calls return consistent results)
#[test]
fn test_enumeration_idempotent() {
    let instance = create_test_instance();

    let result1 = enumerate_adapters_with_info(&instance, Backends::PRIMARY);
    let result2 = enumerate_adapters_with_info(&instance, Backends::PRIMARY);
    let result3 = enumerate_adapters_with_info(&instance, Backends::PRIMARY);

    // All calls should return same count
    assert_eq!(result1.len(), result2.len());
    assert_eq!(result2.len(), result3.len());

    // Backend counts should match
    assert_eq!(result1.backend_counts.vulkan, result2.backend_counts.vulkan);
    assert_eq!(result2.backend_counts.vulkan, result3.backend_counts.vulkan);
    assert_eq!(result1.backend_counts.gl, result2.backend_counts.gl);
    assert_eq!(result2.backend_counts.gl, result3.backend_counts.gl);
}

/// Path T: Concurrent enumeration from multiple threads
///
/// Tests that adapter enumeration is thread-safe.
#[test]
fn test_concurrent_enumeration() {
    // Create a shared instance
    let instance = Arc::new(create_test_instance());
    let backends = Backends::PRIMARY;

    // Spawn multiple threads to enumerate concurrently
    let handles: Vec<_> = (0..4)
        .map(|_| {
            let instance = Arc::clone(&instance);
            thread::spawn(move || {
                enumerate_adapters_with_info(&instance, backends)
            })
        })
        .collect();

    // Collect results
    let results: Vec<_> = handles
        .into_iter()
        .map(|h| h.join().expect("Thread should not panic"))
        .collect();

    // All results should have the same count
    let first_len = results[0].len();
    for result in &results {
        assert_eq!(result.len(), first_len,
            "Concurrent enumeration should return consistent results");
    }
}

/// Test concurrent enumeration with TrinityInstance
#[test]
fn test_concurrent_trinity_instance_enumeration() {
    let handles: Vec<_> = (0..4)
        .map(|_| {
            thread::spawn(|| {
                let instance = TrinityInstance::new();
                instance.enumerate_adapters_detailed()
            })
        })
        .collect();

    let results: Vec<_> = handles
        .into_iter()
        .map(|h| h.join().expect("Thread should not panic"))
        .collect();

    // All results should have the same count (assuming stable hardware)
    let first_len = results[0].len();
    for result in &results {
        assert_eq!(result.len(), first_len);
    }
}

// ============================================================================
// Path U: device_type_description
// ============================================================================

/// Path U: device_type_description returns correct strings
#[test]
fn test_device_type_description_discrete() {
    let desc = device_type_description(DeviceType::DiscreteGpu);
    assert_eq!(desc, "Discrete GPU (dedicated graphics card)");
}

#[test]
fn test_device_type_description_integrated() {
    let desc = device_type_description(DeviceType::IntegratedGpu);
    assert_eq!(desc, "Integrated GPU (shared memory with CPU)");
}

#[test]
fn test_device_type_description_virtual() {
    let desc = device_type_description(DeviceType::VirtualGpu);
    assert_eq!(desc, "Virtual GPU (virtualized environment)");
}

#[test]
fn test_device_type_description_cpu() {
    let desc = device_type_description(DeviceType::Cpu);
    assert_eq!(desc, "CPU (software rendering)");
}

#[test]
fn test_device_type_description_other() {
    let desc = device_type_description(DeviceType::Other);
    assert_eq!(desc, "Other (unknown device type)");
}

// ============================================================================
// Additional Edge Cases
// ============================================================================

/// Test BackendCounts Default trait
#[test]
fn test_backend_counts_default_trait() {
    let counts: BackendCounts = Default::default();
    assert_eq!(counts.vulkan, 0);
    assert_eq!(counts.metal, 0);
    assert_eq!(counts.dx12, 0);
    assert_eq!(counts.gl, 0);
    assert_eq!(counts.webgpu, 0);
}

/// Test BackendCounts Clone trait
#[test]
fn test_backend_counts_clone() {
    let original = BackendCounts {
        vulkan: 5,
        metal: 3,
        dx12: 2,
        gl: 1,
        webgpu: 0,
    };
    let cloned = original.clone();

    assert_eq!(original.vulkan, cloned.vulkan);
    assert_eq!(original.metal, cloned.metal);
    assert_eq!(original.dx12, cloned.dx12);
    assert_eq!(original.gl, cloned.gl);
    assert_eq!(original.webgpu, cloned.webgpu);
}

/// Test BackendCounts Copy trait
#[test]
fn test_backend_counts_copy() {
    let original = BackendCounts {
        vulkan: 5,
        metal: 3,
        dx12: 2,
        gl: 1,
        webgpu: 0,
    };
    let copied = original; // Copy, not move

    // Original should still be usable
    assert_eq!(original.total(), 11);
    assert_eq!(copied.total(), 11);
}

/// Test BackendCounts PartialEq trait
#[test]
fn test_backend_counts_partial_eq() {
    let a = BackendCounts {
        vulkan: 1,
        metal: 2,
        dx12: 3,
        gl: 4,
        webgpu: 5,
    };
    let b = BackendCounts {
        vulkan: 1,
        metal: 2,
        dx12: 3,
        gl: 4,
        webgpu: 5,
    };
    let c = BackendCounts {
        vulkan: 0,
        ..Default::default()
    };

    assert_eq!(a, b);
    assert_ne!(a, c);
}

/// Test EnumerationResult Debug trait
#[test]
fn test_enumeration_result_debug() {
    let result = empty_enumeration_result();
    let debug_str = format!("{:?}", result);

    assert!(debug_str.contains("EnumerationResult"));
    assert!(debug_str.contains("adapters"));
    assert!(debug_str.contains("backend_counts"));
}

/// Test BackendCounts Debug trait
#[test]
fn test_backend_counts_debug() {
    let counts = BackendCounts {
        vulkan: 1,
        metal: 0,
        dx12: 0,
        gl: 2,
        webgpu: 0,
    };
    let debug_str = format!("{:?}", counts);

    assert!(debug_str.contains("BackendCounts"));
    assert!(debug_str.contains("vulkan: 1"));
    assert!(debug_str.contains("gl: 2"));
}

/// Test that all DeviceType variants are handled
#[test]
fn test_all_device_types_have_descriptions() {
    // This test ensures we don't miss any device types
    let device_types = [
        DeviceType::DiscreteGpu,
        DeviceType::IntegratedGpu,
        DeviceType::VirtualGpu,
        DeviceType::Cpu,
        DeviceType::Other,
    ];

    for dt in device_types {
        let desc = device_type_description(dt);
        assert!(!desc.is_empty(), "Device type {:?} should have a description", dt);
    }
}

/// Test filter functions preserve adapter order
#[test]
fn test_filter_preserves_order() {
    let instance = create_test_instance();

    // First enumeration to determine backend and expected order
    let result1 = enumerate_adapters_with_info(&instance, Backends::PRIMARY);

    if result1.len() < 2 {
        return; // Need at least 2 adapters to test order
    }

    // Get all adapters of a specific backend type
    let backend_type = result1.adapters[0].get_info().backend;
    let original_names: Vec<_> = result1.adapters.iter()
        .filter(|a| a.get_info().backend == backend_type)
        .map(|a| a.get_info().name.clone())
        .collect();

    // Second enumeration to test filter
    let result2 = enumerate_adapters_with_info(&instance, Backends::PRIMARY);
    let filtered = filter_by_backend(result2.adapters, backend_type);
    let filtered_names: Vec<_> = filtered.iter()
        .map(|a| a.get_info().name.clone())
        .collect();

    assert_eq!(original_names, filtered_names, "Filter should preserve adapter order");
}

/// Test enumeration with GL backend specifically
#[test]
fn test_enumerate_adapters_with_gl_backend() {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: Backends::GL,
        ..Default::default()
    });
    let result = enumerate_adapters_with_info(&instance, Backends::GL);

    // If adapters found, they should all be GL
    for adapter in &result.adapters {
        let info = adapter.get_info();
        assert_eq!(info.backend, wgpu::Backend::Gl);
    }

    // Only GL count should be non-zero
    assert_eq!(result.backend_counts.vulkan, 0);
    assert_eq!(result.backend_counts.metal, 0);
    assert_eq!(result.backend_counts.dx12, 0);
    assert_eq!(result.backend_counts.webgpu, 0);
}

/// Test that backend counts match actual adapters by backend
#[test]
fn test_backend_counts_accuracy() {
    let instance = create_test_instance();
    let result = enumerate_adapters_with_info(&instance, Backends::PRIMARY);

    // Count adapters manually by backend
    let mut manual_counts = BackendCounts::default();
    for adapter in &result.adapters {
        match adapter.get_info().backend {
            wgpu::Backend::Vulkan => manual_counts.vulkan += 1,
            wgpu::Backend::Metal => manual_counts.metal += 1,
            wgpu::Backend::Dx12 => manual_counts.dx12 += 1,
            wgpu::Backend::Gl => manual_counts.gl += 1,
            wgpu::Backend::BrowserWebGpu => manual_counts.webgpu += 1,
            wgpu::Backend::Empty => {}
        }
    }

    assert_eq!(result.backend_counts, manual_counts,
        "BackendCounts should accurately reflect adapter backends");
}
