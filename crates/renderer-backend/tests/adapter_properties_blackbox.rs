// Blackbox contract tests for T-WGPU-P1.2.2 Adapter Properties
//
// CLEANROOM: No access to implementation files. Tests use only the public API
// exported by `renderer_backend::device`.
//
// Forbidden files (per TESTDEV_BLACKBOX prompt):
//   - crates/renderer-backend/src/device/adapter.rs
//   - crates/renderer-backend/src/device/instance.rs
//   - Any WHITEBOX test file for this task
//
// Contract sources:
//   - docs/WGPU_DOCS/RDC_OUTPUT/PHASE_1_CORE_TODO.md (T-WGPU-P1.2.2)
//
// Acceptance criteria (T-WGPU-P1.2.2):
//   - Vendor ID extraction (NVIDIA, AMD, Intel, Apple, ARM)
//   - DeviceType detection (DiscreteGpu, IntegratedGpu, Cpu)
//   - Driver version when available
//   - Human-readable adapter description
//
// Test design rationale:
//   Equivalence partitioning:
//     - Known vendor IDs (NVIDIA, AMD, Intel, Apple, ARM)
//     - Unknown vendor IDs
//     - Different device types (Discrete, Integrated, Software)
//   Boundary cases:
//     - Zero adapters (empty properties list)
//     - Vendor ID edge values (0x0000, 0xFFFFFFFF)
//   Contract verification:
//     - Vendor struct fields and methods
//     - AdapterProperties struct fields and methods
//     - Device type helper methods

use renderer_backend::device::{
    enumerate_adapters_with_info, AdapterProperties, TrinityInstance, Vendor,
};

// =============================================================================
// 1. Vendor Classification Contract Tests
// =============================================================================

/// Verifies that Vendor can be created from NVIDIA vendor ID.
///
/// Contract: Vendor::from_id(0x10DE) produces a Vendor with name containing "NVIDIA".
#[test]
fn test_vendor_nvidia_id_produces_nvidia_name() {
    const NVIDIA_VENDOR_ID: u32 = 0x10DE;
    let vendor = Vendor::from_id(NVIDIA_VENDOR_ID);

    let name = vendor.name();
    assert!(
        name.to_uppercase().contains("NVIDIA"),
        "NVIDIA vendor ID (0x10DE) should produce name containing 'NVIDIA', got: {}",
        name
    );
}

/// Verifies that Vendor can be created from AMD vendor ID.
///
/// Contract: Vendor::from_id(0x1002) produces a Vendor with name containing "AMD".
#[test]
fn test_vendor_amd_id_produces_amd_name() {
    const AMD_VENDOR_ID: u32 = 0x1002;
    let vendor = Vendor::from_id(AMD_VENDOR_ID);

    let name = vendor.name();
    assert!(
        name.to_uppercase().contains("AMD"),
        "AMD vendor ID (0x1002) should produce name containing 'AMD', got: {}",
        name
    );
}

/// Verifies that Vendor can be created from Intel vendor ID.
///
/// Contract: Vendor::from_id(0x8086) produces a Vendor with name containing "Intel".
#[test]
fn test_vendor_intel_id_produces_intel_name() {
    const INTEL_VENDOR_ID: u32 = 0x8086;
    let vendor = Vendor::from_id(INTEL_VENDOR_ID);

    let name = vendor.name();
    assert!(
        name.to_uppercase().contains("INTEL"),
        "Intel vendor ID (0x8086) should produce name containing 'Intel', got: {}",
        name
    );
}

/// Verifies that Vendor can be created from Apple vendor ID.
///
/// Contract: Vendor::from_id(0x106B) produces a Vendor with name containing "Apple".
#[test]
fn test_vendor_apple_id_produces_apple_name() {
    const APPLE_VENDOR_ID: u32 = 0x106B;
    let vendor = Vendor::from_id(APPLE_VENDOR_ID);

    let name = vendor.name();
    assert!(
        name.to_uppercase().contains("APPLE"),
        "Apple vendor ID (0x106B) should produce name containing 'Apple', got: {}",
        name
    );
}

/// Verifies that Vendor can be created from ARM vendor ID.
///
/// Contract: Vendor::from_id(0x13B5) produces a Vendor with name containing "ARM".
#[test]
fn test_vendor_arm_id_produces_arm_name() {
    const ARM_VENDOR_ID: u32 = 0x13B5;
    let vendor = Vendor::from_id(ARM_VENDOR_ID);

    let name = vendor.name();
    assert!(
        name.to_uppercase().contains("ARM"),
        "ARM vendor ID (0x13B5) should produce name containing 'ARM', got: {}",
        name
    );
}

/// Verifies that unknown vendor IDs are handled gracefully (no panic).
///
/// Contract: Unknown vendor IDs should not panic and should return a valid Vendor.
#[test]
fn test_vendor_unknown_id_does_not_panic() {
    // Test various unknown vendor IDs
    let unknown_ids = [0x0000, 0x1234, 0xDEAD, 0xBEEF, 0xFFFFFFFF];

    for id in unknown_ids {
        let vendor = Vendor::from_id(id);
        let name = vendor.name();
        // Should return a non-empty name (even if "Unknown")
        assert!(
            !name.is_empty(),
            "Unknown vendor ID (0x{:04X}) should produce non-empty name",
            id
        );
    }
}

/// Verifies that Vendor::name() returns a &str type.
///
/// Contract: Vendor.name() method returns a string slice.
#[test]
fn test_vendor_name_returns_str() {
    let vendor = Vendor::from_id(0x10DE);
    let name: &str = vendor.name();
    let _ = name;
}

// =============================================================================
// 2. AdapterProperties Contract Tests
// =============================================================================

/// Verifies that AdapterProperties can be created from a wgpu::Adapter.
///
/// Contract: AdapterProperties::from_adapter() returns valid properties.
#[test]
#[cfg(not(target_arch = "wasm32"))]
fn test_adapter_properties_from_adapter_returns_valid_properties() {
    let instance = TrinityInstance::new();
    let result = enumerate_adapters_with_info(instance.inner(), instance.backends());

    if !result.adapters.is_empty() {
        let adapter = &result.adapters[0];
        let props = AdapterProperties::from_adapter(adapter);

        // Should have a non-empty name
        assert!(
            !props.name.is_empty(),
            "AdapterProperties should have non-empty name"
        );
    }
}

/// Verifies that AdapterProperties has a name field.
///
/// Contract: AdapterProperties.name is accessible and is a String.
#[test]
#[cfg(not(target_arch = "wasm32"))]
fn test_adapter_properties_has_name_field() {
    let instance = TrinityInstance::new();
    let result = enumerate_adapters_with_info(instance.inner(), instance.backends());

    if !result.adapters.is_empty() {
        let adapter = &result.adapters[0];
        let props = AdapterProperties::from_adapter(adapter);

        // Access the name field
        let _name: &String = &props.name;
    }
}

/// Verifies that AdapterProperties has a vendor field.
///
/// Contract: AdapterProperties.vendor is accessible and is a Vendor.
#[test]
#[cfg(not(target_arch = "wasm32"))]
fn test_adapter_properties_has_vendor_field() {
    let instance = TrinityInstance::new();
    let result = enumerate_adapters_with_info(instance.inner(), instance.backends());

    if !result.adapters.is_empty() {
        let adapter = &result.adapters[0];
        let props = AdapterProperties::from_adapter(adapter);

        // Access the vendor field and call name() to verify it's a Vendor
        let _vendor_name: &str = props.vendor.name();
    }
}

/// Verifies that AdapterProperties has a device_type field.
///
/// Contract: AdapterProperties.device_type is accessible and is wgpu::DeviceType.
#[test]
#[cfg(not(target_arch = "wasm32"))]
fn test_adapter_properties_has_device_type_field() {
    let instance = TrinityInstance::new();
    let result = enumerate_adapters_with_info(instance.inner(), instance.backends());

    if !result.adapters.is_empty() {
        let adapter = &result.adapters[0];
        let props = AdapterProperties::from_adapter(adapter);

        // Access the device_type field
        let _device_type: wgpu::DeviceType = props.device_type;
    }
}

/// Verifies that AdapterProperties has a backend field.
///
/// Contract: AdapterProperties.backend is accessible and is wgpu::Backend.
#[test]
#[cfg(not(target_arch = "wasm32"))]
fn test_adapter_properties_has_backend_field() {
    let instance = TrinityInstance::new();
    let result = enumerate_adapters_with_info(instance.inner(), instance.backends());

    if !result.adapters.is_empty() {
        let adapter = &result.adapters[0];
        let props = AdapterProperties::from_adapter(adapter);

        // Access the backend field
        let _backend: wgpu::Backend = props.backend;
    }
}

/// Verifies that AdapterProperties has a driver field.
///
/// Contract: AdapterProperties.driver is accessible and is a String.
#[test]
#[cfg(not(target_arch = "wasm32"))]
fn test_adapter_properties_has_driver_field() {
    let instance = TrinityInstance::new();
    let result = enumerate_adapters_with_info(instance.inner(), instance.backends());

    if !result.adapters.is_empty() {
        let adapter = &result.adapters[0];
        let props = AdapterProperties::from_adapter(adapter);

        // Access the driver field - may be empty but should be accessible
        let _driver: &String = &props.driver;
    }
}

/// Verifies that AdapterProperties has a driver_info field.
///
/// Contract: AdapterProperties.driver_info is accessible and is a String.
#[test]
#[cfg(not(target_arch = "wasm32"))]
fn test_adapter_properties_has_driver_info_field() {
    let instance = TrinityInstance::new();
    let result = enumerate_adapters_with_info(instance.inner(), instance.backends());

    if !result.adapters.is_empty() {
        let adapter = &result.adapters[0];
        let props = AdapterProperties::from_adapter(adapter);

        // Access the driver_info field - may be empty but should be accessible
        let _driver_info: &String = &props.driver_info;
    }
}

// =============================================================================
// 3. Description Contract Tests
// =============================================================================

/// Verifies that AdapterProperties.description() returns a non-empty string.
///
/// Contract: description() returns a human-readable, non-empty string.
#[test]
#[cfg(not(target_arch = "wasm32"))]
fn test_adapter_properties_description_returns_nonempty_string() {
    let instance = TrinityInstance::new();
    let result = enumerate_adapters_with_info(instance.inner(), instance.backends());

    if !result.adapters.is_empty() {
        let adapter = &result.adapters[0];
        let props = AdapterProperties::from_adapter(adapter);

        let description = props.description();
        assert!(
            !description.is_empty(),
            "description() should return non-empty string"
        );
    }
}

/// Verifies that description() includes device type information.
///
/// Contract: description() includes some reference to the device type.
#[test]
#[cfg(not(target_arch = "wasm32"))]
fn test_adapter_properties_description_includes_device_type() {
    let instance = TrinityInstance::new();
    let result = enumerate_adapters_with_info(instance.inner(), instance.backends());

    if !result.adapters.is_empty() {
        let adapter = &result.adapters[0];
        let props = AdapterProperties::from_adapter(adapter);

        let description = props.description().to_lowercase();

        // Should contain some device type indicator
        let has_device_type = description.contains("discrete")
            || description.contains("integrated")
            || description.contains("software")
            || description.contains("cpu")
            || description.contains("virtual")
            || description.contains("gpu")
            || description.contains("other");

        assert!(
            has_device_type,
            "description() should include device type info, got: {}",
            props.description()
        );
    }
}

/// Verifies that description() includes backend information.
///
/// Contract: description() includes some reference to the backend.
#[test]
#[cfg(not(target_arch = "wasm32"))]
fn test_adapter_properties_description_includes_backend() {
    let instance = TrinityInstance::new();
    let result = enumerate_adapters_with_info(instance.inner(), instance.backends());

    if !result.adapters.is_empty() {
        let adapter = &result.adapters[0];
        let props = AdapterProperties::from_adapter(adapter);

        let description = props.description().to_lowercase();

        // Should contain backend name
        let has_backend = description.contains("vulkan")
            || description.contains("metal")
            || description.contains("dx12")
            || description.contains("d3d12")
            || description.contains("opengl")
            || description.contains("webgpu")
            || description.contains("empty");

        assert!(
            has_backend,
            "description() should include backend info, got: {}",
            props.description()
        );
    }
}

// =============================================================================
// 4. Device Type Helper Contract Tests
// =============================================================================

/// Verifies that is_discrete() returns a boolean.
///
/// Contract: is_discrete() returns bool.
#[test]
#[cfg(not(target_arch = "wasm32"))]
fn test_adapter_properties_is_discrete_returns_bool() {
    let instance = TrinityInstance::new();
    let result = enumerate_adapters_with_info(instance.inner(), instance.backends());

    if !result.adapters.is_empty() {
        let adapter = &result.adapters[0];
        let props = AdapterProperties::from_adapter(adapter);

        let _is_discrete: bool = props.is_discrete();
    }
}

/// Verifies that is_integrated() returns a boolean.
///
/// Contract: is_integrated() returns bool.
#[test]
#[cfg(not(target_arch = "wasm32"))]
fn test_adapter_properties_is_integrated_returns_bool() {
    let instance = TrinityInstance::new();
    let result = enumerate_adapters_with_info(instance.inner(), instance.backends());

    if !result.adapters.is_empty() {
        let adapter = &result.adapters[0];
        let props = AdapterProperties::from_adapter(adapter);

        let _is_integrated: bool = props.is_integrated();
    }
}

/// Verifies that is_software() returns a boolean.
///
/// Contract: is_software() returns bool.
#[test]
#[cfg(not(target_arch = "wasm32"))]
fn test_adapter_properties_is_software_returns_bool() {
    let instance = TrinityInstance::new();
    let result = enumerate_adapters_with_info(instance.inner(), instance.backends());

    if !result.adapters.is_empty() {
        let adapter = &result.adapters[0];
        let props = AdapterProperties::from_adapter(adapter);

        let _is_software: bool = props.is_software();
    }
}

/// Verifies that device type helpers are logically consistent for a single adapter.
///
/// Contract: For a single adapter, at most one of is_discrete/is_integrated/is_software
/// should be true (they represent distinct device types).
#[test]
#[cfg(not(target_arch = "wasm32"))]
fn test_adapter_properties_device_type_helpers_mutually_exclusive() {
    let instance = TrinityInstance::new();
    let result = enumerate_adapters_with_info(instance.inner(), instance.backends());

    for adapter in &result.adapters {
        let props = AdapterProperties::from_adapter(adapter);

        let discrete = props.is_discrete();
        let integrated = props.is_integrated();
        let software = props.is_software();

        let true_count = [discrete, integrated, software]
            .iter()
            .filter(|&&v| v)
            .count();

        assert!(
            true_count <= 1,
            "Device type helpers should be mutually exclusive. discrete={}, integrated={}, software={} for adapter: {}",
            discrete,
            integrated,
            software,
            props.name
        );
    }
}

/// Verifies that discrete GPU helper correctly identifies discrete devices.
///
/// Contract: is_discrete() returns true iff device_type is DiscreteGpu.
#[test]
#[cfg(not(target_arch = "wasm32"))]
fn test_adapter_properties_is_discrete_matches_device_type() {
    let instance = TrinityInstance::new();
    let result = enumerate_adapters_with_info(instance.inner(), instance.backends());

    for adapter in &result.adapters {
        let props = AdapterProperties::from_adapter(adapter);

        let is_discrete = props.is_discrete();
        let device_type_is_discrete = matches!(props.device_type, wgpu::DeviceType::DiscreteGpu);

        assert_eq!(
            is_discrete, device_type_is_discrete,
            "is_discrete() should match device_type == DiscreteGpu for adapter: {}",
            props.name
        );
    }
}

/// Verifies that integrated GPU helper correctly identifies integrated devices.
///
/// Contract: is_integrated() returns true iff device_type is IntegratedGpu.
#[test]
#[cfg(not(target_arch = "wasm32"))]
fn test_adapter_properties_is_integrated_matches_device_type() {
    let instance = TrinityInstance::new();
    let result = enumerate_adapters_with_info(instance.inner(), instance.backends());

    for adapter in &result.adapters {
        let props = AdapterProperties::from_adapter(adapter);

        let is_integrated = props.is_integrated();
        let device_type_is_integrated =
            matches!(props.device_type, wgpu::DeviceType::IntegratedGpu);

        assert_eq!(
            is_integrated, device_type_is_integrated,
            "is_integrated() should match device_type == IntegratedGpu for adapter: {}",
            props.name
        );
    }
}

/// Verifies that software renderer helper correctly identifies CPU/software devices.
///
/// Contract: is_software() returns true iff device_type is Cpu.
#[test]
#[cfg(not(target_arch = "wasm32"))]
fn test_adapter_properties_is_software_matches_device_type() {
    let instance = TrinityInstance::new();
    let result = enumerate_adapters_with_info(instance.inner(), instance.backends());

    for adapter in &result.adapters {
        let props = AdapterProperties::from_adapter(adapter);

        let is_software = props.is_software();
        let device_type_is_cpu = matches!(props.device_type, wgpu::DeviceType::Cpu);

        assert_eq!(
            is_software, device_type_is_cpu,
            "is_software() should match device_type == Cpu for adapter: {}",
            props.name
        );
    }
}

// =============================================================================
// 5. EnumerationResult Integration Tests
// =============================================================================

/// Verifies that EnumerationResult.properties() returns Vec<AdapterProperties>.
///
/// Contract: properties() method returns a Vec of AdapterProperties.
#[test]
fn test_enumeration_result_properties_returns_vec() {
    let instance = TrinityInstance::new();
    let result = enumerate_adapters_with_info(instance.inner(), instance.backends());

    let props: Vec<AdapterProperties> = result.properties();
    let _ = props;
}

/// Verifies that properties() count matches adapter count.
///
/// Contract: properties().len() == adapters.len().
#[test]
fn test_enumeration_result_properties_length_matches_adapters() {
    let instance = TrinityInstance::new();
    let result = enumerate_adapters_with_info(instance.inner(), instance.backends());

    let props = result.properties();

    assert_eq!(
        props.len(),
        result.adapters.len(),
        "properties() length should match adapters length"
    );
}

/// Verifies that all properties from enumeration can be iterated and accessed.
///
/// Contract: Can iterate through properties and access all fields without panic.
#[test]
#[cfg(not(target_arch = "wasm32"))]
fn test_enumeration_result_can_iterate_all_properties() {
    let instance = TrinityInstance::new();
    let result = enumerate_adapters_with_info(instance.inner(), instance.backends());

    let props = result.properties();

    for prop in &props {
        // Access all fields to verify they don't panic
        let _name = &prop.name;
        let _vendor_name = prop.vendor.name();
        let _device_type = prop.device_type;
        let _backend = prop.backend;
        let _driver = &prop.driver;
        let _driver_info = &prop.driver_info;
        let _description = prop.description();
        let _is_discrete = prop.is_discrete();
        let _is_integrated = prop.is_integrated();
        let _is_software = prop.is_software();
    }
}

// =============================================================================
// 6. Graceful Handling Tests
// =============================================================================

/// Verifies that properties from real adapters don't panic.
///
/// Contract: AdapterProperties::from_adapter() on any real adapter should not panic.
#[test]
#[cfg(not(target_arch = "wasm32"))]
fn test_adapter_properties_from_real_adapters_no_panic() {
    let instance = TrinityInstance::new();
    let result = enumerate_adapters_with_info(instance.inner(), instance.backends());

    for adapter in &result.adapters {
        // This should not panic for any real adapter
        let props = AdapterProperties::from_adapter(adapter);

        // Accessing description should also not panic
        let _ = props.description();
    }
}

/// Verifies that properties() works with zero adapters (empty result).
///
/// Contract: properties() on empty result returns empty Vec, not panic.
#[test]
fn test_enumeration_result_properties_handles_zero_adapters() {
    let instance = TrinityInstance::with_backends(wgpu::Backends::empty());
    let result = enumerate_adapters_with_info(instance.inner(), instance.backends());

    // Should be empty since we requested no backends
    assert!(
        result.adapters.is_empty(),
        "Should have no adapters with empty backends"
    );

    // properties() should return empty vec, not panic
    let props = result.properties();
    assert!(
        props.is_empty(),
        "properties() should return empty Vec for zero adapters"
    );
}

/// Verifies that Vendor handles edge case vendor IDs gracefully.
///
/// Contract: Edge case vendor IDs (0, max u32) should not panic.
#[test]
fn test_vendor_handles_edge_case_ids() {
    // Minimum value
    let vendor_zero = Vendor::from_id(0);
    assert!(
        !vendor_zero.name().is_empty(),
        "Vendor(0) should have non-empty name"
    );

    // Maximum value
    let vendor_max = Vendor::from_id(u32::MAX);
    assert!(
        !vendor_max.name().is_empty(),
        "Vendor(u32::MAX) should have non-empty name"
    );
}

// =============================================================================
// 7. All Known Vendors Integration Test
// =============================================================================

/// Verifies all known vendor IDs map to expected vendors.
///
/// Contract: All documented vendor IDs produce recognizable vendor names.
#[test]
fn test_all_known_vendor_ids_map_correctly() {
    let known_vendors: [(u32, &str); 5] = [
        (0x10DE, "NVIDIA"),
        (0x1002, "AMD"),
        (0x8086, "INTEL"),
        (0x106B, "APPLE"),
        (0x13B5, "ARM"),
    ];

    for (id, expected_substr) in known_vendors {
        let vendor = Vendor::from_id(id);
        let name = vendor.name().to_uppercase();

        assert!(
            name.contains(expected_substr),
            "Vendor ID 0x{:04X} should map to name containing '{}', got: {}",
            id,
            expected_substr,
            vendor.name()
        );
    }
}

// =============================================================================
// 8. Consistency Tests
// =============================================================================

/// Verifies that multiple calls to from_adapter produce consistent results.
///
/// Contract: from_adapter is deterministic for the same adapter.
#[test]
#[cfg(not(target_arch = "wasm32"))]
fn test_adapter_properties_from_adapter_is_deterministic() {
    let instance = TrinityInstance::new();
    let result = enumerate_adapters_with_info(instance.inner(), instance.backends());

    if !result.adapters.is_empty() {
        let adapter = &result.adapters[0];

        let props1 = AdapterProperties::from_adapter(adapter);
        let props2 = AdapterProperties::from_adapter(adapter);

        assert_eq!(
            props1.name, props2.name,
            "Multiple from_adapter calls should produce same name"
        );
        assert_eq!(
            props1.vendor.name(),
            props2.vendor.name(),
            "Multiple from_adapter calls should produce same vendor"
        );
        assert_eq!(
            props1.device_type, props2.device_type,
            "Multiple from_adapter calls should produce same device_type"
        );
        assert_eq!(
            props1.backend, props2.backend,
            "Multiple from_adapter calls should produce same backend"
        );
    }
}

/// Verifies that multiple calls to description produce consistent results.
///
/// Contract: description() is deterministic for the same properties.
#[test]
#[cfg(not(target_arch = "wasm32"))]
fn test_adapter_properties_description_is_deterministic() {
    let instance = TrinityInstance::new();
    let result = enumerate_adapters_with_info(instance.inner(), instance.backends());

    if !result.adapters.is_empty() {
        let adapter = &result.adapters[0];
        let props = AdapterProperties::from_adapter(adapter);

        let desc1 = props.description();
        let desc2 = props.description();
        let desc3 = props.description();

        assert_eq!(
            desc1, desc2,
            "Multiple description() calls should produce same result"
        );
        assert_eq!(
            desc2, desc3,
            "Multiple description() calls should produce same result"
        );
    }
}

/// Verifies that multiple calls to properties() produce consistent results.
///
/// Contract: properties() is deterministic for the same enumeration result.
#[test]
fn test_enumeration_result_properties_is_deterministic() {
    let instance = TrinityInstance::new();
    let result = enumerate_adapters_with_info(instance.inner(), instance.backends());

    let props1 = result.properties();
    let props2 = result.properties();

    assert_eq!(
        props1.len(),
        props2.len(),
        "Multiple properties() calls should produce same count"
    );

    for (p1, p2) in props1.iter().zip(props2.iter()) {
        assert_eq!(
            p1.name, p2.name,
            "Multiple properties() calls should produce same names"
        );
    }
}
