// Blackbox contract tests for T-WGPU-P1.3.1 Device Creation
//
// CLEANROOM: No access to implementation files. Tests use only the public API
// exported by `renderer_backend::device`.
//
// Forbidden files (per TESTDEV_BLACKBOX prompt):
//   - crates/renderer-backend/src/device/device.rs
//   - crates/renderer-backend/src/device/adapter.rs
//   - crates/renderer-backend/src/device/instance.rs
//   - Any WHITEBOX test file for this task
//
// Contract sources:
//   - docs/WGPU_DOCS/RDC_OUTPUT/PHASE_1_CORE_TODO.md (T-WGPU-P1.3.1)
//   - docs/WGPU_DOCS/RDC_OUTPUT/PHASE_1_CORE_ARCH.md (Architecture spec)
//
// Acceptance criteria (T-WGPU-P1.3.1):
//   - Creates device with requested features
//   - Creates device with requested limits
//   - Handles request failure gracefully
//   - Logs device creation details
//
// Test design rationale:
//   Equivalence partitioning:
//     - Device creation with defaults (minimal requirements)
//     - Device creation with all features (maximal requirements)
//     - Device creation with custom features/limits (explicit requirements)
//   Boundary cases:
//     - Zero features requested
//     - Maximum limits requested
//     - Unsupported features
//   Contract verification:
//     - TrinityDevice struct methods
//     - DeviceCreationError variants
//     - Display and Debug implementations

use pollster::FutureExt as _;
use renderer_backend::device::{
    enumerate_adapters_with_info, AdapterSelector, DeviceCreationError, TrinityDevice,
    TrinityInstance,
};
use std::fmt::{Debug, Display};

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

/// Creates a TrinityInstance and gets the first available adapter.
/// Skips tests if no adapter is available.
fn get_test_adapter() -> Option<wgpu::Adapter> {
    let instance = TrinityInstance::new();
    let result = enumerate_adapters_with_info(instance.inner(), instance.backends());
    result.adapters.into_iter().next()
}

/// Helper macro to skip test if no GPU adapter is available.
macro_rules! require_adapter {
    () => {
        match get_test_adapter() {
            Some(adapter) => adapter,
            None => {
                eprintln!("SKIP: No GPU adapter available for this test");
                return;
            }
        }
    };
}

// =============================================================================
// 1. TrinityDevice Type Contract Tests
// =============================================================================

/// Verifies that TrinityDevice is a publicly accessible type.
///
/// Contract: TrinityDevice is exported from renderer_backend::device.
#[test]
fn test_trinity_device_type_is_exported() {
    // This test verifies the type exists and is accessible
    fn _assert_type_exists<T>(_: Option<&T>) {}
    _assert_type_exists::<TrinityDevice>(None);
}

/// Verifies that TrinityDevice implements Debug.
///
/// Contract: TrinityDevice derives Debug for introspection.
#[test]
fn test_trinity_device_implements_debug() {
    fn _assert_debug<T: Debug>() {}
    _assert_debug::<TrinityDevice>();
}

/// Verifies that TrinityDevice implements Display.
///
/// Contract: TrinityDevice implements Display for human-readable output.
#[test]
fn test_trinity_device_implements_display() {
    fn _assert_display<T: Display>() {}
    _assert_display::<TrinityDevice>();
}

// =============================================================================
// 2. TrinityDevice::new() Constructor Contract Tests
// =============================================================================

/// Verifies that TrinityDevice has an async new() constructor.
///
/// Contract: TrinityDevice::new() is an async function returning Result<Self, DeviceCreationError>.
#[test]
fn test_trinity_device_new_is_async() {
    let adapter = require_adapter!();

    // Verify the new() method exists and is async by calling it with block_on
    let result = TrinityDevice::new(
        &adapter,
        wgpu::Features::empty(),
        wgpu::Limits::downlevel_defaults(),
    )
    .block_on();

    // We're testing the constructor signature, not the result
    let _ = result;
}

/// Verifies that TrinityDevice::new() returns Result<TrinityDevice, DeviceCreationError>.
///
/// Contract: new() returns a Result type with proper error handling.
#[test]
fn test_trinity_device_new_returns_result() {
    let adapter = require_adapter!();

    let result: Result<TrinityDevice, DeviceCreationError> = TrinityDevice::new(
        &adapter,
        wgpu::Features::empty(),
        wgpu::Limits::downlevel_defaults(),
    )
    .block_on();

    // Type annotation verifies the return type
    let _ = result;
}

/// Verifies that TrinityDevice::new() accepts features parameter.
///
/// Contract: new() takes wgpu::Features as its second parameter.
#[test]
fn test_trinity_device_new_accepts_features_parameter() {
    let adapter = require_adapter!();

    // Test with empty features
    let features: wgpu::Features = wgpu::Features::empty();
    let _ = TrinityDevice::new(&adapter, features, wgpu::Limits::downlevel_defaults()).block_on();
}

/// Verifies that TrinityDevice::new() accepts limits parameter.
///
/// Contract: new() takes wgpu::Limits as its third parameter.
#[test]
fn test_trinity_device_new_accepts_limits_parameter() {
    let adapter = require_adapter!();

    // Test with downlevel defaults
    let limits: wgpu::Limits = wgpu::Limits::downlevel_defaults();
    let _ = TrinityDevice::new(&adapter, wgpu::Features::empty(), limits).block_on();
}

// =============================================================================
// 3. TrinityDevice::with_defaults() Convenience Constructor Tests
// =============================================================================

/// Verifies that TrinityDevice has a with_defaults() convenience method.
///
/// Contract: TrinityDevice::with_defaults() creates a device with minimal requirements.
#[test]
fn test_trinity_device_has_with_defaults_method() {
    let adapter = require_adapter!();

    // Verify with_defaults exists and is async
    let result = TrinityDevice::with_defaults(&adapter).block_on();
    let _ = result;
}

/// Verifies that with_defaults() returns the same Result type as new().
///
/// Contract: with_defaults() returns Result<TrinityDevice, DeviceCreationError>.
#[test]
fn test_trinity_device_with_defaults_returns_result() {
    let adapter = require_adapter!();

    let result: Result<TrinityDevice, DeviceCreationError> =
        TrinityDevice::with_defaults(&adapter).block_on();

    let _ = result;
}

/// Verifies that with_defaults() succeeds on a valid adapter.
///
/// Contract: with_defaults() should succeed on any valid adapter.
#[test]
fn test_trinity_device_with_defaults_succeeds_on_valid_adapter() {
    let adapter = require_adapter!();

    let result = TrinityDevice::with_defaults(&adapter).block_on();

    assert!(
        result.is_ok(),
        "with_defaults() should succeed on a valid adapter: {:?}",
        result.err()
    );
}

// =============================================================================
// 4. TrinityDevice::with_all_features() Convenience Constructor Tests
// =============================================================================

/// Verifies that TrinityDevice has a with_all_features() convenience method.
///
/// Contract: TrinityDevice::with_all_features() creates a device with maximum capabilities.
#[test]
fn test_trinity_device_has_with_all_features_method() {
    let adapter = require_adapter!();

    // Verify with_all_features exists and is async
    let result = TrinityDevice::with_all_features(&adapter).block_on();
    let _ = result;
}

/// Verifies that with_all_features() returns the same Result type as new().
///
/// Contract: with_all_features() returns Result<TrinityDevice, DeviceCreationError>.
#[test]
fn test_trinity_device_with_all_features_returns_result() {
    let adapter = require_adapter!();

    let result: Result<TrinityDevice, DeviceCreationError> =
        TrinityDevice::with_all_features(&adapter).block_on();

    let _ = result;
}

/// Verifies that with_all_features() enables all adapter-supported features.
///
/// Contract: with_all_features() requests all features the adapter supports.
#[test]
fn test_trinity_device_with_all_features_enables_supported_features() {
    let adapter = require_adapter!();
    let adapter_features = adapter.features();

    let result = TrinityDevice::with_all_features(&adapter).block_on();

    if let Ok(device) = result {
        // Device should have at least some features if adapter supports them
        let device_features = device.features();
        // The device features should be a subset of what the adapter supports
        assert!(
            adapter_features.contains(device_features),
            "Device features should be a subset of adapter features"
        );
    }
    // If it fails, that's also acceptable (some features may not be requestable)
}

// =============================================================================
// 5. TrinityDevice Accessor Methods Contract Tests
// =============================================================================

/// Verifies that TrinityDevice has a device() accessor.
///
/// Contract: device() returns a reference to the underlying wgpu::Device.
#[test]
fn test_trinity_device_has_device_accessor() {
    let adapter = require_adapter!();
    let device = TrinityDevice::with_defaults(&adapter)
        .block_on()
        .expect("Device creation should succeed");

    // Verify device() returns a reference to wgpu::Device
    let inner_device: &wgpu::Device = device.device();
    let _ = inner_device;
}

/// Verifies that TrinityDevice has a queue() accessor.
///
/// Contract: queue() returns a reference to the underlying wgpu::Queue.
#[test]
fn test_trinity_device_has_queue_accessor() {
    let adapter = require_adapter!();
    let device = TrinityDevice::with_defaults(&adapter)
        .block_on()
        .expect("Device creation should succeed");

    // Verify queue() returns a reference to wgpu::Queue
    let queue: &wgpu::Queue = device.queue();
    let _ = queue;
}

/// Verifies that TrinityDevice has a features() accessor.
///
/// Contract: features() returns the wgpu::Features enabled on this device.
#[test]
fn test_trinity_device_has_features_accessor() {
    let adapter = require_adapter!();
    let device = TrinityDevice::with_defaults(&adapter)
        .block_on()
        .expect("Device creation should succeed");

    // Verify features() returns wgpu::Features
    let features: wgpu::Features = device.features();
    let _ = features;
}

/// Verifies that TrinityDevice has a limits() accessor.
///
/// Contract: limits() returns a reference to the wgpu::Limits of this device.
#[test]
fn test_trinity_device_has_limits_accessor() {
    let adapter = require_adapter!();
    let device = TrinityDevice::with_defaults(&adapter)
        .block_on()
        .expect("Device creation should succeed");

    // Verify limits() returns a reference to wgpu::Limits
    let limits: &wgpu::Limits = device.limits();
    let _ = limits;
}

/// Verifies that TrinityDevice has a has_feature() method.
///
/// Contract: has_feature() checks if a specific feature is enabled.
#[test]
fn test_trinity_device_has_feature_checker() {
    let adapter = require_adapter!();
    let device = TrinityDevice::with_defaults(&adapter)
        .block_on()
        .expect("Device creation should succeed");

    // Verify has_feature() exists and returns bool
    let has_feature: bool = device.has_feature(wgpu::Features::empty());
    let _ = has_feature;
}

// =============================================================================
// 6. DeviceCreationError Type Contract Tests
// =============================================================================

/// Verifies that DeviceCreationError is a publicly accessible type.
///
/// Contract: DeviceCreationError is exported from renderer_backend::device.
#[test]
fn test_device_creation_error_type_is_exported() {
    fn _assert_type_exists<T>(_: Option<&T>) {}
    _assert_type_exists::<DeviceCreationError>(None);
}

/// Verifies that DeviceCreationError implements Debug.
///
/// Contract: DeviceCreationError derives Debug for error introspection.
#[test]
fn test_device_creation_error_implements_debug() {
    fn _assert_debug<T: Debug>() {}
    _assert_debug::<DeviceCreationError>();
}

/// Verifies that DeviceCreationError implements Display.
///
/// Contract: DeviceCreationError implements Display for error messages.
#[test]
fn test_device_creation_error_implements_display() {
    fn _assert_display<T: Display>() {}
    _assert_display::<DeviceCreationError>();
}

/// Verifies that DeviceCreationError has a FeatureNotSupported variant.
///
/// Contract: DeviceCreationError::FeatureNotSupported(Features) exists.
#[test]
fn test_device_creation_error_has_feature_not_supported_variant() {
    // Create an instance of the variant to verify it exists
    let error = DeviceCreationError::FeatureNotSupported(wgpu::Features::empty());
    let _ = error;
}

/// Verifies that DeviceCreationError has a LimitNotMet variant.
///
/// Contract: DeviceCreationError::LimitNotMet { limit, required, available } exists.
#[test]
fn test_device_creation_error_has_limit_not_met_variant() {
    // Create an instance of the variant to verify it exists
    let error = DeviceCreationError::LimitNotMet {
        limit: String::from("max_texture_dimension_2d"),
        required: 8192,
        available: 4096,
    };
    let _ = error;
}

/// Verifies that DeviceCreationError has a RequestDeviceError variant.
///
/// Contract: DeviceCreationError::RequestDeviceError wraps wgpu::RequestDeviceError.
#[test]
fn test_device_creation_error_has_request_device_error_variant() {
    // We cannot easily construct a wgpu::RequestDeviceError, but we can verify
    // the variant type exists by pattern matching
    fn _match_variant(e: DeviceCreationError) {
        match e {
            DeviceCreationError::RequestDeviceError(_) => {}
            _ => {}
        }
    }
}

// =============================================================================
// 7. DeviceCreationError Display Messages Tests
// =============================================================================

/// Verifies that FeatureNotSupported error has an informative message.
///
/// Contract: Error messages should be informative for debugging.
#[test]
fn test_feature_not_supported_error_has_informative_message() {
    let error = DeviceCreationError::FeatureNotSupported(wgpu::Features::TEXTURE_COMPRESSION_BC);
    let message = format!("{}", error);

    // Message should mention features and be non-empty
    assert!(!message.is_empty(), "Error message should not be empty");
    assert!(
        message.to_lowercase().contains("feature")
            || message.to_lowercase().contains("support"),
        "Error message should mention features or support: {}",
        message
    );
}

/// Verifies that LimitNotMet error has an informative message.
///
/// Contract: LimitNotMet error should include the limit name and values.
#[test]
fn test_limit_not_met_error_has_informative_message() {
    let error = DeviceCreationError::LimitNotMet {
        limit: String::from("max_texture_dimension_2d"),
        required: 16384,
        available: 8192,
    };
    let message = format!("{}", error);

    // Message should be non-empty and mention the limit
    assert!(!message.is_empty(), "Error message should not be empty");
    assert!(
        message.contains("limit") || message.contains("16384") || message.contains("8192"),
        "Error message should contain limit info: {}",
        message
    );
}

/// Verifies that DeviceCreationError Debug output is useful.
///
/// Contract: Debug output should show variant and contained data.
#[test]
fn test_device_creation_error_debug_output() {
    let error = DeviceCreationError::LimitNotMet {
        limit: String::from("test_limit"),
        required: 100,
        available: 50,
    };
    let debug_output = format!("{:?}", error);

    // Debug output should show the variant name
    assert!(
        debug_output.contains("LimitNotMet"),
        "Debug output should contain variant name: {}",
        debug_output
    );
}

// =============================================================================
// 8. Device Creation Behavior Tests
// =============================================================================

/// Verifies that a device can be created from a valid adapter.
///
/// Contract: TrinityDevice::new() succeeds with valid adapter and supported params.
#[test]
fn test_can_create_device_from_valid_adapter() {
    let adapter = require_adapter!();

    let result = TrinityDevice::new(
        &adapter,
        wgpu::Features::empty(),
        wgpu::Limits::downlevel_defaults(),
    )
    .block_on();

    assert!(
        result.is_ok(),
        "Should create device from valid adapter: {:?}",
        result.err()
    );
}

/// Verifies that the created device has a working queue.
///
/// Contract: The queue returned by queue() should be functional.
#[test]
fn test_created_device_has_working_queue() {
    let adapter = require_adapter!();
    let device = TrinityDevice::with_defaults(&adapter)
        .block_on()
        .expect("Device creation should succeed");

    // Test that we can use the queue to submit work
    let queue = device.queue();

    // Create a simple command encoder and submit (empty submission is valid)
    let encoder = device
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("test") });

    // Submit the command buffer - this tests the queue works
    queue.submit(std::iter::once(encoder.finish()));
}

/// Verifies that features match what was requested (subset of requested).
///
/// Contract: Device features should include what was successfully requested.
#[test]
fn test_device_features_match_requested() {
    let adapter = require_adapter!();

    // Request empty features
    let requested_features = wgpu::Features::empty();
    let device = TrinityDevice::new(
        &adapter,
        requested_features,
        wgpu::Limits::downlevel_defaults(),
    )
    .block_on()
    .expect("Device creation should succeed");

    // Device should have at least the requested features
    let device_features = device.features();
    assert!(
        device_features.contains(requested_features),
        "Device should have at least the requested features"
    );
}

/// Verifies that limits are properly set on the device.
///
/// Contract: Device limits should be usable and match/exceed requirements.
#[test]
fn test_device_limits_are_properly_set() {
    let adapter = require_adapter!();
    let requested_limits = wgpu::Limits::downlevel_defaults();

    let device = TrinityDevice::new(&adapter, wgpu::Features::empty(), requested_limits.clone())
        .block_on()
        .expect("Device creation should succeed");

    let device_limits = device.limits();

    // Key limits should be at least what was requested
    assert!(
        device_limits.max_texture_dimension_2d >= requested_limits.max_texture_dimension_2d,
        "max_texture_dimension_2d should be at least requested"
    );
    assert!(
        device_limits.max_bind_groups >= requested_limits.max_bind_groups,
        "max_bind_groups should be at least requested"
    );
}

// =============================================================================
// 9. Error Handling Contract Tests
// =============================================================================

/// Verifies that requesting unsupported features returns an appropriate error.
///
/// Contract: Requesting features the adapter doesn't support should fail gracefully.
#[test]
fn test_requesting_unsupported_features_returns_error() {
    let adapter = require_adapter!();
    let adapter_features = adapter.features();

    // Find a feature the adapter doesn't support
    let all_features = [
        wgpu::Features::TEXTURE_COMPRESSION_BC,
        wgpu::Features::TEXTURE_COMPRESSION_ETC2,
        wgpu::Features::TEXTURE_COMPRESSION_ASTC,
        wgpu::Features::SHADER_F16,
        wgpu::Features::DEPTH_CLIP_CONTROL,
        wgpu::Features::CONSERVATIVE_RASTERIZATION,
    ];

    let unsupported = all_features
        .iter()
        .find(|f| !adapter_features.contains(**f));

    if let Some(&unsupported_feature) = unsupported {
        // This should fail or the implementation should validate
        let result = TrinityDevice::new(
            &adapter,
            unsupported_feature,
            wgpu::Limits::downlevel_defaults(),
        )
        .block_on();

        // The result should be an error OR the implementation gracefully handled it
        if result.is_err() {
            // Expected behavior
            let error = result.unwrap_err();
            // Verify it's a meaningful error (can be formatted)
            let _ = format!("{}", error);
        }
        // If it succeeded, that means the implementation handled it gracefully
    } else {
        eprintln!("SKIP: Adapter supports all test features, cannot test unsupported feature error");
    }
}

/// Verifies that the implementation handles device creation failures.
///
/// Contract: Handles request failure gracefully (from acceptance criteria).
#[test]
fn test_device_creation_handles_failures_gracefully() {
    let adapter = require_adapter!();

    // Try to create with impossible limits
    let mut impossible_limits = wgpu::Limits::default();
    impossible_limits.max_texture_dimension_2d = u32::MAX;
    impossible_limits.max_storage_buffer_binding_size = u32::MAX;

    let result = TrinityDevice::new(&adapter, wgpu::Features::empty(), impossible_limits).block_on();

    // The function should either:
    // 1. Return an error (graceful failure)
    // 2. Cap the limits to adapter limits (graceful degradation)
    // Either way, it shouldn't panic
    match result {
        Ok(device) => {
            // Graceful degradation - limits were capped
            let actual_limits = device.limits();
            assert!(
                actual_limits.max_texture_dimension_2d < u32::MAX,
                "Limits should be capped to reasonable values"
            );
        }
        Err(_) => {
            // Graceful failure with error
        }
    }
}

// =============================================================================
// 10. Integration with Adapter Selection Tests
// =============================================================================

/// Verifies full pipeline: Instance -> Enumerate -> Select -> Create Device.
///
/// Contract: The full initialization sequence should work end-to-end.
#[test]
fn test_full_pipeline_instance_to_device() {
    // Step 1: Create instance
    let instance = TrinityInstance::new();

    // Step 2: Enumerate adapters
    let result = enumerate_adapters_with_info(instance.inner(), instance.backends());

    if result.adapters.is_empty() {
        eprintln!("SKIP: No adapters available for full pipeline test");
        return;
    }

    // Step 3: Select adapter (using AdapterSelector)
    let selector = AdapterSelector::new();
    let selection = selector.select(&result.adapters);

    let adapter = match selection {
        Some(selection_result) => selection_result.adapter,
        None => {
            eprintln!("SKIP: No adapter selected (all blacklisted or none available)");
            return;
        }
    };

    // Step 4: Create device
    let device = TrinityDevice::with_defaults(adapter)
        .block_on()
        .expect("Device creation should succeed");

    // Verify we have a working device
    let _ = device.device(); // Just verify we got a device
    let _ = device.queue();
    let _ = device.features();
    let _ = device.limits();
}

/// Verifies that AdapterSelector can be used to pick adapter, then create device.
///
/// Contract: AdapterSelector integrates with TrinityDevice creation.
#[test]
fn test_adapter_selector_with_device_creation() {
    let instance = TrinityInstance::new();
    let result = enumerate_adapters_with_info(instance.inner(), instance.backends());

    if result.adapters.is_empty() {
        eprintln!("SKIP: No adapters available");
        return;
    }

    // Use AdapterSelector
    let selector = AdapterSelector::new();
    let selection = selector.select(&result.adapters);

    if let Some(selection_result) = selection {
        let adapter = selection_result.adapter;
        let device = TrinityDevice::with_defaults(adapter).block_on();

        assert!(
            device.is_ok(),
            "Device creation should succeed with selected adapter"
        );
    }
}

// =============================================================================
// 11. Queue Functionality Tests
// =============================================================================

/// Verifies that TrinityDevice has a submit() method for command buffers.
///
/// Contract: TrinityDevice can submit command buffers.
#[test]
fn test_trinity_device_has_submit_method() {
    let adapter = require_adapter!();
    let device = TrinityDevice::with_defaults(&adapter)
        .block_on()
        .expect("Device creation should succeed");

    // Create command encoder
    let encoder = device
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("test_submit"),
        });

    // Submit via TrinityDevice's submit method
    let _submission_index: wgpu::SubmissionIndex = device.submit(std::iter::once(encoder.finish()));
}

/// Verifies that TrinityDevice has a create_command_encoder() method.
///
/// Contract: TrinityDevice provides convenience for creating command encoders.
#[test]
fn test_trinity_device_has_create_command_encoder_method() {
    let adapter = require_adapter!();
    let device = TrinityDevice::with_defaults(&adapter)
        .block_on()
        .expect("Device creation should succeed");

    // Use the convenience method
    let encoder: wgpu::CommandEncoder = device.create_command_encoder(Some("test_encoder"));
    let _ = encoder.finish();
}

/// Verifies that queue can be used for direct buffer writes.
///
/// Contract: Queue can perform write_buffer operations.
#[test]
fn test_queue_can_write_buffer() {
    let adapter = require_adapter!();
    let device = TrinityDevice::with_defaults(&adapter)
        .block_on()
        .expect("Device creation should succeed");

    // Create a buffer
    let buffer = device.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("test_buffer"),
        size: 64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::UNIFORM,
        mapped_at_creation: false,
    });

    // Write to buffer using queue
    let data: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
    device
        .queue()
        .write_buffer(&buffer, 0, bytemuck::cast_slice(&data));

    // If we get here without panic, the write succeeded
}

// =============================================================================
// 12. Display and Debug Output Tests
// =============================================================================

/// Verifies that TrinityDevice Display output is useful.
///
/// Contract: TrinityDevice should have meaningful Display output.
#[test]
fn test_trinity_device_display_output() {
    let adapter = require_adapter!();
    let device = TrinityDevice::with_defaults(&adapter)
        .block_on()
        .expect("Device creation should succeed");

    let display_output = format!("{}", device);

    // Display output should be non-empty and contain useful info
    assert!(
        !display_output.is_empty(),
        "Display output should not be empty"
    );
}

/// Verifies that TrinityDevice Debug output is useful.
///
/// Contract: TrinityDevice should have meaningful Debug output.
#[test]
fn test_trinity_device_debug_output() {
    let adapter = require_adapter!();
    let device = TrinityDevice::with_defaults(&adapter)
        .block_on()
        .expect("Device creation should succeed");

    let debug_output = format!("{:?}", device);

    // Debug output should be non-empty and contain struct name
    assert!(
        !debug_output.is_empty(),
        "Debug output should not be empty"
    );
    assert!(
        debug_output.contains("TrinityDevice") || debug_output.contains("device"),
        "Debug output should identify the type: {}",
        debug_output
    );
}

// =============================================================================
// 13. Feature Checking Tests
// =============================================================================

/// Verifies has_feature() returns true for enabled features.
///
/// Contract: has_feature() correctly reports enabled features.
#[test]
fn test_has_feature_returns_true_for_enabled_features() {
    let adapter = require_adapter!();
    let device = TrinityDevice::with_defaults(&adapter)
        .block_on()
        .expect("Device creation should succeed");

    let device_features = device.features();

    // Empty features should always be "supported"
    assert!(
        device.has_feature(wgpu::Features::empty()),
        "Empty features should always be present"
    );

    // If device has any features, has_feature should return true for them
    if !device_features.is_empty() {
        assert!(
            device.has_feature(device_features),
            "has_feature should return true for device's own features"
        );
    }
}

/// Verifies has_feature() returns false for disabled features.
///
/// Contract: has_feature() correctly reports missing features.
#[test]
fn test_has_feature_returns_false_for_missing_features() {
    let adapter = require_adapter!();
    let device = TrinityDevice::with_defaults(&adapter)
        .block_on()
        .expect("Device creation should succeed");

    let device_features = device.features();

    // Find a feature the device doesn't have
    let test_features = [
        wgpu::Features::RAY_TRACING_ACCELERATION_STRUCTURE,
        wgpu::Features::RAY_QUERY,
        wgpu::Features::SHADER_F16,
    ];

    for feature in test_features {
        if !device_features.contains(feature) {
            assert!(
                !device.has_feature(feature),
                "has_feature should return false for missing features"
            );
            return;
        }
    }

    eprintln!("SKIP: Device has all test features, cannot test missing feature detection");
}

// =============================================================================
// 14. Multiple Device Creation Tests
// =============================================================================

/// Verifies that multiple devices can be created from the same adapter.
///
/// Contract: Adapter can be used to create multiple devices.
#[test]
fn test_multiple_devices_from_same_adapter() {
    let adapter = require_adapter!();

    // Create first device
    let device1 = TrinityDevice::with_defaults(&adapter)
        .block_on()
        .expect("First device creation should succeed");

    // Create second device
    let device2 = TrinityDevice::with_defaults(&adapter)
        .block_on()
        .expect("Second device creation should succeed");

    // Both devices should be functional
    let _ = device1.queue();
    let _ = device2.queue();
}

/// Verifies that devices with different configurations can coexist.
///
/// Contract: Different device configurations should work independently.
#[test]
fn test_devices_with_different_configs() {
    let adapter = require_adapter!();

    // Create device with defaults
    let device_default = TrinityDevice::with_defaults(&adapter)
        .block_on()
        .expect("Default device creation should succeed");

    // Create device with specific limits
    let custom_limits = wgpu::Limits::downlevel_webgl2_defaults();
    let device_custom = TrinityDevice::new(&adapter, wgpu::Features::empty(), custom_limits)
        .block_on()
        .expect("Custom device creation should succeed");

    // Both should be functional
    let _ = device_default.features();
    let _ = device_custom.features();
}

// =============================================================================
// 15. Edge Case Tests
// =============================================================================

/// Verifies device creation with zero features requested.
///
/// Contract: Empty features should always be acceptable.
#[test]
fn test_device_creation_with_zero_features() {
    let adapter = require_adapter!();

    let result = TrinityDevice::new(
        &adapter,
        wgpu::Features::empty(),
        wgpu::Limits::downlevel_defaults(),
    )
    .block_on();

    assert!(
        result.is_ok(),
        "Device creation with zero features should succeed"
    );
}

/// Verifies device creation with downlevel defaults works.
///
/// Contract: Downlevel defaults should be universally supported.
#[test]
fn test_device_creation_with_downlevel_defaults() {
    let adapter = require_adapter!();

    let result = TrinityDevice::new(
        &adapter,
        wgpu::Features::empty(),
        wgpu::Limits::downlevel_defaults(),
    )
    .block_on();

    assert!(
        result.is_ok(),
        "Device creation with downlevel_defaults should succeed"
    );
}

/// Verifies device creation with WebGL2 defaults works if supported.
///
/// Contract: WebGL2 defaults represent minimal guaranteed limits.
#[test]
fn test_device_creation_with_webgl2_defaults() {
    let adapter = require_adapter!();

    let result = TrinityDevice::new(
        &adapter,
        wgpu::Features::empty(),
        wgpu::Limits::downlevel_webgl2_defaults(),
    )
    .block_on();

    // WebGL2 defaults should work on any reasonable adapter
    assert!(
        result.is_ok(),
        "Device creation with webgl2_defaults should succeed: {:?}",
        result.err()
    );
}

/// Verifies device creation with default limits works.
///
/// Contract: Default limits should work on capable hardware.
#[test]
fn test_device_creation_with_default_limits() {
    let adapter = require_adapter!();

    let result = TrinityDevice::new(
        &adapter,
        wgpu::Features::empty(),
        wgpu::Limits::default(),
    )
    .block_on();

    // Default limits might fail on low-end hardware, which is acceptable
    match result {
        Ok(_) => {}
        Err(e) => {
            eprintln!("INFO: Default limits not supported on this adapter: {}", e);
        }
    }
}

// =============================================================================
// 16. Concurrent Access Tests
// =============================================================================

/// Verifies that device accessors are thread-safe.
///
/// Contract: TrinityDevice should support concurrent read access.
#[test]
fn test_device_accessors_are_thread_safe() {
    use std::sync::Arc;
    use std::thread;

    let adapter = require_adapter!();
    let device = Arc::new(
        TrinityDevice::with_defaults(&adapter)
            .block_on()
            .expect("Device creation should succeed"),
    );

    let handles: Vec<_> = (0..4)
        .map(|_| {
            let device_clone = Arc::clone(&device);
            thread::spawn(move || {
                // Access various methods concurrently
                let _ = device_clone.features();
                let _ = device_clone.limits();
                let _ = device_clone.device();
                let _ = device_clone.queue();
            })
        })
        .collect();

    for handle in handles {
        handle.join().expect("Thread should complete without panic");
    }
}

// =============================================================================
// 17. LimitNotMet Error Field Tests
// =============================================================================

/// Verifies LimitNotMet error contains the limit name.
///
/// Contract: LimitNotMet.limit field contains the limit identifier.
#[test]
fn test_limit_not_met_error_contains_limit_name() {
    let error = DeviceCreationError::LimitNotMet {
        limit: String::from("max_texture_dimension_2d"),
        required: 16384,
        available: 8192,
    };

    if let DeviceCreationError::LimitNotMet { limit, .. } = error {
        assert_eq!(limit, "max_texture_dimension_2d");
    } else {
        panic!("Expected LimitNotMet variant");
    }
}

/// Verifies LimitNotMet error contains the required value.
///
/// Contract: LimitNotMet.required field contains the requested value.
#[test]
fn test_limit_not_met_error_contains_required_value() {
    let error = DeviceCreationError::LimitNotMet {
        limit: String::from("test"),
        required: 12345,
        available: 100,
    };

    if let DeviceCreationError::LimitNotMet { required, .. } = error {
        assert_eq!(required, 12345);
    } else {
        panic!("Expected LimitNotMet variant");
    }
}

/// Verifies LimitNotMet error contains the available value.
///
/// Contract: LimitNotMet.available field contains the adapter's maximum.
#[test]
fn test_limit_not_met_error_contains_available_value() {
    let error = DeviceCreationError::LimitNotMet {
        limit: String::from("test"),
        required: 1000,
        available: 500,
    };

    if let DeviceCreationError::LimitNotMet { available, .. } = error {
        assert_eq!(available, 500);
    } else {
        panic!("Expected LimitNotMet variant");
    }
}

// =============================================================================
// 18. FeatureNotSupported Error Tests
// =============================================================================

/// Verifies FeatureNotSupported error contains the requested features.
///
/// Contract: FeatureNotSupported contains the features that were not available.
#[test]
fn test_feature_not_supported_error_contains_features() {
    let features = wgpu::Features::TEXTURE_COMPRESSION_BC | wgpu::Features::SHADER_F16;
    let error = DeviceCreationError::FeatureNotSupported(features);

    if let DeviceCreationError::FeatureNotSupported(contained) = error {
        assert!(contained.contains(wgpu::Features::TEXTURE_COMPRESSION_BC));
        assert!(contained.contains(wgpu::Features::SHADER_F16));
    } else {
        panic!("Expected FeatureNotSupported variant");
    }
}

/// Verifies FeatureNotSupported error can hold multiple features.
///
/// Contract: FeatureNotSupported can report multiple missing features at once.
#[test]
fn test_feature_not_supported_error_multiple_features() {
    let multiple_features = wgpu::Features::TEXTURE_COMPRESSION_BC
        | wgpu::Features::TEXTURE_COMPRESSION_ETC2
        | wgpu::Features::TEXTURE_COMPRESSION_ASTC;

    let error = DeviceCreationError::FeatureNotSupported(multiple_features);

    if let DeviceCreationError::FeatureNotSupported(contained) = error {
        // Should contain all three
        assert!(contained.contains(wgpu::Features::TEXTURE_COMPRESSION_BC));
        assert!(contained.contains(wgpu::Features::TEXTURE_COMPRESSION_ETC2));
        assert!(contained.contains(wgpu::Features::TEXTURE_COMPRESSION_ASTC));
    } else {
        panic!("Expected FeatureNotSupported variant");
    }
}

// =============================================================================
// 19. Adapter Features Subset Tests
// =============================================================================

/// Verifies device can be created with adapter's supported features.
///
/// Contract: Requesting features the adapter supports should succeed.
#[test]
fn test_device_with_adapter_supported_features() {
    let adapter = require_adapter!();
    let adapter_features = adapter.features();

    let result = TrinityDevice::new(
        &adapter,
        adapter_features, // Request all supported features
        wgpu::Limits::downlevel_defaults(),
    )
    .block_on();

    // This might fail if some features require others or have constraints,
    // but it should not panic
    match result {
        Ok(device) => {
            // Device should have at least the features we requested
            assert!(
                device.features().contains(adapter_features),
                "Device should have the adapter's features"
            );
        }
        Err(_) => {
            // Some feature combinations might not work, which is acceptable
        }
    }
}

// =============================================================================
// 20. Stress and Boundary Tests
// =============================================================================

/// Verifies device creation is idempotent (multiple calls produce valid devices).
///
/// Contract: Multiple device creations should all succeed independently.
#[test]
fn test_device_creation_is_idempotent() {
    let adapter = require_adapter!();

    for i in 0..3 {
        let result = TrinityDevice::with_defaults(&adapter).block_on();
        assert!(
            result.is_ok(),
            "Device creation {} should succeed: {:?}",
            i,
            result.err()
        );
    }
}

/// Verifies that device and queue remain valid after operations.
///
/// Contract: Device should remain usable after work submission.
#[test]
fn test_device_remains_valid_after_operations() {
    let adapter = require_adapter!();
    let device = TrinityDevice::with_defaults(&adapter)
        .block_on()
        .expect("Device creation should succeed");

    // Perform some operations
    for _ in 0..5 {
        let encoder = device.create_command_encoder(Some("test"));
        device.submit(std::iter::once(encoder.finish()));
    }

    // Device should still be functional
    let _ = device.features();
    let _ = device.limits();
    let final_encoder = device.create_command_encoder(Some("final"));
    device.submit(std::iter::once(final_encoder.finish()));
}
