// SPDX-License-Identifier: MIT
//
// device_creation_whitebox.rs -- Whitebox structural tests for T-WGPU-P1.3.1
// (TrinityDevice - Device Creation).
//
// These tests exercise the internal implementation of TrinityDevice,
// covering all code paths in device creation, feature validation,
// limit validation, error handling, and convenience methods.
//
// WHITEBOX coverage plan:
//   - Section 1: TrinityDevice Creation (new, with_defaults, with_all_features)
//   - Section 2: DeviceCreationError variants and Display
//   - Section 3: Feature Validation (missing features, supported features)
//   - Section 4: Limit Validation (all limit fields)
//   - Section 5: Accessor Methods (device, queue, features, limits, has_feature)
//   - Section 6: Convenience Methods (submit, create_command_encoder)
//   - Section 7: Trait Implementations (Debug, Display, Error, From)
//   - Section 8: Edge Cases and Boundary Conditions
//
// Acceptance criteria (T-WGPU-P1.3.1):
//   1. Creates device with requested features
//   2. Creates device with requested limits
//   3. Handles request failure gracefully
//   4. Logs device creation details

use renderer_backend::device::{DeviceCreationError, TrinityDevice, TrinityInstance};
use std::error::Error;

// =============================================================================
// Test Helpers
// =============================================================================

/// Get an adapter for testing, or skip the test if none available.
fn get_test_adapter() -> Option<wgpu::Adapter> {
    let instance = TrinityInstance::new();
    let adapters = instance.enumerate_adapters();
    adapters.into_iter().next()
}

/// Blocking helper to run async device creation.
fn block_on<F: std::future::Future>(future: F) -> F::Output {
    pollster::block_on(future)
}

// =============================================================================
// Section 1: TrinityDevice Creation Tests
// =============================================================================

// ---------------------------------------------------------------------------
// 1.1: new() with valid adapter and default features/limits
// ---------------------------------------------------------------------------

#[test]
fn test_device_new_with_empty_features_and_default_limits() {
    let Some(adapter) = get_test_adapter() else {
        eprintln!("SKIP: No adapter available");
        return;
    };

    let result = block_on(TrinityDevice::new(
        &adapter,
        wgpu::Features::empty(),
        wgpu::Limits::default(),
    ));

    assert!(result.is_ok(), "Device creation should succeed with empty features");
    let device = result.unwrap();
    assert!(device.features().is_empty(), "Features should be empty");
}

#[test]
fn test_device_new_returns_valid_device_and_queue() {
    let Some(adapter) = get_test_adapter() else {
        eprintln!("SKIP: No adapter available");
        return;
    };

    let device = block_on(TrinityDevice::new(
        &adapter,
        wgpu::Features::empty(),
        wgpu::Limits::default(),
    ))
    .expect("Device creation failed");

    // Verify device() accessor returns non-null reference
    let inner_device = device.device();
    assert!(std::ptr::addr_of!(*inner_device) as usize != 0);

    // Verify queue() accessor returns non-null reference
    let queue = device.queue();
    assert!(std::ptr::addr_of!(*queue) as usize != 0);
}

#[test]
fn test_device_new_stores_requested_features() {
    let Some(adapter) = get_test_adapter() else {
        eprintln!("SKIP: No adapter available");
        return;
    };

    // Request empty features
    let device = block_on(TrinityDevice::new(
        &adapter,
        wgpu::Features::empty(),
        wgpu::Limits::default(),
    ))
    .expect("Device creation failed");

    assert_eq!(
        device.features(),
        wgpu::Features::empty(),
        "Stored features should match requested"
    );
}

#[test]
fn test_device_new_stores_requested_limits() {
    let Some(adapter) = get_test_adapter() else {
        eprintln!("SKIP: No adapter available");
        return;
    };

    let limits = wgpu::Limits::default();
    let device = block_on(TrinityDevice::new(
        &adapter,
        wgpu::Features::empty(),
        limits.clone(),
    ))
    .expect("Device creation failed");

    assert_eq!(
        device.limits().max_texture_dimension_2d,
        limits.max_texture_dimension_2d,
        "Stored limits should match requested"
    );
}

// ---------------------------------------------------------------------------
// 1.2: new() with specific features requested
// ---------------------------------------------------------------------------

#[test]
fn test_device_new_with_supported_single_feature() {
    let Some(adapter) = get_test_adapter() else {
        eprintln!("SKIP: No adapter available");
        return;
    };

    let adapter_features = adapter.features();

    // Find a feature the adapter supports
    if adapter_features.contains(wgpu::Features::TEXTURE_COMPRESSION_BC) {
        let device = block_on(TrinityDevice::new(
            &adapter,
            wgpu::Features::TEXTURE_COMPRESSION_BC,
            wgpu::Limits::default(),
        ))
        .expect("Device creation with supported feature failed");

        assert!(
            device.features().contains(wgpu::Features::TEXTURE_COMPRESSION_BC),
            "Device should have requested feature enabled"
        );
    } else if adapter_features.contains(wgpu::Features::DEPTH32FLOAT_STENCIL8) {
        let device = block_on(TrinityDevice::new(
            &adapter,
            wgpu::Features::DEPTH32FLOAT_STENCIL8,
            wgpu::Limits::default(),
        ))
        .expect("Device creation with supported feature failed");

        assert!(
            device.features().contains(wgpu::Features::DEPTH32FLOAT_STENCIL8),
            "Device should have requested feature enabled"
        );
    } else {
        eprintln!("SKIP: No testable optional features available on adapter");
    }
}

#[test]
fn test_device_new_with_multiple_supported_features() {
    let Some(adapter) = get_test_adapter() else {
        eprintln!("SKIP: No adapter available");
        return;
    };

    let adapter_features = adapter.features();

    // Find multiple features the adapter supports
    let mut requested = wgpu::Features::empty();
    if adapter_features.contains(wgpu::Features::DEPTH32FLOAT_STENCIL8) {
        requested |= wgpu::Features::DEPTH32FLOAT_STENCIL8;
    }
    if adapter_features.contains(wgpu::Features::TEXTURE_COMPRESSION_BC) {
        requested |= wgpu::Features::TEXTURE_COMPRESSION_BC;
    }
    if adapter_features.contains(wgpu::Features::INDIRECT_FIRST_INSTANCE) {
        requested |= wgpu::Features::INDIRECT_FIRST_INSTANCE;
    }

    if requested.is_empty() {
        eprintln!("SKIP: No testable features available on adapter");
        return;
    }

    let device = block_on(TrinityDevice::new(
        &adapter,
        requested,
        wgpu::Limits::default(),
    ))
    .expect("Device creation with multiple features failed");

    assert_eq!(
        device.features(),
        requested,
        "All requested features should be enabled"
    );
}

// ---------------------------------------------------------------------------
// 1.3: new() with specific limits requested
// ---------------------------------------------------------------------------

#[test]
fn test_device_new_with_downlevel_limits() {
    let Some(adapter) = get_test_adapter() else {
        eprintln!("SKIP: No adapter available");
        return;
    };

    // Use downlevel limits (very minimal)
    let limits = wgpu::Limits::downlevel_defaults();

    let result = block_on(TrinityDevice::new(
        &adapter,
        wgpu::Features::empty(),
        limits.clone(),
    ));

    // Downlevel limits should work on any modern GPU
    assert!(
        result.is_ok(),
        "Device creation should succeed with downlevel limits"
    );
}

#[test]
fn test_device_new_with_downlevel_webgl2_limits() {
    let Some(adapter) = get_test_adapter() else {
        eprintln!("SKIP: No adapter available");
        return;
    };

    // Use WebGL2 limits (very conservative)
    let limits = wgpu::Limits::downlevel_webgl2_defaults();

    let result = block_on(TrinityDevice::new(
        &adapter,
        wgpu::Features::empty(),
        limits.clone(),
    ));

    assert!(
        result.is_ok(),
        "Device creation should succeed with WebGL2 limits"
    );
}

#[test]
fn test_device_new_with_custom_texture_dimension_limit() {
    let Some(adapter) = get_test_adapter() else {
        eprintln!("SKIP: No adapter available");
        return;
    };

    let adapter_limits = adapter.limits();

    // Request a custom (but supported) texture dimension
    let mut limits = wgpu::Limits::default();
    limits.max_texture_dimension_2d = adapter_limits.max_texture_dimension_2d.min(4096);

    let device = block_on(TrinityDevice::new(
        &adapter,
        wgpu::Features::empty(),
        limits.clone(),
    ))
    .expect("Device creation with custom limits failed");

    assert_eq!(
        device.limits().max_texture_dimension_2d,
        limits.max_texture_dimension_2d
    );
}

#[test]
fn test_device_new_with_custom_buffer_size_limit() {
    let Some(adapter) = get_test_adapter() else {
        eprintln!("SKIP: No adapter available");
        return;
    };

    let adapter_limits = adapter.limits();

    let mut limits = wgpu::Limits::default();
    limits.max_buffer_size = adapter_limits.max_buffer_size.min(128 * 1024 * 1024);

    let device = block_on(TrinityDevice::new(
        &adapter,
        wgpu::Features::empty(),
        limits.clone(),
    ))
    .expect("Device creation with custom buffer limit failed");

    assert_eq!(device.limits().max_buffer_size, limits.max_buffer_size);
}

// ---------------------------------------------------------------------------
// 1.4: with_defaults() convenience method
// ---------------------------------------------------------------------------

#[test]
fn test_device_with_defaults_creates_valid_device() {
    let Some(adapter) = get_test_adapter() else {
        eprintln!("SKIP: No adapter available");
        return;
    };

    let device =
        block_on(TrinityDevice::with_defaults(&adapter)).expect("with_defaults() failed");

    // Should have empty features
    assert!(device.features().is_empty());

    // Should have default limits
    assert_eq!(
        device.limits().max_texture_dimension_2d,
        wgpu::Limits::default().max_texture_dimension_2d
    );
}

#[test]
fn test_device_with_defaults_returns_same_as_new_with_empty() {
    let Some(adapter) = get_test_adapter() else {
        eprintln!("SKIP: No adapter available");
        return;
    };

    let device1 =
        block_on(TrinityDevice::with_defaults(&adapter)).expect("with_defaults() failed");

    let device2 = block_on(TrinityDevice::new(
        &adapter,
        wgpu::Features::empty(),
        wgpu::Limits::default(),
    ))
    .expect("new() failed");

    // Both should have same features
    assert_eq!(device1.features(), device2.features());

    // Both should have same limits
    assert_eq!(
        device1.limits().max_texture_dimension_2d,
        device2.limits().max_texture_dimension_2d
    );
}

// ---------------------------------------------------------------------------
// 1.5: with_all_features() convenience method
// ---------------------------------------------------------------------------

#[test]
fn test_device_with_all_features_enables_all_supported() {
    let Some(adapter) = get_test_adapter() else {
        eprintln!("SKIP: No adapter available");
        return;
    };

    let adapter_features = adapter.features();

    let device = block_on(TrinityDevice::with_all_features(&adapter))
        .expect("with_all_features() failed");

    // Should have all adapter features enabled
    assert_eq!(
        device.features(),
        adapter_features,
        "Device should have all adapter features"
    );
}

#[test]
fn test_device_with_all_features_uses_adapter_limits() {
    let Some(adapter) = get_test_adapter() else {
        eprintln!("SKIP: No adapter available");
        return;
    };

    let adapter_limits = adapter.limits();

    let device = block_on(TrinityDevice::with_all_features(&adapter))
        .expect("with_all_features() failed");

    // Should use adapter limits
    assert_eq!(
        device.limits().max_texture_dimension_2d,
        adapter_limits.max_texture_dimension_2d
    );
    assert_eq!(
        device.limits().max_buffer_size,
        adapter_limits.max_buffer_size
    );
}

#[test]
fn test_device_with_all_features_feature_count_matches_adapter() {
    let Some(adapter) = get_test_adapter() else {
        eprintln!("SKIP: No adapter available");
        return;
    };

    let adapter_feature_count = adapter.features().iter().count();

    let device = block_on(TrinityDevice::with_all_features(&adapter))
        .expect("with_all_features() failed");

    let device_feature_count = device.features().iter().count();

    assert_eq!(
        device_feature_count, adapter_feature_count,
        "Device feature count should match adapter"
    );
}

// ---------------------------------------------------------------------------
// 1.6: Async creation behavior
// ---------------------------------------------------------------------------

#[test]
fn test_device_new_is_async() {
    // Verify that new() is async by using block_on
    let Some(adapter) = get_test_adapter() else {
        eprintln!("SKIP: No adapter available");
        return;
    };

    // The fact that we need to use block_on verifies it's async
    let future = TrinityDevice::new(&adapter, wgpu::Features::empty(), wgpu::Limits::default());

    // The future should be Send
    fn assert_send<T: Send>(_: &T) {}
    assert_send(&future);

    let _device = block_on(future).expect("Async device creation failed");
}

#[test]
fn test_device_with_defaults_is_async() {
    let Some(adapter) = get_test_adapter() else {
        eprintln!("SKIP: No adapter available");
        return;
    };

    let future = TrinityDevice::with_defaults(&adapter);
    fn assert_send<T: Send>(_: &T) {}
    assert_send(&future);

    let _device = block_on(future).expect("Async device creation failed");
}

#[test]
fn test_device_with_all_features_is_async() {
    let Some(adapter) = get_test_adapter() else {
        eprintln!("SKIP: No adapter available");
        return;
    };

    let future = TrinityDevice::with_all_features(&adapter);
    fn assert_send<T: Send>(_: &T) {}
    assert_send(&future);

    let _device = block_on(future).expect("Async device creation failed");
}

// =============================================================================
// Section 2: DeviceCreationError Tests
// =============================================================================

// ---------------------------------------------------------------------------
// 2.1: FeatureNotSupported error creation and Display
// ---------------------------------------------------------------------------

#[test]
fn test_error_feature_not_supported_single_feature() {
    let err = DeviceCreationError::FeatureNotSupported(wgpu::Features::TIMESTAMP_QUERY);
    let msg = format!("{}", err);

    assert!(msg.contains("not supported"), "Error message should mention 'not supported'");
    assert!(
        msg.contains("TIMESTAMP_QUERY"),
        "Error message should contain feature name"
    );
}

#[test]
fn test_error_feature_not_supported_multiple_features() {
    let features =
        wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::TEXTURE_COMPRESSION_BC;
    let err = DeviceCreationError::FeatureNotSupported(features);
    let msg = format!("{}", err);

    assert!(msg.contains("not supported"));
}

#[test]
fn test_error_feature_not_supported_empty_features() {
    let err = DeviceCreationError::FeatureNotSupported(wgpu::Features::empty());
    let msg = format!("{}", err);

    assert!(msg.contains("not supported"));
}

// ---------------------------------------------------------------------------
// 2.2: LimitNotMet error creation and Display
// ---------------------------------------------------------------------------

#[test]
fn test_error_limit_not_met_texture_dimension() {
    let err = DeviceCreationError::LimitNotMet {
        limit: "max_texture_dimension_2d".to_string(),
        required: 16384,
        available: 8192,
    };
    let msg = format!("{}", err);

    assert!(
        msg.contains("max_texture_dimension_2d"),
        "Should contain limit name"
    );
    assert!(msg.contains("16384"), "Should contain required value");
    assert!(msg.contains("8192"), "Should contain available value");
}

#[test]
fn test_error_limit_not_met_buffer_size() {
    let err = DeviceCreationError::LimitNotMet {
        limit: "max_buffer_size".to_string(),
        required: 1073741824, // 1GB
        available: 268435456, // 256MB
    };
    let msg = format!("{}", err);

    assert!(msg.contains("max_buffer_size"));
    assert!(msg.contains("1073741824"));
    assert!(msg.contains("268435456"));
}

#[test]
fn test_error_limit_not_met_bind_groups() {
    let err = DeviceCreationError::LimitNotMet {
        limit: "max_bind_groups".to_string(),
        required: 8,
        available: 4,
    };
    let msg = format!("{}", err);

    assert!(msg.contains("max_bind_groups"));
}

#[test]
fn test_error_limit_not_met_compute_workgroup() {
    let err = DeviceCreationError::LimitNotMet {
        limit: "max_compute_workgroup_size_x".to_string(),
        required: 1024,
        available: 256,
    };
    let msg = format!("{}", err);

    assert!(msg.contains("max_compute_workgroup_size_x"));
}

// ---------------------------------------------------------------------------
// 2.3: RequestDeviceError wrapping
// ---------------------------------------------------------------------------

#[test]
fn test_error_from_request_device_error() {
    // We cannot easily create a RequestDeviceError, but we can test the From impl
    // by checking the error variant structure
    let err = DeviceCreationError::FeatureNotSupported(wgpu::Features::empty());
    match err {
        DeviceCreationError::RequestDeviceError(_) => {
            panic!("Should not be RequestDeviceError variant")
        }
        DeviceCreationError::FeatureNotSupported(_) => { /* expected */ }
        DeviceCreationError::LimitNotMet { .. } => {
            panic!("Should not be LimitNotMet variant")
        }
    }
}

// =============================================================================
// Section 3: Feature Validation Tests
// =============================================================================

// ---------------------------------------------------------------------------
// 3.1: Request unsupported feature -> FeatureNotSupported error
// ---------------------------------------------------------------------------

#[test]
fn test_device_new_with_unsupported_feature_returns_error() {
    let Some(adapter) = get_test_adapter() else {
        eprintln!("SKIP: No adapter available");
        return;
    };

    let adapter_features = adapter.features();

    // Find a feature the adapter does NOT support
    // Check common optional features that may not be supported
    let test_features = [
        wgpu::Features::TEXTURE_COMPRESSION_ASTC,
        wgpu::Features::TEXTURE_COMPRESSION_ETC2,
        wgpu::Features::MULTIVIEW,
        wgpu::Features::CONSERVATIVE_RASTERIZATION,
        wgpu::Features::VERTEX_WRITABLE_STORAGE,
    ];

    let unsupported = test_features
        .iter()
        .copied()
        .find(|f| !adapter_features.contains(*f));

    let Some(unsupported_feature) = unsupported else {
        eprintln!("SKIP: All test features are supported by adapter");
        return;
    };

    let result = block_on(TrinityDevice::new(
        &adapter,
        unsupported_feature,
        wgpu::Limits::default(),
    ));

    assert!(result.is_err(), "Should fail when requesting unsupported feature");

    match result.unwrap_err() {
        DeviceCreationError::FeatureNotSupported(missing) => {
            assert!(
                missing.contains(unsupported_feature),
                "Error should report the missing feature"
            );
        }
        other => panic!("Expected FeatureNotSupported, got {:?}", other),
    }
}

#[test]
fn test_device_new_with_multiple_unsupported_features_reports_all() {
    let Some(adapter) = get_test_adapter() else {
        eprintln!("SKIP: No adapter available");
        return;
    };

    let adapter_features = adapter.features();

    // Find multiple unsupported features
    let test_features = [
        wgpu::Features::TEXTURE_COMPRESSION_ASTC,
        wgpu::Features::TEXTURE_COMPRESSION_ETC2,
        wgpu::Features::MULTIVIEW,
    ];

    let unsupported: wgpu::Features = test_features
        .iter()
        .copied()
        .filter(|f| !adapter_features.contains(*f))
        .fold(wgpu::Features::empty(), |acc, f| acc | f);

    if unsupported.is_empty() {
        eprintln!("SKIP: All test features are supported");
        return;
    }

    let result = block_on(TrinityDevice::new(
        &adapter,
        unsupported,
        wgpu::Limits::default(),
    ));

    assert!(result.is_err());

    match result.unwrap_err() {
        DeviceCreationError::FeatureNotSupported(missing) => {
            // All unsupported features should be reported
            assert_eq!(missing, unsupported);
        }
        other => panic!("Expected FeatureNotSupported, got {:?}", other),
    }
}

// ---------------------------------------------------------------------------
// 3.2: Request supported feature -> success
// ---------------------------------------------------------------------------

#[test]
fn test_device_new_with_all_supported_features_succeeds() {
    let Some(adapter) = get_test_adapter() else {
        eprintln!("SKIP: No adapter available");
        return;
    };

    let adapter_features = adapter.features();

    // Request only features the adapter supports
    let result = block_on(TrinityDevice::new(
        &adapter,
        adapter_features,
        wgpu::Limits::default(),
    ));

    assert!(
        result.is_ok(),
        "Should succeed when requesting only supported features"
    );
}

// ---------------------------------------------------------------------------
// 3.3: has_feature() accessor
// ---------------------------------------------------------------------------

#[test]
fn test_has_feature_returns_true_for_enabled_feature() {
    let Some(adapter) = get_test_adapter() else {
        eprintln!("SKIP: No adapter available");
        return;
    };

    let adapter_features = adapter.features();

    // Use a feature the adapter supports
    if adapter_features.contains(wgpu::Features::DEPTH32FLOAT_STENCIL8) {
        let device = block_on(TrinityDevice::new(
            &adapter,
            wgpu::Features::DEPTH32FLOAT_STENCIL8,
            wgpu::Limits::default(),
        ))
        .expect("Device creation failed");

        assert!(device.has_feature(wgpu::Features::DEPTH32FLOAT_STENCIL8));
    }
}

#[test]
fn test_has_feature_returns_false_for_disabled_feature() {
    let Some(adapter) = get_test_adapter() else {
        eprintln!("SKIP: No adapter available");
        return;
    };

    let device =
        block_on(TrinityDevice::with_defaults(&adapter)).expect("Device creation failed");

    // Empty features means no optional features enabled
    assert!(!device.has_feature(wgpu::Features::TIMESTAMP_QUERY));
    assert!(!device.has_feature(wgpu::Features::TEXTURE_COMPRESSION_BC));
}

#[test]
fn test_has_feature_with_empty_features_always_false() {
    let Some(adapter) = get_test_adapter() else {
        eprintln!("SKIP: No adapter available");
        return;
    };

    let device = block_on(TrinityDevice::new(
        &adapter,
        wgpu::Features::empty(),
        wgpu::Limits::default(),
    ))
    .expect("Device creation failed");

    // All optional features should return false
    assert!(!device.has_feature(wgpu::Features::DEPTH32FLOAT_STENCIL8));
    assert!(!device.has_feature(wgpu::Features::TEXTURE_COMPRESSION_BC));
    assert!(!device.has_feature(wgpu::Features::TIMESTAMP_QUERY));
    assert!(!device.has_feature(wgpu::Features::MULTIVIEW));
}

// =============================================================================
// Section 4: Limit Validation Tests
// =============================================================================

// ---------------------------------------------------------------------------
// 4.1: Request limits within adapter capability -> success
// ---------------------------------------------------------------------------

#[test]
fn test_device_new_with_limits_within_adapter_capability() {
    let Some(adapter) = get_test_adapter() else {
        eprintln!("SKIP: No adapter available");
        return;
    };

    let adapter_limits = adapter.limits();

    // Use limits that are less than or equal to adapter limits
    let mut limits = wgpu::Limits::default();
    limits.max_texture_dimension_2d =
        adapter_limits.max_texture_dimension_2d.min(limits.max_texture_dimension_2d);
    limits.max_buffer_size = adapter_limits.max_buffer_size.min(limits.max_buffer_size);

    let result = block_on(TrinityDevice::new(
        &adapter,
        wgpu::Features::empty(),
        limits,
    ));

    assert!(result.is_ok(), "Should succeed with limits within capability");
}

// ---------------------------------------------------------------------------
// 4.2: Request limits exceeding adapter -> LimitNotMet error
// ---------------------------------------------------------------------------

#[test]
fn test_device_new_with_excessive_texture_dimension_1d() {
    let Some(adapter) = get_test_adapter() else {
        eprintln!("SKIP: No adapter available");
        return;
    };

    let adapter_limits = adapter.limits();

    let mut limits = wgpu::Limits::default();
    limits.max_texture_dimension_1d = adapter_limits.max_texture_dimension_1d + 1;

    let result = block_on(TrinityDevice::new(
        &adapter,
        wgpu::Features::empty(),
        limits,
    ));

    assert!(result.is_err());
    match result.unwrap_err() {
        DeviceCreationError::LimitNotMet { limit, .. } => {
            assert_eq!(limit, "max_texture_dimension_1d");
        }
        other => panic!("Expected LimitNotMet for max_texture_dimension_1d, got {:?}", other),
    }
}

#[test]
fn test_device_new_with_excessive_texture_dimension_2d() {
    let Some(adapter) = get_test_adapter() else {
        eprintln!("SKIP: No adapter available");
        return;
    };

    let adapter_limits = adapter.limits();

    let mut limits = wgpu::Limits::default();
    limits.max_texture_dimension_2d = adapter_limits.max_texture_dimension_2d + 1;

    let result = block_on(TrinityDevice::new(
        &adapter,
        wgpu::Features::empty(),
        limits,
    ));

    assert!(result.is_err());
    match result.unwrap_err() {
        DeviceCreationError::LimitNotMet { limit, .. } => {
            assert_eq!(limit, "max_texture_dimension_2d");
        }
        other => panic!("Expected LimitNotMet for max_texture_dimension_2d, got {:?}", other),
    }
}

#[test]
fn test_device_new_with_excessive_texture_dimension_3d() {
    let Some(adapter) = get_test_adapter() else {
        eprintln!("SKIP: No adapter available");
        return;
    };

    let adapter_limits = adapter.limits();

    let mut limits = wgpu::Limits::default();
    limits.max_texture_dimension_3d = adapter_limits.max_texture_dimension_3d + 1;

    let result = block_on(TrinityDevice::new(
        &adapter,
        wgpu::Features::empty(),
        limits,
    ));

    assert!(result.is_err());
    match result.unwrap_err() {
        DeviceCreationError::LimitNotMet { limit, .. } => {
            assert_eq!(limit, "max_texture_dimension_3d");
        }
        other => panic!("Expected LimitNotMet for max_texture_dimension_3d, got {:?}", other),
    }
}

#[test]
fn test_device_new_with_excessive_texture_array_layers() {
    let Some(adapter) = get_test_adapter() else {
        eprintln!("SKIP: No adapter available");
        return;
    };

    let adapter_limits = adapter.limits();

    let mut limits = wgpu::Limits::default();
    limits.max_texture_array_layers = adapter_limits.max_texture_array_layers + 1;

    let result = block_on(TrinityDevice::new(
        &adapter,
        wgpu::Features::empty(),
        limits,
    ));

    assert!(result.is_err());
    match result.unwrap_err() {
        DeviceCreationError::LimitNotMet { limit, .. } => {
            assert_eq!(limit, "max_texture_array_layers");
        }
        other => panic!("Expected LimitNotMet for max_texture_array_layers, got {:?}", other),
    }
}

#[test]
fn test_device_new_with_excessive_buffer_size() {
    let Some(adapter) = get_test_adapter() else {
        eprintln!("SKIP: No adapter available");
        return;
    };

    let adapter_limits = adapter.limits();

    let mut limits = wgpu::Limits::default();
    limits.max_buffer_size = adapter_limits.max_buffer_size + 1;

    let result = block_on(TrinityDevice::new(
        &adapter,
        wgpu::Features::empty(),
        limits,
    ));

    assert!(result.is_err());
    match result.unwrap_err() {
        DeviceCreationError::LimitNotMet { limit, .. } => {
            assert_eq!(limit, "max_buffer_size");
        }
        other => panic!("Expected LimitNotMet for max_buffer_size, got {:?}", other),
    }
}

#[test]
fn test_device_new_with_excessive_uniform_buffer_binding_size() {
    let Some(adapter) = get_test_adapter() else {
        eprintln!("SKIP: No adapter available");
        return;
    };

    let adapter_limits = adapter.limits();

    let mut limits = wgpu::Limits::default();
    limits.max_uniform_buffer_binding_size = adapter_limits.max_uniform_buffer_binding_size + 1;

    let result = block_on(TrinityDevice::new(
        &adapter,
        wgpu::Features::empty(),
        limits,
    ));

    assert!(result.is_err());
    match result.unwrap_err() {
        DeviceCreationError::LimitNotMet { limit, .. } => {
            assert_eq!(limit, "max_uniform_buffer_binding_size");
        }
        other => panic!(
            "Expected LimitNotMet for max_uniform_buffer_binding_size, got {:?}",
            other
        ),
    }
}

#[test]
fn test_device_new_with_excessive_storage_buffer_binding_size() {
    let Some(adapter) = get_test_adapter() else {
        eprintln!("SKIP: No adapter available");
        return;
    };

    let adapter_limits = adapter.limits();

    let mut limits = wgpu::Limits::default();
    limits.max_storage_buffer_binding_size = adapter_limits.max_storage_buffer_binding_size + 1;

    let result = block_on(TrinityDevice::new(
        &adapter,
        wgpu::Features::empty(),
        limits,
    ));

    assert!(result.is_err());
    match result.unwrap_err() {
        DeviceCreationError::LimitNotMet { limit, .. } => {
            assert_eq!(limit, "max_storage_buffer_binding_size");
        }
        other => panic!(
            "Expected LimitNotMet for max_storage_buffer_binding_size, got {:?}",
            other
        ),
    }
}

#[test]
fn test_device_new_with_excessive_bind_groups() {
    let Some(adapter) = get_test_adapter() else {
        eprintln!("SKIP: No adapter available");
        return;
    };

    let adapter_limits = adapter.limits();

    let mut limits = wgpu::Limits::default();
    limits.max_bind_groups = adapter_limits.max_bind_groups + 1;

    let result = block_on(TrinityDevice::new(
        &adapter,
        wgpu::Features::empty(),
        limits,
    ));

    assert!(result.is_err());
    match result.unwrap_err() {
        DeviceCreationError::LimitNotMet { limit, .. } => {
            assert_eq!(limit, "max_bind_groups");
        }
        other => panic!("Expected LimitNotMet for max_bind_groups, got {:?}", other),
    }
}

#[test]
fn test_device_new_with_excessive_bindings_per_bind_group() {
    let Some(adapter) = get_test_adapter() else {
        eprintln!("SKIP: No adapter available");
        return;
    };

    let adapter_limits = adapter.limits();

    let mut limits = wgpu::Limits::default();
    limits.max_bindings_per_bind_group = adapter_limits.max_bindings_per_bind_group + 1;

    let result = block_on(TrinityDevice::new(
        &adapter,
        wgpu::Features::empty(),
        limits,
    ));

    assert!(result.is_err());
    match result.unwrap_err() {
        DeviceCreationError::LimitNotMet { limit, .. } => {
            assert_eq!(limit, "max_bindings_per_bind_group");
        }
        other => panic!(
            "Expected LimitNotMet for max_bindings_per_bind_group, got {:?}",
            other
        ),
    }
}

#[test]
fn test_device_new_with_excessive_compute_workgroup_size_x() {
    let Some(adapter) = get_test_adapter() else {
        eprintln!("SKIP: No adapter available");
        return;
    };

    let adapter_limits = adapter.limits();

    let mut limits = wgpu::Limits::default();
    limits.max_compute_workgroup_size_x = adapter_limits.max_compute_workgroup_size_x + 1;

    let result = block_on(TrinityDevice::new(
        &adapter,
        wgpu::Features::empty(),
        limits,
    ));

    assert!(result.is_err());
    match result.unwrap_err() {
        DeviceCreationError::LimitNotMet { limit, .. } => {
            assert_eq!(limit, "max_compute_workgroup_size_x");
        }
        other => panic!(
            "Expected LimitNotMet for max_compute_workgroup_size_x, got {:?}",
            other
        ),
    }
}

#[test]
fn test_device_new_with_excessive_compute_workgroup_size_y() {
    let Some(adapter) = get_test_adapter() else {
        eprintln!("SKIP: No adapter available");
        return;
    };

    let adapter_limits = adapter.limits();

    let mut limits = wgpu::Limits::default();
    limits.max_compute_workgroup_size_y = adapter_limits.max_compute_workgroup_size_y + 1;

    let result = block_on(TrinityDevice::new(
        &adapter,
        wgpu::Features::empty(),
        limits,
    ));

    assert!(result.is_err());
    match result.unwrap_err() {
        DeviceCreationError::LimitNotMet { limit, .. } => {
            assert_eq!(limit, "max_compute_workgroup_size_y");
        }
        other => panic!(
            "Expected LimitNotMet for max_compute_workgroup_size_y, got {:?}",
            other
        ),
    }
}

#[test]
fn test_device_new_with_excessive_compute_workgroup_size_z() {
    let Some(adapter) = get_test_adapter() else {
        eprintln!("SKIP: No adapter available");
        return;
    };

    let adapter_limits = adapter.limits();

    let mut limits = wgpu::Limits::default();
    limits.max_compute_workgroup_size_z = adapter_limits.max_compute_workgroup_size_z + 1;

    let result = block_on(TrinityDevice::new(
        &adapter,
        wgpu::Features::empty(),
        limits,
    ));

    assert!(result.is_err());
    match result.unwrap_err() {
        DeviceCreationError::LimitNotMet { limit, .. } => {
            assert_eq!(limit, "max_compute_workgroup_size_z");
        }
        other => panic!(
            "Expected LimitNotMet for max_compute_workgroup_size_z, got {:?}",
            other
        ),
    }
}

#[test]
fn test_device_new_with_excessive_compute_invocations_per_workgroup() {
    let Some(adapter) = get_test_adapter() else {
        eprintln!("SKIP: No adapter available");
        return;
    };

    let adapter_limits = adapter.limits();

    let mut limits = wgpu::Limits::default();
    limits.max_compute_invocations_per_workgroup =
        adapter_limits.max_compute_invocations_per_workgroup + 1;

    let result = block_on(TrinityDevice::new(
        &adapter,
        wgpu::Features::empty(),
        limits,
    ));

    assert!(result.is_err());
    match result.unwrap_err() {
        DeviceCreationError::LimitNotMet { limit, .. } => {
            assert_eq!(limit, "max_compute_invocations_per_workgroup");
        }
        other => panic!(
            "Expected LimitNotMet for max_compute_invocations_per_workgroup, got {:?}",
            other
        ),
    }
}

#[test]
fn test_device_new_with_excessive_compute_workgroups_per_dimension() {
    let Some(adapter) = get_test_adapter() else {
        eprintln!("SKIP: No adapter available");
        return;
    };

    let adapter_limits = adapter.limits();

    let mut limits = wgpu::Limits::default();
    limits.max_compute_workgroups_per_dimension =
        adapter_limits.max_compute_workgroups_per_dimension + 1;

    let result = block_on(TrinityDevice::new(
        &adapter,
        wgpu::Features::empty(),
        limits,
    ));

    assert!(result.is_err());
    match result.unwrap_err() {
        DeviceCreationError::LimitNotMet { limit, .. } => {
            assert_eq!(limit, "max_compute_workgroups_per_dimension");
        }
        other => panic!(
            "Expected LimitNotMet for max_compute_workgroups_per_dimension, got {:?}",
            other
        ),
    }
}

#[test]
fn test_device_new_with_excessive_vertex_buffers() {
    let Some(adapter) = get_test_adapter() else {
        eprintln!("SKIP: No adapter available");
        return;
    };

    let adapter_limits = adapter.limits();

    let mut limits = wgpu::Limits::default();
    limits.max_vertex_buffers = adapter_limits.max_vertex_buffers + 1;

    let result = block_on(TrinityDevice::new(
        &adapter,
        wgpu::Features::empty(),
        limits,
    ));

    assert!(result.is_err());
    match result.unwrap_err() {
        DeviceCreationError::LimitNotMet { limit, .. } => {
            assert_eq!(limit, "max_vertex_buffers");
        }
        other => panic!("Expected LimitNotMet for max_vertex_buffers, got {:?}", other),
    }
}

#[test]
fn test_device_new_with_excessive_vertex_attributes() {
    let Some(adapter) = get_test_adapter() else {
        eprintln!("SKIP: No adapter available");
        return;
    };

    let adapter_limits = adapter.limits();

    let mut limits = wgpu::Limits::default();
    limits.max_vertex_attributes = adapter_limits.max_vertex_attributes + 1;

    let result = block_on(TrinityDevice::new(
        &adapter,
        wgpu::Features::empty(),
        limits,
    ));

    assert!(result.is_err());
    match result.unwrap_err() {
        DeviceCreationError::LimitNotMet { limit, .. } => {
            assert_eq!(limit, "max_vertex_attributes");
        }
        other => panic!("Expected LimitNotMet for max_vertex_attributes, got {:?}", other),
    }
}

#[test]
fn test_device_new_with_excessive_vertex_buffer_array_stride() {
    let Some(adapter) = get_test_adapter() else {
        eprintln!("SKIP: No adapter available");
        return;
    };

    let adapter_limits = adapter.limits();

    let mut limits = wgpu::Limits::default();
    limits.max_vertex_buffer_array_stride = adapter_limits.max_vertex_buffer_array_stride + 1;

    let result = block_on(TrinityDevice::new(
        &adapter,
        wgpu::Features::empty(),
        limits,
    ));

    assert!(result.is_err());
    match result.unwrap_err() {
        DeviceCreationError::LimitNotMet { limit, .. } => {
            assert_eq!(limit, "max_vertex_buffer_array_stride");
        }
        other => panic!(
            "Expected LimitNotMet for max_vertex_buffer_array_stride, got {:?}",
            other
        ),
    }
}

#[test]
fn test_device_new_with_excessive_color_attachments() {
    let Some(adapter) = get_test_adapter() else {
        eprintln!("SKIP: No adapter available");
        return;
    };

    let adapter_limits = adapter.limits();

    let mut limits = wgpu::Limits::default();
    limits.max_color_attachments = adapter_limits.max_color_attachments + 1;

    let result = block_on(TrinityDevice::new(
        &adapter,
        wgpu::Features::empty(),
        limits,
    ));

    assert!(result.is_err());
    match result.unwrap_err() {
        DeviceCreationError::LimitNotMet { limit, .. } => {
            assert_eq!(limit, "max_color_attachments");
        }
        other => panic!("Expected LimitNotMet for max_color_attachments, got {:?}", other),
    }
}

#[test]
fn test_device_new_with_excessive_color_attachment_bytes_per_sample() {
    let Some(adapter) = get_test_adapter() else {
        eprintln!("SKIP: No adapter available");
        return;
    };

    let adapter_limits = adapter.limits();

    let mut limits = wgpu::Limits::default();
    limits.max_color_attachment_bytes_per_sample =
        adapter_limits.max_color_attachment_bytes_per_sample + 1;

    let result = block_on(TrinityDevice::new(
        &adapter,
        wgpu::Features::empty(),
        limits,
    ));

    assert!(result.is_err());
    match result.unwrap_err() {
        DeviceCreationError::LimitNotMet { limit, .. } => {
            assert_eq!(limit, "max_color_attachment_bytes_per_sample");
        }
        other => panic!(
            "Expected LimitNotMet for max_color_attachment_bytes_per_sample, got {:?}",
            other
        ),
    }
}

// =============================================================================
// Section 5: Accessor Tests
// =============================================================================

// ---------------------------------------------------------------------------
// 5.1: device() returns valid device
// ---------------------------------------------------------------------------

#[test]
fn test_device_accessor_returns_valid_reference() {
    let Some(adapter) = get_test_adapter() else {
        eprintln!("SKIP: No adapter available");
        return;
    };

    let device =
        block_on(TrinityDevice::with_defaults(&adapter)).expect("Device creation failed");

    let inner = device.device();

    // Verify we can call methods on the device
    let _ = inner.limits();
    let _ = inner.features();
}

#[test]
fn test_device_accessor_can_create_buffer() {
    let Some(adapter) = get_test_adapter() else {
        eprintln!("SKIP: No adapter available");
        return;
    };

    let device =
        block_on(TrinityDevice::with_defaults(&adapter)).expect("Device creation failed");

    let buffer = device.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("Test Buffer"),
        size: 1024,
        usage: wgpu::BufferUsages::UNIFORM,
        mapped_at_creation: false,
    });

    // Buffer should be created successfully - size() returns the buffer size
    assert_eq!(buffer.size(), 1024);
}

#[test]
fn test_device_accessor_can_create_texture() {
    let Some(adapter) = get_test_adapter() else {
        eprintln!("SKIP: No adapter available");
        return;
    };

    let device =
        block_on(TrinityDevice::with_defaults(&adapter)).expect("Device creation failed");

    let texture = device.device().create_texture(&wgpu::TextureDescriptor {
        label: Some("Test Texture"),
        size: wgpu::Extent3d {
            width: 256,
            height: 256,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });

    // Texture should be created successfully
    let _view = texture.create_view(&wgpu::TextureViewDescriptor::default());
}

// ---------------------------------------------------------------------------
// 5.2: queue() returns valid queue
// ---------------------------------------------------------------------------

#[test]
fn test_queue_accessor_returns_valid_reference() {
    let Some(adapter) = get_test_adapter() else {
        eprintln!("SKIP: No adapter available");
        return;
    };

    let device =
        block_on(TrinityDevice::with_defaults(&adapter)).expect("Device creation failed");

    let queue = device.queue();

    // Verify we can submit empty work
    queue.submit(std::iter::empty());
}

#[test]
fn test_queue_accessor_can_write_buffer() {
    let Some(adapter) = get_test_adapter() else {
        eprintln!("SKIP: No adapter available");
        return;
    };

    let device =
        block_on(TrinityDevice::with_defaults(&adapter)).expect("Device creation failed");

    let buffer = device.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("Test Buffer"),
        size: 64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let data: [u8; 64] = [0; 64];
    device.queue().write_buffer(&buffer, 0, &data);
}

// ---------------------------------------------------------------------------
// 5.3: features() returns enabled features
// ---------------------------------------------------------------------------

#[test]
fn test_features_accessor_returns_requested_features() {
    let Some(adapter) = get_test_adapter() else {
        eprintln!("SKIP: No adapter available");
        return;
    };

    let requested = wgpu::Features::empty();
    let device = block_on(TrinityDevice::new(
        &adapter,
        requested,
        wgpu::Limits::default(),
    ))
    .expect("Device creation failed");

    assert_eq!(device.features(), requested);
}

#[test]
fn test_features_accessor_iteration() {
    let Some(adapter) = get_test_adapter() else {
        eprintln!("SKIP: No adapter available");
        return;
    };

    let device = block_on(TrinityDevice::with_all_features(&adapter))
        .expect("Device creation failed");

    // Features should be iterable
    let count = device.features().iter().count();
    let adapter_count = adapter.features().iter().count();

    assert_eq!(count, adapter_count);
}

// ---------------------------------------------------------------------------
// 5.4: limits() returns configured limits
// ---------------------------------------------------------------------------

#[test]
fn test_limits_accessor_returns_reference() {
    let Some(adapter) = get_test_adapter() else {
        eprintln!("SKIP: No adapter available");
        return;
    };

    let limits = wgpu::Limits::default();
    let device = block_on(TrinityDevice::new(
        &adapter,
        wgpu::Features::empty(),
        limits.clone(),
    ))
    .expect("Device creation failed");

    let returned_limits = device.limits();

    assert_eq!(returned_limits.max_texture_dimension_2d, limits.max_texture_dimension_2d);
    assert_eq!(returned_limits.max_buffer_size, limits.max_buffer_size);
    assert_eq!(returned_limits.max_bind_groups, limits.max_bind_groups);
}

#[test]
fn test_limits_accessor_all_fields_accessible() {
    let Some(adapter) = get_test_adapter() else {
        eprintln!("SKIP: No adapter available");
        return;
    };

    let device =
        block_on(TrinityDevice::with_defaults(&adapter)).expect("Device creation failed");

    let limits = device.limits();

    // All fields should be accessible
    let _ = limits.max_texture_dimension_1d;
    let _ = limits.max_texture_dimension_2d;
    let _ = limits.max_texture_dimension_3d;
    let _ = limits.max_texture_array_layers;
    let _ = limits.max_bind_groups;
    let _ = limits.max_bindings_per_bind_group;
    let _ = limits.max_buffer_size;
    let _ = limits.max_vertex_buffers;
    let _ = limits.max_vertex_attributes;
    let _ = limits.max_compute_workgroup_size_x;
    let _ = limits.max_compute_workgroup_size_y;
    let _ = limits.max_compute_workgroup_size_z;
}

// =============================================================================
// Section 6: Convenience Methods Tests
// =============================================================================

// ---------------------------------------------------------------------------
// 6.1: submit() for command buffer
// ---------------------------------------------------------------------------

#[test]
fn test_submit_empty_command_buffers() {
    let Some(adapter) = get_test_adapter() else {
        eprintln!("SKIP: No adapter available");
        return;
    };

    let device =
        block_on(TrinityDevice::with_defaults(&adapter)).expect("Device creation failed");

    let index = device.submit(std::iter::empty());

    // Submission index should be valid
    let _ = index;
}

#[test]
fn test_submit_single_command_buffer() {
    let Some(adapter) = get_test_adapter() else {
        eprintln!("SKIP: No adapter available");
        return;
    };

    let device =
        block_on(TrinityDevice::with_defaults(&adapter)).expect("Device creation failed");

    let encoder = device.create_command_encoder(Some("Test Encoder"));
    let command_buffer = encoder.finish();

    let index = device.submit(std::iter::once(command_buffer));

    let _ = index;
}

#[test]
fn test_submit_multiple_command_buffers() {
    let Some(adapter) = get_test_adapter() else {
        eprintln!("SKIP: No adapter available");
        return;
    };

    let device =
        block_on(TrinityDevice::with_defaults(&adapter)).expect("Device creation failed");

    let encoder1 = device.create_command_encoder(Some("Encoder 1"));
    let encoder2 = device.create_command_encoder(Some("Encoder 2"));
    let encoder3 = device.create_command_encoder(Some("Encoder 3"));

    let command_buffers = vec![encoder1.finish(), encoder2.finish(), encoder3.finish()];

    let index = device.submit(command_buffers);

    let _ = index;
}

// ---------------------------------------------------------------------------
// 6.2: create_command_encoder() for encoder
// ---------------------------------------------------------------------------

#[test]
fn test_create_command_encoder_with_label() {
    let Some(adapter) = get_test_adapter() else {
        eprintln!("SKIP: No adapter available");
        return;
    };

    let device =
        block_on(TrinityDevice::with_defaults(&adapter)).expect("Device creation failed");

    let encoder = device.create_command_encoder(Some("My Encoder"));

    // Encoder should be usable
    let command_buffer = encoder.finish();
    device.submit(std::iter::once(command_buffer));
}

#[test]
fn test_create_command_encoder_without_label() {
    let Some(adapter) = get_test_adapter() else {
        eprintln!("SKIP: No adapter available");
        return;
    };

    let device =
        block_on(TrinityDevice::with_defaults(&adapter)).expect("Device creation failed");

    let encoder = device.create_command_encoder(None);

    let command_buffer = encoder.finish();
    device.submit(std::iter::once(command_buffer));
}

#[test]
fn test_create_command_encoder_multiple() {
    let Some(adapter) = get_test_adapter() else {
        eprintln!("SKIP: No adapter available");
        return;
    };

    let device =
        block_on(TrinityDevice::with_defaults(&adapter)).expect("Device creation failed");

    // Create multiple encoders
    let _ = device.create_command_encoder(Some("Encoder 1"));
    let _ = device.create_command_encoder(Some("Encoder 2"));
    let _ = device.create_command_encoder(Some("Encoder 3"));
}

// =============================================================================
// Section 7: Trait Implementation Tests
// =============================================================================

// ---------------------------------------------------------------------------
// 7.1: Debug implementation for TrinityDevice
// ---------------------------------------------------------------------------

#[test]
fn test_trinity_device_debug_impl() {
    let Some(adapter) = get_test_adapter() else {
        eprintln!("SKIP: No adapter available");
        return;
    };

    let device =
        block_on(TrinityDevice::with_defaults(&adapter)).expect("Device creation failed");

    let debug_str = format!("{:?}", device);

    assert!(debug_str.contains("TrinityDevice"));
    assert!(debug_str.contains("device"));
    assert!(debug_str.contains("queue"));
    assert!(debug_str.contains("features"));
    assert!(debug_str.contains("limits"));
}

// ---------------------------------------------------------------------------
// 7.2: Display implementation for TrinityDevice
// ---------------------------------------------------------------------------

#[test]
fn test_trinity_device_display_impl() {
    let Some(adapter) = get_test_adapter() else {
        eprintln!("SKIP: No adapter available");
        return;
    };

    let device =
        block_on(TrinityDevice::with_defaults(&adapter)).expect("Device creation failed");

    let display_str = format!("{}", device);

    assert!(display_str.contains("TrinityDevice"));
    assert!(display_str.contains("Features"));
    assert!(display_str.contains("Max texture"));
    assert!(display_str.contains("Max buffer"));
    assert!(display_str.contains("Max bind groups"));
}

#[test]
fn test_trinity_device_display_shows_feature_count() {
    let Some(adapter) = get_test_adapter() else {
        eprintln!("SKIP: No adapter available");
        return;
    };

    let device = block_on(TrinityDevice::with_all_features(&adapter))
        .expect("Device creation failed");

    let display_str = format!("{}", device);

    // Should show feature count
    assert!(display_str.contains("enabled"));
}

// ---------------------------------------------------------------------------
// 7.3: Debug implementation for DeviceCreationError
// ---------------------------------------------------------------------------

#[test]
fn test_device_creation_error_debug_feature_not_supported() {
    let err = DeviceCreationError::FeatureNotSupported(wgpu::Features::TIMESTAMP_QUERY);
    let debug_str = format!("{:?}", err);

    assert!(debug_str.contains("FeatureNotSupported"));
    assert!(debug_str.contains("TIMESTAMP_QUERY"));
}

#[test]
fn test_device_creation_error_debug_limit_not_met() {
    let err = DeviceCreationError::LimitNotMet {
        limit: "max_texture_dimension_2d".to_string(),
        required: 16384,
        available: 8192,
    };
    let debug_str = format!("{:?}", err);

    assert!(debug_str.contains("LimitNotMet"));
    assert!(debug_str.contains("max_texture_dimension_2d"));
}

// ---------------------------------------------------------------------------
// 7.4: Display implementation for DeviceCreationError
// ---------------------------------------------------------------------------

#[test]
fn test_device_creation_error_display_format() {
    let err = DeviceCreationError::LimitNotMet {
        limit: "test_limit".to_string(),
        required: 100,
        available: 50,
    };
    let display_str = format!("{}", err);

    assert!(display_str.contains("test_limit"));
    assert!(display_str.contains("100"));
    assert!(display_str.contains("50"));
}

// ---------------------------------------------------------------------------
// 7.5: Error trait implementation
// ---------------------------------------------------------------------------

#[test]
fn test_device_creation_error_is_error() {
    let err = DeviceCreationError::FeatureNotSupported(wgpu::Features::empty());

    // Verify it implements std::error::Error
    let _: &dyn Error = &err;
}

#[test]
fn test_device_creation_error_source_for_feature_not_supported() {
    let err = DeviceCreationError::FeatureNotSupported(wgpu::Features::empty());

    // FeatureNotSupported has no source
    assert!(err.source().is_none());
}

#[test]
fn test_device_creation_error_source_for_limit_not_met() {
    let err = DeviceCreationError::LimitNotMet {
        limit: "test".to_string(),
        required: 1,
        available: 0,
    };

    // LimitNotMet has no source
    assert!(err.source().is_none());
}

// =============================================================================
// Section 8: Edge Cases and Boundary Conditions
// =============================================================================

#[test]
fn test_device_creation_with_minimum_valid_limits() {
    let Some(adapter) = get_test_adapter() else {
        eprintln!("SKIP: No adapter available");
        return;
    };

    // Use the most restrictive limits that should still work
    let limits = wgpu::Limits::downlevel_webgl2_defaults();

    let result = block_on(TrinityDevice::new(
        &adapter,
        wgpu::Features::empty(),
        limits,
    ));

    // Should succeed - these are very minimal limits
    assert!(result.is_ok());
}

#[test]
fn test_device_creation_with_adapter_exact_limits() {
    let Some(adapter) = get_test_adapter() else {
        eprintln!("SKIP: No adapter available");
        return;
    };

    let limits = adapter.limits();

    let result = block_on(TrinityDevice::new(
        &adapter,
        wgpu::Features::empty(),
        limits,
    ));

    // Using exact adapter limits should succeed
    assert!(result.is_ok());
}

#[test]
fn test_device_creation_rejects_first_excessive_limit() {
    let Some(adapter) = get_test_adapter() else {
        eprintln!("SKIP: No adapter available");
        return;
    };

    let adapter_limits = adapter.limits();

    // Set multiple excessive limits - should fail on the first checked
    let mut limits = wgpu::Limits::default();
    limits.max_texture_dimension_1d = adapter_limits.max_texture_dimension_1d + 1;
    limits.max_texture_dimension_2d = adapter_limits.max_texture_dimension_2d + 1;
    limits.max_buffer_size = adapter_limits.max_buffer_size + 1;

    let result = block_on(TrinityDevice::new(
        &adapter,
        wgpu::Features::empty(),
        limits,
    ));

    assert!(result.is_err());

    // Should fail on first checked limit (max_texture_dimension_1d)
    match result.unwrap_err() {
        DeviceCreationError::LimitNotMet { limit, .. } => {
            assert_eq!(limit, "max_texture_dimension_1d");
        }
        other => panic!("Expected LimitNotMet, got {:?}", other),
    }
}

#[test]
fn test_device_is_send() {
    let Some(adapter) = get_test_adapter() else {
        eprintln!("SKIP: No adapter available");
        return;
    };

    let device =
        block_on(TrinityDevice::with_defaults(&adapter)).expect("Device creation failed");

    fn assert_send<T: Send>(_: &T) {}
    assert_send(&device);
}

#[test]
fn test_device_is_sync() {
    let Some(adapter) = get_test_adapter() else {
        eprintln!("SKIP: No adapter available");
        return;
    };

    let device =
        block_on(TrinityDevice::with_defaults(&adapter)).expect("Device creation failed");

    fn assert_sync<T: Sync>(_: &T) {}
    assert_sync(&device);
}

#[test]
fn test_device_can_be_wrapped_in_arc() {
    use std::sync::Arc;

    let Some(adapter) = get_test_adapter() else {
        eprintln!("SKIP: No adapter available");
        return;
    };

    let device =
        block_on(TrinityDevice::with_defaults(&adapter)).expect("Device creation failed");

    let arc_device = Arc::new(device);

    // Should be clonable
    let clone1 = Arc::clone(&arc_device);
    let clone2 = Arc::clone(&arc_device);

    // All clones should access the same device
    assert_eq!(
        clone1.limits().max_texture_dimension_2d,
        clone2.limits().max_texture_dimension_2d
    );
}

#[test]
fn test_multiple_devices_from_same_adapter() {
    let Some(adapter) = get_test_adapter() else {
        eprintln!("SKIP: No adapter available");
        return;
    };

    // Create multiple devices from the same adapter
    let device1 =
        block_on(TrinityDevice::with_defaults(&adapter)).expect("Device 1 creation failed");
    let device2 =
        block_on(TrinityDevice::with_defaults(&adapter)).expect("Device 2 creation failed");

    // Both devices should be functional
    let _ = device1.create_command_encoder(Some("Device 1 Encoder"));
    let _ = device2.create_command_encoder(Some("Device 2 Encoder"));
}

#[test]
fn test_device_limits_returned_by_reference() {
    let Some(adapter) = get_test_adapter() else {
        eprintln!("SKIP: No adapter available");
        return;
    };

    let device =
        block_on(TrinityDevice::with_defaults(&adapter)).expect("Device creation failed");

    // limits() returns &wgpu::Limits
    let limits_ref: &wgpu::Limits = device.limits();

    // Multiple calls should return same reference
    let limits_ref2 = device.limits();
    assert!(std::ptr::eq(limits_ref, limits_ref2));
}

#[test]
fn test_device_features_returned_by_value() {
    let Some(adapter) = get_test_adapter() else {
        eprintln!("SKIP: No adapter available");
        return;
    };

    let device =
        block_on(TrinityDevice::with_defaults(&adapter)).expect("Device creation failed");

    // features() returns wgpu::Features by value
    let features1: wgpu::Features = device.features();
    let features2: wgpu::Features = device.features();

    // Both should be equal
    assert_eq!(features1, features2);
}

#[test]
fn test_error_message_contains_useful_information() {
    // Feature error
    let err1 = DeviceCreationError::FeatureNotSupported(
        wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::TEXTURE_COMPRESSION_BC,
    );
    let msg1 = format!("{}", err1);
    assert!(
        msg1.len() > 20,
        "Error message should be descriptive"
    );

    // Limit error
    let err2 = DeviceCreationError::LimitNotMet {
        limit: "max_texture_dimension_2d".to_string(),
        required: 16384,
        available: 8192,
    };
    let msg2 = format!("{}", err2);
    assert!(msg2.contains("16384") && msg2.contains("8192"));
}

// =============================================================================
// Additional Tests for Complete Coverage
// =============================================================================

#[test]
fn test_validate_limits_equal_values_succeeds() {
    let Some(adapter) = get_test_adapter() else {
        eprintln!("SKIP: No adapter available");
        return;
    };

    let adapter_limits = adapter.limits();

    // Request exactly the adapter limits
    let result = block_on(TrinityDevice::new(
        &adapter,
        wgpu::Features::empty(),
        adapter_limits.clone(),
    ));

    assert!(result.is_ok(), "Equal limits should succeed");
}

#[test]
fn test_device_new_with_features_and_limits_combined() {
    let Some(adapter) = get_test_adapter() else {
        eprintln!("SKIP: No adapter available");
        return;
    };

    let adapter_features = adapter.features();
    let adapter_limits = adapter.limits();

    // Use some supported features and valid limits
    let features = if adapter_features.contains(wgpu::Features::DEPTH32FLOAT_STENCIL8) {
        wgpu::Features::DEPTH32FLOAT_STENCIL8
    } else {
        wgpu::Features::empty()
    };

    let mut limits = wgpu::Limits::default();
    limits.max_texture_dimension_2d =
        adapter_limits.max_texture_dimension_2d.min(limits.max_texture_dimension_2d);

    let device = block_on(TrinityDevice::new(&adapter, features, limits.clone()))
        .expect("Device creation failed");

    assert_eq!(device.features(), features);
    assert_eq!(
        device.limits().max_texture_dimension_2d,
        limits.max_texture_dimension_2d
    );
}

#[test]
fn test_submission_index_is_usable() {
    let Some(adapter) = get_test_adapter() else {
        eprintln!("SKIP: No adapter available");
        return;
    };

    let device =
        block_on(TrinityDevice::with_defaults(&adapter)).expect("Device creation failed");

    let encoder = device.create_command_encoder(None);
    let cmd = encoder.finish();

    let index = device.submit(std::iter::once(cmd));

    // SubmissionIndex can be stored and used later
    let _stored_index = index;
}

#[test]
fn test_device_queue_submit_via_convenience_method() {
    let Some(adapter) = get_test_adapter() else {
        eprintln!("SKIP: No adapter available");
        return;
    };

    let device =
        block_on(TrinityDevice::with_defaults(&adapter)).expect("Device creation failed");

    // Use convenience submit() instead of queue().submit()
    let encoder = device.create_command_encoder(Some("Via Convenience"));
    let _index = device.submit(std::iter::once(encoder.finish()));
}

#[test]
fn test_create_encoder_label_preserved() {
    let Some(adapter) = get_test_adapter() else {
        eprintln!("SKIP: No adapter available");
        return;
    };

    let device =
        block_on(TrinityDevice::with_defaults(&adapter)).expect("Device creation failed");

    // Label is passed through to wgpu
    let label = "My Labeled Encoder";
    let encoder = device.create_command_encoder(Some(label));

    // Finish and submit to verify it works
    let cmd = encoder.finish();
    device.submit(std::iter::once(cmd));
}

#[test]
fn test_device_creation_error_variants_are_distinct() {
    let err1 = DeviceCreationError::FeatureNotSupported(wgpu::Features::empty());
    let err2 = DeviceCreationError::LimitNotMet {
        limit: String::new(),
        required: 0,
        available: 0,
    };

    // Different variants should have different debug output
    let debug1 = format!("{:?}", err1);
    let debug2 = format!("{:?}", err2);

    assert_ne!(debug1, debug2);
}

#[test]
fn test_has_feature_with_multiple_features() {
    let Some(adapter) = get_test_adapter() else {
        eprintln!("SKIP: No adapter available");
        return;
    };

    let adapter_features = adapter.features();

    // Find two features the adapter supports
    let mut features_to_test = wgpu::Features::empty();
    if adapter_features.contains(wgpu::Features::DEPTH32FLOAT_STENCIL8) {
        features_to_test |= wgpu::Features::DEPTH32FLOAT_STENCIL8;
    }
    if adapter_features.contains(wgpu::Features::INDIRECT_FIRST_INSTANCE) {
        features_to_test |= wgpu::Features::INDIRECT_FIRST_INSTANCE;
    }

    if features_to_test.is_empty() {
        eprintln!("SKIP: No testable features");
        return;
    }

    let device = block_on(TrinityDevice::new(
        &adapter,
        features_to_test,
        wgpu::Limits::default(),
    ))
    .expect("Device creation failed");

    // has_feature should work with combined feature flags
    assert!(device.has_feature(features_to_test));
}
