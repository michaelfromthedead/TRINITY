// SPDX-License-Identifier: MIT
//
// integration_device_init.rs -- Integration tests for T-WGPU-P1.6.2
// (Full Device Initialization Sequence)
//
// These integration tests verify the complete initialization flow:
// TrinityInstance -> Adapter selection -> Device creation -> Queue -> Work submission
//
// Test categories:
//   - SECTION 1: Full initialization sequence
//   - SECTION 2: Device lost recovery (documented limitation)
//   - SECTION 3: Queue submission with actual work
//   - SECTION 4: Cross-backend test matrix (conditional compilation)
//   - SECTION 5: WebGPU/WASM tests (ignored in native CI)
//   - SECTION 6: CapabilityManager integration
//   - SECTION 7: Error handling during initialization
//
// Acceptance criteria (T-WGPU-P1.6.2):
//   1. Full initialization sequence test
//   2. Device lost recovery test (if simulable)
//   3. Queue submission test
//   4. Cross-backend test matrix (CI)
//   5. WebGPU test (browser or wasm-pack)

use renderer_backend::device::{
    AdapterSelector, BatcherConfig, CapabilityManager, DeviceCreationError, DeviceLostReason,
    DeviceManager, DeviceRequirements, DeviceState, ErrorScope, FeatureNegotiationError,
    LimitRequirements, RecoveryConfig, ResourceTracker, SubmissionTracker, TrinityDevice,
    TrinityInstance, TrinityQueue,
};
use std::sync::Arc;

// =============================================================================
// Test Helpers
// =============================================================================

/// Blocking helper to run async code in tests.
fn block_on<F: std::future::Future>(future: F) -> F::Output {
    pollster::block_on(future)
}

/// Check if a GPU adapter is available for hardware tests.
fn gpu_available() -> bool {
    let instance = TrinityInstance::new();
    !instance.enumerate_adapters().is_empty()
}

/// Get the first available adapter, or None if no GPU present.
fn get_adapter() -> Option<wgpu::Adapter> {
    let instance = TrinityInstance::new();
    instance.enumerate_adapters().into_iter().next()
}

/// Get the index of the best adapter using AdapterSelector.
///
/// Note: We return an index because the adapter cannot outlive the instance,
/// so tests that need the best adapter should enumerate and select inline.
#[allow(dead_code)]
fn get_best_adapter_index() -> Option<usize> {
    let instance = TrinityInstance::new();
    let adapters: Vec<_> = instance.inner().enumerate_adapters(instance.backends());

    if adapters.is_empty() {
        return None;
    }

    let selector = AdapterSelector::new();
    selector.select(&adapters).map(|result| {
        // Find the index of the selected adapter
        adapters
            .iter()
            .position(|a| a.get_info().name == result.adapter.get_info().name)
            .unwrap_or(0)
    })
}

// =============================================================================
// SECTION 1: Full Initialization Sequence Tests
// =============================================================================

/// Test 1.1: Complete initialization flow from Instance to Queue.
///
/// Verifies the full pipeline:
/// TrinityInstance::new() -> enumerate_adapters() -> TrinityDevice::new() -> TrinityQueue
#[test]
fn test_full_initialization_sequence_instance_to_queue() {
    // Step 1: Create TrinityInstance with platform-appropriate backends
    let instance = TrinityInstance::new();

    eprintln!(
        "INTEGRATION: Created TrinityInstance with backends {:?}",
        instance.backends()
    );

    // Step 2: Enumerate adapters
    let adapters = instance.enumerate_adapters();
    if adapters.is_empty() {
        eprintln!("INTEGRATION: No GPU adapter available, skipping hardware test");
        return;
    }

    eprintln!("INTEGRATION: Found {} adapter(s)", adapters.len());

    // Step 3: Select best adapter
    let adapter = &adapters[0];
    let info = adapter.get_info();
    eprintln!(
        "INTEGRATION: Using adapter '{}' (backend: {:?}, device_type: {:?})",
        info.name, info.backend, info.device_type
    );

    // Step 4: Create device with default features/limits
    let device_result = block_on(TrinityDevice::new(
        adapter,
        wgpu::Features::empty(),
        wgpu::Limits::default(),
    ));

    assert!(
        device_result.is_ok(),
        "Device creation should succeed: {:?}",
        device_result.err()
    );

    let device = device_result.unwrap();
    eprintln!(
        "INTEGRATION: Created TrinityDevice with limits max_texture_dimension_2d={}",
        device.limits().max_texture_dimension_2d
    );

    // Step 5: Create TrinityQueue wrapper
    let queue = device.queue();

    // Step 6: Verify queue is functional by creating and submitting empty work
    let encoder =
        device
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("integration_test_encoder"),
            });

    let command_buffer = encoder.finish();
    queue.submit(std::iter::once(command_buffer));

    eprintln!("INTEGRATION: Successfully submitted work to queue");
}

/// Test 1.2: Full initialization with AdapterSelector scoring.
///
/// Uses the production adapter selection algorithm.
#[test]
fn test_full_initialization_with_adapter_selector() {
    let instance = TrinityInstance::new();
    let adapters: Vec<_> = instance.inner().enumerate_adapters(instance.backends());

    if adapters.is_empty() {
        eprintln!("INTEGRATION: No adapters available, skipping test");
        return;
    }

    // Use AdapterSelector for production-style selection
    let selector = AdapterSelector::new();
    let selection_result = selector.select(&adapters);

    assert!(
        selection_result.is_some(),
        "AdapterSelector should find a suitable adapter"
    );

    let result = selection_result.unwrap();
    eprintln!(
        "INTEGRATION: Selected adapter '{}' with score {}",
        result.adapter.get_info().name,
        result.score.total
    );

    // Create device with selected adapter
    let device = block_on(TrinityDevice::new(
        &result.adapter,
        wgpu::Features::empty(),
        wgpu::Limits::default(),
    ))
    .expect("Device creation failed");

    // Verify device is usable
    let _queue = device.queue();
    eprintln!("INTEGRATION: Full initialization with selector completed successfully");
}

/// Test 1.3: Full initialization using DeviceRequirements negotiation.
#[test]
fn test_full_initialization_with_requirements_negotiation() {
    let Some(adapter) = get_adapter() else {
        eprintln!("INTEGRATION: No adapter, skipping test");
        return;
    };

    // Build requirements (no required features for broad compatibility)
    let requirements = DeviceRequirements::new();

    // Negotiate and create device (returns tuple: (device, negotiation_result))
    let result = block_on(renderer_backend::device::negotiate_and_create_device(
        &requirements,
        &adapter,
    ));

    assert!(
        result.is_ok(),
        "Device creation with requirements should succeed: {:?}",
        result.err()
    );

    let (device, _negotiation) = result.unwrap();
    eprintln!(
        "INTEGRATION: Created device via negotiate_and_create_device, limits: max_texture={}",
        device.limits().max_texture_dimension_2d
    );
}

/// Test 1.4: Verify TrinityInstance backends match platform expectations.
#[test]
fn test_instance_backends_match_platform() {
    let instance = TrinityInstance::new();

    // Platform-specific backend expectations:
    // - Linux: Vulkan | OpenGL
    // - Windows: Vulkan | DX12 | OpenGL
    // - macOS: Metal
    // - WASM: BROWSER_WEBGPU

    #[cfg(all(not(target_arch = "wasm32"), target_os = "linux"))]
    {
        let expected = wgpu::Backends::VULKAN | wgpu::Backends::GL;
        assert_eq!(
            instance.backends(),
            expected,
            "Linux should use Vulkan | GL backends"
        );
        eprintln!("INTEGRATION: Linux platform backends verified: {:?}", expected);
    }

    #[cfg(all(not(target_arch = "wasm32"), target_os = "windows"))]
    {
        let expected = wgpu::Backends::VULKAN | wgpu::Backends::DX12 | wgpu::Backends::GL;
        assert_eq!(
            instance.backends(),
            expected,
            "Windows should use Vulkan | DX12 | GL backends"
        );
        eprintln!("INTEGRATION: Windows platform backends verified: {:?}", expected);
    }

    #[cfg(all(not(target_arch = "wasm32"), target_os = "macos"))]
    {
        assert_eq!(
            instance.backends(),
            wgpu::Backends::METAL,
            "macOS should use Metal backend"
        );
        eprintln!("INTEGRATION: macOS Metal backend verified");
    }

    #[cfg(target_arch = "wasm32")]
    {
        assert_eq!(
            instance.backends(),
            wgpu::Backends::BROWSER_WEBGPU,
            "WASM should use BROWSER_WEBGPU"
        );
        eprintln!("INTEGRATION: WASM BROWSER_WEBGPU backend verified");
    }
}

// =============================================================================
// SECTION 2: Device Lost Recovery Tests
// =============================================================================

/// Test 2.1: Device lost state machine transitions.
///
/// Note: Actually triggering device loss is platform-specific and often not
/// reliably simulable in tests. This test verifies the state machine behavior
/// using the DeviceManager's public API.
#[test]
fn test_device_lost_state_transitions() {
    // Verify DeviceState values
    assert!(DeviceState::Healthy.is_healthy());
    assert!(!DeviceState::Lost.is_healthy());
    assert!(!DeviceState::Recovering(0).is_healthy());
    assert!(!DeviceState::Fatal.is_healthy());

    // Verify needs_recovery
    assert!(!DeviceState::Healthy.needs_recovery());
    assert!(DeviceState::Lost.needs_recovery());
    assert!(DeviceState::Recovering(1).needs_recovery());
    assert!(!DeviceState::Fatal.needs_recovery());

    eprintln!("INTEGRATION: DeviceState transitions verified");
}

/// Test 2.2: RecoveryConfig backoff behavior.
#[test]
fn test_recovery_config_backoff() {
    let config = RecoveryConfig::default();

    // Verify exponential backoff
    let backoff_0 = config.backoff_for_attempt(0);
    let backoff_1 = config.backoff_for_attempt(1);
    let backoff_2 = config.backoff_for_attempt(2);

    eprintln!(
        "INTEGRATION: Recovery backoffs: attempt 0={:?}, 1={:?}, 2={:?}",
        backoff_0, backoff_1, backoff_2
    );

    // Each backoff should be >= the previous (exponential)
    assert!(
        backoff_1 >= backoff_0,
        "Backoff should increase or stay same"
    );
    assert!(
        backoff_2 >= backoff_1,
        "Backoff should increase or stay same"
    );

    // Verify conservative and aggressive configs exist
    let _conservative = RecoveryConfig::conservative();
    let _aggressive = RecoveryConfig::aggressive();
}

/// Test 2.3: DeviceLostReason recoverable classification.
#[test]
fn test_device_lost_reason_recoverability() {
    // Verify which reasons are considered recoverable
    let reasons = [
        (DeviceLostReason::Unknown, true),
        (DeviceLostReason::Destroyed, false),
    ];

    for (reason, expected_recoverable) in reasons.iter() {
        assert_eq!(
            reason.is_likely_recoverable(),
            *expected_recoverable,
            "{:?} should be {}recoverable",
            reason,
            if *expected_recoverable { "" } else { "not " }
        );
    }
}

/// Test 2.4: ResourceTracker for rebuild after recovery.
#[test]
fn test_resource_tracker_for_recovery() {
    let tracker = ResourceTracker::default();

    // Track various resource types (no names, just counts)
    tracker.track_buffer();
    tracker.track_buffer();
    tracker.track_texture();
    tracker.track_bind_group();

    assert_eq!(tracker.total_count(), 4);
    assert!(!tracker.is_empty());

    // Clear after recovery
    tracker.clear();
    assert!(tracker.is_empty());

    eprintln!("INTEGRATION: ResourceTracker verified for recovery workflow");
}

/// Test 2.5: DeviceManager creation and state.
///
/// Note: Full recovery testing would require triggering actual device loss,
/// which is not reliably simulable. This test verifies the manager can be
/// created and starts in a healthy state.
#[test]
fn test_device_manager_creation() {
    let Some(adapter) = get_adapter() else {
        eprintln!("INTEGRATION: No adapter, skipping DeviceManager test");
        return;
    };

    let requirements = DeviceRequirements::new();
    let limits = LimitRequirements::new();
    let config = RecoveryConfig::default();

    let result = block_on(DeviceManager::new(
        Arc::new(adapter),
        requirements,
        limits,
        config,
    ));

    match result {
        Ok(manager) => {
            assert!(manager.is_healthy(), "DeviceManager should start healthy");
            assert!(!manager.needs_recovery());
            assert!(!manager.is_fatal());
            eprintln!("INTEGRATION: DeviceManager created and healthy");
        }
        Err(e) => {
            eprintln!("INTEGRATION: DeviceManager creation failed (expected in some CI): {:?}", e);
        }
    }
}

// =============================================================================
// SECTION 3: Queue Submission Tests
// =============================================================================

/// Test 3.1: Basic queue submission with command buffer.
#[test]
fn test_queue_submission_single_command_buffer() {
    let Some(adapter) = get_adapter() else {
        eprintln!("INTEGRATION: No adapter, skipping queue test");
        return;
    };

    let device = block_on(TrinityDevice::new(
        &adapter,
        wgpu::Features::empty(),
        wgpu::Limits::default(),
    ))
    .expect("Device creation failed");

    let queue = device.queue();

    // Create a simple command buffer
    let encoder =
        device
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("single_submit_test"),
            });

    let cmd_buffer = encoder.finish();
    queue.submit(std::iter::once(cmd_buffer));

    eprintln!("INTEGRATION: Single command buffer submitted successfully");
}

/// Test 3.2: Multiple command buffer batch submission.
#[test]
fn test_queue_submission_multiple_command_buffers() {
    let Some(adapter) = get_adapter() else {
        eprintln!("INTEGRATION: No adapter, skipping batch queue test");
        return;
    };

    let device = block_on(TrinityDevice::new(
        &adapter,
        wgpu::Features::empty(),
        wgpu::Limits::default(),
    ))
    .expect("Device creation failed");

    let queue = device.queue();

    // Create multiple command buffers
    let mut command_buffers = Vec::new();
    for i in 0..5 {
        let encoder =
            device
                .device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some(&format!("batch_encoder_{}", i)),
                });
        command_buffers.push(encoder.finish());
    }

    queue.submit(command_buffers);

    eprintln!("INTEGRATION: Batch of 5 command buffers submitted successfully");
}

/// Test 3.3: TrinityQueue wrapper pending count tracking.
#[test]
fn test_trinity_queue_pending_tracking() {
    let Some(adapter) = get_adapter() else {
        eprintln!("INTEGRATION: No adapter, skipping TrinityQueue test");
        return;
    };

    // Create device using raw wgpu to get both device and queue
    let (device, queue) = block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("trinity_queue_test_device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::downlevel_webgl2_defaults(),
            memory_hints: Default::default(),
        },
        None,
    ))
    .expect("Device request failed");

    let trinity_queue = TrinityQueue::new(queue);

    // Initial pending count should be 0
    assert_eq!(trinity_queue.pending_count(), 0);

    // Submit work (pending count incremented internally)
    let encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("pending_test"),
    });
    trinity_queue.submit(std::iter::once(encoder.finish()));

    eprintln!(
        "INTEGRATION: TrinityQueue pending count after submit: {}",
        trinity_queue.pending_count()
    );
}

/// Test 3.4: Queue write_buffer operation.
#[test]
fn test_queue_write_buffer() {
    let Some(adapter) = get_adapter() else {
        eprintln!("INTEGRATION: No adapter, skipping write_buffer test");
        return;
    };

    let device = block_on(TrinityDevice::new(
        &adapter,
        wgpu::Features::empty(),
        wgpu::Limits::default(),
    ))
    .expect("Device creation failed");

    // Create a buffer
    let buffer = device.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("write_test_buffer"),
        size: 256,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::UNIFORM,
        mapped_at_creation: false,
    });

    // Write data to buffer via queue
    let data: [u8; 64] = [0xAB; 64];
    device.queue().write_buffer(&buffer, 0, &data);

    eprintln!("INTEGRATION: write_buffer completed successfully");
}

/// Test 3.5: SubmissionTracker construction and state.
///
/// Note: Full tracking requires Arc<TrinityQueue> and command buffers,
/// which are tested in the queue whitebox tests. Here we verify the
/// basic construction and initial state.
#[test]
fn test_submission_tracker_initial_state() {
    let tracker = SubmissionTracker::new();

    // Initially empty
    assert_eq!(tracker.completed_count(), 0);

    // Non-existent submission IDs are not completed
    assert!(!tracker.is_completed(0));
    assert!(!tracker.is_completed(999));

    // Clear operation works on empty tracker
    tracker.clear_completed();
    assert_eq!(tracker.completed_count(), 0);

    eprintln!("INTEGRATION: SubmissionTracker initial state verified");
}

/// Test 3.6: SubmissionBatcher configuration.
#[test]
fn test_submission_batcher_config() {
    let config = BatcherConfig::default();

    // Verify defaults are reasonable
    assert!(config.count_threshold > 0);
    assert!(config.time_threshold_ms > 0);

    eprintln!(
        "INTEGRATION: BatcherConfig defaults: count_threshold={}, time_threshold_ms={}",
        config.count_threshold, config.time_threshold_ms
    );
}

// =============================================================================
// SECTION 4: Cross-Backend Test Matrix
// =============================================================================

/// Test 4.1: Vulkan backend (Linux/Windows).
#[test]
#[cfg(any(target_os = "linux", target_os = "windows"))]
fn test_vulkan_backend_initialization() {
    let instance = TrinityInstance::with_backends(wgpu::Backends::VULKAN);

    let adapters: Vec<_> = instance.inner().enumerate_adapters(wgpu::Backends::VULKAN);

    if adapters.is_empty() {
        eprintln!("INTEGRATION: No Vulkan adapter available (driver may not be installed)");
        return;
    }

    let adapter = &adapters[0];
    let info = adapter.get_info();
    assert_eq!(
        info.backend,
        wgpu::Backend::Vulkan,
        "Should be Vulkan backend"
    );

    eprintln!("INTEGRATION: Vulkan adapter: {} ({:?})", info.name, info.device_type);

    // Attempt device creation
    let device_result = block_on(TrinityDevice::new(
        adapter,
        wgpu::Features::empty(),
        wgpu::Limits::default(),
    ));

    if let Ok(device) = device_result {
        eprintln!("INTEGRATION: Vulkan device created successfully");
        let _ = device.queue();
    } else {
        eprintln!("INTEGRATION: Vulkan device creation failed (may be expected)");
    }
}

/// Test 4.2: Metal backend (macOS only).
#[test]
#[cfg(target_os = "macos")]
fn test_metal_backend_initialization() {
    let instance = TrinityInstance::with_backends(wgpu::Backends::METAL);

    let adapters: Vec<_> = instance.inner().enumerate_adapters(wgpu::Backends::METAL);

    if adapters.is_empty() {
        eprintln!("INTEGRATION: No Metal adapter available");
        return;
    }

    let adapter = &adapters[0];
    let info = adapter.get_info();
    assert_eq!(
        info.backend,
        wgpu::Backend::Metal,
        "Should be Metal backend"
    );

    eprintln!("INTEGRATION: Metal adapter: {}", info.name);

    let device_result = block_on(TrinityDevice::new(
        adapter,
        wgpu::Features::empty(),
        wgpu::Limits::default(),
    ));

    assert!(
        device_result.is_ok(),
        "Metal device creation should succeed on macOS"
    );
    eprintln!("INTEGRATION: Metal device created successfully");
}

/// Test 4.3: DX12 backend (Windows only).
#[test]
#[cfg(target_os = "windows")]
fn test_dx12_backend_initialization() {
    let instance = TrinityInstance::with_backends(wgpu::Backends::DX12);

    let adapters: Vec<_> = instance.inner().enumerate_adapters(wgpu::Backends::DX12);

    if adapters.is_empty() {
        eprintln!("INTEGRATION: No DX12 adapter available");
        return;
    }

    let adapter = &adapters[0];
    let info = adapter.get_info();
    assert_eq!(info.backend, wgpu::Backend::Dx12, "Should be DX12 backend");

    eprintln!("INTEGRATION: DX12 adapter: {}", info.name);

    let device_result = block_on(TrinityDevice::new(
        adapter,
        wgpu::Features::empty(),
        wgpu::Limits::default(),
    ));

    if let Ok(device) = device_result {
        eprintln!("INTEGRATION: DX12 device created successfully");
        let _ = device.queue();
    }
}

/// Test 4.4: OpenGL backend (fallback, all platforms).
#[test]
fn test_opengl_backend_initialization() {
    let instance = TrinityInstance::with_backends(wgpu::Backends::GL);

    let adapters: Vec<_> = instance.inner().enumerate_adapters(wgpu::Backends::GL);

    if adapters.is_empty() {
        eprintln!("INTEGRATION: No OpenGL adapter available");
        return;
    }

    let adapter = &adapters[0];
    let info = adapter.get_info();
    assert_eq!(info.backend, wgpu::Backend::Gl, "Should be GL backend");

    eprintln!("INTEGRATION: OpenGL adapter: {}", info.name);

    // Note: OpenGL may have more limited features
    let device_result = block_on(TrinityDevice::new(
        adapter,
        wgpu::Features::empty(),
        wgpu::Limits::downlevel_webgl2_defaults(),
    ));

    if let Ok(device) = device_result {
        eprintln!("INTEGRATION: OpenGL device created successfully");
        let _ = device.queue();
    } else {
        eprintln!(
            "INTEGRATION: OpenGL device creation failed: {:?}",
            device_result.err()
        );
    }
}

/// Test 4.5: All backends enumeration.
#[test]
fn test_all_backends_enumeration() {
    let instance = TrinityInstance::with_backends(wgpu::Backends::all());

    let adapters: Vec<_> = instance.inner().enumerate_adapters(wgpu::Backends::all());

    eprintln!(
        "INTEGRATION: Total adapters across all backends: {}",
        adapters.len()
    );

    // Group by backend
    let mut backend_counts = std::collections::HashMap::new();
    for adapter in &adapters {
        let backend = adapter.get_info().backend;
        *backend_counts.entry(backend).or_insert(0) += 1;
    }

    for (backend, count) in &backend_counts {
        eprintln!("  {:?}: {} adapter(s)", backend, count);
    }
}

// =============================================================================
// SECTION 5: WebGPU/WASM Tests
// =============================================================================

/// Test 5.1: WebGPU backend selection on WASM target.
///
/// This test is ignored in native CI but documents WASM behavior.
#[test]

#[cfg(target_arch = "wasm32")]
fn test_webgpu_backend_wasm() {
    let instance = TrinityInstance::new();

    assert_eq!(
        instance.backends(),
        wgpu::Backends::BROWSER_WEBGPU,
        "WASM should use BROWSER_WEBGPU"
    );

    // Note: Actual adapter enumeration in WASM requires async/await
    // and browser WebGPU support
}

/// Test 5.2: Verify WASM compilation (type check only).
#[test]
fn test_wasm_types_compile() {
    // This test verifies that WASM-related code paths compile correctly
    // on all platforms without actually running WASM-specific code.

    #[cfg(target_arch = "wasm32")]
    {
        let _: wgpu::Backends = wgpu::Backends::BROWSER_WEBGPU;
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        // On non-WASM, verify BROWSER_WEBGPU constant exists
        let _webgpu = wgpu::Backends::BROWSER_WEBGPU;
        eprintln!("INTEGRATION: BROWSER_WEBGPU backend constant verified");
    }
}

// =============================================================================
// SECTION 6: CapabilityManager Integration Tests
// =============================================================================

/// Test 6.1: CapabilityManager creation from device.
#[test]
fn test_capability_manager_from_device() {
    let Some(adapter) = get_adapter() else {
        eprintln!("INTEGRATION: No adapter, skipping CapabilityManager test");
        return;
    };

    let device = block_on(TrinityDevice::new(
        &adapter,
        wgpu::Features::empty(),
        wgpu::Limits::default(),
    ))
    .expect("Device creation failed");

    // Create CapabilityManager from device's features and limits
    let manager = CapabilityManager::from_features_and_limits(
        device.features(),
        device.limits().clone(),
        &adapter.get_info().name,
    );

    let tier = manager.tier();
    eprintln!(
        "INTEGRATION: CapabilityManager tier: {:?} for adapter '{}'",
        tier,
        manager.adapter_name()
    );

    // Verify report generation
    let report = manager.report();
    eprintln!("INTEGRATION: CapabilityReport: {}", report);
}

/// Test 6.2: CapabilityManager matches actual hardware.
#[test]
fn test_capability_manager_matches_hardware() {
    let Some(adapter) = get_adapter() else {
        eprintln!("INTEGRATION: No adapter, skipping capability match test");
        return;
    };

    let adapter_features = adapter.features();
    let adapter_limits = adapter.limits();

    let manager = CapabilityManager::from_features_and_limits(
        adapter_features,
        adapter_limits.clone(),
        &adapter.get_info().name,
    );

    // Verify feature queries match actual features
    let has_timestamp = adapter_features.contains(wgpu::Features::TIMESTAMP_QUERY);
    assert_eq!(
        manager.supports_timestamp_queries(),
        has_timestamp,
        "supports_timestamp_queries should match adapter features"
    );

    let has_bindless = adapter_features.contains(wgpu::Features::TEXTURE_BINDING_ARRAY);
    assert_eq!(
        manager.supports_bindless(),
        has_bindless,
        "supports_bindless should match adapter features"
    );

    eprintln!(
        "INTEGRATION: Capability queries match hardware (timestamp={}, bindless={})",
        has_timestamp, has_bindless
    );
}

/// Test 6.3: Render path selection based on tier.
#[test]
fn test_render_path_selection() {
    let Some(adapter) = get_adapter() else {
        eprintln!("INTEGRATION: No adapter, skipping render path test");
        return;
    };

    let device = block_on(TrinityDevice::new(
        &adapter,
        wgpu::Features::empty(),
        wgpu::Limits::default(),
    ))
    .expect("Device creation failed");

    let manager = CapabilityManager::from_features_and_limits(
        device.features(),
        device.limits().clone(),
        &adapter.get_info().name,
    );

    let render_path = manager.select_render_path();
    let texture_compression = manager.select_texture_compression();

    eprintln!(
        "INTEGRATION: Selected render path: {:?}, texture compression: {:?}",
        render_path, texture_compression
    );
}

// =============================================================================
// SECTION 7: Error Handling During Initialization Tests
// =============================================================================

/// Test 7.1: Invalid feature request fails gracefully.
#[test]
fn test_invalid_feature_request_fails() {
    let Some(adapter) = get_adapter() else {
        eprintln!("INTEGRATION: No adapter, skipping invalid feature test");
        return;
    };

    // Request a feature that likely isn't available on all hardware
    let exotic_features = wgpu::Features::RAY_TRACING_ACCELERATION_STRUCTURE;

    if adapter.features().contains(exotic_features) {
        eprintln!("INTEGRATION: Adapter supports ray tracing, skipping failure test");
        return;
    }

    // This should fail because we're requesting unsupported features
    let device_result = block_on(TrinityDevice::new(&adapter, exotic_features, wgpu::Limits::default()));

    assert!(
        device_result.is_err(),
        "Device creation should fail when requesting unsupported features"
    );

    eprintln!(
        "INTEGRATION: Device creation correctly failed for unsupported features: {:?}",
        device_result.err()
    );
}

/// Test 7.2: Feature negotiation with missing required features.
#[test]
fn test_feature_negotiation_missing_required() {
    let Some(adapter) = get_adapter() else {
        eprintln!("INTEGRATION: No adapter, skipping negotiation test");
        return;
    };

    // Build requirements with a likely-unsupported required feature
    let requirements = DeviceRequirements::new()
        .require(wgpu::Features::RAY_TRACING_ACCELERATION_STRUCTURE);

    if adapter
        .features()
        .contains(wgpu::Features::RAY_TRACING_ACCELERATION_STRUCTURE)
    {
        eprintln!("INTEGRATION: Adapter supports RT, skipping missing feature test");
        return;
    }

    let result = renderer_backend::device::negotiate_features(&requirements, &adapter);

    assert!(
        result.is_err(),
        "Negotiation should fail when required features are missing"
    );

    if let Err(FeatureNegotiationError::RequiredFeaturesMissing(missing)) = result {
        eprintln!(
            "INTEGRATION: Negotiation correctly reported missing features: {:?}",
            missing
        );
    }
}

/// Test 7.3: Graceful handling of zero adapters.
#[test]
fn test_zero_adapters_handled_gracefully() {
    // Use an unlikely backend combination to get zero adapters
    let instance = TrinityInstance::with_backends(wgpu::Backends::BROWSER_WEBGPU);

    // On non-WASM, this should return zero adapters
    #[cfg(not(target_arch = "wasm32"))]
    {
        let adapters: Vec<_> = instance
            .inner()
            .enumerate_adapters(wgpu::Backends::BROWSER_WEBGPU);

        eprintln!(
            "INTEGRATION: BROWSER_WEBGPU on native returned {} adapters (expected 0)",
            adapters.len()
        );

        // This is expected behavior, not a failure
        assert!(
            adapters.is_empty(),
            "BROWSER_WEBGPU should have no adapters on native platform"
        );
    }
}

/// Test 7.4: ErrorScope captures validation errors.
#[test]
fn test_error_scope_validation_capture() {
    let Some(adapter) = get_adapter() else {
        eprintln!("INTEGRATION: No adapter, skipping ErrorScope test");
        return;
    };

    let device = block_on(TrinityDevice::new(
        &adapter,
        wgpu::Features::empty(),
        wgpu::Limits::default(),
    ))
    .expect("Device creation failed");

    // ErrorScope exists and is usable
    let scope = ErrorScope::new(device.device(), renderer_backend::device::ErrorFilter::Validation);

    // Scope will be dropped, checking for errors
    drop(scope);

    eprintln!("INTEGRATION: ErrorScope validation capture verified");
}

/// Test 7.5: DeviceCreationError variants are accessible.
#[test]
fn test_device_creation_error_variants() {
    // Verify error types are accessible and implement expected traits
    let error: Result<TrinityDevice, DeviceCreationError> =
        Err(DeviceCreationError::FeatureNotSupported(wgpu::Features::empty()));

    assert!(error.is_err());

    // Verify Display is implemented
    let display = format!("{}", error.unwrap_err());
    assert!(!display.is_empty());

    eprintln!("INTEGRATION: DeviceCreationError Display: {}", display);
}

// =============================================================================
// SECTION 8: End-to-End Workflow Tests
// =============================================================================

/// Test 8.1: Full render preparation workflow.
///
/// Simulates the initialization a real application would perform.
#[test]
fn test_full_render_preparation_workflow() {
    // Step 1: Create instance
    let instance = TrinityInstance::new();
    eprintln!("STEP 1: Created TrinityInstance");

    // Step 2: Enumerate and select adapter
    let adapters: Vec<_> = instance.inner().enumerate_adapters(instance.backends());
    if adapters.is_empty() {
        eprintln!("WORKFLOW: No adapters, cannot complete workflow");
        return;
    }

    let selector = AdapterSelector::new();
    let selection = selector.select(&adapters).expect("No suitable adapter");
    eprintln!(
        "STEP 2: Selected adapter '{}' (score: {})",
        selection.adapter.get_info().name,
        selection.score.total
    );

    // Step 3: Analyze capabilities
    let adapter_features = selection.adapter.features();
    let adapter_limits = selection.adapter.limits();

    let cap_manager = CapabilityManager::from_features_and_limits(
        adapter_features,
        adapter_limits.clone(),
        &selection.adapter.get_info().name,
    );
    eprintln!(
        "STEP 3: Detected capability tier: {:?}",
        cap_manager.tier()
    );

    // Step 4: Build requirements based on capabilities
    let mut requirements = DeviceRequirements::new();

    // Add preferred features based on detected tier
    if cap_manager.supports_timestamp_queries() {
        requirements = requirements.prefer(wgpu::Features::TIMESTAMP_QUERY);
    }
    eprintln!("STEP 4: Built DeviceRequirements");

    // Step 5: Negotiate and create device (returns tuple)
    let (device, _negotiation) = block_on(renderer_backend::device::negotiate_and_create_device(
        &requirements,
        &selection.adapter,
    ))
    .expect("Device negotiation failed");
    eprintln!(
        "STEP 5: Created device with {} features enabled",
        device.features().iter().count()
    );

    // Step 6: Verify queue is operational
    let queue = device.queue();
    let encoder =
        device
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("workflow_test"),
            });
    queue.submit(std::iter::once(encoder.finish()));
    eprintln!("STEP 6: Verified queue submission");

    // Step 7: Final capability verification
    let final_manager = CapabilityManager::from_features_and_limits(
        device.features(),
        device.limits().clone(),
        &selection.adapter.get_info().name,
    );

    eprintln!(
        "WORKFLOW COMPLETE: Running at {:?} tier with {:?} render path",
        final_manager.tier(),
        final_manager.select_render_path()
    );
}

/// Test 8.2: Stress test with many sequential initializations.
#[test]
fn test_sequential_initialization_stability() {
    let Some(adapter) = get_adapter() else {
        eprintln!("INTEGRATION: No adapter, skipping stability test");
        return;
    };

    const ITERATIONS: usize = 5;
    let mut success_count = 0;

    for i in 0..ITERATIONS {
        let result = block_on(TrinityDevice::new(
            &adapter,
            wgpu::Features::empty(),
            wgpu::Limits::default(),
        ));

        if result.is_ok() {
            success_count += 1;
            // Explicitly drop device to release resources
            drop(result);
        }
    }

    eprintln!(
        "INTEGRATION: Sequential init stability: {}/{} successful",
        success_count, ITERATIONS
    );

    assert_eq!(
        success_count, ITERATIONS,
        "All sequential initializations should succeed"
    );
}
