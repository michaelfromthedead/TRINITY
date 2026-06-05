// Blackbox contract tests for T-WGPU-P7.3.3 Debug Validation Layer.
//
// CLEANROOM: No src/ access beyond the public API exported by the crate.
// Tests use only `renderer_backend::debug::validation::*` -- no internal fields,
// no private methods, no implementation details.
//
// Test coverage:
//   1.  Validation Level Workflows
//   2.  Feature Configuration
//   3.  Message Processing
//   4.  Validation Scopes
//   5.  Real-World Scenarios
//   6.  Callback System
//   7.  Counter Tracking
//   8.  Severity Filtering
//   9.  Object Tracking
//  10.  Thread Safety

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::thread;

use renderer_backend::debug::{
    SourceLocation, ValidationCallbackRegistry, ValidationFeatures, ValidationLayer,
    ValidationLevel, ValidationMessage, ValidationMessageType, ValidationObject,
    ValidationObjectType, ValidationScope, ValidationSeverity,
};

// ============================================================================
// SECTION 1: Validation Level Workflows
// ============================================================================

#[test]
fn validation_level_development_with_full_validation() {
    let layer = ValidationLayer::new(ValidationLevel::Full);
    assert!(layer.is_enabled());
    assert_eq!(layer.level(), ValidationLevel::Full);
    assert!(layer.features().synchronization_validation);
    assert!(layer.features().shader_validation);
    assert!(layer.features().best_practices_warnings);
}

#[test]
fn validation_level_release_with_disabled_validation() {
    let layer = ValidationLayer::new(ValidationLevel::Disabled);
    assert!(!layer.is_enabled());
    assert_eq!(layer.level(), ValidationLevel::Disabled);
    assert!(!layer.features().any_enabled());
}

#[test]
fn validation_level_debugging_with_verbose_validation() {
    let layer = ValidationLayer::new(ValidationLevel::Verbose);
    assert!(layer.is_enabled());
    assert_eq!(layer.level(), ValidationLevel::Verbose);
    assert!(layer.features().gpu_based_validation);
    assert!(layer.features().printf_to_stdout);
    assert!(layer.features().synchronization_validation);
}

#[test]
fn validation_level_ci_with_basic_validation() {
    let layer = ValidationLayer::new(ValidationLevel::Basic);
    assert!(layer.is_enabled());
    assert_eq!(layer.level(), ValidationLevel::Basic);
    assert!(layer.features().shader_validation);
    assert!(!layer.features().gpu_based_validation);
    assert!(!layer.features().synchronization_validation);
}

#[test]
fn validation_level_from_str_disabled_variants() {
    assert_eq!(ValidationLevel::from_str("disabled"), ValidationLevel::Disabled);
    assert_eq!(ValidationLevel::from_str("none"), ValidationLevel::Disabled);
    assert_eq!(ValidationLevel::from_str("off"), ValidationLevel::Disabled);
    assert_eq!(ValidationLevel::from_str("0"), ValidationLevel::Disabled);
    assert_eq!(ValidationLevel::from_str("false"), ValidationLevel::Disabled);
    assert_eq!(ValidationLevel::from_str("DISABLED"), ValidationLevel::Disabled);
    assert_eq!(ValidationLevel::from_str("OFF"), ValidationLevel::Disabled);
}

#[test]
fn validation_level_from_str_basic_variants() {
    assert_eq!(ValidationLevel::from_str("basic"), ValidationLevel::Basic);
    assert_eq!(ValidationLevel::from_str("min"), ValidationLevel::Basic);
    assert_eq!(ValidationLevel::from_str("1"), ValidationLevel::Basic);
    assert_eq!(ValidationLevel::from_str("BASIC"), ValidationLevel::Basic);
    assert_eq!(ValidationLevel::from_str("MIN"), ValidationLevel::Basic);
}

#[test]
fn validation_level_from_str_full_variants() {
    assert_eq!(ValidationLevel::from_str("full"), ValidationLevel::Full);
    assert_eq!(ValidationLevel::from_str("on"), ValidationLevel::Full);
    assert_eq!(ValidationLevel::from_str("true"), ValidationLevel::Full);
    assert_eq!(ValidationLevel::from_str("2"), ValidationLevel::Full);
    assert_eq!(ValidationLevel::from_str("FULL"), ValidationLevel::Full);
    assert_eq!(ValidationLevel::from_str("ON"), ValidationLevel::Full);
}

#[test]
fn validation_level_from_str_verbose_variants() {
    assert_eq!(ValidationLevel::from_str("verbose"), ValidationLevel::Verbose);
    assert_eq!(ValidationLevel::from_str("max"), ValidationLevel::Verbose);
    assert_eq!(ValidationLevel::from_str("debug"), ValidationLevel::Verbose);
    assert_eq!(ValidationLevel::from_str("3"), ValidationLevel::Verbose);
    assert_eq!(ValidationLevel::from_str("VERBOSE"), ValidationLevel::Verbose);
    assert_eq!(ValidationLevel::from_str("MAX"), ValidationLevel::Verbose);
}

#[test]
fn validation_level_from_str_unknown_defaults_to_basic() {
    assert_eq!(ValidationLevel::from_str("unknown"), ValidationLevel::Basic);
    assert_eq!(ValidationLevel::from_str(""), ValidationLevel::Basic);
    assert_eq!(ValidationLevel::from_str("invalid"), ValidationLevel::Basic);
    assert_eq!(ValidationLevel::from_str("super"), ValidationLevel::Basic);
}

#[test]
fn validation_level_ordering_is_correct() {
    assert!(ValidationLevel::Disabled < ValidationLevel::Basic);
    assert!(ValidationLevel::Basic < ValidationLevel::Full);
    assert!(ValidationLevel::Full < ValidationLevel::Verbose);
}

#[test]
fn validation_level_display_format() {
    assert_eq!(format!("{}", ValidationLevel::Disabled), "Disabled");
    assert_eq!(format!("{}", ValidationLevel::Basic), "Basic");
    assert_eq!(format!("{}", ValidationLevel::Full), "Full");
    assert_eq!(format!("{}", ValidationLevel::Verbose), "Verbose");
}

#[test]
fn validation_level_default_is_disabled() {
    let level: ValidationLevel = Default::default();
    assert_eq!(level, ValidationLevel::Disabled);
}

// ============================================================================
// SECTION 2: Feature Configuration
// ============================================================================

#[test]
fn feature_configuration_gpu_based_validation_only() {
    let mut features = ValidationFeatures::default();
    features.gpu_based_validation = true;
    features.shader_validation = false;
    features.descriptor_indexing_validation = false;
    features.best_practices_warnings = false;

    let layer = ValidationLayer::with_features(ValidationLevel::Full, features);
    assert!(layer.features().gpu_based_validation);
    assert!(!layer.features().shader_validation);
}

#[test]
fn feature_configuration_shader_validation_only() {
    let features = ValidationFeatures {
        gpu_based_validation: false,
        synchronization_validation: false,
        shader_validation: true,
        descriptor_indexing_validation: false,
        best_practices_warnings: false,
        printf_to_stdout: false,
    };

    assert!(features.any_enabled());
    assert_eq!(features.enabled_count(), 1);
}

#[test]
fn feature_configuration_all_features_enabled() {
    let features = ValidationFeatures::all_enabled();

    assert!(features.gpu_based_validation);
    assert!(features.synchronization_validation);
    assert!(features.shader_validation);
    assert!(features.descriptor_indexing_validation);
    assert!(features.best_practices_warnings);
    assert!(features.printf_to_stdout);
    assert_eq!(features.enabled_count(), 6);
}

#[test]
fn feature_configuration_custom_combinations() {
    let features = ValidationFeatures {
        gpu_based_validation: true,
        synchronization_validation: true,
        shader_validation: false,
        descriptor_indexing_validation: false,
        best_practices_warnings: false,
        printf_to_stdout: false,
    };

    assert!(features.any_enabled());
    assert_eq!(features.enabled_count(), 2);
}

#[test]
fn feature_configuration_for_disabled_level() {
    let features = ValidationFeatures::for_level(ValidationLevel::Disabled);
    assert!(!features.any_enabled());
    assert_eq!(features.enabled_count(), 0);
}

#[test]
fn feature_configuration_for_basic_level() {
    let features = ValidationFeatures::for_level(ValidationLevel::Basic);
    assert!(features.shader_validation);
    assert!(features.descriptor_indexing_validation);
    assert!(!features.gpu_based_validation);
    assert!(!features.synchronization_validation);
    assert!(!features.best_practices_warnings);
}

#[test]
fn feature_configuration_for_full_level() {
    let features = ValidationFeatures::for_level(ValidationLevel::Full);
    assert!(features.shader_validation);
    assert!(features.descriptor_indexing_validation);
    assert!(features.synchronization_validation);
    assert!(features.best_practices_warnings);
    assert!(!features.gpu_based_validation);
}

#[test]
fn feature_configuration_for_verbose_level() {
    let features = ValidationFeatures::for_level(ValidationLevel::Verbose);
    assert!(features.gpu_based_validation);
    assert!(features.synchronization_validation);
    assert!(features.shader_validation);
    assert!(features.descriptor_indexing_validation);
    assert!(features.best_practices_warnings);
    assert!(features.printf_to_stdout);
}

#[test]
fn feature_configuration_display_none() {
    let features = ValidationFeatures::for_level(ValidationLevel::Disabled);
    assert_eq!(format!("{}", features), "None");
}

#[test]
fn feature_configuration_display_multiple() {
    let features = ValidationFeatures {
        gpu_based_validation: true,
        synchronization_validation: false,
        shader_validation: true,
        descriptor_indexing_validation: false,
        best_practices_warnings: false,
        printf_to_stdout: false,
    };
    let display = format!("{}", features);
    assert!(display.contains("GPU"));
    assert!(display.contains("Shader"));
    assert!(display.contains("+"));
}

#[test]
fn feature_configuration_default_has_shader_validation() {
    let features = ValidationFeatures::default();
    assert!(features.shader_validation);
    assert!(features.descriptor_indexing_validation);
    assert!(!features.gpu_based_validation);
}

#[test]
fn feature_configuration_new_same_as_default() {
    let new_features = ValidationFeatures::new();
    let default_features = ValidationFeatures::default();
    assert_eq!(new_features, default_features);
}

// ============================================================================
// SECTION 3: Message Processing
// ============================================================================

#[test]
fn message_processing_error_triggers_callback() {
    let layer = ValidationLayer::new(ValidationLevel::Full);
    let received = Arc::new(AtomicBool::new(false));
    let received_clone = received.clone();

    layer.register_callback(Box::new(move |msg| {
        if msg.is_error() {
            received_clone.store(true, Ordering::SeqCst);
        }
    }));

    let msg = ValidationMessage::error("Test error");
    layer.on_message(&msg);

    assert!(received.load(Ordering::SeqCst));
}

#[test]
fn message_processing_warning_filtered_by_threshold() {
    let layer = ValidationLayer::new(ValidationLevel::Disabled);
    let received = Arc::new(AtomicBool::new(false));
    let received_clone = received.clone();

    layer.register_callback(Box::new(move |_| {
        received_clone.store(true, Ordering::SeqCst);
    }));

    // Warnings are below Error threshold for Disabled level
    let msg = ValidationMessage::warning("Test warning");
    layer.on_message(&msg);

    // Warning should be filtered out (threshold is Error for Disabled)
    assert!(!received.load(Ordering::SeqCst));
}

#[test]
fn message_processing_performance_warning_classification() {
    let msg = ValidationMessage::performance("Consider using staging buffer");
    assert_eq!(msg.message_type, ValidationMessageType::Performance);
    assert_eq!(msg.severity, ValidationSeverity::Warning);
    assert!(!msg.is_error());
    assert!(msg.is_warning());
}

#[test]
fn message_processing_multiple_objects_in_message() {
    let msg = ValidationMessage::error("Resource conflict")
        .with_object(ValidationObject::new(ValidationObjectType::Buffer, 1))
        .with_object(ValidationObject::new(ValidationObjectType::Texture, 2))
        .with_object(ValidationObject::new(ValidationObjectType::Sampler, 3));

    assert_eq!(msg.objects.len(), 3);
}

#[test]
fn message_processing_with_objects_batch() {
    let objects = vec![
        ValidationObject::new(ValidationObjectType::Buffer, 1),
        ValidationObject::new(ValidationObjectType::Texture, 2),
    ];

    let msg = ValidationMessage::warning("Multiple resources")
        .with_objects(objects);

    assert_eq!(msg.objects.len(), 2);
}

#[test]
fn message_processing_info_message_creation() {
    let msg = ValidationMessage::info("Device initialized");
    assert_eq!(msg.severity, ValidationSeverity::Info);
    assert_eq!(msg.message_type, ValidationMessageType::General);
    assert!(!msg.is_error());
    assert!(!msg.is_warning());
}

#[test]
fn message_processing_with_message_id() {
    let msg = ValidationMessage::error("Validation error")
        .with_id(1234);
    assert_eq!(msg.message_id, Some(1234));
}

#[test]
fn message_processing_with_source_location() {
    let loc = SourceLocation::new()
        .with_file("test.rs")
        .with_line(42);

    let msg = ValidationMessage::error("Error")
        .with_location(loc);

    assert!(msg.location.is_some());
    let loc = msg.location.as_ref().unwrap();
    assert_eq!(loc.file.as_deref(), Some("test.rs"));
    assert_eq!(loc.line, Some(42));
}

#[test]
fn message_processing_elapsed_time_exists() {
    let msg = ValidationMessage::info("Test");
    let elapsed = msg.elapsed();
    // Just verify it doesn't panic and returns a reasonable value
    assert!(elapsed.as_nanos() >= 0);
}

#[test]
fn message_processing_display_format() {
    let msg = ValidationMessage::error("Invalid buffer usage");
    let display = format!("{}", msg);
    assert!(display.contains("ERROR"));
    assert!(display.contains("Validation"));
    assert!(display.contains("Invalid buffer usage"));
}

#[test]
fn message_processing_meets_threshold() {
    let error = ValidationMessage::error("Error");
    assert!(error.meets_threshold(ValidationSeverity::Warning));
    assert!(error.meets_threshold(ValidationSeverity::Error));

    let warning = ValidationMessage::warning("Warning");
    assert!(warning.meets_threshold(ValidationSeverity::Warning));
    assert!(!warning.meets_threshold(ValidationSeverity::Error));

    let info = ValidationMessage::info("Info");
    assert!(info.meets_threshold(ValidationSeverity::Info));
    assert!(!info.meets_threshold(ValidationSeverity::Warning));
}

// ============================================================================
// SECTION 4: Validation Scopes
// ============================================================================

#[test]
fn validation_scope_frame_validation_scope() {
    let layer = ValidationLayer::new(ValidationLevel::Full);

    {
        let scope = layer.scope("Frame Validation").silent();
        assert_eq!(scope.name(), "Frame Validation");
        assert!(!scope.has_errors());
        assert!(!scope.has_warnings());
    }
}

#[test]
fn validation_scope_draw_call_validation_scope() {
    let layer = ValidationLayer::new(ValidationLevel::Full);

    {
        let scope = layer.scope("Draw Call").silent();

        // Simulate a warning during draw call
        layer.on_message(&ValidationMessage::warning("Unused vertex attribute"));

        assert!(scope.has_warnings());
        assert!(!scope.has_errors());
        assert_eq!(scope.scope_warnings(), 1);
    }
}

#[test]
fn validation_scope_nested_validation_scopes() {
    let layer = ValidationLayer::new(ValidationLevel::Full);

    {
        let outer_scope = layer.scope("Render Pass").silent();

        layer.on_message(&ValidationMessage::warning("Outer warning"));

        {
            let inner_scope = layer.scope("Shadow Pass").silent();

            layer.on_message(&ValidationMessage::error("Inner error"));

            assert_eq!(inner_scope.scope_errors(), 1);
            assert_eq!(inner_scope.scope_warnings(), 0);
        }

        // Outer scope sees all messages from the start
        assert_eq!(outer_scope.scope_errors(), 1);
        assert_eq!(outer_scope.scope_warnings(), 1);
    }
}

#[test]
fn validation_scope_error_reporting() {
    let layer = ValidationLayer::new(ValidationLevel::Full);

    {
        let scope = ValidationScope::new(&layer, "Error Test").silent();

        layer.on_message(&ValidationMessage::error("First error"));
        layer.on_message(&ValidationMessage::error("Second error"));

        assert!(scope.has_errors());
        assert_eq!(scope.scope_errors(), 2);
    }
}

#[test]
fn validation_scope_end_returns_result() {
    let layer = ValidationLayer::new(ValidationLevel::Full);
    let scope = ValidationScope::new(&layer, "Test Scope").silent();

    layer.on_message(&ValidationMessage::warning("Warning"));
    layer.on_message(&ValidationMessage::error("Error"));

    let result = scope.end();

    assert_eq!(result.name, "Test Scope");
    assert_eq!(result.errors, 1);
    assert_eq!(result.warnings, 1);
    assert!(!result.is_clean());
    assert!(result.has_errors());
    assert!(result.has_warnings());
}

#[test]
fn validation_scope_result_is_clean() {
    let layer = ValidationLayer::new(ValidationLevel::Full);
    let scope = ValidationScope::new(&layer, "Clean Scope").silent();

    let result = scope.end();

    assert!(result.is_clean());
    assert!(!result.has_errors());
    assert!(!result.has_warnings());
    assert_eq!(result.errors, 0);
    assert_eq!(result.warnings, 0);
}

#[test]
fn validation_scope_tracks_only_scope_messages() {
    let layer = ValidationLayer::new(ValidationLevel::Full);

    // Pre-scope messages
    layer.on_message(&ValidationMessage::error("Pre-scope error"));
    layer.on_message(&ValidationMessage::warning("Pre-scope warning"));

    {
        let scope = ValidationScope::new(&layer, "Scoped").silent();

        // Scope should not include pre-scope messages
        assert_eq!(scope.scope_errors(), 0);
        assert_eq!(scope.scope_warnings(), 0);

        // Add in-scope messages
        layer.on_message(&ValidationMessage::error("In-scope error"));

        assert_eq!(scope.scope_errors(), 1);
        assert_eq!(scope.scope_warnings(), 0);
    }

    // Total layer counts include all messages
    assert_eq!(layer.error_count(), 2);
    assert_eq!(layer.warning_count(), 1);
}

#[test]
fn validation_scope_elapsed_time() {
    let layer = ValidationLayer::new(ValidationLevel::Full);
    let scope = ValidationScope::new(&layer, "Timed").silent();

    // Just verify elapsed works
    let elapsed = scope.elapsed();
    assert!(elapsed.as_nanos() >= 0);
}

#[test]
fn validation_scope_summary() {
    let layer = ValidationLayer::new(ValidationLevel::Full);
    let scope = ValidationScope::new(&layer, "Summary Test").silent();

    layer.on_message(&ValidationMessage::warning("Test"));

    let summary = scope.summary();
    assert!(summary.contains("Summary Test"));
    assert!(summary.contains("warnings"));
}

#[test]
fn validation_scope_result_display() {
    let layer = ValidationLayer::new(ValidationLevel::Full);
    let scope = ValidationScope::new(&layer, "Display Test").silent();
    let result = scope.end();

    let display = format!("{}", result);
    assert!(display.contains("Display Test"));
}

// ============================================================================
// SECTION 5: Real-World Scenarios
// ============================================================================

#[test]
fn real_world_pipeline_creation_validation() {
    let layer = ValidationLayer::new(ValidationLevel::Full);

    // Simulate pipeline creation validation error
    let msg = ValidationMessage::error("Pipeline creation failed: missing vertex shader")
        .with_object(ValidationObject::new(ValidationObjectType::RenderPipeline, 0x100).with_name("MainPipeline"))
        .with_object(ValidationObject::new(ValidationObjectType::ShaderModule, 0x200).with_name("VertexShader"));

    layer.on_message(&msg);

    assert!(layer.has_errors());
    assert_eq!(layer.error_count(), 1);
}

#[test]
fn real_world_resource_binding_validation() {
    let layer = ValidationLayer::new(ValidationLevel::Full);

    let msg = ValidationMessage::error("Bind group incompatible with pipeline layout")
        .with_object(ValidationObject::new(ValidationObjectType::BindGroup, 0x300))
        .with_object(ValidationObject::new(ValidationObjectType::PipelineLayout, 0x400));

    layer.on_message(&msg);

    assert!(layer.has_errors());
}

#[test]
fn real_world_shader_compilation_error() {
    let layer = ValidationLayer::new(ValidationLevel::Full);

    let loc = SourceLocation::new()
        .with_file("shaders/pbr.wgsl")
        .with_line(42)
        .with_column(10);

    let msg = ValidationMessage::error("Shader compilation failed: undefined identifier 'albedo'")
        .with_object(ValidationObject::new(ValidationObjectType::ShaderModule, 0x500).with_name("PBR"))
        .with_location(loc);

    layer.on_message(&msg);

    assert!(layer.has_errors());
    assert!(msg.location.is_some());
}

#[test]
fn real_world_synchronization_warning() {
    let layer = ValidationLayer::new(ValidationLevel::Full);

    let msg = ValidationMessage::warning("Texture read without proper barrier")
        .with_object(ValidationObject::new(ValidationObjectType::Texture, 0x600).with_name("ShadowMap"));

    layer.on_message(&msg);

    assert!(layer.has_warnings());
    assert!(!layer.has_errors());
}

#[test]
fn real_world_best_practices_violation() {
    let layer = ValidationLayer::new(ValidationLevel::Full);

    let msg = ValidationMessage::new(
        ValidationSeverity::Warning,
        ValidationMessageType::Performance,
        "Consider using a staging buffer for frequent CPU->GPU transfers",
    )
    .with_object(ValidationObject::new(ValidationObjectType::Buffer, 0x700).with_name("UniformBuffer"));

    layer.on_message(&msg);

    assert!(layer.has_warnings());
}

#[test]
fn real_world_multiple_validation_errors_in_frame() {
    let layer = ValidationLayer::new(ValidationLevel::Full);

    {
        let frame_scope = layer.scope("Frame 100").silent();

        layer.on_message(&ValidationMessage::error("Missing bind group slot 0"));
        layer.on_message(&ValidationMessage::error("Vertex buffer too small"));
        layer.on_message(&ValidationMessage::warning("Unused uniform binding"));

        assert_eq!(frame_scope.scope_errors(), 2);
        assert_eq!(frame_scope.scope_warnings(), 1);
    }
}

#[test]
fn real_world_render_pass_validation() {
    let layer = ValidationLayer::new(ValidationLevel::Full);

    {
        let pass_scope = layer.scope("Shadow Pass").silent();

        // No errors in this pass
        assert!(pass_scope.scope_messages() == 0);

        let result = pass_scope.end();
        assert!(result.is_clean());
    }
}

#[test]
fn real_world_compute_dispatch_validation() {
    let layer = ValidationLayer::new(ValidationLevel::Full);

    let msg = ValidationMessage::warning("Dispatch workgroup count exceeds device limit")
        .with_object(ValidationObject::new(ValidationObjectType::ComputePass, 0x800));

    layer.on_message(&msg);

    assert!(layer.has_warnings());
}

#[test]
fn real_world_texture_format_incompatibility() {
    let layer = ValidationLayer::new(ValidationLevel::Full);

    let msg = ValidationMessage::error("Texture format mismatch: expected Rgba8Unorm, got Rgba8UnormSrgb")
        .with_object(ValidationObject::new(ValidationObjectType::Texture, 0x900))
        .with_object(ValidationObject::new(ValidationObjectType::TextureView, 0x901));

    layer.on_message(&msg);

    assert!(layer.has_errors());
}

#[test]
fn real_world_query_set_overflow() {
    let layer = ValidationLayer::new(ValidationLevel::Full);

    let msg = ValidationMessage::error("QuerySet overflow: index 64 exceeds size 64")
        .with_object(ValidationObject::new(ValidationObjectType::QuerySet, 0xA00));

    layer.on_message(&msg);

    assert!(layer.has_errors());
}

// ============================================================================
// SECTION 6: Callback System
// ============================================================================

#[test]
fn callback_system_register_error_handler() {
    let layer = ValidationLayer::new(ValidationLevel::Full);
    let error_count = Arc::new(AtomicU64::new(0));
    let error_count_clone = error_count.clone();

    layer.register_callback(Box::new(move |msg| {
        if msg.is_error() {
            error_count_clone.fetch_add(1, Ordering::SeqCst);
        }
    }));

    layer.on_message(&ValidationMessage::error("Error 1"));
    layer.on_message(&ValidationMessage::warning("Warning 1"));
    layer.on_message(&ValidationMessage::error("Error 2"));

    assert_eq!(error_count.load(Ordering::SeqCst), 2);
}

#[test]
fn callback_system_multiple_concurrent_callbacks() {
    let layer = ValidationLayer::new(ValidationLevel::Full);
    let callback1_count = Arc::new(AtomicU64::new(0));
    let callback2_count = Arc::new(AtomicU64::new(0));
    let callback3_count = Arc::new(AtomicU64::new(0));

    let c1 = callback1_count.clone();
    let c2 = callback2_count.clone();
    let c3 = callback3_count.clone();

    layer.register_callback(Box::new(move |_| {
        c1.fetch_add(1, Ordering::SeqCst);
    }));

    layer.register_callback(Box::new(move |_| {
        c2.fetch_add(1, Ordering::SeqCst);
    }));

    layer.register_callback(Box::new(move |_| {
        c3.fetch_add(1, Ordering::SeqCst);
    }));

    layer.on_message(&ValidationMessage::warning("Test"));

    assert_eq!(callback1_count.load(Ordering::SeqCst), 1);
    assert_eq!(callback2_count.load(Ordering::SeqCst), 1);
    assert_eq!(callback3_count.load(Ordering::SeqCst), 1);
}

#[test]
fn callback_system_invocation_order() {
    use std::sync::Mutex;

    let layer = ValidationLayer::new(ValidationLevel::Full);
    let order = Arc::new(Mutex::new(Vec::new()));

    let order1 = order.clone();
    let order2 = order.clone();
    let order3 = order.clone();

    layer.register_callback(Box::new(move |_| {
        order1.lock().unwrap().push(1);
    }));

    layer.register_callback(Box::new(move |_| {
        order2.lock().unwrap().push(2);
    }));

    layer.register_callback(Box::new(move |_| {
        order3.lock().unwrap().push(3);
    }));

    layer.on_message(&ValidationMessage::info("Test"));

    let result = order.lock().unwrap();
    assert_eq!(*result, vec![1, 2, 3]);
}

#[test]
fn callback_system_clear_callbacks() {
    let registry = ValidationCallbackRegistry::new();
    let count = Arc::new(AtomicU64::new(0));
    let count_clone = count.clone();

    registry.register(Box::new(move |_| {
        count_clone.fetch_add(1, Ordering::SeqCst);
    }));

    assert!(!registry.is_empty());

    registry.clear();

    assert!(registry.is_empty());
    assert_eq!(registry.len(), 0);

    // Invoke should do nothing now
    registry.invoke(&ValidationMessage::info("Test"));
    assert_eq!(count.load(Ordering::SeqCst), 0);
}

#[test]
fn callback_system_registry_len() {
    let registry = ValidationCallbackRegistry::new();
    assert_eq!(registry.len(), 0);

    registry.register(Box::new(|_| {}));
    assert_eq!(registry.len(), 1);

    registry.register(Box::new(|_| {}));
    assert_eq!(registry.len(), 2);
}

#[test]
fn callback_system_registry_is_empty() {
    let registry = ValidationCallbackRegistry::new();
    assert!(registry.is_empty());

    registry.register(Box::new(|_| {}));
    assert!(!registry.is_empty());
}

#[test]
fn callback_system_layer_callbacks_access() {
    let layer = ValidationLayer::new(ValidationLevel::Full);
    let callbacks = layer.callbacks();

    // Should be able to register through the returned reference
    callbacks.register(Box::new(|_| {}));
    assert_eq!(callbacks.len(), 1);
}

// ============================================================================
// SECTION 7: Counter Tracking
// ============================================================================

#[test]
fn counter_tracking_count_errors_per_frame() {
    let layer = ValidationLayer::new(ValidationLevel::Full);

    assert_eq!(layer.error_count(), 0);

    layer.on_message(&ValidationMessage::error("Error 1"));
    assert_eq!(layer.error_count(), 1);

    layer.on_message(&ValidationMessage::error("Error 2"));
    assert_eq!(layer.error_count(), 2);

    layer.on_message(&ValidationMessage::error("Error 3"));
    assert_eq!(layer.error_count(), 3);
}

#[test]
fn counter_tracking_reset_counters_between_frames() {
    let layer = ValidationLayer::new(ValidationLevel::Full);

    layer.on_message(&ValidationMessage::error("Error"));
    layer.on_message(&ValidationMessage::warning("Warning"));
    layer.on_message(&ValidationMessage::info("Info"));

    assert!(layer.has_errors());
    assert!(layer.has_warnings());
    assert!(layer.message_count() > 0);

    layer.reset_counts();

    assert!(!layer.has_errors());
    assert!(!layer.has_warnings());
    assert_eq!(layer.message_count(), 0);
    assert_eq!(layer.error_count(), 0);
    assert_eq!(layer.warning_count(), 0);
}

#[test]
fn counter_tracking_accumulate_warnings() {
    let layer = ValidationLayer::new(ValidationLevel::Full);

    for i in 0..10 {
        layer.on_message(&ValidationMessage::warning(format!("Warning {}", i)));
    }

    assert_eq!(layer.warning_count(), 10);
    assert!(!layer.has_errors());
    assert!(layer.has_warnings());
}

#[test]
fn counter_tracking_summary_generation() {
    let layer = ValidationLayer::new(ValidationLevel::Full);

    layer.on_message(&ValidationMessage::error("E1"));
    layer.on_message(&ValidationMessage::error("E2"));
    layer.on_message(&ValidationMessage::warning("W1"));

    let summary = layer.summary();

    assert!(summary.contains("Full"));
    assert!(summary.contains("2 errors"));
    assert!(summary.contains("1 warnings"));
    assert!(summary.contains("3 total"));
}

#[test]
fn counter_tracking_total_message_count() {
    let layer = ValidationLayer::new(ValidationLevel::Full);

    layer.on_message(&ValidationMessage::error("E"));
    layer.on_message(&ValidationMessage::warning("W"));
    layer.on_message(&ValidationMessage::info("I"));

    assert_eq!(layer.message_count(), 3);
    assert_eq!(layer.error_count(), 1);
    assert_eq!(layer.warning_count(), 1);
}

#[test]
fn counter_tracking_has_errors_false_initially() {
    let layer = ValidationLayer::new(ValidationLevel::Full);
    assert!(!layer.has_errors());
}

#[test]
fn counter_tracking_has_warnings_false_initially() {
    let layer = ValidationLayer::new(ValidationLevel::Full);
    assert!(!layer.has_warnings());
}

#[test]
fn counter_tracking_message_count_zero_initially() {
    let layer = ValidationLayer::new(ValidationLevel::Full);
    assert_eq!(layer.message_count(), 0);
}

// ============================================================================
// SECTION 8: Severity Filtering
// ============================================================================

#[test]
fn severity_filtering_basic_level_errors_only() {
    let layer = ValidationLayer::new(ValidationLevel::Basic);

    // Threshold is Warning for Basic
    layer.on_message(&ValidationMessage::info("Info")); // Filtered
    layer.on_message(&ValidationMessage::warning("Warning")); // Passes
    layer.on_message(&ValidationMessage::error("Error")); // Passes

    assert_eq!(layer.message_count(), 2);
}

#[test]
fn severity_filtering_full_level_errors_and_warnings() {
    let layer = ValidationLayer::new(ValidationLevel::Full);

    // Threshold is Info for Full
    layer.on_message(&ValidationMessage::new(
        ValidationSeverity::Verbose,
        ValidationMessageType::General,
        "Verbose",
    )); // Filtered
    layer.on_message(&ValidationMessage::info("Info")); // Passes
    layer.on_message(&ValidationMessage::warning("Warning")); // Passes

    assert_eq!(layer.message_count(), 2);
}

#[test]
fn severity_filtering_verbose_level_all_messages() {
    let layer = ValidationLayer::new(ValidationLevel::Verbose);

    // Threshold is Verbose for Verbose level
    layer.on_message(&ValidationMessage::new(
        ValidationSeverity::Verbose,
        ValidationMessageType::DebugMarker,
        "Debug",
    ));
    layer.on_message(&ValidationMessage::info("Info"));
    layer.on_message(&ValidationMessage::warning("Warning"));
    layer.on_message(&ValidationMessage::error("Error"));

    assert_eq!(layer.message_count(), 4);
}

#[test]
fn severity_filtering_disabled_level_errors_only() {
    let layer = ValidationLayer::new(ValidationLevel::Disabled);

    // Threshold is Error for Disabled
    layer.on_message(&ValidationMessage::info("Info")); // Filtered
    layer.on_message(&ValidationMessage::warning("Warning")); // Filtered
    layer.on_message(&ValidationMessage::error("Error")); // Passes

    assert_eq!(layer.message_count(), 1);
    assert!(layer.has_errors());
    assert!(!layer.has_warnings());
}

#[test]
fn severity_filtering_custom_thresholds() {
    // Test severity threshold via level
    assert_eq!(
        ValidationLevel::Disabled.severity_threshold(),
        ValidationSeverity::Error
    );
    assert_eq!(
        ValidationLevel::Basic.severity_threshold(),
        ValidationSeverity::Warning
    );
    assert_eq!(
        ValidationLevel::Full.severity_threshold(),
        ValidationSeverity::Info
    );
    assert_eq!(
        ValidationLevel::Verbose.severity_threshold(),
        ValidationSeverity::Verbose
    );
}

#[test]
fn severity_filtering_severity_ordering() {
    assert!(ValidationSeverity::Verbose < ValidationSeverity::Info);
    assert!(ValidationSeverity::Info < ValidationSeverity::Warning);
    assert!(ValidationSeverity::Warning < ValidationSeverity::Error);
}

#[test]
fn severity_filtering_meets_threshold_method() {
    assert!(ValidationSeverity::Error.meets_threshold(ValidationSeverity::Verbose));
    assert!(ValidationSeverity::Error.meets_threshold(ValidationSeverity::Info));
    assert!(ValidationSeverity::Error.meets_threshold(ValidationSeverity::Warning));
    assert!(ValidationSeverity::Error.meets_threshold(ValidationSeverity::Error));

    assert!(!ValidationSeverity::Info.meets_threshold(ValidationSeverity::Warning));
    assert!(!ValidationSeverity::Warning.meets_threshold(ValidationSeverity::Error));
}

#[test]
fn severity_filtering_should_break() {
    assert!(!ValidationSeverity::Verbose.should_break());
    assert!(!ValidationSeverity::Info.should_break());
    assert!(!ValidationSeverity::Warning.should_break());
    assert!(ValidationSeverity::Error.should_break());
}

#[test]
fn severity_filtering_as_str() {
    assert_eq!(ValidationSeverity::Verbose.as_str(), "VERBOSE");
    assert_eq!(ValidationSeverity::Info.as_str(), "INFO");
    assert_eq!(ValidationSeverity::Warning.as_str(), "WARN");
    assert_eq!(ValidationSeverity::Error.as_str(), "ERROR");
}

#[test]
fn severity_filtering_display() {
    assert_eq!(format!("{}", ValidationSeverity::Verbose), "VERBOSE");
    assert_eq!(format!("{}", ValidationSeverity::Info), "INFO");
    assert_eq!(format!("{}", ValidationSeverity::Warning), "WARN");
    assert_eq!(format!("{}", ValidationSeverity::Error), "ERROR");
}

#[test]
fn severity_filtering_as_log_level() {
    assert_eq!(ValidationSeverity::Verbose.as_log_level(), log::Level::Trace);
    assert_eq!(ValidationSeverity::Info.as_log_level(), log::Level::Info);
    assert_eq!(ValidationSeverity::Warning.as_log_level(), log::Level::Warn);
    assert_eq!(ValidationSeverity::Error.as_log_level(), log::Level::Error);
}

// ============================================================================
// SECTION 9: Object Tracking
// ============================================================================

#[test]
fn object_tracking_buffer_validation_error() {
    let obj = ValidationObject::new(ValidationObjectType::Buffer, 0x1234)
        .with_name("UniformBuffer");

    assert_eq!(obj.object_type, ValidationObjectType::Buffer);
    assert_eq!(obj.handle, 0x1234);
    assert_eq!(obj.name.as_deref(), Some("UniformBuffer"));
    assert!(obj.has_name());
    assert!(obj.object_type.is_resource());
}

#[test]
fn object_tracking_texture_validation_error() {
    let obj = ValidationObject::new(ValidationObjectType::Texture, 0x5678)
        .with_name("Albedo");

    assert_eq!(obj.object_type, ValidationObjectType::Texture);
    assert!(obj.object_type.is_resource());
    assert!(!obj.object_type.is_pipeline());
}

#[test]
fn object_tracking_pipeline_validation_error() {
    let obj = ValidationObject::new(ValidationObjectType::RenderPipeline, 0xABCD)
        .with_name("PBRPipeline");

    assert!(obj.object_type.is_pipeline());
    assert!(!obj.object_type.is_resource());
    assert!(!obj.object_type.is_binding());
}

#[test]
fn object_tracking_multiple_objects_per_error() {
    let msg = ValidationMessage::error("Resource binding mismatch")
        .with_object(ValidationObject::new(ValidationObjectType::BindGroup, 1))
        .with_object(ValidationObject::new(ValidationObjectType::BindGroupLayout, 2))
        .with_object(ValidationObject::new(ValidationObjectType::RenderPipeline, 3));

    assert_eq!(msg.objects.len(), 3);
    assert!(msg.objects[0].object_type.is_binding());
    assert!(msg.objects[1].object_type.is_binding());
    assert!(msg.objects[2].object_type.is_pipeline());
}

#[test]
fn object_tracking_object_type_categories() {
    // Pipeline types
    assert!(ValidationObjectType::RenderPipeline.is_pipeline());
    assert!(ValidationObjectType::ComputePipeline.is_pipeline());
    assert!(ValidationObjectType::PipelineLayout.is_pipeline());

    // Resource types
    assert!(ValidationObjectType::Buffer.is_resource());
    assert!(ValidationObjectType::Texture.is_resource());
    assert!(ValidationObjectType::TextureView.is_resource());
    assert!(ValidationObjectType::Sampler.is_resource());

    // Binding types
    assert!(ValidationObjectType::BindGroup.is_binding());
    assert!(ValidationObjectType::BindGroupLayout.is_binding());

    // Command types
    assert!(ValidationObjectType::CommandBuffer.is_command());
    assert!(ValidationObjectType::CommandEncoder.is_command());
    assert!(ValidationObjectType::RenderPass.is_command());
    assert!(ValidationObjectType::ComputePass.is_command());
}

#[test]
fn object_tracking_unknown_object() {
    let obj = ValidationObject::unknown(0xDEADBEEF);
    assert_eq!(obj.object_type, ValidationObjectType::Unknown);
    assert_eq!(obj.handle, 0xDEADBEEF);
    assert!(!obj.has_name());
}

#[test]
fn object_tracking_set_name() {
    let mut obj = ValidationObject::new(ValidationObjectType::Texture, 0x100);
    assert!(!obj.has_name());

    obj.set_name("DepthTexture");
    assert!(obj.has_name());
    assert_eq!(obj.name.as_deref(), Some("DepthTexture"));
}

#[test]
fn object_tracking_display_string_with_name() {
    let obj = ValidationObject::new(ValidationObjectType::Buffer, 0x123)
        .with_name("VertexBuffer");

    let display = obj.display_string();
    assert!(display.contains("Buffer"));
    assert!(display.contains("0x123"));
    assert!(display.contains("VertexBuffer"));
}

#[test]
fn object_tracking_display_string_without_name() {
    let obj = ValidationObject::new(ValidationObjectType::Sampler, 0x456);

    let display = obj.display_string();
    assert!(display.contains("Sampler"));
    assert!(display.contains("0x456"));
}

#[test]
fn object_tracking_display_trait() {
    let obj = ValidationObject::new(ValidationObjectType::ShaderModule, 0x789)
        .with_name("VertexShader");

    let display = format!("{}", obj);
    assert!(display.contains("ShaderModule"));
}

#[test]
fn object_tracking_all_object_types_display() {
    let types = [
        ValidationObjectType::Unknown,
        ValidationObjectType::Buffer,
        ValidationObjectType::Texture,
        ValidationObjectType::TextureView,
        ValidationObjectType::Sampler,
        ValidationObjectType::BindGroup,
        ValidationObjectType::BindGroupLayout,
        ValidationObjectType::RenderPipeline,
        ValidationObjectType::ComputePipeline,
        ValidationObjectType::PipelineLayout,
        ValidationObjectType::ShaderModule,
        ValidationObjectType::CommandBuffer,
        ValidationObjectType::CommandEncoder,
        ValidationObjectType::RenderPass,
        ValidationObjectType::ComputePass,
        ValidationObjectType::QuerySet,
        ValidationObjectType::Surface,
        ValidationObjectType::Device,
        ValidationObjectType::Queue,
        ValidationObjectType::Adapter,
        ValidationObjectType::Instance,
    ];

    for obj_type in &types {
        let display = format!("{}", obj_type);
        assert!(!display.is_empty());
    }
}

#[test]
fn object_tracking_object_type_default() {
    let default: ValidationObjectType = Default::default();
    assert_eq!(default, ValidationObjectType::Unknown);
}

// ============================================================================
// SECTION 10: Thread Safety
// ============================================================================

#[test]
fn thread_safety_concurrent_message_processing() {
    let layer = Arc::new(ValidationLayer::new(ValidationLevel::Full));
    let mut handles = vec![];

    for i in 0..10 {
        let layer_clone = Arc::clone(&layer);
        handles.push(thread::spawn(move || {
            for j in 0..100 {
                let msg = ValidationMessage::warning(format!("Thread {} Warning {}", i, j));
                layer_clone.on_message(&msg);
            }
        }));
    }

    for handle in handles {
        handle.join().unwrap();
    }

    // All 1000 warnings should be counted
    assert_eq!(layer.warning_count(), 1000);
    assert_eq!(layer.message_count(), 1000);
}

#[test]
fn thread_safety_concurrent_callback_registration() {
    let registry = Arc::new(ValidationCallbackRegistry::new());
    let mut handles = vec![];

    for _ in 0..10 {
        let registry_clone = Arc::clone(&registry);
        handles.push(thread::spawn(move || {
            for _ in 0..10 {
                registry_clone.register(Box::new(|_| {}));
            }
        }));
    }

    for handle in handles {
        handle.join().unwrap();
    }

    assert_eq!(registry.len(), 100);
}

#[test]
fn thread_safety_counter_atomicity() {
    let layer = Arc::new(ValidationLayer::new(ValidationLevel::Full));
    let mut handles = vec![];

    for _ in 0..4 {
        let layer_clone = Arc::clone(&layer);
        handles.push(thread::spawn(move || {
            for _ in 0..250 {
                layer_clone.on_message(&ValidationMessage::error("Error"));
            }
        }));
    }

    for handle in handles {
        handle.join().unwrap();
    }

    assert_eq!(layer.error_count(), 1000);
}

#[test]
fn thread_safety_callback_invocation_under_concurrent_messages() {
    let layer = Arc::new(ValidationLayer::new(ValidationLevel::Full));
    let callback_count = Arc::new(AtomicU64::new(0));

    let callback_count_clone = callback_count.clone();
    layer.register_callback(Box::new(move |_| {
        callback_count_clone.fetch_add(1, Ordering::SeqCst);
    }));

    let mut handles = vec![];

    for _ in 0..8 {
        let layer_clone = Arc::clone(&layer);
        handles.push(thread::spawn(move || {
            for _ in 0..125 {
                layer_clone.on_message(&ValidationMessage::info("Test"));
            }
        }));
    }

    for handle in handles {
        handle.join().unwrap();
    }

    assert_eq!(callback_count.load(Ordering::SeqCst), 1000);
}

#[test]
fn thread_safety_concurrent_reset_and_increment() {
    let layer = Arc::new(ValidationLayer::new(ValidationLevel::Full));
    let layer_inc = Arc::clone(&layer);

    let incrementer = thread::spawn(move || {
        for _ in 0..100 {
            layer_inc.on_message(&ValidationMessage::error("Error"));
        }
    });

    // While incrementing, do some resets (just testing no panic/deadlock)
    for _ in 0..10 {
        layer.reset_counts();
    }

    incrementer.join().unwrap();

    // Final state is indeterminate due to racing, but should be consistent
    let errors = layer.error_count();
    let messages = layer.message_count();
    assert!(errors <= 100);
    assert!(messages <= 100);
}

// ============================================================================
// Additional Edge Cases and Negative Tests
// ============================================================================

#[test]
fn edge_case_empty_message() {
    let msg = ValidationMessage::error("");
    assert_eq!(msg.message, "");
    assert!(msg.is_error());
}

#[test]
fn edge_case_very_long_message() {
    let long_msg = "x".repeat(10000);
    let msg = ValidationMessage::error(&long_msg);
    assert_eq!(msg.message.len(), 10000);
}

#[test]
fn edge_case_zero_handle_object() {
    let obj = ValidationObject::new(ValidationObjectType::Buffer, 0);
    assert_eq!(obj.handle, 0);
}

#[test]
fn edge_case_max_handle_object() {
    let obj = ValidationObject::new(ValidationObjectType::Buffer, u64::MAX);
    assert_eq!(obj.handle, u64::MAX);
}

#[test]
fn edge_case_scope_no_messages() {
    let layer = ValidationLayer::new(ValidationLevel::Full);
    let scope = ValidationScope::new(&layer, "Empty").silent();

    assert_eq!(scope.scope_errors(), 0);
    assert_eq!(scope.scope_warnings(), 0);
    assert_eq!(scope.scope_messages(), 0);
    assert!(!scope.has_errors());
    assert!(!scope.has_warnings());
}

#[test]
fn edge_case_layer_debug_format() {
    let layer = ValidationLayer::new(ValidationLevel::Full);
    let debug_str = format!("{:?}", layer);

    assert!(debug_str.contains("ValidationLayer"));
    assert!(debug_str.contains("Full"));
}

#[test]
fn edge_case_registry_debug_format() {
    let registry = ValidationCallbackRegistry::new();
    registry.register(Box::new(|_| {}));

    let debug_str = format!("{:?}", registry);
    assert!(debug_str.contains("ValidationCallbackRegistry"));
    assert!(debug_str.contains("callback_count"));
}

#[test]
fn edge_case_scope_debug_format() {
    let layer = ValidationLayer::new(ValidationLevel::Full);
    let scope = ValidationScope::new(&layer, "Debug Test").silent();

    let debug_str = format!("{:?}", scope);
    assert!(debug_str.contains("ValidationScope"));
    assert!(debug_str.contains("Debug Test"));
}

#[test]
fn edge_case_break_on_error_setting() {
    let layer = ValidationLayer::new(ValidationLevel::Full);

    assert!(!layer.break_on_error());

    layer.set_break_on_error(true);
    assert!(layer.break_on_error());

    layer.set_break_on_error(false);
    assert!(!layer.break_on_error());
}

#[test]
fn edge_case_message_type_classifications() {
    assert!(ValidationMessageType::Validation.is_error());
    assert!(!ValidationMessageType::General.is_error());
    assert!(!ValidationMessageType::Performance.is_error());
    assert!(!ValidationMessageType::DebugMarker.is_error());

    assert!(ValidationMessageType::Performance.is_performance());
    assert!(!ValidationMessageType::Validation.is_performance());
}

#[test]
fn edge_case_message_type_default_severity() {
    assert_eq!(
        ValidationMessageType::General.default_severity(),
        ValidationSeverity::Info
    );
    assert_eq!(
        ValidationMessageType::Validation.default_severity(),
        ValidationSeverity::Error
    );
    assert_eq!(
        ValidationMessageType::Performance.default_severity(),
        ValidationSeverity::Warning
    );
    assert_eq!(
        ValidationMessageType::DebugMarker.default_severity(),
        ValidationSeverity::Verbose
    );
}

#[test]
fn edge_case_message_type_display() {
    assert_eq!(format!("{}", ValidationMessageType::General), "General");
    assert_eq!(format!("{}", ValidationMessageType::Validation), "Validation");
    assert_eq!(format!("{}", ValidationMessageType::Performance), "Performance");
    assert_eq!(format!("{}", ValidationMessageType::DebugMarker), "DebugMarker");
}

#[test]
fn edge_case_message_type_default() {
    let default: ValidationMessageType = Default::default();
    assert_eq!(default, ValidationMessageType::General);
}

#[test]
fn edge_case_severity_default() {
    let default: ValidationSeverity = Default::default();
    assert_eq!(default, ValidationSeverity::Info);
}

#[test]
fn edge_case_source_location_here() {
    let loc = SourceLocation::here();
    assert!(loc.file.is_some());
    assert!(loc.line.is_some());
    assert!(loc.column.is_some());
    assert!(loc.is_available());
}

#[test]
fn edge_case_source_location_empty() {
    let loc = SourceLocation::new();
    assert!(loc.file.is_none());
    assert!(loc.line.is_none());
    assert!(loc.column.is_none());
    assert!(loc.function.is_none());
    assert!(!loc.is_available());
}

#[test]
fn edge_case_source_location_with_function() {
    let loc = SourceLocation::new()
        .with_function("render_frame");

    assert!(loc.is_available());
    assert_eq!(loc.function.as_deref(), Some("render_frame"));
}

#[test]
fn edge_case_source_location_with_column() {
    let loc = SourceLocation::new()
        .with_file("test.rs")
        .with_line(10)
        .with_column(5);

    assert!(loc.is_available());
    assert_eq!(loc.column, Some(5));
}

#[test]
fn edge_case_default_layer() {
    // Default layer depends on debug_assertions, just verify it doesn't panic
    let _layer: ValidationLayer = Default::default();
}

#[test]
fn edge_case_default_features_for_level() {
    let level = ValidationLevel::Full;
    let features = level.default_features();

    assert!(features.shader_validation);
    assert!(features.synchronization_validation);
}

#[test]
fn negative_test_info_not_counted_as_warning() {
    let layer = ValidationLayer::new(ValidationLevel::Full);

    layer.on_message(&ValidationMessage::info("Info message"));

    assert_eq!(layer.message_count(), 1);
    assert_eq!(layer.warning_count(), 0);
    assert_eq!(layer.error_count(), 0);
}

#[test]
fn negative_test_warning_not_counted_as_error() {
    let layer = ValidationLayer::new(ValidationLevel::Full);

    layer.on_message(&ValidationMessage::warning("Warning message"));

    assert!(!layer.has_errors());
    assert!(layer.has_warnings());
}

#[test]
fn negative_test_object_without_name() {
    let obj = ValidationObject::new(ValidationObjectType::Buffer, 0x100);

    assert!(!obj.has_name());
    assert!(obj.name.is_none());
}

#[test]
fn negative_test_filtered_message_not_counted() {
    let layer = ValidationLayer::new(ValidationLevel::Disabled);

    // Info should be filtered for Disabled level
    layer.on_message(&ValidationMessage::info("Should be filtered"));

    assert_eq!(layer.message_count(), 0);
}
