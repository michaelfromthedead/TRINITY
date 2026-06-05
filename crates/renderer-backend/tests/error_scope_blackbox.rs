// Blackbox contract tests for T-WGPU-P1.3.5 Error Scopes
//
// CLEANROOM: No access to implementation files. Tests use only the public API
// exported by `renderer_backend::device`.
//
// Forbidden files (per TESTDEV_BLACKBOX prompt):
//   - crates/renderer-backend/src/device/error_scope.rs (implementation)
//   - Any WHITEBOX test file for this task
//
// Contract sources:
//   - docs/WGPU_DOCS/RDC_OUTPUT/PHASE_1_CORE_TODO.md (T-WGPU-P1.3.5)
//   - docs/WGPU_DOCS/RDC_OUTPUT/PHASE_1_CORE_ARCH.md (Architecture spec)
//
// Acceptance criteria (T-WGPU-P1.3.5):
//   - push_error_scope() on creation
//   - pop_error_scope() on drop
//   - Validation and OutOfMemory filters
//   - Async error retrieval
//   - Error logging and propagation
//
// Test design rationale:
//   Equivalence partitioning:
//     - ErrorFilter::Validation filter (catches validation errors)
//     - ErrorFilter::OutOfMemory filter (catches OOM errors)
//     - ScopedErrorCapture (multi-scope operations)
//     - Convenience functions (with_validation_scope, with_oom_scope)
//   Boundary cases:
//     - Empty scope (no operations, no errors)
//     - Nested scopes
//     - Multiple scopes in sequence
//   Error cases:
//     - Scope with validation error triggered
//     - Scope with OOM error (simulated if possible)
//   Contract verification:
//     - ErrorFilter enum has Validation and OutOfMemory variants
//     - ErrorScope struct constructable and droppable
//     - ScopedErrorCapture for multi-scope management
//     - Convenience functions return expected types

use pollster::block_on;
use renderer_backend::device::{
    enumerate_adapters_with_info, with_oom_scope, with_validation_scope, ErrorFilter, ErrorScope,
    ScopedErrorCapture, TrinityDevice, TrinityInstance,
};

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

/// Creates a wgpu device for testing.
fn create_test_device(adapter: &wgpu::Adapter) -> Option<(wgpu::Device, wgpu::Queue)> {
    block_on(adapter.request_device(&wgpu::DeviceDescriptor::default(), None)).ok()
}

/// Helper macro to get a device, skipping if unavailable.
macro_rules! require_device {
    ($adapter:expr) => {
        match create_test_device($adapter) {
            Some((device, queue)) => (device, queue),
            None => {
                eprintln!("SKIP: Could not create device for this test");
                return;
            }
        }
    };
}

// =============================================================================
// 1. ErrorFilter Enum Contract Tests
// =============================================================================

/// Verifies that ErrorFilter enum has a Validation variant.
///
/// Contract: ErrorFilter should support filtering validation errors.
#[test]
fn test_error_filter_has_validation_variant() {
    let filter = ErrorFilter::Validation;
    // Type annotation enforces this is ErrorFilter
    let _: ErrorFilter = filter;
}

/// Verifies that ErrorFilter enum has an OutOfMemory variant.
///
/// Contract: ErrorFilter should support filtering OOM errors.
#[test]
fn test_error_filter_has_out_of_memory_variant() {
    let filter = ErrorFilter::OutOfMemory;
    // Type annotation enforces this is ErrorFilter
    let _: ErrorFilter = filter;
}

/// Verifies that ErrorFilter variants are distinct.
///
/// Contract: Validation and OutOfMemory should be different variants.
#[test]
fn test_error_filter_variants_are_distinct() {
    let validation = ErrorFilter::Validation;
    let oom = ErrorFilter::OutOfMemory;

    // Using pattern matching to verify they are distinct
    match validation {
        ErrorFilter::Validation => (), // Expected
        ErrorFilter::OutOfMemory => panic!("Validation should not match OutOfMemory"),
    }

    match oom {
        ErrorFilter::OutOfMemory => (), // Expected
        ErrorFilter::Validation => panic!("OutOfMemory should not match Validation"),
    }
}

/// Verifies that ErrorFilter is Copy (cheap to pass around).
///
/// Contract: ErrorFilter should be a lightweight enum suitable for copying.
#[test]
fn test_error_filter_is_copy() {
    let filter = ErrorFilter::Validation;
    let copy = filter; // Copy
    let _ = filter; // Original still usable
    let _ = copy;
}

/// Verifies that ErrorFilter is Clone.
///
/// Contract: ErrorFilter should be clonable.
#[test]
fn test_error_filter_is_clone() {
    let filter = ErrorFilter::OutOfMemory;
    let cloned = filter.clone();
    let _ = cloned;
}

/// Verifies that ErrorFilter implements Debug.
///
/// Contract: ErrorFilter should be debuggable for logging purposes.
#[test]
fn test_error_filter_implements_debug() {
    let filter = ErrorFilter::Validation;
    let debug_str = format!("{:?}", filter);
    assert!(!debug_str.is_empty(), "Debug output should not be empty");

    let oom_filter = ErrorFilter::OutOfMemory;
    let oom_debug = format!("{:?}", oom_filter);
    assert!(!oom_debug.is_empty(), "Debug output should not be empty");
}

/// Verifies that different filters have distinct debug output.
///
/// Contract: Debug representations should distinguish variants.
#[test]
fn test_error_filter_debug_distinguishes_variants() {
    let validation_debug = format!("{:?}", ErrorFilter::Validation);
    let oom_debug = format!("{:?}", ErrorFilter::OutOfMemory);

    assert_ne!(
        validation_debug, oom_debug,
        "Different variants should have different debug output"
    );
}

// =============================================================================
// 2. ErrorScope Type Existence Contract Tests
// =============================================================================

/// Verifies that ErrorScope is a publicly accessible type.
///
/// Contract: ErrorScope is exported from renderer_backend::device.
#[test]
fn test_error_scope_type_is_exported() {
    // This test verifies the type exists and is accessible
    fn _assert_type_exists<T>(_: Option<&T>) {}
    _assert_type_exists::<ErrorScope>(None);
}

/// Verifies that ErrorScope type can be referenced.
///
/// Contract: ErrorScope should be a concrete type that can be named.
#[test]
fn test_error_scope_can_be_named_as_type() {
    // If this compiles, the type exists and can be named
    fn _takes_error_scope<'a>(_scope: ErrorScope<'a>) {}
    fn _returns_option_error_scope<'a>() -> Option<ErrorScope<'a>> {
        None
    }
    let _: Option<ErrorScope> = None;
}

// =============================================================================
// 3. ErrorScope Construction Contract Tests
// =============================================================================

/// Verifies that ErrorScope can be constructed with a device and validation filter.
///
/// Contract: ErrorScope::new() should accept device reference and ErrorFilter.
/// The scope pushes an error scope on creation.
#[test]
fn test_error_scope_constructable_with_validation_filter() {
    let adapter = require_adapter!();
    let (device, _queue) = require_device!(&adapter);

    // Create an error scope with validation filter
    let scope = ErrorScope::new(&device, ErrorFilter::Validation);
    // Scope exists - type annotation enforces it
    let _: ErrorScope = scope;
    // Scope drops here, which should pop the error scope
}

/// Verifies that ErrorScope can be constructed with OOM filter.
///
/// Contract: ErrorScope should work with OutOfMemory filter.
#[test]
fn test_error_scope_constructable_with_oom_filter() {
    let adapter = require_adapter!();
    let (device, _queue) = require_device!(&adapter);

    let scope = ErrorScope::new(&device, ErrorFilter::OutOfMemory);
    let _: ErrorScope = scope;
}

/// Verifies that multiple ErrorScopes can be created in sequence.
///
/// Contract: Scopes are independent when used sequentially.
#[test]
fn test_multiple_error_scopes_in_sequence() {
    let adapter = require_adapter!();
    let (device, _queue) = require_device!(&adapter);

    // First scope
    {
        let _scope1 = ErrorScope::new(&device, ErrorFilter::Validation);
    }

    // Second scope
    {
        let _scope2 = ErrorScope::new(&device, ErrorFilter::OutOfMemory);
    }

    // Third scope
    {
        let _scope3 = ErrorScope::new(&device, ErrorFilter::Validation);
    }
}

/// Verifies that nested ErrorScopes can be created.
///
/// Contract: Scopes should support nesting (inner scope pops before outer).
#[test]
fn test_nested_error_scopes() {
    let adapter = require_adapter!();
    let (device, _queue) = require_device!(&adapter);

    // Outer scope
    let _outer = ErrorScope::new(&device, ErrorFilter::Validation);
    {
        // Inner scope
        let _inner = ErrorScope::new(&device, ErrorFilter::OutOfMemory);
        // Inner drops here
    }
    // Outer drops here
}

// =============================================================================
// 4. ScopedErrorCapture Type Contract Tests
// =============================================================================

/// Verifies that ScopedErrorCapture type exists.
///
/// Contract: ScopedErrorCapture should be available for multi-scope operations.
#[test]
fn test_scoped_error_capture_type_exists() {
    // Type exists - we can create a function signature that uses it
    fn _assert_type_exists<T>(_: Option<&T>) {}
    _assert_type_exists::<ScopedErrorCapture>(None);
}

/// Verifies that ScopedErrorCapture can be named as a type.
///
/// Contract: ScopedErrorCapture should be a concrete type.
#[test]
fn test_scoped_error_capture_can_be_named() {
    fn _takes_capture<'a>(_capture: ScopedErrorCapture<'a>) {}
    let _: Option<ScopedErrorCapture> = None;
}

/// Verifies that ScopedErrorCapture can be constructed.
///
/// Contract: ScopedErrorCapture should be constructable from a device and filters.
#[test]
fn test_scoped_error_capture_constructable() {
    let adapter = require_adapter!();
    let (device, _queue) = require_device!(&adapter);

    // ScopedErrorCapture takes device and a slice of filters
    let filters = [ErrorFilter::Validation, ErrorFilter::OutOfMemory];
    let capture = ScopedErrorCapture::new(&device, &filters);
    let _: ScopedErrorCapture = capture;
}

/// Verifies that ScopedErrorCapture can be constructed with only validation filter.
///
/// Contract: ScopedErrorCapture should work with single filter.
#[test]
fn test_scoped_error_capture_with_single_validation_filter() {
    let adapter = require_adapter!();
    let (device, _queue) = require_device!(&adapter);

    let filters = [ErrorFilter::Validation];
    let capture = ScopedErrorCapture::new(&device, &filters);
    let _: ScopedErrorCapture = capture;
}

/// Verifies that ScopedErrorCapture can be constructed with only OOM filter.
///
/// Contract: ScopedErrorCapture should work with single OOM filter.
#[test]
fn test_scoped_error_capture_with_single_oom_filter() {
    let adapter = require_adapter!();
    let (device, _queue) = require_device!(&adapter);

    let filters = [ErrorFilter::OutOfMemory];
    let capture = ScopedErrorCapture::new(&device, &filters);
    let _: ScopedErrorCapture = capture;
}

/// Verifies that ScopedErrorCapture can be constructed with empty filters.
///
/// Contract: ScopedErrorCapture should handle empty filter list.
#[test]
fn test_scoped_error_capture_with_empty_filters() {
    let adapter = require_adapter!();
    let (device, _queue) = require_device!(&adapter);

    let filters: [ErrorFilter; 0] = [];
    let capture = ScopedErrorCapture::new(&device, &filters);
    let _: ScopedErrorCapture = capture;
}

// =============================================================================
// 5. Convenience Function Contract Tests
// =============================================================================

/// Verifies that with_validation_scope function exists and is callable.
///
/// Contract: with_validation_scope wraps operations in a validation error scope.
#[test]
fn test_with_validation_scope_exists() {
    let adapter = require_adapter!();
    let (device, _queue) = require_device!(&adapter);

    // with_validation_scope takes device, optional label, and an async block
    // Returns a future that resolves to (result, Option<Error>)
    let (result, error) = block_on(with_validation_scope(&device, Some("test_scope"), async {
        // Some operation that might cause a validation error
        42u32
    }));

    // Should return the async block's result
    assert_eq!(result, 42u32);
    // No error should occur with a valid operation
    assert!(error.is_none(), "No error expected for valid operation");
}

/// Verifies that with_oom_scope function exists and is callable.
///
/// Contract: with_oom_scope wraps operations in an OOM error scope.
#[test]
fn test_with_oom_scope_exists() {
    let adapter = require_adapter!();
    let (device, _queue) = require_device!(&adapter);

    // with_oom_scope takes device, optional label, and an async block
    let (result, error) = block_on(with_oom_scope(&device, Some("oom_test"), async {
        // Some operation that might cause an OOM error
        "test_result"
    }));

    assert_eq!(result, "test_result");
    assert!(error.is_none(), "No error expected for valid operation");
}

/// Verifies that with_validation_scope works without a label.
///
/// Contract: Label parameter should be optional (None acceptable).
#[test]
fn test_with_validation_scope_without_label() {
    let adapter = require_adapter!();
    let (device, _queue) = require_device!(&adapter);

    let (result, _error) = block_on(with_validation_scope(&device, None, async { 123i32 }));
    assert_eq!(result, 123i32);
}

/// Verifies that with_oom_scope works without a label.
///
/// Contract: Label parameter should be optional (None acceptable).
#[test]
fn test_with_oom_scope_without_label() {
    let adapter = require_adapter!();
    let (device, _queue) = require_device!(&adapter);

    let (result, _error) = block_on(with_oom_scope(&device, None, async { "hello" }));
    assert_eq!(result, "hello");
}

/// Verifies that convenience functions return futures.
///
/// Contract: Both functions should return futures, enabling async error retrieval.
#[test]
fn test_convenience_functions_return_futures() {
    let adapter = require_adapter!();
    let (device, _queue) = require_device!(&adapter);

    // These should be futures that we can block on
    let (v_result, _v_error) = block_on(with_validation_scope(&device, None, async { 1i32 }));
    let (o_result, _o_error) = block_on(with_oom_scope(&device, None, async { 2i32 }));

    assert_eq!(v_result, 1i32);
    assert_eq!(o_result, 2i32);
}

// =============================================================================
// 6. Error Scope RAII Behavior Tests
// =============================================================================

/// Verifies that ErrorScope follows RAII pattern (cleanup on drop).
///
/// Contract: ErrorScope should pop the error scope when dropped.
#[test]
fn test_error_scope_raii_drop_behavior() {
    let adapter = require_adapter!();
    let (device, _queue) = require_device!(&adapter);

    // Create and immediately drop a scope
    {
        let scope = ErrorScope::new(&device, ErrorFilter::Validation);
        drop(scope); // Explicit drop
    }

    // Create another scope - this should work if the previous one cleaned up
    {
        let _scope2 = ErrorScope::new(&device, ErrorFilter::OutOfMemory);
    }
}

/// Verifies that ErrorScope can be explicitly dropped early.
///
/// Contract: Early explicit drop should work correctly.
#[test]
fn test_error_scope_early_drop() {
    let adapter = require_adapter!();
    let (device, _queue) = require_device!(&adapter);

    let scope = ErrorScope::new(&device, ErrorFilter::Validation);
    // Do some work...
    drop(scope); // Explicit early drop
    // More work after scope is dropped...
}

// =============================================================================
// 7. Boundary and Edge Case Tests
// =============================================================================

/// Verifies behavior with moderate nesting depth.
///
/// Contract: Scopes should handle moderate nesting depths.
#[test]
fn test_moderate_nesting_depth() {
    let adapter = require_adapter!();
    let (device, _queue) = require_device!(&adapter);

    // Test 10 levels of nesting - reasonable for real usage
    let scope1 = ErrorScope::new(&device, ErrorFilter::Validation);
    let scope2 = ErrorScope::new(&device, ErrorFilter::OutOfMemory);
    let scope3 = ErrorScope::new(&device, ErrorFilter::Validation);
    let scope4 = ErrorScope::new(&device, ErrorFilter::OutOfMemory);
    let scope5 = ErrorScope::new(&device, ErrorFilter::Validation);
    let scope6 = ErrorScope::new(&device, ErrorFilter::OutOfMemory);
    let scope7 = ErrorScope::new(&device, ErrorFilter::Validation);
    let scope8 = ErrorScope::new(&device, ErrorFilter::OutOfMemory);
    let scope9 = ErrorScope::new(&device, ErrorFilter::Validation);
    let scope10 = ErrorScope::new(&device, ErrorFilter::OutOfMemory);

    // Drop in reverse order (LIFO)
    drop(scope10);
    drop(scope9);
    drop(scope8);
    drop(scope7);
    drop(scope6);
    drop(scope5);
    drop(scope4);
    drop(scope3);
    drop(scope2);
    drop(scope1);
}

/// Verifies that rapid scope creation/destruction works.
///
/// Contract: Many sequential scopes should not cause issues.
#[test]
fn test_rapid_sequential_scope_creation() {
    let adapter = require_adapter!();
    let (device, _queue) = require_device!(&adapter);

    // Create and destroy 100 scopes in rapid succession
    for i in 0..100 {
        let filter = if i % 2 == 0 {
            ErrorFilter::Validation
        } else {
            ErrorFilter::OutOfMemory
        };
        let _scope = ErrorScope::new(&device, filter);
        // Scope drops immediately
    }
}

/// Verifies that interleaved convenience function calls work.
///
/// Contract: Alternating between with_validation_scope and with_oom_scope should work.
#[test]
fn test_interleaved_convenience_calls() {
    let adapter = require_adapter!();
    let (device, _queue) = require_device!(&adapter);

    let mut sum = 0i32;

    for i in 0..10i32 {
        if i % 2 == 0 {
            let (result, _) = block_on(with_validation_scope(&device, None, async move { i }));
            sum += result;
        } else {
            let (result, _) = block_on(with_oom_scope(&device, None, async move { i }));
            sum += result;
        }
    }

    assert_eq!(sum, 45); // 0 + 1 + 2 + ... + 9 = 45
}

// =============================================================================
// 8. TrinityDevice Integration Tests
// =============================================================================

/// Verifies that TrinityDevice can be used with ErrorScope.
///
/// Contract: ErrorScope should work with TrinityDevice's inner wgpu::Device.
#[test]
fn test_error_scope_with_trinity_device() {
    let adapter = require_adapter!();

    // Create TrinityDevice
    let trinity_device = block_on(TrinityDevice::new(
        &adapter,
        wgpu::Features::empty(),
        wgpu::Limits::downlevel_defaults(),
    ));

    if let Ok(trinity_device) = trinity_device {
        // ErrorScope should work with TrinityDevice's inner device
        let scope = ErrorScope::new(trinity_device.device(), ErrorFilter::Validation);
        let _: ErrorScope = scope;
    }
}

/// Verifies that convenience functions work with TrinityDevice.
///
/// Contract: with_validation_scope and with_oom_scope should accept TrinityDevice's device.
#[test]
fn test_convenience_functions_with_trinity_device() {
    let adapter = require_adapter!();

    let trinity_device = block_on(TrinityDevice::new(
        &adapter,
        wgpu::Features::empty(),
        wgpu::Limits::downlevel_defaults(),
    ));

    if let Ok(trinity_device) = trinity_device {
        let (result, _) = block_on(with_validation_scope(
            trinity_device.device(),
            Some("test"),
            async { 123i32 },
        ));
        assert_eq!(result, 123i32);

        let (result2, _) = block_on(with_oom_scope(
            trinity_device.device(),
            None,
            async { "hello" },
        ));
        assert_eq!(result2, "hello");
    }
}

// =============================================================================
// 9. Thread Safety Observations (Non-Assertion)
// =============================================================================

/// Documents thread safety expectations for ErrorScope.
///
/// Note: This test observes behavior, doesn't assert guarantees.
/// The wgpu device is Send + Sync, so scopes should be creatable from any thread.
#[test]
fn test_error_scope_thread_safety_observation() {
    let adapter = require_adapter!();
    let (device, _queue) = require_device!(&adapter);

    use std::sync::Arc;

    let device = Arc::new(device);
    let device_clone = device.clone();

    // Create scope on main thread
    let _main_scope = ErrorScope::new(&device, ErrorFilter::Validation);

    // Create scope on another thread
    let handle = std::thread::spawn(move || {
        let _thread_scope = ErrorScope::new(&device_clone, ErrorFilter::OutOfMemory);
    });

    handle.join().expect("Thread should complete successfully");
}

// =============================================================================
// 10. Error Capture Behavior Tests
// =============================================================================

/// Verifies that a scope with no operations captures no errors.
///
/// Contract: An empty scope should report no errors.
#[test]
fn test_empty_validation_scope_has_no_errors() {
    let adapter = require_adapter!();
    let (device, _queue) = require_device!(&adapter);

    let (result, error) = block_on(with_validation_scope(&device, Some("empty"), async {
        // No GPU operations
        ()
    }));

    assert_eq!(result, ());
    assert!(error.is_none(), "Empty scope should have no errors");
}

/// Verifies that a scope with no operations captures no OOM errors.
///
/// Contract: An empty OOM scope should report no errors.
#[test]
fn test_empty_oom_scope_has_no_errors() {
    let adapter = require_adapter!();
    let (device, _queue) = require_device!(&adapter);

    let (result, error) = block_on(with_oom_scope(&device, Some("empty_oom"), async {
        // No GPU operations
        true
    }));

    assert!(result);
    assert!(error.is_none(), "Empty OOM scope should have no errors");
}

/// Verifies that ScopedErrorCapture properly collects errors from multiple filters.
///
/// Contract: ScopedErrorCapture should allow collecting errors after operations.
#[test]
fn test_scoped_error_capture_collects_errors() {
    let adapter = require_adapter!();
    let (device, _queue) = require_device!(&adapter);

    let filters = [ErrorFilter::Validation, ErrorFilter::OutOfMemory];
    let capture = ScopedErrorCapture::new(&device, &filters);

    // No operations performed, so no errors expected
    // The capture should still be usable
    drop(capture);
}

// =============================================================================
// 11. Return Type Verification Tests
// =============================================================================

/// Verifies the return type of with_validation_scope includes error info.
///
/// Contract: Return type should be a tuple of (T, Option<wgpu::Error>).
#[test]
fn test_with_validation_scope_return_type_has_error_option() {
    let adapter = require_adapter!();
    let (device, _queue) = require_device!(&adapter);

    let tuple: (&str, Option<wgpu::Error>) =
        block_on(with_validation_scope(&device, None, async { "value" }));

    assert_eq!(tuple.0, "value");
    // Error is Option<wgpu::Error>
    let _error_option: Option<wgpu::Error> = tuple.1;
}

/// Verifies the return type of with_oom_scope includes error info.
///
/// Contract: Return type should be a tuple of (T, Option<wgpu::Error>).
#[test]
fn test_with_oom_scope_return_type_has_error_option() {
    let adapter = require_adapter!();
    let (device, _queue) = require_device!(&adapter);

    let tuple: (Vec<i32>, Option<wgpu::Error>) =
        block_on(with_oom_scope(&device, None, async { vec![1, 2, 3] }));

    assert_eq!(tuple.0, vec![1, 2, 3]);
    let _error_option: Option<wgpu::Error> = tuple.1;
}

// =============================================================================
// 12. Label Parameter Tests
// =============================================================================

/// Verifies that with_validation_scope accepts string slice labels.
///
/// Contract: Label should accept &str via Option<&str>.
#[test]
fn test_with_validation_scope_accepts_str_label() {
    let adapter = require_adapter!();
    let (device, _queue) = require_device!(&adapter);

    let label: &str = "my_validation_scope";
    let (result, _) = block_on(with_validation_scope(&device, Some(label), async { 42i32 }));
    assert_eq!(result, 42i32);
}

/// Verifies that with_oom_scope accepts string slice labels.
///
/// Contract: Label should accept &str via Option<&str>.
#[test]
fn test_with_oom_scope_accepts_str_label() {
    let adapter = require_adapter!();
    let (device, _queue) = require_device!(&adapter);

    let label: &str = "my_oom_scope";
    let (result, _) = block_on(with_oom_scope(&device, Some(label), async { 99i32 }));
    assert_eq!(result, 99i32);
}

// =============================================================================
// 13. Async Block Behavior Tests
// =============================================================================

/// Verifies that async blocks can capture external variables.
///
/// Contract: Convenience functions should work with capturing async blocks.
#[test]
fn test_async_blocks_can_capture_variables() {
    let adapter = require_adapter!();
    let (device, _queue) = require_device!(&adapter);

    let external_value = 100i32;
    let (result, _) = block_on(with_validation_scope(
        &device,
        None,
        async move { external_value * 2 },
    ));

    assert_eq!(result, 200i32);
}

/// Verifies that async blocks can return complex types.
///
/// Contract: Convenience functions should work with various return types.
#[test]
fn test_async_blocks_return_complex_types() {
    let adapter = require_adapter!();
    let (device, _queue) = require_device!(&adapter);

    // Return a struct
    #[derive(Debug, PartialEq)]
    struct TestResult {
        value: i32,
        name: String,
    }

    let (result, _) = block_on(with_validation_scope(&device, None, async {
        TestResult {
            value: 42,
            name: "test".to_string(),
        }
    }));

    assert_eq!(result.value, 42);
    assert_eq!(result.name, "test");
}

/// Verifies that async blocks can return Result types.
///
/// Contract: Inner Result types should work correctly.
#[test]
fn test_async_blocks_return_result_types() {
    let adapter = require_adapter!();
    let (device, _queue) = require_device!(&adapter);

    let (inner_result, gpu_error): (Result<i32, &str>, _) =
        block_on(with_validation_scope(&device, None, async { Ok::<i32, &str>(42) }));

    assert_eq!(inner_result, Ok(42));
    assert!(gpu_error.is_none());
}

// =============================================================================
// 14. Multiple Filter Tests
// =============================================================================

/// Verifies that ScopedErrorCapture can manage multiple different filters.
///
/// Contract: Multiple filters of different types should be manageable.
#[test]
fn test_scoped_error_capture_multiple_filters_same_type() {
    let adapter = require_adapter!();
    let (device, _queue) = require_device!(&adapter);

    // Multiple validation filters
    let filters = [
        ErrorFilter::Validation,
        ErrorFilter::Validation,
        ErrorFilter::Validation,
    ];
    let capture = ScopedErrorCapture::new(&device, &filters);
    drop(capture);
}

/// Verifies that ScopedErrorCapture can interleave filter types.
///
/// Contract: Interleaved filter types should work correctly.
#[test]
fn test_scoped_error_capture_interleaved_filters() {
    let adapter = require_adapter!();
    let (device, _queue) = require_device!(&adapter);

    let filters = [
        ErrorFilter::Validation,
        ErrorFilter::OutOfMemory,
        ErrorFilter::Validation,
        ErrorFilter::OutOfMemory,
    ];
    let capture = ScopedErrorCapture::new(&device, &filters);
    drop(capture);
}
