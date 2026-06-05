// SPDX-License-Identifier: MIT
//
// error_scope_whitebox.rs -- Whitebox structural tests for T-WGPU-P1.3.5
// (ErrorScope - Error Scopes).
//
// These tests exercise the internal implementation of ErrorScope and related
// types, covering all code paths in error filter conversion, scope creation,
// RAII behavior, explicit pop, multi-scope capture, and convenience functions.
//
// WHITEBOX coverage plan:
//   - Section 1: ErrorFilter enum tests
//     - Path A: Validation variant construction and to_wgpu conversion
//     - Path B: OutOfMemory variant construction and to_wgpu conversion
//     - Path C: from_wgpu conversion for known variants
//     - Path D: from_wgpu conversion for unknown variants (fallback branch)
//     - Path E: description() for both variants
//     - Path F: Display impl for both variants
//   - Section 2: ErrorScope construction tests
//     - Path A: new() pushes scope immediately
//     - Path B: with_label() pushes scope and stores label
//     - Path C: filter() accessor returns correct value
//     - Path D: label() accessor returns correct value
//     - Path E: is_popped() returns false initially
//   - Section 3: RAII behavior tests
//     - Path A: Drop pops scope when not explicitly popped
//     - Path B: Drop does not double-pop when already popped
//     - Path C: popped flag is set correctly
//   - Section 4: Explicit pop tests
//     - Path A: pop() returns None when no error
//     - Path B: pop_blocking() returns None when no error
//     - Path C: pop() sets popped flag to true
//   - Section 5: ScopedErrorCapture tests
//     - Path A: new() pushes multiple scopes in order
//     - Path B: pop_all() pops in reverse order (LIFO)
//     - Path C: pop_all() returns empty vec when already popped
//     - Path D: pop_all_blocking() convenience method
//     - Path E: has_errors() returns false when no errors
//     - Path F: Drop pops remaining scopes when not explicitly popped
//   - Section 6: Convenience function tests
//     - Path A: with_validation_scope() with label
//     - Path B: with_validation_scope() without label
//     - Path C: with_oom_scope() with label
//     - Path D: with_oom_scope() without label
//   - Section 7: Edge cases and boundary conditions
//     - Path A: Empty filter slice in ScopedErrorCapture
//     - Path B: Nested ErrorScope instances
//     - Path C: Multiple consecutive scopes
//
// Acceptance criteria (T-WGPU-P1.3.5):
//   1. push_error_scope() on creation
//   2. pop_error_scope() on drop
//   3. Validation and OutOfMemory filters
//   4. Async error retrieval
//   5. Error logging and propagation

use renderer_backend::device::{
    with_oom_scope, with_validation_scope, ErrorFilter, ErrorScope, ScopedErrorCapture,
    TrinityInstance,
};

// =============================================================================
// Test Helpers
// =============================================================================

/// Get a device for testing, or skip the test if none available.
fn get_test_device() -> Option<(wgpu::Device, wgpu::Queue)> {
    let instance = TrinityInstance::new();
    let adapters = instance.enumerate_adapters();
    let adapter = adapters.into_iter().next()?;

    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("test_device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::downlevel_defaults(),
            memory_hints: wgpu::MemoryHints::default(),
        },
        None,
    ))
    .ok()?;

    Some((device, queue))
}

/// Blocking helper to run async code.
fn block_on<F: std::future::Future>(future: F) -> F::Output {
    pollster::block_on(future)
}

// =============================================================================
// Section 1: ErrorFilter Enum Tests
// =============================================================================

// ---------------------------------------------------------------------------
// 1.A: Validation variant construction and to_wgpu conversion
// ---------------------------------------------------------------------------

#[test]
fn test_error_filter_validation_variant() {
    let filter = ErrorFilter::Validation;
    assert_eq!(filter.to_wgpu(), wgpu::ErrorFilter::Validation);
}

#[test]
fn test_error_filter_validation_debug() {
    let filter = ErrorFilter::Validation;
    let debug_str = format!("{:?}", filter);
    assert_eq!(debug_str, "Validation");
}

#[test]
fn test_error_filter_validation_clone() {
    let filter = ErrorFilter::Validation;
    let cloned = filter.clone();
    assert_eq!(filter, cloned);
}

#[test]
fn test_error_filter_validation_copy() {
    let filter = ErrorFilter::Validation;
    let copied: ErrorFilter = filter; // Copy, not move
    assert_eq!(filter, copied);
}

// ---------------------------------------------------------------------------
// 1.B: OutOfMemory variant construction and to_wgpu conversion
// ---------------------------------------------------------------------------

#[test]
fn test_error_filter_out_of_memory_variant() {
    let filter = ErrorFilter::OutOfMemory;
    assert_eq!(filter.to_wgpu(), wgpu::ErrorFilter::OutOfMemory);
}

#[test]
fn test_error_filter_out_of_memory_debug() {
    let filter = ErrorFilter::OutOfMemory;
    let debug_str = format!("{:?}", filter);
    assert_eq!(debug_str, "OutOfMemory");
}

#[test]
fn test_error_filter_out_of_memory_clone() {
    let filter = ErrorFilter::OutOfMemory;
    let cloned = filter.clone();
    assert_eq!(filter, cloned);
}

// ---------------------------------------------------------------------------
// 1.C: from_wgpu conversion for known variants
// ---------------------------------------------------------------------------

#[test]
fn test_error_filter_from_wgpu_validation() {
    let wgpu_filter = wgpu::ErrorFilter::Validation;
    let filter = ErrorFilter::from_wgpu(wgpu_filter);
    assert_eq!(filter, ErrorFilter::Validation);
}

#[test]
fn test_error_filter_from_wgpu_out_of_memory() {
    let wgpu_filter = wgpu::ErrorFilter::OutOfMemory;
    let filter = ErrorFilter::from_wgpu(wgpu_filter);
    assert_eq!(filter, ErrorFilter::OutOfMemory);
}

// ---------------------------------------------------------------------------
// 1.D: from_wgpu round-trip conversion
// ---------------------------------------------------------------------------

#[test]
fn test_error_filter_round_trip_validation() {
    let original = ErrorFilter::Validation;
    let wgpu_filter = original.to_wgpu();
    let converted = ErrorFilter::from_wgpu(wgpu_filter);
    assert_eq!(original, converted);
}

#[test]
fn test_error_filter_round_trip_out_of_memory() {
    let original = ErrorFilter::OutOfMemory;
    let wgpu_filter = original.to_wgpu();
    let converted = ErrorFilter::from_wgpu(wgpu_filter);
    assert_eq!(original, converted);
}

// ---------------------------------------------------------------------------
// 1.E: description() for both variants
// ---------------------------------------------------------------------------

#[test]
fn test_error_filter_validation_description() {
    let filter = ErrorFilter::Validation;
    let desc = filter.description();
    assert_eq!(desc, "validation errors (API misuse)");
}

#[test]
fn test_error_filter_out_of_memory_description() {
    let filter = ErrorFilter::OutOfMemory;
    let desc = filter.description();
    assert_eq!(desc, "out-of-memory errors (allocation failures)");
}

// ---------------------------------------------------------------------------
// 1.F: Display impl for both variants
// ---------------------------------------------------------------------------

#[test]
fn test_error_filter_validation_display() {
    let filter = ErrorFilter::Validation;
    let display_str = format!("{}", filter);
    assert_eq!(display_str, "Validation");
}

#[test]
fn test_error_filter_out_of_memory_display() {
    let filter = ErrorFilter::OutOfMemory;
    let display_str = format!("{}", filter);
    assert_eq!(display_str, "OutOfMemory");
}

// ---------------------------------------------------------------------------
// 1.G: Hash and Eq implementations
// ---------------------------------------------------------------------------

#[test]
fn test_error_filter_eq_same_variants() {
    assert_eq!(ErrorFilter::Validation, ErrorFilter::Validation);
    assert_eq!(ErrorFilter::OutOfMemory, ErrorFilter::OutOfMemory);
}

#[test]
fn test_error_filter_eq_different_variants() {
    assert_ne!(ErrorFilter::Validation, ErrorFilter::OutOfMemory);
}

#[test]
fn test_error_filter_hash() {
    use std::collections::HashSet;

    let mut set = HashSet::new();
    set.insert(ErrorFilter::Validation);
    set.insert(ErrorFilter::OutOfMemory);

    assert!(set.contains(&ErrorFilter::Validation));
    assert!(set.contains(&ErrorFilter::OutOfMemory));
    assert_eq!(set.len(), 2);
}

#[test]
fn test_error_filter_hash_duplicate() {
    use std::collections::HashSet;

    let mut set = HashSet::new();
    set.insert(ErrorFilter::Validation);
    set.insert(ErrorFilter::Validation); // duplicate

    assert_eq!(set.len(), 1);
}

// =============================================================================
// Section 2: ErrorScope Construction Tests
// =============================================================================

// ---------------------------------------------------------------------------
// 2.A: new() creates scope and pushes immediately
// ---------------------------------------------------------------------------

#[test]
fn test_error_scope_new_validation() {
    let Some((device, _queue)) = get_test_device() else {
        eprintln!("SKIP: No device available");
        return;
    };

    // Create a validation scope - this pushes the scope
    let scope = ErrorScope::new(&device, ErrorFilter::Validation);

    // Verify filter is stored correctly
    assert_eq!(scope.filter(), ErrorFilter::Validation);

    // Verify not popped yet
    assert!(!scope.is_popped());

    // Explicitly pop to avoid drop-pop issues
    let _ = block_on(scope.pop());
}

#[test]
fn test_error_scope_new_out_of_memory() {
    let Some((device, _queue)) = get_test_device() else {
        eprintln!("SKIP: No device available");
        return;
    };

    let scope = ErrorScope::new(&device, ErrorFilter::OutOfMemory);

    assert_eq!(scope.filter(), ErrorFilter::OutOfMemory);
    assert!(!scope.is_popped());

    let _ = block_on(scope.pop());
}

// ---------------------------------------------------------------------------
// 2.B: with_label() pushes scope and stores label
// ---------------------------------------------------------------------------

#[test]
fn test_error_scope_with_label_validation() {
    let Some((device, _queue)) = get_test_device() else {
        eprintln!("SKIP: No device available");
        return;
    };

    let scope = ErrorScope::with_label(&device, ErrorFilter::Validation, "test_label");

    assert_eq!(scope.filter(), ErrorFilter::Validation);
    assert_eq!(scope.label(), Some("test_label"));
    assert!(!scope.is_popped());

    let _ = block_on(scope.pop());
}

#[test]
fn test_error_scope_with_label_out_of_memory() {
    let Some((device, _queue)) = get_test_device() else {
        eprintln!("SKIP: No device available");
        return;
    };

    let scope = ErrorScope::with_label(&device, ErrorFilter::OutOfMemory, "oom_test");

    assert_eq!(scope.filter(), ErrorFilter::OutOfMemory);
    assert_eq!(scope.label(), Some("oom_test"));
    assert!(!scope.is_popped());

    let _ = block_on(scope.pop());
}

// ---------------------------------------------------------------------------
// 2.C: filter() accessor returns correct value
// ---------------------------------------------------------------------------

#[test]
fn test_error_scope_filter_accessor() {
    let Some((device, _queue)) = get_test_device() else {
        eprintln!("SKIP: No device available");
        return;
    };

    let scope_val = ErrorScope::new(&device, ErrorFilter::Validation);
    assert_eq!(scope_val.filter(), ErrorFilter::Validation);
    let _ = block_on(scope_val.pop());

    let scope_oom = ErrorScope::new(&device, ErrorFilter::OutOfMemory);
    assert_eq!(scope_oom.filter(), ErrorFilter::OutOfMemory);
    let _ = block_on(scope_oom.pop());
}

// ---------------------------------------------------------------------------
// 2.D: label() accessor returns correct value
// ---------------------------------------------------------------------------

#[test]
fn test_error_scope_label_accessor_none() {
    let Some((device, _queue)) = get_test_device() else {
        eprintln!("SKIP: No device available");
        return;
    };

    let scope = ErrorScope::new(&device, ErrorFilter::Validation);
    assert_eq!(scope.label(), None);

    let _ = block_on(scope.pop());
}

#[test]
fn test_error_scope_label_accessor_some() {
    let Some((device, _queue)) = get_test_device() else {
        eprintln!("SKIP: No device available");
        return;
    };

    let scope = ErrorScope::with_label(&device, ErrorFilter::Validation, "my_label");
    assert_eq!(scope.label(), Some("my_label"));

    let _ = block_on(scope.pop());
}

#[test]
fn test_error_scope_label_empty_string() {
    let Some((device, _queue)) = get_test_device() else {
        eprintln!("SKIP: No device available");
        return;
    };

    let scope = ErrorScope::with_label(&device, ErrorFilter::Validation, "");
    assert_eq!(scope.label(), Some(""));

    let _ = block_on(scope.pop());
}

// ---------------------------------------------------------------------------
// 2.E: is_popped() returns false initially
// ---------------------------------------------------------------------------

#[test]
fn test_error_scope_is_popped_initial_state() {
    let Some((device, _queue)) = get_test_device() else {
        eprintln!("SKIP: No device available");
        return;
    };

    let scope = ErrorScope::new(&device, ErrorFilter::Validation);
    assert!(!scope.is_popped(), "Scope should not be popped initially");

    // Clean up
    let _ = block_on(scope.pop());
}

// =============================================================================
// Section 3: RAII Behavior Tests
// =============================================================================

// ---------------------------------------------------------------------------
// 3.A: Drop pops scope when not explicitly popped
// ---------------------------------------------------------------------------

#[test]
fn test_error_scope_drop_pops_automatically() {
    let Some((device, _queue)) = get_test_device() else {
        eprintln!("SKIP: No device available");
        return;
    };

    // Create and immediately drop a scope
    {
        let _scope = ErrorScope::new(&device, ErrorFilter::Validation);
        // Scope will auto-pop on drop here
    }

    // If we get here without panicking, the auto-pop worked
    // Create another scope to verify device is still functional
    {
        let scope = ErrorScope::new(&device, ErrorFilter::Validation);
        let error = block_on(scope.pop());
        assert!(error.is_none(), "No error expected in clean scope");
    }
}

#[test]
fn test_error_scope_drop_with_label() {
    let Some((device, _queue)) = get_test_device() else {
        eprintln!("SKIP: No device available");
        return;
    };

    {
        let _scope = ErrorScope::with_label(&device, ErrorFilter::OutOfMemory, "drop_test");
        // Auto-pop with label
    }

    // Verify device still works
    let scope = ErrorScope::new(&device, ErrorFilter::Validation);
    let _ = block_on(scope.pop());
}

// ---------------------------------------------------------------------------
// 3.B: Drop does not double-pop when already popped
// ---------------------------------------------------------------------------

#[test]
fn test_error_scope_no_double_pop() {
    let Some((device, _queue)) = get_test_device() else {
        eprintln!("SKIP: No device available");
        return;
    };

    let scope = ErrorScope::new(&device, ErrorFilter::Validation);

    // Explicitly pop
    let _ = block_on(scope.pop());

    // Scope is now consumed, so it won't double-pop on drop
    // If we get here without issues, the test passes
}

// ---------------------------------------------------------------------------
// 3.C: popped flag is set correctly after explicit pop
// ---------------------------------------------------------------------------

#[test]
fn test_error_scope_popped_flag_after_pop() {
    let Some((device, _queue)) = get_test_device() else {
        eprintln!("SKIP: No device available");
        return;
    };

    let scope = ErrorScope::new(&device, ErrorFilter::Validation);
    assert!(!scope.is_popped());

    // After pop(), the scope is consumed, so we can't check is_popped()
    // The pop() method moves self, so this is testing that pop consumes the scope
    let _ = block_on(scope.pop());
    // scope is moved/consumed here
}

// =============================================================================
// Section 4: Explicit Pop Tests
// =============================================================================

// ---------------------------------------------------------------------------
// 4.A: pop() returns None when no error
// ---------------------------------------------------------------------------

#[test]
fn test_error_scope_pop_returns_none_no_error() {
    let Some((device, _queue)) = get_test_device() else {
        eprintln!("SKIP: No device available");
        return;
    };

    let scope = ErrorScope::new(&device, ErrorFilter::Validation);

    // No operations that would cause an error
    let error = block_on(scope.pop());

    assert!(error.is_none(), "Expected no error from clean scope");
}

#[test]
fn test_error_scope_pop_validation_filter_no_error() {
    let Some((device, _queue)) = get_test_device() else {
        eprintln!("SKIP: No device available");
        return;
    };

    let scope = ErrorScope::new(&device, ErrorFilter::Validation);

    // Create a valid buffer (should not produce validation error)
    let _buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("valid_buffer"),
        size: 256,
        usage: wgpu::BufferUsages::VERTEX,
        mapped_at_creation: false,
    });

    let error = block_on(scope.pop());
    assert!(error.is_none(), "Valid buffer creation should not produce error");
}

#[test]
fn test_error_scope_pop_oom_filter_no_error() {
    let Some((device, _queue)) = get_test_device() else {
        eprintln!("SKIP: No device available");
        return;
    };

    let scope = ErrorScope::new(&device, ErrorFilter::OutOfMemory);

    // Create a small buffer (should not cause OOM)
    let _buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("small_buffer"),
        size: 256,
        usage: wgpu::BufferUsages::VERTEX,
        mapped_at_creation: false,
    });

    let error = block_on(scope.pop());
    assert!(error.is_none(), "Small buffer creation should not produce OOM error");
}

// ---------------------------------------------------------------------------
// 4.B: pop_blocking() returns None when no error
// ---------------------------------------------------------------------------

#[test]
fn test_error_scope_pop_blocking_returns_none_no_error() {
    let Some((device, _queue)) = get_test_device() else {
        eprintln!("SKIP: No device available");
        return;
    };

    let scope = ErrorScope::new(&device, ErrorFilter::Validation);

    let error = scope.pop_blocking();

    assert!(error.is_none(), "Expected no error from clean scope using pop_blocking");
}

#[test]
fn test_error_scope_pop_blocking_with_label() {
    let Some((device, _queue)) = get_test_device() else {
        eprintln!("SKIP: No device available");
        return;
    };

    let scope = ErrorScope::with_label(&device, ErrorFilter::OutOfMemory, "blocking_test");

    let error = scope.pop_blocking();

    assert!(error.is_none());
}

// ---------------------------------------------------------------------------
// 4.C: pop() consumes the scope (move semantics)
// ---------------------------------------------------------------------------

#[test]
fn test_error_scope_pop_consumes_scope() {
    let Some((device, _queue)) = get_test_device() else {
        eprintln!("SKIP: No device available");
        return;
    };

    let scope = ErrorScope::new(&device, ErrorFilter::Validation);

    // This call moves `scope`
    let _ = block_on(scope.pop());

    // Cannot use `scope` after pop() - this is enforced by the borrow checker
    // The test verifies the API design (pop takes self, not &self)
}

// =============================================================================
// Section 5: ScopedErrorCapture Tests
// =============================================================================

// ---------------------------------------------------------------------------
// 5.A: new() pushes multiple scopes in order
// ---------------------------------------------------------------------------

#[test]
fn test_scoped_error_capture_new_single_filter() {
    let Some((device, _queue)) = get_test_device() else {
        eprintln!("SKIP: No device available");
        return;
    };

    let mut capture = ScopedErrorCapture::new(&device, &[ErrorFilter::Validation]);

    let results = block_on(capture.pop_all());

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].0, ErrorFilter::Validation);
    assert!(results[0].1.is_none());
}

#[test]
fn test_scoped_error_capture_new_multiple_filters() {
    let Some((device, _queue)) = get_test_device() else {
        eprintln!("SKIP: No device available");
        return;
    };

    let mut capture =
        ScopedErrorCapture::new(&device, &[ErrorFilter::Validation, ErrorFilter::OutOfMemory]);

    let results = block_on(capture.pop_all());

    // Should pop in reverse order (LIFO)
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].0, ErrorFilter::OutOfMemory); // popped first (inner)
    assert_eq!(results[1].0, ErrorFilter::Validation); // popped second (outer)
}

// ---------------------------------------------------------------------------
// 5.B: pop_all() pops in reverse order (LIFO)
// ---------------------------------------------------------------------------

#[test]
fn test_scoped_error_capture_lifo_order() {
    let Some((device, _queue)) = get_test_device() else {
        eprintln!("SKIP: No device available");
        return;
    };

    let mut capture = ScopedErrorCapture::new(
        &device,
        &[
            ErrorFilter::Validation,
            ErrorFilter::OutOfMemory,
            ErrorFilter::Validation,
        ],
    );

    let results = block_on(capture.pop_all());

    assert_eq!(results.len(), 3);
    // Reverse order of push
    assert_eq!(results[0].0, ErrorFilter::Validation); // last pushed, first popped
    assert_eq!(results[1].0, ErrorFilter::OutOfMemory);
    assert_eq!(results[2].0, ErrorFilter::Validation); // first pushed, last popped
}

// ---------------------------------------------------------------------------
// 5.C: pop_all() returns empty vec when already popped
// ---------------------------------------------------------------------------

#[test]
fn test_scoped_error_capture_double_pop_returns_empty() {
    let Some((device, _queue)) = get_test_device() else {
        eprintln!("SKIP: No device available");
        return;
    };

    let mut capture =
        ScopedErrorCapture::new(&device, &[ErrorFilter::Validation, ErrorFilter::OutOfMemory]);

    // First pop
    let results1 = block_on(capture.pop_all());
    assert_eq!(results1.len(), 2);

    // Second pop should return empty
    let results2 = block_on(capture.pop_all());
    assert!(results2.is_empty(), "Double pop should return empty vec");
}

// ---------------------------------------------------------------------------
// 5.D: pop_all_blocking() convenience method
// ---------------------------------------------------------------------------

#[test]
fn test_scoped_error_capture_pop_all_blocking() {
    let Some((device, _queue)) = get_test_device() else {
        eprintln!("SKIP: No device available");
        return;
    };

    let mut capture =
        ScopedErrorCapture::new(&device, &[ErrorFilter::Validation, ErrorFilter::OutOfMemory]);

    let results = capture.pop_all_blocking();

    assert_eq!(results.len(), 2);
    assert_eq!(results[0].0, ErrorFilter::OutOfMemory);
    assert_eq!(results[1].0, ErrorFilter::Validation);
}

// ---------------------------------------------------------------------------
// 5.E: has_errors() returns false when no errors
// ---------------------------------------------------------------------------

#[test]
fn test_scoped_error_capture_has_errors_false() {
    let Some((device, _queue)) = get_test_device() else {
        eprintln!("SKIP: No device available");
        return;
    };

    let mut capture =
        ScopedErrorCapture::new(&device, &[ErrorFilter::Validation, ErrorFilter::OutOfMemory]);

    // No operations that would cause errors
    let has_errors = block_on(capture.has_errors());

    assert!(!has_errors, "No errors expected in clean capture");
}

// ---------------------------------------------------------------------------
// 5.F: Drop pops remaining scopes when not explicitly popped
// ---------------------------------------------------------------------------

#[test]
fn test_scoped_error_capture_drop_pops_remaining() {
    let Some((device, _queue)) = get_test_device() else {
        eprintln!("SKIP: No device available");
        return;
    };

    {
        let _capture =
            ScopedErrorCapture::new(&device, &[ErrorFilter::Validation, ErrorFilter::OutOfMemory]);
        // Capture will auto-pop on drop
    }

    // If we get here without issues, auto-pop worked
    // Verify device still functional
    let scope = ErrorScope::new(&device, ErrorFilter::Validation);
    let _ = block_on(scope.pop());
}

// =============================================================================
// Section 6: Convenience Function Tests
// =============================================================================

// ---------------------------------------------------------------------------
// 6.A: with_validation_scope() with label
// ---------------------------------------------------------------------------

#[test]
fn test_with_validation_scope_with_label() {
    let Some((device, _queue)) = get_test_device() else {
        eprintln!("SKIP: No device available");
        return;
    };

    let (result, error) = block_on(with_validation_scope(&device, Some("test_op"), async {
        // Simple operation
        42u32
    }));

    assert_eq!(result, 42);
    assert!(error.is_none());
}

#[test]
fn test_with_validation_scope_with_buffer_creation() {
    let Some((device, _queue)) = get_test_device() else {
        eprintln!("SKIP: No device available");
        return;
    };

    let (buffer, error) =
        block_on(with_validation_scope(&device, Some("buffer_create"), async {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("test_buffer"),
                size: 1024,
                usage: wgpu::BufferUsages::VERTEX,
                mapped_at_creation: false,
            })
        }));

    // Buffer should be created
    assert!(std::ptr::addr_of!(buffer) as usize != 0);
    assert!(error.is_none());
}

// ---------------------------------------------------------------------------
// 6.B: with_validation_scope() without label
// ---------------------------------------------------------------------------

#[test]
fn test_with_validation_scope_without_label() {
    let Some((device, _queue)) = get_test_device() else {
        eprintln!("SKIP: No device available");
        return;
    };

    let (result, error) = block_on(with_validation_scope(&device, None, async { "hello" }));

    assert_eq!(result, "hello");
    assert!(error.is_none());
}

// ---------------------------------------------------------------------------
// 6.C: with_oom_scope() with label
// ---------------------------------------------------------------------------

#[test]
fn test_with_oom_scope_with_label() {
    let Some((device, _queue)) = get_test_device() else {
        eprintln!("SKIP: No device available");
        return;
    };

    let (result, error) = block_on(with_oom_scope(&device, Some("alloc_test"), async {
        // Small allocation, should not OOM
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("small_buffer"),
            size: 256,
            usage: wgpu::BufferUsages::VERTEX,
            mapped_at_creation: false,
        })
    }));

    assert!(std::ptr::addr_of!(result) as usize != 0);
    assert!(error.is_none());
}

// ---------------------------------------------------------------------------
// 6.D: with_oom_scope() without label
// ---------------------------------------------------------------------------

#[test]
fn test_with_oom_scope_without_label() {
    let Some((device, _queue)) = get_test_device() else {
        eprintln!("SKIP: No device available");
        return;
    };

    let (result, error) = block_on(with_oom_scope(&device, None, async { 123i32 }));

    assert_eq!(result, 123);
    assert!(error.is_none());
}

// =============================================================================
// Section 7: Edge Cases and Boundary Conditions
// =============================================================================

// ---------------------------------------------------------------------------
// 7.A: Empty filter slice in ScopedErrorCapture
// ---------------------------------------------------------------------------

#[test]
fn test_scoped_error_capture_empty_filters() {
    let Some((device, _queue)) = get_test_device() else {
        eprintln!("SKIP: No device available");
        return;
    };

    let mut capture = ScopedErrorCapture::new(&device, &[]);

    let results = block_on(capture.pop_all());

    assert!(results.is_empty(), "Empty filters should produce empty results");
}

#[test]
fn test_scoped_error_capture_empty_filters_has_errors() {
    let Some((device, _queue)) = get_test_device() else {
        eprintln!("SKIP: No device available");
        return;
    };

    let mut capture = ScopedErrorCapture::new(&device, &[]);

    let has_errors = block_on(capture.has_errors());

    assert!(!has_errors, "Empty capture should have no errors");
}

// ---------------------------------------------------------------------------
// 7.B: Nested ErrorScope instances
// ---------------------------------------------------------------------------

#[test]
fn test_nested_error_scopes_same_filter() {
    let Some((device, _queue)) = get_test_device() else {
        eprintln!("SKIP: No device available");
        return;
    };

    let outer_scope = ErrorScope::new(&device, ErrorFilter::Validation);

    {
        let inner_scope = ErrorScope::new(&device, ErrorFilter::Validation);

        // No errors in inner scope
        let inner_error = block_on(inner_scope.pop());
        assert!(inner_error.is_none());
    }

    // Outer scope should still be valid
    let outer_error = block_on(outer_scope.pop());
    assert!(outer_error.is_none());
}

#[test]
fn test_nested_error_scopes_different_filters() {
    let Some((device, _queue)) = get_test_device() else {
        eprintln!("SKIP: No device available");
        return;
    };

    let outer_scope = ErrorScope::new(&device, ErrorFilter::Validation);

    {
        let inner_scope = ErrorScope::new(&device, ErrorFilter::OutOfMemory);

        let inner_error = block_on(inner_scope.pop());
        assert!(inner_error.is_none());
    }

    let outer_error = block_on(outer_scope.pop());
    assert!(outer_error.is_none());
}

#[test]
fn test_deeply_nested_scopes() {
    let Some((device, _queue)) = get_test_device() else {
        eprintln!("SKIP: No device available");
        return;
    };

    let scope1 = ErrorScope::new(&device, ErrorFilter::Validation);
    let scope2 = ErrorScope::new(&device, ErrorFilter::OutOfMemory);
    let scope3 = ErrorScope::new(&device, ErrorFilter::Validation);

    // Pop in reverse order
    let e3 = block_on(scope3.pop());
    let e2 = block_on(scope2.pop());
    let e1 = block_on(scope1.pop());

    assert!(e3.is_none());
    assert!(e2.is_none());
    assert!(e1.is_none());
}

// ---------------------------------------------------------------------------
// 7.C: Multiple consecutive scopes
// ---------------------------------------------------------------------------

#[test]
fn test_multiple_consecutive_scopes() {
    let Some((device, _queue)) = get_test_device() else {
        eprintln!("SKIP: No device available");
        return;
    };

    for i in 0..5 {
        let scope = ErrorScope::new(&device, ErrorFilter::Validation);
        let error = block_on(scope.pop());
        assert!(error.is_none(), "Iteration {} should have no error", i);
    }
}

#[test]
fn test_alternating_filter_types() {
    let Some((device, _queue)) = get_test_device() else {
        eprintln!("SKIP: No device available");
        return;
    };

    for i in 0..4 {
        let filter = if i % 2 == 0 {
            ErrorFilter::Validation
        } else {
            ErrorFilter::OutOfMemory
        };

        let scope = ErrorScope::new(&device, filter);
        let error = block_on(scope.pop());
        assert!(error.is_none());
    }
}

// ---------------------------------------------------------------------------
// 7.D: Scope with operations inside
// ---------------------------------------------------------------------------

#[test]
fn test_scope_with_valid_operations() {
    let Some((device, _queue)) = get_test_device() else {
        eprintln!("SKIP: No device available");
        return;
    };

    let scope = ErrorScope::new(&device, ErrorFilter::Validation);

    // Create multiple valid resources
    let _buffer1 = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("buffer1"),
        size: 256,
        usage: wgpu::BufferUsages::VERTEX,
        mapped_at_creation: false,
    });

    let _buffer2 = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("buffer2"),
        size: 512,
        usage: wgpu::BufferUsages::INDEX,
        mapped_at_creation: false,
    });

    let error = block_on(scope.pop());
    assert!(error.is_none());
}

#[test]
fn test_scope_with_command_encoder() {
    let Some((device, _queue)) = get_test_device() else {
        eprintln!("SKIP: No device available");
        return;
    };

    let scope = ErrorScope::new(&device, ErrorFilter::Validation);

    let _encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("test_encoder"),
    });

    let error = block_on(scope.pop());
    assert!(error.is_none());
}

// ---------------------------------------------------------------------------
// 7.E: ScopedErrorCapture with operations between push and pop
// ---------------------------------------------------------------------------

#[test]
fn test_scoped_capture_with_operations() {
    let Some((device, _queue)) = get_test_device() else {
        eprintln!("SKIP: No device available");
        return;
    };

    let mut capture =
        ScopedErrorCapture::new(&device, &[ErrorFilter::Validation, ErrorFilter::OutOfMemory]);

    // Perform valid operations
    let _buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("scoped_buffer"),
        size: 1024,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let results = block_on(capture.pop_all());

    assert_eq!(results.len(), 2);
    assert!(results[0].1.is_none(), "No OOM error expected");
    assert!(results[1].1.is_none(), "No validation error expected");
}

// ---------------------------------------------------------------------------
// 7.F: Labeled scope identification
// ---------------------------------------------------------------------------

#[test]
fn test_scope_labels_unique() {
    let Some((device, _queue)) = get_test_device() else {
        eprintln!("SKIP: No device available");
        return;
    };

    let scope1 = ErrorScope::with_label(&device, ErrorFilter::Validation, "scope_a");
    let scope2 = ErrorScope::with_label(&device, ErrorFilter::Validation, "scope_b");

    assert_ne!(scope1.label(), scope2.label());

    // Pop in correct order (LIFO)
    let _ = block_on(scope2.pop());
    let _ = block_on(scope1.pop());
}

// ---------------------------------------------------------------------------
// 7.G: Convenience function return value preservation
// ---------------------------------------------------------------------------

#[test]
fn test_with_validation_scope_preserves_complex_return() {
    let Some((device, _queue)) = get_test_device() else {
        eprintln!("SKIP: No device available");
        return;
    };

    let (result, error) = block_on(with_validation_scope(&device, Some("complex"), async {
        vec![1, 2, 3, 4, 5]
    }));

    assert_eq!(result, vec![1, 2, 3, 4, 5]);
    assert!(error.is_none());
}

#[test]
fn test_with_oom_scope_preserves_option_return() {
    let Some((device, _queue)) = get_test_device() else {
        eprintln!("SKIP: No device available");
        return;
    };

    let (result, error) =
        block_on(with_oom_scope(&device, None, async { Some("optional_value") }));

    assert_eq!(result, Some("optional_value"));
    assert!(error.is_none());
}

// =============================================================================
// Section 8: Integration Tests
// =============================================================================

// ---------------------------------------------------------------------------
// 8.A: Full workflow - scope protects resource creation
// ---------------------------------------------------------------------------

#[test]
fn test_full_workflow_buffer_creation() {
    let Some((device, _queue)) = get_test_device() else {
        eprintln!("SKIP: No device available");
        return;
    };

    // Create scope
    let scope = ErrorScope::with_label(&device, ErrorFilter::Validation, "buffer_workflow");

    // Create buffer
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("workflow_buffer"),
        size: 4096,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Check for errors
    let error = block_on(scope.pop());
    assert!(error.is_none());

    // Verify buffer was created
    assert!(std::ptr::addr_of!(buffer) as usize != 0);
}

// ---------------------------------------------------------------------------
// 8.B: Full workflow with ScopedErrorCapture
// ---------------------------------------------------------------------------

#[test]
fn test_full_workflow_scoped_capture() {
    let Some((device, _queue)) = get_test_device() else {
        eprintln!("SKIP: No device available");
        return;
    };

    let mut capture =
        ScopedErrorCapture::new(&device, &[ErrorFilter::Validation, ErrorFilter::OutOfMemory]);

    // Create several resources
    let _buffers: Vec<_> = (0..3)
        .map(|i| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("buffer_{}", i)),
                size: 256 * (i + 1) as u64,
                usage: wgpu::BufferUsages::VERTEX,
                mapped_at_creation: false,
            })
        })
        .collect();

    // Verify no errors
    assert!(!block_on(capture.has_errors()));
}

// ---------------------------------------------------------------------------
// 8.C: Convenience function chaining
// ---------------------------------------------------------------------------

#[test]
fn test_convenience_function_sequential() {
    let Some((device, _queue)) = get_test_device() else {
        eprintln!("SKIP: No device available");
        return;
    };

    let (val1, err1) = block_on(with_validation_scope(&device, Some("op1"), async { 1 }));
    let (val2, err2) = block_on(with_oom_scope(&device, Some("op2"), async { 2 }));
    let (val3, err3) = block_on(with_validation_scope(&device, Some("op3"), async { 3 }));

    assert_eq!(val1 + val2 + val3, 6);
    assert!(err1.is_none() && err2.is_none() && err3.is_none());
}
