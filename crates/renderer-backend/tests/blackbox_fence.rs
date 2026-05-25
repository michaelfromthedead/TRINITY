// Blackbox contract tests for T-FG-5.5 CommandBufferFence.
//
// CLEANROOM: No src/ access beyond the public API exported by the crate.
// Tests use only `renderer_backend::frame_graph::*` -- no internal fields,
// no private methods, no implementation details.
//
// Contract:
//   CommandBufferFence::new(wait_value, signal_value)
//   CommandBufferFence::is_complete() -> bool
//   CommandBufferFence::mark_signaled()
//
// Acceptance criteria:
//   - new() stores both wait/signal values and starts incomplete
//   - is_complete() returns false before mark_signaled()
//   - is_complete() returns true after mark_signaled()
//   - mark_signaled() is idempotent (safe to call multiple times)
//   - Independent fences are fully isolated
//   - Debug output includes all fields
//   - Fence implements Send + Sync for thread-safe backend use
//
// Coverage:
//   1.  Construction -- values stored, initial state incomplete
//   2.  Wait / signal value accessors (field reads)
//   3.  Initial incomplete -- fresh fence is not complete
//   4.  Mark signaled transitions to complete
//   5.  Mark signaled is idempotent
//   6.  Multiple fences are independent
//   7.  Debug format shows all fields
//   8.  Wait / signal at boundary values (zero, max)
//   9.  Repeated new/create for large batches
//  10.  Send + Sync trait bounds

use renderer_backend::frame_graph::CommandBufferFence;

// ===== SECTION 1 -- Construction =====

#[test]
fn fence_constructs_with_wait_and_signal_values() {
    let fence = CommandBufferFence::new(1, 42);
    assert_eq!(fence.wait_value, 1);
    assert_eq!(fence.signal_value, 42);
}

#[test]
fn fence_new_initial_state_is_not_signaled() {
    let fence = CommandBufferFence::new(0, 0);
    assert!(!fence.is_complete(), "fresh fence must report incomplete");
}

#[test]
fn fence_constructs_with_zero_values() {
    let fence = CommandBufferFence::new(0, 0);
    assert_eq!(fence.wait_value, 0);
    assert_eq!(fence.signal_value, 0);
    assert!(!fence.is_complete());
}

#[test]
fn fence_constructs_with_max_values() {
    let fence = CommandBufferFence::new(u64::MAX, u64::MAX);
    assert_eq!(fence.wait_value, u64::MAX);
    assert_eq!(fence.signal_value, u64::MAX);
}

#[test]
fn fence_different_wait_signal_values() {
    let fence = CommandBufferFence::new(7, 13);
    assert_eq!(fence.wait_value, 7);
    assert_eq!(fence.signal_value, 13);
    assert_ne!(fence.wait_value, fence.signal_value);
}

// ===== SECTION 2 -- Wait / signal value accessors =====

#[test]
fn fence_wait_value_accessible_as_public_field() {
    let fence = CommandBufferFence::new(99, 200);
    assert_eq!(fence.wait_value, 99);
}

#[test]
fn fence_signal_value_accessible_as_public_field() {
    let fence = CommandBufferFence::new(100, 256);
    assert_eq!(fence.signal_value, 256);
}

// ===== SECTION 3 -- Initial incomplete state =====

#[test]
fn fence_is_complete_returns_false_before_mark_signaled() {
    let fence = CommandBufferFence::new(1, 2);
    assert!(!fence.is_complete());
}

#[test]
fn fence_is_complete_returns_false_with_zero_values() {
    let fence = CommandBufferFence::new(0, 0);
    assert!(!fence.is_complete());
}

#[test]
fn fence_returns_false_across_repeated_checks() {
    let fence = CommandBufferFence::new(5, 10);
    for _ in 0..10 {
        assert!(!fence.is_complete(), "must remain incomplete until mark_signaled");
    }
}

// ===== SECTION 4 -- Mark signaled transitions to complete =====

#[test]
fn fence_mark_signaled_makes_is_complete_true() {
    let fence = CommandBufferFence::new(1, 1);
    fence.mark_signaled();
    assert!(fence.is_complete());
}

#[test]
fn fence_mark_signaled_preserves_wait_and_signal_values() {
    let fence = CommandBufferFence::new(3, 7);
    fence.mark_signaled();
    assert!(fence.is_complete());
    assert_eq!(fence.wait_value, 3);
    assert_eq!(fence.signal_value, 7);
}

#[test]
fn fence_mark_signaled_then_is_complete_returns_true() {
    let fence = CommandBufferFence::new(0, 1);
    fence.mark_signaled();
    let result = fence.is_complete();
    assert!(result);
}

// ===== SECTION 5 -- Mark signaled is idempotent =====

#[test]
fn fence_mark_signaled_called_twice_stays_complete() {
    let fence = CommandBufferFence::new(2, 4);
    fence.mark_signaled();
    fence.mark_signaled();
    assert!(fence.is_complete());
}

#[test]
fn fence_mark_signaled_called_many_times_stays_complete() {
    let fence = CommandBufferFence::new(1, 1);
    for _ in 0..100 {
        fence.mark_signaled();
    }
    assert!(fence.is_complete());
}

#[test]
fn fence_is_complete_returns_true_after_single_mark_signaled() {
    let fence = CommandBufferFence::new(10, 20);
    fence.mark_signaled();
    for _ in 0..20 {
        assert!(fence.is_complete(), "must stay complete after repeated checks");
    }
}

// ===== SECTION 6 -- Multiple fences are independent =====

#[test]
fn two_fences_independent_states() {
    let f1 = CommandBufferFence::new(1, 10);
    let f2 = CommandBufferFence::new(2, 20);

    assert!(!f1.is_complete());
    assert!(!f2.is_complete());

    f1.mark_signaled();

    assert!(f1.is_complete(), "f1 must be complete after mark");
    assert!(!f2.is_complete(), "f2 must remain incomplete");
}

#[test]
fn multiple_fences_independent_signaling() {
    let fences: Vec<CommandBufferFence> = (0..10)
        .map(|i| CommandBufferFence::new(i, i + 100))
        .collect();

    // Signal every other fence.
    for (idx, fence) in fences.iter().enumerate() {
        if idx % 2 == 0 {
            fence.mark_signaled();
        }
    }

    // Verify independence.
    for (idx, fence) in fences.iter().enumerate() {
        if idx % 2 == 0 {
            assert!(fence.is_complete(), "even-indexed fence {} must be complete", idx);
        } else {
            assert!(!fence.is_complete(), "odd-indexed fence {} must remain incomplete", idx);
        }
    }
}

#[test]
fn fence_batch_construction_and_signaling() {
    let count = 100;
    let fences: Vec<CommandBufferFence> = (0..count)
        .map(|i| CommandBufferFence::new(i as u64, (i * 2) as u64))
        .collect();

    // All start incomplete.
    for (i, f) in fences.iter().enumerate() {
        assert!(!f.is_complete(), "fence {} must start incomplete", i);
    }

    // Mark all signaled.
    for f in &fences {
        f.mark_signaled();
    }

    // All now complete.
    for (i, f) in fences.iter().enumerate() {
        assert!(f.is_complete(), "fence {} must be complete after batch mark", i);
    }
}

// ===== SECTION 7 -- Debug format =====

#[test]
fn fence_debug_format_contains_fields() {
    let fence = CommandBufferFence::new(7, 13);
    let dbg = format!("{:?}", fence);
    assert!(dbg.contains("CommandBufferFence"), "Debug must contain type name");
    assert!(
        dbg.contains("wait_value") || dbg.contains("wait"),
        "Debug must include wait field information"
    );
    assert!(
        dbg.contains("signal_value") || dbg.contains("signal"),
        "Debug must include signal field information"
    );
}

#[test]
fn fence_debug_shows_no_initial_signaled_state() {
    let fence = CommandBufferFence::new(0, 0);
    let dbg = format!("{:?}", fence);
    assert!(dbg.contains("false") || dbg.contains("is_signaled"), "Debug should indicate false");
}

// ===== SECTION 8 -- Boundary values =====

#[test]
fn fence_handles_u64_zero() {
    let fence = CommandBufferFence::new(0, 0);
    assert_eq!(fence.wait_value, 0);
    assert_eq!(fence.signal_value, 0);
    fence.mark_signaled();
    assert!(fence.is_complete());
}

#[test]
fn fence_handles_u64_max() {
    let fence = CommandBufferFence::new(u64::MAX, u64::MAX);
    assert_eq!(fence.wait_value, u64::MAX);
    assert_eq!(fence.signal_value, u64::MAX);
    fence.mark_signaled();
    assert!(fence.is_complete());
}

#[test]
fn fence_handles_mixed_boundary_values() {
    let fence = CommandBufferFence::new(0, u64::MAX);
    assert_eq!(fence.wait_value, 0);
    assert_eq!(fence.signal_value, u64::MAX);
    fence.mark_signaled();
    assert!(fence.is_complete());
}

// ===== SECTION 9 -- Large batch =====

#[test]
fn fence_large_batch_all_independent() {
    let batch_size = 1000;
    let mut fences: Vec<CommandBufferFence> = Vec::with_capacity(batch_size);
    for i in 0..batch_size {
        fences.push(CommandBufferFence::new(i as u64, (batch_size - i) as u64));
    }

    // Verify values.
    for (i, f) in fences.iter().enumerate() {
        assert_eq!(f.wait_value, i as u64);
        assert_eq!(f.signal_value, (batch_size - i) as u64);
    }

    // All start incomplete.
    for f in &fences {
        assert!(!f.is_complete());
    }

    // Signal a few at different positions.
    fences[0].mark_signaled();
    fences[500].mark_signaled();
    fences[999].mark_signaled();

    assert!(fences[0].is_complete());
    assert!(fences[500].is_complete());
    assert!(fences[999].is_complete());

    // Others remain incomplete.
    for i in [1, 2, 498, 499, 501, 502, 997, 998] {
        assert!(!fences[i].is_complete(), "fence {} must remain incomplete", i);
    }
}

// ===== SECTION 10 -- Thread safety: Send + Sync bounds =====

/// Verify that CommandBufferFence is Send (ownership can be transferred
/// between threads).
#[test]
fn fence_is_send() {
    fn assert_send<T: Send>() {}
    assert_send::<CommandBufferFence>();
}

/// Verify that CommandBufferFence is Sync (shared references can be used
/// across threads).
#[test]
fn fence_is_sync() {
    fn assert_sync<T: Sync>() {}
    assert_sync::<CommandBufferFence>();
}

/// Simulate the backend/real-world pattern: create fence, hand it off
/// to a worker thread, worker signals completion, main thread polls.
#[test]
fn fence_cross_thread_signaling() {
    use std::sync::Arc;
    use std::thread;

    let fence = Arc::new(CommandBufferFence::new(1, 1));
    let fence_clone = Arc::clone(&fence);

    let handle = thread::spawn(move || {
        // Simulate GPU work completing.
        fence_clone.mark_signaled();
    });

    handle.join().expect("worker thread panicked");

    assert!(
        fence.is_complete(),
        "fence must be complete after worker signals from another thread"
    );
}

/// Multiple threads each manage their own fence independently.
#[test]
fn fence_multiple_threads_independent_fences() {
    use std::thread;

    let mut handles = Vec::new();
    let fence_count = 16;

    for i in 0..fence_count {
        let handle = thread::spawn(move || {
            let fence = CommandBufferFence::new(i as u64, (i * 10) as u64);
            assert!(!fence.is_complete(), "thread {}: fence starts incomplete", i);
            fence.mark_signaled();
            assert!(fence.is_complete(), "thread {}: fence complete after mark", i);
            (fence.wait_value, fence.signal_value)
        });
        handles.push(handle);
    }

    for (i, handle) in handles.into_iter().enumerate() {
        let (wait, signal) = handle.join().expect("thread panicked");
        assert_eq!(wait, i as u64);
        assert_eq!(signal, (i * 10) as u64);
    }
}
