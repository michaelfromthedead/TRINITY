//! Synchronization tests — fence and barrier operations.
//!
//! Mirrors tests/platform/rhi/test_sync.py.

mod common;

use common::*;
use std::thread;
use std::time::{Duration, Instant};

// =========================================================================
// Fence tests
// =========================================================================

#[test]
fn test_fence_creation() {
    let fence = MockFence::new(0);
    assert_eq!(fence.value(), 0);
    assert!(fence.handle() > 0);
}

#[test]
fn test_fence_initial_value() {
    let fence = MockFence::new(42);
    assert_eq!(fence.value(), 42);
}

#[test]
fn test_fence_signal() {
    let fence = MockFence::new(0);
    fence.signal(10);
    assert_eq!(fence.value(), 10);

    fence.signal(20);
    assert_eq!(fence.value(), 20);
}

#[test]
fn test_fence_is_complete() {
    let fence = MockFence::new(0);

    assert!(fence.is_complete(0));
    assert!(!fence.is_complete(1));

    fence.signal(5);

    assert!(fence.is_complete(0));
    assert!(fence.is_complete(5));
    assert!(!fence.is_complete(6));
}

#[test]
fn test_fence_wait_immediate() {
    let fence = MockFence::new(10);

    // Should return immediately since already at value
    let result = fence.wait(5, 1000);
    assert!(result);
}

#[test]
fn test_fence_wait_timeout() {
    let fence = MockFence::new(0);

    // Should timeout since fence won't be signaled
    let start = Instant::now();
    let result = fence.wait(100, 100);
    let elapsed_ms = start.elapsed().as_millis();

    assert!(!result);
    assert!(elapsed_ms >= 80); // Should wait at least ~100ms
}

#[test]
fn test_fence_wait_signal_threaded() {
    let fence = MockFence::new(0);
    let fence_clone = fence.clone();
    let wait_result = std::sync::Arc::new(std::sync::Mutex::new(None::<bool>));
    let wait_result_clone = wait_result.clone();

    let waiter = thread::spawn(move || {
        let result = fence_clone.wait(5, 1000);
        *wait_result_clone.lock().unwrap() = Some(result);
    });

    let signaller = thread::spawn(move || {
        thread::sleep(Duration::from_millis(100));
        fence.signal(5);
    });

    waiter.join().unwrap();
    signaller.join().unwrap();

    assert_eq!(*wait_result.lock().unwrap(), Some(true));
}

#[test]
fn test_fence_wait_infinite() {
    let fence = MockFence::new(0);
    let fence_clone = fence.clone();

    let signaller = thread::spawn(move || {
        thread::sleep(Duration::from_millis(50));
        fence_clone.signal(1);
    });

    // Wait with infinite timeout (-1)
    let result = fence.wait(1, -1);
    signaller.join().unwrap();

    assert!(result);
}

#[test]
fn test_fence_multiple_threads() {
    let fence = MockFence::new(0);
    let results = std::sync::Arc::new(std::sync::Mutex::new(vec![false, false, false]));

    let mut handles = Vec::new();
    for i in 0..3 {
        let fence_clone = fence.clone();
        let results_clone = results.clone();
        handles.push(thread::spawn(move || {
            let result = fence_clone.wait(10, 1000);
            results_clone.lock().unwrap()[i] = result;
        }));
    }

    thread::sleep(Duration::from_millis(100));
    fence.signal(10);

    for h in handles {
        h.join().unwrap();
    }

    let results = results.lock().unwrap();
    assert!(results.iter().all(|&r| r));
}

#[test]
fn test_fence_incremental_signal() {
    let fence = MockFence::new(0);

    fence.signal(1);
    assert!(fence.is_complete(1));
    assert!(!fence.is_complete(2));

    fence.signal(2);
    assert!(fence.is_complete(2));
    assert!(!fence.is_complete(3));

    fence.signal(3);
    assert!(fence.is_complete(3));
}

#[test]
fn test_fence_reuse() {
    let fence = MockFence::new(0);

    // Signal-wait cycle 1
    fence.signal(1);
    assert!(fence.wait(1, 100));

    // Signal-wait cycle 2
    fence.signal(5);
    assert!(fence.wait(5, 100));
}

#[test]
fn test_fence_default_value_is_zero() {
    let fence = MockFence::new(0);
    assert_eq!(fence.value(), 0);
}

// =========================================================================
// Barrier descriptor tests
// =========================================================================

#[test]
fn test_barrier_desc_transition() {
    let barrier = BarrierDesc {
        ty: BarrierType::Transition,
        resource: None,
        state_before: ResourceState::CopyDst,
        state_after: ResourceState::ShaderResource,
    };

    assert_eq!(barrier.ty, BarrierType::Transition);
    assert_eq!(barrier.state_before, ResourceState::CopyDst);
    assert_eq!(barrier.state_after, ResourceState::ShaderResource);
}

#[test]
fn test_barrier_type_uav_vs_transition() {
    let uav = BarrierDesc {
        ty: BarrierType::Uav,
        resource: None,
        state_before: ResourceState::Undefined,
        state_after: ResourceState::Undefined,
    };
    assert_eq!(uav.ty, BarrierType::Uav);

    let transition = BarrierDesc {
        ty: BarrierType::Transition,
        resource: None,
        state_before: ResourceState::Common,
        state_after: ResourceState::UnorderedAccess,
    };
    assert_eq!(transition.ty, BarrierType::Transition);

    assert_ne!(uav.ty, transition.ty);
}

#[test]
fn test_barrier_desc_alias() {
    let alias = BarrierDesc {
        ty: BarrierType::Aliasing,
        resource: Some(42),
        state_before: ResourceState::Undefined,
        state_after: ResourceState::Common,
    };
    assert_eq!(alias.ty, BarrierType::Aliasing);
    assert_eq!(alias.resource, Some(42));
}

#[test]
fn test_uav_barrier_desc() {
    let barrier = BarrierDesc {
        ty: BarrierType::Uav,
        resource: None,
        state_before: ResourceState::Undefined,
        state_after: ResourceState::Undefined,
    };
    assert_eq!(barrier.ty, BarrierType::Uav);
}

#[test]
fn test_resource_state_transition_render_to_present() {
    let barrier = BarrierDesc {
        ty: BarrierType::Transition,
        resource: None,
        state_before: ResourceState::RenderTarget,
        state_after: ResourceState::Present,
    };

    assert_eq!(barrier.state_before, ResourceState::RenderTarget);
    assert_eq!(barrier.state_after, ResourceState::Present);
}

#[test]
fn test_all_resource_states() {
    let states = [
        ResourceState::Undefined,
        ResourceState::Common,
        ResourceState::RenderTarget,
        ResourceState::DepthWrite,
        ResourceState::DepthRead,
        ResourceState::ShaderResource,
        ResourceState::UnorderedAccess,
        ResourceState::CopySrc,
        ResourceState::CopyDst,
        ResourceState::Present,
    ];

    for &state in &states {
        let barrier = BarrierDesc {
            ty: BarrierType::Transition,
            resource: None,
            state_before: ResourceState::Undefined,
            state_after: state,
        };
        assert_eq!(barrier.state_after, state);
    }
}

#[test]
fn test_all_barrier_types() {
    for ty in &[BarrierType::Transition, BarrierType::Uav, BarrierType::Aliasing] {
        let barrier = BarrierDesc {
            ty: *ty,
            resource: None,
            state_before: ResourceState::Undefined,
            state_after: ResourceState::Common,
        };
        assert_eq!(barrier.ty, *ty);
    }
}
