// SPDX-License-Identifier: MIT
//
// HotReloadableFrameGraph blackbox tests (T-FG-7.8).
//
// Tests the hot-reload frame graph behavior via the public API at
// `renderer_backend::frame_graph::HotReloadableFrameGraph`.
//
// CLEANROOM: No src/ access beyond the public API exported by the crate.
// Tests use only observable behavior -- no internal fields, no private
// methods, no implementation assumptions (ArcSwap, Guard types are
// implementation details).
//
// Coverage:
//   1. Initial graph is accessible after creation
//   2. After swap, subsequent loads see the new graph
//   3. Old graph (from load() before swap) remains valid after swap
//   4. Multiple consecutive swaps always return the latest graph
//   5. swap_and_get_old returns the previous graph
//   6. load_full returns an Arc that remains valid
//   7. Concurrent load/swap operations do not panic

use std::sync::Arc;
use std::thread;

use renderer_backend::frame_graph::{
    mock_pass_graphics, mock_resource_buffer, CompiledFrameGraph, HotReloadableFrameGraph,
    PassIndex, ResourceHandle,
};

// =============================================================================
// Helper: compile a chain of N graphics passes
// =============================================================================

/// Compiles a directed chain of `n` graphics passes with distinct names.
///
/// Each pass i (i > 0) reads the resource written by pass i-1, forming a
/// linear dependency. All passes are `Graphics` type.
fn compile_chain(n: usize) -> CompiledFrameGraph {
    let mut passes = Vec::with_capacity(n);
    let mut resources = Vec::with_capacity(n);

    for i in 0..n {
        let handle = ResourceHandle(i as u32);
        resources.push(mock_resource_buffer(handle, &format!("r{}", i), 64));

        let mut pass = mock_pass_graphics(PassIndex(i), &format!("pass_{}", i), &[handle]);
        pass.access_set.writes.push(handle);
        if i > 0 {
            pass.access_set.reads.push(ResourceHandle((i - 1) as u32));
        }
        passes.push(pass);
    }

    CompiledFrameGraph::compile(passes, resources)
        .unwrap_or_else(|e| panic!("compile_chain({}): {}", n, e))
}

/// Compiles a graph with passes named with a prefix for identification.
fn compile_chain_with_prefix(n: usize, prefix: &str) -> CompiledFrameGraph {
    let mut passes = Vec::with_capacity(n);
    let mut resources = Vec::with_capacity(n);

    for i in 0..n {
        let handle = ResourceHandle(i as u32);
        resources.push(mock_resource_buffer(handle, &format!("{}r{}", prefix, i), 64));

        let mut pass = mock_pass_graphics(
            PassIndex(i),
            &format!("{}pass_{}", prefix, i),
            &[handle],
        );
        pass.access_set.writes.push(handle);
        if i > 0 {
            pass.access_set.reads.push(ResourceHandle((i - 1) as u32));
        }
        passes.push(pass);
    }

    CompiledFrameGraph::compile(passes, resources)
        .unwrap_or_else(|e| panic!("compile_chain({}, {}): {}", n, prefix, e))
}

// =============================================================================
// TEST 1: Initial graph is accessible after creation
// =============================================================================

/// After creating a HotReloadableFrameGraph with an initial graph, the graph
/// can be loaded and accessed immediately.
#[test]
fn test_hot_reload_initial_graph_accessible() {
    let initial_graph = compile_chain(5);
    let hot_reload = HotReloadableFrameGraph::new(initial_graph);

    // load() should return a reference to the graph
    let loaded = hot_reload.load();

    // Observable behavior: we can access the compiled graph data
    assert_eq!(
        loaded.order.len(),
        5,
        "Initial graph must have 5 passes in execution order"
    );

    // Verify pass names are accessible
    let pass_names: Vec<&str> = loaded.passes.iter().map(|p| p.name.as_str()).collect();
    assert_eq!(pass_names.len(), 5, "Must have 5 passes");
    assert_eq!(pass_names[0], "pass_0", "First pass must be pass_0");
    assert_eq!(pass_names[4], "pass_4", "Last pass must be pass_4");

    // Verify resources are accessible
    assert_eq!(
        loaded.resources.len(),
        5,
        "Initial graph must have 5 resources"
    );
}

/// load_full() also returns an accessible initial graph.
#[test]
fn test_hot_reload_initial_graph_accessible_via_load_full() {
    let initial_graph = compile_chain(3);
    let hot_reload = HotReloadableFrameGraph::new(initial_graph);

    let loaded_arc = hot_reload.load_full();

    assert_eq!(
        loaded_arc.order.len(),
        3,
        "load_full() must return graph with 3 passes"
    );
    assert_eq!(
        loaded_arc.passes[0].name,
        "pass_0",
        "First pass name must match"
    );
}

// =============================================================================
// TEST 2: After swap, subsequent loads see the new graph
// =============================================================================

/// After calling swap(), all subsequent load() calls return the new graph.
#[test]
fn test_hot_reload_swap_changes_graph() {
    // Create with initial 2-pass graph
    let initial = compile_chain_with_prefix(2, "old_");
    let hot_reload = HotReloadableFrameGraph::new(initial);

    // Verify initial state
    {
        let loaded = hot_reload.load();
        assert_eq!(loaded.order.len(), 2, "Initial graph has 2 passes");
        assert!(
            loaded.passes[0].name.starts_with("old_"),
            "Initial graph passes start with 'old_'"
        );
    }

    // Swap to a new 6-pass graph
    let new_graph = compile_chain_with_prefix(6, "new_");
    hot_reload.swap(new_graph);

    // All subsequent loads should see the new graph
    {
        let loaded = hot_reload.load();
        assert_eq!(
            loaded.order.len(),
            6,
            "After swap, load() must return new 6-pass graph"
        );
        assert!(
            loaded.passes[0].name.starts_with("new_"),
            "After swap, passes should be from new graph (prefix 'new_')"
        );
    }

    // Verify with load_full() as well
    {
        let loaded_arc = hot_reload.load_full();
        assert_eq!(
            loaded_arc.order.len(),
            6,
            "load_full() must also return the new graph after swap"
        );
    }

    // Multiple loads after swap should all see the new graph
    for _ in 0..5 {
        let loaded = hot_reload.load();
        assert_eq!(
            loaded.order.len(),
            6,
            "Every load after swap must see new graph"
        );
    }
}

// =============================================================================
// TEST 3: Old graph (loaded before swap) remains valid after swap
// =============================================================================

/// If load() is called before swap(), the returned value remains valid and
/// usable even after swap() replaces the internal graph. This is critical
/// for in-flight frame rendering.
#[test]
fn test_hot_reload_old_graph_not_dropped_while_in_use() {
    // Create with initial 4-pass graph
    let initial = compile_chain_with_prefix(4, "inflight_");
    let hot_reload = HotReloadableFrameGraph::new(initial);

    // Simulate an "in-flight frame" by loading before swap
    let inflight_graph = hot_reload.load();

    // Verify we captured the old graph
    assert_eq!(
        inflight_graph.order.len(),
        4,
        "In-flight graph must have 4 passes"
    );
    assert!(
        inflight_graph.passes[0].name.starts_with("inflight_"),
        "In-flight graph must have 'inflight_' prefix"
    );

    // Now swap to a completely different graph
    let replacement = compile_chain_with_prefix(8, "replacement_");
    hot_reload.swap(replacement);

    // The NEW loads should see the replacement graph
    {
        let new_load = hot_reload.load();
        assert_eq!(
            new_load.order.len(),
            8,
            "New loads see the 8-pass replacement"
        );
        assert!(
            new_load.passes[0].name.starts_with("replacement_"),
            "New graph has 'replacement_' prefix"
        );
    }

    // BUT the in-flight graph must STILL be valid and accessible
    // This is the key guarantee for in-flight frame safety
    assert_eq!(
        inflight_graph.order.len(),
        4,
        "In-flight graph must STILL have 4 passes after swap"
    );
    assert!(
        inflight_graph.passes[0].name.starts_with("inflight_"),
        "In-flight graph must STILL have original prefix after swap"
    );

    // Access every field to prove the graph is fully intact
    for (i, pass) in inflight_graph.passes.iter().enumerate() {
        let expected_name = format!("inflight_pass_{}", i);
        assert_eq!(
            pass.name, expected_name,
            "In-flight pass {} must retain original name",
            i
        );
    }
    for (i, resource) in inflight_graph.resources.iter().enumerate() {
        let expected_name = format!("inflight_r{}", i);
        assert_eq!(
            resource.name, expected_name,
            "In-flight resource {} must retain original name",
            i
        );
    }
}

/// Using load_full() before swap also maintains validity.
#[test]
fn test_hot_reload_old_arc_from_load_full_valid_after_swap() {
    let initial = compile_chain(3);
    let hot_reload = HotReloadableFrameGraph::new(initial);

    // Get an Arc before swap
    let inflight_arc = hot_reload.load_full();
    assert_eq!(inflight_arc.order.len(), 3);

    // Swap
    hot_reload.swap(compile_chain(7));

    // New loads see new graph
    assert_eq!(hot_reload.load_full().order.len(), 7);

    // Old Arc still valid
    assert_eq!(
        inflight_arc.order.len(),
        3,
        "Arc from load_full() before swap must remain valid"
    );
}

// =============================================================================
// TEST 4: Multiple consecutive swaps always return the latest graph
// =============================================================================

/// Can swap multiple times in succession, and load() always returns the
/// most recently swapped graph.
#[test]
fn test_hot_reload_multiple_consecutive_swaps() {
    let initial = compile_chain(1);
    let hot_reload = HotReloadableFrameGraph::new(initial);

    // Initial state
    assert_eq!(hot_reload.load().order.len(), 1);

    // First swap: 1 -> 3
    hot_reload.swap(compile_chain(3));
    assert_eq!(
        hot_reload.load().order.len(),
        3,
        "After 1st swap, must see 3-pass graph"
    );

    // Second swap: 3 -> 5
    hot_reload.swap(compile_chain(5));
    assert_eq!(
        hot_reload.load().order.len(),
        5,
        "After 2nd swap, must see 5-pass graph"
    );

    // Third swap: 5 -> 2
    hot_reload.swap(compile_chain(2));
    assert_eq!(
        hot_reload.load().order.len(),
        2,
        "After 3rd swap, must see 2-pass graph"
    );

    // Fourth swap: 2 -> 10
    hot_reload.swap(compile_chain(10));
    assert_eq!(
        hot_reload.load().order.len(),
        10,
        "After 4th swap, must see 10-pass graph"
    );

    // Verify all names match expected pattern
    let loaded = hot_reload.load();
    for (i, pass) in loaded.passes.iter().enumerate() {
        assert_eq!(
            pass.name,
            format!("pass_{}", i),
            "Pass {} name must match",
            i
        );
    }
}

/// Many swaps do not degrade correctness.
#[test]
fn test_hot_reload_many_swaps_maintain_correctness() {
    let hot_reload = HotReloadableFrameGraph::new(compile_chain(1));

    // Perform 50 swaps
    for i in 1..=50 {
        let size = (i % 10) + 1; // Sizes cycle 1-10
        hot_reload.swap(compile_chain(size));

        let loaded = hot_reload.load();
        assert_eq!(
            loaded.order.len(),
            size,
            "Iteration {}: load() must return graph with {} passes",
            i,
            size
        );
    }
}

// =============================================================================
// TEST 5: swap_and_get_old returns the previous graph
// =============================================================================

/// swap_and_get_old atomically swaps and returns an Arc to the old graph.
#[test]
fn test_hot_reload_swap_and_get_old_returns_previous() {
    let initial = compile_chain_with_prefix(3, "version1_");
    let hot_reload = HotReloadableFrameGraph::new(initial);

    // Swap and get the old graph
    let old = hot_reload.swap_and_get_old(compile_chain_with_prefix(5, "version2_"));

    // The returned Arc should be the v1 graph
    assert_eq!(
        old.order.len(),
        3,
        "swap_and_get_old must return the 3-pass v1 graph"
    );
    assert!(
        old.passes[0].name.starts_with("version1_"),
        "Returned graph must be version1"
    );

    // Current graph should be v2
    let current = hot_reload.load();
    assert_eq!(
        current.order.len(),
        5,
        "After swap_and_get_old, current graph must be 5-pass"
    );
    assert!(
        current.passes[0].name.starts_with("version2_"),
        "Current graph must be version2"
    );
}

/// Multiple swap_and_get_old calls chain correctly.
#[test]
fn test_hot_reload_chained_swap_and_get_old() {
    let hot_reload = HotReloadableFrameGraph::new(compile_chain(1));

    let old_v1 = hot_reload.swap_and_get_old(compile_chain(2));
    let old_v2 = hot_reload.swap_and_get_old(compile_chain(3));
    let old_v3 = hot_reload.swap_and_get_old(compile_chain(4));

    // Each returned Arc should have the expected size
    assert_eq!(old_v1.order.len(), 1, "v1 had 1 pass");
    assert_eq!(old_v2.order.len(), 2, "v2 had 2 passes");
    assert_eq!(old_v3.order.len(), 3, "v3 had 3 passes");

    // Current should be v4
    assert_eq!(hot_reload.load().order.len(), 4, "Current is v4 with 4 passes");

    // All old Arcs remain independently valid
    assert_eq!(old_v1.order.len(), 1);
    assert_eq!(old_v2.order.len(), 2);
    assert_eq!(old_v3.order.len(), 3);
}

// =============================================================================
// TEST 6: Concurrent access safety
// =============================================================================

/// Multiple threads can load() concurrently without panicking.
#[test]
fn test_hot_reload_concurrent_loads() {
    let hot_reload = Arc::new(HotReloadableFrameGraph::new(compile_chain(5)));

    let mut handles = Vec::new();

    // Spawn 8 reader threads
    for _ in 0..8 {
        let hr = Arc::clone(&hot_reload);
        handles.push(thread::spawn(move || {
            for _ in 0..100 {
                let loaded = hr.load();
                // Access graph to ensure no data race
                let _ = loaded.order.len();
                let _ = loaded.passes.len();
                for pass in loaded.passes.iter() {
                    let _ = &pass.name;
                }
            }
        }));
    }

    for h in handles {
        h.join().expect("Reader thread panicked");
    }
}

/// Concurrent loads and swaps do not panic.
#[test]
fn test_hot_reload_concurrent_load_and_swap() {
    let hot_reload = Arc::new(HotReloadableFrameGraph::new(compile_chain(3)));

    let mut handles = Vec::new();

    // 4 reader threads
    for _ in 0..4 {
        let hr = Arc::clone(&hot_reload);
        handles.push(thread::spawn(move || {
            for _ in 0..200 {
                let loaded = hr.load();
                // Graph must always have at least 1 pass
                assert!(
                    loaded.order.len() >= 1,
                    "Loaded graph must have at least 1 pass"
                );
            }
        }));
    }

    // 2 writer threads
    for writer_id in 0..2 {
        let hr = Arc::clone(&hot_reload);
        handles.push(thread::spawn(move || {
            for j in 0..20 {
                let size = 2 + ((writer_id * 10 + j) % 8);
                hr.swap(compile_chain(size));
            }
        }));
    }

    for (i, h) in handles.into_iter().enumerate() {
        h.join()
            .unwrap_or_else(|_| panic!("Thread {} panicked", i));
    }

    // Verify final state is valid
    let final_load = hot_reload.load();
    assert!(
        final_load.order.len() >= 2,
        "Final graph must be valid (at least 2 passes)"
    );
}

/// Old graphs from in-flight loads remain valid during concurrent swaps.
#[test]
fn test_hot_reload_inflight_safety_under_concurrency() {
    let hot_reload = Arc::new(HotReloadableFrameGraph::new(compile_chain(5)));

    let mut handles = Vec::new();

    // Threads that hold onto loaded graphs while swaps happen
    for _ in 0..4 {
        let hr = Arc::clone(&hot_reload);
        handles.push(thread::spawn(move || {
            for _ in 0..50 {
                // Capture a graph reference
                let captured = hr.load();
                let captured_len = captured.order.len();

                // Simulate some "rendering work" with the captured graph
                for pass in captured.passes.iter() {
                    let _ = &pass.name;
                }

                // After "work", verify the graph is still intact
                assert_eq!(
                    captured.order.len(),
                    captured_len,
                    "Captured graph must remain consistent during simulated rendering"
                );
            }
        }));
    }

    // Writer thread swapping rapidly
    {
        let hr = Arc::clone(&hot_reload);
        handles.push(thread::spawn(move || {
            for j in 0..30 {
                let size = 3 + (j % 7);
                hr.swap(compile_chain(size));
            }
        }));
    }

    for h in handles {
        h.join().expect("Thread panicked");
    }
}

// =============================================================================
// TEST 7: Empty graph handling
// =============================================================================

/// Can create with an empty graph.
#[test]
fn test_hot_reload_empty_graph() {
    let empty = CompiledFrameGraph::compile(vec![], vec![]).expect("Empty graph compiles");
    let hot_reload = HotReloadableFrameGraph::new(empty);

    let loaded = hot_reload.load();
    assert!(loaded.passes.is_empty(), "Empty graph has no passes");
    assert!(loaded.resources.is_empty(), "Empty graph has no resources");
    assert!(loaded.order.is_empty(), "Empty graph has empty order");
}

/// Can swap from non-empty to empty and back.
#[test]
fn test_hot_reload_swap_to_and_from_empty() {
    let hot_reload = HotReloadableFrameGraph::new(compile_chain(3));
    assert_eq!(hot_reload.load().order.len(), 3);

    // Swap to empty
    let empty = CompiledFrameGraph::compile(vec![], vec![]).expect("Empty graph compiles");
    hot_reload.swap(empty);
    assert!(
        hot_reload.load().order.is_empty(),
        "After swap to empty, order must be empty"
    );

    // Swap back to non-empty
    hot_reload.swap(compile_chain(4));
    assert_eq!(
        hot_reload.load().order.len(),
        4,
        "After swap from empty to non-empty, must have 4 passes"
    );
}

// =============================================================================
// TEST 8: Debug formatting
// =============================================================================

/// Debug formatting does not panic.
#[test]
fn test_hot_reload_debug_format() {
    let hot_reload = HotReloadableFrameGraph::new(compile_chain(2));
    let debug_str = format!("{:?}", hot_reload);
    assert!(
        !debug_str.is_empty(),
        "Debug output must not be empty"
    );
}
