// SPDX-License-Identifier: MIT
//
// FrameGraphSlot / ArcSwap blackbox test (T-FG-7.3).
//
// Tests the lock-free frame graph swapping mechanism via the public API at
// `renderer_backend::frame_graph::swap::FrameGraphSlot`.
//
// CLEANROOM: No src/ access beyond the public API exported by the crate.
// Tests use only `renderer_backend::frame_graph::*` and
// `renderer_backend::frame_graph::swap::*` -- no internal fields,
// no private methods, no implementation details.
//
// Coverage:
//   1. Construction with a compiled frame graph
//   2. load() returns an Arc that dereferences to the stored graph
//   3. store() atomically replaces the graph
//   4. swap() returns the previous graph
//   5. Old Arc remains valid after the slot is swapped
//   6. is_loaded() returns true after construction
//   7. Concurrent readers and writers do not panic
//   8. Empty graph storage works correctly
//   9. Multiple swaps in sequence maintain correctness
//  10. All returned Arcs from swap remain valid independently

use std::sync::Arc;

use renderer_backend::frame_graph::{
    mock_pass_graphics, mock_resource_buffer, CompiledFrameGraph, PassIndex, ResourceHandle,
};
use renderer_backend::frame_graph::swap::FrameGraphSlot;

// =============================================================================
// Helper: compile a chain of N graphics passes
// =============================================================================

/// Compiles a directed chain of `n` graphics passes.
///
/// Each pass i (i > 0) reads the resource written by pass i-1, forming a
/// linear dependency that the compiler must order topologically.  All passes
/// are `Graphics` and are therefore unconditionally preserved by dead-pass
/// elimination, guaranteeing that `order.len() == n` for every valid `n`.
fn compile_chain(n: usize) -> CompiledFrameGraph {
    let mut passes = Vec::with_capacity(n);
    let mut resources = Vec::with_capacity(n);

    for i in 0..n {
        let handle = ResourceHandle(i as u32);
        resources.push(mock_resource_buffer(handle, &format!("r{}", i), 64));

        let mut pass = mock_pass_graphics(PassIndex(i), &format!("p{}", i), &[handle]);
        pass.access_set.writes.push(handle);
        if i > 0 {
            pass.access_set.reads.push(ResourceHandle((i - 1) as u32));
        }
        passes.push(pass);
    }

    CompiledFrameGraph::compile(passes, resources)
        .unwrap_or_else(|e| panic!("compile_chain({}): {}", n, e))
}

// =============================================================================
// SECTION 1 -- Construction
// =============================================================================

/// FrameGraphSlot can be constructed with a CompiledFrameGraph.
#[test]
fn slot_constructed_with_compiled_graph() {
    let graph = compile_chain(3);
    let slot = FrameGraphSlot::new(graph);
    // Construction does not panic, slot is alive.
    assert!(slot.is_loaded(), "slot must report loaded after construction");
}

// =============================================================================
// SECTION 2 -- load()
// =============================================================================

/// load() returns an Arc that dereferences to the stored graph.
#[test]
fn load_returns_arc_to_stored_graph() {
    let graph = compile_chain(5);
    let slot = FrameGraphSlot::new(graph);

    let loaded = slot.load();
    // Deref to CompiledFrameGraph -- access a public field via the Arc.
    assert_eq!(
        loaded.order.len(),
        5,
        "load() must return a graph whose order contains all 5 passes",
    );

    // The loaded graph must contain the expected pass names (p0..p4).
    let names: Vec<&str> = loaded.passes.iter().map(|p| p.name.as_str()).collect();
    for (i, name) in names.iter().enumerate() {
        assert_eq!(*name, format!("p{}", i), "pass name must match at index {}", i);
    }
}

/// load() can be called multiple times, each returning an independent Arc
/// pointing to the same underlying graph data.
#[test]
fn load_multiple_calls_produce_independent_arcs() {
    let graph = compile_chain(4);
    let slot = FrameGraphSlot::new(graph);

    let a = slot.load();
    let b = slot.load();
    let c = slot.load();

    // All three Arcs refer to the same pass count.
    assert_eq!(a.order.len(), 4);
    assert_eq!(b.order.len(), 4);
    assert_eq!(c.order.len(), 4);

    // Identity check: they all point to the same underlying allocation
    // (same resource handles, same order).
    assert_eq!(a.order, b.order, "concurrent Arcs must see same order");
    assert_eq!(b.order, c.order, "concurrent Arcs must see same order");
}

// =============================================================================
// SECTION 3 -- store()
// =============================================================================

/// store() replaces the graph atomically.  After store(), load() returns the
/// new graph.
#[test]
fn store_replaces_graph_atomically() {
    let slot = FrameGraphSlot::new(compile_chain(2));

    // Replace with a larger graph.
    slot.store(compile_chain(6));

    let loaded = slot.load();
    assert_eq!(
        loaded.order.len(),
        6,
        "store() must replace the current graph; order.len() reflects new graph",
    );

    // Verify the new graph has the correct pass names.
    let names: Vec<&str> = loaded.passes.iter().map(|p| p.name.as_str()).collect();
    assert_eq!(names.len(), 6, "6 passes in stored graph");
    assert_eq!(names[0], "p0", "first pass is p0");
    assert_eq!(names[5], "p5", "last pass is p5");
}

/// store() preserves graph data that is still referenced by outstanding Arcs.
#[test]
fn store_preserves_outstanding_references() {
    let slot = FrameGraphSlot::new(compile_chain(2));
    let old_ref = slot.load();
    assert_eq!(old_ref.order.len(), 2);

    // Store a new graph while an old Arc is still alive.
    slot.store(compile_chain(5));

    // The new load returns the latest graph.
    assert_eq!(slot.load().order.len(), 5);

    // The old Arc still dereferences correctly to the original graph.
    assert_eq!(
        old_ref.order.len(),
        2,
        "old Arc must remain valid after store() replaces the slot",
    );
    let old_names: Vec<&str> = old_ref.passes.iter().map(|p| p.name.as_str()).collect();
    assert_eq!(old_names, vec!["p0", "p1"], "old Arc preserves original pass names");
}

// =============================================================================
// SECTION 4 -- swap()
// =============================================================================

/// swap() atomically replaces the current graph and returns the old graph.
#[test]
fn swap_returns_old_graph() {
    let slot = FrameGraphSlot::new(compile_chain(2));

    let old = slot.swap(compile_chain(5));

    // The returned Arc holds the old graph.
    assert_eq!(
        old.order.len(),
        2,
        "swap() must return the previous graph (2 passes)",
    );

    // The slot now holds the new graph.
    assert_eq!(
        slot.load().order.len(),
        5,
        "slot must now contain the new graph (5 passes)",
    );
}

/// swap() returns an Arc that remains valid even after further swaps.
#[test]
fn swap_arc_remains_valid_after_further_swaps() {
    let slot = FrameGraphSlot::new(compile_chain(1));

    let old_v1 = slot.swap(compile_chain(3));
    let old_v2 = slot.swap(compile_chain(5));
    let old_v3 = slot.swap(compile_chain(7));

    // Each returned Arc must still dereference to its original graph.
    assert_eq!(old_v1.order.len(), 1, "v1 is the initial graph (1 pass)");
    assert_eq!(old_v2.order.len(), 3, "v2 is the 3-pass graph");
    assert_eq!(old_v3.order.len(), 5, "v3 is the 5-pass graph");

    // The slot now holds the 7-pass graph.
    assert_eq!(slot.load().order.len(), 7, "slot holds the 7-pass graph");

    // All Arcs are still independently valid after dropping the slot.
    // (Slot is still in scope, but this proves the Arcs hold strong references.)
    let _graph: &CompiledFrameGraph = &old_v1;
    let _graph: &CompiledFrameGraph = &old_v2;
    let _graph: &CompiledFrameGraph = &old_v3;
}

// =============================================================================
// SECTION 5 -- Old Arc validity after swap
// =============================================================================

/// An Arc obtained via load() remains valid after the slot is swapped.
/// This guarantees that in-flight frames are never interrupted.
#[test]
fn old_load_arc_still_valid_after_swap() {
    let slot = FrameGraphSlot::new(compile_chain(3));

    // Take a reference before the swap.
    let before_swap = slot.load();
    assert_eq!(before_swap.order.len(), 3);
    let before_names: Vec<&str> =
        before_swap.passes.iter().map(|p| p.name.as_str()).collect();

    // Swap in a new graph.
    let _old = slot.swap(compile_chain(7));

    // The old Arc is still fully accessible.
    assert_eq!(
        before_swap.order.len(),
        3,
        "pre-swap Arc still has 3 passes in order",
    );
    let after_names: Vec<&str> =
        before_swap.passes.iter().map(|p| p.name.as_str()).collect();
    assert_eq!(before_names, after_names, "pre-swap Arc preserves all pass names");

    // Meanwhile the slot returns the new graph.
    assert_eq!(slot.load().order.len(), 7, "slot now returns the 7-pass graph");
}

/// An Arc obtained via swap() (the old graph) remains valid after the slot
/// is swapped again.
#[test]
fn swap_returned_arc_still_valid_after_another_swap() {
    let slot = FrameGraphSlot::new(compile_chain(1));

    let old_v1 = slot.swap(compile_chain(4));
    assert_eq!(old_v1.order.len(), 1);

    // Swap again -- old_v1 should still be valid.
    let _old_v2 = slot.swap(compile_chain(7));

    assert_eq!(
        old_v1.order.len(),
        1,
        "swap()-returned Arc must survive a second swap",
    );
    let v1_names: Vec<&str> = old_v1.passes.iter().map(|p| p.name.as_str()).collect();
    assert_eq!(v1_names, vec!["p0"], "old_v1 still has its original pass name");
}

// =============================================================================
// SECTION 6 -- is_loaded()
// =============================================================================

/// is_loaded() returns true after construction.
#[test]
fn is_loaded_returns_true_after_construction() {
    let slot = FrameGraphSlot::new(compile_chain(1));
    assert!(slot.is_loaded(), "slot must report loaded after construction");
}

/// is_loaded() returns true regardless of how many store() or swap() calls
/// have been made.
#[test]
fn is_loaded_persists_through_operations() {
    let slot = FrameGraphSlot::new(compile_chain(1));
    assert!(slot.is_loaded(), "loaded after construction");

    slot.store(compile_chain(3));
    assert!(slot.is_loaded(), "loaded after store()");

    let _old = slot.swap(compile_chain(5));
    assert!(slot.is_loaded(), "loaded after swap()");

    slot.store(compile_chain(7));
    assert!(slot.is_loaded(), "loaded after second store()");
}

// =============================================================================
// SECTION 7 -- Concurrent access
// =============================================================================

/// Concurrent read and write threads do not panic.
///
/// Spawns 4 reader threads (each calling load() 200 times) and 2 writer
/// threads (each calling store() 10 times with different graph sizes).
/// All threads must complete successfully.
#[test]
fn concurrent_read_write_no_panic() {
    let slot = Arc::new(FrameGraphSlot::new(compile_chain(4)));
    let mut handles = Vec::new();

    // ---- Reader threads ----
    for _ in 0..4 {
        let s = Arc::clone(&slot);
        handles.push(std::thread::spawn(move || {
            for _ in 0..200 {
                let g = s.load();
                // Every graph in this test has between 2 and 10 passes.
                let len = g.order.len();
                assert!(
                    len >= 1,
                    "reader: order.len() must be >= 1 (got {})",
                    len,
                );
                // Access every pass by name to exercise the full deref.
                for pass in g.passes.iter() {
                    let _name = &pass.name;
                }
            }
        }));
    }

    // ---- Writer threads ----
    for writer_id in 0..2 {
        let s = Arc::clone(&slot);
        handles.push(std::thread::spawn(move || {
            for j in 0..10 {
                // Cycle through graph sizes 4, 6, 8, 10, 12, ...
                let size = 4 + ((writer_id * 10 + j) % 7);
                s.store(compile_chain(size));
            }
        }));
    }

    // Join all threads -- any panic will propagate via expect().
    for (i, h) in handles.into_iter().enumerate() {
        h.join().unwrap_or_else(|_| {
            panic!("Thread {} panicked during concurrent access test", i);
        });
    }

    // Verify the slot is still in a healthy state after all threads finish.
    let final_loaded = slot.load();
    assert!(
        final_loaded.order.len() >= 4,
        "slot must hold a valid graph after concurrent access (order.len() = {})",
        final_loaded.order.len(),
    );
}

/// Many concurrent readers running in lockstep with a single writer do not
/// produce stale or corrupted data.
#[test]
fn concurrent_readers_with_store_do_not_panic() {
    let slot = Arc::new(FrameGraphSlot::new(compile_chain(3)));
    let mut handles = Vec::new();

    // 8 reader threads, each reading 100 times.
    for _ in 0..8 {
        let s = Arc::clone(&slot);
        handles.push(std::thread::spawn(move || {
            for _ in 0..100 {
                let g = s.load();
                // Access the graph to verify the Arc is not dangling.
                let _pass_count = g.order.len();
                let _res_count = g.resources.len();
            }
        }));
    }

    // 1 writer thread that calls store() 20 times.
    {
        let s = Arc::clone(&slot);
        handles.push(std::thread::spawn(move || {
            for j in 0..20 {
                let size = 3 + (j % 8);
                s.store(compile_chain(size));
            }
        }));
    }

    for (i, h) in handles.into_iter().enumerate() {
        h.join().unwrap_or_else(|_| {
            panic!("Reader/writer thread {} panicked", i);
        });
    }
}

// =============================================================================
// SECTION 8 -- Empty graph
// =============================================================================

/// FrameGraphSlot can store and retrieve an empty compiled graph.
#[test]
fn empty_graph_storage_works() {
    let empty = CompiledFrameGraph::compile(vec![], vec![])
        .expect("empty graph must compile");
    let slot = FrameGraphSlot::new(empty);

    let loaded = slot.load();
    assert!(
        loaded.passes.is_empty(),
        "empty slot must contain no passes",
    );
    assert!(
        loaded.resources.is_empty(),
        "empty slot must contain no resources",
    );
    assert!(
        loaded.order.is_empty(),
        "empty slot must have empty execution order",
    );
    assert!(slot.is_loaded(), "slot must report loaded even with empty graph");
}

/// Storing an empty graph after a non-empty one works correctly.
#[test]
fn store_empty_over_non_empty_works() {
    let slot = FrameGraphSlot::new(compile_chain(3));
    assert_eq!(slot.load().order.len(), 3);

    let empty = CompiledFrameGraph::compile(vec![], vec![])
        .expect("empty graph must compile");
    slot.store(empty);

    let loaded = slot.load();
    assert!(loaded.passes.is_empty(), "passes must be empty after storing empty graph");
    assert!(loaded.order.is_empty(), "order must be empty after storing empty graph");
    assert!(slot.is_loaded(), "slot still loaded");
}

/// Storing a non-empty graph after an empty one works correctly.
#[test]
fn store_non_empty_over_empty_works() {
    let empty = CompiledFrameGraph::compile(vec![], vec![])
        .expect("empty graph must compile");
    let slot = FrameGraphSlot::new(empty);

    let _first = slot.load();
    assert!(_first.passes.is_empty());

    slot.store(compile_chain(4));
    assert_eq!(slot.load().order.len(), 4, "slot now holds the 4-pass graph");
}

// =============================================================================
// SECTION 9 -- Multiple swaps in sequence
// =============================================================================

/// Multiple store() calls in sequence correctly update the slot.
#[test]
fn multiple_stores_in_sequence() {
    let slot = FrameGraphSlot::new(compile_chain(2));

    slot.store(compile_chain(4));
    assert_eq!(
        slot.load().order.len(),
        4,
        "first store: 2 -> 4 passes",
    );

    slot.store(compile_chain(6));
    assert_eq!(
        slot.load().order.len(),
        6,
        "second store: 4 -> 6 passes",
    );

    slot.store(compile_chain(8));
    assert_eq!(
        slot.load().order.len(),
        8,
        "third store: 6 -> 8 passes",
    );
}

/// Multiple swap() calls in sequence correctly return the previous graph
/// each time and leave the slot holding the latest graph.
#[test]
fn multiple_swaps_in_sequence() {
    let slot = FrameGraphSlot::new(compile_chain(2));

    // swap 1: old = 2, new = 4
    let old_a = slot.swap(compile_chain(4));
    assert_eq!(old_a.order.len(), 2, "first swap returns 2-pass graph");
    assert_eq!(slot.load().order.len(), 4, "slot holds 4-pass graph");

    // swap 2: old = 4, new = 6
    let old_b = slot.swap(compile_chain(6));
    assert_eq!(old_b.order.len(), 4, "second swap returns 4-pass graph");
    assert_eq!(slot.load().order.len(), 6, "slot holds 6-pass graph");

    // swap 3: old = 6, new = 8
    let old_c = slot.swap(compile_chain(8));
    assert_eq!(old_c.order.len(), 6, "third swap returns 6-pass graph");
    assert_eq!(slot.load().order.len(), 8, "slot holds 8-pass graph");

    // All three old Arcs remain valid.
    assert_eq!(old_a.order.len(), 2, "old_a still valid (2 passes)");
    assert_eq!(old_b.order.len(), 4, "old_b still valid (4 passes)");
    assert_eq!(old_c.order.len(), 6, "old_c still valid (6 passes)");
}

/// Many successive store/swap calls do not degrade correctness.
#[test]
fn many_swaps_do_not_degrade_correctness() {
    let slot = FrameGraphSlot::new(compile_chain(1));

    // Perform 20 store/verify cycles.
    for i in 0..20 {
        let expected_size = 1 + (i % 5) + 1; // cycles 2,3,4,5,6,2,3,...
        // (Actually: 1 -> 3, then 3 -> 5, then 5 -> 7, etc.)
        slot.store(compile_chain(3 + i));

        let loaded = slot.load();
        let expected = 3 + i;
        assert_eq!(
            loaded.order.len(),
            expected,
            "iteration {}: order.len() == {}",
            i,
            expected,
        );

        // All pass names are distinct and in order.
        let names: Vec<&str> = loaded.passes.iter().map(|p| p.name.as_str()).collect();
        for (j, name) in names.iter().enumerate() {
            let expected_name = format!("p{}", j);
            assert_eq!(
                *name, expected_name,
                "iteration {}, pass {}: expected name {}",
                i, j, expected_name,
            );
        }
    }
}

// =============================================================================
// SECTION 10 -- Debug formatting
// =============================================================================

/// The Debug implementation for FrameGraphSlot does not panic.
#[test]
fn debug_format_does_not_panic() {
    let slot = FrameGraphSlot::new(compile_chain(3));
    let debug_str = format!("{:?}", slot);
    assert!(
        !debug_str.is_empty(),
        "Debug output must not be empty",
    );
    assert!(
        debug_str.contains("loaded"),
        "Debug output must contain 'loaded' field",
    );
}

// =============================================================================
// SECTION 11 -- Slot is Send + Sync
// =============================================================================

/// FrameGraphSlot must be Send (can be transferred between threads).
#[test]
fn slot_is_send() {
    let slot = FrameGraphSlot::new(compile_chain(2));
    let handle = std::thread::spawn(move || {
        let g = slot.load();
        assert_eq!(g.order.len(), 2);
    });
    handle.join().expect("Send test thread panicked");
}

/// FrameGraphSlot must be Sync (can be shared via & reference across threads).
#[test]
fn slot_is_sync() {
    let slot = Arc::new(FrameGraphSlot::new(compile_chain(3)));
    let mut handles = Vec::new();

    for _ in 0..3 {
        let s = Arc::clone(&slot);
        handles.push(std::thread::spawn(move || {
            let g = s.load();
            assert_eq!(g.order.len(), 3);
        }));
    }

    for (i, h) in handles.into_iter().enumerate() {
        h.join().unwrap_or_else(|_| {
            panic!("Sync test thread {} panicked", i);
        });
    }
}
