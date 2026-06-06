// Blackbox contract tests for DAG builder + topological sort (T-FG-2.7).
//
// CLEANROOM: No src/ access beyond the public API exported by the crate.
// Tests use only `renderer_backend::frame_graph::*` -- no internal fields,
// no private methods, no implementation details.
//
// Acceptance criteria:
//   1.  build_dag on linear chain of 10 passes produces 9 edges (N-1)
//   2.  build_dag on diamond pattern produces correct edge count (4)
//   3.  topological_sort on linear chain returns sequential PassIndex order
//   4.  topological_sort on diamond returns valid partial order (root-first,
//       merge-last, branches unconstrained)
//   5.  topological_sort on empty input returns Ok(vec![])
//   6.  Cycle detection: topological_sort on manually-created cycle
//       returns Err
//   7.  Stress: 50 passes with varied dependencies, sort completes without
//       error (no assertion on timing)

use std::sync::Arc;
use renderer_backend::frame_graph::{
    build_dag, topological_sort, DispatchSource, EdgeType, EmptyView, InstanceSource, IrEdge,
    IrPass, IrResource, PassFlags, PassIndex, PassType, ResourceAccessSet, ResourceHandle,
    ViewType,
};

// ---------------------------------------------------------------------------
// Helper: create a single IrPass with a given access pattern
// ---------------------------------------------------------------------------

fn make_pass(index: usize, reads: &[ResourceHandle], writes: &[ResourceHandle]) -> IrPass {
    let name = format!("pass_{}", index);
    IrPass {
        index: PassIndex(index),
        name: name.clone(),
        pass_type: PassType::Compute,
        access_set: ResourceAccessSet {
            reads: reads.to_vec(),
            writes: writes.to_vec(),
        },
        color_attachments: Vec::new(),
        depth_stencil: None,
        instance_source: InstanceSource::Direct {
            index_count: 0,
            instance_count: 1,
            base_vertex: 0,
            first_index: 0,
            first_instance: 0,
        },
        dispatch_source: Some(DispatchSource::Direct {
            group_count_x: 1,
            group_count_y: 1,
            group_count_z: 1,
        }),
        view_type: ViewType::Storage,
        view: Arc::new(EmptyView { name }),
        tags: Vec::new(),
        flags: PassFlags::empty(),
    }
}

// ---------------------------------------------------------------------------
// Helper: produce a placeholder IrResource slice (build_dag accepts it but
// currently does not use it —_resources parameter is underscore-prefixed).
// ---------------------------------------------------------------------------

fn placeholder_resources() -> Vec<IrResource> {
    Vec::new()
}

// =============================================================================
// TEST 1 -- build_dag on a linear chain of 10 passes
// =============================================================================

#[test]
fn build_dag_linear_chain_10_passes_produces_9_edges() {
    // Chain: pass_0 writes R0, pass_1 reads R0+writes R1, ..., pass_9 reads R8
    // Each consecutive pair of passes shares exactly one resource.
    // build_dag should produce N-1 = 9 edges, all RAW.
    let mut passes = Vec::with_capacity(10);

    // First pass writes resource 0.
    passes.push(make_pass(0, &[], &[ResourceHandle(0)]));

    // Intermediate passes read previous resource, write next resource.
    for i in 1..9 {
        passes.push(make_pass(
            i,
            &[ResourceHandle((i - 1) as u32)],  // read predecessor
            &[ResourceHandle(i as u32)],         // write our own
        ));
    }

    // Last pass reads resource 8 only (no write needed).
    passes.push(make_pass(9, &[ResourceHandle(8)], &[]));

    let resources = placeholder_resources();
    let edges = build_dag(&passes, &resources);

    // Exactly N-1 edges.
    assert_eq!(
        edges.len(),
        9,
        "linear chain of 10 passes should produce 9 edges, got {}",
        edges.len()
    );

    // Every edge is RAW, from i to i+1, over resource i.  Order is
    // non-deterministic (build_dag uses HashMap internally), so we check
    // existence rather than position.
    let expected: Vec<(PassIndex, PassIndex, ResourceHandle, EdgeType)> = (0..9)
        .map(|i| {
            (
                PassIndex(i),
                PassIndex(i + 1),
                ResourceHandle(i as u32),
                EdgeType::RAW,
            )
        })
        .collect();

    for edge in &edges {
        assert_eq!(
            edge.edge_type,
            EdgeType::RAW,
            "unexpected non-RAW edge: {:?}",
            edge
        );
        let found = expected
            .iter()
            .any(|(from, to, res, _)| edge.from == *from && edge.to == *to && edge.resource == *res);
        assert!(
            found,
            "unexpected or mismatched edge: {:?}",
            edge
        );
    }
}

// =============================================================================
// TEST 2 -- build_dag on a diamond pattern
// =============================================================================

#[test]
fn build_dag_diamond_pattern_produces_4_edges() {
    // Diamond:
    //   pass_0 (root, writes R_A)
    //   ├── pass_1 (left, reads R_A, writes R_B)
    //   ├── pass_2 (right, reads R_A, writes R_C)
    //   └── pass_3 (merge, reads R_B, R_C)
    //
    // Expected edges:
    //   R_A: pass_0(w) → pass_1(r)  RAW
    //   R_A: pass_0(w) → pass_2(r)  RAW
    //   R_B: pass_1(w) → pass_3(r)  RAW
    //   R_C: pass_2(w) → pass_3(r)  RAW
    //   Total: 4

    let r_a = ResourceHandle(0);
    let r_b = ResourceHandle(1);
    let r_c = ResourceHandle(2);

    let passes = vec![
        make_pass(0, &[], &[r_a]),                          // root writes A
        make_pass(1, &[r_a], &[r_b]),                       // left: reads A, writes B
        make_pass(2, &[r_a], &[r_c]),                       // right: reads A, writes C
        make_pass(3, &[r_b, r_c], &[]),                     // merge: reads B and C
    ];

    let resources = placeholder_resources();
    let edges = build_dag(&passes, &resources);

    assert_eq!(
        edges.len(),
        4,
        "diamond pattern should produce 4 edges, got {}",
        edges.len()
    );

    // Verify each expected edge exists in the result.
    let expected: Vec<(PassIndex, PassIndex, ResourceHandle, EdgeType)> = vec![
        (PassIndex(0), PassIndex(1), r_a, EdgeType::RAW),
        (PassIndex(0), PassIndex(2), r_a, EdgeType::RAW),
        (PassIndex(1), PassIndex(3), r_b, EdgeType::RAW),
        (PassIndex(2), PassIndex(3), r_c, EdgeType::RAW),
    ];

    for (from, to, res, etype) in &expected {
        let found = edges.iter().any(|e| {
            e.from == *from && e.to == *to && e.resource == *res && e.edge_type == *etype
        });
        assert!(
            found,
            "missing expected edge: from={:?}, to={:?}, resource={:?}, type={:?}",
            from, to, res, etype
        );
    }

    // Verify no unexpected edges.
    for edge in &edges {
        let found = expected.iter().any(|(from, to, res, etype)| {
            edge.from == *from
                && edge.to == *to
                && edge.resource == *res
                && edge.edge_type == *etype
        });
        assert!(
            found,
            "unexpected edge: from={:?}, to={:?}, resource={:?}, type={:?}",
            edge.from, edge.to, edge.resource, edge.edge_type
        );
    }
}

// =============================================================================
// TEST 3 -- topological_sort on linear chain produces sequential order
// =============================================================================

#[test]
fn topological_sort_linear_chain_returns_sequential_order() {
    // Same chain as test 1: pass_0 writes R0, pass_1 reads R0, ...
    let mut passes = Vec::with_capacity(10);
    passes.push(make_pass(0, &[], &[ResourceHandle(0)]));
    for i in 1..9 {
        passes.push(make_pass(
            i,
            &[ResourceHandle((i - 1) as u32)],
            &[ResourceHandle(i as u32)],
        ));
    }
    passes.push(make_pass(9, &[ResourceHandle(8)], &[]));

    let edges = build_dag(&passes, &placeholder_resources());
    let sorted = topological_sort(&passes, &edges).expect("linear chain should not contain a cycle");

    assert_eq!(
        sorted.len(),
        10,
        "sorted output should contain 10 passes, got {}",
        sorted.len()
    );

    for (i, pass_idx) in sorted.iter().enumerate() {
        assert_eq!(
            *pass_idx,
            PassIndex(i),
            "position {} should be PassIndex({}), got {:?}",
            i,
            i,
            pass_idx
        );
    }
}

// =============================================================================
// TEST 4 -- topological_sort on diamond produces valid partial order
// =============================================================================

#[test]
fn topological_sort_diamond_respects_partial_order() {
    // Same diamond as test 2.
    let r_a = ResourceHandle(0);
    let r_b = ResourceHandle(1);
    let r_c = ResourceHandle(2);

    let passes = vec![
        make_pass(0, &[], &[r_a]),
        make_pass(1, &[r_a], &[r_b]),
        make_pass(2, &[r_a], &[r_c]),
        make_pass(3, &[r_b, r_c], &[]),
    ];

    let edges = build_dag(&passes, &placeholder_resources());
    let sorted =
        topological_sort(&passes, &edges).expect("diamond should not contain a cycle");

    assert_eq!(
        sorted.len(),
        4,
        "sorted output should contain 4 passes, got {}",
        sorted.len()
    );

    // Compute position of each pass in the sorted order.
    fn pos(order: &[PassIndex], target: PassIndex) -> usize {
        order.iter().position(|p| *p == target).unwrap()
    }

    let p0 = pos(&sorted, PassIndex(0));
    let p1 = pos(&sorted, PassIndex(1));
    let p2 = pos(&sorted, PassIndex(2));
    let p3 = pos(&sorted, PassIndex(3));

    // Root must come before left and right branches.
    assert!(
        p0 < p1,
        "root (pass_0) must precede left branch (pass_1)"
    );
    assert!(
        p0 < p2,
        "root (pass_0) must precede right branch (pass_2)"
    );

    // Merge must come after both branches.
    assert!(
        p1 < p3,
        "left branch (pass_1) must precede merge (pass_3)"
    );
    assert!(
        p2 < p3,
        "right branch (pass_2) must precede merge (pass_3)"
    );

    // Left and right branches have no mutual dependency -- either order is valid.
}

// =============================================================================
// TEST 5 -- topological_sort on empty input
// =============================================================================

#[test]
fn topological_sort_empty_input_returns_empty() {
    let passes: Vec<IrPass> = Vec::new();
    let edges: Vec<IrEdge> = Vec::new();
    let result = topological_sort(&passes, &edges);

    assert!(result.is_ok(), "empty input should return Ok, got Err");
    assert!(
        result.unwrap().is_empty(),
        "sorted output for empty input should be empty"
    );
}

// =============================================================================
// TEST 6 -- topological_sort cycle detection
// =============================================================================

#[test]
fn topological_sort_cycle_detection_returns_err() {
    // Three passes with edges that form a cycle: 0 → 1 → 2 → 0.
    let passes = vec![
        make_pass(0, &[], &[]),
        make_pass(1, &[], &[]),
        make_pass(2, &[], &[]),
    ];

    // Manually create a cycle (build_dag cannot produce back-edges because it
    // only goes from lower to higher insertion index).
    let edges = vec![
        IrEdge::new(PassIndex(0), PassIndex(1), ResourceHandle(0), EdgeType::RAW),
        IrEdge::new(PassIndex(1), PassIndex(2), ResourceHandle(1), EdgeType::RAW),
        IrEdge::new(PassIndex(2), PassIndex(0), ResourceHandle(2), EdgeType::RAW),
    ];

    let result = topological_sort(&passes, &edges);

    assert!(
        result.is_err(),
        "cycle should produce Err, got Ok: {:?}",
        result
    );

    // Verify the error message mentions "Cycle".
    let err_msg = result.unwrap_err();
    assert!(
        err_msg.contains("Cycle"),
        "error message should contain 'Cycle', got: {}",
        err_msg
    );
}

// =============================================================================
// TEST 7 -- 50-pass stress test with varied dependencies
// =============================================================================

#[test]
fn topological_sort_stress_50_passes_with_varied_deps() {
    // Build 50 passes where each pass i (i >= 1) reads from:
    //   - the immediate predecessor (i-1)  [chain dependency]
    //   - a further back predecessor (i/2) [branching dependency]
    // Pass 0 writes resource 0.
    // This creates a DAG with a mix of chain and branch edges.
    //
    // The test verifies that topological_sort completes without error and
    // produces exactly 50 entries. No assertion on timing.
    let n: usize = 50;

    let mut passes = Vec::with_capacity(n);
    passes.push(make_pass(0, &[], &[ResourceHandle(0)]));

    for i in 1..n {
        let mut reads = vec![ResourceHandle(i as u32 - 1)];       // predecessor
        let back = ResourceHandle((i / 2) as u32);                 // branching dep
        if back != ResourceHandle(i as u32 - 1) {
            reads.push(back);
        }
        passes.push(make_pass(i, &reads, &[ResourceHandle(i as u32)]));
    }

    let edges = build_dag(&passes, &placeholder_resources());
    let sorted = topological_sort(&passes, &edges)
        .expect("50-pass varied-dep graph should be acyclic");

    assert_eq!(
        sorted.len(),
        n,
        "sorted output should contain {} passes, got {}",
        n,
        sorted.len()
    );

    // Verify the ordering respects every edge.
    let position: std::collections::HashMap<PassIndex, usize> = sorted
        .iter()
        .enumerate()
        .map(|(idx, p)| (*p, idx))
        .collect();

    for edge in &edges {
        let from_pos = position.get(&edge.from).expect("edge.from must be in sorted output");
        let to_pos = position.get(&edge.to).expect("edge.to must be in sorted output");
        assert!(
            from_pos < to_pos,
            "edge {:?}: from {:?} (pos {}) must precede to {:?} (pos {})",
            edge,
            edge.from,
            from_pos,
            edge.to,
            to_pos,
        );
    }
}
