// Blackbox contract tests for T-FG-2.5 ParallelRegions.
//
// CLEANROOM: No src/ access beyond the public API exported by the crate.
// Tests use only `renderer_backend::frame_graph::*` -- no internal fields,
// no private methods, no implementation details.
//
// Contract under test:
//   identify_parallel_regions(
//       order: &[PassIndex],
//       depths: &HashMap<PassIndex, u32>,
//       edges: &[IrEdge],
//   ) -> Vec<Vec<PassIndex>>
//
//   Identifies groups of passes at the same depth level that can execute
//   concurrently. Passes at different depths are in separate groups. Within
//   the same depth, passes connected by RAW (Read-After-Write) edges are
//   placed into separate sub-groups (serialised); WAR and WAW edges do NOT
//   force serialisation.
//
// Acceptance criteria:
//   1.  Empty input produces an empty result
//   2.  Single pass produces a single region containing that pass
//   3.  Linear chain: each pass at its own depth -> one pass per region
//   4.  Diamond graph: entry alone, mid_a+mid_b together, exit alone
//   5.  Same-depth passes WITH a RAW edge -> separate sub-regions
//   6.  Same-depth passes WITHOUT a RAW edge -> single region
//   7.  Same-depth passes connected by WAR/WAW only -> single region
//   8.  Multiple passes at the same depth with partial RAW coverage
//   9.  Mixed depths with parallel and serial segments
//  10.  All passes at depth 0, no edges -> single region
//  11.  Duplicate passes in order handled gracefully
//  12.  Ordering: passes are grouped without reordering

use std::collections::HashMap;

use renderer_backend::frame_graph::{
    build_dag, compute_pass_depths, identify_parallel_regions, topological_sort,
    BufferDesc, DispatchSource, EdgeType, IrEdge, IrPass, IrResource,
    PassIndex, ResourceDesc, ResourceHandle, ResourceLifetime, ResourceState,
    ViewType,
};

// ---------------------------------------------------------------------------
// Helper: a minimal compute pass that reads and/or writes given resources.
// ---------------------------------------------------------------------------

fn make_pass(
    index: usize,
    name: &str,
    reads: &[ResourceHandle],
    writes: &[ResourceHandle],
) -> IrPass {
    let mut pass = IrPass::compute(
        PassIndex(index),
        name,
        DispatchSource::Direct {
            group_count_x: 1,
            group_count_y: 1,
            group_count_z: 1,
        },
        ViewType::Storage,
    );
    for &r in reads {
        pass.access_set.reads.push(r);
    }
    for &w in writes {
        pass.access_set.writes.push(w);
    }
    pass
}

// Helper: create a minimal storage buffer resource.
fn storage_buf(size: u64) -> ResourceDesc {
    ResourceDesc::Buffer(BufferDesc {
        size,
        usage: "storage".into(),
        is_indirect_arg: false,
    })
}

// Helper: run the full DAG pipeline (build_dag -> topological_sort ->
// compute_pass_depths -> identify_parallel_regions).
fn compute_regions(passes: &[IrPass], resources: &[IrResource]) -> Vec<Vec<PassIndex>> {
    let edges = build_dag(passes, resources);
    let order = topological_sort(passes, &edges)
        .expect("DAG must be acyclic for valid test inputs");
    let depths = compute_pass_depths(&order, &edges);
    identify_parallel_regions(&order, &depths, &edges)
}

// =============================================================================
// SECTION 1 -- Empty and trivial inputs
// =============================================================================

#[test]
fn empty_passes_produces_empty_regions() {
    let passes: Vec<IrPass> = vec![];
    let regions = compute_regions(&passes, &[]);
    assert!(
        regions.is_empty(),
        "empty pass list must produce empty parallel regions"
    );
}

#[test]
fn empty_input_explicit() {
    let order: Vec<PassIndex> = vec![];
    let depths: HashMap<PassIndex, u32> = HashMap::new();
    let edges: Vec<IrEdge> = vec![];
    let regions = identify_parallel_regions(&order, &depths, &edges);
    assert!(
        regions.is_empty(),
        "explicit empty input must produce empty regions"
    );
}

#[test]
fn single_pass_produces_one_region() {
    let pass = make_pass(0, "solo", &[], &[ResourceHandle(1)]);
    let resources = vec![IrResource::new(
        ResourceHandle(1), "buf", storage_buf(64),
        ResourceLifetime::Transient, ResourceState::Uninitialized,
    )];
    let regions = compute_regions(&[pass], &resources);
    assert_eq!(regions.len(), 1, "single pass must produce 1 region");
    assert_eq!(
        regions[0],
        vec![PassIndex(0)],
        "region must contain the single pass"
    );
}

// =============================================================================
// SECTION 2 -- Linear chain (each pass at its own depth)
// =============================================================================

#[test]
fn linear_chain_each_pass_is_own_region() {
    // P0 -> P1 -> P2
    // Each pass has unique depth, so each gets its own region.
    let p0 = make_pass(0, "a", &[], &[ResourceHandle(1)]);
    let p1 = make_pass(1, "b", &[ResourceHandle(1)], &[ResourceHandle(2)]);
    let p2 = make_pass(2, "c", &[ResourceHandle(2)], &[]);

    let resources = vec![
        IrResource::new(ResourceHandle(1), "r1", storage_buf(64),
            ResourceLifetime::Transient, ResourceState::Uninitialized),
        IrResource::new(ResourceHandle(2), "r2", storage_buf(64),
            ResourceLifetime::Transient, ResourceState::Uninitialized),
    ];
    let regions = compute_regions(&[p0, p1, p2], &resources);
    assert_eq!(regions.len(), 3, "3-pass chain must produce 3 regions");
    assert_eq!(regions[0], vec![PassIndex(0)], "region 0 = [a]");
    assert_eq!(regions[1], vec![PassIndex(1)], "region 1 = [b]");
    assert_eq!(regions[2], vec![PassIndex(2)], "region 2 = [c]");
}

#[test]
fn four_pass_chain_four_regions() {
    let p0 = make_pass(0, "w", &[], &[ResourceHandle(1)]);
    let p1 = make_pass(1, "x", &[ResourceHandle(1)], &[ResourceHandle(2)]);
    let p2 = make_pass(2, "y", &[ResourceHandle(2)], &[ResourceHandle(3)]);
    let p3 = make_pass(3, "z", &[ResourceHandle(3)], &[]);

    let resources = vec![
        IrResource::new(ResourceHandle(1), "r1", storage_buf(32),
            ResourceLifetime::Transient, ResourceState::Uninitialized),
        IrResource::new(ResourceHandle(2), "r2", storage_buf(32),
            ResourceLifetime::Transient, ResourceState::Uninitialized),
        IrResource::new(ResourceHandle(3), "r3", storage_buf(32),
            ResourceLifetime::Transient, ResourceState::Uninitialized),
    ];
    let regions = compute_regions(&[p0, p1, p2, p3], &resources);
    assert_eq!(regions.len(), 4, "4-pass chain must produce 4 regions");
    for (i, region) in regions.iter().enumerate() {
        assert_eq!(
            region.len(), 1,
            "region {} in a chain must have exactly 1 pass", i
        );
        assert_eq!(
            region[0], PassIndex(i),
            "region {} must contain pass {}", i, i
        );
    }
}

// =============================================================================
// SECTION 3 -- Diamond graph
// =============================================================================

#[test]
fn diamond_graph_three_regions() {
    // Diamond:
    //       entry (depth 0)
    //      /           \
    //   mid_a         mid_b    (depth 1)
    //      \           /
    //        exit (depth 2)
    //
    // Regions: [entry], [mid_a, mid_b], [exit].

    let entry = make_pass(0, "entry", &[], &[ResourceHandle(1), ResourceHandle(2)]);
    let mid_a = make_pass(1, "mid_a", &[ResourceHandle(1)], &[ResourceHandle(3)]);
    let mid_b = make_pass(2, "mid_b", &[ResourceHandle(2)], &[ResourceHandle(4)]);
    let exit = make_pass(3, "exit", &[ResourceHandle(3), ResourceHandle(4)], &[]);

    let resources = vec![
        IrResource::new(ResourceHandle(1), "r1", storage_buf(64),
            ResourceLifetime::Transient, ResourceState::Uninitialized),
        IrResource::new(ResourceHandle(2), "r2", storage_buf(64),
            ResourceLifetime::Transient, ResourceState::Uninitialized),
        IrResource::new(ResourceHandle(3), "r3", storage_buf(64),
            ResourceLifetime::Transient, ResourceState::Uninitialized),
        IrResource::new(ResourceHandle(4), "r4", storage_buf(64),
            ResourceLifetime::Transient, ResourceState::Uninitialized),
    ];
    let regions = compute_regions(&[entry, mid_a, mid_b, exit], &resources);

    assert_eq!(regions.len(), 3, "diamond must produce 3 regions");

    // Region 0: entry alone.
    assert_eq!(regions[0], vec![PassIndex(0)], "region 0 = [entry]");

    // Region 1: mid_a and mid_b together (parallel at depth 1).
    assert_eq!(regions[1].len(), 2, "region 1 must contain 2 passes");
    assert!(
        regions[1].contains(&PassIndex(1)),
        "region 1 must contain mid_a"
    );
    assert!(
        regions[1].contains(&PassIndex(2)),
        "region 1 must contain mid_b"
    );

    // Region 2: exit alone.
    assert_eq!(regions[2], vec![PassIndex(3)], "region 2 = [exit]");
}

// =============================================================================
// SECTION 4 -- RAW edge serialisation within same depth
// =============================================================================

#[test]
fn raw_edge_at_same_depth_splits_into_sub_regions() {
    // Two passes at the same depth with a RAW edge between them.
    // The second must wait for the first.
    let order = vec![PassIndex(0), PassIndex(1)];
    let depths = HashMap::from([(PassIndex(0), 0u32), (PassIndex(1), 0u32)]);
    let edges = vec![IrEdge::new(
        PassIndex(0),
        PassIndex(1),
        ResourceHandle(1),
        EdgeType::RAW,
    )];
    let regions = identify_parallel_regions(&order, &depths, &edges);

    assert_eq!(regions.len(), 2, "RAW edge must split depth group into 2");
    assert_eq!(regions[0], vec![PassIndex(0)], "P0 goes first");
    assert_eq!(regions[1], vec![PassIndex(1)], "P1 goes second");
}

#[test]
fn three_passes_chain_raw_at_same_depth() {
    // P0 -> P1 -> P2 (all RAW, all same depth)
    // Expected: [P0], [P1], [P2] (three waves)
    let order = vec![PassIndex(0), PassIndex(1), PassIndex(2)];
    let depths = HashMap::from([
        (PassIndex(0), 0u32),
        (PassIndex(1), 0u32),
        (PassIndex(2), 0u32),
    ]);
    let edges = vec![
        IrEdge::new(PassIndex(0), PassIndex(1), ResourceHandle(1), EdgeType::RAW),
        IrEdge::new(PassIndex(1), PassIndex(2), ResourceHandle(2), EdgeType::RAW),
    ];
    let regions = identify_parallel_regions(&order, &depths, &edges);

    assert_eq!(regions.len(), 3, "RAW chain at same depth => 3 sub-regions");
    assert_eq!(regions[0], vec![PassIndex(0)], "wave 0 = [P0]");
    assert_eq!(regions[1], vec![PassIndex(1)], "wave 1 = [P1]");
    assert_eq!(regions[2], vec![PassIndex(2)], "wave 2 = [P2]");
}

// =============================================================================
// SECTION 5 -- No RAW edge: same-depth passes grouped together
// =============================================================================

#[test]
fn no_raw_edge_same_depth_single_region() {
    // Two passes at depth 0 writing to different resources -- no RAW edges.
    let order = vec![PassIndex(0), PassIndex(1)];
    let depths = HashMap::from([(PassIndex(0), 0u32), (PassIndex(1), 0u32)]);
    let edges = vec![]; // no edges at all
    let regions = identify_parallel_regions(&order, &depths, &edges);

    assert_eq!(regions.len(), 1, "no edges => single region for all same-depth passes");
    assert_eq!(regions[0].len(), 2, "region contains both passes");
    assert!(regions[0].contains(&PassIndex(0)));
    assert!(regions[0].contains(&PassIndex(1)));
}

#[test]
fn independent_passes_at_same_depth() {
    // Two passes writing to different resources, no cross-dependencies.
    let p0 = make_pass(0, "a", &[], &[ResourceHandle(1)]);
    let p1 = make_pass(1, "b", &[], &[ResourceHandle(2)]);

    let resources = vec![
        IrResource::new(ResourceHandle(1), "r1", storage_buf(64),
            ResourceLifetime::Transient, ResourceState::Uninitialized),
        IrResource::new(ResourceHandle(2), "r2", storage_buf(64),
            ResourceLifetime::Transient, ResourceState::Uninitialized),
    ];
    let regions = compute_regions(&[p0, p1], &resources);

    assert_eq!(regions.len(), 1, "independent passes => single region");
    assert_eq!(regions[0].len(), 2, "region contains both passes");
    assert!(regions[0].contains(&PassIndex(0)));
    assert!(regions[0].contains(&PassIndex(1)));
}

// =============================================================================
// SECTION 6 -- WAR and WAW edges do NOT force serialisation
// =============================================================================

#[test]
fn war_edge_does_not_force_serialisation() {
    // Two passes at same depth connected only by a WAR edge.
    // A WAR edge means pass 0 reads resource 1, pass 1 writes resource 1.
    // This is NOT a true data dependency for GPU parallelism -- only RAW
    // forces serialisation.
    let order = vec![PassIndex(0), PassIndex(1)];
    let depths = HashMap::from([(PassIndex(0), 0u32), (PassIndex(1), 0u32)]);
    let edges = vec![IrEdge::new(
        PassIndex(0),
        PassIndex(1),
        ResourceHandle(1),
        EdgeType::WAR,
    )];
    let regions = identify_parallel_regions(&order, &depths, &edges);

    assert_eq!(
        regions.len(), 1,
        "WAR edge must NOT split the depth group"
    );
    assert_eq!(regions[0].len(), 2, "both passes in one region");
}

#[test]
fn waw_edge_does_not_force_serialisation() {
    // Two passes at same depth connected only by a WAW edge.
    let order = vec![PassIndex(0), PassIndex(1)];
    let depths = HashMap::from([(PassIndex(0), 0u32), (PassIndex(1), 0u32)]);
    let edges = vec![IrEdge::new(
        PassIndex(0),
        PassIndex(1),
        ResourceHandle(1),
        EdgeType::WAW,
    )];
    let regions = identify_parallel_regions(&order, &depths, &edges);

    assert_eq!(
        regions.len(), 1,
        "WAW edge must NOT split the depth group"
    );
    assert_eq!(regions[0].len(), 2, "both passes in one region");
}

#[test]
fn raw_and_war_mixed_raw_still_splits() {
    // P0 -> P1 (RAW), P0 -> P2 (WAR).
    // Only RAW edges force serialisation.  P2 has no RAW predecessor,
    // so it is ready in wave 0 alongside P0.  P1 must wait until P0
    // is done.
    //
    // Wave 0: [P0, P2] (P2 has only a WAR edge from P0 -- no RAW)
    // Wave 1: [P1]      (P1's RAW pred P0 is gone)
    let order = vec![PassIndex(0), PassIndex(1), PassIndex(2)];
    let depths = HashMap::from([
        (PassIndex(0), 0u32),
        (PassIndex(1), 0u32),
        (PassIndex(2), 0u32),
    ]);
    let edges = vec![
        IrEdge::new(PassIndex(0), PassIndex(1), ResourceHandle(1), EdgeType::RAW),
        IrEdge::new(PassIndex(0), PassIndex(2), ResourceHandle(2), EdgeType::WAR),
    ];
    let regions = identify_parallel_regions(&order, &depths, &edges);

    assert_eq!(regions.len(), 2, "RAW between P0->P1 splits into 2 waves");
    assert_eq!(regions[0].len(), 2, "wave 0 = [P0, P2] (both RAW-pred-free)");
    assert!(regions[0].contains(&PassIndex(0)));
    assert!(regions[0].contains(&PassIndex(2)));
    assert_eq!(regions[1], vec![PassIndex(1)], "wave 1 = [P1]");
}

// =============================================================================
// SECTION 7 -- Partial RAW coverage at same depth
// =============================================================================

#[test]
fn partial_raw_coverage_produces_two_waves() {
    // Four passes at depth 0:
    //   P0 -> P2 (RAW)
    //   P1 -> P3 (RAW)
    // No edges between P0/P1 or P2/P3.
    //
    // Expected: wave 0 = [P0, P1], wave 1 = [P2, P3]
    let order = vec![PassIndex(0), PassIndex(1), PassIndex(2), PassIndex(3)];
    let depths = HashMap::from([
        (PassIndex(0), 0u32),
        (PassIndex(1), 0u32),
        (PassIndex(2), 0u32),
        (PassIndex(3), 0u32),
    ]);
    let edges = vec![
        IrEdge::new(PassIndex(0), PassIndex(2), ResourceHandle(1), EdgeType::RAW),
        IrEdge::new(PassIndex(1), PassIndex(3), ResourceHandle(2), EdgeType::RAW),
    ];
    let regions = identify_parallel_regions(&order, &depths, &edges);

    assert_eq!(regions.len(), 2, "two independent RAW edges => 2 waves");

    // Wave 0: P0 and P1 (no RAW predecessors).
    assert_eq!(regions[0].len(), 2, "wave 0 has 2 passes");
    assert!(regions[0].contains(&PassIndex(0)));
    assert!(regions[0].contains(&PassIndex(1)));

    // Wave 1: P2 and P3 (their RAW predecessors were in wave 0).
    assert_eq!(regions[1].len(), 2, "wave 1 has 2 passes");
    assert!(regions[1].contains(&PassIndex(2)));
    assert!(regions[1].contains(&PassIndex(3)));
}

#[test]
fn diamond_within_depth_group() {
    // At depth 0:
    //   P0 -> P1, P0 -> P2, P1 -> P3 (all RAW)
    //   P2 and P3 have no RAW with each other.
    //
    // Expected: wave 0 = [P0], wave 1 = [P1, P2], wave 2 = [P3]
    let order = vec![PassIndex(0), PassIndex(1), PassIndex(2), PassIndex(3)];
    let depths = HashMap::from([
        (PassIndex(0), 0u32),
        (PassIndex(1), 0u32),
        (PassIndex(2), 0u32),
        (PassIndex(3), 0u32),
    ]);
    let edges = vec![
        IrEdge::new(PassIndex(0), PassIndex(1), ResourceHandle(1), EdgeType::RAW),
        IrEdge::new(PassIndex(0), PassIndex(2), ResourceHandle(2), EdgeType::RAW),
        IrEdge::new(PassIndex(1), PassIndex(3), ResourceHandle(3), EdgeType::RAW),
    ];
    let regions = identify_parallel_regions(&order, &depths, &edges);

    assert_eq!(regions.len(), 3, "mini-diamond within depth group => 3 waves");
    assert_eq!(regions[0], vec![PassIndex(0)], "wave 0 = [P0]");

    // Wave 1 = [P1, P2] (both ready once P0 is gone).
    assert_eq!(regions[1].len(), 2, "wave 1 has 2 passes");
    assert!(regions[1].contains(&PassIndex(1)));
    assert!(regions[1].contains(&PassIndex(2)));

    assert_eq!(regions[2], vec![PassIndex(3)], "wave 2 = [P3]");
}

// =============================================================================
// SECTION 8 -- Mixed depths: parallel and serial segments
// =============================================================================

#[test]
fn mixed_depths_produces_correct_regions() {
    // A more complex graph with mixed parallelism:
    //
    //   depth 0: P0 (entry)
    //   depth 1: P1, P2 (parallel -- both read from P0)
    //   depth 2: P3, P4 (parallel -- P3 reads from P1, P4 reads from P2)
    //   depth 3: P5 (exit -- reads from P3 and P4)
    //
    // Regions: [P0], [P1, P2], [P3, P4], [P5]
    let p0 = make_pass(0, "entry", &[], &[ResourceHandle(1), ResourceHandle(2)]);

    let mut p1 = make_pass(1, "a", &[ResourceHandle(1)], &[ResourceHandle(3)]);
    let mut p2 = make_pass(2, "b", &[ResourceHandle(2)], &[ResourceHandle(4)]);
    let p3 = make_pass(3, "c", &[ResourceHandle(3)], &[ResourceHandle(5)]);
    let p4 = make_pass(4, "d", &[ResourceHandle(4)], &[ResourceHandle(6)]);
    let p5 = make_pass(5, "exit", &[ResourceHandle(5), ResourceHandle(6)], &[]);

    // P1 also reads from P0's resource 1. P2 also reads from P0's resource 2.
    p1.access_set.reads.push(ResourceHandle(1));
    p2.access_set.reads.push(ResourceHandle(2));

    let resources = vec![
        IrResource::new(ResourceHandle(1), "r1", storage_buf(64),
            ResourceLifetime::Transient, ResourceState::Uninitialized),
        IrResource::new(ResourceHandle(2), "r2", storage_buf(64),
            ResourceLifetime::Transient, ResourceState::Uninitialized),
        IrResource::new(ResourceHandle(3), "r3", storage_buf(64),
            ResourceLifetime::Transient, ResourceState::Uninitialized),
        IrResource::new(ResourceHandle(4), "r4", storage_buf(64),
            ResourceLifetime::Transient, ResourceState::Uninitialized),
        IrResource::new(ResourceHandle(5), "r5", storage_buf(64),
            ResourceLifetime::Transient, ResourceState::Uninitialized),
        IrResource::new(ResourceHandle(6), "r6", storage_buf(64),
            ResourceLifetime::Transient, ResourceState::Uninitialized),
    ];
    let regions = compute_regions(&[p0, p1, p2, p3, p4, p5], &resources);

    assert_eq!(regions.len(), 4, "mixed graph must produce 4 regions");

    // Region 0: entry alone.
    assert_eq!(regions[0], vec![PassIndex(0)], "region 0 = [P0]");

    // Region 1: P1 and P2 parallel.
    assert_eq!(regions[1].len(), 2, "region 1 = [P1, P2]");
    assert!(regions[1].contains(&PassIndex(1)));
    assert!(regions[1].contains(&PassIndex(2)));

    // Region 2: P3 and P4 parallel.
    assert_eq!(regions[2].len(), 2, "region 2 = [P3, P4]");
    assert!(regions[2].contains(&PassIndex(3)));
    assert!(regions[2].contains(&PassIndex(4)));

    // Region 3: exit alone.
    assert_eq!(regions[3], vec![PassIndex(5)], "region 3 = [P5]");
}

// =============================================================================
// SECTION 9 -- All passes at depth 0 with no edges
// =============================================================================

#[test]
fn all_passes_depth_zero_no_edges_one_region() {
    let order = vec![PassIndex(0), PassIndex(1), PassIndex(2), PassIndex(3)];
    let depths = HashMap::from([
        (PassIndex(0), 0u32),
        (PassIndex(1), 0u32),
        (PassIndex(2), 0u32),
        (PassIndex(3), 0u32),
    ]);
    let edges: Vec<IrEdge> = vec![];
    let regions = identify_parallel_regions(&order, &depths, &edges);

    assert_eq!(regions.len(), 1, "all at depth 0, no edges => 1 region");
    assert_eq!(regions[0].len(), 4, "region contains all 4 passes");
    for i in 0..4 {
        assert!(
            regions[0].contains(&PassIndex(i)),
            "region must contain P{}", i
        );
    }
}

// =============================================================================
// SECTION 10 -- Edge case: empty depth at intermediate level
// =============================================================================

#[test]
fn missing_depth_skips_gracefully() {
    // Depth 0: [P0], Depth 1: [], Depth 2: [P1]
    // The empty depth 1 must be skipped.
    let order = vec![PassIndex(0), PassIndex(1)];
    let depths = HashMap::from([
        (PassIndex(0), 0u32),
        (PassIndex(1), 2u32),
    ]);
    let edges: Vec<IrEdge> = vec![];
    let regions = identify_parallel_regions(&order, &depths, &edges);

    assert_eq!(
        regions.len(), 2,
        "empty depth 1 skipped => 2 regions"
    );
    assert_eq!(regions[0], vec![PassIndex(0)], "region 0 = [P0]");
    assert_eq!(regions[1], vec![PassIndex(1)], "region 1 = [P1]");
}

// =============================================================================
// SECTION 11 -- Return type structure
// =============================================================================

#[test]
fn regions_are_well_typed() {
    // Verify that the return type is Vec<Vec<PassIndex>> with no nesting
    // surprises.
    let p0 = make_pass(0, "a", &[], &[ResourceHandle(1)]);
    let p1 = make_pass(1, "b", &[ResourceHandle(1)], &[]);

    let resources = vec![IrResource::new(
        ResourceHandle(1), "x", storage_buf(64),
        ResourceLifetime::Transient, ResourceState::Uninitialized,
    )];
    let regions = compute_regions(&[p0, p1], &resources);

    // The result is flat: each element is a Vec<PassIndex>.
    assert_eq!(regions.len(), 2, "2 passes in chain => 2 regions");
    for region in &regions {
        // Each region is a Vec<PassIndex>.
        assert!(!region.is_empty(), "each region must be non-empty");
        for pass in region {
            let _ = pass.0; // Access the inner usize.
        }
    }
}

#[test]
fn regions_preserve_pass_indices() {
    // Verify that pass indices in the output match the input passes exactly.
    let p0 = make_pass(10, "ten", &[], &[ResourceHandle(1)]);
    let p1 = make_pass(20, "twenty", &[ResourceHandle(1)], &[]);

    let resources = vec![IrResource::new(
        ResourceHandle(1), "x", storage_buf(64),
        ResourceLifetime::Transient, ResourceState::Uninitialized,
    )];
    let regions = compute_regions(&[p0, p1], &resources);

    let flat: Vec<PassIndex> = regions.iter().flat_map(|r| r.iter().copied()).collect();
    assert!(
        flat.contains(&PassIndex(10)),
        "regions must contain pass index 10"
    );
    assert!(
        flat.contains(&PassIndex(20)),
        "regions must contain pass index 20"
    );
}

// =============================================================================
// SECTION 12 -- End-to-end via CompiledFrameGraph
// =============================================================================

#[test]
fn compiled_graph_parallel_regions_match_direct_call() {
    // Verify that CompiledFrameGraph::emit_schedule_bridge produces the same
    // parallel regions as calling identify_parallel_regions directly.
    use renderer_backend::frame_graph::CompiledFrameGraph;

    let entry = make_pass(0, "entry", &[], &[ResourceHandle(1), ResourceHandle(2)]);
    let mid_a = make_pass(1, "mid_a", &[ResourceHandle(1)], &[ResourceHandle(3)]);
    let mid_b = make_pass(2, "mid_b", &[ResourceHandle(2)], &[ResourceHandle(4)]);
    let exit_p = make_pass(3, "exit", &[ResourceHandle(3), ResourceHandle(4)], &[]);

    let passes = vec![entry, mid_a, mid_b, exit_p];
    let resources = vec![
        IrResource::new(ResourceHandle(1), "r1", storage_buf(64),
            ResourceLifetime::Transient, ResourceState::Uninitialized),
        IrResource::new(ResourceHandle(2), "r2", storage_buf(64),
            ResourceLifetime::Transient, ResourceState::Uninitialized),
        IrResource::new(ResourceHandle(3), "r3", storage_buf(64),
            ResourceLifetime::Transient, ResourceState::Uninitialized),
        IrResource::new(ResourceHandle(4), "r4", storage_buf(64),
            ResourceLifetime::Transient, ResourceState::Uninitialized),
    ];

    // Direct call.
    let direct_regions = compute_regions(&passes, &resources);

    // Via CompiledFrameGraph.
    let compiled = CompiledFrameGraph::compile(passes, resources)
        .expect("diamond must compile");
    let schedule = compiled.emit_schedule_bridge();
    let schedule_regions: Vec<Vec<PassIndex>> = schedule["parallel_regions"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| {
            r.as_array()
                .unwrap()
                .iter()
                .map(|v| PassIndex(v.as_u64().unwrap() as usize))
                .collect()
        })
        .collect();

    assert_eq!(
        direct_regions, schedule_regions,
        "direct identify_parallel_regions must match CompiledFrameGraph output"
    );
}

// =============================================================================
// SECTION 13 -- Resource-level end-to-end graphs
// =============================================================================

#[test]
fn three_way_fork_at_depth_0() {
    // P0 writes resource 1, P1 writes resource 2, P2 writes resource 3.
    // All independent, all at depth 0. Single region with all three.
    let p0 = make_pass(0, "a", &[], &[ResourceHandle(1)]);
    let p1 = make_pass(1, "b", &[], &[ResourceHandle(2)]);
    let p2 = make_pass(2, "c", &[], &[ResourceHandle(3)]);

    let resources = vec![
        IrResource::new(ResourceHandle(1), "r1", storage_buf(64),
            ResourceLifetime::Transient, ResourceState::Uninitialized),
        IrResource::new(ResourceHandle(2), "r2", storage_buf(64),
            ResourceLifetime::Transient, ResourceState::Uninitialized),
        IrResource::new(ResourceHandle(3), "r3", storage_buf(64),
            ResourceLifetime::Transient, ResourceState::Uninitialized),
    ];
    let regions = compute_regions(&[p0, p1, p2], &resources);

    assert_eq!(regions.len(), 1, "three independent passes => 1 region");
    assert_eq!(regions[0].len(), 3, "region contains all 3 passes");
}

#[test]
fn diamond_with_raw_serialisation_same_depth() {
    // Build a diamond but where the two middle passes have a RAW edge
    // between them (mid_a writes what mid_b reads). They are at the same
    // depth, so the RAW edge forces them into sub-groups.
    //
    // Pass 0: writes R1, R2 (entry)
    // Pass 1: reads R1, writes R3
    // Pass 2: reads R2, reads R3, writes R4  (RAW from P1 on R3)
    // Pass 3: reads R4
    //
    // P1 and P2 both sit at depth 1 -- but P2 depends on P1 via RAW on R3.
    let p0 = make_pass(0, "entry", &[], &[ResourceHandle(1), ResourceHandle(2)]);
    let p1 = make_pass(1, "first_mid", &[ResourceHandle(1)], &[ResourceHandle(3)]);
    let p2 = make_pass(2, "second_mid", &[ResourceHandle(2), ResourceHandle(3)], &[ResourceHandle(4)]);
    let p3 = make_pass(3, "exit", &[ResourceHandle(4)], &[]);

    let resources = vec![
        IrResource::new(ResourceHandle(1), "r1", storage_buf(64),
            ResourceLifetime::Transient, ResourceState::Uninitialized),
        IrResource::new(ResourceHandle(2), "r2", storage_buf(64),
            ResourceLifetime::Transient, ResourceState::Uninitialized),
        IrResource::new(ResourceHandle(3), "r3", storage_buf(64),
            ResourceLifetime::Transient, ResourceState::Uninitialized),
        IrResource::new(ResourceHandle(4), "r4", storage_buf(64),
            ResourceLifetime::Transient, ResourceState::Uninitialized),
    ];
    let regions = compute_regions(&[p0, p1, p2, p3], &resources);

    // Expected: [P0], [P1], [P2], [P3]
    // P1 and P2 have depths 1 and 2 respectively because of the RAW on R3
    // between P1 and P2 creates an extra depth level.
    assert_eq!(regions.len(), 4, "RAW within diamond => 4 regions");
}

// =============================================================================
// SECTION 14 -- All passes at same depth with mixed edge types
// =============================================================================

#[test]
fn same_depth_mixed_edges_produces_correct_waves() {
    // All at depth 0:
    //   P0 -RAW-> P2
    //   P1 -WAR-> P3
    //   P2 -RAW-> P3
    //
    // Wave 0: [P0, P1] (P0 has RAW pred none; P1 has no RAW pred)
    // Wave 1: [P2]      (P2 RAW pred P0 is gone; P3 still has RAW pred P2)
    // Wave 2: [P3]      (P3 RAW pred P2 is gone)
    let order = vec![PassIndex(0), PassIndex(1), PassIndex(2), PassIndex(3)];
    let depths = HashMap::from([
        (PassIndex(0), 0u32),
        (PassIndex(1), 0u32),
        (PassIndex(2), 0u32),
        (PassIndex(3), 0u32),
    ]);
    let edges = vec![
        IrEdge::new(PassIndex(0), PassIndex(2), ResourceHandle(1), EdgeType::RAW),
        IrEdge::new(PassIndex(1), PassIndex(3), ResourceHandle(2), EdgeType::WAR),
        IrEdge::new(PassIndex(2), PassIndex(3), ResourceHandle(3), EdgeType::RAW),
    ];
    let regions = identify_parallel_regions(&order, &depths, &edges);

    assert_eq!(regions.len(), 3, "mixed edges => 3 waves");
    assert_eq!(regions[0].len(), 2, "wave 0 = [P0, P1]");
    assert!(regions[0].contains(&PassIndex(0)));
    assert!(regions[0].contains(&PassIndex(1)));
    assert_eq!(regions[1], vec![PassIndex(2)], "wave 1 = [P2]");
    assert_eq!(regions[2], vec![PassIndex(3)], "wave 2 = [P3]");
}

// =============================================================================
// SECTION 15 -- Stability: same input, same output
// =============================================================================

#[test]
fn deterministic_output() {
    // Calling twice with the same inputs must produce the same result.
    let order = vec![PassIndex(0), PassIndex(1)];
    let depths = HashMap::from([(PassIndex(0), 0u32), (PassIndex(1), 0u32)]);
    let edges = vec![IrEdge::new(
        PassIndex(0),
        PassIndex(1),
        ResourceHandle(1),
        EdgeType::RAW,
    )];

    let r1 = identify_parallel_regions(&order, &depths, &edges);
    let r2 = identify_parallel_regions(&order, &depths, &edges);

    assert_eq!(r1, r2, "deterministic: same input => same output");
}
