// SPDX-License-Identifier: MIT
//
// blackbox_liveness.rs -- Blackbox contract tests for T-FG-6.1 transitive
// liveness analysis.
//
// CLEANROOM: No src/ access beyond the public API exported by the crate.
// Tests use only `renderer_backend::frame_graph::*` -- no internal fields,
// no private methods, no implementation details.
//
// Public API under test:
//   compute_transitive_liveness(passes: &[IrPass], edges: &[IrEdge])
//       -> HashSet<PassIndex>
//
// Contract:
//   Graphics passes are ALWAYS live (observable side effects on the
//   framebuffer/swapchain). Compute and Copy passes are live only if their
//   outputs are consumed by a live pass (directly or transitively). Only RAW
//   (read-after-write) edges contribute to the consumption chain; WAR and WAW
//   edges do NOT make a producer pass live.
//
// Coverage:
//   1.  Graphics pass alone is always live (no consumers)
//   2.  Compute pass whose output is consumed by a live pass is live
//   3.  Compute pass without any consumer is NOT live (dead)
//   4.  Copy pass without any consumer is NOT live (dead)
//   5.  Chain: P0 (compute) -> P1 (graphics) -- both live
//   6.  Chain: P0 (compute) -> P1 (compute) -> P2 (graphics) -- all three
//       transitively live
//   7.  Diamond: three compute producers converge on one graphics sink
//       -- all passes live
//   8.  Dead compute pass excluded from liveness set
//   9.  All graphics passes included regardless of consumers
//  10.  Dead middle segment: compute chain where last link is dead
//  11.  WAR and WAW edges do not confer liveness
//  12.  Multiple graphics passes all always live
//  13.  Copy pass whose output is consumed by a live pass is live
//  14.  Mixed set: some compute live (consumed), some dead (unconsumed)
//  15.  Empty input returns empty set

use renderer_backend::frame_graph::{
    compute_transitive_liveness, DispatchSource, EdgeType, InstanceSource, IrEdge,
    IrPass, PassIndex, ResourceHandle, ViewType,
};
use std::collections::HashSet;

// =============================================================================
// Helper: build a pass with a descriptive name
// =============================================================================

fn compute_pass(index: usize, name: &str) -> IrPass {
    IrPass::compute(
        PassIndex(index),
        name,
        DispatchSource::Direct {
            group_count_x: 1,
            group_count_y: 1,
            group_count_z: 1,
        },
        ViewType::Storage,
    )
}

fn graphics_pass(index: usize, name: &str) -> IrPass {
    IrPass::graphics(
        PassIndex(index),
        name,
        vec![],
        None,
        InstanceSource::Direct {
            index_count: 0,
            instance_count: 1,
            base_vertex: 0,
            first_index: 0,
            first_instance: 0,
        },
        ViewType::Texture2D,
    )
}

fn copy_pass(index: usize, name: &str) -> IrPass {
    IrPass::copy(PassIndex(index), name)
}

fn raw_edge(from: usize, to: usize, resource: u32) -> IrEdge {
    IrEdge::new(
        PassIndex(from),
        PassIndex(to),
        ResourceHandle(resource),
        EdgeType::RAW,
    )
}

fn war_edge(from: usize, to: usize, resource: u32) -> IrEdge {
    IrEdge::new(
        PassIndex(from),
        PassIndex(to),
        ResourceHandle(resource),
        EdgeType::WAR,
    )
}

fn waw_edge(from: usize, to: usize, resource: u32) -> IrEdge {
    IrEdge::new(
        PassIndex(from),
        PassIndex(to),
        ResourceHandle(resource),
        EdgeType::WAW,
    )
}

fn set_from(indices: &[usize]) -> HashSet<PassIndex> {
    indices.iter().map(|&i| PassIndex(i)).collect()
}

// =============================================================================
// SECTION 1 -- Graphics pass alone is always live (no consumers)
// =============================================================================

#[test]
fn graphics_pass_alone_is_live() {
    let passes = vec![graphics_pass(0, "gbuffer")];
    let edges = vec![];
    let live = compute_transitive_liveness(&passes, &edges);

    assert!(
        live.contains(&PassIndex(0)),
        "Graphics pass must be live even with no consumers"
    );
    assert_eq!(live.len(), 1, "Only the graphics pass is live");
}

// =============================================================================
// SECTION 2 -- Compute pass with consumer is live
// =============================================================================

#[test]
fn compute_pass_with_consumer_is_live() {
    // P0 (compute, writes R1), P1 (graphics, reads R1 as attachment).
    let passes = vec![compute_pass(0, "compute_a"), graphics_pass(1, "gfx_b")];
    let edges = vec![raw_edge(0, 1, 1)];

    let live = compute_transitive_liveness(&passes, &edges);

    // P1 is graphics -> always live.  P0's output is consumed by live P1 -> live.
    assert_eq!(live, set_from(&[0, 1]));
}

// =============================================================================
// SECTION 3 -- Compute pass without consumer is NOT live (dead)
// =============================================================================

#[test]
fn compute_pass_without_consumer_is_dead() {
    // P0 (compute, writes R1), nobody reads R1.
    let passes = vec![compute_pass(0, "orphan_compute")];
    let edges = vec![];

    let live = compute_transitive_liveness(&passes, &edges);

    assert!(
        !live.contains(&PassIndex(0)),
        "Compute pass with no consumer must be dead"
    );
    assert!(live.is_empty(), "No live passes expected");
}

// =============================================================================
// SECTION 4 -- Copy pass without consumer is NOT live (dead)
// =============================================================================

#[test]
fn copy_pass_without_consumer_is_dead() {
    // P0 (copy, writes R1), nobody reads R1.
    let passes = vec![copy_pass(0, "orphan_copy")];
    let edges = vec![];

    let live = compute_transitive_liveness(&passes, &edges);

    assert!(
        !live.contains(&PassIndex(0)),
        "Copy pass with no consumer must be dead"
    );
    assert!(live.is_empty(), "No live passes expected");
}

// =============================================================================
// SECTION 5 -- Chain: P0 (compute) -> P1 (graphics) -- both live
// =============================================================================

#[test]
fn chain_compute_to_graphics_both_live() {
    // P0 (compute, writes R1), P1 (graphics, reads R1).
    let passes = vec![compute_pass(0, "compute_depth"), graphics_pass(1, "render")];
    let edges = vec![raw_edge(0, 1, 1)];

    let live = compute_transitive_liveness(&passes, &edges);

    assert_eq!(live, set_from(&[0, 1]));
}

// =============================================================================
// SECTION 6 -- Transitive chain through compute to graphics
// =============================================================================

#[test]
fn transitive_chain_all_live() {
    // P0 (compute, writes R1) -> P1 (compute, reads R1, writes R2)
    //   -> P2 (graphics, reads R2)
    let passes = vec![
        compute_pass(0, "cs_a"),
        compute_pass(1, "cs_b"),
        graphics_pass(2, "gfx_sink"),
    ];
    let edges = vec![raw_edge(0, 1, 1), raw_edge(1, 2, 2)];

    let live = compute_transitive_liveness(&passes, &edges);

    // P2 is graphics -> live.  P1 consumed by live P2 -> live.
    // P0 consumed by live P1 -> live.
    assert_eq!(live, set_from(&[0, 1, 2]));
}

#[test]
fn transitive_chain_five_deep() {
    // P0 -> P1 -> P2 -> P3 -> P4 (graphics)
    let mut passes = Vec::new();
    for i in 0..4 {
        passes.push(compute_pass(i, &format!("cs_{}", i)));
    }
    passes.push(graphics_pass(4, "gfx_sink"));

    let mut edges = Vec::new();
    for i in 0..4 {
        edges.push(raw_edge(i, i + 1, (i + 1) as u32));
    }

    let live = compute_transitive_liveness(&passes, &edges);

    assert_eq!(live, set_from(&[0, 1, 2, 3, 4]));
}

// =============================================================================
// SECTION 7 -- Diamond: all passes live when final pass is graphics
// =============================================================================

#[test]
fn diamond_all_live_when_sink_is_graphics() {
    // P0 (compute), P1 (compute), P2 (compute) all write to P3 (graphics).
    let passes = vec![
        compute_pass(0, "cs_left"),
        compute_pass(1, "cs_mid"),
        compute_pass(2, "cs_right"),
        graphics_pass(3, "gfx_merge"),
    ];
    let edges = vec![
        raw_edge(0, 3, 1),
        raw_edge(1, 3, 2),
        raw_edge(2, 3, 3),
    ];

    let live = compute_transitive_liveness(&passes, &edges);

    assert_eq!(live, set_from(&[0, 1, 2, 3]));
}

#[test]
fn diamond_with_transitive_leg() {
    // P0 (compute) writes R1; P1 (compute) reads R1, writes R2;
    // P2 (compute) writes R3; P3 (graphics) reads R2 and R3.
    let passes = vec![
        compute_pass(0, "cs_producer"),
        compute_pass(1, "cs_transit"),
        compute_pass(2, "cs_other"),
        graphics_pass(3, "gfx_merge"),
    ];
    let edges = vec![
        raw_edge(0, 1, 1),
        raw_edge(1, 3, 2),
        raw_edge(2, 3, 3),
    ];

    let live = compute_transitive_liveness(&passes, &edges);

    // P3 is graphics -> live.  P1 consumed by live P3 -> live.
    // P0 consumed by live P1 -> live.  P2 consumed by live P3 -> live.
    assert_eq!(live, set_from(&[0, 1, 2, 3]));
}

// =============================================================================
// SECTION 8 -- Dead compute pass excluded from liveness set
// =============================================================================

#[test]
fn dead_compute_pass_excluded() {
    // P0 (compute, writes R1, nobody reads it), P1 (graphics).
    let passes = vec![compute_pass(0, "dead_compute"), graphics_pass(1, "gfx_pass")];
    let edges = vec![];

    let live = compute_transitive_liveness(&passes, &edges);

    assert!(
        !live.contains(&PassIndex(0)),
        "Compute pass with no consumer must be excluded"
    );
    assert!(
        live.contains(&PassIndex(1)),
        "Graphics pass must still be live"
    );
    assert_eq!(live.len(), 1, "Only the graphics pass is live");
}

// =============================================================================
// SECTION 9 -- All graphics passes included regardless of consumers
// =============================================================================

#[test]
fn multiple_graphics_passes_all_included() {
    let passes = vec![
        graphics_pass(0, "gfx_a"),
        graphics_pass(1, "gfx_b"),
        graphics_pass(2, "gfx_c"),
    ];
    let edges = vec![];

    let live = compute_transitive_liveness(&passes, &edges);

    assert_eq!(live.len(), 3, "All three graphics passes must be live");
    for i in 0..3 {
        assert!(live.contains(&PassIndex(i)));
    }
}

#[test]
fn graphics_pass_with_unrelated_compute_pass() {
    // P0 (graphics), P1 (compute, writes R1, nobody reads it).
    // P0 is live (graphics).  P1 is dead (compute, no consumer).
    let passes = vec![graphics_pass(0, "gfx_pass"), compute_pass(1, "unrelated_cs")];
    let edges = vec![];

    let live = compute_transitive_liveness(&passes, &edges);

    assert_eq!(live, set_from(&[0]));
}

// =============================================================================
// SECTION 10 -- Dead middle segment
// =============================================================================

#[test]
fn dead_middle_segment() {
    // P0 (compute, writes R1) -> P1 (compute, reads R1, writes R2)
    //   -> P2 (compute, reads R2, but nobody reads P2's output)
    // P2 is not graphics and its output is unconsumed -> dead.
    // P1 is consumed by P2 (dead) -> dead.
    // P0 is consumed by P1 (dead) -> dead.
    let passes = vec![
        compute_pass(0, "cs_a"),
        compute_pass(1, "cs_b"),
        compute_pass(2, "cs_c"),
    ];
    let edges = vec![raw_edge(0, 1, 1), raw_edge(1, 2, 2)];

    let live = compute_transitive_liveness(&passes, &edges);

    assert!(live.is_empty(), "No graphics pass in chain; all should be dead");
}

// =============================================================================
// SECTION 11 -- WAR and WAW edges do not confer liveness
// =============================================================================

#[test]
fn war_edge_does_not_make_producer_live() {
    // P0 (compute, reads R1), P1 (graphics, writes R1).
    // WAR edge: P0 reads, P1 writes.  Edge: P0 -> P1 with EdgeType::WAR.
    // P1 is graphics -> live.
    // But only RAW edges are considered for consumption.  The WAR edge from
    // P0 to P1 does NOT convey consumption, so P0 stays dead.
    let passes = vec![compute_pass(0, "cs_reader"), graphics_pass(1, "gfx_writer")];
    let edges = vec![war_edge(0, 1, 1)];

    let live = compute_transitive_liveness(&passes, &edges);

    assert_eq!(live, set_from(&[1]));
}

#[test]
fn waw_edge_does_not_make_producer_live() {
    // P0 (compute, writes R1), P1 (graphics, writes R1).
    // WAW edge: P0 writes, P1 writes.  Edge: P0 -> P1 with EdgeType::WAW.
    // P1 is graphics -> live.
    // WAW is not RAW, so P0 is NOT made live by P1 consuming its output.
    let passes = vec![compute_pass(0, "cs_writer"), graphics_pass(1, "gfx_writer")];
    let edges = vec![waw_edge(0, 1, 1)];

    let live = compute_transitive_liveness(&passes, &edges);

    assert_eq!(live, set_from(&[1]));
}

// =============================================================================
// SECTION 12 -- Multiple graphics passes all always live
// =============================================================================

#[test]
fn ten_graphics_passes_all_live() {
    let passes: Vec<IrPass> = (0..10)
        .map(|i| graphics_pass(i, &format!("gfx_{}", i)))
        .collect();
    let edges = vec![];

    let live = compute_transitive_liveness(&passes, &edges);

    assert_eq!(live.len(), 10, "All 10 graphics passes must be live");
}

// =============================================================================
// SECTION 13 -- Copy pass with consumer is live
// =============================================================================

#[test]
fn copy_pass_with_consumer_is_live() {
    // P0 (copy, writes R1), P1 (graphics, reads R1).
    let passes = vec![copy_pass(0, "copy_op"), graphics_pass(1, "gfx_user")];
    let edges = vec![raw_edge(0, 1, 1)];

    let live = compute_transitive_liveness(&passes, &edges);

    assert_eq!(live, set_from(&[0, 1]));
}

#[test]
fn copy_pass_in_transitive_chain() {
    // P0 (compute, writes R1) -> P1 (copy, reads R1, writes R2)
    //   -> P2 (graphics, reads R2)
    let passes = vec![
        compute_pass(0, "cs_writer"),
        copy_pass(1, "copy_stage"),
        graphics_pass(2, "gfx_sink"),
    ];
    let edges = vec![raw_edge(0, 1, 1), raw_edge(1, 2, 2)];

    let live = compute_transitive_liveness(&passes, &edges);

    assert_eq!(live, set_from(&[0, 1, 2]));
}

// =============================================================================
// SECTION 14 -- Mixed set: some compute live, some dead
// =============================================================================

#[test]
fn mixed_live_and_dead_compute_passes() {
    // P0 (compute, writes R1) -- consumed by P2, should be live.
    // P1 (compute, writes R2) -- NOT consumed by anyone, should be dead.
    // P2 (graphics, reads R1) -- graphics, always live.
    // P3 (compute, writes R3) -- NOT consumed, should be dead.
    // P4 (graphics) -- graphics, always live (no edges to it).
    let passes = vec![
        compute_pass(0, "live_cs"),
        compute_pass(1, "dead_cs_a"),
        graphics_pass(2, "gfx_mid"),
        compute_pass(3, "dead_cs_b"),
        graphics_pass(4, "gfx_tail"),
    ];
    let edges = vec![raw_edge(0, 2, 1)];

    let live = compute_transitive_liveness(&passes, &edges);

    // P0's output consumed by P2 (live graphics) -> P0 is live.
    // P2 is graphics -> live.
    // P4 is graphics -> live.
    // P1 and P3 are compute with no consumers -> dead.
    assert_eq!(live, set_from(&[0, 2, 4]));
}

// =============================================================================
// SECTION 15 -- Empty input returns empty set
// =============================================================================

#[test]
fn empty_input_returns_empty_set() {
    let passes = vec![];
    let edges = vec![];
    let live = compute_transitive_liveness(&passes, &edges);

    assert!(live.is_empty(), "Empty input must produce empty liveness set");
}

// =============================================================================
// SECTION 16 -- WAR edge does not confer liveness (gray-box)
// =============================================================================

#[test]
fn war_edge_alone_does_not_seed_or_propagate() {
    // Compute pass reads R1 (the "R" in WAR). Graphics pass writes R1 (the "W").
    // A WAR edge from P0 (read) to P1 (write) is created, but it does NOT
    // make P0 live (only RAW edges propagate liveness).
    let passes = vec![compute_pass(0, "cs_reader"), graphics_pass(1, "gfx_writer")];
    let edges = vec![war_edge(0, 1, 1)];

    let live = compute_transitive_liveness(&passes, &edges);

    assert_eq!(live, set_from(&[1]), "only graphics pass should be live");
}

// =============================================================================
// SECTION 17 -- WAW edge does not confer liveness (gray-box)
// =============================================================================

#[test]
fn waw_edge_alone_does_not_seed_or_propagate() {
    // Compute pass writes R1 (the first "W" in WAW). Graphics pass also writes R1.
    // A WAW edge from P0 to P1 is created, but it does NOT make P0 live.
    let passes = vec![compute_pass(0, "cs_first"), graphics_pass(1, "gfx_second")];
    let edges = vec![waw_edge(0, 1, 1)];

    let live = compute_transitive_liveness(&passes, &edges);

    assert_eq!(live, set_from(&[1]), "only graphics pass should be live");
}

// =============================================================================
// SECTION 18 -- Mixed RAW and WAR: only RAW propagates (gray-box)
// =============================================================================

#[test]
fn mixed_raw_and_war_only_raw_propagates() {
    // P0 (compute) writes R1 -- consumed via RAW by P2 (graphics).
    // P1 (compute) reads R2 -- WAR edge to P2.
    // P2 (graphics) reads R1, writes R2.
    // P0 is live (RAW consumer is graphics). P1 is dead (WAR does not propagate).
    let passes = vec![
        compute_pass(0, "cs_producer"),
        compute_pass(1, "cs_war_reader"),
        graphics_pass(2, "gfx_sink"),
    ];
    let edges = vec![
        raw_edge(0, 2, 1),
        war_edge(1, 2, 2),
    ];

    let live = compute_transitive_liveness(&passes, &edges);

    assert_eq!(live, set_from(&[0, 2]), "P1 must be dead (WAR does not propagate)");
}

// =============================================================================
// SECTION 19 -- Graphics-to-compute RAW does not make compute live (gray-box)
// =============================================================================

#[test]
fn graphics_producer_does_not_make_consumer_live() {
    // A RAW edge where the PRODUCER is graphics and the CONSUMER is compute.
    // Liveness propagates backward (consumer -> producer). Since the consumer
    // (compute) has no live consumer of its own, it should NOT be live.
    let passes = vec![graphics_pass(0, "gfx_writer"), compute_pass(1, "cs_reader")];
    let edges = vec![raw_edge(0, 1, 1)];

    let live = compute_transitive_liveness(&passes, &edges);

    assert_eq!(live, set_from(&[0]), "only graphics pass should be live");
}

// =============================================================================
// SECTION 20 -- Multiple independent graphics seeds (gray-box)
// =============================================================================

#[test]
fn multiple_independent_graphics_seeds() {
    // Two independent compute -> graphics chains. Both graphics seeds should
    // propagate liveness to their respective upstream compute passes.
    let passes = vec![
        compute_pass(0, "chain0_cs"),
        graphics_pass(1, "chain0_gfx"),
        compute_pass(2, "chain1_cs"),
        graphics_pass(3, "chain1_gfx"),
    ];
    let edges = vec![raw_edge(0, 1, 1), raw_edge(2, 3, 2)];

    let live = compute_transitive_liveness(&passes, &edges);

    assert_eq!(live, set_from(&[0, 1, 2, 3]), "all four passes must be live");
}

// =============================================================================
// SECTION 21 -- Large chain: 50 passes (gray-box)
// =============================================================================

#[test]
fn large_chain_fifty_passes_all_live() {
    let mut passes = Vec::new();
    for i in 0..49 {
        passes.push(compute_pass(i, &format!("cs_{}", i)));
    }
    passes.push(graphics_pass(49, "gfx_sink"));

    let mut edges = Vec::new();
    for i in 0..49 {
        edges.push(raw_edge(i, i + 1, (i + 1) as u32));
    }

    let live = compute_transitive_liveness(&passes, &edges);

    assert_eq!(live.len(), 50, "all 50 passes must be live");
    for i in 0..=49 {
        assert!(live.contains(&PassIndex(i)), "P{} must be live", i);
    }
}

// =============================================================================
// SECTION 22 -- Large: 50 all-graphics, 50 all-compute (gray-box)
// =============================================================================

#[test]
fn fifty_all_graphics_all_live() {
    let passes: Vec<IrPass> = (0..50)
        .map(|i| graphics_pass(i, &format!("gfx_{}", i)))
        .collect();
    let live = compute_transitive_liveness(&passes, &[]);
    assert_eq!(live.len(), 50, "all 50 graphics passes must be live");
}

#[test]
fn fifty_all_compute_no_graphics_all_dead() {
    let passes: Vec<IrPass> = (0..50)
        .map(|i| compute_pass(i, &format!("cs_{}", i)))
        .collect();
    let live = compute_transitive_liveness(&passes, &[]);
    assert!(live.is_empty(), "no graphics pass means all compute passes are dead");
}

// =============================================================================
// SECTION 23 -- Complex interleaved pattern (gray-box)
// =============================================================================

#[test]
fn complex_interleaved_liveness() {
    // Chain 1: P0 (cs) -> P1 (cs) -> P2 (gfx)           -- all live
    // Chain 2: P3 (cs) -> P4 (cs)                        -- all dead (no graphics)
    // Chain 3: P5 (cs) -> P6 (gfx)                       -- all live
    // P7 (cs) orphan                                      -- dead
    // P8 (gfx) standalone                                 -- live
    let passes = vec![
        compute_pass(0,  "c1_a"),
        compute_pass(1,  "c1_b"),
        graphics_pass(2, "c1_sink"),
        compute_pass(3,  "c2_a"),
        compute_pass(4,  "c2_b"),
        compute_pass(5,  "c3_a"),
        graphics_pass(6, "c3_sink"),
        compute_pass(7,  "orphan"),
        graphics_pass(8, "loner"),
    ];
    let edges = vec![
        raw_edge(0, 1, 1),
        raw_edge(1, 2, 2),
        raw_edge(3, 4, 3),
        raw_edge(5, 6, 4),
    ];

    let live = compute_transitive_liveness(&passes, &edges);

    assert_eq!(live, set_from(&[0, 1, 2, 5, 6, 8]));
}

// =============================================================================
// SECTION 24 -- Self-loop compute does not confer liveness (gray-box)
// =============================================================================

#[test]
fn self_loop_compute_not_live() {
    // A compute pass with a RAW self-loop (reads its own output).
    // Self-consumption should not make a pass live.
    let passes = vec![compute_pass(0, "self_loop")];
    let edges = vec![raw_edge(0, 0, 1)];

    let live = compute_transitive_liveness(&passes, &edges);

    assert!(live.is_empty(), "self-loop compute pass must remain dead");
}

// =============================================================================
// SECTION 25 -- Single copy pass dead (no consumer) (gray-box)
// =============================================================================

#[test]
fn single_copy_pass_dead() {
    let passes = vec![copy_pass(0, "orphan_copy")];
    let live = compute_transitive_liveness(&passes, &[]);
    assert!(live.is_empty(), "copy pass without consumer must be dead");
}

// =============================================================================
// SECTION 26 -- Mixed types no edges (gray-box)
// =============================================================================

#[test]
fn mixed_types_no_edges_live_set() {
    // Graphics passes are live; compute and copy passes with no consumers are not.
    let passes = vec![
        graphics_pass(0, "gfx_a"),
        compute_pass(1, "cs_orphan"),
        copy_pass(2, "copy_orphan"),
        graphics_pass(3, "gfx_b"),
    ];
    let live = compute_transitive_liveness(&passes, &[]);
    assert_eq!(live, set_from(&[0, 3]), "only graphics passes should be live");
}

// =============================================================================
// SECTION 27 -- Five-deep compute chain ends in graphics (gray-box)
// =============================================================================

#[test]
fn five_deep_compute_chain_to_graphics() {
    // P0 -> P1 -> P2 -> P3 -> P4 -> P5 (graphics)
    let passes = vec![
        compute_pass(0, "cs_0"),
        compute_pass(1, "cs_1"),
        compute_pass(2, "cs_2"),
        compute_pass(3, "cs_3"),
        compute_pass(4, "cs_4"),
        graphics_pass(5, "gfx_sink"),
    ];
    let edges = vec![
        raw_edge(0, 1, 1),
        raw_edge(1, 2, 2),
        raw_edge(2, 3, 3),
        raw_edge(3, 4, 4),
        raw_edge(4, 5, 5),
    ];
    let live = compute_transitive_liveness(&passes, &edges);
    assert_eq!(live, set_from(&[0, 1, 2, 3, 4, 5]));
}
