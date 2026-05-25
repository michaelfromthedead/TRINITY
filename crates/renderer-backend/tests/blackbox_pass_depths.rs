// SPDX-License-Identifier: MIT
//
// blackbox_pass_depths.rs -- Blackbox contract tests for T-FG-2.4
// compute_pass_depths.
//
// CLEANROOM: No src/ access beyond the public API exported by the crate.
// Tests use only the public types and functions exported by
// `renderer_backend::frame_graph` -- no internal fields, no private methods.
//
// Public API under test:
//   compute_pass_depths(order: &[PassIndex], edges: &[IrEdge])
//       -> HashMap<PassIndex, u32>
//
// Contract (from doc comment on compute_pass_depths):
//   For each pass in topological order:
//   - Entry passes (no predecessors in the edge set) get depth 0.
//   - All other passes get max(predecessor depths) + 1.
//
// Depth uses the longest-path algorithm on the dependency DAG. All edge types
// (RAW, WAR, WAW) contribute to the predecessor set -- any edge type creates an
// ordering dependency that increases depth.
//
// Coverage:
//   1.  Empty order, no edges -> empty depth map
//   2.  Single pass, no edges -> depth 0
//   3.  Two independent passes, no edges -> both depth 0
//   4.  Three independent passes, no edges -> all depth 0
//   5.  Two-pass chain -> depths [0, 1]
//   6.  Linear chain of 4 -> depths [0, 1, 2, 3]
//   7.  Linear chain of 5 -> depths [0, 1, 2, 3, 4]
//   8.  Diamond: P0 feeds P1 and P2, both feed P3 -> P0=0, P1=1, P2=1, P3=2
//   9.  Diamond with asymmetric legs: left leg longer than right -> merge
//       point takes longest-path depth
//  10.  Wide diamond: three middle passes at depth 1, merge at depth 2
//  11.  Fork: one source feeds two children -> source 0, children 1
//  12.  Two independent entries converge on one sink -> entries 0, sink 1
//  13.  Three independent chains of different lengths (2, 3, 1) -> each chain
//       receives correct depths independently
//  14.  All edge types (RAW, WAR, WAW) contribute to depth
//  15.  Order sensitivity: passes earlier in order get correct depths even
//       when a predecessor appears later in the order (should not happen with
//       valid topological order but tests robustness)

use renderer_backend::frame_graph::{
    compute_pass_depths, EdgeType, IrEdge, PassIndex, ResourceHandle,
};
use std::collections::HashMap;

// =============================================================================
// Helpers
// =============================================================================

/// Shorthand for a RAW edge between two passes.
fn raw(from: usize, to: usize, resource: u32) -> IrEdge {
    IrEdge::new(
        PassIndex(from),
        PassIndex(to),
        ResourceHandle(resource),
        EdgeType::RAW,
    )
}

/// Shorthand for a WAR edge between two passes.
fn war(from: usize, to: usize, resource: u32) -> IrEdge {
    IrEdge::new(
        PassIndex(from),
        PassIndex(to),
        ResourceHandle(resource),
        EdgeType::WAR,
    )
}

/// Shorthand for a WAW edge between two passes.
fn waw(from: usize, to: usize, resource: u32) -> IrEdge {
    IrEdge::new(
        PassIndex(from),
        PassIndex(to),
        ResourceHandle(resource),
        EdgeType::WAW,
    )
}

/// Assert that a depth map has the expected (index -> depth) entries and no
/// extra entries.
fn assert_depths(depths: &HashMap<PassIndex, u32>, expected: &[(usize, u32)]) {
    assert_eq!(
        depths.len(),
        expected.len(),
        "Depth map size mismatch: expected {} entries, got {}",
        expected.len(),
        depths.len(),
    );
    for &(idx, exp_depth) in expected {
        let key = PassIndex(idx);
        let actual = depths.get(&key);
        assert_eq!(
            actual,
            Some(&exp_depth),
            "PassIndex({}) depth mismatch: expected {}, got {:?}",
            idx,
            exp_depth,
            actual,
        );
    }
}

// =============================================================================
// SECTION 1 -- Empty and edge cases
// =============================================================================

/// Empty order with no edges produces an empty depth map.
#[test]
fn empty_input_produces_empty_map() {
    let order: Vec<PassIndex> = vec![];
    let edges: Vec<IrEdge> = vec![];

    let depths = compute_pass_depths(&order, &edges);

    assert!(
        depths.is_empty(),
        "Empty input must produce an empty depth map, got {} entries",
        depths.len(),
    );
}

/// A single pass with no edges receives depth 0.
#[test]
fn single_pass_depth_zero() {
    let order = vec![PassIndex(0)];
    let edges = vec![];

    let depths = compute_pass_depths(&order, &edges);

    assert_depths(&depths, &[(0, 0)]);
}

/// Two independent passes with no edges both receive depth 0.
#[test]
fn two_independent_passes_both_depth_zero() {
    let order = vec![PassIndex(0), PassIndex(1)];
    let edges = vec![];

    let depths = compute_pass_depths(&order, &edges);

    assert_depths(&depths, &[(0, 0), (1, 0)]);
}

/// Three independent passes with no edges all receive depth 0.
#[test]
fn three_independent_passes_all_depth_zero() {
    let order = vec![PassIndex(0), PassIndex(1), PassIndex(2)];
    let edges = vec![];

    let depths = compute_pass_depths(&order, &edges);

    assert_depths(&depths, &[(0, 0), (1, 0), (2, 0)]);
}

// =============================================================================
// SECTION 2 -- Linear chains
// =============================================================================

/// A two-pass chain: P0 -> P1. Depths: P0=0, P1=1.
#[test]
fn two_pass_chain() {
    let order = vec![PassIndex(0), PassIndex(1)];
    let edges = vec![raw(0, 1, 1)];

    let depths = compute_pass_depths(&order, &edges);

    assert_depths(&depths, &[(0, 0), (1, 1)]);
}

/// A four-pass chain: P0 -> P1 -> P2 -> P3. Depths: [0, 1, 2, 3].
#[test]
fn chain_of_four() {
    let order = vec![PassIndex(0), PassIndex(1), PassIndex(2), PassIndex(3)];
    let edges = vec![raw(0, 1, 1), raw(1, 2, 2), raw(2, 3, 3)];

    let depths = compute_pass_depths(&order, &edges);

    assert_depths(&depths, &[(0, 0), (1, 1), (2, 2), (3, 3)]);
}

/// A five-pass chain: P0 -> P1 -> P2 -> P3 -> P4. Depths: [0, 1, 2, 3, 4].
#[test]
fn chain_of_five() {
    let order = vec![
        PassIndex(0),
        PassIndex(1),
        PassIndex(2),
        PassIndex(3),
        PassIndex(4),
    ];
    let edges = vec![raw(0, 1, 1), raw(1, 2, 2), raw(2, 3, 3), raw(3, 4, 4)];

    let depths = compute_pass_depths(&order, &edges);

    assert_depths(&depths, &[(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]);
}

// =============================================================================
// SECTION 3 -- Diamond and fan-in / fan-out topologies
// =============================================================================

/// Diamond: P0 feeds P1 and P2; P1 and P2 both feed P3.
///     P1
///    /   \
///  P0     P3
///    \   /
///     P2
/// Depths: P0=0, P1=1, P2=1, P3=2
#[test]
fn diamond_depths() {
    let order = vec![
        PassIndex(0),
        PassIndex(1),
        PassIndex(2),
        PassIndex(3),
    ];
    let edges = vec![raw(0, 1, 1), raw(0, 2, 2), raw(1, 3, 3), raw(2, 3, 4)];

    let depths = compute_pass_depths(&order, &edges);

    assert_depths(&depths, &[(0, 0), (1, 1), (2, 1), (3, 2)]);
}

/// Diamond with asymmetric legs: the left leg is longer than the right leg.
/// The merge point must take the longest-path depth.
///
///     P0 -> P1 -> P2
///      \         /
///       P3 ----/
///
/// Edges: P0->P1, P1->P2, P0->P3, P2->P4, P3->P4
/// Depths: P0=0, P1=1, P3=1, P2=2, P4=max(depth(P2), depth(P3)) + 1 = 3
///
/// Left-leg path length: P0->P1->P2 = depth 2 at P2
/// Right-leg path length: P0->P3 = depth 1 at P3
/// P4 merge takes max(2, 1) + 1 = 3
#[test]
fn diamond_asymmetric_legs_longest_path_wins() {
    let order = vec![
        PassIndex(0),
        PassIndex(1),
        PassIndex(3),
        PassIndex(2),
        PassIndex(4),
    ];
    let edges = vec![
        raw(0, 1, 1),
        raw(1, 2, 2),
        raw(0, 3, 3),
        raw(2, 4, 4),
        raw(3, 4, 5),
    ];

    let depths = compute_pass_depths(&order, &edges);

    assert_depths(&depths, &[(0, 0), (1, 1), (3, 1), (2, 2), (4, 3)]);
}

/// Wide diamond: three middle passes all at depth 1, merging to a single sink.
///
///     P1
///    / | \
///  P0  P2  P4
///    \ | /
///     P3
///
/// Edges: P0->P1, P0->P2, P0->P3, P1->P4, P2->P4, P3->P4
/// Depths: P0=0, P1=1, P2=1, P3=1, P4=2
#[test]
fn wide_diamond_depths() {
    let order = vec![
        PassIndex(0),
        PassIndex(1),
        PassIndex(2),
        PassIndex(3),
        PassIndex(4),
    ];
    let edges = vec![
        raw(0, 1, 1),
        raw(0, 2, 2),
        raw(0, 3, 3),
        raw(1, 4, 4),
        raw(2, 4, 5),
        raw(3, 4, 6),
    ];

    let depths = compute_pass_depths(&order, &edges);

    assert_depths(&depths, &[(0, 0), (1, 1), (2, 1), (3, 1), (4, 2)]);
}

// =============================================================================
// SECTION 4 -- Fork patterns
// =============================================================================

/// Fork: one source pass feeds two children.
/// P0 -> P1, P0 -> P2
/// Depths: P0=0, P1=1, P2=1
#[test]
fn fork_one_to_two() {
    let order = vec![PassIndex(0), PassIndex(1), PassIndex(2)];
    let edges = vec![raw(0, 1, 1), raw(0, 2, 2)];

    let depths = compute_pass_depths(&order, &edges);

    assert_depths(&depths, &[(0, 0), (1, 1), (2, 1)]);
}

/// Two independent entry passes converge on one sink.
/// P0 -> P2, P1 -> P2
/// Depths: P0=0, P1=0, P2=1
#[test]
fn independent_entries_converge() {
    let order = vec![PassIndex(0), PassIndex(1), PassIndex(2)];
    let edges = vec![raw(0, 2, 1), raw(1, 2, 2)];

    let depths = compute_pass_depths(&order, &edges);

    assert_depths(&depths, &[(0, 0), (1, 0), (2, 1)]);
}

// =============================================================================
// SECTION 5 -- Multiple independent chains
// =============================================================================

/// Three independent chains of different lengths in the same graph.
///
/// Chain A: P0 -> P1          (length 2)
/// Chain B: P2 -> P3 -> P4    (length 3)
/// Chain C: P5                 (length 1, no edges)
///
/// Depths:
///   Chain A: P0=0, P1=1
///   Chain B: P2=0, P3=1, P4=2
///   Chain C: P5=0
#[test]
fn multiple_independent_chains() {
    let order = vec![
        PassIndex(0),
        PassIndex(1),
        PassIndex(2),
        PassIndex(3),
        PassIndex(4),
        PassIndex(5),
    ];
    let edges = vec![
        raw(0, 1, 1),
        raw(2, 3, 2),
        raw(3, 4, 3),
    ];

    let depths = compute_pass_depths(&order, &edges);

    assert_depths(
        &depths,
        &[(0, 0), (1, 1), (2, 0), (3, 1), (4, 2), (5, 0)],
    );
}

// =============================================================================
// SECTION 6 -- All edge types contribute to depth
// =============================================================================

/// All three edge types (RAW, WAR, WAW) must contribute to depth assignment.
/// The function treats every edge as a dependency -- edge type is irrelevant.
///
/// P0 ->[RAW] P1 ->[WAR] P2 ->[WAW] P3
///
/// Depths: P0=0, P1=1, P2=2, P3=3
#[test]
fn all_edge_types_contribute_to_depth() {
    let order = vec![
        PassIndex(0),
        PassIndex(1),
        PassIndex(2),
        PassIndex(3),
    ];
    let edges = vec![raw(0, 1, 1), war(1, 2, 2), waw(2, 3, 3)];

    let depths = compute_pass_depths(&order, &edges);

    assert_depths(&depths, &[(0, 0), (1, 1), (2, 2), (3, 3)]);
}

/// Mixed edge types in a diamond topology. All edge types create predecessor
/// relationships regardless of their classification.
///
/// P0 ->[RAW] P1
/// P0 ->[WAR] P2
/// P1 ->[WAW] P3
/// P2 ->[RAW] P3
///
/// Depths: P0=0, P1=1, P2=1, P3=2
#[test]
fn mixed_edge_types_diamond() {
    let order = vec![
        PassIndex(0),
        PassIndex(1),
        PassIndex(2),
        PassIndex(3),
    ];
    let edges = vec![raw(0, 1, 1), war(0, 2, 2), waw(1, 3, 3), raw(2, 3, 4)];

    let depths = compute_pass_depths(&order, &edges);

    assert_depths(&depths, &[(0, 0), (1, 1), (2, 1), (3, 2)]);
}

// =============================================================================
// SECTION 7 -- Order sensitivity and robustness
// =============================================================================

/// Passes that appear earlier in the order but depend on later passes should
/// still compute correctly: the predecessor will not have been assigned a depth
/// yet, so it defaults to 0 for the max calculation. This tests robustness
/// when the order is not a valid topological order.
///
/// order: [P1, P0]  (P1 depends on P0, but P1 comes first)
/// edge: P0 -> P1
///
/// Depths: P1's predecessor P0 has not been seen yet, defaults to 0.
///         P1 = 0 + 1 = 1
///         P0 = 0 (no predecessors)
#[test]
fn reverse_order_robustness() {
    // P1 appears first in order but depends on P0.
    // P0 has no predecessor -> depth 0.
    // P1 has predecessor P0, but since we process topo-order, and P0 hasn't
    // been assigned yet (it appears later), depth(P0) defaults to 0.
    // So P1 = 0 + 1 = 1, and when P0 is processed, P0 = 0.
    let order = vec![PassIndex(1), PassIndex(0)];
    let edges = vec![raw(0, 1, 1)];

    let depths = compute_pass_depths(&order, &edges);

    // P1 is processed first, P0 not yet assigned -> P1 uses default 0 for P0
    // -> P1 = 1. Then P0 has no preds -> P0 = 0.
    assert_depths(&depths, &[(1, 1), (0, 0)]);
}

/// Deep diamond with intermediate passes at multiple depths that merge in a
/// complex pattern. Exercises the longest-path logic thoroughly.
///
///           P1
///          / \
///     P0 --> P3 --> P5
///      \   / \    /
///       P2   P4
///
/// Edges:
///   P0->P1, P0->P2    (fork)
///   P1->P3, P2->P3    (first merge)
///   P1->P4             (bypass from P1)
///   P3->P5, P4->P5    (second merge)
///
/// Depths:
///   P0=0
///   P1=1, P2=1
///   P3=max(1,1)+1=2, P4=max(1)+1=2
///   P5=max(2,2)+1=3
#[test]
fn deep_diamond_complex() {
    let order = vec![
        PassIndex(0),
        PassIndex(1),
        PassIndex(2),
        PassIndex(3),
        PassIndex(4),
        PassIndex(5),
    ];
    let edges = vec![
        raw(0, 1, 1),
        raw(0, 2, 2),
        raw(1, 3, 3),
        raw(2, 3, 4),
        raw(1, 4, 5),
        raw(3, 5, 6),
        raw(4, 5, 7),
    ];

    let depths = compute_pass_depths(&order, &edges);

    assert_depths(
        &depths,
        &[(0, 0), (1, 1), (2, 1), (3, 2), (4, 2), (5, 3)],
    );
}

/// A pass with multiple predecessors at different depths takes the max.
///
/// P0 -> P3
/// P1 -> P3
/// P2 -> P3
///
/// Where P0=0, P1=1, P2=2.
/// P3 = max(0,1,2) + 1 = 3.
#[test]
fn triple_predecessor_at_different_depths() {
    let order = vec![
        PassIndex(0),
        PassIndex(1),
        PassIndex(2),
        PassIndex(3),
    ];
    // Create a chain P0->P1->P2, then all three feed P3.
    let edges = vec![
        raw(0, 1, 1),
        raw(1, 2, 2),
        raw(0, 3, 3),
        raw(1, 3, 4),
        raw(2, 3, 5),
    ];

    let depths = compute_pass_depths(&order, &edges);

    assert_depths(&depths, &[(0, 0), (1, 1), (2, 2), (3, 3)]);
}
