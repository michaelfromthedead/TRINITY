// Blackbox contract tests for T-FG-2.3 TopoValidator.
//
// CLEANROOM: No src/ access beyond the public API exported by the crate.
// Tests use only `renderer_backend::frame_graph::*` -- no internal fields,
// no private methods, no implementation details.
//
// Contract:
//   TopoValidator::validate(order, edges) checks that the given topological
//   ordering respects all dependency edges.
//
//   Signature:
//     pub fn validate(
//         order: &[PassIndex],
//         edges: &[IrEdge],
//     ) -> Result<(), Vec<String>>
//
//   Three checks:
//     1. No duplicates -- every PassIndex appears at most once in `order`.
//     2. All edge endpoints present -- every `from` and `to` in `edges`
//        exists somewhere in `order`.
//     3. Edge ordering -- for every edge, `from` appears at an earlier
//        position than `to` in `order`.
//
//   Returns `Ok(())` on success, or `Err(Vec<String>)` with one human-readable
//   message per violation.
//
//   TopoValidator derives Clone, Debug, Default.
//
// Coverage:
//   1.  Empty order + empty edges -> Ok(())
//   2.  Single pass, no edges -> Ok(())
//   3.  Linear chain valid (0->1->2) -> Ok(())
//   4.  Diamond DAG valid -> Ok(())
//   5.  WAR edges respected -> Ok(())
//   6.  WAW edges respected -> Ok(())
//   7.  All three edge types simultaneously -> Ok(())
//   8.  Non-sequential pass indices (sparse) -> Ok(())
//   9.  Multiple edges over same resource -> Ok(())
//  10.  Duplicate pass at start of order -> Err
//  11.  Duplicate pass at end of order -> Err
//  12.  Multiple identical duplicates -> Err with multiple messages
//  13.  All same pass repeated -> Err
//  14.  Edge from-pass missing from order -> Err
//  15.  Edge to-pass missing from order -> Err
//  16.  Both from and to missing -> Err with two messages
//  17.  Multiple edges with missing endpoints -> Err with multiple messages
//  18.  Simple ordering reversal (1 before 0) -> Err
//  19.  Self-loop edge (from==to) -> Err
//  20.  Multiple ordering violations -> Err with multiple messages
//  21.  Duplicate + missing endpoint combined -> Err with messages for both
//  22.  Missing endpoint + ordering violation combined -> Err with messages
//  23.  Duplicate + missing + ordering violation -> Err with messages for all
//  24.  Empty order with non-empty edges -> Err
//  25.  Error messages mention "duplicate" for check 1 violations
//  26.  Error messages mention "not found" for check 2 violations
//  27.  Error messages mention "violation" for check 3 violations
//  28.  TopoValidator derives Clone (clone works)
//  29.  TopoValidator derives Debug (non-empty output)
//  30.  TopoValidator derives Default (same as new)
//  31.  Valid branching DAG (fan-out) -> Ok(())
//  32.  Valid merging DAG (fan-in) -> Ok(())
//  33.  Complex 6-pass DAG -> Ok(())
//  34.  Reversed 5-pass chain -> Err with 4 violation messages
//  35.  Large batch: 50 passes in valid order -> Ok(())

use renderer_backend::frame_graph::{EdgeType, IrEdge, PassIndex, TopoValidator};

// =============================================================================
// SECTION 1 -- Happy path: valid topological orders
// =============================================================================

#[test]
fn empty_order_empty_edges_returns_ok() {
    let result = TopoValidator::validate(&[], &[]);
    assert!(result.is_ok(), "empty order, empty edges -> Ok");
}

#[test]
fn single_pass_no_edges_returns_ok() {
    let result = TopoValidator::validate(&[PassIndex(0)], &[]);
    assert!(result.is_ok(), "single pass, no edges -> Ok");
}

#[test]
fn linear_chain_valid() {
    let order = &[PassIndex(0), PassIndex(1), PassIndex(2)];
    let edges = &[
        IrEdge::new(PassIndex(0), PassIndex(1), resource(1), EdgeType::RAW),
        IrEdge::new(PassIndex(1), PassIndex(2), resource(2), EdgeType::RAW),
    ];
    let result = TopoValidator::validate(order, edges);
    assert!(result.is_ok(), "linear chain P0->P1->P2 should be valid");
}

#[test]
fn diamond_dag_valid() {
    // Diamond:   1
    //          /   \
    //         0     3
    //          \   /
    //           2
    let order = &[PassIndex(1), PassIndex(0), PassIndex(2), PassIndex(3)];
    let edges = &[
        IrEdge::new(PassIndex(1), PassIndex(0), resource(1), EdgeType::RAW),
        IrEdge::new(PassIndex(1), PassIndex(2), resource(2), EdgeType::RAW),
        IrEdge::new(PassIndex(0), PassIndex(3), resource(3), EdgeType::RAW),
        IrEdge::new(PassIndex(2), PassIndex(3), resource(4), EdgeType::RAW),
    ];
    let result = TopoValidator::validate(order, edges);
    assert!(result.is_ok(), "diamond DAG should be valid");
}

#[test]
fn war_edges_respected() {
    // WAR edges: P0 reads, P1 writes -> P0 must be before P1.
    let order = &[PassIndex(0), PassIndex(1)];
    let edges = &[
        IrEdge::new(PassIndex(0), PassIndex(1), resource(1), EdgeType::WAR),
    ];
    let result = TopoValidator::validate(order, edges);
    assert!(result.is_ok(), "WAR edge P0->P1 respected");
}

#[test]
fn waw_edges_respected() {
    // WAW edges: P0 writes, P1 overwrites -> P0 must be before P1.
    let order = &[PassIndex(0), PassIndex(1)];
    let edges = &[
        IrEdge::new(PassIndex(0), PassIndex(1), resource(1), EdgeType::WAW),
    ];
    let result = TopoValidator::validate(order, edges);
    assert!(result.is_ok(), "WAW edge P0->P1 respected");
}

#[test]
fn all_edge_types_simultaneously() {
    let order = &[PassIndex(0), PassIndex(1), PassIndex(2)];
    let edges = &[
        IrEdge::new(PassIndex(0), PassIndex(1), resource(1), EdgeType::RAW),
        IrEdge::new(PassIndex(1), PassIndex(2), resource(2), EdgeType::WAR),
        IrEdge::new(PassIndex(0), PassIndex(2), resource(3), EdgeType::WAW),
    ];
    let result = TopoValidator::validate(order, edges);
    assert!(result.is_ok(), "all three edge types valid");
}

#[test]
fn non_sequential_pass_indices() {
    // Sparse / non-sequential pass indices.
    let order = &[PassIndex(10), PassIndex(55), PassIndex(200)];
    let edges = &[
        IrEdge::new(PassIndex(10), PassIndex(55), resource(1), EdgeType::RAW),
        IrEdge::new(PassIndex(55), PassIndex(200), resource(2), EdgeType::RAW),
    ];
    let result = TopoValidator::validate(order, edges);
    assert!(result.is_ok(), "non-sequential pass indices valid");
}

#[test]
fn multiple_edges_same_resource() {
    let order = &[PassIndex(0), PassIndex(1), PassIndex(2), PassIndex(3)];
    let res = resource(7);
    let edges = &[
        IrEdge::new(PassIndex(0), PassIndex(1), res, EdgeType::RAW),
        IrEdge::new(PassIndex(1), PassIndex(2), res, EdgeType::RAW),
        IrEdge::new(PassIndex(2), PassIndex(3), res, EdgeType::RAW),
    ];
    let result = TopoValidator::validate(order, edges);
    assert!(result.is_ok(), "multiple edges same resource valid");
}

// =============================================================================
// SECTION 2 -- Error: duplicate passes in order (check 1)
// =============================================================================

#[test]
fn duplicate_pass_at_start() {
    let order = &[PassIndex(0), PassIndex(0), PassIndex(1)];
    let edges = &[IrEdge::new(PassIndex(0), PassIndex(1), resource(1), EdgeType::RAW)];
    let result = TopoValidator::validate(order, edges);
    assert!(result.is_err(), "duplicate at start should fail");
}

#[test]
fn duplicate_pass_at_end() {
    let order = &[PassIndex(0), PassIndex(1), PassIndex(1)];
    let edges = &[IrEdge::new(PassIndex(0), PassIndex(1), resource(1), EdgeType::RAW)];
    let result = TopoValidator::validate(order, edges);
    assert!(result.is_err(), "duplicate at end should fail");
}

#[test]
fn multiple_duplicate_passes() {
    let order = &[PassIndex(0), PassIndex(0), PassIndex(1), PassIndex(1)];
    let edges = &[IrEdge::new(PassIndex(0), PassIndex(1), resource(1), EdgeType::RAW)];
    let result = TopoValidator::validate(order, edges);
    assert!(result.is_err(), "multiple duplicates should fail");

    let errors = result.unwrap_err();
    assert!(errors.len() >= 2, "expected at least 2 error messages for 2 duplicate pairs");
}

#[test]
fn all_same_pass_repeated() {
    let order = &[PassIndex(42), PassIndex(42), PassIndex(42), PassIndex(42)];
    let result = TopoValidator::validate(order, &[]);
    assert!(result.is_err(), "all same pass repeated should fail");
}

// =============================================================================
// SECTION 3 -- Error: missing edge endpoints (check 2)
// =============================================================================

#[test]
fn edge_from_pass_missing() {
    let order = &[PassIndex(0), PassIndex(1)];
    let edges = &[
        // PassIndex(5) is not in order.
        IrEdge::new(PassIndex(5), PassIndex(1), resource(1), EdgeType::RAW),
    ];
    let result = TopoValidator::validate(order, edges);
    assert!(result.is_err(), "missing from-pass should fail");
}

#[test]
fn edge_to_pass_missing() {
    let order = &[PassIndex(0), PassIndex(1)];
    let edges = &[
        // PassIndex(9) is not in order.
        IrEdge::new(PassIndex(0), PassIndex(9), resource(1), EdgeType::RAW),
    ];
    let result = TopoValidator::validate(order, edges);
    assert!(result.is_err(), "missing to-pass should fail");
}

#[test]
fn both_from_and_to_missing() {
    let order = &[PassIndex(0)];
    let edges = &[
        IrEdge::new(PassIndex(10), PassIndex(20), resource(1), EdgeType::RAW),
    ];
    let result = TopoValidator::validate(order, edges);
    assert!(result.is_err(), "both endpoints missing should fail");

    let errors = result.unwrap_err();
    assert_eq!(errors.len(), 2, "expected 2 error messages (from + to)");
}

#[test]
fn multiple_edges_missing_endpoints() {
    let order = &[PassIndex(0)];
    let edges = &[
        IrEdge::new(PassIndex(10), PassIndex(20), resource(1), EdgeType::RAW),
        IrEdge::new(PassIndex(30), PassIndex(40), resource(2), EdgeType::WAR),
    ];
    let result = TopoValidator::validate(order, edges);
    assert!(result.is_err(), "multiple missing endpoints should fail");

    let errors = result.unwrap_err();
    // 2 edges * 2 missing endpoints each = 4 messages.
    assert!(errors.len() >= 4, "expected >=4 error messages, got {}", errors.len());
}

// =============================================================================
// SECTION 4 -- Error: ordering violations (check 3)
// =============================================================================

#[test]
fn simple_order_reversal() {
    // P1 appears before P0, but edge says P0->P1.
    let order = &[PassIndex(1), PassIndex(0)];
    let edges = &[
        IrEdge::new(PassIndex(0), PassIndex(1), resource(1), EdgeType::RAW),
    ];
    let result = TopoValidator::validate(order, edges);
    assert!(result.is_err(), "reversed order should fail");
}

#[test]
fn self_loop_edge() {
    // Self-loop: from==to. Since position[P0] == position[P0], from_pos >= to_pos.
    let order = &[PassIndex(0)];
    let edges = &[
        IrEdge::new(PassIndex(0), PassIndex(0), resource(1), EdgeType::RAW),
    ];
    let result = TopoValidator::validate(order, edges);
    assert!(result.is_err(), "self-loop edge should fail ordering check");
}

#[test]
fn multiple_ordering_violations() {
    // 4-pass chain [3,2,1,0] with edges 0->1, 1->2, 2->3 is entirely reversed.
    let order = &[PassIndex(3), PassIndex(2), PassIndex(1), PassIndex(0)];
    let edges = &[
        IrEdge::new(PassIndex(0), PassIndex(1), resource(1), EdgeType::RAW),
        IrEdge::new(PassIndex(1), PassIndex(2), resource(2), EdgeType::RAW),
        IrEdge::new(PassIndex(2), PassIndex(3), resource(3), EdgeType::RAW),
    ];
    let result = TopoValidator::validate(order, edges);
    assert!(result.is_err(), "entirely reversed chain should fail");

    let errors = result.unwrap_err();
    assert_eq!(errors.len(), 3, "expected 3 ordering violation messages for 3 reversed edges");
}

// =============================================================================
// SECTION 5 -- Combined violations (multiple checks)
// =============================================================================

#[test]
fn duplicate_and_missing_endpoint() {
    let order = &[PassIndex(0), PassIndex(0)];
    let edges = &[
        IrEdge::new(PassIndex(42), PassIndex(0), resource(1), EdgeType::RAW),
    ];
    let result = TopoValidator::validate(order, edges);
    assert!(result.is_err(), "duplicate + missing endpoint should fail");

    let errors = result.unwrap_err();
    // Expected: 1 duplicate message + 1 missing from-pass message.
    assert!(errors.len() >= 2, "expected >=2 messages, got {}", errors.len());
}

#[test]
fn missing_endpoint_and_ordering_violation() {
    let order = &[PassIndex(1), PassIndex(0)];
    let edges = &[
        IrEdge::new(PassIndex(0), PassIndex(1), resource(1), EdgeType::RAW),
        // to-pass missing:
        IrEdge::new(PassIndex(1), PassIndex(99), resource(2), EdgeType::WAR),
    ];
    let result = TopoValidator::validate(order, edges);
    assert!(result.is_err(), "missing endpoint + ordering violation should fail");

    let errors = result.unwrap_err();
    // Expected: 1 ordering violation (P0->P1 reversed) + 1 missing to-pass.
    assert!(errors.len() >= 2, "expected >=2 messages, got {}", errors.len());
}

#[test]
fn duplicate_missing_and_ordering_all_together() {
    let order = &[PassIndex(0), PassIndex(1), PassIndex(0)];
    let edges = &[
        IrEdge::new(PassIndex(0), PassIndex(1), resource(1), EdgeType::RAW),
        IrEdge::new(PassIndex(0), PassIndex(99), resource(2), EdgeType::RAW),
    ];
    let result = TopoValidator::validate(order, edges);
    assert!(result.is_err(), "duplicate + missing + ordering should fail");

    let errors = result.unwrap_err();
    // Expected: at least 1 duplicate, 1 missing endpoint, 1 ordering = at least 3.
    assert!(errors.len() >= 3, "expected >=3 messages, got {}", errors.len());
}

// =============================================================================
// SECTION 6 -- Empty order edge cases
// =============================================================================

#[test]
fn empty_order_with_non_empty_edges() {
    let edges = &[
        IrEdge::new(PassIndex(0), PassIndex(1), resource(1), EdgeType::RAW),
    ];
    let result = TopoValidator::validate(&[], edges);
    assert!(result.is_err(), "empty order with edges should fail");

    let errors = result.unwrap_err();
    // Both from and to are missing.
    assert!(errors.len() >= 2, "expected >=2 messages (from + to missing)");
}

// =============================================================================
// SECTION 7 -- Error message content verification
// =============================================================================

#[test]
fn duplicate_error_message_contains_duplicate() {
    let order = &[PassIndex(0), PassIndex(0)];
    let result = TopoValidator::validate(order, &[]);
    let errors = result.unwrap_err();

    let all_lower: String = errors.join(" ").to_lowercase();
    assert!(
        all_lower.contains("duplicate"),
        "duplicate error message should contain 'duplicate': {:?}",
        errors,
    );
}

#[test]
fn missing_endpoint_message_contains_not_found() {
    let order = &[PassIndex(0)];
    let edges = &[
        IrEdge::new(PassIndex(99), PassIndex(0), resource(1), EdgeType::RAW),
    ];
    let result = TopoValidator::validate(order, edges);
    let errors = result.unwrap_err();

    let all_lower: String = errors.join(" ").to_lowercase();
    assert!(
        all_lower.contains("not found"),
        "missing endpoint message should contain 'not found': {:?}",
        errors,
    );
}

#[test]
fn ordering_violation_message_contains_violation() {
    let order = &[PassIndex(1), PassIndex(0)];
    let edges = &[
        IrEdge::new(PassIndex(0), PassIndex(1), resource(1), EdgeType::RAW),
    ];
    let result = TopoValidator::validate(order, edges);
    let errors = result.unwrap_err();

    let all_lower: String = errors.join(" ").to_lowercase();
    assert!(
        all_lower.contains("violation"),
        "ordering violation message should contain 'violation': {:?}",
        errors,
    );
}

#[test]
fn ordering_message_mentions_edge_type() {
    let order = &[PassIndex(1), PassIndex(0)];
    let edges = &[
        IrEdge::new(PassIndex(0), PassIndex(1), resource(1), EdgeType::WAR),
    ];
    let result = TopoValidator::validate(order, edges);
    let errors = result.unwrap_err();

    let first = &errors[0];
    assert!(
        first.contains("WAR"),
        "ordering message should mention edge type WAR: {}",
        first,
    );
}

// =============================================================================
// SECTION 8 -- TopoValidator derives
// =============================================================================

#[test]
fn topo_validator_clone_works() {
    let a = TopoValidator;
    let b = a.clone();
    // Both must validate identically.
    let result_a = TopoValidator::validate(&[PassIndex(0)], &[]);
    let result_b = TopoValidator::validate(&[PassIndex(0)], &[]);
    assert_eq!(result_a, result_b, "cloned validator produces same results");
    let _ = b; // suppress unused warning
}

#[test]
fn topo_validator_debug_non_empty() {
    let debug = format!("{:?}", TopoValidator);
    assert!(!debug.is_empty(), "TopoValidator Debug output is non-empty");
}

#[test]
fn topo_validator_default_works() {
    let v: TopoValidator = Default::default();
    let result = TopoValidator::validate(&[PassIndex(0), PassIndex(1)], &[]);
    assert!(result.is_ok(), "Default::default() validator works");
    let _ = v;
}

// =============================================================================
// SECTION 9 -- Complex DAG shapes
// =============================================================================

#[test]
fn branching_fan_out_dag_valid() {
    // P0 fans out to P1, P2, P3.
    let order = &[PassIndex(0), PassIndex(1), PassIndex(2), PassIndex(3)];
    let edges = &[
        IrEdge::new(PassIndex(0), PassIndex(1), resource(1), EdgeType::RAW),
        IrEdge::new(PassIndex(0), PassIndex(2), resource(2), EdgeType::WAR),
        IrEdge::new(PassIndex(0), PassIndex(3), resource(3), EdgeType::WAW),
    ];
    let result = TopoValidator::validate(order, edges);
    assert!(result.is_ok(), "fan-out DAG valid");
}

#[test]
fn merging_fan_in_dag_valid() {
    // P0, P1, P2 all feed into P3.
    let order = &[PassIndex(0), PassIndex(1), PassIndex(2), PassIndex(3)];
    let edges = &[
        IrEdge::new(PassIndex(0), PassIndex(3), resource(1), EdgeType::RAW),
        IrEdge::new(PassIndex(1), PassIndex(3), resource(2), EdgeType::RAW),
        IrEdge::new(PassIndex(2), PassIndex(3), resource(3), EdgeType::RAW),
    ];
    let result = TopoValidator::validate(order, edges);
    assert!(result.is_ok(), "fan-in DAG valid");
}

#[test]
fn complex_six_pass_dag_valid() {
    // Complex DAG:
    //   P0 -> P1 -> P3 -> P5
    //    |           ^
    //    v           |
    //   P2 ----------+
    //    |
    //    v
    //   P4
    let order = &[
        PassIndex(0),
        PassIndex(1),
        PassIndex(2),
        PassIndex(3),
        PassIndex(4),
        PassIndex(5),
    ];
    let edges = &[
        IrEdge::new(PassIndex(0), PassIndex(1), resource(1), EdgeType::RAW),
        IrEdge::new(PassIndex(0), PassIndex(2), resource(2), EdgeType::RAW),
        IrEdge::new(PassIndex(1), PassIndex(3), resource(3), EdgeType::RAW),
        IrEdge::new(PassIndex(2), PassIndex(3), resource(4), EdgeType::RAW),
        IrEdge::new(PassIndex(2), PassIndex(4), resource(5), EdgeType::RAW),
        IrEdge::new(PassIndex(3), PassIndex(5), resource(6), EdgeType::RAW),
    ];
    let result = TopoValidator::validate(order, edges);
    assert!(result.is_ok(), "complex 6-pass DAG valid");
}

// =============================================================================
// SECTION 10 -- Wholesale reversed chain
// =============================================================================

#[test]
fn reversed_five_pass_chain() {
    // Full reversal: 5 passes in reverse order with edges 0->1, 1->2, 2->3, 3->4.
    // Every edge is violated.
    let order = &[PassIndex(4), PassIndex(3), PassIndex(2), PassIndex(1), PassIndex(0)];
    let edges = &[
        IrEdge::new(PassIndex(0), PassIndex(1), resource(1), EdgeType::RAW),
        IrEdge::new(PassIndex(1), PassIndex(2), resource(2), EdgeType::WAR),
        IrEdge::new(PassIndex(2), PassIndex(3), resource(3), EdgeType::WAW),
        IrEdge::new(PassIndex(3), PassIndex(4), resource(4), EdgeType::RAW),
    ];
    let result = TopoValidator::validate(order, edges);
    assert!(result.is_err(), "reversed 5-pass chain should fail");

    let errors = result.unwrap_err();
    assert_eq!(
        errors.len(),
        4,
        "expected 4 violation messages for 4 reversed edges",
    );
}

// =============================================================================
// SECTION 11 -- Large batch stress test
// =============================================================================

#[test]
fn fifty_passes_valid_chain() {
    // 50 passes in order with 49 edges forming a linear chain.
    let order: Vec<PassIndex> = (0..50).map(|i| PassIndex(i)).collect();
    let edges: Vec<IrEdge> = (0..49)
        .map(|i| IrEdge::new(PassIndex(i), PassIndex(i + 1), resource(i as u32), EdgeType::RAW))
        .collect();

    let result = TopoValidator::validate(&order, &edges);
    assert!(result.is_ok(), "50-pass valid chain should succeed");
}

#[test]
fn fifty_passes_reversed_chain() {
    // 50 passes in reverse order with 49 edges -> all violated.
    let order: Vec<PassIndex> = (0..50).rev().map(|i| PassIndex(i)).collect();
    let edges: Vec<IrEdge> = (0..49)
        .map(|i| IrEdge::new(PassIndex(i), PassIndex(i + 1), resource(i as u32), EdgeType::RAW))
        .collect();

    let result = TopoValidator::validate(&order, &edges);
    assert!(result.is_err(), "50-pass reversed chain should fail");

    let errors = result.unwrap_err();
    assert_eq!(errors.len(), 49, "all 49 edges should be reported as violations");
}

// =============================================================================
// SECTION 12 -- WAR and WAW ordering violations specifically
// =============================================================================

#[test]
fn war_ordering_violation() {
    // WAR edge P0->P1 but P1 appears first.
    let order = &[PassIndex(1), PassIndex(0)];
    let edges = &[
        IrEdge::new(PassIndex(0), PassIndex(1), resource(1), EdgeType::WAR),
    ];
    let result = TopoValidator::validate(order, edges);
    assert!(result.is_err(), "WAR ordering violation should fail");
}

#[test]
fn waw_ordering_violation() {
    // WAW edge P0->P1 but P1 appears first.
    let order = &[PassIndex(1), PassIndex(0)];
    let edges = &[
        IrEdge::new(PassIndex(0), PassIndex(1), resource(1), EdgeType::WAW),
    ];
    let result = TopoValidator::validate(order, edges);
    assert!(result.is_err(), "WAW ordering violation should fail");
}

// =============================================================================
// SECTION 13 -- Error message reports the pass index
// =============================================================================

#[test]
fn duplicate_message_contains_pass_index() {
    let order = &[PassIndex(42), PassIndex(42)];
    let result = TopoValidator::validate(order, &[]);
    let errors = result.unwrap_err();

    let all = errors.join(" ");
    assert!(
        all.contains("42"),
        "duplicate message should mention the duplicate pass index: {}",
        all,
    );
}

#[test]
fn missing_endpoint_message_contains_pass_index() {
    let order = &[PassIndex(0)];
    let edges = &[
        IrEdge::new(PassIndex(99), PassIndex(0), resource(1), EdgeType::RAW),
    ];
    let result = TopoValidator::validate(order, edges);
    let errors = result.unwrap_err();

    let all = errors.join(" ");
    assert!(
        all.contains("99"),
        "missing endpoint message should mention the missing pass index: {}",
        all,
    );
}

// =============================================================================
// SECTION 14 -- Zero-value and boundary pass indices
// =============================================================================

#[test]
fn pass_index_zero_in_order() {
    let result = TopoValidator::validate(&[PassIndex(0)], &[]);
    assert!(result.is_ok(), "PassIndex(0) in order is valid");
}

#[test]
fn pass_index_max_usize_in_order() {
    let result = TopoValidator::validate(&[PassIndex(usize::MAX)], &[]);
    assert!(result.is_ok(), "PassIndex(usize::MAX) in order is valid");
}

#[test]
fn pass_index_zero_as_edge_endpoint() {
    let order = &[PassIndex(0), PassIndex(1)];
    let edges = &[
        IrEdge::new(PassIndex(0), PassIndex(1), resource(1), EdgeType::RAW),
    ];
    let result = TopoValidator::validate(order, edges);
    assert!(result.is_ok(), "PassIndex(0) as edge endpoint is valid");
}

// =============================================================================
// SECTION 15 -- Error count consistency
// =============================================================================

#[test]
fn no_error_when_no_violations() {
    let result = TopoValidator::validate(&[PassIndex(0), PassIndex(1)], &[]);
    assert!(result.is_ok(), "no violations -> Ok");
    // Ok result carries no error messages; unwrap returns unit.
    assert_eq!(result.unwrap(), (), "Ok value is unit");
}

#[test]
fn single_duplicate_single_error_message() {
    // Exactly 1 duplicate should produce exactly 1 error message about it.
    let order = &[PassIndex(5), PassIndex(5)];
    let result = TopoValidator::validate(order, &[]);
    let errors = result.unwrap_err();
    assert_eq!(errors.len(), 1, "single duplicate -> single error message");
}

// =============================================================================
// Helper
// =============================================================================

fn resource(id: u32) -> renderer_backend::frame_graph::ResourceHandle {
    renderer_backend::frame_graph::ResourceHandle(id)
}
