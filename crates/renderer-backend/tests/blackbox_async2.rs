// Blackbox contract tests for T-FG-5.1 (AsyncExecutionPlan / build_async_plan).
// CLEANROOM: No src/ access beyond the public API.
// Test ID: SDLC TESTDEV_BLACKBOX T-FG-5.1 AsyncExec
//
// Contract:
//   build_async_plan(order, async_passes) -> AsyncExecutionPlan
//   Splits a topological pass order into a graphics_queue (sequential
//   main-queue) and async_queues (HashMap<String, Vec<PassIndex>>) by
//   queue type string. Passes whose index appears in async_passes are
//   moved from the graphics queue into their respective async queues.
//   Relative order within each queue matches the input order.
//
// Areas tested (complementary to blackbox_async_exec.rs):
//   1. Partition correctness -- deterministic, multi-type partition
//   2. Order preservation -- deep interleaving patterns
//   3. Edge cases -- usize::MAX, 10K-order stress, degenerate interleaving
//   4. Determinism -- same inputs -> same outputs
//   5. Round-trip consistency -- async_schedule -> build_async_plan
//   6. Queue type isolation -- no cross-contamination between types

use std::collections::HashSet;

use renderer_backend::frame_graph::{
    build_async_plan, is_async_pass, AsyncExecutionPlan, PassIndex,
};

// =============================================================================
// SECTION 1 -- Partition correctness: deterministic multi-type split
// =============================================================================

#[test]
fn partition_places_passes_into_correct_queue_by_type() {
    // Each pass in async_passes maps to a distinct queue type.
    // The function must place each pass into the queue matching its
    // associated type string, regardless of order position.
    let order = vec![
        PassIndex(0), PassIndex(1), PassIndex(2), PassIndex(3), PassIndex(4),
    ];
    let async_passes = vec![
        (PassIndex(0), "compute".to_string()),
        (PassIndex(1), "copy".to_string()),
        (PassIndex(2), "transfer".to_string()),
        (PassIndex(3), "custom_async".to_string()),
        (PassIndex(4), "compute".to_string()),
    ];

    let plan = build_async_plan(&order, &async_passes);

    // Pass 0 -> compute, Pass 1 -> copy, Pass 2 -> transfer,
    // Pass 3 -> custom_async, Pass 4 -> compute
    assert_eq!(
        plan.async_queues.get("compute"),
        Some(&vec![PassIndex(0), PassIndex(4)]),
        "compute queue must contain passes 0 and 4"
    );
    assert_eq!(
        plan.async_queues.get("copy"),
        Some(&vec![PassIndex(1)]),
        "copy queue must contain pass 1"
    );
    assert_eq!(
        plan.async_queues.get("transfer"),
        Some(&vec![PassIndex(2)]),
        "transfer queue must contain pass 2"
    );
    assert_eq!(
        plan.async_queues.get("custom_async"),
        Some(&vec![PassIndex(3)]),
        "custom_async queue must contain pass 3"
    );

    // graphics_queue must be empty since all passes are async
    assert!(
        plan.graphics_queue.is_empty(),
        "graphics_queue must be empty when all passes are async"
    );
}

#[test]
fn partition_multiple_types_same_queue_keeps_all_passes() {
    // Multiple passes with the same queue type and queue type at
    // non-consecutive positions must all end up in the same queue
    // entry in the correct relative order.
    let order = vec![
        PassIndex(0), PassIndex(1), PassIndex(2),
        PassIndex(3), PassIndex(4), PassIndex(5),
    ];
    let async_passes = vec![
        (PassIndex(0), "compute".to_string()),
        (PassIndex(2), "compute".to_string()),
        (PassIndex(3), "compute".to_string()),
        (PassIndex(5), "compute".to_string()),
    ];

    let plan = build_async_plan(&order, &async_passes);

    let compute_queue = plan.async_queues.get("compute")
        .expect("compute queue must exist");
    assert_eq!(
        compute_queue,
        &vec![PassIndex(0), PassIndex(2), PassIndex(3), PassIndex(5)],
        "compute queue must contain all four passes in input order"
    );

    // graphics_queue gets passes 1 and 4
    assert_eq!(
        plan.graphics_queue,
        vec![PassIndex(1), PassIndex(4)],
        "graphics_queue must contain the non-async passes 1 and 4"
    );
}

#[test]
fn partition_no_async_pass_escapes_to_wrong_queue() {
    // Every async pass must be absent from graphics_queue and present
    // in exactly one async queue.
    let order: Vec<PassIndex> = (0..20).map(PassIndex).collect();
    let async_passes: Vec<(PassIndex, String)> = (0..20)
        .filter(|i| i % 3 == 0)
        .map(|i| (PassIndex(i), "compute".to_string()))
        .collect();

    let plan = build_async_plan(&order, &async_passes);

    let async_set: HashSet<PassIndex> = plan.async_queues.values()
        .flat_map(|v| v.iter())
        .copied()
        .collect();

    // Every async pass in the input must appear in the async queues
    for &(pass, _) in &async_passes {
        assert!(
            async_set.contains(&pass),
            "async pass {:?} must be in some async queue",
            pass
        );
    }

    // No async pass may appear in graphics_queue
    for &(pass, _) in &async_passes {
        assert!(
            !plan.graphics_queue.contains(&pass),
            "async pass {:?} must NOT be in graphics_queue",
            pass
        );
    }
}

#[test]
fn partition_non_async_passes_never_appear_in_async_queues() {
    // Passes NOT listed in async_passes must never appear in any
    // async queue -- they belong exclusively to graphics_queue.
    let order: Vec<PassIndex> = (0..10).map(PassIndex).collect();
    let async_passes = vec![
        (PassIndex(1), "compute".to_string()),
        (PassIndex(3), "copy".to_string()),
        (PassIndex(5), "compute".to_string()),
    ];

    let plan = build_async_plan(&order, &async_passes);

    let all_async: Vec<PassIndex> = plan.async_queues.values()
        .flat_map(|v| v.iter())
        .copied()
        .collect();

    let non_async: Vec<PassIndex> = order.iter()
        .filter(|idx| !async_passes.iter().any(|(p, _)| p == *idx))
        .copied()
        .collect();

    for pass in &non_async {
        assert!(
            !all_async.contains(pass),
            "non-async pass {:?} must not appear in any async queue",
            pass
        );
        assert!(
            plan.graphics_queue.contains(pass),
            "non-async pass {:?} must be in graphics_queue",
            pass
        );
    }
}

// =============================================================================
// SECTION 2 -- Order preservation: deep interleaving patterns
// =============================================================================

#[test]
fn order_preservation_async_passes_at_every_position() {
    // Vary the position of a single async pass across the full order
    // and verify the graphics_queue shrinks and shifts accordingly.
    for async_pos in 0..10 {
        let order: Vec<PassIndex> = (0..10).map(PassIndex).collect();
        let async_passes = vec![(PassIndex(async_pos), "compute".to_string())];

        let plan = build_async_plan(&order, &async_passes);

        let expected_graphics: Vec<PassIndex> = (0..10)
            .filter(|i| *i != async_pos)
            .map(PassIndex)
            .collect();
        assert_eq!(
            plan.graphics_queue, expected_graphics,
            "graphics_queue mismatch when async pass is at position {}",
            async_pos
        );

        let compute_queue = plan.async_queues.get("compute")
            .expect("compute queue must exist");
        assert_eq!(
            compute_queue,
            &vec![PassIndex(async_pos)],
            "compute queue must contain the single async pass at position {}",
            async_pos
        );
    }
}

#[test]
fn order_preservation_block_of_consecutive_async_passes() {
    // A contiguous block of async passes at the beginning, middle,
    // and end must be correctly partitioned and preserve order.
    let cases: Vec<(&str, Vec<PassIndex>, Vec<(PassIndex, String)>)> = vec![
        (
            "async block at start (0-4)",
            (0..10).map(PassIndex).collect(),
            (0..5).map(|i| (PassIndex(i), "compute".to_string())).collect(),
        ),
        (
            "async block in middle (3-7)",
            (0..10).map(PassIndex).collect(),
            (3..8).map(|i| (PassIndex(i), "compute".to_string())).collect(),
        ),
        (
            "async block at end (5-9)",
            (0..10).map(PassIndex).collect(),
            (5..10).map(|i| (PassIndex(i), "compute".to_string())).collect(),
        ),
    ];

    for (label, order, async_passes) in cases {
        let plan = build_async_plan(&order, &async_passes);

        let async_indices: HashSet<PassIndex> = async_passes.iter()
            .map(|(p, _)| *p).collect();

        let expected_graphics: Vec<PassIndex> = order.iter()
            .filter(|idx| !async_indices.contains(idx))
            .copied()
            .collect();
        let expected_async: Vec<PassIndex> = order.iter()
            .filter(|idx| async_indices.contains(idx))
            .copied()
            .collect();

        assert_eq!(
            plan.graphics_queue, expected_graphics,
            "{}: graphics_queue preserves relative order of non-async passes",
            label
        );
        let compute_queue = plan.async_queues.get("compute")
            .unwrap_or_else(|| panic!("{}: compute queue must exist", label));
        assert_eq!(
            compute_queue, &expected_async,
            "{}: compute queue preserves relative order of async passes",
            label
        );
    }
}

#[test]
fn order_preservation_alternating_single_pattern() {
    // Strictly alternating async/non-async: A, G, A, G, A, G, ...
    // Ensures no off-by-one or fencepost errors in partition logic.
    let size = 20;
    let order: Vec<PassIndex> = (0..size).map(PassIndex).collect();
    let async_passes: Vec<(PassIndex, String)> = (0..size)
        .filter(|i| i % 2 == 0)
        .map(|i| (PassIndex(i), "compute".to_string()))
        .collect();

    let plan = build_async_plan(&order, &async_passes);

    // Graphics: odd indices in input order
    let expected_graphics: Vec<PassIndex> = (0..size)
        .filter(|i| i % 2 != 0)
        .map(PassIndex)
        .collect();
    assert_eq!(
        plan.graphics_queue, expected_graphics,
        "alternating pattern: graphics_queue must contain odd-indexed passes"
    );

    // Async (compute): even indices in input order
    let expected_async: Vec<PassIndex> = (0..size)
        .filter(|i| i % 2 == 0)
        .map(PassIndex)
        .collect();
    let compute_queue = plan.async_queues.get("compute")
        .expect("alternating pattern: compute queue must exist");
    assert_eq!(
        compute_queue, &expected_async,
        "alternating pattern: compute queue must contain even-indexed passes"
    );
}

#[test]
fn order_preservation_two_async_types_interleaved() {
    // Two async queue types interleaved: compute, copy, compute, copy, ...
    // Verifies each queue independently preserves input order.
    let order: Vec<PassIndex> = (0..10).map(PassIndex).collect();
    let async_passes = vec![
        (PassIndex(0), "compute".to_string()),
        (PassIndex(1), "copy".to_string()),
        (PassIndex(2), "compute".to_string()),
        (PassIndex(3), "copy".to_string()),
        (PassIndex(4), "compute".to_string()),
        (PassIndex(5), "copy".to_string()),
        (PassIndex(6), "compute".to_string()),
        (PassIndex(7), "copy".to_string()),
        (PassIndex(8), "compute".to_string()),
        (PassIndex(9), "copy".to_string()),
    ];

    let plan = build_async_plan(&order, &async_passes);

    // Compute queue: all even indices in order: 0, 2, 4, 6, 8
    let expected_compute: Vec<PassIndex> = (0..10)
        .filter(|i| i % 2 == 0)
        .map(PassIndex)
        .collect();
    let compute_queue = plan.async_queues.get("compute")
        .expect("interleaved: compute queue must exist");
    assert_eq!(
        compute_queue, &expected_compute,
        "interleaved: compute queue preserves order of even-indexed passes"
    );

    // Copy queue: all odd indices in order: 1, 3, 5, 7, 9
    let expected_copy: Vec<PassIndex> = (0..10)
        .filter(|i| i % 2 != 0)
        .map(PassIndex)
        .collect();
    let copy_queue = plan.async_queues.get("copy")
        .expect("interleaved: copy queue must exist");
    assert_eq!(
        copy_queue, &expected_copy,
        "interleaved: copy queue preserves order of odd-indexed passes"
    );

    // Graphics queue is empty (all passes are async)
    assert!(plan.graphics_queue.is_empty());
}

#[test]
fn order_preservation_interleaved_with_graphics() {
    // Three-way interleaving: compute, graphics, copy, graphics, compute, graphics, ...
    // Tests that the pass-through (graphics_queue) order is correct when
    // async passes of different types are mixed with non-async passes.
    let order = vec![
        PassIndex(0), PassIndex(1), PassIndex(2), PassIndex(3),
        PassIndex(4), PassIndex(5), PassIndex(6), PassIndex(7),
    ];
    let async_passes = vec![
        (PassIndex(0), "compute".to_string()),
        (PassIndex(2), "copy".to_string()),
        (PassIndex(4), "compute".to_string()),
        (PassIndex(6), "copy".to_string()),
    ];

    let plan = build_async_plan(&order, &async_passes);

    // Graphics: non-async passes in input order: 1, 3, 5, 7
    assert_eq!(
        plan.graphics_queue,
        vec![PassIndex(1), PassIndex(3), PassIndex(5), PassIndex(7)],
        "graphics_queue must preserve order of non-async passes"
    );

    // Compute: passes 0, 4
    assert_eq!(
        plan.async_queues.get("compute"),
        Some(&vec![PassIndex(0), PassIndex(4)]),
        "compute queue must contain passes 0 and 4 in order"
    );

    // Copy: passes 2, 6
    assert_eq!(
        plan.async_queues.get("copy"),
        Some(&vec![PassIndex(2), PassIndex(6)]),
        "copy queue must contain passes 2 and 6 in order"
    );
}

// =============================================================================
// SECTION 3 -- Edge cases: extreme and degenerate inputs
// =============================================================================

#[test]
fn edge_pass_index_usize_max() {
    // The function must handle the maximum possible PassIndex value
    // without panicking or producing incorrect results.
    let order = vec![PassIndex(usize::MAX), PassIndex(0), PassIndex(1)];
    let async_passes = vec![
        (PassIndex(usize::MAX), "compute".to_string()),
    ];

    let plan = build_async_plan(&order, &async_passes);

    // usize::MAX goes to compute, 0 and 1 go to graphics in order
    assert_eq!(
        plan.graphics_queue,
        vec![PassIndex(0), PassIndex(1)],
        "graphics_queue must contain passes 0 and 1"
    );
    assert_eq!(
        plan.async_queues.get("compute"),
        Some(&vec![PassIndex(usize::MAX)]),
        "compute queue must contain the usize::MAX pass"
    );

    // is_async_pass must agree
    assert!(
        is_async_pass(PassIndex(usize::MAX), &async_passes),
        "is_async_pass must return true for usize::MAX"
    );
    assert!(
        !is_async_pass(PassIndex(0), &async_passes),
        "is_async_pass must return false for pass 0"
    );
}

#[test]
fn edge_usize_max_and_large_gap_in_order() {
    // Pass indices with extremely large gaps (e.g., MAX-1, 0, MAX, 1).
    // Tests that the partition logic does not rely on contiguity.
    let order = vec![
        PassIndex(usize::MAX - 1),
        PassIndex(0),
        PassIndex(usize::MAX),
        PassIndex(1),
    ];
    let async_passes = vec![
        (PassIndex(usize::MAX - 1), "compute".to_string()),
        (PassIndex(usize::MAX), "copy".to_string()),
    ];

    let plan = build_async_plan(&order, &async_passes);

    assert_eq!(
        plan.graphics_queue,
        vec![PassIndex(0), PassIndex(1)],
        "large gaps: non-async passes 0 and 1 must go to graphics_queue"
    );
    assert_eq!(
        plan.async_queues.get("compute"),
        Some(&vec![PassIndex(usize::MAX - 1)]),
        "large gaps: MAX-1 goes to compute"
    );
    assert_eq!(
        plan.async_queues.get("copy"),
        Some(&vec![PassIndex(usize::MAX)]),
        "large gaps: MAX goes to copy"
    );

    // Total count must match input
    let total = plan.graphics_queue.len()
        + plan.async_queues.values().map(|v| v.len()).sum::<usize>();
    assert_eq!(total, 4, "total passes must match input order length (4)");
}

#[test]
fn edge_all_passes_same_index_in_order() {
    // Duplicate PassIndex values in the order itself.
    let order = vec![
        PassIndex(0), PassIndex(0), PassIndex(0),
    ];
    let async_passes = vec![
        (PassIndex(0), "compute".to_string()),
    ];

    let plan = build_async_plan(&order, &async_passes);

    // All three entries for pass 0 are listed in async_passes,
    // so all should be removed from graphics_queue.
    assert!(
        plan.graphics_queue.is_empty(),
        "graphics_queue must be empty when the only pass index is async"
    );

    let compute_queue = plan.async_queues.get("compute")
        .expect("compute queue must exist");
    // Each occurrence in order produces an entry in the async queue.
    assert_eq!(
        compute_queue.len(),
        3,
        "each duplicate occurrence of pass 0 in order must produce an entry"
    );
    for entry in compute_queue {
        assert_eq!(*entry, PassIndex(0));
    }
}

#[test]
fn edge_async_passes_not_in_order_some_async_present() {
    // async_passes contains entries for passes NOT in the order
    // alongside entries FOR passes in the order.
    let order = vec![PassIndex(1), PassIndex(2)];
    let async_passes = vec![
        (PassIndex(0), "compute".to_string()),  // not in order
        (PassIndex(1), "compute".to_string()),  // in order
        (PassIndex(99), "copy".to_string()),     // not in order
    ];

    let plan = build_async_plan(&order, &async_passes);

    // Pass 1 is async -> compute queue
    // Pass 2 is not in async_passes -> graphics_queue
    // Passes 0 and 99 are not in order -> ignored

    assert_eq!(
        plan.graphics_queue,
        vec![PassIndex(2)],
        "graphics_queue must contain only non-async pass 2"
    );
    assert_eq!(
        plan.async_queues.get("compute"),
        Some(&vec![PassIndex(1)]),
        "compute queue must contain pass 1 (in both order and async_passes)"
    );

    // No copy queue because pass 99 was ignored
    assert!(
        !plan.async_queues.contains_key("copy"),
        "copy queue must not be created when the only copy pass is not in the order"
    );

    let total = plan.graphics_queue.len()
        + plan.async_queues.values().map(|v| v.len()).sum::<usize>();
    assert_eq!(total, 2, "total must equal order.len() which is 2");
}

// =============================================================================
// SECTION 4 -- Determinism: identical inputs produce identical outputs
// =============================================================================

#[test]
fn determinism_same_input_twice() {
    // build_async_plan must be a pure function -- same inputs always
    // produce the same outputs.
    let order = vec![
        PassIndex(5), PassIndex(3), PassIndex(1),
        PassIndex(4), PassIndex(2), PassIndex(0),
    ];
    let async_passes = vec![
        (PassIndex(5), "compute".to_string()),
        (PassIndex(1), "copy".to_string()),
        (PassIndex(0), "compute".to_string()),
    ];

    let plan_a = build_async_plan(&order, &async_passes);
    let plan_b = build_async_plan(&order, &async_passes);

    assert_eq!(
        plan_a.graphics_queue, plan_b.graphics_queue,
        "build_async_plan must be deterministic: graphics_queue mismatch"
    );
    assert_eq!(
        plan_a.async_queues, plan_b.async_queues,
        "build_async_plan must be deterministic: async_queues mismatch"
    );
}

#[test]
fn determinism_large_input_stable() {
    // Determinism under larger load -- ensure no internal randomization
    // or allocation-dependent ordering.
    let order: Vec<PassIndex> = (0..1000).map(PassIndex).collect();
    let async_passes: Vec<(PassIndex, String)> = (0..1000)
        .filter(|i| i % 7 == 0 || i % 13 == 0)
        .map(|i| (PassIndex(i), "compute".to_string()))
        .collect();

    let plan_a = build_async_plan(&order, &async_passes);
    let plan_b = build_async_plan(&order, &async_passes);

    assert_eq!(
        plan_a.graphics_queue, plan_b.graphics_queue,
        "large-input determinism: graphics_queue mismatch"
    );
    assert_eq!(
        plan_a.async_queues, plan_b.async_queues,
        "large-input determinism: async_queues mismatch"
    );

    // Additionally verify that the total counts are consistent
    let total = plan_a.graphics_queue.len()
        + plan_a.async_queues.values().map(|v| v.len()).sum::<usize>();
    assert_eq!(total, order.len(), "total pass count must match order length");
}

// =============================================================================
// SECTION 5 -- 10K-order stress test
// =============================================================================

#[test]
fn stress_large_order_all_non_async() {
    // A 10,000-element order with no async passes.
    let order: Vec<PassIndex> = (0..10_000).map(PassIndex).collect();
    let async_passes: Vec<(PassIndex, String)> = vec![];

    let plan = build_async_plan(&order, &async_passes);

    assert_eq!(
        plan.graphics_queue.len(),
        10_000,
        "all 10K non-async passes go to graphics_queue"
    );
    assert_eq!(
        plan.graphics_queue, order,
        "graphics_queue must exactly match input order when no async passes"
    );
    assert!(
        plan.async_queues.is_empty(),
        "async_queues must be empty when no async passes"
    );
}

#[test]
fn stress_large_order_half_async() {
    // 10,000-element order, every even index is async (compute),
    // every odd index is async (copy).
    let order: Vec<PassIndex> = (0..10_000).map(PassIndex).collect();
    let async_passes: Vec<(PassIndex, String)> = (0..10_000)
        .map(|i| {
            let qtype = if i % 2 == 0 { "compute" } else { "copy" };
            (PassIndex(i), qtype.to_string())
        })
        .collect();

    let plan = build_async_plan(&order, &async_passes);

    // All passes are async, so graphics_queue is empty
    assert!(
        plan.graphics_queue.is_empty(),
        "graphics_queue must be empty when all 10K passes are async"
    );
    assert_eq!(
        plan.async_queues.len(),
        2,
        "two async queue types: compute and copy"
    );

    // compute: even indices in order
    let expected_compute: Vec<PassIndex> = (0..10_000)
        .filter(|i| i % 2 == 0)
        .map(PassIndex)
        .collect();
    assert_eq!(
        plan.async_queues.get("compute"),
        Some(&expected_compute),
        "compute queue preserves order of all 5000 even-indexed passes"
    );

    // copy: odd indices in order
    let expected_copy: Vec<PassIndex> = (0..10_000)
        .filter(|i| i % 2 != 0)
        .map(PassIndex)
        .collect();
    assert_eq!(
        plan.async_queues.get("copy"),
        Some(&expected_copy),
        "copy queue preserves order of all 5000 odd-indexed passes"
    );
}

#[test]
fn stress_large_order_sparse_async() {
    // 10K order, only 3 async passes at sparse positions (beginning,
    // middle, end). Tests that the bulk of passes remain in graphics_queue.
    let order: Vec<PassIndex> = (0..10_000).map(PassIndex).collect();
    let async_passes = vec![
        (PassIndex(0), "compute".to_string()),
        (PassIndex(5_000), "compute".to_string()),
        (PassIndex(9_999), "compute".to_string()),
    ];

    let plan = build_async_plan(&order, &async_passes);

    assert_eq!(
        plan.graphics_queue.len(),
        9_997,
        "only 3 of 10K passes are async, graphics_queue must have 9997"
    );
    assert_eq!(
        plan.async_queues.get("compute"),
        Some(&vec![PassIndex(0), PassIndex(5_000), PassIndex(9_999)]),
        "sparse async passes must appear in compute queue in input order"
    );

    // Confirm the graphics_queue does NOT contain the 3 async indices
    assert!(!plan.graphics_queue.contains(&PassIndex(0)));
    assert!(!plan.graphics_queue.contains(&PassIndex(5_000)));
    assert!(!plan.graphics_queue.contains(&PassIndex(9_999)));
}

// =============================================================================
// SECTION 6 -- Queue type isolation: no cross-contamination
// =============================================================================

#[test]
fn queue_type_isolation_case_sensitivity() {
    // Queue type strings are case-sensitive: "Compute" != "compute".
    // Each should produce a separate queue entry.
    let order = vec![PassIndex(0), PassIndex(1), PassIndex(2)];
    let async_passes = vec![
        (PassIndex(0), "compute".to_string()),
        (PassIndex(1), "Compute".to_string()),
        (PassIndex(2), "COMPUTE".to_string()),
    ];

    let plan = build_async_plan(&order, &async_passes);

    assert_eq!(
        plan.async_queues.len(),
        3,
        "case-variant queue types produce separate queues"
    );
    assert!(plan.async_queues.contains_key("compute"));
    assert!(plan.async_queues.contains_key("Compute"));
    assert!(plan.async_queues.contains_key("COMPUTE"));
    assert!(plan.graphics_queue.is_empty());
}

#[test]
fn queue_type_isolation_whitespace() {
    // Whitespace-preserving type strings: leading/trailing/internal spaces.
    let order = vec![PassIndex(0), PassIndex(1), PassIndex(2), PassIndex(3)];
    let async_passes = vec![
        (PassIndex(0), "compute".to_string()),
        (PassIndex(1), " compute".to_string()),
        (PassIndex(2), "compute ".to_string()),
        (PassIndex(3), "com pute".to_string()),
    ];

    let plan = build_async_plan(&order, &async_passes);

    assert_eq!(plan.async_queues.len(), 4, "all four whitespace variants produce separate queues");
    assert_eq!(plan.async_queues.get("compute"), Some(&vec![PassIndex(0)]));
    assert_eq!(plan.async_queues.get(" compute"), Some(&vec![PassIndex(1)]));
    assert_eq!(plan.async_queues.get("compute "), Some(&vec![PassIndex(2)]));
    assert_eq!(plan.async_queues.get("com pute"), Some(&vec![PassIndex(3)]));

    assert!(plan.graphics_queue.is_empty());
}

#[test]
fn queue_type_isolation_copy_vs_compute() {
    // Mixed compute+copy with non-async passes: ensure each pass ends
    // up in exactly one queue and no queue contains the wrong type.
    let order: Vec<PassIndex> = (0..8).map(PassIndex).collect();
    let async_passes = vec![
        (PassIndex(0), "compute".to_string()),
        (PassIndex(2), "copy".to_string()),
        (PassIndex(4), "compute".to_string()),
        (PassIndex(6), "copy".to_string()),
    ];

    let plan = build_async_plan(&order, &async_passes);

    // Compute queue: only passes tagged "compute" (0, 4)
    let compute_queue = plan.async_queues.get("compute")
        .expect("compute queue must exist");
    for &p in compute_queue {
        let is_compute = async_passes.iter()
            .any(|&(ref idx, ref typ)| *idx == p && typ == "compute");
        assert!(
            is_compute,
            "compute queue must only contain passes tagged 'compute', found {:?}",
            p
        );
    }

    // Copy queue: only passes tagged "copy" (2, 6)
    let copy_queue = plan.async_queues.get("copy")
        .expect("copy queue must exist");
    for &p in copy_queue {
        let is_copy = async_passes.iter()
            .any(|&(ref idx, ref typ)| *idx == p && typ == "copy");
        assert!(
            is_copy,
            "copy queue must only contain passes tagged 'copy', found {:?}",
            p
        );
    }

    // Graphics queue: only passes NOT in async_passes (1, 3, 5, 7)
    for &p in &plan.graphics_queue {
        let is_async = async_passes.iter().any(|(idx, _)| *idx == p);
        assert!(
            !is_async,
            "graphics_queue must not contain any async pass, found {:?}",
            p
        );
    }
}

// =============================================================================
// SECTION 7 -- Edge cases: queue type string extremes
// =============================================================================

#[test]
fn edge_queue_type_empty_string() {
    // An empty-string queue type is valid as a HashMap<String, Vec<PassIndex>> key.
    let order = vec![PassIndex(0)];
    let async_passes = vec![(PassIndex(0), "".to_string())];

    let plan = build_async_plan(&order, &async_passes);

    assert!(
        plan.async_queues.contains_key(""),
        "empty-string queue type must create a queue entry"
    );
    assert_eq!(
        plan.async_queues.get(""),
        Some(&vec![PassIndex(0)]),
        "empty-string queue must contain pass 0"
    );
    assert!(
        plan.graphics_queue.is_empty(),
        "graphics_queue must be empty when the only pass has empty queue type"
    );
}

#[test]
fn edge_queue_type_long_string() {
    // Long queue type strings (~100 chars) must be handled.
    let long_type = "a".repeat(100);
    let order = vec![PassIndex(0), PassIndex(1)];
    let async_passes = vec![
        (PassIndex(0), long_type.clone()),
    ];

    let plan = build_async_plan(&order, &async_passes);

    assert!(
        plan.async_queues.contains_key(&long_type),
        "long queue type string must be preserved as a key"
    );
    assert_eq!(
        plan.async_queues.get(&long_type),
        Some(&vec![PassIndex(0)]),
        "long queue type must contain pass 0"
    );
    assert_eq!(
        plan.graphics_queue,
        vec![PassIndex(1)],
        "graphics_queue must contain the non-async pass 1"
    );
}

#[test]
fn edge_queue_type_special_characters() {
    // Queue type strings with special characters (unicode, symbols).
    let special_types = vec![
        "compute_01",
        "copy-queue",
        "async.queue.42",
        "主队列",
        "🖥️",
        "queue_with_underscores",
        "custom@type#2024",
    ];

    for (i, qtype) in special_types.iter().enumerate() {
        let order = vec![PassIndex(i as u32 as usize)];
        let async_passes = vec![
            (PassIndex(i as u32 as usize), qtype.to_string()),
        ];

        let plan = build_async_plan(&order, &async_passes);

        assert!(
            plan.async_queues.contains_key(*qtype),
            "queue type '{}' must be accepted as a key", qtype
        );
        assert_eq!(
            plan.async_queues.get(*qtype),
            Some(&vec![PassIndex(i as u32 as usize)]),
            "queue type '{}' must contain pass {}", qtype, i
        );
    }
}

// =============================================================================
// SECTION 8 -- Structural invariants and no-panic guarantee for wild inputs
// =============================================================================

#[test]
fn invariants_never_panic_wild_inputs() {
    // Exhaustive no-panic check for a variety of wild inputs.
    // The function must never panic for any well-typed input.
    let wild_orders: &[&[PassIndex]] = &[
        &[],
        &[PassIndex(0)],
        &[PassIndex(usize::MAX)],
        &[PassIndex(usize::MAX), PassIndex(0), PassIndex(usize::MAX)],
    ];

    let wild_async: &[&[(PassIndex, String)]] = &[
        &[],
        &[(PassIndex(0), "compute".to_string())],
        &[(PassIndex(usize::MAX), "".to_string())],
        &[
            (PassIndex(usize::MAX), "compute".to_string()),
            (PassIndex(usize::MAX), "copy".to_string()),
        ],
    ];

    for &order in wild_orders {
        for &async_list in wild_async {
            let plan = build_async_plan(order, async_list);

            // Basic structural invariants must always hold
            let total = plan.graphics_queue.len()
                + plan.async_queues.values().map(|v| v.len()).sum::<usize>();
            assert_eq!(
                total,
                order.len(),
                "total passes in queues must equal order length for input (order={:?}, async={:?})",
                order.len(),
                async_list.len()
            );

            // No pass appears in both graphics and async
            let all_async: HashSet<PassIndex> = plan.async_queues.values()
                .flat_map(|v| v.iter())
                .copied()
                .collect();
            for g in &plan.graphics_queue {
                assert!(
                    !all_async.contains(g),
                    "pass {:?} appears in both graphics_queue and async_queues", g
                );
            }
        }
    }
}

#[test]
fn invariants_all_queue_elements_from_passed_order() {
    // Every PassIndex in graphics_queue AND in every async_queue
    // must be an element of the input order array.
    let order = vec![PassIndex(10), PassIndex(20), PassIndex(30), PassIndex(40)];
    let async_passes = vec![
        (PassIndex(20), "compute".to_string()),
        (PassIndex(40), "copy".to_string()),
    ];

    let plan = build_async_plan(&order, &async_passes);

    for idx in &plan.graphics_queue {
        assert!(
            order.contains(idx),
            "graphics_queue pass {:?} must come from the input order", idx
        );
    }
    for (qtype, queue) in &plan.async_queues {
        for idx in queue {
            assert!(
                order.contains(idx),
                "async queue '{}' pass {:?} must come from the input order", qtype, idx
            );
        }
    }
}

#[test]
fn invariants_default_plan_is_empty() {
    // Manually construct an empty AsyncExecutionPlan since it does not derive Default.
    #[allow(unused_imports)]
    use std::collections::HashMap;
    let plan = AsyncExecutionPlan {
        graphics_queue: vec![],
        async_queues: HashMap::new(),
    };
    assert!(plan.graphics_queue.is_empty());
    assert!(plan.async_queues.is_empty());
}

#[test]
fn invariants_plan_from_empty_input_matches_default() {
    let plan = build_async_plan(&[], &[]);
    assert!(plan.graphics_queue.is_empty(), "graphics_queue should be empty");
    assert!(plan.async_queues.is_empty(), "async_queues should be empty");
}
