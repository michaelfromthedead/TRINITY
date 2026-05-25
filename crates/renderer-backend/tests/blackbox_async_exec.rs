// Blackbox contract tests for T-FG-5.1 (AsyncExecutionPlan).
//
// CLEANROOM: No src/ access beyond the public API exported by the crate.
// Tests use only `renderer_backend::frame_graph::{AsyncExecutionPlan,
// build_async_plan, is_async_pass, PassIndex}` -- no internal fields,
// no private methods, no implementation details.
//
// Contract under test:
//   AsyncExecutionPlan { graphics_queue, async_queues }
//   build_async_plan(order, async_passes) -> AsyncExecutionPlan
//   is_async_pass(pass_idx, async_passes) -> bool
//
//   Separates a topological pass order into a graphics queue (sequential,
//   main queue) and async compute/copy queues. Passes identified as
//   async-eligible are moved from the graphics queue into their respective
//   async queues, preserving relative order within each queue.
//
// Acceptance criteria:
//   1.  build_async_plan returns a plan with both graphics_queue and async_queues
//   2.  Passes NOT in async_passes go to graphics_queue
//   3.  Passes IN async_passes go to async_queues by their queue_type
//   4.  is_async_pass returns true for async passes, false for non-async passes
//   5.  Empty order and empty async_passes produce empty plan
//   6.  Relative order within each queue matches the input order
//   7.  Total pass count = graphics_queue + sum of all async_queues
//   8.  No pass appears in both graphics_queue and async_queues
//   9.  Multiple async queue types are handled (compute, copy)
//  10.  Duplicate pass index in async_passes is handled gracefully
//  11.  is_async_pass handles empty async_passes list
//  12.  All async_queues values are non-empty Vecs

use renderer_backend::frame_graph::{
    build_async_plan, is_async_pass, PassIndex,
};

// =============================================================================
// SECTION 1 -- Plan structure and field presence
// =============================================================================

#[test]
fn build_async_plan_returns_struct_with_both_fields() {
    let order = vec![PassIndex(0)];
    let async_passes: Vec<(PassIndex, String)> = vec![];

    let plan = build_async_plan(&order, &async_passes);

    // The struct has both public fields; verifying they exist and are correct
    // types is implicit in the type system. We confirm they are accessible.
    assert_eq!(plan.graphics_queue.len(), 1,
        "graphics_queue must contain the single non-async pass");
    assert_eq!(plan.async_queues.len(), 0,
        "async_queues must be empty when no async passes");
}

#[test]
fn async_queues_is_hashmap_with_string_keys() {
    let order = vec![PassIndex(0), PassIndex(1)];
    let async_passes = vec![(PassIndex(1), "compute".to_string())];

    let plan = build_async_plan(&order, &async_passes);

    // async_queues is HashMap<String, Vec<PassIndex>>
    assert_eq!(plan.async_queues.len(), 1,
        "must have one async queue entry");
    assert!(plan.async_queues.contains_key("compute"),
        "must contain a 'compute' key");
}

#[test]
fn graphics_queue_is_vec_of_pass_index() {
    let order = vec![PassIndex(0), PassIndex(1), PassIndex(2)];
    let async_passes = vec![(PassIndex(0), "compute".to_string())];

    let plan = build_async_plan(&order, &async_passes);

    // graphics_queue elements are PassIndex
    for (i, idx) in plan.graphics_queue.iter().enumerate() {
        let _: PassIndex = *idx; // type check
        assert_ne!(*idx, PassIndex(0),
            "graphics_queue[{}] must not contain async pass 0", i);
    }

    assert_eq!(plan.graphics_queue.len(), 2,
        "graphics_queue must contain 2 of 3 passes");
}

// =============================================================================
// SECTION 2 -- Pass partitioning: graphics vs async
// =============================================================================

#[test]
fn non_async_passes_go_to_graphics_queue() {
    let order = vec![PassIndex(0), PassIndex(1), PassIndex(2)];
    let async_passes = vec![(PassIndex(1), "compute".to_string())];

    let plan = build_async_plan(&order, &async_passes);

    // Passes 0 and 2 are NOT in async_passes => graphics_queue
    assert!(plan.graphics_queue.contains(&PassIndex(0)),
        "graphics_queue must contain non-async pass 0");
    assert!(plan.graphics_queue.contains(&PassIndex(2)),
        "graphics_queue must contain non-async pass 2");
    assert!(!plan.graphics_queue.contains(&PassIndex(1)),
        "graphics_queue must NOT contain async pass 1");
}

#[test]
fn async_passes_go_to_async_queues() {
    let order = vec![PassIndex(0), PassIndex(1), PassIndex(2)];
    let async_passes = vec![
        (PassIndex(0), "compute".to_string()),
        (PassIndex(2), "compute".to_string()),
    ];

    let plan = build_async_plan(&order, &async_passes);

    let compute_queue = plan.async_queues.get("compute")
        .expect("must have a compute async queue");

    assert!(compute_queue.contains(&PassIndex(0)),
        "compute queue must contain async pass 0");
    assert!(compute_queue.contains(&PassIndex(2)),
        "compute queue must contain async pass 2");
    assert!(!compute_queue.contains(&PassIndex(1)),
        "compute queue must NOT contain non-async pass 1");
}

#[test]
fn async_passes_removed_from_graphics_queue() {
    let order = vec![PassIndex(0), PassIndex(1), PassIndex(2), PassIndex(3)];
    let async_passes = vec![
        (PassIndex(0), "compute".to_string()),
        (PassIndex(2), "copy".to_string()),
    ];

    let plan = build_async_plan(&order, &async_passes);

    // Graphics queue must NOT contain any async pass
    for idx in plan.graphics_queue.iter() {
        assert!(!is_async_pass(*idx, &async_passes),
            "graphics_queue must not contain async pass {:?}", idx);
    }

    // Graphics queue contains only the non-async passes (1 and 3)
    assert_eq!(plan.graphics_queue.len(), 2,
        "graphics_queue should have 2 of 4 passes");
    assert!(plan.graphics_queue.contains(&PassIndex(1)));
    assert!(plan.graphics_queue.contains(&PassIndex(3)));
}

#[test]
fn all_passes_are_accounted_for() {
    let order = vec![
        PassIndex(0), PassIndex(1), PassIndex(2),
        PassIndex(3), PassIndex(4),
    ];
    let async_passes = vec![
        (PassIndex(1), "compute".to_string()),
        (PassIndex(3), "copy".to_string()),
    ];

    let plan = build_async_plan(&order, &async_passes);

    let graphics_count = plan.graphics_queue.len();
    let async_count: usize = plan.async_queues.values()
        .map(|v| v.len())
        .sum();

    assert_eq!(graphics_count + async_count, order.len(),
        "total passes in queues ({} graphics + {} async) must equal input order length ({})",
        graphics_count, async_count, order.len());
}

#[test]
fn no_pass_appears_in_both_graphics_and_async() {
    let order = vec![
        PassIndex(0), PassIndex(1), PassIndex(2),
        PassIndex(3), PassIndex(4),
    ];
    let async_passes = vec![
        (PassIndex(1), "compute".to_string()),
        (PassIndex(3), "copy".to_string()),
    ];

    let plan = build_async_plan(&order, &async_passes);

    // Collect all async pass indices into a single set.
    let all_async: Vec<PassIndex> = plan.async_queues.values()
        .flat_map(|v| v.iter())
        .copied()
        .collect();

    // No graphics pass should appear in the async set.
    for g_idx in &plan.graphics_queue {
        assert!(!all_async.contains(g_idx),
            "pass {:?} must not appear in both graphics_queue and async_queues", g_idx);
    }
}

// =============================================================================
// SECTION 3 -- is_async_pass
// =============================================================================

#[test]
fn is_async_pass_returns_true_for_async_passes() {
    let async_passes = vec![
        (PassIndex(0), "compute".to_string()),
        (PassIndex(2), "copy".to_string()),
    ];

    assert!(is_async_pass(PassIndex(0), &async_passes),
        "pass 0 is in async_passes => must return true");
    assert!(is_async_pass(PassIndex(2), &async_passes),
        "pass 2 is in async_passes => must return true");
}

#[test]
fn is_async_pass_returns_false_for_non_async_passes() {
    let async_passes = vec![
        (PassIndex(1), "compute".to_string()),
    ];

    assert!(!is_async_pass(PassIndex(0), &async_passes),
        "pass 0 is NOT in async_passes => must return false");
    assert!(!is_async_pass(PassIndex(2), &async_passes),
        "pass 2 is NOT in async_passes => must return false");
}

#[test]
fn is_async_pass_handles_empty_list() {
    let async_passes: Vec<(PassIndex, String)> = vec![];

    assert!(!is_async_pass(PassIndex(0), &async_passes),
        "empty async_passes => must return false for any pass");
    assert!(!is_async_pass(PassIndex(42), &async_passes),
        "empty async_passes => must return false for any pass");
}

#[test]
fn is_async_pass_ignores_queue_type_when_checking() {
    // The function checks only pass_index, not the queue_type string.
    let async_passes = vec![
        (PassIndex(0), "compute".to_string()),
        (PassIndex(1), "copy".to_string()),
        (PassIndex(2), "custom_queue".to_string()),
    ];

    assert!(is_async_pass(PassIndex(0), &async_passes));
    assert!(is_async_pass(PassIndex(1), &async_passes));
    assert!(is_async_pass(PassIndex(2), &async_passes));
    assert!(!is_async_pass(PassIndex(3), &async_passes));
}

// =============================================================================
// SECTION 4 -- Empty input
// =============================================================================

#[test]
fn empty_order_produces_empty_plan() {
    let order: Vec<PassIndex> = vec![];
    let async_passes: Vec<(PassIndex, String)> = vec![];

    let plan = build_async_plan(&order, &async_passes);

    assert_eq!(plan.graphics_queue.len(), 0,
        "empty order => empty graphics_queue");
    assert_eq!(plan.async_queues.len(), 0,
        "empty order => empty async_queues");
}

#[test]
fn empty_async_passes_with_non_empty_order_puts_all_in_graphics() {
    let order = vec![PassIndex(0), PassIndex(1), PassIndex(2)];
    let async_passes: Vec<(PassIndex, String)> = vec![];

    let plan = build_async_plan(&order, &async_passes);

    assert_eq!(plan.graphics_queue.len(), 3,
        "all 3 passes must go to graphics_queue");
    assert_eq!(plan.async_queues.len(), 0,
        "no async_queues when async_passes is empty");

    // Order preserved: every pass index present.
    assert!(plan.graphics_queue.contains(&PassIndex(0)));
    assert!(plan.graphics_queue.contains(&PassIndex(1)));
    assert!(plan.graphics_queue.contains(&PassIndex(2)));
}

#[test]
fn all_passes_async_produces_empty_graphics_queue() {
    let order = vec![PassIndex(0), PassIndex(1), PassIndex(2)];
    let async_passes = vec![
        (PassIndex(0), "compute".to_string()),
        (PassIndex(1), "compute".to_string()),
        (PassIndex(2), "copy".to_string()),
    ];

    let plan = build_async_plan(&order, &async_passes);

    assert_eq!(plan.graphics_queue.len(), 0,
        "all passes async => empty graphics_queue");
    assert_eq!(
        plan.async_queues.values().map(|v| v.len()).sum::<usize>(),
        order.len(),
        "all passes must be distributed across async_queues"
    );
}

// =============================================================================
// SECTION 5 -- Order preservation
// =============================================================================

#[test]
fn graphics_queue_preserves_input_order() {
    let order = vec![
        PassIndex(3), PassIndex(0), PassIndex(2), PassIndex(1),
    ];
    let async_passes = vec![(PassIndex(0), "compute".to_string())];

    let plan = build_async_plan(&order, &async_passes);

    // Non-async passes in order: 3, 2, 1 (0 is async).
    assert_eq!(plan.graphics_queue.len(), 3);
    assert_eq!(plan.graphics_queue[0], PassIndex(3),
        "graphics_queue[0] must be pass 3 (first non-async in input order)");
    assert_eq!(plan.graphics_queue[1], PassIndex(2),
        "graphics_queue[1] must be pass 2 (second non-async in input order)");
    assert_eq!(plan.graphics_queue[2], PassIndex(1),
        "graphics_queue[2] must be pass 1 (third non-async in input order)");
}

#[test]
fn async_queue_preserves_input_order() {
    let order = vec![
        PassIndex(0), PassIndex(1), PassIndex(2), PassIndex(3),
        PassIndex(4),
    ];
    let async_passes = vec![
        (PassIndex(1), "compute".to_string()),
        (PassIndex(3), "compute".to_string()),
    ];

    let plan = build_async_plan(&order, &async_passes);

    let compute_queue = plan.async_queues.get("compute")
        .expect("must have compute queue");

    // Async passes appear in input order: 1 then 3.
    assert_eq!(compute_queue.len(), 2);
    assert_eq!(compute_queue[0], PassIndex(1),
        "compute_queue[0] must be pass 1 (first async in input order)");
    assert_eq!(compute_queue[1], PassIndex(3),
        "compute_queue[1] must be pass 3 (second async in input order)");
}

#[test]
fn interleaved_async_passes_preserve_relative_order() {
    // Mixed order: async and non-async passes interleaved.
    let order = vec![
        PassIndex(5), PassIndex(2), PassIndex(7),
        PassIndex(1), PassIndex(3),
    ];
    let async_passes = vec![
        (PassIndex(2), "compute".to_string()),
        (PassIndex(1), "compute".to_string()),
    ];

    let plan = build_async_plan(&order, &async_passes);

    // graphics_queue: 5, 7, 3 (non-async in input order).
    assert_eq!(plan.graphics_queue, vec![
        PassIndex(5), PassIndex(7), PassIndex(3),
    ], "graphics_queue must preserve relative order of non-async passes");

    // compute_queue: 2, 1 (async in input order).
    let compute_queue = plan.async_queues.get("compute")
        .expect("must have compute queue");
    assert_eq!(compute_queue, &vec![
        PassIndex(2), PassIndex(1),
    ], "compute_queue must preserve relative order of async passes");
}

// =============================================================================
// SECTION 6 -- Multiple async queue types (compute and copy)
// =============================================================================

#[test]
fn compute_and_copy_queues_are_separate() {
    let order = vec![
        PassIndex(0), PassIndex(1), PassIndex(2), PassIndex(3),
    ];
    let async_passes = vec![
        (PassIndex(0), "compute".to_string()),
        (PassIndex(2), "copy".to_string()),
    ];

    let plan = build_async_plan(&order, &async_passes);

    assert!(plan.async_queues.contains_key("compute"),
        "must have compute async queue");
    assert!(plan.async_queues.contains_key("copy"),
        "must have copy async queue");

    // Pass 0 goes to compute, pass 2 goes to copy.
    let compute_queue = plan.async_queues.get("compute").unwrap();
    let copy_queue = plan.async_queues.get("copy").unwrap();

    assert_eq!(compute_queue, &vec![PassIndex(0)],
        "compute queue must contain pass 0");
    assert_eq!(copy_queue, &vec![PassIndex(2)],
        "copy queue must contain pass 2");

    // graphics_queue gets the rest (1, 3).
    assert_eq!(plan.graphics_queue, vec![PassIndex(1), PassIndex(3)]);
}

#[test]
fn multiple_passes_in_same_async_queue_type() {
    let order = vec![
        PassIndex(0), PassIndex(1), PassIndex(2), PassIndex(3),
        PassIndex(4),
    ];
    let async_passes = vec![
        (PassIndex(0), "compute".to_string()),
        (PassIndex(2), "compute".to_string()),
        (PassIndex(4), "compute".to_string()),
    ];

    let plan = build_async_plan(&order, &async_passes);

    let compute_queue = plan.async_queues.get("compute")
        .expect("must have compute queue");

    assert_eq!(compute_queue.len(), 3,
        "all three compute passes must be in the same queue");
    assert_eq!(compute_queue[0], PassIndex(0));
    assert_eq!(compute_queue[1], PassIndex(2));
    assert_eq!(compute_queue[2], PassIndex(4));
}

#[test]
fn three_async_queue_types() {
    let order = vec![
        PassIndex(0), PassIndex(1), PassIndex(2), PassIndex(3),
        PassIndex(4),
    ];
    let async_passes = vec![
        (PassIndex(0), "compute".to_string()),
        (PassIndex(2), "copy".to_string()),
        (PassIndex(4), "transfer".to_string()),
    ];

    let plan = build_async_plan(&order, &async_passes);

    assert_eq!(plan.async_queues.len(), 3,
        "must have 3 separate async queue entries");

    assert!(plan.async_queues.contains_key("compute"),
        "must have compute queue");
    assert!(plan.async_queues.contains_key("copy"),
        "must have copy queue");
    assert!(plan.async_queues.contains_key("transfer"),
        "must have transfer queue");

    // Each queue has exactly one pass.
    for queue in plan.async_queues.values() {
        assert_eq!(queue.len(), 1);
    }
}

// =============================================================================
// SECTION 7 -- Edge cases: single pass
// =============================================================================

#[test]
fn single_non_async_pass_goes_to_graphics_queue() {
    let order = vec![PassIndex(0)];
    let async_passes: Vec<(PassIndex, String)> = vec![];

    let plan = build_async_plan(&order, &async_passes);

    assert_eq!(plan.graphics_queue, vec![PassIndex(0)],
        "single non-async pass must go to graphics_queue");
    assert!(plan.async_queues.is_empty(),
        "no async queues for non-async pass");
}

#[test]
fn single_async_pass_goes_to_async_queue() {
    let order = vec![PassIndex(0)];
    let async_passes = vec![(PassIndex(0), "compute".to_string())];

    let plan = build_async_plan(&order, &async_passes);

    assert!(plan.graphics_queue.is_empty(),
        "graphics_queue must be empty when only pass is async");

    let compute_queue = plan.async_queues.get("compute")
        .expect("must have compute queue");
    assert_eq!(compute_queue, &vec![PassIndex(0)],
        "async pass 0 must be in compute queue");
}

// =============================================================================
// SECTION 8 -- Duplicate and edge-case async_passes entries
// =============================================================================

#[test]
fn duplicate_async_pass_entry() {
    let order = vec![PassIndex(0), PassIndex(1)];
    // Duplicate entry: pass 0 appears twice in async_passes.
    let async_passes = vec![
        (PassIndex(0), "compute".to_string()),
        (PassIndex(0), "compute".to_string()),
    ];

    let plan = build_async_plan(&order, &async_passes);

    // Pass 0 should appear in compute queue (likely once, but the contract
    // says it's removed from graphics). The implementation iterates order
    // and checks async_set membership built from async_passes.
    assert!(!plan.graphics_queue.contains(&PassIndex(0)),
        "duplicate async pass 0 must still be removed from graphics_queue");

    let compute_queue = plan.async_queues.get("compute");
    if let Some(queue) = compute_queue {
        // The exact count depends on whether duplicates are de-aliased by the set.
        // The key contract: pass 0 is NOT in graphics_queue.
        assert!(queue.contains(&PassIndex(0)),
            "pass 0 must appear in compute queue");
    } else {
        panic!("compute queue must exist when async_passes has compute entries");
    }
}

#[test]
fn async_pass_with_empty_queue_type_string() {
    let order = vec![PassIndex(0), PassIndex(1)];
    let async_passes = vec![(PassIndex(0), "".to_string())];

    let plan = build_async_plan(&order, &async_passes);

    // The empty-string queue type is valid (it is a String key).
    assert!(plan.async_queues.contains_key(""),
        "empty-string queue type must produce a queue entry");
    assert!(!plan.graphics_queue.contains(&PassIndex(0)),
        "async pass with empty queue type removed from graphics_queue");
}

// =============================================================================
// SECTION 9 -- Non-contiguous and out-of-range pass indices
// =============================================================================

#[test]
fn non_contiguous_pass_indices() {
    let order = vec![PassIndex(10), PassIndex(20), PassIndex(30)];
    let async_passes = vec![(PassIndex(20), "compute".to_string())];

    let plan = build_async_plan(&order, &async_passes);

    assert_eq!(plan.graphics_queue, vec![PassIndex(10), PassIndex(30)],
        "non-contiguous non-async passes in graphics_queue");
    assert_eq!(
        plan.async_queues.get("compute").unwrap(),
        &vec![PassIndex(20)],
        "non-contiguous async pass 20 in compute queue"
    );
}

// =============================================================================
// SECTION 10 -- Structural consistency across all tests
// =============================================================================

#[test]
fn async_queues_values_are_non_empty_vectors() {
    let order = vec![PassIndex(0), PassIndex(1), PassIndex(2)];
    let async_passes = vec![
        (PassIndex(0), "compute".to_string()),
        (PassIndex(2), "copy".to_string()),
    ];

    let plan = build_async_plan(&order, &async_passes);

    for (queue_type, passes) in &plan.async_queues {
        assert!(!passes.is_empty(),
            "async queue '{}' must have at least one pass", queue_type);
    }
}

#[test]
fn all_queue_elements_are_valid_pass_indices() {
    let order = vec![PassIndex(0), PassIndex(1), PassIndex(2)];
    let async_passes = vec![
        (PassIndex(1), "compute".to_string()),
    ];

    let plan = build_async_plan(&order, &async_passes);

    // All pass indices in graphics_queue come from the original order.
    for idx in &plan.graphics_queue {
        assert!(order.contains(idx),
            "graphics_queue pass {:?} must be from the original order", idx);
    }

    // All pass indices in async_queues come from the original order.
    for (queue_type, queue) in &plan.async_queues {
        for idx in queue {
            assert!(order.contains(idx),
                "async queue '{}' pass {:?} must be from the original order",
                queue_type, idx);
        }
    }
}

#[test]
fn build_async_plan_never_panics() {
    // Must never panic for any well-typed input.
    let plan = build_async_plan(&[], &[]);
    assert_eq!(plan.graphics_queue.len(), 0);

    let plan = build_async_plan(&[PassIndex(0)], &[]);
    assert_eq!(plan.graphics_queue.len(), 1);

    let plan = build_async_plan(&[PassIndex(0), PassIndex(1)], &[
        (PassIndex(0), "compute".to_string()),
        (PassIndex(1), "copy".to_string()),
    ]);
    assert_eq!(plan.graphics_queue.len(), 0);
}
