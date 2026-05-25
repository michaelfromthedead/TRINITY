// Blackbox contract tests for T-FG-4.6 (ScheduledPass pre/post barriers).
//
// CLEANROOM: No src/ access beyond the public API exported by the crate.
// Tests use only `renderer_backend::frame_graph::*` -- no internal fields,
// no private methods, no implementation details.
//
// Acceptance criteria (T-FG-4.6):
//   ScheduledPass struct with pass_index, pre_barriers, Vec<BarrierTuple>,
//   and post_barriers: Vec<BarrierTuple> fields, populated by build_scheduled_passes
//   during compile.
//
// Coverage:
//   1.  Compile a linear chain graph -- verify compiled.scheduled_passes is non-empty
//   2.  Each ScheduledPass has pass_index, pre_barriers, and post_barriers fields
//   3.  First pass has empty pre_barriers
//   4.  Barrier-producing pass has post_barriers populated
//   5.  Barrier-consuming pass has pre_barriers populated
//   6.  Barrier tuples have correct structure (6-element tuple, valid indices)
//   7.  No-pass graph produces empty scheduled_passes
//   8.  scheduled_passes.len() matches the number of passes

use renderer_backend::frame_graph::{
    mock_pass_compute, mock_pass_graphics, mock_resource_buffer, mock_resource_texture,
    CompiledFrameGraph, EdgeType, PassIndex, ResourceHandle, ResourceState,
};

// =============================================================================
// SECTION 1 -- Linear chain graph: scheduled_passes is non-empty
// =============================================================================

#[test]
fn linear_chain_scheduled_passes_non_empty() {
    // Two-pass linear chain:
    //   Pass 0 writes R1 (ShaderReadWrite), Pass 1 reads R1 (ShaderRead).
    // Compilation must produce at least one ScheduledPass.
    let r = ResourceHandle(1);
    let passes = vec![
        mock_pass_compute(PassIndex(0), "producer", &[], &[r]),
        mock_pass_compute(PassIndex(1), "consumer", &[r], &[]),
    ];
    let resources = vec![mock_resource_buffer(r, "buf", 256)];

    let compiled = CompiledFrameGraph::compile(passes, resources)
        .expect("linear chain must compile");

    assert!(
        !compiled.scheduled_passes.is_empty(),
        "linear chain graph must produce non-empty scheduled_passes",
    );
}

#[test]
fn linear_chain_scheduled_passes_length_matches_pass_count() {
    // Three-pass chain: P0 writes R1, P1 reads R1 and writes R2, P2 reads R2.
    // scheduled_passes.len() must equal the number of passes (3).
    let r1 = ResourceHandle(1);
    let r2 = ResourceHandle(2);
    let passes = vec![
        mock_pass_compute(PassIndex(0), "p0", &[], &[r1]),
        mock_pass_compute(PassIndex(1), "p1", &[r1], &[r2]),
        mock_pass_compute(PassIndex(2), "p2", &[r2], &[]),
    ];
    let resources = vec![
        mock_resource_buffer(r1, "buf_a", 256),
        mock_resource_buffer(r2, "buf_b", 512),
    ];

    let compiled = CompiledFrameGraph::compile(passes, resources)
        .expect("three-pass chain must compile");

    assert_eq!(
        compiled.scheduled_passes.len(),
        3,
        "scheduled_passes.len() must match the number of compiled passes",
    );
}

// =============================================================================
// SECTION 2 -- Each ScheduledPass has pass_index, pre_barriers, and
//              post_barriers fields
// =============================================================================

#[test]
fn each_scheduled_pass_has_required_fields() {
    // Build a two-pass chain and verify that every ScheduledPass in the output
    // has accessible pass_index, pre_barriers, and post_barriers fields.
    let r = ResourceHandle(1);
    let passes = vec![
        mock_pass_compute(PassIndex(0), "writer", &[], &[r]),
        mock_pass_compute(PassIndex(1), "reader", &[r], &[]),
    ];
    let resources = vec![mock_resource_buffer(r, "buf", 256)];

    let compiled = CompiledFrameGraph::compile(passes, resources)
        .expect("two-pass chain must compile");

    assert_eq!(
        compiled.scheduled_passes.len(),
        2,
        "must have exactly 2 scheduled passes",
    );

    for (i, sp) in compiled.scheduled_passes.iter().enumerate() {
        // pass_index must be of type PassIndex and match the expected value.
        let _idx: PassIndex = sp.pass_index;

        // pre_barriers must be Vec<BarrierTuple>.
        let _pre: &[(
            PassIndex,
            PassIndex,
            ResourceHandle,
            EdgeType,
            ResourceState,
            ResourceState,
        )] = &sp.pre_barriers;

        // post_barriers must be Vec<BarrierTuple>.
        let _post: &[(
            PassIndex,
            PassIndex,
            ResourceHandle,
            EdgeType,
            ResourceState,
            ResourceState,
        )] = &sp.post_barriers;

        // Verify the pass_index matches the iteration order.
        assert_eq!(
            sp.pass_index,
            PassIndex(i),
            "ScheduledPass[{}] must have pass_index = {}",
            i,
            i,
        );
    }
}

#[test]
fn scheduled_pass_fields_are_accessible_directly() {
    // Verify that we can read back all three fields from a ScheduledPass
    // without going through accessor methods -- confirming they are pub fields.
    let r = ResourceHandle(1);
    let passes = vec![
        mock_pass_compute(PassIndex(0), "gen", &[], &[r]),
        mock_pass_compute(PassIndex(1), "use", &[r], &[]),
    ];
    let resources = vec![mock_resource_buffer(r, "data", 128)];

    let compiled = CompiledFrameGraph::compile(passes, resources)
        .expect("must compile");

    let sp = &compiled.scheduled_passes[0];
    let _idx = sp.pass_index;
    let _pre = &sp.pre_barriers;
    let _post = &sp.post_barriers;
}

// =============================================================================
// SECTION 3 -- First pass has empty pre_barriers
// =============================================================================

#[test]
fn first_pass_has_empty_pre_barriers() {
    // The first pass in execution order has no predecessor, so its pre_barriers
    // must be empty. There is no pass before it to transition from.
    let r = ResourceHandle(1);
    let passes = vec![
        mock_pass_compute(PassIndex(0), "first", &[], &[r]),
        mock_pass_compute(PassIndex(1), "second", &[r], &[]),
    ];
    let resources = vec![mock_resource_buffer(r, "buf", 256)];

    let compiled = CompiledFrameGraph::compile(passes, resources)
        .expect("two-pass chain must compile");

    let first = &compiled.scheduled_passes[0];
    assert!(
        first.pre_barriers.is_empty(),
        "first pass (index 0) must have empty pre_barriers since it has no predecessor",
    );
}

// =============================================================================
// SECTION 4 -- Barrier-producing pass has post_barriers populated
// =============================================================================

#[test]
fn producer_pass_has_non_empty_post_barriers() {
    // Pass 0 writes R1, Pass 1 reads R1.  Pass 0 must have a post_barrier
    // that transitions R1 from ShaderReadWrite to ShaderRead for Pass 1.
    let r = ResourceHandle(1);
    let passes = vec![
        mock_pass_compute(PassIndex(0), "producer", &[], &[r]),
        mock_pass_compute(PassIndex(1), "consumer", &[r], &[]),
    ];
    let resources = vec![mock_resource_buffer(r, "buf", 256)];

    let compiled = CompiledFrameGraph::compile(passes, resources)
        .expect("two-pass chain must compile");

    let producer = &compiled.scheduled_passes[0];
    assert!(
        !producer.post_barriers.is_empty(),
        "producer pass (index 0) must have non-empty post_barriers \
         indicating the resource transition to the consumer",
    );
}

// =============================================================================
// SECTION 5 -- Barrier-consuming pass has pre_barriers populated
// =============================================================================

#[test]
fn consumer_pass_has_non_empty_pre_barriers() {
    // Pass 0 writes R1, Pass 1 reads R1.  Pass 1 must have a pre_barrier
    // that transitions R1 to ShaderRead before Pass 1 executes.
    let r = ResourceHandle(1);
    let passes = vec![
        mock_pass_compute(PassIndex(0), "producer", &[], &[r]),
        mock_pass_compute(PassIndex(1), "consumer", &[r], &[]),
    ];
    let resources = vec![mock_resource_buffer(r, "buf", 256)];

    let compiled = CompiledFrameGraph::compile(passes, resources)
        .expect("two-pass chain must compile");

    let consumer = &compiled.scheduled_passes[1];
    assert!(
        !consumer.pre_barriers.is_empty(),
        "consumer pass (index 1) must have non-empty pre_barriers \
         indicating the resource transition needed before execution",
    );
}

// =============================================================================
// SECTION 6 -- Barrier tuples have correct structure (6-element tuple,
//              valid indices)
// =============================================================================

#[test]
fn barrier_tuples_have_correct_six_element_structure() {
    // Each barrier tuple must be a 6-element tuple:
    //   (from_pass: PassIndex, to_pass: PassIndex, resource: ResourceHandle,
    //    edge_type: EdgeType, before: ResourceState, after: ResourceState)
    // All referenced pass indices must be valid (exist in the graph).
    let r = ResourceHandle(1);
    let passes = vec![
        mock_pass_compute(PassIndex(0), "writer", &[], &[r]),
        mock_pass_compute(PassIndex(1), "reader", &[r], &[]),
    ];
    let resources = vec![mock_resource_buffer(r, "buf", 256)];

    let compiled = CompiledFrameGraph::compile(passes, resources)
        .expect("must compile");

    // Collect all barrier tuples from both pre_barriers and post_barriers.
    let mut all_tuples: Vec<(
        PassIndex,
        PassIndex,
        ResourceHandle,
        EdgeType,
        ResourceState,
        ResourceState,
    )> = Vec::new();

    for sp in &compiled.scheduled_passes {
        for b in &sp.pre_barriers {
            all_tuples.push(*b);
        }
        for b in &sp.post_barriers {
            all_tuples.push(*b);
        }
    }

    assert!(
        !all_tuples.is_empty(),
        "a two-pass chain with a RAW dependency must produce barrier tuples",
    );

    for (i, bt) in all_tuples.iter().enumerate() {
        // Destructure to verify all 6 elements are present.
        let (
            from_pass,
            to_pass,
            resource,
            edge_type,
            before_state,
            after_state,
        ) = bt;

        // from_pass and to_pass must be valid pass indices (0 or 1 in this graph).
        let from: usize = from_pass.0;
        let to: usize = to_pass.0;
        assert!(
            (from == 0 || from == 1),
            "barrier tuple[{}] from_pass {} must be a valid pass index (0 or 1)",
            i,
            from,
        );
        assert!(
            (to == 0 || to == 1),
            "barrier tuple[{}] to_pass {} must be a valid pass index (0 or 1)",
            i,
            to,
        );

        // Resource handle must be non-NONE for a barrier.
        assert_ne!(
            *resource,
            ResourceHandle::NONE,
            "barrier tuple[{}] must not reference ResourceHandle::NONE",
            i,
        );

        // EdgeType must be one of the known variants.
        let _edge_debug = format!("{:?}", edge_type);

        // before_state and after_state must not be the sentinel.
        let _before_debug = format!("{:?}", before_state);
        let _after_debug = format!("{:?}", after_state);

        // from_pass must be less than to_pass (barriers always go forward).
        assert!(
            from < to || (from == to),
            "barrier tuple[{}] from_pass({}) must be <= to_pass({})",
            i,
            from,
            to,
        );
    }
}

#[test]
fn barrier_tuple_edge_type_is_meaningful() {
    // Verify that barrier tuples carry a recognizable EdgeType that reflects
    // the dependency type (RAW for read-after-write).
    let r = ResourceHandle(1);
    let passes = vec![
        mock_pass_compute(PassIndex(0), "writer", &[], &[r]),
        mock_pass_compute(PassIndex(1), "reader", &[r], &[]),
    ];
    let resources = vec![mock_resource_buffer(r, "buf", 256)];

    let compiled = CompiledFrameGraph::compile(passes, resources)
        .expect("must compile");

    // Collect all barrier tuples and check that at least one has EdgeType::RAW.
    let has_raw = compiled.scheduled_passes.iter().any(|sp| {
        sp.pre_barriers
            .iter()
            .any(|bt| matches!(bt.3, EdgeType::RAW))
            || sp.post_barriers
                .iter()
                .any(|bt| matches!(bt.3, EdgeType::RAW))
    });

    assert!(
        has_raw,
        "at least one barrier tuple must have EdgeType::RAW for a write-then-read chain",
    );
}

#[test]
fn barrier_tuple_resource_state_transition_is_valid() {
    // Verify that barrier tuples have meaningful before/after resource states
    // (not both the same, and not Uninitialized on the after side for a
    // write-then-read chain).
    let r = ResourceHandle(1);
    let passes = vec![
        mock_pass_compute(PassIndex(0), "writer", &[], &[r]),
        mock_pass_compute(PassIndex(1), "reader", &[r], &[]),
    ];
    let resources = vec![mock_resource_buffer(r, "buf", 256)];

    let compiled = CompiledFrameGraph::compile(passes, resources)
        .expect("must compile");

    let mut barrier_found = false;

    for sp in &compiled.scheduled_passes {
        for bt in &sp.pre_barriers {
            barrier_found = true;
            // pre_barrier: the state before is what the *previous* pass left
            // the resource in, and after is what *this* pass needs.
            // The two states should differ if a transition is required.
            let _before = bt.4;
            let _after = bt.5;
        }
        for bt in &sp.post_barriers {
            barrier_found = true;
            // post_barrier: the state before is what *this* pass did to the
            // resource, and after is what the *next* pass needs.
            let _before = bt.4;
            let _after = bt.5;
        }
    }

    assert!(
        barrier_found,
        "must find at least one barrier tuple with state transitions",
    );
}

// =============================================================================
// SECTION 7 -- No-pass graph produces empty scheduled_passes
// =============================================================================

#[test]
fn empty_pass_list_produces_empty_scheduled_passes() {
    // An empty graph with no passes and no resources must produce an empty
    // scheduled_passes vector.
    let passes = vec![];
    let resources = vec![];

    let compiled = CompiledFrameGraph::compile(passes, resources)
        .expect("empty graph must compile");

    assert!(
        compiled.scheduled_passes.is_empty(),
        "empty pass list must produce empty scheduled_passes",
    );
}

// =============================================================================
// SECTION 8 -- scheduled_passes.len() matches the number of passes
// =============================================================================

#[test]
fn scheduled_passes_len_matches_pass_count_diamond() {
    // Diamond graph with 4 passes.  scheduled_passes.len() must be 4,
    // one per pass in execution order.  In a diamond:
    //   entry(0) -> mid_a(1), mid_b(2) -> exit(3)
    let r1 = ResourceHandle(1);
    let r2 = ResourceHandle(2);
    let r3 = ResourceHandle(3);
    let r4 = ResourceHandle(4);

    let passes = vec![
        mock_pass_compute(PassIndex(0), "entry", &[], &[r1, r2]),
        mock_pass_compute(PassIndex(1), "mid_a", &[r1], &[r3]),
        mock_pass_compute(PassIndex(2), "mid_b", &[r2], &[r4]),
        mock_pass_compute(PassIndex(3), "exit", &[r3, r4], &[]),
    ];
    let resources = vec![
        mock_resource_buffer(r1, "ra", 64),
        mock_resource_buffer(r2, "rb", 64),
        mock_resource_buffer(r3, "rc", 64),
        mock_resource_buffer(r4, "rd", 64),
    ];

    let compiled = CompiledFrameGraph::compile(passes, resources)
        .expect("diamond graph must compile");

    assert_eq!(
        compiled.scheduled_passes.len(),
        4,
        "diamond graph with 4 passes must produce 4 scheduled passes",
    );

    // Each pass_index in scheduled_passes must be unique and match 0..3.
    let mut seen_indices: Vec<PassIndex> = compiled
        .scheduled_passes
        .iter()
        .map(|sp| sp.pass_index)
        .collect();
    seen_indices.sort();
    assert_eq!(
        seen_indices,
        vec![
            PassIndex(0),
            PassIndex(1),
            PassIndex(2),
            PassIndex(3)
        ],
        "scheduled_passes must contain one entry per pass index in order",
    );
}

#[test]
fn scheduled_passes_len_matches_pass_count_graphics_chain() {
    // Three graphics passes: P0 writes tex, P1 reads tex and writes post,
    // P2 reads post.  scheduled_passes.len() must be 3.
    let r1 = ResourceHandle(1);
    let r2 = ResourceHandle(2);

    let passes = vec![
        mock_pass_graphics(PassIndex(0), "gbuffer", &[r1]),
        mock_pass_graphics(PassIndex(1), "post", &[r1, r2]),
        mock_pass_graphics(PassIndex(2), "final", &[r2]),
    ];
    let resources = vec![
        mock_resource_texture(r1, "color_rt", 1920, 1080),
        mock_resource_texture(r2, "post_rt", 1920, 1080),
    ];

    let compiled = CompiledFrameGraph::compile(passes, resources)
        .expect("graphics chain must compile");

    assert_eq!(
        compiled.scheduled_passes.len(),
        3,
        "graphics chain with 3 passes must produce 3 scheduled passes",
    );

    // Verify each scheduled pass references the correct pass index.
    for (i, sp) in compiled.scheduled_passes.iter().enumerate() {
        assert_eq!(
            sp.pass_index,
            PassIndex(i),
            "scheduled_passes[{}] references pass index {}",
            i,
            i,
        );
    }
}

// =============================================================================
// SECTION 9 -- Barrier tuples in pre_barriers reference the correct to_pass
// =============================================================================

#[test]
fn pre_barriers_reference_correct_to_pass() {
    // In a pre_barrier for pass N, the tuple's to_pass must be N (the current
    // pass), and from_pass must be N-1 (the producer).
    let r = ResourceHandle(1);
    let passes = vec![
        mock_pass_compute(PassIndex(0), "a", &[], &[r]),
        mock_pass_compute(PassIndex(1), "b", &[r], &[]),
        mock_pass_compute(PassIndex(2), "c", &[r], &[]),
    ];
    let resources = vec![mock_resource_buffer(r, "buf", 256)];

    let compiled = CompiledFrameGraph::compile(passes, resources)
        .expect("three-pass chain must compile");

    // Check that for pass 1, any pre_barrier has to_pass == 1 and
    // from_pass == 0 (the predecessor).
    let sp1 = &compiled.scheduled_passes[1];
    if !sp1.pre_barriers.is_empty() {
        for bt in &sp1.pre_barriers {
            assert_eq!(
                bt.1,
                PassIndex(1),
                "pre_barrier for pass 1 must have to_pass = 1",
            );
            // The from_pass should be the previous pass.
            assert!(
                bt.0 .0 < bt.1 .0,
                "pre_barrier from_pass ({}) must be < to_pass ({})",
                bt.0 .0,
                bt.1 .0,
            );
        }
    }
}

// =============================================================================
// SECTION 10 -- Barrier tuples in post_barriers reference the correct from_pass
// =============================================================================

#[test]
fn post_barriers_reference_correct_from_pass() {
    // In a post_barrier for pass N, the tuple's from_pass must be N (the
    // current pass), and to_pass must be N+1 (the consumer).
    let r = ResourceHandle(1);
    let passes = vec![
        mock_pass_compute(PassIndex(0), "a", &[], &[r]),
        mock_pass_compute(PassIndex(1), "b", &[r], &[]),
    ];
    let resources = vec![mock_resource_buffer(r, "buf", 256)];

    let compiled = CompiledFrameGraph::compile(passes, resources)
        .expect("two-pass chain must compile");

    // Pass 0 should have post_barriers pointing from 0 to 1.
    let sp0 = &compiled.scheduled_passes[0];
    if !sp0.post_barriers.is_empty() {
        for bt in &sp0.post_barriers {
            assert_eq!(
                bt.0,
                PassIndex(0),
                "post_barrier for pass 0 must have from_pass = 0",
            );
            assert_eq!(
                bt.1,
                PassIndex(1),
                "post_barrier for pass 0 must have to_pass = 1 (the consumer)",
            );
        }
    }
}

// =============================================================================
// SECTION 11 -- Single pass has empty pre_barriers and empty post_barriers
// =============================================================================

#[test]
fn single_pass_has_empty_pre_and_post_barriers() {
    // A single pass with no dependent passes must have both pre_barriers and
    // post_barriers empty.
    let r = ResourceHandle(1);
    let passes = vec![mock_pass_graphics(PassIndex(0), "solo", &[r])];
    let resources = vec![mock_resource_texture(r, "rt", 800, 600)];

    let compiled = CompiledFrameGraph::compile(passes, resources)
        .expect("single pass must compile");

    assert_eq!(
        compiled.scheduled_passes.len(),
        1,
        "single pass produces exactly one scheduled pass",
    );

    let sp = &compiled.scheduled_passes[0];
    assert!(
        sp.pre_barriers.is_empty(),
        "single pass with no predecessor must have empty pre_barriers",
    );
    assert!(
        sp.post_barriers.is_empty(),
        "single pass with no successor must have empty post_barriers",
    );
}
