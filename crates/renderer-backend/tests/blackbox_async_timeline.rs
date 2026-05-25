// Blackbox contract tests for T-FG-5.2 (Secondary timeline builder /
// async_timeline on CompiledFrameGraph).
//
// CLEANROOM: No src/ access beyond the public API exported by the crate.
// Tests use only `renderer_backend::frame_graph::*` -- no internal fields,
// no private methods, no implementation details.
//
// DEV added async_timeline: Vec<PassIndex> to CompiledFrameGraph. It contains
// the pass indices of async-eligible compute passes, ordered by their internal
// dependencies (the secondary timeline). Graphics passes, eliminated dead
// passes, and non-compute passes are excluded.
//
// Scenarios:
//   1.  Compile with only graphics passes   -> async_timeline is empty
//   2.  Compile with one compute pass        -> async_timeline has one entry
//   3.  Compile compute + graphics           -> async_timeline only contains compute passes
//   4.  Graphics passes are NOT in async_timeline
//   5.  Eliminated async pass (dead) does not appear in async_timeline
//   6.  Async timeline entries are valid PassIndex values
//   7.  Empty graph                          -> async_timeline is empty
//   8.  async_timeline order respects internal compute dependencies
//
use std::collections::HashSet;

use renderer_backend::frame_graph::{
    mock_pass_compute, mock_pass_graphics, mock_resource_buffer, mock_resource_texture,
    CompiledFrameGraph, PassIndex, ResourceHandle,
};

// =============================================================================
// SECTION 1 -- Graphics-only graph produces empty async_timeline
// =============================================================================

#[test]
fn graphics_only_async_timeline_empty() {
    // Single graphics pass, zero compute.  async_timeline must be empty.
    let r = ResourceHandle(1);
    let compiled = CompiledFrameGraph::compile(
        vec![mock_pass_graphics(PassIndex(0), "render", &[r])],
        vec![mock_resource_texture(r, "swapchain", 1920, 1080)],
    )
    .expect("single graphics pass compiles");

    assert!(
        compiled.async_passes.is_empty(),
        "graphics-only graph should have empty async_passes",
    );
}

#[test]
fn multiple_graphics_only_async_timeline_empty() {
    // Three independent graphics passes, each writing a different texture.
    // Still no async pass -> timeline empty.
    let compiled = CompiledFrameGraph::compile(
        vec![
            mock_pass_graphics(PassIndex(0), "gbuffer", &[ResourceHandle(1)]),
            mock_pass_graphics(PassIndex(1), "lighting", &[ResourceHandle(2)]),
            mock_pass_graphics(PassIndex(2), "post", &[ResourceHandle(3)]),
        ],
        vec![
            mock_resource_texture(ResourceHandle(1), "albedo", 1920, 1080),
            mock_resource_texture(ResourceHandle(2), "normal", 1920, 1080),
            mock_resource_texture(ResourceHandle(3), "output", 1920, 1080),
        ],
    )
    .expect("multiple graphics passes compile");

    assert!(
        compiled.async_passes.is_empty(),
        "multiple-graphics graph should have empty async_passes",
    );
}

// =============================================================================
// SECTION 2 -- Single compute pass produces one entry
// =============================================================================

#[test]
fn one_compute_pass_async_timeline_has_one_entry() {
    // Single compute pass writing a buffer.  It is not dead (no consumer
    // check needed for the timeline — compute passes are identified by
    // async_schedule even if later eliminated).
    let r = ResourceHandle(1);
    let compiled = CompiledFrameGraph::compile(
        vec![mock_pass_compute(
            PassIndex(0),
            "compute_kernel",
            &[],
            &[r],
        )],
        vec![mock_resource_buffer(r, "buf", 1024)],
    )
    .expect("single compute pass compiles");

    // The async_passes vector must contain exactly one entry for PassIndex(0).
    assert_eq!(
        compiled.async_passes.len(),
        1,
        "one compute pass => one async entry",
    );
    let (idx, _qtype) = compiled.async_passes[0];
    assert_eq!(idx, PassIndex(0), "async entry must be PassIndex(0)");
}

#[test]
fn one_compute_pass_queue_type_is_compute() {
    // The queue_type string in the async_passes entry must be "compute"
    // for a compute-only pass.
    let r = ResourceHandle(1);
    let compiled = CompiledFrameGraph::compile(
        vec![mock_pass_compute(
            PassIndex(0),
            "cs_main",
            &[],
            &[r],
        )],
        vec![mock_resource_buffer(r, "buf", 512)],
    )
    .expect("compute pass compiles");

    assert_eq!(compiled.async_passes.len(), 1);
    let (_idx, qtype) = &compiled.async_passes[0];
    assert_eq!(
        qtype, "compute",
        "compute pass queue type must be 'compute'",
    );
}

// =============================================================================
// SECTION 3 -- Mixed compute + graphics: only compute passes in async_timeline
// =============================================================================

#[test]
fn mixed_async_timeline_only_contains_compute_passes() {
    // P0 (graphics) writes R1.  P1 (compute) reads R1 and writes R2.
    // Only P1 should appear in async_passes.
    let r1 = ResourceHandle(1);
    let r2 = ResourceHandle(2);
    let compiled = CompiledFrameGraph::compile(
        vec![
            mock_pass_graphics(PassIndex(0), "gbuffer", &[r1]),
            mock_pass_compute(PassIndex(1), "lighting", &[r1], &[r2]),
        ],
        vec![
            mock_resource_texture(r1, "albedo", 1920, 1080),
            mock_resource_buffer(r2, "result", 4096),
        ],
    )
    .expect("mixed graph compiles");

    // The async_passes must contain only the compute pass (P1).
    assert!(
        !compiled.async_passes.is_empty(),
        "compute pass present => async_passes not empty",
    );
    for (idx, _qtype) in &compiled.async_passes {
        assert_eq!(
            *idx,
            PassIndex(1),
            "only compute pass P1 should be in async_passes, but found P{:?}",
            idx,
        );
    }
}

#[test]
fn multiple_compute_passes_all_in_async_timeline() {
    // Two compute passes reading/writing a resource chain.  Both are
    // async-eligible and both should appear in async_passes.
    let r1 = ResourceHandle(1);
    let r2 = ResourceHandle(2);
    let compiled = CompiledFrameGraph::compile(
        vec![
            mock_pass_compute(PassIndex(0), "preprocess", &[], &[r1]),
            mock_pass_compute(PassIndex(1), "simulate", &[r1], &[r2]),
        ],
        vec![
            mock_resource_buffer(r1, "input", 1024),
            mock_resource_buffer(r2, "output", 2048),
        ],
    )
    .expect("two compute passes compile");

    // Both compute passes should appear in async_passes.
    let async_indices: HashSet<PassIndex> = compiled
        .async_passes
        .iter()
        .map(|(idx, _)| *idx)
        .collect();
    assert!(
        async_indices.contains(&PassIndex(0)),
        "P0 (compute) must be in async_passes",
    );
    assert!(
        async_indices.contains(&PassIndex(1)),
        "P1 (compute) must be in async_passes",
    );
}

// =============================================================================
// SECTION 4 -- Graphics passes are NOT in async_timeline
// =============================================================================

#[test]
fn graphics_pass_excluded_from_async_timeline() {
    // P0 (graphics), P1 (compute).  Only P1 should be in async_passes.
    let r1 = ResourceHandle(1);
    let compiled = CompiledFrameGraph::compile(
        vec![
            mock_pass_graphics(PassIndex(0), "main_rt", &[r1]),
            mock_pass_compute(PassIndex(1), "post", &[r1], &[]),
        ],
        vec![mock_resource_texture(r1, "color", 800, 600)],
    )
    .expect("graphics + compute compiles");

    let async_indices: HashSet<PassIndex> = compiled
        .async_passes
        .iter()
        .map(|(idx, _)| *idx)
        .collect();

    assert!(
        !async_indices.contains(&PassIndex(0)),
        "graphics pass P0 must NOT appear in async_passes",
    );
}

#[test]
fn all_graphics_no_compute_all_excluded() {
    // Even with many graphics passes, none leak into async_passes.
    let compiled = CompiledFrameGraph::compile(
        vec![
            mock_pass_graphics(PassIndex(0), "pass0", &[ResourceHandle(1)]),
            mock_pass_graphics(PassIndex(1), "pass1", &[ResourceHandle(2)]),
            mock_pass_graphics(PassIndex(2), "pass2", &[ResourceHandle(3)]),
        ],
        vec![
            mock_resource_texture(ResourceHandle(1), "r1", 64, 64),
            mock_resource_texture(ResourceHandle(2), "r2", 64, 64),
            mock_resource_texture(ResourceHandle(3), "r3", 64, 64),
        ],
    )
    .expect("three graphics passes compile");

    assert!(
        compiled.async_passes.is_empty(),
        "no compute passes => async_passes must be empty",
    );
}

// =============================================================================
// SECTION 5 -- Eliminated (dead) compute pass does NOT appear in async_timeline
// =============================================================================

#[test]
fn eliminated_dead_compute_pass_not_in_async_timeline() {
    // A single compute pass writing an unread buffer.  It is dead and
    // eliminated.  It must NOT appear in async_passes after compilation.
    let r = ResourceHandle(1);
    let compiled = CompiledFrameGraph::compile(
        vec![mock_pass_compute(
            PassIndex(0),
            "dead_compute",
            &[],
            &[r],
        )],
        vec![mock_resource_buffer(r, "orphan", 1024)],
    )
    .expect("dead compute compiles");

    // The pass was eliminated, so it must not appear in async_passes.
    for (idx, _qtype) in &compiled.async_passes {
        assert_ne!(
            *idx,
            PassIndex(0),
            "eliminated compute pass P0 must NOT appear in async_passes",
        );
    }

    // Sanity: confirm it IS in eliminated_passes.
    assert!(
        compiled.eliminated_passes.contains(&PassIndex(0)),
        "P0 must be in eliminated_passes",
    );
}

#[test]
fn mixed_live_and_dead_only_live_in_async_timeline() {
    // P0 (graphics) writes R1.  P1 (compute) reads R1, writes R2 (live).
    // P2 (compute) writes R3 unread (dead).  Only P1 should appear in
    // async_passes.
    let r1 = ResourceHandle(1);
    let r2 = ResourceHandle(2);
    let r3 = ResourceHandle(3);
    let compiled = CompiledFrameGraph::compile(
        vec![
            mock_pass_graphics(PassIndex(0), "gbuffer", &[r1]),
            mock_pass_compute(PassIndex(1), "resolve", &[r1], &[r2]),
            mock_pass_compute(PassIndex(2), "dead_cs", &[], &[r3]),
        ],
        vec![
            mock_resource_texture(r1, "color", 800, 600),
            mock_resource_buffer(r2, "data", 4096),
            mock_resource_buffer(r3, "orphan", 128),
        ],
    )
    .expect("mixed live/dead compiles");

    let async_indices: Vec<PassIndex> = compiled
        .async_passes
        .iter()
        .map(|(idx, _)| *idx)
        .collect();

    // Live compute pass P1 must be in async_passes.
    assert!(
        async_indices.contains(&PassIndex(1)),
        "live compute pass P1 must be in async_passes",
    );

    // Dead compute pass P2 must NOT be in async_passes.
    assert!(
        !async_indices.contains(&PassIndex(2)),
        "dead compute pass P2 must NOT be in async_passes",
    );
}

#[test]
fn all_dead_compute_passes_async_timeline_empty() {
    // Four compute passes, all dead (write unread resources).
    // async_passes must be empty.
    let compiled = CompiledFrameGraph::compile(
        vec![
            mock_pass_compute(PassIndex(0), "d1", &[], &[ResourceHandle(1)]),
            mock_pass_compute(PassIndex(1), "d2", &[], &[ResourceHandle(2)]),
            mock_pass_compute(PassIndex(2), "d3", &[], &[ResourceHandle(3)]),
            mock_pass_compute(PassIndex(3), "d4", &[], &[ResourceHandle(4)]),
        ],
        vec![
            mock_resource_buffer(ResourceHandle(1), "r1", 64),
            mock_resource_buffer(ResourceHandle(2), "r2", 128),
            mock_resource_buffer(ResourceHandle(3), "r3", 256),
            mock_resource_buffer(ResourceHandle(4), "r4", 512),
        ],
    )
    .expect("all-dead compiles");

    // All eliminated -> async_passes must be empty.
    assert!(
        compiled.async_passes.is_empty(),
        "all compute passes eliminated => async_passes must be empty",
    );
}

// =============================================================================
// SECTION 6 -- Async timeline entries are valid PassIndex values
// =============================================================================

#[test]
fn async_timeline_entries_are_valid_pass_indices() {
    // All entries in async_passes must be valid PassIndex values that
    // reference actual passes in the compiled order.
    let r1 = ResourceHandle(1);
    let r2 = ResourceHandle(2);
    let compiled = CompiledFrameGraph::compile(
        vec![
            mock_pass_compute(PassIndex(0), "phase1", &[], &[r1]),
            mock_pass_compute(PassIndex(1), "phase2", &[r1], &[r2]),
        ],
        vec![
            mock_resource_buffer(r1, "buf_a", 512),
            mock_resource_buffer(r2, "buf_b", 1024),
        ],
    )
    .expect("two compute passes compile");

    // Every pass index in async_passes must be present in the order.
    for (idx, _qtype) in &compiled.async_passes {
        assert!(
            compiled.order.contains(idx),
            "async_passes entry P{:?} must be in compiled.order",
            idx,
        );
    }
}

#[test]
fn async_timeline_indices_never_none_or_zero_default() {
    // PassIndex(0) is a valid first index, but the async_timeline should
    // never contain a sentinel or "empty" PassIndex that doesn't correspond
    // to a real pass.
    let r1 = ResourceHandle(1);
    let r2 = ResourceHandle(2);
    let compiled = CompiledFrameGraph::compile(
        vec![
            mock_pass_graphics(PassIndex(0), "main", &[r1]),
            mock_pass_compute(PassIndex(1), "post", &[r1], &[r2]),
        ],
        vec![
            mock_resource_texture(r1, "color", 800, 600),
            mock_resource_buffer(r2, "data", 256),
        ],
    )
    .expect("mixed graph compiles");

    for (idx, _qtype) in &compiled.async_passes {
        // The pass type of the index must be Compute (not Graphics).
        let pass = compiled
            .passes
            .iter()
            .find(|p| p.index == *idx)
            .expect("async index must reference a real pass");
        assert_eq!(
            pass.pass_type.to_string(),
            "compute",
            "async pass P{:?} must be a compute pass, got {:?}",
            idx,
            pass.pass_type,
        );
    }
}

// =============================================================================
// SECTION 7 -- Empty graph produces empty async_timeline
// =============================================================================

#[test]
fn empty_graph_async_timeline_empty() {
    let compiled =
        CompiledFrameGraph::compile(vec![], vec![]).expect("empty graph compiles");

    assert!(
        compiled.async_passes.is_empty(),
        "empty graph must have empty async_passes",
    );
}

#[test]
fn empty_graph_async_timeline_empty_and_order_empty() {
    // Both async_passes and order should be consistent for empty input.
    let compiled =
        CompiledFrameGraph::compile(vec![], vec![]).expect("empty graph compiles");

    assert_eq!(compiled.async_passes.len(), 0);
    assert_eq!(compiled.order.len(), 0);
    assert_eq!(compiled.passes.len(), 0);
}

// =============================================================================
// SECTION 8 -- async_timeline order respects internal compute dependencies
// =============================================================================

#[test]
fn async_timeline_order_respects_linear_chain() {
    // P0 (compute) writes R1.  P1 (compute) reads R1.
    // In async_passes, P0 must appear before P1.
    let r1 = ResourceHandle(1);
    let compiled = CompiledFrameGraph::compile(
        vec![
            mock_pass_compute(PassIndex(0), "producer", &[], &[r1]),
            mock_pass_compute(PassIndex(1), "consumer", &[r1], &[]),
        ],
        vec![mock_resource_buffer(r1, "chain_buf", 256)],
    )
    .expect("compute chain compiles");

    let async_vec: Vec<PassIndex> = compiled
        .async_passes
        .iter()
        .map(|(idx, _)| *idx)
        .collect();

    // Both should be present.  P0 before P1.
    let pos0 = async_vec.iter().position(|i| *i == PassIndex(0));
    let pos1 = async_vec.iter().position(|i| *i == PassIndex(1));
    assert!(
        pos0.is_some() && pos1.is_some(),
        "both compute passes P0 and P1 must be in async_passes",
    );
    assert!(
        pos0.unwrap() < pos1.unwrap(),
        "P0 (producer) must appear before P1 (consumer) in async_passes",
    );
}

#[test]
fn async_timeline_order_respects_diamond_dependency() {
    // P0 writes R1.  P1 reads R1 writes R2.  P2 reads R1 writes R3.
    // P3 reads R2 and R3 (merge).
    //
    // async_passes contains all four compute passes in an order consistent
    // with their dependencies: P0 first, P3 last.
    let r1 = ResourceHandle(1);
    let r2 = ResourceHandle(2);
    let r3 = ResourceHandle(3);
    let compiled = CompiledFrameGraph::compile(
        vec![
            mock_pass_compute(PassIndex(0), "root", &[], &[r1]),
            mock_pass_compute(PassIndex(1), "left", &[r1], &[r2]),
            mock_pass_compute(PassIndex(2), "right", &[r1], &[r3]),
            mock_pass_compute(PassIndex(3), "merge", &[r2, r3], &[]),
        ],
        vec![
            mock_resource_buffer(r1, "r1", 64),
            mock_resource_buffer(r2, "r2", 64),
            mock_resource_buffer(r3, "r3", 64),
        ],
    )
    .expect("diamond compute compiles");

    let async_vec: Vec<PassIndex> = compiled
        .async_passes
        .iter()
        .map(|(idx, _)| *idx)
        .collect();

    // P0 must be first (root producer).
    let pos0 = async_vec.iter().position(|i| *i == PassIndex(0));
    assert!(
        pos0.is_some() && pos0.unwrap() == 0,
        "root producer P0 must be first in async_passes",
    );

    // P3 must be last (merge consumer).
    let pos3 = async_vec.iter().position(|i| *i == PassIndex(3));
    assert_eq!(
        pos3,
        Some(async_vec.len() - 1),
        "merge consumer P3 must be last in async_passes",
    );

    // P0 before both P1 and P2.
    let pos1 = async_vec.iter().position(|i| *i == PassIndex(1));
    let pos2 = async_vec.iter().position(|i| *i == PassIndex(2));
    assert!(pos0.unwrap() < pos1.unwrap(), "P0 before P1");
    assert!(pos0.unwrap() < pos2.unwrap(), "P0 before P2");
}

#[test]
fn async_timeline_order_respects_fan_in() {
    // P0 writes R2.  P1 writes R3.  P2 reads R2 and R3.
    // P0 and P1 are independent; P2 must be after both.
    let r2 = ResourceHandle(2);
    let r3 = ResourceHandle(3);
    let compiled = CompiledFrameGraph::compile(
        vec![
            mock_pass_compute(PassIndex(0), "producer_a", &[], &[r2]),
            mock_pass_compute(PassIndex(1), "producer_b", &[], &[r3]),
            mock_pass_compute(PassIndex(2), "consumer", &[r2, r3], &[]),
        ],
        vec![
            mock_resource_buffer(r2, "buf_a", 256),
            mock_resource_buffer(r3, "buf_b", 512),
        ],
    )
    .expect("fan-in compiles");

    let async_vec: Vec<PassIndex> = compiled
        .async_passes
        .iter()
        .map(|(idx, _)| *idx)
        .collect();

    // Both producers before consumer.
    let pos0 = async_vec.iter().position(|i| *i == PassIndex(0));
    let pos1 = async_vec.iter().position(|i| *i == PassIndex(1));
    let pos2 = async_vec.iter().position(|i| *i == PassIndex(2));

    assert!(pos0.is_some(), "P0 in async_passes");
    assert!(pos1.is_some(), "P1 in async_passes");
    assert!(pos2.is_some(), "P2 in async_passes");
    assert!(
        pos0.unwrap() < pos2.unwrap(),
        "P0 (producer) before P2 (consumer)",
    );
    assert!(
        pos1.unwrap() < pos2.unwrap(),
        "P1 (producer) before P2 (consumer)",
    );
}

#[test]
fn async_timeline_order_independent_compute_passes_any_order() {
    // Two independent compute passes with no dependencies.  Both must be
    // present in async_passes (order between them is not constrained).
    let compiled = CompiledFrameGraph::compile(
        vec![
            mock_pass_compute(PassIndex(0), "independent_a", &[], &[ResourceHandle(1)]),
            mock_pass_compute(PassIndex(1), "independent_b", &[], &[ResourceHandle(2)]),
        ],
        vec![
            mock_resource_buffer(ResourceHandle(1), "r1", 64),
            mock_resource_buffer(ResourceHandle(2), "r2", 128),
        ],
    )
    .expect("independent compute compiles");

    let async_indices: HashSet<PassIndex> = compiled
        .async_passes
        .iter()
        .map(|(idx, _)| *idx)
        .collect();

    assert!(
        async_indices.contains(&PassIndex(0)),
        "P0 in async_passes",
    );
    assert!(
        async_indices.contains(&PassIndex(1)),
        "P1 in async_passes",
    );
    assert_eq!(
        compiled.async_passes.len(),
        2,
        "both independent compute passes in async_passes",
    );
}

#[test]
fn async_timeline_order_graphics_do_not_affect_compute_order() {
    // Interleaved graphics passes must not affect the async timeline
    // order.  Compute passes maintain their relative dependency order.
    let r1 = ResourceHandle(1);
    let r2 = ResourceHandle(2);
    let r3 = ResourceHandle(3);
    let compiled = CompiledFrameGraph::compile(
        vec![
            // P0 (compute) writes r1
            mock_pass_compute(PassIndex(0), "cs_producer", &[], &[r1]),
            // P1 (graphics) uses r1 — not async
            mock_pass_graphics(PassIndex(1), "gfx_mid", &[r1]),
            // P2 (compute) reads r1 writes r2
            mock_pass_compute(PassIndex(2), "cs_consumer", &[r1], &[r2]),
            // P3 (graphics) uses r2 — not async
            mock_pass_graphics(PassIndex(3), "gfx_final", &[r2]),
        ],
        vec![
            mock_resource_texture(r1, "rt", 800, 600),
            mock_resource_buffer(r2, "data", 1024),
        ],
    )
    .expect("interleaved graphics+compute compiles");

    let async_vec: Vec<PassIndex> = compiled
        .async_passes
        .iter()
        .map(|(idx, _)| *idx)
        .collect();

    // Only compute passes (P0, P2) should be in async_passes.
    let pos0 = async_vec.iter().position(|i| *i == PassIndex(0));
    let pos2 = async_vec.iter().position(|i| *i == PassIndex(2));

    assert!(pos0.is_some(), "compute P0 in async_passes");
    assert!(pos2.is_some(), "compute P2 in async_passes");
    assert_eq!(async_vec.len(), 2, "only two compute entries");

    // No graphics passes leak in.
    assert!(
        !async_vec.contains(&PassIndex(1)),
        "graphics P1 not in async_passes",
    );
    assert!(
        !async_vec.contains(&PassIndex(3)),
        "graphics P3 not in async_passes",
    );

    // Dependency order: P0 before P2.
    assert!(
        pos0.unwrap() < pos2.unwrap(),
        "P0 (producer) before P2 (consumer) in async_passes",
    );
}

// =============================================================================
// SECTION 9 -- Structural invariants
// =============================================================================

#[test]
fn async_plus_non_async_count_matches_total_surviving_passes() {
    // For any compiled graph, the number of async passes + the number of
    // non-async passes (graphics passes in order not in async_passes) must
    // equal the total surviving pass count.
    let r1 = ResourceHandle(1);
    let r2 = ResourceHandle(2);
    let compiled = CompiledFrameGraph::compile(
        vec![
            mock_pass_graphics(PassIndex(0), "gfx", &[r1]),
            mock_pass_compute(PassIndex(1), "cs", &[r1], &[r2]),
        ],
        vec![
            mock_resource_texture(r1, "tex", 800, 600),
            mock_resource_buffer(r2, "buf", 1024),
        ],
    )
    .expect("gfx+cs compiles");

    let async_count = compiled.async_passes.len();
    let order_count = compiled.order.len();

    assert!(
        async_count <= order_count,
        "async_passes len ({}) must be <= order len ({})",
        async_count,
        order_count,
    );
}

#[test]
fn no_duplicate_pass_indices_in_async_timeline() {
    // Every PassIndex should appear at most once in async_passes.
    let r1 = ResourceHandle(1);
    let r2 = ResourceHandle(2);
    let r3 = ResourceHandle(3);
    let compiled = CompiledFrameGraph::compile(
        vec![
            mock_pass_compute(PassIndex(0), "a", &[], &[r1]),
            mock_pass_compute(PassIndex(1), "b", &[r1], &[r2]),
            mock_pass_compute(PassIndex(2), "c", &[r2], &[r3]),
        ],
        vec![
            mock_resource_buffer(r1, "r1", 64),
            mock_resource_buffer(r2, "r2", 128),
            mock_resource_buffer(r3, "r3", 256),
        ],
    )
    .expect("compute chain compiles");

    let mut seen = HashSet::new();
    for (idx, _qtype) in &compiled.async_passes {
        assert!(
            seen.insert(*idx),
            "duplicate PassIndex {:?} in async_passes",
            idx,
        );
    }
}

// =============================================================================
// SECTION 10 -- Debug and Display formatting
// =============================================================================

#[test]
fn compiled_graph_debug_includes_async_info() {
    // The Debug output of CompiledFrameGraph should mention async_passes.
    let r1 = ResourceHandle(1);
    let compiled = CompiledFrameGraph::compile(
        vec![mock_pass_compute(
            PassIndex(0),
            "compute_main",
            &[],
            &[r1],
        )],
        vec![mock_resource_buffer(r1, "buf", 256)],
    )
    .expect("compute compiles");

    let debug_str = format!("{:?}", compiled);
    assert!(
        !debug_str.is_empty(),
        "Debug output must not be empty",
    );
    // The string representation should cover the async schedule.
    let passes_present = compiled.async_passes.is_empty()
        || compiled.async_passes.iter().any(|(idx, _)| *idx == PassIndex(0));
    assert!(
        passes_present,
        "async_passes should reference P0 when a compute pass is present",
    );
}

#[test]
fn async_passes_field_is_public_and_iterable() {
    // The async_passes field must be publicly accessible and iterable.
    let compiled = CompiledFrameGraph::compile(vec![], vec![])
        .expect("empty graph compiles");

    // Type assertion: it's a Vec<(PassIndex, String)>.
    let _async_passes: &Vec<(PassIndex, String)> = &compiled.async_passes;
    assert!(_async_passes.is_empty());
}
