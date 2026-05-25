// Blackbox contract tests for T-FG-4.7 Barrier state transition tests.
//
// CLEANROOM: No src/ access beyond the public API exported by the crate.
// Tests use only `renderer_backend::frame_graph::*` -- no internal fields,
// no private methods, no implementation details.
//
// Acceptance criteria (T-FG-4.7):
//   - UNINITIALIZED -> ShaderRead first-use barrier exists
//   - UNINITIALIZED -> ColorAttachment first-use barrier exists
//   - ShaderRead -> ShaderRead same-state access produces NO barrier
//   - ShaderRead -> ColorAttachment barrier required (read -> write)
//   - ColorAttachment -> ShaderRead barrier required (write -> read)
//   - ColorAttachment -> Present barrier required
//   - TransferDst -> ShaderRead barrier required
//   - Multiple barriers at same boundary are batched into one BarrierCommand
//   - Empty graph produces zero barriers
//   - Single pass produces zero inter-pass barriers
//
// Coverage:
//   1.  UNINITIALIZED -> ShaderRead first-use barrier via generate_barriers
//   2.  UNINITIALIZED -> ColorAttachment first-use barrier via generate_barriers
//   3.  ShaderRead -> ShaderRead same-state no-op via full compute_barriers chain
//   4.  ShaderRead -> ColorAttachment read-to-write via full compute_barriers chain
//   5.  ColorAttachment -> ShaderRead write-to-read via full compute_barriers chain
//   6.  ColorAttachment -> Present barrier via generate_barriers
//   7.  TransferDst -> ShaderRead barrier via generate_barriers
//   8.  Multiple barriers batched at single pass boundary via compute_barriers
//   9.  Empty graph returns zero barriers
//  10.  Single pass returns zero barriers

use renderer_backend::frame_graph::{
    compute_barriers, generate_barriers, mock_pass_compute, mock_pass_graphics,
    mock_resource_buffer, mock_resource_texture, BarrierCommand, EdgeType, IrEdge, PassIndex,
    ResourceHandle, ResourceState,
};

// =========================================================================
// SECTION 1 -- UNINITIALIZED -> ShaderRead (first use as read)
// =========================================================================

#[test]
fn uninitialized_to_shader_read_first_use_barrier_through_generate() {
    // A resource first read as ShaderRead should produce a barrier from
    // UNINITIALIZED to ShaderRead when the tuple is fed to generate_barriers.
    let r_tex = ResourceHandle(1);
    let resources = vec![mock_resource_texture(r_tex, "first_read", 1920, 1080)];
    let passes = vec![
        mock_pass_compute(PassIndex(0), "initial", &[r_tex], &[]),
        mock_pass_compute(PassIndex(1), "first_read", &[r_tex], &[]),
    ];
    let edges = vec![IrEdge::new(
        PassIndex(0),
        PassIndex(1),
        r_tex,
        EdgeType::RAW,
    )];

    // Manually construct the first-use transition: UNINITIALIZED -> ShaderRead.
    let barrier_tuples = vec![(
        PassIndex(0),
        PassIndex(1),
        r_tex,
        ResourceState::Uninitialized,
        ResourceState::ShaderRead,
    )];

    let commands = generate_barriers(&barrier_tuples, &passes, &edges, &resources);
    assert_eq!(
        commands.len(),
        1,
        "one boundary produces one BarrierCommand"
    );
    assert_eq!(
        commands[0].texture_barriers.len(),
        1,
        "one texture barrier for first-use read",
    );
    assert!(
        commands[0].buffer_barriers.is_empty(),
        "no buffer barriers for texture resource",
    );

    let tb = &commands[0].texture_barriers[0];
    assert_eq!(
        tb.before,
        ResourceState::Uninitialized,
        "before state is Uninitialized",
    );
    assert_eq!(
        tb.after,
        ResourceState::ShaderRead,
        "after state is ShaderRead",
    );
    assert_eq!(tb.resource, r_tex, "barrier targets the correct resource");
}

// =========================================================================
// SECTION 2 -- UNINITIALIZED -> ColorAttachment (first use as render target)
// =========================================================================

#[test]
fn uninitialized_to_color_attachment_first_use_barrier_through_generate() {
    // A resource first used as a render target (ColorAttachment) should produce
    // a barrier from UNINITIALIZED to ColorAttachment.
    let r_tex = ResourceHandle(2);
    let resources = vec![mock_resource_texture(r_tex, "first_rt", 1920, 1080)];
    let passes = vec![
        mock_pass_graphics(PassIndex(0), "initial", &[r_tex]),
        mock_pass_compute(PassIndex(1), "consumer", &[r_tex], &[]),
    ];
    let edges = vec![IrEdge::new(
        PassIndex(0),
        PassIndex(1),
        r_tex,
        EdgeType::RAW,
    )];

    let barrier_tuples = vec![(
        PassIndex(0),
        PassIndex(1),
        r_tex,
        ResourceState::Uninitialized,
        ResourceState::ColorAttachment,
    )];

    let commands = generate_barriers(&barrier_tuples, &passes, &edges, &resources);
    assert_eq!(
        commands.len(),
        1,
        "one boundary produces one BarrierCommand"
    );
    assert_eq!(
        commands[0].texture_barriers.len(),
        1,
        "one texture barrier for first-use render target",
    );

    let tb = &commands[0].texture_barriers[0];
    assert_eq!(
        tb.before,
        ResourceState::Uninitialized,
        "before state is Uninitialized",
    );
    assert_eq!(
        tb.after,
        ResourceState::ColorAttachment,
        "after state is ColorAttachment",
    );
    assert_eq!(tb.resource, r_tex, "barrier targets the correct resource");
}

// =========================================================================
// SECTION 3 -- ShaderRead -> ShaderRead (same-state no-op, optimized away)
// =========================================================================

#[test]
fn shader_read_to_shader_read_same_state_no_barrier() {
    // Both passes read the same resource as ShaderRead -> no state change.
    // compute_barriers should produce no barrier tuple for this resource.
    let r_tex = ResourceHandle(1);
    let resources = vec![mock_resource_texture(r_tex, "shared_read", 800, 600)];
    let passes = vec![
        mock_pass_compute(PassIndex(0), "read_a", &[r_tex], &[]),
        mock_pass_compute(PassIndex(1), "read_b", &[r_tex], &[]),
    ];
    let edges = vec![IrEdge::new(
        PassIndex(0),
        PassIndex(1),
        r_tex,
        EdgeType::RAW,
    )];
    let order = vec![PassIndex(0), PassIndex(1)];

    let barrier_tuples = compute_barriers(&order, &passes, &edges);
    assert!(
        barrier_tuples.is_empty(),
        "same-state ShaderRead -> ShaderRead produces no barrier tuples",
    );

    // Full pipeline: no barrier tuples -> no BarrierCommands.
    let commands = generate_barriers(&barrier_tuples, &passes, &edges, &resources);
    assert!(
        commands.is_empty(),
        "no barrier tuples -> no BarrierCommands",
    );
}

// =========================================================================
// SECTION 4 -- ShaderRead -> ColorAttachment (read to write)
// =========================================================================

#[test]
fn shader_read_to_color_attachment_read_to_write_barrier() {
    // P0 reads the resource as ShaderRead (compute pass read).
    // P1 writes the resource as ColorAttachment (graphics pass attachment).
    // Barrier before=P1 must transition ShaderRead -> ColorAttachment.
    let r_tex = ResourceHandle(1);
    let resources = vec![mock_resource_texture(r_tex, "gbuffer", 1920, 1080)];

    // P0: compute pass that reads r_tex -> ShaderRead state.
    let p0 = mock_pass_compute(PassIndex(0), "pre_read", &[r_tex], &[]);
    // P1: graphics pass that writes r_tex as color attachment -> ColorAttachment state.
    let p1 = mock_pass_graphics(PassIndex(1), "render", &[r_tex]);
    let passes = vec![p0, p1];

    // WAR (Write After Read): P1 writes after P0 reads.
    let edges = vec![IrEdge::new(
        PassIndex(0),
        PassIndex(1),
        r_tex,
        EdgeType::WAR,
    )];
    let order = vec![PassIndex(0), PassIndex(1)];

    let barrier_tuples = compute_barriers(&order, &passes, &edges);
    assert_eq!(
        barrier_tuples.len(),
        1,
        "ShaderRead -> ColorAttachment produces one barrier tuple",
    );

    let (_from, _to, handle, before, after) = barrier_tuples[0];
    assert_eq!(handle, r_tex, "barrier targets the correct resource");
    assert_eq!(
        before,
        ResourceState::ShaderRead,
        "before state is ShaderRead",
    );
    assert_eq!(
        after,
        ResourceState::ColorAttachment,
        "after state is ColorAttachment",
    );

    // Full pipeline: verify BarrierCommand is generated correctly.
    let commands = generate_barriers(&barrier_tuples, &passes, &edges, &resources);
    assert_eq!(commands.len(), 1, "one boundary = one BarrierCommand");
    assert_eq!(commands[0].texture_barriers.len(), 1, "one texture barrier",);
    assert!(commands[0].buffer_barriers.is_empty());
    assert_eq!(
        commands[0].texture_barriers[0].before,
        ResourceState::ShaderRead,
    );
    assert_eq!(
        commands[0].texture_barriers[0].after,
        ResourceState::ColorAttachment,
    );
}

// =========================================================================
// SECTION 5 -- ColorAttachment -> ShaderRead (write to read)
// =========================================================================

#[test]
fn color_attachment_to_shader_read_write_to_read_barrier() {
    // P0 writes the resource as ColorAttachment (graphics pass).
    // P1 reads the resource as ShaderRead (compute pass).
    // Barrier before=P1 must transition ColorAttachment -> ShaderRead.
    let r_tex = ResourceHandle(1);
    let resources = vec![mock_resource_texture(r_tex, "gbuffer", 1920, 1080)];

    let p0 = mock_pass_graphics(PassIndex(0), "render", &[r_tex]);
    let p1 = mock_pass_compute(PassIndex(1), "read", &[r_tex], &[]);
    let passes = vec![p0, p1];

    let edges = vec![IrEdge::new(
        PassIndex(0),
        PassIndex(1),
        r_tex,
        EdgeType::RAW,
    )];
    let order = vec![PassIndex(0), PassIndex(1)];

    let barrier_tuples = compute_barriers(&order, &passes, &edges);
    assert_eq!(
        barrier_tuples.len(),
        1,
        "ColorAttachment -> ShaderRead produces one barrier tuple",
    );

    let (_from, _to, handle, before, after) = barrier_tuples[0];
    assert_eq!(handle, r_tex, "barrier targets the correct resource");
    assert_eq!(
        before,
        ResourceState::ColorAttachment,
        "before state is ColorAttachment",
    );
    assert_eq!(
        after,
        ResourceState::ShaderRead,
        "after state is ShaderRead",
    );

    // Full pipeline: verify BarrierCommand is generated correctly.
    let commands = generate_barriers(&barrier_tuples, &passes, &edges, &resources);
    assert_eq!(commands.len(), 1, "one boundary = one BarrierCommand");
    assert_eq!(commands[0].texture_barriers.len(), 1, "one texture barrier",);
    assert_eq!(
        commands[0].texture_barriers[0].before,
        ResourceState::ColorAttachment,
    );
    assert_eq!(
        commands[0].texture_barriers[0].after,
        ResourceState::ShaderRead,
    );
}

// =========================================================================
// SECTION 6 -- ColorAttachment -> Present (render target to present)
// =========================================================================

#[test]
fn color_attachment_to_present_barrier_through_generate() {
    // A resource transitioning from ColorAttachment to Present needs a barrier.
    // This is the swap-chain image transition that occurs after rendering.
    let r_tex = ResourceHandle(1);
    let resources = vec![mock_resource_texture(r_tex, "swapchain", 1920, 1080)];
    let passes = vec![
        mock_pass_graphics(PassIndex(0), "render", &[r_tex]),
        mock_pass_compute(PassIndex(1), "post", &[r_tex], &[]),
    ];
    let edges = vec![IrEdge::new(
        PassIndex(0),
        PassIndex(1),
        r_tex,
        EdgeType::RAW,
    )];

    let barrier_tuples = vec![(
        PassIndex(0),
        PassIndex(1),
        r_tex,
        ResourceState::ColorAttachment,
        ResourceState::Present,
    )];

    let commands = generate_barriers(&barrier_tuples, &passes, &edges, &resources);
    assert_eq!(commands.len(), 1, "one boundary = one BarrierCommand");
    assert_eq!(
        commands[0].texture_barriers.len(),
        1,
        "one texture barrier for ColorAttachment -> Present",
    );
    assert!(commands[0].buffer_barriers.is_empty());

    let tb = &commands[0].texture_barriers[0];
    assert_eq!(
        tb.before,
        ResourceState::ColorAttachment,
        "before state is ColorAttachment",
    );
    assert_eq!(tb.after, ResourceState::Present, "after state is Present",);
    assert_eq!(tb.resource, r_tex, "barrier targets the correct resource");
}

// =========================================================================
// SECTION 7 -- TransferDst -> ShaderRead (copy destination to shader read)
// =========================================================================

#[test]
fn transfer_dst_to_shader_read_barrier_through_generate() {
    // A resource written as a copy destination and later read as a shader
    // resource needs a barrier from TransferDst to ShaderRead.
    let r_tex = ResourceHandle(1);
    let resources = vec![mock_resource_texture(r_tex, "upload", 1920, 1080)];
    let passes = vec![
        mock_pass_compute(PassIndex(0), "copy_upload", &[], &[r_tex]),
        mock_pass_compute(PassIndex(1), "shader_read", &[r_tex], &[]),
    ];
    let edges = vec![IrEdge::new(
        PassIndex(0),
        PassIndex(1),
        r_tex,
        EdgeType::RAW,
    )];

    let barrier_tuples = vec![(
        PassIndex(0),
        PassIndex(1),
        r_tex,
        ResourceState::TransferDst,
        ResourceState::ShaderRead,
    )];

    let commands = generate_barriers(&barrier_tuples, &passes, &edges, &resources);
    assert_eq!(commands.len(), 1, "one boundary = one BarrierCommand");
    assert_eq!(
        commands[0].texture_barriers.len(),
        1,
        "one texture barrier for TransferDst -> ShaderRead",
    );

    let tb = &commands[0].texture_barriers[0];
    assert_eq!(
        tb.before,
        ResourceState::TransferDst,
        "before state is TransferDst",
    );
    assert_eq!(
        tb.after,
        ResourceState::ShaderRead,
        "after state is ShaderRead",
    );
    assert_eq!(tb.resource, r_tex, "barrier targets the correct resource");
}

// =========================================================================
// SECTION 8 -- Multiple barriers batched at pass boundary
// =========================================================================

#[test]
fn multiple_barriers_batched_at_single_boundary() {
    // Three resources transitioning at the same pass boundary (P0 -> P1).
    // All three barriers should be batched into a single BarrierCommand.
    let r_a = ResourceHandle(1);
    let r_b = ResourceHandle(2);
    let r_c = ResourceHandle(3);
    let resources = vec![
        mock_resource_texture(r_a, "albedo", 1920, 1080),
        mock_resource_texture(r_b, "normal", 1920, 1080),
        mock_resource_texture(r_c, "depth", 1920, 1080),
    ];

    // P0: graphics pass that writes all three as color attachments.
    // P1: compute pass that reads all three.
    let passes = vec![
        mock_pass_graphics(PassIndex(0), "gbuffer", &[r_a, r_b, r_c]),
        {
            let mut p = mock_pass_compute(PassIndex(1), "deferred", &[], &[]);
            p.access_set.reads.push(r_a);
            p.access_set.reads.push(r_b);
            p.access_set.reads.push(r_c);
            p
        },
    ];

    let edges = vec![
        IrEdge::new(PassIndex(0), PassIndex(1), r_a, EdgeType::RAW),
        IrEdge::new(PassIndex(0), PassIndex(1), r_b, EdgeType::RAW),
        IrEdge::new(PassIndex(0), PassIndex(1), r_c, EdgeType::RAW),
    ];
    let order = vec![PassIndex(0), PassIndex(1)];

    let barrier_tuples = compute_barriers(&order, &passes, &edges);
    // Three resources at the same boundary -> three barrier tuples.
    assert_eq!(
        barrier_tuples.len(),
        3,
        "three resources at same boundary = three barrier 5-tuples",
    );

    // Each tuple carries its own ResourceHandle.
    let handles: Vec<ResourceHandle> = barrier_tuples.iter().map(|t| t.2).collect();
    assert!(handles.contains(&r_a), "r_a has a barrier tuple");
    assert!(handles.contains(&r_b), "r_b has a barrier tuple");
    assert!(handles.contains(&r_c), "r_c has a barrier tuple");

    // Full pipeline: generate_barriers produces ONE BarrierCommand
    // (one boundary) containing THREE texture barriers.
    let commands = generate_barriers(&barrier_tuples, &passes, &edges, &resources);
    assert_eq!(
        commands.len(),
        1,
        "one pass boundary = one BarrierCommand (batched)",
    );

    let cmd = &commands[0];
    assert_eq!(
        cmd.texture_barriers.len(),
        3,
        "all three barriers batched into the same BarrierCommand",
    );
    assert!(
        cmd.buffer_barriers.is_empty(),
        "no buffer barriers in texture-only batch",
    );

    // Each barrier in the batch targets the correct resource.
    let batched_handles: Vec<ResourceHandle> =
        cmd.texture_barriers.iter().map(|tb| tb.resource).collect();
    assert!(batched_handles.contains(&r_a), "r_a in batch");
    assert!(batched_handles.contains(&r_b), "r_b in batch");
    assert!(batched_handles.contains(&r_c), "r_c in batch");
}

#[test]
fn multi_resource_batch_includes_texture_and_buffer() {
    // Mixed texture + buffer resources at the same boundary.
    // Both barrier types should be batched into one BarrierCommand.
    let r_tex = ResourceHandle(1);
    let r_buf = ResourceHandle(10);
    let resources = vec![
        mock_resource_texture(r_tex, "color", 1920, 1080),
        mock_resource_buffer(r_buf, "data", 4096),
    ];

    let passes = vec![
        {
            let mut p = mock_pass_graphics(PassIndex(0), "write_both", &[r_tex]);
            p.access_set.writes.push(r_buf);
            p
        },
        {
            let mut p = mock_pass_compute(PassIndex(1), "read_both", &[], &[]);
            p.access_set.reads.push(r_tex);
            p.access_set.reads.push(r_buf);
            p
        },
    ];

    let edges = vec![
        IrEdge::new(PassIndex(0), PassIndex(1), r_tex, EdgeType::RAW),
        IrEdge::new(PassIndex(0), PassIndex(1), r_buf, EdgeType::RAW),
    ];
    let order = vec![PassIndex(0), PassIndex(1)];

    let barrier_tuples = compute_barriers(&order, &passes, &edges);
    assert_eq!(
        barrier_tuples.len(),
        2,
        "two resources = two barrier 5-tuples",
    );

    let commands = generate_barriers(&barrier_tuples, &passes, &edges, &resources);
    assert_eq!(
        commands.len(),
        1,
        "one boundary = one batched BarrierCommand",
    );

    // Both barrier types present in a single command.
    assert_eq!(commands[0].texture_barriers.len(), 1, "one texture barrier",);
    assert_eq!(commands[0].buffer_barriers.len(), 1, "one buffer barrier",);
    assert_eq!(
        commands[0].texture_barriers[0].resource, r_tex,
        "texture barrier for r_tex",
    );
    assert_eq!(
        commands[0].buffer_barriers[0].resource, r_buf,
        "buffer barrier for r_buf",
    );
}

// =========================================================================
// SECTION 9 -- Empty graph -> zero barriers
// =========================================================================

#[test]
fn compute_barriers_empty_graph_zero_barriers() {
    // No passes, no edges, no resources -> no barriers.
    let barrier_tuples = compute_barriers(&[], &[], &[]);
    assert!(
        barrier_tuples.is_empty(),
        "compute_barriers with empty graph returns no barriers",
    );
}

#[test]
fn generate_barriers_empty_input_zero_barriers() {
    // Empty barrier tuples, passes, edges, resources -> no BarrierCommands.
    let commands = generate_barriers(&[], &[], &[], &[]);
    assert!(
        commands.is_empty(),
        "generate_barriers with all empty inputs returns no BarrierCommands",
    );
}

#[test]
fn zero_barrier_tuples_produces_zero_commands() {
    // Explicitly empty barrier tuples with valid passes/resources.
    let r_tex = ResourceHandle(1);
    let resources = vec![mock_resource_texture(r_tex, "unused", 800, 600)];
    let passes = vec![
        mock_pass_compute(PassIndex(0), "p0", &[r_tex], &[]),
        mock_pass_compute(PassIndex(1), "p1", &[r_tex], &[]),
    ];
    let edges = vec![IrEdge::new(
        PassIndex(0),
        PassIndex(1),
        r_tex,
        EdgeType::RAW,
    )];

    let commands = generate_barriers(&[], &passes, &edges, &resources);
    assert!(
        commands.is_empty(),
        "zero barrier tuples -> zero BarrierCommands even with valid resources/passes",
    );
}

// =========================================================================
// SECTION 10 -- Single pass -> zero barriers
// =========================================================================

#[test]
fn single_pass_compute_zero_inter_pass_barriers() {
    // A single compute pass with no pass boundaries should produce no barriers.
    let r_tex = ResourceHandle(1);
    let resources = vec![mock_resource_texture(r_tex, "single", 800, 600)];
    let passes = vec![mock_pass_compute(PassIndex(0), "only_pass", &[r_tex], &[])];
    let order = vec![PassIndex(0)];

    let barrier_tuples = compute_barriers(&order, &passes, &[]);
    assert!(
        barrier_tuples.is_empty(),
        "single compute pass -> no barriers",
    );
}

#[test]
fn single_pass_graphics_zero_inter_pass_barriers() {
    // A single graphics pass with no pass boundaries should produce no barriers.
    let r_tex = ResourceHandle(1);
    let resources = vec![mock_resource_texture(r_tex, "single_rt", 800, 600)];
    let passes = vec![mock_pass_graphics(PassIndex(0), "only_render", &[r_tex])];
    let order = vec![PassIndex(0)];

    let barrier_tuples = compute_barriers(&order, &passes, &[]);
    assert!(
        barrier_tuples.is_empty(),
        "single graphics pass -> no barriers",
    );
}

#[test]
fn single_pass_no_edges_zero_barriers() {
    // A single pass with resources but no edges -> no barriers.
    let r_tex = ResourceHandle(1);
    let resources = vec![mock_resource_texture(r_tex, "lonely", 800, 600)];
    let passes = vec![mock_pass_graphics(PassIndex(0), "only_pass", &[r_tex])];
    let order = vec![PassIndex(0)];

    let barrier_tuples = compute_barriers(&order, &passes, &[]);
    assert!(
        barrier_tuples.is_empty(),
        "single pass with resources but no edges -> no barriers",
    );

    let commands = generate_barriers(&barrier_tuples, &passes, &[], &resources);
    assert!(
        commands.is_empty(),
        "no barrier tuples -> no BarrierCommands",
    );
}

// =========================================================================
// SECTION 11 -- Full pipeline: UNINITIALIZED -> ShaderRead via compute_barriers
//             then generate_barriers (first use through the full API)
// =========================================================================

#[test]
fn first_use_read_pipeline_uninitialized_to_shader_read() {
    // A resource that has never been written starts as UNINITIALIZED.
    // When the first pass reads it, compute_barriers should emit a transition
    // from UNINITIALIZED to ShaderRead.
    //
    // This test validates that the initial state (for a resource read before
    // any write) is correctly handled as UNINITIALIZED.
    let r_tex = ResourceHandle(1);
    let resources = vec![mock_resource_texture(r_tex, "first_use", 800, 600)];
    let passes = vec![
        // P0 does not touch r_tex at all.
        mock_pass_compute(PassIndex(0), "unrelated", &[], &[]),
        // P1 reads r_tex for the first time -> ShaderRead.
        mock_pass_compute(PassIndex(1), "first_reader", &[r_tex], &[]),
        // P2 also reads (no transition from P1).
        mock_pass_compute(PassIndex(2), "second_reader", &[r_tex], &[]),
    ];
    // Edges only between passes that touch the resource.
    let edges = vec![
        IrEdge::new(PassIndex(0), PassIndex(1), r_tex, EdgeType::RAW),
        IrEdge::new(PassIndex(1), PassIndex(2), r_tex, EdgeType::RAW),
    ];
    let order = vec![PassIndex(0), PassIndex(1), PassIndex(2)];

    let barrier_tuples = compute_barriers(&order, &passes, &edges);

    // The first pass to USE the resource produces a transition from
    // UNINITIALIZED to ShaderRead. The second read is same-state (no barrier).
    // Total: 1 barrier for the first-use transition.
    assert_eq!(
        barrier_tuples.len(),
        1,
        "first use of resource produces one barrier (UNINITIALIZED -> ShaderRead)",
    );

    let (from, to, handle, before, after) = barrier_tuples[0];
    assert_eq!(handle, r_tex, "barrier targets the first-use resource");
    assert_eq!(
        before,
        ResourceState::Uninitialized,
        "first-use before state is Uninitialized",
    );
    assert_eq!(
        after,
        ResourceState::ShaderRead,
        "first-use after state is ShaderRead",
    );

    // Full pipeline through generate_barriers.
    let commands = generate_barriers(&barrier_tuples, &passes, &edges, &resources);
    assert_eq!(commands.len(), 1, "one batch at the first-use boundary");
    assert_eq!(commands[0].texture_barriers.len(), 1);
    assert_eq!(
        commands[0].texture_barriers[0].before,
        ResourceState::Uninitialized,
    );
    assert_eq!(
        commands[0].texture_barriers[0].after,
        ResourceState::ShaderRead,
    );
}

// =========================================================================
// SECTION 12 -- Full pipeline: UNINITIALIZED -> ColorAttachment via
//             compute_barriers then generate_barriers (first RT use)
// =========================================================================

#[test]
fn first_use_render_target_pipeline_uninitialized_to_color_attachment() {
    // A resource that has never been written starts as UNINITIALIZED.
    // When the first pass writes it as a render target, compute_barriers
    // should emit a transition from UNINITIALIZED to ColorAttachment.
    let r_tex = ResourceHandle(1);
    let resources = vec![mock_resource_texture(r_tex, "first_rt", 800, 600)];
    let passes = vec![
        // P0 does not touch r_tex at all.
        mock_pass_compute(PassIndex(0), "unrelated", &[], &[]),
        // P1 writes r_tex as a color attachment for the first time.
        mock_pass_graphics(PassIndex(1), "first_render", &[r_tex]),
        // P2 reads the result.
        mock_pass_compute(PassIndex(2), "consumer", &[r_tex], &[]),
    ];
    let edges = vec![
        IrEdge::new(PassIndex(0), PassIndex(1), r_tex, EdgeType::RAW),
        IrEdge::new(PassIndex(1), PassIndex(2), r_tex, EdgeType::RAW),
    ];
    let order = vec![PassIndex(0), PassIndex(1), PassIndex(2)];

    let barrier_tuples = compute_barriers(&order, &passes, &edges);

    // Two transitions: UNINITIALIZED -> ColorAttachment (first use),
    // and ColorAttachment -> ShaderRead (write to read).
    assert_eq!(
        barrier_tuples.len(),
        2,
        "two transitions: first-use + write-to-read",
    );

    // First barrier: UNINITIALIZED -> ColorAttachment (between P0 and P1).
    let (f0, t0, h0, b0, a0) = barrier_tuples[0];
    assert_eq!(f0, PassIndex(0), "first barrier at P0->P1 boundary");
    assert_eq!(t0, PassIndex(1));
    assert_eq!(h0, r_tex);
    assert_eq!(
        b0,
        ResourceState::Uninitialized,
        "first barrier before=Uninitialized"
    );
    assert_eq!(
        a0,
        ResourceState::ColorAttachment,
        "first barrier after=ColorAttachment"
    );

    // Second barrier: ColorAttachment -> ShaderRead (between P1 and P2).
    let (f1, t1, h1, b1, a1) = barrier_tuples[1];
    assert_eq!(f1, PassIndex(1), "second barrier at P1->P2 boundary");
    assert_eq!(t1, PassIndex(2));
    assert_eq!(h1, r_tex);
    assert_eq!(
        b1,
        ResourceState::ColorAttachment,
        "second barrier before=ColorAttachment"
    );
    assert_eq!(
        a1,
        ResourceState::ShaderRead,
        "second barrier after=ShaderRead"
    );

    // Full pipeline through generate_barriers: two boundaries = two commands.
    let commands = generate_barriers(&barrier_tuples, &passes, &edges, &resources);
    assert_eq!(commands.len(), 2, "two boundaries = two BarrierCommands");
}

// =========================================================================
// SECTION 13 -- A -> B -> A redundant elimination via compute_barriers
//             (transitions back and forth produce distinct barriers)
// =========================================================================

#[test]
fn a_to_b_to_a_produces_two_barriers_not_eliminated() {
    // A resource transitions ColorAttachment -> ShaderRead -> ColorAttachment
    // across three passes. Each transition is genuine and must produce its
    // own barrier. (The A->B->A redundant elimination is handled by the
    // BarrierOptimizer, not by compute_barriers.)
    let r_tex = ResourceHandle(1);
    let resources = vec![mock_resource_texture(r_tex, "ping_pong", 800, 600)];
    let passes = vec![
        mock_pass_graphics(PassIndex(0), "write_a", &[r_tex]),
        mock_pass_compute(PassIndex(1), "read_b", &[r_tex], &[]),
        mock_pass_graphics(PassIndex(2), "write_a_again", &[r_tex]),
    ];
    let edges = vec![
        IrEdge::new(PassIndex(0), PassIndex(1), r_tex, EdgeType::RAW),
        IrEdge::new(PassIndex(1), PassIndex(2), r_tex, EdgeType::RAW),
    ];
    let order = vec![PassIndex(0), PassIndex(1), PassIndex(2)];

    let barrier_tuples = compute_barriers(&order, &passes, &edges);

    // Expect 2 barriers:
    //   0->1: ColorAttachment -> ShaderRead
    //   1->2: ShaderRead -> ColorAttachment
    assert_eq!(
        barrier_tuples.len(),
        2,
        "A->B->A across three passes = two distinct barriers",
    );

    // First barrier: ColorAttachment -> ShaderRead.
    assert_eq!(
        barrier_tuples[0].3,
        ResourceState::ColorAttachment,
        "first: before=ColorAttachment",
    );
    assert_eq!(
        barrier_tuples[0].4,
        ResourceState::ShaderRead,
        "first: after=ShaderRead",
    );

    // Second barrier: ShaderRead -> ColorAttachment.
    assert_eq!(
        barrier_tuples[1].3,
        ResourceState::ShaderRead,
        "second: before=ShaderRead",
    );
    assert_eq!(
        barrier_tuples[1].4,
        ResourceState::ColorAttachment,
        "second: after=ColorAttachment",
    );

    // Full pipeline: two boundaries = two BarrierCommands.
    let commands = generate_barriers(&barrier_tuples, &passes, &edges, &resources);
    assert_eq!(commands.len(), 2, "two boundaries = two BarrierCommands");
    assert_eq!(commands[0].texture_barriers.len(), 1);
    assert_eq!(commands[1].texture_barriers.len(), 1);
}
