// Blackbox contract tests for T-FG-4.5 barrier FIX: multi-resource same-boundary.
//
// CLEANROOM: No src/ access beyond the public API exported by the crate.
// Tests use only `renderer_backend::frame_graph::*` -- no internal fields,
// no private methods, no implementation details.
//
// Acceptance criterion (T-FG-4.5 FIX):
//   generate_barriers() now correctly handles multi-resource same-boundary
//   via 5-tuple (PassIndex, PassIndex, ResourceState, ResourceState,
//   ResourceHandle). When two or more edges share the same (from, to) pass
//   boundary but reference different resources, each resource gets its own
//   barrier tuple carrying the correct ResourceHandle directly -- eliminating
//   the prior bug where the edge-resource HashMap (keyed by (from, to) with
//   or_insert) collapsed multi-resource same-boundary barriers to a single
//   resource handle.
//
// 5-tuple contract:
//   - compute_barriers returns Vec<(PassIndex, PassIndex, ResourceState,
//     ResourceState, ResourceHandle)>
//   - generate_barriers accepts &[(PassIndex, PassIndex, ResourceState,
//     ResourceState, ResourceHandle)]
//   - Every barrier tuple carries its own ResourceHandle so that
//     multi-resource boundaries resolve correctly.
//
// Scenarios:
//   1.  Two textures at the same (from, to) boundary -- both barriers present
//   2.  Two buffers at the same boundary -- both present in buffer_barriers
//   3.  Mixed texture + buffer at the same boundary -- both present, correct type
//   4.  Three textures at the same boundary -- all three present
//   5.  Full API chain: compute_barriers -> generate_barriers, multi-resource
//   6.  Multi-resource across two different boundaries
//   7.  Same resource repeated (deduped) alongside distinct resource at same boundary
//   8.  Texture + buffer + texture at same boundary (interleaved types)
//   9.  Edge case: zero resources at boundary produces empty BarrierCommand
//  10.  Full chain with 4+ resources at same boundary
//
use renderer_backend::frame_graph::{
    compute_barriers, generate_barriers, mock_pass_compute, mock_pass_graphics,
    mock_resource_buffer, mock_resource_texture, wgpu_barrier_from_state_transition,
    BarrierCommand, BarrierDescriptor, BufferBarrierDescriptor, BufferDesc, EdgeType, IrEdge,
    IrPass, IrResource, PassIndex, ResourceDesc, ResourceHandle, ResourceState,
    TextureBarrierDescriptor, TextureDesc,
};

// =========================================================================
// SECTION 1 -- Two textures at same boundary
// =========================================================================

#[test]
fn two_textures_same_boundary_both_barriers_present() {
    // P0 writes r_tex_a and r_tex_b (both as ColorAttachment).
    // P1 reads both (ShaderRead). The same (from=P0, to=P1) boundary has
    // TWO texture barriers -- one per resource.
    let r_a = ResourceHandle(1);
    let r_b = ResourceHandle(2);
    let resources = vec![
        mock_resource_texture(r_a, "albedo", 1920, 1080),
        mock_resource_texture(r_b, "normal", 1920, 1080),
    ];
    let edges = vec![
        IrEdge::new(PassIndex(0), PassIndex(1), r_a, EdgeType::RAW),
        IrEdge::new(PassIndex(0), PassIndex(1), r_b, EdgeType::RAW),
    ];
    let passes = vec![
        mock_pass_graphics(PassIndex(0), "gbuffer", &[r_a, r_b]),
        // P1 reads both as shader resources (compute pass with reads).
        {
            let mut p = mock_pass_compute(PassIndex(1), "lighting", &[], &[]);
            p.access_set.reads.push(r_a);
            p.access_set.reads.push(r_b);
            p
        },
    ];
    let order = vec![PassIndex(0), PassIndex(1)];

    let barrier_tuples = compute_barriers(&order, &passes, &edges);
    // Two distinct resources at same boundary -> two barriers.
    assert_eq!(
        barrier_tuples.len(),
        2,
        "two resources at same boundary must produce two barrier tuples",
    );

    // Each tuple carries its own ResourceHandle.
    let handles: Vec<ResourceHandle> = barrier_tuples.iter().map(|t| t.4).collect();
    assert!(
        handles.contains(&r_a),
        "barrier for r_a must be present in 5-tuples",
    );
    assert!(
        handles.contains(&r_b),
        "barrier for r_b must be present in 5-tuples",
    );

    // Feed into generate_barriers -- both barriers must resolve.
    let commands = generate_barriers(&barrier_tuples, &passes, &edges, &resources);
    assert_eq!(commands.len(), 1, "one boundary = one BarrierCommand");

    let cmd = &commands[0];
    assert_eq!(
        cmd.texture_barriers.len(),
        2,
        "both texture barriers at same boundary must be present",
    );
    assert!(cmd.buffer_barriers.is_empty());

    // Each texture barrier references the correct resource.
    let tex_handles: Vec<ResourceHandle> =
        cmd.texture_barriers.iter().map(|tb| tb.resource).collect();
    assert!(tex_handles.contains(&r_a), "r_a texture barrier present");
    assert!(tex_handles.contains(&r_b), "r_b texture barrier present");
}

// =========================================================================
// SECTION 2 -- Two buffers at same boundary
// =========================================================================

#[test]
fn two_buffers_same_boundary_both_barriers_present() {
    let r_buf_a = ResourceHandle(10);
    let r_buf_b = ResourceHandle(11);
    let resources = vec![
        mock_resource_buffer(r_buf_a, "ssbo_a", 4096),
        mock_resource_buffer(r_buf_b, "ssbo_b", 8192),
    ];
    let edges = vec![
        IrEdge::new(PassIndex(0), PassIndex(1), r_buf_a, EdgeType::RAW),
        IrEdge::new(PassIndex(0), PassIndex(1), r_buf_b, EdgeType::RAW),
    ];
    // P0 writes both buffers; P1 reads both.
    let passes = vec![
        mock_pass_compute(PassIndex(0), "writer", &[], &[r_buf_a, r_buf_b]),
        mock_pass_compute(PassIndex(1), "reader", &[r_buf_a, r_buf_b], &[]),
    ];
    let order = vec![PassIndex(0), PassIndex(1)];

    let barrier_tuples = compute_barriers(&order, &passes, &edges);
    assert_eq!(
        barrier_tuples.len(),
        2,
        "two buffer resources at same boundary = two barrier 5-tuples",
    );

    // Verify each 5-tuple carries its ResourceHandle.
    let handles: Vec<ResourceHandle> = barrier_tuples.iter().map(|t| t.4).collect();
    assert!(handles.contains(&r_buf_a));
    assert!(handles.contains(&r_buf_b));

    let commands = generate_barriers(&barrier_tuples, &passes, &edges, &resources);
    assert_eq!(commands.len(), 1, "one boundary = one BarrierCommand");

    let cmd = &commands[0];
    assert!(cmd.texture_barriers.is_empty());
    assert_eq!(
        cmd.buffer_barriers.len(),
        2,
        "both buffer barriers at same boundary must be present",
    );

    let buf_handles: Vec<ResourceHandle> =
        cmd.buffer_barriers.iter().map(|bb| bb.resource).collect();
    assert!(buf_handles.contains(&r_buf_a), "r_buf_a present");
    assert!(buf_handles.contains(&r_buf_b), "r_buf_b present");
}

// =========================================================================
// SECTION 3 -- Mixed texture + buffer at same boundary
// =========================================================================

#[test]
fn mixed_texture_and_buffer_same_boundary() {
    let r_tex = ResourceHandle(1);
    let r_buf = ResourceHandle(10);
    let resources = vec![
        mock_resource_texture(r_tex, "albedo", 1920, 1080),
        mock_resource_buffer(r_buf, "ssbo", 4096),
    ];
    let edges = vec![
        IrEdge::new(PassIndex(0), PassIndex(1), r_tex, EdgeType::RAW),
        IrEdge::new(PassIndex(0), PassIndex(1), r_buf, EdgeType::RAW),
    ];
    // P0 writes texture (graphics color attachment) and buffer (compute storage write).
    // P1 reads both.
    let passes = vec![
        {
            let mut p = mock_pass_graphics(PassIndex(0), "write_both", &[r_tex]);
            p.access_set.writes.push(r_buf); // write buffer to trigger barrier
            p
        },
        {
            let mut p = mock_pass_compute(PassIndex(1), "read_both", &[], &[]);
            p.access_set.reads.push(r_tex);
            p.access_set.reads.push(r_buf);
            p
        },
    ];
    let order = vec![PassIndex(0), PassIndex(1)];

    let barrier_tuples = compute_barriers(&order, &passes, &edges);
    assert_eq!(
        barrier_tuples.len(),
        2,
        "two resources of different types = two barrier 5-tuples",
    );

    // Verify 5-tuple ResourceHandle correctness.
    let handles: Vec<ResourceHandle> = barrier_tuples.iter().map(|t| t.4).collect();
    assert!(
        handles.contains(&r_tex),
        "texture handle in barrier 5-tuples"
    );
    assert!(
        handles.contains(&r_buf),
        "buffer handle in barrier 5-tuples"
    );

    let commands = generate_barriers(&barrier_tuples, &passes, &edges, &resources);
    assert_eq!(commands.len(), 1, "one boundary = one BarrierCommand");

    let cmd = &commands[0];
    // One texture barrier + one buffer barrier at the same boundary.
    assert_eq!(
        cmd.texture_barriers.len(),
        1,
        "texture barrier present at mixed boundary",
    );
    assert_eq!(
        cmd.buffer_barriers.len(),
        1,
        "buffer barrier present at mixed boundary",
    );
    assert_eq!(
        cmd.texture_barriers[0].resource, r_tex,
        "texture barrier targets correct resource handle",
    );
    assert_eq!(
        cmd.buffer_barriers[0].resource, r_buf,
        "buffer barrier targets correct resource handle",
    );
}

// =========================================================================
// SECTION 4 -- Three textures at same boundary
// =========================================================================

#[test]
fn three_textures_same_boundary_all_present() {
    let r1 = ResourceHandle(1);
    let r2 = ResourceHandle(2);
    let r3 = ResourceHandle(3);
    let resources = vec![
        mock_resource_texture(r1, "color", 1920, 1080),
        mock_resource_texture(r2, "normal", 1920, 1080),
        mock_resource_texture(r3, "depth", 1920, 1080),
    ];
    let edges = vec![
        IrEdge::new(PassIndex(0), PassIndex(1), r1, EdgeType::RAW),
        IrEdge::new(PassIndex(0), PassIndex(1), r2, EdgeType::RAW),
        IrEdge::new(PassIndex(0), PassIndex(1), r3, EdgeType::RAW),
    ];
    let passes = vec![
        mock_pass_graphics(PassIndex(0), "gbuffer", &[r1, r2, r3]),
        {
            let mut p = mock_pass_compute(PassIndex(1), "deferred", &[], &[]);
            p.access_set.reads.push(r1);
            p.access_set.reads.push(r2);
            p.access_set.reads.push(r3);
            p
        },
    ];
    let order = vec![PassIndex(0), PassIndex(1)];

    let barrier_tuples = compute_barriers(&order, &passes, &edges);
    assert_eq!(barrier_tuples.len(), 3, "three resources = three 5-tuples");

    // Verify all three handles are present in 5-tuples.
    let handles: Vec<ResourceHandle> = barrier_tuples.iter().map(|t| t.4).collect();
    for r in &[r1, r2, r3] {
        assert!(
            handles.contains(r),
            "ResourceHandle({}) present in barrier 5-tuples",
            r.0
        );
    }

    let commands = generate_barriers(&barrier_tuples, &passes, &edges, &resources);
    assert_eq!(commands.len(), 1, "one boundary");
    assert_eq!(
        commands[0].texture_barriers.len(),
        3,
        "all three texture barriers present",
    );

    let tex_handles: Vec<ResourceHandle> = commands[0]
        .texture_barriers
        .iter()
        .map(|tb| tb.resource)
        .collect();
    for r in &[r1, r2, r3] {
        assert!(
            tex_handles.contains(r),
            "barrier for r{} resolved correctly",
            r.0
        );
    }
}

// =========================================================================
// SECTION 5 -- Full API chain: compute_barriers -> generate_barriers
// =========================================================================

#[test]
fn full_chain_multi_resource_same_boundary() {
    // Full pipeline: build edges, call compute_barriers, then
    // generate_barriers. Verify the 5-tuple carries correct ResourceHandle
    // all the way through to the BarrierCommand.
    let r_tex = ResourceHandle(1);
    let r_buf = ResourceHandle(10);
    let resources = vec![
        mock_resource_texture(r_tex, "gbuffer", 1920, 1080),
        mock_resource_buffer(r_buf, "storage", 4096),
    ];
    let passes = vec![
        {
            let mut p = mock_pass_graphics(PassIndex(0), "render", &[r_tex]);
            p.access_set.writes.push(r_buf);
            p
        },
        {
            let mut p = mock_pass_compute(PassIndex(1), "post", &[], &[]);
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

    // Phase 4: compute barriers (5-tuples with ResourceHandle).
    let barrier_tuples = compute_barriers(&order, &passes, &edges);
    assert_eq!(barrier_tuples.len(), 2, "two resources = two 5-tuples");

    // Verify 5-tuple structure: each entry is (from, to, before, after, handle).
    for &(from, to, _before, _after, handle) in &barrier_tuples {
        assert_eq!(from, PassIndex(0), "from is P0");
        assert_eq!(to, PassIndex(1), "to is P1");
        assert!(
            handle == r_tex || handle == r_buf,
            "handle is one of the two resources",
        );
    }

    // Phase 4b: generate BarrierCommands from the 5-tuples.
    let commands = generate_barriers(&barrier_tuples, &passes, &edges, &resources);
    assert_eq!(commands.len(), 1, "one boundary = one BarrierCommand");

    let cmd = &commands[0];
    assert_eq!(cmd.texture_barriers.len(), 1, "one texture barrier");
    assert_eq!(cmd.buffer_barriers.len(), 1, "one buffer barrier");
}

#[test]
fn full_chain_preserves_state_transition_per_resource() {
    // Verify each 5-tuple carries the correct (before, after) state pair
    // for its specific resource -- distinct resources may have different
    // state transitions at the same boundary.
    let r_write = ResourceHandle(1); // ColorAttachment -> ShaderRead
    let r_read = ResourceHandle(2); // ShaderRead -> ShaderRead (no barrier)
    let resources = vec![
        mock_resource_texture(r_write, "color", 800, 600),
        mock_resource_texture(r_read, "depth", 800, 600),
    ];
    // P0: writes r_write (ColorAttachment), reads r_read (ShaderRead).
    // P1: reads both.
    let passes = vec![
        {
            let mut p = mock_pass_graphics(PassIndex(0), "gbuffer", &[r_write]);
            p.access_set.reads.push(r_read);
            p
        },
        {
            let mut p = mock_pass_compute(PassIndex(1), "lighting", &[], &[]);
            p.access_set.reads.push(r_write);
            p.access_set.reads.push(r_read);
            p
        },
    ];
    let edges = vec![
        IrEdge::new(PassIndex(0), PassIndex(1), r_write, EdgeType::RAW),
        IrEdge::new(PassIndex(0), PassIndex(1), r_read, EdgeType::RAW),
    ];
    let order = vec![PassIndex(0), PassIndex(1)];

    let barrier_tuples = compute_barriers(&order, &passes, &edges);

    // r_write transitions ColorAttachment -> ShaderRead.
    // r_read stays ShaderRead -> ShaderRead (no barrier emitted).
    // Only the resource with an actual state transition gets a barrier.
    assert_eq!(barrier_tuples.len(), 1, "only r_write transitions");

    // The single barrier 5-tuple must reference r_write with correct states.
    let (from, to, before, after, handle) = barrier_tuples[0];
    assert_eq!(from, PassIndex(0));
    assert_eq!(to, PassIndex(1));
    assert_eq!(handle, r_write, "barrier is for r_write, not r_read");
    assert_eq!(before, ResourceState::ColorAttachment);
    assert_eq!(after, ResourceState::ShaderRead);
}

// =========================================================================
// SECTION 6 -- Multi-resource across two different boundaries
// =========================================================================

#[test]
fn multi_resource_across_two_boundaries() {
    // P0 -> P1: r_tex_a and r_tex_b (textures)
    // P1 -> P2: r_buf_c (buffer)
    let r_tex_a = ResourceHandle(1);
    let r_tex_b = ResourceHandle(2);
    let r_buf_c = ResourceHandle(10);
    let resources = vec![
        mock_resource_texture(r_tex_a, "albedo", 1920, 1080),
        mock_resource_texture(r_tex_b, "normal", 1920, 1080),
        mock_resource_buffer(r_buf_c, "data", 4096),
    ];
    let edges = vec![
        IrEdge::new(PassIndex(0), PassIndex(1), r_tex_a, EdgeType::RAW),
        IrEdge::new(PassIndex(0), PassIndex(1), r_tex_b, EdgeType::RAW),
        IrEdge::new(PassIndex(1), PassIndex(2), r_buf_c, EdgeType::RAW),
    ];
    let passes = vec![
        mock_pass_graphics(PassIndex(0), "gbuffer", &[r_tex_a, r_tex_b]),
        {
            let mut p = mock_pass_compute(PassIndex(1), "compute", &[], &[]);
            p.access_set.reads.push(r_tex_a);
            p.access_set.reads.push(r_tex_b);
            p.access_set.writes.push(r_buf_c);
            p
        },
        mock_pass_compute(PassIndex(2), "readout", &[r_buf_c], &[]),
    ];
    let order = vec![PassIndex(0), PassIndex(1), PassIndex(2)];

    let barrier_tuples = compute_barriers(&order, &passes, &edges);
    // P0->P1: 2 barriers (r_tex_a, r_tex_b); P1->P2: 1 barrier (r_buf_c) = 3 total.
    assert_eq!(
        barrier_tuples.len(),
        3,
        "three total barriers across two boundaries"
    );

    // Verify 5-tuple handles.
    let p0p1_handles: Vec<ResourceHandle> = barrier_tuples
        .iter()
        .filter(|&&(from, to, ..)| from == PassIndex(0) && to == PassIndex(1))
        .map(|&(_, _, _, _, h)| h)
        .collect();
    assert_eq!(p0p1_handles.len(), 2, "two resources at P0->P1");
    assert!(p0p1_handles.contains(&r_tex_a));
    assert!(p0p1_handles.contains(&r_tex_b));

    let p1p2_handles: Vec<ResourceHandle> = barrier_tuples
        .iter()
        .filter(|&&(from, to, ..)| from == PassIndex(1) && to == PassIndex(2))
        .map(|&(_, _, _, _, h)| h)
        .collect();
    assert_eq!(p1p2_handles.len(), 1, "one resource at P1->P2");
    assert_eq!(p1p2_handles[0], r_buf_c);

    let commands = generate_barriers(&barrier_tuples, &passes, &edges, &resources);
    assert_eq!(commands.len(), 2, "two boundaries = two BarrierCommands");

    // First boundary (P0->P1): two texture barriers.
    assert_eq!(commands[0].texture_barriers.len(), 2);
    assert!(commands[0].buffer_barriers.is_empty());

    // Second boundary (P1->P2): one buffer barrier.
    assert!(commands[1].texture_barriers.is_empty());
    assert_eq!(commands[1].buffer_barriers.len(), 1);
    assert_eq!(commands[1].buffer_barriers[0].resource, r_buf_c);
}

// =========================================================================
// SECTION 7 -- Deduplication alongside distinct resource at same boundary
// =========================================================================

#[test]
fn dedup_same_resource_distinct_resource_preserved() {
    // Two edges for r_tex_a (same from, to, resource -> deduped in compute_barriers).
    // One edge for r_tex_b (distinct resource -> its own barrier).
    // Result should be: 2 barriers (one per unique resource), NOT 3.
    let r_a = ResourceHandle(1);
    let r_b = ResourceHandle(2);
    let resources = vec![
        mock_resource_texture(r_a, "color", 800, 600),
        mock_resource_texture(r_b, "depth", 800, 600),
    ];
    // Duplicate edges for r_a (same from, to, resource).
    let edges = vec![
        IrEdge::new(PassIndex(0), PassIndex(1), r_a, EdgeType::RAW),
        IrEdge::new(PassIndex(0), PassIndex(1), r_a, EdgeType::RAW), // duplicate
        IrEdge::new(PassIndex(0), PassIndex(1), r_b, EdgeType::RAW),
    ];
    let passes = vec![mock_pass_graphics(PassIndex(0), "gbuffer", &[r_a, r_b]), {
        let mut p = mock_pass_compute(PassIndex(1), "deferred", &[], &[]);
        p.access_set.reads.push(r_a);
        p.access_set.reads.push(r_b);
        p
    }];
    let order = vec![PassIndex(0), PassIndex(1)];

    let barrier_tuples = compute_barriers(&order, &passes, &edges);
    // compute_barriers deduplicates by (from, to, resource), so even though
    // there are 3 edges, only 2 barriers are emitted.
    assert_eq!(
        barrier_tuples.len(),
        2,
        "dedup: duplicate edges for same resource produce one barrier 5-tuple",
    );

    // Both unique resources must have their own barrier 5-tuple.
    let handles: Vec<ResourceHandle> = barrier_tuples.iter().map(|t| t.4).collect();
    assert!(handles.contains(&r_a), "r_a present");
    assert!(handles.contains(&r_b), "r_b present");
    // And r_a appears exactly once.
    let count_a = handles.iter().filter(|&&h| h == r_a).count();
    assert_eq!(count_a, 1, "r_a appears exactly once (deduped)");
}

// =========================================================================
// SECTION 8 -- Interleaved texture + buffer + texture at same boundary
// =========================================================================

#[test]
fn texture_buffer_texture_interleaved_same_boundary() {
    // Three resources at same boundary: texture, buffer, texture.
    // Types interleave, so verify texture_barriers has 2 entries and
    // buffer_barriers has 1 entry, all with correct resource handles.
    let r_tex_a = ResourceHandle(1);
    let r_buf = ResourceHandle(10);
    let r_tex_b = ResourceHandle(2);
    let resources = vec![
        mock_resource_texture(r_tex_a, "albedo", 1920, 1080),
        mock_resource_buffer(r_buf, "ssbo", 4096),
        mock_resource_texture(r_tex_b, "normal", 1920, 1080),
    ];
    let edges = vec![
        IrEdge::new(PassIndex(0), PassIndex(1), r_tex_a, EdgeType::RAW),
        IrEdge::new(PassIndex(0), PassIndex(1), r_buf, EdgeType::RAW),
        IrEdge::new(PassIndex(0), PassIndex(1), r_tex_b, EdgeType::RAW),
    ];
    let passes = vec![
        {
            let mut p = mock_pass_graphics(PassIndex(0), "gbuffer", &[r_tex_a, r_tex_b]);
            p.access_set.writes.push(r_buf);
            p
        },
        {
            let mut p = mock_pass_compute(PassIndex(1), "resolve", &[], &[]);
            p.access_set.reads.push(r_tex_a);
            p.access_set.reads.push(r_buf);
            p.access_set.reads.push(r_tex_b);
            p
        },
    ];
    let order = vec![PassIndex(0), PassIndex(1)];

    let barrier_tuples = compute_barriers(&order, &passes, &edges);
    assert_eq!(
        barrier_tuples.len(),
        3,
        "three resources = three barrier 5-tuples"
    );

    let commands = generate_barriers(&barrier_tuples, &passes, &edges, &resources);
    assert_eq!(commands.len(), 1, "one boundary");

    let cmd = &commands[0];
    // Two textures + one buffer separately listed.
    assert_eq!(
        cmd.texture_barriers.len(),
        2,
        "both texture barriers present",
    );
    assert_eq!(cmd.buffer_barriers.len(), 1, "buffer barrier present",);

    // Verify each texture barrier references a texture handle.
    for tb in &cmd.texture_barriers {
        assert!(
            tb.resource == r_tex_a || tb.resource == r_tex_b,
            "texture barrier handle is one of the texture resources",
        );
    }
    // Verify buffer barrier references the buffer handle.
    assert_eq!(cmd.buffer_barriers[0].resource, r_buf);
}

// =========================================================================
// SECTION 9 -- Edge case: boundary with no barriers produces empty command
// =========================================================================

#[test]
fn boundary_with_no_barriers_produces_no_command() {
    // Two passes where states match -> no barriers.
    let r_tex = ResourceHandle(1);
    let resources = vec![mock_resource_texture(r_tex, "shared", 800, 600)];
    // Both passes read r_tex as ShaderRead -> no transition.
    let edges = vec![IrEdge::new(
        PassIndex(0),
        PassIndex(1),
        r_tex,
        EdgeType::RAW,
    )];
    let passes = vec![
        mock_pass_compute(PassIndex(0), "read_a", &[r_tex], &[]),
        mock_pass_compute(PassIndex(1), "read_b", &[r_tex], &[]),
    ];
    let order = vec![PassIndex(0), PassIndex(1)];

    let barrier_tuples = compute_barriers(&order, &passes, &edges);
    assert!(
        barrier_tuples.is_empty(),
        "no state transition -> no barriers"
    );

    let commands = generate_barriers(&barrier_tuples, &passes, &edges, &resources);
    assert!(commands.is_empty(), "empty input -> empty output");
}

// =========================================================================
// SECTION 10 -- Four+ resources at same boundary
// =========================================================================

#[test]
fn four_textures_same_boundary_all_present() {
    let r = [
        ResourceHandle(1),
        ResourceHandle(2),
        ResourceHandle(3),
        ResourceHandle(4),
    ];
    let resources: Vec<IrResource> = r
        .iter()
        .map(|&h| mock_resource_texture(h, "mrrt", 1920, 1080))
        .collect();
    let edges: Vec<IrEdge> = r
        .iter()
        .map(|&h| IrEdge::new(PassIndex(0), PassIndex(1), h, EdgeType::RAW))
        .collect();
    let passes = vec![mock_pass_graphics(PassIndex(0), "mrt", &r), {
        let mut p = mock_pass_compute(PassIndex(1), "resolve", &[], &[]);
        for &h in &r {
            p.access_set.reads.push(h);
        }
        p
    }];
    let order = vec![PassIndex(0), PassIndex(1)];

    let barrier_tuples = compute_barriers(&order, &passes, &edges);
    assert_eq!(
        barrier_tuples.len(),
        4,
        "four resources = four barrier 5-tuples",
    );

    // Verify all four handles in the 5-tuples.
    let handles: Vec<ResourceHandle> = barrier_tuples.iter().map(|t| t.4).collect();
    for expected in &r {
        assert!(
            handles.contains(expected),
            "ResourceHandle({}) present in 5-tuples",
            expected.0,
        );
    }

    let commands = generate_barriers(&barrier_tuples, &passes, &edges, &resources);
    assert_eq!(commands.len(), 1, "one boundary");
    assert_eq!(
        commands[0].texture_barriers.len(),
        4,
        "all four texture barriers resolved",
    );

    let resolved: Vec<ResourceHandle> = commands[0]
        .texture_barriers
        .iter()
        .map(|tb| tb.resource)
        .collect();
    for expected in &r {
        assert!(
            resolved.contains(expected),
            "ResourceHandle({}) resolved in BarrierCommand",
            expected.0,
        );
    }
}

// =========================================================================
// SECTION 11 -- Resources with same from/to but different edge types
// =========================================================================

#[test]
fn multi_resource_different_edge_types_same_boundary() {
    // Two resources at same boundary, one RAW one WAR. Both still need
    // barriers and each must carry its correct ResourceHandle.
    let r_tex = ResourceHandle(1);
    let r_buf = ResourceHandle(10);
    let resources = vec![
        mock_resource_texture(r_tex, "color", 800, 600),
        mock_resource_buffer(r_buf, "storage", 4096),
    ];
    let edges = vec![
        IrEdge::new(PassIndex(0), PassIndex(1), r_tex, EdgeType::RAW),
        IrEdge::new(PassIndex(0), PassIndex(1), r_buf, EdgeType::WAR),
    ];
    let passes = vec![
        {
            let mut p = mock_pass_graphics(PassIndex(0), "render", &[r_tex]);
            p.access_set.reads.push(r_buf);
            p
        },
        {
            let mut p = mock_pass_compute(PassIndex(1), "post", &[], &[]);
            p.access_set.reads.push(r_tex);
            p.access_set.writes.push(r_buf);
            p
        },
    ];
    let order = vec![PassIndex(0), PassIndex(1)];

    let barrier_tuples = compute_barriers(&order, &passes, &edges);
    // Both resources need barriers (different edge types, different resources).
    assert_eq!(
        barrier_tuples.len(),
        2,
        "two resources with different edge types = two barrier 5-tuples",
    );

    let handles: Vec<ResourceHandle> = barrier_tuples.iter().map(|t| t.4).collect();
    assert!(handles.contains(&r_tex));
    assert!(handles.contains(&r_buf));

    let commands = generate_barriers(&barrier_tuples, &passes, &edges, &resources);
    assert_eq!(commands.len(), 1, "one boundary");
    assert_eq!(commands[0].texture_barriers.len(), 1);
    assert_eq!(commands[0].buffer_barriers.len(), 1);
    assert_eq!(commands[0].texture_barriers[0].resource, r_tex);
    assert_eq!(commands[0].buffer_barriers[0].resource, r_buf);
}
