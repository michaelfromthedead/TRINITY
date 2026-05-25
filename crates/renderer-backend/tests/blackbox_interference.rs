// SPDX-License-Identifier: MIT
//
// blackbox_interference.rs -- Blackbox integration tests for T-FG-3.2
// InterferenceGraph, exercised through the public compilation API.
//
// CLEANROOM: No src/ access beyond the public API exported by the crate.
// Tests use only `renderer_backend::frame_graph::*` -- no internal fields,
// no private methods, no implementation details.
//
// The interference graph maps resources to other resources they conflict with
// (overlapping lifetimes OR incompatible formats).  These tests verify the
// interference graph is correctly built as part of the full compilation
// pipeline, accessible via `CompiledFrameGraph::interference_graph`.
//
// Test strategy:
//   Each test compiles a graph via CompiledFrameGraph::compile(), then
//   inspects the interference graph stored on the compiled output.  This
//   validates the full pipeline: pass ordering -> lifetime computation
//   -> interference graph construction.
//
// Public API under test:
//   CompiledFrameGraph::compile(passes, resources) -> Result<CompiledFrameGraph, String>
//   CompiledFrameGraph::interference_graph (field)
//   InterferenceGraph::interfere(a, b) -> bool
//   InterferenceGraph::neighbors(handle) -> &[ResourceHandle]
//
// Coverage (7 blackbox scenarios):
//   1.  Two resources with non-overlapping lifetimes -> graph is built, no edge
//   2.  Two resources with overlapping lifetimes -> graph shows conflict
//   3.  Multiple resources, verify interference graph structure
//   4.  Interference graph populated in CompiledFrameGraph after compilation
//   5.  Empty graph produces empty interference graph
//   6.  Single resource produces empty interference graph (nothing to conflict with)
//   7.  Non-overlapping resources have no interference edges

use renderer_backend::frame_graph::{
    BufferDesc, CompiledFrameGraph, DispatchSource, InstanceSource, IrPass, IrResource, PassIndex,
    PassType, ResourceAccessSet, ResourceDesc, ResourceHandle, ResourceLifetime, ResourceState,
    TextureDesc, ViewType,
};

// =============================================================================
// Helpers
// =============================================================================

/// Creates a texture resource with given handle, name, format, and dimensions.
fn tex(handle: u32, name: &str, format: &str, w: u32, h: u32) -> IrResource {
    IrResource::new(
        ResourceHandle(handle),
        name,
        ResourceDesc::Texture2D(TextureDesc {
            width: w,
            height: h,
            mip_levels: 1,
            array_layers: 1,
            format: format.into(),
        }),
        ResourceLifetime::Transient,
        ResourceState::Uninitialized,
    )
}

/// Creates a buffer resource with given handle, name, and byte size.
fn buf(handle: u32, name: &str, size: u64) -> IrResource {
    IrResource::new(
        ResourceHandle(handle),
        name,
        ResourceDesc::Buffer(BufferDesc {
            size,
            usage: "storage".into(),
            is_indirect_arg: false,
        }),
        ResourceLifetime::Transient,
        ResourceState::Uninitialized,
    )
}

/// Creates a compute pass with given index, name, reads, and writes.
fn compute_pass(
    idx: usize,
    name: &str,
    reads: &[ResourceHandle],
    writes: &[ResourceHandle],
) -> IrPass {
    IrPass {
        index: PassIndex(idx),
        name: name.into(),
        pass_type: PassType::Compute,
        access_set: ResourceAccessSet {
            reads: reads.to_vec(),
            writes: writes.to_vec(),
        },
        color_attachments: Vec::new(),
        depth_stencil: None,
        instance_source: InstanceSource::Direct {
            index_count: 0,
            instance_count: 1,
            base_vertex: 0,
            first_index: 0,
            first_instance: 0,
        },
        dispatch_source: Some(DispatchSource::Direct {
            group_count_x: 1,
            group_count_y: 1,
            group_count_z: 1,
        }),
        view_type: ViewType::Storage,
        tags: Vec::new(),
    }
}

/// Compiles a graph via `CompiledFrameGraph::compile()` and returns the result.
///
/// The interference graph is populated inside `compile()` as part of Phase 3
/// (resource lifetime analysis).  Callers inspect it via
/// `compiled.interference_graph`.
fn compile_graph(passes: Vec<IrPass>, resources: Vec<IrResource>) -> CompiledFrameGraph {
    CompiledFrameGraph::compile(passes, resources).expect("Graph must compile successfully")
}

// =============================================================================
// SECTION 1 -- Non-overlapping lifetimes: graph is built, no edge
// =============================================================================

/// Two buffers used in disjoint passes (non-overlapping lifetimes).
/// The interference graph is built but there is no edge between the two
/// resources.
#[test]
fn non_overlapping_buffers_graph_built_without_edge() {
    let r0 = ResourceHandle(0);
    let r1 = ResourceHandle(1);

    let passes = vec![
        compute_pass(0, "p0", &[r0], &[]),
        compute_pass(1, "p1", &[r1], &[]),
    ];
    let resources = vec![buf(0, "res_a", 1024), buf(1, "res_b", 2048)];

    let compiled = compile_graph(passes, resources);
    let ig = &compiled.interference_graph;

    // R0 and R1 are non-overlapping -> no interference.
    assert!(
        !ig.interfere(r0, r1),
        "Disjoint resource lifetimes must not interfere",
    );
    assert!(
        ig.neighbors(r0).is_empty(),
        "R0 must have no neighbours (disjoint from R1)",
    );
    assert!(
        ig.neighbors(r1).is_empty(),
        "R1 must have no neighbours (disjoint from R0)",
    );
}

/// Two textures with the SAME format in disjoint passes.
/// Same-format, disjoint lifetimes: no interference edge.
#[test]
fn non_overlapping_same_format_textures_no_edge() {
    let r0 = ResourceHandle(0);
    let r1 = ResourceHandle(1);

    let passes = vec![
        compute_pass(0, "p0", &[r0], &[]),
        compute_pass(1, "p1", &[r1], &[]),
    ];
    let resources = vec![
        tex(0, "tex_a", "rgba8unorm", 100, 100),
        tex(1, "tex_b", "rgba8unorm", 200, 200),
    ];

    let compiled = compile_graph(passes, resources);
    let ig = &compiled.interference_graph;

    assert!(
        !ig.interfere(r0, r1),
        "Same-format textures with disjoint lifetimes must not interfere",
    );
}

// =============================================================================
// SECTION 2 -- Overlapping lifetimes: interference graph shows conflict
// =============================================================================

/// Two buffers used in the same pass: overlapping lifetime produces an
/// interference edge.
#[test]
fn overlapping_buffers_show_conflict() {
    let r0 = ResourceHandle(0);
    let r1 = ResourceHandle(1);

    let passes = vec![compute_pass(0, "shared_pass", &[r0, r1], &[])];
    let resources = vec![buf(0, "a", 1024), buf(1, "b", 2048)];

    let compiled = compile_graph(passes, resources);
    let ig = &compiled.interference_graph;

    assert!(
        ig.interfere(r0, r1),
        "Resources in the same pass must interfere",
    );
    assert!(ig.interfere(r1, r0), "Interference must be symmetric",);
}

/// Two textures with overlapping lifetimes (one written, one read across
/// adjacent passes).  Shows conflict in the interference graph.
#[test]
fn overlapping_textures_show_conflict() {
    let r0 = ResourceHandle(0);
    let r1 = ResourceHandle(1);

    let passes = vec![
        compute_pass(0, "write_a", &[], &[r0]),
        compute_pass(1, "middle", &[r0], &[r1]),
        compute_pass(2, "read_b", &[r1], &[]),
    ];
    let resources = vec![
        tex(0, "rt_a", "rgba8unorm", 100, 100),
        tex(1, "rt_b", "rgba16float", 100, 100),
    ];

    let compiled = compile_graph(passes, resources);
    let ig = &compiled.interference_graph;

    assert!(
        ig.interfere(r0, r1),
        "Textures with overlapping lifetimes must interfere",
    );
}

// =============================================================================
// SECTION 3 -- Multiple resources: verify interference graph structure
// =============================================================================

/// Three buffers, all in the same pass -> complete triangle (all pairwise
/// interference).
#[test]
fn three_resources_all_interfere() {
    let r0 = ResourceHandle(0);
    let r1 = ResourceHandle(1);
    let r2 = ResourceHandle(2);

    let passes = vec![compute_pass(0, "all", &[r0, r1, r2], &[])];
    let resources = vec![buf(0, "a", 1024), buf(1, "b", 2048), buf(2, "c", 4096)];

    let compiled = compile_graph(passes, resources);
    let ig = &compiled.interference_graph;

    // Complete triangle: every pair interferes.
    assert!(ig.interfere(r0, r1), "R0-R1 must interfere");
    assert!(ig.interfere(r0, r2), "R0-R2 must interfere");
    assert!(ig.interfere(r1, r2), "R1-R2 must interfere");

    // Each vertex has degree 2.
    assert_eq!(ig.neighbors(r0).len(), 2, "R0 degree is 2");
    assert_eq!(ig.neighbors(r1).len(), 2, "R1 degree is 2");
    assert_eq!(ig.neighbors(r2).len(), 2, "R2 degree is 2");
}

/// Chain: R0-R1 overlap, R1-R2 overlap, R0-R2 do NOT overlap.
/// Verify the interference graph has only the direct edges.
#[test]
fn chain_interference_direct_edges_only() {
    // Lifetimes: R0=[0,1], R1=[1,2], R2=[2,3]. R0-R1 overlap at 1.
    // R1-R2 overlap at 2. R0-R2: no overlap.
    let r0 = ResourceHandle(0);
    let r1 = ResourceHandle(1);
    let r2 = ResourceHandle(2);

    let passes = vec![
        compute_pass(0, "p0", &[r0], &[]),
        compute_pass(1, "p1", &[r0, r1], &[]),
        compute_pass(2, "p2", &[r1, r2], &[]),
        compute_pass(3, "p3", &[r2], &[]),
    ];
    let resources = vec![
        buf(0, "chain_a", 1024),
        buf(1, "chain_b", 2048),
        buf(2, "chain_c", 4096),
    ];

    let compiled = compile_graph(passes, resources);
    let ig = &compiled.interference_graph;

    // Direct edges exist.
    assert!(ig.interfere(r0, r1), "R0-R1 must interfere (overlap at 1)");
    assert!(ig.interfere(r1, r2), "R1-R2 must interfere (overlap at 2)");

    // No transitive edge.
    assert!(
        !ig.interfere(r0, r2),
        "R0-R2 must NOT directly interfere (no transitive closure)",
    );

    // Verify degrees.
    assert_eq!(ig.neighbors(r0).len(), 1, "R0 degree is 1 (only R1)");
    assert_eq!(ig.neighbors(r1).len(), 2, "R1 degree is 2 (R0 and R2)");
    assert_eq!(ig.neighbors(r2).len(), 1, "R2 degree is 1 (only R1)");
}

// =============================================================================
// SECTION 4 -- Interference graph populated after compilation
// =============================================================================

/// Full compilation pipeline produces a populated interference graph.
/// After compiling a graph with multiple overlapping resources, the
/// interference graph should have entries and edges.
#[test]
fn interference_graph_populated_after_compilation() {
    // R0 written by p0, read by p1.  R1 used in p1 only.  R2 used in p2 only.
    // Lifetimes: R0=[0,1], R1=[1,1], R2=[2,2].
    // R0 and R1 overlap at pass 1 -> interfere.
    // R0 and R2: disjoint -> no interfere.  R1 and R2: disjoint -> no interfere.
    let r0 = ResourceHandle(0);
    let r1 = ResourceHandle(1);
    let r2 = ResourceHandle(2);

    let passes = vec![
        compute_pass(0, "write_a", &[], &[r0]),
        compute_pass(1, "mid", &[r0, r1], &[]),
        compute_pass(2, "last", &[r2], &[]),
    ];
    let resources = vec![buf(0, "a", 1024), buf(1, "b", 2048), buf(2, "c", 4096)];

    let compiled = compile_graph(passes, resources);
    let ig = &compiled.interference_graph;

    // R0-R1 interfere (overlap at pass 1).
    assert!(
        ig.interfere(r0, r1),
        "R0 and R1 must interfere (overlapping lifetimes)",
    );
    // R0-R2 do NOT interfere (disjoint lifetimes).
    assert!(
        !ig.interfere(r0, r2),
        "R0 and R2 must not interfere (disjoint lifetimes)",
    );
    // R1-R2 do NOT interfere (disjoint lifetimes).
    assert!(
        !ig.interfere(r1, r2),
        "R1 and R2 must not interfere (disjoint lifetimes)",
    );

    // R0 has exactly one neighbour (R1).
    assert_eq!(ig.neighbors(r0).len(), 1);
    assert!(ig.neighbors(r0).contains(&r1));

    // R1 has exactly one neighbour (R0).
    assert_eq!(ig.neighbors(r1).len(), 1);
    assert!(ig.neighbors(r1).contains(&r0));

    // R2 has no neighbours (isolated).
    assert!(ig.neighbors(r2).is_empty());
}

/// After compilation, the interference graph correctly reflects format-based
/// interference for textures with different formats (even when lifetimes are
/// disjoint).
#[test]
fn format_mismatch_interference_after_compilation() {
    let r0 = ResourceHandle(0);
    let r1 = ResourceHandle(1);

    // Two textures with different formats, used in disjoint passes.
    // Format mismatch causes interference despite disjoint lifetimes.
    let passes = vec![
        compute_pass(0, "pass0", &[r0], &[]),
        compute_pass(1, "pass1", &[r1], &[]),
    ];
    let resources = vec![
        tex(0, "color", "rgba8unorm", 1920, 1080),
        tex(1, "depth", "r32float", 1920, 1080),
    ];

    let compiled = compile_graph(passes, resources);
    let ig = &compiled.interference_graph;

    assert!(
        ig.interfere(r0, r1),
        "Textures with different formats must interfere even with disjoint lifetimes",
    );
    assert!(
        ig.interfere(r1, r0),
        "Format-based interference must be symmetric",
    );
}

/// After compilation, buffer resources do NOT format-interfere -- they only
/// interfere via lifetime overlap.
#[test]
fn buffers_no_format_interference_after_compilation() {
    let r0 = ResourceHandle(0);
    let r1 = ResourceHandle(1);

    // Two buffers with disjoint lifetimes -> no interference (buffers never
    // format-interfere).
    let passes = vec![
        compute_pass(0, "pass0", &[r0], &[]),
        compute_pass(1, "pass1", &[r1], &[]),
    ];
    let resources = vec![buf(0, "buf_a", 4096), buf(1, "buf_b", 8192)];

    let compiled = compile_graph(passes, resources);
    let ig = &compiled.interference_graph;

    assert!(
        !ig.interfere(r0, r1),
        "Buffers with disjoint lifetimes must not interfere (no format mismatch)",
    );
}

// =============================================================================
// SECTION 5 -- Empty graph produces empty interference graph
// =============================================================================

/// Compiling an empty graph (no passes, no resources) produces an empty
/// interference graph with no entries and no edges.
#[test]
fn empty_graph_empty_interference() {
    let passes: Vec<IrPass> = vec![];
    let resources: Vec<IrResource> = vec![];

    let compiled = compile_graph(passes, resources);
    let ig = &compiled.interference_graph;

    // Interference graph is empty (no edges, no neighbours).
    assert!(
        !ig.interfere(ResourceHandle(0), ResourceHandle(1)),
        "Empty graph: interfere must return false",
    );
    assert!(
        ig.neighbors(ResourceHandle(0)).is_empty(),
        "Empty graph: neighbours must be empty",
    );
    assert!(
        ig.neighbors(ResourceHandle(99)).is_empty(),
        "Empty graph: neighbours for any handle must be empty",
    );
}

/// Compiling with passes but NO resources produces an empty interference
/// graph.
#[test]
fn no_resources_empty_interference() {
    let passes = vec![compute_pass(0, "empty_pass", &[], &[])];
    let resources: Vec<IrResource> = vec![];

    let compiled = compile_graph(passes, resources);
    let ig = &compiled.interference_graph;

    assert!(
        ig.neighbors(ResourceHandle(0)).is_empty(),
        "No resources -> no neighbours",
    );
}

// =============================================================================
// SECTION 6 -- Single resource: empty interference (nothing to conflict with)
// =============================================================================

/// A single resource used by a single pass produces an empty interference
/// graph (no other resources to conflict with).
#[test]
fn single_resource_no_interference() {
    let r0 = ResourceHandle(0);

    let passes = vec![compute_pass(0, "single_pass", &[r0], &[])];
    let resources = vec![buf(0, "lonely", 1024)];

    let compiled = compile_graph(passes, resources);
    let ig = &compiled.interference_graph;

    // No neighbours (nothing to conflict with).
    assert!(
        ig.neighbors(r0).is_empty(),
        "Single resource must have no neighbours",
    );
    // No self-loop.
    assert!(
        !ig.interfere(r0, r0),
        "Single resource must not self-interfere",
    );
}

/// Single resource used across three passes still has no interference
/// neighbours (no other resources exist).
#[test]
fn single_resource_used_across_multiple_passes_no_interference() {
    let r0 = ResourceHandle(0);

    let passes = vec![
        compute_pass(0, "write", &[], &[r0]),
        compute_pass(1, "process", &[r0], &[]),
        compute_pass(2, "read", &[r0], &[]),
    ];
    let resources = vec![tex(0, "shared", "rgba8unorm", 100, 100)];

    let compiled = compile_graph(passes, resources);
    let ig = &compiled.interference_graph;

    // Still no neighbours -- only one resource in the graph.
    assert!(
        ig.neighbors(r0).is_empty(),
        "Single resource never has neighbours regardless of pass count",
    );
    assert!(
        !ig.interfere(r0, r0),
        "No self-interference for single resource",
    );
}

// =============================================================================
// SECTION 7 -- Non-overlapping resources: no interference edges
// =============================================================================

/// Three buffers, each in its own pass (fully sequential).  No pair has
/// overlapping lifetimes -> no interference edges.
#[test]
fn three_sequential_buffers_no_edges() {
    let r0 = ResourceHandle(0);
    let r1 = ResourceHandle(1);
    let r2 = ResourceHandle(2);

    let passes = vec![
        compute_pass(0, "p0", &[r0], &[]),
        compute_pass(1, "p1", &[r1], &[]),
        compute_pass(2, "p2", &[r2], &[]),
    ];
    let resources = vec![buf(0, "a", 1024), buf(1, "b", 2048), buf(2, "c", 4096)];

    let compiled = compile_graph(passes, resources);
    let ig = &compiled.interference_graph;

    // No pair interferes.
    assert!(!ig.interfere(r0, r1), "R0 and R1 must not interfere");
    assert!(!ig.interfere(r0, r2), "R0 and R2 must not interfere");
    assert!(!ig.interfere(r1, r2), "R1 and R2 must not interfere");

    // All neighbours lists are empty.
    assert!(ig.neighbors(r0).is_empty(), "R0 neighbours must be empty");
    assert!(ig.neighbors(r1).is_empty(), "R1 neighbours must be empty");
    assert!(ig.neighbors(r2).is_empty(), "R2 neighbours must be empty");
}

/// Same-format textures in sequential passes.  Same-format textures with
/// disjoint lifetimes do NOT interfere.
#[test]
fn same_format_textures_sequential_no_edges() {
    let r0 = ResourceHandle(0);
    let r1 = ResourceHandle(1);
    let r2 = ResourceHandle(2);

    let passes = vec![
        compute_pass(0, "p0", &[r0], &[]),
        compute_pass(1, "p1", &[r1], &[]),
        compute_pass(2, "p2", &[r2], &[]),
    ];
    let resources = vec![
        tex(0, "a", "rgba8unorm", 100, 100),
        tex(1, "b", "rgba8unorm", 200, 200),
        tex(2, "c", "rgba8unorm", 300, 300),
    ];

    let compiled = compile_graph(passes, resources);
    let ig = &compiled.interference_graph;

    assert!(!ig.interfere(r0, r1), "Same-format, disjoint: no edge");
    assert!(!ig.interfere(r0, r2), "Same-format, disjoint: no edge");
    assert!(!ig.interfere(r1, r2), "Same-format, disjoint: no edge");
}

/// Resources at the edge of non-overlap (adjacent passes): lifetimes [0,0]
/// and [1,1] do NOT overlap, so no interference edge.
#[test]
fn adjacent_passes_no_overlap() {
    let r0 = ResourceHandle(0);
    let r1 = ResourceHandle(1);

    let passes = vec![
        compute_pass(0, "first", &[r0], &[]),
        compute_pass(1, "second", &[r1], &[]),
    ];
    let resources = vec![
        tex(0, "early", "rgba8unorm", 100, 100),
        tex(1, "late", "rgba8unorm", 100, 100),
    ];

    let compiled = compile_graph(passes, resources);
    let ig = &compiled.interference_graph;

    assert!(
        !ig.interfere(r0, r1),
        "Adjacent passes (pass 0 and pass 1) with disjoint lifetimes must not interfere",
    );
}

/// Ten resources, each in its own sequential pass.  No pair of resources
/// shares a pass -> no interference edges in the entire graph.
#[test]
fn ten_sequential_non_overlapping_resources_no_edges() {
    let n: usize = 10;
    let handles: Vec<ResourceHandle> = (0..n).map(|i| ResourceHandle(i as u32)).collect();
    let resources: Vec<IrResource> = (0..n)
        .map(|i| buf(i as u32, &format!("buf_{}", i), 1024))
        .collect();
    let passes: Vec<IrPass> = (0..n)
        .map(|i| compute_pass(i, &format!("p{}", i), &[ResourceHandle(i as u32)], &[]))
        .collect();

    let compiled = compile_graph(passes, resources);
    let ig = &compiled.interference_graph;

    // No pair should interfere.
    for i in 0..n {
        for j in (i + 1)..n {
            assert!(
                !ig.interfere(handles[i], handles[j]),
                "Sequential non-overlapping resources R{} and R{} must not interfere",
                i,
                j,
            );
        }
        assert!(
            ig.neighbors(handles[i]).is_empty(),
            "Sequential non-overlapping resource R{} must have no neighbours",
            i,
        );
    }
}

/// A resource written by pass 0 and read by pass 1 does NOT interfere with
/// a resource only used in pass 2 (lifetimes are disjoint: [0,1] vs [2,2]).
#[test]
fn early_resource_does_not_interfere_with_late_resource() {
    let r_early = ResourceHandle(0);
    let r_late = ResourceHandle(1);

    let passes = vec![
        compute_pass(0, "write_early", &[], &[r_early]),
        compute_pass(1, "read_early", &[r_early], &[]),
        compute_pass(2, "use_late", &[r_late], &[]),
    ];
    let resources = vec![buf(0, "early", 4096), buf(1, "late", 2048)];

    let compiled = compile_graph(passes, resources);
    let ig = &compiled.interference_graph;

    assert!(
        !ig.interfere(r_early, r_late),
        "Early resource must not interfere with late resource (disjoint timelines)",
    );
    assert!(
        !ig.interfere(r_late, r_early),
        "Symmetry: late resource must not interfere with early resource",
    );
}

/// Interference graph is symmetric for all non-overlapping pairs: if
/// interfere(A, B) is false then interfere(B, A) is also false.
#[test]
fn non_overlapping_symmetry() {
    let r0 = ResourceHandle(0);
    let r1 = ResourceHandle(1);

    let passes = vec![
        compute_pass(0, "p0", &[r0], &[]),
        compute_pass(5, "p5", &[r1], &[]),
    ];
    let resources = vec![buf(0, "a", 1024), buf(1, "b", 2048)];

    let compiled = compile_graph(passes, resources);
    let ig = &compiled.interference_graph;

    assert_eq!(
        ig.interfere(r0, r1),
        ig.interfere(r1, r0),
        "Non-interference must be symmetric",
    );
    assert!(!ig.interfere(r0, r1), "Disjoint passes must not interfere",);
}
