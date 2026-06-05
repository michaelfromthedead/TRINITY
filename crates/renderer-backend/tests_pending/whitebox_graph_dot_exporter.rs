//! Whitebox tests for [`GraphDotExporter::to_dot`].
//!
//! T-FG-2.8: DEV implemented `to_dot()` on `GraphDotExporter` to produce
//! Graphviz DOT output from a [`CompiledFrameGraph`].  These tests exercise
//! node labels (pass name + type), edge labels (resource + state transition),
//! dead-pass barrier filtering, DOT escaping, preamble correctness, and the
//! empty-graph edge case.

use renderer_backend::frame_graph::{
    BufferDesc, CompiledFrameGraph, CullStats, DispatchSource, EdgeType, GraphDotExporter,
    InstanceSource, IrPass, IrResource, PassIndex, PassType, ResourceDesc,
    ResourceHandle, ResourceLifetime, ResourceState, TextureDesc, ViewType,
};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Helper: construct a minimal CompiledFrameGraph from parts
// ---------------------------------------------------------------------------

fn make_graph(
    passes: Vec<IrPass>,
    resources: Vec<IrResource>,
    order: Vec<PassIndex>,
    barriers: Vec<(PassIndex, PassIndex, ResourceHandle, EdgeType, ResourceState, ResourceState)>,
) -> CompiledFrameGraph {
    CompiledFrameGraph {
        passes,
        resources,
        edges: vec![],
        order,
        depths: HashMap::new(),
        barriers,
        async_passes: vec![],
        eliminated_passes: vec![],
        cull_stats: CullStats::default(),
        parallel_regions: vec![],
        compilation_time_us: 0,
        ..Default::default()
    }
}

/// Shortcut: a compute pass writing one storage buffer.
fn compute_pass(idx: u32, name: &str, write_handle: Option<u32>) -> IrPass {
    let mut pass = IrPass::compute(
        PassIndex(idx as usize),
        name,
        DispatchSource::Direct { group_count_x: 1, group_count_y: 1, group_count_z: 1 },
        ViewType::Storage,
    );
    if let Some(h) = write_handle {
        pass.access_set.writes.push(ResourceHandle(h));
    }
    pass
}

/// Shortcut: a graphics pass writing one color attachment.
fn graphics_pass(idx: u32, name: &str) -> IrPass {
    IrPass::graphics(
        PassIndex(idx as usize),
        name,
        vec![],
        None,
        InstanceSource::Direct { index_count: 6, instance_count: 1, base_vertex: 0, first_index: 0, first_instance: 0 },
        ViewType::Texture2D,
    )
}

fn resource_texture(handle: u32, name: &str) -> IrResource {
    IrResource::new(
        ResourceHandle(handle),
        name,
        ResourceDesc::Texture2D(TextureDesc {
            width: 256, height: 256, mip_levels: 1, array_layers: 1, format: "rgba8unorm".into(),
        }),
        ResourceLifetime::Transient,
        ResourceState::Uninitialized,
    )
}

fn resource_buffer(handle: u32, name: &str) -> IrResource {
    IrResource::new(
        ResourceHandle(handle),
        name,
        ResourceDesc::Buffer(BufferDesc { size: 4096, usage: "storage".into(), is_indirect_arg: false }),
        ResourceLifetime::Transient,
        ResourceState::Uninitialized,
    )
}

/// Collect just the node lines (P\d+ \[label=...) from a DOT output.
fn node_lines(dot: &str) -> Vec<&str> {
    dot.lines()
        .filter(|l| l.trim().starts_with("P") && l.contains("[label="))
        .map(|l| l.trim())
        .collect()
}

/// Collect just the edge lines (P\d+ -> P\d+ \[label=...) from a DOT output.
fn edge_lines(dot: &str) -> Vec<&str> {
    dot.lines()
        .filter(|l| l.trim().contains("->") && l.contains("[label="))
        .map(|l| l.trim())
        .collect()
}

// ===========================================================================
// 1.  Empty graph -- no passes, no barriers
// ===========================================================================

#[test]
fn empty_graph_produces_valid_dot() {
    let graph = make_graph(vec![], vec![], vec![], vec![]);
    let dot = GraphDotExporter::to_dot(&graph);

    // Preamble.
    assert!(dot.starts_with("digraph FrameGraph {"), "must open with digraph");
    assert!(dot.contains("rankdir=LR;"), "must contain rankdir directive");
    assert!(dot.contains("node [shape=box, style=rounded];"), "must contain node style");
    assert!(dot.ends_with("}\n"), "must close with brace");

    // No nodes, no edges.
    assert!(node_lines(&dot).is_empty(), "no nodes expected in empty graph");
    assert!(edge_lines(&dot).is_empty(), "no edges expected in empty graph");
}

// ===========================================================================
// 2.  Single compute pass node label
// ===========================================================================

#[test]
fn single_compute_pass_node_label() {
    let pass = compute_pass(0, "tonemap", Some(1));
    let res = resource_buffer(1, "framebuf");
    let graph = make_graph(vec![pass], vec![res], vec![PassIndex(0)], vec![]);
    let dot = GraphDotExporter::to_dot(&graph);

    let nodes = node_lines(&dot);
    assert_eq!(nodes.len(), 1, "one node expected");
    assert!(
        nodes[0].contains("P0") && nodes[0].contains("tonemap") && nodes[0].contains("Compute"),
        "node must contain 'P0', 'tonemap', and 'Compute'; got: {}",
        nodes[0],
    );
}

// ===========================================================================
// 3.  Single graphics pass node label
// ===========================================================================

#[test]
fn single_graphics_pass_node_label() {
    let pass = graphics_pass(0, "shadow_map");
    let graph = make_graph(vec![pass], vec![], vec![PassIndex(0)], vec![]);
    let dot = GraphDotExporter::to_dot(&graph);

    let nodes = node_lines(&dot);
    assert_eq!(nodes.len(), 1);
    assert!(
        nodes[0].contains("P0") && nodes[0].contains("shadow_map") && nodes[0].contains("Graphics"),
        "graphics pass node; got: {}",
        nodes[0],
    );
}

// ===========================================================================
// 4.  All four pass types produce correct labels
// ===========================================================================

#[test]
fn all_pass_types_produce_correct_node_labels() {
    let p0 = graphics_pass(0, "gbuffer");          // Graphics
    let p1 = compute_pass(1, "post", None);        // Compute
    let p2 = {
        let mut p = IrPass::compute(
            PassIndex(2),
            "copy_out",
            DispatchSource::Direct { group_count_x: 1, group_count_y: 1, group_count_z: 1 },
            ViewType::Storage,
        );
        p.pass_type = PassType::Copy;
        p
    };
    let p3 = {
        let mut p = IrPass::compute(
            PassIndex(3),
            "rt_reflections",
            DispatchSource::Direct { group_count_x: 1, group_count_y: 1, group_count_z: 1 },
            ViewType::Storage,
        );
        p.pass_type = PassType::RayTracing;
        p
    };

    let order = vec![PassIndex(0), PassIndex(1), PassIndex(2), PassIndex(3)];
    let graph = make_graph(vec![p0, p1, p2, p3], vec![], order, vec![]);
    let dot = GraphDotExporter::to_dot(&graph);
    let nodes = node_lines(&dot);

    assert_eq!(nodes.len(), 4);
    assert!(nodes[0].contains("gbuffer") && nodes[0].contains("Graphics"));
    assert!(nodes[1].contains("post") && nodes[1].contains("Compute"));
    assert!(nodes[2].contains("copy_out") && nodes[2].contains("Copy"));
    assert!(nodes[3].contains("rt_reflections") && nodes[3].contains("RayTracing"));
}

// ===========================================================================
// 5.  Edge label from a barrier between two passes
// ===========================================================================

#[test]
fn barrier_produces_edge_with_resource_and_state_transition() {
    let p0 = compute_pass(0, "write_pass", Some(1));
    let p1 = compute_pass(1, "read_pass", Some(2));
    let res = resource_texture(1, "frame");
    let barriers = vec![(
        PassIndex(0), PassIndex(1),
        ResourceHandle(1),
        EdgeType::RAW,
        ResourceState::ColorAttachment,
        ResourceState::ShaderRead,
    )];
    let graph = make_graph(
        vec![p0, p1], vec![res],
        vec![PassIndex(0), PassIndex(1)],
        barriers,
    );
    let dot = GraphDotExporter::to_dot(&graph);
    let edges = edge_lines(&dot);

    assert_eq!(edges.len(), 1, "one edge expected");
    assert!(
        edges[0].contains("P0 -> P1"),
        "edge must go from P0 to P1; got: {}",
        edges[0],
    );
    assert!(
        edges[0].contains("frame"),
        "edge label must contain resource name 'frame'; got: {}",
        edges[0],
    );
    assert!(
        edges[0].contains("ColorAttachment -> ShaderRead"),
        "edge label must contain state transition 'ColorAttachment -> ShaderRead'; got: {}",
        edges[0],
    );
    assert!(
        edges[0].contains("(1)"),
        "edge label must contain resource handle '(1)'; got: {}",
        edges[0],
    );
}

// ===========================================================================
// 6.  Dead-pass filtering -- barrier involving eliminated pass is omitted
// ===========================================================================

#[test]
fn barriers_involving_dead_passes_are_omitted() {
    let p0 = compute_pass(0, "alive_before", Some(1));
    let p1 = compute_pass(1, "dead_pass", Some(1));
    let p2 = compute_pass(2, "alive_after", Some(1));
    let res = resource_buffer(1, "buf");

    // order: p0, p2 are alive; p1 is eliminated (not in order)
    let barriers = vec![
        (PassIndex(0), PassIndex(1), ResourceHandle(1), EdgeType::RAW, ResourceState::ShaderRead, ResourceState::ShaderReadWrite),
        (PassIndex(1), PassIndex(2), ResourceHandle(1), EdgeType::RAW, ResourceState::ShaderReadWrite, ResourceState::ShaderRead),
        (PassIndex(0), PassIndex(2), ResourceHandle(1), EdgeType::RAW, ResourceState::ShaderRead, ResourceState::ShaderRead),
    ];
    let graph = make_graph(
        vec![p0, p1, p2], vec![res],
        vec![PassIndex(0), PassIndex(2)],  // p1 dead
        barriers,
    );
    let dot = GraphDotExporter::to_dot(&graph);
    let edges = edge_lines(&dot);

    assert_eq!(edges.len(), 1, "only the P0->P2 edge should survive dead-pass filtering");
    assert!(
        edges[0].contains("P0 -> P2"),
        "surviving edge must be P0 -> P2; got: {}",
        edges[0],
    );
}

// ===========================================================================
// 7.  Pass name with special characters is escaped
// ===========================================================================

#[test]
fn pass_name_with_special_chars_is_escaped() {
    // Name containing double-quote, backslash, and explicit \n.
    let mut p0 = compute_pass(0, "my\"pass\\name\nline2", None);
    // Also test with "Graphics" as type for better readability.
    p0.pass_type = PassType::Graphics;
    let graph = make_graph(vec![p0], vec![], vec![PassIndex(0)], vec![]);
    let dot = GraphDotExporter::to_dot(&graph);

    // The label should be: my\"pass\\name\nline2\n(Graphics)  -- all escaped
    let nodes = node_lines(&dot);
    assert_eq!(nodes.len(), 1);
    // In the raw DOT output double-quotes become \", backslashes become \\,
    // and \n stays as \n (DOT renders it as line break).
    assert!(
        nodes[0].contains("my\\\"pass\\\\name"),
        "backslash and quote must be escaped; got: {}",
        nodes[0],
    );
    assert!(nodes[0].contains("\\nline2"), "embedded newline must survive as \\\\n");
    assert!(nodes[0].contains("Graphics"), "pass type must appear");
}

// ===========================================================================
// 8.  Resource name fallback when handle is missing from resource lookup
// ===========================================================================

#[test]
fn edge_label_fallback_when_resource_handle_unknown() {
    let p0 = compute_pass(0, "src", Some(1));
    let p1 = compute_pass(1, "dst", Some(2));
    // Resources list is empty -- handle 1 won't be found.
    let barriers = vec![(
        PassIndex(0), PassIndex(1),
        ResourceHandle(1),
        EdgeType::RAW,
        ResourceState::Uninitialized,
        ResourceState::ShaderRead,
    )];
    let graph = make_graph(
        vec![p0, p1], vec![],
        vec![PassIndex(0), PassIndex(1)],
        barriers,
    );
    let dot = GraphDotExporter::to_dot(&graph);
    let edges = edge_lines(&dot);

    assert_eq!(edges.len(), 1);
    // Fallback: no resource name prefix, just "(1)": Uninitialized -> ShaderRead
    assert!(
        edges[0].contains("(1)"),
        "handle must still appear even without a name; got: {}",
        edges[0],
    );
    assert!(
        !edges[0].contains("(null)"),
        "should not contain '(null)' text",
    );
}

// ===========================================================================
// 9.  Nodes appear in topological order from `order`
// ===========================================================================

#[test]
fn nodes_appear_in_topological_order() {
    let p0 = compute_pass(0, "first", None);
    let p1 = compute_pass(1, "second", None);
    let p2 = compute_pass(2, "third", None);
    // order deliberately shuffled.
    let order = vec![PassIndex(1), PassIndex(2), PassIndex(0)];
    let graph = make_graph(vec![p0, p1, p2], vec![], order, vec![]);
    let dot = GraphDotExporter::to_dot(&graph);

    let nodes = node_lines(&dot);
    assert_eq!(nodes.len(), 3);
    // Must appear in order: P1, P2, P0.
    assert!(nodes[0].starts_with("P1"), "first node must be P1; got: {}", nodes[0]);
    assert!(nodes[1].starts_with("P2"), "second node must be P2; got: {}", nodes[1]);
    assert!(nodes[2].starts_with("P0"), "third node must be P0; got: {}", nodes[2]);
}

// ===========================================================================
// 10. Multiple edges between the same two passes with different resources
// ===========================================================================

#[test]
fn multiple_edges_with_different_resources() {
    let p0 = compute_pass(0, "src", None);
    let p1 = compute_pass(1, "dst", None);
    let r1 = resource_buffer(1, "buf_a");
    let r2 = resource_buffer(2, "buf_b");
    let barriers = vec![
        (PassIndex(0), PassIndex(1), ResourceHandle(1), EdgeType::RAW, ResourceState::ShaderRead, ResourceState::ShaderReadWrite),
        (PassIndex(0), PassIndex(1), ResourceHandle(2), EdgeType::RAW, ResourceState::VertexBuffer, ResourceState::ShaderRead),
    ];
    let graph = make_graph(
        vec![p0, p1], vec![r1, r2],
        vec![PassIndex(0), PassIndex(1)],
        barriers,
    );
    let dot = GraphDotExporter::to_dot(&graph);
    let edges = edge_lines(&dot);

    assert_eq!(edges.len(), 2, "two edges expected for P0->P1");
    assert!(
        edges[0].contains("buf_a") && edges[1].contains("buf_b"),
        "edges must reference different resource names; got: {:?}",
        edges,
    );
}

// ===========================================================================
// 11. Preamble includes all expected DOT directives
// ===========================================================================

#[test]
fn preamble_contains_all_directives() {
    let p0 = compute_pass(0, "a", None);
    let graph = make_graph(vec![p0], vec![], vec![PassIndex(0)], vec![]);
    let dot = GraphDotExporter::to_dot(&graph);

    let preamble_lines: Vec<&str> = dot
        .lines()
        .take_while(|l| !l.trim().starts_with("P"))
        .collect();
    let full_preamble = preamble_lines.join("\n");

    assert!(full_preamble.contains("digraph FrameGraph {"));
    assert!(full_preamble.contains("rankdir=LR;"));
    assert!(full_preamble.contains("node [shape=box, style=rounded];"));
    assert!(full_preamble.contains("edge [fontsize=10];"));
}

// ===========================================================================
// 12. Footer closes the digraph
// ===========================================================================

#[test]
fn footer_closes_digraph() {
    let p0 = compute_pass(0, "a", None);
    let graph = make_graph(vec![p0], vec![], vec![PassIndex(0)], vec![]);
    let dot = GraphDotExporter::to_dot(&graph);
    assert!(dot.ends_with("}\n"), "DOT output must end with '}}\\n'");
}

// ===========================================================================
// 13. Pass not found in passes list is silently skipped (robustness)
// ===========================================================================

#[test]
fn pass_index_in_order_without_corresponding_pass_is_skipped() {
    // order references P0, but passes list is empty.
    let graph = make_graph(vec![], vec![], vec![PassIndex(0)], vec![]);
    let dot = GraphDotExporter::to_dot(&graph);
    // Must not panic, must produce valid DOT with no nodes.
    assert!(node_lines(&dot).is_empty());
    assert!(dot.contains("digraph FrameGraph {"));
}

// ===========================================================================
// 14. Multiple barriers produce correct edge lines
// ===========================================================================

#[test]
fn multiple_barriers_produce_correct_edge_lines() {
    let p0 = compute_pass(0, "a", Some(1));
    let p1 = compute_pass(1, "b", Some(2));
    let p2 = compute_pass(2, "c", Some(3));
    let r1 = resource_texture(1, "tex1");
    let r2 = resource_texture(2, "tex2");
    let r3 = resource_texture(3, "tex3");
    let barriers = vec![
        (PassIndex(0), PassIndex(1), ResourceHandle(1), EdgeType::RAW, ResourceState::ColorAttachment, ResourceState::ShaderRead),
        (PassIndex(1), PassIndex(2), ResourceHandle(2), EdgeType::RAW, ResourceState::ShaderRead, ResourceState::TransferSrc),
    ];
    let graph = make_graph(
        vec![p0, p1, p2], vec![r1, r2, r3],
        vec![PassIndex(0), PassIndex(1), PassIndex(2)],
        barriers,
    );
    let dot = GraphDotExporter::to_dot(&graph);
    let edges = edge_lines(&dot);

    assert_eq!(edges.len(), 2);
    assert!(edges[0].contains("P0 -> P1") && edges[0].contains("tex1"));
    assert!(edges[1].contains("P1 -> P2") && edges[1].contains("tex2"));
}
