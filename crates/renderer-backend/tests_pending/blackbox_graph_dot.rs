// SPDX-License-Identifier: MIT
//
// blackbox_graph_dot.rs -- Blackbox contract tests for T-FG-2.8 GraphDotExporter.
//
// CLEANROOM: No src/ access beyond the public API exported by the crate.
// Tests use only `renderer_backend::frame_graph::*` -- no internal fields,
// no private methods, no implementation details.
//
// Acceptance criterion (T-FG-2.8):
//   GraphDotExporter::to_dot(&compiled) returns a valid DOT digraph string
//   with nodes for surviving passes, edges for pipeline barriers, and
//   well-formed Graphviz syntax.
//
// Contract:
//   - Output starts with "digraph FrameGraph {" and ends with "}\n".
//   - Preamble includes rankdir=LR, node shape, edge fontsize.
//   - Surviving passes appear as nodes labeled "name\n(Type)".
//   - Eliminated (dead) passes do NOT appear as nodes.
//   - Barriers between surviving passes appear as directed edges.
//   - Barriers referencing eliminated passes are omitted.
//   - Edge labels show resource name, handle, and state transition.
//   - Pass names and types are DOT-escaped (quotes, backslashes).
//
// Coverage:
//   1.  Empty graph produces preamble + footer with no nodes or edges
//   2.  Single graphics pass node
//   3.  Single compute pass node (constructed directly)
//   4.  Single copy pass node (constructed directly)
//   5.  Two-pass RAW chain with barrier edge
//   6.  Three-pass sequential chain
//   7.  Fan-in: two producers, one consumer
//   8.  Dead pass (eliminated) omitted from DOT output
//   9.  Graphics pass survives in DOT even without consumer
//  10.  DOT preamble structure
//  11.  Balanced braces and trailing newline
//  12.  Edge label contains resource handle and state transition
//  13.  DOT escaping of special characters in pass names
//  14.  Multiple pass types in one graph (Graphics, Compute, Copy)
//  15.  Resource name appears in edge label
//  16.  Barrier referencing eliminated pass is omitted

use std::collections::HashMap;

use renderer_backend::frame_graph::{
    mock_pass_compute, mock_pass_graphics, mock_resource_buffer, mock_resource_texture,
    CompiledFrameGraph, CullStats, DispatchSource, FrameGraphCompiler, GraphDotExporter, IrPass,
    PassIndex, ResourceHandle, ResourceState, ViewType,
};

// Helper: count node lines (lines with [label=] but without ->
fn count_nodes(dot: &str) -> usize {
    dot.lines()
        .filter(|l| l.contains("[label=") && !l.contains("->"))
        .count()
}

// Helper: count edge lines (lines with ->)
fn count_edges(dot: &str) -> usize {
    dot.lines().filter(|l| l.contains("->")).count()
}

// =============================================================================
// SECTION 1 -- Empty graph: preamble + footer, no nodes, no edges
// =============================================================================

/// An empty compiled graph produces a structurally valid DOT digraph with zero
/// nodes and zero edges.
#[test]
fn empty_graph_dot_has_preamble_and_footer_only() {
    let compiler = FrameGraphCompiler::from_ir(vec![], vec![]);
    let compiled = compiler.expect("empty graph compiles");
    let dot = GraphDotExporter::to_dot(&compiled);

    let lines: Vec<&str> = dot.lines().collect();
    assert_eq!(lines[0], "digraph FrameGraph {", "first line is digraph header");
    assert!(dot.ends_with("}\n"), "output ends with closing brace + newline");

    // No nodes, no edges.
    assert_eq!(count_nodes(&dot), 0, "no nodes in empty graph");
    assert_eq!(count_edges(&dot), 0, "no edges in empty graph");
}

/// The preamble consists of exactly four lines: header, rankdir, node, edge.
#[test]
fn empty_graph_dot_preamble_lines() {
    let compiler = FrameGraphCompiler::from_ir(vec![], vec![]);
    let compiled = compiler.expect("empty graph compiles");
    let dot = GraphDotExporter::to_dot(&compiled);

    let lines: Vec<&str> = dot.lines().collect();
    assert_eq!(lines[0], "digraph FrameGraph {", "line 0: digraph header");
    assert_eq!(lines[1], "    rankdir=LR;", "line 1: rankdir");
    assert_eq!(lines[2], "    node [shape=box, style=rounded];", "line 2: node style");
    assert_eq!(lines[3], "    edge [fontsize=10];", "line 3: edge fontsize");
}

/// Balanced braces check: opening '{' must equal closing '}'.
#[test]
fn empty_graph_dot_balanced_braces() {
    let compiler = FrameGraphCompiler::from_ir(vec![], vec![]);
    let compiled = compiler.expect("empty graph compiles");
    let dot = GraphDotExporter::to_dot(&compiled);

    let open_braces = dot.matches('{').count();
    let close_braces = dot.matches('}').count();
    assert_eq!(
        open_braces, close_braces,
        "balanced braces: {} open, {} close",
        open_braces, close_braces,
    );
}

/// The output must end with a closing brace and a trailing newline.
#[test]
fn empty_graph_dot_trailing_newline() {
    let compiler = FrameGraphCompiler::from_ir(vec![], vec![]);
    let compiled = compiler.expect("empty graph compiles");
    let dot = GraphDotExporter::to_dot(&compiled);

    assert!(dot.ends_with("}\n"), "DOT must end with '}}\\n'");
}

// =============================================================================
// SECTION 2 -- Single pass nodes (direct construction for precise control)
// =============================================================================

/// Construct a minimal CompiledFrameGraph with a single pass.
fn single_pass_compiled(pass: IrPass) -> CompiledFrameGraph {
    let idx = pass.index;
    CompiledFrameGraph {
        passes: vec![pass],
        resources: vec![],
        edges: vec![],
        order: vec![idx],
        depths: HashMap::new(),
        barriers: vec![],
        async_passes: vec![],
        eliminated_passes: vec![],
        cull_stats: CullStats::default(),
        parallel_regions: vec![],
        compilation_time_us: 0,
    }
}

/// A single graphics pass produces one node labeled "name\n(Graphics)".
#[test]
fn single_graphics_pass_node() {
    let r = ResourceHandle(1);
    let passes = vec![mock_pass_graphics(PassIndex(0), "gbuffer", &[r])];
    let resources = vec![mock_resource_texture(r, "color", 1920, 1080)];

    let compiled = FrameGraphCompiler::from_ir(passes, resources)
        .compile()
        .expect("single graphics pass compiles");
    let dot = GraphDotExporter::to_dot(&compiled);

    // The node label contains the pass name and "(Graphics)" type.
    assert!(
        dot.contains(r#"P0 [label="gbuffer\"#),
        "Graphics pass node must contain P0 and gbuffer label"
    );
    assert!(
        dot.contains(r"(Graphics)"),
        "Graphics pass node must contain '(Graphics)' type indicator"
    );

    // Exactly one node (edge lines also contain [label= but we filter those out).
    assert_eq!(count_nodes(&dot), 1, "exactly one node for a single pass");
}

/// A single compute pass produces one node labeled "name\n(Compute)".
#[test]
fn single_compute_pass_node() {
    let pass = IrPass::compute(
        PassIndex(0),
        "lighting",
        DispatchSource::Direct {
            group_count_x: 1,
            group_count_y: 1,
            group_count_z: 1,
        },
        ViewType::Storage,
    );

    let compiled = single_pass_compiled(pass);
    let dot = GraphDotExporter::to_dot(&compiled);

    assert!(
        dot.contains(r#"P0 [label="lighting\"#),
        "Compute pass node must contain P0 and lighting label"
    );
    assert!(
        dot.contains(r"(Compute)"),
        "Compute pass node must contain '(Compute)' type"
    );
    assert_eq!(count_nodes(&dot), 1, "exactly one node");
}

/// A single copy pass produces one node labeled "name\n(Copy)".
#[test]
fn single_copy_pass_node() {
    let pass = IrPass::copy(PassIndex(0), "blit_out");

    let compiled = single_pass_compiled(pass);
    let dot = GraphDotExporter::to_dot(&compiled);

    assert!(
        dot.contains(r#"P0 [label="blit_out\"#),
        "Copy pass node must contain P0 and blit_out label"
    );
    assert!(
        dot.contains(r"(Copy)"),
        "Copy pass node must contain '(Copy)' type"
    );
    assert_eq!(count_nodes(&dot), 1, "exactly one node");
}

// =============================================================================
// SECTION 3 -- Multi-pass chains with barrier edges
// =============================================================================

/// A two-pass chain (P0 writes R1, P1 reads R1) produces a barrier edge
/// from P0 to P1 with a resource transition label.
#[test]
fn two_pass_chain_has_barrier_edge() {
    let r = ResourceHandle(1);
    let passes = vec![
        mock_pass_graphics(PassIndex(0), "gbuffer", &[r]),
        mock_pass_compute(PassIndex(1), "lighting", &[r], &[]),
    ];
    let resources = vec![mock_resource_texture(r, "albedo", 800, 600)];

    let compiled = FrameGraphCompiler::from_ir(passes, resources)
        .compile()
        .expect("two-pass chain compiles");
    let dot = GraphDotExporter::to_dot(&compiled);

    // Two nodes present.
    assert_eq!(count_nodes(&dot), 2, "two nodes for two passes");

    // Edge from P0 -> P1 with label.
    assert!(
        dot.contains("P0 -> P1"),
        "barrier edge must exist from P0 to P1"
    );

    // Exactly one edge line.
    assert_eq!(count_edges(&dot), 1, "exactly one barrier edge in two-pass chain");
}

/// A three-pass sequential chain produces two barrier edges.
#[test]
fn three_pass_chain_has_two_barrier_edges() {
    let r0 = ResourceHandle(1);
    let r1 = ResourceHandle(2);
    let passes = vec![
        mock_pass_graphics(PassIndex(0), "base", &[r0]),
        mock_pass_compute(PassIndex(1), "mid", &[r0], &[r1]),
        mock_pass_compute(PassIndex(2), "final", &[r1], &[]),
    ];
    let resources = vec![
        mock_resource_texture(r0, "g_buffer", 1920, 1080),
        mock_resource_buffer(r1, "intermediate", 4096),
    ];

    let compiled = FrameGraphCompiler::from_ir(passes, resources)
        .compile()
        .expect("three-pass chain compiles");
    let dot = GraphDotExporter::to_dot(&compiled);

    // Three nodes.
    assert_eq!(count_nodes(&dot), 3, "three nodes for three passes");

    // Two edges: P0->P1 and P1->P2.
    assert_eq!(count_edges(&dot), 2, "two barrier edges in three-pass chain");
    assert!(dot.contains("P0 -> P1"), "edge P0 -> P1 present");
    assert!(dot.contains("P1 -> P2"), "edge P1 -> P2 present");
}

/// Fan-in: two producers write separate resources, one consumer reads both.
#[test]
fn fan_in_produces_two_barrier_edges() {
    let r_tex = ResourceHandle(1);
    let r_buf = ResourceHandle(2);
    let passes = vec![
        mock_pass_graphics(PassIndex(0), "render_a", &[r_tex]),
        mock_pass_compute(PassIndex(1), "compute_b", &[], &[r_buf]),
        mock_pass_compute(PassIndex(2), "composite", &[r_tex, r_buf], &[]),
    ];
    let resources = vec![
        mock_resource_texture(r_tex, "tex_a", 800, 600),
        mock_resource_buffer(r_buf, "buf_b", 2048),
    ];

    let compiled = FrameGraphCompiler::from_ir(passes, resources)
        .compile()
        .expect("fan-in compiles");
    let dot = GraphDotExporter::to_dot(&compiled);

    // Three nodes.
    assert_eq!(count_nodes(&dot), 3, "three nodes for fan-in");

    // Two edges: P0->P2 (tex) and P1->P2 (buf).
    assert_eq!(count_edges(&dot), 2, "two barrier edges in fan-in topology");
    assert!(dot.contains("P0 -> P2"), "edge P0 -> P2 present");
    assert!(dot.contains("P1 -> P2"), "edge P1 -> P2 present");
}

// =============================================================================
// SECTION 4 -- Eliminated (dead) passes in DOT output
// =============================================================================

/// A pass NOT in `order` is treated as eliminated and omitted from DOT nodes.
/// Barriers referencing an eliminated pass are also omitted.
#[test]
fn eliminated_pass_omitted_from_dot_nodes_and_edges() {
    // Two surviving passes P0, P2, and one eliminated pass P1.
    // P0 writes R1, P1 (eliminated) writes R2, P2 reads R1.
    // Barrier P0->P2 (for R1) survives.
    // Barrier P1->P2 (for R2) is omitted because P1 is eliminated.
    let r1 = ResourceHandle(1);
    let r2 = ResourceHandle(2);

    let p0 = mock_pass_graphics(PassIndex(0), "alive_a", &[r1]);
    let p1 = IrPass::compute(
        PassIndex(1),
        "dead_pass",
        DispatchSource::Direct {
            group_count_x: 1,
            group_count_y: 1,
            group_count_z: 1,
        },
        ViewType::Storage,
    );
    let p2 = mock_pass_compute(PassIndex(2), "alive_b", &[r1], &[]);

    let compiled = CompiledFrameGraph {
        passes: vec![p0, p1, p2],
        resources: vec![
            mock_resource_texture(r1, "tex", 64, 64),
            mock_resource_buffer(r2, "buf", 1024),
        ],
        edges: vec![],
        order: vec![PassIndex(0), PassIndex(2)],  // P1 eliminated from order
        depths: HashMap::new(),
        barriers: vec![
            (PassIndex(0), PassIndex(2), r1, ResourceState::ColorAttachment, ResourceState::ShaderRead),
            // Barrier referencing P1 (eliminated) should be omitted
            (PassIndex(1), PassIndex(2), r2, ResourceState::ShaderReadWrite, ResourceState::ShaderRead),
        ],
        async_passes: vec![],
        eliminated_passes: vec![PassIndex(1)],
        cull_stats: CullStats::default(),
        parallel_regions: vec![],
        compilation_time_us: 0,
    };

    let dot = GraphDotExporter::to_dot(&compiled);

    // Only surviving passes appear as nodes.
    assert!(
        dot.contains("alive_a"),
        "surviving pass alive_a must appear as node"
    );
    assert!(
        dot.contains("alive_b"),
        "surviving pass alive_b must appear as node"
    );
    assert!(
        !dot.contains("dead_pass"),
        "eliminated pass dead_pass must NOT appear as node"
    );

    // Exactly two nodes.
    assert_eq!(count_nodes(&dot), 2, "two surviving nodes");

    // Only one barrier edge (P0->P2 survives; P1->P2 is omitted).
    assert_eq!(count_edges(&dot), 1, "only the barrier between surviving passes appears");
    assert!(
        dot.contains("P0 -> P2"),
        "edge P0 -> P2 must exist (both survive)"
    );
    assert!(
        !dot.contains("P1 -> P2"),
        "edge P1 -> P2 must be omitted (P1 eliminated)"
    );
}

/// A graphics pass is never eliminated by the compiler, so it appears in DOT.
#[test]
fn graphics_pass_survives_in_dot() {
    let r = ResourceHandle(1);
    let passes = vec![mock_pass_graphics(PassIndex(0), "final_output", &[r])];
    let resources = vec![mock_resource_texture(r, "swapchain", 1920, 1080)];

    let compiled = FrameGraphCompiler::from_ir(passes, resources)
        .compile()
        .expect("graphics pass compiles");
    let dot = GraphDotExporter::to_dot(&compiled);

    // The graphics pass node IS present.
    assert!(
        dot.contains("final_output"),
        "graphics pass node present in DOT output"
    );
    assert!(
        dot.contains("(Graphics)"),
        "graphics pass type indicator present"
    );
    assert_eq!(count_nodes(&dot), 1, "one node for surviving graphics pass");
}

// =============================================================================
// SECTION 5 -- Edge label format: resource info in labels
// =============================================================================

/// Barrier edges contain resource handle numbers and state transitions.
#[test]
fn edge_label_contains_resource_handle() {
    let r = ResourceHandle(42);
    let passes = vec![
        mock_pass_graphics(PassIndex(0), "producer", &[r]),
        mock_pass_compute(PassIndex(1), "consumer", &[r], &[]),
    ];
    let resources = vec![mock_resource_texture(r, "tex", 64, 64)];

    let compiled = FrameGraphCompiler::from_ir(passes, resources)
        .compile()
        .expect("chain compiles");
    let dot = GraphDotExporter::to_dot(&compiled);

    // Find the edge line and verify it contains the resource handle.
    let edge_line = dot
        .lines()
        .find(|l| l.contains("P0 -> P1"))
        .expect("edge P0 -> P1 must exist");

    // The edge label should contain the resource handle number (42).
    assert!(
        edge_line.contains("(42)"),
        "edge label must contain resource handle (42): got {:?}",
        edge_line,
    );

    // The edge must have a label attribute.
    assert!(
        edge_line.contains("[label=\""),
        "edge must have quoted label attribute"
    );
}

/// The resource name appears in the edge label when available.
#[test]
fn edge_label_contains_resource_name() {
    let r = ResourceHandle(7);
    let passes = vec![
        mock_pass_graphics(PassIndex(0), "writer", &[r]),
        mock_pass_compute(PassIndex(1), "reader", &[r], &[]),
    ];
    let resources = vec![mock_resource_texture(r, "my_resource", 512, 512)];

    let compiled = FrameGraphCompiler::from_ir(passes, resources)
        .compile()
        .expect("chain compiles");
    let dot = GraphDotExporter::to_dot(&compiled);

    let edge_line = dot
        .lines()
        .find(|l| l.contains("P0 -> P1"))
        .expect("edge P0 -> P1 must exist");

    assert!(
        edge_line.contains("my_resource"),
        "edge label must contain resource name 'my_resource': got {:?}",
        edge_line,
    );
}

/// Edge labels contain a state transition arrow (-> within the label text).
#[test]
fn edge_label_has_state_transition() {
    let r = ResourceHandle(1);
    let passes = vec![
        mock_pass_graphics(PassIndex(0), "producer", &[r]),
        mock_pass_compute(PassIndex(1), "consumer", &[r], &[]),
    ];
    let resources = vec![mock_resource_texture(r, "tex", 64, 64)];

    let compiled = FrameGraphCompiler::from_ir(passes, resources)
        .compile()
        .expect("chain compiles");
    let dot = GraphDotExporter::to_dot(&compiled);

    let edge_line = dot
        .lines()
        .find(|l| l.contains("P0 -> P1"))
        .expect("edge P0 -> P1 must exist");

    // Edge line has two arrows: one for the graph edge (P0 -> P1) and
    // one for the state transition inside the label (e.g., "ColorAttachment -> ShaderRead").
    let arrow_count = edge_line.matches("->").count();
    assert!(
        arrow_count >= 2,
        "edge line must have at least 2 arrows (direction + state transition): got {}",
        arrow_count,
    );
}

// =============================================================================
// SECTION 6 -- DOT escaping of special characters
// =============================================================================

/// Pass names containing double quotes are DOT-escaped.
#[test]
fn pass_name_with_quotes_escaped() {
    let r = ResourceHandle(1);
    let passes = vec![mock_pass_graphics(PassIndex(0), r#"he said "hello""#, &[r])];
    let resources = vec![mock_resource_texture(r, "tex", 64, 64)];

    let compiled = FrameGraphCompiler::from_ir(passes, resources)
        .compile()
        .expect("pass with quotes compiles");
    let dot = GraphDotExporter::to_dot(&compiled);

    // The node line must escape the double quotes in the label.
    // In the DOT output, each " becomes \".
    assert!(
        dot.contains(r#"he said \"hello\""#),
        "double quotes in pass name must be backslash-escaped in DOT output"
    );
}

/// Pass names containing backslashes are DOT-escaped (doubled).
#[test]
fn pass_name_with_backslash_escaped() {
    let r = ResourceHandle(1);
    let passes = vec![mock_pass_graphics(PassIndex(0), r"path\name", &[r])];
    let resources = vec![mock_resource_texture(r, "tex", 64, 64)];

    let compiled = FrameGraphCompiler::from_ir(passes, resources)
        .compile()
        .expect("pass with backslash compiles");
    let dot = GraphDotExporter::to_dot(&compiled);

    // In the DOT output, the single backslash must be escaped to \\.
    // In the Rust string, we check for \\ (two backslashes) in the DOT output.
    assert!(
        dot.contains(r"path\\name"),
        "backslashes in pass name must be doubled in DOT output"
    );
}

// =============================================================================
// SECTION 7 -- Mixed pass types in one graph
// =============================================================================

/// A graph with Graphics, Compute, and Copy passes emits correctly typed nodes.
///
/// Directly constructed (via CompiledFrameGraph) to avoid compiler dead-pass
/// elimination: the copy pass writes to an unconsumed output, which would
/// cause the compiler to remove it and the compute pass it depends on.
#[test]
fn mixed_graphics_compute_copy_nodes() {
    let r_tex = ResourceHandle(1);
    let r_buf = ResourceHandle(2);
    let r_out = ResourceHandle(3);

    let mut copy_pass = IrPass::copy(PassIndex(2), "output_copy");
    copy_pass.access_set.reads.push(r_buf);
    copy_pass.access_set.writes.push(r_out);

    let p0 = mock_pass_graphics(PassIndex(0), "scene_render", &[r_tex]);
    let p1 = mock_pass_compute(PassIndex(1), "post_process", &[r_tex], &[r_buf]);

    let compiled = CompiledFrameGraph {
        passes: vec![p0, p1, copy_pass],
        resources: vec![
            mock_resource_texture(r_tex, "hdr_color", 1920, 1080),
            mock_resource_buffer(r_buf, "compute_out", 4096),
            mock_resource_buffer(r_out, "final_out", 8192),
        ],
        edges: vec![],
        order: vec![PassIndex(0), PassIndex(1), PassIndex(2)],
        depths: HashMap::new(),
        barriers: vec![
            (PassIndex(0), PassIndex(1), r_tex, ResourceState::ColorAttachment, ResourceState::ShaderRead),
            (PassIndex(1), PassIndex(2), r_buf, ResourceState::ShaderReadWrite, ResourceState::ShaderRead),
        ],
        async_passes: vec![],
        eliminated_passes: vec![],
        cull_stats: CullStats::default(),
        parallel_regions: vec![],
        compilation_time_us: 0,
    };

    let dot = GraphDotExporter::to_dot(&compiled);

    // All three nodes with correct type annotations.
    assert!(
        dot.contains(r"(Graphics)"),
        "graphics pass type in mixed graph"
    );
    assert!(
        dot.contains(r"(Compute)"),
        "compute pass type in mixed graph"
    );
    assert!(
        dot.contains(r"(Copy)"),
        "copy pass type in mixed graph"
    );

    assert_eq!(count_nodes(&dot), 3, "three nodes for three mixed-type passes");

    // Edges exist between dependent passes.
    assert!(
        dot.contains("P0 -> P1"),
        "edge P0 -> P1 for RAW dependency (r_tex)"
    );
    assert!(
        dot.contains("P1 -> P2"),
        "edge P1 -> P2 for RAW dependency (r_buf)"
    );
}

// =============================================================================
// SECTION 8 -- Structural integrity of non-empty DOT output
// =============================================================================

/// Any non-empty graph must have balanced braces.
#[test]
fn non_empty_graph_balanced_braces() {
    let r = ResourceHandle(1);
    let passes = vec![
        mock_pass_graphics(PassIndex(0), "gbuffer", &[r]),
        mock_pass_compute(PassIndex(1), "lighting", &[r], &[]),
    ];
    let resources = vec![mock_resource_texture(r, "albedo", 800, 600)];

    let compiled = FrameGraphCompiler::from_ir(passes, resources)
        .compile()
        .expect("compiles");
    let dot = GraphDotExporter::to_dot(&compiled);

    let open_braces = dot.matches('{').count();
    let close_braces = dot.matches('}').count();
    assert_eq!(
        open_braces, close_braces,
        "balanced braces: {} open, {} close",
        open_braces, close_braces,
    );
}

/// Every node line follows the pattern P<index> [label="..."]; -- semicolon
/// terminated with square-bracket attributes.
#[test]
fn all_node_lines_have_correct_syntax() {
    let r0 = ResourceHandle(1);
    let r1 = ResourceHandle(2);
    let passes = vec![
        mock_pass_graphics(PassIndex(0), "gbuffer", &[r0]),
        mock_pass_compute(PassIndex(1), "lighting", &[r0], &[r1]),
        mock_pass_compute(PassIndex(2), "final", &[r1], &[]),
    ];
    let resources = vec![
        mock_resource_texture(r0, "color", 800, 600),
        mock_resource_buffer(r1, "data", 4096),
    ];

    let compiled = FrameGraphCompiler::from_ir(passes, resources)
        .compile()
        .expect("compiles");
    let dot = GraphDotExporter::to_dot(&compiled);

    for line in dot.lines() {
        let trimmed = line.trim();
        // Identify node lines: start with P, contain [label=, no ->
        if trimmed.starts_with('P')
            && trimmed.contains("[label=")
            && !trimmed.contains("->")
        {
            // Every node line must end with "];"
            assert!(
                trimmed.ends_with("];"),
                "node line must end with '];': got {:?}",
                trimmed,
            );

            // Every node line must start with P<digits>.
            assert!(
                trimmed.starts_with('P'),
                "node line must start with 'P': got {:?}",
                trimmed,
            );
            let after_p = trimmed.trim_start_matches('P');
            let digits: String = after_p.chars().take_while(|c| c.is_ascii_digit()).collect();
            assert!(
                !digits.is_empty(),
                "node line must have numeric index after P: got {:?}",
                trimmed,
            );
        }
    }
}

/// Every edge line follows the pattern P<from> -> P<to> [label="..."];.
#[test]
fn all_edge_lines_have_correct_syntax() {
    let r0 = ResourceHandle(1);
    let r1 = ResourceHandle(2);
    let passes = vec![
        mock_pass_graphics(PassIndex(0), "gbuffer", &[r0]),
        mock_pass_compute(PassIndex(1), "lighting", &[r0], &[r1]),
        mock_pass_compute(PassIndex(2), "final", &[r1], &[]),
    ];
    let resources = vec![
        mock_resource_texture(r0, "color", 800, 600),
        mock_resource_buffer(r1, "data", 4096),
    ];

    let compiled = FrameGraphCompiler::from_ir(passes, resources)
        .compile()
        .expect("compiles");
    let dot = GraphDotExporter::to_dot(&compiled);

    for line in dot.lines() {
        let trimmed = line.trim();
        if trimmed.contains("->") {
            // Every edge line must end with ";"
            assert!(
                trimmed.ends_with(';'),
                "edge line must end with ';': got {:?}",
                trimmed,
            );

            // Every edge line must have a label attribute.
            assert!(
                trimmed.contains("[label="),
                "edge line must have [label=...]: got {:?}",
                trimmed,
            );

            // Edge must reference P<from> -> P<to>.
            assert!(
                trimmed.starts_with('P'),
                "edge line must start with P: got {:?}",
                trimmed,
            );
        }
    }
}

// =============================================================================
// SECTION 9 -- Direct construction edge cases
// =============================================================================

/// Directly constructed graph with resource referenced in barrier but missing
/// from resources list: the handle number appears without a name prefix.
#[test]
fn barrier_with_missing_resource_name_shows_handle_only() {
    let p0 = mock_pass_graphics(PassIndex(0), "pass_a", &[ResourceHandle(1)]);
    let p1 = mock_pass_compute(PassIndex(1), "pass_b", &[ResourceHandle(1)], &[]);

    // Resource R1 is referenced in barriers but absent from resources list.
    let compiled = CompiledFrameGraph {
        passes: vec![p0, p1],
        resources: vec![], // no named resource for handle 1
        edges: vec![],
        order: vec![PassIndex(0), PassIndex(1)],
        depths: HashMap::new(),
        barriers: vec![(
            PassIndex(0),
            PassIndex(1),
            ResourceHandle(1),
            ResourceState::ColorAttachment,
            ResourceState::ShaderRead,
        )],
        async_passes: vec![],
        eliminated_passes: vec![],
        cull_stats: CullStats::default(),
        parallel_regions: vec![],
        compilation_time_us: 0,
    };

    let dot = GraphDotExporter::to_dot(&compiled);

    // Edge exists.
    assert!(dot.contains("P0 -> P1"), "edge must be present");

    // The handle number must appear in the label (no name to display).
    let edge_line = dot
        .lines()
        .find(|l| l.contains("P0 -> P1"))
        .expect("edge line exists");
    assert!(
        edge_line.contains("(1)"),
        "edge label must contain '(1)' for unnamed resource"
    );
}
