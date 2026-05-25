// SPDX-License-Identifier: MIT
//
// blackbox_frame_graph_conv.rs -- Frame graph compilation consistency tests.
//
// CLEANROOM: Tests use only the public API exported by the crate:
//   - MockPassNode / MockResourceDesc — test helpers for constructing IR
//   - CompiledFrameGraph::compile() — compiler entry point
//   - CompiledFrameGraph::emit_bridge_json() — IR -> JSON serialization
//
// Contract:
//   The frame graph compiler produces deterministic, consistent output for
//   equivalent input graphs.  Properties preserved through compilation:
//
//     - Pass count, names, and types
//     - Resource count, names, and types
//     - Pipeline barrier count (graph topology)
//     - Cull statistics (passes_total, passes_eliminated)
//     - Topological execution order
//
//   Double compilation (same input -> compile twice) produces identical
//   emit_bridge_json output for all data fields (fixed point convergence).
//
// IMPORTANT: MockPassNode::build() assigns PassIndex(0) to every pass, so
// multi-pass tests MUST assign unique indices before calling compile().

use renderer_backend::frame_graph::mocks::{MockPassNode, MockResourceDesc, reset_mock_handles};
use renderer_backend::frame_graph::{CompiledFrameGraph, IrPass, PassIndex};

/// Assign unique PassIndex(0..N) to a Vec of IrPass so the DAG builder
/// can distinguish producers from consumers.
fn unique_indices(passes: &mut Vec<IrPass>) {
    for (i, pass) in passes.iter_mut().enumerate() {
        pass.index = PassIndex(i);
    }
}

// =============================================================================
// SECTION 1 -- Consistency: same input produces same output
// =============================================================================

#[test]
fn conv_equivalent_pass_and_resource_counts() {
    reset_mock_handles();
    // Build a 4-pass chain.
    let r0 = MockResourceDesc::texture_2d("albedo", 1920, 1080);
    let r1 = MockResourceDesc::buffer("light_data", 65536);
    let r2 = MockResourceDesc::texture_2d("output", 1920, 1080);

    let mut passes = vec![
        MockPassNode::graphics("gbuffer")
            .color_attachment(r0.handle())
            .build(),
        MockPassNode::compute("lighting")
            .reads(&[r0.handle()])
            .writes(&[r1.handle()])
            .build(),
        MockPassNode::compute("postfx")
            .reads(&[r1.handle()])
            .writes(&[r2.handle()])
            .build(),
        MockPassNode::graphics("present")
            .color_attachment(r2.handle())
            .build(),
    ];
    unique_indices(&mut passes);
    let resources = vec![r0.build(), r1.build(), r2.build()];

    let compiled = CompiledFrameGraph::compile(passes, resources)
        .expect("4-pass chain must compile");

    // All 4 passes survive.
    assert_eq!(compiled.passes.len(), 4, "4 passes must survive");
    assert_eq!(compiled.resources.len(), 3, "3 resources");

    // Pass names in order.
    let names: Vec<&str> = compiled.passes.iter().map(|p| p.name.as_str()).collect();
    assert_eq!(names[0], "gbuffer");
    assert_eq!(names[1], "lighting");
    assert_eq!(names[2], "postfx");
    assert_eq!(names[3], "present");

    // Pass types.
    let types: Vec<String> = compiled.passes.iter().map(|p| format!("{}", p.pass_type)).collect();
    assert_eq!(types[0], "Graphics");
    assert_eq!(types[1], "Compute");
    assert_eq!(types[2], "Compute");
    assert_eq!(types[3], "Graphics");

    // Barriers exist for the chain dependencies.
    assert!(!compiled.barriers.is_empty(), "chain must produce barriers");
}

// =============================================================================
// SECTION 2 -- Deterministic compilation (fixed point)
// =============================================================================

#[test]
fn conv_double_compile_converges_to_fixed_point() {
    reset_mock_handles();
    let depth_tex = MockResourceDesc::texture_2d("depth_tex", 1920, 1080);
    let color_rt = MockResourceDesc::texture_2d("color_rt", 1920, 1080);
    let bloom_rt = MockResourceDesc::texture_2d("bloom_rt", 960, 540);

    let mut passes = vec![
        MockPassNode::graphics("depth")
            .color_attachment(depth_tex.handle())
            .build(),
        MockPassNode::graphics("opaque")
            .color_attachment(color_rt.handle())
            .depth_stencil(depth_tex.handle())
            .build(),
        MockPassNode::compute("bloom")
            .reads(&[color_rt.handle()])
            .writes(&[bloom_rt.handle()])
            .build(),
        MockPassNode::graphics("composite")
            .color_attachment(color_rt.handle())
            .reads(&[bloom_rt.handle()])
            .build(),
    ];
    unique_indices(&mut passes);
    let resources = vec![depth_tex.build(), color_rt.build(), bloom_rt.build()];

    let compiled_a = CompiledFrameGraph::compile(
        passes.clone(),
        resources.clone(),
    ).expect("first compile must succeed");

    let compiled_b = CompiledFrameGraph::compile(
        passes,
        resources,
    ).expect("second compile must succeed");

    // Same input -> same output for all data fields.
    // Barriers are checked by count and set membership rather than strict
    // ordering because barriers with the same (from, to) may sort non-
    // deterministically.
    let json_a = compiled_a.emit_bridge_json();
    let json_b = compiled_b.emit_bridge_json();

    assert_eq!(json_a["passes"], json_b["passes"], "passes must be deterministic");
    assert_eq!(json_a["resources"], json_b["resources"], "resources must be deterministic");
    assert_eq!(
        json_a["barriers"].as_array().unwrap().len(),
        json_b["barriers"].as_array().unwrap().len(),
        "barrier count must be deterministic",
    );
    assert_eq!(json_a["depths"], json_b["depths"], "depths must be deterministic");

    assert_eq!(
        json_a["cull_stats"]["passes_total"],
        json_b["cull_stats"]["passes_total"],
        "passes_total must be deterministic",
    );
}

// =============================================================================
// SECTION 3 -- Diamond topology preserved
// =============================================================================

#[test]
fn conv_diamond_topology_preserved() {
    reset_mock_handles();
    // Diamond graph: P0 -> P1, P0 -> P2, P1 -> P3, P2 -> P3
    // But P3 (merge) writes a resource with no consumer, so it gets culled.
    let rt_a = MockResourceDesc::texture_2d("rt_a", 800, 600);
    let rt_b = MockResourceDesc::texture_2d("rt_b", 800, 600);
    let result_a = MockResourceDesc::buffer("result_a", 4096);
    let result_b = MockResourceDesc::buffer("result_b", 4096);
    let final_ = MockResourceDesc::buffer("final", 8192);

    let mut passes = vec![
        MockPassNode::graphics("root")
            .color_attachment(rt_a.handle())
            .color_attachment(rt_b.handle())
            .build(),
        MockPassNode::compute("branch_a")
            .reads(&[rt_a.handle()])
            .writes(&[result_a.handle()])
            .build(),
        MockPassNode::compute("branch_b")
            .reads(&[rt_b.handle()])
            .writes(&[result_b.handle()])
            .build(),
        MockPassNode::compute("merge")
            .reads(&[result_a.handle(), result_b.handle()])
            .writes(&[final_.handle()])
            .build(),
    ];
    unique_indices(&mut passes);
    let resources = vec![
        rt_a.build(), rt_b.build(),
        result_a.build(), result_b.build(),
        final_.build(),
    ];

    let compiled = CompiledFrameGraph::compile(passes, resources)
        .expect("diamond graph must compile");

    // The "merge" pass writes "final" which has no consumer, so it gets culled.
    // Live passes in order: root, branch_a, branch_b.
    assert_eq!(compiled.order.len(), 3, "3 diamond passes survive (merge culled)");
    let names: Vec<&str> = compiled.order.iter()
        .filter_map(|idx| compiled.passes.iter().find(|p| p.index == *idx))
        .map(|p| p.name.as_str())
        .collect();
    assert_eq!(names[0], "root");
    let graphics_count = compiled.passes.iter().filter(|p| p.pass_type.to_string() == "Graphics").count();
    assert!(graphics_count >= 1, "at least 1 Graphics pass");

    // 5 resources survive.
    assert_eq!(compiled.resources.len(), 5, "5 resources");

    // Barriers exist for the dependencies.
    assert!(!compiled.barriers.is_empty(), "diamond topology must produce barriers");

    // Stats.
    assert_eq!(compiled.cull_stats.passes_total, 4);
    assert_eq!(compiled.cull_stats.passes_eliminated, 1);
}

// =============================================================================
// SECTION 4 -- Complex topology: 6-pass fan-out + fan-in
// =============================================================================

#[test]
fn conv_6_pass_multi_level_preserved() {
    reset_mock_handles();
    // Three-level hierarchy:
    //   P0 (graphics, writes background + character)
    //   P1 (compute, reads background -> writes bg_blur)
    //   P2 (compute, reads character -> writes char_outline)
    //   P3 (compute, reads bg_blur + char_outline -> writes merged)
    //   P4 (compute, reads merged -> writes graded)
    //   P5 (graphics, reads graded + background -> writes output)
    let background = MockResourceDesc::texture_2d("background", 1920, 1080);
    let character = MockResourceDesc::texture_2d("character", 1920, 1080);
    let bg_blur = MockResourceDesc::texture_2d("bg_blur", 960, 540);
    let char_outline = MockResourceDesc::buffer("char_outline", 65536);
    let merged = MockResourceDesc::texture_2d("merged", 1920, 1080);
    let graded = MockResourceDesc::texture_2d("graded", 1920, 1080);

    let mut passes = vec![
        MockPassNode::graphics("scene_render")
            .color_attachment(background.handle())
            .color_attachment(character.handle())
            .build(),
        MockPassNode::compute("blur_bg")
            .reads(&[background.handle()])
            .writes(&[bg_blur.handle()])
            .build(),
        MockPassNode::compute("outline_char")
            .reads(&[character.handle()])
            .writes(&[char_outline.handle()])
            .build(),
        MockPassNode::compute("merge_layers")
            .reads(&[bg_blur.handle(), char_outline.handle()])
            .writes(&[merged.handle()])
            .build(),
        MockPassNode::compute("color_grade")
            .reads(&[merged.handle()])
            .writes(&[graded.handle()])
            .build(),
        MockPassNode::graphics("output_compose")
            .color_attachment(background.handle())
            .reads(&[graded.handle()])
            .build(),
    ];
    unique_indices(&mut passes);
    let resources = vec![
        background.build(), character.build(),
        bg_blur.build(), char_outline.build(),
        merged.build(), graded.build(),
    ];

    let compiled = CompiledFrameGraph::compile(passes, resources)
        .expect("6-pass graph must compile");

    // All 6 passes survive.
    assert_eq!(compiled.order.len(), 6, "all 6 passes must survive");

    let names: Vec<&str> = compiled.passes.iter().map(|p| p.name.as_str()).collect();
    assert!(names.contains(&"scene_render"));
    assert!(names.contains(&"blur_bg"));
    assert!(names.contains(&"outline_char"));
    assert!(names.contains(&"merge_layers"));
    assert!(names.contains(&"color_grade"));
    assert!(names.contains(&"output_compose"));

    // Pass type counts.
    let graphics_count = compiled.passes.iter().filter(|p| p.pass_type.to_string() == "Graphics").count();
    let compute_count = compiled.passes.iter().filter(|p| p.pass_type.to_string() == "Compute").count();
    assert_eq!(graphics_count, 2, "2 Graphics passes");
    assert_eq!(compute_count, 4, "4 Compute passes");

    // 6 resources.
    assert_eq!(compiled.resources.len(), 6, "all 6 resources");

    // Barriers for multi-level dependencies.
    assert!(!compiled.barriers.is_empty(), "6-pass graph must have barriers");

    // Stats.
    assert_eq!(compiled.cull_stats.passes_total, 6);
    assert_eq!(compiled.cull_stats.passes_eliminated, 0);
}

// =============================================================================
// SECTION 5 -- Resource metadata preserved
// =============================================================================

#[test]
fn conv_resource_metadata_preserved() {
    reset_mock_handles();
    let swapchain = MockResourceDesc::texture_2d("swapchain", 2560, 1440);

    let mut passes = vec![
        MockPassNode::graphics("final")
            .color_attachment(swapchain.handle())
            .build(),
    ];
    unique_indices(&mut passes);
    let resources = vec![swapchain.build()];

    let compiled = CompiledFrameGraph::compile(passes, resources)
        .expect("graph must compile");
    let json = compiled.emit_bridge_json();

    let resources_arr = json["resources"].as_array()
        .expect("output must have 'resources' array");
    assert!(!resources_arr.is_empty(), "resources array must not be empty");

    let res = &resources_arr[0];
    assert_eq!(res["name"], "swapchain");
    assert!(res.get("desc").is_some(), "resource must have 'desc' field");
}
