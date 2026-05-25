// SPDX-License-Identifier: MIT
//
// blackbox_ffi_roundtrip.rs -- Blackbox contract tests for T-FG-1.8 FFI round-trip.
//
// CLEANROOM: Tests access only the public API of the crate. No implementation
// files in src/ were read beyond the public function signatures.
//
// Contract under test:
//   CompiledFrameGraph::compile(passes, resources) produces a compiled graph.
//   emit_bridge_json() serialises the compiled graph to a JSON value whose
//   structure matches the Python bridge schema (passes, resources, barriers,
//   depths, cull_stats).
//
// Input graph built via MockPassNode + MockResourceDesc (crate::frame_graph::mocks).
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
// SECTION 1 -- Valid graph compiles and produces valid JSON output
// =============================================================================

#[test]
fn simple_valid_graph_compiles_and_emits_json() {
    reset_mock_handles();
    let r = MockResourceDesc::texture_2d("color_rt", 1920, 1080);
    let mut passes = vec![
        MockPassNode::graphics("Render")
            .color_attachment(r.handle())
            .build(),
    ];
    unique_indices(&mut passes);
    let resources = vec![r.build()];

    let compiled = CompiledFrameGraph::compile(passes, resources)
        .expect("simple graph must compile");
    let json = compiled.emit_bridge_json();

    assert!(
        json.get("passes").is_some(),
        "output must contain 'passes'"
    );
    assert!(
        json.get("resources").is_some(),
        "output must contain 'resources'"
    );
    assert!(
        json.get("barriers").is_some(),
        "output must contain 'barriers'"
    );
    assert!(
        json.get("depths").is_some(),
        "output must contain 'depths'"
    );
    assert!(
        json.get("cull_stats").is_some(),
        "output must contain 'cull_stats'"
    );
}

// =============================================================================
// SECTION 2 -- Pass names survive compile
// =============================================================================

#[test]
fn pass_names_survive_compile() {
    reset_mock_handles();
    let shadow = MockResourceDesc::texture_2d("shadow_atlas", 2048, 2048);
    let hdr = MockResourceDesc::texture_2d("hdr_output", 1920, 1080);

    let mut passes = vec![
        MockPassNode::graphics("ShadowMap")
            .color_attachment(shadow.handle())
            .build(),
        MockPassNode::compute("Lighting")
            .reads(&[shadow.handle()])
            .writes(&[hdr.handle()])
            .build(),
        MockPassNode::graphics("ToneMap")
            .color_attachment(hdr.handle())
            .build(),
    ];
    unique_indices(&mut passes);
    let resources = vec![shadow.build(), hdr.build()];

    let compiled = CompiledFrameGraph::compile(passes, resources)
        .expect("graph must compile");
    let pass_names: Vec<&str> = compiled.passes.iter().map(|p| p.name.as_str()).collect();

    assert!(
        pass_names.contains(&"ShadowMap"),
        "ShadowMap pass name must survive"
    );
    assert!(
        pass_names.contains(&"Lighting"),
        "Lighting pass name must survive"
    );
    assert!(
        pass_names.contains(&"ToneMap"),
        "ToneMap pass name must survive"
    );
}

// =============================================================================
// SECTION 3 -- Resource names survive compile
// =============================================================================

#[test]
fn resource_names_survive_compile() {
    reset_mock_handles();
    let color = MockResourceDesc::texture_2d("color_rt", 1920, 1080);
    let depth = MockResourceDesc::texture_2d("depth_tex", 1920, 1080);
    let buf = MockResourceDesc::buffer("compute_buf", 65536);

    let mut passes = vec![
        MockPassNode::graphics("Render")
            .color_attachment(color.handle())
            .depth_stencil(depth.handle())
            .build(),
    ];
    unique_indices(&mut passes);
    let resources = vec![color.build(), depth.build(), buf.build()];

    let compiled = CompiledFrameGraph::compile(passes, resources)
        .expect("graph must compile");
    let res_names: Vec<&str> = compiled.resources.iter().map(|r| r.name.as_str()).collect();

    assert!(res_names.contains(&"color_rt"));
    assert!(res_names.contains(&"depth_tex"));
    assert!(res_names.contains(&"compute_buf"));
}

// =============================================================================
// SECTION 4 -- Single pass compiles correctly
// =============================================================================

#[test]
fn single_graphics_pass_compiles_correctly() {
    reset_mock_handles();
    let r = MockResourceDesc::texture_2d("output", 1920, 1080);
    let mut passes = vec![
        MockPassNode::graphics("MainRender")
            .color_attachment(r.handle())
            .build(),
    ];
    unique_indices(&mut passes);
    let resources = vec![r.build()];

    let compiled = CompiledFrameGraph::compile(passes, resources)
        .expect("single graphics pass must compile");
    assert_eq!(compiled.passes.len(), 1, "single pass remains 1 pass");
    assert_eq!(compiled.passes[0].name, "MainRender");
    assert_eq!(compiled.passes[0].pass_type.to_string(), "Graphics");
}

// =============================================================================
// SECTION 5 -- Multiple passes preserves pass count
// =============================================================================

#[test]
fn multiple_passes_preserves_pass_count() {
    reset_mock_handles();
    let albedo = MockResourceDesc::texture_2d("albedo", 1920, 1080);
    let normal = MockResourceDesc::texture_2d("normal", 1920, 1080);
    let hdr = MockResourceDesc::texture_2d("hdr", 1920, 1080);
    let velocity = MockResourceDesc::texture_2d("velocity", 1920, 1080);
    let taa_out = MockResourceDesc::texture_2d("taa_output", 1920, 1080);
    let final_out = MockResourceDesc::texture_2d("final_output", 1920, 1080);

    let mut passes = vec![
        MockPassNode::graphics("GBuffer")
            .color_attachment(albedo.handle())
            .color_attachment(normal.handle())
            .build(),
        MockPassNode::compute("Lighting")
            .reads(&[albedo.handle(), normal.handle()])
            .writes(&[hdr.handle()])
            .build(),
        MockPassNode::compute("TAA")
            .reads(&[hdr.handle(), velocity.handle()])
            .writes(&[taa_out.handle()])
            .build(),
        MockPassNode::compute("PostFX")
            .reads(&[taa_out.handle()])
            .writes(&[final_out.handle()])
            .build(),
        MockPassNode::graphics("UIOverlay")
            .color_attachment(final_out.handle())
            .build(),
    ];
    unique_indices(&mut passes);
    let resources = vec![
        albedo.build(), normal.build(), hdr.build(),
        velocity.build(), taa_out.build(), final_out.build(),
    ];

    let compiled = CompiledFrameGraph::compile(passes, resources)
        .expect("5-pass graph must compile");
    assert_eq!(
        compiled.order.len(), 5,
        "all 5 passes must survive in execution order"
    );
}

// =============================================================================
// SECTION 6 -- Graphics pass with depth attachment compiles
// =============================================================================

#[test]
fn graphics_pass_with_depth_attachment_compiles() {
    reset_mock_handles();
    let albedo = MockResourceDesc::texture_2d("gbuffer_albedo", 1920, 1080);
    let depth = MockResourceDesc::texture_2d("depth_tex", 1920, 1080);

    let mut passes = vec![
        MockPassNode::graphics("DepthPrePass")
            .color_attachment(albedo.handle())
            .depth_stencil(depth.handle())
            .build(),
    ];
    unique_indices(&mut passes);
    let resources = vec![albedo.build(), depth.build()];

    let compiled = CompiledFrameGraph::compile(passes, resources)
        .expect("depth attachment pass must compile");
    let pass = &compiled.passes[0];
    assert_eq!(pass.name, "DepthPrePass");
    assert!(pass.depth_stencil.is_some(), "depth_stencil must be present");
}

// =============================================================================
// SECTION 7 -- Cull stats reports pass counts
// =============================================================================

#[test]
fn cull_stats_reports_pass_counts() {
    reset_mock_handles();
    let r = MockResourceDesc::texture_2d("rt", 256, 256);
    let mut passes = vec![
        MockPassNode::graphics("Render")
            .color_attachment(r.handle())
            .build(),
    ];
    unique_indices(&mut passes);
    let resources = vec![r.build()];

    let compiled = CompiledFrameGraph::compile(passes, resources)
        .expect("graph must compile");
    let cs = &compiled.cull_stats;
    assert_eq!(cs.passes_total, 1, "passes_total must be 1");
    assert_eq!(
        cs.passes_eliminated, 0,
        "passes_eliminated must be 0 for a single graphics pass"
    );
}
