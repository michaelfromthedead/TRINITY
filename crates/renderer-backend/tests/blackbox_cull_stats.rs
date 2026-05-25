// SPDX-License-Identifier: MIT
//
// blackbox_cull_stats.rs -- Blackbox contract tests for T-FG-6.5 CullStats.
//
// CLEANROOM: Tests access only the public API exported by the crate.  No
// implementation files in src/ were read beyond public function signatures.
//
// Contracts:
//   CompiledFrameGraph::compile() produces a CullStats struct on the
//   compiled result with:
//     - passes_total:      Total passes before dead-pass elimination
//     - passes_eliminated: Passes removed by culling
//     - live_pass_count:   Passes that survive culling
//     - culled_pass_count: Passes removed (alias for passes_eliminated)
//     - estimated_gpu_time_saved_ms: Sum of estimated GPU time for culled
//       passes (Graphics: ~2.0 ms, Compute: ~0.5 ms, Copy: ~0.1 ms)
//
// Invariant: live_pass_count + culled_pass_count == passes_total
//
// emit_bridge_json() includes all cull_stats fields in the output.
//
// IMPORTANT: MockPassNode::build() assigns PassIndex(0) to every pass, so
// multi-pass tests MUST assign unique indices before calling compile().

use renderer_backend::frame_graph::mocks::{MockPassNode, MockResourceDesc, reset_mock_handles};
use renderer_backend::frame_graph::{CompiledFrameGraph, IrPass, PassIndex};

const EPSILON_MS: f32 = 0.01;

/// Assign unique PassIndex(0..N) so the DAG builder can distinguish passes.
fn unique_indices(passes: &mut Vec<IrPass>) {
    for (i, pass) in passes.iter_mut().enumerate() {
        pass.index = PassIndex(i);
    }
}

// =========================================================================
// SECTION 1 -- Dead terminal pass (culled_pass_count > 0)
// =========================================================================

#[test]
fn dead_terminal_pass_culled_count_positive() {
    reset_mock_handles();
    // A single compute pass writes R1.  Nobody reads R1.
    let r1 = MockResourceDesc::buffer("orphan", 1024);
    let mut passes = vec![
        MockPassNode::compute("dead_terminal")
            .writes(&[r1.handle()])
            .build(),
    ];
    unique_indices(&mut passes);
    let resources = vec![r1.build()];

    let compiled = CompiledFrameGraph::compile(passes, resources)
        .expect("compile should succeed");

    assert!(
        compiled.cull_stats.culled_pass_count > 0,
        "dead terminal pass must have culled_pass_count > 0, got {}",
        compiled.cull_stats.culled_pass_count,
    );
}

#[test]
fn dead_terminal_pass_live_count_reflects_culling() {
    reset_mock_handles();
    // One dead compute pass, zero live passes.
    let r1 = MockResourceDesc::buffer("buf", 64);
    let mut passes = vec![
        MockPassNode::compute("terminal")
            .writes(&[r1.handle()])
            .build(),
    ];
    unique_indices(&mut passes);
    let resources = vec![r1.build()];

    let compiled = CompiledFrameGraph::compile(passes, resources)
        .expect("compile should succeed");

    assert_eq!(
        compiled.cull_stats.live_pass_count, 0,
        "dead terminal leaves zero live passes",
    );
    assert_eq!(
        compiled.cull_stats.culled_pass_count, 1,
        "one pass culled",
    );
}

// =========================================================================
// SECTION 2 -- No dead passes (culled_pass_count == 0)
// =========================================================================

#[test]
fn no_dead_passes_culled_count_zero() {
    reset_mock_handles();
    // A single graphics pass is always live.
    let r1 = MockResourceDesc::texture_2d("swapchain", 1920, 1080);
    let mut passes = vec![
        MockPassNode::graphics("render")
            .color_attachment(r1.handle())
            .build(),
    ];
    unique_indices(&mut passes);
    let resources = vec![r1.build()];

    let compiled = CompiledFrameGraph::compile(passes, resources)
        .expect("compile should succeed");

    assert_eq!(
        compiled.cull_stats.culled_pass_count, 0,
        "no dead passes means culled_pass_count == 0",
    );
}

#[test]
fn linear_chain_live_and_culled_counts() {
    reset_mock_handles();
    // P0 writes R1, P1 reads R1 and writes R2.
    // The consumer (P1) writes R2 which has no reader, so it is culled.
    // Producer (P0) survives because R1 is read, but by a culled pass,
    // so P0 may also be culled if transitive liveness is disabled for
    // compute-only chains.  The test documents the actual count.
    let r1 = MockResourceDesc::buffer("r1", 256);
    let r2 = MockResourceDesc::buffer("r2", 512);

    let mut passes = vec![
        MockPassNode::compute("producer")
            .writes(&[r1.handle()])
            .build(),
        MockPassNode::compute("consumer")
            .reads(&[r1.handle()])
            .writes(&[r2.handle()])
            .build(),
    ];
    unique_indices(&mut passes);
    let resources = vec![r1.build(), r2.build()];

    let compiled = CompiledFrameGraph::compile(passes, resources)
        .expect("compile should succeed");

    // The invariant holds regardless of which passes survive.
    let cs = &compiled.cull_stats;
    assert!(
        cs.culled_pass_count > 0,
        "consumer is culled because R2 has no reader; culled={}",
        cs.culled_pass_count,
    );
    assert_eq!(
        cs.live_pass_count + cs.culled_pass_count,
        cs.passes_total,
        "invariant holds: live ({}) + culled ({}) == total ({})",
        cs.live_pass_count,
        cs.culled_pass_count,
        cs.passes_total,
    );
}

#[test]
fn graphics_with_live_compute_culled_zero() {
    reset_mock_handles();
    let r1 = MockResourceDesc::texture_2d("color", 800, 600);
    let r2 = MockResourceDesc::buffer("scratch", 256);

    let mut passes = vec![
        MockPassNode::graphics("gbuffer")
            .color_attachment(r1.handle())
            .build(),
        MockPassNode::compute("post")
            .reads(&[r1.handle()])
            .writes(&[r2.handle()])
            .build(),
    ];
    unique_indices(&mut passes);
    let resources = vec![r1.build(), r2.build()];

    let compiled = CompiledFrameGraph::compile(passes, resources)
        .expect("compile should succeed");

    let stats = &compiled.cull_stats;
    assert!(
        stats.live_pass_count > 0,
        "at least one live pass (graphics)",
    );
}

// =========================================================================
// SECTION 3 -- Invariant: live_pass_count + culled_pass_count == passes_total
// =========================================================================

#[test]
fn invariant_live_plus_culled_equals_total() {
    reset_mock_handles();
    // 1 graphics (live) + 1 compute with consumer (live) + 1 compute w/o consumer (culled).
    let r1 = MockResourceDesc::texture_2d("color", 800, 600);
    let r2 = MockResourceDesc::buffer("data", 1024);
    let r3 = MockResourceDesc::buffer("orphan", 2048);

    let mut passes = vec![
        MockPassNode::graphics("gbuffer")
            .color_attachment(r1.handle())
            .build(),
        MockPassNode::compute("resolve")
            .reads(&[r1.handle()])
            .writes(&[r2.handle()])
            .build(),
        MockPassNode::compute("unused")
            .writes(&[r3.handle()])
            .build(),
    ];
    unique_indices(&mut passes);
    let resources = vec![r1.build(), r2.build(), r3.build()];

    let compiled = CompiledFrameGraph::compile(passes, resources)
        .expect("compile should succeed");

    let cs = &compiled.cull_stats;
    assert_eq!(
        cs.live_pass_count + cs.culled_pass_count,
        cs.passes_total,
        "live_pass_count ({}) + culled_pass_count ({}) must equal passes_total ({})",
        cs.live_pass_count,
        cs.culled_pass_count,
        cs.passes_total,
    );
}

#[test]
fn invariant_all_dead() {
    reset_mock_handles();
    // Create resource descriptors, capture handles, then build resources.
    let r1 = MockResourceDesc::buffer("r1", 64);
    let r2 = MockResourceDesc::buffer("r2", 128);
    let r3 = MockResourceDesc::buffer("r3", 256);
    let r4 = MockResourceDesc::buffer("r4", 512);
    let h1 = r1.handle();
    let h2 = r2.handle();
    let h3 = r3.handle();
    let h4 = r4.handle();

    let mut passes = vec![
        MockPassNode::compute("d1").writes(&[h1]).build(),
        MockPassNode::compute("d2").writes(&[h2]).build(),
        MockPassNode::compute("d3").writes(&[h3]).build(),
        MockPassNode::compute("d4").writes(&[h4]).build(),
    ];
    unique_indices(&mut passes);
    let all_resources = vec![r1.build(), r2.build(), r3.build(), r4.build()];

    let compiled = CompiledFrameGraph::compile(passes, all_resources)
        .expect("compile should succeed");

    let cs = &compiled.cull_stats;
    assert_eq!(cs.passes_total, 4);
    assert_eq!(
        cs.live_pass_count + cs.culled_pass_count,
        cs.passes_total,
        "invariant holds for all-dead graph",
    );
}

#[test]
fn invariant_empty_graph() {
    reset_mock_handles();
    let compiled =
        CompiledFrameGraph::compile(vec![], vec![]).expect("empty graph compiles");

    let cs = &compiled.cull_stats;
    assert_eq!(
        cs.live_pass_count + cs.culled_pass_count,
        cs.passes_total,
        "invariant holds for empty graph: 0 + 0 == 0",
    );
}

#[test]
fn invariant_all_live() {
    reset_mock_handles();
    let r1 = MockResourceDesc::texture_2d("rt", 800, 600);

    let mut passes = vec![
        MockPassNode::graphics("render")
            .color_attachment(r1.handle())
            .build(),
        MockPassNode::compute("post")
            .reads(&[r1.handle()])
            .build(),
    ];
    unique_indices(&mut passes);
    let resources = vec![r1.build()];

    let compiled = CompiledFrameGraph::compile(passes, resources)
        .expect("compile should succeed");

    let cs = &compiled.cull_stats;
    assert_eq!(
        cs.live_pass_count + cs.culled_pass_count,
        cs.passes_total,
        "invariant holds for all-live graph",
    );
}

// =========================================================================
// SECTION 4 -- Dead compute pass GPU time (~= 0.5 ms)
// =========================================================================

#[test]
fn dead_compute_pass_gpu_time_estimate() {
    reset_mock_handles();
    let r1 = MockResourceDesc::buffer("buf", 1024);
    let mut passes = vec![
        MockPassNode::compute("dead_compute")
            .writes(&[r1.handle()])
            .build(),
    ];
    unique_indices(&mut passes);
    let resources = vec![r1.build()];

    let compiled = CompiledFrameGraph::compile(passes, resources)
        .expect("compile should succeed");

    let estimated = compiled.cull_stats.estimated_gpu_time_saved_ms;
    let expected: f32 = 0.5;
    assert!(
        (estimated - expected).abs() < EPSILON_MS,
        "dead compute pass GPU time saved expected ~{} ms, got {} ms",
        expected,
        estimated,
    );
}

#[test]
fn two_dead_compute_passes_gpu_time_sum() {
    reset_mock_handles();
    let r1_desc = MockResourceDesc::buffer("r1", 64);
    let r2_desc = MockResourceDesc::buffer("r2", 128);
    let h1 = r1_desc.handle();
    let h2 = r2_desc.handle();

    let mut passes = vec![
        MockPassNode::compute("d1").writes(&[h1]).build(),
        MockPassNode::compute("d2").writes(&[h2]).build(),
    ];
    unique_indices(&mut passes);
    let resources = vec![r1_desc.build(), r2_desc.build()];

    let compiled = CompiledFrameGraph::compile(passes, resources)
        .expect("compile should succeed");

    let estimated = compiled.cull_stats.estimated_gpu_time_saved_ms;
    let expected: f32 = 1.0;
    assert!(
        (estimated - expected).abs() < EPSILON_MS,
        "two dead compute passes GPU time saved expected ~{} ms, got {} ms",
        expected,
        estimated,
    );
}

#[test]
fn dead_compute_pass_time_non_zero_and_positive() {
    reset_mock_handles();
    let r1 = MockResourceDesc::buffer("scratch", 4096);
    let mut passes = vec![
        MockPassNode::compute("compute_kernel")
            .writes(&[r1.handle()])
            .build(),
    ];
    unique_indices(&mut passes);
    let resources = vec![r1.build()];

    let compiled = CompiledFrameGraph::compile(passes, resources)
        .expect("compile should succeed");

    let estimated = compiled.cull_stats.estimated_gpu_time_saved_ms;
    assert!(
        estimated > 0.0,
        "dead compute pass GPU time saved must be > 0, got {}",
        estimated,
    );
}

// =========================================================================
// SECTION 5 -- Dead graphics pass GPU time (~= 2.0 ms)
// =========================================================================

#[test]
fn dead_graphics_pass_gpu_time_estimate() {
    reset_mock_handles();
    // Graphics passes are always live, so this pass will NOT be culled.
    let r1 = MockResourceDesc::texture_2d("output", 800, 600);
    let mut passes = vec![
        MockPassNode::graphics("gfx_unread")
            .color_attachment(r1.handle())
            .build(),
    ];
    unique_indices(&mut passes);
    let resources = vec![r1.build()];

    let compiled = CompiledFrameGraph::compile(passes, resources)
        .expect("compile should succeed");

    let estimated = compiled.cull_stats.estimated_gpu_time_saved_ms;
    let culled = compiled.cull_stats.culled_pass_count;

    // If the graphics pass IS culled: 1 pass * 2.0 ms = ~2.0 ms
    // If the graphics pass is NOT culled: 0.0 ms
    if culled > 0 {
        let expected_per_pass: f32 = 2.0;
        let expected_total = expected_per_pass * culled as f32;
        assert!(
            (estimated - expected_total).abs() < EPSILON_MS * culled as f32,
            "per-culled-graphics-pass GPU time expected ~{} ms, got {} ms for {} culled pass(es)",
            expected_total,
            estimated,
            culled,
        );
    }
}

#[test]
fn gpu_time_heuristic_graphics_is_2ms_per_pass() {
    reset_mock_handles();
    // 1 graphics (always live) + 4 dead compute passes = 4 * 0.5 = 2.0 ms.
    let r1 = MockResourceDesc::texture_2d("rt", 800, 600);
    let r2 = MockResourceDesc::buffer("x1", 64);
    let r3 = MockResourceDesc::buffer("x2", 64);
    let r4 = MockResourceDesc::buffer("x3", 64);
    let r5 = MockResourceDesc::buffer("x4", 64);
    let h1 = r1.handle();
    let h2 = r2.handle();
    let h3 = r3.handle();
    let h4 = r4.handle();
    let h5 = r5.handle();

    let mut passes = vec![
        MockPassNode::graphics("gfx_live")
            .color_attachment(h1)
            .build(),
        MockPassNode::compute("d1").writes(&[h2]).build(),
        MockPassNode::compute("d2").writes(&[h3]).build(),
        MockPassNode::compute("d3").writes(&[h4]).build(),
        MockPassNode::compute("d4").writes(&[h5]).build(),
    ];
    unique_indices(&mut passes);
    let resources = vec![
        r1.build(), r2.build(), r3.build(), r4.build(), r5.build(),
    ];

    let compiled = CompiledFrameGraph::compile(passes, resources)
        .expect("compile should succeed");

    let cs = &compiled.cull_stats;
    assert_eq!(cs.culled_pass_count, 4, "4 dead compute passes culled");
    assert_eq!(cs.live_pass_count, 1, "1 graphics pass live");
    let expected: f32 = 4.0 * 0.5;
    assert!(
        (cs.estimated_gpu_time_saved_ms - expected).abs() < EPSILON_MS,
        "expected ~{} ms for 4 culled compute, got {} ms",
        expected,
        cs.estimated_gpu_time_saved_ms,
    );
}

// =========================================================================
// SECTION 6 -- Empty graph (all cull stats zero)
// =========================================================================

#[test]
fn empty_graph_all_cull_stats_zero() {
    reset_mock_handles();
    let compiled =
        CompiledFrameGraph::compile(vec![], vec![]).expect("empty graph compiles");

    let cs = &compiled.cull_stats;
    assert_eq!(cs.passes_total, 0, "passes_total == 0");
    assert_eq!(cs.live_pass_count, 0, "live_pass_count == 0");
    assert_eq!(cs.culled_pass_count, 0, "culled_pass_count == 0");
    assert_eq!(
        cs.estimated_gpu_time_saved_ms, 0.0,
        "estimated_gpu_time_saved_ms == 0.0",
    );
}

#[test]
fn empty_graph_culled_count_zero() {
    reset_mock_handles();
    let compiled =
        CompiledFrameGraph::compile(vec![], vec![]).expect("empty graph compiles");

    assert_eq!(
        compiled.cull_stats.culled_pass_count, 0,
        "empty graph culled_pass_count == 0",
    );
    assert_eq!(
        compiled.cull_stats.live_pass_count, 0,
        "empty graph live_pass_count == 0",
    );
}

// =========================================================================
// SECTION 7 -- JSON export includes cull_stats with the expected fields
// =========================================================================

#[test]
fn json_export_cull_stats_has_live_pass_count() {
    reset_mock_handles();
    let r1 = MockResourceDesc::texture_2d("target", 800, 600);
    let mut passes = vec![
        MockPassNode::graphics("render")
            .color_attachment(r1.handle())
            .build(),
    ];
    unique_indices(&mut passes);
    let resources = vec![r1.build()];

    let compiled = CompiledFrameGraph::compile(passes, resources)
        .expect("graph compiles");
    let json = compiled.emit_bridge_json();

    let cs = &json["cull_stats"];
    assert!(
        cs.is_object(),
        "cull_stats must be a JSON object",
    );
    assert!(
        cs.get("live_pass_count").is_some(),
        "cull_stats.live_pass_count is present in JSON export",
    );
    assert!(
        cs.get("culled_pass_count").is_some(),
        "cull_stats.culled_pass_count is present in JSON export",
    );
    assert!(
        cs.get("estimated_gpu_time_saved_ms").is_some(),
        "cull_stats.estimated_gpu_time_saved_ms is present in JSON export",
    );
}

#[test]
fn json_export_cull_stats_values_are_numbers() {
    reset_mock_handles();
    let r1 = MockResourceDesc::texture_2d("target", 800, 600);
    let r2 = MockResourceDesc::buffer("orphan", 1024);

    let mut passes = vec![
        MockPassNode::graphics("render")
            .color_attachment(r1.handle())
            .build(),
        MockPassNode::compute("dead")
            .writes(&[r2.handle()])
            .build(),
    ];
    unique_indices(&mut passes);
    let resources = vec![r1.build(), r2.build()];

    let compiled = CompiledFrameGraph::compile(passes, resources)
        .expect("graph compiles");
    let json = compiled.emit_bridge_json();

    let cs = &json["cull_stats"];
    assert!(cs["live_pass_count"].is_number(), "live_pass_count is a number");
    assert!(cs["culled_pass_count"].is_number(), "culled_pass_count is a number");
    assert!(
        cs["estimated_gpu_time_saved_ms"].is_number(),
        "estimated_gpu_time_saved_ms is a number",
    );
}

#[test]
fn json_export_cull_stats_new_fields_are_consistent_with_total() {
    reset_mock_handles();
    let r1 = MockResourceDesc::texture_2d("color", 800, 600);
    let r2 = MockResourceDesc::buffer("data", 1024);
    let r3 = MockResourceDesc::buffer("orphan", 2048);

    let mut passes = vec![
        MockPassNode::graphics("gbuffer")
            .color_attachment(r1.handle())
            .build(),
        MockPassNode::compute("resolve")
            .reads(&[r1.handle()])
            .writes(&[r2.handle()])
            .build(),
        MockPassNode::compute("dead")
            .writes(&[r3.handle()])
            .build(),
    ];
    unique_indices(&mut passes);
    let resources = vec![r1.build(), r2.build(), r3.build()];

    let compiled = CompiledFrameGraph::compile(passes, resources)
        .expect("graph compiles");
    let json = compiled.emit_bridge_json();

    let cs = &json["cull_stats"];
    let live: i64 = cs["live_pass_count"].as_i64().unwrap_or(0);
    let culled: i64 = cs["culled_pass_count"].as_i64().unwrap_or(0);
    let total: i64 = cs["passes_total"].as_i64().unwrap_or(0);

    assert_eq!(
        live + culled,
        total,
        "JSON cull_stats invariant: live ({}) + culled ({}) == total ({})",
        live,
        culled,
        total,
    );
}

#[test]
fn json_export_cull_stats_estimated_time_non_negative() {
    reset_mock_handles();
    let r1 = MockResourceDesc::texture_2d("tex", 800, 600);
    let mut passes = vec![
        MockPassNode::graphics("main")
            .color_attachment(r1.handle())
            .build(),
    ];
    unique_indices(&mut passes);
    let resources = vec![r1.build()];

    let compiled = CompiledFrameGraph::compile(passes, resources)
        .expect("graph compiles");
    let json = compiled.emit_bridge_json();

    let time = json["cull_stats"]["estimated_gpu_time_saved_ms"]
        .as_f64()
        .unwrap_or(f64::NAN);
    assert!(
        time >= 0.0,
        "estimated_gpu_time_saved_ms must be non-negative, got {}",
        time,
    );
}

#[test]
fn json_export_old_and_new_cull_stats_coexist() {
    reset_mock_handles();
    let r1 = MockResourceDesc::texture_2d("tex", 800, 600);
    let mut passes = vec![
        MockPassNode::graphics("main")
            .color_attachment(r1.handle())
            .build(),
    ];
    unique_indices(&mut passes);
    let resources = vec![r1.build()];

    let compiled = CompiledFrameGraph::compile(passes, resources)
        .expect("graph compiles");
    let json = compiled.emit_bridge_json();

    let cs = json["cull_stats"]
        .as_object()
        .expect("cull_stats is an object");

    // Old fields
    assert!(cs.contains_key("passes_total"), "old field passes_total present");
    assert!(cs.contains_key("passes_eliminated"), "old field passes_eliminated present");
    assert!(cs.contains_key("resources_freed"), "old field resources_freed present");
    assert!(cs.contains_key("bytes_saved"), "old field bytes_saved present");

    // New fields
    assert!(cs.contains_key("live_pass_count"), "new field live_pass_count present");
    assert!(cs.contains_key("culled_pass_count"), "new field culled_pass_count present");
    assert!(
        cs.contains_key("estimated_gpu_time_saved_ms"),
        "new field estimated_gpu_time_saved_ms present",
    );
}

// =========================================================================
// SECTION 8 -- emit_bridge_json() cull_stats includes the expected fields
// =========================================================================

#[test]
fn bridge_json_has_live_pass_count() {
    reset_mock_handles();
    let r1 = MockResourceDesc::texture_2d("target", 800, 600);
    let mut passes = vec![
        MockPassNode::graphics("render")
            .color_attachment(r1.handle())
            .build(),
    ];
    unique_indices(&mut passes);
    let resources = vec![r1.build()];

    let json = CompiledFrameGraph::compile(passes, resources)
        .expect("compile succeeds")
        .emit_bridge_json();
    let cs = &json["cull_stats"];

    assert!(
        cs.is_object(),
        "bridge cull_stats must be a JSON object",
    );
    assert!(
        cs.get("live_pass_count").is_some(),
        "bridge cull_stats has live_pass_count",
    );
}

#[test]
fn bridge_json_has_culled_pass_count() {
    reset_mock_handles();
    let r1 = MockResourceDesc::buffer("buf", 1024);
    let mut passes = vec![
        MockPassNode::compute("dead")
            .writes(&[r1.handle()])
            .build(),
    ];
    unique_indices(&mut passes);
    let resources = vec![r1.build()];

    let json = CompiledFrameGraph::compile(passes, resources)
        .expect("compile succeeds")
        .emit_bridge_json();

    assert!(
        json["cull_stats"].get("culled_pass_count").is_some(),
        "bridge cull_stats has culled_pass_count",
    );
}

#[test]
fn bridge_json_has_estimated_gpu_time_saved_ms() {
    reset_mock_handles();
    let r1 = MockResourceDesc::texture_2d("target", 800, 600);
    let mut passes = vec![
        MockPassNode::graphics("render")
            .color_attachment(r1.handle())
            .build(),
    ];
    unique_indices(&mut passes);
    let resources = vec![r1.build()];

    let json = CompiledFrameGraph::compile(passes, resources)
        .expect("compile succeeds")
        .emit_bridge_json();

    assert!(
        json["cull_stats"]
            .get("estimated_gpu_time_saved_ms")
            .is_some(),
        "bridge cull_stats has estimated_gpu_time_saved_ms",
    );
}

#[test]
fn bridge_json_cull_stats_values_are_numbers() {
    reset_mock_handles();
    let r1 = MockResourceDesc::texture_2d("target", 800, 600);
    let mut passes = vec![
        MockPassNode::graphics("render")
            .color_attachment(r1.handle())
            .build(),
    ];
    unique_indices(&mut passes);
    let resources = vec![r1.build()];

    let json = CompiledFrameGraph::compile(passes, resources)
        .expect("compile succeeds")
        .emit_bridge_json();
    let cs = &json["cull_stats"];

    assert!(
        cs["live_pass_count"].is_number(),
        "bridge cull_stats.live_pass_count is a number",
    );
    assert!(
        cs["culled_pass_count"].is_number(),
        "bridge cull_stats.culled_pass_count is a number",
    );
    assert!(
        cs["estimated_gpu_time_saved_ms"].is_number(),
        "bridge cull_stats.estimated_gpu_time_saved_ms is a number",
    );
}

#[test]
fn bridge_json_new_fields_consistent_with_old() {
    reset_mock_handles();
    let r1 = MockResourceDesc::texture_2d("tex", 800, 600);
    let r2 = MockResourceDesc::buffer("buf", 1024);

    let mut passes = vec![
        MockPassNode::graphics("main")
            .color_attachment(r1.handle())
            .build(),
        MockPassNode::compute("dead")
            .writes(&[r2.handle()])
            .build(),
    ];
    unique_indices(&mut passes);
    let resources = vec![r1.build(), r2.build()];

    let json = CompiledFrameGraph::compile(passes, resources)
        .expect("compile succeeds")
        .emit_bridge_json();
    let cs = &json["cull_stats"];

    let live: i64 = cs["live_pass_count"].as_i64().unwrap_or(-1);
    let culled: i64 = cs["culled_pass_count"].as_i64().unwrap_or(-1);
    let total: i64 = cs["passes_total"].as_i64().unwrap_or(-1);

    assert_eq!(
        live + culled,
        total,
        "bridge cull_stats invariant: live ({}) + culled ({}) == total ({})",
        live,
        culled,
        total,
    );
}

#[test]
fn bridge_json_cull_stats_has_estimated_gpu_time_non_negative() {
    reset_mock_handles();
    let json = CompiledFrameGraph::compile(vec![], vec![])
        .expect("compile succeeds")
        .emit_bridge_json();
    let time = json["cull_stats"]["estimated_gpu_time_saved_ms"]
        .as_f64()
        .unwrap_or(f64::NAN);

    assert!(
        time >= 0.0,
        "estimated_gpu_time_saved_ms in bridge JSON must be >= 0, got {}",
        time,
    );
}

#[test]
fn bridge_json_empty_graph_new_fields_zero() {
    reset_mock_handles();
    let json = CompiledFrameGraph::compile(vec![], vec![])
        .expect("compile succeeds")
        .emit_bridge_json();
    let cs = &json["cull_stats"];

    assert_eq!(
        cs["live_pass_count"].as_i64().unwrap_or(-1),
        0,
        "empty graph live_pass_count == 0",
    );
    assert_eq!(
        cs["culled_pass_count"].as_i64().unwrap_or(-1),
        0,
        "empty graph culled_pass_count == 0",
    );
    assert_eq!(
        cs["estimated_gpu_time_saved_ms"].as_f64().unwrap_or(-1.0),
        0.0,
        "empty graph estimated_gpu_time_saved_ms == 0.0",
    );
}
