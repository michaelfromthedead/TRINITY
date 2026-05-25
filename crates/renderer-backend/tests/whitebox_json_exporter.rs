//! Whitebox tests for [`CompiledFrameGraph::emit_bridge_json`].
//!
//! DEV implemented `emit_bridge_json()` on `CompiledFrameGraph` to produce a
//! structured JSON value wrapping the pass graph, resource table, compiler
//! barriers, and cull statistics.  These tests verify all expected top-level
//! keys are present, the `cull_stats` sub-object contains every expected
//! field, and the overall structure is correct for both empty and populated
//! graphs.
//!
//! IMPORTANT: MockPassNode::build() assigns PassIndex(0) to every pass, so
//! multi-pass tests MUST assign unique indices before calling compile().

use renderer_backend::frame_graph::mocks::{MockPassNode, MockResourceDesc, reset_mock_handles};
use renderer_backend::frame_graph::{CompiledFrameGraph, IrPass, PassIndex};
use serde_json::Value;
use std::collections::HashSet;

/// Assign unique PassIndex(0..N) to a Vec of IrPass so the DAG builder
/// can distinguish producers from consumers.
fn unique_indices(passes: &mut Vec<IrPass>) {
    for (i, pass) in passes.iter_mut().enumerate() {
        pass.index = PassIndex(i);
    }
}

/// Assert that a serde_json::Value is a JSON object (dictionary).
fn assert_is_object(value: &Value, path: &str) {
    assert!(value.is_object(), "expected `{}` to be a JSON object; got {}", path, value);
}

/// Assert that a serde_json::Value is a JSON array.
fn assert_is_array(value: &Value, path: &str) {
    assert!(value.is_array(), "expected `{}` to be a JSON array; got {}", path, value);
}

// ===========================================================================
// 1.  All expected top-level keys exist for a populated graph
// ===========================================================================

#[test]
fn bridge_json_contains_all_expected_top_level_keys() {
    reset_mock_handles();
    let r0 = MockResourceDesc::texture_2d("framebuf", 256, 256);
    let mut passes = vec![
        MockPassNode::graphics("tonemap")
            .color_attachment(r0.handle())
            .build(),
    ];
    unique_indices(&mut passes);
    let resources = vec![r0.build()];
    let graph = CompiledFrameGraph::compile(passes, resources)
        .expect("graph must compile");
    let json = graph.emit_bridge_json();

    assert_is_object(&json, "root");

    let obj = json.as_object().unwrap();
    let expected_keys: HashSet<&str> = [
        "passes", "resources", "barriers", "async_passes",
        "sync_points", "parallel_regions", "depths", "cull_stats", "validation",
    ].into_iter().collect();

    assert!(
        obj.keys().all(|k| expected_keys.contains(k.as_str())),
        "unexpected key(s) in top-level JSON; keys: {:?}",
        obj.keys().collect::<Vec<_>>(),
    );
    assert_eq!(
        obj.len(),
        expected_keys.len(),
        "emit_bridge_json must return exactly {} top-level keys",
        expected_keys.len(),
    );
}

// ===========================================================================
// 2.  "passes" key is a JSON array
// ===========================================================================

#[test]
fn passes_key_is_a_json_array() {
    reset_mock_handles();
    let r = MockResourceDesc::texture_2d("target", 64, 64);
    let mut passes = vec![
        MockPassNode::graphics("gbuffer")
            .color_attachment(r.handle())
            .build(),
    ];
    unique_indices(&mut passes);
    let resources = vec![r.build()];
    let graph = CompiledFrameGraph::compile(passes, resources)
        .expect("graph must compile");
    let json = graph.emit_bridge_json();

    assert_is_array(&json["passes"], "\"passes\"");
}

// ===========================================================================
// 3.  "resources" key is a JSON array
// ===========================================================================

#[test]
fn resources_key_is_a_json_array() {
    reset_mock_handles();
    let r = MockResourceDesc::texture_2d("albedo", 64, 64);
    let mut passes = vec![
        MockPassNode::graphics("render")
            .color_attachment(r.handle())
            .build(),
    ];
    unique_indices(&mut passes);
    let resources = vec![r.build()];
    let graph = CompiledFrameGraph::compile(passes, resources)
        .expect("graph must compile");
    let json = graph.emit_bridge_json();

    assert_is_array(&json["resources"], "\"resources\"");
}

// ===========================================================================
// 4.  "depths" key is a JSON object
// ===========================================================================

#[test]
fn depths_key_is_a_json_object() {
    reset_mock_handles();
    let r = MockResourceDesc::texture_2d("target", 64, 64);
    let mut passes = vec![
        MockPassNode::graphics("render")
            .color_attachment(r.handle())
            .build(),
    ];
    unique_indices(&mut passes);
    let resources = vec![r.build()];
    let graph = CompiledFrameGraph::compile(passes, resources)
        .expect("graph must compile");
    let json = graph.emit_bridge_json();

    assert_is_object(&json["depths"], "\"depths\"");
}

// ===========================================================================
// 5.  "cull_stats" key is a JSON object with all expected fields
// ===========================================================================

#[test]
fn cull_stats_contains_all_expected_fields() {
    reset_mock_handles();
    let r = MockResourceDesc::texture_2d("target", 64, 64);
    let mut passes = vec![
        MockPassNode::graphics("render")
            .color_attachment(r.handle())
            .build(),
    ];
    unique_indices(&mut passes);
    let resources = vec![r.build()];
    let graph = CompiledFrameGraph::compile(passes, resources)
        .expect("graph must compile");
    let json = graph.emit_bridge_json();

    assert_is_object(&json["cull_stats"], "\"cull_stats\"");

    let stats = json["cull_stats"].as_object().unwrap();
    let expected_keys: &[&str] = &[
        "passes_total",
        "passes_eliminated",
        "resources_freed",
        "bytes_saved",
        "live_pass_count",
        "culled_pass_count",
        "estimated_gpu_time_saved_ms",
    ];

    for key in expected_keys {
        assert!(
            stats.contains_key(*key),
            "cull_stats.{} missing; keys: {:?}",
            key,
            stats.keys().collect::<Vec<_>>(),
        );
    }

    assert_eq!(stats.len(), expected_keys.len(), "cull_stats must contain exactly {} fields", expected_keys.len());
}

// ===========================================================================
// 6.  cull_stats values match compilation for a known graph
// ===========================================================================

#[test]
fn cull_stats_values_match_compilation() {
    reset_mock_handles();
    // One live graphics pass + one dead compute pass.
    // Expected: passes_total=2, live=1, culled=1.
    let r1 = MockResourceDesc::texture_2d("color", 64, 64);
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
    let graph = CompiledFrameGraph::compile(passes, resources)
        .expect("graph must compile");
    let json = graph.emit_bridge_json();

    let stats = &json["cull_stats"];
    assert_eq!(stats["passes_total"], 2, "passes_total mismatch");
    assert_eq!(stats["passes_eliminated"], 1, "passes_eliminated mismatch");
    assert_eq!(stats["live_pass_count"], 1, "live_pass_count mismatch");
    assert_eq!(stats["culled_pass_count"], 1, "culled_pass_count mismatch");
}

// ===========================================================================
// 7.  cull_stats values are zero for an empty graph
// ===========================================================================

#[test]
fn cull_stats_values_are_zero_for_empty_graph() {
    reset_mock_handles();
    let graph = CompiledFrameGraph::compile(vec![], vec![])
        .expect("empty graph must compile");
    let json = graph.emit_bridge_json();

    let stats = &json["cull_stats"];
    assert_eq!(stats["passes_total"], 0);
    assert_eq!(stats["passes_eliminated"], 0);
    assert_eq!(stats["resources_freed"], 0);
    assert_eq!(stats["bytes_saved"], 0);
    assert_eq!(stats["live_pass_count"], 0);
    assert_eq!(stats["culled_pass_count"], 0);
    assert_eq!(stats["estimated_gpu_time_saved_ms"], 0.0);
}

// ===========================================================================
// 8.  resource array length matches number of resources in graph
// ===========================================================================

#[test]
fn resources_array_length_matches_resource_count() {
    reset_mock_handles();
    let r0 = MockResourceDesc::texture_2d("albedo", 64, 64);
    let r1 = MockResourceDesc::buffer("light_grid", 4096);
    let r2 = MockResourceDesc::texture_2d("shadow_map", 128, 128);

    let mut passes = vec![
        MockPassNode::graphics("render")
            .color_attachment(r0.handle())
            .reads(&[r2.handle()])
            .writes(&[r1.handle()])
            .build(),
    ];
    unique_indices(&mut passes);
    let resources = vec![r0.build(), r1.build(), r2.build()];
    let graph = CompiledFrameGraph::compile(passes, resources)
        .expect("graph must compile");
    let json = graph.emit_bridge_json();

    let resources_arr = json["resources"].as_array().unwrap();
    assert_eq!(resources_arr.len(), 3, "expected 3 resources in array");
}

// ===========================================================================
// 9.  cull_stats object does not contain extra keys beyond the expected seven
// ===========================================================================

#[test]
fn cull_stats_has_no_extra_keys() {
    reset_mock_handles();
    let r = MockResourceDesc::texture_2d("target", 64, 64);
    let mut passes = vec![
        MockPassNode::graphics("render")
            .color_attachment(r.handle())
            .build(),
    ];
    unique_indices(&mut passes);
    let resources = vec![r.build()];
    let graph = CompiledFrameGraph::compile(passes, resources)
        .expect("graph must compile");
    let json = graph.emit_bridge_json();

    let stats = json["cull_stats"].as_object().unwrap();
    let allowed: HashSet<&str> = [
        "passes_total",
        "passes_eliminated",
        "resources_freed",
        "bytes_saved",
        "live_pass_count",
        "culled_pass_count",
        "estimated_gpu_time_saved_ms",
    ].into_iter().collect();

    for key in stats.keys() {
        assert!(
            allowed.contains(key.as_str()),
            "unexpected extra key in cull_stats: \"{}\"",
            key,
        );
    }
}

// ===========================================================================
// 10. Empty graph produces correct structure with all expected keys
// ===========================================================================

#[test]
fn empty_graph_still_has_all_expected_keys() {
    reset_mock_handles();
    let graph = CompiledFrameGraph::compile(vec![], vec![])
        .expect("empty graph must compile");
    let json = graph.emit_bridge_json();

    assert_is_object(&json, "root");
    let obj = json.as_object().unwrap();
    assert!(obj.contains_key("passes"), "\"passes\" key missing for empty graph");
    assert!(obj.contains_key("resources"), "\"resources\" key missing for empty graph");
    assert!(obj.contains_key("barriers"), "\"barriers\" key missing for empty graph");
    assert!(obj.contains_key("cull_stats"), "\"cull_stats\" key missing for empty graph");
    assert!(obj.contains_key("depths"), "\"depths\" key missing for empty graph");
    assert!(obj.contains_key("validation"), "\"validation\" key missing for empty graph");
}

// ===========================================================================
// 11. passes array has correct number of entries
// ===========================================================================

#[test]
fn passes_array_has_correct_count() {
    reset_mock_handles();
    let r0 = MockResourceDesc::texture_2d("rt", 64, 64);
    let mut passes = vec![
        MockPassNode::graphics("pass_0")
            .color_attachment(r0.handle())
            .build(),
    ];
    unique_indices(&mut passes);
    let resources = vec![r0.build()];
    let graph = CompiledFrameGraph::compile(passes, resources)
        .expect("graph must compile");
    let json = graph.emit_bridge_json();

    let passes_arr = &json["passes"];
    assert_is_array(passes_arr, "\"passes\"");
    assert_eq!(passes_arr.as_array().unwrap().len(), 1);
}
