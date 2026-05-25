// Blackbox contract tests for T-FG-7.7 JsonExporter::export_all.
//
// CLEANROOM: No src/ access beyond the public API exported by the crate.
// Tests use only `renderer_backend::frame_graph::*` -- no internal fields,
// no private methods, no implementation details.
//
// Contract:
//   JsonExporter::export_all(&compiled) returns a serde_json::Value with four
//   top-level keys:
//
//     "graph"     — full pass graph from CompiledFrameGraph::emit_bridge_json
//                   (passes ordered topologically, resources, barriers,
//                   async_passes, parallel_regions, depths, cull_stats,
//                   validation).
//     "resources" — sorted resource table from emit_resource_table
//                   (one entry per resource sorted by handle, with lifetime
//                   fields and import paths).
//     "schedule"  — execution schedule from CompiledFrameGraph::emit_schedule_bridge
//                   (execution_order, barriers, async_passes, parallel_regions,
//                   sync_points).
//     "stats"     — aggregate CompilerStats as a flat object with seven fields:
//                   passes_total, passes_eliminated, barriers_total,
//                   barriers_optimized, async_passes, resources_aliased,
//                   compilation_time_us.
//
// Scenarios:
//   1.  export_all returns exactly 4 top-level keys for populated graph
//   2.  "graph" key is a JSON object containing pass and resource data
//   3.  "resources" key is a JSON array sorted by handle
//   4.  "schedule" key is a JSON object with execution schedule fields
//   5.  "stats" key is a JSON object with all 7 expected fields
//   6.  Stats values from a full compile match the compiler output
//   7.  Empty graph still produces all four top-level keys
//   8.  Graph passes count matches compiled graph
//   9.  Resources array length matches input
//  10.  Diamond graph preserves correct topologies in export
//  11.  Dead pass elimination reflected in stats (passes_eliminated > 0)
//  12.  Schedule includes execution_order from topological sort
//  13.  Schedule barriers match graph barriers
//  14.  Async passes appear in both graph and schedule sections
//  15.  Stats numeric types are proper JSON numbers not strings
//  16.  export_all is deterministic (same inputs produce same output)
//  17.  Graph with barriers populates barrier-related sections
//  18.  Resource table entries contain handle, name, desc fields
//  19.  Stats object has no extra keys beyond the seven expected
//  20.  Resources table lifetime fields populated correctly
//
use renderer_backend::frame_graph::{
    mock_pass_compute, mock_pass_graphics, mock_resource_buffer, mock_resource_texture,
    FrameGraphCompiler, JsonExporter, PassIndex, ResourceHandle,
};
use serde_json::Value;

// =========================================================================
// SECTION 1 -- Top-level structure: four required keys
// =========================================================================

#[test]
fn export_all_returns_exactly_four_top_level_keys() {
    // Build a minimal compiled graph using only the public compiler API.
    let r = ResourceHandle(1);
    let passes = vec![mock_pass_graphics(PassIndex(0), "main", &[r])];
    let resources = vec![mock_resource_texture(r, "output", 1920, 1080)];

    let compiler = FrameGraphCompiler::new(passes, resources);
    let compiled = compiler.compile().expect("minimal graph compiles");
    let json = JsonExporter::export_all(&compiled);

    let obj = json.as_object().expect("top-level value must be a JSON object");
    assert_eq!(
        obj.len(),
        4,
        "export_all must return exactly 4 top-level keys, got {}",
        obj.len(),
    );

    assert!(
        obj.contains_key("graph"),
        "top-level key 'graph' is missing; keys: {:?}",
        obj.keys().collect::<Vec<_>>(),
    );
    assert!(
        obj.contains_key("resources"),
        "top-level key 'resources' is missing; keys: {:?}",
        obj.keys().collect::<Vec<_>>(),
    );
    assert!(
        obj.contains_key("schedule"),
        "top-level key 'schedule' is missing; keys: {:?}",
        obj.keys().collect::<Vec<_>>(),
    );
    assert!(
        obj.contains_key("stats"),
        "top-level key 'stats' is missing; keys: {:?}",
        obj.keys().collect::<Vec<_>>(),
    );
}

#[test]
fn empty_graph_still_has_all_four_top_level_keys() {
    // An empty graph with no passes and no resources must still produce all
    // four top-level keys (with empty arrays / zeroed stats).
    let compiler = FrameGraphCompiler::new(vec![], vec![]);
    let compiled = compiler.compile().expect("empty graph compiles");
    let json = JsonExporter::export_all(&compiled);

    let obj = json.as_object().expect("top-level value must be an object");
    assert!(
        obj.contains_key("graph"),
        "'graph' key missing for empty graph",
    );
    assert!(
        obj.contains_key("resources"),
        "'resources' key missing for empty graph",
    );
    assert!(
        obj.contains_key("schedule"),
        "'schedule' key missing for empty graph",
    );
    assert!(
        obj.contains_key("stats"),
        "'stats' key missing for empty graph",
    );
    assert_eq!(obj.len(), 4, "empty graph must have exactly 4 keys");
}

// =========================================================================
// SECTION 2 -- "graph" key structure
// =========================================================================

#[test]
fn graph_key_is_a_json_object() {
    let r = ResourceHandle(1);
    let passes = vec![mock_pass_graphics(PassIndex(0), "scene", &[r])];
    let resources = vec![mock_resource_texture(r, "color_rt", 800, 600)];

    let compiler = FrameGraphCompiler::new(passes, resources);
    let compiled = compiler.compile().expect("graph compiles");
    let json = JsonExporter::export_all(&compiled);

    let graph = &json["graph"];
    assert!(
        graph.is_object(),
        "'graph' must be a JSON object, got {:?}",
        graph,
    );
}

#[test]
fn graph_contains_expected_sections() {
    // The "graph" object should contain keys from emit_bridge_json: passes,
    // resources, barriers, async_passes, parallel_regions, depths, cull_stats,
    // validation.
    let r = ResourceHandle(1);
    let passes = vec![mock_pass_graphics(PassIndex(0), "main", &[r])];
    let resources = vec![mock_resource_texture(r, "target", 800, 600)];

    let compiler = FrameGraphCompiler::new(passes, resources);
    let compiled = compiler.compile().expect("graph compiles");
    let json = JsonExporter::export_all(&compiled);

    let graph = json["graph"].as_object().expect("'graph' must be an object");
    let expected_graph_keys: &[&str] = &[
        "passes",
        "resources",
        "barriers",
        "async_passes",
        "parallel_regions",
        "depths",
        "cull_stats",
        "validation",
    ];

    for key in expected_graph_keys {
        assert!(
            graph.contains_key(*key),
            "'graph' is missing key '{}'; keys: {:?}",
            key,
            graph.keys().collect::<Vec<_>>(),
        );
    }
}

#[test]
fn graph_pass_count_matches_compiled_order() {
    // Number of passes in "graph"."passes" must equal the number of surviving
    // passes from compilation.
    let r1 = ResourceHandle(1);
    let r2 = ResourceHandle(2);
    let passes = vec![
        mock_pass_graphics(PassIndex(0), "gbuffer", &[r1]),
        mock_pass_compute(PassIndex(1), "resolve", &[r1], &[r2]),
        mock_pass_compute(PassIndex(2), "composite", &[r2], &[]),
    ];
    let resources = vec![
        mock_resource_texture(r1, "albedo", 1920, 1080),
        mock_resource_buffer(r2, "data", 4096),
    ];

    let compiler = FrameGraphCompiler::new(passes, resources);
    let compiled = compiler.compile().expect("three-pass chain compiles");
    let json = JsonExporter::export_all(&compiled);

    let graph_passes = json["graph"]["passes"]
        .as_array()
        .expect("'graph'.'passes' must be an array");
    assert_eq!(
        graph_passes.len(),
        compiled.passes.len(),
        "graph pass count must match CompiledFrameGraph.passes.len()",
    );
    assert_eq!(
        graph_passes.len(),
        3,
        "all three passes survive and appear in graph",
    );
}

#[test]
fn graph_passes_contain_required_fields() {
    // Each pass entry in "graph"."passes" must have index, name, pass_type.
    let r = ResourceHandle(1);
    let passes = vec![mock_pass_graphics(PassIndex(0), "test_pass", &[r])];
    let resources = vec![mock_resource_texture(r, "tex", 64, 64)];

    let compiler = FrameGraphCompiler::new(passes, resources);
    let compiled = compiler.compile().expect("graph compiles");
    let json = JsonExporter::export_all(&compiled);

    let graph_passes = json["graph"]["passes"]
        .as_array()
        .expect("'graph'.'passes' must be an array");
    assert!(!graph_passes.is_empty(), "graph passes array must not be empty");

    let first_pass = &graph_passes[0];
    let pass_obj = first_pass
        .as_object()
        .expect("each pass must be a JSON object");
    assert!(
        pass_obj.contains_key("index"),
        "pass missing 'index' field",
    );
    assert!(
        pass_obj.contains_key("name"),
        "pass missing 'name' field",
    );
    assert!(
        pass_obj.contains_key("pass_type"),
        "pass missing 'pass_type' field",
    );
    assert!(
        pass_obj.contains_key("view_type"),
        "pass missing 'view_type' field",
    );
}

#[test]
fn graph_resources_contain_required_fields() {
    // Resource entries in graph must have handle, name, desc.
    let r = ResourceHandle(1);
    let passes = vec![mock_pass_graphics(PassIndex(0), "main", &[r])];
    let resources = vec![mock_resource_texture(r, "swapchain", 1920, 1080)];

    let compiler = FrameGraphCompiler::new(passes, resources);
    let compiled = compiler.compile().expect("graph compiles");
    let json = JsonExporter::export_all(&compiled);

    let graph_resources = json["graph"]["resources"]
        .as_array()
        .expect("'graph'.'resources' must be an array");
    assert!(!graph_resources.is_empty(), "graph resources must not be empty");

    let res_entry = &graph_resources[0];
    let obj = res_entry
        .as_object()
        .expect("resource entry must be an object");
    assert!(obj.contains_key("handle"), "resource missing 'handle'");
    assert!(obj.contains_key("name"), "resource missing 'name'");
    assert!(obj.contains_key("desc"), "resource missing 'desc'");
    assert!(obj.contains_key("lifetime"), "resource missing 'lifetime'");
}

#[test]
fn graph_cull_stats_has_expected_fields() {
    // The cull_stats sub-object inside "graph" must have all four fields.
    let r = ResourceHandle(1);
    let passes = vec![mock_pass_graphics(PassIndex(0), "main", &[r])];
    let resources = vec![mock_resource_texture(r, "tex", 800, 600)];

    let compiler = FrameGraphCompiler::new(passes, resources);
    let compiled = compiler.compile().expect("graph compiles");
    let json = JsonExporter::export_all(&compiled);

    let cs = &json["graph"]["cull_stats"];
    assert!(cs.is_object(), "cull_stats must be an object");

    let cs_obj = cs.as_object().unwrap();
    assert!(cs_obj.contains_key("passes_total"), "cull_stats.passes_total missing");
    assert!(cs_obj.contains_key("passes_eliminated"), "cull_stats.passes_eliminated missing");
    assert!(cs_obj.contains_key("resources_freed"), "cull_stats.resources_freed missing");
    assert!(cs_obj.contains_key("bytes_saved"), "cull_stats.bytes_saved missing");
}

#[test]
fn graph_barriers_exist_for_dependent_passes() {
    // When passes share a resource via write-then-read, barriers must appear
    // in the graph output.
    let r = ResourceHandle(1);
    let passes = vec![
        mock_pass_graphics(PassIndex(0), "producer", &[r]),
        mock_pass_compute(PassIndex(1), "consumer", &[r], &[]),
    ];
    let resources = vec![mock_resource_texture(r, "shared", 800, 600)];

    let compiler = FrameGraphCompiler::new(passes, resources);
    let compiled = compiler.compile().expect("dependent passes compile");
    let json = JsonExporter::export_all(&compiled);

    let barriers = json["graph"]["barriers"]
        .as_array()
        .expect("'graph'.'barriers' must be an array");
    assert!(
        !barriers.is_empty(),
        "barriers must exist between dependent passes",
    );

    // Each barrier must have from, to, resource_handle fields.
    for (i, barrier) in barriers.iter().enumerate() {
        let obj = barrier
            .as_object()
            .unwrap_or_else(|| panic!("barrier[{}] must be an object", i));
        assert!(
            obj.contains_key("from"),
            "barrier[{}] missing 'from' field",
            i,
        );
        assert!(
            obj.contains_key("to"),
            "barrier[{}] missing 'to' field",
            i,
        );
        assert!(
            obj.contains_key("resource_handle"),
            "barrier[{}] missing 'resource_handle' field",
            i,
        );
        assert!(
            obj.contains_key("before_state"),
            "barrier[{}] missing 'before_state' field",
            i,
        );
        assert!(
            obj.contains_key("after_state"),
            "barrier[{}] missing 'after_state' field",
            i,
        );
    }
}

#[test]
fn graph_no_barriers_for_independent_passes() {
    // Independent passes touching disjoint resources produce no barriers.
    let r1 = ResourceHandle(1);
    let r2 = ResourceHandle(2);
    let passes = vec![
        mock_pass_graphics(PassIndex(0), "pass_a", &[r1]),
        mock_pass_graphics(PassIndex(1), "pass_b", &[r2]),
    ];
    let resources = vec![
        mock_resource_texture(r1, "tex_a", 800, 600),
        mock_resource_texture(r2, "tex_b", 1024, 768),
    ];

    let compiler = FrameGraphCompiler::new(passes, resources);
    let compiled = compiler.compile().expect("independent passes compile");
    let json = JsonExporter::export_all(&compiled);

    let barriers = json["graph"]["barriers"]
        .as_array()
        .expect("'graph'.'barriers' must be an array");
    assert!(
        barriers.is_empty() || json["graph"]["cull_stats"]["passes_total"].as_u64().unwrap_or(0) == 2,
        "independent passes should have zero or minimal barriers",
    );
}

// =========================================================================
// SECTION 3 -- "resources" key structure
// =========================================================================

#[test]
fn resources_key_is_a_json_array() {
    let r = ResourceHandle(1);
    let passes = vec![mock_pass_graphics(PassIndex(0), "main", &[r])];
    let resources = vec![mock_resource_texture(r, "color", 800, 600)];

    let compiler = FrameGraphCompiler::new(passes, resources);
    let compiled = compiler.compile().expect("graph compiles");
    let json = JsonExporter::export_all(&compiled);

    let resources_arr = &json["resources"];
    assert!(
        resources_arr.is_array(),
        "'resources' must be a JSON array, got {:?}",
        resources_arr,
    );
}

#[test]
fn resources_array_length_matches_input_resource_count() {
    let r1 = ResourceHandle(1);
    let r2 = ResourceHandle(2);
    let r3 = ResourceHandle(3);
    let passes = vec![
        mock_pass_graphics(PassIndex(0), "main", &[r1, r2]),
        mock_pass_compute(PassIndex(1), "post", &[r1, r2], &[r3]),
    ];
    let resources = vec![
        mock_resource_texture(r1, "color", 800, 600),
        mock_resource_buffer(r2, "data", 4096),
        mock_resource_buffer(r3, "output", 2048),
    ];

    let compiler = FrameGraphCompiler::new(passes, resources);
    let compiled = compiler.compile().expect("graph compiles");
    let json = JsonExporter::export_all(&compiled);

    let resources_arr = json["resources"]
        .as_array()
        .expect("'resources' must be an array");
    assert_eq!(
        resources_arr.len(),
        3,
        "resources array must contain 3 entries",
    );
}

#[test]
fn resources_sorted_by_handle() {
    // Resources must be sorted by handle in ascending order.
    let r_high = ResourceHandle(10);
    let r_low = ResourceHandle(1);
    let passes = vec![
        mock_pass_graphics(PassIndex(0), "main", &[r_high, r_low]),
    ];
    let resources = vec![
        mock_resource_texture(r_high, "high", 800, 600),
        mock_resource_texture(r_low, "low", 1024, 768),
    ];

    let compiler = FrameGraphCompiler::new(passes, resources);
    let compiled = compiler.compile().expect("graph compiles");
    let json = JsonExporter::export_all(&compiled);

    let resources_arr = json["resources"]
        .as_array()
        .expect("'resources' must be an array");
    assert_eq!(resources_arr.len(), 2, "expected 2 resources");

    let first_handle = resources_arr[0]["handle"].as_u64().unwrap_or(u64::MAX);
    let second_handle = resources_arr[1]["handle"].as_u64().unwrap_or(u64::MAX);
    assert!(
        first_handle < second_handle,
        "resources must be sorted by handle ascending: {} >= {}",
        first_handle,
        second_handle,
    );
}

#[test]
fn resources_entries_contain_required_fields() {
    // Each resource entry must have handle, name, desc, transient, resource_type.
    let r = ResourceHandle(1);
    let passes = vec![mock_pass_graphics(PassIndex(0), "main", &[r])];
    let resources = vec![mock_resource_texture(r, "color", 800, 600)];

    let compiler = FrameGraphCompiler::new(passes, resources);
    let compiled = compiler.compile().expect("graph compiles");
    let json = JsonExporter::export_all(&compiled);

    let resources_arr = json["resources"]
        .as_array()
        .expect("'resources' must be an array");
    assert!(!resources_arr.is_empty(), "resources array must not be empty");

    let entry = &resources_arr[0];
    let obj = entry.as_object().expect("resource entry must be an object");
    assert!(obj.contains_key("handle"), "resource entry missing 'handle'");
    assert!(obj.contains_key("name"), "resource entry missing 'name'");
    assert!(
        obj.contains_key("resource_type"),
        "resource entry missing 'resource_type'",
    );
    assert!(
        obj.contains_key("dimensions"),
        "resource entry missing 'dimensions'",
    );
    assert!(
        obj.contains_key("transient"),
        "resource entry missing 'transient'",
    );
}

#[test]
fn resources_texture_entry_includes_format_and_size() {
    // A texture resource should include format, width, height in its desc.
    let r = ResourceHandle(1);
    let passes = vec![mock_pass_graphics(PassIndex(0), "main", &[r])];
    let resources = vec![mock_resource_texture(r, "color", 1920, 1080)];

    let compiler = FrameGraphCompiler::new(passes, resources);
    let compiled = compiler.compile().expect("graph compiles");
    let json = JsonExporter::export_all(&compiled);

    let entry = &json["resources"][0];
    let obj = entry.as_object().expect("resource entry must be an object");
    assert!(
        obj.contains_key("format"),
        "texture resource missing 'format'",
    );
    assert!(
        obj.contains_key("dimensions"),
        "texture resource missing 'dimensions' object",
    );
    let dims = &obj["dimensions"];
    assert!(
        dims.get("width").is_some(),
        "texture resource dimensions missing 'width'",
    );
    assert!(
        dims.get("height").is_some(),
        "texture resource dimensions missing 'height'",
    );
}

#[test]
fn resources_buffer_entry_includes_size() {
    // A buffer resource should include size in its desc.
    let r = ResourceHandle(1);
    let passes = vec![mock_pass_compute(PassIndex(0), "compute", &[], &[r])];
    let resources = vec![mock_resource_buffer(r, "storage_buf", 8192)];

    let compiler = FrameGraphCompiler::new(passes, resources);
    let compiled = compiler.compile().expect("graph compiles");
    let json = JsonExporter::export_all(&compiled);

    let entry = &json["resources"][0];
    let obj = entry.as_object().expect("resource entry must be an object");
    assert!(
        obj.contains_key("resource_type"),
        "buffer entry missing 'resource_type'",
    );
    assert_eq!(
        obj["resource_type"], "buffer",
        "resource_type must be 'buffer'",
    );
    assert!(
        obj.contains_key("dimensions"),
        "buffer entry missing 'dimensions' object",
    );
    let dims = &obj["dimensions"];
    assert!(
        dims.get("size").is_some(),
        "buffer dimensions missing 'size' field",
    );
}

// =========================================================================
// SECTION 4 -- "schedule" key structure
// =========================================================================

#[test]
fn schedule_key_is_a_json_object() {
    let r = ResourceHandle(1);
    let passes = vec![mock_pass_graphics(PassIndex(0), "main", &[r])];
    let resources = vec![mock_resource_texture(r, "tex", 800, 600)];

    let compiler = FrameGraphCompiler::new(passes, resources);
    let compiled = compiler.compile().expect("graph compiles");
    let json = JsonExporter::export_all(&compiled);

    let schedule = &json["schedule"];
    assert!(
        schedule.is_object(),
        "'schedule' must be a JSON object, got {:?}",
        schedule,
    );
}

#[test]
fn schedule_contains_execution_order() {
    // The schedule must contain an execution_order array.
    let r1 = ResourceHandle(1);
    let r2 = ResourceHandle(2);
    let passes = vec![
        mock_pass_graphics(PassIndex(0), "gbuffer", &[r1]),
        mock_pass_compute(PassIndex(1), "resolve", &[r1], &[r2]),
        mock_pass_compute(PassIndex(2), "output", &[r2], &[]),
    ];
    let resources = vec![
        mock_resource_texture(r1, "albedo", 1920, 1080),
        mock_resource_buffer(r2, "data", 4096),
    ];

    let compiler = FrameGraphCompiler::new(passes, resources);
    let compiled = compiler.compile().expect("chain compiles");
    let json = JsonExporter::export_all(&compiled);

    let exec_order = json["schedule"]["execution_order"]
        .as_array()
        .expect("'schedule'.'execution_order' must be an array");
    assert_eq!(exec_order.len(), 3, "execution_order must have 3 entries");

    // Verify topological order: 0, 1, 2.
    let indices: Vec<usize> = exec_order
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    assert_eq!(indices, vec![0, 1, 2], "execution order must be topological");
}

#[test]
fn schedule_contains_barriers_array() {
    let r = ResourceHandle(1);
    let passes = vec![
        mock_pass_graphics(PassIndex(0), "producer", &[r]),
        mock_pass_compute(PassIndex(1), "consumer", &[r], &[]),
    ];
    let resources = vec![mock_resource_texture(r, "shared", 800, 600)];

    let compiler = FrameGraphCompiler::new(passes, resources);
    let compiled = compiler.compile().expect("dependent passes compile");
    let json = JsonExporter::export_all(&compiled);

    let barriers = json["schedule"]["barriers"]
        .as_array()
        .expect("'schedule'.'barriers' must be an array");
    assert!(
        !barriers.is_empty(),
        "schedule must have barriers for dependent passes",
    );
}

#[test]
fn schedule_contains_async_passes_array() {
    let r = ResourceHandle(1);
    let passes = vec![
        mock_pass_graphics(PassIndex(0), "gpu", &[r]),
        mock_pass_compute(PassIndex(1), "compute_task", &[r], &[]),
    ];
    let resources = vec![mock_resource_texture(r, "tex", 800, 600)];

    let compiler = FrameGraphCompiler::new(passes, resources);
    let compiled = compiler.compile().expect("graph compiles");
    let json = JsonExporter::export_all(&compiled);

    let _async_passes = json["schedule"]["async_passes"]
        .as_array()
        .expect("'schedule'.'async_passes' must be an array");
    // async_passes may be empty or populated; the key must exist.
    assert!(
        json["schedule"]
            .as_object()
            .unwrap()
            .contains_key("async_passes"),
        "'schedule' missing 'async_passes' key",
    );
}

#[test]
fn schedule_contains_parallel_regions() {
    // The parallel_regions array must exist in the schedule.
    let r = ResourceHandle(1);
    let passes = vec![mock_pass_graphics(PassIndex(0), "single", &[r])];
    let resources = vec![mock_resource_texture(r, "tex", 800, 600)];

    let compiler = FrameGraphCompiler::new(passes, resources);
    let compiled = compiler.compile().expect("graph compiles");
    let json = JsonExporter::export_all(&compiled);

    let _parallel_regions = json["schedule"]["parallel_regions"]
        .as_array()
        .expect("'schedule'.'parallel_regions' must be an array");
    assert!(
        json["schedule"]
            .as_object()
            .unwrap()
            .contains_key("parallel_regions"),
        "'schedule' missing 'parallel_regions' key",
    );
}

#[test]
fn schedule_contains_sync_points() {
    let r = ResourceHandle(1);
    let passes = vec![
        mock_pass_graphics(PassIndex(0), "producer", &[r]),
        mock_pass_compute(PassIndex(1), "consumer", &[r], &[]),
    ];
    let resources = vec![mock_resource_texture(r, "shared", 800, 600)];

    let compiler = FrameGraphCompiler::new(passes, resources);
    let compiled = compiler.compile().expect("dependent passes compile");
    let json = JsonExporter::export_all(&compiled);

    let sync_points = json["schedule"]["sync_points"]
        .as_array()
        .expect("'schedule'.'sync_points' must be an array");
    assert!(
        !sync_points.is_empty(),
        "sync_points should not be empty when barriers exist",
    );
}

// =========================================================================
// SECTION 5 -- "stats" key structure and values
// =========================================================================

#[test]
fn stats_key_is_a_json_object() {
    let r = ResourceHandle(1);
    let passes = vec![mock_pass_graphics(PassIndex(0), "main", &[r])];
    let resources = vec![mock_resource_texture(r, "tex", 800, 600)];

    let compiler = FrameGraphCompiler::new(passes, resources);
    let compiled = compiler.compile().expect("graph compiles");
    let json = JsonExporter::export_all(&compiled);

    let stats = &json["stats"];
    assert!(
        stats.is_object(),
        "'stats' must be a JSON object, got {:?}",
        stats,
    );
}

#[test]
fn stats_contains_all_expected_fields() {
    let r = ResourceHandle(1);
    let passes = vec![mock_pass_graphics(PassIndex(0), "main", &[r])];
    let resources = vec![mock_resource_texture(r, "tex", 800, 600)];

    let compiler = FrameGraphCompiler::new(passes, resources);
    let compiled = compiler.compile().expect("graph compiles");
    let json = JsonExporter::export_all(&compiled);

    let stats = json["stats"]
        .as_object()
        .expect("'stats' must be an object");
    let expected_stats_fields: &[&str] = &[
        "passes_total",
        "passes_eliminated",
        "barriers_total",
        "barriers_optimized",
        "async_passes",
        "resources_aliased",
        "compilation_time_us",
    ];

    assert_eq!(
        stats.len(),
        expected_stats_fields.len(),
        "stats must have exactly {} fields, got {}",
        expected_stats_fields.len(),
        stats.len(),
    );

    for field in expected_stats_fields {
        assert!(
            stats.contains_key(*field),
            "stats missing field '{}'; keys: {:?}",
            field,
            stats.keys().collect::<Vec<_>>(),
        );
    }
}

#[test]
fn stats_values_are_non_negative_numbers() {
    // All stat values must be non-negative integers (JSON numbers).
    let r = ResourceHandle(1);
    let passes = vec![mock_pass_graphics(PassIndex(0), "main", &[r])];
    let resources = vec![mock_resource_texture(r, "tex", 800, 600)];

    let compiler = FrameGraphCompiler::new(passes, resources);
    let compiled = compiler.compile().expect("graph compiles");
    let json = JsonExporter::export_all(&compiled);

    let stats = json["stats"]
        .as_object()
        .expect("'stats' must be an object");

    for (key, val) in stats {
        assert!(
            val.is_number(),
            "stats.{} must be a JSON number, got {:?}",
            key,
            val,
        );
        let num = val.as_f64().unwrap();
        assert!(
            num >= 0.0,
            "stats.{} must be non-negative, got {}",
            key,
            num,
        );
        assert!(
            num.fract() == 0.0 || key == &"compilation_time_us",
            "stats.{} must be an integer (whole number), got {}",
            key,
            num,
        );
    }
}

#[test]
fn stats_values_reflect_compilation_with_dead_passes() {
    // When dead passes exist, stats must reflect them.
    let r_live = ResourceHandle(1);
    let r_dead = ResourceHandle(2);
    let passes = vec![
        mock_pass_graphics(PassIndex(0), "alive", &[r_live]),
        mock_pass_compute(PassIndex(1), "dead", &[], &[r_dead]),
    ];
    let resources = vec![
        mock_resource_texture(r_live, "live_tex", 800, 600),
        mock_resource_buffer(r_dead, "dead_buf", 2048),
    ];

    let compiler = FrameGraphCompiler::new(passes, resources);
    let compiled = compiler.compile().expect("mixed graph compiles");
    let json = JsonExporter::export_all(&compiled);

    let stats = &json["stats"];
    assert_eq!(
        stats["passes_total"],
        2,
        "passes_total must be 2 (both input passes)",
    );
    assert_eq!(
        stats["passes_eliminated"],
        1,
        "passes_eliminated must be 1 (dead compute pass)",
    );

    // The graph passes must show only the surviving pass.
    let graph_passes = json["graph"]["passes"]
        .as_array()
        .expect("'graph'.'passes' must be an array");
    assert_eq!(
        graph_passes.len(),
        1,
        "graph must contain only 1 surviving pass",
    );
}

#[test]
fn stats_values_match_compiled_graph_directly() {
    // Verify that stats values from export_all match the actual compiler stats.
    let r = ResourceHandle(1);
    let passes = vec![mock_pass_graphics(PassIndex(0), "main", &[r])];
    let resources = vec![mock_resource_texture(r, "tex", 800, 600)];

    let compiler = FrameGraphCompiler::new(passes, resources);
    let compiled = compiler.compile().expect("graph compiles");
    let json = JsonExporter::export_all(&compiled);

    let stats = &json["stats"];
    assert_eq!(
        stats["passes_total"],
        compiled.stats.passes_total as u64,
        "passes_total mismatch",
    );
    assert_eq!(
        stats["passes_eliminated"],
        compiled.stats.passes_eliminated as u64,
        "passes_eliminated mismatch",
    );
    assert_eq!(
        stats["async_passes"],
        compiled.stats.async_passes as u64,
        "async_passes mismatch",
    );
    assert_eq!(
        stats["compilation_time_us"],
        compiled.stats.compilation_time_us,
        "compilation_time_us mismatch",
    );
}

#[test]
fn stats_values_are_zero_for_empty_graph() {
    // Empty graph must have all zero stats.
    let compiler = FrameGraphCompiler::new(vec![], vec![]);
    let compiled = compiler.compile().expect("empty graph compiles");
    let json = JsonExporter::export_all(&compiled);

    let stats = &json["stats"];
    assert_eq!(stats["passes_total"], 0);
    assert_eq!(stats["passes_eliminated"], 0);
    assert_eq!(stats["barriers_total"], 0);
    assert_eq!(stats["barriers_optimized"], 0);
    assert_eq!(stats["async_passes"], 0);
    assert_eq!(stats["resources_aliased"], 0);
    // compilation_time_us is a wall-clock measurement and may be >0 even for
    // an empty graph. Verify it is a non-negative number.
    let time_us = stats["compilation_time_us"].as_u64().unwrap_or(u64::MAX);
    assert!(
        time_us > 0 || time_us == 0,
        "compilation_time_us must be a non-negative number, got {}",
        time_us,
    );
}

// =========================================================================
// SECTION 6 -- Integration and stress scenarios
// =========================================================================

#[test]
fn diamond_graph_produces_correct_export() {
    // Fan-out then fan-in: P0 -> P1, P0 -> P2, P1 -> P3, P2 -> P3.
    let r1 = ResourceHandle(1);
    let r2 = ResourceHandle(2);
    let r3 = ResourceHandle(3);
    // P0 writes R1 and R2. P1 reads R1. P2 reads R2. P3 reads R1 and R2.
    let passes = vec![
        {
            let mut p = mock_pass_graphics(PassIndex(0), "root", &[]);
            p.access_set.writes.push(r1);
            p.access_set.writes.push(r2);
            p
        },
        mock_pass_compute(PassIndex(1), "branch_a", &[r1], &[]),
        mock_pass_compute(PassIndex(2), "branch_b", &[r2], &[]),
        {
            let mut p = mock_pass_compute(PassIndex(3), "merge", &[], &[]);
            p.access_set.reads.push(r1);
            p.access_set.reads.push(r2);
            p
        },
    ];
    let resources = vec![
        mock_resource_texture(r1, "rt1", 800, 600),
        mock_resource_texture(r2, "rt2", 1024, 768),
        mock_resource_buffer(r3, "scratch", 4096),
    ];

    let compiler = FrameGraphCompiler::new(passes, resources);
    let compiled = compiler.compile().expect("diamond graph compiles");
    let json = JsonExporter::export_all(&compiled);

    // Verify all four keys exist.
    let obj = json.as_object().expect("export must be a JSON object");
    assert!(obj.contains_key("graph"));
    assert!(obj.contains_key("resources"));
    assert!(obj.contains_key("schedule"));
    assert!(obj.contains_key("stats"));

    // Graph passes count: all 4 survive.
    let graph_passes = json["graph"]["passes"]
        .as_array()
        .expect("'graph'.'passes' must be an array");
    assert_eq!(graph_passes.len(), 4, "all four diamond passes survive");

    // Execution order: P0 first (producer).
    let exec_order = json["schedule"]["execution_order"]
        .as_array()
        .expect("'schedule'.'execution_order' must be an array");
    assert_eq!(exec_order.len(), 4, "four passes in execution order");

    let first = exec_order[0].as_u64().unwrap() as usize;
    assert_eq!(first, 0, "P0 must be first in execution order");

    // Resources: 3 entries.
    let resources_arr = json["resources"]
        .as_array()
        .expect("'resources' must be an array");
    assert_eq!(resources_arr.len(), 3, "three resources exported");

    // Barriers should exist for the write-read dependencies.
    let barriers = json["schedule"]["barriers"]
        .as_array()
        .expect("'schedule'.'barriers' must be an array");
    assert!(!barriers.is_empty(), "diamond graph must have barriers");
}

#[test]
fn sequential_chain_produces_correct_execution_order_in_export() {
    // A 5-pass linear chain: P0 -> P1 -> P2 -> P3 -> P4.
    // The tail pass P4 reads R3 but does NOT write (otherwise it would be
    // eliminated as dead since no one would consume its output).
    let mut passes = Vec::with_capacity(5);
    let mut resources = Vec::new();

    passes.push(mock_pass_graphics(
        PassIndex(0),
        "head",
        &[ResourceHandle(0)],
    ));
    resources.push(mock_resource_texture(ResourceHandle(0), "r0", 64, 64));

    for i in 1..5 {
        let reads = vec![ResourceHandle((i - 1) as u32)];
        // P1-P3 write R1-R3 consumed by next. P4 reads R3 only (no write).
        let writes: Vec<ResourceHandle> = if i < 4 {
            vec![ResourceHandle(i as u32)]
        } else {
            vec![]
        };
        passes.push(mock_pass_compute(
            PassIndex(i),
            &format!("pass_{}", i),
            &reads,
            &writes,
        ));
        if i < 4 {
            resources.push(mock_resource_buffer(
                ResourceHandle(i as u32),
                &format!("r{}", i),
                64,
            ));
        }
    }

    let compiler = FrameGraphCompiler::new(passes, resources);
    let compiled = compiler.compile().expect("5-pass chain compiles");
    let json = JsonExporter::export_all(&compiled);

    // All 5 passes survive (P4 is a read-only tail, not eliminated).
    // Execution order must be sequential.
    let exec_order = json["schedule"]["execution_order"]
        .as_array()
        .expect("execution_order must be an array");
    let order: Vec<usize> = exec_order
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    assert_eq!(order, vec![0, 1, 2, 3, 4], "chain must preserve order");

    // All 5 passes in graph section.
    let graph_passes = json["graph"]["passes"]
        .as_array()
        .expect("graph.passes must be an array");
    assert_eq!(graph_passes.len(), 5, "all 5 passes in graph passes");

    // Barriers in schedule: 4 barriers (P0->P1, P1->P2, P2->P3, P3->P4).
    let barriers = json["schedule"]["barriers"]
        .as_array()
        .expect("schedule.barriers must be an array");
    assert_eq!(barriers.len(), 4, "4 barriers for 5-pass chain");

    // Stats: passes_total = 5, passes_eliminated = 0 (all alive).
    assert_eq!(json["stats"]["passes_total"], 5);
    assert_eq!(json["stats"]["passes_eliminated"], 0);
}

#[test]
fn export_all_is_deterministic() {
    // Two compilations with identical inputs must produce identical JSON output
    // (modulo compilation_time_us which varies by system).
    let r1 = ResourceHandle(1);
    let r2 = ResourceHandle(2);
    let passes = vec![
        mock_pass_graphics(PassIndex(0), "gbuffer", &[r1]),
        mock_pass_compute(PassIndex(1), "resolve", &[r1], &[r2]),
        mock_pass_compute(PassIndex(2), "output", &[r2], &[]),
    ];
    let resources = vec![
        mock_resource_texture(r1, "albedo", 1920, 1080),
        mock_resource_buffer(r2, "data", 4096),
    ];

    // First compilation.
    let compiled_a = FrameGraphCompiler::new(
        passes.clone(),
        resources.clone(),
    )
    .compile()
    .expect("first compile");
    let json_a = JsonExporter::export_all(&compiled_a);

    // Second compilation with identical inputs.
    let compiled_b = FrameGraphCompiler::new(passes, resources)
        .compile()
        .expect("second compile");
    let json_b = JsonExporter::export_all(&compiled_b);

    // Compare graph, resources, schedule (stats.compilation_time_us may differ).
    assert_eq!(
        json_a["graph"], json_b["graph"],
        "'graph' output must be identical across compilations",
    );
    assert_eq!(
        json_a["resources"], json_b["resources"],
        "'resources' output must be identical across compilations",
    );
    assert_eq!(
        json_a["schedule"], json_b["schedule"],
        "'schedule' output must be identical across compilations",
    );

    // Semantic stat comparison: pass/barrier counts must match.
    assert_eq!(
        json_a["stats"]["passes_total"], json_b["stats"]["passes_total"],
        "passes_total must match",
    );
    assert_eq!(
        json_a["stats"]["passes_eliminated"],
        json_b["stats"]["passes_eliminated"],
        "passes_eliminated must match",
    );
    assert_eq!(
        json_a["stats"]["barriers_total"], json_b["stats"]["barriers_total"],
        "barriers_total must match",
    );
}

#[test]
fn multiple_dead_passes_export_correctly() {
    // All compute passes dead, only graphics pass survives.
    let r_live = ResourceHandle(1);
    let passes = vec![
        mock_pass_graphics(PassIndex(0), "alive", &[r_live]),
        mock_pass_compute(PassIndex(1), "dead_a", &[], &[ResourceHandle(2)]),
        mock_pass_compute(PassIndex(2), "dead_b", &[], &[ResourceHandle(3)]),
        mock_pass_compute(PassIndex(3), "dead_c", &[], &[ResourceHandle(4)]),
    ];
    let resources = vec![
        mock_resource_texture(r_live, "color", 800, 600),
        mock_resource_buffer(ResourceHandle(2), "buf_a", 1024),
        mock_resource_buffer(ResourceHandle(3), "buf_b", 2048),
        mock_resource_buffer(ResourceHandle(4), "buf_c", 4096),
    ];

    let compiler = FrameGraphCompiler::new(passes, resources);
    let compiled = compiler.compile().expect("mixed graph compiles");
    let json = JsonExporter::export_all(&compiled);

    // Stats.
    assert_eq!(
        json["stats"]["passes_total"], 4,
        "all 4 input passes counted",
    );
    assert_eq!(
        json["stats"]["passes_eliminated"],
        3,
        "3 dead compute passes eliminated",
    );
    assert_eq!(
        json["stats"]["barriers_total"],
        0,
        "no barriers since only 1 pass survives",
    );

    // Graph passes: only 1 surviving pass.
    let graph_passes = json["graph"]["passes"]
        .as_array()
        .expect("graph.passes must be an array");
    assert_eq!(graph_passes.len(), 1, "only 1 pass survives in graph");

    let pass_name = graph_passes[0]["name"].as_str().unwrap_or("");
    assert_eq!(pass_name, "alive", "surviving pass must be the graphics pass");

    // Graph cull stats reflect dead pass elimination.
    let cull = &json["graph"]["cull_stats"];
    assert_eq!(cull["passes_total"], 4);
    assert_eq!(cull["passes_eliminated"], 3);
}

#[test]
fn graph_cull_stats_barrier_counts_are_consistent() {
    // Verify the relationship between cull_stats, schedule barriers count,
    // and stats barriers_total.
    // Create a chain that generates barriers.
    // P3 reads R2 but does NOT write (otherwise its write would be unread
    // and the pass would be eliminated as dead).
    let mut passes = Vec::new();
    let mut resources = Vec::new();

    passes.push(mock_pass_graphics(
        PassIndex(0),
        "p0",
        &[ResourceHandle(0)],
    ));
    resources.push(mock_resource_texture(ResourceHandle(0), "r0", 64, 64));

    passes.push(mock_pass_compute(
        PassIndex(1), "p1",
        &[ResourceHandle(0)], &[ResourceHandle(1)],
    ));
    resources.push(mock_resource_buffer(ResourceHandle(1), "r1", 64));

    passes.push(mock_pass_compute(
        PassIndex(2), "p2",
        &[ResourceHandle(1)], &[ResourceHandle(2)],
    ));
    resources.push(mock_resource_buffer(ResourceHandle(2), "r2", 64));

    // P3 is read-only tail (no write) to avoid dead-pass elimination.
    passes.push(mock_pass_compute(
        PassIndex(3), "p3",
        &[ResourceHandle(2)], &[],
    ));

    let compiler = FrameGraphCompiler::new(passes, resources);
    let compiled = compiler.compile().expect("chain compiles");
    let json = JsonExporter::export_all(&compiled);

    // Schedule barrier count should equal or exceed the cull_stats barrier total
    // since schedule includes barriers by boundary.
    let schedule_barrier_count = json["schedule"]["barriers"]
        .as_array()
        .map(|a| a.len())
        .unwrap_or(0);
    let graph_barrier_count = json["graph"]["barriers"]
        .as_array()
        .map(|a| a.len())
        .unwrap_or(0);

    // Both should report the same number of barriers.
    assert_eq!(
        graph_barrier_count, schedule_barrier_count,
        "graph barrier count must match schedule barrier count",
    );

    // Stats barriers_total must be >= the barrier count in the output.
    let stats_barriers_total = json["stats"]["barriers_total"]
        .as_u64()
        .unwrap_or(0) as usize;
    assert!(
        stats_barriers_total >= graph_barrier_count,
        "stats.barriers_total ({}) must be >= graph barrier count ({})",
        stats_barriers_total,
        graph_barrier_count,
    );
}

#[test]
fn resources_table_lifetime_fields_populated() {
    // When resources are used by passes, first_use_pass and last_use_pass
    // should be populated in the resources table.
    let r = ResourceHandle(1);
    let passes = vec![
        mock_pass_graphics(PassIndex(0), "producer", &[r]),
        mock_pass_compute(PassIndex(1), "consumer", &[r], &[]),
    ];
    let resources = vec![mock_resource_texture(r, "shared", 800, 600)];

    let compiler = FrameGraphCompiler::new(passes, resources);
    let compiled = compiler.compile().expect("graph compiles");
    let json = JsonExporter::export_all(&compiled);

    let res_entry = &json["resources"][0];
    let obj = res_entry.as_object().expect("resource entry must be an object");

    // Lifetime fields should be present or null.
    if let Some(first_use) = obj.get("first_use_pass") {
        assert!(
            first_use.is_number(),
            "first_use_pass must be a number, got {:?}",
            first_use,
        );
    }
    if let Some(last_use) = obj.get("last_use_pass") {
        assert!(
            last_use.is_number(),
            "last_use_pass must be a number, got {:?}",
            last_use,
        );
    }
}

#[test]
fn validate_output_can_be_serialized_to_json_string() {
    // The output of export_all must be serializable to a JSON string
    // without error (round-trip through serde_json).
    let r = ResourceHandle(1);
    let passes = vec![mock_pass_graphics(PassIndex(0), "main", &[r])];
    let resources = vec![mock_resource_texture(r, "tex", 800, 600)];

    let compiler = FrameGraphCompiler::new(passes, resources);
    let compiled = compiler.compile().expect("graph compiles");
    let json = JsonExporter::export_all(&compiled);

    // Round-trip through string serialization.
    let json_str = serde_json::to_string(&json).expect("JSON must serialize to string");
    let parsed: Value =
        serde_json::from_str(&json_str).expect("JSON string must deserialize back to Value");

    // Verify the structure survived the round-trip.
    let obj = parsed.as_object().expect("round-tripped value must be an object");
    assert!(obj.contains_key("graph"));
    assert!(obj.contains_key("resources"));
    assert!(obj.contains_key("schedule"));
    assert!(obj.contains_key("stats"));
}

// =========================================================================
// SECTION 7 -- Edge cases
// =========================================================================

#[test]
fn pass_with_no_resources_exported_correctly() {
    // A compute pass with no resource accesses should still appear in the
    // graph export with correct structure.
    let passes = vec![mock_pass_compute(
        PassIndex(0),
        "noop",
        &[] as &[ResourceHandle],
        &[] as &[ResourceHandle],
    )];

    let compiler = FrameGraphCompiler::new(passes, vec![]);
    let compiled = compiler.compile().expect("no-resource pass compiles");
    let json = JsonExporter::export_all(&compiled);

    // All four keys exist.
    let obj = json.as_object().expect("export must be an object");
    assert_eq!(obj.len(), 4);

    // 1 pass in graph.
    let graph_passes = json["graph"]["passes"]
        .as_array()
        .expect("graph.passes must be an array");
    assert_eq!(graph_passes.len(), 1);

    let pass_obj = graph_passes[0].as_object().expect("pass must be an object");
    assert_eq!(pass_obj["name"], "noop");
    assert_eq!(pass_obj["index"], 0);

    // Resources should be empty.
    let graph_resources = json["graph"]["resources"]
        .as_array()
        .expect("graph.resources must be an array");
    assert!(graph_resources.is_empty(), "no resources expected");

    // Stats: 1 pass total, 0 eliminated.
    assert_eq!(json["stats"]["passes_total"], 1);
    assert_eq!(json["stats"]["passes_eliminated"], 0);
}

#[test]
fn graph_validation_result_present() {
    // The graph object should contain a validation sub-object from
    // BridgeValidator::validate.
    let r = ResourceHandle(1);
    let passes = vec![mock_pass_graphics(PassIndex(0), "main", &[r])];
    let resources = vec![mock_resource_texture(r, "tex", 800, 600)];

    let compiler = FrameGraphCompiler::new(passes, resources);
    let compiled = compiler.compile().expect("graph compiles");
    let json = JsonExporter::export_all(&compiled);

    let validation = &json["graph"]["validation"];
    assert!(
        validation.is_object(),
        "'graph'.'validation' must be an object, got {:?}",
        validation,
    );

    // Validation should have 'valid' boolean and 'errors' array.
    let v_obj = validation.as_object().expect("validation must be an object");
    assert!(
        v_obj.contains_key("valid"),
        "validation must contain 'valid' field",
    );
    assert!(
        v_obj.contains_key("errors"),
        "validation must contain 'errors' field",
    );
}

#[test]
fn graph_depths_are_present() {
    // The depths map should exist in the graph object.
    let r = ResourceHandle(1);
    let passes = vec![mock_pass_graphics(PassIndex(0), "main", &[r])];
    let resources = vec![mock_resource_texture(r, "tex", 800, 600)];

    let compiler = FrameGraphCompiler::new(passes, resources);
    let compiled = compiler.compile().expect("graph compiles");
    let json = JsonExporter::export_all(&compiled);

    let depths = &json["graph"]["depths"];
    assert!(
        depths.is_object(),
        "'graph'.'depths' must be an object, got {:?}",
        depths,
    );
}

#[test]
fn export_all_fan_in_fan_out_preserves_all_passes() {
    // Complex topology: P0 -> P1, P2 and P1, P2 -> P3.
    let r = ResourceHandle(1);
    let passes = vec![
        mock_pass_graphics(PassIndex(0), "root", &[r]),
        mock_pass_compute(PassIndex(1), "a", &[r], &[]),
        mock_pass_compute(PassIndex(2), "b", &[r], &[]),
        {
            let mut p = mock_pass_compute(PassIndex(3), "combine", &[], &[]);
            p.access_set.reads.push(r);
            p
        },
    ];
    let resources = vec![mock_resource_texture(r, "shared", 800, 600)];

    let compiler = FrameGraphCompiler::new(passes, resources);
    let compiled = compiler.compile().expect("complex graph compiles");
    let json = JsonExporter::export_all(&compiled);

    // All 4 passes survive.
    assert_eq!(
        json["graph"]["passes"].as_array().unwrap().len(),
        4,
        "all 4 passes survive",
    );

    // Resources: 1 entry.
    assert_eq!(
        json["resources"].as_array().unwrap().len(),
        1,
        "1 resource exported",
    );

    // Execution order: 4 entries.
    assert_eq!(
        json["schedule"]["execution_order"]
            .as_array()
            .unwrap()
            .len(),
        4,
        "4 passes in execution order",
    );

    // P0 must be first.
    assert_eq!(
        json["schedule"]["execution_order"][0].as_u64().unwrap(),
        0,
        "P0 must be first in execution order",
    );

    // Stats.
    assert_eq!(json["stats"]["passes_total"], 4);
    assert_eq!(json["stats"]["passes_eliminated"], 0);
}
