// Blackbox contract tests for T-FG-6.6: EdgeType in barrier dedup.
//
// CLEANROOM: No src/ access beyond the public API exported by the crate.
// Tests use only `renderer_backend::frame_graph::*` -- no internal fields,
// no private methods, no implementation details.
//
// Acceptance criteria (T-FG-6.6 fix):
//   Barrier dedup key now includes EdgeType so that different edge types
//   between the same passes for the same resource produce distinct barriers.
//   compute_barriers returns 6-tuples:
//     (PassIndex, PassIndex, ResourceHandle, EdgeType, ResourceState, ResourceState)
//   where EdgeType occupies index 3 (dedup discriminator).
//
// Scenarios:
//   1.  RAW + WAW same resource -> 2 distinct barriers with correct EdgeType
//   2.  Only RAW -> 1 barrier with EdgeType::RAW
//   3.  Only WAW -> 1 barrier with EdgeType::WAW
//   4.  Two RAW edges same passes same resource -> 1 barrier (dedup still works)
//   5.  JSON export includes "edge_type" field
//   6.  JSON edge_type values are valid strings (RAW, WAW, WAR)
//   7.  Empty graph -> barriers empty, no edge_type errors
//   8.  Full pipeline round-trip (compile -> export -> deserialize -> re-compile)

use renderer_backend::frame_graph::{
    compute_barriers, mock_pass_compute, mock_pass_graphics, mock_resource_buffer,
    mock_resource_texture, EdgeType, FrameGraphCompiler, IrEdge, JsonExporter, PassIndex,
    ResourceHandle,
};
use serde_json::Value;

// =========================================================================
// SECTION 1 -- RAW + WAW for same resource -> 2 distinct barriers
// =========================================================================

#[test]
fn raw_and_waw_same_resource_produce_two_barriers() {
    // P0 writes R. P1 both reads and writes R, producing both RAW and WAW
    // edges between P0 and P1 for resource R.
    // With EdgeType in the dedup key, these must produce 2 distinct barriers.
    let r = ResourceHandle(1);
    let passes = vec![
        mock_pass_graphics(PassIndex(0), "producer", &[r]),
        {
            let mut p = mock_pass_compute(PassIndex(1), "consumer", &[], &[]);
            p.access_set.reads.push(r);
            p.access_set.writes.push(r);
            p
        },
    ];
    let resources = vec![mock_resource_texture(r, "tex", 800, 600)];
    let edges = vec![
        IrEdge::new(PassIndex(0), PassIndex(1), r, EdgeType::RAW),
        IrEdge::new(PassIndex(0), PassIndex(1), r, EdgeType::WAW),
    ];
    let order = vec![PassIndex(0), PassIndex(1)];

    let barriers = compute_barriers(&order, &passes, &edges);

    // Two distinct edge types at same (from, to, resource) -> 2 barrier tuples.
    assert_eq!(
        barriers.len(),
        2,
        "RAW+WAW for same resource must produce 2 barrier tuples",
    );

    // Each edge type must appear exactly once.
    let raw_count = barriers.iter().filter(|b| b.3 == EdgeType::RAW).count();
    let waw_count = barriers.iter().filter(|b| b.3 == EdgeType::WAW).count();
    assert_eq!(raw_count, 1, "one RAW barrier must exist");
    assert_eq!(waw_count, 1, "one WAW barrier must exist");

    // Both barriers reference the same passes and resource.
    for (i, b) in barriers.iter().enumerate() {
        assert_eq!(b.0, PassIndex(0), "barrier[{}].from must be P0", i);
        assert_eq!(b.1, PassIndex(1), "barrier[{}].to must be P1", i);
        assert_eq!(b.2, r, "barrier[{}].resource must be R(1)", i);
    }
}

// =========================================================================
// SECTION 2 -- Only RAW -> 1 barrier with EdgeType::RAW
// =========================================================================

#[test]
fn only_raw_produces_one_barrier() {
    // P0 writes R, P1 reads R. Only a RAW edge exists.
    let r = ResourceHandle(1);
    let passes = vec![
        mock_pass_graphics(PassIndex(0), "producer", &[r]),
        mock_pass_compute(PassIndex(1), "consumer", &[r], &[]),
    ];
    let resources = vec![mock_resource_texture(r, "tex", 800, 600)];
    let edges = vec![IrEdge::new(PassIndex(0), PassIndex(1), r, EdgeType::RAW)];
    let order = vec![PassIndex(0), PassIndex(1)];

    let barriers = compute_barriers(&order, &passes, &edges);

    assert_eq!(barriers.len(), 1, "single RAW edge -> 1 barrier");
    assert_eq!(barriers[0].3, EdgeType::RAW, "barrier must be RAW");
    assert_eq!(barriers[0].0, PassIndex(0), "from must be P0");
    assert_eq!(barriers[0].1, PassIndex(1), "to must be P1");
    assert_eq!(barriers[0].2, r, "resource must be R");
}

// =========================================================================
// SECTION 3 -- Only WAW -> 1 barrier with EdgeType::WAW
// =========================================================================

#[test]
fn only_waw_produces_one_barrier() {
    // Both P0 and P1 write R. Only a WAW edge exists.
    let r = ResourceHandle(1);
    let passes = vec![
        mock_pass_graphics(PassIndex(0), "producer", &[r]),
        {
            let mut p = mock_pass_compute(PassIndex(1), "consumer", &[], &[]);
            p.access_set.writes.push(r);
            p
        },
    ];
    let resources = vec![mock_resource_texture(r, "tex", 800, 600)];
    let edges = vec![IrEdge::new(PassIndex(0), PassIndex(1), r, EdgeType::WAW)];
    let order = vec![PassIndex(0), PassIndex(1)];

    let barriers = compute_barriers(&order, &passes, &edges);

    assert_eq!(barriers.len(), 1, "single WAW edge -> 1 barrier");
    assert_eq!(barriers[0].3, EdgeType::WAW, "barrier must be WAW");
    assert_eq!(barriers[0].0, PassIndex(0), "from must be P0");
    assert_eq!(barriers[0].1, PassIndex(1), "to must be P1");
    assert_eq!(barriers[0].2, r, "resource must be R");
}

// =========================================================================
// SECTION 4 -- Two identical RAW edges -> 1 barrier (dedup still works)
// =========================================================================

#[test]
fn duplicate_raw_dedup_to_one_barrier() {
    // Two RAW edges with the same (from, to, resource, edge_type) must be
    // deduped to a single barrier. EdgeType is now part of the dedup key,
    // so matching edge types still collapse into one entry.
    let r = ResourceHandle(1);
    let passes = vec![
        mock_pass_graphics(PassIndex(0), "producer", &[r]),
        mock_pass_compute(PassIndex(1), "consumer", &[r], &[]),
    ];
    let resources = vec![mock_resource_texture(r, "tex", 800, 600)];
    // Two identical RAW edges -- dedup should collapse them.
    let edges = vec![
        IrEdge::new(PassIndex(0), PassIndex(1), r, EdgeType::RAW),
        IrEdge::new(PassIndex(0), PassIndex(1), r, EdgeType::RAW),
    ];
    let order = vec![PassIndex(0), PassIndex(1)];

    let barriers = compute_barriers(&order, &passes, &edges);

    // Same edge type -> still deduped to 1 barrier.
    assert_eq!(
        barriers.len(),
        1,
        "duplicate RAW edges must produce 1 barrier (dedup still works)",
    );
    assert_eq!(barriers[0].3, EdgeType::RAW, "barrier must be RAW");
}

// =========================================================================
// SECTION 5 -- JSON export includes "edge_type" field
// =========================================================================

#[test]
fn json_export_includes_edge_type_field() {
    let r_tex = ResourceHandle(1);
    let r_buf = ResourceHandle(2);
    let passes = vec![
        mock_pass_graphics(PassIndex(0), "gbuffer", &[r_tex]),
        {
            let mut p = mock_pass_compute(PassIndex(1), "post", &[], &[]);
            p.access_set.reads.push(r_tex);
            p.access_set.writes.push(r_tex);
            p.access_set.reads.push(r_buf);
            p
        },
    ];
    let resources = vec![
        mock_resource_texture(r_tex, "color", 800, 600),
        mock_resource_buffer(r_buf, "data", 4096),
    ];

    let compiler = FrameGraphCompiler::from_ir(passes, resources);
    let compiled = compiler.expect("graph with RAW+WAW compiles");
    let json = JsonExporter::export_all(&compiled);

    // Check graph section barriers.
    let graph_barriers = json["graph"]["barriers"]
        .as_array()
        .expect("json['graph']['barriers'] must be an array");
    for (i, barrier) in graph_barriers.iter().enumerate() {
        let obj = barrier
            .as_object()
            .unwrap_or_else(|| panic!("graph barrier[{}] must be an object", i));
        assert!(
            obj.contains_key("edge_type"),
            "graph barrier[{}] must contain 'edge_type' field",
            i,
        );
    }

    // Check schedule section barriers.
    let sched_barriers = json["schedule"]["barriers"]
        .as_array()
        .expect("json['schedule']['barriers'] must be an array");
    for (i, barrier) in sched_barriers.iter().enumerate() {
        let obj = barrier
            .as_object()
            .unwrap_or_else(|| panic!("schedule barrier[{}] must be an object", i));
        assert!(
            obj.contains_key("edge_type"),
            "schedule barrier[{}] must contain 'edge_type' field",
            i,
        );
    }
}

// =========================================================================
// SECTION 6 -- JSON edge_type values are valid strings
// =========================================================================

#[test]
fn json_edge_type_values_are_valid_strings() {
    let r = ResourceHandle(1);
    let passes = vec![
        mock_pass_graphics(PassIndex(0), "producer", &[r]),
        {
            let mut p = mock_pass_compute(PassIndex(1), "consumer", &[], &[]);
            p.access_set.reads.push(r);
            p.access_set.writes.push(r);
            p
        },
    ];
    let resources = vec![mock_resource_texture(r, "tex", 800, 600)];

    let compiler = FrameGraphCompiler::from_ir(passes, resources);
    let compiled = compiler.expect("graph compiles");
    let json = JsonExporter::export_all(&compiled);

    // Collect barriers from both graph and schedule sections.
    let sections = ["graph", "schedule"];
    for section in &sections {
        let barriers = json[section]["barriers"]
            .as_array()
            .unwrap_or_else(|| panic!("json['{}']['barriers'] must be an array", section));
        for (i, barrier) in barriers.iter().enumerate() {
            let obj = barrier.as_object().unwrap_or_else(|| {
                panic!("{} barrier[{}] must be an object", section, i)
            });
            let et = obj["edge_type"].as_str().unwrap_or_else(|| {
                panic!("{} barrier[{}] edge_type must be a string", section, i)
            });
            assert!(
                ["RAW", "WAR", "WAW"].contains(&et),
                "{} barrier[{}] edge_type '{}' must be one of RAW, WAR, WAW",
                section,
                i,
                et,
            );
        }
    }
}

// =========================================================================
// SECTION 7 -- Empty graph edge case
// =========================================================================

#[test]
fn empty_graph_no_barriers() {
    let compiler = FrameGraphCompiler::from_ir(vec![], vec![]);
    let compiled = compiler.expect("empty graph compiles");
    let json = JsonExporter::export_all(&compiled);

    // Graph barriers must be empty.
    let graph_barriers = json["graph"]["barriers"]
        .as_array()
        .expect("json['graph']['barriers'] must be an array");
    assert!(
        graph_barriers.is_empty(),
        "empty graph must produce no graph barriers",
    );

    // Schedule barriers must also be empty.
    let sched_barriers = json["schedule"]["barriers"]
        .as_array()
        .expect("json['schedule']['barriers'] must be an array");
    assert!(
        sched_barriers.is_empty(),
        "empty graph must produce no schedule barriers",
    );

    // Confirm the JSON round-trips without error.
    let json_str = serde_json::to_string(&json).expect("empty graph JSON serialization");
    assert!(!json_str.is_empty(), "JSON output must not be empty");

    // Re-parse and verify barriers remain empty.
    let re_parsed: Value = serde_json::from_str(&json_str).expect("empty graph JSON round-trip");
    assert_eq!(
        re_parsed["graph"]["barriers"]
            .as_array()
            .map(|a| a.len()),
        Some(0),
        "empty graph barriers remain empty after round-trip",
    );
}

// =========================================================================
// SECTION 8 -- Full pipeline round-trip
// =========================================================================

#[test]
fn full_pipeline_round_trip_preserves_edge_type() {
    // Build a simple graph with a RAW dependency.
    let r = ResourceHandle(1);
    let passes = vec![
        mock_pass_graphics(PassIndex(0), "producer", &[r]),
        mock_pass_compute(PassIndex(1), "consumer", &[r], &[]),
    ];
    let resources = vec![mock_resource_texture(r, "tex", 800, 600)];

    // Step 1: Compile.
    let compiler = FrameGraphCompiler::from_ir(passes, resources);
    let compiled = compiler.expect("graph compiles");

    // Step 2: Export to JSON.
    let json = JsonExporter::export_all(&compiled);

    // Step 3: Verify barriers in exported JSON have edge_type.
    let barriers = json["graph"]["barriers"]
        .as_array()
        .expect("graph.barriers must be an array");
    assert!(!barriers.is_empty(), "dependent passes must produce barriers");
    for (i, barrier) in barriers.iter().enumerate() {
        let obj = barrier
            .as_object()
            .unwrap_or_else(|| panic!("barrier[{}] must be an object", i));
        assert!(
            obj.contains_key("edge_type"),
            "barrier[{}] missing edge_type",
            i,
        );
        let et = obj["edge_type"]
            .as_str()
            .unwrap_or_else(|| panic!("barrier[{}] edge_type not a string", i));
        assert!(
            ["RAW", "WAR", "WAW"].contains(&et),
            "barrier[{}] edge_type '{}' invalid",
            i,
            et,
        );
    }

    // Step 4: Serialize JSON to string and deserialize back to passes/resources.
    let json_str =
        serde_json::to_string(&json).expect("JSON serialization succeeds");
    let (rt_passes, rt_resources) =
        renderer_backend::frame_graph::deserialize_from_json(&json_str)
            .expect("deserialize_from_json must succeed");

    // Step 5: Re-compile with the deserialized graph.
    let re_compiler = FrameGraphCompiler::from_ir(rt_passes, rt_resources);
    let re_compiled = re_compiler.expect("re-compilation succeeds");

    // Step 6: Export the re-compiled graph and verify barrier edge_type.
    let re_json = JsonExporter::export_all(&re_compiled);
    let re_barriers = re_json["graph"]["barriers"]
        .as_array()
        .expect("re-compiled graph.barriers must be an array");
    for (i, barrier) in re_barriers.iter().enumerate() {
        let obj = barrier.as_object().unwrap_or_else(|| {
            panic!("re-compiled barrier[{}] must be an object", i)
        });
        assert!(
            obj.contains_key("edge_type"),
            "re-compiled barrier[{}] missing edge_type",
            i,
        );
        assert!(
            obj["edge_type"].as_str().is_some(),
            "re-compiled barrier[{}] edge_type not a string",
            i,
        );
    }
}
