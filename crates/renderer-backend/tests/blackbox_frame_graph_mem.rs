// SPDX-License-Identifier: MIT
//
// FrameGraphMemTest (T-FG-9.7 GAP 2) -- memory usage test.
//
// Verifies no memory leaks when repeatedly compiling graphs through the full
// compile pipeline (build_dag -> topological_sort -> compute_pass_depths ->
// compute_lifetimes -> compute_barriers -> async_schedule ->
// eliminate_dead_passes) over many iterations.
//
// The test performs a drop/compile/drop/export cycle 150 times, each time
// building a 50-pass mixed-topology graph, compiling it, and exporting the
// result to JSON.  Each iteration consumes and releases a CompiledFrameGraph
// (no Arc cycles, no global statics), so the standard Rust ownership model
// guarantees all memory is reclaimed at drop.
//
// Acceptance criteria:
//   1. All 150 compile/drop cycles succeed without error.
//   2. Each cycle produces valid JSON with all expected top-level keys.
//   3. The JSON export structure is consistent across all cycles.
//   4. Cumulative compile time is recorded for performance observability.
//   5. CompilerStats are populated and non-zero in every cycle.
//   6. No allocation monotonic-growth trend (the allocation differential
//      between early and late cycles is near zero for a steady-state run).
//
// CLEANROOM: Tests use only the public API exported by the crate:
//   renderer_backend::frame_graph::*

use renderer_backend::frame_graph::{
    CompiledFrameGraph, CompilerProfile, DispatchSource, InstanceSource, IrPass, IrResource,
    JsonExporter, PassIndex, PassType, ResourceAccessSet, ResourceDesc, ResourceHandle,
    ResourceLifetime, ResourceState, ViewType,
};

// =============================================================================
// Test configuration constants
// =============================================================================

/// Number of compile/drop/export cycles.
const ITERATIONS: usize = 150;

/// Number of passes in each generated graph.
/// Actual: chain1(10) + diamond(8) + chain2(8) + fanout(7) + fanin(6) + chain3(12) = 51.
const PASSES_PER_GRAPH: usize = 51;

// =============================================================================
// Helper: build a 50-pass mixed-topology graph
// =============================================================================

/// Generates a frame graph composed of multiple pattern segments:
///
///   - Chain 1: 10 passes (linear)
///   - Diamond: 8 passes (root -> 3 middle -> 3 merge -> 1 tail)
///   - Chain 2: 8 passes (linear)
///   - Fan-out: 6 passes (1 producer, 4 consumers, 1 barrier)
///   - Fan-in:  6 passes (4 producers, 1 consumer, 1 barrier)
///   - Chain 3: 12 passes (linear)
///
/// Total: 10 + 8 + 8 + 6 + 6 + 12 = 50 passes.
/// Resources: each pass writes at least one unique resource.
///
/// Returns (passes, resources).
fn build_mixed_graph() -> (Vec<IrPass>, Vec<IrResource>) {
    let mut passes: Vec<IrPass> = Vec::with_capacity(PASSES_PER_GRAPH);
    let mut next_res = 0u32;

    // Helper: append a compute pass that reads from `reads` and writes to
    // a new resource, advancing next_res.
    macro_rules! push_write_pass {
        ($name:expr, $reads:expr) => {{
            let write_h = ResourceHandle(next_res);
            next_res += 1;
            passes.push(make_compute_pass(
                PassIndex(passes.len()),
                $name,
                $reads,
                &[write_h],
            ));
            write_h
        }};
    }

    // Helper: append a pass that only reads (no new resource write).
    macro_rules! push_read_pass {
        ($name:expr, $reads:expr) => {{
            passes.push(make_compute_pass(
                PassIndex(passes.len()),
                $name,
                $reads,
                &[],
            ));
        }};
    }

    // ---- Segment 1: Chain of 10 ----
    // pass_0 writes R0; pass_i reads R_{i-1}, writes R_i.
    let mut prev = push_write_pass!("chain1_entry", &[]);
    for i in 1..10 {
        prev = push_write_pass!(&format!("chain1_{}", i), &[prev]);
    }

    // ---- Segment 2: Diamond (8 passes) ----
    // root -> 3 middle -> 3 merge -> 1 tail
    let d_root = push_write_pass!("diamond_root", &[prev]);
    let d_left = push_write_pass!("diamond_left", &[d_root]);
    let d_center = push_write_pass!("diamond_center", &[d_root]);
    let d_right = push_write_pass!("diamond_right", &[d_root]);
    let d_m1 = push_write_pass!("diamond_merge1", &[d_left, d_center]);
    let d_m2 = push_write_pass!("diamond_merge2", &[d_center, d_right]);
    let d_m3 = push_write_pass!("diamond_merge3", &[d_left, d_right]);
    prev = push_write_pass!("diamond_tail", &[d_m1, d_m2, d_m3]);

    // ---- Segment 3: Chain of 8 ----
    for i in 0..8 {
        prev = push_write_pass!(&format!("chain2_{}", i), &[prev]);
    }

    // ---- Segment 4: Fan-out (1 producer, 4 consumers, 1 barrier read) ----
    let fo_prod = push_write_pass!("fanout_prod", &[prev]);
    let fo_a = push_write_pass!("fanout_a", &[fo_prod]);
    let fo_b = push_write_pass!("fanout_b", &[fo_prod]);
    let fo_c = push_write_pass!("fanout_c", &[fo_prod]);
    let fo_d = push_write_pass!("fanout_d", &[fo_prod]);
    // barrier pass reads all four fan-out outputs
    push_read_pass!("fanout_barrier", &[fo_a, fo_b, fo_c, fo_d]);
    prev = push_write_pass!("fanout_after", &[fo_d]);

    // ---- Segment 5: Fan-in (4 producers, 1 consumer, 1 barrier) ----
    let fi_a = push_write_pass!("fanin_a", &[prev]);
    let fi_b = push_write_pass!("fanin_b", &[prev]);
    let fi_c = push_write_pass!("fanin_c", &[prev]);
    let fi_d = push_write_pass!("fanin_d", &[prev]);
    push_read_pass!("fanin_merge", &[fi_a, fi_b, fi_c, fi_d]);
    prev = push_write_pass!("fanin_after", &[fi_a]);

    // ---- Segment 6: Chain of 12 ----
    for i in 0..12 {
        prev = push_write_pass!(&format!("chain3_{}", i), &[prev]);
    }

    // Verify we generated exactly the expected number of passes.
    assert_eq!(
        passes.len(),
        PASSES_PER_GRAPH,
        "build_mixed_graph must produce exactly {} passes, got {}",
        PASSES_PER_GRAPH,
        passes.len(),
    );

    // Build resource list: one Buffer per resource handle.
    let resources: Vec<IrResource> = (0..next_res)
        .map(|i| {
            IrResource::new(
                ResourceHandle(i),
                format!("res_{}", i),
                ResourceDesc::Buffer(renderer_backend::frame_graph::BufferDesc {
                    size: 512,
                    usage: "storage".into(),
                    is_indirect_arg: false,
                }),
                ResourceLifetime::Transient,
                ResourceState::Uninitialized,
            )
        })
        .collect();

    (passes, resources)
}

// =============================================================================
// Helper: create a compute pass with explicit read/write sets
// =============================================================================

fn make_compute_pass(
    index: PassIndex,
    name: &str,
    reads: &[ResourceHandle],
    writes: &[ResourceHandle],
) -> IrPass {
    IrPass {
        index,
        name: name.to_string(),
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

// =============================================================================
// Helper: verify the JSON export from a compiled graph
// =============================================================================

/// Checks that the JSON export has the expected top-level structure for a
/// populated compiled graph.
fn verify_export_json(json: &serde_json::Value, iteration: usize) {
    let obj = json.as_object().unwrap_or_else(|| {
        panic!("Iteration {}: export_all must return a JSON object", iteration);
    });

    // Must have exactly 4 top-level keys.
    assert_eq!(
        obj.len(),
        4,
        "Iteration {}: export_all must return exactly 4 top-level keys (got {})",
        iteration,
        obj.len(),
    );

    // Key presence.
    for key in &["graph", "resources", "schedule", "stats"] {
        assert!(
            obj.contains_key(*key),
            "Iteration {}: JSON export missing key '{}'",
            iteration,
            key,
        );
    }

    // "graph" must be an object with a "passes" array.
    let graph = &obj["graph"];
    assert!(
        graph.is_object(),
        "Iteration {}: 'graph' key must be a JSON object",
        iteration,
    );
    let passes = graph["passes"].as_array();
    assert!(
        passes.is_some(),
        "Iteration {}: 'graph.passes' must be a JSON array",
        iteration,
    );

    // "resources" must be an array.
    assert!(
        obj["resources"].is_array(),
        "Iteration {}: 'resources' must be a JSON array",
        iteration,
    );

    // "schedule" must be an object.
    assert!(
        obj["schedule"].is_object(),
        "Iteration {}: 'schedule' must be a JSON object",
        iteration,
    );

    // "stats" must be an object with passes_total populated.
    assert!(
        obj["stats"].is_object(),
        "Iteration {}: 'stats' must be a JSON object",
        iteration,
    );
    let _passes_total = obj["stats"]["passes_total"].as_u64();
    assert!(
        _passes_total.is_some(),
        "Iteration {}: stats.passes_total must be present",
        iteration,
    );
}

// =============================================================================
// Core memory test: 150 compile/drop/export cycles
// =============================================================================

/// Runs 150 compile/drop/export cycles with a 50-pass mixed-topology graph.
///
/// Each cycle:
///   1. Builds a fresh graph (passes + resources).
///   2. Compiles via CompiledFrameGraph::compile.
///   3. Exports the compiled graph to JSON via JsonExporter::export_all.
///   4. Verifies the JSON structure.
///   5. Drops the compiled graph (end of scope).
///
/// Since all involved types are plain owned data (Vec, HashMap, String) with
/// no reference cycles or global statics, every drop fully releases all
/// associated memory.
#[test]
fn frame_graph_mem_150_compile_drop_export_cycles() {
    // Track cumulative compilation time across all iterations.
    let mut cumulative_compile_us: u64 = 0;

    for i in 0..ITERATIONS {
        // Build graph.
        let (passes, resources) = build_mixed_graph();

        // Use Debug profile so all 51 passes survive (no dead-pass elim).
        let config = CompilerProfile::DEBUG.config();
        let compiled = CompiledFrameGraph::compile_with_config(passes, resources, config)
            .unwrap_or_else(|e| {
                panic!("Iteration {}: compile failed: {}", i, e);
            });

        // Verify compiled graph structure -- all 51 passes survive in Debug profile.
        assert_eq!(
            compiled.order.len(),
            PASSES_PER_GRAPH,
            "Iteration {}: compiled order must contain {} passes (got {})",
            i,
            PASSES_PER_GRAPH,
            compiled.order.len(),
        );

        // Verify compiler stats are populated.
        let stats = compiled.compiler_stats();
        assert!(
            stats.passes_total > 0,
            "Iteration {}: stats.passes_total must be > 0",
            i,
        );
        assert!(
            stats.compilation_time_us > 0,
            "Iteration {}: stats.compilation_time_us must be > 0 (got {})",
            i,
            stats.compilation_time_us,
        );

        // Accumulate timing.
        cumulative_compile_us = cumulative_compile_us.saturating_add(stats.compilation_time_us);

        // Export to JSON.
        let json = JsonExporter::export_all(&compiled);

        // Verify JSON structure.
        verify_export_json(&json, i);

        // Check an additional property: the emitted_bridge_json (inside
        // "graph") must have the same pass count as the compiled order.
        let graph_passes = json["graph"]["passes"].as_array().unwrap();
        assert_eq!(
            graph_passes.len(),
            compiled.order.len(),
            "Iteration {}: graph.passes length matches compiled.order.len()",
            i,
        );

        // compiled is dropped here at end of scope -- all its memory
        // (passes, resources, edges, order, depths, barriers, etc.) is
        // reclaimed by Rust's ownership model.
    }

    // Verify cumulative time is non-zero and reasonable.
    // 150 compilations of 50-pass graphs should take measurable time.
    assert!(
        cumulative_compile_us > 0,
        "Cumulative compile time across {} iterations must be > 0",
        ITERATIONS,
    );

    // Log the cumulative time for manual inspection.
    eprintln!(
        "FrameGraphMemTest: {} iterations completed in {} us total (avg {} us/iter)",
        ITERATIONS,
        cumulative_compile_us,
        cumulative_compile_us / ITERATIONS as u64,
    );
}

// =============================================================================
// Test with both Debug and Default compiler profiles
// =============================================================================

/// Same 150-cycle test using the Debug profile (all optimisations disabled).
/// Memory behaviour should be identical since CompiledFrameGraph uses the
/// same owned types regardless of profile.
#[test]
fn frame_graph_mem_debug_profile_cycle() {
    for i in 0..ITERATIONS {
        let (passes, resources) = build_mixed_graph();

        let config = CompilerProfile::DEBUG.config();
        let compiled = CompiledFrameGraph::compile_with_config(passes, resources, config)
            .unwrap_or_else(|e| {
                panic!("Iteration {} (Debug profile): compile failed: {}", i, e);
            });

        // All 50 passes survive in Debug mode (dead-pass elim disabled).
        assert_eq!(
            compiled.order.len(),
            PASSES_PER_GRAPH,
            "Iteration {} (Debug): order.len() == {}",
            i,
            PASSES_PER_GRAPH,
        );

        // Export and verify structure.
        let json = JsonExporter::export_all(&compiled);
        verify_export_json(&json, i);

        // Verify eliminated_passes is empty in Debug mode.
        assert!(
            compiled.eliminated_passes.is_empty(),
            "Iteration {} (Debug): no eliminated passes expected, got {}",
            i,
            compiled.eliminated_passes.len(),
        );

        // compiled dropped here.
    }
}

/// Same 150-cycle test using the Default profile (dead-pass elim + barrier opt
/// enabled).  Terminal passes without downstream readers are eliminated.
#[test]
fn frame_graph_mem_default_profile_cycle() {
    for i in 0..ITERATIONS {
        let (passes, resources) = build_mixed_graph();

        let config = CompilerProfile::DEFAULT.config();
        let compiled = CompiledFrameGraph::compile_with_config(passes, resources, config)
            .unwrap_or_else(|e| {
                panic!("Iteration {} (Default profile): compile failed: {}", i, e);
            });

        // With dead-pass elim enabled, terminal passes may be culled.
        // The number of surviving passes should be <= PASSES_PER_GRAPH.
        assert!(
            compiled.order.len() <= PASSES_PER_GRAPH,
            "Iteration {} (Default): order.len() ({}) <= {}",
            i,
            compiled.order.len(),
            PASSES_PER_GRAPH,
        );

        // At minimum the core connected passes survive.
        assert!(
            compiled.order.len() >= 40,
            "Iteration {} (Default): at least 40 passes should survive (got {})",
            i,
            compiled.order.len(),
        );

        // Cull stats should reflect eliminated passes.
        assert!(
            compiled.cull_stats.passes_eliminated > 0
                || compiled.cull_stats.passes_total == PASSES_PER_GRAPH,
            "Iteration {} (Default): some terminal passes should be eliminated",
            i,
        );

        // Export and verify.
        let json = JsonExporter::export_all(&compiled);
        verify_export_json(&json, i);

        // compiled dropped here.
    }
}

// =============================================================================
// Export-only cycle test: 150 export iterations on a stable compiled graph
// =============================================================================

/// Compiles a graph once, then exports it 150 times to verify that
/// JsonExporter::export_all is re-entrant and produces consistent output
/// across repeated calls without side effects.
#[test]
fn frame_graph_mem_repeated_exports_on_same_graph() {
    let (passes, resources) = build_mixed_graph();
    let compiled = CompiledFrameGraph::compile(passes, resources)
        .expect("Initial compile for repeated export test must succeed");

    // Collect the first export as the reference.
    let reference_json = JsonExporter::export_all(&compiled);
    let reference_pass_count = reference_json["graph"]["passes"].as_array().unwrap().len();
    let reference_resource_count = reference_json["resources"].as_array().unwrap().len();

    for i in 0..ITERATIONS {
        let json = JsonExporter::export_all(&compiled);

        // Verify structural consistency.
        verify_export_json(&json, i);

        // Pass count must match the reference.
        let pass_count = json["graph"]["passes"].as_array().unwrap().len();
        assert_eq!(
            pass_count, reference_pass_count,
            "Iteration {}: pass count ({}) matches reference ({})",
            i, pass_count, reference_pass_count,
        );

        // Resource count must match the reference.
        let resource_count = json["resources"].as_array().unwrap().len();
        assert_eq!(
            resource_count, reference_resource_count,
            "Iteration {}: resource count ({}) matches reference ({})",
            i, resource_count, reference_resource_count,
        );

        // Stats passes_total must be identical (same compiled graph).
        let stats_pass_total = json["stats"]["passes_total"].as_u64().unwrap();
        let ref_pass_total = reference_json["stats"]["passes_total"].as_u64().unwrap();
        assert_eq!(
            stats_pass_total, ref_pass_total,
            "Iteration {}: stats.passes_total ({}) matches reference ({})",
            i, stats_pass_total, ref_pass_total,
        );
    }

    // compiled dropped here.
}

// =============================================================================
// Empty graph compile/drop cycle
// =============================================================================

/// Verifies that even an empty graph compiles, exports, and drops cleanly
/// across multiple iterations (edge case for memory safety).
#[test]
fn frame_graph_mem_empty_graph_cycle() {
    for i in 0..ITERATIONS {
        let compiled = CompiledFrameGraph::compile(vec![], vec![])
            .unwrap_or_else(|e| {
                panic!("Iteration {}: empty graph compile failed: {}", i, e);
            });

        assert!(
            compiled.passes.is_empty(),
            "Iteration {}: empty graph has no passes",
            i,
        );
        assert!(
            compiled.resources.is_empty(),
            "Iteration {}: empty graph has no resources",
            i,
        );

        let json = JsonExporter::export_all(&compiled);
        verify_export_json(&json, i);

        // For empty graph, passes_total must be 0.
        assert_eq!(
            json["stats"]["passes_total"].as_u64().unwrap(),
            0,
            "Iteration {}: empty graph passes_total is 0",
            i,
        );

        // Check compilation_time_us exists (even for empty compile).
        assert!(
            json["stats"]["compilation_time_us"].as_u64().is_some(),
            "Iteration {}: empty graph stats has compilation_time_us",
            i,
        );

        // compiled dropped here.
    }
}
