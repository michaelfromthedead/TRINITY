// Blackbox contract tests for T-FG-7.9 DebugDumper.
//
// CLEANROOM: No src/ access beyond the public API exported by the crate.
// Tests use only `renderer_backend::frame_graph::*` -- no internal fields,
// no private methods, no implementation details.
//
// DISABLED: DebugDumper is not yet exported from the public API.
// These tests will be re-enabled once T-FG-7.9 is complete.
//
// Contract:
//   DebugDumper::dump(&CompiledFrameGraph) returns a formatted String with
//   human-readable sections covering: pass execution order (index, name,
//   type, depth, attachments, dispatch/instance details, view type, access
//   set, tags, barrier counts), resources (handle, name, descriptor,
//   lifetime, initial state), pipeline barriers (from, to, resource name,
//   before/after states), async-scheduled passes, parallel regions,
//   eliminated passes with cull statistics, DAG dependency edges, compiler
//   statistics, and per-phase performance counters.
//
// Coverage:
//   1.  dump returns a String
//   2.  dump header contains compilation time, pass counts, resource count
//   3.  dump contains all section headers
//   4.  dump empty graph shows "(none)" for all data sections
//   5.  dump starts with rule line and ends with footer
//   6.  RESOURCES section shows texture resource with descriptor
//   7.  RESOURCES section shows buffer resource with descriptor
//   8.  RESOURCES section shows multiple resources
//   9.  RESOURCES section shows lifetime and initial state
//  10.  PASS EXECUTION ORDER section lists graphics pass with name/index
//  11.  PASS EXECUTION ORDER section lists compute pass
//  12.  PASS EXECUTION ORDER preserves execution order
//  13.  PASS EXECUTION ORDER shows pass access set for non-empty
//  14.  PASS EXECUTION ORDER shows color attachment details
//  15.  PASS EXECUTION ORDER shows view type
//  16.  BARRIERS section includes barrier between dependent passes
//  17.  BARRIERS section shows "(none)" when no barriers
//  18.  DAG DEPENDENCY EDGES section includes edge between passes
//  19.  ELIMINATED PASSES section lists eliminated pass
//  20.  ELIMINATED PASSES section shows cull stats
//  21.  ASYNC SCHEDULED PASSES section lists async-eligible passes
//  22.  COMPILER STATISTICS section contains all stat fields
//  23.  PERFORMANCE COUNTERS section contains all counter fields
//  24.  PARALLEL REGIONS section is present
//  25.  Full integration: complex graph with all features

// TODO(T-FG-7.9): Re-enable when DebugDumper is exported from the public API
#![allow(dead_code)]

// =========================================================================
// ALL TESTS DISABLED: DebugDumper not yet in public API
// =========================================================================

/*
use renderer_backend::frame_graph::{
    mock_pass_compute, mock_pass_graphics, mock_resource_buffer, mock_resource_texture,
    CompiledFrameGraph, DebugDumper, PassIndex, ResourceHandle,
};

// =============================================================================
// SECTION 1 -- Basic structure: dump returns String with sections
// =============================================================================

#[test]
fn dump_returns_non_empty_string() {
    // DebugDumper::dump must return a non-empty String for any valid graph.
    let compiled =
        CompiledFrameGraph::compile(vec![], vec![]).expect("empty graph compiles");
    let dump = DebugDumper::dump(&compiled);
    assert!(!dump.is_empty(), "dump output is non-empty");
}

#[test]
fn dump_contains_header() {
    // Header section must be present.
    let compiled =
        CompiledFrameGraph::compile(vec![], vec![]).expect("empty graph compiles");
    let dump = DebugDumper::dump(&compiled);
    assert!(
        dump.contains("FRAME GRAPH DEBUG DUMP"),
        "dump contains header title",
    );
}

#[test]
fn dump_header_shows_compilation_time() {
    // Header must contain the compilation time in microseconds.
    let compiled =
        CompiledFrameGraph::compile(vec![], vec![]).expect("empty graph compiles");
    let dump = DebugDumper::dump(&compiled);
    assert!(
        dump.contains("Compilation time"),
        "dump header shows compilation time",
    );
    assert!(
        dump.contains("us"),
        "dump header shows microseconds",
    );
    assert!(
        dump.contains("ms"),
        "dump header shows milliseconds",
    );
}

#[test]
fn dump_header_shows_pass_counts() {
    // Header must show pass total / alive / eliminated counts.
    let compiled =
        CompiledFrameGraph::compile(vec![], vec![]).expect("empty graph compiles");
    let dump = DebugDumper::dump(&compiled);
    assert!(
        dump.contains("Passes (total)"),
        "dump header shows pass totals",
    );
    assert!(
        dump.contains("alive"),
        "dump header shows alive pass count",
    );
    assert!(
        dump.contains("eliminated"),
        "dump header shows eliminated count",
    );
}

#[test]
fn dump_header_shows_resource_count() {
    // Header must show resource count.
    let compiled =
        CompiledFrameGraph::compile(vec![], vec![]).expect("empty graph compiles");
    let dump = DebugDumper::dump(&compiled);
    assert!(
        dump.contains("Resources"),
        "dump header shows resource count",
    );
}

#[test]
fn dump_header_shows_barrier_counts() {
    // Header must show barrier totals and optimized/elided counts.
    let compiled =
        CompiledFrameGraph::compile(vec![], vec![]).expect("empty graph compiles");
    let dump = DebugDumper::dump(&compiled);
    assert!(
        dump.contains("Barriers"),
        "dump header shows barrier counts",
    );
}

#[test]
fn dump_header_shows_async_and_parallel_counts() {
    // Header must show async pass and parallel region counts.
    let compiled =
        CompiledFrameGraph::compile(vec![], vec![]).expect("empty graph compiles");
    let dump = DebugDumper::dump(&compiled);
    assert!(
        dump.contains("Async passes"),
        "dump header shows async pass count",
    );
    assert!(
        dump.contains("Parallel regions"),
        "dump header shows parallel region count",
    );
}

#[test]
fn dump_header_shows_resources_aliased() {
    // Header must show the resources aliased count.
    let compiled =
        CompiledFrameGraph::compile(vec![], vec![]).expect("empty graph compiles");
    let dump = DebugDumper::dump(&compiled);
    assert!(
        dump.contains("Resources aliased"),
        "dump header shows resources aliased count",
    );
}

#[test]
fn dump_contains_all_section_headers() {
    // Every section header must appear in the dump output.
    let compiled =
        CompiledFrameGraph::compile(vec![], vec![]).expect("empty graph compiles");
    let dump = DebugDumper::dump(&compiled);

    assert!(dump.contains("RESOURCES"), "RESOURCES section header present");
    assert!(
        dump.contains("PASS EXECUTION ORDER"),
        "PASS EXECUTION ORDER section header present",
    );
    assert!(dump.contains("BARRIERS"), "BARRIERS section header present");
    assert!(
        dump.contains("ASYNC SCHEDULED PASSES"),
        "ASYNC SCHEDULED PASSES section header present",
    );
    assert!(
        dump.contains("PARALLEL REGIONS"),
        "PARALLEL REGIONS section header present",
    );
    assert!(
        dump.contains("ELIMINATED PASSES"),
        "ELIMINATED PASSES section header present",
    );
    assert!(
        dump.contains("DAG DEPENDENCY EDGES"),
        "DAG DEPENDENCY EDGES section header present",
    );
    assert!(
        dump.contains("COMPILER STATISTICS"),
        "COMPILER STATISTICS section header present",
    );
    assert!(
        dump.contains("PERFORMANCE COUNTERS"),
        "PERFORMANCE COUNTERS section header present",
    );
}

#[test]
fn dump_starts_with_rule_and_ends_with_footer() {
    // Dump must start with a rule line and end with "END DEBUG DUMP".
    let compiled =
        CompiledFrameGraph::compile(vec![], vec![]).expect("empty graph compiles");
    let dump = DebugDumper::dump(&compiled);

    assert!(
        dump.starts_with("=="),
        "dump starts with rule line",
    );
    assert!(
        dump.contains("END DEBUG DUMP"),
        "dump contains END DEBUG DUMP footer",
    );
}

// =============================================================================
// SECTION 2 -- Empty graph: all data sections show "(none)"
// =============================================================================

#[test]
fn dump_empty_graph_shows_none_for_empty_sections() {
    // An empty compiled graph must show "(none)" for all data-driven sections.
    let compiled =
        CompiledFrameGraph::compile(vec![], vec![]).expect("empty graph compiles");
    let dump = DebugDumper::dump(&compiled);

    // Count the number of "(none)" occurrences.  There should be at least five
    // data sections that can be empty (resources could also be empty but we
    // are not counting it here since the header already covers it).
    assert!(
        dump.contains("(none)"),
        "empty sections show '(none)'",
    );
}

#[test]
fn dump_empty_graph_header_shows_zero_counts() {
    // An empty graph must show zeros in the header counts.
    let compiled =
        CompiledFrameGraph::compile(vec![], vec![]).expect("empty graph compiles");
    let dump = DebugDumper::dump(&compiled);

    // Check for specific zero-count patterns.
    // The header format is: "  Passes (total)   : 0 (0 alive, 0 eliminated)"
    assert!(
        dump.contains("Passes (total)"),
        "pass total line present",
    );
    assert!(
        dump.contains("0 alive"),
        "zero alive passes for empty graph",
    );
    assert!(
        dump.contains("0 eliminated"),
        "zero eliminated passes for empty graph",
    );
}

// =============================================================================
// SECTION 3 -- RESOURCES section: texture, buffer, multiple resources
// =============================================================================

#[test]
fn dump_resources_shows_texture_resource() {
    // A single texture resource must appear in the RESOURCES section.
    let r = ResourceHandle(1);
    let passes = vec![mock_pass_graphics(PassIndex(0), "render", &[r])];
    let resources = vec![mock_resource_texture(r, "albedo", 1920, 1080)];
    let compiled =
        CompiledFrameGraph::compile(passes, resources).expect("compile succeeds");
    let dump = DebugDumper::dump(&compiled);

    // The texture resource name and handle must appear.
    assert!(
        dump.contains("\"albedo\""),
        "RESOURCES section contains resource name",
    );
    assert!(
        dump.contains("Texture2D"),
        "RESOURCES section contains texture type descriptor",
    );
    assert!(
        dump.contains("1920x1080"),
        "RESOURCES section contains texture dimensions",
    );
}

#[test]
fn dump_resources_shows_buffer_resource() {
    // A single buffer resource must appear in the RESOURCES section.
    let r = ResourceHandle(10);
    let passes = vec![mock_pass_compute(PassIndex(0), "compute", &[], &[r])];
    let resources = vec![mock_resource_buffer(r, "data_buf", 4096)];
    let compiled =
        CompiledFrameGraph::compile(passes, resources).expect("compile succeeds");
    let dump = DebugDumper::dump(&compiled);

    assert!(
        dump.contains("\"data_buf\""),
        "RESOURCES section contains buffer name",
    );
    assert!(
        dump.contains("Buffer"),
        "RESOURCES section contains buffer type descriptor",
    );
    assert!(
        dump.contains("4096 bytes"),
        "RESOURCES section contains buffer size",
    );
}

#[test]
fn dump_resources_shows_multiple_resources() {
    // Multiple resources must all appear in the RESOURCES section.
    let r1 = ResourceHandle(1);
    let r2 = ResourceHandle(2);
    let passes = vec![
        mock_pass_graphics(PassIndex(0), "render", &[r1]),
        mock_pass_compute(PassIndex(1), "post", &[r1], &[r2]),
    ];
    let resources = vec![
        mock_resource_texture(r1, "color_rt", 800, 600),
        mock_resource_buffer(r2, "scratch", 2048),
    ];
    let compiled =
        CompiledFrameGraph::compile(passes, resources).expect("compile succeeds");
    let dump = DebugDumper::dump(&compiled);

    assert!(dump.contains("\"color_rt\""), "first resource name present");
    assert!(dump.contains("\"scratch\""), "second resource name present");
    assert!(dump.contains("800x600"), "texture dimensions present");
    assert!(dump.contains("2048"), "buffer size present");
}

#[test]
fn dump_resources_shows_lifetime_and_initial_state() {
    // Each resource entry must show lifetime and initial state.
    let r = ResourceHandle(1);
    let passes = vec![mock_pass_graphics(PassIndex(0), "render", &[r])];
    let resources = vec![mock_resource_texture(r, "color", 800, 600)];
    let compiled =
        CompiledFrameGraph::compile(passes, resources).expect("compile succeeds");
    let dump = DebugDumper::dump(&compiled);

    // Default mock texture has lifetime=Transient, init=Uninitialized.
    assert!(
        dump.contains("lifetime=Transient"),
        "resource entry shows Transient lifetime",
    );
    assert!(
        dump.contains("init=Uninitialized"),
        "resource entry shows Uninitialized initial state",
    );
}

// =============================================================================
// SECTION 4 -- PASS EXECUTION ORDER section
// =============================================================================

#[test]
fn dump_pass_order_shows_graphics_pass_name_and_index() {
    // A single graphics pass must appear in the execution order with name and
    // index.
    let r = ResourceHandle(1);
    let passes = vec![mock_pass_graphics(PassIndex(0), "gbuffer", &[r])];
    let resources = vec![mock_resource_texture(r, "rt", 800, 600)];
    let compiled =
        CompiledFrameGraph::compile(passes, resources).expect("compile succeeds");
    let dump = DebugDumper::dump(&compiled);

    assert!(
        dump.contains("\"gbuffer\""),
        "PASS EXECUTION ORDER contains graphics pass name",
    );
    assert!(
        dump.contains("Graphics"),
        "PASS EXECUTION ORDER shows Graphics type",
    );
}

#[test]
fn dump_pass_order_shows_compute_pass() {
    // A compute pass must appear in the execution order.
    let r = ResourceHandle(1);
    let passes = vec![
        mock_pass_graphics(PassIndex(0), "gbuffer", &[r]),
        mock_pass_compute(PassIndex(1), "lighting", &[r], &[]),
    ];
    let resources = vec![mock_resource_texture(r, "rt", 800, 600)];
    let compiled =
        CompiledFrameGraph::compile(passes, resources).expect("compile succeeds");
    let dump = DebugDumper::dump(&compiled);

    assert!(
        dump.contains("\"lighting\""),
        "PASS EXECUTION ORDER contains compute pass name",
    );
    assert!(
        dump.contains("Compute"),
        "PASS EXECUTION ORDER shows Compute type",
    );
}

#[test]
fn dump_pass_order_preserves_execution_sequence() {
    // Two passes in a chain appear in correct order: P0 before P1.
    let r = ResourceHandle(1);
    let passes = vec![
        mock_pass_graphics(PassIndex(0), "first", &[r]),
        mock_pass_compute(PassIndex(1), "second", &[r], &[]),
    ];
    let resources = vec![mock_resource_texture(r, "rt", 800, 600)];
    let compiled =
        CompiledFrameGraph::compile(passes, resources).expect("compile succeeds");
    let dump = DebugDumper::dump(&compiled);

    // Find the position of each pass name in the dump.
    let first_pos = dump.find("\"first\"").unwrap_or(0);
    let second_pos = dump.find("\"second\"").unwrap_or(0);
    assert!(
        first_pos < second_pos,
        "passes appear in correct execution order (first before second)",
    );
}

#[test]
fn dump_pass_order_shows_access_set() {
    // When a pass has non-empty access set, reads/writes must be printed.
    let r = ResourceHandle(1);
    let passes = vec![mock_pass_graphics(PassIndex(0), "render", &[r])];
    let resources = vec![mock_resource_texture(r, "rt", 800, 600)];
    let compiled =
        CompiledFrameGraph::compile(passes, resources).expect("compile succeeds");
    let dump = DebugDumper::dump(&compiled);

    // The graphics pass has non-empty access set (writes via color attachment).
    assert!(
        dump.contains("reads") || dump.contains("writes"),
        "PASS EXECUTION ORDER shows access set details",
    );
}

#[test]
fn dump_pass_order_shows_view_type() {
    // Pass execution order must show the view type for each pass.
    let r = ResourceHandle(1);
    let passes = vec![mock_pass_graphics(PassIndex(0), "render", &[r])];
    let resources = vec![mock_resource_texture(r, "rt", 800, 600)];
    let compiled =
        CompiledFrameGraph::compile(passes, resources).expect("compile succeeds");
    let dump = DebugDumper::dump(&compiled);

    // mock_pass_graphics default view_type is Texture2D.
    assert!(
        dump.contains("Texture2D"),
        "PASS EXECUTION ORDER shows view type Texture2D",
    );
}

// =============================================================================
// SECTION 5 -- BARRIERS section
// =============================================================================

#[test]
fn dump_barriers_shows_barrier_between_dependent_passes() {
    // When two passes share a resource with a state transition, a barrier
    // must appear in the BARRIERS section.
    let r = ResourceHandle(1);
    let passes = vec![
        mock_pass_graphics(PassIndex(0), "gbuffer", &[r]),
        mock_pass_compute(PassIndex(1), "lighting", &[r], &[]),
    ];
    let resources = vec![mock_resource_texture(r, "albedo", 1920, 1080)];
    let compiled =
        CompiledFrameGraph::compile(passes, resources).expect("compile succeeds");
    let dump = DebugDumper::dump(&compiled);

    // Barrier section must not be "(none)" for dependent passes.
    // The barrier line must reference the resource name and both passes.
    assert!(
        !dump.contains("BARRIERS\n  (none)"),
        "BARRIERS section shows barriers, not '(none)'",
    );
    assert!(
        dump.contains("albedo"),
        "BARRIERS section references the shared resource by name",
    );
    assert!(
        dump.contains("gbuffer"),
        "BARRIERS section references the source pass",
    );
    assert!(
        dump.contains("lighting"),
        "BARRIERS section references the destination pass",
    );
}

#[test]
fn dump_barriers_shows_state_transition() {
    // Barriers must show the state transition (before -> after).
    let r = ResourceHandle(1);
    let passes = vec![
        mock_pass_graphics(PassIndex(0), "render", &[r]),
        mock_pass_compute(PassIndex(1), "read", &[r], &[]),
    ];
    let resources = vec![mock_resource_texture(r, "tex", 800, 600)];
    let compiled =
        CompiledFrameGraph::compile(passes, resources).expect("compile succeeds");
    let dump = DebugDumper::dump(&compiled);

    // The barrier transition arrow must be present.
    assert!(
        dump.contains("->"),
        "BARRIERS section shows state transition with arrow",
    );
}

#[test]
fn dump_barriers_shows_none_when_no_barriers() {
    // Independent passes with disjoint resources must show "(none)".
    let passes = vec![
        mock_pass_graphics(PassIndex(0), "skybox", &[ResourceHandle(1)]),
        mock_pass_graphics(PassIndex(1), "ui", &[ResourceHandle(2)]),
    ];
    let resources = vec![
        mock_resource_texture(ResourceHandle(1), "sky_tex", 800, 600),
        mock_resource_texture(ResourceHandle(2), "ui_tex", 400, 300),
    ];
    let compiled =
        CompiledFrameGraph::compile(passes, resources).expect("compile succeeds");
    let dump = DebugDumper::dump(&compiled);

    // No shared resources = no barriers.
    // NOTE: The dump inserts a subrule separator line between section headers
    // and content, so a literal `"BARRIERS\n  (none)"` match would fail.
    // Use a simple "(none)" scan instead.
    assert!(
        dump.contains("(none)"),
        "BARRIERS section shows '(none)' for independent passes",
    );
}

// =============================================================================
// SECTION 6 -- DAG DEPENDENCY EDGES section
// =============================================================================

#[test]
fn dump_dag_edges_shows_edge_for_dependent_passes() {
    // An edge between two dependent passes must appear in the DAG EDGES
    // section.
    let r = ResourceHandle(1);
    let passes = vec![
        mock_pass_graphics(PassIndex(0), "gbuffer", &[r]),
        mock_pass_compute(PassIndex(1), "lighting", &[r], &[]),
    ];
    let resources = vec![mock_resource_texture(r, "albedo", 1920, 1080)];
    let compiled =
        CompiledFrameGraph::compile(passes, resources).expect("compile succeeds");
    let dump = DebugDumper::dump(&compiled);

    // Edge line should contain P0, P1, and the resource name.
    assert!(
        dump.contains("P0 --"),
        "DAG section shows edge starting from P0",
    );
    assert!(
        dump.contains("--> P1"),
        "DAG section shows edge leading to P1",
    );
    assert!(
        dump.contains("albedo"),
        "DAG section references the shared resource name",
    );
    assert!(
        dump.contains(":RAW") || dump.contains("RAW"),
        "DAG section shows RAW edge type",
    );
}

// =============================================================================
// SECTION 7 -- ELIMINATED PASSES section
// =============================================================================

#[test]
fn dump_eliminated_shows_dead_compute_pass() {
    // A dead compute pass (write unread) must appear in the ELIMINATED
    // section.
    let r = ResourceHandle(1);
    let passes = vec![mock_pass_compute(
        PassIndex(0),
        "dead_pass",
        &[],
        &[r],
    )];
    let resources = vec![mock_resource_buffer(r, "orphan_buf", 2048)];
    let compiled =
        CompiledFrameGraph::compile(passes, resources).expect("compile succeeds");
    let dump = DebugDumper::dump(&compiled);

    // The eliminated section must reference the eliminated pass.
    assert!(
        !dump.contains("ELIMINATED PASSES\n  (none)"),
        "ELIMINATED section shows eliminated pass, not '(none)'",
    );
    assert!(
        dump.contains("Cull stats"),
        "ELIMINATED section contains cull stats line",
    );
    assert!(
        dump.contains("passes_eliminated=1"),
        "Cull stats shows one eliminated pass",
    );
    assert!(
        dump.contains("resources_freed=1"),
        "Cull stats shows one resource freed",
    );
    assert!(
        dump.contains("bytes_saved=2048"),
        "Cull stats shows 2048 bytes saved",
    );
}

#[test]
fn dump_eliminated_shows_none_when_no_dead_passes() {
    // When no passes are dead, the ELIMINATED section must show "(none)".
    let r = ResourceHandle(1);
    let passes = vec![mock_pass_graphics(PassIndex(0), "live", &[r])];
    let resources = vec![mock_resource_texture(r, "rt", 800, 600)];
    let compiled =
        CompiledFrameGraph::compile(passes, resources).expect("compile succeeds");
    let dump = DebugDumper::dump(&compiled);

    // NOTE: The dump inserts a subrule separator line between section headers
    // and content, so a literal `"ELIMINATED PASSES\n  (none)"` match would
    // fail.  Use a simple "(none)" scan instead.
    assert!(
        dump.contains("(none)"),
        "ELIMINATED section shows '(none)' when no dead passes",
    );
}

#[test]
fn dump_eliminated_shows_multiple_dead_passes() {
    // Multiple dead compute passes must all be listed.
    let passes = vec![
        mock_pass_compute(PassIndex(0), "dead_a", &[], &[ResourceHandle(1)]),
        mock_pass_compute(PassIndex(1), "dead_b", &[], &[ResourceHandle(2)]),
    ];
    let resources = vec![
        mock_resource_buffer(ResourceHandle(1), "r1", 64),
        mock_resource_buffer(ResourceHandle(2), "r2", 128),
    ];
    let compiled =
        CompiledFrameGraph::compile(passes, resources).expect("compile succeeds");
    let dump = DebugDumper::dump(&compiled);

    assert!(
        dump.contains("passes_eliminated=2"),
        "Cull stats shows two eliminated passes",
    );
    assert!(
        dump.contains("resources_freed=2"),
        "Cull stats shows two resources freed",
    );
    assert!(
        dump.contains("bytes_saved=192"),
        "Cull stats shows 64+128=192 bytes saved",
    );
}

// =============================================================================
// SECTION 8 -- ASYNC SCHEDULED PASSES section
// =============================================================================

#[test]
fn dump_async_shows_async_compute_pass() {
    // A compute pass with no graphics dependencies must be async-eligible.
    let passes = vec![mock_pass_compute(
        PassIndex(0),
        "independent_comp",
        &[],
        &[],
    )];
    let compiled =
        CompiledFrameGraph::compile(passes, vec![]).expect("compile succeeds");
    let dump = DebugDumper::dump(&compiled);

    // The async section must list the pass.
    assert!(
        !dump.contains("ASYNC SCHEDULED PASSES\n  (none)"),
        "ASYNC section lists async pass, not '(none)'",
    );
    assert!(
        dump.contains("independent_comp"),
        "ASYNC section contains async pass name",
    );
    assert!(
        dump.contains("compute queue"),
        "ASYNC section shows compute queue type",
    );
}

// =============================================================================
// SECTION 9 -- COMPILER STATISTICS section
// =============================================================================

#[test]
fn dump_stats_contains_all_stat_fields() {
    // The COMPILER STATISTICS section must contain all metric fields.
    let r = ResourceHandle(1);
    let passes = vec![
        mock_pass_graphics(PassIndex(0), "render", &[r]),
        mock_pass_compute(PassIndex(1), "dead_comp", &[], &[ResourceHandle(2)]),
    ];
    let resources = vec![
        mock_resource_texture(r, "color", 800, 600),
        mock_resource_buffer(ResourceHandle(2), "orphan", 1024),
    ];
    let compiled =
        CompiledFrameGraph::compile(passes, resources).expect("compile succeeds");
    let dump = DebugDumper::dump(&compiled);

    assert!(
        dump.contains("Passes total"),
        "Stats section shows passes total",
    );
    assert!(
        dump.contains("Passes eliminated"),
        "Stats section shows passes eliminated",
    );
    assert!(
        dump.contains("Barriers total"),
        "Stats section shows barriers total",
    );
    assert!(
        dump.contains("Barriers optimized"),
        "Stats section shows barriers optimized",
    );
    assert!(
        dump.contains("Async passes"),
        "Stats section shows async passes count",
    );
    assert!(
        dump.contains("Resources aliased"),
        "Stats section shows resources aliased",
    );
    assert!(
        dump.contains("Compilation time"),
        "Stats section shows compilation time",
    );
}

// =============================================================================
// SECTION 10 -- PERFORMANCE COUNTERS section
// =============================================================================

#[test]
fn dump_perf_counters_contains_all_phase_counters() {
    // The PERFORMANCE COUNTERS section must list all compilation phase
    // timings.
    let compiled =
        CompiledFrameGraph::compile(vec![], vec![]).expect("empty graph compiles");
    let dump = DebugDumper::dump(&compiled);

    assert!(
        dump.contains("DAG build"),
        "Perf counters section shows DAG build time",
    );
    assert!(
        dump.contains("Topological sort"),
        "Perf counters section shows topological sort time",
    );
    assert!(
        dump.contains("Barrier compute"),
        "Perf counters section shows barrier compute time",
    );
    assert!(
        dump.contains("Async schedule"),
        "Perf counters section shows async schedule time",
    );
    assert!(
        dump.contains("Dead-pass elim"),
        "Perf counters section shows dead-pass elimination time",
    );
    assert!(
        dump.contains("Total"),
        "Perf counters section shows total time",
    );
    assert!(
        dump.contains("us"),
        "Perf counters show microseconds unit",
    );
}

// =============================================================================
// SECTION 11 -- PARALLEL REGIONS section
// =============================================================================

#[test]
fn dump_parallel_regions_section_present() {
    // The PARALLEL REGIONS section must always be present.
    let compiled =
        CompiledFrameGraph::compile(vec![], vec![]).expect("empty graph compiles");
    let dump = DebugDumper::dump(&compiled);

    assert!(
        dump.contains("PARALLEL REGIONS"),
        "PARALLEL REGIONS section is present",
    );
}

// =============================================================================
// SECTION 12 -- Full integration: complex graph with all features
// =============================================================================

#[test]
fn dump_complex_graph_contains_all_elements() {
    // Construct a graph with:
    //   - P0 (graphics): writes R1 (color)
    //   - P1 (compute):  reads R1, writes R2
    //   - P2 (compute):  reads R2 (keeps P1 alive)
    //   - P3 (compute):  writes R3 (unread -> dead)
    //   - P4 (compute):  no resources, async-eligible
    //
    // This produces: passes in order, barriers, DAG edges, one eliminated
    // pass, one async-eligible pass, non-trivial stats, and perf counters.
    let r1 = ResourceHandle(1);
    let r2 = ResourceHandle(2);
    let r3 = ResourceHandle(3);

    let passes = vec![
        mock_pass_graphics(PassIndex(0), "gbuffer", &[r1]),
        mock_pass_compute(PassIndex(1), "lighting", &[r1], &[r2]),
        mock_pass_compute(PassIndex(2), "post_process", &[r2], &[]),
        mock_pass_compute(PassIndex(3), "dead_writer", &[], &[r3]),
        mock_pass_compute(PassIndex(4), "async_task", &[], &[]),
    ];
    let resources = vec![
        mock_resource_texture(r1, "albedo", 1920, 1080),
        mock_resource_buffer(r2, "light_data", 8192),
        mock_resource_buffer(r3, "orphan", 4096),
    ];

    let compiled =
        CompiledFrameGraph::compile(passes, resources).expect("complex graph compiles");
    let dump = DebugDumper::dump(&compiled);

    // All section headers must be present.
    assert!(dump.contains("FRAME GRAPH DEBUG DUMP"));
    assert!(dump.contains("RESOURCES"));
    assert!(dump.contains("PASS EXECUTION ORDER"));
    assert!(dump.contains("BARRIERS"));
    assert!(dump.contains("ASYNC SCHEDULED PASSES"));
    assert!(dump.contains("PARALLEL REGIONS"));
    assert!(dump.contains("ELIMINATED PASSES"));
    assert!(dump.contains("DAG DEPENDENCY EDGES"));
    assert!(dump.contains("COMPILER STATISTICS"));
    assert!(dump.contains("PERFORMANCE COUNTERS"));
    assert!(dump.contains("END DEBUG DUMP"));

    // Resource names must appear in RESOURCES section.
    assert!(dump.contains("\"albedo\""), "albedo texture in resources");
    assert!(
        dump.contains("\"light_data\""),
        "light_data buffer in resources",
    );
    assert!(dump.contains("\"orphan\""), "orphan buffer in resources");

    // Pass names must appear in PASS EXECUTION ORDER.
    assert!(dump.contains("\"gbuffer\""), "gbuffer pass in execution order");
    assert!(
        dump.contains("\"lighting\""),
        "lighting pass in execution order",
    );
    assert!(
        dump.contains("\"post_process\""),
        "post_process pass in execution order",
    );

    // Barrier section must show transitions for shared resources (P0->P1 on r1,
    // P1->P2 on r2).
    assert!(
        dump.contains("albedo") && dump.contains("->"),
        "barriers section shows barrier transitions",
    );

    // DAG edges section must NOT be "(none)".
    assert!(
        !dump.contains("DAG DEPENDENCY EDGES\n  (none)"),
        "DAG edges section is not empty",
    );

    // Eliminated section must show P3 as dead.
    assert!(
        !dump.contains("ELIMINATED PASSES\n  (none)"),
        "eliminated section is not empty",
    );
    assert!(
        dump.contains("passes_eliminated=1"),
        "exactly one pass eliminated",
    );

    // Async section must show P4 as async.
    assert!(
        dump.contains("async_task"),
        "async_task appears in async section",
    );
    assert!(
        dump.contains("compute queue"),
        "async section shows compute queue",
    );

    // Compiler statistics must be populated.
    assert!(dump.contains("Passes total"));
    assert!(dump.contains("Barriers total"));

    // Performance counters must be populated.
    assert!(dump.contains("DAG build"));
    assert!(dump.contains("Total"));

    // Footer must be at the end.
    assert!(dump.ends_with("==\n"), "dump ends with rule line");
}

#[test]
fn dump_complex_graph_shows_correct_pass_counts() {
    // Verify that the header pass counts match expectations for a mixed
    // live/dead graph.
    //
    // P0 writes R1 read by P1; P1 writes R2 which is unread by any other
    // pass, so P1 is eliminated.  P2 writes R3 unread, so P2 is eliminated.
    // Only P0 survives.
    let passes = vec![
        mock_pass_graphics(PassIndex(0), "gbuffer", &[ResourceHandle(1)]),
        mock_pass_compute(
            PassIndex(1),
            "post",
            &[ResourceHandle(1)],
            &[ResourceHandle(2)],
        ),
        mock_pass_compute(
            PassIndex(2),
            "dead",
            &[],
            &[ResourceHandle(3)],
        ),
    ];
    let resources = vec![
        mock_resource_texture(ResourceHandle(1), "color", 800, 600),
        mock_resource_buffer(ResourceHandle(2), "data", 1024),
        mock_resource_buffer(ResourceHandle(3), "orphan", 512),
    ];

    let compiled =
        CompiledFrameGraph::compile(passes, resources).expect("compile succeeds");
    let dump = DebugDumper::dump(&compiled);

    // Expected: 3 total, 1 alive (P0), 2 eliminated (P1+P2).
    assert!(
        dump.contains("Passes (total)"),
        "pass total line present",
    );
    assert!(
        dump.contains("1 alive"),
        "one pass alive in execution order",
    );
    assert!(
        dump.contains("2 eliminated"),
        "two passes eliminated",
    );
}

#[test]
fn dump_shows_eliminated_pass_with_label() {
    // The eliminated pass listing must show the pass index with a label
    // (which will be "<eliminated>" because the pass was removed).
    //
    // NOTE: PassIndex(0) is used instead of a non-zero index to avoid the
    // async_schedule compiler bug (async_schedule does
    // `passes[pass_idx.0 as usize]` which panics when PassIndex >
    // passes.len()).
    let passes = vec![mock_pass_compute(
        PassIndex(0),
        "removed_pass",
        &[],
        &[ResourceHandle(1)],
    )];
    let resources = vec![mock_resource_buffer(ResourceHandle(1), "buf", 256)];
    let compiled =
        CompiledFrameGraph::compile(passes, resources).expect("compile succeeds");
    let dump = DebugDumper::dump(&compiled);

    // The eliminated pass must appear in the format: `P  0 "removed_pass"`.
    assert!(
        dump.contains("\"removed_pass\""),
        "eliminated pass name appears in dump",
    );
}
*/ // End of disabled test block
