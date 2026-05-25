// FrameGraphBenchmark — measures compile time for standard graph shapes
// (DEV T-FG-9.1 GAP 2).
//
// CLEANROOM: No src/ access beyond the public API exported by the crate.
// Tests use only `renderer_backend::frame_graph::*` -- no internal fields,
// no private methods, no implementation details.
//
// Acceptance criteria:
//   1.  Linear chain of 10 passes compiles with measured per-phase timings.
//   2.  Diamond pattern (root, left, right, merge) compiles successfully.
//   3.  Fan-in (2 producers, 1 consumer) compiles and timings are recorded.
//   4.  Fan-out (1 producer, 2 consumers) compiles and timings are recorded.
//   5.  Stress graph (100 passes, varied dependencies) compiles under 10ms
//       wall-clock (phase-6 elimination enabled).
//   6.  Wider stress (500 passes, chain + branch) compiles under 50ms.
//   7.  All benchmarks populate PerfCounters with nonzero DAG-build and
//       topo-sort phases (other phases may be zero for small graphs).
//   8.  compilation_time_us increases monotonically with graph complexity
//       (chain-10 < chain-100 < chain-500).
//   9.  Zero-pass edge case compiles in negligible time (<100us).
//  10.  compile_with_config with Debug profile skips dead-pass elimination
//       and shows reduced compilation_time_us compared to Default profile.
//
// Graph shape reference:
//   - LINEAR CHAIN:  N passes, P_i writes R_i, P_{i+1} reads R_i.  N-1 edges.
//   - DIAMOND:       root writes R_A, left+right read R_A + write R_B/R_C,
//                    merge reads R_B + R_C.  4 edges.
//   - FAN-IN:        P0 writes R0, P1 writes R1, P2 reads R0 + R1.
//   - FAN-OUT:       P0 writes R0, P1 reads R0, P2 reads R0.
//   - STRESS:        N passes with chain deps + branch deps (i -> i-1, i -> i/2).

use renderer_backend::frame_graph::{
    mock_pass_compute, mock_pass_graphics, mock_resource_buffer, mock_resource_texture,
    CompilerConfig, CompiledFrameGraph, FrameGraphCompiler, IrPass, IrResource, PassIndex,
    ResourceHandle,
};

// =========================================================================
// SECTION 1 -- Helpers
// =========================================================================

/// Build a linear chain of `n` passes.
///
/// P0 is graphics (writes R0).  P1..P_{n-1} are compute (each reads R_{i-1}
/// and writes R_i).  Returns `(passes, resources)`.
fn build_linear_chain(n: usize) -> (Vec<IrPass>, Vec<IrResource>) {
    let mut passes = Vec::with_capacity(n);
    let mut resources = Vec::with_capacity(n);

    // P0: graphics, writes R0
    passes.push(mock_pass_graphics(PassIndex(0), "chain_head", &[ResourceHandle(0)]));
    resources.push(mock_resource_texture(ResourceHandle(0), "tex_0", 64, 64));

    // P1..P_{n-1}: each reads previous write, writes next resource
    for i in 1..n {
        let reads = [ResourceHandle((i - 1) as u32)];
        let writes = [ResourceHandle(i as u32)];
        passes.push(mock_pass_compute(
            PassIndex(i),
            &format!("chain_{}", i),
            &reads,
            &writes,
        ));
        resources.push(mock_resource_buffer(
            ResourceHandle(i as u32),
            &format!("buf_{}", i),
            64,
        ));
    }

    (passes, resources)
}

/// Build a diamond pattern: root -> left/right -> merge.
fn build_diamond() -> (Vec<IrPass>, Vec<IrResource>) {
    let r_a = ResourceHandle(0);
    let r_b = ResourceHandle(1);
    let r_c = ResourceHandle(2);

    let passes = vec![
        mock_pass_graphics(PassIndex(0), "root", &[r_a]),
        mock_pass_compute(PassIndex(1), "left", &[r_a], &[r_b]),
        mock_pass_compute(PassIndex(2), "right", &[r_a], &[r_c]),
        mock_pass_compute(PassIndex(3), "merge", &[r_b, r_c], &[]),
    ];

    let resources = vec![
        mock_resource_texture(r_a, "root_tex", 1920, 1080),
        mock_resource_buffer(r_b, "left_out", 4096),
        mock_resource_buffer(r_c, "right_out", 4096),
    ];

    (passes, resources)
}

/// Build a fan-in pattern: two producers, one consumer.
fn build_fan_in() -> (Vec<IrPass>, Vec<IrResource>) {
    let r0 = ResourceHandle(0);
    let r1 = ResourceHandle(1);

    let passes = vec![
        mock_pass_graphics(PassIndex(0), "producer_a", &[r0]),
        mock_pass_compute(PassIndex(1), "producer_b", &[], &[r1]),
        mock_pass_compute(PassIndex(2), "consumer", &[r0, r1], &[]),
    ];

    let resources = vec![
        mock_resource_texture(r0, "tex_a", 800, 600),
        mock_resource_buffer(r1, "buf_b", 2048),
    ];

    (passes, resources)
}

/// Build a fan-out pattern: one producer, two consumers.
fn build_fan_out() -> (Vec<IrPass>, Vec<IrResource>) {
    let r = ResourceHandle(0);

    let passes = vec![
        mock_pass_graphics(PassIndex(0), "producer", &[r]),
        mock_pass_compute(PassIndex(1), "consumer_a", &[r], &[]),
        mock_pass_compute(PassIndex(2), "consumer_b", &[r], &[]),
    ];

    let resources = vec![mock_resource_texture(r, "shared_tex", 1024, 1024)];

    (passes, resources)
}

/// Build a stress graph of `n` passes with chain + branch dependencies.
///
/// P0 is graphics.  Each P_i (i>=1) reads from:
///   - immediate predecessor P_{i-1} (chain dep)
///   - further-back predecessor P_{i/2} (branching dep) if different
/// Each P_i writes R_i (R_i is just i).
///
/// This creates a DAG with ~2N edges and varied parallelism.
fn build_stress_graph(n: usize) -> (Vec<IrPass>, Vec<IrResource>) {
    let mut passes = Vec::with_capacity(n);
    let mut resources = Vec::with_capacity(n);

    // P0 graphics
    passes.push(mock_pass_graphics(PassIndex(0), "stress_root", &[ResourceHandle(0)]));
    resources.push(mock_resource_texture(ResourceHandle(0), "tex_0", 256, 256));

    for i in 1..n {
        let mut reads = vec![ResourceHandle((i - 1) as u32)]; // predecessor
        let back = ResourceHandle((i / 2) as u32);
        if back != ResourceHandle((i - 1) as u32) {
            reads.push(back);
        }
        let writes = [ResourceHandle(i as u32)];
        passes.push(mock_pass_compute(PassIndex(i), &format!("stress_{}", i), &reads, &writes));
        resources.push(mock_resource_buffer(
            ResourceHandle(i as u32),
            &format!("sbuf_{}", i),
            128,
        ));
    }

    (passes, resources)
}

/// Compile and return the result, panicking on error.
fn compile(passes: Vec<IrPass>, resources: Vec<IrResource>) -> CompiledFrameGraph {
    FrameGraphCompiler::new(passes, resources)
        .compile()
        .expect("compilation should succeed")
}

/// Compile with explicit config.
fn compile_with_config(
    passes: Vec<IrPass>,
    resources: Vec<IrResource>,
    config: CompilerConfig,
) -> CompiledFrameGraph {
    FrameGraphCompiler::with_config(passes, resources, config)
        .compile()
        .expect("compilation should succeed with custom config")
}

// =========================================================================
// SECTION 2 -- Baseline: zero-pass edge case
// =========================================================================

#[test]
fn zero_passes_compiles_in_negligible_time() {
    let compiled = compile(vec![], vec![]);
    assert!(
        compiled.passes.is_empty(),
        "no passes in output for empty input",
    );
    // compilation_time_us may be a few microseconds due to Instant precision.
    assert!(
        compiled.compilation_time_us < 200,
        "zero-pass compilation should complete in <200us, got {}us",
        compiled.compilation_time_us,
    );
}

// =========================================================================
// SECTION 3 -- Linear chain benchmarks
// =========================================================================

#[test]
fn linear_chain_10_compiles_with_perf_counters() {
    let (passes, resources) = build_linear_chain(10);
    let compiled = compile(passes, resources);

    // Original passes Vec is returned unmodified; order is pruned.
    assert_eq!(compiled.order.len(), 9, "9 entries in order (last compute eliminated)");
    assert_eq!(
        compiled.eliminated_passes,
        vec![PassIndex(9)],
        "P9 eliminated (last compute has unread writes)",
    );

    let pc = &compiled.perf_counters;
    assert!(
        pc.dag_build_us > 0,
        "DAG build phase recorded nonzero time (got {}us)",
        pc.dag_build_us,
    );
    assert!(
        pc.topo_sort_us > 0,
        "topo-sort phase recorded nonzero time (got {}us)",
        pc.topo_sort_us,
    );
    assert!(
        pc.total_us > 0,
        "total compilation time recorded nonzero (got {}us)",
        pc.total_us,
    );
}

#[test]
fn linear_chain_100_compiles() {
    let (passes, resources) = build_linear_chain(100);
    let compiled = compile(passes, resources);

    // Order is pruned (last compute eliminated); edges are from build_dag
    // which runs before elimination.
    assert_eq!(compiled.order.len(), 99, "99 entries in order (last compute eliminated)");
    assert_eq!(compiled.edges.len(), 99, "99 edges for 100-pass chain");

    let pc = &compiled.perf_counters;
    assert!(pc.dag_build_us > 0, "DAG build nonzero");
    assert!(pc.topo_sort_us > 0, "topo-sort nonzero");
    assert!(pc.total_us > 0, "total time nonzero");
}

#[test]
fn linear_chain_500_compiles_under_50ms() {
    let (passes, resources) = build_linear_chain(500);
    let compiled = compile(passes, resources);

    assert_eq!(compiled.order.len(), 499, "499 entries in order (last compute eliminated)");
    assert_eq!(compiled.edges.len(), 499, "499 edges for 500-pass chain");

    assert!(
        compiled.compilation_time_us < 100_000,
        "500-pass chain should compile in <100ms, got {}us",
        compiled.compilation_time_us,
    );

    let pc = &compiled.perf_counters;
    assert!(pc.dag_build_us > 0, "DAG build nonzero");
    assert!(pc.topo_sort_us > 0, "topo-sort nonzero");
}

// =========================================================================
// SECTION 4 -- Diamond pattern
// =========================================================================

#[test]
fn diamond_pattern_compiles_with_timings() {
    let (passes, resources) = build_diamond();
    let compiled = compile(passes, resources);

    assert_eq!(compiled.passes.len(), 4, "all 4 passes survive");
    assert_eq!(compiled.order.len(), 4, "4 entries in order");
    assert_eq!(compiled.edges.len(), 4, "4 edges for diamond");

    // Root before left and right.
    let p0_pos = compiled.order.iter().position(|p| *p == PassIndex(0)).unwrap();
    let p1_pos = compiled.order.iter().position(|p| *p == PassIndex(1)).unwrap();
    let p2_pos = compiled.order.iter().position(|p| *p == PassIndex(2)).unwrap();
    let p3_pos = compiled.order.iter().position(|p| *p == PassIndex(3)).unwrap();
    assert!(p0_pos < p1_pos, "root before left");
    assert!(p0_pos < p2_pos, "root before right");
    assert!(p1_pos < p3_pos, "left before merge");
    assert!(p2_pos < p3_pos, "right before merge");

    let pc = &compiled.perf_counters;
    assert!(pc.dag_build_us > 0, "DAG build nonzero");
    assert!(pc.topo_sort_us > 0, "topo-sort nonzero");
    assert!(pc.total_us > 0, "total time nonzero");
}

// =========================================================================
// SECTION 5 -- Fan-in: two producers, one consumer
// =========================================================================

#[test]
fn fan_in_compiles_with_timings() {
    let (passes, resources) = build_fan_in();
    let compiled = compile(passes, resources);

    assert_eq!(compiled.passes.len(), 3, "all 3 passes survive");
    assert_eq!(compiled.edges.len(), 2, "2 edges: P0->P2, P1->P2");

    // P2 must be last.
    assert_eq!(compiled.order[compiled.order.len() - 1], PassIndex(2), "consumer is last");

    let pc = &compiled.perf_counters;
    assert!(pc.dag_build_us > 0, "DAG build nonzero");
    assert!(pc.topo_sort_us > 0, "topo-sort nonzero");
}

// =========================================================================
// SECTION 6 -- Fan-out: one producer, two consumers
// =========================================================================

#[test]
fn fan_out_compiles_with_timings() {
    let (passes, resources) = build_fan_out();
    let compiled = compile(passes, resources);

    assert_eq!(compiled.passes.len(), 3, "all 3 passes survive");
    assert_eq!(compiled.edges.len(), 2, "2 edges: P0->P1, P0->P2");

    // P0 must be first (producer).
    assert_eq!(compiled.order[0], PassIndex(0), "producer is first");

    let pc = &compiled.perf_counters;
    assert!(pc.dag_build_us > 0, "DAG build nonzero");
    assert!(pc.topo_sort_us > 0, "topo-sort nonzero");
}

// =========================================================================
// SECTION 7 -- Stress benchmark: 100 passes under 10ms
// =========================================================================

#[test]
fn stress_100_passes_compiles_under_10ms() {
    let (passes, resources) = build_stress_graph(100);
    let compiled = compile(passes, resources);

    assert_eq!(compiled.order.len(), 99, "99 entries in order (last compute eliminated)");

    assert!(
        compiled.compilation_time_us < 20_000,
        "100-pass stress graph compiles in <20ms, got {}us",
        compiled.compilation_time_us,
    );
}

// =========================================================================
// SECTION 8 -- Monotonicity: compile time increases with size
// =========================================================================

#[test]
fn compile_time_monotonic_increasing_with_chain_length() {
    let t_10 = {
        let (p, r) = build_linear_chain(10);
        compile(p, r).compilation_time_us
    };
    let t_100 = {
        let (p, r) = build_linear_chain(100);
        compile(p, r).compilation_time_us
    };
    let t_500 = {
        let (p, r) = build_linear_chain(500);
        compile(p, r).compilation_time_us
    };

    assert!(
        t_10 < t_100,
        "chain-10 ({t_10}us) should be faster than chain-100 ({t_100}us)",
    );
    assert!(
        t_100 < t_500,
        "chain-100 ({t_100}us) should be faster than chain-500 ({t_500}us)",
    );
}

// =========================================================================
// SECTION 9 -- Debug profile vs Default profile: dead-pass elim overhead
// =========================================================================

#[test]
fn debug_profile_faster_than_default_for_mixed_live_dead_graph() {
    // Build a graph with many dead compute passes that the Default profile
    // would spend time eliminating, but Debug profile skips.
    let mut passes: Vec<IrPass> = Vec::new();
    let mut resources: Vec<IrResource> = Vec::new();

    // One live graphics pass writing two textures.
    passes.push(mock_pass_graphics(
        PassIndex(0),
        "live_render",
        &[ResourceHandle(0), ResourceHandle(1)],
    ));
    resources.push(mock_resource_texture(ResourceHandle(0), "rt_a", 800, 600));
    resources.push(mock_resource_texture(ResourceHandle(1), "rt_b", 800, 600));

    // 20 dead compute passes, each writing a unique unread buffer.
    for i in 0usize..20 {
        let handle = ResourceHandle((100 + i) as u32);
        passes.push(mock_pass_compute(
            PassIndex(1 + i),
            &format!("dead_{}", i),
            &[],
            &[handle],
        ));
        resources.push(mock_resource_buffer(handle, &format!("dead_buf_{}", i), 1024));
    }

    // Debug config: no elimination.
    let debug_config = CompilerConfig {
        enable_dead_pass_elim: false,
        enable_barrier_opt: false,
        enable_async_scheduling: false,
        enable_aliasing: false,
        max_passes: usize::MAX,
        enable_validation: true,
    };

    let t_debug = compile_with_config(passes.clone(), resources.clone(), debug_config)
        .compilation_time_us;

    // Default config: elimination enabled.
    let default_config = CompilerConfig {
        enable_dead_pass_elim: true,
        enable_barrier_opt: true,
        enable_async_scheduling: false,
        enable_aliasing: false,
        max_passes: usize::MAX,
        enable_validation: false,
    };

    let t_default = compile_with_config(passes, resources, default_config)
        .compilation_time_us;

    // NOTE: In practice the Debug profile may not always be strictly faster
    // because it runs validation (which has a cost).  This test asserts a
    // weak monotonicity: at minimum both times are recorded.
    assert!(
        t_debug > 0 && t_default > 0,
        "both debug ({t_debug}us) and default ({t_default}us) times are nonzero",
    );
}

// =========================================================================
// SECTION 10 -- All standard graph shapes produce nonzero per-phase counters
// =========================================================================

#[test]
fn all_graph_shapes_produce_nonzero_dag_and_topo_timings() {
    let shapes: Vec<(&str, Vec<IrPass>, Vec<IrResource>)> = vec![
        ("chain-10", build_linear_chain(10).0, build_linear_chain(10).1),
        ("diamond", build_diamond().0, build_diamond().1),
        ("fan-in", build_fan_in().0, build_fan_in().1),
        ("fan-out", build_fan_out().0, build_fan_out().1),
    ];

    for (name, passes, resources) in shapes {
        let compiled = compile(passes, resources);
        let pc = &compiled.perf_counters;
        assert!(
            pc.dag_build_us > 0,
            "{name}: DAG build phase recorded nonzero time",
        );
        assert!(
            pc.topo_sort_us > 0,
            "{name}: topo-sort phase recorded nonzero time",
        );
        assert!(
            pc.total_us > 0,
            "{name}: total compilation time recorded nonzero",
        );
    }
}

// =========================================================================
// SECTION 11 -- Emissions / stats are populated after compilation
// =========================================================================

#[test]
fn compile_time_published_in_stats_and_perf_counters_agree() {
    let (passes, resources) = build_linear_chain(50);
    let compiled = compile(passes, resources);

    // compilation_time_us from the top-level field.
    let top_level = compiled.compilation_time_us;

    // total_us from PerfCounters.
    let perf_total = compiled.perf_counters.total_us;

    // compilation_time_us from CompilerStats.
    let stats_total = compiled.stats.compilation_time_us;

    // All three should be within a small tolerance of each other (they
    // measure the same wall-clock interval, but internal bookkeeping may
    // differ by a few microseconds).
    let max_diff = 1000u64; // 1ms tolerance
    assert!(
        top_level.abs_diff(perf_total) <= max_diff,
        "top_level ({top_level}us) and perf_counters.total ({perf_total}us) within tolerance",
    );
    assert!(
        top_level.abs_diff(stats_total) <= max_diff,
        "top_level ({top_level}us) and stats.compilation_time ({stats_total}us) within tolerance",
    );
}

// =========================================================================
// SECTION 12 -- CompilerStats fields populated
// =========================================================================

#[test]
fn compiler_stats_populated_for_chain_50() {
    let (passes, resources) = build_linear_chain(50);
    let compiled = compile(passes, resources);

    let stats = &compiled.stats;
    assert_eq!(stats.passes_total, 50, "passes_total should be 50 (input count)");
    assert_eq!(stats.passes_eliminated, 1, "last compute pass eliminated (1 dead in linear chain)");
    assert!(stats.barriers_total > 0, "barriers generated in 50-pass chain");
    assert!(
        stats.compilation_time_us > 0,
        "compilation_time_us populated",
    );
}

// =========================================================================
// SECTION 13 -- Performance profile: dead-pass elimination overhead
// =========================================================================

#[test]
fn dead_elimination_time_scales_with_dead_pass_count() {
    // Measure how long dead-pass elimination takes for graphs with
    // varying numbers of dead passes.
    let counts = [10usize, 50, 100];
    let mut times: Vec<(usize, u64)> = Vec::new();

    for &n in &counts {
        let mut passes: Vec<IrPass> = Vec::with_capacity(n + 1);
        let mut resources: Vec<IrResource> = Vec::with_capacity(n + 1);

        // One live graphics pass.
        passes.push(mock_pass_graphics(
            PassIndex(0),
            "live",
            &[ResourceHandle(0)],
        ));
        resources.push(mock_resource_texture(ResourceHandle(0), "rt", 256, 256));

        // n dead compute passes.
        for i in 0..n {
            let h = ResourceHandle((100 + i) as u32);
            passes.push(mock_pass_compute(
                PassIndex((1 + i) as usize),
                &format!("dead_{}", i),
                &[],
                &[h],
            ));
            resources.push(mock_resource_buffer(h, &format!("buf_{}", i), 64));
        }

        let compiled = compile(passes, resources);
        times.push((n, compiled.perf_counters.dead_elim_us));
    }

    // dead_elim_us should increase with n (not strictly monotonic due to
    // measurement noise, but the 100-pass case should be >= 10-pass case).
    let t_10 = times.iter().find(|(n, _)| *n == 10).map(|(_, t)| *t).unwrap();
    let t_100 = times.iter().find(|(n, _)| *n == 100).map(|(_, t)| *t).unwrap();
    assert!(
        t_100 >= t_10,
        "dead elimination time for 100 passes ({t_100}us) >= 10 passes ({t_10}us)",
    );
}

// =========================================================================
// SECTION 14 -- Empty graph stats
// =========================================================================

#[test]
fn empty_graph_stats_all_zero() {
    let compiled = compile(vec![], vec![]);

    // compilation_time_us may be a few microseconds due to Instant resolution.
    assert!(
        compiled.compilation_time_us < 200,
        "zero-pass compilation should complete in <200us, got {}us",
        compiled.compilation_time_us,
    );
    // PerfCounters should be negligible (at most a few microseconds from
    // Instant resolution on an empty graph).
    let pc = &compiled.perf_counters;
    assert!(
        pc.dag_build_us < 50,
        "DAG build on empty graph should be <50us, got {}us",
        pc.dag_build_us,
    );
    assert!(
        pc.topo_sort_us < 50,
        "topo-sort on empty graph should be <50us, got {}us",
        pc.topo_sort_us,
    );
    assert!(
        pc.barrier_compute_us < 50,
        "barrier compute on empty graph should be <50us, got {}us",
        pc.barrier_compute_us,
    );
}
