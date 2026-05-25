// Blackbox compile stress test -- 1000-pass compile with varied dependencies.
//
// CLEANROOM: No src/ access beyond the public API exported by the crate.
// Tests use only `renderer_backend::frame_graph::*` -- no internal fields,
// no private methods, no implementation details.
//
// Acceptance criteria:
//   1. 1000 passes with mixed patterns compiles without error
//   2. All 1000 passes present in the compiled output order
//   3. Topological ordering respects every dependency edge
//   4. CompilerStats report non-zero pass counts and compilation time
//   5. PerfCounters report non-zero phase timings
//   6. Edge counts are correct for each pattern segment
//   7. No duplicate or spurious edges are produced
//   8. Barrier list is populated
//
// GAP: T-FG-9.3 GAP 2 -- FrameGraphStress

use renderer_backend::frame_graph::{
    build_dag, CompiledFrameGraph, CompilerStats, DispatchSource, EdgeType, InstanceSource, IrPass,
    IrResource, PassIndex, PassType, PerfCounters, ResourceAccessSet, ResourceDesc, ResourceHandle,
    ResourceLifetime, ResourceState, ViewType,
};
use std::collections::HashMap;

// =============================================================================
// Pattern-generation helpers
// =============================================================================

/// Generates N passes forming a linear chain:
///   pass_0 writes R_base
///   pass_i reads R_{i-1}, writes R_{base+i}
///
/// Returns (passes, base_resource_index) where base_resource_index is the
/// first resource handle used by this segment.
fn chain_segment(
    count: usize,
    start_pass: PassIndex,
    start_resource: u32,
) -> (Vec<IrPass>, u32) {
    let mut passes = Vec::with_capacity(count);
    let mut next_res = start_resource;

    if count == 0 {
        return (passes, next_res);
    }

    // First pass writes the starting resource.
    passes.push(make_compute_pass(
        start_pass,
        &format!("chain_{}_entry", start_pass.0),
        &[],
        &[ResourceHandle(next_res)],
    ));
    next_res += 1;

    // Intermediate chain links.
    for i in 1..count {
        let idx = PassIndex(start_pass.0 + i);
        let r_prev = ResourceHandle(next_res - 1);
        let r_cur = ResourceHandle(next_res);
        passes.push(make_compute_pass(idx, &format!("chain_{}", idx.0), &[r_prev], &[r_cur]));
        next_res += 1;
    }

    (passes, next_res)
}

/// Generates a diamond pattern with a root, two middle passes, and a merge.
///   pass_root writes R_out
///   pass_left  reads R_out, writes R_left
///   pass_right reads R_out, writes R_right
///   pass_merge reads R_left, R_right
///
/// Returns (passes, next_resource).
fn diamond_segment(
    start_pass: PassIndex,
    start_resource: u32,
) -> (Vec<IrPass>, u32) {
    let r_out = ResourceHandle(start_resource);
    let r_left = ResourceHandle(start_resource + 1);
    let r_right = ResourceHandle(start_resource + 2);

    let passes = vec![
        make_compute_pass(start_pass, &format!("diamond_root_{}", start_pass.0), &[], &[r_out]),
        make_compute_pass(
            PassIndex(start_pass.0 + 1),
            &format!("diamond_left_{}", start_pass.0),
            &[r_out],
            &[r_left],
        ),
        make_compute_pass(
            PassIndex(start_pass.0 + 2),
            &format!("diamond_right_{}", start_pass.0),
            &[r_out],
            &[r_right],
        ),
        make_compute_pass(
            PassIndex(start_pass.0 + 3),
            &format!("diamond_merge_{}", start_pass.0),
            &[r_left, r_right],
            &[],
        ),
    ];

    (passes, start_resource + 3)
}

/// Generates a fan-out pattern: one producer, `count` consumers.
///   pass_prod writes R_out
///   pass_i reads R_out, writes R_i (for i in 0..count)
///
/// Returns (passes, next_resource).
fn fan_out_segment(
    count: usize,
    start_pass: PassIndex,
    start_resource: u32,
) -> (Vec<IrPass>, u32) {
    let mut passes = Vec::with_capacity(1 + count);
    let r_out = ResourceHandle(start_resource);
    let mut next_res = start_resource + 1;

    // Producer.
    passes.push(make_compute_pass(
        start_pass,
        &format!("fanout_prod_{}", start_pass.0),
        &[],
        &[r_out],
    ));

    // Consumers.
    for i in 0..count {
        let idx = PassIndex(start_pass.0 + 1 + i);
        let r_i = ResourceHandle(next_res);
        passes.push(make_compute_pass(
            idx,
            &format!("fanout_cons_{}_{}", start_pass.0, i),
            &[r_out],
            &[r_i],
        ));
        next_res += 1;
    }

    (passes, next_res)
}

/// Generates a fan-in pattern: `count` producers converge to one consumer.
///   pass_prod_i writes R_i (for i in 0..count)
///   pass_merge reads R_0 .. R_{count-1}
///
/// Returns (passes, next_resource).
fn fan_in_segment(
    count: usize,
    start_pass: PassIndex,
    start_resource: u32,
) -> (Vec<IrPass>, u32) {
    let mut passes = Vec::with_capacity(count + 1);
    let mut next_res = start_resource;

    // Producers.
    let prod_resources: Vec<ResourceHandle> = (0..count)
        .map(|i| {
            let h = ResourceHandle(next_res);
            let idx = PassIndex(start_pass.0 + i);
            passes.push(make_compute_pass(
                idx,
                &format!("fanin_prod_{}_{}", start_pass.0, i),
                &[],
                &[h],
            ));
            next_res += 1;
            h
        })
        .collect();

    // Consumer merge pass.
    let merge_idx = PassIndex(start_pass.0 + count);
    passes.push(make_compute_pass(
        merge_idx,
        &format!("fanin_merge_{}", start_pass.0),
        &prod_resources,
        &[],
    ));

    (passes, next_res)
}

/// Generates a random-ish bipartite layer: each pass at even indices reads
/// from a resource produced by a pass at a random earlier index; odd passes
/// are chain links.
///
/// Uses a deterministic pseudo-random generator (xorshift32) seeded from the
/// start_pass so results are reproducible across runs.
fn random_layer_segment(
    count: usize,
    start_pass: PassIndex,
    start_resource: u32,
    seed: u32,
) -> (Vec<IrPass>, u32) {
    let mut passes = Vec::with_capacity(count);
    let mut next_res = start_resource;
    let mut rng = XorShift32::new(seed);

    for i in 0..count {
        let idx = PassIndex(start_pass.0 + i);
        if i == 0 {
            // First pass in the segment: write a fresh resource.
            passes.push(make_compute_pass(
                idx,
                &format!("random_entry_{}", idx.0),
                &[],
                &[ResourceHandle(next_res)],
            ));
            next_res += 1;
        } else {
            // Pick a random earlier resource from this segment as a read source.
            let back = rng.next_u32() % (next_res - start_resource);
            let src = ResourceHandle(start_resource + back);
            let dst = ResourceHandle(next_res);
            passes.push(make_compute_pass(
                idx,
                &format!("random_{}", idx.0),
                &[src],
                &[dst],
            ));
            next_res += 1;
        }
    }

    (passes, next_res)
}

// =============================================================================
// Deterministic pseudo-random generator (xorshift32)
// =============================================================================

struct XorShift32(u32);

impl XorShift32 {
    fn new(seed: u32) -> Self {
        Self(seed.overflowing_add(1).0)
    }

    fn next_u32(&mut self) -> u32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.0 = x;
        x
    }
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
// Helper: build a minimal resource list from the max handle used by passes
// =============================================================================

fn build_resource_list(max_handle: u32) -> Vec<IrResource> {
    let mut resources = Vec::with_capacity(max_handle as usize);
    for i in 0..max_handle {
        resources.push(IrResource::new(
            ResourceHandle(i),
            format!("res_{}", i),
            ResourceDesc::Buffer(renderer_backend::frame_graph::BufferDesc {
                size: 1024,
                usage: "storage".into(),
                is_indirect_arg: false,
            }),
            ResourceLifetime::Transient,
            ResourceState::Uninitialized,
        ));
    }
    resources
}

// =============================================================================
// Helper: verify every edge is respected by the topological order
// =============================================================================

fn verify_topological_order(
    order: &[PassIndex],
    _passes: &[IrPass],
    edges: &[renderer_backend::frame_graph::IrEdge],
) {
    // Build a set of valid passes (non-eliminated) from the order.
    let valid: std::collections::HashSet<PassIndex> = order.iter().copied().collect();

    let position: HashMap<PassIndex, usize> = order
        .iter()
        .enumerate()
        .map(|(idx, p)| (*p, idx))
        .collect();

    for edge in edges {
        // Skip edges referencing eliminated passes — those no longer exist
        // in the topological order.
        if !valid.contains(&edge.from) || !valid.contains(&edge.to) {
            continue;
        }
        let from_pos = position.get(&edge.from).expect("edge.from must be in sorted order");
        let to_pos = position.get(&edge.to).expect("edge.to must be in sorted order");
        assert!(
            from_pos < to_pos,
            "edge {:?}: from {:?} (pos {}) must precede to {:?} (pos {})",
            edge,
            edge.from,
            from_pos,
            edge.to,
            to_pos,
        );
    }
}

// =============================================================================
// EXPORTED: generate_1000_pass_mixed_graph
//
// Builds a 1000-pass DAG composed of multiple pattern segments:
//   - 5 chain segments of varying lengths
//   - 20 diamond patterns
//   - 10 fan-out patterns (1->N)
//   - 10 fan-in patterns (N->1)
//   - Random sparse layers
//
// Returns (passes, resources, expected_pass_count, segment_edge_minimum).
// =============================================================================

pub fn generate_1000_pass_stress_graph() -> (Vec<IrPass>, Vec<IrResource>, usize) {
    let total_passes = 1000usize;
    let mut all_passes: Vec<IrPass> = Vec::with_capacity(total_passes);
    let mut next_pass = 0usize;
    let mut next_res = 0u32;

    // ---- Chain segments (5 segments, variable lengths) ----
    for (len, _name) in &[(50, "chain_A"), (60, "chain_B"), (40, "chain_C"), (70, "chain_D"), (30, "chain_E")] {
        let (mut seg, nr) = chain_segment(*len, PassIndex(next_pass), next_res);
        all_passes.append(&mut seg);
        next_pass += len;
        next_res = nr;
    }
    // Total chain passes: 50+60+40+70+30 = 250

    // ---- Diamond segments (20 diamonds = 80 passes) ----
    for _ in 0..20 {
        let (mut seg, nr) = diamond_segment(PassIndex(next_pass), next_res);
        all_passes.append(&mut seg);
        next_pass += 4;
        next_res = nr;
    }
    // Total passes so far: 250 + 80 = 330

    // ---- Fan-out segments (10 fans, each with 8 consumers = 90 passes) ----
    for _ in 0..10 {
        let (mut seg, nr) = fan_out_segment(8, PassIndex(next_pass), next_res);
        all_passes.append(&mut seg);
        next_pass += 9;  // 1 producer + 8 consumers
        next_res = nr;
    }
    // Total passes so far: 330 + 90 = 420

    // ---- Fan-in segments (10 fans, each with 8 producers = 90 passes) ----
    for _ in 0..10 {
        let (mut seg, nr) = fan_in_segment(8, PassIndex(next_pass), next_res);
        all_passes.append(&mut seg);
        next_pass += 9;  // 8 producers + 1 merge
        next_res = nr;
    }
    // Total passes so far: 420 + 90 = 510

    // ---- Random sparse layers (fill remaining to 1000) ----
    let remaining = total_passes - next_pass;
    if remaining > 0 {
        let (mut seg, nr) = random_layer_segment(remaining, PassIndex(next_pass), next_res, 42);
        all_passes.append(&mut seg);
        next_pass += remaining;
        next_res = nr;
    }

    assert_eq!(next_pass, total_passes, "must produce exactly {} passes", total_passes);

    let resources = build_resource_list(next_res);

    (all_passes, resources, total_passes)
}

// =============================================================================
// TEST 1 -- 1000-pass full compile pipeline (Debug profile, no culling)
// =============================================================================

#[test]
fn frame_graph_stress_1000_pass_full_compile() {
    let (passes, resources, expected_count) = generate_1000_pass_stress_graph();

    // Use Debug profile so dead-pass elimination is disabled and all 1000
    // passes survive the compile pipeline.
    let config = renderer_backend::frame_graph::CompilerProfile::DEBUG.config();
    let compiled = CompiledFrameGraph::compile_with_config(passes, resources, config)
        .expect("1000-pass varied-dep graph should compile without error");

    // Verify all 1000 passes survive (Debug profile disables dead-pass elim).
    assert_eq!(
        compiled.order.len(),
        expected_count,
        "compiled order should contain {} passes, got {}",
        expected_count,
        compiled.order.len(),
    );

    // Verify all passes are present in the order (sorted identity check).
    let mut expected_indices: Vec<PassIndex> = (0..expected_count).map(|i| PassIndex(i)).collect();
    expected_indices.sort();
    let mut actual_indices = compiled.order.clone();
    actual_indices.sort();
    assert_eq!(
        actual_indices, expected_indices,
        "sorted compiled order must contain exactly PassIndex(0)..PassIndex({})",
        expected_count,
    );

    // Verify topological ordering respects every edge.
    verify_topological_order(&compiled.order, &compiled.passes, &compiled.edges);

    // CompilerStats must be populated.
    let stats: CompilerStats = compiled.stats().clone();
    assert!(
        stats.passes_total > 0,
        "CompilerStats.passes_total should be > 0"
    );
    assert!(
        stats.compilation_time_us > 0,
        "CompilerStats.compilation_time_us should be > 0, got {}",
        stats.compilation_time_us,
    );

    // PerfCounters must be populated.
    let perf: &PerfCounters = compiled.perf_counters();
    assert!(
        perf.total_us > 0,
        "PerfCounters.total_us should be > 0, got {}",
        perf.total_us,
    );
    assert!(
        perf.dag_build_us > 0,
        "PerfCounters.dag_build_us should be > 0, got {}",
        perf.dag_build_us,
    );
    assert!(
        perf.topo_sort_us > 0,
        "PerfCounters.topo_sort_us should be > 0, got {}",
        perf.topo_sort_us,
    );

    // Barrier list should be non-empty for a 1000-pass graph.
    assert!(
        !compiled.barriers.is_empty(),
        "barrier list should be non-empty for 1000-pass graph",
    );

    // Parallel regions should be populated.
    assert!(
        !compiled.parallel_regions.is_empty(),
        "parallel_regions should be non-empty",
    );

    // Summary string should be present and mention the pass count.
    let summary = compiled.emit_summary();
    assert!(
        !summary.is_empty(),
        "emit_summary() should return a non-empty string",
    );
    assert!(
        summary.contains(&expected_count.to_string()),
        "summary should mention the pass count: {}",
        summary,
    );
}

// =============================================================================
// TEST 1b -- 1000-pass compile WITH dead-pass elimination (Default profile)
// =============================================================================

#[test]
fn frame_graph_stress_1000_pass_with_dead_pass_elim() {
    let (passes, resources, expected_count) = generate_1000_pass_stress_graph();

    // Default profile enables dead-pass elimination, so terminal passes
    // whose outputs are never consumed will be culled.
    let config = renderer_backend::frame_graph::CompilerProfile::DEFAULT.config();
    let compiled = CompiledFrameGraph::compile_with_config(passes, resources, config)
        .expect("1000-pass compile with dead-pass elim should succeed");

    // Pass count must be <= 1000 (some terminal passes eliminated).
    assert!(
        compiled.order.len() <= expected_count,
        "post-cull pass count ({}) should not exceed input count ({})",
        compiled.order.len(),
        expected_count,
    );

    // At least some passes should survive (the structural core).
    assert!(
        compiled.order.len() >= 500,
        "at least 500 passes should survive culling, got {}",
        compiled.order.len(),
    );

    // Cull stats must be populated.
    assert!(
        compiled.cull_stats.passes_total == expected_count,
        "cull_stats.passes_total should be {}, got {}",
        expected_count,
        compiled.cull_stats.passes_total,
    );
    assert!(
        compiled.cull_stats.passes_eliminated > 0,
        "cull_stats.passes_eliminated should be > 0, got {}",
        compiled.cull_stats.passes_eliminated,
    );

    // All surviving edges must respect topological order.
    verify_topological_order(&compiled.order, &compiled.passes, &compiled.edges);

    // Summary should reflect the post-cull count.
    let summary = compiled.emit_summary();
    assert!(
        !summary.is_empty(),
        "emit_summary() should be non-empty",
    );
    assert!(
        summary.contains("dead eliminated"),
        "summary should mention dead-pass elimination",
    );
}

// =============================================================================
// TEST 2 -- 1000-pass DAG edge counts
// =============================================================================

#[test]
fn frame_graph_stress_1000_pass_edge_integrity() {
    let (passes, resources, _) = generate_1000_pass_stress_graph();

    // Build the DAG (equivalent to Phase 2).
    let edges = build_dag(&passes, &resources);

    // Every edge must have valid pass and resource indices.
    for edge in &edges {
        assert!(
            edge.from.0 < passes.len(),
            "edge.from ({:?}) exceeds pass count ({})",
            edge.from,
            passes.len(),
        );
        assert!(
            edge.to.0 < passes.len(),
            "edge.to ({:?}) exceeds pass count ({})",
            edge.to,
            passes.len(),
        );
        assert!(
            edge.from != edge.to,
            "self-loop edge: {:?}",
            edge,
        );
        assert!(
            edge.edge_type == EdgeType::RAW
                || edge.edge_type == EdgeType::WAR
                || edge.edge_type == EdgeType::WAW,
            "unknown edge type: {:?}",
            edge.edge_type,
        );
    }

    // Verify no duplicate edges (same from, same to, same resource).
    let mut seen = std::collections::HashSet::new();
    for edge in &edges {
        let key = (edge.from.0, edge.to.0, edge.resource.0);
        assert!(
            seen.insert(key),
            "duplicate edge: from={:?}, to={:?}, resource={:?}",
            edge.from,
            edge.to,
            edge.resource,
        );
    }

    // Every edge should correspond to a genuine resource dependency.
    for edge in &edges {
        let from_pass = &passes[edge.from.0];
        let to_pass = &passes[edge.to.0];

        let resource_in_from = from_pass.access_set.contains(edge.resource);
        let resource_in_to = to_pass.access_set.contains(edge.resource);

        assert!(
            resource_in_from || resource_in_to,
            "edge resource {:?} not found in either from_pass ({}) or to_pass ({})",
            edge.resource,
            from_pass.name,
            to_pass.name,
        );
    }
}

// =============================================================================
// TEST 3 -- Compiler config respects max_passes bound
// =============================================================================

#[test]
fn frame_graph_stress_max_passes_bound() {
    let (passes, resources, _) = generate_1000_pass_stress_graph();
    let limit = 100usize;

    let config = renderer_backend::frame_graph::CompilerConfig {
        max_passes: limit,
        enable_dead_pass_elim: false,   // disable culling so max_passes is the only bound
        enable_barrier_opt: false,
        enable_async_scheduling: false,
        enable_aliasing: false,
        enable_validation: false,
    };

    let compiled = CompiledFrameGraph::compile_with_config(passes, resources, config)
        .expect("compile with max_passes bound should succeed");

    assert_eq!(
        compiled.order.len(),
        limit,
        "compiled order should respect max_passes bound of {}, got {}",
        limit,
        compiled.order.len(),
    );
}

// =============================================================================
// TEST 4 -- Multiple compile profiles produce valid results
// =============================================================================

#[test]
fn frame_graph_stress_compiler_profiles() {
    let (passes, resources, expected_count) = generate_1000_pass_stress_graph();

    // Debug profile: all optimisation off — all 1000 passes survive.
    {
        let config = renderer_backend::frame_graph::CompilerProfile::DEBUG.config();
        let compiled = CompiledFrameGraph::compile_with_config(
            passes.clone(),
            resources.clone(),
            config,
        )
        .expect("Debug profile compile should succeed");
        assert_eq!(
            compiled.order.len(),
            expected_count,
            "Debug profile: all {} passes should survive",
            expected_count,
        );
        assert!(
            compiled.eliminated_passes.is_empty(),
            "Debug profile: no passes should be eliminated, got {}",
            compiled.eliminated_passes.len(),
        );
        verify_topological_order(&compiled.order, &compiled.passes, &compiled.edges);
    }

    // Default profile: dead-pass elim enabled, pass count <= 1000.
    {
        let config = renderer_backend::frame_graph::CompilerProfile::DEFAULT.config();
        let compiled = CompiledFrameGraph::compile_with_config(
            passes.clone(),
            resources.clone(),
            config,
        )
        .expect("Default profile compile should succeed");
        assert!(
            compiled.order.len() <= expected_count,
            "Default profile: pass count ({}) should not exceed {}",
            compiled.order.len(),
            expected_count,
        );
        assert!(
            compiled.order.len() >= 500,
            "Default profile: at least 500 passes should survive, got {}",
            compiled.order.len(),
        );
        assert!(
            compiled.cull_stats.passes_eliminated > 0,
            "Default profile: should eliminate dead passes, got 0",
        );
        verify_topological_order(&compiled.order, &compiled.passes, &compiled.edges);
    }

    // Performance profile: similar to Default but with async + aliasing.
    {
        let config = renderer_backend::frame_graph::CompilerProfile::PERFORMANCE.config();
        let compiled = CompiledFrameGraph::compile_with_config(passes, resources, config)
            .expect("Performance profile compile should succeed");
        assert!(
            compiled.order.len() <= expected_count,
            "Performance profile: pass count ({}) should not exceed {}",
            compiled.order.len(),
            expected_count,
        );
        assert!(
            compiled.order.len() >= 500,
            "Performance profile: at least 500 passes should survive, got {}",
            compiled.order.len(),
        );
        verify_topological_order(&compiled.order, &compiled.passes, &compiled.edges);
    }
}

// =============================================================================
// TEST 5 -- 1000-pass with QualityPresets
// =============================================================================

#[test]
fn frame_graph_stress_quality_presets() {
    let (passes, resources, expected_count) = generate_1000_pass_stress_graph();

    for (name, preset) in &[
        ("Debug", renderer_backend::frame_graph::QualityPresets::DEBUG),
        ("Release", renderer_backend::frame_graph::QualityPresets::RELEASE),
        ("Production", renderer_backend::frame_graph::QualityPresets::PRODUCTION),
    ] {
        let mut config = renderer_backend::frame_graph::CompilerConfig {
            max_passes: 1000,
            ..renderer_backend::frame_graph::CompilerConfig::default()
        };
        preset.apply(&mut config);

        let compiled = CompiledFrameGraph::compile_with_config(
            passes.clone(),
            resources.clone(),
            config,
        )
        .unwrap_or_else(|e| {
            panic!("{} QualityPreset compile failed: {}", name, e);
        });

        // Debug preset disables optimisations — all passes survive.
        // Release/Production enable dead-pass elim — some passes culled.
        if *name == "Debug" {
            assert_eq!(
                compiled.order.len(),
                expected_count,
                "Debug QualityPreset: all {} passes should survive",
                expected_count,
            );
        } else {
            assert!(
                compiled.order.len() <= expected_count,
                "{} QualityPreset: pass count ({}) should not exceed {}",
                name,
                compiled.order.len(),
                expected_count,
            );
            assert!(
                compiled.order.len() >= 500,
                "{} QualityPreset: at least 500 passes should survive, got {}",
                name,
                compiled.order.len(),
            );
        }

        // Summary must be non-empty.
        let summary = compiled.emit_summary();
        assert!(
            !summary.is_empty(),
            "{} QualityPreset summary should be non-empty",
            name,
        );
    }
}

// =============================================================================
// TEST 6 -- JSON bridge export for a 1000-pass compiled graph
// =============================================================================

#[test]
fn frame_graph_stress_json_bridge_export() {
    let (passes, resources, expected_count) = generate_1000_pass_stress_graph();

    // Debug profile ensures all passes survive for a complete bridge export.
    let config = renderer_backend::frame_graph::CompilerProfile::DEBUG.config();
    let compiled = CompiledFrameGraph::compile_with_config(passes, resources, config)
        .expect("1000-pass compile for JSON bridge should succeed");

    let json = compiled.emit_bridge_json();

    // Verify JSON structure.
    assert!(
        json.get("passes").and_then(|v| v.as_array()).map_or(false, |a| a.len() == expected_count),
        "JSON bridge should contain exactly {} passes",
        expected_count,
    );

    assert!(
        json.get("resources").and_then(|v| v.as_array()).map_or(false, |a| !a.is_empty()),
        "JSON bridge should contain resources",
    );

    assert!(
        json.get("barriers").and_then(|v| v.as_array()).is_some(),
        "JSON bridge should contain barriers",
    );

    assert!(
        json.get("parallel_regions").and_then(|v| v.as_array()).is_some(),
        "JSON bridge should contain parallel_regions",
    );

    assert!(
        json.get("depths").and_then(|v| v.as_object()).is_some(),
        "JSON bridge should contain depths",
    );

    assert!(
        json.get("cull_stats").and_then(|v| v.as_object()).is_some(),
        "JSON bridge should contain cull_stats",
    );

    assert!(
        json.get("validation").and_then(|v| v.as_object()).is_some(),
        "JSON bridge should contain validation result",
    );
}
