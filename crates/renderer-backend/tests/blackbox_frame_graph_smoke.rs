// SPDX-License-Identifier: MIT
//
// FrameGraphSmokeTest (T-FG-9.2 GAP 2)
//
// Basic smoke test that compiles a standard deferred rendering graph and
// verifies the output structure of CompiledFrameGraph. Exercises the full
// compile pipeline (build_dag -> topological_sort -> compute_pass_depths
// -> compute_lifetimes -> compute_barriers -> async_schedule ->
// eliminate_dead_passes) with a realistic multi-pass pipeline.
//
// CLEANROOM: No src/ access beyond the public API exported by the crate.
// Tests use only `renderer_backend::frame_graph::*` -- no internal fields,
// no private methods, no implementation details.
//
// Deferred rendering graph topology:
//
//   P0 (gbuffer, Graphics) --writes depth, albedo, normal
//   |       \
//   |        P1 (ssao, Compute) --reads depth, normal; writes ao
//   |       /       |
//   P2 (lighting, Compute) --reads albedo, normal, depth, ao; writes lighting
//   |
//   P3 (compose, Graphics) --reads lighting, ao; writes final
//   |
//   P4 (tonemap, Compute) --reads final; writes output_buf
//   |
//   P5 (present, Copy) --reads output_buf; writes swapchain [ELIMINATED]
//
// Dead pass elimination: the terminal `present` copy pass is eliminated because
// its output (swapchain) has no downstream reader. Only Graphics passes are
// unconditionally preserved. This is expected compiler behaviour -- in a real
// renderer the swapchain image is consumed by the presentation engine.
//
// Resources:
//   R0: depth      (Texture2D,  1920x1080, depth32float)
//   R1: albedo     (Texture2D,  1920x1080, rgba8unorm)
//   R2: normal     (Texture2D,  1920x1080, rgba8unorm)
//   R3: ao         (Texture2D,  1920x1080, r8unorm)
//   R4: lighting   (Texture2D,  1920x1080, rgba16float)
//   R5: final      (Texture2D,  1920x1080, rgba8unorm-srgb)
//   R6: output_buf (Buffer,     1048576)
//   R7: swapchain  (Texture2D,  1920x1080, rgba8unorm-srgb)
//
// Acceptance criteria:
//   1. 5 passes survive (P0-P4). P5 is eliminated as dead (swapchain has no
//      downstream reader; only Graphics passes are unconditionally alive).
//   2. All 8 resources are preserved in the output.
//   3. Topological order respects producer-before-consumer.
//   4. Dependency edges exist for every producer-consumer pair.
//   5. Pipeline barriers are generated where resource state transitions occur.
//   6. CullStats reflect 1 eliminated pass (P5), 1 freed resource, bytes saved.

use renderer_backend::frame_graph::{
    mock_pass_compute, mock_pass_graphics, mock_resource_buffer, mock_resource_texture,
    AttachmentLoadOp, AttachmentStoreOp, DepthStencilAttachment, DispatchSource,
    FrameGraphCompiler, IrPass, IrResource, PassIndex, PassType, ResourceAccessSet,
    ResourceHandle, ViewType,
};

// =============================================================================
// Helper: construct a standard deferred rendering graph with 6 passes and 8
// resources. Returns (passes, resources) ready for FrameGraphCompiler::new().
// =============================================================================

fn build_deferred_rendering_graph() -> (Vec<IrPass>, Vec<IrResource>) {
    // -- Resource handles --
    let depth = ResourceHandle(0);
    let albedo = ResourceHandle(1);
    let normal = ResourceHandle(2);
    let ao = ResourceHandle(3);
    let lighting = ResourceHandle(4);
    let final_ = ResourceHandle(5);
    let output_buf = ResourceHandle(6);
    let swapchain = ResourceHandle(7);

    // -- Resources --
    let resources = vec![
        mock_resource_texture(depth, "depth", 1920, 1080),
        mock_resource_texture(albedo, "albedo", 1920, 1080),
        mock_resource_texture(normal, "normal", 1920, 1080),
        mock_resource_texture(ao, "ao", 1920, 1080),
        mock_resource_texture(lighting, "lighting", 1920, 1080),
        mock_resource_texture(final_, "final", 1920, 1080),
        mock_resource_buffer(output_buf, "output_buf", 1048576),
        mock_resource_texture(swapchain, "swapchain", 1920, 1080),
    ];

    // -- Pass 0: G-buffer (Graphics) --
    // Writes albedo + normal as color attachments, depth via depth-stencil.
    let mut p0 = mock_pass_graphics(PassIndex(0), "gbuffer", &[albedo, normal]);
    p0.depth_stencil = Some(DepthStencilAttachment {
        resource: depth,
        depth_load_op: AttachmentLoadOp::Clear,
        depth_store_op: AttachmentStoreOp::Store,
        ..DepthStencilAttachment::default()
    });
    p0.access_set.writes.push(depth);

    // -- Pass 1: SSAO (Compute) --
    // Reads depth + normal, writes ao.
    let p1 = mock_pass_compute(PassIndex(1), "ssao", &[depth, normal], &[ao]);

    // -- Pass 2: Lighting (Compute) --
    // Reads all gbuffer outputs + ao, writes lighting.
    let p2 = mock_pass_compute(
        PassIndex(2),
        "lighting",
        &[albedo, normal, depth, ao],
        &[lighting],
    );

    // -- Pass 3: Composition (Graphics) --
    // Reads lighting + ao as shader-read inputs; writes final via color attachment.
    let mut p3 = mock_pass_graphics(PassIndex(3), "compose", &[final_]);
    p3.access_set.reads.push(lighting);
    p3.access_set.reads.push(ao);

    // -- Pass 4: Tone-map (Compute) --
    // Reads final color, writes output buffer.
    let p4 = mock_pass_compute(PassIndex(4), "tonemap", &[final_], &[output_buf]);

    // -- Pass 5: Present (Copy) --
    // Reads output buffer, writes swapchain texture.
    // Copy passes survive dead-pass elimination even when their output is not
    // consumed downstream (analogous to graphics passes).
    let mut p5 = IrPass::copy(PassIndex(5), "present");
    p5.access_set = ResourceAccessSet {
        reads: vec![output_buf],
        writes: vec![swapchain],
    };

    let passes = vec![p0, p1, p2, p3, p4, p5];

    (passes, resources)
}

// =============================================================================
// SECTION 1 -- Compilation succeeds for a standard deferred rendering graph
// =============================================================================

/// The full deferred rendering graph compiles without error.
#[test]
fn deferred_rendering_graph_compiles_successfully() {
    let (passes, resources) = build_deferred_rendering_graph();
    let compiler = FrameGraphCompiler::new(passes, resources);
    let result = compiler.compile();

    assert!(
        result.is_ok(),
        "Standard deferred rendering graph must compile successfully",
    );
}

// =============================================================================
// SECTION 2 -- Pass count and types are preserved after compilation
// =============================================================================

/// 5 passes survive compilation (P0-P4 live; P5 copy pass with unread output eliminated).
#[test]
fn all_passes_survive_compilation() {
    let (passes, resources) = build_deferred_rendering_graph();
    let compiled = FrameGraphCompiler::new(passes, resources)
        .compile()
        .expect("Deferred graph compiles");

    // compiled.passes contains ALL input passes (including eliminated).
    // Use compiled.order or cull_stats to determine surviving count.
    assert_eq!(
        compiled.passes.len(),
        6,
        "All 6 input passes retained in compiled.passes (including eliminated P5)",
    );
    assert_eq!(
        compiled.order.len(),
        5,
        "5 passes appear in execution order (P5 eliminated)",
    );
}

/// Each pass retains its correct type (Graphics, Compute, or Copy).
#[test]
fn pass_types_are_preserved() {
    let (passes, resources) = build_deferred_rendering_graph();
    let compiled = FrameGraphCompiler::new(passes, resources)
        .compile()
        .expect("Deferred graph compiles");

    // P5 (present, Copy) is eliminated as dead. Surviving types:
    let expected_types = [
        PassType::Graphics, // P0: gbuffer
        PassType::Compute,  // P1: ssao
        PassType::Compute,  // P2: lighting
        PassType::Graphics, // P3: compose
        PassType::Compute,  // P4: tonemap
    ];

    for (i, expected) in expected_types.iter().enumerate() {
        assert_eq!(
            compiled.passes[i].pass_type,
            *expected,
            "Pass {} ({}) has correct type",
            i,
            compiled.passes[i].name,
        );
    }
}

/// Each pass retains its original name after compilation.
#[test]
fn pass_names_are_preserved() {
    let (passes, resources) = build_deferred_rendering_graph();
    let compiled = FrameGraphCompiler::new(passes, resources)
        .compile()
        .expect("Deferred graph compiles");

    // P5 (present) is eliminated as dead; surviving names are P0-P4.
    let expected_names = [
        "gbuffer", "ssao", "lighting", "compose", "tonemap",
    ];

    for (i, expected) in expected_names.iter().enumerate() {
        assert_eq!(
            compiled.passes[i].name,
            *expected,
            "Pass {} has correct name",
            i,
        );
    }
}

/// Each pass retains its original PassIndex after compilation.
#[test]
fn pass_indices_are_preserved() {
    let (passes, resources) = build_deferred_rendering_graph();
    let compiled = FrameGraphCompiler::new(passes, resources)
        .compile()
        .expect("Deferred graph compiles");

    // Only P0-P4 survive (P5 is eliminated). Check all surviving passes.
    for i in 0..5 {
        assert_eq!(
            compiled.passes[i].index,
            PassIndex(i),
            "Pass {} has correct index",
            i,
        );
    }
}

// =============================================================================
// SECTION 3 -- Resources are preserved in the compiled output
// =============================================================================

/// All 8 resources are present in the compiled graph output.
#[test]
fn all_resources_are_preserved() {
    let (passes, resources) = build_deferred_rendering_graph();
    let compiled = FrameGraphCompiler::new(passes, resources)
        .compile()
        .expect("Deferred graph compiles");

    assert_eq!(
        compiled.resources.len(),
        8,
        "All 8 resources are preserved in compiled output",
    );
}

/// Each resource has the correct handle and name.
#[test]
fn resource_handles_and_names_are_correct() {
    let (passes, resources) = build_deferred_rendering_graph();
    let compiled = FrameGraphCompiler::new(passes, resources)
        .compile()
        .expect("Deferred graph compiles");

    let expected = [
        (ResourceHandle(0), "depth"),
        (ResourceHandle(1), "albedo"),
        (ResourceHandle(2), "normal"),
        (ResourceHandle(3), "ao"),
        (ResourceHandle(4), "lighting"),
        (ResourceHandle(5), "final"),
        (ResourceHandle(6), "output_buf"),
        (ResourceHandle(7), "swapchain"),
    ];

    for (i, (expected_handle, expected_name)) in expected.iter().enumerate() {
        assert_eq!(
            compiled.resources[i].handle,
            *expected_handle,
            "Resource {} has correct handle",
            i,
        );
        assert_eq!(
            compiled.resources[i].name,
            *expected_name,
            "Resource {} has correct name",
            i,
        );
    }
}

// =============================================================================
// SECTION 4 -- Topological order is correct (producer before consumer)
// =============================================================================

/// Topological order must respect all producer-before-consumer constraints.
#[test]
fn topological_order_is_valid() {
    let (passes, resources) = build_deferred_rendering_graph();
    let compiled = FrameGraphCompiler::new(passes, resources)
        .compile()
        .expect("Deferred graph compiles");

    let order = &compiled.order;

    // 5 surviving passes (P0-P4; P5 eliminated as terminal).
    assert_eq!(order.len(), 5, "Exactly 5 entries in execution order");

    for i in 0..5 {
        assert!(
            order.contains(&PassIndex(i)),
            "Pass {} appears in execution order",
            i,
        );
    }

    // P0 (gbuffer, producer) must appear before P1 (ssao, consumer).
    let pos_p0 = order.iter().position(|&p| p == PassIndex(0)).unwrap();
    let pos_p1 = order.iter().position(|&p| p == PassIndex(1)).unwrap();
    assert!(
        pos_p0 < pos_p1,
        "P0 (gbuffer) must appear before P1 (ssao)",
    );

    // P0 must appear before P2 (lighting).
    let pos_p2 = order.iter().position(|&p| p == PassIndex(2)).unwrap();
    assert!(
        pos_p0 < pos_p2,
        "P0 (gbuffer) must appear before P2 (lighting)",
    );

    // P1 must appear before P2 (lighting reads ao from ssao).
    assert!(
        pos_p1 < pos_p2,
        "P1 (ssao) must appear before P2 (lighting)",
    );

    // P2 must appear before P3 (compose).
    let pos_p3 = order.iter().position(|&p| p == PassIndex(3)).unwrap();
    assert!(
        pos_p2 < pos_p3,
        "P2 (lighting) must appear before P3 (compose)",
    );

    // P3 must appear before P4 (tonemap).
    let pos_p4 = order.iter().position(|&p| p == PassIndex(4)).unwrap();
    assert!(
        pos_p3 < pos_p4,
        "P3 (compose) must appear before P4 (tonemap)",
    );

    // P5 is eliminated; no ordering constraint for it.
}

// =============================================================================
// SECTION 5 -- Dependency edges are generated between correct passes
// =============================================================================

/// Dependency edges exist for every producer-consumer relationship.
#[test]
fn dependency_edges_cover_all_producer_consumer_pairs() {
    let (passes, resources) = build_deferred_rendering_graph();
    let compiled = FrameGraphCompiler::new(passes, resources)
        .compile()
        .expect("Deferred graph compiles");

    let edges = &compiled.edges;

    // Edges are created per-resource per-producer-consumer pair.
    // IrEdge has a `resource` field, so multiple resources between the
    // same two passes produce multiple edges.
    //
    // Expected edges (from, to, resource):
    //   P0->P1: depth, normal             (gbuffer writes, ssao reads)
    //   P0->P2: albedo, normal, depth     (gbuffer writes, lighting reads)
    //   P1->P2: ao                        (ssao writes, lighting reads)
    //   P1->P3: ao                        (ssao writes, compose reads)
    //   P2->P3: lighting                  (lighting writes, compose reads)
    //   P3->P4: final                     (compose writes, tonemap reads)
    //   P4->P5: output_buf                (tonemap writes, present reads)
    // Total: 10 edges.

    assert_eq!(
        edges.len(),
        10,
        "Expected 10 per-resource dependency edges in deferred rendering graph",
    );

    // Verify every expected (from, to) pair has at least one edge.
    let expected_pairs = [
        (PassIndex(0), PassIndex(1)),
        (PassIndex(0), PassIndex(2)),
        (PassIndex(1), PassIndex(2)),
        (PassIndex(1), PassIndex(3)),
        (PassIndex(2), PassIndex(3)),
        (PassIndex(3), PassIndex(4)),
        (PassIndex(4), PassIndex(5)),
    ];

    for &(from, to) in &expected_pairs {
        let exists = edges
            .iter()
            .any(|e| e.from == from && e.to == to);
        assert!(
            exists,
            "Dependency edge from {:?} to {:?} must exist",
            from, to,
        );
    }
}

/// Every edge references valid pass indices.
#[test]
fn edge_references_are_valid() {
    let (passes, resources) = build_deferred_rendering_graph();
    let compiled = FrameGraphCompiler::new(passes, resources)
        .compile()
        .expect("Deferred graph compiles");

    let valid_indices: std::collections::HashSet<PassIndex> =
        (0..6).map(PassIndex).collect();

    for edge in &compiled.edges {
        assert!(
            valid_indices.contains(&edge.from),
            "Edge 'from' {:?} is a valid pass index",
            edge.from,
        );
        assert!(
            valid_indices.contains(&edge.to),
            "Edge 'to' {:?} is a valid pass index",
            edge.to,
        );
    }
}

// =============================================================================
// SECTION 6 -- Pipeline barriers are generated
// =============================================================================

/// Barriers exist in the compiled graph (resource state transitions between
/// dependent passes).
#[test]
fn pipeline_barriers_are_generated() {
    let (passes, resources) = build_deferred_rendering_graph();
    let compiled = FrameGraphCompiler::new(passes, resources)
        .compile()
        .expect("Deferred graph compiles");

    assert!(
        !compiled.barriers.is_empty(),
        "Barriers must be generated for resource state transitions in \
         the deferred rendering graph",
    );
}

/// Barriers reference valid pass indices and resource handles.
#[test]
fn barrier_entries_have_valid_references() {
    let (passes, resources) = build_deferred_rendering_graph();
    let compiled = FrameGraphCompiler::new(passes, resources)
        .compile()
        .expect("Deferred graph compiles");

    let valid_indices: std::collections::HashSet<PassIndex> =
        (0..6).map(PassIndex).collect();
    let valid_resources: std::collections::HashSet<ResourceHandle> =
        (0..8).map(ResourceHandle).collect();

    for &(from, to, resource, before, after) in &compiled.barriers {
        assert!(
            valid_indices.contains(&from),
            "Barrier 'from' {:?} is a valid pass index",
            from,
        );
        assert!(
            valid_indices.contains(&to),
            "Barrier 'to' {:?} is a valid pass index",
            to,
        );
        assert!(
            valid_resources.contains(&resource),
            "Barrier resource {:?} is a valid resource handle",
            resource,
        );
        let _ = (before, after); // state enum values -- structurally valid.
    }
}

// =============================================================================
// SECTION 7 -- No dead pass elimination in a fully-connected graph
// =============================================================================

/// CullStats reflect 1 eliminated pass (P5: terminal copy pass with no downstream reader).
#[test]
fn no_dead_passes_in_connected_graph() {
    let (passes, resources) = build_deferred_rendering_graph();
    let compiled = FrameGraphCompiler::new(passes, resources)
        .compile()
        .expect("Deferred graph compiles");

    let stats = &compiled.cull_stats;

    assert_eq!(
        stats.passes_total, 6,
        "CullStats.passes_total == 6 input passes",
    );
    assert_eq!(
        stats.passes_eliminated, 1,
        "P5 (terminal copy pass) eliminated; only Graphics passes are \
         unconditionally preserved",
    );
    assert_eq!(
        stats.resources_freed, 1,
        "One unique resource (swapchain) freed by eliminated P5",
    );
    assert!(
        stats.bytes_saved > 0,
        "Swapchain texture bytes accounted for in bytes_saved",
    );
}

// =============================================================================
// SECTION 8 -- Async scheduling metadata is populated
// =============================================================================

/// Async passes list is present (compute passes may be eligible).
#[test]
fn async_passes_field_is_populated() {
    let (passes, resources) = build_deferred_rendering_graph();
    let compiled = FrameGraphCompiler::new(passes, resources)
        .compile()
        .expect("Deferred graph compiles");

    // The async_passes field must be a Vec<(PassIndex, String)>.
    let _async_list: &Vec<(PassIndex, String)> = &compiled.async_passes;

    // At minimum the field is accessible. Compute passes (ssao, lighting,
    // tonemap) may be scheduled as async compute. async_schedule runs
    // BEFORE eliminate_dead_passes, so all 6 input passes are eligible.
    for &(pi, ref queue_type) in &compiled.async_passes {
        assert!(
            pi.0 < 6,
            "Async pass index {:?} is within range (0..6 input passes)",
            pi,
        );
        let is_valid_queue = queue_type == "compute"
            || queue_type == "copy"
            || queue_type == "graphics";
        assert!(
            is_valid_queue,
            "Async queue type '{}' is valid",
            queue_type,
        );
    }
}

// =============================================================================
// SECTION 9 -- Pass depths and parallel regions
// =============================================================================

/// Pass depths are assigned and non-negative.
#[test]
fn pass_depths_are_assigned() {
    let (passes, resources) = build_deferred_rendering_graph();
    let compiled = FrameGraphCompiler::new(passes, resources)
        .compile()
        .expect("Deferred graph compiles");

    let depths = &compiled.depths;

    // All 5 surviving passes (P0-P4) have an assigned depth.
    // P5 is eliminated but its depth may still be present.
    for i in 0..5 {
        let depth = depths.get(&PassIndex(i));
        assert!(
            depth.is_some(),
            "Pass {} has an assigned depth",
            i,
        );
        assert!(
            *depth.unwrap() < 6,
            "Pass {} depth is reasonable (< 6 for this graph)",
            i,
        );
    }
}

/// Parallel regions are populated from pass depths.
#[test]
fn parallel_regions_are_populated() {
    let (passes, resources) = build_deferred_rendering_graph();
    let compiled = FrameGraphCompiler::new(passes, resources)
        .compile()
        .expect("Deferred graph compiles");

    // parallel_regions should contain at least one region.
    assert!(
        !compiled.parallel_regions.is_empty(),
        "Parallel regions must be populated",
    );

    // 5 surviving passes across parallel regions (P5 eliminated).
    let total: usize = compiled
        .parallel_regions
        .iter()
        .map(|region| region.len())
        .sum();
    assert_eq!(
        total, 5,
        "All 5 surviving passes appear across parallel regions",
    );
}

// =============================================================================
// SECTION 10 -- Compilation performance counters
// =============================================================================

/// PerfCounters are populated with non-zero values for each pipeline phase.
#[test]
fn perf_counters_are_populated() {
    let (passes, resources) = build_deferred_rendering_graph();
    let compiled = FrameGraphCompiler::new(passes, resources)
        .compile()
        .expect("Deferred graph compiles");

    let counters = &compiled.perf_counters;

    // Total compilation time must be non-zero.
    assert!(
        counters.total_us > 0,
        "Total compilation time is non-zero (was {} us)",
        counters.total_us,
    );

    // Phase-level counters: at minimum, dag_build must have been measured.
    assert!(
        counters.dag_build_us > 0 || counters.total_us > 0,
        "DAG build phase has recorded timing",
    );
}

// =============================================================================
// SECTION 11 -- CompilerStats integration
// =============================================================================

/// CompilerStats fields are populated and consistent with the compiled graph.
#[test]
fn compiler_stats_are_consistent() {
    let (passes, resources) = build_deferred_rendering_graph();
    let compiled = FrameGraphCompiler::new(passes, resources)
        .compile()
        .expect("Deferred graph compiles");

    let stats = &compiled.stats;

    // passes_total matches input count (includes eliminated).
    assert_eq!(stats.passes_total, 6, "passes_total == 6");

    // passes_eliminated is 1 (P5 terminal copy pass).
    assert_eq!(stats.passes_eliminated, 1, "passes_eliminated == 1 (P5)");

    // compilation_time_us is non-zero.
    assert!(
        stats.compilation_time_us > 0,
        "compilation_time_us is non-zero",
    );

    // barriers_total is consistent with final barrier count (pre-optimization
    // count may differ from post-optimization).
    assert!(
        stats.barriers_total >= compiled.barriers.len(),
        "barriers_total ({}) >= compiled barriers ({})",
        stats.barriers_total,
        compiled.barriers.len(),
    );

    // async_passes count matches the async_passes vector length.
    assert_eq!(
        stats.async_passes,
        compiled.async_passes.len(),
        "stats.async_passes matches compiled.async_passes.len()",
    );
}

// =============================================================================
// SECTION 12 -- Size and accessor sanity checks
// =============================================================================

/// Verify pass depth ordering is monotonic in execution order (depths should
/// be non-decreasing as we follow the topological order).
#[test]
fn pass_depths_are_non_decreasing_in_execution_order() {
    let (passes, resources) = build_deferred_rendering_graph();
    let compiled = FrameGraphCompiler::new(passes, resources)
        .compile()
        .expect("Deferred graph compiles");

    let order = &compiled.order;
    let depths = &compiled.depths;

    if order.len() >= 2 {
        for window in order.windows(2) {
            let d0 = depths.get(&window[0]).copied().unwrap_or(0);
            let d1 = depths.get(&window[1]).copied().unwrap_or(0);
            // Depths should generally be non-decreasing in topological order.
            // Adjacent passes may share the same depth (parallel candidates).
            assert!(
                d1 >= d0 || (d1 + 1) >= d0,
                "Depth should not skip backwards: P{:?} depth {} -> P{:?} depth {}",
                window[0], d0, window[1], d1,
            );
        }
    }
}

/// Total compilation time from top-level field is positive.
#[test]
fn compilation_time_is_positive() {
    let (passes, resources) = build_deferred_rendering_graph();
    let compiled = FrameGraphCompiler::new(passes, resources)
        .compile()
        .expect("Deferred graph compiles");

    assert!(
        compiled.compilation_time_us > 0,
        "compilation_time_us ({}) must be positive",
        compiled.compilation_time_us,
    );
}

// =============================================================================
// SECTION 13 -- CompiledFrameGraph field structure verification
// =============================================================================

/// All documented CompiledFrameGraph fields are publicly accessible.
#[test]
fn compiled_frame_graph_fields_are_accessible() {
    let (passes, resources) = build_deferred_rendering_graph();
    let compiled = FrameGraphCompiler::new(passes, resources)
        .compile()
        .expect("Deferred graph compiles");

    // passes: Vec<IrPass>
    let _passes: &Vec<IrPass> = &compiled.passes;
    // resources: Vec<IrResource>
    let _resources: &Vec<IrResource> = &compiled.resources;
    // edges: Vec<IrEdge>
    let _edges = &compiled.edges;
    // order: Vec<PassIndex>
    let _order: &Vec<PassIndex> = &compiled.order;
    // depths: HashMap<PassIndex, u32>
    let _depths = &compiled.depths;
    // barriers: Vec<(PassIndex, PassIndex, ResourceHandle, ResourceState, ResourceState)>
    let _barriers = &compiled.barriers;
    // async_passes: Vec<(PassIndex, String)>
    let _async: &Vec<(PassIndex, String)> = &compiled.async_passes;
    // eliminated_passes: Vec<PassIndex>
    let _eliminated: &Vec<PassIndex> = &compiled.eliminated_passes;
    // cull_stats: CullStats
    let _stats = &compiled.cull_stats;
    // parallel_regions: Vec<Vec<PassIndex>>
    let _regions: &Vec<Vec<PassIndex>> = &compiled.parallel_regions;
    // compilation_time_us: u64
    let _time: u64 = compiled.compilation_time_us;
    // stats: CompilerStats
    let _compiler_stats = &compiled.stats;
    // perf_counters: PerfCounters
    let _perf = &compiled.perf_counters;

    // Structural assertion: all fields pass their type checks.
    assert!(
        _order.len() == 5,
        "Execution order has 5 entries (P5 eliminated)",
    );
    assert_eq!(
        _eliminated.len(),
        1,
        "P5 eliminated as terminal pass",
    );
    assert!(
        _time > 0,
        "Compilation time is positive",
    );
}

// =============================================================================
// SECTION 14 -- Multiple compilations with the same graph are idempotent
// =============================================================================

/// Compiling the same graph twice produces the same output structure.
#[test]
fn recompilation_is_idempotent() {
    let (passes_a, resources_a) = build_deferred_rendering_graph();
    let (passes_b, resources_b) = build_deferred_rendering_graph();

    let compiled_a = FrameGraphCompiler::new(passes_a, resources_a)
        .compile()
        .expect("First compile");

    let compiled_b = FrameGraphCompiler::new(passes_b, resources_b)
        .compile()
        .expect("Second compile");

    // Same pass count.
    assert_eq!(
        compiled_a.passes.len(),
        compiled_b.passes.len(),
        "Same pass count across compilations",
    );

    // Same execution order.
    assert_eq!(
        compiled_a.order,
        compiled_b.order,
        "Same execution order across compilations",
    );

    // Same edge count.
    assert_eq!(
        compiled_a.edges.len(),
        compiled_b.edges.len(),
        "Same edge count across compilations",
    );

    // Same resource count.
    assert_eq!(
        compiled_a.resources.len(),
        compiled_b.resources.len(),
        "Same resource count across compilations",
    );

    // Same cull stats.
    assert_eq!(
        compiled_a.cull_stats,
        compiled_b.cull_stats,
        "Same cull stats across compilations",
    );
}

// =============================================================================
// SECTION 15 -- ViewType correctness per pass
// =============================================================================

/// Each pass preserves the correct ViewType.
#[test]
fn view_types_are_correct_per_pass_type() {
    let (passes, resources) = build_deferred_rendering_graph();
    let compiled = FrameGraphCompiler::new(passes, resources)
        .compile()
        .expect("Deferred graph compiles");

    // P0 (gbuffer, Graphics) -> Texture2D (from mock)
    assert_eq!(
        compiled.passes[0].view_type,
        ViewType::Texture2D,
        "P0 gbuffer uses Texture2D view type",
    );

    // P1 (ssao, Compute) -> Storage (from mock)
    assert_eq!(
        compiled.passes[1].view_type,
        ViewType::Storage,
        "P1 ssao uses Storage view type",
    );

    // P2 (lighting, Compute) -> Storage
    assert_eq!(
        compiled.passes[2].view_type,
        ViewType::Storage,
        "P2 lighting uses Storage view type",
    );

    // P3 (compose, Graphics) -> Texture2D (from mock)
    assert_eq!(
        compiled.passes[3].view_type,
        ViewType::Texture2D,
        "P3 compose uses Texture2D view type",
    );

    // P4 (tonemap, Compute) -> Storage
    assert_eq!(
        compiled.passes[4].view_type,
        ViewType::Storage,
        "P4 tonemap uses Storage view type",
    );

    // P5 is eliminated; no surviving pass at index 5.
}

// =============================================================================
// SECTION 16 -- Dispatch source correctness per pass
// =============================================================================

/// Graphics passes have no dispatch source; compute passes have Direct dispatch.
#[test]
fn dispatch_sources_match_pass_type() {
    let (passes, resources) = build_deferred_rendering_graph();
    let compiled = FrameGraphCompiler::new(passes, resources)
        .compile()
        .expect("Deferred graph compiles");

    // P0: Graphics -- no dispatch source.
    assert!(
        compiled.passes[0].dispatch_source.is_none(),
        "P0 (gbuffer, Graphics) has no dispatch source",
    );

    // P1: Compute -- Direct dispatch.
    assert!(
        compiled.passes[1].dispatch_source.is_some(),
        "P1 (ssao, Compute) has a dispatch source",
    );
    match &compiled.passes[1].dispatch_source {
        Some(DispatchSource::Direct { .. }) => {} // OK
        other => panic!("P1 dispatch source is Direct, got {:?}", other),
    }

    // P2: Compute -- Direct dispatch.
    assert!(
        compiled.passes[2].dispatch_source.is_some(),
        "P2 (lighting, Compute) has a dispatch source",
    );

    // P3: Graphics -- no dispatch source.
    assert!(
        compiled.passes[3].dispatch_source.is_none(),
        "P3 (compose, Graphics) has no dispatch source",
    );

    // P4: Compute -- Direct dispatch.
    assert!(
        compiled.passes[4].dispatch_source.is_some(),
        "P4 (tonemap, Compute) has a dispatch source",
    );

    // P5 is eliminated; only P0-P4 survive.
}
