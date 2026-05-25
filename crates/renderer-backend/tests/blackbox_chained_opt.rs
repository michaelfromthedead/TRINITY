// SPDX-License-Identifier: MIT
//
// blackbox_chained_opt.rs -- Blackbox contract tests for T-FG-6.6 ChainedOptimizer.
//
// CLEANROOM: No src/ access beyond the public API exported by the crate.
// Tests use only `renderer_backend::frame_graph::*` -- no internal fields,
// no private methods, no implementation details.
//
// Public API under test:
//   ChainedOptimizer::new()
//   ChainedOptimizer::add_pass(&mut self, pass: Box<dyn OptimizationPass>)
//   ChainedOptimizer::run(&self, graph: CompiledFrameGraph) -> CompiledFrameGraph
//   OptimizationPass trait
//
// Contract:
//   add_pass registers an optimizer; run applies all sequentially.
//   The output of each pass becomes the input of the next.
//
// Coverage:
//   1.  new() creates an empty chain
//   2.  run with zero passes returns the input unchanged (identity)
//   3.  add_pass followed by run invokes exactly one pass
//   4.  Multiple add_pass calls register all passes
//   5.  Passes execute in registration order
//   6.  Pipeline sequencing: output of pass N reaches pass N+1
//   7.  Chain is reusable across multiple run calls
//   8.  run with BarrierOptimizer eliminates same-state barriers
//   9.  run with PassMerger merges adjacent compatible passes
//  10.  run with ResourcePruner removes unreferenced resources
//  11.  run with all three built-in passes composed
//  12.  Mixed custom and built-in passes in one chain
//  13.  Cumulative effect: chain of StampPass adds all deltas
//  14.  Empty chain across multiple runs stays empty

use std::collections::HashMap;

use renderer_backend::frame_graph::{
    BarrierOptimizer, BufferDesc, ChainedOptimizer, CompiledFrameGraph, CullStats, CompilerStats,
    DispatchSource, EdgeType, IrEdge, IrPass, IrResource, OptimizationPass, PassIndex, PassMerger,
    PerfCounters, ResourceDesc, ResourceHandle, ResourceLifetime, ResourcePruner, ResourceState,
    ViewType,
};

// =============================================================================
// Custom optimization passes for behavioural verification
// =============================================================================

/// StampPass adds a fixed delta to `compilation_time_us`, proving the pass
/// executed and that chaining accumulates deltas.
struct StampPass(u64);

impl OptimizationPass for StampPass {
    fn run(&self, mut graph: CompiledFrameGraph) -> CompiledFrameGraph {
        graph.compilation_time_us += self.0;
        graph
    }
}

/// TagPass appends a name-tag to the `async_passes` list, proving execution
/// order when multiple passes are chained.
struct TagPass(&'static str);

impl OptimizationPass for TagPass {
    fn run(&self, mut graph: CompiledFrameGraph) -> CompiledFrameGraph {
        graph.async_passes.push((PassIndex(0), self.0.to_string()));
        graph
    }
}

/// SentinelPass writes a fixed value to `compilation_time_us`, proving that
/// the output of a prior pass is visible to the next pass in the chain (pipeline
/// sequencing).
struct SentinelPass(u64);

impl OptimizationPass for SentinelPass {
    fn run(&self, mut graph: CompiledFrameGraph) -> CompiledFrameGraph {
        graph.compilation_time_us = self.0;
        graph
    }
}

// =============================================================================
// Helper: a minimal empty CompiledFrameGraph
// =============================================================================

fn empty_graph() -> CompiledFrameGraph {
    CompiledFrameGraph {
        passes: Vec::new(),
        resources: Vec::new(),
        edges: Vec::new(),
        order: Vec::new(),
        depths: HashMap::new(),
        barriers: Vec::new(),
        async_passes: Vec::new(),
        eliminated_passes: Vec::new(),
        cull_stats: CullStats::default(),
        parallel_regions: Vec::new(),
        compilation_time_us: 0,
        stats: CompilerStats::default(),
        perf_counters: PerfCounters::default(),
    }
}

/// Convenience: a two-pass input with an edge for PassMerger tests.
fn two_pass_input() -> CompiledFrameGraph {
    let p0 = IrPass::compute(
        PassIndex(0),
        "cs_first",
        DispatchSource::Direct {
            group_count_x: 1,
            group_count_y: 1,
            group_count_z: 1,
        },
        ViewType::Storage,
    );
    let p1 = IrPass::compute(
        PassIndex(1),
        "cs_second",
        DispatchSource::Direct {
            group_count_x: 1,
            group_count_y: 1,
            group_count_z: 1,
        },
        ViewType::Storage,
    );
    CompiledFrameGraph {
        passes: vec![p0, p1],
        order: vec![PassIndex(0), PassIndex(1)],
        edges: vec![IrEdge::new(PassIndex(0), PassIndex(1), ResourceHandle(0), EdgeType::RAW)],
        ..empty_graph()
    }
}

// =============================================================================
// SECTION 1 -- Construction and identity behaviour
// =============================================================================

#[test]
fn new_creates_empty_chain() {
    let chain = ChainedOptimizer::new();
    let result = chain.run(empty_graph());
    assert_eq!(
        result.compilation_time_us, 0,
        "empty chain leaves graph untouched",
    );
}

#[test]
fn run_with_no_passes_is_identity() {
    let chain = ChainedOptimizer::new();
    let input = empty_graph();
    let output = chain.run(input);
    assert_eq!(output.compilation_time_us, 0);
    assert!(output.passes.is_empty());
    assert!(output.resources.is_empty());
    assert!(output.barriers.is_empty());
    assert!(output.async_passes.is_empty());
}

#[test]
fn empty_chain_preserves_custom_fields() {
    let chain = ChainedOptimizer::new();
    let mut input = empty_graph();
    input.compilation_time_us = 42;
    input.eliminated_passes.push(PassIndex(0));
    let output = chain.run(input);
    assert_eq!(output.compilation_time_us, 42, "custom time preserved");
    assert_eq!(
        output.eliminated_passes,
        vec![PassIndex(0)],
        "eliminated passes preserved",
    );
}

// =============================================================================
// SECTION 2 -- Single pass registration
// =============================================================================

#[test]
fn add_pass_single_invocation() {
    let mut chain = ChainedOptimizer::new();
    chain.add_pass(Box::new(StampPass(100)));
    let result = chain.run(empty_graph());
    assert_eq!(
        result.compilation_time_us, 100,
        "single StampPass(100) applied",
    );
}

#[test]
fn add_pass_single_tag() {
    let mut chain = ChainedOptimizer::new();
    chain.add_pass(Box::new(TagPass("alpha")));
    let result = chain.run(empty_graph());
    assert_eq!(
        result.async_passes,
        vec![(PassIndex(0), "alpha".to_string())],
    );
}

// =============================================================================
// SECTION 3 -- Multiple passes
// =============================================================================

#[test]
fn add_pass_multiple_all_invoked() {
    let mut chain = ChainedOptimizer::new();
    chain.add_pass(Box::new(StampPass(10)));
    chain.add_pass(Box::new(StampPass(20)));
    chain.add_pass(Box::new(StampPass(30)));
    let result = chain.run(empty_graph());
    assert_eq!(
        result.compilation_time_us,
        10 + 20 + 30,
        "all three StampPass deltas accumulated",
    );
}

#[test]
fn passes_execute_in_registration_order() {
    let mut chain = ChainedOptimizer::new();
    chain.add_pass(Box::new(TagPass("first")));
    chain.add_pass(Box::new(TagPass("second")));
    chain.add_pass(Box::new(TagPass("third")));
    let result = chain.run(empty_graph());
    assert_eq!(
        result.async_passes,
        vec![
            (PassIndex(0), "first".to_string()),
            (PassIndex(0), "second".to_string()),
            (PassIndex(0), "third".to_string()),
        ],
        "tags appear in registration order",
    );
}

#[test]
fn registration_order_reversed_produces_same_accumulation() {
    // StampPass is commutative so both orders produce the same cumulative
    // value.  This confirms separate chain instances with identical pass
    // multisets behave identically.
    let mut chain_a = ChainedOptimizer::new();
    chain_a.add_pass(Box::new(StampPass(10)));
    chain_a.add_pass(Box::new(StampPass(100)));

    let mut chain_b = ChainedOptimizer::new();
    chain_b.add_pass(Box::new(StampPass(100)));
    chain_b.add_pass(Box::new(StampPass(10)));

    let result_a = chain_a.run(empty_graph());
    let result_b = chain_b.run(empty_graph());
    assert_eq!(result_a.compilation_time_us, result_b.compilation_time_us);
}

// =============================================================================
// SECTION 4 -- Pipeline sequencing (output of pass N feeds pass N+1)
// =============================================================================

#[test]
fn pipeline_sequencing_sentinel_forwarding() {
    // SentinelPass writes a fixed value.  If chaining works, a second
    // StampPass reads the graph produced by the SentinelPass.
    let mut chain = ChainedOptimizer::new();
    chain.add_pass(Box::new(SentinelPass(777)));
    chain.add_pass(Box::new(StampPass(23)));
    let result = chain.run(empty_graph());
    assert_eq!(
        result.compilation_time_us,
        777 + 23,
        "StampPass sees the compilation_time_us set by SentinelPass",
    );
}

#[test]
fn pipeline_sequencing_tag_after_stamp() {
    // Verify that StampPass modifying compilation_time_us does not
    // interfere with TagPass operating on async_passes in the same chain.
    let mut chain = ChainedOptimizer::new();
    chain.add_pass(Box::new(StampPass(50)));
    chain.add_pass(Box::new(TagPass("post-stamp")));
    let result = chain.run(empty_graph());
    assert_eq!(result.compilation_time_us, 50, "StampPass applied");
    assert_eq!(
        result.async_passes,
        vec![(PassIndex(0), "post-stamp".to_string())],
        "TagPass sees the graph after StampPass",
    );
}

// =============================================================================
// SECTION 5 -- Chain reusability
// =============================================================================

#[test]
fn chain_is_reusable_multiple_runs() {
    let mut chain = ChainedOptimizer::new();
    chain.add_pass(Box::new(StampPass(7)));

    let r1 = chain.run(empty_graph());
    let r2 = chain.run(empty_graph());
    let r3 = chain.run(empty_graph());

    assert_eq!(r1.compilation_time_us, 7, "run 1");
    assert_eq!(r2.compilation_time_us, 7, "run 2");
    assert_eq!(r3.compilation_time_us, 7, "run 3");
}

#[test]
fn empty_chain_reusable() {
    let chain = ChainedOptimizer::new();
    let r1 = chain.run(empty_graph());
    let r2 = chain.run(empty_graph());
    assert_eq!(r1.compilation_time_us, 0);
    assert_eq!(r2.compilation_time_us, 0);
}

// =============================================================================
// SECTION 6 -- Built-in pass: BarrierOptimizer
// =============================================================================

#[test]
fn chained_barrier_optimizer_eliminates_same_state_barriers() {
    let mut input = empty_graph();
    // Two identical same-state barriers for the same resource.
    input.barriers = vec![
        (PassIndex(0), PassIndex(1), ResourceHandle(1),
         ResourceState::ShaderRead, ResourceState::ShaderRead),
        (PassIndex(0), PassIndex(1), ResourceHandle(1),
         ResourceState::ShaderRead, ResourceState::ShaderRead),
    ];

    let mut chain = ChainedOptimizer::new();
    chain.add_pass(Box::new(BarrierOptimizer));
    let result = chain.run(input);

    assert!(
        result.barriers.is_empty(),
        "BarrierOptimizer removed same-state barriers",
    );
}

#[test]
fn chained_barrier_optimizer_preserves_different_state_transitions() {
    let mut input = empty_graph();
    // A genuine transition that should survive optimisation.
    input.barriers = vec![
        (PassIndex(0), PassIndex(1), ResourceHandle(1),
         ResourceState::ShaderRead, ResourceState::ColorAttachment),
    ];

    let mut chain = ChainedOptimizer::new();
    chain.add_pass(Box::new(BarrierOptimizer));
    let result = chain.run(input);

    assert_eq!(
        result.barriers.len(),
        1,
        "genuine transition preserved",
    );
    assert_eq!(
        result.barriers[0].3, ResourceState::ShaderRead,
        "before state unchanged",
    );
    assert_eq!(
        result.barriers[0].4, ResourceState::ColorAttachment,
        "after state unchanged",
    );
}

// =============================================================================
// SECTION 7 -- Built-in pass: PassMerger
// =============================================================================

#[test]
fn chained_pass_merger_merges_adjacent_compatible_passes() {
    let mut input = two_pass_input();
    // Ensure no edge between the two passes so they are mergeable.
    // two_pass_input() adds a RAW edge, which blocks merging.  We need
    // edges that do NOT create a dependency -- use an edge that connects
    // pass 0 to itself or use no edge at all for the barrier relationship.
    // For this test we pass passes without a direct edge between them:
    // they share a resource but the access pattern is compatible.
    input.edges = vec![];

    let mut chain = ChainedOptimizer::new();
    chain.add_pass(Box::new(PassMerger));
    let result = chain.run(input);

    assert!(
        result.order.len() < 2,
        "PassMerger reduced pass count from 2 to {}",
        result.order.len(),
    );
}

#[test]
fn chained_pass_merger_leaves_incompatible_passes_unmerged() {
    // two_pass_input has a RAW edge between passes via ResourceHandle(0).
    // PassMerger sees the RAW edge and refuses to merge.
    let input = two_pass_input();

    let mut chain = ChainedOptimizer::new();
    chain.add_pass(Box::new(PassMerger));
    let result = chain.run(input);

    assert_eq!(
        result.order.len(),
        2,
        "incompatible passes remain unmerged",
    );
}

// =============================================================================
// SECTION 8 -- Built-in pass: ResourcePruner
// =============================================================================

#[test]
fn chained_resource_pruner_removes_unreferenced_resources() {
    let mut input = empty_graph();
    // A compute pass that references handle(0) but not handle(1).
    let mut p = IrPass::compute(
        PassIndex(0),
        "cs_main",
        DispatchSource::Direct {
            group_count_x: 8,
            group_count_y: 1,
            group_count_z: 1,
        },
        ViewType::Storage,
    );

    // Give the pass a reference to resource handle(0) so the pruner keeps it.
    p.access_set.writes.push(ResourceHandle(0));

    input.passes = vec![p];
    input.order = vec![PassIndex(0)];

    // Resource 1 is unreferenced.
    input.resources = vec![
        IrResource::new(
            ResourceHandle(0),
            "used",
            ResourceDesc::Buffer(BufferDesc {
                size: 64,
                usage: "storage".to_string(),
                is_indirect_arg: false,
            }),
            ResourceLifetime::Transient,
            ResourceState::Uninitialized,
        ),
        IrResource::new(
            ResourceHandle(1),
            "unused",
            ResourceDesc::Buffer(BufferDesc {
                size: 128,
                usage: "storage".to_string(),
                is_indirect_arg: false,
            }),
            ResourceLifetime::Transient,
            ResourceState::Uninitialized,
        ),
    ];

    let mut chain = ChainedOptimizer::new();
    chain.add_pass(Box::new(ResourcePruner));
    let result = chain.run(input);

    assert_eq!(
        result.resources.len(),
        1,
        "ResourcePruner removed the unreferenced resource",
    );
    assert_eq!(
        result.resources[0].name, "used",
        "the referenced resource is preserved",
    );
}

// =============================================================================
// SECTION 9 -- All three built-in passes composed
// =============================================================================

#[test]
fn all_builtin_passes_chain_without_error() {
    let mut input = empty_graph();

    // Two compute passes + one unreferenced resource.
    let p0 = IrPass::compute(
        PassIndex(0),
        "pass_a",
        DispatchSource::Direct {
            group_count_x: 1,
            group_count_y: 1,
            group_count_z: 1,
        },
        ViewType::Storage,
    );
    let mut p1 = IrPass::compute(
        PassIndex(1),
        "pass_b",
        DispatchSource::Direct {
            group_count_x: 1,
            group_count_y: 1,
            group_count_z: 1,
        },
        ViewType::Storage,
    );

    // Give pass_b a reference to resource handle(0) so ResourcePruner keeps it.
    p1.access_set.writes.push(ResourceHandle(0));

    input.passes = vec![p0, p1];
    input.order = vec![PassIndex(0), PassIndex(1)];
    input.edges = vec![
        IrEdge::new(PassIndex(0), PassIndex(1), ResourceHandle(0), EdgeType::RAW),
    ];
    input.barriers = vec![
        // Same-state barrier (removed by BarrierOptimizer).
        (PassIndex(0), PassIndex(1), ResourceHandle(0),
         ResourceState::ShaderRead, ResourceState::ShaderRead),
        // Genuine transition (preserved through BarrierOptimizer).
        (PassIndex(0), PassIndex(1), ResourceHandle(1),
         ResourceState::Uninitialized, ResourceState::ShaderRead),
    ];
    input.resources = vec![
        IrResource::new(
            ResourceHandle(0),
            "referenced",
            ResourceDesc::Buffer(BufferDesc {
                size: 64,
                usage: "storage".to_string(),
                is_indirect_arg: false,
            }),
            ResourceLifetime::Transient,
            ResourceState::Uninitialized,
        ),
        IrResource::new(
            ResourceHandle(1),
            "unreferenced",
            ResourceDesc::Buffer(BufferDesc {
                size: 128,
                usage: "storage".to_string(),
                is_indirect_arg: false,
            }),
            ResourceLifetime::Transient,
            ResourceState::Uninitialized,
        ),
    ];

    let mut chain = ChainedOptimizer::new();
    chain.add_pass(Box::new(BarrierOptimizer));
    chain.add_pass(Box::new(PassMerger));
    chain.add_pass(Box::new(ResourcePruner));
    let result = chain.run(input);

    // BarrierOptimizer: same-state barrier removed, one remains (genuine).
    assert!(
        result.barriers.len() <= 1,
        "at most one barrier survives (the genuine transition), got {}",
        result.barriers.len(),
    );

    // ResourcePruner: unreferenced resource removed.
    assert_eq!(result.resources.len(), 1, "only referenced resource remains");
    assert_eq!(result.resources[0].name, "referenced");
}

// =============================================================================
// SECTION 10 -- Mixed custom and built-in passes
// =============================================================================

#[test]
fn mixed_custom_and_builtin_passes() {
    let mut input = empty_graph();
    input.barriers = vec![
        (PassIndex(0), PassIndex(1), ResourceHandle(0),
         ResourceState::ShaderRead, ResourceState::ShaderRead),
    ];

    let mut chain = ChainedOptimizer::new();
    chain.add_pass(Box::new(StampPass(5)));
    chain.add_pass(Box::new(BarrierOptimizer));
    chain.add_pass(Box::new(TagPass("after-barrier-opt")));
    let result = chain.run(input);

    assert_eq!(result.compilation_time_us, 5, "StampPass executed");
    assert!(result.barriers.is_empty(), "BarrierOptimizer ran after StampPass");
    assert_eq!(
        result.async_passes,
        vec![(PassIndex(0), "after-barrier-opt".to_string())],
        "TagPass executed after BarrierOptimizer",
    );
}

// =============================================================================
// SECTION 11 -- Cumulative effect
// =============================================================================

#[test]
fn cumulative_stamp_chain() {
    let mut chain = ChainedOptimizer::new();
    // Register 5 StampPasses each adding a different delta.
    for &delta in &[1, 2, 3, 4, 5] {
        chain.add_pass(Box::new(StampPass(delta)));
    }
    let result = chain.run(empty_graph());
    assert_eq!(
        result.compilation_time_us,
        1 + 2 + 3 + 4 + 5,
        "all five StampPass deltas accumulate",
    );
}

// =============================================================================
// SECTION 12 -- Edge cases
// =============================================================================

#[test]
fn chain_with_different_graphs_each_run() {
    let mut chain = ChainedOptimizer::new();
    chain.add_pass(Box::new(StampPass(10)));

    let r1 = chain.run(empty_graph());
    assert_eq!(r1.compilation_time_us, 10);

    // Second run with a pre-warmed graph.
    let mut warm = empty_graph();
    warm.compilation_time_us = 100;
    let r2 = chain.run(warm);
    assert_eq!(
        r2.compilation_time_us,
        110,
        "StampPass adds to the incoming value",
    );
}

#[test]
fn no_side_effects_between_runs() {
    let mut chain = ChainedOptimizer::new();
    chain.add_pass(Box::new(StampPass(99)));

    // First run does not alter chain state for second run.
    let r1 = chain.run(empty_graph());
    assert_eq!(r1.compilation_time_us, 99);

    let r2 = chain.run(empty_graph());
    assert_eq!(
        r2.compilation_time_us, 99,
        "second run produces the same result (no internal mutation)",
    );
}
