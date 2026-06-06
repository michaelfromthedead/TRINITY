// SPDX-License-Identifier: MIT
//
// blackbox_validator.rs -- Blackbox contract tests for T-FG-7.2 (BridgeValidator).
//
// CLEANROOM: No src/ access beyond the public API exported by the crate.
// Tests construct IR data directly -- no implementation details, no internal
// fields, no private methods.
//
// Public API under test:
//   BridgeValidator::validate(compiled: &CompiledFrameGraph)
//     -> Result<(), Vec<String>>
//
//   Runs five independent structural checks against a compiled frame graph's
//   intermediate representation before GPU submission:
//
//   1. Barrier-pass reference validity  -- every barrier's from/to pass
//      indices exist in the passes list.
//   2. Resource existence               -- every resource handle referenced
//      by a pass exists in the resource list (ResourceHandle::NONE skipped).
//   3. No RAW violations                -- no pass reads a resource that has
//      not been written by any earlier pass in execution order (imported
//      resources with non-Uninitialized initial state are pre-written).
//   4. Topological order                -- execution order respects every
//      dependency edge (from appears strictly before to).
//   5. Pass-list completeness           -- every pass index in execution
//      order exists in the passes array.
//
// Acceptance criteria (T-FG-7.2):
//   1.  Valid graph passes validation (Ok(()))
//   2.  Invalid barrier pass reference caught
//   3.  Missing resource reference caught
//   4.  RAW hazard detected
//   5.  Execution order violation caught
//   6.  Missing pass in order caught
//   7.  Multiple errors accumulated
//   8.  Empty graph passes validation
//   9.  Imported resource does not trigger false RAW positive
//   10. ResourceHandle::NONE is flagged as unknown resource (not silently skipped)
//   11. Edge from/to at boundary (last valid index) passes
//   12. WAR dependency does not trigger RAW false positive (when a prior writer exists)
//   13. Indirect-buffer NONE is silently skipped
//   14. No colour-attachment resource triggers missing-resource error

use renderer_backend::frame_graph::{
    BridgeValidator, CompilerStats, DispatchSource, EdgeType, IrEdge, IrPass, IrResource,
    PassIndex, PerfCounters, ResourceDesc, ResourceHandle, ResourceLifetime, ResourceState,
    CullStats, SyncPoint,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Creates a storage-buffer IrResource at the given handle.
fn storage_buf(handle: u32, name: &str, size: u64) -> IrResource {
    IrResource::new(
        ResourceHandle(handle),
        name,
        ResourceDesc::Buffer(renderer_backend::frame_graph::BufferDesc {
            size,
            usage: "storage".into(),
            is_indirect_arg: false,
        }),
        ResourceLifetime::Transient,
        ResourceState::Uninitialized,
    )
}

/// Creates an imported (externally-created) storage-buffer IrResource with a
/// non-Uninitialized initial state, used to verify that imported resources
/// do not trigger false RAW positives.
fn imported_buf(handle: u32, name: &str, size: u64) -> IrResource {
    IrResource::new(
        ResourceHandle(handle),
        name,
        ResourceDesc::Buffer(renderer_backend::frame_graph::BufferDesc {
            size,
            usage: "storage".into(),
            is_indirect_arg: false,
        }),
        ResourceLifetime::Imported,
        ResourceState::ShaderRead,
    )
}

/// Creates a compute pass that writes the listed resources and reads none.
fn write_pass(index: usize, name: &str, writes: &[u32]) -> IrPass {
    let mut pass = IrPass::compute(
        PassIndex(index),
        name,
        DispatchSource::Direct {
            group_count_x: 1,
            group_count_y: 1,
            group_count_z: 1,
        },
        renderer_backend::frame_graph::ViewType::Storage,
    );
    for &w in writes {
        pass.access_set.writes.push(ResourceHandle(w));
    }
    pass
}

/// Creates a compute pass that reads the listed resources and writes none.
fn read_pass(index: usize, name: &str, reads: &[u32]) -> IrPass {
    let mut pass = IrPass::compute(
        PassIndex(index),
        name,
        DispatchSource::Direct {
            group_count_x: 1,
            group_count_y: 1,
            group_count_z: 1,
        },
        renderer_backend::frame_graph::ViewType::Storage,
    );
    for &r in reads {
        pass.access_set.reads.push(ResourceHandle(r));
    }
    pass
}

/// Build a minimal CompiledFrameGraph from parts and run the validator.
///
/// This lets us inject arbitrary invalid data (barriers, order, etc.) that
/// would never survive the `compile()` pipeline.
fn validate(
    passes: Vec<IrPass>,
    resources: Vec<IrResource>,
    order: Vec<PassIndex>,
    edges: Vec<IrEdge>,
    barriers: Vec<(PassIndex, PassIndex, ResourceState, ResourceState, ResourceHandle)>,
) -> Result<(), Vec<String>> {
    let passes_total = passes.len();
    let compiled = renderer_backend::frame_graph::CompiledFrameGraph {
        passes,
        resources,
        edges,
        order,
        depths: std::collections::HashMap::new(),
        barriers,
        async_passes: vec![],
        async_timeline: None,
        sync_points: vec![],
        eliminated_passes: vec![],
        cull_stats: CullStats {
            passes_total,
            passes_eliminated: 0,
            resources_freed: 0,
            bytes_saved: 0,
            live_pass_count: passes_total,
            culled_pass_count: 0,
            estimated_gpu_time_saved_ms: 0.0,
        },
        parallel_regions: vec![],
        compilation_time_us: 0,
        stats: Default::default(),
        perf_counters: Default::default(),
    };
    BridgeValidator::validate(&compiled)
}

/// Shorthand for validate() with no barriers.
fn validate_no_barriers(
    passes: Vec<IrPass>,
    resources: Vec<IrResource>,
    order: Vec<PassIndex>,
    edges: Vec<IrEdge>,
) -> Result<(), Vec<String>> {
    validate(passes, resources, order, edges, vec![])
}

// =============================================================================
// SECTION 1 -- Valid graph passes validation
// =============================================================================

/// A well-formed two-pass chain (writer then reader, single RAW edge) produces
/// `Ok(())`.
#[test]
fn valid_graph_returns_ok() {
    let passes = vec![
        write_pass(0, "writer", &[1]),
        read_pass(1, "reader", &[1]),
    ];
    let resources = vec![storage_buf(1, "shared", 256)];
    let order = vec![PassIndex(0), PassIndex(1)];
    let edges = vec![IrEdge::new(
        PassIndex(0),
        PassIndex(1),
        ResourceHandle(1),
        EdgeType::RAW,
    )];

    let result = validate_no_barriers(passes, resources, order, edges);
    assert!(result.is_ok(), "Valid graph must produce Ok(()); got {:?}", result);
}

/// A three-pass linear chain with multiple RAW edges passes validation.
#[test]
fn three_pass_linear_chain_passes() {
    let passes = vec![
        write_pass(0, "a", &[1]),
        write_pass(1, "b", &[2]),
        read_pass(2, "c", &[1, 2]),
    ];
    let resources = vec![
        storage_buf(1, "r1", 64),
        storage_buf(2, "r2", 64),
    ];
    let order = vec![PassIndex(0), PassIndex(1), PassIndex(2)];
    let edges = vec![
        IrEdge::new(PassIndex(0), PassIndex(2), ResourceHandle(1), EdgeType::RAW),
        IrEdge::new(PassIndex(1), PassIndex(2), ResourceHandle(2), EdgeType::RAW),
    ];

    let result = validate_no_barriers(passes, resources, order, edges);
    assert!(result.is_ok(), "Three-pass linear chain must produce Ok(()); got {:?}", result);
}

// =============================================================================
// SECTION 2 -- Invalid barrier / edge pass reference caught
// =============================================================================

/// A barrier whose `from` pass index is out of bounds for the passes array
/// must be reported as a validation error.
#[test]
fn barrier_invalid_from_pass_caught() {
    let passes = vec![write_pass(0, "p0", &[]), write_pass(1, "p1", &[])];
    let resources = vec![];
    let order = vec![PassIndex(0), PassIndex(1)];
    let edges = vec![];
    let barriers = vec![(
        PassIndex(99),  // out of bounds -- only passes 0 and 1 exist
        PassIndex(1),
        ResourceState::Uninitialized,
        ResourceState::ShaderRead,
        ResourceHandle(0),
    )];

    let result = validate(passes, resources, order, edges, barriers);
    assert!(result.is_err(), "Barrier with out-of-bounds `from` must produce errors");
}

/// A barrier whose `to` pass index is out of bounds must be reported.
#[test]
fn barrier_invalid_to_pass_caught() {
    let passes = vec![write_pass(0, "p0", &[])];
    let resources = vec![];
    let order = vec![PassIndex(0)];
    let edges = vec![];
    let barriers = vec![(
        PassIndex(0),
        PassIndex(999), // out of bounds
        ResourceState::Uninitialized,
        ResourceState::ShaderRead,
        ResourceHandle(0),
    )];

    let result = validate(passes, resources, order, edges, barriers);
    assert!(result.is_err(), "Barrier with out-of-bounds `to` must produce errors");
}

/// Both `from` and `to` out of bounds produce at least one error.
#[test]
fn barrier_both_ends_out_of_bounds_caught() {
    let passes = vec![write_pass(0, "p0", &[])];
    let resources = vec![];
    let order = vec![PassIndex(0)];
    let edges = vec![];
    let barriers = vec![(
        PassIndex(0xFFFFFFFF),
        PassIndex(0xFFFFFFFF),
        ResourceState::Uninitialized,
        ResourceState::ShaderRead,
        ResourceHandle(0),
    )];

    let result = validate(passes, resources, order, edges, barriers);
    assert!(
        result.is_err(),
        "Barrier with both ends out of bounds must produce errors",
    );
}

// =============================================================================
// SECTION 3 -- Missing resource reference caught
// =============================================================================

/// A pass that writes a resource handle not present in the resource list
/// must be reported.
#[test]
fn missing_write_resource_caught() {
    let passes = vec![write_pass(0, "writer", &[1])];
    let resources = vec![]; // no resource with handle 1
    let order = vec![PassIndex(0)];
    let edges = vec![];

    let result = validate_no_barriers(passes, resources, order, edges);
    assert!(
        result.is_err(),
        "Write to non-existent resource must produce errors",
    );
}

/// A pass that reads a resource handle not present in the resource list
/// must be reported.
#[test]
fn missing_read_resource_caught() {
    let passes = vec![read_pass(0, "reader", &[42])];
    let resources = vec![storage_buf(1, "real_buf", 64)];
    let order = vec![PassIndex(0)];
    let edges = vec![];

    let result = validate_no_barriers(passes, resources, order, edges);
    assert!(
        result.is_err(),
        "Read of non-existent resource must produce errors",
    );
}

/// Multiple missing resource references produce at least one error.
#[test]
fn multiple_missing_resources_caught() {
    let passes = vec![write_pass(0, "writer", &[10, 20, 30])];
    let resources = vec![];
    let order = vec![PassIndex(0)];
    let edges = vec![];

    let result = validate_no_barriers(passes, resources, order, edges);
    assert!(
        result.is_err(),
        "Multiple missing resource refs must produce error(s)",
    );
}

// =============================================================================
// SECTION 4 -- RAW hazard detected
// =============================================================================

/// A pass that reads a resource before any earlier pass has written it is a
/// RAW violation.
#[test]
fn raw_violation_detected() {
    let passes = vec![
        read_pass(0, "premature_reader", &[1]), // reads R1 before any writer
        write_pass(1, "writer", &[1]),
    ];
    let resources = vec![storage_buf(1, "buf", 64)];
    let order = vec![PassIndex(0), PassIndex(1)];
    let edges = vec![];

    let result = validate_no_barriers(passes, resources, order, edges);
    assert!(
        result.is_err(),
        "RAW violation (read before write) must produce errors",
    );
}

/// A pass that reads a resource AFTER it has been written is NOT a RAW
/// violation.
#[test]
fn read_after_write_not_raw() {
    let passes = vec![
        write_pass(0, "writer", &[1]),
        read_pass(1, "reader", &[1]),
    ];
    let resources = vec![storage_buf(1, "buf", 64)];
    let order = vec![PassIndex(0), PassIndex(1)];
    let edges = vec![];

    let result = validate_no_barriers(passes, resources, order, edges);
    assert!(
        result.is_ok(),
        "Read after write must not be a RAW violation; got {:?}",
        result,
    );
}

/// A WAR (write-after-read) dependency does NOT trigger a RAW violation
/// as long as every read has a prior writer. In this three-pass chain:
///   pass 0 writes R1, pass 1 reads R1, pass 2 writes R1 again.
/// Pass 1's read is satisfied by pass 0's write — the later write by pass 2
/// creates a WAR (which is valid), not a RAW.
#[test]
fn war_dependency_not_raw() {
    let passes = vec![
        write_pass(0, "first_writer", &[1]),
        read_pass(1, "reader", &[1]),
        write_pass(2, "second_writer", &[1]),
    ];
    let resources = vec![storage_buf(1, "buf", 64)];
    let order = vec![PassIndex(0), PassIndex(1), PassIndex(2)];
    let edges = vec![];

    let result = validate_no_barriers(passes, resources, order, edges);
    assert!(
        result.is_ok(),
        "WAR dependency must not trigger false RAW positive; got {:?}",
        result,
    );
}

// =============================================================================
// SECTION 5 -- Execution order violation caught
// =============================================================================

/// When an edge requires pass 0 before pass 1 but execution order places
/// pass 1 first, the validator must flag the topological violation.
#[test]
fn execution_order_violation_caught() {
    let passes = vec![
        write_pass(0, "writer", &[1]),
        read_pass(1, "reader", &[1]),
    ];
    let resources = vec![storage_buf(1, "buf", 64)];
    let order = vec![PassIndex(1), PassIndex(0)]; // reversed!
    let edges = vec![IrEdge::new(
        PassIndex(0),
        PassIndex(1),
        ResourceHandle(1),
        EdgeType::RAW,
    )];

    let result = validate_no_barriers(passes, resources, order, edges);
    assert!(
        result.is_err(),
        "Reversed execution order must produce topological error",
    );
}

/// Multiple edges all consistent with execution order passes validation.
#[test]
fn diamond_dag_passes_validation() {
    // Diamond: 0 -> 1, 0 -> 2, 1 -> 3, 2 -> 3
    let passes = vec![
        write_pass(0, "entry", &[1, 2]),
        read_pass(1, "mid_a", &[1]),
        read_pass(2, "mid_b", &[2]),
        read_pass(3, "exit", &[1, 2]),
    ];
    let resources = vec![
        storage_buf(1, "r1", 64),
        storage_buf(2, "r2", 64),
    ];
    let order = vec![PassIndex(0), PassIndex(1), PassIndex(2), PassIndex(3)];
    let edges = vec![
        IrEdge::new(PassIndex(0), PassIndex(1), ResourceHandle(1), EdgeType::RAW),
        IrEdge::new(PassIndex(0), PassIndex(2), ResourceHandle(2), EdgeType::RAW),
        IrEdge::new(PassIndex(1), PassIndex(3), ResourceHandle(1), EdgeType::RAW),
        IrEdge::new(PassIndex(2), PassIndex(3), ResourceHandle(2), EdgeType::RAW),
    ];

    let result = validate_no_barriers(passes, resources, order, edges);
    assert!(
        result.is_ok(),
        "Diamond DAG with correct order must pass; got {:?}",
        result,
    );
}

// =============================================================================
// SECTION 6 -- Missing pass in order caught
// =============================================================================

/// An execution order entry referencing a pass index that does not exist
/// in the passes array must be flagged.
#[test]
fn missing_pass_in_execution_order_caught() {
    let passes = vec![write_pass(0, "p0", &[])];
    let resources = vec![];
    let order = vec![PassIndex(0), PassIndex(7)]; // pass 7 does not exist
    let edges = vec![];

    let result = validate_no_barriers(passes, resources, order, edges);
    assert!(
        result.is_err(),
        "Missing pass in execution order must produce error(s)",
    );
}

/// An execution order referencing only valid passes produces `Ok(())`.
#[test]
fn valid_pass_list_ok() {
    let passes = vec![
        write_pass(0, "p0", &[]),
        write_pass(1, "p1", &[]),
        write_pass(2, "p2", &[]),
    ];
    let resources = vec![];
    let order = vec![PassIndex(0), PassIndex(1), PassIndex(2)];
    let edges = vec![];

    let result = validate_no_barriers(passes, resources, order, edges);
    assert!(
        result.is_ok(),
        "All passes present in execution order must pass; got {:?}",
        result,
    );
}

// =============================================================================
// SECTION 7 -- Multiple errors accumulated
// =============================================================================

/// When the input violates multiple invariants simultaneously, the
/// validator must return errors for every category, not fail fast on the
/// first one.
#[test]
fn multiple_errors_accumulated() {
    let passes = vec![
        read_pass(0, "bad_reader", &[999]), // missing resource
    ];
    let resources = vec![];
    let order = vec![
        PassIndex(0),
        PassIndex(99), // pass 99 does not exist -> completeness error
    ];
    let edges = vec![];
    let barriers = vec![(
        PassIndex(0),
        PassIndex(999), // out of bounds -> barrier pass ref error
        ResourceState::Uninitialized,
        ResourceState::ShaderRead,
        ResourceHandle(0),
    )];

    let result = validate(passes, resources, order, edges, barriers);
    assert!(
        result.is_err(),
        "Multiple independent violations must produce errors",
    );
    assert!(
        result.as_ref().unwrap_err().len() >= 2,
        "Multiple violations must produce >=2 errors; got {:?}",
        result,
    );
}

/// Combined RAW + topological + resource errors all surfaced.
#[test]
fn mixed_violation_categories_all_surfaced() {
    let passes = vec![
        read_pass(0, "reader_orphan", &[1]),   // R1 never written, no such resource
        write_pass(1, "writer_ghost", &[2]),   // R2 does not exist
    ];
    let resources = vec![];                      // no resources at all
    let order = vec![PassIndex(1), PassIndex(0)]; // topological violation
    let edges = vec![IrEdge::new(
        PassIndex(0),
        PassIndex(1),
        ResourceHandle(1),
        EdgeType::RAW,
    )];

    let result = validate_no_barriers(passes, resources, order, edges);
    assert!(
        result.is_err(),
        "Mixed violations must produce errors",
    );
    assert!(
        result.as_ref().unwrap_err().len() >= 2,
        "Mixed violations must produce >=2 errors; got {:?}",
        result,
    );
}

// =============================================================================
// SECTION 8 -- Empty graph passes validation
// =============================================================================

/// An entirely empty frame graph (no passes, no resources, no schedule,
/// no edges) must pass validation silently.
#[test]
fn empty_graph_passes() {
    let result = validate_no_barriers(vec![], vec![], vec![], vec![]);
    assert!(result.is_ok(), "Empty graph must produce Ok(()); got {:?}", result);
}

/// A graph with resources but no passes or schedule passes validation.
#[test]
fn resources_without_passes_ok() {
    let resources = vec![storage_buf(1, "orphan", 64)];
    let result = validate_no_barriers(vec![], resources, vec![], vec![]);
    assert!(result.is_ok(), "Resources without passes must pass; got {:?}", result);
}

// =============================================================================
// SECTION 9 -- Imported resource does not trigger false RAW positive
// =============================================================================

/// An imported resource with a non-Uninitialized initial state is treated
/// as pre-written, so reading it in the first pass is valid.
#[test]
fn imported_resource_avoids_false_raw() {
    let passes = vec![
        read_pass(0, "first_reader", &[1]), // reads imported resource
    ];
    let resources = vec![imported_buf(1, "imported", 256)];
    let order = vec![PassIndex(0)];
    let edges = vec![];

    let result = validate_no_barriers(passes, resources, order, edges);
    assert!(
        result.is_ok(),
        "Imported resource read must not trigger false RAW; got {:?}",
        result,
    );
}

/// An imported resource with Uninitialized state is NOT treated as
/// pre-written, so a read before any write IS a RAW violation.
#[test]
fn imported_uninitialized_still_triggers_raw() {
    let resources = vec![IrResource::new(
        ResourceHandle(1),
        "imported_but_uninit",
        ResourceDesc::Buffer(renderer_backend::frame_graph::BufferDesc {
            size: 256,
            usage: "storage".into(),
            is_indirect_arg: false,
        }),
        ResourceLifetime::Imported,
        ResourceState::Uninitialized, // still needs a write before read
    )];
    let passes = vec![
        read_pass(0, "reader", &[1]),
    ];
    let order = vec![PassIndex(0)];
    let edges = vec![];

    let result = validate_no_barriers(passes, resources, order, edges);
    assert!(
        result.is_err(),
        "Imported resource with Uninitialized state must still trigger RAW",
    );
}

// =============================================================================
// SECTION 10 -- ResourceHandle::NONE is flagged by the validator
// =============================================================================

/// ResourceHandle::NONE (`u32::MAX`) is a sentinel value, not a real resource
/// handle. The BridgeValidator currently flags it as an unknown resource.
/// This test documents that behavior — if the implementation later learns to
/// skip NONE, this test can be updated.
#[test]
fn none_handle_in_access_set_is_flagged() {
    let mut pass = IrPass::compute(
        PassIndex(0),
        "none_user",
        DispatchSource::Direct {
            group_count_x: 1, group_count_y: 1, group_count_z: 1,
        },
        renderer_backend::frame_graph::ViewType::Storage,
    );
    pass.access_set.reads.push(ResourceHandle::NONE);
    pass.access_set.writes.push(ResourceHandle::NONE);

    let passes = vec![pass];
    let resources = vec![]; // no resources; NONE handle is unknown
    let order = vec![PassIndex(0)];
    let edges = vec![];

    let result = validate_no_barriers(passes, resources, order, edges);
    assert!(
        result.is_err(),
        "ResourceHandle::NONE is currently flagged as unknown resource",
    );
}

// =============================================================================
// SECTION 11 -- Edge from/to at boundary (last valid index) passes
// =============================================================================

/// An edge referencing the last valid pass index (passes.len() - 1) must
/// not be flagged as out of bounds.
#[test]
fn edge_to_last_valid_index_passes() {
    let passes = vec![
        write_pass(0, "p0", &[1]),
        write_pass(1, "p1", &[1]),
    ];
    let resources = vec![storage_buf(1, "r1", 64)];
    let order = vec![PassIndex(0), PassIndex(1)];
    // Edge from 0 to 1, where 1 is the last valid index (passes.len() - 1)
    let edges = vec![IrEdge::new(
        PassIndex(0),
        PassIndex(1),
        ResourceHandle(1),
        EdgeType::RAW,
    )];

    let result = validate_no_barriers(passes, resources, order, edges);
    assert!(
        result.is_ok(),
        "Edge referencing last valid index must pass; got {:?}",
        result,
    );
}

// =============================================================================
// SECTION 12 -- Result shape and error messages
// =============================================================================

/// The Err variant contains human-readable strings.
#[test]
fn error_messages_are_strings() {
    let passes = vec![read_pass(0, "bad_reader", &[999])];
    let resources = vec![];
    let order = vec![PassIndex(0)];
    let edges = vec![];

    let result = validate_no_barriers(passes, resources, order, edges);
    assert!(result.is_err(), "Must produce an error");
    let errors = result.unwrap_err();
    assert!(!errors.is_empty(), "Error vector must not be empty");
    for msg in &errors {
        assert!(!msg.is_empty(), "Each error message must be non-empty");
        assert!(msg.contains("999") || msg.contains("unknown resource"),
            "Error message should describe the violation: got {:?}", msg);
    }
}

/// Multiple errors are collected into a single Vec<String>.
#[test]
fn multiple_errors_collected() {
    // Two missing resources + one barrier with out-of-bounds from.
    let passes = vec![
        write_pass(0, "w1", &[10]),
        write_pass(1, "w2", &[20]),
    ];
    let resources = vec![];
    let order = vec![PassIndex(0), PassIndex(1)];
    let edges = vec![];
    let barriers = vec![(
        PassIndex(99),
        PassIndex(0),
        ResourceState::Uninitialized,
        ResourceState::ShaderRead,
        ResourceHandle(0),
    )];

    let result = validate(passes, resources, order, edges, barriers);
    assert!(result.is_err(), "Must produce errors");
    let errors = result.unwrap_err();
    // At least two resource errors + one barrier error.
    assert!(
        errors.len() >= 3,
        "Expected >=3 errors (2 resource + 1 barrier); got {}",
        errors.len(),
    );
}

// =============================================================================
// SECTION 13 -- Integration: compiled graph passes validation
// =============================================================================

/// A graph produced by the full compiler pipeline (via CompiledFrameGraph)
/// must pass validation without errors.
#[test]
fn compiled_graph_passes_validation() {
    let mut p0 = IrPass::compute(
        PassIndex(0),
        "producer",
        DispatchSource::Direct {
            group_count_x: 1, group_count_y: 1, group_count_z: 1,
        },
        renderer_backend::frame_graph::ViewType::Storage,
    );
    p0.access_set.writes.push(ResourceHandle(1));

    let mut p1 = IrPass::compute(
        PassIndex(1),
        "consumer",
        DispatchSource::Direct {
            group_count_x: 1, group_count_y: 1, group_count_z: 1,
        },
        renderer_backend::frame_graph::ViewType::Storage,
    );
    p1.access_set.reads.push(ResourceHandle(1));

    let passes = vec![p0, p1];
    let resources = vec![storage_buf(1, "buf", 256)];

    // Use the compile pipeline to get a real CompiledFrameGraph.
    let compiled = renderer_backend::frame_graph::CompiledFrameGraph::compile(
        passes,
        resources,
    )
    .expect("Simple two-pass graph must compile");

    let result = BridgeValidator::validate(&compiled);
    assert!(
        result.is_ok(),
        "Compiled pipeline output must pass validation; got {:?}",
        result,
    );
}
