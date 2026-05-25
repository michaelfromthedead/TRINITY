// Blackbox contract tests for T-FG-7.5 (Schedule emit bridge).
//
// CLEANROOM: No src/ access beyond the public API exported by the crate.
// Tests use only `renderer_backend::frame_graph::*` -- no internal fields,
// no private methods, no implementation details.
//
// Contract under test:
//   CompiledFrameGraph::emit_schedule_bridge(&self) -> serde_json::Value
//
//   Serialises the execution schedule as a structured JSON value containing:
//   - "execution_order"  -- flat list of pass indices in topological order
//   - "barriers"         -- {from_pass, to_pass, before_state, after_state}[]
//   - "async_passes"     -- {pass_index, queue_type}[] (compute/copy passes)
//   - "parallel_regions" -- groups of pass indices at same DAG depth
//   - "sync_points"      -- {after_pass, before_pass, barriers[]}[]
//
// Acceptance criteria:
//   1.  Output is valid JSON (serde_json::Value)
//   2.  Output contains all five required top-level keys
//   3.  Execution order is a valid array of numeric pass indices
//   4.  Barriers have from_pass / to_pass / before_state / after_state fields
//   5.  Async passes have pass_index and queue_type fields
//   6.  Parallel regions are arrays of numeric pass indices
//   7.  Sync points have after_pass / before_pass / barriers fields
//   8.  Empty input produces structurally valid output
//   9.  Single pass produces empty barriers, async_passes, sync_points
//  10.  Linear chain produces correct RAW barriers
//  11.  Compute passes appear in async_passes
//  12.  Diamond graph produces both parallel regions and sync points
//  13.  All pass indices in output exist in execution_order
//  14.  Sync points group barriers by (from, to) boundary

use renderer_backend::frame_graph::{
    AttachmentLoadOp, AttachmentStoreOp, BufferDesc, ColorAttachment,
    CompiledFrameGraph, DispatchSource, InstanceSource,
    IrPass, IrResource, PassIndex, ResourceDesc, ResourceHandle,
    ResourceLifetime, ResourceState, TextureDesc, ViewType,
};

// Helper: create a storage buffer resource descriptor.
fn storage_buf(size: u64) -> ResourceDesc {
    ResourceDesc::Buffer(BufferDesc {
        size,
        usage: "storage".into(),
        is_indirect_arg: false,
    })
}

// Helper: create a transient 2D texture resource descriptor.
fn tex_2d(w: u32, h: u32, fmt: &str) -> ResourceDesc {
    ResourceDesc::Texture2D(TextureDesc {
        width: w,
        height: h,
        mip_levels: 1,
        array_layers: 1,
        format: fmt.into(),
    })
}

// =============================================================================
// SECTION 1 -- Output is valid JSON with all required top-level keys
// =============================================================================

#[test]
fn schedule_output_is_valid_json_value() {
    // A simple two-pass linear chain that creates a real compiled graph.
    let mut pass0 = IrPass::compute(
        PassIndex(0),
        "write_phase",
        DispatchSource::Direct { group_count_x: 1, group_count_y: 1, group_count_z: 1 },
        ViewType::Storage,
    );
    pass0.access_set.writes.push(ResourceHandle(1));

    let mut pass1 = IrPass::compute(
        PassIndex(1),
        "read_phase",
        DispatchSource::Direct { group_count_x: 1, group_count_y: 1, group_count_z: 1 },
        ViewType::Storage,
    );
    pass1.access_set.reads.push(ResourceHandle(1));

    let passes = vec![pass0, pass1];
    let resources = vec![IrResource::new(
        ResourceHandle(1), "shared_buf", storage_buf(256),
        ResourceLifetime::Transient, ResourceState::Uninitialized,
    )];

    let compiled = CompiledFrameGraph::compile(passes, resources)
        .expect("simple two-pass graph must compile");

    let schedule = compiled.emit_schedule_bridge();
    // The return type is serde_json::Value — verification that it's valid
    // is implicit in the type system.  We confirm it round-trips cleanly.
    let json_str = serde_json::to_string(&schedule)
        .expect("schedule must serialise to JSON string");
    let _round_trip: serde_json::Value = serde_json::from_str(&json_str)
        .expect("schedule string must deserialise back to valid JSON");
}

#[test]
fn schedule_contains_all_required_keys() {
    let mut pass0 = IrPass::compute(
        PassIndex(0),
        "src",
        DispatchSource::Direct { group_count_x: 1, group_count_y: 1, group_count_z: 1 },
        ViewType::Storage,
    );
    pass0.access_set.writes.push(ResourceHandle(1));

    let mut pass1 = IrPass::compute(
        PassIndex(1),
        "dst",
        DispatchSource::Direct { group_count_x: 1, group_count_y: 1, group_count_z: 1 },
        ViewType::Storage,
    );
    pass1.access_set.reads.push(ResourceHandle(1));

    let passes = vec![pass0, pass1];
    let resources = vec![IrResource::new(
        ResourceHandle(1), "buf", storage_buf(128),
        ResourceLifetime::Transient, ResourceState::Uninitialized,
    )];

    let compiled = CompiledFrameGraph::compile(passes, resources)
        .expect("linear chain must compile");
    let schedule = compiled.emit_schedule_bridge();
    let obj = schedule.as_object()
        .expect("schedule must be a JSON object");

    assert!(obj.contains_key("execution_order"),
        "schedule must contain 'execution_order'");
    assert!(obj.contains_key("barriers"),
        "schedule must contain 'barriers'");
    assert!(obj.contains_key("async_passes"),
        "schedule must contain 'async_passes'");
    assert!(obj.contains_key("parallel_regions"),
        "schedule must contain 'parallel_regions'");
    assert!(obj.contains_key("sync_points"),
        "schedule must contain 'sync_points'");
}

// =============================================================================
// SECTION 2 -- Execution order
// =============================================================================

#[test]
fn execution_order_is_array_of_pass_indices() {
    let mut pass0 = IrPass::compute(
        PassIndex(0), "a",
        DispatchSource::Direct { group_count_x: 1, group_count_y: 1, group_count_z: 1 },
        ViewType::Storage,
    );
    pass0.access_set.writes.push(ResourceHandle(10));

    let mut pass1 = IrPass::compute(
        PassIndex(1), "b",
        DispatchSource::Direct { group_count_x: 1, group_count_y: 1, group_count_z: 1 },
        ViewType::Storage,
    );
    pass1.access_set.reads.push(ResourceHandle(10));

    let passes = vec![pass0, pass1];
    let resources = vec![IrResource::new(
        ResourceHandle(10), "x", storage_buf(64),
        ResourceLifetime::Transient, ResourceState::Uninitialized,
    )];

    let compiled = CompiledFrameGraph::compile(passes, resources).unwrap();
    let schedule = compiled.emit_schedule_bridge();

    let order = schedule["execution_order"].as_array()
        .expect("execution_order must be an array");
    assert!(!order.is_empty(), "execution_order must not be empty");

    for (i, idx) in order.iter().enumerate() {
        let n = idx.as_u64()
            .unwrap_or_else(|| panic!("execution_order[{}] must be a number", i));
        let _ = n as usize; // valid usize
    }
}

#[test]
fn execution_order_matches_expected_topological_sort() {
    // Diamond: entry(0) -> mid_a(1), mid_b(2) -> exit(3)
    let mut e = IrPass::compute(
        PassIndex(0), "entry",
        DispatchSource::Direct { group_count_x: 1, group_count_y: 1, group_count_z: 1 },
        ViewType::Storage,
    );
    e.access_set.writes.push(ResourceHandle(1));
    e.access_set.writes.push(ResourceHandle(2));

    let mut a = IrPass::compute(
        PassIndex(1), "mid_a",
        DispatchSource::Direct { group_count_x: 1, group_count_y: 1, group_count_z: 1 },
        ViewType::Storage,
    );
    a.access_set.reads.push(ResourceHandle(1));
    a.access_set.writes.push(ResourceHandle(3));

    let mut b = IrPass::compute(
        PassIndex(2), "mid_b",
        DispatchSource::Direct { group_count_x: 1, group_count_y: 1, group_count_z: 1 },
        ViewType::Storage,
    );
    b.access_set.reads.push(ResourceHandle(2));
    b.access_set.writes.push(ResourceHandle(4));

    let mut x = IrPass::compute(
        PassIndex(3), "exit",
        DispatchSource::Direct { group_count_x: 1, group_count_y: 1, group_count_z: 1 },
        ViewType::Storage,
    );
    x.access_set.reads.push(ResourceHandle(3));
    x.access_set.reads.push(ResourceHandle(4));

    let passes = vec![e, a, b, x];
    let resources = vec![
        IrResource::new(ResourceHandle(1), "r1", storage_buf(64),
            ResourceLifetime::Transient, ResourceState::Uninitialized),
        IrResource::new(ResourceHandle(2), "r2", storage_buf(64),
            ResourceLifetime::Transient, ResourceState::Uninitialized),
        IrResource::new(ResourceHandle(3), "r3", storage_buf(64),
            ResourceLifetime::Transient, ResourceState::Uninitialized),
        IrResource::new(ResourceHandle(4), "r4", storage_buf(64),
            ResourceLifetime::Transient, ResourceState::Uninitialized),
    ];

    let compiled = CompiledFrameGraph::compile(passes, resources).unwrap();
    let schedule = compiled.emit_schedule_bridge();

    let order: Vec<usize> = schedule["execution_order"].as_array().unwrap().iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();

    // Must start at 0 (entry) and end at 3 (exit)
    assert_eq!(order[0], 0, "first in execution order must be entry (0)");
    assert_eq!(order[order.len() - 1], 3, "last must be exit (3)");

    // Must contain all four passes exactly once.
    assert_eq!(order.len(), 4, "must contain all 4 passes");
    for i in 0..=3 {
        assert!(order.contains(&i), "execution_order must contain pass {}", i);
    }
}

// =============================================================================
// SECTION 3 -- Barriers
// =============================================================================

#[test]
fn barriers_have_all_required_fields() {
    let mut pass0 = IrPass::compute(
        PassIndex(0), "writer",
        DispatchSource::Direct { group_count_x: 1, group_count_y: 1, group_count_z: 1 },
        ViewType::Storage,
    );
    pass0.access_set.writes.push(ResourceHandle(1));

    let mut pass1 = IrPass::compute(
        PassIndex(1), "reader",
        DispatchSource::Direct { group_count_x: 1, group_count_y: 1, group_count_z: 1 },
        ViewType::Storage,
    );
    pass1.access_set.reads.push(ResourceHandle(1));

    let passes = vec![pass0, pass1];
    let resources = vec![IrResource::new(
        ResourceHandle(1), "buf", storage_buf(64),
        ResourceLifetime::Transient, ResourceState::Uninitialized,
    )];

    let compiled = CompiledFrameGraph::compile(passes, resources).unwrap();
    let schedule = compiled.emit_schedule_bridge();

    let barriers = schedule["barriers"].as_array()
        .expect("barriers must be an array");
    assert!(!barriers.is_empty(), "linear chain must produce at least one barrier");

    for (i, b) in barriers.iter().enumerate() {
        let obj = b.as_object()
            .unwrap_or_else(|| panic!("barriers[{}] must be an object", i));
        assert!(obj.contains_key("from_pass"),
            "barriers[{}] must have 'from_pass'", i);
        assert!(obj.contains_key("to_pass"),
            "barriers[{}] must have 'to_pass'", i);
        assert!(obj.contains_key("before_state"),
            "barriers[{}] must have 'before_state'", i);
        assert!(obj.contains_key("after_state"),
            "barriers[{}] must have 'after_state'", i);

        // Values must be the correct types.
        assert!(obj["from_pass"].is_number(),
            "barriers[{}].from_pass must be a number", i);
        assert!(obj["to_pass"].is_number(),
            "barriers[{}].to_pass must be a number", i);
        assert!(obj["before_state"].is_string(),
            "barriers[{}].before_state must be a string", i);
        assert!(obj["after_state"].is_string(),
            "barriers[{}].after_state must be a string", i);
    }
}

#[test]
fn linear_chain_produces_correct_raw_barrier() {
    // Pass 0 writes resource R, Pass 1 reads resource R => RAW barrier.
    let mut pass0 = IrPass::compute(
        PassIndex(0), "gen",
        DispatchSource::Direct { group_count_x: 1, group_count_y: 1, group_count_z: 1 },
        ViewType::Storage,
    );
    pass0.access_set.writes.push(ResourceHandle(99));

    let mut pass1 = IrPass::compute(
        PassIndex(1), "consume",
        DispatchSource::Direct { group_count_x: 1, group_count_y: 1, group_count_z: 1 },
        ViewType::Storage,
    );
    pass1.access_set.reads.push(ResourceHandle(99));

    let passes = vec![pass0, pass1];
    let resources = vec![IrResource::new(
        ResourceHandle(99), "data", storage_buf(1024),
        ResourceLifetime::Transient, ResourceState::Uninitialized,
    )];

    let compiled = CompiledFrameGraph::compile(passes, resources).unwrap();
    let schedule = compiled.emit_schedule_bridge();

    let barriers = schedule["barriers"].as_array().unwrap();
    assert_eq!(barriers.len(), 1, "linear chain should produce exactly one barrier");

    let barrier = &barriers[0];
    assert_eq!(barrier["from_pass"].as_u64().unwrap(), 0,
        "barrier from_pass must be the writer (0)");
    assert_eq!(barrier["to_pass"].as_u64().unwrap(), 1,
        "barrier to_pass must be the reader (1)");
}

#[test]
fn empty_pass_list_produces_no_barriers() {
    let passes: Vec<IrPass> = vec![];
    let resources: Vec<IrResource> = vec![];

    let compiled = CompiledFrameGraph::compile(passes, resources)
        .expect("empty graph must compile");
    let schedule = compiled.emit_schedule_bridge();

    let barriers = schedule["barriers"].as_array()
        .expect("barriers must be present");
    assert!(barriers.is_empty(), "empty graph must have zero barriers");
}

// =============================================================================
// SECTION 4 -- Async passes
// =============================================================================

#[test]
fn async_passes_have_required_fields() {
    let mut pass0 = IrPass::compute(
        PassIndex(0), "comp",
        DispatchSource::Direct { group_count_x: 8, group_count_y: 8, group_count_z: 1 },
        ViewType::Storage,
    );
    pass0.access_set.writes.push(ResourceHandle(1));

    let mut pass1 = IrPass::compute(
        PassIndex(1), "comp2",
        DispatchSource::Direct { group_count_x: 4, group_count_y: 4, group_count_z: 1 },
        ViewType::Storage,
    );
    pass1.access_set.reads.push(ResourceHandle(1));

    let passes = vec![pass0, pass1];
    let resources = vec![IrResource::new(
        ResourceHandle(1), "buf", storage_buf(256),
        ResourceLifetime::Transient, ResourceState::Uninitialized,
    )];

    let compiled = CompiledFrameGraph::compile(passes, resources).unwrap();
    let schedule = compiled.emit_schedule_bridge();

    let async_passes = schedule["async_passes"].as_array()
        .expect("async_passes must be an array");

    for (i, a) in async_passes.iter().enumerate() {
        let obj = a.as_object()
            .unwrap_or_else(|| panic!("async_passes[{}] must be an object", i));
        assert!(obj.contains_key("pass_index"),
            "async_passes[{}] must have 'pass_index'", i);
        assert!(obj.contains_key("queue_type"),
            "async_passes[{}] must have 'queue_type'", i);

        assert!(obj["pass_index"].is_number(),
            "async_passes[{}].pass_index must be a number", i);
        assert!(obj["queue_type"].is_string(),
            "async_passes[{}].queue_type must be a string", i);
    }
}

#[test]
fn compute_passes_appear_in_async_passes() {
    let mut pass0 = IrPass::compute(
        PassIndex(0), "compute_task",
        DispatchSource::Direct { group_count_x: 16, group_count_y: 1, group_count_z: 1 },
        ViewType::Storage,
    );
    pass0.access_set.writes.push(ResourceHandle(1));

    let mut pass1 = IrPass::compute(
        PassIndex(1), "compute_consumer",
        DispatchSource::Direct { group_count_x: 16, group_count_y: 1, group_count_z: 1 },
        ViewType::Storage,
    );
    pass1.access_set.reads.push(ResourceHandle(1));

    let passes = vec![pass0, pass1];
    let resources = vec![IrResource::new(
        ResourceHandle(1), "shared", storage_buf(128),
        ResourceLifetime::Transient, ResourceState::Uninitialized,
    )];

    let compiled = CompiledFrameGraph::compile(passes, resources).unwrap();
    let schedule = compiled.emit_schedule_bridge();

    let async_passes = schedule["async_passes"].as_array().unwrap();
    assert!(!async_passes.is_empty(), "compute passes must appear in async_passes");

    let indices: Vec<u64> = async_passes.iter()
        .map(|a| a["pass_index"].as_u64().unwrap())
        .collect();

    // Pass 0 and pass 1 are both compute — they should be eligible.
    assert!(indices.contains(&0), "compute pass 0 must be in async_passes");
    assert!(indices.contains(&1), "compute pass 1 must be in async_passes");
}

#[test]
fn async_passes_have_queue_type_string() {
    let mut pass0 = IrPass::compute(
        PassIndex(0), "async_comp",
        DispatchSource::Direct { group_count_x: 8, group_count_y: 8, group_count_z: 1 },
        ViewType::Storage,
    );
    pass0.access_set.writes.push(ResourceHandle(1));

    let mut pass1 = IrPass::copy(PassIndex(1), "async_copy");
    pass1.access_set.reads.push(ResourceHandle(1));

    let passes = vec![pass0, pass1];
    let resources = vec![IrResource::new(
        ResourceHandle(1), "data", storage_buf(64),
        ResourceLifetime::Transient, ResourceState::Uninitialized,
    )];

    let compiled = CompiledFrameGraph::compile(passes, resources).unwrap();
    let schedule = compiled.emit_schedule_bridge();

    let async_passes = schedule["async_passes"].as_array().unwrap();
    for a in async_passes {
        let queue = a["queue_type"].as_str().unwrap();
        // Must be a non-empty string describing the queue.
        assert!(!queue.is_empty(), "queue_type must be non-empty");
    }
}

// =============================================================================
// SECTION 5 -- Parallel regions
// =============================================================================

#[test]
fn parallel_regions_are_arrays_of_numeric_indices() {
    // Diamond graph creates parallel regions.
    let mut e = IrPass::compute(
        PassIndex(0), "entry",
        DispatchSource::Direct { group_count_x: 1, group_count_y: 1, group_count_z: 1 },
        ViewType::Storage,
    );
    e.access_set.writes.push(ResourceHandle(1));
    e.access_set.writes.push(ResourceHandle(2));

    let mut a = IrPass::compute(
        PassIndex(1), "branch_a",
        DispatchSource::Direct { group_count_x: 1, group_count_y: 1, group_count_z: 1 },
        ViewType::Storage,
    );
    a.access_set.reads.push(ResourceHandle(1));
    a.access_set.writes.push(ResourceHandle(3));

    let mut b = IrPass::compute(
        PassIndex(2), "branch_b",
        DispatchSource::Direct { group_count_x: 1, group_count_y: 1, group_count_z: 1 },
        ViewType::Storage,
    );
    b.access_set.reads.push(ResourceHandle(2));
    b.access_set.writes.push(ResourceHandle(4));

    let mut x = IrPass::compute(
        PassIndex(3), "exit",
        DispatchSource::Direct { group_count_x: 1, group_count_y: 1, group_count_z: 1 },
        ViewType::Storage,
    );
    x.access_set.reads.push(ResourceHandle(3));
    x.access_set.reads.push(ResourceHandle(4));

    let passes = vec![e, a, b, x];
    let resources = vec![
        IrResource::new(ResourceHandle(1), "r1", storage_buf(64),
            ResourceLifetime::Transient, ResourceState::Uninitialized),
        IrResource::new(ResourceHandle(2), "r2", storage_buf(64),
            ResourceLifetime::Transient, ResourceState::Uninitialized),
        IrResource::new(ResourceHandle(3), "r3", storage_buf(64),
            ResourceLifetime::Transient, ResourceState::Uninitialized),
        IrResource::new(ResourceHandle(4), "r4", storage_buf(64),
            ResourceLifetime::Transient, ResourceState::Uninitialized),
    ];

    let compiled = CompiledFrameGraph::compile(passes, resources).unwrap();
    let schedule = compiled.emit_schedule_bridge();

    let regions = schedule["parallel_regions"].as_array()
        .expect("parallel_regions must be an array");
    assert!(!regions.is_empty(), "diamond graph must have parallel regions");

    for (i, region) in regions.iter().enumerate() {
        let arr = region.as_array()
            .unwrap_or_else(|| panic!("parallel_regions[{}] must be an array", i));
        assert!(!arr.is_empty(), "parallel_regions[{}] must not be empty", i);

        for (j, idx) in arr.iter().enumerate() {
            let _n = idx.as_u64()
                .unwrap_or_else(|| panic!("parallel_regions[{}][{}] must be a number", i, j));
        }
    }
}

#[test]
fn parallel_regions_group_passes_at_same_depth() {
    // Diamond: depth0=[entry], depth1=[mid_a, mid_b], depth2=[exit]
    let mut e = IrPass::compute(
        PassIndex(0), "entry",
        DispatchSource::Direct { group_count_x: 1, group_count_y: 1, group_count_z: 1 },
        ViewType::Storage,
    );
    e.access_set.writes.push(ResourceHandle(1));
    e.access_set.writes.push(ResourceHandle(2));

    let mut a = IrPass::compute(
        PassIndex(1), "mid_a",
        DispatchSource::Direct { group_count_x: 1, group_count_y: 1, group_count_z: 1 },
        ViewType::Storage,
    );
    a.access_set.reads.push(ResourceHandle(1));
    a.access_set.writes.push(ResourceHandle(3));

    let mut b = IrPass::compute(
        PassIndex(2), "mid_b",
        DispatchSource::Direct { group_count_x: 1, group_count_y: 1, group_count_z: 1 },
        ViewType::Storage,
    );
    b.access_set.reads.push(ResourceHandle(2));
    b.access_set.writes.push(ResourceHandle(4));

    let mut x = IrPass::compute(
        PassIndex(3), "exit",
        DispatchSource::Direct { group_count_x: 1, group_count_y: 1, group_count_z: 1 },
        ViewType::Storage,
    );
    x.access_set.reads.push(ResourceHandle(3));
    x.access_set.reads.push(ResourceHandle(4));

    let passes = vec![e, a, b, x];
    let resources = vec![
        IrResource::new(ResourceHandle(1), "r1", storage_buf(64),
            ResourceLifetime::Transient, ResourceState::Uninitialized),
        IrResource::new(ResourceHandle(2), "r2", storage_buf(64),
            ResourceLifetime::Transient, ResourceState::Uninitialized),
        IrResource::new(ResourceHandle(3), "r3", storage_buf(64),
            ResourceLifetime::Transient, ResourceState::Uninitialized),
        IrResource::new(ResourceHandle(4), "r4", storage_buf(64),
            ResourceLifetime::Transient, ResourceState::Uninitialized),
    ];

    let compiled = CompiledFrameGraph::compile(passes, resources).unwrap();
    let schedule = compiled.emit_schedule_bridge();

    let regions: Vec<Vec<usize>> = schedule["parallel_regions"].as_array().unwrap().iter()
        .map(|r| r.as_array().unwrap().iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect())
        .collect();

    // Depth 0: only pass 0 (entry).
    assert_eq!(regions[0], vec![0], "region 0 must be entry alone");

    // Depth 1: contains both mid passes (1 and 2) — they must share a region.
    let depth1 = &regions[1];
    assert_eq!(depth1.len(), 2, "region 1 must contain 2 parallel passes");
    assert!(depth1.contains(&1), "region 1 must contain pass 1");
    assert!(depth1.contains(&2), "region 1 must contain pass 2");

    // Depth 2: only pass 3 (exit).
    assert_eq!(regions[regions.len() - 1], vec![3], "last region must be exit alone");
}

// =============================================================================
// SECTION 6 -- Sync points
// =============================================================================

#[test]
fn sync_points_have_all_required_fields() {
    let mut pass0 = IrPass::compute(
        PassIndex(0), "src",
        DispatchSource::Direct { group_count_x: 1, group_count_y: 1, group_count_z: 1 },
        ViewType::Storage,
    );
    pass0.access_set.writes.push(ResourceHandle(1));

    let mut pass1 = IrPass::compute(
        PassIndex(1), "dst",
        DispatchSource::Direct { group_count_x: 1, group_count_y: 1, group_count_z: 1 },
        ViewType::Storage,
    );
    pass1.access_set.reads.push(ResourceHandle(1));

    let passes = vec![pass0, pass1];
    let resources = vec![IrResource::new(
        ResourceHandle(1), "buf", storage_buf(64),
        ResourceLifetime::Transient, ResourceState::Uninitialized,
    )];

    let compiled = CompiledFrameGraph::compile(passes, resources).unwrap();
    let schedule = compiled.emit_schedule_bridge();

    let sync_points = schedule["sync_points"].as_array()
        .expect("sync_points must be an array");

    for (i, sp) in sync_points.iter().enumerate() {
        let obj = sp.as_object()
            .unwrap_or_else(|| panic!("sync_points[{}] must be an object", i));
        assert!(obj.contains_key("after_pass"),
            "sync_points[{}] must have 'after_pass'", i);
        assert!(obj.contains_key("before_pass"),
            "sync_points[{}] must have 'before_pass'", i);
        assert!(obj.contains_key("barriers"),
            "sync_points[{}] must have 'barriers'", i);

        assert!(obj["after_pass"].is_number(),
            "sync_points[{}].after_pass must be a number", i);
        assert!(obj["before_pass"].is_number(),
            "sync_points[{}].before_pass must be a number", i);
        assert!(obj["barriers"].is_array(),
            "sync_points[{}].barriers must be an array", i);

        // Each barrier in the sync point must have before_state and after_state.
        for (j, b) in obj["barriers"].as_array().unwrap().iter().enumerate() {
            let b_obj = b.as_object()
                .unwrap_or_else(|| panic!("sync_points[{}].barriers[{}] must be an object", i, j));
            assert!(b_obj.contains_key("before_state"),
                "sync_points[{}].barriers[{}] must have 'before_state'", i, j);
            assert!(b_obj.contains_key("after_state"),
                "sync_points[{}].barriers[{}] must have 'after_state'", i, j);
        }
    }
}

#[test]
fn sync_points_group_barriers_by_boundary() {
    // Diamond graph with 4 barriers across 4 unique boundaries.
    let mut e = IrPass::compute(
        PassIndex(0), "entry",
        DispatchSource::Direct { group_count_x: 1, group_count_y: 1, group_count_z: 1 },
        ViewType::Storage,
    );
    e.access_set.writes.push(ResourceHandle(1));
    e.access_set.writes.push(ResourceHandle(2));

    let mut a = IrPass::compute(
        PassIndex(1), "mid_a",
        DispatchSource::Direct { group_count_x: 1, group_count_y: 1, group_count_z: 1 },
        ViewType::Storage,
    );
    a.access_set.reads.push(ResourceHandle(1));
    a.access_set.writes.push(ResourceHandle(3));

    let mut b = IrPass::compute(
        PassIndex(2), "mid_b",
        DispatchSource::Direct { group_count_x: 1, group_count_y: 1, group_count_z: 1 },
        ViewType::Storage,
    );
    b.access_set.reads.push(ResourceHandle(2));
    b.access_set.writes.push(ResourceHandle(4));

    let mut x = IrPass::compute(
        PassIndex(3), "exit",
        DispatchSource::Direct { group_count_x: 1, group_count_y: 1, group_count_z: 1 },
        ViewType::Storage,
    );
    x.access_set.reads.push(ResourceHandle(3));
    x.access_set.reads.push(ResourceHandle(4));

    let passes = vec![e, a, b, x];
    let resources = vec![
        IrResource::new(ResourceHandle(1), "r1", storage_buf(64),
            ResourceLifetime::Transient, ResourceState::Uninitialized),
        IrResource::new(ResourceHandle(2), "r2", storage_buf(64),
            ResourceLifetime::Transient, ResourceState::Uninitialized),
        IrResource::new(ResourceHandle(3), "r3", storage_buf(64),
            ResourceLifetime::Transient, ResourceState::Uninitialized),
        IrResource::new(ResourceHandle(4), "r4", storage_buf(64),
            ResourceLifetime::Transient, ResourceState::Uninitialized),
    ];

    let compiled = CompiledFrameGraph::compile(passes, resources).unwrap();
    let schedule = compiled.emit_schedule_bridge();

    let barriers = schedule["barriers"].as_array().unwrap();
    // Collect all (from, to) pairs from the flat barrier list.
    let barrier_pairs: Vec<(usize, usize)> = barriers.iter()
        .map(|b| (b["from_pass"].as_u64().unwrap() as usize,
                  b["to_pass"].as_u64().unwrap() as usize))
        .collect();

    let sync_points = schedule["sync_points"].as_array().unwrap();
    let sync_boundaries: Vec<(usize, usize)> = sync_points.iter()
        .map(|sp| (sp["after_pass"].as_u64().unwrap() as usize,
                   sp["before_pass"].as_u64().unwrap() as usize))
        .collect();

    // Every flat barrier pair must appear as a sync point boundary.
    for &(from, to) in &barrier_pairs {
        assert!(sync_boundaries.contains(&(from, to)),
            "sync point must exist for barrier boundary ({}, {})", from, to);
    }

    // Every sync point boundary must correspond to a flat barrier pair.
    for &(after, before) in &sync_boundaries {
        assert!(barrier_pairs.contains(&(after, before)),
            "sync point boundary ({}, {}) must have a corresponding barrier", after, before);
    }
}

// =============================================================================
// SECTION 7 -- Structural consistency
// =============================================================================

#[test]
fn all_referenced_pass_indices_exist_in_execution_order() {
    let mut e = IrPass::compute(
        PassIndex(0), "entry",
        DispatchSource::Direct { group_count_x: 1, group_count_y: 1, group_count_z: 1 },
        ViewType::Storage,
    );
    e.access_set.writes.push(ResourceHandle(1));

    let mut a = IrPass::compute(
        PassIndex(1), "mid",
        DispatchSource::Direct { group_count_x: 1, group_count_y: 1, group_count_z: 1 },
        ViewType::Storage,
    );
    a.access_set.reads.push(ResourceHandle(1));
    a.access_set.writes.push(ResourceHandle(2));

    let mut x = IrPass::compute(
        PassIndex(2), "exit",
        DispatchSource::Direct { group_count_x: 1, group_count_y: 1, group_count_z: 1 },
        ViewType::Storage,
    );
    x.access_set.reads.push(ResourceHandle(2));

    let passes = vec![e, a, x];
    let resources = vec![
        IrResource::new(ResourceHandle(1), "r1", storage_buf(64),
            ResourceLifetime::Transient, ResourceState::Uninitialized),
        IrResource::new(ResourceHandle(2), "r2", storage_buf(64),
            ResourceLifetime::Transient, ResourceState::Uninitialized),
    ];

    let compiled = CompiledFrameGraph::compile(passes, resources).unwrap();
    let schedule = compiled.emit_schedule_bridge();

    // Collect all pass indices referenced across the schedule.
    let order: Vec<u64> = schedule["execution_order"].as_array().unwrap().iter()
        .map(|v| v.as_u64().unwrap())
        .collect();

    // Check barrier from_pass and to_pass reference existing indices.
    for b in schedule["barriers"].as_array().unwrap() {
        let from = b["from_pass"].as_u64().unwrap();
        let to = b["to_pass"].as_u64().unwrap();
        assert!(order.contains(&from),
            "barrier references from_pass {} which is not in execution_order", from);
        assert!(order.contains(&to),
            "barrier references to_pass {} which is not in execution_order", to);
    }

    // Check async_passes pass_index references existing indices.
    for a in schedule["async_passes"].as_array().unwrap() {
        let idx = a["pass_index"].as_u64().unwrap();
        assert!(order.contains(&idx),
            "async_pass references pass_index {} not in execution_order", idx);
    }

    // Check parallel region pass indices reference existing indices.
    for region in schedule["parallel_regions"].as_array().unwrap() {
        for v in region.as_array().unwrap() {
            let idx = v.as_u64().unwrap();
            assert!(order.contains(&idx),
                "parallel_region references pass {} not in execution_order", idx);
        }
    }

    // Check sync point after_pass / before_pass reference existing indices.
    for sp in schedule["sync_points"].as_array().unwrap() {
        let after = sp["after_pass"].as_u64().unwrap();
        let before = sp["before_pass"].as_u64().unwrap();
        assert!(order.contains(&after),
            "sync_point after_pass {} not in execution_order", after);
        assert!(order.contains(&before),
            "sync_point before_pass {} not in execution_order", before);
    }
}

// =============================================================================
// SECTION 8 -- Edge cases: empty and single-pass graphs
// =============================================================================

#[test]
fn empty_graph_produces_structurally_valid_schedule() {
    let passes: Vec<IrPass> = vec![];
    let resources: Vec<IrResource> = vec![];

    let compiled = CompiledFrameGraph::compile(passes, resources)
        .expect("empty graph must compile");
    let schedule = compiled.emit_schedule_bridge();
    let obj = schedule.as_object()
        .expect("schedule must be a JSON object");

    assert!(obj.contains_key("execution_order"));
    assert!(obj.contains_key("barriers"));
    assert!(obj.contains_key("async_passes"));
    assert!(obj.contains_key("parallel_regions"));
    assert!(obj.contains_key("sync_points"));

    assert_eq!(obj["execution_order"].as_array().unwrap().len(), 0,
        "empty graph must have empty execution_order");
    assert_eq!(obj["barriers"].as_array().unwrap().len(), 0,
        "empty graph must have no barriers");
    assert_eq!(obj["async_passes"].as_array().unwrap().len(), 0,
        "empty graph must have no async_passes");
    assert_eq!(obj["parallel_regions"].as_array().unwrap().len(), 0,
        "empty graph must have no parallel_regions");
    assert_eq!(obj["sync_points"].as_array().unwrap().len(), 0,
        "empty graph must have no sync_points");
}

#[test]
fn single_graphics_pass_produces_no_barriers_or_sync_points() {
    let pass = IrPass::graphics(
        PassIndex(0),
        "clear_pass",
        vec![ColorAttachment {
            resource: ResourceHandle(1),
            load_op: AttachmentLoadOp::Clear,
            store_op: AttachmentStoreOp::Store,
            ..Default::default()
        }],
        None,
        InstanceSource::Direct {
            index_count: 0,
            instance_count: 1,
            base_vertex: 0,
            first_index: 0,
            first_instance: 0,
        },
        ViewType::Texture2D,
    );

    let resources = vec![IrResource::new(
        ResourceHandle(1), "rt", tex_2d(1920, 1080, "rgba8unorm"),
        ResourceLifetime::Transient, ResourceState::Uninitialized,
    )];

    let compiled = CompiledFrameGraph::compile(vec![pass], resources)
        .expect("single graphics pass must compile");
    let schedule = compiled.emit_schedule_bridge();

    assert_eq!(schedule["execution_order"].as_array().unwrap().len(), 1,
        "single pass must produce execution_order of length 1");
    assert_eq!(schedule["execution_order"][0].as_u64().unwrap(), 0,
        "execution_order must contain pass 0");

    assert_eq!(schedule["barriers"].as_array().unwrap().len(), 0,
        "single pass must have no barriers");
    assert_eq!(schedule["sync_points"].as_array().unwrap().len(), 0,
        "single pass must have no sync points");
}

#[test]
fn single_compute_pass_produces_async_entry() {
    let pass = IrPass::compute(
        PassIndex(0),
        "compute_dispatch",
        DispatchSource::Direct { group_count_x: 32, group_count_y: 1, group_count_z: 1 },
        ViewType::Storage,
    );

    let compiled = CompiledFrameGraph::compile(vec![pass], vec![])
        .expect("single compute pass must compile");
    let schedule = compiled.emit_schedule_bridge();

    assert_eq!(schedule["execution_order"][0].as_u64().unwrap(), 0);

    // A lone compute pass with no resource dependencies may be scheduled async.
    let async_passes = schedule["async_passes"].as_array().unwrap();
    if !async_passes.is_empty() {
        assert_eq!(async_passes[0]["pass_index"].as_u64().unwrap(), 0,
            "async pass must reference pass 0");
        let queue = async_passes[0]["queue_type"].as_str().unwrap();
        assert!(!queue.is_empty(), "queue type must be non-empty");
    }
}

// =============================================================================
// SECTION 9 -- Graphics passes do not appear in async_passes
// =============================================================================

#[test]
fn graphics_pass_not_in_async_passes() {
    let pass = IrPass::graphics(
        PassIndex(0),
        "rasterize",
        vec![ColorAttachment {
            resource: ResourceHandle(1),
            ..Default::default()
        }],
        None,
        InstanceSource::Direct {
            index_count: 36,
            instance_count: 1,
            base_vertex: 0,
            first_index: 0,
            first_instance: 0,
        },
        ViewType::Texture2D,
    );

    let resources = vec![IrResource::new(
        ResourceHandle(1), "rt", tex_2d(800, 600, "rgba8unorm"),
        ResourceLifetime::Transient, ResourceState::Uninitialized,
    )];

    let compiled = CompiledFrameGraph::compile(vec![pass], resources).unwrap();
    let schedule = compiled.emit_schedule_bridge();

    // A lone graphics pass with no dependent pass is never async.
    for a in schedule["async_passes"].as_array().unwrap() {
        let idx = a["pass_index"].as_u64().unwrap();
        assert_ne!(idx, 0,
            "graphics pass 0 must not appear in async_passes");
    }
}

// =============================================================================
// SECTION 10 -- Multi-resource barriers produce correct sync points
// =============================================================================

#[test]
fn multi_resource_barrier_produces_single_sync_point_with_multiple_barriers() {
    // Two resources written by pass 0, both read by pass 1.
    // This produces 2 barriers at the same (0,1) boundary, which should
    // collapse into 1 sync_point with 2 barrier entries.
    let mut pass0 = IrPass::compute(
        PassIndex(0), "multi_writer",
        DispatchSource::Direct { group_count_x: 1, group_count_y: 1, group_count_z: 1 },
        ViewType::Storage,
    );
    pass0.access_set.writes.push(ResourceHandle(1));
    pass0.access_set.writes.push(ResourceHandle(2));

    let mut pass1 = IrPass::compute(
        PassIndex(1), "multi_reader",
        DispatchSource::Direct { group_count_x: 1, group_count_y: 1, group_count_z: 1 },
        ViewType::Storage,
    );
    pass1.access_set.reads.push(ResourceHandle(1));
    pass1.access_set.reads.push(ResourceHandle(2));

    let passes = vec![pass0, pass1];
    let resources = vec![
        IrResource::new(ResourceHandle(1), "a", storage_buf(64),
            ResourceLifetime::Transient, ResourceState::Uninitialized),
        IrResource::new(ResourceHandle(2), "b", storage_buf(64),
            ResourceLifetime::Transient, ResourceState::Uninitialized),
    ];

    let compiled = CompiledFrameGraph::compile(passes, resources).unwrap();
    let schedule = compiled.emit_schedule_bridge();

    let barriers = schedule["barriers"].as_array().unwrap();
    assert_eq!(barriers.len(), 2, "two resources => 2 barriers");

    // Both barriers share the same (0,1) boundary.
    for b in barriers.iter() {
        assert_eq!(b["from_pass"].as_u64().unwrap(), 0);
        assert_eq!(b["to_pass"].as_u64().unwrap(), 1);
    }

    // Sync points collapse (0,1) into a single entry with 2 barrier entries.
    let sync_points = schedule["sync_points"].as_array().unwrap();
    assert_eq!(sync_points.len(), 1,
        "2 barriers at same boundary => 1 sync point");

    let sp = &sync_points[0];
    assert_eq!(sp["after_pass"].as_u64().unwrap(), 0);
    assert_eq!(sp["before_pass"].as_u64().unwrap(), 1);
    assert_eq!(sp["barriers"].as_array().unwrap().len(), 2,
        "sync point barriers must contain both barrier entries");
}

// =============================================================================
// SECTION 11 -- Complex mixed graph
// =============================================================================

#[test]
fn mixed_pass_types_produce_comprehensive_schedule() {
    // Build a 5-pass graph mixing graphics, compute, and copy:
    //   pass 0: graphics — writes rt
    //   pass 1: graphics — reads rt, writes postprocess
    //   pass 2: compute  — reads rt, writes compute_out
    //   pass 3: copy     — reads compute_out to transfer
    //   pass 4: graphics — reads postprocess (final output)
    //
    // A realistic mini-frame.

    let mut g0 = IrPass::graphics(
        PassIndex(0), "gbuffer",
        vec![ColorAttachment {
            resource: ResourceHandle(1),
            load_op: AttachmentLoadOp::Clear,
            store_op: AttachmentStoreOp::Store,
            ..Default::default()
        }],
        None,
        InstanceSource::Direct {
            index_count: 36, instance_count: 1,
            base_vertex: 0, first_index: 0, first_instance: 0,
        },
        ViewType::Texture2D,
    );
    g0.access_set.writes.push(ResourceHandle(1));

    let mut g1 = IrPass::graphics(
        PassIndex(1), "post",
        vec![ColorAttachment {
            resource: ResourceHandle(2),
            load_op: AttachmentLoadOp::Load,
            store_op: AttachmentStoreOp::Store,
            ..Default::default()
        }],
        None,
        InstanceSource::Direct {
            index_count: 36, instance_count: 1,
            base_vertex: 0, first_index: 0, first_instance: 0,
        },
        ViewType::Texture2D,
    );
    // Reads gbuffer output, writes postprocess target.
    g1.access_set.reads.push(ResourceHandle(1));
    g1.access_set.writes.push(ResourceHandle(2));

    let mut c2 = IrPass::compute(
        PassIndex(2), "compute_fx",
        DispatchSource::Direct { group_count_x: 16, group_count_y: 16, group_count_z: 1 },
        ViewType::Storage,
    );
    // Reads gbuffer, writes compute output.
    c2.access_set.reads.push(ResourceHandle(1));
    c2.access_set.writes.push(ResourceHandle(3));

    let mut cp3 = IrPass::copy(PassIndex(3), "download");
    cp3.access_set.reads.push(ResourceHandle(3));

    let mut g4 = IrPass::graphics(
        PassIndex(4), "final_blit",
        vec![ColorAttachment {
            resource: ResourceHandle(4),
            load_op: AttachmentLoadOp::Clear,
            store_op: AttachmentStoreOp::Store,
            ..Default::default()
        }],
        None,
        InstanceSource::Direct {
            index_count: 6, instance_count: 1,
            base_vertex: 0, first_index: 0, first_instance: 0,
        },
        ViewType::Texture2D,
    );
    g4.access_set.reads.push(ResourceHandle(2));

    let passes = vec![g0, g1, c2, cp3, g4];
    let resources = vec![
        IrResource::new(ResourceHandle(1), "gbuffer_rt",
            tex_2d(1920, 1080, "rgba16float"),
            ResourceLifetime::Transient, ResourceState::Uninitialized),
        IrResource::new(ResourceHandle(2), "post_rt",
            tex_2d(1920, 1080, "rgba8unorm"),
            ResourceLifetime::Transient, ResourceState::Uninitialized),
        IrResource::new(ResourceHandle(3), "compute_buf",
            storage_buf(65536),
            ResourceLifetime::Transient, ResourceState::Uninitialized),
        IrResource::new(ResourceHandle(4), "swapchain_rt",
            tex_2d(1920, 1080, "bgra8unorm"),
            ResourceLifetime::Transient, ResourceState::Uninitialized),
    ];

    let compiled = CompiledFrameGraph::compile(passes, resources)
        .expect("mixed graph must compile");
    let schedule = compiled.emit_schedule_bridge();

    // ----- Verify structural integrity -----
    let order: Vec<usize> = schedule["execution_order"].as_array().unwrap().iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    assert_eq!(order.len(), 5, "all 5 passes must be in execution_order");

    // All five pass indices present.
    for i in 0..5 {
        assert!(order.contains(&i), "pass {} must be in execution_order", i);
    }

    // ----- Verify barriers exist for resource dependencies -----
    let barriers = schedule["barriers"].as_array().unwrap();
    assert!(!barriers.is_empty(), "mixed graph must produce barriers");

    let barrier_pairs: Vec<(usize, usize)> = barriers.iter()
        .map(|b| (b["from_pass"].as_u64().unwrap() as usize,
                  b["to_pass"].as_u64().unwrap() as usize))
        .collect();

    // gbuffer(0) -> post(1): RAW on ResourceHandle(1)
    assert!(barrier_pairs.contains(&(0, 1)),
        "must have barrier from gbuffer(0) to post(1)");

    // gbuffer(0) -> compute_fx(2): RAW on ResourceHandle(1)
    assert!(barrier_pairs.contains(&(0, 2)),
        "must have barrier from gbuffer(0) to compute_fx(2)");

    // post(1) -> final_blit(4): RAW on ResourceHandle(2)
    assert!(barrier_pairs.contains(&(1, 4)),
        "must have barrier from post(1) to final_blit(4)");

    // ----- Verify copy pass is in async_passes (compute may or may not be) -----
    let async_indices: Vec<u64> = schedule["async_passes"].as_array().unwrap().iter()
        .map(|a| a["pass_index"].as_u64().unwrap())
        .collect();

    // At minimum the copy pass (3) is eligible for async queue scheduling.
    assert!(async_indices.contains(&3),
        "copy pass 3 must be in async_passes");

    // All async entries must have valid queue_type strings.
    for a in schedule["async_passes"].as_array().unwrap() {
        let queue = a["queue_type"].as_str().unwrap();
        assert!(!queue.is_empty(), "async queue_type must be non-empty");
    }

    // ----- Verify sync points match flat barriers -----
    let sync_points = schedule["sync_points"].as_array().unwrap();
    assert!(!sync_points.is_empty(), "must have sync points");

    let sync_boundaries: Vec<(usize, usize)> = sync_points.iter()
        .map(|sp| (sp["after_pass"].as_u64().unwrap() as usize,
                   sp["before_pass"].as_u64().unwrap() as usize))
        .collect();

    // Every barrier pair maps to a sync point.
    for &(from, to) in &barrier_pairs {
        assert!(sync_boundaries.contains(&(from, to)),
            "no sync point for barrier ({}, {})", from, to);
    }

    // Every sync point maps to at least one barrier.
    for &(after, before) in &sync_boundaries {
        assert!(barrier_pairs.contains(&(after, before)),
            "sync point ({},{}) has no corresponding barrier", after, before);
    }
}

// =============================================================================
// SECTION 12 -- No dangling references in output
// =============================================================================

#[test]
fn schedule_has_no_dangling_references() {
    // Build a graph where passes reference only existing resources.
    let mut p0 = IrPass::graphics(
        PassIndex(0), "draw",
        vec![ColorAttachment {
            resource: ResourceHandle(10),
            load_op: AttachmentLoadOp::Clear,
            store_op: AttachmentStoreOp::Store,
            ..Default::default()
        }],
        None,
        InstanceSource::Direct {
            index_count: 36, instance_count: 1,
            base_vertex: 0, first_index: 0, first_instance: 0,
        },
        ViewType::Texture2D,
    );
    p0.access_set.writes.push(ResourceHandle(10));

    let mut p1 = IrPass::compute(
        PassIndex(1), "compute",
        DispatchSource::Direct { group_count_x: 8, group_count_y: 8, group_count_z: 1 },
        ViewType::Storage,
    );
    p1.access_set.reads.push(ResourceHandle(10));

    let passes = vec![p0, p1];
    let resources = vec![IrResource::new(
        ResourceHandle(10), "shared_rt",
        tex_2d(1024, 1024, "rgba8unorm"),
        ResourceLifetime::Transient, ResourceState::Uninitialized,
    )];

    let compiled = CompiledFrameGraph::compile(passes, resources).unwrap();
    let schedule = compiled.emit_schedule_bridge();

    // All barriers reference valid pass indices.
    let order: Vec<u64> = schedule["execution_order"].as_array().unwrap().iter()
        .map(|v| v.as_u64().unwrap())
        .collect();

    for b in schedule["barriers"].as_array().unwrap() {
        assert!(order.contains(&b["from_pass"].as_u64().unwrap()));
        assert!(order.contains(&b["to_pass"].as_u64().unwrap()));
    }

    // All sync points reference valid pass indices.
    for sp in schedule["sync_points"].as_array().unwrap() {
        assert!(order.contains(&sp["after_pass"].as_u64().unwrap()));
        assert!(order.contains(&sp["before_pass"].as_u64().unwrap()));
    }
}

// =============================================================================
// SECTION 13 -- Uncompilable graph returns error from compile, not emit
// =============================================================================

#[test]
fn emit_schedule_bridge_does_not_panic_on_any_input() {
    // `emit_schedule_bridge` must never panic, even for graphs with unusual
    // dependency patterns that may or may not compile.
    let mut p0 = IrPass::compute(
        PassIndex(0), "a",
        DispatchSource::Direct { group_count_x: 1, group_count_y: 1, group_count_z: 1 },
        ViewType::Storage,
    );
    p0.access_set.writes.push(ResourceHandle(1));
    p0.access_set.reads.push(ResourceHandle(2));

    let mut p1 = IrPass::compute(
        PassIndex(1), "b",
        DispatchSource::Direct { group_count_x: 1, group_count_y: 1, group_count_z: 1 },
        ViewType::Storage,
    );
    p1.access_set.writes.push(ResourceHandle(2));
    p1.access_set.reads.push(ResourceHandle(1));

    let passes = vec![p0, p1];
    let resources = vec![
        IrResource::new(ResourceHandle(1), "x", storage_buf(64),
            ResourceLifetime::Transient, ResourceState::Uninitialized),
        IrResource::new(ResourceHandle(2), "y", storage_buf(64),
            ResourceLifetime::Transient, ResourceState::Uninitialized),
    ];

    // Whether or not this compiles depends on the DAG builder's edge detection.
    // Either path must not panic when calling emit_schedule_bridge.
    if let Ok(compiled) = CompiledFrameGraph::compile(passes, resources) {
        let schedule = compiled.emit_schedule_bridge();
        let obj = schedule.as_object()
            .expect("schedule must be a JSON object");
        assert!(obj.contains_key("execution_order"),
            "must produce execution_order");
    }
    // If it fails to compile, no emit_schedule_bridge call happens — also fine.
}

// =============================================================================
// SECTION 14 -- Duplicate passes
// =============================================================================

#[test]
fn duplicate_pass_indices_still_produce_valid_schedule() {
    // Two passes with the same index: compiler processes both.
    let mut p0a = IrPass::compute(
        PassIndex(0), "first",
        DispatchSource::Direct { group_count_x: 1, group_count_y: 1, group_count_z: 1 },
        ViewType::Storage,
    );
    p0a.access_set.writes.push(ResourceHandle(1));

    let mut p0b = IrPass::compute(
        PassIndex(0), "second",
        DispatchSource::Direct { group_count_x: 1, group_count_y: 1, group_count_z: 1 },
        ViewType::Storage,
    );
    p0b.access_set.reads.push(ResourceHandle(1));

    let passes = vec![p0a, p0b];
    let resources = vec![IrResource::new(
        ResourceHandle(1), "data", storage_buf(64),
        ResourceLifetime::Transient, ResourceState::Uninitialized,
    )];

    // This may compile or not — either way, emit_schedule_bridge must not panic.
    if let Ok(compiled) = CompiledFrameGraph::compile(passes, resources) {
        let schedule = compiled.emit_schedule_bridge();
        let obj = schedule.as_object()
            .expect("schedule must produce an object even with duplicate indices");
        assert!(obj.contains_key("execution_order"),
            "duplicate indices must still produce execution_order");
    }
}
