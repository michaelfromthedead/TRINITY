// SPDX-License-Identifier: MIT
//
// whitebox_bridge_validator.rs -- Whitebox structural tests for T-FG-7.2
// (BridgeValidator).
//
// These tests construct CompiledFrameGraph IR directly (same public API as the
// blackbox suite) but focus on validator-internal edge cases and structural
// combinations that the blackbox suite does not cover.
//
// Acceptance criteria (T-FG-7.2):
//   1.  Valid graph passes validation (Ok(()))
//   2.  Invalid barrier from-pass reference caught
//   3.  Invalid barrier to-pass reference caught
//   4.  Missing resource reference caught
//   5.  RAW hazard detected
//   6.  Topological sort violation caught
//   7.  Missing pass in execution order caught
//   8.  Multiple errors accumulated
//   9.  Valid graph with two-pass pipeline passes
//  10.  Imported resource does not trigger false RAW positive
//  11.  Transient Uninitialized resource triggers RAW on first read
//  12.  ResourceHandle::NONE in access-set read triggers resource error
//  13.  ResourceHandle::NONE in indirect buffer is silently skipped

use std::collections::HashMap;

use renderer_backend::frame_graph::{
    BridgeValidator, BufferDesc, CullStats,
    DispatchSource, EdgeType, InstanceSource, IrEdge, IrPass, IrResource,
    PassIndex, ResourceDesc, ResourceHandle, ResourceLifetime, ResourceState,
    TextureDesc, ViewType,
};

// ---------------------------------------------------------------------------
// Helper
// ---------------------------------------------------------------------------

/// Constructs a minimal valid `CompiledFrameGraph` with one compute pass that
/// writes one transient buffer resource.
fn make_bv_compiled() -> renderer_backend::frame_graph::CompiledFrameGraph {
    let res = IrResource::new(
        ResourceHandle(1),
        "test_res",
        ResourceDesc::Buffer(BufferDesc {
            size: 4096,
            usage: "storage".into(),
            is_indirect_arg: false,
        }),
        ResourceLifetime::Transient,
        ResourceState::Uninitialized,
    );
    let mut pass0 = IrPass::compute(
        PassIndex(0),
        "pass0",
        DispatchSource::Direct {
            group_count_x: 1,
            group_count_y: 1,
            group_count_z: 1,
        },
        ViewType::Storage,
    );
    pass0.access_set.writes.push(ResourceHandle(1));

    renderer_backend::frame_graph::CompiledFrameGraph {
        passes: vec![pass0],
        resources: vec![res],
        edges: vec![],
        order: vec![PassIndex(0)],
        depths: HashMap::new(),
        barriers: vec![],
        async_passes: vec![],
        eliminated_passes: vec![],
        cull_stats: CullStats::default(),
        parallel_regions: vec![],
        ..Default::default()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn test_bv_barrier_invalid_from_pass() {
    let mut graph = make_bv_compiled();
    graph.barriers.push((
        PassIndex(999),
        PassIndex(0),
        ResourceHandle(1),
        EdgeType::RAW,
        ResourceState::Uninitialized,
        ResourceState::ShaderRead,
    ));
    let result = BridgeValidator::validate(&graph);
    assert!(result.is_err());
    let errs = result.unwrap_err();
    assert!(
        errs.iter().any(|e| e.contains("Barrier") && e.contains("from-pass")),
        "Expected error mentioning 'Barrier' and 'from-pass', got: {:?}",
        errs
    );
}

#[test]
fn test_bv_barrier_invalid_to_pass() {
    let mut graph = make_bv_compiled();
    graph.barriers.push((
        PassIndex(0),
        PassIndex(999),
        ResourceHandle(1),
        EdgeType::RAW,
        ResourceState::Uninitialized,
        ResourceState::ShaderRead,
    ));
    let result = BridgeValidator::validate(&graph);
    assert!(result.is_err());
    let errs = result.unwrap_err();
    assert!(
        errs.iter().any(|e| e.contains("Barrier") && e.contains("to-pass")),
        "Expected error mentioning 'Barrier' and 'to-pass', got: {:?}",
        errs
    );
}

#[test]
fn test_bv_resource_handle_not_in_list() {
    let mut graph = make_bv_compiled();
    graph.passes[0].access_set.reads.push(ResourceHandle(42));
    let result = BridgeValidator::validate(&graph);
    assert!(result.is_err());
    let errs = result.unwrap_err();
    assert!(
        errs.iter().any(|e| e.contains("unknown resource handle")),
        "Expected error mentioning 'unknown resource handle', got: {:?}",
        errs
    );
}

#[test]
fn test_bv_raw_hazard_read_before_write() {
    let mut graph = make_bv_compiled();
    graph.passes[0].access_set.reads.push(ResourceHandle(2));
    let result = BridgeValidator::validate(&graph);
    assert!(result.is_err());
    let errs = result.unwrap_err();
    assert!(
        errs.iter().any(|e| e.contains("RAW hazard")),
        "Expected error mentioning 'RAW hazard', got: {:?}",
        errs
    );
}

#[test]
fn test_bv_topological_sort_violation() {
    let mut p0 = IrPass::compute(
        PassIndex(0),
        "pass0",
        DispatchSource::Direct {
            group_count_x: 1,
            group_count_y: 1,
            group_count_z: 1,
        },
        ViewType::Storage,
    );
    let mut p1 = IrPass::compute(
        PassIndex(1),
        "pass1",
        DispatchSource::Direct {
            group_count_x: 1,
            group_count_y: 1,
            group_count_z: 1,
        },
        ViewType::Storage,
    );
    p0.access_set.reads.push(ResourceHandle(1));
    p1.access_set.writes.push(ResourceHandle(1));

    let res = IrResource::new(
        ResourceHandle(1),
        "shared",
        ResourceDesc::Buffer(BufferDesc {
            size: 4096,
            usage: "storage".into(),
            is_indirect_arg: false,
        }),
        ResourceLifetime::Transient,
        ResourceState::Uninitialized,
    );

    // Edge: P1 writes R1, P0 reads R1 => P1 must be before P0.
    // Order has P0(0) before P1(1) => violation.
    let graph = renderer_backend::frame_graph::CompiledFrameGraph {
        passes: vec![p0, p1],
        resources: vec![res],
        edges: vec![IrEdge::new(
            PassIndex(1), PassIndex(0), ResourceHandle(1), EdgeType::RAW,
        )],
        order: vec![PassIndex(0), PassIndex(1)],
        depths: HashMap::new(),
        barriers: vec![],
        async_passes: vec![],
        eliminated_passes: vec![],
        cull_stats: CullStats::default(),
        parallel_regions: vec![],
        ..Default::default()
    };

    let result = BridgeValidator::validate(&graph);
    assert!(result.is_err());
    let errs = result.unwrap_err();
    assert!(
        errs.iter().any(|e| e.contains("Topological sort violation")),
        "Expected error mentioning 'Topological sort violation', got: {:?}",
        errs
    );
}

#[test]
fn test_bv_execution_order_pass_not_in_list() {
    let res = IrResource::new(
        ResourceHandle(1),
        "test_res",
        ResourceDesc::Buffer(BufferDesc {
            size: 4096,
            usage: "storage".into(),
            is_indirect_arg: false,
        }),
        ResourceLifetime::Transient,
        ResourceState::Uninitialized,
    );
    let pass0 = IrPass::compute(
        PassIndex(0),
        "pass0",
        DispatchSource::Direct {
            group_count_x: 1,
            group_count_y: 1,
            group_count_z: 1,
        },
        ViewType::Storage,
    );

    let graph = renderer_backend::frame_graph::CompiledFrameGraph {
        passes: vec![pass0],
        resources: vec![res],
        edges: vec![],
        order: vec![PassIndex(0), PassIndex(1)],
        depths: HashMap::new(),
        barriers: vec![],
        async_passes: vec![],
        eliminated_passes: vec![],
        cull_stats: CullStats::default(),
        parallel_regions: vec![],
        ..Default::default()
    };

    let result = BridgeValidator::validate(&graph);
    assert!(result.is_err());
    let errs = result.unwrap_err();
    assert!(
        errs.iter().any(|e| e.contains("Execution order references pass index")),
        "Expected error mentioning 'Execution order references pass index', got: {:?}",
        errs
    );
}

#[test]
fn test_bv_valid_graph_passes_all_checks() {
    let res = IrResource::new(
        ResourceHandle(1),
        "shared",
        ResourceDesc::Buffer(BufferDesc {
            size: 4096,
            usage: "storage".into(),
            is_indirect_arg: false,
        }),
        ResourceLifetime::Transient,
        ResourceState::Uninitialized,
    );
    let mut p0 = IrPass::compute(
        PassIndex(0),
        "writer",
        DispatchSource::Direct {
            group_count_x: 1,
            group_count_y: 1,
            group_count_z: 1,
        },
        ViewType::Storage,
    );
    p0.access_set.writes.push(ResourceHandle(1));

    let mut p1 = IrPass::compute(
        PassIndex(1),
        "reader",
        DispatchSource::Direct {
            group_count_x: 1,
            group_count_y: 1,
            group_count_z: 1,
        },
        ViewType::Storage,
    );
    p1.access_set.reads.push(ResourceHandle(1));

    let graph = renderer_backend::frame_graph::CompiledFrameGraph {
        passes: vec![p0, p1],
        resources: vec![res],
        edges: vec![IrEdge::new(
            PassIndex(0), PassIndex(1), ResourceHandle(1), EdgeType::RAW,
        )],
        order: vec![PassIndex(0), PassIndex(1)],
        depths: HashMap::new(),
        barriers: vec![],
        async_passes: vec![],
        eliminated_passes: vec![],
        cull_stats: CullStats::default(),
        parallel_regions: vec![],
        ..Default::default()
    };

    let result = BridgeValidator::validate(&graph);
    assert!(
        result.is_ok(),
        "Expected Ok, got Err: {:?}",
        result
    );
}

#[test]
fn test_bv_multiple_errors_accumulated() {
    let mut graph = make_bv_compiled();
    graph.barriers.push((
        PassIndex(999),
        PassIndex(0),
        ResourceHandle(1),
        EdgeType::RAW,
        ResourceState::Uninitialized,
        ResourceState::ShaderRead,
    ));
    graph.passes[0].access_set.reads.push(ResourceHandle(42));
    graph.order.push(PassIndex(999));

    let result = BridgeValidator::validate(&graph);
    assert!(result.is_err());
    let errs = result.unwrap_err();
    assert!(
        errs.len() >= 3,
        "Expected at least 3 errors, got {}: {:?}",
        errs.len(),
        errs
    );
}

#[test]
fn test_bv_imported_resource_treated_as_written() {
    let imported_res = IrResource::new(
        ResourceHandle(1),
        "imported_rt",
        ResourceDesc::Texture2D(TextureDesc {
            width: 256,
            height: 256,
            mip_levels: 1,
            array_layers: 1,
            format: "rgba8unorm".into(),
        }),
        ResourceLifetime::Imported,
        ResourceState::ColorAttachment,
    );
    let mut pass0 = IrPass::compute(
        PassIndex(0),
        "reader",
        DispatchSource::Direct {
            group_count_x: 1,
            group_count_y: 1,
            group_count_z: 1,
        },
        ViewType::Storage,
    );
    pass0.access_set.reads.push(ResourceHandle(1));

    let graph = renderer_backend::frame_graph::CompiledFrameGraph {
        passes: vec![pass0],
        resources: vec![imported_res],
        edges: vec![],
        order: vec![PassIndex(0)],
        depths: HashMap::new(),
        barriers: vec![],
        async_passes: vec![],
        eliminated_passes: vec![],
        cull_stats: CullStats::default(),
        parallel_regions: vec![],
        ..Default::default()
    };

    let result = BridgeValidator::validate(&graph);
    assert!(
        result.is_ok(),
        "Expected Ok for imported resource read, got Err: {:?}",
        result
    );
}

#[test]
fn test_bv_transient_uninitialized_raw_hazard() {
    let trans_res = IrResource::new(
        ResourceHandle(1),
        "transient_rt",
        ResourceDesc::Texture2D(TextureDesc {
            width: 256,
            height: 256,
            mip_levels: 1,
            array_layers: 1,
            format: "rgba8unorm".into(),
        }),
        ResourceLifetime::Transient,
        ResourceState::Uninitialized,
    );
    let mut pass0 = IrPass::compute(
        PassIndex(0),
        "reader",
        DispatchSource::Direct {
            group_count_x: 1,
            group_count_y: 1,
            group_count_z: 1,
        },
        ViewType::Storage,
    );
    pass0.access_set.reads.push(ResourceHandle(1));

    let graph = renderer_backend::frame_graph::CompiledFrameGraph {
        passes: vec![pass0],
        resources: vec![trans_res],
        edges: vec![],
        order: vec![PassIndex(0)],
        depths: HashMap::new(),
        barriers: vec![],
        async_passes: vec![],
        eliminated_passes: vec![],
        cull_stats: CullStats::default(),
        parallel_regions: vec![],
        ..Default::default()
    };

    let result = BridgeValidator::validate(&graph);
    assert!(result.is_err());
    let errs = result.unwrap_err();
    assert!(
        errs.iter().any(|e| e.contains("RAW hazard")),
        "Expected error mentioning 'RAW hazard', got: {:?}",
        errs
    );
}

#[test]
fn test_bv_none_handle_triggers_resource_error() {
    let res = IrResource::new(
        ResourceHandle(1),
        "real_res",
        ResourceDesc::Buffer(BufferDesc {
            size: 4096,
            usage: "storage".into(),
            is_indirect_arg: false,
        }),
        ResourceLifetime::Transient,
        ResourceState::Uninitialized,
    );
    let mut pass0 = IrPass::compute(
        PassIndex(0),
        "pass0",
        DispatchSource::Direct {
            group_count_x: 1,
            group_count_y: 1,
            group_count_z: 1,
        },
        ViewType::Storage,
    );
    pass0.access_set.reads.push(ResourceHandle::NONE);

    let graph = renderer_backend::frame_graph::CompiledFrameGraph {
        passes: vec![pass0],
        resources: vec![res],
        edges: vec![],
        order: vec![PassIndex(0)],
        depths: HashMap::new(),
        barriers: vec![],
        async_passes: vec![],
        eliminated_passes: vec![],
        cull_stats: CullStats::default(),
        parallel_regions: vec![],
        ..Default::default()
    };

    let result = BridgeValidator::validate(&graph);
    assert!(result.is_err());
    let errs = result.unwrap_err();
    assert!(
        errs.iter().any(|e| e.contains("unknown resource handle")),
        "Expected error mentioning 'unknown resource handle', got: {:?}",
        errs
    );
}

#[test]
fn test_bv_none_indirect_buffer_skipped() {
    let res = IrResource::new(
        ResourceHandle(1),
        "real_res",
        ResourceDesc::Buffer(BufferDesc {
            size: 4096,
            usage: "storage".into(),
            is_indirect_arg: false,
        }),
        ResourceLifetime::Transient,
        ResourceState::Uninitialized,
    );
    let pass0 = IrPass::graphics(
        PassIndex(0),
        "gfx_pass",
        vec![],
        None,
        InstanceSource::Indirect {
            buffer: ResourceHandle::NONE,
            offset: 0,
            draw_count: 0,
            stride: 0,
        },
        ViewType::Texture2D,
    );

    let graph = renderer_backend::frame_graph::CompiledFrameGraph {
        passes: vec![pass0],
        resources: vec![res],
        edges: vec![],
        order: vec![PassIndex(0)],
        depths: HashMap::new(),
        barriers: vec![],
        async_passes: vec![],
        eliminated_passes: vec![],
        cull_stats: CullStats::default(),
        parallel_regions: vec![],
        ..Default::default()
    };

    let result = BridgeValidator::validate(&graph);
    assert!(
        result.is_ok(),
        "Expected Ok for NONE indirect buffer, got Err: {:?}",
        result
    );
}
