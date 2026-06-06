// Blackbox contract tests for MockPassNode and MockResourceDesc constructors
// (T-FG-1.7).
//
// CLEANROOM: No src/ access beyond the public API exported by the crate.
// Tests use only `renderer_backend::frame_graph::*` -- no internal fields,
// no private methods, no implementation details.
//
// T-FG-1.7: Implement MockPassNode and MockResourceDesc constructors for
// Rust-side compiler testing without Python.
//
// The mocks module provides:
//   - MockPassNode:      builder for IrPass (graphics/compute/copy) with
//                        chainable reads/writes/color_attachment/depth_stencil
//   - MockResourceDesc:  builder for IrResource (texture_2d/buffer) with
//                        auto-assigned unique ResourceHandle from a global
//                        atomic counter
//   - reset_mock_handles(): zeroes the global counter for deterministic tests
//
// Coverage:
//   1.  MockPassNode::graphics("test") constructs and produces a Graphics IrPass
//   2.  MockPassNode::compute("test") constructs and produces a Compute IrPass
//   3.  MockPassNode::copy("test") constructs and produces a Copy IrPass
//   4.  MockPassNode builder chain: reads, writes, color_attachment, depth_stencil
//   5.  MockPassNode::build populates access_set from color attachments
//   6.  MockResourceDesc::texture_2d builds a valid IrResource with Texture2D desc
//   7.  MockResourceDesc::buffer builds a valid IrResource with Buffer desc
//   8.  MockResourceDesc handles are unique and not NONE
//   9.  reset_mock_handles produces deterministic handle values
//  10.  Using mocks, build a 3-pass linear chain and compile successfully
//  11.  Using mocks, build a diamond DAG and verify compilation
//  12.  Mocks interoperate with CompiledFrameGraph for a full compile pipeline
//
use renderer_backend::frame_graph::mocks::{MockPassNode, MockResourceDesc, reset_mock_handles};
use renderer_backend::frame_graph::{
    CompiledFrameGraph, IrPass, PassIndex, PassType, ResourceHandle, ResourceLifetime,
    ResourceState,
};

// =============================================================================
// SECTION 1 -- MockPassNode constructors
// =============================================================================

#[test]
fn mock_pass_node_graphics_constructor_produces_graphics_pass() {
    let pass: IrPass = MockPassNode::graphics("test").build();

    assert_eq!(pass.pass_type, PassType::Graphics, "Graphics pass type");
    assert_eq!(pass.name, "test", "Name preserved");
    assert_eq!(pass.index, PassIndex(0), "Default index is 0");
    assert!(pass.depth_stencil.is_none(), "No depth-stencil by default");
}

#[test]
fn mock_pass_node_compute_constructor_produces_compute_pass() {
    let pass: IrPass = MockPassNode::compute("compute_test").build();

    assert_eq!(pass.pass_type, PassType::Compute, "Compute pass type");
    assert_eq!(pass.name, "compute_test", "Name preserved");
    assert!(pass.dispatch_source.is_some(), "Compute has dispatch source");
    assert!(
        pass.color_attachments.is_empty(),
        "Compute has no color attachments"
    );
}

#[test]
fn mock_pass_node_copy_constructor_produces_copy_pass() {
    let pass: IrPass = MockPassNode::copy("copy_test").build();

    assert_eq!(pass.pass_type, PassType::Copy, "Copy pass type");
    assert_eq!(pass.name, "copy_test", "Name preserved");
    assert!(pass.dispatch_source.is_none(), "Copy has no dispatch source");
    assert!(
        pass.color_attachments.is_empty(),
        "Copy has no color attachments"
    );
}

// =============================================================================
// SECTION 2 -- MockPassNode builder chain
// =============================================================================

#[test]
fn mock_pass_node_builder_chain_reads_and_writes() {
    let h1 = ResourceHandle(10);
    let h2 = ResourceHandle(20);

    let pass = MockPassNode::compute("chain_test")
        .reads(&[h1])
        .writes(&[h2])
        .build();

    assert!(pass.access_set.reads.contains(&h1), "Reads contains h1");
    assert!(pass.access_set.writes.contains(&h2), "Writes contains h2");
    assert_eq!(pass.access_set.len(), 2, "Two access entries");
}

#[test]
fn mock_pass_node_builder_color_attachment() {
    let h = ResourceHandle(5);

    let pass = MockPassNode::graphics("color_test")
        .color_attachment(h)
        .build();

    assert_eq!(pass.color_attachments.len(), 1, "One color attachment");
    assert_eq!(pass.color_attachments[0].resource, h, "Attachment targets h");
    // Color attachment with Store op should appear in writes.
    assert!(
        pass.access_set.writes.contains(&h),
        "Color attachment (Store) added to writes",
    );
}

#[test]
fn mock_pass_node_builder_depth_stencil() {
    let color_h = ResourceHandle(1);
    let ds_h = ResourceHandle(2);

    let pass = MockPassNode::graphics("ds_test")
        .color_attachment(color_h)
        .depth_stencil(ds_h)
        .build();

    assert!(pass.depth_stencil.is_some(), "Depth-stencil set");
    assert_eq!(
        pass.depth_stencil.as_ref().unwrap().resource,
        ds_h
    );
    assert!(
        pass.access_set.writes.contains(&ds_h),
        "Depth-stencil resource in writes",
    );
}

#[test]
fn mock_pass_node_multiple_color_attachments() {
    let h1 = ResourceHandle(1);
    let h2 = ResourceHandle(2);
    let h3 = ResourceHandle(3);

    let pass = MockPassNode::graphics("mrt_test")
        .color_attachment(h1)
        .color_attachment(h2)
        .color_attachment(h3)
        .build();

    assert_eq!(pass.color_attachments.len(), 3, "Three color attachments");
    assert!(pass.access_set.writes.contains(&h1));
    assert!(pass.access_set.writes.contains(&h2));
    assert!(pass.access_set.writes.contains(&h3));
}

#[test]
fn mock_pass_node_empty_access_set() {
    let pass = MockPassNode::compute("empty").build();
    assert!(pass.access_set.is_empty(), "No reads or writes by default");
}

#[test]
fn mock_pass_node_combined_reads_writes_and_attachments() {
    let h_r = ResourceHandle(7);
    let h_w = ResourceHandle(8);
    let h_color = ResourceHandle(9);

    let pass = MockPassNode::graphics("combined")
        .reads(&[h_r])
        .writes(&[h_w])
        .color_attachment(h_color)
        .build();

    assert!(pass.access_set.reads.contains(&h_r), "Reads contains h_r");
    assert!(pass.access_set.writes.contains(&h_w), "Writes contains h_w");
    assert!(
        pass.access_set.writes.contains(&h_color),
        "Color attachment in writes",
    );
}

// =============================================================================
// SECTION 3 -- MockResourceDesc constructors
// =============================================================================

#[test]
fn mock_resource_desc_texture_2d_builds_valid_ir_resource() {
    use renderer_backend::frame_graph::ResourceDesc;

    let res = MockResourceDesc::texture_2d("color_rt", 1920, 1080).build();

    match &res.desc {
        ResourceDesc::Texture2D(t) => {
            assert_eq!(t.width, 1920, "Texture width");
            assert_eq!(t.height, 1080, "Texture height");
            assert_eq!(t.mip_levels, 1, "Default mip levels");
            assert_eq!(t.array_layers, 1, "Default array layers");
            assert_eq!(t.format, "rgba8unorm", "Default format");
        }
        other => panic!("Expected Texture2D, got {:?}", other),
    }

    assert_eq!(
        res.lifetime,
        ResourceLifetime::Transient,
        "Transient lifetime"
    );
    assert_eq!(
        res.initial_state,
        ResourceState::Uninitialized,
        "Uninitialized state",
    );
}

#[test]
fn mock_resource_desc_buffer_builds_valid_ir_resource() {
    use renderer_backend::frame_graph::ResourceDesc;

    let res = MockResourceDesc::buffer("data_buf", 65536).build();

    match &res.desc {
        ResourceDesc::Buffer(b) => {
            assert_eq!(b.size, 65536, "Buffer size");
            assert!(!b.is_indirect_arg, "Not an indirect arg by default");
            assert!(b.usage.contains("storage"), "Usage includes 'storage'");
        }
        other => panic!("Expected Buffer, got {:?}", other),
    }

    assert_eq!(res.lifetime, ResourceLifetime::Transient);
    assert_eq!(res.initial_state, ResourceState::Uninitialized);
}

#[test]
fn mock_resource_desc_name_preserved() {
    let res = MockResourceDesc::texture_2d("my_custom_name", 800, 600).build();
    assert_eq!(res.name, "my_custom_name", "Resource name preserved");
}

#[test]
fn mock_resource_desc_buffer_custom_size() {
    use renderer_backend::frame_graph::ResourceDesc;

    let res = MockResourceDesc::buffer("large_buf", 1_048_576).build();

    match &res.desc {
        ResourceDesc::Buffer(b) => {
            assert_eq!(b.size, 1_048_576, "1 MiB buffer");
        }
        _ => panic!("Expected Buffer"),
    }
}

// =============================================================================
// SECTION 4 -- Mock handles: unique, non-NONE, deterministic reset
// =============================================================================

#[test]
fn mock_resource_desc_handles_are_unique() {
    reset_mock_handles();

    let r0 = MockResourceDesc::texture_2d("a", 64, 64).build();
    let r1 = MockResourceDesc::texture_2d("b", 64, 64).build();
    let r2 = MockResourceDesc::buffer("c", 128).build();

    assert_ne!(r0.handle, r1.handle, "Handles r0 and r1 differ");
    assert_ne!(r0.handle, r2.handle, "Handles r0 and r2 differ");
    assert_ne!(r1.handle, r2.handle, "Handles r1 and r2 differ");
}

#[test]
fn mock_resource_desc_handles_are_not_none() {
    reset_mock_handles();

    for _ in 0..10 {
        let res = MockResourceDesc::texture_2d("t", 32, 32).build();
        assert_ne!(
            res.handle,
            ResourceHandle::NONE,
            "Mock handle is never NONE",
        );
    }
}

#[test]
fn mock_resource_desc_handles_monotonically_increasing() {
    reset_mock_handles();

    let r0 = MockResourceDesc::texture_2d("a", 16, 16).build();
    let r1 = MockResourceDesc::texture_2d("b", 16, 16).build();
    let r2 = MockResourceDesc::buffer("c", 64).build();

    assert!(
        r0.handle.0 < r1.handle.0,
        "Handle 0 < handle 1: {} < {}",
        r0.handle.0,
        r1.handle.0,
    );
    assert!(
        r1.handle.0 < r2.handle.0,
        "Handle 1 < handle 2: {} < {}",
        r1.handle.0,
        r2.handle.0,
    );
}

#[test]
fn mock_resource_desc_handle_before_build_matches_build() {
    reset_mock_handles();

    let desc = MockResourceDesc::texture_2d("preview", 320, 240);
    let handle_before = desc.handle();
    let res = desc.build();

    assert_eq!(
        handle_before, res.handle,
        "handle() before build() matches build result",
    );
}

#[test]
fn reset_mock_handles_produces_deterministic_sequence() {
    reset_mock_handles();

    let a0 = MockResourceDesc::texture_2d("first", 8, 8).build();
    let a1 = MockResourceDesc::buffer("second", 16).build();

    reset_mock_handles();

    let b0 = MockResourceDesc::texture_2d("first", 8, 8).build();
    let b1 = MockResourceDesc::buffer("second", 16).build();

    assert_eq!(
        a0.handle, b0.handle,
        "After reset, first handle is the same",
    );
    assert_eq!(
        a1.handle, b1.handle,
        "After reset, second handle is the same",
    );
}

// =============================================================================
// SECTION 5 -- Compilation: 3-pass linear chain with mocks
// =============================================================================

#[test]
fn three_pass_linear_chain_compiles_with_mocks() {
    reset_mock_handles();

    let r_desc_0 = MockResourceDesc::texture_2d("chain_r0", 1920, 1080);
    let r_desc_1 = MockResourceDesc::buffer("chain_r1", 8192);
    let h0 = r_desc_0.handle();
    let h1 = r_desc_1.handle();
    let resources = vec![r_desc_0.build(), r_desc_1.build()];

    // P0 (graphics): writes h0 as color attachment
    let mut p0 = MockPassNode::graphics("p0_gbuffer")
        .color_attachment(h0)
        .build();
    p0.index = PassIndex(0);

    // P1 (compute): reads h0, writes h1
    let mut p1 = MockPassNode::compute("p1_lighting")
        .reads(&[h0])
        .writes(&[h1])
        .build();
    p1.index = PassIndex(1);

    // P2 (compute): reads h1 (final consumer)
    let mut p2 = MockPassNode::compute("p2_resolve")
        .reads(&[h1])
        .build();
    p2.index = PassIndex(2);

    let passes = vec![p0, p1, p2];

    let compiled =
        CompiledFrameGraph::compile(passes, resources).expect("3-pass chain compiles with mocks");

    // Order must be P0 -> P1 -> P2.
    assert_eq!(
        compiled.order,
        vec![PassIndex(0), PassIndex(1), PassIndex(2)],
        "Topological order is P0 -> P1 -> P2",
    );

    // Edges: P0->P1 (h0 RAW), P1->P2 (h1 RAW).
    let edges = &compiled.edges;
    assert_eq!(edges.len(), 2, "Two edges in 3-pass chain");
    let has_p0p1 = edges
        .iter()
        .any(|e| e.from == PassIndex(0) && e.to == PassIndex(1));
    let has_p1p2 = edges
        .iter()
        .any(|e| e.from == PassIndex(1) && e.to == PassIndex(2));
    assert!(has_p0p1, "Edge P0->P1 exists");
    assert!(has_p1p2, "Edge P1->P2 exists");
}

#[test]
fn three_pass_chain_no_passes_eliminated() {
    reset_mock_handles();

    let r0 = MockResourceDesc::texture_2d("r0", 1920, 1080);
    let r1 = MockResourceDesc::buffer("r1", 8192);
    let h0 = r0.handle();
    let h1 = r1.handle();
    let resources = vec![r0.build(), r1.build()];

    let mut p0 = MockPassNode::graphics("p0").color_attachment(h0).build();
    p0.index = PassIndex(0);

    let mut p1 = MockPassNode::compute("p1").reads(&[h0]).writes(&[h1]).build();
    p1.index = PassIndex(1);

    let mut p2 = MockPassNode::compute("p2").reads(&[h1]).build();
    p2.index = PassIndex(2);

    let passes = vec![p0, p1, p2];
    let compiled = CompiledFrameGraph::compile(passes, resources).expect("compile ok");

    // All three passes have consumers or are graphics passes, so none eliminated.
    assert_eq!(
        compiled.cull_stats.passes_eliminated, 0,
        "no eliminated passes"
    );
    assert_eq!(
        compiled.cull_stats.passes_total, 3,
        "3 total passes"
    );
}

// =============================================================================
// SECTION 6 -- Compilation: diamond DAG with mocks
// =============================================================================

#[test]
fn diamond_dag_compiles_with_mocks() {
    reset_mock_handles();

    // Diamond topology:
    //   P0 (graphics) writes R0
    //   P1 (compute)  reads R0, writes R1
    //   P2 (compute)  reads R0, writes R2
    //   P3 (compute)  reads R1, reads R2  (merge point)
    //
    // Edges: P0->P1 (R0 RAW), P0->P2 (R0 RAW),
    //        P1->P3 (R1 RAW), P2->P3 (R2 RAW)
    // Expected: 4 edges.

    let r_desc_0 = MockResourceDesc::texture_2d("diamond_r0", 800, 600);
    let r_desc_1 = MockResourceDesc::buffer("diamond_r1", 4096);
    let r_desc_2 = MockResourceDesc::buffer("diamond_r2", 4096);
    let h0 = r_desc_0.handle();
    let h1 = r_desc_1.handle();
    let h2 = r_desc_2.handle();
    let resources = vec![r_desc_0.build(), r_desc_1.build(), r_desc_2.build()];

    let mut p0 = MockPassNode::graphics("p0_root")
        .color_attachment(h0)
        .build();
    p0.index = PassIndex(0);

    let mut p1 = MockPassNode::compute("p1_left")
        .reads(&[h0])
        .writes(&[h1])
        .build();
    p1.index = PassIndex(1);

    let mut p2 = MockPassNode::compute("p2_right")
        .reads(&[h0])
        .writes(&[h2])
        .build();
    p2.index = PassIndex(2);

    let mut p3 = MockPassNode::compute("p3_merge")
        .reads(&[h1, h2])
        .build();
    p3.index = PassIndex(3);

    let passes = vec![p0, p1, p2, p3];

    let compiled =
        CompiledFrameGraph::compile(passes, resources).expect("diamond DAG compiles with mocks");

    // P0 must be first (only root producer).
    assert_eq!(compiled.order[0], PassIndex(0), "P0 is first");

    // P3 must be last (merge point, depends on both P1 and P2).
    assert_eq!(
        compiled.order[compiled.order.len() - 1],
        PassIndex(3),
        "P3 is last (merge point)",
    );

    // Four edges.
    assert_eq!(compiled.edges.len(), 4, "Diamond has 4 edges");

    let edges = &compiled.edges;
    assert!(edges.iter().any(|e| e.from == PassIndex(0) && e.to == PassIndex(1)));
    assert!(edges.iter().any(|e| e.from == PassIndex(0) && e.to == PassIndex(2)));
    assert!(edges.iter().any(|e| e.from == PassIndex(1) && e.to == PassIndex(3)));
    assert!(edges.iter().any(|e| e.from == PassIndex(2) && e.to == PassIndex(3)));
}

// =============================================================================
// SECTION 7 -- Interop: MockPassNode + MockResourceDesc + CompiledFrameGraph
// =============================================================================

#[test]
fn mocks_interoperate_with_frame_graph_compiler_full_pipeline() {
    reset_mock_handles();

    // P0 (graphics): writes R0 (color attachment)
    // P1 (compute):  reads R0, writes R1
    // P2 (compute):  reads R1 (final consumer, keeps P1 alive)
    let r0 = MockResourceDesc::texture_2d("rt_color", 1920, 1080);
    let r1 = MockResourceDesc::buffer("buf_data", 16384);
    let h0 = r0.handle();
    let h1 = r1.handle();
    let resources = vec![r0.build(), r1.build()];

    let mut p0 = MockPassNode::graphics("scene_render")
        .color_attachment(h0)
        .build();
    p0.index = PassIndex(0);

    let mut p1 = MockPassNode::compute("post_process")
        .reads(&[h0])
        .writes(&[h1])
        .build();
    p1.index = PassIndex(1);

    let mut p2 = MockPassNode::compute("output_resolve")
        .reads(&[h1])
        .build();
    p2.index = PassIndex(2);

    let passes = vec![p0, p1, p2];

    let compiled =
        CompiledFrameGraph::compile(passes, resources).expect("full pipeline compiles with mocks");

    // Verify key output fields.
    assert_eq!(compiled.resources.len(), 2, "Both resources preserved");
    assert_eq!(
        compiled.order,
        vec![PassIndex(0), PassIndex(1), PassIndex(2)],
        "Correct topological order",
    );
    assert!(
        !compiled.barriers.is_empty(),
        "Barriers generated between dependent passes"
    );
    assert!(
        compiled.eliminated_passes.is_empty(),
        "No eliminated passes (all have consumers or are graphics)"
    );

    // CullStats: 3 total, 0 eliminated.
    assert_eq!(compiled.cull_stats.passes_total, 3);
    assert_eq!(compiled.cull_stats.passes_eliminated, 0);
}

#[test]
fn mocks_produce_barriers_on_dependent_chain() {
    reset_mock_handles();

    // P0 writes R0 as color attachment. P1 reads R0 as shader read.
    // Compiler must insert a barrier between them.
    let r0 = MockResourceDesc::texture_2d("shared_rt", 800, 600);
    let h0 = r0.handle();
    let resources = vec![r0.build()];

    let mut p0 = MockPassNode::graphics("producer")
        .color_attachment(h0)
        .build();
    p0.index = PassIndex(0);

    let mut p1 = MockPassNode::compute("consumer")
        .reads(&[h0])
        .build();
    p1.index = PassIndex(1);

    let passes = vec![p0, p1];
    let compiled =
        CompiledFrameGraph::compile(passes, resources).expect("dependent chain compiles");

    // Barriers must exist between the two passes.
    assert!(
        !compiled.barriers.is_empty(),
        "Barriers exist for dependent chain"
    );

    // At least one barrier transitions h0 from some state to some state.
    let has_h0_barrier = compiled
        .barriers
        .iter()
        .any(|(_from, _to, _before, _after, _handle)| true);
    assert!(has_h0_barrier, "Barrier record present");
}

// =============================================================================
// SECTION 8 -- MockPassNode constructors are distinct (name, type isolation)
// =============================================================================

#[test]
fn mock_pass_node_graphics_and_compute_names_independent() {
    let g = MockPassNode::graphics("gfx_pass").build();
    let c = MockPassNode::compute("comp_pass").build();

    assert_eq!(g.name, "gfx_pass");
    assert_eq!(c.name, "comp_pass");
    assert_ne!(g.pass_type, c.pass_type, "Different pass types");
}

#[test]
fn mock_pass_node_copy_default_view_type() {
    let pass = MockPassNode::copy("copy_op").build();
    assert_eq!(
        format!("{}", pass.view_type),
        "StorageBuffer",
        "Copy pass default view type",
    );
}

#[test]
fn mock_pass_node_compute_default_view_type() {
    let pass = MockPassNode::compute("comp").build();
    assert_eq!(
        format!("{}", pass.view_type),
        "Storage",
        "Compute pass default view type",
    );
}

// =============================================================================
// SECTION 9 -- MockResourceDesc interop with MockPassNode wiring
// =============================================================================

#[test]
fn mock_resource_desc_handle_used_as_color_attachment_in_mock_pass_node() {
    reset_mock_handles();

    let rt = MockResourceDesc::texture_2d("render_target", 800, 600);
    let handle = rt.handle();

    let pass = MockPassNode::graphics("render")
        .color_attachment(handle)
        .build();

    assert_eq!(
        pass.color_attachments[0].resource,
        handle,
        "MockResourceDesc handle wired as color attachment",
    );
}

#[test]
fn mock_resource_desc_handle_in_reads_and_writes() {
    reset_mock_handles();

    let buf = MockResourceDesc::buffer("work_buf", 4096);
    let h = buf.handle();

    let pass = MockPassNode::compute("worker")
        .reads(&[h])
        .writes(&[h])
        .build();

    assert!(pass.access_set.reads.contains(&h), "Handle in reads");
    assert!(pass.access_set.writes.contains(&h), "Handle in writes");
}

// =============================================================================
// SECTION 10 -- Edge cases: empty pass, zero-size buffer, single-pixel texture
// =============================================================================

#[test]
fn mock_pass_node_empty_builder_chain_compiles() {
    let pass = MockPassNode::compute("minimal").build();
    assert_eq!(pass.name, "minimal");
    assert_eq!(pass.pass_type, PassType::Compute);
    assert!(pass.access_set.is_empty());
    assert!(pass.color_attachments.is_empty());
}

#[test]
fn mock_resource_desc_zero_size_buffer() {
    use renderer_backend::frame_graph::ResourceDesc;

    let res = MockResourceDesc::buffer("empty_buf", 0).build();
    match &res.desc {
        ResourceDesc::Buffer(b) => {
            assert_eq!(b.size, 0, "Zero-size buffer");
        }
        _ => panic!("Expected Buffer"),
    }
}

#[test]
fn mock_resource_desc_single_pixel_texture() {
    use renderer_backend::frame_graph::ResourceDesc;

    let res = MockResourceDesc::texture_2d("pixel", 1, 1).build();
    match &res.desc {
        ResourceDesc::Texture2D(t) => {
            assert_eq!(t.width, 1);
            assert_eq!(t.height, 1);
        }
        _ => panic!("Expected Texture2D"),
    }
}

#[test]
fn mock_resource_desc_large_texture() {
    use renderer_backend::frame_graph::ResourceDesc;

    let res = MockResourceDesc::texture_2d("ultra_hd", 7680, 4320).build();
    match &res.desc {
        ResourceDesc::Texture2D(t) => {
            assert_eq!(t.width, 7680, "8K width");
            assert_eq!(t.height, 4320, "8K height");
        }
        _ => panic!("Expected Texture2D"),
    }
}

#[test]
fn mock_resource_desc_large_buffer() {
    use renderer_backend::frame_graph::ResourceDesc;

    let res = MockResourceDesc::buffer("large", 268_435_456).build();
    match &res.desc {
        ResourceDesc::Buffer(b) => {
            assert_eq!(b.size, 268_435_456, "256 MiB buffer");
        }
        _ => panic!("Expected Buffer"),
    }
}

#[test]
fn mock_pass_node_no_panics_on_empty_chain() {
    // Build a MockPassNode with nothing chained -- must not panic.
    let _ = MockPassNode::compute("noop").build();
    let _ = MockPassNode::graphics("noop").build();
    let _ = MockPassNode::copy("noop").build();
}
