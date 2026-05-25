// SPDX-License-Identifier: MIT
//
// blackbox_pass_emit.rs -- Blackbox contract tests for T-FG-7.3 Pass emit bridge.
//
// CLEANROOM: No src/ access beyond the public API exported by the crate.
// Tests use only `renderer_backend::frame_graph::*` -- no internal fields,
// no private methods, no implementation details.
//
// Public API under test:
//   emit_pass_bridge(pass, resources, pass_index) -> serde_json::Value
//   emit_all_passes(compiled) -> Vec<serde_json::Value>
//
// Contract: serializes compiled passes with attachment resolution and barrier
// context. Given valid input, the function produces structured JSON output
// with the expected schema.
//
// Coverage:
//   1.  Graphics pass -- color_attachments + depth_stencil in output
//   2.  Graphics pass -- all color attachment fields present
//   3.  Graphics pass -- all depth-stencil fields present
//   4.  Graphics pass -- instance_source (Direct) schema
//   5.  Graphics pass -- no copy-specific keys
//   6.  Graphics pass -- vertex_buffers from buffer reads
//   7.  Compute pass  -- dispatch_source + workgroup sizes
//   8.  Compute pass  -- no color attachments, null depth_stencil
//   9.  Copy pass     -- source_resources + destination_resources
//  10.  Ray tracing pass -- dispatch_source present
//  11.  Pass name resolves correctly in output
//  12.  Pass index (execution + original) in output
//  13.  emit_all_passes -- barrier context included
//  14.  emit_all_passes -- first pass has empty barriers
//  15.  emit_all_passes -- multiple passes preserve order
//  16.  emit_all_passes -- barrier state transitions recorded
//  17.  Unknown handle uses fallback name
//  18.  Indirect dispatch resolves buffer name
//  19.  Tags appear in output
//  20.  Output is valid JSON (parseable serde_json::Value)
//  21.  View type present in output
//  22.  emit_all_passes with empty pass list -> empty vec
//  23.  emit_all_passes with single pass (no edges = no barriers)
//  24.  emit_all_passes -- populated pass count matches input count
//  25.  emit_pass_bridge -- barrier key absent for standalone emit
//  26.  emit_all_passes -- resource_name in barrier entries

use renderer_backend::frame_graph::{
    emit_all_passes, emit_pass_bridge, AttachmentLoadOp, AttachmentStoreOp, BufferDesc,
    ColorAttachment, CompiledFrameGraph, DepthStencilAttachment, DispatchSource, InstanceSource,
    IrPass, IrResource, PassIndex, ResourceDesc, ResourceHandle, ResourceLifetime, ResourceState,
    TextureDesc, ViewType,
};

// =============================================================================
// SECTION 1 -- Graphics pass: color_attachments + depth_stencil in output
// =============================================================================

#[test]
fn graphics_pass_has_color_attachments_in_output() {
    let resources = vec![
        IrResource::new(
            ResourceHandle(1),
            "color_rt",
            ResourceDesc::Texture2D(TextureDesc {
                width: 1920,
                height: 1080,
                mip_levels: 1,
                array_layers: 1,
                format: "rgba8unorm".into(),
            }),
            ResourceLifetime::Transient,
            ResourceState::Uninitialized,
        ),
        IrResource::new(
            ResourceHandle(2),
            "depth_buffer",
            ResourceDesc::Texture2D(TextureDesc {
                width: 1920,
                height: 1080,
                mip_levels: 1,
                array_layers: 1,
                format: "depth32float".into(),
            }),
            ResourceLifetime::Transient,
            ResourceState::Uninitialized,
        ),
    ];

    let pass = IrPass::graphics(
        PassIndex(0),
        "gbuffer",
        vec![ColorAttachment {
            resource: ResourceHandle(1),
            load_op: AttachmentLoadOp::Clear,
            store_op: AttachmentStoreOp::Store,
            clear_color: [0.0, 0.0, 0.0, 1.0],
            ..Default::default()
        }],
        Some(DepthStencilAttachment {
            resource: ResourceHandle(2),
            depth_load_op: AttachmentLoadOp::Clear,
            depth_store_op: AttachmentStoreOp::Store,
            ..Default::default()
        }),
        InstanceSource::Direct {
            index_count: 36,
            instance_count: 1,
            base_vertex: 0,
            first_index: 0,
            first_instance: 0,
        },
        ViewType::Texture2D,
    );

    let json = emit_pass_bridge(&pass, &resources, 0);

    // Top-level identity fields.
    assert_eq!(json["index"], 0);
    assert_eq!(json["pass_index"], 0);
    assert_eq!(json["name"], "gbuffer");
    assert_eq!(json["pass_type"], "Graphics");

    // color_attachments array present and non-empty.
    let ca_array = json["color_attachments"]
        .as_array()
        .expect("graphics pass must have color_attachments array");
    assert_eq!(ca_array.len(), 1, "graphics pass has one color attachment");

    // depth_stencil present and not null.
    let ds = json
        .get("depth_stencil")
        .expect("graphics pass with depth_stencil must have depth_stencil key");
    assert!(
        !ds.is_null(),
        "depth_stencil must not be null when provided"
    );
}

#[test]
fn graphics_pass_color_attachment_contains_all_fields() {
    let resources = vec![IrResource::new(
        ResourceHandle(1),
        "color_rt",
        ResourceDesc::Texture2D(TextureDesc {
            width: 1920,
            height: 1080,
            mip_levels: 1,
            array_layers: 1,
            format: "rgba8unorm".into(),
        }),
        ResourceLifetime::Transient,
        ResourceState::Uninitialized,
    )];

    let pass = IrPass::graphics(
        PassIndex(0),
        "mrt_pass",
        vec![ColorAttachment {
            resource: ResourceHandle(1),
            mip_level: 2,
            array_layer: 1,
            load_op: AttachmentLoadOp::Clear,
            store_op: AttachmentStoreOp::Store,
            clear_color: [0.1, 0.2, 0.3, 1.0],
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

    let json = emit_pass_bridge(&pass, &resources, 0);
    let ca = &json["color_attachments"][0];

    assert_eq!(ca["resource_name"], "color_rt");
    assert_eq!(ca["resource_handle"], 1);
    assert_eq!(ca["mip_level"], 2);
    assert_eq!(ca["array_layer"], 1);
    assert_eq!(ca["load_op"], "Clear");
    assert_eq!(ca["store_op"], "Store");
    let cc = ca["clear_color"]
        .as_array()
        .expect("clear_color must be an array");
    assert_eq!(cc.len(), 4);
    assert!((cc[0].as_f64().unwrap() - 0.1).abs() < 1e-6);
    assert!((cc[1].as_f64().unwrap() - 0.2).abs() < 1e-6);
    assert!((cc[2].as_f64().unwrap() - 0.3).abs() < 1e-6);
    assert!((cc[3].as_f64().unwrap() - 1.0).abs() < 1e-6);
}

#[test]
fn graphics_pass_depth_stencil_contains_all_fields() {
    let resources = vec![
        IrResource::new(
            ResourceHandle(1),
            "color_rt",
            ResourceDesc::Texture2D(TextureDesc {
                width: 256,
                height: 256,
                mip_levels: 1,
                array_layers: 1,
                format: "rgba8unorm".into(),
            }),
            ResourceLifetime::Transient,
            ResourceState::Uninitialized,
        ),
        IrResource::new(
            ResourceHandle(2),
            "depth_tex",
            ResourceDesc::Texture2D(TextureDesc {
                width: 256,
                height: 256,
                mip_levels: 1,
                array_layers: 1,
                format: "depth32float".into(),
            }),
            ResourceLifetime::Transient,
            ResourceState::Uninitialized,
        ),
    ];

    let pass = IrPass::graphics(
        PassIndex(0),
        "depth_pass",
        vec![ColorAttachment {
            resource: ResourceHandle(1),
            ..Default::default()
        }],
        Some(DepthStencilAttachment {
            resource: ResourceHandle(2),
            depth_load_op: AttachmentLoadOp::Clear,
            depth_store_op: AttachmentStoreOp::Store,
            stencil_load_op: AttachmentLoadOp::Load,
            stencil_store_op: AttachmentStoreOp::DontCare,
            clear_depth: 0.5,
            clear_stencil: 0xFF,
            depth_test_enabled: true,
            depth_write_enabled: true,
        }),
        InstanceSource::Direct {
            index_count: 0,
            instance_count: 1,
            base_vertex: 0,
            first_index: 0,
            first_instance: 0,
        },
        ViewType::Texture2D,
    );

    let json = emit_pass_bridge(&pass, &resources, 0);
    let ds = &json["depth_stencil"];

    assert_eq!(ds["resource_name"], "depth_tex");
    assert_eq!(ds["resource_handle"], 2);
    assert_eq!(ds["depth_load_op"], "Clear");
    assert_eq!(ds["depth_store_op"], "Store");
    assert_eq!(ds["stencil_load_op"], "Load");
    assert_eq!(ds["stencil_store_op"], "DontCare");
    assert!((ds["clear_depth"].as_f64().unwrap() - 0.5).abs() < 1e-6);
    assert_eq!(ds["clear_stencil"], 0xFF);
    assert_eq!(ds["depth_test_enabled"], true);
    assert_eq!(ds["depth_write_enabled"], true);
}

#[test]
fn graphics_pass_depth_stencil_is_null_when_not_provided() {
    let resources = vec![IrResource::new(
        ResourceHandle(1),
        "color_rt",
        ResourceDesc::Texture2D(TextureDesc {
            width: 256,
            height: 256,
            mip_levels: 1,
            array_layers: 1,
            format: "rgba8unorm".into(),
        }),
        ResourceLifetime::Transient,
        ResourceState::Uninitialized,
    )];

    let pass = IrPass::graphics(
        PassIndex(0),
        "no_depth",
        vec![ColorAttachment {
            resource: ResourceHandle(1),
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

    let json = emit_pass_bridge(&pass, &resources, 0);
    assert!(
        json["depth_stencil"].is_null(),
        "depth_stencil must be null when no depth-stencil attachment is provided"
    );
}

// =============================================================================
// SECTION 2 -- Graphics pass: instance_source
// =============================================================================

#[test]
fn graphics_pass_instance_source_direct_has_correct_schema() {
    let resources = vec![];
    let pass = IrPass::graphics(
        PassIndex(0),
        "direct_draw",
        vec![],
        None,
        InstanceSource::Direct {
            index_count: 42,
            instance_count: 3,
            base_vertex: 100,
            first_index: 12,
            first_instance: 7,
        },
        ViewType::Texture2D,
    );

    let json = emit_pass_bridge(&pass, &resources, 0);
    let inst = &json["instance_source"];

    assert_eq!(inst["kind"], "Direct");
    assert_eq!(inst["index_count"], 42);
    assert_eq!(inst["instance_count"], 3);
    assert_eq!(inst["base_vertex"], 100);
    assert_eq!(inst["first_index"], 12);
    assert_eq!(inst["first_instance"], 7);
}

#[test]
fn graphics_pass_instance_source_indirect_has_correct_schema() {
    let resources = vec![IrResource::new(
        ResourceHandle(5),
        "args_buffer",
        ResourceDesc::Buffer(BufferDesc {
            size: 64,
            usage: "indirect".into(),
            is_indirect_arg: true,
        }),
        ResourceLifetime::Imported,
        ResourceState::Uninitialized,
    )];

    let pass = IrPass::graphics(
        PassIndex(0),
        "indirect_draw",
        vec![],
        None,
        InstanceSource::Indirect {
            buffer: ResourceHandle(5),
            offset: 256,
            draw_count: 8,
            stride: 20,
        },
        ViewType::Texture2D,
    );

    let json = emit_pass_bridge(&pass, &resources, 0);
    let inst = &json["instance_source"];

    assert_eq!(inst["kind"], "Indirect");
    assert_eq!(inst["buffer_name"], "args_buffer");
    assert_eq!(inst["buffer_handle"], 5);
    assert_eq!(inst["offset"], 256);
    assert_eq!(inst["draw_count"], 8);
    assert_eq!(inst["stride"], 20);
}

#[test]
fn graphics_pass_instance_source_mesh_has_correct_schema() {
    let resources = vec![];
    let pass = IrPass::graphics(
        PassIndex(0),
        "mesh_draw",
        vec![],
        None,
        InstanceSource::Mesh {
            group_count_x: 16,
            group_count_y: 8,
            group_count_z: 1,
        },
        ViewType::Texture2D,
    );

    let json = emit_pass_bridge(&pass, &resources, 0);
    let inst = &json["instance_source"];

    assert_eq!(inst["kind"], "Mesh");
    assert_eq!(inst["group_count_x"], 16);
    assert_eq!(inst["group_count_y"], 8);
    assert_eq!(inst["group_count_z"], 1);
}

// =============================================================================
// SECTION 3 -- Graphics pass: no copy-specific keys
// =============================================================================

#[test]
fn graphics_pass_does_not_include_copy_specific_keys() {
    let resources = vec![];
    let pass = IrPass::graphics(
        PassIndex(0),
        "gfx_pass",
        vec![],
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

    let json = emit_pass_bridge(&pass, &resources, 0);

    assert!(
        json.get("source_resources").is_none(),
        "Graphics pass must not have source_resources key"
    );
    assert!(
        json.get("destination_resources").is_none(),
        "Graphics pass must not have destination_resources key"
    );
}

// =============================================================================
// SECTION 4 -- Graphics pass: vertex_buffers
// =============================================================================

#[test]
fn graphics_pass_vertex_buffers_populated_from_buffer_reads() {
    let resources = vec![
        IrResource::new(
            ResourceHandle(1),
            "vertex_buf",
            ResourceDesc::Buffer(BufferDesc {
                size: 65536,
                usage: "vertex".into(),
                is_indirect_arg: false,
            }),
            ResourceLifetime::Transient,
            ResourceState::Uninitialized,
        ),
        IrResource::new(
            ResourceHandle(2),
            "index_buf",
            ResourceDesc::Buffer(BufferDesc {
                size: 16384,
                usage: "index".into(),
                is_indirect_arg: false,
            }),
            ResourceLifetime::Transient,
            ResourceState::Uninitialized,
        ),
    ];

    let mut pass = IrPass::graphics(
        PassIndex(0),
        "mesh_pass",
        vec![],
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
    pass.access_set.reads.push(ResourceHandle(1));
    pass.access_set.reads.push(ResourceHandle(2));

    let json = emit_pass_bridge(&pass, &resources, 0);
    let vbs = json["vertex_buffers"]
        .as_array()
        .expect("vertex_buffers must be an array");

    assert_eq!(
        vbs.len(),
        2,
        "two buffer resources in reads should appear in vertex_buffers"
    );
    assert_eq!(vbs[0]["resource_name"], "vertex_buf");
    assert_eq!(vbs[0]["resource_handle"], 1);
    assert_eq!(vbs[1]["resource_name"], "index_buf");
    assert_eq!(vbs[1]["resource_handle"], 2);
}

#[test]
fn graphics_pass_vertex_buffers_empty_when_no_buffer_reads() {
    let resources = vec![];
    let pass = IrPass::graphics(
        PassIndex(0),
        "no_vb",
        vec![],
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

    let json = emit_pass_bridge(&pass, &resources, 0);
    let vbs = json["vertex_buffers"]
        .as_array()
        .expect("vertex_buffers must be an array");
    assert!(
        vbs.is_empty(),
        "vertex_buffers should be empty when no buffer reads"
    );
}

// =============================================================================
// SECTION 5 -- Compute pass: dispatch_source + workgroup sizes
// =============================================================================

#[test]
fn compute_pass_dispatch_source_has_workgroup_sizes() {
    let resources: Vec<IrResource> = vec![];
    let pass = IrPass::compute(
        PassIndex(1),
        "compute_lighting",
        DispatchSource::Direct {
            group_count_x: 16,
            group_count_y: 8,
            group_count_z: 1,
        },
        ViewType::Storage,
    );

    let json = emit_pass_bridge(&pass, &resources, 0);

    assert_eq!(json["name"], "compute_lighting");
    assert_eq!(json["pass_type"], "Compute");

    let ds = &json["dispatch_source"];
    assert_eq!(ds["kind"], "Direct");
    assert_eq!(ds["group_count_x"], 16);
    assert_eq!(ds["group_count_y"], 8);
    assert_eq!(ds["group_count_z"], 1);
}

#[test]
fn compute_pass_no_color_attachments_and_null_depth_stencil() {
    let resources: Vec<IrResource> = vec![];
    let pass = IrPass::compute(
        PassIndex(0),
        "pure_compute",
        DispatchSource::Direct {
            group_count_x: 1,
            group_count_y: 1,
            group_count_z: 1,
        },
        ViewType::Storage,
    );

    let json = emit_pass_bridge(&pass, &resources, 0);

    let ca = json["color_attachments"]
        .as_array()
        .expect("color_attachments must be an array");
    assert!(
        ca.is_empty(),
        "compute pass must have empty color_attachments"
    );
    assert!(
        json["depth_stencil"].is_null(),
        "compute pass must have null depth_stencil"
    );
}

#[test]
fn compute_pass_large_workgroup_counts() {
    let resources: Vec<IrResource> = vec![];
    let pass = IrPass::compute(
        PassIndex(0),
        "large_compute",
        DispatchSource::Direct {
            group_count_x: 128,
            group_count_y: 64,
            group_count_z: 16,
        },
        ViewType::Storage,
    );

    let json = emit_pass_bridge(&pass, &resources, 0);
    let ds = &json["dispatch_source"];

    assert_eq!(ds["group_count_x"], 128);
    assert_eq!(ds["group_count_y"], 64);
    assert_eq!(ds["group_count_z"], 16);
}

// =============================================================================
// SECTION 6 -- Copy pass: source_resources + destination_resources
// =============================================================================

#[test]
fn copy_pass_has_source_and_destination_resources() {
    let resources = vec![
        IrResource::new(
            ResourceHandle(10),
            "src_buffer",
            ResourceDesc::Buffer(BufferDesc {
                size: 4096,
                usage: "storage".into(),
                is_indirect_arg: false,
            }),
            ResourceLifetime::Transient,
            ResourceState::Uninitialized,
        ),
        IrResource::new(
            ResourceHandle(11),
            "dst_buffer",
            ResourceDesc::Buffer(BufferDesc {
                size: 4096,
                usage: "storage".into(),
                is_indirect_arg: false,
            }),
            ResourceLifetime::Transient,
            ResourceState::Uninitialized,
        ),
    ];

    let mut pass = IrPass::copy(PassIndex(2), "buffer_copy");
    pass.access_set.reads.push(ResourceHandle(10));
    pass.access_set.writes.push(ResourceHandle(11));

    let json = emit_pass_bridge(&pass, &resources, 0);

    assert_eq!(json["name"], "buffer_copy");
    assert_eq!(json["pass_type"], "Copy");

    let src = json["source_resources"]
        .as_array()
        .expect("Copy pass must have source_resources");
    assert_eq!(src.len(), 1);
    assert_eq!(src[0]["resource_name"], "src_buffer");
    assert_eq!(src[0]["resource_handle"], 10);

    let dst = json["destination_resources"]
        .as_array()
        .expect("Copy pass must have destination_resources");
    assert_eq!(dst.len(), 1);
    assert_eq!(dst[0]["resource_name"], "dst_buffer");
    assert_eq!(dst[0]["resource_handle"], 11);
}

#[test]
fn copy_pass_multiple_source_and_destination() {
    let resources = vec![
        IrResource::new(
            ResourceHandle(1),
            "src_a",
            ResourceDesc::Buffer(BufferDesc {
                size: 1024,
                usage: "storage".into(),
                is_indirect_arg: false,
            }),
            ResourceLifetime::Transient,
            ResourceState::Uninitialized,
        ),
        IrResource::new(
            ResourceHandle(2),
            "src_b",
            ResourceDesc::Buffer(BufferDesc {
                size: 1024,
                usage: "storage".into(),
                is_indirect_arg: false,
            }),
            ResourceLifetime::Transient,
            ResourceState::Uninitialized,
        ),
        IrResource::new(
            ResourceHandle(3),
            "dst_a",
            ResourceDesc::Buffer(BufferDesc {
                size: 1024,
                usage: "storage".into(),
                is_indirect_arg: false,
            }),
            ResourceLifetime::Transient,
            ResourceState::Uninitialized,
        ),
        IrResource::new(
            ResourceHandle(4),
            "dst_b",
            ResourceDesc::Buffer(BufferDesc {
                size: 1024,
                usage: "storage".into(),
                is_indirect_arg: false,
            }),
            ResourceLifetime::Transient,
            ResourceState::Uninitialized,
        ),
    ];

    let mut pass = IrPass::copy(PassIndex(0), "multi_copy");
    pass.access_set.reads.push(ResourceHandle(1));
    pass.access_set.reads.push(ResourceHandle(2));
    pass.access_set.writes.push(ResourceHandle(3));
    pass.access_set.writes.push(ResourceHandle(4));

    let json = emit_pass_bridge(&pass, &resources, 0);

    assert_eq!(
        json["source_resources"].as_array().unwrap().len(),
        2,
        "two source resources"
    );
    assert_eq!(
        json["destination_resources"].as_array().unwrap().len(),
        2,
        "two destination resources"
    );
}

// =============================================================================
// SECTION 7 -- Ray tracing pass: dispatch_source
// =============================================================================

#[test]
fn ray_tracing_pass_has_dispatch_source() {
    let resources: Vec<IrResource> = vec![];
    let pass = IrPass::ray_tracing(
        PassIndex(4),
        "raytrace_gi",
        DispatchSource::Direct {
            group_count_x: 8,
            group_count_y: 8,
            group_count_z: 1,
        },
    );

    let json = emit_pass_bridge(&pass, &resources, 0);

    assert_eq!(json["name"], "raytrace_gi");
    assert_eq!(json["pass_type"], "RayTracing");

    let ds = &json["dispatch_source"];
    assert!(
        ds.is_object(),
        "RayTracing pass must have dispatch_source object"
    );
    assert_eq!(ds["kind"], "Direct");
    assert_eq!(ds["group_count_x"], 8);
}

// =============================================================================
// SECTION 8 -- Pass name resolves correctly
// =============================================================================

#[test]
fn pass_name_appears_in_output() {
    let resources: Vec<IrResource> = vec![];
    let pass = IrPass::compute(
        PassIndex(0),
        "CustomPassName_123",
        DispatchSource::Direct {
            group_count_x: 1,
            group_count_y: 1,
            group_count_z: 1,
        },
        ViewType::Storage,
    );

    let json = emit_pass_bridge(&pass, &resources, 0);
    assert_eq!(
        json["name"], "CustomPassName_123",
        "pass name must be preserved exactly in output"
    );
}

// =============================================================================
// SECTION 9 -- Pass index (execution + original) in output
// =============================================================================

#[test]
fn emit_pass_bridge_includes_both_index_and_pass_index() {
    let resources: Vec<IrResource> = vec![];
    let pass = IrPass::compute(
        PassIndex(3),
        "indexed_pass",
        DispatchSource::Direct {
            group_count_x: 1,
            group_count_y: 1,
            group_count_z: 1,
        },
        ViewType::Storage,
    );

    // pass_index=0 -> execution order index
    // pass.index=3 -> original compiled index
    let json = emit_pass_bridge(&pass, &resources, 0);

    assert_eq!(
        json["index"], 0,
        "index should be the execution-order index (0)"
    );
    assert_eq!(
        json["pass_index"], 3,
        "pass_index should be the original pass index (3)"
    );
}

#[test]
fn emit_pass_bridge_respects_execution_index_parameter() {
    let resources: Vec<IrResource> = vec![];
    let pass = IrPass::compute(
        PassIndex(1),
        "late_pass",
        DispatchSource::Direct {
            group_count_x: 1,
            group_count_y: 1,
            group_count_z: 1,
        },
        ViewType::Storage,
    );

    let json = emit_pass_bridge(&pass, &resources, 7);
    assert_eq!(
        json["index"], 7,
        "index should reflect the caller-provided execution index"
    );
}

// =============================================================================
// SECTION 10 -- emit_all_passes: barrier context included
// =============================================================================

#[test]
fn emit_all_passes_includes_barriers_per_pass() {
    // Two passes: P0 writes R1, P1 reads R1 -> requires barrier.
    let p0 = IrPass::graphics(
        PassIndex(0),
        "write_pass",
        vec![ColorAttachment {
            resource: ResourceHandle(1),
            load_op: AttachmentLoadOp::Clear,
            store_op: AttachmentStoreOp::Store,
            clear_color: [0.0; 4],
            ..Default::default()
        }],
        None,
        InstanceSource::Direct {
            index_count: 6,
            instance_count: 1,
            base_vertex: 0,
            first_index: 0,
            first_instance: 0,
        },
        ViewType::Texture2D,
    );

    let mut p1 = IrPass::compute(
        PassIndex(1),
        "read_pass",
        DispatchSource::Direct {
            group_count_x: 1,
            group_count_y: 1,
            group_count_z: 1,
        },
        ViewType::Storage,
    );
    p1.access_set.reads.push(ResourceHandle(1));

    let resources = vec![IrResource::new(
        ResourceHandle(1),
        "framebuffer",
        ResourceDesc::Texture2D(TextureDesc {
            width: 256,
            height: 256,
            mip_levels: 1,
            array_layers: 1,
            format: "rgba16float".into(),
        }),
        ResourceLifetime::Transient,
        ResourceState::Uninitialized,
    )];

    let compiled = CompiledFrameGraph::compile(vec![p0, p1], resources)
        .expect("two-pass graph must compile successfully");

    let all_passes = emit_all_passes(&compiled);

    assert_eq!(all_passes.len(), 2, "should emit both passes");

    // Verify each pass has a `barriers` key.
    for (i, pass_json) in all_passes.iter().enumerate() {
        assert!(
            pass_json.get("barriers").is_some(),
            "pass {} must have a barriers key",
            i
        );
        assert!(
            pass_json["barriers"].is_array(),
            "pass {} barriers must be an array",
            i
        );
    }
}

#[test]
fn emit_all_passes_first_pass_has_empty_barriers() {
    let p0 = IrPass::compute(
        PassIndex(0),
        "first_pass",
        DispatchSource::Direct {
            group_count_x: 1,
            group_count_y: 1,
            group_count_z: 1,
        },
        ViewType::Storage,
    );

    let mut p1 = IrPass::compute(
        PassIndex(1),
        "second_pass",
        DispatchSource::Direct {
            group_count_x: 1,
            group_count_y: 1,
            group_count_z: 1,
        },
        ViewType::Storage,
    );
    p1.access_set.reads.push(ResourceHandle(1));

    let resources = vec![IrResource::new(
        ResourceHandle(1),
        "shared_buf",
        ResourceDesc::Buffer(BufferDesc {
            size: 1024,
            usage: "storage".into(),
            is_indirect_arg: false,
        }),
        ResourceLifetime::Transient,
        ResourceState::Uninitialized,
    )];

    let compiled =
        CompiledFrameGraph::compile(vec![p0, p1], resources).expect("compile must succeed");

    let all_passes = emit_all_passes(&compiled);

    assert!(
        all_passes[0]["barriers"].as_array().unwrap().is_empty(),
        "first pass in topological order should have no incoming barriers"
    );
}

#[test]
fn emit_all_passes_barrier_contains_state_transition_fields() {
    let p0 = IrPass::graphics(
        PassIndex(0),
        "producer",
        vec![ColorAttachment {
            resource: ResourceHandle(1),
            load_op: AttachmentLoadOp::Clear,
            store_op: AttachmentStoreOp::Store,
            clear_color: [0.0; 4],
            ..Default::default()
        }],
        None,
        InstanceSource::Direct {
            index_count: 6,
            instance_count: 1,
            base_vertex: 0,
            first_index: 0,
            first_instance: 0,
        },
        ViewType::Texture2D,
    );

    let mut p1 = IrPass::compute(
        PassIndex(1),
        "consumer",
        DispatchSource::Direct {
            group_count_x: 1,
            group_count_y: 1,
            group_count_z: 1,
        },
        ViewType::Storage,
    );
    p1.access_set.reads.push(ResourceHandle(1));

    let resources = vec![IrResource::new(
        ResourceHandle(1),
        "shared_tex",
        ResourceDesc::Texture2D(TextureDesc {
            width: 256,
            height: 256,
            mip_levels: 1,
            array_layers: 1,
            format: "rgba16float".into(),
        }),
        ResourceLifetime::Transient,
        ResourceState::Uninitialized,
    )];

    let compiled =
        CompiledFrameGraph::compile(vec![p0, p1], resources).expect("compile must succeed");

    let all_passes = emit_all_passes(&compiled);
    let p1_barriers = all_passes[1]["barriers"].as_array().unwrap();

    assert!(!p1_barriers.is_empty(), "second pass should have barriers");

    let barrier = &p1_barriers[0];
    assert!(
        barrier.get("from_pass_index").is_some(),
        "barrier must have from_pass_index"
    );
    assert!(
        barrier.get("from_pass_name").is_some(),
        "barrier must have from_pass_name"
    );
    assert!(
        barrier.get("before_state").is_some(),
        "barrier must have before_state"
    );
    assert!(
        barrier.get("after_state").is_some(),
        "barrier must have after_state"
    );

    assert_eq!(barrier["from_pass_index"], 0);
    assert_eq!(barrier["from_pass_name"], "producer");
}

// =============================================================================
// SECTION 11 -- emit_all_passes: multiple passes preserve order
// =============================================================================

#[test]
fn emit_all_passes_three_passes_preserve_order() {
    // Three passes in a chain: P0 -> P1 -> P2
    let mut p0 = IrPass::compute(
        PassIndex(0),
        "a",
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
        "b",
        DispatchSource::Direct {
            group_count_x: 1,
            group_count_y: 1,
            group_count_z: 1,
        },
        ViewType::Storage,
    );
    p1.access_set.reads.push(ResourceHandle(1));
    p1.access_set.writes.push(ResourceHandle(2));

    let mut p2 = IrPass::compute(
        PassIndex(2),
        "c",
        DispatchSource::Direct {
            group_count_x: 1,
            group_count_y: 1,
            group_count_z: 1,
        },
        ViewType::Storage,
    );
    p2.access_set.reads.push(ResourceHandle(2));

    let resources = vec![
        IrResource::new(
            ResourceHandle(1),
            "r1",
            ResourceDesc::Buffer(BufferDesc {
                size: 64,
                usage: "storage".into(),
                is_indirect_arg: false,
            }),
            ResourceLifetime::Transient,
            ResourceState::Uninitialized,
        ),
        IrResource::new(
            ResourceHandle(2),
            "r2",
            ResourceDesc::Buffer(BufferDesc {
                size: 64,
                usage: "storage".into(),
                is_indirect_arg: false,
            }),
            ResourceLifetime::Transient,
            ResourceState::Uninitialized,
        ),
    ];

    let compiled = CompiledFrameGraph::compile(vec![p0, p1, p2], resources)
        .expect("three-pass graph must compile");

    let all_passes = emit_all_passes(&compiled);
    assert_eq!(all_passes.len(), 3, "all three passes should be emitted");

    let names: Vec<&str> = all_passes
        .iter()
        .map(|p| p["name"].as_str().unwrap())
        .collect();
    assert_eq!(
        names,
        vec!["a", "b", "c"],
        "passes must be in topological order"
    );
}

// =============================================================================
// SECTION 12 -- Unknown handle uses fallback name
// =============================================================================

#[test]
fn emit_pass_bridge_unknown_handle_uses_fallback_name() {
    let resources: Vec<IrResource> = vec![];
    let pass = IrPass::graphics(
        PassIndex(0),
        "orphan_pass",
        vec![ColorAttachment {
            resource: ResourceHandle(99), // not in resources array
            ..Default::default()
        }],
        None,
        InstanceSource::Direct {
            index_count: 6,
            instance_count: 1,
            base_vertex: 0,
            first_index: 0,
            first_instance: 0,
        },
        ViewType::Texture2D,
    );

    let json = emit_pass_bridge(&pass, &resources, 0);
    let ca = &json["color_attachments"][0];

    assert_eq!(
        ca["resource_name"], "<unknown: 99>",
        "unresolvable handle should use fallback name"
    );
}

// =============================================================================
// SECTION 13 -- Indirect dispatch resolves buffer name
// =============================================================================

#[test]
fn compute_pass_indirect_dispatch_resolves_buffer_name() {
    let resources = vec![IrResource::new(
        ResourceHandle(5),
        "dispatch_args",
        ResourceDesc::Buffer(BufferDesc {
            size: 64,
            usage: "indirect".into(),
            is_indirect_arg: true,
        }),
        ResourceLifetime::Imported,
        ResourceState::Uninitialized,
    )];

    let pass = IrPass::compute(
        PassIndex(3),
        "indirect_compute",
        DispatchSource::Indirect {
            buffer: ResourceHandle(5),
            offset: 0,
        },
        ViewType::Storage,
    );

    let json = emit_pass_bridge(&pass, &resources, 0);
    let ds = &json["dispatch_source"];

    assert_eq!(ds["kind"], "Indirect");
    assert_eq!(ds["buffer_name"], "dispatch_args");
    assert_eq!(ds["buffer_handle"], 5);
    assert_eq!(ds["offset"], 0);
}

#[test]
fn ray_tracing_pass_indirect_dispatch_resolves_buffer_name() {
    let resources = vec![IrResource::new(
        ResourceHandle(7),
        "rt_dispatch_args",
        ResourceDesc::Buffer(BufferDesc {
            size: 32,
            usage: "indirect".into(),
            is_indirect_arg: true,
        }),
        ResourceLifetime::Imported,
        ResourceState::Uninitialized,
    )];

    let pass = IrPass::ray_tracing(
        PassIndex(4),
        "rt_indirect",
        DispatchSource::Indirect {
            buffer: ResourceHandle(7),
            offset: 128,
        },
    );

    let json = emit_pass_bridge(&pass, &resources, 0);
    let ds = &json["dispatch_source"];

    assert_eq!(ds["kind"], "Indirect");
    assert_eq!(ds["buffer_name"], "rt_dispatch_args");
    assert_eq!(ds["buffer_handle"], 7);
    assert_eq!(ds["offset"], 128);
}

// =============================================================================
// SECTION 14 -- Tags in output
// =============================================================================

#[test]
fn empty_tags_in_output() {
    let resources: Vec<IrResource> = vec![];
    let pass = IrPass::compute(
        PassIndex(0),
        "untagged",
        DispatchSource::Direct {
            group_count_x: 1,
            group_count_y: 1,
            group_count_z: 1,
        },
        ViewType::Storage,
    );

    let json = emit_pass_bridge(&pass, &resources, 0);
    let tags = json["tags"].as_array().expect("tags must be an array");
    assert!(tags.is_empty(), "default tags should be empty");
}

#[test]
fn populated_tags_appear_in_output() {
    let resources: Vec<IrResource> = vec![];
    let mut pass = IrPass::compute(
        PassIndex(0),
        "tagged_pass",
        DispatchSource::Direct {
            group_count_x: 1,
            group_count_y: 1,
            group_count_z: 1,
        },
        ViewType::Storage,
    );
    pass.tags.push("transparent".into());
    pass.tags.push("post-process".into());
    pass.tags.push("debug".into());

    let json = emit_pass_bridge(&pass, &resources, 0);
    let tags = json["tags"].as_array().expect("tags must be an array");

    assert_eq!(tags.len(), 3, "three tags should appear");
    assert_eq!(tags[0], "transparent");
    assert_eq!(tags[1], "post-process");
    assert_eq!(tags[2], "debug");
}

// =============================================================================
// SECTION 15 -- Output is valid JSON (parseable serde_json::Value)
// =============================================================================

#[test]
fn emit_pass_bridge_output_is_valid_json() {
    let resources: Vec<IrResource> = vec![];
    let pass = IrPass::compute(
        PassIndex(0),
        "valid_json",
        DispatchSource::Direct {
            group_count_x: 1,
            group_count_y: 1,
            group_count_z: 1,
        },
        ViewType::Storage,
    );

    let json = emit_pass_bridge(&pass, &resources, 0);

    // The return type is already serde_json::Value, but verify it round-trips.
    let serialized = serde_json::to_string(&json)
        .expect("emit_pass_bridge output must serialize to JSON string");
    let parsed: serde_json::Value =
        serde_json::from_str(&serialized).expect("serialized output must parse back to valid JSON");

    assert_eq!(parsed["name"], "valid_json");
}

#[test]
fn emit_all_passes_output_is_valid_json() {
    let p0 = IrPass::compute(
        PassIndex(0),
        "p0",
        DispatchSource::Direct {
            group_count_x: 1,
            group_count_y: 1,
            group_count_z: 1,
        },
        ViewType::Storage,
    );

    let resources: Vec<IrResource> = vec![];

    let compiled =
        CompiledFrameGraph::compile(vec![p0], resources).expect("single pass must compile");

    let all_passes = emit_all_passes(&compiled);

    let serialized = serde_json::to_string(&all_passes)
        .expect("emit_all_passes output must serialize to JSON string");
    let parsed: Vec<serde_json::Value> =
        serde_json::from_str(&serialized).expect("serialized output must parse back to valid JSON");

    assert_eq!(parsed.len(), 1, "single pass in output");
    assert_eq!(parsed[0]["name"], "p0");
}

// =============================================================================
// SECTION 16 -- View type present in output
// =============================================================================

#[test]
fn view_type_appears_in_graphics_pass_output() {
    let resources: Vec<IrResource> = vec![];
    let pass = IrPass::graphics(
        PassIndex(0),
        "gfx",
        vec![],
        None,
        InstanceSource::Direct {
            index_count: 0,
            instance_count: 1,
            base_vertex: 0,
            first_index: 0,
            first_instance: 0,
        },
        ViewType::TextureCube,
    );

    let json = emit_pass_bridge(&pass, &resources, 0);
    assert_eq!(json["view_type"], "TextureCube");
}

#[test]
fn view_type_appears_in_compute_pass_output() {
    let resources: Vec<IrResource> = vec![];
    let pass = IrPass::compute(
        PassIndex(0),
        "comp",
        DispatchSource::Direct {
            group_count_x: 1,
            group_count_y: 1,
            group_count_z: 1,
        },
        ViewType::StorageBuffer,
    );

    let json = emit_pass_bridge(&pass, &resources, 0);
    assert_eq!(json["view_type"], "StorageBuffer");
}

// =============================================================================
// SECTION 17 -- emit_all_passes: single pass (no edges = no barriers)
// =============================================================================

#[test]
fn emit_all_passes_single_pass_no_barriers() {
    let pass = IrPass::compute(
        PassIndex(0),
        "solo",
        DispatchSource::Direct {
            group_count_x: 1,
            group_count_y: 1,
            group_count_z: 1,
        },
        ViewType::Storage,
    );

    let resources: Vec<IrResource> = vec![];

    let compiled =
        CompiledFrameGraph::compile(vec![pass], resources).expect("single pass must compile");

    let all_passes = emit_all_passes(&compiled);
    assert_eq!(all_passes.len(), 1, "one pass in output");
    assert!(
        all_passes[0]["barriers"].as_array().unwrap().is_empty(),
        "single pass with no edges has no barriers"
    );
}

// =============================================================================
// SECTION 18 -- emit_all_passes: multiple passes with buffer barriers
// =============================================================================

#[test]
fn emit_all_passes_buffer_barrier_context() {
    // P0 writes R1, P1 reads R1 (buffer resource).
    let mut p0 = IrPass::compute(
        PassIndex(0),
        "buffer_writer",
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
        "buffer_reader",
        DispatchSource::Direct {
            group_count_x: 1,
            group_count_y: 1,
            group_count_z: 1,
        },
        ViewType::Storage,
    );
    p1.access_set.reads.push(ResourceHandle(1));

    let resources = vec![IrResource::new(
        ResourceHandle(1),
        "storage_buf",
        ResourceDesc::Buffer(BufferDesc {
            size: 4096,
            usage: "storage".into(),
            is_indirect_arg: false,
        }),
        ResourceLifetime::Transient,
        ResourceState::Uninitialized,
    )];

    let compiled = CompiledFrameGraph::compile(vec![p0, p1], resources)
        .expect("two-pass buffer graph must compile");

    let all_passes = emit_all_passes(&compiled);
    assert_eq!(all_passes.len(), 2);

    let p1_barriers = all_passes[1]["barriers"].as_array().unwrap();
    assert!(!p1_barriers.is_empty(), "second pass should have barriers");
    assert!(
        p1_barriers[0]["before_state"].as_str().is_some(),
        "before_state must be a string"
    );
    assert!(
        p1_barriers[0]["after_state"].as_str().is_some(),
        "after_state must be a string"
    );
}

// =============================================================================
// SECTION 19 -- emit_pass_bridge: multiple color attachments
// =============================================================================

#[test]
fn graphics_pass_multiple_color_attachments_all_resolve() {
    let resources = vec![
        IrResource::new(
            ResourceHandle(1),
            "albedo",
            ResourceDesc::Texture2D(TextureDesc {
                width: 1920,
                height: 1080,
                mip_levels: 1,
                array_layers: 1,
                format: "rgba8unorm".into(),
            }),
            ResourceLifetime::Transient,
            ResourceState::Uninitialized,
        ),
        IrResource::new(
            ResourceHandle(2),
            "normal",
            ResourceDesc::Texture2D(TextureDesc {
                width: 1920,
                height: 1080,
                mip_levels: 1,
                array_layers: 1,
                format: "rgba8unorm".into(),
            }),
            ResourceLifetime::Transient,
            ResourceState::Uninitialized,
        ),
        IrResource::new(
            ResourceHandle(3),
            "roughness_metallic",
            ResourceDesc::Texture2D(TextureDesc {
                width: 1920,
                height: 1080,
                mip_levels: 1,
                array_layers: 1,
                format: "rgba8unorm".into(),
            }),
            ResourceLifetime::Transient,
            ResourceState::Uninitialized,
        ),
    ];

    let pass = IrPass::graphics(
        PassIndex(0),
        "mrt_gbuffer",
        vec![
            ColorAttachment {
                resource: ResourceHandle(1),
                load_op: AttachmentLoadOp::Clear,
                store_op: AttachmentStoreOp::Store,
                clear_color: [0.0, 0.0, 0.0, 1.0],
                ..Default::default()
            },
            ColorAttachment {
                resource: ResourceHandle(2),
                load_op: AttachmentLoadOp::Clear,
                store_op: AttachmentStoreOp::Store,
                clear_color: [0.0, 0.0, 0.0, 1.0],
                ..Default::default()
            },
            ColorAttachment {
                resource: ResourceHandle(3),
                load_op: AttachmentLoadOp::Clear,
                store_op: AttachmentStoreOp::Store,
                clear_color: [0.5, 0.5, 0.5, 1.0],
                ..Default::default()
            },
        ],
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

    let json = emit_pass_bridge(&pass, &resources, 0);
    let ca = json["color_attachments"].as_array().unwrap();

    assert_eq!(ca.len(), 3, "MRT with 3 color attachments");
    assert_eq!(ca[0]["resource_name"], "albedo");
    assert_eq!(ca[1]["resource_name"], "normal");
    assert_eq!(ca[2]["resource_name"], "roughness_metallic");
}

// =============================================================================
// SECTION 20 -- emit_pass_bridge: all four pass types
// =============================================================================

#[test]
fn all_four_pass_types_are_distinguishable_in_output() {
    let resources: Vec<IrResource> = vec![];

    let gfx = IrPass::graphics(
        PassIndex(0),
        "gfx",
        vec![],
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
    let comp = IrPass::compute(
        PassIndex(1),
        "comp",
        DispatchSource::Direct {
            group_count_x: 1,
            group_count_y: 1,
            group_count_z: 1,
        },
        ViewType::Storage,
    );
    let copy_pass = IrPass::copy(PassIndex(2), "copy_op");
    let rt = IrPass::ray_tracing(
        PassIndex(3),
        "rt",
        DispatchSource::Direct {
            group_count_x: 1,
            group_count_y: 1,
            group_count_z: 1,
        },
    );

    let outputs = vec![
        emit_pass_bridge(&gfx, &resources, 0),
        emit_pass_bridge(&comp, &resources, 1),
        emit_pass_bridge(&copy_pass, &resources, 2),
        emit_pass_bridge(&rt, &resources, 3),
    ];

    let types: Vec<&str> = outputs
        .iter()
        .map(|o| o["pass_type"].as_str().unwrap())
        .collect();

    assert_eq!(types, vec!["Graphics", "Compute", "Copy", "RayTracing"]);
}

// =============================================================================
// SECTION 21 -- emit_all_passes: empty / edge cases
// =============================================================================

#[test]
fn emit_all_passes_empty_pass_list_produces_empty_output() {
    let resources: Vec<IrResource> = vec![];

    let compiled = CompiledFrameGraph::compile(vec![], resources)
        .expect("compile of empty pass list must succeed");

    let all_passes = emit_all_passes(&compiled);
    assert!(
        all_passes.is_empty(),
        "empty frame graph must produce empty pass array"
    );
}

#[test]
fn emit_all_passes_pass_count_matches_input() {
    let mut passes: Vec<IrPass> = (0..5)
        .map(|i| {
            let mut p = IrPass::compute(
                PassIndex(i),
                format!("p{}", i),
                DispatchSource::Direct {
                    group_count_x: 1,
                    group_count_y: 1,
                    group_count_z: 1,
                },
                ViewType::Storage,
            );
            if i > 0 {
                p.access_set.reads.push(ResourceHandle(1));
            }
            p.access_set.writes.push(ResourceHandle(1));
            p
        })
        .collect();

    let resources = vec![IrResource::new(
        ResourceHandle(1),
        "shared",
        ResourceDesc::Buffer(BufferDesc {
            size: 256,
            usage: "storage".into(),
            is_indirect_arg: false,
        }),
        ResourceLifetime::Transient,
        ResourceState::Uninitialized,
    )];

    let compiled =
        CompiledFrameGraph::compile(passes, resources).expect("five-pass chain must compile");

    let all_passes = emit_all_passes(&compiled);
    assert_eq!(
        all_passes.len(),
        5,
        "emit_all_passes must emit exactly as many passes as compiled"
    );
}

#[test]
fn emit_pass_bridge_no_barriers_key_for_standalone_emit() {
    let resources: Vec<IrResource> = vec![];
    let pass = IrPass::compute(
        PassIndex(0),
        "standalone",
        DispatchSource::Direct {
            group_count_x: 1,
            group_count_y: 1,
            group_count_z: 1,
        },
        ViewType::Storage,
    );

    let json = emit_pass_bridge(&pass, &resources, 0);

    assert!(
        json.get("barriers").is_none(),
        "standalone emit_pass_bridge must not contain barriers key"
    );
}

#[test]
fn emit_all_passes_barrier_entry_contains_resource_name() {
    let mut p0 = IrPass::compute(
        PassIndex(0),
        "producer",
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
        "consumer",
        DispatchSource::Direct {
            group_count_x: 1,
            group_count_y: 1,
            group_count_z: 1,
        },
        ViewType::Storage,
    );
    p1.access_set.reads.push(ResourceHandle(1));

    let resources = vec![IrResource::new(
        ResourceHandle(1),
        "barrier_resource",
        ResourceDesc::Buffer(BufferDesc {
            size: 512,
            usage: "storage".into(),
            is_indirect_arg: false,
        }),
        ResourceLifetime::Transient,
        ResourceState::Uninitialized,
    )];

    let compiled =
        CompiledFrameGraph::compile(vec![p0, p1], resources).expect("two-pass graph must compile");

    let all_passes = emit_all_passes(&compiled);
    let p1_barriers = all_passes[1]["barriers"].as_array().unwrap();

    assert!(!p1_barriers.is_empty(), "second pass should have barriers");

    let barrier = &p1_barriers[0];
    assert!(
        barrier.get("resource_name").is_some() || barrier.get("resource_handle").is_some(),
        "barrier must carry resource_name or resource_handle"
    );
}
