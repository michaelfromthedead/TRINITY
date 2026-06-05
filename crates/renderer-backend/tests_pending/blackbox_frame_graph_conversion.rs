// Black-box integration tests for PyPassNode -> IrPass conversion.
//
// These tests validate TryFrom<PyPassNode> for IrPass using ONLY the public
// API surface of renderer_backend::frame_graph::python and
// renderer_backend::frame_graph. No internal fields are accessed.
//
// Cleanroom: written against the spec only, without reading the conversion
// implementation.

use renderer_backend::frame_graph::python::{
    self, ConversionError, PyColorAttachment, PyDepthStencilAttachment,
    PyDispatchSource, PyInstanceSource, PyPassNode, PyPassType, PyViewType,
};
use renderer_backend::frame_graph::{
    AttachmentLoadOp, AttachmentStoreOp, DispatchSource, InstanceSource,
    IrPass, PassIndex, PassType, ResourceHandle, ViewType,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Create a simple graphics pass node with one color attachment and one
/// depth-stencil attachment. Uses `minimal_py_pass_node` which is doc-hidden
/// but available in integration tests.
fn graphics_node(name: &str) -> PyPassNode {
    python::minimal_py_pass_node(name, PyPassType::Graphics)
}

fn compute_node(name: &str) -> PyPassNode {
    python::minimal_py_pass_node(name, PyPassType::Compute)
}

fn copy_node(name: &str) -> PyPassNode {
    python::minimal_py_pass_node(name, PyPassType::Copy)
}

fn rt_node(name: &str) -> PyPassNode {
    python::minimal_py_pass_node(name, PyPassType::RayTracing)
}

// ===========================================================================
// Section 1: ConversionError Display (12 variants)
// ===========================================================================

#[test]
fn error_display_empty_pass_name() {
    let err = ConversionError::EmptyPassName;
    let s = format!("{}", err);
    assert!(!s.is_empty(), "Display should produce a non-empty string");
    // Should mention "empty" or "whitespace"
    assert!(
        s.to_lowercase().contains("empty") || s.to_lowercase().contains("whitespace"),
        "Expected 'empty' or 'whitespace' in: {}",
        s
    );
}

#[test]
fn error_display_invalid_color_handle() {
    let err = ConversionError::InvalidColorAttachmentHandle(999);
    let s = format!("{}", err);
    assert!(s.contains("999"), "Display should include the handle value: {}", s);
}

#[test]
fn error_display_invalid_depth_handle() {
    let err = ConversionError::InvalidDepthStencilHandle(888);
    let s = format!("{}", err);
    assert!(s.contains("888"), "Display should include the handle value: {}", s);
}

#[test]
fn error_display_unknown_color_load_op() {
    let err = ConversionError::UnknownColorLoadOp("FooBar".into());
    let s = format!("{}", err);
    assert!(
        s.contains("FooBar"),
        "Display should include the unknown op string: {}",
        s
    );
}

#[test]
fn error_display_unknown_color_store_op() {
    let err = ConversionError::UnknownColorStoreOp("BazQux".into());
    let s = format!("{}", err);
    assert!(s.contains("BazQux"));
}

#[test]
fn error_display_unknown_depth_load_op() {
    let err = ConversionError::UnknownDepthLoadOp("DepthBad".into());
    let s = format!("{}", err);
    assert!(s.contains("DepthBad"));
}

#[test]
fn error_display_unknown_depth_store_op() {
    let err = ConversionError::UnknownDepthStoreOp("DepthStoreBad".into());
    let s = format!("{}", err);
    assert!(s.contains("DepthStoreBad"));
}

#[test]
fn error_display_unknown_stencil_load_op() {
    let err = ConversionError::UnknownStencilLoadOp("StencilBad".into());
    let s = format!("{}", err);
    assert!(s.contains("StencilBad"));
}

#[test]
fn error_display_unknown_stencil_store_op() {
    let err = ConversionError::UnknownStencilStoreOp("StencilStoreBad".into());
    let s = format!("{}", err);
    assert!(s.contains("StencilStoreBad"));
}

#[test]
fn error_display_missing_dispatch_source() {
    let err = ConversionError::MissingDispatchSource;
    let s = format!("{}", err);
    assert!(
        s.to_lowercase().contains("dispatch"),
        "Expected 'dispatch' in: {}",
        s
    );
}

#[test]
fn error_display_missing_copy_source() {
    let err = ConversionError::MissingCopySource;
    let s = format!("{}", err);
    assert!(
        s.to_lowercase().contains("copy"),
        "Expected 'copy' in: {}",
        s
    );
}

#[test]
fn error_display_missing_copy_destination() {
    let err = ConversionError::MissingCopyDestination;
    let s = format!("{}", err);
    assert!(
        s.to_lowercase().contains("copy") && s.to_lowercase().contains("dest"),
        "Expected 'copy' and 'dest' in: {}",
        s
    );
}

// ===========================================================================
// Section 2: Empty / whitespace name
// ===========================================================================

#[test]
fn empty_name_triggers_empty_pass_name_error() {
    let node = compute_node("");
    let result = IrPass::try_from(node);
    assert!(result.is_err());
    assert_eq!(result.unwrap_err(), ConversionError::EmptyPassName);
}

#[test]
fn whitespace_name_triggers_empty_pass_name_error() {
    let mut node = compute_node("x");
    node.name = "   \t  ".into();
    let result = IrPass::try_from(node);
    assert!(result.is_err());
    assert_eq!(result.unwrap_err(), ConversionError::EmptyPassName);
}

// ===========================================================================
// Section 3: Color attachment NONE handle
// ===========================================================================

#[test]
fn graphics_color_none_handle_rejected() {
    let mut node = graphics_node("color_none");
    node.color_attachments[0].resource = u32::MAX;
    let result = IrPass::try_from(node);
    assert!(result.is_err());
    assert_eq!(
        result.unwrap_err(),
        ConversionError::InvalidColorAttachmentHandle(u32::MAX)
    );
}

#[test]
fn graphics_multiple_colors_one_none_rejected() {
    let mut node = graphics_node("multi_color_none");
    node.color_attachments = vec![
        PyColorAttachment {
            resource: 0,
            load_op: "Clear".into(),
            store_op: "Store".into(),
            ..Default::default()
        },
        PyColorAttachment {
            resource: u32::MAX,
            load_op: "Load".into(),
            store_op: "Store".into(),
            ..Default::default()
        },
    ];
    let result = IrPass::try_from(node);
    assert!(result.is_err());
    assert_eq!(
        result.unwrap_err(),
        ConversionError::InvalidColorAttachmentHandle(u32::MAX)
    );
}

#[test]
fn graphics_valid_color_handle_accepted() {
    let node = graphics_node("valid_color");
    let result = IrPass::try_from(node);
    assert!(result.is_ok());
    let pass = result.unwrap();
    assert_eq!(pass.color_attachments.len(), 1);
    assert_eq!(pass.color_attachments[0].resource, ResourceHandle(0));
}

// ===========================================================================
// Section 4: Depth-stencil NONE handle
// ===========================================================================

#[test]
fn graphics_depth_none_handle_rejected() {
    let mut node = graphics_node("ds_none");
    node.depth_stencil = Some(PyDepthStencilAttachment {
        resource: u32::MAX,
        ..Default::default()
    });
    let result = IrPass::try_from(node);
    assert!(result.is_err());
    assert_eq!(
        result.unwrap_err(),
        ConversionError::InvalidDepthStencilHandle(u32::MAX)
    );
}

#[test]
fn graphics_valid_depth_handle_accepted() {
    let mut node = graphics_node("valid_ds");
    node.depth_stencil = Some(PyDepthStencilAttachment {
        resource: 7,
        ..Default::default()
    });
    let result = IrPass::try_from(node);
    assert!(result.is_ok());
    let ds = result.unwrap().depth_stencil.unwrap();
    assert_eq!(ds.resource, ResourceHandle(7));
}

// ===========================================================================
// Section 5: Unknown colour load/store ops
// ===========================================================================

#[test]
fn unknown_color_load_op_rejected() {
    let mut node = graphics_node("bad_color_load");
    node.color_attachments[0].load_op = "load".into(); // wrong case
    let result = IrPass::try_from(node);
    assert!(result.is_err());
    assert_eq!(
        result.unwrap_err(),
        ConversionError::UnknownColorLoadOp("load".into())
    );
}

#[test]
fn unknown_color_store_op_rejected() {
    let mut node = graphics_node("bad_color_store");
    node.color_attachments[0].store_op = "STORE".into(); // wrong case
    let result = IrPass::try_from(node);
    assert!(result.is_err());
    assert_eq!(
        result.unwrap_err(),
        ConversionError::UnknownColorStoreOp("STORE".into())
    );
}

#[test]
fn invalid_color_op_string_rejected() {
    let mut node = graphics_node("invalid_color_op");
    node.color_attachments[0].load_op = "RandomOp".into();
    let result = IrPass::try_from(node);
    assert!(result.is_err());
    assert_eq!(
        result.unwrap_err(),
        ConversionError::UnknownColorLoadOp("RandomOp".into())
    );
}

// ===========================================================================
// Section 6: Unknown depth ops
// ===========================================================================

#[test]
fn unknown_depth_load_op_rejected() {
    let mut node = graphics_node("bad_depth_load");
    node.depth_stencil = Some(PyDepthStencilAttachment {
        resource: 1,
        depth_load_op: "clear".into(),
        ..Default::default()
    });
    let result = IrPass::try_from(node);
    assert!(result.is_err());
    assert_eq!(
        result.unwrap_err(),
        ConversionError::UnknownDepthLoadOp("clear".into())
    );
}

#[test]
fn unknown_depth_store_op_rejected() {
    let mut node = graphics_node("bad_depth_store");
    node.depth_stencil = Some(PyDepthStencilAttachment {
        resource: 1,
        depth_store_op: "store".into(),
        ..Default::default()
    });
    let result = IrPass::try_from(node);
    assert!(result.is_err());
    assert_eq!(
        result.unwrap_err(),
        ConversionError::UnknownDepthStoreOp("store".into())
    );
}

// ===========================================================================
// Section 7: Unknown stencil ops
// ===========================================================================

#[test]
fn unknown_stencil_load_op_rejected() {
    let mut node = graphics_node("bad_stencil_load");
    node.depth_stencil = Some(PyDepthStencilAttachment {
        resource: 1,
        stencil_load_op: "DONT_CARE".into(),
        ..Default::default()
    });
    let result = IrPass::try_from(node);
    assert!(result.is_err());
    assert_eq!(
        result.unwrap_err(),
        ConversionError::UnknownStencilLoadOp("DONT_CARE".into())
    );
}

#[test]
fn unknown_stencil_store_op_rejected() {
    let mut node = graphics_node("bad_stencil_store");
    node.depth_stencil = Some(PyDepthStencilAttachment {
        resource: 1,
        stencil_store_op: "Dontcare".into(),
        ..Default::default()
    });
    let result = IrPass::try_from(node);
    assert!(result.is_err());
    assert_eq!(
        result.unwrap_err(),
        ConversionError::UnknownStencilStoreOp("Dontcare".into())
    );
}

// ===========================================================================
// Section 8: Missing dispatch source
// ===========================================================================

#[test]
fn compute_without_dispatch_source_rejected() {
    let mut node = compute_node("no_dispatch");
    node.dispatch_source = None;
    let result = IrPass::try_from(node);
    assert!(result.is_err());
    assert_eq!(result.unwrap_err(), ConversionError::MissingDispatchSource);
}

#[test]
fn ray_tracing_without_dispatch_source_rejected() {
    let mut node = rt_node("no_rt_dispatch");
    node.dispatch_source = None;
    let result = IrPass::try_from(node);
    assert!(result.is_err());
    assert_eq!(result.unwrap_err(), ConversionError::MissingDispatchSource);
}

// ===========================================================================
// Section 9: Missing copy source / destination
// ===========================================================================

#[test]
fn copy_without_source_rejected() {
    let mut node = copy_node("no_copy_src");
    node.copy_source = None;
    let result = IrPass::try_from(node);
    assert!(result.is_err());
    assert_eq!(result.unwrap_err(), ConversionError::MissingCopySource);
}

#[test]
fn copy_without_destination_rejected() {
    let mut node = copy_node("no_copy_dst");
    node.copy_dest = None;
    let result = IrPass::try_from(node);
    assert!(result.is_err());
    assert_eq!(result.unwrap_err(), ConversionError::MissingCopyDestination);
}

// ===========================================================================
// Section 10: Graphics success
// ===========================================================================

#[test]
fn graphics_pass_type_and_index() {
    let node = graphics_node("main_render");
    let pass = IrPass::try_from(node).unwrap();
    assert_eq!(pass.index, PassIndex(0));
    assert_eq!(pass.name, "main_render");
    assert_eq!(pass.pass_type, PassType::Graphics);
}

#[test]
fn graphics_pass_color_attachment_fields() {
    let node = graphics_node("gbuffer");
    let pass = IrPass::try_from(node).unwrap();
    assert_eq!(pass.color_attachments.len(), 1);
    let ca = &pass.color_attachments[0];
    assert_eq!(ca.resource, ResourceHandle(0));
    assert_eq!(ca.load_op, AttachmentLoadOp::Clear);
    assert_eq!(ca.store_op, AttachmentStoreOp::Store);
    assert_eq!(ca.mip_level, 0);
    assert_eq!(ca.array_layer, 0);
}

#[test]
fn graphics_pass_depth_stencil_fields() {
    let node = graphics_node("depth_pass");
    let pass = IrPass::try_from(node).unwrap();
    let ds = pass.depth_stencil.as_ref().expect("expected depth-stencil");
    assert_eq!(ds.resource, ResourceHandle(1));
    assert!(ds.depth_test_enabled);
    assert!(ds.depth_write_enabled);
    assert_eq!(ds.clear_depth, 1.0);
}

#[test]
fn graphics_pass_read_only_depth_maps_to_disabled_test_and_write() {
    let mut node = graphics_node("readonly_depth");
    node.depth_stencil = Some(PyDepthStencilAttachment {
        resource: 1,
        depth_load_op: "Load".into(),
        depth_store_op: "DontCare".into(),
        stencil_load_op: "Load".into(),
        stencil_store_op: "DontCare".into(),
        clear_depth: 1.0,
        clear_stencil: 0,
        read_only: true,
    });
    let pass = IrPass::try_from(node).unwrap();
    let ds = pass.depth_stencil.unwrap();
    assert!(!ds.depth_test_enabled, "read_only=true -> depth_test_enabled=false");
    assert!(!ds.depth_write_enabled, "read_only=true -> depth_write_enabled=false");
}

#[test]
fn graphics_pass_no_depth_stencil() {
    let mut node = graphics_node("no_ds");
    node.depth_stencil = None;
    let pass = IrPass::try_from(node).unwrap();
    assert!(pass.depth_stencil.is_none());
}

#[test]
fn graphics_pass_instance_source_defaults() {
    let node = graphics_node("inst_default");
    let pass = IrPass::try_from(node).unwrap();
    match pass.instance_source {
        InstanceSource::Direct {
            index_count,
            instance_count,
            ..
        } => {
            assert_eq!(index_count, 36);
            assert_eq!(instance_count, 1);
        }
        _ => panic!("expected Direct instance source"),
    }
}

// ===========================================================================
// Section 11: Compute success
// ===========================================================================

#[test]
fn compute_pass_type_and_dispatch() {
    let node = compute_node("postfx");
    let pass = IrPass::try_from(node).unwrap();
    assert_eq!(pass.pass_type, PassType::Compute);
    assert!(pass.dispatch_source.is_some());
    assert!(pass.color_attachments.is_empty());
    assert!(pass.depth_stencil.is_none());
    assert!(pass.has_dispatch());
}

#[test]
fn compute_pass_dispatch_source_direct() {
    let node = compute_node("compute_skin");
    let pass = IrPass::try_from(node).unwrap();
    match pass.dispatch_source.unwrap() {
        DispatchSource::Direct {
            group_count_x,
            group_count_y,
            group_count_z,
        } => {
            assert_eq!(group_count_x, 8);
            assert_eq!(group_count_y, 8);
            assert_eq!(group_count_z, 1);
        }
        _ => panic!("expected Direct dispatch source"),
    }
}

// ===========================================================================
// Section 12: Copy success
// ===========================================================================

#[test]
fn copy_pass_type() {
    let node = copy_node("blit");
    let pass = IrPass::try_from(node).unwrap();
    assert_eq!(pass.pass_type, PassType::Copy);
    assert!(pass.dispatch_source.is_none());
    assert!(!pass.has_dispatch());
    assert!(!pass.has_color_attachments());
}

#[test]
fn copy_pass_view_type() {
    let node = copy_node("copy_buffer");
    let pass = IrPass::try_from(node).unwrap();
    assert_eq!(pass.view_type, ViewType::StorageBuffer);
}

// ===========================================================================
// Section 13: Ray-tracing success
// ===========================================================================

#[test]
fn ray_tracing_pass_type() {
    let node = rt_node("rt_gi");
    let pass = IrPass::try_from(node).unwrap();
    assert_eq!(pass.pass_type, PassType::RayTracing);
    assert!(pass.dispatch_source.is_some());
    assert!(pass.has_dispatch());
}

#[test]
fn ray_tracing_pass_view_type() {
    let node = rt_node("rt_ao");
    let pass = IrPass::try_from(node).unwrap();
    assert_eq!(pass.view_type, ViewType::AccelerationStructure);
}

// ===========================================================================
// Section 14: All 9 view types
// ===========================================================================

#[test]
fn all_view_types_round_trip() {
    let cases = vec![
        (PyViewType::Texture2D, ViewType::Texture2D),
        (PyViewType::TextureCube, ViewType::TextureCube),
        (PyViewType::Texture3D, ViewType::Texture3D),
        (PyViewType::Storage, ViewType::Storage),
        (PyViewType::UniformTexel, ViewType::UniformTexel),
        (PyViewType::StorageTexel, ViewType::StorageTexel),
        (PyViewType::UniformBuffer, ViewType::UniformBuffer),
        (PyViewType::StorageBuffer, ViewType::StorageBuffer),
        (PyViewType::AccelerationStructure, ViewType::AccelerationStructure),
    ];
    for (py_vt, ir_vt) in cases {
        let mut node = compute_node("vt_test");
        node.view_type = py_vt;
        let pass = IrPass::try_from(node).unwrap();
        assert_eq!(
            pass.view_type, ir_vt,
            "view type mismatch for {:?}",
            py_vt,
        );
    }
}

// ===========================================================================
// Section 15: Instance source variants (Direct, Indirect, Mesh)
// ===========================================================================

#[test]
fn instance_source_direct_all_fields() {
    let mut node = graphics_node("direct_inst");
    node.instance_source = PyInstanceSource::Direct {
        index_count: 256,
        instance_count: 8,
        base_vertex: 5,
        first_index: 0,
        first_instance: 1,
    };
    let pass = IrPass::try_from(node).unwrap();
    match pass.instance_source {
        InstanceSource::Direct {
            index_count,
            instance_count,
            base_vertex,
            first_index,
            first_instance,
        } => {
            assert_eq!(index_count, 256);
            assert_eq!(instance_count, 8);
            assert_eq!(base_vertex, 5);
            assert_eq!(first_index, 0);
            assert_eq!(first_instance, 1);
        }
        _ => panic!("expected Direct"),
    }
}

#[test]
fn instance_source_indirect_all_fields() {
    let mut node = graphics_node("indirect_inst");
    node.instance_source = PyInstanceSource::Indirect {
        buffer: 10,
        offset: 512,
        draw_count: 16,
        stride: 64,
    };
    let pass = IrPass::try_from(node).unwrap();
    match pass.instance_source {
        InstanceSource::Indirect {
            buffer,
            offset,
            draw_count,
            stride,
        } => {
            assert_eq!(buffer, ResourceHandle(10));
            assert_eq!(offset, 512);
            assert_eq!(draw_count, 16);
            assert_eq!(stride, 64);
        }
        _ => panic!("expected Indirect"),
    }
}

#[test]
fn instance_source_mesh_all_fields() {
    let mut node = graphics_node("mesh_inst");
    node.instance_source = PyInstanceSource::Mesh {
        group_count_x: 32,
        group_count_y: 16,
        group_count_z: 8,
    };
    let pass = IrPass::try_from(node).unwrap();
    match pass.instance_source {
        InstanceSource::Mesh {
            group_count_x,
            group_count_y,
            group_count_z,
        } => {
            assert_eq!(group_count_x, 32);
            assert_eq!(group_count_y, 16);
            assert_eq!(group_count_z, 8);
        }
        _ => panic!("expected Mesh"),
    }
}

// ===========================================================================
// Section 16: Passthrough fields (index, name, tags)
// ===========================================================================

#[test]
fn index_preserved_through_conversion() {
    let mut node = compute_node("indexed");
    node.index = 99;
    let pass = IrPass::try_from(node).unwrap();
    assert_eq!(pass.index, PassIndex(99));
}

#[test]
fn name_preserved_through_conversion() {
    let node = compute_node("special_name_123");
    let pass = IrPass::try_from(node).unwrap();
    assert_eq!(pass.name, "special_name_123");
}

#[test]
fn tags_preserved_through_conversion() {
    let mut node = compute_node("tagged_pass");
    node.tags = vec![
        "transparent".into(),
        "post-process".into(),
        "debug".into(),
    ];
    let pass = IrPass::try_from(node).unwrap();
    assert_eq!(pass.tags.len(), 3);
    assert!(pass.tags.contains(&"transparent".to_string()));
    assert!(pass.tags.contains(&"post-process".to_string()));
    assert!(pass.tags.contains(&"debug".to_string()));
}

#[test]
fn empty_tags_vector() {
    let node = compute_node("no_tags");
    let pass = IrPass::try_from(node).unwrap();
    assert!(pass.tags.is_empty());
}

// ===========================================================================
// Section 17: Access set population
// ===========================================================================

#[test]
fn access_set_reads_filter_none() {
    let mut node = compute_node("reads_filter");
    node.reads = vec![1, u32::MAX, 3, u32::MAX, 5];
    node.writes = vec![];
    let pass = IrPass::try_from(node).unwrap();
    assert!(!pass.access_set.reads.contains(&ResourceHandle::NONE));
    assert_eq!(pass.access_set.reads.len(), 3);
    assert!(pass.access_set.reads.contains(&ResourceHandle(1)));
    assert!(pass.access_set.reads.contains(&ResourceHandle(3)));
    assert!(pass.access_set.reads.contains(&ResourceHandle(5)));
}

#[test]
fn access_set_writes_filter_none() {
    let mut node = compute_node("writes_filter");
    node.reads = vec![];
    node.writes = vec![u32::MAX, 2, u32::MAX, 4];
    let pass = IrPass::try_from(node).unwrap();
    assert!(!pass.access_set.writes.contains(&ResourceHandle::NONE));
    assert_eq!(pass.access_set.writes.len(), 2);
    assert!(pass.access_set.writes.contains(&ResourceHandle(2)));
    assert!(pass.access_set.writes.contains(&ResourceHandle(4)));
}

#[test]
fn access_set_dedup_reads_removed_when_also_in_writes() {
    let mut node = compute_node("dedup");
    node.reads = vec![1, 2, 3, 4];
    node.writes = vec![2, 4, 6];
    let pass = IrPass::try_from(node).unwrap();
    // Reads should no longer contain 2 or 4 (they are in writes)
    assert!(pass.access_set.reads.contains(&ResourceHandle(1)));
    assert!(!pass.access_set.reads.contains(&ResourceHandle(2)));
    assert!(pass.access_set.reads.contains(&ResourceHandle(3)));
    assert!(!pass.access_set.reads.contains(&ResourceHandle(4)));
    // Writes should still contain all their handles
    assert!(pass.access_set.writes.contains(&ResourceHandle(2)));
    assert!(pass.access_set.writes.contains(&ResourceHandle(4)));
    assert!(pass.access_set.writes.contains(&ResourceHandle(6)));
}

// ===========================================================================
// Section 18: DispatchSource variants
// ===========================================================================

#[test]
fn dispatch_source_indirect_fields() {
    let mut node = compute_node("indirect_dispatch");
    node.dispatch_source = Some(PyDispatchSource::Indirect {
        buffer: 8,
        offset: 256,
    });
    let pass = IrPass::try_from(node).unwrap();
    match pass.dispatch_source.unwrap() {
        DispatchSource::Indirect { buffer, offset } => {
            assert_eq!(buffer, ResourceHandle(8));
            assert_eq!(offset, 256);
        }
        _ => panic!("expected Indirect dispatch source"),
    }
}

#[test]
fn dispatch_source_direct_custom_values() {
    let mut node = compute_node("custom_dispatch");
    node.dispatch_source = Some(PyDispatchSource::Direct {
        group_count_x: 128,
        group_count_y: 1,
        group_count_z: 1,
    });
    let pass = IrPass::try_from(node).unwrap();
    match pass.dispatch_source.unwrap() {
        DispatchSource::Direct {
            group_count_x,
            group_count_y,
            group_count_z,
        } => {
            assert_eq!(group_count_x, 128);
            assert_eq!(group_count_y, 1);
            assert_eq!(group_count_z, 1);
        }
        _ => panic!("expected Direct dispatch source"),
    }
}
