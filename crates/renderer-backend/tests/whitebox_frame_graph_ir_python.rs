//! Whitebox tests for PyPassNode to IrPass conversion bridge.
#![cfg(feature = "pyo3")]

use renderer_backend::frame_graph::{
    InstanceSource, IrPass, PassType, ResourceHandle, ViewType,
};
use renderer_backend::frame_graph::python::{
    ConversionError, PyColorAttachment, PyDepthStencilAttachment,
    PyDispatchSource, PyInstanceSource, PyPassNode, PyPassType, PyViewType,
    convert_color_attachment, convert_depth_stencil, convert_dispatch_source,
    convert_instance_source, convert_view_type,
    parse_load_op, parse_store_op, parse_depth_load_op, parse_depth_store_op,
    parse_stencil_load_op, parse_stencil_store_op,
};

fn make_graphics_node() -> PyPassNode {
    PyPassNode {
        name: "gbuffer".to_string(),
        pass_type: PyPassType::Graphics,
        color_attachments: vec![PyColorAttachment { resource: 0, load_op: "Clear".to_string(), store_op: "Store".to_string() }],
        depth_stencil: None,
        reads: vec![1, 2], writes: vec![0],
        instance_source: Some(PyInstanceSource { kind: "Direct".to_string() }),
        dispatch_source: None,
        view_type: PyViewType { kind: "Texture2D".to_string() },
    }
}

// 1. ConversionError Display
#[test] fn display_empty_pass_name() { assert_eq!(ConversionError::EmptyPassName.to_string(), "pass name must not be empty"); }
#[test] fn display_invalid_resource_handle() { assert_eq!(ConversionError::InvalidResourceHandle(42).to_string(), "invalid resource handle: 42"); }
#[test] fn display_invalid_load_op() { assert_eq!(ConversionError::InvalidLoadOp("BAD".into()).to_string(), "invalid load op: BAD"); }
#[test] fn display_invalid_store_op() { assert_eq!(ConversionError::InvalidStoreOp("BAD".into()).to_string(), "invalid store op: BAD"); }
#[test] fn display_invalid_depth_load_op() { assert_eq!(ConversionError::InvalidDepthLoadOp("BAD".into()).to_string(), "invalid depth load op: BAD"); }
#[test] fn display_invalid_depth_store_op() { assert_eq!(ConversionError::InvalidDepthStoreOp("BAD".into()).to_string(), "invalid depth store op: BAD"); }
#[test] fn display_invalid_stencil_load_op() { assert_eq!(ConversionError::InvalidStencilLoadOp("BAD".into()).to_string(), "invalid stencil load op: BAD"); }
#[test] fn display_invalid_stencil_store_op() { assert_eq!(ConversionError::InvalidStencilStoreOp("BAD".into()).to_string(), "invalid stencil store op: BAD"); }
#[test] fn display_invalid_view_type() { assert_eq!(ConversionError::InvalidViewType("WRONG".into()).to_string(), "invalid view type: WRONG"); }
#[test] fn display_invalid_instance_source() { assert_eq!(ConversionError::InvalidInstanceSource("BAD".into()).to_string(), "invalid instance source: BAD"); }
#[test] fn display_invalid_dispatch_source() { assert_eq!(ConversionError::InvalidDispatchSource("BAD".into()).to_string(), "invalid dispatch source: BAD"); }
#[test] fn display_missing_color_attachments() { assert_eq!(ConversionError::MissingColorAttachments.to_string(), "graphics pass must have at least one color attachment"); }
#[test] fn display_attachments_not_allowed() { assert_eq!(ConversionError::AttachmentsNotAllowed(PassType::Compute).to_string(), "attachments not allowed for pass type: Compute"); }
#[test] fn display_missing_dispatch_source() { assert_eq!(ConversionError::MissingDispatchSource.to_string(), "compute pass requires a dispatch source"); }
#[test] fn display_invalid_depth_stencil_handle() { assert_eq!(ConversionError::InvalidDepthStencilHandle.to_string(), "depth/stencil attachment resource handle is NONE"); }

// 2. Parse helpers
#[test] fn parse_load_op_valid() { assert!(parse_load_op("Load").is_ok() && parse_load_op("Clear").is_ok() && parse_load_op("DontCare").is_ok()); }
#[test] fn parse_load_op_invalid() { assert!(matches!(parse_load_op("Bogus").unwrap_err(), ConversionError::InvalidLoadOp(_))); }
#[test] fn parse_store_op_valid() { assert!(parse_store_op("Store").is_ok() && parse_store_op("DontCare").is_ok()); }
#[test] fn parse_store_op_invalid() { assert!(matches!(parse_store_op("Bogus").unwrap_err(), ConversionError::InvalidStoreOp(_))); }
#[test] fn parse_depth_load_op_valid() { assert!(parse_depth_load_op("Load").is_ok() && parse_depth_load_op("Clear").is_ok() && parse_depth_load_op("DontCare").is_ok()); }
#[test] fn parse_depth_load_op_invalid() { assert!(matches!(parse_depth_load_op("Bogus").unwrap_err(), ConversionError::InvalidDepthLoadOp(_))); }
#[test] fn parse_depth_store_op_valid() { assert!(parse_depth_store_op("Store").is_ok() && parse_depth_store_op("DontCare").is_ok()); }
#[test] fn parse_depth_store_op_invalid() { assert!(matches!(parse_depth_store_op("Bogus").unwrap_err(), ConversionError::InvalidDepthStoreOp(_))); }
#[test] fn parse_stencil_load_op_valid() { assert!(parse_stencil_load_op("Load").is_ok() && parse_stencil_load_op("Clear").is_ok() && parse_stencil_load_op("DontCare").is_ok()); }
#[test] fn parse_stencil_load_op_invalid() { assert!(matches!(parse_stencil_load_op("Bogus").unwrap_err(), ConversionError::InvalidStencilLoadOp(_))); }
#[test] fn parse_stencil_store_op_valid() { assert!(parse_stencil_store_op("Store").is_ok() && parse_stencil_store_op("DontCare").is_ok()); }
#[test] fn parse_stencil_store_op_invalid() { assert!(matches!(parse_stencil_store_op("Bogus").unwrap_err(), ConversionError::InvalidStencilStoreOp(_))); }

// 3. View type conversion
#[test] fn view_type_all() { for (s, e) in &[("Texture2D", ViewType::Texture2D), ("TextureCube", ViewType::TextureCube), ("Texture3D", ViewType::Texture3D), ("Storage", ViewType::Storage), ("UniformTexel", ViewType::UniformTexel), ("StorageTexel", ViewType::StorageTexel), ("UniformBuffer", ViewType::UniformBuffer), ("StorageBuffer", ViewType::StorageBuffer), ("AccelerationStructure", ViewType::AccelerationStructure)] { assert_eq!(convert_view_type(&PyViewType { kind: (*s).to_string() }).unwrap(), *e); } }
#[test] fn view_type_invalid() { assert!(matches!(convert_view_type(&PyViewType { kind: "Bad".to_string() }).unwrap_err(), ConversionError::InvalidViewType(_))); }

// 4. Instance source
#[test] fn instance_source_all() { for s in &["Direct", "Indirect", "Mesh"] { assert!(convert_instance_source(&PyInstanceSource { kind: (*s).to_string() }).is_ok()); } }
#[test] fn instance_source_invalid() { assert!(matches!(convert_instance_source(&PyInstanceSource { kind: "Bad".to_string() }).unwrap_err(), ConversionError::InvalidInstanceSource(_))); }

// 5. Dispatch source
#[test] fn dispatch_source_all() { for s in &["Direct", "Indirect"] { assert!(convert_dispatch_source(&PyDispatchSource { kind: (*s).to_string() }).is_ok()); } }
#[test] fn dispatch_source_invalid() { assert!(matches!(convert_dispatch_source(&PyDispatchSource { kind: "Bad".to_string() }).unwrap_err(), ConversionError::InvalidDispatchSource(_))); }

// 6. Color attachment
#[test] fn color_attachment_valid() { let ca = convert_color_attachment(&PyColorAttachment { resource: 7, load_op: "Clear".to_string(), store_op: "Store".to_string() }).unwrap(); assert_eq!(ca.resource.0, 7); }
#[test] fn color_attachment_none_handle() { assert!(matches!(convert_color_attachment(&PyColorAttachment { resource: u32::MAX, load_op: "Clear".to_string(), store_op: "Store".to_string() }).unwrap_err(), ConversionError::InvalidResourceHandle(_))); }
#[test] fn color_attachment_bad_load_op() { assert!(matches!(convert_color_attachment(&PyColorAttachment { resource: 0, load_op: "Bad".to_string(), store_op: "Store".to_string() }).unwrap_err(), ConversionError::InvalidLoadOp(_))); }
#[test] fn color_attachment_bad_store_op() { assert!(matches!(convert_color_attachment(&PyColorAttachment { resource: 0, load_op: "Clear".to_string(), store_op: "Bad".to_string() }).unwrap_err(), ConversionError::InvalidStoreOp(_))); }

// 7. Depth/stencil
#[test] fn depth_stencil_valid() { let ds = convert_depth_stencil(&PyDepthStencilAttachment { resource: 5, depth_load_op: "Load".to_string(), depth_store_op: "Store".to_string(), stencil_load_op: "Clear".to_string(), stencil_store_op: "DontCare".to_string() }).unwrap(); assert_eq!(ds.resource.0, 5); }
#[test] fn depth_stencil_none_handle() { assert!(matches!(convert_depth_stencil(&PyDepthStencilAttachment { resource: u32::MAX, depth_load_op: "Load".to_string(), depth_store_op: "Store".to_string(), stencil_load_op: "Clear".to_string(), stencil_store_op: "DontCare".to_string() }).unwrap_err(), ConversionError::InvalidDepthStencilHandle)); }
#[test] fn depth_stencil_bad_depth_load() { assert!(matches!(convert_depth_stencil(&PyDepthStencilAttachment { resource: 0, depth_load_op: "Bad".to_string(), depth_store_op: "Store".to_string(), stencil_load_op: "Clear".to_string(), stencil_store_op: "DontCare".to_string() }).unwrap_err(), ConversionError::InvalidDepthLoadOp(_))); }
#[test] fn depth_stencil_bad_depth_store() { assert!(matches!(convert_depth_stencil(&PyDepthStencilAttachment { resource: 0, depth_load_op: "Load".to_string(), depth_store_op: "Bad".to_string(), stencil_load_op: "Clear".to_string(), stencil_store_op: "DontCare".to_string() }).unwrap_err(), ConversionError::InvalidDepthStoreOp(_))); }
#[test] fn depth_stencil_bad_stencil_load() { assert!(matches!(convert_depth_stencil(&PyDepthStencilAttachment { resource: 0, depth_load_op: "Load".to_string(), depth_store_op: "Store".to_string(), stencil_load_op: "Bad".to_string(), stencil_store_op: "DontCare".to_string() }).unwrap_err(), ConversionError::InvalidStencilLoadOp(_))); }
#[test] fn depth_stencil_bad_stencil_store() { assert!(matches!(convert_depth_stencil(&PyDepthStencilAttachment { resource: 0, depth_load_op: "Load".to_string(), depth_store_op: "Store".to_string(), stencil_load_op: "Clear".to_string(), stencil_store_op: "Bad".to_string() }).unwrap_err(), ConversionError::InvalidStencilStoreOp(_))); }

// 8. Graphics pass
#[test] fn graphics_minimal() { let pass: IrPass = make_graphics_node().try_into().unwrap(); assert_eq!(pass.name, "gbuffer"); assert_eq!(pass.pass_type, PassType::Graphics); assert_eq!(pass.color_attachments.len(), 1); }
#[test] fn graphics_empty_name() { let mut n = make_graphics_node(); n.name = "".to_string(); assert!(matches!(TryInto::<IrPass>::try_into(n), Err(ConversionError::EmptyPassName))); }
#[test] fn graphics_no_color() { let mut n = make_graphics_node(); n.color_attachments.clear(); assert!(matches!(TryInto::<IrPass>::try_into(n), Err(ConversionError::MissingColorAttachments))); }
#[test] fn graphics_with_depth_stencil() { let mut n = make_graphics_node(); n.depth_stencil = Some(PyDepthStencilAttachment { resource: 10, depth_load_op: "Load".to_string(), depth_store_op: "Store".to_string(), stencil_load_op: "Clear".to_string(), stencil_store_op: "DontCare".to_string() }); let pass: IrPass = n.try_into().unwrap(); assert!(pass.depth_stencil.is_some()); }
#[test] fn graphics_bad_color_handle() { let mut n = make_graphics_node(); n.color_attachments[0].resource = u32::MAX; assert!(matches!(TryInto::<IrPass>::try_into(n), Err(ConversionError::InvalidResourceHandle(_)))); }
#[test] fn graphics_bad_instance_source() { let mut n = make_graphics_node(); n.instance_source = Some(PyInstanceSource { kind: "Bad".to_string() }); assert!(matches!(TryInto::<IrPass>::try_into(n), Err(ConversionError::InvalidInstanceSource(_)))); }
#[test] fn graphics_no_instance_source_default() { let mut n = make_graphics_node(); n.instance_source = None; let pass: IrPass = n.try_into().unwrap(); assert_eq!(pass.instance_source, InstanceSource::Direct { index_count: 0, instance_count: 1, base_vertex: 0, first_index: 0, first_instance: 0 }); }
#[test] fn graphics_multiple_color_attachments() { let mut n = make_graphics_node(); n.color_attachments.push(PyColorAttachment { resource: 1, load_op: "Load".to_string(), store_op: "Store".to_string() }); let pass: IrPass = n.try_into().unwrap(); assert_eq!(pass.color_attachments.len(), 2); }
#[test] fn graphics_read_write_sets() { let mut n = make_graphics_node(); n.reads = vec![10,20,30]; n.writes = vec![40,50]; let pass: IrPass = n.try_into().unwrap(); assert!(pass.access_set.reads.contains(&ResourceHandle(10))); assert!(pass.access_set.writes.contains(&ResourceHandle(40))); }

// 9. Compute pass
#[test] fn compute_valid() { let node = PyPassNode { name: "cs".to_string(), pass_type: PyPassType::Compute, color_attachments: vec![], depth_stencil: None, reads: vec![0], writes: vec![1], instance_source: None, dispatch_source: Some(PyDispatchSource { kind: "Direct".to_string() }), view_type: PyViewType { kind: "Storage".to_string() } }; let pass: IrPass = node.try_into().unwrap(); assert_eq!(pass.pass_type, PassType::Compute); assert!(pass.dispatch_source.is_some()); }
#[test] fn compute_indirect_dispatch() { let node = PyPassNode { name: "cs_ind".to_string(), pass_type: PyPassType::Compute, color_attachments: vec![], depth_stencil: None, reads: vec![], writes: vec![], instance_source: None, dispatch_source: Some(PyDispatchSource { kind: "Indirect".to_string() }), view_type: PyViewType { kind: "Storage".to_string() } }; let pass: IrPass = node.try_into().unwrap(); assert!(pass.dispatch_source.is_some()); }
#[test] fn compute_missing_dispatch() { let node = PyPassNode { name: "cs_bad".to_string(), pass_type: PyPassType::Compute, color_attachments: vec![], depth_stencil: None, reads: vec![], writes: vec![], instance_source: None, dispatch_source: None, view_type: PyViewType { kind: "Storage".to_string() } }; assert!(matches!(TryInto::<IrPass>::try_into(node), Err(ConversionError::MissingDispatchSource))); }
#[test] fn compute_with_attachments() { let node = PyPassNode { name: "cs_att".to_string(), pass_type: PyPassType::Compute, color_attachments: vec![PyColorAttachment { resource: 0, load_op: "Load".to_string(), store_op: "Store".to_string() }], depth_stencil: None, reads: vec![], writes: vec![], instance_source: None, dispatch_source: Some(PyDispatchSource { kind: "Direct".to_string() }), view_type: PyViewType { kind: "Storage".to_string() } }; assert!(matches!(TryInto::<IrPass>::try_into(node), Err(ConversionError::AttachmentsNotAllowed(PassType::Compute)))); }
#[test] fn compute_bad_dispatch_source() { let node = PyPassNode { name: "cs_bad_d".to_string(), pass_type: PyPassType::Compute, color_attachments: vec![], depth_stencil: None, reads: vec![], writes: vec![], instance_source: None, dispatch_source: Some(PyDispatchSource { kind: "Bad".to_string() }), view_type: PyViewType { kind: "Storage".to_string() } }; assert!(matches!(TryInto::<IrPass>::try_into(node), Err(ConversionError::InvalidDispatchSource(_)))); }

// 10. Ray tracing
#[test] fn ray_tracing_valid() { let node = PyPassNode { name: "rt".to_string(), pass_type: PyPassType::RayTracing, color_attachments: vec![], depth_stencil: None, reads: vec![0], writes: vec![1], instance_source: None, dispatch_source: None, view_type: PyViewType { kind: "AccelerationStructure".to_string() } }; let pass: IrPass = node.try_into().unwrap(); assert_eq!(pass.pass_type, PassType::RayTracing); }
#[test] fn ray_tracing_with_attachments() { let node = PyPassNode { name: "rt_att".to_string(), pass_type: PyPassType::RayTracing, color_attachments: vec![PyColorAttachment { resource: 0, load_op: "Load".to_string(), store_op: "Store".to_string() }], depth_stencil: None, reads: vec![], writes: vec![], instance_source: None, dispatch_source: None, view_type: PyViewType { kind: "AccelerationStructure".to_string() } }; assert!(matches!(TryInto::<IrPass>::try_into(node), Err(ConversionError::AttachmentsNotAllowed(PassType::RayTracing)))); }

// 11. Copy pass
#[test] fn copy_valid() { let node = PyPassNode { name: "copy".to_string(), pass_type: PyPassType::Copy, color_attachments: vec![], depth_stencil: None, reads: vec![0], writes: vec![1], instance_source: None, dispatch_source: None, view_type: PyViewType { kind: "StorageBuffer".to_string() } }; let pass: IrPass = node.try_into().unwrap(); assert_eq!(pass.pass_type, PassType::Copy); }

// 12. PyPassType
#[test] fn passtype_from_str() { assert_eq!(PyPassType::from_str("Graphics"), Some(PyPassType::Graphics)); assert_eq!(PyPassType::from_str("Compute"), Some(PyPassType::Compute)); assert_eq!(PyPassType::from_str("Copy"), Some(PyPassType::Copy)); assert_eq!(PyPassType::from_str("RayTracing"), Some(PyPassType::RayTracing)); assert_eq!(PyPassType::from_str("Bad"), None); }
#[test] fn passtype_to_ir() { assert_eq!(PyPassType::Graphics.to_ir(), PassType::Graphics); assert_eq!(PyPassType::Compute.to_ir(), PassType::Compute); assert_eq!(PyPassType::Copy.to_ir(), PassType::Copy); assert_eq!(PyPassType::RayTracing.to_ir(), PassType::RayTracing); }

// 13. Edge cases
#[test] fn empty_resource_sets() { let mut n = make_graphics_node(); n.reads.clear(); n.writes.clear(); let pass: IrPass = n.try_into().unwrap(); assert!(pass.access_set.reads.is_empty() && pass.access_set.writes.is_empty()); }
#[test] fn duplicate_reads() { let mut n = make_graphics_node(); n.reads = vec![1,1,2,2,3]; let pass: IrPass = n.try_into().unwrap(); assert_eq!(pass.access_set.reads.len(), 3); }
#[test] fn duplicate_writes() { let mut n = make_graphics_node(); n.writes = vec![0,0,0]; let pass: IrPass = n.try_into().unwrap(); assert_eq!(pass.access_set.writes.len(), 1); }
#[test] fn max_valid_handle() { let mut n = make_graphics_node(); n.color_attachments[0].resource = u32::MAX - 1; let pass: IrPass = n.try_into().unwrap(); assert_eq!(pass.color_attachments[0].resource.0, u32::MAX - 1); }
#[test] fn invalid_view_type_on_graphics() { let mut n = make_graphics_node(); n.view_type = PyViewType { kind: "Bad".to_string() }; assert!(matches!(TryInto::<IrPass>::try_into(n), Err(ConversionError::InvalidViewType(_)))); }
#[test] fn depth_stencil_defaults() { let mut n = make_graphics_node(); n.depth_stencil = Some(PyDepthStencilAttachment { resource: 7, depth_load_op: "Load".to_string(), depth_store_op: "Store".to_string(), stencil_load_op: "Load".to_string(), stencil_store_op: "DontCare".to_string() }); let pass: IrPass = n.try_into().unwrap(); let ds = pass.depth_stencil.unwrap(); assert!(ds.depth_test_enabled); assert!(ds.depth_write_enabled); assert_eq!(ds.clear_depth, 1.0); }
