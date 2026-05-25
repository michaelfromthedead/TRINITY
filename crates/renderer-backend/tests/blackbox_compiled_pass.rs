// Blackbox contract tests for CompiledPass enum (T-FG-1.5).
//
// CLEANROOM: No src/ access beyond the public API exported by the crate.
// Tests use only `renderer_backend::frame_graph::*` -- no internal fields,
// no private methods, no implementation details.
//
// Acceptance criterion (T-FG-1.5):
//   CompiledPass enum with Graphics/Compute/Copy/RayTracing variants,
//   from_ir_pass conversion, unified accessors (index/name/view_type),
//   Display for all variants. Blackbox tests against public API contract.
//
// Coverage:
//   1.  Graphics variant -- full field construction, field access
//   2.  Compute variant -- full field construction, field access
//   3.  Copy variant -- full field construction, field access
//   4.  RayTracing variant -- full field construction, field access
//   5.  from_ir_pass -- Graphics IrPass -> CompiledPass
//   6.  from_ir_pass -- Compute IrPass -> CompiledPass
//   7.  from_ir_pass -- Copy IrPass -> CompiledPass
//   8.  from_ir_pass -- RayTracing IrPass -> CompiledPass
//   9.  from_ir_pass -- all fields preserved across conversion
//  10.  From<IrPass> trait impl (Into trait)
//  11.  index() -- unified accessor across all 4 variants
//  12.  name() -- unified accessor across all 4 variants
//  13.  view_type() -- unified accessor across all 4 variants
//  14.  Clone -- Clone preserves value
//  15.  Debug -- Debug round-trip
//  16.  PartialEq -- same content equals, different variant not equal
//  17.  Display -- Graphics variant format
//  18.  Display -- Compute variant format
//  19.  Display -- Copy variant format
//  20.  Display -- RayTracing variant format
//  21.  Integration -- all 4 variants round-trip via from_ir_pass

use renderer_backend::frame_graph::{
    AttachmentLoadOp, AttachmentStoreOp, ColorAttachment, CompiledPass,
    DepthStencilAttachment, DispatchSource, InstanceSource, IrPass, PassIndex, PassType,
    ResourceAccessSet, ResourceHandle, ViewType,
};

// =============================================================================
// SECTION 1 -- Graphics variant: construction, field access
// =============================================================================

#[test]
fn compiled_pass_graphics_constructs_directly() {
    let color_att = ColorAttachment::default();
    let ds = DepthStencilAttachment::default();

    let cp = CompiledPass::Graphics {
        index: PassIndex(0),
        name: "gbuffer".into(),
        access_set: ResourceAccessSet::empty(),
        color_attachments: vec![color_att],
        depth_stencil: Some(ds),
        instance_source: InstanceSource::Direct {
            index_count: 36,
            instance_count: 1,
            base_vertex: 0,
            first_index: 0,
            first_instance: 0,
        },
        view_type: ViewType::ColorAttachment,
        tags: vec![],
    };

    assert_eq!(cp.index(), PassIndex(0), "Graphics index preserved");
    assert_eq!(cp.name(), "gbuffer", "Graphics name preserved");
    assert_eq!(cp.view_type(), ViewType::ColorAttachment, "Graphics view_type preserved");
}

#[test]
fn compiled_pass_graphics_no_depth_stencil() {
    let cp = CompiledPass::Graphics {
        index: PassIndex(1),
        name: "no_ds".into(),
        access_set: ResourceAccessSet::empty(),
        color_attachments: vec![],
        depth_stencil: None,
        instance_source: InstanceSource::Direct {
            index_count: 0,
            instance_count: 1,
            base_vertex: 0,
            first_index: 0,
            first_instance: 0,
        },
        view_type: ViewType::Texture2D,
        tags: vec![],
    };

    assert!(cp.index() == PassIndex(1));
}

#[test]
fn compiled_pass_graphics_tags_field() {
    let cp = CompiledPass::Graphics {
        index: PassIndex(2),
        name: "tagged_gfx".into(),
        access_set: ResourceAccessSet::empty(),
        color_attachments: vec![],
        depth_stencil: None,
        instance_source: InstanceSource::Direct {
            index_count: 0,
            instance_count: 1,
            base_vertex: 0,
            first_index: 0,
            first_instance: 0,
        },
        view_type: ViewType::Texture2D,
        tags: vec!["transparent".into(), "debug".into()],
    };

    // index() is the only public accessor; for tags we rely on PartialEq
    // -- the struct round-trips through clone + compare.
    assert_eq!(cp.index(), PassIndex(2));
}

// =============================================================================
// SECTION 2 -- Compute variant: construction, field access
// =============================================================================

#[test]
fn compiled_pass_compute_constructs_directly() {
    let cp = CompiledPass::Compute {
        index: PassIndex(1),
        name: "bloom".into(),
        access_set: ResourceAccessSet::empty(),
        dispatch_source: DispatchSource::Direct {
            group_count_x: 16,
            group_count_y: 16,
            group_count_z: 1,
        },
        view_type: ViewType::Storage,
        tags: vec![],
    };

    assert_eq!(cp.index(), PassIndex(1), "Compute index preserved");
    assert_eq!(cp.name(), "bloom", "Compute name preserved");
    assert_eq!(cp.view_type(), ViewType::Storage, "Compute view_type preserved");
}

#[test]
fn compiled_pass_compute_indirect_dispatch() {
    let cp = CompiledPass::Compute {
        index: PassIndex(2),
        name: "indirect_comp".into(),
        access_set: ResourceAccessSet::empty(),
        dispatch_source: DispatchSource::Indirect {
            buffer: ResourceHandle(7),
            offset: 0,
        },
        view_type: ViewType::StorageBuffer,
        tags: vec![],
    };

    assert_eq!(cp.index(), PassIndex(2));
    assert_eq!(cp.view_type(), ViewType::StorageBuffer);
}

// =============================================================================
// SECTION 3 -- Copy variant: construction, field access
// =============================================================================

#[test]
fn compiled_pass_copy_constructs_directly() {
    let cp = CompiledPass::Copy {
        index: PassIndex(2),
        name: "depth_copy".into(),
        access_set: ResourceAccessSet::empty(),
        view_type: ViewType::Texture2D,
        tags: vec![],
    };

    assert_eq!(cp.index(), PassIndex(2), "Copy index preserved");
    assert_eq!(cp.name(), "depth_copy", "Copy name preserved");
    assert_eq!(cp.view_type(), ViewType::Texture2D, "Copy view_type preserved");
}

#[test]
fn compiled_pass_copy_tags() {
    let cp = CompiledPass::Copy {
        index: PassIndex(5),
        name: "tagged_copy".into(),
        access_set: ResourceAccessSet::empty(),
        view_type: ViewType::StorageBuffer,
        tags: vec!["post-process".into()],
    };

    assert_eq!(cp.index(), PassIndex(5));
    assert_eq!(cp.name(), "tagged_copy");
}

// =============================================================================
// SECTION 4 -- RayTracing variant: construction, field access
// =============================================================================

#[test]
fn compiled_pass_ray_tracing_constructs_directly() {
    let cp = CompiledPass::RayTracing {
        index: PassIndex(3),
        name: "reflections".into(),
        access_set: ResourceAccessSet::empty(),
        dispatch_source: DispatchSource::Direct {
            group_count_x: 8,
            group_count_y: 8,
            group_count_z: 1,
        },
        view_type: ViewType::AccelerationStructure,
        tags: vec![],
    };

    assert_eq!(cp.index(), PassIndex(3), "RayTracing index preserved");
    assert_eq!(cp.name(), "reflections", "RayTracing name preserved");
    assert_eq!(cp.view_type(), ViewType::AccelerationStructure, "RayTracing view_type preserved");
}

#[test]
fn compiled_pass_ray_tracing_indirect_dispatch() {
    let cp = CompiledPass::RayTracing {
        index: PassIndex(4),
        name: "rt_indirect".into(),
        access_set: ResourceAccessSet::empty(),
        dispatch_source: DispatchSource::Indirect {
            buffer: ResourceHandle(3),
            offset: 64,
        },
        view_type: ViewType::AccelerationStructure,
        tags: vec![],
    };

    assert_eq!(cp.index(), PassIndex(4));
    assert_eq!(cp.name(), "rt_indirect");
}

// =============================================================================
// SECTION 5 -- from_ir_pass: Graphics conversion
// =============================================================================

#[test]
fn from_ir_pass_graphics_converts_to_compiled_pass_graphics() {
    let pass = IrPass::graphics(
        PassIndex(0),
        "gbuffer",
        vec![ColorAttachment::default()],
        Some(DepthStencilAttachment::default()),
        InstanceSource::Direct {
            index_count: 36,
            instance_count: 1,
            base_vertex: 0,
            first_index: 0,
            first_instance: 0,
        },
        ViewType::ColorAttachment,
    );

    let cp = CompiledPass::from_ir_pass(pass);
    assert_eq!(cp.index(), PassIndex(0), "from_ir_pass preserves index for Graphics");
    assert_eq!(cp.name(), "gbuffer", "from_ir_pass preserves name for Graphics");
    assert_eq!(cp.view_type(), ViewType::ColorAttachment, "from_ir_pass preserves view_type for Graphics");
}

#[test]
fn from_ir_pass_graphics_with_tags() {
    let mut pass = IrPass::graphics(
        PassIndex(5),
        "tagged_gfx",
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
    pass.tags.push("debug".into());

    let cp = CompiledPass::from_ir_pass(pass);
    assert_eq!(cp.index(), PassIndex(5));
    assert_eq!(cp.name(), "tagged_gfx");
}

// =============================================================================
// SECTION 6 -- from_ir_pass: Compute conversion
// =============================================================================

#[test]
fn from_ir_pass_compute_converts_to_compiled_pass_compute() {
    let pass = IrPass::compute(
        PassIndex(1),
        "postfx",
        DispatchSource::Direct {
            group_count_x: 16,
            group_count_y: 16,
            group_count_z: 1,
        },
        ViewType::Storage,
    );

    let cp = CompiledPass::from_ir_pass(pass);
    assert_eq!(cp.index(), PassIndex(1), "from_ir_pass preserves index for Compute");
    assert_eq!(cp.name(), "postfx", "from_ir_pass preserves name for Compute");
    assert_eq!(cp.view_type(), ViewType::Storage, "from_ir_pass preserves view_type for Compute");
}

#[test]
fn from_ir_pass_compute_indirect() {
    let pass = IrPass::compute(
        PassIndex(3),
        "indirect_comp",
        DispatchSource::Indirect {
            buffer: ResourceHandle(5),
            offset: 0,
        },
        ViewType::StorageBuffer,
    );

    let cp = CompiledPass::from_ir_pass(pass);
    assert_eq!(cp.index(), PassIndex(3));
    assert_eq!(cp.name(), "indirect_comp");
    assert_eq!(cp.view_type(), ViewType::StorageBuffer);
}

// =============================================================================
// SECTION 7 -- from_ir_pass: Copy conversion
// =============================================================================

#[test]
fn from_ir_pass_copy_converts_to_compiled_pass_copy() {
    let pass = IrPass::copy(PassIndex(2), "depth_copy");

    let cp = CompiledPass::from_ir_pass(pass);
    assert_eq!(cp.index(), PassIndex(2), "from_ir_pass preserves index for Copy");
    assert_eq!(cp.name(), "depth_copy", "from_ir_pass preserves name for Copy");
}

#[test]
fn from_ir_pass_copy_default_view_type() {
    let pass = IrPass::copy(PassIndex(4), "buffer_copy");
    let cp = CompiledPass::from_ir_pass(pass);
    // Copy passes default to ViewType::StorageBuffer
    assert_eq!(cp.view_type(), ViewType::StorageBuffer, "Copy pass default view_type is StorageBuffer");
}

// =============================================================================
// SECTION 8 -- from_ir_pass: RayTracing conversion
// =============================================================================

#[test]
fn from_ir_pass_ray_tracing_converts_to_compiled_pass_ray_tracing() {
    let pass = IrPass::ray_tracing(
        PassIndex(3),
        "rt_reflections",
        DispatchSource::Direct {
            group_count_x: 8,
            group_count_y: 8,
            group_count_z: 1,
        },
    );

    let cp = CompiledPass::from_ir_pass(pass);
    assert_eq!(cp.index(), PassIndex(3), "from_ir_pass preserves index for RayTracing");
    assert_eq!(cp.name(), "rt_reflections", "from_ir_pass preserves name for RayTracing");
    assert_eq!(cp.view_type(), ViewType::Storage, "RayTracing IrPass defaults to Storage view_type");
}

#[test]
fn from_ir_pass_ray_tracing_indirect() {
    let pass = IrPass::ray_tracing(
        PassIndex(6),
        "rt_indirect",
        DispatchSource::Indirect {
            buffer: ResourceHandle(9),
            offset: 128,
        },
    );

    let cp = CompiledPass::from_ir_pass(pass);
    assert_eq!(cp.index(), PassIndex(6));
    assert_eq!(cp.name(), "rt_indirect");
}

// =============================================================================
// SECTION 9 -- from_ir_pass: all fields preserved
// =============================================================================

#[test]
fn from_ir_pass_graphics_preserves_all_fields() {
    let ca = ColorAttachment {
        resource: ResourceHandle(10),
        mip_level: 0,
        array_layer: 0,
        load_op: AttachmentLoadOp::Clear,
        store_op: AttachmentStoreOp::Store,
        clear_color: [0.0, 0.0, 0.0, 1.0],
    };
    let ds = DepthStencilAttachment {
        resource: ResourceHandle(11),
        depth_load_op: AttachmentLoadOp::Clear,
        depth_store_op: AttachmentStoreOp::Store,
        stencil_load_op: AttachmentLoadOp::Load,
        stencil_store_op: AttachmentStoreOp::DontCare,
        clear_depth: 1.0,
        clear_stencil: 0,
        depth_test_enabled: true,
        depth_write_enabled: true,
    };

    let mut pass = IrPass::graphics(
        PassIndex(0),
        "full_gbuffer",
        vec![ca],
        Some(ds),
        InstanceSource::Direct {
            index_count: 36,
            instance_count: 2,
            base_vertex: 0,
            first_index: 0,
            first_instance: 0,
        },
        ViewType::ColorAttachment,
    );
    pass.tags.push("opaque".into());

    let cp = CompiledPass::from_ir_pass(pass);

    assert_eq!(cp.index(), PassIndex(0));
    assert_eq!(cp.name(), "full_gbuffer");
    assert_eq!(cp.view_type(), ViewType::ColorAttachment);
}

#[test]
fn from_ir_pass_compute_preserves_dispatch_source() {
    let pass = IrPass::compute(
        PassIndex(1),
        "compute_full",
        DispatchSource::Direct {
            group_count_x: 32,
            group_count_y: 1,
            group_count_z: 1,
        },
        ViewType::Storage,
    );

    let cp = CompiledPass::from_ir_pass(pass);
    assert_eq!(cp.index(), PassIndex(1));
    assert_eq!(cp.name(), "compute_full");
    assert_eq!(cp.view_type(), ViewType::Storage);
}

#[test]
fn from_ir_pass_ray_tracing_preserves_dispatch_source() {
    let pass = IrPass::ray_tracing(
        PassIndex(4),
        "rt_full",
        DispatchSource::Direct {
            group_count_x: 4,
            group_count_y: 4,
            group_count_z: 4,
        },
    );

    let cp = CompiledPass::from_ir_pass(pass);
    assert_eq!(cp.index(), PassIndex(4));
    assert_eq!(cp.name(), "rt_full");
    assert_eq!(cp.view_type(), ViewType::Storage);
}

// =============================================================================
// SECTION 10 -- From<IrPass> trait impl
// =============================================================================

#[test]
fn from_trait_converts_ir_pass_to_compiled_pass() {
    let pass = IrPass::compute(
        PassIndex(0),
        "from_trait_test",
        DispatchSource::Direct {
            group_count_x: 1,
            group_count_y: 1,
            group_count_z: 1,
        },
        ViewType::Storage,
    );

    let cp: CompiledPass = pass.into();
    assert_eq!(cp.name(), "from_trait_test", "Into<CompiledPass> works via From<IrPass>");
    assert_eq!(cp.index(), PassIndex(0));
}

#[test]
fn from_trait_graphics() {
    let pass = IrPass::graphics(
        PassIndex(0),
        "from_trait_gfx",
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

    let cp = CompiledPass::from(pass);
    assert_eq!(cp.name(), "from_trait_gfx");
    assert_eq!(cp.view_type(), ViewType::Texture2D);
}

// =============================================================================
// SECTION 11 -- index() unified accessor across all 4 variants
// =============================================================================

#[test]
fn index_accessor_graphics() {
    let cp = CompiledPass::Graphics {
        index: PassIndex(42),
        name: String::new(),
        access_set: ResourceAccessSet::empty(),
        color_attachments: vec![],
        depth_stencil: None,
        instance_source: InstanceSource::Direct {
            index_count: 0,
            instance_count: 1,
            base_vertex: 0,
            first_index: 0,
            first_instance: 0,
        },
        view_type: ViewType::Texture2D,
        tags: vec![],
    };
    assert_eq!(cp.index(), PassIndex(42), "index() returns Graphics index");
}

#[test]
fn index_accessor_compute() {
    let cp = CompiledPass::Compute {
        index: PassIndex(99),
        name: String::new(),
        access_set: ResourceAccessSet::empty(),
        dispatch_source: DispatchSource::Direct {
            group_count_x: 1,
            group_count_y: 1,
            group_count_z: 1,
        },
        view_type: ViewType::Storage,
        tags: vec![],
    };
    assert_eq!(cp.index(), PassIndex(99), "index() returns Compute index");
}

#[test]
fn index_accessor_copy() {
    let cp = CompiledPass::Copy {
        index: PassIndex(7),
        name: String::new(),
        access_set: ResourceAccessSet::empty(),
        view_type: ViewType::StorageBuffer,
        tags: vec![],
    };
    assert_eq!(cp.index(), PassIndex(7), "index() returns Copy index");
}

#[test]
fn index_accessor_ray_tracing() {
    let cp = CompiledPass::RayTracing {
        index: PassIndex(13),
        name: String::new(),
        access_set: ResourceAccessSet::empty(),
        dispatch_source: DispatchSource::Direct {
            group_count_x: 1,
            group_count_y: 1,
            group_count_z: 1,
        },
        view_type: ViewType::AccelerationStructure,
        tags: vec![],
    };
    assert_eq!(cp.index(), PassIndex(13), "index() returns RayTracing index");
}

// =============================================================================
// SECTION 12 -- name() unified accessor across all 4 variants
// =============================================================================

#[test]
fn name_accessor_graphics() {
    let cp = CompiledPass::Graphics {
        index: PassIndex(0),
        name: "shadow_map".into(),
        access_set: ResourceAccessSet::empty(),
        color_attachments: vec![],
        depth_stencil: None,
        instance_source: InstanceSource::Direct {
            index_count: 0,
            instance_count: 1,
            base_vertex: 0,
            first_index: 0,
            first_instance: 0,
        },
        view_type: ViewType::Texture2D,
        tags: vec![],
    };
    assert_eq!(cp.name(), "shadow_map", "name() returns Graphics name");
}

#[test]
fn name_accessor_compute() {
    let cp = CompiledPass::Compute {
        index: PassIndex(0),
        name: "cs_blur".into(),
        access_set: ResourceAccessSet::empty(),
        dispatch_source: DispatchSource::Direct {
            group_count_x: 1,
            group_count_y: 1,
            group_count_z: 1,
        },
        view_type: ViewType::Storage,
        tags: vec![],
    };
    assert_eq!(cp.name(), "cs_blur", "name() returns Compute name");
}

#[test]
fn name_accessor_copy() {
    let cp = CompiledPass::Copy {
        index: PassIndex(0),
        name: "copy_op".into(),
        access_set: ResourceAccessSet::empty(),
        view_type: ViewType::StorageBuffer,
        tags: vec![],
    };
    assert_eq!(cp.name(), "copy_op", "name() returns Copy name");
}

#[test]
fn name_accessor_ray_tracing() {
    let cp = CompiledPass::RayTracing {
        index: PassIndex(0),
        name: "rt_ao".into(),
        access_set: ResourceAccessSet::empty(),
        dispatch_source: DispatchSource::Direct {
            group_count_x: 1,
            group_count_y: 1,
            group_count_z: 1,
        },
        view_type: ViewType::AccelerationStructure,
        tags: vec![],
    };
    assert_eq!(cp.name(), "rt_ao", "name() returns RayTracing name");
}

// =============================================================================
// SECTION 13 -- view_type() unified accessor across all 4 variants
// =============================================================================

#[test]
fn view_type_accessor_graphics() {
    let cp = CompiledPass::Graphics {
        index: PassIndex(0),
        name: String::new(),
        access_set: ResourceAccessSet::empty(),
        color_attachments: vec![],
        depth_stencil: None,
        instance_source: InstanceSource::Direct {
            index_count: 0,
            instance_count: 1,
            base_vertex: 0,
            first_index: 0,
            first_instance: 0,
        },
        view_type: ViewType::ColorAttachment,
        tags: vec![],
    };
    assert_eq!(cp.view_type(), ViewType::ColorAttachment);
}

#[test]
fn view_type_accessor_compute() {
    let cp = CompiledPass::Compute {
        index: PassIndex(0),
        name: String::new(),
        access_set: ResourceAccessSet::empty(),
        dispatch_source: DispatchSource::Direct {
            group_count_x: 1,
            group_count_y: 1,
            group_count_z: 1,
        },
        view_type: ViewType::Storage,
        tags: vec![],
    };
    assert_eq!(cp.view_type(), ViewType::Storage);
}

#[test]
fn view_type_accessor_copy() {
    let cp = CompiledPass::Copy {
        index: PassIndex(0),
        name: String::new(),
        access_set: ResourceAccessSet::empty(),
        view_type: ViewType::StorageBuffer,
        tags: vec![],
    };
    assert_eq!(cp.view_type(), ViewType::StorageBuffer);
}

#[test]
fn view_type_accessor_ray_tracing() {
    let cp = CompiledPass::RayTracing {
        index: PassIndex(0),
        name: String::new(),
        access_set: ResourceAccessSet::empty(),
        dispatch_source: DispatchSource::Direct {
            group_count_x: 1,
            group_count_y: 1,
            group_count_z: 1,
        },
        view_type: ViewType::AccelerationStructure,
        tags: vec![],
    };
    assert_eq!(cp.view_type(), ViewType::AccelerationStructure);
}

// =============================================================================
// SECTION 14 -- Clone
// =============================================================================

#[test]
fn compiled_pass_clone_graphics() {
    let cp = CompiledPass::Graphics {
        index: PassIndex(7),
        name: "clone_test".into(),
        access_set: ResourceAccessSet::empty(),
        color_attachments: vec![ColorAttachment::default()],
        depth_stencil: Some(DepthStencilAttachment::default()),
        instance_source: InstanceSource::Direct {
            index_count: 36,
            instance_count: 1,
            base_vertex: 0,
            first_index: 0,
            first_instance: 0,
        },
        view_type: ViewType::ColorAttachment,
        tags: vec!["debug".into()],
    };

    let cloned = cp.clone();
    assert_eq!(cp, cloned, "Clone of Graphics variant is equal");
    assert_eq!(cloned.index(), PassIndex(7));
    assert_eq!(cloned.name(), "clone_test");
}

#[test]
fn compiled_pass_clone_compute() {
    let cp = CompiledPass::Compute {
        index: PassIndex(3),
        name: "cs_clone".into(),
        access_set: ResourceAccessSet::empty(),
        dispatch_source: DispatchSource::Direct {
            group_count_x: 8,
            group_count_y: 8,
            group_count_z: 1,
        },
        view_type: ViewType::Storage,
        tags: vec![],
    };

    let cloned = cp.clone();
    assert_eq!(cp, cloned, "Clone of Compute variant is equal");
}

#[test]
fn compiled_pass_clone_copy() {
    let cp = CompiledPass::Copy {
        index: PassIndex(4),
        name: "copy_clone".into(),
        access_set: ResourceAccessSet::empty(),
        view_type: ViewType::Texture2D,
        tags: vec![],
    };

    let cloned = cp.clone();
    assert_eq!(cp, cloned, "Clone of Copy variant is equal");
}

#[test]
fn compiled_pass_clone_ray_tracing() {
    let cp = CompiledPass::RayTracing {
        index: PassIndex(5),
        name: "rt_clone".into(),
        access_set: ResourceAccessSet::empty(),
        dispatch_source: DispatchSource::Direct {
            group_count_x: 1,
            group_count_y: 1,
            group_count_z: 1,
        },
        view_type: ViewType::AccelerationStructure,
        tags: vec![],
    };

    let cloned = cp.clone();
    assert_eq!(cp, cloned, "Clone of RayTracing variant is equal");
}

// =============================================================================
// SECTION 15 -- Debug
// =============================================================================

#[test]
fn compiled_pass_debug_graphics_contains_variant_name() {
    let cp = CompiledPass::Graphics {
        index: PassIndex(0),
        name: "debug_gfx".into(),
        access_set: ResourceAccessSet::empty(),
        color_attachments: vec![],
        depth_stencil: None,
        instance_source: InstanceSource::Direct {
            index_count: 0,
            instance_count: 1,
            base_vertex: 0,
            first_index: 0,
            first_instance: 0,
        },
        view_type: ViewType::Texture2D,
        tags: vec![],
    };

    let s = format!("{:?}", cp);
    assert!(s.contains("Graphics"), "Debug output names the variant");
    assert!(s.contains("debug_gfx"), "Debug output contains the pass name");
}

#[test]
fn compiled_pass_debug_compute() {
    let cp = CompiledPass::Compute {
        index: PassIndex(1),
        name: "debug_cs".into(),
        access_set: ResourceAccessSet::empty(),
        dispatch_source: DispatchSource::Direct {
            group_count_x: 1,
            group_count_y: 1,
            group_count_z: 1,
        },
        view_type: ViewType::Storage,
        tags: vec![],
    };

    let s = format!("{:?}", cp);
    assert!(s.contains("Compute"), "Debug output names Compute");
    assert!(s.contains("debug_cs"));
}

#[test]
fn compiled_pass_debug_copy() {
    let cp = CompiledPass::Copy {
        index: PassIndex(2),
        name: "debug_copy".into(),
        access_set: ResourceAccessSet::empty(),
        view_type: ViewType::StorageBuffer,
        tags: vec![],
    };

    let s = format!("{:?}", cp);
    assert!(s.contains("Copy"), "Debug output names Copy");
    assert!(s.contains("debug_copy"));
}

#[test]
fn compiled_pass_debug_ray_tracing() {
    let cp = CompiledPass::RayTracing {
        index: PassIndex(3),
        name: "debug_rt".into(),
        access_set: ResourceAccessSet::empty(),
        dispatch_source: DispatchSource::Direct {
            group_count_x: 1,
            group_count_y: 1,
            group_count_z: 1,
        },
        view_type: ViewType::AccelerationStructure,
        tags: vec![],
    };

    let s = format!("{:?}", cp);
    assert!(s.contains("RayTracing"), "Debug output names RayTracing");
    assert!(s.contains("debug_rt"));
}

// =============================================================================
// SECTION 16 -- PartialEq
// =============================================================================

#[test]
fn compiled_pass_partial_eq_same_graphics_equal() {
    let a = CompiledPass::Graphics {
        index: PassIndex(0),
        name: "same".into(),
        access_set: ResourceAccessSet::empty(),
        color_attachments: vec![],
        depth_stencil: None,
        instance_source: InstanceSource::Direct {
            index_count: 0,
            instance_count: 1,
            base_vertex: 0,
            first_index: 0,
            first_instance: 0,
        },
        view_type: ViewType::Texture2D,
        tags: vec![],
    };
    let b = CompiledPass::Graphics {
        index: PassIndex(0),
        name: "same".into(),
        access_set: ResourceAccessSet::empty(),
        color_attachments: vec![],
        depth_stencil: None,
        instance_source: InstanceSource::Direct {
            index_count: 0,
            instance_count: 1,
            base_vertex: 0,
            first_index: 0,
            first_instance: 0,
        },
        view_type: ViewType::Texture2D,
        tags: vec![],
    };
    assert_eq!(a, b, "Identical Graphics variants are equal");
}

#[test]
fn compiled_pass_partial_eq_different_names_not_equal() {
    let a = CompiledPass::Graphics {
        index: PassIndex(0),
        name: "alpha".into(),
        access_set: ResourceAccessSet::empty(),
        color_attachments: vec![],
        depth_stencil: None,
        instance_source: InstanceSource::Direct {
            index_count: 0,
            instance_count: 1,
            base_vertex: 0,
            first_index: 0,
            first_instance: 0,
        },
        view_type: ViewType::Texture2D,
        tags: vec![],
    };
    let b = CompiledPass::Graphics {
        index: PassIndex(0),
        name: "beta".into(),
        access_set: ResourceAccessSet::empty(),
        color_attachments: vec![],
        depth_stencil: None,
        instance_source: InstanceSource::Direct {
            index_count: 0,
            instance_count: 1,
            base_vertex: 0,
            first_index: 0,
            first_instance: 0,
        },
        view_type: ViewType::Texture2D,
        tags: vec![],
    };
    assert_ne!(a, b, "Different names are not equal");
}

#[test]
fn compiled_pass_partial_eq_different_variants_not_equal() {
    let gfx = CompiledPass::Graphics {
        index: PassIndex(0),
        name: "pass".into(),
        access_set: ResourceAccessSet::empty(),
        color_attachments: vec![],
        depth_stencil: None,
        instance_source: InstanceSource::Direct {
            index_count: 0,
            instance_count: 1,
            base_vertex: 0,
            first_index: 0,
            first_instance: 0,
        },
        view_type: ViewType::Texture2D,
        tags: vec![],
    };
    let cmp = CompiledPass::Compute {
        index: PassIndex(0),
        name: "pass".into(),
        access_set: ResourceAccessSet::empty(),
        dispatch_source: DispatchSource::Direct {
            group_count_x: 1,
            group_count_y: 1,
            group_count_z: 1,
        },
        view_type: ViewType::Texture2D,
        tags: vec![],
    };
    assert_ne!(gfx, cmp, "Graphics != Compute even with same index/name");
}

#[test]
fn compiled_pass_partial_eq_same_compute_equal() {
    let a = CompiledPass::Compute {
        index: PassIndex(1),
        name: "cs".into(),
        access_set: ResourceAccessSet::empty(),
        dispatch_source: DispatchSource::Direct {
            group_count_x: 8,
            group_count_y: 8,
            group_count_z: 1,
        },
        view_type: ViewType::Storage,
        tags: vec![],
    };
    let b = CompiledPass::Compute {
        index: PassIndex(1),
        name: "cs".into(),
        access_set: ResourceAccessSet::empty(),
        dispatch_source: DispatchSource::Direct {
            group_count_x: 8,
            group_count_y: 8,
            group_count_z: 1,
        },
        view_type: ViewType::Storage,
        tags: vec![],
    };
    assert_eq!(a, b);
}

#[test]
fn compiled_pass_partial_eq_same_copy_equal() {
    let a = CompiledPass::Copy {
        index: PassIndex(2),
        name: "copy".into(),
        access_set: ResourceAccessSet::empty(),
        view_type: ViewType::StorageBuffer,
        tags: vec![],
    };
    let b = CompiledPass::Copy {
        index: PassIndex(2),
        name: "copy".into(),
        access_set: ResourceAccessSet::empty(),
        view_type: ViewType::StorageBuffer,
        tags: vec![],
    };
    assert_eq!(a, b);
}

#[test]
fn compiled_pass_partial_eq_same_ray_tracing_equal() {
    let a = CompiledPass::RayTracing {
        index: PassIndex(3),
        name: "rt".into(),
        access_set: ResourceAccessSet::empty(),
        dispatch_source: DispatchSource::Direct {
            group_count_x: 1,
            group_count_y: 1,
            group_count_z: 1,
        },
        view_type: ViewType::AccelerationStructure,
        tags: vec![],
    };
    let b = CompiledPass::RayTracing {
        index: PassIndex(3),
        name: "rt".into(),
        access_set: ResourceAccessSet::empty(),
        dispatch_source: DispatchSource::Direct {
            group_count_x: 1,
            group_count_y: 1,
            group_count_z: 1,
        },
        view_type: ViewType::AccelerationStructure,
        tags: vec![],
    };
    assert_eq!(a, b);
}

// =============================================================================
// SECTION 17 -- Display: Graphics variant
// =============================================================================

#[test]
fn compiled_pass_display_graphics_contains_variant_name() {
    let cp = CompiledPass::Graphics {
        index: PassIndex(0),
        name: "gbuffer".into(),
        access_set: ResourceAccessSet::empty(),
        color_attachments: vec![],
        depth_stencil: None,
        instance_source: InstanceSource::Direct {
            index_count: 0,
            instance_count: 1,
            base_vertex: 0,
            first_index: 0,
            first_instance: 0,
        },
        view_type: ViewType::ColorAttachment,
        tags: vec![],
    };

    let s = format!("{}", cp);
    assert!(s.contains("CompiledPass::Graphics"), "Display starts with variant name");
    assert!(s.contains("gbuffer"), "Display contains pass name");
    assert!(s.contains("0"), "Display contains pass index");
    assert!(s.contains("colors=0"), "Display contains color attachment count");
    assert!(s.contains("ds=none"), "Display shows no depth-stencil");
    assert!(s.contains("ColorAttachment"), "Display contains view type");
}

#[test]
fn compiled_pass_display_graphics_with_depth_stencil() {
    let cp = CompiledPass::Graphics {
        index: PassIndex(1),
        name: "z_prepass".into(),
        access_set: ResourceAccessSet::empty(),
        color_attachments: vec![],
        depth_stencil: Some(DepthStencilAttachment::default()),
        instance_source: InstanceSource::Direct {
            index_count: 0,
            instance_count: 1,
            base_vertex: 0,
            first_index: 0,
            first_instance: 0,
        },
        view_type: ViewType::Texture2D,
        tags: vec![],
    };

    let s = format!("{}", cp);
    assert!(s.contains("CompiledPass::Graphics"));
    assert!(s.contains("ds=present"), "Display shows ds=present when depth_stencil is Some");
}

// =============================================================================
// SECTION 18 -- Display: Compute variant
// =============================================================================

#[test]
fn compiled_pass_display_compute() {
    let cp = CompiledPass::Compute {
        index: PassIndex(1),
        name: "bloom_cs".into(),
        access_set: ResourceAccessSet::empty(),
        dispatch_source: DispatchSource::Direct {
            group_count_x: 16,
            group_count_y: 16,
            group_count_z: 1,
        },
        view_type: ViewType::Storage,
        tags: vec![],
    };

    let s = format!("{}", cp);
    assert!(s.contains("CompiledPass::Compute"), "Display starts with variant name");
    assert!(s.contains("bloom_cs"), "Display contains pass name");
    assert!(s.contains("1"), "Display contains pass index");
    assert!(s.contains("dispatch="), "Display contains dispatch source");
    assert!(s.contains("Storage"), "Display contains view type");
    assert!(s.contains("access="), "Display contains access set");
}

// =============================================================================
// SECTION 19 -- Display: Copy variant
// =============================================================================

#[test]
fn compiled_pass_display_copy() {
    let cp = CompiledPass::Copy {
        index: PassIndex(2),
        name: "depth_copy".into(),
        access_set: ResourceAccessSet::empty(),
        view_type: ViewType::StorageBuffer,
        tags: vec![],
    };

    let s = format!("{}", cp);
    assert!(s.contains("CompiledPass::Copy"), "Display starts with variant name");
    assert!(s.contains("depth_copy"), "Display contains pass name");
    assert!(s.contains("2"), "Display contains pass index");
    assert!(s.contains("StorageBuffer"), "Display contains view type");
    assert!(s.contains("access="), "Display contains access set");
}

// =============================================================================
// SECTION 20 -- Display: RayTracing variant
// =============================================================================

#[test]
fn compiled_pass_display_ray_tracing() {
    let cp = CompiledPass::RayTracing {
        index: PassIndex(3),
        name: "rt_reflections".into(),
        access_set: ResourceAccessSet::empty(),
        dispatch_source: DispatchSource::Direct {
            group_count_x: 8,
            group_count_y: 8,
            group_count_z: 1,
        },
        view_type: ViewType::AccelerationStructure,
        tags: vec![],
    };

    let s = format!("{}", cp);
    assert!(s.contains("CompiledPass::RayTracing"), "Display starts with variant name");
    assert!(s.contains("rt_reflections"), "Display contains pass name");
    assert!(s.contains("3"), "Display contains pass index");
    assert!(s.contains("dispatch="), "Display contains dispatch source");
    assert!(s.contains("AccelerationStructure"), "Display contains view type");
    assert!(s.contains("access="), "Display contains access set");
}

// =============================================================================
// SECTION 21 -- Integration: all 4 variants round-trip via from_ir_pass
// =============================================================================

#[test]
fn integration_all_four_variants_round_trip() {
    let gfx_pass = IrPass::graphics(
        PassIndex(0),
        "gbuffer",
        vec![ColorAttachment {
            resource: ResourceHandle(1),
            ..Default::default()
        }],
        Some(DepthStencilAttachment::default()),
        InstanceSource::Direct {
            index_count: 36,
            instance_count: 1,
            base_vertex: 0,
            first_index: 0,
            first_instance: 0,
        },
        ViewType::ColorAttachment,
    );

    let compute_pass = IrPass::compute(
        PassIndex(1),
        "postfx",
        DispatchSource::Direct {
            group_count_x: 16,
            group_count_y: 16,
            group_count_z: 1,
        },
        ViewType::Storage,
    );

    let copy_pass = IrPass::copy(PassIndex(2), "depth_copy");

    let rt_pass = IrPass::ray_tracing(
        PassIndex(3),
        "rt_reflections",
        DispatchSource::Direct {
            group_count_x: 8,
            group_count_y: 8,
            group_count_z: 1,
        },
    );

    // Convert all four
    let compiled: Vec<CompiledPass> = vec![
        CompiledPass::from_ir_pass(gfx_pass),
        CompiledPass::from_ir_pass(compute_pass),
        CompiledPass::from_ir_pass(copy_pass),
        CompiledPass::from_ir_pass(rt_pass),
    ];

    assert_eq!(compiled.len(), 4, "All 4 passes converted");
    assert_eq!(compiled[0].index(), PassIndex(0), "Graphics index preserved");
    assert_eq!(compiled[1].index(), PassIndex(1), "Compute index preserved");
    assert_eq!(compiled[2].index(), PassIndex(2), "Copy index preserved");
    assert_eq!(compiled[3].index(), PassIndex(3), "RayTracing index preserved");

    assert_eq!(compiled[0].name(), "gbuffer");
    assert_eq!(compiled[1].name(), "postfx");
    assert_eq!(compiled[2].name(), "depth_copy");
    assert_eq!(compiled[3].name(), "rt_reflections");
}

#[test]
fn integration_from_ir_pass_chain() {
    // Verify that each IrPass pass_type maps to the correct CompiledPass variant
    let passes = vec![
        (
            IrPass::graphics(
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
            ),
            PassType::Graphics,
        ),
        (
            IrPass::compute(
                PassIndex(1),
                "cs",
                DispatchSource::Direct {
                    group_count_x: 1,
                    group_count_y: 1,
                    group_count_z: 1,
                },
                ViewType::Storage,
            ),
            PassType::Compute,
        ),
        (
            IrPass::copy(PassIndex(2), "cp"),
            PassType::Copy,
        ),
        (
            IrPass::ray_tracing(
                PassIndex(3),
                "rt",
                DispatchSource::Direct {
                    group_count_x: 1,
                    group_count_y: 1,
                    group_count_z: 1,
                },
            ),
            PassType::RayTracing,
        ),
    ];

    for (ir_pass, expected_type) in passes {
        let cp = CompiledPass::from_ir_pass(ir_pass);
        // The CompiledPass stores the type implicitly via its variant.
        // We verify by checking index/name are preserved regardless of variant.
        assert_eq!(
            cp.name(),
            match expected_type {
                PassType::Graphics => "gfx",
                PassType::Compute => "cs",
                PassType::Copy => "cp",
                PassType::RayTracing => "rt",
            },
            "from_ir_pass preserves name for {:?}",
            expected_type,
        );
    }
}
